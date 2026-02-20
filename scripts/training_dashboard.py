#!/usr/bin/env python3
"""Training Dashboard - Parse training logs and generate visual report.

Usage:
    python scripts/training_dashboard.py <log_file>
    python scripts/training_dashboard.py  # uses latest training log
"""

import re
import sys
import os
import webbrowser
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def parse_training_log(filepath):
    """Extract metrics from stable-baselines3 training output."""
    with open(filepath, "r") as f:
        content = f.read()

    metrics = []
    blocks = content.split("---")

    current = {}
    for block in blocks:
        for line in block.strip().split("\n"):
            line = line.strip().strip("|").strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) == 2:
                key, val = parts
                key = key.strip()
                val = val.strip()
                try:
                    val = float(val)
                    current[key] = val
                except ValueError:
                    pass

        if "total_timesteps" in current and "ep_rew_mean" in current:
            metrics.append(dict(current))
            current = {}

    return metrics


def generate_html(metrics, log_file):
    """Generate self-contained HTML dashboard with Chart.js."""
    if not metrics:
        return "<h1>No training data found</h1>"

    timesteps = [m.get("total_timesteps", 0) for m in metrics]
    rewards = [m.get("ep_rew_mean", 0) for m in metrics]
    ep_lens = [m.get("ep_len_mean", 0) for m in metrics]
    losses = [m.get("loss", 0) for m in metrics if "loss" in m]
    loss_steps = [m.get("total_timesteps", 0) for m in metrics if "loss" in m]
    stds = [m.get("std", 0) for m in metrics if "std" in m]
    std_steps = [m.get("total_timesteps", 0) for m in metrics if "std" in m]

    latest = metrics[-1]
    total = latest.get("total_timesteps", 0)
    elapsed = latest.get("time_elapsed", 0)
    fps = latest.get("fps", 0)
    best_reward = max(rewards)
    best_idx = rewards.index(best_reward)

    # Format labels as M (millions)
    labels_js = [f"{s/1e6:.1f}" for s in timesteps]
    loss_labels_js = [f"{s/1e6:.1f}" for s in loss_steps]
    std_labels_js = [f"{s/1e6:.1f}" for s in std_steps]

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="60">
<title>JPRobot Training Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0f172a; color: #e2e8f0; padding: 20px; }}
  h1 {{ text-align: center; margin-bottom: 8px; font-size: 24px; color: #38bdf8; }}
  .subtitle {{ text-align: center; color: #64748b; margin-bottom: 20px; font-size: 13px; }}
  .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px; margin-bottom: 24px; }}
  .stat {{ background: #1e293b; border-radius: 10px; padding: 16px; text-align: center; }}
  .stat .label {{ font-size: 12px; color: #94a3b8; text-transform: uppercase; }}
  .stat .value {{ font-size: 28px; font-weight: 700; margin-top: 4px; }}
  .stat .value.green {{ color: #4ade80; }}
  .stat .value.blue {{ color: #38bdf8; }}
  .stat .value.yellow {{ color: #facc15; }}
  .stat .value.purple {{ color: #c084fc; }}
  .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .chart-box {{ background: #1e293b; border-radius: 10px; padding: 16px; }}
  .chart-box h3 {{ font-size: 14px; color: #94a3b8; margin-bottom: 10px; }}
  canvas {{ max-height: 280px; }}
  @media (max-width: 800px) {{ .charts {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<h1>JPRobot 训练面板</h1>
<p class="subtitle">日志: {os.path.basename(log_file)} | 每60秒自动刷新</p>

<div class="summary" style="background:#1e293b; border-radius:10px; padding:16px; margin-bottom:20px; line-height:1.8; font-size:14px; color:#cbd5e1;">
  <b style="color:#38bdf8;">训练总结：</b>
  已训练 <b style="color:#38bdf8;">{total/1e6:.1f}M</b> 步（共需 50M），耗时 <b>{elapsed/3600:.1f}</b> 小时。
  机器狗当前每轮平均奖励 <b style="color:#4ade80;">{rewards[-1]:.0f}</b>（越高越好），
  历史最高 <b style="color:#facc15;">{best_reward:.0f}</b>。
  每轮平均存活 <b style="color:#c084fc;">{ep_lens[-1]:.0f}</b> 步（满分250，越高说明越不容易摔倒）。
  探索系数 std=<b>{stds[-1] if stds else 0:.2f}</b>（从1.0逐渐下降，说明动作越来越确定）。
  {'<b style="color:#4ade80;">训练健康，奖励持续增长。</b>' if rewards[-1] > rewards[0] and rewards[-1] > 0 else '<b style="color:#f87171;">注意：奖励出现下降，可能需要检查。</b>'}
</div>

<div class="stats">
  <div class="stat"><div class="label">训练进度</div><div class="value blue">{total/1e6:.1f}M</div></div>
  <div class="stat"><div class="label">当前奖励</div><div class="value green">{rewards[-1]:.0f}</div></div>
  <div class="stat"><div class="label">最高奖励</div><div class="value yellow">{best_reward:.0f}</div></div>
  <div class="stat"><div class="label">存活步数</div><div class="value purple">{ep_lens[-1]:.0f}/250</div></div>
  <div class="stat"><div class="label">训练速度</div><div class="value blue">{fps:.0f} fps</div></div>
  <div class="stat"><div class="label">已用时间</div><div class="value">{elapsed/3600:.1f}h</div></div>
</div>

<div class="charts">
  <div class="chart-box">
    <h3>奖励曲线（越高越好：机器狗走得越远奖励越高）</h3>
    <canvas id="rewardChart"></canvas>
  </div>
  <div class="chart-box">
    <h3>存活步数（越高越好：250满分表示不摔倒）</h3>
    <canvas id="lenChart"></canvas>
  </div>
  <div class="chart-box">
    <h3>策略损失（训练过程的波动，正常会震荡）</h3>
    <canvas id="lossChart"></canvas>
  </div>
  <div class="chart-box">
    <h3>探索系数（逐渐下降=动作越来越精确）</h3>
    <canvas id="stdChart"></canvas>
  </div>
</div>

<script>
const chartOpts = {{
  responsive: true,
  plugins: {{ legend: {{ display: false }} }},
  scales: {{
    x: {{ title: {{ display: true, text: '训练步数 (百万)', color: '#64748b' }},
          ticks: {{ color: '#64748b' }}, grid: {{ color: '#334155' }} }},
    y: {{ ticks: {{ color: '#64748b' }}, grid: {{ color: '#334155' }} }}
  }}
}};

new Chart(document.getElementById('rewardChart'), {{
  type: 'line',
  data: {{
    labels: {labels_js},
    datasets: [{{ data: {rewards}, borderColor: '#4ade80', borderWidth: 2, pointRadius: 0, fill: false }}]
  }},
  options: chartOpts
}});

new Chart(document.getElementById('lenChart'), {{
  type: 'line',
  data: {{
    labels: {labels_js},
    datasets: [{{ data: {ep_lens}, borderColor: '#c084fc', borderWidth: 2, pointRadius: 0, fill: false }}]
  }},
  options: chartOpts
}});

new Chart(document.getElementById('lossChart'), {{
  type: 'line',
  data: {{
    labels: {loss_labels_js},
    datasets: [{{ data: {losses}, borderColor: '#fb923c', borderWidth: 2, pointRadius: 0, fill: false }}]
  }},
  options: chartOpts
}});

new Chart(document.getElementById('stdChart'), {{
  type: 'line',
  data: {{
    labels: {std_labels_js},
    datasets: [{{ data: {stds}, borderColor: '#38bdf8', borderWidth: 2, pointRadius: 0, fill: false }}]
  }},
  options: chartOpts
}});
</script>
</body>
</html>"""
    return html


def main():
    # Find log file
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        # Default: look for latest training output
        tmp_dir = "/private/tmp/claude-501/-Users-mlamp-Workspace-JPClaw/tasks/"
        log_file = os.path.join(tmp_dir, "b4e9ff9.output")

    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        sys.exit(1)

    print(f"Parsing: {log_file}")
    metrics = parse_training_log(log_file)
    print(f"Found {len(metrics)} data points")

    if not metrics:
        print("No training data found!")
        sys.exit(1)

    # Generate HTML
    html = generate_html(metrics, log_file)
    output_path = os.path.join(
        os.path.dirname(__file__), "..", "trained", "dashboard.html"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)

    print(f"Dashboard saved: {output_path}")

    latest = metrics[-1]
    print(f"\nCurrent: {latest.get('total_timesteps', 0)/1e6:.1f}M steps, "
          f"reward={latest.get('ep_rew_mean', 0):.0f}, "
          f"elapsed={latest.get('time_elapsed', 0)/3600:.1f}h")

    # Open in browser
    webbrowser.open(f"file://{os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
