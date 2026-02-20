#!/usr/bin/env python3
"""Training Dashboard Web Server.

Serves the training dashboard via HTTP, auto-refreshes data from training logs.

Usage:
    python scripts/training_server.py                          # default port 18791
    python scripts/training_server.py --port 8080              # custom port
    python scripts/training_server.py --log /path/to/log.txt   # custom log file
"""

import json
import os
import re
import sys
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Default log file (current training)
DEFAULT_LOG = "/private/tmp/claude-501/-Users-mlamp-Workspace-JPClaw/tasks/b4e9ff9.output"
PROGRESSIVE_STATE = os.path.join(os.path.dirname(__file__), "..", "trained", "progressive_state.json")


def parse_progressive_state():
    """Read progressive training state if available."""
    path = os.path.abspath(PROGRESSIVE_STATE)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def parse_training_log(filepath):
    """Extract metrics from stable-baselines3 training output."""
    if not os.path.exists(filepath):
        return []

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
                try:
                    current[key.strip()] = float(val.strip())
                except ValueError:
                    pass
        if "total_timesteps" in current and "ep_rew_mean" in current:
            metrics.append(dict(current))
            current = {}
    return metrics


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>JPRobot 训练面板</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0f172a; color: #e2e8f0; padding: 20px; }
  h1 { text-align: center; margin-bottom: 6px; font-size: 26px; color: #38bdf8; }
  .subtitle { text-align: center; color: #64748b; margin-bottom: 16px; font-size: 13px; }
  .btn-row { text-align: center; margin-bottom: 20px; }
  .btn { background: #6366f1; color: white; border: none; padding: 10px 24px;
         border-radius: 8px; cursor: pointer; font-size: 14px; margin: 0 6px; }
  .btn:hover { background: #818cf8; }
  .btn:active { background: #4f46e5; }
  .summary { background: #1e293b; border-radius: 10px; padding: 16px; margin-bottom: 20px;
             line-height: 1.8; font-size: 14px; color: #cbd5e1; }
  .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
           gap: 12px; margin-bottom: 24px; }
  .stat { background: #1e293b; border-radius: 10px; padding: 16px; text-align: center; }
  .stat .label { font-size: 12px; color: #94a3b8; }
  .stat .value { font-size: 26px; font-weight: 700; margin-top: 4px; }
  .green { color: #4ade80; } .blue { color: #38bdf8; }
  .yellow { color: #facc15; } .purple { color: #c084fc; }
  .red { color: #f87171; }
  .section-title { font-size: 16px; color: #94a3b8; margin: 20px 0 12px 0; padding-left: 4px;
                   border-left: 3px solid #6366f1; padding-left: 10px; }
  .health { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 12px; margin-bottom: 24px; }
  .health-card { background: #1e293b; border-radius: 10px; padding: 14px; }
  .health-card .h-name { font-size: 13px; color: #94a3b8; margin-bottom: 2px; }
  .health-card .h-cn { font-size: 11px; color: #64748b; margin-bottom: 6px; }
  .health-card .h-val { font-size: 22px; font-weight: 700; }
  .health-card .h-hint { font-size: 11px; margin-top: 4px; }
  .h-good { color: #4ade80; } .h-warn { color: #facc15; } .h-bad { color: #f87171; }
  .charts { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .chart-box { background: #1e293b; border-radius: 10px; padding: 16px; }
  .chart-box h3 { font-size: 14px; color: #94a3b8; margin-bottom: 10px; }
  canvas { max-height: 280px; }
  .status { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 12px; }
  .status.running { background: #166534; color: #4ade80; }
  .status.done { background: #1e3a5f; color: #38bdf8; }
  @media (max-width: 800px) { .charts { grid-template-columns: 1fr; } .health { grid-template-columns: 1fr 1fr; } }
</style>
</head>
<body>
<h1>JPRobot 训练面板</h1>
<p class="subtitle">
  <span id="statusBadge" class="status running">训练中</span>
  &nbsp; 每15秒自动刷新 &nbsp;|&nbsp; <span id="lastUpdate"></span>
</p>

<div class="btn-row">
  <button class="btn" onclick="fetchData()">刷新数据</button>
</div>

<div class="summary" id="summary">加载中...</div>

<div class="stats" id="stats"></div>

<h2 class="section-title" id="stageHistoryTitle" style="display:none;">已完成阶段</h2>
<div id="stageHistory" style="margin-bottom:20px;display:flex;flex-wrap:wrap;gap:8px;"></div>

<h2 class="section-title">训练健康指标 <span style="font-size:12px;color:#64748b;font-weight:400;">（约束条件：ent_coef=0.0 纯策略梯度 / target_kl=0.05 KL超阈值自动停更 / clip_range=0.2 策略每次最多变±20%）</span></h2>
<div class="health" id="health"></div>

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
    <h3>探索系数（逐渐下降 = 动作越来越精确）</h3>
    <canvas id="stdChart"></canvas>
  </div>
</div>

<script>
let charts = {};

const chartOpts = {
  responsive: true,
  animation: { duration: 300 },
  plugins: { legend: { display: false } },
  scales: {
    x: { title: { display: true, text: '训练步数 (百万)', color: '#64748b' },
         ticks: { color: '#64748b', maxTicksLimit: 15 }, grid: { color: '#334155' } },
    y: { ticks: { color: '#64748b' }, grid: { color: '#334155' } }
  }
};

function createChart(id, color) {
  return new Chart(document.getElementById(id), {
    type: 'line',
    data: { labels: [], datasets: [{ data: [], borderColor: color, borderWidth: 2, pointRadius: 0, fill: false }] },
    options: chartOpts
  });
}

function initCharts() {
  charts.reward = createChart('rewardChart', '#4ade80');
  charts.len = createChart('lenChart', '#c084fc');
  charts.loss = createChart('lossChart', '#fb923c');
  charts.std = createChart('stdChart', '#38bdf8');
}

function updateChart(chart, labels, data) {
  chart.data.labels = labels;
  chart.data.datasets[0].data = data;
  chart.update('none');
}

function fetchData() {
  fetch('/api/data')
    .then(r => r.json())
    .then(d => {
      if (!d.metrics || d.metrics.length === 0) return;

      const m = d.metrics;
      const latest = m[m.length - 1];
      const rewards = m.map(x => x.ep_rew_mean || 0);
      const epLens = m.map(x => x.ep_len_mean || 0);
      const labels = m.map(x => (x.total_timesteps / 1e6).toFixed(1));

      const lossData = m.filter(x => x.loss !== undefined);
      const lossLabels = lossData.map(x => (x.total_timesteps / 1e6).toFixed(1));
      const losses = lossData.map(x => x.loss);

      const stdData = m.filter(x => x.std !== undefined);
      const stdLabels = stdData.map(x => (x.total_timesteps / 1e6).toFixed(1));
      const stds = stdData.map(x => x.std);

      const total = latest.total_timesteps || 0;
      const elapsed = latest.time_elapsed || 0;
      const fps = latest.fps || 0;
      const bestReward = Math.max(...rewards);
      const curReward = rewards[rewards.length - 1];
      const curLen = epLens[epLens.length - 1];
      const curStd = stds.length > 0 ? stds[stds.length - 1] : 0;
      // Progressive stage info
      const stage = d.stage;
      let stageNum = 1, totalStages = 1, stageTargetSteps = 50e6;
      if (stage && stage.planned_stages && stage.stage_idx !== undefined) {
        stageNum = stage.stage_idx + 1;
        totalStages = stage.total_stages || 1;
        const prevDoneSteps = stage.total_steps || 0;
        if (stage.stage_idx < stage.planned_stages.length) {
          stageTargetSteps = stage.planned_stages[stage.stage_idx] - prevDoneSteps;
        }
      }
      const stageTotalM = (stageTargetSteps / 1e6).toFixed(0);
      const pct = stageTargetSteps > 0 ? (total / stageTargetSteps * 100).toFixed(0) : 100;
      const eta = fps > 0 && stageTargetSteps > total ? ((stageTargetSteps - total) / fps / 3600).toFixed(1) : '?';

      const healthy = curReward > rewards[0] && curReward > 0;
      const isDone = total >= stageTargetSteps;
      // e.g. "第4阶段→50M" when planned_stages available, else empty
      const cumulativeM = stage && stage.planned_stages && stage.stage_idx < stage.planned_stages.length
        ? (stage.planned_stages[stage.stage_idx] / 1e6).toFixed(0) + 'M'
        : null;
      const finalM = stage && stage.planned_stages && stage.planned_stages.length > 0
        ? (stage.planned_stages[stage.planned_stages.length - 1] / 1e6).toFixed(0) + 'M'
        : null;
      const stageLabel = totalStages > 1
        ? ' ' + stageNum + (cumulativeM ? '→' + cumulativeM : '') + '/' + totalStages + (finalM ? '→' + finalM : '')
        : '';

      document.getElementById('statusBadge').className = 'status ' + (isDone ? 'done' : 'running');
      document.getElementById('statusBadge').textContent = isDone ? '已完成' : ('训练中' + stageLabel + ' ' + pct + '%');

      document.getElementById('summary').innerHTML =
        '<b style="color:#38bdf8;">训练总结：</b>' +
        (stageLabel ? '<b style="color:#c084fc;">' + stageLabel.trim() + '</b>，' : '') +
        '本阶段已训练 <b style="color:#38bdf8;">' + (total/1e6).toFixed(1) + 'M</b> / ' + stageTotalM + 'M 步（' + pct + '%），' +
        '耗时 <b>' + (elapsed/3600).toFixed(1) + '</b> 小时' +
        (isDone ? '。' : '，预计剩余 <b>' + eta + '</b> 小时。') +
        '机器狗当前每轮奖励 <b style="color:#4ade80;">' + curReward.toFixed(0) + '</b>，' +
        '历史最高 <b style="color:#facc15;">' + bestReward.toFixed(0) + '</b>。' +
        '每轮存活 <b style="color:#c084fc;">' + curLen.toFixed(0) + '</b>/250 步。' +
        (healthy
          ? ' <b style="color:#4ade80;">训练健康，奖励持续增长。</b>'
          : ' <b style="color:#f87171;">注意：奖励出现下降，可能需要检查。</b>');

      document.getElementById('stats').innerHTML =
        '<div class="stat"><div class="label">本阶段进度</div><div class="value blue">' + (total/1e6).toFixed(1) + 'M / ' + stageTotalM + 'M</div></div>' +
        '<div class="stat"><div class="label">当前奖励</div><div class="value green">' + curReward.toFixed(0) + '</div></div>' +
        '<div class="stat"><div class="label">最高奖励</div><div class="value yellow">' + bestReward.toFixed(0) + '</div></div>' +
        '<div class="stat"><div class="label">存活步数</div><div class="value purple">' + curLen.toFixed(0) + '/250</div></div>' +
        '<div class="stat"><div class="label">训练速度</div><div class="value blue">' + fps.toFixed(0) + ' fps</div></div>' +
        '<div class="stat"><div class="label">已用时间</div><div class="value">' + (elapsed/3600).toFixed(1) + 'h</div></div>';

      // Training health indicators
      const kl = latest.approx_kl || 0;
      const clip = latest.clip_fraction || 0;
      const lr = latest.learning_rate || 0;
      const ev = latest.explained_variance || 0;
      const targetKl = 0.03;

      function healthCard(name, cn, value, hint, colorClass) {
        return '<div class="health-card">' +
          '<div class="h-name">' + name + '</div>' +
          '<div class="h-cn">' + cn + '</div>' +
          '<div class="h-val ' + colorClass + '">' + value + '</div>' +
          '<div class="h-hint ' + colorClass + '">' + hint + '</div></div>';
      }

      const klColor = kl < 0.03 ? 'h-good' : kl < 0.05 ? 'h-warn' : 'h-bad';
      const klHint = kl < 0.03 ? '正常：策略更新稳定' : kl < 0.05 ? '偏高：接近阈值' : '危险：策略更新过大';

      const clipColor = clip < 0.2 ? 'h-good' : clip < 0.3 ? 'h-warn' : 'h-bad';
      const clipHint = clip < 0.2 ? '正常：大部分样本有效' : clip < 0.3 ? '偏高：部分样本被裁剪' : '危险：大量样本被裁剪';

      const stdColor = curStd < 0.8 ? 'h-good' : curStd < 0.95 ? 'h-warn' : 'h-bad';
      const stdHint = curStd < 0.5 ? '收敛：动作非常精确' : curStd < 0.8 ? '正常：策略逐渐收敛' : '初期：仍在探索中';

      const evColor = ev > 0.8 ? 'h-good' : ev > 0.5 ? 'h-warn' : 'h-bad';
      const evHint = ev > 0.8 ? '优秀：价值预测准确' : ev > 0.5 ? '一般：预测有偏差' : '差：价值网络不准';

      document.getElementById('health').innerHTML =
        healthCard('approx_kl', '策略偏移度（每次更新策略变化多大，越小越稳）', kl.toFixed(4), klHint + '（阈值 ' + targetKl + '）', klColor) +
        healthCard('clip_fraction', '裁剪比例（被裁剪的样本占比，越低越好）', (clip * 100).toFixed(1) + '%', clipHint, clipColor) +
        healthCard('learning_rate', '学习率（控制每次学习的步幅，线性衰减到0）', lr.toFixed(6), '当前进度 ' + pct + '%，lr 已衰减到 ' + (lr/3e-4*100).toFixed(0) + '%', 'h-good') +
        healthCard('std', '探索系数（动作随机性，越低=越精确）', curStd.toFixed(3), stdHint, stdColor) +
        healthCard('explained_variance', '预测准确度（价值网络预测奖励的准确性）', (ev * 100).toFixed(1) + '%', evHint, evColor) +
        '';

      // Render completed stage history
      const histEl = document.getElementById('stageHistory');
      const histTitle = document.getElementById('stageHistoryTitle');
      if (histEl && stage && stage.stages && stage.stages.length > 0) {
        histTitle.style.display = '';
        histEl.innerHTML = stage.stages.map(s => {
          const color = s.healthy ? '#4ade80' : '#f87171';
          const trend = s.metrics.reward_trend >= 0
            ? '<span style="color:#4ade80;">↑' + s.metrics.reward_trend.toFixed(0) + '</span>'
            : '<span style="color:#f87171;">↓' + Math.abs(s.metrics.reward_trend).toFixed(0) + '</span>';
          return '<div style="background:#0f172a;border-radius:8px;padding:10px 16px;border:1px solid #334155;min-width:140px;">' +
            '<div style="font-size:12px;color:#94a3b8;">Stage ' + s.name +
            ' <span style="color:' + color + ';">' + (s.healthy ? '✓' : '✗') + '</span></div>' +
            '<div style="font-size:24px;font-weight:700;color:#4ade80;margin:4px 0;">' + s.metrics.reward_final.toFixed(0) + '</div>' +
            '<div style="font-size:11px;color:#64748b;">趋势 ' + trend + ' · ep_len ' + s.metrics.ep_len_final.toFixed(0) + '</div>' +
            '<div style="font-size:11px;color:#475569;margin-top:4px;">' + s.completed_at.replace('T', ' ') + '</div>' +
            '</div>';
        }).join('');
      } else if (histTitle) {
        histTitle.style.display = 'none';
      }

      updateChart(charts.reward, labels, rewards);
      updateChart(charts.len, labels, epLens);
      updateChart(charts.loss, lossLabels, losses);
      updateChart(charts.std, stdLabels, stds);

      document.getElementById('lastUpdate').textContent = '更新于 ' + new Date().toLocaleTimeString();
    })
    .catch(e => console.error('Fetch error:', e));
}

initCharts();
fetchData();
setInterval(fetchData, 15000);
</script>
</body>
</html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    log_file = DEFAULT_LOG

    def do_GET(self):
        if self.path == "/" or self.path == "/dashboard":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode("utf-8"))

        elif self.path == "/api/data":
            metrics = parse_training_log(self.log_file)
            stage = parse_progressive_state()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"metrics": metrics, "stage": stage}).encode("utf-8"))

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress request logs


def main():
    parser = argparse.ArgumentParser(description="JPRobot Training Dashboard Server")
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    parser.add_argument("--port", type=int, default=int(os.getenv("JPROBOT_DASHBOARD_PORT", "18791")))
    parser.add_argument("--log", type=str, default=DEFAULT_LOG)
    args = parser.parse_args()

    DashboardHandler.log_file = args.log

    HTTPServer.allow_reuse_address = True
    server = HTTPServer(("0.0.0.0", args.port), DashboardHandler)
    print(f"JPRobot Training Dashboard")
    print(f"  http://127.0.0.1:{args.port}/dashboard")
    print(f"  Log: {args.log}")
    print(f"  Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
