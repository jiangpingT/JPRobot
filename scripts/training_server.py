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
import threading
import queue
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Default log file (current training)
DEFAULT_LOG = "/tmp/ab_test/jprobot.log"
PROGRESSIVE_STATE = os.path.join(os.path.dirname(__file__), "..", "trained", "progressive_state.json")
POSTURE_EVAL = os.path.join(os.path.dirname(__file__), "..", "trained", "posture_eval.json")
POSTURE_HISTORY = os.path.join(os.path.dirname(__file__), "..", "trained", "posture_eval_history.jsonl")


def parse_progressive_state():
    """Read progressive training state if available."""
    path = os.path.abspath(PROGRESSIVE_STATE)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def parse_posture_eval():
    """Read posture evaluation metrics if available."""
    path = os.path.abspath(POSTURE_EVAL)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def get_curriculum_plan():
    """Return curriculum stage definitions for dashboard display.

    Parses progressive.py source directly to avoid importing pybullet
    (which env.py triggers on import).
    """
    prog_path = os.path.join(os.path.dirname(__file__), "..", "jprobot", "training", "progressive.py")
    prog_path = os.path.abspath(prog_path)
    if not os.path.exists(prog_path):
        return {}
    try:
        import ast
        with open(prog_path) as f:
            tree = ast.parse(f.read())
        # Execute only the constants we need (no side effects)
        # Collect all top-level assignments that CURRICULA may reference
        ns = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and (
                        target.id == 'CURRICULA'
                        or target.id.startswith('_')
                        and target.id.endswith('_CONFIG')
                    ):
                        exec(compile(ast.Module(body=[node], type_ignores=[]), prog_path, 'exec'), ns)
        return ns.get('CURRICULA', {})
    except Exception:
        return {}


def parse_posture_history():
    """Read posture history JSONL for trend charts."""
    path = os.path.abspath(POSTURE_HISTORY)
    if not os.path.exists(path):
        return []
    points = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    points.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return points


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


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class EvalEngine:
    """Background evaluation engine: runs PyBullet DIRECT, broadcasts frames via queues."""

    def __init__(self, trained_dir):
        self.trained_dir = os.path.abspath(trained_dir)
        self._subscribers = []
        self._lock = threading.Lock()
        self.running = False
        self._thread = None
        self.current_model = None
        self.state = "idle"

    def subscribe(self):
        q = queue.Queue(maxsize=100)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q):
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def _broadcast(self, event_type, data):
        msg = json.dumps(data, ensure_ascii=False)
        with self._lock:
            for q in self._subscribers:
                try:
                    q.put_nowait((event_type, msg))
                except queue.Full:
                    pass  # drop frame to prevent memory leak

    def start(self, model_name="best.zip"):
        if self.running:
            return
        self.running = True
        self.current_model = model_name
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False

    def _find_latest_model(self):
        """Find the most recently modified .zip across all subdirectories."""
        best_path, best_mtime = None, 0
        for subdir in ("checkpoints", "snapshots", ""):
            d = os.path.join(self.trained_dir, subdir) if subdir else self.trained_dir
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                if not f.endswith(".zip"):
                    continue
                full = os.path.join(d, f)
                mt = os.path.getmtime(full)
                if mt > best_mtime:
                    best_mtime, best_path = mt, full
        return best_path

    def _resolve_model_path(self, model_name):
        if model_name == "__latest__":
            return self._find_latest_model()
        # Try snapshots/ first, then checkpoints/, then direct path
        for subdir in ("snapshots", "checkpoints", ""):
            path = os.path.join(self.trained_dir, subdir, model_name) if subdir else os.path.join(self.trained_dir, model_name)
            if os.path.exists(path):
                return path
        return None

    def _run_loop(self):
        import pybullet as p
        from stable_baselines3 import PPO
        from jprobot.training.env import BittleGymEnv

        model_path = self._resolve_model_path(self.current_model)
        if not model_path:
            self._broadcast("status", {"state": "error", "message": f"Model not found: {self.current_model}"})
            self.running = False
            self.state = "idle"
            return

        self.state = "loading"
        self._broadcast("status", {"state": "loading", "model": self.current_model})

        try:
            env = BittleGymEnv(render_mode=None)
            model = PPO.load(model_path)
        except Exception as e:
            self._broadcast("status", {"state": "error", "message": str(e)})
            self.running = False
            self.state = "idle"
            return

        self.state = "running"
        self._broadcast("status", {"state": "running", "model": self.current_model})

        last_mtime = os.path.getmtime(model_path)
        episode = 0

        try:
            while self.running:
                episode += 1
                obs, _ = env.reset()
                total_reward = 0.0
                max_distance = 0.0
                start_x = None

                self._broadcast("episode_start", {"episode": episode})

                step = 0
                terminated = False
                while self.running:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    step += 1
                    total_reward += reward

                    pos, orn = p.getBasePositionAndOrientation(
                        env.robot_id, physicsClientId=env.physics_client
                    )
                    if start_x is None:
                        start_x = pos[0]
                    distance = pos[0] - start_x
                    max_distance = max(max_distance, distance)

                    # Joint angles in action-space order (matches plan mapping)
                    joints = []
                    for ji in env.joint_id:
                        js = p.getJointState(env.robot_id, ji, physicsClientId=env.physics_client)
                        joints.append(round(js[0], 4))

                    self._broadcast("frame", {
                        "step": step,
                        "position": [round(v, 5) for v in pos],
                        "orientation": [round(v, 5) for v in orn],
                        "joints": joints,
                        "reward": round(float(reward), 2),
                    })

                    time.sleep(0.02)

                    if terminated or truncated:
                        break

                self._broadcast("episode_end", {
                    "episode": episode,
                    "total_reward": round(total_reward, 2),
                    "steps": step,
                    "survived": not terminated,
                    "max_distance": round(max_distance, 4),
                })

                # Check if model file was updated (or find newer file in __latest__ mode)
                try:
                    if self.current_model == "__latest__":
                        newest = self._find_latest_model()
                        if newest and (newest != model_path or os.path.getmtime(newest) != last_mtime):
                            model_path = newest
                            last_mtime = os.path.getmtime(model_path)
                            model = PPO.load(model_path)
                            short = os.path.basename(model_path)
                            self._broadcast("model_updated", {
                                "file": short,
                                "mtime": last_mtime,
                            })
                    else:
                        cur_mtime = os.path.getmtime(model_path)
                        if cur_mtime != last_mtime:
                            last_mtime = cur_mtime
                            model = PPO.load(model_path)
                            self._broadcast("model_updated", {
                                "file": self.current_model,
                                "mtime": cur_mtime,
                            })
                except Exception:
                    pass

        finally:
            try:
                env.close()
            except Exception:
                pass
            self.state = "idle"
            self.running = False
            self._broadcast("status", {"state": "stopped"})


# Global eval engine instance (created in main)
eval_engine = None


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
  <a class="btn" href="/viz" style="text-decoration:none;display:inline-block;">3D 可视化</a>
</div>

<div class="summary" id="summary">加载中...</div>

<div class="stats" id="stats"></div>

<div id="curriculumInfo" style="display:none;background:#1e293b;border-radius:10px;padding:14px 18px;margin-bottom:20px;border-left:3px solid #c084fc;">
</div>

<h2 class="section-title" id="postureTitle" style="display:none;">行为分析（机器狗是在走路还是在爬行？）</h2>
<div class="health" id="posture" style="margin-bottom:20px;"></div>

<h2 class="section-title" id="postureTrendTitle" style="display:none;">姿态趋势（课程学习核心指标）</h2>
<div class="charts" id="postureTrendCharts" style="display:none;margin-bottom:24px;">
  <div class="chart-box">
    <h3>身体高度（目标 > 0.07m = 站立）</h3>
    <canvas id="heightChart"></canvas>
  </div>
  <div class="chart-box">
    <h3>手臂触地率（目标 < 30% = 用腿走路）</h3>
    <canvas id="armContactChart"></canvas>
  </div>
</div>

<h2 class="section-title" id="stageHistoryTitle" style="display:none;">已完成阶段</h2>
<div id="stageHistory" style="margin-bottom:20px;display:flex;flex-wrap:wrap;gap:8px;"></div>

<h2 class="section-title">训练健康指标 <span style="font-size:12px;color:#64748b;font-weight:400;">（约束条件：ent_coef=0.0 纯策略梯度 / lr=3e-4 常数学习率 / clip_range=0.2 策略每次最多变±20%）</span></h2>
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

function createChartWithTarget(id, color, targetVal, targetLabel) {
  const opts = JSON.parse(JSON.stringify(chartOpts));
  opts.plugins = {
    legend: { display: false },
    annotation: { annotations: {
      target: { type: 'line', yMin: targetVal, yMax: targetVal,
                borderColor: '#facc15', borderWidth: 1, borderDash: [6, 3],
                label: { display: true, content: targetLabel, position: 'end',
                         color: '#facc15', font: { size: 10 } } }
    }}
  };
  return new Chart(document.getElementById(id), {
    type: 'line',
    data: { labels: [], datasets: [{ data: [], borderColor: color, borderWidth: 2, pointRadius: 0, fill: false }] },
    options: opts
  });
}

function initCharts() {
  charts.reward = createChart('rewardChart', '#4ade80');
  charts.len = createChart('lenChart', '#c084fc');
  charts.loss = createChart('lossChart', '#fb923c');
  charts.std = createChart('stdChart', '#38bdf8');
  charts.height = createChart('heightChart', '#38bdf8');
  charts.armContact = createChart('armContactChart', '#fb923c');
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
      // Progressive stage info — supports both curriculum and classic modes
      const stage = d.stage;
      let stageNum = 1, totalStages = 1, stageTargetSteps = 50e6, currentStageName = '';
      let globalTotalSteps = 50e6, globalDoneSteps = 0;
      const isCurriculum = stage && stage.curriculum;
      const curPlan = isCurriculum && d.curricula && d.curricula[stage.curriculum]
        ? d.curricula[stage.curriculum].stages : null;

      if (isCurriculum && curPlan) {
        const curIdx = stage.curriculum_stage_idx || 0;
        totalStages = curPlan.length;
        stageNum = curIdx + 1;
        globalTotalSteps = curPlan[curPlan.length - 1].target;
        globalDoneSteps = curIdx > 0 ? curPlan[curIdx - 1].target : 0;
        if (curIdx < curPlan.length) {
          const curStageDef = curPlan[curIdx];
          const prevTarget = curIdx > 0 ? curPlan[curIdx - 1].target : 0;
          stageTargetSteps = curStageDef.target - prevTarget;
          currentStageName = curStageDef.name || '';
        }
      } else if (stage && stage.planned_stages && stage.stage_idx !== undefined) {
        stageNum = stage.stage_idx + 1;
        totalStages = stage.total_stages || 1;
        globalTotalSteps = stage.planned_stages[stage.planned_stages.length - 1];
        globalDoneSteps = stage.total_steps || 0;
        if (stage.stage_idx < stage.planned_stages.length) {
          stageTargetSteps = stage.planned_stages[stage.stage_idx] - globalDoneSteps;
        }
      }

      function curStageCn(name) {
        if (!name) return '';
        const n = name.toLowerCase();
        if (n.includes('learn')) return '基础学习';
        if (n.includes('refine')) return '精化训练';
        if (n.includes('forward')) return '前进训练';
        if (n.includes('height')) return '站立训练';
        if (n.includes('stand')) return '站立训练';
        if (n.includes('walk')) return '走路训练';
        return name;
      }
      const stageTypeCn = curStageCn(currentStageName);

      // Overall progress (across all stages)
      const globalCurrent = globalDoneSteps + total;
      const globalPct = globalTotalSteps > 0 ? (globalCurrent / globalTotalSteps * 100).toFixed(0) : 100;
      const globalEta = fps > 0 && globalTotalSteps > globalCurrent
        ? ((globalTotalSteps - globalCurrent) / fps / 3600).toFixed(1) : '?';
      // Current stage progress
      const stagePct = stageTargetSteps > 0 ? (total / stageTargetSteps * 100).toFixed(0) : 100;

      // Health check: use last 20% of rewards to detect real decline vs penalty growth
      const tail5 = rewards.slice(-Math.max(1, Math.floor(rewards.length/5)));
      const head5 = rewards.slice(0, Math.max(1, Math.floor(rewards.length/5)));
      const tailAvg = tail5.reduce((a,b) => a+b, 0) / tail5.length;
      const headAvg = head5.reduce((a,b) => a+b, 0) / head5.length;
      // Reward decline is normal after ~1.5M steps due to penalty_factor growth
      // Only warn if decline happens early (before 50% of training)
      const earlyDecline = (globalCurrent / globalTotalSteps < 0.5) && tailAvg < headAvg - 50;
      const healthy = !earlyDecline && curLen > 50;
      const isDone = globalCurrent >= globalTotalSteps;

      // Status badge
      const stageTag = totalStages > 1
        ? (stageTypeCn ? stageTypeCn + ' ' + stageNum + '/' + totalStages : stageNum + '/' + totalStages)
        : (stageTypeCn || '');
      document.getElementById('statusBadge').className = 'status ' + (isDone ? 'done' : 'running');
      document.getElementById('statusBadge').textContent = isDone
        ? '已完成 · 奖励 ' + bestReward.toFixed(0)
        : ('训练中 ' + (stageTag ? stageTag + ' ' : '') + globalPct + '%');

      // Summary text
      const stageDesc = isCurriculum && totalStages > 1
        ? '<b style="color:#c084fc;">' + stageTypeCn + '（' + currentStageName + '）</b>，第 ' + stageNum + '/' + totalStages + ' 阶段，'
        : (totalStages > 1 ? '<b style="color:#c084fc;">第 ' + stageNum + '/' + totalStages + ' 阶段</b>，' : '');
      const etaText = isDone ? '' : '预计剩余 <b>' + globalEta + '</b> 小时。';
      const healthText = isDone
        ? (curLen > 200
          ? ' <b style="color:#4ade80;">训练完成！机器人已学会行走。</b>'
          : ' <b style="color:#facc15;">训练完成，但存活步数偏低，可能需要额外训练。</b>')
        : (healthy
          ? ' <b style="color:#4ade80;">训练健康。</b>'
          : ' <b style="color:#f87171;">早期奖励下降，可能需要检查。</b>');
      document.getElementById('summary').innerHTML =
        '<b style="color:#38bdf8;">训练总结：</b>' + stageDesc +
        '进度 <b style="color:#38bdf8;">' + (globalCurrent/1e6).toFixed(1) + 'M</b> / ' + (globalTotalSteps/1e6).toFixed(0) + 'M 步（' + globalPct + '%）。' +
        etaText +
        '当前奖励 <b style="color:#4ade80;">' + curReward.toFixed(0) + '</b>，' +
        '历史最高 <b style="color:#facc15;">' + bestReward.toFixed(0) + '</b>。' +
        '存活 <b style="color:#c084fc;">' + curLen.toFixed(0) + '</b>/250 步。' +
        healthText;

      document.getElementById('stats').innerHTML =
        '<div class="stat"><div class="label">整体进度</div><div class="value blue">' + (globalCurrent/1e6).toFixed(1) + 'M / ' + (globalTotalSteps/1e6).toFixed(0) + 'M</div></div>' +
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
        healthCard('approx_kl', '策略偏移度（每次更新策略变化多大，越小越稳）', kl.toFixed(4), klHint + '（无 target_kl 限制，SB3 默认）', klColor) +
        healthCard('clip_fraction', '裁剪比例（被裁剪的样本占比，越低越好）', (clip * 100).toFixed(1) + '%', clipHint, clipColor) +
        healthCard('learning_rate', '学习率（控制每次学习的步幅，固定值）', lr.toFixed(6), '常数学习率 3e-4（SB3 默认值）', 'h-good') +
        healthCard('std', '探索系数（动作随机性，越低=越精确）', curStd.toFixed(3), stdHint, stdColor) +
        healthCard('explained_variance', '预测准确度（价值网络预测奖励的准确性）', (ev * 100).toFixed(1) + '%', evHint, evColor) +
        '';

      // Render curriculum info with full stage plan
      const curInfoEl = document.getElementById('curriculumInfo');
      if (curInfoEl && stage && stage.curriculum) {
        curInfoEl.style.display = '';
        const curName = stage.curriculum;
        const curIdx = stage.curriculum_stage_idx || 0;
        const curTotal = stage.total_stages || 0;
        const completedStages = stage.stages || [];
        // Get plan from curricula definition
        const plan = d.curricula && d.curricula[curName] ? d.curricula[curName].stages : [];

        // Stage name to Chinese
        function stageLabel(name) {
          if (!name) return '未知';
          const n = name.toLowerCase();
          if (n.includes('learn')) return '基础学习';
          if (n.includes('refine')) return '精化训练';
          if (n.includes('forward')) return '前进训练';
          if (n.includes('height')) return '站立训练';
          if (n.includes('stand')) return '站立训练';
          if (n.includes('walk')) return '走路训练';
          return name;
        }
        function stageColor(name) {
          if (!name) return '#94a3b8';
          const n = name.toLowerCase();
          if (n.includes('learn')) return '#38bdf8';
          if (n.includes('refine')) return '#c084fc';
          if (n.includes('forward')) return '#38bdf8';
          if (n.includes('height')) return '#4ade80';
          if (n.includes('stand')) return '#38bdf8';
          if (n.includes('walk')) return '#4ade80';
          return '#94a3b8';
        }

        // Title
        const curDesc = d.curricula[curName] ? d.curricula[curName].description : curName;
        const stageProgress = curTotal > 1
          ? '已完成 ' + curIdx + '/' + curTotal + ' 个阶段'
          : (curIdx >= curTotal ? '已完成' : '训练中');
        let html = '<div style="font-size:14px;color:#c084fc;font-weight:600;margin-bottom:10px;">' +
          '课程：' + curName + ' · ' + stageProgress +
          '<span style="font-size:11px;color:#64748b;margin-left:8px;">(' + curDesc + ')</span></div>';

        // Stage pipeline
        html += '<div style="display:flex;gap:6px;flex-wrap:wrap;align-items:center;">';
        const stageList = plan.length > 0 ? plan : [];
        for (let i = 0; i < Math.max(stageList.length, curTotal); i++) {
          const def = stageList[i] || {};
          const name = def.name || (completedStages[i] ? completedStages[i].name : '阶段' + (i+1));
          const label = stageLabel(name);
          const color = stageColor(name);
          const steps = def.target ? (def.target / 1e6).toFixed(0) + 'M' : '';
          const ent = def.ent_coef !== undefined ? def.ent_coef : '';
          const ec = def.env_config || {};
          const hb = ec.fac_height_bonus || '';

          let status, bg, border, textColor;
          if (i < curIdx) {
            // Completed
            const completed = completedStages[i];
            const healthy = completed && completed.healthy;
            status = healthy ? '✓' : '✗';
            bg = healthy ? color + '15' : '#f8717115';
            border = healthy ? color : '#f87171';
            textColor = color;
          } else if (i === curIdx) {
            // Current
            status = '▶ 训练中';
            bg = color + '25';
            border = color;
            textColor = '#fff';
          } else {
            // Pending
            status = '待训练';
            bg = '#1e293b';
            border = '#334155';
            textColor = '#64748b';
          }

          html += '<div style="background:' + bg + ';border:1px solid ' + border +
            ';border-radius:8px;padding:8px 12px;min-width:120px;text-align:center;">';
          html += '<div style="font-size:11px;color:' + textColor + ';font-weight:600;">' +
            label + '</div>';
          html += '<div style="font-size:10px;color:#94a3b8;margin:2px 0;">' + name;
          if (steps) html += ' · ' + steps + '步';
          html += '</div>';
          // Show key params
          if (ent !== '') {
            html += '<div style="font-size:9px;color:#64748b;">ent=' + ent;
            if (hb) html += ' h_bonus=' + hb;
            html += '</div>';
          }
          html += '<div style="font-size:10px;color:' + (i < curIdx ? (completedStages[i] && completedStages[i].healthy ? '#4ade80' : '#f87171') : textColor) +
            ';margin-top:3px;font-weight:600;">' + status + '</div>';
          // Show reward for completed stages
          if (i < curIdx && completedStages[i]) {
            html += '<div style="font-size:10px;color:#94a3b8;">奖励 ' +
              completedStages[i].metrics.reward_final.toFixed(0) + '</div>';
          }
          html += '</div>';

          // Arrow between stages
          if (i < Math.max(stageList.length, curTotal) - 1) {
            html += '<div style="color:#475569;font-size:16px;">→</div>';
          }
        }
        html += '</div>';

        curInfoEl.innerHTML = html;
      } else if (curInfoEl) {
        curInfoEl.style.display = 'none';
      }

      // Render posture analysis
      const postureEl = document.getElementById('posture');
      const postureTitle = document.getElementById('postureTitle');
      const posture = d.posture;
      if (postureEl && posture) {
        postureTitle.style.display = '';
        const isWalking = posture.behavior === 'walking';
        const behaviorColor = isWalking ? 'h-good' : 'h-warn';
        const behaviorText = isWalking ? '走路' : '爬行';
        const behaviorHint = isWalking ? '身体高度正常，手臂未撑地' : '身体低伏，手臂频繁触地';

        const h = posture.avg_height;
        const heightColor = h > 0.06 ? 'h-good' : h > 0.04 ? 'h-warn' : 'h-bad';
        const heightHint = h > 0.06 ? '站立高度正常（参考：站立≈0.07m）'
                         : h > 0.04 ? '偏低，介于爬行与站立之间'
                         : '爬行高度（身体贴地）';

        const ac = posture.arm_contact_rate * 100;
        const acColor = ac < 20 ? 'h-good' : ac < 50 ? 'h-warn' : 'h-bad';
        const acHint = ac < 20 ? '手臂几乎不触地，走路姿态'
                     : ac < 50 ? '手臂偶尔触地，姿态欠佳'
                     : '手臂频繁撑地，典型爬行';

        const tilt = posture.avg_tilt_deg;
        const tiltColor = tilt < 10 ? 'h-good' : tilt < 20 ? 'h-warn' : 'h-bad';
        const tiltHint = tilt < 10 ? '姿态端正' : tilt < 20 ? '轻微前倾' : '明显前倾/侧倾';

        const ep = posture.episodes_sampled || 0;
        const updatedAt = posture.updated_at ? posture.updated_at.replace('T', ' ') : '';

        postureEl.innerHTML =
          healthCard('行为判断', '基于身体高度和手臂触地率综合判断', behaviorText, behaviorHint + '（样本 ' + ep + ' 局，更新 ' + updatedAt + '）', behaviorColor) +
          healthCard('平均身体高度', '站立≈0.07m / 爬行≈0.03m', h.toFixed(4) + 'm', heightHint, heightColor) +
          healthCard('手臂触地频率', '手臂/肘关节接触地面的步数占比', ac.toFixed(1) + '%', acHint, acColor) +
          healthCard('平均姿态倾角', '机体 roll+pitch 综合倾斜角', tilt.toFixed(1) + '°', tiltHint, tiltColor);
      } else if (postureTitle) {
        postureTitle.style.display = 'none';
      }

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
          // Detect stage type from name, then env_config
          let typeTag = '';
          const sn = (s.name || '').toLowerCase();
          const tagMap = {
            'learn': ['#38bdf8', 'Learn'], 'refine': ['#c084fc', 'Refine'],
            'forward': ['#38bdf8', 'Forward'], 'height': ['#4ade80', 'Height'],
            'stand': ['#38bdf8', 'Stand'], 'walk': ['#4ade80', 'Walk'],
          };
          for (const [key, [tc, tt]] of Object.entries(tagMap)) {
            if (sn.includes(key)) {
              typeTag = '<span style="display:inline-block;background:' + tc + '22;color:' + tc +
                ';padding:1px 6px;border-radius:4px;font-size:10px;margin-left:4px;">' + tt + '</span>';
              break;
            }
          }
          return '<div style="background:#0f172a;border-radius:8px;padding:10px 16px;border:1px solid #334155;min-width:140px;">' +
            '<div style="font-size:12px;color:#94a3b8;">' + s.name + typeTag +
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

      // Render posture trend charts
      const ph = d.posture_history;
      const trendTitle = document.getElementById('postureTrendTitle');
      const trendCharts = document.getElementById('postureTrendCharts');
      if (ph && ph.length > 0 && trendTitle && trendCharts) {
        trendTitle.style.display = '';
        trendCharts.style.display = '';
        // Downsample to max 200 points for performance
        const maxPts = 200;
        const stride = Math.max(1, Math.floor(ph.length / maxPts));
        const sampled = ph.filter((_, i) => i % stride === 0);
        const phLabels = sampled.map(p => (p.step / 1e6).toFixed(1));
        const heights = sampled.map(p => p.h);
        const armContacts = sampled.map(p => p.ac * 100);
        updateChart(charts.height, phLabels, heights);
        updateChart(charts.armContact, phLabels, armContacts);
      } else if (trendTitle) {
        trendTitle.style.display = 'none';
        if (trendCharts) trendCharts.style.display = 'none';
      }

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


VISUALIZATION_HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>JPRobot 3D 可视化</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #0f172a; overflow: hidden; font-family: -apple-system, BlinkMacSystemFont, sans-serif; }
  #canvas3d { width: 100vw; height: 100vh; display: block; }

  #controls {
    position: fixed; top: 12px; right: 12px; z-index: 10;
    background: rgba(15,23,42,0.92); border-radius: 10px; padding: 12px 16px;
    color: #e2e8f0; font-size: 13px; min-width: 200px;
    backdrop-filter: blur(8px); border: 1px solid #334155;
  }
  #controls select, #controls button {
    display: block; width: 100%; margin-top: 8px; padding: 7px 10px;
    border-radius: 6px; border: 1px solid #475569; background: #1e293b;
    color: #e2e8f0; font-size: 13px; cursor: pointer;
  }
  #controls button { background: #6366f1; border-color: #6366f1; font-weight: 600; }
  #controls button:hover { background: #818cf8; }
  #controls button.stop { background: #dc2626; border-color: #dc2626; }
  #controls button.stop:hover { background: #ef4444; }

  #stats {
    position: fixed; top: 12px; left: 12px; z-index: 10;
    background: rgba(15,23,42,0.92); border-radius: 10px; padding: 14px 18px;
    color: #e2e8f0; font-size: 13px; line-height: 1.7;
    backdrop-filter: blur(8px); border: 1px solid #334155; min-width: 180px;
  }
  #stats .label { color: #94a3b8; }
  #stats .val { color: #38bdf8; font-weight: 600; }
  #stats .val.green { color: #4ade80; }
  #stats .val.yellow { color: #facc15; }

  #connStatus {
    position: fixed; bottom: 12px; left: 12px; z-index: 10;
    padding: 4px 12px; border-radius: 12px; font-size: 11px;
    background: #334155; color: #94a3b8;
  }
  #connStatus.connected { background: #166534; color: #4ade80; }
  #connStatus.error { background: #7f1d1d; color: #fca5a5; }

  #backBtn {
    position: fixed; bottom: 12px; right: 12px; z-index: 10;
    padding: 6px 16px; border-radius: 8px; font-size: 12px;
    background: #1e293b; color: #94a3b8; text-decoration: none;
    border: 1px solid #334155;
  }
  #backBtn:hover { color: #e2e8f0; background: #334155; }
</style>
</head>
<body>

<canvas id="canvas3d"></canvas>

<div id="stats">
  <div><span class="label">Episode: </span><span class="val" id="sEpisode">-</span></div>
  <div><span class="label">Step: </span><span class="val" id="sStep">-</span></div>
  <div><span class="label">Reward: </span><span class="val green" id="sReward">-</span></div>
  <div><span class="label">Distance: </span><span class="val yellow" id="sDistance">-</span></div>
  <div><span class="label">Speed: </span><span class="val" id="sSpeed">-</span></div>
  <div><span class="label">Model: </span><span class="val" id="sModel">-</span></div>
</div>

<div id="controls">
  <label style="color:#94a3b8;">模型选择</label>
  <select id="modelSelect"><option value="best.zip">best.zip</option></select>
  <button id="startBtn" onclick="doControl('start')">开始</button>
</div>

<div id="connStatus">未连接</div>
<a id="backBtn" href="/dashboard">← 训练面板</a>

<script type="importmap">
{
  "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
  }
}
</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ─── Scene setup ────────────────────────────────────────────
const canvas = document.getElementById('canvas3d');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a2e);
scene.fog = new THREE.Fog(0x1a1a2e, 2, 6);

// Z-up camera (match PyBullet)
const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.01, 20);
camera.up.set(0, 0, 1);
camera.position.set(0.25, -0.3, 0.25);

const controls = new OrbitControls(camera, canvas);
controls.target.set(0, 0, 0.05);
controls.enableDamping = true;
controls.dampingFactor = 0.1;
controls.update();

// ─── Lights ─────────────────────────────────────────────────
scene.add(new THREE.AmbientLight(0xffffff, 0.6));
const dirLight = new THREE.DirectionalLight(0xffffff, 1.0);
dirLight.position.set(1, -1, 2);
dirLight.castShadow = true;
dirLight.shadow.mapSize.set(1024, 1024);
dirLight.shadow.camera.left = -1; dirLight.shadow.camera.right = 1;
dirLight.shadow.camera.top = 1; dirLight.shadow.camera.bottom = -1;
scene.add(dirLight);

// ─── Ground plane ───────────────────────────────────────────
const groundGeo = new THREE.PlaneGeometry(6, 2);
const groundMat = new THREE.MeshStandardMaterial({ color: 0x2a2a3e, roughness: 0.9 });
const ground = new THREE.Mesh(groundGeo, groundMat);
ground.receiveShadow = true;
ground.position.set(1.5, 0, -0.001);
scene.add(ground);

// Grid lines (10cm spacing) on XY plane at Z=0
const gridGroup = new THREE.Group();
const gridMat = new THREE.LineBasicMaterial({ color: 0x3a3a5e, transparent: true, opacity: 0.5 });
for (let i = -5; i <= 55; i++) {
  const x = i * 0.1;
  const pts = [new THREE.Vector3(x, -1, 0.0005), new THREE.Vector3(x, 1, 0.0005)];
  gridGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), gridMat));
}
for (let j = -10; j <= 10; j++) {
  const y = j * 0.1;
  const pts = [new THREE.Vector3(-0.5, y, 0.0005), new THREE.Vector3(5.5, y, 0.0005)];
  gridGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), gridMat));
}
scene.add(gridGroup);

// ─── Robot model ────────────────────────────────────────────
// URDF dimensions (meters)
const TORSO_SIZE = [0.105, 0.08, 0.02];
const UPPER_R = 0.003, UPPER_L = 0.045;
const LOWER_R = 0.003, LOWER_L = 0.05;
const PAW_R = 0.006, PAW_L = 0.008;

// Colors
const matTorso = new THREE.MeshStandardMaterial({ color: 0xf0c040, roughness: 0.5 });
const matBattery = new THREE.MeshStandardMaterial({ color: 0x222222, roughness: 0.6 });
const matUpper = new THREE.MeshStandardMaterial({ color: 0x4488cc, roughness: 0.4 });
const matLower = new THREE.MeshStandardMaterial({ color: 0x222222, roughness: 0.5 });
const matPaw = new THREE.MeshStandardMaterial({ color: 0x111111, roughness: 0.6 });

const robotGroup = new THREE.Group();
scene.add(robotGroup);

// Torso
const torsoMesh = new THREE.Mesh(new THREE.BoxGeometry(...TORSO_SIZE), matTorso);
torsoMesh.castShadow = true;
robotGroup.add(torsoMesh);

// Battery
const batteryMesh = new THREE.Mesh(new THREE.BoxGeometry(0.09, 0.04, 0.015), matBattery);
batteryMesh.position.set(0, 0, -0.015);
robotGroup.add(batteryMesh);

// Leg origins from URDF (xyz relative to torso center)
const legConfig = [
  { name: 'FL', ox:  0.055, oy:  0.05, elbowOy: -0.015, sign: 1 },  // front-left
  { name: 'FR', ox:  0.055, oy: -0.05, elbowOy:  0.015, sign: -1 }, // front-right
  { name: 'BR', ox: -0.055, oy: -0.05, elbowOy:  0.015, sign: -1 }, // back-right
  { name: 'BL', ox: -0.055, oy:  0.05, elbowOy: -0.015, sign: 1 },  // back-left
];

// Build leg hierarchy: shoulderPivot → upperLimb → elbowStatic → elbowPivot → lowerLimb → paw
const legs = {};
legConfig.forEach(cfg => {
  // Shoulder pivot (at URDF joint origin, rotates around Y)
  const shoulderPivot = new THREE.Group();
  shoulderPivot.position.set(cfg.ox, cfg.oy, 0);
  robotGroup.add(shoulderPivot);

  // Upper limb (cylinder along Z, offset so top is at pivot)
  const upperGeo = new THREE.CylinderGeometry(UPPER_R, UPPER_R, UPPER_L, 8);
  const upperMesh = new THREE.Mesh(upperGeo, matUpper);
  // Cylinder default axis is Y; we need it along -Z
  // Rotate cylinder 90° around X to align with -Z
  upperMesh.rotation.x = Math.PI / 2;
  upperMesh.position.set(0, 0, -UPPER_L / 2);
  upperMesh.castShadow = true;
  shoulderPivot.add(upperMesh);

  // Elbow static transform (URDF: rpy(0, -pi/2, 0) at elbow origin)
  const elbowStatic = new THREE.Group();
  elbowStatic.position.set(0, cfg.elbowOy, -UPPER_L);
  // Apply URDF rpy(0, -pi/2, 0) — in Three.js ZYX convention, this is rotation around Y by -pi/2
  // But since our coordinate system is Z-up, we need to convert:
  // URDF rpy(0, -pi/2, 0) means pitch = -90° which rotates the child joint axis
  elbowStatic.rotation.y = -Math.PI / 2;
  shoulderPivot.add(elbowStatic);

  // Elbow pivot (dynamic rotation around Y axis = joint angle)
  const elbowPivot = new THREE.Group();
  elbowStatic.add(elbowPivot);

  // Lower limb
  const lowerGeo = new THREE.CylinderGeometry(LOWER_R, LOWER_R, LOWER_L, 8);
  const lowerMesh = new THREE.Mesh(lowerGeo, matLower);
  lowerMesh.rotation.x = Math.PI / 2;
  lowerMesh.position.set(0, 0, -LOWER_L / 2);
  lowerMesh.castShadow = true;
  elbowPivot.add(lowerMesh);

  // Paw (fixed at end of lower limb)
  const pawGeo = new THREE.CylinderGeometry(PAW_R, PAW_R, PAW_L, 8);
  const pawMesh = new THREE.Mesh(pawGeo, matPaw);
  pawMesh.rotation.x = Math.PI / 2;
  pawMesh.position.set(0, 0, -LOWER_L - PAW_L / 2);
  elbowPivot.add(pawMesh);

  legs[cfg.name] = { shoulderPivot, elbowPivot };
});

// Joint mapping from SSE joints[] to Three.js pivots
// joints[0]=shoulder_left(FL), [1]=elbow_left(FL),
// joints[2]=shoulder_right(FR), [3]=elbow_right(FR),
// joints[4]=hip_right(BR), [5]=knee_right(BR),
// joints[6]=hip_left(BL), [7]=knee_left(BL)
const jointMap = [
  { pivot: legs.FL.shoulderPivot, prop: 'y' },  // 0: shoulder_left
  { pivot: legs.FL.elbowPivot,    prop: 'y' },  // 1: elbow_left
  { pivot: legs.FR.shoulderPivot, prop: 'y' },  // 2: shoulder_right
  { pivot: legs.FR.elbowPivot,    prop: 'y' },  // 3: elbow_right
  { pivot: legs.BR.shoulderPivot, prop: 'y' },  // 4: hip_right
  { pivot: legs.BR.elbowPivot,    prop: 'y' },  // 5: knee_right
  { pivot: legs.BL.shoulderPivot, prop: 'y' },  // 6: hip_left
  { pivot: legs.BL.elbowPivot,    prop: 'y' },  // 7: knee_left
];

// ─── Trajectory trail ───────────────────────────────────────
const MAX_TRAIL = 2000;
const trailPositions = new Float32Array(MAX_TRAIL * 3);
const trailColors = new Float32Array(MAX_TRAIL * 3);
const trailGeo = new THREE.BufferGeometry();
trailGeo.setAttribute('position', new THREE.BufferAttribute(trailPositions, 3));
trailGeo.setAttribute('color', new THREE.BufferAttribute(trailColors, 3));
const trailMat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.7 });
const trailLine = new THREE.Line(trailGeo, trailMat);
scene.add(trailLine);
let trailCount = 0;

function addTrailPoint(x, y, z) {
  if (trailCount >= MAX_TRAIL) return;
  const i = trailCount * 3;
  trailPositions[i] = x; trailPositions[i+1] = y; trailPositions[i+2] = z + 0.002;
  // Green to cyan gradient
  const t = Math.min(trailCount / 250, 1);
  trailColors[i] = 0.2 * t; trailColors[i+1] = 0.9 - 0.2*t; trailColors[i+2] = 0.3 + 0.5*t;
  trailCount++;
  trailGeo.setDrawRange(0, trailCount);
  trailGeo.attributes.position.needsUpdate = true;
  trailGeo.attributes.color.needsUpdate = true;
}

function resetTrail() {
  trailCount = 0;
  trailGeo.setDrawRange(0, 0);
}

// ─── State ──────────────────────────────────────────────────
let currentEpisode = 0, currentStep = 0, totalReward = 0, maxDistance = 0;
let prevX = 0, speed = 0, modelName = '-';
let cameraTarget = new THREE.Vector3(0, 0, 0.05);
let isRunning = false;

// ─── SSE connection ─────────────────────────────────────────
let evtSource = null;
function connectSSE() {
  if (evtSource) evtSource.close();
  evtSource = new EventSource('/api/viz/stream');
  const cs = document.getElementById('connStatus');

  evtSource.onopen = () => { cs.textContent = '已连接'; cs.className = 'connected'; };
  evtSource.onerror = () => { cs.textContent = '连接断开'; cs.className = 'error'; };

  evtSource.addEventListener('status', e => {
    const d = JSON.parse(e.data);
    if (d.model) modelName = d.model === '__latest__' ? '\u5b9e\u65f6' : d.model;
    document.getElementById('sModel').textContent = modelName;
    if (d.state === 'running') {
      isRunning = true;
      updateBtn(true);
    } else if (d.state === 'stopped' || d.state === 'error') {
      isRunning = false;
      updateBtn(false);
    }
  });

  evtSource.addEventListener('episode_start', e => {
    const d = JSON.parse(e.data);
    currentEpisode = d.episode;
    currentStep = 0; totalReward = 0; maxDistance = 0; prevX = 0; speed = 0;
    resetTrail();
    document.getElementById('sEpisode').textContent = currentEpisode;
  });

  evtSource.addEventListener('frame', e => {
    const d = JSON.parse(e.data);
    currentStep = d.step;
    totalReward += d.reward;

    // Update robot position & orientation (PyBullet: x=forward, y=left, z=up)
    const [px, py, pz] = d.position;
    const [qx, qy, qz, qw] = d.orientation;
    robotGroup.position.set(px, py, pz);
    robotGroup.quaternion.set(qx, qy, qz, qw);

    // Update joint angles
    d.joints.forEach((angle, i) => {
      if (jointMap[i]) {
        jointMap[i].pivot.rotation[jointMap[i].prop] = angle;
      }
    });

    // Distance & speed
    const dist = px - (d.step === 1 ? px : 0);
    maxDistance = Math.max(maxDistance, dist);
    speed = (px - prevX) / 0.02;
    prevX = px;

    // Trail (every 3 steps)
    if (d.step % 3 === 0) addTrailPoint(px, py, pz);

    // Camera follow (smooth lerp)
    cameraTarget.lerp(new THREE.Vector3(px, py, 0.05), 0.05);
    controls.target.copy(cameraTarget);

    // Update stats
    document.getElementById('sStep').textContent = currentStep + '/250';
    document.getElementById('sReward').textContent = totalReward.toFixed(0);
    document.getElementById('sDistance').textContent = maxDistance.toFixed(3) + 'm';
    document.getElementById('sSpeed').textContent = speed.toFixed(3) + ' m/s';
  });

  evtSource.addEventListener('episode_end', e => {
    const d = JSON.parse(e.data);
    document.getElementById('sReward').textContent = d.total_reward.toFixed(0) + (d.survived ? ' ✓' : ' ✗');
    document.getElementById('sDistance').textContent = d.max_distance.toFixed(3) + 'm';
    // Timelapse: advance to next model after episode ends
    if (timelapseQueue.length > 0) {
      timelapseIdx++;
      if (timelapseIdx < timelapseQueue.length) {
        const next = timelapseQueue[timelapseIdx];
        document.getElementById('sModel').textContent = '\u23f3 ' + next.name;
        setTimeout(() => {
          fetch('/api/viz/control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: 'start', model: next.file }),
          });
        }, 1000);
      } else {
        timelapseQueue = [];
        document.getElementById('sModel').textContent = '\u2705 过程回放完成';
      }
    }
  });

  evtSource.addEventListener('model_updated', e => {
    const d = JSON.parse(e.data);
    modelName = d.file;
    document.getElementById('sModel').textContent = modelName + ' (更新)';
  });
}

// ─── Controls ───────────────────────────────────────────────
function updateBtn(running) {
  const btn = document.getElementById('startBtn');
  if (running) {
    btn.textContent = '停止';
    btn.className = 'stop';
    btn.setAttribute('onclick', "doControl('stop')");
  } else {
    btn.textContent = '开始';
    btn.className = '';
    btn.setAttribute('onclick', "doControl('start')");
  }
}

let timelapseQueue = [];
let timelapseIdx = 0;

window.doControl = function(action) {
  const sel = document.getElementById('modelSelect');
  const model = sel.value;

  if (action === 'start' && model === '__timelapse__') {
    // Timelapse mode: fetch model list, then play sequentially
    fetch('/api/viz/timelapse').then(r => r.json()).then(models => {
      if (!models.length) { alert('没有可用的过程快照'); return; }
      timelapseQueue = models;
      timelapseIdx = 0;
      document.getElementById('sModel').textContent = '\u23f3 ' + models[0].name;
      fetch('/api/viz/control', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'start', model: models[0].file }),
      });
    });
    return;
  }

  const body = { action };
  if (action === 'start') {
    timelapseQueue = [];
    body.model = model;
  }
  fetch('/api/viz/control', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
};

// Auto-advance timelapse when episode ends (step reaches 250+)
let lastTimelapseStep = 0;
function checkTimelapseAdvance(step) {
  if (timelapseQueue.length === 0) return;
  if (step < lastTimelapseStep && lastTimelapseStep > 200) {
    // Episode just reset → advance to next model
    timelapseIdx++;
    if (timelapseIdx < timelapseQueue.length) {
      const next = timelapseQueue[timelapseIdx];
      document.getElementById('sModel').textContent = '\u23f3 ' + next.name;
      fetch('/api/viz/control', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'start', model: next.file }),
      });
    } else {
      // All done
      timelapseQueue = [];
      document.getElementById('sModel').textContent = '\u2705 过程回放完成';
    }
  }
  lastTimelapseStep = step;
}

// Load snapshot list
fetch('/api/viz/snapshots')
  .then(r => r.json())
  .then(list => {
    const sel = document.getElementById('modelSelect');
    sel.innerHTML = '';
    list.forEach(name => {
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = name === '__latest__' ? '\u25cf \u5b9e\u65f6 (\u6700\u65b0\u68c0\u67e5\u70b9)'
        : name === '__timelapse__' ? '\u23f3 \u8fc7\u7a0b\u56de\u653e (\u524d\u671f\u2192\u4e2d\u671f\u2192\u6700\u7ec8)'
        : name;
      sel.appendChild(opt);
    });
  });

// ─── Render loop ────────────────────────────────────────────
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// Start SSE
connectSSE();
</script>
</body>
</html>"""


def _list_snapshots(trained_dir):
    """List available model snapshots for visualization."""
    snapshots = ["__latest__"]  # live/latest always first
    snap_dir = os.path.join(trained_dir, "snapshots")
    ckpt_dir = os.path.join(trained_dir, "checkpoints")

    # best.zip second
    if os.path.exists(os.path.join(snap_dir, "best.zip")):
        snapshots.append("best.zip")

    # Timelapse option (auto-picks early / mid / late snapshots)
    snapshots.append("__timelapse__")

    # stage snapshots
    if os.path.isdir(snap_dir):
        for f in sorted(os.listdir(snap_dir)):
            if f.endswith(".zip") and f != "best.zip" and f not in snapshots:
                snapshots.append(f)

    # checkpoints
    if os.path.isdir(ckpt_dir):
        for f in sorted(os.listdir(ckpt_dir)):
            if f.endswith(".zip"):
                snapshots.append(f)

    return snapshots


def _pick_timelapse_models(trained_dir):
    """Pick 3 representative models: early (~0.1M), mid (~1M), final (best).

    Returns list of {"name": display_name, "file": filename} dicts.
    """
    import re
    snap_dir = os.path.join(trained_dir, "snapshots")
    if not os.path.isdir(snap_dir):
        return []

    # Parse step_X.XM_rew_Y.zip files with positive reward
    candidates = []
    for f in os.listdir(snap_dir):
        m = re.match(r"step_([\d.]+)M_rew_(-?\d+)\.zip", f)
        if m:
            step_m = float(m.group(1))
            rew = int(m.group(2))
            if rew > 0:
                candidates.append({"file": f, "step": step_m, "rew": rew})

    if not candidates:
        return []

    candidates.sort(key=lambda x: x["step"])
    max_step = max(c["step"] for c in candidates)

    def pick_near(target_m):
        """Pick the snapshot with highest reward near target step."""
        nearby = [c for c in candidates if abs(c["step"] - target_m) <= max(0.2, target_m * 0.3)]
        if not nearby:
            nearby = sorted(candidates, key=lambda c: abs(c["step"] - target_m))[:3]
        return max(nearby, key=lambda c: c["rew"])

    # Early: ~5% of training, Mid: ~50%, Late: best
    early = pick_near(max_step * 0.05) if max_step > 0.3 else candidates[0]
    mid = pick_near(max_step * 0.5)
    late = max(candidates, key=lambda c: c["rew"])

    # Deduplicate
    result = []
    seen = set()
    for label, c in [("前期", early), ("中期", mid), ("最终", late)]:
        if c["file"] not in seen:
            result.append({
                "name": f"{label} ({c['step']:.1f}M步, 奖励{c['rew']})",
                "file": c["file"],
            })
            seen.add(c["file"])

    return result


class DashboardHandler(BaseHTTPRequestHandler):
    log_file = DEFAULT_LOG
    trained_dir = os.path.join(os.path.dirname(__file__), "..", "trained")

    def do_GET(self):
        if self.path == "/" or self.path == "/dashboard":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode("utf-8"))

        elif self.path == "/api/data":
            metrics = parse_training_log(self.log_file)
            stage = parse_progressive_state()
            posture = parse_posture_eval()
            posture_history = parse_posture_history()
            curricula = get_curriculum_plan()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({
                "metrics": metrics, "stage": stage,
                "posture": posture, "posture_history": posture_history,
                "curricula": curricula,
            }).encode("utf-8"))

        elif self.path == "/viz":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(VISUALIZATION_HTML.encode("utf-8"))

        elif self.path == "/api/viz/snapshots":
            snapshots = _list_snapshots(self.trained_dir)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(snapshots).encode("utf-8"))

        elif self.path == "/api/viz/timelapse":
            models = _pick_timelapse_models(self.trained_dir)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(models).encode("utf-8"))

        elif self.path == "/api/viz/stream":
            self._handle_sse()

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/viz/control":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length > 0 else {}
            action = body.get("action", "")

            global eval_engine
            if eval_engine is None:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b'{"error":"eval engine not initialized"}')
                return

            if action == "start":
                model = body.get("model", "best.zip")
                if eval_engine.running:
                    eval_engine.stop()
                    # Wait briefly for thread to finish
                    time.sleep(0.3)
                eval_engine.start(model)
                resp = {"ok": True, "model": model}
            elif action == "stop":
                eval_engine.stop()
                resp = {"ok": True}
            else:
                resp = {"error": f"unknown action: {action}"}

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def _handle_sse(self):
        """Server-Sent Events stream for visualization frames."""
        global eval_engine
        if eval_engine is None:
            self.send_response(503)
            self.end_headers()
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        q = eval_engine.subscribe()
        try:
            # Send current status immediately
            status_data = json.dumps({
                "state": eval_engine.state,
                "model": eval_engine.current_model or "",
            })
            self.wfile.write(f"event: status\ndata: {status_data}\n\n".encode())
            self.wfile.flush()

            while True:
                try:
                    event_type, data = q.get(timeout=15)
                    self.wfile.write(f"event: {event_type}\ndata: {data}\n\n".encode())
                    self.wfile.flush()
                except queue.Empty:
                    # Heartbeat to keep connection alive
                    self.wfile.write(b": heartbeat\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            eval_engine.unsubscribe(q)

    def log_message(self, format, *args):
        pass  # Suppress request logs


def main():
    global eval_engine

    parser = argparse.ArgumentParser(description="JPRobot Training Dashboard Server")
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    parser.add_argument("--port", type=int, default=int(os.getenv("JPROBOT_DASHBOARD_PORT", "18791")))
    parser.add_argument("--log", type=str, default=DEFAULT_LOG)
    args = parser.parse_args()

    DashboardHandler.log_file = args.log
    trained_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "trained"))
    DashboardHandler.trained_dir = trained_dir

    eval_engine = EvalEngine(trained_dir)

    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer(("0.0.0.0", args.port), DashboardHandler)
    print(f"JPRobot Training Dashboard")
    print(f"  Dashboard: http://127.0.0.1:{args.port}/dashboard")
    print(f"  3D Viz:    http://127.0.0.1:{args.port}/viz")
    print(f"  Log: {args.log}")
    print(f"  Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        eval_engine.stop()
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
