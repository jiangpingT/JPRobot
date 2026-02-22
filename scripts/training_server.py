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
import zipfile
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Default log file (current training)
DEFAULT_LOG = "/tmp/ab_test/jprobot.log"
PROGRESSIVE_STATE = os.path.join(os.path.dirname(__file__), "..", "trained", "progressive_state.json")
POSTURE_EVAL = os.path.join(os.path.dirname(__file__), "..", "trained", "posture_eval.json")
POSTURE_HISTORY = os.path.join(os.path.dirname(__file__), "..", "trained", "posture_eval_history.jsonl")
LIVE_PROGRESS = os.path.join(os.path.dirname(__file__), "..", "trained", "live_progress.json")
FIXED_EVAL = os.path.join(os.path.dirname(__file__), "..", "trained", "fixed_direction_eval.json")


def parse_fixed_eval():
    """Read fixed-direction evaluation results if available."""
    path = os.path.abspath(FIXED_EVAL)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def parse_progressive_state():
    """Read progressive training state if available."""
    path = os.path.abspath(PROGRESSIVE_STATE)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def parse_live_progress():
    """Read live training progress written by LiveProgressCallback (every rollout)."""
    path = os.path.abspath(LIVE_PROGRESS)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


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
        self._last_good_latest = None

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
        old_thread = None
        with self._lock:
            # If already running same model, no-op.
            if self.running and self.current_model == model_name:
                return
            # If running another model, request stop and wait outside lock.
            if self.running and self._thread is not None:
                self.running = False
                old_thread = self._thread

        if old_thread and old_thread.is_alive():
            old_thread.join(timeout=2.0)

        with self._lock:
            self.running = True
            self.current_model = model_name
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()

    def stop(self):
        t = None
        with self._lock:
            self.running = False
            t = self._thread
        if t and t.is_alive():
            t.join(timeout=2.0)

    def _iter_models_by_mtime(self):
        """Yield .zip model paths sorted by mtime descending."""
        candidates = []
        for subdir in ("checkpoints", "snapshots", ""):
            d = os.path.join(self.trained_dir, subdir) if subdir else self.trained_dir
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                if not f.endswith(".zip"):
                    continue
                full = os.path.join(d, f)
                try:
                    mt = os.path.getmtime(full)
                except OSError:
                    continue
                candidates.append((mt, full))
        candidates.sort(key=lambda x: x[0], reverse=True)
        for _, path in candidates:
            yield path

    @staticmethod
    def _is_readable_zip(path):
        """Return True if path is a complete readable zip file."""
        try:
            if not os.path.exists(path) or os.path.getsize(path) <= 0:
                return False
            if not zipfile.is_zipfile(path):
                return False
            with zipfile.ZipFile(path, "r") as zf:
                zf.namelist()
            return True
        except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile):
            return False

    def _find_latest_model(self):
        """Find the most recent *readable* model zip."""
        for path in self._iter_models_by_mtime():
            if self._is_readable_zip(path):
                self._last_good_latest = path
                return path
        return self._last_good_latest

    def _resolve_model_path(self, model_name):
        if model_name == "__latest__":
            # Real-time mode should follow the newest checkpoint/snapshot.
            # Skip files that are still being written and keep last good model.
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
                start_xy = None
                target_dir = getattr(env, "target_dir_xy", None)
                target_name = getattr(env, "target_name", "?")

                self._broadcast("episode_start", {"episode": episode, "direction": target_name})

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
                    if start_xy is None:
                        start_xy = (pos[0], pos[1])
                    # 沿目标方向的真实位移（多方向训练时比单轴 X 更准确）
                    delta = (pos[0] - start_xy[0], pos[1] - start_xy[1])
                    if target_dir is not None:
                        import numpy as _np
                        distance = float(_np.dot(delta, target_dir))
                    else:
                        distance = delta[0]
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
                except Exception as e:
                    self._broadcast("status", {"state": "error", "message": str(e)})
                    break

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

<div id="fixedEvalPanel" style="display:none;background:#1e293b;border-radius:10px;padding:14px 18px;margin-bottom:20px;border-left:3px solid #38bdf8;">
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
      const curPlan = isCurriculum
        ? (d.curricula && d.curricula[stage.curriculum]
            ? d.curricula[stage.curriculum].stages
            : (stage.curriculum_stages || null))
        : null;

      if (isCurriculum && curPlan) {
        const curIdx = stage.curriculum_stage_idx || 0;
        totalStages = curPlan.length;
        stageNum = Math.min(curIdx + 1, totalStages);
        globalTotalSteps = curPlan[curPlan.length - 1].target;
        if (curIdx >= totalStages) {
          // All stages done: lock at 100%, don't double-count
          globalDoneSteps = globalTotalSteps;
        } else {
          globalDoneSteps = curIdx > 0 ? curPlan[curIdx - 1].target : 0;
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
      // For curriculum runs, prefer persisted progressive_state total_steps
      // to avoid stale log timesteps falsely showing 100% done.
      // If total_steps is 0 (written only at stage completion), fall back to
      // live_progress.total_timesteps written every rollout by LiveProgressCallback.
      const lp = d.live_progress;
      const liveSteps = lp && lp.total_timesteps ? lp.total_timesteps : 0;
      const stageLocalSteps = Math.min(total, Math.max(0, stageTargetSteps));
      let globalCurrent = Math.min(globalDoneSteps + stageLocalSteps, globalTotalSteps);
      if (isCurriculum && stage && typeof stage.total_steps === 'number') {
        const persistedSteps = stage.total_steps || 0;
        // Use live_progress when persisted value is 0 (training in progress, not yet written)
        const effectiveSteps = persistedSteps > 0 ? persistedSteps
          : (liveSteps > 0 ? globalDoneSteps + liveSteps : globalCurrent);
        globalCurrent = Math.min(Math.max(0, effectiveSteps), globalTotalSteps);
      }
      const globalPct = globalTotalSteps > 0 ? (globalCurrent / globalTotalSteps * 100).toFixed(0) : 100;
      const globalEta = fps > 0 && globalTotalSteps > globalCurrent
        ? ((globalTotalSteps - globalCurrent) / fps / 3600).toFixed(1) : '?';
      // Current stage progress
      const stageLocalCurrent = isCurriculum && stage && typeof stage.total_steps === 'number'
        ? Math.max(0, (stage.total_steps > 0 ? stage.total_steps : liveSteps > 0 ? liveSteps : total) - globalDoneSteps)
        : total;
      const stagePct = stageTargetSteps > 0
        ? (Math.min(stageLocalCurrent, stageTargetSteps) / stageTargetSteps * 100).toFixed(0)
        : 100;

      // Health check: use last 20% of rewards to detect real decline vs penalty growth
      const tail5 = rewards.slice(-Math.max(1, Math.floor(rewards.length/5)));
      const head5 = rewards.slice(0, Math.max(1, Math.floor(rewards.length/5)));
      const tailAvg = tail5.reduce((a,b) => a+b, 0) / tail5.length;
      const headAvg = head5.reduce((a,b) => a+b, 0) / head5.length;
      // Reward decline is normal after ~1.5M steps due to penalty_factor growth
      // Only warn if decline happens early (before 50% of training)
      const earlyDecline = (globalCurrent / globalTotalSteps < 0.5) && tailAvg < headAvg - 50;
      const healthy = !earlyDecline && curLen > 50;
      const isDone = isCurriculum
        ? ((stage.curriculum_stage_idx || 0) >= totalStages)
        : (globalCurrent >= globalTotalSteps);

      // Prefer curriculum/progressive stage metrics for headline stats.
      // This avoids mixing legacy log history with the current run summary.
      // During training (stage.last_metrics is empty), fall back to live_progress.ep_rew_mean.
      let displayCurReward = curReward;
      let displayBestReward = bestReward;
      let displayLen = curLen;
      if (stage && stage.last_metrics) {
        if (stage.last_metrics.reward_final !== undefined) {
          displayCurReward = stage.last_metrics.reward_final;
        }
        if (stage.last_metrics.reward_best !== undefined) {
          displayBestReward = stage.last_metrics.reward_best;
        }
        if (stage.last_metrics.ep_len_final !== undefined) {
          displayLen = stage.last_metrics.ep_len_final;
        }
      }
      // If stage metrics not yet available (training in progress) and live_progress exists, use it
      if (lp && lp.ep_rew_mean !== null && lp.ep_rew_mean !== undefined) {
        if (!stage || !stage.last_metrics || stage.last_metrics.reward_final === undefined) {
          displayCurReward = lp.ep_rew_mean;
        }
      }

      function cleanText(v) {
        if (v === undefined || v === null) return '';
        const s = String(v).trim();
        if (!s || s === '()' || s === '[]' || s.toLowerCase() === 'none' || s.toLowerCase() === 'null') {
          return '';
        }
        return s;
      }
      const curriculumName = isCurriculum && stage ? cleanText(stage.curriculum || '') : '';
      // Status badge
      const stageTag = totalStages > 1
        ? (stageTypeCn ? stageTypeCn + ' ' + stageNum + '/' + totalStages : stageNum + '/' + totalStages)
        : (stageTypeCn || '');
      document.getElementById('statusBadge').className = 'status ' + (isDone ? 'done' : 'running');
      document.getElementById('statusBadge').textContent = isDone
        ? ('已完成' + (curriculumName ? ' · ' + curriculumName : '') + ' · 奖励 ' + displayBestReward.toFixed(0))
        : ('训练中 ' + (stageTag ? stageTag + ' ' : '') + globalPct + '%');

      // Summary text
      let stageDesc = '';
      if (isCurriculum && totalStages > 1) {
        const currentStageClean = cleanText(currentStageName);
        if (stageTypeCn || currentStageClean) {
          const stageLabel = stageTypeCn || currentStageClean;
          stageDesc = '<b style="color:#c084fc;">' + stageLabel + '</b>，第 ' + stageNum + '/' + totalStages + ' 阶段，';
        } else if (curriculumName) {
          stageDesc = '<b style="color:#c084fc;">课程 ' + curriculumName + '</b>，第 ' + stageNum + '/' + totalStages + ' 阶段，';
        } else {
          stageDesc = '<b style="color:#c084fc;">第 ' + stageNum + '/' + totalStages + ' 阶段</b>，';
        }
      } else if (totalStages > 1) {
        stageDesc = '<b style="color:#c084fc;">第 ' + stageNum + '/' + totalStages + ' 阶段</b>，';
      }
      const etaText = isDone ? '' : '预计剩余 <b>' + globalEta + '</b> 小时。';
      const healthText = isDone
        ? (curLen > 200
          ? ' <b style="color:#4ade80;">训练已结束，可进行验收评估。</b>'
          : ' <b style="color:#facc15;">训练已结束，但稳定性偏弱，建议继续调参或续训。</b>')
        : (healthy
          ? ' <b style="color:#4ade80;">训练健康。</b>'
          : ' <b style="color:#f87171;">早期奖励下降，可能需要检查。</b>');
      document.getElementById('summary').innerHTML =
        '<b style="color:#38bdf8;">训练总结：</b>' + stageDesc +
        '进度 <b style="color:#38bdf8;">' + (globalCurrent/1e6).toFixed(1) + 'M</b> / ' + (globalTotalSteps/1e6).toFixed(0) + 'M 步（' + globalPct + '%）。' +
        etaText +
        '当前奖励 <b style="color:#4ade80;">' + displayCurReward.toFixed(0) + '</b>，' +
        '历史最高 <b style="color:#facc15;">' + displayBestReward.toFixed(0) + '</b>。' +
        '存活 <b style="color:#c084fc;">' + displayLen.toFixed(0) + '</b>/250 步。' +
        healthText;

      document.getElementById('stats').innerHTML =
        '<div class="stat"><div class="label">整体进度</div><div class="value blue">' + (globalCurrent/1e6).toFixed(1) + 'M / ' + (globalTotalSteps/1e6).toFixed(0) + 'M</div></div>' +
        '<div class="stat"><div class="label">当前奖励</div><div class="value green">' + displayCurReward.toFixed(0) + '</div></div>' +
        '<div class="stat"><div class="label">最高奖励</div><div class="value yellow">' + displayBestReward.toFixed(0) + '</div></div>' +
        '<div class="stat"><div class="label">存活步数</div><div class="value purple">' + displayLen.toFixed(0) + '/250</div></div>' +
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
        const doneAllStages = curIdx >= curTotal;
        const activeIdx = doneAllStages ? Math.max(0, curTotal - 1) : curIdx;
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
          if (i === activeIdx) {
            // Active stage (or last stage when all completed)
            status = doneAllStages ? '✓ 已完成' : '▶ 训练中';
            bg = '#38bdf825';
            border = '#38bdf8';
            textColor = '#38bdf8';
          } else if (i < curIdx) {
            // Completed
            const completed = completedStages[i];
            const healthy = completed && completed.healthy;
            status = healthy ? '✓' : '✗';
            bg = '#1e293b';
            border = '#64748b';
            textColor = '#94a3b8';
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

      // Render fixed-direction evaluation panel
      const fepEl = document.getElementById('fixedEvalPanel');
      const fe = d.fixed_eval;
      if (fepEl && fe && fe.results) {
        fepEl.style.display = '';
        const feTime = fe.evaluated_at ? fe.evaluated_at.replace('T', ' ') : '';
        const feEps = fe.episodes_per_direction || 0;
        let feHtml = '<div style="font-size:14px;color:#38bdf8;font-weight:600;margin-bottom:10px;">' +
          '验收评估（确定性推理） ' +
          '<span style="font-size:11px;color:#64748b;font-weight:400;">' + feTime +
          ' · ' + feEps + '局/方向</span></div>';

        const dirs = ['forward', 'backward', 'left', 'right'];
        const dirCN = {forward:'向前', backward:'向后', left:'向左', right:'向右'};
        const weakest = fe.weakest_direction;
        const bestRew = Math.max(...dirs.map(d => (fe.results[d] || {}).mean_episode_reward || 0));

        feHtml += '<div style="display:flex;flex-direction:column;gap:6px;margin-bottom:12px;">';
        // Sort by reward descending for display
        const sortedDirs = dirs.slice().sort(
          (a, b) => (fe.results[b] || {}).mean_episode_reward - (fe.results[a] || {}).mean_episode_reward
        );
        for (const dir of sortedDirs) {
          const r = fe.results[dir];
          if (!r) continue;
          const rew = r.mean_episode_reward || 0;
          const prog = r.median_target_progress || 0;
          const isWeak = dir === weakest;
          const barW = bestRew > 0 ? Math.round(rew / bestRew * 100) : 0;
          const barColor = isWeak ? '#f87171' : '#38bdf8';
          const flagColor = isWeak ? '#f87171' : '#4ade80';
          const flag = isWeak ? '⚠' : '✓';
          feHtml += '<div style="display:flex;align-items:center;gap:8px;">';
          feHtml += '<span style="min-width:36px;font-size:13px;color:#94a3b8;">' + (dirCN[dir] || dir) + '</span>';
          feHtml += '<div style="flex:1;background:#0f172a;border-radius:4px;height:14px;overflow:hidden;">' +
            '<div style="width:' + barW + '%;background:' + barColor + ';height:100%;border-radius:4px;"></div></div>';
          feHtml += '<span style="min-width:44px;font-size:13px;color:#e2e8f0;text-align:right;">' + rew.toFixed(0) + '分</span>';
          feHtml += '<span style="min-width:70px;font-size:11px;color:#64748b;">进度 ' + prog.toFixed(5) + '</span>';
          feHtml += '<span style="font-size:14px;color:' + flagColor + ';">' + flag + '</span>';
          feHtml += '</div>';
        }
        feHtml += '</div>';

        // Posture summary table
        feHtml += '<table style="width:100%;border-collapse:collapse;margin-bottom:12px;font-size:13px;">';
        feHtml += '<tr style="color:#64748b;font-size:11px;">' +
          '<th style="text-align:left;padding:4px 8px;border-bottom:1px solid #334155;">方向</th>' +
          '<th style="text-align:right;padding:4px 8px;border-bottom:1px solid #334155;">得分</th>' +
          '<th style="text-align:right;padding:4px 8px;border-bottom:1px solid #334155;">手臂触地率</th>' +
          '<th style="text-align:center;padding:4px 8px;border-bottom:1px solid #334155;">状态</th>' +
          '</tr>';
        for (const dir of sortedDirs) {
          const r = fe.results[dir];
          if (!r) continue;
          const rew = r.mean_episode_reward || 0;
          const ac = (r.arm_contact_rate || 0) * 100;
          const isWalking = ac < 30;
          const statusText = isWalking ? '走路姿态' : '爬行';
          const statusColor = isWalking ? '#4ade80' : '#facc15';
          const isWeak = dir === weakest;
          const dirColor = isWeak ? '#f87171' : '#e2e8f0';
          feHtml += '<tr style="border-bottom:1px solid #1e293b;">' +
            '<td style="padding:5px 8px;color:' + dirColor + ';">' + (dirCN[dir] || dir) + '</td>' +
            '<td style="padding:5px 8px;text-align:right;color:' + dirColor + ';font-weight:600;">' + rew.toFixed(0) + '</td>' +
            '<td style="padding:5px 8px;text-align:right;color:#94a3b8;">' + ac.toFixed(1) + '%</td>' +
            '<td style="padding:5px 8px;text-align:center;color:' + statusColor + ';">' + statusText + '</td>' +
            '</tr>';
        }
        feHtml += '</table>';

        const weakRew = (fe.results[weakest] || {}).mean_episode_reward || 0;
        const ratio = bestRew > 0 ? (weakRew / bestRew * 100).toFixed(0) : 0;
        feHtml += '<div style="font-size:12px;color:#f87171;margin-bottom:6px;">' +
          '最弱方向：' + (dirCN[weakest] || weakest) + '（' + weakRew.toFixed(0) + '分，仅为最强的 ' + ratio + '%）</div>';

        const rec = fe.recommendation || '';
        feHtml += '<div style="font-size:12px;color:#94a3b8;">建议：' + rec + '</div>';
        if (rec.startsWith('refine_')) {
          feHtml += '<div style="font-size:11px;color:#475569;margin-top:4px;font-family:monospace;">' +
            'python -m jprobot.training.progressive --curriculum ' + rec + ' --auto</div>';
        }

        fepEl.innerHTML = feHtml;
      } else if (fepEl) {
        fepEl.style.display = 'none';
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
  <button id="foxBtn" onclick="toggleFoxMode()" style="background:#0f766e;border-color:#0f766e;margin-top:4px;font-size:12px;">🦊 Fox 模型</button>
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
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

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
// 环境光（暖色调，模拟室内漫反射）
const ambientLight = new THREE.AmbientLight(0xfff0e0, 0.7);
scene.add(ambientLight);

// 主光源（偏暖白，从右上方照射，产生阴影）
const dirLight = new THREE.DirectionalLight(0xfffbe8, 1.4);
dirLight.position.set(0.8, -1.2, 2.5);
dirLight.castShadow = true;
dirLight.shadow.mapSize.set(2048, 2048);
dirLight.shadow.camera.left = -0.5; dirLight.shadow.camera.right = 0.5;
dirLight.shadow.camera.top = 0.5; dirLight.shadow.camera.bottom = -0.5;
dirLight.shadow.bias = -0.001;
scene.add(dirLight);

// 轮廓补光（冷蓝色，从左后方，增强立体感）
const rimLight = new THREE.DirectionalLight(0xaad4ff, 0.5);
rimLight.position.set(-1, 1.5, 1);
scene.add(rimLight);

// 底部反射光（模拟地面反射的暖色）
const fillLight = new THREE.DirectionalLight(0xffd090, 0.3);
fillLight.position.set(0, 0, -1);
scene.add(fillLight);

// ─── Ground plane ───────────────────────────────────────────
const groundGeo = new THREE.PlaneGeometry(6, 2);
const groundMat = new THREE.MeshToonMaterial({ color: 0x1a1a30 });
const ground = new THREE.Mesh(groundGeo, groundMat);
ground.receiveShadow = true;
ground.position.set(1.5, 0, -0.001);
scene.add(ground);

// GridHelper 网格线地板（用 Three.js 内置 GridHelper，旋转到 XY 平面）
const gridHelper = new THREE.GridHelper(6, 60, 0x3a3aff, 0x2a2a55);
gridHelper.rotation.x = Math.PI / 2;  // 旋转到 XY 平面（Z-up）
gridHelper.position.set(1.5, 0, 0.001);
gridHelper.material.transparent = true;
gridHelper.material.opacity = 0.45;
scene.add(gridHelper);

// ─── Robot model ────────────────────────────────────────────
// URDF dimensions (meters) — 身体比例微调：头偏大、腿细一点、身体更椭圆
const TORSO_SIZE = [0.11, 0.075, 0.025];   // 身体略高（更椭圆感）
const UPPER_R = 0.004, UPPER_L = 0.045;    // 腿略粗一点点显得卡通
const LOWER_R = 0.003, LOWER_L = 0.05;
const PAW_R = 0.007, PAW_L = 0.009;        // 爪子略大

// ─── 卡通材质（MeshToonMaterial，橘猫配色）───────────────────
// 橘猫主色：橘黄色身体
const matTorso   = new THREE.MeshToonMaterial({ color: 0xe8721a });
// 白色肚子（胸腹部装饰）
const matBelly    = new THREE.MeshToonMaterial({ color: 0xf5f0e8 });
// 前腿：橘色上肢
const matUpperFront = new THREE.MeshToonMaterial({ color: 0xe8721a });
// 后腿：深橘上肢
const matUpperBack  = new THREE.MeshToonMaterial({ color: 0xc95e10 });
// 下肢：深棕色
const matLower    = new THREE.MeshToonMaterial({ color: 0x8b4513 });
// 爪子：米白色
const matPaw      = new THREE.MeshToonMaterial({ color: 0xf5deb3 });
// 电池/背部装饰：深灰
const matBattery  = new THREE.MeshToonMaterial({ color: 0x2a2a2a });
// 耳朵：橘色外耳
const matEarOuter = new THREE.MeshToonMaterial({ color: 0xe8721a });
// 耳朵内：粉色内耳
const matEarInner = new THREE.MeshToonMaterial({ color: 0xff9eb5 });
// 尾巴：橘棕渐变用橘色
const matTail     = new THREE.MeshToonMaterial({ color: 0xc95e10 });

const robotGroup = new THREE.Group();
scene.add(robotGroup);

// ── 身体（椭球感：用 SphereGeometry 拉伸 + BoxGeometry 叠加）──
// 主身体用 BoxGeometry，加圆角感用 scale
const torsoMesh = new THREE.Mesh(
  new THREE.BoxGeometry(...TORSO_SIZE),
  matTorso
);
torsoMesh.castShadow = true;
robotGroup.add(torsoMesh);

// 身体侧面加圆润感（左右两个小半球）
const sideBallGeo = new THREE.SphereGeometry(0.013, 8, 6);
[-1, 1].forEach(sign => {
  const sideBall = new THREE.Mesh(sideBallGeo, matTorso);
  sideBall.position.set(0, sign * 0.037, 0);
  sideBall.scale.set(1.4, 0.7, 0.8);
  sideBall.castShadow = true;
  robotGroup.add(sideBall);
});

// 白色肚皮（正面小平面）
const bellyMesh = new THREE.Mesh(
  new THREE.BoxGeometry(0.07, 0.05, 0.003),
  matBelly
);
bellyMesh.position.set(0, 0, 0.013);
robotGroup.add(bellyMesh);

// ── 头部（大头，更像猫）──
const headGroup = new THREE.Group();
headGroup.position.set(0.062, 0, 0.012);
robotGroup.add(headGroup);

// 头部主体（球形，偏大）
const headMesh = new THREE.Mesh(
  new THREE.SphereGeometry(0.028, 12, 10),
  matTorso
);
headMesh.scale.set(1.1, 0.95, 1.0);
headMesh.castShadow = true;
headGroup.add(headMesh);

// 脸部白色区域
const faceMesh = new THREE.Mesh(
  new THREE.SphereGeometry(0.019, 10, 8),
  matBelly
);
faceMesh.position.set(0.016, 0, -0.002);
faceMesh.scale.set(0.6, 0.85, 0.9);
headGroup.add(faceMesh);

// 耳朵（两个锥形，左右各一）
const earGeo = new THREE.ConeGeometry(0.010, 0.022, 4);  // 四棱锥，像猫耳
const earInnerGeo = new THREE.ConeGeometry(0.006, 0.015, 4);
[1, -1].forEach(sign => {
  // 外耳
  const ear = new THREE.Mesh(earGeo, matEarOuter);
  ear.position.set(0.005, sign * 0.020, 0.022);
  ear.rotation.x = sign * 0.25;  // 略微外八
  ear.castShadow = true;
  headGroup.add(ear);
  // 粉色内耳
  const earInner = new THREE.Mesh(earInnerGeo, matEarInner);
  earInner.position.set(0.007, sign * 0.020, 0.022);
  earInner.rotation.x = sign * 0.25;
  headGroup.add(earInner);
});

// ── 尾巴（弯曲感：用多段 CylinderGeometry 拼接）──
const tailGroup = new THREE.Group();
tailGroup.position.set(-0.058, 0, 0.005);
robotGroup.add(tailGroup);

// 尾巴根部
const tail1 = new THREE.Mesh(
  new THREE.CylinderGeometry(0.006, 0.008, 0.03, 8),
  matTail
);
tail1.rotation.x = -Math.PI * 0.35;  // 向上翘
tail1.position.set(-0.005, 0, 0.012);
tail1.castShadow = true;
tailGroup.add(tail1);

// 尾巴中段（弯折）
const tail2 = new THREE.Mesh(
  new THREE.CylinderGeometry(0.004, 0.006, 0.028, 8),
  matTail
);
tail2.position.set(-0.008, 0, 0.033);
tail2.rotation.x = -Math.PI * 0.15;
tail2.castShadow = true;
tailGroup.add(tail2);

// 尾巴尖（细）
const tail3 = new THREE.Mesh(
  new THREE.CylinderGeometry(0.002, 0.004, 0.018, 8),
  matBelly  // 尾巴尖白色
);
tail3.position.set(-0.010, 0, 0.055);
tail3.rotation.x = Math.PI * 0.1;
tailGroup.add(tail3);

// Battery / 背部装饰
const batteryMesh = new THREE.Mesh(new THREE.BoxGeometry(0.085, 0.038, 0.012), matBattery);
batteryMesh.position.set(0, 0, -0.016);
robotGroup.add(batteryMesh);

// Leg origins from URDF (xyz relative to torso center)
const legConfig = [
  { name: 'FL', ox:  0.055, oy:  0.05, elbowOy: -0.015, sign: 1,  isFront: true },
  { name: 'FR', ox:  0.055, oy: -0.05, elbowOy:  0.015, sign: -1, isFront: true },
  { name: 'BR', ox: -0.055, oy: -0.05, elbowOy:  0.015, sign: -1, isFront: false },
  { name: 'BL', ox: -0.055, oy:  0.05, elbowOy: -0.015, sign: 1,  isFront: false },
];

// Build leg hierarchy: shoulderPivot → upperLimb → elbowStatic → elbowPivot → lowerLimb → paw
const legs = {};
legConfig.forEach(cfg => {
  // 前腿用橘色，后腿用深橘色
  const matUp = cfg.isFront ? matUpperFront : matUpperBack;

  // Shoulder pivot (at URDF joint origin, rotates around Y)
  const shoulderPivot = new THREE.Group();
  shoulderPivot.position.set(cfg.ox, cfg.oy, 0);
  robotGroup.add(shoulderPivot);

  // 肩部关节小球（卡通风格）
  const shoulderBall = new THREE.Mesh(
    new THREE.SphereGeometry(0.006, 8, 6),
    matUp
  );
  shoulderPivot.add(shoulderBall);

  // Upper limb (cylinder along Z, offset so top is at pivot)
  // 用较大半径 + 较多分段，视觉上更圆润
  const upperGeo = new THREE.CylinderGeometry(UPPER_R, UPPER_R * 0.8, UPPER_L, 10);
  const upperMesh = new THREE.Mesh(upperGeo, matUp);
  upperMesh.rotation.x = Math.PI / 2;
  upperMesh.position.set(0, 0, -UPPER_L / 2);
  upperMesh.castShadow = true;
  shoulderPivot.add(upperMesh);

  // Elbow static transform (URDF: rpy(0, -pi/2, 0) at elbow origin)
  const elbowStatic = new THREE.Group();
  elbowStatic.position.set(0, cfg.elbowOy, -UPPER_L);
  elbowStatic.rotation.y = -Math.PI / 2;
  shoulderPivot.add(elbowStatic);

  // Elbow pivot (dynamic rotation around Y axis = joint angle)
  const elbowPivot = new THREE.Group();
  elbowStatic.add(elbowPivot);

  // 肘部关节小球
  const elbowBall = new THREE.Mesh(
    new THREE.SphereGeometry(0.005, 8, 6),
    matLower
  );
  elbowPivot.add(elbowBall);

  // Lower limb（略细，锥形感，上粗下细）
  const lowerGeo = new THREE.CylinderGeometry(LOWER_R * 0.7, LOWER_R, LOWER_L, 10);
  const lowerMesh = new THREE.Mesh(lowerGeo, matLower);
  lowerMesh.rotation.x = Math.PI / 2;
  lowerMesh.position.set(0, 0, -LOWER_L / 2);
  lowerMesh.castShadow = true;
  elbowPivot.add(lowerMesh);

  // Paw（爪子：扁球形，更可爱）
  const pawMesh = new THREE.Mesh(
    new THREE.SphereGeometry(PAW_R, 8, 6),
    matPaw
  );
  pawMesh.scale.set(1.2, 1.0, 0.7);
  pawMesh.position.set(0, 0, -LOWER_L - PAW_R * 0.7);
  pawMesh.castShadow = true;
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

// ─── Fox GLB Model (Route B) ────────────────────────────────
// Bone mapping: BittleX joint index → Fox skeleton bone name + rotation axis
// Left side joints use axis 'x' scale +1; Right side use scale -1 (mirrored)
const FOX_JOINT_MAP = [
  { bone: 'b_LeftUpperArm_09',  axis: 'x', scale:  1.0 },  // 0: FL shoulder
  { bone: 'b_LeftForeArm_010',  axis: 'x', scale:  1.0 },  // 1: FL elbow
  { bone: 'b_RightUpperArm_06', axis: 'x', scale: -1.0 },  // 2: FR shoulder
  { bone: 'b_RightForeArm_07',  axis: 'x', scale: -1.0 },  // 3: FR elbow
  { bone: 'b_RightLeg01_019',   axis: 'x', scale: -1.0 },  // 4: BR hip
  { bone: 'b_RightLeg02_020',   axis: 'x', scale: -1.0 },  // 5: BR knee
  { bone: 'b_LeftLeg01_015',    axis: 'x', scale:  1.0 },  // 6: BL hip
  { bone: 'b_LeftLeg02_016',    axis: 'x', scale:  1.0 },  // 7: BL knee
];

let foxMode = false;
let foxBones = {};
let foxRestAngles = {};

// foxWrapper: receives PyBullet position/quaternion (same coordinate system as robotGroup)
const foxWrapper = new THREE.Group();
foxWrapper.visible = false;
scene.add(foxWrapper);

const foxLoader = new GLTFLoader();
foxLoader.load('/assets/Fox.glb', (gltf) => {
  const foxModel = gltf.scene;

  // Fox.glb is glTF Y-up; our scene is Z-up (PyBullet).
  // Rotate to align: X=-90° flips Y-up→Z-up; Z=180° makes fox face X-forward.
  foxModel.rotation.x = -Math.PI / 2;
  foxModel.rotation.z = Math.PI;

  // Scale: Fox model is ~200 glTF units tall.
  // BittleX torso = 0.11m → fox scaled to similar size (~0.12m body length).
  foxModel.scale.set(0.00065, 0.00065, 0.00065);

  foxWrapper.add(foxModel);

  // Collect all Bone nodes from the skeleton
  gltf.scene.traverse(obj => {
    if (obj.isBone) foxBones[obj.name] = obj;
  });
  console.log('Fox GLB loaded. Bones found:', Object.keys(foxBones));

  // Save rest pose rotations so we apply joint angles as deltas
  FOX_JOINT_MAP.forEach(jm => {
    const b = foxBones[jm.bone];
    if (b) {
      foxRestAngles[jm.bone] = { x: b.rotation.x, y: b.rotation.y, z: b.rotation.z };
    } else {
      console.warn('Fox bone not found in skeleton:', jm.bone);
    }
  });
}, undefined, (err) => {
  console.error('Failed to load Fox.glb:', err);
});

window.toggleFoxMode = function() {
  foxMode = !foxMode;
  robotGroup.visible = !foxMode;
  foxWrapper.visible = foxMode;
  const btn = document.getElementById('foxBtn');
  btn.textContent = foxMode ? '🐱 卡通猫' : '🦊 Fox 模型';
  btn.style.background = foxMode ? '#e8721a' : '#0f766e';
  btn.style.borderColor = foxMode ? '#c95e10' : '#0f766e';
};

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
      if (d.state === 'error') {
        const msg = d.message || '\u53ef\u89c6\u5316\u5f15\u64ce\u542f\u52a8\u5931\u8d25';
        cs.textContent = '\u9519\u8bef: ' + msg;
        cs.className = 'error';
      }
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

    // Fox wrapper follows same position/orientation as robot
    foxWrapper.position.set(px, py, pz);
    foxWrapper.quaternion.set(qx, qy, qz, qw);

    // Update joint angles (toon cat)
    d.joints.forEach((angle, i) => {
      if (jointMap[i]) {
        jointMap[i].pivot.rotation[jointMap[i].prop] = angle;
      }
    });

    // Update Fox skeleton bones (Route B)
    if (foxMode && Object.keys(foxBones).length > 0) {
      d.joints.forEach((angle, i) => {
        const jm = FOX_JOINT_MAP[i];
        if (jm && foxBones[jm.bone]) {
          const rest = foxRestAngles[jm.bone] || { x: 0, y: 0, z: 0 };
          foxBones[jm.bone].rotation[jm.axis] = rest[jm.axis] + angle * jm.scale;
        }
      });
    }

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
        : name === '__timelapse__' ? '\u23f3 \u8fc7\u7a0b\u56de\u653e (\u524d\u671f\u00d73\u2192\u4e2d\u671f\u2192\u6700\u7ec8)'
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

    def pick_first_at(target_m):
        """Pick the first snapshot at or after target_m steps."""
        after = [c for c in candidates if c["step"] >= target_m]
        if after:
            return after[0]  # candidates already sorted by step, take earliest
        # fallback: closest one
        return min(candidates, key=lambda c: abs(c["step"] - target_m))

    # 前期: first snapshot at ~0.1M, ~0.2M, ~0.3M
    early_batch = [pick_first_at(0.1), pick_first_at(0.2), pick_first_at(0.3)]
    # 中期: first snapshot with reward in 500-600 range; fallback to ~60%
    mid_candidates = [c for c in candidates if 500 <= c["rew"] <= 600]
    mid = mid_candidates[0] if mid_candidates else pick_near(max_step * 0.6)
    # 最终: best reward snapshot
    late = max(candidates, key=lambda c: c["rew"])

    # Deduplicate
    result = []
    seen = set()

    # Add 前期 1/2/3
    for idx, c in enumerate(early_batch):
        if c["file"] not in seen:
            result.append({
                "name": f"前期{idx + 1}/3 ({c['step']:.1f}M步, 奖励{c['rew']})",
                "file": c["file"],
            })
            seen.add(c["file"])

    # Add 中期 and 最终
    for label, c in [("中期", mid), ("最终", late)]:
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
            live_progress = parse_live_progress()
            fixed_eval = parse_fixed_eval()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({
                "metrics": metrics, "stage": stage,
                "posture": posture, "posture_history": posture_history,
                "curricula": curricula, "live_progress": live_progress,
                "fixed_eval": fixed_eval,
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

        elif self.path == "/assets/Fox.glb":
            glb_path = "/Users/mlamp/Workspace/cat_model/Fox.glb"
            if os.path.exists(glb_path):
                with open(glb_path, "rb") as f:
                    data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "model/gltf-binary")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_response(404)
                self.end_headers()

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
    pid = os.getpid()
    cmd_hint = f"conda run -n jprobot python scripts/training_server.py --port {args.port}"
    print(f"JPRobot Training Dashboard")
    print(f"  PID: {pid}")
    print(f"  Port: {args.port}")
    print(f"  Trained dir: {trained_dir}")
    print(f"  Dashboard: http://127.0.0.1:{args.port}/dashboard")
    print(f"  3D Viz:    http://127.0.0.1:{args.port}/viz")
    print(f"  Log: {args.log}")
    print(f"  Recommended foreground start: {cmd_hint}")
    print(f"  Note: If 127.0.0.1:{args.port} is not LISTEN, dashboard/viz will refuse connection.")
    print(f"  Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"\n[FATAL] training_server crashed: {type(e).__name__}: {e}")
        raise
    finally:
        try:
            eval_engine.stop()
        except Exception:
            pass
        server.server_close()


if __name__ == "__main__":
    main()
