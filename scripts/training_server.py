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

DASHBOARD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "trained"))
MOE_EVAL = os.path.join(os.path.dirname(__file__), "..", "trained", "moe_eval.json")


def find_live_dashboard() -> dict | None:
    """扫描 DASHBOARD_DIR 下所有 live_dashboard.json，返回 mtime 最新的那个。"""
    best_mtime = 0.0
    best_spec   = None
    best_dir    = None
    for root, _dirs, files in os.walk(DASHBOARD_DIR):
        if "live_dashboard.json" in files:
            p = os.path.join(root, "live_dashboard.json")
            try:
                mtime = os.path.getmtime(p)
                if mtime > best_mtime:
                    with open(p, encoding="utf-8") as f:
                        spec = json.load(f)
                    best_mtime = mtime
                    best_spec  = spec
                    best_dir   = root
            except (OSError, json.JSONDecodeError):
                pass
    if best_spec is not None:
        # 把 dir 附到 spec，供 parse_history 定位 history_file
        best_spec["_dir"] = best_dir
    return best_spec


def parse_history(spec: dict) -> list:
    """读 spec['history_file'] 对应的 JSONL，返回 list of dict。"""
    history_file = spec.get("history_file")
    base_dir     = spec.get("_dir")
    if not history_file or not base_dir:
        return []
    path = os.path.join(base_dir, history_file)
    if not os.path.exists(path):
        return []
    points = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        points.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except OSError:
        pass
    return points


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
            return self._find_latest_model()
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "trained"))
        # humanoid 系列：_run_loop_humanoid 内部用 self.current_model 决定路径
        if model_name == "__route_a__":
            return os.path.join(base, "route_a_v3", "snapshots", "best.zip")
        if model_name == "__route_b__":
            return os.path.join(base, "route_b_v4", "snapshots", "best.zip")
        if model_name == "__humanoid_sac__":
            return os.path.join(base, "humanoid_sac", "best.zip")
        if model_name == "__humanoid_velocity__":
            return os.path.join(base, "humanoid_velocity", "best.zip")
        if model_name == "__humanoid_upright_c2__":
            return os.path.join(base, "humanoid_upright", "C2", "best.zip")
        sac_ckpt_m = re.match(r'^__humanoid_sac_ckpt_(\d+)__$', model_name)
        if sac_ckpt_m:
            steps = sac_ckpt_m.group(1)
            return os.path.join(base, "humanoid_sac", "checkpoints", f"humanoid_sac_{steps}_steps.zip")
        if model_name == "__backflip_latest__":
            versions = _list_backflip_versions(base)
            if versions:
                _, bf_dir = versions[0]
                return os.path.join(base, bf_dir, "full", "best.zip")
            return None
        # Backflip 各训练阶段
        for bf_ver, bf_dir in _list_backflip_versions(base):
            for bf_stage in ("jump", "rotate", "land", "full"):
                if model_name == f"__backflip_{bf_ver}_{bf_stage}__":
                    return os.path.join(base, bf_dir, bf_stage, "best.zip")
        # Try snapshots/ first, then checkpoints/, then direct path
        for subdir in ("snapshots", "checkpoints", ""):
            path = os.path.join(self.trained_dir, subdir, model_name) if subdir else os.path.join(self.trained_dir, model_name)
            if os.path.exists(path):
                return path
        return None

    def _run_loop(self):
        if self.current_model == "__moe__":
            self._run_loop_moe()
            return

        if self.current_model == "__humanoid_sac__":
            self._run_loop_humanoid_sac()
            return

        if self.current_model == "__humanoid_velocity__":
            self._run_loop_humanoid_velocity()
            return

        if self.current_model == "__humanoid_upright_c2__":
            c2_path = self._resolve_model_path("__humanoid_upright_c2__")
            self._run_loop_humanoid_sac(model_path=c2_path)
            return

        sac_ckpt_m = re.match(r'^__humanoid_sac_ckpt_(\d+)__$', self.current_model)
        if sac_ckpt_m:
            ckpt_path = self._resolve_model_path(self.current_model)
            self._run_loop_humanoid_sac(model_path=ckpt_path)
            return

        import pybullet as p
        from stable_baselines3 import PPO

        # Route A v3 → BittleGymEnvV2（direction-reward，obs=254）
        if self.current_model == "__route_a__":
            from jprobot.training.env_v2 import BittleGymEnvV2 as _EnvCls
            _env_kwargs = {"render_mode": None}
        # Route B v4 → BittleGymEnvVelocityV2（velocity-tracking，obs=254）
        elif self.current_model == "__route_b__":
            from jprobot.training.env_velocity_v2 import BittleGymEnvVelocityV2 as _EnvCls
            _env_kwargs = {"render_mode": None}
        # Backflip 各阶段 → BittleBackflipEnv（obs=23，training_phase="full" 全奖励可视化）
        elif self.current_model.startswith("__backflip"):
            from jprobot.training.env_backflip import BittleBackflipEnv as _EnvCls
            _env_kwargs = {"render_mode": None, "training_phase": "full"}
        else:
            from jprobot.training.env import BittleGymEnv as _EnvCls
            _env_kwargs = {"render_mode": None}

        model_path = self._resolve_model_path(self.current_model)
        if not model_path:
            self._broadcast("status", {"state": "error", "message": f"Model not found: {self.current_model}"})
            self.running = False
            self.state = "idle"
            return

        self.state = "loading"
        self._broadcast("status", {"state": "loading", "model": self.current_model})

        try:
            env = _EnvCls(**_env_kwargs)
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

    def _run_loop_moe(self):
        """MoE 可视化：向前用 Route A，其余用 Route B（obs 自动转换）。

        路径从 moe_eval.json 读取（model_a / model_b 绝对路径）。
        env: BittleGymEnvV2（与 Route A 训练环境完全一致，obs=254）。
        obs 转换：Route A env obs[248:250](target_dir) → vel_cmd_scaled(2)，
                  obs[250:254](feet_contact) 直接复用（两个环境相同）。
        """
        import numpy as _np
        import pybullet as p
        from stable_baselines3 import PPO
        from jprobot.training.env_v2 import BittleGymEnvV2

        # 读取 moe_eval.json 获取两个模型的绝对路径
        moe_path = os.path.abspath(MOE_EVAL)
        if not os.path.exists(moe_path):
            self._broadcast("status", {"state": "error", "message": f"moe_eval.json 不存在: {moe_path}"})
            self.running = False
            self.state = "idle"
            return

        with open(moe_path) as f:
            moe_data = json.load(f)
        model_a_path = moe_data["model_a"]
        model_b_path = moe_data["model_b"]

        _VEL_CMD_MAP = {
            "forward":  _np.array([ 0.25,  0.0], dtype=_np.float32),
            "backward": _np.array([-0.25,  0.0], dtype=_np.float32),
            "left":     _np.array([ 0.0,   0.25], dtype=_np.float32),
            "right":    _np.array([ 0.0,  -0.25], dtype=_np.float32),
        }
        _VEL_SCALE = 2.5

        self.state = "loading"
        self._broadcast("status", {"state": "loading", "model": "__moe__"})
        try:
            model_a = PPO.load(model_a_path)
            model_b = PPO.load(model_b_path)
            b_obs_dim = model_b.observation_space.shape[0]
            env = BittleGymEnvV2(render_mode=None)
        except Exception as e:
            self._broadcast("status", {"state": "error", "message": str(e)})
            self.running = False
            self.state = "idle"
            return

        self.state = "running"
        self._broadcast("status", {
            "state": "running", "model": "__moe__",
            "model_a": os.path.basename(os.path.dirname(model_a_path)) + "/best.zip",
            "model_b": os.path.basename(os.path.dirname(model_b_path)) + "/best.zip",
        })

        episode = 0
        try:
            while self.running:
                episode += 1
                obs, _ = env.reset()
                total_reward = 0.0
                max_distance = 0.0
                start_xy = None

                target_name = getattr(env, "target_name", "forward")
                target_dir  = getattr(env, "target_dir_xy", _np.array([1.0, 0.0]))
                use_model   = "A" if target_name == "forward" else "B"
                vel_cmd     = _VEL_CMD_MAP.get(target_name, _np.array([0.25, 0.0]))

                self._broadcast("episode_start", {
                    "episode": episode,
                    "direction": target_name,
                    "model_used": use_model,
                })

                step = 0
                while self.running:
                    if use_model == "A":
                        action, _ = model_a.predict(obs, deterministic=True)
                    else:
                        vel_scaled = _np.clip(vel_cmd * _VEL_SCALE, -1.0, 1.0)
                        if b_obs_dim == 254:
                            obs_b = _np.concatenate([obs[:248], vel_scaled, obs[250:254]]).astype(_np.float32)
                        else:
                            obs_b = _np.concatenate([obs[:248], vel_scaled]).astype(_np.float32)
                        action, _ = model_b.predict(obs_b, deterministic=True)

                    obs, reward, terminated, truncated, info = env.step(action)
                    step += 1
                    total_reward += reward

                    pos, orn = p.getBasePositionAndOrientation(
                        env.robot_id, physicsClientId=env.physics_client
                    )
                    if start_xy is None:
                        start_xy = (pos[0], pos[1])
                    delta = (pos[0] - start_xy[0], pos[1] - start_xy[1])
                    distance = float(_np.dot(delta, target_dir))
                    max_distance = max(max_distance, distance)

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
                        "model_used": use_model,
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
                    "direction": target_name,
                    "model_used": use_model,
                })
        finally:
            try:
                env.close()
            except Exception:
                pass
            self.state = "idle"
            self.running = False
            self._broadcast("status", {"state": "stopped"})


    def _run_loop_humanoid(self):
        """人形机器人可视化：使用 MuJoCo 引擎推理。

        与四足机器人的区别：
        - 使用 mujoco.MjData 读取状态，而非 pybullet.getBasePositionAndOrientation
        - 位姿四元数从 MuJoCo 格式 [w,x,y,z] 转成 Three.js 格式 [x,y,z,w]
        - 帧数据中携带 robot_type='humanoid' 让前端切换渲染模式
        """
        # macOS MuJoCo + OpenMP 双库冲突修复（与 train_humanoid.py 保持一致）
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        import numpy as _np
        from stable_baselines3 import PPO
        from jprobot.training.env_humanoid import HumanoidEnv

        base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "trained"))
        # 根据 current_model 选择对应版本的路径和环境类
        if self.current_model == "__humanoid_v4__":
            model_path = os.path.join(base, "humanoid_v4", "best.zip")
            from jprobot.training.env_humanoid_v4 import HumanoidEnvV4 as _HEnv
        elif self.current_model == "__humanoid_v3__":
            model_path = os.path.join(base, "humanoid_v3", "best.zip")
            from jprobot.training.env_humanoid_v3 import HumanoidEnvV3 as _HEnv
        elif self.current_model == "__humanoid_v2__":
            model_path = os.path.join(base, "humanoid_v2", "best.zip")
            from jprobot.training.env_humanoid_v2 import HumanoidEnvV2 as _HEnv
        else:
            model_path = os.path.join(base, "humanoid_v1", "best.zip")
            _HEnv = HumanoidEnv

        if not self._is_readable_zip(model_path):
            self._broadcast("status", {"state": "error", "message": f"Model not found: {model_path}"})
            self.running = False
            self.state = "idle"
            return

        self.state = "loading"
        self._broadcast("status", {"state": "loading", "model": self.current_model})

        try:
            env = _HEnv(render_mode=None)
            model = PPO.load(model_path)
        except Exception as e:
            self._broadcast("status", {"state": "error", "message": str(e)})
            self.running = False
            self.state = "idle"
            return

        self.state = "running"
        self._broadcast("status", {"state": "running", "model": "__humanoid_v1__"})

        last_mtime = os.path.getmtime(model_path)
        episode = 0

        try:
            while self.running:
                episode += 1
                obs, _ = env.reset()
                total_reward = 0.0
                start_x = None
                max_distance = 0.0

                self._broadcast("episode_start", {"episode": episode, "direction": "forward"})

                step = 0
                while self.running:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    step += 1
                    total_reward += reward

                    # MuJoCo qpos: [x, y, z, w, qx, qy, qz, joint0..joint16]
                    pos = env.data.qpos[:3].tolist()
                    # MuJoCo 四元数格式 [w,x,y,z] → Three.js 格式 [x,y,z,w]
                    orn_mj = env.data.qpos[3:7]
                    orn_xyzw = [float(orn_mj[1]), float(orn_mj[2]),
                                float(orn_mj[3]), float(orn_mj[0])]
                    joints = [round(float(v), 4) for v in env.data.qpos[7:]]  # 17 关节角

                    if start_x is None:
                        start_x = pos[0]
                    distance = pos[0] - start_x
                    max_distance = max(max_distance, distance)

                    self._broadcast("frame", {
                        "step": step,
                        "position": [round(v, 5) for v in pos],
                        "orientation": [round(v, 5) for v in orn_xyzw],
                        "joints": joints,
                        "reward": round(float(reward), 2),
                        "robot_type": "humanoid",  # 告知前端用人形渲染
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

                # 检查模型文件是否更新
                try:
                    cur_mtime = os.path.getmtime(model_path)
                    if cur_mtime != last_mtime:
                        last_mtime = cur_mtime
                        model = PPO.load(model_path)
                        self._broadcast("model_updated", {
                            "file": "humanoid_v1/best.zip",
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

    def _run_loop_humanoid_sac(self, model_path=None):
        """SAC 人形机器人可视化：Gymnasium Humanoid-v4 原生环境 + SAC 策略。

        model_path=None → 使用 humanoid_sac/best.zip（训练最佳，支持热加载）
        model_path=<path> → 使用指定 checkpoint（固定模型，不热加载）
        """
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        import gymnasium as _gym
        from stable_baselines3 import SAC

        base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "trained"))
        is_live = model_path is None  # live=True 时每局结束后检查 best.zip 是否更新
        if model_path is None:
            model_path = os.path.join(base, "humanoid_sac", "best.zip")

        if not self._is_readable_zip(model_path):
            self._broadcast("status", {"state": "error", "message": f"Model not found: {model_path}"})
            self.running = False
            self.state = "idle"
            return

        self.state = "loading"
        self._broadcast("status", {"state": "loading", "model": "__humanoid_sac__"})

        try:
            env = _gym.make("Humanoid-v4")
            model = SAC.load(model_path)
        except Exception as e:
            self._broadcast("status", {"state": "error", "message": str(e)})
            self.running = False
            self.state = "idle"
            return

        self.state = "running"
        self._broadcast("status", {"state": "running", "model": "__humanoid_sac__"})

        last_mtime = os.path.getmtime(model_path)
        episode = 0

        try:
            while self.running:
                episode += 1
                obs, _ = env.reset()
                total_reward = 0.0
                start_x = None
                max_distance = 0.0

                self._broadcast("episode_start", {"episode": episode, "direction": "forward"})

                step = 0
                while self.running:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    step += 1
                    total_reward += reward

                    # Gymnasium Humanoid-v4 qpos: [x, y, z, w, qx, qy, qz, joint0..joint16]
                    pos = env.unwrapped.data.qpos[:3].tolist()
                    orn_mj = env.unwrapped.data.qpos[3:7]
                    orn_xyzw = [float(orn_mj[1]), float(orn_mj[2]),
                                float(orn_mj[3]), float(orn_mj[0])]
                    joints = [round(float(v), 4) for v in env.unwrapped.data.qpos[7:]]  # 17 关节角

                    if start_x is None:
                        start_x = pos[0]
                    distance = pos[0] - start_x
                    max_distance = max(max_distance, distance)

                    self._broadcast("frame", {
                        "step": step,
                        "position": [round(v, 5) for v in pos],
                        "orientation": [round(v, 5) for v in orn_xyzw],
                        "joints": joints,
                        "reward": round(float(reward), 2),
                        "robot_type": "humanoid",
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

                # 热加载：仅对 best.zip（live 模式）有效，checkpoint 是固定模型
                if is_live:
                    try:
                        cur_mtime = os.path.getmtime(model_path)
                        if cur_mtime != last_mtime:
                            last_mtime = cur_mtime
                            model = SAC.load(model_path)
                            self._broadcast("model_updated", {
                                "file": "humanoid_sac/best.zip",
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

    def _run_loop_humanoid_velocity(self):
        """速度命令跟随人形机器人可视化：HumanoidVelocityEnv + SAC 策略。

        与 _run_loop_humanoid_sac 的区别：
          - 使用 HumanoidVelocityEnv（379维obs = 376 + 3速度命令）
          - 每局广播 vel_cmd 和 vel_actual 供前端 HUD 显示
          - frame 数据多了 vel_cmd/vel_actual/vel_error 字段
        """
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        from stable_baselines3 import SAC
        import sys as _sys
        _sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from jprobot.training.env_humanoid_velocity import HumanoidVelocityEnv

        base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "trained"))
        model_path = os.path.join(base, "humanoid_velocity", "best.zip")

        if not self._is_readable_zip(model_path):
            self._broadcast("status", {"state": "error", "message": f"Model not found: {model_path}"})
            self.running = False
            self.state = "idle"
            return

        self.state = "loading"
        self._broadcast("status", {"state": "loading", "model": "__humanoid_velocity__"})

        try:
            env = HumanoidVelocityEnv()
            model = SAC.load(model_path)
        except Exception as e:
            self._broadcast("status", {"state": "error", "message": str(e)})
            self.running = False
            self.state = "idle"
            return

        self.state = "running"
        self._broadcast("status", {"state": "running", "model": "__humanoid_velocity__"})

        last_mtime = os.path.getmtime(model_path)
        episode = 0

        try:
            while self.running:
                episode += 1
                obs, info = env.reset()
                cmd_vx = env.cmd_vx
                cmd_vy = env.cmd_vy
                cmd_wz = env.cmd_wz
                total_reward = 0.0
                start_x = None
                max_distance = 0.0

                self._broadcast("episode_start", {
                    "episode": episode,
                    "direction": f"cmd({cmd_vx:+.2f},{cmd_vy:+.2f},{cmd_wz:+.2f})",
                    "vel_cmd": [round(cmd_vx, 3), round(cmd_vy, 3), round(cmd_wz, 3)],
                })

                step = 0
                while self.running:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    step += 1
                    total_reward += float(reward)

                    pos = env.unwrapped.data.qpos[:3].tolist()
                    orn_mj = env.unwrapped.data.qpos[3:7]
                    orn_xyzw = [float(orn_mj[1]), float(orn_mj[2]),
                                float(orn_mj[3]), float(orn_mj[0])]
                    joints = [round(float(v), 4) for v in env.unwrapped.data.qpos[7:]]

                    if start_x is None:
                        start_x = pos[0]
                    distance = pos[0] - start_x
                    max_distance = max(max_distance, distance)

                    vel_actual = info.get("vel_actual", (0.0, 0.0, 0.0))
                    vel_error  = float(info.get("vel_error", 0.0))

                    self._broadcast("frame", {
                        "step": step,
                        "position": [round(v, 5) for v in pos],
                        "orientation": [round(v, 5) for v in orn_xyzw],
                        "joints": joints,
                        "reward": round(float(reward), 2),
                        "robot_type": "humanoid",
                        "vel_cmd":    [round(cmd_vx, 3), round(cmd_vy, 3), round(cmd_wz, 3)],
                        "vel_actual": [round(float(vel_actual[0]), 3),
                                       round(float(vel_actual[1]), 3),
                                       round(float(vel_actual[2]), 3)],
                        "vel_error":  round(vel_error, 4),
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

                # 热加载：每局结束后检查 best.zip 是否更新
                try:
                    cur_mtime = os.path.getmtime(model_path)
                    if cur_mtime != last_mtime:
                        last_mtime = cur_mtime
                        model = SAC.load(model_path)
                        self._broadcast("model_updated", {
                            "file": "humanoid_velocity/best.zip",
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
  .no-data { text-align: center; color: #475569; margin-top: 80px; font-size: 18px; }
  .progress-wrap { background: #1e293b; border-radius: 10px; padding: 16px; margin-bottom: 20px; }
  .progress-label { font-size: 13px; color: #94a3b8; margin-bottom: 8px; }
  .progress-bar-bg { background: #0f172a; border-radius: 6px; height: 18px; overflow: hidden; }
  .progress-bar { background: linear-gradient(90deg,#6366f1,#38bdf8); height: 100%;
                  border-radius: 6px; transition: width 0.5s; }
  .stages { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 12px; margin-bottom: 24px; }
  .stage-card { background: #1e293b; border-radius: 10px; padding: 14px;
                border: 2px solid transparent; }
  .stage-card.done { border-color: #4ade80; }
  .stage-card .s-label { font-size: 13px; color: #94a3b8; margin-bottom: 2px; }
  .stage-card .s-name  { font-size: 16px; font-weight: 700; color: #e2e8f0; }
  .stage-card .s-rew   { font-size: 22px; font-weight: 700; color: #38bdf8; margin-top: 4px; }
  .stage-card .s-note  { font-size: 11px; color: #64748b; margin-top: 3px; }
  .stage-card .s-done  { font-size: 11px; color: #4ade80; margin-top: 3px; }
  .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
             gap: 12px; margin-bottom: 24px; }
  .metric-card { background: #1e293b; border-radius: 10px; padding: 14px; text-align: center; }
  .metric-card .m-label { font-size: 13px; font-weight: 600; color: #cbd5e1; letter-spacing: 0.3px; }
  .metric-card .m-sublabel { font-size: 10px; color: #64748b; margin-top: 2px; }
  .metric-card .m-value { font-size: 26px; font-weight: 700; margin-top: 6px; }
  .color-green  { color: #4ade80; }
  .color-blue   { color: #38bdf8; }
  .color-yellow { color: #facc15; }
  .color-purple { color: #c084fc; }
  .color-red    { color: #f87171; }
  .chart-box { background: #1e293b; border-radius: 10px; padding: 16px; margin-bottom: 20px; }
  .chart-box h3 { font-size: 14px; color: #94a3b8; margin-bottom: 10px; }
  canvas { max-height: 260px; }
  .section-title { font-size: 14px; color: #94a3b8; margin: 20px 0 10px 0;
                   padding-left: 10px; border-left: 3px solid #6366f1; }
  .viz-link { display: inline-block; margin: 0 auto; background: #1e293b;
              color: #38bdf8; padding: 8px 20px; border-radius: 8px; text-decoration: none;
              border: 1px solid #334155; font-size: 13px; }
  .viz-link:hover { background: #334155; }
  .top-bar { display: flex; justify-content: space-between; align-items: center;
             margin-bottom: 20px; }
  .updated { font-size: 11px; color: #475569; }
</style>
</head>
<body>
<div class="top-bar">
  <div>
    <h1 id="pageTitle">JPRobot 训练面板</h1>
    <div class="subtitle" id="pageSubtitle"></div>
  </div>
  <a class="viz-link" href="/viz">3D 可视化 →</a>
</div>

<div id="app">
  <div class="no-data" id="noData" style="display:none;">暂无训练数据<br><small style="font-size:13px;color:#334155;">等待 live_dashboard.json...</small></div>

  <div id="mainContent" style="display:none;">
    <!-- 进度条 -->
    <div class="progress-wrap">
      <div class="progress-label" id="progressLabel">进度</div>
      <div class="progress-bar-bg">
        <div class="progress-bar" id="progressBar" style="width:0%"></div>
      </div>
    </div>

    <!-- 阶段卡片 -->
    <div class="section-title">训练阶段</div>
    <div class="stages" id="stagesGrid"></div>

    <!-- 指标卡片 -->
    <div class="section-title">当前指标</div>
    <div class="metrics" id="metricsGrid"></div>

    <!-- 奖励曲线 -->
    <div class="section-title">奖励曲线</div>
    <div class="chart-box">
      <h3>ep_rew_mean vs 训练步数</h3>
      <canvas id="rewChart"></canvas>
    </div>
  </div>
</div>

<div class="updated" id="updatedAt" style="text-align:right;margin-top:12px;"></div>

<script>
let rewChart = null;

function colorClass(c) {
  const map = {green:'color-green', blue:'color-blue', yellow:'color-yellow',
               purple:'color-purple', red:'color-red'};
  return map[c] || 'color-blue';
}

function fmtSteps(n) {
  if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(0) + 'K';
  return String(n);
}

function renderSpec(spec) {
  // Title
  document.getElementById('pageTitle').textContent = spec.title || 'JPRobot 训练面板';

  // Progress
  const prog = spec.progress || {};
  const cur  = prog.current_steps || 0;
  const tot  = prog.total_steps   || 1;
  const pct  = Math.min(100, (cur / tot * 100)).toFixed(1);
  document.getElementById('progressLabel').textContent =
    `总进度  ${fmtSteps(cur)} / ${fmtSteps(tot)}  (${pct}%)`;
  document.getElementById('progressBar').style.width = pct + '%';

  // Stages
  const sg = document.getElementById('stagesGrid');
  sg.innerHTML = '';
  for (const s of (spec.stages || [])) {
    const card = document.createElement('div');
    card.className = 'stage-card' + (s.done ? ' done' : '');
    card.innerHTML = `
      <div class="s-label">${s.name}</div>
      <div class="s-name">${s.label || s.name}</div>
      <div class="s-rew">${s.reward !== undefined ? s.reward : '-'}</div>
      ${s.note ? `<div class="s-note">${s.note}</div>` : ''}
      <div class="s-done">${s.done ? '✓ 已完成' : '训练中...'}</div>
    `;
    sg.appendChild(card);
  }

  // Metrics
  const mg = document.getElementById('metricsGrid');
  mg.innerHTML = '';
  for (const m of (spec.metrics || [])) {
    const card = document.createElement('div');
    card.className = 'metric-card';
    card.innerHTML = `
      <div class="m-label">${m.label}</div>
      ${m.sublabel ? `<div class="m-sublabel">${m.sublabel}</div>` : ''}
      <div class="m-value ${colorClass(m.color)}">${m.value}</div>
    `;
    mg.appendChild(card);
  }

  // Updated at
  document.getElementById('updatedAt').textContent =
    spec.updated_at ? `更新于 ${spec.updated_at}` : '';
}

function renderHistory(history) {
  const canvas = document.getElementById('rewChart');
  const labels = history.map(p => fmtSteps(p.total_timesteps));
  const data   = history.map(p => p.ep_rew_mean);

  // Color points by stage
  const stageColors = {jump:'#6366f1', rotate:'#38bdf8', land:'#4ade80', full:'#facc15'};
  const pointColors = history.map(p => stageColors[p.stage] || '#94a3b8');

  if (rewChart) rewChart.destroy();
  rewChart = new Chart(canvas, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: '奖励 (ep_rew_mean)',
        data,
        borderColor: '#38bdf8',
        backgroundColor: 'rgba(56,189,248,0.1)',
        pointBackgroundColor: pointColors,
        pointRadius: history.length > 200 ? 0 : 3,
        borderWidth: 2,
        tension: 0.3,
        fill: true,
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: '#94a3b8' } } },
      scales: {
        x: { ticks: { color: '#64748b', maxTicksLimit: 10 },
             grid: { color: '#1e293b' } },
        y: { ticks: { color: '#64748b' },
             grid: { color: '#334155' } },
      }
    }
  });
}

async function fetchData() {
  try {
    const r = await fetch('/api/data');
    if (!r.ok) return;
    const d = await r.json();

    const hasData = d.spec && d.spec.title;
    document.getElementById('noData').style.display      = hasData ? 'none'  : 'block';
    document.getElementById('mainContent').style.display = hasData ? 'block' : 'none';
    document.getElementById('pageSubtitle').textContent  = hasData ? (d.spec.run_id || '') : '';

    if (hasData) {
      renderSpec(d.spec);
      if (d.history && d.history.length > 0) renderHistory(d.history);
    }
  } catch(e) {
    console.error('fetchData error:', e);
  }
}

fetchData();
setInterval(fetchData, 10000);
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
  <div id="sMoeRow" style="display:none;"><span class="label">Expert: </span><span class="val" id="sMoeExpert" style="font-weight:600;">-</span></div>
  <div id="sVelRow" style="display:none;">
    <div><span class="label">Cmd: </span><span class="val" id="sVelCmd" style="color:#a78bfa;">-</span></div>
    <div><span class="label">Vel: </span><span class="val" id="sVelActual" style="color:#34d399;">-</span></div>
    <div><span class="label">Err: </span><span class="val" id="sVelError" style="color:#fb923c;">-</span></div>
  </div>
</div>

<div id="controls">
  <label style="color:#94a3b8;">模型选择</label>
  <select id="modelSelect"><option value="best.zip">best.zip</option></select>
  <button id="startBtn" onclick="doControl('start')">开始</button>
  <button id="foxBtn" onclick="toggleFoxMode()" style="background:#0f766e;border-color:#0f766e;margin-top:4px;font-size:12px;">🦊 Fox 模型</button>
  <button id="simpleBtn" onclick="toggleSimpleMode()" style="background:#334155;border-color:#475569;margin-top:4px;font-size:12px;">⬛ 简化</button>
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
scene.background = new THREE.Color(0x3a4060);
scene.fog = new THREE.Fog(0x3a4060, 8, 20);

// Z-up camera (match PyBullet)
const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.01, 30);
camera.up.set(0, 0, 1);
camera.position.set(0.25, -0.3, 0.25);

const controls = new OrbitControls(camera, canvas);
controls.target.set(0, 0, 0.05);
controls.enableDamping = true;
controls.dampingFactor = 0.1;
controls.update();

// ─── Lights ─────────────────────────────────────────────────
// 环境光（提升整体亮度，让黑色猫体与背景有对比）
const ambientLight = new THREE.AmbientLight(0xfff0e0, 1.6);
scene.add(ambientLight);

// 主光源（加强正面打光，让黑色表面有光泽感）
const dirLight = new THREE.DirectionalLight(0xfffbe8, 2.2);
dirLight.position.set(0.8, -1.2, 2.5);
dirLight.castShadow = true;
dirLight.shadow.mapSize.set(2048, 2048);
dirLight.shadow.camera.left = -0.5; dirLight.shadow.camera.right = 0.5;
dirLight.shadow.camera.top = 0.5; dirLight.shadow.camera.bottom = -0.5;
dirLight.shadow.bias = -0.001;
scene.add(dirLight);

// 轮廓补光（冷蓝色，增强立体感）
const rimLight = new THREE.DirectionalLight(0xaad4ff, 1.0);
rimLight.position.set(-1, 1.5, 1);
scene.add(rimLight);

// 底部反射光（暖色）
const fillLight = new THREE.DirectionalLight(0xffd090, 0.7);
fillLight.position.set(0, 0, -1);
scene.add(fillLight);

// ─── Ground plane ───────────────────────────────────────────
// 地面和格子做成 200×200m，并跟随机器人移动，机器人跑多远都不会"飞出空间"
const groundGeo = new THREE.PlaneGeometry(200, 200);
const groundMat = new THREE.MeshToonMaterial({ color: 0x2e3358 });
const ground = new THREE.Mesh(groundGeo, groundMat);
ground.receiveShadow = true;
ground.position.set(0, 0, -0.001);
scene.add(ground);

// GridHelper：200m 范围，每格 1m（200 格）
const gridHelper = new THREE.GridHelper(200, 200, 0x6677dd, 0x4455aa);
gridHelper.rotation.x = Math.PI / 2;
gridHelper.position.set(0, 0, 0.001);
gridHelper.material.transparent = true;
gridHelper.material.opacity = 0.6;
scene.add(gridHelper);

// ─── Robot model ────────────────────────────────────────────
// URDF dimensions (meters) — 身体比例微调：头偏大、腿细一点、身体更椭圆
const TORSO_SIZE = [0.11, 0.075, 0.025];   // 身体略高（更椭圆感）
const UPPER_R = 0.004, UPPER_L = 0.045;    // 腿略粗一点点显得卡通
const LOWER_R = 0.003, LOWER_L = 0.05;
const PAW_R = 0.007, PAW_L = 0.009;        // 爪子略大

// ─── 卡通材质（MeshToonMaterial，黑白猫配色 / 燕尾服猫）───────
// 身体主色：黑色
const matTorso   = new THREE.MeshToonMaterial({ color: 0x1c1c1c });
// 胸腹部：白色
const matBelly    = new THREE.MeshToonMaterial({ color: 0xf0f0f0 });
// 前腿：白色（胸部延伸到前腿）
const matUpperFront = new THREE.MeshToonMaterial({ color: 0xf0f0f0 });
// 后腿：黑色
const matUpperBack  = new THREE.MeshToonMaterial({ color: 0x1c1c1c });
// 下肢：深灰黑色
const matLower    = new THREE.MeshToonMaterial({ color: 0x2a2a2a });
// 爪子：白色
const matPaw      = new THREE.MeshToonMaterial({ color: 0xf5f5f5 });
// 电池/背部装饰：深灰
const matBattery  = new THREE.MeshToonMaterial({ color: 0x111111 });
// 耳朵：黑色外耳
const matEarOuter = new THREE.MeshToonMaterial({ color: 0x1c1c1c });
// 耳朵内：粉色内耳
const matEarInner = new THREE.MeshToonMaterial({ color: 0xff9eb5 });
// 尾巴：黑色
const matTail     = new THREE.MeshToonMaterial({ color: 0x1c1c1c });

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

// ─── Simple Robot Group（原始简化版，无猫造型）──────────────────
// 用 MeshLambertMaterial 纯色几何体，和最早版本一致
const smBodyMat  = new THREE.MeshLambertMaterial({ color: 0x4488cc });
const smLegMat   = new THREE.MeshLambertMaterial({ color: 0x5599dd });
const smLowerMat = new THREE.MeshLambertMaterial({ color: 0x2255aa });
const smPawMat   = new THREE.MeshLambertMaterial({ color: 0x88bbee });

const simpleGroup = new THREE.Group();
simpleGroup.visible = false;
scene.add(simpleGroup);

// 躯干
simpleGroup.add((() => {
  const m = new THREE.Mesh(new THREE.BoxGeometry(...TORSO_SIZE), smBodyMat);
  m.castShadow = true; return m;
})());
// 电池块
simpleGroup.add((() => {
  const m = new THREE.Mesh(new THREE.BoxGeometry(0.085, 0.038, 0.012), smBodyMat);
  m.position.set(0, 0, -0.016); m.castShadow = true; return m;
})());

// 四条腿（和猫完全相同的 URDF 尺寸，但无装饰）
const smLegs = {};
legConfig.forEach(cfg => {
  const shoulderPivot = new THREE.Group();
  shoulderPivot.position.set(cfg.ox, cfg.oy, 0);
  simpleGroup.add(shoulderPivot);

  const upper = new THREE.Mesh(
    new THREE.CylinderGeometry(UPPER_R * 1.2, UPPER_R * 1.2, UPPER_L, 8), smLegMat);
  upper.rotation.x = Math.PI / 2;
  upper.position.set(0, 0, -UPPER_L / 2);
  upper.castShadow = true;
  shoulderPivot.add(upper);

  const elbowStatic = new THREE.Group();
  elbowStatic.position.set(0, cfg.elbowOy, -UPPER_L);
  elbowStatic.rotation.y = -Math.PI / 2;
  shoulderPivot.add(elbowStatic);

  const elbowPivot = new THREE.Group();
  elbowStatic.add(elbowPivot);

  const lower = new THREE.Mesh(
    new THREE.CylinderGeometry(LOWER_R, LOWER_R * 1.2, LOWER_L, 8), smLowerMat);
  lower.rotation.x = Math.PI / 2;
  lower.position.set(0, 0, -LOWER_L / 2);
  lower.castShadow = true;
  elbowPivot.add(lower);

  const paw = new THREE.Mesh(new THREE.SphereGeometry(PAW_R, 7, 5), smPawMat);
  paw.position.set(0, 0, -LOWER_L - PAW_R * 0.7);
  elbowPivot.add(paw);

  smLegs[cfg.name] = { shoulderPivot, elbowPivot };
});

const simpleJointMap = [
  { pivot: smLegs.FL.shoulderPivot, prop: 'y' },
  { pivot: smLegs.FL.elbowPivot,    prop: 'y' },
  { pivot: smLegs.FR.shoulderPivot, prop: 'y' },
  { pivot: smLegs.FR.elbowPivot,    prop: 'y' },
  { pivot: smLegs.BR.shoulderPivot, prop: 'y' },
  { pivot: smLegs.BR.elbowPivot,    prop: 'y' },
  { pivot: smLegs.BL.shoulderPivot, prop: 'y' },
  { pivot: smLegs.BL.elbowPivot,    prop: 'y' },
];

// ─── Humanoid Robot Group（人形棍人，MuJoCo Z-up）────────────
// Position = pelvis/root center（MuJoCo qpos[0:3]）。
// 所有偏移量相对 pelvis 中心（z=0 = 骨盆）。
// 人形机器人站立时 pelvis z ≈ 0.93m，头顶约在 z=1.7m。
const humanoidGroup = new THREE.Group();
humanoidGroup.visible = false;
humanoidGroup.scale.set(1.25, 1.25, 1.25);  // 放大 25%，更容易看清
scene.add(humanoidGroup);

// ── 人形材质：Phong 材质 = 有光影立体感，区别于卡通平面 ──────
// 颜色方案：橙色躯干 + 电蓝肢体 + 金色关节 + 银白头部（参考 Gymnasium Humanoid 风格）
const matHCore   = new THREE.MeshPhongMaterial({ color: 0xff5500, shininess: 70,  specular: 0x441100 });  // 躯干：亮橙
const matHPelvis = new THREE.MeshPhongMaterial({ color: 0x334455, shininess: 40,  specular: 0x111111 });  // 骨盆：钢铁深灰
const matHLimb   = new THREE.MeshPhongMaterial({ color: 0x0099ff, shininess: 90,  specular: 0x003366 });  // 大腿/上臂：电蓝
const matHFore   = new THREE.MeshPhongMaterial({ color: 0x44ccff, shininess: 70,  specular: 0x003366 });  // 小腿/前臂：浅蓝
const matHHead   = new THREE.MeshPhongMaterial({ color: 0xdde8f0, shininess: 100, specular: 0x888899 });  // 头：银白
const matHJoint  = new THREE.MeshPhongMaterial({ color: 0xffcc00, shininess: 120, specular: 0x664400 });  // 关节球：金色
const matHFoot   = new THREE.MeshPhongMaterial({ color: 0x445566, shininess: 50,  specular: 0x222233 });  // 脚：暗钢色
const matHBody   = matHCore;  // 兼容旧引用

// 辅助函数：生成沿 Z 轴方向的圆柱（CylinderGeometry 默认沿 Y，旋转 90° 对齐 Z）
function _hCyl(r, h, mat) {
  const g = new THREE.CylinderGeometry(r, r, h, 16);  // 16 段 = 更圆滑
  const m = new THREE.Mesh(g, mat);
  m.rotation.x = Math.PI / 2;
  m.castShadow = true;
  return m;
}
function _hSphere(r, mat) {
  const m = new THREE.Mesh(new THREE.SphereGeometry(r, 16, 12), mat);  // 高分辨率球
  m.castShadow = true;
  return m;
}

// 身体躯干（pelvis 向上 0→0.32m）
const hTorsoLow = _hCyl(0.11, 0.20, matHBody);
hTorsoLow.position.set(0, 0, 0.10);
humanoidGroup.add(hTorsoLow);

const hTorsoUp = _hCyl(0.09, 0.18, matHBody);
hTorsoUp.position.set(0, 0, 0.29);
humanoidGroup.add(hTorsoUp);

// 颈部
const hNeck = _hCyl(0.04, 0.10, matHBody);
hNeck.position.set(0, 0, 0.43);
humanoidGroup.add(hNeck);

// 头部（球，z=0.58m）
const hHead = _hSphere(0.12, matHHead);
hHead.position.set(0, 0, 0.62);
humanoidGroup.add(hHead);

// 骨盆（pelvis 向下 z= -0.06）
const hPelvis = _hCyl(0.13, 0.12, matHPelvis);
hPelvis.position.set(0, 0, -0.06);
humanoidGroup.add(hPelvis);

// ── 腰部 pivot（abdomen 3 自由度）─────────────────────────────
// 腰部关节位于 z=+0.12（骨盆正上方）
const hWaistPivot = new THREE.Group();
hWaistPivot.position.set(0, 0, 0.12);
humanoidGroup.add(hWaistPivot);

// ── 右腿（Y 轴负方向 = 右）──────────────────────────────────
const hRHipPivot = new THREE.Group();
hRHipPivot.position.set(0, -0.09, -0.04);
humanoidGroup.add(hRHipPivot);

const hRThigh = _hCyl(0.06, 0.35, matHLimb);
hRThigh.position.set(0, 0, -0.175);
hRHipPivot.add(hRThigh);
hRHipPivot.add((() => { const b = _hSphere(0.07, matHJoint); b.position.set(0,0,0); return b; })());

const hRKneePivot = new THREE.Group();
hRKneePivot.position.set(0, 0, -0.35);
hRHipPivot.add(hRKneePivot);

const hRShin = _hCyl(0.05, 0.30, matHFore);
hRShin.position.set(0, 0, -0.15);
hRKneePivot.add(hRShin);
hRKneePivot.add((() => { const b = _hSphere(0.055, matHJoint); b.position.set(0,0,0); return b; })());

const hRFoot = new THREE.Mesh(new THREE.BoxGeometry(0.22, 0.09, 0.07), matHFoot);
hRFoot.position.set(0.06, 0, -0.33);
hRKneePivot.add(hRFoot);

// ── 左腿（Y 轴正方向 = 左）──────────────────────────────────
const hLHipPivot = new THREE.Group();
hLHipPivot.position.set(0, 0.09, -0.04);
humanoidGroup.add(hLHipPivot);

const hLThigh = _hCyl(0.06, 0.35, matHLimb);
hLThigh.position.set(0, 0, -0.175);
hLHipPivot.add(hLThigh);
hLHipPivot.add((() => { const b = _hSphere(0.07, matHJoint); b.position.set(0,0,0); return b; })());

const hLKneePivot = new THREE.Group();
hLKneePivot.position.set(0, 0, -0.35);
hLHipPivot.add(hLKneePivot);

const hLShin = _hCyl(0.05, 0.30, matHFore);
hLShin.position.set(0, 0, -0.15);
hLKneePivot.add(hLShin);
hLKneePivot.add((() => { const b = _hSphere(0.055, matHJoint); b.position.set(0,0,0); return b; })());

const hLFoot = new THREE.Mesh(new THREE.BoxGeometry(0.22, 0.09, 0.07), matHFoot);
hLFoot.position.set(0.06, 0, -0.33);
hLKneePivot.add(hLFoot);

// ── 右臂（肩部在 z=+0.42，Y 轴负方向 = 右）─────────────────
const hRShoulderPivot = new THREE.Group();
hRShoulderPivot.position.set(0, -0.20, 0.38);
humanoidGroup.add(hRShoulderPivot);

const hRUpperArm = _hCyl(0.04, 0.27, matHLimb);
hRUpperArm.position.set(0, 0, -0.135);
hRShoulderPivot.add(hRUpperArm);
hRShoulderPivot.add((() => { const b = _hSphere(0.05, matHJoint); b.position.set(0,0,0); return b; })());

const hRElbowPivot = new THREE.Group();
hRElbowPivot.position.set(0, 0, -0.27);
hRShoulderPivot.add(hRElbowPivot);

const hRForearm = _hCyl(0.033, 0.23, matHFore);
hRForearm.position.set(0, 0, -0.115);
hRElbowPivot.add(hRForearm);
hRElbowPivot.add((() => { const b = _hSphere(0.04, matHJoint); b.position.set(0,0,0); return b; })());

// ── 左臂（Y 轴正方向 = 左）──────────────────────────────────
const hLShoulderPivot = new THREE.Group();
hLShoulderPivot.position.set(0, 0.20, 0.38);
humanoidGroup.add(hLShoulderPivot);

const hLUpperArm = _hCyl(0.04, 0.27, matHLimb);
hLUpperArm.position.set(0, 0, -0.135);
hLShoulderPivot.add(hLUpperArm);
hLShoulderPivot.add((() => { const b = _hSphere(0.05, matHJoint); b.position.set(0,0,0); return b; })());

const hLElbowPivot = new THREE.Group();
hLElbowPivot.position.set(0, 0, -0.27);
hLShoulderPivot.add(hLElbowPivot);

const hLForearm = _hCyl(0.033, 0.23, matHFore);
hLForearm.position.set(0, 0, -0.115);
hLElbowPivot.add(hLForearm);
hLElbowPivot.add((() => { const b = _hSphere(0.04, matHJoint); b.position.set(0,0,0); return b; })());

// 关节映射：17 个 MuJoCo 马达 → Three.js pivot + 旋转轴
// MuJoCo actuator 顺序（与 humanoid.xml 一致）：
//   0-2:   abdomen_y/z/x（腰部 3 DOF）
//   3-6:   right_hip_x/z/y + right_knee
//   7-10:  left_hip_x/z/y  + left_knee
//   11-13: right_shoulder1/2 + right_elbow
//   14-16: left_shoulder1/2  + left_elbow
const humanoidJointMap = [
  { pivot: hWaistPivot,     axis: 'y' },  // 0: abdomen_y（前后弯腰）
  { pivot: hWaistPivot,     axis: 'z' },  // 1: abdomen_z（扭腰）
  { pivot: hWaistPivot,     axis: 'x' },  // 2: abdomen_x（侧弯）
  { pivot: hRHipPivot,      axis: 'x' },  // 3: right_hip_x
  { pivot: hRHipPivot,      axis: 'z' },  // 4: right_hip_z
  { pivot: hRHipPivot,      axis: 'y' },  // 5: right_hip_y（髋部摆腿）
  { pivot: hRKneePivot,     axis: 'x' },  // 6: right_knee（膝盖弯曲）
  { pivot: hLHipPivot,      axis: 'x' },  // 7: left_hip_x
  { pivot: hLHipPivot,      axis: 'z' },  // 8: left_hip_z
  { pivot: hLHipPivot,      axis: 'y' },  // 9: left_hip_y
  { pivot: hLKneePivot,     axis: 'x' },  // 10: left_knee
  { pivot: hRShoulderPivot, axis: 'y' },  // 11: right_shoulder1
  { pivot: hRShoulderPivot, axis: 'x' },  // 12: right_shoulder2
  { pivot: hRElbowPivot,    axis: 'x' },  // 13: right_elbow
  { pivot: hLShoulderPivot, axis: 'y' },  // 14: left_shoulder1
  { pivot: hLShoulderPivot, axis: 'x' },  // 15: left_shoulder2
  { pivot: hLElbowPivot,    axis: 'x' },  // 16: left_elbow
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
let foxMixer = null;   // AnimationMixer for Fox built-in walk animation
let foxWalkAction = null;

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

  // Scale: ~0.0013 makes Fox visually similar in size to the toon cat.
  foxModel.scale.set(0.0013, 0.0013, 0.0013);

  foxWrapper.add(foxModel);

  // Use Fox's built-in Walk animation driven by robot speed
  if (gltf.animations && gltf.animations.length > 0) {
    foxMixer = new THREE.AnimationMixer(foxModel);
    const walkClip = THREE.AnimationClip.findByName(gltf.animations, 'Walk')
                  || gltf.animations[0];
    foxWalkAction = foxMixer.clipAction(walkClip);
    foxWalkAction.play();
    foxWalkAction.timeScale = 0;  // start paused; speed set by robot velocity
    console.log('Fox animations:', gltf.animations.map(a => a.name));
  }
  console.log('Fox GLB loaded with AnimationMixer.');
}, undefined, (err) => {
  console.error('Failed to load Fox.glb:', err);
});

let simpleMode = false;
let humanoidMode = false;

window.toggleFoxMode = function() {
  if (simpleMode) return;  // 简化模式下不切换
  foxMode = !foxMode;
  robotGroup.visible = !foxMode;
  foxWrapper.visible = foxMode;
  const btn = document.getElementById('foxBtn');
  btn.textContent = foxMode ? '🐱 卡通猫' : '🦊 Fox 模型';
  btn.style.background = foxMode ? '#e8721a' : '#0f766e';
  btn.style.borderColor = foxMode ? '#c95e10' : '#0f766e';
};

window.toggleSimpleMode = function() {
  simpleMode = !simpleMode;
  const btn = document.getElementById('simpleBtn');
  const foxBtn = document.getElementById('foxBtn');
  if (simpleMode) {
    robotGroup.visible = false;
    foxWrapper.visible = false;
    simpleGroup.visible = true;
    btn.textContent = '⬛ 退出简化';
    btn.style.background = '#64748b';
    foxBtn.style.opacity = '0.35';
    foxBtn.style.pointerEvents = 'none';
  } else {
    robotGroup.visible = !foxMode;
    foxWrapper.visible = foxMode;
    simpleGroup.visible = false;
    btn.textContent = '⬛ 简化';
    btn.style.background = '#334155';
    foxBtn.style.opacity = '1';
    foxBtn.style.pointerEvents = 'auto';
  }
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
    // MoE mode: show which expert and direction
    const moeRow = document.getElementById('sMoeRow');
    const moeExpert = document.getElementById('sMoeExpert');
    if (d.model_used && moeRow && moeExpert) {
      moeRow.style.display = '';
      const dirCN = {forward:'向前', backward:'向后', left:'向左', right:'向右'};
      const cn = dirCN[d.direction] || d.direction || '';
      const color = d.model_used === 'A' ? '#4ade80' : '#38bdf8';
      moeExpert.style.color = color;
      moeExpert.textContent = '[' + d.model_used + '] ' + cn;
    } else if (moeRow) {
      moeRow.style.display = 'none';
    }
  });

  evtSource.addEventListener('frame', e => {
    const d = JSON.parse(e.data);
    currentStep = d.step;
    totalReward += d.reward;

    const [px, py, pz] = d.position;
    const [qx, qy, qz, qw] = d.orientation;

    // ── 自动切换显示模式（无需手动点按钮）──────────────────────
    // 每次首帧检测到 robot_type 变化就自动同步，和四足完全一致的体验
    const isHumanoidFrame = d.robot_type === 'humanoid';
    if (isHumanoidFrame && !humanoidMode) {
      // 收到人形数据 → 自动进入人形模式
      humanoidMode = true;
      humanoidGroup.visible = true;
      robotGroup.visible = false;
      foxWrapper.visible = false;
      simpleGroup.visible = false;
      camera.position.set(2.5, -3.5, 2.5);
      controls.target.set(0, 0, 1.0);
      controls.update();
    } else if (!isHumanoidFrame && humanoidMode) {
      // 收到四足数据 → 自动退出人形模式
      humanoidMode = false;
      humanoidGroup.visible = false;
      robotGroup.visible = !foxMode && !simpleMode;
      foxWrapper.visible = foxMode && !simpleMode;
      simpleGroup.visible = simpleMode;
      camera.position.set(0.25, -0.3, 0.25);
      controls.target.set(0, 0, 0.05);
      controls.update();
    }

    if (d.robot_type === 'humanoid') {
      // ── 人形机器人（MuJoCo 引擎）────────────────────────────
      // MuJoCo qpos[:3] = torso（根体）位置。我们的 Three.js 几何体以 pelvis 为原点，
      // 脚底在 group local z ≈ -0.755。MuJoCo 的腿更长（≈1.32m），
      // 差值 ≈ 0.64m，用常量 H_GND 补偿，让脚尽量踩到地面。
      const H_GND = 0.64;
      humanoidGroup.position.set(px, py, pz - H_GND);
      humanoidGroup.quaternion.set(qx, qy, qz, qw);

      // 把 17 个关节角度写入对应 pivot 的旋转轴
      if (d.joints) d.joints.forEach((angle, i) => {
        if (humanoidJointMap[i]) {
          humanoidJointMap[i].pivot.rotation[humanoidJointMap[i].axis] = angle;
        }
      });

      // 相机跟随：和四足一样用 controls.target，目标对准腰部高度
      // OrbitControls 会自动保持相机与目标的相对位置，不需要单独移动 camera.position
      cameraTarget.lerp(new THREE.Vector3(px, py, pz - H_GND + 0.7), 0.05);

    } else {
      // ── 四足机器人（PyBullet 引擎）──────────────────────────
      robotGroup.position.set(px, py, pz);
      robotGroup.quaternion.set(qx, qy, qz, qw);

      foxWrapper.position.set(px, py, pz);
      foxWrapper.quaternion.set(qx, qy, qz, qw);

      d.joints.forEach((angle, i) => {
        if (jointMap[i]) jointMap[i].pivot.rotation[jointMap[i].prop] = angle;
      });

      simpleGroup.position.set(px, py, pz);
      simpleGroup.quaternion.set(qx, qy, qz, qw);
      d.joints.forEach((angle, i) => {
        if (simpleJointMap[i]) simpleJointMap[i].pivot.rotation[simpleJointMap[i].prop] = angle;
      });

      // Fox Walk 动画速度：机器人爬行 0.1-0.3 m/s → timeScale 0.5-1.5
      if (foxMode && foxWalkAction) {
        foxWalkAction.timeScale = Math.min(2.5, Math.max(0, Math.abs(speed) * 5));
      }

      cameraTarget.lerp(new THREE.Vector3(px, py, 0.05), 0.05);
    }

    // ── 通用：距离 / 速度 / 轨迹 / 统计 ────────────────────────
    const dist = px - (d.step === 1 ? px : 0);
    maxDistance = Math.max(maxDistance, dist);
    speed = (px - prevX) / 0.02;
    prevX = px;

    if (d.step % 3 === 0) addTrailPoint(px, py, pz);

    controls.target.copy(cameraTarget);

    // 地面跟随机器人，保证机器人始终在地面中央（0.1 平滑追踪）
    ground.position.x    += (px - ground.position.x)    * 0.1;
    ground.position.y    += (py - ground.position.y)    * 0.1;
    gridHelper.position.x += (px - gridHelper.position.x) * 0.1;
    gridHelper.position.y += (py - gridHelper.position.y) * 0.1;

    const maxSteps = d.robot_type === 'humanoid' ? '1000' : '250';
    document.getElementById('sStep').textContent = currentStep + '/' + maxSteps;
    document.getElementById('sReward').textContent = totalReward.toFixed(0);
    document.getElementById('sDistance').textContent = maxDistance.toFixed(3) + 'm';
    document.getElementById('sSpeed').textContent = speed.toFixed(3) + ' m/s';

    // 速度命令 HUD（仅速度跟随模式显示）
    const velRow = document.getElementById('sVelRow');
    if (d.vel_cmd) {
      velRow.style.display = '';
      const c = d.vel_cmd;
      const a = d.vel_actual || [0,0,0];
      document.getElementById('sVelCmd').textContent =
        `vx${c[0]>=0?'+':''}${c[0].toFixed(2)} vy${c[1]>=0?'+':''}${c[1].toFixed(2)} wz${c[2]>=0?'+':''}${c[2].toFixed(2)}`;
      document.getElementById('sVelActual').textContent =
        `vx${a[0]>=0?'+':''}${a[0].toFixed(2)} vy${a[1]>=0?'+':''}${a[1].toFixed(2)} wz${a[2]>=0?'+':''}${a[2].toFixed(2)}`;
      const err = d.vel_error !== undefined ? d.vel_error : null;
      document.getElementById('sVelError').textContent = err !== null ? err.toFixed(4) : '-';
    } else {
      velRow.style.display = 'none';
    }
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

    // 点击"开始"时自动切换模式：
    // - 选了人形模型 → 自动进入 humanoidMode
    // - 选了非人形模型但当前在 humanoidMode → 自动退出，无需手动点"退出人形"
    const isHumanoidModel = model && model.toLowerCase().includes('humanoid');
    if (isHumanoidModel && !humanoidMode) {
      humanoidMode = true;
      humanoidGroup.visible = true;
      robotGroup.visible = false;
      foxWrapper.visible = false;
      simpleGroup.visible = false;
      camera.position.set(2.5, -3.5, 2.5);
      controls.target.set(0, 0, 1.0);
      controls.update();
    } else if (!isHumanoidModel && humanoidMode) {
      humanoidMode = false;
      humanoidGroup.visible = false;
      robotGroup.visible = !foxMode && !simpleMode;
      foxWrapper.visible = foxMode && !simpleMode;
      simpleGroup.visible = simpleMode;
      camera.position.set(0.25, -0.3, 0.25);
      controls.target.set(0, 0, 0.05);
      controls.update();
    }
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
      const bfLabels = {
        jump: '起跳', rotate: '旋转', land: '落地', full: '完整后空翻'
      };
      const bfIconList = ['🔵', '🟠', '🟢', '⚡', '🔴', '🟣', '🟡', '🩵'];
      const bfIcons = v => bfIconList[(parseInt(v.slice(1)) - 1) % bfIconList.length];
      const bfMatch = name.match(/^__backflip_(v\d+)_(\w+)__$/);
      const sacCkptMatch = name.match(/^__humanoid_sac_ckpt_(\d+)__$/);
      opt.textContent = name === '__moe__'                  ? '✨ MoE 融合（A前进 + B后退/侧移）'
        : name === '__route_a__'                           ? '🟢 Route A v3（前进最强）'
        : name === '__route_b__'                           ? '🟦 Route B v4（多方向稳定）'
        : name === '__humanoid_upright_c2__'               ? '🧍 Humanoid 直立 C2（软奖励直立版）'
        : name === '__humanoid_velocity__'                 ? '🏃 Humanoid 万向行走（速度命令跟随）'
        : name === '__humanoid_sac__'                      ? '🚶 Humanoid SAC — best（弯腰基线版）'
        : name === '__backflip_latest__'                   ? '🤸 Backflip — best（最新最佳）'
        : sacCkptMatch ? `🚶 Humanoid SAC — ${(parseInt(sacCkptMatch[1])/1e6).toFixed(1)}M 步快照`
        : bfMatch ? `${bfIcons(bfMatch[1])} Backflip ${bfMatch[1].toUpperCase()} — ${bfLabels[bfMatch[2]] || bfMatch[2]}`
        : name === '__latest__'                     ? '● 实时（训练中）'
        : name;
      sel.appendChild(opt);
    });
  });

// ─── Render loop ────────────────────────────────────────────
const clock = new THREE.Clock();
function animate() {
  requestAnimationFrame(animate);
  const delta = clock.getDelta();
  if (foxMixer) foxMixer.update(delta);
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


def _list_backflip_versions(base):
    """Scan trained/ for backflip_vN dirs, return [(ver_str, dir_name)] newest-first.

    Handles:
      backflip_vN  → "vN"  (N = 1, 2, 3, ...)
      backflip     → "v2"  (legacy naming, used as fallback if backflip_v2 absent)
    """
    seen = {}
    if os.path.isdir(base):
        for name in os.listdir(base):
            m = re.match(r'^backflip_v(\d+)$', name)
            if m and os.path.isdir(os.path.join(base, name)):
                n = int(m.group(1))
                seen[n] = (n, f"v{n}", name)
    return [(label, dirname) for _, label, dirname in sorted(seen.values(), reverse=True)]


def _list_snapshots(trained_dir):
    """List meaningful model options for visualization.

    训练结束后语义清晰的三档：
      __moe__      ✨ MoE 融合（Route A 向前 + Route B 多方向）
      __route_a__  Route A v3 best（向前最强，direction-reward）
      __route_b__  Route B v4 best（多方向稳定，velocity-tracking）

    只在有对应文件时显示，避免噪音。
    training 进行中额外提供 __latest__（实时跟随最新 checkpoint）。
    """
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "trained"))
    snapshots = []

    # 1. MoE — 最推荐，排第一
    if os.path.exists(os.path.abspath(MOE_EVAL)):
        snapshots.append("__moe__")

    # 2. Route A v3 best（向前最强）
    route_a_best = os.path.join(base, "route_a_v3", "snapshots", "best.zip")
    if os.path.exists(route_a_best):
        snapshots.append("__route_a__")

    # 3. Route B v4 best（多方向稳定）
    route_b_best = os.path.join(base, "route_b_v4", "snapshots", "best.zip")
    if os.path.exists(route_b_best):
        snapshots.append("__route_b__")

    # 4. 固定置顶：人形机器人模型（不参与时间混排）
    # C2 直立版（更新，排前面）
    c2_best = os.path.join(base, "humanoid_upright", "C2", "best.zip")
    if os.path.exists(c2_best):
        snapshots.append("__humanoid_upright_c2__")

    # 速度命令跟随版（万向行走，379维obs）
    vel_best = os.path.join(base, "humanoid_velocity", "best.zip")
    if os.path.exists(vel_best):
        snapshots.append("__humanoid_velocity__")

    sac_best = os.path.join(base, "humanoid_sac", "best.zip")
    if os.path.exists(sac_best):
        snapshots.append("__humanoid_sac__")

    bf_versions = _list_backflip_versions(base)
    if bf_versions:
        _bf_latest_ver, _bf_latest_dir = bf_versions[0]
        bf_latest_path = os.path.join(base, _bf_latest_dir, "full", "best.zip")
        if os.path.exists(bf_latest_path):
            snapshots.append("__backflip_latest__")

    # 5. 时间混排：Humanoid SAC checkpoints + Backflip 各版本各阶段（最新在前）
    timed = []  # [(mtime, token)]

    # Humanoid SAC checkpoints（不含 best.zip，已置顶）
    sac_ckpt_dir = os.path.join(base, "humanoid_sac", "checkpoints")
    if os.path.isdir(sac_ckpt_dir):
        for fname in os.listdir(sac_ckpt_dir):
            m = re.match(r'^humanoid_sac_(\d+)_steps\.zip$', fname)
            if m:
                steps = int(m.group(1))
                fpath = os.path.join(sac_ckpt_dir, fname)
                try:
                    mt = os.path.getmtime(fpath)
                except OSError:
                    mt = 0
                timed.append((mt, f"__humanoid_sac_ckpt_{steps}__"))

    # Backflip 各版本各阶段（不含 best，已置顶）
    for bf_ver, bf_dir in bf_versions:
        for bf_stage in ("full", "land", "rotate", "jump"):
            bf_best = os.path.join(base, bf_dir, bf_stage, "best.zip")
            if os.path.exists(bf_best):
                try:
                    mt = os.path.getmtime(bf_best)
                except OSError:
                    mt = 0
                timed.append((mt, f"__backflip_{bf_ver}_{bf_stage}__"))

    timed.sort(key=lambda x: x[0], reverse=True)
    snapshots.extend(token for _, token in timed)

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
    trained_dir = os.path.join(os.path.dirname(__file__), "..", "trained")

    def do_GET(self):
        if self.path == "/" or self.path == "/dashboard":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode("utf-8"))

        elif self.path == "/api/data":
            spec    = find_live_dashboard()
            history = parse_history(spec) if spec else []
            # strip internal _dir key before sending
            if spec is not None:
                spec = {k: v for k, v in spec.items() if k != "_dir"}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({
                "spec": spec,
                "history": history,
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
    parser.add_argument(
        "--run-id", type=str, default=None, dest="run_id",
        help=(
            "Restrict dashboard scan to trained/<run-id>/ (e.g. --run-id backflip). "
            "Without this, scans all of trained/ for the newest live_dashboard.json."
        ),
    )
    args = parser.parse_args()

    _base_trained = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "trained"))
    if args.run_id:
        trained_dir = os.path.join(_base_trained, args.run_id)
        globals()["DASHBOARD_DIR"] = trained_dir
        print(f"[Dashboard] --run-id={args.run_id!r} → scanning {trained_dir}")
    else:
        trained_dir = _base_trained
    DashboardHandler.trained_dir = trained_dir

    eval_engine = EvalEngine(trained_dir)

    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer(("0.0.0.0", args.port), DashboardHandler)
    pid = os.getpid()
    print(f"JPRobot Training Dashboard")
    print(f"  PID: {pid}")
    print(f"  Port: {args.port}")
    print(f"  Dashboard scan dir: {DASHBOARD_DIR}")
    print(f"  Dashboard: http://127.0.0.1:{args.port}/dashboard")
    print(f"  3D Viz:    http://127.0.0.1:{args.port}/viz")
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
