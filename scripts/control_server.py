#!/usr/bin/env python3
"""JPRobot 机器人控制服务器。

接收动作命令，生成 PyBullet 仿真动画 GIF 返回。

Usage:
    python scripts/control_server.py          # default port 18792
    python scripts/control_server.py --port 9000
"""

import difflib
import io
import json
import os
import re
import sys
import argparse
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from socketserver import ThreadingMixIn

import numpy as np
import pybullet as p
import pybullet_data
import imageio
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jprobot.robot.skills import SkillRegistry, SkillType

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
DEFAULT_STEPS = 5
MAX_STEPS = 60        # MoE 模式最多 60步（env 约 251 步后自然终止）
MAX_TOTAL_FRAMES = 60 # 固件关键帧 GIF 最大帧数（MoE 不受此限制）

TRAINED_ROOT = Path(__file__).parent.parent / "trained"

GIF_W, GIF_H = 480, 360

# 初始站姿（度），用于 POSTURE 插值起点，也是 URDF 坐标系的站立参考
INIT_POSE_DEG = [50, 0, 50, 0, 50, 0, 50, 0]

# 每帧渲染前跑多少次物理仿真步
# PyBullet 默认 timestep=1/240s，跑 8 步 ≈ 33ms 物理时间/帧（接近 30fps 真实感）
# force 足够大时，关节才能在这 8 步内真正到达目标角度
PHYSICS_STEPS_PER_FRAME = 8
PHYSICS_JOINT_FORCE = 200.0             # 固件关键帧：大力，跨帧角度跳跃需要大力到位
PHYSICS_JOINT_FORCE_PROGRAMMATIC = 10.0 # 程序化正弦步态：小力，角度变化平滑不需要大力

# ---------------------------------------------------------------------------
# 关节控制模式（按技能类型）
#
# "physics"   → setJointMotorControl2：受物理约束，力不足时到不了目标角度。
#               适合步态/姿态，仿真更真实，与真机行为一致。
#               接真机时：这里的仿真和串口指令是两套系统，无需切换。
#
# "kinematic" → resetJointState：直接强制设置角度，跳过物理约束。
#               适合行为动作（翻转、击掌），保证视觉上完整还原固件快照。
#
# 若未来需要全部切回 physics（例如做仿真精度对比），
# 把下面三行的 "kinematic" 改成 "physics" 即可。
# ---------------------------------------------------------------------------
JOINT_CONTROL_MODE: dict[SkillType, str] = {
    SkillType.GAIT:     "physics",    # 步态：物理仿真，脚步自然
    SkillType.POSTURE:  "physics",    # 姿态：物理仿真，插值平滑
    SkillType.BEHAVIOR: "kinematic",  # 行为：运动学，固件快照直接播放
}


# ---------------------------------------------------------------------------
# 带步数的步态正则（匹配用户输入中的数字步数）
# ---------------------------------------------------------------------------
GAIT_PATTERNS = [
    (r"强化.*?(\d+)步",    "rl"),       # RL 策略优先匹配
    (r"转身.*后.*?(\d+)步", "turn_bk"), # 转身向后走
    (r"转身向后走",         "turn_bk"), # 无步数版本（用 DEFAULT_STEPS）
    (r"向左走(\d+)步",     "wkL"),
    (r"向右走(\d+)步",     "wkR"),
    (r"向前走(\d+)步",     "wkF"),
    (r"后退(\d+)步",       "bk"),
    (r"小跑(\d+)步",       "trF"),
    (r"(\d+)步",           "wkF"),      # fallback：有步数就默认向前走
]

# RL 模型选择逻辑
def _resolve_rl_model_path(model_type: str = "best") -> Path:
    """
    解析 RL 模型路径，与 viz/dashboard 三个按钮对应：
        - "best"    → best.zip（最佳 eval 快照，SnapshotCallback 维护）
        - "latest"  → 最近修改的 step_*.zip 或 checkpoint（训练中实时更新）
        - "process" → step_*.zip 中间点（约 40% 进度，展示学习过程）
    """
    import re as _re
    trained_dir = Path(__file__).parent.parent / "trained"
    snap_dir = trained_dir / "snapshots"
    ckpt_dir = trained_dir / "checkpoints"

    if model_type == "best":
        return snap_dir / "best.zip"

    elif model_type == "latest":
        # 训练过程产生的模型按修改时间排序：step_*.zip > checkpoints/*.zip
        all_zips = list(snap_dir.glob("step_*.zip")) + list(ckpt_dir.glob("*.zip"))
        if all_zips:
            return max(all_zips, key=lambda p: p.stat().st_mtime)
        # 训练结束后保存的 bittle_ppo.zip
        bpp = trained_dir / "bittle_ppo.zip"
        return bpp if bpp.exists() else snap_dir / "best.zip"

    elif model_type == "process":
        # 找所有奖励为正的 step_*.zip，取约 40% 进度处的快照（早中期学习阶段）
        candidates = []
        for p in snap_dir.glob("step_*.zip"):
            m = _re.match(r"step_([\d.]+)M_rew_(-?\d+)\.zip", p.name)
            if m and int(m.group(2)) > 0:
                candidates.append((float(m.group(1)), p))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            mid_idx = max(0, len(candidates) * 2 // 5)  # ~40% 进度
            return candidates[mid_idx][1]
        # fallback：curriculum 阶段快照（取时间最早的，代表过程起点）
        curr = sorted(snap_dir.glob("curriculum_*.zip"), key=lambda p: p.stat().st_mtime)
        return curr[0] if curr else snap_dir / "best.zip"

    return snap_dir / "best.zip"


RL_MODEL_PATH = _resolve_rl_model_path("best")

# ---------------------------------------------------------------------------
# 技能注册表 & 关键帧数据（模块级单例）
# ---------------------------------------------------------------------------
_SKILL_REGISTRY = SkillRegistry()
_KEYFRAMES: dict = {}
SIM_LOCK = threading.Lock()


def _mirror_frames(frames: list[list[float]]) -> list[list[float]]:
    """模仿固件 mirror()：将左转步态镜像为右转步态。

    固件在硬件顺序（hw8-hw15）上交换相邻对，但我们的数据已经过 REORDER
    变换为仿真顺序，左右腿对应关系如下：
        lf_shoulder(0) ↔ rf_shoulder(2)
        lf_knee(1)     ↔ rf_knee(3)
        rb_hip(4)      ↔ lb_hip(6)
        rb_knee(5)     ↔ lb_knee(7)
    """
    result = []
    for frame in frames:
        m = list(frame)
        m[0], m[2] = m[2], m[0]   # lf_shoulder ↔ rf_shoulder
        m[1], m[3] = m[3], m[1]   # lf_knee     ↔ rf_knee
        m[4], m[6] = m[6], m[4]   # rb_hip      ↔ lb_hip
        m[5], m[7] = m[7], m[5]   # rb_knee     ↔ lb_knee
        result.append(m)
    return result


def _load_keyframes() -> None:
    global _KEYFRAMES
    p_json = Path(__file__).parent.parent / "jprobot" / "data" / "skills_keyframes.json"
    if p_json.exists():
        _KEYFRAMES = json.loads(p_json.read_text())
        # 固件没有 wkR，用 mirror(wkL) 实时生成（与真机逻辑一致）
        if "wkL" in _KEYFRAMES and "wkR" not in _KEYFRAMES:
            wkL = _KEYFRAMES["wkL"]
            _KEYFRAMES["wkR"] = {
                "num_frames": wkL["num_frames"],
                "delay_ms":   wkL["delay_ms"],
                "frames":     _mirror_frames(wkL["frames"]),
            }
            print("已生成 wkR（mirror of wkL）")
        print(f"已加载关键帧：{len(_KEYFRAMES)} 个技能")
    else:
        print("WARNING: skills_keyframes.json not found, using programmatic gait only")


_load_keyframes()


# ---------------------------------------------------------------------------
# 命令解析：三层匹配
# ---------------------------------------------------------------------------
def parse_command(text: str) -> tuple[str, int, str]:
    """
    解析命令文本，返回 (skill_code, steps, model_type)。

    model_type: "best" | "latest" | "process"
        强化: 使用 best（最佳模型）
        实时强化: 使用 latest（最新检查点）
        过程强化: 使用 process（过程回放）
    """
    model_type = "best"  # 默认用最佳模型

    # 检测模型类型前缀
    if "实时" in text:
        model_type = "latest"
    elif "过程" in text:
        model_type = "process"

    # 层 1：带步数的步态（正则）
    for pattern, gait in GAIT_PATTERNS:
        m = re.search(pattern, text)
        if m:
            steps = min(int(m.group(1)), MAX_STEPS) if m.lastindex else DEFAULT_STEPS
            return gait, steps, model_type

    # 层 2：精确中文名 / 英文 code 匹配
    for skill in _SKILL_REGISTRY.all_skills:
        if skill.name_cn in text or skill.code.lower() in text.lower():
            return skill.code, DEFAULT_STEPS, model_type

    # 层 3：模糊匹配（difflib）
    cn_names = {s.name_cn: s.code for s in _SKILL_REGISTRY.all_skills}
    matches = difflib.get_close_matches(text, cn_names.keys(), n=1, cutoff=0.5)
    if matches:
        return cn_names[matches[0]], DEFAULT_STEPS, model_type

    return "wkF", DEFAULT_STEPS, model_type   # 终极 fallback


# ---------------------------------------------------------------------------
# 帧序列构建
# ---------------------------------------------------------------------------
def _programmatic_frames(gait: str, steps: int) -> list[list[float]]:
    """程序化正弦步态（度），当关键帧数据不存在时使用。"""
    total = steps * 3
    amp_shoulder = 23.0  # deg
    amp_elbow    = 29.0  # deg
    base_shoulder = 50.0
    base_elbow    = 0.0

    bias = 0.0
    if gait == "wkL":
        bias = 8.6
    elif gait == "wkR":
        bias = -8.6

    if gait == "trF":
        amp_shoulder = 31.5
        amp_elbow    = 37.2

    result = []
    for frame_idx in range(total):
        phase = (frame_idx / total) * 2 * np.pi * steps
        sin_a = np.sin(phase)
        sin_b = np.sin(phase + np.pi)
        if gait == "bk":
            sin_a, sin_b = -sin_a, -sin_b
        row = [
            base_shoulder + amp_shoulder * sin_a,
            base_elbow    + amp_elbow    * max(0, sin_a),
            base_shoulder + amp_shoulder * sin_b + bias,
            base_elbow    + amp_elbow    * max(0, sin_b),
            base_shoulder + amp_shoulder * sin_a + bias,
            base_elbow    + amp_elbow    * max(0, sin_a),
            base_shoulder + amp_shoulder * sin_b,
            base_elbow    + amp_elbow    * max(0, sin_b),
        ]
        result.append(row)
    return result


def build_frame_sequence(skill_code: str, steps: int) -> list[list[float]]:
    """返回要播放的关节角序列（度）。"""
    kf = _KEYFRAMES.get(skill_code)
    skill = _SKILL_REGISTRY.get(skill_code)

    if not kf:
        return _programmatic_frames(skill_code, steps)

    frames = kf["frames"]
    skill_type = skill.skill_type if skill else SkillType.GAIT

    if skill_code == "turn_bk":
        # 转身向后走：先播一轮 trL（转身），再拼 bk（后退）
        turn_frames = _KEYFRAMES.get("trL", {}).get("frames", [])[:MAX_TOTAL_FRAMES]
        bk_frames   = _KEYFRAMES.get("bk",  {}).get("frames", [])
        remaining = max(0, MAX_TOTAL_FRAMES - len(turn_frames))
        bk_target = min(steps * 3, remaining)
        if not bk_frames or bk_target == 0:
            return turn_frames
        bk_repeated = (bk_frames * (bk_target // len(bk_frames) + 1))[:bk_target]
        return (turn_frames + bk_repeated)[:MAX_TOTAL_FRAMES]

    if skill_type == SkillType.GAIT:
        target = min(steps * 3, MAX_TOTAL_FRAMES)
        repeated = (frames * (target // len(frames) + 1))[:target]
        return repeated

    elif skill_type == SkillType.POSTURE:
        target_pose = frames[-1]
        interp = []
        for t in range(20):
            a = t / 19
            interp.append([
                INIT_POSE_DEG[j] * (1 - a) + target_pose[j] * a
                for j in range(8)
            ])
        return interp + [target_pose] * 10

    else:  # BEHAVIOR
        return frames[:MAX_TOTAL_FRAMES]


# ---------------------------------------------------------------------------
# RL / MoE 策略推理渲染
# ---------------------------------------------------------------------------

# MoE 方向 → vel_cmd（与 moe_policy.py 保持一致）
_MOE_DIR_TO_VEL_CMD = {
    "forward":  np.array([ 0.25,  0.0], dtype=np.float32),
    "backward": np.array([-0.25,  0.0], dtype=np.float32),
    "left":     np.array([ 0.0,   0.25], dtype=np.float32),
    "right":    np.array([ 0.0,  -0.25], dtype=np.float32),
}
_MOE_VEL_CMD_SCALE = 2.5

_MOE_DIR_CN = {
    "forward": "前进", "backward": "后退", "left": "左行", "right": "右行",
}


def _simulate_moe_to_gif(steps: int, target_name: str = "forward") -> bytes:
    """用 MoE 策略（Route A + Route B）跑推理，返回 GIF 二进制。

    路由规则（从 moe_eval.json 读取）：
        向前 → Route A（BittleGymEnvV2 原始 254-dim obs）
        其余 → Route B（obs 前 248 维共享 + scaled vel_cmd 替换后 6 维）

    moe_eval.json 不存在时自动降级到旧版单模型推理。
    """
    from stable_baselines3 import PPO
    from jprobot.training.env_v2 import BittleGymEnvV2

    trained_dir = Path(__file__).parent.parent / "trained"
    moe_eval_path = trained_dir / "route_a_v3" / "moe_eval.json"

    if not moe_eval_path.exists():
        print(f"[MoE] moe_eval.json 不存在，降级到旧版单模型推理")
        return _simulate_rl_to_gif(steps, "best", target_name)

    with open(moe_eval_path) as f:
        moe_cfg = json.load(f)

    model_a_path = moe_cfg["model_a"]
    model_b_path = moe_cfg["model_b"]
    routing = moe_cfg.get("routing", {})
    use_model_id = routing.get(target_name, "B")

    print(f"[MoE] 加载 Route A: {model_a_path}")
    print(f"[MoE] 加载 Route B: {model_b_path}")
    print(f"[MoE] 方向={target_name}, 使用模型={use_model_id}")

    model_a = PPO.load(model_a_path)
    model_b = PPO.load(model_b_path)
    b_obs_dim = model_b.observation_space.shape[0]

    vel_cmd = _MOE_DIR_TO_VEL_CMD.get(target_name, _MOE_DIR_TO_VEL_CMD["forward"])
    vel_cmd_scaled = np.clip(vel_cmd * _MOE_VEL_CMD_SCALE, -1.0, 1.0)

    env = BittleGymEnvV2(render_mode="rgb_array")
    obs, _ = env.reset(options={"target_name": target_name})

    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0, 0, 0.05],
        distance=0.45, yaw=45, pitch=-30, roll=0, upAxisIndex=2,
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60, aspect=GIF_W / GIF_H, nearVal=0.1, farVal=100,
    )

    dir_cn = _MOE_DIR_CN.get(target_name, target_name)
    label = f"MoE {dir_cn} [{'A' if use_model_id == 'A' else 'B'}]  JPRobot"

    frames_gif = []
    # MoE 不受 MAX_TOTAL_FRAMES 限制：env 在约 251 步后自然终止，实际最多 ~250 帧
    # steps * 6 让用户可以用较小步数生成较短 GIF（每步约 6 个仿真帧）
    n_frames = steps * 6
    try:
        for _ in range(n_frames):
            if use_model_id == "A":
                action, _ = model_a.predict(obs, deterministic=True)
            else:
                # obs 转换：Route A 254-dim → Route B 所需维度
                # 前 248 维（body state + lin_vel + joint_history）两环境物理一致
                # 后 6 维（target_dir[2] + feet_contact[4]）替换为 vel_cmd_scaled + feet_contact
                if b_obs_dim == 254:
                    obs_b = np.concatenate([obs[:248], vel_cmd_scaled, obs[250:254]]).astype(np.float32)
                else:
                    obs_b = np.concatenate([obs[:248], vel_cmd_scaled]).astype(np.float32)
                action, _ = model_b.predict(obs_b, deterministic=True)

            obs, _, terminated, truncated, _ = env.step(action)

            _, _, rgba, _, _ = p.getCameraImage(
                GIF_W, GIF_H, view_matrix, proj_matrix,
                renderer=p.ER_TINY_RENDERER,
                physicsClientId=env.physics_client,
            )
            rgb = np.array(rgba, dtype=np.uint8).reshape(GIF_H, GIF_W, 4)[:, :, :3]

            img = Image.fromarray(rgb)
            draw = ImageDraw.Draw(img)
            draw.text(
                (GIF_W - 8, GIF_H - 8), label,
                fill=(255, 255, 255), stroke_fill=(0, 0, 0),
                stroke_width=2, anchor="rb",
            )
            frames_gif.append(np.array(img))

            if terminated or truncated:
                break
    finally:
        env.close()

    buf = io.BytesIO()
    imageio.mimsave(buf, frames_gif, format="GIF", fps=15, loop=0)
    return buf.getvalue()


def _simulate_rl_to_gif(steps: int, model_type: str = "best", target_name: str = "forward") -> bytes:
    """旧版单模型推理（BittleGymEnv v1，obs=248）。保留作降级备用。"""
    from stable_baselines3 import PPO
    from jprobot.training.env import BittleGymEnv

    model_path = _resolve_rl_model_path(model_type)
    model = PPO.load(str(model_path))
    env = BittleGymEnv(render_mode="rgb_array")
    obs, _ = env.reset(options={"target_name": target_name})

    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0, 0, 0.05],
        distance=0.45, yaw=45, pitch=-30, roll=0, upAxisIndex=2,
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60, aspect=GIF_W / GIF_H, nearVal=0.1, farVal=100,
    )

    frames_gif = []
    n_frames = min(steps * 6, MAX_TOTAL_FRAMES)
    try:
        for _ in range(n_frames):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)

            _, _, rgba, _, _ = p.getCameraImage(
                GIF_W, GIF_H, view_matrix, proj_matrix,
                renderer=p.ER_TINY_RENDERER,
                physicsClientId=env.physics_client,
            )
            rgb = np.array(rgba, dtype=np.uint8).reshape(GIF_H, GIF_W, 4)[:, :, :3]

            img = Image.fromarray(rgb)
            draw = ImageDraw.Draw(img)
            draw.text(
                (GIF_W - 8, GIF_H - 8), "强化学习  JPRobot",
                fill=(255, 255, 255), stroke_fill=(0, 0, 0),
                stroke_width=2, anchor="rb",
            )
            frames_gif.append(np.array(img))

            if terminated or truncated:
                break
    finally:
        env.close()

    buf = io.BytesIO()
    imageio.mimsave(buf, frames_gif, format="GIF", fps=15, loop=0)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 公共帧→GIF 辅助（backflip_rl / humanoid 共用）
# ---------------------------------------------------------------------------
def _frames_to_gif(frames: list, caption: str = "", fps: int = 15) -> bytes:
    """把 RGB numpy 帧列表加水印后编码为 GIF 字节。"""
    pil_frames = []
    for f in frames:
        img = Image.fromarray(f)
        draw = ImageDraw.Draw(img)
        draw.text(
            (img.width - 8, img.height - 8),
            caption,
            fill=(255, 255, 255),
            stroke_fill=(0, 0, 0),
            stroke_width=2,
            anchor="rb",
        )
        pil_frames.append(np.array(img))
    buf = io.BytesIO()
    imageio.mimsave(buf, pil_frames, format="GIF", fps=fps, loop=0)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# RL 后空翻仿真（backflip_v89/v90/v64）
# ---------------------------------------------------------------------------
def _simulate_backflip_to_gif(steps: int = 150) -> bytes:
    """加载最佳后空翻 RL 模型，仿真并返回 GIF 字节。"""
    from stable_baselines3 import PPO
    from jprobot.training.env_backflip import BittleBackflipEnv

    # 按优先级查找可用模型（v89 = confirmed demo model）
    ver_found = None
    for ver in ["backflip_v89", "backflip_v90", "backflip_v64"]:
        model_path = TRAINED_ROOT / ver / "full" / "best.zip"
        if model_path.exists():
            ver_found = ver
            break

    if ver_found is None:
        raise FileNotFoundError("找不到任何 backflip RL 模型（v89/v90/v64）")

    print(f"[backflip_rl] 加载模型: {ver_found}/full/best.zip")
    model = PPO.load(str(TRAINED_ROOT / ver_found / "full" / "best.zip"))
    env = BittleBackflipEnv(render_mode="rgb_array", training_phase="full",
                            disable_rsi=True)
    try:
        obs, _ = env.reset()
        frames = []
        for _ in range(steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            if terminated or truncated:
                break
    finally:
        env.close()

    caption = f"JPRobot RL Backflip ({ver_found})"
    return _frames_to_gif(frames, caption=caption, fps=15)


# ---------------------------------------------------------------------------
# 人形机器人方向解析 + 仿真
# ---------------------------------------------------------------------------
def _parse_humanoid_direction(text: str) -> tuple:
    """从自然语言解析 (vx, vy, wz)。"""
    if any(kw in text for kw in ["向左走", "左移", "侧左"]):   return  0.3,  0.4, 0.0
    if any(kw in text for kw in ["向右走", "右移", "侧右"]):   return  0.3, -0.4, 0.0
    if any(kw in text for kw in ["后退", "向后走", "倒退"]):   return -0.5,  0.0, 0.0
    if any(kw in text for kw in ["左转", "向左转"]):            return  0.3,  0.0, 0.6
    if any(kw in text for kw in ["右转", "向右转"]):            return  0.3,  0.0,-0.6
    return 0.8, 0.0, 0.0  # 默认：向前走


def _humanoid_dir_label(vx: float, vy: float, wz: float) -> str:
    if wz > 0.1:   return "左转"
    if wz < -0.1:  return "右转"
    if vx < 0:     return "后退"
    if vy > 0.1:   return "左移"
    if vy < -0.1:  return "右移"
    return "前进"


def _simulate_humanoid_to_gif(vx: float = 0.8, vy: float = 0.0, wz: float = 0.0,
                               steps: int = 100, mode: str = "walk") -> bytes:
    """在独立子进程中运行人形仿真并返回 GIF 字节。

    macOS 上 MuJoCo rgb_array 渲染需要主线程（OpenGL/Metal 限制），
    HTTP server 的 worker 线程中会卡死。通过 subprocess 绕过此限制：
    子进程是独立主线程，可正常调用 env.render()。

    mode: "walk"（默认，万向行走模型）| "run"（跑步模型，高速+腾空相）
    """
    import subprocess
    worker = Path(__file__).parent / "humanoid_sim_worker.py"
    print(f"[humanoid] 启动子进程仿真  vx={vx} vy={vy} wz={wz} steps={steps} mode={mode}")
    result = subprocess.run(
        [sys.executable, str(worker), str(vx), str(vy), str(wz), str(steps), mode],
        capture_output=True,
        timeout=120,
        cwd=str(Path(__file__).parent.parent),
    )
    if result.returncode != 0:
        err = result.stderr.decode(errors="replace")
        raise RuntimeError(f"humanoid worker 失败: {err[-500:]}")
    return result.stdout


# ---------------------------------------------------------------------------
# 仿真 & GIF 渲染
# ---------------------------------------------------------------------------
def simulate_to_gif(command: str) -> bytes:
    """运行 PyBullet DIRECT 仿真，返回 GIF 二进制。"""
    with SIM_LOCK:
        # ── 人形机器人路由（优先检测，避免被步态解析器截获）──────────────────
        if any(kw in command for kw in ["人形", "人型", "humanoid"]):
            # 跑步：专用高速模型（腾空相，vx=1.5m/s）
            if any(kw in command for kw in ["跑步", "奔跑", "跑起来", "快跑", "跑", "run"]):
                return _simulate_humanoid_to_gif(1.5, 0.0, 0.0, mode="run")
            vx, vy, wz = _parse_humanoid_direction(command)
            return _simulate_humanoid_to_gif(vx, vy, wz)

        # ── RL 后空翻路由（区别于固件 bf）────────────────────────────────────
        if any(kw in command for kw in ["强化后空翻", "RL后空翻", "强化翻", "rl后空翻"]):
            return _simulate_backflip_to_gif()

        skill_code, steps, model_type = parse_command(command)

        # MoE / RL 策略走单独渲染路径
        if skill_code == "rl":
            # 从命令文本中解析目标方向
            if "向后" in command or "后退" in command:
                rl_dir = "backward"
            elif "向左" in command:
                rl_dir = "left"
            elif "向右" in command:
                rl_dir = "right"
            else:
                rl_dir = "forward"  # 默认向前
            # 优先使用 MoE 双模型推理（降级到旧版单模型若 moe_eval.json 不存在）
            return _simulate_moe_to_gif(steps, target_name=rl_dir)

        frame_sequence_deg = build_frame_sequence(skill_code, steps)

        physics_client = p.connect(p.DIRECT)
        try:
            p.setGravity(0, 0, -9.81, physicsClientId=physics_client)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.loadURDF("plane.urdf", physicsClientId=physics_client)

            urdf_path = Path(__file__).parent.parent / "jprobot" / "models" / "bittle_esp32.urdf"
            robot_id = p.loadURDF(
                str(urdf_path),
                [0, 0, 0.08],
                [0, 0, 0, 1],
                flags=p.URDF_USE_SELF_COLLISION,
                physicsClientId=physics_client,
            )

            # 动态发现关节（避免硬编码 Bug）
            joint_ids = []
            for j in range(p.getNumJoints(robot_id, physicsClientId=physics_client)):
                info = p.getJointInfo(robot_id, j, physicsClientId=physics_client)
                if info[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                    joint_ids.append(j)
                    p.changeDynamics(
                        robot_id, j,
                        maxJointVelocity=np.pi * 10,
                        physicsClientId=physics_client,
                    )

            # 设置初始姿态
            init_rad = np.deg2rad(np.array(INIT_POSE_DEG, dtype=float))
            for i, jid in enumerate(joint_ids[:8]):
                p.resetJointState(robot_id, jid, init_rad[i],
                                physicsClientId=physics_client)

            # 稳定初始姿态
            for _ in range(30):
                p.stepSimulation(physicsClientId=physics_client)

            # 按技能类型选配置
            skill = _SKILL_REGISTRY.get(skill_code)
            label = f"{skill.name_cn}  JPRobot" if skill else "JPRobot"
            skill_type = skill.skill_type if skill else SkillType.GAIT
            kinematic = JOINT_CONTROL_MODE.get(skill_type, "physics") == "kinematic"

            # 力控制：固件关键帧用大力（角度跳跃幅度大），程序化正弦用小力（平滑变化）
            has_firmware = _KEYFRAMES.get(skill_code) is not None
            joint_force = PHYSICS_JOINT_FORCE if has_firmware else PHYSICS_JOINT_FORCE_PROGRAMMATIC

            # 相机参数（固定视角）
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0, 0, 0.05],
                distance=0.45,
                yaw=45,
                pitch=-30,
                roll=0,
                upAxisIndex=2,
            )
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60,
                aspect=GIF_W / GIF_H,
                nearVal=0.1,
                farVal=100,
            )

            frames_gif = []
            for frame_angles_deg in frame_sequence_deg:
                target_rad = np.deg2rad(np.array(frame_angles_deg, dtype=float))
                for i, jid in enumerate(joint_ids[:8]):
                    if kinematic:
                        # 运动学模式：直接强制设置角度，跳过物理约束
                        # 适合 BEHAVIOR（翻转等），保证固件快照完整还原
                        p.resetJointState(
                            robot_id, jid, target_rad[i],
                            physicsClientId=physics_client,
                        )
                    else:
                        # 物理模式：电机驱动到目标角度，受力/惯性/重力约束
                        # 适合 GAIT/POSTURE，仿真更真实，与真机串口指令行为一致
                        p.setJointMotorControl2(
                            robot_id, jid, p.POSITION_CONTROL,
                            targetPosition=target_rad[i], force=joint_force,
                            physicsClientId=physics_client,
                        )
                # 每帧跑多步物理仿真，让关节有足够时间到达目标角度
                for _ in range(PHYSICS_STEPS_PER_FRAME):
                    p.stepSimulation(physicsClientId=physics_client)

                # 渲染
                _, _, rgba, _, _ = p.getCameraImage(
                    GIF_W, GIF_H,
                    view_matrix, proj_matrix,
                    renderer=p.ER_TINY_RENDERER,
                    physicsClientId=physics_client,
                )
                rgb = np.array(rgba, dtype=np.uint8).reshape(GIF_H, GIF_W, 4)[:, :, :3]

                # 水印（右下角）
                img = Image.fromarray(rgb)
                draw = ImageDraw.Draw(img)
                draw.text(
                    (GIF_W - 8, GIF_H - 8),
                    label,
                    fill=(255, 255, 255),
                    stroke_fill=(0, 0, 0),
                    stroke_width=2,
                    anchor="rb",
                )
                frames_gif.append(np.array(img))

        finally:
            p.disconnect(physics_client)

        # 生成 GIF
        buf = io.BytesIO()
        imageio.mimsave(buf, frames_gif, format="GIF", fps=15, loop=0)
        return buf.getvalue()


# ---------------------------------------------------------------------------
# HTTP 服务器
# ---------------------------------------------------------------------------
class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class ControlHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path == "/health":
            self._json({"status": "ok"})
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/simulate":
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            try:
                body = json.loads(raw)
            except json.JSONDecodeError:
                self._json({"error": "invalid json"}, status=400)
                return
            command = body.get("command", "").strip()
            if not command:
                self._json({"error": "command is required"}, status=400)
                return
            started = time.time()
            skill_code, steps, model_type = parse_command(command)
            try:
                gif_bytes = simulate_to_gif(command)
                elapsed_ms = int((time.time() - started) * 1000)
                print(
                    f"[simulate] cmd={command!r} skill={skill_code} steps={steps} model={model_type} "
                    f"elapsed_ms={elapsed_ms} size={len(gif_bytes)}B"
                )
                self.send_response(200)
                self.send_header("Content-Type", "image/gif")
                self.send_header("Content-Length", str(len(gif_bytes)))
                self.end_headers()
                self.wfile.write(gif_bytes)
            except Exception as e:
                print(
                    f"[simulate][error] cmd={command!r} skill={skill_code} steps={steps} model={model_type} "
                    f"err={type(e).__name__}: {e}"
                )
                self._json({"error": str(e)}, status=500)
        else:
            self.send_response(404)
            self.end_headers()

    def _json(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass  # 静默请求日志


def main():
    parser = argparse.ArgumentParser(description="JPRobot Control Server")
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    except ImportError:
        pass
    parser.add_argument("--port", type=int,
                        default=int(os.getenv("JPROBOT_CONTROL_PORT", "18792")))
    args = parser.parse_args()

    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer(("0.0.0.0", args.port), ControlHandler)
    print(f"JPRobot Control Server")
    print(f"  Simulate: http://127.0.0.1:{args.port}/simulate")
    print(f"  Health:   http://127.0.0.1:{args.port}/health")
    print(f"  Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
