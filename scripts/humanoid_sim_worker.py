#!/usr/bin/env python3
"""人形仿真 worker — 独立进程运行。

macOS 上 MuJoCo rgb_array 渲染需要主线程，在 HTTP server worker 线程中会卡死。
此脚本作为独立进程被 control_server.py 调用，是自己进程的主线程，可正常渲染。

用法：python humanoid_sim_worker.py <vx> <vy> <wz> <steps> [mode]
  mode: walk（默认）| run（跑步模型，高速，腾空相）
输出：GIF 字节写入 stdout（二进制）
"""

import io
import sys
from pathlib import Path

import imageio
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import SAC
from jprobot.training.env_humanoid_velocity import HumanoidVelocityEnv
from jprobot.training.env_humanoid_run import HumanoidRunEnv


def main():
    if len(sys.argv) < 5:
        sys.exit("Usage: humanoid_sim_worker.py <vx> <vy> <wz> <steps> [mode]")

    vx    = float(sys.argv[1])
    vy    = float(sys.argv[2])
    wz    = float(sys.argv[3])
    steps = int(sys.argv[4])
    mode  = sys.argv[5] if len(sys.argv) > 5 else "walk"

    if mode == "run":
        model_path = Path(__file__).parent.parent / "trained" / "humanoid_run" / "best.zip"
        model = SAC.load(str(model_path))
        env = HumanoidRunEnv(render_mode="rgb_array")
    else:
        model_path = Path(__file__).parent.parent / "trained" / "humanoid_velocity" / "best.zip"
        model = SAC.load(str(model_path))
        env = HumanoidVelocityEnv(render_mode="rgb_array")

    try:
        obs, _ = env.reset()
        env.cmd_vx, env.cmd_vy, env.cmd_wz = vx, vy, wz
        base_obs = obs[:376]
        obs = env._aug_obs(base_obs)

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

    if mode == "run":
        label = "跑步"
    elif wz > 0.1:   label = "左转"
    elif wz < -0.1: label = "右转"
    elif vx < 0:   label = "后退"
    elif vy > 0.1: label = "左移"
    elif vy < -0.1: label = "右移"
    else:           label = "前进"

    # 降分辨率：480×480 → 320×240（缩小文件体积，Telegram 友好）
    TARGET_W, TARGET_H = 320, 240
    pil_frames = []
    for f in frames:
        img = Image.fromarray(f).resize((TARGET_W, TARGET_H), Image.LANCZOS)
        draw = ImageDraw.Draw(img)
        draw.text(
            (TARGET_W - 8, TARGET_H - 8),
            f"JPRobot Humanoid {label}",
            fill=(255, 255, 255), stroke_fill=(0, 0, 0),
            stroke_width=2, anchor="rb",
        )
        pil_frames.append(np.array(img))

    buf = io.BytesIO()
    imageio.mimsave(buf, pil_frames, format="GIF", fps=15, loop=0)
    sys.stdout.buffer.write(buf.getvalue())


if __name__ == "__main__":
    main()
