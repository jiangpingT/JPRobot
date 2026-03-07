#!/usr/bin/env python3
"""人形机器人跑步可视化 — 带速度命令和腾空相实时显示。

用法：
    python scripts/viz_humanoid_run.py                          # 默认模型，3 局
    python scripts/viz_humanoid_run.py --episodes 5 --speed 0.5 # 慢放看步态
    python scripts/viz_humanoid_run.py --model trained/humanoid_run/best.zip
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

sys.path.insert(0, str(Path(__file__).parent.parent))
from jprobot.training.env_humanoid_run import HumanoidRunEnv

DEFAULT_MODEL = "trained/humanoid_run/best.zip"


def run_viz(model_path: str, n_episodes: int, speed: float):
    print(f"\n{'='*60}")
    print(f"人形机器人跑步可视化")
    print(f"  模型: {model_path}")
    print(f"  局数: {n_episodes}  速度倍率: {speed}x")
    print(f"  绿色条 = 腾空相（双脚离地）")
    print(f"  关闭渲染窗口可退出")
    print(f"{'='*60}\n")

    env = HumanoidRunEnv(render_mode="human")
    model = SAC.load(model_path, device="cpu")

    for ep in range(n_episodes):
        obs, _ = env.reset()
        cmd_vx = env.cmd_vx
        cmd_vy = env.cmd_vy
        cmd_wz = env.cmd_wz
        print(f"第 {ep+1}/{n_episodes} 局  速度命令: vx={cmd_vx:+.2f} vy={cmd_vy:+.2f} wz={cmd_wz:+.2f}")

        done = False
        steps = 0
        total_rew = 0.0
        airborne_steps = 0
        vx_list = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, terminated, truncated, info = env.step(action)

            steps += 1
            total_rew += rew
            if info.get("airborne", False):
                airborne_steps += 1
            vx_actual = info.get("vel_actual", (0.0,))[0]
            vx_list.append(float(vx_actual))
            done = terminated or truncated

            if speed < 2.0:
                time.sleep(max(0, 0.003 / speed))

        airborne_pct = 100.0 * airborne_steps / max(1, steps)
        print(
            f"  步数={steps}  奖励={total_rew:.0f}  "
            f"airborne={airborne_pct:.1f}%  "
            f"vx均值={np.mean(vx_list):+.2f}m/s"
        )

    env.close()
    print("\n可视化结束。")


def main():
    parser = argparse.ArgumentParser(description="人形机器人跑步可视化")
    parser.add_argument("--model",    type=str,   default=DEFAULT_MODEL)
    parser.add_argument("--episodes", type=int,   default=3)
    parser.add_argument("--speed",    type=float, default=1.0,
                        help="播放速度倍率（0.5=慢放，2.0=快放）")
    args = parser.parse_args()
    run_viz(args.model, args.episodes, args.speed)


if __name__ == "__main__":
    main()
