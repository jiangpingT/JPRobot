#!/usr/bin/env python3
"""人形机器人可视化脚本 — 打开 MuJoCo 渲染窗口看机器人实际行走效果。

用法：
    python scripts/viz_humanoid.py                                    # 默认看 C2
    python scripts/viz_humanoid.py --model trained/humanoid_sac/best.zip  # 对比 v1
    python scripts/viz_humanoid.py --model trained/humanoid_upright/B2/best.zip
    python scripts/viz_humanoid.py --episodes 3 --speed 1.0          # 正常速度
"""

import argparse
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC

sys.path.insert(0, str(Path(__file__).parent.parent))

DEFAULT_MODEL = "trained/humanoid_upright/C2/best.zip"


def run_viz(model_path: str, n_episodes: int, speed: float):
    print(f"\n{'='*60}")
    print(f"人形机器人可视化")
    print(f"  模型: {model_path}")
    print(f"  局数: {n_episodes}  速度倍率: {speed}x")
    print(f"  关闭渲染窗口可退出")
    print(f"{'='*60}\n")

    env = gym.make("Humanoid-v4", render_mode="human")
    model = SAC.load(model_path, device="cpu")

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        z_list = []
        total_rew = 0.0

        print(f"第 {ep+1}/{n_episodes} 局开始...")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, terminated, truncated, _ = env.step(action)
            env.render()

            torso_z = float(obs[0])
            z_list.append(torso_z)
            total_rew += rew
            steps += 1
            done = terminated or truncated

            # 控制播放速度（speed<1 慢放，speed>1 快放）
            if speed < 2.0:
                time.sleep(max(0, 0.002 / speed))

        print(f"  步数={steps}  平均z={np.mean(z_list):.3f}m  "
              f"最低z={np.min(z_list):.3f}m  奖励={total_rew:.0f}")

    env.close()
    print("\n可视化结束。")


def main():
    parser = argparse.ArgumentParser(description="人形机器人行走可视化")
    parser.add_argument("--model",    type=str,   default=DEFAULT_MODEL)
    parser.add_argument("--episodes", type=int,   default=5)
    parser.add_argument("--speed",    type=float, default=1.0,
                        help="播放速度倍率（0.5=慢放，2.0=快放）")
    args = parser.parse_args()
    run_viz(args.model, args.episodes, args.speed)


if __name__ == "__main__":
    main()
