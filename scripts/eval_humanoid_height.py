#!/usr/bin/env python3
"""人形机器人躯干高度评估脚本

评估指定模型在行走过程中的实际躯干 z 坐标（站高），
统计均值、最小值、最大值及分布，判断是否真正接近直立（z≈1.3m）。

用法：
    python scripts/eval_humanoid_height.py                          # 默认评估 C2
    python scripts/eval_humanoid_height.py --model trained/humanoid_upright/B2/best.zip
    python scripts/eval_humanoid_height.py --model trained/humanoid_sac/best.zip --episodes 5
"""

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC

sys.path.insert(0, str(Path(__file__).parent.parent))

DEFAULT_MODEL = "trained/humanoid_upright/C2/best.zip"


def eval_height(model_path: str, n_episodes: int, healthy_z_min: float):
    print(f"\n{'='*60}")
    print(f"人形机器人躯干高度评估")
    print(f"  模型: {model_path}")
    print(f"  局数: {n_episodes}")
    print(f"  环境 healthy_z_min: {healthy_z_min}")
    print(f"{'='*60}\n")

    env = gym.make("Humanoid-v4", healthy_z_range=(healthy_z_min, 2.1))
    model = SAC.load(model_path, device="cpu")

    all_z       = []   # 所有步的 z 值
    ep_z_means  = []   # 每局平均 z
    ep_z_mins   = []   # 每局最低 z
    ep_lens     = []   # 每局步数
    ep_rews     = []   # 每局奖励（原始，不含软约束）

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_z = []
        total_rew = 0.0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, terminated, truncated, _ = env.step(action)
            torso_z = float(obs[0])   # Humanoid-v4 obs[0] = 躯干 z 坐标
            ep_z.append(torso_z)
            total_rew += rew
            steps += 1
            done = terminated or truncated

        ep_mean_z = float(np.mean(ep_z))
        ep_min_z  = float(np.min(ep_z))
        ep_z_means.append(ep_mean_z)
        ep_z_mins.append(ep_min_z)
        ep_lens.append(steps)
        ep_rews.append(total_rew)
        all_z.extend(ep_z)

        print(f"  局 {ep+1:2d}/{n_episodes}  步数={steps:4d}  "
              f"z均值={ep_mean_z:.3f}m  z最低={ep_min_z:.3f}m  "
              f"奖励={total_rew:.0f}")

    env.close()

    # ── 汇总统计 ──────────────────────────────────────────────
    all_z = np.array(all_z)
    print(f"\n{'='*60}")
    print(f"【躯干高度汇总】")
    print(f"  总采样步数   : {len(all_z)}")
    print(f"  全程平均 z   : {np.mean(all_z):.4f} m")
    print(f"  全程中位数 z : {np.median(all_z):.4f} m")
    print(f"  全程最低 z   : {np.min(all_z):.4f} m")
    print(f"  全程最高 z   : {np.max(all_z):.4f} m")
    print(f"  标准差       : {np.std(all_z):.4f} m")

    # 各高度区间占比
    print(f"\n【高度分布】")
    brackets = [
        (0.0,  1.1,  "z < 1.1m  （危险区）"),
        (1.1,  1.2,  "1.1 ≤ z < 1.2m（A阶段水平）"),
        (1.2,  1.25, "1.2 ≤ z < 1.25m（B阶段水平）"),
        (1.25, 1.3,  "1.25 ≤ z < 1.3m（B2阶段水平）"),
        (1.3,  1.4,  "1.3 ≤ z < 1.4m（目标直立区✅）"),
        (1.4,  9.9,  "z ≥ 1.4m  （充分直立✅✅）"),
    ]
    for lo, hi, label in brackets:
        count = np.sum((all_z >= lo) & (all_z < hi))
        pct   = count / len(all_z) * 100
        bar   = "█" * int(pct / 2)
        print(f"  {label:30s}  {pct:5.1f}%  {bar}")

    print(f"\n【每局平均】")
    print(f"  平均步数   : {np.mean(ep_lens):.1f}")
    print(f"  平均奖励   : {np.mean(ep_rews):.1f}")
    print(f"  平均 z 均值: {np.mean(ep_z_means):.4f} m")
    print(f"  平均 z 最低: {np.mean(ep_z_mins):.4f} m")

    # 直立达标判断
    pct_above_13 = np.sum(all_z >= 1.3) / len(all_z) * 100
    pct_above_12 = np.sum(all_z >= 1.2) / len(all_z) * 100
    print(f"\n【直立达标】")
    print(f"  z ≥ 1.2m 的时间占比: {pct_above_12:.1f}%")
    print(f"  z ≥ 1.3m 的时间占比: {pct_above_13:.1f}%")
    if pct_above_13 >= 50:
        print(f"  ✅ 超过半数时间保持真正直立（z≥1.3m）！")
    elif pct_above_13 >= 20:
        print(f"  🔶 部分时间达到直立，还需继续训练")
    else:
        print(f"  ❌ 大多数时间仍低于 1.3m，直立训练效果有限")

    print(f"{'='*60}\n")

    print("【名词备注】")
    print("  torso_z    — 躯干 z 坐标 / 机器人从地面到躯干中心的高度（米）")
    print("  obs[0]     — Humanoid-v4 第0维观测值，即躯干 z")
    print("  healthy_z  — 健康高度范围门槛 / z 低于此值 episode 终止")
    print("  直立参考    — 完全站直时 z ≈ 1.3-1.4m；v1 弯腰走 z ≈ 1.17m")


def main():
    parser = argparse.ArgumentParser(description="人形机器人躯干高度评估")
    parser.add_argument("--model",    type=str, default=DEFAULT_MODEL)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--healthy-z", type=float, default=1.0,
                        help="评估时环境的 healthy_z_min（不影响模型，只影响episode终止）")
    args = parser.parse_args()
    eval_height(args.model, args.episodes, args.healthy_z)


if __name__ == "__main__":
    main()
