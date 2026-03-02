#!/usr/bin/env python3
"""人形机器人速度命令跟随评估 — 万向行走质量检验。

## 评估什么

训练产出 trained/humanoid_velocity/best.zip 之后，我们知道 ep_len 高（不摔），
但不知道"速度命令是否真正被跟随"。本脚本回答这个问题。

## 输出指标

  vel_error_mean — 每步 ||v_actual - v_cmd|| 的均值（越小越好，目标 <0.5）
  vel_reward_mean — 每步速度追踪奖励均值（越高越好，满分 W_VEL）
  按命令方向分类：前进 / 后退 / 左移 / 右移 / 左转 / 右转

## 速度读取方式

MuJoCo free joint（根节点 6DoF）的 qvel：
  [0]: vx — 躯干在全局 X 方向的线速度（m/s），正值=前进
  [1]: vy — 躯干在全局 Y 方向的线速度（m/s），正值=左移
  [5]: wz — 绕 Z 轴的角速度（rad/s），正值=左转

## 用法

  python scripts/eval_humanoid_velocity.py               # 默认 20 局
  python scripts/eval_humanoid_velocity.py --episodes 5  # 快速验证（约 1 分钟）
  python scripts/eval_humanoid_velocity.py --model trained/humanoid_velocity/best.zip
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import SAC
from jprobot.training.env_humanoid_velocity import HumanoidVelocityEnv


DEFAULT_MODEL = Path(__file__).parent.parent / "trained" / "humanoid_velocity" / "best.zip"


def classify_cmd(vx: float, vy: float, wz: float) -> str:
    """将速度命令分类为语义方向，用于分组统计。"""
    # 以绝对值最大的分量为主方向
    magnitudes = {
        "前进": max(0.0, vx),
        "后退": max(0.0, -vx),
        "左移": max(0.0, vy),
        "右移": max(0.0, -vy),
        "左转": max(0.0, wz),
        "右转": max(0.0, -wz),
    }
    dominant = max(magnitudes, key=magnitudes.get)
    # 如果所有分量都很小（接近零命令），标记为"静止"
    if max(magnitudes.values()) < 0.1:
        return "静止"
    return dominant


def evaluate(model_path: str, n_episodes: int) -> None:
    print(f"{'='*65}")
    print(f"人形机器人速度命令跟随评估")
    print(f"  模型: {model_path}")
    print(f"  局数: {n_episodes}")
    print()
    print(f"  【指标说明】")
    print(f"    vel_error  — 每步速度误差 ||v_actual-v_cmd||（越小越好，目标<0.5）")
    print(f"    vel_reward — 每步速度追踪奖励（满分 W_VEL，越高越好）")
    print(f"    ep_len     — 每局存活步数（满分 1000）")
    print(f"    ep_rew     — 每局累计奖励")
    print(f"{'='*65}")

    if not Path(model_path).exists():
        print(f"[错误] 模型文件不存在: {model_path}")
        return

    print("加载模型和环境...")
    env = HumanoidVelocityEnv()
    model = SAC.load(model_path)
    print("加载完成，开始评估...\n")

    # 按方向分组收集数据
    direction_stats: dict[str, list] = {}

    all_vel_errors = []
    all_vel_rewards = []
    all_ep_lens = []
    all_ep_rews = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        cmd_vx = env.cmd_vx
        cmd_vy = env.cmd_vy
        cmd_wz = env.cmd_wz
        direction = classify_cmd(cmd_vx, cmd_vy, cmd_wz)

        ep_vel_errors = []
        ep_vel_rewards = []
        ep_reward = 0.0
        step = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            ep_reward += float(reward)

            vel_error = float(info.get("vel_error", 0.0))
            vel_reward = float(info.get("vel_reward", 0.0))
            ep_vel_errors.append(vel_error)
            ep_vel_rewards.append(vel_reward)

            if terminated or truncated:
                break

        ep_vel_error_mean = float(np.mean(ep_vel_errors))
        ep_vel_reward_mean = float(np.mean(ep_vel_rewards))

        print(
            f"  Ep {ep+1:3d} | {direction:4s} "
            f"cmd=({cmd_vx:+.2f},{cmd_vy:+.2f},{cmd_wz:+.2f}) "
            f"| ep_len={step:4d} | ep_rew={ep_reward:7.1f} "
            f"| vel_err={ep_vel_error_mean:.3f} | vel_rew={ep_vel_reward_mean:.3f}"
        )

        all_vel_errors.append(ep_vel_error_mean)
        all_vel_rewards.append(ep_vel_reward_mean)
        all_ep_lens.append(step)
        all_ep_rews.append(ep_reward)

        if direction not in direction_stats:
            direction_stats[direction] = []
        direction_stats[direction].append({
            "vel_error": ep_vel_error_mean,
            "vel_reward": ep_vel_reward_mean,
            "ep_len": step,
            "ep_rew": ep_reward,
        })

    env.close()

    # ── 汇总报告 ──────────────────────────────────────────────────────────
    print()
    print(f"{'='*65}")
    print(f"[总体结果] {n_episodes} 局均值")
    print(f"  ep_len     = {np.mean(all_ep_lens):.1f}   (满分 1000)")
    print(f"  ep_rew     = {np.mean(all_ep_rews):.1f}")
    print(f"  vel_error  = {np.mean(all_vel_errors):.4f}  (越小越好，<0.5 优秀)")
    print(f"  vel_reward = {np.mean(all_vel_rewards):.4f}  (满分 W_VEL)")
    print()

    # 按方向分组报告
    if len(direction_stats) > 1:
        print(f"[分方向结果]")
        for dir_name, stats in sorted(direction_stats.items()):
            n = len(stats)
            avg_err = np.mean([s["vel_error"] for s in stats])
            avg_rew = np.mean([s["vel_reward"] for s in stats])
            avg_len = np.mean([s["ep_len"] for s in stats])
            print(
                f"  {dir_name:4s} ({n:2d}局): "
                f"vel_err={avg_err:.4f}  vel_rew={avg_rew:.4f}  ep_len={avg_len:.0f}"
            )
        print()

    # 质量判断
    overall_err = float(np.mean(all_vel_errors))
    overall_rew = float(np.mean(all_vel_rewards))
    print(f"[质量判断]")
    if overall_err < 0.3:
        print(f"  速度跟随：✅ 优秀（vel_error={overall_err:.3f} < 0.3）")
    elif overall_err < 0.7:
        print(f"  速度跟随：⚠️  中等（vel_error={overall_err:.3f}，目标 <0.3）")
    else:
        print(f"  速度跟随：❌ 较差（vel_error={overall_err:.3f}，建议继续训练）")

    if overall_rew > 2.5:
        print(f"  追踪奖励：✅ 优秀（vel_reward={overall_rew:.3f} > 2.5，接近满分 W_VEL）")
    elif overall_rew > 1.5:
        print(f"  追踪奖励：⚠️  中等（vel_reward={overall_rew:.3f}，满分 W_VEL）")
    else:
        print(f"  追踪奖励：❌ 较差（vel_reward={overall_rew:.3f}）")

    if float(np.mean(all_ep_lens)) > 800:
        print(f"  存活能力：✅ 优秀（ep_len={np.mean(all_ep_lens):.0f} > 800）")
    elif float(np.mean(all_ep_lens)) > 400:
        print(f"  存活能力：⚠️  中等（ep_len={np.mean(all_ep_lens):.0f}）")
    else:
        print(f"  存活能力：❌ 差（ep_len={np.mean(all_ep_lens):.0f}，频繁摔倒）")

    print()
    print(f"【名词备注】")
    print(f"  vel_error  — velocity error，速度误差。计算公式：(vx-vx_cmd)²+(vy-vy_cmd)²+0.5*(wz-wz_cmd)²")
    print(f"               why: 衡量机器人实际速度与指令速度的差距，是万向行走质量的核心指标")
    print(f"               how: 越小表示跟随越精确；0=完美跟随，>1=基本不跟随")
    print(f"  vel_reward — 速度追踪奖励。公式：W_VEL*exp(-vel_error/SIGMA)，当前 W_VEL=5.0, SIGMA=1.0")
    print(f"               why: 和训练时的奖励函数相同，高 vel_reward 代表策略真正在跟随命令")
    print(f"               how: 满分 W_VEL（误差=0），实际 >2.5 属于优秀")
    print(f"  ep_len     — episode length，每局存活步数。Humanoid 每步 0.005s，满步 1000=5秒")
    print(f"               why: 不摔倒是跟随命令的前提，ep_len 低说明策略还不稳定")
    print(f"  ep_rew     — episode reward，每局累计奖励，含速度追踪+存活+健康")
    print(f"  vx/vy/wz   — 躯干速度：vx=前后(m/s)，vy=左右(m/s)，wz=转弯角速度(rad/s)")
    print(f"{'='*65}")


def main() -> None:
    parser = argparse.ArgumentParser(description="人形机器人速度命令跟随评估（万向行走）")
    parser.add_argument(
        "--model", type=str,
        default=str(DEFAULT_MODEL),
        help=f"模型路径（默认 {DEFAULT_MODEL}）",
    )
    parser.add_argument(
        "--episodes", type=int, default=20,
        help="评估局数（默认 20，5 局约 1 分钟快速验证）",
    )
    args = parser.parse_args()
    evaluate(args.model, args.episodes)


if __name__ == "__main__":
    main()
