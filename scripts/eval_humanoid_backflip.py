#!/usr/bin/env python3
"""人形机器人后空翻评估脚本。

核心指标：
  success%      — 后空翻成功率（backward_rot≥286°且落地）
  rotation_deg  — 平均累积旋转度数（空中峰值）
  rot@land_deg  — 落地瞬间累积旋转度数（核心指标，目标 360°）
  ep_len        — 每局步数（正常 50-80，超时 200 表示失败）
  ep_rew        — 每局总奖励

用法：
  python scripts/eval_humanoid_backflip.py                     # 默认 20 局
  python scripts/eval_humanoid_backflip.py --episodes 5        # 快速验证
  python scripts/eval_humanoid_backflip.py --model trained/humanoid_backflip/best.zip
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import SAC
from jprobot.training.env_humanoid_backflip import HumanoidBackflipEnv


DEFAULT_MODEL = Path(__file__).parent.parent / "trained" / "humanoid_backflip" / "best.zip"


def evaluate(model_path: str, n_episodes: int) -> None:
    print(f"{'='*65}")
    print(f"人形机器人后空翻评估")
    print(f"  模型: {model_path}")
    print(f"  局数: {n_episodes}")
    print()
    print(f"  【指标说明】")
    print(f"    success%     — 后空翻成功率（rotation≥286°且落地）")
    print(f"    rotation     — 空中累积旋转度数（正值=向后翻）")
    print(f"    rot@land     — 落地瞬间旋转度数（核心，目标360°）")
    print(f"    ep_len       — 每局步数（满分情况约50-80步）")
    print(f"    ep_rew       — 每局总奖励")
    print(f"{'='*65}")

    if not Path(model_path).exists():
        print(f"[错误] 模型文件不存在: {model_path}")
        return

    print("加载模型和环境...")
    env = HumanoidBackflipEnv()
    model = SAC.load(model_path, env=env)
    print("加载完成，开始评估...\n")

    all_success      = []
    all_rotation     = []
    all_rot_at_land  = []
    all_ep_lens      = []
    all_ep_rews      = []

    for ep in range(n_episodes):
        obs, info = env.reset()

        ep_reward    = 0.0
        step         = 0
        rot_at_land  = 0.0
        max_rotation = 0.0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            ep_reward    += float(reward)
            max_rotation = max(max_rotation, info.get("backward_rot_deg", 0.0))

            # 记录落地瞬间的旋转角度
            if info.get("just_landed", False):
                rot_at_land = info.get("backward_rot_deg", 0.0)

            if terminated or truncated:
                break

        success = info.get("success", False)

        print(
            f"  Ep {ep+1:3d} | "
            f"{'✅' if success else '❌'} "
            f"rot={max_rotation:5.1f}°  "
            f"rot@land={rot_at_land:5.1f}°  "
            f"ep_len={step:3d}  "
            f"ep_rew={ep_reward:7.1f}"
        )

        all_success.append(1 if success else 0)
        all_rotation.append(max_rotation)
        all_rot_at_land.append(rot_at_land)
        all_ep_lens.append(step)
        all_ep_rews.append(ep_reward)

    env.close()

    # ── 汇总报告 ──────────────────────────────────────────────────────────────
    success_pct   = float(np.mean(all_success)) * 100.0
    mean_rotation = float(np.mean(all_rotation))
    mean_rot_land = float(np.mean(all_rot_at_land))
    mean_len      = float(np.mean(all_ep_lens))
    mean_rew      = float(np.mean(all_ep_rews))

    print()
    print(f"{'='*65}")
    print(f"[总体结果] {n_episodes} 局均值")
    print(f"  success%    = {success_pct:.1f}%   (目标: 100%)")
    print(f"  rotation    = {mean_rotation:.1f}°   (空中峰值，目标: ≥360°)")
    print(f"  rot@land    = {mean_rot_land:.1f}°   (落地时，核心指标，目标: 360°)")
    print(f"  ep_len      = {mean_len:.1f}   (目标: 50-80)")
    print(f"  ep_rew      = {mean_rew:.1f}")
    print()

    print(f"[阶段判断]")
    # 成功率判断
    if success_pct >= 90:
        print(f"  成功率：✅ 优秀（{success_pct:.0f}% ≥ 90%）")
    elif success_pct >= 50:
        print(f"  成功率：⚠️  中等（{success_pct:.0f}%，需继续训练）")
    elif success_pct >= 10:
        print(f"  成功率：⚠️  初现（{success_pct:.0f}%，旋转策略正在形成）")
    else:
        print(f"  成功率：❌ 未达标（{success_pct:.0f}%，<10%）")

    # 旋转量判断
    if mean_rotation >= 355:
        print(f"  旋转量：✅ 接近完美（{mean_rotation:.1f}° ≥ 355°）")
    elif mean_rotation >= 286:
        print(f"  旋转量：✅ 达标（{mean_rotation:.1f}° ≥ 286°，后空翻完成）")
    elif mean_rotation >= 180:
        print(f"  旋转量：⚠️  中段（{mean_rotation:.1f}°，已过半，继续推进）")
    elif mean_rotation >= 90:
        print(f"  旋转量：⚠️  初段（{mean_rotation:.1f}°，已腾空旋转）")
    else:
        print(f"  旋转量：❌ 起步（{mean_rotation:.1f}°，旋转未形成）")

    # 落地精确度判断
    if mean_rot_land >= 355:
        print(f"  落地精度：✅ 接近完美（rot@land={mean_rot_land:.1f}°）")
    elif mean_rot_land >= 340:
        print(f"  落地精度：⚠️  良好（rot@land={mean_rot_land:.1f}°，最后几度需微调）")
    elif mean_rot_land > 0:
        print(f"  落地精度：⚠️  有待提升（rot@land={mean_rot_land:.1f}°）")
    else:
        print(f"  落地精度：❌ 未落地（rot@land=0，还未成功落地）")

    print()
    print(f"【名词备注】")
    print(f"  success%   — 后空翻成功率。what: 旋转≥286°且落地的局比例")
    print(f"               why: 286°（5.0 rad）是机器人旋转完成后空翻的最低门槛")
    print(f"               how: 100% = 每局都翻成功，0% = 从未成功")
    print(f"  rotation   — 空中累积旋转度数。what: 整局中向后翻转的最大总角度")
    print(f"               why: 机器人需转约360°才算真正完成后空翻")
    print(f"               how: <90° 旋转未形成，>360° 超旋转（会有惩罚）")
    print(f"  rot@land   — 落地瞬间旋转角度。what: 双脚接触地面那一刻的累积旋转")
    print(f"               why: 四足经验：空中峰值360°但落地时只有347°，需让翻转和落地时机同步")
    print(f"               how: 目标360°，差距越小越好")
    print(f"  ep_len     — 每局步数。Humanoid-v4 每步 0.015s，200步=3秒（超时）")
    print(f"               成功后空翻约需50步（0.75s）+80步站稳=130步，正常范围")
    print(f"  SAC        — Soft Actor-Critic，样本效率高的 off-policy 强化学习算法")
    print(f"               why: 17 DOF 连续控制任务，SAC 比 PPO 样本效率高5-10倍")
    print(f"{'='*65}")


def main() -> None:
    parser = argparse.ArgumentParser(description="人形机器人后空翻评估")
    parser.add_argument("--model",    type=str, default=str(DEFAULT_MODEL))
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()
    evaluate(args.model, args.episodes)


if __name__ == "__main__":
    main()
