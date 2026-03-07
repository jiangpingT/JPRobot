#!/usr/bin/env python3
"""人形机器人跑步评估 — 速度跟随 + 腾空相质量检验。

新增指标（相比万向行走评估）：
  airborne_pct — 每局腾空相占比（双脚同时离地的步数比例，越高说明越接近真正跑步）
  vx_actual    — 每局实际平均前进速度（m/s）
  vx_cmd       — 每局目标前进速度（m/s）

用法：
  python scripts/eval_humanoid_run.py               # 默认 20 局
  python scripts/eval_humanoid_run.py --episodes 5  # 快速验证
  python scripts/eval_humanoid_run.py --model trained/humanoid_run/best.zip
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import SAC
from jprobot.training.env_humanoid_run import HumanoidRunEnv


DEFAULT_MODEL = Path(__file__).parent.parent / "trained" / "humanoid_run" / "best.zip"


def evaluate(model_path: str, n_episodes: int) -> None:
    print(f"{'='*65}")
    print(f"人形机器人跑步评估（速度追踪 + 腾空相）")
    print(f"  模型: {model_path}")
    print(f"  局数: {n_episodes}")
    print()
    print(f"  【指标说明】")
    print(f"    vel_error   — 速度误差（越小越好）")
    print(f"    airborne%   — 腾空相占比（双脚同时离地，越高越接近跑步步态）")
    print(f"    vx_actual   — 实际平均前进速度（m/s）")
    print(f"    vx_cmd      — 目标前进速度（m/s）")
    print(f"    ep_len      — 每局存活步数（满分 1000）")
    print(f"{'='*65}")

    if not Path(model_path).exists():
        print(f"[错误] 模型文件不存在: {model_path}")
        return

    print("加载模型和环境...")
    env = HumanoidRunEnv()
    model = SAC.load(model_path, env=env)
    print("加载完成，开始评估...\n")

    all_vel_errors   = []
    all_vel_rewards  = []
    all_ep_lens      = []
    all_ep_rews      = []
    all_airborne_pct = []
    all_vx_actual    = []
    all_vx_cmd       = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        cmd_vx = env.cmd_vx
        cmd_vy = env.cmd_vy
        cmd_wz = env.cmd_wz

        ep_vel_errors   = []
        ep_vel_rewards  = []
        ep_airborne     = []
        ep_vx_actual    = []
        ep_reward       = 0.0
        step            = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            ep_reward += float(reward)

            ep_vel_errors.append(float(info.get("vel_error", 0.0)))
            ep_vel_rewards.append(float(info.get("vel_reward", 0.0)))
            ep_airborne.append(1 if info.get("airborne", False) else 0)
            vx_actual = info.get("vel_actual", (0.0, 0.0, 0.0))[0]
            ep_vx_actual.append(float(vx_actual))

            if terminated or truncated:
                break

        vel_error_mean  = float(np.mean(ep_vel_errors))
        vel_reward_mean = float(np.mean(ep_vel_rewards))
        airborne_pct    = float(np.mean(ep_airborne)) * 100.0
        vx_mean         = float(np.mean(ep_vx_actual))

        print(
            f"  Ep {ep+1:3d} | cmd_vx={cmd_vx:+.2f} "
            f"| ep_len={step:4d} | ep_rew={ep_reward:7.1f} "
            f"| vel_err={vel_error_mean:.3f} "
            f"| airborne={airborne_pct:5.1f}% "
            f"| vx={vx_mean:+.2f}m/s"
        )

        all_vel_errors.append(vel_error_mean)
        all_vel_rewards.append(vel_reward_mean)
        all_ep_lens.append(step)
        all_ep_rews.append(ep_reward)
        all_airborne_pct.append(airborne_pct)
        all_vx_actual.append(vx_mean)
        all_vx_cmd.append(cmd_vx)

    env.close()

    # ── 汇总报告 ──────────────────────────────────────────────────────────
    mean_err     = float(np.mean(all_vel_errors))
    mean_rew_vel = float(np.mean(all_vel_rewards))
    mean_len     = float(np.mean(all_ep_lens))
    mean_airborne= float(np.mean(all_airborne_pct))
    mean_vx      = float(np.mean(all_vx_actual))
    mean_vx_cmd  = float(np.mean(all_vx_cmd))

    print()
    print(f"{'='*65}")
    print(f"[总体结果] {n_episodes} 局均值")
    print(f"  ep_len      = {mean_len:.1f}   (满分 1000)")
    print(f"  ep_rew      = {float(np.mean(all_ep_rews)):.1f}")
    print(f"  vel_error   = {mean_err:.4f}  (越小越好)")
    print(f"  vel_reward  = {mean_rew_vel:.4f}  (满分 W_VEL=5.0)")
    print(f"  airborne%   = {mean_airborne:.1f}%   (双脚腾空占比，>10% 接近跑步步态)")
    print(f"  vx_actual   = {mean_vx:+.3f} m/s  (实际平均前进速度)")
    print(f"  vx_cmd      = {mean_vx_cmd:+.3f} m/s  (目标平均前进速度)")
    print()

    print(f"[质量判断]")
    if mean_len > 800:
        print(f"  存活能力：✅ 优秀（ep_len={mean_len:.0f} > 800）")
    elif mean_len > 400:
        print(f"  存活能力：⚠️  中等（ep_len={mean_len:.0f}）")
    else:
        print(f"  存活能力：❌ 差（ep_len={mean_len:.0f}，频繁摔倒）")

    if mean_err < 1.0:
        print(f"  速度跟随：✅ 优秀（vel_error={mean_err:.3f} < 1.0）")
    elif mean_err < 3.0:
        print(f"  速度跟随：⚠️  中等（vel_error={mean_err:.3f}）")
    else:
        print(f"  速度跟随：❌ 较差（vel_error={mean_err:.3f}，速度命令基本未跟随）")

    if mean_airborne >= 20.0:
        print(f"  腾空步态：✅ 优秀（airborne={mean_airborne:.1f}% ≥ 20%，明显跑步步态）")
    elif mean_airborne >= 5.0:
        print(f"  腾空步态：⚠️  初现（airborne={mean_airborne:.1f}%，有腾空但不稳定）")
    else:
        print(f"  腾空步态：❌ 未出现（airborne={mean_airborne:.1f}%，仍是行走步态）")

    speed_ratio = mean_vx / max(0.01, mean_vx_cmd)
    if speed_ratio >= 0.7:
        print(f"  速度达成：✅ 优秀（实际/目标={speed_ratio:.0%}，{mean_vx:.2f}/{mean_vx_cmd:.2f} m/s）")
    elif speed_ratio >= 0.4:
        print(f"  速度达成：⚠️  中等（实际/目标={speed_ratio:.0%}，{mean_vx:.2f}/{mean_vx_cmd:.2f} m/s）")
    else:
        print(f"  速度达成：❌ 较差（实际/目标={speed_ratio:.0%}，{mean_vx:.2f}/{mean_vx_cmd:.2f} m/s）")

    print()
    print(f"【名词备注】")
    print(f"  airborne%  — 腾空相占比。what: 每局中双脚同时离地的步数比例")
    print(f"               why: 行走步态至少有一只脚在地，跑步必须有腾空期")
    print(f"               how: >20% 是明显跑步步态，5-20% 是过渡状态，<5% 基本是行走")
    print(f"  vx_actual  — actual forward velocity，实际前进速度（m/s）")
    print(f"               why: 目标速度可能超出物理上限，需看实际达到了多少")
    print(f"               how: MuJoCo Humanoid-v4 物理上限约 1.2-1.5 m/s")
    print(f"  vel_error  — 速度追踪误差。公式：(vx-vx_cmd)²+(vy-vy_cmd)²+WZ_WEIGHT*(wz-wz_cmd)²")
    print(f"  cfrc_ext   — contact force external，MuJoCo 外部接触力数组，用于判断脚是否着地")
    print(f"{'='*65}")


def main() -> None:
    parser = argparse.ArgumentParser(description="人形机器人跑步评估")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()
    evaluate(args.model, args.episodes)


if __name__ == "__main__":
    main()
