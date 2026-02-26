#!/usr/bin/env python3
"""人形机器人确定性评估脚本。

使用训练好的策略（无探索噪声）运行若干局，汇报关键指标：
  ep_len     — 每局步数（目标 >500）
  fall_rate  — 摔倒率（terminated / total，越低越好）
  forward_m  — 每局前进距离（米）
  avg_z      — 平均躯干高度（越接近 1.4m 越好）

用法：
    conda activate jprobot
    python scripts/fixed_eval_humanoid.py              # 评估 v3（默认）
    python scripts/fixed_eval_humanoid.py --version v1
    python scripts/fixed_eval_humanoid.py --version v2 --episodes 10
    python scripts/fixed_eval_humanoid.py --model trained/humanoid_v3/checkpoints/xxx.zip
"""

import argparse
import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

TRAINED_DIR = Path(__file__).parent.parent / "trained"

# ────────────────────────────────────────────────
# 指标说明（方便姜哥一眼看懂）
# ────────────────────────────────────────────────
_METRIC_HELP = """
指标说明
  ep_len     每局步数     — 每步 0.01s，500 步 = 5 秒。目标 >500
  fall_rate  摔倒率       — terminated（z<1.0m 判定摔倒）占总局数比例
  forward_m  前进距离     — 每局 x 轴净位移（米）
  avg_z      平均躯干高度 — 接近 1.4m = 站得直，接近 1.0m = 快摔了
  ep_reward  累计奖励     — 整局总 reward（参考值）
"""


def _resolve_env_and_path(version: str | None, model_override: str | None):
    """根据版本选择环境类 + 模型路径。"""
    ver = (version or "v4").lower().lstrip("humanoid_")
    if ver == "v1":
        from jprobot.training.env_humanoid import HumanoidEnv as EnvCls
        model_path = TRAINED_DIR / "humanoid_v1" / "best.zip"
        out_path = TRAINED_DIR / "humanoid_v1" / "eval_result.json"
    elif ver == "v2":
        from jprobot.training.env_humanoid_v2 import HumanoidEnvV2 as EnvCls
        model_path = TRAINED_DIR / "humanoid_v2" / "best.zip"
        out_path = TRAINED_DIR / "humanoid_v2" / "eval_result.json"
    elif ver == "v3":
        from jprobot.training.env_humanoid_v3 import HumanoidEnvV3 as EnvCls
        model_path = TRAINED_DIR / "humanoid_v3" / "best.zip"
        out_path = TRAINED_DIR / "humanoid_v3" / "eval_result.json"
    elif ver == "v4":
        from jprobot.training.env_humanoid_v4 import HumanoidEnvV4 as EnvCls
        model_path = TRAINED_DIR / "humanoid_v4" / "best.zip"
        out_path = TRAINED_DIR / "humanoid_v4" / "eval_result.json"
    else:
        raise ValueError(f"未知版本: {version!r}，可选 v1 / v2 / v3 / v4")

    if model_override:
        model_path = Path(model_override)
        out_path = model_path.parent / "eval_result.json"

    return EnvCls, model_path, out_path


def run_eval(
    version: str | None = "v3",
    episodes: int = 20,
    model_override: str | None = None,
) -> dict:
    from stable_baselines3 import PPO

    EnvCls, model_path, out_path = _resolve_env_and_path(version, model_override)

    print(_METRIC_HELP)
    print(f"{'='*55}")
    print(f"人形机器人验收评估 — {EnvCls.__name__}")
    print(f"  模型路径: {model_path}")
    print(f"  评估局数: {episodes}")
    print(f"{'='*55}\n")

    if not model_path.exists():
        print(f"[错误] 模型文件不存在: {model_path}")
        print("  请先跑训练：python scripts/train_humanoid_v3.py")
        sys.exit(1)

    model = PPO.load(str(model_path))
    env = EnvCls(render_mode=None)

    ep_lens = []
    ep_rewards = []
    forward_dists = []
    avg_heights = []
    falls = 0

    for ep in range(episodes):
        obs, _ = env.reset()
        x_start = float(env.data.qpos[0])
        total_reward = 0.0
        z_history = []
        terminated = False
        step = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            z_history.append(float(env.data.qpos[2]))
            step += 1
            if terminated or truncated:
                break

        x_end = float(env.data.qpos[0])
        forward_m = x_end - x_start
        avg_z = float(np.mean(z_history)) if z_history else 0.0

        ep_lens.append(step)
        ep_rewards.append(total_reward)
        forward_dists.append(forward_m)
        avg_heights.append(avg_z)
        if terminated:
            falls += 1

        status = "摔" if terminated else "时间到"
        print(f"  局 {ep+1:3d}/{episodes}: "
              f"步数={step:4d}  "
              f"奖励={total_reward:7.1f}  "
              f"前进={forward_m:5.2f}m  "
              f"平均高度={avg_z:.3f}m  "
              f"[{status}]")

    env.close()

    # ────────── 汇总 ──────────
    mean_ep_len    = float(np.mean(ep_lens))
    median_ep_len  = float(np.median(ep_lens))
    mean_reward    = float(np.mean(ep_rewards))
    mean_forward   = float(np.mean(forward_dists))
    mean_height    = float(np.mean(avg_heights))
    fall_rate      = falls / episodes

    target_ok = mean_ep_len >= 500

    print(f"\n{'='*55}")
    print(f"验收汇总（{episodes} 局）")
    print(f"{'='*55}")
    print(f"  ep_len 均值  : {mean_ep_len:.1f}  （中位 {median_ep_len:.0f}）  {'✓ 达标' if target_ok else '✗ 未达标（目标 ≥500）'}")
    print(f"  摔倒率       : {fall_rate*100:.0f}%  （{falls}/{episodes} 局摔倒）")
    print(f"  前进距离均值 : {mean_forward:.3f} m/局")
    print(f"  平均躯干高度 : {mean_height:.3f} m  （初始 1.4m，阈值 1.0m）")
    print(f"  累计奖励均值 : {mean_reward:.1f}")
    print(f"{'='*55}")

    result = {
        "model": str(model_path),
        "env_class": EnvCls.__name__,
        "episodes": episodes,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "metrics": {
            "ep_len_mean":   round(mean_ep_len, 1),
            "ep_len_median": round(median_ep_len, 0),
            "fall_rate":     round(fall_rate, 3),
            "forward_m_mean": round(mean_forward, 4),
            "avg_z_mean":    round(mean_height, 4),
            "ep_reward_mean": round(mean_reward, 1),
        },
        "target_met": target_ok,   # ep_len ≥ 500
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n结果已写入: {out_path}")

    # 与历史对比
    history_path = out_path.parent / "eval_history.jsonl"
    with open(history_path, "a") as f:
        f.write(json.dumps({
            "evaluated_at": result["evaluated_at"],
            **result["metrics"],
        }) + "\n")

    print(_METRIC_HELP)
    return result


def main():
    parser = argparse.ArgumentParser(description="人形机器人固定评估")
    parser.add_argument("--version",  type=str, default="v3",
                        help="环境版本 v1 / v2 / v3（默认 v3）")
    parser.add_argument("--episodes", type=int, default=20,
                        help="评估局数（默认 20）")
    parser.add_argument("--model",    type=str, default=None,
                        help="直接指定模型路径（覆盖 --version 的默认路径）")
    args = parser.parse_args()
    run_eval(version=args.version, episodes=args.episodes, model_override=args.model)


if __name__ == "__main__":
    main()
