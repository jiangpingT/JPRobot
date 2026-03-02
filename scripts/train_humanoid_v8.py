#!/usr/bin/env python3
"""人形机器人 PPO 训练脚本 V8 — 稳定超参 + Gymnasium 原生 + VecNormalize。

## 所有失败版本的共同规律

| 版本 | n_steps | 峰值ep_len | 峰值时步数 | 后续   |
|------|---------|-----------|-----------|--------|
| v4   | 2048    | ~105      | ~3M       | 稳定   |
| v5   | 512     | 104       | 2.34M     | 崩到28 |
| v6   | 512     | 127       | 4.65M     | 降至109|
| v7a  | 512     | 127       | 2.55M     | 崩到76 |
| v7b  | 512     | 125       | 2.57M     | 崩至92 |

## 根因：n_steps=512 造成训练不稳定

n_steps=512 × 8 envs = 每轮 4096 步。
ep_len≈100 时，每轮只有 ~32 个完整 episode。
32 个 episode 的平均奖励方差很高（法则：样本越少，方差越大）。
高方差梯度 → 策略更新偶发大偏移 → 进入坏区域 → 崩溃。

n_steps=2048 × 8 envs = 每轮 16384 步 ≈ 163 个完整 episode。
163 个 episode 的平均奖励方差低 5 倍 → 梯度稳定 → v4 不崩溃。

## V8 策略

回到 v4 的稳定超参（n_steps=2048），同时加入两个真正有用的改进：
1. Gymnasium Humanoid-v4（376维，包含 cinert/qfrc_act）- 提升 ep_len 上限
2. VecNormalize（norm_obs+norm_reward）- 归一化各维度尺度

不改动 n_steps/batch_size/lr/ent_coef - 这些在 v4 上已验证稳定。

## 预期

- v4 在 Gymnasium Humanoid-v4 的论文基准：15M 步 → ep_len ~300-500
- v8 在 v4 稳定超参 + VecNorm + Gymnasium：保守估计 ep_len ~200-300 at 15M
- 不崩溃 > 一次性到 150 后崩溃

用法：
    conda activate jprobot
    python scripts/train_humanoid_v8.py             # 默认 15M 步
    python scripts/train_humanoid_v8.py --steps 5   # 快速验证
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).parent.parent))

TRAINED_DIR  = Path(__file__).parent.parent / "trained" / "humanoid_v8"
VECNORM_PATH = TRAINED_DIR / "vecnormalize.pkl"

# ── v4 的稳定超参 + 少量安全改进 ─────────────────────────────────────────
PPO_KWARGS = dict(
    n_steps=2048,          # v4 验证稳定：每轮 16384 步 ≈ 163 episode，梯度方差低
    batch_size=256,        # v4 验证稳定
    n_epochs=10,
    learning_rate=3e-4,    # v4 验证稳定
    ent_coef=0.01,         # v4 验证稳定（0.00481 太小，降低了探索，可能是崩溃原因之一）
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,        # v4 验证稳定（0.185 过于保守）
    policy_kwargs=dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=nn.Tanh,  # 新增：Tanh 比 ReLU 稳定（有界）
        ortho_init=False,       # 新增：Gymnasium Humanoid 标准做法
        log_std_init=-2,        # 新增：初始动作更精确
    ),
    seed=42,
)


class HumanoidV8Callback(BaseCallback):
    def __init__(self, total_steps: int, out_dir: Path, vec_norm: VecNormalize):
        super().__init__(verbose=0)
        self.total_steps  = total_steps
        self.out_dir      = out_dir
        self.vec_norm     = vec_norm
        self._best_reward = -np.inf
        self._last_written: str | None = None

    def _on_rollout_end(self) -> None:
        buf = self.model.ep_info_buffer
        if not buf:
            return
        rew_mean = float(np.mean([ep["r"] for ep in buf]))
        len_mean = float(np.mean([ep["l"] for ep in buf]))
        now_str  = datetime.now().isoformat(timespec="seconds")

        if self.num_timesteps % 100_000 < 2048 * 8:
            print(f"  [humanoid_v8] {self.num_timesteps/1e6:.2f}M / "
                  f"{self.total_steps/1e6:.1f}M  "
                  f"rew={rew_mean:.1f}  ep_len={len_mean:.0f}")

        try:
            with open(self.out_dir / "progress.json", "w") as f:
                json.dump({
                    "stage": "full",
                    "timesteps": self.num_timesteps,
                    "total_timesteps": self.total_steps,
                    "progress": self.num_timesteps / max(1, self.total_steps),
                    "ep_rew_mean": rew_mean,
                    "ep_len_mean": len_mean,
                    "updated_at": now_str,
                }, f, indent=2)
        except OSError:
            pass

        if now_str != self._last_written:
            self._last_written = now_str
            self._write_live_dashboard(rew_mean, len_mean, now_str)

        if rew_mean > self._best_reward:
            self._best_reward = rew_mean
            self.model.save(self.out_dir / "best.zip")
            self.vec_norm.save(str(VECNORM_PATH))

    def _write_live_dashboard(self, rew_mean: float, len_mean: float, now_str: str) -> None:
        spec = {
            "title": "人形机器人训练 v8（稳定超参+Gymnasium+VecNorm）",
            "run_id": "humanoid_v8",
            "updated_at": now_str,
            "progress": {
                "current_steps": self.num_timesteps,
                "total_steps": self.total_steps,
            },
            "stages": [{
                "name": "full",
                "label": "行走 V8",
                "done": self.num_timesteps >= self.total_steps,
                "reward": round(rew_mean, 1),
                "note": "n_steps=2048稳定超参+Gymnasium 376维+VecNorm+Tanh",
            }],
            "metrics": [
                {"label": "Reward",   "sublabel": "当前奖励",  "value": round(rew_mean, 1),  "color": "green"},
                {"label": "ep_len",   "sublabel": "每局步数",  "value": round(len_mean, 0),  "color": "orange"},
                {"label": "Progress", "sublabel": "训练进度",
                 "value": f"{self.num_timesteps/self.total_steps*100:.1f}%",               "color": "blue"},
                {"label": "Best",     "sublabel": "历史最佳",  "value": round(self._best_reward, 1), "color": "yellow"},
            ],
            "history_file": "metrics_history.jsonl",
        }
        try:
            with open(TRAINED_DIR / "live_dashboard.json", "w") as f:
                json.dump(spec, f, ensure_ascii=False, indent=2)
        except OSError:
            pass
        try:
            with open(TRAINED_DIR / "metrics_history.jsonl", "a") as f:
                f.write(json.dumps({
                    "total_timesteps": self.num_timesteps,
                    "ep_rew_mean": round(rew_mean, 1),
                    "ep_len_mean": round(len_mean, 1),
                    "stage": "full",
                }) + "\n")
        except OSError:
            pass

    def _on_step(self) -> bool:
        return True


def train(total_steps: int, n_envs: int) -> None:
    TRAINED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"人形机器人训练 v8（稳定超参 + Gymnasium + VecNorm）")
    print(f"  步数: {total_steps/1e6:.1f}M  并行环境: {n_envs}")
    print(f"  环境: Gymnasium Humanoid-v4（376 维 obs）")
    print(f"  n_steps: {PPO_KWARGS['n_steps']}（v4 验证稳定，每轮163ep）")
    print(f"  batch_size: {PPO_KWARGS['batch_size']}  lr: {PPO_KWARGS['learning_rate']}")
    print(f"  VecNormalize: norm_obs=True, norm_reward=True")
    print(f"  activation: Tanh  ortho_init: False")
    print(f"  输出: {TRAINED_DIR}")
    print(f"{'='*60}")

    vec_env = make_vec_env(
        "Humanoid-v4",
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,
    )

    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        gamma=PPO_KWARGS["gamma"],
    )

    print("从零初始化策略网络（MlpPolicy，376维输入，Tanh激活）...")
    model = PPO("MlpPolicy", vec_env, verbose=0, device="cpu", **PPO_KWARGS)

    cb   = HumanoidV8Callback(total_steps, TRAINED_DIR, vec_env)
    ckpt = CheckpointCallback(
        save_freq=max(500_000 // n_envs, 1),
        save_path=str(TRAINED_DIR / "checkpoints"),
        name_prefix="humanoid_v8",
        verbose=0,
    )

    print("开始训练...")
    model.learn(
        total_timesteps=total_steps,
        callback=[cb, ckpt],
        reset_num_timesteps=True,
        progress_bar=True,
    )

    model.save(TRAINED_DIR / "final.zip")
    vec_env.save(str(TRAINED_DIR / "vecnormalize_final.pkl"))
    print(f"训练完成！模型: {TRAINED_DIR}/final.zip")
    vec_env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="人形机器人 PPO 训练 V8")
    parser.add_argument("--steps", type=int, default=15_000_000, help="训练总步数")
    parser.add_argument("--envs",  type=int, default=8,           help="并行环境数")
    args = parser.parse_args()
    train(args.steps, args.envs)


if __name__ == "__main__":
    main()
