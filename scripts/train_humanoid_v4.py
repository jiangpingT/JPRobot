#!/usr/bin/env python3
"""人形机器人 PPO 训练脚本 V4 — 扩展观测空间（203 维），突破 ep_len 瓶颈。

V3 问题：ep_len 卡在 ~140 步（每百万步仅增 13 步），无法到达目标 500 步。
根因：47 维观测缺少身体帧速度（cvel）和接触力（cfrc_ext），agent 感知不足以学平衡。

V4 改动：
    - env: HumanoidEnvV4（观测 47→203 维，加 cvel+cfrc_ext）
    - 奖励：完全复用 V3（高度维持 + 交替步态），奖励设计没问题
    - 从 SCRATCH 开始（obs 维度变了，V3 权重不兼容）
    - 步数: 15M（任务更难，需要更多探索）

用法：
    conda activate jprobot
    python scripts/train_humanoid_v4.py             # 默认 15M 步
    python scripts/train_humanoid_v4.py --steps 5   # 快速验证
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

sys.path.insert(0, str(Path(__file__).parent.parent))
from jprobot.training.env_humanoid_v4 import HumanoidEnvV4

TRAINED_DIR = Path(__file__).parent.parent / "trained" / "humanoid_v4"

PPO_KWARGS = dict(
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    learning_rate=3e-4,
    ent_coef=0.01,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    policy_kwargs=dict(net_arch=[256, 256]),
    seed=42,
)


class HumanoidV4Callback(BaseCallback):
    def __init__(self, total_steps: int, out_dir: Path):
        super().__init__(verbose=0)
        self.total_steps = total_steps
        self.out_dir = out_dir
        self._best_reward = -np.inf
        self._last_written: str | None = None

    def _on_rollout_end(self) -> None:
        buf = self.model.ep_info_buffer
        if not buf:
            return
        rew_mean = float(np.mean([ep["r"] for ep in buf]))
        len_mean = float(np.mean([ep["l"] for ep in buf]))
        now_str  = datetime.now().isoformat(timespec="seconds")

        if self.num_timesteps % 100_000 < 2048:
            print(f"  [humanoid_v4] {self.num_timesteps/1e6:.2f}M / "
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

    def _write_live_dashboard(self, rew_mean: float, len_mean: float, now_str: str) -> None:
        spec = {
            "title": "人形机器人训练 v4（扩展观测 203 维）",
            "run_id": "humanoid_v4",
            "updated_at": now_str,
            "progress": {
                "current_steps": self.num_timesteps,
                "total_steps": self.total_steps,
            },
            "stages": [{
                "name": "full",
                "label": "行走 V4",
                "done": self.num_timesteps >= self.total_steps,
                "reward": round(rew_mean, 1),
                "note": "扩展obs(cvel+cfrc_ext)+V3奖励",
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
    print(f"人形机器人训练 v4（扩展观测 203 维）")
    print(f"  步数: {total_steps/1e6:.1f}M  并行环境: {n_envs}")
    print(f"  obs 维度: 47 → 203（+cvel 78 + cfrc_ext 78）")
    print(f"  obs 新增: 身体帧速度（感知倒向）+ 接触力（感知蹬地）")
    print(f"  奖励: 复用 V3（高度维持+交替步态）")
    print(f"  从 SCRATCH 开始（obs 维度变化，无法迁移 V3 权重）")
    print(f"  输出: {TRAINED_DIR}")
    print(f"{'='*60}")

    vec_env = make_vec_env(
        HumanoidEnvV4,
        n_envs=n_envs,
        env_kwargs={"render_mode": None},
        vec_env_cls=SubprocVecEnv,
    )

    print("从头初始化策略网络...")
    model = PPO("MlpPolicy", vec_env, verbose=0, device="cpu", **PPO_KWARGS)

    cb = HumanoidV4Callback(total_steps, TRAINED_DIR)
    ckpt_cb = CheckpointCallback(
        save_freq=max(500_000 // n_envs, 1),
        save_path=str(TRAINED_DIR / "checkpoints"),
        name_prefix="humanoid_v4",
        verbose=0,
    )

    print("开始训练...")
    model.learn(
        total_timesteps=total_steps,
        callback=[cb, ckpt_cb],
        reset_num_timesteps=True,
        progress_bar=True,
    )

    model.save(TRAINED_DIR / "final.zip")
    vec_env.close()

    print(f"\n训练完成！  最佳奖励: {cb._best_reward:.1f}")
    print(f"最佳模型: {TRAINED_DIR / 'best.zip'}")


def main():
    parser = argparse.ArgumentParser(description="人形机器人训练 V4（扩展观测）")
    parser.add_argument("--steps", type=float, default=15.0)
    parser.add_argument("--envs",  type=int,   default=8)
    args = parser.parse_args()
    train(int(args.steps * 1_000_000), args.envs)


if __name__ == "__main__":
    main()
