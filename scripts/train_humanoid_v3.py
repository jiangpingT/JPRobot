#!/usr/bin/env python3
"""人形机器人 PPO 训练脚本 V3 — 高度维持 + 交替步态奖励。

V2 问题：ep_len 仍卡在 ~110 步——诊断发现 z 缓慢下沉（每步降 0.003m），
         agent 没有"主动对抗重力"的明确激励。

V3 改动：
    - env: HumanoidEnvV3（高度维持 + 交替步态奖励）
    - 从 V1 best 热启动（V2 训练中，V1 weights 已包含基础平衡知识）
    - ent_coef: 0.01（同 V2）
    - 步数: 10M

用法：
    conda activate jprobot
    python scripts/train_humanoid_v3.py                # 默认从 V1 热启动
    python scripts/train_humanoid_v3.py --steps 5      # 快速验证
    python scripts/train_humanoid_v3.py --resume trained/humanoid_v2/best.zip
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
from jprobot.training.env_humanoid_v3 import HumanoidEnvV3

TRAINED_DIR = Path(__file__).parent.parent / "trained" / "humanoid_v3"
V1_BEST     = Path(__file__).parent.parent / "trained" / "humanoid_v1" / "best.zip"
V2_BEST     = Path(__file__).parent.parent / "trained" / "humanoid_v2" / "best.zip"

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


class HumanoidV3Callback(BaseCallback):
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
            print(f"  [humanoid_v3] {self.num_timesteps/1e6:.2f}M / "
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
            "title": "人形机器人训练 v3（高度+步态）",
            "run_id": "humanoid_v3",
            "updated_at": now_str,
            "progress": {
                "current_steps": self.num_timesteps,
                "total_steps": self.total_steps,
            },
            "stages": [{
                "name": "full",
                "label": "行走 V3",
                "done": self.num_timesteps >= self.total_steps,
                "reward": round(rew_mean, 1),
                "note": "高度维持+交替步态",
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


def train(total_steps: int, n_envs: int, resume_path: str | None) -> None:
    TRAINED_DIR.mkdir(parents=True, exist_ok=True)

    # 优先从 V2 热启动，其次 V1
    if not resume_path or resume_path.lower() == "none":
        if V2_BEST.exists():
            resume_path = str(V2_BEST)
            print(f"[自动] 使用 V2 best 热启动: {V2_BEST}")
        elif V1_BEST.exists():
            resume_path = str(V1_BEST)
            print(f"[自动] V2 不存在，使用 V1 best 热启动: {V1_BEST}")

    print(f"{'='*60}")
    print(f"人形机器人训练 v3（高度维持 + 交替步态）")
    print(f"  步数: {total_steps/1e6:.1f}M  并行环境: {n_envs}")
    print(f"  续训: {resume_path}")
    print(f"  输出: {TRAINED_DIR}")
    print(f"  关键改动: height_reward + alternating_gait_reward")
    print(f"{'='*60}")

    vec_env = make_vec_env(
        HumanoidEnvV3,
        n_envs=n_envs,
        env_kwargs={"render_mode": None},
        vec_env_cls=SubprocVecEnv,
    )

    if resume_path and Path(resume_path).exists():
        print(f"热启动：从 {resume_path} 迁移策略网络...")
        model = PPO.load(
            resume_path, env=vec_env, device="cpu",
            ent_coef=PPO_KWARGS["ent_coef"],
            learning_rate=PPO_KWARGS["learning_rate"],
        )
    else:
        print("从头初始化")
        model = PPO("MlpPolicy", vec_env, verbose=0, device="cpu", **PPO_KWARGS)

    cb = HumanoidV3Callback(total_steps, TRAINED_DIR)
    ckpt_cb = CheckpointCallback(
        save_freq=max(500_000 // n_envs, 1),
        save_path=str(TRAINED_DIR / "checkpoints"),
        name_prefix="humanoid_v3",
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
    parser = argparse.ArgumentParser(description="人形机器人训练 V3")
    parser.add_argument("--steps",  type=float, default=10.0)
    parser.add_argument("--envs",   type=int,   default=8)
    parser.add_argument("--resume", type=str,   default=None,
                        help="续训路径（默认优先 V2 best，其次 V1 best）")
    args = parser.parse_args()
    train(int(args.steps * 1_000_000), args.envs, args.resume)


if __name__ == "__main__":
    main()
