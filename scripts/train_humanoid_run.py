#!/usr/bin/env python3
"""人形机器人跑步训练 — 高速行走 + 腾空相（flight phase）。

在万向行走 v6 基础上两处核心改动：
  1. VX_RANGE 扩展到 3.5 m/s（跑步速度）
  2. 腾空相奖励：双脚同时腾空且 vx > 1.5 m/s 时给额外 +2.0 奖励

建议热启自 trained/humanoid_velocity/best_v6_backup.zip（obs=379，最优行走模型）。
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

sys.path.insert(0, str(Path(__file__).parent.parent))
from jprobot.training.env_humanoid_run import HumanoidRunEnv

TRAINED_DIR = Path(__file__).parent.parent / "trained" / "humanoid_run"


def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


SAC_KWARGS = dict(
    batch_size=512,
    buffer_size=1_000_000,
    learning_starts=10_000,
    gradient_steps=1,
    train_freq=1,
    ent_coef="auto",
    gamma=0.99,
    tau=0.005,
    learning_rate=linear_schedule(7.3e-4),
    use_sde=True,
    sde_sample_freq=64,
    policy_kwargs=dict(
        log_std_init=-3.67,
        net_arch=[400, 300],
    ),
    seed=42,
    verbose=0,
)


class RunTrackingCallback(BaseCallback):
    """训练回调：打印日志 + 写 progress.json。"""

    def __init__(self, total_steps: int, out_dir: Path):
        super().__init__(verbose=0)
        self.total_steps  = total_steps
        self.out_dir      = out_dir
        self._best_reward = -np.inf

    def _on_step(self) -> bool:
        if self.num_timesteps % 10_000 != 0:
            return True
        buf = self.model.ep_info_buffer
        if not buf:
            return True

        rew_mean = float(np.mean([ep["r"] for ep in buf]))
        len_mean = float(np.mean([ep["l"] for ep in buf]))
        now_str  = datetime.now().isoformat(timespec="seconds")

        print(
            f"  [run] {self.num_timesteps/1e6:.2f}M / "
            f"{self.total_steps/1e6:.1f}M  "
            f"rew={rew_mean:.1f}  ep_len={len_mean:.0f}"
        )

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

        if rew_mean > self._best_reward:
            self._best_reward = rew_mean
            self.model.save(self.out_dir / "best.zip")

        return True


def train(total_steps: int, n_envs: int, resume_path: str | None = None) -> None:
    TRAINED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{'='*65}")
    print(f"人形机器人跑步训练（高速万向 + 腾空相奖励）")
    print(f"  步数: {total_steps/1e6:.1f}M  并行环境: {n_envs}")
    print(f"  算法: SAC  环境: HumanoidRunEnv（379维 obs）")
    print(f"  速度命令: vx∈[-0.5,2.0]m/s  70%采样高速>0.5m/s")
    print(f"  腾空相奖励: +2.0（双脚腾空 + vx>0.5 m/s 时触发）")
    print(f"  直立奖励:   +3.0（v3新增，躯干高度 1.0→1.3m 线性，解决爬行gaming）")
    print(f"  输出: {TRAINED_DIR}")
    if resume_path:
        print(f"  热启: {resume_path}")
    print()
    print(f"  【指标说明】")
    print(f"    ep_len    — 每局存活步数（目标 >800）")
    print(f"    Reward    — 速度追踪 + 腾空相 + 存活奖励之和")
    print(f"    airborne  — 腾空相奖励（每步有腾空时 +2.0）")
    print(f"{'='*65}")

    vec_env = make_vec_env(
        lambda: HumanoidRunEnv(),
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,
    )

    if resume_path:
        print(f"从 checkpoint 热启（obs=379）: {resume_path}")
        model = SAC("MlpPolicy", vec_env, device="cpu", **SAC_KWARGS)
        loaded = SAC.load(resume_path)
        model.policy.load_state_dict(loaded.policy.state_dict())
        del loaded
    else:
        print("从零初始化 SAC 策略网络（obs=379）...")
        model = SAC("MlpPolicy", vec_env, device="cpu", **SAC_KWARGS)

    cb   = RunTrackingCallback(total_steps, TRAINED_DIR)
    ckpt = CheckpointCallback(
        save_freq=max(200_000 // n_envs, 1),
        save_path=str(TRAINED_DIR / "checkpoints"),
        name_prefix="humanoid_run",
        verbose=0,
    )

    print("开始训练...")
    model.learn(
        total_timesteps=total_steps,
        callback=[cb, ckpt],
        reset_num_timesteps=True,
        progress_bar=False,
    )

    model.save(TRAINED_DIR / "final.zip")
    print(f"\n训练完成！  最佳: {TRAINED_DIR}/best.zip")
    vec_env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="人形机器人跑步训练")
    parser.add_argument("--steps",  type=int, default=5_000_000)
    parser.add_argument("--envs",   type=int, default=4)
    parser.add_argument("--resume", type=str, default=None,
                        help="热启 checkpoint 路径（建议 trained/humanoid_velocity/best_v6_backup.zip）")
    args = parser.parse_args()
    train(args.steps, args.envs, resume_path=args.resume)


if __name__ == "__main__":
    main()
