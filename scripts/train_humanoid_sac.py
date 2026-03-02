#!/usr/bin/env python3
"""人形机器人 SAC 训练脚本 — RL Zoo 调优配置，2M 步目标 ep_len=500+。

## 为什么从 PPO 切换到 SAC

PPO（Proximal Policy Optimization）的根本限制：
  - On-policy：每次只能利用当前策略收集的数据，样本效率低
  - Humanoid-v4 需要 10M+ 步 PPO（GPU）才能到 ep_len=500
  - CPU 上 PPO 训练速度~2500 it/s → 15M 步需要 1.7 小时，结果只有 ep_len=130
  - 根本原因：PPO 在困难的连续控制任务上需要非常多样本才能学会稳定步态

SAC（Soft Actor-Critic）的优势：
  - Off-policy：可以从 Replay Buffer 中反复学习历史经验，样本效率高 5-10x
  - 自动熵调节（ent_coef='auto'）：自适应探索，不需要手动调熵系数
  - SDE（State Dependent Exploration）：更有效的探索策略，尤其适合连续控制
  - RL Zoo 基准：Gymnasium Humanoid-v4，SAC 2M 步即可到 ep_len=500+

## RL Zoo SAC 超参（完整照搬 Humanoid-v4 配置）

来源：rl-baselines3-zoo hyperparams/sac.yml - Humanoid-v4:
  n_timesteps: 2_000_000
  policy: MlpPolicy
  batch_size: 512
  buffer_size: 1_000_000
  learning_starts: 10_000
  gradient_steps: 1
  train_freq: 1
  ent_coef: auto
  gamma: 0.99
  tau: 0.005
  learning_rate: lin_schedule(7.3e-4)  # 线性衰减到 0
  policy_kwargs: dict(log_std_init=-3.67, net_arch=[400, 300])
  use_sde: True       # State Dependent Exploration（关键！）
  sde_sample_freq: 64
  normalize: false    # SAC 不用 VecNormalize

## 为什么 SAC 不需要 VecNormalize

SAC 通过自动熵调节（alpha × entropy）来稳定训练，不像 PPO 那样
对观测尺度差异敏感（PPO 用 GAE 计算优势，对尺度很敏感）。
SAC 的 Critic 直接学 Q 值，对观测尺度的鲁棒性更强。

## State Dependent Exploration (SDE)

use_sde=True 是 SAC 在 Humanoid 上收敛的关键！
标准 SAC 使用固定的高斯噪声（与状态无关），
SDE 让噪声参数本身成为状态的函数（学习"在哪种姿态下需要多大探索"），
大幅改善步态探索效率。

用法：
    conda activate jprobot
    python scripts/train_humanoid_sac.py           # 默认 2M 步
    python scripts/train_humanoid_sac.py --steps 5000  # 快速验证
    python scripts/train_humanoid_sac.py --steps 5000000  # 5M 步（更稳妥）
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

TRAINED_DIR    = Path(__file__).parent.parent / "trained" / "humanoid_sac"
TRAINED_DIR_V2 = Path(__file__).parent.parent / "trained" / "humanoid_sac_v2"


def linear_schedule(initial_value: float):
    """SAC 的线性学习率衰减：从 initial_value 线性衰减到 0。"""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


# ── RL Zoo SAC 超参（Humanoid-v4，完整照搬）─────────────────────────────
SAC_KWARGS = dict(
    batch_size=512,           # 比 PPO 的 256 大，利用 replay buffer
    buffer_size=1_000_000,    # 100 万步 replay buffer（SAC 核心：可反复学习）
    learning_starts=10_000,   # 先收集 10k 步经验再开始学习（避免过早学偏）
    gradient_steps=1,         # 每 env step 做 1 次梯度更新
    train_freq=1,             # 每 1 env step 触发 1 次训练
    ent_coef="auto",          # 自动调节熵系数（alpha）：自适应探索强度
    gamma=0.99,
    tau=0.005,                # 目标网络软更新速率（每步只更新 0.5%）
    learning_rate=linear_schedule(7.3e-4),  # 线性衰减，开始时 7.3e-4
    use_sde=True,             # SDE：状态相关探索（Humanoid 关键！）
    sde_sample_freq=64,       # 每 64 步重采样一次探索噪声
    policy_kwargs=dict(
        log_std_init=-3.67,   # 初始动作标准差 exp(-3.67) ≈ 0.025（精确）
        net_arch=[400, 300],  # SAC 用 [400, 300] 比 PPO 的 [256, 256] 更大
    ),
    seed=42,
    verbose=0,
)


class HumanoidSACCallback(BaseCallback):
    def __init__(self, total_steps: int, out_dir: Path):
        super().__init__(verbose=0)
        self.total_steps  = total_steps
        self.out_dir      = out_dir
        self._best_reward = -np.inf
        self._last_written: str | None = None

    def _on_step(self) -> bool:
        # SAC 是 step-based，每 10k 步打印一次
        if self.num_timesteps % 10_000 == 0:
            buf = self.model.ep_info_buffer
            if buf:
                rew_mean = float(np.mean([ep["r"] for ep in buf]))
                len_mean = float(np.mean([ep["l"] for ep in buf]))
                now_str  = datetime.now().isoformat(timespec="seconds")
                print(f"  [humanoid_sac] {self.num_timesteps/1e6:.2f}M / "
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

        return True

    def _write_live_dashboard(self, rew_mean: float, len_mean: float, now_str: str) -> None:
        spec = {
            "title": "人形机器人训练 SAC（RL Zoo 调优，目标2M步→500+）",
            "run_id": "humanoid_sac",
            "updated_at": now_str,
            "progress": {
                "current_steps": self.num_timesteps,
                "total_steps": self.total_steps,
            },
            "stages": [{
                "name": "full",
                "label": "SAC行走",
                "done": self.num_timesteps >= self.total_steps,
                "reward": round(rew_mean, 1),
                "note": "SAC off-policy，样本效率5-10x PPO",
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


def train(total_steps: int, n_envs: int, resume_path: str | None = None,
          healthy_z_min: float = 1.0, version: str = "v1") -> None:
    out_dir = TRAINED_DIR_V2 if version == "v2" else TRAINED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    healthy_z_range = (healthy_z_min, 2.1)

    print(f"{'='*60}")
    print(f"人形机器人训练 SAC — {version.upper()}")
    print(f"  步数: {total_steps/1e6:.1f}M  并行环境: {n_envs}")
    print(f"  算法: SAC（off-policy，样本效率5-10x PPO）")
    print(f"  环境: Gymnasium Humanoid-v4（376 维 obs）")
    print(f"  healthy_z_range: {healthy_z_range}  ← 直立高度约束")
    print(f"    v1 默认=(1.0,2.0)，弯腰到1m也算健康（鬼畜步态根因）")
    print(f"    v2 改为=(1.3,2.1)，必须保持躯干≥1.3m，强迫直立")
    print(f"  SDE: True  ent_coef: auto  buffer: 1M")
    print(f"  输出: {out_dir}")
    if resume_path:
        print(f"  热启: {resume_path}")
    print(f"{'='*60}")

    vec_env = make_vec_env(
        "Humanoid-v4",
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"healthy_z_range": healthy_z_range},
    )

    if resume_path:
        print(f"热启加载: {resume_path}")
        model = SAC.load(resume_path, env=vec_env, device="cpu")
        reset_num = True   # v2 是新任务，步数从零重计
        learn_steps = total_steps
    else:
        print("从零初始化 SAC 策略网络...")
        model = SAC(
            "MlpPolicy",
            vec_env,
            device="cpu",
            **SAC_KWARGS,
        )
        reset_num = True
        learn_steps = total_steps

    cb   = HumanoidSACCallback(total_steps, out_dir)
    ckpt = CheckpointCallback(
        save_freq=max(200_000 // n_envs, 1),
        save_path=str(out_dir / "checkpoints"),
        name_prefix="humanoid_sac",
        verbose=0,
    )

    print("开始训练（SAC：每步都学习，收敛快）...")
    model.learn(
        total_timesteps=learn_steps,
        callback=[cb, ckpt],
        reset_num_timesteps=reset_num,
        progress_bar=True,
    )

    model.save(out_dir / "final.zip")
    print(f"训练完成！模型: {out_dir}/final.zip")
    vec_env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="人形机器人 SAC 训练")
    parser.add_argument("--steps",        type=int,   default=5_000_000, help="训练总步数（默认5M）")
    parser.add_argument("--envs",         type=int,   default=4,         help="并行环境数（SAC 建议 1-4）")
    parser.add_argument("--resume",       type=str,   default=None,      help="热启模型 zip 路径")
    parser.add_argument("--healthy-z",    type=float, default=1.0,       help="躯干最低健康高度（默认1.0，v2 建议1.3）")
    parser.add_argument("--version",      type=str,   default="v1",      help="版本标识（v1/v2），决定输出目录")
    args = parser.parse_args()
    train(args.steps, args.envs, args.resume, args.healthy_z, args.version)


if __name__ == "__main__":
    main()
