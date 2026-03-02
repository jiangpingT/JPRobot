#!/usr/bin/env python3
"""人形机器人直立行走训练脚本 — 阶梯式 healthy_z 课程。

## 为什么不能直接从 z=1.0 跳到 z=1.3

v1 的弯腰步态躯干高度 z≈1.17m，直接改 healthy_z=1.3 等于强制要求
机器人在 10 步内提升 0.13m 的站姿高度——这对 SAC 来说梯度几乎为零，
430k 步 ep_len 仍是 10.79 证明了这一点。

## 阶梯方案

每个阶段都从上一阶段最好模型热启，逐步提高高度要求：

  阶段A (z≥1.1, 1M步): v1 策略 z=1.17 勉强能通过，
                         agent 学会"稍微挺直一点点"才能活更久
  阶段B (z≥1.2, 2M步): 中间台阶，弯腰走不行了，
                         agent 必须明显挺直
  阶段C (z≥1.3, 2M步): 最终目标，接近真正直立

## 奖励函数增强（软约束，配合硬约束）

在原始 healthy_reward + forward_reward 基础上，加一层 RewardWrapper：
  height_bonus = max(0, torso_z - z_target) * W_HEIGHT
  upright_bonus = torso_z / 1.4 * W_UPRIGHT

软约束的好处：即使 z 没到硬门槛，也有连续梯度引导 agent 往上走。

用法：
    conda activate jprobot
    python scripts/train_humanoid_upright.py          # 完整三阶段课程
    python scripts/train_humanoid_upright.py --stage B  # 只跑 B 阶段
    python scripts/train_humanoid_upright.py --stage B --resume trained/humanoid_upright/A/best.zip
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

sys.path.insert(0, str(Path(__file__).parent.parent))

TRAINED_DIR   = Path(__file__).parent.parent / "trained" / "humanoid_upright"
V1_BEST       = Path(__file__).parent.parent / "trained" / "humanoid_sac" / "best.zip"

# ── 阶梯课程定义 ──────────────────────────────────────────────────────────
# (阶段名, healthy_z_min, 训练步数, 高度软奖励权重)
CURRICULUM = [
    ("A",  1.1,  1_000_000, 2.0),  # v1 z=1.17 勉强能活，让 agent 学"稍微挺直"
    ("B",  1.2,  2_000_000, 5.0),  # 明显需要挺直，软奖励更强
    ("B2", 1.25, 1_000_000, 6.5),  # 中间台阶：B→C 跨度太大，加 1.25 过渡
    ("C",  1.3,  2_000_000, 8.0),  # 硬约束版（已验证失败，ep_len=10）
    ("C2", 1.1,  3_000_000, 20.0), # 软约束版：宽松生存门槛 + 超强软奖励推向 z=1.3
]
STAGE_NAMES = [s for s, _, _, _ in CURRICULUM]

SAC_KWARGS = dict(
    batch_size=512,
    buffer_size=1_000_000,
    learning_starts=10_000,
    gradient_steps=1,
    train_freq=1,
    ent_coef="auto",
    gamma=0.99,
    tau=0.005,
    learning_rate=3e-4,           # 热启后用固定 lr（比原来的线性衰减更稳）
    use_sde=True,
    sde_sample_freq=64,
    policy_kwargs=dict(
        log_std_init=-3.67,
        net_arch=[400, 300],
    ),
    seed=42,
    verbose=0,
)


class UprightRewardWrapper(gym.Wrapper):
    """在原始奖励基础上加入直立软约束。

    原始 Humanoid-v4 奖励 = forward_reward + healthy_reward + ctrl_cost
    新增：
      height_bonus  = max(0, torso_z - 1.0) * w_height
                      torso_z 越高，奖励越多（线性正向激励）
      upright_bonus = torso_z / 1.4 * w_upright
                      鼓励躯干保持在正常直立高度（1.4m）附近
    """

    def __init__(self, env: gym.Env, w_height: float = 5.0, w_upright: float = 3.0):
        super().__init__(env)
        self.w_height  = w_height
        self.w_upright = w_upright

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Humanoid-v4 obs[0] = torso z 坐标（相对地面高度，直立约 1.3-1.4m）
        torso_z = float(obs[0])
        height_bonus  = max(0.0, torso_z - 1.0) * self.w_height
        upright_bonus = (torso_z / 1.4) * self.w_upright
        reward = reward + height_bonus + upright_bonus
        return obs, reward, terminated, truncated, info


class UprightCallback(BaseCallback):
    def __init__(self, stage: str, total_steps: int, out_dir: Path):
        super().__init__(verbose=0)
        self.stage       = stage
        self.total_steps = total_steps
        self.out_dir     = out_dir
        self._best       = -np.inf

    def _on_step(self) -> bool:
        if self.num_timesteps % 5_000 == 0:
            buf = self.model.ep_info_buffer
            n_ep = len(buf)
            now  = datetime.now().isoformat(timespec="seconds")
            if buf:
                rew  = float(np.mean([e["r"] for e in buf]))
                llen = float(np.mean([e["l"] for e in buf]))
                print(f"  [{self.stage}] {self.num_timesteps/1e6:.2f}M/"
                      f"{self.total_steps/1e6:.1f}M  "
                      f"rew={rew:.0f}  ep_len={llen:.0f}  buf={n_ep}")
                try:
                    with open(self.out_dir / "progress.json", "w") as f:
                        json.dump({
                            "stage": self.stage,
                            "timesteps": self.num_timesteps,
                            "total_timesteps": self.total_steps,
                            "progress": self.num_timesteps / self.total_steps,
                            "ep_rew_mean": rew,
                            "ep_len_mean": llen,
                            "updated_at": now,
                        }, f, indent=2)
                except OSError:
                    pass
                if rew > self._best:
                    self._best = rew
                    self.model.save(self.out_dir / "best.zip")
            else:
                print(f"  [{self.stage}] {self.num_timesteps} steps, buf空 ({now})")
        return True


def make_env(healthy_z_min: float, w_height: float, w_upright: float):
    """工厂函数：创建带软奖励的 Humanoid 环境。"""
    def _make():
        env = gym.make(
            "Humanoid-v4",
            healthy_z_range=(healthy_z_min, 2.1),
        )
        env = UprightRewardWrapper(env, w_height=w_height, w_upright=w_upright)
        # Monitor 是必须的：它在每个 episode 结束时在 info 里写入 "episode" key，
        # SB3 的 ep_info_buffer 依赖这个 key 来统计平均奖励/步长。
        # 不加 Monitor → ep_info_buffer 永远为空 → 回调无法输出任何进度。
        return Monitor(env)
    return _make


def train_stage(stage: str, healthy_z_min: float, total_steps: int,
                w_height: float, resume_path: Path | None, n_envs: int) -> Path:
    out_dir = TRAINED_DIR / stage
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"直立训练 — 阶段 {stage}")
    print(f"  healthy_z ≥ {healthy_z_min}m  步数: {total_steps/1e6:.1f}M  n_envs: {n_envs}")
    print(f"  软奖励: height_bonus×{w_height}  upright_bonus×{w_height*0.6:.1f}")
    if resume_path:
        print(f"  热启: {resume_path}")
    print(f"  输出: {out_dir}")
    print('='*60)

    vec_env = SubprocVecEnv([
        make_env(healthy_z_min, w_height, w_height * 0.6)
        for _ in range(n_envs)
    ])

    if resume_path and resume_path.exists():
        print(f"热启：创建新模型 + 复制策略权重自 {resume_path}")
        # 用新环境创建全新 SAC（不调 SAC.load，避免 macOS forkserver 下的卡死）
        model = SAC("MlpPolicy", vec_env, device="cpu", **SAC_KWARGS)
        # 只加载策略权重（actor / critic / log_std）
        _src = SAC.load(resume_path, device="cpu")  # 无 env → 轻量加载
        model.policy.load_state_dict(_src.policy.state_dict())
        del _src
        print("策略权重已复制，开始训练")
    else:
        print("从零初始化 SAC...")
        model = SAC("MlpPolicy", vec_env, device="cpu", **SAC_KWARGS)

    cb   = UprightCallback(stage, total_steps, out_dir)
    ckpt = CheckpointCallback(
        save_freq=max(200_000 // n_envs, 1),
        save_path=str(out_dir / "checkpoints"),
        name_prefix=f"upright_{stage}",
        verbose=0,
    )

    model.learn(
        total_timesteps=total_steps,
        callback=[cb, ckpt],
        reset_num_timesteps=True,
        progress_bar=False,
    )

    model.save(out_dir / "final.zip")
    print(f"\n阶段 {stage} 完成！最佳模型: {out_dir / 'best.zip'}")
    vec_env.close()
    return out_dir / "best.zip"


def main():
    parser = argparse.ArgumentParser(description="人形机器人直立行走课程训练")
    parser.add_argument("--stage",  choices=STAGE_NAMES + ["all"], default="all",
                        help="训练阶段（A/B/C 或 all）")
    parser.add_argument("--resume", type=Path, default=None,
                        help="指定热启模型（仅对 --stage 指定阶段生效）")
    parser.add_argument("--envs",   type=int,  default=4,
                        help="并行环境数（SAC 建议 1-4）")
    args = parser.parse_args()

    TRAINED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"直立训练目录: {TRAINED_DIR}")
    print(f"阶梯课程: A(z≥1.1, 1M) → B(z≥1.2, 2M) → C(z≥1.3, 2M)")

    if args.stage == "all":
        # 从 v1 best.zip 开始热启
        prev = args.resume or (V1_BEST if V1_BEST.exists() else None)
        if prev:
            print(f"从 v1 best.zip 热启: {prev}")
        for stage, z_min, steps, w_h in CURRICULUM:
            prev = train_stage(stage, z_min, steps, w_h, prev, args.envs)
        print("\n完整课程训练结束！最终模型:", TRAINED_DIR / "C" / "best.zip")
    else:
        cfg = {s: (z, t, w) for s, z, t, w in CURRICULUM}
        z_min, steps, w_h = cfg[args.stage]
        resume = args.resume
        if resume is None and args.stage != "A":
            prev_stage = STAGE_NAMES[STAGE_NAMES.index(args.stage) - 1]
            auto = TRAINED_DIR / prev_stage / "best.zip"
            if auto.exists():
                resume = auto
                print(f"自动加载上一阶段: {resume}")
        if resume is None and args.stage == "A":
            resume = V1_BEST if V1_BEST.exists() else None
        train_stage(args.stage, z_min, steps, w_h, resume, args.envs)


if __name__ == "__main__":
    main()
