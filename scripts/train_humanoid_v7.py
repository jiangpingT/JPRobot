#!/usr/bin/env python3
"""人形机器人 PPO 训练脚本 V7 — 原生 Gymnasium Humanoid-v4 + VecNormalize + RL Zoo 超参。

V6 失败根因（ep_len 卡在 100-125，增速 2.3/M，需要 170M 步才能到 500）：
    我们的 203 维观测跳过了关键的 cinert（130 维，质量/惯性矩阵）：
      - cinert：每个身体段在全局坐标系下的质量、质心、惯性矩阵 10 个量
      - 没有 cinert：agent 不知道"躯干比四肢重 20 倍"，无法规划正确的动量传递
      - 没有 qfrc_act（17 维，执行力矩）：agent 不知道上一步实际施加了多大力矩

V7 核心改动：
    直接用 Gymnasium 原生 Humanoid-v4 环境（gymnasium.make("Humanoid-v4")）：
      - 观测空间：376 维（完整状态，Deepmind 专为 humanoid 设计）
        qpos(22) + qvel(23) + cinert(130) + cvel(78) + qfrc_act(17) + cfrc_ext(84) + clip剩余(2)
      - 奖励：proven 标准奖励（存活+前进+控制惩罚+接触惩罚）
        healthy_reward=5.0, forward_weight=1.25, ctrl_cost=0.1, contact_cost=5e-7
      - 已知基准：RL Zoo PPO 在 Gymnasium Humanoid-v4 上，10M 步内可达 ep_len=500+
    超参数：完全照搬 RL Zoo 的 Humanoid 调优配置（与 V6 相同）
    归一化：VecNormalize（同 V6，防止 376 维不同尺度的梯度混乱）

为什么放弃自定义奖励：
    Height reward + alternating_gait 的设计思路是对的，
    但缺少 cinert 的 203 维 obs 让 agent 无法充分利用这些奖励信号，
    导致策略停留在"延迟摔倒"（110步）而不是"真正行走"（500步+）。
    Gymnasium 标准奖励 + 标准 obs 已被证明可以产生真实行走，优先采用。

基准参考（来自 RL Zoo benchmark）：
    - 5M 步时：ep_len ~300-400
    - 10M 步时：ep_len ~500+，奖励 ~6000
    本次目标：15M 步，验收标准 ep_len > 500

用法：
    conda activate jprobot
    python scripts/train_humanoid_v7.py             # 默认 15M 步
    python scripts/train_humanoid_v7.py --steps 5   # 快速验证
    python scripts/train_humanoid_v7.py --envs 4    # 少用 CPU
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import gymnasium as gym
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).parent.parent))

TRAINED_DIR   = Path(__file__).parent.parent / "trained" / "humanoid_v7"
VECNORM_PATH  = TRAINED_DIR / "vecnormalize.pkl"

# ── RL Zoo 为 Gymnasium Humanoid-v4 调优的完整 PPO 超参数 ─────────────────
# 来源：https://github.com/DLR-RM/rl-baselines3-zoo
# 这套配置在 Gymnasium Humanoid-v4 上已验证 10M 步内可达 ep_len=500+
PPO_KWARGS = dict(
    n_steps=512,           # 512 × 8 envs = 4096 步/更新（4x 更频繁）
    batch_size=64,         # 更小 mini-batch，更多梯度更新（4096/64=64次/rollout）
    n_epochs=10,
    learning_rate=2.55e-4, # RL Zoo 调优值
    ent_coef=0.00481,      # RL Zoo 调优值（V4 的 0.01 太大）
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.185,      # RL Zoo 调优值（保守）
    max_grad_norm=0.9,     # RL Zoo 额外：梯度裁剪
    vf_coef=0.871,         # RL Zoo 额外：Critic 损失系数（Humanoid 专用）
    policy_kwargs=dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        log_std_init=-2,      # 初始动作更精确（exp(-2)≈0.14 标准差）
        activation_fn=nn.Tanh,  # RL Zoo 关键：Tanh 比 ReLU 更稳定（有界 [-1,1]，防梯度爆炸）
        ortho_init=False,       # RL Zoo 关键：Humanoid 不用正交初始化，标准随机初始化更好
    ),
    seed=42,
)


class HumanoidV7Callback(BaseCallback):
    """每个 rollout 打印进度，写 JSON，保存最佳模型 + VecNormalize。"""

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

        if self.num_timesteps % 100_000 < 512 * 8:
            print(f"  [humanoid_v7] {self.num_timesteps/1e6:.2f}M / "
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
            "title": "人形机器人训练 v7（Gymnasium Humanoid-v4 原生）",
            "run_id": "humanoid_v7",
            "updated_at": now_str,
            "progress": {
                "current_steps": self.num_timesteps,
                "total_steps": self.total_steps,
            },
            "stages": [{
                "name": "full",
                "label": "行走 V7",
                "done": self.num_timesteps >= self.total_steps,
                "reward": round(rew_mean, 1),
                "note": "Gymnasium原生obs(376维)+VecNormalize+RL Zoo超参",
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
    print(f"人形机器人训练 v7（Gymnasium Humanoid-v4 原生）")
    print(f"  步数: {total_steps/1e6:.1f}M  并行环境: {n_envs}")
    print(f"  环境: gymnasium Humanoid-v4（376 维 obs）")
    print(f"  VecNormalize: 开（norm_obs=True，clip_obs=10.0）")
    print(f"  n_steps: {PPO_KWARGS['n_steps']}  batch_size: {PPO_KWARGS['batch_size']}")
    print(f"  ent_coef: {PPO_KWARGS['ent_coef']}  lr: {PPO_KWARGS['learning_rate']}")
    print(f"  从 SCRATCH 开始")
    print(f"  输出: {TRAINED_DIR}")
    print(f"  activation: Tanh  ortho_init: False  norm_reward: True")
    print(f"  基准：RL Zoo 10M 步 → ep_len=500+")
    print(f"{'='*60}")

    vec_env = make_vec_env(
        "Humanoid-v4",
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,
    )

    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,   # RL Zoo 关键：归一化奖励防止 Critic 被大值压垮（1000步×5.0=5000）
        clip_obs=10.0,
        gamma=PPO_KWARGS["gamma"],
    )

    print("从零初始化策略网络（MlpPolicy，376维输入）...")
    model = PPO("MlpPolicy", vec_env, verbose=0, device="cpu", **PPO_KWARGS)

    cb    = HumanoidV7Callback(total_steps, TRAINED_DIR, vec_env)
    ckpt  = CheckpointCallback(
        save_freq=max(500_000 // n_envs, 1),
        save_path=str(TRAINED_DIR / "checkpoints"),
        name_prefix="humanoid_v7",
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
    parser = argparse.ArgumentParser(description="人形机器人 PPO 训练 V7")
    parser.add_argument("--steps", type=int, default=15_000_000, help="训练总步数")
    parser.add_argument("--envs",  type=int, default=8,           help="并行环境数")
    args = parser.parse_args()
    train(args.steps, args.envs)


if __name__ == "__main__":
    main()
