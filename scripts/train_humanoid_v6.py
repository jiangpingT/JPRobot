#!/usr/bin/env python3
"""人形机器人 PPO 训练脚本 V6 — VecNormalize + RL Zoo 超参数，从零开始。

V5 崩溃根因分析（ep_len: 104 → 28，在 3.4M 步后开始）：
    203 维观测的各维度尺度差异极大：
      - qpos: 关节角 ~[-π, π]（量级 1）
      - qvel: 角速度 ~[-10, 10] rad/s（量级 10）
      - cvel: 身体帧速度 ~[-20, 20]（量级 20）
      - cfrc_ext: 接触力 ~[-500, 500] N（量级 500！）
    没有 VecNormalize 时，cfrc_ext 的梯度量级比 qpos 大 500 倍，
    PPO 更新严重偏向接触力维度，导致策略逐渐偏斜，最终崩溃。

V6 核心改动：
    1. VecNormalize（最关键！）：
       - norm_obs=True：运行均值/方差归一化所有观测维度
       - norm_reward=False：奖励不归一化（我们的奖励尺度已合理）
       - clip_obs=10.0：归一化后 clip 到 [-10, 10]，防止离群值
    2. RL Zoo 为 Gymnasium Humanoid 调优的超参数：
       - n_steps: 512（4x 更频繁更新）
       - batch_size: 64（更小 mini-batch，更多梯度更新次数）
       - ent_coef: 0.00481（比 V4 的 0.01 小 2 倍，更精确收敛）
       - learning_rate: 2.55e-4（比 V4 的 3e-4 稍低）
       - clip_range: 0.185（更保守的策略更新）
    3. 从 SCRATCH 开始：
       VecNormalize 改变了网络的有效输入分布，V4/V5 权重不兼容。
       必须从零学，但有了 VecNormalize 应能更快收敛。
    4. 保存 VecNormalize stats：
       评估时需要加载同一份归一化参数，否则输入分布不匹配。

理论预期：
    RL Zoo 的 Gymnasium Humanoid-v4 benchmark 在相同超参数下，
    10M 步内可达 ep_len=500+，奖励 ~6000。
    我们用自定义奖励（高度维持+交替步态），目标也是 ep_len > 500。

用法：
    conda activate jprobot
    python scripts/train_humanoid_v6.py             # 默认 15M 步
    python scripts/train_humanoid_v6.py --steps 5   # 快速验证（5 个 episode）
    python scripts/train_humanoid_v6.py --envs 4    # 少用 CPU
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
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).parent.parent))
from jprobot.training.env_humanoid_v4 import HumanoidEnvV4

TRAINED_DIR = Path(__file__).parent.parent / "trained" / "humanoid_v6"
VECNORM_PATH = TRAINED_DIR / "vecnormalize.pkl"  # 保存归一化统计，评估时必须加载

# ── RL Zoo 为 Gymnasium Humanoid 调优的 PPO 超参数 ─────────────────────────
# 来源：https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml
# Humanoid-v4 配置（已适配我们的环境）
PPO_KWARGS = dict(
    n_steps=512,          # 512 × 8 envs = 4096 步/更新（比 V4 的 16384 快 4 倍）
    batch_size=64,        # 更小的 mini-batch → 每次更新 4096/64=64 个 mini-batch（更充分利用数据）
    n_epochs=10,          # 每批数据重复训练 10 轮
    learning_rate=2.55e-4, # RL Zoo 调优值（略低于 V4 的 3e-4）
    ent_coef=0.00481,     # RL Zoo 调优值（比 V4 的 0.01 小，允许策略更精确收敛）
    gamma=0.99,           # 折扣因子（与 V4 相同，保守选择）
    gae_lambda=0.95,      # GAE-λ（广义优势估计的衰减系数，标准值）
    clip_range=0.185,     # PPO clip（比 V4 的 0.2 稍保守）
    max_grad_norm=0.9,    # RL Zoo 额外：梯度裁剪上限（防止大梯度更新）
    vf_coef=0.871,        # RL Zoo 额外：Critic 损失系数（Humanoid 专用调优值）
    policy_kwargs=dict(
        net_arch=[256, 256],  # 与 V4 相同（两层 256 神经元 MLP）
        log_std_init=-2,      # RL Zoo：初始动作标准差更小（exp(-2)≈0.14），动作更精确
    ),
    seed=42,
)


class HumanoidV6Callback(BaseCallback):
    """每个 rollout 打印进度，写 JSON 供监控，保存最佳模型。"""

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
            print(f"  [humanoid_v6] {self.num_timesteps/1e6:.2f}M / "
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
            # 同步保存 VecNormalize 归一化统计（评估时必须一起加载）
            self.vec_norm.save(str(VECNORM_PATH))

    def _write_live_dashboard(self, rew_mean: float, len_mean: float, now_str: str) -> None:
        spec = {
            "title": "人形机器人训练 v6（VecNormalize + RL Zoo 超参）",
            "run_id": "humanoid_v6",
            "updated_at": now_str,
            "progress": {
                "current_steps": self.num_timesteps,
                "total_steps": self.total_steps,
            },
            "stages": [{
                "name": "full",
                "label": "行走 V6",
                "done": self.num_timesteps >= self.total_steps,
                "reward": round(rew_mean, 1),
                "note": "VecNormalize+RL Zoo超参，从零开始",
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
    print(f"人形机器人训练 v6（VecNormalize + RL Zoo 超参）")
    print(f"  步数: {total_steps/1e6:.1f}M  并行环境: {n_envs}")
    print(f"  obs 维度: 203（同 V4）")
    print(f"  VecNormalize: 开（norm_obs=True，clip_obs=10.0）")
    print(f"  n_steps: {PPO_KWARGS['n_steps']}  batch_size: {PPO_KWARGS['batch_size']}")
    print(f"  ent_coef: {PPO_KWARGS['ent_coef']}  lr: {PPO_KWARGS['learning_rate']}")
    print(f"  从 SCRATCH 开始（VecNormalize 不兼容旧权重）")
    print(f"  输出: {TRAINED_DIR}")
    print(f"{'='*60}")

    vec_env = make_vec_env(
        HumanoidEnvV4,
        n_envs=n_envs,
        env_kwargs={"render_mode": None},
        vec_env_cls=SubprocVecEnv,
    )

    # VecNormalize：在 SubprocVecEnv 外层包裹归一化层
    # norm_obs=True：对每个观测维度独立做 running mean/std 归一化
    # norm_reward=False：不归一化奖励（我们的奖励尺度合理，不需要）
    # clip_obs=10.0：归一化后 clip，防止离群值（接触力偶发大值）扰乱训练
    # gamma=0.99：必须与 PPO 的 gamma 一致，用于奖励折扣（即使 norm_reward=False 也要对齐）
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        gamma=PPO_KWARGS["gamma"],
    )

    print("从零初始化策略网络（MlpPolicy）...")
    model = PPO("MlpPolicy", vec_env, verbose=0, device="cpu", **PPO_KWARGS)

    cb = HumanoidV6Callback(total_steps, TRAINED_DIR, vec_env)
    ckpt_cb = CheckpointCallback(
        save_freq=max(500_000 // n_envs, 1),
        save_path=str(TRAINED_DIR / "checkpoints"),
        name_prefix="humanoid_v6",
        verbose=0,
    )

    print("开始训练...")
    model.learn(
        total_timesteps=total_steps,
        callback=[cb, ckpt_cb],
        reset_num_timesteps=True,
        progress_bar=True,
    )

    # 训练结束：保存最终模型 + 归一化统计
    model.save(TRAINED_DIR / "final.zip")
    vec_env.save(str(TRAINED_DIR / "vecnormalize_final.pkl"))
    print(f"训练完成！模型: {TRAINED_DIR}/final.zip")
    print(f"VecNormalize: {TRAINED_DIR}/vecnormalize_final.pkl")
    vec_env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="人形机器人 PPO 训练 V6")
    parser.add_argument("--steps", type=int, default=15_000_000, help="训练总步数")
    parser.add_argument("--envs",  type=int, default=8,           help="并行环境数")
    args = parser.parse_args()
    train(args.steps, args.envs)


if __name__ == "__main__":
    main()
