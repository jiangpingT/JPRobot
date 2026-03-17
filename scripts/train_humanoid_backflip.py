#!/usr/bin/env python3
"""人形机器人后空翻训练脚本。

算法：SAC（与人形行走/跑步一致，17 DOF 连续控制 off-policy 更高效）
环境：HumanoidBackflipEnv（obs=382，Humanoid-v4 + 6 维后空翻信号）
热启：支持从 humanoid_sac/best.zip（376 维）或其他 382 维 checkpoint 热启

## 热启动观测维度迁移（376 → 382）

当从 376 维旧模型热启时，SAC 策略网络的输入层需要扩展：
  Actor 输入层：[hidden, 376] → [hidden, 382]（新增 6 列零初始化）
  Critic 输入层：[hidden, 376+17] → [hidden, 382+17]（obs 部分新增 6 列零）

权重迁移逻辑：
  1. 创建新模型（obs=382）
  2. 加载旧模型（obs=376）
  3. 对每层：shape 匹配则直接复制；输入层则填充零列

用法：
  # 从 humanoid_sac/best.zip 热启（推荐，已知会站立的基础模型）
  python scripts/train_humanoid_backflip.py --steps 5000000 \\
      --resume trained/humanoid_sac/best.zip --obs-expand 376

  # 从同维度 checkpoint 续训
  python scripts/train_humanoid_backflip.py --steps 5000000 \\
      --resume trained/humanoid_backflip/best.zip

  # 从零开始
  python scripts/train_humanoid_backflip.py --steps 5000000
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

sys.path.insert(0, str(Path(__file__).parent.parent))
from jprobot.training.env_humanoid_backflip import HumanoidBackflipEnv

TRAINED_DIR = Path(__file__).parent.parent / "trained" / "humanoid_backflip"
ACTION_DIM  = 17   # Humanoid-v4 动作维度（17 个关节执行器）
NEW_OBS_DIM = 382  # 新环境观测维度（376 + 6）


def linear_schedule(initial_value: float):
    """线性学习率调度：随训练进度从 initial_value 线性衰减到 0。"""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


# SAC 超参数（与人形跑步 run_v4 一致，已在 17 DOF 任务验证有效）
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
    use_sde=True,          # 状态相关探索（Humanoid 任务必须）
    sde_sample_freq=64,
    policy_kwargs=dict(
        log_std_init=-3.67,
        net_arch=[400, 300],
    ),
    seed=42,
    verbose=0,
)


def transfer_weights_obs_expanded(old_model: SAC, new_model: SAC,
                                  old_obs_dim: int, new_obs_dim: int,
                                  action_dim: int) -> None:
    """将旧模型权重迁移到新模型，处理 obs 维度从 old_obs_dim 扩展到 new_obs_dim。

    处理策略：
      - 形状完全匹配的层：直接复制
      - Actor 输入层 [hidden, old_obs_dim] → [hidden, new_obs_dim]：
          前 old_obs_dim 列复制，后 extra_dim 列保持零
      - Critic 输入层 [hidden, old_obs_dim+action_dim] → [hidden, new_obs_dim+action_dim]：
          obs 部分（前 old_obs_dim 列）复制，
          新增 obs 维（extra_dim 列）零初始化，
          action 部分（后 action_dim 列）复制（位置后移 extra_dim）

    为什么 extra_dim 对应的权重初始化为 0？
      新增的 6 维 obs（累积旋转、高度、速度、脚接触、成功标志）在旧模型中不存在。
      零初始化意味着这些维度初始对策略输出没有影响，
      训练过程中 SAC 会通过梯度逐渐学习如何利用这些信号。
    """
    extra = new_obs_dim - old_obs_dim
    old_sd = old_model.policy.state_dict()
    new_sd = new_model.policy.state_dict()
    transferred = 0

    for key in new_sd:
        if key not in old_sd:
            continue
        old_w = old_sd[key]
        new_w = new_sd[key]

        if old_w.shape == new_w.shape:
            # 形状匹配：直接复制（所有中间层、输出层、bias 都走这里）
            new_sd[key] = old_w.clone()
            transferred += 1

        elif len(old_w.shape) == 2:
            old_r, old_c = old_w.shape
            new_r, new_c = new_w.shape

            if old_r == new_r and new_c == old_c + extra:
                # Actor 输入层或 feature extractor 层：列数 = obs_dim
                # 情况：[hidden, old_obs_dim] → [hidden, new_obs_dim]
                t = torch.zeros_like(new_w)
                t[:, :old_c] = old_w
                new_sd[key] = t
                transferred += 1

            elif (old_r == new_r
                  and old_c == old_obs_dim + action_dim
                  and new_c == new_obs_dim + action_dim):
                # Critic 输入层：列数 = obs_dim + action_dim
                # 情况：[hidden, old_obs+act] → [hidden, new_obs+act]
                # 需要把 action 列移位 extra 个位置
                t = torch.zeros_like(new_w)
                t[:, :old_obs_dim]   = old_w[:, :old_obs_dim]   # obs 部分
                # t[:, old_obs_dim:new_obs_dim] = 0  # 新 obs 维（已是零）
                t[:, new_obs_dim:]   = old_w[:, old_obs_dim:]   # action 部分（后移）
                new_sd[key] = t
                transferred += 1

        elif len(old_w.shape) == 1 and old_w.shape == new_w.shape:
            # 1D bias：形状匹配则直接复制（已在上面处理）
            new_sd[key] = old_w.clone()
            transferred += 1

    new_model.policy.load_state_dict(new_sd)
    print(f"  权重迁移完成：{transferred}/{len(new_sd)} 层成功复制")


class BackflipTrackingCallback(BaseCallback):
    """训练回调：每 10K 步打印日志 + 写 progress.json + 保存最优模型。"""

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
            f"  [backflip] {self.num_timesteps/1e6:.2f}M / "
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


def train(total_steps: int, n_envs: int,
          resume_path: str | None = None,
          obs_expand_from: int | None = None) -> None:
    """主训练函数。

    Args:
        total_steps:      训练步数
        n_envs:           并行环境数
        resume_path:      热启 checkpoint 路径（None = 从零开始）
        obs_expand_from:  旧模型的 obs 维度（仅在 376→382 维度迁移时需要）
    """
    TRAINED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{'='*65}")
    print(f"人形机器人后空翻训练（SAC + 17 DOF Humanoid-v4）")
    print(f"  步数: {total_steps/1e6:.1f}M  并行环境: {n_envs}")
    print(f"  算法: SAC  obs: 382 维（376 Humanoid-v4 + 6 后空翻信号）")
    print(f"  目标: rotation≥286°落地，最终 rot@land=360°")
    if resume_path:
        print(f"  热启: {resume_path}")
        if obs_expand_from:
            print(f"  维度迁移: {obs_expand_from} → {NEW_OBS_DIM}")
    print()
    print(f"  【指标说明】")
    print(f"    ep_len    — 每局存活步数（目标 50-80，后空翻+站稳）")
    print(f"    Reward    — 起跳+旋转+落地+成功+站稳奖励之和")
    print(f"    rotation  — 累积旋转度数（目标 ≥ 360°）")
    print(f"    success%  — 后空翻成功率（rotation≥286°且落地）")
    print(f"{'='*65}")

    vec_env = make_vec_env(
        lambda: HumanoidBackflipEnv(),
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,
    )

    if resume_path and obs_expand_from and obs_expand_from != NEW_OBS_DIM:
        # ── obs 维度扩展热启（376 → 382）────────────────────────────────────
        print(f"obs 维度扩展热启：{obs_expand_from} → {NEW_OBS_DIM}")
        new_model = SAC("MlpPolicy", vec_env, device="cpu", **SAC_KWARGS)
        old_model = SAC.load(resume_path, device="cpu")
        transfer_weights_obs_expanded(
            old_model, new_model,
            old_obs_dim=obs_expand_from,
            new_obs_dim=NEW_OBS_DIM,
            action_dim=ACTION_DIM,
        )
        del old_model
        model = new_model

    elif resume_path:
        # ── 同维度热启（续训）────────────────────────────────────────────────
        print(f"同维度热启（obs={NEW_OBS_DIM}）: {resume_path}")
        model = SAC("MlpPolicy", vec_env, device="cpu", **SAC_KWARGS)
        loaded = SAC.load(resume_path, device="cpu")
        model.policy.load_state_dict(loaded.policy.state_dict())
        del loaded

    else:
        # ── 从零初始化 ────────────────────────────────────────────────────────
        print(f"从零初始化（obs={NEW_OBS_DIM}）...")
        model = SAC("MlpPolicy", vec_env, device="cpu", **SAC_KWARGS)

    cb   = BackflipTrackingCallback(total_steps, TRAINED_DIR)
    ckpt = CheckpointCallback(
        save_freq=max(200_000 // n_envs, 1),
        save_path=str(TRAINED_DIR / "checkpoints"),
        name_prefix="humanoid_backflip",
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
    parser = argparse.ArgumentParser(description="人形机器人后空翻训练")
    parser.add_argument("--steps",  type=int, default=5_000_000,
                        help="训练步数（默认 5M）")
    parser.add_argument("--envs",   type=int, default=4,
                        help="并行环境数（默认 4）")
    parser.add_argument("--resume", type=str, default=None,
                        help="热启 checkpoint 路径（如 trained/humanoid_sac/best.zip）")
    parser.add_argument("--obs-expand", type=int, default=None,
                        dest="obs_expand",
                        help="旧模型 obs 维度（用于维度迁移，如 --obs-expand 376）")
    args = parser.parse_args()

    # 自动推断维度迁移：如果提供了旧模型路径但未指定 obs-expand，
    # 尝试检测旧模型的 obs 维度
    obs_expand = args.obs_expand
    if args.resume and obs_expand is None:
        try:
            tmp = SAC.load(args.resume, device="cpu")
            old_obs = tmp.observation_space.shape[0]
            del tmp
            if old_obs != NEW_OBS_DIM:
                print(f"自动检测：旧模型 obs={old_obs}，新模型 obs={NEW_OBS_DIM}，将进行维度迁移")
                obs_expand = old_obs
        except Exception as e:
            print(f"警告：无法自动检测旧模型维度（{e}），将尝试同维度热启")

    train(args.steps, args.envs,
          resume_path=args.resume,
          obs_expand_from=obs_expand)


if __name__ == "__main__":
    main()
