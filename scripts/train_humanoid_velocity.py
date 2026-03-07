#!/usr/bin/env python3
"""人形机器人速度命令跟随训练 — 万向行走（前后左右转）。

## 本脚本做什么

训练一个"万向行走"策略：接受速度命令 (vx_cmd, vy_cmd, wz_cmd)，
机器人能按命令向前后左右行走、原地转弯。

一个模型替代多个单向策略，是机器人落地部署的必备能力。

## 与 train_humanoid_sac.py 的区别

| 维度 | train_humanoid_sac.py | 本脚本 |
|------|----------------------|--------|
| 环境 | Gymnasium Humanoid-v4（376 维，固定前进） | HumanoidVelocityEnv（379 维，速度命令） |
| 奖励 | 固定前进奖励 1.25*vx | 速度追踪 3.0*exp(-err²/0.25) |
| 总步数 | 2-5M | 3M（要学三个方向，需要更多步） |
| VecEnv 创建 | make_vec_env("Humanoid-v4") | make_vec_env(lambda: ...) |
| 输出目录 | trained/humanoid_sac/ | trained/humanoid_velocity/ |

## SAC 超参复用（已验证有效）

与 train_humanoid_sac.py 完全相同的超参：
  batch_size=512, buffer_size=1M, learning_starts=10k
  use_sde=True, ent_coef='auto', lr=linear_schedule(7.3e-4)
  net_arch=[400, 300], log_std_init=-3.67

## 用法

    conda activate jprobot
    python scripts/train_humanoid_velocity.py           # 默认 3M 步
    python scripts/train_humanoid_velocity.py --steps 5000  # 快速验证
    python scripts/train_humanoid_velocity.py --steps 5000000 --envs 4

## 预期过程

  前 3 万步：ep_len ≈ 20-30（比 forward-only 早期更低，同时学 3 个方向）
  30 万步时：ep_len > 100（开始学会在命令方向行走）
  3M 步时：ep_len > 300，vel_reward 稳定增长
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

from jprobot.training.env_humanoid_velocity import HumanoidVelocityEnv

TRAINED_DIR = Path(__file__).parent.parent / "trained" / "humanoid_velocity"


def linear_schedule(initial_value: float):
    """SAC 的线性学习率衰减：从 initial_value 线性衰减到 0。

    progress_remaining 从 1.0（训练开始）降到 0.0（训练结束）。
    乘以 initial_value 就是当前学习率。
    """
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


# ── SAC 超参（完全复用 train_humanoid_sac.py，已验证有效）──────────────
SAC_KWARGS = dict(
    batch_size=512,           # replay buffer 每次采样 512 条经验训练
    buffer_size=1_000_000,    # 存 100 万步经验（off-policy 核心：可反复学习）
    learning_starts=10_000,   # 先收集 1 万步探索经验再开始学习
    gradient_steps=1,         # 每步环境交互做 1 次梯度更新
    train_freq=1,             # 每 1 步触发 1 次训练
    ent_coef="auto",          # 自动调节熵系数（自适应探索）
    gamma=0.99,               # 折扣因子（重视长期收益）
    tau=0.005,                # 目标网络软更新速率
    learning_rate=linear_schedule(7.3e-4),
    use_sde=True,             # 状态相关探索（Humanoid 关键，固定噪声不够）
    sde_sample_freq=64,       # 每 64 步重采样探索噪声
    policy_kwargs=dict(
        log_std_init=-3.67,   # 初始动作标准差 ≈ 0.025（精确动作，不乱抖）
        net_arch=[400, 300],  # 更大的网络（复杂任务需要更多容量）
    ),
    seed=42,
    verbose=0,
)


class VelocityTrackingCallback(BaseCallback):
    """训练回调：打印日志 + 写 Dashboard 数据。"""

    def __init__(self, total_steps: int, out_dir: Path):
        super().__init__(verbose=0)
        self.total_steps  = total_steps
        self.out_dir      = out_dir
        self._best_reward = -np.inf
        self._last_written: str | None = None

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
            f"  [velocity] {self.num_timesteps/1e6:.2f}M / "
            f"{self.total_steps/1e6:.1f}M  "
            f"rew={rew_mean:.1f}  ep_len={len_mean:.0f}"
        )

        # 写进度文件（training_server.py 读这个显示 Dashboard）
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

        # 避免同一秒多次写 Dashboard（节省 IO）
        if now_str != self._last_written:
            self._last_written = now_str
            self._write_live_dashboard(rew_mean, len_mean, now_str)

        # 保存最佳模型
        if rew_mean > self._best_reward:
            self._best_reward = rew_mean
            self.model.save(self.out_dir / "best.zip")

        return True

    def _write_live_dashboard(self, rew_mean: float, len_mean: float, now_str: str) -> None:
        """写 live_dashboard.json 供 training_server.py 展示。"""
        spec = {
            "title": "人形机器人速度命令跟随（万向行走）",
            "run_id": "humanoid_velocity",
            "updated_at": now_str,
            "progress": {
                "current_steps": self.num_timesteps,
                "total_steps": self.total_steps,
            },
            "stages": [{
                "name": "full",
                "label": "速度命令跟随",
                "done": self.num_timesteps >= self.total_steps,
                "reward": round(rew_mean, 1),
                "note": "379维obs=376+[vx_cmd,vy_cmd,wz_cmd]，替换固定前进奖励",
            }],
            "metrics": [
                {"label": "Reward",    "sublabel": "当前奖励",  "value": round(rew_mean, 1),  "color": "green"},
                {"label": "ep_len",    "sublabel": "每局步数",  "value": round(len_mean, 0),  "color": "orange"},
                {"label": "Progress",  "sublabel": "训练进度",
                 "value": f"{self.num_timesteps/self.total_steps*100:.1f}%",                  "color": "blue"},
                {"label": "Best",      "sublabel": "历史最佳",  "value": round(self._best_reward, 1), "color": "yellow"},
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


def train(total_steps: int, n_envs: int, resume_path: str | None = None) -> None:
    TRAINED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{'='*65}")
    print(f"人形机器人速度命令跟随训练（万向行走）")
    print(f"  步数: {total_steps/1e6:.1f}M  并行环境: {n_envs}")
    print(f"  算法: SAC（off-policy，样本效率5-10x PPO）")
    print(f"  环境: HumanoidVelocityEnv（388 维 obs = 376 + 3 速度命令 + 9 误差历史[v7]）")
    print(f"  速度命令: vx∈[-1,2] m/s  vy∈[-0.5,0.5] m/s  wz∈[-1,1] rad/s")
    print(f"  奖励: 去掉固定前进奖励，换速度追踪 3.0*exp(-err/0.25)")
    print(f"  SDE: True  ent_coef: auto  buffer: 1M")
    print(f"  输出: {TRAINED_DIR}")
    if resume_path:
        print(f"  续训: {resume_path}")
    print()
    print(f"  【指标说明】")
    print(f"    ep_len  — 每局存活步数（目标 >300）")
    print(f"    Reward  — 速度追踪 + 存活 + 健康奖励之和")
    print(f"    Best    — 历史最高 Reward（对应 best.zip）")
    print(f"{'='*65}")

    # VecEnv 创建：Wrapper 不能直接传字符串，用 lambda 包一层
    # （lambda: HumanoidVelocityEnv() 每次调用都新建一个环境实例）
    vec_env = make_vec_env(
        lambda: HumanoidVelocityEnv(),
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,
    )

    if resume_path:
        print(f"从 checkpoint 续训（obs=388）: {resume_path}")
        # 先新建 SAC（绑定 env），再把 checkpoint 权重加载进来
        model = SAC("MlpPolicy", vec_env, device="cpu", **SAC_KWARGS)
        loaded = SAC.load(resume_path)
        model.policy.load_state_dict(loaded.policy.state_dict())
        del loaded
    else:
        print("从零初始化 SAC 策略网络（obs=388，v7 Method A）...")
        model = SAC(
            "MlpPolicy",
            vec_env,
            device="cpu",
            **SAC_KWARGS,
        )

    cb   = VelocityTrackingCallback(total_steps, TRAINED_DIR)
    ckpt = CheckpointCallback(
        save_freq=max(200_000 // n_envs, 1),
        save_path=str(TRAINED_DIR / "checkpoints"),
        name_prefix="humanoid_velocity",
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
    print(f"\n训练完成！")
    print(f"  最终模型: {TRAINED_DIR}/final.zip")
    print(f"  最佳模型: {TRAINED_DIR}/best.zip")
    vec_env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="人形机器人速度命令跟随训练（万向行走）")
    parser.add_argument("--steps", type=int, default=3_000_000,
                        help="训练总步数（默认3M，速度命令比单向更难，需要更多步）")
    parser.add_argument("--envs",  type=int, default=4,
                        help="并行环境数（SAC 建议 1-4，默认4）")
    parser.add_argument("--resume", type=str, default=None,
                        help="从已有 checkpoint 续训，传入 .zip 路径（如 trained/humanoid_velocity/best.zip）")
    args = parser.parse_args()
    train(args.steps, args.envs, resume_path=args.resume)


if __name__ == "__main__":
    main()
