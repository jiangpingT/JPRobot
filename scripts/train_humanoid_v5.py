#!/usr/bin/env python3
"""人形机器人 PPO 训练脚本 V5 — 正确超参数 + V4 热启动。

V4 问题：ep_len 卡在 ~100（3M 步后无突破）。
根因：超参数完全错误。n_steps=2048 导致每次更新间隔太长；
      gamma=0.99 导致 agent 过于重视遥远的未来（对不稳定任务有害）。

V5 核心改动：
    1. 超参数：完全换用 RL Zoo 为 Gymnasium Humanoid 调优的配置
       - n_steps: 2048 → 512     （4倍更频繁的更新）
       - n_epochs: 10 → 20       （每批数据训练更充分）
       - gamma: 0.99 → 0.95      （更关注近期平衡，适合不稳定任务）
       - ent_coef: 0.01 → 0.0004 （更小，允许策略精确收敛）
       - clip_range: 0.2 → 0.1   （更保守的策略更新）
       - lr: 3e-4 → 2e-4         （更稳定的学习）
       - log_std_init: -2        （初始动作更精确，避免乱动）
    2. 从 V4 best.zip 热启动（obs 相同 203 维，直接兼容）
    3. 同样的环境（HumanoidEnvV4，保留 V3 奖励）

为什么这些超参数对 Humanoid 至关重要：
    - n_steps=512：人形机器人 ep_len ~100 步，2048 steps/update 意味着每次更新
      跨越 20 个 episode，梯度信号被平均得太稀疏。512 步让 agent 每 5 个 episode
      就更新一次，梯度信号更集中。
    - gamma=0.95：Humanoid 摔倒通常在 100 步内，0.99 折扣使第 100 步的惩罚
      几乎没有影响（0.99^100 = 0.37 折扣），0.95 则更直接（0.95^100 = 0.006）。
    - normalize_advantage=False：Humanoid 的奖励尺度固定（healthy=5），
      归一化会掩盖"此 episode 明显更好"的信号。

用法：
    conda activate jprobot
    python scripts/train_humanoid_v5.py         # 默认 15M 步，从 V4 热启动
    python scripts/train_humanoid_v5.py --steps 2  # 快速验证
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

TRAINED_DIR = Path(__file__).parent.parent / "trained" / "humanoid_v5"
V4_BEST = Path(__file__).parent.parent / "trained" / "humanoid_v4" / "best.zip"

# ── V5 超参数：保守单项修改，只改 n_steps ────────────────────────────────────
# 同时改多个超参（gamma/ent_coef/normalize_advantage）会让旧 Critic 完全失配，
# 导致热启动后 ep_len 持续下降（V5_v0 实测：92→83→72，崩溃）。
#
# 最安全的做法：只改 n_steps 2048→512（最有影响的单项）
# 原理：每次更新间隔从 2048×8=16384 步降到 512×8=4096 步，
#       对于 ep_len~100 的任务，更新频率从每 160 个 ep 一次 → 每 40 个 ep 一次。
#       梯度信号更集中，更快看到平衡改进的效果。
# 其他参数保持与 V4 完全相同，确保 Critic 不失配。
PPO_KWARGS = dict(
    n_steps=512,        # 核心改动：2048→512（4x 更频繁的更新）
    batch_size=256,     # 同 V4
    n_epochs=10,        # 同 V4（保持 Critic 兼容性）
    learning_rate=3e-4, # 同 V4
    ent_coef=0.01,      # 同 V4（保持探索能力）
    gamma=0.99,         # 同 V4（防止 Critic 预测失配）
    gae_lambda=0.95,    # 同 V4
    clip_range=0.2,     # 同 V4
    policy_kwargs=dict(
        net_arch=[256, 256],   # 同 V4（热启动兼容性）
    ),
    seed=42,
)


class HumanoidV5Callback(BaseCallback):
    def __init__(self, total_steps: int, out_dir: Path, start_steps: int = 0):
        super().__init__(verbose=0)
        self.total_steps = total_steps
        self.out_dir = out_dir
        self.start_steps = start_steps   # V4 已训步数（用于显示总进度）
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
            total_so_far = self.start_steps + self.num_timesteps
            print(f"  [humanoid_v5] V5步={self.num_timesteps/1e6:.2f}M "
                  f"（总计~{total_so_far/1e6:.1f}M）  "
                  f"rew={rew_mean:.1f}  ep_len={len_mean:.0f}")

        try:
            with open(self.out_dir / "progress.json", "w") as f:
                json.dump({
                    "stage": "v5_finetune",
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
            "title": "人形机器人训练 v5（n_steps=512 + V4 热启动）",
            "run_id": "humanoid_v5",
            "updated_at": now_str,
            "progress": {
                "current_steps": self.num_timesteps,
                "total_steps": self.total_steps,
            },
            "stages": [{
                "name": "v5_finetune",
                "label": "V5精调",
                "done": self.num_timesteps >= self.total_steps,
                "reward": round(rew_mean, 1),
                "note": "n_steps=512，其他同V4，从V4 best热启动",
            }],
            "metrics": [
                {"label": "Reward",   "sublabel": "当前奖励",  "value": round(rew_mean, 1),   "color": "green"},
                {"label": "ep_len",   "sublabel": "每局步数",  "value": round(len_mean, 0),   "color": "orange"},
                {"label": "Progress", "sublabel": "V5训练进度",
                 "value": f"{self.num_timesteps/self.total_steps*100:.1f}%",                  "color": "blue"},
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
                    "timesteps": self.num_timesteps,
                    "ep_rew_mean": round(rew_mean, 1),
                    "ep_len_mean": round(len_mean, 1),
                    "stage": "v5_finetune",
                }) + "\n")
        except OSError:
            pass

    def _on_step(self) -> bool:
        return True


def train(total_steps: int, n_envs: int, start_fresh: bool) -> None:
    TRAINED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"人形机器人训练 V5（RL Zoo 超参数 + V4 热启动）")
    print(f"  V5 新步数: {total_steps/1e6:.1f}M  并行环境: {n_envs}")
    print(f"  超参变化: n_steps 2048→512（只改这一项，其他与V4相同）")
    print(f"  启动方式: {'从零开始' if start_fresh else 'V4 热启动 (best.zip)'}")
    print(f"  输出: {TRAINED_DIR}")
    print(f"{'='*60}")

    vec_env = make_vec_env(
        HumanoidEnvV4,
        n_envs=n_envs,
        env_kwargs={"render_mode": None},
        vec_env_cls=SubprocVecEnv,
    )

    start_steps = 0
    if not start_fresh and V4_BEST.exists():
        print(f"从 V4 best.zip 热启动: {V4_BEST}")
        model = PPO.load(str(V4_BEST), env=vec_env, device="cpu", **PPO_KWARGS)
        start_steps = 3_000_000   # V4 已训约 3M 步
        print(f"  热启动成功！在 V4 权重基础上用新超参继续训练。")
    else:
        print("从零初始化策略网络（V4 best 不存在或强制重启）...")
        model = PPO("MlpPolicy", vec_env, verbose=0, device="cpu", **PPO_KWARGS)

    cb = HumanoidV5Callback(total_steps, TRAINED_DIR, start_steps=start_steps)
    ckpt_cb = CheckpointCallback(
        save_freq=max(500_000 // n_envs, 1),
        save_path=str(TRAINED_DIR / "checkpoints"),
        name_prefix="humanoid_v5",
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
    parser = argparse.ArgumentParser(description="人形机器人训练 V5（RL Zoo 超参数）")
    parser.add_argument("--steps", type=float, default=15.0, help="训练步数（百万），默认15M")
    parser.add_argument("--envs",  type=int,   default=8,    help="并行环境数，默认8")
    parser.add_argument("--fresh", action="store_true",      help="从零开始（不热启动）")
    args = parser.parse_args()
    train(int(args.steps * 1_000_000), args.envs, start_fresh=args.fresh)


if __name__ == "__main__":
    main()
