#!/usr/bin/env python3
"""人形机器人 PPO 训练脚本（第一版）。

目标：让 MuJoCo 人形机器人学会向 x 方向行走。
训练规模：5M 步 × 8 并行环境（M4 Pro，MuJoCo 比 PyBullet 轻，8 个够用）

对比 train_backflip.py 的主要差异：
    - 环境：HumanoidEnv（MuJoCo）而非 BittleBackflipEnv（PyBullet）
    - 阶段：1 个阶段（full），无课程学习
    - 步数：5M（第一版先跑通，验证能学会走）
    - 输出：trained/humanoid_v1/

用法：
    conda activate jprobot
    python scripts/train_humanoid.py
    python scripts/train_humanoid.py --envs 4     # 减少 CPU 占用
    python scripts/train_humanoid.py --steps 10   # 快速冒烟测试（单位 M）
    python scripts/train_humanoid.py --resume trained/humanoid_v1/best.zip
"""

import os
# macOS 上 MuJoCo 与 OpenMP 双库冲突修复
# 根因：MuJoCo 自带 libomp，SubprocVecEnv 子进程再次加载时报冲突崩溃
# 参考：https://github.com/openai/mujoco-py/issues/268
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
from jprobot.training.env_humanoid import HumanoidEnv

# ── 目录 ─────────────────────────────────────────────────────
TRAINED_DIR = Path(__file__).parent.parent / "trained" / "humanoid_v1"

# ── 训练配置 ─────────────────────────────────────────────────
# net_arch=[256,256]：与四足 env 保持一致，方便横向对比
# ent_coef=0.005：轻微探索（人形需要比后空翻更多探索，但比随机行走少）
PPO_KWARGS = dict(
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    learning_rate=3e-4,
    ent_coef=0.005,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    policy_kwargs=dict(net_arch=[256, 256]),
    seed=42,
)


class HumanoidProgressCallback(BaseCallback):
    """每个 rollout 打印进度，并写 live_dashboard.json + metrics_history.jsonl。

    live_dashboard.json 的格式与 train_backflip.py 完全一致，
    Dashboard 会自动识别最新文件并显示。
    """

    def __init__(self, total_steps: int, out_dir: Path):
        super().__init__(verbose=0)
        self.total_steps = total_steps
        self.out_dir = out_dir
        self._best_reward = -np.inf
        self._last_written_at: str | None = None

    def _on_rollout_end(self) -> None:
        buf = self.model.ep_info_buffer
        if not buf:
            return

        rew_mean = float(np.mean([ep["r"] for ep in buf]))
        len_mean = float(np.mean([ep["l"] for ep in buf]))
        progress = self.num_timesteps / max(1, self.total_steps)
        now_str = datetime.now().isoformat(timespec="seconds")

        # 每 10 万步打印一次
        if self.num_timesteps % 100_000 < 2048:
            print(
                f"  [humanoid] {self.num_timesteps/1e6:.2f}M / "
                f"{self.total_steps/1e6:.1f}M  "
                f"rew={rew_mean:.1f}  ep_len={len_mean:.0f}"
            )

        # 写 progress.json（供历史兼容）
        progress_data = {
            "stage": "full",
            "timesteps": self.num_timesteps,
            "total_timesteps": self.total_steps,
            "progress": progress,
            "ep_rew_mean": rew_mean,
            "ep_len_mean": len_mean,
            "updated_at": now_str,
        }
        try:
            with open(self.out_dir / "progress.json", "w") as f:
                json.dump(progress_data, f, indent=2)
        except OSError:
            pass

        # 写 live_dashboard.json（Dashboard 识别）
        if now_str != self._last_written_at:
            self._last_written_at = now_str
            self._write_live_dashboard(rew_mean, len_mean, now_str)

        # 保存最佳模型
        if rew_mean > self._best_reward:
            self._best_reward = rew_mean
            self.model.save(self.out_dir / "best.zip")

    def _write_live_dashboard(self, rew_mean: float, len_mean: float, now_str: str) -> None:
        """写 trained/humanoid_v1/live_dashboard.json + 追加 metrics_history.jsonl。"""

        spec = {
            "title": "人形机器人训练 v1",
            "run_id": "humanoid_v1",
            "updated_at": now_str,
            "progress": {
                "current_steps": self.num_timesteps,
                "total_steps": self.total_steps,
            },
            # 单阶段训练，只有 full 阶段
            "stages": [
                {
                    "name": "full",
                    "label": "行走",
                    "done": self.num_timesteps >= self.total_steps,
                    "reward": round(rew_mean, 1),
                    "note": "",
                }
            ],
            # 指标卡片（与 Dashboard 期望格式完全一致）
            "metrics": [
                {"label": "Reward",   "sublabel": "当前奖励",  "value": round(rew_mean, 1), "color": "green"},
                {"label": "ep_len",   "sublabel": "每局步数",  "value": round(len_mean, 0), "color": "orange"},
                {"label": "Progress", "sublabel": "训练进度",
                 "value": f"{self.num_timesteps/self.total_steps*100:.1f}%",    "color": "blue"},
                {"label": "Best",     "sublabel": "历史最佳",  "value": round(self._best_reward, 1), "color": "yellow"},
            ],
            "history_file": "metrics_history.jsonl",
        }

        try:
            with open(TRAINED_DIR / "live_dashboard.json", "w") as f:
                json.dump(spec, f, ensure_ascii=False, indent=2)
        except OSError:
            pass

        # 追加 metrics_history.jsonl（历史曲线）
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


def train(total_steps: int, n_envs: int, resume_path: Path | None) -> None:
    """训练主流程。"""

    TRAINED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{'='*60}")
    print(f"人形机器人训练 v1")
    print(f"  步数: {total_steps/1e6:.1f}M  并行环境: {n_envs}")
    print(f"  输出目录: {TRAINED_DIR}")
    if resume_path:
        print(f"  续训模型: {resume_path}")
    print(f"{'='*60}")

    # 创建并行环境
    # SubprocVecEnv 在 macOS 用 spawn，env_fn 必须可 pickle，
    # 直接传类（而非 lambda）即可。
    vec_env = make_vec_env(
        HumanoidEnv,
        n_envs=n_envs,
        env_kwargs={"render_mode": None},
        vec_env_cls=SubprocVecEnv,
    )

    # 加载或新建模型
    if resume_path and resume_path.exists():
        print(f"加载续训模型: {resume_path}")
        model = PPO.load(resume_path, env=vec_env, device="cpu", **{
            k: v for k, v in PPO_KWARGS.items()
            if k in ("ent_coef",)  # 只更新可在 load 时覆盖的参数
        })
    else:
        print("从头初始化模型")
        model = PPO("MlpPolicy", vec_env, verbose=0, device="cpu", **PPO_KWARGS)

    # Callbacks
    progress_cb = HumanoidProgressCallback(total_steps, TRAINED_DIR)
    checkpoint_cb = CheckpointCallback(
        save_freq=max(500_000 // n_envs, 1),
        save_path=str(TRAINED_DIR / "checkpoints"),
        name_prefix="humanoid",
        verbose=0,
    )

    print("开始训练...")
    model.learn(
        total_timesteps=total_steps,
        callback=[progress_cb, checkpoint_cb],
        reset_num_timesteps=(resume_path is None),
        progress_bar=True,
    )

    # 保存最终模型
    final_path = TRAINED_DIR / "final.zip"
    model.save(final_path)
    vec_env.close()

    print(f"\n训练完成！")
    print(f"  最佳奖励: {progress_cb._best_reward:.1f}")
    print(f"  最佳模型: {TRAINED_DIR / 'best.zip'}")
    print(f"  最终模型: {final_path}")


def main():
    parser = argparse.ArgumentParser(description="人形机器人 PPO 训练（第一版）")
    parser.add_argument(
        "--steps", type=float, default=5.0,
        help="训练步数（单位 M，默认 5.0 = 5M 步）",
    )
    parser.add_argument(
        "--envs", type=int, default=8,
        help="并行环境数量（默认 8，MuJoCo 比 PyBullet 轻）",
    )
    parser.add_argument(
        "--resume", type=Path, default=None,
        help="从指定模型续训（例如 trained/humanoid_v1/best.zip）",
    )
    args = parser.parse_args()

    total_steps = int(args.steps * 1_000_000)
    train(total_steps, args.envs, args.resume)


if __name__ == "__main__":
    main()
