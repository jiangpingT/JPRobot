#!/usr/bin/env python3
"""BittleX 后空翻 PPO 课程训练脚本。

后空翻训练的核心挑战：
  随机动作几乎不可能"偶然"发现完整的后空翻序列。
  解决方案：4 阶段课程学习（Curriculum Learning），逐步放开奖励目标：

  阶段 1（jump）：只奖励起跳高度 → agent 学会产生向上速度
  阶段 2（rotate）：加入旋转奖励 → agent 学会在腾空后向后旋转
  阶段 3（land）：加入落地奖励 → agent 学会旋转后回到地面
  阶段 4（full）：全部奖励 + 成功 bonus → 精化完整后空翻

每个阶段在前一阶段模型基础上继续训练（迁移学习）。

用法：
    conda activate jprobot
    python scripts/train_backflip.py                 # 完整 4 阶段课程
    python scripts/train_backflip.py --stage jump    # 只跑第一阶段
    python scripts/train_backflip.py --stage rotate --resume trained/backflip/jump/best.zip
    python scripts/train_backflip.py --envs 4        # 少用 CPU
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

sys.path.insert(0, str(Path(__file__).parent.parent))
from jprobot.training.env_backflip import BittleBackflipEnv

# ── 目录结构 ──────────────────────────────────────────────────────────────
TRAINED_DIR = Path(__file__).parent.parent / "trained" / "backflip"

# ── 课程定义 ──────────────────────────────────────────────────────────────
# 每个阶段：(training_phase, timesteps, ent_coef)
# ent_coef（熵系数）控制探索量：
#   - 初期高熵（0.02）→ 更多随机探索，有助于发现新动作
#   - 后期低熵（0.005）→ 更多利用已学策略，精化动作质量
CURRICULUM = [
    ("jump",   1_000_000, 0.02),   # 阶段1: 1M步，高探索，只学起跳
    ("rotate", 1_500_000, 0.01),   # 阶段2: 1.5M步，学旋转
    ("land",   2_000_000, 0.008),  # 阶段3: 2M步，学落地
    ("full",   2_000_000, 0.005),  # 阶段4: 2M步，精化完整后空翻
]

STAGE_ORDER = ["jump", "rotate", "land", "full"]


_STAGE_LABELS = {"jump": "起跳", "rotate": "旋转", "land": "落地", "full": "完整"}
_STAGE_TOTAL_STEPS = {
    "jump":   1_000_000,
    "rotate": 1_500_000,
    "land":   2_000_000,
    "full":   2_000_000,
}
_CURRICULUM_CUMULATIVE = {
    "jump":   1_000_000,
    "rotate": 2_500_000,
    "land":   4_500_000,
    "full":   6_500_000,
}
_TOTAL_STEPS = 6_500_000


class BackflipProgressCallback(BaseCallback):
    """每个 rollout 打印进度，并写 JSON 供监控。"""

    def __init__(self, stage: str, total_steps: int, out_dir: Path):
        super().__init__(verbose=0)
        self.stage      = stage
        self.total_steps = total_steps
        self.out_dir    = out_dir
        self._best_reward = -np.inf
        self._last_written_at: str | None = None

    def _on_rollout_end(self) -> None:
        buf = self.model.ep_info_buffer
        if not buf:
            return

        rew_mean = float(np.mean([ep["r"] for ep in buf]))
        len_mean = float(np.mean([ep["l"] for ep in buf]))
        progress = self.num_timesteps / max(1, self.total_steps)
        now_str  = datetime.now().isoformat(timespec="seconds")

        # 每 10 万步打印一次进度
        if self.num_timesteps % 100_000 < 2048:
            print(f"  [{self.stage}] {self.num_timesteps/1e6:.2f}M / "
                  f"{self.total_steps/1e6:.1f}M  "
                  f"rew={rew_mean:.1f}  ep_len={len_mean:.0f}")

        # 写进度 JSON（供旧逻辑兼容）
        data = {
            "stage": self.stage,
            "timesteps": self.num_timesteps,
            "total_timesteps": self.total_steps,
            "progress": progress,
            "ep_rew_mean": rew_mean,
            "ep_len_mean": len_mean,
            "updated_at": now_str,
        }
        try:
            with open(self.out_dir / "progress.json", "w") as f:
                json.dump(data, f, indent=2)
        except OSError:
            pass

        # 写 live_dashboard.json + 追加 metrics_history.jsonl
        if now_str != self._last_written_at:
            self._last_written_at = now_str
            self._write_live_dashboard(rew_mean, len_mean, now_str)

        # 保存最佳模型
        if rew_mean > self._best_reward:
            self._best_reward = rew_mean
            self.model.save(self.out_dir / "best.zip")

    def _write_live_dashboard(self, rew_mean: float, len_mean: float, now_str: str) -> None:
        """写 trained/backflip/live_dashboard.json 和追加 metrics_history.jsonl。"""
        base_dir = TRAINED_DIR  # trained/backflip/
        stage_order = ["jump", "rotate", "land", "full"]

        # 累计全局步数
        prev_stages = stage_order[: stage_order.index(self.stage)]
        global_steps = sum(_STAGE_TOTAL_STEPS[s] for s in prev_stages) + self.num_timesteps
        global_steps = min(global_steps, _TOTAL_STEPS)

        # 扫描各阶段 fixed_eval.json 拿历史奖励和 note
        stages_spec = []
        for s in stage_order:
            eval_path = base_dir / s / "fixed_eval.json"
            prog_path = base_dir / s / "progress.json"

            # 该阶段奖励：优先 fixed_eval，否则 progress.json，否则当前
            reward_val: float | None = None
            note = ""
            if eval_path.exists():
                try:
                    with open(eval_path) as f:
                        ev = json.load(f)
                    m = ev.get("metrics", {})
                    reward_val = m.get("mean_episode_reward")
                    rot = m.get("mean_max_rotation_deg")
                    if rot is not None:
                        note = f"rot={rot:.1f}°"
                except (OSError, json.JSONDecodeError):
                    pass
            if reward_val is None and prog_path.exists():
                try:
                    with open(prog_path) as f:
                        pg = json.load(f)
                    reward_val = pg.get("ep_rew_mean")
                except (OSError, json.JSONDecodeError):
                    pass
            if reward_val is None and s == self.stage:
                reward_val = rew_mean

            # 是否已完成
            if s == self.stage:
                done = self.num_timesteps >= self.total_steps
            else:
                done = prog_path.exists() and stage_order.index(s) < stage_order.index(self.stage)

            stages_spec.append({
                "name":   s,
                "label":  _STAGE_LABELS.get(s, s),
                "done":   done,
                "reward": round(reward_val, 1) if reward_val is not None else 0,
                "note":   note,
            })

        # metrics 卡片（从当前阶段 fixed_eval 或 progress 读）
        cur_eval_path = base_dir / self.stage / "fixed_eval.json"
        success_rate_str = "N/A"
        launch_rate_str  = "N/A"
        rotation_str     = "N/A"
        if cur_eval_path.exists():
            try:
                with open(cur_eval_path) as f:
                    ev = json.load(f)
                m = ev.get("metrics", {})
                sr = m.get("success_rate")
                lr = m.get("launch_rate")
                rot = m.get("mean_max_rotation_deg")
                if sr  is not None: success_rate_str = f"{sr*100:.0f}%"
                if lr  is not None: launch_rate_str  = f"{lr*100:.0f}%"
                if rot is not None: rotation_str     = f"{rot:.1f}°"
            except (OSError, json.JSONDecodeError):
                pass

        metrics_spec = [
            {"label": "当前奖励", "value": round(rew_mean, 1), "color": "green"},
            {"label": "旋转",     "value": rotation_str,       "color": "blue"},
            {"label": "成功率",   "value": success_rate_str,   "color": "yellow"},
            {"label": "起跳率",   "value": launch_rate_str,    "color": "green"},
        ]

        spec = {
            "title":    "后空翻训练",
            "run_id":   "backflip",
            "updated_at": now_str,
            "progress": {
                "current_steps": global_steps,
                "total_steps":   _TOTAL_STEPS,
            },
            "stages":   stages_spec,
            "metrics":  metrics_spec,
            "history_file": "metrics_history.jsonl",
        }

        try:
            with open(base_dir / "live_dashboard.json", "w") as f:
                json.dump(spec, f, ensure_ascii=False, indent=2)
        except OSError:
            pass

        # 追加 metrics_history.jsonl
        hist_path = base_dir / "metrics_history.jsonl"
        try:
            with open(hist_path, "a") as f:
                f.write(json.dumps({
                    "total_timesteps": global_steps,
                    "ep_rew_mean":     round(rew_mean, 1),
                    "ep_len_mean":     round(len_mean, 1),
                    "stage":           self.stage,
                }) + "\n")
        except OSError:
            pass

    def _on_step(self) -> bool:
        return True


def train_stage(stage: str, timesteps: int, ent_coef: float,
                resume_path: Path | None, n_envs: int) -> Path:
    """训练单个课程阶段，返回最终模型路径。"""

    out_dir = TRAINED_DIR / stage
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"阶段 [{stage}]  目标: {timesteps/1e6:.1f}M 步  "
          f"ent_coef={ent_coef}  n_envs={n_envs}")
    if resume_path:
        print(f"续训模型: {resume_path}")
    print(f"输出目录: {out_dir}")
    print('='*60)

    # 创建并行环境
    # 注意：SubprocVecEnv 在 macOS 用 spawn/forkserver，env_fn 必须可 pickle。
    # 局部 lambda/闭包不可 pickle，改用 env_class + env_kwargs 方式。
    vec_env = make_vec_env(
        BittleBackflipEnv,
        n_envs=n_envs,
        env_kwargs={"render_mode": None, "training_phase": stage},
        vec_env_cls=SubprocVecEnv,
    )

    # 加载或新建模型
    if resume_path and resume_path.exists():
        print(f"加载模型: {resume_path}")
        model = PPO.load(resume_path, env=vec_env, device="cpu", ent_coef=ent_coef)
    else:
        print("从头初始化模型")
        model = PPO(
            "MlpPolicy", vec_env,
            verbose=0,
            device="cpu",   # MPS 对 256×256 小网络比 CPU 慢 2.3×（实测）
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            learning_rate=3e-4,
            ent_coef=ent_coef,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            policy_kwargs=dict(
                net_arch=[256, 256],  # 与步态 env 一致
            ),
            seed=42,
        )

    # Callbacks
    progress_cb = BackflipProgressCallback(stage, timesteps, out_dir)
    checkpoint_cb = CheckpointCallback(
        save_freq=max(500_000 // n_envs, 1),
        save_path=str(out_dir / "checkpoints"),
        name_prefix=f"bf_{stage}",
        verbose=0,
    )

    print(f"开始训练...")
    model.learn(
        total_timesteps=timesteps,
        callback=[progress_cb, checkpoint_cb],
        reset_num_timesteps=(resume_path is None),
        progress_bar=True,
    )

    # 保存最终模型
    final_path = out_dir / "final.zip"
    model.save(final_path)
    vec_env.close()

    best_reward = progress_cb._best_reward
    print(f"\n阶段 [{stage}] 完成！  最佳奖励: {best_reward:.1f}")
    print(f"最佳模型: {out_dir / 'best.zip'}")
    return out_dir / "best.zip"


def eval_stage(stage: str, best_model: Path):
    """训练阶段结束后自动跑 fixed_eval，打印结论。"""
    from scripts.fixed_eval_backflip import run_backflip_eval, _print_summary
    print(f"\n[AutoEval] 阶段 [{stage}] 训练完成，开始验收评估（20 局，确定性推理）...")
    try:
        # 对比上次同阶段的结果
        prev_path = TRAINED_DIR / stage / "fixed_eval.json"
        prev = None
        if prev_path.exists():
            import json
            with open(prev_path) as f:
                prev = json.load(f)
        result = run_backflip_eval(best_model, stage=stage, episodes=20)
        _print_summary(result, prev)
    except Exception as e:
        print(f"[AutoEval] 验收评估失败（不影响训练）: {e}")


def main():
    parser = argparse.ArgumentParser(description="BittleX 后空翻课程训练")
    parser.add_argument("--stage", choices=STAGE_ORDER + ["all"], default="all",
                        help="要训练的阶段（默认 all = 完整课程）")
    parser.add_argument("--resume", type=Path, default=None,
                        help="从指定模型续训（仅对 --stage 指定的阶段生效）")
    parser.add_argument("--envs", type=int, default=12,
                        help="并行环境数量（默认 12，适合 M4 Pro 14 核）")
    parser.add_argument("--no-eval", action="store_true",
                        help="跳过阶段后的自动验收评估")
    args = parser.parse_args()

    TRAINED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"后空翻训练目录: {TRAINED_DIR}")
    print(f"device: cpu（MPS 对小网络反而慢 2.3×，详见 /tmp/mps_bench.py）")

    if args.stage == "all":
        prev_model = args.resume
        for stage, timesteps, ent_coef in CURRICULUM:
            best = train_stage(stage, timesteps, ent_coef, prev_model, args.envs)
            if not args.no_eval:
                eval_stage(stage, best)
            prev_model = best
        print("\n完整课程训练结束！")
        print(f"最终模型: {TRAINED_DIR / 'full' / 'best.zip'}")
        print(f"验收报告: {TRAINED_DIR / 'full' / 'fixed_eval.json'}")
    else:
        stage_cfg = {s: (t, e) for s, t, e in CURRICULUM}
        timesteps, ent_coef = stage_cfg[args.stage]
        best = train_stage(args.stage, timesteps, ent_coef, args.resume, args.envs)
        if not args.no_eval:
            eval_stage(args.stage, best)


if __name__ == "__main__":
    main()
