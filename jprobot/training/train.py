"""PPO training script for BittleX locomotion.

Snapshot system:
- trained/snapshots/best.zip          always the current best model
- trained/snapshots/step_Nm_rew_R.zip named snapshot every time reward improves
- trained/snapshots/manifest.json     history log (step, reward, timestamp, file)
- trained/checkpoints/                periodic checkpoint every 2M steps (safety net)

Usage:
    python -m jprobot.training.train
    python -m jprobot.training.train --timesteps 50000000 --envs 8
    python -m jprobot.training.train --resume trained/snapshots/best.zip
"""

import argparse
import json
import os
import shutil
from datetime import datetime

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from .env import BittleGymEnv


class LiveProgressCallback(BaseCallback):
    """每个 rollout 结束后将实时进度写入 trained/live_progress.json。

    Dashboard 读取此文件显示训练中的步数/奖励，解决 progressive_state.json
    在阶段完成前 total_steps=0 导致进度条为 0% 的问题。
    """

    def __init__(self, output_path: str, total_timesteps: int):
        super().__init__(verbose=0)
        self.output_path = output_path
        self.total_timesteps = total_timesteps

    def _on_rollout_end(self) -> None:
        current = self.num_timesteps
        ep_rew_mean = None
        if self.model.ep_info_buffer:
            ep_rew_mean = float(np.mean([ep["r"] for ep in self.model.ep_info_buffer]))
        data = {
            "total_timesteps": current,
            "total_timesteps_target": self.total_timesteps,
            "progress": current / max(1, self.total_timesteps),
            "ep_rew_mean": ep_rew_mean,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        try:
            with open(self.output_path, "w") as f:
                json.dump(data, f)
        except OSError:
            pass

    def _on_step(self) -> bool:
        return True


class MetricsTracker(BaseCallback):
    """Collects rolling reward/ep_len metrics over a training stage.

    After training, call .summary() to get health-check data.
    """

    def __init__(self):
        super().__init__(verbose=0)
        self.rewards: list[float] = []
        self.ep_lens: list[float] = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if self.model.ep_info_buffer:
            self.rewards.append(float(np.mean([ep["r"] for ep in self.model.ep_info_buffer])))
            self.ep_lens.append(float(np.mean([ep["l"] for ep in self.model.ep_info_buffer])))

    def summary(self) -> dict:
        if not self.rewards:
            return {}
        n = len(self.rewards)
        split = max(1, n // 5)
        return {
            "reward_final": self.rewards[-1],
            "reward_best": max(self.rewards),
            "reward_start": float(np.mean(self.rewards[:split])),
            "reward_end": float(np.mean(self.rewards[-split:])),
            "reward_trend": float(np.mean(self.rewards[-split:]) - np.mean(self.rewards[:split])),
            "ep_len_final": self.ep_lens[-1] if self.ep_lens else 0.0,
        }


class PostureMetricsCallback(BaseCallback):
    """Tracks per-episode posture metrics (height, arm contact, tilt) from env info.

    Writes a rolling summary to trained/posture_eval.json after every rollout,
    so the dashboard can show whether the robot is walking or crawling.
    Also appends to posture_history.jsonl for trend charts.
    """

    def __init__(self, output_path: str, window: int = 100):
        super().__init__(verbose=0)
        self.output_path = output_path
        self.history_path = output_path.replace('.json', '_history.jsonl')
        self.window = window
        self._heights: list[float] = []
        self._arm_contacts: list[float] = []
        self._tilts: list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            posture = info.get('posture')
            if posture:
                self._heights.append(posture['avg_height'])
                self._arm_contacts.append(posture['arm_contact_rate'])
                self._tilts.append(posture['avg_tilt_deg'])
        return True

    def _on_rollout_end(self) -> None:
        if not self._heights:
            return
        w = self.window
        avg_h = float(np.mean(self._heights[-w:]))
        avg_ac = float(np.mean(self._arm_contacts[-w:]))
        avg_tilt = float(np.mean(self._tilts[-w:]))
        # Simple heuristic: walking = high body + low arm contact
        behavior = 'walking' if avg_h > 0.06 and avg_ac < 0.3 else 'crawling'
        data = {
            'avg_height': round(avg_h, 4),
            'arm_contact_rate': round(avg_ac, 3),
            'avg_tilt_deg': round(avg_tilt, 1),
            'behavior': behavior,
            'episodes_sampled': len(self._heights),
            'updated_at': datetime.now().isoformat(timespec='seconds'),
        }
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, 'w') as f:
            json.dump(data, f, indent=2)

        # Append to history for trend charts
        history_point = {
            'step': self.num_timesteps,
            'h': round(avg_h, 4),
            'ac': round(avg_ac, 3),
            'tilt': round(avg_tilt, 1),
        }
        with open(self.history_path, 'a') as f:
            f.write(json.dumps(history_point) + '\n')


class DirectionMetricsCallback(BaseCallback):
    """Tracks per-direction target progress and episode length from env info."""

    def __init__(self, output_path: str, window: int = 200):
        super().__init__(verbose=0)
        self.output_path = output_path
        self.history_path = output_path.replace(".json", "_history.jsonl")
        self.window = window
        self._events: list[dict] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            direction = info.get("direction")
            if not direction:
                continue
            self._events.append({
                "target_name": direction.get("target_name", "unknown"),
                "avg_target_progress": float(direction.get("avg_target_progress", 0.0)),
                "ep_len": float(direction.get("ep_len", 0.0)),
            })
        return True

    def _on_rollout_end(self) -> None:
        if not self._events:
            return
        events = self._events[-self.window:]
        by_dir: dict[str, list[dict]] = {}
        for e in events:
            by_dir.setdefault(e["target_name"], []).append(e)

        per_direction = {}
        for name, rows in by_dir.items():
            progresses = [r["avg_target_progress"] for r in rows]
            ep_lens = [r["ep_len"] for r in rows]
            per_direction[name] = {
                "episodes": len(rows),
                "median_target_progress": float(np.median(progresses)),
                "mean_target_progress": float(np.mean(progresses)),
                "mean_ep_len": float(np.mean(ep_lens)),
            }

        data = {
            "window_episodes": len(events),
            "per_direction": per_direction,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(data, f, indent=2)

        history_point = {
            "step": self.num_timesteps,
            "updated_at": data["updated_at"],
            "per_direction": {
                k: {
                    "median_target_progress": round(v["median_target_progress"], 6),
                    "mean_target_progress": round(v["mean_target_progress"], 6),
                    "mean_ep_len": round(v["mean_ep_len"], 2),
                    "episodes": v["episodes"],
                }
                for k, v in per_direction.items()
            },
        }
        with open(self.history_path, "a") as f:
            f.write(json.dumps(history_point) + "\n")


class SnapshotCallback(BaseCallback):
    """Saves a named snapshot every time mean reward hits a new high.

    Directory layout::

        snapshot_dir/
            best.zip                       ← always the current best
            step_5.0M_rew_234.zip          ← named snapshot
            manifest.json                  ← history log
    """

    def __init__(self, snapshot_dir: str, min_improvement: float = 5.0,
                 reset_best: bool = False, verbose: int = 1):
        super().__init__(verbose)
        self.snapshot_dir = snapshot_dir
        self.min_improvement = min_improvement
        self.reset_best = reset_best
        self.best_reward = -float("inf")
        self.manifest_path = os.path.join(snapshot_dir, "manifest.json")
        self.manifest: list[dict] = []

    def _init_callback(self) -> None:
        os.makedirs(self.snapshot_dir, exist_ok=True)
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path) as f:
                self.manifest = json.load(f)
            if self.manifest and not self.reset_best:
                self.best_reward = max(e["reward"] for e in self.manifest)
                print(f"[Snapshot] Restored best reward from manifest: {self.best_reward:.0f}")
            elif self.reset_best:
                print(f"[Snapshot] reset_best=True, starting from -inf (curriculum stage change)")

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if not self.model.ep_info_buffer:
            return

        mean_reward = float(np.mean([ep["r"] for ep in self.model.ep_info_buffer]))

        if mean_reward <= self.best_reward + self.min_improvement:
            return

        self.best_reward = mean_reward
        steps_m = self.num_timesteps / 1e6
        name = f"step_{steps_m:.1f}M_rew_{mean_reward:.0f}"

        snapshot_path = os.path.join(self.snapshot_dir, name + ".zip")
        best_path = os.path.join(self.snapshot_dir, "best.zip")

        self.model.save(snapshot_path)
        shutil.copy2(snapshot_path, best_path)

        entry = {
            "step": self.num_timesteps,
            "reward": round(mean_reward, 1),
            "file": name + ".zip",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        self.manifest.append(entry)
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)

        if self.verbose:
            print(
                f"\n[Snapshot] New best {mean_reward:.0f} at {steps_m:.1f}M steps"
                f" → {name}.zip"
            )


def train(
    total_timesteps: int = 50_000_000,
    parallel_envs: int = 8,
    net_arch: list[int] = None,
    save_path: str = None,
    resume_from: str = None,
    seed: int = 42,
    return_metrics: bool = False,
    env_config: dict = None,
    ent_coef: float = 0.0,
    env_class=None,
    trained_dir: str = None,
):
    """Train a PPO agent for BittleX locomotion.

    Args:
        return_metrics: If True, returns (model, metrics_dict) instead of model.
        env_config: Override reward weights for curriculum stages (passed to env).
        ent_coef: Entropy coefficient for PPO (0.0 = pure policy gradient,
                  >0 encourages exploration).
        env_class: Env class to use (default: BittleGymEnv). Pass BittleGymEnvV2
                   or BittleGymEnvVelocity for alternate observation spaces.
                   NOTE: models are NOT transferable between env classes (different obs dims).
    """
    env_cls = env_class if env_class is not None else BittleGymEnv
    if net_arch is None:
        net_arch = [256, 256]

    if trained_dir is None:
        trained_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "trained")
        )
    else:
        trained_dir = os.path.abspath(trained_dir)
    os.makedirs(trained_dir, exist_ok=True)
    if save_path is None:
        save_path = os.path.join(trained_dir, "bittle_ppo")

    snapshot_dir = os.path.join(trained_dir, "snapshots")
    checkpoint_dir = os.path.join(trained_dir, "checkpoints")
    posture_eval_path = os.path.join(trained_dir, "posture_eval.json")
    direction_eval_path = os.path.join(trained_dir, "direction_eval.json")

    print("[JPRobot Training]")
    print(f"  Timesteps:     {total_timesteps:,}")
    print(f"  Parallel envs: {parallel_envs}")
    print(f"  Network arch:  {net_arch}")
    print(f"  ent_coef:      {ent_coef}")
    if env_config:
        print(f"  env_config:    {env_config}")
        if "direction_mode" in env_config or "target_dir" in env_config:
            print("  NOTE: direction-conditioned training enabled; reward not comparable with legacy forward-only runs.")
    print(f"  Snapshots:     {snapshot_dir}  (best model + named on improvement)")
    print(f"  Checkpoints:   {checkpoint_dir}  (every 2M steps)")
    print()

    # Build env_kwargs for vectorised environments
    env_kwargs = {"config": env_config} if env_config else {}

    print(f"  Env class:     {env_cls.__name__}")

    # Check environment (skip when resuming — env hasn't changed)
    if not resume_from:
        print("Checking environment...")
        env = env_cls(**env_kwargs)
        check_env(env)
        env.close()
        print("Environment check passed.")

    # Vectorised environments
    print(f"Creating {parallel_envs} parallel environments...")
    vec_env = make_vec_env(
        env_cls, n_envs=parallel_envs, vec_env_cls=SubprocVecEnv,
        env_kwargs=env_kwargs or None,
    )

    # PPO setup
    custom_arch = dict(net_arch=net_arch)
    n_steps = int(2048 * 8 / parallel_envs)

    if resume_from:
        print(f"Resuming from {resume_from}...")
        model = PPO.load(
            resume_from,
            vec_env,
            policy_kwargs=custom_arch,
            n_steps=n_steps,
            learning_rate=3e-4,
            ent_coef=ent_coef,
            verbose=1,
        )
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            seed=seed,
            policy_kwargs=custom_arch,
            n_steps=n_steps,
            learning_rate=3e-4,
            ent_coef=ent_coef,
            verbose=1,
        )

    # Callbacks
    snapshot_cb = SnapshotCallback(
        snapshot_dir=snapshot_dir, min_improvement=5.0,
        reset_best=(env_config is not None),
    )
    checkpoint_cb = CheckpointCallback(
        save_freq=max(2_000_000 // parallel_envs, 1),
        save_path=checkpoint_dir,
        name_prefix="bittle_ppo",
    )
    metrics_cb = MetricsTracker()
    posture_cb = PostureMetricsCallback(output_path=posture_eval_path)
    direction_cb = DirectionMetricsCallback(output_path=direction_eval_path)
    live_progress_cb = LiveProgressCallback(
        output_path=os.path.join(trained_dir, "live_progress.json"),
        total_timesteps=total_timesteps,
    )
    callbacks = CallbackList([snapshot_cb, checkpoint_cb, metrics_cb, posture_cb, direction_cb, live_progress_cb])

    # Train
    print(f"Starting training for {total_timesteps:,} steps...")
    print(f"  learning_rate: 3e-4 (constant)")
    print(f"  ent_coef:      {ent_coef}")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        reset_num_timesteps=True,   # always restart lr schedule per stage
    )

    # Final save
    model.save(save_path)
    print(f"\nFinal model saved to {save_path}.zip")

    vec_env.close()

    if return_metrics:
        return model, metrics_cb.summary()
    return model


def main():
    parser = argparse.ArgumentParser(description="Train BittleX locomotion with PPO")
    parser.add_argument("--timesteps", type=int, default=50_000_000)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--arch", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from snapshot, e.g. trained/snapshots/best.zip")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        total_timesteps=args.timesteps,
        parallel_envs=args.envs,
        net_arch=args.arch,
        save_path=args.save,
        resume_from=args.resume,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
