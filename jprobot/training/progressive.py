"""Progressive training orchestrator for BittleX.

Trains in cumulative stages: 1M → 5M → 10M → 50M → 100M steps.
After each stage an automatic health check decides whether to proceed.
All state is persisted so training can be stopped and resumed at any time.

State file:  trained/progressive_state.json
Snapshots:   trained/snapshots/stage_1M.zip, stage_5M.zip, ...
Best model:  trained/snapshots/best.zip  (always current best)

Usage:
    # Start from scratch
    python -m jprobot.training.progressive

    # Resume where we left off (reads state file automatically)
    python -m jprobot.training.progressive --resume

    # Custom stages (cumulative targets in millions)
    python -m jprobot.training.progressive --stages 1 5 10 50

    # Skip health check and always proceed
    python -m jprobot.training.progressive --auto
"""

import argparse
import json
import os
import shutil
from datetime import datetime

from .train import train

# ── Cumulative step targets (M = million) ────────────────────────────────────
DEFAULT_STAGES = [1_000_000, 5_000_000, 10_000_000, 50_000_000, 100_000_000]

# ── Health check thresholds ───────────────────────────────────────────────────
HEALTH = {
    "min_reward_final": 0,      # final reward must be positive
    "min_ep_len": 25,           # robot must survive > 25 steps
    "max_trend_drop": -30,      # end-of-stage reward must not fall > 30 vs start
    "max_cross_stage_drop": -20,# final reward must not fall > 20 vs previous stage
}


# ─────────────────────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────────────────────

def check_health(metrics: dict, prev_metrics: dict | None = None) -> tuple[bool, list[str]]:
    """Evaluate whether a stage's training result is healthy.

    Returns (is_healthy, list_of_issues).
    """
    issues = []

    final = metrics.get("reward_final", 0)
    trend = metrics.get("reward_trend", 0)
    ep_len = metrics.get("ep_len_final", 0)

    if final < HEALTH["min_reward_final"]:
        issues.append(f"Reward negative: {final:.0f} < 0")

    if ep_len < HEALTH["min_ep_len"]:
        issues.append(f"Episode length too low: {ep_len:.0f} < {HEALTH['min_ep_len']}")

    if trend < HEALTH["max_trend_drop"]:
        issues.append(
            f"Reward declined within stage: end={metrics.get('reward_end', 0):.0f}"
            f" vs start={metrics.get('reward_start', 0):.0f} (Δ{trend:+.0f})"
        )

    if prev_metrics:
        prev_final = prev_metrics.get("reward_final", 0)
        drop = final - prev_final
        if drop < HEALTH["max_cross_stage_drop"]:
            issues.append(
                f"Reward dropped from previous stage: {prev_final:.0f} → {final:.0f}"
                f" (Δ{drop:+.0f})"
            )

    return len(issues) == 0, issues


# ─────────────────────────────────────────────────────────────────────────────
# Progressive trainer
# ─────────────────────────────────────────────────────────────────────────────

class ProgressiveTrainer:
    """Runs training in progressive stages with automatic health checks."""

    def __init__(self, parallel_envs: int = 8, seed: int = 42):
        trained_dir = os.path.join(os.path.dirname(__file__), "..", "..", "trained")
        self.trained_dir = os.path.abspath(trained_dir)
        self.snapshot_dir = os.path.join(self.trained_dir, "snapshots")
        self.state_path = os.path.join(self.trained_dir, "progressive_state.json")
        self.parallel_envs = parallel_envs
        self.seed = seed

    # ── State persistence ────────────────────────────────────────────────────

    def _load_state(self) -> dict:
        if os.path.exists(self.state_path):
            with open(self.state_path) as f:
                state = json.load(f)
            print(f"[Progressive] Loaded state: stage {state['stage_idx']}/{state['total_stages']}"
                  f", {state['total_steps']:,} steps so far")
            return state
        return {
            "stage_idx": 0,
            "total_stages": 0,
            "total_steps": 0,
            "best_model": None,
            "last_metrics": None,
            "stages": [],
            "started_at": datetime.now().isoformat(timespec="seconds"),
        }

    def _save_state(self, state: dict) -> None:
        with open(self.state_path, "w") as f:
            json.dump(state, f, indent=2)

    # ── Display helpers ───────────────────────────────────────────────────────

    def _print_stage_header(self, idx: int, total: int, name: str,
                            stage_steps: int, best_model: str | None) -> None:
        print(f"\n{'═' * 60}")
        print(f"  STAGE {idx}/{total}  →  {name} cumulative steps")
        print(f"  This stage:  {stage_steps / 1e6:.1f}M additional steps")
        src = best_model if best_model else "scratch"
        print(f"  Starting from: {src}")
        print(f"{'═' * 60}\n")

    def _print_health_report(self, name: str, metrics: dict,
                             healthy: bool, issues: list[str]) -> None:
        status = "HEALTHY ✓" if healthy else "UNHEALTHY ✗"
        bar = "─" * 60
        print(f"\n{bar}")
        print(f"  Stage {name} — {status}")
        print(f"  Reward :  final={metrics.get('reward_final', 0):.0f}"
              f"  best={metrics.get('reward_best', 0):.0f}"
              f"  trend={metrics.get('reward_trend', 0):+.0f}")
        print(f"  ep_len :  {metrics.get('ep_len_final', 0):.0f} steps")
        if issues:
            print(f"  Issues :")
            for issue in issues:
                print(f"    • {issue}")
        print(bar)

    # ── Main run loop ─────────────────────────────────────────────────────────

    def run(self, stages: list[int] = None, auto_proceed: bool = False) -> None:
        """Execute progressive training.

        Args:
            stages: Cumulative step targets. Defaults to DEFAULT_STAGES.
            auto_proceed: If True, skip interactive prompt and always continue
                          to the next stage when health check passes.
        """
        if stages is None:
            stages = DEFAULT_STAGES

        state = self._load_state()
        state["total_stages"] = len(stages)
        state["planned_stages"] = stages
        self._save_state(state)  # persist plan immediately so dashboard knows current stage target
        start_idx = state["stage_idx"]

        if start_idx >= len(stages):
            print("[Progressive] All stages already completed.")
            return

        os.makedirs(self.snapshot_dir, exist_ok=True)

        for i, target in enumerate(stages[start_idx:], start=start_idx):
            stage_steps = target - state["total_steps"]
            stage_name = f"{target // 1_000_000}M"

            self._print_stage_header(
                idx=i + 1,
                total=len(stages),
                name=stage_name,
                stage_steps=stage_steps,
                best_model=state["best_model"],
            )

            # ── Train ────────────────────────────────────────────────────────
            _, metrics = train(
                total_timesteps=stage_steps,
                parallel_envs=self.parallel_envs,
                resume_from=state["best_model"],
                seed=self.seed,
                return_metrics=True,
            )

            # ── Stage snapshot ───────────────────────────────────────────────
            # best.zip is already updated by SnapshotCallback inside train().
            # Save an additional immutable stage snapshot.
            best_zip = os.path.join(self.snapshot_dir, "best.zip")
            stage_snap = os.path.join(self.snapshot_dir, f"stage_{stage_name}.zip")
            if os.path.exists(best_zip):
                shutil.copy2(best_zip, stage_snap)
                print(f"[Progressive] Stage snapshot → {stage_snap}")

            # ── Persist state ────────────────────────────────────────────────
            healthy, issues = check_health(metrics, state["last_metrics"])

            state["stage_idx"] = i + 1
            state["total_steps"] = target
            state["best_model"] = best_zip if os.path.exists(best_zip) else None
            state["last_metrics"] = metrics
            state["stages"].append({
                "name": stage_name,
                "target_steps": target,
                "stage_steps": stage_steps,
                "snapshot": stage_snap,
                "metrics": metrics,
                "healthy": healthy,
                "issues": issues,
                "completed_at": datetime.now().isoformat(timespec="seconds"),
            })
            self._save_state(state)

            # ── Health report ────────────────────────────────────────────────
            self._print_health_report(stage_name, metrics, healthy, issues)

            if not healthy:
                print(
                    "\n[Progressive] Stopped due to unhealthy stage.\n"
                    "  Adjust hyperparameters or reward function, then resume:\n"
                    f"    python -m jprobot.training.progressive --resume\n"
                )
                return

            if i + 1 >= len(stages):
                print("\n[Progressive] All stages completed!")
                return

            # ── Decide whether to proceed ────────────────────────────────────
            next_name = f"{stages[i + 1] // 1_000_000}M"
            if not auto_proceed:
                answer = input(
                    f"\n  Proceed to next stage ({next_name} steps)? [Y/n]: "
                ).strip().lower()
                if answer == "n":
                    print("[Progressive] Stopped by user. Resume anytime with --resume.")
                    return

            print(f"\n[Progressive] Proceeding to stage {next_name}...\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Progressive PPO training: 1M → 5M → 10M → 50M → 100M steps"
    )
    parser.add_argument(
        "--stages", type=int, nargs="+",
        default=[s // 1_000_000 for s in DEFAULT_STAGES],
        metavar="M",
        help="Cumulative targets in millions (default: 1 5 10 50 100)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from saved progressive_state.json",
    )
    parser.add_argument(
        "--auto", action="store_true",
        help="Auto-proceed through all stages without interactive prompt",
    )
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    stages = [m * 1_000_000 for m in args.stages]

    trainer = ProgressiveTrainer(parallel_envs=args.envs, seed=args.seed)

    if not args.resume:
        # Fresh start: wipe old state
        state_path = trainer.state_path
        if os.path.exists(state_path):
            os.remove(state_path)
            print("[Progressive] Cleared previous state (fresh start).")

    trainer.run(stages=stages, auto_proceed=args.auto)


if __name__ == "__main__":
    main()
