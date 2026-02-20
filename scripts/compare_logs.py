#!/usr/bin/env python3
"""Real-time comparison of A/B training logs.

Parses SB3 PPO log output from both baseline (opencat-gym) and JPRobot,
prints a side-by-side comparison table every 30 seconds.

Usage:
    python scripts/compare_logs.py
    python scripts/compare_logs.py --interval 10   # faster refresh
    python scripts/compare_logs.py --once           # single snapshot
"""

import argparse
import os
import re
import sys
import time
from datetime import datetime

BASELINE_LOG = "/tmp/ab_test/baseline.log"
JPROBOT_LOG = "/tmp/ab_test/jprobot.log"

# SB3 PPO log format:
# | rollout/                |             |
# |    ep_len_mean          | 245         |
# |    ep_rew_mean          | 123         |
# | time/                   |             |
# |    total_timesteps      | 16384       |

METRIC_RE = re.compile(
    r"\|\s+(ep_rew_mean|ep_len_mean|total_timesteps)\s+\|\s+([\d.e+-]+)\s*\|"
)


def parse_log(path: str) -> list[dict]:
    """Parse SB3 PPO log file into a list of metric snapshots."""
    if not os.path.exists(path):
        return []

    snapshots = []
    current = {}

    with open(path) as f:
        for line in f:
            m = METRIC_RE.search(line)
            if m:
                key = m.group(1)
                val = m.group(2)
                try:
                    current[key] = float(val)
                except ValueError:
                    pass

                # When we see total_timesteps, that's the last metric in a block
                if key == "total_timesteps" and "ep_rew_mean" in current:
                    snapshots.append(current.copy())
                    current = {}

    # Capture any remaining partial block
    if "ep_rew_mean" in current:
        snapshots.append(current)

    return snapshots


def format_number(val: float, width: int = 10) -> str:
    """Format number for display."""
    if abs(val) >= 1e6:
        return f"{val / 1e6:.1f}M".rjust(width)
    elif abs(val) >= 1e3:
        return f"{val / 1e3:.0f}K".rjust(width)
    else:
        return f"{val:.1f}".rjust(width)


def print_comparison(baseline: list[dict], jprobot: list[dict]) -> None:
    """Print side-by-side comparison table."""
    now = datetime.now().strftime("%H:%M:%S")

    # Clear screen
    print("\033[2J\033[H", end="")

    print(f"{'=' * 70}")
    print(f"  A/B Training Comparison  [{now}]")
    print(f"{'=' * 70}")
    print()

    # Status
    b_status = f"{len(baseline)} updates" if baseline else "NOT STARTED"
    j_status = f"{len(jprobot)} updates" if jprobot else "NOT STARTED"
    print(f"  Baseline (opencat-gym): {b_status}")
    print(f"  JPRobot (our code):     {j_status}")
    print()

    if not baseline and not jprobot:
        print("  Waiting for training data...")
        return

    # Header
    print(f"  {'Metric':<20} {'Baseline':>12} {'JPRobot':>12} {'Delta':>12}")
    print(f"  {'-' * 20} {'-' * 12} {'-' * 12} {'-' * 12}")

    # Latest values
    b_latest = baseline[-1] if baseline else {}
    j_latest = jprobot[-1] if jprobot else {}

    for key, label in [
        ("total_timesteps", "Steps"),
        ("ep_rew_mean", "Reward (mean)"),
        ("ep_len_mean", "Episode Len"),
    ]:
        b_val = b_latest.get(key, 0)
        j_val = j_latest.get(key, 0)
        delta = j_val - b_val if b_val and j_val else 0

        b_str = format_number(b_val) if b_val else "    -".rjust(12)
        j_str = format_number(j_val) if j_val else "    -".rjust(12)

        if key == "total_timesteps":
            d_str = ""
        elif delta > 0:
            d_str = f"+{delta:.1f}".rjust(12)
        elif delta < 0:
            d_str = f"{delta:.1f}".rjust(12)
        else:
            d_str = "0".rjust(12)

        print(f"  {label:<20} {b_str} {j_str} {d_str}")

    # Reward trend (last 5 updates)
    print()
    print(f"  {'Reward Trend (last 5 updates)':}")
    print(f"  {'':>4} {'Baseline':>12} {'JPRobot':>12}")

    max_len = max(len(baseline), len(jprobot))
    start = max(0, max_len - 5)

    for i in range(start, max_len):
        b_rew = baseline[i]["ep_rew_mean"] if i < len(baseline) else None
        j_rew = jprobot[i]["ep_rew_mean"] if i < len(jprobot) else None

        b_str = f"{b_rew:.1f}".rjust(12) if b_rew is not None else "-".rjust(12)
        j_str = f"{j_rew:.1f}".rjust(12) if j_rew is not None else "-".rjust(12)

        print(f"  {i + 1:>4} {b_str} {j_str}")

    # Best reward
    print()
    if baseline:
        b_best = max(s["ep_rew_mean"] for s in baseline)
        b_best_step = next(
            s["total_timesteps"]
            for s in baseline
            if s["ep_rew_mean"] == b_best
        )
        print(f"  Baseline best: {b_best:.1f} @ {b_best_step / 1e6:.1f}M steps")

    if jprobot:
        j_best = max(s["ep_rew_mean"] for s in jprobot)
        j_best_step = next(
            s["total_timesteps"]
            for s in jprobot
            if s["ep_rew_mean"] == j_best
        )
        print(f"  JPRobot best:  {j_best:.1f} @ {j_best_step / 1e6:.1f}M steps")

    # Verdict
    print()
    if baseline and jprobot:
        b_rew = baseline[-1]["ep_rew_mean"]
        j_rew = jprobot[-1]["ep_rew_mean"]
        if j_rew > b_rew * 1.1:
            print("  Verdict: JPRobot is AHEAD")
        elif b_rew > j_rew * 1.1:
            print("  Verdict: Baseline is AHEAD")
        else:
            print("  Verdict: ROUGHLY EQUAL")

    print()
    print(f"  PID files: /tmp/ab_test/{{baseline,jprobot}}.pid")
    print(f"  Stop: bash scripts/run_ab_test.sh stop")


def main():
    parser = argparse.ArgumentParser(description="Compare A/B training logs")
    parser.add_argument("--interval", type=int, default=30, help="Refresh interval in seconds")
    parser.add_argument("--once", action="store_true", help="Print once and exit")
    parser.add_argument("--baseline-log", default=BASELINE_LOG)
    parser.add_argument("--jprobot-log", default=JPROBOT_LOG)
    args = parser.parse_args()

    try:
        while True:
            baseline = parse_log(args.baseline_log)
            jprobot = parse_log(args.jprobot_log)
            print_comparison(baseline, jprobot)

            if args.once:
                break

            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
