#!/usr/bin/env python3
"""Fixed-direction evaluation script for BittleX locomotion.

Evaluates the model with deterministic policy (no exploration noise) on each of
the 4 cardinal directions. Supports Route A (direction env) and Route B (velocity env).

Usage:
    python scripts/fixed_eval.py                                      # default env, best.zip
    python scripts/fixed_eval.py --run-id route_a --env-class BittleGymEnvV2
    python scripts/fixed_eval.py --run-id route_b --env-class BittleGymEnvVelocity
    python scripts/fixed_eval.py --episodes 5                         # quick check
"""

import argparse
import json
import os
import sys

from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DIRECTIONS = ["forward", "backward", "left", "right"]
TRAINED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "trained"))

_DIR_CN = {
    "forward": "向前",
    "backward": "向后",
    "left": "向左",
    "right": "向右",
}

# Direction → velocity command for Route B (velocity tracking env)
# Speed 0.25 m/s matches Route B's training range and gives comparable distance to Route A
_DIR_TO_VEL_CMD = {
    "forward":  [0.25, 0.0],
    "backward": [-0.25, 0.0],
    "left":     [0.0, 0.25],
    "right":    [0.0, -0.25],
}

# Direction → unit vector for projecting XY displacement (used in Route B target_progress)
_DIR_TO_UNIT = {
    "forward":  np.array([1.0, 0.0]),
    "backward": np.array([-1.0, 0.0]),
    "left":     np.array([0.0, 1.0]),
    "right":    np.array([0.0, -1.0]),
}


def _resolve_env_class(name: str | None):
    """Return env class by name string."""
    if name is None or name == "BittleGymEnv":
        from jprobot.training.env import BittleGymEnv
        return BittleGymEnv
    if name == "BittleGymEnvV2":
        from jprobot.training.env_v2 import BittleGymEnvV2
        return BittleGymEnvV2
    if name == "BittleGymEnvVelocity":
        from jprobot.training.env_velocity import BittleGymEnvVelocity
        return BittleGymEnvVelocity
    if name == "BittleGymEnvVelocityV2":
        from jprobot.training.env_velocity_v2 import BittleGymEnvVelocityV2
        return BittleGymEnvVelocityV2
    raise ValueError(f"Unknown env_class: {name!r}")


def _get_robot_pos_xy(env) -> np.ndarray:
    """Read current robot XY position directly from PyBullet."""
    import pybullet as p
    pos, _ = p.getBasePositionAndOrientation(env.robot_id, physicsClientId=env.physics_client)
    return np.array(pos[:2], dtype=np.float64)


def run_fixed_eval(
    model_path: str = None,
    episodes_per_direction: int = 20,
    env_class=None,
    output_path: str = None,
) -> dict:
    """Run fixed-direction evaluation with deterministic policy.

    Args:
        model_path: Path to model zip.
        episodes_per_direction: Number of episodes per direction.
        env_class: Gym environment class (BittleGymEnv / BittleGymEnvV2 / BittleGymEnvVelocity).
                   Defaults to BittleGymEnv.
        output_path: Where to write fixed_direction_eval.json.

    Returns:
        Result dict.
    """
    from stable_baselines3 import PPO

    if env_class is None:
        from jprobot.training.env import BittleGymEnv
        env_class = BittleGymEnv

    if model_path is None:
        model_path = os.path.join(TRAINED_DIR, "snapshots", "best.zip")
    if output_path is None:
        output_path = os.path.join(TRAINED_DIR, "fixed_direction_eval.json")

    model_path = os.path.abspath(model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    is_velocity_env = env_class.__name__ in ("BittleGymEnvVelocity", "BittleGymEnvVelocityV2")

    print(f"[FixedEval] Loading model:   {model_path}")
    print(f"[FixedEval] Env class:       {env_class.__name__}")
    print(f"[FixedEval] Output path:     {output_path}")
    model = PPO.load(model_path)

    print("[FixedEval] Creating environment (DIRECT mode)...")
    env = env_class(render_mode=None)

    results = {}
    for direction in DIRECTIONS:
        vel_cmd = _DIR_TO_VEL_CMD[direction]
        dir_unit = _DIR_TO_UNIT[direction]

        print(f"\n[FixedEval] Direction: {_DIR_CN[direction]} ({direction}), {episodes_per_direction} episodes...")
        episode_rewards = []
        episode_lens = []
        posture_heights = []
        posture_arm_contacts = []
        target_progresses = []

        for ep in range(episodes_per_direction):
            if is_velocity_env:
                obs, _ = env.reset(options={"vel_cmd": vel_cmd})
                start_pos = _get_robot_pos_xy(env)
            else:
                obs, _ = env.reset(options={"target_name": direction})

            total_reward = 0.0
            ep_len = 0
            done = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                ep_len += 1
                done = terminated or truncated

                if done:
                    posture = info.get("posture")
                    if posture:
                        posture_heights.append(posture["avg_height"])
                        posture_arm_contacts.append(posture["arm_contact_rate"])

                    if is_velocity_env:
                        # Compute net displacement projected onto the intended direction
                        # (PyBullet state is still valid immediately after done=True)
                        end_pos = _get_robot_pos_xy(env)
                        total_disp = float(np.dot(end_pos - start_pos, dir_unit))
                        # avg_target_progress = per-step average (matches Route A metric)
                        avg_tp = total_disp / max(ep_len, 1)
                        target_progresses.append(avg_tp)
                        velocity_info = info.get("velocity", {})
                        episode_lens.append(velocity_info.get("ep_len", ep_len))
                    else:
                        direction_info = info.get("direction")
                        if direction_info:
                            target_progresses.append(direction_info["avg_target_progress"])
                            episode_lens.append(direction_info["ep_len"])

            episode_rewards.append(total_reward)
            print(f"  ep {ep + 1:2d}/{episodes_per_direction}: reward={total_reward:.0f}", end="\r")

        print()

        median_tp = float(np.median(target_progresses)) if target_progresses else 0.0
        mean_ep_len = float(np.mean(episode_lens)) if episode_lens else 0.0
        mean_rew = float(np.mean(episode_rewards))
        avg_height = float(np.mean(posture_heights)) if posture_heights else 0.0
        arm_contact = float(np.mean(posture_arm_contacts)) if posture_arm_contacts else 0.0

        results[direction] = {
            "episodes": episodes_per_direction,
            "median_target_progress": round(median_tp, 6),
            "mean_ep_len": round(mean_ep_len, 1),
            "mean_episode_reward": round(mean_rew, 1),
            "arm_contact_rate": round(arm_contact, 3),
            "avg_height": round(avg_height, 4),
        }
        print(
            f"  {direction}: reward={mean_rew:.0f}, "
            f"progress={median_tp:.5f}, ep_len={mean_ep_len:.0f}, "
            f"height={avg_height:.4f}, arm_contact={arm_contact:.3f}"
        )

    env.close()

    weakest = min(DIRECTIONS, key=lambda d: results[d]["mean_episode_reward"])
    best = max(DIRECTIONS, key=lambda d: results[d]["mean_episode_reward"])
    best_reward = results[best]["mean_episode_reward"]
    weakest_reward = results[weakest]["mean_episode_reward"]

    ratio = weakest_reward / max(1.0, best_reward)
    rewards_all = [results[d]["mean_episode_reward"] for d in DIRECTIONS]
    avg_reward = float(sum(rewards_all) / len(rewards_all))

    if ratio < 0.7:
        recommendation = f"refine_{weakest}"
    elif avg_reward > 800:
        recommendation = "充分训练"
    else:
        recommendation = f"refine_{weakest}"

    output = {
        "model": os.path.relpath(model_path),
        "env_class": env_class.__name__,
        "episodes_per_direction": episodes_per_direction,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "results": results,
        "weakest_direction": weakest,
        "recommendation": recommendation,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"[FixedEval] 结果已写入 {output_path}")

    return output


def _print_summary(result: dict, prev_results: dict | None = None):
    """Print bar chart summary and comparison."""
    directions = DIRECTIONS
    best = max(directions, key=lambda d: result["results"][d]["mean_episode_reward"])
    weakest = result["weakest_direction"]
    best_rew = result["results"][best]["mean_episode_reward"]
    weak_rew = result["results"][weakest]["mean_episode_reward"]
    ratio = weak_rew / max(1.0, best_rew)

    if prev_results and prev_results.get("results"):
        print(f"\n[FixedEval] 与上次对比 (vs {prev_results.get('evaluated_at', '?')}):")
        for direction in directions:
            curr_rew = result["results"][direction]["mean_episode_reward"]
            if direction in prev_results["results"]:
                prev_rew = prev_results["results"][direction]["mean_episode_reward"]
                delta = curr_rew - prev_rew
                arrow = "↑" if delta >= 0 else "↓"
                print(f"  {direction:>10}: {prev_rew:.0f} → {curr_rew:.0f} ({arrow}{abs(delta):.0f})")
            else:
                print(f"  {direction:>10}: {curr_rew:.0f} (无历史)")

    print(f"\n[FixedEval] 各方向验收结果（{result.get('env_class','?')}）：")
    for direction in directions:
        r = result["results"][direction]["mean_episode_reward"]
        prog = result["results"][direction]["median_target_progress"]
        bar_len = int(r / max(best_rew, 1) * 20)
        bar = "█" * bar_len + " " * (20 - bar_len)
        flag = "⚠" if direction == weakest else "✓"
        cn = _DIR_CN.get(direction, direction)
        print(f"  {cn:>3}  {bar}  {r:.0f}分  进度 {prog:.5f}  {flag}")

    print(f"\n  最弱方向：{weakest}（{weak_rew:.0f}分，仅为最强的 {ratio * 100:.0f}%）")
    print(f"  建议：{result['recommendation']}")

    if result["recommendation"].startswith("refine_"):
        print(f"\n  下一步命令：")
        print(f"    python -m jprobot.training.progressive \\")
        print(f"      --curriculum {result['recommendation']} --auto")


def main():
    parser = argparse.ArgumentParser(
        description="Fixed-direction evaluation (deterministic policy, no exploration)"
    )
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model zip (overrides --run-id)")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Training run ID (e.g. route_a). Sets model and output paths.")
    parser.add_argument("--env-class", type=str, default=None,
                        help="Env class: BittleGymEnv | BittleGymEnvV2 | BittleGymEnvVelocity | BittleGymEnvVelocityV2")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Episodes per direction (default: 20)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (overrides default)")
    args = parser.parse_args()

    # Resolve paths
    run_id = args.run_id
    if run_id:
        run_dir = os.path.join(TRAINED_DIR, run_id)
        model_path = args.model or os.path.join(run_dir, "snapshots", "best.zip")
        output_path = args.output or os.path.join(run_dir, "fixed_direction_eval.json")
    else:
        model_path = args.model
        output_path = args.output or os.path.join(TRAINED_DIR, "fixed_direction_eval.json")

    env_class = _resolve_env_class(args.env_class)

    # Load previous results for comparison
    prev_results = None
    if output_path and os.path.exists(output_path):
        try:
            with open(output_path) as f:
                prev_results = json.load(f)
            print(f"[FixedEval] 上次评估：{prev_results.get('evaluated_at', '?')}")
        except (json.JSONDecodeError, OSError):
            pass

    result = run_fixed_eval(
        model_path=model_path,
        episodes_per_direction=args.episodes,
        env_class=env_class,
        output_path=output_path,
    )

    _print_summary(result, prev_results)
    return result


if __name__ == "__main__":
    main()
