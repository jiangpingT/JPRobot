"""Visualize a trained PPO agent controlling BittleX in simulation.

Enhanced visualization with:
- Smooth camera tracking
- Shadow, lighting, ground grid with distance markers
- Real-time stats overlay (reward, distance, speed, steps)
- Trajectory trail with color gradient
- Best distance flag marker
- Video recording support

Usage:
    python -m jprobot.training.enjoy
    python -m jprobot.training.enjoy --model trained/bittle_ppo.zip
    python -m jprobot.training.enjoy --record   # save video
"""

import argparse
import os
import time

import pybullet as p
from stable_baselines3 import PPO

from .env import BittleGymEnv


def setup_visual(physics_client, robot_id):
    """Configure PyBullet for better visual presentation."""
    # Enable shadows, hide default GUI panels
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=physics_client)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=physics_client)
    p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0, physicsClientId=physics_client)
    p.configureDebugVisualizer(
        p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0, physicsClientId=physics_client
    )

    # Lighting
    p.configureDebugVisualizer(
        lightPosition=[2, 2, 3], physicsClientId=physics_client
    )

    # Light grey ground
    p.changeVisualShape(
        0, -1, rgbaColor=[0.85, 0.88, 0.90, 1.0],
        physicsClientId=physics_client,
    )

    # Color the robot body (orange-red, more visible)
    num_joints = p.getNumJoints(robot_id, physicsClientId=physics_client)
    # Base link
    p.changeVisualShape(
        robot_id, -1, rgbaColor=[0.9, 0.45, 0.15, 1.0],
        physicsClientId=physics_client,
    )
    # Joint links: alternate colors for legs
    for j in range(num_joints):
        if j % 2 == 0:
            color = [0.2, 0.2, 0.2, 1.0]  # dark joints
        else:
            color = [0.85, 0.4, 0.1, 1.0]  # orange links
        p.changeVisualShape(
            robot_id, j, rgbaColor=color,
            physicsClientId=physics_client,
        )

    # Ground grid lines (10cm spacing)
    grid_color = [0.6, 0.65, 0.7]
    for i in range(-2, 30):
        x = i * 0.1
        p.addUserDebugLine(
            [x, -0.5, 0.001], [x, 0.5, 0.001],
            lineColorRGB=grid_color, lineWidth=1,
            physicsClientId=physics_client,
        )
    for j in range(-5, 6):
        y = j * 0.1
        p.addUserDebugLine(
            [-0.2, y, 0.001], [3.0, y, 0.001],
            lineColorRGB=grid_color, lineWidth=1,
            physicsClientId=physics_client,
        )

    # Distance markers every 0.5m (thicker line + label)
    for i in range(0, 30):
        x = i * 0.1
        if i % 5 == 0:
            p.addUserDebugText(
                f"{x:.1f}m", [x, -0.55, 0.01],
                textColorRGB=[0.3, 0.4, 0.5], textSize=1.0,
                physicsClientId=physics_client,
            )
            p.addUserDebugLine(
                [x, -0.5, 0.001], [x, 0.5, 0.001],
                lineColorRGB=[0.4, 0.45, 0.5], lineWidth=2,
                physicsClientId=physics_client,
            )

    # Green start line
    p.addUserDebugLine(
        [0, -0.3, 0.002], [0, 0.3, 0.002],
        lineColorRGB=[0.2, 0.8, 0.3], lineWidth=3,
        physicsClientId=physics_client,
    )
    p.addUserDebugText(
        "START", [0, 0.35, 0.01],
        textColorRGB=[0.2, 0.8, 0.3], textSize=1.2,
        physicsClientId=physics_client,
    )


def update_camera(physics_client, robot_id, smooth_pos):
    """Smooth camera tracking the robot. Returns (x, y, z) position."""
    pos, _ = p.getBasePositionAndOrientation(robot_id, physicsClientId=physics_client)

    # Lerp for smooth follow
    smooth_pos[0] = smooth_pos[0] * 0.9 + pos[0] * 0.1
    smooth_pos[1] = smooth_pos[1] * 0.9 + pos[1] * 0.1

    p.resetDebugVisualizerCamera(
        cameraDistance=0.35,
        cameraYaw=50,
        cameraPitch=-25,
        cameraTargetPosition=[smooth_pos[0], smooth_pos[1], 0.05],
        physicsClientId=physics_client,
    )
    return pos


def enjoy(model_path: str = None, episodes: int = 5, max_steps: int = 500,
          record: bool = False):
    """Run trained model in simulation with enhanced visualization."""
    if model_path is None:
        model_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "trained", "bittle_ppo.zip"
        )

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Train a model first with: python -m jprobot.training.train")
        return

    print(f"[JPRobot] Loading model from {model_path}")
    model = PPO.load(model_path)

    env = BittleGymEnv(render_mode="human")
    stats_ids = {}
    best_flag_id = None

    for episode in range(episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        max_distance = 0.0
        prev_distance = 0.0
        trail_points = []
        smooth_pos = [0.0, 0.0]
        log_id = None

        # Setup visuals after reset creates the physics world
        setup_visual(env.physics_client, env.robot_id)

        # Initial camera
        p.resetDebugVisualizerCamera(
            cameraDistance=0.35, cameraYaw=50, cameraPitch=-25,
            cameraTargetPosition=[0, 0, 0.05],
            physicsClientId=env.physics_client,
        )

        # Video recording
        if record:
            video_path = os.path.join(
                os.path.dirname(model_path),
                f"episode_{episode + 1}.mp4",
            )
            log_id = p.startStateLogging(
                p.STATE_LOGGING_VIDEO_MP4, video_path,
                physicsClientId=env.physics_client,
            )
            print(f"  Recording to {video_path}")

        print(f"\n{'='*40}")
        print(f"  Episode {episode + 1}/{episodes}")
        print(f"{'='*40}")

        terminated = False
        step = 0

        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Track position and speed
            pos = update_camera(env.physics_client, env.robot_id, smooth_pos)
            distance = pos[0]
            speed = (distance - prev_distance) / 0.02 if step > 0 else 0
            prev_distance = distance

            # Update best distance flag
            if distance > max_distance:
                max_distance = distance
                if best_flag_id is not None:
                    p.removeUserDebugItem(best_flag_id,
                                          physicsClientId=env.physics_client)
                best_flag_id = p.addUserDebugText(
                    f"BEST {max_distance:.2f}m",
                    [max_distance, 0, 0.18],
                    textColorRGB=[1.0, 0.85, 0.2], textSize=1.0,
                    physicsClientId=env.physics_client,
                )

            # Draw trajectory trail (every 3 steps)
            if step % 3 == 0:
                trail_points.append([pos[0], pos[1], 0.005])
                if len(trail_points) >= 2:
                    t = min(step / 250, 1.0)
                    color = [t, 1.0 - 0.3 * t, 0.2]
                    p.addUserDebugLine(
                        trail_points[-2], trail_points[-1],
                        lineColorRGB=color, lineWidth=3,
                        physicsClientId=env.physics_client,
                    )

            # Update stats overlay (every 5 steps)
            if step % 5 == 0:
                for key in list(stats_ids):
                    p.removeUserDebugItem(stats_ids[key],
                                          physicsClientId=env.physics_client)

                tx = smooth_pos[0] + 0.05
                ty = 0.4
                stats_ids["title"] = p.addUserDebugText(
                    f"Episode {episode + 1}/{episodes}",
                    [tx, ty, 0.25],
                    textColorRGB=[0.2, 0.7, 1.0], textSize=1.5,
                    physicsClientId=env.physics_client,
                )
                stats_ids["reward"] = p.addUserDebugText(
                    f"Reward: {total_reward:.0f}",
                    [tx, ty, 0.22],
                    textColorRGB=[0.3, 0.9, 0.4], textSize=1.2,
                    physicsClientId=env.physics_client,
                )
                stats_ids["distance"] = p.addUserDebugText(
                    f"Distance: {distance:.3f}m",
                    [tx, ty, 0.19],
                    textColorRGB=[1.0, 0.85, 0.3], textSize=1.2,
                    physicsClientId=env.physics_client,
                )
                stats_ids["speed"] = p.addUserDebugText(
                    f"Speed: {speed:.3f} m/s",
                    [tx, ty, 0.16],
                    textColorRGB=[0.6, 0.8, 1.0], textSize=1.0,
                    physicsClientId=env.physics_client,
                )
                stats_ids["steps"] = p.addUserDebugText(
                    f"Steps: {step + 1}/250",
                    [tx, ty, 0.13],
                    textColorRGB=[0.7, 0.7, 0.8], textSize=1.0,
                    physicsClientId=env.physics_client,
                )

            time.sleep(0.02)

            if terminated or truncated:
                break

        # Episode result: survived if not terminated (truncated = walked full 250 steps)
        survived = not terminated
        status = "SURVIVED 250 steps!" if survived else f"FALLEN at step {step + 1}"
        status_color = [0.3, 0.9, 0.4] if survived else [1.0, 0.3, 0.3]

        # Clear stats overlay
        for key in list(stats_ids):
            p.removeUserDebugItem(stats_ids[key],
                                  physicsClientId=env.physics_client)
        stats_ids.clear()

        # Show final result text
        p.addUserDebugText(
            status,
            [pos[0], 0, 0.28],
            textColorRGB=status_color, textSize=1.8,
            physicsClientId=env.physics_client,
        )
        p.addUserDebugText(
            f"Reward: {total_reward:.0f}  |  Distance: {max_distance:.3f}m",
            [pos[0], 0, 0.24],
            textColorRGB=[0.8, 0.8, 0.9], textSize=1.3,
            physicsClientId=env.physics_client,
        )

        print(f"  Status:   {status}")
        print(f"  Reward:   {total_reward:.0f}")
        print(f"  Distance: {max_distance:.3f}m")
        print(f"  Steps:    {step + 1}")

        if record and log_id is not None:
            p.stopStateLogging(log_id, physicsClientId=env.physics_client)

        # Pause between episodes
        time.sleep(2.5)

    env.close()
    print(f"\n{'='*40}")
    print("  Done!")
    print(f"{'='*40}")


def main():
    parser = argparse.ArgumentParser(description="Visualize trained BittleX agent")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained model (.zip)")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to run")
    parser.add_argument("--steps", type=int, default=500,
                        help="Maximum steps per episode")
    parser.add_argument("--record", action="store_true",
                        help="Record video (saved to trained/ directory)")
    args = parser.parse_args()

    enjoy(
        model_path=args.model,
        episodes=args.episodes,
        max_steps=args.steps,
        record=args.record,
    )


if __name__ == "__main__":
    main()
