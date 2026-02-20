#!/usr/bin/env python3
"""Demo: PyBullet simulation of BittleX.

Run this to see the BittleX URDF model in PyBullet simulation.
No real robot or training needed - just visualizes the model.

Usage: python scripts/demo_simulation.py
"""

import time
import math

import pybullet as p
import pybullet_data
import numpy as np


def main():
    print("=== JPRobot Simulation Demo ===\n")
    print("Loading BittleX model in PyBullet...")
    print("Controls: Mouse drag to rotate, scroll to zoom")
    print("Press Ctrl+C to exit\n")

    # Setup
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # Camera setup
    p.resetDebugVisualizerCamera(
        cameraDistance=0.3,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0.05],
    )

    # Load ground and robot
    p.loadURDF("plane.urdf")

    import os
    urdf_path = os.path.join(
        os.path.dirname(__file__), "..", "jprobot", "models", "bittle_esp32.urdf"
    )
    robot_id = p.loadURDF(
        urdf_path,
        [0, 0, 0.08],
        p.getQuaternionFromEuler([0, 0, 0]),
        flags=p.URDF_USE_SELF_COLLISION,
    )

    # Print joint info
    n_joints = p.getNumJoints(robot_id)
    print(f"Robot loaded with {n_joints} joints:")
    for i in range(n_joints):
        info = p.getJointInfo(robot_id, i)
        joint_name = info[1].decode("utf-8")
        joint_type = ["revolute", "prismatic", "spherical", "planar", "fixed"][info[2]]
        print(f"  [{i}] {joint_name} ({joint_type})")

    # Initialize to standing pose (matching training: shoulders/hips=50°, elbows/knees=0°)
    init_pattern = [1, 0, 1, 0, 1, 0, 1, 0]  # alternating: upper=50°, lower=0°
    revolute_joints = []
    joint_idx = 0
    for i in range(n_joints):
        if p.getJointInfo(robot_id, i)[2] == 0:  # revolute
            revolute_joints.append(i)
            angle = math.radians(init_pattern[joint_idx] * 50)
            p.resetJointState(robot_id, i, angle)
            p.changeDynamics(robot_id, i, maxJointVelocity=math.pi * 10)
            joint_idx += 1

    print(f"\nActuated joints: {revolute_joints}")
    print("\nRunning simple walking demo...")

    # Simple walking pattern demo
    try:
        step = 0
        while True:
            t = step * 0.02

            # Simple sinusoidal gait pattern
            for i, joint_idx in enumerate(revolute_joints):
                phase = math.pi * (i % 2)  # Alternating phase
                if i < 4:  # Shoulder/hip joints
                    angle = math.radians(50 + 20 * math.sin(2 * t + phase))
                else:  # Knee/elbow joints
                    angle = math.radians(50 + 15 * math.sin(2 * t + phase + math.pi / 2))

                p.setJointMotorControl2(
                    robot_id, joint_idx,
                    p.POSITION_CONTROL,
                    targetPosition=angle,
                    force=0.2,
                )

            p.stepSimulation()
            time.sleep(1 / 240)
            step += 1

    except KeyboardInterrupt:
        print("\nExiting...")

    p.disconnect()
    print("Done!")


if __name__ == "__main__":
    main()
