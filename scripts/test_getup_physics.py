#!/usr/bin/env python3
"""物理可行性测试：BittleX 能否从趴地姿态站起来？

不用 RL，直接用 PD 控制尝试把关节推向站立帧（BF_FRAMES_RAD[4]），
观察身体能抬到多高。

结论：
  height_final > 0.05m → 物理可行，RL 应该能学到
  height_final ≈ 0.02m → 物理不可行，关节力矩不够
"""

import sys
import time
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data

sys.path.insert(0, str(Path(__file__).parent.parent))
from jprobot.training.env_backflip import BF_FRAMES_RAD

URDF_PATH = Path(__file__).parent.parent / "jprobot" / "models" / "bittle_esp32.urdf"
TARGET_JOINTS = BF_FRAMES_RAD[4]  # 站稳帧：[30°, 30°, ...] × 8

def find_revolute_joints(robot_id):
    joint_ids = []
    for i in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, i)
        if info[2] == p.JOINT_REVOLUTE:
            joint_ids.append(i)
    return joint_ids[:8]

def run_test(prone_height=0.04, n_steps=500):
    client = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.8, physicsClientId=client)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf", physicsClientId=client)

    robot = p.loadURDF(str(URDF_PATH), [0, 0, prone_height],
                       p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=client)

    joints = find_revolute_joints(robot)

    # 初始关节设为趴地帧（BF_FRAMES_RAD[3]）+ 随机扰动
    init_joints = BF_FRAMES_RAD[3] + np.random.uniform(-0.2, 0.2, 8)
    for i, jid in enumerate(joints):
        p.resetJointState(robot, jid, init_joints[i], physicsClientId=client)

    # 稳定 5 步
    for _ in range(5):
        p.stepSimulation(physicsClientId=client)

    pos_init, _ = p.getBasePositionAndOrientation(robot, physicsClientId=client)
    h_init = pos_init[2]

    # 用 PD 控制把关节推向站稳帧
    heights = []
    for step in range(n_steps):
        for i, jid in enumerate(joints):
            p.setJointMotorControl2(
                robot, jid,
                controlMode=p.POSITION_CONTROL,
                targetPosition=TARGET_JOINTS[i],
                force=0.35,          # BittleX 舵机最大力矩约 0.3-0.4 Nm
                maxVelocity=3.0,
                physicsClientId=client,
            )
        p.stepSimulation(physicsClientId=client)
        pos, _ = p.getBasePositionAndOrientation(robot, physicsClientId=client)
        heights.append(pos[2])

    h_final = np.mean(heights[-50:])  # 最后50步均值
    h_peak  = max(heights)

    p.disconnect(client)
    return h_init, h_final, h_peak

if __name__ == "__main__":
    print("=" * 50)
    print("  BittleX 起身物理可行性测试")
    print("  目标关节：BF_FRAMES_RAD[4]（站稳帧，30°×8）")
    print("=" * 50)

    results = []
    for trial in range(5):
        h_init, h_final, h_peak = run_test()
        results.append((h_init, h_final, h_peak))
        print(f"  试验 {trial+1}: 初始={h_init:.4f}m  最终={h_final:.4f}m  峰值={h_peak:.4f}m")

    avg_init  = np.mean([r[0] for r in results])
    avg_final = np.mean([r[1] for r in results])
    avg_peak  = np.mean([r[2] for r in results])

    print()
    print(f"  平均初始高度: {avg_init:.4f}m")
    print(f"  平均最终高度: {avg_final:.4f}m")
    print(f"  平均峰值高度: {avg_peak:.4f}m")
    print()

    if avg_final > 0.05:
        print("  ✅ 结论：物理上可行！PD 控制能抬起身体，RL 应该能学到。")
    elif avg_peak > 0.05:
        print("  ⚠️  结论：峰值可行但不稳定，RL 需要学更精细的动作序列。")
    else:
        print("  ❌ 结论：物理不可行！关节力矩不够，0.0226m 是物理上限。")
        print("       → 需要放弃'完全站立'目标，或修改物理参数（力矩/摩擦）。")
