#!/usr/bin/env python3
"""测试从真实后空翻落地状态（h=0.024m，joints=[73,-72,73,-72,109,-73,110,-73]°）能否站起来。"""
import sys, numpy as np
sys.path.insert(0, '.')
import pybullet as p, pybullet_data
from jprobot.training.env_backflip import BF_FRAMES_RAD

URDF = "jprobot/models/bittle_esp32.urdf"
ACTUAL_LANDING = np.deg2rad([73., -72., 73., -72., 109., -73., 110., -73.])
TARGET = BF_FRAMES_RAD[4]

print("=== 从真实落地状态（h=0.024m）起身测试 ===")
results = []
for trial in range(5):
    cli = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.8, physicsClientId=cli)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf", physicsClientId=cli)
    robot = p.loadURDF(URDF, [0, 0, 0.024], p.getQuaternionFromEuler([0, 0, 0]), physicsClientId=cli)
    joints = [i for i in range(p.getNumJoints(robot, physicsClientId=cli))
              if p.getJointInfo(robot, i, physicsClientId=cli)[2] == p.JOINT_REVOLUTE][:8]
    init_j = ACTUAL_LANDING + np.random.uniform(-0.3, 0.3, 8)
    for i, jid in enumerate(joints):
        p.resetJointState(robot, jid, init_j[i], physicsClientId=cli)
    for _ in range(5):
        p.stepSimulation(physicsClientId=cli)
    pos0, _ = p.getBasePositionAndOrientation(robot, physicsClientId=cli)
    heights = []
    for _ in range(500):
        for i, jid in enumerate(joints):
            p.setJointMotorControl2(robot, jid, p.POSITION_CONTROL,
                                    TARGET[i], force=0.35, maxVelocity=3.0, physicsClientId=cli)
        p.stepSimulation(physicsClientId=cli)
        pos, _ = p.getBasePositionAndOrientation(robot, physicsClientId=cli)
        heights.append(pos[2])
    h0 = pos0[2]; hf = np.mean(heights[-50:]); hp = max(heights)
    results.append((h0, hf, hp))
    p.disconnect(cli)
    print(f"  试验{trial+1}: 初始={h0:.4f}m  最终={hf:.4f}m  峰值={hp:.4f}m")

avg_hf = np.mean([r[1] for r in results])
print(f"\n  平均最终高度: {avg_hf:.4f}m")
print("  ✅ 物理可行！" if avg_hf > 0.05 else "  ❌ 物理障碍——关节力矩不足以从此状态站立")
