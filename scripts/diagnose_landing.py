#!/usr/bin/env python3
"""诊断后空翻成功后第1帧的真实机器人状态。"""
import sys, numpy as np
sys.path.insert(0, '.')
from stable_baselines3 import PPO
from jprobot.training.env_backflip import BittleBackflipEnv, BF_FRAMES_RAD
import pybullet as p

model = PPO.load("trained/backflip_v71/full/best.zip", device="cpu")
env   = BittleBackflipEnv(training_phase="full", render_mode=None, disable_rsi=True)

records = []
for ep in range(5):
    obs, _ = env.reset()
    recorded = False
    for _ in range(300):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        if info.get("success") and not recorded:
            pos, orn = p.getBasePositionAndOrientation(env.robot_id, physicsClientId=env.physics_client)
            lin_vel, ang_vel = p.getBaseVelocity(env.robot_id, physicsClientId=env.physics_client)
            js = p.getJointStates(env.robot_id, env.joint_id[:8], physicsClientId=env.physics_client)
            euler = p.getEulerFromQuaternion(orn)
            records.append({
                "h": pos[2], "pitch": np.degrees(euler[1]), "roll": np.degrees(euler[0]),
                "vz": lin_vel[2], "wn": np.linalg.norm(ang_vel),
                "joints": np.degrees([s[0] for s in js]),
            })
            recorded = True
        if term or trunc:
            break

env.close()
print("\n=== 后空翻成功后第1帧 ===")
for i, r in enumerate(records):
    print(f"EP{i+1}: h={r['h']:.4f}m  pitch={r['pitch']:.1f}°  roll={r['roll']:.1f}°  vz={r['vz']:.3f}  w={r['wn']:.3f}")
    print(f"       joints: {np.round(r['joints'],1)}")
print(f"\n目标帧4: {np.round(np.degrees(BF_FRAMES_RAD[4]),1)}")
avg_j = np.mean([r['joints'] for r in records], axis=0)
print(f"平均实际: {np.round(avg_j,1)}")
print(f"差值:     {np.round(np.degrees(BF_FRAMES_RAD[4])-avg_j,1)}")
