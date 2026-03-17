#!/usr/bin/env python3
"""
V80 准备脚本：将 V64 的 obs_dim=23 策略网络权重扩展到 obs_dim=24。
不依赖 pybullet（避免与正在训练的进程冲突），直接操作 PyTorch 权重。
"""
import sys
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, '.')

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

if __name__ == '__main__':
    V64_PATH = Path("trained/backflip_v64/full/best.zip")
    OUT_DIR  = Path("trained/backflip_v80_init/full")
    OUT_PATH = OUT_DIR / "init.zip"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"加载 V64 模型（obs_dim=23）: {V64_PATH}")
    v64_model = PPO.load(str(V64_PATH), device="cpu")
    v64_state = v64_model.policy.state_dict()

    print(f"\n需要扩展的权重层：")
    for key, val in v64_state.items():
        if val.ndim == 2 and val.shape[1] == 23:
            print(f"  {key}: {val.shape}")

    # 扩展权重：为 success_flag 维度添加零列
    new_state = {}
    for key, val in v64_state.items():
        if val.ndim == 2 and val.shape[1] == 23:
            new_val = torch.zeros(val.shape[0], 24, dtype=val.dtype)
            new_val[:, :23] = val
            new_state[key] = new_val
            print(f"  扩展: {key}  {val.shape} → {new_val.shape}")
        else:
            new_state[key] = val

    # 创建纯 gymnasium 的 dummy env（不用 pybullet）
    print(f"\n创建 obs_dim=24 的 dummy 环境...")
    obs_space_24 = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32)
    act_space    = gym.spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)

    class DummyEnv(gym.Env):
        observation_space = obs_space_24
        action_space      = act_space
        def reset(self, **kwargs):
            return np.zeros(24, dtype=np.float32), {}
        def step(self, action):
            return np.zeros(24, dtype=np.float32), 0.0, True, False, {}

    dummy_vec = DummyVecEnv([DummyEnv])
    print(f"  obs_space: {dummy_vec.observation_space}")

    # 创建新 PPO 模型（obs_dim=24）
    new_model = PPO(
        "MlpPolicy", dummy_vec,
        verbose=0, device="cpu",
        n_steps=2048, batch_size=256, n_epochs=10,
        learning_rate=3e-4, ent_coef=0.010, gamma=0.99,
        gae_lambda=0.95, clip_range=0.2,
        policy_kwargs=dict(net_arch=[256, 256]),
        seed=42,
    )

    # 加载扩展后的权重
    print(f"\n加载扩展权重到 obs_dim=24 模型...")
    missing, unexpected = new_model.policy.load_state_dict(new_state, strict=False)
    if missing:
        print(f"  [INFO] 缺失键（随机初始化）: {missing}")
    if unexpected:
        print(f"  [WARN] 多余键（忽略）: {unexpected}")

    # 验证一下权重是否正确
    key0 = "mlp_extractor.policy_net.0.weight"
    loaded = new_model.policy.state_dict()[key0]
    original = v64_state[key0]
    assert loaded.shape == (256, 24), f"形状错误: {loaded.shape}"
    assert torch.allclose(loaded[:, :23], original), "权重拷贝失败！"
    assert torch.all(loaded[:, 23] == 0), "第24维未初始化为0！"
    print(f"  ✅ 权重验证通过: {key0} {loaded.shape}")

    new_model.save(str(OUT_PATH))
    dummy_vec.close()

    print(f"\n✅ V80 初始化模型已保存: {OUT_PATH}")
    print(f"   - obs_dim: 23 → 24（第24维 = success_flag）")
    print(f"   - success_flag 权重列 = 0（训练中梯度逐渐激活）")
    print(f"   - 其余权重保留 V64 完整翻转知识")
