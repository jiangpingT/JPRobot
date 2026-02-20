# ADR-0001: 使用 PPO 算法进行强化学习训练

**状态**：已采纳
**日期**：2026-02-19
**作者**：JPRobot Team

---

## 背景

BittleX 四足机器狗的步态控制是一个连续动作空间问题，需要同时控制 8 个关节的角度。传统 PID 控制器难以适应复杂地形，预设动作序列缺乏泛化能力。因此选择强化学习方法来端到端学习步态策略。

强化学习算法选型时面临以下候选方案：
1. PPO（Proximal Policy Optimization）
2. SAC（Soft Actor-Critic）
3. TD3（Twin Delayed DDPG）
4. TRPO（Trust Region Policy Optimization）

## 决策

选用 **PPO（近端策略优化）** 算法，通过 `stable-baselines3` 库实现。

## 理由

| 维度 | PPO | SAC | TD3 |
|------|-----|-----|-----|
| 训练稳定性 | 高 | 中 | 中 |
| 超参敏感度 | 低 | 高 | 高 |
| On-policy 样本效率 | 中 | 高 | 高 |
| 工程成熟度 | 高（SB3官方支持）| 高 | 中 |
| 适合渐进课程学习 | 是 | 是 | 较难 |

PPO 的核心优势：
- **训练稳定**：裁剪目标函数（clip ratio=0.2）防止策略更新幅度过大
- **超参鲁棒**：对学习率、批大小等超参不敏感，适合长周期实验
- **断点续训友好**：On-policy 特性使模型快照可直接作为续训起点
- **社区生态**：stable-baselines3 提供完善的 PPO 实现，与 Gymnasium 无缝集成

关键超参配置：
```python
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    ent_coef=0.0,
    target_kl=0.05,
)
```

`target_kl=0.05` 作为早停条件，确保每次策略更新不过激。

## 后果

**正面影响**：
- 训练过程可预期，便于诊断和调试
- 快照文件（`.zip`）可在任意阶段续训
- 超参数量少，降低调参成本

**负面影响**：
- On-policy 算法样本效率低于 SAC，同等计算资源下需要更多训练时间
- 无法利用经验回放池，历史数据不可复用

**实测结果**：
- Stage 1（1M步）：reward 21.5，ep_len 105
- Stage 2（5M步）：reward 52.4，ep_len 63.7
- Stage 3（10M步）：reward 61.5，ep_len 64
- Stage 4 重启后（11.1M步）：reward 509，ep_len 248/250（参数调优后）
