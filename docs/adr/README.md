# Architecture Decision Records (ADR)

本目录记录 JPRobot 项目中的重要架构决策。

## 什么是 ADR？

ADR（Architecture Decision Record）用于记录重要的架构决策，包括：
- **背景**：为什么需要做决策
- **决策**：选择了什么方案
- **理由**：为什么选这个方案
- **后果**：实施后的影响与权衡

## 记录列表

| ADR | 标题 | 状态 |
|-----|------|------|
| [ADR-0001](ADR-0001-ppo-algorithm.md) | 使用 PPO 算法进行强化学习训练 | 已采纳 |
| [ADR-0002](ADR-0002-progressive-training.md) | 渐进式分阶段训练策略 | 已采纳 |
| [ADR-0003](ADR-0003-pybullet-simulation.md) | 使用 PyBullet 作为物理仿真引擎 | 已采纳 |
| [ADR-0004](ADR-0004-reward-function.md) | 奖励函数设计：存活奖励 + 姿态加权前进奖励 | 已采纳 |
| [ADR-0005](ADR-0005-penalty-ramp.md) | 惩罚渐进机制（Penalty Ramp）设计 | 已采纳 |
| [ADR-0006](ADR-0006-snapshot-system.md) | 训练快照与断点续训系统 | 已采纳 |
| [ADR-0007](ADR-0007-llm-dual-mode.md) | LLM 双模式支持（Function Calling + 提示解析） | 已采纳 |

## 状态说明

- **草稿**：决策尚未最终确定
- **已采纳**：决策已实施
- **已废弃**：决策已被新方案取代
- **已取代**：由后续 ADR 取代，见对应记录
