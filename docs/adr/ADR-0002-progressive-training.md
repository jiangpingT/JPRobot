# ADR-0002: 渐进式分阶段训练策略

**状态**：已采纳
**日期**：2026-02-19
**作者**：JPRobot Team

---

## 背景

四足步态学习从随机初始化策略开始，面临稀疏奖励问题：机器人在站立和迈步之前几乎无法获得前进奖励。直接用 100M 步训练目标进行单次训练，早期阶段缺乏有效监督，容易陷入局部最优（如原地抖动）。

同时，训练过程中需要逐步提升任务难度（例如惩罚系数渐进加强），单阶段训练难以平衡早期探索和后期精细化。

## 决策

采用**渐进式分阶段（Progressive Staged）训练策略**，将总训练分为 5 个累计步数里程碑：

| 阶段 | 累计步数目标 | 本阶段步数 | 主要目标 |
|------|------------|----------|---------|
| 1    | 1M         | 1M       | 学会基本站立和迈步 |
| 2    | 5M         | 4M       | 步态趋于稳定 |
| 3    | 10M        | 5M       | 前进奖励开始显现 |
| 4    | 50M        | 40M      | 步态精细化，惩罚系数逐步加强 |
| 5    | 100M       | 50M      | 最终目标，完全稳定步态 |

阶段间自动健康检查，不健康则停止并支持 `--resume` 续训。

## 实现

`jprobot/training/progressive.py` 中的 `ProgressiveTrainer` 控制：

```python
stages = [1_000_000, 5_000_000, 10_000_000, 50_000_000, 100_000_000]

for stage in stages[start_idx:]:
    trainer.train(stage_steps)
    metrics = health_check(metrics)
    if not metrics["healthy"]:
        break
    save_snapshot(stage_name)
    update_state(stage_idx + 1)
```

状态持久化到 `trained/progressive_state.json`，包含：
- `stage_idx`：下次训练的起始阶段
- `total_steps`：已完成的总步数
- `planned_stages`：完整阶段计划（供 Dashboard 计算进度）
- `stages`：各阶段历史指标

## 理由

- **课程学习（Curriculum Learning）**：由简到难，避免早期稀疏奖励导致的探索困难
- **快速反馈**：每个里程碑后自动健康检查，早发现训练异常
- **断点续训**：每阶段结束自动保存快照，任意阶段可恢复
- **灵活调整**：发现问题时可回滚阶段索引，用新参数重训特定阶段

## 后果

**正面影响**：
- Stage 1 结束时（1M步）reward 从 3.9 升至 22.6，证明渐进策略有效引导探索
- 断点续训机制在参数调整（PENALTY_STEPS 15M→100M）后成功从 best.zip 恢复

**负面影响**：
- 阶段边界处策略可能出现短暂退步（新惩罚系数引入）
- `planned_stages` 字段需要在训练进程启动时写入，跨阶段重启需手动确认

**关键经验**：
- Stage 4（50M）首次因 PENALTY_STEPS=15M 过短导致崩溃（reward 从 +75 跌至 -70）
- 调整 PENALTY_STEPS=100M 后，从 best.zip 续训，11.1M 步时 reward 达 509，ep_len 248/250
