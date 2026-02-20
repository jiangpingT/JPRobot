# ADR-0005: 惩罚渐进机制（Penalty Ramp）设计

**状态**：已采纳
**日期**：2026-02-19
**作者**：JPRobot Team

---

## 背景

强化学习训练初期，机器人需要大量探索：过早引入强惩罚会抑制探索，导致策略陷入保守局部最优。但训练后期，若无惩罚，机器人可能出现关节过度运动、能量浪费等不良行为。

需要一种机制：**早期惩罚弱，随训练推进逐步加强**。

## 决策

实现线性渐进惩罚（Penalty Ramp）：

```python
PENALTY_STEPS = 100_000_000  # 每个 env 实例的步数，penalty 从 0 到最大值

# 在每个 env reset/step 中：
penalty_factor = min(1.0, self.step_counter_session / PENALTY_STEPS)

# 关节惩罚项：
joint_deviation = sum(abs(action - neutral_pose) for action in actions)
reward -= penalty_factor * 0.1 * joint_deviation
```

关键点：`step_counter_session` 是**每个 env 实例**从 0 开始计数，不是全局步数。这意味着每次渐进训练启动新阶段时，惩罚从 0 重新开始。

## 理由

**线性渐进的优势**：
- 简单可解释：在第 X 步时，penalty_factor = X / PENALTY_STEPS
- 无跳跃：不存在阈值触发导致的奖励突变
- 可调：只需修改 `PENALTY_STEPS` 即可控制渐进速度

**PENALTY_STEPS = 100M 的选择依据**：

Stage 4 的目标步数为 40M（总步数 50M - 前3阶段的 10M）。以 100M 为满值：
- 在 Stage 4 结束时（40M 步），penalty_factor ≈ 40%
- 在 Stage 5 结束时（90M 步），penalty_factor ≈ 90%

这确保惩罚在整个训练周期内平滑增加，不会在某个阶段突然变强。

## 历史错误与教训

**PENALTY_STEPS = 15M（初始值）的失败**：

```
Stage 4 开始时（step_counter_session = 0）：penalty_factor = 0%  → reward +75
当 step_counter_session = 3.4M 时：penalty_factor = 22.7%        → reward 急剧下降
当 step_counter_session = 5M 时：penalty_factor = 33%            → reward 变负
```

图表观察：ep_len 从 235 跌至 127，reward 从 +75 跌至 -70。机器人学会了"快速倒下"以避免更多惩罚。

根因：15M 步时惩罚就达到满值，但 Stage 4 需要 40M 步，导致后 25M 步完全在满惩罚下训练，策略完全退化。

**修复方案**：PENALTY_STEPS 从 15M 提升到 100M，保证 Stage 4/5 全程惩罚系数温和渐进。

## 后果

**正面影响**：
- 修复后立即见效：从 best.zip（reward=61）续训，11.1M 步时 reward 跃升至 509
- ep_len 从 63 稳定提升至 248/250（接近最大 episode 长度）
- 训练曲线稳定上升，无明显崩溃

**负面影响**：
- 在不同渐进阶段重新启动时，`step_counter_session` 从 0 开始，早期惩罚重置可能导致策略短暂退步
- 100M 步的满值对 100M 总训练目标来说几乎永远不会达到满强度（仅在 Stage 5 末期接近满值）
