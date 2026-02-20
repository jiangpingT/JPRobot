# ADR-0004: 奖励函数设计：存活奖励 + 姿态加权前进奖励

**状态**：已采纳
**日期**：2026-02-19
**作者**：JPRobot Team

---

## 背景

四足步态学习的奖励设计需要平衡多个目标：
1. **鼓励前进**：机器人需要向目标方向运动
2. **保持稳定**：避免摔倒、翻滚
3. **持续存活**：不能快速结束 episode
4. **抑制不良行为**：防止原地转圈、过度关节运动

过于简单的奖励（只有前进奖励）导致早期探索困难；过于复杂的奖励（大量惩罚项）导致优化困难。

## 决策

采用**三层奖励结构**：

```python
reward = FAC_SURVIVAL                                    # 基础存活奖励
reward += FAC_MOVEMENT * movement_forward * posture_factor  # 姿态加权前进奖励
reward -= penalty_factor * joint_penalty                 # 渐进关节惩罚
```

关键常数（`jprobot/training/env.py`）：

| 常数 | 当前值 | 说明 |
|------|--------|------|
| `FAC_SURVIVAL` | 2.0 | 每步存活奖励（绝对值） |
| `FAC_MOVEMENT` | 1000.0 | 前进奖励放大系数 |
| `PENALTY_STEPS` | 100,000,000 | 惩罚从 0 到最大值的步数 |

## 各项设计理由

### 1. 存活奖励（FAC_SURVIVAL = 2.0）

确保即使前进缓慢，机器人也有动力保持站立而非直接倒下：
```python
reward = FAC_SURVIVAL  # 每步 +2.0，episode 越长总奖励越高
```

历史演变：
- 初始值 0.5：前进奖励（~1.0/步）压倒存活奖励，机器人倾向激进动作
- 调整为 2.0：存活奖励显著，机器人更倾向保持站立

### 2. 姿态加权前进奖励

```python
movement_forward = pos_after[0] - pos_before[0]  # X 轴位移（米/步）
posture_factor = max(0, 1 - abs(euler[1]) / 0.5)  # pitch 偏差越大奖励越低
reward += FAC_MOVEMENT * movement_forward * posture_factor
```

`posture_factor` 的作用：机器人向前移动但身体大幅倾斜时，奖励折扣。鼓励稳定步态而非"摔倒式前进"。

`FAC_MOVEMENT = 1000.0`：典型前进速度约 0.001~0.002 m/step，乘以 1000 后奖励为 1~2/步，与存活奖励量级相当。

### 3. 终止条件（Episode 终止阈值）

```python
terminated = abs(euler[0]) > 0.873 or abs(euler[1]) > 0.873  # roll/pitch > 50°
```

历史演变：
- 初始值 40°（0.7 rad）：过于严格，机器人一旦倾斜即终止，难以学习恢复
- 调整为 50°（0.873 rad）：给予机器人更多恢复机会

## 后果

**正面影响**：
- 姿态加权因子有效抑制"摔倒式前进"行为
- FAC_SURVIVAL=2.0 使 episode 长度稳定在 200+ 步（最大 250 步）
- 奖励信号清晰，训练过程可解释

**调优历史**：

| 参数组合 | reward | ep_len | 问题 |
|---------|--------|--------|------|
| FAC_SURVIVAL=0.5, PENALTY_STEPS=15M | 75（峰值），后跌至-70 | 235→127→64 | 惩罚过早压制 |
| FAC_SURVIVAL=2.0, PENALTY_STEPS=100M | 509（11.1M步） | 248/250 | 当前最优 |

**负面影响**：
- `posture_factor` 基于 pitch 单轴，roll 轴倾斜未纳入奖励折扣（仅作终止条件）
- 关节惩罚项权重需要根据 PENALTY_STEPS 调整，存在过期惩罚风险
