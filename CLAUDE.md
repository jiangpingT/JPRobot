# JPRobot - Claude Code 项目记忆

## 用户学习偏好（重要，每次对话都要遵守）

> 机器人领域对用户完全陌生。所有与**机械结构、运动控制、物理仿真、硬件接口**相关的代码，
> 用户都需要详细解释背后的原理，不能只说"做了什么"，要说"为什么这样做、物理上意味着什么"。
> 学习越细致越好，用户在慢慢积累这个领域的知识。

具体要求：
- URDF、关节、舵机、力矩等硬件概念 → 每次都解释清楚
- PyBullet API（getJointStates、setJointMotorControl 等）→ 说明物理含义
- 奖励函数的每个项 → 解释它在鼓励/惩罚什么行为
- 不要假设用户已经知道任何机器人知识

## 项目概述
**具身智能机器人研究项目**，覆盖四足机器人与人形机器人两条技术路线。

- **四足机器人**：BittleX 平台，基于 opencat-gym 改进，PyBullet 物理仿真 + Stable Baselines3 PPO 训练行走策略
- **人形机器人**：正在规划/探索阶段，目标是将具身智能技术延伸到双足人形平台

## 关键架构
- `jprobot/training/env.py` — Gym 环境 v1（匹配原版 opencat-gym，obs=248）
- `jprobot/training/env_v2.py` — Route A 环境（obs=254，+lin_vel_xy +feet_contact_state）
- `jprobot/training/env_velocity.py` — Route B 环境（obs=250，速度命令追踪范式）
- `jprobot/training/progressive.py` — 课程学习编排器（支持 --run-id 并行训练）
- `jprobot/training/train.py` — PPO 训练入口（支持 env_class / trained_dir 参数）
- `jprobot/models/bittle_esp32.urdf` — 机器人 URDF 模型
- `scripts/training_server.py` — Web Dashboard + 3D 可视化服务器（端口 18791）
- `scripts/fixed_eval.py` — 固定方向验收评估（确定性推理，支持 --run-id / --env-class）

## 训练参数（已验证可产生行走）
- 2M 步 / 8 并行环境 / seed=42
- PPO: net_arch=[256,256], lr=3e-4, ent_coef=0.0
- 奖励: FAC_MOVEMENT=1000, FAC_STABILITY=0.1, FAC_ARM_CONTACT=0.01
- penalty_factor 线性增长: 0 → 1.0 over 2M steps
- 期望结果: reward ~1100-1200, 前进 ~1.2m/episode, 0% 手臂触地

## 后空翻训练历史（backflip 专项）

### 最终成果 Phase 2：V64 🎉 落地后恢复站姿！（2026-03-01）
- **路径**：`trained/backflip_v64/full/best.zip`
- **指标**：success=100%, rotation=375.5°, rot@land=358.9°, ep_len=100, reward=10002.7
- **训练方式**：从 V63 热启动，5M步，ent_coef=0.006，W_POST_STAND=50+W_POST_HEIGHT=30
- **意义**：机器人在后空翻落地后，脚着地、身体水平保持（不再仰躺），部分恢复站姿

### 最终成果 Phase 1：V60 🎉 rot@land=360.2° 完美落地！（2026-03-01）
- **路径**：`trained/backflip_v60/full/best.zip`
- **指标**：success=100%, rotation=372.3°, rot@land=360.2°, ep_len=61
- **训练方式**：从 V59 热启动，5M步，ent_coef=0.006，W_LAND_TIMING=2000
- **意义**：机器人在正好完成360°翻转的瞬间落地，物理意义上的"完美后空翻"

### 里程碑：V54 360.3° 完整后空翻（2026-03-01）
- **路径**：`trained/backflip_v54/full/best.zip`
- **指标**：success=100%, rotation=360.3°, rot@land=347°, ep_len=61
- **意义**：首次突破360°旋转，但落地时仍差13°（空中峰值360°，落地时347°）

### 里程碑版本（V41 阶段成果）
- **路径**：`trained/backflip_v41/full/best.zip`
- **指标**：success=100%, rotation=332.1°, ep_len=61, ep_rew=2597
- **意义**：确立"一次性直立加成"反 gaming 机制，是后续一切的基础

### 版本演进路线

| 版本 | rotation | success | gaming 路径 | 关键改动 |
|------|----------|---------|------------|---------|
| V33 | 412° | 100% | 超速(13.65r/s) | 过旋转不惩罚 |
| V37 | 318.4° | 100% | ✅ | W_OVERROT=500, ROTATION_TARGET=2π |
| V38 | 258.8° | 0% | 落地259°收W_LANDING至超时 | MIN_BACKFLIP_STEPS=0（失败） |
| V39 | 284.1° | 0% | 精停于286°门槛下 | 恢复MIN=60，从V37热启 |
| V40 | 259.5° | 0% | 落地259°收W_STABILITY至超时 | W_STABILITY=2.0（失败） |
| V41 | 332.1° | 100% | ✅ | W_UPRIGHTNESS_BONUS=0.5，一次性 |

---

## 后空翻 360° 突破完整复盘（2026-03-01）

### 问题背景
V41 在 332° 达到稳定成功后，rotation 陷入固化——无论怎么调整奖励，策略都卡在某个局部最优，拒绝旋转更多。从 332° 到 360° 走了整整 13 个版本（V42→V54）。

### 三阶段突破路线

**第一阶段：引入旋转完整度奖励（V46-V48）**

问题根因：成功门槛是 286°，agent 旋转到 338° 就落地——因为落地后 episode 结束，多转没有额外奖励。

解法：新增 `W_ROT_COMPLETENESS`，在成功时额外奖励旋转完整度：
```python
# 成功时额外奖励（286°→360° 线性/平方公式）
rot_ratio = min(1.0, max(0.0, (rot_deg - 286.0) / (360.0 - 286.0)))
rot_completeness = rot_ratio ** 2
reward += W_ROT_COMPLETENESS * rot_completeness
```
效果：W_ROT_COMPLETENESS 1000→2000→3000，rotation 随之从 338.9°→344.4°→353.8°→356.1°。

**卡墙**：到 356.1° 后完全固化（std=0）。根因是"286° 基线"的致命缺陷：
- 在 356° 时，agent 已拿到 89.8% 的完整度奖励
- 冲到 360° 只多 10.2%（约 306 分）——梯度太弱，推不动固化策略

---

**第二阶段：公式基线重构 + 门槛倒逼（V49）**

洞察：公式基线应该贴近目标，而非贴近成功门槛。把基线从 286° 提高到 350°：
```python
rot_ratio = min(1.0, max(0.0, (rot_deg - 350.0) / (360.0 - 350.0)))
```
效果：在 356° 时只拿到 37.2% 奖励，360° 还有 62.8% = 1884 分差价，梯度强了 6 倍。

但：策略仍然固化在 356.1°。`ent_coef=0.003` 下 PPO 完全没有探索，梯度再陡也推不动。

**失败实验（V50-V51）**：
- V50：ent_coef=0.008 + 从低版本(V46)热启 → 成功门槛 350° 无法达到 → 0% 成功，训练崩溃
- V51：ent_coef=0.01 + 从 V49 热启 → 探索过强，逃逸到劣质局部最优(340.7°)，退步

---

**第三阶段：ent_coef 甜蜜点搜索（V52-V54）**

核心洞察：需要找到"刚好能打破固化，又不足以退步"的 ent_coef 值。用二分法逼近：

| ent_coef | 热启 | 结果 | 判断 |
|----------|------|------|------|
| 0.003 | V48 | 356.1°（固化） | 太弱 |
| 0.010 | V49 | 340.7°（退步） | 太强 |
| **0.005** | V48 | **358.7°**（+2.6°！） | ✅ 甜蜜点 |
| 0.005 | V52 | 357.1°（小退步） | 在盆地内震荡 |
| **0.007** | V52 | **360.3°**（达成！） | ✅ 最终突破值 |

规律：每个局部最优盆地都有自己的"逃逸 ent_coef"。从 356° 逃出需要 0.005，从 358.7° 盆地逃出需要 0.007。

### 最终有效配置（V54）

```python
# env_backflip.py
W_ROT_COMPLETENESS = 3000.0          # 旋转完整度奖励权重
ROTATION_COMPLETE  = 5.0             # rad≈286°，成功门槛（宽松）

# 成功时奖励计算
rot_ratio = min(1.0, max(0.0, (rot_deg - 350.0) / (360.0 - 350.0)))
rot_completeness = rot_ratio ** 2    # 平方：最后10°梯度极陡
reward += W_ROT_COMPLETENESS * rot_completeness

# train_backflip.py
ent_coef = 0.007   # 突破358.7°盆地的甜蜜点
# 热启动自 V52(358.7°) → trained/backflip_v52/full/best.zip
```

### 三条核心规律（跨项目通用）

**规律一：消灭持续收入流**
任何 per-step 正奖励都会成为 gaming 温床。agent 宁可停在门槛以下收持续奖励，也不愿一次性完成任务。所有奖励应绑定到"成功瞬间"。

**规律二：公式基线贴近目标**
旋转完整度公式的基线应设在当前策略已能稳定达到的位置，让目标区间的梯度最陡。基线越靠近目标，奖励函数对"最后几度"越敏感。

**规律三：ent_coef 分层突破**
PPO 的局部最优"盆地"有深浅之别。浅盆地用低 ent_coef（0.005）可以逃出，深盆地需要更高（0.007）。但过高（0.01+）会导致策略跌出盆地落入差的局部最优。正确做法：
1. 先用 0.005 尝试
2. 如果只是盆地内震荡，升到 0.007
3. 如果退步，降回来换热启点（更早版本）

---

### gaming 根本规律（必须记住）
**Per-step 正奖励 = 持续收入流 = gaming 温床**
- agent 宁可落地259°站稳收W_LANDING(10/步)×61步=610分，也不愿翻完拿W_SUCCESS(1000分一次)
- 折扣+风险权衡后，gaming往往比真实完成更"合理"
- **解法**：所有站姿奖励绑定到"成功瞬间"（一次性），消灭持续收入流

### V41 关键代码（env_backflip.py）
```python
W_STABILITY         = 0.0    # 彻底关闭per-step站稳奖励
W_UPRIGHTNESS_BONUS = 0.5    # 成功时直立加成系数
W_OVERROT           = 500.0  # 过旋转惩罚（V37引入）
ROTATION_TARGET     = 2*math.pi  # 360°，超过此值开始惩罚
MIN_BACKFLIP_STEPS  = 60     # 步数门槛（V39恢复）

# 成功奖励（step函数中）
if self.training_phase == "full" and self._success:
    uprightness = math.exp(-2.0 * (pitch**2 + roll**2))
    reward += W_SUCCESS * (1.0 + W_UPRIGHTNESS_BONUS * uprightness)
    # 直立落地→1500, 翻倒落地→1000, gaming(259°)→820（差距明确）
```

---

## 落地稳定性优化完整复盘（2026-03-01）

### 问题背景
V54 实现了 rotation=360.3° 后，发现一个"最后13°"的隐患：机器人在空中峰值达到 360°，但落地时（height < 0.10m）只转到 347°。空中旋转和落地时刻不同步——机器人实际上是"转过了头再落地"的。

目标：把 rot@land（落地瞬间累积旋转）从 347° 推到 360°，让旋转完成和落地同步。

---

### 发现1：W_LANDING 是死代码

检查 env_backflip.py 时发现致命 bug：`_airborne` 是单向 True 标志（一旦离地就永远为 True），从未重置为 False。

```python
# 这段条件永远不会触发！
if not self._airborne:   # 起跳后 _airborne=True，此条件永远 False
    reward += W_LANDING
```

`ROTATION_GATE=352°` 也是死代码，完全无效。V54 的落地奖励从未触发过。

---

### 发现2：梯度盲区（V56 教训）

W_ROT_COMPLETENESS 公式基线=350°，而实际 rot@land=342-347°——整个有效范围在基线之下！

```python
# 基线350°时：rot@land=347° → rot_ratio = (347-350)/10 = 负数 → 截断到 0
# 梯度=0！agent 无法分辨 342° 和 349° 的落地有什么区别
rot_ratio = min(1.0, max(0.0, (rot_deg - 350.0) / (360.0 - 350.0)))
```

这就是为什么 V55 一动不动：奖励函数对 rot@land 的整个改进空间毫无感知。

---

### 突破：双梯度修复（V57）

**修复1：基线下移（350°→335°）**，让当前 rot@land 范围（342-347°）落入有梯度区间：
```python
# V57修复：基线335°，342°落地也能得到梯度
rot_ratio = min(1.0, max(0.0, (rot_deg - 335.0) / (360.0 - 335.0)))
```

**修复2：新增 W_LAND_TIMING=2000**，在 `just_landed`（高度首次低于0.10m）瞬间一次性触发：
```python
# 落地瞬间额外奖励，基线330°→360°（平方公式）
if self._landed_after_launch and not was_landed_before:
    rot_at_land = math.degrees(max(0.0, -self._pitch_accumulated))
    land_timing_ratio = min(1.0, max(0.0, (rot_at_land - 330.0) / (360.0 - 330.0)))
    reward += W_LAND_TIMING * land_timing_ratio ** 2
```

347°→360° 差价对比：
| 落地角度 | W_ROT_COMPLETENESS | W_LAND_TIMING | 合计 |
|---------|-------------------|--------------|------|
| 347° | 691 | 642 | **1333** |
| 360° | 3000 | 2000 | **5000** |
| 差价 | | | **3667分** |

V57 结果：rot@land=351.5°（+4.5°！首次突破）

---

### ent_coef 甜蜜点（落地优化阶段）

落地稳定性阶段的 ent_coef 甜蜜点与旋转突破阶段不同：

| ent_coef | 效果 | 判断 |
|----------|------|------|
| 0.007 | rot@land=342.3°（退步） | 太强，逃到差的局部最优 |
| **0.005** | 完全固化（std=0） | 太弱 |
| **0.006** | 3次连续突破 | ✅ 甜蜜点 |

规律：旋转突破阶段甜蜜点=0.007，落地稳定阶段甜蜜点=0.006。不同优化目标有不同的 ent_coef 要求。

---

### 物理洞察：策略性过旋转

机器人发展出一种独特策略：在空中故意多转约12°，然后落地前减速。

```
V57: 空中峰值364.7° → 落地351.5°（差13.2°）
V58: 空中峰值369.1° → 落地357.4°（差11.7°）
V59: 空中峰值371.0° → 落地359.0°（差12.0°）
V60: 空中峰值372.3° → 落地360.2°（差12.1°）
```

减速差（~12°）是物理常数，由机器人伸腿减速动作决定。机器人学会了"瞄准372°，落在360°"的策略。

---

### V54→V60 完整进化路线

| 版本 | rot@land | 关键改动 |
|------|----------|---------|
| V54 | 347.0° | 基准线（旋转360°已达成） |
| V55 | 347.2° | ROTATION_GATE调参（死代码，无效） |
| V56 | 342.3° | ent_coef=0.007（退步） |
| V57 | 351.5° | 双梯度修复突破（+4.5°！） |
| V58 | 357.4° | ent_coef=0.006甜蜜点（+5.9°！） |
| V59 | 359.0° | 继续爬升（距360°仅1°） |
| V60 | 360.2° | 完美落地！任务达成！ |

### 新增核心规律

**规律四：基线必须在当前策略可达范围内**
奖励基线（如 W_ROT_COMPLETENESS 的起算点）必须低于当前策略实际达到的值，否则梯度=0，agent 无法改进。

**规律五：不同优化目标有不同 ent_coef 甜蜜点**
旋转完整性阶段需要 0.007 打破固化；落地精确性阶段需要 0.006（0.007 反而退步）。每进入新的优化目标，都要重新搜索最优 ent_coef。

---

## 落地后站立优化复盘（V61-V64，2026-03-01）

### 问题背景
V60 完美落地后，发现机器人落地后仰躺不能站立。增加 post-success 站立机制。

### 核心机制（V61 新增）
```python
# 成功后额外运行 POST_SUCCESS_STEPS 步，给站稳奖励
if self._success and self._post_success_steps < POST_SUCCESS_STEPS:
    uprightness_now = math.exp(-2.0 * (pitch ** 2 + roll ** 2))
    height_ratio = min(1.0, max(0.0, (height - 0.04) / (0.10 - 0.04)))
    reward += W_POST_STAND * uprightness_now + W_POST_HEIGHT * height_ratio
    self._post_success_steps += 1
# 站稳阶段结束后才终止 episode（ep_len 从61→100）
```

### 版本进化

| 版本 | W_POST_STAND | W_POST_HEIGHT | avg uprightness | 视觉结果 |
|------|-------------|--------------|-----------------|---------|
| V61 | 10 | 0 | ~0 | 仰躺，梯度太弱 |
| V62 | 25 | 0 | 0.75（22°倾斜） | 趴地，progress |
| V63 | 50 | 0 | ~0.95 | 平趴骗分！uprightness盲区 |
| V64 | 50 | 30 | ~0.95+height_ratio=0.20 | 脚着地，身体略抬起 |

### 关键发现：uprightness 盲区
`uprightness = exp(-2*(pitch²+roll²))` 只测体态角度，无法区分：
- "四腿站立"（body在0.10m高处，pitch=0）→ uprightness=1.0 ✅
- "趴在地上"（body在0.04m贴地，pitch=0）→ uprightness=1.0 ❌（骗分！）

解法：增加 `height_ratio = (height-0.04)/(0.10-0.04)` 奖励身体离地高度。

### 最终成果（V64）
- ep_len=100±0（40步站稳阶段正常运行）
- reward=10002.7（历史最高）
- 脚着地、身体水平（不仰躺），body高度≈0.052m（vs目标0.10m）
- 未完全站立（物理约束：从倒地恢复到完全站立需要更长训练）

### 规律六：奖励指标盲区需要多维信号
单一角度奖励（uprightness）被 agent 用"趴地"行为满足，需要额外的高度信号闭合盲区。
设计站立奖励时应同时激励：角度（body水平）+ 高度（body离地）+ 脚接触（足部着地）。

---

## 人形万向行走完整复盘（2026-03-03 完成，2026-03-05 归档）

### 最终成果
- **最优模型**：`trained/humanoid_velocity/best_v6_backup.zip`（vel_error=0.4469）
- **环境**：`jprobot/training/env_humanoid_velocity.py`（obs=379 维）
- **算法**：SAC，4 并行环境，线性 lr=7.3e-4，use_sde=True

### 版本进化完整记录

| 版本 | vel_error | 关键改动 | 累计步数 | 结论 |
|------|-----------|---------|---------|------|
| v1 | 1.639 | SIGMA=0.25（太小） | 3M | ❌ 无梯度，机器人不走 |
| v2 | 0.582 | SIGMA=1.0，W_VEL=5.0 | +5M（热启v1） | ✅ 基准，零摔倒 |
| v3 | 0.539 | VX_RANGE 2.0→1.2 | +3M（热启v2） | ✅ 改善7%，3M优于v2的5M |
| v4 | 0.514 | wz左转偏置70%+WZ_ERROR_WEIGHT=0.70 | +3M（热启v3） | ✅ 改善4.6% |
| v5 | 0.712 | SIGMA 1.0→0.75（失败） | +3M（热启v4） | ❌ 退步38% |
| **v6** | **0.4469** | Method B（误差变化率奖励W_ERROR_DELTA=1.0） | +3M（热启v4） | ✅ **历史最优，首破0.45** |
| v7 | 0.617 | Method A（3步误差历史obs 379→388） | 5M（从零） | ❌ 未超v6 |
| v7热启 | 0.570 | v7继续热启 | +3M | ❌ 仍未超v6 |

### 核心规律（万向行走）

**规律一：SIGMA 必须与初始误差同量级**
- SIGMA=0.25 时，vel_error=1.6 给出奖励≈0.004（梯度消失）
- SIGMA=1.0 使任何速度方向都有梯度，是行走训练的正确设置
- **禁止随意改 SIGMA**：改了必须同步修改评估脚本，否则比较无意义

**规律二：速度命令范围不能超过物理上限**
- MuJoCo Humanoid-v4 实际可达最高速约 1.2-1.5 m/s
- VX_RANGE=2.0 导致大量"追不上"的命令，拖累整体 vel_error
- 缩小到 1.2 后 3M 步即超越 5M 步的旧版本

**规律三：方向偏置采样是双刃剑**
- v4 左转偏置（70%）：左转略改善，后退严重退步（0.495→0.615）
- 偏置采样等于压缩其他方向训练数据，必然此消彼长
- v6 恢复均匀采样 + delta_reward 双效，左转反而大幅改善（0.584→0.363）

**规律四：Method B（误差变化率）是目前最有效的改进**
```python
# 每步奖励加入误差下降趋势
error_delta  = vel_error - self._prev_vel_error
delta_reward = W_ERROR_DELTA * (-error_delta)   # 误差减小→正奖励
self._prev_vel_error = vel_error
```
物理意义：鼓励策略"持续在变好"，而不只是"当前误差多小"

**规律五：热启积累比单次大步数更有效**
- v6 = 在 v4（9M累计）基础上热启3M，总12M → vel_error=0.447
- v7 = 从零5M+热启3M，总8M → vel_error=0.570
- 每次热启保留了所有历史学习成果，是 off-policy SAC 的核心优势

**规律六：obs 维度变化=无法热启，代价极大**
- Method A（历史观测）把 obs 379→388，必须从零训练
- 8M 步从零仍不如 12M 步热启积累
- 在这个任务上，架构升级的收益<热启积累的收益

### v6 关键配置（最优参数）
```python
VX_RANGE      = (-1.0, 1.2)   # m/s，贴近物理上限
WZ_LEFT_BIAS  = 0.5            # 均匀采样
W_VEL         = 5.0
SIGMA         = 1.0            # 核心参数，不要动
WZ_ERROR_WEIGHT = 0.70
W_ERROR_DELTA = 1.0            # Method B，关键改进
```

---

## 人形跑步训练完整复盘（2026-03-05~07 完成归档）

### 最终成果（run_v4）🎉
- **模型**：`trained/humanoid_run/best.zip`
- **环境**：`jprobot/training/env_humanoid_run.py`（obs=379，兼容v6热启）
- **脚本**：`scripts/train_humanoid_run.py`、`scripts/eval_humanoid_run.py`、`scripts/viz_humanoid_run.py`
- **验收（20局）**：ep_len=1000 满分，airborne%=26.8%，vx=1.00 m/s，ep_rew=13885（历史最高）
- **视觉**：Playwright 截图确认机器人完全直立跑步（头顶最高点，双脚交替踏地）✅

### 版本进化路线

| 版本 | ep_rew | airborne% | vx | 问题 | 关键改动 |
|------|--------|-----------|-----|------|---------|
| run_v1 | — | 16.4% | 0.50 m/s | 腾空奖励从未触发 | 基础版，VX_RUN_THRESHOLD=1.5（超物理上限）|
| run_v2 | 8586 | 22.6% | 0.74 m/s | 爬行gaming（水平滑行） | VX_RUN_THRESHOLD 1.5→0.5，VX_RANGE 上限 3.5→2.0 |
| run_v3 | 10474 | 31.4% | 1.10 m/s | 仍前倾约45° | W_UPRIGHT=3.0（torso_z 线性直立奖励） |
| **run_v4** | **13885** | **26.8%** | **1.00 m/s** | ✅ 达标 | W_UPRIGHT 3.0→6.0（超过速度追踪权重，直立优先）|

### 三大核心规律（本项目新增）

**规律七：阈值必须在物理可达范围内**
MuJoCo Humanoid-v4 最高速约 1.2-1.5 m/s（关节力矩硬限制）。
VX_RUN_THRESHOLD=1.5 导致腾空奖励永远触发不了——梯度为零，等于死代码。
降到 0.5 后腾空奖励真正触发，airborne% 从 16% 升到 22%+。
**规律：所有门槛/阈值设计前，先查物理上限，不要拍脑袋。**

**规律八：直立奖励权重必须超过速度追踪权重才能主导行为**
W_UPRIGHT=3.0 < W_VEL=5.0：机器人爬行（拿速度分）+ 轻微直立（拿部分直立分），合计更优。
W_UPRIGHT=6.0 > W_VEL=5.0：直立成为最大单项奖励，机器人必须站立才能最大化总奖励。
**规律：要改变行为优先级，对应奖励权重必须明确超过竞争项，而非接近。**

**规律九：爬行 gaming 的识别方法**
- 数据看起来好（vel_error 小、airborne 高）但视觉截图机器人水平
- torso_z ≈ 1.0-1.1m（刚好在终止线以上，但远低于正常站立 1.25m）
- 速度 vx 合理但姿势畸形
- **诊断工具**：一定要用 Playwright/viz 视觉验证，数字好看不等于动作对**

### 腾空检测实现（已验证）
```python
import mujoco

def _both_feet_airborne(self) -> bool:
    """利用 cfrc_ext（外部接触力）检测双脚是否腾空。"""
    data  = self.env.unwrapped.data
    model = self.env.unwrapped.model
    lf = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_foot")
    rf = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_foot")
    lf_force = float(np.linalg.norm(data.cfrc_ext[lf, :3]))
    rf_force = float(np.linalg.norm(data.cfrc_ext[rf, :3]))
    return lf_force < CONTACT_THRESH and rf_force < CONTACT_THRESH
```
- `cfrc_ext[body_id, :3]` = 该 body 所受外部接触力向量（世界坐标系）
- `CONTACT_THRESH=1.0 N`：脚踩地时力 >>1N，腾空时 ≈0
- body 名称：`"left_foot"` / `"right_foot"`（Humanoid-v4 标准命名）

### run_v4 关键配置（最优参数）
```python
# env_humanoid_run.py
VX_RANGE          = (-0.5, 2.0)   # m/s，贴近物理上限
VX_RUN_THRESHOLD  = 0.5           # m/s，必须低于实际可达速度
W_AIRBORNE        = 2.0           # 腾空相奖励
W_VEL             = 5.0           # 速度追踪奖励
W_UPRIGHT         = 6.0           # 直立奖励（v4：必须 > W_VEL 才能主导行为）
UPRIGHT_Z_MIN     = 1.0           # Humanoid-v4 健康下限
UPRIGHT_Z_TARGET  = 1.3           # 完全直立约 1.25-1.35m
W_ERROR_DELTA     = 1.0           # 误差变化率奖励（继承自万向行走 v6）
```

---

## 血泪教训（11 Bug 复盘）

### 致命级（必须牢记）

0. **奖励函数必须堵死 gaming 捷径**（Backflip v1 教训，2026-02-23）
   Backflip v1 训练 6.5M 步、旋转始终卡在 60°、成功率 0%。
   根因：落地奖励（W_LANDING=10）无旋转门槛，agent 发现"跳起不翻直接落地"
   能同时拿 W_JUMP + W_LANDING + W_POSE_GUIDE，总收益远超费力旋转。
   修复：① W_ROTATION: 5→15；② 落地奖励加 90° 旋转门槛；③ 腾空不旋转每步 -0.5。
   规律：每项奖励都要问"agent 能不做 X 也拿到这个奖励吗？"

1. **永远不要硬编码 PyBullet 关节索引**
   URDF 中 fixed 关节也占编号。必须用 `getJointInfo` 动态发现 REVOLUTE 关节。
   错误: `joint_indices = [0,4,1,5,2,6,3,7]` — 3 个指向 FIXED 关节，后腿瘫痪。
   正确: 遍历 getNumJoints 过滤 JOINT_REVOLUTE → `[1,2,4,5,7,8,10,11]`

2. **物理量必须检查单位和缩放**
   角速度 raw 值 ~5-10 rad/s，原版先 ×0.1 再 clip[-1,1] 再平方。
   缺少 ×0.1 导致 stability penalty 大 100 倍，agent 学会"不动"。

3. **opencat-gym 的 arm_contact 是逐 link 计数**
   原版对 [1,2,4,5] 四个 link 逐一调用 getContactPoints，每个触地 +1（最多 +4/步）。
   用 break 只记 1 次会导致手臂惩罚弱 4 倍。

### 严重级

4. **闭环 vs 开环关节控制**
   必须每步从 PyBullet `getJointStates` 读实际角度，不能用 Python 变量追踪。
   开环累积误差，agent 看不到碰撞/外力导致的偏差。

5. **stepSimulation 时序必须匹配原版**
   原版 3 次 step 分散在不同位置（模拟串口延迟），不能集中到一起。

6. **resetSimulation() vs 快速 reset**
   原版每个 episode 完全重建物理世界。快速 reset 可能残留碰撞/摩擦状态。

7. **初始姿态 [1,0,1,0,1,0,1,0]*50**
   肩/髋=50°，肘/膝=0°（自然站姿）。全部设 50° 不是正确站姿。

### 中等级

8. URDF effort/velocity 应为 0（由 changeDynamics 设置 maxJointVelocity）
9. start_pos = [0,0,0.08]（不是 0.1）
10. joint_history 初始化应填入初始姿态（不是全零）

## 调试方法论

- **A/B 对比测试是最有效的调试手段**: 同时运行原版和我们的代码，逐步对比 reward 曲线
- 前 9 步 reward 完全一致 → 证明代码已等价
- 比猜测/单独改参数有效 100 倍

## 最佳启动 / 重启方式

### 一键训练栈（推荐，自动带 Dashboard + 守护进程）

```bash
# 新训练（训练 + Dashboard 一起启动，Dashboard 掉线自动拉起）
bash scripts/train.sh --curriculum multidir_v3_right_refine --auto

# 断点续训（训练中断后用这个，不丢进度）
bash scripts/train.sh --resume --curriculum multidir_v3_right_refine --auto

# 并行两条路线（各开一个终端）
bash scripts/train.sh --curriculum env_v2_continue --run-id route_a --auto
bash scripts/train.sh --curriculum velocity_v2    --run-id route_b --auto
```

训练日志自动写入 `./log/train_YYYYMMDD_HHMMSS.log`，Dashboard 在 http://127.0.0.1:18791/dashboard

---

### Dashboard 单独管理

```bash
bash scripts/dev_dashboard.sh start      # 前台启动（默认，Ctrl+C 停止）
bash scripts/dev_dashboard.sh start-bg   # 后台启动（挂机用）
bash scripts/dev_dashboard.sh stop       # 停止
bash scripts/dev_dashboard.sh restart    # 重启（Dashboard 卡了用这个）
bash scripts/dev_dashboard.sh status     # 查看是否在跑
```

---

### 评估 / 验收

```bash
# 快速检查（5局，约1分钟）
python scripts/fixed_eval.py --episodes 5

# 正式验收（20局/方向）
python scripts/fixed_eval.py --run-id route_a --env-class BittleGymEnvV2
python scripts/fixed_eval.py --run-id route_b --env-class BittleGymEnvVelocity
```

---

### 其他常用

```bash
# 本地 PyBullet GUI 可视化（看机器人实际跑起来什么样）
python -m jprobot.training.enjoy

# A/B 对比测试（调试奖励函数用）
bash scripts/run_ab_test.sh
python scripts/compare_logs.py
```

---

## 运行命令（底层，不常用）

```bash
# 直接调训练入口（不带 Dashboard）
conda activate jprobot
python -m jprobot.training.progressive --curriculum simple --auto
```

## 文件清理提醒
trained/snapshots/ 会积累大量快照文件（几百个），定期清理只保留:
- best.zip（最佳模型）
- curriculum_*.zip（阶段快照）
- 少量里程碑快照
