# JPRobot 训练手册

> 本文档记录 BittleX 强化学习训练的完整工程设计，包括架构、命令、参数、调试指南和已知问题。
>
> 变更联动请同时参考：`docs/JPROBOT_CHANGE_CHECKLIST.md`

---

## 快速参考

### 推荐方式：一键脚本（自动重启 Dashboard）

```bash
# 从头开始，全自动跑完所有阶段，同时启动 Dashboard
./scripts/train.sh --auto

# 从上次断点续训
./scripts/train.sh --resume --auto

# 从头开始，每阶段结束手动确认
./scripts/train.sh
```

> `train.sh` 会自动：① 杀掉旧 Dashboard → ② 后台启动训练 → ③ 5秒后重启 Dashboard 指向新日志。
> Dashboard 地址：http://127.0.0.1:18791/dashboard

### 手动方式

```bash
# 渐进训练：1M → 5M → 10M → 50M → 100M 步，全自动
KMP_DUPLICATE_LIB_OK=TRUE python -m jprobot.training.progressive --auto

# 渐进训练，每阶段结束手动确认是否继续
KMP_DUPLICATE_LIB_OK=TRUE python -m jprobot.training.progressive

# 中途停了，从上次进度恢复
KMP_DUPLICATE_LIB_OK=TRUE python -m jprobot.training.progressive --resume

# 自定义阶段（单位：百万步）
KMP_DUPLICATE_LIB_OK=TRUE python -m jprobot.training.progressive --stages 1 5 10 --auto

# 直接训练（不走渐进控制）
KMP_DUPLICATE_LIB_OK=TRUE python -m jprobot.training.train --timesteps 5000000

# 从快照续训
KMP_DUPLICATE_LIB_OK=TRUE python -m jprobot.training.train --resume trained/snapshots/best.zip

# 多方向单模型课程（2M 快速验证）
KMP_DUPLICATE_LIB_OK=TRUE python -m jprobot.training.progressive --curriculum multidir_v1 --auto

# 多方向精调课程（基于当前 best.zip，再训 2M）
KMP_DUPLICATE_LIB_OK=TRUE python -m jprobot.training.progressive --curriculum multidir_v2_refine --auto

# 右方向定向精调（短程 1M 验证）
KMP_DUPLICATE_LIB_OK=TRUE python -m jprobot.training.progressive --curriculum multidir_v3_right_refine --auto

# 可视化已训练模型
KMP_DUPLICATE_LIB_OK=TRUE python -m jprobot.training.enjoy --model trained/snapshots/best.zip --episodes 3

# 单独启动 Dashboard（手动指定日志）
python scripts/training_server.py --log /path/to/task.output
```

> **注意**：macOS 上必须加 `KMP_DUPLICATE_LIB_OK=TRUE`，否则 OpenMP 冲突崩溃。
> **Python 环境**：`/opt/homebrew/Caskroom/miniforge/base/envs/jprobot/bin/python`
> **注意**：多方向训练把观测从 246 维升级到 248 维，旧 `best.zip` 不能直接续训。

---

## 文件结构

```
jprobot/training/
├── env.py          仿真环境（状态/动作/奖励）
├── train.py        单次 PPO 训练 + 快照系统
├── progressive.py  渐进阶段控制器
└── enjoy.py        可视化已训练模型

trained/
├── bittle_ppo.zip              最终模型（每次训练结束覆盖）
├── progressive_state.json      渐进训练完整进度（用于 --resume）
├── direction_eval.json         多方向评估摘要（每轮 rollout 更新）
├── direction_eval_history.jsonl 多方向评估时间序列
├── snapshots/
│   ├── best.zip                当前历史最优模型（始终维护）
│   ├── step_5.0M_rew_234.zip   奖励创新高时自动命名保存
│   ├── stage_1M.zip            每个阶段结束时的快照
│   └── manifest.json           所有快照的历史记录
└── checkpoints/
    └── bittle_ppo_*_steps.zip  每 2M 步周期性保存（安全网）
```

---

## 模块架构

### multidir_v1 — 多方向单模型课程

目标：单一 PPO 策略学会按目标方向移动（前/后/左/右）。

阶段设计（累计步数）：
1. `0.5M`：固定前进（稳定步态）
2. `1.2M`：偏向前进混合采样（`[0.5, 0.2, 0.15, 0.15]`）
3. `2.0M`：均匀四方向采样（`[0.25, 0.25, 0.25, 0.25]`）

运行命令：

```bash
KMP_DUPLICATE_LIB_OK=TRUE python -m jprobot.training.progressive --curriculum multidir_v1 --auto
```

### multidir_v2_refine — 多方向精调课程（弱方向补强）

目标：在已有 `multidir_v1` 模型基础上，重点提升 `left/right/backward` 的稳定性与进度。

阶段设计（累计步数）：
1. `1.0M`：弱方向偏置采样（`[0.2, 0.2, 0.3, 0.3]`，顺序为前/后/左/右）
2. `2.0M`：均匀四方向再平衡（`[0.25, 0.25, 0.25, 0.25]`）

运行命令：

```bash
KMP_DUPLICATE_LIB_OK=TRUE python -m jprobot.training.progressive --curriculum multidir_v2_refine --auto
```

### 方向评估口径说明（务必区分）

训练目录里有两类方向评估，作用不同：

1. `trained/direction_eval.json`（训练中窗口统计）
- 由训练回调自动写入。
- 口径是最近窗口（默认 200 episodes）聚合。
- 适合看趋势：四方向是否整体在变好。
- 不适合作为最终验收结论（受训练探索噪声和窗口混合影响）。

2. `trained/fixed_direction_eval.json`（训练后固定方向回放）
- 训练结束后手动评估生成（通常每方向固定 N 局）。
- 使用同一模型（一般 `trained/snapshots/best.zip`）做定向离线验收。
- 适合做版本对比和上线前判断，能更真实暴露单方向短板。

建议解读顺序：
- 先看 `fixed_direction_eval.json`（验收结论）。
- 再看 `direction_eval.json`（训练趋势是否一致）。

### env.py — 仿真环境

**观测空间（248 维）**

| 维度 | 内容 | 范围 |
|---|---|---|
| 0-3 | 机体姿态四元数 (x,y,z,w) | [-1, 1] |
| 4-5 | 角速度 roll/pitch (×0.1) | [-1, 1] |
| 6-245 | 关节角历史 30帧 × 8关节（归一化） | [-1, 1] |
| 246-247 | 目标方向向量 `(dx, dy)` | [-1, 1] |

**动作空间（8 维连续）**

每个动作是关节角度的增量（最大 ±11°），映射到 8 个关节：
`shoulder_left, elbow_left, shoulder_right, elbow_right, hip_right, knee_right, hip_left, knee_left`

**奖励函数**

```
总奖励 = 目标方向移动奖励 - penalty_factor × (平滑惩罚 + 稳定惩罚 + 手臂接触惩罚 [+ 可选高度项])

目标方向移动奖励 = FAC_MOVEMENT × dot([Δx, Δy], [dx, dy])
  其中 `[dx, dy]` 是本回合目标方向向量（前/后/左/右）

penalty_factor = min(1.0, step_counter_session / PENALTY_STEPS)
  # 从 0 线性增长到 1，让机器狗先学会走路再被要求走得好
```

**关键超参数**

| 参数 | 当前值 | 说明 |
|---|---|---|
| `FAC_SURVIVAL` | 0.5 | 每步固定存活奖励（鼓励活得更久） |
| `FAC_MOVEMENT` | 1000.0 | 前进奖励放大系数 |
| `FAC_ORIENTATION` | 5.0 | 姿态倾斜惩罚（越小越宽松） |
| `FAC_HEIGHT` | 100.0 | 低于最小高度惩罚（防止爬行） |
| `FAC_ARM_CONTACT` | 2.0 | 手臂/肘部触地惩罚 |
| `PENALTY_STEPS` | 15,000,000 | 惩罚从 0 到满值的步数（越大越宽松） |
| `MIN_HEIGHT` | 0.045 m | 机体最小可接受高度 |
| `EPISODE_LENGTH` | 250 | 每轮最大步数 |
| `STEP_ANGLE` | 11° | 每步最大关节角变化 |

**终止条件**：roll 或 pitch 超过 50°（≈0.873 rad）时 terminated=True，当步奖励清零。（之前是 40°，放宽后给机器狗更多恢复机会）

**注意**：`step_counter_session` 在每个 env 实例的生命周期内累计（不随 episode 重置），但每次新建 env 时从 0 开始。在渐进训练中，每个 stage 新建 env，惩罚系数从 0 重新爬坡——这是有意为之，给每个 stage 提供温和的早期学习环境。

---

### train.py — 单次训练

**PPO 超参数**

| 参数 | 值 | 说明 |
|---|---|---|
| `learning_rate` | `linear_schedule(3e-4 → 0)` | 线性衰减到 0，随进度下降 |
| `target_kl` | 0.05 | KL 上限，超过则停止本次更新（SB3 早停阈值 1.5×） |
| `ent_coef` | 0.0 | 熵奖励系数，0 表示纯策略梯度 |
| `n_steps` | 2048 | 每个 env 每次 rollout 的步数 |
| `net_arch` | [256, 256] | 策略/价值网络隐藏层 |
| `gamma` | 0.99（SB3 默认） | 折扣系数 |
| `gae_lambda` | 0.95（SB3 默认） | GAE 参数 |

**快照回调（SnapshotCallback）**

每次 `ep_rew_mean` 创历史新高（提升 > 5）时触发：
- 保存命名快照 `step_NM_rew_R.zip`
- 复制为 `best.zip`
- 追加记录到 `manifest.json`

**指标跟踪（MetricsTracker）**

收集每个 rollout 的 `ep_rew_mean` 和 `ep_len_mean`，生成 summary：
- `reward_final`：最后一次 rollout 的奖励
- `reward_best`：整个 stage 的最高奖励
- `reward_trend`：stage 末 20% 均值 - 首 20% 均值（正数说明在涨）
- `ep_len_final`：最后一次 rollout 的存活步数

---

### progressive.py — 渐进训练控制器

**阶段（Stages）**

默认阶段（累计步数）：`1M → 5M → 10M → 50M → 100M`

每个 stage 的实际训练步数 = 本 stage 累计目标 - 上个 stage 累计目标。
例如从 1M 到 5M，实际训练 4M 步。

**健康检查（每个 stage 结束后自动执行）**

| 规则 | 阈值 | 含义 |
|---|---|---|
| `min_reward_final` | > 0 | 末期奖励必须为正 |
| `min_ep_len` | > 25 步 | 机器狗不能太早摔倒 |
| `max_trend_drop` | > -30 | stage 内奖励不能持续下滑 |
| `max_cross_stage_drop` | > -20 | 相比上个 stage 不能大幅退步 |

不健康 → 自动停止，等待人工调整后 `--resume`。

**状态持久化（progressive_state.json）**

记录：已完成 stage 数、总步数、各 stage 指标和健康结论、最优模型路径。
只要这个文件存在，`--resume` 就能从上次停的地方继续，无需任何其他操作。

---

## 训练历史记录

| 日期 | 关键事件 | 结论 |
|---|---|---|
| 2025-02-18 | 第一次测试训练 (`bittle_ppo_test.zip`) | 机器狗稳定但几乎不前进（reward≈24, steps=250） |
| 2025-02-19 早 | 50M 步训练 (`bittle_ppo_50m.zip`) | std=1.37，实为坏模型，不可用于续训 |
| 2025-02-19 下午 | 从零新训练 18M 步被强制终止 | reward=-96，ep_len=47。根因：target_kl=0.03 太紧 + 惩罚太重，KL 每次都在 step 0 触发，策略无法更新 |
| 2025-02-19 晚 | 修复参数，工程化快照和渐进训练 | target_kl→0.05，PENALTY_STEPS→15M，惩罚系数降低，reset()不再重连 PyBullet |

---

## 训练诊断指南

### 训练健康的标志

```
ep_rew_mean:  正值且趋势向上
ep_len_mean:  在增长（机器狗活得越来越久）
std:          在缓慢下降（动作越来越确定）
approx_kl:    稳定在 0.01~0.04 之间
explained_variance: > 0.5（最好 > 0.8）
Early stopping: 发生在 step 2+ 而非 step 0
```

### 训练失败的信号

| 现象 | 可能原因 | 处理方式 |
|---|---|---|
| `ep_rew_mean` 持续为负且下滑 | 惩罚压过奖励 | 降低 `FAC_ORIENTATION/HEIGHT/ARM_CONTACT`，增大 `PENALTY_STEPS` |
| 每次都 `Early stopping at step 0` | KL 太紧 | `target_kl` 从 0.03 调到 0.05~0.08 |
| `std` 不降反升（> 0.95） | 策略发散 | 检查奖励量级，降低学习率 |
| `ep_len_mean` < 30 步且不改善 | 机器狗根本没学会站立 | 从 `bittle_ppo_test.zip`（能站稳的模型）续训 |
| `explained_variance` < 0 | 价值网络完全失效 | 重启训练，检查奖励是否异常大 |

### 续训后奖励异常低

原因：续训时 `step_counter_session=0`，`penalty_factor` 从 0 开始，奖励看起来比之前高——这是正常现象，不代表模型退步。等 penalty 爬坡结束后（约 `PENALTY_STEPS / n_envs` 总步数），指标才具可比性。

---

## 已知限制（暂未修复）

1. **config/robot_config.yaml 未接入训练**：`env.py` 的奖励系数是硬编码常量，yaml 里的值不起作用。如需修改参数直接编辑 `env.py`。

2. **观测没有偏航角（yaw）速度**：机器狗无法感知自身左右偏转，长距离训练可能产生偏航。后续可在观测里加入 yaw 角速度。

3. **域随机化默认关闭**：`RANDOM_GYRO/MASS/FRICTION = 0`。仿真到真实机器人的迁移（sim-to-real）会有较大差距。后期可在 50M+ 步阶段逐步开启。

4. **惩罚系数在 stage 之间重置**：渐进训练每个 stage 新建 env，`step_counter_session` 重置为 0，惩罚重新从 0 爬坡。短 stage（如 1M 步）的惩罚实际上几乎为零。这是有意设计。

5. **地形始终是平地**：后续可在晚期 stage 引入坡面或障碍。
