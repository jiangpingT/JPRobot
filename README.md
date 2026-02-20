# JPRobot

> BittleX 四足机器狗智能控制与强化学习训练框架

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-Private-red)](.)

---

## 简介

JPRobot 是一个面向 **Petoi BittleX** 四足机器狗的完整工程框架，集成三大能力：

- **LLM 自然语言控制**：用中文直接指挥机器狗做动作
- **强化学习训练**：在 PyBullet 仿真中训练步态策略，逐步迁移到真机
- **渐进式训练管理**：自动分阶段训练、健康检查、断点续训

---

## 快速开始

### 环境准备

```bash
# 克隆项目
git clone https://github.com/jiangpingT/JPRobot.git
cd JPRobot

# 配置环境变量
cp .env.example .env
# 编辑 .env 填入 API Key

# 安装依赖（推荐 conda 环境）
pip install -r requirements.txt
```

> macOS 上运行训练需加环境变量：`KMP_DUPLICATE_LIB_OK=TRUE`

### 自然语言控制机器狗

```bash
# 连接真机（自动检测串口）
python scripts/demo_agent.py

# 仿真模式（无需硬件）
python scripts/demo_agent.py --sim
```

### 强化学习训练

```bash
# 从头开始全自动训练（推荐）
./scripts/train.sh --auto

# 从断点续训
./scripts/train.sh --resume --auto

# 查看实时训练 Dashboard
open http://127.0.0.1:18791/dashboard
```

### 可视化已训练模型

```bash
KMP_DUPLICATE_LIB_OK=TRUE python -m jprobot.training.enjoy \
  --model trained/snapshots/best.zip --episodes 3
```

---

## 项目结构

```
JPRobot/
├── jprobot/                    # 核心包
│   ├── agent/                  # LLM 智能控制层
│   │   ├── brain.py            # RobotBrain：自然语言 → 机器人指令
│   │   └── tools.py            # Function Calling 工具定义
│   ├── robot/                  # 硬件通信层
│   │   ├── connection.py       # 串口连接管理
│   │   ├── commander.py        # BittleCommander：高级控制 API
│   │   └── skills.py           # 56 个预设动作库
│   ├── training/               # 强化学习训练层
│   │   ├── env.py              # BittleGymEnv：PyBullet 仿真环境
│   │   ├── train.py            # PPO 训练 + 快照系统
│   │   ├── progressive.py      # 渐进训练控制器
│   │   └── enjoy.py            # 模型可视化
│   └── models/
│       └── bittle_esp32.urdf   # BittleX 机器人模型
├── scripts/
│   ├── train.sh                # 一键训练脚本（含 Dashboard 管理）
│   ├── training_server.py      # 训练 Dashboard 后端
│   ├── demo_agent.py           # 自然语言交互 Demo
│   ├── demo_simulation.py      # PyBullet 仿真 Demo
│   └── demo_control.py         # 直接硬件控制 Demo
├── config/
│   └── robot_config.yaml       # 机器人与训练配置
├── trained/                    # 训练输出（不提交 .zip）
│   ├── snapshots/              # 命名快照 + best.zip
│   ├── checkpoints/            # 周期性检查点（每 2M 步）
│   └── progressive_state.json  # 渐进训练进度（用于续训）
├── docs/
│   └── adr/                    # 架构决策记录
├── TRAINING.md                 # 训练完整手册
└── .env.example                # 环境变量模板
```

---

## 架构概览

```
用户自然语言输入
       │
       ▼
  RobotBrain (LLM)
  ├── Function Calling 模式（Claude / GPT-4o）
  └── 提示解析模式（其他模型）
       │
       ▼
  BittleCommander
  ├── perform(skill_code)   → SkillRegistry（56 个预设动作）
  ├── move_joint()          → 单关节控制
  └── execute_sequence()    → 动作序列编排
       │
       ▼
  SerialConnection（115200 bps）
  └── BittleX 真机 / PyBullet 仿真
```

---

## 强化学习训练

训练采用 **PPO（近端策略优化）**，在 PyBullet 物理仿真中逐步学习四足步态。

### 渐进训练阶段

| 阶段 | 累计步数 | 说明 |
|------|---------|------|
| 1    | 1M      | 学会基本站立和迈步 |
| 2    | 5M      | 步态趋于稳定 |
| 3    | 10M     | 前进奖励开始显现 |
| 4    | 50M     | 步态精细化，惩罚逐步加强 |
| 5    | 100M    | 最终目标 |

每阶段结束后自动健康检查，训练异常时自动停止并支持 `--resume` 续训。详见 [TRAINING.md](TRAINING.md)。

### 观测空间（246 维）

| 维度 | 内容 |
|------|------|
| 0–3  | 机体姿态四元数 |
| 4–5  | 角速度 roll/pitch |
| 6–245 | 关节角历史（30帧 × 8关节） |

### 关键超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `FAC_MOVEMENT` | 1000.0 | 前进奖励放大系数 |
| `FAC_SURVIVAL` | 2.0 | 每步存活奖励 |
| `PENALTY_STEPS` | 100M | 惩罚从 0 到满值的步数 |
| `target_kl` | 0.05 | KL 散度上限 |

---

## 文档索引

| 文档 | 内容 |
|------|------|
| [TRAINING.md](TRAINING.md) | 训练命令、参数说明、诊断指南、历史记录 |
| [docs/adr/](docs/adr/) | 架构决策记录（ADR） |

---

## 依赖

- Python 3.12
- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) — PPO 实现
- [PyBullet](https://pybullet.org) — 物理仿真
- [Gymnasium](https://gymnasium.farama.org) — RL 环境接口
- [OpenAI SDK](https://github.com/openai/openai-python) — LLM 调用
- PySerial — 串口通信
