# JPRobot - Claude Code 项目记忆

## 项目概述
BittleX 四足机器人强化学习训练系统，基于 opencat-gym 改进。
使用 PyBullet 物理仿真 + Stable Baselines3 PPO 训练行走策略。

## 关键架构
- `jprobot/training/env.py` — Gym 环境（核心，匹配原版 opencat-gym）
- `jprobot/training/progressive.py` — 课程学习编排器
- `jprobot/training/train.py` — PPO 训练入口（含快照/检查点回调）
- `jprobot/models/bittle_esp32.urdf` — 机器人 URDF 模型
- `scripts/training_server.py` — Web Dashboard + 3D 可视化服务器（端口 18791）

## 训练参数（已验证可产生行走）
- 2M 步 / 8 并行环境 / seed=42
- PPO: net_arch=[256,256], lr=3e-4, ent_coef=0.0
- 奖励: FAC_MOVEMENT=1000, FAC_STABILITY=0.1, FAC_ARM_CONTACT=0.01
- penalty_factor 线性增长: 0 → 1.0 over 2M steps
- 期望结果: reward ~1100-1200, 前进 ~1.2m/episode, 0% 手臂触地

## 血泪教训（10 Bug 复盘）

### 致命级（必须牢记）

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

## 运行命令

```bash
# 训练（conda activate jprobot）
python -m jprobot.training.progressive --curriculum simple --auto

# Dashboard + 3D 可视化
python scripts/training_server.py
# 浏览器打开: http://127.0.0.1:18791/dashboard 和 /viz

# 本地可视化（PyBullet GUI）
python -m jprobot.training.enjoy

# A/B 对比测试
bash scripts/run_ab_test.sh
python scripts/compare_logs.py
```

## 文件清理提醒
trained/snapshots/ 会积累大量快照文件（几百个），定期清理只保留:
- best.zip（最佳模型）
- curriculum_*.zip（阶段快照）
- 少量里程碑快照
