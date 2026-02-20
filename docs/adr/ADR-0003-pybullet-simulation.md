# ADR-0003: 使用 PyBullet 作为物理仿真引擎

**状态**：已采纳
**日期**：2026-02-19
**作者**：JPRobot Team

---

## 背景

强化学习需要大量仿真交互（数百万步）才能收敛，直接在真机上训练：
1. 速度极慢（真机实时运行，无法加速）
2. 存在安全风险（探索阶段动作随机，可能损坏硬件）
3. 成本高（磨损、维护）

需要选择一个物理仿真引擎来替代真机训练环境。

候选方案：
1. **PyBullet**：开源，轻量，Python 原生 API
2. **MuJoCo**：高精度，OpenAI 标准，但商业授权
3. **Isaac Gym / IsaacSim**：GPU 加速，但需要 NVIDIA GPU
4. **Gazebo**：ROS 生态，较重量级

## 决策

选用 **PyBullet** 作为物理仿真引擎，通过 URDF 文件加载 BittleX 机器人模型。

## 理由

| 维度 | PyBullet | MuJoCo | Isaac Gym |
|------|---------|--------|-----------|
| 许可证 | BSD（免费）| 免费（2022年后）| 免费（需NVIDIA）|
| 安装复杂度 | pip install | 中等 | 高 |
| Python API | 原生 | 原生 | 原生 |
| CPU 并行 | 支持 | 支持 | 仅 GPU |
| URDF 支持 | 完整 | 部分 | 完整 |
| macOS 支持 | 是 | 是 | 否 |

选择 PyBullet 的核心原因：
- **零门槛安装**：`pip install pybullet`，无需额外依赖
- **URDF 原生支持**：BittleX 有现成 URDF 模型（`jprobot/models/bittle_esp32.urdf`）
- **macOS 兼容**：开发者在 macOS 上工作，Isaac Gym 不支持
- **Gymnasium 集成**：与 `gymnasium.Env` 接口无缝配合

仿真环境实现在 `jprobot/training/env.py`，核心接口：
```python
class BittleGymEnv(gymnasium.Env):
    def __init__(self, render_mode=None, ...):
        self.physics_client = pybullet.connect(pybullet.DIRECT)  # 无头模式
        self.robot_id = pybullet.loadURDF("bittle_esp32.urdf")

    def step(self, action):
        pybullet.setJointMotorControl2(...)  # 驱动关节
        pybullet.stepSimulation()
        obs = self._get_obs()
        reward = self._calculate_reward()
        return obs, reward, terminated, truncated, info
```

## 后果

**正面影响**：
- macOS 上可直接运行，无需专用 GPU 硬件
- 仿真速度比真机快约 10-20 倍（CPU 模式）
- URDF 模型与真机参数对齐，物理特性接近真实

**负面影响**：
- 仿真-真实差距（Sim-to-Real Gap）：PyBullet 的摩擦力、关节阻尼模型与真机存在偏差
- 无 GPU 加速，并发环境数受 CPU 核心数限制
- `KMP_DUPLICATE_LIB_OK=TRUE` 环境变量在 macOS 上必须设置（PyBullet 与 MKL 冲突）

**待解决**：
- 真机迁移（Sim-to-Real Transfer）策略尚未实现，需领域随机化（Domain Randomization）
