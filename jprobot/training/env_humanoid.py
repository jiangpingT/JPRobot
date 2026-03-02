"""MuJoCo 人形机器人 Gymnasium 环境。

对标 Gymnasium Humanoid-v4，直接调用 mujoco Python 包（不通过 Gymnasium 中间层），
设计风格与 env.py（PyBullet 四足）保持一致。

观测空间（47 维，精简版）：
    qpos[2:]        — 20 维（去掉 x/y，保留高度 + 四元数[4] + 17 关节角）
    qvel            — 23 维（线速度 3 + 角速度 3 + 17 关节速度）
    foot_contacts   — 4 维（左右脚 × 2 个触地传感器，二进制）

动作空间（17 维，连续 [-0.4, 0.4]）：
    直接映射 MuJoCo 的 17 个马达力矩（与 Humanoid-v4 相同量程）

奖励（4 项）：
    healthy_reward  — 保持直立，+5.0/步
    forward_reward  — 质心 x 方向速度 × 1.25
    ctrl_cost       — 控制成本（大力矩惩罚），-0.1 × sum(action²)
    lateral_penalty — 侧偏惩罚（偏离 x 轴），-0.5 × |y_pos|

终止条件：
    躯干高度 < 1.0m 或 > 2.0m → terminated（摔倒或弹飞）
    1000 步 → truncated（时间到）
"""

import os
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

# 模型 XML 路径
_MODEL_PATH = str(Path(__file__).parent.parent / "models" / "humanoid.xml")

# 超参数
EPISODE_LENGTH = 1000          # 最大步数
FRAME_SKIP = 5                 # 每次 step 执行的物理子步数（同 Humanoid-v4）
HEALTHY_Z_MIN = 1.0            # 躯干最低高度（低于此即摔倒）
HEALTHY_Z_MAX = 2.0            # 躯干最高高度（超过此即弹飞）

# 奖励权重
W_HEALTHY = 5.0                # 存活奖励（每步）
W_FORWARD = 1.25               # 前进速度奖励系数
W_CTRL = 0.1                   # 控制成本惩罚系数
W_LATERAL = 0.5                # 侧偏惩罚系数

# 脚部接触 body name（在 humanoid.xml 中对应 left_foot / right_foot）
_FOOT_BODY_NAMES = ["left_foot", "right_foot"]


class HumanoidEnv(gym.Env):
    """MuJoCo 人形机器人环境，面向 PPO 强化学习训练。

    坐标系（MuJoCo 默认 Z-up）：
        x → 前进方向
        y → 左方向
        z → 上方向

    qpos 布局（28 维）：
        [0:3]  质心位置 x, y, z
        [3:7]  质心姿态四元数 w, x, y, z
        [7:24] 17 个关节角度（弧度）

    qvel 布局（27 维）：
        [0:3]  线速度 vx, vy, vz
        [3:6]  角速度 ωx, ωy, ωz
        [6:23] 17 个关节角速度

    obs 取 qpos[2:]（去掉 x/y）+ qvel + foot_contacts = 47 维。
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        super().__init__()

        # 加载 MuJoCo 模型
        self.model = mujoco.MjModel.from_xml_path(_MODEL_PATH)
        self.data = mujoco.MjData(self.model)

        # render_mode（目前不实现渲染，占位）
        self.render_mode = render_mode

        # 动作空间：17 个马达力矩，范围 [-0.4, 0.4]
        # 为什么是 17？humanoid.xml 里有 17 个 actuator（腰部 3 + 髋部 6 + 膝部 2 + 肩部 4 + 肘部 2）
        n_actions = self.model.nu  # nu = number of actuators
        self.action_space = gym.spaces.Box(
            low=-0.4, high=0.4, shape=(n_actions,), dtype=np.float32
        )

        # 观测空间：47 维，无界（用 ±inf 声明）
        # qpos[2:] = 26 维，qvel = 23 维（注意：qvel 比 qpos 少 1，因为四元数速度用 3 维角速度代替）
        # qpos 有 28 维，去掉 x/y 后 26 维
        # qvel 有 27 维（线速度 3 + 角速度 3 + 关节速度 17 = 23）
        # 但 MuJoCo 的 qvel 实际 = nv = 27
        # foot_contacts = 4 维（2 脚 × 2 传感器，但 humanoid.xml 只有 left_foot + right_foot 两个 body，每个 1 维）
        obs_dim = (self.model.nq - 2) + self.model.nv + len(_FOOT_BODY_NAMES)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # 预查 foot body id（避免每步重复查找）
        self._foot_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in _FOOT_BODY_NAMES
        ]

        self._step_count = 0

    # ──────────────────────────────────────────────────────────
    # Gym 接口
    # ──────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # 重置 MuJoCo 状态（零初始化）
        mujoco.mj_resetData(self.model, self.data)

        # 轻微随机化初始姿态，避免策略对固定初始条件过拟合
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

        # 对 qpos 和 qvel 加小扰动（同 Humanoid-v4 默认 noise_scale=0.01）
        noise_scale = 0.01
        self.data.qpos[:] += rng.uniform(-noise_scale, noise_scale, self.model.nq)
        self.data.qvel[:] += rng.uniform(-noise_scale, noise_scale, self.model.nv)

        # 前向模拟一步以更新 contact/geom
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        """执行一步环境交互。

        物理含义：
            1. 把 action（力矩指令）写入 data.ctrl
            2. 执行 FRAME_SKIP 个物理子步（共 5 × 0.002s = 0.01s 真实时间）
            3. 计算奖励、判断是否终止

        frame_skip=5 的含义：控制频率 = 1/0.01s = 100 Hz，
        每秒做 100 次决策（与 Humanoid-v4 一致）。
        """
        # 记录动作前质心位置（用于计算速度）
        x_pos_before = self.data.qpos[0]

        # 写入控制指令（力矩）
        self.data.ctrl[:] = np.clip(action, -0.4, 0.4)

        # 执行 frame_skip 个物理步
        mujoco.mj_step(self.model, self.data, nstep=FRAME_SKIP)

        self._step_count += 1

        # 获取观测
        obs = self._get_obs()

        # 计算奖励
        reward = self._compute_reward(action, x_pos_before)

        # 终止条件
        terminated = not self._is_healthy()
        truncated = self._step_count >= EPISODE_LENGTH

        info = {
            "x_pos": float(self.data.qpos[0]),
            "z_pos": float(self.data.qpos[2]),
            "step": self._step_count,
        }

        return obs, reward, terminated, truncated, info

    def close(self):
        pass

    # ──────────────────────────────────────────────────────────
    # 内部方法
    # ──────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """构建 47 维观测向量。

        为什么去掉 qpos[0:2]（x/y）？
            x/y 是绝对位置，对策略没用（策略只关心如何动，不关心在哪）。
            去掉后 agent 自动学会平移不变的步态。
        """
        qpos = self.data.qpos[2:].copy()    # 26 维：z + 四元数(4) + 17关节角
        qvel = self.data.qvel.copy()         # 27 维：线速度(3) + 角速度(3) + 关节速度(17+4=20... actually nv=27)
        contacts = self._get_foot_contacts() # 2 维：左右脚是否接触地面
        return np.concatenate([qpos, qvel, contacts]).astype(np.float32)

    def _get_foot_contacts(self) -> np.ndarray:
        """检测左右脚是否接触地面，返回二进制向量。

        为什么需要接触信息？
            告诉 agent 哪只脚踩在地上，有助于学习正确的步态相位。
            类比四足 env 的 feet_contact_state。
        """
        contacts = np.zeros(len(_FOOT_BODY_NAMES), dtype=np.float32)
        for c in range(self.data.ncon):
            contact = self.data.contact[c]
            # 检查 contact 的两个 geom 是否属于脚 body
            for j, body_id in enumerate(self._foot_body_ids):
                g1 = contact.geom[0]
                g2 = contact.geom[1]
                b1 = self.model.geom_bodyid[g1]
                b2 = self.model.geom_bodyid[g2]
                if b1 == body_id or b2 == body_id:
                    contacts[j] = 1.0
        return contacts

    def _is_healthy(self) -> bool:
        """判断机器人是否处于健康（未摔倒）状态。

        判据：躯干（root body）z 轴高度在合理范围内。
        z < 1.0m 通常意味着摔倒；z > 2.0m 意味着被弹飞（仿真不稳定）。
        """
        z = float(self.data.qpos[2])
        return HEALTHY_Z_MIN <= z <= HEALTHY_Z_MAX

    def _compute_reward(self, action: np.ndarray, x_pos_before: float) -> float:
        """计算单步奖励，共 4 项。

        1. healthy_reward（+5.0/步）
           为什么？鼓励 agent 保持直立。如果只有前进奖励，agent 可能选择倒地滚动。

        2. forward_reward（正比于 x 速度）
           为什么？核心训练目标：让机器人向 x 方向行走。
           x_velocity = Δx / (frame_skip × dt)，单位 m/s。

        3. ctrl_cost（负，正比于力矩平方和）
           为什么？惩罚大力矩，促使 agent 找到节能的步态，避免抖动。

        4. lateral_penalty（负，正比于 |y 位置|）
           为什么？阻止机器人走斜，保持沿 x 轴直线行走。
           Humanoid-v4 没有这项，我们新增它来提高行走稳定性。
        """
        # 1. 存活奖励
        healthy_reward = W_HEALTHY if self._is_healthy() else 0.0

        # 2. 前进奖励（x 方向速度）
        # MuJoCo timestep = model.opt.timestep（通常 0.002s），frame_skip=5 → 每 step 0.01s
        dt = self.model.opt.timestep * FRAME_SKIP
        x_velocity = (float(self.data.qpos[0]) - x_pos_before) / dt
        forward_reward = W_FORWARD * x_velocity

        # 3. 控制成本（大力矩惩罚）
        ctrl_cost = W_CTRL * float(np.sum(action ** 2))

        # 4. 侧偏惩罚（偏离 x 轴）
        y_pos = float(self.data.qpos[1])
        lateral_penalty = W_LATERAL * abs(y_pos)

        return healthy_reward + forward_reward - ctrl_cost - lateral_penalty
