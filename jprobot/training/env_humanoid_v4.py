"""人形机器人 Gym 环境 V4 — 扩展观测空间，突破 ep_len 瓶颈。

V3 问题（ep_len 卡在 ~140 步）：
    - V3 奖励设计正确（高度维持+交替步态），但 ep_len 每百万步仅增 ~13 步
    - 根因：47 维观测空间缺少关键感知信息，agent 无法感知身体动态
    - 具体缺失：身体帧速度（cvel）+ 地面反力（cfrc_ext）

V4 核心改动：
    1. 扩展观测到 203 维（47 → +78 cvel + 78 cfrc_ext = 203）
       - cvel（身体帧速度）：每个身体段在自身局部坐标下的6DOF速度
         物理直觉：知道"我的腰正在向左倾斜多快"，才能主动抵抗
       - cfrc_ext（外部接触力）：每个身体段受到的外力（主要是地面反作用力）
         物理直觉：知道"脚踩地推力有多大"，才能学会踩地蹬起
    2. 保留 V3 的全部奖励设计（高度维持 + 交替步态）
    3. 从 SCRATCH 训练（观测空间变了，无法迁移权重）

为什么 cvel 和 cfrc_ext 是最重要的？
    - 没有 cvel：agent 只能看到关节角度，不知道哪个方向在倒
    - 没有 cfrc_ext：agent 不知道脚踩地是否产生了有效的上推力
    - Gymnasium Humanoid-v4 用 376 维观测，其中这两项占 78+78=156 维

对比 Gymnasium Humanoid-v4：
    - Gymnasium: qpos(22)+qvel(23)+cinert(130)+cvel(78)+qfrc_act(17)+cfrc_ext(84)=354
    - V4: qpos(22)+qvel(23)+foot_contacts(2)+cvel(78)+cfrc_ext(78)=203
    - 我们去掉了复杂的 cinert（质量惯性矩阵），只加最关键的两个
"""

import numpy as np

from jprobot.training.env_humanoid import (
    HumanoidEnv,
    FRAME_SKIP,
    _FOOT_BODY_NAMES,
    HEALTHY_Z_MIN,
)
import mujoco

# 初始站立高度
_Z_STAND = 1.4

# 复用 V3 奖励权重（奖励设计没问题，只是 obs 不够）
V4_W_HEALTHY     = 5.0
V4_W_FORWARD     = 2.5
V4_W_CTRL        = 0.05
V4_W_HEIGHT      = 3.0
V4_W_ALT_GAIT    = 2.0

# 身体数量（humanoid.xml 共 14 个 body，跳过 root body）
_N_BODY_SKIP_ROOT = 13  # 14 - 1


class HumanoidEnvV4(HumanoidEnv):
    """V4 人形环境：扩展观测空间（203 维），突破平衡感知瓶颈。

    观测空间拆解（共 203 维）：
        qpos[2:]       22 维  关节角度+高度+姿态四元数（去掉 x/y 全局坐标）
        qvel           23 维  关节速度+线速度+角速度
        foot_contacts   2 维  左右脚二进制接触信号
        ─────────────── 以上为 V1-V3 原有 47 维 ────────────────
        cvel[1:]       78 维  每个身体段的 6DOF 身体帧速度（[v, ω]×13 bodies）
                              → 关键：感知哪个方向在倒，倒得多快
        cfrc_ext[1:]   78 维  每个身体段受到的外部力（[力, 力矩]×13 bodies）
                              → 关键：感知脚踩地的反作用力，学会蹬地
    """

    def __init__(self, render_mode=None):
        # 先调用父类初始化（obs_dim=47）
        super().__init__(render_mode=render_mode)

        # 计算新 obs 维度：47 + cvel(78) + cfrc_ext(78) = 203
        _base_obs_dim = (self.model.nq - 2) + self.model.nv + len(_FOOT_BODY_NAMES)
        _cvel_dim = _N_BODY_SKIP_ROOT * 6    # 13×6 = 78
        _cfrc_dim = _N_BODY_SKIP_ROOT * 6    # 13×6 = 78
        _new_obs_dim = _base_obs_dim + _cvel_dim + _cfrc_dim  # 47+78+78 = 203

        import gymnasium as gym
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (_new_obs_dim,), np.float32
        )

    def _get_obs(self) -> np.ndarray:
        """扩展观测：47 维基础 + 78 维 cvel + 78 维 cfrc_ext = 203 维。"""
        # 基础 obs（与 V1-V3 相同）
        qpos = self.data.qpos[2:].copy()     # 22 维
        qvel = self.data.qvel.copy()          # 23 维

        # 脚部接触（二进制）
        foot_contacts = self._get_foot_contacts().astype(np.float32)  # 2 维

        # 新增 1: 身体帧速度（每个 body 在自身局部坐标下的 [vx,vy,vz,ωx,ωy,ωz]）
        # shape: (14, 6)，跳过 root body → (13, 6) → flatten → 78 维
        cvel = self.data.cvel[1:].flatten().copy()  # 78 维

        # 新增 2: 外部接触力（地面反作用力等）
        # shape: (14, 6)，跳过 root body → (13, 6) → flatten → 78 维
        cfrc_ext = self.data.cfrc_ext[1:].flatten().copy()  # 78 维

        return np.concatenate([qpos, qvel, foot_contacts, cvel, cfrc_ext]).astype(np.float32)

    def _compute_reward(self, action: np.ndarray, x_pos_before: float) -> float:
        """复用 V3 奖励（5 项）：高度维持 + 交替步态。"""
        dt = self.model.opt.timestep * FRAME_SKIP

        # 1. 存活奖励
        healthy_reward = V4_W_HEALTHY if self._is_healthy() else 0.0

        # 2. 前进速度奖励
        x_velocity = (float(self.data.qpos[0]) - x_pos_before) / dt
        forward_reward = V4_W_FORWARD * x_velocity

        # 3. 控制成本
        ctrl_cost = V4_W_CTRL * float(np.sum(action ** 2))

        # 4. 高度维持奖励
        z = float(self.data.qpos[2])
        z_range = _Z_STAND - HEALTHY_Z_MIN  # 0.4
        height_fraction = max(0.0, (z - HEALTHY_Z_MIN) / z_range)
        height_reward = V4_W_HEIGHT * height_fraction

        # 5. 交替步态奖励
        foot_contacts = self._get_foot_contacts()
        alternating = abs(float(foot_contacts[0]) - float(foot_contacts[1]))
        alt_gait_reward = V4_W_ALT_GAIT * alternating * max(0.0, x_velocity)

        return (healthy_reward + forward_reward + height_reward
                + alt_gait_reward - ctrl_cost)
