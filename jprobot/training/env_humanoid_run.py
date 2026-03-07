#!/usr/bin/env python3
"""人形机器人跑步环境 — 高速万向行走 + 腾空相奖励。

在 env_humanoid_velocity.py（万向行走 v6）基础上两处核心改动：
  1. VX_RANGE 扩展到 3.5 m/s，让 agent 学习跑步速度
  2. 新增腾空相奖励（flight phase reward）：双脚同时离地+速度达阈值才给奖励

## 跑步 vs 行走的物理区别

行走：任意时刻至少有一只脚在地面（支撑相 + 摆动相交替）
跑步：每步都有短暂的双脚腾空期（flight phase / aerial phase）

腾空相是判断机器人是否真正"跑起来"的关键特征。仅仅速度快但双脚
始终贴地（快速蹦跶）不算跑步；只有在腾空时才会产生跑步的视觉效果。

## 腾空检测方法

利用 MuJoCo 的 data.cfrc_ext：
  - cfrc_ext[body_id] = 该 body 所受外部接触力（[3]force + [3]torque）
  - 脚踩地时，接触力不为零；完全腾空时接触力近零
  - 双脚接触力 < CONTACT_THRESH → 判断为腾空

## 防 gaming 设计

腾空奖励需要同时满足：
  1. vx > VX_RUN_THRESHOLD（真正在跑，不是原地乱跳）
  2. 双脚均无接触（真正腾空，不是单脚离地的行走摆动相）
  否则给 0 奖励，不惩罚（不影响已有的行走技能）
"""

import mujoco
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class HumanoidRunEnv(gym.Wrapper):
    """跑步速度命令跟随的人形机器人环境。

    包裹 Gymnasium Humanoid-v4，在 HumanoidVelocityEnv（v6）基础上
    新增腾空相奖励，训练机器人以 2~3.5 m/s 的跑步速度移动。
    """

    # ── 速度命令范围（跑步版）──────────────────────────────────────────────
    VX_RANGE = (-0.5, 2.0)   # m/s，v2: 3.5→2.0（贴近 MuJoCo Humanoid-v4 物理上限 ~1.5m/s）
    VY_RANGE = (-0.5, 0.5)   # m/s，横向移动
    WZ_RANGE = (-1.0, 1.0)   # rad/s，转弯
    WZ_LEFT_BIAS = 0.5       # 均匀采样（不偏置）

    # ── 速度追踪奖励参数（继承自 v6）──────────────────────────────────────
    W_VEL = 5.0              # 速度追踪奖励权重
    SIGMA = 1.0              # 奖励宽容度（误差=0时满分，=1时约37%）
    WZ_ERROR_WEIGHT = 0.70   # 转弯误差权重
    W_ERROR_DELTA = 1.0      # 误差变化率奖励权重（Method B，v6验证有效）
    FORWARD_REWARD_WEIGHT = 1.25  # Humanoid-v4 原版前进奖励系数（用于移除）

    # ── 腾空相奖励参数（跑步新增）──────────────────────────────────────────
    W_AIRBORNE = 2.0         # 腾空相奖励权重（每步双脚腾空时给分）
    VX_RUN_THRESHOLD = 0.5  # m/s，v2: 1.5→0.5（降低至物理可达范围，让腾空奖励真正触发）
    CONTACT_THRESH = 1.0     # N，接触力低于此值判断为腾空（单位：牛顿）

    # ── 直立姿态奖励参数（v3 新增，解决"超人扑倒式滑行"gaming）────────────
    # 问题：机器人发现横躺滑行比直立跑步更容易拿速度追踪分
    # 解法：奖励躯干保持站立高度，强迫机器人维持直立姿态
    # torso_z（躯干 z 坐标）物理含义：
    #   - 完全直立站立：约 1.25m
    #   - 严重前倾（45°）：约 1.0-1.1m（接近健康下限）
    #   - 横躺（健康边界）：约 1.0m（低于此值 episode 终止）
    # 奖励公式：(torso_z - 1.0) / 0.3，在 [0, W_UPRIGHT] 间线性插值
    W_UPRIGHT = 6.0          # 直立奖励权重（v4: 3.0→6.0，超过速度追踪W_VEL=5.0，强迫直立优先）
    UPRIGHT_Z_MIN = 1.0      # 最低健康高度（Humanoid-v4 的 healthy_z_range 下限）
    UPRIGHT_Z_TARGET = 1.3   # 目标站立高度（完全直立约 1.25-1.35m）

    def __init__(self, render_mode=None):
        env = gym.make("Humanoid-v4", render_mode=render_mode)
        super().__init__(env)

        # 扩展观测空间：376 + 3(速度命令) = 379 维（与 v6 相同，可热启）
        orig_low  = env.observation_space.low
        orig_high = env.observation_space.high
        cmd_low  = np.array([self.VX_RANGE[0], self.VY_RANGE[0], self.WZ_RANGE[0]], dtype=np.float32)
        cmd_high = np.array([self.VX_RANGE[1], self.VY_RANGE[1], self.WZ_RANGE[1]], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.concatenate([orig_low, cmd_low]),
            high=np.concatenate([orig_high, cmd_high]),
            dtype=np.float32,
        )

        # 速度命令
        self.cmd_vx: float = 0.0
        self.cmd_vy: float = 0.0
        self.cmd_wz: float = 0.0
        # Method B：上一步误差（用于误差变化率奖励）
        self._prev_vel_error: float = 0.0
        # 腾空检测：缓存脚的 body ID（首次 step 时懒初始化）
        self._left_foot_id: int | None = None
        self._right_foot_id: int | None = None

    # ── 内部工具 ──────────────────────────────────────────────────────────

    def _sample_cmd(self) -> None:
        """随机采样速度命令（跑步版：70% 概率采样 vx > 1.5 m/s 的高速命令）。

        高速命令采样偏置：确保有足够的跑步训练样本，否则模型会以慢速行走
        完成大多数命令，永远触发不到腾空奖励的梯度。
        """
        # 70% 概率：采样跑步速度（vx > VX_RUN_THRESHOLD）
        if self.np_random.random() < 0.7:
            self.cmd_vx = float(self.np_random.uniform(self.VX_RUN_THRESHOLD, self.VX_RANGE[1]))
        else:
            # 30% 概率：采样全范围（保留慢速和后退的训练覆盖）
            self.cmd_vx = float(self.np_random.uniform(self.VX_RANGE[0], self.VX_RANGE[1]))
        self.cmd_vy = float(self.np_random.uniform(*self.VY_RANGE))
        if self.np_random.random() < self.WZ_LEFT_BIAS:
            self.cmd_wz = float(self.np_random.uniform(0.0, self.WZ_RANGE[1]))
        else:
            self.cmd_wz = float(self.np_random.uniform(self.WZ_RANGE[0], 0.0))

    def _aug_obs(self, obs: np.ndarray) -> np.ndarray:
        """把速度命令拼接到原始观测末尾，返回 379 维 float32。"""
        cmd = np.array([self.cmd_vx, self.cmd_vy, self.cmd_wz], dtype=np.float32)
        return np.concatenate([obs.astype(np.float32), cmd])

    def _get_robot_vel(self):
        """读取躯干速度：vx（前进），vy（侧移），wz（转弯角速度）。"""
        qvel = self.env.unwrapped.data.qvel
        return float(qvel[0]), float(qvel[1]), float(qvel[5])

    def _init_foot_ids(self):
        """首次调用时懒初始化脚的 body ID（避免 __init__ 时模型未就绪）。"""
        model = self.env.unwrapped.model
        self._left_foot_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_foot")
        self._right_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_foot")

    def _both_feet_airborne(self) -> bool:
        """检测双脚是否同时腾空（均无地面接触力）。

        data.cfrc_ext[body_id] = 该 body 所受外部接触力向量（世界坐标系）。
          - 脚踩地时：法向接触力产生显著数值
          - 完全腾空时：接触力趋近于零
        取力向量的 L2 范数，与 CONTACT_THRESH 比较。
        """
        if self._left_foot_id is None:
            self._init_foot_ids()
        data = self.env.unwrapped.data
        lf_force = float(np.linalg.norm(data.cfrc_ext[self._left_foot_id, :3]))
        rf_force = float(np.linalg.norm(data.cfrc_ext[self._right_foot_id, :3]))
        return lf_force < self.CONTACT_THRESH and rf_force < self.CONTACT_THRESH

    # ── Gym 接口 ──────────────────────────────────────────────────────────

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._sample_cmd()
        self._prev_vel_error = 0.0
        info["vel_cmd"] = (self.cmd_vx, self.cmd_vy, self.cmd_wz)
        return self._aug_obs(obs), info

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)

        vx, vy, wz = self._get_robot_vel()

        # ── 去掉原版固定前进奖励 ──────────────────────────────────────────
        original_forward = self.FORWARD_REWARD_WEIGHT * vx

        # ── 速度追踪奖励 ──────────────────────────────────────────────────
        vel_error = (
            (vx - self.cmd_vx) ** 2
            + (vy - self.cmd_vy) ** 2
            + self.WZ_ERROR_WEIGHT * (wz - self.cmd_wz) ** 2
        )
        vel_reward = self.W_VEL * np.exp(-vel_error / self.SIGMA)

        # ── 误差变化率奖励（Method B，v6 验证有效）──────────────────────────
        error_delta  = vel_error - self._prev_vel_error
        delta_reward = self.W_ERROR_DELTA * (-error_delta)
        self._prev_vel_error = vel_error

        # ── 腾空相奖励（跑步新增）────────────────────────────────────────
        airborne = self._both_feet_airborne()
        if vx > self.VX_RUN_THRESHOLD and airborne:
            airborne_reward = self.W_AIRBORNE
        else:
            airborne_reward = 0.0

        # ── 直立姿态奖励（v3 新增）────────────────────────────────────────
        # 读取躯干 z 坐标（MuJoCo free joint 的 qpos[2] = 根节点世界坐标 z）
        # 线性插值：高度在 [UPRIGHT_Z_MIN, UPRIGHT_Z_TARGET] 间给 [0, W_UPRIGHT] 分
        torso_z = float(self.env.unwrapped.data.qpos[2])
        upright_ratio = min(1.0, max(0.0,
            (torso_z - self.UPRIGHT_Z_MIN) / (self.UPRIGHT_Z_TARGET - self.UPRIGHT_Z_MIN)
        ))
        upright_reward = self.W_UPRIGHT * upright_ratio

        # ── 最终奖励 ──────────────────────────────────────────────────────
        reward = base_reward - original_forward + vel_reward + delta_reward + airborne_reward + upright_reward

        info["vel_cmd"]       = (self.cmd_vx, self.cmd_vy, self.cmd_wz)
        info["vel_actual"]    = (vx, vy, wz)
        info["vel_error"]     = vel_error
        info["vel_reward"]    = vel_reward
        info["delta_reward"]  = delta_reward
        info["airborne"]        = airborne
        info["airborne_reward"] = airborne_reward
        info["torso_z"]         = torso_z
        info["upright_reward"]  = upright_reward

        return self._aug_obs(obs), reward, terminated, truncated, info
