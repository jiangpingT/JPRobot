#!/usr/bin/env python3
"""人形机器人速度命令环境 — 万向行走（前后左右转）。

## 设计思路

Gymnasium Humanoid-v4 只能"固定向前走"，因为它的奖励函数里有：
    forward_reward = 1.25 * vx   （鼓励尽量快地向 +X 方向前进）

要训练"万向行走"，需要做三件事：
  1. 去掉固定的 forward_reward
  2. 换成速度命令追踪奖励（vel_tracking_reward）：
       按 (vx_cmd, vy_cmd, wz_cmd) 命令，只要机器人速度贴近命令就给奖励
  3. 把速度命令拼进观测，让 agent 知道"该往哪走"

## 观测空间

原始 Humanoid-v4 观测：376 维（关节角、关节速度、躯干姿态等）
+ 速度命令：3 维 [vx_cmd, vy_cmd, wz_cmd]
= 379 维

## 速度追踪奖励公式

vel_error = (vx - vx_cmd)² + (vy - vy_cmd)² + 0.5 * (wz - wz_cmd)²
vel_tracking_reward = W_VEL * exp(-vel_error / SIGMA)

  - 当机器人速度完全匹配命令时，vel_error=0 → 奖励最大=W_VEL=3.0
  - 用指数衰减（而非线性惩罚）是因为：指数在误差小时梯度大（精细调整有收益），
    误差大时梯度小（崩了也不会无限惩罚，训练稳定）
  - SIGMA=0.25：误差=0.5 时奖励已降至约 37%，激励机器人追求精度

## 速度命令范围

vx_cmd ∈ [-1.0, 2.0] m/s   （负=后退，正=前进，最快 2m/s）
vy_cmd ∈ [-0.5, 0.5] m/s   （横移，人形侧移较慢）
wz_cmd ∈ [-1.0, 1.0] rad/s （绕 Z 轴转弯角速度）

每局 reset 时随机采样一次命令，整局保持不变（命令切换留给后续课程）。
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class HumanoidVelocityEnv(gym.Wrapper):
    """速度命令跟随的人形机器人环境。

    包裹 Gymnasium Humanoid-v4，把速度命令拼进观测，
    用速度追踪奖励替换原版的固定前进奖励。
    """

    # ── 速度命令范围 ───────────────────────────────────────────────────────
    VX_RANGE = (-1.0, 1.2)   # m/s，前进 / 后退（v3: 2.0→1.2，贴近物理上限，消灭追不上的高速命令）
    VY_RANGE = (-0.5, 0.5)   # m/s，横向移动（侧步）
    WZ_RANGE = (-1.0, 1.0)   # rad/s，原地转弯
    WZ_LEFT_BIAS = 0.5       # v5: 恢复均匀采样（v4=0.7左转偏置→副作用大，后退退步）

    # ── 奖励参数 ───────────────────────────────────────────────────────────
    W_VEL = 5.0               # 速度追踪奖励权重（v2: 3.0→5.0，让追踪比存活更有竞争力）
    SIGMA = 1.0               # v6: 恢复1.0（v5 SIGMA=0.75失败，评估vel_error退步0.2）
                              # 教训：SIGMA改变会造成训练/评估标准不一致，需同步改评估脚本
                              # 或者维持SIGMA不变，改其他手段提升精度
    WZ_ERROR_WEIGHT = 0.70   # v4: 转弯误差权重（v1-v3: 0.5 → v4: 0.70，加强转弯追踪梯度）
    W_ERROR_DELTA = 1.0      # v6: 误差变化率奖励权重（误差减小时奖励，增大时惩罚）
    HISTORY_STEPS = 3        # v7: 保留最近 N 步的速度误差历史，让 agent 看到趋势

    # Gymnasium Humanoid-v4 源码中 forward_reward_weight 默认值
    # 参考：gymnasium/envs/mujoco/humanoid_v4.py
    FORWARD_REWARD_WEIGHT = 1.25

    def __init__(self, render_mode=None):
        env = gym.make("Humanoid-v4", render_mode=render_mode)
        super().__init__(env)

        # 扩展观测空间：376 → 379
        orig_low  = env.observation_space.low
        orig_high = env.observation_space.high

        # 速度命令的边界
        cmd_low  = np.array([self.VX_RANGE[0], self.VY_RANGE[0], self.WZ_RANGE[0]], dtype=np.float32)
        cmd_high = np.array([self.VX_RANGE[1], self.VY_RANGE[1], self.WZ_RANGE[1]], dtype=np.float32)

        # v7: 误差历史的边界（误差最大约为命令范围的 2 倍，用 ±3 留余量）
        hist_low  = np.full(self.HISTORY_STEPS * 3, -3.0, dtype=np.float32)
        hist_high = np.full(self.HISTORY_STEPS * 3,  3.0, dtype=np.float32)

        # 扩展观测空间：376 + 3(速度命令) + 9(误差历史 3步×3分量) = 388 维
        self.observation_space = spaces.Box(
            low=np.concatenate([orig_low, cmd_low, hist_low]),
            high=np.concatenate([orig_high, cmd_high, hist_high]),
            dtype=np.float32,
        )

        # 速度命令（每局随机采样一次）
        self.cmd_vx: float = 0.0
        self.cmd_vy: float = 0.0
        self.cmd_wz: float = 0.0
        # v6: 上一步的速度误差（用于计算误差变化率）
        self._prev_vel_error: float = 0.0
        # v7: 误差历史缓冲区（shape=[HISTORY_STEPS, 3]，每行=[vx_err, vy_err, wz_err]）
        self._error_history = np.zeros((self.HISTORY_STEPS, 3), dtype=np.float32)

    # ── 内部工具 ──────────────────────────────────────────────────────────

    def _sample_cmd(self) -> None:
        """在每局 reset 时随机采样速度命令。

        v4 左转偏置：WZ_LEFT_BIAS=0.7 概率采样 wz∈(0,1]（左转），
        其余采样 wz∈[-1,0)（右转/直行），增加左转训练样本。
        """
        self.cmd_vx = float(self.np_random.uniform(*self.VX_RANGE))
        self.cmd_vy = float(self.np_random.uniform(*self.VY_RANGE))
        if self.np_random.random() < self.WZ_LEFT_BIAS:
            # 左转：wz 正值
            self.cmd_wz = float(self.np_random.uniform(0.0, self.WZ_RANGE[1]))
        else:
            # 右转 / 直行：wz 非正值
            self.cmd_wz = float(self.np_random.uniform(self.WZ_RANGE[0], 0.0))

    def _aug_obs(self, obs: np.ndarray) -> np.ndarray:
        """把速度命令和误差历史拼接到原始观测末尾，返回 388 维 float32。

        结构：[376 原始观测] + [3 速度命令] + [9 误差历史（3步×3分量）]
        误差历史：最旧的一行在前，最新的一行在末尾（时间序，便于网络学习趋势）
        """
        cmd = np.array([self.cmd_vx, self.cmd_vy, self.cmd_wz], dtype=np.float32)
        return np.concatenate([obs.astype(np.float32), cmd, self._error_history.flatten()])

    def _get_robot_vel(self):
        """从 MuJoCo 状态中读取机器人根节点的线速度和角速度。

        qvel 的含义（MuJoCo free joint，6DoF 根节点）：
          [0]: vx — 躯干在全局 X 方向的线速度（前进速度）
          [1]: vy — 躯干在全局 Y 方向的线速度（侧移速度）
          [2]: vz — 躯干在 Z 方向的线速度（上下跳）
          [3]: wx，[4]: wy，[5]: wz — 绕 X/Y/Z 轴的角速度（wz=转弯）

        unwrapped 是 Gymnasium 的属性，穿透所有 Wrapper 拿到原始 MuJoCo env，
        .data 是 mujoco.MjData 对象，存着仿真的实时状态。
        """
        qvel = self.env.unwrapped.data.qvel
        vx = float(qvel[0])
        vy = float(qvel[1])
        wz = float(qvel[5])
        return vx, vy, wz

    # ── Gym 接口 ─────────────────────────────────────────────────────────

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._sample_cmd()
        self._prev_vel_error = 0.0  # v6: 每局开始时清零上步误差
        self._error_history[:] = 0.0  # v7: 每局开始时清空误差历史
        info["vel_cmd"] = (self.cmd_vx, self.cmd_vy, self.cmd_wz)
        return self._aug_obs(obs), info

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)

        vx, vy, wz = self._get_robot_vel()

        # ── 去掉原版固定前进奖励 ──────────────────────────────────────────
        # Gymnasium Humanoid-v4 的 forward_reward = 1.25 * vx
        # 这个奖励只鼓励向 +X 前进，和"万向行走"目标冲突，必须移除。
        original_forward = self.FORWARD_REWARD_WEIGHT * vx

        # ── 速度追踪奖励 ──────────────────────────────────────────────────
        # vel_error：实际速度与命令之差（wz 权重 0.5，转弯比线速度容错更大）
        vel_error = (
            (vx - self.cmd_vx) ** 2
            + (vy - self.cmd_vy) ** 2
            + self.WZ_ERROR_WEIGHT * (wz - self.cmd_wz) ** 2  # v4: 0.5→0.70
        )
        # 指数衰减：误差=0 时奖励最大=3.0，误差越大奖励越接近 0
        vel_reward = self.W_VEL * np.exp(-vel_error / self.SIGMA)

        # ── 误差变化率奖励（v6 新增）─────────────────────────────────────────
        # error_delta > 0 表示误差在增大（变差），< 0 表示误差在减小（变好）
        # 乘以 -1 后：误差减小 → 正奖励，误差增大 → 负惩罚
        # 物理含义：鼓励策略"持续向正确方向修正"，而不只是"当前误差有多大"
        error_delta = vel_error - self._prev_vel_error
        delta_reward = self.W_ERROR_DELTA * (-error_delta)
        self._prev_vel_error = vel_error

        # v7: 更新误差历史缓冲区（滚动覆盖最旧一行，写入当前步误差向量）
        err_vec = np.array([vx - self.cmd_vx, vy - self.cmd_vy, wz - self.cmd_wz], dtype=np.float32)
        self._error_history = np.roll(self._error_history, -1, axis=0)
        self._error_history[-1] = err_vec

        # 最终奖励：保留原版的存活奖励、健康奖励、能量惩罚等，只换掉前进部分
        reward = base_reward - original_forward + vel_reward + delta_reward

        info["vel_cmd"]    = (self.cmd_vx, self.cmd_vy, self.cmd_wz)
        info["vel_actual"] = (vx, vy, wz)
        info["vel_error"]  = vel_error
        info["vel_reward"] = vel_reward
        info["delta_reward"] = delta_reward

        return self._aug_obs(obs), reward, terminated, truncated, info
