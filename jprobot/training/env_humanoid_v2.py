"""人形机器人 Gym 环境 V2 — 改进奖励，打破 V1 plateau。

V1 问题（ep_len 在 ~110 步 plateau，约 1 秒就摔倒）：
    1. lateral_penalty 惩罚 y 坐标绝对值 → 随机探索时 y 漂移积累惩罚，
       agent 学会"原地不动"比"向前探索"更安全，抑制了步态学习
    2. W_FORWARD=1.25 太弱 → "稳定但不走" 的奖励高于 "走路但不稳"
    3. ctrl_cost=0.1 过强 → agent 不敢输出产生有效步态所需的大力矩

V2 改动（相比 V1）：
    1. 去掉 lateral_penalty（解锁探索空间）
    2. W_FORWARD: 1.25 → 2.5（更强的前进激励）
    3. ctrl_cost 系数: 0.1 → 0.05（允许更有力的步态动作）
    4. 新增 gait_reward: max(vx, 0) × 脚接触比例 × 1.0
       — 专门奖励"脚踩地同时向前走"，引导步态形成

obs/action 空间与 V1 完全相同（47维 / 17维），可从 V1 best.zip 热启动。
"""

import numpy as np

from jprobot.training.env_humanoid import (
    HumanoidEnv,
    FRAME_SKIP,
    _FOOT_BODY_NAMES,
)

# V2 奖励权重
V2_W_HEALTHY = 5.0    # 不变
V2_W_FORWARD = 2.5    # V1: 1.25 → 2.5（翻倍，更强前进激励）
V2_W_CTRL    = 0.05   # V1: 0.10 → 0.05（减半，允许更大动作）
V2_W_GAIT    = 1.0    # 新增：步态奖励系数


class HumanoidEnvV2(HumanoidEnv):
    """V2 人形环境：更强前进激励 + 步态奖励 + 无侧偏惩罚。

    与 V1 唯一区别是 _compute_reward，obs/action 空间完全一致。
    """

    def _compute_reward(self, action: np.ndarray, x_pos_before: float) -> float:
        """V2 奖励：4 项（去掉 lateral_penalty，新增 gait_reward）。

        1. healthy_reward  — 保持直立每步 +5.0（不变）
        2. forward_reward  — 前进速度 × 2.5（V1 的 2 倍）
        3. ctrl_cost       — 大力矩惩罚 × 0.05（V1 的一半）
        4. gait_reward     — max(vx, 0) × 脚接触比例 × 1.0
             物理含义：脚踩在地上往前走才给奖励，
             — vx>0：只奖励前进，不奖励倒退
             — 脚接触比例：两脚都踩地得 1.0，一脚得 0.5，都腾空得 0
             — 这个奖励直接引导 agent 发展出"有支撑的步态"
        """
        dt = self.model.opt.timestep * FRAME_SKIP

        # 1. 存活奖励
        healthy_reward = V2_W_HEALTHY if self._is_healthy() else 0.0

        # 2. 前进速度奖励（系数翻倍）
        x_velocity = (float(self.data.qpos[0]) - x_pos_before) / dt
        forward_reward = V2_W_FORWARD * x_velocity

        # 3. 控制成本（减半）
        ctrl_cost = V2_W_CTRL * float(np.sum(action ** 2))

        # 4. 步态奖励：脚踩地 × 前进速度
        foot_contacts = self._get_foot_contacts()
        foot_ratio = float(foot_contacts.sum()) / max(1, len(_FOOT_BODY_NAMES))
        gait_reward = V2_W_GAIT * max(0.0, x_velocity) * foot_ratio

        # 去掉 V1 的 lateral_penalty，解锁 y 方向探索自由度
        return healthy_reward + forward_reward - ctrl_cost + gait_reward
