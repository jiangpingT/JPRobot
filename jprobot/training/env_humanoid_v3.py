"""人形机器人 Gym 环境 V3 — 高度维持 + 交替步态奖励，打破"缓慢下沉"问题。

V2 诊断（ep_len 仍卡在 ~110 步）：
    - 零动作存活 40 步，V1 策略存活 127 步——说明策略在用力，但 z 仍缓慢下沉
    - z 从初始 1.4m 以 ~0.003m/步 速度下沉到 1.0m（阈值）后触发终止
    - 根因：奖励函数没有明确惩罚"缓慢下沉"，也没有奖励"踩地反弹"

V3 改动（相比 V2）：
    1. 新增 height_reward：主动奖励维持高 z，z 越接近初始 1.4m 奖励越高
       → 直接对抗"缓慢下沉"，给 agent 明确的"向上推"信号
    2. 改进 gait_reward → alternating_gait：奖励交替单脚接触 × 前进速度
       → 激励真正的行走步态（一脚踩地，一脚摆动），而非双脚同时落地的"站立"
    3. 保留 V2 的 ctrl_cost=0.05（较小，允许有力的步态动作）
    4. 保留 V2 的 W_FORWARD=2.5
    5. 去掉 V2 的简单 gait_reward（被新 alternating_gait 取代）

奖励设计物理直觉：
    - height_reward：让机器人主动弯腿后蹬地，像人迈步时会先下蹲再蹬起
    - alternating_gait：完美的行走步态是"左脚→右脚→左脚..."循环，
      两脚同时在地 = 站立（低奖励），一脚在地一脚摆 = 行走（高奖励）
"""

import numpy as np

from jprobot.training.env_humanoid import (
    HumanoidEnv,
    FRAME_SKIP,
    _FOOT_BODY_NAMES,
    HEALTHY_Z_MIN,
)

# 初始站立高度（humanoid.xml 默认 z=1.4m）
_Z_STAND = 1.4

# V3 奖励权重
V3_W_HEALTHY     = 5.0    # 存活奖励（不变）
V3_W_FORWARD     = 2.5    # 前进速度（同 V2）
V3_W_CTRL        = 0.05   # 控制成本（同 V2，较小）
V3_W_HEIGHT      = 3.0    # 高度维持奖励（新增）
V3_W_ALT_GAIT    = 2.0    # 交替步态奖励（替换 V2 的 gait_reward）


class HumanoidEnvV3(HumanoidEnv):
    """V3 人形环境：高度维持 + 交替步态奖励，对抗"缓慢下沉"问题。"""

    def _compute_reward(self, action: np.ndarray, x_pos_before: float) -> float:
        """V3 奖励：5 项。

        1. healthy_reward  — 保持直立 +5.0/步（不变）
        2. forward_reward  — 前进速度 × 2.5（同 V2）
        3. ctrl_cost       — 大力矩惩罚 × 0.05（同 V2）
        4. height_reward   — 维持高 z 奖励（新增）
               = V3_W_HEIGHT × max(0, z - z_min) / (z_stand - z_min)
               = 在 z=z_min 时为 0，在 z=z_stand=1.4m 时为 V3_W_HEIGHT=3.0
               物理直觉：鼓励机器人维持直立身高，而不是"矮身子挣扎"
        5. alternating_gait_reward — 交替步态奖励（替换 V2 的 gait_reward）
               = V3_W_ALT_GAIT × alternating × max(0, vx)
               alternating = |left_contact - right_contact| ∈ {0, 1}
               — 只有一脚在地时为 1（标准行走步态），两脚同时在/离地为 0
               物理直觉：奖励"一脚撑地，一脚迈步"的走路动作
        """
        dt = self.model.opt.timestep * FRAME_SKIP

        # 1. 存活奖励
        healthy_reward = V3_W_HEALTHY if self._is_healthy() else 0.0

        # 2. 前进速度奖励
        x_velocity = (float(self.data.qpos[0]) - x_pos_before) / dt
        forward_reward = V3_W_FORWARD * x_velocity

        # 3. 控制成本
        ctrl_cost = V3_W_CTRL * float(np.sum(action ** 2))

        # 4. 高度维持奖励（对抗"缓慢下沉"）
        z = float(self.data.qpos[2])
        z_range = _Z_STAND - HEALTHY_Z_MIN  # = 1.4 - 1.0 = 0.4
        height_fraction = max(0.0, (z - HEALTHY_Z_MIN) / z_range)  # 0→1
        height_reward = V3_W_HEIGHT * height_fraction

        # 5. 交替步态奖励
        foot_contacts = self._get_foot_contacts()
        # alternating = 1 当且仅当一脚在地（完美行走步态）
        alternating = abs(float(foot_contacts[0]) - float(foot_contacts[1]))
        alt_gait_reward = V3_W_ALT_GAIT * alternating * max(0.0, x_velocity)

        return (healthy_reward + forward_reward + height_reward
                + alt_gait_reward - ctrl_cost)
