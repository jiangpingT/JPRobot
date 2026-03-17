"""人形机器人后空翻（Backflip）环境 — 基于 Gymnasium Humanoid-v4。

核心设计与四足后空翻（env_backflip.py）一致，针对 17 DOF 人形做关键适配：
  1. 包裹 Humanoid-v4（terminate_when_unhealthy=False）
     ↳ 默认情况下 torso_z < 1.0m 就终止 episode，后空翻必须关闭
  2. 双脚接触检测：cfrc_ext（MuJoCo 接触力数组，跑步环境已验证）
  3. 俯仰积分：qvel[4]（世界坐标系 Y 轴角速度）× dt，与四足 ang_vel[1] 等价
     ↳ 向后翻转时 qvel[4] < 0，backward_rot = max(0, -accumulated)
  4. 四阶段奖励：起跳 → 旋转（腾空）→ 落地 → 成功+站稳
  5. 所有四足 anti-gaming 机制完整移植（里程碑、单调增量、超旋转惩罚）

## 物理时间参数
  Humanoid-v4: frame_skip=5, physics timestep=0.003s → env dt = 0.015s/step
  200步 = 3秒（四足120步×0.021s≈2.5秒，等价）

## 坐标系约定
  x: 前方，z: 上方，y: 左侧（右手系）
  后空翻 = 绕 Y 轴顺时针旋转（从 +Y 看）= qvel[4] < 0

## 观测空间（382 维）：
  [0:376]  Humanoid-v4 原生观测（qpos/qvel/cinert/cvel/qfrc_actuator/cfrc_ext）
  [376]    accumulated_pitch / (2π)   # 归一化累积俯仰旋转量（后空翻核心信号）
  [377]    torso_z                     # 躯干高度 m（站立≈1.25m，倒地≈0.3m）
  [378]    vz                          # 垂直速度 m/s（>0 上升，<0 下落）
  [379]    left_foot_contact           # 左脚接触地面：0=腾空，1=着地
  [380]    right_foot_contact          # 右脚接触地面：0=腾空，1=着地
  [381]    success_flag                # 后空翻成功标志（0=翻转阶段，1=站立阶段）

热启动：支持从 humanoid_sac/best.zip（376维）迁移权重，新增 6 维零初始化。
"""

import math

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


# ── 物理时间参数 ────────────────────────────────────────────────────────────
_DT = 0.015   # Humanoid-v4: frame_skip=5 × physics_dt=0.003s = 0.015s/step

# ── 状态阈值 ────────────────────────────────────────────────────────────────
VZ_LAUNCH_THRESH   = 0.5                 # m/s，向上速度超过此值 = 已起跳
ROTATION_COMPLETE  = 5.0                 # rad ≈ 286°，后空翻成功门槛（宽松）
ROTATION_TARGET    = 2 * math.pi         # rad = 360°，理想旋转目标
MIN_BACKFLIP_STEPS = 50                  # 至少经历此步数才能触发 success
EPISODE_LENGTH     = 200                 # 总步长 = 3 秒
POST_SUCCESS_STEPS = 80                  # 成功后站稳阶段持续步数 = 1.2 秒
ROTATION_GATE      = math.radians(286)   # rad，落地奖励最低旋转门槛
CONTACT_THRESH     = 1.0                 # N，接触力低于此值 = 脚腾空

# ── 旋转里程碑（rad）────────────────────────────────────────────────────────
# 腾空时每越过一个角度门槛给一次性奖励，防止振荡 gaming（正转-反转-正转）
MILESTONES_RAD = [math.pi / 2, math.pi, 3 * math.pi / 2]  # 90°, 180°, 270°

# ── 奖励权重 ────────────────────────────────────────────────────────────────
# 起跳
W_JUMP              = 3.0    # 向上速度奖励（地面起跳阶段）
# 旋转（仅腾空时生效）
W_ROTATION          = 20.0   # 累积旋转量奖励（v1原始值，强信号引导旋转行为形成）
W_ROT_STEP          = 8.0    # 增量旋转奖励（单调递增，防振荡）
W_MILESTONE         = 60.0   # 里程碑绝对奖励（等于原 3×20=60，绝对值解耦）
W_NO_SPIN           = 0.3    # 腾空不旋转惩罚（每步）
W_ANTI_ROLL         = 2.0    # 防侧倒惩罚（|roll|>30° 时生效）
W_OVERROT           = 200.0  # 超 360° 惩罚（保守值，四足=500）
# 落地
W_LANDING           = 1.0    # 脚着地一次性奖励（需旋转达到 ROTATION_GATE）
W_LAND_TIMING       = 1000.0 # 落地时机精准奖励（基线 330°→360° 平方公式）
# 成功
W_SUCCESS           = 500.0  # 后空翻完成一次性大奖
W_UPRIGHTNESS_BONUS = 1.0    # 成功时直立姿态加成系数
W_ROT_COMPLETENESS  = 3000.0 # 成功时旋转完整度（基线 335°→360° 平方公式）
# 站稳
W_POST_STAND        = 20.0   # 站稳奖励（uprightness × height 门控，每步）
W_POST_HEIGHT       = 10.0   # 高度奖励（torso_z 接近站立高度，每步）
# 惩罚
W_LATERAL_FALL      = 5.0    # 侧倒终止惩罚

# ── 站稳高度参数（Humanoid-v4）────────────────────────────────────────────
# 倒地（仰躺）: torso_z ≈ 0.3-0.5m
# 完全站立:     torso_z ≈ 1.25m
POST_HEIGHT_MIN    = 0.3    # m，最低参考高度（约等于倒地时躯干高度）
POST_HEIGHT_TARGET = 1.25   # m，目标站立高度

# Humanoid-v4 前进奖励系数（用于从 base_reward 中减去，我们用自定义奖励代替）
FORWARD_REWARD_WEIGHT = 1.25


class HumanoidBackflipEnv(gym.Wrapper):
    """人形机器人后空翻 Gym 环境。

    包裹 Gymnasium Humanoid-v4，禁用不健康终止，添加后空翻专用奖励。

    设计原则（继承自四足后空翻的血泪教训）：
      1. 所有正奖励绑定到"成功瞬间"或"单调进度"，消灭 per-step 持续收入流
      2. 里程碑奖励一次性（防振荡 gaming）
      3. W_OVERROT 惩罚超 360° 旋转（防超旋转 gaming）
      4. success_flag 在 obs 中明确标记成功状态（解决策略无法区分翻转/站立阶段）
    """

    def __init__(self, render_mode=None):
        # terminate_when_unhealthy=False：关闭默认的 torso_z 低于 1.0m 终止
        # 后空翻过程中躯干会经过极低位置（翻转时约 0.3-0.5m），必须关闭
        env = gym.make("Humanoid-v4", render_mode=render_mode,
                       terminate_when_unhealthy=False)
        super().__init__(env)

        # ── 扩展观测空间：376 + 6 = 382 维 ──────────────────────────────────
        orig_low  = env.observation_space.low
        orig_high = env.observation_space.high
        # 新增 6 维的合理范围
        extra_low  = np.array([-10.0, 0.0, -10.0, 0.0, 0.0, 0.0], dtype=np.float32)
        extra_high = np.array([ 10.0, 5.0,  10.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.concatenate([orig_low, extra_low]),
            high=np.concatenate([orig_high, extra_high]),
            dtype=np.float32,
        )

        # episode 状态变量（在 reset() 中清零）
        self.step_counter           = 0
        self._launched              = False   # 已检测到起跳冲量（vz > 阈值）
        self._airborne              = False   # 双脚已腾空
        self._landed_after_launch   = False   # 起跳腾空后已重新接地
        self._success               = False   # 后空翻成功
        self._success_bonus_given   = False   # 成功 bonus 只给一次
        self._post_success_steps    = 0       # 站稳阶段计数器
        self._pitch_accumulated     = 0.0     # 累积俯仰旋转量（rad，负值=向后翻）
        self._prev_max_rot_airborne = 0.0     # 腾空阶段历史最大向后旋转量（单调递增）
        self._milestones_hit        = set()   # 已触发的里程碑集合（防重复奖励）

        # 脚的 body ID（懒初始化，__init__ 时模型可能未完全就绪）
        self._left_foot_id  = None
        self._right_foot_id = None

    # ── 内部工具 ──────────────────────────────────────────────────────────────

    def _init_foot_ids(self):
        """首次调用时初始化脚的 MuJoCo body ID。"""
        model = self.env.unwrapped.model
        self._left_foot_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_foot")
        self._right_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_foot")

    def _get_foot_contacts(self):
        """返回 (lf_contact, rf_contact)：True = 脚着地（接触力 >= 阈值）。

        cfrc_ext[body_id, :3] = 该 body 所受外部接触力的前三维（力，非力矩）。
        力的 L2 范数 >= CONTACT_THRESH(1N) → 视为着地。
        """
        if self._left_foot_id is None:
            self._init_foot_ids()
        data = self.env.unwrapped.data
        lf_force = float(np.linalg.norm(data.cfrc_ext[self._left_foot_id, :3]))
        rf_force = float(np.linalg.norm(data.cfrc_ext[self._right_foot_id, :3]))
        return lf_force >= CONTACT_THRESH, rf_force >= CONTACT_THRESH

    def _get_torso_state(self):
        """读取躯干状态：(torso_z, vz, pitch_ang_vel, roll)。

        MuJoCo free joint（Humanoid-v4 根节点）的 qpos/qvel 布局：
          qpos[0:3]  = 位置 (x, y, z)
          qpos[3:7]  = 四元数 (w, x, y, z)
          qvel[0:3]  = 线速度 (vx, vy, vz)（世界坐标系）
          qvel[3:6]  = 角速度 (wx, wy, wz)（世界坐标系）

        pitch_ang_vel = qvel[4]（世界 Y 轴角速度）：
          向后翻转 = 绕 Y 轴顺时针（从 +Y 观察）→ qvel[4] < 0
          与四足 PyBullet 的 ang_vel[1] 符号约定完全一致
        """
        data = self.env.unwrapped.data
        torso_z       = float(data.qpos[2])
        vz            = float(data.qvel[2])
        pitch_ang_vel = float(data.qvel[4])   # Y 轴角速度（向后翻 < 0）

        # 横滚角（绕 X 轴）：从四元数 [w, x, y, z] 计算
        # roll = atan2(2*(w*x + y*z), 1 - 2*(x² + y²))
        qw = float(data.qpos[3])
        qx = float(data.qpos[4])
        qy = float(data.qpos[5])
        qz = float(data.qpos[6])
        roll = math.atan2(2.0 * (qw * qx + qy * qz),
                          1.0 - 2.0 * (qx * qx + qy * qy))

        # 即时俯仰角（用于直立度计算，不是累积旋转量）
        # pitch = asin(2*(w*y - z*x))
        sinp = 2.0 * (qw * qy - qz * qx)
        sinp = max(-1.0, min(1.0, sinp))   # 数值安全截断
        pitch = math.asin(sinp)

        return torso_z, vz, pitch_ang_vel, roll, pitch

    def _aug_obs(self, obs: np.ndarray, torso_z: float, vz: float,
                 lf_contact: bool, rf_contact: bool) -> np.ndarray:
        """将 6 维后空翻专用信号拼接到原始 376 维观测末尾。"""
        extra = np.array([
            self._pitch_accumulated / (2 * math.pi),  # 归一化累积旋转（-1=翻了一圈向后）
            torso_z,                                   # 躯干高度 m
            vz,                                        # 垂直速度 m/s
            1.0 if lf_contact else 0.0,                # 左脚接触
            1.0 if rf_contact else 0.0,                # 右脚接触
            1.0 if self._success else 0.0,             # 成功标志
        ], dtype=np.float32)
        return np.concatenate([obs.astype(np.float32), extra])

    # ── Gym 接口 ──────────────────────────────────────────────────────────────

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # 重置所有 episode 状态
        self.step_counter           = 0
        self._launched              = False
        self._airborne              = False
        self._landed_after_launch   = False
        self._success               = False
        self._success_bonus_given   = False
        self._post_success_steps    = 0
        self._pitch_accumulated     = 0.0
        self._prev_max_rot_airborne = 0.0
        self._milestones_hit        = set()

        torso_z, vz, _, _, _ = self._get_torso_state()
        lf_c, rf_c = self._get_foot_contacts()
        return self._aug_obs(obs, torso_z, vz, lf_c, rf_c), info

    def step(self, action):
        was_landed_before = self._landed_after_launch  # 落地边沿检测

        # 执行动作（17 DOF 直接传递给 Humanoid-v4）
        obs, base_reward, _base_term, _base_trunc, base_info = self.env.step(action)

        # ── 读取机体状态 ────────────────────────────────────────────────────
        torso_z, vz, pitch_ang_vel, roll, pitch = self._get_torso_state()
        lf_contact, rf_contact = self._get_foot_contacts()
        any_foot_contact = lf_contact or rf_contact

        # ── 俯仰积分（仅双脚腾空时积分）────────────────────────────────
        # v2 修复：只在双脚均离地时积分，防止地面滚动时虚增 backward_rot。
        # 原版 bug：全程积分，机器人落地后在地面滚动也计入 backward_rot，
        # 导致 gaming：小跳57°落地 → 地面滚到107° → per-step旋转奖励持续发放。
        # 向后翻转时 pitch_ang_vel < 0，_pitch_accumulated 变负
        if not any_foot_contact:
            self._pitch_accumulated += pitch_ang_vel * _DT
        backward_rot = max(0.0, -self._pitch_accumulated)

        # ── 状态机更新 ──────────────────────────────────────────────────────
        # 起跳检测：向上速度超过阈值
        if vz > VZ_LAUNCH_THRESH:
            self._launched = True
        # 腾空检测：起跳后双脚均离地
        if self._launched and not any_foot_contact:
            self._airborne = True
        # 落地检测：曾腾空后重新有脚接触地面
        if self._airborne and any_foot_contact:
            self._landed_after_launch = True
        # 成功判断：向后旋转量达标 + 落地 + 至少经历 MIN_BACKFLIP_STEPS 步
        if (backward_rot >= ROTATION_COMPLETE
                and self._landed_after_launch
                and self.step_counter >= MIN_BACKFLIP_STEPS):
            self._success = True

        # ── 奖励计算 ────────────────────────────────────────────────────────
        reward = 0.0

        # 1. 起跳奖励：向上速度越大越好（鼓励爆发起跳）
        #    仅在地面起跳阶段（已起跳但未腾空时）生效
        if vz > 0 and not self._airborne:
            reward += W_JUMP * vz

        # 2. 旋转奖励（仅腾空 AND 尚未落地时）
        # v2 修复：加 not _landed_after_launch，切断"落地后继续滚动拿旋转奖励"的 gaming。
        # _airborne 是单向 latch（落地后仍为 True），必须额外加落地检测来截断奖励。
        if self._airborne and not self._landed_after_launch:
            # 2a. 累积旋转量奖励：连续平滑信号
            rot_progress = min(1.0, backward_rot / ROTATION_COMPLETE)
            reward += W_ROTATION * rot_progress

            # 2b. 增量旋转奖励：只奖励历史最大值的新增部分（单调，防振荡）
            delta = backward_rot - self._prev_max_rot_airborne
            if delta > 0:
                self._prev_max_rot_airborne = backward_rot
                reward += W_ROT_STEP * delta

            # 2c. 里程碑一次性奖励：每越过 90°/180°/270° 给一次
            for m in MILESTONES_RAD:
                if backward_rot >= m and m not in self._milestones_hit:
                    self._milestones_hit.add(m)
                    reward += W_MILESTONE  # v3: 绝对值，不再乘W_ROTATION（解耦）

            # 2d. 腾空不旋转惩罚：防止在空中僵直等落下
            if backward_rot < 0.1:
                reward -= W_NO_SPIN

            # 2e. 超旋转惩罚（>360°）：防止超旋 gaming
            if backward_rot > ROTATION_TARGET:
                reward -= W_OVERROT * (backward_rot - ROTATION_TARGET)

            # 2f. 防侧倒惩罚（|roll| > 30°）：防止陀螺进动导致侧翻
            roll_excess = abs(roll) - 0.52  # 0.52 rad ≈ 30° 容限
            if roll_excess > 0:
                reward -= W_ANTI_ROLL * roll_excess

        # 3. 落地奖励（一次性，just_landed 时触发）
        if self._landed_after_launch and not was_landed_before:
            # 3a. 落地时机精准奖励：鼓励转够 360° 才落地（基线 330°→360° 平方）
            rot_at_land_deg = math.degrees(backward_rot)
            land_ratio = min(1.0, max(0.0,
                (rot_at_land_deg - 330.0) / (360.0 - 330.0)
            ))
            reward += W_LAND_TIMING * land_ratio ** 2

            # 3b. 脚着地一次性奖励（需旋转达到门槛）
            if backward_rot >= ROTATION_GATE:
                n_feet = (1 if lf_contact else 0) + (1 if rf_contact else 0)
                reward += W_LANDING * (n_feet / 2)

        # 4. 成功 bonus（一次性，翻转完成瞬间触发）
        if self._success and not self._success_bonus_given:
            self._success_bonus_given = True
            # 直立姿态加成：以即时俯仰+横滚角衡量（落地时接近直立则更高）
            uprightness = math.exp(-2.0 * (pitch ** 2 + roll ** 2))
            reward += W_SUCCESS * (1.0 + W_UPRIGHTNESS_BONUS * uprightness)
            # 旋转完整度奖励（基线 335°→360° 平方，梯度在最后 25° 极陡）
            rot_deg = math.degrees(backward_rot)
            rot_ratio = min(1.0, max(0.0, (rot_deg - 335.0) / (360.0 - 335.0)))
            reward += W_ROT_COMPLETENESS * rot_ratio ** 2

        # 5. 站稳阶段（success 后 POST_SUCCESS_STEPS 步）
        #    同时激励角度（uprightness）+ 高度（torso_z 接近站立高度）
        #    uprightness × height 相乘 = 高度门控，防止趴地骗直立分
        if self._success and self._post_success_steps < POST_SUCCESS_STEPS:
            uprightness  = math.exp(-2.0 * (pitch ** 2 + roll ** 2))
            height_ratio = min(1.0, max(0.0,
                (torso_z - POST_HEIGHT_MIN) / (POST_HEIGHT_TARGET - POST_HEIGHT_MIN)
            ))
            reward += (W_POST_STAND * uprightness * height_ratio
                     + W_POST_HEIGHT * height_ratio)
            self._post_success_steps += 1

        # ── 终止判断 ────────────────────────────────────────────────────────
        self.step_counter += 1
        terminated = truncated = False

        # 侧倒终止（|roll| > 90°）：只因侧翻终止，不因俯仰终止（允许完整翻转）
        if abs(roll) > math.pi / 2:
            terminated = True
            reward -= W_LATERAL_FALL

        # 超时截断
        if self.step_counter >= EPISODE_LENGTH:
            truncated = True

        # 站稳阶段结束后终止
        if self._success and self._post_success_steps >= POST_SUCCESS_STEPS:
            terminated = True

        info = {
            "success":      self._success,
            "rotation_deg": math.degrees(abs(self._pitch_accumulated)),
            "backward_rot_deg": math.degrees(backward_rot),
            "torso_z":      torso_z,
            "airborne":     self._airborne,
            "landed":       self._landed_after_launch,
            "just_landed":  self._landed_after_launch and not was_landed_before,
            "lf_contact":   lf_contact,
            "rf_contact":   rf_contact,
        }

        return self._aug_obs(obs, torso_z, vz, lf_contact, rf_contact), reward, terminated, truncated, info
