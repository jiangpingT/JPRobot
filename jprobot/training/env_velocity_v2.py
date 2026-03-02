"""Gymnasium environment for BittleX — velocity command tracking v2 (Route B v3).

Key additions vs env_velocity.py (v1):

1. feet_air_time reward  ← 核心修复
   Learned from legged_gym / unitree_rl_gym SOTA.
   给奖励的条件：脚在空中 > threshold 秒后落地。
   反骗分原理：站着不动 → 脚不离地 → air_time = 0 → 此项奖励为 0。
   和速度追踪奖励互补：速度追踪梯度"模糊"（站着能得60%），
   feet_air_time 梯度"清晰"（不抬腿就是零，不可骗）。
   命令 norm < 0.1 时自动关零（站立任务不惩罚）。

2. alive bonus
   每步固定奖励 FAC_ALIVE，摔倒终止则失去后续所有 alive 奖励。
   强化"活着比骗分更重要"。

3. feet_contact_state 加入 obs（254 维，与 Route A 对齐）
   网络能感知当前步态相位（哪只脚在地），有助于学习协调步态。
   obs = body(6) + lin_vel_xy(2) + joint_history(240) + vel_cmd(2) + feet_contact(4) = 254

为什么不改 sigma/vel_cmd 参数：
   三次 Route B 精调实验已证明参数调整无法解决根本问题（梯度信号弱）。
   feet_air_time 从奖励结构上修复，不需要改其他参数。
"""

import os
import math

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data


# === Constants（与 env_velocity.py 一致，除非标注修改）===
GUI_MODE = False
EPISODE_LENGTH = 250
STEP_ANGLE = 11
JOINT_LIMIT = 110
MOTOR_FORCE = 0.2
MOTOR_VELOCITY = 10 * math.pi
INIT_ANGLE = 50
LENGTH_JOINT_HISTORY = 30
SIMULATION_STEPS = 3

# Velocity tracking reward（同 v1）
FAC_VEL_TRACKING = 5.0
TRACKING_SIGMA = 0.15      # 保持 v2 修复后的 0.15（比 legged_gym 默认 0.25 更严格）

# Velocity command ranges（同 v1）
VEL_CMD_RANGE_X = [-0.35, 0.35]
VEL_CMD_RANGE_Y = [-0.25, 0.25]
VEL_CMD_MIN_NORM = 0.15    # 保持 0.15（防止零速命令）

# Observation scaling（同 v1）
LIN_VEL_SCALE = 2.5
VEL_CMD_SCALE = 2.5

# Penalty weights（同 v1）
FAC_SMOOTH_1 = 1.0
FAC_SMOOTH_2 = 1.0
FAC_STABILITY = 0.1
FAC_Z_VELOCITY = 0.0
FAC_ARM_CONTACT = 0.03
PENALTY_STEPS = 2_000_000

# ── 新增：feet_air_time 奖励 ───────────────────────────────────────────────────
# 每只脚落地时，奖励 = max(0, 空中时长 - threshold) × FAC_FEET_AIR_TIME
# 物理含义：超过 threshold 秒的空中时长才被奖励，确保真正的步态（不只是颤抖）
#
# 参数校准（BittleX 小四足）：
#   dt_env = 3 × (1/240s) ≈ 0.0125s per env step
#   正常步态下脚在空中约 0.15-0.30s，threshold = 0.10s 基本都能超过
#   FAC_FEET_AIR_TIME = 5.0 → 正常走路每局贡献约 20-50 分（速度追踪上限 1250）
#   量级比速度追踪小，但梯度方向清晰，不可通过静止获得
FAC_FEET_AIR_TIME = 5.0
FEET_AIR_TIME_THRESHOLD = 0.10   # seconds：空中时长超过此值才给奖励

# 每个 env step 对应的物理时间（用于 feet_air_time 积分）
DT_ENV = SIMULATION_STEPS / 240.0   # ≈ 0.0125s

# ── 新增：alive bonus ─────────────────────────────────────────────────────────
# 每步固定奖励，摔倒则失去后续所有 alive 奖励
# 0.2/步 × 250步 = 50分/局（适度，不过度主导奖励结构）
FAC_ALIVE = 0.2

# Paw link 索引（与 env_v2.py 相同，已验证）
# [left_front_paw, right_front_paw, right_back_paw, left_back_paw]
_PAW_LINK_INDICES = [3, 6, 9, 12]

RANDOM_GYRO = 0
RANDOM_JOINT_ANGS = 0
_START_POS = [0, 0, 0.08]


class BittleGymEnvVelocityV2(gym.Env):
    """BittleX velocity tracking env v2 (Route B v3).

    obs = 254：body(6) + lin_vel_xy(2) + joint_history(240) + vel_cmd(2) + feet_contact(4)
    与 Route A (BittleGymEnvV2, obs=254) 维度对齐，方便未来 MoE 融合。
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None, config=None):
        super().__init__()

        self.render_mode = render_mode
        self.config = config or {}

        self.gui_mode = self.config.get("gui_mode", GUI_MODE)
        self.episode_length = self.config.get("episode_length", EPISODE_LENGTH)

        self.n_joints = 8
        # obs: body(6) + lin_vel_xy(2) + joint_history(240) + vel_cmd(2) + feet_contact(4) = 254
        obs_dim = 6 + 2 + LENGTH_JOINT_HISTORY * self.n_joints + 2 + 4

        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_joints,), dtype=np.float32
        )

        self.robot_id = None
        self.physics_client = None
        self.joint_id = []
        self._bound_ang = np.deg2rad(JOINT_LIMIT)
        self.step_counter = 0
        self.step_counter_session = 0
        self.joint_angles_norm = np.zeros(self.n_joints)
        self.joint_history = np.zeros((LENGTH_JOINT_HISTORY, self.n_joints))
        self.prev_joint_angles = np.zeros(self.n_joints)
        self.prev_prev_joint_angles = np.zeros(self.n_joints)
        self._state_robot = np.zeros(6)
        self._lin_vel_xy = np.zeros(2, dtype=np.float32)
        self.vel_cmd_xy = np.zeros(2, dtype=np.float32)

        # feet_air_time 状态
        self._feet_air_time = np.zeros(4, dtype=np.float32)    # 每只脚离地时长（秒）
        self._prev_feet_contact = np.ones(4, dtype=bool)       # 上一步是否接触地面
        self._feet_contact = np.ones(4, dtype=np.float32)      # 当前接触状态（用于 obs）

        self.arm_contact = 0
        self._ep_heights: list[float] = []
        self._ep_arm_contacts: list[bool] = []
        self._ep_tilts: list[float] = []
        self._ep_tracking_errors: list[float] = []
        self._ep_vel_rewards: list[float] = []
        self._ep_air_time_rewards: list[float] = []
        self._last_arm_contact: bool = False

        self.urdf_path = os.path.join(
            os.path.dirname(__file__), "..", "models", "bittle_esp32.urdf"
        )

    def _sample_vel_cmd(self) -> np.ndarray:
        """Sample velocity command for this episode."""
        if "fixed_vel_cmd" in self.config:
            return np.array(self.config["fixed_vel_cmd"], dtype=np.float32)

        range_x = self.config.get("vel_cmd_range_x", VEL_CMD_RANGE_X)
        range_y = self.config.get("vel_cmd_range_y", VEL_CMD_RANGE_Y)
        min_norm = self.config.get("vel_cmd_min_norm", VEL_CMD_MIN_NORM)

        vx = float(self.np_random.uniform(range_x[0], range_x[1]))
        vy = float(self.np_random.uniform(range_y[0], range_y[1]))
        cmd = np.array([vx, vy], dtype=np.float32)

        if float(np.linalg.norm(cmd)) < min_norm:
            cmd = np.zeros(2, dtype=np.float32)

        return cmd

    def _get_lin_vel_xy(self) -> np.ndarray:
        lin_vel = p.getBaseVelocity(self.robot_id, physicsClientId=self.physics_client)[0]
        return np.clip(np.array(lin_vel[:2], dtype=np.float32) * LIN_VEL_SCALE, -1.0, 1.0)

    def _get_feet_contact(self) -> np.ndarray:
        """检测四只爪子是否接触地面，返回 [4] float32 二值数组。

        link 索引 [3,6,9,12] = left_front/right_front/right_back/left_back paw。
        与 env_v2.py 使用相同索引（已验证）。
        """
        contact = np.zeros(4, dtype=np.float32)
        for i, link_idx in enumerate(_PAW_LINK_INDICES):
            pts = p.getContactPoints(
                bodyA=self.robot_id, linkIndexA=link_idx,
                physicsClientId=self.physics_client,
            )
            if pts:
                contact[i] = 1.0
        return contact

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.physics_client is None:
            if self.gui_mode or (self.render_mode == "human"):
                self.physics_client = p.connect(p.GUI)
            else:
                self.physics_client = p.connect(p.DIRECT)

        p.resetSimulation(physicsClientId=self.physics_client)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=self.physics_client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf", physicsClientId=self.physics_client)

        start_orient = p.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = p.loadURDF(
            self.urdf_path, _START_POS, start_orient,
            flags=p.URDF_USE_SELF_COLLISION,
            physicsClientId=self.physics_client,
        )

        self.joint_id = []
        for j in range(p.getNumJoints(self.robot_id, physicsClientId=self.physics_client)):
            info = p.getJointInfo(self.robot_id, j, physicsClientId=self.physics_client)
            if info[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                self.joint_id.append(j)
                p.changeDynamics(
                    self.robot_id, j,
                    maxJointVelocity=np.pi * 10,
                    physicsClientId=self.physics_client,
                )

        init_angs = np.deg2rad(np.array([1, 0, 1, 0, 1, 0, 1, 0]) * INIT_ANGLE)
        for i, j in enumerate(self.joint_id):
            p.resetJointState(self.robot_id, j, init_angs[i], physicsClientId=self.physics_client)

        joint_states = p.getJointStates(self.robot_id, self.joint_id, physicsClientId=self.physics_client)
        joint_angs_norm = np.array([s[0] for s in joint_states]) / self._bound_ang
        self.joint_angles_norm = joint_angs_norm.copy()
        self.prev_joint_angles = joint_angs_norm.copy()
        self.prev_prev_joint_angles = joint_angs_norm.copy()
        self.joint_history = np.tile(joint_angs_norm, (LENGTH_JOINT_HISTORY, 1))

        state_ang = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.physics_client)[1]
        state_vel = np.asarray(p.getBaseVelocity(self.robot_id, physicsClientId=self.physics_client)[1])
        state_vel_scaled = np.clip(state_vel[0:2] * 0.1, -1, 1)
        self._state_robot = np.concatenate((state_ang, state_vel_scaled))

        self._lin_vel_xy = np.zeros(2, dtype=np.float32)

        # Reset feet_air_time state（每局重置）
        self._feet_air_time[:] = 0.0
        self._prev_feet_contact[:] = True   # 初始姿态四脚着地
        self._feet_contact[:] = 1.0

        if options and "vel_cmd" in options:
            self.vel_cmd_xy = np.array(options["vel_cmd"], dtype=np.float32)
        else:
            self.vel_cmd_xy = self._sample_vel_cmd()

        self.step_counter = 0
        self.arm_contact = 0
        self._ep_heights.clear()
        self._ep_arm_contacts.clear()
        self._ep_tilts.clear()
        self._ep_tracking_errors.clear()
        self._ep_vel_rewards.clear()
        self._ep_air_time_rewards.clear()

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.physics_client)

        vel_cmd_obs = np.clip(self.vel_cmd_xy * VEL_CMD_SCALE, -1.0, 1.0)
        observation = np.concatenate((
            self._state_robot,            # 6
            self._lin_vel_xy,             # 2
            self.joint_history.flatten(), # 240
            vel_cmd_obs,                  # 2
            self._feet_contact,           # 4  ← NEW
        ))
        return observation.astype(np.float32), {}

    def step(self, action):
        last_pose = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.physics_client)
        last_position = last_pose[0]

        joint_states = p.getJointStates(self.robot_id, self.joint_id, physicsClientId=self.physics_client)
        joint_angs = np.array([s[0] for s in joint_states])

        ds = np.deg2rad(STEP_ANGLE)
        joint_angs += action * ds
        joint_angs = np.clip(joint_angs, -self._bound_ang, self._bound_ang)
        joint_angs_deg = np.rad2deg(joint_angs).round()
        joint_angs = np.deg2rad(joint_angs_deg)

        p.stepSimulation(physicsClientId=self.physics_client)

        arm_link_indices = [1, 2, 4, 5]
        self._last_arm_contact = False
        for idx in arm_link_indices:
            if p.getContactPoints(bodyA=self.robot_id, linkIndexA=idx, physicsClientId=self.physics_client):
                self.arm_contact += 1
                self._last_arm_contact = True

        p.setJointMotorControlArray(
            self.robot_id, self.joint_id, p.POSITION_CONTROL, joint_angs,
            forces=np.ones(self.n_joints) * MOTOR_FORCE,
            physicsClientId=self.physics_client,
        )

        p.stepSimulation(physicsClientId=self.physics_client)

        joint_angs_norm = joint_angs / self._bound_ang

        if self.step_counter % 2 == 0:
            normalized = self._randomize(joint_angs_norm, RANDOM_JOINT_ANGS)
            self.joint_history = np.roll(self.joint_history, -1, axis=0)
            self.joint_history[-1] = normalized

        self.prev_prev_joint_angles = self.prev_joint_angles.copy()
        self.prev_joint_angles = self.joint_angles_norm.copy()
        self.joint_angles_norm = joint_angs_norm.copy()

        state_pos, state_ang = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.physics_client)

        p.stepSimulation(physicsClientId=self.physics_client)

        euler = p.getEulerFromQuaternion(state_ang)
        state_vel_ang = np.asarray(p.getBaseVelocity(self.robot_id, physicsClientId=self.physics_client)[1])
        state_vel_scaled = np.clip(state_vel_ang[0:2] * 0.1, -1, 1)
        self._state_robot = np.concatenate((state_ang, state_vel_scaled))

        lin_vel_world = np.array(
            p.getBaseVelocity(self.robot_id, physicsClientId=self.physics_client)[0][:2],
            dtype=np.float32,
        )
        self._lin_vel_xy = np.clip(lin_vel_world * LIN_VEL_SCALE, -1.0, 1.0)

        # ── Feet contact & air time ───────────────────────────────────────────
        feet_contact = self._get_feet_contact()          # [4] float32, 0/1
        feet_in_contact = feet_contact.astype(bool)      # [4] bool

        # first_contact：上一步在空中，这一步刚着地
        first_contact = feet_in_contact & ~self._prev_feet_contact

        # feet_air_time reward：只在有运动指令时生效（命令 norm > 0.1）
        # 解决骗分根源：不抬腿 → air_time 永远为 0 → 此项奖励为 0
        cmd_norm = float(np.linalg.norm(self.vel_cmd_xy))
        fac_air_time = self.config.get("fac_feet_air_time", FAC_FEET_AIR_TIME)
        if cmd_norm > 0.1:
            air_time_reward = float(np.sum(
                np.maximum(0.0, self._feet_air_time - FEET_AIR_TIME_THRESHOLD) * first_contact
            )) * fac_air_time
        else:
            air_time_reward = 0.0

        # 更新 air_time：不在地 → 累加；刚落地 → 归零
        self._feet_air_time += DT_ENV * (~feet_in_contact).astype(np.float32)
        self._feet_air_time *= (~first_contact).astype(np.float32)
        self._prev_feet_contact = feet_in_contact.copy()
        self._feet_contact = feet_contact.copy()

        # ── Velocity tracking reward ─────────────────────────────────────────
        vel_error_sq = float(np.sum((self.vel_cmd_xy - lin_vel_world) ** 2))
        tracking_sigma = self.config.get("tracking_sigma", TRACKING_SIGMA)
        fac_vel = self.config.get("fac_vel_tracking", FAC_VEL_TRACKING)
        vel_reward = fac_vel * float(np.exp(-vel_error_sq / tracking_sigma))

        # ── Alive bonus ───────────────────────────────────────────────────────
        fac_alive = self.config.get("fac_alive", FAC_ALIVE)
        alive_reward = fac_alive

        # ── Penalty terms ────────────────────────────────────────────────────
        fac_smooth_1 = self.config.get("fac_smooth_1", FAC_SMOOTH_1)
        fac_smooth_2 = self.config.get("fac_smooth_2", FAC_SMOOTH_2)
        smooth_penalty = float(np.sum(
            fac_smooth_1 * (self.joint_angles_norm - self.prev_joint_angles) ** 2
            + fac_smooth_2 * (self.joint_angles_norm
                              - 2 * self.prev_joint_angles
                              + self.prev_prev_joint_angles) ** 2
        ))

        fac_stability = self.config.get("fac_stability", FAC_STABILITY)
        z_velocity = float(p.getBaseVelocity(self.robot_id, physicsClientId=self.physics_client)[0][2])
        body_stability = (fac_stability * float(state_vel_scaled[0] ** 2 + state_vel_scaled[1] ** 2)
                          + self.config.get("fac_z_velocity", FAC_Z_VELOCITY) * z_velocity ** 2)

        fac_arm_contact = self.config.get("fac_arm_contact", FAC_ARM_CONTACT)

        penalty_scale = self.step_counter_session / self.config.get("penalty_steps", PENALTY_STEPS)
        reward = (alive_reward + vel_reward + air_time_reward
                  - penalty_scale * (smooth_penalty + body_stability + fac_arm_contact * self.arm_contact))

        # Metrics tracking
        self._ep_heights.append(state_pos[2])
        self._ep_tilts.append(abs(euler[0]) + abs(euler[1]))
        self._ep_arm_contacts.append(self._last_arm_contact)
        self._ep_tracking_errors.append(vel_error_sq)
        self._ep_vel_rewards.append(vel_reward)
        self._ep_air_time_rewards.append(air_time_reward)

        self.step_counter += 1
        terminated = False
        truncated = False
        info = {}

        height_term = self.config.get("height_termination_threshold", 0.0)
        if self.step_counter > self.episode_length:
            self.step_counter_session += self.step_counter
            truncated = True
        elif abs(euler[0]) > 1.3 or abs(euler[1]) > 1.3:
            self.step_counter_session += self.step_counter
            reward = 0
            terminated = True
        elif height_term > 0.0 and state_pos[2] < height_term:
            # 躯体高度低于阈值 → 机器人趴下爬行 → 立即终止（防止爬行骗分）
            # 物理含义：BittleX 正常站立高度约 0.08m；低于 0.04m = 躯体几乎贴地
            self.step_counter_session += self.step_counter
            reward = 0
            terminated = True

        if (terminated or truncated) and self._ep_heights:
            info['posture'] = {
                'avg_height': float(np.mean(self._ep_heights)),
                'arm_contact_rate': float(np.mean(self._ep_arm_contacts)),
                'avg_tilt_deg': float(np.degrees(np.mean(self._ep_tilts))),
            }
            info['velocity'] = {
                'vel_cmd': self.vel_cmd_xy.tolist(),
                'mean_tracking_error_sq': float(np.mean(self._ep_tracking_errors)),
                'mean_vel_reward': float(np.mean(self._ep_vel_rewards)),
                'mean_air_time_reward': float(np.mean(self._ep_air_time_rewards)),
                'ep_len': int(self.step_counter),
            }

        vel_cmd_obs = np.clip(self.vel_cmd_xy * VEL_CMD_SCALE, -1.0, 1.0)
        observation = np.concatenate((
            self._state_robot,            # 6
            self._lin_vel_xy,             # 2
            self.joint_history.flatten(), # 240
            vel_cmd_obs,                  # 2
            self._feet_contact,           # 4  ← NEW
        ))
        return observation.astype(np.float32), float(reward), terminated, truncated, info

    @staticmethod
    def _randomize(value, percentage):
        if percentage <= 0:
            return value
        percentage /= 100
        return value * (1 + percentage * (2 * np.random.rand(*value.shape) - 1))

    def render(self):
        if self.render_mode == "human":
            pass
        elif self.render_mode == "rgb_array":
            width, height = 320, 240
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0, 0, 0.05],
                distance=0.3, yaw=45, pitch=-30, roll=0, upAxisIndex=2,
            )
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60, aspect=width / height, nearVal=0.1, farVal=100,
            )
            _, _, img, _, _ = p.getCameraImage(
                width, height, view_matrix, proj_matrix,
                physicsClientId=self.physics_client,
            )
            return np.array(img, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]

    def close(self):
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None
