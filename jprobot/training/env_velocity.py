"""Gymnasium environment for BittleX — velocity command tracking paradigm (Route B).

This is an ALTERNATIVE to env.py's direction-conditioned locomotion.
The key difference: instead of "move in direction X", the command is "move at speed V m/s".

=== Route A vs Route B comparison ===

Route A (env.py / env_v2.py) — Direction conditioning:
  command: target_dir_xy = unit vector [1, 0] (forward)
  reward:  FAC_MOVEMENT * dot(delta_position, target_dir)
  agent learns: move AS FAR AS POSSIBLE in this direction (no speed constraint)
  result: agent may learn to slide/shuffle/crawl if it moves far enough

Route B (this file) — Velocity tracking:
  command: vel_cmd_xy = [vx, vy] in m/s, e.g. [0.3, 0.0]
  reward:  FAC_VEL * exp(-||vel_cmd - actual_vel||² / sigma)
  agent learns: match THIS EXACT SPEED (not "go fast", but "go at 0.3 m/s")
  result: agent must regulate speed, which forces consistent gait at each speed

Why velocity tracking is used in SOTA (legged_gym, Unitree, ETH):
  1. Zero-velocity command → standing/balance reward (no direction bias)
  2. Speed curriculum: start with slow commands, gradually increase speed
  3. Transfers to hardware: real robot receives velocity commands (like joystick input)
  4. Separates "can it move" from "can it move at the right speed"

Velocity tracking reward formula (from legged_gym):
  vel_error = ||[vx_cmd, vy_cmd] - [vx_actual, vy_actual]||²
  reward = exp(-vel_error / sigma)
  sigma = 0.25  (lower sigma = more precise required, exponential decay)

  Examples with sigma=0.25:
  - Perfect match (error=0):   exp(0) = 1.0  (maximum)
  - 0.2 m/s off in one axis:  exp(-0.04/0.25) = exp(-0.16) ≈ 0.85
  - 0.5 m/s off in one axis:  exp(-0.25/0.25) = exp(-1.0)  ≈ 0.37
  - Completely wrong speed:    exp(-large) → 0.0

State space (250 dimensions):
    - Body orientation quaternion (4)
    - Angular velocity roll/pitch, scaled (2)
    - Linear velocity xy, actual (scaled) (2)  ← velocity feedback for tracking
    - Joint angle history: 30 timesteps x 8 joints (240)
    - Velocity command [vx_cmd, vy_cmd], scaled (2)  ← replaces target_dir

Action space (8 dimensions, continuous [-1, 1]):
    - Joint angle increments for 8 actuated joints
"""

import os
import math

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data


# === Constants ===
GUI_MODE = False
EPISODE_LENGTH = 250
STEP_ANGLE = 11
JOINT_LIMIT = 110
MOTOR_FORCE = 0.2
MOTOR_VELOCITY = 10 * math.pi
INIT_ANGLE = 50
LENGTH_JOINT_HISTORY = 30
SIMULATION_STEPS = 3

# Velocity command tracking reward
# FAC_VEL_TRACKING × exp(-error/sigma) per step
# Max per episode: 250 × FAC_VEL_TRACKING (when perfect tracking throughout)
# Target: similar magnitude to Route A (~1200), so FAC_VEL_TRACKING = 5.0
FAC_VEL_TRACKING = 5.0
TRACKING_SIGMA = 0.25   # From legged_gym. Lower = stricter speed matching required.

# Velocity command ranges (m/s in world frame)
# BittleX nominal speed ~0.1-0.3 m/s during walking gait
VEL_CMD_RANGE_X = [-0.35, 0.35]   # forward/backward
VEL_CMD_RANGE_Y = [-0.25, 0.25]   # lateral (left/right)
VEL_CMD_MIN_NORM = 0.1            # zero-out commands smaller than this (standing)

# Actual linear velocity scaling (for observation)
LIN_VEL_SCALE = 2.5               # scales 0.4 m/s → 1.0 in obs

# Command scaling (for observation, same scale as LIN_VEL_SCALE so policy sees apples-to-apples)
VEL_CMD_SCALE = 2.5               # cmd obs = vel_cmd * LIN_VEL_SCALE

# Penalty weights (identical to env.py for comparability)
FAC_SMOOTH_1 = 1.0
FAC_SMOOTH_2 = 1.0
FAC_STABILITY = 0.1
FAC_Z_VELOCITY = 0.0
FAC_ARM_CONTACT = 0.03            # Same as env_v2.py (stronger than original 0.01)
FAC_SURVIVAL = 0.0
FAC_HEIGHT_BONUS = 0.0
PENALTY_STEPS = 2_000_000

RANDOM_GYRO = 0
RANDOM_JOINT_ANGS = 0

_START_POS = [0, 0, 0.08]


class BittleGymEnvVelocity(gym.Env):
    """BittleX env with velocity command tracking reward (Route B).

    Training usage:
        env = BittleGymEnvVelocity(config={
            "vel_cmd_range_x": [-0.3, 0.3],   # start slower, increase over curriculum
            "vel_cmd_range_y": [-0.2, 0.2],
        })
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None, config=None):
        super().__init__()

        self.render_mode = render_mode
        self.config = config or {}

        self.gui_mode = self.config.get("gui_mode", GUI_MODE)
        self.episode_length = self.config.get("episode_length", EPISODE_LENGTH)

        self.n_joints = 8
        # obs: body(6) + lin_vel_xy(2) + joint_history(240) + vel_cmd(2) = 250
        obs_dim = 6 + 2 + LENGTH_JOINT_HISTORY * self.n_joints + 2  # 250

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
        self.vel_cmd_xy = np.zeros(2, dtype=np.float32)   # current velocity command

        self.arm_contact = 0
        self._ep_heights: list[float] = []
        self._ep_arm_contacts: list[bool] = []
        self._ep_tilts: list[float] = []
        self._ep_tracking_errors: list[float] = []       # velocity tracking error per step
        self._ep_vel_rewards: list[float] = []
        self._last_arm_contact: bool = False

        self.urdf_path = os.path.join(
            os.path.dirname(__file__), "..", "models", "bittle_esp32.urdf"
        )

    def _sample_vel_cmd(self) -> np.ndarray:
        """Sample a velocity command for this episode.

        Config options:
          fixed_vel_cmd: [vx, vy]  — always use this command (bootstrap stage)
          vel_cmd_range_x: [lo, hi] — vx range in m/s
          vel_cmd_range_y: [lo, hi] — vy range in m/s
          vel_cmd_min_norm: float   — zero out commands smaller than this

        Returns:
            vel_cmd: [vx, vy] in m/s (world frame)
        """
        # Fixed command mode: used in early bootstrap stages to guarantee
        # the robot always gets the same direction target (e.g. forward at 0.25 m/s).
        if "fixed_vel_cmd" in self.config:
            return np.array(self.config["fixed_vel_cmd"], dtype=np.float32)

        range_x = self.config.get("vel_cmd_range_x", VEL_CMD_RANGE_X)
        range_y = self.config.get("vel_cmd_range_y", VEL_CMD_RANGE_Y)
        min_norm = self.config.get("vel_cmd_min_norm", VEL_CMD_MIN_NORM)

        vx = float(self.np_random.uniform(range_x[0], range_x[1]))
        vy = float(self.np_random.uniform(range_y[0], range_y[1]))
        cmd = np.array([vx, vy], dtype=np.float32)

        # Zero out very small commands → standing task
        if float(np.linalg.norm(cmd)) < min_norm:
            cmd = np.zeros(2, dtype=np.float32)

        return cmd

    def _get_lin_vel_xy(self) -> np.ndarray:
        """Read world-frame linear velocity, scale and clip to [-1, 1]."""
        lin_vel = p.getBaseVelocity(
            self.robot_id, physicsClientId=self.physics_client
        )[0]
        return np.clip(
            np.array(lin_vel[:2], dtype=np.float32) * LIN_VEL_SCALE,
            -1.0, 1.0
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.physics_client is None:
            if self.gui_mode or (self.render_mode == "human"):
                self.physics_client = p.connect(p.GUI)
            else:
                self.physics_client = p.connect(p.DIRECT)

        p.resetSimulation(physicsClientId=self.physics_client)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0,
                                   physicsClientId=self.physics_client)
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
            p.resetJointState(self.robot_id, j, init_angs[i],
                              physicsClientId=self.physics_client)

        joint_states = p.getJointStates(
            self.robot_id, self.joint_id, physicsClientId=self.physics_client,
        )
        joint_angs_norm = np.array([s[0] for s in joint_states]) / self._bound_ang
        self.joint_angles_norm = joint_angs_norm.copy()
        self.prev_joint_angles = joint_angs_norm.copy()
        self.prev_prev_joint_angles = joint_angs_norm.copy()
        self.joint_history = np.tile(joint_angs_norm, (LENGTH_JOINT_HISTORY, 1))

        state_ang = p.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=self.physics_client
        )[1]
        state_vel = np.asarray(
            p.getBaseVelocity(self.robot_id, physicsClientId=self.physics_client)[1]
        )
        state_vel_scaled = np.clip(state_vel[0:2] * 0.1, -1, 1)
        self._state_robot = np.concatenate((state_ang, state_vel_scaled))

        self._lin_vel_xy = np.zeros(2, dtype=np.float32)

        # Sample velocity command for this episode (or accept override)
        if options and "vel_cmd" in options:
            # Allow explicit command for evaluation: e.g., options={"vel_cmd": [0.3, 0.0]}
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

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1,
                                   physicsClientId=self.physics_client)

        # Observation: vel_cmd scaled same as lin_vel for apples-to-apples comparison
        vel_cmd_obs = np.clip(self.vel_cmd_xy * VEL_CMD_SCALE, -1.0, 1.0)

        observation = np.concatenate((
            self._state_robot,      # 6
            self._lin_vel_xy,       # 2: actual velocity (feedback)
            self.joint_history.flatten(),  # 240
            vel_cmd_obs,            # 2: command (what we want to achieve)
        ))
        return observation.astype(np.float32), {}

    def step(self, action):
        last_pose = p.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=self.physics_client
        )
        last_position = last_pose[0]

        joint_states = p.getJointStates(
            self.robot_id, self.joint_id, physicsClientId=self.physics_client,
        )
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
            if p.getContactPoints(bodyA=self.robot_id, linkIndexA=idx,
                                  physicsClientId=self.physics_client):
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

        state_pos, state_ang = p.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=self.physics_client
        )

        p.stepSimulation(physicsClientId=self.physics_client)

        euler = p.getEulerFromQuaternion(state_ang)
        state_vel_ang = np.asarray(
            p.getBaseVelocity(self.robot_id, physicsClientId=self.physics_client)[1]
        )
        state_vel_scaled = np.clip(state_vel_ang[0:2] * 0.1, -1, 1)
        self._state_robot = np.concatenate((state_ang, state_vel_scaled))

        # Read actual linear velocity (world frame)
        lin_vel_world = np.array(
            p.getBaseVelocity(self.robot_id, physicsClientId=self.physics_client)[0][:2],
            dtype=np.float32,
        )
        self._lin_vel_xy = np.clip(lin_vel_world * LIN_VEL_SCALE, -1.0, 1.0)

        # === Velocity tracking reward ===
        # Compare actual velocity vs commanded velocity (both in raw m/s)
        vel_error_sq = float(np.sum((self.vel_cmd_xy - lin_vel_world) ** 2))
        tracking_sigma = self.config.get("tracking_sigma", TRACKING_SIGMA)
        fac_vel = self.config.get("fac_vel_tracking", FAC_VEL_TRACKING)
        vel_reward = fac_vel * float(np.exp(-vel_error_sq / tracking_sigma))

        # === Penalty terms (same structure as env.py) ===
        fac_smooth_1 = self.config.get("fac_smooth_1", FAC_SMOOTH_1)
        fac_smooth_2 = self.config.get("fac_smooth_2", FAC_SMOOTH_2)
        smooth_penalty = float(np.sum(
            fac_smooth_1 * (self.joint_angles_norm - self.prev_joint_angles) ** 2
            + fac_smooth_2 * (self.joint_angles_norm
                              - 2 * self.prev_joint_angles
                              + self.prev_prev_joint_angles) ** 2
        ))

        fac_stability = self.config.get("fac_stability", FAC_STABILITY)
        z_velocity = float(
            p.getBaseVelocity(self.robot_id, physicsClientId=self.physics_client)[0][2]
        )
        body_stability = (fac_stability * float(state_vel_scaled[0] ** 2 + state_vel_scaled[1] ** 2)
                          + self.config.get("fac_z_velocity", FAC_Z_VELOCITY) * z_velocity ** 2)

        fac_arm_contact = self.config.get("fac_arm_contact", FAC_ARM_CONTACT)

        reward = self.config.get("fac_survival", FAC_SURVIVAL) + vel_reward

        fac_height_bonus = self.config.get("fac_height_bonus", FAC_HEIGHT_BONUS)
        if fac_height_bonus > 0:
            reward += fac_height_bonus * state_pos[2]

        reward -= self.step_counter_session / self.config.get("penalty_steps", PENALTY_STEPS) * (
            smooth_penalty + body_stability + fac_arm_contact * self.arm_contact
        )

        # Track episode metrics
        self._ep_heights.append(state_pos[2])
        self._ep_tilts.append(abs(euler[0]) + abs(euler[1]))
        self._ep_arm_contacts.append(self._last_arm_contact)
        self._ep_tracking_errors.append(vel_error_sq)
        self._ep_vel_rewards.append(vel_reward)

        self.step_counter += 1
        terminated = False
        truncated = False
        info = {}

        if self.step_counter > self.episode_length:
            self.step_counter_session += self.step_counter
            truncated = True
        elif abs(euler[0]) > 1.3 or abs(euler[1]) > 1.3:
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
                'ep_len': int(self.step_counter),
            }

        vel_cmd_obs = np.clip(self.vel_cmd_xy * VEL_CMD_SCALE, -1.0, 1.0)
        observation = np.concatenate((
            self._state_robot,      # 6
            self._lin_vel_xy,       # 2
            self.joint_history.flatten(),  # 240
            vel_cmd_obs,            # 2
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

