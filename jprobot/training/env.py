"""Gymnasium environment for BittleX reinforcement learning.

Based on ger01d/opencat-gym with improvements:
- Configurable reward weights via YAML config
- Domain randomization support
- Cleaner state representation
- Better documentation

State space (246 dimensions):
    - Body orientation quaternion (4)
    - Angular velocity roll/pitch (2)
    - Joint angle history: 30 timesteps x 8 joints (240)

Action space (8 dimensions, continuous [-1, 1]):
    - Joint angle increments for 8 actuated joints
    - Mapped to: shoulder_left, elbow_left, shoulder_right, elbow_right,
                 hip_right, knee_right, hip_left, knee_left

Reference:
    Lee et al., "Learning quadrupedal locomotion over challenging terrain",
    Science Robotics, 2020.
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
STEP_ANGLE = 11              # Max angle change per step (degrees)
JOINT_LIMIT = 110            # Max joint angle (degrees)
MOTOR_FORCE = 0.2            # Max motor torque (N)
MOTOR_VELOCITY = 10 * math.pi  # Max angular velocity (rad/s)
INIT_ANGLE = 50              # Initial joint angle (degrees)
LENGTH_JOINT_HISTORY = 30    # Timesteps of joint history in observation
SIMULATION_STEPS = 3         # Physics steps per env step (simulates serial delay)

# Reward weights (matching original opencat-gym defaults)
FAC_SURVIVAL = 0.0          # No survival reward (avoids encouraging stillness)
FAC_MOVEMENT = 1000.0
FAC_SMOOTH_1 = 1.0
FAC_SMOOTH_2 = 1.0
FAC_STABILITY = 0.1
FAC_Z_VELOCITY = 0.0
FAC_CLEARANCE = 0.0
FAC_SLIP = 0.0
FAC_ARM_CONTACT = 0.01      # Cumulative arm contact (original value)
FAC_ORIENTATION = 0.0       # No orientation penalty (original has none)
FAC_HEIGHT_BONUS = 0.0      # No height bonus (original has none)
MIN_HEIGHT = 0.045          # Minimum acceptable body height (m)
PENALTY_STEPS = 2_000_000   # 2M steps to full penalty (original value)

# Domain randomization (0 = off)
RANDOM_GYRO = 0
RANDOM_JOINT_ANGS = 0
RANDOM_MASS = 0
RANDOM_FRICTION = 0

_INIT_RAD = math.radians(INIT_ANGLE)
_START_POS = [0, 0, 0.08]
_START_ORI = [0, 0, 0, 1]   # identity quaternion (p.getQuaternionFromEuler([0,0,0]))


class BittleGymEnv(gym.Env):
    """OpenAI Gymnasium environment for BittleX locomotion training."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None, config=None):
        super().__init__()

        self.render_mode = render_mode
        self.config = config or {}

        # Override constants from config
        self.gui_mode = self.config.get("gui_mode", GUI_MODE)
        self.episode_length = self.config.get("episode_length", EPISODE_LENGTH)

        # State and action dimensions
        self.n_joints = 8
        obs_dim = LENGTH_JOINT_HISTORY * self.n_joints + 6  # 246

        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_joints,), dtype=np.float32
        )

        # State variables (joint_id populated in reset via dynamic discovery)
        self.robot_id = None
        self.physics_client = None
        self.joint_id = []             # PyBullet IDs of revolute joints (discovered in reset)
        self._bound_ang = np.deg2rad(JOINT_LIMIT)
        self.step_counter = 0
        self.step_counter_session = 0
        self.joint_angles_norm = np.zeros(self.n_joints)   # normalized [-1, 1]
        self.joint_history = np.zeros((LENGTH_JOINT_HISTORY, self.n_joints))
        self.prev_joint_angles = np.zeros(self.n_joints)   # normalized
        self.prev_prev_joint_angles = np.zeros(self.n_joints)  # normalized
        self._state_robot = np.zeros(6)  # quaternion(4) + ang_vel_scaled(2)

        # Cumulative arm contact counter (reset per episode)
        self.arm_contact = 0

        # Posture tracking per episode (written to info at episode end for dashboard)
        self._ep_heights: list[float] = []
        self._ep_arm_contacts: list[bool] = []
        self._ep_tilts: list[float] = []
        self._last_arm_contact: bool = False

        # URDF model path
        self.urdf_path = os.path.join(
            os.path.dirname(__file__), "..", "models", "bittle_esp32.urdf"
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Connect once, resetSimulation every episode (matching original)
        if self.physics_client is None:
            if self.gui_mode or (self.render_mode == "human"):
                self.physics_client = p.connect(p.GUI)
            else:
                self.physics_client = p.connect(p.DIRECT)

        # Full reset: reload entire physics scene (matching original resetSimulation)
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

        # Discover revolute/prismatic joints and set maxJointVelocity (matching original)
        self.joint_id = []
        for j in range(p.getNumJoints(self.robot_id, physicsClientId=self.physics_client)):
            info = p.getJointInfo(self.robot_id, j, physicsClientId=self.physics_client)
            joint_type = info[2]
            if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                self.joint_id.append(j)
                p.changeDynamics(
                    self.robot_id, j,
                    maxJointVelocity=np.pi * 10,
                    physicsClientId=self.physics_client,
                )

        # Initial pose: [50°, 0°, 50°, 0°, 50°, 0°, 50°, 0°] (matching original)
        # shoulder=50°, elbow=0°, shoulder=50°, elbow=0°, hip=50°, knee=0°, hip=50°, knee=0°
        init_angs = np.deg2rad(np.array([1, 0, 1, 0, 1, 0, 1, 0]) * INIT_ANGLE)
        for i, j in enumerate(self.joint_id):
            p.resetJointState(self.robot_id, j, init_angs[i],
                              physicsClientId=self.physics_client)

        # Read actual joint states and normalize
        joint_states = p.getJointStates(
            self.robot_id, self.joint_id,
            physicsClientId=self.physics_client,
        )
        joint_angs_norm = np.array([s[0] for s in joint_states]) / self._bound_ang
        self.joint_angles_norm = joint_angs_norm.copy()
        self.prev_joint_angles = joint_angs_norm.copy()
        self.prev_prev_joint_angles = joint_angs_norm.copy()

        # Initialize history with starting pose (matching original: np.tile)
        self.joint_history = np.tile(joint_angs_norm, (LENGTH_JOINT_HISTORY, 1))

        # Read robot state (quaternion + scaled angular velocity)
        state_ang = p.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=self.physics_client
        )[1]
        state_vel = np.asarray(
            p.getBaseVelocity(self.robot_id, physicsClientId=self.physics_client)[1]
        )
        state_vel_scaled = np.clip(state_vel[0:2] * 0.1, -1, 1)
        self._state_robot = np.concatenate((state_ang, state_vel_scaled))

        self.step_counter = 0
        self.arm_contact = 0
        self._ep_heights.clear()
        self._ep_arm_contacts.clear()
        self._ep_tilts.clear()

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1,
                                   physicsClientId=self.physics_client)

        observation = np.concatenate((self._state_robot, self.joint_history.flatten()))
        return observation.astype(np.float32), {}

    def step(self, action):
        # Read last position BEFORE any simulation (matching original)
        last_position = p.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=self.physics_client
        )[0][0]

        # Read actual joint angles from PyBullet (matching original: closed-loop feedback)
        joint_states = p.getJointStates(
            self.robot_id, self.joint_id,
            physicsClientId=self.physics_client,
        )
        joint_angs = np.array([s[0] for s in joint_states])  # radians

        # Apply action: angle increment in radians
        ds = np.deg2rad(STEP_ANGLE)
        joint_angs += action * ds

        # Clip to joint limits
        joint_angs = np.clip(joint_angs, -self._bound_ang, self._bound_ang)

        # Round to integer degrees (matching real hardware)
        joint_angs_deg = np.rad2deg(joint_angs).round()
        joint_angs = np.deg2rad(joint_angs_deg)

        # stepSimulation #1: simulate serial delay (matching original timing)
        p.stepSimulation(physicsClientId=self.physics_client)

        # Check arm contact (between step 1 and step 2, matching original)
        arm_link_indices = [1, 2, 4, 5]
        self._last_arm_contact = False
        for idx in arm_link_indices:
            if p.getContactPoints(bodyA=self.robot_id, linkIndexA=idx,
                                  physicsClientId=self.physics_client):
                self.arm_contact += 1
                self._last_arm_contact = True

        # Set motor targets (matching original: setJointMotorControlArray, no maxVelocity)
        p.setJointMotorControlArray(
            self.robot_id,
            self.joint_id,
            p.POSITION_CONTROL,
            joint_angs,
            forces=np.ones(self.n_joints) * MOTOR_FORCE,
            physicsClientId=self.physics_client,
        )

        # stepSimulation #2: data transfer delay (matching original timing)
        p.stepSimulation(physicsClientId=self.physics_client)

        # Normalize joint angles for observation and smoothness
        joint_angs_norm = joint_angs / self._bound_ang

        # Update joint history (every 2 steps)
        if self.step_counter % 2 == 0:
            normalized = self._randomize(joint_angs_norm, RANDOM_JOINT_ANGS)
            self.joint_history = np.roll(self.joint_history, -1, axis=0)
            self.joint_history[-1] = normalized

        # Update recent angles buffer (3-step history for smoothness)
        self.prev_prev_joint_angles = self.prev_joint_angles.copy()
        self.prev_joint_angles = self.joint_angles_norm.copy()
        self.joint_angles_norm = joint_angs_norm.copy()

        # Read robot state: position and orientation
        state_pos, state_ang = p.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=self.physics_client
        )

        # stepSimulation #3: serial communication delay (matching original timing)
        p.stepSimulation(physicsClientId=self.physics_client)

        euler = p.getEulerFromQuaternion(state_ang)
        state_vel = np.asarray(
            p.getBaseVelocity(self.robot_id, physicsClientId=self.physics_client)[1]
        )
        state_vel_scaled = np.clip(state_vel[0:2] * 0.1, -1, 1)
        self._state_robot = np.concatenate((state_ang, state_vel_scaled))

        current_position = p.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=self.physics_client
        )[0][0]

        # === Calculate reward (matching original structure) ===
        fac_movement = self.config.get("fac_movement", FAC_MOVEMENT)
        movement_forward = current_position - last_position

        # Smoothness penalty (normalized values)
        fac_smooth_1 = self.config.get("fac_smooth_1", FAC_SMOOTH_1)
        fac_smooth_2 = self.config.get("fac_smooth_2", FAC_SMOOTH_2)
        smooth_penalty = np.sum(
            fac_smooth_1 * (self.joint_angles_norm - self.prev_joint_angles) ** 2
            + fac_smooth_2 * (self.joint_angles_norm
                              - 2 * self.prev_joint_angles
                              + self.prev_prev_joint_angles) ** 2
        )

        # Body stability (already scaled and clipped)
        fac_stability = self.config.get("fac_stability", FAC_STABILITY)
        z_velocity = p.getBaseVelocity(
            self.robot_id, physicsClientId=self.physics_client
        )[0][2]
        body_stability = (fac_stability * (state_vel_scaled[0] ** 2 + state_vel_scaled[1] ** 2)
                          + self.config.get("fac_z_velocity", FAC_Z_VELOCITY) * z_velocity ** 2)

        # Arm contact penalty
        fac_arm_contact = self.config.get("fac_arm_contact", FAC_ARM_CONTACT)

        # Survival reward (default 0)
        reward = self.config.get("fac_survival", FAC_SURVIVAL)

        # Forward reward
        reward += fac_movement * movement_forward

        # Height bonus (default 0)
        fac_height_bonus = self.config.get("fac_height_bonus", FAC_HEIGHT_BONUS)
        if fac_height_bonus > 0:
            reward += fac_height_bonus * state_pos[2]

        # Progressive penalty factor
        reward -= self.step_counter_session / self.config.get("penalty_steps", PENALTY_STEPS) * (
            smooth_penalty + body_stability
            + fac_arm_contact * self.arm_contact
        )

        # Track posture metrics
        self._ep_heights.append(state_pos[2])
        self._ep_tilts.append(abs(euler[0]) + abs(euler[1]))
        self._ep_arm_contacts.append(self._last_arm_contact)

        # Check termination (matching original: step_counter after reward)
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

        # Build observation
        observation = np.concatenate((self._state_robot, self.joint_history.flatten()))

        return observation.astype(np.float32), float(reward), terminated, truncated, info

    @staticmethod
    def _randomize(value, percentage):
        """Randomize value within percentage boundaries (matching original)."""
        if percentage <= 0:
            return value
        percentage /= 100
        return value * (1 + percentage * (2 * np.random.rand(*value.shape) - 1))

    def render(self):
        if self.render_mode == "human":
            pass  # PyBullet GUI handles rendering
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
