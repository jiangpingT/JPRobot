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

# Reward weights
FAC_SURVIVAL = 2.0          # Per-step survival reward (encourages staying alive longer)
FAC_MOVEMENT = 1000.0
FAC_SMOOTH_1 = 1.0
FAC_SMOOTH_2 = 1.0
FAC_STABILITY = 0.1
FAC_Z_VELOCITY = 0.0
FAC_CLEARANCE = 0.0
FAC_SLIP = 0.0
FAC_ARM_CONTACT = 2.0       # Arm/elbow ground contact
FAC_ORIENTATION = 5.0       # Body tilt penalty (abs roll + abs pitch)
FAC_HEIGHT_BONUS = 40.0     # Per-step bonus for body height above MIN_HEIGHT
MIN_HEIGHT = 0.045          # Minimum acceptable body height (m)
PENALTY_STEPS = 100_000_000  # Per-env steps where penalty reaches full strength

# Domain randomization (0 = off)
RANDOM_GYRO = 0
RANDOM_JOINT_ANGS = 0
RANDOM_MASS = 0
RANDOM_FRICTION = 0

_INIT_RAD = math.radians(INIT_ANGLE)
_START_POS = [0, 0, 0.1]
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

        # Joint mapping: action index -> PyBullet joint index
        # Order: shoulder_left, elbow_left, shoulder_right, elbow_right,
        #        hip_right, knee_right, hip_left, knee_left
        self.joint_indices = [0, 4, 1, 5, 2, 6, 3, 7]

        # State variables
        self.robot_id = None
        self.physics_client = None       # kept alive across episodes
        self.step_counter = 0
        self.step_counter_session = 0
        self.joint_angles = np.zeros(self.n_joints)
        self.joint_history = np.zeros((LENGTH_JOINT_HISTORY, self.n_joints))
        self.prev_joint_angles = np.zeros(self.n_joints)
        self.prev_prev_joint_angles = np.zeros(self.n_joints)
        self.prev_position = np.zeros(3)

        # Posture tracking per episode (written to info at episode end for dashboard)
        self._ep_heights: list[float] = []
        self._ep_arm_contacts: list[bool] = []
        self._ep_tilts: list[float] = []
        self._last_arm_contact: bool = False

        # URDF model path
        self.urdf_path = os.path.join(
            os.path.dirname(__file__), "..", "models", "bittle_esp32.urdf"
        )

    def _connect(self) -> None:
        """Connect to PyBullet and load the scene. Called once per env lifetime."""
        if self.gui_mode or (self.render_mode == "human"):
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        p.loadURDF("plane.urdf", physicsClientId=self.physics_client)

        self.robot_id = p.loadURDF(
            self.urdf_path,
            _START_POS,
            _START_ORI,
            flags=p.URDF_USE_SELF_COLLISION,
            physicsClientId=self.physics_client,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Connect and load URDF only on the first reset; reuse afterwards.
        if self.physics_client is None:
            self._connect()
        else:
            # Fast reset: restore body pose and velocity without reloading URDF.
            p.resetBasePositionAndOrientation(
                self.robot_id, _START_POS, _START_ORI,
                physicsClientId=self.physics_client,
            )
            p.resetBaseVelocity(
                self.robot_id, [0, 0, 0], [0, 0, 0],
                physicsClientId=self.physics_client,
            )

        # Reset joints to standing pose (targetVelocity=0 clears joint velocity)
        for i in range(self.n_joints):
            p.resetJointState(
                self.robot_id, self.joint_indices[i], _INIT_RAD, 0,
                physicsClientId=self.physics_client,
            )

        self.joint_angles = np.full(self.n_joints, INIT_ANGLE, dtype=np.float32)
        self.joint_history = np.zeros((LENGTH_JOINT_HISTORY, self.n_joints))
        self.prev_joint_angles = self.joint_angles.copy()
        self.prev_prev_joint_angles = self.joint_angles.copy()

        # Let the robot settle
        for _ in range(50):
            p.stepSimulation(physicsClientId=self.physics_client)

        self.prev_position = np.array(
            p.getBasePositionAndOrientation(self.robot_id,
                                            physicsClientId=self.physics_client)[0]
        )
        self.step_counter = 0
        self._ep_heights.clear()
        self._ep_arm_contacts.clear()
        self._ep_tilts.clear()

        return self._get_observation(), {}

    def step(self, action):
        self.step_counter += 1
        self.step_counter_session += 1

        # Store previous angles for smoothness calculation
        self.prev_prev_joint_angles = self.prev_joint_angles.copy()
        self.prev_joint_angles = self.joint_angles.copy()

        # Apply action: angle increment
        ds = math.radians(STEP_ANGLE)
        angle_increments = action * ds
        self.joint_angles = self.joint_angles + np.degrees(angle_increments)

        # Clip to joint limits and round to integer (matching real hardware)
        self.joint_angles = np.clip(self.joint_angles, -JOINT_LIMIT, JOINT_LIMIT)
        self.joint_angles = np.round(self.joint_angles)

        # Apply domain randomization
        target_angles = self.joint_angles.copy()
        if RANDOM_JOINT_ANGS > 0:
            target_angles += np.random.uniform(
                -RANDOM_JOINT_ANGS, RANDOM_JOINT_ANGS, self.n_joints
            )

        # Set motor targets
        for i in range(self.n_joints):
            joint_idx = self.joint_indices[i]
            target_rad = math.radians(target_angles[i])
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=target_rad,
                force=MOTOR_FORCE,
                maxVelocity=MOTOR_VELOCITY,
                physicsClientId=self.physics_client,
            )

        # Step simulation multiple times (simulates serial communication delay)
        for _ in range(SIMULATION_STEPS):
            p.stepSimulation(physicsClientId=self.physics_client)

        # Query physics state once — shared by reward and observation
        position, orientation = p.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=self.physics_client
        )
        velocity = p.getBaseVelocity(self.robot_id, physicsClientId=self.physics_client)
        euler = p.getEulerFromQuaternion(orientation)

        # Update joint history (every 2 steps)
        if self.step_counter % 2 == 0:
            normalized = self.joint_angles / JOINT_LIMIT
            self.joint_history = np.roll(self.joint_history, -1, axis=0)
            self.joint_history[-1] = normalized

        # Calculate reward
        reward = self._calculate_reward(position, euler, velocity)

        # Track posture metrics (uses _last_arm_contact set by _calculate_reward)
        self._ep_heights.append(position[2])
        self._ep_tilts.append(abs(euler[0]) + abs(euler[1]))
        self._ep_arm_contacts.append(self._last_arm_contact)

        # Check termination (50° tilt threshold, was 40°/0.7 rad)
        terminated = abs(euler[0]) > 0.873 or abs(euler[1]) > 0.873
        truncated = self.step_counter >= self.episode_length

        if terminated:
            reward = 0

        self.prev_position = np.array(position)

        info = {}
        if (terminated or truncated) and self._ep_heights:
            info['posture'] = {
                'avg_height': float(np.mean(self._ep_heights)),
                'arm_contact_rate': float(np.mean(self._ep_arm_contacts)),
                'avg_tilt_deg': float(np.degrees(np.mean(self._ep_tilts))),
            }

        return self._get_observation(orientation, velocity), float(reward), terminated, truncated, info

    def _get_observation(self, orientation=None, velocity=None):
        """Build observation vector (246 dimensions).

        Accepts pre-fetched orientation/velocity from step() to avoid
        a redundant PyBullet query.
        """
        if orientation is None:
            _, orientation = p.getBasePositionAndOrientation(
                self.robot_id, physicsClientId=self.physics_client
            )
            velocity = p.getBaseVelocity(self.robot_id, physicsClientId=self.physics_client)

        # Body orientation quaternion (4 values)
        quat = np.array(orientation, dtype=np.float32)

        # Angular velocity roll/pitch (2 values, clipped to [-1, 1])
        ang_vel = np.array([
            np.clip(velocity[1][0] * 0.1, -1, 1),
            np.clip(velocity[1][1] * 0.1, -1, 1),
        ], dtype=np.float32)

        if RANDOM_GYRO > 0:
            ang_vel += np.random.uniform(-RANDOM_GYRO, RANDOM_GYRO, 2)
            ang_vel = np.clip(ang_vel, -1, 1)

        # Joint history (flattened: 30 x 8 = 240 values)
        history_flat = self.joint_history.flatten().astype(np.float32)

        return np.concatenate([quat, ang_vel, history_flat])

    def _calculate_reward(self, position, euler, velocity):
        """Calculate reward with posture-gated forward reward.

        Key design: forward movement is only rewarded when the robot is upright.
        This prevents the "rolling exploit" where tumbling forward earns reward.

        All reward weights are read from self.config with fallback to module-level
        defaults, so curriculum stages can override any weight without code changes.
        """
        position = np.array(position)

        # === Posture quality: 1.0 when upright, 0 when tilted > ~35° ===
        tilt = abs(euler[0]) + abs(euler[1])
        posture_factor = max(0.0, 1.0 - tilt / 0.6)

        # === Survival reward (per step, always positive, encourages staying alive) ===
        reward = self.config.get("fac_survival", FAC_SURVIVAL)

        # === Forward reward (gated by posture) ===
        fac_movement = self.config.get("fac_movement", FAC_MOVEMENT)
        movement_forward = position[0] - self.prev_position[0]
        reward += fac_movement * movement_forward * posture_factor

        # === Height bonus (always active, encourages standing tall) ===
        fac_height_bonus = self.config.get("fac_height_bonus", FAC_HEIGHT_BONUS)
        reward += fac_height_bonus * position[2]

        # === Penalty factor: support hard override from config ===
        pf_override = self.config.get("penalty_factor", None)
        if pf_override is not None:
            penalty_factor = pf_override
        else:
            penalty_factor = min(1.0, self.step_counter_session / PENALTY_STEPS)

        # Smoothness penalty (1st order: angle velocity)
        angle_diff = self.joint_angles - self.prev_joint_angles
        smooth_1 = self.config.get("fac_smooth_1", FAC_SMOOTH_1) * np.sum(np.abs(angle_diff)) / self.n_joints

        # Smoothness penalty (2nd order: angle acceleration)
        angle_accel = (self.joint_angles - 2 * self.prev_joint_angles
                       + self.prev_prev_joint_angles)
        smooth_2 = self.config.get("fac_smooth_2", FAC_SMOOTH_2) * np.sum(np.abs(angle_accel)) / self.n_joints

        smooth_penalty = smooth_1 + smooth_2

        # Body stability penalty (angular velocity)
        ang_vel = velocity[1]
        stability_penalty = self.config.get("fac_stability", FAC_STABILITY) * (abs(ang_vel[0]) + abs(ang_vel[1]))

        # Z-velocity penalty
        z_vel_penalty = self.config.get("fac_z_velocity", FAC_Z_VELOCITY) * abs(velocity[0][2])

        # Orientation penalty (absolute tilt angle)
        orientation_penalty = self.config.get("fac_orientation", FAC_ORIENTATION) * tilt

        # Arm contact penalty (check if arm/elbow links touch ground)
        fac_arm_contact = self.config.get("fac_arm_contact", FAC_ARM_CONTACT)
        arm_contact_penalty = 0.0
        contact_points = p.getContactPoints(
            self.robot_id, physicsClientId=self.physics_client
        )
        arm_link_indices = {0, 1, 4, 5}  # Upper/lower arm links
        self._last_arm_contact = False
        for contact in contact_points:
            if contact[3] in arm_link_indices or contact[4] in arm_link_indices:
                arm_contact_penalty = fac_arm_contact
                self._last_arm_contact = True
                break

        # Apply progressive penalty
        total_penalty = penalty_factor * (
            smooth_penalty + stability_penalty + z_vel_penalty
            + orientation_penalty + arm_contact_penalty
        )

        reward -= total_penalty
        return reward

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
