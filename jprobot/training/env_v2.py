"""Gymnasium environment for BittleX reinforcement learning — v2.

Changes from env.py (v1):
- obs_dim 248 → 254: adds lin_vel_xy (2) + feet_contact_state (4)
- FAC_ARM_CONTACT: 0.01 → 0.03 (stronger crawling penalty)
- New paw contact detection using fixed-joint link indices [3, 6, 9, 12]

State space (254 dimensions):
    - Body orientation quaternion (4)
    - Angular velocity roll/pitch, scaled (2)
    - Linear velocity xy, scaled (2)       ← NEW
    - Joint angle history: 30 timesteps x 8 joints (240)
    - Target direction vector dx/dy (2)
    - Feet contact state: [FL, FR, BR, BL] binary (4)  ← NEW

Why lin_vel_xy?
    The policy can see HOW FAST it is currently moving, not just which direction.
    This closes the velocity feedback loop: if the robot is commanded to go forward
    but is actually sliding sideways, the policy sees the discrepancy and can correct.
    Without this, the policy has to infer velocity from joint history alone.

Why feet_contact_state?
    Quadruped gaits (trot, walk, pace) are defined by which feet are in contact with
    the ground at each moment. By providing this directly, the policy can learn to
    coordinate legs in a specific gait pattern rather than discovering it from scratch.
    The 4 binary bits encode: [front-left, front-right, back-right, back-left] contact.

Paw link indices (from URDF joint analysis):
    Joint 3 (fixed): paw_front_left  → link index 3 = left_front_paw
    Joint 6 (fixed): paw_front_right → link index 6 = right_front_paw
    Joint 9 (fixed): paw_back_right  → link index 9 = right_back_paw
    Joint 12 (fixed): paw_back_left  → link index 12 = left_back_paw

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
STEP_ANGLE = 11              # Max angle change per step (degrees)
JOINT_LIMIT = 110            # Max joint angle (degrees)
MOTOR_FORCE = 0.2            # Max motor torque (N·m)
MOTOR_VELOCITY = 10 * math.pi
INIT_ANGLE = 50
LENGTH_JOINT_HISTORY = 30
SIMULATION_STEPS = 3

# Reward weights
FAC_SURVIVAL = 0.0
FAC_MOVEMENT = 1000.0
FAC_SMOOTH_1 = 1.0
FAC_SMOOTH_2 = 1.0
FAC_STABILITY = 0.1
FAC_Z_VELOCITY = 0.0
FAC_CLEARANCE = 0.0
FAC_SLIP = 0.0
FAC_ARM_CONTACT = 0.03      # v2: 3× stronger crawling penalty than v1 (0.01)
FAC_ORIENTATION = 0.0
FAC_HEIGHT_BONUS = 0.0
MIN_HEIGHT = 0.045
PENALTY_STEPS = 2_000_000

# Linear velocity scaling
# Robot moves ~0.005 m/env_step at normal gait, max ~0.4 m/s
# Multiplying by 2.5 maps 0.4 m/s → 1.0 (full obs range)
LIN_VEL_SCALE = 2.5

# Domain randomization (0 = off)
RANDOM_GYRO = 0
RANDOM_JOINT_ANGS = 0
RANDOM_MASS = 0
RANDOM_FRICTION = 0

_INIT_RAD = math.radians(INIT_ANGLE)
_START_POS = [0, 0, 0.08]
_START_ORI = [0, 0, 0, 1]

_DIRECTION_MAP = {
    "forward":  np.array([1.0,  0.0], dtype=np.float32),
    "backward": np.array([-1.0, 0.0], dtype=np.float32),
    "left":     np.array([0.0,  1.0], dtype=np.float32),
    "right":    np.array([0.0, -1.0], dtype=np.float32),
}
_DIRECTION_NAMES = list(_DIRECTION_MAP.keys())
_DEFAULT_DIRECTION_PROBS = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
_CURRICULUM_DEFAULT_DIRECTION_PROBS = np.array([0.5, 0.2, 0.15, 0.15], dtype=np.float32)

# Paw link indices (fixed-joint links that are the actual foot tips)
# These are NOT in self.joint_id (which only contains revolute joints)
# but PyBullet still tracks them as rigid body links for contact detection.
_PAW_LINK_INDICES = [3, 6, 9, 12]  # FL, FR, BR, BL


class BittleGymEnvV2(gym.Env):
    """BittleX env v2 — adds lin_vel_xy + feet_contact_state to observation."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None, config=None):
        super().__init__()

        self.render_mode = render_mode
        self.config = config or {}

        self.gui_mode = self.config.get("gui_mode", GUI_MODE)
        self.episode_length = self.config.get("episode_length", EPISODE_LENGTH)

        self.n_joints = 8
        # obs layout: body(6) + lin_vel_xy(2) + joint_history(240) + target_dir(2) + feet(4) = 254
        obs_dim = 6 + 2 + LENGTH_JOINT_HISTORY * self.n_joints + 2 + 4  # 254

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
        self._state_robot = np.zeros(6)         # quat(4) + ang_vel_scaled(2)
        self._lin_vel_xy = np.zeros(2, dtype=np.float32)   # NEW: linear vel xy
        self._feet_contact = np.zeros(4, dtype=np.float32)  # NEW: [FL, FR, BR, BL]
        self.target_dir_xy = np.array([1.0, 0.0], dtype=np.float32)
        self.target_name = "forward"

        self.arm_contact = 0
        self._ep_heights: list[float] = []
        self._ep_arm_contacts: list[bool] = []
        self._ep_tilts: list[float] = []
        self._ep_target_progress: list[float] = []
        self._last_arm_contact: bool = False

        self.urdf_path = os.path.join(
            os.path.dirname(__file__), "..", "models", "bittle_esp32.urdf"
        )

    def _normalize_target_dir(self, target):
        arr = np.asarray(target, dtype=np.float32).reshape(-1)
        if arr.shape[0] != 2:
            raise ValueError(f"target_dir must have 2 elements, got {arr.shape[0]}")
        norm = float(np.linalg.norm(arr))
        if norm < 1e-8:
            raise ValueError("target_dir norm must be > 0")
        return arr / norm

    def _resolve_target_direction(self, options=None):
        options = options or {}

        if "target_dir" in options:
            target = self._normalize_target_dir(options["target_dir"])
            return "custom", target

        if "target_name" in options:
            name = str(options["target_name"]).lower()
            if name not in _DIRECTION_MAP:
                raise ValueError(f"Unknown target_name: {name}")
            return name, _DIRECTION_MAP[name].copy()

        if "target_dir" in self.config:
            target = self._normalize_target_dir(self.config["target_dir"])
            return "custom", target

        mode = str(self.config.get("direction_mode", "random")).lower()
        if mode == "fixed":
            fixed_name = str(self.config.get("fixed_direction", "forward")).lower()
            if fixed_name not in _DIRECTION_MAP:
                fixed_name = "forward"
            return fixed_name, _DIRECTION_MAP[fixed_name].copy()

        probs_cfg = self.config.get("direction_probs", None)
        if probs_cfg is not None:
            probs = np.asarray(probs_cfg, dtype=np.float32).reshape(-1)
        elif mode == "curriculum":
            probs = _CURRICULUM_DEFAULT_DIRECTION_PROBS.copy()
        else:
            probs = _DEFAULT_DIRECTION_PROBS.copy()

        if probs.shape[0] != len(_DIRECTION_NAMES) or float(np.sum(probs)) <= 0:
            probs = _DEFAULT_DIRECTION_PROBS.copy()
        probs = probs / np.sum(probs)

        idx = int(self.np_random.choice(len(_DIRECTION_NAMES), p=probs))
        name = _DIRECTION_NAMES[idx]
        return name, _DIRECTION_MAP[name].copy()

    def _get_lin_vel_xy(self):
        """Read world-frame linear velocity, scale and clip to [-1, 1]."""
        lin_vel_world = p.getBaseVelocity(
            self.robot_id, physicsClientId=self.physics_client
        )[0]  # [0] = linear, [1] = angular
        return np.clip(
            np.array(lin_vel_world[:2], dtype=np.float32) * LIN_VEL_SCALE,
            -1.0, 1.0
        )

    def _get_feet_contact(self):
        """Return binary contact state for the 4 paw links.

        PyBullet getContactPoints returns a list of contact tuples when the
        specified link is touching any other body. Empty list = no contact.
        We check each paw link independently.

        Contact forces are tiny (<<1N) when the foot is barely grazing.
        PyBullet reports ANY contact, which is fine for binary gait state.
        """
        contact = np.zeros(4, dtype=np.float32)
        for i, link_idx in enumerate(_PAW_LINK_INDICES):
            pts = p.getContactPoints(
                bodyA=self.robot_id,
                linkIndexA=link_idx,
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

        # On reset, velocity is zero (just placed), contact not yet checked
        self._lin_vel_xy = np.zeros(2, dtype=np.float32)
        self._feet_contact = np.zeros(4, dtype=np.float32)

        self.step_counter = 0
        self.arm_contact = 0
        self._ep_heights.clear()
        self._ep_arm_contacts.clear()
        self._ep_tilts.clear()
        self._ep_target_progress.clear()
        self.target_name, self.target_dir_xy = self._resolve_target_direction(options)

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1,
                                   physicsClientId=self.physics_client)

        observation = np.concatenate((
            self._state_robot,      # 6: quat + ang_vel_rp
            self._lin_vel_xy,       # 2: lin_vel_xy (NEW)
            self.joint_history.flatten(),  # 240
            self.target_dir_xy,     # 2
            self._feet_contact,     # 4: feet contact (NEW)
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

        # Arm contact check (between sim step 1 and 2)
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
        state_vel = np.asarray(
            p.getBaseVelocity(self.robot_id, physicsClientId=self.physics_client)[1]
        )
        state_vel_scaled = np.clip(state_vel[0:2] * 0.1, -1, 1)
        self._state_robot = np.concatenate((state_ang, state_vel_scaled))

        current_position = p.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=self.physics_client
        )[0]

        # Read new obs components after all 3 sim steps
        self._lin_vel_xy = self._get_lin_vel_xy()
        self._feet_contact = self._get_feet_contact()

        # === Reward ===
        fac_movement = self.config.get("fac_movement", FAC_MOVEMENT)
        delta_xy = np.array(
            [current_position[0] - last_position[0],
             current_position[1] - last_position[1]],
            dtype=np.float32,
        )
        movement_along_target = float(np.dot(delta_xy, self.target_dir_xy))

        fac_smooth_1 = self.config.get("fac_smooth_1", FAC_SMOOTH_1)
        fac_smooth_2 = self.config.get("fac_smooth_2", FAC_SMOOTH_2)
        smooth_penalty = np.sum(
            fac_smooth_1 * (self.joint_angles_norm - self.prev_joint_angles) ** 2
            + fac_smooth_2 * (self.joint_angles_norm
                              - 2 * self.prev_joint_angles
                              + self.prev_prev_joint_angles) ** 2
        )

        fac_stability = self.config.get("fac_stability", FAC_STABILITY)
        z_velocity = p.getBaseVelocity(
            self.robot_id, physicsClientId=self.physics_client
        )[0][2]
        body_stability = (fac_stability * (state_vel_scaled[0] ** 2 + state_vel_scaled[1] ** 2)
                          + self.config.get("fac_z_velocity", FAC_Z_VELOCITY) * z_velocity ** 2)

        fac_arm_contact = self.config.get("fac_arm_contact", FAC_ARM_CONTACT)

        reward = self.config.get("fac_survival", FAC_SURVIVAL)
        reward += fac_movement * movement_along_target

        fac_height_bonus = self.config.get("fac_height_bonus", FAC_HEIGHT_BONUS)
        if fac_height_bonus > 0:
            reward += fac_height_bonus * state_pos[2]

        reward -= self.step_counter_session / self.config.get("penalty_steps", PENALTY_STEPS) * (
            smooth_penalty + body_stability + fac_arm_contact * self.arm_contact
        )

        self._ep_heights.append(state_pos[2])
        self._ep_tilts.append(abs(euler[0]) + abs(euler[1]))
        self._ep_arm_contacts.append(self._last_arm_contact)
        self._ep_target_progress.append(movement_along_target)

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
            info['direction'] = {
                'target_name': self.target_name,
                'target_dir': self.target_dir_xy.tolist(),
                'avg_target_progress': float(np.mean(self._ep_target_progress))
                if self._ep_target_progress else 0.0,
                'ep_len': int(self.step_counter),
            }

        if self.config.get("only_positive_rewards", False):
            reward = max(0.0, reward)

        observation = np.concatenate((
            self._state_robot,      # 6
            self._lin_vel_xy,       # 2 (NEW)
            self.joint_history.flatten(),  # 240
            self.target_dir_xy,     # 2
            self._feet_contact,     # 4 (NEW)
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
