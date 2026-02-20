"""High-level commander for Petoi BittleX.

Provides a clean Python API over the serial protocol.
Based on OpenCat serialMaster protocol documentation.

Joint index mapping (BittleX):
    0: Head pan (yaw)
    1: Head tilt (pitch)
    2-7: Reserved / extra DOF
    8: Left front shoulder
    9: Right front shoulder
    10: Right back hip
    11: Left back hip
    12: Left front elbow/knee
    13: Right front elbow/knee
    14: Right back knee
    15: Left back knee
"""

import time
from typing import Optional

from .connection import SerialConnection
from .skills import SkillRegistry, Skill


class BittleCommander:
    """High-level control API for BittleX robot."""

    # Joint name to index mapping
    JOINTS = {
        "head_pan": 0,
        "head_tilt": 1,
        "left_front_shoulder": 8,
        "right_front_shoulder": 9,
        "right_back_hip": 10,
        "left_back_hip": 11,
        "left_front_knee": 12,
        "right_front_knee": 13,
        "right_back_knee": 14,
        "left_back_knee": 15,
    }

    def __init__(self, connection: Optional[SerialConnection] = None):
        self.conn = connection or SerialConnection()
        self.skills = SkillRegistry()
        self._current_skill: Optional[str] = None

    def connect(self) -> bool:
        """Connect to robot."""
        return self.conn.connect()

    def disconnect(self):
        """Disconnect from robot."""
        self.conn.disconnect()

    # === Skill Execution ===

    def perform(self, skill_code: str, delay: float = 0) -> str:
        """Execute a named skill.

        Args:
            skill_code: Skill code (e.g., "sit", "wkF", "bf").
            delay: Seconds to wait after executing.

        Returns:
            Robot response string.
        """
        response = self.conn.send_ascii(f"k{skill_code}")
        self._current_skill = skill_code
        if delay > 0:
            time.sleep(delay)
        return response

    # === Common Actions (convenience methods) ===

    def sit(self) -> str:
        return self.perform("sit")

    def stand(self) -> str:
        return self.perform("balance")

    def stand_up(self) -> str:
        return self.perform("up")

    def rest(self) -> str:
        """Rest and shut down servos."""
        return self.perform("rest")

    def walk_forward(self) -> str:
        return self.perform("wkF")

    def walk_left(self) -> str:
        return self.perform("wkL")

    def walk_backward(self) -> str:
        return self.perform("bk")

    def trot_forward(self) -> str:
        return self.perform("trF")

    def crawl_forward(self) -> str:
        return self.perform("crF")

    def jump(self) -> str:
        return self.perform("jmp")

    def back_flip(self) -> str:
        return self.perform("bf")

    def front_flip(self) -> str:
        return self.perform("ff")

    def push_up(self) -> str:
        return self.perform("pu")

    def hand_shake(self) -> str:
        return self.perform("hsk")

    def say_hi(self) -> str:
        return self.perform("hi")

    def play_dead(self) -> str:
        return self.perform("pd")

    def moon_walk(self) -> str:
        return self.perform("mw")

    def nod(self) -> str:
        return self.perform("nd")

    def roll_over(self) -> str:
        return self.perform("rl")

    # === Joint Control ===

    def move_joint(self, joint: int | str, angle: int, delay: float = 0) -> str:
        """Move a single joint to target angle.

        Args:
            joint: Joint index (0-15) or name string.
            angle: Target angle in degrees.
            delay: Seconds to wait after moving.
        """
        if isinstance(joint, str):
            joint = self.JOINTS[joint]
        response = self.conn.send_ascii(f"m{joint} {angle}")
        if delay > 0:
            time.sleep(delay)
        return response

    def move_joints_simultaneous(self, joint_angles: dict[int | str, int],
                                  delay: float = 0) -> str:
        """Move multiple joints simultaneously.

        Args:
            joint_angles: {joint_index_or_name: angle_degrees}
        """
        resolved = {}
        for k, v in joint_angles.items():
            idx = self.JOINTS[k] if isinstance(k, str) else k
            resolved[idx] = v
        response = self.conn.send_indexed_angles(resolved)
        if delay > 0:
            time.sleep(delay)
        return response

    def move_joints_sequential(self, joint_angles: list[tuple[int, int]],
                                delay: float = 0) -> str:
        """Move joints one by one in sequence.

        Args:
            joint_angles: [(joint_index, angle), ...]
        """
        parts = []
        for idx, ang in joint_angles:
            parts.extend([str(idx), str(ang)])
        cmd = "m " + " ".join(["m"] + parts)
        response = self.conn.send_ascii(cmd)
        if delay > 0:
            time.sleep(delay)
        return response

    def set_all_joints(self, angles: list[int], delay: float = 0) -> str:
        """Set all 16 joint angles at once using binary protocol.

        Args:
            angles: List of 16 int8 angles.
        """
        response = self.conn.send_joint_angles(angles)
        if delay > 0:
            time.sleep(delay)
        return response

    def move_head(self, pan: int = 0, tilt: int = 0) -> str:
        """Move the head.

        Args:
            pan: Horizontal angle (-70 to 70, 0=center).
            tilt: Vertical angle (-30 to 80, 0=level).
        """
        return self.move_joints_simultaneous({0: pan, 1: tilt})

    # === Gyroscope ===

    def gyro_on(self) -> str:
        """Enable IMU-based self-balancing."""
        return self.conn.send_ascii("G")

    def gyro_off(self) -> str:
        """Disable IMU-based self-balancing."""
        return self.conn.send_ascii("g")

    # === Buzzer ===

    def beep(self, tone: int = 20, duration: int = 50) -> str:
        """Play a single beep.

        Args:
            tone: Musical tone value (0-50).
            duration: Duration (0-255).
        """
        return self.conn.send_ascii(f"b{tone} {duration}")

    def play_melody(self, notes: list[tuple[int, int]]) -> str:
        """Play a melody sequence.

        Args:
            notes: [(tone, duration), ...]
        """
        parts = ["b"]
        for tone, dur in notes:
            parts.extend([str(tone), str(dur)])
        return self.conn.send_ascii(" ".join(parts))

    # === System ===

    def query_model(self) -> str:
        """Query robot model and version."""
        return self.conn.send_ascii("?")

    def query_joints(self) -> str:
        """Query current joint angles."""
        return self.conn.send_ascii("j")

    def pause(self) -> str:
        """Pause current action."""
        return self.conn.send_ascii("p")

    def calibrate(self) -> str:
        """Enter calibration mode."""
        return self.conn.send_ascii("c")

    def save_calibration(self) -> str:
        """Save calibration values."""
        return self.conn.send_ascii("s")

    # === Task Scheduling ===

    def execute_sequence(self, tasks: list[tuple]) -> list[str]:
        """Execute a sequence of commands with delays.

        Based on OpenCat serialMaster's testSchedule format.

        Args:
            tasks: List of (command, [params], delay) tuples.
                   Examples:
                   [("kbalance", 2),
                    ("m", [0, -20], 1.5),
                    ("i", [8, 50, 9, 50], 3)]

        Returns:
            List of responses.
        """
        responses = []
        for task in tasks:
            if len(task) == 2:
                cmd, delay = task
                params = None
            else:
                cmd, params, delay = task

            if cmd.startswith("k"):
                resp = self.perform(cmd[1:])
            elif cmd in ("m", "i", "M", "I") and params:
                parts = " ".join(str(p) for p in params)
                resp = self.conn.send_ascii(f"{cmd}{parts}")
            elif cmd == "d":
                resp = self.rest()
            elif cmd == "b" and params:
                resp = self.beep(*params)
            else:
                resp = self.conn.send_ascii(cmd)

            responses.append(resp)
            if delay > 0:
                time.sleep(delay)

        return responses

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()
