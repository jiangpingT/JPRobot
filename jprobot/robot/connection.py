"""Serial/Bluetooth connection manager for Petoi BittleX.

Based on OpenCat serial protocol:
- ASCII mode (lowercase tokens): command + params + '\\n'
- Binary mode (uppercase tokens): command + bytes + '~'
- Baud rate: 115200
"""

import time
import struct
import threading
from typing import Optional

import serial
import serial.tools.list_ports


class SerialConnection:
    """Manage serial connection to BittleX robot."""

    BAUD_RATE = 115200
    TIMEOUT = 1.0
    # Known USB-serial chip vendors for Petoi boards
    KNOWN_VENDORS = ["1A86", "10C4", "0403", "2341"]

    def __init__(self, port: Optional[str] = None, baud_rate: int = BAUD_RATE):
        self.port = port
        self.baud_rate = baud_rate
        self.serial: Optional[serial.Serial] = None
        self._lock = threading.Lock()
        self._connected = False

    @staticmethod
    def list_ports() -> list[dict]:
        """List all available serial ports with device info."""
        ports = []
        for p in serial.tools.list_ports.comports():
            ports.append({
                "device": p.device,
                "description": p.description,
                "hwid": p.hwid,
                "vid": f"{p.vid:04X}" if p.vid else None,
                "pid": f"{p.pid:04X}" if p.pid else None,
            })
        return ports

    @staticmethod
    def auto_detect() -> Optional[str]:
        """Auto-detect BittleX serial port by known USB vendor IDs."""
        for p in serial.tools.list_ports.comports():
            if p.vid:
                vid = f"{p.vid:04X}"
                if vid in SerialConnection.KNOWN_VENDORS:
                    return p.device
        # Fallback: return first available port
        ports = list(serial.tools.list_ports.comports())
        if ports:
            return ports[0].device
        return None

    def connect(self) -> bool:
        """Connect to BittleX. Auto-detects port if not specified."""
        if self._connected:
            return True

        port = self.port or self.auto_detect()
        if not port:
            print("[JPRobot] No serial port found. Is BittleX connected?")
            return False

        try:
            self.serial = serial.Serial(
                port=port,
                baudrate=self.baud_rate,
                timeout=self.TIMEOUT,
            )
            self.port = port
            self._connected = True
            # Wait for Arduino bootloader
            time.sleep(2)
            # Drain any startup messages
            self._drain()
            print(f"[JPRobot] Connected to {port} @ {self.baud_rate}")
            return True
        except serial.SerialException as e:
            print(f"[JPRobot] Connection failed: {e}")
            return False

    def disconnect(self):
        """Close serial connection."""
        if self.serial and self.serial.is_open:
            self.serial.close()
        self._connected = False
        print("[JPRobot] Disconnected")

    @property
    def is_connected(self) -> bool:
        return self._connected and self.serial is not None and self.serial.is_open

    def send_ascii(self, command: str) -> str:
        """Send ASCII-mode command (lowercase token).

        Format: <token><params>\\n
        Examples:
            send_ascii("kbalance")   -> execute 'balance' skill
            send_ascii("ksit")       -> sit down
            send_ascii("m0 45")      -> rotate joint 0 to 45 degrees
            send_ascii("i0 70 8 -20") -> simultaneous joint control
        """
        with self._lock:
            if not self.is_connected:
                return ""
            data = (command.strip() + "\n").encode("utf-8")
            self.serial.write(data)
            time.sleep(0.1)
            return self._read_response()

    def send_binary(self, token: str, data: bytes) -> str:
        """Send binary-mode command (uppercase token).

        Format: <Token><bytes>~
        Examples:
            send_binary("L", bytes([20,0,0,...]))  -> set all 16 joints
            send_binary("K", skill_data_bytes)     -> upload custom skill
        """
        with self._lock:
            if not self.is_connected:
                return ""
            payload = token.encode("utf-8") + data + b"~"
            self.serial.write(payload)
            time.sleep(0.1)
            return self._read_response()

    def send_joint_angles(self, angles: list[int]) -> str:
        """Send all 16 joint angles using binary L command.

        Args:
            angles: list of 16 int8 values (-128 to 127), one per joint.
        """
        if len(angles) != 16:
            raise ValueError(f"Expected 16 angles, got {len(angles)}")
        data = struct.pack("16b", *angles)
        return self.send_binary("L", data)

    def send_indexed_angles(self, joint_angles: dict[int, int]) -> str:
        """Send specific joint angles using ASCII i command.

        Args:
            joint_angles: dict mapping joint_index -> angle_degrees.
        """
        parts = " ".join(f"{idx} {ang}" for idx, ang in joint_angles.items())
        return self.send_ascii(f"i{parts}")

    def _read_response(self, timeout: float = 0.5) -> str:
        """Read response from robot until timeout."""
        end_time = time.time() + timeout
        response = b""
        while time.time() < end_time:
            if self.serial.in_waiting:
                response += self.serial.read(self.serial.in_waiting)
                time.sleep(0.05)
            else:
                time.sleep(0.01)
        return response.decode("utf-8", errors="ignore").strip()

    def _drain(self):
        """Drain any pending data in the serial buffer."""
        if self.serial and self.serial.in_waiting:
            self.serial.read(self.serial.in_waiting)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()

    def __del__(self):
        self.disconnect()
