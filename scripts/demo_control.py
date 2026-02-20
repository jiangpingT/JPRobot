#!/usr/bin/env python3
"""Demo: Basic BittleX control via serial.

Run this after connecting BittleX via USB.
Usage: python scripts/demo_control.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jprobot.robot import BittleCommander


def main():
    print("=== JPRobot Basic Control Demo ===\n")

    with BittleCommander() as robot:
        print("Connected! Running demo sequence...\n")

        # Demo sequence
        sequence = [
            ("kbalance", 2),            # Stand up
            ("khi", 2),                 # Say hi
            ("kwkF", 3),               # Walk forward
            ("ksit", 2),               # Sit down
            ("knd", 1),                # Nod
            ("kpu", 3),                # Push up
            ("kbf", 3),                # Back flip!
            ("ksit", 2),               # Sit back down
        ]

        robot.execute_sequence(sequence)

        # Direct joint control demo
        print("\nHead movement demo...")
        robot.move_head(pan=-50, tilt=0)     # Look left
        import time; time.sleep(1)
        robot.move_head(pan=50, tilt=0)      # Look right
        time.sleep(1)
        robot.move_head(pan=0, tilt=30)      # Look up
        time.sleep(1)
        robot.move_head(pan=0, tilt=0)       # Center
        time.sleep(1)

        # Rest
        robot.rest()
        print("\nDemo complete!")


if __name__ == "__main__":
    main()
