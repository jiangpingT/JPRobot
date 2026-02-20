#!/usr/bin/env python3
"""Demo: LLM-powered natural language robot control.

Talk to your BittleX in natural language. The AI brain translates
your commands into robot actions using Function Calling.

Usage:
    # With real robot connected:
    python scripts/demo_agent.py

    # Simulation mode (no robot needed):
    python scripts/demo_agent.py --sim
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from jprobot.robot import BittleCommander
from jprobot.agent import RobotBrain

def main():
    default_model = os.getenv("JPROBOT_MODEL", "claude-sonnet-4-6")

    parser = argparse.ArgumentParser(description="JPRobot AI Agent Demo")
    parser.add_argument("--sim", action="store_true",
                        help="Run in simulation mode (no real robot)")
    parser.add_argument("--model", type=str, default=default_model,
                        help="LLM model to use")
    args = parser.parse_args()

    print("=== JPRobot AI Agent Demo ===")
    print(f"Model: {args.model}")
    print(f"Mode: {'Simulation' if args.sim else 'Real Robot'}")
    print("Type 'quit' to exit, 'reset' to clear history.\n")

    # Setup commander (None for simulation)
    commander = None
    if not args.sim:
        commander = BittleCommander()
        if not commander.connect():
            print("Failed to connect to robot. Running in simulation mode.")
            commander = None

    # Setup AI brain
    brain = RobotBrain(
        commander=commander,
        model=args.model,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    # Chat loop
    while True:
        try:
            user_input = input("\n你: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if user_input.lower() == "reset":
            brain.reset()
            print("[History cleared]")
            continue

        try:
            reply = brain.chat(user_input)
            print(f"\nJPRobot: {reply}")
        except Exception as e:
            print(f"\n[Error] {e}")

    # Cleanup
    if commander:
        commander.rest()
        commander.disconnect()
    print("\nBye!")


if __name__ == "__main__":
    main()
