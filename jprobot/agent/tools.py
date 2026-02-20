"""OpenAI Function Calling tool definitions for robot control.

Replaces the regex-based approach in PetoiBittleChatGPT with structured
Function Calling for reliable intent-to-command translation.
"""

# Tool definitions for OpenAI Function Calling API
ROBOT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "perform_skill",
            "description": "Execute a robot skill/action by its code name. "
                           "Use this for predefined motions like walking, sitting, flipping, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_code": {
                        "type": "string",
                        "description": "The skill code to execute. Common codes: "
                                       "sit (sit down), balance (stand), wkF (walk forward), "
                                       "bk (walk backward), trF (trot), bf (back flip), "
                                       "ff (front flip), pu (push up), hsk (hand shake), "
                                       "hi (say hi), pd (play dead), mw (moon walk), "
                                       "jmp (jump), rl (roll over), nd (nod), "
                                       "ang (angry), gdb (good boy), rest (rest/sleep)",
                    },
                    "delay": {
                        "type": "number",
                        "description": "Seconds to wait after executing (default 1.0)",
                        "default": 1.0,
                    },
                },
                "required": ["skill_code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_head",
            "description": "Move the robot's head to look in a direction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pan": {
                        "type": "integer",
                        "description": "Horizontal angle: negative=left, 0=center, positive=right. Range: -70 to 70.",
                        "default": 0,
                    },
                    "tilt": {
                        "type": "integer",
                        "description": "Vertical angle: negative=down, 0=level, positive=up. Range: -30 to 80.",
                        "default": 0,
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_joint",
            "description": "Move a specific joint to a target angle. "
                           "Joint indices: 0=head_pan, 1=head_tilt, "
                           "8=left_front_shoulder, 9=right_front_shoulder, "
                           "10=right_back_hip, 11=left_back_hip, "
                           "12=left_front_knee, 13=right_front_knee, "
                           "14=right_back_knee, 15=left_back_knee",
            "parameters": {
                "type": "object",
                "properties": {
                    "joint": {
                        "type": "integer",
                        "description": "Joint index (0-15)",
                    },
                    "angle": {
                        "type": "integer",
                        "description": "Target angle in degrees (-125 to 125)",
                    },
                },
                "required": ["joint", "angle"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_sequence",
            "description": "Execute a sequence of actions with delays between them. "
                           "Use this for choreographed routines or multi-step tasks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "actions": {
                        "type": "array",
                        "description": "List of actions to perform in order",
                        "items": {
                            "type": "object",
                            "properties": {
                                "skill_code": {
                                    "type": "string",
                                    "description": "Skill code to execute",
                                },
                                "delay": {
                                    "type": "number",
                                    "description": "Seconds to wait after this action",
                                    "default": 1.0,
                                },
                            },
                            "required": ["skill_code"],
                        },
                    },
                },
                "required": ["actions"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "beep",
            "description": "Make the robot beep/play a sound.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tone": {
                        "type": "integer",
                        "description": "Musical tone value (0-50)",
                        "default": 20,
                    },
                    "duration": {
                        "type": "integer",
                        "description": "Duration (0-255)",
                        "default": 50,
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_status",
            "description": "Query the robot's current status (model, joint angles, etc.)",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_type": {
                        "type": "string",
                        "enum": ["model", "joints"],
                        "description": "'model' to get robot model info, 'joints' to get current joint angles",
                    },
                },
                "required": ["query_type"],
            },
        },
    },
]
