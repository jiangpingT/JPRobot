"""LLM-powered robot brain with dual mode support.

Mode 1: Function Calling (OpenAI, GPT-4o, etc.)
Mode 2: Prompt-based parsing (Gemini, other models without tool support)

Auto-detects which mode to use based on whether the model supports tools.
"""

import json
import re
import os
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv

from ..robot.commander import BittleCommander
from ..robot.skills import SkillRegistry
from .tools import ROBOT_TOOLS

load_dotenv()


class RobotBrain:
    """AI brain that understands natural language and controls the robot."""

    PROMPT_SYSTEM = """你是 JPRobot，一个智能机器狗助手。你控制着一个 Petoi BittleX 四足机器人。

你的职责：
1. 理解用户的自然语言指令（中文或英文）
2. 将指令翻译为机器人动作命令
3. 回复时简洁友好，像一只聪明活泼的小狗

当你需要控制机器人时，在回复中用 <<ACTION:指令代码>> 格式嵌入命令。
可以嵌入多个命令，它们会按顺序执行。

例如：
- 用户说"向前走" → 回复中包含 <<ACTION:wkF>>
- 用户说"翻跟头" → 回复中包含 <<ACTION:bf>>
- 用户说"先站起来再走几步" → 回复中包含 <<ACTION:balance>> 和 <<ACTION:wkF>>
- 用户说"摇摇头" → 回复中包含 <<ACTION:wh>>

如果用户只是聊天，不需要控制机器人，就正常对话，不要加 ACTION 标签。

{skill_list}
"""

    TOOL_SYSTEM = """你是 JPRobot，一个智能机器狗助手。你控制着一个 Petoi BittleX 四足机器人。

你的职责：
1. 理解用户的自然语言指令（中文或英文）
2. 调用合适的工具来控制机器人执行动作
3. 回复时简洁友好，像一只聪明活泼的小狗

注意事项：
- 如果用户说"走"、"向前"等，使用 walk forward (wkF)
- 如果用户说"跑"、"快走"，使用 trot forward (trF)
- 如果用户说"翻跟头"，根据语境选择 back flip (bf) 或 front flip (ff)
- 如果用户要求多个动作，使用 execute_sequence 编排动作序列
- 如果不确定用户要什么，先问清楚再执行

{skill_list}
"""

    def __init__(
        self,
        commander: Optional[BittleCommander] = None,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
        use_tools: Optional[bool] = None,
    ):
        self.commander = commander
        self.model = model
        self.skills = SkillRegistry()

        # Initialize OpenAI client
        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)

        # Auto-detect tool support or use provided setting
        if use_tools is not None:
            self.use_tools = use_tools
        else:
            self.use_tools = self._detect_tool_support()

        # Build system prompt
        if system_prompt:
            self.system_prompt = system_prompt.format(
                skill_list=self.skills.to_prompt_text()
            )
        elif self.use_tools:
            self.system_prompt = self.TOOL_SYSTEM.format(
                skill_list=self.skills.to_prompt_text()
            )
        else:
            self.system_prompt = self.PROMPT_SYSTEM.format(
                skill_list=self.skills.to_prompt_text()
            )

        # Conversation history
        self.messages: list[dict] = [
            {"role": "system", "content": self.system_prompt}
        ]
        self.max_history = 20

        mode_str = "Function Calling" if self.use_tools else "Prompt Parsing"
        print(f"[JPRobot Brain] Model: {model}, Mode: {mode_str}")

    def _detect_tool_support(self) -> bool:
        """Auto-detect if the model supports function calling."""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "hi"}],
                tools=[{
                    "type": "function",
                    "function": {
                        "name": "test",
                        "description": "test",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }],
                max_tokens=10,
            )
            return True
        except Exception:
            return False

    def chat(self, user_input: str) -> str:
        """Process natural language input and execute robot commands."""
        if self.use_tools:
            return self._chat_with_tools(user_input)
        else:
            return self._chat_with_prompt(user_input)

    # === Prompt-based mode (for models without Function Calling) ===

    def _chat_with_prompt(self, user_input: str) -> str:
        """Chat using prompt-based command extraction (<<ACTION:code>>)."""
        self.messages.append({"role": "user", "content": user_input})
        self._trim_history()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
        )

        reply = response.choices[0].message.content

        # Extract and execute ACTION commands
        actions = re.findall(r"<<ACTION:(\w+)>>", reply)
        action_results = []
        for action_code in actions:
            result = self._execute_action(action_code)
            action_results.append(result)

        # Clean ACTION tags from display text
        clean_reply = re.sub(r"\s*<<ACTION:\w+>>\s*", " ", reply).strip()

        # Append execution results
        if action_results:
            executed = ", ".join(
                f"{r['skill']}({'OK' if r['success'] else 'FAIL'})"
                for r in action_results
            )
            clean_reply += f"\n[Executed: {executed}]"

        self.messages.append({"role": "assistant", "content": clean_reply})
        return clean_reply

    def _execute_action(self, skill_code: str) -> dict:
        """Execute a single action by skill code."""
        skill = self.skills.get(skill_code)
        if not skill:
            return {"success": False, "skill": skill_code, "error": "unknown skill"}

        if self.commander:
            try:
                self.commander.perform(skill_code, delay=1.0)
                return {"success": True, "skill": skill.name}
            except Exception as e:
                return {"success": False, "skill": skill.name, "error": str(e)}

        return {"success": True, "skill": skill.name, "mode": "simulation"}

    # === Function Calling mode (for OpenAI, etc.) ===

    def _chat_with_tools(self, user_input: str) -> str:
        """Chat using OpenAI Function Calling."""
        self.messages.append({"role": "user", "content": user_input})
        self._trim_history()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=ROBOT_TOOLS,
            tool_choice="auto",
        )

        message = response.choices[0].message

        if message.tool_calls:
            self.messages.append(message.model_dump())

            for tool_call in message.tool_calls:
                result = self._execute_tool(
                    tool_call.function.name,
                    json.loads(tool_call.function.arguments),
                )
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })

            final_response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=ROBOT_TOOLS,
            )
            reply = final_response.choices[0].message.content
        else:
            reply = message.content

        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def _execute_tool(self, function_name: str, arguments: dict) -> dict:
        """Execute a tool call and return the result."""
        try:
            if function_name == "perform_skill":
                skill_code = arguments["skill_code"]
                delay = arguments.get("delay", 1.0)
                skill = self.skills.get(skill_code)
                if not skill:
                    return {"success": False, "error": f"Unknown skill: {skill_code}"}
                if self.commander:
                    resp = self.commander.perform(skill_code, delay=delay)
                    return {"success": True, "skill": skill.name, "response": resp}
                return {"success": True, "skill": skill.name, "mode": "simulation"}

            elif function_name == "move_head":
                pan = arguments.get("pan", 0)
                tilt = arguments.get("tilt", 0)
                if self.commander:
                    resp = self.commander.move_head(pan, tilt)
                    return {"success": True, "pan": pan, "tilt": tilt, "response": resp}
                return {"success": True, "pan": pan, "tilt": tilt, "mode": "simulation"}

            elif function_name == "move_joint":
                joint = arguments["joint"]
                angle = arguments["angle"]
                if self.commander:
                    resp = self.commander.move_joint(joint, angle)
                    return {"success": True, "joint": joint, "angle": angle, "response": resp}
                return {"success": True, "joint": joint, "angle": angle, "mode": "simulation"}

            elif function_name == "execute_sequence":
                actions = arguments["actions"]
                results = []
                for action in actions:
                    skill_code = action["skill_code"]
                    delay = action.get("delay", 1.0)
                    if self.commander:
                        resp = self.commander.perform(skill_code, delay=delay)
                        results.append({"skill": skill_code, "response": resp})
                    else:
                        results.append({"skill": skill_code, "mode": "simulation"})
                return {"success": True, "sequence": results}

            elif function_name == "beep":
                tone = arguments.get("tone", 20)
                duration = arguments.get("duration", 50)
                if self.commander:
                    resp = self.commander.beep(tone, duration)
                    return {"success": True, "response": resp}
                return {"success": True, "mode": "simulation"}

            elif function_name == "query_status":
                query_type = arguments["query_type"]
                if self.commander:
                    if query_type == "model":
                        resp = self.commander.query_model()
                    else:
                        resp = self.commander.query_joints()
                    return {"success": True, "data": resp}
                return {"success": True, "data": "simulation mode", "mode": "simulation"}

            else:
                return {"success": False, "error": f"Unknown function: {function_name}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _trim_history(self):
        """Keep conversation history within max_history limit."""
        if len(self.messages) > self.max_history + 1:
            self.messages = [self.messages[0]] + self.messages[-(self.max_history):]

    def reset(self):
        """Reset conversation history."""
        self.messages = [{"role": "system", "content": self.system_prompt}]
