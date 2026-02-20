# ADR-0007: LLM 双模式支持（Function Calling + 提示解析）

**状态**：已采纳
**日期**：2026-02-19
**作者**：JPRobot Team

---

## 背景

JPRobot 的核心功能之一是**自然语言控制机器狗**：用户输入中文指令，系统解析意图并执行对应动作。

不同 LLM 提供商对工具调用（Function Calling）的支持程度不同：
- **Claude / GPT-4o**：原生支持 Function Calling，返回结构化 JSON 工具调用
- **国产模型（Qwen、DeepSeek 等）**：部分支持，部分仅支持文本输出
- **轻量级本地模型**：通常不支持 Function Calling

需要一套兼容多种 LLM 的控制框架。

## 决策

`RobotBrain`（`jprobot/agent/brain.py`）实现**双模式自动切换**：

```python
class RobotBrain:
    def process_command(self, user_input: str):
        # 模式1：Function Calling（Claude / GPT-4o）
        if self._supports_function_calling():
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=ROBOT_TOOLS,  # 来自 tools.py
            )
            if response.choices[0].message.tool_calls:
                return self._execute_tool_calls(response)

        # 模式2：提示解析（fallback）
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages + [PARSING_PROMPT],
        )
        return self._parse_text_response(response.choices[0].message.content)
```

工具定义（`jprobot/agent/tools.py`）：
```python
ROBOT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "perform_skill",
            "description": "执行预设技能动作",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_code": {"type": "string", "enum": SKILL_CODES}
                }
            }
        }
    },
    # move_joint, execute_sequence, ...
]
```

## 理由

**为什么需要双模式**：
- Claude Sonnet 4.6（默认模型）原生支持 Function Calling，精准可靠
- 保留提示解析 fallback，确保接入其他 LLM 时无需修改上层逻辑
- 统一 OpenAI SDK 接口：无论哪家 LLM，均通过 `OPENAI_BASE_URL` 兼容适配

**配置方式（`.env`）**：
```env
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://vibe.deepminer.ai/v1  # 可替换为任意 OpenAI 兼容端点
JPROBOT_MODEL=claude-sonnet-4-6               # 可替换为任意模型名
```

## 后果

**正面影响**：
- 切换模型只需修改 `.env`，核心逻辑无需改动
- Function Calling 模式精准度高，误判率低（Claude Sonnet 实测几乎无误判）
- 提示解析模式作为保险，确保系统在降级环境中仍可运行

**负面影响**：
- 提示解析模式依赖模型生成格式规范的文本，鲁棒性弱于 Function Calling
- `_supports_function_calling()` 的判断逻辑需要维护模型白名单

**扩展方向**：
- 流式输出（Streaming）支持，减少用户等待感知
- 多轮对话上下文管理（当前每条指令独立，无历史记忆）
- 语音输入集成（STT → RobotBrain → BittleCommander）
