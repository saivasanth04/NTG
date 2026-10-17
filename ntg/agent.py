import json
from dataclasses import dataclass, field
from typing import Any

from .tool_parser import parse_tool_call
from .types import ChatRequest


@dataclass(frozen=True)
class AgentResult:
    agent: str
    answer: str
    routing: dict[str, Any]
    tool_history: list[dict[str, Any]] = field(default_factory=list)
    routing_history: list[dict[str, Any]] = field(default_factory=list)


class Agent:
    MAX_TOOL_CALLS = 5
    MAX_STEPS = 6
    RUN_TIMEOUT = 120

    def __init__(self, name, system_prompt, llm, *, tools=None, max_tokens=2048):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Agent name must be nonempty")
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("System prompt must be nonempty")
        if type(max_tokens) is not int or not 1 <= max_tokens <= 32768:
            raise ValueError("max_tokens must be an integer from 1 through 32768")
        self.max_tokens = max_tokens
        self.name = name
        self.system_prompt = system_prompt
        self.llm = llm
        self.tools = tools

    async def run(self, task, context=None):
        if not isinstance(task, str) or not task.strip():
            raise ValueError("Task must be a nonempty string")
        if context is not None and not isinstance(context, str):
            raise TypeError("Context must be a string or None")
        if self.tools is not None:
            return await self._run_tools(task, context)

        # JSON encodes task/context boundaries; it does not make prompt
        # injection impossible. Context remains untrusted reference data.
        user_input = json.dumps({"task": task, "context": context})
        response = await self.llm.chat(ChatRequest(
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_input},
            ],
            max_tokens=self.max_tokens,
            deadline_seconds=120,
        ))

        # RoutingService validates the default schema before returning.
        # Check the boundary too, so a substituted LLM fails clearly.
        try:
            content = response["choices"][0]["message"]["content"]
            data = json.loads(content)
            if not isinstance(data, dict) or set(data) != {"answer"}:
                raise ValueError("Unexpected output fields")
            answer = data["answer"]
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("Answer must be a nonempty string")
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValueError("Agent received invalid structured output") from exc

        return AgentResult(
            agent=self.name,
            answer=answer,
            routing=response.get("routing", {}),
            routing_history=[response.get("routing", {})],
        )

    @staticmethod
    def _tool_step(response, schema):
        decision = parse_tool_call(response)
        return decision.content, json.loads(decision.content)

    async def _run_tools(self, task, context):
        from .react_agent import ReActAgent

        agent = ReActAgent(
            self.llm, self.tools, max_steps=self.MAX_STEPS,
            name=self.name, system_prompt=self.system_prompt,
            max_tool_calls=self.MAX_TOOL_CALLS,
        )
        agent.RUN_TIMEOUT = self.RUN_TIMEOUT
        return await agent.run(task, context)
