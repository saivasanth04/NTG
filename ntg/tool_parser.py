"""Parse structured model decisions without executing any actions."""
import json
from dataclasses import dataclass

import jsonschema

from .types import tool_response_format


@dataclass(frozen=True)
class ToolDecision:
    content: str
    tool: str | None
    arguments: dict[str, str]
    answer: str | None


def parse_tool_call(response):
    try:
        content = response["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise ValueError("Content must be a JSON string")
        step = json.loads(content)
        jsonschema.validate(step, tool_response_format()["json_schema"]["schema"])
        name, args, answer = step["tool"], step["arguments"], step["answer"]
        if name is None:
            valid = (args["path"] is None and args["content"] is None
                     and isinstance(answer, str) and bool(answer.strip()))
            arguments = {}
        else:
            valid = (answer is None and isinstance(args["path"], str)
                     and bool(args["path"].strip())
                     and (isinstance(args["content"], str) if name == "write_file"
                          else args["content"] is None))
            arguments = {"path": args["path"]}
            if name == "write_file":
                arguments["content"] = args["content"]
        if not valid:
            raise ValueError("Contradictory tool step")
    except (KeyError, IndexError, TypeError, ValueError, jsonschema.ValidationError) as exc:
        raise ValueError("Agent received invalid tool step") from exc
    return ToolDecision(content, name, arguments, answer)
