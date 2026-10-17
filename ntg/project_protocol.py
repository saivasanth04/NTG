"""Small JSON protocols for existing agent roles in project mode."""
import json
from copy import deepcopy
from .types import ChatRequest


def structured(name, schema):
    return {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}


def obj(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


ISSUE = obj({"severity": {"type": "string", "enum": ["high", "medium", "low"]},
             "file": {"type": "string"}, "description": {"type": "string", "maxLength": 1200}})
STEP = structured("project_step", obj({
    "action": {"type": "string", "enum": ["read", "search", "list", "repo_search", "repo_dependencies", "edit", "create", "done", "split", "approve", "reject"]},
    "path": {"type": ["string", "null"]}, "start": {"type": ["integer", "null"]},
    "end": {"type": ["integer", "null"]}, "sha256": {"type": ["string", "null"]},
    "text": {"type": ["string", "null"], "maxLength": 16000},
    "note": {"type": "string", "maxLength": 1200},
    "issues": {"type": "array", "maxItems": 12, "items": ISSUE},
}))
CHECK = obj({"name": {"type": "string"}, "arguments": {"type": "array", "maxItems": 16, "items": {"type": "string"}},
             "exit_code": {"type": "integer"}, "stdout_contains": {"type": "array", "maxItems": 8, "items": {"type": "string"}}})
PLAN = structured("project_plan", obj({"tasks": {"type": "array", "minItems": 1, "maxItems": 6,
    "items": obj({"id": {"type": "string", "pattern": "^T[1-6]$"},
        "description": {"type": "string", "minLength": 1, "maxLength": 1800},
        "dependencies": {"type": "array", "maxItems": 5, "items": {"type": "string", "pattern": "^T[1-6]$"}},
        "kind": {"type": "string", "enum": ["leaf", "component"]},
        "files": {"type": "array", "minItems": 1, "maxItems": 12, "items": {"type": "string"}},
        "runtime_files": {"type": "array", "maxItems": 12, "items": {"type": "string"}},
        "contract": {"type": "string", "maxLength": 1800},
        "acceptance": {"type": "array", "minItems": 1, "maxItems": 8, "items": {"type": "string"}},
        "checks": {"type": "array", "maxItems": 6, "items": CHECK},
        "research_questions": {"type": "array", "maxItems": 3, "items": {"type": "string", "maxLength": 300}},
    })}}))


async def ask(llm, role, instructions, packet, response_format=STEP, max_tokens=2048):
    if response_format is STEP:
        response_format = deepcopy(STEP)
        response_format["json_schema"]["name"] = "project_step_"+role
        properties = response_format["json_schema"]["schema"]["properties"]
        properties["action"]["enum"] = (["read", "search", "list", "repo_search", "repo_dependencies", "edit", "create", "done", "split"]
                                         if role == "coder" else ["read", "search", "list", "repo_search", "repo_dependencies", "approve", "reject", "split"])
        if role == "coder":
            properties["issues"]["maxItems"] = 0
    response = await llm.chat(ChatRequest(model_class={"coder": "coding", "planner": "reasoning", "reviewer": "reasoning"}[role],
        messages=[{"role": "system", "content": instructions},
                  {"role": "user", "content": json.dumps(packet, ensure_ascii=False)}],
        response_format=response_format, max_tokens=max_tokens, deadline_seconds=120))
    # Also validate injected clients at the role boundary.
    import jsonschema
    try:
        data = json.loads(response["choices"][0]["message"]["content"])
        jsonschema.validate(data, response_format["json_schema"]["schema"])
    except (KeyError, IndexError, TypeError, ValueError, jsonschema.ValidationError):
        raise ValueError("invalid_project_response") from None
    return data, response.get("routing", {})
