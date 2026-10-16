"""Opt-in, small live probe. Prints/saves routing metadata, never prompts or model text."""
import asyncio
import json
import time
from pathlib import Path

from ntg import ChatRequest, RoutingError, RoutingService


async def main():
    service = await RoutingService.from_env()
    requests = [
        ("chat", ChatRequest(messages=[{"role": "user", "content": "Reply OK."}], max_tokens=64)),
        ("tools", ChatRequest(messages=[{"role": "user", "content": "Call the ping tool with value hello."}],
            tools=[{"type": "function", "function": {"name": "ping", "description": "Echo a value",
                "parameters": {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]}}}],
            tool_choice="required", max_tokens=256)),
        ("json_schema", ChatRequest(messages=[{"role": "user", "content": "Return an object with value equal to 1."}],
            response_format={"type": "json_schema", "json_schema": {"name": "value", "strict": True,
                "schema": {"type": "object", "properties": {"value": {"type": "integer"}},
                    "required": ["value"], "additionalProperties": False}}}, max_tokens=256)),
    ]
    results = []
    for label, request in requests:
        start = time.monotonic()
        try:
            result = await service.chat(request)
            item = {"probe": label, "outcome": "success", "routing": result["routing"]}
            if label == "tools":
                item["returned_tool_calls"] = bool(result["choices"][0]["message"].get("tool_calls"))
            results.append(item)
        except RoutingError as exc:
            results.append({"probe": label, "outcome": exc.code, "history": exc.history})
        results[-1]["wall_seconds"] = round(time.monotonic() - start, 3)
    # Two simultaneous callers exercise the real scheduler and provider path.
    async def concurrent_probe(index):
        try:
            result = await service.chat(requests[0][1])
            return {"probe": f"concurrent_{index}", "outcome": "success", "routing": result["routing"]}
        except RoutingError as exc:
            return {"probe": f"concurrent_{index}", "outcome": exc.code, "history": exc.history}
    results.extend(await asyncio.gather(concurrent_probe(1), concurrent_probe(2)))
    Path("runtime").mkdir(exist_ok=True)
    Path("runtime/live-validation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


asyncio.run(main())
