import argparse
import asyncio
import json
from pathlib import Path

from .config import accounts_from_env, load_policy
from .discovery import discover
from .engine import RoutingService
from .types import ChatRequest, RoutingError


async def execute(args):
    accounts = accounts_from_env(args.env)
    deployments, report = await discover(accounts, load_policy(args.policy))
    if args.command == "discover":
        data = {"accounts": report, "deployments": [d.public() for d in deployments]}
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(json.dumps({"accounts": len(accounts), "deployments": len(deployments),
                          "free_verified": sum(d.free == "verified" for d in deployments), "output": args.output}))
    else:
        router = RoutingService(deployments)
        try:
            result = await router.chat(ChatRequest(messages=[{"role": "user", "content": "Reply with OK."}],
                                      max_tokens=32, deadline_seconds=30, model=args.model))
            # Smoke never prints model text. Only safe routing and usage metadata.
            print(json.dumps(result["routing"], indent=2))
        except RoutingError as exc:
            print(json.dumps(exc.public(), indent=2))
            return 1
    return 0


def main():
    parser = argparse.ArgumentParser(description="NTG multi-provider routing")
    parser.add_argument("--env", default=".env")
    parser.add_argument("--policy", default="policy.json")
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("discover")
    d.add_argument("--output", default="runtime/catalog.json")
    s = sub.add_parser("smoke")
    s.add_argument("--model", default="auto")
    sub.add_parser("serve")
    args = parser.parse_args()
    if args.command == "serve":
        if args.env != ".env" or args.policy != "policy.json":
            parser.error("serve uses .env and policy.json in the working directory")
        import uvicorn
        from .gateway import create_app
        uvicorn.run(create_app(), host="127.0.0.1", port=8000, workers=1, access_log=False, log_level="critical")
    else:
        raise SystemExit(asyncio.run(execute(args)))
