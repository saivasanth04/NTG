import argparse
import asyncio
import json
from .router import UnifiedNTGRouter, RoutingError

async def main():
    parser = argparse.ArgumentParser(description="Free-use NTG async smart router")
    parser.add_argument("prompt", nargs="?")
    parser.add_argument("--env", default=".env")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--provider", action="append")
    parser.add_argument("--fallback-model", action="append", default=[])
    parser.add_argument("--deadline", type=float, default=60)
    parser.add_argument("--attempts", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()
    router = UnifiedNTGRouter(args.env)
    if args.status:
        print(json.dumps(router.availability(), indent=2))
        return
    if not args.prompt:
        parser.error("Provide a prompt or --status")
    try:
        result = await router.acompletion([{"role": "user", "content": args.prompt}], model=args.model,
            providers=args.provider, fallback_models=args.fallback_model,
            allow_model_fallback=bool(args.fallback_model), max_attempts=args.attempts,
            deadline_seconds=args.deadline, max_tokens=args.max_tokens, stream=args.stream)
        if args.stream:
            try:
                async for chunk in result.response:
                    print(chunk.model_dump_json())
            finally:
                await result.response.aclose()
            print(json.dumps({k: v for k, v in vars(result).items() if k != "response"}))
        else:
            print(json.dumps({**vars(result), "response": result.response.model_dump()}, indent=2))
    except RoutingError as error:
        print(json.dumps(error.details, indent=2))
        raise SystemExit(1)

asyncio.run(main())
