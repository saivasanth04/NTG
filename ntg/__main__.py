import argparse
import asyncio
import json
from pathlib import Path
from .router import UnifiedNTGRouter, RoutingError

async def main():
    parser = argparse.ArgumentParser(description="NTG async smart router with provider-enforced access")
    parser.add_argument("prompt", nargs="?")
    parser.add_argument("--env", default=".env")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--provider", action="append")
    parser.add_argument("--fallback-model", action="append", default=[])
    parser.add_argument("--exact-model", action="store_true", help="Disable automatic model fallback")
    parser.add_argument("--deadline", type=float, default=60)
    parser.add_argument("--attempts", type=int, default=None, help="Maximum attempts; default covers the configured pool within the deadline")
    parser.add_argument("--max-tokens", type=int, default=None, help="Optional output cap; default uses the provider's limit")
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()
    router = UnifiedNTGRouter(args.env, routing_state_path=Path(args.env).resolve().parent / '.ntg-routing.json')
    if args.status:
        print(json.dumps(router.availability(), indent=2))
        return
    if not args.prompt:
        parser.error("Provide a prompt or --status")
    output_params = {"max_tokens": args.max_tokens} if args.max_tokens is not None else {}
    try:
        result = await router.acompletion([{"role": "user", "content": args.prompt}], model=args.model,
            providers=args.provider, fallback_models=args.fallback_model,
            allow_model_fallback=not args.exact_model, max_attempts=args.attempts,
            deadline_seconds=args.deadline, stream=args.stream, **output_params)
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
