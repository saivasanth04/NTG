import asyncio

from ntg import ChatRequest, RoutingService


async def main():
    router = await RoutingService.from_env()
    result = await router.chat(ChatRequest(
        messages=[{"role": "user", "content": "Write a Python function that adds two integers."}],
        max_tokens=256,
    ))
    print(result["choices"][0]["message"]["content"])
    print(result["routing"])


asyncio.run(main())
