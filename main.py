"""NTG command-line interface."""

from __future__ import annotations

import asyncio
import sys

from ntg import (
    Account,
    DEFAULT_KEYS,
    OpenRouterFallbackRouter,
)

from ntg.config import (
    LITELLM,
    OPENROUTER,
)

from ntg.diagnostics import (
    print_account_status,
    print_divider,
    print_pool_summary,
)


def build_accounts() -> list[Account]:

    return [
        Account(
            name=f"account_{index}",
            api_key=key,
            order=index,
            rpm_limit=(
                OPENROUTER.requests_per_minute
            ),
            rpd_limit=(
                OPENROUTER.free_requests_per_day
            ),
        )

        for index, key
        in enumerate(
            DEFAULT_KEYS,
            start=1,
        )
    ]


async def run(
    prompt: str | None = None,
) -> None:

    accounts = build_accounts()

    router = OpenRouterFallbackRouter(
        accounts
    )

    print_divider(
        "NTG — OPENROUTER FREE MULTI-ACCOUNT GATEWAY"
    )

    print(
        f"OpenRouter model   : "
        f"{OPENROUTER.free_model}"
    )

    print(
        f"LiteLLM model      : "
        f"{LITELLM.model_name}"
    )

    print(
        f"Routing strategy   : "
        f"{LITELLM.routing_strategy}"
    )

    print(
        f"Configured accounts: "
        f"{len(accounts)}"
    )

    print(
        "Account routing    : LiteLLM"
    )

    print(
        "Model routing      : OpenRouter free router"
    )

    print(
        "Automatic retries  : disabled"
    )

    print_pool_summary(
        accounts
    )

    print_account_status(
        accounts
    )

    if prompt is not None:

        await router.aask(
            prompt
        )

        print_account_status(
            accounts
        )

        return

    while True:

        try:
            value = input(
                "\nNTG> "
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError,
        ):
            print(
                "\nExiting."
            )
            return

        if value.lower() in {
            "exit",
            "quit",
        }:
            return

        if not value:
            continue

        await router.aask(
            value
        )

        print_account_status(
            accounts
        )


def main() -> None:

    prompt = (
        " ".join(sys.argv[1:]).strip()
        or None
    )

    asyncio.run(
        run(prompt)
    )


if __name__ == "__main__":
    main()