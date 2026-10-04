"""Core multi-account fallback router orchestrating LiteLLM requests."""

import logging
from typing import Any, List, Optional

import litellm
from litellm import Router

from ntg.config import LITELLM_MODEL_NAME, OPENROUTER_FREE_MODEL
from ntg.diagnostics import print_account_status, print_divider, print_rate_limit_details
from ntg.exceptions import (
    CATEGORY_AUTH_ERROR,
    CATEGORY_DAILY_QUOTA,
    CATEGORY_PROVIDER_LIMIT,
    CATEGORY_SERVER_ERROR,
    classify_openrouter_error,
)
from ntg.models import Account


class OpenRouterFallbackRouter:
    """Manages multi-account priority failover and rate-limit classification."""

    def __init__(self, accounts: List[Account]):
        self.accounts = accounts

        # Configure LiteLLM logging
        litellm.suppress_debug_info = True
        litellm.set_verbose = False
        logging.getLogger("LiteLLM").setLevel(logging.ERROR)

        active = [acc.to_deployment() for acc in accounts if acc.available]
        self.router = Router(
            model_list=active or [accounts[0].to_deployment()],
            num_retries=0,
        )

    def refresh(self) -> None:
        """Refreshes cooldown states across all registered accounts."""
        for acc in self.accounts:
            acc.refresh()

    def ask(self, prompt: str) -> Optional[Any]:
        """Dispatches completion request across accounts with smart rate-limit failover."""
        self.refresh()
        print_divider("NEW REQUEST")
        print(f"Model: {OPENROUTER_FREE_MODEL}")

        available = [acc for acc in self.accounts if acc.available]
        if not available:
            print("\n[Router] No OpenRouter accounts are currently available.")
            print_account_status(self.accounts)
            return None

        # Sort accounts by priority order
        available.sort(key=lambda a: a.order)

        for attempt, account in enumerate(available, start=1):
            account.attempts += 1
            print(f"\n[Router] Attempt {attempt} using '{account.name}'")
            print(f"[Router] Available accounts: {', '.join(a.name for a in self.accounts if a.available)}")

            # Direct LiteLLM router deployment to target active account
            self.router.set_model_list([account.to_deployment()])

            try:
                response = self.router.completion(
                    model=LITELLM_MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    num_retries=0,
                )
                account.successes += 1
                print_divider("REQUEST SUCCESSFUL")
                print(f"Account: {account.name}")
                print(f"Response model: {getattr(response, 'model', None)}\n")
                print(response.choices[0].message.content)
                return response

            except Exception as error:
                account.failures += 1
                info = classify_openrouter_error(error)

                print_divider("REQUEST FAILED")
                print(f"Account: {account.name}")
                print(f"Classification: {info['category']}")
                print(f"LiteLLM error: {error}")
                print_rate_limit_details(info)

                category = info["category"]
                if category == CATEGORY_DAILY_QUOTA:
                    account.rate_limits += 1
                    print("\n>>> DEFINITIVE ACCOUNT-LEVEL LIMIT: Free daily quota exhausted.")
                    account.block(
                        "OpenRouter free daily quota exhausted",
                        until=info.get("reset_timestamp"),
                    )
                    continue

                if category == CATEGORY_AUTH_ERROR:
                    print("\n>>> Authentication failure on account. Cooling down for 1 hour.")
                    account.block("Authentication/authorization failure", cooldown=3600.0)
                    continue

                if category == CATEGORY_SERVER_ERROR:
                    print("\n>>> Upstream server error. Cooling down for 30s and trying next account.")
                    account.block("Temporary upstream server failure", cooldown=30.0)
                    continue

                if category == CATEGORY_PROVIDER_LIMIT:
                    account.rate_limits += 1
                    print("\n>>> Upstream PROVIDER-level 429. Will NOT burn second account.")
                    return None

                print("\n>>> Unrecoverable or unknown error. Aborting without retry.")
                return None

        print("\nAll OpenRouter accounts exhausted.")
        print_account_status(self.accounts)
        return None
