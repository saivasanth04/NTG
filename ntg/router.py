"""Core multi-account fallback router orchestrating capacity-aware LiteLLM requests."""

import logging
import time
from typing import Any, List, Optional

import litellm
from litellm import Router

from ntg.config import LITELLM_MODEL_NAME, OPENROUTER_FREE_MODEL
from ntg.diagnostics import (
    print_account_status,
    print_divider,
    print_rate_limit_details,
    print_request_execution,
)
from ntg.exceptions import (
    CATEGORY_AUTH_ERROR,
    CATEGORY_DAILY_QUOTA,
    CATEGORY_PROVIDER_LIMIT,
    CATEGORY_SERVER_ERROR,
    classify_openrouter_error,
)
from ntg.models import Account, utc_string
from ntg.persistence import (
    clean_expired_timestamps,
    load_history,
    record_request_timestamp,
    save_history,
)


class OpenRouterFallbackRouter:
    """Capacity-aware multi-account fallback router built on LiteLLM Router."""

    def __init__(self, accounts: List[Account]):
        self.accounts = accounts

        # Configure LiteLLM logging
        litellm.suppress_debug_info = True
        litellm.set_verbose = False
        logging.getLogger("LiteLLM").setLevel(logging.ERROR)

        # Register all account deployments with LiteLLM
        all_deployments = [acc.to_deployment() for acc in accounts]
        self.router = Router(
            model_list=all_deployments,
            num_retries=0,
        )
        self.load_state()

    def load_state(self) -> None:
        """Loads request timestamp history from persistence and cleans expired records."""
        now = time.time()
        history = load_history()
        cleaned_history = clean_expired_timestamps(history, now=now)
        if cleaned_history != history:
            save_history(cleaned_history)

        for acc in self.accounts:
            acc.timestamps = cleaned_history.get(acc.name, [])
            acc.refresh(now)

    def ask(self, prompt: str) -> Optional[Any]:
        """Dispatches request using capacity-aware selection, recording timestamp before sending."""
        self.load_state()
        now = time.time()

        # Identify routeable accounts
        routeable = [acc for acc in self.accounts if acc.is_routeable(now)]

        if not routeable:
            print_divider("NO AVAILABLE ACCOUNTS")
            print("All configured OpenRouter accounts are currently exhausted or cooling down.")
            
            # Find earliest reset timestamp
            resets = []
            for acc in self.accounts:
                rpm = acc.rpm_info(now)
                rpd = acc.rpd_info(now)
                if rpm["reset_seconds"] > 0:
                    resets.append(rpm["reset_seconds"])
                if rpd["reset_seconds"] > 0:
                    resets.append(rpd["reset_seconds"])
                if acc.blocked_until > now:
                    resets.append(acc.blocked_until - now)

            if resets:
                next_capacity = min(resets)
                print(f"Next available capacity in: {next_capacity:.1f}s ({utc_string(now + next_capacity)})")
            print_account_status(self.accounts)
            return None

        # Sort accounts by available capacity (RPM remaining, then RPD remaining)
        routeable.sort(
            key=lambda a: (
                a.rpm_info(now)["remaining"],
                a.rpd_info(now)["remaining"],
                -a.attempts,
                -a.order,
            ),
            reverse=True,
        )

        selected_account = routeable[0]

        # ----------------------------------------------------
        # RECORD TIMESTAMP BEFORE SENDING REQUEST
        # ----------------------------------------------------
        ts = record_request_timestamp(selected_account.name, ts=now)
        selected_account.timestamps.append(ts)
        selected_account.attempts += 1

        # Sync LiteLLM router deployment target
        self.router.set_model_list([selected_account.to_deployment()])

        try:
            response = self.router.completion(
                model=LITELLM_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                num_retries=0,
            )
            selected_account.successes += 1

            # Extract response model name
            resp_model = getattr(response, "model", None)
            hidden = getattr(response, "_hidden_params", {})
            if isinstance(hidden, dict) and hidden.get("response_model"):
                resp_model = hidden["response_model"]

            print_request_execution(selected_account, response_model=resp_model)
            print("\n" + str(response.choices[0].message.content))
            return response

        except Exception as error:
            selected_account.failures += 1
            info = classify_openrouter_error(error)

            print_divider("REQUEST FAILED")
            print(f"Selected account : {selected_account.name}")
            print(f"Classification   : {info['category']}")
            print(f"LiteLLM error    : {error}")
            print_rate_limit_details(info)

            category = info["category"]
            if category == CATEGORY_DAILY_QUOTA:
                selected_account.rate_limits += 1
                print("\n>>> DEFINITIVE ACCOUNT-LEVEL LIMIT: Free daily quota exhausted.")
                selected_account.block(
                    "OpenRouter free daily quota exhausted",
                    until=info.get("reset_timestamp"),
                )
                return None

            if category == CATEGORY_AUTH_ERROR:
                print("\n>>> Authentication failure on account. Cooling down for 1 hour.")
                selected_account.block("Authentication/authorization failure", cooldown=3600.0)
                return None

            if category == CATEGORY_SERVER_ERROR:
                print("\n>>> Upstream server error. Cooling down for 30s.")
                selected_account.block("Temporary upstream server failure", cooldown=30.0)
                return None

            if category == CATEGORY_PROVIDER_LIMIT:
                selected_account.rate_limits += 1
                print("\n>>> Upstream PROVIDER-level 429. Will NOT burn secondary accounts.")
                return None

            print("\n>>> Unrecoverable or unknown error. Aborting without retry.")
            return None