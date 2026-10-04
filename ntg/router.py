"""Dynamic fallback router placing multiple OpenRouter deployments behind one LiteLLM logical model."""

import logging
import time
from typing import Any, Dict, List, Optional

import litellm
from litellm import Router

from ntg.config import LITELLM_MODEL_NAME
from ntg.diagnostics import (
    print_account_status,
    print_divider,
    print_rate_limit_details,
)
from ntg.exceptions import (
    CATEGORY_AUTH_ERROR,
    CATEGORY_DAILY_QUOTA,
    CATEGORY_PROVIDER_LIMIT,
    CATEGORY_SERVER_ERROR,
    classify_openrouter_error,
)
from ntg.models import Account, utc_string


class OpenRouterFallbackRouter:
    """Dynamic OpenRouter fallback router reacting to real OpenRouter errors."""

    def __init__(self, accounts: List[Account]):
        self.accounts = accounts
        self.account_map: Dict[str, Account] = {acc.name: acc for acc in accounts}

        # Configure LiteLLM logging
        litellm.suppress_debug_info = True
        litellm.set_verbose = False
        logging.getLogger("LiteLLM").setLevel(logging.ERROR)

        # Register LiteLLM callbacks for failure and success
        litellm.failure_callback = [self._handle_litellm_failure]
        litellm.success_callback = [self._handle_litellm_success]

        # Initial deployment setup behind single logical model
        valid_deployments = [acc.to_deployment() for acc in accounts if acc.api_key and acc.api_key.strip()]
        self.router = Router(
            model_list=valid_deployments if valid_deployments else [],
            num_retries=max(0, len(valid_deployments) - 1),
            cooldown_time=60,
        )

    def _get_account_from_kwargs(self, kwargs: Dict[str, Any]) -> Optional[Account]:
        """Extracts Account instance from callback kwargs metadata or model_info."""
        lp = kwargs.get("litellm_params", {})
        meta = lp.get("metadata", {}) if isinstance(lp, dict) else {}
        mi = meta.get("model_info") if isinstance(meta, dict) else {}
        if not mi and isinstance(lp, dict):
            mi = lp.get("model_info", {})

        account_name = None
        if isinstance(mi, dict):
            account_name = mi.get("account")
        if not account_name and isinstance(meta, dict):
            account_name = meta.get("account_name")

        if account_name and account_name in self.account_map:
            return self.account_map[account_name]

        api_key = lp.get("api_key") if isinstance(lp, dict) else None
        if api_key:
            for acc in self.accounts:
                if acc.api_key == api_key:
                    return acc
        return None

    def _handle_litellm_failure(
        self, kwargs: Dict[str, Any], exception: Exception, start_time: float, end_time: float
    ) -> None:
        """Reactive callback executed when a deployment fails during LiteLLM router execution."""
        account = self._get_account_from_kwargs(kwargs)
        exc = kwargs.get("exception") or exception
        info = classify_openrouter_error(exc) if exc else {}

        if account:
            account.attempts += 1
            account.failures += 1

            category = info.get("category", "UNKNOWN_ERROR")
            print_divider("DEPLOYMENT FAILED (REACTIVE ERROR)")
            print(f"Failed Account : {account.name}")
            print(f"Classification : {category}")
            print(f"LiteLLM error  : {exc}")
            if info:
                print_rate_limit_details(info)

            # React dynamically to real error classification
            if category == CATEGORY_DAILY_QUOTA:
                account.rate_limits += 1
                reset_ts = info.get("reset_timestamp")
                print(f"\n>>> REAL ERROR REACT: Daily quota exhausted on {account.name}. Blocking account.")
                account.block("OpenRouter free daily quota exhausted", until=reset_ts, cooldown=86400.0)

            elif category == CATEGORY_AUTH_ERROR:
                print(f"\n>>> REAL ERROR REACT: Authentication failure on {account.name}. Cooling down 1 hour.")
                account.block("Authentication failure", cooldown=3600.0)

            elif category == CATEGORY_SERVER_ERROR:
                print(f"\n>>> REAL ERROR REACT: Server error on {account.name}. Cooling down 30 seconds.")
                account.block("Upstream server error", cooldown=30.0)

            elif category == CATEGORY_PROVIDER_LIMIT:
                account.rate_limits += 1
                print(f"\n>>> REAL ERROR REACT: Upstream provider 429 on {account.name}. Cooling down 60 seconds.")
                account.block("Upstream provider rate limit (429)", cooldown=60.0)

            else:
                account.rate_limits += 1
                print(f"\n>>> REAL ERROR REACT: Request error on {account.name}. Cooling down 60 seconds.")
                account.block(f"Request failure: {category}", cooldown=60.0)

    def _handle_litellm_success(
        self, kwargs: Dict[str, Any], response: Any, start_time: float, end_time: float
    ) -> None:
        """Callback executed when a deployment succeeds."""
        account = self._get_account_from_kwargs(kwargs)
        if account:
            account.attempts += 1
            account.successes += 1

    def ask(self, prompt: str) -> Optional[Any]:
        """Dispatches request via LiteLLM logical model router with reactive error fallback."""
        now = time.time()
        available_accounts = [acc for acc in self.accounts if acc.is_available(now)]

        if not available_accounts:
            print_divider("NO AVAILABLE ACCOUNTS")
            print("All configured OpenRouter accounts are currently cooling down or missing API keys.")
            resets = [acc.remaining_cooldown for acc in self.accounts if acc.remaining_cooldown > 0]
            if resets:
                next_capacity = min(resets)
                print(f"Next available account in: {next_capacity:.1f}s ({utc_string(now + next_capacity)})")
            print_account_status(self.accounts)
            return None

        # Update LiteLLM Router model list with available deployments behind single logical model
        active_deployments = [acc.to_deployment() for acc in available_accounts]
        self.router.set_model_list(active_deployments)

        num_retries = max(0, len(active_deployments) - 1)

        try:
            response = self.router.completion(
                model=LITELLM_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                num_retries=num_retries,
            )

            # Identify fulfilling account from response metadata or hidden params
            resp_model = getattr(response, "model", None)
            hidden = getattr(response, "_hidden_params", {})
            if isinstance(hidden, dict) and hidden.get("response_model"):
                resp_model = hidden["response_model"]

            # Output response content
            if hasattr(response, "choices") and response.choices:
                print("\n" + str(response.choices[0].message.content))

            return response

        except Exception as error:
            print_divider("ALL DEPLOYMENTS EXHAUSTED")
            print(f"LiteLLM logical model '{LITELLM_MODEL_NAME}' failed on all available deployments.")
            print(f"Final exception: {error}")
            print_account_status(self.accounts)
            return None