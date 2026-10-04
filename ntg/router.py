"""NTG multi-account LiteLLM router.

LiteLLM chooses the OpenRouter account deployment.
OpenRouter's openrouter/free router chooses the actual free model/provider.
NTG handles account state and OpenRouter-specific quota interpretation.
"""

from __future__ import annotations

import logging
from typing import Any

import litellm
from litellm import Router

from ntg.config import (
    LITELLM,
    LITELLM_MODEL_NAME,
    OPENROUTER_FREE_MODEL,
)
from ntg.diagnostics import (
    print_divider,
    print_rate_limit_details,
)
from ntg.exceptions import (
    CATEGORY_AUTH_ERROR,
    CATEGORY_DAILY_QUOTA,
    CATEGORY_PROVIDER_LIMIT,
    CATEGORY_REQUEST_ERROR,
    CATEGORY_SERVER_ERROR,
    CATEGORY_UNKNOWN_LIMIT,
    classify_openrouter_error,
)
from ntg.models import Account


class OpenRouterFallbackRouter:
    """Final NTG pool-based router."""

    def __init__(
        self,
        accounts: list[Account],
    ):
        if not accounts:
            raise ValueError(
                "At least one OpenRouter account is required."
            )

        self.accounts = accounts

        litellm.suppress_debug_info = True
        litellm.set_verbose = False

        logging.getLogger(
            "LiteLLM"
        ).setLevel(logging.ERROR)

        # IMPORTANT:
        # All accounts stay in ONE LiteLLM model group.
        #
        # We do NOT replace the model list per request.
        #
        # This lets LiteLLM perform deployment selection.
        self.router = Router(
            model_list=[
                account.to_deployment()
                for account in accounts
            ],
            routing_strategy=(
                LITELLM.routing_strategy
            ),
            num_retries=LITELLM.retries,
            default_max_parallel_requests=(
                LITELLM.max_parallel_requests
            ),
            cooldown_time=(
                LITELLM.cooldown_time_seconds
            ),
        )

    def refresh(self) -> None:
        for account in self.accounts:
            account.refresh()

    def _healthy_accounts(
        self,
    ) -> list[Account]:
        self.refresh()

        return [
            account
            for account in self.accounts
            if account.can_route()
        ]

    def _account_from_response(
        self,
        response: Any,
    ) -> Account | None:
        """Resolve the serving account when LiteLLM exposes deployment ID."""

        hidden = (
            getattr(
                response,
                "_hidden_params",
                {},
            )
            or {}
        )

        model_id = hidden.get(
            "model_id"
        )

        if model_id:
            model_id = str(model_id)

            for account in self.accounts:
                if model_id == (
                    f"ntg-{account.name}"
                ):
                    return account

        return None

    async def aask(
        self,
        prompt: str,
    ) -> Any:
        return await self._request_async(
            prompt
        )

    def ask(
        self,
        prompt: str,
    ) -> Any:
        return self._request_sync(
            prompt
        )

    async def _request_async(
        self,
        prompt: str,
    ) -> Any:

        self._validate_prompt(prompt)

        self.refresh()

        if not self._healthy_accounts():
            self._print_no_accounts()
            return None

        print_divider(
            "NEW REQUEST"
        )

        print(
            f"Model group        : "
            f"{LITELLM_MODEL_NAME}"
        )

        print(
            f"OpenRouter model   : "
            f"{OPENROUTER_FREE_MODEL}"
        )

        print(
            f"Accounts in pool   : "
            f"{len(self.accounts)}"
        )

        print(
            "Account routing    : "
            "LiteLLM"
        )

        print(
            "Model routing      : "
            "OpenRouter free router"
        )

        print(
            "Automatic retries  : "
            "DISABLED"
        )

        try:
            response = (
                await self.router.acompletion(
                    model=LITELLM_MODEL_NAME,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    num_retries=0,
                )
            )

            account = (
                self._account_from_response(
                    response
                )
            )

            if account:
                account.mark_attempt()
                account.mark_success()

            self._print_success(
                response,
                account,
            )

            return response

        except Exception as error:
            self._handle_error(error)

            return None

    def _request_sync(
        self,
        prompt: str,
    ) -> Any:

        self._validate_prompt(prompt)

        self.refresh()

        if not self._healthy_accounts():
            self._print_no_accounts()
            return None

        print_divider(
            "NEW REQUEST"
        )

        print(
            f"Model group        : "
            f"{LITELLM_MODEL_NAME}"
        )

        print(
            f"OpenRouter model   : "
            f"{OPENROUTER_FREE_MODEL}"
        )

        print(
            f"Accounts in pool   : "
            f"{len(self.accounts)}"
        )

        print(
            "Account routing    : "
            "LiteLLM"
        )

        print(
            "Model routing      : "
            "OpenRouter free router"
        )

        print(
            "Automatic retries  : "
            "DISABLED"
        )

        try:
            response = (
                self.router.completion(
                    model=LITELLM_MODEL_NAME,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    num_retries=0,
                )
            )

            account = (
                self._account_from_response(
                    response
                )
            )

            if account:
                account.mark_attempt()
                account.mark_success()

            self._print_success(
                response,
                account,
            )

            return response

        except Exception as error:
            self._handle_error(error)

            return None

    @staticmethod
    def _validate_prompt(
        prompt: str,
    ) -> None:

        if (
            not isinstance(prompt, str)
            or not prompt.strip()
        ):
            raise ValueError(
                "Prompt must not be empty."
            )

    def _handle_error(
        self,
        error: Exception,
    ) -> None:

        info = classify_openrouter_error(
            error
        )

        print_divider(
            "REQUEST FAILED"
        )

        print(
            f"Classification    : "
            f"{info['category']}"
        )

        print(
            f"Status code       : "
            f"{info['status_code']}"
        )

        print(
            f"LiteLLM error     : "
            f"{error}"
        )

        print_rate_limit_details(
            info
        )

        category = info[
            "category"
        ]

        # IMPORTANT:
        # We do not guess the account if LiteLLM
        # does not expose the selected deployment.
        account = (
            self._account_from_error(
                error
            )
        )

        if category == CATEGORY_DAILY_QUOTA:

            if account:

                account.mark_attempt()
                account.mark_failure()
                account.mark_rate_limit()

                reset = info.get(
                    "reset_timestamp"
                )

                account.block(
                    "OpenRouter free daily "
                    "quota exhausted",
                    until=reset,
                )

            print(
                "\n>>> Definitive OpenRouter "
                "daily quota exhaustion."
            )

            print(
                ">>> Only the affected "
                "deployment should be removed."
            )

            return

        if category == CATEGORY_PROVIDER_LIMIT:

            if account:

                account.mark_attempt()
                account.mark_failure()
                account.mark_rate_limit()

            print(
                "\n>>> Provider-level 429."
            )

            print(
                ">>> This is NOT evidence "
                "that the account's daily "
                "quota is exhausted."
            )

            print(
                ">>> No blind retry is performed."
            )

            return

        if category == CATEGORY_UNKNOWN_LIMIT:

            if account:

                account.mark_attempt()
                account.mark_failure()
                account.mark_rate_limit()

            print(
                "\n>>> Unknown 429."
            )

            print(
                ">>> No automatic retry to "
                "avoid wasting free quota."
            )

            return

        if category == CATEGORY_AUTH_ERROR:

            if account:

                account.mark_attempt()
                account.mark_failure()

                account.block(
                    "Authentication failure",
                    cooldown=3600,
                )

            print(
                "\n>>> Authentication failure."
            )

            return

        if category == CATEGORY_SERVER_ERROR:

            if account:

                account.mark_attempt()
                account.mark_failure()

                account.block(
                    "Temporary upstream "
                    "server failure",
                    cooldown=(
                        LITELLM.cooldown_time_seconds
                    ),
                )

            print(
                "\n>>> Temporary upstream "
                "server failure."
            )

            return

        if category == CATEGORY_REQUEST_ERROR:

            if account:

                account.mark_attempt()
                account.mark_failure()

            print(
                "\n>>> Request-level error."
            )

            print(
                ">>> Fix the request instead "
                "of rotating accounts."
            )

            return

        print(
            "\n>>> Unknown error."
        )

        print(
            ">>> No blind retry performed."
        )

    def _account_from_error(
        self,
        error: Exception,
    ) -> Account | None:

        response = getattr(
            error,
            "response",
            None,
        )

        if response is None:
            return None

        return self._account_from_response(
            response
        )

    @staticmethod
    def _print_success(
        response: Any,
        account: Account | None,
    ) -> None:

        print_divider(
            "REQUEST SUCCESSFUL"
        )

        print(
            f"Account           : "
            f"{account.name if account else 'unknown'}"
        )

        print(
            f"Response model    : "
            f"{getattr(response, 'model', None)}"
        )

        try:
            content = (
                response
                .choices[0]
                .message
                .content
            )

        except (
            AttributeError,
            IndexError,
            KeyError,
        ):
            content = None

        print(
            f"\n{content}"
        )

    @staticmethod
    def _print_no_accounts() -> None:

        print_divider(
            "NO AVAILABLE ACCOUNTS"
        )

        print(
            "All configured OpenRouter "
            "accounts are locally exhausted "
            "or unavailable."
        )