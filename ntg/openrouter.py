"""OpenRouter metadata client."""

from __future__ import annotations

from typing import Any

import requests

from ntg.config import OPENROUTER
from ntg.models import Account


class OpenRouterClient:
    """Read-only OpenRouter account metadata client."""

    def __init__(
        self,
        timeout: float | None = None,
    ):
        self.timeout = (
            timeout
            or OPENROUTER.http_timeout_seconds
        )

    def get_key_info(
        self,
        account: Account,
    ) -> dict[str, Any] | None:
        try:

            response = requests.get(
                OPENROUTER.key_url,
                headers={
                    "Authorization":
                        f"Bearer {account.api_key}"
                },
                timeout=self.timeout,
            )

            if not response.ok:
                return None

            payload = response.json()

            data = payload.get(
                "data"
            )

            if isinstance(
                data,
                dict,
            ):
                return data

        except (
            requests.RequestException,
            ValueError,
        ):
            return None

        return None