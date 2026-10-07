"""Data models and deployment state management."""

from dataclasses import dataclass
from datetime import datetime, timezone
import time
from typing import Any, Dict, Optional

from ntg.config import (
    MODEL_GROUP_GEMINI,
    MODEL_GROUP_OPENROUTER,
    OPENROUTER_ACTUAL_MODEL,
    OPENROUTER_LITELLM_MODEL,
)


def utc_string(timestamp: Optional[float] = None) -> str:
    """Format Unix timestamp (or current time) as an ISO-8601 UTC string."""
    ts = timestamp if timestamp is not None else time.time()
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()


@dataclass
class Deployment:
    """Provider-neutral deployment representation with metadata and health state."""

    id: str
    provider: str
    account: str
    model: str
    logical_model: str
    litellm_model: str
    api_key: str
    available: bool = True
    blocked_until: float = 0.0
    blocked_reason: Optional[str] = None
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    rate_limits: int = 0

    @property
    def deployment_id(self) -> str:
        """Alias for id."""
        return self.id

    @property
    def name(self) -> str:
        """Alias for model name (backward compatibility)."""
        return self.model

    @property
    def account_name(self) -> str:
        """Alias for account (backward compatibility)."""
        return self.account

    @property
    def order(self) -> int:
        """Legacy order attribute."""
        return 1

    @property
    def display_name(self) -> str:
        """Formatted human-readable deployment label without exposing secrets."""
        return f"{self.provider.title()} {self.account} ({self.model})"

    def refresh(self, now: Optional[float] = None) -> bool:
        """Restores availability if the cooldown timer has elapsed."""
        current_time = now if now is not None else time.time()
        if not self.available and self.blocked_until > 0 and current_time >= self.blocked_until:
            print(f"\n[Router] {self.display_name} cooldown expired. Deployment restored.")
            self.available = True
            self.blocked_until = 0.0
            self.blocked_reason = None
        return self.available and bool(self.api_key and self.api_key.strip())

    def block(
        self,
        reason: str,
        until: Optional[float] = None,
        cooldown: Optional[float] = None,
    ) -> None:
        """Blocks deployment until a given timestamp or relative cooldown seconds."""
        self.available = False
        self.blocked_reason = reason
        self.blocked_until = until if until else (time.time() + (cooldown or 60.0))
        print(f"\n[Router] BLOCKED {self.display_name}")
        print(f"  Reason: {reason}")
        print(f"  Until : {utc_string(self.blocked_until)}")

    @property
    def remaining_cooldown(self) -> float:
        """Returns the number of seconds remaining in explicit cooldown."""
        return max(0.0, self.blocked_until - time.time()) if self.blocked_until > 0 else 0.0

    def is_available(self, now: Optional[float] = None) -> bool:
        """Deployment is available when API key is provided and not in cooldown."""
        current_time = now if now is not None else time.time()
        return self.refresh(current_time)

    def to_litellm_dict(self) -> Dict[str, Any]:
        """Converts deployment into LiteLLM router dictionary format."""
        return {
            "model_name": self.logical_model,
            "litellm_params": {
                "model": self.litellm_model,
                "api_key": self.api_key,
                "metadata": {
                    "deployment_id": self.id,
                    "provider": self.provider,
                    "account": self.account,
                    "model": self.model,
                    "logical_model": self.logical_model,
                },
            },
            "model_info": {
                "id": self.id,
                "provider": self.provider,
                "account": self.account,
                "model": self.model,
                "logical_model": self.logical_model,
            },
        }

    def to_deployment(self) -> Dict[str, Any]:
        """Backward-compatible alias for to_litellm_dict."""
        return self.to_litellm_dict()


class Account(Deployment):
    """Backwards-compatible OpenRouter account deployment."""

    def __init__(
        self,
        name: str,
        api_key: str,
        order: int = 1,
        available: bool = True,
        blocked_until: float = 0.0,
        blocked_reason: Optional[str] = None,
        attempts: int = 0,
        successes: int = 0,
        failures: int = 0,
        rate_limits: int = 0,
    ):
        super().__init__(
            id=f"openrouter-{name}",
            provider=MODEL_GROUP_OPENROUTER,
            account=name,
            model=OPENROUTER_ACTUAL_MODEL,
            logical_model=MODEL_GROUP_OPENROUTER,
            litellm_model=OPENROUTER_LITELLM_MODEL,
            api_key=api_key,
            available=available and bool(api_key and api_key.strip()),
            blocked_until=blocked_until,
            blocked_reason=blocked_reason,
            attempts=attempts,
            successes=successes,
            failures=failures,
            rate_limits=rate_limits,
        )


class GeminiDeployment(Deployment):
    """Backwards-compatible Gemini account/model deployment."""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        account_name: str = "account_1",
        order: int = 1,
        available: bool = True,
        blocked_until: float = 0.0,
        blocked_reason: Optional[str] = None,
        attempts: int = 0,
        successes: int = 0,
        failures: int = 0,
        rate_limits: int = 0,
    ):
        super().__init__(
            id=f"gemini-{account_name}-{model_name}",
            provider=MODEL_GROUP_GEMINI,
            account=account_name,
            model=model_name,
            logical_model=MODEL_GROUP_GEMINI,
            litellm_model=f"gemini/{model_name}",
            api_key=api_key,
            available=available and bool(api_key and api_key.strip()),
            blocked_until=blocked_until,
            blocked_reason=blocked_reason,
            attempts=attempts,
            successes=successes,
            failures=failures,
            rate_limits=rate_limits,
        )