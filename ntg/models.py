"""Data models and deployment state management."""

from dataclasses import dataclass
from datetime import datetime, timezone
import time
from typing import Any, Dict, Optional

from ntg.config import LITELLM_MODEL_NAME, OPENROUTER_FREE_MODEL


def utc_string(timestamp: Optional[float] = None) -> str:
    """Format Unix timestamp (or current time) as an ISO-8601 UTC string."""
    ts = timestamp if timestamp is not None else time.time()
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()


@dataclass
class Deployment:
    """Represents a router deployment (OpenRouter account or Gemini model) with runtime health state."""

    id: str
    provider: str
    name: str
    litellm_model: str
    api_key: str
    order: int = 1
    available: bool = True
    blocked_until: float = 0.0
    blocked_reason: Optional[str] = None
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    rate_limits: int = 0

    @property
    def display_name(self) -> str:
        """Formatted human-readable deployment label without API keys."""
        if self.provider == "openrouter":
            return f"OpenRouter {self.name}"
        return f"Gemini {self.name}"

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

    def to_deployment(self) -> Dict[str, Any]:
        """Converts deployment into LiteLLM router dictionary format."""
        return {
            "model_name": LITELLM_MODEL_NAME,
            "litellm_params": {
                "model": self.litellm_model,
                "api_key": self.api_key,
                "metadata": {
                    "deployment_id": self.id,
                    "provider": self.provider,
                    "account_name": self.name,
                    "deployment_name": self.name,
                },
            },
            "model_info": {
                "id": self.id,
                "provider": self.provider,
                "account": self.name,
                "name": self.name,
                "order": self.order,
            },
        }


class Account(Deployment):
    """Represents an OpenRouter account deployment (subclass of Deployment for backwards compatibility)."""

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
            provider="openrouter",
            name=name,
            litellm_model=OPENROUTER_FREE_MODEL,
            api_key=api_key,
            order=order,
            available=available,
            blocked_until=blocked_until,
            blocked_reason=blocked_reason,
            attempts=attempts,
            successes=successes,
            failures=failures,
            rate_limits=rate_limits,
        )


class GeminiDeployment(Deployment):
    """Represents a dynamically discovered Gemini model deployment."""

    def __init__(
        self,
        model_name: str,
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
            id=f"gemini-{model_name}",
            provider="gemini",
            name=model_name,
            litellm_model=f"gemini/{model_name}",
            api_key=api_key,
            order=order,
            available=available,
            blocked_until=blocked_until,
            blocked_reason=blocked_reason,
            attempts=attempts,
            successes=successes,
            failures=failures,
            rate_limits=rate_limits,
        )