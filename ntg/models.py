"""Data models and account state management."""

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
class Account:
    """Represents an OpenRouter account with rate-limit and cooldown tracking."""

    name: str
    api_key: str
    order: int = 1
    available: bool = True
    blocked_until: float = 0.0
    blocked_reason: Optional[str] = None
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    rate_limits: int = 0

    def refresh(self) -> bool:
        """Restores availability if the cooldown timer has elapsed."""
        if not self.available and self.blocked_until > 0 and time.time() >= self.blocked_until:
            print(f"\n[Router] {self.name} cooldown expired. Account restored.")
            self.available = True
            self.blocked_until = 0.0
            self.blocked_reason = None
        return self.available

    def block(
        self,
        reason: str,
        until: Optional[float] = None,
        cooldown: Optional[float] = None,
    ) -> None:
        """Blocks account until a given timestamp or relative cooldown seconds."""
        self.available = False
        self.blocked_reason = reason
        self.blocked_until = until if until else (time.time() + (cooldown or 60.0))
        print(f"\n[Router] BLOCKED {self.name}")
        print(f"  Reason: {reason}")
        print(f"  Until : {utc_string(self.blocked_until)}")

    @property
    def remaining_cooldown(self) -> float:
        """Returns the number of seconds remaining in cooldown."""
        return max(0.0, self.blocked_until - time.time()) if self.blocked_until > 0 else 0.0

    def to_deployment(self) -> Dict[str, Any]:
        """Converts account into LiteLLM deployment dictionary."""
        return {
            "model_name": LITELLM_MODEL_NAME,
            "litellm_params": {
                "model": OPENROUTER_FREE_MODEL,
                "api_key": self.api_key,
            },
            "model_info": {
                "account": self.name,
                "order": self.order,
            },
        }
