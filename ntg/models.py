"""Data models and account state management."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import time
from typing import Any, Dict, List, Optional

from ntg.config import LITELLM_MODEL_NAME, OPENROUTER_FREE_MODEL
from ntg.quota import calculate_rpd, calculate_rpm


def utc_string(timestamp: Optional[float] = None) -> str:
    """Format Unix timestamp (or current time) as an ISO-8601 UTC string."""
    ts = timestamp if timestamp is not None else time.time()
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()


@dataclass
class Account:
    """Represents an OpenRouter account with persistent request timestamps and cooldown state."""

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
    timestamps: List[float] = field(default_factory=list)

    def refresh(self, now: Optional[float] = None) -> bool:
        """Restores availability if the cooldown timer has elapsed."""
        current_time = now if now is not None else time.time()
        if not self.available and self.blocked_until > 0 and current_time >= self.blocked_until:
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
        """Returns the number of seconds remaining in explicit cooldown."""
        return max(0.0, self.blocked_until - time.time()) if self.blocked_until > 0 else 0.0

    def rpm_info(self, now: Optional[float] = None) -> Dict[str, float]:
        """Returns RPM usage and reset metrics."""
        return calculate_rpm(self.timestamps, now=now)

    def rpd_info(self, now: Optional[float] = None) -> Dict[str, float]:
        """Returns RPD usage and reset metrics."""
        return calculate_rpd(self.timestamps, now=now)

    def is_routeable(self, now: Optional[float] = None) -> bool:
        """Account is routeable when available, no cooldown, and remaining RPM & RPD > 0."""
        current_time = now if now is not None else time.time()
        self.refresh(current_time)
        if not self.available:
            return False
        if current_time < self.blocked_until:
            return False
        rpm = self.rpm_info(current_time)
        rpd = self.rpd_info(current_time)
        return rpm["remaining"] > 0 and rpd["remaining"] > 0

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
                "id": f"ntg-{self.name}",
            },
        }