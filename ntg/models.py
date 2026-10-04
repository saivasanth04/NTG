"""Runtime account state and LiteLLM deployment representation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import time
from typing import Any


def utc_string(timestamp: float | None = None) -> str:
    """Format a Unix timestamp as ISO-8601 UTC."""

    ts = time.time() if timestamp is None else timestamp

    return datetime.fromtimestamp(
        ts,
        timezone.utc,
    ).isoformat()


@dataclass
class Account:
    """Represents one OpenRouter API-key deployment."""

    name: str
    api_key: str
    order: int

    rpm_limit: int = 20
    rpd_limit: int = 50
    weight: int = 1

    available: bool = True

    blocked_until: float = 0.0
    blocked_reason: str | None = None

    attempts: int = 0
    successes: int = 0
    failures: int = 0
    rate_limits: int = 0

    # Local diagnostic RPD count.
    requests_today: int = 0

    # Local rolling RPM timestamps.
    request_timestamps: deque[float] = field(
        default_factory=deque
    )

    def refresh(self, now: float | None = None) -> bool:
        """Restore account after cooldown expiration."""

        current = time.time() if now is None else now

        if (
            not self.available
            and self.blocked_until
            and current >= self.blocked_until
        ):
            self.available = True
            self.blocked_until = 0.0
            self.blocked_reason = None

        self._prune_rpm(current)

        return self.available

    def _prune_rpm(self, now: float | None = None) -> None:
        current = time.time() if now is None else now

        cutoff = current - 60.0

        while (
            self.request_timestamps
            and self.request_timestamps[0] <= cutoff
        ):
            self.request_timestamps.popleft()

    @property
    def remaining_cooldown(self) -> float:
        return max(
            0.0,
            self.blocked_until - time.time(),
        )

    @property
    def local_rpm_used(self) -> int:
        self._prune_rpm()

        return len(self.request_timestamps)

    @property
    def local_rpm_remaining(self) -> int:
        return max(
            0,
            self.rpm_limit - self.local_rpm_used,
        )

    @property
    def local_rpd_remaining(self) -> int:
        return max(
            0,
            self.rpd_limit - self.requests_today,
        )

    def mark_attempt(self) -> None:
        """Record a request attempt.

        This is deliberately counted before the provider returns,
        because failed free-tier attempts may consume quota.
        """

        self.attempts += 1
        self.requests_today += 1

        self.request_timestamps.append(time.time())

    def mark_success(self) -> None:
        self.successes += 1

    def mark_failure(self) -> None:
        self.failures += 1

    def mark_rate_limit(self) -> None:
        self.rate_limits += 1

    def block(
        self,
        reason: str,
        *,
        until: float | None = None,
        cooldown: float | None = None,
    ) -> None:
        """Block this account/deployment only."""

        now = time.time()

        if until is not None and until > now:
            target = until
        else:
            target = now + (cooldown or 60.0)

        self.available = False
        self.blocked_until = target
        self.blocked_reason = reason

    def can_route(self) -> bool:
        """Return whether the account is currently usable locally."""

        self.refresh()

        return (
            self.available
            and self.local_rpd_remaining > 0
        )

    def to_deployment(self) -> dict[str, Any]:
        """Build the LiteLLM model-list deployment."""

        return {
            "model_name": "openrouter-free",

            "litellm_params": {
                "model": "openrouter/openrouter/free",
                "api_key": self.api_key,

                # Used by LiteLLM's rate-aware router.
                "rpm": self.rpm_limit,

                # Prevent unlimited concurrency on a deployment.
                "max_parallel_requests": self.rpm_limit,
            },

            "model_info": {
                "id": f"ntg-{self.name}",
                "account": self.name,
                "order": self.order,

                # NTG metadata; OpenRouter remains authoritative.
                "rpd": self.rpd_limit,
                "rpm": self.rpm_limit,
            },

            # LiteLLM deployment-level RPM.
            "rpm": self.rpm_limit,

            "weight": self.weight,
        }