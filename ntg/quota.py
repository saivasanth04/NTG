"""Rolling RPM/RPD window calculations and reset timing."""

import time
from typing import Dict, List, Optional, Tuple

from ntg.config import ACCOUNT_MAX_RPD, ACCOUNT_MAX_RPM


def calculate_rpm(
    timestamps: List[float],
    now: Optional[float] = None,
    max_rpm: int = ACCOUNT_MAX_RPM,
) -> Dict[str, float]:
    """Calculates rolling RPM usage, remaining capacity, and reset timing over a 60s window."""
    current_time = now if now is not None else time.time()
    window_cutoff = current_time - 60.0

    active_timestamps = [ts for ts in timestamps if ts > window_cutoff]
    rpm_used = len(active_timestamps)
    rpm_remaining = max(0, max_rpm - rpm_used)

    reset_seconds = 0.0
    if active_timestamps:
        oldest_ts = min(active_timestamps)
        reset_seconds = max(0.0, (oldest_ts + 60.0) - current_time)

    return {
        "used": rpm_used,
        "remaining": rpm_remaining,
        "max": max_rpm,
        "reset_seconds": reset_seconds,
    }


def calculate_rpd(
    timestamps: List[float],
    now: Optional[float] = None,
    max_rpd: int = ACCOUNT_MAX_RPD,
) -> Dict[str, float]:
    """Calculates rolling RPD usage, remaining capacity, and reset timing over an 86400s window."""
    current_time = now if now is not None else time.time()
    window_cutoff = current_time - 86400.0

    active_timestamps = [ts for ts in timestamps if ts > window_cutoff]
    rpd_used = len(active_timestamps)
    rpd_remaining = max(0, max_rpd - rpd_used)

    reset_seconds = 0.0
    if active_timestamps:
        oldest_ts = min(active_timestamps)
        reset_seconds = max(0.0, (oldest_ts + 86400.0) - current_time)

    return {
        "used": rpd_used,
        "remaining": rpd_remaining,
        "max": max_rpd,
        "reset_seconds": reset_seconds,
    }