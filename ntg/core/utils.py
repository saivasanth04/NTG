"""Shared parsing, timestamp, and secret sanitization utilities."""

from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import re
import time
from typing import Any, Optional


def parse_duration_string(val: str) -> Optional[float]:
    """Parse duration strings like '6m0s', '12s', '250ms', '1.5s', '1h30m' into seconds."""
    if not val or not isinstance(val, str):
        return None
    val = val.strip().rstrip(".,;:!?()[]{}").strip().lower()

    try:
        return max(1.0, float(val))
    except ValueError:
        pass

    ms_match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*ms", val)
    if ms_match:
        return max(0.5, float(ms_match.group(1)) / 1000.0)

    pattern = r"^(?:(\d+(?:\.\d+)?)\s*h)?\s*(?:(\d+(?:\.\d+)?)\s*m)?\s*(?:(\d+(?:\.\d+)?)\s*s)?$"
    m = re.match(pattern, val)
    if m and (m.group(1) or m.group(2) or m.group(3)):
        hours = float(m.group(1) or 0)
        minutes = float(m.group(2) or 0)
        seconds = float(m.group(3) or 0)
        total = hours * 3600.0 + minutes * 60.0 + seconds
        return max(1.0, total)

    return None


def parse_reset_header(raw: Any) -> tuple[Optional[float], Optional[float]]:
    """Parse a reset header or metadata value into (reset_timestamp, cooldown_seconds).

    Supports:
    - Numeric epoch microseconds (> 1e14)
    - Numeric epoch milliseconds (> 1e11)
    - Numeric epoch seconds (> 1e8)
    - Numeric relative seconds (<= 1e7)
    - Duration strings ('6m0s', '250ms', '1.5s', '1h30m')
    - ISO-8601 datetime strings ('2026-10-08T00:50:00Z', '2026-10-08T00:50:00+00:00')
    - HTTP date format (RFC 2822 / RFC 7231, e.g. 'Wed, 21 Oct 2026 07:28:00 GMT')
    """
    if raw is None:
        return None, None

    now = time.time()

    if isinstance(raw, (int, float)):
        num = float(raw)
        if num > 1e14:
            ts = num / 1e6
            return ts, max(0.5, ts - now)
        elif num > 1e11:
            ts = num / 1000.0
            return ts, max(0.5, ts - now)
        elif num > 1e8:
            ts = num
            return ts, max(0.5, ts - now)
        elif num >= 0:
            cd = max(0.5, num)
            return now + cd, cd
        return None, None

    if isinstance(raw, datetime):
        ts = raw.timestamp()
        return ts, max(0.5, ts - now)

    raw_str = str(raw).strip().rstrip(".,;:!?()[]{}").strip()
    if not raw_str:
        return None, None

    # 1. Check for duration string first (e.g. 6m0s or 500ms or 1.5s)
    if re.search(r"[a-z]", raw_str.lower()):
        dur = parse_duration_string(raw_str)
        if dur is not None:
            return now + dur, max(0.5, dur)

    # 2. Check numeric string values
    try:
        num = float(raw_str)
        if num > 1e14:
            ts = num / 1e6
            return ts, max(0.5, ts - now)
        elif num > 1e11:
            ts = num / 1000.0
            return ts, max(0.5, ts - now)
        elif num > 1e8:
            ts = num
            return ts, max(0.5, ts - now)
        elif num >= 0:
            cd = max(0.5, num)
            return now + cd, cd
    except ValueError:
        pass

    # 3. Check ISO-8601 datetime format
    try:
        iso_str = raw_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(iso_str)
        ts = dt.timestamp()
        return ts, max(0.5, ts - now)
    except Exception:
        pass

    # 4. Check HTTP date format (RFC 2822 / RFC 7231)
    try:
        dt = parsedate_to_datetime(raw_str)
        ts = dt.timestamp()
        return ts, max(0.5, ts - now)
    except Exception:
        pass

    return None, None


def utc_string(timestamp: Optional[float] = None) -> str:
    """Format Unix timestamp (or current time) as an ISO-8601 UTC string."""
    ts = timestamp if timestamp is not None else time.time()
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()


def sanitize_secret(message: Any, secret: Optional[str] = None) -> str:
    """Redact raw API keys and secret tokens from strings."""
    if not message:
        return ""
    text = str(message)
    if secret and secret.strip() and len(secret.strip()) >= 3:
        text = text.replace(secret.strip(), "***REDACTED***")
    text = re.sub(r"(Bearer\s+)[A-Za-z0-9_\-\.]{8,}", r"\1***REDACTED***", text, flags=re.IGNORECASE)
    text = re.sub(r"(sk-[a-zA-Z0-9_\-]{8,})", r"***REDACTED***", text)
    text = re.sub(r"(key[=:\"'\s]+)[A-Za-z0-9_\-]{16,}", r"\1***REDACTED***", text, flags=re.IGNORECASE)
    return text
