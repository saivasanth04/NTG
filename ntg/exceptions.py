"""OpenRouter/LiteLLM/Gemini/Groq/NVIDIA/Cohere error extraction and classification."""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

CATEGORY_DAILY_QUOTA = "ACCOUNT_DAILY_QUOTA_EXHAUSTED"
CATEGORY_PROVIDER_LIMIT = "UPSTREAM_PROVIDER_RATE_LIMIT"
CATEGORY_UNKNOWN_LIMIT = "UNKNOWN_RATE_LIMIT"
CATEGORY_AUTH_ERROR = "ACCOUNT_AUTHENTICATION_ERROR"
CATEGORY_SERVER_ERROR = "UPSTREAM_SERVER_ERROR"
CATEGORY_REQUEST_ERROR = "REQUEST_ERROR"
CATEGORY_NOT_FOUND = "MODEL_NOT_FOUND"
CATEGORY_UNKNOWN_ERROR = "UNKNOWN_ERROR"


@dataclass
class ParsedErrorInfo:
    """Structured information extracted from provider or LiteLLM exceptions."""

    category: str
    status_code: int | None
    is_retryable: bool
    should_quarantine: bool
    cooldown_seconds: float
    reset_timestamp: float | None
    limit_type: str | None  # "rpm", "rpd", "tpm", "quota", "auth", "server", "bad_request", "not_found"
    quota_scope: str  # "deployment", "account", "provider"
    provider_name: str | None
    message: str
    limit: str | None = None
    remaining: str | None = None
    remedy: str | None = None
    raw_payload: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict representation for backward compatibility."""
        res = asdict(self)
        res["payload"] = self.raw_payload
        return res


def extract_error_payload(error: Exception) -> dict[str, Any] | None:
    """Extract a structured JSON payload from an exception."""
    for attr in ("response", "body"):
        value = getattr(error, attr, None)
        if isinstance(value, dict):
            return value
        if hasattr(value, "json") and callable(value.json):
            try:
                parsed = value.json()
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    return parsed
            except (TypeError, ValueError, json.JSONDecodeError):
                pass

    text = str(error)
    match = re.search(r'\{"error"\s*:', text)
    if not match:
        return None

    try:
        parsed, _ = json.JSONDecoder().raw_decode(text[match.start() :])
        if isinstance(parsed, dict):
            return parsed
    except (TypeError, ValueError, json.JSONDecodeError):
        pass

    return None


def _header(headers: Any, name: str) -> Any:
    """Case-insensitive HTTP header lookup."""
    if not isinstance(headers, dict) and not hasattr(headers, "items"):
        return None
    try:
        d = dict(headers)
    except Exception:
        return None

    if name in d:
        return d[name]
    wanted = name.lower()
    for key, value in d.items():
        if str(key).lower() == wanted:
            return value
    return None


def parse_duration_string(val: str) -> float | None:
    """Parse duration strings like '6m0s', '12s', '250ms', '1.5s', '1h30m' into seconds."""
    if not val or not isinstance(val, str):
        return None
    val = val.strip().lower()

    # Try pure float first
    try:
        return max(1.0, float(val))
    except ValueError:
        pass

    # Milliseconds e.g. "250ms"
    ms_match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*ms", val)
    if ms_match:
        return max(0.5, float(ms_match.group(1)) / 1000.0)

    # Complex duration e.g. "1h2m3.5s" or "6m0s"
    pattern = r"^(?:(\d+(?:\.\d+)?)\s*h)?\s*(?:(\d+(?:\.\d+)?)\s*m)?\s*(?:(\d+(?:\.\d+)?)\s*s)?$"
    m = re.match(pattern, val)
    if m and (m.group(1) or m.group(2) or m.group(3)):
        hours = float(m.group(1) or 0)
        minutes = float(m.group(2) or 0)
        seconds = float(m.group(3) or 0)
        total = hours * 3600.0 + minutes * 60.0 + seconds
        return max(1.0, total)

    return None


def parse_reset_header(raw: Any) -> tuple[float | None, float | None]:
    """Parse a reset header into (reset_timestamp, cooldown_seconds).

    Supports:
    - Numeric epoch milliseconds (> 1e11)
    - Numeric epoch seconds (> 1e9)
    - Relative seconds (< 1e7)
    - Duration strings ('6m0s', '250ms')
    - HTTP date format (RFC 2822)
    """
    if raw is None:
        return None, None

    now = time.time()
    raw_str = str(raw).strip()

    # Check for duration string first (e.g. 6m0s or 500ms)
    duration = parse_duration_string(raw_str)
    if duration is not None and not raw_str.replace(".", "", 1).isdigit():
        return now + duration, duration

    # Check numeric values
    try:
        num = float(raw_str)
        if num > 1e11:
            # Epoch milliseconds
            ts = num / 1000.0
            cd = max(1.0, ts - now)
            return ts, cd
        elif num > 1e9:
            # Epoch seconds
            ts = num
            cd = max(1.0, ts - now)
            return ts, cd
        else:
            # Relative seconds
            cd = max(1.0, num)
            return now + cd, cd
    except ValueError:
        pass

    # Check HTTP date format (e.g., 'Wed, 21 Oct 2026 07:28:00 GMT')
    try:
        dt = parsedate_to_datetime(raw_str)
        ts = dt.timestamp()
        cd = max(1.0, ts - now)
        return ts, cd
    except Exception:
        pass

    return None, None


def seconds_until_utc_midnight() -> float:
    """Calculate seconds until 00:00:00 UTC next day."""
    now = datetime.now(timezone.utc)
    # Next day 00:00 UTC
    target = datetime(now.year, now.month, now.day, tzinfo=timezone.utc).timestamp() + 86400.0
    return max(60.0, target - time.time())


def parse_provider_error(error: Exception, provider: str = "") -> ParsedErrorInfo:
    """Extract structured, actionable diagnostic and routing metadata from an exception."""
    p_lower = (provider or "").lower()
    payload = extract_error_payload(error) or {}
    err_data = payload.get("error") if isinstance(payload.get("error"), dict) else {}
    metadata = err_data.get("metadata") if isinstance(err_data.get("metadata"), dict) else {}
    headers = metadata.get("headers") if isinstance(metadata.get("headers"), dict) else {}

    # Check if error has response headers directly
    resp = getattr(error, "response", None)
    if resp is not None and hasattr(resp, "headers"):
        resp_headers = getattr(resp, "headers")
        if isinstance(resp_headers, dict) or hasattr(resp_headers, "items"):
            # Merge with response headers
            try:
                for k, v in dict(resp_headers).items():
                    if k not in headers:
                        headers[k] = v
            except Exception:
                pass

    status_code = getattr(error, "status_code", None)
    if status_code is None and resp is not None:
        status_code = getattr(resp, "status_code", None)

    err_str = str(error)
    err_lower = err_str.lower()
    msg = err_data.get("message") or err_str

    limit_source = metadata.get("limit_source")
    provider_name = metadata.get("provider_name") or provider or None
    remedy = metadata.get("remedy_hint")

    # Header extraction
    retry_after_raw = _header(headers, "retry-after")
    reset_raw = (
        _header(headers, "x-ratelimit-reset")
        or _header(headers, "x-ratelimit-reset-requests")
        or _header(headers, "x-ratelimit-reset-tokens")
    )
    limit_val = _header(headers, "x-ratelimit-limit") or _header(headers, "x-ratelimit-limit-requests")
    rem_val = _header(headers, "x-ratelimit-remaining") or _header(headers, "x-ratelimit-remaining-requests")

    # Determine reset timestamp and cooldown
    reset_ts = None
    cooldown = None

    if retry_after_raw is not None:
        reset_ts, cooldown = parse_reset_header(retry_after_raw)
    if cooldown is None and reset_raw is not None:
        reset_ts, cooldown = parse_reset_header(reset_raw)

    # Classify error type
    # 1. Status 400 Bad Request
    if status_code == 400 or "bad_request" in err_lower or "invalid_argument" in err_lower:
        return ParsedErrorInfo(
            category=CATEGORY_REQUEST_ERROR,
            status_code=400,
            is_retryable=False,
            should_quarantine=False,
            cooldown_seconds=0.0,
            reset_timestamp=None,
            limit_type="bad_request",
            quota_scope="deployment",
            provider_name=provider_name,
            message=msg,
            raw_payload=payload,
        )

    # 2. Status 404 Not Found (Model unavailable/deleted)
    if status_code == 404 or "not found" in err_lower or "model_not_found" in err_lower:
        return ParsedErrorInfo(
            category=CATEGORY_NOT_FOUND,
            status_code=404,
            is_retryable=True,
            should_quarantine=True,
            cooldown_seconds=86400.0,
            reset_timestamp=None,
            limit_type="not_found",
            quota_scope="deployment",
            provider_name=provider_name,
            message=msg,
            raw_payload=payload,
        )

    # 3. Status 401/403 Authentication / Permission Error
    if (
        status_code in (401, 403)
        or "401" in err_lower
        or "403" in err_lower
        or "unauthorized" in err_lower
        or "invalid api key" in err_lower
        or "api_key_invalid" in err_lower
        or "permission_denied" in err_lower
        or "authentication" in err_lower
    ):
        return ParsedErrorInfo(
            category=CATEGORY_AUTH_ERROR,
            status_code=status_code or 401,
            is_retryable=True,
            should_quarantine=True,
            cooldown_seconds=86400.0,
            reset_timestamp=None,
            limit_type="auth",
            quota_scope="account",
            provider_name=provider_name,
            message=msg,
            raw_payload=payload,
        )

    # 4. Status 429 Rate Limit / Quota Exhaustion
    is_429 = (
        status_code == 429
        or "429" in err_lower
        or "rate limit" in err_lower
        or "ratelimit" in err_lower
        or "quota" in err_lower
        or "too many requests" in err_lower
        or "resource_exhausted" in err_lower
    )

    if is_429:
        is_daily = (
            limit_source == "openrouter_free_tier_daily"
            or "daily" in err_lower
            or "per day" in err_lower
            or "rpd" in err_lower
            or "free tier limit" in err_lower
        )
        is_rpm = "per minute" in err_lower or "rpm" in err_lower
        is_tpm = "token" in err_lower or "tpm" in err_lower

        if is_daily:
            category = CATEGORY_DAILY_QUOTA
            limit_type = "rpd"
            quota_scope = "account"
            if cooldown is None:
                cooldown = seconds_until_utc_midnight()
                reset_ts = time.time() + cooldown
        elif is_rpm:
            category = CATEGORY_PROVIDER_LIMIT
            limit_type = "rpm"
            quota_scope = "deployment"
            if cooldown is None:
                cooldown = 60.0
                reset_ts = time.time() + cooldown
        elif is_tpm:
            category = CATEGORY_PROVIDER_LIMIT
            limit_type = "tpm"
            quota_scope = "deployment"
            if cooldown is None:
                cooldown = 30.0
                reset_ts = time.time() + cooldown
        else:
            category = CATEGORY_PROVIDER_LIMIT if provider_name else CATEGORY_UNKNOWN_LIMIT
            limit_type = "quota"
            quota_scope = "deployment"
            if cooldown is None:
                cooldown = 60.0
                reset_ts = time.time() + cooldown

        return ParsedErrorInfo(
            category=category,
            status_code=429,
            is_retryable=True,
            should_quarantine=False,
            cooldown_seconds=max(1.0, cooldown),
            reset_timestamp=reset_ts,
            limit_type=limit_type,
            quota_scope=quota_scope,
            provider_name=provider_name,
            message=msg,
            limit=limit_val,
            remaining=rem_val,
            remedy=remedy,
            raw_payload=payload,
        )

    # 5. Status 500 / 502 / 503 / 504 / Timeout Upstream Server Error
    if (
        status_code in (500, 502, 503, 504)
        or "500" in err_lower
        or "502" in err_lower
        or "503" in err_lower
        or "504" in err_lower
        or "timeout" in err_lower
        or "timed out" in err_lower
        or "connection error" in err_lower
        or "service unavailable" in err_lower
        or "internal server error" in err_lower
    ):
        return ParsedErrorInfo(
            category=CATEGORY_SERVER_ERROR,
            status_code=status_code or 503,
            is_retryable=True,
            should_quarantine=False,
            cooldown_seconds=15.0,
            reset_timestamp=time.time() + 15.0,
            limit_type="server",
            quota_scope="deployment",
            provider_name=provider_name,
            message=msg,
            raw_payload=payload,
        )

    # 6. Fallback Unknown Error
    return ParsedErrorInfo(
        category=CATEGORY_UNKNOWN_ERROR,
        status_code=status_code,
        is_retryable=False,
        should_quarantine=False,
        cooldown_seconds=10.0,
        reset_timestamp=None,
        limit_type="unknown",
        quota_scope="deployment",
        provider_name=provider_name,
        message=msg,
        raw_payload=payload,
    )


def classify_openrouter_error(error: Exception) -> dict[str, Any]:
    """Classify an OpenRouter/LiteLLM exception (returns dict for backward compat)."""
    return parse_provider_error(error, provider="openrouter").to_dict()


def classify_gemini_error(error: Exception) -> dict[str, Any]:
    """Classify a Gemini exception (returns dict for backward compat)."""
    return parse_provider_error(error, provider="gemini").to_dict()


def classify_provider_error(error: Exception, provider: str = "") -> dict[str, Any]:
    """Unified error classifier across all providers (returns dict for backward compat)."""
    return parse_provider_error(error, provider=provider).to_dict()