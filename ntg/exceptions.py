"""OpenRouter/LiteLLM/Gemini/Groq/NVIDIA/Cohere error extraction and classification."""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

from ntg.models import QuotaScope, parse_duration_string, parse_reset_header, sanitize_secret

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
    quota_scope: str  # "provider", "account", "model", "account_model", "unknown"
    provider_name: str | None
    message: str
    limit: str | None = None
    remaining: str | None = None
    remedy: str | None = None
    raw_payload: dict[str, Any] | None = None
    rpm_limit: int | None = None
    rpm_remaining: int | None = None
    rpd_limit: int | None = None
    rpd_remaining: int | None = None
    reset_source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict representation for backward compatibility."""
        res = asdict(self)
        res["payload"] = self.raw_payload
        res["reset_at"] = self.reset_timestamp
        res["retry_after"] = self.cooldown_seconds
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


def resolve_rate_limit_reset(
    headers: dict[str, Any],
    metadata: dict[str, Any],
    error: Exception | None = None,
    err_lower: str = "",
    is_daily: bool = False,
) -> tuple[float, float, str]:
    """Resolve (reset_timestamp, cooldown_seconds, priority_source) using strict priority:

    Priority 1: Retry-After header
    Priority 2: Authoritative reset timestamp header (x-ratelimit-reset, ratelimit-reset)
    Priority 3: Provider-specific rate-limit headers (x-ratelimit-reset-requests, x-ratelimit-reset-tokens, etc.)
    Priority 4: Structured LiteLLM/provider metadata & error text duration
    Priority 5: Conservative fallback cooldown only when no authoritative information exists.
    """
    now = time.time()

    # Priority 1: Retry-After header
    retry_after_raw = _header(headers, "retry-after")
    if retry_after_raw is not None:
        ts, cd = parse_reset_header(retry_after_raw)
        if cd is not None and ts is not None:
            return ts, max(0.5, cd), "retry_after_header"

    # Priority 2: Authoritative reset timestamp header
    for h_name in ("x-ratelimit-reset", "ratelimit-reset"):
        raw_val = _header(headers, h_name)
        if raw_val is not None:
            ts, cd = parse_reset_header(raw_val)
            if cd is not None and ts is not None:
                return ts, max(0.5, cd), f"reset_timestamp_header:{h_name}"

    # Priority 3: Provider-specific rate-limit headers
    for h_name in (
        "x-ratelimit-reset-requests",
        "x-ratelimit-reset-tokens",
        "x-ratelimit-reset-requests-day",
        "x-ratelimit-reset-day",
        "x-goog-ratelimit-reset",
        "anthropic-ratelimit-requests-reset",
        "anthropic-ratelimit-tokens-reset",
        "openai-ratelimit-reset-requests",
        "openai-ratelimit-reset-tokens",
    ):
        raw_val = _header(headers, h_name)
        if raw_val is not None:
            ts, cd = parse_reset_header(raw_val)
            if cd is not None and ts is not None:
                return ts, max(0.5, cd), f"provider_header:{h_name}"

    # Priority 4: Structured LiteLLM / provider metadata & error text duration
    # 4a. Exception attributes and metadata dict
    meta_candidates: list[Any] = []
    if error is not None:
        for attr in ("retry_after", "reset_at", "reset_timestamp", "retry_after_seconds"):
            val = getattr(error, attr, None)
            if val is not None:
                meta_candidates.append(val)
    if isinstance(metadata, dict):
        for key in ("retry_after", "retry_after_seconds", "reset_at", "reset_timestamp", "reset", "reset_seconds"):
            val = metadata.get(key)
            if val is not None:
                meta_candidates.append(val)

    for meta_val in meta_candidates:
        ts, cd = parse_reset_header(meta_val)
        if cd is not None and ts is not None:
            return ts, max(0.5, cd), "structured_metadata"

    # 4b. Error text duration (e.g. "try again in 6m0s", "resets in 45s", "wait 12.5s")
    if err_lower:
        retry_match = re.search(r"(?:try again in|resets? in|wait)\s+([0-9a-z.]+)", err_lower)
        if retry_match:
            dur = parse_duration_string(retry_match.group(1))
            if dur is not None:
                return now + dur, max(0.5, dur), "error_text_duration"

    # Priority 5: Conservative fallback cooldown only when no authoritative info exists
    if is_daily:
        cd = seconds_until_utc_midnight()
        return now + cd, cd, "fallback_utc_midnight"

    # Conservative short fallback for RPM / unknown limits (default 60s)
    cd = 60.0
    return now + cd, cd, "fallback_conservative_short"


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

    # Collect all available headers from exception and response objects
    for h_src in [
        getattr(error, "headers", None),
        getattr(error, "response_headers", None),
        payload.get("headers"),
        metadata.get("headers"),
        getattr(getattr(error, "response", None), "headers", None),
        getattr(getattr(error, "raw_response", None), "headers", None),
    ]:
        if h_src and (isinstance(h_src, dict) or hasattr(h_src, "items")):
            try:
                for k, v in dict(h_src).items():
                    if k not in headers:
                        headers[k] = v
            except Exception:
                pass

    status_code = getattr(error, "status_code", None)
    resp = getattr(error, "response", None) or getattr(error, "raw_response", None)
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

    # Header-level numerical quota values
    rpm_limit = None
    rpm_remaining = None
    rpd_limit = None
    rpd_remaining = None

    raw_req_lim = _header(headers, "x-ratelimit-limit-requests")
    if raw_req_lim and str(raw_req_lim).isdigit():
        rpm_limit = int(raw_req_lim)

    raw_req_rem = _header(headers, "x-ratelimit-remaining-requests")
    if raw_req_rem and str(raw_req_rem).isdigit():
        rpm_remaining = int(raw_req_rem)

    raw_rpd_lim = _header(headers, "x-ratelimit-limit-requests-day") or _header(headers, "x-ratelimit-limit-day")
    if raw_rpd_lim and str(raw_rpd_lim).isdigit():
        rpd_limit = int(raw_rpd_lim)

    raw_rpd_rem = _header(headers, "x-ratelimit-remaining-requests-day") or _header(headers, "x-ratelimit-remaining-day")
    if raw_rpd_rem and str(raw_rpd_rem).isdigit():
        rpd_remaining = int(raw_rpd_rem)

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
            quota_scope=QuotaScope.UNKNOWN,
            provider_name=provider_name,
            message=msg,
            raw_payload=payload,
        )

    # 2. Status 404 Not Found (Distinguish Model Not Found vs General 404)
    if status_code == 404 or "not found" in err_lower or "model_not_found" in err_lower:
        is_model_specific = (
            (
                "model" in err_lower
                and any(
                    kw in err_lower
                    for kw in (
                        "not found",
                        "does not exist",
                        "not exist",
                        "access",
                        "unknown",
                        "unavailable",
                        "recognized",
                        "invalid",
                        "could not find",
                        "no such",
                    )
                )
            )
            or any(
                pattern in err_lower
                for pattern in (
                    "model not found",
                    "model_not_found",
                    "models/",
                )
            )
            or (isinstance(err_data, dict) and err_data.get("code") == "model_not_found")
        )

        if is_model_specific:
            return ParsedErrorInfo(
                category=CATEGORY_NOT_FOUND,
                status_code=404,
                is_retryable=False,
                should_quarantine=True,
                cooldown_seconds=86400.0,
                reset_timestamp=None,
                limit_type="not_found",
                quota_scope=QuotaScope.MODEL,
                provider_name=provider_name,
                message=msg,
                raw_payload=payload,
            )
        else:
            # General 404 (endpoint / path / route not found): Client request error, do NOT quarantine model
            return ParsedErrorInfo(
                category=CATEGORY_REQUEST_ERROR,
                status_code=404,
                is_retryable=False,
                should_quarantine=False,
                cooldown_seconds=0.0,
                reset_timestamp=None,
                limit_type="not_found",
                quota_scope=QuotaScope.UNKNOWN,
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
        safe_msg = sanitize_secret(msg)
        return ParsedErrorInfo(
            category=CATEGORY_AUTH_ERROR,
            status_code=status_code or 401,
            is_retryable=False,
            should_quarantine=True,
            cooldown_seconds=0.0,
            reset_timestamp=None,
            limit_type="auth",
            quota_scope=QuotaScope.ACCOUNT,
            provider_name=provider_name,
            message=safe_msg,
            raw_payload=payload,
            reset_source="auth_quarantine",
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
            or "requests/day" in err_lower
            or (rpd_remaining is not None and rpd_remaining == 0)
        )
        is_rpm = "per minute" in err_lower or "rpm" in err_lower
        is_tpm = "token" in err_lower or "tpm" in err_lower

        # Extract limits/used from error text if present (e.g. Groq "Limit 30, Used 30")
        limit_match = re.search(r"limit\s+(\d+)", err_lower)
        used_match = re.search(r"used\s+(\d+)", err_lower)
        parsed_text_limit = int(limit_match.group(1)) if limit_match else None
        parsed_text_used = int(used_match.group(1)) if used_match else None
        parsed_text_rem = max(0, parsed_text_limit - parsed_text_used) if (parsed_text_limit is not None and parsed_text_used is not None) else None

        # 1. Distinguish Category, limit_type, and quota_scope
        if is_daily:
            category = CATEGORY_DAILY_QUOTA
            limit_type = "rpd"
            rpd_remaining = 0
            day_match = re.search(r"(\d+)\s*requests/day", err_lower)
            if day_match:
                rpd_limit = rpd_limit or int(day_match.group(1))
            rpd_limit = rpd_limit or parsed_text_limit

            if p_lower == "groq" or "groq" in err_lower:
                quota_scope = QuotaScope.ACCOUNT_MODEL
            elif p_lower in ("openrouter", "gemini", "cohere", "nvidia"):
                quota_scope = QuotaScope.ACCOUNT
            else:
                quota_scope = QuotaScope.ACCOUNT if provider_name else QuotaScope.UNKNOWN

        elif is_rpm or is_tpm or p_lower in ("groq", "gemini", "openrouter", "cohere", "nvidia"):
            category = CATEGORY_PROVIDER_LIMIT
            limit_type = "tpm" if is_tpm else "rpm"
            rpm_limit = rpm_limit or parsed_text_limit
            rpm_remaining = rpm_remaining if rpm_remaining is not None else (parsed_text_rem if parsed_text_rem is not None else 0)

            if p_lower == "groq" or "groq" in err_lower or "model `" in err_lower:
                quota_scope = QuotaScope.ACCOUNT_MODEL
            elif p_lower == "gemini" or "generativelanguage" in err_lower:
                quota_scope = QuotaScope.ACCOUNT_MODEL
            elif p_lower in ("openrouter", "cohere", "nvidia"):
                quota_scope = QuotaScope.ACCOUNT
            else:
                quota_scope = QuotaScope.PROVIDER if provider_name else QuotaScope.UNKNOWN

        else:
            # Unknown 429
            category = CATEGORY_UNKNOWN_LIMIT
            limit_type = "quota"
            quota_scope = QuotaScope.UNKNOWN

        # 2. Resolve reset timestamp and cooldown via strict 5-level priority
        reset_ts, cooldown, reset_source = resolve_rate_limit_reset(
            headers=headers,
            metadata=metadata,
            error=error,
            err_lower=err_lower,
            is_daily=is_daily,
        )

        return ParsedErrorInfo(
            category=category,
            status_code=429,
            is_retryable=True,
            should_quarantine=False,
            cooldown_seconds=cooldown,
            reset_timestamp=reset_ts,
            limit_type=limit_type,
            quota_scope=quota_scope,
            provider_name=provider_name,
            message=msg,
            limit=limit_val,
            remaining=rem_val,
            remedy=remedy,
            raw_payload=payload,
            rpm_limit=rpm_limit,
            rpm_remaining=rpm_remaining,
            rpd_limit=rpd_limit,
            rpd_remaining=rpd_remaining,
            reset_source=reset_source,
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


# Alias for convenience
classify_exception = parse_provider_error