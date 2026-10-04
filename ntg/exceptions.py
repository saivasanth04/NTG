"""OpenRouter/LiteLLM/Gemini error extraction and classification."""

from __future__ import annotations

import json
import re
from typing import Any

CATEGORY_DAILY_QUOTA = "ACCOUNT_DAILY_QUOTA_EXHAUSTED"
CATEGORY_PROVIDER_LIMIT = "UPSTREAM_PROVIDER_RATE_LIMIT"
CATEGORY_UNKNOWN_LIMIT = "UNKNOWN_RATE_LIMIT"
CATEGORY_AUTH_ERROR = "ACCOUNT_AUTHENTICATION_ERROR"
CATEGORY_SERVER_ERROR = "UPSTREAM_SERVER_ERROR"
CATEGORY_REQUEST_ERROR = "REQUEST_ERROR"
CATEGORY_UNKNOWN_ERROR = "UNKNOWN_ERROR"


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
        parsed, _ = json.JSONDecoder().raw_decode(text[match.start():])
        if isinstance(parsed, dict):
            return parsed
    except (TypeError, ValueError, json.JSONDecodeError):
        pass

    return None


def _header(headers: dict[str, Any], name: str) -> Any:
    """Case-insensitive HTTP header lookup."""
    if name in headers:
        return headers[name]
    wanted = name.lower()
    for key, value in headers.items():
        if str(key).lower() == wanted:
            return value
    return None


def classify_openrouter_error(error: Exception) -> dict[str, Any]:
    """Classify an OpenRouter/LiteLLM exception based on HTTP response metadata."""
    payload = extract_error_payload(error) or {}
    err_data = payload.get("error") if isinstance(payload.get("error"), dict) else {}
    metadata = err_data.get("metadata") if isinstance(err_data.get("metadata"), dict) else {}
    headers = metadata.get("headers") if isinstance(metadata.get("headers"), dict) else {}

    status_code = getattr(error, "status_code", None)
    if status_code is None:
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)

    limit_source = metadata.get("limit_source")
    provider_name = metadata.get("provider_name")
    reset_raw = _header(headers, "X-RateLimit-Reset")

    reset_timestamp = None
    if reset_raw is not None:
        try:
            reset_timestamp = float(reset_raw) / 1000.0
        except (TypeError, ValueError):
            pass

    if limit_source == "openrouter_free_tier_daily":
        category = CATEGORY_DAILY_QUOTA
    elif status_code in (401, 403):
        category = CATEGORY_AUTH_ERROR
    elif provider_name and status_code == 429:
        category = CATEGORY_PROVIDER_LIMIT
    elif status_code == 429:
        category = CATEGORY_UNKNOWN_LIMIT
    elif status_code in (500, 502, 503, 504):
        category = CATEGORY_SERVER_ERROR
    elif status_code in (400, 404):
        category = CATEGORY_REQUEST_ERROR
    else:
        category = CATEGORY_UNKNOWN_ERROR

    return {
        "category": category,
        "status_code": status_code,
        "limit_source": limit_source,
        "provider_name": provider_name,
        "limit": _header(headers, "X-RateLimit-Limit"),
        "remaining": _header(headers, "X-RateLimit-Remaining"),
        "reset": reset_raw,
        "reset_timestamp": reset_timestamp,
        "remedy": metadata.get("remedy_hint"),
        "message": err_data.get("message"),
        "payload": payload,
    }


def classify_gemini_error(error: Exception) -> dict[str, Any]:
    """Classify a Gemini exception based on status code or exception message."""
    status_code = getattr(error, "status_code", None)
    if status_code is None:
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)

    err_str = str(error)
    err_lower = err_str.lower()

    if (
        status_code == 429
        or "429" in err_lower
        or "resource_exhausted" in err_lower
        or "quota" in err_lower
        or "rate_limit" in err_lower
        or "ratelimit" in err_lower
        or "too many requests" in err_lower
    ):
        category = CATEGORY_PROVIDER_LIMIT
    elif (
        status_code in (401, 403)
        or "401" in err_lower
        or "403" in err_lower
        or "unauthorized" in err_lower
        or "permission_denied" in err_lower
        or "api_key_invalid" in err_lower
        or "invalid api key" in err_lower
    ):
        category = CATEGORY_AUTH_ERROR
    elif (
        status_code in (500, 502, 503, 504)
        or "500" in err_lower
        or "503" in err_lower
        or "internal" in err_lower
        or "unavailable" in err_lower
        or "service unavailable" in err_lower
    ):
        category = CATEGORY_SERVER_ERROR
    elif status_code in (400, 404) or "invalid_argument" in err_lower or "not found" in err_lower:
        category = CATEGORY_REQUEST_ERROR
    else:
        category = CATEGORY_UNKNOWN_ERROR

    return {
        "category": category,
        "status_code": status_code,
        "message": err_str,
    }