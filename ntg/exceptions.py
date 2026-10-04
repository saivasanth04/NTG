"""Error parsing and classification for OpenRouter and LiteLLM exceptions."""

import json
import re
from typing import Any, Dict, Optional

CATEGORY_DAILY_QUOTA = "ACCOUNT_DAILY_QUOTA_EXHAUSTED"
CATEGORY_PROVIDER_LIMIT = "UPSTREAM_PROVIDER_RATE_LIMIT"
CATEGORY_UNKNOWN_LIMIT = "UNKNOWN_RATE_LIMIT"
CATEGORY_AUTH_ERROR = "ACCOUNT_AUTHENTICATION_ERROR"
CATEGORY_SERVER_ERROR = "UPSTREAM_SERVER_ERROR"
CATEGORY_UNKNOWN_ERROR = "UNKNOWN_ERROR"


def extract_error_payload(error: Exception) -> Optional[Dict[str, Any]]:
    """Safely extracts JSON error payload from LiteLLM / OpenRouter exceptions in O(N)."""
    # 1. Direct object or response attributes
    for attr in ("response", "body"):
        val = getattr(error, attr, None)
        if isinstance(val, dict):
            return val
        if hasattr(val, "json") and callable(val.json):
            try:
                res = val.json()
                if isinstance(res, dict):
                    return res
            except Exception:
                pass
        if isinstance(val, str) and "{" in val:
            try:
                res = json.loads(val)
                if isinstance(res, dict):
                    return res
            except Exception:
                pass

    # 2. Extract from stringified exception text
    text = str(error)
    match = re.search(r'\{"error"\s*:', text)
    if match:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[match.start():])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return None


def classify_openrouter_error(error: Exception) -> Dict[str, Any]:
    """Classifies OpenRouter/LiteLLM error and extracts rate limit headers/metadata."""
    payload = extract_error_payload(error) or {}
    err_data = payload.get("error", {})
    metadata = err_data.get("metadata", {})
    headers = metadata.get("headers", {})

    status_code = getattr(error, "status_code", None)
    limit_source = metadata.get("limit_source")
    provider_name = metadata.get("provider_name")
    reset_header = headers.get("X-RateLimit-Reset")

    reset_ts = None
    if reset_header:
        try:
            reset_ts = float(reset_header) / 1000.0  # OpenRouter returns ms timestamps
        except (ValueError, TypeError):
            pass

    # Categorize error
    if limit_source == "openrouter_free_tier_daily":
        category = CATEGORY_DAILY_QUOTA
    elif provider_name:
        category = CATEGORY_PROVIDER_LIMIT
    elif status_code == 429:
        category = CATEGORY_UNKNOWN_LIMIT
    elif status_code in (401, 403):
        category = CATEGORY_AUTH_ERROR
    elif status_code in (500, 502, 503, 504):
        category = CATEGORY_SERVER_ERROR
    else:
        category = CATEGORY_UNKNOWN_ERROR

    return {
        "category": category,
        "limit_source": limit_source,
        "provider_name": provider_name,
        "limit": headers.get("X-RateLimit-Limit"),
        "remaining": headers.get("X-RateLimit-Remaining"),
        "reset": reset_header,
        "reset_timestamp": reset_ts,
        "remedy": metadata.get("remedy_hint"),
        "payload": payload,
    }
