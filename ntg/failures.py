import re
import math
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime


def duration(value):
    if value is None:
        return None
    value = str(value).strip()
    try:
        number = float(value)
        return max(0.0, number) if math.isfinite(number) else None
    except ValueError:
        pass
    try:
        return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
    except (ValueError, TypeError, OverflowError):
        pass
    parts = re.fullmatch(r"(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m)?(?:(\d+(?:\.\d+)?)s)?", value)
    if parts and any(parts.groups()):
        return sum(float(v or 0) * factor for v, factor in zip(parts.groups(), (3600, 60, 1)))
    return None


def normalized_headers(headers):
    # LiteLLM may prefix forwarded headers.
    result = {}
    for k, v in (headers or {}).items():
        key = k.lower()
        for marker in ("retry-after", "x-ratelimit-"):
            pos = key.find(marker)
            if pos >= 0:
                result[key[pos:]] = str(v)
    return result


@dataclass
class Failure:
    code: str
    scope: str
    delay: float
    retry: bool = True


def classify(exc, attempt):
    if type(exc).__name__ in ("UnsupportedParamsError", "ContextWindowExceededError"):
        return Failure("unsupported_capability", "deployment", 3600)
    status = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status is None:
        status = getattr(response, "status_code", None)
    headers = normalized_headers(getattr(response, "headers", {}) or getattr(exc, "headers", {}))
    retry_after = duration(headers.get("retry-after"))
    backoff = min(30, 2 ** attempt)
    if status == 401:
        return Failure("invalid_credentials", "account", 86400)
    if status == 402:
        return Failure("quota_exhausted", "group", retry_after if retry_after is not None else 3600)
    if status in (403, 404):
        return Failure("model_inaccessible", "deployment", 3600)
    if status == 429:
        # Ambiguous quota scope: conservatively block the ownership group, not just the key.
        return Failure("rate_limited", "group", retry_after if retry_after is not None else 60)
    if status in (400, 413, 422):
        # Never replay a malformed / semantically invalid request across all providers.
        return Failure("invalid_request", "request", 0, False)
    if status in (501,):
        return Failure("unsupported_capability", "deployment", 3600)
    if status in (502, 504):
        return Failure("provider_unavailable", "provider", backoff)
    if status in (408, 500, 503) or isinstance(exc, (TimeoutError, ConnectionError)) or "Timeout" in type(exc).__name__:
        return Failure("timeout" if status == 408 or isinstance(exc, TimeoutError) or "Timeout" in type(exc).__name__ else "overloaded", "deployment", backoff)
    return Failure("upstream_error", "deployment", backoff)
