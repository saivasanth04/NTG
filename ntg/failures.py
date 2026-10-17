import re
import math
import time
import json
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


def reset_duration(value):
    """Reset headers can carry epoch seconds/milliseconds or relative durations."""
    try:
        number = float(value)
        if not math.isfinite(number):
            return None
        if number >= 1_000_000_000_000:
            return max(0.0, number / 1000 - time.time())
        if number >= 1_000_000_000:
            return max(0.0, number - time.time())
    except (ValueError, TypeError):
        pass
    return duration(value)


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

    @property
    def category(self):
        if self.code in ("quota_exhausted", "invalid_credentials"):
            return "ACCOUNT_EXHAUSTED"
        if self.code in ("model_inaccessible", "unsupported_capability"):
            return "MODEL_UNAVAILABLE"
        if self.code == "invalid_request":
            return "INVALID_REQUEST"
        return "TEMPORARY_FAILURE"


def upstream_code(exc):
    """Expose only a provider's short error enum, never prompts or raw exception text."""
    try:
        body = getattr(exc, "body", None)
        if body is None:
            body = exc.response.json()
        if isinstance(body, str):
            body = json.loads(body)
        error = body.get("error", body)
        code = error.get("code")
        return code if isinstance(code, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", code) else None
    except (AttributeError, TypeError, ValueError):
        return None


def classify(exc, attempt):
    if type(exc).__name__ == "ContextWindowExceededError":
        return Failure("context_exceeded", "request", 0)
    if type(exc).__name__ == "UnsupportedParamsError":
        return Failure("unsupported_capability", "deployment", 60)
    status = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status is None:
        status = getattr(response, "status_code", None)
    headers = normalized_headers(getattr(response, "headers", {}) or getattr(exc, "headers", {}))
    retry_after = duration(headers.get("retry-after"))
    backoff = retry_after if retry_after is not None else min(30, 2 ** attempt)
    if status == 401:
        return Failure("invalid_credentials", "account", 60)
    if status == 402:
        return Failure("quota_exhausted", "account", retry_after if retry_after is not None else 60)
    if status in (403, 404):
        return Failure("model_inaccessible", "deployment", 60)
    if status == 429:
        # Ambiguous quota scope: conservatively block the ownership group, not just the key.
        return Failure("rate_limited", "group", retry_after if retry_after is not None else 60)
    if status == 400:
        # An explicit upstream grammar limitation is model capability evidence,
        # unlike arbitrary bad-request errors, which must remain terminal.
        message = str(exc).casefold()
        code = upstream_code(exc)
        if code in ("json_validate_failed", "tool_use_failed", "failed_generation") or "failed to generate json" in message:
            return Failure("invalid_structured_output", "deployment", 30)
        if code in ("insufficient_quota", "insufficient_credits", "billing_hard_limit_reached"):
            return Failure("quota_exhausted", "account", 60)
        if any(term in message for term in ("response_format", "json_schema", "grammar error")):
            return Failure("unsupported_capability", "deployment", 60)
        if any(term in message for term in ("max_tokens", "max_completion_tokens", "context length")) and any(
                term in message for term in ("limit", "maximum", "exceed", "must be", "too large")):
            return Failure("context_exceeded", "request", 0)
        if "grammar error" in message and "unimplemented keys" in message:
            return Failure("unsupported_capability", "deployment", 60)
        # Provider-specific incompatibility is not a malformed global request.
        if any(term in message for term in ("not supported", "does not support", "unsupported", "not available")) and any(
                term in message for term in ("response_format", "json_schema", "structured output", "tool_choice", "tools", "model")):
            return Failure("unsupported_capability", "deployment", 60)
    if status in (400, 413, 422):
        # Never replay a malformed / semantically invalid request across all providers.
        return Failure("invalid_request", "request", 0, False)
    if status in (501,):
        return Failure("unsupported_capability", "deployment", 60)
    if status in (502, 504):
        return Failure("provider_unavailable", "provider", backoff)
    if status in (408, 500, 503) or isinstance(exc, (TimeoutError, ConnectionError)) or "Timeout" in type(exc).__name__:
        return Failure("timeout" if status == 408 or isinstance(exc, TimeoutError) or "Timeout" in type(exc).__name__ else "overloaded", "deployment", backoff)
    return Failure("upstream_error", "deployment", backoff)
