"""Only retry eligibility and recent failures; no quota accounting."""
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import random
import re
import time

@dataclass
class Block:
    category: str
    until: float | None
    failures: int = 1

def duration(value):
    if isinstance(value, dict) and "seconds" in value:
        return float(value["seconds"]) + float(value.get("nanos", 0)) / 1e9
    try:
        return float(value)
    except (TypeError, ValueError):
        parts = re.findall(r"([\d.]+)(ms|s|m|h|d)", str(value))
        return sum(float(n) * {"ms": .001, "s": 1, "m": 60, "h": 3600, "d": 86400}[u]
                   for n, u in parts) if parts else None

def classify(error, provider=None):
    response = getattr(error, "response", None)
    headers = {k.lower(): v for k, v in dict(getattr(response, "headers", {}) or {}).items()}
    body = getattr(error, "body", None)
    if body is None and response is not None:
        try:
            body = response.json()
        except Exception:
            pass
    text = (str(error) + " " + json.dumps(body, default=str)).lower()
    code = getattr(error, "status_code", None) or getattr(response, "status_code", None)
    now = time.time()
    times = []
    for name, value in headers.items():
        if name == "retry-after":
            try:
                times.append(now + float(value))
            except ValueError:
                try:
                    times.append(parsedate_to_datetime(value).timestamp())
                except (ValueError, TypeError):
                    pass
        elif "reset" in name:
            seconds = duration(value)
            if seconds is not None:
                times.append(seconds / 1000 if seconds > 1e12 else seconds if seconds > 1e9 else now + seconds)
            else:
                try:
                    times.append(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())
                except ValueError:
                    pass
    # Google RetryInfo and arbitrary nested structured retry/reset fields.
    def visit(node):
        if isinstance(node, dict):
            for key, val in node.items():
                if key.lower() in ("retrydelay", "retry_after", "retryafter"):
                    sec = duration(val)
                    if sec is not None:
                        times.append(now + sec)
                if key.lower() in ("reset_at", "resettime", "reset_time"):
                    try:
                        times.append(float(val) if str(val).replace('.', '').isdigit() else
                                     datetime.fromisoformat(val.replace("Z", "+00:00")).timestamp())
                    except (ValueError, TypeError):
                        pass
                visit(val)
        elif isinstance(node, list):
            for val in node:
                visit(val)
    visit(body)
    retry = max((t for t in times if t >= now), default=None)
    if code == 401 or any(s in text for s in ("invalid api key", "api_key_invalid", "unauthenticated")):
        category = "invalid_credentials"
    elif code == 402 or any(s in text for s in ("insufficient credit", "credit balance", "depleted", "payment required")):
        category = "depleted_credits"
    elif any(s in text for s in ("perday", "per_day", "per day", "daily", "permonth", "monthly", "per_month", "per month", "neuron")) and (code == 429 or "quota" in text or "limit" in text):
        category = "long_term_exhaustion"
    elif code == 429 or "resource_exhausted" in text:
        category = "throttled"
    elif code in (403, 404, 410) or any(s in text for s in ("model_decommissioned", "deprecated model", "model_not_found")):
        category = "inaccessible_model"
    elif isinstance(error, TimeoutError) or "timeout" in type(error).__name__.lower():
        category = "timeout"
    elif code and code >= 500:
        category = "server_failure"
    elif code in (400, 422):
        category = "malformed_request"
    else:
        category = "transport_failure"
    if provider == "gemini" and category == "long_term_exhaustion" and retry is not None and retry - now < 3600:
        retry = None  # A minute Retry-After does not establish a daily/monthly reset.
    scope = "model"
    if category == "invalid_credentials":
        scope = "account"
    elif category == "depleted_credits":
        scope = "group"
    elif category in ("throttled", "long_term_exhaustion"):
        # Defaults follow the documented quota scope, not the mere presence of
        # a model name in an error. Explicit shared quotas override defaults.
        scope = "group_model" if provider in ("gemini", "groq") else "group"
        if provider == "cohere" and category == "throttled":
            scope = "group_model"
        if any(s in text for s in ("permodel", "per_model", "per model", "per-model")):
            scope = "group_model"
        model_quota, shared_quota = False, False
        def quota_scope(node):
            nonlocal model_quota, shared_quota
            if isinstance(node, dict):
                dimensions = node.get("quotaDimensions", {})
                quota_id = str(node.get("quotaId", "")).lower()
                if quota_id or dimensions:
                    if dimensions.get("model") or "permodel" in quota_id:
                        model_quota = True
                    else:
                        shared_quota = True
                for val in node.values():
                    quota_scope(val)
            elif isinstance(node, list):
                for val in node:
                    quota_scope(val)
        quota_scope(body)
        if model_quota:
            scope = "group_model"
        if shared_quota or any(s in text for s in ("all models", "all_models", "account-wide", "project-wide", "organization-wide")):
            scope = "group"
        if provider == "cohere" and category == "long_term_exhaustion":
            scope = "group"
        if provider == "cloudflare" and "neuron" in text:
            scope = "group"
        if provider == "openrouter":
            if "upstream_provider_shared_pool" in text or "rate-limited upstream" in text:
                scope = "provider_model"
            if category == "long_term_exhaustion" or "free-models-per" in text or "free model daily" in text:
                scope = "group"
    return category, retry, scope

def backoff(failures):
    return random.uniform(.5, 1) * min(60, 2 ** min(failures, 6))
