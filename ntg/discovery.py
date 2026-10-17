"""Explicit catalog refresh. Catalog presence never means usable quota or free access."""
import time
from decimal import Decimal, InvalidOperation

import httpx

from .types import Deployment


def zero_pricing(pricing):
    # Require input/output pricing and reject ANY other nonzero/unknown charge.
    try:
        return bool(pricing) and {"prompt", "completion"} <= pricing.keys() and all(
            Decimal(str(v)) == 0 for v in pricing.values())
    except (InvalidOperation, TypeError):
        return False


async def list_models(client, account):
    p = account.provider
    headers = {"Authorization": f"Bearer {account.key}"}
    if p == "gemini":
        url = "https://generativelanguage.googleapis.com/v1beta/models"
        headers = {"x-goog-api-key": account.key}
        field, token = "models", "pageToken"
    elif p == "groq":
        url, field, token = "https://api.groq.com/openai/v1/models", "data", None
    elif p == "openrouter":
        url, field, token = "https://openrouter.ai/api/v1/models", "data", None
    elif p == "cohere":
        url, field, token = "https://api.cohere.com/v1/models", "models", "page_token"
    else:
        if not account.account_id:
            return [], "missing_account_id"
        url = f"https://api.cloudflare.com/client/v4/accounts/{account.account_id}/ai/models/search"
        field, token = "result", None
    rows, params = [], {}
    for _ in range(20):  # bounded pagination, no unbounded refresh loops
        response = await client.get(url, headers=headers, params=params)
        if response.status_code != 200:
            return [], f"http_{response.status_code}"
        data = response.json()
        if data.get("success") is False:
            return [], "provider_error"
        rows.extend(data.get(field, []))
        next_page = data.get("nextPageToken") or data.get("next_page_token")
        if not token or not next_page:
            break
        params[token] = next_page
    return rows, "listed"


def deployment(account, model, row, override):
    p = account.provider
    d = Deployment(account, model, listed=bool(row), checked_at=time.time())
    # Advisory name hints rank models, never deny a provider/model an attempt.
    name = model.casefold()
    if any(tag in name for tag in ("coder", "code", "reasoning", "gpt-oss", "deepseek-r1", "qwq")):
        d.model_classes.update({"coding", "reasoning"})
    if any(tag in name for tag in ("flash", "lite", "mini", "small")):
        d.model_classes.add("fast")
    if any(tag in name for tag in ("safeguard", "moderation", "translate", "embedding", "rerank", "prompt-guard")):
        # Still callable, but a product-family substring is not evidence of coding suitability.
        d.model_classes.difference_update({"coding", "reasoning"})
    if row:
        d.max_input = int(row.get("inputTokenLimit") or 0)
        d.context = int(row.get("context_length") or row.get("context_window") or 0)
        d.limit_source = "provider_catalog"
        d.max_output = int(row.get("outputTokenLimit") or row.get("top_provider", {}).get("max_completion_tokens") or 0)
        if p == "gemini":
            d.capabilities = {"chat"} if "generateContent" in row.get("supportedGenerationMethods", []) else set()
        elif p == "cohere":
            d.capabilities = {"chat"} if "chat" in row.get("endpoints", []) else set()
        elif p == "openrouter":
            modalities = row.get("architecture", {}).get("output_modalities", ["text"])
            d.capabilities = {"chat"} if "text" in modalities else set()
            inputs = row.get("architecture", {}).get("input_modalities", [])
            if "image" in inputs:
                d.capabilities.add("vision")
            if "audio" in inputs:
                d.capabilities.add("audio_input")
            params = row.get("supported_parameters", [])
            if "tools" in params:
                d.capabilities.add("tools")
            if "response_format" in params:
                d.capabilities.add("json")
            if "structured_outputs" in params:
                d.capabilities.add("json_schema")
            # Informational pricing evidence only; never an inference admission gate.
            if model.endswith(":free") and zero_pricing(row.get("pricing")):
                d.free, d.evidence = "verified", "OpenRouter catalog: explicit :free and all pricing fields zero"
    if override:
        priority = override.get("priority", 0)
        classes = override.get("model_classes", ["general"])
        if (type(priority) is not int or not 0 <= priority <= 1000
                or not isinstance(classes, list) or any(not isinstance(c, str) for c in classes)
                or not set(classes) <= {"general", "fast", "reasoning", "coding"}):
            raise ValueError("Invalid model routing metadata")
        d.priority, d.model_classes = priority, set(classes)
        d.capabilities = set(override.get("capabilities", d.capabilities))
        d.context = int(override.get("context", d.context))
        d.max_output = int(override.get("max_output", d.max_output))
        d.max_input = int(override.get("max_input", d.max_input))
        d.limit_source = "operator_policy"
        d.tokenizer_file = override.get("tokenizer_file")
        if p != "openrouter" and override.get("free_eligible") and account.free_verified:
            d.free, d.evidence = "verified", "operator verified free-only account and documented model policy"
    return d


async def discover(accounts, policy, client=None):
    owned = client is None
    client = client or httpx.AsyncClient(timeout=15, follow_redirects=False)
    deployments, report = [], []
    try:
        # Sequential per account: low-impact metadata only, never completion probes.
        for account in accounts:
            try:
                rows, status = await list_models(client, account)
            except (httpx.HTTPError, ValueError, TypeError):
                rows, status = [], "discovery_unavailable"
            lookup = {((r.get("name") if account.provider == "cloudflare" else r.get("id"))
                       or r.get("name", "")).removeprefix("models/"): r for r in rows}
            names = list(dict.fromkeys([*account.models, *lookup]))
            for model in names:
                override = policy.get("models", {}).get(f"{account.provider}/{model}", {})
                d = deployment(account, model, lookup.get(model), override)
                # Explicit configuration remains attemptable even if catalog discovery fails.
                if d.listed or override or model in account.models:
                    deployments.append(d)
            quota = {"status": "unknown"}
            if account.provider == "openrouter":
                try:
                    response = await client.get("https://openrouter.ai/api/v1/key",
                        headers={"Authorization": f"Bearer {account.key}"})
                    if response.status_code == 200:
                        data = response.json().get("data", {})
                        # Explicit whitelist; never save labels or the raw key response.
                        quota = {k: data[k] for k in ("is_free_tier", "limit_remaining", "free_model_daily_requests") if k in data}
                        quota["status"] = "observed_not_reserved"
                        daily = data.get("free_model_daily_requests") or {}
                        if daily.get("remaining") == 0:
                            # Reset time is not guaranteed by this endpoint; do not invent capacity.
                            for item in deployments:
                                if item.account is account and item.model.endswith(":free"):
                                    # Recent observation, not an invented daily reset time.
                                    item.available_after = time.time() + 60
                                    item.available_reason = "quota_exhausted"
                    else:
                        quota = {"status": f"http_{response.status_code}"}
                except (httpx.HTTPError, ValueError):
                    pass
            report.append({"account": account.alias, "provider": account.provider, "status": status,
                           "listed_count": len(rows), "configured_count": len(account.models),
                           "configured_not_listed": sorted(set(account.models) - lookup.keys()),
                           "free_account_verified": account.free_verified, "quota": quota})
    finally:
        if owned:
            await client.aclose()
    return deployments, report
