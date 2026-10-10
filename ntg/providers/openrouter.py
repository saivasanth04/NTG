"""OpenRouter capability inference and deployment builder."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import requests

from ntg.core.config import (
    MODEL_GROUP_OPENROUTER,
    OPENROUTER_ACTUAL_MODEL,
    OPENROUTER_KEYS,
    OPENROUTER_LITELLM_MODEL,
)
from ntg.core.models import Deployment, ModelCapabilities, QuotaInfo, QuotaScope
from ntg.providers.base import DiscoveryErrorCategory, apply_stored_state, classify_discovery_error


def infer_openrouter_capabilities(
    model_id: str = OPENROUTER_ACTUAL_MODEL,
    state_manager: Optional[Any] = None,
) -> ModelCapabilities:
    """Infer capabilities for an OpenRouter model using authoritative metadata from OpenRouter API or cache."""
    cache_key = f"openrouter_model_{model_id.replace('/', '_').replace(':', '_')}"
    model_meta = None
    if state_manager:
        cached = state_manager.get_cached_discovery(cache_key)
        if isinstance(cached, dict):
            model_meta = cached

    if not model_meta:
        try:
            url = "https://openrouter.ai/api/v1/models"
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                match = next((m for m in data if m.get("id") == model_id), None)
                if match:
                    model_meta = match
                    if state_manager:
                        state_manager.set_cached_discovery(cache_key, match, is_stale=False)
            else:
                cat, detail = classify_discovery_error(
                    Exception(f"HTTP {resp.status_code}"),
                    status_code=resp.status_code,
                )
                if state_manager:
                    state_manager.mark_discovery_stale(cache_key, reason=detail, error_type=cat.value)
        except Exception as err:
            cat, detail = classify_discovery_error(err)
            if cat == DiscoveryErrorCategory.PROGRAMMING_ERROR:
                raise err
            if state_manager:
                state_manager.mark_discovery_stale(cache_key, reason=detail, error_type=cat.value)

    if isinstance(model_meta, dict):
        context_window = int(model_meta.get("context_length") or 4096)
        arch = model_meta.get("architecture") or {}
        input_mods = arch.get("input_modalities", [])
        modality = str(arch.get("modality", ""))
        vision: Optional[bool] = (
            ("image" in input_mods or "image" in modality) if (input_mods or modality) else None
        )
        params = model_meta.get("supported_parameters")
        if isinstance(params, list):
            tool_calling: Optional[bool] = "tools" in params
            reasoning: Optional[bool] = (
                True
                if (
                    "reasoning" in params
                    or "include_reasoning" in params
                    or bool(model_meta.get("reasoning"))
                )
                else None
            )
            structured_output: Optional[bool] = (
                "structured_outputs" in params or "response_format" in params
            )
        else:
            tool_calling = None
            reasoning = None
            structured_output = None

        return ModelCapabilities(
            coding=None,
            reasoning=reasoning,
            vision=vision,
            tool_calling=tool_calling,
            structured_output=structured_output,
            streaming=True,
            context_window=context_window,
        )

    return ModelCapabilities(
        coding=None,
        reasoning=None,
        vision=None,
        tool_calling=None,
        structured_output=None,
        streaming=True,
        context_window=4096,
    )


def build_openrouter_deployments(
    keys: Optional[Dict[str, str]] = None,
    state_manager: Optional[Any] = None,
) -> List[Deployment]:
    """Create 5 independent LiteLLM deployments for OpenRouter accounts."""
    keys_to_use = keys if keys is not None else OPENROUTER_KEYS
    deployments: List[Deployment] = []

    capabilities = infer_openrouter_capabilities(OPENROUTER_ACTUAL_MODEL, state_manager=state_manager)

    for i in range(1, 6):
        acc_name = f"account_{i}"
        key = keys_to_use.get(acc_name, "").strip()

        deployment = Deployment(
            id=f"openrouter-{acc_name}",
            provider=MODEL_GROUP_OPENROUTER,
            account=acc_name,
            model=OPENROUTER_ACTUAL_MODEL,
            logical_model=MODEL_GROUP_OPENROUTER,
            litellm_model=OPENROUTER_LITELLM_MODEL,
            api_key=key,
            capabilities=capabilities,
            quota=QuotaInfo(quota_scope=QuotaScope.ACCOUNT),
        )
        apply_stored_state(deployment, state_manager)
        deployments.append(deployment)

    return deployments
