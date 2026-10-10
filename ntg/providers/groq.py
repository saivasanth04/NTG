"""Groq dynamic model discovery, capability inference, and deployment builder."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Set
import requests

from ntg.core.config import GROQ_KEYS, MODEL_GROUP_GROQ
from ntg.core.models import (
    CircuitState,
    Deployment,
    ModelCapabilities,
    QuotaInfo,
    QuotaScope,
)
from ntg.core.utils import sanitize_secret
from ntg.providers.base import DiscoveryErrorCategory, apply_stored_state, classify_discovery_error

logger = logging.getLogger("ntg.providers.groq")


def infer_groq_capabilities(model_id: str, context_window: int = 8192) -> ModelCapabilities:
    """Infer normalized ModelCapabilities for a Groq model using provider metadata."""
    return ModelCapabilities(
        coding=None,
        reasoning=None,
        vision=None,
        tool_calling=None,
        structured_output=None,
        streaming=True,
        context_window=context_window,
    )


def discover_groq_models(
    api_key: str,
    account_name: str = "",
    state_manager: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Dynamically discover usable text/chat models for a given Groq API key.

    Uses authoritative metadata (e.g. context_window, active status) from the Groq API.
    Saves successful discovery into StateManager cache; on discovery failure/offline,
    falls back to last-known-good cached discovery for this account.
    """
    if not api_key or not api_key.strip():
        return []

    cache_key = f"groq_{account_name}" if account_name else "groq_default"
    url = "https://api.groq.com/openai/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "User-Agent": "NTG-LiteLLM-Router/1.0",
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        status_code = response.status_code

        if status_code == 200:
            try:
                payload = response.json()
            except Exception as json_err:
                cat, detail = classify_discovery_error(json_err, status_code=200)
                logger.warning(
                    "[Groq Discovery] %s for %s: %s",
                    cat.value,
                    account_name or "default",
                    sanitize_secret(detail, api_key),
                )
                if state_manager:
                    state_manager.mark_discovery_stale(
                        cache_key,
                        reason=detail,
                        error_type=cat.value,
                        last_error=sanitize_secret(str(json_err), api_key),
                    )
                    cached = state_manager.get_cached_discovery(cache_key)
                    if cached:
                        return cached
                return []

            model_entries = payload.get("data", [])
            if not isinstance(model_entries, list):
                detail = "Malformed provider response: 'data' is not a list"
                logger.warning(
                    "[Groq Discovery] malformed_response for %s: %s",
                    account_name or "default",
                    detail,
                )
                if state_manager:
                    state_manager.mark_discovery_stale(
                        cache_key,
                        reason=detail,
                        error_type=DiscoveryErrorCategory.MALFORMED_RESPONSE.value,
                    )
                    cached = state_manager.get_cached_discovery(cache_key)
                    if cached:
                        return cached
                return []

            excluded_keywords = [
                "whisper",
                "transcribe",
                "audio",
                "speech",
                "tts",
                "text-to-speech",
                "embed",
                "embedding",
                "guard",
                "moderation",
                "image",
                "imagen",
                "veo",
                "clip",
                "dall-e",
                "robotics",
                "computer-use",
            ]

            discovered: List[Dict[str, Any]] = []
            seen: Set[str] = set()

            for entry in model_entries:
                if not isinstance(entry, dict):
                    continue

                model_id = entry.get("id", "").strip()
                if not model_id or model_id in seen:
                    continue

                if entry.get("active") is False:
                    continue

                model_id_lower = model_id.lower()
                if any(exc in model_id_lower for exc in excluded_keywords):
                    continue

                seen.add(model_id)
                context_window = int(entry.get("context_window", 8192))
                discovered.append({
                    "id": model_id,
                    "context_window": context_window,
                })

            if discovered and state_manager:
                state_manager.set_cached_discovery(cache_key, discovered, is_stale=False)

            return discovered

        else:
            err_msg = f"HTTP {status_code}: {response.text[:200]}"
            cat, detail = classify_discovery_error(Exception(err_msg), status_code=status_code)
            logger.warning(
                "[Groq Discovery] %s for %s: %s",
                cat.value,
                account_name or "default",
                sanitize_secret(detail, api_key),
            )
            if state_manager:
                if cat == DiscoveryErrorCategory.AUTH_FAILURE:
                    state_manager.mark_discovery_stale(
                        cache_key,
                        reason=detail,
                        error_type=cat.value,
                        last_error=sanitize_secret(err_msg, api_key),
                    )
                    return []
                else:
                    state_manager.mark_discovery_stale(
                        cache_key,
                        reason=detail,
                        error_type=cat.value,
                        last_error=sanitize_secret(err_msg, api_key),
                    )
                    cached = state_manager.get_cached_discovery(cache_key)
                    if cached:
                        return cached
            return []

    except Exception as err:
        cat, detail = classify_discovery_error(err)
        if cat == DiscoveryErrorCategory.PROGRAMMING_ERROR:
            logger.error(
                "[Groq Discovery] Programming error for %s: %s",
                account_name or "default",
                err,
                exc_info=True,
            )
            raise err

        logger.warning(
            "[Groq Discovery] %s for %s: %s",
            cat.value,
            account_name or "default",
            sanitize_secret(detail, api_key),
        )
        if state_manager:
            state_manager.mark_discovery_stale(
                cache_key,
                reason=detail,
                error_type=cat.value,
                last_error=sanitize_secret(str(err), api_key),
            )
            cached = state_manager.get_cached_discovery(cache_key)
            if cached:
                return cached
        return []


def build_groq_deployments(
    keys: Optional[Dict[str, str]] = None,
    state_manager: Optional[Any] = None,
) -> List[Deployment]:
    """Dynamically discover chat-capable models and create deployments per Groq account."""
    keys_to_use = keys if keys is not None else GROQ_KEYS
    deployments: List[Deployment] = []

    for i in range(1, 6):
        acc_name = f"account_{i}"
        key = keys_to_use.get(acc_name, "").strip()
        if not key:
            continue

        try:
            discovered_models = discover_groq_models(
                key,
                account_name=acc_name,
                state_manager=state_manager,
            )
            cache_key = f"groq_{acc_name}"
            is_cache_stale = state_manager.is_discovery_stale(cache_key) if state_manager else False
            entry_meta = state_manager.get_discovery_entry(cache_key) if state_manager else None
            stale_reason = entry_meta.get("stale_reason") if entry_meta else None
            is_auth_err = entry_meta.get("error_type") == "auth_failure" if entry_meta else False

            for item in discovered_models:
                if isinstance(item, dict):
                    model_id = item.get("id", "")
                    context_window = item.get("context_window", 8192)
                    is_stale = item.get("is_stale", is_cache_stale)
                    item_stale_reason = item.get("stale_reason", stale_reason)
                else:
                    model_id = str(item)
                    context_window = 8192
                    is_stale = is_cache_stale
                    item_stale_reason = stale_reason

                if not model_id:
                    continue

                dep_id = f"groq-{acc_name}-{model_id}"
                caps = infer_groq_capabilities(model_id, context_window=context_window)
                deployment = Deployment(
                    id=dep_id,
                    provider=MODEL_GROUP_GROQ,
                    account=acc_name,
                    model=model_id,
                    logical_model=MODEL_GROUP_GROQ,
                    litellm_model=f"groq/{model_id}",
                    api_key=key,
                    capabilities=caps,
                    quota=QuotaInfo(quota_scope=QuotaScope.ACCOUNT_MODEL),
                    is_stale=is_stale,
                    stale_reason=item_stale_reason,
                    last_discovered_at=time.time(),
                )
                if is_auth_err:
                    deployment.circuit_state = CircuitState.AUTH_FAILED
                    deployment.state_reason = item_stale_reason or "Authentication failed during discovery"

                apply_stored_state(deployment, state_manager)
                deployments.append(deployment)
        except (TypeError, AttributeError, NameError, SyntaxError):
            raise
        except Exception as err:
            logger.warning(
                "Error building Groq deployments for %s: %s",
                acc_name,
                sanitize_secret(str(err), key),
            )
            continue

    return deployments
