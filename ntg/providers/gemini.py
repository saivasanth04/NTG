"""Gemini dynamic model discovery, capability inference, and deployment builder."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Union
from google import genai

from ntg.core.config import (
    DEFAULT_GEMINI_KEYS,
    GEMINI_API_KEY,
    GEMINI_KEYS,
    MODEL_GROUP_GEMINI,
    load_env_file,
)
from ntg.core.models import (
    CircuitState,
    Deployment,
    ModelCapabilities,
    QuotaInfo,
    QuotaScope,
)
from ntg.core.utils import sanitize_secret
from ntg.providers.base import DiscoveryErrorCategory, apply_stored_state, classify_discovery_error

load_env_file()

logger = logging.getLogger("ntg.providers.gemini")


def discover_gemini_models(
    api_key: Optional[str] = None,
    account_name: str = "",
    state_manager: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Discover models available to the Google API key using Google GenAI SDK."""
    key = (
        api_key
        or GEMINI_API_KEY
        or os.getenv("GEMINI_API_KEY_1")
        or os.getenv("Gemini_API_KEY_1", "")
    )
    if not key or key == "YOUR_NEW_GOOGLE_API_KEY":
        return []

    cache_key = f"gemini_{account_name}" if account_name else "gemini_default"

    try:
        client = genai.Client(api_key=key)
        models: List[Dict[str, Any]] = []
        for model in client.models.list():
            model_name = model.name
            if model_name.startswith("models/"):
                model_name = model_name[len("models/") :]

            supported_actions = getattr(model, "supported_actions", None)
            input_token_limit = getattr(model, "input_token_limit", None)
            description = getattr(model, "description", None)
            thinking = getattr(model, "thinking", None)

            models.append({
                "name": model_name,
                "supported_actions": supported_actions,
                "input_token_limit": input_token_limit,
                "description": description,
                "thinking": thinking,
            })

        if models and state_manager:
            state_manager.set_cached_discovery(cache_key, models, is_stale=False)

        return models
    except Exception as err:
        cat, detail = classify_discovery_error(err)
        if cat == DiscoveryErrorCategory.PROGRAMMING_ERROR:
            logger.error("[Gemini Discovery Error] Programming error: %s", err)
            raise err

        logger.warning(
            "[Gemini Discovery Error] %s for %s: %s",
            cat.value,
            account_name or "default",
            sanitize_secret(detail, key),
        )
        if state_manager:
            state_manager.mark_discovery_stale(
                cache_key,
                reason=detail,
                error_type=cat.value,
                last_error=sanitize_secret(str(err), key),
            )
            cached = state_manager.get_cached_discovery(cache_key)
            if cached and isinstance(cached, list):
                return cached
        return []


# Backward-compatible alias
discover_models = discover_gemini_models


def infer_gemini_capabilities(model_dict: Dict[str, Any]) -> ModelCapabilities:
    """Infer normalized ModelCapabilities for a Gemini model using authoritative SDK metadata."""
    input_limit = model_dict.get("input_token_limit")
    context_window = int(input_limit) if input_limit and input_limit > 0 else 4096

    sdk_thinking = model_dict.get("thinking")
    reasoning: Optional[bool] = bool(sdk_thinking) if sdk_thinking is not None else None

    return ModelCapabilities(
        coding=None,
        reasoning=reasoning,
        vision=None,
        tool_calling=None,
        structured_output=None,
        streaming=True,
        context_window=context_window,
    )


def supports_generate_content(model: Dict[str, Any]) -> bool:
    """Return True only for models capable of normal text content generation.

    Excludes embedding, image, TTS, audio, video, music, transcription, robotics, etc.
    """
    name = model["name"].lower()

    excluded_patterns = [
        "embedding",
        "embed",
        "imagen",
        "image",
        "tts",
        "text-to-speech",
        "speech",
        "audio",
        "veo",
        "music",
        "lyria",
        "transcribe",
        "robotics",
        "computer-use",
        "bidi",
        "live",
        "realtime",
    ]

    for pattern in excluded_patterns:
        if pattern in name:
            return False

    supported_actions = model.get("supported_actions")
    if supported_actions:
        normalized = {str(action).lower().replace("_", "") for action in supported_actions}
        if "generatecontent" in normalized:
            return True
        return False

    if "gemini" in name:
        return True

    return False


def build_gemini_deployments(
    keys: Optional[Union[str, Dict[str, str]]] = None,
    state_manager: Optional[Any] = None,
) -> List[Deployment]:
    """Dynamically discover and build Gemini deployments across configured Gemini accounts."""
    gemini_keys: Dict[str, str] = {}

    if isinstance(keys, str):
        if keys and keys != "YOUR_NEW_GOOGLE_API_KEY":
            gemini_keys = {"account_1": keys}
    elif isinstance(keys, dict):
        gemini_keys = {
            acc: k for acc, k in keys.items() if k and k != "YOUR_NEW_GOOGLE_API_KEY"
        }
    else:
        source_keys = GEMINI_KEYS or DEFAULT_GEMINI_KEYS
        gemini_keys = {
            acc: k for acc, k in source_keys.items() if k and k != "YOUR_NEW_GOOGLE_API_KEY"
        }
        if not gemini_keys and GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_NEW_GOOGLE_API_KEY":
            gemini_keys = {"account_1": GEMINI_API_KEY}

    if not gemini_keys:
        return []

    deployments: List[Deployment] = []

    try:
        for acc_name, key in gemini_keys.items():
            if not key or key == "YOUR_NEW_GOOGLE_API_KEY":
                continue

            cache_key = f"gemini_{acc_name}"
            discovered = discover_gemini_models(
                key,
                account_name=acc_name,
                state_manager=state_manager,
            )
            is_cache_stale = (
                state_manager.is_discovery_stale(cache_key) if state_manager else False
            )
            entry_meta = (
                state_manager.get_discovery_entry(cache_key) if state_manager else None
            )
            stale_reason = entry_meta.get("stale_reason") if entry_meta else None
            is_auth_err = (
                entry_meta.get("error_type") == "auth_failure" if entry_meta else False
            )

            seen_models = set()

            for model in discovered:
                if not supports_generate_content(model):
                    continue

                model_name = model["name"]
                if model_name in seen_models:
                    continue
                seen_models.add(model_name)

                caps = infer_gemini_capabilities(model)
                is_stale = model.get("is_stale", is_cache_stale)
                item_stale_reason = model.get("stale_reason", stale_reason)

                dep = Deployment(
                    id=f"gemini-{acc_name}-{model_name}",
                    provider=MODEL_GROUP_GEMINI,
                    account=acc_name,
                    model=model_name,
                    logical_model=MODEL_GROUP_GEMINI,
                    litellm_model=f"gemini/{model_name}",
                    api_key=key,
                    capabilities=caps,
                    quota=QuotaInfo(quota_scope=QuotaScope.ACCOUNT_MODEL),
                    is_stale=is_stale,
                    stale_reason=item_stale_reason,
                    circuit_state=(
                        CircuitState.AUTH_FAILED if is_auth_err else CircuitState.HEALTHY
                    ),
                    state_reason=(
                        (item_stale_reason or "Authentication failed during discovery")
                        if is_auth_err
                        else None
                    ),
                    last_discovered_at=time.time(),
                )
                apply_stored_state(dep, state_manager)
                deployments.append(dep)
    except (TypeError, AttributeError, NameError, SyntaxError):
        raise
    except Exception as err:
        logger.warning("Error building Gemini deployments: %s", sanitize_secret(str(err)))
        return []

    return deployments
