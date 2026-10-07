"""Provider discovery and deployment building layer."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Set
import requests

from ntg.config import (
    COHERE_ACTUAL_MODEL,
    COHERE_KEYS,
    COHERE_LITELLM_MODEL,
    GEMINI_KEYS,
    GROQ_KEYS,
    MODEL_GROUP_COHERE,
    MODEL_GROUP_GEMINI,
    MODEL_GROUP_GROQ,
    MODEL_GROUP_NVIDIA,
    MODEL_GROUP_OPENROUTER,
    NVIDIA_ACTUAL_MODEL,
    NVIDIA_KEYS,
    NVIDIA_LITELLM_MODEL,
    OPENROUTER_ACTUAL_MODEL,
    OPENROUTER_KEYS,
    OPENROUTER_LITELLM_MODEL,
)
from ntg.models import CircuitState, Deployment, ModelCapabilities, QuotaInfo
from ntg.state import StateManager


def infer_groq_capabilities(model_id: str, context_window: int = 8192) -> ModelCapabilities:
    """Infer capabilities for a Groq model using provider metadata and model characteristics."""
    m_lower = model_id.lower()

    # Tool calling support on Groq
    tool_calling = any(
        kw in m_lower
        for kw in (
            "tool",
            "llama-3.3-70b",
            "llama-3.1-8b",
            "llama-3.1-70b",
            "mixtral-8x7b",
            "qwen-2.5",
        )
    )

    # Coding capabilities
    coding = any(
        kw in m_lower
        for kw in ("coder", "code", "llama-3.3", "llama-3.1-70b", "deepseek", "qwen-2.5")
    )

    # Reasoning / thinking capabilities
    reasoning = any(
        kw in m_lower
        for kw in ("r1", "reasoning", "thinking", "deepseek", "70b")
    )

    # Vision capabilities
    vision = any(kw in m_lower for kw in ("vision", "llava", "scenecap"))

    return ModelCapabilities(
        coding=coding,
        reasoning=reasoning,
        vision=vision,
        tool_calling=tool_calling,
        structured_output=True,
        streaming=True,
        context_window=context_window,
    )


def discover_groq_models(
    api_key: str,
    account_name: str = "",
    state_manager: Optional[StateManager] = None,
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
        if response.status_code == 200:
            payload = response.json()
            model_entries = payload.get("data", [])
            if isinstance(model_entries, list):
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
                    state_manager.set_cached_discovery(cache_key, discovered)

                return discovered

    except Exception:
        # Network/timeout during discovery - fall back to cache below
        pass

    # Fallback to last-known-good cached models if available
    if state_manager:
        cached = state_manager.get_cached_discovery(cache_key)
        if cached and isinstance(cached, list):
            return cached

    return []


def _apply_stored_state(deployment: Deployment, state_manager: Optional[StateManager]) -> None:
    """Restore persisted circuit state, cooldown, and metrics from StateManager."""
    if not state_manager:
        return

    state_manager.apply_to_deployment(deployment)
    deployment.metrics = state_manager.get_metrics(deployment.id)


def build_openrouter_deployments(
    keys: Optional[Dict[str, str]] = None,
    state_manager: Optional[StateManager] = None,
) -> List[Deployment]:
    """Create 5 independent LiteLLM deployments for OpenRouter accounts."""
    keys_to_use = keys if keys is not None else OPENROUTER_KEYS
    deployments: List[Deployment] = []

    # Model capabilities for OpenRouter nemotron
    capabilities = ModelCapabilities(
        coding=True,
        reasoning=True,
        vision=False,
        tool_calling=False,
        structured_output=True,
        streaming=True,
        context_window=131072,
    )

    for i in range(1, 6):
        acc_name = f"account_{i}"
        key = keys_to_use.get(acc_name, "").strip()
        has_key = bool(key)

        deployment = Deployment(
            id=f"openrouter-{acc_name}",
            provider=MODEL_GROUP_OPENROUTER,
            account=acc_name,
            model=OPENROUTER_ACTUAL_MODEL,
            logical_model=MODEL_GROUP_OPENROUTER,
            litellm_model=OPENROUTER_LITELLM_MODEL,
            api_key=key,
            capabilities=capabilities,
            quota=QuotaInfo(quota_scope="account"),
        )
        _apply_stored_state(deployment, state_manager)
        deployments.append(deployment)

    return deployments


def build_groq_deployments(
    keys: Optional[Dict[str, str]] = None,
    state_manager: Optional[StateManager] = None,
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
            discovered_models = discover_groq_models(key, account_name=acc_name, state_manager=state_manager)
            for item in discovered_models:
                if isinstance(item, dict):
                    model_id = item.get("id", "")
                    context_window = item.get("context_window", 8192)
                else:
                    model_id = str(item)
                    context_window = 8192

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
                    quota=QuotaInfo(quota_scope="deployment"),
                )
                _apply_stored_state(deployment, state_manager)
                deployments.append(deployment)
        except Exception:
            # Skip failing key; do not crash router initialization
            continue

    return deployments


def build_nvidia_deployments(
    keys: Optional[Dict[str, str]] = None,
    state_manager: Optional[StateManager] = None,
) -> List[Deployment]:
    """Create one LiteLLM deployment per configured NVIDIA account using NVIDIA NIM."""
    keys_to_use = keys if keys is not None else NVIDIA_KEYS
    deployments: List[Deployment] = []

    capabilities = ModelCapabilities(
        coding=True,
        reasoning=True,
        vision=False,
        tool_calling=True,
        structured_output=True,
        streaming=True,
        context_window=131072,
    )

    for i in range(1, 5):
        acc_name = f"account_{i}"
        key = keys_to_use.get(acc_name, "").strip()
        if not key:
            continue

        deployment = Deployment(
            id=f"nvidia-{acc_name}",
            provider=MODEL_GROUP_NVIDIA,
            account=acc_name,
            model=NVIDIA_ACTUAL_MODEL,
            logical_model=MODEL_GROUP_NVIDIA,
            litellm_model=NVIDIA_LITELLM_MODEL,
            api_key=key,
            api_base="https://integrate.api.nvidia.com/v1",
            capabilities=capabilities,
            quota=QuotaInfo(quota_scope="account"),
        )
        _apply_stored_state(deployment, state_manager)
        deployments.append(deployment)

    return deployments


def build_cohere_deployments(
    keys: Optional[Dict[str, str]] = None,
    state_manager: Optional[StateManager] = None,
) -> List[Deployment]:
    """Create one LiteLLM deployment per configured Cohere account using native Cohere provider."""
    keys_to_use = keys if keys is not None else COHERE_KEYS
    deployments: List[Deployment] = []

    capabilities = ModelCapabilities(
        coding=True,
        reasoning=True,
        vision=False,
        tool_calling=True,
        structured_output=True,
        streaming=True,
        context_window=128000,
    )

    for i in range(1, 6):
        acc_name = f"account_{i}"
        key = keys_to_use.get(acc_name, "").strip()
        if not key:
            continue

        deployment = Deployment(
            id=f"cohere-{acc_name}",
            provider=MODEL_GROUP_COHERE,
            account=acc_name,
            model=COHERE_ACTUAL_MODEL,
            logical_model=MODEL_GROUP_COHERE,
            litellm_model=COHERE_LITELLM_MODEL,
            api_key=key,
            capabilities=capabilities,
            quota=QuotaInfo(quota_scope="account"),
        )
        _apply_stored_state(deployment, state_manager)
        deployments.append(deployment)

    return deployments


def build_gemini_deployments(
    keys: Optional[Dict[str, str]] = None,
    state_manager: Optional[StateManager] = None,
) -> List[Deployment]:
    """Discover and build Gemini deployments across configured Gemini accounts."""
    keys_to_use = keys if keys is not None else GEMINI_KEYS
    active_keys = {acc: k for acc, k in keys_to_use.items() if k and k != "YOUR_NEW_GOOGLE_API_KEY"}
    if not active_keys:
        return []

    gemini_caps = ModelCapabilities(
        coding=True,
        reasoning=True,
        vision=True,
        tool_calling=True,
        structured_output=True,
        streaming=True,
        context_window=1000000,
    )

    try:
        from gemini import build_gemini_deployments as gemini_builder

        raw_deps = gemini_builder(active_keys)
        deployments: List[Deployment] = []
        for gdep in raw_deps:
            dep = Deployment(
                id=gdep.id,
                provider=MODEL_GROUP_GEMINI,
                account=gdep.account,
                model=gdep.model,
                logical_model=MODEL_GROUP_GEMINI,
                litellm_model=gdep.litellm_model,
                api_key=gdep.api_key,
                capabilities=gemini_caps,
                quota=QuotaInfo(quota_scope="account"),
            )
            _apply_stored_state(dep, state_manager)
            deployments.append(dep)
        return deployments
    except Exception:
        return []


def build_all_deployments(
    openrouter_keys: Optional[Dict[str, str]] = None,
    groq_keys: Optional[Dict[str, str]] = None,
    nvidia_keys: Optional[Dict[str, str]] = None,
    cohere_keys: Optional[Dict[str, str]] = None,
    gemini_keys: Optional[Dict[str, str]] = None,
    state_manager: Optional[StateManager] = None,
) -> List[Deployment]:
    """Build all deployments across all configured providers."""
    all_deployments: List[Deployment] = []

    # 1. OpenRouter
    all_deployments.extend(build_openrouter_deployments(openrouter_keys, state_manager=state_manager))

    # 2. Groq
    all_deployments.extend(build_groq_deployments(groq_keys, state_manager=state_manager))

    # 3. NVIDIA
    all_deployments.extend(build_nvidia_deployments(nvidia_keys, state_manager=state_manager))

    # 4. Cohere
    all_deployments.extend(build_cohere_deployments(cohere_keys, state_manager=state_manager))

    # 5. Gemini (if configured)
    all_deployments.extend(build_gemini_deployments(gemini_keys, state_manager=state_manager))

    return all_deployments
