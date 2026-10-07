"""Provider discovery and deployment building layer."""

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
from ntg.models import Deployment


def discover_groq_models(api_key: str) -> List[str]:
    """Dynamically discover usable text/chat models for a given Groq API key.

    Filters out embedding, audio, transcription, guard, and non-chat models.
    Returns an empty list on failure without exposing the API key.
    """
    if not api_key or not api_key.strip():
        return []

    url = "https://api.groq.com/openai/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "User-Agent": "NTG-LiteLLM-Router/1.0",
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return []

        payload = response.json()
        model_entries = payload.get("data", [])
        if not isinstance(model_entries, list):
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

        chat_models: List[str] = []
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
            chat_models.append(model_id)

        return chat_models

    except Exception:
        # Never log or expose API keys
        return []


def build_openrouter_deployments(
    keys: Optional[Dict[str, str]] = None,
) -> List[Deployment]:
    """Create 5 independent LiteLLM deployments for OpenRouter accounts."""
    keys_to_use = keys if keys is not None else OPENROUTER_KEYS
    deployments: List[Deployment] = []

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
            available=has_key,
        )
        deployments.append(deployment)

    return deployments


def build_groq_deployments(
    keys: Optional[Dict[str, str]] = None,
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
            discovered_models = discover_groq_models(key)
            for model_id in discovered_models:
                dep_id = f"groq-{acc_name}-{model_id}"
                deployment = Deployment(
                    id=dep_id,
                    provider=MODEL_GROUP_GROQ,
                    account=acc_name,
                    model=model_id,
                    logical_model=MODEL_GROUP_GROQ,
                    litellm_model=f"groq/{model_id}",
                    api_key=key,
                    available=True,
                )
                deployments.append(deployment)
        except Exception:
            # Skip only the failing key; do not crash initialization
            continue

    return deployments


def build_nvidia_deployments(
    keys: Optional[Dict[str, str]] = None,
) -> List[Deployment]:
    """Create one LiteLLM deployment per configured NVIDIA account using NVIDIA NIM."""
    keys_to_use = keys if keys is not None else NVIDIA_KEYS
    deployments: List[Deployment] = []

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
            available=True,
        )
        deployments.append(deployment)

    return deployments


def build_cohere_deployments(
    keys: Optional[Dict[str, str]] = None,
) -> List[Deployment]:
    """Create one LiteLLM deployment per configured Cohere account using native Cohere provider."""
    keys_to_use = keys if keys is not None else COHERE_KEYS
    deployments: List[Deployment] = []

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
            available=True,
        )
        deployments.append(deployment)

    return deployments


def build_gemini_deployments(
    keys: Optional[Dict[str, str]] = None,
) -> List[Deployment]:
    """Discover and build Gemini deployments across configured Gemini accounts."""
    keys_to_use = keys if keys is not None else GEMINI_KEYS
    active_keys = {acc: k for acc, k in keys_to_use.items() if k and k != "YOUR_NEW_GOOGLE_API_KEY"}
    if not active_keys:
        return []

    try:
        from gemini import build_gemini_deployments as gemini_builder
        raw_deps = gemini_builder(active_keys)
        deployments: List[Deployment] = []
        for gdep in raw_deps:
            # Wrap as provider-neutral Deployment if needed
            dep = Deployment(
                id=gdep.id,
                provider=MODEL_GROUP_GEMINI,
                account=gdep.account,
                model=gdep.model,
                logical_model=MODEL_GROUP_GEMINI,
                litellm_model=gdep.litellm_model,
                api_key=gdep.api_key,
                available=gdep.is_available(),
            )
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
) -> List[Deployment]:
    """Build all deployments across all configured providers."""
    all_deployments: List[Deployment] = []

    # 1. OpenRouter
    all_deployments.extend(build_openrouter_deployments(openrouter_keys))

    # 2. Groq
    all_deployments.extend(build_groq_deployments(groq_keys))

    # 3. NVIDIA
    all_deployments.extend(build_nvidia_deployments(nvidia_keys))

    # 4. Cohere
    all_deployments.extend(build_cohere_deployments(cohere_keys))

    # 5. Gemini (if configured)
    all_deployments.extend(build_gemini_deployments(gemini_keys))

    return all_deployments
