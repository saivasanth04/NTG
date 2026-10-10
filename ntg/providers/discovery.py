"""Unified multi-provider model discovery and deployment orchestration."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ntg.core.models import Deployment
from ntg.providers.base import (
    DiscoveryErrorCategory,
    _apply_stored_state,
    apply_stored_state,
    classify_discovery_error,
)
from ntg.providers.cohere import build_cohere_deployments, infer_cohere_capabilities
from ntg.providers.gemini import (
    build_gemini_deployments,
    discover_gemini_models,
    discover_models,
    infer_gemini_capabilities,
    supports_generate_content,
)
from ntg.providers.groq import (
    build_groq_deployments,
    discover_groq_models,
    infer_groq_capabilities,
)
from ntg.providers.nvidia import build_nvidia_deployments, infer_nvidia_capabilities
from ntg.providers.openrouter import (
    build_openrouter_deployments,
    infer_openrouter_capabilities,
)


def build_all_deployments(
    openrouter_keys: Optional[Dict[str, str]] = None,
    groq_keys: Optional[Dict[str, str]] = None,
    nvidia_keys: Optional[Dict[str, str]] = None,
    cohere_keys: Optional[Dict[str, str]] = None,
    gemini_keys: Optional[Dict[str, str]] = None,
    state_manager: Optional[Any] = None,
) -> List[Deployment]:
    """Build all deployments across all configured providers."""
    all_deployments: List[Deployment] = []

    # 1. OpenRouter
    all_deployments.extend(
        build_openrouter_deployments(openrouter_keys, state_manager=state_manager)
    )

    # 2. Groq
    all_deployments.extend(
        build_groq_deployments(groq_keys, state_manager=state_manager)
    )

    # 3. NVIDIA
    all_deployments.extend(
        build_nvidia_deployments(nvidia_keys, state_manager=state_manager)
    )

    # 4. Cohere
    all_deployments.extend(
        build_cohere_deployments(cohere_keys, state_manager=state_manager)
    )

    # 5. Gemini
    all_deployments.extend(
        build_gemini_deployments(gemini_keys, state_manager=state_manager)
    )

    return all_deployments


__all__ = [
    "DiscoveryErrorCategory",
    "classify_discovery_error",
    "apply_stored_state",
    "_apply_stored_state",
    "infer_openrouter_capabilities",
    "build_openrouter_deployments",
    "infer_groq_capabilities",
    "discover_groq_models",
    "build_groq_deployments",
    "infer_nvidia_capabilities",
    "build_nvidia_deployments",
    "infer_cohere_capabilities",
    "build_cohere_deployments",
    "discover_gemini_models",
    "discover_models",
    "infer_gemini_capabilities",
    "supports_generate_content",
    "build_gemini_deployments",
    "build_all_deployments",
]
