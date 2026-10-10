"""Provider adapters, model discovery, and deployment builders."""

from ntg.providers.base import (
    DiscoveryErrorCategory,
    _apply_stored_state,
    apply_stored_state,
    classify_discovery_error,
)
from ntg.providers.cohere import build_cohere_deployments, infer_cohere_capabilities
from ntg.providers.discovery import build_all_deployments
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
