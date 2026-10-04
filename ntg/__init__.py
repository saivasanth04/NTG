"""NTG - Unified OpenRouter + Gemini Smart Router."""

from ntg.config import (
    DEFAULT_GEMINI_KEYS,
    DEFAULT_KEYS,
    GEMINI_API_KEY,
    LITELLM_MODEL_NAME,
    OPENROUTER_FREE_MODEL,
)
from ntg.diagnostics import print_account_status, print_banner, print_request_execution
from ntg.exceptions import (
    classify_gemini_error,
    classify_openrouter_error,
    extract_error_payload,
)
from ntg.models import Account, Deployment, GeminiDeployment, utc_string
from ntg.router import OpenRouterFallbackRouter, UnifiedNTGRouter

__all__ = [
    "Deployment",
    "Account",
    "GeminiDeployment",
    "UnifiedNTGRouter",
    "OpenRouterFallbackRouter",
    "OPENROUTER_FREE_MODEL",
    "LITELLM_MODEL_NAME",
    "DEFAULT_KEYS",
    "DEFAULT_GEMINI_KEYS",
    "GEMINI_API_KEY",
    "extract_error_payload",
    "classify_openrouter_error",
    "classify_gemini_error",
    "print_account_status",
    "print_banner",
    "print_request_execution",
    "utc_string",
]