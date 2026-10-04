"""NTG - Resilient OpenRouter Multi-Account Fallback Router with LiteLLM."""

from ntg.config import (
    DEFAULT_KEYS,
    LITELLM_MODEL_NAME,
    OPENROUTER_FREE_MODEL,
    OPENROUTER_KEY_URL,
)
from ntg.diagnostics import print_account_status, print_rate_limit_details
from ntg.exceptions import classify_openrouter_error, extract_error_payload
from ntg.models import Account, utc_string
from ntg.router import OpenRouterFallbackRouter

__all__ = [
    "Account",
    "OpenRouterFallbackRouter",
    "OPENROUTER_FREE_MODEL",
    "LITELLM_MODEL_NAME",
    "OPENROUTER_KEY_URL",
    "DEFAULT_KEYS",
    "extract_error_payload",
    "classify_openrouter_error",
    "print_account_status",
    "print_rate_limit_details",
    "utc_string",
]
