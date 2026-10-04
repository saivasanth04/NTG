"""NTG - Capacity-Aware OpenRouter Multi-Account Gateway."""

from ntg.config import (
    DEFAULT_KEYS,
    LITELLM_MODEL_NAME,
    OPENROUTER_FREE_MODEL,
)
from ntg.diagnostics import print_account_status, print_banner, print_request_execution
from ntg.exceptions import classify_openrouter_error, extract_error_payload
from ntg.models import Account, utc_string
from ntg.persistence import load_history, record_request_timestamp, save_history
from ntg.quota import calculate_rpd, calculate_rpm
from ntg.router import OpenRouterFallbackRouter

__all__ = [
    "Account",
    "OpenRouterFallbackRouter",
    "OPENROUTER_FREE_MODEL",
    "LITELLM_MODEL_NAME",
    "DEFAULT_KEYS",
    "calculate_rpm",
    "calculate_rpd",
    "load_history",
    "save_history",
    "record_request_timestamp",
    "extract_error_payload",
    "classify_openrouter_error",
    "print_account_status",
    "print_banner",
    "print_request_execution",
    "utc_string",
]