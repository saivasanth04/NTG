"""Public NTG package API."""

from ntg.config import (
    DEFAULT_KEYS,
    LITELLM,
    LITELLM_MODEL_NAME,
    OPENROUTER,
    OPENROUTER_FREE_MODEL,
    OPENROUTER_KEY_URL,
)

from ntg.diagnostics import (
    print_account_status,
    print_divider,
    print_pool_summary,
    print_rate_limit_details,
)

from ntg.exceptions import (
    classify_openrouter_error,
    extract_error_payload,
)

from ntg.models import (
    Account,
    utc_string,
)

from ntg.router import (
    OpenRouterFallbackRouter,
)


__all__ = [
    "Account",
    "OpenRouterFallbackRouter",
    "OPENROUTER",
    "LITELLM",
    "OPENROUTER_FREE_MODEL",
    "OPENROUTER_KEY_URL",
    "LITELLM_MODEL_NAME",
    "DEFAULT_KEYS",
    "classify_openrouter_error",
    "extract_error_payload",
    "print_account_status",
    "print_divider",
    "print_pool_summary",
    "print_rate_limit_details",
    "utc_string",
]