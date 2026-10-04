"""OpenRouter provider metadata and response parsing helpers."""

from typing import Any, Dict, Optional

from ntg.config import OPENROUTER_FREE_MODEL


def is_openrouter_free_model(model_name: str) -> bool:
    """Checks if model identifier targets OpenRouter free router."""
    return model_name == OPENROUTER_FREE_MODEL


def parse_provider_metadata(response_headers: Dict[str, Any]) -> Dict[str, Any]:
    """Parses OpenRouter specific rate-limit and provider headers."""
    return {
        "limit": response_headers.get("x-ratelimit-limit"),
        "remaining": response_headers.get("x-ratelimit-remaining"),
        "reset": response_headers.get("x-ratelimit-reset"),
    }