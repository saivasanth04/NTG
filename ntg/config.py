"""Configuration settings for OpenRouter multi-account router."""

import os
from typing import List


def _load_env_file(filepath: str = ".env") -> None:
    """Lightweight .env file loader without external dependencies."""
    if not os.path.exists(filepath):
        # Look in parent directories if not in current working dir
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(base_dir, ".env")

    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, val = line.split("=", 1)
                        key = key.strip()
                        val = val.strip().strip("'\"")
                        if key and key not in os.environ:
                            os.environ[key] = val
        except Exception:
            pass


# Load environment variables if present
_load_env_file()

# Model identifiers
OPENROUTER_FREE_MODEL: str = "openrouter/openrouter/free"
LITELLM_MODEL_NAME: str = "openrouter-free"
OPENROUTER_KEY_URL: str = "https://openrouter.ai/api/v1/key"

# Default API keys loaded from environment
OPENROUTER_API_KEY_1: str = os.getenv("OPENROUTER_API_KEY_1", "")
OPENROUTER_API_KEY_2: str = os.getenv("OPENROUTER_API_KEY_2", "")

DEFAULT_KEYS: List[str] = [
    key for key in [OPENROUTER_API_KEY_1, OPENROUTER_API_KEY_2] if key
]
