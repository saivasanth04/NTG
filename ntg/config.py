"""Configuration settings for OpenRouter multi-account router."""

import os
from typing import Dict

def _load_env_file(filepath: str = ".env") -> None:
    """Lightweight .env file loader without external dependencies."""
    if not os.path.exists(filepath):
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

# OpenRouter and LiteLLM model parameters
OPENROUTER_FREE_MODEL: str = "openrouter/openrouter/free"
LITELLM_MODEL_NAME: str = "openrouter-free"
OPENROUTER_KEY_URL: str = "https://openrouter.ai/api/v1/key"

# Account limit parameters
ACCOUNT_MAX_RPM: int = 20
ACCOUNT_MAX_RPD: int = 50

# Persistence directory and file
NTG_DIR: str = ".ntg"
HISTORY_FILE_PATH: str = os.path.join(NTG_DIR, "request_history.json")

# Retrieve up to 5 OpenRouter API keys from environment
DEFAULT_KEYS: Dict[str, str] = {}
for i in range(1, 6):
    key_name = f"account_{i}"
    env_val = os.getenv(f"OPENROUTER_API_KEY_{i}", "")
    if env_val:
        DEFAULT_KEYS[key_name] = env_val