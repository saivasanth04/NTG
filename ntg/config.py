"""Configuration settings for multi-provider LiteLLM router."""

import os
from typing import Dict, List, Optional


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


def _get_env_key(key_candidates: List[str]) -> str:
    """Check a list of environment variable names and return the first non-empty value."""
    for key in key_candidates:
        val = os.getenv(key, "").strip().strip("'\"")
        if val:
            return val
    return ""


# Logical model group identifiers
MODEL_GROUP_AUTO: str = "auto"
MODEL_GROUP_NTG_AUTO: str = "ntg-auto"
MODEL_GROUP_OPENROUTER: str = "openrouter"
MODEL_GROUP_GROQ: str = "groq"
MODEL_GROUP_NVIDIA: str = "nvidia"
MODEL_GROUP_COHERE: str = "cohere"
MODEL_GROUP_GEMINI: str = "gemini"

# Default logical model for requests when none is specified (global multi-provider pool)
DEFAULT_LOGICAL_MODEL: str = MODEL_GROUP_AUTO

# OpenRouter configuration
OPENROUTER_ACTUAL_MODEL: str = "nvidia/nemotron-3-ultra-550b-a55b:free"
OPENROUTER_LITELLM_MODEL: str = f"openrouter/{OPENROUTER_ACTUAL_MODEL}"
OPENROUTER_FREE_MODEL: str = OPENROUTER_LITELLM_MODEL  # Backward compatibility
LITELLM_MODEL_NAME: str = DEFAULT_LOGICAL_MODEL  # Backward compatibility

# NVIDIA NIM configuration
NVIDIA_ACTUAL_MODEL: str = "nvidia/nemotron-3.5-lightning-30b-a3b"
NVIDIA_LITELLM_MODEL: str = f"nvidia_nim/{NVIDIA_ACTUAL_MODEL}"

# Cohere configuration
COHERE_ACTUAL_MODEL: str = "north-mini-code-1-0"
COHERE_LITELLM_MODEL: str = f"cohere/{COHERE_ACTUAL_MODEL}"

# Load OpenRouter API keys (up to 5 accounts)
OPENROUTER_KEYS: Dict[str, str] = {}
for i in range(1, 6):
    key_val = _get_env_key([f"OPENROUTER_API_KEY_{i}", f"OpenRouter_API_KEY_{i}"])
    OPENROUTER_KEYS[f"account_{i}"] = key_val

# Load Groq API keys (up to 5 accounts)
GROQ_KEYS: Dict[str, str] = {}
for i in range(1, 6):
    key_val = _get_env_key([
        f"Grok_API_KEY_{i}",
        f"GROK_API_KEY_{i}",
        f"GROQ_API_KEY_{i}",
        f"Groq_API_KEY_{i}",
    ])
    GROQ_KEYS[f"account_{i}"] = key_val

# Load NVIDIA NIM API keys (up to 4 accounts)
NVIDIA_KEYS: Dict[str, str] = {}
for i in range(1, 5):
    key_val = _get_env_key([
        f"Nvida_API_KEY_{i}",
        f"NVIDA_API_KEY_{i}",
        f"NVIDIA_API_KEY_{i}",
        f"Nvidia_API_KEY_{i}",
    ])
    NVIDIA_KEYS[f"account_{i}"] = key_val

# Load Cohere API keys (up to 5 accounts)
COHERE_KEYS: Dict[str, str] = {}
for i in range(1, 6):
    key_val = _get_env_key([f"Cohere_API_KEY_{i}", f"COHERE_API_KEY_{i}"])
    COHERE_KEYS[f"account_{i}"] = key_val

# Load Gemini API keys (up to 4 accounts)
GEMINI_KEYS: Dict[str, str] = {}
for i in range(1, 5):
    key_val = _get_env_key([f"Gemini_API_KEY_{i}", f"GEMINI_API_KEY_{i}"])
    GEMINI_KEYS[f"account_{i}"] = key_val

# Backward-compatible references
DEFAULT_KEYS: Dict[str, str] = {k: v for k, v in OPENROUTER_KEYS.items() if v}
DEFAULT_GEMINI_KEYS: Dict[str, str] = {k: v for k, v in GEMINI_KEYS.items() if v}
GEMINI_API_KEY: str = GEMINI_KEYS.get("account_1", "")