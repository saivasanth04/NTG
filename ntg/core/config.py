"""Core configuration settings and environment loader for NTG."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


def load_env_file(filepath: str | Path = ".env") -> None:
    """Lightweight .env file loader without external dependencies."""
    candidate = Path(filepath)
    if not candidate.exists():
        candidate = PROJECT_ROOT / ".env"

    if candidate.exists():
        try:
            with candidate.open("r", encoding="utf-8") as f:
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


# Backward-compatible private alias
_load_env_file = load_env_file

# Load environment variables on import if present
load_env_file()


def get_env_key(key_candidates: List[str]) -> str:
    """Check a list of environment variable names and return the first non-empty value."""
    for key in key_candidates:
        val = os.getenv(key, "").strip().strip("'\"")
        if val:
            return val
    return ""


_get_env_key = get_env_key

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

# Bounded retry budget configuration (never derived from deployment count)
DEFAULT_NUM_RETRIES: int = 1
MAX_RETRY_BUDGET: int = 2

# Deployment circuit breaker configuration
CIRCUIT_BREAKER_MAX_FAILURES: int = int(os.getenv("NTG_CIRCUIT_BREAKER_MAX_FAILURES", "3"))
CIRCUIT_BREAKER_RECOVERY_TIME: float = float(os.getenv("NTG_CIRCUIT_BREAKER_RECOVERY_TIME", "30.0"))
CIRCUIT_BREAKER_HALF_OPEN_PROBES: int = 1

# Explicit cross-provider fallback mappings (empty by default so provider-specific
# requests never jump to unrelated providers unless explicitly configured)
PROVIDER_FALLBACKS: Dict[str, List[str]] = {}

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
    OPENROUTER_KEYS[f"account_{i}"] = get_env_key([
        f"OPENROUTER_API_KEY_{i}",
        f"OpenRouter_API_KEY_{i}",
    ])

# Load Groq API keys (up to 5 accounts)
GROQ_KEYS: Dict[str, str] = {}
for i in range(1, 6):
    GROQ_KEYS[f"account_{i}"] = get_env_key([
        f"GROQ_API_KEY_{i}",
        f"Groq_API_KEY_{i}",
        f"Grok_API_KEY_{i}",
        f"GROK_API_KEY_{i}",
    ])

# Load NVIDIA NIM API keys (up to 4 accounts)
NVIDIA_KEYS: Dict[str, str] = {}
for i in range(1, 5):
    NVIDIA_KEYS[f"account_{i}"] = get_env_key([
        f"NVIDIA_API_KEY_{i}",
        f"Nvidia_API_KEY_{i}",
        f"Nvida_API_KEY_{i}",
        f"NVIDA_API_KEY_{i}",
    ])

# Load Cohere API keys (up to 5 accounts)
COHERE_KEYS: Dict[str, str] = {}
for i in range(1, 6):
    COHERE_KEYS[f"account_{i}"] = get_env_key([
        f"COHERE_API_KEY_{i}",
        f"Cohere_API_KEY_{i}",
    ])

# Load Gemini API keys (up to 4 accounts)
GEMINI_KEYS: Dict[str, str] = {}
for i in range(1, 5):
    GEMINI_KEYS[f"account_{i}"] = get_env_key([
        f"GEMINI_API_KEY_{i}",
        f"Gemini_API_KEY_{i}",
    ])

# Backward-compatible references
DEFAULT_KEYS: Dict[str, str] = {k: v for k, v in OPENROUTER_KEYS.items() if v}
DEFAULT_GEMINI_KEYS: Dict[str, str] = {k: v for k, v in GEMINI_KEYS.items() if v}
GEMINI_API_KEY: str = GEMINI_KEYS.get("account_1", "")
