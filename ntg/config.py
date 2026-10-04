"""Central NTG configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"


def _load_env_file(path: Path = ENV_FILE) -> None:
    """Load a tiny .env file without adding a dependency."""
    if not path.is_file():
        return

    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()

            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)

            key = key.strip()
            value = value.strip().strip("'\"")

            if key and key not in os.environ:
                os.environ[key] = value

    except OSError:
        return


_load_env_file()


@dataclass(frozen=True)
class OpenRouterSettings:
    free_model: str = "openrouter/openrouter/free"
    key_url: str = "https://openrouter.ai/api/v1/key"

    # OpenRouter free-tier limits
    free_requests_per_day: int = 50
    requests_per_minute: int = 20

    http_timeout_seconds: float = 10.0


@dataclass(frozen=True)
class LiteLLMSettings:
    # All account deployments belong to this same model group.
    model_name: str = "openrouter-free"

    # LiteLLM chooses the OpenRouter account deployment.
    routing_strategy: str = "usage-based-routing-v2"

    # Deliberately disabled to avoid wasting free quota.
    retries: int = 0

    rpm_per_deployment: int = 20

    cooldown_time_seconds: float = 30.0

    max_parallel_requests: int = 20


OPENROUTER = OpenRouterSettings()
LITELLM = LiteLLMSettings()


def load_api_keys(max_keys: int = 5) -> list[str]:
    """Load up to five OpenRouter API keys in numeric order."""

    keys: list[str] = []

    for index in range(1, max_keys + 1):
        value = os.getenv(f"OPENROUTER_API_KEY_{index}", "").strip()

        if value:
            keys.append(value)

    if not keys:
        raise RuntimeError(
            "No OpenRouter API keys configured. "
            "Set OPENROUTER_API_KEY_1 ... OPENROUTER_API_KEY_5 "
            "in .env or the environment."
        )

    return keys


OPENROUTER_FREE_MODEL = OPENROUTER.free_model
OPENROUTER_KEY_URL = OPENROUTER.key_url
LITELLM_MODEL_NAME = LITELLM.model_name

DEFAULT_KEYS = load_api_keys()