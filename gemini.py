"""Gemini dynamic model discovery and deployment builder."""

import os
from typing import Any, Dict, List, Optional, Union
from google import genai

from ntg.config import _load_env_file, DEFAULT_GEMINI_KEYS, GEMINI_API_KEY
from ntg.models import GeminiDeployment

_load_env_file()


def discover_models(api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """Discover models available to the Google API key using Google SDK."""
    key = api_key or GEMINI_API_KEY or os.getenv("Gemini_API_KEY_1") or os.getenv("GEMINI_API_KEY_1", "")
    if not key or key == "YOUR_NEW_GOOGLE_API_KEY":
        return []

    try:
        client = genai.Client(api_key=key)
        models = []
        for model in client.models.list():
            model_name = model.name
            if model_name.startswith("models/"):
                model_name = model_name[len("models/"):]

            supported_actions = getattr(model, "supported_actions", None)
            models.append({
                "name": model_name,
                "supported_actions": supported_actions,
            })
        return models
    except Exception as err:
        print(f"[Gemini Discovery Error] Failed to list models: {err}")
        return []


def supports_generate_content(model: Dict[str, Any]) -> bool:
    """Return True only for models capable of normal text content generation.

    Excludes embedding, image, TTS, audio, video, music, transcription, robotics, etc.
    """
    name = model["name"].lower()

    excluded_patterns = [
        "embedding",
        "embed",
        "imagen",
        "image",
        "tts",
        "text-to-speech",
        "speech",
        "audio",
        "veo",
        "music",
        "lyria",
        "transcribe",
        "robotics",
        "computer-use",
        "bidi",
        "live",
        "realtime",
    ]

    for pattern in excluded_patterns:
        if pattern in name:
            return False

    supported_actions = model.get("supported_actions")
    if supported_actions:
        normalized = {
            str(action).lower().replace("_", "")
            for action in supported_actions
        }
        if "generatecontent" in normalized:
            return True
        return False

    if "gemini" in name:
        return True

    return False


def build_gemini_deployments(
    keys_input: Optional[Union[str, Dict[str, str]]] = None
) -> List[GeminiDeployment]:
    """Dynamically discover compatible Gemini models for EACH configured Gemini API key."""
    gemini_keys: Dict[str, str] = {}

    if isinstance(keys_input, str):
        if keys_input and keys_input != "YOUR_NEW_GOOGLE_API_KEY":
            gemini_keys = {"account_1": keys_input}
    elif isinstance(keys_input, dict):
        gemini_keys = keys_input
    else:
        gemini_keys = DEFAULT_GEMINI_KEYS
        if not gemini_keys and GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_NEW_GOOGLE_API_KEY":
            gemini_keys = {"account_1": GEMINI_API_KEY}

    deployments: List[GeminiDeployment] = []
    order = 1

    for acc_name, key in gemini_keys.items():
        if not key or key == "YOUR_NEW_GOOGLE_API_KEY":
            continue

        discovered = discover_models(key)
        seen_models = set()

        for model in discovered:
            if not supports_generate_content(model):
                continue

            model_name = model["name"]
            if model_name in seen_models:
                continue
            seen_models.add(model_name)

            deployment = GeminiDeployment(
                account_name=acc_name,
                model_name=model_name,
                api_key=key,
                order=order,
            )
            deployments.append(deployment)
            order += 1

    return deployments