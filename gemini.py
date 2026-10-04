"""Gemini dynamic model discovery and deployment builder."""

import os
from typing import Any, Dict, List, Optional
from google import genai

from ntg.config import _load_env_file, GEMINI_API_KEY
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


def build_gemini_deployments(api_key: Optional[str] = None) -> List[GeminiDeployment]:
    """Dynamically discover compatible Gemini models and build GeminiDeployment objects."""
    key = api_key or GEMINI_API_KEY or os.getenv("Gemini_API_KEY_1") or os.getenv("GEMINI_API_KEY_1", "")
    if not key or key == "YOUR_NEW_GOOGLE_API_KEY":
        return []

    discovered = discover_models(key)
    deployments = []
    seen = set()

    order = 1
    for model in discovered:
        if not supports_generate_content(model):
            continue

        model_name = model["name"]
        if model_name in seen:
            continue
        seen.add(model_name)

        deployment = GeminiDeployment(
            model_name=model_name,
            api_key=key,
            order=order,
        )
        deployments.append(deployment)
        order += 1

    return deployments


def main():
    """CLI entry point delegating to main unified router."""
    from main import main as run_unified_main
    run_unified_main()


if __name__ == "__main__":
    main()