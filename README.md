# NTG: Unified OpenRouter + Gemini Smart Router

A lightweight, dynamic, self-adapting smart router combining 5 OpenRouter accounts and dynamically discovered Google Gemini models into **one global routing pool** (`ntg-auto`).

## Architecture & Philosophy

- **Unified Global Pool (`ntg-auto`)**: All 5 OpenRouter accounts and all dynamically discovered, text-generation compatible Gemini models exist as peer deployments within a single global pool.
- **Deterministic Global Fallback Loop**: `UnifiedNTGRouter.ask()` explicitly manages deployment selection, failure classification, and per-deployment cooldowns. The router removes failed deployments from the current request's candidate pool and immediately attempts another healthy peer deployment.
- **Dynamic Gemini Discovery**: Automatically lists and filters models using the Google GenAI SDK (`google.genai`), excluding non-chat models (embeddings, image, TTS, audio, video, music, transcription, robotics, etc.).
- **Reactive Error Handling & Cooldown**: Directly reacts to real HTTP status codes and error payloads:
  - OpenRouter daily quota exhaustion -> Block until reported reset time
  - Authentication failure (401/403) -> 1-hour cooldown
  - Upstream server errors (500/502/503/504) -> 30-second cooldown
  - Upstream provider rate limits (429) -> 60-second cooldown
- **State Isolation**: When a deployment fails, only that specific deployment (OpenRouter account or Gemini model) enters cooldown. Other accounts and Gemini models remain active.

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Set your OpenRouter and Gemini API keys in environment variables or a `.env` file:

```bash
export OPENROUTER_API_KEY_1="sk-or-v1-..."
export OPENROUTER_API_KEY_2="sk-or-v1-..."
export Gemini_API_KEY_1="AIzaSy..."
```

## Usage

### Interactive CLI

```bash
python main.py
```

### CLI with Direct Prompt

```bash
python main.py "Explain quantum computing in simple terms"
```

### Programmatic Python Usage

```python
from ntg import UnifiedNTGRouter

router = UnifiedNTGRouter()
response = router.ask("Hello world!")
```
