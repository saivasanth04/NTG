# NTG: Multi-Provider LiteLLM Smart Router

A clean, high-performance, self-adapting multi-provider router powered by **LiteLLM**. All configured accounts and discovered models across OpenRouter, Groq, NVIDIA, Cohere, and Gemini exist as independent deployments within a unified LiteLLM Router pool.

## Architecture & Features

- **LiteLLM as the Routing Authority**: Deployment selection, intra-provider load balancing, retries, and cross-provider fallbacks are managed directly by LiteLLM Router (`simple-shuffle` routing strategy).
- **Multi-Provider Deployments**:
  - **OpenRouter** (`openrouter`): 5 independent deployments using `nvidia/nemotron-3-ultra-550b-a55b:free`.
  - **Groq** (`groq`): Per-account dynamic model discovery automatically listing and filtering usable text/chat models.
  - **NVIDIA** (`nvidia`): Native NVIDIA NIM deployments using `nvidia/nemotron-3.5-lightning-30b-a3b`.
  - **Cohere** (`cohere`): Native Cohere deployments using `north-mini-code-1-0`.
  - **Gemini** (`gemini`): Dynamic discovery across configured Gemini accounts.
- **Fair Load Balancing**: Distributes requests fairly across healthy deployments instead of sequentially exhausting account 1 before account 2.
- **Isolated Deployment Health & Cooldowns**: Failures (rate limits, auth errors, server errors, quota exhaustion) cool down ONLY the affected deployment, preserving health of peer deployments.
- **Graceful Fault Tolerance**: Missing keys or temporary discovery errors on one key/provider do not crash initialization.
- **Privacy & Security**: Zero API keys or secrets are exposed in logs, diagnostics, or source code.

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Configure your provider API keys in a `.env` file or environment variables:

```bash
# OpenRouter (up to 5 accounts)
OPENROUTER_API_KEY_1=...
OPENROUTER_API_KEY_2=...

# Groq (up to 5 accounts, dynamic model discovery)
Grok_API_KEY_1=...

# NVIDIA NIM (up to 4 accounts)
Nvida_API_KEY_1=...

# Cohere (up to 5 accounts)
Cohere_API_KEY_1=...

# Gemini (up to 4 accounts)
Gemini_API_KEY_1=...
```

## Usage

### Interactive CLI

```bash
python main.py
```

### CLI with Direct Prompt and Model Selection

```bash
# Default logical model
python main.py "Explain quantum computing in simple terms"

# Specific provider logical group
python main.py --model groq "Explain quantum computing in simple terms"
python main.py --model nvidia "Write a quicksort implementation in Python"
python main.py --model cohere "Summarize this paragraph"
```

### Programmatic Python Usage

```python
from ntg import UnifiedNTGRouter

router = UnifiedNTGRouter()

# Route via default logical provider group
response = router.ask("Hello world!")

# Route via specific logical provider group
response = router.ask("Hello Groq!", model="groq")
response = router.ask("Hello NVIDIA!", model="nvidia")
response = router.ask("Hello Cohere!", model="cohere")
```

