# NTG: Dynamic OpenRouter Fallback Router

A lightweight, dynamic, self-adapting fallback router for OpenRouter models built on LiteLLM.

## Architecture & Philosophy

- **Five OpenRouter Deployments Behind One LiteLLM Logical Model**: All configured OpenRouter account API keys act as deployments registered under a single logical model (`openrouter-free`). LiteLLM handles deployment selection and retry management.
- **Reactive Real Error Failover**: Instead of predicting or maintaining model quotas locally, NTG reacts directly to real HTTP errors returned by OpenRouter:
  - Account daily quotas (`openrouter_free_tier_daily` / 429) -> Block account until reset
  - Authentication errors (401/403) -> 1-hour cooldown
  - Upstream server errors (500/502/503/504) -> 30-second cooldown
  - Upstream provider throttles -> 60-second cooldown
- **Dynamic Self-Adapting Router**: Small, clean codebase without complex local quota prediction counters or timestamp persistence.
- **Fast JSON Error Extraction**: Robust parsing of structured error responses from LiteLLM exceptions.
- **Modular Project Structure**:
  - `ntg/config.py`: Centralized model and API key configuration.
  - `ntg/models.py`: Strongly-typed `Account` data model.
  - `ntg/exceptions.py`: Real error classification engine.
  - `ntg/diagnostics.py`: Status tables and rate-limit diagnostics.
  - `ntg/router.py`: LiteLLM Router integration with reactive failure callbacks.
  - `main.py`: Interactive CLI entry point.

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Set your OpenRouter API keys in environment variables or a `.env` file:

```bash
export OPENROUTER_API_KEY_1="sk-or-v1-..."
export OPENROUTER_API_KEY_2="sk-or-v1-..."
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
from ntg import Account, OpenRouterFallbackRouter

accounts = [
    Account(name="primary", api_key="sk-or-...", order=1),
    Account(name="backup", api_key="sk-or-...", order=2),
]

router = OpenRouterFallbackRouter(accounts)
response = router.ask("Hello world!")
```
