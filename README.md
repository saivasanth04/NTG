# NTG: Resilient OpenRouter Multi-Account Fallback Router

A high-reliability, multi-account fallback routing system for OpenRouter free models using LiteLLM.

## Architecture & Features

- **Multi-Account Cooldown & Quota Management**: Tracks rate limits, daily quotas, and automatic cooldown recovery per account.
- **Granular Error Classification**: Distinguishes between:
  - Account daily quotas (`openrouter_free_tier_daily`)
  - Upstream provider-level throttles
  - Authentication errors (401/403)
  - Upstream server errors (500/502/503/504)
- **Zero-Waste Fallback Routing**: Prevents burning secondary accounts on upstream provider throttles, while smoothly failing over on account-specific quota exhaustion.
- **Fast $O(N)$ JSON Payload Extraction**: Robust parsing of structured error responses from LiteLLM exceptions without performance bottlenecks.
- **Modular Project Structure**:
  - `ntg/config.py`: Centralized model and API key configuration.
  - `ntg/models.py`: Strongly-typed `Account` data model.
  - `ntg/exceptions.py`: Error parsing and rate-limit classification.
  - `ntg/diagnostics.py`: Status tables and rate-limit diagnostics.
  - `ntg/router.py`: Priority failover engine built on LiteLLM Router.
  - `main.py`: Command-line interface.
  - `test.py`: Backwards-compatible testing module.

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
