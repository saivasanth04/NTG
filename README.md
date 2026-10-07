# NTG: Multi-Provider LiteLLM Smart Router

A clean, production-grade, highly resilient multi-provider AI model router that unifies OpenRouter, Groq, NVIDIA NIM, Cohere, and Gemini into an intelligent routing fabric powered by **LiteLLM**.

---

## Architecture & Responsibilities

NTG cleanly divides responsibilities between infrastructure routing and domain intelligence:

```
┌────────────────────────────────────────────────────────┐
│                          NTG                           │
│  - Capability Filtering (coding, reasoning, vision...)  │
│  - Eligibility & Context Window Verification           │
│  - Circuit Breaker (HEALTHY ↔ OPEN ↔ HALF_OPEN)        │
│  - Quota Intelligence (RPM/RPD, account scopes)        │
│  - Atomic State Persistence (.ntg/state.json)          │
│  - Telemetry & Diagnostics (user vs upstream metrics)  │
└──────────────────────────┬─────────────────────────────┘
                           │ Dispatches eligible requests
┌──────────────────────────▼─────────────────────────────┐
│                     LiteLLM Router                     │
│  - Distributed Routing (simple-shuffle & weights)      │
│  - Bounded Retries (RetryPolicy, max 2 retries)        │
│  - Upstream Cooldown Cache                             │
│  - Compatible Fallback Chains                          │
└────────────────────────────────────────────────────────┘
```

---

## Key Features

1. **Single Routing Authority**: LiteLLM owns deployment selection, distributed load balancing, and failover. NTG enforces eligibility and tracks observability without competing routing engines.
2. **Real Global `auto` Routing**: Requesting `auto` or `ntg-auto` balances across **all** healthy deployments from all configured providers, not merely a single provider alias.
3. **Capability-Aware Routing**: Deployments declare normalized capabilities:
   - `coding`: High-capability coding models.
   - `reasoning`: Deep reasoning/thinking models (e.g., Nemotron, R1).
   - `vision`: Multimodal image processing models.
   - `tool_calling`: Function calling / tool execution support.
   - `context_window`: Verified token capacity.
4. **Resilient Circuit Breaker**:
   - `HEALTHY`: Serving traffic normally.
   - `OPEN`: Paused after failure or rate limits (authoritative `Retry-After` / reset duration).
   - `HALF_OPEN`: Controlled single-request probing before returning to `HEALTHY`.
   - `QUARANTINED`: Suspended on authentication (401/403) or missing model (404) errors.
5. **Accurate Quota & Error Handling**:
   - Parses HTTP `Retry-After` (seconds and RFC 2822 dates) and `X-RateLimit-*` headers.
   - Distinguishes RPM, RPD (daily free tier resets at 00:00 UTC), and upstream limits.
   - Propagates account-level limits to peer deployments sharing the same provider account.
   - 400 Bad Request errors are never retried.
6. **Thread-Safe Concurrency**: Zero shared mutable request state. Uses Python `contextvars` and event callbacks to guarantee atomic deployment attribution under high concurrency.
7. **Local State Persistence**: Circuit breaker status, cooldown expirations, and discovery caches are atomically persisted to `.ntg/state.json` across process restarts without external database dependencies.
8. **Dynamic Discovery with Caching**: Groq and Gemini models are discovered dynamically; discovery failures fall back to last-known-good cached models without disabling providers.
9. **Zero Secret Exposure**: API keys are never printed, logged, or exposed in diagnostics.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Configuration

Place your provider keys in `.env` (keys are loaded dynamically and never committed):

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

---

## Usage

### Interactive CLI

```bash
python main.py
```

### CLI with Model & Capability Flags

```bash
# Global auto load-balanced routing
python main.py "Explain the Byzantine Generals Problem"

# Target specific logical provider
python main.py --model groq "Generate a summary"
python main.py --model nvidia "Optimize this algorithm"
python main.py --model cohere "Draft a documentation outline"
python main.py --model openrouter "Analyze trade-offs"

# Target specific capability
python main.py --capability coding "Write a Red-Black tree in Rust"
python main.py --capability vision "Describe this architecture diagram"
python main.py --capability reasoning "Solve this logic puzzle step by step"

# View deployment health & circuit states
python main.py --status

# Reset persisted circuit breaker & metrics
python main.py --reset-state
```

### Programmatic Python API

```python
from ntg import UnifiedNTGRouter, ModelCapabilities

router = UnifiedNTGRouter()

# 1. Standard global routing (simple-shuffle across all healthy providers)
response = router.ask("Hello world!")

# 2. Capability-aware routing
response = router.ask(
    "Implement binary search",
    capabilities=ModelCapabilities(coding=True),
)

# 3. Provider-specific routing
response = router.ask("Fast inference request", model="groq")

# 4. OpenAI / LiteLLM standard chat completion
response = router.completion(
    messages=[{"role": "user", "content": "Explain concurrency"}],
    model="auto",
    capabilities={"coding": True},
)
```
