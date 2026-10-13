# Free text-model review — October 13, 2026

All 24 configured credentials successfully retrieved their model catalogs. Catalog presence confirms current identifiers, not a successful inference or remaining quota. Pools combine those catalogs with the providers' published free-plan rules and the user's confirmation that billing is disabled. Every account's selected models are recorded in [model-catalog-audit.json](model-catalog-audit.json); the reusable pool snapshot is [free-model-pools.json](free-model-pools.json).

| Provider | Accounts | Models per account | Quota handling |
| --- | ---: | ---: | --- |
| Gemini | 4 | 10 | Project/model cooldown; explicit project-wide quotas block the shared project |
| Groq | 5 | 3 | Organization/model cooldown; explicit shared organization quotas block the group |
| OpenRouter | 5 | 15 | Account-wide free caps; upstream model throttling blocks that model across keys |
| Cloudflare | 5 | 25 | Text-generation task limit and daily neuron allowance are shared by account |
| Cohere | 5 | 17 | Per-model chat cooldown; monthly trial exhaustion blocks the account/group |

## Gemini

The pool includes Gemini 3.8 Flash, 3.6 Flash, 3.5 Flash-Lite, 3.1 Flash-Lite, 3 Flash Preview, 2.5 Flash, 2.5 Flash-Lite, 2.5 Pro, and the two listed Gemma 4 instruction models. Each is returned by all four keys and has documented free text generation. Paid-only Pro previews, floating aliases, and image/audio/video-only endpoints are excluded. Catalogs can include models with no usable quota on a particular project; inference errors remain authoritative.

Quotas attach to projects rather than individual keys, and vary by model. Keys belonging to one project must use the same `GEMINI_N_QUOTA_GROUP`. A daily model quota does not disable other models on that project. A short RetryInfo delay accompanying daily exhaustion does not establish the daily reset.

Sources: [API pricing](https://ai.google.dev/gemini-api/docs/pricing), [quota scope](https://ai.google.dev/gemini-api/docs/rate-limits).

## Groq

The general chat pool is `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, and `qwen/qwen3.8-27b`, intersecting the Free Plan table with the live catalog. Enterprise Llama IDs are removed. Speech and safety-only models are unsuitable as general chat fallbacks. `allam-2-7b` is catalog-visible but absent from the published Free Plan table, so it is not selected.

Limits are model-specific at the organization level. Keys sharing an organization must share `GROQ_N_QUOTA_GROUP`; changing a key does not create an independent allowance. Provider reset headers remain authoritative, including rolling daily request resets.

Sources: [Free Plan table and headers](https://console.groq.com/docs/rate-limits), [current model tiers](https://console.groq.com/docs/models).

## OpenRouter

The pool includes `openrouter/free` and 14 catalog-listed `:free` text-output variants with zero metadata prices. Safety-only models are excluded. Account free limits cover the pool; changing models does not replenish those limits. Upstream model saturation is handled separately so the next model remains usable. The public catalog is shared across keys and does not verify an individual key's authentication or account permissions.

Sources: [free routing](https://openrouter.ai/docs/guides/routing/routers/free-router), [rate-limit scope](https://openrouter.ai/docs/api_reference/limits), [live catalog](https://openrouter.ai/api/v1/models).

## Cloudflare

The pool includes 25 catalog text-generation models without the `require_workers_paid` flag. The default is `@cf/meta/llama-3.1-8b-instruct-fp8`. Seven paid-only frontier models are excluded, along with safety-only and Clef decision models requiring specialized inputs. Catalog pagination was checked; later pages returned no additional visible models despite a larger total that includes hidden/retired entries.

All free models consume the account's shared daily neuron allocation. A generic text-generation rate limit is shared by the task; changing models cannot bypass it. Explicit per-model errors cool down only that model. Cloudflare keys are automatically grouped by a hash of the configured account ID.

Sources: [free allocation and paid exceptions](https://developers.cloudflare.com/workers-ai/platform/pricing/), [task/model limits](https://developers.cloudflare.com/workers-ai/platform/limits/).

## Cohere

All 17 current non-finetuned Chat models returned by the evaluation keys are included, covering Command, Aya, North and Tiny Aya. The default remains `command-a-03-2025`. Embedding, reranking and deprecated aliases are excluded. Translation-oriented models remain in the pool as requested, though their output may be less suitable for a general conversation. Evaluation keys are subject to the provider's trial-use terms.

Short-term chat limits are per model; the trial monthly call allowance is shared. A monthly cap skips remaining models for the affected group. Keys belonging to the same trial account must share `COHERE_N_QUOTA_GROUP`.

Sources: [trial and per-model limits](https://docs.cohere.com/docs/rate-limits), [model families](https://docs.cohere.com/docs/models), [trial access](https://docs.cohere.com/docs/cohere-faqs).

## Routing behavior

Starting providers rotate in the initial order Gemini → Groq → OpenRouter → Cloudflare → Cohere. Each provider rotates its accounts, and each account rotates its models. Callers can supply another order with `provider_order` or restrict/order providers through `providers` (`--provider` in the CLI).

Within a request, start from the reserved rotation position, try every available model on that account, then the remaining accounts of that provider, then the next provider. Shared cooldowns skip routes that cannot succeed. No quota counters are stored. The CLI persists rotation positions with a process lock; operational cooldowns remain in memory. An explicit configured model is tried first; automatic fallback then uses the configured free pools. An unlisted/deprecated model is not injected into a pool. `--exact-model` disables model changes; `--fallback-model` narrows fallback candidates. `--provider` remains a strict restriction.

The default attempt budget covers the whole selected pool, subject to the overall deadline (60 seconds by default) and transport timeouts (at most 15 seconds per attempt). A deadline can stop traversal before every route has been tried. Once a stream has emitted data, it is never replayed on another model.

Refresh catalog evidence with `python scripts/audit_models.py`. This read-only audit intersects live IDs with the reviewed snapshot; it does not rewrite `.env` or infer new models' free eligibility. Recheck official pricing/tier documentation before adding new IDs.

Model-specific cooldowns and account/group cooldowns are operational responses to actual API errors. No account is disabled because a local free-plan declaration is missing. Billing remains disabled in the provider accounts as confirmed by the user.

## Validation results

The deterministic test suite covers fair rotation with unequal pools, separate CLI-style processes and concurrent instances, alongside quota-aware fallback. Small live completion samples succeeded for Groq (`openai/gpt-oss-120b`), Cloudflare (`@cf/meta/llama-3.1-8b-instruct-fp8`), Cohere (`command-a-03-2025`), OpenRouter (free router selected `nvidia/nemotron-3-ultra-550b-a55b:free`) and Gemini (`gemini-3.6-flash` after one fallback). A follow-up check of Gemini's first route, `gemini-3.8-flash`, returned 503 server failure; it remains in the documented free pool and is cooled down operationally rather than treated as a nonexistent model. These samples verify the transport and actual fallback, not every selected model/account combination.
