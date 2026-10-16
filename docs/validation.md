# Validation results

Measured during the 16–17 October 2026 work session. These results are a small observation window, not a reliability benchmark or future availability prediction.

## Controlled tests

`.venv\Scripts\python -m pytest -q` passed **36 tests** in **4.73 seconds** on the final recorded rerun. Tests use injected transports and mocked HTTP, with no inference charges. Coverage includes:

- Arbitrary account counts (40 slots), duplicate-key quota grouping, unknown and nonzero pricing rejected.
- Discovery pagination, access/free separation, expired catalog evidence.
- Tool/schema/context filtering and preservation during fallback.
- Fair account distribution: 20 sequential requests split **10 / 10** between an account with ten models and an account with one model.
- 401 account, 402/429 group, 403/404 deployment, 502 provider, timeout/overload deployment cooldown scopes; invalid requests stop without retry.
- Retry-After numeric/date and reset-duration parsing; successful exhausted-token headers preemptively cool the group.
- Shared quota concurrency: 12 simultaneous callers in one group observed **peak one** active provider call.
- Queue overflow, waiting deadline, caller cancellation and reservation cleanup; queue remains bounded even when group capacity is below global capacity.
- Timeout fallback and single-deployment bounded retry after cooldown.
- JSON/schema response validation and fallback.
- Authentication, sanitized validation errors, OpenAI response shape; real OpenAI SDK against an ASGI gateway with simulated provider output.
- LiteLLM transport arguments, disabled nested retries/caching and OpenRouter zero-price constraints.
- Vision requirements inferred from image inputs; mandatory tool calls cannot silently return plain text; provider unsupported-parameter errors permit compatible fallback; a policy override cannot admit paid OpenRouter models.

The isolated virtual environment installed successfully and `pip check` reported no broken dependencies. An exact-value scan compared configured API keys/account identifiers with 32 deliverable/runtime files and found no leaks. The scan reports filenames only and can be repeated with `python scripts/check_secrets.py`.

## Live discovery

All 24 configured account slots completed metadata discovery. These are catalog results, not inference access tests:

| Provider | Slots | Catalog models per account | Configured names absent per account |
|---|---:|---:|---:|
| Gemini | 4 | 63 | 0 |
| Groq | 5 | 11 | 0 |
| OpenRouter | 5 | 466 | 1 |
| Cloudflare | 5 | 71 | 0 |
| Cohere | 5 | 20 | 6 |

After allowlist intersection there were **305 deployment records** and **65 automatically free-eligible records**, all OpenRouter. The absent OpenRouter name was `apodex/apodex-1.1-mini:free`. Cohere's six missing configured names were the North Mini Code / North Small Translate names and four Tiny Aya variants. They were not admitted by guessing aliases. Cloudflare matching was verified against its model `name` field, not internal UUID.

OpenRouter `/api/v1/key` reported `is_free_tier=true` for all five slots and a free-model daily ceiling of 50, with observed remaining counts ranging from 31 to 37 in one snapshot. These figures are observations only: they are not hardcoded or reserved, can change outside this process, and do not prove independent owner quotas. Other provider billing status was **not verified**, so inference remained disabled for those providers.

Secret-free local metadata is in ignored `runtime/catalog.json`. Catalog prices and capabilities can change after this report.

## Live inference

The initial targeted smoke call to `cohere/north-mini-code:free` succeeded through LiteLLM in **4.413s**, reporting 36 total tokens and provider cost zero. This was a transport response, not a coding benchmark.

The subsequent bounded `scripts/validate_live.py` run produced:

| Probe | Observed result | Logical-request latency |
|---|---|---:|
| Chat | Inaccessible `thinkingmachines/inkling-small:free` deployment, then successful fallback to `liquid/lfm-2.5-2.6b:free` on another account | 0.861s |
| Required tool calling | Three attempts exhausted: inaccessible model, rate limit, inaccessible model. No tool-call success claimed. | 0.549s |
| JSON schema | `dots-studio/dots-3-note-preview:free` succeeded; JSON parsed and validated against the requested schema | 2.824s |
| Concurrent caller 1 | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free` succeeded | 1.357s |
| Concurrent caller 2 | `dots-studio/dots-3-note-preview:free` succeeded | 2.016s |

Four of those five logical requests succeeded. Every successful response reported provider cost zero. Successful output may be truncated at the small probe output cap; success here means a normalized API response (and schema validation when requested), not a quality assessment. Live fallback, inaccessible models and one 429 were observed. Timeouts, invalid keys, full-queue conditions and broad outages were simulated rather than deliberately induced against providers. No live tool execution occurred.

Detailed safe routing histories/usage remain in ignored `runtime/live-validation.json`; prompts, model text and credentials were not saved. That run preceded a small metadata improvement that maps the provider-reported cost into `routing.cost_usd`; original artifacts therefore show `cost_usd=null` while their usage records contain `cost=0`.

## Remaining limitations

- Live inference was limited to explicit zero-priced OpenRouter variants. Gemini, Groq, Cohere and Cloudflare adapters were not live-inference verified because free-only billing evidence was absent.
- No sustained load, coding correctness, end-to-end multi-agent tool loop or long-context benchmark was run. The live tool-calling probe failed within its retry budget.
- Cooldown scope defaults are conservative; provider error ambiguity cannot always distinguish model and account restrictions.
- No invoice audit, account ownership audit, TLS remote deployment, persistent quotas, dashboard, embeddings or image-generation validation is claimed.
- In-memory state and single-process coordination are deliberate limitations. Restarts lose runtime state; catalog evidence expires after 24 hours.
