# Smart routing audit and design

## Audit findings

The former router had several independent owners of a single request:

* LiteLLM shuffle selected the first deployment; NTG shuffled accounts and built
  another fallback ordering. Ranking ignored available quota, latency, and load.
* Private `_completion`, `should_retry_this_error`, and retry sleep methods were
  replaced on a live LiteLLM Router. Selection, cooldowns and retry budgets could
  consequently disagree with NTG's own state.
* Pool construction reserved HALF_OPEN probes on every candidate. Unused
  candidates could consume all recovery capacity. Coding preferences removed
  otherwise valid deployments instead of changing their score.
* Callbacks inferred attempts from event counts and thread-local state, then
  synthesized callbacks when events appeared missing. Asynchronous callbacks
  could count twice, attribute events to another router, or arrive after teardown.
* Every classified failure updated quota fields, including server failures.
  Ordinary successful-response reset headers also made deployments unavailable.
  Remaining quota was not reserved at dispatch, and exhausted counters were not
  properly refreshed when windows expired.
* Success callbacks unconditionally reset health and quota. Old successes could
  erase a newer auth quarantine or account cooldown. Quota propagation could
  replace auth quarantine, and a matching API key could propagate quota between
  different providers.
* Caller fallback mappings were flattened without respecting their sources;
  global fallback destinations were accepted but never expanded.
* Context estimation counted only message text divided by four. It omitted tool
  schemas, tool call arguments and Unicode overhead; arbitrary `base64,` text was
  treated as an image. Misspelled capability names silently disappeared.
* Discovery often left all protocol capabilities unknown even where the installed
  LiteLLM metadata had explicit evidence. Unknown coding suitability was treated
  inconsistently depending on whether any other deployment advertised coding.
* Persistence wrote portions of one outcome separately, did not restore failure
  streaks or credential fingerprints, and assumed numeric JSON fields were valid.
  The default Deployment repr exposed the API key.
* The agent caught runtime router validation errors inside a signature-inspection
  handler and invoked the router again without capability requirements.

## Ownership and data flow

`requirements.py` normalizes request semantics. `policy.py` implements capability
compatibility, model matching, quota boundaries, and an explicit score.
`engine.py` compiles provider/model tiers and executes a single bounded attempt
loop. `lifecycle.py` atomically admits each selected attempt and applies its
outcome. `state.py` persists coherent snapshots.

LiteLLM's public `completion()` function adapts provider requests. Its nested
retries are disabled. The existing `router` attribute and model registry remain
for compatibility, but they do not choose deployments or execute NTG requests.
No private LiteLLM methods are replaced. NTG's compatibility logger is not added
to global callback lists, and ignores managed, foreign, unidentified, and closed
requests. Managed attempt metrics and state are recorded synchronously.

Planning acquires no probe slots. Only dispatch claims a HALF_OPEN probe and
reserves known RPM/RPD remaining counts within their recorded scope. Routing,
admission, topology mutations and outcome application share one process-local
reentrant lock. Provider I/O happens outside that lock. The executor rechecks
eligibility and topology before every attempt.

## Requirements and ranking

Wire capabilities (tools, schema output, images, audio and streaming) require
explicit positive evidence. Provider discovery is authoritative; unknown fields
can be filled from the installed LiteLLM model metadata, without a new network
lookup or model-name substring guesses. Unknown protocol support never passes
eligibility. Unsupported native file/video inputs fail explicitly.

Coding is task suitability rather than a distinct protocol. An explicit coding
request excludes known negative suitability, while unknown suitability remains
usable with a lower score. This preserves the agent's coding integration without
pretending that unknown means verified. Reasoning requests require positive
evidence. Soft coding hints retain all otherwise compatible fallback candidates.

Context estimates include the serialized request, schemas and completion budget.
The dependency-free text estimate uses UTF-8 byte counts as a conservative BPE
upper bound. Image/audio parts have a 4,096-token planning allowance instead of
counting base64 bytes. These are estimates, not provider tokenization guarantees;
callers can supply `min_context` to reserve additional multimodal capacity.

Within a requested model/provider tier, the score is (dispatch counts are normalized
against the least-used eligible account so distribution pressure does not vanish
in a long-lived process):

| Component | Weight | Interpretation |
| --- | ---: | --- |
| Capability suitability | 25 | Verified task fit beats unknown suitability |
| Quota headroom | 25 | Minimum known RPM/RPD remaining fraction; unknown = 0.5 |
| Health | 25 | Failure streak penalty, HALF_OPEN and stale discovery penalties |
| Latency | 10 | Inverse of successful-attempt latency EWMA |
| Account distribution | 10 | Inverse of relative dispatch count plus twice current in-flight load |
| Configured deployment weight | 5 | Bounded normalized nonnegative weight |

Deployment ID breaks ties. The same configuration and runtime state always yield
the same ordering. Failover prefers accounts not yet attempted within each tier.
Explicit model/provider choices and configured provider order remain stronger
preferences than score. Caller fallback edges are resolved only when reachable
from the requested source, expanded to compatible deployments, and deduplicated;
cycles cannot create unlimited attempts. An explicitly empty fallback list disables
automatic widening beyond the requested target.

## Attempts, recovery and persistence

Each compatible deployment is visited at most once for failover. After fresh
candidates are exhausted, transient server/transport failures may repeat only
within both the request retry budget and the appropriate RetryPolicy allowance.
The total bound is the planned deployment count plus `num_retries` (maximum 2
extra calls). Auth, quota and unavailable-model failures advance immediately;
permanent request errors and unknown/programming errors stop without damaging
circuit health. Model-specific capability/context mismatches fail over without
quarantine. Server failures update circuit failure streaks, never quota. Explicit
server `Retry-After` hints open the circuit until the delay elapses; ordinary
transient repeats use bounded exponential backoff outside the admission lock.

Auth quarantine is credential/account scoped; unavailable models are deployment
scoped. Quota boundaries are provider, account, model, account+model, or local when
unknown. A cooldown cannot override quarantine. A credential fingerprint and
dispatch time prevent older outcomes from erasing stronger, newer state.
Account auth records also protect newly discovered deployments. Key rotation
releases only the failed credential's quarantine.

RPM and RPD reset timestamps describe quota windows independently of an explicit
429 embargo. Positive headroom remains usable before the window reset. Known
counters are reserved at dispatch; reservations and outcomes are persisted in
atomic snapshots. Missing window deadlines use conservative 60-second/24-hour
expiry for observed counters. Successful responses can refresh authoritative
headroom and reset information.

Streams use the same attempt loop. Errors before the first chunk can fail over.
After output begins, errors propagate without replaying the request. Success is
recorded only on complete exhaustion; closing a stream releases the probe and
in-flight reservation without inventing a success. Consumers should exhaust or
close returned streams.

Version 2 snapshots remain compatible with the previous JSON layout. They add
quota window deadlines, failure streaks, latency and one-way credential
fingerprints. Writes flush and fsync a temporary file before replacement;
malformed numeric state is ignored safely. Request bodies and credential fields
remain excluded. Known credentials are redacted exactly from all persisted
strings, and credentials are excluded from Deployment repr and CLI errors.

This is a single-process router, not a distributed quota service. File snapshots
recover state after restart; independent processes sharing a state file do not
coordinate admission and should use distinct state files. Latency measures complete
successful attempts (including full stream consumption), not time to first token.

## Verification

Run `python -m unittest discover -s tests -v` for offline behavioral tests covering
ranking, quota/account reservations, concurrent dispatch, classification,
failover, probe ownership, caller fallback edges, streaming, stale outcomes,
credential rotation, state recovery and secret redaction. The integration test
uses the installed LiteLLM's official mock transport for completion and streaming.
No live credentials, provider requests or paid quota are needed.
