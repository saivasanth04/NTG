import asyncio
import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass

import jsonschema

from .failures import classify, reset_duration, normalized_headers
from .types import ChatRequest, RoutingError


@dataclass
class State:
    active: int = 0
    calls: int = 0
    latency: float = 1.0
    successes: int = 0
    failures: int = 0
    last_failure: str | None = None
    last_failure_at: float | None = None
    last_failure_scope: str | None = None
    last_status: int | None = None
    last_success_at: float | None = None


class RoutingService:
    """One event loop / one process. Admission and reservations are atomic between awaits.

    Unknown quota permits one conservative probe per ownership group, never invented RPM.
    Bounded retries are owned here; LiteLLM retries are explicitly disabled.
    """
    def __init__(self, deployments, transport=None, *, max_active=4, max_queue=16,
                 group_concurrency=1, max_attempts=3, attempt_timeout=20,
                 catalog_ttl=86400, clock=time.monotonic, rng=None, token_counter=None):
        if min(max_active, group_concurrency, max_attempts, attempt_timeout, catalog_ttl) <= 0 or max_queue < 0:
            raise ValueError("Invalid routing limits")
        self.deployments = deployments
        if transport is None:
            from .transport import LiteLLMTransport
            transport = LiteLLMTransport()
        self.transport = transport
        self.token_counter = token_counter
        self.max_active, self.max_queue = max_active, max_queue
        self.group_concurrency, self.max_attempts = group_concurrency, max_attempts
        self.attempt_timeout, self.catalog_ttl = attempt_timeout, catalog_ttl
        self.clock, self.rng = clock, rng or random.Random()
        self.states = defaultdict(State)
        self.group_active = Counter()
        self.account_calls = Counter()
        self.cooldowns = {}
        self.cooldown_reasons = {}
        self.provider_calls = Counter()
        self.model_successes = Counter()
        self.discovery_report = []
        for d in deployments:
            if d.available_after > time.time():
                self._cool(d, "deployment", d.available_after - time.time(), d.available_reason or "temporary_unavailable")
        self.admitted = 0
        self.active = 0
        self.waiting = 0
        self.condition = asyncio.Condition()
        self.metrics = Counter()

    @classmethod
    async def from_env(cls, env_path=".env", policy_path="policy.json", **kwargs):
        from .config import accounts_from_env, load_policy
        from .discovery import discover
        deployments, report = await discover(accounts_from_env(env_path), load_policy(policy_path))
        service = cls(deployments, **kwargs)
        service.discovery_report = report
        return service

    @staticmethod
    def _identity(d):
        return (d.account.provider, d.account.alias, d.model)

    @staticmethod
    def _group(d):
        return (d.account.provider, d.account.quota_group or "provider-shared")

    def _keys(self, d):
        return {"deployment": ("deployment", self._identity(d)),
                "account": ("account", (d.account.provider, d.account.alias)),
                "group": ("group", self._group(d)), "provider": ("provider", d.account.provider)}

    def _ready_at(self, d):
        return max((self.cooldowns.get(k, 0) for k in self._keys(d).values()), default=0)

    def _eligible(self, request, history=None):
        required = {"chat"} | request.required_capabilities
        for message in request.messages:
            content = message.get("content")
            if isinstance(content, list):
                for part in content:
                    kind = part.get("type") if isinstance(part, dict) else None
                    if kind == "image_url":
                        required.add("vision")
                    elif kind in ("input_audio", "audio"):
                        required.add("audio_input")
                    elif kind not in ("text",):
                        raise self._request_error("unsupported_content_type")
        if request.tools or request.tool_choice is not None or any(m.get("role") == "tool" or m.get("tool_calls") for m in request.messages):
            required.add("tools")
        if request.response_format:
            fmt = request.response_format.get("type")
            if fmt not in ("text", "json_object", "json_schema"):
                raise self._request_error("unsupported_response_format")
            if fmt != "text":
                required.add("json_schema" if fmt == "json_schema" else "json")
        candidates = []
        for d in self.deployments:
            reason = None
            budget = None
            if request.model != "auto" and request.model not in (d.model, d.id):
                reason = "explicit_model_selector"
            elif not (required - {"json", "json_schema"}) <= d.capabilities:
                reason = "missing_capabilities"
            else:
                from .budget import packet_budget
                budget = packet_budget(d, request, self.token_counter)
                reason = budget["reason"]
            if reason:
                if history is not None:
                    entry = self._event(d, reason, "INELIGIBLE", event="excluded")
                    if reason == "missing_capabilities":
                        entry["missing_capabilities"] = sorted(required - d.capabilities)
                    if budget:
                        entry["budget"] = budget
                    history.append(entry)
            else:
                candidates.append(d)
        if history is not None:
            self._compact_history(history)
        return candidates

    @staticmethod
    def _compact_history(history):
        """Aggregate catalog/cooldown diagnostics; retain every actual API attempt."""
        grouped, compact = {}, []
        for entry in history:
            if entry["event"] not in ("excluded", "skipped"):
                compact.append(entry)
                continue
            key = (entry["event"], entry["provider"], entry["account"], entry["outcome"], entry.get("scope"))
            if key in grouped:
                first = grouped[key]
                first["deployment_count"] += entry.get("deployment_count", 1)
                if "retry_after_seconds" in entry:
                    first["retry_after_seconds"] = max(first.get("retry_after_seconds", 0), entry["retry_after_seconds"])
            else:
                first = dict(entry, deployment_count=entry.get("deployment_count", 1))
                first["model_is_example"] = True
                grouped[key] = first
                compact.append(first)
        history[:] = compact

    @staticmethod
    def _event(d, outcome, category, *, event="skipped", scope=None, status=None, latency_ms=0):
        return dict(provider=d.account.provider if d else None,
                    account=d.account.alias if d else None, model=d.model if d else None,
                    event=event, attempted=event == "attempt", outcome=outcome,
                    result=str(status) if status is not None else outcome,
                    category=category, scope=scope, latency_ms=latency_ms)

    def _request_error(self, code):
        return RoutingError(code, 400, [self._event(None, code, "INVALID_REQUEST", event="request_rejected")])

    def _blocking(self, d):
        active = [(self.cooldowns.get(key, 0), self.cooldown_reasons.get(key, "temporary_unavailable"), scope)
                  for scope, key in self._keys(d).items() if self.cooldowns.get(key, 0) > self.clock()]
        return max(active, default=(0, None, None), key=lambda item: item[0])

    def _pool_failure(self, candidates, history):
        reasons = [self._blocking(d)[1] or self.states[self._identity(d)].last_failure for d in candidates]
        quota_only = candidates and all(r == "quota_exhausted" for r in reasons)
        size_only = candidates and all(r in ("context_exceeded", "output_truncated") for r in reasons)
        code = "context_capacity_exceeded" if size_only else "quota_exhausted" if quota_only else "global_pool_exhausted"
        history.append(self._event(None, code, "GLOBAL_POOL_EXHAUSTED", event="pool_exhausted"))
        return RoutingError(code, 429 if code == "quota_exhausted" else 503, history)

    async def _reserve(self, candidates, tried, deadline, request, history, skipped, exhausted_providers):
        async with self.condition:
            while True:
                now = self.clock()
                if now >= deadline:
                    raise RoutingError("request_deadline", 504, history)
                ready = []
                for d in candidates:
                    if self._identity(d) in tried:
                        continue
                    until, reason, scope = self._blocking(d)
                    if until > now:
                        marker = (self._identity(d), reason, until)
                        if marker not in skipped:
                            skipped.add(marker)
                            category = "ACCOUNT_EXHAUSTED" if reason in ("quota_exhausted", "invalid_credentials") else (
                                "MODEL_UNAVAILABLE" if reason in ("model_inaccessible", "unsupported_capability") else "TEMPORARY_FAILURE")
                            entry = self._event(d, reason, category, scope=scope)
                            entry["retry_after_seconds"] = round(until - now, 2)
                            entry["evidence"] = ("provider_metadata" if d.available_reason and d.available_after > time.time()
                                                 else "runtime_cooldown")
                            history.append(entry)
                    else:
                        ready.append(d)
                self._compact_history(history)
                providers = {d.account.provider for d in candidates}
                for provider in sorted(providers):
                    if provider not in {d.account.provider for d in ready} and provider not in exhausted_providers:
                        exhausted_providers.add(provider)
                        representative = next(d for d in candidates if d.account.provider == provider)
                        history.append(self._event(representative, "provider_exhausted", "PROVIDER_EXHAUSTED",
                                                   event="provider_exhausted", scope="provider"))
                if not ready:
                    raise self._pool_failure(candidates, history)
                available = [d for d in ready if self.group_active[self._group(d)] < self.group_concurrency]
                if available and self.active < self.max_active:
                    tried_providers = {d.account.provider for d in candidates if self._identity(d) in tried}
                    self.rng.shuffle(available)
                    def score(d):
                        state = self.states[self._identity(d)]
                        recent_failure = (state.last_failure_scope != "request" and state.last_failure_at is not None and now - state.last_failure_at < 60
                                          and (state.last_success_at is None or state.last_failure_at > state.last_success_at))
                        return (d.account.provider in tried_providers, recent_failure,
                                bool(request.preferred_provider and request.preferred_provider != d.account.provider),
                                bool(request.preferred_model and request.preferred_model != d.model),
                                request.model_class not in d.model_classes,
                                not self.model_successes[(d.account.provider, d.model)],
                                bool((request.response_format or {}).get("type") == "json_schema" and "json_schema" not in d.capabilities),
                                d.priority,
                                bool(d.account.models and d.model not in d.account.models),
                                self.provider_calls[d.account.provider], self.account_calls[(d.account.provider, d.account.alias)],
                                state.calls, state.latency)
                    d = min(available, key=score)
                    self.active += 1
                    self.group_active[self._group(d)] += 1
                    self.states[self._identity(d)].active += 1
                    self.states[self._identity(d)].calls += 1
                    self.account_calls[(d.account.provider, d.account.alias)] += 1
                    self.provider_calls[d.account.provider] += 1
                    return d
                if self.waiting >= self.max_queue:
                    self.metrics["queue_rejected"] += 1
                    raise RoutingError("queue_full", 429, history)
                self.waiting += 1
                try:
                    next_recovery = [self._ready_at(d) - now for d in candidates
                                     if self._identity(d) not in tried and self._ready_at(d) > now]
                    await asyncio.wait_for(self.condition.wait(), min([deadline - now] + next_recovery))
                except TimeoutError:
                    pass
                finally:
                    self.waiting -= 1

    async def _release(self, d):
        async with self.condition:
            self.active -= 1
            self.group_active[self._group(d)] -= 1
            self.states[self._identity(d)].active -= 1
            self.condition.notify_all()

    def _cool(self, d, scope, seconds, reason="temporary_unavailable"):
        key = self._keys(d)[scope]
        until = self.clock() + seconds
        if until >= self.cooldowns.get(key, 0):
            self.cooldowns[key] = until
            self.cooldown_reasons[key] = reason

    def _headers(self, d, headers):
        h = normalized_headers(headers)
        for kind in ("requests", "tokens"):
            remaining = h.get(f"x-ratelimit-remaining-{kind}")
            try:
                exhausted = remaining is not None and float(remaining) <= 0
            except ValueError:
                exhausted = False
            if exhausted:
                reset = reset_duration(h.get(f"x-ratelimit-reset-{kind}"))
                self._cool(d, "group", reset if reset is not None else 60, "rate_limited")

    def _validate_output(self, request, response):
        choices = response.get("choices") or []
        message = choices[0].get("message", {}) if choices else {}
        content = message.get("content")
        if not (isinstance(content, str) and content.strip()) and not message.get("tool_calls"):
            raise RoutingError("empty_completion", 502)
        choice = request.tool_choice
        if choice == "required" or isinstance(choice, dict):
            calls = (response.get("choices") or [{}])[0].get("message", {}).get("tool_calls") or []
            name = choice.get("function", {}).get("name") if isinstance(choice, dict) else None
            if not calls or (name and not any(c.get("function", {}).get("name") == name for c in calls)):
                raise RoutingError("invalid_structured_output", 502)
        fmt = request.response_format or {}
        if fmt.get("type") in ("json_object", "json_schema"):
            try:
                content = response["choices"][0]["message"].get("content")
                data = json.loads(content)
                if fmt["type"] == "json_schema":
                    jsonschema.validate(data, fmt["json_schema"]["schema"])
            except (ValueError, TypeError, KeyError, IndexError, jsonschema.ValidationError):
                raise RoutingError("invalid_structured_output", 502) from None

    async def chat(self, request: ChatRequest | dict):
        request = ChatRequest.model_validate(request) if isinstance(request, dict) else request
        if request.stream:
            raise self._request_error("streaming_not_supported")
        if request.tool_choice is not None and not request.tools:
            raise self._request_error("tool_choice_requires_tools")
        if (request.response_format or {}).get("type") == "json_schema":
            try:
                jsonschema.Draft202012Validator.check_schema(request.response_format["json_schema"]["schema"])
            except (KeyError, TypeError, jsonschema.SchemaError):
                raise self._request_error("invalid_json_schema") from None
        history = []
        candidates = self._eligible(request, history)
        if not candidates:
            size_limited = any(h["outcome"] in ("context_limit", "output_limit") for h in history)
            code = "context_capacity_exceeded" if size_limited else "no_compatible_deployment"
            history.append(self._event(None, code, "GLOBAL_POOL_EXHAUSTED", event="pool_exhausted"))
            raise RoutingError(code, 503, history)
        if self.admitted >= self.max_active + self.max_queue:
            self.metrics["queue_rejected"] += 1
            history.append(self._event(None, "queue_full", "TEMPORARY_FAILURE", event="request_stopped"))
            raise RoutingError("queue_full", 429, history)
        self.admitted += 1
        start = self.clock()
        deadline = start + request.deadline_seconds
        tried, skipped, exhausted_providers = set(), set(), set()
        try:
            # Cover the whole compatible pool once, rather than spending a small
            # retry budget on one provider. The caller deadline remains the hard bound.
            for attempt in range(max(self.max_attempts, len(candidates))):
                try:
                    d = await self._reserve(candidates, tried, deadline, request, history, skipped, exhausted_providers)
                except RoutingError as exc:
                    if exc.code in ("request_deadline", "queue_full"):
                        history.append(self._event(None, exc.code, "TEMPORARY_FAILURE", event="request_stopped"))
                    exc.history = history
                    raise
                tried.add(self._identity(d))
                t = self.clock()
                try:
                    timeout = min(self.attempt_timeout, deadline - t)
                    if timeout <= 0:
                        raise RoutingError("request_deadline", 504, history)
                    response, headers = await asyncio.wait_for(self.transport.complete(d, request, timeout), timeout)
                    if any(c.get("finish_reason") in ("length", "max_tokens") for c in response.get("choices", [])):
                        raise RoutingError("output_truncated", 502)
                    self._validate_output(request, response)
                    elapsed = self.clock() - t
                    state = self.states[self._identity(d)]
                    state.latency = .7 * state.latency + .3 * elapsed
                    state.successes += 1
                    self.model_successes[(d.account.provider, d.model)] += 1
                    state.last_success_at = self.clock()
                    d.access = "observed_success"
                    self._headers(d, headers)
                    self.metrics["success"] += 1
                    usage = response.get("usage", {})
                    self.metrics["total_tokens"] += usage.get("total_tokens", 0) or 0
                    history.append(self._event(d, "success", "SUCCESS", event="attempt",
                                               latency_ms=round(elapsed * 1000, 2)))
                    reported_cost = usage.get("cost")
                    if not isinstance(reported_cost, (int, float)) or not math.isfinite(reported_cost) or reported_cost < 0:
                        reported_cost = None
                    response["routing"] = dict(provider=d.account.provider, model=d.model,
                        account=d.account.alias, latency_ms=round((self.clock() - start) * 1000, 2),
                        usage=usage, history=history, cost_usd=reported_cost,
                        cost_basis="provider_reported" if reported_cost is not None else "unknown")
                    return response
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    if isinstance(exc, RoutingError):
                        failure_code = exc.code
                        if exc.code not in ("invalid_structured_output", "empty_completion", "output_truncated"):
                            raise
                        from .failures import Failure
                        failure = Failure(failure_code, "request" if failure_code == "output_truncated" else "deployment", 30)
                    else:
                        failure = classify(exc, attempt)
                    state = self.states[self._identity(d)]
                    state.failures += 1
                    state.last_failure, state.last_failure_at = failure.code, self.clock()
                    state.last_failure_scope = failure.scope
                    state.last_status = (getattr(exc, "status_code", None)
                                         or getattr(getattr(exc, "response", None), "status_code", None))
                    self.metrics[failure.code] += 1
                    if failure.code == "model_inaccessible":
                        d.access = "temporarily_denied"
                    history.append(self._event(d, failure.code, failure.category, event="attempt",
                        scope=failure.scope, status=state.last_status,
                        latency_ms=round((self.clock() - t) * 1000, 2)))
                    from .failures import upstream_code
                    safe_code = upstream_code(exc)
                    if safe_code:
                        history[-1]["upstream_code"] = safe_code
                    if not failure.retry:
                        raise RoutingError(failure.code, 400, history) from None
                    # Positive jitter never shortens a provider Retry-After instruction.
                    if failure.scope != "request":
                        self._cool(d, failure.scope, failure.delay + self.rng.uniform(.05, .5), failure.code)
                finally:
                    await self._release(d)
            # Record the final failed deployment's provider exhaustion as well.
            await self._reserve(candidates, tried, deadline, request, history, skipped, exhausted_providers)
            raise self._pool_failure(candidates, history)
        finally:
            self.admitted -= 1

    def status(self):
        return dict(active=self.active, waiting=self.waiting, admitted=self.admitted, metrics=dict(self.metrics),
                    discovery=self.discovery_report,
                    deployments=[{**d.public(), "runtime": {
                        "cooldown_seconds": round(max(0, self._ready_at(d) - self.clock()), 2),
                        "active": self.states[self._identity(d)].active, "calls": self.states[self._identity(d)].calls,
                        "successes": self.states[self._identity(d)].successes, "failures": self.states[self._identity(d)].failures,
                        "latency_ewma_ms": round(self.states[self._identity(d)].latency * 1000, 2),
                        "availability": "cooldown" if self._ready_at(d) > self.clock() else "ready",
                        "cooldown_reason": self._blocking(d)[1],
                        "last_failure": self.states[self._identity(d)].last_failure,
                        "last_status": self.states[self._identity(d)].last_status}} for d in self.deployments])
