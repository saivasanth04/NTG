import asyncio
import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass

import jsonschema

from .failures import classify, duration, normalized_headers
from .types import ChatRequest, RoutingError


@dataclass
class State:
    active: int = 0
    calls: int = 0
    latency: float = 1.0
    successes: int = 0
    failures: int = 0


class RoutingService:
    """One event loop / one process. Admission and reservations are atomic between awaits.

    Unknown quota permits one conservative probe per ownership group, never invented RPM.
    Bounded retries are owned here; LiteLLM retries are explicitly disabled.
    """
    def __init__(self, deployments, transport=None, *, max_active=4, max_queue=16,
                 group_concurrency=1, max_attempts=3, attempt_timeout=20,
                 catalog_ttl=86400, clock=time.monotonic, rng=None):
        if min(max_active, group_concurrency, max_attempts, attempt_timeout, catalog_ttl) <= 0 or max_queue < 0:
            raise ValueError("Invalid routing limits")
        self.deployments = deployments
        if transport is None:
            from .transport import LiteLLMTransport
            transport = LiteLLMTransport()
        self.transport = transport
        self.max_active, self.max_queue = max_active, max_queue
        self.group_concurrency, self.max_attempts = group_concurrency, max_attempts
        self.attempt_timeout, self.catalog_ttl = attempt_timeout, catalog_ttl
        self.clock, self.rng = clock, rng or random.Random()
        self.states = defaultdict(State)
        self.group_active = Counter()
        self.account_calls = Counter()
        self.cooldowns = {}
        for d in deployments:
            if d.available_after > time.time():
                self._cool(d, "group", d.available_after - time.time())
        self.admitted = 0
        self.active = 0
        self.waiting = 0
        self.condition = asyncio.Condition()
        self.metrics = Counter()

    @classmethod
    async def from_env(cls, env_path=".env", policy_path="policy.json", **kwargs):
        from .config import accounts_from_env, load_policy
        from .discovery import discover
        deployments, _ = await discover(accounts_from_env(env_path), load_policy(policy_path))
        return cls(deployments, **kwargs)

    def _keys(self, d):
        return {"deployment": ("deployment", d.id), "account": ("account", d.account.alias),
                "group": ("group", d.account.quota_group), "provider": ("provider", d.account.provider)}

    def _ready_at(self, d):
        return max((self.cooldowns.get(k, 0) for k in self._keys(d).values()), default=0)

    def _eligible(self, request):
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
                        raise RoutingError("unsupported_content_type", 400)
        if request.tools or request.tool_choice is not None or any(m.get("role") == "tool" or m.get("tool_calls") for m in request.messages):
            required.add("tools")
        if request.response_format:
            fmt = request.response_format.get("type")
            if fmt not in ("text", "json_object", "json_schema"):
                raise RoutingError("unsupported_response_format", 400)
            if fmt != "text":
                required.add("json_schema" if fmt == "json_schema" else "json")
        # UTF-8 byte count is deliberately conservative for mixed-language code, tools and JSON.
        estimate = len(json.dumps({"messages": request.messages, "tools": request.tools,
                                   "response_format": request.response_format}, ensure_ascii=False).encode()) + 64 * len(request.messages)
        needed = max(estimate, request.input_tokens or 0) + request.max_tokens
        return [d for d in self.deployments if d.free == "verified"
                and time.time() - d.checked_at < self.catalog_ttl
                and required <= d.capabilities and d.context >= needed
                and (not d.max_output or d.max_output >= request.max_tokens)
                and (request.model == "auto" or request.model in (d.model, d.id))]

    async def _reserve(self, candidates, tried, deadline):
        async with self.condition:
            while True:
                now = self.clock()
                if now >= deadline:
                    raise RoutingError("capacity_wait_deadline", 504)
                ready = [d for d in candidates if d.id not in tried
                         and self._ready_at(d) <= now
                         and self.group_active[d.account.quota_group] < self.group_concurrency]
                if ready and self.active < self.max_active:
                    tried_models = {(d.account.provider, d.model) for d in candidates if d.id in tried}
                    # Try a different model after failure, then balance account usage,
                    # deployment usage, measured latency and jitter.
                    # Accounts with many models do not receive proportionally more traffic.
                    self.rng.shuffle(ready)
                    d = min(ready, key=lambda x: ((x.account.provider, x.model) in tried_models,
                            self.account_calls[x.account.alias],
                            self.states[x.id].calls, self.states[x.id].latency))
                    self.active += 1
                    self.group_active[d.account.quota_group] += 1
                    self.states[d.id].active += 1
                    self.states[d.id].calls += 1
                    self.account_calls[d.account.alias] += 1
                    return d
                remaining = [d for d in candidates if d.id not in tried]
                if not remaining:
                    raise RoutingError("suitable_deployments_exhausted")
                # Event-driven wakeup on release or nearest cooldown/deadline; no health polling.
                cooldowns = [self._ready_at(d) - now for d in remaining if self._ready_at(d) > now]
                wait = min([deadline - now] + cooldowns)
                if self.waiting >= self.max_queue:
                    self.metrics["queue_rejected"] += 1
                    raise RoutingError("queue_full", 429)
                self.waiting += 1
                try:
                    await asyncio.wait_for(self.condition.wait(), wait)
                except TimeoutError:
                    pass
                finally:
                    self.waiting -= 1

    async def _release(self, d):
        async with self.condition:
            self.active -= 1
            self.group_active[d.account.quota_group] -= 1
            self.states[d.id].active -= 1
            self.condition.notify_all()

    def _cool(self, d, scope, seconds):
        key = self._keys(d)[scope]
        self.cooldowns[key] = max(self.cooldowns.get(key, 0), self.clock() + seconds)

    def _headers(self, d, headers):
        h = normalized_headers(headers)
        for kind in ("requests", "tokens"):
            remaining = h.get(f"x-ratelimit-remaining-{kind}")
            try:
                exhausted = remaining is not None and float(remaining) <= 0
            except ValueError:
                exhausted = False
            if exhausted:
                reset = duration(h.get(f"x-ratelimit-reset-{kind}"))
                self._cool(d, "group", reset if reset is not None else 60)

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
            raise RoutingError("streaming_not_supported", 400)
        if request.tool_choice is not None and not request.tools:
            raise RoutingError("tool_choice_requires_tools", 400)
        if (request.response_format or {}).get("type") == "json_schema":
            try:
                jsonschema.Draft202012Validator.check_schema(request.response_format["json_schema"]["schema"])
            except (KeyError, TypeError, jsonschema.SchemaError):
                raise RoutingError("invalid_json_schema", 400) from None
        candidates = self._eligible(request)
        if not candidates:
            raise RoutingError("no_eligible_free_deployment")
        if self.admitted >= self.max_active + self.max_queue:
            self.metrics["queue_rejected"] += 1
            raise RoutingError("queue_full", 429)
        self.admitted += 1
        start = self.clock()
        deadline = start + request.deadline_seconds
        history, tried = [], set()
        try:
            for attempt in range(self.max_attempts):
                if len(tried) == len(candidates):
                    tried.clear()  # bounded retry only after every alternative was considered
                try:
                    d = await self._reserve(candidates, tried, deadline)
                except RoutingError as exc:
                    exc.history = history
                    raise
                tried.add(d.id)
                t = self.clock()
                try:
                    timeout = min(self.attempt_timeout, deadline - t)
                    if timeout <= 0:
                        raise RoutingError("request_deadline", 504, history)
                    response, headers = await asyncio.wait_for(self.transport.complete(d, request, timeout), timeout)
                    self._validate_output(request, response)
                    elapsed = self.clock() - t
                    state = self.states[d.id]
                    state.latency = .7 * state.latency + .3 * elapsed
                    state.successes += 1
                    d.access = "observed_success"
                    self._headers(d, headers)
                    self.metrics["success"] += 1
                    usage = response.get("usage", {})
                    self.metrics["total_tokens"] += usage.get("total_tokens", 0) or 0
                    history.append(dict(provider=d.account.provider, model=d.model, account=d.account.alias,
                                        outcome="success", latency_ms=round(elapsed * 1000, 2)))
                    reported_cost = usage.get("cost")
                    if not isinstance(reported_cost, (int, float)) or not math.isfinite(reported_cost) or reported_cost < 0:
                        reported_cost = None
                    response["routing"] = dict(provider=d.account.provider, model=d.model,
                        account=d.account.alias, latency_ms=round((self.clock() - start) * 1000, 2),
                        usage=usage, history=history, cost_usd=reported_cost,
                        cost_basis="provider_reported" if reported_cost is not None else "unknown; free-only policy")
                    return response
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    if isinstance(exc, RoutingError):
                        failure_code = exc.code
                        if exc.code not in ("invalid_structured_output", "empty_completion"):
                            raise
                        from .failures import Failure
                        failure = Failure(failure_code, "deployment", 30)
                    else:
                        failure = classify(exc, attempt)
                    self.states[d.id].failures += 1
                    self.metrics[failure.code] += 1
                    if failure.code == "model_inaccessible":
                        d.access = "denied"
                    history.append(dict(provider=d.account.provider, model=d.model, account=d.account.alias,
                        outcome=failure.code, scope=failure.scope,
                        latency_ms=round((self.clock() - t) * 1000, 2)))
                    if not failure.retry:
                        raise RoutingError(failure.code, 400, history) from None
                    # Positive jitter never shortens a provider Retry-After instruction.
                    self._cool(d, failure.scope, failure.delay + self.rng.uniform(.05, .5))
                finally:
                    await self._release(d)
            raise RoutingError("attempts_exhausted", 503, history)
        finally:
            self.admitted -= 1

    def status(self):
        return dict(active=self.active, waiting=self.waiting, admitted=self.admitted, metrics=dict(self.metrics),
                    deployments=[{**d.public(), "runtime": {
                        "cooldown_seconds": round(max(0, self._ready_at(d) - self.clock()), 2),
                        "active": self.states[d.id].active, "calls": self.states[d.id].calls,
                        "successes": self.states[d.id].successes, "failures": self.states[d.id].failures,
                        "latency_ewma_ms": round(self.states[d.id].latency * 1000, 2)}} for d in self.deployments])
