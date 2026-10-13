"""NTG owns model/account/provider fallback; LiteLLM converts protocols."""
import asyncio
from dataclasses import dataclass
import time
from .config import load_accounts, eligibility
from .state import Block, classify, backoff
from .transport import send, compatible, ManagedStream
from .scheduler import RoundRobin

class RoutingError(Exception):
    def __init__(self, failures, availability, reason="No eligible route succeeded"):
        retries = [r["retry_at"] for r in availability if r.get("retry_at")]
        self.details = {"error": "routing_failed", "reason": reason, "failures": failures,
                        "availability": availability, "earliest_retry_at": min(retries) if retries else None}
        super().__init__(reason)

@dataclass
class CompletionResult:
    response: object
    provider: str
    account_id: str
    model: str
    attempts: int
    latency_ms: float
    routing_reason: str

class UnifiedNTGRouter:
    def __init__(self, env_path=".env", *, accounts=None, transport=send, capability_check=compatible,
                 provider_order=("gemini", "groq", "openrouter", "cloudflare", "cohere"), routing_state_path=None):
        self.env_path = env_path
        self.accounts = load_accounts(env_path) if accounts is None else accounts
        self.transport = transport
        self.capability_check = capability_check
        self.provider_order = tuple(provider_order)
        self.scheduler = RoundRobin(routing_state_path)
        self.blocks = {}
        self.inflight = {}
        self.seen = set()
        self.lock = asyncio.Lock()

    def _keys(self, account, model):
        return [("account", account.id), ("model", account.id, model),
                ("group", account.provider, account.group),
                ("group_model", account.provider, account.group, model),
                ("provider_model", account.provider, model)]

    def _block(self, account, model):
        now = time.time()
        return next((self.blocks[k] for k in self._keys(account, model)
                     if k in self.blocks and (self.blocks[k].until is None or self.blocks[k].until > now)), None)

    def availability(self):
        rows = []
        for account in self.accounts:
            for model in account.models or ("unconfigured",):
                reason = eligibility(account, model)
                block = self._block(account, model)
                status = ("disabled" if reason else "disabled" if block and block.until is None else
                          "temporarily_blocked" if block else "eligible" if (account.id, model) in self.seen else "unknown")
                rows.append({"provider": account.provider, "account_id": account.masked, "model": model,
                             "status": status, "reason": reason or (block.category if block else "untried" if status == "unknown" else "last request succeeded"),
                             "retry_at": block.until if block else None})
        return rows

    async def reload(self):
        async with self.lock:
            if any(self.inflight.values()):
                raise RuntimeError("Reload requires no active completions/streams")
            self.accounts = load_accounts(self.env_path)
            self.blocks.clear()
            self.seen.clear()

    async def reenable(self, account_id, model=None):
        """Session action; credentials require reload, not manual override."""
        async with self.lock:
            account = next(a for a in self.accounts if a.id == account_id)
            keys = set(k for m in ((model,) if model else account.models) for k in self._keys(account, m))
            for key in keys:
                if key in self.blocks and self.blocks[key].category != "invalid_credentials":
                    del self.blocks[key]

    async def _failure(self, account, model, error):
        category, until, scope = classify(error, account.provider)
        keys = dict(zip(("account", "model", "group", "group_model", "provider_model"), self._keys(account, model)))
        async with self.lock:
            key = keys[scope]
            old = self.blocks.get(key)
            failures = (old.failures if old else 0) + 1
            if category == "malformed_request":
                return category
            if category in ("invalid_credentials", "inaccessible_model", "depleted_credits"):
                until = None
            elif category != "long_term_exhaustion" and until is None:
                until = time.time() + backoff(failures)
            self.blocks[key] = Block(category, until, failures)
        return category

    async def acompletion(self, messages, *, model=None, providers=None, allow_model_fallback=True,
                          fallback_models=(), max_attempts=None, deadline_seconds=60, input_tokens=None,
                          **params):
        if not messages or (max_attempts is not None and max_attempts < 1) or not 0 < deadline_seconds < float("inf"):
            raise ValueError("Messages, positive attempts and a finite positive deadline are required")
        allowed = {"temperature", "top_p", "max_tokens", "max_completion_tokens", "tools", "tool_choice",
                   "response_format", "stream", "stop", "seed", "presence_penalty", "frequency_penalty", "n"}
        if set(params) - allowed:
            raise ValueError("Unsupported parameters: " + ", ".join(sorted(set(params) - allowed)))
        if any(tool.get("type") != "function" for tool in params.get("tools", [])):
            raise ValueError("Only caller-provided function tools are allowed")
        if any(not isinstance(params[k], int) or params[k] < 1 for k in ("max_tokens", "max_completion_tokens") if k in params):
            raise ValueError("Output limits must be positive integers")
        start = time.monotonic()
        end = start + deadline_seconds
        tried, failures = set(), []
        model_pools = {}
        provider_names = {a.provider for a in self.accounts}
        order = list(dict.fromkeys(providers or self.provider_order))
        order.extend(p for p in sorted(provider_names) if p not in order)
        # A known requested model starts with its provider, while unrestricted
        # fallback can still reach other providers' configured free pools.
        preferred = next((a.provider for a in self.accounts if model in a.models or
                          any(model == f"{a.provider}/{m}" for m in a.models)), None)
        if preferred and not providers:
            order.remove(preferred)
            order.insert(0, preferred)
        for a in self.accounts:
            pool = list(a.models)
            exact = [m for m in pool if model in (m, f"{a.provider}/{m}")]
            pool = exact + [m for m in pool if m not in exact]
            if fallback_models:
                pool = [m for m in pool if m in exact or m in fallback_models or f"{a.provider}/{m}" in fallback_models]
            if model is not None and not allow_model_fallback:
                pool = exact
            model_pools[a.id] = tuple(pool)
        ordered_accounts = sorted(self.accounts, key=lambda a: (order.index(a.provider), int(a.id.rsplit('-', 1)[-1])))
        if max_attempts is None:
            max_attempts = sum(len(model_pools[a.id]) for a in ordered_accounts)
        plan = None
        for attempt in range(1, max_attempts + 1):
            async with self.lock:
                routes = []
                for a in ordered_accounts:
                    if providers and a.provider not in providers:
                        continue
                    for m in model_pools[a.id]:
                        if (a.id, m) in tried or eligibility(a, m) or self._block(a, m):
                            continue
                        try:
                            supported = self.capability_check(a.provider, m, messages, params, input_tokens)
                        except Exception:
                            supported = False
                        if supported:
                            routes.append((a, m))
                if not routes or time.monotonic() >= end:
                    break
                if plan is None:
                    plan = await asyncio.to_thread(self.scheduler.plan, routes, order, model)
                available = {(a.id, m) for a, m in routes}
                routes = [(a, m) for a, m in plan if (a.id, m) in available]
                if not routes or time.monotonic() >= end:
                    break
                # Exhaust this account's models, then its provider's accounts,
                # before moving to the next provider. Cooldowns skip only the
                # quota scope actually reported by the service.
                a, m = routes[0]
                tried.add((a.id, m))
                self.inflight[a.id] = self.inflight.get(a.id, 0) + 1
                self.inflight[(a.id, m)] = self.inflight.get((a.id, m), 0) + 1
            handed_off = False
            response = None
            try:
                async with asyncio.timeout_at(end):
                    response = await self.transport(a, m, messages, max(.001, min(15, end - time.monotonic())), **params)
                    first = await anext(response) if params.get("stream") else None
                actual = getattr(first if first is not None else response, "model", None) or m
                self.seen.add((a.id, m))
                reason = "round-robin provider, account and model; model then account then provider fallback" + ("; fallback after provider failure" if attempt > 1 else "")
                if params.get("stream"):
                    upstream = response
                    async def close_stream(account=a, selected_model=m, source=upstream):
                        try:
                            if hasattr(source, "aclose"):
                                await source.aclose()
                        finally:
                            await self._release(account, selected_model)
                    response = ManagedStream(self._stream(upstream, None, a, m, end),
                                             close_stream, first)
                    handed_off = True
                return CompletionResult(response, a.provider, a.masked, actual, attempt,
                                        (time.monotonic() - start) * 1000, reason)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                category = await self._failure(a, m, error)
                failures.append({"provider": a.provider, "account_id": a.masked, "model": m,
                                 "category": category, "status_code": getattr(error, "status_code", None)})
            finally:
                if not handed_off:
                    try:
                        if params.get("stream") and response is not None and hasattr(response, "aclose"):
                            await response.aclose()
                    finally:
                        await self._release(a, m)
        raise RoutingError(failures, self.availability())

    async def _release(self, a, m):
        async with self.lock:
            self.inflight[a.id] -= 1
            self.inflight[(a.id, m)] -= 1

    async def _stream(self, upstream, first, a, m, end):
        try:
            while True:
                async with asyncio.timeout_at(end):
                    try:
                        chunk = await anext(upstream)
                    except StopAsyncIteration:
                        break
                yield chunk
        except Exception as error:
            category = await self._failure(a, m, error)
            raise RoutingError([{"provider": a.provider, "account_id": a.masked, "model": m,
                                 "category": category}], self.availability(), "Stream failed; request was not replayed") from None
        finally:
            try:
                if hasattr(upstream, "aclose"):
                    await upstream.aclose()
            finally:
                pass  # ManagedStream releases the reservation exactly once.
