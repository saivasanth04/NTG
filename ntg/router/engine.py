"""NTG owns fair admission and fallback; LiteLLM owns protocol conversion."""
import asyncio
from dataclasses import dataclass
import time
from .config import load_accounts, eligibility
from .state import Block, classify, backoff
from .transport import send, compatible, ManagedStream, openrouter_metadata

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
    def __init__(self, env_path=".env", *, accounts=None, transport=send, capability_check=compatible):
        self.env_path = env_path
        self.accounts = load_accounts(env_path) if accounts is None else accounts
        self.transport = transport
        self.capability_check = capability_check
        self.blocks = {}
        self.inflight = {}
        self.seen = set()
        self.cursor = 0
        self.model_cursors = {}
        self.lock = asyncio.Lock()

    def _keys(self, account, model):
        return [("account", account.id), ("model", account.id, model),
                ("group", account.provider, account.group),
                ("group_model", account.provider, account.group, model)]

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
        category, until, scope = classify(error)
        keys = dict(zip(("account", "model", "group", "group_model"), self._keys(account, model)))
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

    async def acompletion(self, messages, *, model=None, providers=None, allow_model_fallback=False,
                          fallback_models=(), max_attempts=6, deadline_seconds=60, input_tokens=None,
                          **params):
        if not messages or max_attempts < 1 or not 0 < deadline_seconds < float("inf"):
            raise ValueError("Messages, positive attempts and a finite positive deadline are required")
        allowed = {"temperature", "top_p", "max_tokens", "max_completion_tokens", "tools", "tool_choice",
                   "response_format", "stream", "stop", "seed", "presence_penalty", "frequency_penalty", "n"}
        if set(params) - allowed:
            raise ValueError("Unsupported parameters: " + ", ".join(sorted(set(params) - allowed)))
        if any(tool.get("type") != "function" for tool in params.get("tools", [])):
            raise ValueError("Only caller-provided function tools are allowed")
        if "max_tokens" not in params and "max_completion_tokens" not in params:
            params["max_tokens"] = 256
        if any(not isinstance(params[k], int) or params[k] < 1 for k in ("max_tokens", "max_completion_tokens") if k in params):
            raise ValueError("Output limits must be positive integers")
        start = time.monotonic()
        end = start + deadline_seconds
        tried, failures = set(), []
        metadata = {}
        if self.transport is send and any(a.provider == "openrouter" and
                (not providers or "openrouter" in providers) and any(not eligibility(a, m) for m in a.models)
                for a in self.accounts):
            try:
                async with asyncio.timeout_at(end):
                    metadata = await openrouter_metadata(min(10, deadline_seconds))
            except Exception:
                failures.append({"provider": "openrouter", "category": "free_metadata_unavailable"})
        for attempt in range(1, max_attempts + 1):
            async with self.lock:
                routes = []
                for a in self.accounts:
                    if providers and a.provider not in providers:
                        continue
                    for m in a.models:
                        qualified = f"{a.provider}/{m}"
                        exact = model in (m, qualified)
                        fallback = allow_model_fallback and (m in fallback_models or qualified in fallback_models)
                        if model is not None and not exact and not fallback:
                            continue
                        if model is None and fallback_models and m not in fallback_models and qualified not in fallback_models:
                            continue
                        if (a.id, m) in tried or eligibility(a, m) or self._block(a, m):
                            continue
                        try:
                            if self.capability_check is compatible and a.provider == "openrouter":
                                info = metadata.get(m)
                                supported = bool(info) and all(k == "stream" or k in info["parameters"] for k in params) and compatible(
                                    a.provider, m, messages, params, input_tokens, metadata=info)
                            else:
                                supported = self.capability_check(a.provider, m, messages, params, input_tokens)
                        except Exception:
                            supported = False
                        if supported:
                            routes.append((a, m))
                if not routes or time.monotonic() >= end:
                    break
                # Account load first, then route load; cyclic tie-breaking across the entire compatible pool.
                minimum = min(self.inflight.get(a.id, 0) for a, m in routes)
                ties = [(a, m) for a, m in routes if self.inflight.get(a.id, 0) == minimum]
                universe = [a.id for a in self.accounts]
                a = min((a for a, m in ties), key=lambda a: (universe.index(a.id) - self.cursor) % len(universe))
                self.cursor = (universe.index(a.id) + 1) % len(universe)
                models = [m for candidate, m in ties if candidate.id == a.id]
                model_load = min(self.inflight.get((a.id, m), 0) for m in models)
                m = min((m for m in models if self.inflight.get((a.id, m), 0) == model_load),
                        key=lambda m: (a.models.index(m) - self.model_cursors.get(a.id, 0)) % len(a.models))
                self.model_cursors[a.id] = (a.models.index(m) + 1) % len(a.models)
                tried.add((a.id, m))
                self.inflight[a.id] = self.inflight.get(a.id, 0) + 1
                self.inflight[(a.id, m)] = self.inflight.get((a.id, m), 0) + 1
            handed_off = False
            response = None
            try:
                async with asyncio.timeout_at(end):
                    response = await self.transport(a, m, messages, max(.001, end - time.monotonic()), **params)
                    first = await anext(response) if params.get("stream") else None
                actual = getattr(first if first is not None else response, "model", None) or m
                self.seen.add((a.id, m))
                reason = "least in-flight with round-robin tie break" + ("; fallback after provider failure" if attempt > 1 else "")
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
                if category == "malformed_request":
                    break
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
