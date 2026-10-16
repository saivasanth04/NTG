import asyncio
import json
import random
import time

import httpx
import pytest

from ntg.config import accounts_from_env
from ntg.discovery import deployment, discover, zero_pricing
from ntg.engine import RoutingService
from ntg.failures import classify, duration
from ntg.gateway import create_app
from ntg.types import Account, ChatRequest, Deployment, RoutingError


def dep(alias="a", model="m", group=None, caps=None, provider="groq"):
    a = Account(provider, alias, "never-print-this-key", quota_group=group or alias)
    return Deployment(a, model, caps or {"chat", "tools", "json", "json_schema"},
                      100000, 8192, True, "unknown", "verified", "test fixture", time.time())


def req(**kwargs):
    return ChatRequest(messages=[{"role": "user", "content": "hello"}], **kwargs)


def answer(content="OK", headers=None):
    return ({"id": "test", "object": "chat.completion", "created": 1, "model": "m",
             "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}}, headers or {})


class Fake:
    def __init__(self, action=None):
        self.calls = []
        self.action = action

    async def complete(self, d, r, timeout):
        self.calls.append(d.id)
        if self.action:
            return await self.action(d, r, timeout)
        return answer()


class Upstream(Exception):
    def __init__(self, status, headers=None):
        self.status_code = status
        self.headers = headers or {}
        super().__init__("SECRET provider error with sensitive prompt")


def test_accounts_arbitrary_count_duplicate_keys_shared_default():
    values = {f"GROQ_API_KEY_{i}": f"key-{i}" for i in range(1, 40)}
    values.update(GROQ_API_KEY_40="key-1", GROQ_40_QUOTA_GROUP="fake-independent")
    accounts = accounts_from_env(values=values)
    assert len(accounts) == 40
    assert len({a.quota_group for a in accounts}) == 1
    assert "key-1" not in repr(dep())


def test_pricing_fail_closed():
    assert zero_pricing({"prompt": "0", "completion": "0"})
    for pricing in ({}, {"prompt": "0"}, {"prompt": "0", "completion": "0.1"},
                    {"prompt": "0", "completion": "0", "request": "1"},
                    {"prompt": "unknown", "completion": "0"}):
        assert not zero_pricing(pricing)


def test_discovery_free_access_separate():
    a = dep().account
    d = deployment(a, "m", {"context_window": 4096}, {})
    assert d.listed and d.access == "unknown" and d.free == "unknown"
    policy = {"context": 4096, "capabilities": ["chat", "tools"], "free_eligible": True,
              "source": "https://console.groq.com/docs/models", "reviewed_on": "2026-10-16"}
    assert deployment(a, "m", {}, policy).free == "unknown"
    a.free_verified = True
    assert deployment(a, "m", {}, policy).free == "verified"


async def test_discovery_pagination_and_failed_listing():
    def handler(request):
        assert "key" not in request.url.params
        if request.url.params.get("pageToken") == "next":
            return httpx.Response(200, json={"models": [{"name": "models/b"}]})
        return httpx.Response(200, json={"models": [{"name": "models/a"}], "nextPageToken": "next"})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        ds, report = await discover([Account("gemini", "g", "secret")], {"models": {}}, client)
    assert len(ds) == 2 and report[0]["listed_count"] == 2
    assert all(d.free == "unknown" for d in ds)


async def test_success_and_metadata_no_secrets():
    service = RoutingService([dep()], Fake())
    result = await service.chat(req())
    assert result["routing"]["account"] == "a"
    assert result["routing"]["usage"]["total_tokens"] == 6
    assert "never-print" not in json.dumps(service.status())


@pytest.mark.parametrize("kwargs,caps", [({"tools": [{"type": "function"}]}, {"chat"}),
    ({"response_format": {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}}}, {"chat", "json"}),
    ({"required_capabilities": {"vision"}}, {"chat"}), ({"input_tokens": 200000}, {"chat"})])
async def test_capability_and_context_filter(kwargs, caps):
    transport = Fake()
    service = RoutingService([dep(caps=caps)], transport)
    with pytest.raises(RoutingError, match="no_eligible"):
        await service.chat(req(**kwargs))
    assert not transport.calls


async def test_fair_distribution_not_model_count_weighted():
    transport = Fake()
    service = RoutingService([dep("a", f"m{i}") for i in range(10)] + [dep("b")], transport)
    for _ in range(20):
        await service.chat(req())
    assert service.account_calls == {"a": 10, "b": 10}


async def test_rate_limit_fallback_shared_cooldown():
    async def action(d, r, timeout):
        if d.account.alias in ("a", "b"):
            raise Upstream(429, {"Retry-After": "120"})
        return answer()
    service = RoutingService([dep("a", group="same"), dep("b", group="same"), dep("c")], Fake(action), rng=random.Random(1))
    # Force initial failure, then another ownership group remains eligible.
    service.account_calls["c"] = 10
    result = await service.chat(req())
    assert result["routing"]["account"] == "c"
    assert len(result["routing"]["history"]) == 2
    assert service._ready_at(service.deployments[0]) - service.clock() >= 119
    assert "SECRET" not in json.dumps(result)


@pytest.mark.parametrize("status,scope", [(401,"account"), (402,"group"), (429,"group"),
    (403,"deployment"), (404,"deployment"), (502,"provider"), (503,"deployment")])
def test_failure_scopes(status, scope):
    assert classify(Upstream(status), 0).scope == scope


async def test_invalid_request_not_retried():
    async def fail(*args):
        raise Upstream(400)
    transport = Fake(fail)
    service = RoutingService([dep("a"), dep("b")], transport)
    with pytest.raises(RoutingError, match="invalid_request"):
        await service.chat(req())
    assert len(transport.calls) == 1


async def test_success_headers_preemptive_cooldown():
    async def action(*args):
        return answer(headers={"x-ratelimit-remaining-tokens": "0", "x-ratelimit-reset-tokens": "2m3s"})
    d = dep()
    service = RoutingService([d], Fake(action))
    await service.chat(req())
    assert service._ready_at(d) - service.clock() > 122


async def test_deadline_and_capacity_exhaustion():
    service = RoutingService([dep()], Fake())
    service._cool(service.deployments[0], "group", 60)
    start = time.monotonic()
    with pytest.raises(RoutingError, match="capacity_wait_deadline"):
        await service.chat(req(deadline_seconds=.03))
    assert time.monotonic() - start < .2
    assert service.admitted == service.active == 0


async def test_bounded_queue_concurrency_and_cancellation():
    entered, release = asyncio.Event(), asyncio.Event()
    active, peak = 0, 0
    async def block(*args):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        entered.set()
        try:
            await release.wait()
            return answer()
        finally:
            active -= 1
    service = RoutingService([dep()], Fake(block), max_active=1, max_queue=1)
    first = asyncio.create_task(service.chat(req()))
    await entered.wait()
    second = asyncio.create_task(service.chat(req()))
    await asyncio.sleep(.01)
    with pytest.raises(RoutingError, match="queue_full"):
        await service.chat(req())
    second.cancel()
    with pytest.raises(asyncio.CancelledError):
        await second
    release.set()
    await first
    assert peak == 1 and service.admitted == service.active == 0


async def test_shared_group_concurrency():
    active, peak = 0, 0
    async def action(*args):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(.005)
        active -= 1
        return answer()
    service = RoutingService([dep("a", group="one"), dep("b", group="one")], Fake(action))
    await asyncio.gather(*(service.chat(req()) for _ in range(12)))
    assert peak == 1


async def test_timeout_fallback():
    async def action(d, *args):
        if d.account.alias == "a":
            await asyncio.sleep(1)
        return answer()
    service = RoutingService([dep("a"), dep("b")], Fake(action), attempt_timeout=.01)
    service.account_calls["b"] = 1
    result = await service.chat(req())
    assert result["routing"]["history"][0]["outcome"] == "timeout"
    assert result["routing"]["account"] == "b"


async def test_structured_output_validation_and_fallback():
    async def action(d, *args):
        return answer("not JSON" if d.account.alias == "a" else '{"value": 1}')
    service = RoutingService([dep("a"), dep("b")], Fake(action))
    service.account_calls["b"] = 1
    result = await service.chat(req(response_format={"type": "json_schema", "json_schema": {
        "name": "test", "schema": {"type": "object", "required": ["value"], "properties": {"value": {"type": "integer"}}}}}))
    assert result["routing"]["history"][0]["outcome"] == "invalid_structured_output"


async def test_gateway_auth_validation_and_openai_shape():
    service = RoutingService([dep()], Fake())
    token = "x" * 32
    app = create_app(service, token)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.get("/status")).status_code == 401
        headers = {"Authorization": f"Bearer {token}"}
        good = await client.post("/v1/chat/completions", headers=headers, json=req().model_dump(mode="json"))
        assert good.status_code == 200 and good.json()["choices"][0]["message"]["content"] == "OK"
        bad = await client.post("/v1/chat/completions", headers=headers, json={"messages": "private secret"})
        assert bad.status_code == 400 and "private secret" not in bad.text
        assert (await client.get("/v1/models", headers=headers)).json()["object"] == "list"


def test_retry_after_dates_and_duration():
    assert duration("1m2.5s") == 62.5
    assert duration("bogus") is None
    assert duration("Thu, 01 Jan 2099 00:00:00 GMT") > 100


async def test_expired_free_evidence_rejected():
    d = dep()
    d.checked_at = time.time() - 90000
    with pytest.raises(RoutingError, match="no_eligible"):
        await RoutingService([d], Fake()).chat(req())


async def test_queue_bound_when_groups_limit_active_slots():
    service = RoutingService([dep()], Fake(), max_active=4, max_queue=1)
    service._cool(service.deployments[0], "group", 10)
    first = asyncio.create_task(service.chat(req(deadline_seconds=.05)))
    await asyncio.sleep(.01)
    with pytest.raises(RoutingError, match="queue_full"):
        await service.chat(req())
    with pytest.raises(RoutingError, match="capacity_wait_deadline"):
        await first
    assert service.waiting == 0


async def test_single_deployment_bounded_retry_after_cooldown():
    transport = Fake()
    async def action(*args):
        if len(transport.calls) == 1:
            raise TimeoutError()
        return answer()
    transport.action = action
    service = RoutingService([dep()], transport)
    # Make transient cooldown deterministic and short without weakening production Retry-After.
    original_cool = service._cool
    service._cool = lambda d, scope, seconds: original_cool(d, scope, .005)
    result = await service.chat(req())
    assert len(result["routing"]["history"]) == 2


async def test_transport_uses_litellm_and_no_nested_retries(monkeypatch):
    from ntg.transport import LiteLLMTransport, litellm
    seen = {}
    class Response:
        _hidden_params = {"additional_headers": {"x-ratelimit-remaining-tokens": "0"}}
        def model_dump(self, **kwargs):
            return answer()[0]
    async def fake_completion(**kwargs):
        seen.update(kwargs)
        return Response()
    monkeypatch.setattr(litellm, "acompletion", fake_completion)
    d = dep(provider="openrouter", model="model:free")
    _, headers = await LiteLLMTransport().complete(d, req(), 2)
    assert seen["num_retries"] == 0 and seen["caching"] is False
    assert seen["extra_body"]["provider"]["require_parameters"] is True
    assert seen["extra_body"]["provider"]["max_price"]["prompt"] == 0
    assert headers["x-ratelimit-remaining-tokens"] == "0"


async def test_openai_sdk_gateway_integration():
    from openai import AsyncOpenAI
    token = "x" * 32
    app = create_app(RoutingService([dep()], Fake()), token)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as http:
        client = AsyncOpenAI(base_url="http://test/v1", api_key=token, http_client=http, max_retries=0)
        result = await client.chat.completions.create(model="auto", messages=[{"role": "user", "content": "hello"}])
        assert result.choices[0].message.content == "OK"
        assert result.model_extra["routing"]["account"] == "a"


async def test_vision_inferred_and_filtered():
    service = RoutingService([dep(caps={"chat"})], Fake())
    request = ChatRequest(messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}]}])
    with pytest.raises(RoutingError, match="no_eligible"):
        await service.chat(request)


def test_provider_unsupported_params_can_fallback():
    error_type = type("UnsupportedParamsError", (Upstream,), {})
    failure = classify(error_type(400), 0)
    assert failure.scope == "deployment" and failure.retry


async def test_required_tool_output_not_silently_accepted():
    service = RoutingService([dep()], Fake(), max_attempts=1)
    with pytest.raises(RoutingError, match="attempts_exhausted"):
        await service.chat(req(tools=[{"type": "function"}], tool_choice="required"))


def test_paid_openrouter_policy_cannot_override_catalog():
    a = Account("openrouter", "a", "secret", free_verified=True)
    d = deployment(a, "paid/model", {"pricing": {"prompt": "1", "completion": "1"}},
                   {"source": "https://openrouter.ai", "reviewed_on": "2026-10-16", "free_eligible": True})
    assert d.free != "verified"
