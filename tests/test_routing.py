"""Offline behavioral tests: transport calls are mocked, routing is real."""

import json
import io
from contextlib import redirect_stdout
import os
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ntg.core.exceptions import (parse_provider_error, CATEGORY_SERVER_ERROR,
                                 CATEGORY_REQUEST_ERROR, CATEGORY_UNKNOWN_ERROR)
from ntg.core.models import CircuitState, Deployment, ModelCapabilities, QuotaInfo, QuotaScope
from ntg.router.engine import UnifiedNTGRouter
from ntg.router.policy import score
from ntg.router.requirements import extract_request_requirements, NoEligibleDeploymentsError
from ntg.router.state import StateManager, clean_for_persistence
from ntg.providers.base import enrich_capabilities


class ProviderError(Exception):
    def __init__(self, status, message, headers=None):
        super().__init__(message)
        self.status_code = status
        self.headers = headers or {}


class FakeRouter:
    def __init__(self, model_list, **kwargs):
        self.model_list = model_list
    def set_model_list(self, models):
        self.model_list = models


def deployment(dep_id, account=None, provider="groq", **kwargs):
    return Deployment(id=dep_id, provider=provider, account=account or dep_id,
        model="model-" + dep_id, logical_model=provider, litellm_model=provider + "/" + dep_id,
        api_key="private-credential-" + (account or dep_id),
        capabilities=kwargs.pop("capabilities", ModelCapabilities(coding=True, streaming=True,
            tool_calling=True, vision=True, structured_output=True, context_window=20000)), **kwargs)


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.transport = patch("ntg.router.engine.litellm.completion")
        self.call = self.transport.start()
        self.addCleanup(self.transport.stop)
        self.call.side_effect = lambda **kw: SimpleNamespace(choices=[], model=kw["model"])
        self.pool = patch("ntg.router.engine.Router", FakeRouter)
        self.pool.start()
        self.addCleanup(self.pool.stop)

    def router(self, deps, **kwargs):
        state = StateManager(str(Path(self.temp.name) / (str(id(deps)) + ".json")))
        router = UnifiedNTGRouter(accounts=deps, state_manager=state, **kwargs)
        self.addCleanup(router.close)
        return router

    def test_determinism_and_account_distribution(self):
        a, b = deployment("a"), deployment("b")
        router = self.router([b, a])
        messages = [{"role": "user", "content": "hello"}]
        self.assertIs(router.completion(messages)._ntg_deployment, a)
        self.assertIs(router.completion(messages)._ntg_deployment, b)
        self.assertEqual([a.metrics.user_requests, b.metrics.user_requests], [1, 1])

    def test_ranking_uses_quota_latency_health_and_fit(self):
        router = self.router([])
        req = extract_request_requirements(prompt="```code```")
        a, b = deployment("a"), deployment("b")
        a.quota = QuotaInfo(rpm_limit=100, rpm_remaining=1)
        a.latency_ewma, b.latency_ewma = 10, 0.1
        a.consecutive_failures = 1
        a.capabilities.coding = None
        self.assertGreater(score(b, req, {}, {}).total, score(a, req, {}, {}).total)
        self.assertEqual(router._rank_eligible_deployments([a, b], requirements=req), [b, a])

    def test_soft_coding_keeps_fallbacks(self):
        a, b = deployment("a"), deployment("b")
        b.capabilities.coding = None
        router = self.router([a, b], num_retries=0)
        self.call.side_effect = [ProviderError(503, "overloaded"), SimpleNamespace(choices=[])]
        self.assertIs(router.completion([{"role": "user", "content": "```python```"}])._ntg_deployment, b)

    def test_protocol_unknown_is_ineligible(self):
        a = deployment("a")
        a.capabilities.vision = None
        router = self.router([a])
        with self.assertRaises(NoEligibleDeploymentsError):
            router.completion([{"role": "user", "content": [{"type": "input_image", "image_url": "https://example.test/image"}]}])
        self.call.assert_not_called()

    def test_account_auth_quarantine_and_failover(self):
        a, b, c = deployment("a", "shared"), deployment("b", "shared"), deployment("c")
        router = self.router([a, b, c], num_retries=0)
        self.call.side_effect = [ProviderError(401, "invalid api key " + a.api_key), SimpleNamespace(choices=[])]
        self.assertIs(router.completion([])._ntg_deployment, c)
        self.assertEqual(a.circuit_state, CircuitState.AUTH_FAILED)
        self.assertEqual(b.circuit_state, CircuitState.AUTH_FAILED)
        self.assertEqual(b.metrics.upstream_attempts, 0)
        self.assertNotIn(a.api_key, Path(router.state_manager.state_file).read_text())

    def test_model_quarantine_does_not_quarantine_account(self):
        a, b = deployment("a", "shared"), deployment("b", "shared")
        router = self.router([a, b])
        self.call.side_effect = [ProviderError(404, "model not found"), SimpleNamespace(choices=[])]
        self.assertIs(router.completion([])._ntg_deployment, b)
        self.assertEqual(a.circuit_state, CircuitState.QUARANTINED)
        self.assertEqual(b.circuit_state, CircuitState.HEALTHY)

    def test_bad_request_stops_without_poisoning_health(self):
        a, b = deployment("a"), deployment("b")
        router = self.router([a, b])
        self.call.side_effect = ProviderError(400, "invalid response format")
        with self.assertRaises(ProviderError):
            router.completion([])
        self.assertEqual(self.call.call_count, 1)
        self.assertEqual(a.consecutive_failures, 0)
        self.assertIsNone(a.quota.reset_at)

    def test_unknown_errors_do_not_retry_or_trip(self):
        a = deployment("a")
        router = self.router([a])
        self.call.side_effect = ValueError("programming bug")
        with self.assertRaises(ValueError):
            router.completion([])
        self.assertEqual(self.call.call_count, 1)
        self.assertEqual(a.consecutive_failures, 0)

    def test_transient_retry_is_bounded_and_has_no_quota_cooldown(self):
        a = deployment("a", max_consecutive_failures=10)
        router = self.router([a], num_retries=2)
        self.call.side_effect = ProviderError(503, "service unavailable")
        with self.assertRaises(ProviderError):
            router.completion([])
        # Default RetryPolicy allows one server repeat, below global max=2.
        self.assertEqual(self.call.call_count, 2)
        self.assertIsNone(a.quota.reset_at)
        self.assertEqual(a.metrics.upstream_attempts, 2)
        self.assertEqual(a.metrics.user_requests, 1)

    def test_failover_precedes_repeat_and_changes_account(self):
        a, b, c = deployment("a", "one"), deployment("b", "one"), deployment("c", "two")
        router = self.router([a, b, c], num_retries=0)
        self.call.side_effect = [ProviderError(503, "service unavailable"), SimpleNamespace(choices=[])]
        self.assertIs(router.completion([])._ntg_deployment, c)

    def test_explicit_provider_order(self):
        a, b, c = deployment("a"), deployment("b", provider="gemini"), deployment("c", provider="nvidia")
        router = self.router([c, b, a], provider_fallbacks={"groq": ["gemini", "nvidia"]})
        self.call.side_effect = [ProviderError(503, "service unavailable"), SimpleNamespace(choices=[])]
        self.assertIs(router.completion([], model="groq")._ntg_deployment, b)

    def test_unrelated_caller_fallback_edge_is_ignored(self):
        a, b = deployment("a"), deployment("b", provider="gemini")
        router = self.router([a, b], num_retries=0)
        self.call.side_effect = ProviderError(503, "service unavailable")
        with self.assertRaises(ProviderError):
            router.completion([], model="a", fallbacks=[{"unrelated": ["b"]}])
        self.assertEqual(self.call.call_count, 1)

    def test_caller_fallback_cycles_and_global_destinations(self):
        a, b = deployment("a"), deployment("b", provider="gemini")
        router = self.router([a, b], num_retries=0)
        self.call.side_effect = [ProviderError(503, "service unavailable"), SimpleNamespace(choices=[])]
        response = router.completion([], model="a", fallbacks=[{"a": ["auto"]}, {"b": ["a"]}])
        self.assertIs(response._ntg_deployment, b)
        self.assertEqual(self.call.call_count, 2)

    def test_pool_inspection_does_not_claim_probes(self):
        a, b = deployment("a"), deployment("b")
        a.transition_to_half_open()
        b.transition_to_half_open()
        router = self.router([a, b])
        router._build_request_pool_and_fallbacks("auto", extract_request_requirements(), claim_probe=True)
        self.assertEqual(a.half_open_probes + b.half_open_probes, 0)

    def test_half_open_single_probe_concurrent_admission(self):
        a = deployment("a")
        a.transition_to_half_open()
        router = self.router([a])
        self.assertIsNotNone(router.lifecycle.admit(a, "first"))
        self.assertIsNone(router.lifecycle.admit(a, "second"))
        router.lifecycle.release(a, "first")
        self.assertEqual(a.half_open_probes, 0)

    def test_quota_last_request_reserved_atomically(self):
        a, b = deployment("a", "shared"), deployment("b", "shared")
        for dep in (a, b):
            dep.quota = QuotaInfo(rpm_limit=10, rpm_remaining=1, quota_scope=QuotaScope.ACCOUNT)
        router = self.router([a, b])
        self.assertIsNotNone(router.lifecycle.admit(a, "one"))
        self.assertIsNone(router.lifecycle.admit(b, "two"))
        router.lifecycle.release(a, "one")

    def test_quota_429_scope_and_retry_after(self):
        a, b, c = deployment("a", "shared", "openrouter"), deployment("b", "shared", "openrouter"), deployment("c")
        router = self.router([a, b, c], num_retries=0)
        self.call.side_effect = [ProviderError(429, "rate limit", {"Retry-After": "60"}), SimpleNamespace(choices=[])]
        self.assertIs(router.completion([])._ntg_deployment, c)
        self.assertEqual(a.circuit_state, CircuitState.OPEN)
        self.assertEqual(b.circuit_state, CircuitState.OPEN)
        self.assertGreater(a.quota.reset_at, time.time())

    def test_late_success_cannot_erase_quarantine_or_cooldown(self):
        a = deployment("a")
        router = self.router([a])
        started, fingerprint = router.lifecycle.admit(a, "old")
        a.transition_to_quarantine("bad auth", CircuitState.AUTH_FAILED)
        router.lifecycle.finish(a, "old", started, fingerprint, response=SimpleNamespace())
        self.assertEqual(a.circuit_state, CircuitState.AUTH_FAILED)
        router.lifecycle.release(a, "old")

    def test_old_credentials_cannot_quarantine_rotated_key(self):
        a = deployment("a")
        router = self.router([a])
        started, fingerprint = router.lifecycle.admit(a, "old")
        a.api_key = "replacement-credential"
        router.lifecycle.finish(a, "old", started, fingerprint, error=parse_provider_error(ProviderError(401, "unauthorized"), "groq"))
        self.assertEqual(a.circuit_state, CircuitState.HEALTHY)
        router.lifecycle.release(a, "old")

    def test_stream_success_recorded_only_after_exhaustion(self):
        a = deployment("a")
        a.transition_to_half_open()
        router = self.router([a])
        self.call.side_effect = lambda **kw: iter(["one", "two"])
        stream = router.completion([], stream=True)
        self.assertEqual(a.metrics.successes, 0)
        self.assertEqual(next(stream), "one")
        self.assertEqual(a.half_open_probes, 1)
        self.assertEqual(list(stream), ["two"])
        self.assertEqual(a.metrics.successes, 1)
        self.assertEqual(a.half_open_probes, 0)
        self.assertFalse(router._active_requests)

    def test_stream_close_releases_resources_without_claiming_success(self):
        a = deployment("a")
        a.transition_to_half_open()
        router = self.router([a])
        self.call.side_effect = lambda **kw: iter(["one", "two"])
        stream = router.completion([], stream=True)
        next(stream)
        stream.close()
        self.assertEqual(a.metrics.successes, 0)
        self.assertEqual(a.half_open_probes, 0)
        self.assertEqual(a.metrics.user_requests, 1)
        self.assertFalse(router._active_requests)

    def test_stream_failover_before_first_chunk(self):
        a, b = deployment("a"), deployment("b")
        router = self.router([a, b])
        def failing():
            raise ProviderError(503, "overloaded")
            yield
        self.call.side_effect = [failing(), iter(["good"])]
        self.assertEqual(list(router.completion([], stream=True)), ["good"])
        self.assertEqual(self.call.call_count, 2)

    def test_stream_never_replays_after_first_chunk(self):
        a, b = deployment("a"), deployment("b")
        router = self.router([a, b])
        def partial():
            yield "one"
            raise ProviderError(503, "overloaded")
        self.call.side_effect = lambda **kw: partial()
        stream = router.completion([], stream=True)
        next(stream)
        with self.assertRaises(ProviderError):
            next(stream)
        self.assertEqual(self.call.call_count, 1)
        self.assertFalse(router._active_requests)

    def test_managed_or_foreign_callbacks_cannot_double_count(self):
        a = deployment("a")
        router = self.router([a])
        response = router.completion([])
        for managed, router_id in ((True, router.router_id), (False, "other")):
            kwargs = {"metadata": {"ntg_managed": managed, "ntg_router_id": router_id, "deployment_id": a.id}, "litellm_call_id": "late"}
            router.telemetry_logger.log_success_event(kwargs, response, 0, 1)
        self.assertEqual(a.metrics.successes, 1)
        self.assertEqual(a.metrics.upstream_attempts, 1)

    def test_snapshot_restore_failure_count_latency_and_rotated_auth(self):
        a = deployment("a")
        router = self.router([a])
        a.consecutive_failures, a.latency_ewma = 2, 0.2
        a.transition_to_quarantine("unauthorized", CircuitState.AUTH_FAILED)
        router.state_manager.snapshot_deployment(a)
        restored = deployment("a")
        state = StateManager(router.state_manager.state_file)
        state.apply_to_deployment(restored)
        self.assertEqual(restored.consecutive_failures, 2)
        self.assertEqual(restored.latency_ewma, 0.2)
        restored.api_key = "rotated-credential"
        restored.refresh()
        self.assertEqual(restored.circuit_state, CircuitState.HEALTHY)

    def test_transport_options_do_not_override_selected_credentials(self):
        a = deployment("a")
        router = self.router([a])
        router.completion([], api_key="override", api_base="https://wrong", temperature=0.2, num_retries=99)
        kw = self.call.call_args.kwargs
        self.assertEqual(kw["api_key"], a.api_key)
        self.assertEqual(kw["temperature"], 0.2)
        self.assertEqual(kw["num_retries"], 0)

    def test_concurrent_requests_reserve_distinct_accounts(self):
        a, b = deployment("a"), deployment("b")
        router = self.router([a, b])
        barrier = threading.Barrier(2)
        def transport(**kwargs):
            barrier.wait(timeout=5)
            return SimpleNamespace(choices=[], model=kwargs["model"])
        self.call.side_effect = transport
        results, errors = [], []
        def run():
            try:
                results.append(router.completion([])._ntg_deployment.id)
            except Exception as error:
                errors.append(error)
        threads = [threading.Thread(target=run) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertFalse(errors)
        self.assertCountEqual(results, ["a", "b"])
        self.assertFalse(router._active_requests)
        self.assertEqual(sum(router.lifecycle.inflight.values()), 0)

    def test_new_model_inherits_persisted_account_auth_quarantine(self):
        a = deployment("a", "shared")
        router = self.router([a])
        router._propagate_account_auth_quarantine("groq", "shared", "unauthorized", a.api_key)
        new = deployment("new", "shared")
        router.add_deployment(new)
        self.assertEqual(new.circuit_state, CircuitState.AUTH_FAILED)
        self.assertIsNone(router.lifecycle.admit(new, "blocked"))

    def test_scoped_quota_cannot_clear_auth_quarantine(self):
        a = deployment("a")
        a.transition_to_quarantine("bad auth", CircuitState.AUTH_FAILED)
        router = self.router([a])
        router._propagate_account_quota_exhaustion("groq", a.account, time.time() + 60, "quota")
        self.assertEqual(a.circuit_state, CircuitState.AUTH_FAILED)

    def test_account_revalidation_preserves_model_quarantine(self):
        a, b = deployment("a", "shared"), deployment("b", "shared")
        a.transition_to_quarantine("bad auth", CircuitState.AUTH_FAILED)
        b.transition_to_quarantine("model retired")
        router = self.router([a, b])
        with patch.object(router, "rediscover_models"):
            result = router.revalidate_account("groq", "shared", force=True, probe_fn=lambda d: True)
        self.assertTrue(result["a"])
        self.assertFalse(result["b"])
        self.assertEqual(b.circuit_state, CircuitState.QUARANTINED)

    def test_capability_failure_fails_over_without_quarantine(self):
        a, b = deployment("a"), deployment("b")
        router = self.router([a, b], num_retries=0)
        self.call.side_effect = [ProviderError(400, "model does not support tools"), SimpleNamespace(choices=[])]
        self.assertIs(router.completion([], tools=[{"type": "function"}])._ntg_deployment, b)
        self.assertEqual(a.circuit_state, CircuitState.HEALTHY)
        self.assertEqual(a.consecutive_failures, 0)

    def test_model_permission_does_not_quarantine_whole_account(self):
        a, b = deployment("a", "shared"), deployment("b", "shared")
        router = self.router([a, b], num_retries=0)
        self.call.side_effect = [ProviderError(403, "no access to this model"), SimpleNamespace(choices=[])]
        self.assertIs(router.completion([])._ntg_deployment, b)
        self.assertEqual(a.circuit_state, CircuitState.QUARANTINED)
        self.assertEqual(b.circuit_state, CircuitState.HEALTHY)

    def test_late_auth_has_precedence_over_quota_cooldown(self):
        a = deployment("a")
        router = self.router([a])
        started, fingerprint = router.lifecycle.admit(a, "old")
        a.transition_to_open(60, "quota", reset_at=time.time() + 60)
        router.lifecycle.finish(a, "old", started, fingerprint, error=parse_provider_error(ProviderError(401, "unauthorized"), "groq"))
        self.assertEqual(a.circuit_state, CircuitState.AUTH_FAILED)
        router.lifecycle.release(a, "old")

    def test_late_rate_limit_cannot_shorten_newer_cooldown(self):
        a = deployment("a")
        router = self.router([a])
        started, fingerprint = router.lifecycle.admit(a, "old")
        deadline = time.time() + 300
        a.transition_to_open(300, "quota", reset_at=deadline)
        info = parse_provider_error(ProviderError(429, "rate limit", {"Retry-After": "10"}), "groq")
        router.lifecycle.finish(a, "old", started, fingerprint, error=info)
        self.assertGreaterEqual(a.quota.reset_at, deadline)
        router.lifecycle.release(a, "old")

    def test_response_headers_do_not_refund_concurrent_reservations(self):
        a, b = deployment("a", "shared", "openrouter"), deployment("b", "shared", "openrouter")
        for dep in (a, b):
            dep.quota = QuotaInfo(rpm_remaining=5, rpm_limit=10, quota_scope=QuotaScope.ACCOUNT)
        router = self.router([a, b])
        started, fingerprint = router.lifecycle.admit(a, "first")
        router.lifecycle.admit(b, "second")
        response = SimpleNamespace(headers={"x-ratelimit-remaining-requests": "4"})
        router.lifecycle.finish(a, "first", started, fingerprint, response=response)
        self.assertEqual(a.quota.rpm_remaining, 3)
        self.assertEqual(b.quota.rpm_remaining, 3)
        router.lifecycle.release(a, "first")
        router.lifecycle.release(b, "second")

    def test_explicit_empty_provider_policy_disables_model_widening(self):
        a, b = deployment("a"), deployment("b")
        router = self.router([a, b], num_retries=0, provider_fallbacks={})
        self.call.side_effect = ProviderError(503, "overloaded")
        with self.assertRaises(ProviderError):
            router.completion([], model="a")
        self.assertEqual(self.call.call_count, 1)

    def test_cli_flags_and_agent_completion_integration(self):
        from ntg.cli.app import run_cli
        from ntg.agent.orchestrator import ArchitectureAwareAgent
        a = deployment("a")
        router = self.router([a])
        with patch("ntg.router.engine.UnifiedNTGRouter", return_value=router), redirect_stdout(io.StringIO()) as output:
            run_cli(["--help"])
            run_cli(["--router-only", "--model", "auto", "--capability", "coding", "hello"])
        for flag in ("--rediscover", "--reset-state", "--router-only", "--model", "--capability"):
            self.assertIn(flag, output.getvalue())
        agent = object.__new__(ArchitectureAwareAgent)
        agent._router = router
        agent.default_capabilities = ["coding"]
        self.assertIs(agent._invoke_router("hello", model="auto")._ntg_deployment, a)

    def test_agent_does_not_retry_validation_errors_without_capabilities(self):
        from ntg.agent.orchestrator import ArchitectureAwareAgent
        from unittest.mock import Mock
        agent = object.__new__(ArchitectureAwareAgent)
        agent._router = SimpleNamespace(completion=Mock(side_effect=ValueError("invalid requirements")))
        agent.default_capabilities = ["coding"]
        with self.assertRaises(ValueError):
            agent._invoke_router("hello", capabilities=["bad"])
        self.assertEqual(agent.router.completion.call_count, 1)
        self.assertEqual(agent.router.completion.call_args.kwargs["capabilities"], ["bad"])

    def test_server_retry_after_is_circuit_backoff_without_quota_pollution(self):
        a = deployment("a")
        router = self.router([a], num_retries=2)
        self.call.side_effect = ProviderError(503, "overloaded", {"Retry-After": "60"})
        with self.assertRaises(ProviderError):
            router.completion([])
        self.assertEqual(self.call.call_count, 1)
        self.assertEqual(a.circuit_state, CircuitState.OPEN)
        self.assertIsNone(a.quota.reset_at)

    def test_reasoning_generation_configuration_is_preserved(self):
        a = deployment("a")
        a.capabilities.reasoning = True
        router = self.router([a])
        router.completion([], reasoning={"effort": "high"})
        self.assertEqual(self.call.call_args.kwargs["reasoning"], {"effort": "high"})


class RequirementsAndStateTests(unittest.TestCase):
    def test_context_counts_schemas_tools_and_unicode(self):
        req = extract_request_requirements(messages=[{"role": "assistant", "content": "你好", "tool_calls": [{"function": {"arguments": "x" * 5000}}]}],
            tools=[{"type": "function", "function": {"parameters": {"description": "z" * 3000}}}], max_tokens=100)
        self.assertGreater(req.capabilities.context_window, 8000)
        self.assertTrue(req.capabilities.tool_calling)

    def test_plain_base64_text_is_not_an_image(self):
        req = extract_request_requirements(prompt="Explain the text base64, and data:image/")
        self.assertIsNone(req.capabilities.vision)

    def test_none_tool_choice_does_not_require_tools(self):
        req = extract_request_requirements(tool_choice="none")
        self.assertIsNone(req.capabilities.tool_calling)

    def test_capability_input_is_copied_and_unknown_rejected(self):
        caps = ModelCapabilities(coding=True, context_window=4096)
        req = extract_request_requirements(capabilities=caps, min_context=8000)
        self.assertEqual(caps.context_window, 4096)
        self.assertEqual(req.capabilities.context_window, 8000)
        with self.assertRaises(ValueError):
            extract_request_requirements(capabilities=["typo"])

    def test_positive_remaining_with_reset_header_is_available(self):
        a = deployment("a")
        a.quota.update_from_headers({"x-ratelimit-remaining-requests": "5", "x-ratelimit-reset-requests": "60"}, provider="groq")
        self.assertTrue(a.is_available())
        self.assertIsNone(a.quota.reset_at)
        self.assertIsNotNone(a.quota.rpm_reset_at)

    def test_exhausted_counter_recovers_after_window(self):
        quota = QuotaInfo(rpm_remaining=0, rpm_reset_at=100)
        self.assertTrue(quota.exhausted(99))
        self.assertFalse(quota.exhausted(101))

    def test_status_precedes_error_keywords(self):
        self.assertEqual(parse_provider_error(ProviderError(503, "model unavailable quota authentication"), "groq").category, CATEGORY_SERVER_ERROR)
        self.assertEqual(parse_provider_error(ProviderError(400, "invalid quota argument"), "groq").category, CATEGORY_REQUEST_ERROR)
        self.assertEqual(parse_provider_error(ValueError("bug"), "groq").category, CATEGORY_UNKNOWN_ERROR)
        self.assertEqual(parse_provider_error(ProviderError(400, "invalid token count 429"), "groq").category, CATEGORY_REQUEST_ERROR)

    def test_persistence_scrubs_payloads_and_keys(self):
        data = clean_for_persistence({"api_key": "secret", "messages": ["private"], "quota_scope": "account", "state_reason": "Bearer abcdefghijklmnop"})
        self.assertNotIn("api_key", data)
        self.assertNotIn("messages", data)
        self.assertNotIn("abcdefghijklmnop", data["state_reason"])

    def test_audio_requires_verified_audio_support(self):
        req = extract_request_requirements(messages=[{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": "base64", "format": "wav"}}]}], modalities=["text", "audio"])
        self.assertTrue(req.capabilities.audio_input)
        self.assertTrue(req.capabilities.audio_output)
        self.assertFalse(deployment("a").capabilities.satisfies(req.capabilities))

    def test_local_metadata_enrichment_keeps_provider_evidence(self):
        with patch("litellm.model_cost", {"groq/example": {"supports_vision": True, "supports_function_calling": True}}):
            caps = enrich_capabilities("groq/example", ModelCapabilities(vision=False))
            self.assertFalse(caps.vision)
            self.assertTrue(caps.tool_calling)
            self.assertIsNone(enrich_capabilities("groq/unknown", ModelCapabilities()).tool_calling)

    def test_corrupt_numeric_state_is_safely_normalized(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            path.write_text(json.dumps({"deployments": {"a": {"circuit_state": "OPEN", "circuit_open_until": "broken", "state_updated_at": "broken"}}, "quota": {"a": {"rpm_remaining": "broken"}}, "metrics": {"a": {"successes": "broken"}}}))
            state = StateManager(str(path))
            dep = deployment("a")
            state.apply_to_deployment(dep)
            self.assertEqual(dep.circuit_state, CircuitState.HALF_OPEN)
            self.assertIsNone(dep.quota.rpm_remaining)
            self.assertEqual(dep.metrics.successes, 0)

    def test_deployment_repr_does_not_expose_api_key(self):
        dep = deployment("a")
        self.assertNotIn(dep.api_key, repr(dep))


class LiteLLMIntegrationTests(unittest.TestCase):
    """Exercise the installed LiteLLM adapter with its official mock transport."""
    def test_completion_and_stream_with_real_litellm(self):
        with tempfile.TemporaryDirectory() as directory:
            dep = deployment("adapter", provider="openai")
            dep.litellm_model = "openai/gpt-4o-mini"
            router = UnifiedNTGRouter(accounts=[dep], state_manager=StateManager(str(Path(directory) / "state.json")))
            try:
                with patch("requests.sessions.Session.request", side_effect=AssertionError("Network forbidden")):
                    response = router.completion([{"role": "user", "content": "hello"}], mock_response="ok")
                    self.assertEqual(response.choices[0].message.content, "ok")
                    self.assertIs(response._ntg_deployment, dep)
                    chunks = list(router.completion([{"role": "user", "content": "hello"}], mock_response="stream", stream=True))
                    self.assertTrue(chunks)
                self.assertEqual(dep.metrics.successes, 2)
                self.assertEqual(dep.metrics.user_requests, 2)
            finally:
                router.close()


if __name__ == "__main__":
    unittest.main()
