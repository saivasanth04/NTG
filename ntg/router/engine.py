"""LiteLLM-based multi-provider smart routing engine with NTG intelligence."""

from __future__ import annotations

import logging
from functools import wraps
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set
import uuid

import litellm
from litellm import RetryPolicy, Router

from ntg.cli.diagnostics import (
    print_account_status,
    print_divider,
    print_request_execution,
)
from ntg.core.config import (
    DEFAULT_LOGICAL_MODEL,
    DEFAULT_NUM_RETRIES,
    GEMINI_KEYS,
    GROQ_KEYS,
    MAX_RETRY_BUDGET,
    MODEL_GROUP_AUTO,
    MODEL_GROUP_GEMINI,
    MODEL_GROUP_GROQ,
    MODEL_GROUP_NTG_AUTO,
    PROVIDER_FALLBACKS,
)
from ntg.core.exceptions import (
    CATEGORY_AUTH_ERROR,
    CATEGORY_REQUEST_ERROR,
    CATEGORY_SERVER_ERROR,
    parse_provider_error,
)
from ntg.core.models import (
    CircuitState,
    Deployment,
    DeploymentMetrics,
    ModelCapabilities,
    QuotaScope,
)
from ntg.core.utils import sanitize_secret, utc_string
from ntg.providers.discovery import build_all_deployments
from ntg.providers.gemini import discover_gemini_models
from ntg.providers.groq import discover_groq_models
from ntg.router.requirements import (
    NoEligibleDeploymentsError,
    RequestRequirements,
    extract_request_requirements,
    parse_capabilities,
)
from ntg.router.policy import CAPABILITY_GROUPS, GLOBAL_GROUPS, compatible, matches, rank
from ntg.router.lifecycle import RoutingLifecycle
from ntg.router.state import StateManager
from ntg.router.telemetry import ActiveRequest, NTGTelemetryLogger


def _routing_locked(method):
    @wraps(method)
    def locked(self, *args, **kwargs):
        with self.lifecycle.lock:
            return method(self, *args, **kwargs)
    return locked


class UnifiedNTGRouter:
    """NTG owns deterministic routing and lifecycle; LiteLLM adapts provider calls."""

    def __init__(
        self,
        accounts: Optional[List[Deployment]] = None,
        default_model: Optional[str] = None,
        num_retries: Optional[int] = None,
        retry_policy: Optional[RetryPolicy] = None,
        provider_fallbacks: Optional[Dict[str, List[str]]] = None,
        openrouter_keys: Optional[Dict[str, str]] = None,
        groq_keys: Optional[Dict[str, str]] = None,
        nvidia_keys: Optional[Dict[str, str]] = None,
        cohere_keys: Optional[Dict[str, str]] = None,
        gemini_keys: Optional[Dict[str, str]] = None,
        gemini_api_key: Optional[str] = None,
        state_manager: Optional[StateManager] = None,
    ):
        litellm.suppress_debug_info = True
        litellm.set_verbose = False
        logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
        logging.getLogger("LiteLLM Router").setLevel(logging.CRITICAL)

        self.state_manager = state_manager or StateManager()

        self.provider_fallbacks = (
            dict(provider_fallbacks)
            if provider_fallbacks is not None
            else dict(PROVIDER_FALLBACKS)
        )

        # 1. Build all deployments across providers and restore persisted state
        if accounts is not None:
            self.deployments: List[Deployment] = list(accounts)
            for dep in self.deployments:
                self.state_manager.apply_to_deployment(dep)
                dep.metrics = self.state_manager.get_metrics(dep.id)
        else:
            g_keys = gemini_keys
            if g_keys is None and gemini_api_key and gemini_api_key != "YOUR_NEW_GOOGLE_API_KEY":
                g_keys = {"account_1": gemini_api_key}

            self.deployments = build_all_deployments(
                openrouter_keys=openrouter_keys,
                groq_keys=groq_keys,
                nvidia_keys=nvidia_keys,
                cohere_keys=cohere_keys,
                gemini_keys=g_keys,
                state_manager=self.state_manager,
            )

        self.deployment_map: Dict[str, Deployment] = {dep.id: dep for dep in self.deployments}
        if len(self.deployment_map) != len(self.deployments):
            raise ValueError("Deployment IDs must be unique")
        self.state_manager.register_secrets(dep.api_key for dep in self.deployments)

        # Legacy LiteLLM registry is retained for compatibility and inspection.
        # Its selection strategy is never used by the NTG request executor.
        model_list = self._model_entries(self.deployments)

        # 3. Resolve default model
        self.default_model = default_model or DEFAULT_LOGICAL_MODEL

        # 4. Configure bounded retry budget
        if num_retries is not None:
            self.num_retries = max(0, min(num_retries, MAX_RETRY_BUDGET))
        else:
            self.num_retries = DEFAULT_NUM_RETRIES

        self.retry_policy = retry_policy or RetryPolicy(
            BadRequestErrorRetries=0,
            ContentPolicyViolationErrorRetries=0,
            AuthenticationErrorRetries=0,
            RateLimitErrorRetries=1,
            TimeoutErrorRetries=1,
            InternalServerErrorRetries=1,
        )

        # 5. Retain the public LiteLLM model registry with nested retries disabled
        self.router = Router(
            model_list=model_list,
            routing_strategy="simple-shuffle",
            cooldown_time=60.0,
            allowed_fails=1,
            num_retries=0,
            retry_policy=None,
            default_max_parallel_requests=None,
            fallbacks=None,
            max_fallbacks=max(100, len(self.deployments) * 2 + 20),
            retry_after=0,
            model_group_alias={
                MODEL_GROUP_NTG_AUTO: MODEL_GROUP_AUTO,
            },
        )
        self.router.max_fallbacks = max(100, len(self.deployments) * 2 + 20)
        self.router.retry_after = 0

        # 6. Synchronization and request context
        self._model_pool_lock = threading.Lock()
        self._last_pool_fingerprint: Optional[tuple] = tuple(
            sorted((dep.id, dep.routing_config_fingerprint()) for dep in self.deployments)
        )
        self._request_context = threading.local()
        self._active_requests: Dict[str, ActiveRequest] = {}
        self._active_requests_lock = threading.Lock()
        self.unknown_metrics = self.state_manager.get_metrics("unknown")

        self.router_id = uuid.uuid4().hex[:8]

        # LiteLLM is a transport; NTG owns selection, admission and outcomes.
        # No global callback mutation or private Router monkey-patches.
        self.lifecycle = RoutingLifecycle(self)
        self.telemetry_logger = NTGTelemetryLogger(self)

    @staticmethod
    def _model_entries(deployments: List[Deployment]) -> List[Dict[str, Any]]:
        entries = []
        for dep in deployments:
            if not dep.is_pool_active():
                continue
            groups = [dep.logical_model, MODEL_GROUP_AUTO, dep.id]
            groups.extend(name for name in sorted(CAPABILITY_GROUPS) if getattr(dep.capabilities, name) is True)
            for group in dict.fromkeys(groups):
                entries.append(dep.to_litellm_dict(group_override=group))
        return entries

    def close(self) -> None:
        """Deregister callbacks and release resources."""
        if hasattr(self, "telemetry_logger") and self.telemetry_logger:
            litellm.callbacks = [cb for cb in litellm.callbacks if cb is not self.telemetry_logger]
            litellm.input_callback = [
                cb for cb in litellm.input_callback if cb is not self.telemetry_logger
            ]
        with self._active_requests_lock:
            self._active_requests.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @_routing_locked
    def add_deployment(self, deployment: Deployment) -> None:
        """Add a deployment to the router and synchronize the LiteLLM model pool."""
        if deployment.id in self.deployment_map:
            self.update_deployment(deployment)
            return
        self.state_manager.register_secrets([deployment.api_key])
        self.state_manager.apply_to_deployment(deployment)
        deployment.metrics = self.state_manager.get_metrics(deployment.id)
        if deployment not in self.deployments:
            self.deployments.append(deployment)
        self.deployment_map[deployment.id] = deployment
        self.router.max_fallbacks = max(100, len(self.deployments) * 2 + 20)
        self._sync_router_model_pool()

    @_routing_locked
    def remove_deployment(self, deployment_id: str) -> bool:
        """Remove a deployment from the router and synchronize the LiteLLM model pool."""
        initial_len = len(self.deployments)
        self.deployments = [d for d in self.deployments if d.id != deployment_id]
        self.deployment_map.pop(deployment_id, None)
        if len(self.deployments) < initial_len:
            self._sync_router_model_pool()
            return True
        return False

    @_routing_locked
    def update_deployment(self, deployment: Deployment) -> bool:
        """Update an existing deployment and synchronize the LiteLLM model pool."""
        for i, d in enumerate(self.deployments):
            if d.id == deployment.id:
                self.state_manager.register_secrets([deployment.api_key])
                self.deployments[i] = deployment
                self.deployment_map[deployment.id] = deployment
                self._sync_router_model_pool()
                return True
        return False

    def _register_active_request(
        self, request_id: str, selected_dep: Optional[Deployment] = None
    ) -> ActiveRequest:
        """Register active request context for cross-thread attempt coordination."""
        with self._active_requests_lock:
            req = ActiveRequest(
                request_id=request_id,
                router_id=self.router_id,
                selected_dep=selected_dep,
            )
            self._active_requests[request_id] = req
            return req

    def _get_active_request(self, request_id: Optional[str]) -> Optional[ActiveRequest]:
        """Lookup active request context by request ID."""
        if not request_id:
            return None
        with self._active_requests_lock:
            return self._active_requests.get(request_id)

    def _unregister_active_request(self, request_id: str) -> Optional[ActiveRequest]:
        """Unregister active request context when request terminates."""
        with self._active_requests_lock:
            return self._active_requests.pop(request_id, None)

    @_routing_locked
    def _sync_router_model_pool(self) -> None:
        """Fast, thread-safe synchronization of the shared LiteLLM model pool."""
        with self._model_pool_lock:
            fingerprint_items = [
                (dep.id, dep.routing_config_fingerprint()) for dep in self.deployments
            ]
            pool_fingerprint = tuple(sorted(fingerprint_items))
            if self._last_pool_fingerprint == pool_fingerprint:
                return

            self.deployment_map = {dep.id: dep for dep in self.deployments}
            self.router.max_fallbacks = max(100, len(self.deployments) * 2 + 20)

            self.router.set_model_list(self._model_entries(self.deployments))
            self._last_pool_fingerprint = pool_fingerprint

    def _record_request_attempt(self, deployment: Deployment) -> None:
        """Track deployment attempted in thread-local request context and record attempt metric."""
        attempts = getattr(self._request_context, "attempted_deployments", None)
        if attempts is not None and isinstance(attempts, list):
            attempts.append(deployment)
        deployment.metrics.record_attempt()

    def _propagate_account_auth_quarantine(
        self,
        provider: str,
        account: str,
        reason: str,
        failed_key: Optional[str] = None,
    ) -> None:
        """Propagate an authentication failure quarantine across all deployments under the same account."""
        self.lifecycle.quarantine_account(provider, account, reason, failed_key)
        self._sync_router_model_pool()

    def _probe_deployment_credentials(self, deployment: Deployment) -> bool:
        """Probe deployment credentials with a minimal lightweight request."""
        if not deployment.api_key or not deployment.api_key.strip():
            return False

        try:
            litellm.completion(
                model=deployment.litellm_model,
                messages=[{"role": "user", "content": "ping"}],
                api_key=deployment.api_key,
                api_base=deployment.api_base,
                max_tokens=1,
                timeout=10,
                num_retries=0,
            )
            return True
        except Exception as e:
            info = parse_provider_error(e, provider=deployment.provider)
            if info.category == CATEGORY_AUTH_ERROR:
                return False
            if deployment.is_auth_failed and info.status_code in (429, 400, 404):
                return True
            return False

    def revalidate_deployment(
        self,
        deployment_id: str,
        force: bool = False,
        probe_fn: Optional[Callable[[Deployment], bool]] = None,
    ) -> bool:
        """Attempt to revalidate a quarantined deployment."""
        dep = next(
            (d for d in self.deployments if d.id == deployment_id or d.model == deployment_id),
            None,
        )
        if not dep:
            return False

        if not dep.is_quarantined:
            return True

        now = time.time()
        if not force and dep.last_auth_probe_at > 0:
            elapsed = now - dep.last_auth_probe_at
            if elapsed < dep.auth_probe_backoff:
                return False

        dep.last_auth_probe_at = now

        probe = probe_fn if probe_fn is not None else self._probe_deployment_credentials
        success = probe(dep)

        if success:
            self.lifecycle.recover(dep, "Credentials revalidated; returned to service")
            self._sync_router_model_pool()
            return True
        else:
            with dep._state_lock:
                dep.auth_probe_backoff = min(3600.0, dep.auth_probe_backoff * 2.0)
                dep.state_reason = "Revalidation probe failed: credentials invalid"
            self.lifecycle.persist([dep])
            return False

    def revalidate_account(
        self,
        provider: str,
        account: str,
        force: bool = False,
        probe_fn: Optional[Callable[[Deployment], bool]] = None,
    ) -> Dict[str, bool]:
        """Attempt to revalidate all deployments belonging to an account."""
        account_deps = [
            d for d in self.deployments if d.provider == provider and d.account == account
        ]
        if not account_deps:
            return {}

        probe_target = next((d for d in account_deps if d.is_auth_failed),
                            next((d for d in account_deps if d.is_quarantined), account_deps[0]))
        success = self.revalidate_deployment(probe_target.id, force=force, probe_fn=probe_fn)

        results: Dict[str, bool] = {}
        for dep in account_deps:
            if success and (dep.is_auth_failed or dep is probe_target):
                self.lifecycle.recover(dep, "Account credentials revalidated; returned to service")
                results[dep.id] = True
            elif success and not dep.is_quarantined:
                results[dep.id] = True
            else:
                results[dep.id] = False

        if success:
            self._sync_router_model_pool()
            try:
                self.rediscover_models(provider=provider, account=account, force=True)
            except Exception:
                pass

        return results

    def rediscover_models(
        self,
        provider: Optional[str] = None,
        account: Optional[str] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Retry discovery through existing application lifecycle for stale or uninitialized accounts."""
        results: Dict[str, Any] = {"success": {}, "failed": {}}

        # 1. Groq rediscovery
        if provider is None or provider == MODEL_GROUP_GROQ:
            groq_accounts = {
                dep.account: dep.api_key
                for dep in self.deployments
                if dep.provider == MODEL_GROUP_GROQ and dep.api_key
            }
            for acc, key in GROQ_KEYS.items():
                if key and acc not in groq_accounts:
                    groq_accounts[acc] = key

            for acc_name, key in groq_accounts.items():
                if account and acc_name != account:
                    continue

                cache_key = f"groq_{acc_name}"
                is_stale = self.state_manager.is_discovery_stale(cache_key)
                if not force and not is_stale:
                    continue

                try:
                    discovered = discover_groq_models(
                        key, account_name=acc_name, state_manager=self.state_manager
                    )
                    if discovered and not self.state_manager.is_discovery_stale(cache_key):
                        for dep in self.deployments:
                            if dep.provider == MODEL_GROUP_GROQ and dep.account == acc_name:
                                dep.is_stale = False
                                dep.stale_reason = None
                        results["success"][f"{MODEL_GROUP_GROQ}_{acc_name}"] = len(discovered)
                    else:
                        results["failed"][f"{MODEL_GROUP_GROQ}_{acc_name}"] = (
                            "Discovery failed or remained stale"
                        )
                except Exception as err:
                    results["failed"][f"{MODEL_GROUP_GROQ}_{acc_name}"] = sanitize_secret(
                        str(err), key
                    )

        # 2. Gemini rediscovery
        if provider is None or provider == MODEL_GROUP_GEMINI:
            gemini_accounts = {
                dep.account: dep.api_key
                for dep in self.deployments
                if dep.provider == MODEL_GROUP_GEMINI and dep.api_key
            }
            for acc, key in GEMINI_KEYS.items():
                if key and key != "YOUR_NEW_GOOGLE_API_KEY" and acc not in gemini_accounts:
                    gemini_accounts[acc] = key

            for acc_name, key in gemini_accounts.items():
                if account and acc_name != account:
                    continue

                cache_key = f"gemini_{acc_name}"
                is_stale = self.state_manager.is_discovery_stale(cache_key)
                if not force and not is_stale:
                    continue

                try:
                    discovered = discover_gemini_models(
                        key, account_name=acc_name, state_manager=self.state_manager
                    )
                    if discovered and not self.state_manager.is_discovery_stale(cache_key):
                        for dep in self.deployments:
                            if dep.provider == MODEL_GROUP_GEMINI and dep.account == acc_name:
                                dep.is_stale = False
                                dep.stale_reason = None
                        results["success"][f"{MODEL_GROUP_GEMINI}_{acc_name}"] = len(discovered)
                    else:
                        results["failed"][f"{MODEL_GROUP_GEMINI}_{acc_name}"] = (
                            "Discovery failed or remained stale"
                        )
                except Exception as err:
                    results["failed"][f"{MODEL_GROUP_GEMINI}_{acc_name}"] = sanitize_secret(
                        str(err), key
                    )

        self._sync_router_model_pool()
        return results

    def reset_auth_quarantine(
        self,
        provider: Optional[str] = None,
        account: Optional[str] = None,
    ) -> int:
        """Administratively reset authentication quarantine for specified provider/account or all deployments."""
        reset_count = 0
        for dep in self.deployments:
            if provider and dep.provider != provider:
                continue
            if account and dep.account != account:
                continue
            if dep.is_quarantined:
                self.lifecycle.recover(dep, "Quarantine reset administratively")
                reset_count += 1
        if reset_count > 0:
            self._sync_router_model_pool()
        return reset_count

    def _propagate_account_quota_exhaustion(
        self,
        provider: str,
        account: str,
        reset_at: float,
        reason: str,
        api_key: Optional[str] = None,
        quota_scope: str = QuotaScope.ACCOUNT,
    ) -> None:
        """Propagate an account-wide or provider-wide quota exhaustion across all affected deployments."""
        affected = [d for d in self.deployments if d.provider == provider
                    and (quota_scope == QuotaScope.PROVIDER or d.account == account
                         or api_key and d.api_key == api_key)]
        self.lifecycle.cooldown(affected, reset_at, reason, quota_scope)
        self._sync_router_model_pool()

    def _propagate_account_cooldown(
        self,
        provider: str,
        account: str,
        cooldown_until: float,
        reason: str,
    ) -> None:
        """Backward-compatibility alias for _propagate_account_quota_exhaustion."""
        self._propagate_account_quota_exhaustion(provider, account, cooldown_until, reason)

    def get_global_pool(self, now: Optional[float] = None) -> List[Deployment]:
        """Return all active, eligible deployments across all configured providers."""
        ts = now if now is not None else time.time()
        pool: List[Deployment] = []
        for dep in self.deployments:
            dep.refresh(ts)
            if not dep.api_key or not dep.api_key.strip():
                continue
            if dep.is_quarantined:
                continue
            if not dep.is_available(ts):
                continue
            pool.append(dep)
        return pool

    def get_eligible_deployments(
        self,
        model_group: str = DEFAULT_LOGICAL_MODEL,
        capabilities: Optional[ModelCapabilities | dict | list[str] | str] = None,
        min_context: Optional[int] = None,
        now: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Deployment]:
        """Public helper to inspect eligible deployments for any model group or requirements."""
        reqs = extract_request_requirements(
            model=model_group,
            capabilities=capabilities,
            min_context=min_context,
            **kwargs,
        )
        return self._filter_eligible_deployments(
            model_group=model_group,
            requirements=reqs,
            now=now,
        )

    @_routing_locked
    def _filter_eligible_deployments(
        self, model_group: str, requirements: Optional[RequestRequirements] = None,
        capabilities: Optional[ModelCapabilities | dict | list[str] | str] = None,
        min_context: Optional[int] = None, now: Optional[float] = None,
        claim_probe: bool = False, claimed_probes: Optional[List[Deployment]] = None,
        request_id: Optional[str] = None,
    ) -> List[Deployment]:
        """Inspect eligibility. Probe slots are acquired only immediately before dispatch."""
        ts = time.time() if now is None else now
        reqs = requirements or extract_request_requirements(model=model_group,
            capabilities=capabilities, min_context=min_context)
        result = []
        for dep in self.deployments:
            dep.refresh(ts)
            if matches(dep, model_group) and compatible(dep, reqs) and dep.is_available(ts, request_id):
                result.append(dep)
        return result

    def _rank_eligible_deployments(
        self, eligible: List[Deployment], primary_group: Optional[str] = None,
        requirements: Optional[RequestRequirements] = None,
    ) -> List[Deployment]:
        with self.lifecycle.lock:
            return rank(eligible, requirements or RequestRequirements(),
                        self.lifecycle.account_attempts, self.lifecycle.inflight)

    def _order_tier_deployments_account_first(
        self, deployments: List[Deployment], exclude_id: Optional[str] = None,
        deprioritize_account: Optional[str] = None,
    ) -> List[Deployment]:
        """Deterministic account interleaving retained for compatibility."""
        ordered = self._rank_eligible_deployments([d for d in deployments if d.id != exclude_id])
        result, seen = [], set()
        while ordered:
            round_deps = []
            for dep in ordered:
                key = (dep.provider, dep.account)
                if key not in seen:
                    round_deps.append(dep)
                    seen.add(key)
            round_deps.sort(key=lambda d: d.account == deprioritize_account)
            result.extend(round_deps)
            chosen = {d.id for d in round_deps}
            ordered = [d for d in ordered if d.id not in chosen]
            seen.clear()
        return result

    def _validate_user_fallbacks(
        self, user_fallbacks: Any, requirements: RequestRequirements,
        now: Optional[float] = None, claim_probe: bool = False,
        claimed_probes: Optional[List[Deployment]] = None,
        request_id: Optional[str] = None,
    ) -> List[Dict[str, List[str]]]:
        if not isinstance(user_fallbacks, list):
            raise ValueError("fallbacks must be a list of source-to-destination mappings")
        validated = []
        for item in user_fallbacks:
            if not isinstance(item, dict):
                raise ValueError("Each fallback mapping must be a dictionary")
            for source, destinations in item.items():
                if not isinstance(source, str) or not isinstance(destinations, list) or not all(isinstance(d, str) for d in destinations):
                    raise ValueError("Fallback sources and destinations must be model names")
                ids = []
                for group in destinations:
                    candidates = self._rank_eligible_deployments(self._filter_eligible_deployments(
                        group, requirements, now=now, request_id=request_id), requirements=requirements)
                    ids.extend(d.id for d in candidates if d.id not in ids)
                validated.append({source: ids})
        return validated

    def _request_tiers(self, target_model: str, requirements: RequestRequirements,
                       caller_fallbacks: Any = None, request_id: Optional[str] = None,
                       now: Optional[float] = None) -> List[List[Deployment]]:
        """Preserve requested model/provider preference, then configured provider order."""
        tiers = [self._filter_eligible_deployments(target_model, requirements, now=now, request_id=request_id)]
        matched = sorted([d for d in self.deployments if matches(d, target_model)], key=lambda d: d.id)
        if caller_fallbacks is not None:
            mappings = self._validate_user_fallbacks(caller_fallbacks, requirements, now=now, request_id=request_id)
            # Resolve only edges reachable from this request. Cycles cannot duplicate attempts.
            sources = {target_model}
            sources.update(d.id for d in tiers[0])
            visited = set()
            while sources - visited:
                frontier = sources - visited
                visited.update(frontier)
                destinations = []
                for mapping in mappings:
                    for source, ids in mapping.items():
                        if source in frontier or any(matches(d, source) for d in matched if d.id in frontier):
                            destinations.extend(ids)
                if not destinations:
                    continue
                tier = [self.deployment_map[i] for i in dict.fromkeys(destinations)]
                tiers.append(tier)
                matched.extend(tier)
                sources.update(destinations)
        elif target_model not in GLOBAL_GROUPS and target_model not in CAPABILITY_GROUPS:
            providers = list(dict.fromkeys(d.provider for d in matched))
            if target_model in self.provider_fallbacks and not providers:
                providers = [target_model]
            for provider in providers:
                if provider in self.provider_fallbacks:
                    tiers.append(self._filter_eligible_deployments(provider, requirements, now=now, request_id=request_id))
                for fallback in self.provider_fallbacks.get(provider, []):
                    tiers.append(self._filter_eligible_deployments(fallback, requirements, now=now, request_id=request_id))
        seen = set()
        result = []
        for tier in tiers:
            unique = [d for d in tier if d.id not in seen]
            seen.update(d.id for d in unique)
            if unique:
                result.append(unique)
        return result

    def _build_request_pool_and_fallbacks(
        self, target_model: str, requirements: RequestRequirements,
        caller_fallbacks: Any = None, now: Optional[float] = None,
        claim_probe: bool = False, claimed_probes: Optional[List[Deployment]] = None,
        request_id: Optional[str] = None,
    ) -> tuple[Optional[Deployment], Optional[List[Dict[str, List[str]]]], List[Deployment]]:
        tiers = self._request_tiers(target_model, requirements, caller_fallbacks, request_id, now)
        ordered = [d for tier in tiers for d in self._rank_eligible_deployments(tier, requirements=requirements)]
        if not ordered:
            return None, None, []
        return ordered[0], self._build_compatible_fallbacks(ordered), ordered

    def _build_compatible_fallbacks(self, ordered_eligible: List[Deployment]) -> Optional[List[Dict[str, List[str]]]]:
        if len(ordered_eligible) < 2:
            return None
        return [{ordered_eligible[0].id: [d.id for d in ordered_eligible[1:]]}]

    @staticmethod
    def _parse_capabilities(
        caps: Optional[ModelCapabilities | dict | list[str] | str] = None,
    ) -> Optional[ModelCapabilities]:
        """Normalize capability parameter into a ModelCapabilities object."""
        return parse_capabilities(caps)

    def _get_response_deployment(
        self,
        response: Any,
        eligible: Optional[List[Deployment]] = None,
    ) -> Optional[Deployment]:
        """Extract fulfilling deployment strictly from verified response metadata."""
        if response is None:
            return None

        dep = getattr(response, "_ntg_deployment", None)
        if dep is not None and isinstance(dep, Deployment):
            return dep

        model_info = getattr(response, "model_info", None)
        if isinstance(model_info, dict):
            dep_id = model_info.get("id")
            if dep_id and dep_id in self.deployment_map:
                dep = self.deployment_map[dep_id]
                try:
                    setattr(response, "_ntg_deployment", dep)
                except Exception:
                    pass
                return dep

        hidden = getattr(response, "_hidden_params", {}) or {}
        if isinstance(hidden, dict):
            dep_id = hidden.get("model_id")
            if not dep_id and isinstance(hidden.get("model_info"), dict):
                dep_id = hidden.get("model_info", {}).get("id")
            if dep_id and dep_id in self.deployment_map:
                dep = self.deployment_map[dep_id]
                try:
                    setattr(response, "_ntg_deployment", dep)
                except Exception:
                    pass
                return dep

        return None

    def _record_user_request_metric(
        self,
        active_req: Optional[ActiveRequest],
        fulfilling_dep: Optional[Deployment],
        selected_dep: Optional[Deployment],
    ) -> None:
        """Atomically increment user_requests metric exactly once for this public request."""
        if active_req:
            with active_req.lock:
                if active_req.user_request_recorded:
                    return
                active_req.user_request_recorded = True

            attempted = (
                active_req.attempted_deployments
                if active_req.attempted_deployments
                else getattr(self._request_context, "attempted_deployments", [])
            )
        else:
            attempted = getattr(self._request_context, "attempted_deployments", [])

        attempted = list({dep.id: dep for dep in attempted}.values())
        target_dep = None
        if fulfilling_dep:
            target_dep = fulfilling_dep
        elif selected_dep and selected_dep in attempted and len(attempted) == 1:
            target_dep = selected_dep
        elif len(attempted) == 1:
            target_dep = attempted[0]

        if target_dep:
            target_dep.record_user_request()
            self.state_manager.record_metrics(target_dep.id, target_dep.metrics)
        else:
            self.unknown_metrics.increment_user_requests()
            self.state_manager.record_metrics("unknown", self.unknown_metrics)

    def _retry_allowance(self, error: Exception, effective_retries: int) -> int:
        """RetryPolicy constrains transient repeats; failover visits each candidate once."""
        name = "TimeoutErrorRetries" if isinstance(error, TimeoutError) or "timeout" in type(error).__name__.lower() else "InternalServerErrorRetries"
        limit = getattr(self.retry_policy, name, None)
        return min(effective_retries, limit) if limit is not None else effective_retries

    def _execute_plan(self, messages: List[Dict[str, Any]], tiers: List[List[Deployment]],
                      requirements: RequestRequirements, effective_retries: int,
                      call_kwargs: Dict[str, Any], stream: bool):
        """One attempt loop for completions and streams. Never replay emitted chunks."""
        request_id = uuid.uuid4().hex
        active = self._register_active_request(request_id)
        attempted_ids: Set[str] = set()
        attempted_accounts: Set[tuple[str, str]] = set()
        retry_candidates: Dict[str, int] = {}
        retry_delays: Dict[str, float] = {}
        retries_used = 0
        final_error: Optional[Exception] = None
        fulfilling_dep = None
        emitted = False
        try:
            while True:
                dep = None
                admission = None
                retry_target = None
                with self.lifecycle.lock:
                    for tier in tiers:
                        available = []
                        for candidate in tier:
                            candidate.refresh()
                            if (candidate.id not in attempted_ids
                                and self.deployment_map.get(candidate.id) is candidate
                                and compatible(candidate, requirements) and candidate.is_available(request_id=request_id)):
                                available.append(candidate)
                        ordered = self._rank_eligible_deployments(available, requirements=requirements)
                        ordered.sort(key=lambda d: (d.provider, d.account) in attempted_accounts)
                        for candidate in ordered:
                            admission = self.lifecycle.admit(candidate, request_id)
                            if admission:
                                dep = candidate
                                break
                        if dep:
                            break
                    if dep is None and retries_used < effective_retries:
                        candidates = [self.deployment_map[i] for i, budget in retry_candidates.items()
                                      if budget > 0 and i in self.deployment_map]
                        for candidate in self._rank_eligible_deployments(candidates, requirements=requirements):
                            candidate.refresh()
                            if compatible(candidate, requirements) and candidate.is_available(request_id=request_id):
                                retry_target = candidate
                                break
                if retry_target is not None:
                    # Wait outside the routing lock, then revalidate/admit. Never
                    # reserve quota or a probe slot during retry backoff.
                    time.sleep(max(retry_delays.get(retry_target.id, 0.0), 0.1 * 2 ** retries_used))
                    with self.lifecycle.lock:
                        admission = self.lifecycle.admit(retry_target, request_id)
                        if admission:
                            dep = retry_target
                            retry_candidates[dep.id] -= 1
                            retries_used += 1
                        else:
                            retry_candidates.pop(retry_target.id, None)
                            continue
                if dep is None:
                    if final_error:
                        raise final_error
                    raise NoEligibleDeploymentsError("Routing capacity changed before dispatch")
                attempted_ids.add(dep.id)
                attempted_accounts.add((dep.provider, dep.account))
                active.attempted_deployments.append(dep)
                active.selected_dep = active.selected_dep or dep
                started, key_hash = admission
                response = None
                try:
                    transport_kwargs = dict(call_kwargs)
                    # Routing controls cannot override the selected deployment or trigger
                    # nested retries in LiteLLM. All other generation options pass through.
                    for key in ("api_key", "api_base", "base_url", "custom_llm_provider", "model_list", "model_info", "fallbacks",
                                "context_window_fallbacks", "content_policy_fallbacks", "retry_policy", "max_retries"):
                        transport_kwargs.pop(key, None)
                    metadata = dict(transport_kwargs.pop("metadata", {}) or {})
                    metadata.update(ntg_request_id=request_id, ntg_router_id=self.router_id,
                                    deployment_id=dep.id, ntg_managed=True)
                    response = litellm.completion(model=dep.litellm_model, messages=messages,
                        api_key=dep.api_key, api_base=dep.api_base, metadata=metadata,
                        num_retries=0, **transport_kwargs)
                    if response is None:
                        raise RuntimeError("Provider returned no completion")
                    if stream:
                        for chunk in response:
                            emitted = True
                            yield chunk
                    self.lifecycle.finish(dep, request_id, started, key_hash, response=response)
                    fulfilling_dep = dep
                    active.fulfilling_dep = dep
                    if not stream:
                        try:
                            setattr(response, "_ntg_deployment", dep)
                        except Exception:
                            pass
                        yield response
                    return
                except Exception as error:
                    info = parse_provider_error(error, provider=dep.provider)
                    self.lifecycle.finish(dep, request_id, started, key_hash, error=info)
                    final_error = error
                    if emitted or (not info.is_retryable and not info.should_quarantine):
                        raise
                    if info.category == CATEGORY_SERVER_ERROR:
                        # Long Retry-After hints are respected by declining an
                        # in-request repeat, rather than ignoring them or blocking
                        # an interactive request for minutes. Fresh failover remains.
                        allowance = self._retry_allowance(error, effective_retries) if info.cooldown_seconds <= 2 else 0
                        retry_candidates.setdefault(dep.id, allowance)
                        retry_delays[dep.id] = info.cooldown_seconds
                finally:
                    if stream and response is not None and hasattr(response, "close"):
                        try:
                            response.close()
                        except Exception:
                            pass
                    self.lifecycle.release(dep, request_id)
        finally:
            self._record_user_request_metric(active, fulfilling_dep, active.selected_dep)
            self._unregister_active_request(request_id)

    def _route_request(
        self, messages: list[dict[str, Any]], model: Optional[str] = None,
        capabilities: Optional[ModelCapabilities | dict | list[str] | str] = None,
        min_context: Optional[int] = None, prompt: Optional[str] = None, **kwargs: Any,
    ) -> tuple[Optional[Any], Optional[Deployment], Optional[Exception], RequestRequirements, float]:
        """Compile requirements and a side-effect-free plan, then execute bounded attempts."""
        now = time.time()
        target = model or self.default_model
        if target in GLOBAL_GROUPS:
            target = MODEL_GROUP_AUTO
        requirements = extract_request_requirements(messages=messages, prompt=prompt,
            model=target, capabilities=capabilities, min_context=min_context, **kwargs)
        retry_value = kwargs.pop("num_retries", self.num_retries)
        if retry_value is None:
            retry_value = self.num_retries
        if isinstance(retry_value, bool) or not isinstance(retry_value, int):
            raise ValueError("num_retries must be an integer")
        retries = min(MAX_RETRY_BUDGET, max(0, retry_value))
        caller_fallbacks = kwargs.pop("fallbacks", None)
        if kwargs.get("metadata") is not None and not isinstance(kwargs["metadata"], dict):
            raise ValueError("metadata must be a dictionary")
        if kwargs.pop("streaming", False):
            kwargs["stream"] = True
        if isinstance(kwargs.get("reasoning"), bool):
            kwargs.pop("reasoning")
        for control in ("coding", "task", "streaming", "images"):
            # images is detected but not a standard chat payload; reject instead of dropping it.
            if control == "images" and kwargs.get(control):
                raise ValueError("Put images in message content parts")
            kwargs.pop(control, None)
        with self.lifecycle.lock:
            tiers = self._request_tiers(target, requirements, caller_fallbacks, now=now)
        if not tiers:
            self.unknown_metrics.increment_user_requests()
            self.state_manager.record_metrics("unknown", self.unknown_metrics)
            error = NoEligibleDeploymentsError(f"No eligible deployments for model='{target}' satisfying requirements: {requirements.describe()}.")
            return None, None, error, requirements, now
        execution = self._execute_plan(messages, tiers, requirements, retries, kwargs, bool(kwargs.get("stream")))
        if kwargs.get("stream"):
            return execution, None, None, requirements, now
        try:
            response = next(execution)
            return response, self._get_response_deployment(response), None, requirements, now
        except Exception as error:
            return None, None, error, requirements, now
        finally:
            execution.close()

    def ask(
        self,
        prompt: str,
        model: Optional[str] = None,
        capabilities: Optional[ModelCapabilities | dict | list[str] | str] = None,
        min_context: Optional[int] = None,
        **kwargs: Any,
    ) -> Optional[Any]:
        """Route a user prompt through NTG's unified attempt pipeline."""
        target_model = model or self.default_model
        if target_model in ("ntg-auto", MODEL_GROUP_NTG_AUTO):
            target_model = MODEL_GROUP_AUTO

        messages = [{"role": "user", "content": prompt}]
        response, deployment, error, reqs, now = self._route_request(
            messages=messages,
            model=target_model,
            capabilities=capabilities,
            min_context=min_context,
            prompt=prompt,
            **kwargs,
        )

        if error is not None:
            if isinstance(error, NoEligibleDeploymentsError):
                print_divider("NO ELIGIBLE DEPLOYMENTS")
                print(
                    f"No available deployments satisfy request for model group '{target_model}'"
                )
                print(f"Required capabilities: {reqs.describe()}")
                if min_context:
                    print(f"Required context     : >={min_context:,}")

                resets = [
                    dep.remaining_cooldown
                    for dep in self.deployments
                    if dep.remaining_cooldown > 0
                ]
                if resets:
                    next_capacity = min(resets)
                    print(
                        f"Next available deployment capacity in: {next_capacity:.1f}s ({utc_string(now + next_capacity)})"
                    )
                print_account_status(self.deployments)
                return None
            else:
                print_divider("ALL ELIGIBLE DEPLOYMENTS EXHAUSTED")
                print(
                    f"Request failed across healthy deployments for model group '{target_model}'."
                )
                safe_error = str(error)
                for dep in self.deployments:
                    safe_error = sanitize_secret(safe_error, dep.api_key)
                print(f"Final error: {safe_error}")
                print_account_status(self.deployments)
                return None

        if hasattr(response, "choices") and response.choices:
            print("\n" + str(response.choices[0].message.content))

        if deployment:
            print_request_execution(deployment, getattr(response, "model", None))

        return response

    def completion(
        self,
        messages: list[dict[str, Any]],
        model: Optional[str] = None,
        capabilities: Optional[ModelCapabilities | dict | list[str] | str] = None,
        min_context: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        """Execute chat completion conforming to standard OpenAI/LiteLLM signature."""
        response, _, error, _, _ = self._route_request(
            messages=messages,
            model=model,
            capabilities=capabilities,
            min_context=min_context,
            **kwargs,
        )
        if error is not None:
            raise error
        return response


OpenRouterFallbackRouter = UnifiedNTGRouter
