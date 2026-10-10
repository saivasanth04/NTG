"""LiteLLM custom logger integration for telemetry, circuit breaker updates, and quota intelligence."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import threading
import time
from typing import TYPE_CHECKING, Any, List, Optional, Set
import weakref

from litellm.integrations.custom_logger import CustomLogger

from ntg.core.exceptions import (
    CATEGORY_AUTH_ERROR,
    CATEGORY_DAILY_QUOTA,
    CATEGORY_PROVIDER_LIMIT,
    CATEGORY_REQUEST_ERROR,
    CATEGORY_UNKNOWN_LIMIT,
    parse_provider_error,
)
from ntg.core.models import CircuitState, Deployment, QuotaScope
from ntg.core.utils import sanitize_secret

if TYPE_CHECKING:
    from ntg.router.engine import UnifiedNTGRouter

logger = logging.getLogger("ntg.router.telemetry")


@dataclass
class ActiveRequest:
    """Request-level coordination tracking for concurrent attempts and probe management."""

    request_id: str
    router_id: str
    selected_dep: Optional[Deployment] = None
    attempted_deployments: List[Deployment] = field(default_factory=list)
    recorded_attempt_ids: Set[str] = field(default_factory=set)
    active_dispatch_id: Optional[str] = None
    dispatch_counter: int = 0
    dispatched_at: float = 0.0
    user_request_recorded: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


class NTGTelemetryLogger(CustomLogger):
    """LiteLLM custom logger callback for telemetry, circuit breaker updates, and quota intelligence.

    Extracts per-attempt deployment metadata directly from event kwargs to ensure
    100% thread safety and eliminate shared mutable request state.
    """

    def __init__(self, router: UnifiedNTGRouter):
        super().__init__()
        self._router_ref = weakref.ref(router)

    @property
    def router(self) -> Optional[UnifiedNTGRouter]:
        return self._router_ref()

    def _get_request_identity(
        self, kwargs: Optional[dict]
    ) -> tuple[Optional[str], Optional[str], Optional[ActiveRequest]]:
        """Extract request ID, attempt ID, and ActiveRequest from event kwargs.

        Guarantees router isolation: if metadata contains ntg_router_id and it does not match
        this router instance, returns (None, None, None) immediately.
        """
        router = self.router
        if router is None:
            return None, None, None

        kw = kwargs or {}
        req_router_id = (
            kw.get("litellm_params", {}).get("metadata", {}).get("ntg_router_id")
            or kw.get("metadata", {}).get("ntg_router_id")
        )
        if req_router_id and req_router_id != router.router_id:
            return None, None, None

        req_id = (
            kw.get("litellm_params", {}).get("metadata", {}).get("ntg_request_id")
            or kw.get("metadata", {}).get("ntg_request_id")
        )
        if not req_id:
            req_id = getattr(router._request_context, "active_request_id", None)

        active_req = router._get_active_request(req_id) if req_id else None
        call_id = kw.get("litellm_call_id")
        return req_id, call_id, active_req

    def _extract_deployment(
        self, kwargs: Optional[dict] = None, response_obj: Any = None
    ) -> tuple[Optional[str], Optional[Deployment]]:
        """Extract deployment ID and Deployment instance from event kwargs or response object."""
        router = self.router
        if router is None or (not kwargs and response_obj is None):
            return None, None
        kw = kwargs or {}

        cand_ids: List[str] = []
        cand_ids.append(kw.get("litellm_params", {}).get("metadata", {}).get("deployment_id"))
        cand_ids.append(kw.get("metadata", {}).get("deployment_id"))
        cand_ids.append(kw.get("litellm_params", {}).get("model_info", {}).get("id"))
        cand_ids.append(kw.get("model_info", {}).get("id"))
        cand_ids.append(kw.get("litellm_params", {}).get("metadata", {}).get("model_info", {}).get("id"))
        cand_ids.append(kw.get("litellm_params", {}).get("metadata", {}).get("deployment_model_name"))
        cand_ids.append(kw.get("litellm_params", {}).get("metadata", {}).get("model_group"))
        cand_ids.append(kw.get("litellm_params", {}).get("model"))
        cand_ids.append(kw.get("model"))

        if response_obj is not None:
            hidden = getattr(response_obj, "_hidden_params", {}) or {}
            if isinstance(hidden, dict):
                cand_ids.append(hidden.get("deployment_id"))
                cand_ids.append(hidden.get("model_id"))
                if isinstance(hidden.get("model_info"), dict):
                    cand_ids.append(hidden.get("model_info", {}).get("id"))
                    cand_ids.append(hidden.get("model_info", {}).get("deployment_id"))

        exc = kw.get("exception")
        if not exc and isinstance(response_obj, Exception):
            exc = response_obj
        if exc and hasattr(exc, "model_id"):
            cand_ids.append(getattr(exc, "model_id"))

        for cand in cand_ids:
            if cand and cand in router.deployment_map:
                return cand, router.deployment_map[cand]

        if hasattr(router, "router") and hasattr(router.router, "model_list"):
            for cand in cand_ids:
                if not cand:
                    continue
                for entry in router.router.model_list:
                    if entry.get("model_info", {}).get("id") == cand:
                        dep_id = (
                            entry.get("litellm_params", {}).get("metadata", {}).get("deployment_id")
                            or entry.get("litellm_params", {}).get("model_info", {}).get("id")
                        )
                        if dep_id and dep_id in router.deployment_map:
                            return dep_id, router.deployment_map[dep_id]

        return None, None

    def log_pre_api_call(self, model: Any, messages: Any, kwargs: Any = None) -> None:
        """LiteLLM pre-API call callback: records attempt dispatch per actual upstream call."""
        router = self.router
        if router is None:
            return

        try:
            kw = kwargs if isinstance(kwargs, dict) else {}
            req_id, call_id, active_req = self._get_request_identity(kw)
            if not req_id and not call_id:
                return

            _, deployment = self._extract_deployment(kw)

            if active_req:
                with active_req.lock:
                    if call_id:
                        attempt_id = call_id
                    else:
                        active_req.dispatch_counter += 1
                        attempt_id = f"{active_req.request_id}_dispatch_{active_req.dispatch_counter}"
                        active_req.active_dispatch_id = attempt_id
                        if isinstance(kw, dict):
                            kw["ntg_attempt_id"] = attempt_id

                    if attempt_id in active_req.recorded_attempt_ids:
                        return
                    active_req.recorded_attempt_ids.add(attempt_id)
                    active_req.dispatched_at = time.time()
                    if deployment:
                        active_req.attempted_deployments.append(deployment)
            else:
                recorded = getattr(router._request_context, "recorded_call_ids", None)
                if recorded is not None and isinstance(recorded, set):
                    if call_id and call_id in recorded:
                        return
                    if call_id:
                        recorded.add(call_id)
                if deployment:
                    attempts = getattr(router._request_context, "attempted_deployments", None)
                    if attempts is not None and isinstance(attempts, list):
                        attempts.append(deployment)

            if deployment:
                deployment.record_attempt()
                router.state_manager.record_metrics(deployment.id, deployment.metrics)
            else:
                router.unknown_metrics.record_attempt()
                router.state_manager.record_metrics("unknown", router.unknown_metrics)
        except Exception as e:
            logger.warning("Error in NTGTelemetryLogger.log_pre_api_call: %s", e)

    def log_success_event(self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any) -> None:
        """Handle upstream success callback."""
        router = self.router
        if router is None:
            return

        try:
            kw = kwargs if isinstance(kwargs, dict) else {}
            req_id, call_id, active_req = self._get_request_identity(kw)
            if not req_id and not call_id:
                return

            _, deployment = self._extract_deployment(kw, response_obj)

            if active_req:
                with active_req.lock:
                    if call_id:
                        attempt_id = call_id
                    elif kw.get("ntg_attempt_id"):
                        attempt_id = kw["ntg_attempt_id"]
                    elif active_req.active_dispatch_id:
                        attempt_id = active_req.active_dispatch_id
                        active_req.active_dispatch_id = None
                    else:
                        active_req.dispatch_counter += 1
                        attempt_id = f"{active_req.request_id}_dispatch_{active_req.dispatch_counter}"

                    if attempt_id not in active_req.recorded_attempt_ids:
                        active_req.recorded_attempt_ids.add(attempt_id)
                        if deployment:
                            active_req.attempted_deployments.append(deployment)
                            deployment.record_attempt()
                            router.state_manager.record_metrics(deployment.id, deployment.metrics)
                        else:
                            router.unknown_metrics.record_attempt()
                            router.state_manager.record_metrics("unknown", router.unknown_metrics)
            else:
                recorded = getattr(router._request_context, "recorded_call_ids", None)
                if recorded is not None and isinstance(recorded, set):
                    if call_id and call_id in recorded:
                        pass
                    elif call_id:
                        recorded.add(call_id)
                        if deployment:
                            router._record_request_attempt(deployment)
                        else:
                            router.unknown_metrics.record_attempt()
                elif not call_id:
                    if deployment:
                        router._record_request_attempt(deployment)
                    else:
                        router.unknown_metrics.record_attempt()

            if deployment:
                deployment.metrics.record_success()
                deployment.record_success(request_id=req_id)

                headers = None
                if response_obj is not None:
                    headers = getattr(response_obj, "_response_headers", None) or getattr(
                        response_obj, "headers", None
                    )
                if not headers and isinstance(kw, dict):
                    headers = kw.get("response_headers")

                if headers:
                    deployment.quota.update_from_headers(
                        headers,
                        provider=deployment.provider,
                        model=deployment.model,
                    )
                    router.state_manager.record_quota(deployment.id, deployment.quota)

                router.state_manager.record_circuit_state(deployment.id, CircuitState.HEALTHY)
                router.state_manager.record_metrics(deployment.id, deployment.metrics)

                if response_obj is not None:
                    try:
                        setattr(response_obj, "_ntg_deployment", deployment)
                    except Exception:
                        pass
            else:
                router.unknown_metrics.record_success()
                router.state_manager.record_metrics("unknown", router.unknown_metrics)
        except Exception as e:
            logger.warning("Error in NTGTelemetryLogger.log_success_event: %s", e)

    def log_failure_event(self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any) -> None:
        """Handle upstream failure callback with granular error classification."""
        router = self.router
        if router is None:
            return

        try:
            kw = kwargs if isinstance(kwargs, dict) else {}
            req_id, call_id, active_req = self._get_request_identity(kw)
            if not req_id and not call_id:
                return

            _, deployment = self._extract_deployment(kw, response_obj)
            exc = kw.get("exception") if isinstance(kw, dict) else None
            if not exc and isinstance(response_obj, Exception):
                exc = response_obj

            if active_req:
                with active_req.lock:
                    if call_id:
                        attempt_id = call_id
                    elif kw.get("ntg_attempt_id"):
                        attempt_id = kw["ntg_attempt_id"]
                    elif active_req.active_dispatch_id:
                        attempt_id = active_req.active_dispatch_id
                        active_req.active_dispatch_id = None
                    else:
                        active_req.dispatch_counter += 1
                        attempt_id = f"{active_req.request_id}_dispatch_{active_req.dispatch_counter}"

                    if attempt_id not in active_req.recorded_attempt_ids:
                        active_req.recorded_attempt_ids.add(attempt_id)
                        if deployment:
                            active_req.attempted_deployments.append(deployment)
                            deployment.record_attempt()
                            router.state_manager.record_metrics(deployment.id, deployment.metrics)
                        else:
                            router.unknown_metrics.record_attempt()
                            router.state_manager.record_metrics("unknown", router.unknown_metrics)
            else:
                recorded = getattr(router._request_context, "recorded_call_ids", None)
                if recorded is not None and isinstance(recorded, set):
                    if call_id and call_id in recorded:
                        pass
                    elif call_id:
                        recorded.add(call_id)
                        if deployment:
                            router._record_request_attempt(deployment)
                        else:
                            router.unknown_metrics.record_attempt()
                elif not call_id:
                    if deployment:
                        router._record_request_attempt(deployment)
                    else:
                        router.unknown_metrics.record_attempt()

            if not deployment:
                router.unknown_metrics.record_failure()
                router.state_manager.record_metrics("unknown", router.unknown_metrics)
                return

            dispatched_at = 0.0
            if active_req and active_req.dispatched_at > 0:
                dispatched_at = active_req.dispatched_at
            elif isinstance(start_time, (int, float)) and start_time > 0:
                dispatched_at = float(start_time)
            elif hasattr(start_time, "timestamp"):
                try:
                    dispatched_at = start_time.timestamp()
                except Exception:
                    pass
            elif isinstance(kw.get("start_time"), (int, float)):
                dispatched_at = float(kw["start_time"])

            is_late_callback = bool(
                dispatched_at > 0
                and deployment.state_updated_at > dispatched_at
                and deployment.circuit_state == CircuitState.HEALTHY
            )

            if exc:
                info = parse_provider_error(exc, provider=deployment.provider)

                deployment.update_quota(
                    rpm_limit=info.rpm_limit,
                    rpm_remaining=info.rpm_remaining,
                    rpd_limit=info.rpd_limit,
                    rpd_remaining=info.rpd_remaining,
                    reset_at=info.reset_timestamp,
                    retry_after=info.cooldown_seconds,
                    quota_scope=info.quota_scope,
                    limit_type=info.limit_type,
                )
                router.state_manager.record_quota(deployment.id, deployment.quota)

                if info.category in (
                    CATEGORY_PROVIDER_LIMIT,
                    CATEGORY_DAILY_QUOTA,
                    CATEGORY_UNKNOWN_LIMIT,
                ):
                    deployment.metrics.record_rate_limit()
                    deployment.metrics.record_failure()
                elif info.category == CATEGORY_AUTH_ERROR:
                    deployment.metrics.record_auth_failure()
                    deployment.metrics.record_failure()
                else:
                    deployment.metrics.record_failure()

                if is_late_callback:
                    router.state_manager.record_metrics(deployment.id, deployment.metrics)
                    return

                if info.category == CATEGORY_AUTH_ERROR:
                    safe_reason = sanitize_secret(
                        f"Authentication failure: {info.message}", deployment.api_key
                    )
                    router._propagate_account_auth_quarantine(
                        provider=deployment.provider,
                        account=deployment.account,
                        reason=safe_reason,
                        failed_key=deployment.api_key,
                    )

                elif info.should_quarantine:
                    safe_reason = sanitize_secret(
                        f"Model unavailable: {info.message}", deployment.api_key
                    )
                    deployment.transition_to_quarantine(
                        reason=safe_reason, state=CircuitState.QUARANTINED
                    )
                    router.state_manager.record_quarantine(
                        deployment.id, deployment.state_reason, state=CircuitState.QUARANTINED
                    )
                    router.state_manager.record_circuit_state(
                        deployment.id, CircuitState.QUARANTINED, reason=deployment.state_reason
                    )
                    router._sync_router_model_pool()

                elif info.status_code == 429 or info.category in (
                    CATEGORY_DAILY_QUOTA,
                    CATEGORY_PROVIDER_LIMIT,
                    CATEGORY_UNKNOWN_LIMIT,
                ):
                    rate_reason = (
                        f"Daily quota exhausted: {info.message}"
                        if info.category == CATEGORY_DAILY_QUOTA
                        else f"Rate limit reached ({info.limit_type}): {info.message}"
                    )
                    deployment.transition_to_open(
                        cooldown_seconds=info.cooldown_seconds,
                        reason=rate_reason,
                        reset_at=info.reset_timestamp,
                        request_id=req_id,
                    )
                    router.state_manager.record_cooldown(
                        deployment.id,
                        cooldown_seconds_or_until=deployment.quota.reset_at,
                        reason_or_limit_type=info.limit_type or "429",
                        reset_timestamp=deployment.quota.reset_at,
                        quota_scope=info.quota_scope,
                    )
                    router.state_manager.record_circuit_state(
                        deployment.id, CircuitState.OPEN, reason=deployment.state_reason
                    )

                    if info.quota_scope in (QuotaScope.ACCOUNT, QuotaScope.PROVIDER):
                        router._propagate_account_quota_exhaustion(
                            provider=deployment.provider,
                            account=deployment.account,
                            reset_at=deployment.quota.reset_at,
                            reason=info.message,
                            api_key=deployment.api_key,
                            quota_scope=info.quota_scope,
                        )
                    else:
                        router._sync_router_model_pool()

                elif info.category == CATEGORY_REQUEST_ERROR:
                    pass

                else:
                    deployment.record_failure(request_id=req_id)
                    if deployment.circuit_state == CircuitState.OPEN:
                        router.state_manager.record_circuit_state(
                            deployment.id,
                            CircuitState.OPEN,
                            reason=deployment.state_reason or "Qualifying upstream failure",
                            recovery_time=deployment.circuit_open_until,
                        )
                        router._sync_router_model_pool()
            else:
                deployment.metrics.record_failure()
                if not is_late_callback:
                    deployment.record_failure(request_id=req_id)
                    if deployment.circuit_state == CircuitState.OPEN:
                        router.state_manager.record_circuit_state(
                            deployment.id,
                            CircuitState.OPEN,
                            reason=deployment.state_reason or "Qualifying upstream failure",
                            recovery_time=deployment.circuit_open_until,
                        )
                        router._sync_router_model_pool()

            router.state_manager.record_metrics(deployment.id, deployment.metrics)
        except Exception as e:
            logger.warning("Error in NTGTelemetryLogger.log_failure_event: %s", e)

    async def async_log_pre_api_call(self, model: Any, messages: Any, kwargs: Any = None) -> None:
        """Async LiteLLM pre-API call callback: delegates to log_pre_api_call."""
        self.log_pre_api_call(model, messages, kwargs)

    async def async_log_success_event(
        self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        """Async LiteLLM success callback: delegates to log_success_event."""
        self.log_success_event(kwargs, response_obj, start_time, end_time)

    async def async_log_failure_event(
        self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        """Async LiteLLM failure callback: delegates to log_failure_event."""
        self.log_failure_event(kwargs, response_obj, start_time, end_time)
