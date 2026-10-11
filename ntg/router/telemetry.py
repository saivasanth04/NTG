"""LiteLLM custom logger integration for telemetry, circuit breaker updates, and quota intelligence."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import threading
from typing import TYPE_CHECKING, Any, List, Optional, Set
import weakref

from litellm.integrations.custom_logger import CustomLogger

from ntg.core.exceptions import parse_provider_error
from ntg.core.models import Deployment

if TYPE_CHECKING:
    from ntg.router.engine import UnifiedNTGRouter

logger = logging.getLogger("ntg.router.telemetry")


@dataclass
class ActiveRequest:
    """Request-level coordination tracking for concurrent attempts and probe management."""

    request_id: str
    router_id: str
    selected_dep: Optional[Deployment] = None
    fulfilling_dep: Optional[Deployment] = None
    attempted_deployments: List[Deployment] = field(default_factory=list)
    recorded_attempt_ids: Set[str] = field(default_factory=set)
    completed_attempt_ids: Set[str] = field(default_factory=set)
    fallback_list: Optional[List[Any]] = None
    fallback_targets: List[str] = field(default_factory=list)
    active_dispatch_id: Optional[str] = None
    dispatch_counter: int = 0
    callback_events_count: int = 0
    dispatched_at: float = 0.0
    user_request_recorded: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


class NTGTelemetryLogger(CustomLogger):
    """Compatibility observer for explicitly identified, active NTG requests.

    Managed attempts are recorded synchronously by the executor. Async LiteLLM
    callbacks must never record them twice or mutate state after request teardown.
    This logger is intentionally not installed in LiteLLM's global callback lists.
    """

    def __init__(self, router: UnifiedNTGRouter):
        super().__init__()
        self._router_ref = weakref.ref(router)
        self._attempts = {}
        self._lock = threading.RLock()

    @property
    def router(self) -> Optional[UnifiedNTGRouter]:
        return self._router_ref()

    def _get_request_identity(self, kwargs):
        router = self.router
        kw = kwargs if isinstance(kwargs, dict) else {}
        meta = kw.get("metadata") or kw.get("litellm_params", {}).get("metadata", {}) or {}
        if router is None or meta.get("ntg_router_id") != router.router_id or meta.get("ntg_managed"):
            return None, None, None
        request_id = meta.get("ntg_request_id")
        active = router._get_active_request(request_id)
        if active is None:
            return None, None, None
        return request_id, kw.get("litellm_call_id"), active

    def _extract_deployment(self, kwargs=None, response_obj=None):
        kw = kwargs or {}
        meta = kw.get("metadata") or kw.get("litellm_params", {}).get("metadata", {}) or {}
        dep_id = meta.get("deployment_id")
        router = self.router
        return dep_id, router.deployment_map.get(dep_id) if router else None

    def log_pre_api_call(self, model, messages, kwargs=None):
        request_id, call_id, active = self._get_request_identity(kwargs)
        _, dep = self._extract_deployment(kwargs)
        if active is None or not call_id or dep is None:
            return
        with self._lock:
            if call_id in active.recorded_attempt_ids:
                return
            admission = self.router.lifecycle.admit(dep, request_id)
            if admission is None:
                return
            active.recorded_attempt_ids.add(call_id)
            active.attempted_deployments.append(dep)
            self._attempts[call_id] = (dep, admission)

    def _finish(self, kwargs, response, failed):
        request_id, call_id, active = self._get_request_identity(kwargs)
        if active is None or not call_id:
            return
        with self._lock:
            attempt = self._attempts.pop(call_id, None)
            if attempt is None:
                return
            dep, (started, key_hash) = attempt
            error = kwargs.get("exception") or response if failed else None
            if failed and not isinstance(error, Exception):
                error = RuntimeError(str(error))
            info = parse_provider_error(error, provider=dep.provider) if failed else None
            try:
                self.router.lifecycle.finish(dep, request_id, started, key_hash, response, info)
                active.completed_attempt_ids.add(call_id)
                if not failed:
                    active.fulfilling_dep = dep
            finally:
                self.router.lifecycle.release(dep, request_id)

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._finish(kwargs, response_obj, False)

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._finish(kwargs, response_obj, True)

    async def async_log_pre_api_call(self, model, messages, kwargs=None):
        self.log_pre_api_call(model, messages, kwargs)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.log_success_event(kwargs, response_obj, start_time, end_time)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self.log_failure_event(kwargs, response_obj, start_time, end_time)
