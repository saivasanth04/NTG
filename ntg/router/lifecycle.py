"""The single owner of admission, scoped quota reservations and attempt outcomes."""

from __future__ import annotations

import hashlib
import threading
import time
from typing import Any, Sequence

from ntg.core.exceptions import (CATEGORY_AUTH_ERROR, CATEGORY_DAILY_QUOTA,
    CATEGORY_PROVIDER_LIMIT, CATEGORY_UNKNOWN_LIMIT, CATEGORY_SERVER_ERROR, ParsedErrorInfo)
from ntg.core.models import CircuitState, Deployment
from ntg.core.utils import sanitize_secret
from ntg.router.policy import quota_peers

RATE_LIMITS = frozenset((CATEGORY_DAILY_QUOTA, CATEGORY_PROVIDER_LIMIT, CATEGORY_UNKNOWN_LIMIT))


class RoutingLifecycle:
    def __init__(self, router: Any):
        self.router = router
        self.lock = threading.RLock()
        self.account_attempts: dict[tuple[str, str], int] = {}
        self.inflight: dict[tuple[str, str], int] = {}
        self._started_monotonic: dict[tuple[str, str], float] = {}

    def quarantine_account(self, provider: str, account: str, reason: str,
                           failed_key: str | None = None) -> list[Deployment]:
        with self.lock:
            affected = [d for d in self.router.deployments if d.provider == provider
                        and (d.account == account or failed_key and d.api_key == failed_key)]
            for peer in affected:
                peer.transition_to_quarantine(reason, CircuitState.AUTH_FAILED, failed_key=failed_key)
            records = {}
            for peer in affected:
                records[f"{peer.provider}:{peer.account}"] = {
                    "auth_fingerprint": peer.auth_failed_key_hash,
                    "state_reason": peer.state_reason, "state_updated_at": peer.state_updated_at,
                }
            self.router.state_manager.snapshot_deployments(affected, records)
            return affected

    def cooldown(self, deployments: Sequence[Deployment], reset_at: float,
                 reason: str, quota_scope: str) -> None:
        with self.lock:
            for deployment in deployments:
                if deployment.is_quarantined:
                    continue
                deployment.quota.quota_scope = quota_scope
                deadline = max(reset_at, deployment.quota.reset_at or 0.0,
                               deployment.circuit_open_until)
                deployment.transition_to_open(max(0.0, deadline - time.time()),
                                             reason, reset_at=deadline)
            self.persist(deployments)

    def recover(self, deployment: Deployment, reason: str) -> None:
        """Administrative/auth recovery cannot forgive independently exhausted quota."""
        with self.lock:
            was_auth_failed = deployment.is_auth_failed
            reset_at, retry_after = deployment.quota.reset_at, deployment.quota.retry_after
            deployment.transition_to_healthy(reason)
            deployment.quota.reset_at, deployment.quota.retry_after = reset_at, retry_after
            records = {f"{deployment.provider}:{deployment.account}": {}} if was_auth_failed else None
            self.router.state_manager.snapshot_deployments([deployment], records)

    def admit(self, deployment: Deployment, request_id: str) -> tuple[float, str] | None:
        """Reserve only the chosen deployment, atomically with account distribution."""
        with self.lock:
            now = time.time()
            account_auth = self.router.state_manager.get_account_auth(deployment.provider, deployment.account)
            fingerprint = hashlib.sha256(deployment.api_key.encode()).hexdigest()
            if account_auth and account_auth.get("auth_fingerprint") == fingerprint:
                deployment.transition_to_quarantine(account_auth.get("state_reason", "Account authentication failed"),
                    CircuitState.AUTH_FAILED)
                return None
            peers = quota_peers(deployment, self.router.deployments)
            for peer in peers:
                peer.refresh(now)
                if peer.quota.exhausted(now):
                    return None
            if not deployment.admit_to_request_pool(now, request_id=request_id):
                return None
            for peer in peers:
                for prefix in ("rpm", "rpd"):
                    value = getattr(peer.quota, f"{prefix}_remaining")
                    if value is not None:
                        setattr(peer.quota, f"{prefix}_remaining", max(0, value - 1))
            key = (deployment.provider, deployment.account)
            self.account_attempts[key] = self.account_attempts.get(key, 0) + 1
            self.inflight[key] = self.inflight.get(key, 0) + 1
            deployment.record_attempt()
            self._started_monotonic[(deployment.id, request_id)] = time.monotonic()
            self.router.state_manager.snapshot_deployments(peers)
            return now, hashlib.sha256(deployment.api_key.encode()).hexdigest()

    def release(self, deployment: Deployment, request_id: str) -> None:
        with self.lock:
            key = (deployment.provider, deployment.account)
            self.inflight[key] = max(0, self.inflight.get(key, 0) - 1)
            deployment.release_half_open_probe(request_id=request_id)
            self._started_monotonic.pop((deployment.id, request_id), None)

    @staticmethod
    def _update_headers(deployment: Deployment, headers: Any) -> None:
        """A response cannot refund reservations belonging to concurrent attempts."""
        deployment.quota.refresh(time.time())
        prior = {name: getattr(deployment.quota, name) for name in ("rpm_remaining", "rpd_remaining")}
        deployment.quota.update_from_headers(headers, provider=deployment.provider, model=deployment.model)
        for name, previous in prior.items():
            reported = getattr(deployment.quota, name)
            if previous is not None and reported is not None:
                setattr(deployment.quota, name, min(previous, reported))

    def finish(self, deployment: Deployment, request_id: str, started: float,
               key_hash: str, response: Any = None, error: ParsedErrorInfo | None = None) -> None:
        with self.lock:
            current_hash = hashlib.sha256(deployment.api_key.encode()).hexdigest()
            stale = current_hash != key_hash or (deployment.state_updated_at > started
                and deployment.circuit_state in (CircuitState.OPEN, CircuitState.QUARANTINED, CircuitState.AUTH_FAILED))
            affected = [deployment]
            if error is None:
                deployment.metrics.record_success()
                elapsed = max(0.0, time.monotonic() - self._started_monotonic.get(
                    (deployment.id, request_id), time.monotonic()))
                deployment.latency_ewma = elapsed if deployment.latency_ewma is None else 0.2 * elapsed + 0.8 * deployment.latency_ewma
                if not stale:
                    # Success restores health but cannot erase a live quota window.
                    quota_reset, retry_after = deployment.quota.reset_at, deployment.quota.retry_after
                    deployment.record_success(request_id=request_id)
                    deployment.quota.reset_at, deployment.quota.retry_after = quota_reset, retry_after
                    hidden = getattr(response, "_hidden_params", {})
                    headers = (getattr(response, "_response_headers", None)
                               or getattr(response, "headers", None)
                               or (hidden.get("additional_headers") if isinstance(hidden, dict) else None))
                    if headers:
                        self._update_headers(deployment, headers)
                        affected = quota_peers(deployment, self.router.deployments)
                        for peer in affected:
                            if peer is not deployment:
                                self._update_headers(peer, headers)
            else:
                deployment.metrics.record_failure()
                if error.category in RATE_LIMITS:
                    deployment.metrics.record_rate_limit()
                if error.category == CATEGORY_AUTH_ERROR:
                    deployment.metrics.record_auth_failure()
                apply_error = current_hash == key_hash and (not stale
                    or error.category == CATEGORY_AUTH_ERROR
                    or error.category in RATE_LIMITS and not deployment.is_quarantined)
                if apply_error:
                    reason = sanitize_secret(error.message, deployment.api_key)
                    if error.category == CATEGORY_AUTH_ERROR:
                        affected = self.quarantine_account(deployment.provider,
                            deployment.account, reason, deployment.api_key)
                    elif error.should_quarantine:
                        deployment.transition_to_quarantine(reason, CircuitState.QUARANTINED)
                    elif error.category in RATE_LIMITS:
                        deployment.quota.quota_scope = error.quota_scope
                        affected = quota_peers(deployment, self.router.deployments)
                        for peer in affected:
                            # Quarantine has precedence over quota cooldown.
                            if peer.is_quarantined:
                                continue
                            deadline = max(error.reset_timestamp or time.time() + error.cooldown_seconds,
                                           peer.quota.reset_at or 0.0, peer.circuit_open_until)
                            peer.update_quota(rpm_limit=error.rpm_limit, rpm_remaining=error.rpm_remaining,
                                rpd_limit=error.rpd_limit, rpd_remaining=error.rpd_remaining,
                                reset_at=deadline, retry_after=max(0.0, deadline - time.time()),
                                quota_scope=error.quota_scope, limit_type=error.limit_type)
                            peer.transition_to_open(max(0.0, deadline - time.time()), reason,
                                reset_at=deadline, request_id=request_id)
                    elif error.category == CATEGORY_SERVER_ERROR:
                        deployment.record_failure(request_id=request_id)
                        if error.cooldown_seconds > 0:
                            deployment.transition_to_open(max(error.cooldown_seconds,
                                deployment.circuit_open_until - time.time()), reason,
                                request_id=request_id)
            self.router.state_manager.snapshot_deployments(affected)

    def persist(self, deployments: Sequence[Deployment]) -> None:
        with self.lock:
            self.router.state_manager.snapshot_deployments(deployments)
