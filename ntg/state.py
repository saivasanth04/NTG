"""State persistence and circuit breaker management across restarts."""

from __future__ import annotations

import json
import logging
import os
import time
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ntg.state")


class CircuitState(str, Enum):
    HEALTHY = "HEALTHY"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"
    QUARANTINED = "QUARANTINED"


class StateManager:
    """Manages persistent deployment circuit breaker, metrics, and quota state across restarts."""

    def __init__(self, state_file: Optional[str] = None):
        if state_file:
            self.state_file = state_file
        else:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ntg_dir = os.path.join(base_dir, ".ntg")
            os.makedirs(ntg_dir, exist_ok=True)
            self.state_file = os.path.join(ntg_dir, "state.json")

        self.state_data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        """Load state data safely from file."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
            except Exception as err:
                logger.warning("Could not read state file %s: %s", self.state_file, err)

        return {
            "deployments": {},
            "accounts": {},
            "discovery_cache": {},
            "metrics": {},
        }

    def save(self) -> None:
        """Persist current state data atomically."""
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            temp_file = f"{self.state_file}.tmp"
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(self.state_data, f, indent=2)
            if os.path.exists(self.state_file):
                os.replace(temp_file, self.state_file)
            else:
                os.rename(temp_file, self.state_file)
        except Exception as err:
            logger.warning("Could not save state file: %s", err)

    def get_cached_discovery(self, key: str) -> List[Any]:
        """Retrieve last-known-good models for a provider account."""
        cache = self.state_data.get("discovery_cache", {})
        return cache.get(key, [])

    def set_cached_discovery(self, key: str, models: List[Any]) -> None:
        """Cache discovered models for a provider account."""
        if "discovery_cache" not in self.state_data:
            self.state_data["discovery_cache"] = {}
        self.state_data["discovery_cache"][key] = list(models)
        self.save()

    def get_circuit_state(self, dep_id: str) -> Optional[CircuitState]:
        """Get circuit state for a deployment ID."""
        saved = self.state_data.get("deployments", {}).get(dep_id, {})
        c_str = saved.get("circuit_state")
        if c_str:
            try:
                return CircuitState(c_str)
            except ValueError:
                pass
        return None

    def get_cooldown_until(self, dep_id: str) -> float:
        """Get cooldown expiration timestamp for a deployment ID."""
        saved = self.state_data.get("deployments", {}).get(dep_id, {})
        return saved.get("reset_at", 0.0)

    def record_circuit_state(self, dep_id: str, state: CircuitState, reason: str = "") -> None:
        """Record circuit state for a deployment ID."""
        if "deployments" not in self.state_data:
            self.state_data["deployments"] = {}
        if dep_id not in self.state_data["deployments"]:
            self.state_data["deployments"][dep_id] = {}
        self.state_data["deployments"][dep_id]["circuit_state"] = state.value
        if reason:
            self.state_data["deployments"][dep_id]["state_reason"] = reason
        self.state_data["deployments"][dep_id]["state_updated_at"] = time.time()
        self.save()

    def record_cooldown(
        self,
        dep_or_id: Any,
        cooldown_seconds_or_until: float,
        reason_or_limit_type: str = "",
        reset_timestamp: Optional[float] = None,
        quota_scope: str = "deployment",
    ) -> None:
        """Record cooldown for either Deployment instance or deployment ID."""
        dep_id = getattr(dep_or_id, "id", str(dep_or_id))
        now = time.time()

        if cooldown_seconds_or_until > 1e8:
            reset_at = cooldown_seconds_or_until
        else:
            cooldown_secs = max(1.0, cooldown_seconds_or_until)
            reset_at = reset_timestamp if reset_timestamp and reset_timestamp > now else (now + cooldown_secs)

        if hasattr(dep_or_id, "circuit_state"):
            dep_or_id.circuit_state = CircuitState.OPEN
            if hasattr(dep_or_id, "quota"):
                dep_or_id.quota.reset_at = reset_at
                dep_or_id.quota.quota_scope = quota_scope
            if hasattr(dep_or_id, "state_reason"):
                dep_or_id.state_reason = reason_or_limit_type

        if "deployments" not in self.state_data:
            self.state_data["deployments"] = {}
        if dep_id not in self.state_data["deployments"]:
            self.state_data["deployments"][dep_id] = {}

        self.state_data["deployments"][dep_id].update({
            "circuit_state": CircuitState.OPEN.value,
            "state_reason": reason_or_limit_type,
            "reset_at": reset_at,
            "quota_scope": quota_scope,
            "state_updated_at": now,
        })
        self.save()

    def record_quarantine(self, dep_or_id: Any, reason: str) -> None:
        """Quarantine a deployment due to invalid credentials or permanent errors."""
        dep_id = getattr(dep_or_id, "id", str(dep_or_id))
        now = time.time()

        if hasattr(dep_or_id, "circuit_state"):
            dep_or_id.circuit_state = CircuitState.QUARANTINED
            if hasattr(dep_or_id, "state_reason"):
                dep_or_id.state_reason = reason

        if "deployments" not in self.state_data:
            self.state_data["deployments"] = {}
        if dep_id not in self.state_data["deployments"]:
            self.state_data["deployments"][dep_id] = {}

        self.state_data["deployments"][dep_id].update({
            "circuit_state": CircuitState.QUARANTINED.value,
            "state_reason": reason,
            "state_updated_at": now,
        })
        self.save()

    def record_metrics(self, dep_id: str, metrics: Any) -> None:
        """Record telemetry metrics for a deployment ID."""
        if "metrics" not in self.state_data:
            self.state_data["metrics"] = {}
        if hasattr(metrics, "__dict__"):
            self.state_data["metrics"][dep_id] = dict(metrics.__dict__)
        elif isinstance(metrics, dict):
            self.state_data["metrics"][dep_id] = dict(metrics)
        self.save()

    def get_metrics(self, dep_id: str) -> Any:
        """Get restored DeploymentMetrics for a deployment ID."""
        from ntg.models import DeploymentMetrics

        raw = self.state_data.get("metrics", {}).get(dep_id, {})
        if raw and isinstance(raw, dict):
            return DeploymentMetrics(**{k: v for k, v in raw.items() if hasattr(DeploymentMetrics, k)})
        return DeploymentMetrics()

    def get_quota(self, dep_id: str) -> Any:
        """Get restored QuotaInfo for a deployment ID."""
        from ntg.models import QuotaInfo

        saved = self.state_data.get("deployments", {}).get(dep_id, {})
        if saved and isinstance(saved, dict):
            return QuotaInfo(
                reset_at=saved.get("reset_at"),
                quota_scope=saved.get("quota_scope", "deployment"),
            )
        return QuotaInfo()

    def record_success(self, deployment: Any) -> None:
        """Record success and transition HALF_OPEN deployments back to HEALTHY."""
        dep_id = getattr(deployment, "id", str(deployment))
        if hasattr(deployment, "circuit_state") and deployment.circuit_state == CircuitState.HALF_OPEN:
            deployment.circuit_state = CircuitState.HEALTHY
            if hasattr(deployment, "state_reason"):
                deployment.state_reason = None
            if hasattr(deployment, "half_open_probes"):
                deployment.half_open_probes = 0

        deps = self.state_data.get("deployments", {})
        if dep_id in deps:
            deps.pop(dep_id, None)
            self.save()

    def apply_to_deployment(self, deployment: Any) -> None:
        """Apply persisted state (quarantine, cooldown, circuit breaker) to a deployment."""
        deps = self.state_data.get("deployments", {})
        saved = deps.get(deployment.id)
        if not saved:
            return

        now = time.time()
        circuit_str = saved.get("circuit_state", CircuitState.HEALTHY.value)
        reset_at = saved.get("reset_at", 0.0)

        # Handle quarantine
        if circuit_str == CircuitState.QUARANTINED.value:
            deployment.circuit_state = CircuitState.QUARANTINED
            deployment.state_reason = saved.get("state_reason", "Quarantined due to authentication error")
            deployment.state_updated_at = saved.get("state_updated_at", now)
            return

        # Handle cooldown (circuit OPEN)
        if circuit_str == CircuitState.OPEN.value and reset_at > 0:
            if now < reset_at:
                deployment.circuit_state = CircuitState.OPEN
                deployment.quota.reset_at = reset_at
                deployment.state_reason = saved.get("state_reason", "Rate limited / cooldown active")
                deployment.state_updated_at = saved.get("state_updated_at", now)
            else:
                # Transition to HALF_OPEN for controlled probing
                deployment.circuit_state = CircuitState.HALF_OPEN
                deployment.state_reason = "Cooldown elapsed; probing deployment health"
                deployment.state_updated_at = now
                deployment.half_open_probes = 1

    def reset_all(self) -> None:
        """Reset all persisted state."""
        self.state_data = {
            "deployments": {},
            "accounts": {},
            "discovery_cache": {},
            "metrics": {},
        }
        self.save()
