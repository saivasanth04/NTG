"""State persistence and circuit breaker management across restarts."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ntg.state")


class CircuitState(str, Enum):
    HEALTHY = "HEALTHY"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"
    QUARANTINED = "QUARANTINED"
    AUTH_FAILED = "AUTH_FAILED"


FORBIDDEN_KEY_PATTERNS = {
    "api_key",
    "key",
    "authorization",
    "token",
    "secret",
    "password",
    "credentials",
    "headers",
    "messages",
    "prompt",
    "request",
}

SAFE_EXACT_KEYS = {
    "quota_scope",
    "limit_type",
    "model",
    "account",
    "provider",
    "circuit_state",
    "state_reason",
    "is_stale",
    "stale_reason",
    "cached_at",
    "error_type",
    "last_error",
    "models",
    "user_requests",
    "upstream_attempts",
    "successes",
    "failures",
    "rate_limits",
    "auth_failures",
    "auth_errors",
}


def _clean_for_persistence(obj: Any) -> Any:
    """Recursively scrub secrets, credentials, and request bodies before saving (Rules 1, 2, 3)."""
    if isinstance(obj, dict):
        cleaned: Dict[str, Any] = {}
        for k, v in obj.items():
            k_str = str(k).lower()
            if k_str in SAFE_EXACT_KEYS:
                cleaned[k] = _clean_for_persistence(v)
            elif any(pat in k_str for pat in FORBIDDEN_KEY_PATTERNS):
                # Rule 1, 2, 3: Omit secret keys, authorization headers, and request content
                continue
            else:
                cleaned[k] = _clean_for_persistence(v)
        return cleaned
    elif isinstance(obj, list):
        return [_clean_for_persistence(item) for item in obj]
    elif isinstance(obj, str):
        try:
            from ntg.models import sanitize_secret
            return sanitize_secret(obj)
        except ImportError:
            return obj
    return obj


class StateManager:
    """Manages persistent deployment circuit breaker, metrics, and quota state across restarts.

    Guarantees:
    - Never persists API keys, Authorization headers, or request content (Rules 1, 2, 3).
    - Loads state on startup and discards expired cooldowns (Rules 4, 5).
    - Handles missing or corrupted state files without crashing (Rules 6, 7).
    - Thread-safe atomic writes via temporary file replacement and in-memory RLock (Rule 10).
    - Integrates with the existing Deployment model as the source of truth (Rule 9).
    """

    def __init__(self, state_file: Optional[str] = None):
        self._lock = threading.RLock()
        if state_file:
            self.state_file = state_file
        else:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ntg_dir = os.path.join(base_dir, ".ntg")
            os.makedirs(ntg_dir, exist_ok=True)
            self.state_file = os.path.join(ntg_dir, "state.json")

        self.state_data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        """Load state data safely from file, discarding expired information (Rules 4, 5, 6, 7)."""
        data: Dict[str, Any] = {}
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        data = loaded
            except Exception as err:
                logger.warning(
                    "Could not read state file %s (corrupted or unreadable): %s",
                    self.state_file,
                    err,
                )
                try:
                    corrupted_backup = f"{self.state_file}.corrupted.{int(time.time())}"
                    os.rename(self.state_file, corrupted_backup)
                except Exception:
                    pass

        # Populate baseline structure if missing or corrupt
        for section in ("deployments", "accounts", "discovery_cache", "metrics", "quota"):
            if section not in data or not isinstance(data[section], dict):
                data[section] = {}

        now = time.time()

        # Rule 5: Discard expired cooldown and reset information on startup
        deployments = data.get("deployments", {})
        for dep_id, entry in list(deployments.items()):
            if not isinstance(entry, dict):
                continue
            if entry.get("circuit_state") == CircuitState.OPEN.value:
                open_until = entry.get("circuit_open_until", 0.0) or entry.get("reset_at", 0.0)
                if open_until > 0 and now >= open_until:
                    # Expired cooldown: discard expired cooldown information
                    entry.pop("reset_at", None)
                    entry.pop("circuit_open_until", None)
                    entry["circuit_state"] = CircuitState.HALF_OPEN.value
                    entry["state_reason"] = "Expired cooldown discarded on startup; ready for probe"

        quotas = data.get("quota", {})
        for dep_id, q_entry in list(quotas.items()):
            if not isinstance(q_entry, dict):
                continue
            q_reset = q_entry.get("reset_at")
            if q_reset and isinstance(q_reset, (int, float)) and now >= q_reset:
                q_entry["reset_at"] = None
                q_entry["retry_after"] = 0.0

        return data

    def save(self) -> None:
        """Persist current state data atomically and safely (Rules 1, 2, 3, 10)."""
        with self._lock:
            temp_file = None
            try:
                os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
                clean_data = _clean_for_persistence(self.state_data)
                clean_data["version"] = 1
                clean_data["updated_at"] = time.time()

                # Process- and thread-unique temp file to avoid concurrent collisions
                temp_file = f"{self.state_file}.tmp.{os.getpid()}.{threading.get_ident()}"
                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(clean_data, f, indent=2)

                if os.path.exists(self.state_file):
                    os.replace(temp_file, self.state_file)
                else:
                    os.rename(temp_file, self.state_file)
            except Exception as err:
                logger.warning("Could not save state file: %s", err)
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except Exception:
                        pass

    def snapshot_deployment(self, deployment: Any) -> None:
        """Persist runtime state directly from a Deployment instance (Rule 9)."""
        with self._lock:
            dep_id = getattr(deployment, "id", str(deployment))
            now = time.time()
            if "deployments" not in self.state_data:
                self.state_data["deployments"] = {}
            if dep_id not in self.state_data["deployments"]:
                self.state_data["deployments"][dep_id] = {}

            c_state = getattr(deployment, "circuit_state", CircuitState.HEALTHY)
            c_val = c_state.value if hasattr(c_state, "value") else str(c_state)

            quota_obj = getattr(deployment, "quota", None)
            reset_at = getattr(quota_obj, "reset_at", 0.0) if quota_obj else 0.0
            quota_scope = getattr(quota_obj, "quota_scope", "deployment") if quota_obj else "deployment"

            self.state_data["deployments"][dep_id].update({
                "circuit_state": c_val,
                "state_reason": getattr(deployment, "state_reason", None),
                "circuit_open_until": getattr(deployment, "circuit_open_until", 0.0),
                "reset_at": reset_at or 0.0,
                "quota_scope": quota_scope,
                "state_updated_at": getattr(deployment, "state_updated_at", now) or now,
                "last_available": deployment.is_available(now) if hasattr(deployment, "is_available") else True,
            })

            if hasattr(deployment, "metrics"):
                self.record_metrics(dep_id, deployment.metrics)
            if hasattr(deployment, "quota"):
                self.record_quota(dep_id, deployment.quota)
            self.save()

    def get_cached_discovery(self, key: str) -> List[Any]:
        """Retrieve last-known-good models for a provider account.

        Handles both legacy list cache entries and rich metadata dict entries.
        """
        with self._lock:
            cache = self.state_data.get("discovery_cache", {})
            entry = cache.get(key)
            if entry is None:
                return []
            if isinstance(entry, dict) and "models" in entry:
                raw_models = entry["models"]
                is_stale = entry.get("is_stale", False)
                stale_reason = entry.get("stale_reason")
                if isinstance(raw_models, list):
                    res = []
                    for item in raw_models:
                        if isinstance(item, dict) and is_stale:
                            d = dict(item)
                            d["is_stale"] = True
                            if stale_reason:
                                d["stale_reason"] = stale_reason
                            res.append(d)
                        else:
                            res.append(item)
                    return res
                elif isinstance(raw_models, dict):
                    if is_stale:
                        d = dict(raw_models)
                        d["is_stale"] = True
                        if stale_reason:
                            d["stale_reason"] = stale_reason
                        return d  # type: ignore
                    return raw_models  # type: ignore
                return []
            elif isinstance(entry, list):
                return list(entry)
            elif isinstance(entry, dict):
                return entry  # type: ignore
            return []

    def get_discovery_entry(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve complete discovery metadata entry for a cache key."""
        with self._lock:
            cache = self.state_data.get("discovery_cache", {})
            entry = cache.get(key)
            if entry is None:
                return None
            if isinstance(entry, dict) and "models" in entry:
                return dict(entry)
            elif isinstance(entry, list):
                return {
                    "models": list(entry),
                    "cached_at": 0.0,
                    "is_stale": False,
                    "stale_reason": None,
                    "error_type": None,
                    "last_error": None,
                }
            elif isinstance(entry, dict):
                return {
                    "models": dict(entry),
                    "cached_at": 0.0,
                    "is_stale": False,
                    "stale_reason": None,
                    "error_type": None,
                    "last_error": None,
                }
            return None

    def set_cached_discovery(
        self,
        key: str,
        models: Any,
        is_stale: bool = False,
        stale_reason: Optional[str] = None,
        error_type: Optional[str] = None,
        last_error: Optional[str] = None,
    ) -> None:
        """Cache discovered models for a provider account with staleness metadata."""
        with self._lock:
            if "discovery_cache" not in self.state_data:
                self.state_data["discovery_cache"] = {}
            entry = {
                "models": list(models) if isinstance(models, list) else models,
                "cached_at": time.time(),
                "is_stale": is_stale,
                "stale_reason": stale_reason,
                "error_type": error_type,
                "last_error": last_error,
            }
            self.state_data["discovery_cache"][key] = entry
            self.save()

    def mark_discovery_stale(
        self,
        key: str,
        reason: str,
        error_type: str = "temporary_failure",
        last_error: Optional[str] = None,
    ) -> Optional[List[Any]]:
        """Mark cached discovery as stale instead of deleting it (Rules 1, 2). Returns cached models."""
        with self._lock:
            if "discovery_cache" not in self.state_data:
                self.state_data["discovery_cache"] = {}
            cache = self.state_data["discovery_cache"]
            entry = cache.get(key)
            if not entry:
                new_entry = {
                    "models": [],
                    "cached_at": time.time(),
                    "is_stale": True,
                    "stale_reason": reason,
                    "error_type": error_type,
                    "last_error": last_error,
                }
                cache[key] = new_entry
                self.save()
                return []

            if isinstance(entry, dict) and "models" in entry:
                entry["is_stale"] = True
                entry["stale_reason"] = reason
                entry["error_type"] = error_type
                if last_error:
                    entry["last_error"] = last_error
                self.save()
                return self.get_cached_discovery(key)
            elif isinstance(entry, (list, dict)):
                models = list(entry) if isinstance(entry, list) else entry
                new_entry = {
                    "models": models,
                    "cached_at": time.time(),
                    "is_stale": True,
                    "stale_reason": reason,
                    "error_type": error_type,
                    "last_error": last_error,
                }
                cache[key] = new_entry
                self.save()
                return self.get_cached_discovery(key)
            return None

    def is_discovery_stale(self, key: str) -> bool:
        """Return True if cached discovery for key is marked stale."""
        with self._lock:
            entry = self.get_discovery_entry(key)
            return bool(entry.get("is_stale")) if entry else False

    def get_circuit_state(self, dep_id: str) -> Optional[CircuitState]:
        """Get circuit state for a deployment ID."""
        with self._lock:
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
        with self._lock:
            saved = self.state_data.get("deployments", {}).get(dep_id, {})
            return saved.get("reset_at", 0.0)

    def record_circuit_state(
        self,
        dep_id: str,
        state: CircuitState,
        reason: str = "",
        recovery_time: float = 0.0,
    ) -> None:
        """Record circuit state for a deployment ID."""
        with self._lock:
            if "deployments" not in self.state_data:
                self.state_data["deployments"] = {}
            if dep_id not in self.state_data["deployments"]:
                self.state_data["deployments"][dep_id] = {}
            self.state_data["deployments"][dep_id]["circuit_state"] = state.value
            if state == CircuitState.HEALTHY:
                self.state_data["deployments"][dep_id]["circuit_open_until"] = 0.0
                self.state_data["deployments"][dep_id].pop("reset_at", None)
                self.state_data["deployments"][dep_id]["state_reason"] = reason or "Healthy"
            else:
                if reason:
                    self.state_data["deployments"][dep_id]["state_reason"] = reason
                if recovery_time > 0:
                    self.state_data["deployments"][dep_id]["circuit_open_until"] = recovery_time
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
        """Persist cooldown for deployment ID or instance (strictly persistence, no live mutations)."""
        with self._lock:
            dep_id = getattr(dep_or_id, "id", str(dep_or_id))
            now = time.time()

            if cooldown_seconds_or_until > 1e8:
                reset_at = cooldown_seconds_or_until
            else:
                cooldown_secs = max(1.0, cooldown_seconds_or_until)
                reset_at = reset_timestamp if reset_timestamp and reset_timestamp > now else (now + cooldown_secs)

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

    def record_quarantine(
        self,
        dep_or_id: Any,
        reason: str,
        state: CircuitState = CircuitState.AUTH_FAILED,
    ) -> None:
        """Persist quarantine state for deployment ID (strictly persistence, no live mutations)."""
        with self._lock:
            dep_id = getattr(dep_or_id, "id", str(dep_or_id))
            now = time.time()

            if "deployments" not in self.state_data:
                self.state_data["deployments"] = {}
            if dep_id not in self.state_data["deployments"]:
                self.state_data["deployments"][dep_id] = {}

            self.state_data["deployments"][dep_id].update({
                "circuit_state": state.value,
                "state_reason": reason,
                "state_updated_at": now,
            })
            self.save()

    def record_metrics(self, dep_id: str, metrics: Any) -> None:
        """Record telemetry metrics for a deployment ID (Rules 6, 8, 10)."""
        with self._lock:
            if "metrics" not in self.state_data:
                self.state_data["metrics"] = {}
            if hasattr(metrics, "to_dict"):
                self.state_data["metrics"][dep_id] = metrics.to_dict()
            elif hasattr(metrics, "__dict__"):
                self.state_data["metrics"][dep_id] = {
                    k.lstrip("_"): v for k, v in metrics.__dict__.items() if not k.startswith("_lock")
                }
            elif isinstance(metrics, dict):
                self.state_data["metrics"][dep_id] = dict(metrics)
            self.save()

    def get_metrics(self, dep_id: str) -> Any:
        """Get restored DeploymentMetrics for a deployment ID."""
        with self._lock:
            from ntg.models import DeploymentMetrics

            raw = self.state_data.get("metrics", {}).get(dep_id, {})
            if raw and isinstance(raw, dict):
                clean = {}
                for k in ("user_requests", "upstream_attempts", "successes", "failures", "rate_limits", "auth_failures"):
                    if k in raw:
                        clean[k] = raw[k]
                if "auth_failures" not in clean and "auth_errors" in raw:
                    clean["auth_failures"] = raw["auth_errors"]
                return DeploymentMetrics(**clean)
            return DeploymentMetrics()

    def record_quota(self, dep_id: str, quota: Any = None, **kwargs) -> None:
        """Record normalized quota metadata for a deployment ID."""
        with self._lock:
            if "quota" not in self.state_data:
                self.state_data["quota"] = {}
            data = {}
            if hasattr(quota, "to_dict"):
                data = quota.to_dict()
            elif isinstance(quota, dict):
                data = dict(quota)
            if kwargs:
                data.update(kwargs)
            if data:
                self.state_data["quota"][dep_id] = data
                self.save()

    def get_quota(self, dep_id: str) -> Any:
        """Get restored QuotaInfo for a deployment ID."""
        with self._lock:
            from ntg.models import QuotaInfo

            raw = self.state_data.get("quota", {}).get(dep_id, {})
            if raw and isinstance(raw, dict):
                return QuotaInfo(
                    rpm_limit=raw.get("rpm_limit"),
                    rpm_remaining=raw.get("rpm_remaining"),
                    rpd_limit=raw.get("rpd_limit"),
                    rpd_remaining=raw.get("rpd_remaining"),
                    reset_at=raw.get("reset_at"),
                    retry_after=raw.get("retry_after"),
                    quota_scope=raw.get("quota_scope", "unknown"),
                    limit_type=raw.get("limit_type"),
                    last_updated=raw.get("last_updated", 0.0),
                )
            saved = self.state_data.get("deployments", {}).get(dep_id, {})
            if saved and isinstance(saved, dict):
                return QuotaInfo(
                    reset_at=saved.get("reset_at"),
                    quota_scope=saved.get("quota_scope", "unknown"),
                )
            return QuotaInfo()

    def record_success(self, dep_or_id: Any) -> None:
        """Persist recovery / healthy state for deployment (strictly persistence, no live mutations)."""
        with self._lock:
            dep_id = getattr(dep_or_id, "id", str(dep_or_id))
            deps = self.state_data.get("deployments", {})
            if dep_id in deps:
                deps[dep_id]["circuit_state"] = CircuitState.HEALTHY.value
                deps[dep_id]["circuit_open_until"] = 0.0
                deps[dep_id].pop("reset_at", None)
                deps[dep_id]["state_reason"] = "Recovered to HEALTHY"
                deps[dep_id]["state_updated_at"] = time.time()
                self.save()

    def apply_to_deployment(self, deployment: Any) -> None:
        """Apply persisted state (quarantine, cooldown, circuit breaker, quota) to a deployment."""
        with self._lock:
            # Restore persisted quota
            if deployment.id in self.state_data.get("quota", {}):
                deployment.quota = self.get_quota(deployment.id)

            # Restore metrics if deployment has zero counters
            if hasattr(deployment, "metrics") and deployment.id in self.state_data.get("metrics", {}):
                if deployment.metrics.user_requests == 0 and deployment.metrics.upstream_attempts == 0:
                    deployment.metrics = self.get_metrics(deployment.id)

            deps = self.state_data.get("deployments", {})
            saved = deps.get(deployment.id)
            if not saved:
                return

            now = time.time()
            saved_updated_at = saved.get("state_updated_at", 0.0)
            if getattr(deployment, "state_updated_at", 0.0) > saved_updated_at and saved_updated_at > 0:
                # Live state is newer than persisted state: do not overwrite newer live state
                return

            circuit_str = saved.get("circuit_state", CircuitState.HEALTHY.value)
            reset_at = saved.get("reset_at", 0.0)
            open_until = saved.get("circuit_open_until", 0.0) or reset_at

            # Handle quarantine and authentication failure
            if circuit_str in (CircuitState.QUARANTINED.value, CircuitState.AUTH_FAILED.value):
                try:
                    deployment.circuit_state = CircuitState(circuit_str)
                except ValueError:
                    deployment.circuit_state = CircuitState.AUTH_FAILED
                deployment.state_reason = saved.get("state_reason", "Quarantined due to authentication error")
                deployment.state_updated_at = saved_updated_at or now
                return

            # Handle circuit HEALTHY
            if circuit_str == CircuitState.HEALTHY.value:
                deployment.circuit_state = CircuitState.HEALTHY
                deployment.circuit_open_until = 0.0
                deployment.state_reason = saved.get("state_reason", "Healthy")
                deployment.state_updated_at = saved_updated_at or now
                return

            # Handle circuit HALF_OPEN
            if circuit_str == CircuitState.HALF_OPEN.value:
                deployment.circuit_state = CircuitState.HALF_OPEN
                deployment.circuit_open_until = 0.0
                deployment.state_reason = saved.get("state_reason", "Ready for probe")
                deployment.state_updated_at = saved_updated_at or now
                deployment.half_open_probes = 0
                if hasattr(deployment, "_active_probe_request_ids"):
                    deployment._active_probe_request_ids.clear()
                return

            # Handle circuit OPEN (server breaker or quota cooldown)
            if circuit_str == CircuitState.OPEN.value:
                if open_until > 0 and now < open_until:
                    deployment.circuit_state = CircuitState.OPEN
                    deployment.circuit_open_until = open_until
                    if reset_at > 0:
                        deployment.quota.reset_at = reset_at
                    deployment.state_reason = saved.get("state_reason", "Circuit breaker OPEN / cooldown active")
                    deployment.state_updated_at = saved_updated_at or now
                else:
                    # Transition to HALF_OPEN for controlled probing (Rule 3)
                    deployment.circuit_state = CircuitState.HALF_OPEN
                    deployment.circuit_open_until = 0.0
                    deployment.state_reason = "Recovery condition reached; entering HALF_OPEN for controlled probe"
                    deployment.state_updated_at = now
                    deployment.half_open_probes = 0
                    if hasattr(deployment, "_active_probe_request_ids"):
                        deployment._active_probe_request_ids.clear()

    def reset_all(self) -> None:
        """Reset all persisted state."""
        with self._lock:
            self.state_data = {
                "deployments": {},
                "accounts": {},
                "discovery_cache": {},
                "metrics": {},
                "quota": {},
            }
            self.save()
