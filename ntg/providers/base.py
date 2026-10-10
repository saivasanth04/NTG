"""Shared provider discovery error classification and state restoration helpers."""

from __future__ import annotations

from enum import Enum
import json
from typing import Any, Optional
import requests

from ntg.core.models import Deployment


class DiscoveryErrorCategory(str, Enum):
    """Normalized categories for provider discovery errors."""

    AUTH_FAILURE = "auth_failure"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    NETWORK_FAILURE = "network_failure"
    MALFORMED_RESPONSE = "malformed_response"
    PROGRAMMING_ERROR = "programming_error"


def classify_discovery_error(
    err: Exception,
    status_code: Optional[int] = None,
) -> tuple[DiscoveryErrorCategory, str]:
    """Distinguish discovery error types.

    Categories:
    - AUTH_FAILURE: 401, 403, invalid API key, unauthenticated, forbidden
    - PROVIDER_UNAVAILABLE: 500, 502, 503, 504, 520, 521, 522, 524, 529, service unavailable, bad gateway
    - NETWORK_FAILURE: ConnectionError, Timeout, DNS error, socket error
    - MALFORMED_RESPONSE: JSONDecodeError, unexpected response schema
    - PROGRAMMING_ERROR: TypeError, AttributeError, NameError, SyntaxError
    """
    if isinstance(err, (TypeError, AttributeError, NameError, SyntaxError)):
        return DiscoveryErrorCategory.PROGRAMMING_ERROR, f"Programming error: {type(err).__name__}: {err}"

    code = status_code
    if code is None:
        if hasattr(err, "status_code") and isinstance(err.status_code, int):
            code = err.status_code
        elif hasattr(err, "code") and isinstance(err.code, int):
            code = err.code
        elif hasattr(err, "response") and hasattr(err.response, "status_code"):
            code = getattr(err.response, "status_code", None)

    msg = str(err).lower()

    if code in (401, 403) or any(
        k in msg
        for k in (
            "unauthenticated",
            "unauthorized",
            "api_key_invalid",
            "invalid api key",
            "invalid_api_key",
            "permission_denied",
            "forbidden",
            "401 unauthorized",
            "403 forbidden",
        )
    ):
        return DiscoveryErrorCategory.AUTH_FAILURE, f"Authentication failure (HTTP {code or 401}): {err}"

    if code in (500, 502, 503, 504, 520, 521, 522, 524, 529) or any(
        k in msg
        for k in (
            "503",
            "502",
            "504",
            "unavailable",
            "bad gateway",
            "gateway timeout",
            "service unavailable",
            "internal server error",
            "overloaded",
        )
    ):
        return DiscoveryErrorCategory.PROVIDER_UNAVAILABLE, f"Provider unavailable (HTTP {code or 503}): {err}"

    if isinstance(err, (json.JSONDecodeError, ValueError)) and any(
        k in msg for k in ("json", "expecting value", "decode", "schema", "malformed")
    ):
        return DiscoveryErrorCategory.MALFORMED_RESPONSE, f"Malformed provider response: {err}"

    if isinstance(
        err,
        (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            TimeoutError,
            ConnectionError,
            OSError,
        ),
    ) or any(
        k in msg
        for k in (
            "connection",
            "timeout",
            "timed out",
            "dns",
            "name resolution",
            "failed to establish",
            "socket",
            "network",
        )
    ):
        return DiscoveryErrorCategory.NETWORK_FAILURE, f"Temporary network failure: {err}"

    if "json" in msg or "decode" in msg or "malformed" in msg:
        return DiscoveryErrorCategory.MALFORMED_RESPONSE, f"Malformed provider response: {err}"

    if code and code >= 500:
        return DiscoveryErrorCategory.PROVIDER_UNAVAILABLE, f"Provider unavailable (HTTP {code}): {err}"
    if code and (code == 401 or code == 403):
        return DiscoveryErrorCategory.AUTH_FAILURE, f"Authentication failure (HTTP {code}): {err}"

    return DiscoveryErrorCategory.NETWORK_FAILURE, f"Temporary discovery error: {err}"


def apply_stored_state(deployment: Deployment, state_manager: Optional[Any]) -> None:
    """Restore persisted circuit state, cooldown, and metrics from StateManager."""
    if not state_manager:
        return

    state_manager.apply_to_deployment(deployment)
    deployment.metrics = state_manager.get_metrics(deployment.id)


_apply_stored_state = apply_stored_state
