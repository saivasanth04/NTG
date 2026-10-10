"""Smart routing engine, request requirements, telemetry callbacks, and persistent state."""

from ntg.router.engine import OpenRouterFallbackRouter, UnifiedNTGRouter
from ntg.router.requirements import (
    NoEligibleDeploymentsError,
    RequestRequirements,
    extract_request_requirements,
    parse_capabilities,
)
from ntg.router.state import CircuitState, StateManager, clean_for_persistence
from ntg.router.telemetry import ActiveRequest, NTGTelemetryLogger

__all__ = [
    "UnifiedNTGRouter",
    "OpenRouterFallbackRouter",
    "NoEligibleDeploymentsError",
    "RequestRequirements",
    "extract_request_requirements",
    "parse_capabilities",
    "StateManager",
    "CircuitState",
    "clean_for_persistence",
    "ActiveRequest",
    "NTGTelemetryLogger",
]
