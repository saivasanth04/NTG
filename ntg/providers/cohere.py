"""Cohere capability inference and deployment builder."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ntg.core.config import (
    COHERE_ACTUAL_MODEL,
    COHERE_KEYS,
    COHERE_LITELLM_MODEL,
    MODEL_GROUP_COHERE,
)
from ntg.core.models import Deployment, ModelCapabilities, QuotaInfo, QuotaScope
from ntg.providers.base import apply_stored_state


def infer_cohere_capabilities(model_id: str = COHERE_ACTUAL_MODEL) -> ModelCapabilities:
    """Infer normalized ModelCapabilities for a Cohere model."""
    return ModelCapabilities(
        coding=None,
        reasoning=None,
        vision=None,
        tool_calling=None,
        structured_output=None,
        streaming=True,
        context_window=128000,
    )


def build_cohere_deployments(
    keys: Optional[Dict[str, str]] = None,
    state_manager: Optional[Any] = None,
) -> List[Deployment]:
    """Create one LiteLLM deployment per configured Cohere account using native Cohere provider."""
    keys_to_use = keys if keys is not None else COHERE_KEYS
    deployments: List[Deployment] = []

    capabilities = infer_cohere_capabilities(COHERE_ACTUAL_MODEL)

    for i in range(1, 6):
        acc_name = f"account_{i}"
        key = keys_to_use.get(acc_name, "").strip()
        if not key:
            continue

        deployment = Deployment(
            id=f"cohere-{acc_name}",
            provider=MODEL_GROUP_COHERE,
            account=acc_name,
            model=COHERE_ACTUAL_MODEL,
            logical_model=MODEL_GROUP_COHERE,
            litellm_model=COHERE_LITELLM_MODEL,
            api_key=key,
            capabilities=capabilities,
            quota=QuotaInfo(quota_scope=QuotaScope.ACCOUNT),
        )
        apply_stored_state(deployment, state_manager)
        deployments.append(deployment)

    return deployments
