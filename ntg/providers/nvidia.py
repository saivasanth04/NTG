"""NVIDIA NIM capability inference and deployment builder."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ntg.core.config import (
    MODEL_GROUP_NVIDIA,
    NVIDIA_ACTUAL_MODEL,
    NVIDIA_KEYS,
    NVIDIA_LITELLM_MODEL,
)
from ntg.core.models import Deployment, ModelCapabilities, QuotaInfo, QuotaScope
from ntg.providers.base import apply_stored_state


def infer_nvidia_capabilities(model_id: str = NVIDIA_ACTUAL_MODEL) -> ModelCapabilities:
    """Infer normalized ModelCapabilities for an NVIDIA NIM model."""
    return ModelCapabilities(
        coding=None,
        reasoning=None,
        vision=None,
        tool_calling=None,
        structured_output=None,
        streaming=True,
        context_window=131072,
    )


def build_nvidia_deployments(
    keys: Optional[Dict[str, str]] = None,
    state_manager: Optional[Any] = None,
) -> List[Deployment]:
    """Create one LiteLLM deployment per configured NVIDIA account using NVIDIA NIM."""
    keys_to_use = keys if keys is not None else NVIDIA_KEYS
    deployments: List[Deployment] = []

    capabilities = infer_nvidia_capabilities(NVIDIA_ACTUAL_MODEL)

    for i in range(1, 5):
        acc_name = f"account_{i}"
        key = keys_to_use.get(acc_name, "").strip()
        if not key:
            continue

        deployment = Deployment(
            id=f"nvidia-{acc_name}",
            provider=MODEL_GROUP_NVIDIA,
            account=acc_name,
            model=NVIDIA_ACTUAL_MODEL,
            logical_model=MODEL_GROUP_NVIDIA,
            litellm_model=NVIDIA_LITELLM_MODEL,
            api_key=key,
            api_base="https://integrate.api.nvidia.com/v1",
            capabilities=capabilities,
            quota=QuotaInfo(quota_scope=QuotaScope.ACCOUNT),
        )
        apply_stored_state(deployment, state_manager)
        deployments.append(deployment)

    return deployments
