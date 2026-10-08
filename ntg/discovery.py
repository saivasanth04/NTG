"""Provider discovery and deployment building layer."""

from __future__ import annotations

import json
import logging
import os
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Set
import requests

from ntg.config import (
    COHERE_ACTUAL_MODEL,
    COHERE_KEYS,
    COHERE_LITELLM_MODEL,
    GEMINI_KEYS,
    GROQ_KEYS,
    MODEL_GROUP_COHERE,
    MODEL_GROUP_GEMINI,
    MODEL_GROUP_GROQ,
    MODEL_GROUP_NVIDIA,
    MODEL_GROUP_OPENROUTER,
    NVIDIA_ACTUAL_MODEL,
    NVIDIA_KEYS,
    NVIDIA_LITELLM_MODEL,
    OPENROUTER_ACTUAL_MODEL,
    OPENROUTER_KEYS,
    OPENROUTER_LITELLM_MODEL,
)
from ntg.models import (
    CircuitState,
    Deployment,
    ModelCapabilities,
    QuotaInfo,
    QuotaScope,
    sanitize_secret,
)
from ntg.state import StateManager

logger = logging.getLogger("ntg.discovery")


class DiscoveryErrorCategory(str, Enum):
    """Normalized categories for provider discovery errors (Rule 6)."""
    AUTH_FAILURE = "auth_failure"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    NETWORK_FAILURE = "network_failure"
    MALFORMED_RESPONSE = "malformed_response"
    PROGRAMMING_ERROR = "programming_error"


def classify_discovery_error(
    err: Exception,
    status_code: Optional[int] = None,
) -> tuple[DiscoveryErrorCategory, str]:
    """Distinguish discovery error types (Rule 6).

    Categories:
    - AUTH_FAILURE: 401, 403, invalid API key, unauthenticated, forbidden
    - PROVIDER_UNAVAILABLE: 500, 502, 503, 504, 520, 521, 522, 524, 529, service unavailable, bad gateway
    - NETWORK_FAILURE: ConnectionError, Timeout, DNS error, socket error
    - MALFORMED_RESPONSE: JSONDecodeError, unexpected response schema
    - PROGRAMMING_ERROR: TypeError, AttributeError, NameError, KeyError, IndexError, SyntaxError
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

    if code in (401, 403) or any(k in msg for k in ("unauthenticated", "unauthorized", "api_key_invalid", "invalid api key", "invalid_api_key", "permission_denied", "forbidden", "401 unauthorized", "403 forbidden")):
        return DiscoveryErrorCategory.AUTH_FAILURE, f"Authentication failure (HTTP {code or 401}): {err}"

    if code in (500, 502, 503, 504, 520, 521, 522, 524, 529) or any(k in msg for k in ("503", "502", "504", "unavailable", "bad gateway", "gateway timeout", "service unavailable", "internal server error", "overloaded")):
        return DiscoveryErrorCategory.PROVIDER_UNAVAILABLE, f"Provider unavailable (HTTP {code or 503}): {err}"

    if isinstance(err, (json.JSONDecodeError, ValueError)) and any(k in msg for k in ("json", "expecting value", "decode", "schema", "malformed")):
        return DiscoveryErrorCategory.MALFORMED_RESPONSE, f"Malformed provider response: {err}"

    if isinstance(err, (requests.exceptions.Timeout, requests.exceptions.ConnectionError, TimeoutError, ConnectionError, OSError)) or any(k in msg for k in ("connection", "timeout", "timed out", "dns", "name resolution", "failed to establish", "socket", "network")):
        return DiscoveryErrorCategory.NETWORK_FAILURE, f"Temporary network failure: {err}"

    if "json" in msg or "decode" in msg or "malformed" in msg:
        return DiscoveryErrorCategory.MALFORMED_RESPONSE, f"Malformed provider response: {err}"

    if code and code >= 500:
        return DiscoveryErrorCategory.PROVIDER_UNAVAILABLE, f"Provider unavailable (HTTP {code}): {err}"
    if code and (code == 401 or code == 403):
        return DiscoveryErrorCategory.AUTH_FAILURE, f"Authentication failure (HTTP {code}): {err}"

    return DiscoveryErrorCategory.NETWORK_FAILURE, f"Temporary discovery error: {err}"


def infer_groq_capabilities(model_id: str, context_window: int = 8192) -> ModelCapabilities:
    """Infer normalized ModelCapabilities for a Groq model using provider metadata and model characteristics."""
    m_lower = model_id.lower()

    # Tool calling support on Groq: supported on Llama 3.1/3.3, Mixtral; disabled on DeepSeek R1 distill
    if any(kw in m_lower for kw in ("r1", "deepseek-r1")):
        tool_calling: Optional[bool] = False
    elif any(kw in m_lower for kw in ("tool", "llama-3.3", "llama-3.1", "mixtral-8x7b", "qwen-2.5")):
        tool_calling = True
    elif "guard" in m_lower or "whisper" in m_lower:
        tool_calling = False
    else:
        tool_calling = None

    # Coding capabilities
    if any(kw in m_lower for kw in ("coder", "code", "llama-3.3", "llama-3.1-70b", "deepseek", "qwen-2.5")):
        coding: Optional[bool] = True
    elif any(kw in m_lower for kw in ("llama-3.1", "gemma", "mistral", "mixtral")):
        coding = True
    elif "guard" in m_lower or "whisper" in m_lower:
        coding = False
    else:
        coding = None

    # Reasoning / thinking capabilities
    if any(kw in m_lower for kw in ("r1", "reasoning", "thinking")):
        reasoning: Optional[bool] = True
    elif any(kw in m_lower for kw in ("llama-3.3-70b", "llama-3.1-70b", "deepseek")):
        reasoning = True
    elif "guard" in m_lower or "whisper" in m_lower or "8b" in m_lower:
        reasoning = False
    else:
        reasoning = None

    # Vision capabilities
    if any(kw in m_lower for kw in ("vision", "llava", "scenecap")):
        vision: Optional[bool] = True
    else:
        vision = False

    # Structured output (Groq chat API supports json_object / json_schema for standard chat models)
    structured_output: Optional[bool] = False if ("guard" in m_lower or "whisper" in m_lower) else True

    # Streaming
    streaming: Optional[bool] = True

    return ModelCapabilities(
        coding=coding,
        reasoning=reasoning,
        vision=vision,
        tool_calling=tool_calling,
        structured_output=structured_output,
        streaming=streaming,
        context_window=context_window,
    )


def discover_groq_models(
    api_key: str,
    account_name: str = "",
    state_manager: Optional[StateManager] = None,
) -> List[Dict[str, Any]]:
    """Dynamically discover usable text/chat models for a given Groq API key.

    Uses authoritative metadata (e.g. context_window, active status) from the Groq API.
    Saves successful discovery into StateManager cache; on discovery failure/offline,
    falls back to last-known-good cached discovery for this account.
    """
    if not api_key or not api_key.strip():
        return []

    cache_key = f"groq_{account_name}" if account_name else "groq_default"
    url = "https://api.groq.com/openai/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "User-Agent": "NTG-LiteLLM-Router/1.0",
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        status_code = response.status_code

        if status_code == 200:
            try:
                payload = response.json()
            except Exception as json_err:
                cat, detail = classify_discovery_error(json_err, status_code=200)
                logger.warning(
                    "[Groq Discovery] %s for %s: %s",
                    cat.value,
                    account_name or "default",
                    sanitize_secret(detail, api_key),
                )
                if state_manager:
                    state_manager.mark_discovery_stale(
                        cache_key,
                        reason=detail,
                        error_type=cat.value,
                        last_error=sanitize_secret(str(json_err), api_key),
                    )
                    cached = state_manager.get_cached_discovery(cache_key)
                    if cached:
                        return cached
                return []

            model_entries = payload.get("data", [])
            if not isinstance(model_entries, list):
                detail = "Malformed provider response: 'data' is not a list"
                logger.warning(
                    "[Groq Discovery] malformed_response for %s: %s",
                    account_name or "default",
                    detail,
                )
                if state_manager:
                    state_manager.mark_discovery_stale(
                        cache_key,
                        reason=detail,
                        error_type=DiscoveryErrorCategory.MALFORMED_RESPONSE.value,
                    )
                    cached = state_manager.get_cached_discovery(cache_key)
                    if cached:
                        return cached
                return []

            excluded_keywords = [
                "whisper",
                "transcribe",
                "audio",
                "speech",
                "tts",
                "text-to-speech",
                "embed",
                "embedding",
                "guard",
                "moderation",
                "image",
                "imagen",
                "veo",
                "clip",
                "dall-e",
                "robotics",
                "computer-use",
            ]

            discovered: List[Dict[str, Any]] = []
            seen: Set[str] = set()

            for entry in model_entries:
                if not isinstance(entry, dict):
                    continue

                model_id = entry.get("id", "").strip()
                if not model_id or model_id in seen:
                    continue

                if entry.get("active") is False:
                    continue

                model_id_lower = model_id.lower()
                if any(exc in model_id_lower for exc in excluded_keywords):
                    continue

                seen.add(model_id)
                context_window = int(entry.get("context_window", 8192))
                discovered.append({
                    "id": model_id,
                    "context_window": context_window,
                })

            if discovered and state_manager:
                state_manager.set_cached_discovery(cache_key, discovered, is_stale=False)

            return discovered

        else:
            err_msg = f"HTTP {status_code}: {response.text[:200]}"
            cat, detail = classify_discovery_error(Exception(err_msg), status_code=status_code)
            logger.warning(
                "[Groq Discovery] %s for %s: %s",
                cat.value,
                account_name or "default",
                sanitize_secret(detail, api_key),
            )
            if state_manager:
                if cat == DiscoveryErrorCategory.AUTH_FAILURE:
                    state_manager.mark_discovery_stale(
                        cache_key,
                        reason=detail,
                        error_type=cat.value,
                        last_error=sanitize_secret(err_msg, api_key),
                    )
                    return []
                else:
                    state_manager.mark_discovery_stale(
                        cache_key,
                        reason=detail,
                        error_type=cat.value,
                        last_error=sanitize_secret(err_msg, api_key),
                    )
                    cached = state_manager.get_cached_discovery(cache_key)
                    if cached:
                        return cached
            return []

    except Exception as err:
        cat, detail = classify_discovery_error(err)
        if cat == DiscoveryErrorCategory.PROGRAMMING_ERROR:
            logger.error(
                "[Groq Discovery] Programming error for %s: %s",
                account_name or "default",
                err,
                exc_info=True,
            )
            raise err

        logger.warning(
            "[Groq Discovery] %s for %s: %s",
            cat.value,
            account_name or "default",
            sanitize_secret(detail, api_key),
        )
        if state_manager:
            state_manager.mark_discovery_stale(
                cache_key,
                reason=detail,
                error_type=cat.value,
                last_error=sanitize_secret(str(err), api_key),
            )
            cached = state_manager.get_cached_discovery(cache_key)
            if cached:
                return cached
        return []


def _apply_stored_state(deployment: Deployment, state_manager: Optional[StateManager]) -> None:
    """Restore persisted circuit state, cooldown, and metrics from StateManager."""
    if not state_manager:
        return

    state_manager.apply_to_deployment(deployment)
    deployment.metrics = state_manager.get_metrics(deployment.id)


def infer_openrouter_capabilities(
    model_id: str = OPENROUTER_ACTUAL_MODEL,
    state_manager: Optional[StateManager] = None,
) -> ModelCapabilities:
    """Infer capabilities for an OpenRouter model using authoritative metadata from OpenRouter API or cache."""
    cache_key = f"openrouter_model_{model_id.replace('/', '_').replace(':', '_')}"
    model_meta = None
    if state_manager:
        cached = state_manager.get_cached_discovery(cache_key)
        if isinstance(cached, dict):
            model_meta = cached

    if not model_meta:
        try:
            url = "https://openrouter.ai/api/v1/models"
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                match = next((m for m in data if m.get("id") == model_id), None)
                if match:
                    model_meta = match
                    if state_manager:
                        state_manager.set_cached_discovery(cache_key, match, is_stale=False)
            else:
                cat, detail = classify_discovery_error(Exception(f"HTTP {resp.status_code}"), status_code=resp.status_code)
                if state_manager:
                    state_manager.mark_discovery_stale(cache_key, reason=detail, error_type=cat.value)
        except Exception as err:
            cat, detail = classify_discovery_error(err)
            if cat == DiscoveryErrorCategory.PROGRAMMING_ERROR:
                raise err
            if state_manager:
                state_manager.mark_discovery_stale(cache_key, reason=detail, error_type=cat.value)

    if isinstance(model_meta, dict):
        context_window = int(model_meta.get("context_length") or 1000000)
        arch = model_meta.get("architecture", {}) or {}
        input_mods = arch.get("input_modalities", [])
        vision: Optional[bool] = ("image" in input_mods) if input_mods else False
        params = model_meta.get("supported_parameters", []) or []
        tool_calling: Optional[bool] = ("tools" in params) if params else True
        reasoning: Optional[bool] = (
            ("reasoning" in params or "include_reasoning" in params or bool(model_meta.get("reasoning")))
            if params
            else True
        )
        structured_output: Optional[bool] = (
            ("structured_outputs" in params or "response_format" in params)
            if params
            else True
        )
        coding: Optional[bool] = True
        return ModelCapabilities(
            coding=coding,
            reasoning=reasoning,
            vision=vision,
            tool_calling=tool_calling,
            structured_output=structured_output,
            streaming=True,
            context_window=context_window,
        )

    # Authoritative known specs for default Nemotron 3 Ultra
    return ModelCapabilities(
        coding=True,
        reasoning=True,
        vision=False,
        tool_calling=True,
        structured_output=True,
        streaming=True,
        context_window=1000000,
    )


def build_openrouter_deployments(
    keys: Optional[Dict[str, str]] = None,
    state_manager: Optional[StateManager] = None,
) -> List[Deployment]:
    """Create 5 independent LiteLLM deployments for OpenRouter accounts."""
    keys_to_use = keys if keys is not None else OPENROUTER_KEYS
    deployments: List[Deployment] = []

    capabilities = infer_openrouter_capabilities(OPENROUTER_ACTUAL_MODEL, state_manager=state_manager)

    for i in range(1, 6):
        acc_name = f"account_{i}"
        key = keys_to_use.get(acc_name, "").strip()

        deployment = Deployment(
            id=f"openrouter-{acc_name}",
            provider=MODEL_GROUP_OPENROUTER,
            account=acc_name,
            model=OPENROUTER_ACTUAL_MODEL,
            logical_model=MODEL_GROUP_OPENROUTER,
            litellm_model=OPENROUTER_LITELLM_MODEL,
            api_key=key,
            capabilities=capabilities,
            quota=QuotaInfo(quota_scope=QuotaScope.ACCOUNT),
        )
        _apply_stored_state(deployment, state_manager)
        deployments.append(deployment)

    return deployments


def build_groq_deployments(
    keys: Optional[Dict[str, str]] = None,
    state_manager: Optional[StateManager] = None,
) -> List[Deployment]:
    """Dynamically discover chat-capable models and create deployments per Groq account."""
    keys_to_use = keys if keys is not None else GROQ_KEYS
    deployments: List[Deployment] = []

    for i in range(1, 6):
        acc_name = f"account_{i}"
        key = keys_to_use.get(acc_name, "").strip()
        if not key:
            continue

        try:
            discovered_models = discover_groq_models(key, account_name=acc_name, state_manager=state_manager)
            cache_key = f"groq_{acc_name}"
            is_cache_stale = state_manager.is_discovery_stale(cache_key) if state_manager else False
            entry_meta = state_manager.get_discovery_entry(cache_key) if state_manager else None
            stale_reason = entry_meta.get("stale_reason") if entry_meta else None
            is_auth_err = entry_meta.get("error_type") == "auth_failure" if entry_meta else False

            for item in discovered_models:
                if isinstance(item, dict):
                    model_id = item.get("id", "")
                    context_window = item.get("context_window", 8192)
                    is_stale = item.get("is_stale", is_cache_stale)
                    item_stale_reason = item.get("stale_reason", stale_reason)
                else:
                    model_id = str(item)
                    context_window = 8192
                    is_stale = is_cache_stale
                    item_stale_reason = stale_reason

                if not model_id:
                    continue

                dep_id = f"groq-{acc_name}-{model_id}"
                caps = infer_groq_capabilities(model_id, context_window=context_window)
                deployment = Deployment(
                    id=dep_id,
                    provider=MODEL_GROUP_GROQ,
                    account=acc_name,
                    model=model_id,
                    logical_model=MODEL_GROUP_GROQ,
                    litellm_model=f"groq/{model_id}",
                    api_key=key,
                    capabilities=caps,
                    quota=QuotaInfo(quota_scope=QuotaScope.ACCOUNT_MODEL),
                    is_stale=is_stale,
                    stale_reason=item_stale_reason,
                    last_discovered_at=time.time(),
                )
                if is_auth_err:
                    deployment.circuit_state = CircuitState.AUTH_FAILED
                    deployment.state_reason = item_stale_reason or "Authentication failed during discovery"

                _apply_stored_state(deployment, state_manager)
                deployments.append(deployment)
        except (TypeError, AttributeError, NameError, SyntaxError):
            raise
        except Exception as err:
            logger.warning("Error building Groq deployments for %s: %s", acc_name, sanitize_secret(str(err), key))
            continue

    return deployments


def infer_nvidia_capabilities(model_id: str = NVIDIA_ACTUAL_MODEL) -> ModelCapabilities:
    """Infer normalized ModelCapabilities for an NVIDIA NIM model."""
    m_lower = model_id.lower()

    coding: Optional[bool] = True if ("nemotron" in m_lower or "llama" in m_lower or "code" in m_lower) else None
    reasoning: Optional[bool] = True if ("nemotron" in m_lower or "reason" in m_lower or "r1" in m_lower) else None
    vision: Optional[bool] = True if ("vision" in m_lower or "multimodal" in m_lower or "fuyu" in m_lower) else False
    tool_calling: Optional[bool] = True if ("nemotron" in m_lower or "llama" in m_lower) else None
    structured_output: Optional[bool] = True
    streaming: Optional[bool] = True
    context_window = 262144 if "nemotron-3.5" in m_lower else 131072

    return ModelCapabilities(
        coding=coding,
        reasoning=reasoning,
        vision=vision,
        tool_calling=tool_calling,
        structured_output=structured_output,
        streaming=streaming,
        context_window=context_window,
    )


def build_nvidia_deployments(
    keys: Optional[Dict[str, str]] = None,
    state_manager: Optional[StateManager] = None,
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
        _apply_stored_state(deployment, state_manager)
        deployments.append(deployment)

    return deployments


def infer_cohere_capabilities(model_id: str = COHERE_ACTUAL_MODEL) -> ModelCapabilities:
    """Infer normalized ModelCapabilities for a Cohere model."""
    m_lower = model_id.lower()

    coding: Optional[bool] = True if ("code" in m_lower or "command" in m_lower) else None
    reasoning: Optional[bool] = True if ("code" in m_lower or "command-r" in m_lower or "reason" in m_lower) else None
    vision: Optional[bool] = True if "vision" in m_lower else False
    tool_calling: Optional[bool] = True if ("command" in m_lower or "code" in m_lower) else None
    structured_output: Optional[bool] = True
    streaming: Optional[bool] = True
    context_window = 128000

    return ModelCapabilities(
        coding=coding,
        reasoning=reasoning,
        vision=vision,
        tool_calling=tool_calling,
        structured_output=structured_output,
        streaming=streaming,
        context_window=context_window,
    )


def build_cohere_deployments(
    keys: Optional[Dict[str, str]] = None,
    state_manager: Optional[StateManager] = None,
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
        _apply_stored_state(deployment, state_manager)
        deployments.append(deployment)

    return deployments


def build_gemini_deployments(
    keys: Optional[Dict[str, str]] = None,
    state_manager: Optional[StateManager] = None,
) -> List[Deployment]:
    """Discover and build Gemini deployments across configured Gemini accounts."""
    keys_to_use = keys if keys is not None else GEMINI_KEYS
    active_keys = {acc: k for acc, k in keys_to_use.items() if k and k != "YOUR_NEW_GOOGLE_API_KEY"}
    if not active_keys:
        return []

    try:
        from gemini import build_gemini_deployments as gemini_builder

        raw_deps = gemini_builder(active_keys, state_manager=state_manager)
        deployments: List[Deployment] = []
        for gdep in raw_deps:
            caps = getattr(gdep, "capabilities", None) or ModelCapabilities(
                coding=True,
                reasoning=True,
                vision=True,
                tool_calling=True,
                structured_output=True,
                streaming=True,
                context_window=1048576,
            )
            dep = Deployment(
                id=gdep.id,
                provider=MODEL_GROUP_GEMINI,
                account=gdep.account,
                model=gdep.model,
                logical_model=MODEL_GROUP_GEMINI,
                litellm_model=gdep.litellm_model,
                api_key=gdep.api_key,
                capabilities=caps,
                quota=QuotaInfo(quota_scope=QuotaScope.ACCOUNT_MODEL),
                is_stale=getattr(gdep, "is_stale", False),
                stale_reason=getattr(gdep, "stale_reason", None),
                circuit_state=getattr(gdep, "circuit_state", CircuitState.HEALTHY),
                state_reason=getattr(gdep, "state_reason", None),
                last_discovered_at=time.time(),
            )
            _apply_stored_state(dep, state_manager)
            deployments.append(dep)
        return deployments
    except (TypeError, AttributeError, NameError, SyntaxError):
        raise
    except Exception as err:
        logger.warning("Error building Gemini deployments: %s", sanitize_secret(str(err)))
        return []


def build_all_deployments(
    openrouter_keys: Optional[Dict[str, str]] = None,
    groq_keys: Optional[Dict[str, str]] = None,
    nvidia_keys: Optional[Dict[str, str]] = None,
    cohere_keys: Optional[Dict[str, str]] = None,
    gemini_keys: Optional[Dict[str, str]] = None,
    state_manager: Optional[StateManager] = None,
) -> List[Deployment]:
    """Build all deployments across all configured providers."""
    all_deployments: List[Deployment] = []

    # 1. OpenRouter
    all_deployments.extend(build_openrouter_deployments(openrouter_keys, state_manager=state_manager))

    # 2. Groq
    all_deployments.extend(build_groq_deployments(groq_keys, state_manager=state_manager))

    # 3. NVIDIA
    all_deployments.extend(build_nvidia_deployments(nvidia_keys, state_manager=state_manager))

    # 4. Cohere
    all_deployments.extend(build_cohere_deployments(cohere_keys, state_manager=state_manager))

    # 5. Gemini (if configured)
    all_deployments.extend(build_gemini_deployments(gemini_keys, state_manager=state_manager))

    return all_deployments
