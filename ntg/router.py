"""LiteLLM-based multi-provider smart router."""

import logging
import time
from typing import Any, Dict, List, Optional

import litellm
from litellm import Router

from ntg.config import (
    DEFAULT_LOGICAL_MODEL,
    MODEL_GROUP_COHERE,
    MODEL_GROUP_GEMINI,
    MODEL_GROUP_GROQ,
    MODEL_GROUP_NVIDIA,
    MODEL_GROUP_OPENROUTER,
)
from ntg.diagnostics import (
    print_account_status,
    print_divider,
    print_rate_limit_details,
    print_request_execution,
)
from ntg.discovery import build_all_deployments
from ntg.exceptions import (
    CATEGORY_AUTH_ERROR,
    CATEGORY_DAILY_QUOTA,
    CATEGORY_PROVIDER_LIMIT,
    CATEGORY_SERVER_ERROR,
    classify_provider_error,
)
from ntg.models import Deployment, utc_string


class NTGLiteLLMRouter(Router):
    """LiteLLM Router subclass that captures deployment execution and isolated failures."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_selected_deployment: Optional[Dict[str, Any]] = None
        self._failure_listener = None

    def _update_kwargs_with_deployment(
        self,
        deployment: dict,
        kwargs: dict,
        function_name: Optional[str] = None,
    ) -> None:
        super()._update_kwargs_with_deployment(deployment, kwargs, function_name)
        self._last_selected_deployment = deployment

    def _completion(self, model: str, messages: list[dict[str, str]], **kwargs):
        try:
            return super()._completion(model, messages, **kwargs)
        except Exception as exc:
            if self._last_selected_deployment and self._failure_listener:
                dep_info = self._last_selected_deployment.get("model_info", {})
                dep_id = dep_info.get("id")
                if dep_id:
                    self._failure_listener(dep_id, exc)
            raise


class UnifiedNTGRouter:
    """Unified multi-provider router delegating routing and load balancing directly to LiteLLM."""

    def __init__(
        self,
        accounts: Optional[List[Deployment]] = None,
        default_model: Optional[str] = None,
        openrouter_keys: Optional[Dict[str, str]] = None,
        groq_keys: Optional[Dict[str, str]] = None,
        nvidia_keys: Optional[Dict[str, str]] = None,
        cohere_keys: Optional[Dict[str, str]] = None,
        gemini_keys: Optional[Dict[str, str]] = None,
        gemini_api_key: Optional[str] = None,
    ):
        # Configure logging to suppress noisy LiteLLM model registration notices
        litellm.suppress_debug_info = True
        litellm.set_verbose = False
        logging.getLogger("LiteLLM").setLevel(logging.ERROR)
        logging.getLogger("LiteLLM Router").setLevel(logging.ERROR)

        # 1. Build all deployments across providers
        if accounts is not None:
            self.deployments: List[Deployment] = list(accounts)
        else:
            g_keys = gemini_keys
            if g_keys is None and gemini_api_key and gemini_api_key != "YOUR_NEW_GOOGLE_API_KEY":
                g_keys = {"account_1": gemini_api_key}

            self.deployments = build_all_deployments(
                openrouter_keys=openrouter_keys,
                groq_keys=groq_keys,
                nvidia_keys=nvidia_keys,
                cohere_keys=cohere_keys,
                gemini_keys=g_keys,
            )

        self.deployment_map: Dict[str, Deployment] = {dep.id: dep for dep in self.deployments}

        # 2. Collect valid deployments for the LiteLLM Router deployment pool
        active_deployments = [
            dep for dep in self.deployments
            if dep.api_key and dep.api_key.strip() and dep.available
        ]
        model_list = [dep.to_litellm_dict() for dep in active_deployments]

        # 3. Discover available logical groups and configure fallbacks
        logical_groups = list(dict.fromkeys(dep.logical_model for dep in active_deployments))
        fallbacks: List[Dict[str, List[str]]] = []
        for group in logical_groups:
            other_groups = [g for g in logical_groups if g != group]
            if other_groups:
                fallbacks.append({group: other_groups})

        # 4. Resolve default model and model aliases
        if default_model:
            self.default_model = default_model
        elif logical_groups:
            self.default_model = logical_groups[0]
        else:
            self.default_model = DEFAULT_LOGICAL_MODEL

        model_group_alias: Dict[str, str] = {
            "auto": self.default_model,
            "ntg-auto": self.default_model,
        }

        # 5. Instantiate LiteLLM Router
        num_retries = max(1, len(active_deployments) - 1) if active_deployments else 1

        self.router = NTGLiteLLMRouter(
            model_list=model_list,
            routing_strategy="simple-shuffle",
            cooldown_time=60.0,
            allowed_fails=1,
            num_retries=num_retries,
            fallbacks=fallbacks,
            model_group_alias=model_group_alias,
        )
        self.router._failure_listener = self._handle_deployment_failure

    def _handle_deployment_failure(self, dep_id: str, error: Exception) -> None:
        """Isolated reactive error handling when LiteLLM reports a deployment failure."""
        deployment = self.deployment_map.get(dep_id)
        provider = deployment.provider if deployment else ""
        info = classify_provider_error(error, provider)
        category = info.get("category", "UNKNOWN_ERROR")

        # Determine cooldown time based on error classification
        if category == CATEGORY_DAILY_QUOTA:
            reset_ts = info.get("reset_timestamp")
            cooldown_secs = max(0.0, reset_ts - time.time()) if reset_ts else 86400.0
            reason = "Daily quota exhausted"
        elif category == CATEGORY_AUTH_ERROR:
            cooldown_secs = 3600.0
            reason = "Authentication failure"
        elif category == CATEGORY_SERVER_ERROR:
            cooldown_secs = 30.0
            reason = "Upstream server error"
        elif category == CATEGORY_PROVIDER_LIMIT:
            cooldown_secs = 60.0
            reason = "Provider rate limit (429)"
        else:
            cooldown_secs = 60.0
            reason = f"Request failure: {category}"

        # Place the failing deployment into LiteLLM's cooldown cache
        self.router.cooldown_cache.add_deployment_to_cooldown(
            model_id=dep_id,
            original_exception=error,
            exception_status=getattr(error, "status_code", 500) or 500,
            cooldown_time=cooldown_secs,
        )

        # Update deployment health and diagnostic tracking in NTG
        if deployment:
            deployment.attempts += 1
            deployment.failures += 1
            if category in (CATEGORY_PROVIDER_LIMIT, CATEGORY_DAILY_QUOTA):
                deployment.rate_limits += 1
            deployment.block(reason, cooldown=cooldown_secs)

            print_divider("DEPLOYMENT FAILED (REACTIVE ERROR)")
            print(f"Failed Deployment: {deployment.display_name}")
            print(f"Classification   : {category}")
            print(f"Error            : {error}")
            if info:
                print_rate_limit_details(info)

    def ask(self, prompt: str, model: Optional[str] = None, **kwargs) -> Optional[Any]:
        """Route request through LiteLLM Router using the selected logical model group."""
        target_model = model or self.default_model

        # Refresh cooldown status on all deployments
        now = time.time()
        for dep in self.deployments:
            dep.refresh(now)

        # Check if there are active deployments
        healthy_active = [
            d for d in self.deployments
            if d.api_key and d.api_key.strip() and d.is_available(now)
        ]
        if not healthy_active:
            print_divider("NO AVAILABLE DEPLOYMENTS")
            print("All configured deployments are currently cooling down or missing API keys.")
            resets = [dep.remaining_cooldown for dep in self.deployments if dep.remaining_cooldown > 0]
            if resets:
                next_capacity = min(resets)
                print(f"Next available deployment capacity in: {next_capacity:.1f}s ({utc_string(now + next_capacity)})")
            print_account_status(self.deployments)
            return None

        try:
            # Delegate routing, load-balancing, and failover completely to LiteLLM Router
            response = self.router.completion(
                model=target_model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )

            # Record success on the deployment that handled the request
            selected_info = (self.router._last_selected_deployment or {}).get("model_info", {})
            dep_id = selected_info.get("id")
            deployment = self.deployment_map.get(dep_id)

            if deployment:
                deployment.attempts += 1
                deployment.successes += 1

            if hasattr(response, "choices") and response.choices:
                print("\n" + str(response.choices[0].message.content))

            if deployment:
                print_request_execution(deployment, getattr(response, "model", None))

            return response

        except Exception as error:
            print_divider("ALL DEPLOYMENTS EXHAUSTED")
            print(f"Request failed across healthy deployments for model group '{target_model}'.")
            print(f"Final error: {error}")
            print_account_status(self.deployments)
            return None


# Maintain backwards compatibility
OpenRouterFallbackRouter = UnifiedNTGRouter