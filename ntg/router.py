"""Dynamic fallback router placing multiple OpenRouter and Gemini deployments behind one unified LiteLLM logical model."""

import logging
import time
from typing import Any, Dict, List, Optional

import litellm
from litellm import Router

from ntg.config import DEFAULT_KEYS, GEMINI_API_KEY, LITELLM_MODEL_NAME
from ntg.diagnostics import (
    print_account_status,
    print_divider,
    print_rate_limit_details,
    print_request_execution,
)
from ntg.exceptions import (
    CATEGORY_AUTH_ERROR,
    CATEGORY_DAILY_QUOTA,
    CATEGORY_PROVIDER_LIMIT,
    CATEGORY_SERVER_ERROR,
    classify_gemini_error,
    classify_openrouter_error,
)
from ntg.models import Account, Deployment, utc_string


class UnifiedNTGRouter:
    """Unified OpenRouter + Gemini Smart Router reacting to real provider errors."""

    def __init__(
        self,
        accounts: Optional[List[Deployment]] = None,
        gemini_api_key: Optional[str] = None,
    ):
        self.deployments: List[Deployment] = []

        # 1. Add OpenRouter account deployments
        if accounts is not None:
            openrouter_deps = accounts
        else:
            openrouter_deps = [
                Account(name=name, api_key=key, order=i + 1)
                for i, (name, key) in enumerate(DEFAULT_KEYS.items())
            ]
            while len(openrouter_deps) < 5:
                idx = len(openrouter_deps) + 1
                openrouter_deps.append(
                    Account(
                        name=f"account_{idx}",
                        api_key="",
                        order=idx,
                        available=False,
                    )
                )

        self.deployments.extend(openrouter_deps)

        # 2. Discover and add Gemini deployments dynamically
        key = gemini_api_key or GEMINI_API_KEY
        if key and key != "YOUR_NEW_GOOGLE_API_KEY":
            try:
                from gemini import build_gemini_deployments
                gemini_deps = build_gemini_deployments(key)
                for i, gdep in enumerate(gemini_deps):
                    gdep.order = len(openrouter_deps) + i + 1
                self.deployments.extend(gemini_deps)
            except Exception as err:
                print(f"[Router Warning] Could not dynamically discover Gemini models: {err}")

        self.deployment_map: Dict[str, Deployment] = {dep.id: dep for dep in self.deployments}
        self._last_success_deployment: Optional[Deployment] = None

        # Configure LiteLLM logging
        litellm.suppress_debug_info = True
        litellm.set_verbose = False
        logging.getLogger("LiteLLM").setLevel(logging.ERROR)

        # Register LiteLLM callbacks
        litellm.failure_callback = [self._handle_litellm_failure]
        litellm.success_callback = [self._handle_litellm_success]

        # Initial LiteLLM Router setup
        valid_deployments = [dep.to_deployment() for dep in self.deployments if dep.api_key and dep.api_key.strip()]
        self.router = Router(
            model_list=valid_deployments if valid_deployments else [],
            routing_strategy="simple-shuffle",
            allowed_fails=0,
            cooldown_time=60,
        )

    def _get_deployment_from_kwargs(self, kwargs: Dict[str, Any]) -> Optional[Deployment]:
        """Extracts Deployment instance from callback kwargs metadata or model_info."""
        lp = kwargs.get("litellm_params", {})
        if not isinstance(lp, dict):
            lp = {}

        meta = lp.get("metadata", {})
        if not isinstance(meta, dict):
            meta = {}

        mi = meta.get("model_info", {})
        if not mi:
            mi = lp.get("model_info", {})
        if not isinstance(mi, dict):
            mi = {}

        dep_id = mi.get("id") or meta.get("deployment_id")
        if dep_id and dep_id in self.deployment_map:
            return self.deployment_map[dep_id]

        acc_name = mi.get("account") or meta.get("account_name")
        if acc_name:
            target_id = f"openrouter-{acc_name}"
            if target_id in self.deployment_map:
                return self.deployment_map[target_id]

        api_key = lp.get("api_key")
        if api_key:
            for dep in self.deployments:
                if dep.api_key == api_key:
                    return dep

        return None

    def _handle_litellm_failure(
        self, kwargs: Dict[str, Any], exception: Exception, start_time: float, end_time: float
    ) -> None:
        """Reactive callback executed when a deployment fails during LiteLLM router execution."""
        deployment = self._get_deployment_from_kwargs(kwargs)
        exc = kwargs.get("exception") or exception

        if deployment:
            deployment.attempts += 1
            deployment.failures += 1

            if deployment.provider == "gemini":
                info = classify_gemini_error(exc) if exc else {}
            else:
                info = classify_openrouter_error(exc) if exc else {}

            category = info.get("category", "UNKNOWN_ERROR")
            print_divider("DEPLOYMENT FAILED (REACTIVE ERROR)")
            print(f"Failed Deployment: {deployment.display_name} ({deployment.provider.title()})")
            print(f"Classification   : {category}")
            print(f"Error            : {exc}")
            if info:
                print_rate_limit_details(info)

            # Apply cooldown/block ONLY to the failed deployment
            if category == CATEGORY_DAILY_QUOTA:
                deployment.rate_limits += 1
                reset_ts = info.get("reset_timestamp")
                print(f"\n>>> REAL ERROR REACT: Daily quota exhausted on {deployment.display_name}. Blocking deployment.")
                deployment.block("Daily quota exhausted", until=reset_ts, cooldown=86400.0)

            elif category == CATEGORY_AUTH_ERROR:
                print(f"\n>>> REAL ERROR REACT: Authentication failure on {deployment.display_name}. Cooling down 1 hour.")
                deployment.block("Authentication failure", cooldown=3600.0)

            elif category == CATEGORY_SERVER_ERROR:
                print(f"\n>>> REAL ERROR REACT: Upstream server error on {deployment.display_name}. Cooling down 30 seconds.")
                deployment.block("Upstream server error", cooldown=30.0)

            elif category == CATEGORY_PROVIDER_LIMIT:
                deployment.rate_limits += 1
                print(f"\n>>> REAL ERROR REACT: Provider rate limit (429) on {deployment.display_name}. Cooling down 60 seconds.")
                deployment.block("Provider rate limit (429)", cooldown=60.0)

            else:
                deployment.rate_limits += 1
                print(f"\n>>> REAL ERROR REACT: Request failure on {deployment.display_name}. Cooling down 60 seconds.")
                deployment.block(f"Request failure: {category}", cooldown=60.0)

    def _handle_litellm_success(
        self, kwargs: Dict[str, Any], response: Any, start_time: float, end_time: float
    ) -> None:
        """Callback executed when a deployment succeeds."""
        deployment = self._get_deployment_from_kwargs(kwargs)
        if deployment:
            deployment.attempts += 1
            deployment.successes += 1
            self._last_success_deployment = deployment

    def ask(self, prompt: str) -> Optional[Any]:
        """Dispatches request via unified LiteLLM router across all healthy OpenRouter & Gemini deployments."""
        now = time.time()
        available_deployments = [dep for dep in self.deployments if dep.is_available(now)]

        if not available_deployments:
            print_divider("NO AVAILABLE DEPLOYMENTS")
            print("All configured OpenRouter accounts and Gemini models are currently cooling down or missing API keys.")
            resets = [dep.remaining_cooldown for dep in self.deployments if dep.remaining_cooldown > 0]
            if resets:
                next_capacity = min(resets)
                print(f"Next available deployment capacity in: {next_capacity:.1f}s ({utc_string(now + next_capacity)})")
            print_account_status(self.deployments)
            return None

        # Update LiteLLM Router model list with currently healthy deployments
        active_litellm_deps = [dep.to_deployment() for dep in available_deployments]
        self.router.set_model_list(active_litellm_deps)

        num_retries = max(0, len(active_litellm_deps) - 1)
        self._last_success_deployment = None

        try:
            response = self.router.completion(
                model=LITELLM_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                num_retries=num_retries,
            )

            # Identify fulfilling deployment
            fulfilling_dep = self._last_success_deployment
            resp_model = getattr(response, "model", None)

            # Output response content
            if hasattr(response, "choices") and response.choices:
                print("\n" + str(response.choices[0].message.content))

            if fulfilling_dep:
                print_request_execution(fulfilling_dep, resp_model)

            return response

        except Exception as error:
            print_divider("ALL DEPLOYMENTS EXHAUSTED")
            print(f"Unified logical model '{LITELLM_MODEL_NAME}' failed on all available deployments.")
            print(f"Final exception: {error}")
            print_account_status(self.deployments)
            return None


# Maintain backwards compatibility
OpenRouterFallbackRouter = UnifiedNTGRouter