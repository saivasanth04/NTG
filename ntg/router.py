"""Dynamic fallback router placing multiple OpenRouter and Gemini deployments in one deterministic global pool."""

import logging
import random
import time
from typing import Any, Dict, List, Optional

import litellm

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
    """Unified OpenRouter + Gemini Smart Router with explicit NTG-owned global fallback loop."""

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

        # Configure LiteLLM logging
        litellm.suppress_debug_info = True
        litellm.set_verbose = False
        logging.getLogger("LiteLLM").setLevel(logging.ERROR)

    def _handle_deployment_failure(self, deployment: Deployment, error: Exception) -> None:
        """Process failure for a specific deployment, classifying error and applying isolated cooldown."""
        deployment.attempts += 1
        deployment.failures += 1

        if deployment.provider == "gemini":
            info = classify_gemini_error(error)
        else:
            info = classify_openrouter_error(error)

        category = info.get("category", "UNKNOWN_ERROR")
        print_divider("DEPLOYMENT FAILED (REACTIVE ERROR)")
        print(f"Failed Deployment: {deployment.display_name} ({deployment.provider.title()})")
        print(f"Classification   : {category}")
        print(f"Error            : {error}")
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

    def ask(self, prompt: str) -> Optional[Any]:
        """Explicit NTG-owned fallback loop across all healthy peer deployments in the global pool."""
        now = time.time()
        # Build candidate pool of currently healthy deployments for this request
        candidate_pool = [dep for dep in self.deployments if dep.is_available(now)]

        if not candidate_pool:
            print_divider("NO AVAILABLE DEPLOYMENTS")
            print("All configured OpenRouter accounts and Gemini models are currently cooling down or missing API keys.")
            resets = [dep.remaining_cooldown for dep in self.deployments if dep.remaining_cooldown > 0]
            if resets:
                next_capacity = min(resets)
                print(f"Next available deployment capacity in: {next_capacity:.1f}s ({utc_string(now + next_capacity)})")
            print_account_status(self.deployments)
            return None

        # Shuffle candidate pool so traffic is fairly distributed among healthy peer deployments
        random.shuffle(candidate_pool)

        # Single NTG-controlled fallback loop over the candidate pool
        while candidate_pool:
            # Select one deployment and immediately remove it from candidate pool to guarantee no retry during current request
            deployment = candidate_pool.pop(0)

            try:
                # Attempt single request with the chosen deployment (LiteLLM num_retries=0)
                response = litellm.completion(
                    model=deployment.litellm_model,
                    api_key=deployment.api_key,
                    messages=[{"role": "user", "content": prompt}],
                    metadata={
                        "deployment_id": deployment.id,
                        "provider": deployment.provider,
                        "account_name": deployment.name,
                        "deployment_name": deployment.name,
                    },
                    num_retries=0,
                )

                # SUCCESS: Record state and return immediately
                deployment.attempts += 1
                deployment.successes += 1
                resp_model = getattr(response, "model", None)

                if hasattr(response, "choices") and response.choices:
                    print("\n" + str(response.choices[0].message.content))

                print_request_execution(deployment, resp_model)
                return response

            except Exception as error:
                # FAILURE: Reliably process failure for the specific deployment and block ONLY it
                self._handle_deployment_failure(deployment, error)
                # Continue loop to try another healthy deployment from candidate_pool

        # Total exhaustion of candidate pool
        print_divider("ALL DEPLOYMENTS EXHAUSTED")
        print(f"Unified logical model '{LITELLM_MODEL_NAME}' failed on all candidate deployments.")
        print_account_status(self.deployments)
        return None


# Maintain backwards compatibility
OpenRouterFallbackRouter = UnifiedNTGRouter