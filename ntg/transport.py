"""LiteLLM owns request translation, response normalization and provider errors."""
import logging
import os

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
os.environ.setdefault("LITELLM_TELEMETRY", "False")
import litellm

litellm.suppress_debug_info = True
litellm.set_verbose = False
litellm.turn_off_message_logging = True
litellm.drop_params = False
litellm.telemetry = False
for name in ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy", "httpx", "httpcore", "openai"):
    logging.getLogger(name).setLevel(logging.CRITICAL)


class LiteLLMTransport:
    async def complete(self, deployment, request, timeout):
        a = deployment.account
        kwargs = dict(model=f"{a.provider}/{deployment.model}", api_key=a.key,
                      messages=request.messages, max_tokens=request.max_tokens,
                      timeout=timeout, num_retries=0, caching=False)
        if a.provider == "cohere":
            kwargs["model"] = f"cohere_chat/{deployment.model}"
        if a.provider == "cloudflare":
            # OpenAI-compatible Workers AI endpoint avoids global account-id environment state.
            kwargs["model"] = f"openai/{deployment.model}"
            kwargs["api_base"] = f"https://api.cloudflare.com/client/v4/accounts/{a.account_id}/ai/v1"
        for key in ("temperature", "tools", "tool_choice", "response_format"):
            value = getattr(request, key)
            if value is not None:
                kwargs[key] = value
        if a.provider == "openrouter":
            # Enforce parameter support and zero-priced upstream providers as well as :free ID.
            kwargs["extra_body"] = {"provider": {"require_parameters": True,
                "max_price": {"prompt": 0, "completion": 0}}}
        response = await litellm.acompletion(**kwargs)
        headers = (getattr(response, "_hidden_params", {}) or {}).get("additional_headers", {})
        raw = response.model_dump(exclude_none=True)
        # Do not forward arbitrary adapter internals or upstream debug fields.
        allowed = {"id", "object", "created", "model", "choices", "usage", "system_fingerprint", "service_tier"}
        return {k: v for k, v in raw.items() if k in allowed}, headers
