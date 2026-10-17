"""LiteLLM owns request translation, response normalization and provider errors."""
from copy import deepcopy
import json
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


def provider_response_format(provider, response_format):
    """Keep host schema strict; omit a known unsupported upstream grammar keyword."""
    if provider != "openrouter" or response_format.get("type") != "json_schema":
        return response_format
    wire = deepcopy(response_format)

    def visit(schema):
        if not isinstance(schema, dict):
            return
        # Some OpenRouter upstream grammar compilers reject uniqueItems outright.
        # RoutingService still validates every completion against the original schema.
        schema.pop("uniqueItems", None)
        for key in ("properties", "patternProperties", "$defs", "definitions", "dependentSchemas"):
            for child in schema.get(key, {}).values():
                visit(child)
        for key in ("items", "additionalProperties", "contains", "not", "if", "then", "else"):
            child = schema.get(key)
            if isinstance(child, list):
                for item in child:
                    visit(item)
            else:
                visit(child)
        for key in ("anyOf", "allOf", "oneOf", "prefixItems"):
            for child in schema.get(key, []):
                visit(child)

    visit(wire.get("json_schema", {}).get("schema"))
    return wire


def completion_kwargs(deployment, request, timeout):
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
            kwargs[key] = (provider_response_format(a.provider, value)
                           if key == "response_format" else value)
    fmt = request.response_format or {}
    native = "json_schema" if fmt.get("type") == "json_schema" else "json"
    if fmt.get("type") in ("json_schema", "json_object") and native not in deployment.capabilities:
        # Native schema support is optional; host validation remains mandatory.
        kwargs.pop("response_format", None)
        schema = fmt.get("json_schema", {}).get("schema")
        instruction = "Return only valid JSON, without Markdown or commentary."
        if schema is not None:
            instruction += " Match this JSON Schema exactly: " + json.dumps(schema)
        kwargs["messages"] = [{"role": "system", "content": instruction}, *deepcopy(request.messages)]
    if a.provider == "openrouter":
        kwargs["extra_body"] = {"provider": {"require_parameters": True}}
    return kwargs


class LiteLLMTransport:
    async def complete(self, deployment, request, timeout):
        kwargs = completion_kwargs(deployment, request, timeout)
        fmt = request.response_format or {}
        response = await litellm.acompletion(**kwargs)
        headers = (getattr(response, "_hidden_params", {}) or {}).get("additional_headers", {})
        raw = response.model_dump(exclude_none=True, warnings=False)
        # Some adapters return decoded structured content rather than JSON text.
        # Normalize its representation; the host still validates the original schema.
        if fmt.get("type") in ("json_schema", "json_object"):
            for choice in raw.get("choices", []):
                message = choice.get("message", {})
                if isinstance(message.get("content"), dict):
                    message["content"] = json.dumps(message["content"], ensure_ascii=False)
        # Do not forward arbitrary adapter internals or upstream debug fields.
        allowed = {"id", "object", "created", "model", "choices", "usage", "system_fingerprint", "service_tier"}
        return {k: v for k, v in raw.items() if k in allowed}, headers
