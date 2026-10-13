"""LiteLLM transport plus a read-only OpenRouter price check."""
import logging
import os
import json
import httpx
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
import litellm
from .config import PREFIX

litellm.suppress_debug_info = True
litellm.set_verbose = False
litellm.drop_params = False
for name in ("LiteLLM", "LiteLLM Router", "httpx", "httpcore"):
    logging.getLogger(name).setLevel(logging.CRITICAL)

async def send(account, model, messages, timeout, **params):
    kwargs = dict(model=f"{PREFIX[account.provider]}/{model}", api_key=account.key,
                  messages=messages, timeout=timeout, num_retries=0, max_retries=0,
                  drop_params=False, **params)
    if account.provider == "cloudflare":
        kwargs["api_base"] = f"https://api.cloudflare.com/client/v4/accounts/{account.account_id}/ai/v1"
        # This LiteLLM version ignores injected clients in its Cloudflare path.
        # Use its OpenAI-compatible adapter with a native SDK client instead.
        kwargs["model"] = f"openai/{model}"
    if account.provider == "openrouter":
        # Native metadata adapter: explicit variant plus zero prices, no inference probe.
        async with httpx.AsyncClient(timeout=timeout) as client:
            reply = await client.get("https://openrouter.ai/api/v1/models")
            reply.raise_for_status()
            row = next((r for r in reply.json()["data"] if r["id"] == model), None)
            prices = row.get("pricing", {}) if row else {}
            if not prices or any(float(v) != 0 for v in prices.values()):
                raise ValueError("Free model pricing is unknown or nonzero")
        kwargs["extra_body"] = {"provider": {"allow_fallbacks": False, "require_parameters": True,
                                             "max_price": {"prompt": 0, "completion": 0}}}
    # Cloudflare's LiteLLM exception drops headers. Capture the real HTTP error
    # without a second request, and attach it to the normalized exception.
    if account.provider != "cloudflare":
        return await litellm.acompletion(**kwargs)
    from openai import AsyncOpenAI
    captured = []
    async def capture(reply):
        if reply.status_code >= 400:
            await reply.aread()
            captured.append(reply)
    http_client = httpx.AsyncClient(timeout=timeout, event_hooks={"response": [capture]})
    client = AsyncOpenAI(api_key=account.key, base_url=kwargs["api_base"],
                         max_retries=0, http_client=http_client)
    try:
        result = await litellm.acompletion(client=client, **kwargs)
    except BaseException as error:
        if captured and isinstance(error, Exception):
            error.response = captured[-1]
            error.status_code = captured[-1].status_code
            try:
                error.body = captured[-1].json()
            except ValueError:
                pass
        await client.close()
        raise
    if params.get("stream"):
        return ManagedStream(result, client.close)
    await client.close()
    return result

class ManagedStream:
    """Close even when never iterated; avoids leaked in-flight reservations."""
    def __init__(self, upstream, close, first=None):
        self.upstream, self.close, self.first = upstream, close, first
        self.closed = False
    def __aiter__(self):
        return self
    async def __anext__(self):
        if self.closed:
            raise StopAsyncIteration
        if self.first is not None:
            first, self.first = self.first, None
            return first
        try:
            return await anext(self.upstream)
        except BaseException:
            await self.aclose()
            raise
    async def aclose(self):
        if not self.closed:
            self.closed = True
            try:
                if hasattr(self.upstream, "aclose"):
                    await self.upstream.aclose()
            finally:
                await self.close()

def compatible(provider, model, messages, params, input_tokens=None, metadata=None):
    full = f"{PREFIX[provider]}/{model}"
    supported = metadata["parameters"] if metadata is not None else (litellm.get_supported_openai_params(model=full) or [])
    if metadata is not None:
        supported = [*supported, "stream"]
    if any(k not in supported for k in params):
        return False
    info = litellm.model_cost.get(full, litellm.model_cost.get(model, {}))
    if metadata is not None:
        info = metadata
    if provider == "cloudflare" and model == "@cf/meta/llama-3.1-8b-instruct-fp8":
        # Documented context and parameters; conservative output policy ceiling,
        # not a rate limit or an assertion about free billing eligibility.
        info = {"max_input_tokens": 32000, "max_output_tokens": 4096}
        supported = ["temperature", "top_p", "max_tokens", "stream", "seed",
                     "frequency_penalty", "presence_penalty"]
        if any(k not in supported for k in params):
            return False
    if params.get("tools") and info.get("supports_function_calling") is not True:
        return False
    if params.get("response_format") and info.get("supports_response_schema") is not True:
        return False
    # Multimodal requires a separate verified adapter; never silently convert content.
    if any(not isinstance(m.get("content", ""), (str, type(None))) for m in messages):
        return False
    output = params.get("max_tokens", params.get("max_completion_tokens", 256))
    maximum = info.get("max_output_tokens")
    context = info.get("max_input_tokens") or info.get("max_tokens")
    if not maximum or not context or output > maximum:
        return False
    # Conservative local estimate avoids downloading provider tokenizers. JSON
    # bytes include role/tool overhead; the provider still validates exact usage.
    estimate = len(json.dumps({"messages": messages, "tools": params.get("tools")},
                              ensure_ascii=False).encode("utf-8")) + 32 * len(messages)
    # Tokenizer estimates do not establish exact provider context usage; caller may supply a stricter count.
    return max(estimate, input_tokens or 0) + output <= context

async def openrouter_metadata(timeout):
    """Request-time model capabilities/pricing, never quota discovery."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get("https://openrouter.ai/api/v1/models")
        response.raise_for_status()
        result = {}
        for row in response.json()["data"]:
            prices = row.get("pricing", {})
            if (row["id"] != "openrouter/free" and not row["id"].endswith(":free")) or not prices or any(float(v) != 0 for v in prices.values()):
                continue
            parameters = row.get("supported_parameters", [])
            result[row["id"]] = {"max_input_tokens": row.get("context_length"),
                # The free router advertises no fixed output limit. Bound our
                # requests by policy; the selected provider validates its limit.
                "max_output_tokens": (4096 if row["id"] == "openrouter/free" else
                    row.get("top_provider", {}).get("max_completion_tokens")),
                "supports_function_calling": "tools" in parameters,
                "supports_response_schema": "structured_outputs" in parameters,
                "parameters": parameters}
        return result
