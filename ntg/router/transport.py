"""LiteLLM transport; provider APIs enforce model access and billing."""
import logging
import os
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
        if model == "@cf/qwen/qwen3.8-27b" and "max_tokens" in kwargs:
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
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

def compatible(provider, model, messages, params, input_tokens=None):
    """Keep the text-only interface; let APIs validate models and limits.

    Bundled model catalogs can lag behind providers and cannot establish
    account access. Unknown models and parameters are sent unchanged.
    """
    return all(isinstance(m.get("content", ""), (str, type(None))) for m in messages)
