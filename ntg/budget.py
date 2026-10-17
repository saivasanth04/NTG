"""Candidate-specific packet estimates; no task rewriting or hidden truncation."""
import hashlib
import json
from functools import lru_cache


def _cached_encoding(name):
    """Use only an already loaded or hash-verified local tiktoken cache."""
    import os
    import re
    import tempfile
    import tiktoken
    from tiktoken import registry
    loaded = registry.ENCODINGS.get(name)
    if loaded is not None:
        return loaded
    # The supported constructors expose a single cache URL and expected SHA-256.
    # If a later tiktoken release changes this layout, use the byte estimate safely.
    from tiktoken_ext import openai_public
    constructor = getattr(openai_public, name, None)
    constants = getattr(getattr(constructor, "__code__", None), "co_consts", ())
    urls = [v for v in constants if isinstance(v, str) and v.startswith("https://")]
    hashes = [v for v in constants if isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v)]
    if len(urls) != 1 or len(hashes) != 1:
        return None
    from pathlib import Path
    root = os.environ.get("TIKTOKEN_CACHE_DIR", os.environ.get("DATA_GYM_CACHE_DIR", str(Path(tempfile.gettempdir()) / "data-gym-cache")))
    if not root:
        return None
    cache = Path(root) / hashlib.sha1(urls[0].encode()).hexdigest()
    try:
        if hashlib.sha256(cache.read_bytes()).hexdigest() != hashes[0]:
            return None
    except OSError:
        return None
    return tiktoken.get_encoding(name)


@lru_cache(maxsize=256)
def _count(tokenizer, payload):
    try:
        encoding = _cached_encoding(tokenizer)
    except (ImportError, ValueError, OSError):
        encoding = None
    if encoding is None:
        return len(payload.encode("utf-8")), "byte_upper_estimate"
    count = len(encoding.encode(payload, disallowed_special=()))
    return count, "model_estimate" if tokenizer == "o200k_base" else "fallback_estimate"


@lru_cache(maxsize=64)
def _local_count(path, modified, payload):
    from tokenizers import Tokenizer
    return len(Tokenizer.from_file(path).encode(payload).ids), "local_model_estimate"


def packet_budget(deployment, request, counter=None):
    from .transport import completion_kwargs
    wire = completion_kwargs(deployment, request, 1)
    messages = list(wire["messages"])
    extra = {k: wire[k] for k in ("tools", "tool_choice", "response_format") if k in wire}
    if extra:
        messages.append({"role": "system", "content": json.dumps(extra, ensure_ascii=False)})
    payload = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
    if counter:
        count, provenance = counter(deployment, messages)
    elif deployment.tokenizer_file:
        from pathlib import Path
        try:
            tokenizer = Path(deployment.tokenizer_file)
            count, provenance = _local_count(str(tokenizer), tokenizer.stat().st_mtime_ns, payload)
        except Exception:
            count, provenance = _count("cl100k_base", payload)
    else:
        count, provenance = _count("o200k_base" if deployment.account.provider == "openai" and deployment.model.startswith(("gpt-4o", "gpt-4.1", "gpt-5")) else "cl100k_base", payload)
    count = max(int(count), request.input_tokens or 0)
    margin = max(256, (count * (10 if provenance in ("model_estimate", "local_model_estimate") else 20) + 99) // 100)
    reason = None
    if deployment.max_input and count + margin > deployment.max_input:
        reason = "context_limit"
    elif deployment.context and count + margin + request.max_tokens > deployment.context:
        reason = "context_limit"
    elif deployment.max_output and request.max_tokens > deployment.max_output:
        reason = "output_limit"
    return dict(input_tokens=count, margin_tokens=margin, output_tokens=request.max_tokens,
                provenance=provenance, packet_sha256=hashlib.sha256(payload.encode()).hexdigest(), reason=reason)
