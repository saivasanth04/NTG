"""Request capability requirement extraction and eligibility errors."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set
import json

from ntg.core.models import ModelCapabilities


class NoEligibleDeploymentsError(RuntimeError):
    """Raised when no deployments satisfy strict request requirements or eligibility checks."""

    pass


def parse_capabilities(
    caps: Optional[ModelCapabilities | dict | list[str] | str] = None,
) -> Optional[ModelCapabilities]:
    """Normalize capability parameter into a ModelCapabilities object."""
    if caps is None:
        return None
    if isinstance(caps, ModelCapabilities):
        return ModelCapabilities(**caps.to_dict())
    if isinstance(caps, dict):
        return ModelCapabilities(**caps)
    if isinstance(caps, list):
        valid_fields = {
            "coding",
            "reasoning",
            "vision",
            "tool_calling",
            "structured_output",
            "streaming",
            "audio_input",
            "audio_output",
        }
        unknown = set(caps) - valid_fields
        if unknown:
            raise ValueError(f"Unknown capabilities: {sorted(unknown)}")
        return ModelCapabilities(**{c: True for c in caps})
    if isinstance(caps, str):
        c_clean = caps.strip().lower()
        if c_clean in ("coding", "code"):
            return ModelCapabilities(coding=True)
        if c_clean in ("reasoning", "reason"):
            return ModelCapabilities(reasoning=True)
        if c_clean in ("vision", "image"):
            return ModelCapabilities(vision=True)
        if c_clean in ("tool_calling", "tools", "tool"):
            return ModelCapabilities(tool_calling=True)
        if c_clean in ("structured_output", "structured", "json"):
            return ModelCapabilities(structured_output=True)
        if c_clean in ("streaming", "stream"):
            return ModelCapabilities(streaming=True)
        if c_clean in ("audio_input", "audio_output"):
            return ModelCapabilities(**{c_clean: True})
    raise ValueError("Invalid capabilities parameter")


class RequestRequirements:
    """Extracted capability and constraint requirements from a user request."""

    def __init__(
        self,
        capabilities: Optional[ModelCapabilities] = None,
        coding_preferred: bool = False,
        strict_requirements: Optional[Set[str]] = None,
    ):
        self.capabilities = capabilities or ModelCapabilities()
        self.coding_preferred = coding_preferred
        self.strict_requirements = strict_requirements or set()

    def describe(self) -> str:
        """Human-readable summary of active requirements."""
        reqs = []
        if self.capabilities.tool_calling is True:
            reqs.append("tool_calling")
        if self.capabilities.structured_output is True:
            reqs.append("structured_output")
        if self.capabilities.vision is True:
            reqs.append("vision")
        if self.capabilities.streaming is True:
            reqs.append("streaming")
        if self.capabilities.audio_input is True:
            reqs.append("audio_input")
        if self.capabilities.audio_output is True:
            reqs.append("audio_output")
        if self.capabilities.reasoning is True:
            reqs.append("reasoning")
        if self.capabilities.coding is True:
            reqs.append("coding")
        elif self.coding_preferred:
            reqs.append("coding (preferred)")
        if self.capabilities.context_window > 4096:
            reqs.append(f"context_window>={self.capabilities.context_window:,}")
        return ", ".join(reqs) if reqs else "none"


def extract_request_requirements(
    messages: Optional[List[Dict[str, Any]]] = None,
    prompt: Optional[str] = None,
    model: Optional[str] = None,
    capabilities: Optional[ModelCapabilities | dict | list[str] | str] = None,
    min_context: Optional[int] = None,
    **kwargs: Any,
) -> RequestRequirements:
    """Extract deterministic capability requirements from OpenAI/LiteLLM request arguments."""
    req_caps = ModelCapabilities()
    strict_reqs: Set[str] = set()
    coding_preferred = False
    if model in ("vision", "tool_calling", "structured_output", "streaming", "audio_input", "audio_output"):
        setattr(req_caps, model, True)
        strict_reqs.add(model)

    # 1. Tools / functions requested -> require tool_calling
    tools = kwargs.get("tools")
    functions = kwargs.get("functions")
    tool_choice = kwargs.get("tool_choice")
    has_tools = bool(
        tools or functions or (tool_choice is not None and tool_choice != "none")
    )
    if not has_tools and messages and isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict):
                if msg.get("role") in ("tool", "function") or msg.get("tool_calls") or msg.get("function_call"):
                    has_tools = True
                    break
    if has_tools:
        req_caps.tool_calling = True
        strict_reqs.add("tool_calling")

    # 2. Structured output requested -> require structured_output
    response_format = kwargs.get("response_format")
    schema = kwargs.get("schema")
    fmt = kwargs.get("format")
    has_structured_response_format = bool(
        response_format is not None
        and not (
            (isinstance(response_format, dict) and response_format.get("type") == "text")
            or response_format == "text"
        )
    )
    if (
        has_structured_response_format
        or schema is not None
        or fmt in ("json", "json_object", "json_schema")
        or kwargs.get("json_schema") is not None
        or kwargs.get("guided_json") is not None
    ):
        req_caps.structured_output = True
        strict_reqs.add("structured_output")

    # 3. Image / multimodal input -> require vision
    has_image = False
    if kwargs.get("images"):
        has_image = True
    if messages and isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        part_type = str(part.get("type", "")).lower()
                        if part_type in ("image_url", "image", "input_image") or "image_url" in part:
                            has_image = True
                        if part_type in ("input_audio", "audio_url"):
                            req_caps.audio_input = True
                            strict_reqs.add("audio_input")
                        if part_type in ("file", "input_file", "video", "video_url"):
                            raise ValueError(f"Unsupported input modality: {part_type}")

    if has_image:
        req_caps.vision = True
        strict_reqs.add("vision")
    if "audio" in (kwargs.get("modalities") or []) or kwargs.get("audio") is not None:
        req_caps.audio_output = True
        strict_reqs.add("audio_output")

    # 4. Streaming requested -> require streaming
    if kwargs.get("stream") is True or kwargs.get("streaming") is True:
        req_caps.streaming = True
        strict_reqs.add("streaming")

    # 5. Reasoning-specific routing
    if (
        model == "reasoning"
        or kwargs.get("reasoning") is True
        or isinstance(kwargs.get("reasoning"), dict) and bool(kwargs["reasoning"]) and kwargs["reasoning"].get("enabled") is not False
        or kwargs.get("reasoning_effort") not in (None, "none", False)
    ):
        req_caps.reasoning = True
        strict_reqs.add("reasoning")

    # 6. Coding-oriented requests
    if model == "coding" or kwargs.get("coding") is True or kwargs.get("task") in ("coding", "code"):
        req_caps.coding = True
        strict_reqs.add("coding")
    else:
        text_to_check = prompt or ""
        if not text_to_check and messages and isinstance(messages, list):
            for m in messages:
                c = m.get("content") if isinstance(m, dict) else ""
                if isinstance(c, str):
                    text_to_check += " " + c
        if "```" in text_to_check or "def " in text_to_check or "class " in text_to_check:
            coding_preferred = True

    # 7. Context window requirement (explicit min_context and estimated prompt + completion tokens)
    # Include protocol overhead, function arguments and schemas. UTF-8 bytes are a
    # conservative dependency-free upper bound for text BPE tokens, not chars/4.
    # Image payload bytes are excluded; reserve a documented per-image allowance.
    def token_bound(value: Any) -> int:
        if isinstance(value, str):
            return len(value.encode("utf-8"))
        if isinstance(value, list):
            return sum(token_bound(item) for item in value)
        if isinstance(value, dict):
            if value.get("type") in ("image", "image_url", "input_image"):
                return 4096
            if value.get("type") in ("input_audio", "audio_url"):
                return 4096
            return sum(token_bound(k) + token_bound(v) for k, v in value.items()) + 8
        return len(json.dumps(value, default=str))

    extra_tokens = 0
    for name in ("max_tokens", "max_completion_tokens"):
        value = kwargs.get(name)
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
            extra_tokens = max(extra_tokens, value)
    if min_context is not None and (isinstance(min_context, bool) or not isinstance(min_context, int) or min_context < 0):
        raise ValueError("min_context must be a nonnegative integer")
    estimated_tokens = token_bound(messages or prompt or "") + extra_tokens
    for name in ("tools", "functions", "response_format", "schema", "json_schema", "guided_json"):
        if kwargs.get(name) is not None:
            estimated_tokens += token_bound(kwargs[name])

    effective_min_context = max(min_context or 0, estimated_tokens)
    if effective_min_context > 0:
        req_caps.context_window = effective_min_context
        if (min_context and min_context > 0) or effective_min_context > 4096:
            strict_reqs.add("context_window")

    # 8. Explicit capabilities argument
    if capabilities is not None:
        explicit_parsed = parse_capabilities(capabilities)
        if explicit_parsed:
            if explicit_parsed.coding is True:
                req_caps.coding = True
                strict_reqs.add("coding")
            if explicit_parsed.reasoning is True:
                req_caps.reasoning = True
                strict_reqs.add("reasoning")
            if explicit_parsed.vision is True:
                req_caps.vision = True
                strict_reqs.add("vision")
            if explicit_parsed.tool_calling is True:
                req_caps.tool_calling = True
                strict_reqs.add("tool_calling")
            if explicit_parsed.structured_output is True:
                req_caps.structured_output = True
                strict_reqs.add("structured_output")
            if explicit_parsed.streaming is True:
                req_caps.streaming = True
                strict_reqs.add("streaming")
            for name in ("audio_input", "audio_output"):
                if getattr(explicit_parsed, name) is True:
                    setattr(req_caps, name, True)
                    strict_reqs.add(name)
            if explicit_parsed.context_window > req_caps.context_window:
                req_caps.context_window = explicit_parsed.context_window
                strict_reqs.add("context_window")

    return RequestRequirements(
        capabilities=req_caps,
        coding_preferred=coding_preferred,
        strict_requirements=strict_reqs,
    )
