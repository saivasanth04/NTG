"""Request capability requirement extraction and eligibility errors."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

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
        return caps
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
        }
        return ModelCapabilities(**{c: True for c in caps if c in valid_fields})
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
    return None


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

    # 1. Tools / functions requested -> require tool_calling
    tools = kwargs.get("tools")
    functions = kwargs.get("functions")
    tool_choice = kwargs.get("tool_choice")
    has_tools = bool(
        (tools and len(tools) > 0) or (functions and len(functions) > 0) or tool_choice
    )
    if not has_tools and messages and isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict):
                if msg.get("role") in ("tool", "function") or msg.get("tool_calls"):
                    has_tools = True
                    break
    if has_tools:
        req_caps.tool_calling = True
        strict_reqs.add("tool_calling")

    # 2. Structured output requested -> require structured_output
    response_format = kwargs.get("response_format")
    schema = kwargs.get("schema")
    fmt = kwargs.get("format")
    if (
        response_format is not None
        or schema is not None
        or fmt == "json"
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
                        if part_type in ("image_url", "image") or "image_url" in part:
                            has_image = True
                            break
            elif isinstance(content, str) and ("data:image/" in content or "base64," in content):
                has_image = True
            if has_image:
                break
    if prompt and ("data:image/" in prompt or "base64," in prompt):
        has_image = True

    if has_image:
        req_caps.vision = True
        strict_reqs.add("vision")

    # 4. Streaming requested -> require streaming
    if kwargs.get("stream") is True or kwargs.get("streaming") is True:
        req_caps.streaming = True
        strict_reqs.add("streaming")

    # 5. Reasoning-specific routing
    if (
        model == "reasoning"
        or kwargs.get("reasoning") is True
        or kwargs.get("reasoning_effort") is not None
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

    # 7. Context window requirement
    if min_context and min_context > 0:
        req_caps.context_window = min_context
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
            if explicit_parsed.context_window > req_caps.context_window:
                req_caps.context_window = explicit_parsed.context_window
                strict_reqs.add("context_window")

    return RequestRequirements(
        capabilities=req_caps,
        coding_preferred=coding_preferred,
        strict_requirements=strict_reqs,
    )
