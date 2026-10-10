"""Architecture-aware change planner and human-in-the-loop approval lifecycle."""

from __future__ import annotations

import json
from typing import Any


def extract_response_content(planning_response: Any) -> str:
    """Safely extract text content from a LiteLLM ModelResponse, dict, or string."""
    if planning_response is None:
        raise ValueError("planning_response cannot be None.")

    if isinstance(planning_response, str):
        return planning_response

    choices = getattr(planning_response, "choices", None)
    if choices is not None:
        if not choices:
            return ""
        first = choices[0]
        message = getattr(first, "message", None)
        if message is not None:
            content = getattr(message, "content", None)
            if content is None and isinstance(message, dict):
                content = message.get("content")
            return str(content) if content is not None else ""
        if isinstance(first, dict):
            msg = first.get("message") or {}
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
            return str(content) if content is not None else ""

    if isinstance(planning_response, dict):
        if "choices" in planning_response and planning_response["choices"]:
            first = planning_response["choices"][0]
            if isinstance(first, dict):
                msg = first.get("message") or {}
                content = msg.get("content", "")
                return str(content) if content is not None else ""
        if "content" in planning_response:
            val = planning_response["content"]
            return str(val) if val is not None else ""
        if "plan" in planning_response:
            val = planning_response["plan"]
            return str(val) if val is not None else ""

    raise ValueError(
        "Unsupported planning_response format; expected LiteLLM response, dict, or str."
    )


def build_planning_prompt(
    request: str,
    context: dict[str, Any] | None = None,
) -> str:
    """Construct an architecture-aware planning prompt combining the user request
    with structural context from Graphify and Codebase Memory MCP.
    """
    if not isinstance(request, str) or not request.strip():
        raise ValueError("request must be a non-empty string.")

    sections: list[str] = [
        "You are an architecture-aware AI coding agent for this repository.",
        "Analyze the user request and the repository knowledge-graph context below,",
        "then produce a clear, reviewable implementation plan covering:",
        "1. Architectural Impact & Affected Components",
        "2. Files & Symbols to Modify",
        "3. Step-by-Step Implementation Plan",
        "4. Verification & Post-Change Knowledge Sync Strategy",
        "",
        f"## User Request\n{request.strip()}",
    ]

    if context:
        graphify_ctx = context.get("graphify_context")
        if graphify_ctx:
            sections.append(f"## Graphify Knowledge Graph Context\n{graphify_ctx}")

        arch_ctx = context.get("memory_architecture")
        if arch_ctx:
            formatted_arch = (
                json.dumps(arch_ctx, indent=2)
                if isinstance(arch_ctx, (dict, list))
                else str(arch_ctx)
            )
            sections.append(f"## Codebase Memory Architecture Overview\n{formatted_arch}")

        symbols_ctx = context.get("memory_symbols")
        if symbols_ctx:
            formatted_symbols = (
                json.dumps(symbols_ctx, indent=2)
                if isinstance(symbols_ctx, (dict, list))
                else str(symbols_ctx)
            )
            sections.append(f"## Matching Codebase Symbols\n{formatted_symbols}")

    return "\n\n".join(sections)


def build_query_prompt(
    query: str,
    context: dict[str, Any] | None = None,
) -> str:
    """Construct an architecture-grounded prompt for answering a query about a directory."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string.")

    sections: list[str] = [
        "You are an architecture-aware AI codebase assistant.",
        "Use the repository knowledge-graph and symbol context below to answer the user's question accurately and concisely.",
        "Reference specific files, modules, classes, and functions from the codebase context where relevant.",
        "",
        f"## User Question\n{query.strip()}",
    ]

    if context:
        graphify_ctx = context.get("graphify_context")
        if graphify_ctx:
            sections.append(f"## Graphify Knowledge Graph Context\n{graphify_ctx}")

        arch_ctx = context.get("memory_architecture")
        if arch_ctx:
            formatted_arch = (
                json.dumps(arch_ctx, indent=2)
                if isinstance(arch_ctx, (dict, list))
                else str(arch_ctx)
            )
            sections.append(f"## Codebase Memory Architecture Overview\n{formatted_arch}")

        symbols_ctx = context.get("memory_symbols")
        if symbols_ctx:
            formatted_symbols = (
                json.dumps(symbols_ctx, indent=2)
                if isinstance(symbols_ctx, (dict, list))
                else str(symbols_ctx)
            )
            sections.append(f"## Matching Codebase Symbols\n{formatted_symbols}")

    return "\n\n".join(sections)


def build_change_plan(
    request: str,
    planning_response: Any,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a reviewable plan from the model's planning response."""
    if not isinstance(request, str) or not request.strip():
        raise ValueError("request must be a non-empty string.")

    content = extract_response_content(planning_response)

    plan_dict: dict[str, Any] = {
        "request": request,
        "plan": content or "",
        "status": "awaiting_review",
        "approved": False,
        "files_changed": [],
        "validation_results": [],
    }
    if context is not None:
        plan_dict["context"] = context
    return plan_dict


def approve_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Explicitly approve a plan after a human has reviewed it.
    Calling this function alone does not edit source files.
    """
    if not isinstance(plan, dict):
        raise ValueError("plan must be a dictionary.")
    if plan.get("status") != "awaiting_review":
        raise ValueError("Only a pending plan can be approved.")

    plan["approved"] = True
    plan["status"] = "approved"
    return plan


def reject_plan(plan: dict[str, Any], reason: str = "") -> dict[str, Any]:
    """Explicitly reject a pending plan so it cannot be executed."""
    if not isinstance(plan, dict):
        raise ValueError("plan must be a dictionary.")
    if plan.get("status") != "awaiting_review":
        raise ValueError("Only a pending plan can be rejected.")

    plan["approved"] = False
    plan["status"] = "rejected"
    if reason:
        plan["rejection_reason"] = reason
    return plan


def require_approved_plan(plan: dict[str, Any]) -> None:
    """Guard that raises PermissionError unless the plan has been explicitly approved."""
    if not isinstance(plan, dict):
        raise ValueError("plan must be a dictionary.")
    if not plan.get("approved") or plan.get("status") not in (
        "approved",
        "executed",
        "verified",
        "verification_failed",
    ):
        raise PermissionError("Plan must be explicitly approved before executing changes.")


def record_files_changed(plan: dict[str, Any], files: list[str]) -> dict[str, Any]:
    """Record modified file paths on an approved plan."""
    require_approved_plan(plan)
    existing = list(plan.get("files_changed") or [])
    for file_path in files:
        if file_path not in existing:
            existing.append(file_path)
    plan["files_changed"] = existing
    plan["status"] = "executed"
    return plan


def record_validation_result(
    plan: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    """Append a verification check result to the plan and update its status."""
    if not isinstance(plan, dict):
        raise ValueError("plan must be a dictionary.")
    results = list(plan.get("validation_results") or [])
    results.append(result)
    plan["validation_results"] = results

    all_passed = all(bool(r.get("passed", False)) for r in results)
    if plan.get("approved"):
        plan["status"] = "verified" if all_passed else "verification_failed"
    return plan
