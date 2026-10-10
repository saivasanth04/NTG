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


def _append_context_sections(
    sections: list[str],
    context: dict[str, Any] | None,
) -> None:
    """Append repository identity, inventory, reading order, Graphify, Codebase Memory, and diagnostics."""
    if not context:
        return

    meta_lines: list[str] = []
    if context.get("repo_root"):
        meta_lines.append(f"- **Repository Root**: `{context['repo_root']}`")
    if context.get("project"):
        meta_lines.append(f"- **Codebase Memory Project**: `{context['project']}`")
    coverage = context.get("coverage")
    if isinstance(coverage, dict):
        meta_lines.append(
            f"- **Coverage**: {coverage.get('total_repo_files', 0)} total files "
            f"({coverage.get('python_files', 0)} Python modules, "
            f"{coverage.get('config_doc_files', 0)} config/doc files); "
            f"Graphify fresh={coverage.get('graphify_fresh')}; "
            f"Codebase Memory indexed={coverage.get('memory_indexed')}"
        )
    errors = context.get("errors")
    if isinstance(errors, list) and errors:
        meta_lines.append(
            "- **Retrieval Diagnostics / Warnings**: " + "; ".join(str(e) for e in errors)
        )
    else:
        meta_lines.append("- **Retrieval Diagnostics**: All knowledge sources healthy (0 errors)")

    if meta_lines:
        sections.append("## Repository Identity & Index Diagnostics\n" + "\n".join(meta_lines))

    formatted_inv = context.get("formatted_inventory")
    if isinstance(formatted_inv, str) and formatted_inv.strip():
        sections.append(
            "## Verified Repository File Inventory & Dependency-Aware Reading Order\n"
            + formatted_inv.strip()
        )
    elif isinstance(context.get("reading_order"), list) and context["reading_order"]:
        ro_text = json.dumps(context["reading_order"], indent=2)
        sections.append(f"## Dependency-Aware Reading Order\n{ro_text}")

    graph_report = context.get("graphify_report")
    if isinstance(graph_report, str) and graph_report.strip():
        sections.append(f"## Graphify Architecture Report Summary\n{graph_report.strip()}")

    graphify_ctx = context.get("graphify_context")
    if graphify_ctx:
        sections.append(f"## Graphify Knowledge Graph Context\n{graphify_ctx}")

    arch_ctx = context.get("memory_architecture") or context.get("architecture_overview")
    if arch_ctx:
        formatted_arch = (
            json.dumps(arch_ctx, indent=2)
            if isinstance(arch_ctx, (dict, list))
            else str(arch_ctx)
        )
        sections.append(f"## Codebase Memory Architecture Overview\n{formatted_arch}")

    symbols_ctx = context.get("memory_symbols") or context.get("memory_context")
    if symbols_ctx:
        formatted_symbols = (
            json.dumps(symbols_ctx, indent=2)
            if isinstance(symbols_ctx, (dict, list))
            else str(symbols_ctx)
        )
        sections.append(f"## Matching Codebase Symbols\n{formatted_symbols}")


def build_planning_prompt(
    request: str,
    context: dict[str, Any] | None = None,
) -> str:
    """Construct an architecture-aware planning prompt combining the user request
    with deterministic repository inventory, Graphify, and Codebase Memory MCP context.
    """
    if not isinstance(request, str) or not request.strip():
        raise ValueError("request must be a non-empty string.")

    sections: list[str] = [
        "You are an architecture-aware AI coding agent for this repository.",
        "Analyze the user request and the verified repository inventory, dependency graph, Graphify context, and Codebase Memory MCP context below,",
        "then produce a clear, reviewable implementation plan covering:",
        "1. Architectural Impact & Affected Components",
        "2. Files & Symbols to Modify",
        "3. Step-by-Step Implementation Plan",
        "4. Verification & Post-Change Knowledge Sync Strategy",
        "",
        "Ground your plan strictly in the actual files, classes, functions, and dependencies listed below. Never invent non-existent files or symbols.",
        "",
        f"## User Request\n{request.strip()}",
    ]

    _append_context_sections(sections, context)
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
        "Use the verified repository file inventory, AST dependency analysis, Graphify knowledge graph, and Codebase Memory MCP symbol context below to answer the user's question accurately, thoroughly, and concisely.",
        "Strict Grounding Rules:",
        "1. Reference the exact file paths, modules, classes, functions, and internal dependencies from the Verified Repository File Inventory below.",
        "2. Never invent, guess, or hallucinate files, modules, classes, or dependencies that are not present in the inventory.",
        "3. Cover the complete repository across all architectural layers (configuration/docs, entry points, `ntg/core/`, `ntg/providers/`, `ntg/router/`, `ntg/agent/`, `ntg/cli/`, and compatibility shims).",
        "4. When presenting a file reading order, follow the dependency-aware progression from entry points and project configuration through core foundation modules, provider adapters, smart routing/state/telemetry, architecture-aware agent modules, and CLI/diagnostics, explaining what each file does and why it appears in that order.",
        "5. If any retrieval warnings or coverage gaps appear in the diagnostics section, state them explicitly rather than guessing.",
        "",
        f"## User Question\n{query.strip()}",
    ]

    _append_context_sections(sections, context)
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
