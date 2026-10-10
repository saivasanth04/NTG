"""Architecture-aware change planner, evidence validator, and human-in-the-loop approval lifecycle."""

from __future__ import annotations

import json
import re
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


def _evaluate_source_verification(context: dict[str, Any] | None) -> dict[str, Any]:
    """Evaluate per-source verification state without assuming unverified sources are healthy."""
    if not isinstance(context, dict):
        return {
            "inventory_verified": False,
            "graphify_verified": False,
            "graphify_fresh": False,
            "memory_verified": False,
            "memory_fresh": False,
            "errors": ["Context dictionary is missing"],
            "all_verified": False,
        }

    index_status = context.get("index_status")
    index_dict = index_status if isinstance(index_status, dict) else {}
    graph_st = index_dict.get("graphify") if isinstance(index_dict.get("graphify"), dict) else {}
    mem_st = index_dict.get("memory") if isinstance(index_dict.get("memory"), dict) else {}
    coverage = context.get("coverage") if isinstance(context.get("coverage"), dict) else {}
    raw_errors = context.get("errors")
    errors = [str(e) for e in raw_errors] if isinstance(raw_errors, list) else []

    inventory_verified = bool(
        coverage.get("complete_inventory_available")
        and isinstance(context.get("repository_inventory"), dict)
    )
    graphify_verified = bool(graph_st.get("verified"))
    graphify_fresh = bool(graphify_verified and graph_st.get("fresh"))
    memory_verified = bool(mem_st.get("verified"))
    memory_fresh = bool(memory_verified and mem_st.get("fresh"))

    all_verified = bool(
        inventory_verified
        and graphify_fresh
        and memory_fresh
        and len(errors) == 0
    )

    return {
        "inventory_verified": inventory_verified,
        "graphify_verified": graphify_verified,
        "graphify_fresh": graphify_fresh,
        "memory_verified": memory_verified,
        "memory_fresh": memory_fresh,
        "errors": errors,
        "all_verified": all_verified,
    }


def _append_context_sections(
    sections: list[str],
    context: dict[str, Any] | None,
) -> None:
    """Append repository identity, inventory, reading order, Graphify, Codebase Memory, and diagnostics."""
    if not context:
        return

    ver = _evaluate_source_verification(context)
    coverage = context.get("coverage") if isinstance(context.get("coverage"), dict) else {}
    meta_lines: list[str] = []

    if context.get("repo_root"):
        meta_lines.append(f"- **Repository Root**: `{context['repo_root']}`")
    if context.get("project"):
        meta_lines.append(f"- **Codebase Memory Project**: `{context['project']}`")

    if coverage:
        meta_lines.append(
            f"- **Analyzed Files**: {coverage.get('total_repo_files', 0)} total files "
            f"({coverage.get('python_files', 0)} Python modules, "
            f"{coverage.get('config_doc_files', 0)} config/doc files; "
            f"{coverage.get('excluded_dirs_count', 0)} excluded directories, "
            f"{coverage.get('excluded_files_count', 0)} excluded files)"
        )

    if ver["all_verified"]:
        meta_lines.append(
            "- **Verification Status**: All knowledge sources actively verified and fresh "
            "(AST Inventory=verified, Graphify=verified/fresh, Codebase Memory MCP=verified/fresh, 0 errors)"
        )
    else:
        meta_lines.append(
            "- **Verification Status**: Degraded or unverified sources detected "
            f"(AST Inventory={'verified' if ver['inventory_verified'] else 'unverified'}, "
            f"Graphify={'verified/fresh' if ver['graphify_fresh'] else 'unverified/stale'}, "
            f"Codebase Memory MCP={'verified/fresh' if ver['memory_fresh'] else 'unverified/stale'}, "
            f"errors={len(ver['errors'])})"
        )

    if ver["errors"]:
        meta_lines.append(
            "- **Retrieval Diagnostics / Errors**: " + "; ".join(ver["errors"])
        )

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
        "3. Cover every relevant file in the repository across all architectural layers: Project Overview & Configuration (`README.md`, `pyproject.toml`, `requirements.txt`, `.env.example`, `.gitignore`), Execution Entry Points (`main.py`, `ntg/__main__.py`, `test.py`, `ntg/__init__.py`), Core Foundation (`ntg/core/*`), Provider Adapters (`ntg/providers/*`), Smart Routing/State/Telemetry (`ntg/router/*`), Architecture-Aware Coding Agent (`ntg/agent/*`), CLI/Diagnostics (`ntg/cli/*`), and Compatibility Shims (`ntg/verifier.py`, `ntg/agent.py`).",
        "4. When presenting a file reading order, follow the dependency-aware progression from project overview and entry points through core foundation modules, provider adapters, smart routing/state/telemetry, architecture-aware agent modules, CLI/diagnostics, and compatibility shims, explaining what each file does and why it appears in that order.",
        "5. Explicitly mention excluded directories/files (such as `.env`, `.ntg/`, `graphify-out/`, `__pycache__/`) and any retrieval warnings or coverage limitations from the diagnostics section rather than guessing.",
        "",
        f"## User Question\n{query.strip()}",
    ]

    _append_context_sections(sections, context)
    return "\n\n".join(sections)


def _is_repo_wide_query(query: str) -> bool:
    """Return True if the query asks for a project-wide explanation, file inventory, or reading order."""
    q_lower = query.lower()
    keywords = (
        "sequence of files",
        "seqence of files",
        "reading order",
        "files that i should read",
        "understand this project",
        "understand the project",
        "explain this project",
        "explain the codebase",
        "all files",
        "architecture",
        "walkthrough",
    )
    return any(k in q_lower for k in keywords)


def validate_query_answer(
    answer: str,
    query: str,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate an LLM answer against the verified repository file inventory and index status."""
    ver = _evaluate_source_verification(context)
    inv = (
        context.get("repository_inventory")
        if isinstance(context, dict) and isinstance(context.get("repository_inventory"), dict)
        else {}
    )
    files_list = inv.get("files") if isinstance(inv.get("files"), list) else []
    reading_order = inv.get("reading_order") if isinstance(inv.get("reading_order"), list) else []

    all_repo_paths: list[str] = [
        str(f["path"]) for f in files_list if isinstance(f, dict) and f.get("path")
    ]
    py_repo_paths: list[str] = [
        str(f["path"])
        for f in files_list
        if isinstance(f, dict) and f.get("path") and f.get("type") == "python"
    ]
    valid_path_set = set(all_repo_paths)
    valid_basenames = {p.split("/")[-1] for p in all_repo_paths}

    answer_text = answer or ""
    covered_files: list[str] = []
    omitted_files: list[str] = []

    for path in all_repo_paths:
        if path in answer_text or path.replace("/", "\\") in answer_text:
            covered_files.append(path)

    is_broad = _is_repo_wide_query(query)
    if is_broad and all_repo_paths:
        ordered_paths = [
            str(item["path"])
            for item in reading_order
            if isinstance(item, dict) and item.get("path")
        ] or all_repo_paths
        for p in ordered_paths:
            if p not in covered_files:
                omitted_files.append(p)

    # Check for hallucinated or deleted .py path references
    hallucinated_files: list[str] = []
    candidate_py_refs = re.findall(
        r"(?<![A-Za-z0-9_./\\-])((?:ntg/[A-Za-z0-9_/.-]+|[A-Za-z0-9_-]+)\.py)\b",
        answer_text,
    )
    for ref in candidate_py_refs:
        norm_ref = ref.replace("\\", "/").lstrip("./")
        if "/" in norm_ref:
            if norm_ref not in valid_path_set and norm_ref not in hallucinated_files:
                hallucinated_files.append(norm_ref)
        else:
            if norm_ref not in valid_basenames and norm_ref not in hallucinated_files:
                hallucinated_files.append(norm_ref)

    covered_py = [p for p in covered_files if p in py_repo_paths]
    warnings: list[str] = []

    if ver["errors"]:
        warnings.extend(ver["errors"])
    if not ver["graphify_fresh"]:
        warnings.append("Graphify index was not verified as fresh.")
    if not ver["memory_fresh"]:
        warnings.append("Codebase Memory MCP index was not verified as fresh.")
    if omitted_files:
        warnings.append(
            f"LLM response omitted {len(omitted_files)} file(s) from the verified inventory: {', '.join(omitted_files)}"
        )
    if hallucinated_files:
        warnings.append(
            f"LLM response referenced {len(hallucinated_files)} non-existent/deleted file path(s): {', '.join(hallucinated_files)}"
        )

    critical_retrieval_failed = bool(
        not ver["inventory_verified"]
        or not ver["graphify_fresh"]
        or not ver["memory_fresh"]
        or len(ver["errors"]) > 0
    )

    is_complete = bool(
        answer_text.strip()
        and not critical_retrieval_failed
        and not omitted_files
        and not hallucinated_files
    )

    return {
        "complete": is_complete,
        "is_repo_wide_query": is_broad,
        "retrieval_healthy": ver["all_verified"],
        "critical_retrieval_failed": critical_retrieval_failed,
        "inventory_verified": ver["inventory_verified"],
        "graphify_verified": ver["graphify_verified"],
        "graphify_fresh": ver["graphify_fresh"],
        "memory_verified": ver["memory_verified"],
        "memory_fresh": ver["memory_fresh"],
        "total_repo_files": len(all_repo_paths),
        "covered_files_count": len(covered_files),
        "total_python_files": len(py_repo_paths),
        "covered_python_files_count": len(covered_py),
        "covered_files": covered_files,
        "omitted_files": omitted_files,
        "hallucinated_files": hallucinated_files,
        "retrieval_errors": ver["errors"],
        "warnings": warnings,
    }


def finalize_grounded_answer(
    answer: str,
    query: str,
    context: dict[str, Any] | None = None,
    validation: dict[str, Any] | None = None,
    strict: bool = False,
    include_diagnostics: bool = False,
) -> str:
    """Validate and finalize an answer against verified inventory evidence.
    - In strict mode, raises RuntimeError if critical retrieval/indexing failed or answer is empty.
    - Supplements any omitted repository files using factual AST inventory metadata.
    - Flags any hallucinated/deleted file references.
    - Appends verification and coverage diagnostics when requested or when warnings exist.
    """
    val = validation or validate_query_answer(answer=answer, query=query, context=context)

    if strict and (val["critical_retrieval_failed"] or not (answer and answer.strip())):
        err_msg = "; ".join(val["warnings"]) or "Critical retrieval/indexing failure or empty answer"
        raise RuntimeError(
            f"Strict mode: refusing to present answer as complete due to verification failure: {err_msg}"
        )

    output_blocks: list[str] = [(answer or "").strip()]
    inv = (
        context.get("repository_inventory")
        if isinstance(context, dict) and isinstance(context.get("repository_inventory"), dict)
        else {}
    )
    reading_order = inv.get("reading_order") if isinstance(inv.get("reading_order"), list) else []
    ro_by_path = {
        str(item["path"]): item
        for item in reading_order
        if isinstance(item, dict) and item.get("path")
    }

    # Supplement any omitted files from the verified AST inventory so coverage is 100% complete
    if val["omitted_files"] and ro_by_path:
        supp_lines: list[str] = [
            "---",
            "### Verified Inventory Supplement (Additional Repository Files)",
            "The following repository files were verified by the deterministic AST inventory and are included for 100% repository coverage:",
        ]
        for path in val["omitted_files"]:
            item = ro_by_path.get(path)
            if not item:
                continue
            deps = item.get("depends_on") or []
            deps_str = ", ".join(f"`{d}`" for d in deps) if deps else "None (leaf/standalone)"
            supp_lines.append(
                f"- **`{path}`** (*{item.get('layer', 'Repository')}*, step {item.get('step')}): "
                f"{item.get('summary', '')} (Internal dependencies: {deps_str})"
            )
        output_blocks.append("\n".join(supp_lines))

    if val["hallucinated_files"]:
        output_blocks.append(
            "---\n### Evidence Grounding Warning\n"
            "The following file paths mentioned above do not exist in the verified repository inventory and should be disregarded: "
            + ", ".join(f"`{h}`" for h in val["hallucinated_files"])
        )

    if include_diagnostics or val["retrieval_errors"] or val["critical_retrieval_failed"]:
        coverage = (
            context.get("coverage")
            if isinstance(context, dict) and isinstance(context.get("coverage"), dict)
            else {}
        )
        index_status = (
            context.get("index_status")
            if isinstance(context, dict) and isinstance(context.get("index_status"), dict)
            else {}
        )
        graph_st = (
            index_status.get("graphify")
            if isinstance(index_status.get("graphify"), dict)
            else {}
        )
        mem_st = (
            index_status.get("memory")
            if isinstance(index_status.get("memory"), dict)
            else {}
        )
        excl_dirs = [d["path"] for d in coverage.get("excluded_dirs", []) if isinstance(d, dict)]
        excl_files = [
            f"{f['path']} ({f['reason']})"
            for f in coverage.get("excluded_files", [])
            if isinstance(f, dict)
        ]

        effective_covered = (
            val["total_repo_files"]
            if (val["is_repo_wide_query"] and ro_by_path)
            else val["covered_files_count"]
        )

        diag_lines: list[str] = [
            "---",
            "### Verification & Coverage Diagnostics",
            f"- **Repository Root**: `{context.get('repo_root', '') if isinstance(context, dict) else ''}`",
            f"- **Graphify Index**: verified=`{val['graphify_verified']}`, fresh=`{val['graphify_fresh']}`, "
            f"nodes={graph_st.get('node_count', 0)}, edges={graph_st.get('edge_count', 0)}, "
            f"indexed_files={len(graph_st.get('indexed_files', []))}, "
            f"deleted_files={len(graph_st.get('deleted_files', []))}, "
            f"missing_files={len(graph_st.get('missing_files', []))}",
            f"- **Codebase Memory MCP Index**: project=`{mem_st.get('project', '')}`, "
            f"verified=`{val['memory_verified']}`, fresh=`{val['memory_fresh']}`, "
            f"indexed_files={len(mem_st.get('indexed_files', []))}, "
            f"deleted_files={len(mem_st.get('deleted_files', []))}, "
            f"missing_files={len(mem_st.get('missing_files', []))}, "
            f"stale_files={len(mem_st.get('stale_files', []))}",
            f"- **Repository Inventory Coverage**: {effective_covered}/{val['total_repo_files']} files covered "
            f"({val['total_python_files']} Python modules, {coverage.get('config_doc_files', 0)} config/doc files; "
            f"initial LLM omissions supplemented={len(val['omitted_files'])}; "
            f"hallucinated paths={len(val['hallucinated_files'])})",
            f"- **Excluded Directories**: {', '.join(f'`{d}`' for d in excl_dirs) or 'None'}",
            f"- **Excluded Files**: {', '.join(f'`{f}`' for f in excl_files) or 'None'}",
            f"- **Retrieval Errors**: {'; '.join(val['retrieval_errors']) if val['retrieval_errors'] else 'None (0 errors)'}",
        ]
        output_blocks.append("\n".join(diag_lines))

    return "\n\n".join(b for b in output_blocks if b)


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
