"""Architecture-aware change planner, deterministic evidence validator, and human-in-the-loop approval lifecycle."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any


_STOP_WORDS: frozenset[str] = frozenset(
    {
        "this",
        "that",
        "with",
        "from",
        "into",
        "file",
        "files",
        "module",
        "modules",
        "read",
        "reading",
        "should",
        "understand",
        "project",
        "codebase",
        "used",
        "uses",
        "using",
        "provides",
        "provide",
        "defines",
        "define",
        "contains",
        "contain",
        "step",
        "layer",
        "order",
        "after",
        "before",
        "when",
        "where",
        "which",
        "their",
        "there",
        "these",
        "those",
        "also",
        "only",
        "each",
        "both",
        "such",
        "than",
        "then",
        "them",
        "they",
        "have",
        "has",
        "been",
        "were",
        "will",
        "would",
        "could",
    }
)


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
            content = (
                msg.get("content")
                if isinstance(msg, dict)
                else getattr(msg, "content", "")
            )
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
    """Evaluate per-source verification and fingerprint state without assuming unverified sources are healthy."""
    if not isinstance(context, dict):
        return {
            "inventory_verified": False,
            "graphify_verified": False,
            "graphify_fresh": False,
            "graphify_fingerprint_verified": False,
            "graphify_ast_symbols_verified": False,
            "memory_verified": False,
            "memory_fresh": False,
            "memory_fingerprint_verified": False,
            "errors": ["Context dictionary is missing"],
            "all_verified": False,
        }

    index_status = context.get("index_status")
    index_dict = index_status if isinstance(index_status, dict) else {}
    graph_st = (
        index_dict.get("graphify")
        if isinstance(index_dict.get("graphify"), dict)
        else {}
    )
    mem_st = (
        index_dict.get("memory") if isinstance(index_dict.get("memory"), dict) else {}
    )
    coverage = (
        context.get("coverage") if isinstance(context.get("coverage"), dict) else {}
    )
    raw_errors = context.get("errors")
    errors = [str(e) for e in raw_errors] if isinstance(raw_errors, list) else []

    inventory_verified = bool(
        coverage.get("complete_inventory_available")
        and isinstance(context.get("repository_inventory"), dict)
    )
    graphify_fp = bool(graph_st.get("fingerprint_verified"))
    graphify_ast = bool(graph_st.get("ast_symbols_verified"))
    graphify_verified = bool(graph_st.get("verified") and graphify_fp and graphify_ast)
    graphify_fresh = bool(graphify_verified and graph_st.get("fresh"))

    memory_fp = bool(mem_st.get("fingerprint_verified"))
    memory_verified = bool(mem_st.get("verified") and memory_fp)
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
        "graphify_fingerprint_verified": graphify_fp,
        "graphify_ast_symbols_verified": graphify_ast,
        "memory_verified": memory_verified,
        "memory_fresh": memory_fresh,
        "memory_fingerprint_verified": memory_fp,
        "errors": errors,
        "all_verified": all_verified,
    }


def _append_context_sections(
    sections: list[str],
    context: dict[str, Any] | None,
    include_execution_flows: bool = False,
) -> None:
    """Append repository identity, inventory, reading order, Graphify, Codebase Memory, and diagnostics."""
    if not context:
        return

    ver = _evaluate_source_verification(context)
    coverage = (
        context.get("coverage") if isinstance(context.get("coverage"), dict) else {}
    )
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
            "(AST Inventory=verified, Graphify=SHA-256 & AST verified/fresh, "
            "Codebase Memory MCP=SHA-256 & symbol verified/fresh, 0 errors)"
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
        sections.append(
            "## Repository Identity & Index Diagnostics\n" + "\n".join(meta_lines)
        )

    formatted_inv = context.get("formatted_inventory")
    if isinstance(formatted_inv, str) and formatted_inv.strip():
        sections.append(
            "## Verified Repository File Inventory & Dependency-Aware Reading Order\n"
            + formatted_inv.strip()
        )
    elif isinstance(context.get("reading_order"), list) and context["reading_order"]:
        ro_text = json.dumps(context["reading_order"], indent=2)
        sections.append(f"## Dependency-Aware Reading Order\n{ro_text}")

    if include_execution_flows and (
        not isinstance(formatted_inv, str)
        or "Verified Runtime Execution Flows" not in formatted_inv
    ):
        inv_dict = (
            context.get("repository_inventory")
            if isinstance(context.get("repository_inventory"), dict)
            else {}
        )
        exec_flows = (
            context.get("execution_flows")
            or inv_dict.get("execution_flows")
            or []
        )
        if isinstance(exec_flows, list) and exec_flows:
            flow_lines: list[str] = []
            for flow in exec_flows:
                if not isinstance(flow, dict):
                    continue
                ep = flow.get("entry_point")
                ecalls = flow.get("entry_calls") or []
                ecall_str = f" (entry calls: `{', '.join(ecalls)}`)" if ecalls else ""
                hop_strs = [
                    f"`{h['from']}` -> `{h['to']}`"
                    + (
                        f" [{', '.join(h['invoked_symbols'][:4])}]"
                        if h.get("invoked_symbols")
                        else ""
                    )
                    for h in (flow.get("hops") or [])[:8]
                    if isinstance(h, dict)
                ]
                flow_lines.append(
                    f"- **Entry `{ep}`**{ecall_str}: " + " ; ".join(hop_strs)
                )
            if flow_lines:
                sections.append(
                    "### Verified Runtime Execution Flows (AST Call-Graph Traces)\n"
                    + "\n".join(flow_lines)
                )

    graph_report = context.get("graphify_report")
    if isinstance(graph_report, str) and graph_report.strip():
        sections.append(
            f"## Graphify Architecture Report Summary\n{graph_report.strip()}"
        )

    graphify_ctx = context.get("graphify_context")
    if graphify_ctx:
        sections.append(f"## Graphify Knowledge Graph Context\n{graphify_ctx}")

    arch_ctx = context.get("memory_architecture") or context.get(
        "architecture_overview"
    )
    if arch_ctx:
        formatted_arch = (
            json.dumps(arch_ctx, indent=2)
            if isinstance(arch_ctx, (dict, list))
            else str(arch_ctx)
        )
        sections.append(
            f"## Codebase Memory Architecture Overview\n{formatted_arch}"
        )

    symbols_ctx = context.get("memory_symbols") or context.get("memory_context")
    if symbols_ctx:
        formatted_symbols = (
            json.dumps(symbols_ctx, indent=2)
            if isinstance(symbols_ctx, (dict, list))
            else str(symbols_ctx)
        )
        sections.append(f"## Matching Codebase Symbols\n{formatted_symbols}")

    scoped_excerpts = context.get("scoped_source_excerpts")
    if isinstance(scoped_excerpts, dict) and scoped_excerpts:
        excerpt_blocks = [
            f"### `{rel_p}`\n```python\n{snippet.strip()}\n```"
            for rel_p, snippet in scoped_excerpts.items()
            if isinstance(snippet, str) and snippet.strip()
        ]
        if excerpt_blocks:
            sections.append(
                "## Scoped Source Code Excerpts (Direct Source Inspection)\n"
                + "\n\n".join(excerpt_blocks)
            )
    elif isinstance(scoped_excerpts, str) and scoped_excerpts.strip():
        sections.append(
            "## Scoped Source Code Excerpts (Direct Source Inspection)\n"
            + scoped_excerpts.strip()
        )


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


_GENERIC_SNAKE_TERMS: frozenset[str] = frozenset(
    {
        "reading_order",
        "file_path",
        "file_paths",
        "code_base",
        "source_code",
        "entry_point",
        "entry_points",
        "api_key",
        "api_keys",
        "rate_limit",
        "rate_limits",
        "circuit_breaker",
        "circuit_breakers",
        "step_by_step",
        "end_to_end",
        "high_level",
        "top_level",
        "command_line",
        "open_router",
        "error_handling",
        "data_flow",
        "call_graph",
    }
)

_PYTHON_BUILTIN_IDENTIFIERS: frozenset[str] = frozenset(
    {
        "True",
        "False",
        "None",
        "Any",
        "Optional",
        "Union",
        "Callable",
        "Iterable",
        "Iterator",
        "Sequence",
        "Mapping",
        "Path",
        "dict",
        "list",
        "set",
        "frozenset",
        "tuple",
        "str",
        "int",
        "float",
        "bool",
        "bytes",
        "object",
        "type",
        "Exception",
        "RuntimeError",
        "ValueError",
        "TypeError",
        "KeyError",
        "OSError",
        "PermissionError",
        "FileNotFoundError",
        "SyntaxError",
        "dataclass",
        "Enum",
        "main",
        "self",
        "cls",
        "args",
        "kwargs",
    }
)


def _is_repo_wide_query(query: str) -> bool:
    """Return True if the query asks for a project-wide explanation, file inventory, or reading order."""
    q_lower = (query or "").lower()
    keywords = (
        "sequence of files",
        "seqence of files",
        "reading order",
        "order of files",
        "order to read",
        "files that i should read",
        "files to read",
        "which files to read",
        "understand this project",
        "understand the project",
        "understand this repo",
        "understand the codebase",
        "explain this project",
        "explain the project",
        "explain the codebase",
        "explain this repo",
        "all files",
        "every file",
        "repository overview",
        "project overview",
        "architecture",
        "walkthrough",
    )
    return any(k in q_lower for k in keywords)


def _is_early_tier_file(
    path: str,
    rec: dict[str, Any] | None,
    ro_item: dict[str, Any] | None,
) -> bool:
    """Dynamically determine whether `path` belongs to Layer 0 (config/doc) or Layer 1
    (execution entry points & top-level package facade) from the repository inventory.
    """
    if isinstance(ro_item, dict) and isinstance(ro_item.get("layer_rank"), int):
        return int(ro_item["layer_rank"]) < 2
    if isinstance(rec, dict) and rec.get("type") != "python":
        return True
    if not path.endswith(".py"):
        return True
    parts = Path(path).parts
    pkg_parts = parts[1:] if len(parts) > 1 and parts[0] == "src" else parts
    fname = parts[-1]
    if len(pkg_parts) == 1:
        return True
    if fname == "__main__.py":
        return True
    if len(pkg_parts) == 2 and fname == "__init__.py":
        return True
    return False


def analyze_query_requirements(
    query: str,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Dynamically analyze the user's question against the verified repository inventory
    to determine query scope, target files, queried symbols, unmatched targets,
    architectural layers, cross-package dependency constraints, and requested deliverables.
    """
    q_text = (query or "").strip()
    q_lower = q_text.lower()

    inv = (
        context.get("repository_inventory")
        if isinstance(context, dict)
        and isinstance(context.get("repository_inventory"), dict)
        else {}
    )
    files_list = inv.get("files") if isinstance(inv.get("files"), list) else []
    reading_order = (
        inv.get("reading_order") if isinstance(inv.get("reading_order"), list) else []
    )
    excluded_dirs = (
        inv.get("excluded_dirs") if isinstance(inv.get("excluded_dirs"), list) else []
    )
    excluded_files = (
        inv.get("excluded_files") if isinstance(inv.get("excluded_files"), list) else []
    )

    files_by_path: dict[str, dict[str, Any]] = {
        str(f["path"]): f
        for f in files_list
        if isinstance(f, dict) and f.get("path")
    }
    ro_by_path: dict[str, dict[str, Any]] = {
        str(item["path"]): item
        for item in reading_order
        if isinstance(item, dict) and item.get("path")
    }
    ordered_repo_paths: list[str] = [
        str(item["path"])
        for item in reading_order
        if isinstance(item, dict) and item.get("path") in files_by_path
    ]
    for p in files_by_path:
        if p not in ordered_repo_paths:
            ordered_repo_paths.append(p)

    # Dynamically group architectural layers from reading_order
    layers: list[dict[str, Any]] = []
    for item in reading_order:
        if not isinstance(item, dict) or not item.get("path"):
            continue
        l_name = str(item.get("layer") or "Modules")
        step_num = int(item.get("step") or (len(layers) + 1))
        if not layers or layers[-1]["layer"] != l_name:
            layers.append(
                {
                    "layer": l_name,
                    "layer_rank": item.get("layer_rank", 99),
                    "start_step": step_num,
                    "end_step": step_num,
                    "files": [str(item["path"])],
                }
            )
        else:
            layers[-1]["end_step"] = step_num
            layers[-1]["files"].append(str(item["path"]))

    # Dynamically detect cross-package dependency promotions in reading_order
    cross_package_deps: list[dict[str, Any]] = []
    for item in reading_order:
        if not isinstance(item, dict) or not item.get("path"):
            continue
        mod_path = str(item["path"])
        if _is_early_tier_file(mod_path, files_by_path.get(mod_path), item):
            continue
        mod_step = int(item.get("step") or 0)
        for dep_path in item.get("depends_on") or []:
            dep_item = ro_by_path.get(dep_path)
            if not dep_item or _is_early_tier_file(
                dep_path, files_by_path.get(dep_path), dep_item
            ):
                continue
            dep_step = int(dep_item.get("step") or 0)
            if (
                Path(dep_path).parent != Path(mod_path).parent
                and dep_item.get("layer") == item.get("layer")
                and dep_step < mod_step
            ):
                cross_package_deps.append(
                    {
                        "dependency": dep_path,
                        "dependent": mod_path,
                        "dependency_step": dep_step,
                        "dependent_step": mod_step,
                    }
                )

    # Build repository-wide symbol & module maps
    symbol_to_files: dict[str, set[str]] = {}
    symbol_lower_to_canonical: dict[str, str] = {}
    all_module_stems: set[str] = set()
    all_package_dirs: set[str] = set()

    for path, rec in files_by_path.items():
        stem = Path(path).stem.lower()
        if stem and stem != "__init__":
            all_module_stems.add(stem)
        parent_str = Path(path).parent.as_posix().lower()
        if parent_str and parent_str != ".":
            all_package_dirs.add(parent_str)
            for part in Path(path).parent.parts:
                all_package_dirs.add(part.lower())

        for cls in rec.get("classes", []):
            cname = str(cls.get("name") or "")
            if cname:
                symbol_to_files.setdefault(cname, set()).add(path)
                symbol_lower_to_canonical[cname.lower()] = cname
                for m in cls.get("methods", []):
                    if isinstance(m, str) and not m.startswith("__"):
                        symbol_to_files.setdefault(m, set()).add(path)
                        symbol_lower_to_canonical.setdefault(m.lower(), m)
        for fn in rec.get("functions", []):
            fname = str(fn.get("name") or "")
            if fname:
                symbol_to_files.setdefault(fname, set()).add(path)
                symbol_lower_to_canonical[fname.lower()] = fname
        for const in rec.get("constants", []):
            if isinstance(const, str) and const:
                symbol_to_files.setdefault(const, set()).add(path)
                symbol_lower_to_canonical.setdefault(const.lower(), const)
        for exp in rec.get("exports", []):
            if isinstance(exp, str) and exp:
                symbol_to_files.setdefault(exp, set()).add(path)
                symbol_lower_to_canonical.setdefault(exp.lower(), exp)

    # 1. Detect explicit file/directory mentions and unmatched file references in `query`
    explicit_target_files: set[str] = set()
    unmatched_targets: list[str] = []
    valid_path_set = set(ordered_repo_paths)
    basename_to_paths: dict[str, list[str]] = {}
    for p in ordered_repo_paths:
        basename_to_paths.setdefault(Path(p).name.lower(), []).append(p)

    excluded_file_names = {
        str(f.get("path") or "").lower()
        for f in excluded_files
        if isinstance(f, dict)
    }

    for p in ordered_repo_paths:
        if p.lower() in q_lower or p.replace("/", "\\").lower() in q_lower:
            explicit_target_files.add(p)

    # Check explicit `.py` file tokens in the query
    for py_match in re.findall(
        r"(?<![A-Za-z0-9_./\\-])((?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.py)\b",
        q_text,
    ):
        norm_py = py_match.replace("\\", "/").lstrip("./")
        if norm_py in valid_path_set:
            explicit_target_files.add(norm_py)
        elif "/" not in norm_py and norm_py.lower() in basename_to_paths:
            for matched_p in basename_to_paths[norm_py.lower()]:
                explicit_target_files.add(matched_p)
        elif norm_py.lower() not in excluded_file_names:
            if norm_py not in unmatched_targets:
                unmatched_targets.append(norm_py)

    # Check explicit package directory prefixes (e.g., "ntg/router", "ntg/providers")
    for pkg_dir in sorted(all_package_dirs, key=len, reverse=True):
        if "/" in pkg_dir and pkg_dir in q_lower:
            for p in ordered_repo_paths:
                if p.lower().startswith(pkg_dir + "/"):
                    explicit_target_files.add(p)

    # 2. Detect explicit code symbol mentions and unmatched code identifiers in `query`
    queried_symbols: list[str] = []
    backtick_tokens = re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", q_text)
    camel_tokens = re.findall(
        r"\b([A-Z][a-z0-9]+(?:[A-Z][A-Za-z0-9]*)+)\b", q_text
    )
    snake_tokens = re.findall(r"\b([a-z_][a-z0-9]*_[a-z0-9_]+)\b", q_text)

    candidate_code_tokens: list[str] = []
    for tok in [*backtick_tokens, *camel_tokens, *snake_tokens]:
        if tok not in candidate_code_tokens:
            candidate_code_tokens.append(tok)

    for tok in candidate_code_tokens:
        tok_low = tok.lower()
        if tok in symbol_to_files:
            if tok not in queried_symbols:
                queried_symbols.append(tok)
            explicit_target_files.update(symbol_to_files[tok])
        elif tok_low in symbol_lower_to_canonical:
            canon = symbol_lower_to_canonical[tok_low]
            if canon not in queried_symbols:
                queried_symbols.append(canon)
            explicit_target_files.update(symbol_to_files[canon])
        elif (
            tok_low in all_module_stems
            or tok_low in all_package_dirs
            or tok_low in _GENERIC_SNAKE_TERMS
            or tok in _PYTHON_BUILTIN_IDENTIFIERS
        ):
            for p in ordered_repo_paths:
                if Path(p).stem.lower() == tok_low:
                    explicit_target_files.add(p)
        else:
            if tok not in unmatched_targets:
                unmatched_targets.append(tok)

    # Also match exact symbol names (length >= 5) mentioned without backticks
    for word in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]{4,})\b", q_text):
        if word in symbol_to_files and word not in _STOP_WORDS:
            if word not in queried_symbols:
                queried_symbols.append(word)
            explicit_target_files.update(symbol_to_files[word])

    # 3. Determine scope (`repo_wide` vs `targeted`) and `target_files`
    broad_intent = _is_repo_wide_query(q_text)
    if broad_intent and not unmatched_targets:
        scope = "repo_wide"
        is_repo_wide = True
        target_files = list(ordered_repo_paths)
    elif explicit_target_files:
        scope = "targeted"
        is_repo_wide = False
        # Include direct internal dependencies of explicitly targeted files when dependency/flow context is requested
        expanded = set(explicit_target_files)
        if any(
            k in q_lower
            for k in ("depend", "import", "order", "sequence", "flow", "call")
        ):
            for p in list(explicit_target_files):
                rec = files_by_path.get(p, {})
                for dep in rec.get("internal_dependencies") or []:
                    if dep in files_by_path:
                        expanded.add(dep)
        target_files = [p for p in ordered_repo_paths if p in expanded]
    elif unmatched_targets:
        scope = "targeted"
        is_repo_wide = False
        target_files = []
    else:
        # Keyword relevance scoring across the verified inventory
        q_non_stop = {
            t
            for t in re.findall(r"[a-z0-9_]{4,}", q_lower)
            if t not in _STOP_WORDS
        }
        scored_paths: list[str] = []
        if q_non_stop:
            for p in ordered_repo_paths:
                anchors = _build_file_anchor_tokens(
                    files_by_path[p], ro_by_path.get(p)
                )
                overlap = q_non_stop & anchors
                if len(overlap) >= 2 or (
                    len(q_non_stop) == 1 and len(overlap) == 1
                ):
                    scored_paths.append(p)
        if scored_paths and len(scored_paths) < len(ordered_repo_paths):
            scope = "targeted"
            is_repo_wide = False
            target_files = scored_paths
        else:
            scope = "repo_wide"
            is_repo_wide = True
            target_files = list(ordered_repo_paths)

    # 4. Determine requested deliverables from the user's question
    requested_deliverables: list[str] = []
    if any(
        k in q_lower
        for k in (
            "sequence of files",
            "seqence of files",
            "reading order",
            "order of files",
            "order to read",
            "files that i should read",
            "files to read",
            "which files to read",
            "step by step",
            "step-by-step",
            "walkthrough",
        )
    ):
        requested_deliverables.append("reading_order")

    if (
        any(
            k in q_lower
            for k in (
                "description",
                "describe",
                "explain",
                "summary",
                "summarize",
                "what does",
                "what each",
                "purpose",
                "role",
                "understand",
                "overview",
            )
        )
        or is_repo_wide
        or "reading_order" in requested_deliverables
    ):
        requested_deliverables.append("file_descriptions")

    if (
        any(
            k in q_lower
            for k in (
                "class",
                "classes",
                "function",
                "functions",
                "method",
                "methods",
                "symbol",
                "symbols",
                "api",
                "exports",
                "signature",
            )
        )
        or bool(queried_symbols)
    ):
        requested_deliverables.append("symbol_details")

    if any(
        k in q_lower
        for k in (
            "depend",
            "dependencies",
            "dependency",
            "import",
            "imports",
            "call",
            "flow",
            "interact",
            "relationship",
            "architecture",
            "layer",
            "layers",
        )
    ):
        requested_deliverables.append("dependency_relationships")

    if any(
        k in q_lower
        for k in (
            "execution flow",
            "call graph",
            "call chain",
            "runtime flow",
            "trace execution",
            "how does execution flow",
            "execution path",
        )
    ):
        requested_deliverables.append("execution_flow")

    if is_repo_wide or any(
        k in q_lower
        for k in ("excluded", "ignored", ".gitignore", ".env", "coverage", "limitation")
    ):
        requested_deliverables.append("exclusions_or_limitations")

    if not requested_deliverables:
        requested_deliverables.append("file_descriptions")

    return {
        "scope": scope,
        "is_repo_wide": is_repo_wide,
        "target_files": target_files,
        "queried_symbols": queried_symbols,
        "unmatched_targets": unmatched_targets,
        "requested_deliverables": requested_deliverables,
        "layers": layers,
        "cross_package_dependencies": cross_package_deps,
        "execution_flows": (
            inv.get("execution_flows")
            if isinstance(inv.get("execution_flows"), list)
            else []
        ),
        "call_graph_edges": (
            inv.get("call_graph_edges")
            if isinstance(inv.get("call_graph_edges"), list)
            else []
        ),
        "excluded_dirs": [
            str(d.get("path"))
            for d in excluded_dirs
            if isinstance(d, dict) and d.get("path")
        ],
        "excluded_files": [
            str(f.get("path"))
            for f in excluded_files
            if isinstance(f, dict) and f.get("path")
        ],
    }


def build_query_prompt(
    query: str,
    context: dict[str, Any] | None = None,
) -> str:
    """Construct a question-aware, architecture-grounded prompt dynamically derived
    from the verified repository inventory without hard-coded file or step assumptions.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string.")

    req = analyze_query_requirements(query=query, context=context)
    target_files = req["target_files"]
    total_target = len(target_files)
    layers = req["layers"]
    cross_deps = req["cross_package_dependencies"]
    deliverables = req["requested_deliverables"]

    rules: list[str] = [
        "1. Reference the exact file paths, modules, classes, functions, and internal dependencies from the Verified Repository File Inventory below.",
        "2. Never invent, guess, or hallucinate files, modules, classes, functions, or dependencies that are not present in the verified inventory.",
    ]

    if req["unmatched_targets"]:
        unmatched_str = ", ".join(f"`{u}`" for u in req["unmatched_targets"])
        rules.append(
            f"3. The user question references target(s) not found in the verified repository inventory ({unmatched_str}). Explicitly state that these target(s) do not exist in this repository instead of inventing details."
        )
    elif req["is_repo_wide"] and total_target > 0:
        layer_summaries: list[str] = []
        for l_info in layers:
            l_files = l_info.get("files") or []
            files_str = ", ".join(f"`{p}`" for p in l_files)
            layer_summaries.append(
                f"{l_info['layer']} (steps {l_info['start_step']}–{l_info['end_step']}: {files_str})"
            )
        layers_joined = "; ".join(layer_summaries) if layer_summaries else f"{total_target} repository files"
        rules.append(
            f"3. Cover every single file in the Verified Repository File Inventory ({total_target} files total) across all dynamically discovered architectural layers without omitting or grouping any files: {layers_joined}."
        )
    elif total_target > 0:
        targets_str = ", ".join(f"`{p}`" for p in target_files)
        rules.append(
            f"3. Cover all relevant target file(s) identified for this question ({targets_str}) using their verified AST symbols, docstrings, and dependencies."
        )

    if "reading_order" in deliverables or req["is_repo_wide"]:
        cross_note = ""
        if cross_deps:
            cross_clauses = [
                f"keep `{cd['dependency']}` (step {cd['dependency_step']}) before `{cd['dependent']}` (step {cd['dependent_step']}) because `{cd['dependent']}` imports `{cd['dependency']}`"
                for cd in cross_deps
            ]
            cross_note = f" (in particular, {'; '.join(cross_clauses)})"
        step_range_str = (
            f"steps 1 through {total_target}"
            if total_target > 0
            else "the dependency-aware reading order"
        )
        rules.append(
            f"4. Preserve the EXACT numbered dependency-aware reading order ({step_range_str}) from the Verified Repository File Inventory below without reordering or grouping any steps{cross_note}. Write each file as its own separate numbered step (`1. \\`path\\`: ...` through `{total_target}. \\`path\\`: ...`) with its backticked relative path, actual classes/functions/exports, and a concise evidence-grounded description of what the file does."
        )
    else:
        deliv_str = ", ".join(deliverables)
        rules.append(
            f"4. Address every requested deliverable ({deliv_str}): provide evidence-grounded descriptions, actual AST classes/functions/exports, and internal dependency relationships for the relevant modules."
        )

    excl_items = [
        *[f"`{d}/`" for d in req["excluded_dirs"]],
        *[f"`{f}`" for f in req["excluded_files"]],
    ]
    excl_clause = (
        f" (such as {', '.join(excl_items[:6])})" if excl_items else ""
    )
    rules.append(
        f"5. Explicitly mention excluded directories/files{excl_clause} and any retrieval warnings or coverage limitations from the diagnostics section rather than guessing."
    )

    sections: list[str] = [
        "You are an architecture-aware AI codebase assistant.",
        "Use the verified repository file inventory, AST dependency analysis, Graphify knowledge graph, and Codebase Memory MCP symbol context below to answer the user's question accurately, thoroughly, and concisely.",
        "Strict Grounding Rules:",
        *rules,
        "",
        f"## User Question\n{query.strip()}",
    ]

    _append_context_sections(
        sections,
        context,
        include_execution_flows=("execution_flow" in deliverables),
    )
    return "\n\n".join(sections)


def _strip_diagnostic_footers(answer_text: str) -> str:
    """Remove generated diagnostic/warning banners and footers before validating answer content."""
    text = answer_text or ""
    # Strip leading degraded banner if present
    text = re.sub(
        r"^>\s*\[!WARNING\]\s*\*\*DEGRADED / PARTIALLY VERIFIED RESULT\*\*.*?(?:\n\n|\Z)",
        "",
        text,
        flags=re.DOTALL,
    )
    # Strip trailing diagnostic and warning sections
    for marker in (
        "\n---\n### Evidence Grounding Warning",
        "\n### Evidence Grounding Warning",
        "\n---\n### Verification & Coverage Diagnostics",
        "\n### Verification & Coverage Diagnostics",
    ):
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    return text.strip()


def _build_file_anchor_tokens(
    file_rec: dict[str, Any],
    ro_item: dict[str, Any] | None,
) -> set[str]:
    """Build deterministic set of valid grounding tokens (symbols, signatures, calls, raises, dependencies, domain keywords) for a file."""
    path = str(file_rec.get("path") or "")
    anchors: set[str] = set()

    # Path stem and parent package tokens
    stem = Path(path).stem.lower()
    if stem and stem != "__init__" and len(stem) >= 3:
        anchors.add(stem)
        for part in stem.split("_"):
            if len(part) >= 3:
                anchors.add(part)
    parent_name = Path(path).parent.name.lower()
    if parent_name and len(parent_name) >= 3:
        anchors.add(parent_name)

    # AST classes, functions, methods, signatures, calls, raises, exports, constants
    for cls in file_rec.get("classes", []):
        if isinstance(cls, dict) and cls.get("name"):
            cname = str(cls["name"]).lower()
            anchors.add(cname)
            for m in cls.get("methods", []):
                if isinstance(m, str) and len(m) >= 3 and not m.startswith("__"):
                    anchors.add(m.lower())
            for md in cls.get("method_details", []) or []:
                if isinstance(md, dict):
                    if md.get("docstring"):
                        for tok in re.findall(
                            r"[a-z0-9_]{4,}", str(md["docstring"]).lower()
                        ):
                            if tok not in _STOP_WORDS:
                                anchors.add(tok)
                    for call_sym in md.get("calls", []) or []:
                        if isinstance(call_sym, str) and len(call_sym) >= 3:
                            anchors.add(call_sym.lower())
                    for exc_sym in md.get("raises", []) or []:
                        if isinstance(exc_sym, str) and len(exc_sym) >= 3:
                            anchors.add(exc_sym.lower())
            if cls.get("docstring"):
                for tok in re.findall(r"[a-z0-9_]{4,}", str(cls["docstring"]).lower()):
                    if tok not in _STOP_WORDS:
                        anchors.add(tok)

    for fn in file_rec.get("functions", []):
        if isinstance(fn, dict) and fn.get("name"):
            fname = str(fn["name"]).lower()
            anchors.add(fname)
            for part in fname.split("_"):
                if len(part) >= 4 and part not in _STOP_WORDS:
                    anchors.add(part)
            for p_name in fn.get("params", []) or []:
                if isinstance(p_name, str) and len(p_name) >= 4 and p_name not in _STOP_WORDS:
                    anchors.add(p_name.lower())
            for call_sym in fn.get("calls", []) or []:
                if isinstance(call_sym, str) and len(call_sym) >= 3:
                    anchors.add(call_sym.lower())
            for exc_sym in fn.get("raises", []) or []:
                if isinstance(exc_sym, str) and len(exc_sym) >= 3:
                    anchors.add(exc_sym.lower())
            if fn.get("docstring"):
                for tok in re.findall(r"[a-z0-9_]{4,}", str(fn["docstring"]).lower()):
                    if tok not in _STOP_WORDS:
                        anchors.add(tok)

    for call_sym in file_rec.get("all_calls", []) or []:
        if isinstance(call_sym, str) and len(call_sym) >= 3:
            anchors.add(call_sym.lower())

    for exc_sym in file_rec.get("raises", []) or []:
        if isinstance(exc_sym, str) and len(exc_sym) >= 3:
            anchors.add(exc_sym.lower())

    for exp in file_rec.get("exports", []):
        if isinstance(exp, str) and len(exp) >= 3:
            anchors.add(exp.lower())

    for const in file_rec.get("constants", []):
        if isinstance(const, str) and len(const) >= 3:
            anchors.add(const.lower())
            for part in const.lower().split("_"):
                if len(part) >= 4 and part not in _STOP_WORDS:
                    anchors.add(part)

    for dep in file_rec.get("internal_dependencies", []):
        if isinstance(dep, str):
            anchors.add(dep.lower())
            dep_stem = Path(dep).stem.lower()
            if dep_stem != "__init__" and len(dep_stem) >= 3:
                anchors.add(dep_stem)

    for ext in file_rec.get("external_dependencies", []):
        if isinstance(ext, str) and len(ext) >= 3:
            anchors.add(ext.lower())

    # Summary, docstring, and layer tokens
    text_sources = [
        str(file_rec.get("summary") or ""),
        str(file_rec.get("docstring") or ""),
        str((ro_item or {}).get("summary") or ""),
        str((ro_item or {}).get("layer") or ""),
    ]
    combined_doc_lower = " ".join(text_sources).lower()
    for src in text_sources:
        for tok in re.findall(r"[a-z0-9_]{4,}", src.lower()):
            if tok not in _STOP_WORDS:
                anchors.add(tok)

    # Dynamically detect package initializer facades and compatibility shims from inventory metadata
    is_facade_or_shim = (
        path.endswith("__init__.py")
        or bool(file_rec.get("exports"))
        or (isinstance(ro_item, dict) and ro_item.get("layer_rank") == 7)
        or any(
            kw in combined_doc_lower
            for kw in ("compatibility", "shim", "re-export", "facade")
        )
    )
    if is_facade_or_shim:
        anchors.update(
            {
                "re-export",
                "re-exports",
                "reexport",
                "reexports",
                "export",
                "exports",
                "facade",
                "package",
                "initializer",
                "compatibility",
                "shim",
                "public",
                "symbols",
                "canonical",
            }
        )

    if file_rec.get("type") == "config_or_doc":
        anchors.update(
            {
                "documentation",
                "readme",
                "overview",
                "guide",
                "architecture",
                "configuration",
                "config",
                "dependencies",
                "requirements",
                "packages",
                "environment",
                "template",
                "keys",
                "ignore",
                "rules",
                "build",
                "metadata",
                "entry",
            }
        )

    return anchors


def _extract_file_entry_lines(
    answer_body: str,
    all_repo_paths: list[str],
) -> tuple[dict[str, str], dict[str, str], list[str]]:
    """Extract per-file primary lines, multi-line description windows, and the observed reading order
    from `answer_body`. Prioritizes numbered reading-order step lines over indented sub-bullets or
    forward-references in 'Imports'/'Delegates to' clauses.
    """
    if not all_repo_paths or not answer_body:
        return {}, {}, []

    lines = answer_body.splitlines()
    sorted_paths_by_len = sorted(all_repo_paths, key=len, reverse=True)
    escaped_alts = "|".join(
        re.escape(p) + "|" + re.escape(p.replace("/", "\\"))
        for p in sorted_paths_by_len
    )

    # Numbered step line (e.g., "1. `path`: ...", "**1.** `path`", "### 34. **`path`**", "- **Step 34**: `path`")
    numbered_step_re = re.compile(
        r"^\s*(?:#{1,6}\s*)?(?:[-*+•]\s*)?(?:\*\*|\*|_|\s)*(?:Step\s+)?(\d+)(?:\*\*|\*|_|\s)*[:.)–—-](?:\*\*|\*|_|\s|\[|`)*("
        + escaped_alts
        + r")(?![A-Za-z0-9_./\\-])",
        re.IGNORECASE,
    )

    # Supplemented inventory line (e.g., "- **`path`** (*Layer*, step 18): ...")
    supplement_step_re = re.compile(
        r"^\s*[-*+•]\s*\*\*`("
        + escaped_alts
        + r")`\*\*\s*\([^)]*?\bstep\s+(\d+)\)",
        re.IGNORECASE,
    )

    # Top-level unnumbered bullet or heading introducing a file path as the primary subject
    bullet_item_re = re.compile(
        r"^(\s*)(?:[-*+•]|#{1,6})\s*(?:\*\*|`|\*|_|\s)*("
        + escaped_alts
        + r")(?:\*\*|`|\*|_|\s)*(?:[:—–(-]|\s+is\b|\s+defines\b|\s+provides\b|\s+implements\b|\s+contains\b|\s+re-exports\b|\s+manages\b|\s+handles\b|\s+serves\b|\s+-|\s*$)"
    )

    numbered_idx_by_path: dict[str, int] = {}
    numbered_step_by_path: dict[str, int] = {}
    supplement_step_by_path: dict[str, int] = {}
    bullet_idx_by_path: dict[str, int] = {}
    all_subject_indices: set[int] = set()

    for idx, line in enumerate(lines):
        m_supp = supplement_step_re.search(line)
        if m_supp:
            norm_p = m_supp.group(1).replace("\\", "/")
            step_n = int(m_supp.group(2))
            all_subject_indices.add(idx)
            numbered_idx_by_path[norm_p] = idx
            supplement_step_by_path[norm_p] = step_n
            continue

        m_num = numbered_step_re.search(line)
        if m_num:
            step_n = int(m_num.group(1))
            norm_p = m_num.group(2).replace("\\", "/")
            all_subject_indices.add(idx)
            if norm_p not in numbered_idx_by_path:
                numbered_idx_by_path[norm_p] = idx
                numbered_step_by_path[norm_p] = step_n
            continue

        m_bul = bullet_item_re.search(line)
        if m_bul:
            indent = len(m_bul.group(1) or "")
            norm_p = m_bul.group(2).replace("\\", "/")
            # Do not let an indented sub-bullet override an existing top-level bullet
            if norm_p not in bullet_idx_by_path or indent == 0:
                if norm_p not in bullet_idx_by_path:
                    bullet_idx_by_path[norm_p] = idx
                    all_subject_indices.add(idx)

    # Choose the canonical subject line index for each file:
    # numbered step line always wins over an unnumbered sub-bullet mention.
    chosen_idx_by_path: dict[str, int] = {}
    for path in all_repo_paths:
        if path in numbered_idx_by_path:
            chosen_idx_by_path[path] = numbered_idx_by_path[path]
        elif path in bullet_idx_by_path:
            chosen_idx_by_path[path] = bullet_idx_by_path[path]

    # Preserve observed line order for the main answer, while inserting any deterministic
    # supplement entries at their canonical step position relative to numbered steps.
    main_ordered = [
        p
        for p, _ in sorted(
            (
                (p, i)
                for p, i in chosen_idx_by_path.items()
                if p not in supplement_step_by_path
            ),
            key=lambda kv: kv[1],
        )
    ]
    if supplement_step_by_path and numbered_step_by_path:
        combined_with_keys: list[tuple[float, int, str]] = []
        for rank_idx, p in enumerate(main_ordered):
            s_num = float(numbered_step_by_path.get(p, rank_idx + 1))
            combined_with_keys.append((s_num, chosen_idx_by_path[p], p))
        for p, s_num in supplement_step_by_path.items():
            combined_with_keys.append((float(s_num) - 0.1, chosen_idx_by_path[p], p))
        combined_with_keys.sort(key=lambda item: (item[0], item[1]))
        structured_order = [p for _, _, p in combined_with_keys]
    else:
        structured_order = [
            p
            for p, _ in sorted(chosen_idx_by_path.items(), key=lambda kv: kv[1])
        ]

    primary_line_by_path: dict[str, str] = {}
    desc_window_by_path: dict[str, list[str]] = {}

    for norm_p, idx in chosen_idx_by_path.items():
        primary_line_by_path[norm_p] = lines[idx]
        window = [lines[idx]]
        for lookahead in range(idx + 1, min(idx + 5, len(lines))):
            if lookahead in all_subject_indices:
                break
            nxt = lines[lookahead].strip()
            if not nxt or nxt.startswith("#"):
                break
            window.append(lines[lookahead])
        desc_window_by_path[norm_p] = window

    # Fallback for any file mentioned only inline in prose (not on a numbered/bullet step line)
    fallback_offsets: list[tuple[int, str]] = []
    for path in all_repo_paths:
        slash_pos = answer_body.find(path)
        bslash_pos = answer_body.find(path.replace("/", "\\"))
        positions = [p for p in (slash_pos, bslash_pos) if p != -1]
        if not positions:
            continue
        first_pos = min(positions)
        if path not in chosen_idx_by_path:
            fallback_offsets.append((first_pos, path))
            for idx, line in enumerate(lines):
                if path in line or path.replace("/", "\\") in line:
                    primary_line_by_path[path] = line
                    win = [line]
                    if idx + 1 < len(lines) and (idx + 1) not in all_subject_indices:
                        win.append(lines[idx + 1])
                    desc_window_by_path[path] = win
                    break

    fallback_offsets.sort(key=lambda item: item[0])
    observed_order = structured_order + [p for _, p in fallback_offsets]

    joined_windows = {
        p: " ".join(w) for p, w in desc_window_by_path.items()
    }
    return primary_line_by_path, joined_windows, observed_order


def _validate_file_descriptions(
    covered_files: list[str],
    primary_lines: dict[str, str],
    desc_windows: dict[str, str],
    files_by_path: dict[str, dict[str, Any]],
    ro_by_path: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Deterministically validate per-file descriptions against AST inventory metadata
    and detect symbol/dependency misattributions or invented symbol claims.
    """
    ungrounded_descriptions: list[dict[str, Any]] = []
    unsupported_claims: list[dict[str, Any]] = []
    files_with_ast_symbol_mentions: set[str] = set()

    # Build repository-wide map of top-level classes and functions to their defining and related files
    symbol_definers: dict[str, set[str]] = {}
    symbol_related_files: dict[str, set[str]] = {}
    all_repo_valid_identifiers: set[str] = set(_PYTHON_BUILTIN_IDENTIFIERS)

    for path, rec in files_by_path.items():
        deps = set(rec.get("internal_dependencies") or [])
        importers = set(rec.get("imported_by") or [])
        exports = set(rec.get("exports") or [])

        stem = Path(path).stem
        if stem:
            all_repo_valid_identifiers.add(stem)
        for part in Path(path).parts[:-1]:
            all_repo_valid_identifiers.add(part)
        for ext in rec.get("external_dependencies") or []:
            if isinstance(ext, str):
                all_repo_valid_identifiers.add(ext)

        for cls in rec.get("classes", []):
            cname = str(cls.get("name") or "")
            if cname:
                all_repo_valid_identifiers.add(cname)
                for m in cls.get("methods", []):
                    if isinstance(m, str):
                        all_repo_valid_identifiers.add(m)
            if len(cname) >= 5:
                symbol_definers.setdefault(cname, set()).add(path)
                symbol_related_files.setdefault(cname, set()).update(
                    {path, *deps, *importers}
                )
        for fn in rec.get("functions", []):
            fname = str(fn.get("name") or "")
            if fname:
                all_repo_valid_identifiers.add(fname)
            if len(fname) >= 5 and not fname.startswith("__"):
                symbol_definers.setdefault(fname, set()).add(path)
                symbol_related_files.setdefault(fname, set()).update(
                    {path, *deps, *importers}
                )
        for const in rec.get("constants", []):
            if isinstance(const, str) and const:
                all_repo_valid_identifiers.add(const)
        for exp in exports:
            if isinstance(exp, str) and exp:
                all_repo_valid_identifiers.add(exp)
            if len(exp) >= 5:
                symbol_related_files.setdefault(exp, set()).update(
                    {path, *deps, *importers}
                )

    # Propagate symbol exports to shims/importers of re-exporting `__init__.py` packages
    for sym in list(symbol_definers.keys()):
        for path, rec in files_by_path.items():
            if sym in set(rec.get("exports") or []):
                symbol_related_files[sym].add(path)
                symbol_related_files[sym].update(rec.get("imported_by") or [])

    explicit_def_re = re.compile(
        r"\b(?:defines|implements|declares|Classes\s*:|Functions\s*:)\s*(?:the\s+)?(?:class|function|dataclass|enum)?\s*`?([A-Za-z_][A-Za-z0-9_]*)`?",
        re.IGNORECASE,
    )
    symbol_list_line_re = re.compile(
        r"(?:Classes/Functions(?:/Exports)?|Classes|Functions)\s*\*?\*?\s*:\s*(.+)",
        re.IGNORECASE,
    )
    dep_claim_re = re.compile(
        r"(?:Imports\s+from\s+repo|Internal\s+dependencies|Depends\s+on)\s*\*?\*?\s*:\s*([^\n|]+)",
        re.IGNORECASE,
    )
    call_claim_re = re.compile(
        r"\b(?:calls|invokes)\s+(?:the\s+)?(?:function\s+|method\s+|class\s+)?`([A-Za-z_][A-Za-z0-9_]*)`",
        re.IGNORECASE,
    )

    grounded_count = 0
    for path in covered_files:
        rec = files_by_path.get(path)
        if not rec:
            continue
        ro_item = ro_by_path.get(path)
        window_text = desc_windows.get(path, "")
        primary_line = primary_lines.get(path, window_text)

        # Strip the file path itself from the window text to inspect explanatory tokens
        stripped_desc = (
            window_text.replace(path, " ")
            .replace(path.replace("/", "\\"), " ")
            .strip()
        )
        desc_tokens = re.findall(r"[A-Za-z0-9_-]{3,}", stripped_desc.lower())
        non_stop_tokens = [t for t in desc_tokens if t not in _STOP_WORDS]

        anchors = _build_file_anchor_tokens(rec, ro_item)
        desc_lower = stripped_desc.lower()
        has_anchor = any(a in desc_lower for a in anchors if len(a) >= 3)

        if len(non_stop_tokens) < 2 or not has_anchor:
            ungrounded_descriptions.append(
                {
                    "file": path,
                    "reason": "File description lacks substantive AST symbol, dependency, or summary anchors",
                    "excerpt": primary_line[:160],
                }
            )
        else:
            grounded_count += 1

        file_own_symbols = (
            {str(c["name"]) for c in rec.get("classes", []) if c.get("name")}
            | {
                str(m)
                for c in rec.get("classes", [])
                for m in (c.get("methods") or [])
                if m
            }
            | {str(f["name"]) for f in rec.get("functions", []) if f.get("name")}
            | {str(e) for e in rec.get("exports", []) if e}
            | {str(k) for k in rec.get("constants", []) if k}
        )

        # Track whether at least one actual AST symbol of `path` is mentioned in its description window
        if any(sym in window_text for sym in file_own_symbols):
            files_with_ast_symbol_mentions.add(path)

        # Check for symbol misattribution in the primary description clause for `path`
        own_clause = re.split(
            r"\b(?:imported\s+by|used\s+by|called\s+by|consumed\s+by|before\s+reading|after\s+reading|unlike|whereas|precedes|follows)\b",
            window_text,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        def_clause = re.split(
            r"\b(?:imports\s+from\s+repo|imports\s*:|depends\s+on|internal\s+dependencies|delegates\s+to|invokes)\b",
            own_clause,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]

        # 1. Explicit "defines/implements <Symbol>" claim check (catches both misattributed and invented symbols)
        for m in explicit_def_re.finditer(def_clause):
            claimed_sym = m.group(1)
            if (
                claimed_sym in _PYTHON_BUILTIN_IDENTIFIERS
                or claimed_sym.lower() in _STOP_WORDS
                or len(claimed_sym) < 4
            ):
                continue
            if (
                claimed_sym in symbol_definers
                and claimed_sym not in file_own_symbols
                and path not in symbol_definers[claimed_sym]
            ):
                actual_owners = sorted(symbol_definers[claimed_sym])
                unsupported_claims.append(
                    {
                        "file": path,
                        "symbol": claimed_sym,
                        "actual_defined_in": actual_owners,
                        "reason": (
                            f"Description for '{path}' claims definition/ownership of '{claimed_sym}', "
                            f"which is actually defined in {', '.join(actual_owners)}"
                        ),
                    }
                )
            elif (
                rec.get("type") == "python"
                and claimed_sym not in file_own_symbols
                and claimed_sym not in all_repo_valid_identifiers
                and (
                    "_" in claimed_sym
                    or bool(re.match(r"^[A-Z][a-z0-9]+[A-Z]", claimed_sym))
                )
            ):
                unsupported_claims.append(
                    {
                        "file": path,
                        "symbol": claimed_sym,
                        "actual_defined_in": [],
                        "reason": (
                            f"Description for '{path}' claims non-existent symbol '{claimed_sym}' "
                            "not found in the verified AST inventory"
                        ),
                    }
                )

        # 2. Check backticked identifiers inside explicit `Classes/Functions/Exports:` lines for Python files
        if rec.get("type") == "python":
            for m_list in symbol_list_line_re.finditer(window_text):
                list_segment = m_list.group(1)
                for bt_sym in re.findall(
                    r"`([A-Za-z_][A-Za-z0-9_]*)`", list_segment
                ):
                    if (
                        bt_sym in file_own_symbols
                        or bt_sym in _PYTHON_BUILTIN_IDENTIFIERS
                        or path in symbol_related_files.get(bt_sym, set())
                        or bt_sym == Path(path).stem
                    ):
                        continue
                    if bt_sym in symbol_definers:
                        actual_owners = sorted(symbol_definers[bt_sym])
                        unsupported_claims.append(
                            {
                                "file": path,
                                "symbol": bt_sym,
                                "actual_defined_in": actual_owners,
                                "reason": (
                                    f"Symbol list for '{path}' claims '{bt_sym}', "
                                    f"which belongs to {', '.join(actual_owners)}"
                                ),
                            }
                        )
                    elif bt_sym not in all_repo_valid_identifiers:
                        unsupported_claims.append(
                            {
                                "file": path,
                                "symbol": bt_sym,
                                "actual_defined_in": [],
                                "reason": (
                                    f"Symbol list for '{path}' claims non-existent symbol '{bt_sym}' "
                                    "not found in the verified AST inventory"
                                ),
                            }
                        )

        # 3. Unrelated distinctive symbol attribution check in `def_clause`
        already_flagged = {
            c["symbol"] for c in unsupported_claims if c.get("file") == path
        }
        for sym, definers in symbol_definers.items():
            if len(sym) < 6 or sym in already_flagged or sym in file_own_symbols:
                continue
            if path in symbol_related_files.get(sym, set()):
                continue
            if re.search(
                rf"(?<![A-Za-z0-9_.]){re.escape(sym)}(?![A-Za-z0-9_])",
                def_clause,
            ):
                actual_owners = sorted(definers)
                unsupported_claims.append(
                    {
                        "file": path,
                        "symbol": sym,
                        "actual_defined_in": actual_owners,
                        "reason": (
                            f"Description for '{path}' attributes unrelated symbol '{sym}' "
                            f"(defined in {', '.join(actual_owners)} and neither defined nor imported by '{path}')"
                        ),
                    }
                )

        # 4. Contradicted internal dependency & runtime call claim checks
        if rec.get("type") == "python":
            actual_deps = set(rec.get("internal_dependencies") or [])
            actual_importers = set(rec.get("imported_by") or [])
            actual_calls = set(rec.get("all_calls") or [])
            for m_dep in dep_claim_re.finditer(window_text):
                dep_clause_text = m_dep.group(1)
                for other_p in files_by_path:
                    if other_p == path:
                        continue
                    if other_p in dep_clause_text:
                        if (
                            other_p not in actual_deps
                            and other_p not in actual_importers
                        ):
                            unsupported_claims.append(
                                {
                                    "file": path,
                                    "symbol": other_p,
                                    "actual_defined_in": [other_p],
                                    "reason": (
                                        f"Description for '{path}' claims internal dependency on '{other_p}', "
                                        f"which is not imported by '{path}' in the verified AST dependency graph"
                                    ),
                                }
                            )

            for m_call in call_claim_re.finditer(own_clause):
                called_sym = m_call.group(1)
                if (
                    called_sym in _PYTHON_BUILTIN_IDENTIFIERS
                    or called_sym in actual_calls
                    or called_sym in file_own_symbols
                    or path in symbol_related_files.get(called_sym, set())
                ):
                    continue
                if called_sym in symbol_definers:
                    actual_owners = sorted(symbol_definers[called_sym])
                    unsupported_claims.append(
                        {
                            "file": path,
                            "symbol": called_sym,
                            "actual_defined_in": actual_owners,
                            "reason": (
                                f"Description for '{path}' claims runtime call to '{called_sym}', "
                                f"which is defined in {', '.join(actual_owners)} and neither imported nor called in '{path}' AST"
                            ),
                        }
                    )
                elif (
                    called_sym not in all_repo_valid_identifiers
                    and (
                        "_" in called_sym
                        or bool(re.match(r"^[A-Z][a-z0-9]+[A-Z]", called_sym))
                    )
                ):
                    unsupported_claims.append(
                        {
                            "file": path,
                            "symbol": called_sym,
                            "actual_defined_in": [],
                            "reason": (
                                f"Description for '{path}' claims runtime call to non-existent symbol '{called_sym}' "
                                "not found in the verified AST inventory"
                            ),
                        }
                    )

    return {
        "checked_files_count": len(covered_files),
        "grounded_files_count": grounded_count,
        "files_with_ast_symbol_mentions_count": len(files_with_ast_symbol_mentions),
        "files_with_ast_symbol_mentions": sorted(files_with_ast_symbol_mentions),
        "ungrounded_descriptions": ungrounded_descriptions,
        "unsupported_claims": unsupported_claims,
    }


def _validate_reading_order(
    observed_order: list[str],
    files_by_path: dict[str, dict[str, Any]],
    ro_by_path: dict[str, dict[str, Any]],
    validate_order: bool,
) -> dict[str, Any]:
    """Validate that in the observed reading order, internal dependencies precede the
    implementation modules that import them. Entry-point and configuration files in
    Layers 0–1 are derived dynamically from the repository inventory.
    """
    pos_by_path = {path: idx + 1 for idx, path in enumerate(observed_order)}
    violations: list[dict[str, Any]] = []
    checked_edges = 0
    satisfied_edges = 0

    if not validate_order or len(observed_order) < 2:
        return {
            "checked_dependency_edges": 0,
            "satisfied_dependency_edges": 0,
            "ordering_valid": True,
            "ordering_violations": [],
            "observed_order": observed_order,
        }

    # Dynamically derive early-tier files (Layer 0 config/docs and Layer 1 entry points / top-level facade)
    early_tier_files = {
        p
        for p in files_by_path
        if _is_early_tier_file(p, files_by_path.get(p), ro_by_path.get(p))
    }

    for mod_path in observed_order:
        if mod_path in early_tier_files:
            continue
        mod_rec = files_by_path.get(mod_path)
        if not mod_rec or mod_rec.get("type") != "python":
            continue

        mod_pos = pos_by_path[mod_path]
        mod_deps = mod_rec.get("internal_dependencies") or []

        for dep_path in mod_deps:
            if dep_path in early_tier_files or dep_path not in pos_by_path:
                continue
            dep_rec = files_by_path.get(dep_path)
            if not dep_rec:
                continue
            # Skip mutual imports if any exist
            if mod_path in (dep_rec.get("internal_dependencies") or []):
                continue
            # Subpackage __init__.py re-export facades are placed at the end of their subpackage
            if (
                dep_path.endswith("/__init__.py")
                and Path(dep_path).parent == Path(mod_path).parent
            ):
                continue

            checked_edges += 1
            dep_pos = pos_by_path[dep_path]
            if dep_pos < mod_pos:
                satisfied_edges += 1
            else:
                violations.append(
                    {
                        "dependent": mod_path,
                        "dependency": dep_path,
                        "dependent_position": mod_pos,
                        "dependency_position": dep_pos,
                        "reason": (
                            f"'{mod_path}' (position {mod_pos}) imports '{dep_path}' "
                            f"(position {dep_pos}), so '{dep_path}' should precede '{mod_path}'"
                        ),
                    }
                )

    return {
        "checked_dependency_edges": checked_edges,
        "satisfied_dependency_edges": satisfied_edges,
        "ordering_valid": len(violations) == 0,
        "ordering_violations": violations,
        "observed_order": observed_order,
    }


def _validate_deliverables(
    answer_body: str,
    full_answer_text: str,
    req: dict[str, Any],
    ver: dict[str, Any],
    files_by_path: dict[str, dict[str, Any]],
    covered_files: list[str],
    omitted_files: list[str],
    observed_order: list[str],
    desc_val: dict[str, Any],
    order_val: dict[str, Any],
) -> dict[str, Any]:
    """Verify that every deliverable requested by the user's question is satisfied by
    verified AST, dependency, and inventory evidence rather than heuristic path mentions.
    """
    requested = req["requested_deliverables"]
    target_files = req["target_files"]
    covered_targets = [p for p in target_files if p in covered_files]
    queried_symbols = req["queried_symbols"]
    unmatched_targets = req["unmatched_targets"]

    missing_evidence: list[str] = []
    if not ver["inventory_verified"] or not files_by_path:
        missing_evidence.append(
            "Verified repository inventory is unavailable or empty."
        )
    if unmatched_targets:
        missing_evidence.append(
            "Requested target(s) do not exist in the verified repository inventory: "
            + ", ".join(unmatched_targets)
        )

    missing_queried_symbols = [
        sym
        for sym in queried_symbols
        if not re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(sym)}(?![A-Za-z0-9_])",
            answer_body,
        )
    ]
    if missing_queried_symbols:
        missing_evidence.append(
            "Answer omitted requested repository symbol(s): "
            + ", ".join(missing_queried_symbols)
        )

    satisfied_deliverables: list[str] = []
    unsatisfied_deliverables: list[dict[str, str]] = []

    for deliv in requested:
        if deliv == "reading_order":
            if (
                target_files
                and not omitted_files
                and order_val["ordering_valid"]
                and len(observed_order) >= min(2, len(target_files))
            ):
                satisfied_deliverables.append(deliv)
            else:
                reasons: list[str] = []
                if not target_files:
                    reasons.append("no verified target files available")
                if omitted_files:
                    reasons.append(f"{len(omitted_files)} target file(s) omitted")
                if not order_val["ordering_valid"]:
                    reasons.append(
                        f"{len(order_val['ordering_violations'])} dependency ordering violation(s)"
                    )
                if len(observed_order) < min(2, len(target_files)):
                    reasons.append("sequential reading order not present")
                unsatisfied_deliverables.append(
                    {
                        "deliverable": deliv,
                        "reason": "; ".join(reasons) or "reading order incomplete",
                    }
                )

        elif deliv == "file_descriptions":
            if (
                target_files
                and len(covered_targets) == len(target_files)
                and not omitted_files
                and not desc_val["ungrounded_descriptions"]
                and not desc_val["unsupported_claims"]
            ):
                satisfied_deliverables.append(deliv)
            else:
                reasons = []
                if not target_files:
                    reasons.append("no verified target files matched")
                if omitted_files:
                    reasons.append(f"{len(omitted_files)} file(s) omitted")
                if desc_val["ungrounded_descriptions"]:
                    reasons.append(
                        f"{len(desc_val['ungrounded_descriptions'])} file(s) lack grounded descriptions"
                    )
                if desc_val["unsupported_claims"]:
                    reasons.append(
                        f"{len(desc_val['unsupported_claims'])} unsupported symbol/dependency claim(s)"
                    )
                unsatisfied_deliverables.append(
                    {
                        "deliverable": deliv,
                        "reason": "; ".join(reasons) or "file descriptions incomplete",
                    }
                )

        elif deliv == "symbol_details":
            target_py_with_syms = [
                p
                for p in covered_targets
                if files_by_path.get(p, {}).get("type") == "python"
                and (
                    files_by_path.get(p, {}).get("classes")
                    or files_by_path.get(p, {}).get("functions")
                )
            ]
            mentioned_sym_files = set(
                desc_val.get("files_with_ast_symbol_mentions") or []
            )
            covered_with_syms = [
                p for p in target_py_with_syms if p in mentioned_sym_files
            ]
            if (
                not missing_queried_symbols
                and not desc_val["unsupported_claims"]
                and (
                    not target_py_with_syms
                    or len(covered_with_syms) >= max(1, int(0.8 * len(target_py_with_syms)))
                )
            ):
                satisfied_deliverables.append(deliv)
            else:
                reasons = []
                if missing_queried_symbols:
                    reasons.append(
                        f"missing queried symbol(s): {', '.join(missing_queried_symbols)}"
                    )
                if desc_val["unsupported_claims"]:
                    reasons.append(
                        f"{len(desc_val['unsupported_claims'])} unsupported symbol claim(s)"
                    )
                if target_py_with_syms and len(covered_with_syms) < max(
                    1, int(0.8 * len(target_py_with_syms))
                ):
                    reasons.append(
                        f"only {len(covered_with_syms)}/{len(target_py_with_syms)} Python modules include AST symbol details"
                    )
                unsatisfied_deliverables.append(
                    {
                        "deliverable": deliv,
                        "reason": "; ".join(reasons) or "symbol details incomplete",
                    }
                )

        elif deliv == "dependency_relationships":
            target_py_with_deps = [
                p
                for p in covered_targets
                if files_by_path.get(p, {}).get("type") == "python"
                and (
                    files_by_path.get(p, {}).get("internal_dependencies")
                    or files_by_path.get(p, {}).get("imported_by")
                )
            ]
            body_low = answer_body.lower()
            has_dep_evidence = any(
                kw in body_low
                for kw in (
                    "import",
                    "depend",
                    "layer",
                    "precede",
                    "delegate",
                    "invoke",
                )
            )
            if (
                order_val["ordering_valid"]
                and not desc_val["unsupported_claims"]
                and (not target_py_with_deps or has_dep_evidence)
            ):
                satisfied_deliverables.append(deliv)
            else:
                reasons = []
                if not order_val["ordering_valid"]:
                    reasons.append("dependency ordering violated")
                if desc_val["unsupported_claims"]:
                    reasons.append("contradicted dependency/symbol claim detected")
                if target_py_with_deps and not has_dep_evidence:
                    reasons.append(
                        "answer does not describe internal import/dependency relationships"
                    )
                unsatisfied_deliverables.append(
                    {
                        "deliverable": deliv,
                        "reason": "; ".join(reasons)
                        or "dependency relationships incomplete",
                    }
                )

        elif deliv == "execution_flow":
            exec_flows = req.get("execution_flows") or []
            call_edges = req.get("call_graph_edges") or []
            target_set = set(target_files)
            relevant_edges = [
                e
                for e in call_edges
                if e.get("caller_file") in target_set
                or e.get("callee_file") in target_set
            ]
            has_flow_evidence = True
            if relevant_edges or exec_flows:
                has_flow_evidence = any(
                    (
                        e.get("caller_file", "") in answer_body
                        and e.get("callee_file", "") in answer_body
                    )
                    or any(
                        sym in answer_body
                        for sym in (e.get("invoked_symbols") or [])
                    )
                    for e in (relevant_edges or call_edges)
                )
            if (
                order_val["ordering_valid"]
                and not desc_val["unsupported_claims"]
                and has_flow_evidence
            ):
                satisfied_deliverables.append(deliv)
            else:
                reasons = []
                if not order_val["ordering_valid"]:
                    reasons.append("dependency ordering violated")
                if desc_val["unsupported_claims"]:
                    reasons.append("contradicted call/symbol claim detected")
                if not has_flow_evidence:
                    reasons.append(
                        "answer does not trace verified AST call-graph edges or entry-point execution flow"
                    )
                unsatisfied_deliverables.append(
                    {
                        "deliverable": deliv,
                        "reason": "; ".join(reasons) or "execution flow incomplete",
                    }
                )

        elif deliv == "exclusions_or_limitations":
            has_exclusions_in_repo = bool(
                req["excluded_dirs"] or req["excluded_files"]
            )
            combined_low = (full_answer_text or answer_body).lower()
            has_excl_mention = (
                not has_exclusions_in_repo
                or "excluded" in combined_low
                or "ignore" in combined_low
                or any(d.lower() in combined_low for d in req["excluded_dirs"])
                or any(f.lower() in combined_low for f in req["excluded_files"])
            )
            if has_excl_mention:
                satisfied_deliverables.append(deliv)
            else:
                unsatisfied_deliverables.append(
                    {
                        "deliverable": deliv,
                        "reason": "excluded directories/files were not reported",
                    }
                )

    return {
        "scope": req["scope"],
        "requested_deliverables": requested,
        "satisfied_deliverables": satisfied_deliverables,
        "unsatisfied_deliverables": unsatisfied_deliverables,
        "deliverables_satisfied": (
            len(unsatisfied_deliverables) == 0 and len(missing_evidence) == 0
        ),
        "target_files": target_files,
        "target_files_count": len(target_files),
        "covered_target_files_count": len(covered_targets),
        "queried_symbols": queried_symbols,
        "missing_queried_symbols": missing_queried_symbols,
        "unmatched_targets": unmatched_targets,
        "missing_evidence": missing_evidence,
    }


def validate_query_answer(
    answer: str,
    query: str,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate an LLM answer against the user's requested deliverables and scope,
    the verified repository file inventory, AST symbol/dependency evidence,
    reading-order dependency precedence, and index freshness status.
    """
    ver = _evaluate_source_verification(context)
    req = analyze_query_requirements(query=query, context=context)

    inv = (
        context.get("repository_inventory")
        if isinstance(context, dict)
        and isinstance(context.get("repository_inventory"), dict)
        else {}
    )
    files_list = inv.get("files") if isinstance(inv.get("files"), list) else []
    reading_order = (
        inv.get("reading_order") if isinstance(inv.get("reading_order"), list) else []
    )

    files_by_path: dict[str, dict[str, Any]] = {
        str(f["path"]): f
        for f in files_list
        for _ in [None]
        if isinstance(f, dict) and f.get("path")
    }
    ro_by_path: dict[str, dict[str, Any]] = {
        str(item["path"]): item
        for item in reading_order
        if isinstance(item, dict) and item.get("path")
    }

    all_repo_paths: list[str] = list(ro_by_path.keys()) or list(files_by_path.keys())
    for p in files_by_path:
        if p not in all_repo_paths:
            all_repo_paths.append(p)

    py_repo_paths: list[str] = [
        p for p, f in files_by_path.items() if f.get("type") == "python"
    ]
    valid_path_set = set(all_repo_paths)
    valid_basenames = {p.split("/")[-1] for p in all_repo_paths}
    top_pkg_prefixes = {
        p.split("/")[0] for p in all_repo_paths if "/" in p
    }

    answer_body = _strip_diagnostic_footers(answer or "")
    covered_files: list[str] = []
    omitted_files: list[str] = []

    for path in all_repo_paths:
        if path in answer_body or path.replace("/", "\\") in answer_body:
            covered_files.append(path)

    target_files = req["target_files"]
    for p in target_files:
        if p not in covered_files:
            omitted_files.append(p)

    # Check for hallucinated or deleted .py path references in answer_body (repository-agnostic)
    hallucinated_files: list[str] = []
    unmatched_set = set(req["unmatched_targets"])
    candidate_py_refs = re.findall(
        r"(?<![A-Za-z0-9_./\\-])((?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.py|[A-Za-z0-9_-]+\.py)\b",
        answer_body,
    )
    for ref in candidate_py_refs:
        norm_ref = ref.replace("\\", "/").lstrip("./")
        if norm_ref in unmatched_set:
            # If the user asked about a non-existent path, it is already tracked in `unmatched_targets` / `missing_evidence`
            continue
        if "/" in norm_ref:
            ref_top = norm_ref.split("/")[0]
            if (
                (not top_pkg_prefixes or ref_top in top_pkg_prefixes or ref_top in ("src", "app", "lib", "tests"))
                and norm_ref not in valid_path_set
                and norm_ref not in hallucinated_files
            ):
                hallucinated_files.append(norm_ref)
        else:
            if norm_ref not in valid_basenames and norm_ref not in hallucinated_files:
                hallucinated_files.append(norm_ref)

    # Extract per-file descriptions and observed reading order
    primary_lines, desc_windows, observed_order = _extract_file_entry_lines(
        answer_body=answer_body,
        all_repo_paths=all_repo_paths,
    )

    desc_val = _validate_file_descriptions(
        covered_files=covered_files,
        primary_lines=primary_lines,
        desc_windows=desc_windows,
        files_by_path=files_by_path,
        ro_by_path=ro_by_path,
    )
    validate_order_flag = bool(
        req["is_repo_wide"] or "reading_order" in req["requested_deliverables"]
    )
    order_val = _validate_reading_order(
        observed_order=observed_order,
        files_by_path=files_by_path,
        ro_by_path=ro_by_path,
        validate_order=validate_order_flag,
    )
    deliverable_val = _validate_deliverables(
        answer_body=answer_body,
        full_answer_text=answer or "",
        req=req,
        ver=ver,
        files_by_path=files_by_path,
        covered_files=covered_files,
        omitted_files=omitted_files,
        observed_order=observed_order,
        desc_val=desc_val,
        order_val=order_val,
    )

    unsupported_claims = desc_val["unsupported_claims"]
    ungrounded_descriptions = desc_val["ungrounded_descriptions"]
    ordering_violations = order_val["ordering_violations"]
    unsatisfied_deliverables = deliverable_val["unsatisfied_deliverables"]
    missing_evidence = deliverable_val["missing_evidence"]

    covered_py = [p for p in covered_files if p in py_repo_paths]
    warnings: list[str] = []
    uncertainty_notes: list[str] = []

    if ver["errors"]:
        warnings.extend(ver["errors"])
        uncertainty_notes.append(
            "One or more knowledge retrieval or index verification steps reported errors."
        )
    if not ver["graphify_fresh"]:
        warnings.append(
            "Graphify index was not verified as fresh with matching SHA-256 and AST symbol fingerprints."
        )
        uncertainty_notes.append(
            "Graphify structural graph context could not be verified against current source fingerprints."
        )
    if not ver["memory_fresh"]:
        warnings.append(
            "Codebase Memory MCP index was not verified as fresh with matching SHA-256 and AST symbol fingerprints."
        )
        uncertainty_notes.append(
            "Codebase Memory MCP architecture/symbol index could not be verified against current source fingerprints."
        )
    if missing_evidence:
        warnings.extend(missing_evidence)
        uncertainty_notes.extend(missing_evidence)
    if omitted_files:
        warnings.append(
            f"Answer omitted {len(omitted_files)} required file(s) from the verified inventory: {', '.join(omitted_files)}"
        )
        uncertainty_notes.append(
            f"Omitted repository files required deterministic supplementation ({len(omitted_files)} file(s))."
        )
    if hallucinated_files:
        warnings.append(
            f"Answer referenced {len(hallucinated_files)} non-existent/deleted file path(s): {', '.join(hallucinated_files)}"
        )
        uncertainty_notes.append(
            f"Detected non-existent file references: {', '.join(hallucinated_files)}."
        )
    if unsupported_claims:
        warnings.append(
            f"Answer contained {len(unsupported_claims)} unsupported symbol/dependency claim(s): "
            + "; ".join(c["reason"] for c in unsupported_claims[:5])
        )
        uncertainty_notes.append(
            "Detected symbol ownership or dependency claims contradicting the deterministic AST inventory."
        )
    if ungrounded_descriptions:
        warnings.append(
            f"Answer contained {len(ungrounded_descriptions)} file mention(s) without AST/summary-grounded descriptions: "
            + ", ".join(u["file"] for u in ungrounded_descriptions[:8])
        )
        uncertainty_notes.append(
            "One or more mentioned files lacked AST symbol, dependency, or summary-grounded descriptions."
        )
    if ordering_violations:
        warnings.append(
            f"Reading order contained {len(ordering_violations)} dependency precedence violation(s): "
            + "; ".join(v["reason"] for v in ordering_violations[:5])
        )
        uncertainty_notes.append(
            "One or more dependent modules appeared before their internal dependencies in the reading order."
        )
    if unsatisfied_deliverables:
        warnings.append(
            "Answer did not satisfy all requested deliverable(s): "
            + "; ".join(
                f"{u['deliverable']} ({u['reason']})"
                for u in unsatisfied_deliverables
            )
        )
        uncertainty_notes.append(
            "One or more user-requested deliverables were incomplete or unverified."
        )

    epistemic_limitations: list[str] = [
        "Deterministic validation proves repository root identity, SHA-256 file fingerprints, AST symbol/signature/call/exception/import inventory, cross-module AST call-graph edges, entry-point execution flow traces, file path existence, AST/summary grounding in file descriptions, absence of cross-file symbol/call/dependency misattribution, topological dependency precedence across implementation modules, and coverage of requested deliverables.",
        "Deterministic validation cannot prove unbounded natural-language prose equivalence or dynamic runtime reflection/external I/O behavior beyond static AST, call-graph, and docstring evidence.",
    ]

    critical_retrieval_failed = bool(
        not ver["inventory_verified"]
        or not ver["graphify_fresh"]
        or not ver["memory_fresh"]
        or len(ver["errors"]) > 0
    )

    is_complete = bool(
        answer_body.strip()
        and not critical_retrieval_failed
        and len(target_files) > 0
        and len(covered_files) > 0
        and not missing_evidence
        and not omitted_files
        and not hallucinated_files
        and not unsupported_claims
        and not ungrounded_descriptions
        and not ordering_violations
        and deliverable_val["deliverables_satisfied"]
    )

    if is_complete:
        status_label = "verified_complete"
    elif critical_retrieval_failed:
        status_label = "degraded"
    else:
        status_label = "incomplete"

    return {
        "complete": is_complete,
        "status": status_label,
        "query_scope": req["scope"],
        "is_repo_wide_query": req["is_repo_wide"],
        "requested_deliverables": req["requested_deliverables"],
        "deliverable_validation": deliverable_val,
        "missing_evidence": missing_evidence,
        "unmatched_targets": req["unmatched_targets"],
        "retrieval_healthy": ver["all_verified"],
        "critical_retrieval_failed": critical_retrieval_failed,
        "inventory_verified": ver["inventory_verified"],
        "graphify_verified": ver["graphify_verified"],
        "graphify_fresh": ver["graphify_fresh"],
        "graphify_fingerprint_verified": ver["graphify_fingerprint_verified"],
        "graphify_ast_symbols_verified": ver["graphify_ast_symbols_verified"],
        "memory_verified": ver["memory_verified"],
        "memory_fresh": ver["memory_fresh"],
        "memory_fingerprint_verified": ver["memory_fingerprint_verified"],
        "total_repo_files": len(all_repo_paths),
        "target_files_count": len(target_files),
        "covered_files_count": len(covered_files),
        "total_python_files": len(py_repo_paths),
        "covered_python_files_count": len(covered_py),
        "covered_files": covered_files,
        "omitted_files": omitted_files,
        "hallucinated_files": hallucinated_files,
        "description_validation": desc_val,
        "ordering_validation": order_val,
        "unsupported_claims": unsupported_claims,
        "ungrounded_descriptions": ungrounded_descriptions,
        "ordering_violations": ordering_violations,
        "unsatisfied_deliverables": unsatisfied_deliverables,
        "uncertainty_notes": uncertainty_notes,
        "epistemic_limitations": epistemic_limitations,
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
    """Validate and finalize an answer against verified inventory and dependency evidence.
    - In strict mode (`strict=True`), raises `RuntimeError` if critical retrieval/indexing failed,
      required evidence is missing/unverifiable, or the finalized answer is incomplete.
    - In non-strict mode (`strict=False`), supplements missing file/symbol/dependency facts only
      from the verified AST inventory when safe (no contradictions), and explicitly marks any
      remaining incomplete or degraded result with a visible warning banner and diagnostics.
    """
    val = validation or validate_query_answer(
        answer=answer, query=query, context=context
    )

    has_contradictory_or_missing_evidence = bool(
        val["critical_retrieval_failed"]
        or not (answer and answer.strip())
        or val["hallucinated_files"]
        or val["unsupported_claims"]
        or val["ordering_violations"]
        or val.get("missing_evidence")
    )

    if strict and has_contradictory_or_missing_evidence:
        err_msg = (
            "; ".join(val["warnings"])
            or "Critical retrieval/indexing failure, missing evidence, unsupported claim, ordering violation, or empty answer"
        )
        raise RuntimeError(
            f"Strict mode: refusing to present answer as complete due to verification failure: {err_msg}"
        )

    output_blocks: list[str] = []

    inv = (
        context.get("repository_inventory")
        if isinstance(context, dict)
        and isinstance(context.get("repository_inventory"), dict)
        else {}
    )
    files_list = inv.get("files") if isinstance(inv.get("files"), list) else []
    reading_order = (
        inv.get("reading_order") if isinstance(inv.get("reading_order"), list) else []
    )
    files_by_path: dict[str, dict[str, Any]] = {
        str(f["path"]): f
        for f in files_list
        if isinstance(f, dict) and f.get("path")
    }
    ro_by_path = {
        str(item["path"]): item
        for item in reading_order
        if isinstance(item, dict) and item.get("path")
    }

    body_text = (answer or "").strip()

    # Supplement from the verified AST inventory ONLY when the inventory is verified and no contradictory claims exist
    can_supplement = bool(
        body_text
        and val["inventory_verified"]
        and not val["hallucinated_files"]
        and not val["unsupported_claims"]
        and not val["ordering_violations"]
        and not val.get("unmatched_targets")
    )

    if can_supplement and val["omitted_files"] and (ro_by_path or files_by_path):
        supp_lines: list[str] = [
            "### Verified Inventory Supplement (Additional Repository Files)",
            "The following repository files were verified by the deterministic AST inventory and are included for complete coverage:",
        ]
        for idx, path in enumerate(val["omitted_files"], start=1):
            item = ro_by_path.get(path) or {}
            rec = files_by_path.get(path) or {}
            step_num = item.get("step") or idx
            layer_name = item.get("layer") or "Repository"
            summary_str = item.get("summary") or rec.get("summary") or ""
            deps = item.get("depends_on") or rec.get("internal_dependencies") or []
            deps_str = (
                ", ".join(f"`{d}`" for d in deps)
                if deps
                else "None (leaf/standalone)"
            )
            cls_list = item.get("classes") or [
                c["name"] for c in rec.get("classes", []) if c.get("name")
            ]
            fn_list = item.get("functions") or [
                f["name"] for f in rec.get("functions", []) if f.get("name")
            ]
            sym_parts: list[str] = []
            if cls_list:
                sym_parts.append(f"Classes: {', '.join(f'`{c}`' for c in cls_list)}")
            if fn_list:
                sym_parts.append(
                    f"Functions: {', '.join(f'`{f}`' for f in fn_list[:8])}"
                )
            sym_str = f" [{' | '.join(sym_parts)}]" if sym_parts else ""
            supp_lines.append(
                f"- **`{path}`** (*{layer_name}*, step {step_num}): "
                f"{summary_str}{sym_str} (Imports from repo: {deps_str})"
            )
        body_text = body_text + "\n\n---\n" + "\n".join(supp_lines)

    # Re-evaluate validation after any safe deterministic inventory supplement
    post_val = validate_query_answer(answer=body_text, query=query, context=context)

    if strict and not post_val["complete"]:
        err_msg = (
            "; ".join(post_val["warnings"])
            or "Answer failed post-validation completeness checks"
        )
        raise RuntimeError(
            f"Strict mode: refusing to present incomplete or unverified answer: {err_msg}"
        )

    if not post_val["complete"]:
        reasons = (
            "; ".join(post_val["warnings"])
            or "Unverified sources or incomplete evidence"
        )
        output_blocks.append(
            f"> [!WARNING] **DEGRADED / PARTIALLY VERIFIED RESULT** (`status={post_val['status']}`): "
            f"This response could not be verified as complete. Reasons: {reasons}"
        )

    output_blocks.append(body_text)

    if (
        post_val["hallucinated_files"]
        or post_val["unsupported_claims"]
        or post_val["ordering_violations"]
        or post_val["ungrounded_descriptions"]
        or post_val.get("missing_evidence")
        or post_val.get("unsatisfied_deliverables")
    ):
        warn_lines: list[str] = ["---", "### Evidence Grounding Warning"]
        if post_val.get("missing_evidence"):
            warn_lines.append(
                "- **Missing or Unverifiable Evidence**: "
                + "; ".join(post_val["missing_evidence"])
            )
        if post_val["hallucinated_files"]:
            warn_lines.append(
                "- **Non-Existent / Deleted File Paths Mentioned**: "
                + ", ".join(f"`{h}`" for h in post_val["hallucinated_files"])
            )
        if post_val["unsupported_claims"]:
            warn_lines.append(
                "- **Unsupported Symbol / Dependency Claims**: "
                + "; ".join(c["reason"] for c in post_val["unsupported_claims"])
            )
        if post_val["ordering_violations"]:
            warn_lines.append(
                "- **Dependency Precedence Violations in Reading Order**: "
                + "; ".join(v["reason"] for v in post_val["ordering_violations"])
            )
        if post_val["ungrounded_descriptions"]:
            warn_lines.append(
                "- **Files Mentioned Without Substantive AST/Summary Grounding**: "
                + ", ".join(
                    f"`{u['file']}`" for u in post_val["ungrounded_descriptions"]
                )
            )
        if post_val.get("unsatisfied_deliverables"):
            warn_lines.append(
                "- **Unsatisfied Requested Deliverables**: "
                + "; ".join(
                    f"`{u['deliverable']}` ({u['reason']})"
                    for u in post_val["unsatisfied_deliverables"]
                )
            )
        output_blocks.append("\n".join(warn_lines))

    if (
        include_diagnostics
        or not post_val["complete"]
        or post_val["retrieval_errors"]
        or post_val["critical_retrieval_failed"]
    ):
        coverage = (
            context.get("coverage")
            if isinstance(context, dict) and isinstance(context.get("coverage"), dict)
            else {}
        )
        index_status = (
            context.get("index_status")
            if isinstance(context, dict)
            and isinstance(context.get("index_status"), dict)
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
        excl_dirs = [
            d["path"]
            for d in coverage.get("excluded_dirs", [])
            if isinstance(d, dict)
        ]
        excl_files = [
            f"{f['path']} ({f['reason']})"
            for f in coverage.get("excluded_files", [])
            if isinstance(f, dict)
        ]
        desc_v = post_val["description_validation"]
        ord_v = post_val["ordering_validation"]
        deliv_v = post_val["deliverable_validation"]

        diag_lines: list[str] = [
            "---",
            "### Verification & Coverage Diagnostics",
            f"- **Completeness Status**: `{post_val['status']}` (`complete={post_val['complete']}`)",
            f"- **Query Scope & Deliverables**: scope=`{post_val['query_scope']}`, "
            f"requested={deliv_v['requested_deliverables']}, "
            f"satisfied={deliv_v['satisfied_deliverables']}, "
            f"unsatisfied={[u['deliverable'] for u in deliv_v['unsatisfied_deliverables']]}",
            f"- **Repository Root**: `{context.get('repo_root', '') if isinstance(context, dict) else ''}`",
            f"- **Graphify Index**: verified=`{post_val['graphify_verified']}`, fresh=`{post_val['graphify_fresh']}`, "
            f"sha256_fingerprint_verified=`{post_val['graphify_fingerprint_verified']}`, "
            f"ast_symbols_verified=`{post_val['graphify_ast_symbols_verified']}`, "
            f"nodes={graph_st.get('node_count', 0)}, edges={graph_st.get('edge_count', 0)}, "
            f"indexed_files={len(graph_st.get('indexed_files', []))}, "
            f"deleted_files={len(graph_st.get('deleted_files', []))}, "
            f"missing_files={len(graph_st.get('missing_files', []))}, "
            f"stale_files={len(graph_st.get('stale_files', []))}",
            f"- **Codebase Memory MCP Index**: project=`{mem_st.get('project', '')}`, "
            f"verified=`{post_val['memory_verified']}`, fresh=`{post_val['memory_fresh']}`, "
            f"sha256_fingerprint_verified=`{post_val['memory_fingerprint_verified']}`, "
            f"indexed_files={len(mem_st.get('indexed_files', []))}, "
            f"deleted_files={len(mem_st.get('deleted_files', []))}, "
            f"missing_files={len(mem_st.get('missing_files', []))}, "
            f"stale_files={len(mem_st.get('stale_files', []))}",
            f"- **Repository Inventory Coverage**: {post_val['covered_files_count']}/{post_val['total_repo_files']} repo files covered "
            f"(target files: {deliv_v['covered_target_files_count']}/{deliv_v['target_files_count']}, "
            f"{post_val['covered_python_files_count']}/{post_val['total_python_files']} Python modules, "
            f"{coverage.get('config_doc_files', 0)} config/doc files; "
            f"initial LLM omissions supplemented={len(val['omitted_files']) if can_supplement else 0}; "
            f"remaining omissions={len(post_val['omitted_files'])}; "
            f"hallucinated paths={len(post_val['hallucinated_files'])})",
            f"- **Evidence & Description Validation**: {desc_v['grounded_files_count']}/{desc_v['checked_files_count']} covered files grounded in AST/summary evidence "
            f"(unsupported symbol/dependency claims={len(post_val['unsupported_claims'])}, "
            f"ungrounded descriptions={len(post_val['ungrounded_descriptions'])})",
            f"- **Reading-Order Dependency Validation**: {ord_v['satisfied_dependency_edges']}/{ord_v['checked_dependency_edges']} internal dependency edges satisfied "
            f"(ordering_valid=`{ord_v['ordering_valid']}`, violations={len(post_val['ordering_violations'])})",
            f"- **Excluded Directories**: {', '.join(f'`{d}`' for d in excl_dirs) or 'None'}",
            f"- **Excluded Files**: {', '.join(f'`{f}`' for f in excl_files) or 'None'}",
            f"- **Retrieval Errors**: {'; '.join(post_val['retrieval_errors']) if post_val['retrieval_errors'] else 'None (0 errors)'}",
            f"- **Epistemic Scope**: {post_val['epistemic_limitations'][1]}",
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
        raise PermissionError(
            "Plan must be explicitly approved before executing changes."
        )


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
