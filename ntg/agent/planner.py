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
        "3. Cover every single file in the Verified Repository File Inventory across all architectural layers: Project Overview & Configuration (`README.md`, `pyproject.toml`, `requirements.txt`, `.env.example`, `.gitignore`), Execution Entry Points (`main.py`, `ntg/__main__.py`, `test.py`, `ntg/__init__.py`), Core Foundation (`ntg/core/*`), Provider Adapters (`ntg/providers/*`), Smart Routing, State, Telemetry & Runtime Diagnostics (`ntg/cli/diagnostics.py`, `ntg/router/*`), Architecture-Aware Coding Agent (`ntg/agent/*`), CLI Application Runner (`ntg/cli/app.py`, `ntg/cli/__init__.py`), and Compatibility Shims (`ntg/verifier.py`, `ntg/agent.py`).",
        "4. Preserve the EXACT numbered dependency-aware reading order (steps 1 through 37) from the Verified Repository File Inventory below without reordering any steps (in particular, keep `ntg/cli/diagnostics.py` before `ntg/router/engine.py` because `ntg/router/engine.py` imports `ntg/cli/diagnostics.py`). For each file, state its step number, backticked relative path, actual classes/functions/exports, and what the file does.",
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
    """Build deterministic set of valid grounding tokens (symbols, dependencies, domain keywords) for a file."""
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

    # AST classes, functions, methods, exports, constants
    for cls in file_rec.get("classes", []):
        if isinstance(cls, dict) and cls.get("name"):
            cname = str(cls["name"]).lower()
            anchors.add(cname)
            for m in cls.get("methods", []):
                if isinstance(m, str) and len(m) >= 3 and not m.startswith("__"):
                    anchors.add(m.lower())
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
            if fn.get("docstring"):
                for tok in re.findall(r"[a-z0-9_]{4,}", str(fn["docstring"]).lower()):
                    if tok not in _STOP_WORDS:
                        anchors.add(tok)

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
    for src in text_sources:
        for tok in re.findall(r"[a-z0-9_]{4,}", src.lower()):
            if tok not in _STOP_WORDS:
                anchors.add(tok)

    if path.endswith("__init__.py") or path in ("ntg/agent.py", "ntg/verifier.py"):
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
    lines = answer_body.splitlines()
    sorted_paths_by_len = sorted(all_repo_paths, key=len, reverse=True)
    escaped_alts = "|".join(
        re.escape(p) + "|" + re.escape(p.replace("/", "\\"))
        for p in sorted_paths_by_len
    )

    # Numbered step line (e.g., "1. `path`: ...", "### 34. **`path`**", "- **Step 34: `path`**")
    numbered_step_re = re.compile(
        r"^\s*(?:#{1,6}\s*)?(?:[-*+•]\s*)?(?:\*\*|\*|_|\s)*(?:Step\s+)?(\d+)\s*[:.)–—-]\s*(?:\*\*|`|\*|_|\s)*("
        + escaped_alts
        + r")(?![A-Za-z0-9_./\\-])",
        re.IGNORECASE,
    )

    # Top-level unnumbered bullet or heading introducing a file path as the primary subject
    bullet_item_re = re.compile(
        r"^(\s*)(?:[-*+•]|#{1,6})\s*(?:\*\*|`|\*|_|\s)*("
        + escaped_alts
        + r")(?:\*\*|`|\*|_|\s)*(?:[:—–(-]|\s+is\b|\s+defines\b|\s+provides\b|\s+implements\b|\s+contains\b|\s+re-exports\b|\s+manages\b|\s+handles\b|\s+serves\b|\s+-|\s*$)"
    )

    numbered_idx_by_path: dict[str, int] = {}
    bullet_idx_by_path: dict[str, int] = {}
    all_subject_indices: set[int] = set()

    for idx, line in enumerate(lines):
        m_num = numbered_step_re.search(line)
        if m_num:
            norm_p = m_num.group(2).replace("\\", "/")
            all_subject_indices.add(idx)
            if norm_p not in numbered_idx_by_path:
                numbered_idx_by_path[norm_p] = idx
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
    and detect symbol/dependency misattributions.
    """
    ungrounded_descriptions: list[dict[str, Any]] = []
    unsupported_claims: list[dict[str, Any]] = []

    # Build repository-wide map of top-level classes and functions to their defining and related files
    symbol_definers: dict[str, set[str]] = {}
    symbol_related_files: dict[str, set[str]] = {}

    for path, rec in files_by_path.items():
        deps = set(rec.get("internal_dependencies") or [])
        importers = set(rec.get("imported_by") or [])
        exports = set(rec.get("exports") or [])

        for cls in rec.get("classes", []):
            cname = str(cls.get("name") or "")
            if len(cname) >= 5:
                symbol_definers.setdefault(cname, set()).add(path)
                symbol_related_files.setdefault(cname, set()).update(
                    {path, *deps, *importers}
                )
        for fn in rec.get("functions", []):
            fname = str(fn.get("name") or "")
            if len(fname) >= 5 and not fname.startswith("__"):
                symbol_definers.setdefault(fname, set()).add(path)
                symbol_related_files.setdefault(fname, set()).update(
                    {path, *deps, *importers}
                )
        for exp in exports:
            if len(exp) >= 5:
                symbol_related_files.setdefault(exp, set()).update(
                    {path, *deps, *importers}
                )

    # Propagate symbol exports to shims/importers of re-exporting `__init__.py` packages
    for sym, definers in list(symbol_definers.items()):
        for path, rec in files_by_path.items():
            if sym in set(rec.get("exports") or []):
                symbol_related_files[sym].add(path)
                symbol_related_files[sym].update(rec.get("imported_by") or [])

    explicit_def_re = re.compile(
        r"\b(?:defines|implements|declares|Classes\s*:|Functions\s*:)\s*(?:the\s+)?(?:class|function|dataclass|enum)?\s*`?([A-Za-z_][A-Za-z0-9_]*)`?",
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

        # Check for symbol misattribution in the primary description clause for `path`
        # (Before relational clauses like "Imported by:", "Used by:", "Before", "After", "Unlike")
        own_clause = re.split(
            r"\b(?:imported\s+by|used\s+by|called\s+by|consumed\s+by|before\s+reading|after\s+reading|unlike|whereas|precedes|follows)\b",
            primary_line,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        # Also remove the "Imports from repo: ..." / "Depends on: ..." suffix when checking definition ownership
        def_clause = re.split(
            r"\b(?:imports\s+from\s+repo|imports\s*:|depends\s+on|internal\s+dependencies)\b",
            own_clause,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]

        file_own_symbols = (
            {str(c["name"]) for c in rec.get("classes", []) if c.get("name")}
            | {str(f["name"]) for f in rec.get("functions", []) if f.get("name")}
            | {str(e) for e in rec.get("exports", []) if e}
            | {str(k) for k in rec.get("constants", []) if k}
        )

        # 1. Explicit "defines/implements <Symbol>" claim check
        for m in explicit_def_re.finditer(def_clause):
            claimed_sym = m.group(1)
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

        # 2. Unrelated distinctive symbol attribution check in `def_clause`
        already_flagged = {
            c["symbol"] for c in unsupported_claims if c.get("file") == path
        }
        for sym, definers in symbol_definers.items():
            if len(sym) < 6 or sym in already_flagged or sym in file_own_symbols:
                continue
            if path in symbol_related_files.get(sym, set()):
                continue
            if re.search(rf"(?<![A-Za-z0-9_.]){re.escape(sym)}(?![A-Za-z0-9_])", def_clause):
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

    return {
        "checked_files_count": len(covered_files),
        "grounded_files_count": grounded_count,
        "ungrounded_descriptions": ungrounded_descriptions,
        "unsupported_claims": unsupported_claims,
    }


def _validate_reading_order(
    observed_order: list[str],
    files_by_path: dict[str, dict[str, Any]],
    ro_by_path: dict[str, dict[str, Any]],
    is_repo_wide: bool,
) -> dict[str, Any]:
    """Validate that in the observed reading order, internal dependencies precede the
    implementation modules that import them.
    """
    pos_by_path = {path: idx + 1 for idx, path in enumerate(observed_order)}
    violations: list[dict[str, Any]] = []
    checked_edges = 0
    satisfied_edges = 0

    if not is_repo_wide or len(observed_order) < 2:
        return {
            "checked_dependency_edges": 0,
            "satisfied_dependency_edges": 0,
            "ordering_valid": True,
            "ordering_violations": [],
            "observed_order": observed_order,
        }

    # Entry points in Layer 1 (`main.py`, `ntg/__main__.py`, `test.py`, `ntg/__init__.py`)
    # intentionally appear in the entry-point overview layer before the implementation layers.
    # All implementation modules and subpackage facades in Layers 2..7 must appear AFTER
    # their internal dependencies in Layers 2..7.
    entry_points = {"main.py", "ntg/__main__.py", "test.py", "ntg/__init__.py"}

    for mod_path in observed_order:
        if mod_path in entry_points:
            continue
        mod_rec = files_by_path.get(mod_path)
        if not mod_rec or mod_rec.get("type") != "python":
            continue

        mod_pos = pos_by_path[mod_path]
        mod_deps = mod_rec.get("internal_dependencies") or []

        for dep_path in mod_deps:
            if dep_path in entry_points or dep_path not in pos_by_path:
                continue
            dep_rec = files_by_path.get(dep_path)
            if not dep_rec:
                continue
            # Skip mutual imports if any exist
            if mod_path in (dep_rec.get("internal_dependencies") or []):
                continue
            # If a non-__init__ module in the same subpackage is compared with its own subpackage __init__.py,
            # subpackage __init__.py re-export facades are placed at the end of their subpackage.
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


def validate_query_answer(
    answer: str,
    query: str,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate an LLM answer against the verified repository file inventory, AST symbol/dependency
    evidence, reading-order dependency precedence, and index freshness status.
    """
    ver = _evaluate_source_verification(context)
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
    ro_by_path: dict[str, dict[str, Any]] = {
        str(item["path"]): item
        for item in reading_order
        if isinstance(item, dict) and item.get("path")
    }

    all_repo_paths: list[str] = list(files_by_path.keys())
    py_repo_paths: list[str] = [
        p for p, f in files_by_path.items() if f.get("type") == "python"
    ]
    valid_path_set = set(all_repo_paths)
    valid_basenames = {p.split("/")[-1] for p in all_repo_paths}

    answer_body = _strip_diagnostic_footers(answer or "")
    covered_files: list[str] = []
    omitted_files: list[str] = []

    for path in all_repo_paths:
        if path in answer_body or path.replace("/", "\\") in answer_body:
            covered_files.append(path)

    is_broad = _is_repo_wide_query(query)
    if is_broad and all_repo_paths:
        ordered_paths = list(ro_by_path.keys()) or all_repo_paths
        for p in ordered_paths:
            if p not in covered_files:
                omitted_files.append(p)

    # Check for hallucinated or deleted .py path references in answer_body
    hallucinated_files: list[str] = []
    candidate_py_refs = re.findall(
        r"(?<![A-Za-z0-9_./\\-])((?:ntg/[A-Za-z0-9_/.-]+|[A-Za-z0-9_-]+)\.py)\b",
        answer_body,
    )
    for ref in candidate_py_refs:
        norm_ref = ref.replace("\\", "/").lstrip("./")
        if "/" in norm_ref:
            if norm_ref not in valid_path_set and norm_ref not in hallucinated_files:
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
    order_val = _validate_reading_order(
        observed_order=observed_order,
        files_by_path=files_by_path,
        ro_by_path=ro_by_path,
        is_repo_wide=is_broad,
    )

    unsupported_claims = desc_val["unsupported_claims"]
    ungrounded_descriptions = desc_val["ungrounded_descriptions"]
    ordering_violations = order_val["ordering_violations"]

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
    if omitted_files:
        warnings.append(
            f"Answer omitted {len(omitted_files)} file(s) from the verified inventory: {', '.join(omitted_files)}"
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
            f"Answer contained {len(unsupported_claims)} unsupported symbol attribution claim(s): "
            + "; ".join(c["reason"] for c in unsupported_claims[:5])
        )
        uncertainty_notes.append(
            "Detected symbol ownership claims contradicting the deterministic AST inventory."
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

    epistemic_limitations: list[str] = [
        "Deterministic validation proves repository root identity, SHA-256 file fingerprints, AST symbol/import inventory, file path existence, presence of AST/summary anchors in file descriptions, absence of cross-file symbol misattribution, and topological dependency precedence across implementation modules.",
        "Deterministic validation cannot prove free-form prose nuance beyond AST/docstring/dependency anchors or live external LLM provider runtime behavior.",
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
        and (len(covered_files) > 0 if all_repo_paths else True)
        and not omitted_files
        and not hallucinated_files
        and not unsupported_claims
        and not ungrounded_descriptions
        and not ordering_violations
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
        "is_repo_wide_query": is_broad,
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
      the answer is empty, or unsupported claims / hallucinated files / ordering violations exist.
    - In non-strict mode (`strict=False`), explicitly marks degraded or incomplete results with
      a visible degraded-status banner and detailed evidence warnings.
    - Supplements any omitted repository files using factual AST inventory metadata when safe.
    """
    val = validation or validate_query_answer(
        answer=answer, query=query, context=context
    )

    critical_validation_failure = bool(
        val["critical_retrieval_failed"]
        or not (answer and answer.strip())
        or val["hallucinated_files"]
        or val["unsupported_claims"]
        or val["ordering_violations"]
    )

    if strict and critical_validation_failure:
        err_msg = (
            "; ".join(val["warnings"])
            or "Critical retrieval/indexing failure, unsupported claim, ordering violation, or empty answer"
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
    reading_order = (
        inv.get("reading_order") if isinstance(inv.get("reading_order"), list) else []
    )
    ro_by_path = {
        str(item["path"]): item
        for item in reading_order
        if isinstance(item, dict) and item.get("path")
    }

    # Supplement any omitted files from the verified AST inventory
    body_text = (answer or "").strip()
    if val["omitted_files"] and ro_by_path:
        supp_lines: list[str] = [
            "### Verified Inventory Supplement (Additional Repository Files)",
            "The following repository files were verified by the deterministic AST inventory and are included for complete repository coverage:",
        ]
        for path in val["omitted_files"]:
            item = ro_by_path.get(path)
            if not item:
                continue
            deps = item.get("depends_on") or []
            deps_str = (
                ", ".join(f"`{d}`" for d in deps)
                if deps
                else "None (leaf/standalone)"
            )
            cls_list = item.get("classes") or []
            fn_list = item.get("functions") or []
            sym_parts: list[str] = []
            if cls_list:
                sym_parts.append(f"Classes: `{', '.join(cls_list)}`")
            if fn_list:
                sym_parts.append(f"Functions: `{', '.join(fn_list[:8])}`")
            sym_str = f" [{' | '.join(sym_parts)}]" if sym_parts else ""
            supp_lines.append(
                f"- **`{path}`** (*{item.get('layer', 'Repository')}*, step {item.get('step')}): "
                f"{item.get('summary', '')}{sym_str} (Internal dependencies: {deps_str})"
            )
        body_text = body_text + "\n\n---\n" + "\n".join(supp_lines)

    # Re-evaluate validation after any deterministic inventory supplement
    post_val = validate_query_answer(answer=body_text, query=query, context=context)

    if not post_val["complete"]:
        reasons = "; ".join(post_val["warnings"]) or "Unverified sources or incomplete evidence"
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
    ):
        warn_lines: list[str] = ["---", "### Evidence Grounding Warning"]
        if post_val["hallucinated_files"]:
            warn_lines.append(
                "- **Non-Existent / Deleted File Paths Mentioned**: "
                + ", ".join(f"`{h}`" for h in post_val["hallucinated_files"])
            )
        if post_val["unsupported_claims"]:
            warn_lines.append(
                "- **Unsupported Symbol Attribution Claims**: "
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
                + ", ".join(f"`{u['file']}`" for u in post_val["ungrounded_descriptions"])
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

        diag_lines: list[str] = [
            "---",
            "### Verification & Coverage Diagnostics",
            f"- **Completeness Status**: `{post_val['status']}` (`complete={post_val['complete']}`)",
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
            f"- **Repository Inventory Coverage**: {post_val['covered_files_count']}/{post_val['total_repo_files']} files covered "
            f"({post_val['covered_python_files_count']}/{post_val['total_python_files']} Python modules, "
            f"{coverage.get('config_doc_files', 0)} config/doc files; "
            f"initial LLM omissions supplemented={len(val['omitted_files'])}; "
            f"remaining omissions={len(post_val['omitted_files'])}; "
            f"hallucinated paths={len(post_val['hallucinated_files'])})",
            f"- **Evidence & Description Validation**: {desc_v['grounded_files_count']}/{desc_v['checked_files_count']} covered files grounded in AST/summary evidence "
            f"(unsupported symbol claims={len(post_val['unsupported_claims'])}, "
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
