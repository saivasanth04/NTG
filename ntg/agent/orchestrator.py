"""Architecture-aware AI coding agent orchestrating Graphify, Codebase Memory MCP, planning, verification, and NTG smart routing."""

from __future__ import annotations

import inspect
from pathlib import Path
import re
import sys
from typing import Any

from ntg.agent.code_intelligence import CodeIntelligence
from ntg.agent.planner import (
    analyze_query_requirements,
    approve_plan,
    build_change_plan,
    build_planning_prompt,
    build_query_prompt,
    extract_response_content,
    finalize_grounded_answer,
    record_files_changed,
    reject_plan,
    require_approved_plan,
    validate_change_plan,
    validate_query_answer,
    verify_retrieval_context,
)
from ntg.agent.verifier import CodeVerifier
from ntg.router.engine import UnifiedNTGRouter


def _ensure_utf8_console() -> None:
    """Ensure Windows stdout/stderr can print UTF-8 responses without cp1252 crashes."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                enc = getattr(stream, "encoding", "") or ""
                if enc.lower().replace("-", "") != "utf8":
                    stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_INFORMATIONAL_STARTER_RE = re.compile(
    r"^\s*(?:what|where|why|which|who|when|how\s+(?:does|do|is|are|many|can\s+i\s+understand)|"
    r"explain|describe|summarize|list|show|walk\s+through|trace|analyze|inspect|"
    r"give\s+me\s+an?\s+overview|tell\s+me\s+about)\b",
    re.IGNORECASE,
)

_EXPLICIT_PLAN_DIRECTIVE_RE = re.compile(
    r"\b(?:create\s+a?\s*(?:change\s+|implementation\s+)?plan|"
    r"generate\s+a?\s*(?:change\s+|implementation\s+)?plan|"
    r"build\s+a?\s*(?:change\s+|implementation\s+)?plan|"
    r"write\s+a?\s*(?:change\s+|implementation\s+)?plan|"
    r"implementation\s+plan\s+(?:for|to)|"
    r"change\s+plan\s+(?:for|to)|"
    r"plan\s+to\s+(?:implement|refactor|fix|update|modify|add|remove|replace|rewrite|migrate))\b",
    re.IGNORECASE,
)

_CODING_ACTION_RE = re.compile(
    r"(?:^\s*|\b(?:please|can\s+you|could\s+you|we\s+need\s+to|i\s+want\s+to|help\s+me)\s+)"
    r"(implement|refactor|fix|update|modify|add|remove|delete|replace|rewrite|migrate|"
    r"rename|patch|integrate|optimize|extend|wire|unify|enforce|introduce|edit|convert)\b",
    re.IGNORECASE,
)


def classify_request(
    request: str,
    explicit_mode: str | None = None,
) -> dict[str, Any]:
    """Classify a repository request as `'informational'` or `'coding'` without
    creating another agent or framework.
    """
    if not isinstance(request, str) or not request.strip():
        raise ValueError("request must be a non-empty string.")

    text = request.strip()
    if explicit_mode is not None and str(explicit_mode).strip():
        norm = str(explicit_mode).strip().lower()
        if norm in ("coding", "plan", "change", "code_change", "edit", "implement"):
            return {
                "request_type": "coding",
                "explicit": True,
                "matched_indicators": [norm],
                "reason": f"Explicit mode '{norm}' selected coding workflow.",
            }
        if norm in ("informational", "query", "ask", "question", "info"):
            return {
                "request_type": "informational",
                "explicit": True,
                "matched_indicators": [norm],
                "reason": f"Explicit mode '{norm}' selected informational workflow.",
            }
        if norm != "auto":
            raise ValueError(
                f"Unsupported request mode '{explicit_mode}'. Expected 'informational', 'coding', or 'auto'."
            )

    matched_indicators: list[str] = []
    plan_match = _EXPLICIT_PLAN_DIRECTIVE_RE.search(text)
    if plan_match:
        matched_indicators.append(plan_match.group(0).strip().lower())
        return {
            "request_type": "coding",
            "explicit": False,
            "matched_indicators": matched_indicators,
            "reason": "Request explicitly asks for an implementation or change plan.",
        }

    is_info_starter = bool(_INFORMATIONAL_STARTER_RE.search(text))
    coding_matches = [m.group(1).lower() for m in _CODING_ACTION_RE.finditer(text)]

    if is_info_starter and not plan_match:
        return {
            "request_type": "informational",
            "explicit": False,
            "matched_indicators": ["informational_starter"],
            "reason": "Request is phrased as an informational repository query.",
        }

    if coding_matches:
        matched_indicators.extend(dict.fromkeys(coding_matches))
        return {
            "request_type": "coding",
            "explicit": False,
            "matched_indicators": matched_indicators,
            "reason": "Request contains code-modification action directives.",
        }

    return {
        "request_type": "informational",
        "explicit": False,
        "matched_indicators": [],
        "reason": "Defaulted to informational query workflow.",
    }


class ArchitectureAwareAgent:
    """Architecture-aware AI coding agent that:
    1. Queries Graphify and Codebase Memory MCP to understand repository structure.
    2. Uses UnifiedNTGRouter to generate architecture-aware implementation plans.
    3. Enforces explicit plan approval before modifying any repository files.
    4. Verifies changes (AST/bytecode syntax checks + verification commands).
    5. Keeps Graphify and Codebase Memory MCP synchronized after changes.
    """

    def __init__(
        self,
        repo_root: str | Path = ".",
        router: UnifiedNTGRouter | Any | None = None,
        intelligence: CodeIntelligence | None = None,
        verifier: CodeVerifier | None = None,
        project: str | None = None,
        default_capabilities: list[str] | None = None,
        memory_project: str | None = None,
    ) -> None:
        _ensure_utf8_console()
        self.repo_root = Path(repo_root).expanduser().resolve()
        if not self.repo_root.exists() or not self.repo_root.is_dir():
            raise ValueError(f"Invalid repository root: {repo_root}")

        effective_proj = project or memory_project
        self._router = router
        self.intelligence = intelligence or CodeIntelligence(
            self.repo_root, default_project=effective_proj
        )
        self.code = self.intelligence
        self.verifier = verifier or CodeVerifier(self.repo_root)
        self.project = effective_proj
        self.memory_project = effective_proj
        self.default_capabilities = (
            list(default_capabilities) if default_capabilities is not None else None
        )
        self.last_result: dict[str, Any] | None = None

    @property
    def router(self) -> UnifiedNTGRouter | Any:
        """Lazily initialize UnifiedNTGRouter if not injected at construction."""
        if self._router is None:
            self._router = UnifiedNTGRouter()
        return self._router

    def classify_request(
        self,
        request: str,
        explicit_mode: str | None = None,
    ) -> dict[str, Any]:
        """Classify a user prompt as `'informational'` or `'coding'`."""
        return classify_request(request=request, explicit_mode=explicit_mode)

    def gather_context(
        self,
        request: str,
        symbol_pattern: str | None = None,
        project: str | None = None,
        auto_build: bool = False,
        strict: bool = True,
    ) -> dict[str, Any]:
        """Collect structural context from Graphify and Codebase Memory MCP."""
        effective_project = project or self.project
        return self.intelligence.gather_context(
            question=request,
            project=effective_project,
            symbol_pattern=symbol_pattern,
            auto_build=auto_build,
            strict=strict,
        )

    def _evaluate_context_verification(
        self,
        context: dict[str, Any],
        require_indexes: bool = True,
    ) -> dict[str, Any]:
        """Verify repository inventory, relevant index freshness, coverage,
        retrieval failures, excluded/unreadable files, and request-specific evidence.
        Index health alone does not imply sufficient evidence for a request.
        """
        context_errors = [str(e) for e in context.get("errors", [])]
        retrieval_operations = {
            "graphify_query_succeeded": bool(context.get("graphify_context")),
            "memory_architecture_succeeded": bool(
                context.get("memory_architecture")
                or context.get("architecture_overview")
            ),
            "memory_symbol_query_succeeded": bool(
                context.get("memory_symbols") or context.get("memory_context")
            ),
            "inventory_verified": (
                context.get("coverage", {}).get("inventory_status") == "verified"
            ),
            "targeted_retrieval_performed": bool(
                context.get("scoped_source_excerpts")
            ),
            "retrieval_errors": context_errors,
        }

        index_status = context.get("index_status") or {}
        graph_st = (
            index_status.get("graphify")
            if isinstance(index_status.get("graphify"), dict)
            else {}
        )
        memory_st = (
            index_status.get("memory")
            if isinstance(index_status.get("memory"), dict)
            else {}
        )

        graph_ok = bool(
            graph_st.get("verified")
            and graph_st.get("fresh")
            and graph_st.get("fingerprint_verified")
            and graph_st.get("ast_symbols_verified")
        )
        memory_ok = bool(
            memory_st.get("verified")
            and memory_st.get("fresh")
            and memory_st.get("fingerprint_verified")
        )

        inventory = context.get("repository_inventory")
        coverage = (
            context.get("coverage")
            if isinstance(context.get("coverage"), dict)
            else {}
        )
        discovered_count = coverage.get("discovered_files_count", -1)
        analyzed_count = coverage.get("analyzed_files_count", -2)

        inventory_ok = bool(
            isinstance(inventory, dict)
            and inventory.get("total_files", 0) > 0
            and coverage.get("inventory_status") == "verified"
            and coverage.get("inventory_missing_files_count", 1) == 0
            and coverage.get("inventory_unreadable_files_count", 1) == 0
            and discovered_count >= 0
            and analyzed_count == discovered_count
        )

        indexes_ok = (graph_ok and memory_ok) if require_indexes else True
        prerequisites_ok = inventory_ok and indexes_ok

        retrieval_ev = (
            context.get("retrieval_evidence")
            if isinstance(context.get("retrieval_evidence"), dict)
            else {}
        )
        evidence_sufficient = bool(
            prerequisites_ok
            and not context_errors
            and retrieval_ev.get("evidence_sufficient", True)
        )
        retrieval_verification = verify_retrieval_context(context, strict=False)

        return {
            "prerequisites_ok": prerequisites_ok,
            "inventory_ok": inventory_ok,
            "indexes_ok": indexes_ok,
            "graphify_ok": graph_ok,
            "memory_ok": memory_ok,
            "require_indexes": require_indexes,
            "context_errors": context_errors,
            "evidence_sufficient": evidence_sufficient,
            "insufficiency_reasons": list(
                retrieval_ev.get("insufficiency_reasons") or []
            ),
            "limitations": list(retrieval_ev.get("limitations") or []),
            "unreadable_files": list(
                coverage.get("inventory_unreadable_files") or []
            ),
            "excluded_dirs": list(coverage.get("excluded_dirs") or []),
            "excluded_files": list(coverage.get("excluded_files") or []),
            "retrieval_operations": retrieval_operations,
            "retrieval_verification": retrieval_verification,
        }

    def _prepare_request_context(
        self,
        request: str,
        request_type: str = "informational",
        symbol_pattern: str | None = None,
        project: str | None = None,
        auto_build_graph: bool = True,
        require_indexes: bool = True,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Gather repository context once per request (or reuse an already-gathered
        context package), perform bounded targeted source retrieval when initial
        summaries or graph metadata are insufficient, and attach verification metadata.
        """
        if context is None:
            context = self.gather_context(
                request=request,
                symbol_pattern=symbol_pattern,
                project=project,
                auto_build=auto_build_graph,
                strict=False,
            )

        req_info = analyze_query_requirements(query=request, context=context)
        retrieval_ev = (
            context.get("retrieval_evidence")
            if isinstance(context.get("retrieval_evidence"), dict)
            else {}
        )
        scoped_excerpts = (
            context.get("scoped_source_excerpts")
            if isinstance(context.get("scoped_source_excerpts"), dict)
            else {}
        )

        target_files = (
            list(req_info.get("target_files") or [])
            if not req_info.get("is_repo_wide")
            else list(retrieval_ev.get("targeted_modules") or [])
        )
        queried_symbols = list(req_info.get("queried_symbols") or [])
        if (
            isinstance(symbol_pattern, str)
            and symbol_pattern.strip()
            and symbol_pattern.strip() not in queried_symbols
        ):
            queried_symbols.append(symbol_pattern.strip())

        caller_files = list(retrieval_ev.get("caller_callee_files") or [])
        dep_files = list(retrieval_ev.get("dependency_files") or [])
        config_files = list(retrieval_ev.get("config_files") or [])

        needs_targeted_retrieval = False
        if not retrieval_ev:
            needs_targeted_retrieval = True
        elif not retrieval_ev.get("evidence_sufficient", False) and target_files:
            needs_targeted_retrieval = True
        elif any(p not in scoped_excerpts for p in target_files[:6]):
            needs_targeted_retrieval = True
        elif request_type == "coding":
            related_files = (target_files + caller_files + dep_files + config_files)[:8]
            if not scoped_excerpts or any(
                p not in scoped_excerpts for p in related_files
            ):
                needs_targeted_retrieval = True

        if needs_targeted_retrieval and hasattr(
            self.intelligence, "retrieve_targeted_evidence"
        ):
            self.intelligence.retrieve_targeted_evidence(
                context=context,
                question=request,
                target_files=target_files or None,
                target_symbols=queried_symbols or None,
                include_dependencies=True,
                include_callers=True,
                include_config=True,
                max_files=8 if request_type == "coding" else 6,
                max_lines_per_file=140 if request_type == "coding" else 110,
            )

        context["request_requirements"] = req_info
        context["request_verification"] = self._evaluate_context_verification(
            context=context,
            require_indexes=require_indexes,
        )
        return context

    def _attempt_targeted_evidence_recovery(
        self,
        context: dict[str, Any],
        request: str,
        draft_text: str = "",
        require_indexes: bool = True,
    ) -> bool:
        """Perform a bounded targeted source retrieval pass if validation or draft
        analysis references repository modules/symbols whose source excerpts were
        not yet retrieved. Returns True if new excerpts were added.
        """
        if not isinstance(context, dict) or not hasattr(
            self.intelligence, "retrieve_targeted_evidence"
        ):
            return False

        retrieval_ev = (
            context.get("retrieval_evidence")
            if isinstance(context.get("retrieval_evidence"), dict)
            else {}
        )
        rounds = int(retrieval_ev.get("targeted_expansion_rounds", 0) or 0)
        if rounds >= 2:
            return False

        inv = (
            context.get("repository_inventory")
            if isinstance(context.get("repository_inventory"), dict)
            else {}
        )
        inv_files = [
            f for f in (inv.get("files") or []) if isinstance(f, dict) and f.get("path")
        ]
        if not inv_files:
            return False

        valid_paths = {str(f["path"]) for f in inv_files}
        existing_excerpts = (
            dict(context.get("scoped_source_excerpts") or {})
            if isinstance(context.get("scoped_source_excerpts"), dict)
            else {}
        )

        combined_text = f"{request}\n{draft_text}"
        candidate_files: list[str] = []
        for p in sorted(valid_paths):
            if p in combined_text and p not in existing_excerpts:
                candidate_files.append(p)

        req_info = context.get("request_requirements") or analyze_query_requirements(
            query=request, context=context
        )
        if not req_info.get("is_repo_wide"):
            for p in req_info.get("target_files") or []:
                if p in valid_paths and p not in existing_excerpts and p not in candidate_files:
                    candidate_files.append(p)

        if not candidate_files and existing_excerpts:
            return False

        before_keys = set(existing_excerpts.keys())
        self.intelligence.retrieve_targeted_evidence(
            context=context,
            question=request,
            target_files=candidate_files or None,
            target_symbols=list(req_info.get("queried_symbols") or []) or None,
            include_dependencies=True,
            include_callers=True,
            include_config=True,
            max_files=10,
            max_lines_per_file=140,
        )
        context["request_verification"] = self._evaluate_context_verification(
            context=context,
            require_indexes=require_indexes,
        )
        after_keys = set((context.get("scoped_source_excerpts") or {}).keys())
        return bool(after_keys - before_keys)

    def _invoke_router(
        self,
        prompt: str,
        model: str | None = None,
        capabilities: list[str] | None = None,
    ) -> Any:
        """Route a planning or coding prompt through the NTG smart router."""
        router_obj = self.router
        effective_caps = (
            capabilities if capabilities is not None else self.default_capabilities
        )
        messages = [{"role": "user", "content": prompt}]

        if hasattr(router_obj, "completion"):
            completion_fn = router_obj.completion
            try:
                sig = inspect.signature(completion_fn)
            except (ValueError, TypeError):
                return completion_fn(messages=messages)
            params = sig.parameters
            accepts_var_kw = any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in params.values()
            )
            kwargs: dict[str, Any] = {}
            if "messages" in params or accepts_var_kw:
                kwargs["messages"] = messages
            if model is not None and ("model" in params or accepts_var_kw):
                kwargs["model"] = model
            if effective_caps and ("capabilities" in params or accepts_var_kw):
                kwargs["capabilities"] = effective_caps
            # Provider/validation errors must propagate. Retrying the call inside a
            # signature-inspection handler could silently discard requirements.
            return completion_fn(**kwargs) if kwargs else completion_fn(messages=messages)

        if hasattr(router_obj, "ask"):
            return router_obj.ask(prompt, model=model, capabilities=effective_caps)

        raise RuntimeError("Configured router does not support 'completion' or 'ask'.")

    def create_plan(
        self,
        request: str,
        symbol_pattern: str | None = None,
        project: str | None = None,
        model: str | None = None,
        capabilities: list[str] | None = None,
        auto_build_graph: bool = True,
        strict: bool = True,
        require_indexes: bool = True,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Gather codebase intelligence from Graphify and Codebase Memory MCP,
        verify that retrieval and indexes are healthy (refusing to invoke the model
        if retrieval failed), route an architecture-aware planning prompt through
        UnifiedNTGRouter, and return a reviewable plan gated on human approval.
        """
        if not isinstance(request, str) or not request.strip():
            raise ValueError("request must be a non-empty string.")

        return self.handle_request(
            request=request,
            mode="coding",
            symbol_pattern=symbol_pattern,
            project=project,
            model=model,
            capabilities=capabilities,
            auto_build_graph=auto_build_graph,
            strict=strict,
            require_indexes=require_indexes,
            context=context,
        )

    def approve(self, plan: dict[str, Any]) -> dict[str, Any]:
        """Explicitly approve a pending plan for execution."""
        return approve_plan(plan)

    def reject(self, plan: dict[str, Any], reason: str = "") -> dict[str, Any]:
        """Explicitly reject a pending plan."""
        return reject_plan(plan, reason=reason)

    def apply_changes(
        self,
        plan: dict[str, Any],
        file_changes: dict[str, str],
    ) -> dict[str, Any]:
        """Apply a mapping of `{relative_or_repo_path: new_content}` to disk.
        Strictly requires `plan` to be approved first and prevents writing outside `repo_root`.
        """
        require_approved_plan(plan)
        if not isinstance(file_changes, dict) or not file_changes:
            raise ValueError(
                "file_changes must be a non-empty dictionary of {path: content}."
            )

        resolved_writes: list[tuple[Path, str, str]] = []
        for raw_path, new_content in file_changes.items():
            if not isinstance(raw_path, str) or not raw_path.strip():
                raise ValueError("File path must be a non-empty string.")
            if not isinstance(new_content, str):
                raise ValueError(f"Content for {raw_path} must be a string.")

            candidate = Path(raw_path)
            target = (
                (self.repo_root / candidate).resolve()
                if not candidate.is_absolute()
                else candidate.resolve()
            )
            try:
                rel_path = target.relative_to(self.repo_root)
            except ValueError as exc:
                raise ValueError(
                    f"Refusing to write outside repository root: {raw_path}"
                ) from exc

            rel_parts = set(rel_path.parts)
            if ".git" in rel_parts:
                raise ValueError(f"Refusing to modify .git directory: {raw_path}")

            resolved_writes.append(
                (target, str(rel_path).replace("\\", "/"), new_content)
            )

        changed_rel_paths: list[str] = []
        for target_path, rel_str, content in resolved_writes:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(content, encoding="utf-8")
            changed_rel_paths.append(rel_str)

        record_files_changed(plan, changed_rel_paths)
        return plan

    def sync_knowledge(
        self,
        project: str | None = None,
        plan: dict[str, Any] | None = None,
        save_plan_to_graph: bool = False,
    ) -> dict[str, Any]:
        """Refresh Graphify and Codebase Memory MCP indexes so the agent's
        codebase knowledge stays up to date after code changes.
        """
        effective_project = project or self.project
        sync_result = self.intelligence.refresh_knowledge(project=effective_project)

        if (
            save_plan_to_graph
            and isinstance(plan, dict)
            and plan.get("request")
            and plan.get("plan")
        ):
            try:
                self.intelligence.graphify_save_result(
                    question=str(plan["request"]),
                    answer=str(plan["plan"]),
                    query_type="change_plan",
                )
                sync_result["saved_plan_to_graph"] = True
            except Exception as exc:
                sync_result["saved_plan_to_graph"] = False
                sync_result["save_error"] = str(exc)

        if isinstance(plan, dict):
            plan["knowledge_synced"] = bool(sync_result.get("synced", False))
            plan["knowledge_sync_result"] = sync_result

        return sync_result

    def verify(
        self,
        plan: dict[str, Any],
        commands: list[list[str]] | None = None,
        refresh_knowledge: bool = True,
        project: str | None = None,
    ) -> dict[str, Any]:
        """Verify a plan's implementation (syntax + optional test/lint commands)
        and refresh codebase knowledge graphs when verification succeeds.
        """
        verification = self.verifier.verify_plan(plan, commands=commands)
        if verification["passed"] and refresh_knowledge:
            sync_res = self.sync_knowledge(project=project, plan=plan)
            verification["knowledge_sync"] = sync_res
        return verification

    def execute_plan(
        self,
        plan: dict[str, Any],
        file_changes: dict[str, str] | None = None,
        verification_commands: list[list[str]] | None = None,
        refresh_knowledge: bool = True,
        project: str | None = None,
    ) -> dict[str, Any]:
        """Execute an approved plan end-to-end:
        1. Verify the plan has been explicitly approved.
        2. Apply any provided `file_changes`.
        3. Run syntax and command verification.
        4. Refresh Graphify and Codebase Memory MCP knowledge when verification passes.
        """
        require_approved_plan(plan)
        if file_changes:
            self.apply_changes(plan, file_changes)

        return self.verify(
            plan=plan,
            commands=verification_commands,
            refresh_knowledge=refresh_knowledge,
            project=project,
        )

    def _execute_coding_request(
        self,
        request: str,
        context: dict[str, Any],
        classification: dict[str, Any],
        model: str | None = None,
        capabilities: list[str] | None = None,
        strict: bool = True,
        require_indexes: bool = True,
    ) -> dict[str, Any]:
        """Execute the coding-plan workflow using an already-gathered and verified context package."""
        req_ver = (
            context.get("request_verification")
            if isinstance(context.get("request_verification"), dict)
            else self._evaluate_context_verification(
                context=context, require_indexes=require_indexes
            )
        )
        ver = req_ver["retrieval_verification"]
        prerequisites_ok = bool(req_ver["prerequisites_ok"])
        inventory_ok = bool(req_ver["inventory_ok"])
        graph_ok = bool(req_ver["graphify_ok"])
        memory_ok = bool(req_ver["memory_ok"])
        context_errors = list(req_ver["context_errors"])
        retrieval_operations = dict(req_ver["retrieval_operations"])
        retrieval_policy = {
            "strict": strict,
            "require_indexes": require_indexes,
            "prerequisites_ok": prerequisites_ok,
            "inventory_ok": inventory_ok,
            "graphify_ok": graph_ok,
            "memory_ok": memory_ok,
            "evidence_sufficient": bool(req_ver.get("evidence_sufficient", False)),
        }

        if not ver["all_verified"] or not prerequisites_ok or context_errors:
            degraded_plan = build_change_plan(
                request=request,
                planning_response="",
                context=context,
                strict=False,
            )
            degraded_plan.update(
                {
                    "repo_root": str(self.repo_root),
                    "project": context.get("project"),
                    "request_type": "coding",
                    "classification": classification,
                    "retrieval_failed": True,
                    "retrieval_evidence": context.get("retrieval_evidence"),
                    "index_status": context.get("index_status"),
                    "coverage": context.get("coverage"),
                    "retrieval_policy": retrieval_policy,
                    "retrieval_operations": retrieval_operations,
                    "response": None,
                }
            )
            self.last_result = degraded_plan
            if strict:
                verify_retrieval_context(context, strict=True)
                details = list(context_errors) or [
                    "Required retrieval prerequisites were not satisfied."
                ]
                raise RuntimeError(
                    "Strict coding plan refused before LLM generation because required "
                    "context is incomplete: " + "; ".join(details)
                )
            return degraded_plan

        prompt = build_planning_prompt(
            request=request,
            context=context,
            strict=strict,
        )
        response = self._invoke_router(
            prompt=prompt,
            model=model,
            capabilities=capabilities,
        )
        plan_dict = build_change_plan(
            request=request,
            planning_response=response,
            context=context,
            strict=strict,
        )

        if not plan_dict.get("complete", False):
            recovered = self._attempt_targeted_evidence_recovery(
                context=context,
                request=request,
                draft_text=str(plan_dict.get("plan") or ""),
                require_indexes=require_indexes,
            )
            if recovered:
                req_ver = context["request_verification"]
                retrieval_operations = dict(req_ver["retrieval_operations"])
                retrieval_policy["evidence_sufficient"] = bool(
                    req_ver.get("evidence_sufficient", False)
                )
                plan_dict = build_change_plan(
                    request=request,
                    planning_response=response,
                    context=context,
                    strict=strict,
                )

        plan_dict.update(
            {
                "repo_root": str(self.repo_root),
                "project": context.get("project"),
                "request_type": "coding",
                "classification": classification,
                "retrieval_failed": bool(plan_dict.get("retrieval_blocked", False)),
                "retrieval_evidence": context.get("retrieval_evidence"),
                "index_status": context.get("index_status"),
                "coverage": context.get("coverage"),
                "retrieval_policy": retrieval_policy,
                "retrieval_operations": retrieval_operations,
                "response": response,
            }
        )
        self.last_result = plan_dict
        return plan_dict

    def _execute_informational_request(
        self,
        query: str,
        context: dict[str, Any],
        classification: dict[str, Any],
        model: str | None = None,
        capabilities: list[str] | None = None,
        strict: bool = True,
        include_diagnostics: bool = True,
        require_indexes: bool = True,
    ) -> dict[str, Any]:
        """Execute the informational query workflow using an already-gathered and verified context package."""
        req_ver = (
            context.get("request_verification")
            if isinstance(context.get("request_verification"), dict)
            else self._evaluate_context_verification(
                context=context, require_indexes=require_indexes
            )
        )
        prerequisites_ok = bool(req_ver["prerequisites_ok"])
        inventory_ok = bool(req_ver["inventory_ok"])
        graph_ok = bool(req_ver["graphify_ok"])
        memory_ok = bool(req_ver["memory_ok"])
        context_errors = list(req_ver["context_errors"])
        retrieval_operations = dict(req_ver["retrieval_operations"])

        if strict and (not prerequisites_ok or context_errors):
            details = list(context_errors)

            if not inventory_ok:
                details.append(
                    "Repository inventory was not verified against the discoverable filesystem."
                )
            if require_indexes and not graph_ok:
                details.append("Graphify is not verified and fresh.")
            if require_indexes and not memory_ok:
                details.append("Codebase Memory MCP is not verified and fresh.")

            details = list(dict.fromkeys(details)) or [
                "Required retrieval prerequisites were not satisfied."
            ]
            raise RuntimeError(
                "Strict query refused before LLM generation because required "
                "context is incomplete: " + "; ".join(details)
            )

        model_invoked = False
        response: Any = None
        raw_answer = ""

        if prerequisites_ok and not context_errors:
            prompt = build_query_prompt(
                query=query,
                context=context,
                strict=strict,
            )
            response = self._invoke_router(
                prompt=prompt,
                model=model,
                capabilities=capabilities,
            )
            model_invoked = True
            raw_answer = extract_response_content(response)

        initial_validation = validate_query_answer(
            answer=raw_answer,
            query=query,
            context=context,
        )
        initial_validation["model_invoked"] = model_invoked

        if not initial_validation.get("complete", False):
            recovered = self._attempt_targeted_evidence_recovery(
                context=context,
                request=query,
                draft_text=raw_answer,
                require_indexes=require_indexes,
            )
            if recovered:
                req_ver = context["request_verification"]
                retrieval_operations = dict(req_ver["retrieval_operations"])
                initial_validation = validate_query_answer(
                    answer=raw_answer,
                    query=query,
                    context=context,
                )
                initial_validation["model_invoked"] = model_invoked

        answer = finalize_grounded_answer(
            answer=raw_answer,
            query=query,
            context=context,
            validation=initial_validation,
            strict=strict,
            include_diagnostics=include_diagnostics,
        )
        final_validation = validate_query_answer(
            answer=answer,
            query=query,
            context=context,
        )
        final_validation["model_invoked"] = model_invoked

        if strict and not final_validation["complete"]:
            err_msg = (
                "; ".join(final_validation.get("warnings") or [])
                or "Finalized answer failed strict completeness verification"
            )
            raise RuntimeError(
                f"Strict mode: refusing to present incomplete or unverified answer: {err_msg}"
            )

        answer_complete = bool(final_validation.get("complete", False))
        retrieval_failed = bool(
            final_validation.get("critical_retrieval_failed", True)
        )

        if retrieval_failed:
            overall_status = "degraded"
        elif answer_complete:
            overall_status = "complete"
        else:
            overall_status = "incomplete"

        result = {
            "repo_root": str(self.repo_root),
            "project": context.get("project"),
            "request": query,
            "query": query,
            "request_type": "informational",
            "classification": classification,
            "answer": answer,
            "raw_answer": raw_answer,
            "query_scope": final_validation.get("query_scope"),
            "is_repo_wide_query": final_validation.get("is_repo_wide_query", False),
            "answer_complete": answer_complete,
            "retrieval_failed": retrieval_failed,
            "retrieval_blocked": bool(
                final_validation.get("retrieval_blocked", retrieval_failed)
            ),
            "complete": final_validation["complete"],
            "status": overall_status,
            "validation": final_validation,
            "initial_validation": initial_validation,
            "index_status": context.get("index_status"),
            "coverage": context.get("coverage"),
            "retrieval_evidence": context.get("retrieval_evidence"),
            "errors": context.get("errors", []),
            "retrieval_policy": {
                "strict": strict,
                "require_indexes": require_indexes,
                "prerequisites_ok": prerequisites_ok,
                "inventory_ok": inventory_ok,
                "graphify_ok": graph_ok,
                "memory_ok": memory_ok,
                "evidence_sufficient": bool(req_ver.get("evidence_sufficient", False)),
            },
            "retrieval_operations": retrieval_operations,
            "model_invoked": model_invoked,
            "context": context,
            "response": response,
        }

        self.last_result = result
        return result

    def handle_request(
        self,
        request: str,
        mode: str | None = None,
        symbol_pattern: str | None = None,
        project: str | None = None,
        model: str | None = None,
        capabilities: list[str] | None = None,
        auto_build_graph: bool = True,
        strict: bool = True,
        include_diagnostics: bool = True,
        require_indexes: bool = True,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Primary repository-aware entry point unifying informational queries and
        coding change-plan requests around a single gathered, verified context package.
        """
        if not isinstance(request, str) or not request.strip():
            raise ValueError("request must be a non-empty string.")

        classification = self.classify_request(
            request=request,
            explicit_mode=mode,
        )
        request_type = classification["request_type"]

        shared_context = self._prepare_request_context(
            request=request,
            request_type=request_type,
            symbol_pattern=symbol_pattern,
            project=project,
            auto_build_graph=auto_build_graph,
            require_indexes=require_indexes,
            context=context,
        )

        if request_type == "coding":
            return self._execute_coding_request(
                request=request,
                context=shared_context,
                classification=classification,
                model=model,
                capabilities=capabilities,
                strict=strict,
                require_indexes=require_indexes,
            )

        return self._execute_informational_request(
            query=request,
            context=shared_context,
            classification=classification,
            model=model,
            capabilities=capabilities,
            strict=strict,
            include_diagnostics=include_diagnostics,
            require_indexes=require_indexes,
        )

    def ask(
        self,
        query: str,
        symbol_pattern: str | None = None,
        project: str | None = None,
        model: str | None = None,
        capabilities: list[str] | None = None,
        auto_build_graph: bool = True,
        strict: bool = True,
        include_diagnostics: bool = True,
        require_indexes: bool = True,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Answer a repository query with explicit retrieval and completeness policy.

        Strict mode refuses LLM generation when required evidence is incomplete.
        Best-effort mode may generate from partial context, but validation must not
        label the result complete unless all required checks pass.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")

        return self.handle_request(
            request=query,
            mode="informational",
            symbol_pattern=symbol_pattern,
            project=project,
            model=model,
            capabilities=capabilities,
            auto_build_graph=auto_build_graph,
            strict=strict,
            include_diagnostics=include_diagnostics,
            require_indexes=require_indexes,
            context=context,
        )


def query_directory_structured(
    directory: str | Path,
    query: str,
    symbol_pattern: str | None = None,
    model: str | None = None,
    capabilities: list[str] | None = None,
    auto_build_graph: bool = True,
    strict: bool = True,
    include_diagnostics: bool = True,
    require_indexes: bool = True,
) -> dict[str, Any]:
    """Return the complete structured answer and diagnostics for a repository query."""
    agent = ArchitectureAwareAgent(repo_root=directory)
    return agent.ask(
        query=query,
        symbol_pattern=symbol_pattern,
        model=model,
        capabilities=capabilities,
        auto_build_graph=auto_build_graph,
        strict=strict,
        include_diagnostics=include_diagnostics,
        require_indexes=require_indexes,
    )


def query_directory(
    directory: str | Path,
    query: str,
    symbol_pattern: str | None = None,
    model: str | None = None,
    capabilities: list[str] | None = None,
    auto_build_graph: bool = True,
    strict: bool = True,
    include_diagnostics: bool = True,
    return_structured: bool = True,
    require_indexes: bool = True,
) -> str | dict[str, Any]:
    """Query a repository; structured diagnostics are returned by default.

    Set return_structured=False only for legacy callers that explicitly need a string.
    Strict mode refuses model generation if required evidence sources are unavailable.
    """
    result = query_directory_structured(
        directory=directory,
        query=query,
        symbol_pattern=symbol_pattern,
        model=model,
        capabilities=capabilities,
        auto_build_graph=auto_build_graph,
        strict=strict,
        include_diagnostics=include_diagnostics,
        require_indexes=require_indexes,
    )
    if return_structured:
        return result
    return result["answer"]


CodingAgent = ArchitectureAwareAgent
