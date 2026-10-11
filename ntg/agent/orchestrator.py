"""Architecture-aware AI coding agent orchestrating Graphify, Codebase Memory MCP, planning, verification, and NTG smart routing."""

from __future__ import annotations

import inspect
from pathlib import Path
import sys
from typing import Any

from ntg.agent.code_intelligence import CodeIntelligence
from ntg.agent.planner import (
    approve_plan,
    build_change_plan,
    build_planning_prompt,
    build_query_prompt,
    extract_response_content,
    finalize_grounded_answer,
    record_files_changed,
    reject_plan,
    require_approved_plan,
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
                kwargs: dict[str, Any] = {}
                params = sig.parameters
                accepts_var_kw = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
                )
                if "messages" in params or accepts_var_kw:
                    kwargs["messages"] = messages
                if model is not None and ("model" in params or accepts_var_kw):
                    kwargs["model"] = model
                if effective_caps and ("capabilities" in params or accepts_var_kw):
                    kwargs["capabilities"] = effective_caps
                if kwargs:
                    return completion_fn(**kwargs)
            except (ValueError, TypeError):
                pass
            return completion_fn(messages=messages)

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
    ) -> dict[str, Any]:
        """Gather codebase intelligence from Graphify and Codebase Memory MCP,
        verify that retrieval and indexes are healthy (refusing to invoke the model
        if retrieval failed), route an architecture-aware planning prompt through
        UnifiedNTGRouter, and return a reviewable plan gated on human approval.
        """
        if not isinstance(request, str) or not request.strip():
            raise ValueError("request must be a non-empty string.")

        context = self.gather_context(
            request=request,
            symbol_pattern=symbol_pattern,
            project=project,
            auto_build=auto_build_graph,
            strict=False,
        )
        ver = verify_retrieval_context(context, strict=False)
        if not ver["all_verified"]:
            degraded_plan = build_change_plan(
                request=request,
                planning_response="",
                context=context,
                strict=False,
            )
            self.last_result = degraded_plan
            if strict:
                verify_retrieval_context(context, strict=True)
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
        self.last_result = plan_dict
        return plan_dict

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
    ) -> dict[str, Any]:
        """Answer a repository query with explicit retrieval and completeness policy.

        Strict mode refuses LLM generation when required evidence is incomplete.
        Best-effort mode may generate from partial context, but validation must not
        label the result complete unless all required checks pass.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")

        # Keep retrieval errors available as structured data so policy is enforced here.
        context = self.gather_context(
            request=query,
            symbol_pattern=symbol_pattern,
            project=project,
            auto_build=auto_build_graph,
            strict=False,
        )

        context_errors = [str(e) for e in context.get("errors", [])]
        retrieval_operations = {
            "graphify_query_succeeded": bool(
                context.get("graphify_context")
            ),
            "memory_architecture_succeeded": bool(
                context.get("memory_architecture")
                or context.get("architecture_overview")
            ),
            "memory_symbol_query_succeeded": bool(
                context.get("memory_symbols")
                or context.get("memory_context")
            ),
            "inventory_verified": (
                context.get("coverage", {}).get("inventory_status")
                == "verified"
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

        prompt = build_query_prompt(query=query, context=context)
        response = self._invoke_router(
            prompt=prompt,
            model=model,
            capabilities=capabilities,
        )
        raw_answer = extract_response_content(response)

        initial_validation = validate_query_answer(
            answer=raw_answer,
            query=query,
            context=context,
        )
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

        if final_validation["complete"]:
            overall_status = "complete"
        elif final_validation["critical_retrieval_failed"]:
            overall_status = "degraded"
        else:
            overall_status = "incomplete"

        result = {
            "repo_root": str(self.repo_root),
            "project": context.get("project"),
            "query": query,
            "answer": answer,
            "raw_answer": raw_answer,
            "complete": final_validation["complete"],
            "status": overall_status,
            "validation": final_validation,
            "initial_validation": initial_validation,
            "index_status": context.get("index_status"),
            "coverage": context.get("coverage"),
            "errors": context.get("errors", []),
            "retrieval_policy": {
                "strict": strict,
                "require_indexes": require_indexes,
                "prerequisites_ok": prerequisites_ok,
                "inventory_ok": inventory_ok,
                "graphify_ok": graph_ok,
                "memory_ok": memory_ok,
            },
            "retrieval_operations": retrieval_operations,
            "model_invoked": True,
            "context": context,
            "response": response,
        }

        self.last_result = result
        return result


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
