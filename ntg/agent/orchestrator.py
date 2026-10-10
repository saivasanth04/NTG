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
    record_files_changed,
    reject_plan,
    require_approved_plan,
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
    ) -> dict[str, Any]:
        """Collect structural context from Graphify and Codebase Memory MCP."""
        effective_project = project or self.project
        return self.intelligence.gather_context(
            question=request,
            project=effective_project,
            symbol_pattern=symbol_pattern,
            auto_build=auto_build,
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
        auto_build_graph: bool = False,
    ) -> dict[str, Any]:
        """Gather codebase intelligence from Graphify and Codebase Memory MCP,
        route an architecture-aware planning prompt through UnifiedNTGRouter,
        and return a reviewable plan gated on human approval.
        """
        if not isinstance(request, str) or not request.strip():
            raise ValueError("request must be a non-empty string.")

        context = self.gather_context(
            request=request,
            symbol_pattern=symbol_pattern,
            project=project,
            auto_build=auto_build_graph,
        )
        prompt = build_planning_prompt(request=request, context=context)
        response = self._invoke_router(
            prompt=prompt,
            model=model,
            capabilities=capabilities,
        )
        return build_change_plan(
            request=request,
            planning_response=response,
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

            resolved_writes.append((target, str(rel_path).replace("\\", "/"), new_content))

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
    ) -> dict[str, Any]:
        """Answer a natural-language query about `self.repo_root` using Graphify,
        Codebase Memory MCP, and UnifiedNTGRouter.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")

        context = self.gather_context(
            request=query,
            symbol_pattern=symbol_pattern,
            project=project,
            auto_build=auto_build_graph,
        )
        prompt = build_query_prompt(query=query, context=context)
        response = self._invoke_router(
            prompt=prompt,
            model=model,
            capabilities=capabilities,
        )
        answer = extract_response_content(response)
        return {
            "repo_root": str(self.repo_root),
            "project": context.get("project"),
            "query": query,
            "answer": answer,
            "index_status": context.get("index_status"),
            "coverage": context.get("coverage"),
            "errors": context.get("errors", []),
            "context": context,
            "response": response,
        }


def query_directory(
    directory: str | Path,
    query: str,
    symbol_pattern: str | None = None,
    model: str | None = None,
    capabilities: list[str] | None = None,
    auto_build_graph: bool = True,
) -> str:
    """Query any target directory using NTG's CodeIntelligence and UnifiedNTGRouter
    and return the answer text.
    """
    agent = ArchitectureAwareAgent(repo_root=directory)
    result = agent.ask(
        query=query,
        symbol_pattern=symbol_pattern,
        model=model,
        capabilities=capabilities,
        auto_build_graph=auto_build_graph,
    )
    return result["answer"]


CodingAgent = ArchitectureAwareAgent

