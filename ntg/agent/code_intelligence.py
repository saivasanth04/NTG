"""Adapter for Graphify and Codebase Memory MCP architecture & symbol intelligence."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
from typing import Any


class CodeIntelligence:
    """Adapter for Graphify and Codebase Memory MCP queries and knowledge updates."""

    def __init__(self, repo_root: str | Path, default_project: str | None = None):
        self.repo_root = Path(repo_root).resolve()

        if not self.repo_root.is_dir():
            raise ValueError("Repository root must be an existing directory.")

        self.default_project = (
            default_project.strip()
            if isinstance(default_project, str) and default_project.strip()
            else None
        )

    def _run(self, args: list[str], timeout: int = 30) -> str:
        """Run a fixed local tool command without invoking a shell."""
        if not args or not all(isinstance(a, str) and a for a in args):
            raise ValueError("Command arguments must be a non-empty list of strings.")

        try:
            result = subprocess.run(
                args,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                shell=False,
                stdin=subprocess.DEVNULL,
            )
        except TypeError:
            result = subprocess.run(
                args,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                shell=False,
            )
        except FileNotFoundError as exc:
            resolved = shutil.which(args[0])
            if not resolved:
                raise RuntimeError(
                    f"Required code intelligence executable not found: {args[0]}"
                ) from exc
            result = subprocess.run(
                [resolved, *args[1:]],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                shell=False,
                stdin=subprocess.DEVNULL,
            )

        if result.returncode != 0:
            err_detail = (result.stderr or result.stdout or "").strip()[-2000:]
            raise RuntimeError(
                f"Code intelligence command failed ({result.returncode}): {err_detail}"
            )

        return result.stdout

    # ------------------------------------------------------------------
    # Graphify Operations
    # ------------------------------------------------------------------

    @property
    def graph_file(self) -> Path:
        """Path to the Graphify knowledge graph JSON file."""
        return self.repo_root / "graphify-out" / "graph.json"

    def graphify_query(self, question: str, budget: int = 1500, dfs: bool = False) -> str:
        """Retrieve broader graph-based architecture context."""
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        if not self.graph_file.is_file():
            raise RuntimeError("Graphify graph is missing. Build it before querying.")

        cmd = ["graphify", "query", question, "--budget", str(budget)]
        if dfs:
            cmd.append("--dfs")
        return self._run(cmd)

    def graphify_explain(self, node: str) -> str:
        """Retrieve a plain-language explanation of a graph node and its neighbors."""
        if not node or not node.strip():
            raise ValueError("Node name cannot be empty.")

        if not self.graph_file.is_file():
            raise RuntimeError("Graphify graph is missing. Build it before querying.")

        return self._run(["graphify", "explain", node.strip()])

    def graphify_path(self, source: str, target: str) -> str:
        """Find the shortest dependency path between two nodes in the Graphify graph."""
        if not source or not source.strip():
            raise ValueError("Source node cannot be empty.")
        if not target or not target.strip():
            raise ValueError("Target node cannot be empty.")

        if not self.graph_file.is_file():
            raise RuntimeError("Graphify graph is missing. Build it before querying.")

        return self._run(["graphify", "path", source.strip(), target.strip()])

    def graphify_update(self, target_path: str = ".") -> str:
        """Re-extract code files and update the Graphify graph (no LLM required)."""
        if not target_path or not target_path.strip():
            raise ValueError("Target path cannot be empty.")
        return self._run(["graphify", "update", target_path.strip()], timeout=60)

    def ensure_graphify_graph(self) -> Path:
        """Ensure graphify-out/graph.json exists, building it if missing."""
        if not self.graph_file.is_file():
            self.graphify_update(".")
        return self.graph_file

    def graphify_save_result(
        self,
        question: str,
        answer: str,
        query_type: str = "query",
        nodes: list[str] | None = None,
    ) -> str:
        """Save an architectural Q&A result to graphify-out/memory/ for graph feedback."""
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")
        if not answer or not answer.strip():
            raise ValueError("Answer cannot be empty.")

        cmd = [
            "graphify",
            "save-result",
            "--question",
            question.strip(),
            "--answer",
            answer.strip(),
            "--type",
            query_type,
        ]
        if nodes:
            clean_nodes = [n.strip() for n in nodes if isinstance(n, str) and n.strip()]
            if clean_nodes:
                cmd.extend(["--nodes", *clean_nodes])
        return self._run(cmd)

    # ------------------------------------------------------------------
    # Codebase Memory MCP Operations
    # ------------------------------------------------------------------

    def memory_query(self, project: str, symbol_pattern: str) -> str:
        """Search indexed code symbols using Codebase Memory MCP."""
        if not project or not project.strip():
            raise ValueError("Project name cannot be empty.")

        if not symbol_pattern or not symbol_pattern.strip():
            raise ValueError("Symbol pattern cannot be empty.")

        return self._run(
            [
                "codebase-memory-mcp",
                "cli",
                "search_graph",
                "--project",
                project,
                "--name-pattern",
                symbol_pattern,
                "--format",
                "json",
            ]
        )

    def memory_index(self, project: str | None = None, mode: str = "fast") -> str:
        """Index or refresh the repository in Codebase Memory MCP."""
        cmd = [
            "codebase-memory-mcp",
            "cli",
            "index_repository",
            "--repo-path",
            str(self.repo_root),
            "--mode",
            mode,
        ]
        target_name = (
            project.strip()
            if isinstance(project, str) and project.strip()
            else self.default_project
        )
        if target_name:
            cmd.extend(["--name", target_name])

        output = self._run(cmd, timeout=60)
        try:
            parsed = json.loads(output)
            if isinstance(parsed, dict) and isinstance(parsed.get("project"), str):
                self.default_project = parsed["project"]
        except (ValueError, TypeError):
            pass
        return output

    def resolve_memory_project(self, preferred_project: str | None = None) -> str:
        """Resolve the indexed project name corresponding to repo_root in Codebase Memory MCP."""
        try:
            out = self._run(
                ["codebase-memory-mcp", "cli", "list_projects", "--format", "json"]
            )
            data = json.loads(out)
            projects = data.get("projects", []) if isinstance(data, dict) else []
            for entry in projects:
                if not isinstance(entry, dict):
                    continue
                root_str = entry.get("root_path")
                name = entry.get("name")
                if root_str and name:
                    try:
                        if Path(root_str).resolve() == self.repo_root:
                            self.default_project = str(name)
                            return str(name)
                    except Exception:
                        pass
            if preferred_project:
                for entry in projects:
                    if isinstance(entry, dict) and entry.get("name") == preferred_project:
                        return preferred_project
        except Exception:
            pass

        return (
            (
                preferred_project.strip()
                if isinstance(preferred_project, str) and preferred_project.strip()
                else None
            )
            or self.default_project
            or self.repo_root.name
        )

    def memory_architecture(self, project: str, aspects: str = "overview") -> str:
        """Retrieve architectural structure from Codebase Memory MCP."""
        if not project or not project.strip():
            raise ValueError("Project name cannot be empty.")

        return self._run(
            [
                "codebase-memory-mcp",
                "cli",
                "get_architecture",
                "--project",
                project.strip(),
                "--format",
                "json",
            ]
        )

    def memory_trace_path(
        self,
        project: str,
        function_name: str,
        direction: str = "both",
        depth: int = 2,
    ) -> str:
        """Trace inbound/outbound call paths for a function or method."""
        if not project or not project.strip():
            raise ValueError("Project name cannot be empty.")
        if not function_name or not function_name.strip():
            raise ValueError("Function name cannot be empty.")

        return self._run(
            [
                "codebase-memory-mcp",
                "cli",
                "trace_path",
                "--project",
                project.strip(),
                "--function-name",
                function_name.strip(),
                "--direction",
                direction,
                "--depth",
                str(depth),
                "--format",
                "json",
            ]
        )

    def memory_detect_changes(self, project: str, base_branch: str = "main") -> str:
        """Detect changed symbols and their architectural impact radius."""
        if not project or not project.strip():
            raise ValueError("Project name cannot be empty.")

        return self._run(
            [
                "codebase-memory-mcp",
                "cli",
                "detect_changes",
                "--project",
                project.strip(),
                "--base-branch",
                base_branch,
                "--format",
                "json",
            ]
        )

    def memory_get_snippet(self, project: str, qualified_name: str) -> str:
        """Retrieve source code snippet for a qualified symbol name."""
        if not project or not project.strip():
            raise ValueError("Project name cannot be empty.")
        if not qualified_name or not qualified_name.strip():
            raise ValueError("Qualified name cannot be empty.")

        return self._run(
            [
                "codebase-memory-mcp",
                "cli",
                "get_code_snippet",
                "--project",
                project.strip(),
                "--qualified-name",
                qualified_name.strip(),
                "--format",
                "json",
            ]
        )

    # ------------------------------------------------------------------
    # Unified Context & Knowledge Synchronization
    # ------------------------------------------------------------------

    def refresh_knowledge(self, project: str | None = None) -> dict[str, Any]:
        """Rebuild/update both Graphify and Codebase Memory MCP indices after code changes."""
        errors: list[str] = []
        graphify_ok = False
        memory_ok = False
        graphify_out = ""
        memory_out = ""

        try:
            graphify_out = self.graphify_update(".")
            graphify_ok = True
        except Exception as err:
            errors.append(f"graphify_update: {err}")

        try:
            memory_out = self.memory_index(project=project)
            memory_ok = True
        except Exception as err:
            errors.append(f"memory_index: {err}")

        effective_proj = self.default_project or project or self.repo_root.name
        return {
            "synced": graphify_ok and memory_ok,
            "graphify_updated": graphify_ok,
            "memory_indexed": memory_ok,
            "project": effective_proj,
            "graphify": {
                "updated": graphify_ok,
                "output": graphify_out.strip(),
            },
            "memory": {
                "indexed": memory_ok,
                "project": effective_proj,
                "output": memory_out.strip(),
            },
            "graphify_output": graphify_out,
            "memory_output": memory_out,
            "errors": errors,
        }

    def gather_context(
        self,
        question: str,
        project: str | None = None,
        symbol_pattern: str | None = None,
        auto_build: bool = False,
    ) -> dict[str, Any]:
        """Gather combined architectural and symbol context from Graphify and Codebase Memory MCP."""
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        errors: list[str] = []
        graph_ctx = ""
        memory_ctx = ""
        arch_ctx = ""

        try:
            if auto_build and not self.graph_file.is_file():
                self.ensure_graphify_graph()
            graph_ctx = self.graphify_query(question)
        except Exception as err:
            errors.append(f"graphify_query: {err}")

        resolved_proj = self.resolve_memory_project(project)
        try:
            arch_ctx = self.memory_architecture(resolved_proj)
        except Exception as err:
            if auto_build:
                try:
                    self.memory_index(project=project)
                    resolved_proj = self.resolve_memory_project(project)
                    arch_ctx = self.memory_architecture(resolved_proj)
                except Exception as retry_err:
                    errors.append(f"memory_architecture: {retry_err}")
            else:
                errors.append(f"memory_architecture: {err}")

        if symbol_pattern and symbol_pattern.strip():
            try:
                memory_ctx = self.memory_query(resolved_proj, symbol_pattern.strip())
            except Exception as err:
                errors.append(f"memory_query: {err}")

        return {
            "question": question,
            "project": resolved_proj,
            "graphify_context": graph_ctx,
            "memory_context": memory_ctx,
            "memory_symbols": memory_ctx,
            "architecture_overview": arch_ctx,
            "memory_architecture": arch_ctx,
            "errors": errors,
        }
