"""Adapter for Graphify, Codebase Memory MCP, and deterministic AST repository intelligence."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any

_IGNORED_DIRS_REASONS: dict[str, str] = {
    ".git": "Git version-control metadata directory",
    "__pycache__": "Python bytecode cache directory",
    "venv": "Python virtual environment directory",
    ".venv": "Python virtual environment directory",
    "env": "Python virtual environment directory",
    ".env": "Environment directory",
    "node_modules": "Node.js dependency directory",
    ".ntg": "NTG runtime router state and discovery cache directory",
    "graphify-out": "Generated Graphify output and AST cache directory",
    ".codebase-memory": "Codebase Memory MCP local index directory",
    ".pytest_cache": "Pytest cache directory",
    ".mypy_cache": "Mypy type-checking cache directory",
    ".ruff_cache": "Ruff linter cache directory",
    "build": "Build artifact directory",
    "dist": "Distribution artifact directory",
    ".idea": "IDE configuration directory",
    ".vscode": "Editor configuration directory",
}

_IGNORED_DIRS: frozenset[str] = frozenset(_IGNORED_DIRS_REASONS.keys())

_SECRET_FILENAMES: frozenset[str] = frozenset(
    {
        ".env",
        ".env.local",
        ".env.development",
        ".env.production",
    }
)

_BINARY_OR_CACHE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".pyc",
        ".pyo",
        ".pyd",
        ".so",
        ".dll",
        ".exe",
        ".bin",
        ".db",
        ".sqlite",
        ".sqlite3",
        ".whl",
        ".tar",
        ".gz",
        ".zip",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".ico",
        ".pdf",
    }
)

_CONFIG_DOC_FILENAMES: frozenset[str] = frozenset(
    {
        "README.md",
        "pyproject.toml",
        "requirements.txt",
        ".env.example",
        ".gitignore",
        "setup.py",
        "setup.cfg",
        "Makefile",
        "Dockerfile",
        "LICENSE",
        "MANIFEST.in",
    }
)

_CONFIG_DOC_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".md",
        ".rst",
        ".txt",
        ".toml",
        ".yaml",
        ".yml",
        ".json",
        ".ini",
        ".cfg",
        ".sh",
        ".ps1",
        ".bat",
    }
)


def _canonical_project_slug(repo_root: Path) -> str:
    """Compute the canonical Codebase Memory MCP project slug for a repository path."""
    resolved = str(repo_root.resolve())
    return re.sub(r"[^A-Za-z0-9]+", "-", resolved).strip("-")


def _first_line(text: str | None) -> str:
    """Return the first non-empty summary sentence/paragraph from a docstring."""
    if not text:
        return ""
    lines: list[str] = []
    for raw in text.strip().splitlines():
        stripped = raw.strip()
        if not stripped:
            if lines:
                break
            continue
        lines.append(stripped)
        if stripped.endswith("."):
            break
    summary = " ".join(lines).strip()
    return summary[:280]


class CodeIntelligence:
    """Adapter for Graphify, Codebase Memory MCP, and deterministic repository inventory."""

    def __init__(self, repo_root: str | Path, default_project: str | None = None):
        self.repo_root = Path(repo_root).expanduser().resolve()

        if not self.repo_root.is_dir():
            raise ValueError("Repository root must be an existing directory.")

        self.default_project = (
            default_project.strip()
            if isinstance(default_project, str) and default_project.strip()
            else None
        )

    def _resolve_executable(self, executable: str) -> str:
        """Resolve a CLI executable path reliably, preferring native binaries on Windows."""
        exe_str = executable.strip()
        if not exe_str:
            raise ValueError("Executable name cannot be empty.")

        stem = Path(exe_str).stem.lower()
        which_hit = shutil.which(exe_str)

        if os.name == "nt" and stem == "codebase-memory-mcp":
            candidates: list[Path] = []
            if which_hit:
                w_path = Path(which_hit).resolve()
                if w_path.suffix.lower() == ".exe" and w_path.is_file():
                    return str(w_path)
                candidates.append(
                    w_path.parent
                    / "node_modules"
                    / "codebase-memory-mcp"
                    / "bin"
                    / "codebase-memory-mcp.exe"
                )
                candidates.append(
                    w_path.parent.parent
                    / "codebase-memory-mcp"
                    / "bin"
                    / "codebase-memory-mcp.exe"
                )
            appdata = os.environ.get("APPDATA", "").strip()
            if appdata:
                candidates.append(
                    Path(appdata)
                    / "npm"
                    / "node_modules"
                    / "codebase-memory-mcp"
                    / "bin"
                    / "codebase-memory-mcp.exe"
                )
            for cand in candidates:
                if cand.is_file():
                    return str(cand)

        if which_hit:
            return which_hit

        return exe_str

    def _run(self, args: list[str], timeout: int = 30) -> str:
        """Run a fixed local tool command safely without invoking a shell."""
        if not args or not all(isinstance(a, str) and a for a in args):
            raise ValueError("Command arguments must be a non-empty list of strings.")

        cmd_args = list(args)
        is_real_subprocess = getattr(subprocess.run, "__module__", "") == "subprocess"
        if os.name == "nt" and is_real_subprocess:
            resolved_first = self._resolve_executable(cmd_args[0])
            if resolved_first != cmd_args[0]:
                cmd_args[0] = resolved_first

        try:
            result = subprocess.run(
                cmd_args,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
                shell=False,
                stdin=subprocess.DEVNULL,
            )
        except TypeError:
            result = subprocess.run(
                cmd_args,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                shell=False,
            )
        except FileNotFoundError as exc:
            resolved = self._resolve_executable(args[0])
            if resolved == args[0] and not shutil.which(resolved):
                if Path(args[0]).stem.lower() == "graphify" and sys.executable:
                    fallback_cmd = [sys.executable, "-m", "graphify", *args[1:]]
                    result = subprocess.run(
                        fallback_cmd,
                        cwd=self.repo_root,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=timeout,
                        check=False,
                        shell=False,
                        stdin=subprocess.DEVNULL,
                    )
                else:
                    raise RuntimeError(
                        f"Required code intelligence executable not found: {args[0]}"
                    ) from exc
            else:
                result = subprocess.run(
                    [resolved, *args[1:]],
                    cwd=self.repo_root,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
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
    # Repository Discovery & Exclusions
    # ------------------------------------------------------------------

    def _discover_repo_files_with_exclusions(
        self,
    ) -> tuple[list[Path], list[Path], list[dict[str, str]], list[dict[str, str]]]:
        """Discover `(python_files, config_doc_files, excluded_dirs, excluded_files)` under `self.repo_root`."""
        py_files: list[Path] = []
        other_files: list[Path] = []
        excluded_dirs: list[dict[str, str]] = []
        excluded_files: list[dict[str, str]] = []
        seen_excluded_dirs: set[str] = set()

        for current_root, dirs, files in os.walk(self.repo_root):
            root_path = Path(current_root)
            kept_dirs: list[str] = []
            for d in sorted(dirs):
                d_path = root_path / d
                rel_d = d_path.relative_to(self.repo_root).as_posix()
                if d in _IGNORED_DIRS:
                    # Record top-level or package-level excluded directories concisely
                    key = d if d == "__pycache__" else rel_d
                    if key not in seen_excluded_dirs:
                        seen_excluded_dirs.add(key)
                        excluded_dirs.append(
                            {
                                "path": "**/__pycache__" if d == "__pycache__" else rel_d,
                                "reason": _IGNORED_DIRS_REASONS.get(d, "Excluded directory"),
                            }
                        )
                    continue
                if d.endswith(".egg-info"):
                    if rel_d not in seen_excluded_dirs:
                        seen_excluded_dirs.add(rel_d)
                        excluded_dirs.append(
                            {"path": rel_d, "reason": "Packaging egg-info metadata directory"}
                        )
                    continue
                kept_dirs.append(d)
            dirs[:] = kept_dirs

            for fname in sorted(files):
                fpath = root_path / fname
                rel_f = fpath.relative_to(self.repo_root).as_posix()
                rel_parts = fpath.relative_to(self.repo_root).parts
                if any(p in _IGNORED_DIRS or p.endswith(".egg-info") for p in rel_parts[:-1]):
                    continue

                lower_name = fname.lower()
                suffix = fpath.suffix.lower()

                if lower_name in _SECRET_FILENAMES or (
                    lower_name.startswith(".env.") and lower_name != ".env.example"
                ):
                    excluded_files.append(
                        {
                            "path": rel_f,
                            "reason": "Excluded secret/environment credential file",
                        }
                    )
                    continue

                if suffix in _BINARY_OR_CACHE_EXTENSIONS:
                    excluded_files.append(
                        {
                            "path": rel_f,
                            "reason": f"Excluded binary or compiled artifact ({suffix})",
                        }
                    )
                    continue

                if suffix == ".py":
                    py_files.append(fpath)
                elif fname in _CONFIG_DOC_FILENAMES or suffix in _CONFIG_DOC_EXTENSIONS:
                    other_files.append(fpath)
                else:
                    excluded_files.append(
                        {
                            "path": rel_f,
                            "reason": "Non-source/non-configuration file",
                        }
                    )

        py_files.sort(key=lambda p: p.relative_to(self.repo_root).as_posix())
        other_files.sort(key=lambda p: p.relative_to(self.repo_root).as_posix())
        return py_files, other_files, excluded_dirs, excluded_files

    def _discover_repo_files(self) -> tuple[list[Path], list[Path]]:
        """Return sorted lists of `(python_files, config_and_doc_files)` under `self.repo_root`."""
        py_files, other_files, _, _ = self._discover_repo_files_with_exclusions()
        return py_files, other_files

    # ------------------------------------------------------------------
    # Graphify Operations & Freshness / Contamination Validation
    # ------------------------------------------------------------------

    @property
    def graph_file(self) -> Path:
        """Path to the Graphify knowledge graph JSON file."""
        return self.repo_root / "graphify-out" / "graph.json"

    @property
    def graph_report_file(self) -> Path:
        """Path to the Graphify markdown report file."""
        return self.repo_root / "graphify-out" / "GRAPH_REPORT.md"

    def inspect_graphify_status(self) -> dict[str, Any]:
        """Validate `graphify-out/graph.json` against the current repository root and files."""
        status: dict[str, Any] = {
            "graph_file": str(self.graph_file),
            "verified": True,
            "exists": self.graph_file.is_file(),
            "valid": False,
            "fresh": False,
            "needs_refresh": True,
            "contaminated": False,
            "mismatched_root": False,
            "node_count": 0,
            "edge_count": 0,
            "indexed_files": [],
            "deleted_files": [],
            "missing_files": [],
            "modified_files": [],
            "reasons": [],
        }

        if not self.graph_file.is_file():
            status["reasons"].append("graphify-out/graph.json does not exist")
            return status

        root_marker = self.repo_root / "graphify-out" / ".graphify_root"
        if root_marker.is_file():
            try:
                recorded_root = root_marker.read_text(encoding="utf-8").strip()
                if recorded_root and recorded_root != ".":
                    cand_root = Path(recorded_root)
                    if cand_root.is_absolute() and cand_root.resolve() != self.repo_root:
                        status["mismatched_root"] = True
                        status["contaminated"] = True
                        status["reasons"].append(
                            f".graphify_root points to {cand_root} instead of {self.repo_root}"
                        )
            except OSError:
                pass

        try:
            raw_data = json.loads(self.graph_file.read_text(encoding="utf-8"))
        except Exception as exc:
            status["reasons"].append(f"Invalid JSON in graph.json: {exc}")
            status["contaminated"] = True
            return status

        if not isinstance(raw_data, dict) or not isinstance(raw_data.get("nodes"), list):
            status["reasons"].append("graph.json is missing a 'nodes' list")
            status["contaminated"] = True
            return status

        nodes = raw_data.get("nodes", [])
        edges = raw_data.get("links", raw_data.get("edges", []))
        status["valid"] = len(nodes) > 0
        status["node_count"] = len(nodes)
        status["edge_count"] = len(edges) if isinstance(edges, list) else 0

        graph_source_files: set[str] = set()
        deleted_files: set[str] = set()
        untagged_code_nodes = 0

        for node in nodes:
            if not isinstance(node, dict):
                continue
            sf = node.get("source_file")
            if not isinstance(sf, str) or not sf.strip():
                continue
            norm_sf = sf.strip().replace("\\", "/")
            if norm_sf.startswith("./"):
                norm_sf = norm_sf[2:]
            cand = Path(norm_sf)
            resolved_sf = (
                cand.resolve()
                if cand.is_absolute()
                else (self.repo_root / norm_sf).resolve()
            )
            try:
                rel_sf = resolved_sf.relative_to(self.repo_root).as_posix()
            except ValueError:
                deleted_files.add(norm_sf)
                status["mismatched_root"] = True
                continue

            graph_source_files.add(rel_sf)
            if not resolved_sf.is_file():
                deleted_files.add(rel_sf)
            elif node.get("file_type") == "code" and not node.get("_origin"):
                untagged_code_nodes += 1

        py_files, _ = self._discover_repo_files()
        try:
            graph_mtime = self.graph_file.stat().st_mtime
        except OSError:
            graph_mtime = 0.0

        missing_files: list[str] = []
        modified_files: list[str] = []

        for py_path in py_files:
            rel_py = py_path.relative_to(self.repo_root).as_posix()
            try:
                content = py_path.read_text(encoding="utf-8", errors="replace").strip()
                mtime = py_path.stat().st_mtime
            except OSError:
                continue

            if not content:
                continue

            has_ast_symbols = False
            try:
                tree = ast.parse(content)
                has_ast_symbols = any(
                    isinstance(
                        stmt,
                        (
                            ast.FunctionDef,
                            ast.AsyncFunctionDef,
                            ast.ClassDef,
                            ast.Assign,
                            ast.AnnAssign,
                            ast.Expr,
                        ),
                    )
                    for stmt in tree.body
                )
            except SyntaxError:
                has_ast_symbols = True

            if has_ast_symbols and rel_py not in graph_source_files:
                missing_files.append(rel_py)

            if mtime > graph_mtime + 1.0:
                modified_files.append(rel_py)

        status["indexed_files"] = sorted(graph_source_files)
        status["deleted_files"] = sorted(deleted_files)
        status["missing_files"] = sorted(missing_files)
        status["modified_files"] = sorted(modified_files)

        if deleted_files:
            status["contaminated"] = True
            status["reasons"].append(
                f"Graph references {len(deleted_files)} deleted file(s): {', '.join(sorted(deleted_files)[:8])}"
            )
        if missing_files:
            status["reasons"].append(
                f"Graph is missing {len(missing_files)} current file(s): {', '.join(sorted(missing_files)[:8])}"
            )
        if modified_files:
            status["reasons"].append(
                f"{len(modified_files)} file(s) modified since graph.json was built"
            )
            if untagged_code_nodes > 0:
                status["contaminated"] = True
                status["reasons"].append(
                    "Legacy graph nodes lack '_origin' metadata for clean incremental eviction"
                )

        needs_refresh = bool(
            not status["valid"]
            or status["contaminated"]
            or status["mismatched_root"]
            or deleted_files
            or missing_files
            or modified_files
            or status["node_count"] == 0
        )
        status["needs_refresh"] = needs_refresh
        status["fresh"] = status["valid"] and not needs_refresh
        return status

    def _clean_contaminated_graphify_artifacts(self) -> None:
        """Remove contaminated `graph.json` and legacy flat cache files prior to a clean rebuild."""
        if self.graph_file.is_file():
            try:
                self.graph_file.unlink()
            except OSError:
                pass
        cache_dir = self.repo_root / "graphify-out" / "cache"
        if cache_dir.is_dir():
            for item in cache_dir.glob("*.json"):
                if item.is_file():
                    try:
                        item.unlink()
                    except OSError:
                        pass

    def graphify_query(self, question: str, budget: int = 2500, dfs: bool = False) -> str:
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

    def graphify_update(
        self,
        target_path: str = ".",
        force: bool = True,
        clean_if_contaminated: bool = True,
    ) -> str:
        """Re-extract code files and update the Graphify graph (no LLM required)."""
        if not target_path or not target_path.strip():
            raise ValueError("Target path cannot be empty.")

        if clean_if_contaminated and self.graph_file.is_file():
            status = self.inspect_graphify_status()
            if status.get("contaminated") or force:
                self._clean_contaminated_graphify_artifacts()

        cmd = ["graphify", "update", target_path.strip()]
        return self._run(cmd, timeout=90)

    def ensure_graphify_graph(self, force_refresh: bool = False) -> Path:
        """Ensure `graphify-out/graph.json` exists and is verified fresh and uncontaminated."""
        status = self.inspect_graphify_status()
        if force_refresh or status["needs_refresh"]:
            if status.get("contaminated"):
                self._clean_contaminated_graphify_artifacts()
            self.graphify_update(".", force=True, clean_if_contaminated=False)
            post_status = self.inspect_graphify_status()
            if not post_status["valid"]:
                raise RuntimeError(
                    f"Graphify rebuild did not produce a valid graph: {'; '.join(post_status['reasons'])}"
                )
            if post_status["deleted_files"]:
                raise RuntimeError(
                    f"Graphify graph still contains deleted files after rebuild: {post_status['deleted_files']}"
                )
            if post_status["missing_files"]:
                raise RuntimeError(
                    f"Graphify graph is missing repository files after rebuild: {post_status['missing_files']}"
                )
        return self.graph_file

    def read_graphify_report_summary(self, max_chars: int = 3500) -> str:
        """Read high-signal sections (Summary, God Nodes, Surprising Connections) from GRAPH_REPORT.md."""
        if not self.graph_report_file.is_file():
            return ""
        try:
            text = self.graph_report_file.read_text(encoding="utf-8", errors="replace").strip()
            return text[:max_chars]
        except OSError:
            return ""

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
    # Codebase Memory MCP Operations & Index Integrity Validation
    # ------------------------------------------------------------------

    def list_memory_projects(self) -> list[dict[str, Any]]:
        """Return the list of indexed projects from Codebase Memory MCP."""
        out = self._run(
            [
                "codebase-memory-mcp",
                "cli",
                "--quiet",
                "list_projects",
                json.dumps({"format": "json"}),
            ]
        )
        data = json.loads(out)
        if isinstance(data, dict) and isinstance(data.get("projects"), list):
            return [p for p in data["projects"] if isinstance(p, dict)]
        return []

    def is_memory_project_indexed(self, project: str | None = None) -> tuple[bool, str | None]:
        """Check whether `self.repo_root` is currently indexed in Codebase Memory MCP."""
        try:
            projects = self.list_memory_projects()
        except Exception:
            return False, None

        canonical_slug = _canonical_project_slug(self.repo_root)
        matching_names: list[str] = []
        for entry in projects:
            root_str = entry.get("root_path")
            name = entry.get("name")
            if not isinstance(root_str, str) or not isinstance(name, str) or not name.strip():
                continue
            try:
                if Path(root_str).resolve() == self.repo_root:
                    matching_names.append(name.strip())
            except Exception:
                continue

        if not matching_names:
            return False, None

        if project and project.strip() in matching_names:
            return True, project.strip()
        if canonical_slug in matching_names:
            return True, canonical_slug
        for name in matching_names:
            if name.lower() == canonical_slug.lower():
                return True, name
        matching_names.sort(key=len)
        return True, matching_names[0]

    def resolve_memory_project(
        self,
        preferred_project: str | None = None,
        projects: list[dict[str, Any]] | None = None,
    ) -> str:
        """Resolve the indexed project name corresponding to `self.repo_root` in Codebase Memory MCP.
        Never silently selects an unrelated project belonging to a different directory.
        """
        pref = (
            preferred_project.strip()
            if isinstance(preferred_project, str) and preferred_project.strip()
            else None
        )
        canonical_slug = _canonical_project_slug(self.repo_root)

        try:
            proj_list = projects if projects is not None else self.list_memory_projects()
            matching_repo_projects: list[str] = []
            unrelated_project_names: set[str] = set()

            for entry in proj_list:
                root_str = entry.get("root_path")
                name = entry.get("name")
                if not isinstance(name, str) or not name.strip():
                    continue
                clean_name = name.strip()
                if isinstance(root_str, str) and root_str.strip():
                    try:
                        if Path(root_str).resolve() == self.repo_root:
                            matching_repo_projects.append(clean_name)
                        else:
                            unrelated_project_names.add(clean_name)
                    except Exception:
                        unrelated_project_names.add(clean_name)

            if matching_repo_projects:
                if pref and pref in matching_repo_projects:
                    self.default_project = pref
                    return pref
                if canonical_slug in matching_repo_projects:
                    self.default_project = canonical_slug
                    return canonical_slug
                for candidate in matching_repo_projects:
                    if candidate.lower() == canonical_slug.lower():
                        self.default_project = candidate
                        return candidate
                if self.default_project and self.default_project in matching_repo_projects:
                    return self.default_project
                matching_repo_projects.sort(key=len)
                self.default_project = matching_repo_projects[0]
                return matching_repo_projects[0]

            if pref and pref not in unrelated_project_names:
                return pref
        except Exception:
            if pref:
                return pref

        if self.default_project:
            return self.default_project
        return canonical_slug

    def inspect_memory_status(self, project: str | None = None) -> dict[str, Any]:
        """Validate the Codebase Memory MCP index against `self.repo_root` and current source files."""
        canonical_slug = _canonical_project_slug(self.repo_root)
        status: dict[str, Any] = {
            "project": project or self.default_project or canonical_slug,
            "canonical_project": canonical_slug,
            "verified": True,
            "exists": False,
            "indexed": False,
            "valid": False,
            "fresh": False,
            "needs_refresh": True,
            "contaminated": False,
            "mismatched_root": False,
            "indexed_files": [],
            "deleted_files": [],
            "missing_files": [],
            "stale_files": [],
            "reasons": [],
        }

        try:
            projects = self.list_memory_projects()
        except Exception as exc:
            status["reasons"].append(f"Failed to list Codebase Memory projects: {exc}")
            return status

        matching_entries: list[dict[str, Any]] = []
        pref = project.strip() if isinstance(project, str) and project.strip() else None

        for entry in projects:
            root_str = entry.get("root_path")
            name = entry.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            clean_name = name.strip()
            if isinstance(root_str, str) and root_str.strip():
                try:
                    resolved_entry_root = Path(root_str).resolve()
                    if resolved_entry_root == self.repo_root:
                        matching_entries.append(entry)
                    elif pref and clean_name == pref:
                        status["mismatched_root"] = True
                        status["contaminated"] = True
                        status["reasons"].append(
                            f"Preferred project '{pref}' points to unrelated root {resolved_entry_root} instead of {self.repo_root}"
                        )
                except Exception:
                    continue

        if not matching_entries:
            status["reasons"].append(
                f"Repository root {self.repo_root} is not indexed in Codebase Memory MCP"
            )
            return status

        resolved_proj = self.resolve_memory_project(project, projects=projects)
        status["project"] = resolved_proj
        status["exists"] = True
        status["indexed"] = True

        try:
            raw_files = self.memory_query(
                resolved_proj,
                ".*",
                limit=500,
                label="File",
            )
            file_data = json.loads(raw_files)
            raw_modules = self.memory_query(
                resolved_proj,
                ".*",
                limit=500,
                label="Module",
            )
            mod_data = json.loads(raw_modules)
        except Exception as exc:
            status["reasons"].append(f"Failed to query File/Module nodes for '{resolved_proj}': {exc}")
            return status

        if not isinstance(mod_data, dict) or not isinstance(mod_data.get("groups"), list):
            status["reasons"].append(f"Unexpected Module query payload for '{resolved_proj}'")
            return status

        indexed_files: set[str] = set()
        deleted_files: set[str] = set()
        stale_files: set[str] = set()
        indexed_line_counts: dict[str, int] = {}

        combined_groups: list[Any] = []
        if isinstance(file_data, dict) and isinstance(file_data.get("groups"), list):
            combined_groups.extend(file_data["groups"])
        combined_groups.extend(mod_data.get("groups", []))

        for grp in combined_groups:
            if not isinstance(grp, dict):
                continue
            rel_file = grp.get("file")
            if not isinstance(rel_file, str) or not rel_file.strip():
                continue
            norm_rel = rel_file.strip().replace("\\", "/")
            indexed_files.add(norm_rel)

            rows = grp.get("rows")
            if isinstance(rows, list) and rows and isinstance(rows[0], list) and len(rows[0]) >= 3:
                line_range = str(rows[0][2] or "")
                if "-" in line_range:
                    try:
                        end_line = int(line_range.split("-")[-1])
                        indexed_line_counts[norm_rel] = end_line
                    except ValueError:
                        pass

            disk_path = (self.repo_root / norm_rel).resolve()
            try:
                disk_path.relative_to(self.repo_root)
            except ValueError:
                deleted_files.add(norm_rel)
                status["mismatched_root"] = True
                continue

            if not disk_path.is_file():
                deleted_files.add(norm_rel)

        py_files, _ = self._discover_repo_files()
        missing_files: list[str] = []

        for py_path in py_files:
            rel_py = py_path.relative_to(self.repo_root).as_posix()
            try:
                text = py_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            if not text.strip():
                continue

            if rel_py not in indexed_files:
                missing_files.append(rel_py)
            elif rel_py in indexed_line_counts:
                disk_lines = len(text.splitlines())
                idx_lines = indexed_line_counts[rel_py]
                if abs(disk_lines - idx_lines) > 1:
                    stale_files.add(rel_py)

        status["valid"] = len(indexed_files) > 0
        status["indexed_files"] = sorted(indexed_files)
        status["deleted_files"] = sorted(deleted_files)
        status["missing_files"] = sorted(missing_files)
        status["stale_files"] = sorted(stale_files)

        if deleted_files:
            status["contaminated"] = True
            status["reasons"].append(
                f"Codebase Memory index references {len(deleted_files)} deleted file(s): {', '.join(sorted(deleted_files)[:8])}"
            )
        if missing_files:
            status["reasons"].append(
                f"Codebase Memory index is missing {len(missing_files)} current Python file(s): {', '.join(sorted(missing_files)[:8])}"
            )
        if stale_files:
            status["reasons"].append(
                f"Codebase Memory index has outdated line counts for {len(stale_files)} modified file(s): {', '.join(sorted(stale_files)[:8])}"
            )

        needs_refresh = bool(
            not status["valid"]
            or status["contaminated"]
            or status["mismatched_root"]
            or deleted_files
            or missing_files
            or stale_files
        )
        status["needs_refresh"] = needs_refresh
        status["fresh"] = status["valid"] and not needs_refresh
        return status

    def ensure_memory_index(
        self,
        project: str | None = None,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        """Ensure Codebase Memory MCP index for `self.repo_root` exists, is fresh, and is verified."""
        status = self.inspect_memory_status(project)
        if force_refresh or status["needs_refresh"]:
            self.memory_index(project=project, mode="fast")
            post_status = self.inspect_memory_status(project)
            if not post_status["valid"]:
                raise RuntimeError(
                    f"Codebase Memory indexing did not produce a valid index: {'; '.join(post_status['reasons'])}"
                )
            if post_status["deleted_files"]:
                raise RuntimeError(
                    f"Codebase Memory index still contains deleted files after re-index: {post_status['deleted_files']}"
                )
            if post_status["missing_files"]:
                raise RuntimeError(
                    f"Codebase Memory index is missing files after re-index: {post_status['missing_files']}"
                )
            return post_status
        return status

    def memory_query(
        self,
        project: str,
        symbol_pattern: str,
        limit: int = 100,
        label: str | None = None,
    ) -> str:
        """Search indexed code symbols using Codebase Memory MCP JSON CLI."""
        if not project or not project.strip():
            raise ValueError("Project name cannot be empty.")

        if not symbol_pattern or not symbol_pattern.strip():
            raise ValueError("Symbol pattern cannot be empty.")

        payload: dict[str, Any] = {
            "project": project.strip(),
            "name_pattern": symbol_pattern.strip(),
            "limit": int(limit),
            "format": "json",
        }
        if isinstance(label, str) and label.strip():
            payload["label"] = label.strip()

        return self._run(
            [
                "codebase-memory-mcp",
                "cli",
                "--quiet",
                "search_graph",
                json.dumps(payload),
            ]
        )

    def memory_index(self, project: str | None = None, mode: str = "fast") -> str:
        """Index or refresh the repository in Codebase Memory MCP."""
        payload: dict[str, Any] = {
            "repo_path": str(self.repo_root),
            "mode": mode,
            "format": "json",
        }
        output = self._run(
            [
                "codebase-memory-mcp",
                "cli",
                "--quiet",
                "index_repository",
                json.dumps(payload),
            ],
            timeout=90,
        )
        try:
            parsed = json.loads(output)
            if isinstance(parsed, dict) and isinstance(parsed.get("project"), str):
                self.default_project = parsed["project"].strip()
        except (ValueError, TypeError):
            pass
        return output

    def memory_architecture(
        self,
        project: str,
        aspects: str | list[str] = "all",
    ) -> str:
        """Retrieve architectural structure (file tree, layers, packages, hotspots, boundaries)
        from Codebase Memory MCP.
        """
        if not project or not project.strip():
            raise ValueError("Project name cannot be empty.")

        if isinstance(aspects, list):
            raw_aspects = [str(a).strip() for a in aspects if str(a).strip()]
        elif isinstance(aspects, str) and aspects.strip():
            raw_aspects = [a.strip() for a in aspects.split(",") if a.strip()]
        else:
            raw_aspects = ["all"]

        normalized_aspects = [
            "all" if a.lower() in ("overview", "all", "*") else a for a in raw_aspects
        ]
        if not normalized_aspects or "all" in normalized_aspects:
            normalized_aspects = ["all"]

        payload: dict[str, Any] = {
            "project": project.strip(),
            "aspects": normalized_aspects,
            "format": "json",
        }
        return self._run(
            [
                "codebase-memory-mcp",
                "cli",
                "--quiet",
                "get_architecture",
                json.dumps(payload),
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

        payload: dict[str, Any] = {
            "project": project.strip(),
            "function_name": function_name.strip(),
            "direction": direction,
            "depth": int(depth),
            "format": "json",
        }
        return self._run(
            [
                "codebase-memory-mcp",
                "cli",
                "--quiet",
                "trace_path",
                json.dumps(payload),
            ]
        )

    def memory_detect_changes(self, project: str, base_branch: str = "main") -> str:
        """Detect changed symbols and their architectural impact radius."""
        if not project or not project.strip():
            raise ValueError("Project name cannot be empty.")

        payload: dict[str, Any] = {
            "project": project.strip(),
            "base_branch": base_branch,
            "format": "json",
        }
        return self._run(
            [
                "codebase-memory-mcp",
                "cli",
                "--quiet",
                "detect_changes",
                json.dumps(payload),
            ]
        )

    def memory_get_snippet(self, project: str, qualified_name: str) -> str:
        """Retrieve source code snippet for a qualified symbol name."""
        if not project or not project.strip():
            raise ValueError("Project name cannot be empty.")
        if not qualified_name or not qualified_name.strip():
            raise ValueError("Qualified name cannot be empty.")

        payload: dict[str, Any] = {
            "project": project.strip(),
            "qualified_name": qualified_name.strip(),
            "format": "json",
        }
        return self._run(
            [
                "codebase-memory-mcp",
                "cli",
                "--quiet",
                "get_code_snippet",
                json.dumps(payload),
            ]
        )

    # ------------------------------------------------------------------
    # Deterministic Repository Inventory & Dependency-Aware Reading Order
    # ------------------------------------------------------------------

    def _module_map_for_repo(self, py_files: list[Path]) -> dict[str, str]:
        """Map dotted Python module names to relative POSIX file paths in `self.repo_root`."""
        mod_to_rel: dict[str, str] = {}
        for py_path in py_files:
            rel = py_path.relative_to(self.repo_root).as_posix()
            no_ext = rel[:-3] if rel.endswith(".py") else rel
            parts = no_ext.split("/")
            if parts[-1] == "__init__":
                pkg_parts = parts[:-1]
                if pkg_parts:
                    mod_to_rel[".".join(pkg_parts)] = rel
            else:
                mod_to_rel[".".join(parts)] = rel
        return mod_to_rel

    def _resolve_import_targets(
        self,
        current_rel: str,
        module_name: str | None,
        names: list[str],
        level: int,
        mod_to_rel: dict[str, str],
    ) -> tuple[set[str], set[str]]:
        """Resolve an import statement into `(internal_rel_paths, external_top_packages)`."""
        internal: set[str] = set()
        external: set[str] = set()

        current_parts = current_rel[:-3].split("/") if current_rel.endswith(".py") else []
        if current_parts and current_parts[-1] == "__init__":
            current_pkg_parts = current_parts[:-1]
        else:
            current_pkg_parts = current_parts[:-1]

        base_mod = module_name or ""
        if level > 0:
            up = max(0, len(current_pkg_parts) - (level - 1))
            prefix_parts = current_pkg_parts[:up]
            if base_mod:
                base_mod = ".".join([*prefix_parts, base_mod])
            else:
                base_mod = ".".join(prefix_parts)

        matched_internal = False
        if base_mod:
            if base_mod in mod_to_rel:
                target_rel = mod_to_rel[base_mod]
                if target_rel != current_rel:
                    internal.add(target_rel)
                matched_internal = True

            for imported_symbol in names:
                if imported_symbol == "*":
                    continue
                sub_cand = f"{base_mod}.{imported_symbol}"
                if sub_cand in mod_to_rel:
                    target_rel = mod_to_rel[sub_cand]
                    if target_rel != current_rel:
                        internal.add(target_rel)
                    matched_internal = True

            if not matched_internal and "." in base_mod:
                parts = base_mod.split(".")
                for i in range(len(parts) - 1, 0, -1):
                    prefix = ".".join(parts[:i])
                    if prefix in mod_to_rel:
                        target_rel = mod_to_rel[prefix]
                        if target_rel != current_rel:
                            internal.add(target_rel)
                        matched_internal = True
                        break

        if not matched_internal and level == 0:
            if base_mod:
                top_pkg = base_mod.split(".")[0]
                if top_pkg:
                    external.add(top_pkg)
            else:
                for imported_symbol in names:
                    top_pkg = imported_symbol.split(".")[0]
                    if top_pkg in mod_to_rel:
                        target_rel = mod_to_rel[top_pkg]
                        if target_rel != current_rel:
                            internal.add(target_rel)
                    elif top_pkg:
                        external.add(top_pkg)

        return internal, external

    def _analyze_python_file(
        self,
        py_path: Path,
        mod_to_rel: dict[str, str],
    ) -> dict[str, Any]:
        """Extract docstring, classes, functions, imports, and entry-point info from a Python file."""
        rel_path = py_path.relative_to(self.repo_root).as_posix()
        try:
            source = py_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            source = ""

        line_count = len(source.splitlines())
        info: dict[str, Any] = {
            "path": rel_path,
            "type": "python",
            "lines": line_count,
            "docstring": "",
            "summary": "",
            "classes": [],
            "functions": [],
            "exports": [],
            "internal_dependencies": [],
            "external_dependencies": [],
            "depended_on_by": [],
            "is_entry_point": rel_path in ("main.py", "test.py") or rel_path.endswith("__main__.py"),
            "parse_error": None,
        }

        if not source.strip():
            info["summary"] = "Empty module marker."
            return info

        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            info["parse_error"] = f"SyntaxError at line {exc.lineno}: {exc.msg}"
            info["summary"] = f"Python source file (syntax error at line {exc.lineno})."
            return info

        doc = _first_line(ast.get_docstring(tree))
        info["docstring"] = doc

        classes: list[dict[str, Any]] = []
        functions: list[dict[str, Any]] = []
        exports: list[str] = []
        internal_deps: set[str] = set()
        external_deps: set[str] = set()

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                methods = [
                    item.name
                    for item in node.body
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and (not item.name.startswith("_") or item.name == "__init__")
                ]
                bases: list[str] = []
                for b in node.bases:
                    try:
                        bases.append(ast.unparse(b))
                    except Exception:
                        pass
                classes.append(
                    {
                        "name": node.name,
                        "bases": bases,
                        "methods": methods[:12],
                        "docstring": _first_line(ast.get_docstring(node)),
                        "line": node.lineno,
                    }
                )
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(
                    {
                        "name": node.name,
                        "docstring": _first_line(ast.get_docstring(node)),
                        "line": node.lineno,
                    }
                )
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, (ast.List, ast.Tuple, ast.Set)):
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                    exports.append(elt.value)
            elif isinstance(node, ast.If):
                try:
                    cond_str = ast.unparse(node.test)
                    if "__name__" in cond_str and "__main__" in cond_str:
                        info["is_entry_point"] = True
                except Exception:
                    pass

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
                int_d, ext_d = self._resolve_import_targets(
                    rel_path, None, names, 0, mod_to_rel
                )
                internal_deps.update(int_d)
                external_deps.update(ext_d)
            elif isinstance(node, ast.ImportFrom):
                names = [alias.name for alias in node.names]
                int_d, ext_d = self._resolve_import_targets(
                    rel_path, node.module, names, node.level or 0, mod_to_rel
                )
                internal_deps.update(int_d)
                external_deps.update(ext_d)

        info["classes"] = classes
        info["functions"] = functions
        info["exports"] = exports or [c["name"] for c in classes] + [
            f["name"] for f in functions if not f["name"].startswith("_")
        ]
        info["internal_dependencies"] = sorted(internal_deps)
        info["external_dependencies"] = sorted(external_deps)

        if doc:
            info["summary"] = doc
        elif rel_path == "test.py":
            info["summary"] = (
                "Top-level verification/demo script that invokes `query_directory(...)` "
                "from `ntg` to explain the codebase and output a dependency-aware file reading order."
            )
        elif classes or functions:
            sym_list = [c["name"] for c in classes] + [f["name"] for f in functions]
            info["summary"] = f"Defines {', '.join(sym_list[:6])}."
        elif exports:
            info["summary"] = f"Package interface re-exporting {', '.join(exports[:8])}."
        else:
            info["summary"] = f"Python module `{rel_path}`."

        return info

    def _analyze_config_or_doc_file(self, fpath: Path) -> dict[str, Any]:
        """Extract a factual summary of a non-Python project configuration or documentation file."""
        rel_path = fpath.relative_to(self.repo_root).as_posix()
        try:
            text = fpath.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""

        lines = text.splitlines()
        summary = f"Project file `{rel_path}`."
        fname = fpath.name

        if fname == "README.md":
            headings = [
                ln.lstrip("#").strip()
                for ln in lines
                if ln.strip().startswith("#")
            ]
            title = headings[0] if headings else "README"
            summary = (
                f"Primary project documentation ({title}) covering architecture, "
                "multi-provider routing, CLI commands, and setup."
            )
        elif fname == "pyproject.toml":
            summary = (
                "Python package build & metadata configuration (`ntg` v2.0.0, Python >=3.9, "
                "dependencies: litellm, httpx, pydantic, python-dotenv; console script `ntg = ntg.cli.app:main`)."
            )
        elif fname == "requirements.txt":
            pkgs = [
                ln.split(">=")[0].split("==")[0].strip()
                for ln in lines
                if ln.strip() and not ln.strip().startswith("#")
            ]
            summary = (
                f"Runtime and development dependency specification ({', '.join(pkgs[:8])})."
                if pkgs
                else "Python dependency requirements file."
            )
        elif fname == ".env.example":
            env_keys = [
                ln.split("=")[0].strip()
                for ln in lines
                if "=" in ln and not ln.strip().startswith("#")
            ]
            summary = (
                f"Template environment configuration documenting API key and runtime variables "
                f"({', '.join(env_keys[:8])})."
                if env_keys
                else "Environment variable template file."
            )
        elif fname == ".gitignore":
            summary = "Git ignore rules excluding virtualenvs, caches, `.env`, `.ntg/`, and build artifacts."

        return {
            "path": rel_path,
            "type": "doc" if fname.lower().endswith(".md") else "config",
            "lines": len(lines),
            "docstring": "",
            "summary": summary,
            "classes": [],
            "functions": [],
            "exports": [],
            "internal_dependencies": [],
            "external_dependencies": [],
            "depended_on_by": [],
            "is_entry_point": False,
            "parse_error": None,
        }

    def _classify_layer(self, file_info: dict[str, Any]) -> tuple[int, str, int]:
        """Classify a file into `(layer_order, layer_name, intra_layer_priority)` for reading order."""
        rel = file_info["path"]
        ftype = file_info["type"]
        parts = rel.split("/")
        is_init = parts[-1] == "__init__.py"

        if ftype in ("doc", "config"):
            order_map = {
                "README.md": 0,
                "pyproject.toml": 1,
                "requirements.txt": 2,
                ".env.example": 3,
                ".gitignore": 4,
            }
            return (1, "1. Project Overview & Configuration", order_map.get(rel, 10))

        if rel in ("main.py", "ntg/__main__.py", "test.py"):
            ep_map = {"main.py": 0, "ntg/__main__.py": 1, "test.py": 2}
            return (2, "2. Execution Entry Points", ep_map.get(rel, 5))

        if rel == "ntg/__init__.py":
            return (2, "2. Execution Entry Points", 3)

        if len(parts) >= 2 and parts[0] == "ntg":
            subpkg = parts[1]
            if subpkg == "core":
                prio_map = {
                    "ntg/core/utils.py": 0,
                    "ntg/core/config.py": 1,
                    "ntg/core/models.py": 2,
                    "ntg/core/exceptions.py": 3,
                    "ntg/core/__init__.py": 9,
                }
                return (
                    3,
                    "3. Core Foundation Layer (ntg/core/)",
                    prio_map.get(rel, 8 if is_init else 4),
                )
            if subpkg == "providers":
                prio_map = {
                    "ntg/providers/base.py": 0,
                    "ntg/providers/openrouter.py": 1,
                    "ntg/providers/groq.py": 2,
                    "ntg/providers/gemini.py": 3,
                    "ntg/providers/nvidia.py": 4,
                    "ntg/providers/cohere.py": 5,
                    "ntg/providers/discovery.py": 6,
                    "ntg/providers/__init__.py": 9,
                }
                return (
                    4,
                    "4. Provider Discovery & Adapter Layer (ntg/providers/)",
                    prio_map.get(rel, 8 if is_init else 5),
                )
            if subpkg == "router":
                prio_map = {
                    "ntg/router/requirements.py": 0,
                    "ntg/router/state.py": 1,
                    "ntg/router/telemetry.py": 2,
                    "ntg/router/engine.py": 3,
                    "ntg/router/__init__.py": 9,
                }
                return (
                    5,
                    "5. Smart Routing, Quota State & Telemetry Layer (ntg/router/)",
                    prio_map.get(rel, 8 if is_init else 4),
                )
            if subpkg == "agent":
                prio_map = {
                    "ntg/agent/code_intelligence.py": 0,
                    "ntg/agent/planner.py": 1,
                    "ntg/agent/verifier.py": 2,
                    "ntg/agent/orchestrator.py": 3,
                    "ntg/agent/__init__.py": 9,
                }
                return (
                    6,
                    "6. Architecture-Aware Coding Agent Layer (ntg/agent/)",
                    prio_map.get(rel, 8 if is_init else 4),
                )
            if subpkg == "cli":
                prio_map = {
                    "ntg/cli/diagnostics.py": 0,
                    "ntg/cli/app.py": 1,
                    "ntg/cli/__init__.py": 9,
                }
                return (
                    7,
                    "7. CLI & Diagnostics Layer (ntg/cli/)",
                    prio_map.get(rel, 8 if is_init else 4),
                )

        return (
            8,
            "8. Compatibility & Supporting Modules",
            9 if is_init else 5,
        )

    def _compute_reading_order(
        self,
        files_by_path: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Compute a deterministic, dependency-aware reading order across all project files."""
        layers: dict[int, tuple[str, list[dict[str, Any]]]] = {}
        for fpath, info in files_by_path.items():
            layer_num, layer_name, _ = self._classify_layer(info)
            if layer_num not in layers:
                layers[layer_num] = (layer_name, [])
            layers[layer_num][1].append(info)

        ordered_items: list[dict[str, Any]] = []
        step = 1

        for layer_num in sorted(layers.keys()):
            layer_name, layer_files = layers[layer_num]
            layer_paths = {f["path"] for f in layer_files}

            in_degree: dict[str, int] = {p: 0 for p in layer_paths}
            adj: dict[str, list[str]] = {p: [] for p in layer_paths}

            for f in layer_files:
                p = f["path"]
                for dep in f["internal_dependencies"]:
                    if dep in layer_paths and dep != p:
                        adj[dep].append(p)
                        in_degree[p] += 1

            ready = [p for p, deg in in_degree.items() if deg == 0]
            ready.sort(
                key=lambda p: (
                    self._classify_layer(files_by_path[p])[2],
                    len(files_by_path[p]["internal_dependencies"]),
                    p,
                )
            )

            sorted_layer_paths: list[str] = []
            while ready:
                curr = ready.pop(0)
                sorted_layer_paths.append(curr)
                for nxt in adj[curr]:
                    in_degree[nxt] -= 1
                    if in_degree[nxt] == 0:
                        ready.append(nxt)
                        ready.sort(
                            key=lambda p: (
                                self._classify_layer(files_by_path[p])[2],
                                len(files_by_path[p]["internal_dependencies"]),
                                p,
                            )
                        )

            if len(sorted_layer_paths) < len(layer_paths):
                remaining = [p for p in layer_paths if p not in sorted_layer_paths]
                remaining.sort(
                    key=lambda p: (
                        self._classify_layer(files_by_path[p])[2],
                        len(files_by_path[p]["internal_dependencies"]),
                        p,
                    )
                )
                sorted_layer_paths.extend(remaining)

            for p in sorted_layer_paths:
                info = files_by_path[p]
                deps = info["internal_dependencies"]
                rev_deps = info["depended_on_by"]
                if info["type"] in ("doc", "config"):
                    rationale = "Project metadata, configuration, and setup context read before source code."
                elif info["is_entry_point"]:
                    rationale = (
                        f"Top-level entry point invoking {', '.join(deps[:4])}."
                        if deps
                        else "Top-level execution entry point."
                    )
                elif not deps:
                    rationale = (
                        f"Zero internal dependencies (foundational leaf module); imported by {len(rev_deps)} module(s)."
                        if rev_deps
                        else "Zero internal dependencies (standalone module)."
                    )
                else:
                    rationale = (
                        f"Depends on {', '.join(deps[:5])}"
                        + (f" (+{len(deps) - 5} more)" if len(deps) > 5 else "")
                        + (
                            f"; imported by {', '.join(rev_deps[:4])}"
                            + (f" (+{len(rev_deps) - 4} more)" if len(rev_deps) > 4 else "")
                            if rev_deps
                            else "."
                        )
                    )

                key_symbols = [c["name"] for c in info["classes"]] + [
                    fn["name"] for fn in info["functions"] if not fn["name"].startswith("_")
                ]
                ordered_items.append(
                    {
                        "step": step,
                        "path": p,
                        "layer": layer_name,
                        "summary": info["summary"],
                        "key_symbols": key_symbols[:10],
                        "depends_on": deps,
                        "depended_on_by": rev_deps,
                        "rationale": rationale,
                    }
                )
                step += 1

        return ordered_items

    def build_repository_inventory(self) -> dict[str, Any]:
        """Build a complete, deterministic AST and filesystem inventory of the repository,
        including symbol map, dependency map, reading order, excluded files/dirs, and limitations.
        """
        py_files, other_files, excluded_dirs, excluded_files = (
            self._discover_repo_files_with_exclusions()
        )
        mod_to_rel = self._module_map_for_repo(py_files)

        files_by_path: dict[str, dict[str, Any]] = {}
        for fpath in other_files:
            info = self._analyze_config_or_doc_file(fpath)
            files_by_path[info["path"]] = info

        parse_errors: list[str] = []
        for py_path in py_files:
            info = self._analyze_python_file(py_path, mod_to_rel)
            files_by_path[info["path"]] = info
            if info.get("parse_error"):
                parse_errors.append(f"{info['path']}: {info['parse_error']}")

        for src_path, info in files_by_path.items():
            for target_dep in info["internal_dependencies"]:
                if (
                    target_dep in files_by_path
                    and src_path not in files_by_path[target_dep]["depended_on_by"]
                ):
                    files_by_path[target_dep]["depended_on_by"].append(src_path)

        symbol_map: dict[str, dict[str, Any]] = {}
        dependency_map: dict[str, dict[str, list[str]]] = {}

        for rel_path, info in sorted(files_by_path.items()):
            info["depended_on_by"].sort()
            dependency_map[rel_path] = {
                "depends_on": info["internal_dependencies"],
                "depended_on_by": info["depended_on_by"],
                "external_dependencies": info["external_dependencies"],
            }
            for cls in info.get("classes", []):
                symbol_map[f"{rel_path}:{cls['name']}"] = {
                    "name": cls["name"],
                    "kind": "class",
                    "file": rel_path,
                    "line": cls.get("line", 1),
                    "methods": cls.get("methods", []),
                    "docstring": cls.get("docstring", ""),
                }
            for fn in info.get("functions", []):
                symbol_map[f"{rel_path}:{fn['name']}"] = {
                    "name": fn["name"],
                    "kind": "function",
                    "file": rel_path,
                    "line": fn.get("line", 1),
                    "docstring": fn.get("docstring", ""),
                }

        reading_order = self._compute_reading_order(files_by_path)
        entry_points = [
            p for p, info in sorted(files_by_path.items()) if info.get("is_entry_point")
        ]

        coverage_limitations: list[str] = []
        if excluded_files:
            excl_f_summary = ", ".join(
                f"{item['path']} ({item['reason']})" for item in excluded_files[:10]
            )
            coverage_limitations.append(f"Excluded files: {excl_f_summary}")
        if excluded_dirs:
            excl_d_summary = ", ".join(
                f"{item['path']} ({item['reason']})" for item in excluded_dirs[:10]
            )
            coverage_limitations.append(f"Excluded directories: {excl_d_summary}")
        if parse_errors:
            coverage_limitations.append(
                f"Files with syntax/parse errors ({len(parse_errors)}): {'; '.join(parse_errors)}"
            )
        coverage_limitations.append(
            "Static AST dependency analysis captures all module-level and function-level `import`/`from ... import` edges; dynamic runtime dispatch (e.g., LiteLLM provider callback hooks) is supplemented via Graphify and Codebase Memory MCP."
        )

        return {
            "repo_root": str(self.repo_root),
            "repo_name": self.repo_root.name,
            "total_files": len(files_by_path),
            "python_file_count": len(py_files),
            "config_doc_file_count": len(other_files),
            "entry_points": entry_points,
            "files": [files_by_path[k] for k in sorted(files_by_path.keys())],
            "symbol_map": symbol_map,
            "dependency_map": dependency_map,
            "reading_order": reading_order,
            "excluded_dirs": excluded_dirs,
            "excluded_files": excluded_files,
            "parse_errors": parse_errors,
            "coverage_limitations": coverage_limitations,
        }

    def format_inventory_for_prompt(
        self,
        inventory: dict[str, Any] | None = None,
    ) -> str:
        """Render the complete file inventory, dependency-aware reading order, and exclusions as Markdown."""
        inv = inventory or self.build_repository_inventory()
        excl_dirs = [d["path"] for d in inv.get("excluded_dirs", [])]
        excl_files = [f"{f['path']} ({f['reason']})" for f in inv.get("excluded_files", [])]

        lines: list[str] = [
            f"- **Repository Root**: `{inv['repo_root']}`",
            f"- **Total Analyzed Project Files**: {inv['total_files']} ({inv['python_file_count']} Python modules, {inv['config_doc_file_count']} config/documentation files)",
            f"- **Execution Entry Points**: {', '.join(f'`{ep}`' for ep in inv['entry_points']) or 'None'}",
            f"- **Excluded Directories**: {', '.join(f'`{d}`' for d in excl_dirs) or 'None'}",
            f"- **Excluded Files**: {', '.join(f'`{f}`' for f in excl_files) or 'None'}",
            "",
            "### Dependency-Aware Reading Order & Complete File Inventory",
        ]

        current_layer = ""
        files_lookup = {f["path"]: f for f in inv.get("files", [])}

        for item in inv.get("reading_order", []):
            layer = item["layer"]
            if layer != current_layer:
                current_layer = layer
                lines.append(f"\n#### {current_layer}")

            path = item["path"]
            f_info = files_lookup.get(path, {})
            classes = [
                f"{c['name']}({', '.join(c['methods'][:5])})"
                if c.get("methods")
                else c["name"]
                for c in f_info.get("classes", [])
            ]
            funcs = [
                fn["name"]
                for fn in f_info.get("functions", [])
                if not fn["name"].startswith("_")
            ]
            deps = item.get("depends_on", [])
            rev_deps = item.get("depended_on_by", [])
            ext_deps = f_info.get("external_dependencies", [])

            details: list[str] = [
                f"**`{path}`** ({f_info.get('lines', 0)} lines): {item['summary']}"
            ]
            if classes:
                details.append(f"  - Classes: `{', '.join(classes)}`")
            if funcs:
                details.append(f"  - Functions: `{', '.join(funcs[:10])}`")
            if deps:
                details.append(
                    f"  - Internal Imports (`depends_on`): {', '.join(f'`{d}`' for d in deps)}"
                )
            else:
                details.append(
                    "  - Internal Imports (`depends_on`): None (leaf / standalone file)"
                )
            if rev_deps:
                details.append(
                    f"  - Imported By (`depended_on_by`): {', '.join(f'`{r}`' for r in rev_deps[:8])}"
                    + (f" (+{len(rev_deps) - 8} more)" if len(rev_deps) > 8 else "")
                )
            if ext_deps:
                details.append(f"  - External/Stdlib Imports: `{', '.join(ext_deps[:8])}`")

            lines.append(f"{item['step']}. " + "\n".join(details))

        limitations = inv.get("coverage_limitations", [])
        if limitations:
            lines.append("\n### Explicit Coverage & Static Analysis Notes")
            for lim in limitations:
                lines.append(f"- {lim}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Unified Context & Knowledge Synchronization
    # ------------------------------------------------------------------

    def refresh_knowledge(self, project: str | None = None) -> dict[str, Any]:
        """Rebuild/update both Graphify and Codebase Memory MCP indices and verify both after refresh."""
        errors: list[str] = []
        graphify_ok = False
        memory_ok = False
        graphify_out = ""
        memory_out = ""
        post_graph: dict[str, Any] = {}
        post_memory: dict[str, Any] = {}

        try:
            graphify_out = self.graphify_update(".", force=True, clean_if_contaminated=True)
            post_graph = self.inspect_graphify_status()
            graphify_ok = bool(post_graph["fresh"] and not post_graph["deleted_files"])
            if not graphify_ok:
                errors.append(
                    f"graphify_update post-refresh verification failed: {'; '.join(post_graph['reasons'])}"
                )
        except Exception as err:
            errors.append(f"graphify_update: {err}")

        try:
            memory_out = self.memory_index(project=project)
            post_memory = self.inspect_memory_status(project=project)
            memory_ok = bool(post_memory["fresh"] and not post_memory["deleted_files"])
            if not memory_ok:
                errors.append(
                    f"memory_index post-refresh verification failed: {'; '.join(post_memory['reasons'])}"
                )
        except Exception as err:
            errors.append(f"memory_index: {err}")

        effective_proj = post_memory.get("project") or self.resolve_memory_project(project)
        return {
            "synced": graphify_ok and memory_ok,
            "graphify_updated": graphify_ok,
            "memory_indexed": memory_ok,
            "project": effective_proj,
            "graphify": {
                "updated": graphify_ok,
                "verified": post_graph.get("fresh", False),
                "status": post_graph,
                "output": graphify_out.strip(),
            },
            "memory": {
                "indexed": memory_ok,
                "verified": post_memory.get("fresh", False),
                "project": effective_proj,
                "status": post_memory,
                "output": memory_out.strip(),
            },
            "graphify_output": graphify_out,
            "memory_output": memory_out,
            "errors": errors,
        }

    @staticmethod
    def _summarize_memory_architecture(raw_arch: str) -> str:
        """Compact Codebase Memory MCP `get_architecture` JSON output for high-signal prompting."""
        if not raw_arch or not raw_arch.strip():
            return ""
        try:
            data = json.loads(raw_arch)
        except Exception:
            return raw_arch[:4000]

        if not isinstance(data, dict):
            return raw_arch[:4000]

        compact: dict[str, Any] = {}
        for key in ("languages", "node_labels", "edge_types", "packages", "layers", "boundaries"):
            if key in data:
                compact[key] = data[key]
        if isinstance(data.get("hotspots"), list):
            compact["hotspots"] = data["hotspots"][:15]
        if isinstance(data.get("clusters"), list):
            compact["clusters"] = data["clusters"][:12]
        if isinstance(data.get("file_tree"), dict):
            compact["file_tree"] = data["file_tree"]

        return json.dumps(compact, indent=2)

    @staticmethod
    def _summarize_memory_symbols(raw_symbols: str) -> str:
        """Compact Codebase Memory MCP `search_graph` JSON output for high-signal prompting."""
        if not raw_symbols or not raw_symbols.strip():
            return ""
        try:
            data = json.loads(raw_symbols)
        except Exception:
            return raw_symbols[:4000]

        if not isinstance(data, dict):
            return raw_symbols[:4000]

        return json.dumps(data, indent=2)[:4500]

    def gather_context(
        self,
        question: str,
        project: str | None = None,
        symbol_pattern: str | None = None,
        auto_build: bool = False,
        strict: bool = False,
    ) -> dict[str, Any]:
        """Gather combined deterministic AST inventory, Graphify context, and Codebase Memory MCP
        architecture & symbol intelligence. Validates both indexes before and after refresh and
        never silently uses stale or contaminated index data.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        errors: list[str] = []
        graph_ctx = ""
        graph_report = ""
        memory_ctx = ""
        arch_ctx = ""

        # 1. Build deterministic repository inventory and dependency-aware reading order
        inventory = self.build_repository_inventory()
        formatted_inventory = self.format_inventory_for_prompt(inventory)
        if inventory.get("parse_errors"):
            errors.append(
                f"inventory_parse_errors: {'; '.join(inventory['parse_errors'])}"
            )

        # 2. Validate Graphify freshness & contamination; refresh only when needed and verify after refresh
        graph_status = self.inspect_graphify_status()
        try:
            if graph_status["needs_refresh"]:
                if auto_build:
                    self.ensure_graphify_graph(force_refresh=True)
                    graph_status = self.inspect_graphify_status()
                else:
                    reasons_str = "; ".join(graph_status["reasons"]) or "stale or missing graph"
                    errors.append(
                        f"graphify_status: Graphify index is not fresh ({reasons_str}) and auto_build=False"
                    )

            # Never silently use contaminated or invalid Graphify data
            if graph_status["valid"] and not graph_status["contaminated"] and graph_status["fresh"]:
                graph_ctx = self.graphify_query(question, budget=2500)
                graph_report = self.read_graphify_report_summary(max_chars=3000)
            elif graph_status["contaminated"]:
                errors.append(
                    f"graphify_contaminated: Refusing to use contaminated Graphify index ({'; '.join(graph_status['reasons'])})"
                )
        except Exception as err:
            errors.append(f"graphify_query: {err}")

        # 3. Validate Codebase Memory MCP index against repo_root and current files; refresh only when needed
        memory_status = self.inspect_memory_status(project)
        resolved_proj = memory_status["project"]

        try:
            if memory_status["needs_refresh"]:
                if auto_build:
                    memory_status = self.ensure_memory_index(project=project, force_refresh=True)
                    resolved_proj = memory_status["project"]
                else:
                    reasons_str = "; ".join(memory_status["reasons"]) or "stale or missing index"
                    errors.append(
                        f"memory_status: Codebase Memory index is not fresh ({reasons_str}) and auto_build=False"
                    )

            # Never silently use contaminated or invalid Codebase Memory MCP data
            if memory_status["valid"] and not memory_status["contaminated"] and memory_status["fresh"]:
                raw_arch = self.memory_architecture(resolved_proj, aspects="all")
                arch_ctx = self._summarize_memory_architecture(raw_arch)

                effective_pattern = (
                    symbol_pattern.strip()
                    if isinstance(symbol_pattern, str) and symbol_pattern.strip()
                    else ".*"
                )
                raw_symbols = self.memory_query(
                    resolved_proj,
                    effective_pattern,
                    limit=80,
                )
                memory_ctx = self._summarize_memory_symbols(raw_symbols)
            elif memory_status["contaminated"]:
                errors.append(
                    f"memory_contaminated: Refusing to use contaminated Codebase Memory index ({'; '.join(memory_status['reasons'])})"
                )
        except Exception as err:
            errors.append(f"memory_retrieval: {err}")

        all_sources_verified = bool(
            graph_status.get("verified")
            and graph_status.get("fresh")
            and memory_status.get("verified")
            and memory_status.get("fresh")
            and not errors
        )

        coverage = {
            "total_repo_files": inventory["total_files"],
            "python_files": inventory["python_file_count"],
            "config_doc_files": inventory["config_doc_file_count"],
            "excluded_dirs_count": len(inventory.get("excluded_dirs", [])),
            "excluded_files_count": len(inventory.get("excluded_files", [])),
            "excluded_dirs": inventory.get("excluded_dirs", []),
            "excluded_files": inventory.get("excluded_files", []),
            "coverage_limitations": inventory.get("coverage_limitations", []),
            "graphify_verified": bool(graph_status.get("verified")),
            "graphify_fresh": bool(graph_status.get("fresh")),
            "graphify_indexed_files": len(graph_status.get("indexed_files", [])),
            "graphify_deleted_files": graph_status.get("deleted_files", []),
            "graphify_missing_files": graph_status.get("missing_files", []),
            "memory_project": resolved_proj,
            "memory_verified": bool(memory_status.get("verified")),
            "memory_indexed": bool(memory_status.get("indexed")),
            "memory_fresh": bool(memory_status.get("fresh")),
            "memory_indexed_files": len(memory_status.get("indexed_files", [])),
            "memory_deleted_files": memory_status.get("deleted_files", []),
            "memory_missing_files": memory_status.get("missing_files", []),
            "memory_stale_files": memory_status.get("stale_files", []),
            "complete_inventory_available": True,
            "all_sources_verified": all_sources_verified,
        }

        if strict and not all_sources_verified:
            raise RuntimeError(
                "Strict mode: critical indexing or retrieval verification failed: "
                + ("; ".join(errors) if errors else "unverified knowledge sources")
            )

        return {
            "repo_root": str(self.repo_root),
            "question": question,
            "project": resolved_proj,
            "index_status": {
                "graphify": graph_status,
                "memory": memory_status,
                "memory_project": resolved_proj,
                "memory_indexed": bool(memory_status.get("indexed")),
                "memory_fresh": bool(memory_status.get("fresh")),
                "all_verified": all_sources_verified,
            },
            "coverage": coverage,
            "repository_inventory": inventory,
            "reading_order": inventory["reading_order"],
            "formatted_inventory": formatted_inventory,
            "graphify_context": graph_ctx,
            "graphify_report": graph_report,
            "memory_context": memory_ctx,
            "memory_symbols": memory_ctx,
            "architecture_overview": arch_ctx,
            "memory_architecture": arch_ctx,
            "errors": errors,
        }
