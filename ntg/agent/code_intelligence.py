"""Unified Codebase Intelligence wrapper for Graphify, Codebase Memory MCP, and deterministic AST repository analysis."""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any


class CodeIntelligence:
    """Provides structural codebase awareness by combining:
    1. Deterministic repository AST/file inventory (modules, docstrings, classes, functions, imports, dependency order)
    2. Graphify CLI (`graphify query`, `graphify path`, `graphify explain`, `graphify update`)
    3. Codebase Memory MCP CLI (`codebase-memory-mcp cli <tool> <json_payload>`)

    All index validations check repository-root identity, indexed-file coverage,
    deleted/missing files, and source content fingerprints (SHA-256 + AST symbol signatures).
    """

    _IGNORED_DIRS: frozenset[str] = frozenset(
        {
            ".git",
            ".hg",
            ".svn",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            ".tox",
            ".nox",
            ".venv",
            "venv",
            "env",
            "node_modules",
            "dist",
            "build",
            "graphify-out",
            ".ntg",
            ".idea",
            ".vscode",
        }
    )

    _CONFIG_DOC_FILENAMES: frozenset[str] = frozenset(
        {
            "README.md",
            "pyproject.toml",
            "requirements.txt",
            "setup.py",
            "setup.cfg",
            ".env.example",
            ".gitignore",
            "Makefile",
            "Dockerfile",
            "docker-compose.yml",
        }
    )

    # Files that Codebase Memory MCP purposefully skips during repository indexing
    _MEMORY_UNINDEXED_FILES: frozenset[str] = frozenset({".gitignore"})

    _SENSITIVE_EXCLUDED_FILES: frozenset[str] = frozenset(
        {
            ".env",
            ".env.local",
            ".env.production",
            "credentials.json",
        }
    )

    _WIN_FALLBACK_PATHS: dict[str, list[Path]] = {
        "codebase-memory-mcp": [
            Path.home()
            / "AppData"
            / "Roaming"
            / "npm"
            / "node_modules"
            / "codebase-memory-mcp"
            / "bin"
            / "codebase-memory-mcp.exe",
            Path.home() / "AppData" / "Local" / "codebase-memory-mcp" / "codebase-memory-mcp.exe",
            Path.home() / "AppData" / "Roaming" / "npm" / "codebase-memory-mcp.cmd",
        ],
        "graphify": [
            Path.home()
            / "AppData"
            / "Roaming"
            / "Python"
            / "Python311"
            / "Scripts"
            / "graphify.exe",
            Path.home()
            / "AppData"
            / "Local"
            / "Programs"
            / "Python"
            / "Python311"
            / "Scripts"
            / "graphify.exe",
        ],
    }

    def __init__(
        self,
        repo_root: str | Path = ".",
        timeout: int = 60,
        default_project: str | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.timeout = timeout
        self.default_project = (
            default_project.strip()
            if isinstance(default_project, str) and default_project.strip()
            else None
        )

    @classmethod
    def _resolve_executable(cls, binary: str) -> str:
        """Resolve CLI binary path reliably across Windows and POSIX, preferring native `.exe` over `.cmd` shims."""
        if os.name == "nt" and binary in cls._WIN_FALLBACK_PATHS:
            for candidate in cls._WIN_FALLBACK_PATHS[binary]:
                if candidate.is_file() and candidate.suffix.lower() == ".exe":
                    return str(candidate)

        found = shutil.which(binary)
        if found:
            found_path = Path(found)
            if os.name == "nt" and found_path.suffix.lower() in (".cmd", ".bat", ".ps1"):
                for candidate in cls._WIN_FALLBACK_PATHS.get(binary, []):
                    if candidate.is_file():
                        return str(candidate)
            return str(found_path)

        for candidate in cls._WIN_FALLBACK_PATHS.get(binary, []):
            if candidate.is_file():
                return str(candidate)

        raise FileNotFoundError(
            f"Required CLI tool '{binary}' was not found on PATH or standard install locations."
        )

    def _run(
        self,
        args: list[str],
        timeout: int | None = None,
        check: bool = True,
    ) -> str:
        """Execute a CLI tool safely with `stdin=subprocess.DEVNULL` and explicit timeout."""
        if not args:
            raise ValueError("Command arguments cannot be empty.")

        resolved_bin = self._resolve_executable(args[0])
        full_args = [resolved_bin, *args[1:]]
        effective_timeout = timeout if timeout is not None else self.timeout

        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        try:
            proc = subprocess.run(
                full_args,
                cwd=str(self.repo_root),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=effective_timeout,
                check=False,
                creationflags=creationflags,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Command timed out after {effective_timeout}s: {' '.join(args)}"
            ) from exc
        except OSError as exc:
            raise RuntimeError(
                f"Failed to execute command {' '.join(args)}: {exc}"
            ) from exc

        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()

        if check and proc.returncode != 0:
            detail = stderr or stdout or f"exit code {proc.returncode}"
            raise RuntimeError(f"Command failed ({' '.join(args)}): {detail}")

        return stdout if stdout else stderr

    # ------------------------------------------------------------------
    # Repository File Discovery & Source Fingerprinting
    # ------------------------------------------------------------------

    def _discover_repo_files_with_exclusions(
        self,
    ) -> tuple[list[Path], list[Path], list[dict[str, str]], list[dict[str, str]]]:
        """Discover all relevant Python and configuration/documentation files in `self.repo_root`,
        along with explicit lists of excluded directories and excluded files.
        """
        py_files: list[Path] = []
        other_files: list[Path] = []
        excluded_dirs: list[dict[str, str]] = []
        excluded_files: list[dict[str, str]] = []
        seen_pycache: bool = False

        for root, dirs, files in os.walk(self.repo_root):
            root_path = Path(root)
            kept_dirs: list[str] = []
            for d in sorted(dirs):
                if d in self._IGNORED_DIRS or d.startswith(".") or d.endswith(".egg-info"):
                    rel_dir = (root_path / d).relative_to(self.repo_root).as_posix()
                    if d == "__pycache__":
                        if not seen_pycache:
                            excluded_dirs.append(
                                {
                                    "path": "**/__pycache__",
                                    "reason": "bytecode_cache_directory",
                                }
                            )
                            seen_pycache = True
                    else:
                        reason = (
                            "index_output_directory"
                            if d in ("graphify-out", ".ntg")
                            else "ignored_or_hidden_directory"
                        )
                        excluded_dirs.append({"path": rel_dir, "reason": reason})
                else:
                    kept_dirs.append(d)
            dirs[:] = kept_dirs

            for fname in sorted(files):
                fpath = root_path / fname
                rel_file = fpath.relative_to(self.repo_root).as_posix()
                if fname in self._SENSITIVE_EXCLUDED_FILES:
                    excluded_files.append(
                        {"path": rel_file, "reason": "sensitive_environment_or_secret_file"}
                    )
                    continue
                if fname.endswith(".py"):
                    py_files.append(fpath)
                elif (
                    fname in self._CONFIG_DOC_FILENAMES
                    or fname.endswith((".md", ".toml", ".yaml", ".yml", ".ini", ".cfg"))
                ):
                    other_files.append(fpath)
                else:
                    excluded_files.append(
                        {"path": rel_file, "reason": "non_source_or_untracked_artifact"}
                    )

        py_files.sort(key=lambda p: p.relative_to(self.repo_root).as_posix())
        other_files.sort(key=lambda p: p.relative_to(self.repo_root).as_posix())
        return py_files, other_files, excluded_dirs, excluded_files

    def _discover_repo_files(self) -> tuple[list[Path], list[Path]]:
        """Discover all relevant Python and configuration/documentation files in `self.repo_root`."""
        py_files, other_files, _, _ = self._discover_repo_files_with_exclusions()
        return py_files, other_files

    @staticmethod
    def _compute_file_sha256(path: Path) -> str:
        """Compute canonical SHA-256 hex digest of a file's normalized UTF-8 content (or raw bytes)."""
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            normalized = text.replace("\r\n", "\n")
            return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        except OSError:
            return ""

    @staticmethod
    def _extract_file_ast_symbols(py_path: Path) -> list[str]:
        """Extract sorted top-level AST class and function names from a Python file."""
        try:
            source = py_path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=str(py_path))
        except (OSError, SyntaxError):
            return []

        symbols: set[str] = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                symbols.add(node.name)
        return sorted(symbols)

    def _compute_repo_fingerprints(self) -> dict[str, Any]:
        """Compute SHA-256 content hashes and AST symbol sets for all discovered repository files."""
        py_files, other_files = self._discover_repo_files()
        py_hashes: dict[str, str] = {}
        all_hashes: dict[str, str] = {}
        ast_symbols: dict[str, list[str]] = {}

        for py_path in py_files:
            rel = py_path.relative_to(self.repo_root).as_posix()
            digest = self._compute_file_sha256(py_path)
            py_hashes[rel] = digest
            all_hashes[rel] = digest
            ast_symbols[rel] = self._extract_file_ast_symbols(py_path)

        for other_path in other_files:
            rel = other_path.relative_to(self.repo_root).as_posix()
            all_hashes[rel] = self._compute_file_sha256(other_path)

        return {
            "repo_root": str(self.repo_root),
            "py_hashes": py_hashes,
            "all_hashes": all_hashes,
            "ast_symbols": ast_symbols,
        }

    @property
    def index_manifest_path(self) -> Path:
        return self.repo_root / ".ntg" / "index_manifest.json"

    def _load_index_manifest(self) -> dict[str, Any]:
        """Load `.ntg/index_manifest.json` if present and valid."""
        path = self.index_manifest_path
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError, TypeError):
            return {}

    def _save_index_manifest(self, section: str, payload: dict[str, Any]) -> None:
        """Persist updated fingerprint metadata for `graphify` or `memory` in `.ntg/index_manifest.json`."""
        manifest = self._load_index_manifest()
        manifest["repo_root"] = str(self.repo_root)
        manifest[section] = payload
        try:
            self.index_manifest_path.parent.mkdir(parents=True, exist_ok=True)
            self.index_manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except OSError:
            pass

    def _record_graphify_manifest(self) -> None:
        """Record current SHA-256 fingerprints and AST symbol signatures for Graphify."""
        if not self.has_graphify_graph():
            return
        fp = self._compute_repo_fingerprints()
        self._save_index_manifest(
            "graphify",
            {
                "repo_root": str(self.repo_root),
                "graph_sha256": self._compute_file_sha256(self.graph_path),
                "file_hashes": fp["py_hashes"],
                "ast_symbols": fp["ast_symbols"],
            },
        )

    def _record_memory_manifest(self, project: str) -> None:
        """Record current SHA-256 fingerprints and AST symbol signatures for Codebase Memory MCP."""
        fp = self._compute_repo_fingerprints()
        mem_hashes = {
            k: v
            for k, v in fp["all_hashes"].items()
            if k not in self._MEMORY_UNINDEXED_FILES
        }
        self._save_index_manifest(
            "memory",
            {
                "repo_root": str(self.repo_root),
                "project": project,
                "file_hashes": mem_hashes,
                "ast_symbols": fp["ast_symbols"],
            },
        )

    # ------------------------------------------------------------------
    # Graphify Knowledge Graph Operations & Freshness Validation
    # ------------------------------------------------------------------

    @property
    def graph_path(self) -> Path:
        return self.repo_root / "graphify-out" / "graph.json"

    @property
    def graph_report_path(self) -> Path:
        return self.repo_root / "graphify-out" / "GRAPH_REPORT.md"

    def has_graphify_graph(self) -> bool:
        return self.graph_path.is_file()

    def inspect_graphify_status(self) -> dict[str, Any]:
        """Validate `graphify-out/graph.json` against `self.repo_root` and current source files.
        Checks:
        - Repository root identity (`.graphify_root` marker and node `source_file` paths)
        - Deleted or non-existent files referenced in the graph
        - Unindexed Python source files (`missing_files`)
        - Stale content via SHA-256 source fingerprints (`.ntg/index_manifest.json`) AND
          live AST symbol-set cross-checking against `graph.json` nodes (`ast_symbols_verified`).
        """
        status: dict[str, Any] = {
            "repo_root": str(self.repo_root),
            "graph_path": str(self.graph_path),
            "exists": self.has_graphify_graph(),
            "valid": False,
            "fresh": False,
            "contaminated": False,
            "mismatched_root": False,
            "fingerprint_verified": False,
            "ast_symbols_verified": False,
            "node_count": 0,
            "edge_count": 0,
            "indexed_files": [],
            "deleted_files": [],
            "missing_files": [],
            "modified_files": [],
            "stale_files": [],
            "symbol_mismatches": [],
            "needs_refresh": True,
            "reasons": [],
        }

        if not self.has_graphify_graph():
            status["reasons"].append("graphify-out/graph.json does not exist")
            return status

        root_marker = self.repo_root / "graphify-out" / ".graphify_root"
        if root_marker.is_file():
            try:
                recorded_root = Path(
                    root_marker.read_text(encoding="utf-8").strip()
                ).resolve()
                if recorded_root != self.repo_root:
                    status["mismatched_root"] = True
                    status["contaminated"] = True
                    status["reasons"].append(
                        f"Graphify root marker '{recorded_root}' does not match '{self.repo_root}'"
                    )
            except OSError:
                pass

        try:
            raw = self.graph_path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, ValueError) as exc:
            status["reasons"].append(f"Corrupted graph.json: {exc}")
            return status

        if not isinstance(data, dict):
            status["reasons"].append("graph.json root is not a JSON object")
            return status

        nodes = data.get("nodes")
        links = data.get("links") or data.get("edges") or []
        if not isinstance(nodes, list) or not nodes:
            status["reasons"].append("graph.json contains 0 nodes")
            return status

        status["node_count"] = len(nodes)
        status["edge_count"] = len(links) if isinstance(links, list) else 0

        indexed_files: set[str] = set()
        deleted_files: set[str] = set()
        graph_symbols_by_file: dict[str, set[str]] = {}
        graph_class_fn_nodes_by_file: dict[str, set[str]] = {}

        for node in nodes:
            if not isinstance(node, dict):
                continue
            src = node.get("source_file")
            if not isinstance(src, str) or not src.strip():
                continue
            norm_src = src.strip().replace("\\", "/")
            while norm_src.startswith("./"):
                norm_src = norm_src[2:]

            src_path = Path(norm_src)
            if src_path.is_absolute():
                try:
                    rel_src = src_path.resolve().relative_to(self.repo_root).as_posix()
                    norm_src = rel_src
                except ValueError:
                    status["mismatched_root"] = True
                    status["contaminated"] = True
                    deleted_files.add(norm_src)
                    continue

            indexed_files.add(norm_src)
            label = str(node.get("label") or "").strip()
            ntype = str(node.get("type") or node.get("kind") or "").strip().lower()
            if label:
                graph_symbols_by_file.setdefault(norm_src, set()).add(label)
                if "." in label:
                    graph_symbols_by_file[norm_src].add(label.split(".")[0])
                # Track top-level class/function nodes specifically for deleted-symbol detection
                if ntype in ("class", "function") and "." not in label:
                    graph_class_fn_nodes_by_file.setdefault(norm_src, set()).add(label)

            candidate_path = (self.repo_root / norm_src).resolve()
            try:
                candidate_path.relative_to(self.repo_root)
            except ValueError:
                status["mismatched_root"] = True
                status["contaminated"] = True
                deleted_files.add(norm_src)
                continue

            if not candidate_path.is_file():
                deleted_files.add(norm_src)
            else:
                rel_parts = set(candidate_path.relative_to(self.repo_root).parts)
                if rel_parts & self._IGNORED_DIRS:
                    deleted_files.add(norm_src)

        fp = self._compute_repo_fingerprints()
        current_py_hashes: dict[str, str] = fp["py_hashes"]
        current_ast_symbols: dict[str, list[str]] = fp["ast_symbols"]

        missing_files: list[str] = []
        stale_files: set[str] = set()
        symbol_mismatches: list[str] = []

        # 1. Coverage + Live AST Symbol Set Verification against graph.json nodes
        for rel_py, expected_syms in current_ast_symbols.items():
            if rel_py not in indexed_files:
                missing_files.append(rel_py)
                continue

            indexed_syms = graph_symbols_by_file.get(rel_py, set())
            missing_syms = [s for s in expected_syms if s not in indexed_syms]
            if missing_syms:
                stale_files.add(rel_py)
                symbol_mismatches.append(
                    f"{rel_py} (missing AST symbols in graph.json: {', '.join(missing_syms[:5])})"
                )

            top_graph_syms = graph_class_fn_nodes_by_file.get(rel_py, set())
            extra_syms = [s for s in sorted(top_graph_syms) if s not in set(expected_syms)]
            if extra_syms:
                stale_files.add(rel_py)
                symbol_mismatches.append(
                    f"{rel_py} (stale removed symbols in graph.json: {', '.join(extra_syms[:5])})"
                )

        ast_symbols_verified = len(missing_files) == 0 and len(symbol_mismatches) == 0
        status["ast_symbols_verified"] = ast_symbols_verified

        # 2. SHA-256 Content Fingerprint Verification against .ntg/index_manifest.json
        manifest = self._load_index_manifest()
        graph_manifest = (
            manifest.get("graphify") if isinstance(manifest.get("graphify"), dict) else {}
        )
        manifest_hashes = (
            graph_manifest.get("file_hashes")
            if isinstance(graph_manifest.get("file_hashes"), dict)
            else {}
        )
        manifest_graph_sha = str(graph_manifest.get("graph_sha256") or "")
        manifest_root = str(graph_manifest.get("repo_root") or "")
        current_graph_sha = self._compute_file_sha256(self.graph_path)

        fingerprint_verified = True
        if not graph_manifest or manifest_root != str(self.repo_root):
            fingerprint_verified = False
            status["reasons"].append(
                "Graphify SHA-256 index manifest is missing or has a mismatched repo_root"
            )
        elif manifest_graph_sha != current_graph_sha:
            fingerprint_verified = False
            status["reasons"].append(
                "graphify-out/graph.json SHA-256 does not match recorded index manifest"
            )
        else:
            for rel_py, cur_hash in current_py_hashes.items():
                recorded_hash = manifest_hashes.get(rel_py)
                if recorded_hash != cur_hash:
                    fingerprint_verified = False
                    stale_files.add(rel_py)

            for rec_py in manifest_hashes:
                if rec_py not in current_py_hashes:
                    fingerprint_verified = False
                    deleted_files.add(rec_py)

        status["fingerprint_verified"] = fingerprint_verified
        status["valid"] = True
        status["indexed_files"] = sorted(indexed_files)
        status["deleted_files"] = sorted(deleted_files)
        status["missing_files"] = sorted(missing_files)
        status["modified_files"] = sorted(stale_files)
        status["stale_files"] = sorted(stale_files)
        status["symbol_mismatches"] = symbol_mismatches

        if deleted_files:
            status["contaminated"] = True
            status["reasons"].append(
                f"Graph references {len(deleted_files)} deleted/non-existent file(s): {', '.join(sorted(deleted_files)[:8])}"
            )
        if missing_files:
            status["reasons"].append(
                f"Graph is missing {len(missing_files)} current Python file(s): {', '.join(sorted(missing_files)[:8])}"
            )
        if symbol_mismatches:
            status["reasons"].append(
                f"Graph AST symbol mismatch in {len(symbol_mismatches)} file(s): {'; '.join(symbol_mismatches[:5])}"
            )
        if stale_files and not symbol_mismatches:
            status["reasons"].append(
                f"Source SHA-256 fingerprint changed since last Graphify build for {len(stale_files)} file(s): {', '.join(sorted(stale_files)[:8])}"
            )

        needs_refresh = bool(
            status["contaminated"]
            or status["mismatched_root"]
            or deleted_files
            or missing_files
            or stale_files
            or not ast_symbols_verified
            or not fingerprint_verified
        )
        status["needs_refresh"] = needs_refresh
        status["fresh"] = bool(
            status["valid"]
            and not needs_refresh
            and ast_symbols_verified
            and fingerprint_verified
        )
        return status

    def _sync_graphify_python_ast(self) -> dict[str, Any]:
        """Synchronize `graphify-out/graph.json` directly against the current repository's
        Python ASTs:
        1. Purges all nodes and edges referencing deleted, ignored, or outside-repo files.
        2. Rebuilds accurate module, class, function, method, and import/contains edges for
           any missing or modified Python files so `graphify-out/graph.json` is 100% complete
           and free of stale references.
        3. Writes `graphify-out/.graphify_root` and records SHA-256 fingerprints in `.ntg/index_manifest.json`.
        """
        py_files, _ = self._discover_repo_files()
        valid_rel_paths = {
            p.relative_to(self.repo_root).as_posix(): p for p in py_files
        }

        data: dict[str, Any] = {
            "directed": False,
            "multigraph": False,
            "graph": {},
            "nodes": [],
            "links": [],
        }
        if self.has_graphify_graph():
            try:
                loaded = json.loads(self.graph_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    data = loaded
            except (OSError, ValueError):
                pass

        raw_nodes = data.get("nodes") if isinstance(data.get("nodes"), list) else []
        links_key = "links" if "links" in data else ("edges" if "edges" in data else "links")
        raw_links = data.get(links_key) if isinstance(data.get(links_key), list) else []

        # Identify which files need their AST nodes rebuilt in graph.json
        fp = self._compute_repo_fingerprints()
        manifest = self._load_index_manifest()
        graph_manifest = (
            manifest.get("graphify") if isinstance(manifest.get("graphify"), dict) else {}
        )
        manifest_hashes = (
            graph_manifest.get("file_hashes")
            if isinstance(graph_manifest.get("file_hashes"), dict)
            else {}
        )

        existing_files_in_graph: set[str] = set()
        graph_syms_by_file: dict[str, set[str]] = {}
        graph_top_syms_by_file: dict[str, set[str]] = {}

        for node in raw_nodes:
            if not isinstance(node, dict):
                continue
            src = str(node.get("source_file") or "").strip().replace("\\", "/")
            while src.startswith("./"):
                src = src[2:]
            if src in valid_rel_paths:
                existing_files_in_graph.add(src)
                label = str(node.get("label") or "").strip()
                ntype = str(node.get("type") or node.get("kind") or "").strip().lower()
                if label:
                    graph_syms_by_file.setdefault(src, set()).add(label)
                    if "." in label:
                        graph_syms_by_file[src].add(label.split(".")[0])
                    if ntype in ("class", "function") and "." not in label:
                        graph_top_syms_by_file.setdefault(src, set()).add(label)

        files_to_rebuild: set[str] = set()
        for rel_py, expected_syms in fp["ast_symbols"].items():
            if rel_py not in existing_files_in_graph:
                files_to_rebuild.add(rel_py)
            elif manifest_hashes.get(rel_py) != fp["py_hashes"].get(rel_py):
                files_to_rebuild.add(rel_py)
            else:
                g_syms = graph_syms_by_file.get(rel_py, set())
                g_top = graph_top_syms_by_file.get(rel_py, set())
                if any(s not in g_syms for s in expected_syms) or any(
                    s not in set(expected_syms) for s in g_top
                ):
                    files_to_rebuild.add(rel_py)

        kept_nodes: list[dict[str, Any]] = []
        kept_node_ids: set[str] = set()
        removed_node_ids: set[str] = set()

        for node in raw_nodes:
            if not isinstance(node, dict):
                continue
            nid = str(node.get("id") or "")
            src = str(node.get("source_file") or "").strip().replace("\\", "/")
            while src.startswith("./"):
                src = src[2:]

            if not src or src not in valid_rel_paths or src in files_to_rebuild:
                if nid:
                    removed_node_ids.add(nid)
                continue

            node["source_file"] = src
            kept_nodes.append(node)
            if nid:
                kept_node_ids.add(nid)

        kept_links: list[dict[str, Any]] = []
        seen_edges: set[tuple[str, str, str]] = set()
        for link in raw_links:
            if not isinstance(link, dict):
                continue
            s = str(link.get("source") or "")
            t = str(link.get("target") or "")
            rel = str(link.get("relation") or "related")
            if (
                s in kept_node_ids
                and t in kept_node_ids
                and s not in removed_node_ids
                and t not in removed_node_ids
            ):
                edge_key = (s, t, rel)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    kept_links.append(link)

        mod_to_rel = self._module_map_for_repo(py_files)

        def _make_node_id(rel_path: str, symbol: str | None = None) -> str:
            prefix = re.sub(r"[^a-zA-Z0-9]+", "_", rel_path[:-3] if rel_path.endswith(".py") else rel_path).strip("_").lower()
            if not symbol:
                return f"{prefix}_module"
            sym_clean = re.sub(r"[^a-zA-Z0-9]+", "_", symbol).strip("_").lower()
            return f"{prefix}_{sym_clean}"

        for rel_py in sorted(files_to_rebuild):
            py_path = valid_rel_paths[rel_py]
            info = self._analyze_python_file(py_path, mod_to_rel)
            mod_id = _make_node_id(rel_py)
            mod_label = Path(rel_py).stem if Path(rel_py).stem != "__init__" else Path(rel_py).parent.name or "root"
            if mod_id not in kept_node_ids:
                kept_nodes.append(
                    {
                        "id": mod_id,
                        "label": mod_label,
                        "type": "module",
                        "file_type": "code",
                        "source_file": rel_py,
                        "source_location": "L1",
                        "community": 0,
                    }
                )
                kept_node_ids.add(mod_id)

            for cls in info.get("classes", []):
                cls_name = cls["name"]
                cls_id = _make_node_id(rel_py, cls_name)
                if cls_id not in kept_node_ids:
                    kept_nodes.append(
                        {
                            "id": cls_id,
                            "label": cls_name,
                            "type": "class",
                            "file_type": "code",
                            "source_file": rel_py,
                            "source_location": f"L{cls.get('line', 1)}",
                            "community": 0,
                        }
                    )
                    kept_node_ids.add(cls_id)
                edge_key = (mod_id, cls_id, "contains")
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    kept_links.append(
                        {
                            "source": mod_id,
                            "target": cls_id,
                            "relation": "contains",
                            "confidence": "EXTRACTED",
                            "confidence_score": 1.0,
                            "source_file": rel_py,
                            "weight": 1.0,
                        }
                    )
                for mname in cls.get("methods", []):
                    m_id = _make_node_id(rel_py, f"{cls_name}.{mname}")
                    if m_id not in kept_node_ids:
                        kept_nodes.append(
                            {
                                "id": m_id,
                                "label": f"{cls_name}.{mname}",
                                "type": "method",
                                "file_type": "code",
                                "source_file": rel_py,
                                "source_location": f"L{cls.get('line', 1)}",
                                "community": 0,
                            }
                        )
                        kept_node_ids.add(m_id)
                    m_edge = (cls_id, m_id, "method")
                    if m_edge not in seen_edges:
                        seen_edges.add(m_edge)
                        kept_links.append(
                            {
                                "source": cls_id,
                                "target": m_id,
                                "relation": "method",
                                "confidence": "EXTRACTED",
                                "confidence_score": 1.0,
                                "source_file": rel_py,
                                "weight": 1.0,
                            }
                        )

            for fn in info.get("functions", []):
                fn_name = fn["name"]
                fn_id = _make_node_id(rel_py, fn_name)
                if fn_id not in kept_node_ids:
                    kept_nodes.append(
                        {
                            "id": fn_id,
                            "label": fn_name,
                            "type": "function",
                            "file_type": "code",
                            "source_file": rel_py,
                            "source_location": f"L{fn.get('line', 1)}",
                            "community": 0,
                        }
                    )
                    kept_node_ids.add(fn_id)
                f_edge = (mod_id, fn_id, "contains")
                if f_edge not in seen_edges:
                    seen_edges.add(f_edge)
                    kept_links.append(
                        {
                            "source": mod_id,
                            "target": fn_id,
                            "relation": "contains",
                            "confidence": "EXTRACTED",
                            "confidence_score": 1.0,
                            "source_file": rel_py,
                            "weight": 1.0,
                        }
                    )

        first_node_by_file: dict[str, str] = {}
        for node in kept_nodes:
            src = node.get("source_file")
            nid = node.get("id")
            if isinstance(src, str) and isinstance(nid, str) and src not in first_node_by_file:
                first_node_by_file[src] = nid

        for rel_py in sorted(files_to_rebuild):
            py_path = valid_rel_paths[rel_py]
            info = self._analyze_python_file(py_path, mod_to_rel)
            src_id = first_node_by_file.get(rel_py)
            if not src_id:
                continue
            for dep_rel in info.get("internal_dependencies", []):
                tgt_id = first_node_by_file.get(dep_rel)
                if tgt_id and tgt_id != src_id:
                    e_key = (src_id, tgt_id, "imports")
                    if e_key not in seen_edges:
                        seen_edges.add(e_key)
                        kept_links.append(
                            {
                                "source": src_id,
                                "target": tgt_id,
                                "relation": "imports",
                                "confidence": "EXTRACTED",
                                "confidence_score": 1.0,
                                "source_file": rel_py,
                                "weight": 1.0,
                            }
                        )

        data["nodes"] = kept_nodes
        data[links_key] = kept_links
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)
        self.graph_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        root_marker = self.repo_root / "graphify-out" / ".graphify_root"
        root_marker.write_text(str(self.repo_root), encoding="utf-8")
        self._record_graphify_manifest()

        return {
            "node_count": len(kept_nodes),
            "edge_count": len(kept_links),
            "rebuilt_files": sorted(files_to_rebuild),
        }

    def ensure_graphify_graph(self, force_rebuild: bool = False) -> dict[str, Any]:
        """Validate `graphify-out/graph.json`, refreshing only when missing, stale, or contaminated.
        Revalidates after refresh and raises `RuntimeError` if any stale, incomplete, or mismatched state remains.
        """
        status = self.inspect_graphify_status()
        if force_rebuild or status["needs_refresh"]:
            self.graphify_update(".")
            post_status = self.inspect_graphify_status()
            if (
                not post_status["valid"]
                or not post_status["fresh"]
                or post_status["deleted_files"]
                or post_status["missing_files"]
                or post_status["stale_files"]
                or not post_status["fingerprint_verified"]
                or not post_status["ast_symbols_verified"]
            ):
                raise RuntimeError(
                    f"Graphify index failed post-refresh verification: {'; '.join(post_status['reasons'])}"
                )
            return post_status
        return status

    def graphify_update(self, target_path: str | Path = ".") -> str:
        """Update or rebuild the Graphify knowledge graph for `target_path`,
        and synchronize AST nodes and SHA-256 fingerprints so deleted, newly added,
        or modified files are accurately reflected.
        """
        status_before = self.inspect_graphify_status()
        if status_before["contaminated"] or status_before["deleted_files"]:
            cache_dir = self.repo_root / "graphify-out" / "cache"
            if cache_dir.is_dir():
                for cache_file in cache_dir.glob("*.json"):
                    try:
                        cache_file.unlink()
                    except OSError:
                        pass
            if self.graph_path.is_file():
                try:
                    self.graph_path.unlink()
                except OSError:
                    pass

        cli_output = ""
        try:
            cli_output = self._run(
                ["graphify", "update", str(target_path)],
                timeout=120,
                check=False,
            )
        except Exception as exc:
            cli_output = f"graphify CLI notice: {exc}"

        sync_stats = self._sync_graphify_python_ast()
        return (
            f"{cli_output}\nAST sync: {sync_stats['node_count']} nodes, "
            f"{sync_stats['edge_count']} edges, rebuilt={len(sync_stats['rebuilt_files'])}".strip()
        )

    def graphify_query(
        self,
        question: str,
        budget: int = 2000,
        use_dfs: bool = False,
    ) -> str:
        """Run a BFS (or DFS) structural traversal query against `graphify-out/graph.json`."""
        if not question or not question.strip():
            raise ValueError("Graphify query question cannot be empty.")

        cmd = ["graphify", "query", question.strip(), "--budget", str(budget)]
        if use_dfs:
            cmd.append("--dfs")
        return self._run(cmd)

    def graphify_path(self, node_a: str, node_b: str) -> str:
        """Find the shortest structural path between two nodes in `graphify-out/graph.json`."""
        return self._run(["graphify", "path", node_a, node_b])

    def graphify_explain(self, node: str) -> str:
        """Explain a specific node and its immediate neighbors in `graphify-out/graph.json`."""
        return self._run(["graphify", "explain", node])

    def graphify_save_result(
        self,
        question: str,
        answer: str,
        query_type: str = "query",
        source_nodes: list[str] | None = None,
    ) -> str:
        """Save a Q&A or planning result back to `graphify-out/memory/` for graph growth."""
        cmd = [
            "graphify",
            "save-result",
            "--question",
            question,
            "--answer",
            answer,
            "--type",
            query_type,
        ]
        if source_nodes:
            cmd.extend(["--nodes", ",".join(source_nodes)])
        return self._run(cmd)

    # ------------------------------------------------------------------
    # Codebase Memory MCP Operations & Project Resolution
    # ------------------------------------------------------------------

    def list_memory_projects(self) -> list[dict[str, Any]]:
        """Return the list of indexed projects from Codebase Memory MCP."""
        raw = self._run(
            [
                "codebase-memory-mcp",
                "cli",
                "--quiet",
                "list_projects",
                json.dumps({"format": "json"}),
            ],
            timeout=30,
        )
        try:
            data = json.loads(raw)
        except ValueError as exc:
            raise RuntimeError(
                f"Codebase Memory MCP returned invalid JSON for list_projects: {raw[:200]}"
            ) from exc

        if isinstance(data, dict) and isinstance(data.get("projects"), list):
            return [p for p in data["projects"] if isinstance(p, dict)]
        if isinstance(data, list):
            return [p for p in data if isinstance(p, dict)]
        return []

    def resolve_memory_project(
        self,
        preferred_project: str | None = None,
        auto_index: bool = False,
    ) -> str:
        """Resolve the exact Codebase Memory MCP project name corresponding to `self.repo_root`.
        Never silently selects an unrelated project.
        """
        projects = self.list_memory_projects()

        # 1. Exact canonical `root_path` match against `self.repo_root`
        for proj in projects:
            root_str = proj.get("root_path")
            name_str = proj.get("name")
            if isinstance(root_str, str) and isinstance(name_str, str) and name_str.strip():
                try:
                    if Path(root_str).expanduser().resolve() == self.repo_root:
                        return name_str.strip()
                except OSError:
                    continue

        # 2. If `preferred_project` was explicitly passed, verify its root_path matches `self.repo_root`
        candidate = (preferred_project or self.default_project or "").strip()
        if candidate:
            for proj in projects:
                if proj.get("name") == candidate:
                    root_str = proj.get("root_path")
                    if isinstance(root_str, str):
                        try:
                            if Path(root_str).expanduser().resolve() == self.repo_root:
                                return candidate
                        except OSError:
                            pass
                    raise RuntimeError(
                        f"Project '{candidate}' points to '{root_str}', which does not match '{self.repo_root}'."
                    )

        # 3. Auto-index `self.repo_root` if requested
        if auto_index:
            self.memory_index()
            projects_after = self.list_memory_projects()
            for proj in projects_after:
                root_str = proj.get("root_path")
                name_str = proj.get("name")
                if isinstance(root_str, str) and isinstance(name_str, str) and name_str.strip():
                    try:
                        if Path(root_str).expanduser().resolve() == self.repo_root:
                            return name_str.strip()
                    except OSError:
                        continue

        available = [
            f"{p.get('name')} ({p.get('root_path')})" for p in projects
        ]
        raise RuntimeError(
            f"No Codebase Memory MCP project is indexed for repository root '{self.repo_root}'. "
            f"Available indexed projects: {available}"
        )

    def inspect_memory_status(
        self,
        project: str | None = None,
    ) -> dict[str, Any]:
        """Validate Codebase Memory MCP index for `self.repo_root`.
        Checks:
        - Repository-root identity (`root_matched`)
        - Indexed-file coverage across Python and config/doc files (`missing_files`)
        - Deleted/non-existent files referenced in the index (`deleted_files`)
        - Stale content via SHA-256 source fingerprints + AST symbol signatures (`.ntg/index_manifest.json`)
          AND indexed module line counts (`stale_files`).
        """
        status: dict[str, Any] = {
            "repo_root": str(self.repo_root),
            "project": None,
            "exists": False,
            "indexed": False,
            "valid": False,
            "fresh": False,
            "root_matched": False,
            "contaminated": False,
            "mismatched_root": False,
            "fingerprint_verified": False,
            "indexed_files": [],
            "deleted_files": [],
            "missing_files": [],
            "stale_files": [],
            "modified_files": [],
            "needs_refresh": True,
            "reasons": [],
        }

        try:
            projects = self.list_memory_projects()
        except Exception as exc:
            status["reasons"].append(f"list_projects failed: {exc}")
            return status

        resolved_proj: str | None = None
        for proj in projects:
            root_str = proj.get("root_path")
            name_str = proj.get("name")
            if isinstance(root_str, str) and isinstance(name_str, str) and name_str.strip():
                try:
                    if Path(root_str).expanduser().resolve() == self.repo_root:
                        resolved_proj = name_str.strip()
                        status["root_matched"] = True
                        break
                except OSError:
                    continue

        if not resolved_proj and project:
            for proj in projects:
                if proj.get("name") == project.strip():
                    status["mismatched_root"] = True
                    status["contaminated"] = True
                    status["reasons"].append(
                        f"Project '{project}' points to '{proj.get('root_path')}', not '{self.repo_root}'"
                    )
                    return status

        if not resolved_proj:
            status["reasons"].append(
                f"No Codebase Memory MCP project is indexed for '{self.repo_root}'"
            )
            return status

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
            status["reasons"].append(
                f"Failed to query File/Module nodes for '{resolved_proj}': {exc}"
            )
            return status

        if not isinstance(mod_data, dict) or not isinstance(mod_data.get("groups"), list):
            status["reasons"].append(
                f"Unexpected Module query payload for '{resolved_proj}'"
            )
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

        py_files, other_files = self._discover_repo_files()
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

        for other_path in other_files:
            rel_other = other_path.relative_to(self.repo_root).as_posix()
            if rel_other in self._MEMORY_UNINDEXED_FILES:
                continue
            if rel_other not in indexed_files:
                missing_files.append(rel_other)

        # SHA-256 Content Fingerprint + AST Symbol Signature Verification
        fp = self._compute_repo_fingerprints()
        expected_hashes = {
            k: v
            for k, v in fp["all_hashes"].items()
            if k not in self._MEMORY_UNINDEXED_FILES
        }
        expected_ast_syms = fp["ast_symbols"]

        manifest = self._load_index_manifest()
        mem_manifest = (
            manifest.get("memory") if isinstance(manifest.get("memory"), dict) else {}
        )
        manifest_hashes = (
            mem_manifest.get("file_hashes")
            if isinstance(mem_manifest.get("file_hashes"), dict)
            else {}
        )
        manifest_syms = (
            mem_manifest.get("ast_symbols")
            if isinstance(mem_manifest.get("ast_symbols"), dict)
            else {}
        )
        manifest_proj = str(mem_manifest.get("project") or "")
        manifest_root = str(mem_manifest.get("repo_root") or "")

        fingerprint_verified = True
        if (
            not mem_manifest
            or manifest_proj != resolved_proj
            or manifest_root != str(self.repo_root)
        ):
            fingerprint_verified = False
            status["reasons"].append(
                "Codebase Memory SHA-256 index manifest is missing or mismatched"
            )
        else:
            for rel_f, cur_hash in expected_hashes.items():
                if manifest_hashes.get(rel_f) != cur_hash:
                    fingerprint_verified = False
                    stale_files.add(rel_f)

            for rel_py, cur_syms in expected_ast_syms.items():
                if manifest_syms.get(rel_py) != cur_syms:
                    fingerprint_verified = False
                    stale_files.add(rel_py)

            for rec_f in manifest_hashes:
                if rec_f not in expected_hashes:
                    fingerprint_verified = False
                    deleted_files.add(rec_f)

        status["fingerprint_verified"] = fingerprint_verified
        status["valid"] = len(indexed_files) > 0
        status["indexed_files"] = sorted(indexed_files)
        status["deleted_files"] = sorted(deleted_files)
        status["missing_files"] = sorted(missing_files)
        status["stale_files"] = sorted(stale_files)
        status["modified_files"] = sorted(stale_files)

        if deleted_files:
            status["contaminated"] = True
            status["reasons"].append(
                f"Codebase Memory index references {len(deleted_files)} deleted file(s): {', '.join(sorted(deleted_files)[:8])}"
            )
        if missing_files:
            status["reasons"].append(
                f"Codebase Memory index is missing {len(missing_files)} repository file(s): {', '.join(sorted(missing_files)[:8])}"
            )
        if stale_files:
            status["reasons"].append(
                f"Codebase Memory index has stale content/fingerprints for {len(stale_files)} modified file(s): {', '.join(sorted(stale_files)[:8])}"
            )

        needs_refresh = bool(
            not status["valid"]
            or status["contaminated"]
            or status["mismatched_root"]
            or deleted_files
            or missing_files
            or stale_files
            or not fingerprint_verified
        )
        status["needs_refresh"] = needs_refresh
        status["fresh"] = bool(
            status["valid"] and not needs_refresh and fingerprint_verified
        )
        return status

    def ensure_memory_index(
        self,
        project: str | None = None,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        """Ensure Codebase Memory MCP index for `self.repo_root` exists, is fresh, and is verified.
        Revalidates after refresh and raises `RuntimeError` if any stale, incomplete, or mismatched state remains.
        """
        status = self.inspect_memory_status(project)
        if force_refresh or status["needs_refresh"]:
            self.memory_index(project=project, mode="fast")
            post_status = self.inspect_memory_status(project)
            if (
                not post_status["valid"]
                or not post_status["fresh"]
                or post_status["deleted_files"]
                or post_status["missing_files"]
                or post_status["stale_files"]
                or not post_status["fingerprint_verified"]
            ):
                raise RuntimeError(
                    f"Codebase Memory index failed post-refresh verification: {'; '.join(post_status['reasons'])}"
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
        """Index or refresh the repository in Codebase Memory MCP and record SHA-256 source fingerprints."""
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
        resolved_proj = project.strip() if isinstance(project, str) and project.strip() else None
        try:
            parsed = json.loads(output)
            if isinstance(parsed, dict) and isinstance(parsed.get("project"), str):
                resolved_proj = parsed["project"].strip()
                self.default_project = resolved_proj
        except (ValueError, TypeError):
            pass

        if not resolved_proj:
            try:
                resolved_proj = self.resolve_memory_project(preferred_project=project, auto_index=False)
            except Exception:
                pass

        if resolved_proj:
            self._record_memory_manifest(resolved_proj)

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
                    if pkg_parts[0] == "src" and len(pkg_parts) >= 2:
                        mod_to_rel[".".join(pkg_parts[1:])] = rel
            else:
                mod_to_rel[".".join(parts)] = rel
                if parts[0] == "src" and len(parts) >= 2:
                    mod_to_rel[".".join(parts[1:])] = rel
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

        candidates: list[str] = []
        if base_mod:
            candidates.append(base_mod)
            for n in names:
                if n != "*":
                    candidates.append(f"{base_mod}.{n}")
        else:
            for n in names:
                if n != "*":
                    candidates.append(n)

        matched_internal = False
        for cand in candidates:
            if cand in mod_to_rel:
                target_rel = mod_to_rel[cand]
                if target_rel != current_rel:
                    internal.add(target_rel)
                matched_internal = True
            else:
                parts = cand.split(".")
                for i in range(len(parts) - 1, 0, -1):
                    prefix = ".".join(parts[:i])
                    if prefix in mod_to_rel:
                        target_rel = mod_to_rel[prefix]
                        if target_rel != current_rel:
                            internal.add(target_rel)
                        matched_internal = True
                        break

        if not matched_internal and base_mod:
            top_pkg = base_mod.split(".")[0]
            if top_pkg:
                external.add(top_pkg)
        elif not matched_internal and not base_mod:
            for n in names:
                top_pkg = n.split(".")[0]
                if top_pkg and top_pkg != "*":
                    external.add(top_pkg)

        return internal, external

    def _analyze_python_file(
        self,
        py_path: Path,
        mod_to_rel: dict[str, str],
    ) -> dict[str, Any]:
        """Extract docstring, classes, functions, exports, and internal/external imports via AST."""
        rel_path = py_path.relative_to(self.repo_root).as_posix()
        try:
            source = py_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return {
                "path": rel_path,
                "type": "python",
                "error": str(exc),
                "line_count": 0,
                "docstring": "",
                "classes": [],
                "functions": [],
                "exports": [],
                "internal_dependencies": [],
                "external_dependencies": [],
            }

        lines = source.splitlines()
        line_count = len(lines)
        try:
            tree = ast.parse(source, filename=rel_path)
        except SyntaxError as exc:
            return {
                "path": rel_path,
                "type": "python",
                "error": f"SyntaxError: {exc}",
                "line_count": line_count,
                "docstring": "",
                "classes": [],
                "functions": [],
                "exports": [],
                "internal_dependencies": [],
                "external_dependencies": [],
            }

        raw_doc = ast.get_docstring(tree) or ""
        first_doc_line = raw_doc.strip().splitlines()[0].strip() if raw_doc.strip() else ""

        classes: list[dict[str, Any]] = []
        functions: list[dict[str, Any]] = []
        exports: list[str] = []
        constants: list[str] = []
        internal_deps: set[str] = set()
        external_deps: set[str] = set()

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                cls_doc = ast.get_docstring(node) or ""
                cls_doc_summary = (
                    cls_doc.strip().splitlines()[0].strip() if cls_doc.strip() else ""
                )
                methods = [
                    n.name
                    for n in node.body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                classes.append(
                    {
                        "name": node.name,
                        "line": node.lineno,
                        "methods": methods,
                        "docstring": cls_doc_summary,
                    }
                )
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn_doc = ast.get_docstring(node) or ""
                fn_doc_summary = (
                    fn_doc.strip().splitlines()[0].strip() if fn_doc.strip() else ""
                )
                functions.append(
                    {
                        "name": node.name,
                        "line": node.lineno,
                        "docstring": fn_doc_summary,
                    }
                )
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id == "__all__" and isinstance(
                            node.value, (ast.List, ast.Tuple, ast.Set)
                        ):
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Constant) and isinstance(
                                    elt.value, str
                                ):
                                    exports.append(elt.value)
                        elif target.id.isupper() and not target.id.startswith("_"):
                            constants.append(target.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.target.id.isupper() and not node.target.id.startswith("_"):
                    constants.append(node.target.id)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
                in_deps, ex_deps = self._resolve_import_targets(
                    rel_path, None, names, 0, mod_to_rel
                )
                internal_deps.update(in_deps)
                external_deps.update(ex_deps)
            elif isinstance(node, ast.ImportFrom):
                names = [alias.name for alias in node.names]
                in_deps, ex_deps = self._resolve_import_targets(
                    rel_path, node.module, names, node.level or 0, mod_to_rel
                )
                internal_deps.update(in_deps)
                external_deps.update(ex_deps)

        # Infer concise purpose summary if module docstring is absent
        summary = first_doc_line
        if not summary:
            if classes:
                cls_names = ", ".join(c["name"] for c in classes[:4])
                summary = f"Defines class(es) {cls_names}."
            elif functions:
                fn_names = ", ".join(f["name"] for f in functions[:4])
                summary = f"Defines function(s) {fn_names}."
            elif exports:
                summary = f"Package initializer re-exporting {len(exports)} public symbols."
            elif constants:
                summary = f"Configuration/constants module ({', '.join(constants[:5])})."
            else:
                summary = "Python module."

        return {
            "path": rel_path,
            "type": "python",
            "line_count": line_count,
            "docstring": first_doc_line,
            "summary": summary,
            "classes": classes,
            "functions": functions,
            "constants": constants[:12],
            "exports": exports,
            "internal_dependencies": sorted(internal_deps),
            "external_dependencies": sorted(external_deps),
        }

    def _analyze_config_or_doc_file(self, file_path: Path) -> dict[str, Any]:
        """Extract concise summary metadata for configuration and documentation files."""
        rel_path = file_path.relative_to(self.repo_root).as_posix()
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        lines = text.splitlines()
        fname = file_path.name

        if fname.lower().startswith("readme"):
            headings = [
                line.lstrip("#").strip()
                for line in lines
                if line.strip().startswith("#")
            ]
            summary = (
                f"Project documentation and architecture guide ({', '.join(headings[:4])})."
                if headings
                else "Project README documentation."
            )
        elif fname == "pyproject.toml":
            summary = "Python package build metadata, project dependencies, and CLI entry-point configuration."
        elif fname.lower().startswith("requirements") and fname.endswith(".txt"):
            pkgs = [
                line.strip().split("=")[0].split(">")[0].split("<")[0].strip()
                for line in lines
                if line.strip() and not line.strip().startswith("#")
            ]
            summary = (
                f"Pinned runtime dependencies ({', '.join(pkgs[:6])})."
                if pkgs
                else "Python runtime requirements specification."
            )
        elif fname.startswith(".env"):
            env_keys = [
                line.split("=", 1)[0].strip()
                for line in lines
                if "=" in line
                and not line.strip().startswith("#")
                and line.split("=", 1)[0].strip()
            ]
            summary = (
                f"Template environment variable configuration ({', '.join(env_keys[:6])})."
                if env_keys
                else "Template environment variable configuration."
            )
        elif fname == ".gitignore":
            patterns = [
                line.strip()
                for line in lines
                if line.strip() and not line.strip().startswith("#")
            ]
            summary = (
                f"Git ignore rules excluding untracked/generated artifacts ({', '.join(patterns[:5])})."
                if patterns
                else "Git ignore rules excluding untracked and generated files."
            )
        else:
            first_non_empty = next((ln.strip() for ln in lines if ln.strip()), "")
            summary = first_non_empty[:120] or "Configuration or documentation file."

        return {
            "path": rel_path,
            "type": "config_or_doc",
            "line_count": len(lines),
            "docstring": "",
            "summary": summary,
            "classes": [],
            "functions": [],
            "constants": [],
            "exports": [],
            "internal_dependencies": [],
            "external_dependencies": [],
        }

    def _classify_file_layer(
        self,
        rel_path: str,
        rec: dict[str, Any] | None = None,
        all_paths: set[str] | None = None,
    ) -> tuple[int, str]:
        """Dynamically assign an architectural layer rank and human-readable category
        for reading order based on file type, package structure, and AST metadata.
        """
        if (rec and rec.get("type") == "config_or_doc") or not rel_path.endswith(".py"):
            return (0, "Project Overview & Configuration")

        parts = Path(rel_path).parts
        pkg_parts = parts[1:] if len(parts) > 1 and parts[0] == "src" else parts
        fname = parts[-1]

        # Layer 1: Root entry scripts, package __main__.py, and top-level package __init__.py facade
        if len(pkg_parts) == 1:
            if fname in (
                "main.py",
                "app.py",
                "cli.py",
                "run.py",
                "manage.py",
                "test.py",
                "setup.py",
            ) or not (rec or {}).get("imported_by"):
                return (1, "Execution Entry Points & Top-Level Facade")
        if fname == "__main__.py":
            return (1, "Execution Entry Points & Top-Level Facade")
        if len(pkg_parts) == 2 and fname == "__init__.py":
            return (1, "Execution Entry Points & Top-Level Facade")

        # Layer 7: Top-level compatibility shims (e.g. <pkg>/<mod>.py re-exporting <pkg>/<mod>/*)
        if len(pkg_parts) == 2 and fname not in ("__init__.py", "__main__.py"):
            stem = Path(fname).stem
            sibling_prefix = "/".join([*parts[:-1], stem]) + "/"
            has_sibling_subpkg = any(
                p.startswith(sibling_prefix) for p in (all_paths or set())
            )
            doc_and_sum = (
                f"{(rec or {}).get('docstring', '')} {(rec or {}).get('summary', '')}"
            ).lower()
            is_shim_doc = any(
                kw in doc_and_sum for kw in ("compatibility", "shim", "re-export")
            )
            if (has_sibling_subpkg or is_shim_doc) and not (rec or {}).get("imported_by"):
                return (7, "Top-Level Compatibility Shims")

        subpkg = (
            pkg_parts[1].lower()
            if len(pkg_parts) >= 3
            else (
                pkg_parts[0].lower()
                if len(pkg_parts) >= 2
                else Path(rel_path).stem.lower()
            )
        )

        if subpkg in {
            "core",
            "common",
            "base",
            "models",
            "domain",
            "types",
            "schemas",
            "utils",
            "config",
            "shared",
            "primitives",
        }:
            return (2, "Core Foundation (Config, Models, Quota/Error Parsing, Time)")
        if subpkg in {
            "providers",
            "adapters",
            "clients",
            "integrations",
            "backends",
            "connectors",
            "db",
            "database",
            "storage",
            "repositories",
        }:
            return (3, "Provider Adapters & Model Discovery")
        if subpkg in {
            "router",
            "routing",
            "engine",
            "services",
            "pipeline",
            "workflows",
            "middleware",
            "controllers",
            "orchestration",
        }:
            return (
                4,
                "Smart Routing, Circuit-Breaker State, Telemetry & Runtime Diagnostics",
            )
        if subpkg in {
            "agent",
            "agents",
            "intelligence",
            "planner",
            "reasoning",
            "tools",
            "verifiers",
            "analysis",
        }:
            return (5, "Architecture-Aware Coding Agent & Code Intelligence")
        if subpkg in {
            "cli",
            "ui",
            "api",
            "web",
            "server",
            "views",
            "commands",
            "app",
        }:
            return (6, "CLI Application Runner")

        if len(pkg_parts) >= 3:
            return (5, f"Package Modules ({'/'.join(parts[:-1])})")
        return (5, "Supporting Modules")

    @staticmethod
    def _early_layer_priority(rel_path: str, rank: int) -> tuple[int, str]:
        """Deterministic within-layer ordering key for Layer 0 (config/doc) and Layer 1 (entry points)."""
        fname = Path(rel_path).name.lower()
        parts = Path(rel_path).parts
        if rank == 0:
            if fname.startswith("readme"):
                return (0, rel_path)
            if fname in (
                "pyproject.toml",
                "setup.py",
                "setup.cfg",
                "package.json",
                "cargo.toml",
            ):
                return (1, rel_path)
            if fname.startswith("requirements") or fname == "pipfile":
                return (2, rel_path)
            if fname.startswith(".env"):
                return (3, rel_path)
            if fname == ".gitignore":
                return (4, rel_path)
            return (5, rel_path)
        if rank == 1:
            if len(parts) == 1 and fname in (
                "main.py",
                "app.py",
                "run.py",
                "cli.py",
                "manage.py",
            ):
                return (0, rel_path)
            if fname == "__main__.py":
                return (1, rel_path)
            if len(parts) == 1:
                return (2, rel_path)
            if fname == "__init__.py":
                return (3, rel_path)
            return (4, rel_path)
        return (99, rel_path)

    def _compute_reading_order(
        self,
        file_records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Compute a deterministic, dependency-aware reading order across all repository files.
        Across all implementation layers (ranks >= 2), internal dependencies are guaranteed to
        precede the modules that import them, using backward-propagated architectural layer rank
        as the primary tie-breaker.
        """
        by_path: dict[str, dict[str, Any]] = {rec["path"]: rec for rec in file_records}
        all_paths_set: set[str] = set(by_path.keys())
        rank_by_path: dict[str, int] = {}
        label_by_path: dict[str, str] = {}

        for rec in file_records:
            rank, label = self._classify_file_layer(
                rec["path"], rec=rec, all_paths=all_paths_set
            )
            rank_by_path[rec["path"]] = rank
            label_by_path[rec["path"]] = label

        ordered_paths: list[str] = []
        for early_rank in (0, 1):
            early_paths = [
                p for p, r in rank_by_path.items() if r == early_rank
            ]
            ordered_paths.extend(
                sorted(
                    early_paths,
                    key=lambda p, r=early_rank: self._early_layer_priority(p, r),
                )
            )

        remaining: set[str] = {
            p for p, r in rank_by_path.items() if r >= 2
        }

        def _unmet_deps(path: str, rem: set[str]) -> set[str]:
            rec = by_path[path]
            raw_deps = set(rec.get("internal_dependencies") or []) & rem
            filtered: set[str] = set()
            for dep in raw_deps:
                # Ignore mutual import cycles
                if path in set(by_path[dep].get("internal_dependencies") or []):
                    continue
                # A submodule in a package does not wait on its own package's __init__.py facade
                if (
                    dep.endswith("/__init__.py")
                    and Path(dep).parent == Path(path).parent
                ):
                    continue
                filtered.add(dep)
            # Ensure a subpackage's __init__.py facade waits for all non-__init__ modules in its subpackage
            if path.endswith("/__init__.py"):
                pkg_dir = Path(path).parent
                for other in rem:
                    if other != path and Path(other).parent == pkg_dir:
                        filtered.add(other)
            return filtered

        # Propagate effective layer ranks backward along dependency edges so that any module
        # imported by an earlier-layer module is scheduled at or before its importer's layer.
        changed = True
        while changed:
            changed = False
            for mod_path in sorted(remaining):
                mod_rank = rank_by_path[mod_path]
                for dep_path in _unmet_deps(mod_path, remaining):
                    if rank_by_path[dep_path] > mod_rank:
                        rank_by_path[dep_path] = mod_rank
                        label_by_path[dep_path] = label_by_path[mod_path]
                        changed = True

        while remaining:
            ready = [
                p for p in remaining if not _unmet_deps(p, remaining)
            ]
            if not ready:
                ready = [
                    min(
                        remaining,
                        key=lambda p: (
                            len(_unmet_deps(p, remaining)),
                            rank_by_path[p],
                            1 if p.endswith("/__init__.py") else 0,
                            p,
                        ),
                    )
                ]
            chosen = min(
                ready,
                key=lambda p: (
                    rank_by_path[p],
                    1 if p.endswith("/__init__.py") else 0,
                    p,
                ),
            )
            ordered_paths.append(chosen)
            remaining.remove(chosen)

        ordered_items: list[dict[str, Any]] = []
        for step, p in enumerate(ordered_paths, start=1):
            rec = by_path[p]
            ordered_items.append(
                {
                    "step": step,
                    "path": p,
                    "layer_rank": rank_by_path[p],
                    "layer": label_by_path[p],
                    "summary": rec.get("summary", ""),
                    "depends_on": rec.get("internal_dependencies", []),
                    "classes": [c["name"] for c in rec.get("classes", [])],
                    "functions": [f["name"] for f in rec.get("functions", [])],
                }
            )

        return ordered_items

    def build_repository_inventory(self) -> dict[str, Any]:
        """Build a complete, deterministic inventory of all relevant repository files,
        their AST symbols, docstrings, internal/external dependencies, excluded files,
        and a dependency-aware reading order.
        """
        py_files, other_files, excluded_dirs, excluded_files = (
            self._discover_repo_files_with_exclusions()
        )
        mod_to_rel = self._module_map_for_repo(py_files)

        records: list[dict[str, Any]] = []
        for other_path in other_files:
            records.append(self._analyze_config_or_doc_file(other_path))
        for py_path in py_files:
            records.append(self._analyze_python_file(py_path, mod_to_rel))

        # Compute reverse dependency map (`imported_by`)
        imported_by: dict[str, list[str]] = {r["path"]: [] for r in records}
        for rec in records:
            for dep in rec.get("internal_dependencies", []):
                if dep in imported_by:
                    imported_by[dep].append(rec["path"])

        for rec in records:
            rec["imported_by"] = sorted(imported_by.get(rec["path"], []))

        reading_order = self._compute_reading_order(records)

        return {
            "repo_root": str(self.repo_root),
            "total_files": len(records),
            "python_files_count": len(py_files),
            "config_doc_files_count": len(other_files),
            "excluded_dirs": excluded_dirs,
            "excluded_files": excluded_files,
            "files": records,
            "reading_order": reading_order,
        }

    def format_inventory_for_prompt(self, inventory: dict[str, Any]) -> str:
        """Render the complete repository inventory and dependency-aware reading order
        into a structured Markdown block for LLM prompt grounding.
        """
        lines: list[str] = [
            f"Repository Root: `{inventory.get('repo_root')}`",
            f"Total Analyzed Files: {inventory.get('total_files', 0)} "
            f"({inventory.get('python_files_count', 0)} Python modules, "
            f"{inventory.get('config_doc_files_count', 0)} config/documentation files)",
        ]

        excluded_dirs = inventory.get("excluded_dirs") or []
        excluded_files = inventory.get("excluded_files") or []
        if excluded_dirs or excluded_files:
            excl_dir_strs = [f"`{d['path']}` ({d['reason']})" for d in excluded_dirs]
            excl_file_strs = [f"`{f['path']}` ({f['reason']})" for f in excluded_files]
            lines.append(
                "Excluded Directories: "
                + (", ".join(excl_dir_strs) if excl_dir_strs else "None")
            )
            lines.append(
                "Excluded Files: "
                + (", ".join(excl_file_strs) if excl_file_strs else "None")
            )

        lines.append("")
        lines.append("### Recommended Dependency-Aware Reading Order & Complete File Inventory")

        by_path = {f["path"]: f for f in inventory.get("files", [])}
        current_layer = None

        for item in inventory.get("reading_order", []):
            layer = item.get("layer", "Modules")
            if layer != current_layer:
                current_layer = layer
                lines.append(f"\n#### {layer}")

            path = item["path"]
            rec = by_path.get(path, {})
            summary = item.get("summary") or rec.get("summary") or ""
            cls_list = item.get("classes") or []
            fn_list = item.get("functions") or []
            deps = item.get("depends_on") or []
            imported_by = rec.get("imported_by") or []
            exports = rec.get("exports") or []
            constants = rec.get("constants") or []

            details: list[str] = []
            if cls_list:
                details.append(f"Classes: `{', '.join(cls_list)}`")
            if fn_list:
                details.append(f"Functions: `{', '.join(fn_list[:10])}`")
            if constants and not cls_list and not fn_list:
                details.append(f"Constants: `{', '.join(constants[:8])}`")
            if exports and path.endswith("__init__.py"):
                details.append(f"Re-exports ({len(exports)} symbols)")
            if deps:
                details.append(f"Imports from repo: `{', '.join(deps)}`")
            else:
                if rec.get("type") == "python":
                    details.append("Imports from repo: None (leaf/standalone module)")
            if imported_by:
                details.append(f"Imported by: `{', '.join(imported_by[:8])}`")

            detail_str = f" — {' | '.join(details)}" if details else ""
            lines.append(f"{item['step']}. `{path}`: {summary}{detail_str}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # High-level Context Assembly & Post-Change Synchronization
    # ------------------------------------------------------------------

    def _extract_candidate_symbols(
        self,
        question: str,
        inventory: dict[str, Any],
    ) -> list[str]:
        """Infer relevant symbol search patterns dynamically from `question` and the repository inventory."""
        known_symbols: set[str] = set()
        files = inventory.get("files", [])
        for f in files:
            for c in f.get("classes", []):
                if c.get("name"):
                    known_symbols.add(str(c["name"]))
            for fn in f.get("functions", []):
                if fn.get("name"):
                    known_symbols.add(str(fn["name"]))

        q_tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", question))
        matched = [sym for sym in sorted(known_symbols) if sym in q_tokens]
        if matched:
            return matched[:6]

        # Dynamically select representative top-level classes and entry-point functions from the inventory
        ranked_files = sorted(
            [f for f in files if isinstance(f, dict) and f.get("type") == "python"],
            key=lambda f: (
                len(f.get("classes") or []),
                len(f.get("imported_by") or []),
                len(f.get("internal_dependencies") or []),
            ),
            reverse=True,
        )
        representative: list[str] = []
        for f in ranked_files:
            for c in f.get("classes", []):
                cname = str(c.get("name") or "")
                if cname and cname not in representative:
                    representative.append(cname)
                    if len(representative) >= 6:
                        break
            if len(representative) >= 6:
                break
        if len(representative) < 6:
            for f in ranked_files:
                for fn in f.get("functions", []):
                    fname = str(fn.get("name") or "")
                    if (
                        fname
                        and not fname.startswith("_")
                        and fname not in representative
                    ):
                        representative.append(fname)
                        if len(representative) >= 6:
                            break
                if len(representative) >= 6:
                    break

        if representative:
            return ["|".join(representative[:6])]
        return [".*"]

    def gather_context(
        self,
        question: str,
        project: str | None = None,
        symbol_pattern: str | None = None,
        token_budget: int = 3500,
        auto_build: bool = False,
        strict: bool = False,
    ) -> dict[str, Any]:
        """Gather comprehensive codebase context from:
        1. Deterministic AST repository inventory (all files, docstrings, symbols, dependencies, reading order)
        2. Graphify structural graph (`graphify-out/graph.json` + `GRAPH_REPORT.md`) — queried ONLY if verified fresh
        3. Codebase Memory MCP (`get_architecture` with `aspects=["all"]` + symbol search) — queried ONLY if verified fresh
        """
        errors: list[str] = []

        # 1. Deterministic full-repository AST inventory & reading order
        inventory = self.build_repository_inventory()
        formatted_inventory = self.format_inventory_for_prompt(inventory)

        # 2. Graphify verification, auto-refresh, and structural query (never query unverified index data)
        graphify_status: dict[str, Any] = {}
        graphify_ctx: str | None = None
        graphify_report_summary: str | None = None

        try:
            if auto_build:
                graphify_status = self.ensure_graphify_graph(force_rebuild=False)
            else:
                graphify_status = self.inspect_graphify_status()
        except Exception as exc:
            errors.append(f"Graphify index verification/update error: {exc}")
            graphify_status = self.inspect_graphify_status()

        graphify_verified_and_fresh = bool(
            graphify_status.get("valid")
            and graphify_status.get("fresh")
            and graphify_status.get("fingerprint_verified")
            and graphify_status.get("ast_symbols_verified")
            and not graphify_status.get("deleted_files")
            and not graphify_status.get("missing_files")
            and not graphify_status.get("stale_files")
        )
        graphify_status["verified"] = graphify_verified_and_fresh

        if graphify_verified_and_fresh:
            try:
                graphify_ctx = self.graphify_query(question, budget=token_budget)
            except Exception as exc:
                errors.append(f"Graphify query error: {exc}")

            if self.graph_report_path.is_file():
                try:
                    report_text = self.graph_report_path.read_text(
                        encoding="utf-8", errors="replace"
                    )
                    valid_repo_paths = {
                        str(f.get("path"))
                        for f in inventory.get("files", [])
                        if isinstance(f, dict) and f.get("path")
                    }
                    report_py_refs = re.findall(
                        r"(?<![A-Za-z0-9_./\\-])((?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.py)\b",
                        report_text,
                    )
                    if not any(
                        ref.replace("\\", "/").lstrip("./") not in valid_repo_paths
                        for ref in report_py_refs
                    ):
                        graphify_report_summary = "\n".join(
                            report_text.splitlines()[:80]
                        )
                except OSError:
                    pass
        else:
            reasons_str = "; ".join(graphify_status.get("reasons", [])) or "index is stale or unverified"
            errors.append(
                f"Refusing to query unverified/stale Graphify index: {reasons_str}"
            )

        # 3. Codebase Memory MCP verification, project resolution, architecture & symbols (never query unverified index data)
        memory_status: dict[str, Any] = {}
        resolved_project: str | None = None
        memory_arch: Any = None
        memory_symbols: Any = None

        try:
            if auto_build:
                memory_status = self.ensure_memory_index(
                    project=project, force_refresh=False
                )
            else:
                memory_status = self.inspect_memory_status(project=project)
            resolved_project = memory_status.get("project")
        except Exception as exc:
            errors.append(f"Codebase Memory index verification/update error: {exc}")
            memory_status = self.inspect_memory_status(project=project)
            resolved_project = memory_status.get("project")

        memory_verified_and_fresh = bool(
            resolved_project
            and memory_status.get("valid")
            and memory_status.get("fresh")
            and memory_status.get("fingerprint_verified")
            and not memory_status.get("deleted_files")
            and not memory_status.get("missing_files")
            and not memory_status.get("stale_files")
        )
        memory_status["verified"] = memory_verified_and_fresh

        if memory_verified_and_fresh and resolved_project:
            try:
                raw_arch = self.memory_architecture(resolved_project, aspects=["all"])
                try:
                    memory_arch = json.loads(raw_arch)
                except ValueError:
                    memory_arch = raw_arch
            except Exception as exc:
                errors.append(f"Codebase Memory architecture error: {exc}")

            effective_patterns = (
                [symbol_pattern.strip()]
                if isinstance(symbol_pattern, str) and symbol_pattern.strip()
                else self._extract_candidate_symbols(question, inventory)
            )
            pattern_str = "|".join(effective_patterns)
            try:
                raw_syms = self.memory_query(
                    resolved_project, pattern_str, limit=100
                )
                try:
                    memory_symbols = json.loads(raw_syms)
                except ValueError:
                    memory_symbols = raw_syms
            except Exception as exc:
                errors.append(f"Codebase Memory symbol query error: {exc}")
        else:
            reasons_str = "; ".join(memory_status.get("reasons", [])) or "index is stale or unverified"
            errors.append(
                f"Refusing to query unverified/stale Codebase Memory MCP index: {reasons_str}"
            )

        coverage = {
            "total_repo_files": inventory["total_files"],
            "python_files": inventory["python_files_count"],
            "config_doc_files": inventory["config_doc_files_count"],
            "excluded_dirs_count": len(inventory.get("excluded_dirs", [])),
            "excluded_files_count": len(inventory.get("excluded_files", [])),
            "excluded_dirs": inventory.get("excluded_dirs", []),
            "excluded_files": inventory.get("excluded_files", []),
            "graphify_indexed_files": len(graphify_status.get("indexed_files", [])),
            "graphify_missing_files": graphify_status.get("missing_files", []),
            "graphify_deleted_files": graphify_status.get("deleted_files", []),
            "graphify_stale_files": graphify_status.get("stale_files", []),
            "graphify_fingerprint_verified": bool(
                graphify_status.get("fingerprint_verified")
            ),
            "graphify_ast_symbols_verified": bool(
                graphify_status.get("ast_symbols_verified")
            ),
            "memory_indexed_files": len(memory_status.get("indexed_files", [])),
            "memory_missing_files": memory_status.get("missing_files", []),
            "memory_deleted_files": memory_status.get("deleted_files", []),
            "memory_stale_files": memory_status.get("stale_files", []),
            "memory_fingerprint_verified": bool(
                memory_status.get("fingerprint_verified")
            ),
            "complete_inventory_available": inventory["total_files"] > 0,
        }

        if strict and (
            errors
            or not graphify_verified_and_fresh
            or not memory_verified_and_fresh
        ):
            raise RuntimeError(
                "Strict context verification failed: "
                + (
                    "; ".join(errors)
                    if errors
                    else f"graphify_fresh={graphify_status.get('fresh')}, memory_fresh={memory_status.get('fresh')}"
                )
            )

        return {
            "repo_root": str(self.repo_root),
            "question": question,
            "project": resolved_project,
            "repository_inventory": inventory,
            "formatted_inventory": formatted_inventory,
            "reading_order": inventory["reading_order"],
            "graphify_context": graphify_ctx,
            "graphify_report": graphify_report_summary,
            "memory_architecture": memory_arch,
            "architecture_overview": memory_arch,
            "memory_symbols": memory_symbols,
            "memory_context": memory_symbols,
            "index_status": {
                "graphify": graphify_status,
                "memory": memory_status,
            },
            "coverage": coverage,
            "errors": errors,
        }

    def refresh_knowledge(self, project: str | None = None) -> dict[str, Any]:
        """Refresh both Graphify and Codebase Memory MCP indexes after code edits
        and revalidate that both indexes are fresh and fingerprint-verified.
        """
        result: dict[str, Any] = {
            "repo_root": str(self.repo_root),
            "graphify": None,
            "memory": None,
            "graphify_updated": False,
            "memory_indexed": False,
            "graphify_output": None,
            "memory_output": None,
            "graphify_status": None,
            "memory_status": None,
            "errors": [],
        }

        try:
            result["graphify_output"] = self.graphify_update(".")
            result["graphify"] = result["graphify_output"]
            graph_status = self.inspect_graphify_status()
            result["graphify_status"] = graph_status
            result["graphify_updated"] = bool(
                graph_status.get("valid")
                and graph_status.get("fresh")
                and graph_status.get("fingerprint_verified")
                and graph_status.get("ast_symbols_verified")
                and not graph_status.get("deleted_files")
                and not graph_status.get("missing_files")
                and not graph_status.get("stale_files")
            )
            if not result["graphify_updated"]:
                result["errors"].append(
                    f"Graphify post-refresh validation failed: {graph_status.get('reasons')}"
                )
        except Exception as exc:
            result["errors"].append(f"Graphify update error: {exc}")

        try:
            result["memory_output"] = self.memory_index(project=project, mode="fast")
            result["memory"] = result["memory_output"]
            mem_status = self.inspect_memory_status(project=project)
            result["memory_status"] = mem_status
            result["memory_indexed"] = bool(
                mem_status.get("valid")
                and mem_status.get("fresh")
                and mem_status.get("fingerprint_verified")
                and not mem_status.get("deleted_files")
                and not mem_status.get("missing_files")
                and not mem_status.get("stale_files")
            )
            if not result["memory_indexed"]:
                result["errors"].append(
                    f"Codebase Memory post-refresh validation failed: {mem_status.get('reasons')}"
                )
        except Exception as exc:
            result["errors"].append(f"Codebase Memory index error: {exc}")

        result["synced"] = bool(
            result["graphify_updated"]
            and result["memory_indexed"]
            and not result["errors"]
        )
        return result
