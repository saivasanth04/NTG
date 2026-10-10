"""Implementation verification and validation utilities for NTG coding agent."""

from __future__ import annotations

import ast
from pathlib import Path
import py_compile
import shutil
import subprocess
from typing import Any

from ntg.planner import record_validation_result

_EXCLUDED_DIRS = {
    ".git",
    "__pycache__",
    "venv",
    ".venv",
    "env",
    "node_modules",
    ".ntg",
    "graphify-out",
    ".codebase-memory",
}

_DISALLOWED_SHELL_BINARIES = {
    "sh",
    "bash",
    "zsh",
    "cmd",
    "cmd.exe",
    "powershell",
    "powershell.exe",
    "pwsh",
    "pwsh.exe",
}


def _resolve_repo_root(repo_root: str | Path) -> Path:
    root = Path(repo_root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Invalid repository root: {repo_root}")
    return root


def _resolve_repo_file(root: Path, file_path: str | Path) -> Path:
    candidate = Path(file_path)
    resolved = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"File path escapes repository root: {file_path}") from exc
    return resolved


def _run_Subprocess(
    args: list[str],
    cwd: Path,
    timeout: int = 60,
) -> subprocess.CompletedProcess[str]:
    """Run a command with shell=False and Windows shim resolution fallback."""
    if not args or any(not isinstance(a, str) or not a.strip() for a in args):
        raise ValueError("Invalid command arguments.")

    base_kwargs: dict[str, Any] = {
        "cwd": cwd,
        "capture_output": True,
        "text": True,
        "check": False,
        "shell": False,
        "timeout": timeout,
    }

    def _invoke(cmd_args: list[str]) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(cmd_args, stdin=subprocess.DEVNULL, **base_kwargs)
        except TypeError:
            return subprocess.run(cmd_args, **base_kwargs)

    try:
        return _invoke(args)
    except FileNotFoundError:
        resolved = shutil.which(args[0])
        if resolved and resolved != args[0]:
            return _invoke([resolved, *args[1:]])
        raise


def verify_python_syntax(
    repo_root: str | Path,
    files: list[str] | None = None,
) -> dict[str, Any]:
    """
    Validate Python syntax using both AST parsing and bytecode compilation.
    If `files` is omitted or empty, checks all Python files in `repo_root`.
    """
    root = _resolve_repo_root(repo_root)
    target_paths: list[Path] = []

    if files:
        for rel_or_abs in files:
            resolved = _resolve_repo_file(root, rel_or_abs)
            if resolved.suffix == ".py":
                target_paths.append(resolved)
    else:
        for path in sorted(root.rglob("*.py")):
            rel_parts = set(path.relative_to(root).parts)
            if rel_parts & _EXCLUDED_DIRS:
                continue
            target_paths.append(path)

    checked_files: list[str] = []
    errors: list[dict[str, Any]] = []

    for path in target_paths:
        rel_str = str(path.relative_to(root)).replace("\\", "/")
        checked_files.append(rel_str)
        if not path.exists():
            errors.append(
                {
                    "file": rel_str,
                    "error": "File does not exist.",
                }
            )
            continue

        try:
            source = path.read_text(encoding="utf-8")
            ast.parse(source, filename=rel_str)
            py_compile.compile(str(path), doraise=True)
        except Exception as exc:
            errors.append(
                {
                    "file": rel_str,
                    "error": str(exc),
                }
            )

    return {
        "check": "python_syntax",
        "passed": len(errors) == 0,
        "checked_files": checked_files,
        "errors": errors,
    }


def run_verification_command(
    repo_root: str | Path,
    command: list[str],
    timeout: int = 60,
) -> dict[str, Any]:
    """
    Execute an explicit verification command argv list inside `repo_root`
    without invoking a shell.
    """
    root = _resolve_repo_root(repo_root)
    if not isinstance(command, list) or not command:
        raise ValueError("command must be a non-empty list of strings.")
    if any(not isinstance(part, str) or not part.strip() for part in command):
        raise ValueError("All command arguments must be non-empty strings.")

    binary_name = Path(command[0]).name.lower()
    if binary_name in _DISALLOWED_SHELL_BINARIES:
        raise ValueError(f"Direct shell invocation is not permitted in verification commands: {command[0]}")

    proc = _run_Subprocess(command, cwd=root, timeout=timeout)
    return {
        "check": "command",
        "command": list(command),
        "passed": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": (proc.stdout or "").strip(),
        "stderr": (proc.stderr or "").strip(),
    }


def inspect_git_diff(repo_root: str | Path) -> dict[str, Any]:
    """Return modified and untracked files reported by Git in `repo_root`."""
    root = _resolve_repo_root(repo_root)
    status_proc = _run_Subprocess(["git", "status", "--porcelain"], cwd=root, timeout=30)
    diff_proc = _run_Subprocess(["git", "diff", "--name-only"], cwd=root, timeout=30)

    changed_files: list[str] = []
    if diff_proc.returncode == 0 and diff_proc.stdout:
        for line in diff_proc.stdout.splitlines():
            cleaned = line.strip()
            if cleaned:
                changed_files.append(cleaned)

    status_entries: list[str] = []
    if status_proc.returncode == 0 and status_proc.stdout:
        for line in status_proc.stdout.splitlines():
            if line.strip():
                status_entries.append(line.rstrip())

    return {
        "check": "git_status",
        "passed": status_proc.returncode == 0,
        "changed_files": changed_files,
        "status_entries": status_entries,
    }


class CodeVerifier:
    """Verifies code changes against syntax, git diff state, and test/check commands."""

    def __init__(self, repo_root: str | Path) -> None:
        self.repo_root = _resolve_repo_root(repo_root)

    def verify_syntax(self, files: list[str] | None = None) -> dict[str, Any]:
        """Run AST and py_compile syntax verification."""
        return verify_python_syntax(self.repo_root, files=files)

    def run_command(self, command: list[str], timeout: int = 60) -> dict[str, Any]:
        """Run a verification command safely without a shell."""
        return run_verification_command(self.repo_root, command=command, timeout=timeout)

    def git_status(self) -> dict[str, Any]:
        """Inspect repository git diff and porcelain status."""
        return inspect_git_diff(self.repo_root)

    def verify_plan(
        self,
        plan: dict[str, Any],
        commands: list[list[str]] | None = None,
        timeout: int = 60,
    ) -> dict[str, Any]:
        """
        Verify a plan by running Python syntax checks on changed files (or the whole repo)
        followed by any explicit verification commands, recording all results on the plan.
        """
        if not isinstance(plan, dict):
            raise ValueError("plan must be a dictionary.")

        changed_files = plan.get("files_changed") or None
        results: list[dict[str, Any]] = []

        syntax_result = self.verify_syntax(files=changed_files)
        results.append(syntax_result)
        record_validation_result(plan, syntax_result)

        for cmd in commands or []:
            cmd_result = self.run_command(cmd, timeout=timeout)
            results.append(cmd_result)
            record_validation_result(plan, cmd_result)

        overall_passed = all(bool(r.get("passed", False)) for r in results)
        return {
            "passed": overall_passed,
            "results": results,
            "plan": plan,
        }
