"""Verification helpers and compatibility exports for repository syntax, command, and diff checks."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
from typing import Any

from ntg.agent.verifier import (
    CodeVerifier,
    inspect_git_diff,
    run_verification_command,
    verify_python_syntax,
)


def run_checked(
    repo_root: str | Path,
    command: list[str],
    timeout: int = 120,
) -> dict[str, Any]:
    """Run an explicitly selected validation command safely with DEVNULL stdin."""
    root = Path(repo_root).resolve()

    if not root.is_dir():
        raise ValueError("Repository root must exist.")

    try:
        result = subprocess.run(
            command,
            cwd=root,
            shell=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError as exc:
        resolved = shutil.which(command[0]) if command else None
        if not resolved:
            raise RuntimeError(f"Executable not found: {command[0] if command else ''}") from exc
        result = subprocess.run(
            [resolved, *command[1:]],
            cwd=root,
            shell=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            stdin=subprocess.DEVNULL,
        )

    return {
        "command": command,
        "exit_code": result.returncode,
        "passed": result.returncode == 0,
        "stdout": (result.stdout or "")[-5000:],
        "stderr": (result.stderr or "")[-5000:],
    }


def inspect_diff(repo_root: str | Path) -> dict[str, Any]:
    """Inspect the working-tree diff; does not modify source files."""
    return run_checked(
        repo_root,
        ["git", "diff", "--check"],
    )


def refresh_graphify(repo_root: str | Path) -> dict[str, Any]:
    """Refresh Graphify after an approved code change."""
    return run_checked(
        repo_root,
        ["graphify", "update", "."],
        timeout=300,
    )


__all__ = [
    "CodeVerifier",
    "verify_python_syntax",
    "run_verification_command",
    "inspect_git_diff",
    "run_checked",
    "inspect_diff",
    "refresh_graphify",
]

