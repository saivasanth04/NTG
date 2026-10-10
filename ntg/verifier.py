
# ntg/verifier.py
from __future__ import annotations

import subprocess
from pathlib import Path


def run_checked(
    repo_root: str | Path,
    command: list[str],
    timeout: int = 120,
) -> dict:
    """Run an explicitly selected validation command."""
    root = Path(repo_root).resolve()

    if not root.is_dir():
        raise ValueError("Repository root must exist.")

    result = subprocess.run(
        command,
        cwd=root,
        shell=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )

    return {
        "command": command,
        "exit_code": result.returncode,
        "passed": result.returncode == 0,
        "stdout": result.stdout[-5000:],
        "stderr": result.stderr[-5000:],
    }


def inspect_diff(repo_root: str | Path) -> dict:
    """Inspect the working-tree diff; does not modify source files."""
    return run_checked(
        repo_root,
        ["git", "diff", "--check"],
    )


def refresh_graphify(repo_root: str | Path) -> dict:
    """Refresh Graphify after an approved code change."""
    return run_checked(
        repo_root,
        ["graphify", "update", "."],
        timeout=300,
    )
