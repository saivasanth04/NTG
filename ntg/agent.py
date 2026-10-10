"""Compatibility module re-exporting the canonical NTG architecture-aware agent API."""

from __future__ import annotations

from ntg.agent.code_intelligence import CodeIntelligence
from ntg.agent.orchestrator import (
    ArchitectureAwareAgent,
    CodingAgent,
    query_directory,
)
from ntg.agent.planner import (
    approve_plan,
    build_change_plan,
    build_planning_prompt,
    build_query_prompt,
    extract_response_content,
    finalize_grounded_answer,
    record_files_changed,
    record_validation_result,
    reject_plan,
    require_approved_plan,
    validate_query_answer,
)
from ntg.agent.verifier import (
    CodeVerifier,
    inspect_git_diff,
    run_verification_command,
    verify_python_syntax,
)

__all__ = [
    "ArchitectureAwareAgent",
    "CodingAgent",
    "CodeIntelligence",
    "CodeVerifier",
    "query_directory",
    "extract_response_content",
    "build_planning_prompt",
    "build_query_prompt",
    "validate_query_answer",
    "finalize_grounded_answer",
    "build_change_plan",
    "approve_plan",
    "reject_plan",
    "require_approved_plan",
    "record_files_changed",
    "record_validation_result",
    "verify_python_syntax",
    "run_verification_command",
    "inspect_git_diff",
]

