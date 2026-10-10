"""Architecture-aware AI coding agent, code intelligence, planning, and verification."""

from ntg.agent.code_intelligence import CodeIntelligence
from ntg.agent.orchestrator import (
    ArchitectureAwareAgent,
    CodingAgent,
    query_directory,
    query_directory_structured,
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
    "query_directory",
    "query_directory_structured",
    "CodeIntelligence",
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
    "CodeVerifier",
    "verify_python_syntax",
    "run_verification_command",
    "inspect_git_diff",
]
