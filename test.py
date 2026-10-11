from typing import Any

from ntg import query_directory, query_directory_structured


def structured_answer_from_directory(
    directory: str,
    query: str,
    strict: bool = True,
) -> dict[str, Any]:
    """Return the structured result, including verification and retrieval diagnostics."""
    return query_directory_structured(
        directory=directory,
        query=query,
        model="auto",
        capabilities=["coding"],
        auto_build_graph=True,
        strict=strict,
        include_diagnostics=True,
    )


def answer_from_directory(directory: str, query: str, strict: bool = True) -> str:
    """
    Return the answer as a string.

    `return_structured=False` is explicit because query_directory may default to
    returning a structured result in the updated API.
    """
    result = query_directory(
        directory=directory,
        query=query,
        model="auto",
        capabilities=["coding"],
        auto_build_graph=True,
        strict=strict,
        include_diagnostics=True,
        return_structured=False,
    )

    if not isinstance(result, str):
        raise TypeError(
            "query_directory(..., return_structured=False) was expected to return "
            f"a string, but returned {type(result).__name__}."
        )
    return result


if __name__ == "__main__":
    target_directory = r"c:\Users\perur\Desktop\NTG"
    user_query = (
        "I want to learn and understand this project. Provide the sequence of "
        "files I should read to understand the project, with a brief description "
        "of each file."
    )

    print(f"Analyzing directory: {target_directory}")
    print(f"Query: {user_query}\n")

    structured_result = structured_answer_from_directory(
        target_directory,
        user_query,
        strict=True,
    )

    print("=== Structured Verification Summary ===")
    print(f"Status: {structured_result.get('status')}")
    print(f"Complete: {structured_result.get('complete')}")
    print(f"Model Invoked: {structured_result.get('model_invoked')}")
    print(f"Retrieval Blocked: {structured_result.get('retrieval_blocked')}")
    print(f"Retrieval Errors: {len(structured_result.get('errors', []))}")
    print("\n=== Answer ===")
    print(structured_result.get("answer", ""))
