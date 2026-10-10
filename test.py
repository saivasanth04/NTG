from ntg import ArchitectureAwareAgent, query_directory


def answer_from_directory(directory: str, query: str, strict: bool = True) -> str:
    """
    1. Builds/verifies the Graphify knowledge graph, Codebase Memory MCP index, and AST inventory for `directory`.
    2. Routes the architecture-grounded prompt through UnifiedNTGRouter (Groq/Gemini/OpenRouter/etc.).
    3. Validates file coverage and factual grounding, and returns the final answer with diagnostics.
    """
    return query_directory(
        directory=directory,
        query=query,
        model="auto",              # or "groq", "gemini", "openrouter", "nvidia", "cohere"
        capabilities=["coding"],   # routes to coding-capable deployments
        auto_build_graph=True,     # automatically indexes/refreshes the directory when needed
        strict=strict,             # refuses to mark answer complete if critical indexing/retrieval fails
        include_diagnostics=True,  # appends verified index status, coverage, and retrieval diagnostics
    )


if __name__ == "__main__":
    target_directory = r"c:\Users\perur\Desktop\NTG"
    user_query = "i want to learn and understand this project,can you provide the seqence of files that i should read to understand the project and also provide a brief description of each file"

    print(f"Analyzing directory: {target_directory}")
    print(f"Query: {user_query}\n")

    answer = answer_from_directory(target_directory, user_query)
    print("=== Answer ===")
    print(answer)