from ntg import ArchitectureAwareAgent, query_directory


def answer_from_directory(directory: str, query: str) -> str:
    """
    1. Builds/queries the Graphify knowledge graph & Codebase Memory MCP index for `directory`.
    2. Routes the architecture-grounded prompt through UnifiedNTGRouter (Groq/Gemini/OpenRouter/etc.).
    3. Returns the final answer.
    """
    return query_directory(
        directory=directory,
        query=query,
        model="auto",              # or "groq", "gemini", "openrouter", "nvidia", "cohere"
        capabilities=["coding"],   # routes to coding-capable deployments
        auto_build_graph=True,     # automatically indexes the directory if not yet indexed
    )


if __name__ == "__main__":
    target_directory = r"c:\Users\perur\Desktop\NTG"
    user_query = "i want to learn and understand this project,can you provide the seqence of files that i should read to understand the project and also provide a brief description of each file"

    print(f"Analyzing directory: {target_directory}")
    print(f"Query: {user_query}\n")

    answer = answer_from_directory(target_directory, user_query)
    print("=== Answer ===")
    print(answer)