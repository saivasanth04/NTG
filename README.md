# NTG: Architecture-Aware AI Coding Agent & Multi-Provider LiteLLM Smart Router

A production-grade, architecture-aware AI coding agent and resilient multi-provider LLM routing fabric. NTG combines **Graphify** and **Codebase Memory MCP** for deep repository knowledge with **LiteLLM** smart routing across OpenRouter, Groq, NVIDIA NIM, Cohere, and Gemini.

---

## System Architecture

### 1. Architecture-Aware Coding Agent Workflow

```
USER CODING REQUEST
    ↓
CODE INTELLIGENCE (Graphify Knowledge Graph + Codebase Memory MCP Symbol/Call Graph)
    ↓
ARCHITECTURE-AWARE PLANNER (Builds structured prompt + routes via UnifiedNTGRouter [coding])
    ↓
HUMAN-IN-THE-LOOP APPROVAL GATE (awaiting_review → approved / rejected)
    ↓
SAFE CHANGE APPLICATION (Repository boundary enforcement + file tracking)
    ↓
IMPLEMENTATION VERIFIER (AST + py_compile syntax verification + shell-free test commands)
    ↓
KNOWLEDGE GRAPH SYNC (graphify update . + codebase-memory-mcp cli index_repository)
```

### 2. Multi-Provider Smart Routing Fabric

```
AGENT / USER PROMPT
    ↓
REQUEST REQUIREMENTS (tools, structured output, vision, streaming, coding, reasoning)
    ↓
NTG ELIGIBILITY FILTER (capability + quota + auth quarantine + circuit state + availability)
    ↓
ELIGIBLE DEPLOYMENT POOL (filtered by health, capabilities, quotas, and circuit state)
    ↓
LITELLM ROUTER (persistent router lifecycle, simple-shuffle load balancing, bounded retry, failover)
    ↓
UPSTREAM PROVIDER / ACCOUNT / MODEL
    ↓
RESPONSE
    ↓
NTG TELEMETRY & PERSISTENT STATE (.ntg/state.json)
```

LiteLLM is the **sole routing authority** (load balancing, retries, and failovers). NTG serves as the **code intelligence, planning, verification, eligibility, quota intelligence, circuit state, and telemetry layer** with zero duplicate routing loops.

---

## Key Features

### Architecture-Aware Coding Agent (`ntg/agent.py`, `ntg/code_intelligence.py`, `ntg/planner.py`, `ntg/verifier.py`)
1. **Dual Knowledge-Graph Intelligence (`CodeIntelligence`)**:
   - **Graphify (`graphify`)**: Queries community-clustered architecture graphs (`graphify query`), node explanations (`graphify explain`), shortest dependency paths (`graphify path`), and incremental code-graph updates (`graphify update .`).
   - **Codebase Memory MCP (`codebase-memory-mcp`)**: Queries indexed AST/call-graph symbols (`search_graph`), architectural overviews (`get_architecture`), call paths (`trace_path`), change impact (`detect_changes`), and code snippets (`get_code_snippet`).
2. **Human-in-the-Loop Change Planning (`planner`)**:
   - Combines Graphify and Codebase Memory context into structured planning prompts.
   - Generates reviewable plans in `awaiting_review` status (`approved=False`).
   - Enforces explicit approval (`approve_plan` / `require_approved_plan`) before any file writes can occur.
3. **Safe Execution & Verification (`CodeVerifier`)**:
   - Enforces strict repository-root path containment (blocks `..` path traversal and `.git` tampering).
   - Validates Python files via `ast.parse` and `py_compile`.
   - Executes verification commands with `shell=False` and rejects direct shell binaries.
4. **Automatic Knowledge Synchronization**:
   - Automatically refreshes both `graphify-out/graph.json` and the Codebase Memory MCP index after verified changes so codebase knowledge never drifts.

### Multi-Provider Smart Router (`ntg/router.py`)
1. **Single Routing Authority & Persistent Router**: LiteLLM owns deployment selection, distributed load balancing, and failover across requests using a long-lived Router lifecycle.
2. **Real Global `auto` Routing**: Requesting `auto` or `ntg-auto` balances across **all** healthy deployments from all configured providers.
3. **Capability-Aware Routing**: Deployments declare normalized capabilities (`coding`, `reasoning`, `vision`, `tool_calling`, `structured_output`, `streaming`, `context_window`).
4. **Resilient Circuit Breaker**:
   - `HEALTHY`: Serving traffic normally.
   - `OPEN`: Paused after failure or rate limits (authoritative `Retry-After` / reset duration).
   - `HALF_OPEN`: Controlled single-request probing (atomic reservation) before returning to `HEALTHY`.
   - `AUTH_FAILED`: Suspended on authentication failure (401/403) or missing model (404) errors.
5. **Accurate Quota & Error Handling**:
   - Parses HTTP `Retry-After` and `X-RateLimit-*` headers.
   - Distinguishes RPM, RPD, and account-wide vs. deployment-scoped limits.
6. **Thread-Safe Concurrency & Exact-Once Telemetry**: Zero shared mutable request state; attributes every attempt to the actual deployment via LiteLLM callback metadata.
7. **Local State Persistence**: Circuit breaker status, cooldown expirations, quota metadata, and discovery caches are atomically persisted to `.ntg/state.json`. API keys are never persisted or logged.

---

## Installation

```bash
pip install -r requirements.txt
```

Ensure the code intelligence CLI tools are available on your `PATH` for full graph capabilities:
- **Graphify**: `pip install graphifyy` (`graphify --help`)
- **Codebase Memory MCP**: `codebase-memory-mcp --version`

---

## Configuration

Place your provider keys in `.env` (keys are loaded dynamically and never committed):

```bash
# OpenRouter (up to 5 accounts)
OPENROUTER_API_KEY_1=...
OPENROUTER_API_KEY_2=...

# Groq (up to 5 accounts, dynamic model discovery)
Grok_API_KEY_1=...

# NVIDIA NIM (up to 4 accounts)
Nvida_API_KEY_1=...

# Cohere (up to 5 accounts)
Cohere_API_KEY_1=...

# Gemini (up to 4 accounts)
Gemini_API_KEY_1=...
```

---

## CLI Usage

### Interactive CLI

```bash
python main.py
```

### Architecture-Aware Coding Agent CLI Flags

```bash
# Generate an architecture-aware change plan using Graphify + Codebase Memory MCP
python main.py --plan "Add retry budget telemetry to UnifiedNTGRouter"

# Verify Python syntax (AST + bytecode compilation) across the repository
python main.py --verify

# Rebuild/synchronize Graphify knowledge graph and Codebase Memory MCP index
python main.py --refresh-knowledge
```

### Smart Router CLI Flags

```bash
# Global auto load-balanced routing
python main.py "Explain the Byzantine Generals Problem"

# Target specific logical provider
python main.py --model groq "Generate a summary"
python main.py --model nvidia "Optimize this algorithm"
python main.py --model cohere "Draft a documentation outline"
python main.py --model openrouter "Analyze trade-offs"

# Target specific capability
python main.py --capability coding "Write a Red-Black tree in Rust"
python main.py --capability vision "Describe this architecture diagram"
python main.py --capability reasoning "Solve this logic puzzle step by step"

# View deployment health & circuit states
python main.py --status

# Force live model rediscovery across provider accounts
python main.py --rediscover

# Reset persisted circuit breaker & metrics
python main.py --reset-state
```

---

## Programmatic Python API

### 1. Architecture-Aware Coding Agent (`ArchitectureAwareAgent`)

```python
from ntg import ArchitectureAwareAgent

agent = ArchitectureAwareAgent(repo_root=".")

# 1. Gather context from Graphify + Codebase Memory MCP and create a reviewable plan
plan = agent.create_plan(
    request="Add helper method to inspect active circuit breakers",
    symbol_pattern="UnifiedNTGRouter",
    auto_build_graph=True,
)
print(plan["plan"])  # Status: "awaiting_review", approved: False

# 2. Explicitly approve the plan after human review
agent.approve(plan)

# 3. Execute approved changes, run verification, and auto-sync Graphify + Codebase Memory MCP
result = agent.execute_plan(
    plan,
    file_changes={
        # "ntg/example.py": "...",
    },
    verification_commands=[
        ["python", "main.py", "--verify"],
    ],
    refresh_knowledge=True,
)
print("Verified:", result["passed"], "Knowledge synced:", plan.get("knowledge_synced"))
```

### 2. Direct Code Intelligence (`CodeIntelligence`)

```python
from ntg import CodeIntelligence

intel = CodeIntelligence(".")

# Query Graphify knowledge graph
print(intel.graphify_query("How does UnifiedNTGRouter handle failovers?"))
print(intel.graphify_explain("UnifiedNTGRouter"))

# Query Codebase Memory MCP symbol graph
project = intel.resolve_memory_project()
print(intel.memory_query(project, "UnifiedNTGRouter"))
print(intel.memory_architecture(project))

# Synchronize both knowledge stores after edits
intel.refresh_knowledge()
```

### 3. Multi-Provider Smart Router (`UnifiedNTGRouter`)

```python
from ntg import UnifiedNTGRouter, ModelCapabilities

router = UnifiedNTGRouter()

# 1. Standard global routing (simple-shuffle across all healthy providers)
response = router.ask("Hello world!")

# 2. Capability-aware routing
response = router.ask(
    "Implement binary search",
    capabilities=ModelCapabilities(coding=True),
)

# 3. Provider-specific routing
response = router.ask("Fast inference request", model="groq")

# 4. OpenAI / LiteLLM standard chat completion
response = router.completion(
    messages=[{"role": "user", "content": "Explain concurrency"}],
    model="auto",
    capabilities={"coding": True},
)
```

