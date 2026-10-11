# NTG: Architecture-Aware AI Coding Agent & Multi-Provider LiteLLM Smart Router

A production-grade, architecture-aware AI coding agent, evidence-grounded repository Q&A engine, and resilient multi-provider LLM routing fabric. NTG combines **deterministic AST/dependency analysis**, **Graphify** knowledge graphs, and **Codebase Memory MCP** symbol/call graphs with **LiteLLM** smart routing across **OpenRouter**, **Groq**, **NVIDIA NIM**, **Cohere**, and **Google Gemini**.

---

## Project Structure

The repository comprises **37 analyzed files** (**32 Python modules** and **5 configuration/documentation files**), plus local runtime state and graph index directories:

```text
NTG/
├── .env.example                 # Standardized UPPER_SNAKE_CASE multi-account API key template
├── .gitignore                   # Git exclusion rules (.env, .ntg/, graphify-out/, caches)
├── pyproject.toml               # PEP 621 package metadata, dependencies, and `ntg` CLI entrypoint
├── requirements.txt             # Pinned runtime Python dependencies (litellm, google-genai, requests)
├── README.md                    # Comprehensive project & architecture documentation
├── main.py                      # Root CLI entrypoint script (`python main.py`)
├── test.py                      # Repository query & structured verification harness (`query_directory_structured`)
├── .ntg/                        # Local runtime persistence (git-ignored)
│   ├── state.json               # Persisted circuit breaker states, cooldowns, quotas, metrics & discovery cache
│   └── index_manifest.json      # SHA-256 content hashes & AST symbol signatures for Graphify and Codebase Memory
├── graphify-out/                # Graphify knowledge graph artifacts
│   ├── graph.json               # Community-clustered AST & structural knowledge graph
│   ├── GRAPH_REPORT.md          # Generated architectural graph report
│   └── .graphify_root           # Canonical repository-root identity marker
└── ntg/                         # Main Python package
    ├── __init__.py              # Unified public package API facade (re-exports 90+ symbols)
    ├── __main__.py              # Module execution entrypoint (`python -m ntg`)
    ├── agent.py                 # Top-level compatibility shim re-exporting canonical `ntg.agent` symbols
    ├── verifier.py              # Verification helpers (`run_checked`, `inspect_diff`, `refresh_graphify`) & compatibility exports
    ├── core/                    # Core configuration, domain models, error classification & shared utilities
    │   ├── __init__.py          # Core package exports
    │   ├── config.py            # Zero-dependency .env loader, logical model groups, retry & circuit constants, provider keys
    │   ├── exceptions.py        # Provider error extraction, classification (`ParsedErrorInfo`), and 5-tier rate-limit reset resolution
    │   ├── models.py            # `Deployment`, `CircuitState`, `ModelCapabilities`, `QuotaInfo`, `QuotaScope`, `DeploymentMetrics`
    │   └── utils.py             # Duration/reset header parsing (`parse_reset_header`), UTC timestamps, and secret redaction
    ├── providers/               # Provider-specific model discovery, capability inference & deployment builders
    │   ├── __init__.py          # Provider package exports
    │   ├── base.py              # `DiscoveryErrorCategory`, discovery error classifier, and persisted state restoration
    │   ├── cohere.py            # Cohere capability inference and multi-account deployment builder (`north-mini-code-1-0`)
    │   ├── discovery.py         # Unified multi-provider deployment orchestrator (`build_all_deployments`)
    │   ├── gemini.py            # Gemini GenAI SDK dynamic model discovery, content-generation filter & deployment builder
    │   ├── groq.py              # Groq `/openai/v1/models` live discovery, cache fallback & deployment builder
    │   ├── nvidia.py            # NVIDIA NIM capability inference and deployment builder (`nvidia/nemotron-3.5-lightning-30b-a3b`)
    │   └── openrouter.py        # OpenRouter `/api/v1/models` metadata capability inference & deployment builder (`nvidia/nemotron-3-ultra-550b-a55b:free`)
    ├── router/                  # Smart routing engine, request requirement extraction, persistent state & telemetry
    │   ├── __init__.py          # Router package exports
    │   ├── engine.py            # `UnifiedNTGRouter` (`OpenRouterFallbackRouter` alias) multi-provider routing engine
    │   ├── requirements.py      # `RequestRequirements`, `extract_request_requirements`, `parse_capabilities`, `NoEligibleDeploymentsError`
    │   ├── state.py             # `StateManager` atomic persistence, secret scrubbing (`clean_for_persistence`) & cooldown recovery
    │   └── telemetry.py         # `NTGTelemetryLogger` (`CustomLogger`) & `ActiveRequest` thread-safe per-attempt telemetry
    ├── agent/                   # Architecture-aware AI coding agent, triple-source intelligence, planning & verification
    │   ├── __init__.py          # Agent package exports
    │   ├── code_intelligence.py # `CodeIntelligence` (AST repository inventory, Graphify CLI, Codebase Memory MCP, SHA-256/AST sync)
    │   ├── orchestrator.py      # `ArchitectureAwareAgent` (`CodingAgent` alias), `query_directory`, `query_directory_structured`
    │   ├── planner.py           # Query requirement analysis, prompt builders, deterministic answer validation/repair & plan approval gate
    │   └── verifier.py          # `CodeVerifier`, `verify_python_syntax`, `run_verification_command`, `inspect_git_diff`
    └── cli/                     # Command-line interface & console diagnostics
        ├── __init__.py          # CLI package exports
        ├── app.py               # `run_cli` (`main` alias) CLI argument parser and command dispatcher
        └── diagnostics.py       # Console banners, deployment status tables, execution summaries & rate-limit reports
```

---

## System Architecture

### 1. Evidence-Grounded Repository Q&A Pipeline (`agent.ask` / `query_directory`)

```text
USER REPOSITORY QUESTION (CLI `-d <path>` or `query_directory_structured`)
    ↓
TRIPLE-SOURCE CODE INTELLIGENCE (`CodeIntelligence.gather_context`)
  ├─ Deterministic AST Inventory (37 files, classes, functions, signatures, calls, raises, imports, reading order)
  ├─ Graphify Knowledge Graph (`graphify-out/graph.json` — verified via `.graphify_root`, SHA-256 & live AST symbols)
  └─ Codebase Memory MCP (`get_architecture` + `search_graph` — verified via canonical `root_path` & SHA-256 manifest)
    ↓
PRE-GENERATION VERIFICATION GATE (`verify_retrieval_context`)
  └─ Strict mode blocks LLM invocation if any knowledge source is stale, contaminated, or unverified
    ↓
DYNAMIC QUERY REQUIREMENT ANALYSIS (`analyze_query_requirements` + `build_query_prompt`)
  └─ Resolves scope (`repo_wide` vs `targeted`), target files, queried symbols, and requested deliverables
    ↓
SMART ROUTING (`UnifiedNTGRouter.completion`)
    ↓
POST-GENERATION VALIDATION & DETERMINISTIC REPAIR (`validate_query_answer` + `finalize_grounded_answer`)
  ├─ Checks omitted files, hallucinated paths, ungrounded descriptions, unsupported symbol/call/env claims, and reading-order precedence
  ├─ Deterministically repairs omitted target files and reading-order sequence when verified AST evidence is complete
  └─ Appends transparent Verification & Coverage Diagnostics and Epistemic Scope disclosures
```

### 2. Architecture-Aware Coding Agent Workflow (`create_plan` → `approve` → `execute_plan`)

```text
USER CODING REQUEST (`--plan` or `agent.create_plan`)
    ↓
TRIPLE-SOURCE CODE INTELLIGENCE + PRE-GENERATION VERIFICATION GATE
    ↓
ARCHITECTURE-AWARE PLANNER (`build_planning_prompt` + `build_change_plan`)
    ↓
HUMAN-IN-THE-LOOP APPROVAL GATE (`status="awaiting_review"`, `approved=False` → `approve_plan` / `reject_plan`)
    ↓
SAFE CHANGE APPLICATION (`apply_changes` — repository root boundary enforcement + blocks `.git` tampering)
    ↓
IMPLEMENTATION VERIFIER (`CodeVerifier.verify_plan` — `ast.parse` + `py_compile` + shell-free verification commands)
    ↓
AUTOMATIC KNOWLEDGE SYNC (`refresh_knowledge` — `graphify update .` + AST sync + `codebase-memory-mcp` re-indexing)
```

### 3. Multi-Provider Smart Routing Fabric (`UnifiedNTGRouter`)

```text
AGENT / USER PROMPT
    ↓
REQUEST REQUIREMENTS (`extract_request_requirements`: tools, structured_output, vision, streaming, coding, reasoning, min_context)
    ↓
NTG ELIGIBILITY FILTER (verified protocol capabilities + quota + auth quarantine + circuit state)
    ↓
DETERMINISTIC POLICY (capability fit + quota headroom + health + latency + account distribution)
    ↓
NTG ATTEMPT LOOP (dispatch-time probe/quota reservation, account/provider failover, bounded extra retries [default=1, max=2])
    ↓
UPSTREAM PROVIDER / ACCOUNT / MODEL (OpenRouter, Groq, NVIDIA NIM, Cohere, Gemini)
    ↓
SYNCHRONOUS ATTEMPT OUTCOMES & COHERENT ATOMIC PERSISTENCE (`.ntg/state.json`)
```

NTG owns **deterministic ranking, bounded retry/failover, scoped quota reservations, circuit recovery, and secret-scrubbed persistence**. LiteLLM provides the public provider transport. See [the routing audit and design](docs/smart-routing.md) for the scoring formula, state ownership, compatibility details, and operational limits.

---

## Key Features

### 1. Triple-Source Codebase Intelligence (`ntg/agent/code_intelligence.py`)
- **Deterministic AST & Dependency Inventory (`build_repository_inventory`)**:
  - Walks the repository while excluding generated/cache directories (`.git`, `__pycache__`, `.ntg`, `graphify-out`, virtual environments) and sensitive files (`.env`, `credentials.json`).
  - Parses all Python modules via `ast` to extract docstrings, classes, methods, functions, parameter/return signatures, raised exceptions, non-trivial call targets, `__all__` exports, constants, internal repository dependencies, and external packages.
  - Computes **cross-module AST call-graph edges** (`call_graph_edges` — including transitive resolution through `__init__.py` re-export facades) and **multi-hop runtime execution flows** (`execution_flows`) starting from Layer 1 entry points.
  - Computes a **topological, dependency-aware reading order** across **8 architectural layers (Layers 0–7)** where internal dependencies are guaranteed to precede their dependents.
- **Graphify Knowledge Graph Integration (`graphify`)**:
  - Executes `graphify query`, `graphify explain`, `graphify path`, `graphify update`, and `graphify save-result`.
  - Validates graph integrity (`inspect_graphify_status`) against `.graphify_root`, deleted files, missing Python files, live AST top-level class/function symbol sets (`ast_symbols_verified`), and SHA-256 content hashes in `.ntg/index_manifest.json`.
  - Includes a deterministic AST graph synchronizer (`_sync_graphify_python_ast`) that purges stale/deleted nodes and rebuilds accurate module/class/function/method/import nodes and edges.
- **Codebase Memory MCP Integration (`codebase-memory-mcp`)**:
  - Executes `list_projects`, `index_repository`, `search_graph`, `get_architecture`, `trace_path`, `detect_changes`, and `get_code_snippet` via CLI with native Windows `.exe` resolution.
  - Enforces strict repository root matching (`resolve_memory_project`) so an unrelated indexed project is never silently queried.
  - Verifies index freshness (`inspect_memory_status`) via indexed file coverage, line-count drift detection, and SHA-256 + AST symbol fingerprints in `.ntg/index_manifest.json`.

### 2. Evidence-Grounded Q&A, Planning & Verification (`ntg/agent/planner.py`, `ntg/agent/verifier.py`)
- **Strict Retrieval Gate (`verify_retrieval_context`)**: Refuses to call the LLM when any required index or inventory check fails (`strict=True` by default).
- **Deterministic Answer Validator (`validate_query_answer`)**:
  - Verifies 100% target file coverage (`omitted_files`) and flags any non-existent `.py` paths (`hallucinated_files`).
  - Validates every file description against AST symbol, call, exception, dependency, and docstring anchor tokens (`ungrounded_descriptions`).
  - Detects cross-file symbol ownership misattribution, contradicted import/call claims, and invented `.env` variable names (`unsupported_claims`).
  - Verifies reading-order uniqueness and topological dependency precedence across implementation modules (`ordering_violations`).
- **Grounded Structural Repair (`finalize_grounded_answer`)**: When the verified inventory holds complete evidence and the LLM output has no hallucinated paths or contradictory claims, deterministically supplements omitted files at their exact topological step positions and re-verifies completeness.
- **Human-in-the-Loop Change Planning & Safe Verification (`ArchitectureAwareAgent`, `CodeVerifier`)**:
  - Plans start in `status="awaiting_review"` (`approved=False`). File modifications (`apply_changes`, `execute_plan`) raise `PermissionError` until `approve_plan(plan)` is called.
  - Blocks path traversal outside `repo_root` and refuses writes inside `.git`.
  - Validates Python syntax via both `ast.parse` and `py_compile.compile`, and runs verification commands with `shell=False`, `stdin=subprocess.DEVNULL`, and a strict ban on shell binaries (`sh`, `bash`, `cmd`, `powershell`, `pwsh`).

### 3. Multi-Provider Smart Router (`ntg/router/`, `ntg/providers/`, `ntg/core/`)
- **5 Providers & Multi-Account Pool (`ntg/core/config.py`, `ntg/providers/`)**:
  - **OpenRouter**: Up to 5 accounts (`OPENROUTER_API_KEY_1..5`), default model `nvidia/nemotron-3-ultra-550b-a55b:free`, live capability inference from `https://openrouter.ai/api/v1/models`.
  - **Groq**: Up to 5 accounts (`GROQ_API_KEY_1..5`), dynamic chat-model discovery via `https://api.groq.com/openai/v1/models` (excluding Whisper/TTS/moderation/embedding models) with persistent last-known-good cache fallback.
  - **NVIDIA NIM**: Up to 4 accounts (`NVIDIA_API_KEY_1..4`), model `nvidia/nemotron-3.5-lightning-30b-a3b` via `https://integrate.api.nvidia.com/v1`.
  - **Cohere**: Up to 5 accounts (`COHERE_API_KEY_1..5`), model `north-mini-code-1-0` (`128,000` token context window).
  - **Google Gemini**: Up to 4 accounts (`GEMINI_API_KEY_1..4`), dynamic discovery via `google-genai` SDK (`client.models.list()`), filtering for `generateContent` text models (`supports_generate_content`).
- **Capability-Aware Request Extraction (`ntg/router/requirements.py`)**:
  - Automatically extracts requirements for `tool_calling` (`tools`/`functions`), `structured_output` (`response_format`/`schema`/`json_schema`), `vision` (typed image content), audio input/output, `streaming`, `reasoning`, `coding`, and `context_window`.
- **Circuit Breaker & Atomic `HALF_OPEN` Probing (`ntg/core/models.py`)**:
  - States: `HEALTHY` → `OPEN` → `HALF_OPEN` → `HEALTHY`, plus `QUARANTINED` (model 404) and `AUTH_FAILED` (account 401/403).
  - `HALF_OPEN` state enforces atomic single-request probe reservation (`claim_half_open_probe` / `release_half_open_probe`); probe failure doubles the cooldown backoff, while probe success restores `HEALTHY`.
  - Automatic key-rotation detection: if a quarantined `AUTH_FAILED` deployment's API key hash changes on refresh, quarantine is automatically lifted.
- **5-Tier Rate-Limit Reset Resolution & Quota Scopes (`ntg/core/exceptions.py`)**:
  - Resolves cooldowns using strict priority: (1) `Retry-After` header, (2) `X-RateLimit-Reset` / `RateLimit-Reset`, (3) provider-specific reset headers, (4) structured metadata / error duration strings (`6m0s`, `250ms`, ISO-8601, RFC 2822), and (5) conservative fallback (UTC midnight for daily quotas, `60s` for short-term limits).
  - Propagates `QuotaScope.ACCOUNT` (OpenRouter, Cohere, NVIDIA) and `QuotaScope.PROVIDER` cooldowns or `AUTH_FAILED` quarantines across all sibling deployments sharing the affected account.
- **Zero-Secret Atomic State Persistence (`ntg/router/state.py`)**:
  - Atomically writes coherent `.ntg/state.json` snapshots via flushed PID/thread-scoped temporary file replacement while recursively scrubbing keys matching `api_key`, `token`, `secret`, `authorization`, `headers`, `messages`, or `prompt`, and redacting `Bearer` / `sk-` tokens via `sanitize_secret`.

---

## Dependency-Aware Repository Reading Order

NTG's deterministic analyzer organizes all **37 repository files** into **8 architectural layers** so that every internal dependency is read before the modules that import it:

1. **Layer 0 — Project Overview & Configuration** (`README.md`, `pyproject.toml`, `requirements.txt`, `.env.example`, `.gitignore`)
2. **Layer 1 — Execution Entry Points & Top-Level Facade** (`main.py`, `ntg/__main__.py`, `test.py`, `ntg/__init__.py`)
3. **Layer 2 — Core Foundation** (`ntg/core/config.py`, `ntg/core/utils.py`, `ntg/core/models.py`, `ntg/core/exceptions.py`, `ntg/core/__init__.py`)
4. **Layer 3 — Provider Adapters & Model Discovery** (`ntg/providers/base.py`, `ntg/providers/cohere.py`, `ntg/providers/gemini.py`, `ntg/providers/groq.py`, `ntg/providers/nvidia.py`, `ntg/providers/openrouter.py`, `ntg/providers/discovery.py`, `ntg/providers/__init__.py`)
5. **Layer 4 — Smart Routing, Circuit-Breaker State, Telemetry & Runtime Diagnostics** (`ntg/cli/diagnostics.py`, `ntg/router/requirements.py`, `ntg/router/state.py`, `ntg/router/telemetry.py`, `ntg/router/engine.py`, `ntg/router/__init__.py`)
6. **Layer 5 — Architecture-Aware Coding Agent & Code Intelligence** (`ntg/agent/code_intelligence.py`, `ntg/agent/planner.py`, `ntg/agent/verifier.py`, `ntg/agent/orchestrator.py`, `ntg/agent/__init__.py`)
7. **Layer 6 — CLI Application Runner** (`ntg/cli/app.py`, `ntg/cli/__init__.py`)
8. **Layer 7 — Top-Level Compatibility Shims** (`ntg/agent.py`, `ntg/verifier.py`)

---

## Installation

Requires **Python >= 3.10**.

```bash
pip install -r requirements.txt
# Or install as an editable package with the `ntg` CLI script entrypoint:
pip install -e .
```

Ensure the external code intelligence CLI tools are installed and available on your `PATH` (or standard Windows install paths) for full triple-source verification:
- **Graphify**: `pip install graphifyy` (`graphify --help`)
- **Codebase Memory MCP**: `codebase-memory-mcp --version`

---

## Configuration

Copy `.env.example` to `.env` and populate your provider API keys (keys are loaded dynamically on startup and never committed or persisted to `.ntg/state.json`):

```bash
# OpenRouter (up to 5 accounts)
OPENROUTER_API_KEY_1=your_openrouter_api_key_1_here
OPENROUTER_API_KEY_2=your_openrouter_api_key_2_here

# Groq (up to 5 accounts, dynamic model discovery)
GROQ_API_KEY_1=your_groq_api_key_1_here

# NVIDIA NIM (up to 4 accounts)
NVIDIA_API_KEY_1=your_nvidia_api_key_1_here

# Cohere (up to 5 accounts)
COHERE_API_KEY_1=your_cohere_api_key_1_here

# Google Gemini (up to 4 accounts, dynamic SDK discovery)
GEMINI_API_KEY_1=your_gemini_api_key_1_here

# Optional Circuit Breaker Tuning
NTG_CIRCUIT_BREAKER_MAX_FAILURES=3
NTG_CIRCUIT_BREAKER_RECOVERY_TIME=30.0
```

---

## CLI Usage

You can invoke NTG via `python main.py`, `python -m ntg`, or the installed `ntg` console script.

### 1. Evidence-Grounded Repository Q&A & Coding Agent CLI

```bash
# Query a repository/directory with strict triple-source verification and reading-order grounding
python main.py -d . "Provide the sequence of files I should read to understand this project, with a brief description of each file."

# Run the standalone structured verification harness script
python test.py

# Generate an architecture-aware change plan (gated on human approval)
python main.py -d . --plan "Add retry budget telemetry to UnifiedNTGRouter"

# Run in best-effort (non-strict) mode when external indexes are unavailable
python main.py -d . --non-strict "Summarize the router package"

# Verify Python syntax (AST + bytecode compilation) across the repository
python main.py --verify
python main.py -d /path/to/repo --verify

# Rebuild and synchronize Graphify knowledge graph and Codebase Memory MCP index
python main.py --refresh-knowledge
python main.py -d /path/to/repo --refresh-knowledge
```

### 2. Multi-Provider Smart Router CLI

```bash
# Interactive prompt mode
python main.py

# Global auto load-balanced routing across all healthy providers
python main.py "Explain the Byzantine Generals Problem"

# Route to a specific logical provider group
python main.py --model groq "Generate a concise summary"
python main.py --model nvidia "Optimize this matrix multiplication routine"
python main.py --model cohere "Draft a module specification"
python main.py --model openrouter "Compare consensus algorithms"
python main.py --model gemini "Summarize key architectural trade-offs"

# Filter by required model capabilities (comma-separated)
python main.py --capability coding "Write a Red-Black tree in Rust"
python main.py --capabilities coding,reasoning "Design a lock-free ring buffer"

# Display detailed deployment health, circuit states, capabilities, telemetry & quotas
python main.py --status

# Force live model rediscovery across Groq and Gemini accounts
python main.py --rediscover

# Reset persisted routing state, circuit breakers, and telemetry metrics
python main.py --reset-state
```

---

## Programmatic Python API

### 1. Evidence-Grounded Repository Q&A (`query_directory_structured` & `query_directory`)

```python
from ntg import query_directory, query_directory_structured

# 1. Structured result with full verification, coverage, and retrieval diagnostics
result = query_directory_structured(
    directory=".",
    query="Provide the sequence of files I should read to understand the project, with a brief description of each file.",
    model="auto",
    capabilities=["coding"],
    auto_build_graph=True,
    strict=True,
    include_diagnostics=True,
)
print("Status:", result["status"], "| Complete:", result["complete"])
print(result["answer"])

# 2. Direct string return for callers that only need the finalized markdown text
answer_text = query_directory(
    directory=".",
    query="How does UnifiedNTGRouter handle rate-limit cooldowns?",
    return_structured=False,
)
```

### 2. Architecture-Aware Coding Agent (`ArchitectureAwareAgent`)

```python
from ntg import ArchitectureAwareAgent

agent = ArchitectureAwareAgent(repo_root=".")

# 1. Gather triple-source context and generate a reviewable change plan
plan = agent.create_plan(
    request="Add helper method to inspect active circuit breakers",
    symbol_pattern="UnifiedNTGRouter",
    auto_build_graph=True,
    strict=True,
)
print(plan["status"], plan["approved"])  # "awaiting_review", False
print(plan["plan"])

# 2. Explicitly approve the plan after human review (or call agent.reject(plan, reason="..."))
agent.approve(plan)

# 3. Apply changes safely inside repo_root, run syntax & command verification, and sync graphs
verification = agent.execute_plan(
    plan,
    file_changes={
        # "ntg/router/engine.py": "...",
    },
    verification_commands=[
        ["python", "main.py", "--verify"],
    ],
    refresh_knowledge=True,
)
print("Passed:", verification["passed"], "| Plan status:", plan["status"])
```

### 3. Direct Code Intelligence (`CodeIntelligence`)

```python
from ntg import CodeIntelligence

intel = CodeIntelligence(".")

# Build deterministic AST inventory, dependency reading order, and runtime execution flows
inventory = intel.build_repository_inventory()
print(f"Analyzed {inventory['total_files']} files ({inventory['python_files_count']} Python modules)")

# Inspect and ensure Graphify & Codebase Memory MCP freshness
print(intel.inspect_graphify_status())
project = intel.resolve_memory_project(auto_index=True)
print(intel.inspect_memory_status(project))

# Query Graphify and Codebase Memory MCP directly
print(intel.graphify_query("How does UnifiedNTGRouter handle failovers?"))
print(intel.memory_architecture(project))
print(intel.memory_query(project, "UnifiedNTGRouter"))

# Synchronize both knowledge stores and update `.ntg/index_manifest.json`
intel.refresh_knowledge(project=project)
```

### 4. Multi-Provider Smart Router (`UnifiedNTGRouter`)

```python
from ntg import ModelCapabilities, UnifiedNTGRouter

router = UnifiedNTGRouter()

# 1. Global load-balanced routing (`auto` / `ntg-auto`)
response = router.ask("Hello world!")

# 2. Capability-aware and minimum-context routing
response = router.ask(
    "Implement a thread-safe LRU cache in Python",
    capabilities=ModelCapabilities(coding=True, streaming=True),
    min_context=32000,
)

# 3. Provider-specific routing
response = router.ask("Fast inference request", model="groq")

# 4. Standard OpenAI / LiteLLM chat completion interface
response = router.completion(
    messages=[{"role": "user", "content": "Explain circuit breaker state machines"}],
    model="auto",
    capabilities={"coding": True},
)
```
