---
goal: Rebuild the existing NTG router with response-driven free-use routing
date_created: 2026-10-13
status: Completed
---
# Introduction

User approved the design in chat. Keep the existing router path and class.

## 1. Requirements & Constraints

- **REQ-001**: Fair async routing, one deadline, bounded fallback and no persistent quota accounting.
- **SEC-001**: Ignore credentials and exclude unknown billing eligibility.
- **CON-001**: No test files or live inference probes.

## 2. Implementation Steps

| Task | Description | Completed | Date |
|---|---|---|---|
| TASK-001 | Load numbered credentials in ntg/router/config.py and migrate .env | yes | 2026-10-13 |
| TASK-002 | Replace engine.py and state.py with fair selection and scoped operational blocks | yes | 2026-10-13 |
| TASK-003 | Implement transport.py, ntg/__main__.py and README.md | yes | 2026-10-13 |
| TASK-004 | Run temporary distribution/fallback/stream simulations and syntax checks | yes | 2026-10-13 |

## 3. Alternatives

- **ALT-001**: Retaining the old persistent quota state violates the requested constraints.

## 4. Dependencies

- **DEP-001**: LiteLLM 1.98.0, python-dotenv and httpx.

## 5. Files

- **FILE-001**: ntg/router/{config,state,transport,engine}.py and ntg/__main__.py.

## 6. Testing

- **TEST-001**: Temporary inline simulations exercise distribution, scopes, retry budgets and streaming.

## 7. Risks & Assumptions

- **RISK-001**: Provider APIs do not universally attest billing plan; explicit verified configuration is required.

## 8. Related Specifications / Further Reading

See README.md for official provider references and the approved chat design.
