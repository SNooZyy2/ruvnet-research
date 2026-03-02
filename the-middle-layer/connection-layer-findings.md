# The Middle Layer: Connection Layer Discovery

**Date**: 2026-03-01
**Context**: After 131 research sessions and 1,631 DEEP file reads (~10.5% of 15,559 total files), we discovered a systematic blind spot: the priority queue (`smart_priority_gaps`) optimizes for algorithmic component files (high LOC, many dependencies) and systematically deprioritizes the integration/wiring layer — the files that answer "how does this actually connect and run?"

This document catalogs all discovered connection-layer files across all repos and prioritizes them for reading.

---

## The Problem

Our research system ranked files by proximity to already-understood code, LOC, and dependency connections. This biased toward:
- Large Rust algorithm files (temporal-tensor, ruQu, HNSW, MinCut kernels)
- TypeScript library modules (backends, controllers, engines)

And biased against:
- CLI entrypoints (small, few connections to algorithm files)
- MCP server bootstraps (different directory tree)
- Factory/registry files (low LOC, high fan-in)
- Docker/deployment configs (not in source trees)
- CI workflows (not tracked in research DB)

**Result**: We deeply understand the components but have never read the code that wires them together at runtime.

---

## Search Methodology

Three parallel searches were conducted:

1. **Pass 1 — Structural Entrypoints**: package.json `bin` fields, Dockerfiles, docker-compose, Makefiles, `build.rs`, CI workflows, launch/start/deploy scripts, examples/demos
2. **Pass 2 — Multi-Import Hubs**: Files importing from 3+ internal packages, factories, registries, routers, dispatchers, MCP tool registration hubs
3. **Pass 3 — DB Cross-Reference**: Queried research DB for integration-shaped filenames at NOT_TOUCHED/SURFACE depth, identified ghost DEEP files (marked DEEP, 0 lines read), counted remaining unread files per package

### Yield
- ~75 high-value integration files identified
- 507 integration-shaped source files remain NOT_TOUCHED across all packages
- 7,105 total NOT_TOUCHED files (~90% of the codebase)
- 8+ files marked DEEP in DB with 0 lines actually read (metadata inflation)

---

## Prioritized File List

### TIER 1: What Actually Runs When a User Types a Command

These files ARE the product. Everything else is a library.

| # | File | LOC | DB Status | Why It Matters |
|---|------|-----|-----------|---------------|
| 1 | `ruvector/npm/packages/ruvector/bin/cli.js` | 7,357 | NOT_TOUCHED | **THE ruvector CLI.** What `npx ruvector` executes. 7K LOC = not a stub. |
| 2 | `ruvector/npm/packages/ruvector/bin/mcp-server.js` | 3,007 | NOT_TOUCHED | **THE ruvector MCP server.** How MCP tools reach ruvector at runtime. |
| 3 | `claude-flow/v3/@claude-flow/cli/bin/cli.js` | 156 | NOT_TOUCHED | **THE claude-flow v3 entrypoint.** What runs when you type `claude-flow`. |
| 4 | `claude-flow/v3/@claude-flow/cli/bin/mcp-server.js` | 189 | NOT_TOUCHED | **THE claude-flow MCP server entrypoint.** |
| 5 | `claude-flow/ruflo/bin/ruflo.js` | 50 | NOT_TOUCHED | **The ruflo rebrand.** Where does it point? |
| 6 | `ruvector/npm/packages/rvlite/bin/cli.js` | 1,686 | NOT_TOUCHED | rvlite CLI — lightweight vector ops. |
| 7 | `ruvector/npm/packages/ruvllm/bin/cli.js` | 1,005 | NOT_TOUCHED | ruvllm CLI — LLM integration. |

**Subtotal: ~13,450 LOC across 7 files**

---

### TIER 2: Where Subsystems Actually Wire Together

These files decide at runtime which backend gets used, which controllers initialize, which code path executes.

| # | File | LOC | DB Status | Why It Matters |
|---|------|-----|-----------|---------------|
| 8 | `agentdb/src/mcp/agentdb-mcp-server.ts` | 2,367 | DEEP (0 lines read!) | **THE AgentDB MCP hub.** 10 controllers wired to MCP protocol. Never actually read. |
| 9 | `claude-flow/v3/@claude-flow/memory/src/controller-registry.ts` | 1,026 | NOT_TOUCHED | **29 controllers** lifecycle-managed. Most comprehensive registry found. |
| 10 | `claude-flow/v3/@claude-flow/memory/src/agentdb-adapter.ts` | 1,038 | NOT_TOUCHED | V3 AgentDB adapter — how claude-flow talks to AgentDB. |
| 11 | `claude-flow/v3/@claude-flow/memory/src/memory-bridge.ts` | 1,773 | NOT_TOUCHED | V3 memory bridge — could be the missing TS-to-Rust connector. |
| 12 | `claude-flow/v3/@claude-flow/memory/src/memory-initializer.ts` | 2,564 | NOT_TOUCHED | What bootstraps the entire memory system at startup. |
| 13 | `ruvllm/src/claude_flow/hnsw_router.rs` | 1,287 | DEEP (0 lines read!) | Semantic HNSW task router. ruvector-core + SONA integration. |
| 14 | `ruvllm/src/claude_flow/claude_integration.rs` | 1,341 | DEEP (0 lines read!) | Primary Rust-side claude-flow integration point. |
| 15 | `ruvllm/src/claude_flow/model_router.rs` | 1,322 | DEEP (0 lines read!) | Model routing — the Rust side of ADR-008. |
| 16 | `agentic-flow/src/services/sona-agentdb-integration.ts` | 463 | NOT_TOUCHED | **Explicit cross-repo bridge**: @ruvector/sona + agentdb. |
| 17 | `ruvector/npm/packages/ruvector/src/core/intelligence-engine.ts` | 1,233 | Partial (R117) | The "brain" — agentdb + sona + onnx + parallel-intelligence. |
| 18 | `agentdb/src/backends/factory.ts` | 344 | Unknown | **Backend selector**: RuVector native > RuVector WASM > RVF > HNSWLib auto-detect. |
| 19 | `ruvllm/src/ruvector_integration.rs` | 1,099 | DEEP | Primary Rust multi-crate integration (ruvector-core + sona + claude_flow). Already read. |
| 20 | `ruvector/npm/packages/ruvector/src/core/index.ts` | 56 | NOT_TOUCHED | Master barrel export — 20+ subsystems exposed through one import. |

**Subtotal: ~15,913 LOC across 13 files (excluding already-read #19)**

---

### TIER 3: MCP Tool Registration (Where Features Become Callable)

These files map internal code to the tools users actually invoke via MCP protocol.

| # | File | LOC | DB Status | Why It Matters |
|---|------|-----|-----------|---------------|
| 21 | `claude-flow/v3/mcp/tools/index.ts` | 445 | NOT_TOUCHED | **256 tools** aggregated from 12 tool groups via `getAllTools()`. |
| 22 | `claude-flow/v3/mcp/server.ts` | 792 | NOT_TOUCHED | V3 MCP server bootstrap — ToolRegistry + SessionManager + ConnectionPool. |
| 23 | `claude-flow/v3/@claude-flow/mcp/src/server.ts` | 1,134 | NOT_TOUCHED | Library-level MCP server with 9 sub-registries including SamplingManager. |
| 24 | `claude-flow/v3/@claude-flow/cli/src/commands/index.ts` | 398 | NOT_TOUCHED | 30+ CLI commands lazy-loaded — the full V3 command surface. |
| 25 | `agentic-flow/src/mcp/standalone-stdio.ts` | 812 | NOT_TOUCHED | agentic-flow MCP server via FastMCP — 66+ agent types registered. |
| 26 | `sublinear-time-solver/src/mcp/server.ts` | 1,327 | NOT_TOUCHED | Sublinear MCP server — 10+ tool modules wired together. |
| 27 | `claude-flow/v2/src/mcp/server.ts` | 646 | NOT_TOUCHED | V2 MCP server — integrates claude-flow-tools + swarm-tools + ruv-swarm-tools. |

**Subtotal: ~5,554 LOC across 7 files**

---

### TIER 4: Deployment / CI Ground Truth

If these run in CI, the code works at least enough to pass pipeline checks.

| # | File | LOC | DB Status | Why It Matters |
|---|------|-----|-----------|---------------|
| 28 | `ruvector/.github/workflows/release.yml` | 621 | NOT_TOUCHED | What gets built and published to npm. |
| 29 | `ruvector/.github/workflows/publish-all.yml` | 552 | NOT_TOUCHED | Multi-package publish — shows which packages are real. |
| 30 | `claude-flow/.github/workflows/ci.yml` | 228 | NOT_TOUCHED | What tests actually run in claude-flow CI. |
| 31 | `claude-flow/.github/workflows/v3-ci.yml` | 157 | NOT_TOUCHED | V3-specific CI pipeline. |
| 32 | `ruvector/tests/integration/distributed/docker-compose.yml` | — | NOT_TOUCHED | Distributed integration test setup. |
| 33 | `agentic-flow/agentic-flow/deployment/docker-compose.yml` | — | NOT_TOUCHED | Production deployment config. |
| 34 | `ruvector/crates/ruvllm/tests/e2e_integration_test.rs` | 1,536 | NOT_TOUCHED | ruvllm end-to-end integration tests. |
| 35 | `ruvector/crates/prime-radiant/tests/ruvllm_integration_tests.rs` | 1,394 | NOT_TOUCHED | Prime-radiant ↔ ruvllm integration tests. |

**Subtotal: ~4,488 LOC across 8 files**

---

### TIER 5: Ghost DEEP Files (Metadata Inflation)

These files are marked DEEP in the research DB but have **0 lines actually read**. DEEP file count (1,631) may be inflated.

| File | LOC | Lines Read | Impact |
|------|-----|-----------|--------|
| `ruvllm/src/claude_flow/claude_integration.rs` | 1,341 | 0 | Overstates ruvllm coverage |
| `ruvllm/src/claude_flow/hnsw_router.rs` | 1,288 | 0 | Overstates ruvllm coverage |
| `ruvllm/src/claude_flow/model_router.rs` | 1,322 | 0 | Overstates ruvllm coverage |
| `agentdb/src/mcp/agentdb-mcp-server.ts` | 2,368 | 0 | Overstates agentdb coverage |
| `agentic-flow/src/agentdb/cli/agentdb-cli.ts` | 862 | 0 | Overstates agentic-flow coverage |
| `agentic-flow/src/cli/commands/hooks.ts` | 1,149 | 0 | Overstates agentic-flow coverage |
| `agentic-flow/src/proxy/anthropic-to-openrouter.ts` | 775 | 0 | Overstates agentic-flow coverage |
| `src/browser/HNSWIndex.ts` | 495 | 0 | Overstates browser coverage |
| `ruvector-core/src/index/hnsw.rs` | 482 | 0 | Overstates core coverage |

**Total phantom DEEP: ~10,082 LOC across 9 files**

---

## Additional Context from Pass 1

### Infrastructure Scale (suggests real deployment, not just marketing)

| Category | Count |
|----------|-------|
| CLI entrypoints (package.json bin fields) | ~35 distinct commands |
| Rust src/bin/ executables | ~50+ binary targets |
| Dockerfiles | ~110 files |
| docker-compose files | ~50 files |
| GitHub CI workflow files | ~90 files |
| Start/run/deploy/setup scripts | ~50 files |
| build.rs files (FFI/codegen) | 23 files |
| Makefiles (project-level) | 5 files |

### Key CLI Commands Discovered

| Command | Entry File | Repo |
|---------|-----------|------|
| `ruvector` | `npm/packages/ruvector/bin/cli.js` | ruvector |
| `ruvllm` | `npm/packages/ruvllm/bin/cli.js` | ruvector |
| `rvlite` | `npm/packages/rvlite/bin/cli.js` | ruvector |
| `claude-flow` | `v3/@claude-flow/cli/bin/cli.js` | claude-flow |
| `claude-flow-mcp` | `v3/@claude-flow/cli/bin/mcp-server.js` | claude-flow |
| `ruflo` | `ruflo/bin/ruflo.js` | claude-flow |
| `coflow` | `packages/coflow/bin/coflow.js` | claude-flow |
| `ruv-swarm` | `ruv-swarm/npm/bin/ruv-swarm-secure.js` | ruv-FANN |
| `agentic-flow` | `agentic-flow/src/cli-proxy.ts` | agentic-flow |
| `agentdb` | `agentic-flow/src/agentdb/cli/agentdb-cli.ts` | agentic-flow |
| `sublinear` | `dist/cli/index.js` | sublinear-time-solver |
| `agentic-synth` | `npm/packages/agentic-synth/bin/cli.js` | ruvector |
| `ruvbot` | `npm/packages/ruvbot/bin/ruvbot.js` | ruvector |
| `spiking-neural` / `snn` | `npm/packages/spiking-neural/bin/cli.js` | ruvector |
| `rudag` | `npm/packages/rudag/bin/cli.js` | ruvector |

### Multi-Import Hubs (Files Wiring 3+ Subsystems)

| File | Subsystems Connected | LOC |
|------|---------------------|-----|
| `ruvector/npm/packages/ruvector/src/core/index.ts` | GNN + AgentDB + SONA + ONNX + Router + Graph + Cluster + RVF + 12 more | 56 |
| `agentdb/src/mcp/agentdb-mcp-server.ts` | CausalMemory + Reflexion + Skills + NightlyLearner + EmbeddingService + ReasoningBank + BatchOps + security | 2,367 |
| `claude-flow/v3/@claude-flow/memory/src/controller-registry.ts` | 13 AgentDB controllers + 16 CLI memory controllers | 1,025 |
| `ruvllm/src/ruvector_integration.rs` | ruvector_core (HNSW) + ruvector_sona (ReasoningBank) + claude_flow (AgentRouter) + internal sona | 1,099 |
| `ruvllm/src/claude_flow/hnsw_router.rs` | ruvector_core::hnsw + ruvector_core::types + crate::sona (SonaIntegration) | 1,287 |
| `claude-flow/v3/mcp/tools/index.ts` | agents + swarm + memory + config + hooks + tasks + system + session + workers + sona + federation + v2compat | 445 |
| `agentic-flow/src/services/sona-agentdb-integration.ts` | @ruvector/sona (Rust WASM) + agentdb (vector DB) | 463 |
| `ruvector/npm/packages/ruvector/src/core/intelligence-engine.ts` | agentdb-fast + sona-wrapper + onnx-embedder + parallel-intelligence | 1,233 |
| `sublinear-time-solver/src/mcp/server.ts` | SublinearSolver + Matrix + Temporal + PsychoSymbolic + Domain + Consciousness + WASM | 1,327 |
| `agentdb/src/backends/factory.ts` | RuVectorBackend + RvfBackend + HNSWLibBackend (3-way auto-detect) | 344 |

### Unread Integration Clusters

| Cluster | Package | Files | Total LOC | Status |
|---------|---------|-------|-----------|--------|
| `npm/packages/agentic-integration/` | ruvector-rust | ~8 files | ~4,800 | ALL NOT_TOUCHED |
| `neuro-divergent-registry/` | ruv-fann-rust | ~5 files | ~3,900 | ALL NOT_TOUCHED |
| `cuda-wasm/src/neural_integration/` | ruv-fann-rust | ~7 files | ~4,700 | ALL NOT_TOUCHED |
| `v3/@claude-flow/memory/src/` | claude-flow-cli | ~10 files | ~10,000 | ALL NOT_TOUCHED |
| `v3/@claude-flow/cli/src/commands/` | claude-flow-cli | ~30 files | ~12,000 | ALL NOT_TOUCHED |
| `prime-radiant/src/ruvllm_integration/` | ruvector-rust | ~4 files | ~3,500 | ALL NOT_TOUCHED |

---

## Remaining Unread Files Per Package

| Package | NOT_TOUCHED Count | Total LOC |
|---------|-------------------|-----------|
| ruvector-rust | 2,792 | 1,003,001 |
| agentic-flow-rust | 2,099 | 767,175 |
| ruv-fann-rust | 824 | 263,009 |
| claude-flow-cli | 462 | 170,774 |
| claude-config | 365 | 73,476 |
| sublinear-rust | 246 | 90,435 |
| agentic-flow | 227 | 38,556 |
| agentdb | 87 | 23,212 |
| **TOTAL** | **7,105** | **~2.4M LOC** |

---

## Reading Plan

### Phase A: The Front Door (1 session, ~13K LOC)
Read Tier 1 files (#1-7). These answer: "What code actually executes when a user interacts with the system?"

**Expected outcome**: Either discover a working integration path we've never seen, or confirm the CLIs are thin wrappers over the broken adapters we already know about.

### Phase B: The Wiring (2 sessions, ~16K LOC)
Read Tier 2 files (#8-20). These answer: "How do subsystems find and call each other at runtime?"

**Expected outcome**: Understand the factory logic, controller lifecycle, and whether the V3 memory layer has a different (working?) code path than the broken V1/V2 adapters.

### Phase C: The Tools (1 session, ~5.5K LOC)
Read Tier 3 files (#21-27). These answer: "What happens when an MCP tool is invoked?"

**Expected outcome**: Trace from MCP tool call → handler → backend. See whether MCP tools bypass the broken TS adapters.

### Phase D: Ground Truth (1 session, ~4.5K LOC)
Read Tier 4 files (#28-35). These answer: "Does this actually build and run in CI? What does deployment look like?"

**Expected outcome**: CI workflows reveal which packages actually build, which tests pass, which Docker containers actually run.

### Phase E: Cleanup (0.5 sessions)
Fix ghost DEEP files (#Tier 5). Correct the DB to reflect actual read status. Recount true DEEP coverage.

**Total: ~5 sessions, ~39K LOC, answering the question we've spent 131 sessions circling around.**

---

## Why This Was Missed

The `smart_priority_gaps` view ranks files by:
- Connection to already-DEEP files (co-location signal)
- LOC (bigger = higher priority)
- Domain relevance tags

This creates a feedback loop: we read algorithm files → those become DEEP → nearby algorithm files rank higher → we read more algorithm files. The integration layer lives in **different directory trees** (`bin/`, `mcp/`, `v3/@claude-flow/`, deployment configs) with few connections to the algorithm files we've been reading.

It's like searching for roads by looking near the densest clusters of buildings. The roads are between the clusters — that's why we never found them.
