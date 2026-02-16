# R63 Execution Plan: Cross-Package Integration — AgentDB Bridge + Swarm Runtime + MCP Wrappers

**Date**: 2026-02-16
**Session ID**: 63
**Focus**: AgentDB's formal integration layer in custom-src (OWN_CODE: ruvector-backend-adapter, reflexion-service, reflexion-memory-adapter), ruv-swarm TS/Rust runtime (cli-diagnostics, agent.ts, MCP types), psycho-symbolic MCP integration wrappers (graph-reasoner, types), psycho-symbolic Rust extractors (preferences.rs), and ReasoningBank MCP tools in Rust (tools.rs)
**Parallel with**: R62 (no file overlap -- R63 = custom-src + ruv-swarm + sublinear-rust crates/ + agentic-flow-rust; R62 = sublinear-rust src/ root files only)

## IMPORTANT: Parallel Execution Notice

This plan runs IN PARALLEL with R62. The file lists are strictly non-overlapping:
- **R63 covers**: custom-src agentdb-integration (3 TS files), ruv-fann-rust ruv-swarm (3 files: 2 TS/JS + 1 Rust), sublinear-rust crates/psycho-symbolic-reasoner (3 files: 2 MCP TS + 1 Rust extractor), agentic-flow-rust reasoningbank-mcp (1 Rust file)
- **R62 covers**: sublinear-rust src/ files only (lib.rs, error.rs, solver_core.rs, graph/mod.rs, simd_ops.rs, solver/forward_push.rs, sublinear/johnson_lindenstrauss.rs, convergence/convergence-detector.js, mcp/tools/solver.ts, reasongraph/advanced-reasoning-engine.ts)
- **ZERO shared files** between R62 and R63
- **R63 has NO src/lib.rs, src/error.rs, src/solver/, src/solver_core.rs, src/graph/, src/simd_ops.rs, src/sublinear/, src/convergence/, src/mcp/, or src/reasongraph/ files**
- **R62 has NO agentdb-integration, ruv-swarm, crates/psycho-symbolic-reasoner, or reasoningbank files**
- Do NOT read or analyze any file from R62's list (see R62-plan.md for that list)

## Rationale

- **AgentDB integration is OWN_CODE and never examined**: These 3 files in custom-src (`~/claude-flow-self-implemented/src/`) are the ONLY OWN_CODE files in the top priority queue. They represent the formal AgentDB integration layer built on top of the library code. R20 found the ROOT CAUSE of broken AgentDB search (EmbeddingService never initialized), R48 found 3 disconnected AgentDB layers, R61 found 5 layers. Do these OWN_CODE files bridge the disconnect?
  - `ruvector-backend-adapter.ts` (374 LOC): Adapts ruvector as a storage backend for AgentDB. This is the formal bridge between ruvector-core's HNSW and AgentDB's vector search
  - `reflexion-service.ts` (330 LOC): Service layer for reflexion (self-improving agents). Connects to ReasoningBank?
  - `reflexion-memory-adapter.ts` (328 LOC): Memory adapter for reflexion. Part of the 5 disconnected AgentDB layers?
- **ruv-swarm TS runtime is the swarm's JS surface**: R50 found ruv-swarm Rust BIMODAL (memory.rs 95%, spawn.rs 8%). The TS/JS runtime (cli-diagnostics, agent.ts) is the consumer-facing layer. R31 found CLI = demonstration framework -- do these files confirm or reverse that?
  - `cli-diagnostics.js` (364 LOC): Diagnostic tooling for ruv-swarm CLI. Genuine debugging or theatrical?
  - `agent.ts` (342 LOC): The Agent class/interface for ruv-swarm. Core runtime object
  - `ruv-swarm-mcp/types.rs` (305 LOC): Rust MCP type definitions for ruv-swarm. Matches R61's genuine MCP pattern?
- **Psycho-symbolic MCP integration is first examination**: R55 found psycho-symbolic Rust 3-4x better than TS. The MCP integration layer wraps the Rust extractors for MCP consumers. Does the MCP quality match?
  - `graph-reasoner.ts` (386 LOC): MCP wrapper around graph-based reasoning. Genuine or theatrical?
  - `types/index.ts` (362 LOC): Type definitions for the MCP integration. Reveals the API surface
  - `preferences.rs` (337 LOC): Rust preference extraction. Quality of the Rust extractors
- **ReasoningBank MCP tools.rs is the current Rust MCP**: R61 found tools_old.rs (85-90%) correctly deprecated. tools.rs (333 LOC) is the CURRENT version. Does it improve on the deprecated one?

## Target: 10 files, ~3,461 LOC

---

### Cluster A: AgentDB Integration Layer — OWN_CODE (3 files, ~1,032 LOC)

The ONLY OWN_CODE files in the priority queue. These are in `~/claude-flow-self-implemented/src/agentdb-integration/` — custom code built on top of the library. This is the formal integration layer that should bridge AgentDB's 5 disconnected layers.

| # | File ID | File | LOC | Domain | Package |
|---|---------|------|-----|--------|---------|
| 1 | 2293 | `agentdb-integration/infrastructure/adapters/ruvector-backend-adapter.ts` | 374 | memory-and-learning | custom-src |
| 2 | 2279 | `agentdb-integration/episodic/services/reflexion-service.ts` | 330 | memory-and-learning | custom-src |
| 3 | 2275 | `agentdb-integration/episodic/adapters/reflexion-memory-adapter.ts` | 328 | memory-and-learning | custom-src |

**Full paths**:
1. `~/claude-flow-self-implemented/src/agentdb-integration/infrastructure/adapters/ruvector-backend-adapter.ts`
2. `~/claude-flow-self-implemented/src/agentdb-integration/episodic/services/reflexion-service.ts`
3. `~/claude-flow-self-implemented/src/agentdb-integration/episodic/adapters/reflexion-memory-adapter.ts`

**Key questions**:
- `ruvector-backend-adapter.ts`: Does this bridge ruvector-core to AgentDB?
  - R20 ROOT CAUSE: EmbeddingService never initialized, causing broken AgentDB search. Does this adapter properly initialize the embedding pipeline?
  - Does it import from ruvector-core's HNSW (the genuine one at 98-100%)?
  - Does it implement the AgentDB VectorBackend interface?
  - Does it handle vector serialization/deserialization correctly?
  - Does it use REAL embeddings or fall back to hash-based (the systemic 9+ occurrence pattern)?
  - At 374 LOC, is this the REAL bridge or another wrapper in the 5-layer stack?
  - This is the single most important file for the AgentDB disconnect story
- `reflexion-service.ts`: What reflexion capabilities does this provide?
  - Does it implement genuine self-reflection (trajectory evaluation, strategy adaptation)?
  - Does it connect to ReasoningBank (R57's 4th disconnected layer)?
  - Does it use the reflexionController from agentdb-wrapper.ts (R51)?
  - At 330 LOC, is there enough for real reflexion logic vs a pass-through?
  - Does it persist reflexion data or operate in-memory only?
  - Does it reference any of the 5 disconnected AgentDB layers?
- `reflexion-memory-adapter.ts`: How does reflexion store memories?
  - Does it implement a genuine memory adapter pattern (store, retrieve, query)?
  - Does it use the ruvector-backend-adapter (file 1) or a different storage backend?
  - Does it connect to the agentdb-fast.ts direct vectorBackend access (R51)?
  - Does it handle episodic memory (sequences of experiences) vs semantic memory (knowledge)?
  - Does it use hash-based embeddings (10th occurrence?) or real ones?
  - Is this the 6th disconnected AgentDB layer or does it unify existing ones?

**Follow-up context**:
- R20: ROOT CAUSE = EmbeddingService never initialized -> broken AgentDB search
- R48: 3 disconnected AgentDB layers
- R51: agentdb-wrapper.ts (85-88%), agentdb-fast.ts (72-78%), 9th hash embeddings
- R57: ReasoningBank = 4th disconnected data layer
- R61: ExplainableRecall = 5th disconnected AgentDB layer
- R55: R20 ROOT CAUSE PROVEN (persistence.js 95-98%)
- custom-src files have NEVER been examined in any session

---

### Cluster B: ruv-swarm Runtime Layer (3 files, ~1,011 LOC)

The consumer-facing ruv-swarm runtime: CLI diagnostics, the Agent class, and MCP type definitions. R50 found Rust bimodal (memory.rs 95%, spawn.rs 8%). Does the TS/JS layer follow the same split?

| # | File ID | File | LOC | Domain | Package |
|---|---------|------|-----|--------|---------|
| 4 | 9616 | `ruv-swarm/npm/src/cli-diagnostics.js` | 364 | swarm-coordination | ruv-fann-rust |
| 5 | 9607 | `ruv-swarm/npm/src/agent.ts` | 342 | swarm-coordination | ruv-fann-rust |
| 6 | 8976 | `ruv-swarm/crates/ruv-swarm-mcp/src/types.rs` | 305 | swarm-coordination | ruv-fann-rust |

**Full paths**:
4. `~/repos/ruv-FANN/ruv-swarm/npm/src/cli-diagnostics.js`
5. `~/repos/ruv-FANN/ruv-swarm/npm/src/agent.ts`
6. `~/repos/ruv-FANN/ruv-swarm/crates/ruv-swarm-mcp/src/types.rs`

**Key questions**:
- `cli-diagnostics.js`: What diagnostic capabilities does the ruv-swarm CLI have?
  - R31 found CLI = demonstration framework. Does diagnostics confirm (theatrical status displays) or reverse (genuine health checks)?
  - Does it inspect running swarm state (agent counts, memory usage, task queues)?
  - Does it connect to the MCP server for diagnostics or query local state only?
  - At 364 LOC in JS, is there enough for real diagnostic tooling?
  - Does it report genuine metrics or fabricated ones (like R56's metrics_collector 40-45%)?
  - R46 found Goalie REVERSAL (internal engines real) -- could cli-diagnostics show similar reversal?
- `agent.ts`: What is the ruv-swarm Agent abstraction?
  - This is the core Agent class/interface for the TS runtime. What capabilities does it define?
  - Does it implement lifecycle (spawn, execute, terminate)?
  - Does it connect to the MCP system or operate standalone?
  - Does it align with agentic-flow's agent-manager pattern (R51: FILES NOT FOUND)?
  - At 342 LOC, does it define a rich agent model or minimal interface?
  - Does it handle communication between agents (message passing, shared state)?
  - R50 found ruv-swarm Rust spawn.rs at 8% -- does the TS agent.ts compensate?
- `types.rs`: How does ruv-swarm define its MCP types?
  - R61 found tools_old.rs 85-90% in ReasoningBank MCP. Does ruv-swarm MCP match that quality?
  - Does it define comprehensive MCP tool types (tool definitions, input schemas, response types)?
  - Does it use serde for serialization?
  - Does it align with agentic-flow's MCP types or define independent ones?
  - At 305 LOC in Rust, is there enough for a complete type system?

**Follow-up context**:
- R50: ruv-swarm Rust bimodal (memory.rs 95%, spawn.rs 8%). Goalie genuine
- R31: CLI = demonstration framework
- R51: agent-manager.ts + claude-code-wrapper.ts FILES NOT FOUND (repo refactored)
- R61: tools_old.rs 85-90%, output.rs 92-95% ANTI-FACADE
- R46: Goalie REVERSAL (internal engines real, cli.ts 88-92%)

---

### Cluster C: Psycho-Symbolic MCP Integration (3 files, ~1,085 LOC)

The MCP integration layer for the psycho-symbolic reasoner. R55 found psycho-symbolic Rust 3-4x better than TS. These files include both MCP TS wrappers and a Rust extractor -- does the quality gap hold?

| # | File ID | File | LOC | Domain | Package |
|---|---------|------|-----|--------|---------|
| 7 | 14029 | `crates/psycho-symbolic-reasoner/mcp-integration/src/wrappers/graph-reasoner.ts` | 386 | memory-and-learning | sublinear-rust |
| 8 | 14036 | `crates/psycho-symbolic-reasoner/mcp-integration/types/index.ts` | 362 | memory-and-learning | sublinear-rust |
| 9 | 14004 | `crates/psycho-symbolic-reasoner/extractors/src/preferences.rs` | 337 | memory-and-learning | sublinear-rust |

**Full paths**:
7. `~/repos/sublinear-time-solver/crates/psycho-symbolic-reasoner/mcp-integration/src/wrappers/graph-reasoner.ts`
8. `~/repos/sublinear-time-solver/crates/psycho-symbolic-reasoner/mcp-integration/types/index.ts`
9. `~/repos/sublinear-time-solver/crates/psycho-symbolic-reasoner/extractors/src/preferences.rs`

**Key questions**:
- `graph-reasoner.ts`: Does the MCP wrapper expose genuine graph reasoning?
  - R55 found psycho-symbolic Rust 3-4x better than TS. Does this TS wrapper maintain quality?
  - Does it call into the Rust psycho-symbolic reasoner via FFI/WASM or reimplement?
  - Does it use genuine graph traversal (BFS, DFS, inference chains) or theatrical reasoning?
  - Does it define proper MCP tool schemas for graph reasoning operations?
  - Does it integrate with the solver's graph infrastructure (graph/mod.rs in R62)?
  - At 386 LOC, is there substance or mostly boilerplate MCP scaffolding?
- `types/index.ts`: What is the MCP integration type surface?
  - At 362 LOC, this is a substantial type file. What entities does it define?
  - Does it include types for graph nodes, edges, reasoning chains, evidence, confidence?
  - Does it align with the Rust types or define a separate TS type hierarchy?
  - Does it use proper TypeScript patterns (interfaces, generics, discriminated unions)?
  - Are there types for MCP tool inputs/outputs that match the MCP specification?
- `preferences.rs`: How does the Rust preference extractor work?
  - This is the Rust side of the psycho-symbolic reasoner. R55 found Rust 3-4x better quality
  - Does it implement genuine preference extraction (sentiment analysis, ranking, utility functions)?
  - Does it use NLP primitives or simple heuristics?
  - Does it use SIMD or other optimized operations?
  - At 337 LOC, is there enough for a real extractor vs a stub?
  - Does it output structured preference data (key-value, ranked lists, weighted preferences)?

**Follow-up context**:
- R55: Psycho-symbolic Rust 3-4x better than TS
- R47: Consciousness BIMODAL (infra 75-95% vs theory 0-5%). Psycho-symbolic relates to consciousness subsystem
- R58: memory-manager.ts 5th MISLABELED (in same psycho-symbolic area)
- R61: MCP tools BIMODAL (82-92% vs 18-28%)
- Psycho-symbolic MCP integration has NEVER been examined

---

### Cluster D: ReasoningBank MCP — Current Rust Implementation (1 file, ~333 LOC)

The current (non-deprecated) Rust MCP tools for ReasoningBank. R61 found tools_old.rs (85-90%) correctly deprecated. This is the replacement.

| # | File ID | File | LOC | Domain | Package |
|---|---------|------|-----|--------|---------|
| 10 | 13453 | `reasoningbank/crates/reasoningbank-mcp/src/tools.rs` | 333 | memory-and-learning | agentic-flow-rust |

**Full paths**:
10. `~/repos/agentic-flow/reasoningbank/crates/reasoningbank-mcp/src/tools.rs`

**Key questions**:
- `tools.rs`: Does the current ReasoningBank MCP improve on tools_old.rs?
  - R61 found tools_old.rs 85-90% (production MCP handlers, correctly deprecated sync->async)
  - Does tools.rs implement async handlers (the reason tools_old was deprecated)?
  - Does it use proper Rust MCP SDK (rmcp, tower) or raw JSON?
  - Does it define the same 4 tools + 2 resources, or expanded set?
  - Does it connect to the ReasoningBank data layer (R57's .swarm/memory.db)?
  - Does it handle trajectory storage, verdict queries, pattern search?
  - At 333 LOC (vs tools_old.rs 419 LOC), is it more concise or reduced in scope?
  - Does it bridge to any of the 5 disconnected AgentDB layers?
  - Does it use real embeddings or hash-based (potential 10th occurrence)?

**Follow-up context**:
- R61: tools_old.rs 85-90%, correctly deprecated sync->async
- R57: queries.ts 85-90% PRODUCTION-READY 7-table schema. ReasoningBank = 4th disconnected data layer
- R43: ReasoningBank core APIs genuine
- R51: 256 MCP tools in mcp-client.js genuine
- Hash-based embeddings: 9+ occurrences across Rust+JS

---

## Expected Outcomes

- **AgentDB bridge verdict**: Do the OWN_CODE agentdb-integration files bridge the 5 disconnected AgentDB layers? Does ruvector-backend-adapter properly initialize embeddings (R20 root cause)?
- **6th AgentDB layer?**: Whether reflexion-memory-adapter creates a 6th disconnected layer or unifies existing ones
- **Hash-based embeddings 10th**: Whether ruvector-backend-adapter uses real or hash-based embeddings
- **ruv-swarm TS quality**: Whether the TS runtime matches the Rust bimodal pattern (95% vs 8%) or is uniformly one quality level
- **CLI demonstration vs genuine**: Whether cli-diagnostics confirms R31 "CLI = demonstration framework" or reverses like R46's Goalie
- **Psycho-symbolic TS vs Rust gap**: Whether MCP TS wrappers maintain quality vs Rust extractors (R55 found 3-4x gap)
- **ReasoningBank MCP evolution**: Whether tools.rs improves on tools_old.rs (85-90%) with async handlers
- **MCP type quality**: Whether ruv-swarm and psycho-symbolic MCP types follow genuine patterns
- **Cross-package integration map**: How custom-src, ruv-fann-rust, sublinear-rust, and agentic-flow-rust actually connect (or don't) via MCP

## Stats Target

- ~10 file reads, ~3,461 LOC
- DEEP files: 1,049 -> ~1,059
- Expected findings: 55-75 (10 files across 4 packages, integration-heavy)

## Cross-Session Notes

- **ZERO overlap with R62**: R62 covers sublinear-rust src/ root files only. No shared files
- **Extends R20**: ruvector-backend-adapter directly tests the ROOT CAUSE (EmbeddingService initialization)
- **Extends R48/R51/R57/R61**: AgentDB integration files extend the 5-disconnected-layers investigation
- **Extends R50**: ruv-swarm TS files extend the Rust bimodal (95% vs 8%) finding
- **Extends R31**: cli-diagnostics extends "CLI = demonstration framework" finding
- **Extends R55**: psycho-symbolic MCP tests the Rust-vs-TS quality gap
- **Extends R61**: tools.rs extends tools_old.rs (85-90%) investigation
- **NEW: custom-src agentdb-integration** = first OWN_CODE deep reads ever
- **NEW: ruv-swarm agent.ts** = core Agent class never examined
- **NEW: psycho-symbolic MCP integration** = first MCP wrapper examination for this crate
- **NEW: ReasoningBank tools.rs** = current (non-deprecated) Rust MCP
- Combined DEEP files from R62+R63: 1,049 -> ~1,069 (approximately +20)
