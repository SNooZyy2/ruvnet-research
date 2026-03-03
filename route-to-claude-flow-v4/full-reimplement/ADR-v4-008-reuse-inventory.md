# ADR-v4-008: v4 Reuse Inventory

> **Status**: Proposed (revised 2026-03-03 with R115-R142 findings)
> **Date**: 2026-02-19 (original), 2026-03-03 (updated)
> **Deciders**: Research project lead
> **Supersedes**: None
> **Related**: ADR-v4-007 (Subsystem Graph), GENUINE-ASSETS.md, README-REALITY-CHECK.md, SPEC.md

---

## Context

The research project has produced three extraction-oriented artifacts:
- **GENUINE-ASSETS.md**: File-level copy list (85%+ realness files)
- **README-REALITY-CHECK.md**: Feature-level verdicts (GENUINE / PARTIALLY REAL / FABRICATED)
- **SPEC.md**: Build plan with phases and estimated effort

What's missing is the **decision matrix** that connects these three: for each capability v4 needs, which specific implementation should be used, what's its condition, and what work is needed to integrate it? GENUINE-ASSETS tells you *what* to copy. REALITY-CHECK tells you *what's real*. Neither tells you *what to do* — the mapping from v4 requirement to implementation action.

Additionally, the existing artifacts are organized differently:
- GENUINE-ASSETS: organized by source repo/crate
- REALITY-CHECK: organized by marketing claim category
- SPEC: organized by implementation phase

None are organized by **v4 capability** — the natural axis for making build decisions.

## Decision

Build a **v4 Reuse Inventory** — a single artifact (~5-8 pages) organized by v4 capability that maps each required capability to:

1. The best existing implementation(s)
2. Their realness scores and conditions
3. The specific action (COPY / ADAPT / FIX / REWRITE / BUILD)
4. Estimated effort and dependencies on other capabilities
5. The subsystem(s) involved (referencing ADR-v4-007 IDs)

This inventory becomes the **implementation planning artifact** — the document you consult when deciding what to build next.

## Inventory Structure

### Capability Categories

Capabilities are grouped by v4 architectural layer (matching SPEC.md section 4):

```
Layer 5: Claude Code CLI (consumer — not modified)
Layer 4: MCP Server (tool registry + request routing)
Layer 3: DDD Bounded Contexts (search, episodic, skill)
Layer 2: Infrastructure (embedding, vector backend, reasoning, persistence)
Layer 1: Rust Crates (algorithms, data structures, WASM exports)
Layer 0: Build & Tooling (wasm-pack, cargo, npm)
```

### Per-Capability Entry Format

Each capability entry contains:

```markdown
### [CAPABILITY-ID] Capability Name

**Layer**: N | **Priority**: P0/P1/P2 | **Action**: COPY/ADAPT/FIX/REWRITE/BUILD
**Subsystems**: R1, T4 (from ADR-v4-007)

| Field | Value |
|-------|-------|
| v4 Requirement | What v4 needs this capability to do |
| Best Implementation | Exact file path(s) and their realness scores |
| Condition | What works, what's broken, key findings |
| Action Required | Specific steps to make it v4-ready |
| Dependencies | Other capabilities that must be ready first |
| Effort | S (hours) / M (1-2 days) / L (3-5 days) / XL (1+ week) |
| Risk | What could go wrong |
| Evidence | Key research sessions and finding IDs |
```

## Complete Inventory

### Layer 1: Rust Crates

---

#### [L1-01] HNSW Vector Search

**Layer**: 1 | **Priority**: P0 | **Action**: COPY
**Subsystems**: R1

| Field | Value |
|-------|-------|
| v4 Requirement | Nearest-neighbor search over embedding vectors. Core of all semantic operations. |
| Best Implementation | `ruvector-core/src/hnsw.rs` (92-98%), `simd.rs` (92-98%) |
| Condition | Genuine HNSW wrapping hnsw_rs. Real SIMD dispatch (AVX-512/AVX2/NEON). Vendored hnsw_rs 98-100% (R52). 3 distinct HNSW implementations exist but ruvector-core is canonical. |
| Action Required | Copy crate. Verify `cargo test` passes standalone. Ensure `wasm_bindgen` exports from `lib.rs` not `lib_simple.rs` (ADR-v4-004). |
| Dependencies | None (standalone crate) |
| Effort | S — copy and verify |
| Risk | LOW — most thoroughly validated subsystem |
| Evidence | R12-R15, R52, R90, R106 |

---

#### [L1-02] Product Quantization + Conformal Prediction

**Layer**: 1 | **Priority**: P1 | **Action**: COPY
**Subsystems**: R1

| Field | Value |
|-------|-------|
| v4 Requirement | Memory-efficient approximate search (PQ) and calibrated uncertainty (conformal). |
| Best Implementation | `ruvector-core/src/advanced_features/product_quantization.rs` (88-92%, 551 LOC), `conformal_prediction.rs` (88-93%, 505 LOC) |
| Condition | Real k-means++, Lloyd's, ADC lookup tables. Valid Vovk et al. quantile formula. Both are standalone features within ruvector-core. |
| Action Required | Included in L1-01 crate copy. No separate action. |
| Dependencies | L1-01 (same crate) |
| Effort | S — comes with crate |
| Risk | LOW |
| Evidence | R90 |

---

#### [L1-03] Sublinear PageRank

**Layer**: 1 | **Priority**: P1 | **Action**: COPY
**Subsystems**: R10

| Field | Value |
|-------|-------|
| v4 Requirement | Fast approximate PageRank for memory graph ranking and knowledge retrieval. |
| Best Implementation | `sublinear-time-solver/src/sublinear/backward_push.rs` (95%+) |
| Condition | Genuine O(1/epsilon) sublinear. Confirmed by R56 SMOKING GUN analysis. Only genuine sublinear algorithm in the ecosystem (8 false claims elsewhere). |
| Action Required | Copy file. Add `wasm_bindgen` export. WARNING: the `SublinearSolver` TS wrapper routes through `lib_simple.rs` (facade) — bypass it entirely (ADR-v4-004). |
| Dependencies | None (standalone algorithm) |
| Effort | S — copy and export |
| Risk | LOW for algorithm. MEDIUM for WASM binding (complex types across boundary). |
| Evidence | R56, R70, R81, R85, R88 |

---

#### [L1-04] Temporal Analysis

**Layer**: 1 | **Priority**: P2 | **Action**: COPY
**Subsystems**: R8

| Field | Value |
|-------|-------|
| v4 Requirement | Time series analysis for performance monitoring, trend detection, anomaly alerting. |
| Best Implementation | `sublinear-time-solver/crates/temporal-tensor/` (93%, entire crate, 213 tests) |
| Condition | PRODUCTION-READY. Highest quality crate in ecosystem. All files >= 88%. |
| Action Required | Copy entire crate. Verify `cargo test`. |
| Dependencies | None (standalone crate) |
| Effort | S — copy and verify |
| Risk | LOW — best-tested crate |
| Evidence | R37 |

---

#### [L1-05] Micro-LoRA + EWC++

**Layer**: 1 | **Priority**: P1 | **Action**: COPY + FIX
**Subsystems**: R13

| Field | Value |
|-------|-------|
| v4 Requirement | On-device fine-tuning with continual learning (no catastrophic forgetting). |
| Best Implementation | `micro_lora.rs` (92-95%) in sona crate |
| Condition | Real NEON SIMD LoRA. EWC++ core math genuine. BUT: R106 training.rs shows Fisher information never updated during training (static EWC, not truly "++"). **R141 UPDATE**: sona Rust crate (10,582 LOC) as a whole FAILS cargo check (broken workspace integration). micro_lora.rs is genuine but must be extracted individually, not copied as part of the sona crate. |
| Action Required | Copy micro_lora.rs. FIX: wire Fisher updates in training loop for true continual adaptation. ~50-100 LOC change. |
| Dependencies | None for core. L2-01 (embeddings) for training data. |
| Effort | S for copy, M for Fisher fix |
| Risk | MEDIUM — Fisher update wiring is algorithmic, not boilerplate |
| Evidence | R37, R106, README-REALITY-CHECK.md |

---

#### [L1-06] Raft Consensus

**Layer**: 1 | **Priority**: P2 | **Action**: COPY
**Subsystems**: R12

| Field | Value |
|-------|-------|
| v4 Requirement | Distributed consensus for multi-agent state agreement. |
| Best Implementation | RAC crate (92%) + `p2p.rs` (92-95%, real libp2p) |
| Condition | Genuine Raft with leader election. Real libp2p transport (R44 reversal). |
| Action Required | Copy RAC crate. For v4 MVP, this is optional — Claude Code Task tool handles coordination. |
| Dependencies | None (standalone) |
| Effort | S — copy. Integration into v4 coordination is L effort. |
| Risk | LOW for copy. HIGH for integration (needs transport layer design). |
| Evidence | R42, R44 |

---

#### [L1-07] Batch Inference Engine

**Layer**: 1 | **Priority**: P1 | **Action**: COPY + FIX
**Subsystems**: R15

| Field | Value |
|-------|-------|
| v4 Requirement | Efficient batched LLM inference with continuous batching. |
| Best Implementation | `ruvllm/serving/` (6/6 DEEP, ~90% avg — BEST ruvllm subsystem). Key files: `batch.rs` (90-95%), `scheduler.rs` (90-92%), `kv_cache_manager.rs` (88-92%). |
| Condition | Real vLLM/Orca architecture. Correct merge_prefill_decode(), TokenBudget dual-gate, PagedAttention. 3 bugs: deadlock risk (double RwLock), memory estimate 2x too low, broken swap_out accounting. **R141 CRITICAL**: ruvllm (120,345 LOC) as a whole FAILS cargo check — largest crate in repo cannot compile. Individual files are algorithmically genuine but CANNOT be extracted via `cargo build`. |
| Action Required | CHERRY-PICK individual .rs files. Cannot copy crate. Would need to create a new minimal crate wrapping batch.rs + scheduler.rs + kv_cache_manager.rs with minimal dependencies. |
| Dependencies | L1-01 for KV cache vector operations |
| Effort | L — cherry-pick + new crate scaffolding + 3 targeted bug fixes (upgraded from M) |
| Risk | HIGH — the files are genuine but isolating them from a 120K LOC non-compiling crate is non-trivial (upgraded from MEDIUM) |
| Evidence | R35, R106, R107, **R141 (compilation audit)** |

---

#### [L1-08] Hyperbolic Geometry

**Layer**: 1 | **Priority**: P2 | **Action**: COPY
**Subsystems**: R6

| Field | Value |
|-------|-------|
| v4 Requirement | Hierarchical embedding space for tree-structured data (agent hierarchies, knowledge graphs). |
| Best Implementation | `hyperbolic-hnsw` crate (88-95%, COMPLETE per R99). Also: ruvector-attention hyperbolic module (90-93%), SQL hyperbolic (88-92%). |
| Condition | Genuine Poincare ball, Lorentz model, mixed curvature. 21 DEEP files across 4 crates. CRITICAL: zero manifold validation at any layer. |
| Action Required | Copy hyperbolic-hnsw crate. Add manifold validation (~100 LOC guard layer). |
| Dependencies | L1-01 (HNSW core) |
| Effort | S for copy, S for validation layer |
| Risk | LOW — math is solid, validation is straightforward |
| Evidence | R92, R97-R101 |

---

### Layer 2: Infrastructure

---

#### [L2-01] Embedding Service

**Layer**: 2 | **Priority**: P0 | **Action**: ADAPT (revised from BUILD — R117)
**Subsystems**: T4 (target), all subsystems (consumer)

| Field | Value |
|-------|-------|
| v4 Requirement | Convert text to dense vectors for semantic search. THE critical missing piece (R20 root cause). |
| Best Implementation | **R117 UPDATE**: `onnx-embedder.ts` in the ruvector umbrella package (~400 LOC, 85-90%) implements REAL ONNX embeddings via Tract/WASM with `all-MiniLM-L6-v2` model (384-dim). Also: R136 confirmed `agentdb-mcp-server.ts` initializes EmbeddingService with Xenova/all-MiniLM-L6-v2 (Pipeline 1 path only). |
| Condition | `onnx-embedder.ts` exists and works. The issue was never "no code exists" — it was "the code exists but is never wired into the runtime path." R135 confirmed `memory-initializer.ts` has 3 fallback paths for embedding but the real one (@xenova/transformers) is tried first. |
| Action Required | ADAPT `onnx-embedder.ts` — wire it into the v4 infrastructure layer at startup. Fail fast if model unavailable (ADR-v4-003). ~50 LOC of wiring, not 200 LOC from scratch. |
| Dependencies | None (foundational service) |
| Effort | S — adapt existing implementation (downgraded from M) |
| Risk | LOW — working code exists, just needs wiring (downgraded from MEDIUM) |
| Evidence | R20, R52, R65, R84, R88, **R117 (onnx-embedder found), R135 (fallback paths traced), R136 (agentdb-mcp-server initializes it)** |

---

#### [L2-02] Vector Backend (RuVectorBackend)

**Layer**: 2 | **Priority**: P0 | **Action**: COPY + FIX
**Subsystems**: T4

| Field | Value |
|-------|-------|
| v4 Requirement | Bridge between TS services and Rust HNSW. Handles vector storage, search, and lifecycle. |
| Best Implementation | `agentdb/src/backends/ruvector/RuVectorBackend.ts` (88-92%, ~500 LOC, R91 revised to 72-78% — discrepancy due to upstream dependency issues vs code quality) |
| Condition | Adaptive HNSW params, Semaphore for concurrency, BufferPool, path security (FORBIDDEN_PATH_PATTERNS). Works correctly when given real embeddings. |
| Action Required | Copy. Wire to L2-01 (EmbeddingService) at initialization. Remove upstream @claude-flow imports. |
| Dependencies | L2-01 (embeddings), L1-01 (HNSW via WASM or FFI) |
| Effort | S — copy + rewire imports |
| Risk | LOW — code is solid, problem was always the missing embeddings |
| Evidence | R20, R88, R91 |

---

#### [L2-03] ReasoningBank

**Layer**: 2 | **Priority**: P0 | **Action**: COPY
**Subsystems**: T5, R18

| Field | Value |
|-------|-------|
| v4 Requirement | Pattern storage, retrieval, and judgment from past agent decisions. Core of self-learning. |
| Best Implementation | TS: `reasoningbank-types.ts`, `pre-task.ts`, `async_learner` (92-95%). Rust: ReasoningBank in ruvllm (92-95%, WASM-compatible). |
| Condition | Both implementations genuine. Statistical ranking, decay coefficients, MMR search. DeepMind-style MaTTS search. v1->v2 migration complete. R102: PatternStore uses REAL VectorDB (first non-hash semantic store). |
| Action Required | Copy both TS and Rust implementations. Wire TS version into hook pipeline (L4-02). |
| Dependencies | L2-01 (for semantic pattern matching), L4-02 (hooks for triggering) |
| Effort | S — copy, already well-integrated with hooks |
| Risk | LOW — most validated learning subsystem |
| Evidence | R67, R73-R78, R83, R102 |

---

#### [L2-04] Persistence (SQLite)

**Layer**: 2 | **Priority**: P0 | **Action**: ADAPT
**Subsystems**: T6

| Field | Value |
|-------|-------|
| v4 Requirement | Single, unified data store for all agent state, memories, skills, and search indices. |
| Best Implementation | Self-implemented `database-adapter.ts` (215 LOC), `schema-migrator.ts` (233 LOC), `agentdb-schema.sql` (167 LOC) from `claude-flow-self-implemented/`. Also: `sqlite-pool.ts` (92%, R45). |
| Condition | 12 disconnected persistence layers in the ecosystem. None compose. The self-implemented DDD pattern is the right architecture — it just needs to be the ONLY one. |
| Action Required | Adapt self-implemented persistence layer. Strip upstream imports. Merge sqlite-pool connection management. Consolidate schema from all 12 layers into one. |
| Dependencies | None (foundational) |
| Effort | M — schema consolidation is the main work |
| Risk | MEDIUM — schema design affects all other layers. Get it wrong and everything is coupled. |
| Evidence | ADR-v4-002, R45, R85-R87, R108 |

---

### Layer 3: DDD Bounded Contexts

---

#### [L3-01] Search Context (Hybrid BM25 + Vector)

**Layer**: 3 | **Priority**: P0 | **Action**: ADAPT
**Subsystems**: T6

| Field | Value |
|-------|-------|
| v4 Requirement | Semantic search over agent memories, skills, and knowledge. |
| Best Implementation | Self-implemented: `hybrid-search-service.ts` (366 LOC), `search-pipeline.ts` (287 LOC), `bm25-index.ts` (269 LOC), `mmr-adapter.ts` (141 LOC) |
| Condition | DDD architecture sound. BM25 + vector fusion with MMR diversity. Problem was upstream deps, not domain model. |
| Action Required | Strip upstream imports. Retarget to L2-02 (RuVectorBackend) and L2-04 (SQLite). Also: extract `sanitizeFTS5Query()` (~60 LOC) from `~/OCR-Provenance/src/services/search/bm25.ts:869-928` into `src/utils/fts5-sanitizer.ts` — prevents FTS5 injection, no equivalent in self-impl. See Rejected Alternatives section. |
| Dependencies | L2-01, L2-02, L2-04 |
| Effort | M — import retargeting + integration tests |
| Risk | LOW — domain model already proven |
| Evidence | SPEC.md Phase 2, OCR-Provenance assessment 2026-03-03 |

---

#### [L3-02] Episodic Context (Reflexion/Experience Replay)

**Layer**: 3 | **Priority**: P1 | **Action**: ADAPT
**Subsystems**: T6

| Field | Value |
|-------|-------|
| v4 Requirement | Store and replay agent experiences for learning from past sessions. |
| Best Implementation | Self-implemented: `reflexion-service.ts` (330 LOC), `reflexion-memory-adapter.ts` (328 LOC), `episode-repository.ts` (322 LOC) |
| Condition | Sound DDD. Connected to ReasoningBank for pattern extraction. |
| Action Required | Strip upstream imports. Retarget to L2-03 (ReasoningBank) and L2-04 (SQLite). |
| Dependencies | L2-03, L2-04 |
| Effort | M |
| Risk | LOW |
| Evidence | SPEC.md Phase 2 |

---

#### [L3-03] Skill Context (Library + Consolidation)

**Layer**: 3 | **Priority**: P1 | **Action**: ADAPT
**Subsystems**: T6

| Field | Value |
|-------|-------|
| v4 Requirement | Extract, store, and suggest reusable skills from agent work. |
| Best Implementation | Self-implemented: `skill-library-service.ts` (483 LOC), `skill-repository.ts` (331 LOC), `consolidation-service.ts` (273 LOC) |
| Condition | Sound DDD. Consolidation service handles dedup and refinement. |
| Action Required | Strip upstream imports. Retarget to L2-04 (SQLite). |
| Dependencies | L2-04 |
| Effort | M |
| Risk | LOW |
| Evidence | SPEC.md Phase 2 |

---

### Layer 4: MCP + Hooks

---

#### [L4-01] MCP Server

**Layer**: 4 | **Priority**: P0 | **Action**: ADAPT + CONSOLIDATE
**Subsystems**: T1

| Field | Value |
|-------|-------|
| v4 Requirement | Tool registry and request routing between Claude Code and v4 services. |
| Best Implementation | Existing MCP server (R51: 256 tools, GENUINE). Uses `@modelcontextprotocol/sdk`. |
| Condition | 6 parallel MCP protocols exist — only `@modelcontextprotocol/sdk` should survive (ADR-v4-001). The existing server works but registers tools from ALL subsystems including dead ones. |
| Action Required | Fork existing MCP server. Remove tool registrations for dead subsystems. Add tool registrations for L3 contexts. |
| Dependencies | L3-01, L3-02, L3-03 (tools to register) |
| Effort | M — selective pruning + new tool wiring |
| Risk | LOW — MCP server architecture is proven |
| Evidence | R51, ADR-v4-001 |

---

#### [L4-02] Hook Pipeline

**Layer**: 4 | **Priority**: P0 | **Action**: COPY
**Subsystems**: T2

| Field | Value |
|-------|-------|
| v4 Requirement | Lifecycle hooks for pre-task, post-task, pre-edit, etc. Backbone of self-learning and model routing. |
| Best Implementation | Existing hook pipeline (R19: 98.1% — one of the most genuine subsystems) |
| Condition | Production-quality core. Already integrated with ReasoningBank via pre-task hook. Model routing works through hooks. **R140 UPDATE**: hooks.ts is 4,530 LOC with 30 real MCP wrapper subcommands (not 17 documented). `token-optimize` has hardcoded +200 fake savings. `pre-task` has REAL ADR-008 3-tier routing via `enhanced-model-router.js`. Statusline (470 LOC) uses `dbSizeKB/2` heuristic for vector count — not real DB query. |
| Action Required | Copy core pipeline. PRUNE: remove fake token savings from `token-optimize`, remove statusline heuristic. Keep: all 30 real subcommands, ADR-008 routing, ReasoningBank integration. |
| Dependencies | None (other things depend on it) |
| Effort | M |
| Risk | LOW |
| Evidence | R19 |

---

#### [L4-03] Execution Engine (NEW — R140)

**Layer**: 4 | **Priority**: P1 | **Action**: ADAPT
**Subsystems**: T1

| Field | Value |
|-------|-------|
| v4 Requirement | Spawn and manage agent worker processes for multi-agent coordination. |
| Best Implementation | `HeadlessWorkerExecutor` in claude-flow-cli (~600 LOC, 78-83% genuine, R140). Real process pool, context caching, output parsing. |
| Condition | Agent execution = `spawn('claude', ['--print', prompt])` subprocess. Three-tier chain: ContainerWorkerPool (Docker) → worker-daemon → HeadlessWorkerExecutor. HeadlessWorkerExecutor is the genuine core. CRITICAL BUG: `buildWorkerCommand()` silently drops prompt+contextPatterns. 9/12 worker-daemon types are facade stubs. |
| Action Required | Extract HeadlessWorkerExecutor. Fix buildWorkerCommand() context dropping. Discard worker-daemon facade stubs. Optionally keep ContainerWorkerPool for Docker support. |
| Dependencies | L4-01 (MCP server for tool routing to workers) |
| Effort | M — extract, fix bug, integrate with v4 MCP |
| Risk | MEDIUM — process pool management has edge cases (zombie processes, resource leaks) |
| Evidence | R140 |

---

#### [L4-04] Bayesian Agent Routing (NEW — R140)

**Layer**: 4 | **Priority**: P1 | **Action**: COPY
**Subsystems**: T1

| Field | Value |
|-------|-------|
| v4 Requirement | Learn which agent types perform best for which task categories over time. |
| Best Implementation | `sona-optimizer.ts` (842 LOC, 72-78%, R140). Genuine Bayesian agent-routing with temporal decay and Thompson sampling. |
| Condition | The ONLY V3 memory subsystem actually wired into the hooks pipeline. Updates beliefs based on observed outcomes. Zero HNSW/ruvector connection (operates independently). |
| Action Required | Copy as-is. Already works through hooks. |
| Dependencies | L4-02 (hooks pipeline for triggering) |
| Effort | S — copy |
| Risk | LOW — self-contained, already proven in V3 runtime |
| Evidence | R140 |

---

#### [L4-05] Cryptographic Provenance (NEW — R122-R124)

**Layer**: 4 | **Priority**: P2 | **Action**: COPY + BRIDGE
**Subsystems**: R1 (RVF crates)

| Field | Value |
|-------|-------|
| v4 Requirement | Tamper-evident audit trail for agent actions and vector DB operations. |
| Best Implementation | RVF witness chains: `witness.rs` (SHAKE-256), `store.rs`, `write_path.rs` in rvf-runtime (85-92%, R122-R124). Also `rvf-node/lib.rs` NAPI bridge (85-90%, R121). |
| Condition | Genuine cryptographic provenance. Each entry links to predecessor via hash chain. Ed25519 signatures available in RAC crate. NAPI bridge exists for Node.js access. |
| Action Required | Copy RVF crates. Use existing NAPI bridge. Wire witness chain into v4 persistence layer (L2-04) for audit trail on all state changes. |
| Dependencies | L2-04 (persistence), L0-01 (NAPI build) |
| Effort | M — copy + integration wiring |
| Risk | MEDIUM — NAPI bridge exists but witness chain integration with SQLite needs design |
| Evidence | R121-R124 |

---

#### [L3-04] Agent Lifecycle Unification (NEW — synthesis docs)

**Layer**: 3 | **Priority**: P1 | **Action**: DESIGN
**Subsystems**: T1

| Field | Value |
|-------|-------|
| v4 Requirement | Single canonical pattern for agent creation, execution, lifecycle management, and teardown. |
| Best Implementation | Three incompatible patterns exist: (1) CLI-based AgentManager (spawn/terminate/status), (2) LongRunningAgent (220 LOC — real budget enforcement, checkpointing, provider failover), (3) EphemeralAgent (fire-and-forget). Additionally, `claudeFlowAgent.js` (116 LOC) provides real Claude Agent SDK integration via `query()` with streaming. |
| Condition | All three patterns work independently but share zero code. AgentManager is tightly coupled to CLI. LongRunningAgent has the most production-quality features. claudeFlowAgent.js is the canonical Claude SDK path. |
| Action Required | DESIGN unified interface. Keep LongRunningAgent's budget enforcement + claudeFlowAgent.js's SDK integration. Discard CLI-specific AgentManager coupling. |
| Dependencies | L4-03 (Execution Engine) |
| Effort | M — interface design + adapter wiring |
| Risk | MEDIUM — three codebases to reconcile, risk of feature regression |
| Evidence | agent-lifecycle domain analysis, R140 |

---

#### [L2-05] ReasoningBank Consolidation (NEW — synthesis docs)

**Layer**: 2 | **Priority**: P1 | **Action**: CONSOLIDATE
**Subsystems**: T5, R18

| Field | Value |
|-------|-------|
| v4 Requirement | Single ReasoningBank implementation with pattern storage, judgment, distillation, and consolidation. |
| Best Implementation | 4 independent implementations with zero code sharing: (1) claude-flow TS (92-95%), (2) agentic-flow TS with 5-step pipeline (Retrieve→Judge→Distill→Consolidate→MaTTS), (3) agentdb TS, (4) Rust workspace: core (88-92%), storage (94%), learning (95-98% BEST learning code), mcp (93-95%). |
| Condition | Best candidates: agentic-flow's 5-step pipeline for the workflow + Rust workspace for the algorithms. Risk: `distill.ts` in agentic-flow uses DeepSeek LLM while `reasoningbank-learning` uses gradient descent — zero integration between approaches. reasoningbank-mcp fails compilation (6 errors, mismatched StorageConfig types). |
| Action Required | CONSOLIDATE: Use agentic-flow pipeline as workflow orchestrator. Wire Rust learning crate (95-98%) for the actual learning. Fix reasoningbank-mcp StorageConfig types. Strip DeepSeek dependency from distill.ts (replace with local model or remove). |
| Dependencies | L2-01 (embeddings for pattern matching), L2-04 (persistence) |
| Effort | L — 4 implementations to reconcile, StorageConfig fix, DeepSeek removal |
| Risk | HIGH — most fragmented subsystem in the ecosystem. Easy to pick wrong "winner." |
| Evidence | memory-and-learning domain, agentic-flow domain, R67-R78, R83, R102 |

---

#### [L2-06] Security Fixes (NEW — synthesis docs)

**Layer**: 2 | **Priority**: P0 | **Action**: FIX
**Subsystems**: Multiple

| Field | Value |
|-------|-------|
| v4 Requirement | Eliminate critical security vulnerabilities before any code extraction. |
| Best Implementation | N/A — these are bugs to fix, not features to copy. |
| Condition | 5 critical security issues identified across synthesis domains: (1) 5 command injection vulns in `independent_verification_system.ts` via `execSync` with unvalidated input (memory-and-learning C52, R47). (2) WAL commit flag never set in `storage/file.rs` — deletions non-durable (ruvector C38, R108). (3) Ed25519 hardcoded example keys + unencrypted private key storage (ruvector C36/C66). (4) Path traversal validation is a no-op in `controller-registry.ts` — `path.resolve()` normalizes BEFORE check (agentdb-integration C53, R136). (5) `intelligence.ts` O(n²) `compactPatterns()` blocks event loop at scale (R140). |
| Action Required | FIX all 5 before extracting affected files. For files in the "NEVER Copy" list (intelligence.ts), ensure replacements don't inherit the same bugs. For files being copied (storage/file.rs), fix in-place before extraction. |
| Dependencies | None — should be done first |
| Effort | M — targeted fixes, each is 10-50 LOC |
| Risk | LOW for individual fixes. HIGH if skipped — these are RCE and data-loss class bugs. |
| Evidence | ruvector C38 (R108), memory-and-learning C52 (R47), ruvector C36/C66 (R108/R111), agentdb-integration C53 (R136), R140 |

---

### Layer 0: Build & Tooling

---

#### [L0-01] NAPI/WASM Build Pipeline

**Layer**: 0 | **Priority**: P0 | **Action**: BUILD (NAPI primary, WASM fallback)
**Subsystems**: None (new)

| Field | Value |
|-------|-------|
| v4 Requirement | Compile Rust crates to NAPI (Node.js) and optionally WASM (browser) for use in TS layer. |
| Best Implementation | **R116-R117 UPDATE**: The ruvector NAPI binary WORKS (`napi-rs`). `onnx-embedder.ts` uses NAPI successfully. WASM path: 60% genuine (R56-R60), 40% theatrical. NAPI is the preferred bridge for Node.js. WASM remains needed only for browser targets. |
| Condition | `wasm-pack` works individually on crates. The problem is the build config, not the tooling. |
| Action Required | BUILD: Primary path via `napi-rs` (proven working in ruvector umbrella, R116-R117). Create NAPI bindings for each Tier 1 crate. WASM as secondary target for browser compatibility using `wasm-pack build` with `lib.rs` (ADR-v4-004). |
| Dependencies | L1-* crates must be copied first |
| Effort | M — NAPI is less effort than pure WASM (downgraded from L) |
| Risk | MEDIUM — NAPI binding is simpler than WASM (no serialization boundary), but each crate still needs individual binding design. Reduced from HIGH. |
| Evidence | R85, ADR-v4-004, **R116-R117 (NAPI proven working)** |

---

## Priority Ordering

### P0 — Must have for MVP (blocks everything else)

| ID | Capability | Action | Effort | Critical Path? |
|----|-----------|--------|--------|---------------|
| L2-01 | Embedding Service | ADAPT | S | YES — R20 root cause |
| L2-04 | Persistence (SQLite) | ADAPT | M | YES — all state depends on this |
| L1-01 | HNSW Vector Search | COPY | S | YES — core search |
| L2-02 | Vector Backend | COPY+FIX | S | YES — bridges L1-01 to L3-01 |
| L2-03 | ReasoningBank | COPY | S | NO — but enables self-learning |
| L2-06 | Security Fixes | FIX | M | YES — must fix before extracting affected files |
| L4-02 | Hook Pipeline | COPY+PRUNE | M | NO — but enables model routing |
| L4-01 | MCP Server | ADAPT | M | YES — user-facing surface |
| L3-01 | Search Context | ADAPT | M | YES — primary user capability |
| L0-01 | NAPI/WASM Build | BUILD | M | YES — connects Rust to TS |

**P0 Critical Path**: L2-04 -> L2-01 (ADAPT, not BUILD) -> L1-01 -> L0-01 (NAPI primary) -> L2-02 -> L3-01 -> L4-01

> **R115-R142 revision**: L2-01 effort reduced from M to S (onnx-embedder.ts found). L0-01 effort reduced from L to M (NAPI proven). L1-07 effort increased from M to L (ruvllm doesn't compile). Net effect: P0 path is ~1-2 days shorter than original estimate.

### P1 — Important for self-learning and inference

| ID | Capability | Action | Effort |
|----|-----------|--------|--------|
| L1-03 | Sublinear PageRank | COPY | S |
| L1-05 | Micro-LoRA + EWC++ | COPY+FIX | M |
| L1-07 | Batch Inference | CHERRY-PICK + FIX | L (upgraded — ruvllm doesn't compile, R141) |
| L3-02 | Episodic Context | ADAPT | M |
| L3-03 | Skill Context | ADAPT | M |
| L3-04 | Agent Lifecycle Unification | DESIGN | M |
| L2-05 | ReasoningBank Consolidation | CONSOLIDATE | L |

### P2 — Nice to have, genuinely useful

| ID | Capability | Action | Effort |
|----|-----------|--------|--------|
| L1-02 | PQ + Conformal | COPY | S |
| L1-04 | Temporal Analysis | COPY | S |
| L1-06 | Raft Consensus | COPY | S |
| L1-08 | Hyperbolic Geometry | COPY | S |

## Capabilities NOT in Inventory (Fabricated — Do Not Build)

These were claimed in the README but have no genuine implementation:

| Claimed Capability | Why Excluded | Reference |
|-------------------|-------------|-----------|
| 9 RL Algorithms | All identical tabular Q-values | REALITY-CHECK |
| LearningBridge | Zero code exists | REALITY-CHECK |
| IPFS Marketplace | Fake CID generation | REALITY-CHECK |
| Byzantine Consensus | coordination.rs 15-25% FACADE | R84 |
| CRDT Sync | No CRDTs found | REALITY-CHECK |
| Int8 Quantization "3.92x" | Returns empty Vec | R82, R87 |
| Agent Booster "352x" | Console.log facades | REALITY-CHECK |
| SWE-Bench "84.8%" | Generates English, can't execute | REALITY-CHECK |
| Multi-Agent Collusion Detection | No code found | REALITY-CHECK |
| V3 AgentDB Integration | `agentdb-adapter.ts` is plain Map, not AgentDB. Bridge not compiled. | R135-R136 |
| V3 CI/CD Validation | All pipelines use continue-on-error. Cannot trust green builds. | R139 |
| intelligence.ts | O(n) brute-force facade with 14+ consumers | R140 |

## Rejected Alternatives (Evaluated, Not Adopted)

Real implementations that were assessed but superseded by better-positioned inventory items. Documented here to prevent re-evaluation during build.

### OCR-Provenance (~/OCR-Provenance) — Assessed 2026-03-03

Source: Fork of ChrisRoyse/OCR-Provenance. 121/121 tests pass, clean TS build. MIT-compatible.

| Subsystem | LOC | Verdict | Superseded By | Rationale |
|-----------|-----|---------|---------------|-----------|
| Provenance Chain (chain-hash, tracker, verifier) | ~1,563 | **REDUNDANT** | L4-05 (RVF witness chains) | RVF uses stronger crypto (SHAKE-256 vs SHA-256), has working NAPI bridge, lives in Rust layer v4 already targets. Adding a parallel TS provenance chain recreates the "N disconnected subsystems" anti-pattern. |
| Hybrid Search (BM25 + RRF fusion) | ~1,200 | **MOSTLY REDUNDANT** | L3-01 (self-impl DDD search) | Self-impl search is already designed for v4's DDD architecture and agent memory schema. OCR-Provenance search tuned for OCR documents (quality multipliers, VLM extraction) — wrong domain. |
| Hash Utilities (hash.ts) | 156 | **TRIVIALLY REIMPLEMENTABLE** | Node.js `crypto` | `crypto.createHash('sha256')` is one line. Streaming hash is `createReadStream().pipe(createHash())`. Not worth a formal reuse item. |
| FTS5 Query Sanitizer (in bm25.ts:869-928) | ~60 | **EXTRACT** | — (no existing equivalent) | Genuine gap-filler. Strips FTS5 metacharacters, preserves AND/OR/NOT, prevents negative-only queries. Extract into `src/utils/fts5-sanitizer.ts` during L3-01 implementation. |

**Key lesson**: The OCR-Provenance codebase is genuinely well-built (~2,400 LOC, 121 tests), but 3 of 4 subsystems duplicate capabilities already in the v4 inventory with better-positioned implementations. Adopting them would increase complexity without adding capability — the exact anti-pattern 142 research sessions identified.

**Action item**: When implementing L3-01 (Search Context), extract `sanitizeFTS5Query()` from `~/OCR-Provenance/src/services/search/bm25.ts` lines 869-928. This is the sole genuinely useful piece.

## Maintenance

This inventory should be updated when:
1. A new research session changes a realness score significantly (>10% shift)
2. A v4 implementation phase completes (mark capabilities as DONE)
3. A new capability is discovered that wasn't in the original README claims
4. A COPY capability fails `cargo test` (escalate to FIX or REWRITE)

## Consequences

**Positive**:
- Single source of truth for "what do I build next?"
- Organized by v4 need, not by source repo — natural for implementation planning
- Priority ordering + critical path enables sequenced execution
- Explicit "do not build" list prevents wasted effort on fabricated features

**Negative**:
- Partially overlaps with GENUINE-ASSETS.md (file list) and REALITY-CHECK.md (verdicts)
- Must be kept in sync with those docs (or they become stale)

**Neutral**:
- This document is expected to shrink over time as capabilities are implemented and marked DONE
- P2 items may be deferred to v5 without impacting v4 MVP
