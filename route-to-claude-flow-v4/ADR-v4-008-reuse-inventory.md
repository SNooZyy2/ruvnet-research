# ADR-v4-008: v4 Reuse Inventory

> **Status**: Proposed
> **Date**: 2026-02-19
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
| Condition | Real NEON SIMD LoRA. EWC++ core math genuine. BUT: R106 training.rs shows Fisher information never updated during training (static EWC, not truly "++"). |
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
| Condition | Real vLLM/Orca architecture. Correct merge_prefill_decode(), TokenBudget dual-gate, PagedAttention. 3 bugs: deadlock risk (double RwLock), memory estimate 2x too low, broken swap_out accounting. |
| Action Required | Copy serving/ directory. FIX 3 bugs before production use. |
| Dependencies | L1-01 for KV cache vector operations |
| Effort | M — copy + 3 targeted bug fixes |
| Risk | MEDIUM — bugs are known and localized |
| Evidence | R35, R106, R107 |

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

**Layer**: 2 | **Priority**: P0 | **Action**: BUILD
**Subsystems**: T4 (target), all subsystems (consumer)

| Field | Value |
|-------|-------|
| v4 Requirement | Convert text to dense vectors for semantic search. THE critical missing piece (R20 root cause). |
| Best Implementation | **None that works.** R20 confirmed: EmbeddingService was never initialized. 16+ hash-based embedding placeholders exist but produce garbage. R102 found ONE exception: ruvllm reasoning_bank PatternStore uses real VectorDB — but it's isolated. |
| Condition | The entire ecosystem's semantic search is broken because this one service was never wired. |
| Action Required | BUILD new EmbeddingService using `@xenova/transformers` with `all-MiniLM-L6-v2` model (~80MB). Fail fast — no hash fallback (ADR-v4-003). ~200 LOC. |
| Dependencies | None (foundational service) |
| Effort | M — ~200 LOC but requires model loading, batching, error handling |
| Risk | MEDIUM — model size (~80MB) may be too large for CLI. Mitigation: lazy-load on first search. |
| Evidence | R20, R52, R65, R84, R88 |

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
| Action Required | Strip upstream imports. Retarget to L2-02 (RuVectorBackend) and L2-04 (SQLite). |
| Dependencies | L2-01, L2-02, L2-04 |
| Effort | M — import retargeting + integration tests |
| Risk | LOW — domain model already proven |
| Evidence | SPEC.md Phase 2 |

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
| Condition | Production-quality. Already integrated with ReasoningBank via pre-task hook. Model routing works through hooks. |
| Action Required | Copy as-is. This is the glue layer — it already works. |
| Dependencies | None (other things depend on it) |
| Effort | S |
| Risk | LOW |
| Evidence | R19 |

---

### Layer 0: Build & Tooling

---

#### [L0-01] WASM Build Pipeline

**Layer**: 0 | **Priority**: P0 | **Action**: BUILD
**Subsystems**: None (new)

| Field | Value |
|-------|-------|
| v4 Requirement | Compile Rust crates to WASM for use in Node.js/TS layer. |
| Best Implementation | No single working pipeline exists. 60% of existing WASM is genuine (R56-R60), 40% is theatrical. `lib_simple.rs` facade deliberately excludes genuine algorithms (R85). |
| Condition | `wasm-pack` works individually on crates. The problem is the build config, not the tooling. |
| Action Required | BUILD: Create `wasm-pack build` configs for each Tier 1 crate, targeting `lib.rs` (ADR-v4-004). Test each export individually. |
| Dependencies | L1-* crates must be copied first |
| Effort | L — each crate needs individual wasm_bindgen annotations and boundary type design |
| Risk | HIGH — complex Rust types (generics, lifetimes) across WASM boundary are the #1 integration risk |
| Evidence | R85, ADR-v4-004 |

---

## Priority Ordering

### P0 — Must have for MVP (blocks everything else)

| ID | Capability | Action | Effort | Critical Path? |
|----|-----------|--------|--------|---------------|
| L2-01 | Embedding Service | BUILD | M | YES — R20 root cause |
| L2-04 | Persistence (SQLite) | ADAPT | M | YES — all state depends on this |
| L1-01 | HNSW Vector Search | COPY | S | YES — core search |
| L2-02 | Vector Backend | COPY+FIX | S | YES — bridges L1-01 to L3-01 |
| L2-03 | ReasoningBank | COPY | S | NO — but enables self-learning |
| L4-02 | Hook Pipeline | COPY | S | NO — but enables model routing |
| L4-01 | MCP Server | ADAPT | M | YES — user-facing surface |
| L3-01 | Search Context | ADAPT | M | YES — primary user capability |
| L0-01 | WASM Build | BUILD | L | YES — connects Rust to TS |

**P0 Critical Path**: L2-04 -> L2-01 -> L1-01 -> L0-01 -> L2-02 -> L3-01 -> L4-01

### P1 — Important for self-learning and inference

| ID | Capability | Action | Effort |
|----|-----------|--------|--------|
| L1-03 | Sublinear PageRank | COPY | S |
| L1-05 | Micro-LoRA + EWC++ | COPY+FIX | M |
| L1-07 | Batch Inference | COPY+FIX | M |
| L3-02 | Episodic Context | ADAPT | M |
| L3-03 | Skill Context | ADAPT | M |

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
