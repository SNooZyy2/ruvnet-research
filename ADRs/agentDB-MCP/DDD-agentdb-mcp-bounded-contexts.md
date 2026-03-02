# DDD Analysis: AgentDB MCP Bounded Contexts

**Date**: 2026-02-21
**Scope**: `packages/agentdb/` (140,908 LOC, 439 files) + integration layers
**Evidence**: 59 DEEP files, 1,507 findings, 195 dependency edges, 63 sessions

---

## 1. Domain Map

The AgentDB ecosystem decomposes into **9 bounded contexts** identified through 114 sessions of deep-read analysis. Each context has clear responsibility boundaries but **critical integration failures** at the seams.

```
┌─────────────────────────────────────────────────────────────┐
│                    AgentDB Domain Map                        │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  MCP Server   │  │  Embedding   │  │  Vector Backend  │  │
│  │  Context      │──│  Context     │──│  Context         │  │
│  │  (7,954 LOC)  │  │  (9,735 LOC) │  │  (6,359 LOC)    │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────────┘  │
│         │                 │                  │              │
│  ┌──────▼───────┐  ┌──────▼───────┐  ┌──────▼───────────┐  │
│  │  Controller   │  │  Search      │  │  Optimization    │  │
│  │  Context      │  │  Context     │  │  Context         │  │
│  │  (21,572 LOC) │  │  (2,571 LOC) │  │  (5,638 LOC)    │  │
│  └──────┬───────┘  └──────────────┘  └──────────────────┘  │
│         │                                                   │
│  ┌──────▼───────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  Integration  │  │  Simulation  │  │  Benchmark       │  │
│  │  Layer        │  │  Context     │  │  Context         │  │
│  │  (8,176 LOC)  │  │  (51,839 LOC)│  │  (16,466 LOC)   │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Bounded Context Inventory

| Context | Files | LOC | DEEP | Role | Quality |
|---------|-------|-----|------|------|---------|
| **MCP Server** | 7 | 7,954 | 4 | Tool registration, request routing, stdio transport | 72-88% |
| **Controllers** | 44 | 21,572 | 16 | Domain logic: Reflexion, Skill, Causal, HNSW, Attention, Learning | 65-92% |
| **Services** | 17 | 9,735 | 6 | EmbeddingService, AttentionService, LLMRouter, Auth | 60-88% |
| **Integration Layer** | 53 | 8,176 | 7 | agentdb-integration: adapters, episodic, MCP tools, bootstrap | 55-85% |
| **Vector Backends** | 20 | 6,359 | 1 | RuVectorBackend, HNSW adapters, vector-backend interface | 72-88% |
| **Optimization** | 12 | 5,638 | 3 | Quantization, BatchOps, SIMD vector ops | 70-85% |
| **Search** | 8 | 2,571 | 1 | HybridSearch, semantic + keyword fusion | 65-78% |
| **Benchmark** | 47 | 16,466 | 3 | Performance suites, ruvector-benchmark | 88-92% |
| **Simulation** | 191 | 51,839 | 3 | Latent-space scenarios, reports, synthetic workloads | 60-75% |

### External Bounded Context: Prime-Radiant (Aspirational Convergence Target)

Prime-radiant is NOT inside `packages/agentdb/` but is the most architecturally ambitious crate in the ecosystem and the intended long-term convergence point. **"The cathedral; the TypeScript layer is the bazaar."**

| Module | Files DEEP | Quality | What It Does |
|--------|-----------|---------|--------------|
| substrate/ | 5+ | ~85% | SheafGraph, SheafNode (StateVector), SheafEdge, RestrictionMap |
| coherence/ | 6 | ~85% | CoherenceEngine: energy minimization. Spectral drift. Real CSR sparse matrices |
| cohomology/ | 9/9 COMPLETE | ~83% | Simplicial complex, cocycles, coboundaries, Hodge Laplacian, diffusion |
| execution/ | 4 COMPLETE | ~89% | Action→Gate→Ladder→Executor. Permit/Defer/Deny. BEST module |
| governance/ | 5/5 COMPLETE | ~88% | PolicyBundle, WitnessRecord (blake3 hash chains). Signatures never verified |
| storage/ | 4/4 COMPLETE | ~86% | FileStorage (WAL), InMemory, Postgres (feature-gated, never wired) |
| hyperbolic/ | 5/5 COMPLETE | ~82% | Poincare ball geometry, Frechet mean. Brute-force search (no HNSW) |
| ruvllm_integration/ | 3+ | ~83% | CoherenceValidator, MemoryLayer, WitnessLog — bridges to ruvllm |

**Why it matters for AgentDB DDD**: Prime-radiant's `ruvllm_integration/` bridges to ruvllm's quality system, and its execution `gate.rs` mirrors `cognitum-gate-tilezero/decision.rs` (same pattern as MCP #7). It has 143 internal dependencies — deeply self-coherent — but zero Cargo.toml workspace deps. It's architecturally aspirational code that can't compile against its intended targets.

**Critical gaps preventing convergence**: No HNSW (O(n) brute-force), no embeddings, no working Postgres, 3 broken Laplacian eigensolvers, `is_coboundary()` always false → all H^n computations inflated. See ADR-MCP-001 Horizon 3 for the 5-step convergence plan.

### Near-Miss External Context: Psycho-Symbolic Reasoner MCP (~#8)

`crates/psycho-symbolic-reasoner/mcp-integration/src/index.ts` — 16 tools using official @modelcontextprotocol/sdk (same as #4). Would be a 5th functional island if wired. Currently disconnected.

---

## 2. Context Details

### 2.1 MCP Server Context (THE CRITICAL PATH)

**Aggregate Root**: `AgentDBMCPServer`
**Location**: `packages/agentdb/src/mcp/agentdb-mcp-server.ts` (2,368 LOC)
**Status**: **THE ONLY WORKING END-TO-END PATH** (R20 confirmed)

#### Tool Groups (27-34 tools)

| Group | Count | Tools | Status |
|-------|-------|-------|--------|
| Core Vector DB | 5 | init, insert, insert_batch, search, delete | WORKING |
| Frontier Memory | 8 | reflexion_store/retrieve, skill_create/search, causal_add_edge/query, recall_with_certificate, learner_discover, db_stats | WORKING |
| Learning System | 10 | start/end_session, predict, feedback, train, metrics, transfer, explain, experience_record, reward_signal | PARTIALLY (9 RL algorithms) |
| AgentDB Core | 5 | stats, pattern_store, pattern_search, pattern_stats, clear_cache | WORKING |
| Batch Ops | 3 | skill_batch, reflexion_batch, pattern_batch | WORKING |
| Attention | 4 | compute, benchmark, configure, metrics | **FABRICATED** (Math.random metrics, R91) |

#### Critical Finding: Attention Tools Are Facades

```
Finding ID 418 (CRITICAL/fabrication):
  attention-tools-handlers.ts L293-299:
  totalCalls = Math.floor(Math.random()*10000) + 1000
  avgLatency = Math.random()*10 + 1
  avgMemory = Math.random()*50 + 10
  successRate = 0.95 + Math.random()*0.05
  cacheHitRate = 0.6 + Math.random()*0.3
  → 100% fabricated metrics returned to MCP consumers
```

Additional attention findings:
- Flash, linear, and performer attention **all compute identical dot product** (R91)
- Sparse attention uses `Math.random() > 0.9` to zero scores — random dropout, not real sparsity
- Handlers exported as **template literal strings, not functions** — defeats TypeScript type system
- Poincare distance produces `Infinity` for normalized vectors (no projection to Poincare ball)
- Config set action merges but **never persists** — next call returns defaults

#### Domain Events

```typescript
// Events emitted by MCP Server context
EmbeddingServiceInitialized { model: 'all-MiniLM-L6-v2', dim: 384 }
ToolRegistered { name: string, group: string }
VectorInserted { id: string, embedding: Float32Array }
SearchExecuted { query: string, k: number, results: number }
EpisodeStored { sessionId: string, task: string, reward: number }
LearningSessionStarted { algorithm: RLAlgorithm }
```

#### Invariants
1. `EmbeddingService` MUST be initialized before any vector operation
2. Episode storage MUST generate real embeddings (not hash fallback)
3. Batch operations MUST use database transactions
4. Security errors MUST be wrapped, not leaked to MCP consumers

---

### 2.2 Embedding Context (THE ROOT CAUSE)

**Aggregate Root**: `EmbeddingService`
**Location**: `packages/agentdb/src/controllers/EmbeddingService.ts` (168 LOC) + `services/enhanced-embeddings.ts` (1,436 LOC)

#### The Dual-Path Architecture

```
                    EmbeddingService
                    /              \
        initialize()           mockEmbedding()
           |                        |
    @xenova/transformers       sin/cos hash
    all-MiniLM-L6-v2         (deterministic but
    384-dim ONNX               semantically void)
           |                        |
       Float32Array            Float32Array
       (REAL)                  (FAKE)
```

**R20 Root Cause**: The claude-flow bridge never calls `initialize()`. It falls back to `mockEmbedding()` silently — no error, no warning, just meaningless vectors.

**R84 Smoking Gun**: `src/cli/commands/install-embeddings.ts` proves embeddings are an **optional manual install** (`claude-flow install-embeddings`). This design decision is the systemic root cause of 14+ hash-based embedding fallbacks across the codebase.

#### Value Objects

```typescript
interface Embedding {
  vector: Float32Array;     // 384-dim from MiniLM or 768/1536 from others
  model: string;            // 'all-MiniLM-L6-v2' | 'mock'
  timestamp: number;
}

interface EmbeddingConfig {
  modelName: string;        // Default: 'Xenova/all-MiniLM-L6-v2'
  dimension: number;        // Default: 384
  cacheEnabled: boolean;
  batchSize: number;
}
```

#### Cross-Context Dependency: The Hash Epidemic

| Consumer | Uses Real? | Falls Back To |
|----------|-----------|---------------|
| AgentDB MCP (#4) | **YES** (auto-init) | N/A |
| claude-flow MCP (#1) | NO | mockEmbedding() |
| agentdb-integration bootstrap | CONDITIONAL | hashEmbedder (SHA-256) |
| attention-tools-handlers | NO | char-code hashing |
| ruvector-backend.ts | NO | Map-based in-memory |
| agentic-flow wrappers | NO | agentdb-service-fallback |

---

### 2.3 Controller Context (Domain Logic)

**16 DEEP controllers** form the richest bounded context. Each controller is a self-contained domain service.

| Controller | LOC | DEEP | Quality | Purpose |
|-----------|-----|------|---------|---------|
| `LearningSystem.ts` | 1,288 | YES | ~75% | 9 RL algorithms: Q-learning, SARSA, DQN, PPO, Actor-Critic, Decision Transformer, MCTS, Policy-Gradient, Model-Based |
| `ReflexionMemory.ts` | 1,115 | YES | ~78% | Episodic memory with reward-weighted retrieval |
| `SkillLibrary.ts` | 925 | YES | ~80% | Skill extraction, search, and reuse |
| `CausalMemoryGraph.ts` | 876 | YES | ~75% | Causal edge tracking with graph queries |
| `AttentionService.ts` (services/) | 1,523 | YES | ~65% | Multi-head attention (BROKEN: fabricated metrics) |
| `AttentionService.ts` (controllers/) | 771 | YES | ~60% | Duplicate of above (ANTI-PATTERN) |
| `HNSWIndex.ts` | 582 | YES | ~85% | Hierarchical Navigable Small World index |
| `NightlyLearner.ts` | 665 | YES | ~72% | Offline learning from accumulated episodes |
| `ReasoningBank.ts` | 676 | YES | ~78% | Pattern storage and retrieval |
| `ExplainableRecall.ts` | 747 | YES | ~80% | Recall with provenance certificates |
| `MemoryController.ts` | 462 | YES | ~72% | Application-layer memory orchestration |
| `SyncCoordinator.ts` | 717 | YES | ~68% | QUIC-based multi-agent sync |
| `QUICClient.ts` | 668 | YES | ~65% | QUIC transport client |
| `MultiHeadAttentionController.ts` | 494 | YES | ~62% | Dead in production pipeline (R112) |
| `CrossAttentionController.ts` | 467 | YES | ~62% | Dead in production pipeline (R114) |

#### Critical Anti-Patterns Detected

1. **Duplicate AttentionService**: Two files (`services/AttentionService.ts` and `controllers/AttentionService.ts`) with overlapping responsibility — 2,294 combined LOC
2. **Both attention controllers dead**: `MultiHeadAttentionController` and `CrossAttentionController` are architecturally present but unreachable from the MCP tool pipeline (R112, R114)
3. **NightlyLearner never scheduled**: Offline learning requires external cron trigger that doesn't exist in deployment config
4. **SyncCoordinator/QUICClient**: QUIC transport is **genuine in Rust** (quinn) but **stub in TypeScript** (R40) — the TS QUIC types are aspirational
5. **Two parallel episodic memory systems** (R104): `context_manager` composes only 2/5 siblings. Episodes stored in one system invisible to the other. Two independent episode aggregates with no shared identity.
6. **4+ independent HNSW indexes never compose**: Each subsystem creates its own HNSW instance. Vectors stored in one are invisible to queries against another. Distinct from the embedding problem — even with real embeddings, index fragmentation prevents cross-subsystem search.
7. **Zero inter-MCP composition primitives**: None of the 7 MCP servers have a tool-routing protocol, shared state bus, or event bridge. Consolidation requires building composition from scratch — there is no "just wire them together" path.

---

### 2.4 Integration Layer Context

**Location**: `agentdb-integration/` (53 files, 8,176 LOC, 7 DEEP)

This is the **bridge layer** between AgentDB native and claude-flow. It follows Clean Architecture with adapters, services, and repositories.

#### Architecture

```
agentdb-integration/
├── episodic/
│   ├── adapters/
│   │   └── reflexion-memory-adapter.ts    (328 LOC, DEEP)
│   ├── aggregates/
│   │   └── episode.ts
│   ├── repositories/
│   │   └── episode-repository.ts
│   └── services/
│       └── reflexion-service.ts           (330 LOC, DEEP)
├── infrastructure/
│   ├── adapters/
│   │   ├── real-embedding-adapter.ts      (153 LOC, DEEP)
│   │   ├── ruvector-backend-adapter.ts    (374 LOC, DEEP)
│   │   └── vector-backend-adapter.interface.ts (49 LOC, DEEP)
│   └── jobs/
│       └── vector-migration-job.ts        (203 LOC, DEEP)
└── mcp-tools/
    ├── mcp-reflexion-retrieve.ts          (93 LOC, DEEP)
    └── mcp-reflexion-store.ts             (92 LOC, DEEP)
```

#### Key Finding: Dual Retrieval Architecture (R65)

The `reflexion-memory-adapter.ts` implements **intentional dual retrieval**:
1. **Vector search path** (L144-167): Uses `embedder.embed()` → `vectorBackend.searchAsync()`
2. **Controller fallback** (L217-245): Direct controller access when vector search unavailable

This is NOT a bug — it's the design. But the fallback is hash-based, defeating semantic search.

#### Integration Invariant

```
bootstrap.ts: episodicEmbedder = realEmbedder ?? hashEmbedder
```

If `@xenova/transformers` is unavailable, the entire episodic system silently degrades to hash-based retrieval. No error, no metric, no signal to the user.

---

### 2.5 Vector Backend Context

**Location**: `packages/agentdb/src/backends/` (20 files, 6,359 LOC, 1 DEEP)

| Backend | LOC | Status | Notes |
|---------|-----|--------|-------|
| `ruvector/RuVectorBackend.ts` | 971 (DEEP) | 72-78% (R91) | Real HNSW via hnswlib-node. DOWN from initial assessment |
| HNSW adapters | ~2,000 | Various | hnswlib-node bindings |
| SQLite backend | ~1,500 | Working | better-sqlite3 persistence |
| Memory backend | ~800 | Working | In-memory for testing |

#### RuVectorBackend Quality Issues (R91)

- Genuine HNSW with real cosine similarity search
- BUT: Quality downgraded to 72-78% due to missing batch optimization
- No connection to Rust `ruvector-core` SIMD (despite name implying it)
- Uses pure JavaScript vector operations — no AVX/NEON acceleration

#### The VectorBackendAdapter Contract (R65)

```typescript
interface VectorBackendAdapter {
  // Required
  addVector(id: string, vector: number[], metadata?: any): Promise<void>;
  search(query: number[], k: number, filter?: any): Promise<SearchResult[]>;
  removeVector(id: string): Promise<void>;
  initialize(config: any): Promise<void>;
  getStats(): Promise<BackendStats>;
  // Optional
  batchAdd?(vectors: VectorBatch[]): Promise<void>;
  optimize?(): Promise<void>;
}
```

This is the **foundational contract** for all AgentDB vector persistence. It's adapter-agnostic (no embedding dimension spec), which is both a strength (flexibility) and weakness (no type safety on dimensionality).

---

### 2.6 Search Context

**Location**: `packages/agentdb/src/search/` (8 files, 2,571 LOC, 1 DEEP)

| Component | LOC | Status |
|-----------|-----|--------|
| `HybridSearch.ts` | 1,062 (DEEP) | 65-78% |
| Keyword search | ~500 | Working |
| Semantic search | ~600 | Depends on EmbeddingService |
| Fusion layer | ~400 | Score normalization |

HybridSearch combines keyword (BM25-like) and semantic (vector cosine) results using reciprocal rank fusion. Quality drops to ~65% because the semantic path requires real embeddings — when running through claude-flow main MCP, it degrades to hash-based "semantic" search that returns random-seeming results.

---

### 2.7 Optimization Context

**Location**: `packages/agentdb/src/optimizations/` + `src/quantization/` + `src/simd/` (12 files, 5,638 LOC, 3 DEEP)

| Component | LOC | Status |
|-----------|-----|--------|
| `Quantization.ts` | 996 (DEEP) | ~78% |
| `BatchOperations.ts` | 809 (DEEP) | ~75% |
| `vector-quantization.ts` | 1,529 (DEEP) | ~72% |
| `simd-vector-ops.ts` | 1,287 (DEEP) | ~70% |

**Key Finding**: SIMD vector ops are **JavaScript-only** approximations, not actual SIMD. The name is misleading. Real SIMD exists only in the Rust `ruvector-core` crate (AVX-512/AVX2/NEON).

---

## 3. Aggregate Boundaries & Invariants

### 3.1 AgentDB Core Aggregate

```
AgentDBMCPServer (Root)
  ├── EmbeddingService         [MUST init before ops]
  ├── HNSWIndex                [Real HNSW via hnswlib-node]
  ├── MemoryController         [Orchestrates controllers]
  ├── ReflexionMemory          [Episodic store/retrieve]
  ├── SkillLibrary             [Skill extraction]
  ├── CausalMemoryGraph        [Graph queries]
  ├── ReasoningBank            [Pattern store]
  ├── LearningSystem           [9 RL algorithms]
  └── ExplainableRecall        [Provenance certificates]
```

### 3.2 Cross-Aggregate Invariants

| Invariant | Enforced? | Evidence |
|-----------|-----------|----------|
| EmbeddingService initialized before vector ops | **YES** (in #4 only) | agentdb-mcp-server.ts L196-201 |
| Episode storage generates real embeddings | **CONDITIONAL** | bootstrap.ts falls back to hash |
| Batch ops use DB transactions | **YES** | agentdb-mcp-server.ts |
| Attention metrics are real | **NO** | Math.random() fabrication (R91) |
| QUIC sync is functional | **NO** | TS stub, Rust genuine (R40) |
| Learning session state persists | **PARTIAL** | In-memory maps, no cross-session |

---

## 4. Context Integration Map

### 4.1 Upstream/Downstream Relationships

```
[claude-flow main MCP #1]  ──(degraded)──▶  [AgentDB via fallback]
                                                    │
[AgentDB Native MCP #4]  ──(working)──▶  [EmbeddingService]
         │                                          │
         ├──▶ [Controllers]                         │
         ├──▶ [Vector Backend] ◀──(real embed)──────┘
         ├──▶ [Search Context]
         └──▶ [Optimization]

[Integration Layer]  ──(bridge)──▶  [AgentDB Native]
         │
         └──(conditional)──▶  [Real Embedding Adapter]
                                       │
                              [Hash Embedding Adapter] (fallback)

[mcp-gate #7]  ──(no connection)──▶  [AgentDB]
[ruv-swarm #5]  ──(no connection)──▶  [AgentDB]
[ReasoningBank #6]  ──(no connection)──▶  [AgentDB]
[psycho-symbolic ~#8]  ──(no connection)──▶  [AgentDB]

[prime-radiant]  ──(aspirational)──▶  [ruvllm]
         │
         └──(ruvllm_integration/ refs types)──▶  [ruvllm memory types]
              BUT Cargo.toml doesn't declare dep → can't compile
```

### 4.2 Anti-Corruption Layers Needed

1. **MCP Composition Layer**: Build inter-MCP tool routing protocol — no composition primitives exist today. Must route vector/memory tool calls from #1 → #4 without duplicating tool registration. Consider event bridge for cross-server state.
2. **Embedding Guarantee**: Reject operations when EmbeddingService is in mock mode (no silent degradation)
3. **HNSW Federation Layer**: Cross-index query router so vectors in one HNSW instance are discoverable from another. Alternatively, consolidate all into shared ruvector-core instance.
4. **Episodic Memory Reconciliation**: Bridge between two parallel episodic systems (R104) — single episode identity across both stores.
5. **Attention Facade Firewall**: Replace fabricated attention metrics with real measurements or honest "not implemented" responses
6. **QUIC Transport Adapter**: Abstract TS QUIC stubs behind interface matching Rust quinn implementation
7. **Prime-Radiant Bridge**: Typed adapter between prime-radiant's `StateVector` world and AgentDB's `Float32Array` embedding world — dimension mismatch (64 vs 384) must be resolved at this boundary.

---

## 5. The 13 Disconnected Persistence Layers

Each independently stores data with no reconciliation mechanism:

| # | Location | Storage | Connected To |
|---|----------|---------|-------------|
| 1 | AgentDB SQLite (episode_embeddings) | better-sqlite3 | MCP #4 only |
| 2 | AgentDB HNSW index (hnswlib-node) | File-backed index | MCP #4 only |
| 3 | claude-flow memory CLI | JSON file (.claude-flow/data/) | Nothing (R84) |
| 4 | ReasoningBank JS (queries.js) | .swarm/memory.db | Nothing (R73) |
| 5 | ReasoningBank hooks (post-task) | patterns/pattern_embeddings tables | Nothing (R73) |
| 6 | agentic-flow long-running-agent | In-memory array | Nothing (R41) |
| 7 | worker-agent-integration | In-memory Maps | Nothing |
| 8 | ReasoningBank Rust workspace | Separate SQLite | Nothing (R78) |
| 9 | .claude/helpers/memory.js | JSON file | Nothing (R84) |
| 10 | prime-radiant FileStorage | WAL-backed file | Nothing (R108) |
| 11 | prime-radiant InMemoryStorage | Heap | Nothing (test only) |
| 12 | prime-radiant PostgresStorage | Feature-gated, never wired | Nothing (R107) |
| 13 | policy_store.rs | Cache-only (broken get) | Nothing (R114) |

**Consolidation Target**: Layers 1-2 (AgentDB native) should be the canonical persistence, with others migrating data through the `VectorBackendAdapter` interface.

---

## 6. Ubiquitous Language

| Term | Definition | Context |
|------|-----------|---------|
| **Episode** | A task execution record with input, output, reward, critique, and embedding | Reflexion Memory |
| **Skill** | An extracted reusable capability from successful episodes | Skill Library |
| **Embedding** | A 384-dim Float32Array from all-MiniLM-L6-v2 (or fake hash) | Embedding Service |
| **Vector Backend** | Abstraction over HNSW index + persistence | Backend Context |
| **Reflexion** | Store-retrieve loop for experience replay (NOT genuine self-reflection — R65) | Integration Layer |
| **Causal Edge** | Directed dependency between episodes in the causal graph | Causal Memory |
| **Recall Certificate** | Provenance chain showing why a memory was retrieved | Explainable Recall |
| **Frontier Memory** | Combined episodic + skill + causal memory system | MCP Tool Group |
| **Mock Embedding** | sin/cos hash fallback that produces semantically meaningless vectors | THE BUG |
| **Hash Embedder** | SHA-256 based embedding fallback in integration layer | THE OTHER BUG |
| **Nightly Learner** | Offline batch processing of accumulated episodes | Controller (unscheduled) |
| **Pattern Store** | Key-value store for learned patterns with vector search | ReasoningBank |

---

## 7. Recommended DDD Refactoring

### 7.1 Phase 1: Enforce Embedding Invariant
- Remove silent fallback to `mockEmbedding()`
- Fail loudly when `EmbeddingService.initialize()` hasn't been called
- Add health check endpoint to MCP server

### 7.2 Phase 2: Merge Duplicate Contexts
- Merge `services/AttentionService.ts` and `controllers/AttentionService.ts`
- Remove fabricated attention metrics entirely
- Either implement real attention or remove the tools

### 7.3 Phase 3: Anti-Corruption Layer for MCP Router
- Implement MCP tool routing: vector/memory → #4, session/config → #1
- Add OpenTelemetry tracing across MCP boundaries
- Implement circuit breaker for embedding model cold-start

### 7.4 Phase 4: Persistence Consolidation
- Migrate layers 3-9 to use `VectorBackendAdapter` from #4
- Implement event-sourced sync between AgentDB SQLite and external consumers
- Add CDC (Change Data Capture) for cross-layer consistency

---

## 8. Appendix: Session Evidence Chain

| Session | Key Discovery |
|---------|--------------|
| R6 | Main MCP has 256 tools, many degraded to no-ops |
| R18 | AgentDB integration deep-read — found native MCP architecture |
| R20 | **ROOT CAUSE**: EmbeddingService never initialized in claude-flow bridge |
| R25 | AgentDB TS sources, claude-flow memory subsystem mapped |
| R40 | LLMRouter NOT connected to ADR-008. Real QUIC in Rust only |
| R51 | Servers #2, #3 confirmed as CLI wrappers via execSync |
| R63 | AgentDB bridge + swarm runtime integration mapped |
| R65 | Dual retrieval architecture (vector + controller fallback) confirmed |
| R72 | ruv-swarm Rust MCP (#5) identified as distinct SDK |
| R78 | ReasoningBank Rust MCP (#6) hand-rolled, zero integration |
| R84 | `install-embeddings.ts` proves embeddings are optional manual install |
| R91 | Attention metrics 100% fabricated (Math.random). RuVectorBackend DOWN |
| R96 | MemoryController — 10th disconnected persistence layer |
| R104 | `claude_flow_bridge.rs` imports zero vector/HNSW code — spawns subprocess |
| R112 | Both attention controllers dead in production pipeline |
| R114 | mcp-gate crate COMPLETE at 91%. 7th MCP confirmed. 13th persistence layer |
