# AgentDB Package Analysis

**Package**: `agentdb` on npm (inside agentic-flow monorepo at `packages/agentdb`)
**Version**: 2.0.0-alpha.3.4 (installed), 92 versions published (1.0.0 through 2.0.0-alpha.3.4)
**Size**: 68 MB installed
**Language**: TypeScript compiled to JavaScript
**License**: MIT
**Analysis Date**: 2026-02-08

---

## Executive Summary

AgentDB is a **frontier memory database for AI agents** with 23 controllers, 33+ MCP
tools, 24 simulation scenarios, a browser build, and comprehensive security middleware.
It is the richest single package in the ruvnet ecosystem in terms of genuine, implemented
algorithms.

Key findings:

1. **18 of 23 controllers are GENUINE implementations** — not stubs. They reference
   real papers (Reflexion, Voyager, Flash Attention, MMR) and implement the algorithms.
2. **3 controllers are PARTIAL** — QUIC (TCP, not real QUIC), WASMVectorSearch
   (missing WASM module), LearningSystem (tabular RL with misleading deep RL names).
3. **Zero controllers are stubs** — every controller has functional code.
4. **Security is defense-in-depth** — 4 security layers, whitelist SQL/Cypher injection
   prevention, Argon2id auth, path traversal hardening, rate limiting, audit logging.
5. **Claude-flow uses NONE of it** — zero imports from agentdb in claude-flow source.
   Claude-flow has its own simpler ReasoningBank, memory system, and hooks.

---

## 1. Architecture Overview

### Core Class: `AgentDB`
- 184 lines of TypeScript (`src/core/AgentDB.ts`)
- Initializes: better-sqlite3 (WAL mode) with sql.js WASM fallback
- Vector backend: auto-selects `@ruvector/core` > `hnswlib-node` > brute-force
- Default embedding: MiniLM-L6-v2 via `@xenova/transformers` (384 dimensions)
- Controllers initialized: ReflexionMemory, SkillLibrary, CausalMemoryGraph
- Config: `dbPath`, `namespace`, `forceWasm`, `vectorBackend`, `vectorDimension`

### Backend Factory (`backends/factory.js`)
Priority chain: `ruvector` → `hnswlib` → brute-force fallback

| Backend | Implementation | Speed |
|---------|---------------|-------|
| RuVectorBackend | `@ruvector/core` native Rust HNSW | 150x vs SQLite |
| HNSWLibBackend | `hnswlib-node` (lazy-loaded) | ~60x vs SQLite |
| Brute-force | JS cosine similarity loop | 1x baseline |

### Entry Point Exports
155 lines of exports covering: 8 core controllers, 3 attention controllers,
embedding services, WASM search, quantization (8-bit/4-bit/PQ), hybrid search
(BM25 + vector), benchmarking suite, coordination layer, LLM router, security
validators.

---

## 2. All 23 Controllers — Detailed Assessment

### Research-Grounded Controllers (Fully Implemented)

| # | Controller | Lines | Paper/Algorithm | Assessment |
|---|-----------|-------|----------------|------------|
| 1 | ReflexionMemory | 815 | Reflexion (Shinn et al., arxiv:2303.11366) | **GENUINE** |
| 2 | SkillLibrary | 697 | Voyager (Wang et al., arxiv:2305.16291) | **GENUINE** |
| 3 | CausalMemoryGraph | 602 | Doubly Robust Estimator (Robins et al., 1994) | **GENUINE** |
| 4 | NightlyLearner | 533 | Doubly Robust causal discovery | **GENUINE** |
| 5 | AttentionService | 517 | Flash Attention + GraphRoPE + MoE | **GENUINE** |
| 6 | ExplainableRecall | 517 | Minimal Hitting Set + Merkle provenance | **GENUINE** |
| 7 | ReasoningBank | 494 | Pattern storage + EWC consolidation | **GENUINE** |
| 8 | CausalRecall | 355 | Utility-based reranking (alpha*sim + beta*uplift) | **GENUINE** |
| 9 | SelfAttentionController | 204 | Scaled Dot-Product (Vaswani 2017) | **GENUINE** |
| 10 | CrossAttentionController | 324 | Cross-Attention (Vaswani 2017) | **GENUINE** |
| 11 | MultiHeadAttentionController | 335 | Multi-Head Attention (Vaswani 2017) | **GENUINE**¹ |
| 12 | MemoryController | 302 | Unified memory + attention orchestrator | **GENUINE** |
| 13 | HNSWIndex | 436 | HNSW (Malkov & Yashunin, 2018) | **GENUINE** |
| 14 | LearningSystem | 928 | Q-Learning/SARSA/UCB1 (tabular RL) | **GENUINE**² |
| 15 | MMRDiversityRanker | 129 | MMR (Carbonell & Goldstein, 1998) | **GENUINE** |
| 16 | MetadataFilter | 242 | MongoDB-style query operators | **GENUINE** |
| 17 | ContextSynthesizer | 207 | Heuristic synthesis | **GENUINE** |
| 18 | EmbeddingService | 141 | transformers.js / OpenAI | **GENUINE** |

¹ Random projections, never trained — provides diversity but not learned attention.
² Claims 9 RL algorithms but DQN/PPO/Actor-Critic/DecisionTransformer reduce to tabular.

### Partially Implemented Controllers

| # | Controller | Lines | Issue |
|---|-----------|-------|-------|
| 19 | QUICServer | 382 | TCP, not real QUIC (RFC 9000) |
| 20 | QUICClient | 488 | TCP client, not QUIC |
| 21 | WASMVectorSearch | 335 | WASM module doesn't exist; "ANN index" is brute-force |
| 22 | SyncCoordinator | 552 | Real sync logic, but depends on fake QUIC |
| 23 | EnhancedEmbeddingService | 118 | Thin wrapper, WASM acceleration unavailable |

### Total: 9,697 lines across 23 controllers

---

## 3. Strongest Subsystems

### Causal Inference Pipeline
`CausalMemoryGraph → NightlyLearner → CausalRecall → ExplainableRecall`

This is a cohesive, well-engineered pipeline:
- **CausalMemoryGraph**: A/B experiment lifecycle, doubly robust estimation, knowledge graph
- **NightlyLearner**: Background job discovering causal edges, calculating uplift, pruning
- **CausalRecall**: Utility formula `U = α*similarity + β*uplift - γ*latencyCost`
- **ExplainableRecall**: Merkle-proof provenance chains, minimal hitting set justifications

### Reflexion + Voyager Core
- **ReflexionMemory** (815 lines): Episodic memory with self-critique, based on the Reflexion
  paper's cycle of act → reflect → critique → improve
- **SkillLibrary** (697 lines): Skill extraction, graph-based relationships, consolidation,
  based on Voyager's automated curriculum

### Attention Subsystem
- Three controllers (Self, Cross, MultiHead) implement textbook attention correctly
- AttentionService adds Flash Attention (block-wise), GraphRoPE, and MoE routing
- All pure JavaScript on Float32Arrays — correct algorithms, no GPU

---

## 4. CLI and MCP Tools

### CLI Commands (30+ top-level)
Major categories: `mcp` (server), `init`, `status`, `doctor`, `migrate`,
`vector-search`, `export`/`import`, `simulate`, `attention`, `causal`,
`recall`, `learner`, `reflexion`, `skill`, `sync`, `train`, `optimize-memory`

### MCP Server (33+ tools)
Organized into categories:

| Category | Tools | Key Operations |
|----------|-------|----------------|
| Core Vector DB | 5 | init, insert, insert_batch, search, delete |
| Frontier Memory | 7 | reflexion_store/retrieve, skill_create/search, causal_add_edge/query, recall_with_certificate |
| Learning v1.3 | 5 | start_session, end_session, predict, feedback, train |
| Learning v1.4 | 5 | metrics, transfer, explain, experience_record, reward_signal |
| Core v1.3 | 5 | stats, pattern_store/search/stats, clear_cache |
| Batch Ops | 3 | skill_create_batch, reflexion_store_batch, pattern_store_batch |
| Attention | 3 | compute, benchmark, optimize |

---

## 5. Simulation Framework

24 scenarios across 3 directories:

### Core Scenarios (17)
- `multi-agent-swarm`: WAL concurrency testing with N agents
- `reflexion-learning`: Episodic memory store/retrieve cycle
- `consciousness-explorer`: Multi-layered consciousness model
- `stock-market-emergence`: 100-agent trading (momentum, value, contrarian, HFT)
- `research-swarm`: Distributed literature review + hypothesis generation
- Plus: causal-reasoning, graph-traversal, skill-evolution, strange-loops, etc.

### Latent Space Scenarios (8)
attention-analysis, clustering, HNSW exploration, hypergraph, neural-augmentation,
quantum-hybrid, self-organizing-hnsw, traversal-optimization

### Domain Examples (6)
e-commerce recommendations, IoT sensor networks, medical imaging,
robotics navigation, scientific research, trading systems

---

## 6. Browser Build

Self-contained UMD bundle (~35KB minified, ~12KB gzipped) with:
- ProductQuantization (PQ8/PQ16/PQ32, 4-32x memory reduction)
- HNSWIndex (10-20x faster search)
- GraphNeuralNetwork, MaximalMarginalRelevance, TensorCompression
- AttentionBrowser with Flash Attention
- Feature detection (IndexedDB, BroadcastChannel, WebWorkers, WASM SIMD)
- 6 configuration presets (small/medium/large/memory/speed/quality optimized)
- Lazy-loaded WASM module (~157KB) with mock fallback

---

## 7. Security Architecture

### Defense in Depth — 4 Layers

**Layer 1: Input Validation** (`security/input-validation.js`)
- Whitelist-based table/column/pragma validation (14 allowed tables)
- XSS pattern detection (`<script`, `javascript:`, `on*=` event handlers)
- Null byte rejection, control character filtering
- Parameterized SQL building (`buildSafeWhereClause`, `buildSafeSetClause`)

**Layer 2: Vector Validation** (`security/validation.js`)
- MAX_VECTORS: 10M, MAX_DIMENSION: 4096, MAX_BATCH_SIZE: 10K
- NaN/Infinity checking on every vector element
- Path traversal prevention in vector IDs
- Cypher injection prevention (blocks `'`, `"`, `;`, `{`, `}`)
- Metadata sanitization (strips 14 sensitive field patterns)

**Layer 3: Path Security** (`security/path-security.js`)
- Resolves to absolute paths, rejects `..` traversal
- Symlink protection (rejects symlink writes without explicit flag)
- Atomic writes via temp file + rename
- TempFileManager with process exit cleanup

**Layer 4: Resource Limits** (`security/limits.js`)
- ResourceTracker: 16GB memory cap
- RateLimiter: Token bucket (insert=100/s, search=1K/s, delete=50/s)
- CircuitBreaker: 5 failures → open, 60s reset
- 30s query timeout

### Authentication
- **Auth Service**: Argon2id password hashing (64MB memory, 3 iterations)
- **Token Service**: JWT HS256, 15-min access / 7-day refresh tokens
- **API Keys**: `agdb_live_<64 hex>` / `agdb_test_<64 hex>` format
- **Rate Limiting Middleware**: 7 Express rate limiters (general=100/15min, auth=5/15min)
- **Security Headers**: Helmet.js with strict CSP, HSTS, permissions policy

### Security Strengths
1. Prototype pollution prevention (`__proto__`, `constructor`, `prototype` blocked)
2. Constant-time comparison for auth (crypto.timingSafeEqual)
3. AES-256-GCM encryption available
4. PBKDF2 key derivation (100K iterations)
5. Structured audit logging with auto-rotation

### Security Weaknesses
1. In-memory auth stores (users, sessions, revocations) — not persisted
2. API key validation is O(n) over all stored keys
3. Service account tokens have no expiration
4. Mock embeddings in AgentDB-Fast are non-semantic

---

## 8. Wrappers and Fallbacks

### GNN Wrapper (`wrappers/gnn-wrapper.js`)
Fixes broken `@ruvector/gnn` APIs:
- Auto-converts `number[]` to `Float32Array` for native module
- Falls back to pure JS matrix multiplication when native fails
- `RuvectorLayer`: neural layer with Xavier init, 4 activations
- `TensorCompress`: 5 compression levels (none/half/pq8/pq4/binary)

### Attention Fallbacks (`wrappers/attention-fallbacks.js`, 1,484 lines)
Largest wrapper — provides pure JS alternatives because `@ruvector/attention` is broken:
- MultiHeadAttention (standard + 8x unrolled optimized)
- FlashAttention (tiled/chunked + online softmax optimized with backward pass)
- LinearAttention (O(n) with ELU feature map)
- HyperbolicAttention (Poincare ball model)
- MoEAttention (Mixture of Experts with top-k gating)
- BufferPool: Float32Array reuse (64MB cap, 64 buffers/dimension)

### AgentDB-Fast (`wrappers/agentdb-fast.js`)
Bypasses CLI overhead:
- CLI: ~2,350ms/op → Direct API: ~10-50ms/op (50-200x speedup)
- Default embeddings are mock hash-based (NOT semantic)
- Episode + Pattern storage via vector backend directly

### Services Layer (7 services)
1. **AttentionService** (50KB): 4 attention mechanisms with NAPI > WASM > JS fallback
2. **LLMRouter** (20KB): 5 providers (RuvLLM, OpenRouter, Gemini, Claude, ONNX)
3. **Federated Learning** (11KB): EphemeralLearningAgent + FederatedCoordinator
4. **Auth Service** (13KB): Argon2id, API keys, session management
5. **Token Service** (11KB): JWT with rotation and revocation
6. **Audit Logger** (10KB): Structured logging with SOC2/GDPR/HIPAA claims
7. **Enhanced Embeddings** (35KB): Multi-provider with LRU cache (100K entries)

---

## 9. Hybrid Search (Vector + Keyword)

Combines vector similarity with BM25 keyword search:

| Fusion Method | Description |
|--------------|-------------|
| **RRF** (default) | Reciprocal Rank Fusion: `RRF(d) = Σ(weight / (60 + rank(d)))` |
| **Linear** | `0.7 * vectorScore + 0.3 * keywordScore` |
| **Max** | `max(vectorScore, keywordScore)` per document |

KeywordIndex: Full BM25 with tokenization, stopword removal, inverted index.

---

## 10. Quantization

### Scalar Quantization
- 8-bit: `quantize8bit()` / `dequantize8bit()` — 4x compression
- 4-bit: `quantize4bit()` / `dequantize4bit()` — 8x compression

### Product Quantization
- `ProductQuantizer`: Configurable subvectors, centroids, codebook training
- Factory functions: `createScalar8BitStore()`, `createScalar4BitStore()`,
  `createProductQuantizedStore()`
- `QuantizedVectorStore`: Full CRUD with quantized storage

---

## 11. npm Version History

| Range | Count | Notes |
|-------|-------|-------|
| 1.0.x | 12 | Initial release series |
| 1.1.x | 10 | Feature additions |
| 1.2.x | 1 | Single release |
| 1.3.x | 18 | Major feature period (RL, learning) |
| 1.4.x | 14 | Learning v2, attention |
| 1.5.x | 10 | QUIC, sync, coordination |
| 1.6.x | 2 | Pre-alpha stabilization |
| 2.0.0-alpha | 26 | Major rewrite (current) |

92 total versions. Rapid iteration — 26 alpha releases in the 2.0 series alone.

---

## 12. What Claude-Flow Actually Uses from AgentDB

**Answer: Nothing directly.**

Claude-flow has its own:
- `LocalReasoningBank` (in-memory Map + JSON file) — not agentdb's ReasoningBank
- Memory system backed by `~/.swarm/memory.db` (its own SQLite schema)
- Hooks pipeline for learning — not agentdb's LearningSystem
- Agent templates — not agentdb's simulation scenarios

AgentDB is installed as a transitive dependency via agentic-flow, but claude-flow
imports zero modules from it. The agentdb `[AgentDB Patch] Controller index not
found` warning on startup confirms the package loads but is unused.

---

## 13. Interplay with Other ruvnet Packages

```
AgentDB 2.0.0-alpha.3.4
├── Uses: better-sqlite3 (relational storage)
├── Uses: @ruvector/core (HNSW vector search via RuVectorBackend)
├── Uses: @ruvector/gnn (wrapped by gnn-wrapper, broken native)
├── Uses: @ruvector/attention (wrapped by attention-native, broken native)
├── Uses: @xenova/transformers (local embeddings)
├── Uses: hnswlib-node (HNSWLibBackend fallback)
├── Uses: sql.js (WASM SQLite fallback)
├── Bundled by: agentic-flow (in packages/agentdb)
├── Loaded by: claude-flow (as transitive dep, UNUSED)
└── Has its own: ReasoningBank (3rd implementation, separate from agentic-flow's and claude-flow's)
```

### The Three ReasoningBanks Problem (Updated)

| ReasoningBank | Location | Backing Store | Lines | Used? |
|--------------|----------|--------------|-------|-------|
| claude-flow's | `dist/src/mcp-tools/hooks-tools.js` | In-memory Map + JSON | ~200 | YES |
| agentic-flow's | `dist/reasoning-bank.js` | SQLite via DeepMind paper | ~600 | Partially |
| agentdb's | `controllers/ReasoningBank.js` | SQLite + embeddings | ~494 | NO |

---

## 14. Missing Functionality / Installation Gaps

| Feature | Status | Impact |
|---------|--------|--------|
| WASM module (`reasoningbank_wasm.js`) | Does not exist | WASMVectorSearch falls back to JS |
| `@ruvector/gnn` native | Broken API | GNN wrapper provides JS fallback |
| `@ruvector/attention` native | Broken | attention-fallbacks.js provides JS (1,484 lines) |
| `hnswlib-node` | Not installed | HNSWLibBackend unavailable, falls to RuVector or brute-force |
| `@xenova/transformers` model | Download fails | EmbeddingService uses mock embeddings |
| QUIC protocol | TCP implementation | Not real QUIC (RFC 9000) |
| In-memory auth stores | Not production-ready | Users, sessions, tokens lost on restart |

---

## 15. Verdict

AgentDB is the **most genuinely implemented** package in the ruvnet ecosystem.
Unlike agentic-flow (which has many stubs) or ruvector (which wraps hnsw_rs),
agentdb's controllers implement real algorithms from real papers. The causal
inference pipeline, attention subsystem, and RL system represent thousands of
lines of working code.

However, none of it is used by claude-flow. The package exists as a transitive
dependency that loads, prints a warning, and does nothing. If claude-flow's V3
roadmap includes "frontier memory" features (ADR-006, ADR-009), agentdb is the
most ready-to-integrate component — particularly its ReflexionMemory,
SkillLibrary, and HybridSearch.

**Sophistication Rating**: 7/10 — Genuine algorithms with real paper references,
but hampered by broken native dependencies, missing WASM modules, and in-memory
auth stores.
