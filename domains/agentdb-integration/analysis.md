# AgentDB Integration Domain Analysis

> **Priority**: MEDIUM | **Coverage**: ~15.3% (79/517 DEEP) | **Status**: In Progress
> **Last updated**: 2026-02-15 (Session R41)

## 1. Current State Summary

AgentDB is a 507-file / 153K LOC vector database with agent learning capabilities. Despite claude-flow listing it as an optional dependency, **none of its 23 controllers are called** — 140K+ LOC of genuinely sophisticated code sits unused. The codebase is **~90% authentic** with production-grade search algorithms, real security implementation, and working neural attention mechanisms.

**Top-level verdicts:**

- **Best-in-ecosystem search implementation**: HybridSearch (BM25 + HNSW + fusion strategies) is production-ready and surpasses all other ruvnet search code.
- **Production-grade quantization**: K-means++ PQ with 8/4-bit scalar quantization rivals standalone vector databases.
- **Genuine neural attention**: MultiHead and CrossAttention controllers implement real transformer-style attention from scratch (inference-only, random weights).
- **Solid security model**: Argon2id hashing, SQL injection whitelists, JWT tokens, brute-force protection — comprehensive and correct.
- **Complete facade in MCP tools layer**: Goalie subsystem (856 LOC) imports GoapPlanner and reasoning engines but calls NONE of them.
- **Systemic embedding degradation**: Hash-based embedding fallback silently breaks all semantic search features.
- **Critical bugs in core controllers**: LearningSystem RL is cosmetic (9 algorithms → 1 implementation), CausalMemoryGraph statistics are mathematically wrong.
- **Three parallel AgentDB systems**: Native standalone MCP server, agentic-flow wrapper, claude-flow patched bridge — only native works correctly.

The integration gap is organizational, not technical. AgentDB quality exceeds the rest of the ruvnet ecosystem across search, quantization, security, and attention.

## 2. File Registry

### AgentDB CLI & MCP Server

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| agentdb-cli.js | agentdb | 3,039 | 95% | DEEP | 14 commands + 60+ subcommands, EmbeddingService initialized | R32 |
| agentdb-mcp-server.js | agentdb | 2,368 | 98% | DEEP | 27+ tools fully implemented, correct EmbeddingService | R32 |
| agentdb-cli.ts | agentdb | 3,422 | 98% | DEEP | Complete command surface, 35+ subcommands | R16 |

### Core Controllers

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| ReasoningBank.ts | agentdb | 676 | 98% | DEEP | Canonical implementation, v1/v2 dual-mode | R16 |
| HNSWIndex.ts | agentdb | 582 | 96% | DEEP | Wraps hnswlib-node C++ library, lazy-loading | R16, R40 |
| SkillLibrary.ts | agentdb | 925 | 90% | DEEP | Composite scoring, pattern extraction is TF word counting | R16 |
| ExplainableRecall.ts | agentdb | 747 | 88% | DEEP | Merkle tree provenance, greedy set cover | R16, R22 |
| NightlyLearner.ts | agentdb | 665 | 80% | DEEP | SQL path works, attention path broken by R20 | R16, R40 |
| LearningSystem.ts | agentdb | 1,288 | 55% | DEEP | 9 RL algorithms = 1 Q-value dict, no neural nets | R8, R22 |
| CausalMemoryGraph.ts | agentdb | 876 | 65% | DEEP | Wrong t-CDF formula, fake correlation via session count | R8, R22 |
| MemoryController.ts | agentdb | 462 | 95% | DEEP | Attention orchestration, temporal decay weighting | R16 |
| ReflexionMemory.ts | agentdb | 1,115 | 65% | DEEP | Storage works, missing judge function (breaks arXiv paper) | R8 |

### Search & Optimization

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| HybridSearch.ts | agentdb | 1,062 | 95% | DEEP | BEST SEARCH CODE. Correct BM25, 3 fusion strategies | R16 |
| Quantization.ts | agentdb | 996 | 98% | DEEP | Per-dimension min/max, O(1) swap-removal | R16, R22 |
| vector-quantization.ts | agentdb | 1,529 | 95% | DEEP | Global min/max, async K-means++. Duplicates Quantization.ts | R8, R22 |
| BatchOperations.ts | agentdb | 809 | 92% | DEEP | SQL injection prevention, transaction management | R16, R22 |
| WASMVectorSearch.ts | agentdb | 458 | 70% | DEEP | WASM module missing, JS fallback is correct | R16 |
| CausalRecall.ts | agentdb | 506 | 75% | DEEP | Reranking formula sound, depends on broken CausalMemoryGraph | R16 |
| BenchmarkSuite.ts | agentdb | 1,361 | 95% | DEEP | Production framework, quantization benchmark crashes | R16, R22 |
| BenchmarkSuite.js | agentdb | 984 | 100% | DEEP | performance.now ×28, zero fakes | R32 |

### Security & Infrastructure

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| validation.ts | agentdb | 557 | 95% | DEEP | Path traversal blocking, 13 sensitive field regexes | R16 |
| input-validation.ts | agentdb | 544 | 98% | DEEP | Whitelist SQL injection prevention, parameterized builders | R16 |
| auth.service.ts | agentdb | 668 | 92% | DEEP | Argon2id, 5-attempt lockout. In-memory storage only | R16, R22 |
| token.service.ts | agentdb | 492 | 96% | DEEP | JWT HS256, 15min/7d TTL. In-memory revocation list | R16 |
| telemetry.ts | agentdb | 545 | 85% | DEEP | OTel framework, SDK init stubbed, no exporters | R16 |

### Attention Mechanisms

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| MultiHeadAttentionController.ts | agentdb | 494 | 98% | DEEP | Real scaled dot-product, Xavier init, 4 aggregation strategies | R16 |
| CrossAttentionController.ts | agentdb | 467 | 98% | DEEP | Multi-context attention, namespace-based stores | R16 |
| AttentionService.ts | agentdb | 771 | 80% | DEEP | NAPI→WASM→JS fallback. JS MHA is single-head | R22 |
| attention-fallbacks.ts | agentdb | 1,953 | 92% | DEEP | HyperbolicAttention correct Poincaré distance (TS source) | R22 |
| attention-tools-handlers.ts | agentdb | 587 | 40% | DEEP | ALL metrics Math.random(), handlers are template strings | R40 |

### Embeddings & Vectors

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| enhanced-embeddings.ts | agentdb | 1,436 | 90% | DEEP | O(1) LRU, multi-provider. Falls back to hash mock at L1109 | R8, R22 |
| RuVectorBackend.ts | agentdb | 971 | 90% | DEEP | Production-ready, correct distance conversion | R8 |
| simd-vector-ops.ts | agentdb | 1,287 | 0% SIMD | DEEP | NOT SIMD — scalar 8x loop unrolling. WASM detected but unused | R8, R22 |

### LLM & Intelligence

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| LLMRouter.ts | agentdb | 660 | 78% | DEEP | Priority-based lookup, NOT ML. No connection to ADR-008 | R40 |

### Tests

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| specification-tools.test.ts | agentdb | 2,222 | 90% | DEEP | 105-test vitest suite, real better-sqlite3 + Xenova | R22 |
| ruvector-integration.test.ts | agentdb | 1,590 | 95% | DEEP | BEST test file in ecosystem | R22 |

### Synchronization & CRDT

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| QUICClient.ts | agentdb | 668 | 25% | DEEP | ENTIRELY STUB. sendRequest returns hardcoded success after 100ms | R22 |
| SyncCoordinator.ts | agentdb | 717 | 55% | DEEP | Real orchestration, routes through stub QUICClient | R22 |
| quic.ts | agentdb | 773 | 95% | DEEP | Textbook CRDTs: GCounter, LWWRegister, ORSet | R22 |

### Analysis & Clustering

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| clustering-analysis.ts | agentdb | 797 | 85% | DEEP | Production Louvain, Label Propagation. Agent metrics facades | R22, R41 |
| traversal-optimization.ts | agentdb | 783 | 82% | DEEP | Beam search real, recall values HARDCODED | R22, R41 |
| self-organizing-hnsw.ts | agentdb | 681 | 80% | DEEP | MPC adaptation production-grade, recall Math.random | R22, R41 |
| hypergraph-exploration.ts | agentdb | 707 | 78% | DEEP | Real hypergraph, 5 collaboration patterns, structural metrics faked | R41 |

### Agentic-Flow Wrappers

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| agentdb-wrapper-enhanced.ts | agentic-flow | 899 | 85% | DEEP | FIXES R18 — proper embedder chain for agentic-flow | R32 |
| enhanced-booster-tools.ts | agentic-flow | 533 | 90% | DEEP | 6-strategy selection, tiered compression | R32 |
| standalone-stdio.ts | agentic-flow | 813 | 95% | DEEP | Thin npx delegation (15 tools). Cache clear is STUB | R32 |
| edge-full.ts | agentic-flow | 943 | 75% | DEEP | 6-module WASM toolkit. JS fallback is CHARACTER HASHING | R32 |
| reasoningbank_wasm_bg.js | agentic-flow | 556 | 100% | DEEP | wasm-bindgen auto-generated, 5 async methods | R32 |

## 3. Findings Registry

### 3a. CRITICAL Findings

| ID | Description | File(s) | Session | Status |
|----|-------------|---------|---------|--------|
| C1 | **Completely unused** — 140K+ LOC dead weight. claude-flow never calls any of 23 AgentDB controllers | All AgentDB | R8 | Open |
| C2 | **Missing WASM module** — reasoningbank_wasm.js doesn't exist, brute-force JS fallback | WASMVectorSearch.ts | R8 | Open |
| C3 | **Broken native deps** — 1,484 lines of JS fallback for broken @ruvector APIs | Multiple | R8 | Open |
| C4 | **CausalMemoryGraph statistics broken** — Wrong t-CDF formula, hardcoded tInverse=1.96. All p-values unreliable | CausalMemoryGraph.ts | R8 | Open. Confirmed R22 in TS source |
| C5 | **LearningSystem RL is cosmetic** — DQN without neural network; all 9 algorithms reduce to identical Q-value dict | LearningSystem.ts | R8 | Open. Confirmed R22 in TS source |
| C6 | **Attention MCP metrics 100% fabricated** — Math.random() for totalCalls, latencies, memory, success rates | attention-tools-handlers.ts | R16, R40 | Open |
| C7 | **Quantization benchmark crashes** — BenchmarkSuite.ts L809 interface mismatch causes runtime error | BenchmarkSuite.ts | R16 | Open |
| C8 | **QUICClient is entirely stub** — sendRequest returns hardcoded success after 100ms sleep, no QUIC protocol | QUICClient.ts | R22 | Open |
| C9 | **Traversal recall values hardcoded** — beam=0.948, dynamic-k=0.941, greedy=0.882 are constants not computed | traversal-optimization.ts | R22, R41 | Open |
| C10 | **Leiden clustering is no-op** — refinementPhase() does nothing beyond Louvain | clustering-analysis.ts | R22 | Open |
| C11 | **Latent-space HNSW metrics simulated** — recall=0.92+random*0.05, adaptationSpeed hardcoded to 5.5 | self-organizing-hnsw.ts | R22, R41 | Open |
| C12 | **agentdb_clear_cache tool is STUB** — Returns mock JSON message, no actual clearing | standalone-stdio.ts | R32 | Open |
| C13 | **edge-full.ts JS embedding fallback is CHARACTER HASHING** — charCodeAt-based, NOT semantic. WASM/ONNX required | edge-full.ts | R32 | Open |
| C14 | **NightlyLearner discover() returns empty** — Public API creates edges internally but always returns [] | NightlyLearner.ts | R40 | Open |
| C15 | **MCP attention encoding is hash-based** — encodeQueryVector() uses charCodeAt, not semantic embeddings | attention-tools-handlers.ts | R40 | Open |
| C16 | **Self-organizing HNSW recall fabricated** — 0.92 + Math.random()*0.05. MPC adaptation genuine but outcomes simulated | self-organizing-hnsw.ts | R41 | Open |

### 3b. HIGH Findings

| ID | Description | File(s) | Session | Status |
|----|-------------|---------|---------|--------|
| H1 | **QUIC is TCP** — Names are misleading | QUICClient/Server | R8 | Open |
| H2 | **SIMD is fake** — Functions use only scalar 8x loop unrolling, WASM detected but unused | simd-vector-ops.ts | R8, R22 | Open |
| H3 | **ReflexionMemory breaks its paper** — Missing judge/feedback loop per arXiv:2303.11366 | ReflexionMemory.ts | R8 | Open |
| H4 | **enhanced-embeddings silent degradation** — Falls back to hash mock without warning | enhanced-embeddings.ts | R8, R22 | Open |
| H5 | **CLI is 98% real** — 35+ subcommands, 60+ methods, all fully implemented | agentdb-cli.ts | R16 | Open (Positive) |
| H6 | **34 MCP tools registered** — Complete vector DB + frontier + learning + attention surface | agentdb-mcp-server.ts | R16 | Open (Positive) |
| H7 | **ReasoningBank is canonical** — v1/v2 dual-mode, cosine similarity, SQLite + VectorBackend | ReasoningBank.ts | R16 | Open (Positive) |
| H8 | **HNSWIndex wraps real hnswlib-node** — C++ HNSW library, not reimplemented | HNSWIndex.ts | R16 | Open (Positive) |
| H9 | **HybridSearch is production-grade** — Correct BM25, three fusion strategies | HybridSearch.ts | R16 | Open (Positive) |
| H10 | **Quantization K-means++ is correct** — ADC with precomputed lookup tables | Quantization.ts | R16 | Open (Positive) |
| H11 | **Security model is solid** — Argon2id, SQL injection prevention, JWT, brute force protection | validation.ts +3 | R16 | Open (Positive) |
| H12 | **Attention mechanisms are GENUINE** — Real scaled dot-product, Xavier init, stable softmax | MultiHeadAttentionController +1 | R16 | Open (Positive) |
| H13 | **NightlyLearner DR formula incomplete** — Missing control group adjustment term | NightlyLearner.ts | R16 | Open |
| H14 | **SkillLibrary "ML" is keyword counting** — Claims ML-inspired but uses TF word counting | SkillLibrary.ts | R16 | Open |
| H15 | **Auth/token use in-memory storage** — Users, sessions, API keys, revocation all in Maps | auth.service.ts, token.service.ts | R16, R22 | Open |
| H16 | **Telemetry OTel framework not wired** — SDK init stubbed, no exporters connected | telemetry.ts | R16 | Open |
| H17 | **SyncCoordinator real but routes through stub** — Real orchestration depends on non-functional QUICClient | SyncCoordinator.ts | R22 | Open |
| H18 | **ReasoningBank O(N*M) scan** — getEmbeddingsForVectorIds uses map scan instead of reverse index | ReasoningBank.ts | R22 | Open |
| H19 | **Spectral clustering wrong** — Uses raw embeddings instead of graph Laplacian eigenvectors | clustering-analysis.ts | R22 | Open |
| H20 | **Duplicate quantization modules** — vector-quantization.ts and Quantization.ts with different approaches | agentdb | R22 | Open |
| H21 | **All 8 native controllers require EmbeddingService** — Without it, ~90% of AgentDB features non-functional | agentdb-mcp-server.js | R32 | Open |
| H22 | **BenchmarkSuite uses REAL timing** — performance.now ×28, zero fake benchmarks | BenchmarkSuite.js | R32 | Open (Positive) |
| H23 | **Cypher queries REQUIRE WASM** — edge-full.ts JS mode throws on any Cypher/SPARQL, no parser fallback | edge-full.ts | R32 | Open |
| H24 | **EnhancedAgentDBWrapper fixes R18** — Properly chains reflexionController→embedder→vectorBackend | agentdb-wrapper-enhanced.ts | R32 | Open (Positive) |
| H25 | **MultiDatabaseCoordinator sync is simulated** — Health checks hardcoded, no transactional guarantees, last-write-wins | persistence-pooled.js | R33 | Open |
| H26 | **LLMRouter has NO connection to ADR-008** — Separate priority-based system from claude-flow's 3-tier routing | LLMRouter.ts | R40 | Open |
| H27 | **NightlyLearner dead dependencies** — ReflexionMemory and SkillLibrary constructed but never called | NightlyLearner.ts | R40 | Open |
| H28 | **LLMRouter constructor timing bug** — RuvLLM can never be default provider due to async init race | LLMRouter.ts | R40 | Open |
| H29 | **Simulations NOT connected to production HNSWIndex** — All 4 latent-space files build own HNSW in TypeScript | clustering-analysis +3 | R41 | Open |
| H30 | **14 Math.random facade metrics** — Secondary metrics across latent-space simulations use baselines + noise | clustering-analysis +3 | R41 | Open |
| H31 | **Hypergraph path query trivial** — Returns [start, midpoint, end] instead of real graph traversal | hypergraph-exploration.ts | R41 | Open |

## 4. Positives Registry

| Description | File(s) | Session |
|-------------|---------|---------|
| **HybridSearch** is the best search implementation across entire ruvnet codebase | HybridSearch.ts | R16 |
| **Quantization** (K-means++ PQ, 8/4-bit scalar) is production-ready | Quantization.ts, vector-quantization.ts | R8, R16 |
| **ReasoningBank** is well-architected canonical implementation | ReasoningBank.ts | R16 |
| **HNSWIndex** wraps real C++ hnswlib for genuine ANN search | HNSWIndex.ts | R16, R40 |
| **Security** model is comprehensive and correct | validation.ts, input-validation.ts, auth.service.ts, token.service.ts | R16 |
| **Attention** controllers implement real transformer-style neural attention from scratch | MultiHeadAttentionController, CrossAttentionController | R16 |
| **CLI** exposes 35+ working commands covering full AgentDB feature set | agentdb-cli.ts | R16 |
| **MCP Server** registers 27+ tools with proper input validation | agentdb-mcp-server.js | R16, R32 |
| **ExplainableRecall** has genuine Merkle tree provenance and greedy set cover | ExplainableRecall.ts | R16 |
| **BenchmarkSuite.ts** is best-quality file in AgentDB (95%) — production benchmarking | BenchmarkSuite.ts | R22 |
| **ruvector-integration.test.ts** is best test file in entire ecosystem (95%) | ruvector-integration.test.ts | R22 |
| **quic.ts types** contain textbook-correct CRDT implementations | quic.ts | R22 |
| **HyperbolicAttention TS source** uses CORRECT Poincaré distance (compilation degraded it) | attention-fallbacks.ts | R22 |
| **BenchmarkSuite.js** (compiled): 100% real, zero fakes | BenchmarkSuite.js | R32 |
| **agentdb-wrapper-enhanced.ts** resolves R18 for agentic-flow with proper initialization | agentdb-wrapper-enhanced.ts | R32 |
| **enhanced-booster-tools.ts** has genuine 6-strategy learning with tiered compression | enhanced-booster-tools.ts | R32 |
| **NightlyLearner SQL path** discovers causal edges independently of embeddings | NightlyLearner.ts | R40 |
| **Louvain implementation** is production-grade with correct Newman modularity Q formula | clustering-analysis.ts | R41 |
| **MPC adaptation** is cutting-edge: state-space prediction, 97.9% degradation prevention | self-organizing-hnsw.ts | R41 |
| **Beam search** is genuine multi-layer traversal with empirically optimized width | traversal-optimization.ts | R41 |
| **Hypergraph construction** with 5 collaboration patterns is well-designed research | hypergraph-exploration.ts | R41 |

## 5. Subsystem Sections

### 5a. Architecture Overview

AgentDB consists of three parallel implementations:

| Implementation | Package | Storage | Components | Status |
|---------------|---------|---------|------------|--------|
| **Native MCP Server** | agentdb | SQLite + EmbeddingService | 27+ tools, all controllers initialized | FUNCTIONAL |
| **Agentic-Flow Wrapper** | agentic-flow | Delegates to native | 15 tools via npx, enhanced wrapper | FUNCTIONAL (via R32 fix) |
| **Claude-Flow Bridge** | claude-flow | agentdb-service-fallback | 6 tools, NO EmbeddingService | BROKEN (R18) |

The native standalone MCP server is the canonical architecture. R18 deep-read revealed why claude-flow bridge fails: agentdb-mcp-server.ts is designed as a standalone process that initializes `@xenova/transformers` for real embeddings on startup. The bridge layer (agentdb-tools.js) bypasses EmbeddingService, causing writes to succeed but reads to return empty (R18).

### 5b. Search Architecture

**HybridSearch.ts is the best search code in the ruvnet universe** (R16). Implements three-stage retrieval:

1. **BM25 keyword search** via in-memory TypeScript inverted index (KeywordIndex class) — NOT SQLite FTS. IDF formula correct, k1=1.2, b=0.75 standard, document length normalization (L316-356).
2. **Vector search** via HNSW or VectorBackend with cosine similarity.
3. **Fusion strategies** (R16): RRF (1/(k+rank)), Linear (α*vector + β*keyword), Max (element-wise).

All mathematically sound, O(n*m) complexity. The in-memory BM25 index explains why search fails through claude-flow bridge — index requires explicit `keywordIndex.add(id, text)` calls that never happen in bridge mode (R18).

**HNSWIndex.ts** (R16, R40) wraps `hnswlib-node` (C++ library) with lazy-loading, persistence, and filter support. Distance-to-similarity conversion correct (cosine=1-distance, L2=exp(-distance)). Delete is a stub — hnswlib doesn't support deletion, code tracks but never rebuilds. Directly affected by R20 broken EmbeddingService: if `pattern_embeddings` contains hash-based garbage, HNSW faithfully indexes garbage (R40).

**Quantization** has two production-grade implementations (R16, R22):
- **Quantization.ts** (996 LOC): Per-dimension min/max (more accurate), O(1) swap-removal, auto-reindex at 20% deletion threshold.
- **vector-quantization.ts** (1,529 LOC): Global min/max normalization, async K-means++, PQ with precomputed lookup tables.

Both are best-in-ecosystem code. Creates maintenance burden (H20).

### 5c. Security Model

**Solid and comprehensive** (R16). Four-layer defense:

1. **validation.ts** (557 LOC, 95%): NaN/Infinity prevention, path traversal blocking via regex, 13 sensitive field regexes (API_KEY, password, token, etc.), Cypher injection prevention, 21 security limits.
2. **input-validation.ts** (544 LOC, 98%): Whitelist SQL injection prevention (13 tables, per-table columns, 11 pragmas), parameterized query builders.
3. **auth.service.ts** (668 LOC, 92%): Argon2id hashing, 5-attempt lockout, username enumeration prevention, API key rotation. **In-memory storage** — users/sessions lost on restart (R16, R22, H15).
4. **token.service.ts** (492 LOC, 96%): JWT HS256 via jsonwebtoken, 15min access / 7d refresh tokens, 32-char secret minimum, revocation list with auto-cleanup. **In-memory** — flagged for Redis (R16, H15).

Security is architecturally sound but operationally limited by in-memory storage.

### 5d. Attention Mechanisms

**Genuine neural computation** — the most surprising R16 finding (confirmed R22 in TypeScript source):

**MultiHeadAttentionController.ts** (494 LOC, 98%): Xavier init, scaled dot-product attention (1/sqrt(d_k)), numerically stable softmax, 4 aggregation strategies (mean/max/concat/first). Implements transformer-style attention from scratch without external neural libraries.

**CrossAttentionController.ts** (467 LOC, 98%): Multi-context attention, weighted sum output, namespace-based context stores.

Both are inference-only with random weights (not trainable). AttentionService.ts (771 LOC, 80%) provides NAPI→WASM→JS fallback chain. JS MHA is single-head; Hyperbolic/MoE fallbacks reduce to standard MHA (R22).

**attention-fallbacks.ts** (1,953 LOC, 92%) contains correct Poincaré ball distance in HyperbolicAttention TypeScript source — compilation degraded it to Euclidean approximation (R22). Flash backward pass is correct.

**attention-tools-handlers.ts** (587 LOC, 40%) is a complete facade: ALL metrics are Math.random() (totalCalls, latencies, memory, success rates at L293-299). Handlers are exported as template literal strings, not functions — defeats TypeScript's type system (R40, C6, C15).

### 5e. Core Controller Quality Spectrum

| Quality Tier | Components | Real% | Notes |
|--------------|------------|-------|-------|
| **Production** | ReasoningBank, HNSWIndex, Quantization, HybridSearch, Security | 95-98% | Best code in ruvnet |
| **Solid** | ExplainableRecall, BatchOperations, MemoryController | 85-95% | Production-ready with gaps |
| **Partial** | NightlyLearner, CausalRecall, SkillLibrary | 75-90% | Real core, incomplete features |
| **Broken** | LearningSystem, CausalMemoryGraph | 55-65% | Critical bugs, cosmetic implementations |
| **Stub** | QUICClient, WASMVectorSearch | 0-70% | Missing dependencies |

**LearningSystem.ts** (R8, R22) claims 9 RL algorithms (Q-learning, SARSA, DQN, PPO, Actor-Critic, Policy Gradient, Decision Transformer, Model-Based, MCTS) but ALL reduce to identical tabular Q-value dictionary updates. DQN has no neural network. PPO/Actor-Critic are running averages. Bug confirmed in TypeScript source — not a compilation artifact (R22, C5).

**CausalMemoryGraph.ts** (R8, R22) claims Pearl's do-calculus but implements none. t-distribution CDF is wrong (L851), tInverse hardcoded to 1.96 ignoring degrees of freedom. calculateCorrelation() is fake — uses session count instead of real correlation. All p-values and confidence intervals unreliable. Bug confirmed in TypeScript source (R22, C4).

**ReflexionMemory.ts** (R8) storage works but breaks arXiv:2303.11366 — missing judge function that synthesizes critique from trajectories. Core paper loop (RETRIEVE → JUDGE → DISTILL → CONSOLIDATE) is broken (H3).

### 5f. Latent-Space Research Simulations

Four standalone research simulations (R41) — NOT connected to production HNSWIndex.ts (H29). All build HNSW-like graphs in pure TypeScript for algorithm validation and parameter tuning.

**Weighted average: 81% real** (R41). Core algorithms are textbook-correct:

| File | Algorithm | Quality | Validation |
|------|-----------|---------|------------|
| clustering-analysis.ts | Louvain community detection | Production | Resolution=1.2 → Q=0.758, purity=89.1% |
| traversal-optimization.ts | Beam search, DynamicKSearch | Real | Beam-width=5 optimal, 94.8% recall |
| self-organizing-hnsw.ts | MPC adaptation | Cutting-edge | Control horizon=5, 97.9% degradation prevention |
| hypergraph-exploration.ts | Hypergraph construction | Well-designed | 5 collaboration patterns, 3.7x compression |

**Empirically validated configurations** (R41): Louvain optimal resolution, beam search width, MPC parameters, hypergraph compression ratio are publishable findings. MPC-based HNSW adaptation is cutting-edge research.

**14 Math.random facade metrics** (R41, H30) — secondary metrics use `baseline + Math.random()*range`. CRITICAL: recall values in traversal-optimization.ts are HARDCODED constants (beam:94.8%, dynamic-k:94.1%, greedy:88.2%) not computed from ground truth (C9). self-organizing-hnsw.ts recall is `0.92 + Math.random()*0.05` (C16).

### 5g. LLM Routing & Intelligence Layer

**LLMRouter.ts** (660 LOC, 78%) is NOT intelligent routing (R40). Uses priority-based lookup table (quality→balanced→cost→speed→privacy mapping to providers), not ML-based. **Has NO connection to claude-flow's ADR-008 3-tier model routing** — completely parallel systems (H26). Constructor timing bug: selectDefaultProvider() checks ruvllmAvailable which is always false at construction due to async init race (H28).

**NightlyLearner.ts** (665 LOC, 75-80%) has two independent paths (R16, R40):
1. **SQL path** (L340-417): discoverCausalEdges() works independently of embeddings — functional regardless of R20 broken EmbeddingService.
2. **Attention path** (L243): consolidateEpisodes() calls embedder.embed() — DIRECTLY AFFECTED by R20. Attention-based causal discovery is meaningless with hash embeddings (R40).

Dead dependencies: ReflexionMemory and SkillLibrary constructed at L84-85 but NEVER used in any method (R40, H27). Public API discover() creates edges internally but always returns empty array (R40, C14).

**Doubly-robust estimator** (L385) only processes treated observations — not truly doubly robust, missing control group adjustment term (R16, H13).

### 5h. Synchronization & CRDT

**QUICClient.ts** (668 LOC, 25%) is ENTIRELY STUB (R22). sendRequest() returns hardcoded `{success:true}` after 100ms sleep. No QUIC protocol implementation. QUIC is TCP — names are misleading (H1).

**SyncCoordinator.ts** (717 LOC, 55%) has real orchestration logic (change detection, sync state, auto-sync intervals) but routes through stub QUICClient, making it non-functional (R22, H17).

**quic.ts** (773 LOC, 95%) contains textbook-correct CRDT implementations: GCounter, LWWRegister, ORSet, VectorClock helpers (R22). Types are production-quality but unused due to broken client/server.

**MultiDatabaseCoordinator** in persistence-pooled.js (42%, R33) claims cross-database sync but health checks return hardcoded healthy, conflict resolution uses last-write-wins without vector clocks, no transactional guarantees (H25).

### 5g. R18 Native vs Patched Architecture

R18 deep-read revealed the root cause of broken claude-flow integration:

**Native architecture** (FUNCTIONAL):
```
agentdb-mcp-server.ts  ← #!/usr/bin/env node (STANDALONE process)
  ├── EmbeddingService(Xenova/all-MiniLM-L6-v2, 384d)  ← INITIALIZED at startup (L219-224)
  ├── ReflexionMemory(db, embedder, ...)
  ├── SkillLibrary(db, embedder)
  └── All 27+ tools with proper embeddings
```

**Patched architecture** (BROKEN):
```
claude-flow mcp-server.js
  ├── agentdb-tools.js (598 LOC)  ← bridges via createRequire()
  │     └── agentdb-service-fallback  ← silent degradation path
  └── 6 tools exposed (vs 27 native)
      → NO EmbeddingService initialized
      → Episodes stored WITHOUT embeddings
      → episode_embeddings stays EMPTY
      → SQL JOIN returns zero rows
      → Search returns empty array
```

Tool coverage gap: User exposes 6/27 native tools, 2 work (stats), 3 are broken (search/retrieve/suggest), 1 is custom (R18).

**agentic-flow resolution** (R32): agentdb-wrapper-enhanced.ts properly chains reflexionController→embedder→vectorBackend. EnhancedAgentDBWrapper fixes R18 for agentic-flow users (H24).

### 5h. Infrastructure & Telemetry

**telemetry.ts** (545 LOC, 85%, R16): OpenTelemetry integration with proper metric instruments (histogram, counters, gauge), @traced decorator. BUT: SDK initialization is stubbed (empty instrumentations array), no OTLP exporter connected (H16).

**BenchmarkSuite** has two implementations:
- **BenchmarkSuite.ts** (1,361 LOC, 95%, R16, R22): Best-quality file in AgentDB. Complete framework with percentile latency, 5% regression threshold. Quantization benchmark would crash due to interface mismatch at L809 (C7).
- **BenchmarkSuite.js** (984 LOC, 100%, R32): Compiled version with performance.now ×28, zero fake benchmarks across all 5 classes (H22).

## 6. Cross-Domain Dependencies

- **memory-and-learning domain**: LearningSystem, ReasoningBank, ReflexionMemory overlap heavily
- **ruvector domain**: RuVectorBackend, HNSW indices, quantization modules
- **agentic-flow domain**: Wrapper implementations (enhanced-booster-tools, agentdb-wrapper-enhanced, edge-full)
- **claude-flow-cli domain**: Patched bridge layer (agentdb-tools.js), broken integration

## 7. Knowledge Gaps

- ~429 files still NOT_TOUCHED (mostly dist/ compiled mirrors, additional test files, CLI helpers)
- Remaining controller source files not yet deep-read
- Browser bundle and schema files
- Additional simulation scenarios beyond latent-space (if any exist)
- Integration between temporal-tensor/agentdb.rs (Rust-side) and TypeScript AgentDB

## 8. Session Log

### R8 (2026-02-09): Initial AgentDB deep-read
7 files, 8,594 LOC. Established HybridSearch as best-in-ecosystem, identified broken LearningSystem/CausalMemoryGraph, discovered AgentDB is completely unused by claude-flow despite being dependency.

### R16 (2026-02-14): CLI & MCP surface area
52 files analyzed. Revealed complete CLI command surface (35+ subcommands), 34 MCP tools, genuine neural attention, production-grade security model, canonical ReasoningBank implementation.

### R18 (2026-02-14): Native architecture deep-read
Identified root cause of broken claude-flow integration: missing EmbeddingService initialization. Native MCP server is functional standalone; bridge layer silently degrades to fallback.

### R22 (2026-02-15): TypeScript source confirmation
22 files, ~22K LOC. Confirmed LearningSystem and CausalMemoryGraph bugs exist in TS source. HyperbolicAttention correct in source (compilation degraded it). QUICClient entirely stub. Duplicate quantization modules.

### R32 (2026-02-15): Compiled JS + agentic-flow wrappers
8 files, ~8,330 LOC. Confirmed native CLI/MCP have correct EmbeddingService init. agentic-flow wrapper fixes R18 issue. edge-full.ts JS fallback is character hashing.

### R33 (2026-02-15): Swarm infrastructure
MultiDatabaseCoordinator sync simulation discovered (42% real) — health checks hardcoded, no transactional guarantees.

### R40 (2026-02-15): Intelligence layer
4 files. LLMRouter has NO connection to ADR-008. NightlyLearner SQL path works independently of embeddings. Attention MCP tools metrics all Math.random().

### R41 (2026-02-15): Latent-space simulations
4 files, 2,968 LOC. Genuine research algorithms (Louvain, beam search, MPC adaptation) with empirically validated configurations. 14 Math.random facade metrics. NOT connected to production HNSWIndex.
