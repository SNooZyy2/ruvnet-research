# AgentDB Integration — Section 2: File Registry

> Part of [AgentDB Integration Domain Analysis](analysis.md). See index for full document map.

## 2. File Registry

### AgentDB CLI & MCP Server

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| agentdb-cli.js | agentdb | 3,039 | 95% | DEEP | 14 commands + 60+ subcommands, EmbeddingService initialized | R32 |
| agentdb-mcp-server.js | agentdb | 2,368 | 98% | DEEP | 27+ tools fully implemented, correct EmbeddingService | R32 |
| agentdb-mcp-server.ts (TS source) | agentdb | 2,367 | 82-87% | DEEP | 32 MCP tools. EmbeddingService IS initialized (Xenova/all-MiniLM-L6-v2, 384-dim, Pipeline 1). Does NOT use backend factory — uses ReflexionMemory built-in brute-force cosine. Causal graph API data model broken (hardcoded fromMemoryId=0/toMemoryId=0). Version mismatch (v1.3.0 constructor vs v2.0.0 banner). 12 dependency edges. Ghost DEEP corrected | R137 |
| agentdb-cli.ts | agentdb | 3,422 | 98% | DEEP | Complete command surface, 35+ subcommands | R16 |

### claude-flow CLI Entry Points

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| v3/@claude-flow/cli/bin/cli.js | claude-flow | 156 | N/A | DEEP | Cold dispatcher, two disjoint bootstrap paths (MCP/CLI), zero subsystem init at boot | R135 |
| v3/@claude-flow/cli/bin/mcp-server.js | claude-flow | 189 | ~72-78% | DEEP | Near-duplicate MCP server, false resources capability ({subscribe:true} with zero handlers), async message ordering race | R135 |

### Backend Factory

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| src/backends/factory.ts | agentdb (root) | 235 | 85-88% | DEEP | 2-tier backend selection (ruvector > hnswlib). Clean dynamic import detection, lazy HNSWLib loading, isNative?() check. SIMPLER than packages/agentdb 5-tier factory (ID 12809). No RVF/sql.js fallback — hard error without either dep | R139 |

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
| MemoryController.ts | agentdb | 462 | 72% | DEEP | Pure in-memory Map store (10th disconnected persistence layer). No EmbeddingService — callers supply embeddings. VectorBackend optional (null default → O(n) JS cosine fallback). delete() BUG: removes from Map but NOT VectorBackend. CrossAttentionController initialized but NEVER called in search(). THREE attention controllers unconditionally initialized. Attention combination hardcoded 0.5*base + 0.5*(attention/2) | R16, R96 |
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
| CrossAttentionController.ts | agentdb | 467 | 62-68% | DEEP | Math sound (scaled dot-product, stable softmax) but architecturally DEAD. VectorBackend insert-only. computeCrossAttention() NEVER invoked from production. No learned weights (no W_q/W_k/W_v) | R16, R114 |
| AttentionService.ts | agentdb | 1,523 | 60-65% | DEEP | 4 genuine JS math fallbacks (FlashAttention, HyperbolicAttention, GraphRoPE, MoEAttention). WASM/NAPI never compiled. db param DEAD CODE — zero SQL ops. All mechanisms enabled:false by default. 3 real downstream consumers (NightlyLearner, CausalMemoryGraph, ExplainableRecall) | R22, R91 |
| attention-fallbacks.ts | agentdb | 1,953 | 92% | DEEP | HyperbolicAttention correct Poincaré distance (TS source) | R22 |
| attention-tools-handlers.ts | agentdb | 587 | 40% | DEEP | ALL metrics Math.random(), handlers are template strings | R40 |

### Embeddings & Vectors

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| enhanced-embeddings.ts | agentdb | 1,436 | 90% | DEEP | O(1) LRU, multi-provider. Falls back to hash mock at L1109 | R8, R22 |
| RuVectorBackend.ts | agentdb | 971 | 72-78% | DEEP | Pure adapter, zero own HNSW code. Genuine: Semaphore (FIFO), BufferPool, validatePath, prototype pollution defense. Problems: updateAdaptiveParams() only adjusts efSearch (not M/efConstruction); insertBatchParallel() creates local semaphore bypassing instance-level; mmap not wired to hot paths; L2 similarity uncalibrated. REVISED DOWN from 88-92% (R88). | R8, R91 |
| RuVectorBackend.js | agentdb | 776 | 88-92% | DEEP | GENUINE ruvector integration. Dynamic imports of `ruvector`/`@ruvector/core`. Real HNSW ops (insert/search/remove via VectorDB). Adaptive HNSW parameters. Production security (path validation, pollution protection). Parallel batch insert with semaphore. RESCUES AgentDB credibility. REVERSES R44 ruvector-backend.ts (12%) | R50 |
| src/backends/ruvector/index.ts | agentdb | 10 | 88-92% | DEEP | 10-line barrel re-export of RuVectorBackend + RuVectorLearning. Entry point for ruvector backend package. See RuVectorBackend.ts (~500 LOC) for implementation detail | R88 |
| simd-vector-ops.ts | agentdb | 1,287 | 0% SIMD | DEEP | NOT SIMD — scalar 8x loop unrolling. WASM detected but unused | R8, R22 |
| SqlJsRvfBackend.ts | agentdb | 457 | 88-92% | DEEP | sql.js WASM fallback RVF backend. O(n) brute-force search (no HNSW). Raw SQLite .rvf files incompatible with native RvfBackend. getDatabase() leaks encapsulation for unified mode. Pending flush() fire-and-forget race. Fallback last in factory 4-tier chain | R115 |
| NativeAccelerator.ts | agentdb | 489 | 72-78% | DEEP | ADR-007 Phase 1 capability bridge for @ruvector packages. 11 lazy-loaders via Promise.allSettled, JS fallbacks via SimdFallbacks.ts. CRITICAL: 3 API mismatches make AdamWOptimizer/InfoNceLoss/TensorCompress always fall back to JS. graphTxAvailable + graphCypherAvailable always false. @ruvector/rvf-wasm not installed. Genuine: SIMD via ruvllm NAPI works if ruvllm binary present | R115 |

### RVF Backend (v3)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| factory.ts | agentdb | 235 | 88-92% | DEEP | **5-tier chain** (RuVector native > WASM > RVF SDK > HNSWLib > sql.js RVF, not 4). Reality: steps 1-3 fail (not installed), step 4 if hnswlib-node present, else step 5. SelfLearningRvfBackend NEVER instantiated. BackendType union omits `sqljsrvf`. No memoization of detection probes. AgentDB.ts sole runtime caller | R115, R137 |
| RvfBackend.ts | agentdb | 749 | 75-80% | DEEP | Native @ruvector/rvf HNSW backend. CRITICAL: remove() fire-and-forget (C31), auto-flush drops writes (C32). search() always throws on sync contract. distanceToSimilarity() inverts ranking for ip metric. embedKernel/embedEbpf accept arbitrary binary with no size limit. | R115 |
| SelfLearningRvfBackend.ts | agentdb | 487 | 65-72% | DEEP | ADR-006 primary class. CRITICAL: contrastive projection never applied (C33), negative mining always empty (C34), _learningRate dead variable (C35). SONA delegation is real. exportedNowhere — unreachable from all public entry points. | R115 |
| NativeAccelerator.ts | agentdb | 489 | 72-78% | DEEP | ADR-007 Phase 1 capability bridge. 11 lazy-loaders, JS fallbacks via SimdFallbacks.ts. CRITICAL: TensorCompress static check always fails (C28), AdamWOptimizer 7-arg vs 2-arg (C29), InfoNceLoss 4-arg vs 3-arg (C30). graphTxAvailable/graphCypherAvailable permanently false. @ruvector/rvf-wasm not installed. | R115 |
| ContrastiveTrainer.ts | agentdb | 559 | 87-90% | DEEP | Real InfoNCE loss, analytical backprop via chain rule, AdamW with L2 decay. Positives created via Gaussian noise injection (not genuine relevant pairs). Used by SelfLearningRvfBackend but trained projection never applied. | R115 |
| SonaLearningBackend.ts | agentdb | 357 | 82-88% | DEEP | Real Rust N-API delegation to @ruvector/sona. All 3 SONA loops (beginTrajectory/addContext/endTrajectory). CRITICAL: applyBaseLora() 1-arg vs 2-arg N-API (C36). addContext() passes JSON.stringify as contextId (H77). Package absent from agentdb node_modules — create() returns null in production deployments. | R115 |
| FederatedSessionManager.ts | agentdb | 526 | 68-75% | DEEP | CRITICAL: aggregate() argument type mismatch — passes EphemeralAgent not exportData (C37). FedAvg NOT implemented — sequential SGD replays (C38). updateMasterLora() gradient formula invalid. EWC is constant L2 not Fisher-weighted (H80). No differential privacy. | R115 |
| SemanticQueryRouter.ts | agentdb | 456 | 62-68% | DEEP | CRITICAL: @ruvector/router not installed — always runs brute-force fallback, never HNSW path (C39). 12th parallel routing system. save/load delegate to unresolved NativeAccelerator property. Parallel to agentic-flow SemanticRouter — no cross-import. | R115 |
| AdaptiveIndexTuner.ts | agentdb | 631 | 72-78% | DEEP | 5-tier compression (hot/warm/cold/frozen/binary). ADR-010 bandit integration genuine but broken (recordReward never called). CRITICAL: binary decompression non-invertible (C40), Matryoshka truncation assumes MRL (C41). updateFrequency() mixed-format decompression bug. | R115 |
| SolverBandit.ts | agentdb | 270 | 55-62% | DEEP | Thompson Sampling Beta distribution genuine. CRITICAL: recordReward() never called — stays at Beta(1,1) random forever (C42). CRITICAL: README claims 5 controller integrations; grep finds zero such imports (C43). Unknown-arm exploration can beat well-trained arm 10% of time. sampleBeta() O(1/p) latency spikes. | R115 |
| SimdFallbacks.ts | agentdb | 254 | 82-88% | DEEP | JS fallback implementations for 6 native ops. jsAdamWStep applies weight decay before gradient correction (wrong AdamW order) (H84). jsInfoNceLoss mathematically correct. Scalar 8x loop unrolling (ILP, not hardware SIMD). | R115 |
| RvfSolver.ts | agentdb | 312 | 72-78% | DEEP | Thompson Sampling + constraint propagation puzzles. @ruvector/rvf-solver not installed — isAvailable() always false in practice. Solver trains on synthetic puzzles unrelated to actual embedding optimization. | R115 |
| FilterBuilder.ts | agentdb | 209 | 92% | DEEP | Injection-safe predicate DSL. 8 operator types (eq/ne/gt/lt/gte/lte/contains/startsWith). Parameterized query generation. Shared with validation.ts security layer. Best file in RVF subsystem. | R115 |
| SqlJsRvfBackend.ts | agentdb | 457 | 88-92% | DEEP | sql.js WASM SQLite fallback. Full ACID transactions, both sync/async interfaces. CRITICAL: O(n) brute-force search (C26), .rvf format incompatible with native RvfBackend (C27). Insert async race. getDatabase() encapsulation break for db-fallback.ts unified mode. | R115 |
| WasmStoreBridge.ts | agentdb | 83 | 88-92% | DEEP | Clean delegation pattern — routes 4 store ops to SqlJsRvfBackend. Minimal adapter, no business logic. | R115 |
| validation.ts (rvf) | agentdb | 82 | 90-95% | DEEP | Prototype-pollution scrub, comprehensive path hardening, shared across RvfBackend + SqlJsRvfBackend. | R115 |
| db-fallback.ts | agentdb | 297 | 72-78% | DEEP | better-sqlite3 compatibility shim for sql.js. Cognitive Container bridge pattern. CRITICAL: zero knowledge of native .rvf HNSW format — will corrupt binary files opened as SQLite (C44). Module-level singleton leak (H87). transaction() async incompatibility (H88). | R115 |
| rvf.ts (CLI) | agentdb | 501 | 78-84% | DEEP | CLI entry point for RVF commands. solver subcommand uses AgentDBSolver. rvf.ts-produced .rvf files are SQLite — consistent with SqlJsRvfBackend not native RvfBackend. | R115 |
| ModelCacheLoader.ts | agentdb | 144 | 82-88% | DEEP | ADR-003 confirmed: .rvf files ARE SQLite (sql.js). model_assets table holds ONNX model BLOBs. envPath check verifies directory not file — could return invalid path pointing to no ONNX file (H89). | R115 |
| wasm-loader.ts | agentdb | 163 | 75-82% | DEEP | 9 RVF classes exported here but absent from index.ts. SelfLearningRvfBackend absent from both — ADR-006 primary class unreachable. Two-exit-point export strategy creates silent inconsistency. | R115 |

### ruvbot / ruvector NPX / RVF Bridge (R118)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| npm/packages/ruvbot/src/RuvBot.ts | ruvector-rust | 781 | 30-40% | DEEP | FACADE: Zero @ruvector imports despite monorepo co-location. Header claims "RuVector's WASM" — false. Plain LLM API proxy (OpenRouter > Google AI > Anthropic). All platform integrations (Slack/Discord/Webhook) are TODO stubs. NOT the optimizer app | R118 |
| npm/packages/ruvbot/README.md | ruvector-rust | 1,400 | — | MEDIUM | Claims RVF microkernel with Linux 6.6.80 bzImage (extraordinary). SONA/WASM benchmarks are marketing facades. Zero hyperbolic/attention at npm layer. Static "560/571 tests" badge (not CI-linked) | R118 |
| npm/packages/ruvbot/docs/adr/ADR-006-wasm-integration.md | ruvector-rust | 776 | — | DEEP | "Accepted" but aspirational design only. All TS code is pseudocode. ZERO ed25519/plugin signing (redirected to RVF package). 6 unverified WASM npm packages named | R118 |
| npm/packages/ruvector/bin/cli.js | ruvector-rust | 7,357 | 40-50% | MEDIUM | 50+ commands but core requires `dist/` that doesn't exist. `graph --query` display-only stub. RVF commands delegate to unverified @ruvector/rvf. No QR-seed loading. No RVF execution runtime | R118 |
| npm/packages/rvf-mcp-server/src/server.ts | ruvector-rust | 569 | 55-65% | DEEP | 8th MCP confirmed. Protocol layer real (McpServer, 10 tools, dual transport). RVF backend is pure in-memory JS Map — @ruvector/rvf declared but never imported. rvf_compact is no-op. rvf_query is O(n) linear | R118 |
| packages/agentdb/docs/adrs/ADR-003-rvf-native-format-integration.md | agentic-flow | 1,400 | — | MEDIUM | "Proposed" but substantial implementation exists (14 files in backends/rvf/). 3-stage migration confirmed. CRITICAL: SqlJsRvfBackend .rvf = SQLite not RVF binary. Rust rvf-adapter-agentdb verified | R118 |
| crates/rvf/rvf-adapters/agentdb/src/pattern_store.rs | ruvector-rust | 457 | 65-70% | DEEP | Genuine write path to RVF via RvfStore. BROKEN read: metadata HashMap never reloaded on open(). next_id resets to 1 causing collision. Tests never test persistence (TempDir only). 6 unit tests | R118 |
| crates/rvf/rvf-adapters/agentdb/src/vector_store.rs | ruvector-rust | 327 | 60-65% | DEEP | O(n) brute-force query via RvfStore (CRITICAL). ef_search param threaded but never used. HNSW index exists in index_adapter.rs but disconnected from query path. get_vector() is O(n) per retrieval | R118 |

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
| QUICClient.ts | agentdb | 668 | 42% | DEEP | FACADE but ALGORITHMIC — zero QUIC protocol (admits "reference implementation" L108), genuine exponential backoff retry, connection pooling, batch processing with progress callbacks. Upgraded from 25% (R22) | R22, R48 |
| SyncCoordinator.ts | agentdb | 717 | 55% | DEEP | Real orchestration, routes through stub QUICClient | R22 |
| quic.ts | agentdb | 773 | 95% | DEEP | Production-grade distributed types: VectorClock, CRDTs (GCounter/LWWRegister/ORSet), full reconciliation protocol with Merkle verification, JWT auth with 12 RBAC scopes, X.509 node registration. CRDT merge functions correct (commutative, idempotent, associative) | R22, R48 |

### Analysis & Clustering

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| clustering-analysis.ts | agentdb | 797 | 85% | DEEP | Production Louvain, Label Propagation. Agent metrics facades | R22, R41 |
| traversal-optimization.ts | agentdb | 783 | 82% | DEEP | Beam search real, recall values HARDCODED | R22, R41 |
| self-organizing-hnsw.ts | agentdb | 681 | 80% | DEEP | MPC adaptation production-grade, recall Math.random | R22, R41 |
| hypergraph-exploration.ts | agentdb | 707 | 78% | DEEP | Real hypergraph, 5 collaboration patterns, structural metrics faked | R41 |

### Latent-Space Neural Augmentation

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| neural-augmentation.ts | agentdb | 605 | 70% | DEEP | MIXED. Real gradient descent, topology optimization, RL navigation (30% over greedy). 5 core metrics Math.random(). GNN weights random, never trained. Standalone testbed, NOT connected to production HNSWIndex | R43 |

### CLI Operations

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| attention.ts | agentdb | 657 | 63% | DEEP | MIXED — real multi-head attention math (dot product, Poincaré distance, sparse masking), genuine benchmark loops, but 9th hash-based embedding occurrence (charCodeAt L550-565), computeAttention() returns simulated results, optimizeMechanism() returns fabricated gains | R48 |
| config-manager.ts | agentdb | 628 | 78% | DEEP | PRODUCTION-QUALITY — Ajv JSON Schema validation (9 subsections), 3-priority config cascade (.agentdb.json > ~/.agentdb/config.json > preset), 11 AGENTDB_* env var overrides, semantic validation warnings. Preset profiles contain EXACT values from R35-R37 simulation discoveries (8.2x HNSW, 12.4% attention, 96.8% recall) | R48 |
| health-monitor.ts | agentdb | 514 | 99% | DEEP | PRODUCTION-QUALITY — real os.totalmem/freemem/cpus, process.memoryUsage, v8.getHeapStatistics. LINEAR REGRESSION memory leak detection (slope>10MB + 80% consistent growth). MPC self-healing (GC/workload reduction/restart/abort). EventEmitter pattern | R48 |
| simulation-runner.ts | agentdb | 580 | 84% | DEEP | GENUINE infrastructure with fallback mocking — 40+ scenario registry with lazy loaders, dynamic import with 5-path search, metric normalization adapts 3 output formats. Coherence via coefficient of variation. Falls back to createMockScenario() when real scenarios unavailable | R48 |

### Agentic-Flow Wrappers

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| agentdb-wrapper-enhanced.ts | agentic-flow | 899 | 85% | DEEP | FIXES R18 — proper embedder chain for agentic-flow | R32 |
| enhanced-booster-tools.ts | agentic-flow | 533 | 90% | DEEP | 6-strategy selection, tiered compression | R32 |
| standalone-stdio.ts | agentic-flow | 813 | 95% | DEEP | Thin npx delegation (15 tools). Cache clear is STUB | R32 |
| edge-full.ts | agentic-flow | 943 | 75% | DEEP | 6-module WASM toolkit. JS fallback is CHARACTER HASHING | R32 |
| reasoningbank_wasm_bg.js | agentic-flow | 556 | 100% | DEEP | wasm-bindgen auto-generated, 5 async methods | R32 |
| sona-agentdb-integration.ts | agentic-flow | 458 | 62-68% | DEEP | **DEAD CODE**: imports `SonaEngine` from `@ruvector/sona` and `agentdb` — NEITHER installed. Zero production consumers, no barrel export. "150x-12,500x" performance claim is hardcoded comment string (no benchmark). Dual-path query (AgentDB HNSW + SONA findPatterns + merge) architecturally genuine but unreachable. Source of marketing claims propagated across 50+ docs | R137 |

### AIDefence Security Module (R92)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| npm/packages/ruvbot/src/security/AIDefenceGuard.ts | ruvbot | 763 | 82-88% | DEEP | STANDALONE — 28 genuine regex patterns. `aidefence` npm dep listed but never imported. enablePolicyVerification has ZERO implementation behind it. behaviorBaseline in-memory only | R92 |
| npm/packages/ruvbot/tests/unit/security/aidefence-guard.test.ts | ruvbot | 235 | — | DEEP | 37 tests, all mocks, zero verification of actual behavioral analysis methods. Tests never exercise enableBehavioralAnalysis path | R92 |
| simulation/scenarios/aidefence-integration.ts | agentdb | 166 | 25% | DEEP | Simulation-only scenario scaffold. Hardcoded threat data, commented-out causal links, EmbeddingService loaded but unused in threat analysis. NOT runtime integration | R92 |

### Prime-Radiant Storage

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| crates/prime-radiant/src/storage/postgres.rs | ruvector-rust | 1,082 | 78-82% | DEEP | Race condition in store_witness(), full-scan find_similar(), no sheaf structure. Feature-gated (sqlx postgres) | R107 |
| crates/prime-radiant/src/storage/file.rs | ruvector-rust | 804 | 85-90% | DEEP | WAL with Blake3 checksums. CRITICAL: commit_wal() never sets committed=true. find_similar() O(n) scan | R108 |
| crates/prime-radiant/src/storage/memory.rs | ruvector-rust | 731 | 88-92% | DEEP | parking_lot::RwLock. store_witness() bug: never writes witnesses_by_action mapping. 9 unit tests | R108 |
| crates/prime-radiant/src/storage/mod.rs | ruvector-rust | 576 | 82-86% | DEEP | HybridStorage = FileStorage only (no postgres field). StorageFactory only creates InMemoryStorage | R108 |

### V3 MCP Tool Layer (R138)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| v3/mcp/tools/index.ts | claude-flow | 445 | 82% | DEEP | Central V3 MCP tool hub: 82 tools from 12 groups via getAllTools(). SONA tools (14) are full facades — fabricated speedup calc (`estimatedBruteForce = searchLatency * 1000`), LoRA handlers are no-ops (output = input.input). SONAState is in-memory Maps only, zero persistence, zero AgentDB. memory-tools.ts claims ADR-006 but has NO agentdb import. 4 groups connected to real backends (hooks, worker, federation, agent) | R138 |
| v3/mcp/server.ts | claude-flow | 792 | 88-92% | DEEP | V3 MCP server bootstrap. Calls getAllTools() to register all 82 tools. Zero calls to memory-initializer, AgentDB, or EmbeddingService — boots completely standalone with no memory backend initialization | R138 |
| v3/@claude-flow/mcp/src/server.ts | claude-flow | 1,134 | 88-92% | DEEP | Library MCP server: 14 methods, 9 sub-registries, 4 built-in tools. Most sophisticated MCP implementation in project. Zero memory/AgentDB/embedding references anywhere in file or imports — pure protocol shell. All domain tools must be externally registered | R138 |

### V3 @claude-flow/memory (R136)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| v3/@claude-flow/memory/src/agentdb-adapter.ts | claude-flow | 1,038 | 68-73% | DEEP | MISNAMED: ZERO AgentDB connection — plain Map<string,MemoryEntry> in-memory store. Persistence PLACEHOLDER (loadFromDisk/saveToDisk empty stubs). R20 NOT fixed (embeddingGenerator optional, 2/4 factories omit it). Cache manager production-quality (O(1) LRU, TTL, L1/L2). COMPETES with agentdb-backend.ts | R136 |
| v3/@claude-flow/memory/src/controller-registry.ts | claude-flow | 1,026 | 78-82% | DEEP | 29 controllers declared, 28 in init levels, 4 null placeholders. ~19 delegate to AgentDB via dynamic import('agentdb'). CRITICAL path traversal no-op. createEmbeddingService() fallback returns zero-filled Float32Array. AgentDB typed as `any`. Level-ordered init with Promise.allSettled GENUINE | R136 |
| v3/@claude-flow/memory/src/index.ts | claude-flow | 595 | 78-82% | DEEP | UnifiedMemoryService sole public API, hardcoded to AgentDBAdapter. createHybridService misleading — AgentDB-only, not hybrid per ADR-009. Two parallel module systems: flat barrel vs orphaned DDD subdirectories | R136 |
| v3/@claude-flow/memory/src/memory-initializer.ts | claude-flow | 2,564 | 72-78% | DEEP | OWN 3-tier embedding pipeline separate from AgentDB/EmbeddingService. options.backend config written to metadata but NEVER influences backend selection. HNSW lazy-loaded via @ruvector/core — silent degrade to SQLite | R136 |
| v3/@claude-flow/memory/src/memory-bridge.ts | claude-flow | 1,773 | 82-85% quality / 0% runtime | DEEP | ENTIRE FILE NOT COMPILED into npm dist — dead at runtime. Intended AgentDB V3 bridge via ControllerRegistry. All 28 functions return null at runtime. BM25 + hybrid scoring implementation correct but unreachable | R136 |
| v3/@claude-flow/memory/src/hnsw-index.ts | claude-flow | 1,014 | 72-78% | DEEP | Pure JS HNSW — no native hnswlib-node or ruvector. Default path via UnifiedMemoryService ALWAYS uses this. CRITICAL: getRandomLevel() wrong (p=0.5 vs standard p=1/M). Zero persistence | R136 |
| v3/@claude-flow/memory/src/auto-memory-bridge.ts | claude-flow | 957 | 82-88% | DEEP | No direct HNSW integration despite claims. Uses keyword/tag-based classification, not embeddings for retrieval | R136 |
