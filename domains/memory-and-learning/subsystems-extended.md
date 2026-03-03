## 5. Memory-and-Learning Subsystem Overview

(Updated R73 to reflect post-task.ts findings on hook registration mismatch and matts.ts orphaning)

### Cluster A: ReasoningBank TS (Judge→Distill→Consolidate→MaTTS)

**Status (R73 updated)**: 60-88% genuine, FOUR-tier layering, hook registration broken, matts.ts orphaned

**Architecture**:
- `index.ts` (136 LOC, ~75%): v1.7.1 barrel BYPASSES backend-selector, hardcodes HybridReasoningBank. Dual API (class + functions).
- `index.js` (101 LOC, 100%): v1.0.0 pure JS barrel. FOURTH ReasoningBank layer. backend-selector routes HERE, not to HybridBackend.
- `wasm-adapter.ts` (170 LOC, 92-95%): **10th GENUINE WASM** bridge to Rust ReasoningBank workspace (R67: core 88-92%, storage 94%, learning 95-98%).
- `config.ts`/`config.js` (241/170 LOC, 80-85%): Authoritative config. Embedding provider mismatch, no db_path config.
- `embeddings.js` (114 LOC, ~60%): **14th hash-based embedding**. 3rd independent embedding system in ReasoningBank alone.
- `schema.ts`/`schema.js` (89/5 LOC, 92-95%/0%): Type-safe bridge to queries.ts 7-table schema. schema.js = empty stub.

**Algorithms**:
- `post-task.ts` (128 LOC, 78-82%): **HOOK MISMATCH**. Pipeline complete but hook-handler.cjs stub bypasses judge→distill→consolidate.
- `judge.js` (129 LOC, 54%): BIMODAL. LLM-as-Judge genuine (78-82%) but bypasses MCP store, creating parallel data flow. Security risk: API key in headers.
- `distill.ts` (230 LOC, 78-82%): LLM-based knowledge extraction. 13th hash fallback. Template fallback broken.
- `consolidate.ts` (259 LOC, 82-86%): LSH bucketing, batch queries, contradiction detection. No EWC++, no matts integration.
- `matts.ts` (309 LOC, 85-88%): **ORPHANED**. Test-Time Scaling correct. Zero calls from post-task. 6th routing system (ModelRouter).

**Utilities**:
- `pii-scrubber.js` (99 LOC, 88%): Genuine 14 regex patterns. Production-ready.
- `mmr.ts` (80 LOC, 90%+): **GENUINE MMR** (Carbonell & Goldstein 1998). Correct cosine similarity.

**Issues**:
- **FOUR ReasoningBank layers**: index.ts → HybridBackend → index.js → core. Zero unification.
- **3 independent embedding systems** in ReasoningBank alone: embedding-service.ts (R51), embeddings.ts (R72), embeddings.js (R73).
- **5th disconnected persistence layer**: patterns/pattern_embeddings vs trajectories tables.
- **Hook registration broken**: post-task.ts never executes, matts.ts orphaned.
- **backend-selector BYPASSED**: Routes to old index.js, not HybridBackend.

### 5l. ruvllm Context Module (R104)

The `crates/ruvllm/src/context/` module is now fully DEEP (7/7 files). Architecture:

- **IntelligentContextManager** (context_manager.rs, 82-88%): Top-level orchestrator. Composes `AgenticMemory` + `SemanticToolCache` as owned members. Does NOT own `EpisodicMemory`, `WorkingMemory`, or `ClaudeFlowBridge` despite co-location. Fake recency (hardcoded 3600s). All ModelTokenLimit variants return 200_000.
- **AgenticMemory** (agentic_memory.rs, R102 DEEP): 4-type memory (Working/Episodic/Semantic/Procedural). consolidate() returns zeros stub. 1st non-hash HNSW store.
- **SemanticToolCache** (semantic_cache.rs, 88-92%): **2nd non-hash HNSW store in ruvllm**. MD5 exact path + HNSW cosine path. Tool-specific (not general KV). LRU eviction removes ONE entry per call.
- **EpisodicMemory** (episodic_memory.rs, 88-92%): **3rd non-hash HNSW store**. Standalone — NOT referenced by AgenticMemory.episodic (parallel systems). MemoryCompressor: genuine top-K by reward + centroid; compress_embedding() = truncation.
- **WorkingMemory** (working_memory.rs, 82-88%): Attention decorative — decay computed, eviction is FIFO ignoring weights. O(n) eviction scan on overflow (C177).
- **ClaudeFlowBridge** (claude_flow_bridge.rs, 65-72%): CLI-subprocess shell adapter. Every call spawns `npx @claude-flow`. No Rust API. 5th routing surface. No EmbeddingService (C179). Multiple correctness bugs (stats before success, fabricated created_at, line count for sync).
- **reasoning_bank/consolidation.rs** (88-92%): Genuine EWC++. `&self` bug blocks actual mutation (C178). consolidate_fisher() destroys per-pattern Fisher by averaging to index 0.
- **reasoning_bank/trajectory.rs** (78-82%): TrajectoryRecorder builder. Hardcoded quality scores. context_embedding never populated.
- **claude_flow/flow_optimizer.rs** (70-75%): Hardcoded improvement percentages. 18th pseudo-embedding (sinusoidal sweep).
- **claude_flow/task_classifier.rs** (72-78%): Keyword pattern-list classifier (no ML). No ADR-008 wiring.

**Integration gaps**: context_manager.rs composes only 2 of 7 context siblings. ClaudeFlowBridge adds a 5th parallel routing path with no connection to ADR-008's 3-tier model routing. TWO parallel episodic systems (standalone EpisodicMemory vs AgenticMemory.episodic) with zero cross-reference. R20 root cause deepened: ClaudeFlowBridge (Rust) has zero EmbeddingService, extending the gap into the Rust crate layer (C179).

### 5m. prime-radiant Coherence Module (R105)

The `crates/prime-radiant/src/coherence/` module is now fully DEEP (5/5 files): energy, history, incremental, spectral, mod. Implements sheaf Laplacian coherence energy tracking over a knowledge graph.

- **CoherenceEngine** (engine.rs, R37 DEEP): Entry point. Delegates to Energy/Spectral subsystems.
- **energy.rs** (88-93%): E(S) = sum(w_e * |r_e|^2). Three-tier SIMD dispatch (scalar/wide-crate/config). Blake3 fingerprint for staleness. Zero-allocation compute_residual_into(). Hotspot ranking via sort. debug_assert_eq! (release unsafe). f32 accumulation risk on large graphs.
- **history.rs** (85-90%): EnergyHistory with running-sum for O(1) statistics. OLS trend (integer x-axis). Anomaly detection with configurable threshold. clear() bug (doesn't reset total_entries/anomaly_count). 12 unit tests.
- **incremental.rs** (83-88%): Dirty-edge cache, full_recompute_threshold (30%), par_iter (shared &self reference risk). O(n) history trim (should VecDeque). energy_trend() reversal bug in iteration order. 6 unit tests.
- **spectral.rs** (75-80%): **CRITICAL DEFLATION BUG (C180)** — deflate_matrix() uses wrong formula (lambda*I instead of lambda*v*v^T). Finds largest, not smallest eigenvalue. Single-step drift noisy. Two implementations (nalgebra vs power-iteration) via feature flag. 8 tests pass despite math being wrong.
- **mod.rs** (88-92%): Flat re-export of 24+ symbols. ResidualCache alias signals API migration. ASCII architecture diagram and sheaf formula in module doc.

**Verdict**: Coherence energy infrastructure is genuine (energy/history/incremental ~86% weighted). Spectral drift detection is architecturally sound but mathematically broken at the deflation step, making multi-eigenvalue coherence analysis unreliable.

### 5n. ruvector-attention Training Module (R109)

The `crates/ruvector-attention/src/training/` module is now fully DEEP (4/4 files): optimizer, loss, curriculum, mining. These implement standard metric-learning training infrastructure.

**Quality gradient**: optimizer.rs (88-92%) > loss.rs (85-90%) > curriculum.rs (85-90%) > mining.rs (78-84%)

**optimizer.rs (88-92%)**: Implements Optimizer trait (Send+Sync) with SGD+Nesterov, Adam, AdamW, and LearningRateScheduler (linear warmup + cosine decay). All algorithm math is textbook correct — Adam bias correction follows Kingma & Ba 2014, AdamW uses proper decoupled weight decay. SGD velocity buffer auto-resizes on parameter dimension mismatch (silently resets momentum state — no warning). No optimizer state serialization (training cannot be checkpointed mid-run). Optimizers operate on raw &mut [f32] slices with no autograd graph — callers must compute and supply gradients externally.

**loss.rs (85-90%)**: InfoNCE loss gradient derivation for anchor is mathematically correct (quotient rule through cosine similarity, numerically stable log-sum-exp). LocalContrastiveLoss implements classic max(0, d_pos - d_neg + margin) triplet loss with correct hinge subgradient and 3 reduction modes. SpectralRegularization contributes to loss but returns zero gradient silently (H243 — makes spectral regularization gradient-free in composed training loops).

**curriculum.rs (85-90%)**: CurriculumScheduler and TemperatureAnnealing with Exponential/Linear/Cosine/Constant modes. Core logic is sound for the common case. ln(0) bug with final_temp=0 in Exponential mode produces NaN output (H244 — full annealing to zero is a common target).

**mining.rs (78-84%)**: Four triplet mining strategies. Core distance computations correct. Hardcoded seed=42 in both stochastic strategies defeats the statistical diversity purpose of randomized mining (H245). InBatchMiner and HardNegativeMiner cannot be composed due to incompatible interfaces (H246 — index-based vs embedding-slice-based APIs with no bridge).

**Verdict**: Training module is textbook correct at the algorithm level (optimizer math, loss gradients, annealing schedules all match literature). Three boundary bugs degrade practical utility: SpectralRegularization silent zero gradient, ln(0) in annealing, and non-stochastic deterministic mining.

### 5o. prime-radiant Cohomology Neural Layer (R109–R110)

The `crates/prime-radiant/src/cohomology/` submodule has grown with neural.rs (R109) and diffusion.rs (R110).

**cohomology/neural.rs (82-87%)**: Three-layer sheaf GNN architecture.
- SheafNeuralLayer: Xavier init, linear transform, sheaf Laplacian diffusion (x = x - alpha * L * x), activation, optional residual (dimension-gated), layer normalization. 5-step pipeline matches graph diffusion literature. Theoretically sound.
- SheafConvolution: Self-weight (W_self) and neighbor-weight (W_neigh) matrices matching GraphSAGE pattern, mean neighbor aggregation. Restriction maps default to identity — comment explicitly says "For identity restriction, just add neighbor" (H247). This makes the layer standard GCN, not a true sheaf convolution.
- CohomologyPooling: 5 strategies (Mean/Max/Sum/Attention/TopK). "Attention" pooling is L2-norm weighted (not learned). spectral_weighting flag declared but never used (H248 — dead configuration).
- GELU activation uses sigmoid approximation constant 1.702 (lower accuracy than tanh-based 0.044715 variant). 4 tests cover shape correctness and basic math.

**cohomology/diffusion.rs (75-80%)**: Explicit Euler sheaf diffusion with two CRITICAL bugs:
- C184: Hard clamping after gradient step destroys monotone energy decrease — the mathematical foundation of diffusion-based optimization.
- C185: Adaptive diffusion initializes best_section before inner trial loop — returns unchanged section when all candidate steps are uphill.
- Additional HIGH issues: no CFL stability check (H249), residual_obstruction is a heuristic threshold not a cohomological projection (H250), energy computation assumes identity restrictions (H251 in transport/centroid_ot.rs, but also relevant here: compute_node_energies wrong for general restrictions).

**transport/centroid_ot.rs (78-84%)**: K-means centroid OT approximation is functional. Full Sinkhorn implementation present but orphaned (#[allow(dead_code)], never called) — config fields sinkhorn_reg and sinkhorn_iterations stored but unused (H251).

**Verdict**: The neural layer architecture is theoretically well-designed but the sheaf-specific innovations are not yet implemented (restriction maps default to identity, spectral weighting dead). Diffusion has two correctness-breaking bugs that invalidate convergence guarantees. Transport provides a working centroid approximation but buries the mathematically rigorous Sinkhorn implementation as dead code.

### 5p. V3 Memory Layer (@claude-flow/memory) (R136)

The V3 memory layer is the runtime "brain" of claude-flow V3 — 7 files, ~8,967 LOC. It represents the INTENDED production memory system but is structurally broken at multiple layers.

**Architecture (designed vs actual)**:

The DESIGNED architecture (ADR-053) routes through memory-bridge.ts (28 bridge functions, hybrid BM25+semantic scoring, MutationGuard, ExplainableRecall) → ControllerRegistry (29 controllers with level-ordered init) → AgentDB native layer. This path scores 82-85% on code quality.

The ACTUAL runtime path is: index.ts UnifiedMemoryService (14 x 1-line delegation wrappers) → AgentDBAdapter (MISNAMED — plain Map<string, MemoryEntry>, zero AgentDB connection) → hnsw-index.ts (PURE JAVASCRIPT HNSW from scratch). memory-bridge.ts is NOT compiled into the npm dist (C191) and all 28 bridge functions return null, causing the entire system to fall back to memory-initializer.ts's sql.js path.

**File quality gradient**:
- auto-memory-bridge.ts (82-88%): HIGHEST — genuine ADR-048 bidirectional sync, PageRank, label propagation, 40+ tests
- memory-bridge.ts (82-85% code / 0% runtime): Correct BM25, hybrid scoring, but dead at runtime
- controller-registry.ts (78-82%): Genuine Promise.allSettled graceful degradation, but security flaws (C192)
- index.ts (78-82%): Clean barrel but misleading createHybridService naming
- memory-initializer.ts (72-78%): Own 3-tier embedding pipeline (14th parallel subsystem), SQL injection (C193)
- hnsw-index.ts (72-78%): Correct heaps and Algorithm 2 search, but broken level distribution (C187) and zero persistence (C188)
- agentdb-adapter.ts (68-73%): Misnamed, persistence stubs, R20 root cause replicated

**THREE embedding pipelines**: (a) memory-initializer.ts: @xenova/transformers → agentic-flow → hash (never references EmbeddingService); (b) auto-memory-bridge.ts LearningBridge: sin(hash) 768-dim; (c) EmbeddingService: never connected. Dimension mismatch: metadata 768 vs xenova 384 vs hash 128.

**Key findings (8C, 12H)**:
- C186-C188: Parallel embedding subsystem, broken HNSW level distribution, zero vector persistence
- C189-C191: AgentDBAdapter misnamed/no persistence, memory-bridge dead at runtime
- C192-C193: Path traversal no-op, SQL injection
- H252-H253: R20 root cause replicated, default path never reaches native HNSW
- H254-H263: Facade quantization, shallow health, zero-vector fallback, dimension mismatch, dead DDD layers

**Verdict**: The V3 memory layer is architecturally ambitious (ADR-053 7-phase bridge, 29 controllers, hybrid scoring) but the intended architecture is dead at runtime. The actual runtime path is an in-memory Map with a pure JS HNSW that has a fundamentally broken level distribution and zero persistence. R20's root cause (embeddings never initialized) is structurally replicated. The best code (memory-bridge.ts) is not compiled; the worst code (agentdb-adapter.ts) is the sole runtime backend.

### 5q. V3 Intelligence Layer — intelligence.ts and sona-optimizer.ts (R140)

R140 deep-reads the two memory-adjacent files in `v3/@claude-flow/cli/src/memory/` that implement the CLI-visible intelligence and routing optimization layers.

**intelligence.ts (985 LOC, 60-65% weighted)**:

The module header (line 8) claims "Pattern search: O(log n) with HNSW" but `LocalReasoningBank.findSimilar()` (lines 357-385) is a brute-force linear scan with cosine similarity — O(n) over the patternList array. Zero HNSW index is instantiated, zero hnswlib-node import, zero @ruvector import. This is a CRITICAL false complexity claim (C194, confirmed by R140 DEEP read).

`LocalSonaCoordinator` implements a pre-allocated circular buffer for signal recording — O(1) insertion, genuine and benchmarked at <0.05ms in headless.ts (10,000 iterations). However `SonaConfig` exposes `loraLearningRate` (0.001), `loraRank` (8), and `ewcLambda` (0.4) fields — none of which are ever read or used in any code path. The SONA branding is applied to what is literally a circular buffer with a JSON persistence layer (C195, H264).

The module is NOT dead code — it has 14+ real consumers: index.ts re-exports (line 583), runtime/headless.ts benchmark harness (line 24), commands/neural.ts (9 import sites), and mcp-tools/embeddings-tools.ts (4 import sites). `compactPatterns()` runs an O(n^2) all-pairs cosine similarity with no indexing for up to maxPatterns=5000 (12.5M comparisons, H264). `recordStep()` has a genuine 2-tier embedding fallback: memory-bridge.ts → memory-initializer.ts (H272). Zero imports of @ruvector/*, hnswlib-node, or hnsw_router.rs — no native acceleration path (H265).

**sona-optimizer.ts (842 LOC, 78-82%)**:

SONAOptimizer implements real Bayesian-style confidence updates: success increments `conf += CONFIDENCE_INCREMENT * (1 - conf)`, failure decrements `conf -= CONFIDENCE_DECREMENT * conf`. Temporal decay via `exp(-DECAY_RATE * days)`. Pattern pruning by `score = confidence * recency`. This is a genuine ML feedback loop for agent routing decisions (H266, finding tagged positive).

Critical naming conflict: `sona-optimizer.ts` is an AGENT-ROUTING optimizer (learns which agent type to dispatch). `sona-tools.ts` claims to provide HNSW vector search speedup (R138 finding: fabricated via `estimatedBruteForce = searchLatency * 1000`). Neither imports the other. The "SONA" brand covers two entirely unrelated subsystems (H267). This confirms the systemic finding that claude-flow V3 has TWO independent SONA subsystems with zero architectural connection.

`getSONAOptimizer()` singleton is invoked in `hooks-tools.ts` at `hooksTrajectoryEnd` handler, `hooksIntelligence` handler, and `getSONAStats` — making sona-optimizer.ts the ONLY V3 memory subsystem actually wired into the hooks pipeline end-to-end (H271). The Q-learning integration lazy-loads `q-learning-router.js` from `../ruvector/q-learning-router.js` (882 LOC) but that file attempts `@ruvector/core` native import — likely absent in production, silently degrading to JS Q-table (H268).

**Performance bug**: `saveToDisk()` is synchronous `writeFileSync` called on every `processTrajectoryOutcome` invocation, with no debounce or throttle despite the comment at line 314 saying "Persist to disk (debounced)". For rapid trajectory processing in the hooks pipeline, this is a blocking I/O bottleneck (H270).

**Subsystem verdict**: intelligence.ts is a genuine-but-mislabeled subsystem (circular buffer sold as HNSW, O(n) sold as O(log n), dead LoRA config). sona-optimizer.ts is the most genuine V3 memory module — real Bayesian routing optimization, the only end-to-end hooks pipeline connection, with a fixable sync-IO bug as the main production concern. The two SONA subsystems (optimizer + tools) share a brand name but serve entirely different purposes and should not be confused.

### 5r. Rust Compilation Audit Results (R141)

R141 systematically ran `cargo check` and `cargo test --lib` across all 4 workspaces: reasoningbank, sublinear-time-solver, ruvector (agentdb context), ruv-FANN. The following findings are directly relevant to memory-and-learning:

**reasoningbank workspace** (4 of 5 crates PASS, 1 FAIL):
- reasoningbank-core: PASS check+test, 12/12 tests green. 773 LOC.
- reasoningbank-storage: PASS check+test, 9/9 tests green. 1,403 LOC. SQLite/async/migrations confirmed working.
- reasoningbank-learning: PASS check+test, 7/7 tests green. 788 LOC. 8 deprecation warnings (AsyncLearner self-references). Confirms R85 assessment.
- reasoningbank-network: PASS check+test, 18/18 tests green. 2,647 LOC. QUIC, NeuralBus, gossip, streams confirmed compiling.
- **reasoningbank-mcp: FAIL (6 errors, C197)** — Two distinct StorageConfig types used interchangeably. MCP entry point for the Rust ReasoningBank layer is broken and unusable.
- reasoningbank-wasm: FAIL native check (cfg-gated wasm module), PASS wasm32-unknown-unknown — expected for WASM-only crate.

**sublinear-time-solver workspace** (mixed):
- Root crate: PASS check (413 warnings), FAIL test (7 errors: f64 not Ord, missing EntanglementValidator methods, H275). Tests have NEVER run.
- bit-parallel-search: PASS check+test. 4/4 tests pass. Only fully clean crate.
- **neural-network-implementation: FAIL check (106 errors, C196)** — ModelParams trait not dyn-compatible (E0038). 17,294-LOC "best code in project" has never compiled.
- psycho-symbolic-reasoner: PASS check+test, but 0 tests across 3 crates (hollow test suite).
- temporal-compare: PASS check+test, 3/3 tests green (quantization).
- **temporal-lead-solver: FAIL check (21 errors, H273)** — E0616 private nalgebra .data field access (7x).
- **rustc-hyperopt: FAIL check (3 errors, H274)** — blake3::Hash missing LowerHex.
- temporal-neural-solver-wasm: PASS wasm32 check.
- wasm-solver: PASS wasm32 check.

**Overall Rust audit verdict for this domain**: The ReasoningBank workspace is the most production-ready Rust cluster (4/5 crates compile and have passing tests). The sublinear-time-solver workspace is severely degraded — the flagship neural-network-implementation crate and 3 others fail compilation. The "never compiled" pattern is systemic: 106 errors in neural-network-implementation prove the "best code in project" claim was based on code reading alone, not compilation evidence.

