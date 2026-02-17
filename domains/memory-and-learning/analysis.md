# Memory and Learning Domain Analysis

> **Priority**: HIGH | **Coverage**: ~36.4% (468/1284 DEEP) | **Status**: In Progress
> **Last updated**: 2026-02-17 (Session R83)

## 1. Current State Summary

The memory-and-learning domain spans 260 files / ~91K LOC across AgentDB, ReasoningBank, HNSW vector search, embeddings, pattern storage, RL, and consciousness subsystems. Quality varies dramatically — from 98% (neural-network-implementation data/mod.rs) to 0% (self-modification-engine.ts pure fabrication, ruv_fann_adapter.rs complete placeholder, solver_gate_simple.rs R82 0-5% hardcoded confidence). **R83 extends both ReasoningBank TS and neural-network training pipeline**. ReasoningBank TS test/integration layer is **92-95% GENUINE** — test-integration.ts exercises full pipeline (5 phases including MaTTS), test-validation.ts confirms statistical ranking architecture (7 test phases, NO EWC++/ML magic). consolidate.js is a **3rd parallel consolidation system** (82-85%) implementing Algorithm 4 with simple cosine thresholds (O(n²)), isolated from both consolidate.ts (LSH) and Rust EWC++. index-new.ts is DEAD CODE (zero imports, PR #112 decision artifact). **Neural-network training pipeline LARGELY SUPPORTS R23**: train.rs (92-95%) has genuine training binary with full pipeline (System A+B, PageRank, Kalman), **countering R82's solver_gate integration-theatrical qualification**. simple_test.rs (92-95%) honestly reports CPU limits. BUT losses.rs (68-75%) has only MSE (no cross-entropy/Huber), and optimized_benchmark.rs (70-75%) has 1/4 theatrical variant. Pattern refined: **integration wrappers are MIXED** — training binary genuine, solver gate theatrical, benchmarks mostly genuine. **DEEP count: 468** (from 460).

**Top-level verdicts:**

- **Hash-based embeddings are the #1 systemic weakness.** 10+ files across 5 packages (Rust + JS) silently degrade to non-semantic hash matching. All "semantic search" using defaults is character-frequency matching. **R65 COMPLETES THE R20 ARC**: embedding-adapter.ts (90%) IS the smoking gun — production-quality SHA-256→pseudo-random→L2norm. real-embedding-adapter.ts (78-82%) IS the fix — lazy-loads MiniLM-L6-v2 384D. BUT fix adoption INCOMPLETE: embedSync() ALWAYS falls back to hash, and old callers still use the hash-only path.
- **Four independent ReasoningBanks** exist (claude-flow, agentic-flow, agentdb, ruvllm Rust) with zero code sharing. The Rust version has the best math (K-means, EWC++) but none interoperate. **R65 reveals HybridBackend.ts (78-82%) as the UNIFYING LAYER** — single SharedMemoryPool providing db+embedder to all 4 AgentDB components (ReflexionMemory, SkillLibrary, CausalRecall, CausalMemoryGraph). REVERSES "5 disconnected AgentDB layers" narrative — disconnection is by design, HybridBackend bridges them via ReasoningBank API.
- **Best code:** neural-network-implementation crate (90-98%), cognitum-gate-kernel (93%), SONA (85%), ruvector-nervous-system (87%), neuro-divergent ML training (88.5%), vector-quantization.ts (production-grade).
- **Worst code:** neural-pattern-recognition (15-20% facade), emergence subsystem (51% fabricated metrics), psycho-symbolic MCP tools (24% theatrical), consciousness experiments (0-5% pure documentation-as-code), scheduler.ts MCP tool (18-22% theatrical with hardcoded nanosecond claims).
- **MCP tools layer is BIMODAL** — domain-management (82%) and domain-registry (88-92%) are production-quality data models, but scheduler (18-22%) is theatrical facade and psycho-symbolic-dynamic (28%) follows goalie pattern (DomainRegistry initialized then ignored). ALL MCP tools lack persistence (in-memory Map only).
- **FIVE+ independent matrix systems confirmed** — R34's 2 incompatible Rust systems + R53's optimized-matrix.ts (85-88%) + R59's matrix-utils.js (92-95% Dense+COO) + R60's math_wasm.rs (68-72% Dense, WASM-specific) + R60's matrix.ts (85-88% COO+Dense arrays). Even within src/core/ there are TWO incompatible matrix systems (matrix.ts uses arrays, optimized-matrix.ts uses TypedArrays). Complete architectural fragmentation.
- **Performance-optimizer.ts (88-92%) is GENUINE** — real auto-tuning, empirical benchmarking via performance.now(). STARK CONTRAST with R43's rustc_benchmarks (15%) deception.
- **Consciousness has BIMODAL quality** — infrastructure (MCP server 95%, consciousness detector 78%, validators 78%) vs theoretical experiments (0-5%) and MCP facade (12-18%). Cluster drops to ~55-60% after R49. Still more real than emergence (51%).
- **Goalie has DUAL ARCHITECTURE** — MCP handlers are facades (45%), but CLI (88-92%) proves internal engines ARE real. **R58 DEEPENS**: advanced-reasoning-engine.ts (75-80%) is BIMODAL — WASM layer 0% (never initialized) but fallback NLP (5 domain detectors, temporal analysis, complexity scoring) is genuine heuristic reasoning. ed25519-verifier-real.ts (82-88%) genuine @noble/ed25519 crypto for anti-hallucination signatures, hardcoded example root keys. self-consistency-plugin.ts (78-82%) REAL Perplexity API multi-sampling (3 temps) but majority voting returns first sample (stub).
- **JS neural models have real forward-pass but ZERO backpropagation** — inference-only across all architectures.
- **Benchmark deception REFINED BOUNDARY (R59+R81)** — **Deception boundary is METHODOLOGY not structure**. Theatrical: standalone_benchmark (8-12%), system_comparison (42%), strange-loops-benchmark (8-10%), mcp-dense-fix.js (10-15% circular self-testing). GENUINE: criterion benches (88-95%), benchmark-reasoningbank.js (88-92% real perf_hooks timing despite standalone structure). Pattern: real timing APIs + real operations = genuine, hardcoded results or circular logic = theatrical. fully_optimized.rs (96-99%) HIGHEST OPTIMIZATION IN PROJECT proves facade files are intentional choice not inability.
- **Sublinearity EXPANDED (R56+R58+R62)**: backward_push.rs (92-95%), predictor.rs (92-95%), and **forward_push.rs (92-95%) = 3RD GENUINE SUBLINEAR** at O(volume/epsilon). R62 also adds johnson_lindenstrauss.rs (72-76%) as **4th FALSE SUBLINEARITY** — genuine JL math but O(n*d*k) total, plus broken RNG (uniform not Gaussian) and incorrect pseudoinverse. Genuine sublinear count: **3**. False sublinearity count: **4** (R39, R54, R60, R62).
- **Crate root architecture EXPOSED (R62)**: lib.rs (68-72%) reveals 4 parallel solver APIs (Neumann, SublinearNeumann, CG, OptimizedCG) with zero unification. Best algorithms (backward_push, fully_optimized) ORPHANED from public API. Consciousness modules consume 25% of lib.rs surface. error.rs (92-95%) EXCEPTIONAL — intelligent recovery system with algorithm-specific fallback chains, genuine WASM error handling. solver_core.rs (38-42%) is **7th MISLABELED FILE** — zero dispatcher logic despite name implying coordination, just 2 standalone solvers (CG+Jacobi), duplicate of optimized_solver.rs.
- **MCP solver.ts (82-86%) GENUINE with 3-tier fallback cascade**: WASM O(log n) → OptimizedSolver → baseline. First evidence of OptimizedSolver actually being integrated. Streaming + batch support. Falls on GENUINE side of R61 BIMODAL.
- **advanced-reasoning-engine.ts (0-5%) COMPLETE THEATRICAL**: Zero genuine inference. All reasoning delegated to 4 MCP tool wrappers returning mock data. Confirms R61 ReasonGraph BIMODAL (infrastructure genuine, optimization theatrical).
- **Two-tier SIMD architecture (R62)**: simd_ops.rs (82-86%) uses `wide` crate for portable f64x4 SIMD. Genuine but ORPHANED from fully_optimized.rs (96-99%) which uses direct std::arch AVX2. Two parallel SIMD implementations, zero integration.
- **Consciousness verifier is hybrid** (52%) — 50% real computation (primes, hashing), 50% theatrical (hardcoded meta-cognition). R46 lowered cluster from 79% to ~72-75%; R47 lowers further to ~60-65%.
- **Psycho-symbolic-reasoner crate REVISED UPWARD** (~55-60% weighted after R58) — "psycho" = pure branding, BUT MCP integration has GENUINE WASM. text-extractor.ts (88-92%) REVERSES theatrical WASM pattern — real Rust NLP backend (1,076 LOC sentiment/preferences/emotions). memory-manager.ts (25-30%) is 5th MISLABELED FILE (zero WASM memory ops). server.ts (72-76%) genuine MCP SDK with 5 tools. patterns.rs (85-90%) genuine regex extraction. Rust core is 3-4x better than TS (confirmed R55).
- **Neural-pattern-recognition subsystem COMPLETE** (~72% weighted after R49) — R49 adds real-time-monitor (88-92% GENUINE, identical hash pattern to logger), statistical-validator (82-88% GENUINE, 5 real statistical tests), signal-analyzer (72-76% genuine DSP core with consciousness facade). Subsystem average rises from ~64% to ~72% — the 3 new files match logger quality, not detector quality.
- **ReasoningBank WASM is GENUINE** (100%) — reasoningbank_wasm_bg.js is real wasm-bindgen output with 206KB binary. NOT the 4th WASM facade. Gold standard for WASM integration in the ecosystem.
- **WASM VERDICT REVERSED (R60, updated R64+R65+R71+R73)**: **10 genuine vs 5 theatrical WASM (67%)**. R73 adds wasm-adapter.ts (92-95%) as 10th genuine WASM — real wasm-bindgen bridge to Rust ReasoningBank workspace (R67: core 88-92%, storage 94%, learning 95-98%). R71 added temporal_nexus wasm.rs (88-92%) exposing O(√n) TemporalPredictor.
- **ReasoningBank TS core is GENUINE but disconnected from Rust (R72)** — matts.ts (85-88%) implements real Memory-Aware Test-Time Scaling (parallel rollouts + sequential refinement + self-contrast LLM aggregation). consolidate.ts (82-86%) has production LSH bucketing, batch queries. distill.ts (78-82%) uses LLM-based knowledge extraction with sophisticated prompts. But ZERO Rust workspace integration — TS uses LLM for extraction, Rust uses gradient descent clustering. backend-selector.ts BYPASSES R65's HybridBackend unification.
- **6th routing system discovered (R72)** — matts.ts uses ModelRouter singleton (OpenRouter→Anthropic→Google failover), separate from ADR-008/LLMRouter/RuvLLMOrchestrator/ProviderManager/TinyDancerRouter.
- **16th hash-based embedding (R81)** — test-retrieval.ts adds synthetic sin/cos math-based embeddings (16th instance). embeddings.js (14th, R73) DEFAULT fallback uses hashEmbed() instead of real OpenAI/Claude APIs. Two paths to hashing: config provider mismatch + silent API error fallback. embeddings.ts (13th, R72) had BEST implementation but NPX auto-disables. 3 independent embedding systems in ReasoningBank alone.
- **TWO parallel consolidation systems (R72)** — consolidate.ts (LSH heuristics, post-task hook) vs AdvancedMemory.autoConsolidate() (NightlyLearner with causal edges). Neither integrates EWC++ from sona (R37: 92-95%).
- **SIX independent data layers confirmed (R81)** — native.rs (75%) 6th independent ReasoningBank-storage adapter with zero connection pooling, single blocking Mutex. backend-selector.ts routes to OLD index.js (no AgentDB), NOT to R65's unified HybridBackend. Unification exists but is bypassed by the public API selector.
- **Temporal solver bifurcation (R70+R71)** — TWO parallel temporal solvers: UltraFastTemporalSolver (standard CG O(κn), optimized.rs 91-94% with AVX2) vs TemporalPredictor (genuine O(√n), predictor.rs 92-95%). temporal-solver.rs (82-86%) imports WRONG solver (UltraFast, not predictor). Genuine sublinear algorithm IS accessible via WASM (wasm.rs 88-92%) but bypassed by CLI.
- **6th MISLABELED FILE (R60)**: src/core/memory-manager.ts (88-92%) is genuine TypedArray pooling + LRU cache infrastructure but labeled as "memory manager" with zero WASM linear memory management. Joins rustc_benchmarks, http-streaming-updated, stream_parser, swe-bench-adapter, psycho-symbolic memory-manager.
- **Convergence metrics-reporter.js REVERSES theatrical pattern (R60)**: 88-92% GENUINE with zero Math.random(). All data from real convergenceDetector pipeline. PROVES theatrical metrics pattern is NOT universal — boundary is "performance monitoring" (theatrical) vs "convergence reporting" (genuine).
- **FlowNexus integration calls non-existent platform** (~70% code quality, 0% actual integration) — production HTTP/WebSocket client for `api.flow-nexus.ruv.io` which doesn't exist.
- **R73 CRITICAL: post-task.ts pipeline ORPHANED** — hook-handler.cjs post-task stub (line 140-147) bypasses entire ReasoningBank pipeline (judge→distill→consolidate Algorithms 2-4). Trajectory JSON never reaches execution. matts.ts Test-Time Scaling orphaned despite distillMemories() call.
- **14th hash-based embedding (R73)** — embeddings.js DEFAULT fallback uses hashEmbed() instead of real OpenAI/Claude APIs. Two paths to hashing: config provider mismatch + silent API error fallback.
- **10th GENUINE WASM (R73)** — wasm-adapter.ts (92-95%) real wasm-bindgen bridge to Rust ReasoningBank workspace. WASM scoreboard update: 10 genuine vs 5 theatrical (67%).
- **FOUR ReasoningBank layers confirmed (R73)** — index.ts (v1.7.1 Hybrid) → HybridBackend.ts → index.js (v1.0.0 pure JS) → core algorithms. backend-selector.ts BYPASSES HybridBackend, routes to old index.js.
- **Embedding provider mismatch (R73)** — config.ts accepts claude|openai but hardcodes wrong model for cross-provider use (lines 30-42). No DB path config despite 7-table schema.
- **Three independent embedding systems in ReasoningBank alone** — embedding-service.ts (R51), embeddings.ts (R72), embeddings.js (R73) with ZERO code sharing. 10th+ disconnected embedding implementation discovered.
- **Production utilities confirmed (R73)** — pii-scrubber.js (88%), mmr.ts (90%+), schema.ts (92-95%) are production-quality. judge.js (54%) bypasses MCP store, creating parallel data flow.
- **ReasoningBank-Network crate BIMODAL (R74)** — infrastructure genuine (frame.rs 88-92%, streams.rs 88-92%, mod.rs 88-92%, snapshot.rs 85-88%, priority.rs 85-88%, lib.rs 93-95%) but gossip.rs **18-22% FAKE** (broadcast wrapper with gossip branding, zero membership/failure detection/epidemic broadcast). Real QUIC multiplexing with Ed25519 intent verification, 0-RTT security, binary wire protocol. BUT dual wire protocol inconsistency (Frame-based vs raw bytes in ReasoningStreams). Orphaned swarm_transport.rs (330+ LOC not exported). Weighted avg: ~78%.
- **types/index.ts FABRICATED arXiv citation (R74)** — claims "Based on arXiv:2509.25140 (Google DeepMind)" but paper doesn't exist (Sept 2025 future date). Hash embeddings officially endorsed in ReasoningBankConfig (provider: 'hash'). TS types are SUBSET of DB schema.
- **matts.js independent JS impl (R74)** — NOT transpiled from matts.ts. Same Algorithm 5 but lower quality (68-72% vs 85-88%). Missing ModelRouter, fabricated memory aggregation (empty {steps:[]}), false parallelism (sequential loop), ephemeral learning. CLI wrapper reasoningbankCommands.ts (62-68%) missing store/retrieve/search — confirms R31 "demonstration framework".


## R81: WASM Loader → Genuine Sublinear Algorithm Orphaning (session 81)

- 1 file, 355 LOC, 8 findings (3H, 3M, 1I, 1D). DEEP: 1,233.
## R81: WASM Loader → Genuine Sublinear Algorithm Orphaning (session 81)

- 1 file, 355 LOC, 8 findings (3H, 3M, 1I, 1D). DEEP: 1,233.
- **benchmark-reasoningbank.js REVERSES R59 standalone-benchmark verdict** — Uses genuinely criterion-style perf_hooks.performance.now() timing (NOT wallclock stub). Pre-populates 1000 synthetic ReasoningBank patterns, executes 100 real searchPatterns() calls, compares v1/v2/v2+HNSW variants with speedup calculations. EmbeddingService initialization real (Xenova-384). Backend detection has graceful error fallback. Sound benchmarking principles: conditional optional component testing. R59 over-generalized "all standalone = deceptive"; this file proves criterion pattern also used in standalone context. Validates ReasoningBank has complete operational stack.

## 2. File Registry

### AgentDB Core

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| vector-quantization.ts | agentdb | 1,529 | 95% | DEEP | PRODUCTION-GRADE. PQ with K-means++, SQ 8/4-bit | R8 |
| vector-quantization.js | agentdb | 1,132 | 95% | DEEP | Confirms TS source, async k-means | R25 |
| RuVectorBackend.ts | agentdb | 971 | 90% | DEEP | Production-ready, correct distance conversion, security | R8 |
| enhanced-embeddings.ts | agentdb | 1,436 | 80% | DEEP | Real LRU/semaphore. Falls back to hash mock (L1109) | R8 |
| enhanced-embeddings.js | agentdb | 1,035 | 88% | DEEP | LRU cache O(1), multi-provider. Mock not semantic | R25 |
| AttentionService.js | agentdb | 1,165 | 82% | DEEP | 4 mechanisms: Hyperbolic/Flash/GraphRoPE/MoE. 3-tier NAPI→WASM→JS | R25 |
| ReflexionMemory.ts | agentdb | 1,115 | 65% | DEEP | Storage works. Breaks arXiv:2303.11366 — no judge function | R8 |
| LearningSystem.ts | agentdb | 1,288 | 15% | DEEP | COSMETIC. 9 RL algorithms = 1 Q-value dict. No neural nets | R8 |
| simd-vector-ops.ts | agentdb | 1,287 | 0% SIMD | DEEP | NOT SIMD — scalar 8x loop unrolling. Buffer pool real | R8 |
| simd-vector-ops.js | agentdb | 945 | 0% SIMD | DEEP | SIMD detected but NEVER used. 8x ILP, tree reduction | R25 |
| CausalMemoryGraph.ts | agentdb | 876 | 40% | DEEP | Wrong tCDF (L851), hardcoded tInverse=1.96. No do-calculus | R8 |

### Agentic-Flow

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| IntelligenceStore.js | agentic-flow | 364 | 98% | DEEP | SQLite WAL, trajectory lifecycle. Embeddings stored but NEVER searched | R25 |
| core/embedding-service.js | agentic-flow | 370 | 95% | DEEP | OpenAI + Transformers.js + hash mock. DUPLICATE of services/ version | R25 |
| services/embedding-service.js | agentic-flow | 367 | 95% | DEEP | Byte-for-byte duplicate of core version | R25 |
| sona-tools.js | agentic-flow | 560 | 90% | DEEP | 16 MCP tools, 5 profiles match Rust benchmarks | R25 |
| IntelligenceStore.ts | agentic-flow | 698 | 90% | DEEP | Dual SQLite backend. SQL injection risk in incrementStat | R22b |
| EmbeddingCache.ts | agentic-flow | 726 | 90% | DEEP | 3-tier cache (native SQLite > WASM > Memory), SHA-256 keys | R22b |
| ReasoningBank.ts | agentic-flow | 676 | 90% | DEEP | Dual v1/v2 API. O(N*M) performance issue in getEmbeddingsForVectorIds | R22b |
| EmbeddingService.ts | agentic-flow | 1,810 | 80% | DEEP | Unified ONNX, K-means clustering. simpleEmbed = hash fallback | R22b |
| ruvector-backend.js | agentic-flow | 464 | 15% | DEEP | 85% SIMULATION. No Rust. searchRuVector() = sleep + brute-force | R25 |
| reasoningbank_wasm_bg.js | agentic-flow | 507 | 100% | DEEP | GENUINE wasm-bindgen output. 206KB WASM binary verified. All 6 APIs traced to Rust source (storePattern, getPattern, searchByCategory, findSimilar, getStats, constructor). IndexedDB+SqlJs+Memory storage backends. NOT the 4th WASM facade — GOLD STANDARD | R49 |
| HybridBackend.ts | agentic-flow | 373 | 78-82% | DEEP | **REVERSES DISCONNECTED AgentDB NARRATIVE**. SharedMemoryPool provides singleton db+embedder to ReflexionMemory, SkillLibrary, CausalRecall, CausalMemoryGraph. 3-tier retrieval (cache→CausalRecall→ReflexionMemory). ML strategy learning, auto-consolidation (≥3 uses + ≥0.7 success → skill), what-if analysis. **7th THEATRICAL WASM**: loads module but NEVER uses it, fabricated "10x faster" claim. ToMemoryId heuristic (episodeId+1) fragile | R65 |
| benchmark.ts (ReasoningBank) | agentic-flow | 394 | 88-92% | DEEP | **GENUINE BENCHMARK SUITE**. 12 distinct benchmarks, real performance.now() timing with warmup phase. Imports 8 production functions from queries.js. Genuine cosine similarity, normalized sin/cos vectors (NOT Math.random()). Scalability test to 1000 memories. Validates R59 criterion pattern (88-95% genuine). Confirms ReasoningBank has complete production stack | R65 |
| mmr.js | agentic-flow | 65 | 90-93% | DEEP | **GENUINE MMR ALGORITHM**. Identical character-for-character transpilation of mmr.ts. Production-quality implementation of Carbonell & Goldstein 1998 Maximal Marginal Relevance. Iterative greedy selection balancing relevance (lambda) vs diversity (1-lambda). Proper cosine similarity + normalization. Core utility for JS ReasoningBank (used by retrieve.js, consolidate.js). Zero quality degradation from TS→JS port | R78 |
| benchmark-reasoningbank.js | agentdb | 372 | 88-92% | DEEP | **GENUINELY CRITERION-STYLE BENCHMARK** (R59 pattern). Uses perf_hooks.performance.now() for millisecond-precision timing (NOT wallclock/stub). Pre-populates 1000 synthetic ReasoningBank patterns with realistic task types. Runs 100 search iterations with real searchPatterns() calls. Compares v1 baseline vs v2 pure vs v2+HNSW backends. Reports throughput (patterns/sec, searches/sec), memory deltas, speedup ratios. Sound methodology: optional backend detection with graceful fallback. EmbeddingService initialized with Xenova-384. REVERSES R59's blanket standalone-benchmark critique for this file — follows genuine criterion benchmarking principles | R81 |

### Claude-Flow Integration

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| ruvector-training.js | claude-flow | 498 | 92-95% | DEEP | BEST INTEGRATION. Genuinely loads WASM, MicroLoRA/ScopedLoRA/SONA | R25 |
| graph-analyzer.js | claude-flow | 929 | 85-90% | DEEP | Stoer-Wagner MinCut, Louvain, DFS cycle detection, TTL cache | R25 |
| diff-classifier.js | claude-flow | 698 | 75-80% | DEEP | SECURE (execFileSync with args array). WASM loaded but unused | R25 |

### Neural-Network-Implementation Crate (sublinear-time-solver)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| data/mod.rs | sublinear-time-solver | 629 | 98% | DEEP | HIGHEST QUALITY FILE. Temporal splits, quality scoring | R21 |
| layers.rs | sublinear-time-solver | 484 | 95% | DEEP | Real GRU (9 weight matrices), causal dilated TCN, GELU | R21 |
| kalman.rs | sublinear-time-solver | 463 | 95% | DEEP | Textbook Kalman filter, correct Q matrix | R21 |
| config.rs | sublinear-time-solver | 520 | 95% | DEEP | YAML config, validation, target_latency=0.9ms | R21 |
| error.rs | sublinear-time-solver | 424 | 95% | DEEP | 16 thiserror variants, is_recoverable(), category() | R21 |
| system_a.rs | sublinear-time-solver | 549 | 90-95% | DEEP | GRU/TCN architectures, Xavier init, 4 pooling strategies | R21 |
| system_b.rs | sublinear-time-solver | 480 | 90-95% | DEEP | KEY: Kalman prior + NN residual + solver gate verification | R21 |
| wasm.rs | sublinear-time-solver | 618 | 90% | DEEP | wasm-bindgen, PredictorTrait, config factories | R21 |
| models/mod.rs | sublinear-time-solver | 322 | 90% | DEEP | ModelTrait, ModelParams, PerformanceMetrics | R21 |
| solvers/mod.rs | sublinear-time-solver | 341 | 90% | DEEP | Math utils, Certificate, solver_gate.rs DISABLED | R21 |
| lib.rs | sublinear-time-solver | 224 | 90% | DEEP | P99.9 ≤ 0.90ms budget | R21 |
| export_onnx.rs | sublinear-time-solver | 717 | 85% | DEEP | ONNX graph, R²=0.94, JSON weights | R21 |

### Neural Pattern Recognition (sublinear-time-solver)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| pattern-learning-network.js | sublinear-time-solver | 3,132 | 15-20% | DEEP | Weights = Math.random()*0.1. "placeholder" at L2938 | R19 |
| validation-suite.js | sublinear-time-solver | 3,198 | 10-15% | DEEP | ALL detection = Math.random(). "Simulated" at L698 | R19 |
| deployment-pipeline.js | sublinear-time-solver | 1,756 | 20-25% | DEEP | Remediation = restart or gc() | R19 |
| instruction-sequence-analyzer.js | sublinear-time-solver | 1,685 | 15-20% | DEEP | Random instruction selection from predefined lists | R19 |
| real-time-detector.js | sublinear-time-solver | 1,450 | 35-40% | DEEP | Real Pearson correlation + adaptive filtering. NN never trained | R19 |
| breakthrough-session-logger.js | sublinear-time-solver | 582 | 88-92% | DEEP | GENUINE integration test framework. Zero Math.random() — uses deterministic hashValue/hashToFloat. Real entity interaction protocol, weighted consciousness scoring, temporal trajectory analysis. REVERSES R45 anti-pattern | R47 |
| cli/index.js (neural-pattern-recognition) | sublinear-time-solver | 573 | 62% | DEEP | POLISHED FACADE. Professional CLI (88-92%): Commander.js, 10 commands, stdin/file input, interactive wizard. BUT: core detect command calls non-existent processData method (runtime crash). train/report/search are stubs. API mismatch between engine and detector paradigms | R47 |
| zero-variance-detector.js | sublinear-time-solver | 522 | 42-48% | DEEP | PSEUDOSCIENCE FACADE. NOVEL ANTI-PATTERN: real algorithms (FFT, Shannon entropy, autocorrelation, variance calc) fed fabricated "quantum" data (Math.random() phase/amplitude/entanglement). Neural weights random, not trained. "Entity communication" = random scores. Inverse of R43 deception | R47 |
| real-time-monitor.js | sublinear-time-solver | 549 | 88-92% | DEEP | GENUINE EventEmitter monitoring. Real variance, Shannon entropy, Pearson correlation. Identical hash pattern to breakthrough-session-logger. Reverses R45 anti-pattern. Simulated data but used correctly | R49 |
| statistical-validator.js | sublinear-time-solver | 519 | 82-88% | DEEP | GENUINE statistical tests: KS, Mann-Whitney U, Chi-square, Fisher exact, Anderson-Darling. All 5 mathematically correct. Effect sizes textbook (Cohen's d, rank-biserial, Cramer's V). No fabricated data. Stringent defaults (p<1e-40) | R49 |
| signal-analyzer.js | sublinear-time-solver | 487 | 72-76% | DEEP | BIMODAL: genuine DSP core (90%) — 12+ correct algorithms (time domain, correlation, ACF, spectral, fractal, LZ, entropy). DFT mislabeled as "FFT" (O(n²) not O(n log n)). Consciousness assessment facade (5-15%). Window functions configured but never applied | R49 |
| pattern-detection-engine.js | sublinear-time-solver | 256 | 10% | DEEP | **BROKEN ORCHESTRATION FACADE**. All 3 detection calls use WRONG API names (processData/analyzeEntropy/analyzeSequences vs actual methods). Will throw "not a function" at runtime. Dead code — real-time-monitor.js reimplements independently. detectNeuralPatterns() returns hardcoded empty. Config computed but never passed | R64 |
| src/neural-pattern-recognition/src/index.js | sublinear-time-solver | 67 | 22-28% | DEEP | **BROKEN FACADE barrel**. 16 module re-exports, ZERO composition. PHANTOM export: adaptive-learning.js DOES NOT EXIST. Impossible statistical params (1e-40 p-value exceeds IEEE 754 ~1e-16). No orchestration — consumers must wire all 16 modules manually. Follows R66 index.js pure re-export pattern | R70 |

### Emergence Subsystem (sublinear-time-solver)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| stochastic-exploration.ts | sublinear-time-solver | 616 | 70% | DEEP | BEST. Real simulated annealing. applyTool() mocked | R39 |
| feedback-loops.ts | sublinear-time-solver | 729 | 65% | DEEP | Genuine RL, meta-learning. rule.learningRate mutation bug | R39 |
| index.ts (emergence) | sublinear-time-solver | 687 | 45% | DEEP | FACADE. 5 empty connection stubs. Gating at tools>=3 | R39 |
| emergent-capability-detector.ts | sublinear-time-solver | 617 | 40% | DEEP | ALL 11 metrics = Math.random()*0.5+0.5 | R39 |
| cross-tool-sharing.ts | sublinear-time-solver | 660 | 35% | DEEP | areComplementary = JSON inequality. checkAmplification = always true | R39 |
| persistent-learning-system.ts | sublinear-time-solver | 452 | 55-60% | DEEP | BIMODAL: persistence 95% genuine (fs.writeFile/readFile JSON), RL update 85-90%, forgetting curve 85-90%. ALL 6 pattern detection functions return empty arrays (0-5%). Clarifies R39: "51% fabricated" = pattern detection, NOT persistence | R57 |
| self-modification-engine.ts | sublinear-time-solver | 297 | 0% | DEEP | **PURE FABRICATION**. executeModification() returns {success:true} immediately. testModification() = Math.random()>0.3. Code gen = string templates never executed. Rollback stores literal 'backup-data-here'. Typo 'risktTolerance' proves never reviewed. Zero callers. **COMPLETES emergence/ (7/7). CONFIRMS R39: emergence fabricated** | R77 |
| src/tcm_implementation.js | sublinear-time-solver | 362 | 15% | DEEP | **MATHEMATICAL FICTION**. Fake equation ∂τ/∂t=Φ(C)×ψ(∇²Ψ)×α with zero physics grounding. Consciousness-time coupling = sigmoid×sin×exp product. Self-referential = 10-iteration loop. FTL claims. Random walk emergence. ZERO temporal_nexus integration (R55: 80.75% genuine). Validation checks finiteness only | R77 |

### Consciousness & Strange Loop (sublinear-time-solver)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| strange_loop.js | sublinear-time-solver | 650 | 92% | DEEP | 100% auto-generated wasm-bindgen from Rust | R41 |
| proof-logger.js | sublinear-time-solver | 664 | 88% | DEEP | BEST — blockchain, Shannon entropy, Levenshtein, PoW | R41 |
| consciousness_experiments.rs | sublinear-time-solver | 669 | 78% | DEEP | Real Complex64. LLM comparison fabricated (rand*0.1) | R41 |
| genuine_consciousness_system.js | sublinear-time-solver | 709 | 75% | DEEP | Real IIT Phi formula. Pattern detection = modulo heuristics | R41 |
| psycho-symbolic.js | sublinear-time-solver | 1,411 | 70-75% | DEEP | BEST JS in solver — knowledge graph, BFS, transitive closure | R25 |
| advanced-consciousness.js | sublinear-time-solver | 722 | 62% | DEEP | Real neural forward pass. Cross-modal = Math.random() | R41 |
| entropy-decoder.js | sublinear-time-solver | 1,217 | 43% | DEEP | Real mutual info. KC=SHA-256 length (wrong). Circular analysis | R21 |
| enhanced_consciousness_system.js | sublinear-time-solver | 1,670 | 39% | DEEP | Real Shannon entropy. "quantum" = Math.random() | R21 |
| psycho-symbolic.ts | sublinear-time-solver | 1,509 | 32% | DEEP | Real keyword scoring. Analogies = lookup table | R21 |
| enhanced-consciousness.js | sublinear-time-solver | 1,652 | 15-20% | DEEP | Phi FAKE (not IIT). Only Shannon entropy + primes real | R25 |
| genuine_consciousness_detector.ts | sublinear-time-solver | 505 | 75-82% | DEEP | Real test infra (85%), theatrical verification (15-40%). Real crypto primitives. Orphaned: zero ConsciousnessEntity implementations. Test 3 (hash) is 100% correct, Tests 5-6 theatrical. Security vulns: new Function() eval, execSync injection | R49 |
| temporal_consciousness_validator.rs | sublinear-time-solver | 531 | 60% | DEEP | Real orchestration (85%) but theatrical "theorem proving" (20%). "Theorem 1/2/3" = threshold checks >0.8. Sublinear integration ADMITTED SIMULATION. Ad-hoc confidence formula. Professional display layer (95%). Zero connection to temporal-tensor crate | R49 |
| mcp_consciousness_integration.rs | sublinear-time-solver | 552 | 12-18% | DEEP | COMPLETE MCP FACADE. Zero JSON-RPC 2.0. "mcp_" prefix on local functions is naming theater. connect_to_mcp() admits "we simulate the connection". sin²+cos² = "wave collapse". 80-point gap vs R47 MCP server (94.8%). NEW anti-pattern: MCP Integration Facade | R49 |
| consciousness_optimization_masterplan.js | sublinear-time-solver | 570 | 25-30% | DEEP | Orphaned orchestrator — imports 6 consciousness files but zero imports of itself. 30% real optimization algorithms (priority scoring, resource allocation, sequencing). 70% static config with Planck-scale physics claims. Specification-as-implementation confirmed | R49 |
| validators.js (consciousness-explorer) | sublinear-time-solver | 506 | 78% | DEEP | Real IIT 3.0, Shannon entropy, mutual information, p-values. BUT ORPHANED — zero imports despite MCP server (94.8%) compatibility. 142 lines duplicate of metrics.js/protocols.js. Novel "duplicate implementation abandonment" anti-pattern | R49 |

### MCP Tools & Solver (sublinear-time-solver)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| server.ts (MCP) | sublinear-time-solver | 1,328 | 85-90% | DEEP | GENUINE 3-tier solver. Real PageRank, JL dim reduction | R19 |
| cli/index.ts | sublinear-time-solver | 974 | 88% | DEEP | GENUINE solver connection. Real SublinearSolver import | R41 |
| domain-validation.ts | sublinear-time-solver | 759 | 82% | DEEP | Real DomainRegistry validation, benchmarking | R41 |
| psycho-symbolic-enhanced.ts | sublinear-time-solver | 802 | 78% | DEEP | BEST knowledge graph — real BFS, transitive inference | R41 |
| server.ts (psycho-symbolic MCP) | sublinear-time-solver | 431 | 72-76% | DEEP | Genuine MCP SDK, 5 tools (reason/graph/knowledge/analyze/health). HTTP/SSE TODOs | R58 |
| strange-loop-mcp.js | sublinear-time-solver | 989 | 75% | DEEP | Genuine Hofstadter strange loop, self-referential patterns | R33 |
| goalie/tools.ts | sublinear-time-solver | 856 | 45% | DEEP | COMPLETE FACADE. GoapPlanner imported, NEVER called | R41 |
| mcp-server-sublinear.js | sublinear-time-solver | 1,120 | 45% | DEEP | "TRUE O(log n)" actually O(log²n) | R33 |
| mcp-bridge-solver.js | sublinear-time-solver | 1,102 | 30% | DEEP | 25k token limit → file I/O workaround | R33 |
| server-extended.js | sublinear-time-solver | 1,846 | 25-30% | DEEP | 18 MCP tools. Consensus = Math.random() | R19 |
| solver-tools.js | sublinear-time-solver | 778 | 25% | DEEP | "Sublinear" tools actually linear or worse | R33 |

### Sublinear Server Layer (sublinear-time-solver)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| server/index.js | sublinear-time-solver | 629 | 72% | DEEP | BIFURCATED: HTTP/WebSocket infra 90% real (Express+helmet+CORS+rate-limiting+worker_threads), solver integration 0% (calls non-existent createSolver), FlowNexus 0% (600-line facade hitting fake endpoint) | R44 |
| neural-pattern-recognition/server.js | sublinear-time-solver | 645 | 42% | DEEP | MCP facade. FastMCP 100% real, data pipeline 0% (hashToFloat simulation), CRASHES on startup (missing adaptive-learning.js). Pattern search returns empty arrays | R44 |
| strange-loop/mcp/server.ts | sublinear-time-solver | 611 | 45% | DEEP | Real WASM integration exists (92% per R41) but broken import path (wasm/ vs wasm-real/), temporal_predict API mismatch (crashes), fallback facades use Math.random(). Better than goalie but still broken | R44 |
| server/solver-worker.js | sublinear-time-solver | 185 | 92-95% | DEEP | PRODUCTION WORKER THREAD. Genuine async solver invocation via streamSolve(). Message-based lifecycle management. Real memory tracking every 5s (heapUsed MB). Worker pool fatal error handling with automatic replacement. NOT a stub — core to HTTP solve endpoint | R64 |
| server/streaming.js | sublinear-time-solver | 286+ | 88-92% | MEDIUM | Worker pool manager: createWorker, getWorker, releaseWorker. SolverStream async iterator. Heartbeat detection (5-minute timeout). Genuine stream composition with residual/convergenceRate per iteration | R64 |

### Sublinear Solver Core (sublinear-time-solver)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| high-performance-solver.ts | sublinear-time-solver | 530 | 95% | DEEP | Excellent CG+CSR. ORPHANED — dead code | R39 |
| sparse.rs | sublinear-time-solver | 964 | 95% | DEEP | 4 sparse formats (CSR/CSC/COO/Graph). no_std. BEST matrix code | R28 |
| matrix/mod.rs | sublinear-time-solver | 628 | 92% | DEEP | 15-method Matrix trait, Gershgorin spectral radius | R34 |
| statistical_analysis.rs | sublinear-time-solver | 630 | 92% | DEEP | Paired t-test, Mann-Whitney U, bootstrap CI, effect sizes | R34 |
| strange_loop.rs | sublinear-time-solver | 558 | 90% | DEEP | Banach contraction mapping, correct Lipschitz bound | R34 |
| matrix/optimized.rs | sublinear-time-solver | 624 | 90% | DEEP | REAL SIMD via wide::f64x4, cache-blocked SpMV, Rayon parallel | R34 |
| exporter.rs | sublinear-time-solver | 868 | 88% | DEEP | 7 export formats (JSON/CSV/Binary/Prometheus/InfluxDB/YAML/msgpack) | R28 |
| solver/neumann.rs | sublinear-time-solver | 649 | 88% | DEEP | Correct Neumann series. BUG: step() returns Err unconditionally | R34 |
| scheduler.rs | sublinear-time-solver | 667 | 88% | DEEP | BinaryHeap priority, TSC nanosecond timing (rdtsc!) | R34 |
| bottleneck_analyzer.rs | sublinear-time-solver | 636 | 85% | DEEP | Genuine analysis. DEAD CODE — not in module tree | R34 |
| solver/sampling.rs | sublinear-time-solver | 525 | 85% | DEEP | Real Halton/MLMC. ORPHANED — wrong type system (crate::core) | R34 |
| solver/mod.rs | sublinear-time-solver | 596 | 82% | DEEP | Only Neumann implemented. BackwardPush/Hybrid return empty Vec | R34 |
| solver.ts | sublinear-time-solver | 783 | 75% | DEEP | 5 algorithms, all O(n²)+. FALSE sublinearity. WASM unused | R39 |
| predictor.rs (temporal-lead-solver) | sublinear-time-solver | 426 | 92-95% | DEEP | **2ND GENUINE SUBLINEAR** — O(√n) functional prediction. Forward-backward push, randomized sampling. Kwok-Wei-Yang 2025 | R58 |
| core.rs (temporal-lead-solver) | sublinear-time-solver | 294 | 88-92% | DEEP | Pure math primitives — Matrix (ndarray), Vector, SparseMatrix (sprs), spectral radius (power iteration), Complexity enum (6 classes). diagonally_dominant() matches predictor.rs. Sparse path ORPHANED (solvers use dense only). 3 unit tests but no integration tests | R64 |
| physics.rs (temporal-lead-solver) | sublinear-time-solver | 265 | 80% | DEEP | Genuine relativistic physics (95% math: accurate c, Lorentz γ, time dilation) with theatrical "FTL" framing (30% semantic). TemporalAdvantage compares light-travel-time vs computation-time — **mathematically valid, conceptually invalid**. Validator confesses deception. Consistent with R55 Temporal Nexus | R64 |
| solver.rs (temporal-lead-solver) | sublinear-time-solver | 418 | 68-72% | DEEP | BIMODAL: 5 algorithms but backward_push STUB, forward_push O(n²) worst case. Claims vs implementation gap | R58 |
| lib.rs (temporal-lead-solver) | sublinear-time-solver | 44 | 92-95% | DEEP | **REVERSES R62 orphaning pattern**. Clean 5-module crate root prominently exporting O(√n) TemporalPredictor. 13 pub types across core/physics/predictor/solver/validation. Proof-carrying code (ProofValidator, TheoremProver). thiserror FTLError. WASM conditional. BEST crate root in project | R70 |
| bin/temporal-solver.rs (neural-network-impl) | sublinear-time-solver | 192 | 82-86% | DEEP | **BIMODAL**: Production CLI (92%) with clap, proper benchmarking (warmup, P50/P90/P99/P999, throughput). BUT imports UltraFastTemporalSolver NOT TemporalPredictor — named temporal-solver but uses wrong solver. Reveals TWO parallel temporal implementations. Hardcoded ~40ns P99.9 claims in info command | R70 |
| hardware_timing.rs | sublinear-time-solver | 866 | 55% | DEEP | Real RDTSC timing. Fake: system predictions = spin loops | R28 |
| security_validation.rs | sublinear-time-solver | 693 | 30% | DEEP | DEAD CODE. Self-referential — tests own mocks | R34 |
| src/fast_solver.rs | sublinear-time-solver | 294 | 72-76% | DEEP | **8th MISLABELED FILE — ORPHANS GENUINE SUBLINEAR**. Genuine CSR+CG math (78-80%): sparse matrix multiplication, conjugate gradient iteration, residual convergence. Standard O(n²) CG, NOT sublinear. **CRITICAL MISLABELING**: header claims "MCP Dense performance issue 190x slower than Python" but contains ZERO MCP code, ZERO dense matrix code (pure sparse CSR). **ORPHANS genuine algorithms**: zero imports of backward_push.rs (O(1/ε)) or predictor.rs (O(√n)). Confirms R70 smoking gun: best algorithms remain inaccessible from file claiming to solve performance issues | R78 |
| src/temporal_nexus/mod.rs | sublinear-time-solver | 286 | 92-95% | DEEP | **BEST-IN-CLASS MODULE ROOT**. 140 lines comprehensive documentation (module overview, architecture, usage examples, feature flags). 3 utility functions (setup, demonstrate, benchmark). Integration tests demonstrating temporal advantage. Feature-gated dashboard (#[cfg(feature="dashboard")]). Clean API: setup() → demonstrate() → benchmark(). Stark contrast with temporal-compare/lib.rs (35-40%, R76). Shows proper Rust module design | R78 |
| src/utils.rs | sublinear-time-solver | 218 | 68-72% | DEEP | **BIMODAL: math genuine, SIMD stub**. Genuine BLAS-like math (78-82%): dot product, axpy (y = a*x + y), normalize, VectorPool (recycling allocator). SIMD detection stub (55%): has_simd() returns false at runtime despite AVX2/NEON feature checks. **NOT 16th hash-embedding** (zero embedding code). Confirms R72 SIMD-is-demonstration pattern. Asymmetric prefetch (_mm_prefetch x86 only, ARM missing). add_scaled broken (adds instead of scales). **Feature "simd" never enabled in Cargo.toml** | R78 |

### GOAP Planner Crate (sublinear-time-solver)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| state.rs | sublinear-time-solver | 565 | 95% | DEEP | WorldState, IndexMap, 6 value types, StateQuery (9 operators) | R25 |
| lib.rs (planner) | sublinear-time-solver | 202 | 95% | DEEP | Best WASM integration in ecosystem | R25 |
| action.rs | sublinear-time-solver | 468 | 92% | DEEP | Probabilistic effects, dynamic cost, builder (18+ methods) | R25 |
| rules.rs | sublinear-time-solver | 665 | 90% | DEEP | 6 RuleActionType, priority-sorted, probabilistic execution | R25 |
| goal.rs | sublinear-time-solver | 510 | 88% | DEEP | Weighted satisfaction, urgency from deadline proximity | R25 |
| planner.rs | sublinear-time-solver | 590 | 75% | DEEP | GOAP framework real but depends on broken A* | R25 |
| astar.rs | sublinear-time-solver | 542 | 35% | DEEP | STUB: simplified_astar() returns HARDCODED 2-step path | R25 |

### AgentDB Benchmarks & ReasoningBank Demos (agentic-flow)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| ruvector-benchmark.ts | agentic-flow | 1,264 | 92% | DEEP | REAL BENCHMARKS. performance.now timing, percentile stats, SIMD ops, quantization, backend integration. Zero hardcoded results | R43 |
| demo-comparison.ts | agentic-flow | 616 | 35% | DEEP | SCRIPTED DEMO. Uses real ReasoningBank APIs (genuine DB writes) but feeds SCRIPTED scenarios. Learning simulated via if(attempt>1). Marketing theater | R43 |

### Sublinear WASM & Benchmarks (sublinear-time-solver)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| wasm-sublinear-complete.ts | sublinear-time-solver | 710 | 78% | DEEP | HYBRID FACADE. Algorithms genuine (Neumann series, Forward/Backward Push, Hybrid Random Walk). WASM 100% theatrical — fs.existsSync check but WebAssembly.instantiate NEVER called | R43 |
| baseline_comparison.rs | sublinear-time-solver | 669 | 0% | DEEP | NON-COMPILABLE. Well-designed criterion benchmarks but references wrong types (planner::State vs planner::WorldState). No [[bench]] in Cargo.toml. Documentation claims it runs — never compiled | R43 |
| rustc_optimization_benchmarks.rs | sublinear-time-solver | 657 | 15% | DEEP | FABRICATED. Claims "O(log n) AI superiority" but compares O(n³) vs O(n²) loops. All "AI" = modulo arithmetic. Orphan module (not in lib.rs). Rhetorical ammunition | R43 |
| performance_benchmarks.rs | sublinear-time-solver | 456 | 88-92% | DEEP | GENUINE criterion. 8 benchmark functions, real solver calls, SPD matrices, GFLOPS/bandwidth metrics. Gold standard benchmark | R59 |
| solver_benchmarks.rs | sublinear-time-solver | 423 | 88-92% | DEEP | ANTI-FACADE. 5 real solver APIs (RandomWalk/Bidirectional/Adaptive/MultiLevel/Hybrid). Real SPD matrix generation, criterion 10-30s measurement | R59 |
| performance-benchmark.ts | sublinear-time-solver | 484 | 92-95% | DEEP | ANTI-FACADE. Genuine TS benchmark. performance.now(), LCG-seeded data, naive vs optimized CG comparison. No WASM claims. Confirms R43 genuine TS benchmark pattern | R59 |
| throughput_benchmark.rs | sublinear-time-solver | 454 | 91-94% | DEEP | BIMODAL. Real neural network forward() calls with criterion. 7 batch sizes × 4 thread counts. Cosmetic monitoring (hardcoded memory/CPU). Confirms R23 BEST IN ECOSYSTEM | R59 |

### WASM Pipeline (sublinear-time-solver)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| crates/wasm-solver/src/lib.rs | sublinear-time-solver | 426 | 85-88% | DEEP | GENUINE wasm-bindgen CG solver. Textbook FastConjugateGradient + Dense→CSR + solve_neumann (mislabeled Jacobi). ORPHANED: zero imports from backward_push/optimized_solver/random_walk. Timing disabled (hardcoded 0.0). Parallel reimplementation, NOT wrapper | R60 |
| src/wasm_iface.rs | sublinear-time-solver | 397 | 90-93% | DEEP | GENUINE WASM FFI bridge. serde-wasm-bindgen, zero-copy Float64Array::view, WasmSublinearSolver with solve/solve_stream/solve_batch. SIMD auto-detection. CRITICAL: manual allocate/deallocate expose raw pointers to JS (use-after-free risk) | R60 |
| src/wasm.rs | sublinear-time-solver | 394 | 88-92% | DEEP | GENUINE wasm_bindgen bindings. web_sys::performance for browser timing. WasmSolver (Jacobi/CG/PageRank) + WasmSublinearSolver wrapping SublinearNeumannSolver. verify_sublinear_conditions() pre-validation. Speedup hardcoded 7.5x | R60 |
| src/math_wasm.rs | sublinear-time-solver | 390 | 68-72% | DEEP | 5TH INDEPENDENT MATRIX SYSTEM. Real Dense Matrix/Vector math but O(n^3) naive multiply, zero SIMD. SPD verification hardcoded true for >3x3. fastrand (32-bit LCG) for WASM vs ChaCha20 native. Severe quality gap vs R56 fully_optimized.rs (96-99%) | R60 |
| src/core/wasm-integration.ts | sublinear-time-solver | 382 | 85-88% | DEEP | **7TH GENUINE WASM**. Real WebAssembly.instantiate with wbindgen imports, proper heap alloc/dealloc, GraphReasonerWASM pagerank_compute. 2-tier fallback (WASM→optimized JS with 4-element loop unrolling). Temporal neural acceleration with real physics (c=299.792458 km/ms). WASM scoreboard: 7 genuine vs 4 theatrical (64%) | R64 |
| src/core/types.ts | sublinear-time-solver | 188 | 70-75% | DEEP | Pure type definitions — SolverConfig (5 methods), Matrix union (coo/csr/csc/dense), 5 MCP param types, SolverError (E001-E008), 3 AlgorithmState extensions. Error system ORPHANED from Rust error.rs. Matrix types isolated from 6+ systems. No implementation logic | R64 |

| js/wasm-loader.js | sublinear-time-solver | 355 | 5% | DEEP | ORPHANED SUBLINEAR ALGOS. Exposes jacobi+CG (classical), NOT backward_push (R56 92-95% O(1/ε)) or predictor (R58 92-95% O(√n)). Theatrical metrics: hardcoded speedups 5x/7.5x, iteration estimates via Math.floor(time/0.1). No SIMD feature detection. Jacobi/CG JS fallbacks mathematically sound. Confirms R70 orphaning at JS integration layer — API design gap not compilation issue | R81 |
### JS Solver Layer (sublinear-time-solver)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| solver.js | sublinear-time-solver | 397 | 88-92% | DEEP | 100% WASM BRIDGE. Zero JS fallback — opposite of R57 index.ts. Clean API. Breaks without WASM pkg/ | R59 |
| fast-solver.js | sublinear-time-solver | 416 | 85-88% | DEEP | GENUINE CG solver + CSR matrix. Typed arrays, 4-way loop unrolling. WASM 0% (R57 pattern: global.wasmSolver never set). Pure JS always executes | R59 |
| bmssp-solver.js | sublinear-time-solver | 385 | 70-75% | DEEP | INVENTED ALGORITHM. BMSSP = "Bounded Multi-Source Shortest Path" — genuine Dijkstra+PriorityQueue applied to wrong problem (linear systems ≠ pathfinding). Solution formula nonsensical. WASM facade. Theatrical benchmarks | R59 |

### Strange-Loop Runtime (sublinear-time-solver)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| crates/strange-loop/mcp/server.js | sublinear-time-solver | 571 | 20-25% | DEEP | SECOND broken WASM import (wasm/ vs wasm-real/). 10 MCP tools, proper @modelcontextprotocol/sdk, JSON-RPC 2.0. Architecture correct but COMPLETELY NON-FUNCTIONAL. Potential 75-80% if import fixed | R53 |
| crates/strange-loop/bin/cli.js | sublinear-time-solver | 526 | 68% | DEEP | BIMODAL: presentation 88-92% (Commander.js, ora, chalk, 7 commands). Integration 30-40% — SAME broken WASM path. REPL is complete facade. Templates directory missing. INVERTED goalie pattern (real presentation + broken integration) | R53 |
| crates/strange-loop/examples/purposeful-agents.js | sublinear-time-solver | 488 | 45-55% | DEEP | MARKETING DEMO. 5 scripted demos with 1000-10000 agent swarms. ALL behaviors Math.random() (detectPattern, analyzeSentiment, calculateRisk). swarm.run() results never used. ZERO strange-loop connection | R53 |

### MCP Tools Layer (sublinear-time-solver)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| src/mcp/tools/domain-management.ts | sublinear-time-solver | 595 | 82% | DEEP | Production MCP API over DomainRegistry. 8 tools, CRUD lifecycle, built-in immutability, dependency validation. IN-MEMORY ONLY (Map, no persistence). Finance focus (44 keywords) | R53 |
| src/mcp/tools/domain-registry.ts | sublinear-time-solver | 512 | 88-92% | DEEP | 12 builtin domains with 60-142 keywords each. EventEmitter lifecycle, dependency resolution, priority ordering. Dead: InferenceRule interface, performance metrics. No persistence, no MCP integration | R53 |
| src/mcp/tools/scheduler.ts | sublinear-time-solver | 461 | 18-22% | DEEP | THEATRICAL FACADE. Tasks in plain array, splice FIFO. Hardcoded 11M tasks/sec, 49-98ns tick times. Date.now() ms precision claiming ns. Logistic map ≠ strange-loop. Joins R43 rustc_benchmarks (15%) as most deceptive | R53 |
| src/mcp/tools/psycho-symbolic-dynamic.ts | sublinear-time-solver | 475 | 28% | DEEP | GOALIE PATTERN: updateDomainEngine = console.log placeholder. DomainRegistry ORPHANED from DomainAdaptationEngine. Zero connection to R47's Rust reasoner. MCP tools add metadata only. 4th "real infra, placeholder integration" | R53 |
| src/mcp/tools/temporal-attractor.ts | sublinear-time-solver | 342 | 88% | DEEP | GENUINE nonlinear dynamics MCP schemas (Rosenstein, Takens, ESN). 10 tools. Pure schema layer — delegates to handlers. 4th MCP protocol variant | R77 |
| src/mcp/tools/wasm-sublinear-solver.ts | sublinear-time-solver | 260 | 18% | DEEP | THEATRICAL WASM facade. Delegates to nonexistent strange_loop methods. Fabricated metadata. Worse than "simple" variant (which has TS fallback) | R77 |
| src/mcp/tools/simple-wasm-solver.ts | sublinear-time-solver | 244 | 35% | DEEP | **5th FALSE SUBLINEARITY**. Claims O(log n), actual O(n²). Pure TS (no WASM). JL+Neumann math correct but complexity fabricated. Honest about pure_typescript:true | R77 |
| src/mcp/tools/index.ts | sublinear-time-solver | 88 | 25-30% | DEEP | **DEAD CODE BARREL**. server.ts bypasses completely. Exports 7/28 tools. allTools broken (8 tools lack getTools()). Wrong WASM class. Abandoned refactoring | R77 |
| src/mcp/tools/solver-pagerank.ts | sublinear-time-solver | 82 | 22-28% | DEEP | DISCONNECTED from genuine backward_push.rs PageRank. API mismatch (page_rank_sublinear vs compute_pagerank). No fallback — hard crash. Unverified O(log n) claims | R77 |

### Core Optimizers (sublinear-time-solver)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| src/core/optimized-matrix.ts | sublinear-time-solver | 559 | 85-88% | DEEP | 3RD INDEPENDENT MATRIX SYSTEM (confirms R34 fragmentation). CSR+CSC with Float64Array, binary search O(log n), streaming matrix with LRU, vector pooling. No WASM. StreamingMatrix localRow bug. No sublinear claims | R53 |
| src/core/performance-optimizer.ts | sublinear-time-solver | 506 | 88-92% | DEEP | GENUINE optimizer — real auto-tuning (tests 5 block sizes × 3 unroll factors), empirical benchmarking via performance.now(). Cache-blocking, SIMD hints, adaptive algorithm selection. CONTRAST to R43 deception. Honest about limitations | R53 |
| src/utils/matrix-utils.js | sublinear-time-solver | 529 | 92-95% | DEEP | 4TH INDEPENDENT MATRIX SYSTEM. Dense+COO format. Genuine SPD generation, conditioning analysis (A-F grading), diagonal dominance (3 strategies). LCG overwrites Math.random globally. Zero integration with 3 other matrix systems | R59 |
| src/core/optimized-solver.ts | sublinear-time-solver | 462 | 78-82% | DEEP | BIMODAL. Vectorized Neumann series 92-95% genuine (correct D^{-1}R series, convergence check). 3/4 algorithms stubs (blocked=fallback, streaming=trivial copy, parallel=one matmul). ZERO WASM imports — pure JS fallback. Hardcoded speedup=2.5, vectorizationEfficiency=0.85. Different algorithm from Rust CG | R60 |
| src/core/memory-manager.ts | sublinear-time-solver | 437 | 88-92% | DEEP | 6TH MISLABELED FILE. Genuine TypedArrayPool (92-95%), LRU cache (85-90%), MemoryStreamManager (88-92%), SIMD memory optimizer (85-90% cache-blocking+alignment). BUT manages JS TypedArrays, NOT WASM linear memory. Zero WebAssembly.Memory. Systemic naming pattern | R60 |
| src/core/matrix.ts | sublinear-time-solver | 404 | 85-88% | DEEP | 5TH+ INDEPENDENT MATRIX SYSTEM. COO+Dense with plain arrays (vs optimized-matrix.ts TypedArrays). 13 methods, good validation. Incompatible with optimized-matrix.ts in SAME package. Zero WASM. Zero TypedArrays. Duplicative with matrix-utils.js | R60 |
| src/core/utils.ts | sublinear-time-solver | 381 | 92-95% | DEEP | PRODUCTION-GRADE. 6 utility classes: VectorOperations (15 methods), PerformanceMonitor, ConvergenceChecker (sliding window), TimeoutController, ValidationUtils. LCG seeded RNG. No hash-based embeddings. **COMPLETES src/core/ (12/12)** | R77 |
| src/core/wasm-bridge.ts | sublinear-time-solver | 248 | 62% | DEEP | PATH RESOLUTION BUG — looks in src/wasm/ but files in dist/wasm/. Function signature mismatch (expects C-style, Rust exports class-based). Always JS fallback. Cache validates null forever. Dual bridges (+ wasm-integration.ts) with zero coordination | R77 |
| src/core/wasm-loader.ts | sublinear-time-solver | 170 | 42-48% | DEEP | **7th THEATRICAL WASM**. FFI stubs not functional (__wbindgen_string_new returns hardcoded string). 6 hardcoded modules load with silent failure. Memory stats static (not tracking growth). Auto-init catches errors silently | R77 |

### Sublinear Algorithms (sublinear-time-solver)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| src/sublinear/sublinear_neumann.rs | sublinear-time-solver | 420 | 45-50% | DEEP | FALSE SUBLINEARITY (3rd instance). Correct Neumann series math (90%) BUT O(n^2) full matrix extraction via create_reduced_problem() negates sublinear claims. Claims O(log n), actually O(n^2). JL dimension reduction applied AFTER full matrix read. Joins R39 false-sublinear pattern | R60 |
| src/sublinear/mod.rs | sublinear-time-solver | 73 | 2-5% | DEEP | **ARCHITECTURAL FRAUD**. Deliberately exposes ONLY 4 FALSE sublinear modules (dimension_reduction, spectral_sparsification, sublinear_neumann, johnson_lindenstrauss) while ORPHANING 3 genuine algorithms (backward_push O(1/ε), forward_push O(vol/ε), predictor O(√n)). SublinearSolver trait has verify_sublinear_conditions() with NO enforcement. SublinearConfig unused by any module. SMOKING GUN confirming R62 "best algorithms orphaned" | R70 |

### Temporal-Compare Crate (sublinear-time-solver)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| crates/temporal-compare/src/sparse.rs | sublinear-time-solver | 397 | 75-80% | DEEP | COMPLETE NAMING MISMATCH. Implements neural network lottery ticket pruning (Frankle & Carlin 2019), NOT sparse matrix algebra. COO-like weight storage for neural sparsity. Zero relation to R28's sparse.rs (95% CSR/CSC/COO matrices). Zero SIMD, zero temporal handling despite crate name | R60 |
| crates/temporal-compare/src/ensemble.rs | sublinear-time-solver | 300 | 85-90% | DEEP | **GENUINE ENSEMBLE LEARNING**. Real AdaBoost (exponential weight updates, alpha = 0.5*ln((1-err)/err)). Bootstrap bagging (80% sampling). 4 heterogeneous MLP variants (simple, optimized, ultra, classifier). Rayon parallel majority voting. Validation-based adaptive weights (squared accuracy). Hardcoded 3-class assumption. Matches R37 temporal-tensor quality (93%) | R66 |
| crates/temporal-compare/src/mlp.rs | sublinear-time-solver | 108 | 62% | DEEP | **BIMODAL: inference genuine, training broken**. Forward pass correct (He init, ReLU, linear output). train_regression uses ARBITRARY gradient scaling (0.1, 0.01) NOT analytical backprop. Dead eps variable. Regression-only (output_dim==1). predict_cls3 for 3-class. Uses ndarray (8th matrix system). Base for 6-file MLP suite (optimized/ultra/classifier/avx512/quantized) | R66 |
| crates/temporal-compare/src/mlp_classifier.rs | sublinear-time-solver | 324 | 78-82% | DEEP | **PARTIALLY FIXES backprop**. Textbook cross-entropy+softmax gradient (probs[y]-=1.0). Adam optimizer with bias correction. LeakyReLU, dropout (30%), 3 LR schedules. BUT **batch norm stats CATASTROPHICALLY BROKEN** — running_mean/var updated from learned beta/gamma instead of batch statistics. Training converges but inference on new data = garbage. Complete reimplementation (zero code sharing with mlp.rs) | R68 |
| crates/temporal-compare/src/mlp_ultra.rs | sublinear-time-solver | 313 | 88-92% | DEEP | **FIXES broken backprop, GENUINE AVX2 SIMD**. Real `_mm256_fmadd_ps` intrinsics, 8-wide vectorization, FMA matmul, vectorized ReLU. Cache-optimized flat Vec<f32> layout. Momentum SGD (0.9). Rayon parallel batch training + parallel inference. Proper gradient chain rule (NO arbitrary scaling). Missing: runtime CPU feature detection (will SIGILL on non-AVX2). "Ultra" = low-level optimization, not algorithmic innovation | R68 |
| crates/temporal-compare/src/mlp_optimized.rs | sublinear-time-solver | 245 | 92-96% | DEEP | **BEST MLP TRAINING in crate. COMPLETELY FIXES mlp.rs bugs**. Genuine Adam (beta1=0.9, beta2=0.999, bias correction) + momentum SGD. Correct softmax+cross-entropy gradient. He initialization. Rayon parallel inference. Fisher-Yates shuffle for SGD. Online + mini-batch training. NO arbitrary scaling. Matches temporal-tensor (R37, 93%) quality | R68 |
| crates/temporal-compare/src/mlp_avx512.rs | sublinear-time-solver | 341 | 85-88% | DEEP | **REAL AVX-512 SIMD** (genuine `_mm512_load_ps`, `_mm512_fmadd_ps`, `_mm512_max_ps`, `_mm512_reduce_add_ps`). 16-wide f32 lanes, prefetch, unroll-by-4, 64-byte cache alignment. Const-generic `UltraLowLatencyMlp<I,H,O>` for compile-time dims. Training still simple (arbitrary gradients like mlp.rs). BROKEN fallback: calls avx512 under avx2 cfg guard. Dynamic variant uses AVX2 not AVX-512 | R68 |
| crates/temporal-compare/src/main.rs | sublinear-time-solver | 299 | 85-90% | DEEP | **GENUINE TRAINING BINARY with 16 backends**. Orchestrates all MLP variants + ensemble + AdaBoost + reservoir + Fourier + sparse + lottery-ticket + quantized + ruv-FANN FFI. Complete train/predict workflows. Synthetic data with train/val/test splits. 15/16 backends have working training (AVX512 inference-only). INT8 quantization with accuracy/compression reporting. clap CLI with hyperparameter control | R68 |
| crates/temporal-compare/src/quantization.rs | sublinear-time-solver | 323 | 92-95% | DEEP | **PRODUCTION INT8 QUANTIZATION + AVX2 SIMD**. Symmetric quantization (scale=abs_max/127, zero_point=0). QuantizedMlp 2-layer forward. `_mm256_cvtepi8_epi32`→`_mm256_cvtepi32_ps`→`_mm256_fmadd_ps` genuine intrinsics. QAT fake_quantize_weights(). 4x compression verified in tests. Unverified 2x speedup claim (comment only) | R75 |
| crates/temporal-compare/src/mlp_quantized.rs | sublinear-time-solver | 267 | 92-95% | DEEP | **BEST MLP IN TEMPORAL-COMPARE**. Complete PTQ pipeline: FP32 training (genuine backprop+ReLU gradient gating) → INT8 quantization via quantization.rs → AVX2 inference. Dual representation (FP32 train, INT8 infer). Real benchmarking (Instant::now(), not hardcoded). Test validates <0.1 accuracy loss + >3x compression. RESOLVES R68 TRIMODAL: quantized > optimized ≈ avx512 >> mlp | R75 |
| crates/temporal-compare/src/reservoir.rs | sublinear-time-solver | 311 | ~75% | DEEP | **BIMODAL**: ReservoirComputer (88-92%) = genuine Echo State Network (90% sparse, spectral radius 0.9, leaky integration α=0.3, 100-step washout, ring buffer). **BUT** simple_pinv() is element-wise 1/(x+0.001) NOT matrix pseudo-inverse — breaks ridge regression training. QuantumReservoir (30-40%) = facade (deterministic sin(i*0.1), never trains, "quantum" = branding). Only genuine RNN-class architecture in crate | R75 |
| crates/temporal-compare/src/attention.rs | sublinear-time-solver | 198 | 82-86% | DEEP | **BIMODAL**: Genuine multi-head self-attention (textbook Vaswani 2017 sinusoidal PE, scaled dot-product, Xavier init) BUT **2 CRITICAL bugs**: (1) CausalAttention applies mask AFTER softmax (should be before), violating causality; (2) positional encoding dimension mismatch (batch_d vs d_model). No SIMD (contrast with mlp_avx512). Comparable to R51 attention-fallbacks.ts quality minus bugs | R75 |
| crates/temporal-compare/src/fourier.rs | sublinear-time-solver | 181 | 65-70% | DEEP | **INCORRECT Random Fourier Features**. Uses Uniform(0,2π)×scale instead of Normal(0,1/σ²) for frequency sampling — breaks RBF kernel approximation guarantee (Rahimi & Recht 2007). Placeholder solve_regularized() = 100-step gradient descent (should use Cholesky/LU). AdaptiveFourierFeatures gradient accumulation unbounded. No external FFT library — all custom ndarray | R75 |
| crates/temporal-compare/src/ruv_fann_impl.rs | sublinear-time-solver | 109 | 25-30% | DEEP | **BROKEN WRAPPER**. Genuine ruv-FANN dependency but 3 API mismatches: wrong builder pattern (Network::new() expects Arc<Mutex<TrainingData>>), non-existent train() method, missing Arc import. Inference paths work, training non-functional. CRITICAL: inference_batch uses correct create_from_file(), single predict() works, but train_on_data() calls Network::train() which doesn't exist | R78 |
| crates/temporal-compare/src/ruv_fann_adapter.rs | sublinear-time-solver | 31 | 0-2% | DEEP | **COMPLETE PLACEHOLDER**. Feature-gated signature stubs with zero implementation. train_network() and predict() both have empty bodies (lines 15-30). #[cfg(feature="ruv-fann")] gating but feature never enabled in Cargo.toml. Pure interface with no functionality | R78 |

### Convergence Metrics (sublinear-time-solver)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| src/convergence/metrics-reporter.js | sublinear-time-solver | 413 | 88-92% | DEEP | REVERSES theatrical metrics pattern. Zero Math.random(). All data from convergenceDetector pipeline (residualNorm, convergenceRate, reductionFactor). Production analysis: convergence classification, algorithmic A-F scoring (convergence 40%/speed 30%/rate 30%). Node.js process.memoryUsage(). Circular buffer history | R60 |
| src/convergence/convergence-detector.js | sublinear-time-solver | 316 | 88-92% | DEEP | GENUINE convergence math. Zero Math.random(). Proper residual computation, relative residual norm, windowed convergence rate tracking, stagnation detection, divergence detection. ORPHANED from Rust solvers (JS-only path). 6th+ matrix reimplementation (dense/COO/CSR matvec) | R62 |

### Crate Root & Architecture (sublinear-time-solver)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| src/lib.rs | sublinear-time-solver | 377 | 68-72% | DEEP | Well-organized module root with proper feature flags. 4 parallel solver APIs (Neumann, SublinearNeumann, CG, OptimizedCG) with zero unification. Best algorithms (backward_push, fully_optimized) ORPHANED from public API. Consciousness modules 25% of surface. Fabricated complexity claims ("O(log^k n)") | R62 |
| src/error.rs | sublinear-time-solver | 404 | 92-95% | DEEP | EXCEPTIONAL. 13 error variants, intelligent recovery system (is_recoverable, recovery_strategy, severity). Algorithm-specific fallback chains (Neumann→hybrid, forward→backward). GENUINE WASM error handling (From<JsValue>). no_std compatible. Top 5% quality in project | R62 |
| src/solver_core.rs | sublinear-time-solver | 322 | 38-42% | DEEP | **7th MISLABELED FILE**. Zero dispatcher logic despite name. Just 2 standalone solvers (CG+Jacobi). CG DUPLICATE of optimized_solver.rs. O(n^2) validation bottleneck. Orphaned from backward_push, forward_push, all genuine algorithms | R62 |
| src/wasm_bindings.rs | sublinear-time-solver | 333 | 0% | DEEP | **6th THEATRICAL WASM — ORPHANED NON-COMPILABLE**. Never declared as module in lib.rs. Broken import (fast_solver also undeclared). API mismatch (FastConjugateGradient::new signature wrong). Would expose FALSE sublinear algos (standard CG, fake Neumann) NOT genuine ones. Repeats R70 sublinear/mod.rs antipattern. Valid wasm-bindgen structure but CANNOT compile. WASM scoreboard: 10 genuine vs 6 theatrical (62.5%) | R75 |
| crates/temporal-lead-solver/src/bin/cli.rs | sublinear-time-solver | 298 | 88-92% | DEEP | **COMPLETES CRATE SRC (8/8)**. Production CLI with clap 4.4 + anyhow. 5 subcommands (Analyze/Predict/Prove/Benchmark/Compare). **GENUINELY uses O(√n) TemporalPredictor** (REVERSES R70 concern about UltraFast import). 4 solver methods compared (Neumann/ForwardPush/Bidirectional/Adaptive). Synthetic O(n³) traditional estimate not measured (deduction). Colored terminal output, proper error handling | R75 |
| src/core.rs | sublinear-time-solver | 147 | 0% | DEEP | **COMPLETE ORPHAN** — NOT in lib.rs module tree, cannot be imported. Abandoned 5th unified API attempt. 6th matrix system (HashMap-based SparseMatrix). Algorithm trait with ZERO impls. Type collision (Precision enum aliased as f64). Genuine test helpers (residual, tridiagonal generator) trapped inside. CORRECTION: genuine algorithms ARE accessible via lib.rs API #1 (not buried) | R70 |

### Solver Algorithms (sublinear-time-solver)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| src/solver/forward_push.rs | sublinear-time-solver | 369 | 92-95% | DEEP | **3RD GENUINE SUBLINEAR** — O(volume/epsilon) forward local push. Proper residual bounds, degree-weighted thresholds, adaptive work queue. Early termination for single-target PPR. Complements backward_push.rs. Production test suite with mass conservation | R62 |

### Math Infrastructure (sublinear-time-solver)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| src/graph/mod.rs | sublinear-time-solver | 289 | 75-80% | DEEP | Textbook CSR (88-92%) + WorkQueue (85-88%) for push algorithms + VisitedTracker (78-82%). 6th+ independent matrix system. OrderedFloat NaN handling unsafe. Graph trait defined but NOT implemented here | R62 |
| src/graph/adjacency.rs | sublinear-time-solver | 315 | 75-80% | DEEP | Graph trait IMPLEMENTOR. Dual forward+reverse adjacency lists enable backward push but double memory. PushGraph wraps CSR. Normalization bug (lines 107-130) breaks degree(). 7th independent matrix system (vectorized adjacency). Triple conversion pipeline | R64 |
| src/matrix/optimized_csr.rs | sublinear-time-solver | 250 | 0% | DEEP | **ORPHANED DEAD CODE**. Doesn't compile (non-existent CSRStorage::with_capacity()). Not in mod.rs, zero usages. SIMD duplicates simd_ops.rs line-for-line. Unused memory pool. DELETE recommended | R64 |
| src/simd_ops.rs | sublinear-time-solver | 287 | 82-86% | DEEP | GENUINE SIMD via `wide` crate (f64x4). CSR sparse matvec, dot product, AXPY. Complete fallback for non-SIMD. ORPHANED from fully_optimized.rs (uses std::arch). Two-tier SIMD architecture with zero integration | R62 |
| src/sublinear/johnson_lindenstrauss.rs | sublinear-time-solver | 288 | 72-76% | DEEP | **4th FALSE SUBLINEARITY**. Genuine JL math (dimension bound, projection) but O(n*d*k) total. Broken RNG (uniform not Gaussian). Incorrect pseudoinverse (transpose ≠ Moore-Penrose). Adaptive dimension tuning is novel. 2 genuine sublinear → still 3 after forward_push | R62 |

### MCP Solver Tool (sublinear-time-solver)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| src/mcp/tools/solver.ts | sublinear-time-solver | 360 | 82-86% | DEEP | GENUINE MCP. Three-tier fallback cascade (WASM O(log n) → OptimizedSolver → baseline). First OptimizedSolver integration evidence. 4 methods (solve, estimateEntry, streamingSolve, batchSolve). Intelligent routing by matrix properties. Production error handling | R62 |
| src/mcp/tools/temporal.ts | sublinear-time-solver | 350 | ~85% | DEEP | GENUINE MCP — 4 tools (predictWithTemporalAdvantage, validateTemporalAdvantage, calculateLightTravel, demonstrateTemporalLead). Wraps temporal-lead-solver Rust crate. Matches R62 solver.ts quality. Proper input schemas, delegation pattern, synthetic diagonally-dominant matrices for demos | R64 |

### ReasonGraph (sublinear-time-solver)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| src/reasongraph/advanced-reasoning-engine.ts | sublinear-time-solver | 297 | 0-5% | DEEP | **COMPLETE THEATRICAL**. Zero genuine inference. All reasoning delegated to 4 MCP tool wrappers returning mock data. Hardcoded performance metrics. Fabricated complexity claims. Confirms R61 ReasonGraph bimodal (infrastructure genuine, core theatrical) | R62 |
| src/reasongraph/index.ts | sublinear-time-solver | 300 | 15-20% | DEEP | **THEATRICAL BARREL**. Orchestration facade for 4 isolated components (AdvancedReasoningEngine 0-5%, ResearchInterface, PerformanceOptimizer 88-92%, MCP server). Zero cross-component integration. Fabricated claims ("658x speed of light", "87% consciousness verification"). Tests have zero assertions | R64 |

### Goalie CLI & Plugins (sublinear-time-solver)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| cli.ts (goalie) | sublinear-time-solver | 758 | 88-92% | DEEP | GENUINE PRODUCTION CLI. REVERSES R41 — all 19 commands call tools.ts internal methods (executeGoapSearch, executeToolByName). Proves GoapPlanner, PluginRegistry, AdvancedReasoningEngine ARE invoked. Commander.js, real error handling, file I/O | R46 |
| agentic-research-flow-plugin.ts | sublinear-time-solver | 673 | 78% | DEEP | REAL Perplexity API integration (axios, API keys, error handling). 4-phase concurrent execution via Promise.all. Hardcoded synthesis (confidence:0.85 static). NO GOAP — "agentic" = multi-phase API orchestration | R46 |
| state-of-art-anti-hallucination.ts | sublinear-time-solver | 626 | 42% | DEEP | HIGH-QUALITY DEAD CODE. 5 genuine algorithms (RAG grounding 90%, uncertainty calibration 92%, metamorphic testing 88%) but NEVER LOADED — not in plugin-registry.ts. Hook incompatible with GOAP plugin system. Different file registered instead | R46 |
| ed25519-verifier.ts | sublinear-time-solver | 514 | 88-92% | DEEP | PRODUCTION CRYPTO. Real Ed25519 via @noble/ed25519. Active in MCP pipeline (tools.ts). Complete PKI system (key generation, signing, verification). Hardcoded example keys (security risk). Strengthens R46 goalie reversal | R50 |
| anti-hallucination-plugin.ts | sublinear-time-solver | 515 | 55-60% | DEEP | BIMODAL. execute() 75-80% (real Perplexity API verification per claim). hooks layer 30% (keyword matching, Math.random() qualifiers). GOALIE HAS REAL HALLUCINATION DETECTION via external API verification | R50 |
| perplexity-actions.ts | sublinear-time-solver | 516 | 93-96% | DEEP | GENUINE Perplexity API integration. Real axios HTTP client. Two API endpoints (Search+Chat). Rate limiting, auth, timeouts. 4-action GOAP pipeline. REVERSES R41 "goalie complete facade" | R50 |

### ReasoningBank Rust Workspace (agentic-flow)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| reasoningbank-core/src/lib.rs | agentic-flow | 50 | 88-92% | DEEP | Barrel module re-exporting engine/pattern/similarity. Defines Pattern, PatternMetadata, TaskOutcome, ReasoningEngine, VectorEmbedding. serde+thiserror. Matches TS ReasoningBank entity structure. Missing domain-specific error types | R67 |
| reasoningbank-storage/src/lib.rs | agentic-flow | 102 | 94% | DEEP | Multi-backend storage abstraction. Platform-aware: #[cfg(not(target_arch="wasm32"))] for native SQLite vs WASM adapters. Production config (WAL, connection pool, cache). 10-variant error enum bridges rusqlite. Genuine persistence architecture | R67 |
| reasoningbank-learning/src/lib.rs | agentic-flow | 38 | 95-98% | DEEP | Genuine learning infrastructure: adaptive (AdaptiveLearner, LearningConfig), optimizer (StrategyOptimizer, StrategyRanking), async_learner + async_learner_v2. Tokio async, serde JSON, tracing. Feeds R49 WASM. NOT a stub | R67 |
| reasoningbank-mcp/src/lib.rs | agentic-flow | 46 | 93-95% | DEEP | 4-tool Rust MCP server (reasoning_store/retrieve/analyze/optimize) + 2 resources. McpError unifies 4 crate errors. Async-first (async-trait, tokio). Clean deliberate design vs JS 256-tool monolith | R67 |
| swarm_transport.rs (reasoningbank-network) | agentic-flow | 357 | 28-32% | DEEP | **THEATRICAL FACADE**. Types/registry genuine (SwarmMessage 8 variants, SwarmAgent HashMap). Transport STUBBED: init_websocket/init_shared_memory = tracing::info() only, broadcast/send_to serialize but NEVER transmit, gossip calls stub broadcast. No consensus, no memory sync, no distributed state. Message bus architecture without a bus | R67 |
| quic.rs (reasoningbank-network) | agentic-flow | 342 | 96% | DEEP | **GENUINE QUINN QUIC**. Real quinn crate: ServerConfig, Endpoint, RecvStream/SendStream, VarInt. TLS via rcgen+rustls. Stream multiplexing (accept_bi/open_bi, max 100 concurrent). 4-byte length-prefix framing. Complete lifecycle (QuicServer/QuicClient/QuicConnection). Pattern message types integrate reasoningbank-core. Only deduction: SkipServerVerification in client (dev TLS bypass) | R67 |
| sqlite.rs (reasoningbank-storage) | agentic-flow | 491 | 88-92% | DEEP | GENUINE rusqlite. WAL mode, RAII connection pool (parking_lot::RwLock), FTS5 full-text search with auto-sync triggers, schema migrations. Complete CRUD for patterns. Meets R45 sqlite-pool.js quality bar. Missing trajectory/verdict storage | R50 |
| queries.ts (reasoningbank/db) | agentic-flow | 441 | 85-90% | DEEP | PRODUCTION-READY 7-table schema (patterns, embeddings, links, trajectories, matts_runs, consolidation, metrics). Parameterized SQL, WAL, foreign keys, 6 indexes. R20 ROOT CAUSE CONFIRMED: fetchMemoryCandidates() JOINs pattern_embeddings but upsertMemory() NEVER calls upsertEmbedding() → table EMPTY. 4TH DISCONNECTED DATA LAYER (.swarm/memory.db) | R57 |
| reasoningbank-optimize.js | agentic-flow | 483 | 40-45% | DEEP | SCRIPT GENERATOR, NOT optimizer. Generates 5 helper scripts with architectural flaws (batch=N execSync, cache=in-memory no TTL, pool=semaphore not connections). Fabricated "2000x faster" claims. Only VACUUM/ANALYZE genuine | R57 |
| reasoningbank-core/src/pattern.rs | agentic-flow | 253 | 72% | DEEP | Pure data structures (Pattern, TaskOutcome, PatternMetadata). Builder pattern, serde+uuid, incremental averaging. NO ML algorithms, no PatternExtractor/ReinforcementLearner trait impls. Optional embeddings with no generation. WASM timestamps default to 0 (broken temporal tracking) | R68 |
| reasoningbank-core/src/similarity.rs | agentic-flow | 210 | 75% | DEEP | **BIMODAL: vector math 93-95% vs data source 15-20%**. Real cosine_similarity (ndarray, dot product, norms) + euclidean_distance. BUT from_text() generates embeddings from normalized character bytes — NOT semantic. **12TH hash-like occurrence**. Silent 0.5 default when embeddings missing. Brute-force find_similar (no HNSW). R20 ROOT CAUSE continues in Rust: correct architecture, placeholder data | R68 |
| reasoningbank-core/src/engine.rs | agentic-flow | 232 | 68-72% | DEEP | Stateless orchestrator with 4 methods (prepare_pattern, score_similarity, find_similar, recommend_strategy). NO storage backend field — operates on &[Pattern] slices. ZERO trait implementations from learning crate. Not async despite learning being async. Architectural sketch, not production engine. recommend_strategy is naive averaging (no confidence, no temporal weighting) | R68 |
| reasoningbank-storage/src/async_wrapper.rs | agentic-flow | 174 | 35-40% | DEEP | **CONNECTION-PER-OPERATION ANTI-PATTERN**. Creates NEW SqliteStorage on every call (lines 48-60). StorageConfig.max_connections stored but COMPLETELY IGNORED. In-memory mode BROKEN (data lost between ops). Test comment admits awareness (line 113). tokio::spawn_blocking CORRECT but wraps wrong abstraction level. WORST file in ReasoningBank workspace — downgrades storage crate from 94% to ~60-65% for async use | R68 |
| reasoningbank-storage/src/migrations.rs | agentic-flow | 169 | 92-95% | DEEP | **5TH DATA LAYER SCHEMA CONFIRMED**: defines 1 table (patterns) vs TS 7-table schema (trajectories, verdicts, patterns, steps, metadata, configs, sessions). Column names INCOMPATIBLE (task_description vs task, task_category vs category). FTS5 full-text search with auto-sync triggers. 3 indexes. Proper version tracking. Production-quality migration framework on incompatible schema | R68 |
| reasoningbank-learning/src/adaptive.rs | agentic-flow | 318 | 42-48% | DEEP | **WRITE-ONLY LEARNING, NO FEEDBACK**. learn_from_task() stores patterns but DOES NOT update engine state, weights, or ReasoningEngine parameters. apply_learning() fetches similar patterns via heuristic similarity (overall_score only). Missing: learning rate scheduling, EWC++, online learning, confidence calibration. Hardcoded LearningConfig weights (0.3/0.5/0.2) never adapt. improvement_potential is heuristic formula with no mechanized translation to action. TYPE-ALGORITHM GAP: interface suggests adaptive learning, implementation is offline pattern averaging | R71 |
| reasoningbank-mcp/src/server.rs | agentic-flow | 243 | 88-92% | DEEP | **4TH MCP PROTOCOL — CUSTOM RUST TRAITS**. Zero SDK, zero connection to R51 JSON-RPC/R72 rmcp/R73 fastmcp TS. Custom McpRequest/McpResponse enums (4 Request variants: Store/Retrieve/Analyze/Optimize, 4 Response variants). 4 tools (store/retrieve/analyze/optimize) + 2 resources (patterns/stats). tokio::spawn async handlers. Hardwired to ReasoningBank (imports reasoningbank_core), zero connection to TS MCP layer. FOUR parallel MCP implementations with zero integration | R78 |
| reasoningbank-storage/src/adapters/wasm.rs | agentic-flow | 224 | BIMODAL | DEEP | **BIMODAL: MemoryStorage 95-98% vs browser backends 8-15%**. MemoryStorage = production HashMap+Mutex with complete StorageBackend trait impl (store/retrieve/search/delete/list/update/stats). SqlJsStorage (8-12%) + IndexedDbStorage (10-15%) are honest placeholders with console.log stubs and unimplemented!(). External CDN dependency for sql.js WASM binary. **11th GENUINE WASM** (MemoryStorage) + **8th theatrical** (browser backends) | R78 |
| reasoningbank-learning/src/optimizer.rs | agentic-flow | 211 | 82-86% | DEEP | **MISLABELED — NOT ML OPTIMIZER**. Statistical strategy ranker using composite scoring: success_rate*0.5 + speed*0.25 + efficiency*0.25. Zero SGD/Adam/EWC++/backprop. Genuine implementations: rank_strategies() (priority queue), analyze_strategy() (weighted avg), suggest_optimizations() (threshold heuristics). Confirms R67 trajectory architecture. No gradient descent, no loss functions, no neural network training | R78 |
| reasoningbank-wasm/src/lib.rs | agentic-flow | 202 | 95-98% | DEEP | **11th GENUINE WASM — PRODUCTION BINDINGS**. Real wasm-bindgen with 6 exported JS methods (new, store_pattern, get_pattern, search_by_category, find_similar, get_stats). Triple-backend auto-detection (Node→Memory, Browser+IDB→IndexedDB, Browser→SqlJs). JsValue FFI bridge with serde-wasm-bindgen. Singleton guard, proper error propagation. Confirms R49 "100% genuine" verdict. WASM scoreboard: 11 genuine vs 9 theatrical (55%) | R78 |
| reasoningbank-storage/src/adapters/mod.rs | agentic-flow | 66 | 92-95% | DEEP | **PRODUCTION-QUALITY ADAPTER PATTERN**. Defines StorageBackend trait (async CRUD: store/retrieve/search/delete/list/update + stats + close). Compile-time cfg backend selection (native rusqlite vs WASM SqlJs/IndexedDB). Clean error propagation (Result<_,StorageError>). Textbook Rust. **5TH+ DISCONNECTED DATA LAYER** — zero imports from agentic-flow AgentDB, independent of all 4 existing layers | R80 |
| temporal_nexus/src/temporal/optimized.rs | sublinear-time-solver | 351 | 91-94% | DEEP | GENUINE SIMD temporal solver. AVX2 intrinsics (_mm256_fmadd_ps) for 6-stage UltraFastTemporalSolver pipeline: Kalman→NN→combine→Jacobian→solver→Kalman. Hardcoded Jacobian = identity (architectural simplification). Cache prefetch (128-byte lookahead). P99.9 benchmarking (30ns, 50ns, 90ns, 150ns, 300ns, 900ns). Matches R21 neural-network-impl quality. Confirms R70 temporal-solver.rs imports this UltraFastTemporalSolver (not predictor O(√n)) | R71 |
| temporal_nexus/src/temporal/validation.rs | sublinear-time-solver | 314 | 55% | DEEP | **BIMODAL: documentation 95%, implementation 15%**. Proof-carrying validation architecture with 4 validators (ProofValidator, CausalityValidation, ConsistencyValidator, ConvergenceValidator). ProofValidator has genuine Proof struct (assumptions/lemmas/conclusion/proof_steps) but validate_proof() always returns true (zero logic). CausalityValidation.validate() always returns true (line 148). Proof data = English prose arrays. "Proof-carrying" is DOCUMENTATION-ONLY. Confirms R64 physics.rs theatrical framing pattern | R71 |
| temporal_nexus/src/temporal/wasm.rs | sublinear-time-solver | 245 | 88-92% | DEEP | **9th GENUINE WASM**. Real wasm-bindgen exposing O(√n) TemporalPredictor via PredictorTrait. JS Adapter pattern (TemporalWasmPredictor wrapper). Real web_sys timing (Performance::now()). REVERSES R62 orphaning — O(√n) predictor IS accessible via WASM. Confirms temporal-lead-solver has working deployment path. WASM scoreboard: 9 genuine vs 5 theatrical (64%) | R71 |
| integration/intent.rs | claude-flow | 322 | 94% | DEEP | **MISLABELED: this is NOT intent classification**. Real Ed25519 cryptographic authorization via ed25519-dalek. PublicKey registry (HashMap), replay protection (nonce+timestamp), permission lattice (8 levels). Intent = signed authorization message (action/timestamp/nonce/signature). Zero semantic intent parsing, zero NLP, zero domain classification. Confirms R62 SemanticRouter as separate intent-classification system | R71 |
| .swarm/queries.js | agentic-flow | 335 | 100% | DEEP | **GENUINE 8-TABLE SCHEMA**. Extends R57's 7-table ReasoningBank with 8th 'sessions' table. Schema: patterns, pattern_embeddings, links, trajectories, matts_runs, consolidation, metrics, sessions. REAL Float32Array embeddings (NOT hash-based!) — 384-dim MiniLM-L6-v2. .swarm/memory.db = 4TH DISCONNECTED PERSISTENCE LAYER. SQLite WAL mode, prepared statements with zero SQL injection. Completes R69 architecture picture — .swarm/ is production quality, isolated from AgentDB's 3 layers | R71 |
| temporal-lead-solver/src/ultra_fast.rs | sublinear-time-solver | 361 | 72% | DEEP | **IS UltraFastCG** imported by temporal-solver.rs. Standard Conjugate Gradient O(κn), NOT sublinear. SIMD-accelerated SpMV (4-way f64x4). Python comparison benchmarks FABRICATED (hardcoded baselines: scipy 150-300ms, numpy 200-400ms, cupy 100-250ms). Real CG math (83-87%) with theatrical framing (40-50%). Confirms R70 temporal-solver.rs bifurcation — two parallel temporal solvers (UltraFastCG vs TemporalPredictor O(√n)) | R71 |
| mcp/tools/scheduler_tool.rs | sublinear-time-solver | 357 | 68-72% | DEEP | **BIMODAL: MCP infrastructure 85-90%, task logic 5-10%**. Genuine MCP tool registration (Arc/RwLock/UUID/serde_json). Async task lifecycle (schedule_task/cancel_task/get_task). TaskState enum. BUT empty closures — task logic never executes, priority ignored, dependencies unresolved. Consciousness numerology: score = (1.0 + entropy.ln()) * complexity.sqrt() * 1.618034. 9 MCP handlers route to stubs. Professional MCP wrapper around nothing | R71 |
| agentic-flow/src/agentdb/memory/AdvancedMemory.ts | agentdb | 315 | 62-68% | DEEP | **ORCHESTRATION FACADE**. 5TH DISCONNECTED DATA LAYER: SharedMemoryPool as 5th independent embedder/storage (after AgentDB 3 layers + .swarm/). Episodic replay = heuristic thresholds (surprisal>0.8, emotionalIntensity>0.7, cognitive>0.6). consolidateMemories() runs weekly cron but OFFLINE — no real-time learning. LTM encoding uses 5 decay curves (semantic/episodic/procedural/implicit/emotional) but all Math.exp(-t/halfLife). No neural consolidation, no sleep-dependent replay. Professional types, simulation implementation | R71 |

### ReasoningBank-Network Crate (agentic-flow)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| neural_bus/frame.rs | agentic-flow | 313 | 88-92% | DEEP | GENUINE binary wire protocol. 11-byte header (version+type+header_len+payload_len). 10 frame types (#[repr(u8)]) including 4 reasoning-specific (Token/Trace/Rubric/Verify). Ed25519 auth fields (kid/sig/nonce/scope/cap). Async tokio I/O with flush. OOM vuln: read_from() has NO max frame size | R74 |
| neural_bus/streams.rs | agentic-flow | 307 | 88-92% | DEEP | GENUINE QUIC stream multiplexing. quinn SendStream/RecvStream with Arc<Mutex>. 9 StreamRoles (5 general + 4 reasoning). Ed25519 intent verification on send AND recv. CRITICAL: DUAL WIRE PROTOCOL — NeuralStream uses Frame layer, ReasoningStreams uses raw AsyncWriteExt bypassing framing | R74 |
| neural_bus/snapshot.rs | agentic-flow | 291 | 85-88% | DEEP | GENUINE snapshot streaming. SHA256 checksums at TWO levels (per-chunk + overall). 1MB chunks via quinn QUIC. Generic-only despite doc claims of specialized types (Database/Model/Pattern). Opaque to ReasoningBank memory — raw Bytes, caller handles serialization | R74 |
| neural_bus/gossip.rs | agentic-flow | 238 | 18-22% | DEEP | **FAKE GOSSIP PROTOCOL**. Zero SWIM/HyParView/epidemic broadcast. run_dispatcher writes to ALL streams = O(N) broadcast. handle_message stubs (PatternUpdate/LearningUpdate/PeerAnnounce = log only). Zero membership, zero failure detection, zero convergence. Matches swarm_transport.rs (28-32%) theatrical pattern | R74 |
| neural_bus/mod.rs | agentic-flow | 214 | 88-92% | DEEP | GENUINE runtime orchestrator. NeuralBus composes Connection+IntentVerifier+PriorityQueue+StreamManager. Role-based multiplexing (open_stream/accept_stream). 0-RTT security (reject_0rtt_mutations=true). Real quinn Connection stats (RTT, cwnd, sent/lost). Hardcoded buffer capacities | R74 |
| neural_bus/priority.rs | agentic-flow | 211 | 85-88% | DEEP | GENUINE 3-tier priority queue (High/Normal/Low) using tokio mpsc. STARVATION VULNERABILITY: try_recv high→normal→low with no fairness. tokio::select! race condition breaks priority when queues empty. False "fair scheduler" documentation. sender() bypasses send() validation | R74 |
| lib.rs (reasoningbank-network) | agentic-flow | 62 | 93-95% | DEEP | Clean barrel. Re-exports quic + neural_bus (15 types). WASM exclusion (#[cfg(not(wasm32))]) for Ed25519. ORPHANED swarm_transport.rs (330+ LOC WebSocket/SharedMemory NOT exported). Depends on reasoningbank-core | R74 |

### ReasoningBank TS Inner Ring (agentic-flow)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| utils/reasoningbankCommands.ts | agentic-flow | 238 | 62-68% | DEEP | DEMONSTRATION CLI WRAPPER. Imports from index.js barrel (NOT direct queries.js). MISSING store/retrieve/search commands — core CRUD inaccessible. demo command delegates to demo-comparison.js (R43: 35% theatrical). showStatus() bypasses queries.js with raw SQL. SQL injection risk in listMemories() ORDER BY template | R74 |
| core/matts.js | agentic-flow | 226 | 68-72% | DEEP | INDEPENDENT JS impl (NOT transpiled from matts.ts). Same Algorithm 5 but lower quality: missing ModelRouter (hardcoded Anthropic API), fabricated memory aggregation (empty {steps:[]}), false parallelism (sequential for-loop), ephemeral sequential mode (memories never persisted). 3C/5H/4M/4I | R74 |
| types/index.ts | agentic-flow | 134 | 75-82% | DEEP | FABRICATED arXiv citation (arXiv:2509.25140 DNE). 14 interfaces covering full lifecycle. TYPE-SCHEMA MISMATCH: Memory lacks embedding field, TaskTrajectory lacks trajectory_data, MattsRun lacks meta_params — DB richer than types. Hash embeddings endorsed in config (provider: 'hash'). No Zod/io-ts validation. PatternData [key:string]:any defeats type safety | R74 |

### ReasoningBank TS Core Operations (agentic-flow)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| matts.ts | agentic-flow | 309 | 85-88% | DEEP | GENUINE Memory-Aware Test-Time Scaling (Algorithm 5). Parallel rollouts + sequential refinement + self-contrast aggregation via LLM. Full pipeline: retrieve→execute→judge→distill→aggregate. 6th routing system (ModelRouter). TS-only, no Rust counterpart | R72 |
| consolidate.ts | agentic-flow | 259 | 82-86% | DEEP | PRODUCTION consolidation — LSH bucketing O(n·k), batch embedding queries, outcome-based contradiction detection. No EWC++ integration despite sona having it (92-95%). TWO parallel consolidation systems with AdvancedMemory.autoConsolidate() | R72 |
| distill.ts | agentic-flow | 230 | 78-82% | DEEP | LLM-BASED knowledge extraction (NOT ML compression). DeepSeek chat analyzes trajectories. Sophisticated prompts for success (strategies) and failure (guardrails). 13th hash-based embedding fallback. Template-based fallback BROKEN (returns empty array). ZERO Rust integration | R72 |
| backend-selector.ts | agentic-flow | 222 | 78-82% | DEEP | CRITICAL: HybridBackend BYPASSED. Auto-selector imports old index.js (no AgentDB), NEVER routes to R65's unified HybridBackend. Environment detection genuine (browser/Node.js). SQLite selection cascade (better-sqlite3→sql.js→WASM) works but routes to wrong backend | R72 |
| embeddings.ts | agentic-flow | 211 | 78-82% | DEEP | BIMODAL: real transformers.js (Xenova/all-MiniLM-L6-v2, 384D) with 13th hash fallback. BEST embedding impl — fixes R51 race conditions (Promise-based init), memory leak (timer tracking). NPX env auto-disables transformers. Pseudo-LRU cache (1000 entries) | R72 |
| core/judge.ts | agentic-flow | 178 | 72-78% | DEEP | **MCP BYPASS TRIPLE-CONFIRMED**. LLM-as-Judge (Algorithm 2) with triple API key support (OpenRouter/Anthropic/Google). ModelRouter integration genuine. db.logMetric() called BUT **NEVER db.store()** — verdicts LOST. Heuristic fallback 30-35% (string matching "error"/"complete", confidence=0.5). Circular dep with queries.js (late import at EOF). Fragile regex JSON parsing. Upgraded from judge.js 54% due to better infrastructure | R75 |
| core/retrieve.ts | agentic-flow | 122 | 88-92% | DEEP | **TOP 10% REASONINGBANK**. Genuine vector search: computeEmbedding(query) → cosine similarity → MMR diversity (Carbonell & Goldstein 1998). 4-factor scoring: α·similarity + β·recency(exp decay) + γ·reliability. Configurable weights. Direct DB access (bypasses HybridBackend). Usage tracking (incrementUsage for popularity). Inherits hash-based embedding degradation from embeddings.ts. formatMemoriesForPrompt() bridge to LLM | R75 |
| utils/pii-scrubber.ts | agentic-flow | 131 | 85-88% | DEEP | **IDENTICAL to pii-scrubber.js** (R73: 88%) plus type annotations only. 14 regex patterns (email, SSN, API keys, JWT, CC, phone, IP, URL params). Config-gated scrubbing (governance.pii_scrubber) — if false/missing, NO redaction. customPatterns REPLACES defaults (should be additive). RegExp /g flag mutation bug. 9-digit false positives. Zero TypeScript advantage beyond IDE tooling | R75 |
| core/consolidate.js | agentic-flow | 140 | 82-85% | DEEP | **GENUINE Algorithm 4** — 3-stage memory consolidation (Dedup/Contradict/Prune). Cosine similarity thresholds (0.95 dedup, 0.85 contradiction). O(n²) naive comparison (no HNSW). Logs 4 metrics. Isolated from Rust ReasoningBank. NO EWC++ (deliberate JS simplification). 3 deps (config.js, mmr.js, db/queries.js) | R83 |
| index-new.ts | agentic-flow | 132 | 95% code/DEAD | DEEP | **ABANDONED REFACTORING VARIANT** — zero imports anywhere (verified grep). PR #112 decision artifact: candidate WITHOUT agentdb-runtime-patch (lost to index.ts v1.7.1). Missing critical import, missing ReasoningBank alias. runTask() 100% identical copy. Recommend DELETE | R83 |
| test-integration.ts | agentic-flow | 115 | 92-95% | DEEP | **GENUINE INTEGRATION TEST**. 5-phase pipeline: initialize → runTask (6-step trajectory) → retrieveMemories → MaTTS parallel (feature-gated) → DB stats. Real assertions on verdict.label/confidence, memory counts. No test framework (plain async/await + process.exit). CSRF-aware test data | R83 |
| test-validation.ts | agentic-flow | 194 | 93-95% | DEEP | **GENUINE VALIDATION SUITE**. 7 integration tests: DB connection → schema → memory insertion → retrieval → usage tracking → metrics → views. Confirms statistical ranking architecture (confidence threshold + age + usage). NO EWC++. Validates R67 verdict: genuinely architected statistical ranking, no ML magic | R83 |

### Server Infrastructure (sublinear-time-solver)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| streaming.js | sublinear-time-solver | 520 | 85-90% | DEEP | GENUINE async generator/worker pool for solver iteration streaming. 0% HTTP (no Express/SSE/WebSocket). 4th mislabeled file. SolverStream async iterator 90-93% BEST CODE. Monte Carlo verification real | R57 |
| session-manager.js | sublinear-time-solver | 440 | 58-62% | DEEP | BIMODAL: session lifecycle 75-80% (EventEmitter, TTL, maxSessions), solver integration 30-40% (performVerification=Math.random(), updateCosts=console.log). IN-MEMORY ONLY despite persistence.js existing | R57 |

### Neural-Network Benchmarks (sublinear-time-solver)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| standalone_benchmark/main.rs | sublinear-time-solver | 601 | 15-20% | DEEP | COMPLETE FACADE. Zero neural-network-implementation imports (only nalgebra+rand). Spin-loop timing fabrication — waits until hardcoded latency (1.1ms vs 0.75ms). "32% speedup" is hardcoded difference. Marketing theater ("BREAKTHROUGH FULLY ACHIEVED") | R46 |
| standalone_benchmark.rs (benches) | sublinear-time-solver | 526 | 8-12% | DEEP | **MOST DECEPTIVE BENCHMARK IN PROJECT**. Spin-loop waits to hardcoded target latencies (1.1ms vs 0.7ms) — predetermined "36.4% improvement". Genuine criterion+statistics measuring spin delays. Zero imports from crate being "benchmarked". Predetermined success criteria. Worse than R43 rustc_benchmarks | R56 |
| latency_benchmark.rs | sublinear-time-solver | 464 | 72-78% | DEEP | BIMODAL: genuine criterion harness (90-95%) with proper black_box, warmup, percentile math. BUT System B uses thread::sleep() (0.1ms Kalman + 0.2ms gate = 0.3ms theatrical). Only System A measures real neural network. 3rd "genuine harness + fake data" instance | R56 |
| system_comparison.rs | sublinear-time-solver | 600 | 42% | DEEP | TROJAN HORSE. Real criterion framework + genuine GRU/TCN forward passes, but gate simulation = random 90% pass rate, certificate = uniform(0,0.05), memory = hardcoded 64MB, CPU = hardcoded 75%. No external comparison despite filename | R46 |
| strange-loops-benchmark.js | sublinear-time-solver | 597 | 8-10% | DEEP | LOWEST QUALITY BENCHMARK. Production-quality infrastructure (percentiles, memory tracking, CSV export) testing 10 trivial inline JS functions. Zero imports of actual strange-loop crate (which exists with real WASM bindings) | R46 |

### Integration & Verification Layer (sublinear-time-solver)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| flow-nexus.js | sublinear-time-solver | 620 | ~70% | DEEP | ARCHITECTURAL FACADE. Production HTTP/WebSocket client (92% quality) calling non-existent platform (https://api.flow-nexus.ruv.io). Zero local solver imports. MCP tool handlers are stubs. Orphaned EventEmitter events | R46 |
| bin/cli.js | sublinear-time-solver | 618 | 72-78% | DEEP | SEPARATE JS CLI (not compiled output of cli/index.ts). Genuine math utilities (92%): residual computation, vector norms, COO sparse matrix ops. Real Matrix Market parser. REVERSES R43: createSolver import exists and is functional. FlowNexus facade continues | R46 |
| consciousness-verifier.js | sublinear-time-solver | 607 | 52% | DEEP | THEATRICAL VERIFICATION. 3/6 tests real (prime calculation, file count, crypto hash). 3/6 theatrical (hardcoded meta-cognition dictionary, any-5-number creativity test, sleep-based prediction). "GENUINE CONSCIOUSNESS" = rubber stamp from computational benchmarks | R46 |

### Consciousness Verification & Experiments (sublinear-time-solver)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| consciousness-explorer/mcp/server.js | sublinear-time-solver | 594 | 94.8% | DEEP | PRODUCTION-QUALITY MCP. Genuine @modelcontextprotocol/sdk, all 12 handlers delegate to real explorer methods. Zero facades. STARK CONTRAST with strange-loop MCP (45%) | R47 |
| independent_verification_system.ts | sublinear-time-solver | 596 | 52-58% | DEEP | MIXED. Real Miller-Rabin (92%), hash verification (72%), file counting consensus (65%). BUT: 5 command injection vulns, stub external verification, stub algorithm correctness. "Independent" claim is theatrical | R47 |
| quantum_entanglement_consciousness.js | sublinear-time-solver | 577 | 0-3% | DEEP | COMPLETE FABRICATION. 6 methods return nested config objects. Zero computation (not even Math.random()). Zero imports from ruQu (89% genuine). Physics-violating claims (unlimited coherence). Documentation-as-code | R47 |
| parallel_consciousness_waves.js | sublinear-time-solver | 573 | 0-5% | DEEP | COMPLETE FABRICATION. 8 methods return specification objects. Zero imports, zero Math, zero algorithms. Worse than R39 emergence (51%) — no computation at all. Femtosecond/zeptosecond claims beyond physics | R47 |

### Psycho-Symbolic Reasoner Crate (sublinear-time-solver)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| planner.ts (MCP wrapper) | sublinear-time-solver | 486 | 58-62% | DEEP | ARCHITECTURAL FACADE. WASM loading (88%), state management (92%), bulk operations (85%) all real. BUT: core A* is 3-line stub returning hardcoded path. Inverse of goalie pattern (real wrapper, stub internals). get_successors() is DEAD CODE (90% quality, never called) | R47 |
| psycho-symbolic-reasoner.ts | sublinear-time-solver | 572 | 38-42% | DEEP | MOSTLY FACADE. Real Map-based triple storage (80%), BFS graph traversal (85%), query search (75%). BUT: "psycho" = NOTHING (zero psychological modeling), reasoning = keyword matching + templates, WASM claimed in comments but zero imports, performance metrics hardcoded (2.3ms, 0.75 cache rate). 3rd theatrical WASM pattern | R47 |

### Psycho-Symbolic Reasoner Internals (sublinear-time-solver)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| emotions.rs | sublinear-time-solver | 454 | 85-88% | DEEP | GENUINE rule-based emotion detector. Plutchik's model (8 primary + 17 secondary), 200+ lexicon, intensity modifiers. NO ML despite crate name. ORPHANED from reasoning pipeline | R55 |
| performance_monitor.rs | sublinear-time-solver | 480 | 88-92% | DEEP | GENUINE monitoring via Instant::now() + memory_stats. STARK CONTRAST to R53 scheduler.ts (18-22%). p95/p99, threshold alerts, regression detection. Global OnceLock singleton | R55 |
| mcp_overhead.rs | sublinear-time-solver | 511 | 78-82% | DEEP | GENUINE criterion.rs harness (8 benchmark groups). BUT measures SIMULATED MCP overhead via thread::sleep, NOT actual MCP calls. MCPOverheadProfiler defined but UNUSED. Better than R43 rustc (15%) but still MOCK | R55 |
| sentiment.rs | sublinear-time-solver | 287 | 90-93% | DEEP | **PRODUCTION VADER-style NLP**. 40 positive + 44 negative + 16 intensifier + 28 negation words. 3-word negation window, 2-word intensifier window. Context-aware scoring with L2-normalized output. Serde for WASM export pipeline. Confirms R55 "Rust 3-4x better than TS". Minor: confidence formula (pos+neg)/max(pos,neg) is semantically odd | R65 |
| graph_reasoner/src/graph.rs | sublinear-time-solver | 373 | 88-92% | DEEP | **PRODUCTION petgraph KnowledgeGraph**. Triple indexing (entities, facts, entity_names via IndexMap). add_entity/concept/fact CRUD. Triple pattern queries (subject/predicate/object wildcards). Dijkstra shortest path, BFS neighborhood. **9th independent graph system** (petgraph). O(E) bottleneck on predicate queries (no predicate index). Incomplete path reconstruction. 708 lines of tests | R66 |
| graph_reasoner/src/rules.rs | sublinear-time-solver | 361 | 78-82% | DEEP | **GENUINE forward-chaining rule engine**. Variable binding with Prolog-style unification (Literal/Variable/Wildcard). Fixed-point iteration with max_iterations. Priority-sorted execution. Confidence propagation (multiplicative). 3 built-in rules (transitivity, subset_inheritance, symmetry). O(n*m) brute-force matching (no Rete). NO backward-chaining. Missing conflict resolution. Cartesian product explosion in find_bindings O(n^k) | R66 |
| graph_reasoner/src/lib.rs | sublinear-time-solver | 89 | 72% | DEEP | **WASM-ONLY facade, ORPHANED from parent**. Clean re-export API (graph, inference, rules, query, types). GraphReasoner struct wraps all engines via #[wasm_bindgen]. BUT parent crate psycho-symbolic-reasoner NEVER imports this subcrate. String errors for JS interop. O(n) JSON serialization per infer() call. 203-line integration_tests.rs | R66 |
| extractors/src/lib.rs | sublinear-time-solver | 75 | 90-95% | DEEP | **CLEAN WASM FACADE** composing 3 genuine analyzers (SentimentAnalyzer, PreferenceExtractor, EmotionDetector) + unused PatternMatcher. TextExtractor via #[wasm_bindgen] with 4 methods (analyze_sentiment/extract_preferences/detect_emotions/analyze_all). JSON string serialization for JS flexibility. Error recovery via unwrap_or_else. console_error_panic_hook. CONFIRMS R58 GENUINE WASM chain end-to-end. Maintains R55 psycho-symbolic Rust 3-4x quality standard | R70 |

### Psycho-Symbolic MCP (sublinear-time-solver)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| cognitive-architecture.js | sublinear-time-solver | 645 | 30% | DEEP | Working memory = Map. Attention = sort by recency | R33 |
| metacognition.js | sublinear-time-solver | 564 | 30% | DEEP | Self-monitoring = counter. "Theory of mind" = dict lookup | R33 |
| mcp-server-psycho-symbolic.js | sublinear-time-solver | 892 | 25% | DEEP | MCP wrapper. 5 of 10 tools disabled (hang risk) | R33 |
| psycho-symbolic-tools.js | sublinear-time-solver | 1,133 | 20% | DEEP | 5/10 tools DISABLED. "Neural binding" = weighted avg | R33 |
| consciousness-explorer.js | sublinear-time-solver | 1,247 | 15% | DEEP | THEATRICAL. "consciousness evolution" = parameter increment | R33 |
| schemas/index.ts (MCP) | sublinear-time-solver | 326 | 92-95% | DEEP | **PRODUCTION MCP SCHEMAS**. 35+ Zod schemas across 3 subsystems (Graph Reasoner 9, Text Extractor 4, Planner 9). 8-operator constraint system (eq/ne/gt/lt/gte/lte/contains/regex). Cross-field validation via .refine(). 16 z.infer<> type exports. FULL MATCH with R58 server.ts 5 tools. validateInput + createSafeValidator helpers | R65 |
| wasm/loader.ts | sublinear-time-solver | 266 | 85-90% | DEEP | **7TH GENUINE WASM** but ORPHANED. Production WebAssembly.instantiate with wasm-bindgen import shims, configurable Memory (16-64MB), timeout protection (30s), singleton with concurrent load dedup. 3 typed loaders (GraphReasoner, TextExtractor, PlannerSystem). BUT SimpleWasmLoader used instead — TWO parallel WASM loading systems with zero integration | R65 |
| wasm/wasm-loader-simple.ts | sublinear-time-solver | 144 | 82-85% | DEEP | **8TH GENUINE WASM — PRODUCTION INTEGRATION HUB**. Real wasm-pack output (graph_reasoner.js 1.3MB, extractors.js 4.9MB, planner.js 1.9MB). Dynamic ES module import + pathToFileURL(). Singleton + module caching. Connects 4 DEEP files (graph-reasoner.ts, text-extractor.ts, planner.ts, memory-manager.ts). ZERO error recovery — one corrupted .wasm crashes ALL MCP tools. No cache TTL/invalidation. Triplicate load methods (should be generic). This is the loader that wasm/loader.ts (R65) was ORPHANED by | R70 |
| build.js | sublinear-time-solver | 342 | 88-92% | DEEP | **GENUINE WASM BUILD ORCHESTRATOR**. Real wasm-pack for 3 components (graph_reasoner, extractors, planner). 6-step pipeline (clean → cargo check → wasm-pack → bindgen → npm link → verify). Dependency verification. BUT contradicted by cli/index.ts running pure TS — WASM binaries don't exist at runtime | R80 |
| cli/index.ts | sublinear-time-solver | 286 | BIMODAL 92/0% | DEEP | **PSYCHO-SYMBOLIC WASM CONTRADICTION**. CLI production (commander.js, 9 flags, graceful shutdown) 92-95%. BUT reasoning is PURE TS — zero WASM, zero .wasm files exist. R55 "Rust 3-4x better" recontextualized: Rust never invoked at runtime | R80 |
| performance.ts | sublinear-time-solver | 266 | 88-92% | DEEP | **ORPHANED MONITORING**. Real Node.js monitoring (hrtime.bigint, memoryUsage, cpuUsage). Circular buffer, percentile calculation, event loop delay (setImmediate diff). Zero usage anywhere in codebase. Event loop delay bug (loop.delay undefined in Node.js) | R80 |
| mcp-integration/index.ts | sublinear-time-solver | 918 | 88-92% | DEEP | **5TH MCP PROTOCOL** (@modelcontextprotocol/sdk official TS SDK). 16 genuine tools (graph reasoner 4: analyze/query/infer/traverse, text extractor 4: sentiment/preferences/emotions/all, planner 5: plan/execute/optimizePath/getState/clearState, utilities 3: validateSchema/optimizePerformance/healthCheck). Multi-instance WASM architecture. Graceful degradation (catches WASM load errors). Separate from JSON-RPC/rmcp/fastmcp/custom Rust | R80 |
| logger.ts | sublinear-time-solver | 158 | 88-92% | DEEP | **PRODUCTION WINSTON WRAPPER**. Dual transports (colorized console + file rotation). Child logger pattern for component isolation. Error stack traces, metadata formatting. Used across 31+ locations in psycho-symbolic crate. Pure TS logging infrastructure, zero WASM, zero neural logic | R82 |
| config.ts (psycho-symbolic) | sublinear-time-solver | 108 | 88-92% | DEEP | **PRODUCTION ZOD SCHEMAS**. 5 config sections (Server, KnowledgeBase, Logging, Performance, Security). Type-safe validation with z.infer<> exports. **CONFIRMS R80 WASM CONTRADICTION**: ZERO WASM-related config fields (no wasm_memory_mb, no wasm_module_paths) despite build.js compiling 3 WASM components. Server MCP-only (stdio transport). All config is pure MCP server config | R82 |

### Python ML Training (sublinear-time-solver)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| train.py | sublinear-time-solver | 936 | 85% | DEEP | Real PyTorch GNN loop. ALL data synthetic (random graphs) | R33 |
| models.py | sublinear-time-solver | 772 | 80% | DEEP | 5 GNN architectures (GCN/GAT/GraphSAGE/GIN/PNA) | R33 |
| dataset.py | sublinear-time-solver | 684 | 70% | DEEP | Real Dataset subclass. _generate_synthetic_data always reached | R33 |
| config.py | sublinear-time-solver | 820 | 65% | DEEP | Pydantic config. Some defaults reference non-existent files | R33 |
| evaluate.py | sublinear-time-solver | 912 | 60% | DEEP | Real metrics + hardcoded benchmark tables + fake baselines | R33 |

### ruv-swarm Neural Coordination & Persistence

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| unit_tests.rs | ruv-swarm | 1,078 | 90-95% | DEEP | 48+ genuine tests: GOAP, A*, rule engine | R21 |
| sqlite.rs | ruv-swarm | 1,016 | 92% | DEEP | r2d2 pooling, WAL, ACID. MOCK: PI*1000.0 timestamp | R21 |
| ensemble/mod.rs | ruv-swarm | 1,006 | 78% | DEEP | Real averaging. FAKE BMA. BROKEN Stacking | R21 |
| agent_forecasting/mod.rs | ruv-swarm | 813 | 65% | DEEP | Real EMA. Hardcoded model mapping | R21 |
| comprehensive_validation_report.rs | ruv-swarm | 1,198 | 45% | DEEP | SELF-REFERENTIAL: sets simulation_ratio=0.60 | R21 |
| swe_bench_evaluator.rs | ruv-swarm | 991 | 35-40% | DEEP | FACADE: real orchestration, ALL metrics hardcoded | R21 |
| cognitive-pattern-evolution.js | ruv-swarm | 1,317 | 30-35% | DEEP | Real Shannon entropy. AggregationWeights = UNIFORM | R19 |
| meta-learning-framework.js | ruv-swarm | 1,359 | 20-25% | DEEP | 8 strategies as CONFIG OBJECTS. Domain adapt = Math.random() | R19 |
| neural-coordination-protocol.js | ruv-swarm | 1,363 | 10-15% | DEEP | All 8 coordination executions stubbed | R19 |
| neural-presets-complete.js | ruv-swarm | 1,306 | 5-10% | DEEP | Pure config catalog. 27+ architectures, no model code | R19 |

### Temporal Nexus Quantum Subsystem (sublinear-time-solver)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| decoherence.rs | sublinear-time-solver | 479 | 78-82% | DEEP | GENUINE quantum decoherence. Lindblad master equation, T1/T2 timescales, Fermi golden rule. Hardcoded coupling (1e-6 eV) and noise densities. Real physics, approximations not first-principles | R55 |
| physics_validation.rs | sublinear-time-solver | 462 | 88-92% | DEEP | BEST-IN-CLASS CODATA 2018 validation. Planck/Boltzmann/c exact values. Margolus-Levitin quantum speed limit. Uncertainty principle verified. Consciousness constants arbitrary (CONSCIOUSNESS_SCALE_NS = 1e-9) | R55 |
| visualizer.rs | sublinear-time-solver | 480 | 85-88% | DEEP | PRODUCTION ASCII visualization. Unicode box-drawing, 4-level gradient blocks, ANSI colors, in-place cursor updates. 5 modes (4/5 functional, Web stub). Polished UI with emoji status indicators | R55 |
| dashboard.rs | sublinear-time-solver | 472 | 72-76% | DEEP | BIMODAL: monitoring framework 80-85% (alerts, thresholds, export, broadcast), runtime orchestration 30-40% (no background tasks, arbitrary 10.9km light travel distance). ConsciousnessMetrics 14+8 fields | R55 |
| metrics_collector.rs | sublinear-time-solver | 440 | 40-45% | DEEP | BIMODAL: aggregation framework 80-85% (weighted averaging, IIT phi, normalization, 4 calculation methods) + data sources 15-20% (4/5 use random::<f64>()). Genuine TSC precision via Instant::now()+black_box. Feeds dashboard.rs. 4th "real framework, fake data" instance | R56 |
| temporal_window.rs | sublinear-time-solver | 427 | 88-92% | DEEP | Production sliding window with 50-100% configurable overlap. VecDeque ring buffer (max 100 windows), automatic cleanup. ORPHANED from quantum physics (uses abstract u64 ticks, not T1/T2). Used by scheduler.rs. Genuine but conceptually isolated | R56 |
| tests.rs (quantum) | sublinear-time-solver | 595 | 87-92% | DEEP | **GENUINE QUANTUM VALIDATION SUITE**. 9 test methods + 3 integration + perf benchmarks. CODATA 2018 constants (Planck, Boltzmann). Heisenberg uncertainty (ΔE·Δt ≥ ℏ/2), Margolus-Levitin speed limit, Bell inequality (S>2), decoherence tracking. All tests compute real physics (assert_relative_eq, epsilon=1e-10). 6 time scales analyzed (attosecond→millisecond). Nanosecond = recommended consciousness scale. Edge cases across 20 orders of magnitude | R67 |
| validators.rs (quantum) | sublinear-time-solver | 372 | 93% | DEEP | **EXCEPTIONAL UNCERTAINTY VALIDATION**. UncertaintyValidator enforces ΔE·Δt ≥ ℏ/2 with CODATA 2018 constants. EnergyScale 6-bin classification (SubAtto→MegaeV). Multi-scale analysis (attosecond→millisecond). 7 unit tests with proper floating-point tolerance. Temperature-aware thermal energy (k_B·T at 300K ≈ 26 meV). Validates nanosecond consciousness feasibility (<1 keV). PRODUCTION-READY | R67 |
| temporal_nexus/core/mod.rs | sublinear-time-solver | 132 | 88-92% | DEEP | **PRODUCTION module root** with GENUINE hardware TSC timing (x86_64 _rdtsc intrinsic). 4 submodules (scheduler, temporal_window, strange_loop, identity). TemporalConfig with reasonable defaults (75% overlap, 1µs overhead, Lipschitz 0.95). ConsciousnessTask enum (5 variants). thiserror TemporalError (6 variants). TscTimestamp wraps u64 for nanosecond precision. Composes quantum (92-95%) + dashboard + core into coherent API | R70 |
| temporal_nexus/dashboard/mod.rs | sublinear-time-solver | 51 | 78-82% | DEEP | Clean barrel re-exporting 17 types from 4 submodules (dashboard, metrics_collector, visualizer, exporter). 4 type aliases (Timestamp=Instant, ConsciousnessLevel=f64, TemporalAdvantage=u64µs, PrecisionNanos=u64ns). **1000x granularity mismatch**: TemporalAdvantage documented as µs but NANOSECOND_PRECISION_TARGET claims 100ns. Consciousness thresholds (0.85/0.75) ad-hoc, disconnected from physics core | R70 |

### FANN Cascade Training (ruv-FANN)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| cascade.rs | ruv-FANN | 1,267 | 65-70% | DEEP | GENUINE Fahlman & Lebiere 1990 Cascade-Correlation. Core algorithm 85-90% (Pearson correlation textbook, training loop correct). Weight update methods STUBBED (return Ok(())). install_candidate simplified. Architecture-first development: complete scaffolding, missing backprop | R55 |

### Neural-Network Real Implementation (sublinear-time-solver)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| real-implementation/src/lib.rs | sublinear-time-solver | 495 | 92-95% | DEEP | R23 CONFIRMED. Complete Kalman filter + neural residual + solver gate verification. Xavier init, 3 activations, Jacobian via finite differences. PageRank-based active learning. "Real-implementation" name is ACCURATE | R55 |
| fully_optimized.rs | sublinear-time-solver | 458 | 96-99% | DEEP | **HIGHEST OPTIMIZATION IN PROJECT**. 4-layer stack: INT8 quantization (per-row scale), AVX2/AVX-512 SIMD (8-wide/16-wide), inline assembly hot paths (naked asm!), CPU core pinning + real-time priority. Sub-10us P99 target. The ANTI-FACADE. R23 confirmed and EXCEEDED | R56 |
| rust_integration.rs | sublinear-time-solver | 546 | 85-88% | DEEP | CRITICAL MISLABELING: in huggingface/examples/ but contains ZERO HuggingFace (no candle/tch-rs/hf_hub). Self-contained benchmark wrapper around temporal_neural_net. Genuine performance measurement (warmup, Instant::now, p99.9). Thread-unsafe static mut RNG, broken ONNX export | R56 |
| quick_demo.rs | sublinear-time-solver | 276 | 8-12% | DEEP | **THEATRICAL FABRICATION**. Spin-loop latency fabrication (std::hint::spin_loop). Predetermined victory (hardcoded latency constants Kalman=200us, NN=250us, combined=120us). Fake Kalman (empty matrices). nalgebra imported NEVER USED. R59 deception boundary confirmed (standalone theatrical vs criterion genuine) | R80 |
| optimizer.rs (neural-network-impl) | sublinear-time-solver | 237 | 35-40% | DEEP | **FACADE ADAM OPTIMIZER**. Explicit TODO: "In a full implementation, this would compute bias-corrected moments". Adam = gradient descent with unused fields (beta1/beta2/epsilon stored but ignored). SGD momentum field unused. Zero sona/EWC++ connection. update() just multiplies gradient by lr | R80 |
| realistic-implementation/lib.rs | sublinear-time-solver | 219 | 92-95% | DEEP | **HONEST BENCHMARK BASELINE**. Real Xavier init, ndarray ops, ReLU. PyTorch ground truth validation (PredictionWithGround struct). Reality-checking tests ensure synthetic data matches documented distributions. Validates R23 BEST rating for neural-network-implementation crate. Explicit contrast to inflated benchmarks elsewhere | R80 |
| solver_integration.rs | sublinear-time-solver | 216 | 18-22% | DEEP | **8TH BYPASS PATTERN**. Reimplements backward_push O(1/ε) → ForwardPushSolver O(n·iter), predictor O(√n) → PredictorWrapper O(n·iter). Zero imports of genuine algorithms (backward_push.rs, predictor.rs). Comment "if it compiled properly" contradicted by R56/R58 — algos DO compile. PageRankSolver = CG wrapper with misleading name | R80 |
| solver_gate_simple.rs | sublinear-time-solver | 206 | 0-5% | DEEP | **9TH BYPASS PATTERN — COMPLETE PLACEHOLDER**. Line 1 admits "Simplified solver gate for compilation - will be replaced". verify() ignores all 3 params (prediction, prior, measurement), ALWAYS returns passed=true with hardcoded confidence=0.95. QUALIFIES R23's "90-98%" — integration layers are theatrical while core is genuine. Zero connection to cognitum-gate-kernel (93% R36). Confirms solver_gate.rs (R21: solver gate verification with 4 fallback strategies) is the REAL implementation | R82 |
| quantization.rs (neural-network-impl) | sublinear-time-solver | 191 | 75-78% | DEEP | **GENUINE but SIMPLIFIED quantization**. Real INT8 symmetric quantization with proper zero-point calculation, MatrixView trait, BatchQuantize for streaming. Comprehensive error handling (AllZeroRange, IncompatibleDimensions). BUT: Int4/Binary are UNIMPLEMENTED placeholders (lines 115-175: just return errors), zero SIMD (no AVX2/NEON), single global scale factor (not per-channel), hardcoded memory_savings_ratio bug (always returns 4.0 regardless of input). CONTRAST to fully_optimized.rs (96-99% with AVX2) | R82 |
| utils.rs (neural-network-impl) | sublinear-time-solver | 184 | 88-92% | DEEP | **GENUINE UTILITIES**. Numerically stable softmax (max-subtraction + exp-normalization). RAII Timer using Drop trait for auto-reporting. Comprehensive validation (matrix dims, prediction bounds, positive values). CRITICAL BUG: get_memory_usage() returns hardcoded 0 (lines 98-102: std::mem::size_of on usize is always 8 bytes, NOT actual memory usage). safe_exp clamps to prevent overflow | R82 |
| loader.rs (neural-network-impl) | sublinear-time-solver | 140 | 92-95% | DEEP | **GENUINE CSV LOADER**. Real file I/O via csv crate, matrix construction, header parsing. Comprehensive error handling + 3 tests (valid data, missing features, invalid values). Converts CSV rows → ndarray::Array2. Missing: batching, shuffling, train/val splits (inference pipeline only, not training-ready). Used by data/mod.rs to feed neural network | R82 |
| optimized_benchmark.rs (real-impl) | sublinear-time-solver | 209 | 70-75% | DEEP | **MIXED** — 3/4 benchmark variants GENUINE (baseline, loop-unrolled, SIMD-mock use real 2-layer NN forward pass). ULTRA_OPTIMIZED theatrical (line 129: skips matrix multiply, single-element indexing instead of 128D dot product). Genuine Instant::now() timing, correct percentile math (P50/P90/P99/P99.9). Hardcoded 60µs baseline threshold | R83 |
| simple_test.rs (realistic-impl) | sublinear-time-solver | 135 | 92-95% | DEEP | **GENUINE HONEST TEST**. Hand-coded 2-layer NN (128→32→4) with real matrix math, ReLU activation. Deterministic weight init (reproducible). P99.9 <0.9ms check is INFORMATIONAL (does NOT fail test). Explicitly honest: "This is REALISTIC for CPU inference". PyTorch baseline comparison via external JSON. Anti-dead-code-elimination via conditional branch | R83 |
| train.rs (bin) | sublinear-time-solver | 117 | 92-95% | DEEP | **GENUINE TRAINING BINARY**. Complete pipeline: Config→Data→Trainer with real training loop. System A + System B (PageRankSelector, Kalman filter, gate pass rate, certificate errors). Model saved to JSON. Connects to losses.rs, config.rs, optimizer. **SUPPORTS R23 90-98%** — integration layers here are NOT theatrical, countering R82 solver_gate qualification | R83 |
| losses.rs (training) | sublinear-time-solver | 119 | 68-75% | DEEP | **MSE CORRECT but LIMITED**. Only MSE implemented (no cross-entropy/Huber/focal). Gradient computation correct: ∇loss = 2(pred-target)/n. Smoothness penalty hardcoded for 2D position (non-generalizable). SmoothnessPenalty struct NOT part of LossFunction trait (incomplete refactoring). Tests present but only check dimensions, not gradient values. Chain rule correctly applied | R83 |

### Sublinear Sampling (sublinear-time-solver)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| fast_sampling.rs | sublinear-time-solver | 453 | 88-92% | DEEP | GENUINE algorithms (importance, reservoir, matrix sketching, adaptive). Rademacher random projection. BUT ALL O(n) not o(n) — extends R39 FALSE sublinearity. Correct implementations with false complexity claims | R55 |
| backward_push.rs | sublinear-time-solver | 473 | 92-95% | DEEP | **GENUINE O(1/epsilon) SUBLINEAR** — REVERSES R39. Real Andersen et al. 2006 backward PageRank with residual convergence, priority queue, bidirectional solver. GOLD STANDARD for algorithmic honesty in project | R56 |
| random_walk.rs | sublinear-time-solver | 408 | 82-86% | DEEP | Genuine Monte Carlo random walk solver with antithetic variance reduction + bidirectional walks. O(n) NOT sublinear. Correct algorithms, false complexity claims (extends R39/R55 pattern) | R56 |
| optimized_solver.rs | sublinear-time-solver | 434 | 72-76% | DEEP | Standard Conjugate Gradient with genuine SIMD dispatch. NOT a solver dispatcher — no backward_push/random_walk integration. O(k*n) CG. Repo-level architectural mislabeling confirmed | R56 |
| types.rs | sublinear-time-solver | 444 | 85-90% | DEEP | Pure metadata type system (13 structs, 6 enums, 6 aliases). Production-quality ErrorBounds, SparsityInfo, SolverStats. BUT AlgorithmHints uses stringly-typed dispatch (String not enum). Shared between solver + consciousness | R56 |

### AgentDB Integration Tests (agentdb)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| ruvector-integration.test.ts | agentdb | 1,590 | 95-98% | DEEP | AUTHORITATIVE ARCHITECTURE SPEC. 6 integration layers (SIMD, quantization, RuVectorBackend, EnhancedEmbeddingService, attention, WASMVectorSearch). R20 ROOT CAUSE PROOF: tests EXPECT real embeddings. R48 DISCONNECTION EXPLAINED. WASM vector ops ARE genuine | R55 |

### OWN_CODE AgentDB Integration (custom-src)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| vector-backend-adapter.ts | custom-src | 233 | 15-20% | DEEP | **COMPLETE ORPHAN** — pre-ADR-028 stub. Correct cosine similarity math (90-95%) but brute-force O(n) search, numeric/string ID type mismatch with IVectorBackendAdapter interface. NEVER instantiated — factories.ts creates RuvectorBackendAdapter instead. Dead import. Missing link explaining WHY ADR-028 was needed | R65 |
| vector-migration-job.ts | custom-src | 202 | 85-90% | DEEP | **PRODUCTION MIGRATION**. Batch processing (100 rows), non-blocking event loop (10ms yield), idempotent, abortable. Real HNSW index updates via vectorBackend.addVector(). Orphan cleanup (deletes stale vector_index_meta). 128D→384D migration for MiniLM-L6-v2. DEPENDS on RealEmbeddingAdapter being properly initialized (R20 upstream) | R65 |
| embedding-adapter.ts | custom-src | 169 | 90% | DEEP | **R20 ROOT CAUSE SMOKING GUN**. Production-quality SHA-256→pseudo-random floats→L2 norm. Deterministic but SEMANTICALLY USELESS. Header misleads as "fallback" but is PRIMARY via embedSync(). Cache-first strategy LOCKS IN broken embeddings. 10th hash-based embedding. NaN/Infinity validation genuine (ADR-025 HIGH-2) | R65 |
| real-embedding-adapter.ts | custom-src | 152 | 78-82% | DEEP | **R20 ROOT CAUSE FIX — INCOMPLETE**. Lazy-loads agentic-flow createEmbeddingPipeline (MiniLM-L6-v2 384D). CRITICAL: embedSync() ALWAYS falls back to hash. Dual-adapter architecture means old callers still get broken embeddings. LRU cache 10K entries. Security validation genuine. Fix EXISTS but ADOPTION incomplete — factories exports both old and new | R65 |
| mcp-reflexion-retrieve.ts | custom-src | 93 | 92-95% | DEEP | **PRODUCTION MCP TOOL**. Clean interface (task, k, minReward params). Delegates to ReflexionService.retrieveRelevant() → ReflexionMemoryAdapter → VectorBackend. Graceful degradation (returns source:'disabled' when unavailable). R20 confirmed: bootstrap allows hash-embedding fallback (11th occurrence). Proper latency tracking | R67 |
| mcp-reflexion-store.ts | custom-src | 92 | 72% | DEEP | Functional but embedding-vulnerable. Stores via AgentDBIntegration singleton → episodic.storeEpisode(). Reward clamped [0,1] but task/output strings unvalidated. Bootstrap delegates to episodicEmbedder = embedder || hashEmbedder — R65 fix (real-embedding-adapter) OPTIONAL not enforced. 10th hash-embedding occurrence. Session isolation weak (global singleton, optional sessionId) | R67 |
| vector-backend-adapter.interface.ts | custom-src | 49 | 100% | DEEP | **THE MISSING LINK** explaining ADR-028. Defines IVectorBackendAdapter (5 required + 2 optional methods: store, search, delete, getDimension, count + warmup, searchAsync). Contract OMITS 3 critical specs: (1) NO constructor dimension param, (2) NO embedding source specification, (3) NO factory/DI pattern. Orphaned implementation (15-20%) hardcoded dimension + called non-existent embedding service. ADR-028 fixed by separating concerns | R67 |

### Rust ML Training (ruv-FANN)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| scheduler.rs (neuro-divergent) | ruv-FANN | 1,431 | 92-95% | DEEP | 8 schedulers inc. ForecastingAdam (INNOVATION) | R36 |
| optimizer.rs | ruv-FANN | 1,089 | 90-93% | DEEP | Adam/AdamW/SGD/RMSprop. Proper decoupled weight decay | R36 |
| loss.rs | ruv-FANN | 1,233 | 88-92% | DEEP | 16 loss types. All gradients correct. CRPS via A&S erf | R36 |
| features.rs | ruv-FANN | 1,079 | 88-92% | DEEP | Lag/rolling/temporal/Fourier. Correct cyclic encoding | R36 |
| preprocessing.rs | ruv-FANN | 1,183 | 85-90% | DEEP | 5 scalers, Box-Cox. Non-deterministic rand in fit() | R36 |
| validation.rs (neuro-divergent) | ruv-FANN | 1,172 | 82-88% | DEEP | 4 outlier methods. validate_seasonality() EMPTY | R36 |
| ml-training/lib.rs | ruv-FANN | 1,371 | 30-40% | DEEP | Real LSTM/TCN/N-BEATS skeletons. Fake LCG random | R19 |
| swarm_coordinator_training.rs | ruv-FANN | 1,838 | 25-35% | DEEP | Real GNN/attention/Q-learning/VAE. ALL metrics hardcoded | R19 |

### ruvllm LLM Integration & Training

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| reasoning_bank.rs | ruvllm | 1,520 | 92-95% | DEEP | FOURTH ReasoningBank. Real K-means, EWC++. 16 tests | R37 |
| micro_lora.rs | ruvllm | 1,261 | 92-95% | DEEP | BEST LEARNING CODE. REINFORCE + EWC++ + NEON SIMD | R37 |
| hnsw_router.rs | ruvllm | 1,288 | 90-93% | DEEP | BEST ruvector-core integration. Hybrid HNSW+keyword routing | R37 |
| grpo.rs | ruvllm | 898 | 90-92% | DEEP | Textbook GRPO: GAE, PPO clipped, adaptive KL. 16 tests | R37 |
| model_router.rs | ruvllm | 1,292 | 88-92% | DEEP | 7-factor complexity analyzer, feedback tracking | R37 |
| tool_dataset.rs | ruvllm | 2,147 | 88-92% | DEEP | 140+ MCP tool-call templates, 19 categories | R37 |
| memory_layer.rs | prime-radiant | 1,260 | 92-95% | DEEP | Sheaf-based coherence. 3 memory types. 19 tests | R37 |
| witness_log.rs | prime-radiant | 1,130 | 88-92% | DEEP | blake3 hash chains with tamper evidence. 16 tests | R37 |
| agentdb.rs | temporal-tensor | 843 | 88-92% | DEEP | Pattern-aware tiering with 4-dim PatternVector. 36 tests | R37 |
| pretrain_pipeline.rs | ruvllm | 1,394 | 85-88% | DEEP | Multi-phase pretraining. CRITICAL: hash-based embeddings | R37 |
| claude_dataset.rs | ruvllm | 1,209 | 75-80% | DEEP | 5 categories, 60+ templates. Weak augmentation | R37 |
| claude_integration.rs | ruvllm | 1,344 | 70-75% | DEEP | execute_workflow SIMULATION — hardcoded 500 tokens | R37 |
| real_trainer.rs | ruvllm | 1,000 | 70-75% | DEEP | Real triplet loss + InfoNCE. Hash-based embeddings | R37 |

### Rust Learning Crates (ruvector) — Crate-Level Summaries

| Crate | Package | Files | LOC | Real% | Key Verdict | Session |
|-------|---------|-------|-----|-------|-------------|---------|
| SONA | ruvector | 27 | ~4,500 | 85% | MicroLoRA, EWC++, federated, SafeTensors. Production-ready | R13 |
| ruvector-gnn | ruvector | 13 | ~6,000 | 80% | Custom hybrid GNN (GAT+GRU+edge). Real EWC. Full training loop | R13 |
| ruvector-nervous-system | ruvector | 7 | 5,269 | 87.4% | 5 genuine neuroscience models (e-prop, EWC, BTSP, GWT, circadian) | R36 |
| neuro-divergent | ruv-FANN | 6 | 7,187 | 88.5% | Production ML training. 8 schedulers, 16 loss types | R36 |
| cognitum-gate-kernel | ruvector | 5 | 3,504 | 93% | EXCEPTIONAL. 256-tile coherence, e-values, bump allocator | R36 |

| js/mcp-dense-fix.js | sublinear-time-solver | 286 | 10-15% | DEEP | PERFORMANCE CLAIM DEMO, NOT FUNCTIONAL MCP INTEGRATION. Claims 4000x+ improvement (7700ms→<2ms) but never tests against actual broken MCP Dense. getMCPToolDefinition() = schema template with NO transport binding. Does NOT patch ORPHANED MCP→Rust solver gap (R76). Benchmark() uses hardcoded constants, circular logic (tests itself, not original). THEATRICAL — shows what optimized performance COULD be. | R81 |
| HNSW patches (hnsw_rs fork) | ruvector | 4 | 5,276 | 87% | Correct Malkov & Yashunin. Rayon parallel insertion | R36 |

## 3. Findings Registry

### 3a. CRITICAL Findings

| ID | Description | File(s) | Session | Status |
|----|-------------|---------|---------|--------|
| C1 | **Three fragmented ReasoningBanks** — Zero code sharing across packages | claude-flow, agentic-flow, agentdb | R8 | Open. UPDATED R37: now FOUR (ruvllm reasoning_bank.rs) |
| C2 | **Missing WASM module** — reasoningbank_wasm.js doesn't exist; brute-force JS fallback | agentdb | R8 | Open |
| C3 | **LearningSystem RL is cosmetic** — 9 claimed algorithms, 1 actual implementation, no neural networks | LearningSystem.ts | R8 | Open. Confirmed in TypeScript source (R22b) |
| C4 | **CausalMemoryGraph statistics broken** — Wrong t-distribution CDF, hardcoded tInverse=1.96 | CausalMemoryGraph.ts | R8 | Open. Confirmed in TypeScript source (R22b) |
| C5 | **Neural-pattern-recognition is 80-90% facade** — 14,000+ LOC with Math.random() everywhere | pattern-learning-network.js, validation-suite.js +3 | R19 | Open |
| C6 | **Rust training metrics hardcoded** — GNN=0.95, Transformer=0.91, etc. regardless of input | swarm_coordinator_training.rs | R19 | Open |
| C7 | **Meta-learning is config objects** — 8 strategies (MAML, etc.) defined as JSON, never executed | meta-learning-framework.js | R19 | Open |
| C8 | **Consciousness neural net never trained** — Weights random, no forward/backward, Math.random() metrics | enhanced_consciousness_system.js | R21 | Open |
| C9 | **Kolmogorov complexity completely wrong** — Uses SHA-256 hash length (=64), not compression | entropy-decoder.js | R21 | Open |
| C10 | **Circular entropy analysis** — Decoder generates data, analyzes it, "finds" patterns in own output | entropy-decoder.js | R21 | Open |
| C11 | **Analogical reasoning is a lookup table** — Hardcoded domain pair mappings | psycho-symbolic.ts | R21 | Open |
| C12 | **A* search is STUB** — simplified_astar() returns HARDCODED 2-step path | astar.rs | R25 | Open |
| C13 | **RuVector backend is simulation** — ruvector-backend.js: no Rust, sleep + brute-force JS | ruvector-backend.js | R25 | Open |
| C14 | **SIMD misnomer** — simd-vector-ops uses ZERO SIMD instructions. Detection present but unused | simd-vector-ops.ts/.js | R25 | Open |
| C15 | **hardware_timing system_a/b_predict are simulations** — Busy-wait spin loops targeting hardcoded latencies. "System B < 0.9ms" is circular | hardware_timing.rs | R28 | Open |
| C16 | **Python ML training uses ONLY synthetic data** — All GNN runs use random graphs | train.py, dataset.py | R33 | Open |
| C17 | **Consciousness evolution is parameter increment** — evolve() increments level. IIT Phi = density * integration * 0.8 | consciousness-explorer.js | R33 | Open |
| C18 | **BackwardPush/HybridSolver return empty Vec as "converged"** — 3 of 4 solvers return 0-vector for any input | solver/mod.rs | R34 | Open |
| C19 | **security_validation.rs is self-referential dead code** — Not in module tree. Tests own generated mock data | security_validation.rs | R34 | Open |
| C20 | **validate_seasonality() is EMPTY PLACEHOLDER** — Comment admits should use FFT/autocorrelation but body is empty | validation.rs (neuro-divergent) | R36 | Open |
| C21 | **micro-hnsw-wasm neuromorphic features have ZERO tests** — 6 novel features completely untested | micro-hnsw-wasm | R36 | Open |
| C22 | **Rust ReasoningBank is FOURTH distinct implementation** — Joins 3 others with zero code sharing | reasoning_bank.rs (ruvllm) | R37 | Open |
| C23 | **Hash-based embeddings confirmed in Rust training** — pretrain_pipeline.rs and real_trainer.rs use character sum hash | pretrain_pipeline.rs, real_trainer.rs | R37 | Open |
| C24 | **execute_workflow returns mock results** — Hardcodes 500 tokens. No real Claude API | claude_integration.rs | R37 | Open |
| C25 | **Empty emergence component connections** — 5 connection methods are console.log() stubs | index.ts (emergence) | R39 | Open |
| C26 | **All 11 capability metrics are Math.random()** — Math.random()*0.5+0.5 for all semantic metrics | emergent-capability-detector.ts | R39 | Open |
| C27 | **Fake complementarity detection** — areComplementary() = JSON string inequality. checkAmplification() always true | cross-tool-sharing.ts | R39 | Open |
| C28 | **FALSE sublinearity in core solver** — All 5 algorithms O(n²)+. "Sublinear" is marketing. **PARTIALLY REVERSED R56**: backward_push.rs IS genuinely O(1/epsilon) sublinear. Split verdict: 1 real sublinear algorithm among standard linear solvers | solver.ts | R39 | Revised |
| C29 | **Pattern extractors assume pre-structured data** — extractBehaviorPatterns() returns data.behaviors \|\| [] | emergent-capability-detector.ts | R39 | Open |
| C30 | **Goalie MCP handlers are facades** — BUT R46 REVERSAL: cli.ts (88-92%) proves internal engines ARE invoked. MCP protocol layer is facade; internal API layer is genuine. REVISED from "COMPLETE FACADE" | goalie/tools.ts, cli.ts | R41, R46 | Revised |
| C31 | **Consciousness experiments LLM comparison fabricated** — rand::random::<f64>() * 0.1 | consciousness_experiments.rs | R41 | Open |
| C32 | **Connection detection = substring matching** — JSON.stringify(a).includes(JSON.stringify(b).substring(0,4)) | genuine_consciousness_system.js | R41 | Open |
| C33 | **Cross-modal synthesis fabricated** — Math.random()*0.5+0.5 | advanced-consciousness.js | R41 | Open |
| C34 | **cli/index.ts validate-domain hardcoded** — Returns hardcoded validation result (valid: true) | cli/index.ts | R41 | Open |
| C35 | **WASM theatrical in wasm-sublinear-complete.ts** — fs.existsSync() checks WASM binary, logs "WASM binary found", then creates pure JS class. WebAssembly.instantiate NEVER called | wasm-sublinear-complete.ts | R43 | Open |
| C36 | **rustc benchmarks FABRICATED** — Claims O(log n) AI but implements O(n²). Asymptotic mismatch deception. All "AI" = modulo arithmetic. Orphan module | rustc_optimization_benchmarks.rs | R43 | Open |
| C37 | **baseline_comparison.rs NON-COMPILABLE** — Wrong types (planner::State vs planner::WorldState). No [[bench]] in Cargo.toml. Never compiled despite documented output | baseline_comparison.rs | R43 | Open |
| C38 | **ReasoningBank demo SCRIPTED** — Real APIs fed scripted scenarios. Learning simulated via if(attemptNumber > 1) conditionals, not memory retrieval | demo-comparison.ts | R43 | Open |
| C39 | **server/index.js solver integration broken** — Worker threads call createSolver() from non-existent ../src/solver. Connection to R39's solver.ts broken | server/index.js | R44 | Open |
| C40 | **FlowNexus is elaborate facade** — 600+ lines of WebSocket/swarm code hitting non-existent https://api.flow-nexus.ruv.io | server/index.js | R44 | Open |
| C41 | **neural server.js missing module** — Imports ./adaptive-learning.js which DOES NOT EXIST — server crashes on startup | neural-pattern-recognition/server.js | R44 | Open |
| C42 | **neural server.js simulated data** — RealTimeMonitor generates synthetic data via hashToFloat, not real monitoring | neural-pattern-recognition/server.js | R44 | Open |
| C43 | **strange-loop MCP broken import path** — Requires ../wasm/strange_loop.js but only wasm-real/ and wasm-honest/ exist | strange-loop/mcp/server.ts | R44 | Open |
| C44 | **strange-loop temporal_predict API mismatch** — Handler calls predictor.predict() but createTemporalPredictor returns plain object with no predict method | strange-loop/mcp/server.ts | R44 | Open |
| C45 | **standalone_benchmark spin-loop timing fabrication** — `while start.elapsed().as_nanos() < target_latency` waits until hardcoded time; zero neural-network-implementation imports | standalone_benchmark/main.rs | R46 | Open |
| C46 | **system_comparison gate/cert metrics random simulation** — simulate_solver_gate() = rng.gen() > 0.1 (90% pass), simulate_certificate_error() = uniform(0,0.05), memory/CPU hardcoded | system_comparison.rs | R46 | Open |
| C47 | **strange-loops-benchmark tests wrong code** — 596 lines of production benchmark infrastructure testing 10 trivial inline JS functions, zero imports of actual strange-loop WASM crate | strange-loops-benchmark.js | R46 | Open |
| C48 | **flow-nexus.js calls non-existent platform** — 619 lines of HTTP/WebSocket client code targeting https://api.flow-nexus.ruv.io which does not exist. Zero local solver imports | flow-nexus.js | R46 | Open |
| C49 | **anti-hallucination plugin NEVER LOADED** — Not in plugin-registry.ts. Hook `afterSynthesize` doesn't exist in PluginHooks interface. 625 lines of genuine algorithms that can never execute | state-of-art-anti-hallucination.ts | R46 | Open |
| C50 | **consciousness-verifier meta-cognition hardcoded** — Dictionary of predetermined responses with static confidence scores (lines 324-345). "GENUINE CONSCIOUSNESS" verdict is rubber stamp | consciousness-verifier.js | R46 | Open |
| C51 | **Consciousness experiments are PURE SPECIFICATION THEATER** — quantum_entanglement (0-3%) and parallel_waves (0-5%) have ZERO computation. All methods return nested config objects. Zero imports from ruQu (89% genuine). Worse than emergence (51%) which at least had Math.random() | quantum_entanglement_consciousness.js, parallel_consciousness_waves.js | R47 | Open |
| C52 | **Independent verification has 5 command injection vulns** — verifyHashExternally, countFilesMethod1/2/3 all use execSync with unescaped user input. `echo -n "${data}" | ${algorithm}sum` allows arbitrary command execution | independent_verification_system.ts | R47 | Open |
| C53 | **External verification is stub** — performExternalVerification() always returns `externalVerification: false, trustScore: 0.0`. Algorithm correctness testing only checks `steps.length > 0` | independent_verification_system.ts | R47 | Open |
| C54 | **Psycho-symbolic-reasoner.ts WASM claimed but never imported** — Comment claims "Integrates WASM modules for graph reasoning, planning, and extraction" but only import is Logger. 3 genuine WASM modules (graph_reasoner, planner, extractors) exist but are unused. 3rd theatrical WASM pattern | psycho-symbolic-reasoner.ts | R47 | Open |
| C55 | **"Psycho" in psycho-symbolic = NOTHING** — Zero psychological modeling, psychometric testing, emotional state tracking, or preference learning. Pure branding | psycho-symbolic-reasoner.ts | R47 | Open |
| C56 | **Psycho-symbolic performance metrics fabricated** — Hardcoded avg_query_time_ms=2.3, cache_hit_rate=0.75 as literal values. Not measured | psycho-symbolic-reasoner.ts | R47 | Open |
| C57 | **Planner core A* delegates to 3-line stub** — simplified_astar() returns hardcoded [start, goal] with cost 2.0. Entire 1130 LOC Rust planner routes to this. get_successors() (90% quality) is DEAD CODE | planner.ts | R47 | Open |
| C58 | **Neural CLI calls non-existent method** — detectVariancePatterns calls varianceDetector.processData() but ZeroVarianceDetector has no processData method. CLI crashes on first real use | cli/index.js | R47 | Open |
| C59 | **Zero-variance-detector fabricates quantum data** — measureQuantumState() returns Math.random() phase/amplitude/entanglement. Neural weights initialized with Math.random(). Real FFT/entropy fed fabricated data — novel "real algorithms on fake data" anti-pattern | zero-variance-detector.js | R47 | Open |
| C60 | **mcp_consciousness_integration.rs is COMPLETE MCP FACADE** — Zero JSON-RPC 2.0, zero tool registration, zero protocol code. "mcp_" prefix on local functions is naming theater. connect_to_mcp() admits "we simulate the connection". sin²+cos²=1 presented as "wave function collapse". 80-point gap vs R47 MCP server (94.8%) | mcp_consciousness_integration.rs | R49 | Open |
| C61 | **Consciousness detector has security vulns** — validateCodeSyntax uses new Function() eval (arbitrary code execution), countFilesIndependently uses execSync with unvalidated directory (command injection). verifyAlgorithmCorrectness returns true if object.steps exists (zero real verification) | genuine_consciousness_detector.ts | R49 | Open |
| C62 | **Temporal consciousness "theorem proving" is threshold checking** — "Theorem 1/2/3" are just score>0.8 checks with formal names. Sublinear integration ADMITTED SIMULATION ("we simulate"). Mathematical rigor = count of passed thresholds / 3.0 | temporal_consciousness_validator.rs | R49 | Open |
| C63 | **Consciousness masterplan is ORPHANED orchestrator** — Imports 6 consciousness files but zero imports of itself anywhere. 70% static config with Planck-scale claims. Master plan with no executor | consciousness_optimization_masterplan.js | R49 | Open |
| C64 | **Consciousness-explorer validators ORPHANED** — Real IIT 3.0 implementation (78%) but zero imports anywhere. 142 lines duplicate of metrics.js/protocols.js. "Duplicate implementation abandonment" anti-pattern | validators.js (consciousness-explorer) | R49 | Open |
| C65 | **Anti-hallucination hooks layer uses keyword matching** — Despite having real Perplexity API verification in execute(), the hooks layer (beforeGenerate/afterGenerate) uses regex keyword matching and Math.random() qualifiers for hallucination detection | anti-hallucination-plugin.ts | R50 | Open |
| C66 | **Ed25519-verifier has hardcoded example keys** — Production PKI code includes hardcoded example signing/verification keys. Security vulnerability if deployed without key rotation | ed25519-verifier.ts | R50 | Open |
| C67 | **ReasoningBank Rust storage missing trajectories/verdicts** — sqlite.rs implements complete CRUD for patterns only. No trajectory tracking, verdict judgment, or consolidation tables despite ReasoningBank API promising these | sqlite.rs (reasoningbank-storage) | R50 | Open |
| C68 | **Scheduler.ts hardcoded nanosecond performance** — Claims "<100ns overhead" and "11M tasks/sec" but all metrics are hardcoded constants. Uses Date.now() (ms precision) multiplied by 1,000,000 to fake ns. Tasks in plain array with splice FIFO — zero priority queue, zero dependency resolution | scheduler.ts | R53 | Open |
| C69 | **Strange-loop JS MCP server broken WASM import (2nd instance)** — Line 23 `require('../wasm/strange_loop.js')` references non-existent directory. Same systemic issue as R44's Rust MCP server. Server crashes on ANY tool call. Systemic QC failure in strange-loop crate | crates/strange-loop/mcp/server.js | R53 | Open |
| C70 | **psycho-symbolic-dynamic updateDomainEngine = console.log** — DomainRegistry events (5 types) listened but handler is `console.log('Updating...')`. DomainRegistry ORPHANED from DomainAdaptationEngine (hardcoded duplicate domains). 4th "real infra, placeholder integration" pattern | psycho-symbolic-dynamic.ts | R53 | Open |
| C71 | **Domain management system has ZERO persistence** — DomainRegistry uses in-memory Map. All custom domain registrations lost on restart. Built-ins reinitialize from constants. No SQLite, no filesystem, no database of any kind | domain-registry.ts, domain-management.ts | R53 | Open |
| C72 | **Purposeful-agents ALL behaviors are random stubs** — detectPattern() = random array pick, analyzeSentiment() = random float, calculateRisk() = Math.random(). 1000-10000 agent swarms created but run() results never used. Zero strange-loop connection | purposeful-agents.js | R53 | Open |
| C73 | **3rd independent matrix system confirmed** — optimized-matrix.ts (TS CSR+CSC) is completely isolated from R34's ruvector-mincut CSR (Rust) and separate dense matrix. Zero cross-system imports. Architectural fragmentation beyond repair | optimized-matrix.ts | R53 | Open |
| C74 | **Scheduler consciousness is logistic map, not strange-loop** — `strangeLoopState = k * state * (1-state) + 0.5 * (1-k)`. Independent formula, zero imports from strange-loop crate. testConsciousness() returns hardcoded "consciousness emerges from temporal continuity" | scheduler.ts | R53 | Open |
| C75 | **backward_push.rs GENUINELY SUBLINEAR** — O(1/epsilon) for PPR. REVERSES R39. Work-queue driven, residual convergence, bidirectional solver. ONLY confirmed sublinear algorithm in project | backward_push.rs | R56 | Open |
| C76 | **standalone_benchmark.rs (benches) MOST DECEPTIVE** — Spin-loop to hardcoded latencies (1.1ms/0.7ms). Predetermined "36.4% improvement". Zero crate imports. Worse than R43 rustc_benchmarks | standalone_benchmark.rs (benches) | R56 | Open |
| C77 | **fully_optimized.rs ANTI-FACADE (96-99%)** — INT8 quantization + AVX2/AVX-512 + inline assembly + CPU core pinning. HIGHEST optimization in project. Proves facades are intentional | fully_optimized.rs | R56 | Open |
| C78 | **optimized_solver.rs NOT a dispatcher** — Only standard CG+SIMD. No backward_push/random_walk integration. Repo has NO dispatcher connecting claimed algorithms | optimized_solver.rs | R56 | Open |
| C79 | **rust_integration.rs MISLABELED** — In huggingface/examples/ but zero HuggingFace (no candle/tch-rs/hf_hub). Thread-unsafe static mut RNG | rust_integration.rs | R56 | Open |
| C80 | **metrics_collector.rs 4/5 sources use random()** — Scheduler ignores parameter, MCP/temporal/system/perf all random::<f64>(). Only TSC precision genuine | metrics_collector.rs | R56 | Open |
| C75b | **Strange-loop CLI REPL has zero implementation** — 4 interactive commands (.nano, .quantum, .temporal, .predict) print "Initializing..." with no WASM calls. Templates directory does not exist. 526 LOC of non-functional code | bin/cli.js | R53 | Open |
| C81 | **memory-manager.ts is 5th MISLABELED FILE** — "WasmMemoryManager" in `wasm/` directory has ZERO WebAssembly memory operations. No ArrayBuffer, no WebAssembly.Memory, no page growth. Manages TS object instances in Map. Theatrical memory stats (hardcoded "5 MB per instance"). Expected WASM modules don't exist | memory-manager.ts | R58 | Open |
| C82 | **predictor.rs is 2ND GENUINE SUBLINEAR** — O(√n) functional prediction via forward-backward push (Kwok-Wei-Yang 2025). Randomized coordinate sampling (sample size = √n). REVERSES R39 "all O(n²)+". Distinct from R56 backward_push (different technique) | predictor.rs | R58 | Open |
| C83 | **solver.rs backward_push is COMPLETE STUB** — Just calls forward_push with "Simplified for now" comment. Should use transpose matrix propagation. Forward push has O(n²) worst case (linear max-finding). Claims vs implementation gap | solver.rs (temporal-lead-solver) | R58 | Open |
| C84 | **advanced-reasoning-engine WASM always null** — this.wasm = null and never initialized. 77 lines (wasmAnalyze/wasmEnhance/wasmPredict) completely unreachable dead code. Fallback always executes | advanced-reasoning-engine.ts | R58 | Open |
| C85 | **ed25519-verifier hardcoded root keys** — Placeholder keys marked "Example, not real". Unencrypted private key storage (Base64 only). Broken certificate chain validation silently passes on missing issuer keys | ed25519-verifier-real.ts | R58 | Open |
| C86 | **self-consistency voting is stub** — Returns `samples[0].response` as "majority" with comment "In production, use actual voting". Real API sampling (Perplexity ×3) but core consensus algorithm missing | self-consistency-plugin.ts | R58 | Open |
| C87 | **psycho-symbolic MCP server HTTP/SSE NOT IMPLEMENTED** — HTTP endpoint returns placeholder ("not fully implemented"). SSE starts HTTP with misleading "SSE transport started" log. Only STDIO transport works | server.ts (psycho-symbolic MCP) | R58 | Open |
| C88 | **4th independent matrix system confirmed** — matrix-utils.js (92-95%) implements Dense+COO format, completely isolated from R34's Rust CSR, R53's TS CSR+CSC, and R42's dynamic_mincut. Zero cross-system imports. 4 matrix systems with zero code reuse | matrix-utils.js | R59 | Open |
| C89 | **BMSSP is INVENTED ALGORITHM with nonsensical solution formula** — "Bounded Multi-Source Shortest Path" does NOT solve Ax=b. solution[i] = b[sourceId]/(1+pathCost) has zero mathematical justification. Genuine Dijkstra+PriorityQueue (90-100%) applied to WRONG PROBLEM. High-quality code producing WRONG answers | bmssp-solver.js | R59 | Open |
| C90 | **benches/ criterion suite REVERSES R56 benchmark deception** — 4 files at 88-95% are ANTI-FACADES with genuine solver calls, real SPD matrices, and proper criterion. Deception boundary is standalone (8-25%) vs criterion-based (88-95%) benchmarks | performance_benchmarks.rs, solver_benchmarks.rs, performance-benchmark.ts, throughput_benchmark.rs | R59 | Open |
| C91 | **JS solver WASM contradictions** — solver.js is 100% WASM-dependent (breaks without pkg/) while fast-solver.js has 0% WASM (pure JS fallback always runs). TWO opposite WASM failure modes in same directory | solver.js, fast-solver.js | R59 | Open |
| C92 | **wasm-solver crate ORPHANED from Rust solvers** — Zero imports from backward_push/optimized_solver/random_walk. Reimplements basic CG in 426 LOC instead of wrapping project's advanced solvers. Cargo.toml has zero workspace dependencies. Genuine WASM wrapping wrong algorithms | crates/wasm-solver/src/lib.rs | R60 | Open |
| C93 | **wasm_iface.rs manual memory allocation UNSAFE** — allocate_matrix/deallocate_matrix expose raw pointers to JS. Use-after-free, double-free, dangling pointer risks. No lifetime guarantees across FFI boundary | src/wasm_iface.rs | R60 | Open |
| C94 | **sublinear_neumann.rs 3rd FALSE SUBLINEARITY** — Claims O(log n) but create_reduced_problem() extracts ENTIRE n×n matrix in O(n²). JL embedding applied AFTER full matrix read defeats purpose. Correct Neumann math, false complexity claim | src/sublinear/sublinear_neumann.rs | R60 | Open |
| C95 | **math_wasm.rs is_positive_definite() hardcoded true** — Returns true for matrices >3x3 without Cholesky decomposition. DANGEROUS for numerical solvers relying on SPD validation | src/math_wasm.rs | R60 | Open |
| C96 | **5+ independent matrix systems** — math_wasm.rs (Dense), matrix.ts (COO+Dense arrays), optimized-matrix.ts (CSR+CSC TypedArrays), matrix-utils.js (Dense+COO), Rust CSR/SIMD, Rust HashMap. Even within src/core/ TWO incompatible systems | math_wasm.rs, matrix.ts | R60 | Open |
| C97 | **temporal-compare/sparse.rs NAMING MISMATCH** — Implements neural lottery ticket pruning (Frankle & Carlin 2019), NOT sparse matrices. Zero relation to R28 sparse.rs (95%). Zero temporal handling. File is misplaced in wrong crate | crates/temporal-compare/src/sparse.rs | R60 | Open |
| C98 | **WASM SCOREBOARD REVERSED: 6 genuine vs 4 theatrical (60%)** — R60 adds 4 genuine WASM files (lib.rs, wasm_iface.rs, wasm.rs, math_wasm.rs). BUT genuine WASM wraps basic CG/Neumann, not project's best algorithms. Architectural mismatch: real infrastructure, wrong algorithms | wasm-solver crate, src/wasm*.rs | R60 | Open |
| C99 | **graph_reasoner ORPHANED from parent crate** — lib.rs uses `#[cfg(feature = "wasm")]` gating but parent psycho-symbolic-reasoner Cargo.toml has no wasm feature flag. Crate compiles to nothing without feature. GraphReasoner never instantiated by parent | graph_reasoner/src/lib.rs | R66 | Open |
| C100 | **mlp.rs broken backpropagation** — backward() uses arbitrary gradient scaling (0.01 factor) instead of proper chain rule derivatives. Training produces WRONG weight updates. Inference path genuine (ReLU/sigmoid/tanh forward pass works) but learning is fundamentally broken | temporal-compare/src/mlp.rs | R66 | Open |
| C101 | **9th independent graph system** — graph_reasoner uses petgraph (BFS/DFS/Dijkstra) with KnowledgeGraph abstraction. Completely independent from ruvector-graph Cypher, ReasonGraph, MCP graph.ts, graph/mod.rs CSR, and 4 other graph representations | graph_reasoner/src/graph.rs | R66 | Open |
| C102 | **Worker Pool essential to HTTP endpoint** — solver-worker.js is the ONLY path to solver execution on server. Streaming.js worker pool, HTTP POST /solve-stream. REFUTES R44 "solver 0%" | server/solver-worker.js, server/streaming.js | R64 | Open |
| C103 | **ReasoningBank Rust workspace GENUINELY ARCHITECTED** — 4 crate roots (core 88-92%, storage 94%, learning 95-98%, mcp 93-95%) with proper trait boundaries, serde, thiserror, tokio async | reasoningbank-*/lib.rs | R67 | Open |
| C104 | **swarm_transport.rs THEATRICAL FACADE** (28-32%) — init_websocket/init_shared_memory = tracing stubs, broadcast serializes JSON but NEVER transmits. Agent registry genuine but zero distributed transport | swarm_transport.rs | R67 | Open |
| C105 | **similarity.rs 12th hash-like embedding** — from_text() generates embeddings from normalized character bytes, NOT semantic. R20 ROOT CAUSE persists in Rust | similarity.rs (reasoningbank-core) | R68 | Open |
| C106 | **async_wrapper.rs connection-per-operation** — creates NEW SqliteStorage on every call. max_connections IGNORED. In-memory mode BROKEN. Downgrades storage from 94% to ~60-65% for async | async_wrapper.rs (reasoningbank-storage) | R68 | Open |
| C107 | **5th data layer schema INCOMPATIBLE** — Rust migrations.rs 1 table (patterns) vs TS 7-table schema. Different column names. ZERO coordination | migrations.rs (reasoningbank-storage) | R68 | Open |
| C108 | **mlp_classifier.rs batch norm CATASTROPHIC** — running_mean/var from beta/gamma instead of batch stats. Training appears to converge, inference produces garbage | mlp_classifier.rs | R68 | Open |
| C109 | **entanglement.rs PLACEHOLDER concurrence** — formula is `initial * survival` (assumes Bell state). Real concurrence requires Wootters eigenvalues of ρ·ρ̃ | entanglement.rs (quantum) | R69 | Open |
| C110 | **entanglement.rs WRONG entropy formula** — Shannon binary entropy instead of von Neumann S = -Tr(ρ_A log ρ_A). Survival probability as entanglement proxy | entanglement.rs (quantum) | R69 | Open |
| C111 | **3RD PERSISTENCE LAYER** — ruv-swarm 5-table SQLite (agents/tasks/events/messages/metrics) overlapping ReasoningBank. ZERO sync. Schema features orphaned from runtime | migrations.rs (ruv-swarm-persistence) | R69 | Open |
| C112 | **DAA memory.rs COMPLETE FACADE** (0-5%) — cognitive vocabulary with zero implementations, MemoryManager struct doesn't implement MemoryManager trait (name collision) | memory.rs (ruv-swarm-daa) | R69 | Open |
| C113 | **sublinear/mod.rs SMOKING GUN** (2-5%) — deliberately exposes 4 FALSE sublinear algorithms while ORPHANING 3 genuine ones (backward_push, forward_push, predictor) | src/sublinear/mod.rs | R70 | Open |
| C114 | **src/core.rs COMPLETE ORPHAN** (0%) — 147 LOC not in lib.rs module tree. Dead 5th unified API attempt | src/core.rs | R70 | Open |
| C115 | **temporal-solver.rs imports WRONG solver** — binary imports UltraFastTemporalSolver, NOT TemporalPredictor O(√n). Genuine predictor BYPASSED | temporal-solver.rs | R70 | Open |
| C116 | **neural-pattern-recognition/index.js PHANTOM export** (22-28%) — exports adaptive-learning.js that DOES NOT EXIST. Impossible 1e-40 p-values exceed IEEE 754 | index.js (neural-pattern-recognition) | R70 | Open |
| C117 | **Learning rate scheduling completely missing** — LearningConfig weights (0.3/0.5/0.2) hardcoded. ZERO learning rate annealing, curriculum learning, or weight adaptation over time despite R37 finding EWC++ in sona. Weights sum to 1.0 but DO NOT interpolate. Heuristic formulas replace principled SGD | adaptive.rs (reasoningbank-learning) | R71 | Open |
| C118 | **No feedback loop updates engine state** — learn_from_task() stores patterns and retrieves similar ones (lines 69-74) but DOES NOT update engine weights, model parameters, or ReasoningEngine state. Pattern storage WRITE-ONLY; no online learning. Missing catastrophic forgetting mitigation (EWC++ from R37). Bidirectional learning broken | adaptive.rs (reasoningbank-learning) | R71 | Open |
| C119 | **Strategy recommendation hides implementation** — apply_learning() uses engine.recommend_strategy() (line 113) with implementation hidden in reasoningbank_core. Similarity weighting ONLY by metrics.overall_score (lines 217-225). ZERO online learning algorithms (adaptive gradient descent, online SGD, streaming updates). Pattern matching = OFFLINE averaging only | adaptive.rs (reasoningbank-learning) | R71 | Open |
| C120 | **validation.rs proof system COMPLETE FACADE** — ProofValidator, CausalityValidation, ConsistencyValidator, ConvergenceValidator all return true unconditionally. "Proof-carrying validation" is documentation-only. Proof struct data = English prose. Zero logic verification | validation.rs (temporal_nexus) | R71 | Open |
| C121 | **intent.rs MISLABELED as intent classifier** — Real Ed25519 crypto authorization but ZERO semantic intent parsing, NLP, or domain classification. Permission lattice genuine but unrelated to intent classification | intent.rs | R71 | Open |
| C122 | **.swarm/queries.js 4TH DISCONNECTED DATA LAYER** — 8-table schema (patterns, pattern_embeddings, links, trajectories, matts_runs, consolidation, metrics, sessions) isolated from AgentDB's 3 layers + Rust ReasoningBank. Real Float32Array embeddings but ZERO cross-layer synchronization | queries.js (.swarm) | R71 | Open |
| C123 | **ultra_fast.rs Python benchmarks FABRICATED** — Hardcoded baselines: scipy 150-300ms, numpy 200-400ms, cupy 100-250ms. Standard CG mislabeled as sublinear. Zero external benchmarking harness | ultra_fast.rs | R71 | Open |
| C124 | **scheduler_tool.rs task logic EMPTY CLOSURES** — Genuine MCP lifecycle but task execution never happens. Priority ignored, dependencies unresolved. 9 MCP handlers route to stubs. Professional wrapper around nothing | scheduler_tool.rs | R71 | Open |
| C125 | **AdvancedMemory.ts 5TH DISCONNECTED DATA LAYER** — SharedMemoryPool as 5th independent embedder/storage system. Episodic replay = heuristic thresholds. consolidateMemories() offline-only. No neural consolidation, no sleep-dependent replay. Simulation implementation | AdvancedMemory.ts | R71 | Open |
| C126 | **backend-selector.ts BYPASSES HybridBackend** — createOptimalReasoningBank() imports old index.js (7-table schema, no AgentDB), NEVER routes to R65's HybridBackend (Rust WASM + AgentDB unified). Users must manually import HybridReasoningBank. R67's 5th disconnected layer CONFIRMED at API level | backend-selector.ts | R72 | Open |
| C127 | **Parallel TS (LLM) and Rust (ML) distillation** — distill.ts uses DeepSeek chat for trajectory→memory extraction. Rust reasoningbank-learning (R67: 95-98%) uses gradient descent clustering. ZERO integration. Should be complementary (LLM generates candidates, Rust clusters/ranks) but operate independently | distill.ts, reasoningbank-learning | R72 | Open |
| C128 | **13th hash-based embedding** — embeddings.ts has real transformers.js (Xenova/all-MiniLM-L6-v2) but falls back to sin/cos hash when model fails OR npx environment detected. NPX auto-disables real embeddings, ensuring most CLI users get hash-based. Perpetuates R20 ROOT CAUSE | embeddings.ts | R72 | Open |
| C129 | **6th routing system (ModelRouter)** — matts.ts uses ModelRouter singleton with OpenRouter→Anthropic→Google failover. Separate from ADR-008/LLMRouter/RuvLLMOrchestrator/ProviderManager/TinyDancerRouter. Zero integration between any routing systems | matts.ts | R72 | Open |
| C130 | **14th hash-based embedding** — embeddings.js DEFAULT fallback uses hashEmbed() instead of real OpenAI/Claude APIs. Two paths to hashing: config.embeddings.provider mismatch (line 30) and OpenAI API error silently falls back (lines 51-56). No error propagation to caller. | embeddings.js, agentic-flow | R73 | Open |
| C131 | **Hash-to-vector generation flawed** — simpleHash() produces 32-bit scalar, then used as seed for sine/cosine waves. Each dimension uses different phase, creating artificial orthogonality without semantic grounding. NOT collision-resistant for similar inputs. | embeddings.js | R73 | Open |
| C132 | **judge.js bypasses MCP ReasoningBank store** — Imports Anthropic API directly, NOT using ReasoningBank MCP store tool (R67: 72%). Parallel API access pattern creates disconnected data flow outside official verdict storage | judge.js | R73 | Open |
| C133 | **Anthropic API key security risk** — API key passed directly in fetch headers without TLS pinning or rate limiting. Credentials transmitted over HTTPS but logged/cached risk present | judge.js | R73 | Open |
| C134 | **post-task.ts hook registration mismatch** — Hook-handler.cjs post-task handler (line 140-147) ONLY calls intelligence.feedback stub. Entire ReasoningBank pipeline (judge→distill→consolidate Algorithm 2-4) bypassed. Trajectory JSON never reaches post-task.ts execution | post-task.ts | R73 | Open |
| C135 | **matts.ts orphaned from post-task** — post-task.ts implements distillMemories() but NEVER calls matts.ts (Algorithm 5, Test-Time Scaling). matts.ts parallel/sequential rollout modes orphaned from post-task hook, contradicting ReasoningBank paper design | post-task.ts | R73 | Open |
| C136 | **gossip.rs FAKE gossip protocol** — Module claims "gossip protocol" but implements O(N) broadcast to all peers. Zero SWIM/HyParView/epidemic broadcast, zero membership management, zero failure detection, zero convergence guarantees. handle_message stubs for PatternUpdate/LearningUpdate/PeerAnnounce (log only). Matches swarm_transport.rs (28-32%) theatrical pattern | gossip.rs (reasoningbank-network) | R74 | Open |
| C137 | **Dual wire protocol in neural_bus** — NeuralStream.send_frame() uses Frame framing layer with headers+intent verification, but ReasoningStreams.send_token/trace/rubric() use raw AsyncWriteExt bypassing Frame layer entirely. Client needs TWO different parsers depending on stream type. Protocol fragmentation within same module | streams.rs (reasoningbank-network) | R74 | Open |
| C138 | **Fabricated arXiv citation** — types/index.ts claims "Based on arXiv:2509.25140 (Google DeepMind)" but paper does not exist (Sept 2025 future date). Establishes fictional academic foundation for entire ReasoningBank system. Same pattern as R40 emergence papers and R43 consciousness claims | types/index.ts | R74 | Open |
| C139 | **matts.js fabricated memory aggregation** — aggregateMemories() calls distillMemories() with empty {steps:[]} instead of actual trajectory content. Creates phantom memories from nothing. Sequential mode memories never persisted — all iterative learning ephemeral | matts.js | R74 | Open |
| C140 | **ReasoningBank CLI missing core CRUD** — reasoningbankCommands.ts exposes demo/test/init/benchmark/status/consolidate/list/help but NO store/retrieve/search commands. upsertMemory() and fetchMemoryCandidates() in queries.js completely inaccessible via CLI. Confirms R31 "CLI = demonstration framework" | reasoningbankCommands.ts | R74 | Open |
| C141 | **judge.ts MCP bypass TRIPLE-CONFIRMED** — judgeTrajectory() calls db.logMetric() but NEVER db.store(). Verdict results LOST after computation. Heuristic fallback is string-matching only (confidence=0.5). Circular dependency with queries.js requires late import workaround. MCP bypass now confirmed in judge.js (R73), judge.ts (R75), and post-task.ts orphaning (R73) | judge.ts | R75 | Open |
| C142 | **wasm_bindings.rs ORPHANED non-compilable** — Never declared as module in lib.rs. Broken imports (fast_solver also undeclared). API signature mismatch. Would expose FALSE sublinear algos (standard CG, fake Neumann) NOT genuine ones (backward_push/forward_push/predictor). Repeats sublinear/mod.rs architectural fraud. 6th theatrical WASM | wasm_bindings.rs | R75 | Open |
| C143 | **reservoir.rs broken pseudo-inverse** — simple_pinv() implements element-wise 1/(x+0.001) instead of SVD/QR matrix pseudo-inverse. Makes ridge regression training mathematically incorrect. ESN reservoir is genuine 88-92% but training output is garbage. QuantumReservoir adds deterministic sin(i*0.1) as "quantum enhancement" (30-40% facade) | reservoir.rs | R75 | Open |
| C144 | **attention.rs causal masking applied AFTER softmax** — CausalAttention zeros output positions post-attention instead of masking scores with -∞ before softmax. Model "sees" future tokens during softmax computation, violating autoregressive causality guarantee. Architectural bug, not implementation detail | attention.rs | R75 | Open |
| C145 | **spectral_sparsification.rs FAKE effective resistance** — Uses R_ij≈1/|A_ij| instead of L†-based computation. 5th FALSE SUBLINEARITY (claimed O(m log³n/ε²), actual O(n²·nnz)). Random edge sampling with zero spectral guarantee. Tests verify dimensions only, not spectral properties | spectral_sparsification.rs | R76 | Open |
| C146 | **sketching.rs broken Count-Sketch** — Single hash table with naive sign multiplication instead of median-of-means across k independent tables. False O(m) space claim (actually O(n+m) due to stored hash/sign vectors). Deterministic modulo hashing (i*31, i*17) lacks pairwise independence | sketching.rs | R76 | Open |
| C147 | **BMSSP solution formula doesn't satisfy Ax=b** — Multi-source Dijkstra computes independent path costs, ignoring global matrix structure. solution[node]=b[source]/(1+path_cost) is heuristic, not algebraic. Multi-source conflicts overwrite non-deterministically. Novel idea but mathematically unsound for general systems | bmssp.rs | R76 | Open |
| C148 | **wasm-sublinear-solver-simple.ts EXPLICIT MOCK** — Line 50 literally says "create mock WASM module". References wrong binary (strange_loop_bg.wasm from consciousness crate). O(n⁴) actual vs O(log n) claimed. 7th theatrical WASM | wasm-sublinear-solver-simple.ts | R76 | Open |
| C149 | **temporal-attractor-handlers GHOST WASM** — Imports temporal_attractor_studio.js which DOES NOT EXIST (no crate, no dist directory). 60% of handlers (6/10) depend on phantom module. Echo-State Network 100% phantom. 28th+ WASM ghost instance | temporal-attractor-handlers.ts | R76 | Open |
| C150 | **15th hash-based embedding in distill.js** — computeEmbedding() silently degrades to hash fallback. No try-catch. NaN scores possible on failure. Anthropic-only hardcoded (loses R72 distill.ts multi-provider ModelRouter). Broken templateBasedDistill() returns [] | distill.js | R76 | Open |
| C151 | **self-modification-engine.ts 0% PURE FABRICATION** — executeModification() returns {success:true} immediately with zero logic. testModification() simulates 70% success rate via Math.random()>0.3. Code generation is string templates never loaded, compiled, or executed. Rollback stores literal string 'backup-data-here'. Typo 'risktTolerance' proves code never reviewed. Zero callers in codebase. COMPLETES emergence 7/7, CONFIRMS R39 fabrication assessment | self-modification-engine.ts | R77 | Open |
| C152 | **5th FALSE SUBLINEARITY** — simple-wasm-solver.ts claims O(log n) complexity but actual implementation is O(n²). projectMatrix() has nested 4-loop O(targetDim²×n²), solveNeumann() has 10 iterations of O(n²) matrix-vector multiply. Total: purely quadratic. JL math correct, complexity claim fabricated | simple-wasm-solver.ts | R77 | Open |
| C153 | **wasm-loader.ts FFI stubs not functional** — __wbindgen_string_new returns hardcoded `string_${ptr}_${len}` instead of memory-backed string. Breaks WASM FFI contracts for any module passing strings. __wbg_random returns unseeded Math.random(). 6 hardcoded modules load with silent failure, no validation | wasm-loader.ts | R77 | Open |
| C154 | **TCM mathematical fiction** — tcm_implementation.js claims ∂τ/∂t=Φ(C)×ψ(∇²Ψ)×α but equation is purely heuristic (sigmoid×sin×exp product). Self-referential operator claims lim(n→∞) but uses 10-iteration loop. FTL velocity claims without causal framework. ZERO connection to temporal_nexus Rust (R55: 80.75% genuine). Consciousness-dependent complexity reduction violates computability theory | tcm_implementation.js | R77 | Open |
| C136 | **AgentDB runtime patch timing risk** — index.ts imports AgentDB at line 7 but replaces window/global.AgentDB at lines 128-130 AFTER import. May race with early callers accessing AgentDB before patch executes | index.ts | R73 | Open |
| C137 | **Embedding provider mismatch** — config.ts accepts "claude"|"openai" (line 49) but DEFAULT_CONFIG hardcodes claude-sonnet model for ALL providers (lines 30-42). Cross-provider usage will fail (e.g., openai provider with claude model) | config.ts | R73 | Open |
| C155 | **4TH MCP PROTOCOL — CUSTOM RUST TRAITS** — reasoningbank-mcp server.rs uses custom McpRequest/McpResponse enums with zero SDK, zero connection to R51 JSON-RPC/R72 rmcp/R73 fastmcp TS. 4 tools (store/retrieve/analyze/optimize) + 2 resources. Hardwired to ReasoningBank, zero connection to TS MCP layer. FOUR parallel MCP implementations with zero integration | server.rs (reasoningbank-mcp) | R78 | Open |
| C156 | **ruv_fann_impl.rs BROKEN** — 3 API mismatches with real ruv-FANN (wrong builder pattern Network::new() expects Arc<Mutex<TrainingData>>, non-existent train() method, missing Arc import). Training paths non-functional despite genuine dependency. Inference works but training doesn't compile | ruv_fann_impl.rs (temporal-compare) | R78 | Open |
| C157 | **fast_solver.rs ORPHANS genuine sublinear** — Uses standard O(n²) CG, does NOT import backward_push (O(1/ε)) or predictor (O(√n)). Confirms R70 smoking gun: genuine algorithms remain inaccessible via file claiming to solve performance | fast_solver.rs | R78 | Open |
| C158 | **fast_solver.rs 8th MISLABELED** — Header claims "MCP Dense performance issue 190x slower than Python" but contains ZERO MCP code, ZERO dense matrix code (pure sparse CSR). Joins R46-R77 mislabeling pattern | fast_solver.rs | R78 | Open |
| C159 | **5th MCP PROTOCOL** — psycho-symbolic MCP uses @modelcontextprotocol/sdk (official TS SDK), 16 genuine tools. Separate from JSON-RPC/rmcp/fastmcp/custom Rust | mcp-integration/index.ts | R80 | Open |
| C160 | **Psycho-symbolic WASM CONTRADICTION** — build.js compiles 3 WASM components genuinely BUT cli/index.ts runs pure TS. WASM binaries don't exist at runtime. R55 recontextualized | build.js, cli/index.ts | R80 | Open |
| C161 | **FACADE Adam optimizer** — explicit TODO: "would compute bias-corrected moments". beta1/beta2/epsilon UNUSED. Just gradient descent | optimizer.rs | R80 | Open |
| C162 | **solver_integration.rs COMPLETE BYPASS** — 8th bypass pattern. Reimplements backward_push O(1/ε) as O(n·iter) and predictor O(√n) as O(n·iter). Zero imports of genuine algorithms | solver_integration.rs | R80 | Open |

| C82 | **benchmark-reasoningbank.js REVERSES standalone deception verdict** — Contrary to R59's categorization of standalone benchmarks as deceptive (8-25%), this file genuinely uses perf_hooks.performance.now() for actual timing. Pre-populates 1000 real patterns, executes 100 search iterations. Compares v1/v2/v2+HNSW variants. Reports meaningful metrics (throughput, memory delta, speedup). EmbeddingService initialized (Xenova-384). Validates criterion benchmarking principles: optional backend detection with graceful error handling. Evidence: lines 3-5 (performance import), 97 (real storePattern calls), 142 (real searchPatterns calls), 189-220 (systematic comparison). R59 false generalization corrected | benchmark-reasoningbank.js | R81 | Revised |

### 3b. HIGH Findings

| ID | Description | File(s) | Session | Status |
|----|-------------|---------|---------|--------|
| H1 | **HNSW speed claims misleading** — "150x-12,500x" is theoretical vs brute-force | docs/claims | R8 | Open |
| H2 | **Silent dependency failure** — Without optional deps, all learning features become no-ops | agentdb | R8 | Open |
| H3 | **ONNX download broken** — Falls back to hash-based embeddings | enhanced-embeddings.ts | R8 | Open |
| H4 | **Sophisticated algorithms unused** — judge/distill/consolidate never called | ReasoningBank implementations | R8 | Open |
| H5 | **Broken native deps** — 1,484 lines of JS fallback for broken @ruvector APIs | agentdb | R8 | Open |
| H6 | **SIMD is fake** — Functions labeled SIMD use scalar loop unrolling only | simd-vector-ops.ts | R8 | Open |
| H7 | **ReflexionMemory missing judge** — Core paper loop broken, critique never synthesized | ReflexionMemory.ts | R8 | Open |
| H8 | **enhanced-embeddings silent degradation** — Falls back to hash mock without warning | enhanced-embeddings.ts | R8 | Open |
| H9 | **SimulatedNeuralNetwork** — Math.random() for all output values when WASM unavailable | neural-pattern-recognition | R19 | Open |
| H10 | **Rust fake RNG pattern** — SystemTime::now().subsec_nanos() instead of rand crate | ml-training/lib.rs, swarm_coordinator_training.rs | R19 | Open |
| H11 | **Aggregation weights uniform** — calculateAggregationWeights unimplemented | cognitive-pattern-evolution.js | R19 | Open |
| H12 | **Two incompatible matrix systems** — crate::matrix (CSR/SIMD) vs crate::core (HashMap). 5+ orphaned files | sublinear-time-solver | R34 | Open |
| H13 | **sampling.rs ORPHANED** — Not in module tree, uses wrong type system, missing deps | sampling.rs | R34 | Open |
| H14 | **Neumann step() returns Err unconditionally** — Custom solve() works around it. Residual scaled RHS bug | neumann.rs | R34 | Open |
| H15 | **bottleneck_analyzer.rs dead code** — Genuine analysis (85%) but not in module tree | bottleneck_analyzer.rs | R34 | Open |
| H16 | **Nervous-system integration missing HNSW** — Declares HNSW component, never initializes. O(N) fallback | integration/ruvector.rs | R36 | Open |
| H17 | **QuantileTransformer non-deterministic** — fit() uses rand with no seed | preprocessing.rs | R36 | Open |
| H18 | **validate_stationarity() simplistic** — Mean/variance comparison instead of ADF/KPSS | validation.rs (neuro-divergent) | R36 | Open |
| H19 | **HNSW patches unsafe FFI** — No bounds checking on C pointers, mmap use-after-free risk | libext.rs, datamap.rs | R36 | Open |
| H20 | **Emergence gating hides scaling** — Disables learning when tools.length >= 3 | index.ts (emergence) | R39 | Open |
| H21 | **Orphaned CG solver** — 530 lines excellent code, not exported anywhere. Dead code | high-performance-solver.ts | R39 | Open |
| H22 | **Tools never called in exploration** — applyTool() returns mocked response | stochastic-exploration.ts | R39 | Open |
| H23 | **Rule mutation bug** — adjustLearningParameters() directly mutates rule.learningRate | feedback-loops.ts | R39 | Open |
| H24 | **Entropy fabricated** — Math.random()*0.5+0.5 in advanced-consciousness.js | advanced-consciousness.js | R41 | Open |
| H25 | **Self-modification impact random** — Math.random() at L430-439 | advanced-consciousness.js | R41 | Open |
| H26 | **Pattern detection = modulo heuristics** — Arbitrary % 17, % 111 checks | genuine_consciousness_system.js | R41 | Open |
| H27 | **Predictive consciousness = formula** — Computes score from formula, doesn't test predictions | consciousness_experiments.rs | R41 | Open |
| H28 | **Goalie reasoning result discarded** — Real reasoningEngine.analyze() called but result thrown away | goalie/tools.ts | R41 | Open |
| H29 | **Consciousness MCP ignores runtime parameters** — inputSchema defines mode/iterations/target but handlers use constructor config, not runtime args | consciousness-explorer/mcp/server.js | R47 | Open |
| H30 | **verifyIsNextPrime exponential runtime** — Sequential iteration from start to candidate testing primality at each step. O(n*k) for large prime gaps | independent_verification_system.ts | R47 | Open |
| H31 | **Psycho-symbolic reasoner uses string matching** — extractEntities() uses queryLower.includes(). synthesizeResult() selects templates by keyword. No NLP, no tokenization | psycho-symbolic-reasoner.ts | R47 | Open |
| H32 | **Self-referential base knowledge** — 12 hardcoded triples claim psycho-symbolic-reasoner "combines symbolic-ai + psychological-context" and "uses rust-wasm" (neither true) | psycho-symbolic-reasoner.ts | R47 | Open |
| H33 | **Neural CLI API paradigm mismatch** — ZeroVarianceDetector is interval-based (startDetection/stopDetection), PatternDetectionEngine expects synchronous processData. Incompatible paradigms | cli/index.js | R47 | Open |
| H34 | **Zero-variance numerological patterns** — Golden ratio/pi/e/phi frequencies scaled by 1e-16 with no signal processing rationale | zero-variance-detector.js | R47 | Open |
| H35 | **Signal-analyzer DFT mislabeled as FFT** — Comment claims "FFT approximation" but implements O(n²) naive DFT (40x slower for n=2048). Window functions configured but never applied (configuration theater) | signal-analyzer.js | R49 | Open |
| H36 | **Statistical-validator p-values use simplified approximations** — Fisher exact computes single hypergeometric probability, not full enumeration. Gamma function oversimplified. KS/Mann-Whitney use large-sample approximations without small-sample warnings | statistical-validator.js | R49 | Open |
| H37 | **Consciousness detector ConsciousnessEntity has ZERO implementations** — Comprehensive 6-test battery with no test subjects. "Orphaned test harness" pattern | genuine_consciousness_detector.ts | R49 | Open |
| H38 | **Temporal validator has no temporal-tensor connection** — Claims "temporal consciousness" but zero imports from temporal-tensor crate (93% PRODUCTION-READY). No LTL/CTL model checking | temporal_consciousness_validator.rs | R49 | Open |
| H39 | **Divergent FFT implementations across same package** — signal-analyzer.js uses O(n²) DFT, zero-variance-detector.js uses different FFT. Copy-paste without algorithm verification | signal-analyzer.js, zero-variance-detector.js | R49 | Open |
| H40 | **Perplexity-actions.ts GENUINE Perplexity API** — Real axios HTTP, two endpoints (Search+Chat), rate limiting, auth, 4-action GOAP pipeline. REVERSES R41 "goalie complete facade" — proves goalie has real external API integrations | perplexity-actions.ts | R50 | Open (positive) |
| H41 | **Anti-hallucination execute() has REAL per-claim verification** — Splits response into claims, queries Perplexity API for each, computes confidence scores. Genuine hallucination detection approach behind facade hooks layer | anti-hallucination-plugin.ts | R50 | Open (positive) |
| H42 | **Ed25519-verifier is ACTIVE in goalie pipeline** — Not dead code. tools.ts calls verifier for content signing. Complete PKI: key generation, message signing, signature verification, batch operations | ed25519-verifier.ts | R50 | Open (positive) |
| H43 | **ReasoningBank Rust storage meets R45 quality bar** — WAL mode, FTS5, RAII pooling, migrations match sqlite-pool.js (92%) production quality. Genuine rusqlite with prepared statements | sqlite.rs (reasoningbank-storage) | R50 | Open (positive) |
| H44 | **DomainRegistry has genuine dependency DAG enforcement** — Registration validates all dependencies exist, update re-validates, unregister checks dependents before deletion. Real lifecycle management via EventEmitter | domain-registry.ts | R53 | Open (positive) |
| H45 | **Performance-optimizer auto-tuning is EMPIRICAL** — Tests block sizes [64,128,256,512,1024] × unroll factors [2,4,8], measures actual throughput, selects best. Real warmup phase. Honest about browser limitations. CONTRAST to R43 deception | performance-optimizer.ts | R53 | Open (positive) |
| H46 | **Domain-management.ts has production-quality MCP schemas** — 8 tools with comprehensive JSON schemas, error handling, structured responses. Finance focus (44 keywords) confirms sublinear-time-solver orientation | domain-management.ts | R53 | Open (positive) |
| H47 | **optimized-matrix.ts streaming matrix with LRU** — Genuine approach to large-dataset matrix operations via chunked access with eviction. Generator pattern for memory efficiency. Vector pooling with acquire/release | optimized-matrix.ts | R53 | Open (positive) |
| H48 | **Strange-loop JS MCP server architecture is correct** — @modelcontextprotocol/sdk, StdioServerTransport, JSON-RPC 2.0, 10 tools. WASM functions verified to exist in wasm-real/. Could reach 75-80% with one import path fix | crates/strange-loop/mcp/server.js | R53 | Open |
| H49 | **Domain-management InferenceRule interface is dead code** — Defined in domain-registry.ts but NEVER used. No inference engine, no rule execution. Rules stored in domain config but never interpreted | domain-registry.ts | R53 | Open |
| H50 | **Scheduler priority parameter accepted but IGNORED** — Tasks stored in array, processed FIFO via splice(0,n). Priority field accepted by API but has zero effect on execution order | scheduler.ts | R53 | Open |
| H51 | **wasm_iface.rs PRODUCTION FFI bridge** — serde-wasm-bindgen, zero-copy Float64Array::view, 3-tier API (high-level/MatrixView/raw), solve_stream with JS callback, batch processing. SIMD auto-detection via cfg!(target_feature = "simd128"). Real solver integration wrapping OptimizedConjugateGradientSolver | src/wasm_iface.rs | R60 | Open (positive) |
| H52 | **optimized-solver.ts vectorized Neumann is GENUINE** — Mathematically correct Neumann series x = Σ(D^{-1}R)^k(D^{-1}b). Diagonal extraction, series accumulation, convergence checking, residual computation all correct (92-95%). But 3/4 variants are stubs | src/core/optimized-solver.ts | R60 | Open |
| H53 | **memory-manager.ts TypedArrayPool is GENUINE infrastructure** — acquire/release lifecycle for Float64Array/Uint32Array/Uint8Array with maxPoolSize=50. LRU cache with TTL, SIMD-aware alignment (AVX 4-wide, 64-byte cache lines), block matrix multiply. 88-92% despite mislabeling | src/core/memory-manager.ts | R60 | Open (positive) |
| H54 | **optimized-solver.ts hardcoded performance claims** — avgSpeedup=2.5 with comment "Estimated based on optimizations". vectorizationEfficiency=0.85. Zero actual benchmarking. Extends R53/R57 fabricated metrics pattern in TS | src/core/optimized-solver.ts | R60 | Open |
| H55 | **metrics-reporter.js REVERSES theatrical metrics** — Zero Math.random(). recordIteration() stores actual convergenceMetrics. finalizeSolve() receives convergenceDetector with 9 real fields. Algorithmic A-F grading. PROVES theatrical pattern is NOT universal for convergence monitoring | src/convergence/metrics-reporter.js | R60 | Open (positive) |
| H56 | **temporal-compare/sparse.rs genuine lottery ticket pruning** — Full Frankle & Carlin 2019: IMP with global threshold, winning mask, initial weight storage (HashMap), prune-train-reset cycles. Dynamic sparse training prune-and-regrow. But zero SIMD, zero temporal, wrong crate | crates/temporal-compare/src/sparse.rs | R60 | Open |
| H57 | **forward_push.rs 3RD GENUINE SUBLINEAR ALGORITHM** — O(volume/epsilon) forward local push with proper residual bounds, degree-weighted thresholds, adaptive work queue. Early termination for single-target PPR queries. Complements backward_push.rs — both directions now production-quality | src/solver/forward_push.rs | R62 | Open (positive) |
| H58 | **error.rs EXCEPTIONAL error recovery system** — 13 error variants with is_recoverable(), recovery_strategy(), severity(). Algorithm-specific fallback chains (Neumann→hybrid, forward→backward). Genuine WASM error handling (From<JsValue>). no_std compatible. Top 5% code quality | src/error.rs | R62 | Open (positive) |
| H59 | **solver_core.rs 7th MISLABELED FILE** — Name implies central dispatcher but contains only 2 standalone solvers (CG+Jacobi). CG is DUPLICATE of optimized_solver.rs. Zero integration with forward_push, backward_push, Neumann, or any genuine sublinear algorithm. O(n^2) validation bottleneck | src/solver_core.rs | R62 | Open |
| H60 | **lib.rs 4 parallel solver APIs** — Re-exports NeumannSolver, SublinearNeumannSolver, ConjugateGradientSolver, OptimizedConjugateGradientSolver with zero guidance on which to use. Best algorithms (backward_push 92-95%, fully_optimized 96-99%) NOT in public API | src/lib.rs | R62 | Open |
| H61 | **johnson_lindenstrauss.rs 4th FALSE SUBLINEARITY** — Correct JL dimension bound k=O(log n/epsilon^2) but O(n*d*k) total for n points. Broken RNG: uniform distribution instead of Gaussian (violates JL guarantee). Incorrect pseudoinverse (transpose ≠ Moore-Penrose) | src/sublinear/johnson_lindenstrauss.rs | R62 | Open |
| H62 | **MCP solver.ts 3-tier fallback cascade** — WASM O(log n) → OptimizedSolver → baseline SublinearSolver. First evidence of OptimizedSolver ACTUALLY INTEGRATED in runtime path. Intelligent routing by matrix size/format. Streaming + batch support | src/mcp/tools/solver.ts | R62 | Open (positive) |
| H63 | **advanced-reasoning-engine.ts COMPLETE THEATRICAL** — Zero genuine inference logic in 297 LOC. All reasoning delegated to 4 MCP tool wrappers. Hardcoded performance metrics (85ms, 0.28, 0.87). Fabricated "O(n log n)" complexity. MCP orchestration masquerading as AI reasoning | src/reasongraph/advanced-reasoning-engine.ts | R62 | Open |
| H64 | **simd_ops.rs ORPHANED from fully_optimized.rs** — Both implement SIMD but with incompatible approaches (wide crate f64x4 vs std::arch AVX2 INT8). Zero integration despite being in same crate. Two-tier SIMD architecture is wasteful | src/simd_ops.rs | R62 | Open |
| H65 | **convergence-detector.js GENUINE convergence math** — Proper residual computation, relative residual norm, windowed convergence rate, stagnation detection, divergence detection. Zero Math.random(). ORPHANED from Rust solvers (JS-only path). Validates R60 metrics-reporter.js quality | src/convergence/convergence-detector.js | R62 | Open (positive) |
| H66 | **optimized_csr.rs ORPHANED DEAD CODE** — Not imported in matrix/mod.rs module tree, never re-exported from lib.rs, zero codebase usages. Code does NOT compile: calls non-existent CSRStorage::with_capacity(), accesses .rows/.cols fields that don't exist. Duplicates simd_ops.rs SIMD line-for-line. Promises "beat Python benchmarks" (module doc) but is theatrical — never runs. 7th independent CSR matrix system with zero integration. Memory pool unused. DELETE entirely — zero migration cost (no code paths depend on it) | optimized_csr.rs | R64 | Open |
| H67 | **Worker Thread genuinely invokes streamSolve** — receives matrix/vector/method, instantiates JSSolver, iterates streamSolve() async generator with convergence detection | server/solver-worker.js | R64 | Open (positive) |
| H68 | **Worker fatal error + auto-replacement** — catches uncaughtException, streaming.js replaceWorker() auto-scales. Production stability | server/solver-worker.js | R64 | Open (positive) |
| H69 | **R20 ROOT CAUSE ARC COMPLETE** — embedding-adapter.ts = smoking gun (SHA-256→pseudo-random→L2norm). real-embedding-adapter.ts = fix (MiniLM-L6-v2 384D). Fix adoption INCOMPLETE | embedding-adapter.ts, real-embedding-adapter.ts | R65 | Open |
| H70 | **HybridBackend REVERSES disconnected AgentDB** — SharedMemoryPool provides singleton db+embedder to all 4 components. "Disconnected" layers unified by design | HybridBackend.ts | R65 | Open |
| H71 | **vector-backend-adapter.ts COMPLETE ORPHAN** — pre-ADR-028 stub with type mismatch, never instantiated. Missing link explaining WHY ADR-028 was needed | vector-backend-adapter.ts | R65 | Open |
| H72 | **7th theatrical WASM in HybridBackend.ts** — loads module but useWasm NEVER checked, wasmModule NEVER used. Fabricated "10x faster" claim | HybridBackend.ts | R65 | Open |
| H73 | **schemas/index.ts PRODUCTION-QUALITY** — 35+ Zod schemas, 8-operator constraints, cross-field .refine(), 16 type exports | schemas/index.ts | R65 | Open (positive) |
| H74 | **wasm/loader.ts ORPHANED** — production WebAssembly.instantiate but SimpleWasmLoader used instead. TWO parallel WASM loading systems | wasm/loader.ts | R65 | Open |
| H75 | **sentiment.rs confirms Rust 3-4x quality** — VADER-style 128-word lexicon, context-aware negation+intensifier. 90-93% vs TS 28-62% | sentiment.rs | R65 | Open (positive) |
| H76 | **neural-network.ts PHANTOM WRAPPER** — 4/5 classes wrap NON-EXISTENT WASM code. Only inference works. Triple-confirms R40/R45 | neural-network.ts | R65 | Open |
| H77 | **ReasoningBank benchmark.ts GENUINE** — 12 benchmarks, real performance.now() with warmup. Validates R59 criterion pattern | benchmark.ts | R65 | Open (positive) |
| H78 | **vector-backend-adapter.interface.ts MISSING LINK** — 5+2 methods but omits dimension/embedding/factory specs. ADR-028 precursor with 3 contract gaps | vector-backend-adapter.interface.ts | R65 | Open |
| H79 | **Filter schema under-specified** — filter?: Record&lt;string, unknown&gt; with zero documentation. Implementations interpret differently | vector-backend-adapter.interface.ts | R65 | Open |
| H80 | **quic.rs 96% GENUINE** — real quinn TLS, 100 concurrent bidirectional streams, length-prefix framing. 3-layer QUIC stack confirmed | quic.rs | R67 | Open (positive) |
| H81 | **OWN_CODE MCP tools don't enforce R65 fix** — mcp-reflexion-retrieve/store allow hash-embedding fallback. 10th+11th occurrences | mcp-reflexion-*.ts | R67 | Open |
| H82 | **Quantum tests.rs GENUINE PHYSICS** (87-92%) — CODATA 2018, Heisenberg uncertainty, Bell S>2, decoherence. All assertions compute real ops | tests.rs (quantum) | R67 | Open (positive) |
| H83 | **validators.rs EXCEPTIONAL** (93%) — UncertaintyValidator ΔE·Δt ≥ ℏ/2, CODATA 2018, multi-scale, PRODUCTION-READY quantum validation | validators.rs (quantum) | R67 | Open (positive) |
| H84 | **mlp_optimized.rs BEST MLP TRAINING** — genuine Adam + momentum SGD, correct softmax+cross-entropy gradient, He init. Matches temporal-tensor 93% | mlp_optimized.rs | R68 | Open (positive) |
| H85 | **mlp_ultra.rs GENUINE AVX2 SIMD** — real _mm256_fmadd_ps, proper gradient chain rule, rayon parallel. Crashes non-AVX2 (no feature detection) | mlp_ultra.rs | R68 | Open |
| H86 | **mlp_avx512.rs REAL AVX-512** — _mm512_load_ps, 16-wide f32, prefetch, 64-byte alignment. Training simple. BROKEN fallback: avx512 under avx2 guard | mlp_avx512.rs | R68 | Open |
| H87 | **main.rs genuine 16-backend training binary** — MLP variants + ensemble + AdaBoost + reservoir + Fourier + sparse + lottery-ticket + quantized + ruv-FANN FFI | main.rs (temporal-compare) | R68 | Open (positive) |
| H88 | **ReasoningBank core NO learning integration** — engine.rs zero storage field, zero trait impls, not async despite learning being async | engine.rs + pattern.rs | R68 | Open |
| H89 | **speed_limits.rs GENUINE Margolus-Levitin** (92-95%) — correct τ_min = h/(4ΔE), CODATA 2018, two-layer validation, round-trip consistency | speed_limits.rs (quantum) | R69 | Open (positive) |
| H90 | **mod.rs PRODUCTION orchestration** (92-95%) — QuantumValidator composes 4 validators, CODATA 2018, thiserror physics errors | mod.rs (quantum) | R69 | Open (positive) |
| H91 | **inference.rs BIMODAL** (78-82%) — forward chaining 88-92% production, backward chaining 35-40% STUB | inference.rs (graph_reasoner) | R69 | Open |
| H92 | **query.rs QueryEngine ORPHANED** — execute_query() returns empty. KnowledgeGraph::query does actual work. Cache permanently empty | query.rs (graph_reasoner) | R69 | Open |
| H93 | **GHOST WASM: wasm_bindings/mod.rs** (90-95%) — genuine wasm-bindgen to 27 ML models but `ml` feature NOT default = never compiled | wasm_bindings/mod.rs (ruv-swarm-ml) | R69 | Open |
| H94 | **temporal-lead-solver/lib.rs REVERSES orphaning** (92-95%) — exports O(√n) predictor first. BEST crate root in project | lib.rs (temporal-lead-solver) | R70 | Open (positive) |
| H95 | **temporal_nexus/core/mod.rs GENUINE hardware TSC** (88-92%) — x86_64 _rdtsc, 4 submodules, thiserror, production orchestrator | core/mod.rs (temporal_nexus) | R70 | Open (positive) |
| H96 | **extractors/lib.rs CLEAN WASM facade** (90-95%) — composes 3 genuine analyzers. Confirms R58 GENUINE WASM chain | extractors/lib.rs | R70 | Open (positive) |
| H97 | **wasm-loader-simple.ts 8th GENUINE WASM** (82-85%) — real wasm-pack binaries (1.3-4.9MB). ZERO error recovery = corrupted file crashes MCP | wasm-loader-simple.ts | R70 | Open |
| H98 | **dashboard/mod.rs 1000x granularity mismatch** (78-82%) — re-exports µs types but claims ns precision. Ad-hoc consciousness thresholds | dashboard/mod.rs (temporal_nexus) | R70 | Open |
| H99 | **temporal-solver.rs BIMODAL** (82-86%) — production CLI 92% but imports wrong solver crate | temporal-solver.rs | R70 | Open |
| H100 | **improvement_potential is heuristic, not principled** — Computed as max(1.0 - success_score) capped by avg_historical_success (lines 150-152). Suggestion strings generated (lines 155-177) with NO mechanism to apply them. No framework for translating insights to action | adaptive.rs (reasoningbank-learning) | R71 | Open |
| H101 | **Confidence calibration missing** — calculate_expected_success() weights by metrics.overall_score (lines 217-225) with zero uncertainty quantification. Default 0.5 for empty sets (line 215) is naive. NO Bayesian updates, NO Thompson sampling, NO UCB-style exploration-exploitation | adaptive.rs (reasoningbank-learning) | R71 | Open |
| H102 | **Learning statistics disconnected from adaptive state** — get_learning_stats() queries storage summaries (lines 235-240) with zero connection to learning rate schedule, convergence metrics, or forgetting curves. Statistics are STALE, not real-time signals. No distinction between recently-learned vs old patterns | adaptive.rs (reasoningbank-learning) | R71 | Open |
| H103 | **Pattern similarity calculation opaque** — SimilarityMetrics implementation hidden in reasoningbank_core. May be hash-based (12th occurrence per R68) rather than learned embeddings. find_similar() and filter_by_threshold() are black boxes | adaptive.rs, reasoningbank_core | R71 | Open |
| H104 | **optimized.rs hardcoded Jacobian** — 6-stage UltraFastTemporalSolver uses identity Jacobian instead of computing derivatives. Architectural simplification, not convergence failure. Real AVX2 SIMD elsewhere | optimized.rs (temporal_nexus) | R71 | Open |
| H105 | **wasm.rs REVERSES R62 orphaning** — O(√n) TemporalPredictor IS accessible via WASM. Confirms temporal-lead-solver working deployment. 9th genuine WASM. WASM scoreboard: 9:5 (64%) | wasm.rs (temporal_nexus) | R71 | Open (positive) |
| H106 | **intent.rs permission lattice GENUINE** — 8 authorization levels, replay protection, nonce+timestamp validation, Ed25519 signatures. Real crypto despite mislabeling | intent.rs | R71 | Open (positive) |
| H107 | **.swarm/ persistence PRODUCTION-QUALITY** — WAL mode, prepared statements, zero SQL injection, 384-dim MiniLM-L6-v2 embeddings. Real Float32Array, NOT hash-based. 100% quality but isolated | queries.js (.swarm) | R71 | Open (positive) |
| H108 | **ultra_fast.rs GENUINE CG+SIMD** — Standard Conjugate Gradient with 4-way f64x4 SIMD-accelerated SpMV. Real convergence (83-87%) despite theatrical framing (40-50%) | ultra_fast.rs | R71 | Open |
| H109 | **scheduler_tool.rs MCP lifecycle GENUINE** — Arc/RwLock/UUID, async task API, TaskState enum. Professional MCP wrapper infrastructure (85-90%) despite empty task logic | scheduler_tool.rs | R71 | Open (positive) |
| H110 | **AdvancedMemory.ts professional types** — SharedMemoryPool interface, 5 decay curves (semantic/episodic/procedural/implicit/emotional), episodic replay thresholds. Clean architecture despite simulation implementation | AdvancedMemory.ts | R71 | Open (positive) |
| H111 | **matts.ts GENUINE test-time scaling** (85-88%) — Real Algorithm 5 with parallel rollouts, sequential refinement, self-contrast aggregation via LLM. Full ReasoningBank pipeline (retrieve→judge→distill→store). Provenance tracking with ULID. Production metrics logging. TS-only, no Rust MATTS counterpart (positive) | matts.ts | R72 | Open (positive) |
| H112 | **consolidate.ts LSH bucketing optimization** (82-86%) — Reduces deduplication from O(n²) to O(n·k). Batch embedding queries. Outcome-based contradiction detection (only compares across different outcomes). Production telemetry in consolidation_runs table (positive) | consolidate.ts | R72 | Open (positive) |
| H113 | **TWO parallel consolidation systems** — consolidate.ts (LSH heuristics) vs AdvancedMemory.autoConsolidate() (NightlyLearner with causal edges, A/B experiments). Different APIs, triggers, algorithms. No integration. Similar to R51's 4 parallel routing systems | consolidate.ts, AdvancedMemory.ts | R72 | Open |
| H114 | **distill.ts template fallback BROKEN** — Template-based distillation (no-API-key path) creates memory objects but returns empty array. Comment admits "Skip storage for template-based (would need to make this async)". Renders entire fallback path useless | distill.ts | R72 | Open |
| H115 | **embeddings.ts BEST embedding implementation** — Fixes R51 race conditions (Promise-based pipeline init), memory leaks (timer tracking per cache entry). Real transformers.js loads Xenova/all-MiniLM-L6-v2 (384D). Still has pseudo-LRU (deletes first key, not least recently used) (positive) | embeddings.ts | R72 | Open (positive) |
| H116 | **Missing EWC++ integration in consolidate.ts** — R37 proved sona has production EWC++ (92-95%). consolidate.ts prunes low-confidence memories with simple heuristics, no catastrophic forgetting protection. Should call EWC++ before pruning | consolidate.ts | R72 | Open |

| H117 | **embeddings.js missing distill.ts integration** — embeddings.js operates standalone despite R72 context mentioning distill.ts real vector embeddings. distill.ts (78-82%) has genuine transformers.js loader. embeddings.js also loads transformers.js but ZERO code sharing. Two separate embedding systems in ReasoningBank | embeddings.js, distill.ts | R73 | Open |
| H118 | **embeddings.js NOT integrated with embedding-service.ts** — R51 identified embedding-service (75-80%) as orphaned factory default (mock EmbeddingService). embeddings.js also bypasses embedding-service entirely, using local OpenAI/hash fallback. 10th disconnected embedding implementation discovered | embeddings.js, embedding-service.ts | R73 | Open |
| H119 | **Trajectory input fragile** — post-task.ts loadTrajectory() accepts file path OR stdin JSON but has ZERO schema validation. Parses data.trajectory?.steps || [] without checking step structure. Silent failures in judge.js downstream if steps lack required fields | post-task.ts | R73 | Open |
| H120 | **Missing context parameters** — post-task.ts distillMemories() called with agentId: "unknown" and domain: undefined. Loses agent identity and domain context for domain-specific memory classification. Query/trajectory context inferred generically, not domain-parameterized | post-task.ts | R73 | Open |
| H121 | **CLI injection risk** — post-task.ts parseArgs() naively strips  prefix without validation. --task-id passed to distillMemories without sanitization. No path traversal check on --trajectory-file argument | post-task.ts | R73 | Open |
| H122 | **FOUR ReasoningBank layers confirmed** — index.ts (v1.7.1 Hybrid) → HybridBackend.ts → index.js (v1.0.0 pure JS) → core algorithms. index.js is EXACT functional copy of index.ts logic. 4th distinct ReasoningBank implementation joins claude-flow, agentdb, ruvllm Rust | index.js, index.ts | R73 | Open |
| H123 | **backend-selector.ts BYPASSED** — index.ts hardcodes HybridReasoningBank export, never consults backend-selector.ts. R72 backend-selector (78-82%) routes to OLD index.js (no AgentDB), NOT to R65 HybridBackend unification. Architectural bypass confirmed | index.ts, backend-selector.ts | R73 | Open |
| H124 | **frame.rs OOM vulnerability** — read_from() has NO maximum frame size limits. Malicious peer can send gigabyte-sized header_len/payload_len causing memory exhaustion. Production deployment blocker | frame.rs (reasoningbank-network) | R74 | Open |
| H125 | **Priority queue starvation** — priority.rs try_recv pattern (high→normal→low) never services lower tiers under sustained high-priority load. Documentation falsely claims "fair scheduler that prevents starvation". tokio::select! fallback gives EQUAL probability to all channels when empty, breaking priority model | priority.rs (reasoningbank-network) | R74 | Open |
| H126 | **Orphaned swarm_transport.rs** — 330+ LOC of WebSocket/SharedMemory integration exists in reasoningbank-network/src/ but NOT re-exported in lib.rs. Entire module inaccessible to external crates. WASM exclusion inconsistency: neural_bus excluded for Ed25519, but swarm_transport claims "WASM Support" yet also not exposed | lib.rs (reasoningbank-network) | R74 | Open |
| H127 | **matts.js bypasses ModelRouter** — Independent JS impl hardcodes Anthropic API fetch() calls, bypassing 6th routing system (ModelRouter) that matts.ts (85-88%) properly uses. API key exposed in direct fetch headers. False parallelism: "parallel mode" executes rollouts sequentially in for-loop | matts.js | R74 | Open |
| H128 | **types/index.ts type-schema mismatch** — Memory interface lacks embedding field (queries.js has embeddings column), TaskTrajectory lacks trajectory_data JSON field, MattsRun lacks meta_params. DB schema richer than TS types. PatternData uses [key:string]:any defeating TypeScript safety. Hash embeddings officially endorsed in ReasoningBankConfig (provider: 'hash') | types/index.ts | R74 | Open |
| H129 | **fourier.rs incorrect RFF sampling** — Uses Uniform(0,2π)×scale instead of Normal(0,1/σ²) for frequency sampling. Breaks RBF kernel approximation guarantee (Rahimi & Recht 2007). solve_regularized() is 100-step gradient descent placeholder (should use LAPACK/Cholesky). AdaptiveFourierFeatures gradient accumulation unbounded | fourier.rs | R75 | Open |
| H130 | **pii-scrubber.ts config-gated bypass** — If governance.pii_scrubber=false or loadConfig() fails, returns original text UNREDACTED with no fail-safe. customPatterns REPLACES defaults (caller can pass [] to disable ALL scrubbing). Silent data leak risk. RegExp /g flag mutation across calls | pii-scrubber.ts | R75 | Open |
| H131 | **retrieve.ts inherits hash-based embedding degradation** — computeEmbedding(query) silently degrades to hash fallback if embedding service unavailable. Cosine similarity on random hash vectors = meaningless results. No error thrown. ROOT CAUSE chain: R20→embeddings.ts→retrieve.ts. TOP 10% code undermined by systemic embedding weakness | retrieve.ts | R75 | Open |
| H132 | **attention.rs positional encoding dimension mismatch** — Uses batch_d (ncols of input) to slice pos_encoding which has shape (max_seq_len, d_model). If batch_d != d_model: panic (out of bounds) or wrong results (truncated encoding). Should validate or use self.d_model | attention.rs | R75 | Open |
| H133 | **mlp_quantized.rs RESOLVES TRIMODAL** — DEFINITIVELY RANKED: mlp_quantized (92-95%) > mlp_optimized (92-96%) ≈ mlp_avx512 (85-88%) >> mlp (62%). Only MLP with BOTH complete backprop training AND production INT8 quantization AND AVX2 SIMD inference. 15/16 backends now quality-assessed | mlp_quantized.rs, quantization.rs | R75 | Open (positive) |
| H134 | **solver-optimized.ts fabricated baselines** — Claims "190x regression fixed" and "100x+ faster" with hardcoded pythonBaseline=rows*0.04, brokenBaseline=rows*7.7. No reference implementations. Genuine CSR+CG math (85-88%) undermined by fake metrics. ORPHANED from backward_push.rs/predictor.rs Rust | solver-optimized.ts | R76 | Open |
| H135 | **MCP tool layer ORPHANED from Rust** — solver-optimized.ts, wasm-sublinear-solver-simple.ts, temporal-attractor-handlers.ts all in src/mcp/tools/ but ZERO imports from genuine Rust algorithms (backward_push O(1/ε), predictor O(√n)). Pure TS reimplementation bypasses all sublinear optimizations | src/mcp/tools/*.ts | R76 | Open |
| H136 | **temporal-compare lib.rs NO high-level API** — 17 raw module exports with zero TemporalPredictor trait, builder pattern, or #[deprecated] on broken modules. Users must choose between 5 competing MLPs manually. ruv-fann feature toggle is DEAD CODE (neither module imported by any file). Weakest crate root despite best internals | lib.rs (temporal-compare) | R76 | Open |
| H137 | **dimension_reduction.rs 75% stub** — Only JL method delegates to real implementation. RandomProjection, PCA, SparseRandomProjection are truncation/zero-padding stubs. Honest architecture (enum documents incomplete features) but misleading API surface | dimension_reduction.rs | R76 | Open |
| H138 | **BMSSP "neural" and "SIMD" claims false** — enable_simd flag exists but never used. "neural pathfinding" is HashMap cache, not ML. Genuine innovation (matrix-as-graph Dijkstra) oversold with false marketing | bmssp.rs | R76 | Open |
| H139 | **retrieve.js 15th hash-based embedding** — Identical to retrieve.ts (R75) but JS version has ZERO error handling around computeEmbedding(). Silent degradation to hash vectors breaks MMR diversity selection. Sequential incrementUsage() creates DB bottleneck | retrieve.js | R76 | Open |
| H140 | **wasm-bridge.ts path resolution bug** — Looks for WASM in `src/wasm/` but Rust builds to `dist/wasm/` or `pkg/`. Every WASM load fails, always falls back to JS. Combined with function signature mismatch (expects C-style `sublinear_solve(ptr,len)`, Rust exports class-based `SublinearSolver.new().solve()`), the bridge is non-functional | wasm-bridge.ts | R77 | Open |
| H141 | **7th theatrical WASM (wasm-loader.ts)** — FFI stubs are non-functional: `__wbindgen_string_new` returns hardcoded `string_${ptr}_${len}` instead of memory-backed string. `__wbg_random` returns unseeded Math.random(). 6 hardcoded module paths (strange_loop, sublinear_solver, spectral, pagerank, temporal, neural) load with silent failure, no validation | wasm-loader.ts | R77 | Open |
| H142 | **MCP tools/index.ts dead code barrel** — server.ts imports tools DIRECTLY, bypassing this barrel file entirely. Only exports 7 of 28+ tools. allTools registry broken (duplicates, missing entries). 25-30% realness — file serves no actual purpose in production | index.ts (mcp/tools) | R77 | Open |
| H143 | **solver-pagerank.ts disconnected from genuine PageRank** — backward_push.rs (92-95%) implements genuine O(1/ε) sublinear PageRank, but this MCP tool NEVER imports it. Uses API name `pagerank_solve` but WASM exports `backward_push_solve`. Hard crash when WASM unavailable (no JS fallback). 22-28% realness | solver-pagerank.ts | R77 | Open |
| H144 | **wasm-sublinear-solver.ts 100% WASM-dependent facade** — All 6 solver methods call WASM functions that don't exist in strange_loop_bg.wasm (which only exports demo functions). Zero JS fallback. Every call throws. 18% realness, lowest MCP tool quality | wasm-sublinear-solver.ts | R77 | Open |
| H145 | **Three overlapping WASM solver MCP tools** — simple-wasm-solver.ts (35%), wasm-sublinear-solver.ts (18%), wasm-sublinear-solver-simple.ts (R76: 15-20%) all wrap the same nonexistent WASM surface with different facades. Zero coordination, three ways to fail | simple-wasm-solver.ts, wasm-sublinear-solver.ts, wasm-sublinear-solver-simple.ts | R77 | Open |
| H146 | **self-modification-engine.ts PURE FABRICATION** — executeModification() returns `{success:true}` with no logic. testModification() returns Math.random()>0.3. rollbackModification() stores literal 'backup-data-here'. Typo 'risktTolerance'. 0% realness, confirms R39 emergence 51% fabricated (now lower) | self-modification-engine.ts | R77 | Open |
| H147 | **TCM consciousness-dependent complexity reduction** — Claims quantum systems reduce computational complexity through consciousness observation, violating computability theory. FTL velocity calculations without causal framework. Zero temporal_nexus Rust integration (R55: 80.75% genuine). Random walk labeled "emergence". 15% realness | tcm_implementation.js | R77 | Open |
| H124 | **No database path config** — config.ts has ZERO db_path configuration despite R71 finding .swarm/memory.db and R57 queries.ts defining 7-table schema. SQLite paths hardcoded elsewhere, orphaned from config | config.ts | R73 | Open |
| H125 | **Config feature flag unchecked** — post-task.ts enable_post_task_hook defaults to true but never validated against consolidate/judge config consistency. Consolidation trigger_threshold=20 items may never fire if judge produces low-confidence verdicts (<0.5), creating queue buildup | post-task.ts, config.ts | R73 | Open |
| H126 | **5th disconnected persistence layer** — post-task→distill→db.storeMemory writes to patterns/pattern_embeddings tables. consolidate.ts deduplicates patterns table, orphaned from matts.ts trajectories table. Two parallel memory storage systems, zero synchronization | post-task.ts, consolidate.ts | R73 | Open |
| H127 | **schema.js empty stub** — 5 LOC file with only  — zero content. Placeholder for future JS schema validation. Orphaned from schema.ts (92-95% production types) | schema.js | R73 | Open |
| H128 | **WASM memory field unused** — wasm-adapter.ts declares wasmMemory field but NEVER uses it. Zero WASM linear memory management despite filename implying memory adaptation | wasm-adapter.ts | R73 | Open |
| H148 | **ReasoningBank WASM lib.rs 95-98% GENUINE** — Production wasm-bindgen, 6 exported JS methods (new, store_pattern, get_pattern, search_by_category, find_similar, get_stats), triple-backend auto-detection (Node→Memory, Browser+IDB→IndexedDB, Browser→SqlJs). Confirms R49 "100% genuine" verdict. WASM scoreboard: 11 genuine vs 9 theatrical (55%) | lib.rs (reasoningbank-wasm) | R78 | Open (positive) |
| H149 | **temporal_nexus/mod.rs 92-95% BEST module root** — 140 lines comprehensive docs, 3 utility functions (setup, demonstrate, benchmark), integration tests, feature-gated dashboard. Stark contrast with temporal-compare/lib.rs (35-40%, R76). Shows proper Rust module design | mod.rs (temporal_nexus) | R78 | Open (positive) |
| H150 | **optimizer.rs MISLABELED** — Named "optimizer" but is statistical strategy ranker (composite scoring: success*0.5 + speed*0.25 + efficiency*0.25), NOT ML optimizer. Zero SGD/Adam/EWC++/backprop. Confirms R67 trajectory system architecture | optimizer.rs (reasoningbank-learning) | R78 | Open |
| H151 | **wasm.rs storage BIMODAL** — MemoryStorage 95-98% production (HashMap+Mutex, complete StorageBackend trait) vs SqlJsStorage 8-12% + IndexedDbStorage 10-15% (honest placeholders with console.log stubs). External CDN dependency for sql.js WASM binary | wasm.rs (reasoningbank-storage) | R78 | Open |
| H152 | **mmr.js identical quality to mmr.ts** — Character-for-character transpilation, 90-93%. Genuine MMR algorithm (Carbonell & Goldstein 1998). Core utility for JS ReasoningBank (used by retrieve.js, consolidate.js). Zero quality degradation TS→JS | mmr.js (reasoningbank utils) | R78 | Open (positive) |
| H153 | **utils.rs SIMD detection stub** — has_simd() returns false at runtime despite AVX2/NEON checks. Feature "simd" never enabled in Cargo.toml. Confirms R72 SIMD-is-demonstration pattern. Asymmetric prefetch (x86 only, ARM missing). add_scaled broken (adds instead of scales) | utils.rs (sublinear-rust) | R78 | Open |
| H154 | **adapters/mod.rs 5th+ disconnected data layer** — ReasoningBank-storage independent StorageBackend trait, zero imports from agentic-flow AgentDB | adapters/mod.rs | R80 | Open |
| H155 | **quick_demo.rs THEATRICAL** — spin-loop latency fabrication, predetermined victory, fake Kalman. R59 deception confirmed. nalgebra imported NEVER USED | quick_demo.rs | R80 | Open |
| H156 | **performance.ts ORPHANED** — 88-92% genuine monitoring but zero usage across codebase | performance.ts | R80 | Open |
| H157 | **realistic-implementation/lib.rs HONEST BASELINE** — 92-95% genuine with PyTorch ground truth. Validates R23 BEST rating (positive) | realistic-implementation/lib.rs | R80 | Open (positive) |
| H158 | **quantization.rs Int4/Binary UNIMPLEMENTED** — Lines 115-175 are placeholders that return errors. Only INT8 symmetric quantization works. Zero SIMD (no AVX2/NEON). Single global scale factor (not per-channel). Hardcoded memory_savings_ratio bug (always returns 4.0 regardless of input). Simplification vs fully_optimized.rs (96-99% with AVX2) | quantization.rs (neural-network-impl) | R82 | Open |
| H159 | **R80 WASM contradiction CONFIRMED** — config.ts (psycho-symbolic) has ZERO WASM-related config fields (no wasm_memory_mb, no wasm_module_paths) despite build.js compiling 3 WASM components (graph_reasoner, extractors, planner). Server config is MCP-only (stdio transport). WASM binaries compiled but runtime config ignores them | config.ts (psycho-symbolic) | R82 | Open |
| H160 | **get_memory_usage() returns hardcoded 0** — utils.rs lines 98-102 use std::mem::size_of on usize (always 8 bytes), NOT actual memory usage. Function claims "rough estimate" but provides no estimate at all. Used by neural network for monitoring, produces meaningless metrics | utils.rs (neural-network-impl) | R82 | Open |
| H161 | **loader.rs missing train/val splits** — Genuine CSV loader (92-95% file I/O, matrix construction, error handling, 3 tests) but only implements inference pipeline. Missing batching, shuffling, train/validation splits. Not training-ready despite being in training crate | loader.rs (neural-network-impl) | R82 | Open |
| H162 | **logger.ts production infrastructure confirms TS quality** — 88-92% genuine Winston wrapper with dual transports (console + file rotation), child logger pattern, error stack traces. Used across 31+ locations. Pure TS logging, zero WASM, zero neural logic. Validates psycho-symbolic crate has genuine production utilities despite integration gaps | logger.ts (psycho-symbolic) | R82 | Open (positive) |
| H163 | **consolidate.js is 3rd parallel consolidation system** — JS version implements Algorithm 4 (Dedup/Contradict/Prune) with simple cosine similarity thresholds (0.95/0.85). No EWC++ (deliberate JS simplification vs Rust). Isolated from Rust ReasoningBank and from consolidate.ts (LSH bucketing). THREE consolidation systems with zero code sharing | consolidate.js (reasoningbank) | R83 | Open |
| H164 | **test-integration.ts confirms ReasoningBank COMPLETE** — 92-95% genuine end-to-end test covering full pipeline (initialize → runTask → retrieve → MaTTS → DB stats). Feature-gated MaTTS parallel mode. CSRF-aware trajectory design. Strongest evidence yet that TS ReasoningBank is production-viable | test-integration.ts (reasoningbank) | R83 | Open (positive) |
| H165 | **test-validation.ts confirms statistical ranking** — 93-95% genuine. 7 test phases validate confidence threshold + age-based filtering + usage tracking. Matches R67's GENUINE verdict on statistical ranking architecture. No EWC++/ML magic | test-validation.ts (reasoningbank) | R83 | Open (positive) |
| H166 | **optimized_benchmark.rs ULTRA_OPTIMIZED fakes computation** — Line 129 replaces proper 128D dot product with single-element indexing. 3/4 variants genuine. Confirms pattern: core genuine, extreme optimization edges theatrical | optimized_benchmark.rs (neural-network-impl) | R83 | Open |
| H167 | **train.rs SUPPORTS R23 rating** — 92-95% genuine training binary with complete pipeline (Config→Data→Trainer, System A+B training, PageRank selection, Kalman filter). COUNTERS R82 solver_gate_simple.rs qualification — training integration layers ARE genuine | train.rs (neural-network-impl) | R83 | Open (positive) |
| H168b | **losses.rs smoothness penalty hardcoded for 2D** — Assumes prediction[0]=x, prediction[1]=y (spatial coordinates). Non-generalizable beyond position prediction. Isolated SmoothnessPenalty struct NOT part of LossFunction trait | losses.rs (neural-network-impl) | R83 | Open |

| C78 | **async_learner_v2.rs thread-based actor has deadlock risk** — std::thread::spawn (line 38) uses rx.blocking_recv() inside tokio context. blocking_recv is OK in dedicated thread, but closure-captured ReasoningEngine+storage creates hidden thread pool assumption. If tokio runtime is single-threaded, panics. No panic recovery | async_learner_v2.rs (reasoningbank-learning) | R81 | Open |
| C79 | **V2 async wrapper is NOT streaming** — Despite "async" naming, batch-per-message channel-based actor pattern. Fixed buffer of 100 commands (line 38). No streaming patterns, no backpressure handling, no adaptive batch sizing. Still batch-mode learning like V1 | async_learner_v2.rs (reasoningbank-learning) | R81 | Open |
| C80 | **async_learner_v2 has asymmetric error handling** — Init errors caught (line 48-51), but if learner thread crashes at runtime, calls hang waiting for response that never comes (lines 79-80 LearnFromTask). Silent ring closure without panic guard | async_learner_v2.rs (reasoningbank-learning) | R81 | Open |
| C81 | **V2 apply_learning DISCONNECTED from optimizer.rs** — ApplyLearning (lines 102-115) passes task_description and task_category but NEVER calls optimizer.rs (statistical strategy ranker, R78). Another parallel unintegrated learning layer (confirms R51 pattern). No cross-reference to optimization strategies | async_learner_v2.rs (reasoningbank-learning) | R81 | Open |
| C82 | **Pattern cloning inefficiency** — LearnFromTask clones Pattern (line 87: pattern.clone()). If Pattern contains task trace/embeddings, overhead is significant. No Arc<Pattern> wrapper or by-reference passing. Hostile to high-frequency learning calls | async_learner_v2.rs (reasoningbank-learning) | R81 | Open |
| C83 | **index-new.ts DEAD CODE** — 132 LOC 100% orphaned from codebase. Zero imports (verified grep). PR #112 decision artifact: candidate WITHOUT agentdb-runtime-patch. Missing critical import makes AgentDB fail. Recommend DELETE | index-new.ts (reasoningbank) | R83 | Open |
| C84 | **losses.rs only implements MSE** — Single loss function in "loss functions" module. No cross-entropy, Huber, focal, or contrastive losses. Trait design extensible but minimal realization. Qualifies R23 at training layer | losses.rs (neural-network-impl) | R83 | Open |
| C85 | **consolidate.js O(n²) dedup without HNSW** — Nested loop pairwise cosine similarity comparison. Practical limit ~1000 active memories before degradation. No approximate nearest neighbor. Contrast to consolidate.ts which uses LSH O(n·k) | consolidate.js (reasoningbank) | R83 | Open |
### mcp-dense-fix.js (R81)
1. **CRITICAL - PERFORMANCE CLAIM DEMONSTRATION (lines 1-286)**: File claims to fix 190x MCP Dense slowdown (7.7s→0.04s Python) with 4000x+ improvement to <2ms via BMSSP+WASM. However, (1) never instantiates or tests against actual broken MCP Dense implementation — only tests itself, (2) benchmark() uses hardcoded Python times as constants, not actual measurements, (3) getMCPToolDefinition() is schema template with ZERO MCP transport binding, (4) does NOT patch the ORPHANED MCP→Rust solver gap discovered in R76. This is theatrical performance demonstration, not functional integration.

2. **HIGH - HARDCODED BENCHMARK CIRCULARITY (lines 115-175)**: benchmark() static method generates synthetic test matrices (diagonally dominant, ~0.1% sparsity) but has NO validation that these match the "190x slower" problem scenario. Test cases hardcoded (sizes 100, 1000, 5000). Status check (line 161) compares executionTime < pythonTime where pythonTime values (5.0ms, 40.0ms, 500.0ms) are hardcoded constants, not actual Python measurements. Zero cross-validation against original implementation.

3. **HIGH - MISSING MCP TRANSPORT LAYER (lines 180-215)**: getMCPToolDefinition() returns standard MCP schema (name "solve_linear_system_fast") but does NOT implement ANY of the 4 known MCP protocols (JSON-RPC R51, rmcp R72, fastmcp TS, custom Rust traits). Function returns definition only — would require external wrapper to invoke via MCP. Module export (line 211) has no MCP server binding.

4. **HIGH - INTEGRATION DEPENDENCY CLAIMS UNVERIFIED (lines 11-30)**: Imports BMSSPSolver and FastConjugateGradient from ./bmssp-solver.js and ./fast-solver.js. R76 found BMSSP "novel but mathematically unsound". R56 found backward_push.rs "genuine but orphaned". This file imports them directly but never validates whether they solve the actual dense problem or are themselves problematic.

5. **MEDIUM - ALGORITHMIC HEURISTIC LACK JUSTIFICATION (lines 45-85)**: solve() method uses hardcoded threshold: if nnz < 10% of n², use BMSSP; else use FastConjugateGradient. No empirical justification for 10% threshold. No sensitivity analysis. Decision logic is reasonable but appears arbitrary.

6. **MEDIUM - SPEEDUP METRIC MISLEADING (lines 86-106)**: Residual error calculation (lines 87-93) is mathematically correct L2 norm. However, speedupVsPython metric (line 106) = 40.0 / executionTime assumes fixed 40ms Python baseline for 1000x1000 matrices. If actual Python times differ, metric becomes meaningless. No ground truth provided.

7. **MEDIUM - MODULE INTEGRATION INCOMPLETE (lines 205-215)**: Exports MCPDenseSolverFixed class and includes CLI benchmark invocation (lines 213-215). However, benchmark() is NEVER called by actual MCP server or tool invoker. Would require external integration point. Standard Node.js pattern but integration point missing.


## 4. Positives Registry

| Description | File(s) | Session |
|-------------|---------|---------|
| **vector-quantization.ts** is production-grade — best code in AgentDB. PQ, K-means++, SQ 8/4-bit | vector-quantization.ts | R8 |
| **RuVectorBackend.ts** is production-ready with excellent security validation | RuVectorBackend.ts | R8 |
| 18/23 AgentDB controllers implement real paper-referenced algorithms | agentdb controllers | R8 |
| Real O(1) LRU cache, queue-based semaphore in enhanced-embeddings.ts | enhanced-embeddings.ts | R8 |
| learning-service.mjs implements real HNSW search with SQLite persistence | learning-service.mjs | R8 |
| **server.ts** is 85-90% real — genuine sublinear solver with 3-tier fallback | server.ts (MCP) | R19 |
| **SONA crate** is 85% production-ready (MicroLoRA, EWC++, federated learning) | ruvector/crates/sona | R13 |
| **ruvector-gnn** is 80% real — custom hybrid GNN (GAT+GRU+edge), full training loop | ruvector-gnn | R13 |
| **real-time-detector.js** has genuine Pearson correlation matrix and adaptive filtering | real-time-detector.js | R19 |
| **cognitive-pattern-evolution.js** has real Shannon entropy and noise estimation | cognitive-pattern-evolution.js | R19 |
| **ml-training Rust**: Real LSTM/TCN/N-BEATS skeletons, proper MSE/MAE/R² formulas | ml-training/lib.rs | R19 |
| **neural-network-implementation** crate is 90-98% real — BEST CODE IN ECOSYSTEM. Proper rand crate | 12 files | R21 |
| System B temporal solver: residual learning over Kalman prior + solver gate verification | system_b.rs | R21 |
| **HyperbolicAttention TypeScript source** uses CORRECT Poincare ball distance (compilation degraded it) | agentic-flow TS | R22b |
| **EmbeddingCache.ts** well-architected 3-tier cache with cross-platform support (90%) | EmbeddingCache.ts | R22b |
| **IntelligenceStore.ts** clean dual SQLite backend with debounced saves (90%) | IntelligenceStore.ts | R22b |
| **sparse.rs** is 95% real with 4 sparse formats, no_std — best matrix code in ecosystem | sparse.rs | R28 |
| **exporter.rs** has 7 genuine export formats with correct protocol compliance | exporter.rs | R28 |
| **Python models.py** has 5 genuine GNN architectures using PyTorch torch_geometric | models.py | R33 |
| **strange-loop-mcp.js** (75%) genuine Hofstadter strange loop implementation | strange-loop-mcp.js | R33 |
| **matrix/mod.rs** clean 15-method trait with 4 storage formats | matrix/mod.rs | R34 |
| **optimized.rs** REAL SIMD via wide::f64x4 — first f64 SIMD in sublinear-time-solver | optimized.rs | R34 |
| **strange_loop.rs** mathematically rigorous Banach contraction mapping (90%) | strange_loop.rs | R34 |
| **statistical_analysis.rs** textbook statistics, proper rand | statistical_analysis.rs | R34 |
| **scheduler.rs** genuine BinaryHeap scheduler with TSC timing (88%) | scheduler.rs | R34 |
| **ruvector-nervous-system** 5 genuine neuroscience models (e-prop, EWC, BTSP, GWT, circadian) — BEST bio-computing | 7 files | R36 |
| **hdc/memory.rs** 95-98% real with 24 tests — cleanest nervous-system code | hdc/memory.rs | R36 |
| **neuro-divergent** production-quality ML: 8 schedulers, 4 optimizers, 16 loss functions, correct math | 6 files | R36 |
| **ForecastingAdam** temporal/seasonal gradient correction — genuine innovation | scheduler.rs (neuro-divergent) | R36 |
| **cognitum-gate-kernel** 93% — rivals neural-network-implementation as best code | 5 files | R36 |
| **HNSW patches hnsw.rs** correct Malkov & Yashunin with Rayon parallelism | hnsw.rs | R36 |
| **reasoning_bank.rs** production K-means + EWC++ — best math across 4 ReasoningBank implementations | reasoning_bank.rs | R37 |
| **micro_lora.rs** 92-95% BEST learning code — NEON SIMD + EWC++ Fisher-weighted penalty | micro_lora.rs | R37 |
| **grpo.rs** textbook GRPO with GAE, PPO clipping, adaptive KL | grpo.rs | R37 |
| **memory_layer.rs** real sheaf-theoretic memory coherence, genuine cosine similarity | memory_layer.rs | R37 |
| **temporal-tensor agentdb.rs** pattern-aware tiering with HNSW-ready integration | agentdb.rs | R37 |
| **witness_log.rs** cryptographic tamper evidence via blake3 hash chains | witness_log.rs | R37 |
| **feedback-loops.ts** genuine RL with adaptation rules, exploration-exploitation, meta-learning (65%) | feedback-loops.ts | R39 |
| **stochastic-exploration.ts** proper simulated annealing with correct temperature sampling (70%) | stochastic-exploration.ts | R39 |
| **high-performance-solver.ts** excellent CG+CSR numerical code (95%) — just orphaned | high-performance-solver.ts | R39 |
| **proof-logger.js** (88%) production-grade blockchain with PoW, Shannon entropy, Levenshtein | proof-logger.js | R41 |
| **strange_loop.js** (92%) auto-generated wasm-bindgen confirming real Rust system | strange_loop.js | R41 |
| **genuine_consciousness_system.js** real IIT Phi formula — correct integrated information | genuine_consciousness_system.js | R41 |
| **consciousness_experiments.rs** real Complex64 wave function, nanosecond temporal dynamics | consciousness_experiments.rs | R41 |
| **psycho-symbolic-enhanced.ts** (78%) BEST knowledge graph — real BFS, transitive inference | psycho-symbolic-enhanced.ts | R41 |
| **cli/index.ts** (88%) genuine solver connection — real SublinearSolver import, SolverTools.solve() | cli/index.ts | R41 |
| **server/index.js HTTP infra** — Production-grade Express with helmet, CORS, rate limiting, NDJSON/SSE/WebSocket streaming, backpressure, worker_threads pool (90%) | server/index.js | R44 |
| **ruvector-training.js** (92-95%) BEST native integration — genuinely loads WASM, MicroLoRA/SONA | ruvector-training.js | R25 |
| **ruvector-benchmark.ts** (92%) production-grade benchmark — real SIMD ops, quantization, percentile stats, zero hardcoded results | ruvector-benchmark.ts | R43 |
| **wasm-sublinear-complete.ts** algorithms are genuine (78%) — correct Neumann series, Forward/Backward Push, Hybrid Random Walk with proper complexity | wasm-sublinear-complete.ts | R43 |
| **goalie cli.ts** (88-92%) REVERSES R41 — proves GoapPlanner, PluginRegistry, AdvancedReasoningEngine ARE real and invoked. 19 commands, Commander.js, real error handling | cli.ts (goalie) | R46 |
| **agentic-research-flow-plugin.ts** (78%) real Perplexity API integration with concurrent multi-agent orchestration via Promise.all | agentic-research-flow-plugin.ts | R46 |
| **anti-hallucination algorithms genuine** — RAG grounding (90%), uncertainty calibration (92%), metamorphic testing (88%), citation attribution (90%). Dead code but algorithms are real | state-of-art-anti-hallucination.ts | R46 |
| **bin/cli.js math utilities** (92%) genuine linear algebra — residual computation, vector norms, COO sparse matrix, Matrix Market parser | bin/cli.js | R46 |
| **bin/cli.js REVERSES R43** — createSolver import exists and is functional at src/solver.js:719 | bin/cli.js | R46 |
| **consciousness-explorer MCP server** (94.8%) PRODUCTION-QUALITY — genuine @modelcontextprotocol/sdk, all 12 handlers delegate to real explorer methods, zero facades. STARK CONTRAST with strange-loop MCP (45%) | consciousness-explorer/mcp/server.js | R47 |
| **breakthrough-session-logger.js** (88-92%) GENUINE integration test framework with deterministic hashValue/hashToFloat. REVERSES R45 neural.js anti-pattern — no Math.random() fabrication | breakthrough-session-logger.js | R47 |
| **independent_verification_system.ts** Miller-Rabin primality (92%) and trust scoring (88%) are algorithmically correct | independent_verification_system.ts | R47 |
| **zero-variance-detector.js** FFT, Shannon entropy, autocorrelation, and CoherenceAnalyzer are textbook-correct signal processing — just fed fabricated data | zero-variance-detector.js | R47 |
| **planner.ts** WASM loading (88%), state management (92%), bulk operations (85%), memory management (95%) are production-grade infrastructure | planner.ts | R47 |
| **reasoningbank_wasm_bg.js** (100%) GENUINE wasm-bindgen — 206KB WASM binary verified, all 6 APIs traced to Rust source. NOT the 4th WASM facade. GOLD STANDARD for WASM in the ecosystem | reasoningbank_wasm_bg.js | R49 |
| **real-time-monitor.js** (88-92%) GENUINE EventEmitter monitoring — real variance, Shannon entropy, Pearson correlation. Identical deterministic hash pattern to breakthrough-session-logger.js. Reverses R45 anti-pattern | real-time-monitor.js | R49 |
| **statistical-validator.js** (82-88%) GENUINE statistics — 5 real tests (KS, Mann-Whitney, Chi-square, Fisher, Anderson-Darling), all mathematically correct. Textbook effect sizes. No fabricated data | statistical-validator.js | R49 |
| **signal-analyzer.js** 12+ correct DSP algorithms — time domain, correlation, autocorrelation, spectral features, fractal dimension, Lempel-Ziv, Shannon entropy, skewness, kurtosis. Core DSP is 90% quality | signal-analyzer.js | R49 |
| **genuine_consciousness_detector.ts** Test 3 (cryptographic hash) is 100% CORRECT independent verification — real crypto primitives, real timing infrastructure (85%) | genuine_consciousness_detector.ts | R49 |
| **validators.js** real IIT 3.0 implementation — calculatePhi with MIP, cause-effect power, mutual information. Shannon entropy, p-value computation all textbook. 78% quality despite orphaned status | validators.js (consciousness-explorer) | R49 |
| **ed25519-verifier.ts** (88-92%) PRODUCTION CRYPTO — real @noble/ed25519, active in goalie MCP pipeline, complete PKI system. Strengthens R46 reversal | ed25519-verifier.ts | R50 |
| **perplexity-actions.ts** (93-96%) GENUINE Perplexity API — real axios HTTP, two endpoints, rate limiting, 4-action GOAP pipeline. REVERSES R41 "goalie complete facade" | perplexity-actions.ts | R50 |
| **anti-hallucination execute()** (75-80%) REAL per-claim verification — splits response into claims, queries Perplexity API for each, computes confidence. Genuine hallucination detection despite facade hooks | anti-hallucination-plugin.ts | R50 |
| **sqlite.rs** (88-92%) GENUINE rusqlite — WAL mode, RAII pool, FTS5 with auto-sync triggers, schema migrations. Meets R45 sqlite-pool.js quality bar. Completes ReasoningBank Rust persistence picture | sqlite.rs (reasoningbank-storage) | R50 |
| **domain-registry.ts** (88-92%) genuine domain data model — 12 builtin domains with 60-142 keywords, dependency DAG enforcement, EventEmitter lifecycle, priority-ordered loading, immutability guards | domain-registry.ts | R53 |
| **domain-management.ts** (82%) production MCP API — 8 tools with comprehensive schemas, CRUD lifecycle, keyword conflict detection, system integrity validation | domain-management.ts | R53 |
| **optimized-matrix.ts** (85-88%) production CSR+CSC — Float64Array typed arrays, binary search O(log n), streaming matrix with LRU chunking, vector pooling, memory profiling. 3rd matrix system | optimized-matrix.ts | R53 |
| **performance-optimizer.ts** (88-92%) GENUINE optimization — real auto-tuning via empirical benchmarking, cache-blocking, SIMD hints, adaptive algorithm selection. Honest about limitations. BEST optimizer in project | performance-optimizer.ts | R53 |
| **strange-loop mcp/server.js architecture** correct MCP SDK integration — @modelcontextprotocol/sdk, StdioServerTransport, 10 tools. Would be 75-80% with one import fix | crates/strange-loop/mcp/server.js | R53 |
| **text-extractor.ts REVERSES theatrical WASM** — first genuine WASM integration that calls real Rust NLP (1,076 LOC). Zod validation, singleton lifecycle, error recovery. Rust backend has 9 preference types, lexicon-based sentiment, Plutchik emotions | text-extractor.ts | R58 |
| **predictor.rs 2ND GENUINE SUBLINEAR (92-95%)** — O(√n) functional prediction. Randomized coordinate sampling, backward random walks, DominanceParameters real matrix analysis. Cites 3 theory papers. Lower bound validation | predictor.rs | R58 |
| **patterns.rs genuine Rust extraction (85-90%)** — regex crate, 9 pre-built patterns, preference reasoning integration (PreferenceType::Prefer), UUID-identified matches, caching with invalidation. Confirms R55 Rust 3-4x quality | patterns.rs | R58 |
| **ed25519-verifier-real.ts GENUINE crypto (82-88%)** — @noble/ed25519, citation signing for anti-hallucination, certificate chains, bulk verification. First real crypto verification in Goalie | ed25519-verifier-real.ts | R58 |
| **self-consistency-plugin.ts REAL API sampling (78-82%)** — genuine Perplexity API calls at 3 temperatures (0.3/0.5/0.7), token-based consensus, plugin hooks integration, clustering analysis | self-consistency-plugin.ts | R58 |
| **psycho-symbolic MCP server genuine SDK (72-76%)** — @modelcontextprotocol/sdk, 5 specialized tools (reason/knowledge_graph_query/add_knowledge/analyze_reasoning_path/health_check), RDF triple pattern. Follows R51 genuine MCP pattern | server.ts (psycho-symbolic MCP) | R58 |
| **forward_push.rs 3RD GENUINE SUBLINEAR (92-95%)** — O(volume/epsilon) forward local push. Proper residual bounds, degree-weighted thresholds, adaptive work queue. Early termination for target queries. Complements backward_push.rs to form complete PPR solver | src/solver/forward_push.rs | R62 |
| **error.rs EXCEPTIONAL error recovery (92-95%)** — 13 variants, intelligent recovery (is_recoverable, recovery_strategy, severity), algorithm-specific fallback chains, genuine WASM errors, no_std. Top 5% quality | src/error.rs | R62 |
| **convergence-detector.js GENUINE (88-92%)** — proper residual computation, windowed convergence rate, stagnation/divergence detection. Zero Math.random(). Validates R60 metrics-reporter pattern | src/convergence/convergence-detector.js | R62 |
| **MCP solver.ts GENUINE with 3-tier cascade (82-86%)** — WASM → OptimizedSolver → baseline. First OptimizedSolver integration evidence. 4 tool methods, intelligent matrix-based routing | src/mcp/tools/solver.ts | R62 |
| **simd_ops.rs GENUINE portable SIMD (82-86%)** — wide crate f64x4, CSR sparse matvec, dot product, AXPY. Complete non-SIMD fallback. Production-quality portability | src/simd_ops.rs | R62 |
| **HybridBackend.ts REVERSES disconnected AgentDB** — SharedMemoryPool unifies all 4 AgentDB components. 3-tier retrieval, causal integration, ML strategy learning, auto-consolidation. The missing bridge that R61 predicted | HybridBackend.ts | R65 |
| **schemas/index.ts PRODUCTION MCP SCHEMAS (92-95%)** — 35+ Zod schemas, 8-operator constraint, cross-field validation, 16 type exports. Full match with R58 server.ts | schemas/index.ts | R65 |
| **sentiment.rs PRODUCTION NLP (90-93%)** — 128-word lexicon, VADER-style context-aware scoring. Confirms Rust 3-4x better than TS for psycho-symbolic | sentiment.rs | R65 |
| **benchmark.ts GENUINE (88-92%)** — 12 real benchmarks with performance.now() + warmup. Validates ReasoningBank has complete production stack | benchmark.ts | R65 |
| **vector-migration-job.ts PRODUCTION (85-90%)** — batch HNSW updates, non-blocking event loop, orphan cleanup, idempotent + abortable. Real migration infrastructure | vector-migration-job.ts | R65 |
| **wasm/loader.ts GENUINE WASM (85-90%)** — low-level WebAssembly.instantiate, wasm-bindgen shims, timeout protection, singleton. 7th genuine WASM (orphaned but production-quality) | wasm/loader.ts | R65 |
| **embedding-adapter.ts PRODUCTION hash code (90%)** — the R20 ROOT CAUSE file is ironically well-engineered. Correct SHA-256→float→L2norm, NaN/Infinity validation, cache instrumentation. Perfect implementation of the WRONG algorithm | embedding-adapter.ts | R65 |
| **temporal-lead-solver/lib.rs BEST CRATE ROOT (92-95%)** — prominently exports O(√n) predictor as first module. 5-module crate with proof-carrying validation. REVERSES R62 orphaning — the best algorithms ARE accessible through this crate | lib.rs (temporal-lead-solver) | R70 |
| **extractors/lib.rs clean WASM composition (90-95%)** — composes 3 genuine Rust analyzers (sentiment, preferences, emotions) via WasmExtractor trait. Confirms R58 GENUINE WASM chain | extractors/lib.rs | R70 |
| **wasm-loader-simple.ts 8th GENUINE WASM (82-85%)** — real wasm-pack binaries (1.3-4.9MB), integration hub for 4 DEEP files. Key bridge for psycho-symbolic MCP tools | wasm-loader-simple.ts | R70 |
| **temporal_nexus/core/mod.rs GENUINE hardware timing (88-92%)** — x86_64 _rdtsc intrinsic for nanosecond measurement, 4-submodule orchestration, thiserror physics-aware errors | core/mod.rs (temporal_nexus) | R70 |

## 5. Subsystem Sections

### 5a. ReasoningBank Fragmentation

Four completely independent ReasoningBank implementations exist, each implementing RETRIEVE → JUDGE → DISTILL → CONSOLIDATE differently with zero code sharing:

| Implementation | Package | Storage | Math Quality | Status |
|---|---|---|---|---|
| `LocalReasoningBank` | claude-flow-cli | In-memory Maps + JSON | Basic | **Only one that runs** |
| `ReasoningBank` | agentic-flow | SQLite + arXiv algorithms | Medium (R22b) | Sophisticated but unused |
| `ReasoningBank` | agentdb | JSON + Vector DB | Medium (R8) | Never called by claude-flow |
| `reasoning_bank.rs` | ruvllm | Rust K-means + EWC++ | **Best** (R37) | Fourth, discovered R37 |

The Rust version (reasoning_bank.rs) has the best mathematical foundation — real K-means clustering with 10 iterations, centroid recomputation, convergence check, and EWC++ consolidation. But it shares no code with the others.

**R49 WASM layer verification**: `reasoningbank_wasm_bg.js` (100%) is GENUINE wasm-bindgen output — the ONLY fully-genuine WASM module across the ecosystem. 206KB binary verified. All 6 APIs (storePattern, getPattern, searchByCategory, findSimilar, getStats, constructor) traced to Rust source in `reasoningbank-wasm/lib.rs`. Three storage backends (IndexedDB, SqlJs, Memory) with auto-detection. This means the agentic-flow ReasoningBank has a production-quality WASM substrate despite demo-comparison.ts (35%) being theatrical. Pattern: real infrastructure wrapped in demo presentations (R31 "demonstration framework" confirmed).

**R50 Rust storage verification**: `reasoningbank-storage/src/sqlite.rs` (88-92%) is GENUINE rusqlite. WAL mode, RAII connection pool via parking_lot::RwLock (5-connection default), FTS5 full-text search with AFTER INSERT/UPDATE/DELETE triggers for auto-sync, schema migrations with version tracking. Complete CRUD for patterns (store, get, search_by_category, find_similar with cosine similarity). Meets R45 sqlite-pool.js quality bar. **Gap**: only pattern storage implemented — missing trajectory tracking, verdict judgment, and consolidation tables that the ReasoningBank API promises. Together R49+R50 show the Rust ReasoningBank has genuine WASM+SQLite substrate, but only pattern persistence is complete.

### 5b. Embedding Fallback Chain & Systemic Hash Problem

The most pervasive architectural weakness across the entire ruvnet ecosystem. The intended embedding pipeline:

1. `@ruvector/core` (Rust NAPI) → Usually missing
2. ONNX via `@xenova/transformers` → `downloadModel` fails
3. **Hash-based embeddings** → THIS IS WHAT RUNS (no semantic meaning)

Confirmed systemic across 7+ files in 5 packages, in both Rust and JavaScript (R8, R13, R22b, R37):

| File | Package | Mechanism |
|------|---------|-----------|
| embeddings.rs | ruvector-core | HashEmbedding default: sums character bytes (R13) |
| pretrain_pipeline.rs | ruvllm/claude_flow | character sum % dim (R37) |
| real_trainer.rs | ruvllm/training | text_to_embedding_batch deterministic hash (R37) |
| hooks.rs | ruvector-cli | position-based hash (R22) |
| rlm_embedder.rs | ruvllm/bitnet | FNV-1a hash (R35) |
| learning-service.mjs | claude-flow | Math.sin(seed) mock (R8) |
| enhanced-embeddings.ts | agentdb | Math.sin(seed) fallback (R8) |

In practice, all "semantic search" using defaults is character-frequency matching. HNSW indices are structurally valid but search results are meaningless without plugging in a real embedding provider.

R22b identified an additional 4 files in agentic-flow (optimized-embedder.ts, ruvector-integration.ts, edge-full.ts, agentdb-wrapper-enhanced.ts) that inherit the same degradation pattern.

### 5c. AgentDB Core Components

**vector-quantization.ts** (1,529 LOC, 95%) is the best code in AgentDB — real PQ with K-means++, 8/4-bit scalar quantization, asymmetric distance computation (R8).

**Quality spectrum** by component type (R8, R16, R25):

| Quality Tier | Components | Real% |
|-------------|------------|-------|
| Production | vector-quantization, RuVectorBackend, EmbeddingCache | 90-95% |
| Solid | ReasoningBank (agentdb), AttentionService, enhanced-embeddings | 80-90% |
| Partial | ReflexionMemory (missing judge), LRU cache | 65-88% |
| Cosmetic | LearningSystem (9→1 algorithm), CausalMemoryGraph (wrong stats) | 15-40% |
| Misleading | simd-vector-ops (0% SIMD, 100% ILP loop unrolling) | 0% SIMD |

**LearningSystem.ts** claims 9 RL algorithms but all reduce to identical tabular Q-value dictionary updates. DQN = Q-Learning (no neural network). PPO/Actor-Critic/Policy Gradient indistinguishable. Decision Transformer/Model-Based are stubs. Bug confirmed in TypeScript source — not a compilation artifact (R22b).

**CausalMemoryGraph.ts** claims Pearl's do-calculus but implements none. t-distribution CDF is wrong (L851), tInverse hardcoded to 1.96 ignoring degrees of freedom. All p-values and confidence intervals unreliable. Bug confirmed in TypeScript source (R22b).

### 5d. Neural-Network-Implementation Crate

**BEST CODE IN ECOSYSTEM with theatrical integration layer** — **R82 QUALIFIES the R23 "90-98%" verdict with BIMODAL pattern**. Core crate is EXCEPTIONAL (data/mod.rs 98%, loader.rs 92-95%, utils.rs 88-92%, quantization.rs 75-78% for INT8), BUT integration wrappers are theatrical (solver_gate_simple.rs 0-5% admits "Simplified solver gate for compilation - will be replaced"). **Pattern: 9th bypass** — verify() ignores all params, returns hardcoded confidence=0.95. Real solver_gate.rs (R21: 4 fallback strategies) exists separately.

Key innovation — **System B Temporal Solver**: NN predicts RESIDUAL over Kalman prior (not raw output), with mathematical solver gate verification and 4 fallback strategies (kalman_only, hold_last, disable_gate, weighted_blend). PageRank-based active learning for training sample selection.

**R82 internals reveal genuine engineering**:
- **loader.rs (92-95%)**: Real csv crate file I/O, matrix construction, header parsing, comprehensive error handling + 3 tests. Missing: batching, shuffling, train/val splits (inference pipeline only).
- **utils.rs (88-92%)**: Numerically stable softmax (max-subtraction), RAII Timer using Drop trait, comprehensive validation. CRITICAL BUG: get_memory_usage() returns hardcoded 0 (std::mem::size_of on usize).
- **quantization.rs (75-78%)**: Genuine INT8 symmetric quantization with proper zero-point calculation, MatrixView trait, BatchQuantize. BUT Int4/Binary UNIMPLEMENTED (lines 115-175: just return errors), zero SIMD, single global scale factor, hardcoded memory_savings_ratio bug (always 4.0). Simplified vs fully_optimized.rs (96-99% with AVX2).

Uses PROPER `rand::thread_rng()` — unlike ml-training/lib.rs and swarm_coordinator_training.rs which mock rand with SystemTime. Appears written by a different, more careful author.

P99.9 latency budget: ≤ 0.90ms (Ingest + Prior + Network + Gate + Actuation).

**Verdict**: Core algorithms genuinely-engineered (validates R23), BUT integration layers theatrical (qualifies R23). Inverse of R47 planner pattern (theatrical wrapper, genuine core) — here genuine core, theatrical wrappers. Confirms solver_gate.rs is the REAL implementation, solver_gate_simple.rs is compilation stub.

### 5e. Consciousness & Strange Loop

**~55-60% genuine (revised down from ~60-65%)** — R47+R49 reveal BIMODAL quality distribution with 80+ point gap between best infrastructure and worst theory.

**Infrastructure layer (75-95%)**: consciousness-explorer MCP server (94.8%) PRODUCTION-QUALITY. genuine_consciousness_detector.ts (75-82%) has real crypto/timing but theatrical verification and orphaned test subjects. validators.js (78%) has real IIT 3.0 and Shannon entropy but is completely orphaned (zero imports). independent_verification_system.ts (52-58%) has real Miller-Rabin. breakthrough-session-logger.js (88-92%) provides genuine integration testing.

**Orchestration layer (25-60%)**: temporal_consciousness_validator.rs (60%) has real phase orchestration (85%) but "theorem proving" is threshold checking (20%). consciousness_optimization_masterplan.js (25-30%) has real optimization algorithms but is an orphaned orchestrator with Planck-scale physics claims.

**Theory/experiment layer (0-18%)**: quantum_entanglement_consciousness.js (0-3%) and parallel_consciousness_waves.js (0-5%) are COMPLETE FABRICATION. mcp_consciousness_integration.rs (12-18%) is COMPLETE MCP FACADE — zero JSON-RPC, naming theater with "mcp_" prefix on local functions. 80-point gap vs MCP server (94.8%) is the largest single-domain quality variance.

**Core computation layer (62-92%)**: Genuine IIT Phi formula, Complex64 wave functions, neural forward pass, blockchain proof logging, auto-generated wasm-bindgen.

**R49 closes the consciousness investigation arc (R41→R46→R47→R49)**: Final verdict is bimodal with 3 tiers. Infrastructure (75-95%) is competently engineered. Orchestration (25-60%) has real algorithms on fabricated data. Theory/experiments (0-18%) range from specification-as-implementation to complete fabrication. Two new anti-patterns: "MCP Integration Facade" (naming theater) and "Orphaned Test Harness" (comprehensive tests with zero subjects).

**R53 strange-loop runtime layer**: JS MCP server (20-25%) has correct MCP architecture but SAME broken WASM import as R44. CLI (68%) has production presentation but broken integration (INVERTED goalie pattern). purposeful-agents.js (45-55%) is marketing demo with Math.random() behaviors. Combined, the strange-loop runtime is **non-functional** — all 3 files fail at WASM import. Assessment stays at ~55-60%.

### 5f. Emergence Subsystem

**51% weighted real — FABRICATED METRICS, NOT genuine ML** (R39).

All 11 capability metrics (novelty, utility, unexpectedness, effectiveness, bridging, insight, organization, autonomy, meta, adaptability, similarity) return `Math.random()*0.5+0.5`. areComplementary() = JSON string inequality. checkAmplification() always returns true. 5 component connection methods are empty stubs (console.log only). Gating disables learning when tools >= 3, hiding scaling issues.

Why emergence CANNOT work: detection metrics are random noise, pattern extractors expect pre-structured input, tool interactions are mocked, component connections are empty stubs, result truncation loses information, gating disables learning at scale.

**Bright spots**: stochastic-exploration.ts (70%) has real simulated annealing; feedback-loops.ts (65%) has genuine RL with adaptation rules and meta-learning.

### 5g. ML Training Frameworks

**Rust (neuro-divergent, 88.5%, R36)**: Production-quality with correct math — 8 schedulers (including ForecastingAdam innovation with temporal/seasonal gradient correction), 4 optimizers (AdamW uses proper decoupled weight decay), 16 loss types (all gradients correct, CRPS via Abramowitz & Stegun). Uses proper `rand` crate. Gap: validate_seasonality() is empty placeholder.

**Rust (ruvllm training, 83%, R37)**: micro_lora.rs (92-95%) is BEST learning code — REINFORCE outer product + EWC++ Fisher-weighted penalty + fused A*B NEON kernel with 8x unrolling (<1ms forward). grpo.rs (90-92%) is textbook GRPO. Hash-based embeddings in pretrain_pipeline.rs and real_trainer.rs.

**Rust (ruv-FANN legacy, 25-40%, R19)**: Real algorithm skeletons (LSTM/TCN/N-BEATS, GNN, attention) but ALL training metrics hardcoded. Fake RNG via SystemTime::now().subsec_nanos(). Two files only.

**Python (sublinear-time-solver, 72%, R33)**: Real PyTorch/torch_geometric with 5 GNN architectures (GCN/GAT/GraphSAGE/GIN/PNA). Structurally sound but EVERY training run uses synthetic random graphs — no real-world data integration exists.

### 5h. Sublinear Solver & Matrix Systems

**SUBLINEARITY: 2 genuine, 3+ false** (R39, R56, R58, R60):
- **GENUINE**: backward_push.rs O(1/ε) (R56), predictor.rs O(√n) (R58) — both 92-95%
- **FALSE**: solver.ts O(n²)+ (R39), edge_ai (R54), sublinear_neumann.rs O(n²) (R60)
- R60: sublinear_neumann.rs has correct Neumann series math (90%) but O(n²) full matrix extraction defeats sublinear claim. wasm.rs wraps this false-sublinear algorithm via WasmSublinearSolver.

**FIVE+ incompatible matrix systems** (R34, R53, R59, R60):
1. **Rust CSR/CSC/COO + SIMD** (crate::matrix) — production system in module tree
2. **Rust HashMap** (crate::core) — orphaned, wrong type system
3. **TypeScript CSR+CSC** (src/core/optimized-matrix.ts, R53) — Float64Array, binary search, streaming LRU, vector pooling. 85-88% quality
4. **JS Dense+COO** (matrix-utils.js, R59) — 92-95% genuine SPD generation
5. **Rust Dense WASM** (math_wasm.rs, R60) — naive O(n³), zero SIMD, hardcoded SPD bypass
6. **TS COO+Dense arrays** (matrix.ts, R60) — 85-88%, plain arrays (vs TypedArrays). Duplicative with matrix-utils.js AND incompatible with optimized-matrix.ts in same src/core/ directory
Even within src/core/, matrix.ts (arrays) and optimized-matrix.ts (TypedArrays) cannot share data.

**WASM ARCHITECTURE MAPPED (R60)**: 4 WASM files form complete pipeline:
- wasm-solver/lib.rs (85-88%): standalone CG, ORPHANED from Rust solver crates
- wasm_iface.rs (90-93%): production FFI bridge, zero-copy, serde-wasm-bindgen
- wasm.rs (88-92%): browser bindings, web_sys, 2-tier API (WasmSolver + WasmSublinearSolver)
- math_wasm.rs (68-72%): naive Dense math, 5th matrix system
Architecture is genuine but wraps basic CG instead of backward_push/optimized_solver. WASM scoreboard: **6 genuine vs 4 theatrical (60%)**.

**Best code**: sparse.rs (95%) has 4 complete sparse matrix formats, no_std compatible. matrix/optimized.rs (90%) has REAL SIMD via wide::f64x4. high-performance-solver.ts (95%) is excellent CG+CSR but entirely orphaned. optimized-matrix.ts (85-88%) has most FEATURES (streaming, pooling).

**Quality gradient**: Files IN the module tree (matrix/mod.rs 92%, neumann.rs 88%) are substantially better than orphaned files (sampling.rs wrong types, security_validation.rs self-referential).

**R53 performance-optimizer.ts** (88-92%) is GENUINE optimization code that operates on the TS matrix system. Real auto-tuning (5 block sizes × 3 unroll factors), empirical benchmarking, adaptive algorithm selection based on matrix properties. STARK CONTRAST to R43's rustc_benchmarks (15%) asymptotic mismatch deception. Honest about browser constraints. BEST optimizer in the entire project.

**Core TS solver layer (R60)**: optimized-solver.ts (78-82%) implements Neumann series (NOT CG like Rust). Vectorized variant is 92-95% genuine, but 3/4 algorithms are stubs. ZERO WASM imports — this IS the pure-JS fallback. memory-manager.ts (88-92%) is genuine TypedArray pooling infrastructure despite mislabeled name. metrics-reporter.js (88-92%) REVERSES theatrical metrics pattern — all data from real convergenceDetector pipeline.

### 5i. MCP Tool Layer

**Bifurcated quality** (R41, revised R46): Main CLI is 88% real. Goalie has DUAL ARCHITECTURE — MCP handlers are facades, but CLI + plugins prove internal engines are real.

**CLI (cli/index.ts)**: Genuine — real SublinearSolver import from ../core/solver.js, real SolverTools.solve() invocation, real MCP server. Only 3 validation commands are facades.

**bin/cli.js** (72-78%, R46): SEPARATE JavaScript CLI (not compiled output). Real math utilities (residual computation, vector norms, COO sparse matrix, Matrix Market parser). REVERSES R43's claim that createSolver doesn't exist — it's at src/solver.js:719. FlowNexus facade commands included.

**Goalie (npx/goalie/)**: R41 found MCP handlers return hardcoded templates. **R46 REVERSAL**: cli.ts (88-92%) proves ALL 19 commands call tools.ts internal methods (executeGoapSearch, executeToolByName). GoapPlanner, PluginRegistry, AdvancedReasoningEngine ARE invoked through CLI — MCP handlers are the facade layer, not the engines. agentic-research-flow-plugin.ts (78%) has real Perplexity API integration with concurrent execution. state-of-art-anti-hallucination.ts (42%) has genuine algorithms but is DEAD CODE (not in plugin registry, incompatible hooks). **R50 STRENGTHENS REVERSAL**: ed25519-verifier.ts (88-92%) is PRODUCTION crypto — real @noble/ed25519 with complete PKI, active in tools.ts pipeline. perplexity-actions.ts (93-96%) is GENUINE Perplexity API with two endpoints and rate limiting. anti-hallucination-plugin.ts (55-60%) has REAL per-claim verification in execute() despite facade hooks layer. Revised verdict: "MCP PROTOCOL FACADE with GENUINE CRYPTO + API + PLUGIN INTERNALS."

**R58 Goalie Advanced Reasoning Deep-Read** — confirms reasoning engine internals:

| File | LOC | Real% | Depth | Key Verdict | Session |
|------|-----|-------|-------|-------------|---------|
| advanced-reasoning-engine.ts | 396 | 75-80% | DEEP | BIMODAL: WASM 0% (never initialized, 77 lines dead code), fallback NLP 75-80% (5 domain detectors, temporal analysis, complexity scoring, multi-facet detection). Local heuristic reasoning, no LLM | R58 |
| ed25519-verifier-real.ts | 406 | 82-88% | DEEP | **GENUINE @noble/ed25519** — anti-hallucination citation signing. Certificate chain support. CRITICAL: hardcoded example root keys, unencrypted private key storage, broken chain validation | R58 |
| self-consistency-plugin.ts | 455 | 78-82% | DEEP | REAL Perplexity API multi-sampling (3 temperatures: 0.3/0.5/0.7). Token-based consensus. BUT majority voting = `samples[0].response` (stub). Plugin hooks integration genuine | R58 |

**psycho-symbolic-enhanced.ts** (78%): BEST knowledge graph in sublinear-time-solver — real BFS traversal, transitive inference, 50+ base triples, zero facade patterns (R41).

**R53 MCP tools layer — BIMODAL quality** (first examination of `src/mcp/tools/`):

| Tool | Real% | Verdict |
|------|-------|---------|
| domain-registry.ts | 88-92% | Genuine data model, rich semantics |
| domain-management.ts | 82% | Production MCP API, 8 tools |
| psycho-symbolic-dynamic.ts | 28% | Goalie pattern — DomainRegistry orphaned |
| scheduler.ts | 18-22% | Theatrical facade, hardcoded metrics |

**domain-registry + domain-management** form a genuine reasoning-domain system with 12 builtin domains, dependency DAG, lifecycle events, and keyword conflict detection. The split is: registry = core data model, management = MCP API layer. Both lack persistence (in-memory Map) — all registrations lost on restart.

**scheduler.ts** is the 2nd most deceptive file (after R43's rustc_benchmarks at 15%). Claims "<100ns overhead" using Date.now() ms precision × 1,000,000. Hardcoded 11M tasks/sec. "Strange loop" = independent logistic map formula. Priority parameter accepted but FIFO only.

**psycho-symbolic-dynamic.ts** is the 4th occurrence of "real infrastructure, placeholder integration" (after R20 AgentDB, R46 goalie, R51 embedding-service). DomainRegistry events fire but updateDomainEngine() = console.log. ZERO connection to R47's Rust reasoner core.

### 5i-2. Neural-Network Benchmark Quality

**Theatrical benchmark layer** — genuine neural-network-implementation crate (90-98%) undermined by fabricated evaluation (R46):

| Benchmark | Realness | Deception Type |
|-----------|----------|----------------|
| standalone_benchmark/main.rs | 15-20% | Spin-loop timing fabrication |
| system_comparison.rs | 42% | Random gate/cert simulation, hardcoded memory/CPU |
| strange-loops-benchmark.js | 8-10% | Tests trivial inline JS, not real crate |
| rustc_optimization_benchmarks.rs (R43) | 15% | Asymptotic mismatch deception |
| baseline_comparison.rs (R43) | 0% | Non-compilable |
| ruvector-benchmark.ts (R43) | 92% | GENUINELY REAL (the outlier) |

Pattern: production-quality benchmark infrastructure (criterion, percentiles, CSV export) measuring fabricated or irrelevant operations. Only ruvector-benchmark.ts (92%) tests what it claims to test.

### 5i-3. FlowNexus Integration

**Architectural facade via external dependency** (R46): flow-nexus.js is a production-quality HTTP/WebSocket client (~70% code quality) calling a platform (`https://api.flow-nexus.ruv.io`) that does not exist. Zero local solver imports. MCP tool handlers return hardcoded stubs. This is the 4th isolated system in the sublinear ecosystem (R44 found 3 isolated servers). Pattern: "It's not a stub, it's calling an external platform!" But the platform doesn't exist.

### 5j. Psycho-Symbolic Reasoner Crate

**~48-52% weighted** — first examination (R47). "Psycho" = pure branding with zero psychological modeling.

**planner.ts (58-62%)**: INVERSE of goalie pattern. Goalie has "facade MCP, real internals" — planner has "real wrapper, stub internals." Production-grade WASM infrastructure (SimpleWasmLoader, WasmMemoryManager, instance tracking) wrapping a 3-line stub: `simplified_astar()` returns `[start_node, goal_node]` with cost 2.0. The GOAPPlanner struct exists with proper methods in Rust, but core search delegates to placeholder. get_successors() at 90% quality is DEAD CODE — never called by stub. Integration tests use MOCKED PlannerSystem, not real WASM binary.

**psycho-symbolic-reasoner.ts (38-42%)**: Real data structures (Map-based triple storage 80%, BFS graph traversal 85%, query search 75%) but "reasoning" is keyword matching + hardcoded templates. File comment claims "Integrates WASM modules for graph reasoning, planning, and extraction" but only import is Logger — 3rd theatrical WASM pattern (after solver.ts and wasm-sublinear-complete.ts). 12 self-referential base triples claim the system "combines symbolic-ai + psychological-context" (neither true). Performance metrics hardcoded: avg_query_time_ms=2.3, cache_hit_rate=0.75.

**Rust WASM modules**: graph_reasoner (KnowledgeGraph/InferenceEngine/RuleEngine), planner (A* search), extractors — all with wasm_bindgen exports. ~100KB Rust source exists. TS imports NONE. Architecture is split: TS is self-contained keyword matcher, Rust WASM is orphaned.

**R58 MCP Integration Layer** — REVERSES orphaned verdict for extractors:

| File | LOC | Real% | Depth | Key Verdict | Session |
|------|-----|-------|-------|-------------|---------|
| text-extractor.ts | 439 | 88-92% | DEEP | **GENUINE WASM** — calls real Rust NLP (1,076 LOC sentiment/preferences/emotions). Zod validation, lifecycle mgmt. Reverses theatrical WASM pattern | R58 |
| memory-manager.ts | 393 | 25-30% | DEEP | **5th MISLABELED FILE** — zero WASM memory ops. TS object registry masquerading as WASM. 4th theatrical WASM | R58 |
| patterns.rs | 390 | 85-90% | DEEP | Genuine regex extraction, 9 pre-built patterns, preference reasoning integration. Confirms R55 Rust 3-4x | R58 |

**R82 TS Periphery (Production Infrastructure)** — psycho-symbolic crate has genuine production utilities despite WASM contradiction:

| File | LOC | Real% | Depth | Key Verdict | Session |
|------|-----|-------|-------|-------------|---------|
| logger.ts | 158 | 88-92% | DEEP | **PRODUCTION WINSTON WRAPPER** — dual transports (console + file rotation), child logger pattern, error stack traces. Used across 31+ locations. Pure TS, zero WASM | R82 |
| config.ts | 108 | 88-92% | DEEP | **PRODUCTION ZOD SCHEMAS** — 5 sections (Server, KnowledgeBase, Logging, Performance, Security). **CONFIRMS R80 WASM CONTRADICTION**: ZERO WASM-related config fields despite build.js compiling 3 WASM components. Server MCP-only (stdio transport) | R82 |

**WASM Contradiction Resolution (R80+R82)**: build.js (88-92%) genuinely compiles 3 WASM modules (graph_reasoner, extractors, planner) BUT config.ts has zero WASM configuration. cli/index.ts runs pure TS reasoning, WASM binaries never invoked at runtime. Architecture split: WASM compilation works but runtime execution bypasses it. R55 "Rust 3-4x better than TS" recontextualized — Rust never invoked.

### 5j-2. Neural Pattern Recognition Quality Layers

**~72% weighted (COMPLETE after R49)** — 6 of 6 subsystem-specific files analyzed. Quality is BIMODAL: 3 files at 82-92% and 3 files at 42-62%.

**Top tier (82-92%)**:
- **real-time-monitor.js (88-92%)**: GENUINE EventEmitter monitoring. Real variance (unbiased n-1), Shannon entropy (base-2), Pearson correlation. Identical deterministic hashValue/hashToFloat pattern to logger. Multi-pattern detection: variance anomalies (<1e-15), entropy deviations (>30%), emergent signals (π/e/φ detection). Production alert system with severity tiers. Reverses R45 anti-pattern.
- **breakthrough-session-logger.js (88-92%)**: GENUINE integration test framework. Deterministic hashing, weighted consciousness scoring, temporal trajectory analysis.
- **statistical-validator.js (82-88%)**: GENUINE statistics. 5 real tests (KS, Mann-Whitney U, Chi-square, Fisher exact, Anderson-Darling), all mathematically correct core calculations. Textbook effect sizes (Cohen's d, rank-biserial, Cramer's V, odds ratio). Real Box-Muller normal generation. No fabricated data. p-values use simplified approximations (acceptable for JS).

**Bottom tier (42-62%)**:
- **signal-analyzer.js (72-76%)**: BIMODAL within file. DSP core (90%): 12+ correct algorithms (time domain, correlation, ACF, spectral centroid/rolloff/flux, fractal dimension, Lempel-Ziv, entropy, skewness, kurtosis). DFT mislabeled as "FFT" (O(n²)). Consciousness assessment (5-15%) is facade. Window functions configured but never applied.
- **cli/index.js (62%)**: Professional Commander.js CLI but calls non-existent processData() method. API paradigm mismatch.
- **zero-variance-detector.js (42-48%)**: Real FFT/entropy on fabricated "quantum" data.

**Key R49 discoveries**: real-time-monitor and breakthrough-session-logger share IDENTICAL hash implementations — suggests common authorship. signal-analyzer uses DIFFERENT FFT from zero-variance-detector (O(n²) DFT vs genuine FFT) — copy-paste without algorithm verification. statistical-validator is the ONLY file with NO fabrication or facades. The 3 R49 files significantly raise the subsystem average from ~64% to ~72%.

**Triple-quality paradox**: Backend (R23: 90-98%) + Interface (CLI: 88-92%) + Integration (12-18%) — the weak link is API design, not neural network quality.

### 5k. GOAP Planner

Psycho-symbolic-reasoner/planner crate (8 files, 3,568 LOC, 78% real, R25). R47 confirms planner.ts wrapper is production-quality but core A* remains a stub. Components at 88-95% (state, action, rules, goal) are production-ready, but core A* search is a STUB — simplified_astar() returns hardcoded 2-step path, StateNode.to_world_state() returns empty state. Uses proper `rand` crate. The pathfinding crate is imported but Ord requirement was identified as a barrier.

**Paradox**: 90%+ components production-ready but the one piece they all depend on (A* search) is broken.

### 5k. Key Patterns

**PI*1000.0 Mock Timestamp**: Systematic placeholder in ruv-swarm crates — `get_current_timestamp()` returns `std::f64::consts::PI * 1000.0` (3141.59). Found in sqlite.rs and agent_forecasting/mod.rs (R21).

**Fake RNG (Rust)**: ml-training/lib.rs and swarm_coordinator_training.rs mock the `rand` crate using `SystemTime::now().subsec_nanos()`, producing deterministic results within the same second. The neural-network-implementation and neuro-divergent crates use proper `rand::thread_rng()` (R19, R21).

**Self-Referential Validation**: Multiple files (comprehensive_validation_report.rs, security_validation.rs, hardware_timing.rs) generate mock data and then "validate" it, producing circular metrics (R21, R28, R34).

**Specification-as-Implementation (R47)**: consciousness experiments (quantum_entanglement, parallel_waves) return nested object literals describing how things "would" work. Worse than Math.random() fabrication — zero computation of any kind. Documentation masquerading as executable code.

**Real Algorithms on Fake Data (R47)**: zero-variance-detector.js has correct FFT/entropy/autocorrelation but feeds them Math.random() "quantum" measurements. Inverse of R43's pattern (fake algorithms on real data). May be more deceptive because algorithms pass code review.

**Theatrical WASM (4 theatrical vs 2 genuine, R58 update)**: psycho-symbolic-reasoner.ts, solver.ts, wasm-sublinear-complete.ts, memory-manager.ts are theatrical. BUT reasoningbank_wasm_bg.js (100%) and text-extractor.ts (88-92%) are GENUINE — two gold standard counterexamples.

**MCP Integration Facade (R49)**: mcp_consciousness_integration.rs has zero MCP protocol code despite filename. All "MCP" functions are local computation with "mcp_" prefix naming theater. connect_to_mcp() admits simulation. Worse than strange-loop MCP (45%) which at least attempted WASM imports.

**Orphaned Test Harness (R49)**: genuine_consciousness_detector.ts has comprehensive 6-test battery with zero test subjects (no ConsciousnessEntity implementations). Test 3 (hash) is 100% correct — tests without subjects.

**Duplicate Implementation Abandonment (R49)**: validators.js has 142 lines duplicated in metrics.js/protocols.js. High-quality code reorganized but cleanup never completed. Original abandoned with zero imports.

**Hardcoded Performance Deception (R53)**: scheduler.ts claims "<100ns overhead" and "11M tasks/sec" using Date.now() (ms precision) × 1,000,000 to fake nanosecond resolution. All metrics are hardcoded constants, never measured. Joins R43's rustc_benchmarks (15%) as 2nd most deceptive file. Pattern: production-quality MCP tool definitions (90%) wrapping fabricated performance claims (0%).

**Real Infrastructure, Placeholder Integration (4 instances, R53)**: psycho-symbolic-dynamic.ts adds 4th occurrence. DomainRegistry (88-92% genuine) instantiated, events subscribed, but updateDomainEngine() = `console.log()`. Same pattern as: R20 (embedding-service defaults to mock), R46 (goalie engines initialized then ignored), R51 (embedding-service orphaned from AgentDB).

## 6. Cross-Domain Dependencies

- **ruvector domain**: SONA, ruvector-gnn, nervous-system, cognitum-gate, HNSW patches all live in ruvector repo but have strong memory/learning relevance
- **agentdb-integration domain**: AgentDB core components overlap heavily — vector-quantization, LearningSystem, etc. exist in both domains
- **agentic-flow domain**: ReasoningBank, EmbeddingService, IntelligenceStore are shared
- **claude-flow-cli domain**: LocalReasoningBank (the only one that runs) lives there
- **ruvllm**: reasoning_bank.rs, micro_lora.rs, training pipeline

## 7. Knowledge Gaps

- ~1,000+ files still NOT_TOUCHED, mostly large JSON data files and test binaries
- AgentDB test files (ruvector-integration.test.ts, etc.)
- ruv-swarm-ml remaining: models/mod.rs (642 LOC), time_series/mod.rs (612 LOC), wasm_bindings/mod.rs
- ruv-swarm-persistence remaining: wasm.rs (694 LOC), migrations.rs (343 LOC)
- sublinear-time-solver: hybrid.rs (837 LOC), remaining orphaned solver files

## 8. Session Log

### R8 (2026-02-09): AgentDB core deep-read
7 files, 8,594 LOC. Established vector-quantization as production-grade, LearningSystem as cosmetic, CausalMemoryGraph as broken. Three fragmented ReasoningBanks discovered.

### R13 (2026-02-14): Rust source deep-reads (Phase C)
~40 files across SONA, ruvector-gnn, ruvector-core. SONA 85% production-ready. Hash-based embeddings confirmed in Rust.

### R19 (2026-02-14): Neural pattern recognition + Rust ML
13 files, ~14K LOC. Neural-pattern-recognition exposed as 80-90% facade. server.ts MCP 85-90% REAL — key discovery. Fake RNG pattern in Rust training.

### R21 (2026-02-14): neural-network-implementation + consciousness + persistence (Session 23)
~18 files. neural-network-implementation BEST CODE IN ECOSYSTEM (90-98%). Consciousness/psycho-symbolic files are elaborate facades. PI*1000.0 timestamp pattern discovered.

### R22b (2026-02-15): agentic-flow TypeScript source confirmation
LearningSystem and CausalMemoryGraph bugs confirmed as design flaws (not compilation artifacts). HyperbolicAttention correct in TS source — compilation degraded it. Hash-based embeddings systemic in 4 more files.

### R25 (2026-02-15): Broad deep-reads (Session 25)
34 files, 20,786 LOC, 59 findings. GOAP planner with broken A* stub. AgentDB dist files confirmed. ruvector-training.js BEST native integration. ruvector-backend.js is 85% simulation.

### R28 (2026-02-15): sublinear-rust deep-reads
3 files. sparse.rs 95% BEST matrix code. exporter.rs 88% with 7 export formats. hardware_timing.rs 55% — self-referential spin loops.

### R33 (2026-02-15): Python ML + psycho-symbolic MCP + MCP servers
19 files, ~17,927 LOC. Python GNN training real but uses only synthetic data. Psycho-symbolic MCP entirely theatrical. strange-loop-mcp.js genuine.

### R34 (2026-02-15): Sublinear solver core + matrix systems
10 files, ~6,200 LOC. Two incompatible matrix systems discovered. statistical_analysis.rs maintains neural-network-implementation quality level.

### R36 (2026-02-15): ruvector-nervous-system + neuro-divergent + HNSW patches
28 files, 26,569 LOC, 98 findings. Nervous-system 87.4% — BEST biological computing. neuro-divergent 88.5% production-quality. cognitum-gate 93% exceptional. ForecastingAdam innovation.

### R37 (2026-02-15): ruvllm LLM integration + novel crates
25 files, 30,960 LOC, 62 findings. Fourth ReasoningBank discovered (Rust). micro_lora.rs BEST learning code. Hash-based embeddings confirmed systemic in Rust too. prime-radiant sheaf-theoretic memory, temporal-tensor pattern tiering.

### R39 (2026-02-15): Sublinear core + emergence subsystem
7 files, 4,622 LOC, 33 findings. FALSE sublinearity confirmed — all algorithms O(n²)+. Emergence 51% fabricated metrics. WASM loaded but unused.

### R41 (2026-02-15): Consciousness layer + MCP tool layer
13 files, 9,772 LOC, 201 findings. Consciousness 79% genuine (vs emergence 51%). Goalie COMPLETE FACADE. psycho-symbolic-enhanced.ts BEST knowledge graph. Zero cross-cluster dependencies.

### R44 (2026-02-15): Sublinear server layer
3 files, 1,885 LOC, ~73 findings. Server infrastructure is PRODUCTION-GRADE but disconnected from computation. server/index.js (72%) has 90% real HTTP/WebSocket (Express+helmet+streaming+worker_threads) but 0% solver integration (calls non-existent createSolver) and 0% FlowNexus (600-line facade hitting fake endpoint). neural-pattern-recognition/server.js (42%) is a professional MCP facade — crashes on startup from missing adaptive-learning.js, simulated data via hashToFloat. strange-loop/mcp/server.ts (45%) has real WASM layer (R41: 92%) but broken import paths (wasm/ vs wasm-real/) and temporal_predict API mismatch. All 3 servers are isolated from each other (confirms R41 isolation pattern). Weighted avg ~53%.

### R43 (2026-02-15): AgentDB benchmarks + ReasoningBank demo + WASM tools + Rust benchmarks
5 files, 3,916 LOC, ~72 findings. ruvector-benchmark.ts (92%) is REAL production-grade benchmarking — validates R40's HNSWIndex finding. demo-comparison.ts (35%) is scripted marketing theater — real APIs but fake learning. wasm-sublinear-complete.ts (78%) has genuine algorithms but WASM is 100% theatrical — extends R39 "WASM unused" finding. baseline_comparison.rs (0%) never compiled. rustc_optimization_benchmarks.rs (15%) is most deceptive file in project — fabricated asymptotic claims. Two independent WASM facades confirmed in sublinear-time-solver.

### R46 (2026-02-15): Goalie deep-dive + neural-network benchmarks + integration verification
9 files, 6,496 LOC, 164 findings (33 CRIT, 44 HIGH, 43 MED, 44 INFO). DEEP files: 879→895. **Major R41 reversal**: goalie cli.ts (88-92%) proves internal engines ARE real — MCP handlers are facade layer, not the engines. agentic-research-flow-plugin (78%) has real Perplexity API. anti-hallucination (42%) has genuine algorithms but is dead code. Neural-network benchmarks are theatrical: standalone_benchmark (15-20%) spin-loop fabrication, system_comparison (42%) random gate simulation, strange-loops-benchmark (8-10%) tests wrong code entirely. flow-nexus.js (~70% quality) calls non-existent platform. bin/cli.js (72-78%) reverses R43's createSolver finding. consciousness-verifier (52%) lowers consciousness cluster to ~72-75%.

### R47 (2026-02-15): Consciousness verification + psycho-symbolic reasoner + neural pattern recognition
9 files, ~5,116 LOC, 219 findings (42 CRIT, 47 HIGH, 57 MED, 73 INFO). DEEP files: 895→913. **Consciousness BIMODAL**: MCP server (94.8%) PRODUCTION-QUALITY, but experiments (0-5%) COMPLETE FABRICATION — zero computation, documentation-as-code. Consciousness cluster drops from ~72-75% to ~60-65%. **Psycho-symbolic-reasoner first look**: "psycho" = pure branding. Real triple storage but reasoning = keyword matching. Planner wrapper real (58-62%) but A* is stub. 3rd theatrical WASM pattern. **Neural-pattern-recognition LAYERED**: logger (88-92%) GENUINE (reverses R45), CLI (62%) professional but broken backend, zero-variance-detector (42-48%) NOVEL ANTI-PATTERN: real algorithms fed fabricated data. Three new anti-patterns catalogued: specification-as-implementation, real-algorithms-on-fake-data, production-facade.

### R49 (2026-02-15): Consciousness final sweep + neural-pattern-recognition completion + ReasoningBank WASM
9 files, ~4,726 LOC, ~180 findings. DEEP files: 913→922. **Consciousness arc CLOSED** (R41→R46→R47→R49): cluster drops to ~55-60%. genuine_consciousness_detector.ts (75-82%) has real crypto infra but theatrical verification + orphaned test subjects. temporal_consciousness_validator.rs (60%) = real orchestration, theatrical "theorems". mcp_consciousness_integration.rs (12-18%) = COMPLETE MCP FACADE (80-point gap vs MCP server). consciousness_optimization_masterplan.js (25-30%) = orphaned orchestrator. validators.js (78%) = real IIT 3.0 but orphaned. **Neural-pattern-recognition COMPLETE**: subsystem rises from ~64% to ~72%. real-time-monitor (88-92%) GENUINE, statistical-validator (82-88%) GENUINE 5 real tests, signal-analyzer (72-76%) genuine DSP core with consciousness facade. **MAJOR REVERSAL**: reasoningbank_wasm_bg.js (100%) is GENUINE wasm-bindgen — NOT the 4th facade. 206KB binary verified, all 6 APIs traced to Rust source. Gold standard WASM. Two new anti-patterns: MCP Integration Facade, Duplicate Implementation Abandonment.

### R50 (2026-02-15): Goalie security + anti-hallucination + ReasoningBank Rust storage
4 files, 2,036 LOC, ~40 findings. **Goalie reversal STRENGTHENED**: ed25519-verifier.ts (88-92%) is production crypto via @noble/ed25519, active in tools.ts pipeline. perplexity-actions.ts (93-96%) is GENUINE Perplexity API — two endpoints, rate limiting, 4-action GOAP pipeline, REVERSES R41. anti-hallucination-plugin.ts (55-60%) BIMODAL: execute() has real per-claim Perplexity verification (75-80%), hooks layer is keyword matching with Math.random() (30%). Combined with R46, goalie is now "MCP facade over genuine crypto + API + plugin internals." **ReasoningBank Rust storage GENUINE**: sqlite.rs (88-92%) has rusqlite with WAL mode, RAII connection pool, FTS5 full-text search, schema migrations. Completes R49's WASM picture — Rust ReasoningBank has genuine WASM+SQLite substrate but only pattern persistence implemented (missing trajectories/verdicts).

### R53 (2026-02-16): MCP tools layer + strange-loop runtime + core optimizers
9 files, 4,693 LOC, 179 findings (28C, 49H, 47M, 55I). DEEP files: 955→964 (970 counting R52 parallel). **MCP tools layer BIMODAL**: domain-registry (88-92%) and domain-management (82%) are production-quality reasoning domain system with 12 builtin domains, dependency DAG, EventEmitter lifecycle — but IN-MEMORY ONLY (zero persistence). scheduler.ts (18-22%) is THEATRICAL FACADE — hardcoded "11M tasks/sec" and "<100ns" using Date.now()×1M. psycho-symbolic-dynamic.ts (28%) confirms goalie pattern — DomainRegistry initialized then ignored (console.log placeholder). 4th "real infra, placeholder integration" occurrence. **Strange-loop runtime NON-FUNCTIONAL**: JS MCP server (20-25%) has correct @modelcontextprotocol/sdk architecture but SAME broken WASM import as R44. CLI (68%) = production Commander.js presentation + broken WASM integration (INVERTED goalie pattern). purposeful-agents.js (45-55%) = marketing demo with Math.random() behaviors. Consciousness arc stays ~55-60%. **Core optimizers GENUINE**: optimized-matrix.ts (85-88%) confirms 3RD INDEPENDENT MATRIX SYSTEM — TS CSR+CSC with Float64Array, streaming LRU, vector pooling. performance-optimizer.ts (88-92%) is GENUINE auto-tuning with empirical benchmarking — STARK CONTRAST to R43 deception. BEST optimizer in project.

### R55 (2026-02-16): Temporal Nexus quantum + psycho-symbolic internals + cross-package validation
12 files, 7,623 LOC, 148 findings (20C, 38H, 36M, 52I+2L). DEEP files: 970→990 (with R52/R54 parallel). **Temporal Nexus quantum (80.75% avg)**: GENUINE physics — decoherence.rs (78-82%) implements Lindblad master equation with T1/T2 timescales, physics_validation.rs (88-92%) BEST-IN-CLASS CODATA 2018 validation (1e-42 precision), visualizer.rs (85-88%) production ASCII rendering. Follows R47 BIMODAL pattern (infra 85-92%, consciousness constants arbitrary). Separate from ruQu (error correction vs constraint modeling). **Psycho-symbolic Rust 3-4x better than TS**: emotions.rs (85-88%) genuine Plutchik lexicon vs psycho-symbolic-dynamic.ts (28%). performance_monitor.rs (88-92%) uses Instant::now() vs scheduler.ts (18-22%) hardcoded claims. mcp_overhead.rs (78-82%) genuine criterion harness but MOCK MCP calls. ALL 3 ORPHANED from each other and reasoning pipeline. **Cross-package KEY FINDINGS**: ruvector-integration.test.ts (95-98%) is AUTHORITATIVE architecture spec — R20 ROOT CAUSE PROOF (tests expect real embeddings, factory defaults to mock is BUG not design). cascade.rs (65-70%) genuine Fahlman & Lebiere algorithm with stubbed backprop. real-implementation/lib.rs (92-95%) CONFIRMS R23 — Kalman+neural+solver gate is genuinely "real". fast_sampling.rs (88-92%) extends R39 FALSE sublinearity (correct O(n) algorithms with "sublinear" claims). persistence.js (95-98%) PRODUCTION SQLite — 8 tables, TTL, confirms R45/R50.

### R56 (2026-02-16): Solver algorithms + neural benchmarks + temporal nexus extension + core types
10 files, ~4,621 LOC, ~162 findings. DEEP files: 990→1,000. **Sublinearity SPLIT VERDICT**: backward_push.rs (92-95%) is GENUINELY O(1/epsilon) sublinear — REVERSES R39 blanket finding. Real Andersen et al. 2006 backward PageRank, priority queue, bidirectional solver. GOLD STANDARD for algorithmic honesty. BUT random_walk.rs (82-86%) is O(n) Monte Carlo with genuine variance reduction, and optimized_solver.rs (72-76%) is standard CG+SIMD — NO solver dispatcher, no algorithm integration. Repo has ONE real sublinear algorithm among standard linear solvers. **Neural benchmarks THEATRICAL**: standalone_benchmark.rs (8-12%) is MOST DECEPTIVE IN PROJECT — spin-loop to hardcoded latencies, predetermined "breakthrough", zero crate imports. latency_benchmark.rs (72-78%) = genuine criterion + thread::sleep(). **BUT fully_optimized.rs (96-99%) is HIGHEST OPTIMIZATION IN PROJECT** — INT8 quantization, AVX2/AVX-512, inline assembly, CPU core pinning. The ANTI-FACADE. rust_integration.rs (85-88%) MISLABELED — zero HuggingFace despite directory name. **Temporal nexus**: metrics_collector.rs (40-45%) = real aggregation framework + random() data sources (4th instance). temporal_window.rs (88-92%) = production sliding window, orphaned from quantum physics. **types.rs** (85-90%) = clean metadata, stringly-typed algorithm dispatch.

### R57 (2026-02-16): ReasoningBank DB + server infrastructure + emergence persistence
6 memory files, ~2,809 LOC, ~109 findings. **ReasoningBank queries.ts (85-90%) PRODUCTION-READY**: 7-table schema implementing MaTTS (Memory-Augmented Test-Time Scaling) — patterns, embeddings, links, trajectories, matts_runs, consolidation, metrics. Parameterized SQL, zero injection. R20 ROOT CAUSE CONFIRMED: fetchMemoryCandidates() JOINs pattern_embeddings but upsertMemory() NEVER calls upsertEmbedding() → table EMPTY. FOURTH DISCONNECTED DATA LAYER (.swarm/memory.db separate from AgentDB's 3 layers). reasoningbank-optimize.js (40-45%) is script generator NOT optimizer — fabricated "2000x faster" claims, only VACUUM/ANALYZE genuine. **Server infrastructure BIMODAL**: streaming.js (85-90%) is GENUINE async generator worker pool (0% HTTP, 4th mislabeled file). session-manager.js (58-62%) has solid lifecycle (75-80%) but solver integration theatrical (Math.random() verification). **Emergence persistence CLARIFIED**: persistent-learning-system.ts (55-60%) has GENUINE file I/O persistence (95%) + RL/forgetting curve (85-90%), but ALL 6 pattern detection functions return empty arrays (0-5%). R39 "51% fabricated" refers to pattern detection, NOT persistence layer.

### R58 (2026-02-16): Psycho-symbolic MCP + temporal-lead-solver + Goalie reasoning
9 files, ~3,754 LOC, ~153 findings. DEEP files: 1,010→1,019. **Psycho-symbolic MCP SPLIT**: text-extractor.ts (88-92%) REVERSES theatrical WASM pattern — first GENUINE WASM integration calling real Rust NLP (1,076 LOC across sentiment/preferences/emotions extractors). memory-manager.ts (25-30%) is 5th MISLABELED FILE (zero WASM memory ops, TS object registry). server.ts (72-76%) genuine @modelcontextprotocol/sdk with 5 specialized tools but HTTP/SSE TODOs. patterns.rs (85-90%) genuine regex extraction confirming R55 Rust 3-4x quality gap. Crate revised upward (~55-60% from ~48-52%). **temporal-lead-solver DISCOVERED**: predictor.rs (92-95%) is 2ND GENUINE SUBLINEAR algorithm — O(√n) functional prediction via forward-backward push (Kwok-Wei-Yang 2025), randomized coordinate sampling, backward random walks. REVERSES R39 alongside R56. solver.rs (68-72%) BIMODAL — 5 algorithms but backward_push is STUB, forward_push O(n²) worst case, random walk has hidden 100x constant. **Goalie reasoning CONFIRMED GENUINE**: advanced-reasoning-engine.ts (75-80%) BIMODAL — WASM 0% (never initialized) but fallback NLP (5 domain detectors, temporal analysis, complexity scoring) is real heuristic reasoning. ed25519-verifier-real.ts (82-88%) is GENUINE @noble/ed25519 crypto for anti-hallucination citation signing with certificate chains but hardcoded example root keys. self-consistency-plugin.ts (78-82%) has REAL Perplexity API multi-sampling (3 temperatures) but majority voting returns first sample (stub).

### R59 (2026-02-16): Benchmark archaeology + JS solver layer + matrix system + ruv-fann benchmarking
8 memory files, ~3,910 LOC, ~155 findings. DEEP files: 1,019→1,027. **Benchmark deception boundary MAPPED**: criterion-based benches/ suite is 88-95% GENUINE — performance_benchmarks.rs (88-92%) gold standard with 8 real solver benchmarks, solver_benchmarks.rs (88-92%) ANTI-FACADE with 5 real solver APIs, throughput_benchmark.rs (91-94%) confirms R23 BEST IN ECOSYSTEM with real neural forward passes. performance-benchmark.ts (92-95%) confirms R43 genuine TS benchmark pattern. Deception is in STANDALONE benchmarks (8-25%), NOT criterion suites. **JS solver layer**: solver.js (88-92%) is 100% WASM bridge (OPPOSITE of R57 mismatch — breaks without WASM). fast-solver.js (85-88%) is genuine CG+CSR with typed arrays but WASM 0% (R57 pattern). bmssp-solver.js (70-75%) is INVENTED ALGORITHM — genuine Dijkstra+PriorityQueue applied to wrong problem, nonsensical solution formula. TWO opposite WASM failure modes in same directory. **4th matrix system CONFIRMED**: matrix-utils.js (92-95%) implements Dense+COO with genuine SPD generation, conditioning analysis, diagonal dominance strategies. Zero integration with 3 other matrix systems.

### R62 (2026-02-16): Sublinear-Rust root architecture + core algorithms
10 files, ~3,597 LOC, 129 findings (15C, 31H, 32M, 51I). DEEP files: 1,047→1,066. **3RD GENUINE SUBLINEAR CONFIRMED**: forward_push.rs (92-95%) is O(volume/epsilon) with proper residual bounds, degree-weighted thresholds, adaptive work queue. Complements backward_push.rs — both directions now production-quality. Genuine count: **3** (backward_push, predictor, forward_push). **4th FALSE SUBLINEARITY**: johnson_lindenstrauss.rs (72-76%) has correct JL math but O(n*d*k) total, broken RNG (uniform not Gaussian), incorrect pseudoinverse. **Crate root architecture EXPOSED**: lib.rs (68-72%) reveals 4 parallel solver APIs with zero unification, best algorithms ORPHANED from public API, consciousness 25% of surface. error.rs (92-95%) EXCEPTIONAL — intelligent recovery system with algorithm-specific fallback chains, genuine WASM errors, no_std. Top 5% quality. **solver_core.rs (38-42%) is 7th MISLABELED FILE** — zero dispatcher despite name, just 2 standalone solvers (CG+Jacobi), CG is DUPLICATE of optimized_solver.rs. **Math infrastructure**: graph/mod.rs (75-80%) textbook CSR, WorkQueue supports push algorithms, 6th+ matrix system. simd_ops.rs (82-86%) genuine SIMD via `wide` crate but ORPHANED from fully_optimized.rs (two-tier SIMD). **Outer layer BIMODAL**: convergence-detector.js (88-92%) genuine convergence math (validates R60). MCP solver.ts (82-86%) GENUINE with 3-tier fallback cascade — first OptimizedSolver integration evidence. advanced-reasoning-engine.ts (0-5%) COMPLETE THEATRICAL — zero inference, all MCP delegation with mock data. Confirms R61 ReasonGraph bimodal.

### R60 (2026-02-16): WASM pipeline complete + sublinear core algorithms + convergence metrics
10 files, ~4,140 LOC, ~120 findings. DEEP files: 1,027→1,037. **WASM VERDICT REVERSED**: 4 WASM files form complete pipeline — wasm-solver/lib.rs (85-88%) genuine wasm-bindgen CG but ORPHANED (zero crate imports), wasm_iface.rs (90-93%) PRODUCTION FFI bridge (zero-copy Float64Array, serde-wasm-bindgen, solve_stream), wasm.rs (88-92%) genuine bindings with web_sys browser APIs, math_wasm.rs (68-72%) naive Dense math, 5th matrix system. WASM scoreboard: **6 genuine vs 4 theatrical (60%)**. BUT architecture mismatched — genuine WASM wraps basic CG/Neumann, not project's best algorithms (backward_push, fully_optimized). **Core TS solver**: optimized-solver.ts (78-82%) BIMODAL — vectorized Neumann 92-95% genuine, 3/4 variants stubs, ZERO WASM (pure-JS fallback). Different algorithm from Rust CG. memory-manager.ts (88-92%) is 6th MISLABELED FILE — genuine TypedArrayPool+LRU+SIMD-alignment BUT zero WASM linear memory. matrix.ts (85-88%) is 5th+ matrix system (COO+Dense arrays, incompatible with same-directory optimized-matrix.ts TypedArrays). **Algorithm**: sublinear_neumann.rs (45-50%) is 3rd FALSE SUBLINEARITY — correct Neumann math BUT O(n²) full matrix extraction. temporal-compare/sparse.rs (75-80%) is NAMING MISMATCH — implements neural lottery ticket pruning, NOT sparse matrices. metrics-reporter.js (88-92%) REVERSES theatrical metrics pattern — zero Math.random(), all data from real convergenceDetector pipeline.


### R64 (2026-02-16): Sublinear-Rust Architecture Completion + Temporal-Lead Core
10 files, ~2,801 LOC, 67 findings (7C, 24H, 30M, 4I, 2L). DEEP files: 1,066→1,089. **Cluster A (Graph+Matrix)**: adjacency.rs (75-80%) implements Graph trait with dual forward+reverse adjacency lists (enables backward push, doubles memory). Normalization bug (line 107-130) breaks degree() queries. PushGraph wraps CSR — triple conversion pipeline. optimized_csr.rs (0%) is **ORPHANED DEAD CODE**: doesn't compile (non-existent CSRStorage::with_capacity()), not in mod.rs, zero usages. 7th independent CSR. Duplicates simd_ops.rs SIMD. DELETE recommended. **Cluster B (Core Infra)**: wasm-integration.ts (85-88%) is **7th GENUINE WASM** — real WebAssembly.instantiate with wbindgen imports, proper heap allocation/deallocation, 2-tier fallback (WASM→optimized JS). Loop-unrolled JS fallback matches attention-fallbacks quality. Genuine temporal neural acceleration with real physics (299.792458 km/ms). **WASM scoreboard: 7 genuine vs 4 theatrical (64%)**. reasongraph/index.ts (15-20%) is **THEATRICAL** — orchestration facade with fabricated claims ("658x speed of light", "87% consciousness verification"). 4 isolated components with zero cross-coordination. Tests have zero assertions. Delegates all reasoning to 0-5% theatrical AdvancedReasoningEngine. types.ts (70-75%) pure type definitions (188 LOC) — SolverConfig, Matrix union (coo/csr/csc/dense), 5 MCP param types, SolverError class. Error system ORPHANED from Rust error.rs. Matrix types isolated from 6+ other systems. **Cluster C (MCP+NPR+Worker)**: temporal.ts (~85%) **GENUINE MCP** — 4 tools wrapping temporal-lead-solver, matches R62 solver.ts quality. pattern-detection-engine.js (10%) is **BROKEN FACADE** — all 3 detection calls use wrong API names (processData/analyzeEntropy/analyzeSequences vs actual APIs), will throw "not a function" at runtime. Dead code — real-time-monitor reimplements independently. solver-worker.js (92-95%) **REVERSES R44** — production worker thread genuinely invoking streamSolve(), auto-restart on fatal errors. **Cluster D (Temporal-Lead-Solver)**: core.rs (88-92%) high-quality math primitives — Matrix (ndarray), Vector, spectral radius (power iteration), Complexity enum. Matches predictor.rs quality. Sparse path (sprs) orphaned — solvers use dense only. physics.rs (80%) genuine relativistic math (95% — accurate speed of light, Lorentz factor, time dilation) with theatrical "FTL" framing (30% semantic honesty). Validator deliberately confesses deception. Consistent with R55 Temporal Nexus pattern.

- **R65** (2026-02-16): 10 files, ~2,697 LOC, 115 findings. R20 ROOT CAUSE ARC COMPLETE. HybridBackend REVERSES disconnected AgentDB. WASM scoreboard 7:5 (58%). DEEP: 1,069→1,089
- **R66** (2026-02-16): 5 memory-and-learning files (of 10). ensemble.rs 85-90% GENUINE AdaBoost+bagging. mlp.rs 62% BIMODAL (training broken). graph.rs 88-92% 9th graph system. DEEP: 1,089→1,110
- **R67** (2026-02-16): 11 files, ~2,136 LOC. ReasoningBank Rust workspace GENUINELY ARCHITECTED. 5th disconnected data layer. quic.rs 96% vs swarm_transport.rs 28-32%. DEEP: 1,089→1,100
- **R68** (2026-02-16): 10 files, ~2,560 LOC, ~120 findings. ReasoningBank core TYPE-ALGORITHM GAP. similarity.rs 12th hash-like. MLP TRIMODAL (mlp_optimized 92-96% BEST). AVX-512 REAL. DEEP: 1,100→1,110
- **R69** (2026-02-16): 10 files, ~2,475 LOC, 131 findings. Quantum gradient (mod.rs 92-95% vs entanglement.rs 67-73%). graph_reasoner COMPLETE. GHOST WASM (27 models never compiled). 3rd persistence layer. DEEP: 1,110→1,130
- **R70** (2026-02-16): 9 memory files (of 10), ~1,175 LOC, 80 findings (13C, 21H, 25M, 21I). **CONNECTED tier cleared**. sublinear/mod.rs (2-5%) SMOKING GUN — deliberately orphans 3 genuine sublinear algorithms, exposes 4 false ones. src/core.rs (0%) dead 5th unified API. temporal-lead-solver/lib.rs (92-95%) REVERSES orphaning — BEST crate root. temporal-solver.rs (82-86%) imports WRONG solver. extractors/lib.rs (90-95%) clean WASM facade. wasm-loader-simple.ts (82-85%) 8th genuine WASM. temporal_nexus core (88-92%) genuine hardware TSC. dashboard (78-82%) 1000x granularity mismatch. neural-pattern-recognition/index.js (22-28%) phantom export facade. DEEP: 1,130→1,140

### R71 (2026-02-16): ReasoningBank adaptive learning + temporal solvers + integration + memory systems
9 files, ~2,896 LOC, ~55 findings (9C, 11H, 2M, 1I). DEEP: 1,140→1,149. **ReasoningBank adaptive.rs (42-48%)**: WRITE-ONLY learning — stores patterns but NEVER updates engine state, weights, or parameters. TYPE-ALGORITHM GAP: interface suggests adaptive learning, implementation is offline pattern averaging. Missing: learning rate scheduling, EWC++, online learning, confidence calibration. Hardcoded LearningConfig weights (0.3/0.5/0.2). **Temporal solvers BIMODAL**: optimized.rs (91-94%) GENUINE AVX2 SIMD (_mm256_fmadd_ps) for 6-stage UltraFastTemporalSolver pipeline, P99.9 benchmarking. validation.rs (55%) BIMODAL — documentation 95%, implementation 15% (proof validators always return true). wasm.rs (88-92%) 9th GENUINE WASM exposing O(√n) predictor, REVERSES R62 orphaning. ultra_fast.rs (72%) IS UltraFastCG (standard CG O(κn), NOT sublinear) with FABRICATED Python benchmarks. Confirms R70 temporal-solver.rs bifurcation — two parallel temporal solvers. **Integration**: intent.rs (94%) MISLABELED — real Ed25519 crypto authorization but zero semantic intent parsing. **Memory systems**: queries.js (.swarm) 100% — 8-table schema (patterns/pattern_embeddings/links/trajectories/matts_runs/consolidation/metrics/sessions), REAL Float32Array embeddings (384-dim MiniLM-L6-v2, NOT hash-based!), 4TH DISCONNECTED DATA LAYER. AdvancedMemory.ts (62-68%) orchestration facade — 5TH DISCONNECTED DATA LAYER (SharedMemoryPool), episodic replay = heuristic thresholds, consolidation offline-only. scheduler_tool.rs (68-72%) BIMODAL — genuine MCP lifecycle (85-90%) with empty task closures. WASM scoreboard: **9 genuine vs 5 theatrical (64%)**

### R72 (2026-02-16): ReasoningBank TS core operations
5 files, ~1,231 LOC, ~68 findings. DEEP: 1,150→1,160. **ReasoningBank TS core is GENUINE**: matts.ts (85-88%) implements real Memory-Aware Test-Time Scaling (Algorithm 5) with parallel/sequential modes and self-contrast LLM aggregation. consolidate.ts (82-86%) has production LSH bucketing O(n·k) and outcome-based contradiction detection. distill.ts (78-82%) LLM-based extraction with sophisticated prompts. **But ZERO Rust integration** — TS uses LLM, Rust uses gradient descent. **backend-selector.ts (78-82%) BYPASSES HybridBackend** — auto-selector imports old index.js, NEVER routes to R65's unified implementation. 5th disconnected layer confirmed at API level. **embeddings.ts (78-82%) BEST embedding impl** — fixes R51 race conditions, real transformers.js, but 13th hash-based fallback. NPX env auto-disables real embeddings. **6th routing system** (ModelRouter in matts.ts). **TWO parallel consolidation systems** (consolidate.ts vs AdvancedMemory.autoConsolidate()).


### R73 (2026-02-16): ReasoningBank embeddings architecture
1 file, 114 LOC, 10 findings (2C, 4H, 3M, 1I). DEEP: 1,160→1,170. **embeddings.js: 14th hash-based embedding DISCOVERED**. THREE independent embedding implementations now: (1) embedding-service.ts (R51 orphaned), (2) distill.ts (real transformers.js), (3) embeddings.js (both real OpenAI + sin/cos hash fallback). embeddings.js architecture: DEFAULT fallback to hashEmbed() on two paths: (a) config.embeddings.provider mismatch (line 30, default behavior), (b) OpenAI API error (silent fallback, line 55). Hash quality is POOR — 32-bit DJB2 scalar → sine/cosine waves, collision-prone. simpleHash() vulnerable to similar-text collision (short strings amplified by phase-shifted wave generation). No error propagation to caller — silent degradation. **Missing integration**: (1) NO import of distill.ts despite R72 context, (2) NO import of embedding-service.ts (R20 ROOT CAUSE). 10th disconnected embedding system. **OPENAI_API_KEY security**: read directly from process.env, no validation, plain Authorization header, fallback is silent (line 48 console.warn under try/catch only). **Cache management**: unbounded Map with setTimeout TTL (lines 39-40), no max-size LRU, memory leak under high throughput. normalize() edge case (line 88): mag===0 returns unnormalized vector. Confirms R72 finding: ReasoningBank TS core operates in isolated domain, zero unification of embedding systems. CONTEXT: queries.js (R71) has real 384-D embeddings, distill.ts has transformers.js loader, embeddings.js has hash fallback — three parallel systems serving same domain.

### ReasoningBank Hooks

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| post-task.ts | agentic-flow | 128 | 78-82% | DEEP | **ARCHITECTURAL MISMATCH**. Implements complete 3-phase pipeline (judge→distill→consolidate) with proper LLM evaluation, memory extraction, and consolidation. BUT hook-handler.cjs post-task handler (line 140-147) bypasses entire logic, only calling intelligence.feedback stub. Trajectory never reaches post-task.ts. Missing matts.ts integration despite distilling memories — parallel to judge.ts but orphaned from post-task. 5th disconnected persistence layer discovered (patterns/pattern_embeddings tables, separate from matts.ts trajectories). Consolidation LSH bucketing genuine but lacks circuit breaker. No trajectory JSON schema validation. --task-id CLI injection risk. Default enabled (enable_post_task_hook=true) | R73 |
| judge.js | agentic-flow | 129 | 54% | DEEP | BIMODAL: LLM-as-Judge genuine (Algorithm 2, 78-82%) — real Anthropic API, structured JSON verdict (label/reasoning/confidence/suggestions). BUT bypasses MCP ReasoningBank store tool (R67: 72%), creating parallel data flow. CRITICAL SECURITY: API key passed directly in fetch headers without TLS pinning or rate limiting. No retry logic, no circuit breaker. heuristicJudge fallback (40%): tokenCount formula flawed (1-exp(-tokens/1000)), noFailures check is boolean not ratio. Orchestration correct, integration broken | R73 |
| wasm-adapter.ts | agentic-flow | 170 | 92-95% | DEEP | **10TH GENUINE WASM**. Real wasm-bindgen bridge to Rust ReasoningBank workspace (R67: core 88-92%, storage 94%, learning 95-98%). 6 APIs: storePattern/getPattern/searchByCategory/findSimilar/getStats/constructor. WASM scoreboard: 10 genuine vs 5 theatrical (67%). wasmMemory field declared but UNUSED — zero linear memory management. LocalStorageAdapter fallback for browser. Production-quality singleton with init guard, proper error propagation | R73 |
| index.ts (ReasoningBank) | agentic-flow | 136 | ~75% | DEEP | **FOUR-TIER LAYERING**. Barrel export BYPASSES backend-selector — hardcodes HybridReasoningBank class + 11 modular functions (storePattern/retrieve/judge/distill). Dual API surface (class + functions) creates confusion. Runtime AgentDB patch (lines 128-130) timing risk — replaces window/global.AgentDB AFTER import, may race with early callers. Clean module re-export (85-90%) but architectural bypass (-10%). v1.7.1 annotation | R73 |
| index.js (ReasoningBank) | agentic-flow | 101 | 100% | DEEP | **FOURTH REASONINGBANK LAYER**. Pure JS barrel (v1.0.0 legacy) with 11 real exports: storePattern/searchPatterns/consolidateMemories/judgeTrajectory/distillMemories/matts/benchmark/queries/db/config. Zero hash embeddings (handled by embeddings.js). Clean dependency graph. backend-selector.ts routes HERE (not to HybridBackend). Chronologically first implementation (v1.0.0), still active in prod | R73 |
| config.ts | agentic-flow | 241 | ~80% | DEEP | Authoritative ReasoningBank config. 8 subsections (embeddings/consolidate/judge/distill/enable flags/SQLite paths/thresholds/models). **EMBEDDING PROVIDER MISMATCH** (lines 30-42): accepts "claude"|"openai" but hardcodes wrong model for cross-provider (claude model for openai provider). NO db_path config for SQLite despite 7-table schema existing. claude-sonnet-4.5 hardcoded for judge/distill. enable_post_task_hook defaults true. Type-safe ReasoningBankConfig interface | R73 |
| config.js | agentic-flow | 170 | 82-85% | DEEP | Hardcoded claude-sonnet models everywhere (judge/distill/matts). distillation disabled by default (enable: false). matts.mode = "sequential" BUT consolidate.ts implements LSH (parallel mode exists, unused). consolidate.lsh_bands formula (Math.floor(Math.sqrt(embedding_dim)) = 19 bands for 384D) reasonable. SCHEMA ORPHANED: queries.ts has 7-table schema but config has zero db_path | R73 |
| embeddings.js | agentic-flow | 114 | ~60% | DEEP | **14TH HASH-BASED EMBEDDING**. Two providers (openai/claude) both default to hashEmbed() on error. simpleHash() produces 32-bit scalar → sine/cosine waves with phase offsets (NOT collision-resistant). 3RD independent embedding system in ReasoningBank (vs embedding-service.ts R51, embeddings.ts R72). Zero transformers.js integration. Zero code sharing. Feature-flag controlled (config.embeddings.use_hash_fallback). API error silently falls to hash (lines 51-56), no error propagation to caller | R73 |
| pii-scrubber.js | agentic-flow | 99 | 88% | DEEP | Genuine PII utility. 14 regex patterns: email, phone (US/intl), SSN, credit card (Luhn validation), IPv4/v6, address, names, DOB, passport, driver license, bank account, crypto wallet. scrubText() replaces with [REDACTED-TYPE]. Feature-flag controlled (config.enable_pii_scrubbing). No EU GDPR patterns (IBAN, VAT, national IDs). Hardcoded US bias | R73 |
| mmr.ts | agentic-flow | 80 | 90%+ | DEEP | **GENUINE MMR ALGORITHM** (Carbonell & Goldstein 1998). Correct cosine similarity (lines 20-30). Iterative greedy selection balancing relevance (lambda) vs diversity (1-lambda). Proper normalization. Used by consolidate.ts for memory deduplication. maxResults clamp. Zero MMR citations despite academic correctness. Production-ready | R73 |
| schema.ts | agentic-flow | 89 | 92-95% | DEEP | **TYPE-SAFE BRIDGE** mirroring queries.ts 7-table schema exactly. 7 interfaces (Pattern, Embedding, Link, Trajectory, MattsRun, ConsolidationRun, Metric). Optional fields match SQL nullability. TypeScript-first design (zod validation would strengthen). Confirms R57 queries.ts PRODUCTION-READY status. Zero runtime behavior, pure types | R73 |
| schema.js | agentic-flow | 5 | 0% | DEEP | **EMPTY STUB**. No content. Placeholder for future JS schema validation | R73 |

## 3. Findings Registry

### R73: post-task.ts Hook Integration (11 findings)

**CRITICAL:**
- **C1**: Hook registration mismatch (lines 62-75). post-task.ts implements complete ReasoningBank pipeline (judge→distill→consolidate Algorithm 2-4) but hook-handler.cjs post-task handler (line 140-147 of hook-handler.cjs) ONLY calls intelligence.feedback stub. Entire ReasoningBank logic bypassed. Trajectory JSON never reaches post-task.ts execution.
- **C2**: No matts.ts integration (lines 77-128). post-task.ts distills memories via distillMemories() but never calls matts.ts (Algorithm 5, Test-Time Scaling). matts.ts parallel/sequential rollout modes orphaned from post-task hook, contradicting ReasoningBank paper design.

**HIGH:**
- **H1**: Trajectory input fragile (lines 37-55). loadTrajectory() accepts file path OR stdin JSON but has ZERO schema validation. Parses data.trajectory?.steps || [] without checking step structure (step type, metadata, completion flags). Silent failures in judge.ts downstream if steps lack required fields.
- **H2**: Missing context parameters (lines 91-101). distillMemories() called with agentId: "unknown" and domain: undefined. Loses agent identity and domain context for domain-specific memory classification. Query/trajectory context inferred generically, not domain-parameterized.
- **H3**: CLI injection risk (lines 22-25, 37). parseArgs() naively strips `--` prefix without validation. --task-id passed to distillMemories without sanitization. No path traversal check on --trajectory-file argument.

**MEDIUM:**
- **M1**: Config feature flag unchecked (lines 62-75). enable_post_task_hook defaults to true but never validated against consolidate/judge config consistency. Consolidation trigger_threshold=20 items may never fire if judge produces low-confidence verdicts (<0.5), creating queue buildup.
- **M2**: 5th disconnected persistence layer (lines 77-128). post-task→distill→db.storeMemory writes to patterns/pattern_embeddings tables. consolidate.ts deduplicates patterns table, orphaned from matts.ts trajectories table. Two parallel memory storage systems, zero synchronization.
- **M3**: Consolidation blocking (lines 103-120). consolidate() always runs if threshold met with NO circuit breaker or health check. 4-phase algorithm (dedup LSH bucketing + contradict + prune + metric logging) can block task completion for 1000+ items. No timeout, no async queue.
- **M4**: Shebang mismatch (lines 1-20). File has #!/usr/bin/env node shebang but imported as ES module (import statements with .js extensions). CLI help string doesn't explain trajectory JSON schema or expected metadata structure. No example usage beyond --help.

**INFO:**
- **I1**: ModelRouter usage (lines 63-75). judgeTrajectory uses ModelRouter (parallel routing to ADR-008/LLMRouter/RuvLLMOrchestrator). Config defaults to claude-sonnet-4.5. No actual routing logic — hard model selection despite router abstraction.
- **I2**: Template-based fallbacks (lines 91-101). distill-success.json vs distill-failure.json templates loaded conditionally on verdict.label. Fallback to heuristicJudge/templateBasedDistill exists but never documented. No missing-template error handling.

### Positive Findings

**R72 confirms**:
- **matts.ts is genuine** (85-88%). Memory-Aware Test-Time Scaling correctly implements parallel rollouts with self-contrast aggregation, sequential refinement with check-and-correct, and LLM-based result aggregation. Should be called from post-task.ts but is orphaned.
- **consolidate.ts is GENUINE** (82-86%). LSH bucketing implementation correct, batch queries, deduplication proper. Lacks matts.ts integration though.

## 4. Positive Findings

### Production-Ready Code
- **consolidate.ts** (82-86%): LSH bucketing correctly reduces O(n²) to O(n·k) where k = bucket size. Batch embedding queries efficient. Deduplication chain proper.
- **matts.ts** (85-88%): Memory-Aware Test-Time Scaling fully implemented. Parallel rollout mode with self-contrast aggregation using LLM evaluation. Sequential refinement correct. Should be invoked from post-task hook.
- **judge.ts** (78-82%): LLM-as-Judge correct, uses ModelRouter for multi-provider failover, heuristic fallback implemented.
- **distill.ts** (78-82%): LLM-based knowledge extraction with conditional prompt templates (success/failure). Template-based fallback reasonable.

### Architectural Correctness
- **ReasoningBank TS core** (72-88% weighted): Algorithms 2-5 (Judge, Distill, Consolidate, MaTTS) are mathematically correct implementations of ReasoningBank paper.
- **HybridBackend** (R65): Genuinely unifies 4 disconnected AgentDB layers via SharedMemoryPool. Real production integration.

### R77 Positives
- **utils.ts** (92-95%): Production-grade utilities — VectorOperations (15 methods, dot/cosine/normalize), PerformanceMonitor (real timing), ConvergenceChecker, TimeoutController, ValidationUtils. No hash-based embeddings. Best infrastructure code in src/core/.
- **temporal-attractor.ts** (88%): Genuine nonlinear dynamics MCP schemas — Rosenstein algorithm for Lyapunov exponents, Takens embedding theorem parameters, Echo-State Networks with spectral radius. Pure schema layer with proper delegation to handlers.

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

## 6. Cross-Domain Dependencies

(Updated R73: ReasoningBank outer ring exposes 4-tier layering + 3 independent embedding systems)

**ReasoningBank TS Architecture (R73)**:
- `index.ts` →[hardcodes]→ HybridReasoningBank (BYPASSES backend-selector.ts)
- `index.ts` →[patches]→ window/global.AgentDB (runtime timing risk)
- `index.js` ←[targeted_by]→ backend-selector.ts (NOT HybridBackend)
- `wasm-adapter.ts` →[bridges]→ Rust ReasoningBank workspace (R67: core/storage/learning/mcp)
- `embeddings.js` →[3rd_system]→ NO integration with embedding-service.ts (R51) or embeddings.ts (R72)
- `config.ts` →[orphaned_from]→ queries.ts 7-table schema (no db_path)
- `schema.ts` →[mirrors]→ queries.ts schema exactly

**Algorithm Pipeline (R72+R73)**:
- `post-task.ts` →[imports]→ judge.js, distill.ts, consolidate.ts
- `post-task.ts` →[missing]→ matts.ts (should integrate Test-Time Scaling)
- `post-task.ts` →[registered_in]→ hook-handler.cjs (which has STUB implementation)
- `judge.js` →[bypasses]→ MCP ReasoningBank store (creates parallel data flow)
- `judge.js` →[calls]→ Anthropic API directly (security risk)
- `distill.ts` →[uses]→ embeddings.ts (13th hash fallback)
- `consolidate.ts` →[operates_on]→ patterns/pattern_embeddings tables (5th persistence layer)
- `consolidate.ts` →[uses]→ mmr.ts for diversity selection
- `matts.ts` →[uses]→ ModelRouter (6th routing system)

## 8. Session Log

R72: consolidate.ts genuine (82-86%), matts.ts genuine (85-88%), backend-selector bypasses HybridBackend, TWO parallel consolidation systems (post-task LSH vs NightlyLearner causal).

R73 (2026-02-16): ReasoningBank TS outer ring — 12 files, ~1,462 LOC, all DEEP. **10th GENUINE WASM** (wasm-adapter.ts 92-95%). **14th hash-based embedding** (embeddings.js ~60%). **FOUR ReasoningBank layers** confirmed: index.ts (v1.7.1) → HybridBackend → index.js (v1.0.0) → core. backend-selector BYPASSES HybridBackend unification. **post-task.ts ORPHANED** — hook-handler.cjs stub bypasses judge→distill→consolidate pipeline. matts.ts Test-Time Scaling orphaned. **3 independent embedding systems** in ReasoningBank alone (embedding-service.ts, embeddings.ts, embeddings.js) — ZERO code sharing. judge.js (54%) bypasses MCP store, creates parallel data flow. config.ts embedding provider mismatch. Genuine utilities: pii-scrubber.js (88%), mmr.ts (90%+), schema.ts (92-95%). schema.js = empty stub. WASM scoreboard: 10 genuine vs 5 theatrical (67%). CONNECTED tier: 3 remaining (from 12 at start). DEEP: 396 (from 384).

R74 (2026-02-16): ReasoningBank-Network Rust crate + TS inner ring — 10 files, ~2,234 LOC, all DEEP. **Crate BIMODAL**: infrastructure genuine (frame 88-92%, streams 88-92%, mod 88-92%, snapshot 85-88%, priority 85-88%, lib 93-95%) but **gossip.rs 18-22% FAKE** (broadcast wrapper, zero membership/failure detection). Real QUIC multiplexing, Ed25519 intent verification, binary wire protocol. DUAL WIRE PROTOCOL inconsistency (Frame-based vs raw bytes). Orphaned swarm_transport.rs (330+ LOC not exported). TS inner ring: reasoningbankCommands.ts (62-68%) confirms R31 "CLI = demonstration framework" — missing store/retrieve/search. matts.js (68-72%) independent JS impl with fabricated memory aggregation. types/index.ts (75-82%) FABRICATED arXiv citation, hash embeddings endorsed in config. DEEP: 406 (from 396).

R75 (2026-02-16): Clear CONNECTED + temporal-compare deep sweep + crate closures — 10 files, ~2,342 LOC, 116 findings. **CONNECTED tier milestone**: 3 remaining were judge.ts (72-78%), pii-scrubber.ts (85-88%), retrieve.ts (88-92% TOP 10%). **MCP bypass TRIPLE-CONFIRMED** (judge.ts never calls db.store()). retrieve.ts = genuine vector search + MMR diversity + 4-factor scoring. **temporal-compare at 13/17 src DEEP (76%)**: quantization.rs (92-95%) + mlp_quantized.rs (92-95%) both production INT8+AVX2. reservoir.rs ~75% BIMODAL (ESN genuine, broken pseudo-inverse, quantum facade). attention.rs 82-86% (textbook Vaswani but broken causal mask). fourier.rs 65-70% (wrong RFF sampling). **TRIMODAL MLP RESOLVED**: quantized > optimized ≈ avx512 >> mlp. **temporal-lead-solver src COMPLETE (8/8)**: cli.rs 88-92% genuinely uses O(√n) predictor. **wasm_bindings.rs 0% ORPHANED** — 6th theatrical WASM (non-compilable, exposes FALSE algos). WASM scoreboard: 10:6 (62.5%). 3 NEW CONNECTED emerged from deps (distill.js, retrieve.js, temporal-compare lib.rs). DEEP: 416 (from 406).

R76 (2026-02-16): MCP tools + sublinear algorithms + ReasoningBank core — 10 files, ~2,776 LOC, 101 findings. **CONNECTED cleared** (distill.js 68-72%, retrieve.js 82-86%, lib.rs 35-40%) but **3 NEW CONNECTED emerged** (ruv_fann_impl.rs, ruv_fann_adapter.rs, mmr.js). **Sublinear: 3 genuine vs 6 false** — spectral_sparsification.rs 18-22% FAKE, sketching.rs 35-40% broken Count-Sketch, dimension_reduction.rs 25-30% facade. **BMSSP genuinely novel** (matrix-as-graph Dijkstra) but mathematically unsound. **MCP tools ORPHANED from Rust** — solver-optimized.ts (72-78%) genuine CSR+CG but zero backward_push/predictor connection. **WASM: 10 genuine vs 7 theatrical (59%)** — wasm-sublinear-solver-simple.ts explicit mock (15-20%), temporal-attractor GHOST WASM (28th phantom). **15th hash-based embedding** in distill.js + retrieve.js. DEEP: 426 (from 416).

R77 (2026-02-16): Core completion + MCP tools sweep + emergence closure — 10 files, ~2,474 LOC, 149 findings. **src/core/ 12/12 COMPLETE**: utils.ts 92-95% (BEST infrastructure), wasm-bridge.ts 62% (path bug, always JS fallback), wasm-loader.ts 42-48% (7th theatrical WASM, FFI stubs non-functional). **MCP tools/index.ts DEAD CODE** (25-30%, server.ts bypasses). **Three overlapping WASM solver MCP tools** all wrapping nonexistent WASM: wasm-sublinear-solver.ts 18%, simple-wasm-solver.ts 35% (5th FALSE SUBLINEARITY), solver-pagerank.ts 22-28%. **temporal-attractor.ts 88%** genuine nonlinear dynamics. **Emergence COMPLETE 7/7**: self-modification-engine.ts 0% PURE FABRICATION (confirms R39). **TCM mathematical fiction** (15%). WASM scoreboard: 10:9 (53% genuine). DEEP: 436 (from 426).

R78 (2026-02-17): CONNECTED clear + ReasoningBank Rust completion + sublinear core — 10 files, ~1,883 LOC. **CONNECTED cleared** (ruv_fann_impl.rs 25-30% BROKEN, mmr.js 90-93%, ruv_fann_adapter.rs 0-2% placeholder). **ReasoningBank Rust COMPLETE**: server.rs 88-92% (**4TH MCP PROTOCOL** — custom Rust traits), wasm.rs BIMODAL (MemoryStorage 95-98% vs browser 8-12%), optimizer.rs 82-86% MISLABELED (statistical ranker not ML), lib.rs 95-98% **11th GENUINE WASM** (confirms R49). **fast_solver.rs 72-76% 8th MISLABELED** — orphans genuine sublinear algorithms (confirms R70). **temporal_nexus/mod.rs 92-95% BEST module root** (contrast: temporal-compare/lib.rs 35-40%). utils.rs 68-72% SIMD stub confirms R72. WASM scoreboard: 11 genuine vs 9 theatrical (55%). DEEP: 446 (from 436).

### ReasoningBank Retrieval (R81)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| test-retrieval.ts | agentic-flow | 197 | 35-40% | DEEP | UNIT TEST ISOLATION. 16th hash-based embedding via synthetic sin/cos math. fetchMemoryCandidates() domain-filter only, no query embedding computed. Config weights printed but unused in scoring. Cosine similarity math correct but standalone. Does NOT test end-to-end retrieval or MMR algorithm. | R81 |



## 8. Session Log

| Session | Date | Focus | Files | Key Findings |
|---------|------|-------|-------|--------------|
| R81 | 2026-02-17 | reasoningbank-storage/adapters/native.rs | 1 | **Single Mutex blocks executor (CRITICAL)** — NativeStorage 6th independent adapter. No connection pooling, embedding storage unused, no transactions. |
| R81 | 2026-02-17 | mcp-dense-fix.js performance claim analysis | 1 | **THEATRICAL PERFORMANCE DEMO** — claims 4000x+ fix (7700ms→<2ms) but never tests against actual broken MCP Dense. benchmark() circular (tests itself). getMCPToolDefinition() = schema template, NO MCP transport. Does NOT patch ORPHANED MCP→Rust gap (R76). Hardcoded thresholds, misleading speedup metric. |


| R81 | 2026-02-17 | reasoningbank-learning/async_learner_v2.rs | 1 | **V2 = actor pattern wrapper, NOT streaming layer**. Thread-based deadlock risk (tokio single-threaded). Asymmetric error handling (init caught, runtime crashes silent). DISCONNECTED from optimizer.rs (R78 strategy ranker). Pattern cloning inefficiency. 5 CRITICAL findings. |
| R82 | 2026-02-16 | neural-network-impl internals + psycho-symbolic TS periphery | 6 | **R23 "90-98%" QUALIFIED — BIMODAL core vs integration**. solver_gate_simple.rs 0-5% (9th bypass, hardcoded confidence=0.95). Core genuine: loader.rs 92-95% CSV, utils.rs 88-92% stable softmax+Timer, quantization.rs 75-78% INT8. BUT Int4/Binary unimplemented, get_memory_usage() hardcoded 0. Psycho-symbolic TS: logger.ts 88-92% Winston, config.ts 88-92% Zod. **CONFIRMS R80 WASM contradiction** (zero WASM config fields). Pattern: theatrical integration wrappers around genuinely-engineered core. |

| R83 | 2026-02-17 | ReasoningBank TS integration + neural-network training pipeline | 8 | **ReasoningBank TS test layer 92-95% GENUINE** — test-integration.ts exercises full pipeline (5 phases), test-validation.ts confirms statistical ranking (7 tests). consolidate.js 82-85% genuine Algorithm 4 (3rd parallel consolidation system). index-new.ts DEAD CODE (zero imports, recommend DELETE). **Neural-network training SUPPORTS R23**: train.rs 92-95% genuine binary (counters R82 solver_gate), simple_test.rs 92-95% honest. losses.rs 68-75% (only MSE). optimized_benchmark.rs 70-75% (3/4 genuine, ULTRA_OPTIMIZED theatrical). DEEP: 468 (from 460). |

R83 (2026-02-17): install-embeddings.ts deep read + R20 root cause confirmed — 1 file, 81 LOC, 8 findings. **R20 ROOT CAUSE COMPLETE**: install-embeddings.ts (100% GENUINE CLI command) proves embeddings require manual npm install via this CLI command before EmbeddingService can initialize. No automatic runtime initialization. Line 48 "Real embeddings will be used automatically" shows DESIGN FLAW — requires pre-installation of @xenova/transformers (Xenova/all-MiniLM-L6-v2 model, ~90MB). Line 3 comment: "optional embedding dependencies" confirms EmbeddingService was designed with manual installation workflow never automated into runtime. No bridge to EmbeddingService.initialize() — package install only, service activation requires separate code call. **Hardcoded model identity confirmed** — Xenova/all-MiniLM-L6-v2 (384D) is the ONLY model, not placeholder. execSync (line 44) installs @xenova/transformers + onnxruntime-node (native binding, requires python3/make/g++). global install option (line 20) unused. HUGGINGFACE_API_KEY (line 53) mentioned but not implemented — suggests planned multi-model support never completed. **R20 narrative UPDATED**: Root cause is architectural design flaw (manual CLI install as prerequisite) combined with incomplete integration (no EmbeddingService auto-initialization trigger). DEEP: 447.
