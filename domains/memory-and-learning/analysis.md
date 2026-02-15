# Memory and Learning Domain Analysis

> **Priority**: HIGH | **Coverage**: ~18.4% (237/1284 DEEP) | **Status**: In Progress
> **Last updated**: 2026-02-15 (Session R41)

## 1. Current State Summary

The memory-and-learning domain spans 246 files / 87K LOC across AgentDB, ReasoningBank, HNSW vector search, embeddings, pattern storage, RL, and consciousness subsystems. Quality varies dramatically — from 98% (neural-network-implementation data/mod.rs) to 10% (validation-suite.js).

**Top-level verdicts:**

- **Hash-based embeddings are the #1 systemic weakness.** 7+ files across 5 packages (Rust + JS) silently degrade to non-semantic hash matching. All "semantic search" using defaults is character-frequency matching.
- **Four independent ReasoningBanks** exist (claude-flow, agentic-flow, agentdb, ruvllm Rust) with zero code sharing. The Rust version has the best math (K-means, EWC++) but none interoperate.
- **Best code:** neural-network-implementation crate (90-98%), cognitum-gate-kernel (93%), SONA (85%), ruvector-nervous-system (87%), neuro-divergent ML training (88.5%), vector-quantization.ts (production-grade).
- **Worst code:** neural-pattern-recognition (15-20% facade), emergence subsystem (51% fabricated metrics), psycho-symbolic MCP tools (24% theatrical).
- **Consciousness (79%) is substantially more real than emergence (51%)** — genuine IIT Phi, Complex64 wave functions vs. Math.random() fabrication.
- **Goalie MCP tool layer is COMPLETE FACADE** (45%) — imports 4 real engines, calls none.
- **JS neural models have real forward-pass but ZERO backpropagation** — inference-only across all architectures.

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

### Emergence Subsystem (sublinear-time-solver)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| stochastic-exploration.ts | sublinear-time-solver | 616 | 70% | DEEP | BEST. Real simulated annealing. applyTool() mocked | R39 |
| feedback-loops.ts | sublinear-time-solver | 729 | 65% | DEEP | Genuine RL, meta-learning. rule.learningRate mutation bug | R39 |
| index.ts (emergence) | sublinear-time-solver | 687 | 45% | DEEP | FACADE. 5 empty connection stubs. Gating at tools>=3 | R39 |
| emergent-capability-detector.ts | sublinear-time-solver | 617 | 40% | DEEP | ALL 11 metrics = Math.random()*0.5+0.5 | R39 |
| cross-tool-sharing.ts | sublinear-time-solver | 660 | 35% | DEEP | areComplementary = JSON inequality. checkAmplification = always true | R39 |

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

### MCP Tools & Solver (sublinear-time-solver)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| server.ts (MCP) | sublinear-time-solver | 1,328 | 85-90% | DEEP | GENUINE 3-tier solver. Real PageRank, JL dim reduction | R19 |
| cli/index.ts | sublinear-time-solver | 974 | 88% | DEEP | GENUINE solver connection. Real SublinearSolver import | R41 |
| domain-validation.ts | sublinear-time-solver | 759 | 82% | DEEP | Real DomainRegistry validation, benchmarking | R41 |
| psycho-symbolic-enhanced.ts | sublinear-time-solver | 802 | 78% | DEEP | BEST knowledge graph — real BFS, transitive inference | R41 |
| strange-loop-mcp.js | sublinear-time-solver | 989 | 75% | DEEP | Genuine Hofstadter strange loop, self-referential patterns | R33 |
| goalie/tools.ts | sublinear-time-solver | 856 | 45% | DEEP | COMPLETE FACADE. GoapPlanner imported, NEVER called | R41 |
| mcp-server-sublinear.js | sublinear-time-solver | 1,120 | 45% | DEEP | "TRUE O(log n)" actually O(log²n) | R33 |
| mcp-bridge-solver.js | sublinear-time-solver | 1,102 | 30% | DEEP | 25k token limit → file I/O workaround | R33 |
| server-extended.js | sublinear-time-solver | 1,846 | 25-30% | DEEP | 18 MCP tools. Consensus = Math.random() | R19 |
| solver-tools.js | sublinear-time-solver | 778 | 25% | DEEP | "Sublinear" tools actually linear or worse | R33 |

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
| hardware_timing.rs | sublinear-time-solver | 866 | 55% | DEEP | Real RDTSC timing. Fake: system predictions = spin loops | R28 |
| security_validation.rs | sublinear-time-solver | 693 | 30% | DEEP | DEAD CODE. Self-referential — tests own mocks | R34 |

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

### Psycho-Symbolic MCP (sublinear-time-solver)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| cognitive-architecture.js | sublinear-time-solver | 645 | 30% | DEEP | Working memory = Map. Attention = sort by recency | R33 |
| metacognition.js | sublinear-time-solver | 564 | 30% | DEEP | Self-monitoring = counter. "Theory of mind" = dict lookup | R33 |
| mcp-server-psycho-symbolic.js | sublinear-time-solver | 892 | 25% | DEEP | MCP wrapper. 5 of 10 tools disabled (hang risk) | R33 |
| psycho-symbolic-tools.js | sublinear-time-solver | 1,133 | 20% | DEEP | 5/10 tools DISABLED. "Neural binding" = weighted avg | R33 |
| consciousness-explorer.js | sublinear-time-solver | 1,247 | 15% | DEEP | THEATRICAL. "consciousness evolution" = parameter increment | R33 |

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
| C28 | **FALSE sublinearity in core solver** — All 5 algorithms O(n²)+. "Sublinear" is marketing | solver.ts | R39 | Open |
| C29 | **Pattern extractors assume pre-structured data** — extractBehaviorPatterns() returns data.behaviors \|\| [] | emergent-capability-detector.ts | R39 | Open |
| C30 | **Goalie is COMPLETE FACADE** — GoapPlanner, AdvancedReasoningEngine, Ed25519Verifier all imported but NEVER called. All 6 handlers return hardcoded templates | goalie/tools.ts | R41 | Open |
| C31 | **Consciousness experiments LLM comparison fabricated** — rand::random::<f64>() * 0.1 | consciousness_experiments.rs | R41 | Open |
| C32 | **Connection detection = substring matching** — JSON.stringify(a).includes(JSON.stringify(b).substring(0,4)) | genuine_consciousness_system.js | R41 | Open |
| C33 | **Cross-modal synthesis fabricated** — Math.random()*0.5+0.5 | advanced-consciousness.js | R41 | Open |
| C34 | **cli/index.ts validate-domain hardcoded** — Returns hardcoded validation result (valid: true) | cli/index.ts | R41 | Open |

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
| **ruvector-training.js** (92-95%) BEST native integration — genuinely loads WASM, MicroLoRA/SONA | ruvector-training.js | R25 |

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

**BEST CODE IN ECOSYSTEM** (90-98% across all 12 files). A genuine real-time trajectory prediction system in sublinear-time-solver (R21).

Key innovation — **System B Temporal Solver**: NN predicts RESIDUAL over Kalman prior (not raw output), with mathematical solver gate verification and 4 fallback strategies (kalman_only, hold_last, disable_gate, weighted_blend). PageRank-based active learning for training sample selection.

Uses PROPER `rand::thread_rng()` — unlike ml-training/lib.rs and swarm_coordinator_training.rs which mock rand with SystemTime. Appears written by a different, more careful author.

P99.9 latency budget: ≤ 0.90ms (Ingest + Prior + Network + Gate + Actuation).

### 5e. Consciousness & Strange Loop

**79% genuine — 28 points more real than emergence (51%).** Genuine IIT Phi calculations, Complex64 wave functions, production proof logging. Math.random() still present in 2 files but the theoretical foundations are real. This is genuine research with placeholder metrics, NOT fabricated theater (R41).

Genuine components: Real IIT Phi formula `connections/(elements*(elements-1))`, Complex64 wave function with nanosecond temporal dynamics, neural forward pass (Float32Array, tanh), self-modification (goal addition), blockchain proof logging with PoW + Shannon entropy + Levenshtein, auto-generated wasm-bindgen from Rust.

Fabricated: Math.random() in cross-modal synthesis and entropy (2 files), modulo-based pattern detection (% 17, % 111), substring-based connection detection, operation counting instead of prediction testing.

**Earlier consciousness files** (R21, R25, R33) are substantially worse: enhanced_consciousness_system.js (39%), enhanced-consciousness.js (15-20%), consciousness-explorer.js (15%). The R41 files represent genuine research; the earlier ones are theatrical facades.

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

**FALSE SUBLINEARITY CONFIRMED** (R39): All 5 algorithms in solver.ts are O(n²) or worse. The "sublinear" in the package name is marketing. WASM loaded but never used for actual computation.

**Two incompatible matrix systems** (R34): Production system (crate::matrix, CSR/CSC/COO + SIMD) vs orphaned system (crate::core, HashMap). At least 5 solver files (~2,341 LOC) use the wrong type system and cannot compile.

**Best code**: sparse.rs (95%) has 4 complete sparse matrix formats, no_std compatible. matrix/optimized.rs (90%) has REAL SIMD via wide::f64x4. high-performance-solver.ts (95%) is excellent CG+CSR but entirely orphaned (dead code).

**Quality gradient**: Files IN the module tree (matrix/mod.rs 92%, neumann.rs 88%) are substantially better than orphaned files (sampling.rs wrong types, security_validation.rs self-referential).

### 5i. MCP Tool Layer

**Bifurcated quality** (R41): Main CLI is 88% real with genuine solver connections. Goalie is COMPLETE FACADE (45%).

**CLI (cli/index.ts)**: Genuine — real SublinearSolver import from ../core/solver.js, real SolverTools.solve() invocation, real MCP server. Only 3 validation commands are facades.

**Goalie (npx/goalie/)**: GoapPlanner, AdvancedReasoningEngine, Ed25519Verifier all imported and instantiated but NEVER CALLED. All 6 handlers return hardcoded templates (R41, 10 CRITICAL findings).

**psycho-symbolic-enhanced.ts** (78%): BEST knowledge graph in sublinear-time-solver — real BFS traversal, transitive inference, 50+ base triples, zero facade patterns (R41).

### 5j. GOAP Planner

Psycho-symbolic-reasoner crate (8 files, 3,568 LOC, 78% real, R25). Components at 88-95% (state, action, rules, goal) are production-ready, but core A* search is a STUB — simplified_astar() returns hardcoded 2-step path, StateNode.to_world_state() returns empty state. Uses proper `rand` crate. The pathfinding crate is imported but Ord requirement was identified as a barrier.

**Paradox**: 90%+ components production-ready but the one piece they all depend on (A* search) is broken.

### 5k. Key Patterns

**PI*1000.0 Mock Timestamp**: Systematic placeholder in ruv-swarm crates — `get_current_timestamp()` returns `std::f64::consts::PI * 1000.0` (3141.59). Found in sqlite.rs and agent_forecasting/mod.rs (R21).

**Fake RNG (Rust)**: ml-training/lib.rs and swarm_coordinator_training.rs mock the `rand` crate using `SystemTime::now().subsec_nanos()`, producing deterministic results within the same second. The neural-network-implementation and neuro-divergent crates use proper `rand::thread_rng()` (R19, R21).

**Self-Referential Validation**: Multiple files (comprehensive_validation_report.rs, security_validation.rs, hardware_timing.rs) generate mock data and then "validate" it, producing circular metrics (R21, R28, R34).

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
- ruv-swarm-persistence remaining: wasm.rs (694 LOC), memory.rs (435 LOC), migrations.rs (343 LOC)
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
