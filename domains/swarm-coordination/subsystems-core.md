# Swarm Coordination — Section 5a: Subsystems (Core)

> Part of [Swarm Coordination Domain Analysis](analysis.md)

## 5. Subsystem Sections

### 5a. Swarm Coordination Architecture

Swarm coordination operates at four distinct layers, each with different reality levels:

| Layer | Components | Status | Evidence |
|-------|-----------|--------|----------|
| **Agent Templates** | Coordinator/consensus .md files | **REAL** (accurate algorithms) | CRDT, BFT, threshold crypto all textbook-correct (R9, R10) |
| **P2P Crypto** | p2p-swarm-v2.js | **REAL** (Ed25519, AES-256-GCM) | Production-grade crypto, task execution stubbed (R9, R22) |
| **Shell Coordination** | swarm-comms.sh, swarm-monitor.sh | **REAL** (file-based IPC) | Works but primitive; race conditions (R9) |
| **Distributed Systems** | Federation, MultiDBCoordinator, SyncCoordinator | **FABRICATED** | All return empty arrays, hardcoded data, Math.random() (R9, R22b) |

Real coordination happens through Claude Code's Task tool parallelism + file-based message passing (swarm-comms.sh), not through distributed protocols. MCP tools claiming to report swarm state return hardcoded or random values.

### 5b. P2P Swarm Layer

p2p-swarm-v2.ts (2,280 LOC, 75-80%) is the flagship swarm coordination implementation. Production-grade cryptography includes Ed25519 signing, X25519 ECDH + HKDF key derivation, AES-256-GCM encryption with auth tags, per-sender nonce tracking with replay protection, canonical JSON serialization for deterministic signatures preventing malleability attacks (R22).

Registry-based identity model (NEVER trust keys from envelopes, always resolve from verified member registry) is sound security design. Two-layer encryption: swarm envelope key (broadcast) + per-peer session keys (direct channels). Heartbeat/membership system: 20s heartbeat interval, 60s timeout, negative caching. Task claim conflict resolution: signed claims with 45s TTL, stale claim overwrite (R22).

**Critical gaps**: Task execution has no Wasmtime integration, hardcodes {status:'success', fuelUsed:1000}. WebRTC handleOffer/handleAnswer/handleICE only log messages — zero P2P direct channels. IPFS CIDs are fake `Qm${hash.slice(0,44)}` — NOT real IPFS. Gun relay health is passive failure tracking only, no proactive ping/pong (R22).

p2p-swarm-wasm.js (315 LOC, 0%) imports from non-existent ruvector-edge.js with no fallback. All methods crash if WASM unavailable (R9).

### 5c. Federation System

FederationHubServer.js (437 LOC, 45%) has functional WebSocket server and SQLite metadata storage, but JWT auth is BYPASSED ("TODO: Verify JWT"), AgentDB = null causing crashes when storePattern() called, and vector clock never resets causing unbounded growth (R9).

FederationHub.js (284 LOC, 5%) is entirely simulated: sendSyncMessage() returns [], getLocalChanges() returns [], applyUpdate() has empty switch cases. QUIC is placeholder: "actual implementation requires quiche or similar" (R9).

QUICClient.ts (668 LOC, 25%) confirms the pattern — TypeScript source has hardcoded `{success: true, data: [], count: 0}` after 100ms setTimeout. Connection pool is plain Map with no QUIC protocol. The ONLY real distributed-systems code is in quic.ts types (773 LOC, 95%) with textbook CRDTs (GCounter, LWWRegister, ORSet) (R22b).

SyncCoordinator.ts (717 LOC, 55%) has more genuine logic than initially suspected: change detection via timestamp queries, sync state persistence (SQL upsert), bidirectional sync flow, auto-sync timer. But all operations route through QUICClient which returns empty data. Infrastructure is designed but non-functional due to dependency (R22b).

Supabase schema is well-designed with pgvector (1536-dim embeddings), RLS policies, HNSW index. Missing: programmatic realtime activation, client-side context for RLS (R9).

### 5d. Coordination Core

MultiDatabaseCoordinator (TS: 1,108 LOC 25%, JS: 803 LOC 42%) uses delay(10) instead of network I/O for sync, simulates conflicts with Math.random() < 0.01, health check returns Math.random() > 0.05 (95% uptime). No vector clocks, CRDTs, or causal ordering — just LWW timestamps (R9, R33).

swarm-comms.sh (354 LOC, 70%) implements inter-agent communication via JSON files in `.claude-flow/swarm/queue/`, routed to mailbox directories. Priority-based (0-3), supports unicast and broadcast. **Critical**: race conditions in connection pool (jq on single file, non-atomic). Consensus voting creates files but has no actual quorum logic (R9).

swarm-monitor.sh (218 LOC, 20%) fabricates agent count as `(process_count / 2)` heuristic. Uses real pgrep but interpretation is fabricated (R9).

### 5e. Agent Templates

The 12 agent template .md files are algorithmically accurate but implementation-incomplete:

**Excellent (90-95% design quality)**: crdt-synchronizer.md (1,005 LOC) — textbook CRDTs with RGA merge oversimplified and delta computation undefined. quorum-manager.md (831 LOC) — sound BFT math (ceil(2n/3)+1), network clustering undefined, hardcoded scoring weights. security-manager.md (625 LOC) — gold standard crypto documentation, ZKP library missing, Lagrange coefficients for threshold signatures undefined. mesh-coordinator.md (971 LOC) — real gossip, work-stealing, auction, GraphRoPE, BFS, Byzantine detection. hierarchical-coordinator.md (718 LOC) — hyperbolic attention, depth/sibling encoding, weighted consensus. performance-benchmarker.md (859 LOC) — throughput ramp, p50-p99.99, CPU/memory profiling (R9, R10).

**Good (70-85%)**: adaptive-coordinator.md (1,133 LOC) — sophisticated concepts, all 5 attention mechanisms delegate to undefined service. topology-optimizer.md (816 LOC) — GA (pop=100, 500 gen), simulated annealing, METIS-like partitioning. consensus-coordinator.md (346 LOC) — PageRank voting, depends on non-existent MCP tool (R9, R10).

**Stubs (10%)**: byzantine-coordinator.md, gossip-coordinator.md, raft-manager.md (all 71 LOC) — mention algorithms but no pseudocode or implementation detail (R10).

### 5f. SKILL.md Files

v3-swarm-coordination.md (340 LOC, 90%) is the most concrete actionable blueprint — 15-agent hierarchical structure tied to actual v3 ADRs (spawn researcher -> break into 5 subtasks -> spawn coder/reviewer/tester per subtask). hive-mind-advanced.md (713 LOC, 80%) documents real CLI tools with 3 consensus algorithms (Raft, Byzantine, CRDT). swarm-advanced.md (974 LOC, 50%) over-promises — ~30% references non-existent MCP functions. flow-nexus-swarm.md (611 LOC, 40%) requires external MCP server. swarm-orchestration.md (180 LOC, 30%) is skeleton needing 3-4x expansion (R10).

### 5g. Rust Swarm Crates

The ruv-FANN repository contains 11 Rust crates for swarm coordination. Quality varies dramatically:

**Best infrastructure (85-95%)**: ruv-swarm-transport (websocket.rs 88-92%, in_process.rs 92%, shared_memory.rs 85-88%) — production WebSocket with exponential backoff, DashMap-based agent registry, ring buffer with atomic head/tail. ruv-swarm-benchmarking (storage.rs 95-98%, comparator.rs 88-92%) — exceptional SQL schema with 10 normalized tables and 9 indexes, real Welch's t-test and Cohen's d. ruv-swarm-mcp validation (validation.rs 92-95%, tools.rs 95-98%) — path traversal protection, null byte prevention, 11 production tool schemas (R31, R34).

**Moderate quality (60-80%)**: ruv-swarm-wasm cognitive layer (agent_neural.rs 80-85%, simd_optimizer.rs 85-90%, cognitive_diversity_wasm.rs 75-80%) — genuinely trains ruv_fann networks with IncrementalBackprop, real f32x4 WASM SIMD128 intrinsics, real Shannon diversity index. ruv-swarm-persistence (sqlite.rs 92%, ensemble/mod.rs 78%, agent_forecasting/mod.rs 65%) — r2d2 pooling with WAL and ACID, real averaging and EMA, but PI*1000.0 mock timestamp (R21, R31).

**Facades (5-35%)**: ruv-swarm-daa (gpu.rs 15%, daa_gpu_agent_framework.rs 5-8%, coordination_protocols.rs 30%, lib.rs 55%, neural.rs 0-5%, patterns.rs 15-20%) — ZERO GPU ops despite 3 GPU-named files, all 11 types from ruv_fann::webgpu don't exist, seek_consensus() sets consensus_reached=true unconditionally, orchestrate_task() hardcodes success:true. **R82**: neural.rs is PURE METADATA FACADE (HashMap storage, zero computation), patterns.rs defines 6 cognitive styles but 67% undefined with zero behavior. types.rs (78-82%) RESOLVES R69 ghost model mismatch — AgentType = 5 agent ROLES not 27 neural models. ruv-swarm-wasm neural (neural_swarm_coordinator.rs 15-20%, swarm_orchestration_wasm.rs 20-25%, learning_integration.rs 30-40%) — all 4 training modes return hardcoded loss curves [0.5,0.3,0.2,0.15,0.1], 4 optimization algorithms return pattern.clone(). **R82**: neural.rs (85-90%) REVERSES R80 activation.rs with correct ruv-fann API (R28, R29, R34, R82).

**Interface drift epidemic**: Three-way API mismatches — handlers.rs (R28) has ~12 wrong method calls to orchestrator.rs (R31), tools.rs schemas (R31) don't match validation.rs strategies (4 vs 6), lib.rs (R31) has 85% commented out disconnecting the entire MCP server. Components developed independently and never integrated.

### 5h. npm JavaScript Swarm Layer

The ruv-swarm npm package (R16, R29, R33) demonstrates an **inverted quality gradient** — JS orchestration layer (88% real average) is more production-ready than the Rust execution layer (25-35% real average).

**Production-grade (88-95%)**: ruv-swarm-secure-heartbeat.js (1,549 LOC, 92%) — production MCP stdio server with JSON-RPC 2.0, restart circuit breaker, regex input sanitization, CommandSanitizer. schemas.js (864 LOC, 95%) — production recursive validator with 25+ MCP tool schemas, UUID validation, input sanitization. persistence-pooled.js (695 LOC, 92%) — 8-table schema, exponential backoff retry, TTL cleanup, SQLite VACUUM. mcp-daa-tools.js (735 LOC, 90%) — 10 MCP tools wrapping real daaService with proper error handling (R29, R33).

**Genuine coordination (84-88%)**: daa-cognition.js (977 LOC, 88%) — REAL Byzantine-tolerant consensus protocol with weighted voting, real distributed learning with pattern extraction + peer aggregation, emergent pattern detection (occurrence>0.7, diversity>0.5). neural-agent.js (830 LOC, 84%) — REAL neural network with Xavier/Glorot init, forward/backward with momentum, 4 activations, real feature engineering (12+ input dims), cognitive pattern modifiers affect analysis. claude-flow-enhanced.js (840 LOC, 85%) — real dependency graph analysis with topological sort and circular dependency detection, batching violation enforcement (SIMD speedup values hardcoded 3.2, 4.1) (R29).

**Partial implementation (65-78%)**: index-enhanced.js (734 LOC, 65%) — orchestrator with WASM fallback, Agent.execute() is stub. wasm-memory-optimizer.js (784 LOC, 78%) — real buddy allocator with block merging + compaction, but SIMD functions are placeholders. mcp-tools-enhanced.js (2,863 LOC, 70%) — real persistence/WASM benchmarks/error handling, fabricated neural_train/agent_metrics/swarm_monitor (R16, R33).

**Facades (10-30%)**: neural-coordination-protocol.js (1,363 LOC, 10-15%) — 8 coordination executions stubbed, all return {success:true}. neural-network-manager.js (1,938 LOC, 15-20%) — SimulatedNeuralNetwork uses Math.random() when WASM fails (R19).

The JS layer correctly delegates to WASM/native backends via MCP tools — the problem is those backends don't compute (neural_swarm_coordinator.rs returns hardcoded loss curves, learning_integration.rs optimization algorithms return pattern.clone()).

**R45+R48 npm operational layer** (9 files, 6,270 LOC, ~76% weighted average): Confirms the inverted quality gradient at the npm package level. sqlite-pool.js (587 LOC, 92%) is a genuine production connection pool — WAL mode, separate read/write connections, worker thread pool, health monitoring with EventEmitter, auto-recovery, and prepared statement caching. wasm-loader.js (602 LOC, 82%) has real WebAssembly.instantiate() with wasm-bindgen integration and 4-strategy path resolution (local/npm/global/inline), but creates facade placeholder API as last-resort fallback. neural.js (574 LOC, 28%) introduces a NEW anti-pattern: "WASM delegation with ignored results" — it calls neural_train() but immediately overwrites return values with Math.random() formulas, making the WASM call purely performative. performance-benchmarks.js (899 LOC, 62%) has real SIMD/WASM/browser benchmarks (78-95%) but complete setTimeout facades for neural/Claude/parallel tests (15-20%). mcp-workflows.js (991 LOC, 72%) is a genuine JSON-RPC 2.0 WebSocket client with proper request/response correlation, but is orphaned — the Rust MCP server it expects has disabled handlers (R31 C24). generate-docs.js (954 LOC, 87%) parses real source files via regex extraction. claude-simulator.js (602 LOC, 88%) is production-quality MCP test infrastructure (Prometheus, Winston, chaos injection) but its docker-compose reveals the self-contained demo loop — simulator connects to test/docker-mcp-validation.js, not real MCP server.

**R48 additions**: diagnostics.js (533 LOC, 87%) is genuine system diagnostics — collects real metrics (process.memoryUsage, process.cpuUsage, performance.now, process._getActiveHandles/_getActiveRequests for event loop monitoring), performs error classification with hourly failure distribution, provides actionable recommendations (failure rate >10%, memory >500MB, handle count >50), includes self-test harness, used by cli-diagnostics.js. errors.js (528 LOC, 90%) is a genuine error taxonomy with 11 typed error classes defining clear boundaries between WASM, SQLite, neural, MCP, network, and concurrency layers. Each error class has context-aware getSuggestions() (e.g., ValidationError checks expectedType, SwarmError checks error context, NetworkError checks HTTP status). ErrorFactory provides single entry point for error creation + wrapping. **Used extensively** by mcp-tools-enhanced.js (26 import/usage sites).

These two files **complete the ruv-swarm npm source picture**. The genuine infrastructure layer now totals 8 files (sqlite-pool 92%, errors 90%, simulator 88%, docs 87%, diagnostics 87%, claude-flow-enhanced 85%, wasm-loader 82%, mcp-workflows 72%) vs 1 facade (neural 28%). Infrastructure-to-facade ratio: **8:1 genuine**.

**R85 npm runtime additions (4 files, 530 LOC)**: build.js (167 LOC, 90-95%) confirms the WASM build pipeline is REAL — dual SIMD compilation (standard + SIMD-optimized), wasm-opt pass, TypeScript definitions generated at build time. Pre-built npm/wasm/ artifacts are legitimately compiled from Rust crates. This upgrades confidence in WASM-adjacent npm code. hooks/cli.js (82 LOC, 75-80%) is the CLI entry point for all ruv-swarm hook invocations — a thin JSON router with custom arg parser, delegating entirely to index.js (1,899 LOC) which holds all hook logic. github-coordinator/claude-hooks.js (162 LOC, 71%) is BIMODAL: GitHub hook integration is genuine (5 hook types, real GitHub API calls via GHCoordinator), but is completely isolated from claude-flow's hook system and ADR-008 model routing. In-memory state limits coordination to single-process lifetime. ruv-swarm-memory.js (0%) finalizes as the 8th disconnected persistence layer — pure CLI demo with hardcoded fabricated metrics, zero persistent backend. **Build layer confirmed genuine; runtime coordination layer remains fragmented.**

### 5i. Neural Models

**JS neural models (R40)**: 4 files (lstm.js 85%, transformer.js 83%, gnn.js 81%, base.js 75%) implement genuine neural network algorithms — not facades. Correct 4-gate LSTM (Hochreiter 1997), multi-head attention with sinusoidal positional encoding (Vaswani 2017), MPNN with GRU-gated updates (Gilmer 2017). Math is correct for forward passes. However, **no file implements backpropagation**. Training runs forward passes and computes loss, but backward() inherited from base class is a console.log stub. Weights never update. Two files return hardcoded accuracy values (lstm.js 0.864, gnn.js 0.96). ZERO connection to Rust neural-network-implementation crate — no WASM, no NAPI, no FFI bindings. Pure standalone JS. The JS neural models are inference-only counterparts to production Rust implementations. Genuine algorithmic understanding but cannot train models.

**Critical bug**: Transformer learning rate uses Math.sqrt(step) instead of 1/Math.sqrt(step) — training would diverge (transformer.js:321).

**R91 neural preset layer (neural-presets-complete.js, 1,306 LOC, 52-58%)**: The "complete" presets file is BIMODAL at the file level. Lines 1-774 contain `COMPLETE_NEURAL_PRESETS` — 27 architecture families as nested config objects with textbook-accurate hyperparameters (BERT 768d/12H/12L/3072-FFN, EfficientNet-B0 compound scaling coefficients, DDPM 1000 timesteps cosine schedule, GPT-2 medium 1024-dim). These are LOOKUP TABLES ONLY: no forward pass, no weight initialization, no tensor operations. `expectedAccuracy`, `inferenceTime`, and `memoryUsage` are hardcoded paper benchmarks. `calculatePresetScore()` is broken: uses `parseInt`/`parseFloat` on range strings like "5ms/step" -> NaN. Lines 780-1305 contain `CognitivePatternSelector` + `NeuralAdaptationEngine` with real algorithmic logic — multi-context branching (creativity/precision/adaptation), diversity enforcement, top-5 scored recommendations, and accuracy-delta-based hyperparameter suggestions drawn from adaptation history. The file is genuinely integrated (4 call sites in neural-network-manager.js) but `CognitivePatternEvolution` and `MetaLearningFramework` are imported and instantiated but never invoked — dead import anti-pattern. `crossSessionMemory` is an in-memory Map that resets on every instantiation, confirming the 9th disconnected persistence layer. **This file is the definitive source of the R69 "27 ghost models" count**: 27 architecture configs exist as data, but the WASM backend implements only a small fraction.

**Rust neural (R21, R19)**: neural-network-implementation crate (90-98%, sublinear-time-solver) is BEST CODE IN ECOSYSTEM. Genuine GRU (9 weight matrices), causal dilated TCN, GELU. System B Temporal Solver: NN predicts residual over Kalman prior (not raw output), with solver gate verification and 4 fallback strategies. P99.9 latency budget <= 0.90ms. Uses proper rand::thread_rng(). swarm_coordinator_training.rs (25-35%, ruv-swarm-ml) has real GNN/attention/Q-learning/VAE algorithm skeletons, but ALL 5 training metrics hardcoded (GNN=0.95, Transformer=0.91). Fake rand via SystemTime::now().subsec_nanos().

### 5j. Python ML Training

5 Python files (4,319 LOC, 72% real average, R33) use genuine PyTorch infrastructure but ALL training data is synthetic/fabricated. train_ensemble_improved.py (969 LOC, 85%) — BEST Python file with GATConv, NoisyLinear, Beta-VAE with curriculum learning, real gradient descent. hyperparameter_optimizer.py (858 LOC, 68%) — real Bayesian GP minimization via scikit-optimize, but ALL 5 model evaluators are SIMULATED (not actual training). train_lstm_coding_optimizer.py (853 LOC, 78%) — real seq2seq with Luong attention + copy mechanism, but data = HARDCODED coding templates (lines 84-134). enhanced_strategies.py (820 LOC, 62%) — 4 real decomposition strategies (waterfall/agile/feature/component), MockModel returns random predictions. train_ensemble.py (819 LOC, 71%) — base version of improved, real ensemble training, RL uses np.random.normal(0.5, 0.2) for rewards. Accuracy claims (95%, 85%, 88%) are aspirational, unvalidated on real data.

### 5k. Workflow Execution

claude_integration.rs (1,344 LOC, 70-75%, ruvllm, R37) implements workflow orchestration for coordinated agent tasks. ClaudeModel enum with real API pricing and context windows, ToolDefinition with complete schema, WorkflowExecution with retry/timeout/state tracking, TokenBudget with per-step allocation and overage detection — all production-grade architecture. **Critical simulation**: execute_workflow() hardcodes `tokens_used: 500`, generates fake response text, no real API calls. Combined with JS RuvSwarm's placeholder message-passing (R31 index-enhanced.js Agent.execute() stub), means zero functional swarm execution across both languages.

### 5l. Benchmarking & SWE-bench

ruv-swarm-benchmarking crate (R31): storage.rs (795 LOC, 95-98%) is BEST SQL — 10 normalized tables with CHECK constraints, foreign keys, 9 performance indexes, full async CRUD via sqlx, real environment capture. comparator.rs (584 LOC, 88-92%) — real Welch's t-test (Welch-Satterthwaite DOF), Cohen's d effect size via statrs. **Critical**: n=1 comparisons have hardcoded p_value=0.01, CI=(0.1,0.3), effect_size=0.5 — fake statistics for single-run benchmarks. stream_parser.rs (602 LOC, 85-90%) parses Claude Code stream-json (8 event types), thinking duration estimated at hardcoded 50ms/token. realtime.rs (521 LOC, 85-90%) — production Axum WebSocket monitoring, DashMap concurrent run tracking, missing static/monitor.html. lib.rs (552 LOC, 75-80%) — **Critical**: build_command() generates ENGLISH PROMPTS ("solve SWE-bench instance X using ML-optimized swarm coordination") instead of CLI flags. Benchmark cannot execute.

SWE-bench adapter (R31): prompts.rs (534 LOC, 98%) — BEST quality file with 4 difficulty-based Claude Code prompt templates, token estimation, section-aware truncation, zero stubs. loader.rs (493 LOC, 75%) — difficulty scoring is REAL (weighted formula), download_instance() returns MOCK data (repo: "mock/repo"). lib.rs (580 LOC, 70%) — framework architecture complete, **Critical**: evaluate_instance() returns hardcoded mock output="Mock execution output", patch="diff fix".

CLI commands (R31): init.rs (538 LOC, 65%) — interactive config real, actual spawning simulated (sleep 200-500ms). status.rs (687 LOC, 60%) — display logic production-ready, loads stale JSON files not live swarm state. orchestrate.rs (662 LOC, 45%) — 4 strategies architecturally correct, execute_subtask() sleeps 1-2s and returns success:true, build_consensus() hardcodes agreement_level: 0.85.

**Systemic finding**: CLI layer inversion — prompt generation (98%) > persistence (95%) > data loading (75%) > framework integration (70%) > config generation (65%) > status display (60%) > orchestration execution (45%). The further from actual task execution, the more real the code becomes.
