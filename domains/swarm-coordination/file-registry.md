# Swarm Coordination — Section 2: File Registry

> Part of [Swarm Coordination Domain Analysis](analysis.md)

## 2. File Registry

### P2P Swarm (agentic-flow + claude-flow)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| p2p-swarm-v2.ts | agentic-flow | 2,280 | 75-80% | DEEP | Production crypto (Ed25519, AES-256-GCM), task executor stub | R22 |
| p2p-swarm-v2.js | agentic-flow | 1,787 | 75-80% | DEEP | Compiled version confirms TS findings | R9 |
| p2p-swarm-wasm.js | agentic-flow | 315 | 0% | DEEP | BROKEN. Imports non-existent ruvector-edge.js | R9 |
| p2p-swarm-tools.js | claude-flow | 600+ | 85% | DEEP | 12 MCP tools with Zod validation, correct wrappers | R9 |
| p2p-swarm-hooks.js | claude-flow | 400+ | 85% | DEEP | 9 hooks with proper error handling | R9 |

### Federation System (agentic-flow)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| FederationHubServer.js | agentic-flow | 437 | 45% | DEEP | WebSocket works, JWT bypassed, AgentDB null crash | R9 |
| FederationHub.js | agentic-flow | 284 | 5% | DEEP | Entirely simulated, all sync methods return [] | R9 |
| realtime-federation.js | agentic-flow | 400+ | 70% | DEEP | Supabase listeners correct, realtime needs manual enable | R9 |
| debug-stream.js | agentic-flow | 350+ | 95% | DEEP | Fully functional observability | R9 |
| agent-debug-stream.js | agentic-flow | 350+ | 95% | DEEP | Fully functional observability | R9 |

### Coordination Core (agentic-flow)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| MultiDatabaseCoordinator.ts | agentic-flow | 1,108 | 25% | DEEP | delay(10) instead of sync, Math.random() conflicts | R9 |
| MultiDatabaseCoordinator.js | agentic-flow | 803 | 42% | DEEP | Confirms TS findings, 1% hardcoded conflict rate | R33 |
| SyncCoordinator.ts | agentic-flow | 717 | 55% | DEEP | Real change detection but QUICClient returns empty data | R22b |
| QUICClient.ts | agentic-flow | 668 | 25% | DEEP | Stub — returns hardcoded {success:true, data:[]} | R22b |
| quic.ts | agentic-flow | 773 | 95% | DEEP | Textbook CRDTs (GCounter, LWWRegister, ORSet) only | R22b |
| transport-router.js | claude-flow | 375 | 60% | DEEP | HTTP/2 real, QUIC fabricated | R10 |
| dispatch-service.ts | agentic-flow | 1,212 | 80% | DEEP | 12 real worker types, vectorization stub | R22b |
| intelligence-bridge.ts | agentic-flow | 1,371 | 70% | DEEP | Math.random()*0.1 activations pollute trajectories | R22b |
| attention-fallbacks.ts | agentic-flow | 1,953 | 85-90% | DEEP | Real Flash Attention + backward pass. SIMD is 8x loop unroll | R22 |

### Shell Coordination (claude-flow)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| swarm-comms.sh | claude-flow | 354 | 70% | DEEP | File-based IPC works, race conditions in jq operations | R9 |
| swarm-monitor.sh | claude-flow | 218 | 20% | DEEP | Agent count = (process_count/2) heuristic | R9 |

### Agent Templates (claude-flow)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| crdt-synchronizer.md | claude-flow | 1,005 | 90% | DEEP | Textbook CRDTs, RGA merge oversimplified | R9 |
| quorum-manager.md | claude-flow | 831 | 90% | DEEP | Sound BFT math, hardcoded scoring weights | R9 |
| security-manager.md | claude-flow | 625 | 95% | DEEP | Gold standard crypto docs, ZKP library missing | R9 |
| adaptive-coordinator.md | claude-flow | 1,133 | 80% | DEEP | Sophisticated concepts, all 5 attention mechanisms delegate to undefined service | R9 |
| mesh-coordinator.md | claude-flow | 971 | 95% | DEEP | Real gossip, work-stealing, auction, GraphRoPE, BFS, Byzantine detection | R10 |
| hierarchical-coordinator.md | claude-flow | 718 | 95% | DEEP | Hyperbolic attention, depth/sibling encoding, weighted consensus | R10 |
| performance-benchmarker.md | claude-flow | 859 | 95% | DEEP | Throughput ramp, p50-p99.99, CPU/memory profiling | R10 |
| topology-optimizer.md | claude-flow | 816 | 85% | DEEP | GA (pop=100, 500 gen), simulated annealing, METIS-like partitioning | R10 |
| consensus-coordinator.md | claude-flow | 346 | 70% | DEEP | PageRank voting, depends on non-existent MCP tool | R10 |
| byzantine-coordinator.md | claude-flow | 71 | 10% | DEEP | Stub — PBFT mentioned only | R10 |
| gossip-coordinator.md | claude-flow | 71 | 10% | DEEP | Stub — lists push/pull, zero algorithmic detail | R10 |
| raft-manager.md | claude-flow | 71 | 10% | DEEP | Stub — leader election mentioned, no pseudocode | R10 |

### SKILL.md Files (claude-flow)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| v3-swarm-coordination.md | claude-flow | 340 | 90% | DEEP | Best — concrete 15-agent blueprint tied to v3 ADRs | R10 |
| hive-mind-advanced.md | claude-flow | 713 | 80% | DEEP | Real CLI tools documented, 3 consensus algorithms | R10 |
| swarm-advanced.md | claude-flow | 974 | 50% | DEEP | ~30% references non-existent MCP functions | R10 |
| flow-nexus-swarm.md | claude-flow | 611 | 40% | DEEP | Over-promises — requires external MCP server | R10 |
| swarm-orchestration.md | claude-flow | 180 | 30% | DEEP | Skeleton — needs 3-4x expansion | R10 |

### GitHub Swarm Templates (claude-flow)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| swarm-issue.md | claude-flow | 559 | 85% | DEEP | Very good, 1 portability bug (GNU date) | R10 |
| swarm-pr.md | claude-flow | 412 | 80% | DEEP | Very good | R10 |
| code-review-swarm.md | claude-flow | 323 | 90% | DEEP | Excellent reasoning blueprint | R10 |
| release-swarm.md | claude-flow | 573 | 60% | DEEP | Good, 3 CLI issues | R10 |
| multi-repo-swarm.md | claude-flow | 537 | 50% | DEEP | Medium — fragile cross-platform | R10 |

### dist/ Implementation Files (claude-flow)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| supabase-adapter-debug.js | claude-flow | 401 | 95% | DEEP | Production-grade Supabase integration | R10 |
| e2b-swarm.js | claude-flow | 366 | 90% | DEEP | Real E2B sandbox orchestration (requires API key) | R10 |
| transport-router.js | claude-flow | 375 | 60% | DEEP | HTTP/2 real, QUIC fabricated | R10 |
| swarm-learning-optimizer.js | claude-flow | 351 | 20% | DEEP | Reward calculations invented, speedup predictions ungrounded | R10 |
| swarm.js (CLI) | claude-flow | 325 | 30% | DEEP | P2P backend missing, will crash | R10 |

### AgentDB Simulations

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| voting-system-consensus.ts | agentdb | 252 | 70% | DEEP | Real code, coalition counting bug, limited RCV | R9 |
| research-swarm.ts | agentdb | 188 | 40% | DEEP | Real DB, fake research, hardcoded outcomes | R9 |
| lean-agentic-swarm.ts | agentdb | 183 | 70% | DEEP | Real concurrency, coordinator query-only | R9 |
| multi-agent-swarm.ts | agentdb | 147 | 30% | DEEP | Invalid test — no real contention | R9 |
| neural-augmentation.js | agentdb | 472 | 52% | DEEP | BIMODAL: graph infra 85-90%, neural 15-20% (GNN=Math.random(), RL=deterministic formula). Hardcoded "+29.4% improvement" | R57 |

### ruv-swarm npm Entry Point + SWE-bench Adapter (ruv-FANN)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| index.ts (npm/src) | ruv-swarm | 457 | 72-76% | DEEP | BIMODAL: production SDK 88-92% (events, topology, metrics), WASM integration 0% (API mismatch, Node fallback always executes). Genuine pure-JS fallback works | R57 |
| performance.js (npm/src) | ruv-swarm | 458 | 25-30% | DEEP | THEATRICAL. All WASM/swarm/neural metrics = Math.random(). optimize() = console.log + setTimeout. R53 scheduler.ts pattern extended | R57 |
| stream_parser.rs (swe-bench-adapter) | ruv-swarm | 439 | 75-80% | DEEP | COMPLETE MISLABELING. 0% SWE-bench — parses Claude Code CLI metrics. Genuine async streaming (tokio mpsc), multi-stream management. 4th mislabeled file | R57 |
| benchmarking.rs (swe-bench-adapter) | ruv-swarm | 430 | 20-25% | DEEP | THEATRICAL. simulate_execution() = sleep(10ms). Hardcoded memory/profile data. Valid statistics on fake data. Extends R43 benchmark deception | R57 |
| ruv-swarm-memory.js (bin) | ruv-swarm | 119 | 0% | DEEP | CLI DEMONSTRATION only, no persistent backend. Fabricated metrics. 8th DISCONNECTED PERSISTENCE LAYER | R84 |
| npm/scripts/build.js | ruv-swarm | 167 | 90-95% | DEEP | GENUINE Rust→WASM build orchestrator. Dual SIMD compilation, wasm-opt, TypeScript def generation. Real dependency validation. | R85 |
| npm/src/github-coordinator/claude-hooks.js | ruv-swarm | 162 | 71% | DEEP | BIMODAL: 5 real GitHub hook types but NO claude-flow/MCP connection. Placeholder conflict detection. In-memory state. | R85 |
| npm/src/hooks/cli.js | ruv-swarm | 82 | 75-80% | DEEP | Thin CLI wrapper delegating to index.js. 15+ hook types via JSON stdout protocol. Custom arg parser. Exit codes 0/1/2. | R85 |
| validate-error-handling.js | ruv-swarm npm | 143 | 60-65% | DEEP | Smoke test masquerading as validation suite. Tests component existence (assertions) but 0% error recovery testing. No MCP/agent/memory failure simulation | R86 |
| test-memory-storage.js | ruv-swarm npm | 57 | 88-92% | DEEP | Tests GENUINE SwarmPersistence backed by better-sqlite3. Multi-session memory with TTL. CONTRADICTS R84 single-session finding (test-mcp-persistence.js 40-50%) | R86 |
| test-pr34-local.js | ruv-swarm npm | 118 | 0% | DEEP | COMPLETE FACADE — imports non-existent src/onboarding/index.js. Phantom classes (DefaultClaudeDetector, MCPServerConfig). File crashes on import. 9th test facade | R87 |
| verify-db-updates.js | ruv-swarm npm | 57 | 88-92% | DEEP | GENUINE DB verification. Real better-sqlite3 queries against ruv-swarm.db (swarms, agents, tasks, agent_memory). Multi-session history. PARTIALLY REVERSES R85 0% assessment of ruv-swarm-memory.js — DB operations real, metrics reporting theatrical | R87 |
| test-wasm-loading.js | ruv-swarm npm | 48 | 95-98% | DEEP | GENUINE WASM test. Real WebAssembly.instantiate(), actual export invocation (create_swarm_orchestrator), memory inspection. Core WASM binary functional. Validates R84 build.rs. 3-strategy WasmModuleLoader (eager/progressive/on-demand) | R87 |
| memory-config.js | ruv-swarm npm | 42 | ~40% | DEEP | 9th DISCONNECTED MEMORY LAYER. Static config export with 6 hardcoded pattern profiles (250-300MB). No backend, no TTL, no lifecycle management. lazyLoading flag without implementation. In-memory only | R87 |
| env-template.js | ruv-swarm npm | 39 | ~35% | DEEP | Security-concerning defaults: RUV_SWARM_REMOTE_EXECUTION=true, AUTO_COMMIT=true, no Claude API key (confirms R43 "setup toolkit not API"). All features enabled by default, no conservative-by-default design | R87 |

### ruv-fann Benchmarking (ruv-FANN)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| claude_executor.rs | ruv-fann-rust | 387 | 75-80% | DEEP | BIMODAL: genuine process spawning (tokio::process::Command), async timeout/kill, batch parallel (buffer_unordered). BUT SWE-Bench extraction hardcoded zeros ("Would need to parse"). ORPHANED module. REVERSES R57 swe-bench-adapter theatrical pattern | R59 |
| metrics.rs | ruv-fann-rust | 383 | 88-92% | DEEP | GENUINE metrics infra. Instant::now() timing, p95/p99 percentile calculation, 14+ metric categories. Placeholder derived metrics (coordination overhead=0, ML inference=0, code quality hardcoded). Matches R55 performance_monitor.rs pattern | R59 |

### ruv-swarm-core Rust Crate (ruv-FANN)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| sqlite.rs | ruv-swarm | 1,016 | 92% | DEEP | r2d2 pooling, WAL, ACID. MOCK: PI*1000.0 timestamp | R21 |
| ensemble/mod.rs | ruv-swarm | 1,006 | 78% | DEEP | Real averaging. FAKE BMA. BROKEN Stacking | R21 |
| agent_forecasting/mod.rs | ruv-swarm | 813 | 65% | DEEP | Real EMA. Hardcoded model mapping | R21 |
| swe_bench_evaluator.rs | ruv-swarm | 991 | 35-40% | DEEP | Real orchestration, ALL metrics hardcoded | R21 |
| comprehensive_validation_report.rs | ruv-swarm | 1,198 | 45% | DEEP | Self-referential — sets simulation_ratio=0.60 | R21 |
| unit_tests.rs | ruv-swarm | 1,078 | 90-95% | DEEP | 48+ genuine tests: GOAP, A*, rule engine | R21 |

### ruv-swarm npm JS (ruv-swarm)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| mcp-tools-enhanced.js | ruv-swarm | 2,863 | 70% | DEEP | Real persistence/WASM, fabricated neural_train/agent_metrics/swarm_monitor | R16 |
| ruv-swarm-secure-heartbeat.js | ruv-swarm | 1,549 | 92% | DEEP | Production MCP stdio server. JSON-RPC 2.0, circuit breaker | R29 |
| daa-cognition.js | ruv-swarm | 977 | 88% | DEEP | Real Byzantine-tolerant consensus, distributed learning, emergent pattern detection | R29 |
| claude-flow-enhanced.js | ruv-swarm | 840 | 85% | DEEP | Real dependency graph + topological sort. SIMD speedup hardcoded | R29 |
| neural-agent.js | ruv-swarm | 830 | 84% | DEEP | Real neural network: Xavier init, forward/backward with momentum | R29 |
| mcp-daa-tools.js | ruv-swarm | 735 | 90% | DEEP | 10 MCP tools wrapping real daaService | R29 |
| schemas.js | ruv-swarm | 864 | 95% | DEEP | Production recursive validator with 25+ MCP tool schemas | R33 |
| MultiDatabaseCoordinator.js | ruv-swarm | 803 | 42% | DEEP | Sync simulated, 1% hardcoded conflict rate | R33 |
| wasm-memory-optimizer.js | ruv-swarm | 784 | 78% | DEEP | Real buddy allocator. SIMD functions are placeholders | R33 |
| index-enhanced.js | ruv-swarm | 734 | 65% | DEEP | Orchestrator with WASM fallback. Agent.execute() stub | R33 |
| persistence-pooled.js | ruv-swarm | 695 | 92% | DEEP | 8-table schema, exponential backoff, TTL cleanup, VACUUM | R33 |
| neural-coordination-protocol.js | ruv-swarm | 1,363 | 10-15% | DEEP | 8 coordination executions stubbed | R19 |
| neural-network-manager.js | ruv-swarm | 1,938 | 15-20% | DEEP | SimulatedNeuralNetwork uses Math.random() when WASM fails | R19 |
| hooks/index.js | ruv-swarm | 1,900 | 25-30% | DEEP | Real git commit, fabricated trainPatternsFromEdit | R19 |
| wasm-loader.js | ruv-swarm | 602 | 82% | DEEP | GENUINE WASM loader — real WebAssembly.instantiate(), wasm-bindgen, 4-strategy path resolution. Facade fallback only on total failure | R45 |
| sqlite-pool.js | ruv-swarm | 587 | 92% | DEEP | GENUINE production pool — WAL mode, worker threads, health monitoring, auto-recovery. Zero red flags | R45 |
| neural.js | ruv-swarm | 574 | 28% | DEEP | MOSTLY FACADE — WASM calls exist but returns IGNORED, overwritten with Math.random(). Training/export/patterns all fabricated | R45 |
| performance-benchmarks.js | ruv-swarm | 899 | 62% | DEEP | MIXED — SIMD/WASM/browser benchmarks real (78-95%), neural/Claude/parallel are setTimeout facades (15-20%) | R45 |
| mcp-workflows.js | ruv-swarm | 991 | 72% | DEEP | MOSTLY REAL — genuine JSON-RPC 2.0 client, 5 workflows. ORPHANED: Rust MCP backend disabled | R45 |
| generate-docs.js | ruv-swarm | 954 | 87% | DEEP | GENUINE regex-based source parser, extracts real API signatures. Only CLI docs hardcoded template | R45 |
| claude-simulator.js | ruv-swarm | 602 | 88% | DEEP | GENUINE MCP test client — WebSocket, Prometheus, chaos injection. PROVES self-contained demo loop | R45 |
| diagnostics.js | ruv-swarm | 533 | 87% | DEEP | GENUINE system diagnostics — real process.memoryUsage/cpuUsage/performance.now, process._getActiveHandles/_getActiveRequests for event loop monitoring. Pattern detection, actionable recommendations (thresholds: >10% failure, >500MB memory, >50 handles). Self-test harness | R48 |
| errors.js | ruv-swarm | 528 | 90% | DEEP | GENUINE error taxonomy — 11 typed error classes (Validation/Swarm/Agent/Task/Neural/Wasm/Network/Persistence/Resource/Concurrency + base). ErrorFactory pattern, ErrorContext enrichment. Used EXTENSIVELY by mcp-tools-enhanced.js (26 import sites). Context-aware getSuggestions() per error type | R48 |
| test-pr34-local.js | ruv-swarm | 118 | 0% | DEEP | COMPLETE FACADE — Imports non-existent src/onboarding/index.js module. DefaultClaudeDetector, MCPServerConfig, MCPConfig phantom classes. File fails to execute. Test suite unconditional pass if no errors thrown (smoke test only). PR#34 feature stub w/o implementation. | R87 |

### ruv-swarm Neural Models JS (ruv-swarm)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| lstm.js | ruv-swarm | 551 | 85% | DEEP | Correct 4-gate LSTM, bidirectional, Xavier init. Hardcoded accuracy 0.864 | R40 |
| transformer.js | ruv-swarm | 515 | 83% | DEEP | Correct multi-head attention, sinusoidal PE. LR formula inverted | R40 |
| gnn.js | ruv-swarm | 447 | 81% | DEEP | Genuine MPNN with GRU-gated updates. Hardcoded accuracy 0.96 | R40 |
| base.js | ruv-swarm | 269 | 75% | DEEP | Float32Array tensor system, matmul, activations, dropout. backward() stub | R40 |
| neural-presets-complete.js | ruv-swarm | 1,306 | 52-58% | DEEP | BIMODAL: 27 arch configs as lookup tables (35-40%) + real CognitivePatternSelector/NeuralAdaptationEngine (65-70%). 9th disconnected persistence (crossSessionMemory). Dead imports. SOURCE of R69 "27 ghost models" count | R91 |

### ruv-swarm-mcp Rust Crate (ruv-FANN)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| handlers.rs | ruv-swarm-mcp | 951 | 0% | DEEP | Won't compile — ~12 API mismatches with orchestrator | R28 |
| orchestrator.rs | ruv-swarm-mcp | 594 | 90-92% | DEEP | Real SQLite persistence, hybrid metrics, agent ID mismatch | R31 |
| lib.rs | ruv-swarm-mcp | 494 | 30-35% | DEEP | 85% commented out. WebSocket handler disabled | R31 |
| tools.rs | ruv-swarm-mcp | 482 | 95-98% | DEEP | 11 tool schemas, production-ready. All handlers None | R31 |
| validation.rs | ruv-swarm-mcp | 479 | 92-95% | DEEP | Path traversal protection, null byte prevention. Schema mismatch (4 vs 6 strategies) | R31 |
| limits.rs | ruv-swarm-mcp | 449 | 90% | DEEP | Production enforcement. Absurd defaults (10M agents, 1TB memory, 30 days) | R34 |

### ruv-swarm-daa Rust Crate (ruv-FANN)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| gpu.rs | ruv-swarm-daa | 901 | 15% | DEEP | Dead code — syntax error, not in module tree, phantom webgpu feature | R28 |
| daa_gpu_agent_framework.rs | ruv-swarm-daa | 856 | 5-8% | DEEP | ZERO GPU ops. All 11 types from ruv_fann::webgpu don't exist | R29 |
| learning.rs | ruv-swarm-daa | 806 | 60-70% | DEEP | Best file. Proficiency EMA real. 5 adaptation strategies hardcoded. Memory leak | R29 |
| coordination_protocols.rs | ruv-swarm-daa | 762 | 30% | DEEP | seek_consensus() sets consensus_reached=true unconditionally | R29 |
| agent.rs | ruv-swarm-daa | 758 | 50-60% | DEEP | Lifecycle real. All 6 cognitive process_* methods return Ok(true) immediately | R29 |
| adaptation.rs | ruv-swarm-daa | 735 | 20-30% | DEEP | Traits only. NaN bug in normalize(). Async/sync collision | R29 |
| bin/daa-coordinator.rs | ruv-swarm-daa | 465 | 65% | DEEP | Daemon skeleton. select_optimal_agent() returns first HashMap key | R34 |
| lib.rs (DAA) | ruv-swarm-daa | 460 | 55% | DEEP | orchestrate_task() hardcodes success:true, 100ms, 0.95 efficiency | R34 |
| traits.rs | ruv-swarm-daa | 402 | 80% | DEEP | 5 sophisticated traits, ZERO implementations | R34 |
| types.rs | ruv-swarm-daa | 132 | 78-82% | DEEP | RESOLVES R69 — AgentType = 5 ROLES not 27 models. DecisionContext production-quality. NeuralNetworkManager STUB | R82 |
| neural.rs | ruv-swarm-daa | 94 | 0-5% | DEEP | PURE METADATA FACADE. HashMap storage, ZERO computation. Confirms R69 GHOST WASM | R82 |
| patterns.rs | ruv-swarm-daa | 81 | 15-20% | DEEP | 6 cognitive styles defined, 67% undefined. Zero behavior logic | R82 |
| telemetry.rs | ruv-swarm-daa | 45 | 25-35% | DEEP | FACADE — no real OpenTelemetry. Just Vec<TelemetryEvent> wrapped in Arc<RwLock>. Unbounded buffer, type-unsafe metadata. DAA facade tier (joins neural.rs, coordination.rs) | R86 |

### ruv-swarm-wasm Rust Crate (ruv-FANN)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| neural_swarm_coordinator.rs | ruv-swarm-wasm | 791 | 15-20% | DEEP | All 4 training modes return hardcoded loss curves | R29 |
| swarm_orchestration_wasm.rs | ruv-swarm-wasm | 757 | 20-25% | DEEP | execute_distributed_task() always returns {status:"initiated"} | R29 |
| utils.rs | ruv-swarm-wasm | 300 | 88% | DEEP | GENUINE WASM utility — JS interop, feature detection, performance timing, portable SIMD via wide::f32x4. SIMD runtime detection stub (returns true). Memory usage placeholder (hardcoded 64KB) | R72 |
| neural.rs | ruv-swarm-wasm | 155 | 85-90% | DEEP | REVERSES R80 activation.rs — correct ruv-fann API (NetworkBuilder, run(), get_weights/set_weights, 17 activations). Inference-only. Metrics facade | R82 |

### ruv-swarm-cli Rust Crate (ruv-FANN)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| config.rs | ruv-swarm-cli | 335 | 88-92% | DEEP | PRODUCTION config — 5-layer hierarchical loading (defaults→global→profile→custom→env), 6 structs, confirms R70's 3 backends. Profile-based init (dev/prod/test). Two type-safety gaps (String not enum) | R72 |
| main.rs | ruv-swarm-cli | 308 | 82-86% | DEEP | BIMODAL: clap 4.5 CLI (90-95%) with 5 commands, shell completions, tracing, env vars. BUT ZERO core crate integration — sleep() execution, "Simulated result" JSON. Extends R31 demo framework to Rust | R72 |
| monitor.rs | ruv-swarm-cli | 68 | 5-10% | DEEP | CLI skeleton — display_monitoring_data() = "not yet implemented" placeholder. Watch mode loops but collects zero metrics. Confirms R31/R71 "CLI = demo framework" | R86 |

### ruv-swarm-daa WASM (ruv-FANN)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| wasm_simple.rs | ruv-swarm-daa | 268 | 22-28% | DEEP | THEATRICAL WASM — genuine wasm-bindgen but facade functionality. Decision=string formatting, adaptation=if-else multiplier. Inherits R69 memory.rs 0-5% pattern. Two parallel WASM impls (wasm.rs 735 LOC vs this) | R72 |

### ruv-swarm-mcp Service Layer (ruv-FANN)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| service.rs | ruv-swarm-mcp | 256 | 88-92% | DEEP | GENUINE MCP — rmcp SDK v0.2.1, THIRD MCP protocol (macro-based #[tool]). 11 domain-specific tools. 2-layer delegation: orchestrator → core crate + persistence. In-memory session storage (not persistent) | R72 |
| main.rs | ruv-swarm-mcp | 46 | 35-45% | DEEP | 6th MCP protocol — rmcp SDK v0.2.1. HTTP/WebSocket INCOMPLETE (7 modules disabled in lib.rs), stdio FUNCTIONAL. Tool registry defined (11 tools) but implementation stubs | R86 |
| stdio.rs | ruv-swarm-mcp | 36 | 90-95% | DEEP | GENUINE MCP binary — correct stderr logging for protocol integrity, current_thread tokio, service composition: SwarmOrchestrator→RealSwarmService→rmcp stdio transport | R86 |
| lib.rs (WASM) | ruv-swarm-wasm | 722 | 40-50% | DEEP | WasmNeuralNetwork forward pass REAL (17 activations). Forecasting naive | R29 |
| learning_integration.rs | ruv-swarm-wasm | 736 | 30-40% | DEEP | "GPU" methods have ZERO GPU ops. All 4 optimization algorithms return pattern.clone() | R29 |
| wasm.rs (DAA) | ruv-swarm-wasm | 736 | 45-55% | DEEP | Agent management genuine. Resource optimize() cosmetic | R29 |
| simd_ops.rs | ruv-swarm-wasm | 419 | 72-78% | DEEP | Real portable SIMD via `wide::f32x4` (not native intrinsics). 8 real SIMD operations (dot product, add, scale, relu, sigmoid, tanh). Matrix multiply is SCALAR triple-loop. Downgraded vs ruvector-core (fixed 4-wide, no AVX-512/AVX2 specialization). Real benchmarking infra | R50 |
| simd_optimizer.rs | ruv-swarm-wasm | 595 | 85-90% | DEEP | BEST SIMD — real f32x4 WASM SIMD128. tanh/gelu SCALAR despite names | R31 |
| cognitive_diversity_wasm.rs | ruv-swarm-wasm | 639 | 75-80% | DEEP | Real Shannon diversity, 5 cognitive patterns. Optimization plan hardcoded +0.3/+0.2/+0.15 | R31 |
| agent_neural.rs | ruv-swarm-wasm | 552 | 80-85% | DEEP | Genuine ruv_fann bridge. Trains IncrementalBackprop. 4/5 metrics placeholders | R31 |
| cognitive_neural_architectures.rs | ruv-swarm-wasm | 482 | 60-65% | DEEP | Detailed encoder/processor specs. IntegrationStrategy NEVER USED | R31 |

### ruv-swarm-wasm-unified Rust Crate (ruv-FANN)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| utils/simd.rs | ruv-swarm-wasm-unified | 369 | 75% | DEEP | REAL WASM SIMD128 for add/mul/dot/relu. Sigmoid and matmul NOT SIMD | R22 |
| utils/bridge.rs | ruv-swarm-wasm-unified | 224 | 80% | DEEP | Genuine JS↔Rust type conversion, SharedArrayBuffer | R22 |
| utils/memory.rs | ruv-swarm-wasm-unified | 183 | 35% | DEEP | Pool creation real but allocate/deallocate/compact all no-ops | R22 |
| core/agent.rs | ruv-swarm-wasm-unified | 147 | 55% | DEEP | Wraps DynamicAgent. Cognitive patterns FAKE (set no-op, get returns "convergent") | R22 |
| utils/mod.rs | ruv-swarm-wasm-unified | 121 | 75% | DEEP | Genuine SIMD/Worker detection, real memory usage | R22 |
| lib.rs | ruv-swarm-wasm-unified | 85 | 70% | DEEP | Standard WASM config | R22 |
| neural/mod.rs | ruv-swarm-wasm-unified | 27 | 5% | DEEP | EMPTY STUB — JS glue advertises 18 activations, Rust is empty | R22 |
| forecasting/mod.rs | ruv-swarm-wasm-unified | 25 | 5% | DEEP | EMPTY STUB — lists 10 models, implements zero | R22 |

### ruv-swarm-transport Rust Crate (ruv-FANN)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| protocol.rs | ruv-swarm-transport | 378 | 92-95% | DEEP | PRODUCTION WIRE PROTOCOL. MessagePack (rmp_serde) + JSON dual codecs, complete state machine (handshake/flow-control/compression/disconnect), distributed RPC with UUID correlation IDs, TTL-based routing, priority queuing (0-255). Used by 3 transport backends | R50 |
| websocket.rs | ruv-swarm-transport | 678 | 88-92% | DEEP | Production — exponential backoff, gzip, real-time stats. 137-line code duplication | R31 |
| shared_memory.rs | ruv-swarm-transport | 482 | 85-88% | DEEP | Ring buffer with atomic head/tail. Misleadingly named "lock-free" (uses Mutex). 1ms polling | R31 |
| in_process.rs | ruv-swarm-transport | 424 | 92% | DEEP | BEST transport. DashMap registry, mpsc+broadcast, bincode validation | R34 |

### ruv-swarm-benchmarking Rust Crate (ruv-FANN)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| storage.rs | ruv-swarm-benchmarking | 795 | 95-98% | DEEP | BEST SQL in batch. 10 normalized tables, CHECK constraints, 9 indexes | R31 |
| comparator.rs | ruv-swarm-benchmarking | 584 | 88-92% | DEEP | Real Welch's t-test, Cohen's d. n=1 comparisons hardcoded p_value=0.01 | R31 |
| stream_parser.rs | ruv-swarm-benchmarking | 602 | 85-90% | DEEP | Parses Claude Code stream-json. Thinking duration hardcoded 50ms/token | R31 |
| realtime.rs | ruv-swarm-benchmarking | 521 | 85-90% | DEEP | Production Axum WebSocket server. Missing static/monitor.html | R31 |
| lib.rs | ruv-swarm-benchmarking | 552 | 75-80% | DEEP | build_command() generates ENGLISH PROMPTS not CLI flags. Cannot execute | R31 |

### ruv-swarm-ml Rust Crate (ruv-FANN)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| time_series/mod.rs | ruv-swarm-ml | 612 | 90-92% | DEEP | 7 genuine transformations, real autocorrelation. Seasonality strength hardcoded 0.5 | R31 |
| models/mod.rs | ruv-swarm-ml | 642 | 70-75% | DEEP | 27 SOTA model metadata. create_model() delegates to unknown neural_models | R31 |
| gpu_learning_engine.rs | ruv-swarm-ml | 1,628 | 5-10% | DEEP | ZERO GPU ops. 27+ models promised, 0 implemented. 280+ struct defaults | R19 |
| swarm_coordinator_training.rs | ruv-swarm-ml | 1,838 | 25-35% | DEEP | Real GNN/attention/Q-learning/VAE algorithms. ALL 5 metrics hardcoded | R19 |
| ml-training/lib.rs | ruv-swarm-ml | 1,371 | 30-40% | DEEP | Real LSTM/TCN/N-BEATS skeletons. Fake LCG random | R19 |

### ruv-swarm-persistence Rust Crate (ruv-FANN)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| memory.rs | ruv-swarm-persistence | 434 | 95-98% | DEEP | PRODUCTION-QUALITY. 28/28 Storage trait methods, parking_lot::RwLock, atomic task claiming, multi-dimensional event ordering. Test harness for concurrent_tests.rs (100 agents). Three-backend architecture (Memory/SQLite/WASM) | R50 |
| wasm.rs (persistence) | ruv-swarm-persistence | 694 | 95% | DEEP | Production IndexedDB via rexie. Only get_storage_size() stub | R31 |
| lib.rs (persistence) | ruv-swarm-persistence | 250 | 88-92% | DEEP | Production trait-based persistence. 28 async CRUD methods via Storage trait, 3 backends (SQLite/IndexedDB/in-memory), QueryBuilder with SQL injection prevention, connection pooling | R70 |
| models.rs | ruv-swarm-persistence | 333 | 92-95% | DEEP | Production 5-table models, serde, builder pattern, retry logic | R71 |
| migrations.rs | ruv-swarm-persistence | 334 | 92-95% | DEEP | Production SQLite migrations. CREATE TABLE IF NOT EXISTS, foreign keys, timestamps, UUIDs. 3-backend architecture (SQLite/WASM/Memory). COMPLETES persistence crate picture | R69 |

### ruv-swarm SWE-bench Adapter + CLI (ruv-FANN)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| prompts.rs | ruv-swarm-adapter | 534 | 98% | DEEP | BEST quality file. 4 difficulty-based Claude Code prompts | R31 |
| evaluation.rs | swe-bench-adapter | 469 | 62% | DEEP | BIMODAL: git patch/test execution 85-95% REAL (TokioCommand, git apply, sandbox). Dataset loading 0% (mock instances, hardcoded results). SWE-Bench = theatrical evaluation | R50 |
| loader.rs | ruv-swarm-adapter | 493 | 75% | DEEP | Difficulty scoring real. download_instance() returns MOCK data | R31 |
| lib.rs (adapter) | ruv-swarm-adapter | 580 | 70% | DEEP | Framework complete. evaluate_instance() hardcoded mock | R31 |
| spawn.rs | ruv-swarm-cli | 412 | 8-12% | DEEP | COMPLETE FACADE. ZERO process spawning — all 5 operations are tokio::time::sleep() delays. Agents = JSON metadata objects. Comment admits "In a real implementation". DEFINITIVELY CONFIRMS R31 | R50 |
| init.rs | ruv-swarm-cli | 538 | 65% | DEEP | Interactive config real. Actual spawning simulated (sleep) | R31 |
| status.rs | ruv-swarm-cli | 687 | 60% | DEEP | Display logic production-ready. Loads stale JSON not live state | R31 |
| orchestrate.rs | ruv-swarm-cli | 662 | 45% | DEEP | 4 strategies architecturally correct. execute_subtask() sleeps 1-2s | R31 |

### claude-parser Rust Crate (ruv-FANN)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| lib.rs | claude-parser | 788 | 85-88% | DEEP | Claude Code stream-json parser. Metric estimates hardcoded. 8 tests | R31 |

### ruv-swarm Claude Integration (ruv-FANN)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| claude-integration/index.js | ruv-swarm | 209 | 72% | DEEP | MIXED — orchestrator wiring core/docs/remote. No real API calls, delegates to Claude CLI via execSync | R43 |
| claude-integration/advanced-commands.js | ruv-swarm | 561 | 83% | DEEP | REAL GENERATOR — creates 9 markdown files in .claude/commands/. Content is aspirational templates | R43 |
| claude-integration/remote.js | ruv-swarm | 408 | 15% | DEEP | COMPLETE FACADE — zero network transport. Generates local wrapper scripts (bash/batch/PowerShell). "Remote" = SSH env detection only | R43 |
| claude-integration/docs.js | ruv-swarm | 1,548 | 78% | DEEP | GENUINE GENERATOR — real file merging, backups, writes 20+ command files. 41% functional code, 32% templates | R43 |
| claude-integration/core.js | ruv-swarm | 112 | 72-76% | DEEP | CLI WRAPPER not MCP client. **CRITICAL: defaults to --dangerously-skip-permissions**. MCP registration via `claude mcp add`. Orphaned file checks for artifacts never created by this module | R66 |

### ruv-swarm npm Runtime Layer (ruv-FANN)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| npm/src/index.js | ruv-swarm | 405 | 28-32% | DEEP | **PHANTOM API WRAPPER**. WASM API mismatch (expects RuvSwarm class, WASM exports create_swarm_orchestrator function). Namespace collision with index-enhanced.js. WorkerPool 100% TODO stubs. WASMLoader SIMD detection genuine (75%). Confirms R57 WASM API mismatch | R66 |
| npm/src/utils.ts | ruv-swarm | 286 | 85-90% | DEEP | **GENUINE TS, 0% RUST INTEGRATION**. 9 utility functions all work correctly. deepClone handles Date/Array/Map/Set. retryWithBackoff exponential. recommendTopology heuristic if-else. Hardcoded cognitive profiles (10 agent types, static 0.0-1.0). Imports only TS types, zero FFI/WASM | R66 |
| npm/src/logging-config.js | ruv-swarm | 179 | 75-80% | DEEP | GENUINE structured logging. 10 component namespaces (mcp-server, swarm-core, agent, neural, etc.). Singleton logger factory. Correlation ID child loggers for distributed tracing. Runtime reconfiguration. MCP stdio mode integration. NO swarm-wide log aggregation | R66 |
| npm/src/types.ts | ruv-swarm | 164 | 88-92% | DEEP | **PRODUCTION-QUALITY type definitions**. Novel 6D CognitiveProfile (analytical/creative/systematic/intuitive/collaborative/independent). WasmModule interface defines full lifecycle (createSwarm/addAgent/assignTask/getState/destroy). AgentMemory in-memory only (Map, zero HNSW/AgentDB). SwarmState CENTRALIZED coordinator pattern. 9 event types on SwarmEventEmitter | R66 |
| npm/src/singleton-container.js | ruv-swarm | 183 | ~85% | DEEP | **GENUINE IoC CONTAINER** — factory registration, lazy singletons, dependency chain resolution, proper process cleanup. Does NOT wire 6 routing systems (ADR-008/LLMRouter/RuvLLMOrchestrator/ProviderManager/SemanticRouter/ModelRouter) — pure framework with zero hardcoded registrations. Missing circular dependency detection | R81 |
| npm/src/security.js | ruv-swarm | 218 | ~65% | DEEP | **MIXED** — real SHA256 hashing, SecurityError class. BUT WASM integrity bypass (checksums.json writeable, no signature verification), insufficient command sanitizer (regex too permissive), DependencyVerifier checks version only (no checksums/hashes/signatures) | R81 |
| npm/src/neural-models/index.js | ruv-swarm | 273 | ~15% | DEEP | **BARREL + TRAINING FACADE** — exports 8 model classes (Transformer, CNN, GRU, Autoencoder, GNN, ResNet, VAE, LSTM) but ZERO Rust integration. Pure JS reimplementation, NOT wrapper. backward() in base.js only logs, does NOT update weights. 234 lines MODEL_PRESETS dead documentation. Contradicts R40 inference-bridge assumption | R81 |
| npm/src/github-coordinator/gh-cli-coordinator.js | ruv-swarm | 260 | 88-92% | DEEP | **PRODUCTION GitHub CLI wrapper** — real execSync('gh ...') calls for all GitHub operations. Issue-based swarm task distribution with swarm-* label prefix. 1-hour lock expiry via SQLite. Five sequential claimTask operations (label, comment, DB record). NOT connected to 6 routing systems (operates at different abstraction layer: GitHub API distribution vs model selection). Infrastructure-ready task distribution layer | R81 |
| crates/ruv-swarm-wasm/src/swarm.rs | ruv-swarm | 190 | 0% | DEEP | **ORPHANED COMPILATION ERROR**. Imports non-existent enums AgentType, CoordinationMode, SwarmStrategy from crate root. RuvSwarm struct complete but NOT exported from lib.rs. Methods have TODO comments with placeholder results. Zero usage | R79 |
| crates/ruv-swarm-wasm/src/simd_tests.rs | ruv-swarm | 273 | 88-92% | DEEP | **GENUINE WASM SIMD** — wasm_bindgen_test, 5 real test functions with math assertions (tolerance checks). 3 exported JS-callable suites. Tests simd_ops.rs. Incomplete WasmNeuralNetwork test | R79 |
| crates/ruv-swarm-wasm/src/training.rs | ruv-swarm | 253 | ~80% | DEEP | **GENUINE 4/5** — IncrementalBackprop, BatchBackprop, Rprop, Quickprop real from ruv_fann. SARPROP silent fallback to RPROP. wasm-bindgen correct. Training history + convergence loop | R79 |
| crates/ruv-swarm-wasm/src/agent.rs | ruv-swarm | 200 | 28-32% | DEEP | **FACADE** — JsAgent.execute() mock setTimeout (100ms). Violates core Agent trait (sync vs async, missing health_check/status). get_capabilities() hardcoded per enum. agent_neural.rs uses base only as ID carrier | R79 |
| crates/ruv-swarm-wasm/src/memory_pool.rs | ruv-swarm | 185 | 78-82% | DEEP | **GENUINE 3-tier pool** (64KB/256KB/1MB) with wasm-bindgen. total_allocated never decrements. Silent block rejection on size mismatch. Hardcoded 50/30/10 blocks. fill(0) security good | R79 |
| crates/ruv-swarm-wasm/src/activation.rs | ruv-swarm | 82 | 35-40% | DEEP | **BROKEN** — 2 API mismatches (wrong method name + wrong param count). 18/25 activation functions. Genuine design, broken execution | R80 |
| crates/ruv-swarm-ml/src/models/neural_bridge.rs | ruv-swarm | 234 | 87% | DEEP | **GENUINE ruv-fann bridge** — Adam optimizer, sliding window time series, ModelType factory dispatch. predict() returns zeros (stub), load_parameters() no-op | R79 |
| crates/ruv-swarm-mcp/src/error.rs | ruv-swarm | 194 | 88-92% | DEEP | **Protocol-agnostic error bridge** — triple representation (string/JSON-RPC/HTTP). Session-aware tracing with credential stripping. String-matching classification fragile | R79 |
| crates/ruv-swarm-transport/src/lib.rs | ruv-swarm | 178 | 90%+ | DEEP | **Clean barrel** — polymorphic async Transport trait (8 methods), DashMap runtime registry, 7 error variants. No feature gates. COMPLETES transport/src/ (5/5 DEEP) | R79 |
| crates/ruv-swarm-wasm/src/cascade.rs | ruv-swarm | 154 | 88-92% | DEEP | **15th GENUINE WASM**. Real cascade correlation algorithm via ruv_fann::CascadeTrainer. Proper wasm-bindgen with JsValue marshaling. 12 hyperparameters (num_candidates=8, 5 activation functions). Sound algorithm (candidate pool rotation, correlation-driven selection, dual learning phases). Adaptive hidden neuron growth (max 30) | R83 |
| npm/src/sqlite-worker.js | ruv-swarm | 141 | 45-50% | DEEP | **PARTIAL FACADE — 7th DISCONNECTED PERSISTENCE LAYER**. Worker_threads architecture genuine (parentPort, message passing, cleanup handlers). BUT DB opened readonly while accepting INSERT/UPDATE/DELETE commands (functionally broken for writes). WAL/mmap config wasted on readonly. No connection pooling (single instance). Zero integration with npm runtime memory system. Unbounded prepared statement cache | R83 |
| npm/src/mcp-tools-benchmarks.js | ruv-swarm | 328 | MISLABELED | DEEP | **NOT MCP benchmarks** — 8th mislabeled file. Generic JS micro-benchmarks (array alloc, JSON parse, matrix multiply). Sound performance.now() but wrong target. Zero MCP tool invocations | R79 |
| npm/src/benchmark.js | ruv-swarm | 267 | 0-5% | DEEP | **DEEPEST FABRICATION** — 100% setTimeout synthetic data. All 6 benchmarks pass by design (delays < targets). RuvSwarm initialized but never used. WORSE than R59 standalone pattern | R79 |

### Python ML Training (sublinear-time-solver)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| train_ensemble_improved.py | sublinear-time-solver | 969 | 85% | DEEP | BEST Python. GATConv, NoisyLinear, Beta-VAE with curriculum learning | R33 |
| hyperparameter_optimizer.py | sublinear-time-solver | 858 | 68% | DEEP | Real Bayesian GP. ALL 5 model evaluators SIMULATED | R33 |
| train_lstm_coding_optimizer.py | sublinear-time-solver | 853 | 78% | DEEP | Real seq2seq with Luong attention. Data = hardcoded templates | R33 |
| enhanced_strategies.py | sublinear-time-solver | 820 | 62% | DEEP | 4 real decomposition strategies. MockModel returns random | R33 |
| train_ensemble.py | sublinear-time-solver | 819 | 71% | DEEP | Base version. Real ensemble. RL uses np.random.normal(0.5,0.2) | R33 |

### ruvllm Workflow Execution (ruvllm)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| claude_integration.rs | ruvllm | 1,344 | 70-75% | DEEP | Complete workflow orchestrator. execute_workflow() hardcodes 500 tokens | R37 |

### V3 Execution Engine (claude-flow V3) (R140)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| v3/@claude-flow/cli/src/services/headless-worker-executor.ts | claude-flow | 1,342 | 78-83% | DEEP | **GENUINE subprocess executor** — spawns `claude --print <prompt>` via child_process.spawn. Process pool (maxConcurrent=2), pending queue, context caching, EventEmitter monitoring all genuine. ZERO MCP protocol, ZERO memory/AgentDB. Prompt injected as raw CLI arg (argument injection risk). Double-timeout bug (two independent setTimeout on same process). audit worker ships .env* in contextPatterns (secrets to Anthropic API + local logs). 6/8 worker types disabled=false by default | R140 |
| v3/@claude-flow/cli/src/services/worker-daemon.ts | claude-flow | 942 | 65-70% | DEEP | **NOT a daemon** — foreground Node.js class with setTimeout scheduling, no fork/PID file/backgrounding. 9/12 local worker implementations are FACADE stubs (write static JSON). Full integration with headless-worker-executor.ts with graceful fallback. maxConcurrent=2, pending queue. State persistence to .claude-flow/daemon-state.json | R140 |
| v3/@claude-flow/cli/src/services/claim-service.ts | claude-flow | 1,118 | 70-75% | DEEP | **File-based JSON persistence** to .claude-flow/claims/claims.json. No distributed coordination — single-machine only. COMPETES with claims-tools.ts (MCP) — two independent implementations of same logic. Incompatible claimant formats (2-part vs 3-part) cause cross-system verification failures. First-writer-wins with no atomicity guarantees. rebalance() stub (moved array always empty) | R140 |
| v3/@claude-flow/cli/src/services/container-worker-pool.ts | claude-flow | 783 | 72-78% | DEEP | **Real Docker CLI integration** via child_process (not dockerode). Dynamic pool: min=1, max=3, idle timeout 5min. Three-tier chain: ContainerWorkerPool → worker-daemon → HeadlessWorkerExecutor. CRITICAL BUG: prompt and contextPatterns silently dropped in buildWorkerCommand(). ANTHROPIC_API_KEY visible via docker inspect (-e flag). Hardcoded image ghcr.io/ruvnet/claude-flow-headless:latest | R140 |
