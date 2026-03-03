# Genuine Assets — Exact File Paths for Extraction

> Copy these files verbatim into claude-flow v4. Every file listed here scored 85%+ realness across 90 research sessions.

## Rust Crates (~/repos/)

### ruvector-core (~/repos/ruvector/crates/ruvector-core/)

```
# Core HNSW (92-98%)
src/hnsw.rs
src/simd.rs

# Advanced features (85-93%)
src/advanced_features/product_quantization.rs    # 551 LOC, 88-92%
src/advanced_features/conformal_prediction.rs    # 505 LOC, 88-93%
src/advanced/hypergraph.rs                       # 551 LOC, 85-90%

# DO NOT COPY:
# src/advanced/tda.rs — 60-70%, mislabeled (no persistent homology)
```

### temporal-tensor (~/repos/sublinear-time-solver/crates/temporal-tensor/)

```
# ENTIRE CRATE — 93%, 213 tests, production-ready
# Copy the whole directory
```

### ruQu quantum (~/repos/sublinear-time-solver/crates/ruqu-core/)

```
# Core QEC (91-95%)
# Copy the whole crate — decoder.rs is 95-98%
```

### backward_push (~/repos/sublinear-time-solver/)

```
# Genuine O(1/epsilon) sublinear PageRank (95%+)
src/sublinear/backward_push.rs
```

### bit-parallel-search (~/repos/sublinear-time-solver/crates/bit-parallel-search/)

```
# Textbook Shift-Or (92-95%)
bit-parallel-search/src/lib.rs    # 198 LOC
```

### micro_lora + EWC++ (~/repos/sublinear-time-solver/)

```
# NEON SIMD LoRA with continual learning (92-95%)
crates/*/src/micro_lora.rs
```

### RAC consensus (~/repos/sublinear-time-solver/)

```
# Raft + real libp2p (92%)
# Includes p2p.rs which was confirmed genuine in R44
```

### RVF Store + Runtime (~/repos/ruvector/crates/rvf-runtime/ and rvf-store/)

```
# Cryptographic witness chains — R122-R124 (85-92%)
store.rs           # Core .rvf store operations
witness.rs         # SHAKE-256 witness chain, tamper-evident audit trail
write_path.rs      # Write path with provenance
hnsw.rs            # HNSW within RVF container
```

### rvf-node NAPI Bridge (~/repos/ruvector/crates/rvf-node/)

```
# Native Node.js bridge for RVF — R121 (85-90%)
lib.rs             # napi-rs bindings for RVF operations
```

### ruvector-domain-expansion (~/repos/ruvector/crates/ruvector-domain-expansion/)

```
# Thompson Sampling + cross-domain transfer — R128 (82-88%)
# Copy the whole crate
```

### ruvector-raft (~/repos/ruvector/crates/ruvector-raft/)

```
# Raft consensus — R129 (78-85%)
# Separate from RAC implementation in sublinear-time-solver
```

### shard partitioner (~/repos/ruvector/crates/ruvector-graph/)

```
# EdgeCutMinimizer with xxh3/blake3 (70-80%)
# Only the partitioning algorithms — NOT the transport stubs
src/distributed/shard.rs    # 596 LOC
```

### temporal-compare (~/repos/sublinear-time-solver/crates/temporal-compare/)

```
# Stochastic AR(1) with regime shifts (92-95%)
src/data.rs     # 92-95%
src/baseline.rs # 90-95%
```

## TypeScript/JavaScript

### RuVectorBackend (~/node_modules/agentdb/src/backends/ruvector/)

```
# Genuine HNSW integration (88-92%)
index.ts              # 10 LOC barrel
RuVectorBackend.ts    # ~500 LOC — adaptive HNSW, Semaphore, BufferPool, path security
```

### ReasoningBank TS (various locations in claude-flow-cli)

```
# Statistical ranking (92-95%)
# Search for: reasoningbank-types.ts, pre-task.ts, async_learner
```

### sona-optimizer.ts (claude-flow-cli dist/)

```
# Genuinely functional Bayesian agent-routing — R140 (72-78%)
# ONLY V3 memory subsystem wired into hooks pipeline
# Real Thompson sampling, temporal decay, Bayesian updates
sona-optimizer.ts    # ~842 LOC
```

### onnx-embedder.ts (ruvector umbrella package)

```
# REAL ONNX embeddings via Tract/WASM — R117 (85-90%)
# Potential R20 fix without building from scratch
onnx-embedder.ts     # ~400 LOC
```

### HeadlessWorkerExecutor (claude-flow-cli dist/)

```
# Real process pool with context caching — R140 (78-83%)
# CRITICAL BUG: buildWorkerCommand() drops prompt+contextPatterns
# Fix the bug, reuse the pool architecture
```

### Self-implemented DDD (~/claude-flow-self-implemented/src/agentdb-integration/)

```
# Infrastructure adapters (ADAPT — strip upstream imports)
infrastructure/adapters/ruvector-backend-adapter.ts     # 374 LOC, DEEP
infrastructure/adapters/vector-backend-adapter.ts       # 230 LOC, DEEP
infrastructure/adapters/embedding-adapter.ts            # 170 LOC, DEEP
infrastructure/adapters/real-embedding-adapter.ts       # 153 LOC, DEEP
infrastructure/adapters/database-adapter.ts             # 215 LOC
infrastructure/schema/schema-migrator.ts                # 233 LOC
infrastructure/schema/agentdb-schema.sql                # 167 LOC
infrastructure/factories.ts                             # 179 LOC

# Episodic/reflexion (ADAPT)
episodic/services/reflexion-service.ts                  # 330 LOC, DEEP
episodic/adapters/reflexion-memory-adapter.ts           # 328 LOC, DEEP
episodic/repositories/episode-repository.ts             # 322 LOC
episodic/aggregates/episode.ts                          # 92 LOC

# Skill library (ADAPT)
skill/services/skill-library-service.ts                 # 483 LOC
skill/repositories/skill-repository.ts                  # 331 LOC
skill/services/consolidation-service.ts                 # 273 LOC
skill/adapters/skill-library-adapter.ts                 # 269 LOC

# Search pipeline (ADAPT)
search/services/hybrid-search-service.ts                # 366 LOC
search/aggregates/search-pipeline.ts                    # 287 LOC
search/services/bm25-index.ts                           # 269 LOC
search/repositories/search-log-repository.ts            # 223 LOC
search/adapters/mmr-adapter.ts                          # 141 LOC

# Security (KEEP AS-IS)
security/input-validator.ts                             # 270 LOC

# Events (KEEP AS-IS)
events/domain-events.ts                                 # 217 LOC
events/event-bus.ts                                     # 138 LOC

# MCP tools (RETARGET to new MCP server)
mcp-tools/mcp-search-hybrid.ts                          # 101 LOC
mcp-tools/mcp-skill-suggest.ts                          # 94 LOC
mcp-tools/mcp-reflexion-retrieve.ts                     # 93 LOC, DEEP
mcp-tools/mcp-reflexion-store.ts                        # 92 LOC, DEEP
mcp-tools/mcp-skill-extract.ts                          # 90 LOC

# Types (KEEP)
types/skill.types.ts                                    # 170 LOC
types/common.types.ts                                   # 154 LOC
types/episodic.types.ts                                 # 152 LOC
types/search.types.ts                                   # 124 LOC

# Tests (~/claude-flow-self-implemented/tests/) — 49 files, 9,075 LOC
# ADAPT all test files to new interfaces
```

## Synthesis-Doc Assets (Not Previously Listed)

> Added from 14 domain synthesis documents (~10,800 lines of accumulated knowledge).
> Scores marked (est.) are qualitative "REAL/production-grade" classifications without formal numeric assessment.

### Hook Pipeline (~/repos/claude-flow/ dist/)

```
# standard-checkpoint-hooks.sh (190 LOC, REAL est. 90-95%)
# Git checkpoint system: stash, tags, metadata JSON, session summaries
# Edge case: L81 git diff HEAD~1 fails on first commit (MEDIUM)
standard-checkpoint-hooks.sh

# ralph-loop stop-hook.sh (178 LOC, REAL est. 90-95%)
# Atomic file updates, Perl regex
stop-hook.sh

# intelligence.cjs (917 LOC, REAL est. 85-90%)
# Actual PageRank with power iteration. Genuine graph algorithm.
intelligence.cjs

# learning-service.mjs (1,145 LOC, REAL est. 85-90%)
# Working HNSW search infrastructure + SQLite persistence
# CAVEAT: embedding function is Math.sin(seed) mock per ruvector+memory-and-learning domains
# The HNSW/SQLite infrastructure is real; the embeddings feeding it are not
learning-service.mjs
```

### Agent Lifecycle (~/repos/claude-flow/ dist/)

```
# LongRunningAgent (220 LOC) — budget enforcement, checkpointing, provider failover
# Real budget enforcement and graceful shutdown
# Worth keeping for v4 agent lifecycle unification
agent-lifecycle/LongRunningAgent

# claudeFlowAgent.js (116 LOC) — REAL Claude Agent SDK integration
# Uses @anthropic-ai/claude-agent-sdk query() with streaming
# 4 exported agent functions, withRetry() wrapper
# Canonical path for v4 agent integration
claudeFlowAgent.js
```

### AgentDB (~/node_modules/agentdb/src/)

```
# ContrastiveTrainer.ts (559 LOC, 87-90%)
# Real InfoNCE loss, analytical backprop via chain rule, AdamW with L2 decay
# CAVEAT: trainer.project() NEVER called (C34) — trained weights have zero effect in production
ContrastiveTrainer.ts

# FilterBuilder.ts (209 LOC, 92%)
# Injection-safe predicate DSL, 8 operator types
# Best file in RVF subsystem
FilterBuilder.ts

# SqlJsRvfBackend.ts (457 LOC, 88-92%)
# Full ACID via sql.js WASM fallback
# CAVEATS: O(n) brute-force search (C25), .rvf format incompatible with native RvfBackend (C26)
SqlJsRvfBackend.ts
```

### RuVector PostgreSQL (~/repos/ruvector/)

```
# PostgreSQL management suite (4,211 LOC combined, 85-95%)
# backup.js, optimize.js, benchmark.js, status.js
# Operational tooling for ruvector-postgres deployment
postgres/backup.js
postgres/optimize.js
postgres/benchmark.js
postgres/status.js
```

### Agentic-Flow / Sublinear-Time-Solver

```
# consciousness-explorer MCP server (594 LOC, 94.8%)
# PRODUCTION-QUALITY MCP exemplar: genuine @modelcontextprotocol/sdk
# All 12 handlers delegate to real explorer methods. Zero facades.
# STARK CONTRAST with consciousness-explorer.js itself (15% — theatrical)
# Use as TEMPLATE for v4 MCP server design
consciousness-explorer/mcp/server.js
```

### Swarm Coordination (~/repos/ruv-FANN/ruv-swarm/)

```
# ruv-swarm-persistence crate (~93% weighted)
# 3-backend trait architecture: SQLite (rusqlite), IndexedDB (rexie), in-memory
# lib.rs (250 LOC, 88-92%), memory.rs (95-98%), wasm.rs (95%), migrations.rs (92-95%)
# 28 async CRUD methods, QueryBuilder with SQL injection prevention
# CAVEAT: compilation blocked by ruv-fann ^0.1.5 vs 0.2.0 version pin (R141)
ruv-swarm-persistence/
```

### Memory & Learning (~/repos/ruv-FANN/ + agentic-flow)

```
# ReasoningBank Rust workspace (4 crates)
# core (88-92%), storage (94%), learning (95-98% — BEST learning code in project), mcp (93-95%)
# learning crate: AdaptiveLearner, StrategyOptimizer, async_learner_v2
# CAVEAT: reasoningbank-mcp fails compilation (C197, 6 errors — mismatched StorageConfig types)
# Other 4 crates pass cargo check and tests
reasoningbank-core/
reasoningbank-storage/
reasoningbank-learning/    # 95-98% — highest quality
reasoningbank-mcp/         # FAILS compilation — fix StorageConfig types first
```

### RuVector Advanced (~/repos/ruvector/crates/ruvector-edge-net/)

```
# edge-net/federated.rs (1,218 LOC, 95-98%)
# BEST federated learning in project
# Byzantine-robust (MAD+median), differential privacy (Gaussian ε,δ-DP)
# TopK compression with error feedback (arXiv:1712.01887)
# Reputation-weighted FedAvg, 13 tests
federated.rs
```

## Files to NEVER Copy

```
# Theatrical WASM (13 stubs)
ANY file named lib.rs in */wasm/ directories under 100 LOC

# Hash-based embeddings (16+ instances)
ANY file containing: hash_embedding, hashCode, fnv1a, djb2 used for vector generation

# lib_simple.rs — the facade that excludes genuine algorithms
sublinear-time-solver/src/lib_simple.rs

# SublinearSolver TS wrapper — routes through theatrical facade
sublinear-time-solver/src/index.ts (the barrel that re-exports SublinearSolver)

# Distributed transport stubs
ruvector-graph/src/distributed/rpc.rs        # 15-20%
ruvector-graph/src/distributed/coordinator.rs # 30-35%
ruvector-graph/src/distributed/gossip.rs     # transport portion only
ruvector-graph/src/distributed/federation.rs # execute_on_cluster stub

# Fabricated systems
ANY EmergenceSystem, consciousness theory, or "superluminal" code

# intelligence.ts facade — R140 CRITICAL
# Claims O(log n) HNSW, actual O(n) brute-force. 14+ consumers. LoRA/EWC config NEVER used.
claude-flow-cli dist/src/intelligence.ts (or intelligence.js)

# sona-tools.ts fake speedup — R138/R140
# Fabricates "1000x speedup" via estimatedBruteForce = searchLatency * 1000
ANY sona-tools.ts or SONA MCP tool handlers that compute estimatedBruteForce

# V3 memory layer — R135-R136
# agentdb-adapter.ts is a plain Map<string, MemoryEntry> — NOT AgentDB
# memory-bridge.ts (1,773 LOC genuine) but NOT compiled into npm dist
v3/@claude-flow/memory/ (entire published package — rebuild from scratch)

# worker-daemon facade stubs — R140
# 9/12 local worker types are stubs
worker-daemon.ts (keep HeadlessWorkerExecutor only)

# neural-network-implementation — R141 COMPILATION AUDIT
# Previously rated 75-85%, actually UNCOMPILABLE (106 cargo errors)
sublinear-time-solver/crates/neural-network-implementation/ (ENTIRE CRATE)

# ruvllm whole-crate — R141 COMPILATION AUDIT
# 120K LOC, largest crate, FAILS cargo check. Extract individual files only.
# DO copy: batch.rs, scheduler.rs, kv_cache_manager.rs (algorithmically genuine)
# DO NOT copy: the crate as a whole
ruvllm/ (as a whole crate — cherry-pick genuine files instead)

# ruv-swarm crates — R141 COMPILATION AUDIT
# All 14 blocked by ruv-fann ^0.1.5 vs 0.2.0 version pin
ruv-fann-rust/ruv-swarm/ (all 14 sub-crates until version pin fixed)

# CLI demo skeletons
ANY file with todo!("not yet implemented") as primary logic

# Upstream packages
@claude-flow/guidance (entire package)
agentic-flow npm (entire package)
```

## R141 Compilation Audit — Binary Truth Signal

The Rust compilation audit (R141) provides definitive pass/fail for every crate. Key results for extraction decisions:

### Confirmed Compilable (safe to COPY)
- ruvector-core ✓ (compiles, tests pass)
- ruvector-nervous-system ✓ (359 tests pass)
- temporal-tensor ✓ (213 tests pass)
- ruQu / ruqu-core ✓ (compiles, tests pass)
- backward_push ✓ (within sublinear-time-solver, compiles)
- bit-parallel-search ✓ (compiles, tests pass)
- micro_lora ✓ (within sona algorithms, compiles)
- RAC consensus ✓ (compiles)
- temporal-compare ✓ (compiles, tests pass)
- hyperbolic-hnsw ✓ (compiles)

### FAILS Compilation (do NOT copy as whole crate)
- ruvllm ✗ (120,345 LOC — largest crate, cannot compile)
- neural-network-implementation ✗ (17,294 LOC — 106 errors, dyn-incompatible traits)
- sona (Rust crate) ✗ (10,582 LOC — broken workspace integration)
- All 14 ruv-swarm crates ✗ (ruv-fann version pin mismatch)
- reasoningbank-mcp ✗ (6 errors — mismatched StorageConfig types)
- cognitum-gate-kernel ✗ (21 errors — private nalgebra field access)

### Partial Failures (compile but tests fail)
- agent-booster: 6/25 tests fail (strategy selection logic wrong)
- agentic-jujutsu: 83/88 pass, 5 fail (CRITICAL: ML-DSA signature verification doesn't reject invalid sigs)
