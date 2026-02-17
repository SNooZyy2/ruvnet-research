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

# CLI demo skeletons
ANY file with todo!("not yet implemented") as primary logic

# Upstream packages
@claude-flow/guidance (entire package)
agentic-flow npm (entire package)
```
