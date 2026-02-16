# R65 Execution Plan: OWN_CODE AgentDB Adapters + ReasoningBank Deep + Psycho-Symbolic MCP + Swarm Neural

**Date**: 2026-02-16
**Session ID**: 65
**Focus**: The ONLY remaining OWN_CODE files in the priority queue (4 AgentDB adapter/job files), agentic-flow ReasoningBank internals (HybridBackend, benchmark), psycho-symbolic MCP integration (schemas, WASM loader, sentiment extractor), and ruv-swarm neural-network runtime
**Parallel with**: R64 (no file overlap — R65 = custom-src + agentic-flow-rust + psycho-symbolic crates/ + ruv-swarm; R64 = sublinear-rust src/ + crates/temporal-lead-solver)

## IMPORTANT: Parallel Execution Notice

This plan runs IN PARALLEL with R64. The file lists are strictly non-overlapping:
- **R65 covers**: custom-src `agentdb-integration/` adapters (4 TS files), agentic-flow-rust `agentic-flow/src/reasoningbank/` (2 TS files), sublinear-rust `crates/psycho-symbolic-reasoner/` (3 files: schemas, WASM loader, sentiment), ruv-fann-rust `ruv-swarm/npm/src/` (1 TS file)
- **R64 covers**: sublinear-rust `src/` files (graph/adjacency, matrix/optimized_csr, core/, mcp/tools/temporal, reasongraph/index, neural-pattern-recognition, server/solver-worker) + `crates/temporal-lead-solver/` (core.rs, physics.rs)
- **ZERO shared files** between R64 and R65
- Do NOT read or analyze any file from R64's list (see R64-plan.md for that list)

## Rationale

- **OWN_CODE AgentDB adapters are the ONLY custom integration code remaining**: 4 files in `~/claude-flow-self-implemented/src/agentdb-integration/` are OWN_CODE (written on top of library code). R20 found ROOT CAUSE of broken AgentDB search (EmbeddingService never initialized), R48/R61 found 5 disconnected AgentDB layers. R63 read 3 OWN_CODE files (ruvector-backend-adapter, reflexion-service, reflexion-memory-adapter). These 4 remaining OWN_CODE files complete the AgentDB integration picture:
  - `vector-backend-adapter.ts` (230 LOC): The formal vector storage adapter — how does OWN_CODE connect to ruvector?
  - `vector-migration-job.ts` (203 LOC): Migration job for vector storage — reveals data model evolution
  - `embedding-adapter.ts` (170 LOC): Embedding adapter — does it fix the R20 root cause?
  - `real-embedding-adapter.ts` (153 LOC): "real" embedding adapter — the name suggests fixing the mock/hash fallback
- **ReasoningBank internals never examined in agentic-flow**: R57 found queries.ts (85-90%) PRODUCTION-READY. R50 found sqlite.rs (88-92%) genuine Rust storage. But HybridBackend.ts and benchmark.ts in the same ReasoningBank directory have never been read:
  - `HybridBackend.ts` (374 LOC): Hybrid storage backend — combines SQLite + in-memory? Could bridge disconnected layers
  - `benchmark.ts` (395 LOC): ReasoningBank benchmarks — genuine or fabricated like R43's standalone benchmarks?
- **Psycho-symbolic MCP integration extends R58/R63**: R58 found server.ts (72-76%) genuine MCP SDK. R63 read types/index.ts and graph-reasoner.ts wrapper. These 3 remaining files complete the MCP integration:
  - `schemas/index.ts` (327 LOC): MCP tool schemas — defines the full API surface
  - `src/wasm/loader.ts` (267 LOC): WASM loader for psycho-symbolic — genuine or theatrical (R60: 60% WASM genuine)?
  - `extractors/src/sentiment.rs` (288 LOC): Rust sentiment extraction — R55 found Rust 3-4x better than TS
- **ruv-swarm neural-network.ts is the swarm's neural layer**: R50 found ruv-swarm Rust BIMODAL (memory.rs 95%, spawn.rs 8%). R63 read agent.ts. neural-network.ts (296 LOC) is the neural subsystem — genuine ML or facade?

## Target: 10 files, ~2,703 LOC

---

### Cluster A: OWN_CODE AgentDB Adapters (4 files, ~756 LOC)

The ONLY remaining OWN_CODE files in the priority queue. These complete the AgentDB integration picture started in R63 (which read ruvector-backend-adapter, reflexion-service, reflexion-memory-adapter).

| # | File ID | File | LOC | Domain | Package |
|---|---------|------|-----|--------|---------|
| 1 | 2295 | `agentdb-integration/infrastructure/adapters/vector-backend-adapter.ts` | 230 | memory-and-learning | custom-src |
| 2 | 2299 | `agentdb-integration/infrastructure/jobs/vector-migration-job.ts` | 203 | memory-and-learning | custom-src |
| 3 | 2289 | `agentdb-integration/infrastructure/adapters/embedding-adapter.ts` | 170 | memory-and-learning | custom-src |
| 4 | 2292 | `agentdb-integration/infrastructure/adapters/real-embedding-adapter.ts` | 153 | memory-and-learning | custom-src |

**Full paths**:
1. `~/claude-flow-self-implemented/src/agentdb-integration/infrastructure/adapters/vector-backend-adapter.ts`
2. `~/claude-flow-self-implemented/src/agentdb-integration/infrastructure/jobs/vector-migration-job.ts`
3. `~/claude-flow-self-implemented/src/agentdb-integration/infrastructure/adapters/embedding-adapter.ts`
4. `~/claude-flow-self-implemented/src/agentdb-integration/infrastructure/adapters/real-embedding-adapter.ts`

**Key questions**:
- `vector-backend-adapter.ts`: How does OWN_CODE adapt ruvector as vector storage? Does it implement the vector-backend-adapter.interface.ts? Does it use RuVectorBackend.ts (R8: 90%)?
- `vector-migration-job.ts`: What does migration look like? Schema evolution? Data reindexing? Does it handle HNSW index rebuilds?
- `embedding-adapter.ts`: Does this adapter default to the hash-based mock (R20 ROOT CAUSE)? Or does it enforce real embeddings?
- `real-embedding-adapter.ts`: The name "real" suggests this was created to FIX the mock embedding problem. Does it use OpenAI/ONNX embeddings? Is this the R20 ROOT CAUSE FIX?

**Follow-up context**:
- R20: ROOT CAUSE — EmbeddingService never initialized → hash-based → broken search
- R63: ruvector-backend-adapter.ts read (separate file from vector-backend-adapter.ts)
- R8: RuVectorBackend.ts (90%) production-ready
- R22b: EmbeddingService.ts (80%) — unified ONNX, hash fallback
- R55: ruvector-integration.test.ts (95-98%) AUTHORITATIVE spec — tests expect real embeddings

---

### Cluster B: ReasoningBank Internals (2 files, ~769 LOC)

The HybridBackend and benchmark for ReasoningBank in agentic-flow. R57 found queries.ts (85-90%) PRODUCTION-READY with 7-table schema. These 2 files complete the ReasoningBank TS picture.

| # | File ID | File | LOC | Domain | Package |
|---|---------|------|-----|--------|---------|
| 5 | 10805 | `agentic-flow/src/reasoningbank/benchmark.ts` | 395 | memory-and-learning | agentic-flow-rust |
| 6 | 10801 | `agentic-flow/src/reasoningbank/HybridBackend.ts` | 374 | memory-and-learning | agentic-flow-rust |

**Full paths**:
5. `~/repos/agentic-flow/src/reasoningbank/benchmark.ts`
6. `~/repos/agentic-flow/src/reasoningbank/HybridBackend.ts`

**Key questions**:
- `benchmark.ts`: Is this genuine benchmarking (like R59's criterion suite 88-95%) or theatrical (like R43's standalone_benchmark 8-12%)? Does it use performance.now() with real solver calls? Does it benchmark the actual ReasoningBank operations (store/retrieve/search)?
- `HybridBackend.ts`: What backends does "hybrid" combine? SQLite + in-memory? Does it bridge the 4th disconnected data layer (R57)? Does it implement the storage interface from queries.ts?

**Follow-up context**:
- R57: queries.ts (85-90%) PRODUCTION-READY 7-table MaTTS schema. 4TH DISCONNECTED DATA LAYER
- R43: demo-comparison.ts (35%) SCRIPTED DEMO. ReasoningBank core APIs genuine
- R50: sqlite.rs (88-92%) GENUINE rusqlite Rust storage
- R59: Benchmark deception boundary — criterion 88-95% vs standalone 8-25%

---

### Cluster C: Psycho-Symbolic MCP Integration (3 files, ~882 LOC)

The MCP integration layer for the psycho-symbolic reasoner. R58 found server.ts (72-76%) genuine MCP SDK. R63 read types/index.ts and graph-reasoner.ts. These 3 files complete the picture.

| # | File ID | File | LOC | Domain | Package |
|---|---------|------|-----|--------|---------|
| 7 | 14023 | `crates/psycho-symbolic-reasoner/mcp-integration/schemas/index.ts` | 327 | memory-and-learning | sublinear-rust |
| 8 | 14026 | `crates/psycho-symbolic-reasoner/mcp-integration/src/wasm/loader.ts` | 267 | memory-and-learning | sublinear-rust |
| 9 | 14005 | `crates/psycho-symbolic-reasoner/extractors/src/sentiment.rs` | 288 | memory-and-learning | sublinear-rust |

**Full paths**:
7. `~/repos/sublinear-time-solver/crates/psycho-symbolic-reasoner/mcp-integration/schemas/index.ts`
8. `~/repos/sublinear-time-solver/crates/psycho-symbolic-reasoner/mcp-integration/src/wasm/loader.ts`
9. `~/repos/sublinear-time-solver/crates/psycho-symbolic-reasoner/extractors/src/sentiment.rs`

**Key questions**:
- `schemas/index.ts`: What MCP tool schemas are defined? How many tools? Does it match the 5 tools from server.ts (R58)? Are schemas complete with inputSchema/outputSchema?
- `src/wasm/loader.ts`: Does this WASM loader genuinely load the Rust psycho-symbolic WASM? R58 found text-extractor.ts (88-92%) REVERSED theatrical WASM. Does the loader match? Or is it the 5th theatrical WASM?
- `sentiment.rs`: R55 found Rust 3-4x better than TS for psycho-symbolic. R58 found text-extractor.ts GENUINE WASM calling Rust NLP. Does sentiment.rs have genuine lexicon-based sentiment analysis? R63 read preferences.rs in the same extractors directory

**Follow-up context**:
- R58: text-extractor.ts (88-92%) REVERSES theatrical WASM. server.ts (72-76%) genuine MCP SDK, 5 tools
- R55: psycho-symbolic Rust 3-4x better than TS
- R63: preferences.rs read, types/index.ts read, graph-reasoner.ts wrapper read
- R60: WASM scoreboard 6 genuine vs 4 theatrical (60%)

---

### Cluster D: Swarm Neural Layer (1 file, ~296 LOC)

The neural-network module for ruv-swarm's TypeScript runtime. R50 found ruv-swarm Rust BIMODAL (memory.rs 95%, spawn.rs 8%). R63 read agent.ts and cli-diagnostics.js.

| # | File ID | File | LOC | Domain | Package |
|---|---------|------|-----|--------|---------|
| 10 | 9660 | `ruv-swarm/npm/src/neural-network.ts` | 296 | swarm-coordination | ruv-fann-rust |

**Full path**:
10. `~/repos/ruv-FANN/ruv-swarm/npm/src/neural-network.ts`

**Key questions**:
- Does it implement genuine neural network operations (forward pass, backpropagation, weight updates)?
- Or is it a facade like R40's JS neural models (inference works, training facade)?
- Does it connect to ruv-fann-rust's Rust neural network implementation or operate independently?
- Does it match R45's neural.js (28% facade) pattern or reverse it?
- At 296 LOC, is there enough for real ML (vs a wrapper/stub)?

**Follow-up context**:
- R50: ruv-swarm Rust BIMODAL (memory.rs 95%, spawn.rs 8%)
- R40: JS neural models — inference works, training facade
- R45: neural.js (28%) facade — Infra-vs-intelligence split confirmed
- R63: agent.ts read (ruv-swarm TS runtime)

---

## Expected Outcomes

- **R20 ROOT CAUSE resolution**: Does real-embedding-adapter.ts FIX the hash-based mock default?
- **OWN_CODE quality**: First assessment of 4 custom AgentDB integration files — production or prototype?
- **ReasoningBank hybrid storage**: Whether HybridBackend.ts bridges disconnected data layers
- **Benchmark quality**: Whether ReasoningBank benchmark.ts is genuine or theatrical
- **Psycho-symbolic WASM verdict**: Whether wasm/loader.ts is genuine (like text-extractor) or theatrical
- **Sentiment Rust quality**: Whether sentiment.rs matches R55's "Rust 3-4x better" finding
- **Swarm neural quality**: Whether neural-network.ts has real ML or is another JS facade

## Stats Target

- ~10 file reads, ~2,703 LOC
- DEEP files: 1,066 → ~1,076
- Expected findings: 40-60

## Cross-Session Notes

- **ZERO overlap with R64**: R64 covers sublinear-rust src/ + crates/temporal-lead-solver only
- **Extends R63**: R63 read 3 OWN_CODE files + 3 ruv-swarm files + 4 psycho-symbolic/reasoningbank files. R65 reads the REMAINING OWN_CODE and cross-package files
- **Extends R20**: embedding-adapter.ts and real-embedding-adapter.ts directly address ROOT CAUSE
- **Extends R57**: HybridBackend.ts and benchmark.ts complete ReasoningBank TS picture
- **Extends R58**: schemas/index.ts, wasm/loader.ts, sentiment.rs complete psycho-symbolic MCP
- **Extends R50**: neural-network.ts extends ruv-swarm investigation
- **Combined DEEP files from R64+R65**: 1,066 → ~1,086 (approximately +20)
