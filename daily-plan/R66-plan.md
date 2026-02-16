# R66 Execution Plan: ruv-swarm npm Runtime + Psycho-Symbolic Graph Reasoner + Temporal-Compare ML

**Date**: 2026-02-16
**Session ID**: 66
**Focus**: ruv-swarm npm JS/TS runtime layer (entry point, utils, types, logging, claude-integration), psycho-symbolic graph_reasoner Rust crate (graph representation, rule engine, crate root), temporal-compare ML ensemble and MLP
**Parallel with**: R67 (no file overlap -- R66 = ruv-fann-rust ruv-swarm/npm/src/ + sublinear-rust crates/; R67 = agentic-flow-rust reasoningbank/ + custom-src + sublinear-rust src/temporal_nexus/)

## IMPORTANT: Parallel Execution Notice

This plan runs IN PARALLEL with R67. The file lists are strictly non-overlapping:
- **R66 covers**: ruv-fann-rust ruv-swarm/npm/src/ (5 JS/TS files), sublinear-rust crates/psycho-symbolic-reasoner/graph_reasoner/src/ (3 Rust files), sublinear-rust crates/temporal-compare/src/ (2 Rust files)
- **R67 covers**: agentic-flow-rust reasoningbank/crates/ (6 Rust files), custom-src agentdb-integration/ (3 TS files), sublinear-rust src/temporal_nexus/quantum/ (2 Rust files)
- **ZERO shared files** between R66 and R67
- **R66 has NO agentic-flow-rust, custom-src, or src/temporal_nexus/ files**
- **R67 has NO ruv-fann-rust, crates/psycho-symbolic-reasoner/, or crates/temporal-compare/ files**
- Do NOT read or analyze any file from R67's list (see R67-plan.md for that list)

## Rationale

- **ruv-swarm npm is the JS consumer of the Rust swarm**: R50 found ruv-swarm Rust BIMODAL (memory.rs 95% vs spawn.rs 8%). The npm/src/ directory is the JS/TS binding layer that consumers actually import. Does the JS layer connect to genuine Rust code, or is it another facade? index.js (405 LOC) is the entry point, utils.ts (286 LOC) and types.ts (164 LOC) define the API surface, logging-config.js (179 LOC) configures observability, and claude-integration/core.js (112 LOC) wires up the Claude Code integration (R43 found claude-integration/ is setup toolkit, not API -- does ruv-swarm's version differ?)
- **graph_reasoner is the unexplored Rust graph logic**: R55 found psycho-symbolic Rust 3-4x better quality than TS. graph_reasoner/ is a complete subcrate within the psycho-symbolic-reasoner with graph.rs (373 LOC) defining the data structures, rules.rs (361 LOC) implementing the rule engine, and lib.rs (89 LOC) as the crate root. R65 found sentiment.rs (90-93%) for the NLP side -- is the GRAPH side equally genuine?
- **temporal-compare ML crate is completely untouched**: ensemble.rs (300 LOC) and mlp.rs (108 LOC) are ML model implementations. R37 found temporal-tensor HIGHEST QUALITY CRATE (93%). temporal-compare is a sibling crate in the same workspace -- does it match that quality? At 14+ files in the crate, ensemble.rs and mlp.rs are the core model files

## Target: 10 files, ~2,377 LOC

---

### Cluster A: ruv-swarm npm Runtime Layer (5 files, ~1,146 LOC)

The JS/TS consumer-facing layer of ruv-swarm. R50 found Rust-side memory.rs 95% genuine but spawn.rs 8% facade. This is the npm package that users `npm install` -- it must bridge to the Rust backend or reimplement functionality in JS.

| # | File ID | File | LOC | Domain | Package |
|---|---------|------|-----|--------|---------|
| 1 | 9631 | `ruv-swarm/npm/src/index.js` | 405 | swarm-coordination | ruv-fann-rust |
| 2 | 9672 | `ruv-swarm/npm/src/utils.ts` | 286 | swarm-coordination | ruv-fann-rust |
| 3 | 9634 | `ruv-swarm/npm/src/logging-config.js` | 179 | swarm-coordination | ruv-fann-rust |
| 4 | 9671 | `ruv-swarm/npm/src/types.ts` | 164 | swarm-coordination | ruv-fann-rust |
| 5 | 9611 | `ruv-swarm/npm/src/claude-integration/core.js` | 112 | swarm-coordination | ruv-fann-rust |

**Full paths**:
1. `~/repos/ruv-FANN/ruv-swarm/npm/src/index.js`
2. `~/repos/ruv-FANN/ruv-swarm/npm/src/utils.ts`
3. `~/repos/ruv-FANN/ruv-swarm/npm/src/logging-config.js`
4. `~/repos/ruv-FANN/ruv-swarm/npm/src/types.ts`
5. `~/repos/ruv-FANN/ruv-swarm/npm/src/claude-integration/core.js`

**Key questions**:
- `index.js`: How does the npm entry point wire up ruv-swarm?
  - Does it import native Rust bindings (NAPI, wasm-bindgen) or reimplement in JS?
  - R50 found memory.rs 95% but spawn.rs 8% -- does index.js call memory or spawn?
  - Does it export a coherent API (init, spawn, coordinate) or just re-export submodules?
  - R57 found ruv-swarm index.ts "WASM API MISMATCH = 0% WASM" -- is this the same file or a TS rewrite?
  - At 405 LOC, is there enough code for real orchestration vs just config + re-exports?
- `utils.ts`: What utility functions does ruv-swarm need?
  - Does it implement genuine swarm utilities (agent ID generation, message serialization, topology helpers)?
  - Or is it generic helpers (retry, sleep, format) that could be in any package?
  - Does it import from or interact with the Rust crate via FFI/WASM?
  - Are the TypeScript types aligned with the Rust SwarmConfig, AgentState types?
- `logging-config.js`: What logging infrastructure does ruv-swarm use?
  - Is this a genuine structured logging setup (winston, pino) with swarm-specific context?
  - Does it implement log aggregation across swarm agents?
  - Or is it a minimal console.log wrapper?
- `types.ts`: What type definitions does ruv-swarm expose?
  - Are these type-only definitions matching the Rust structs (SwarmConfig, AgentMessage, etc.)?
  - Does it define the full swarm API surface (spawn, terminate, coordinate, consensus)?
  - R50 found bimodal quality in Rust -- do the types cover only the genuine 95% parts?
  - Are there types that reference WASM or native bindings?
- `claude-integration/core.js`: How does ruv-swarm integrate with Claude Code?
  - R43 found the main claude-integration/ is a setup toolkit, NOT API -- does ruv-swarm's version differ?
  - Does it call `claude` CLI subprocess, use MCP, or define hook integration?
  - At 112 LOC, what functionality can it realistically implement?
  - Does it connect to the R51 MCP server (256 tools) or define its own integration path?

---

### Cluster B: Psycho-Symbolic Graph Reasoner (3 files, ~823 LOC)

The graph reasoning subcrate of the psycho-symbolic-reasoner. R55 found the Rust psycho-symbolic code is 3-4x better quality than the TS equivalent. R65 found sentiment.rs (90-93%) for NLP. This cluster examines the GRAPH reasoning side -- symbolic rule application over graph structures.

| # | File ID | File | LOC | Domain | Package |
|---|---------|------|-----|--------|---------|
| 6 | 14010 | `crates/psycho-symbolic-reasoner/graph_reasoner/src/graph.rs` | 373 | memory-and-learning | sublinear-rust |
| 7 | 14014 | `crates/psycho-symbolic-reasoner/graph_reasoner/src/rules.rs` | 361 | memory-and-learning | sublinear-rust |
| 8 | 14012 | `crates/psycho-symbolic-reasoner/graph_reasoner/src/lib.rs` | 89 | memory-and-learning | sublinear-rust |

**Full paths**:
6. `~/repos/sublinear-time-solver/crates/psycho-symbolic-reasoner/graph_reasoner/src/graph.rs`
7. `~/repos/sublinear-time-solver/crates/psycho-symbolic-reasoner/graph_reasoner/src/rules.rs`
8. `~/repos/sublinear-time-solver/crates/psycho-symbolic-reasoner/graph_reasoner/src/lib.rs`

**Key questions**:
- `graph.rs`: How is the reasoning graph represented?
  - Does it use a proper graph data structure (adjacency list, CSR, petgraph)?
  - R34/R59/R62 found 7+ independent matrix/graph systems -- is this another one?
  - Does it define nodes and edges with semantic types (concept, relation, inference)?
  - Does it support graph operations needed for reasoning (traversal, subgraph extraction, path finding)?
  - Is the graph mutable (for incremental reasoning) or immutable (for batch processing)?
  - How does it relate to ruvector-graph's Cypher executor (R38 confirmed working)?
- `rules.rs`: How does the symbolic rule engine work?
  - Does it implement forward-chaining, backward-chaining, or both?
  - Are rules defined as pattern → action with proper binding variables?
  - Does it support rule priorities, conflict resolution, or Rete-like optimization?
  - How does it interact with graph.rs -- does it query the graph or modify it?
  - R41 found consciousness code 79% genuine -- are these the symbolic reasoning rules that support it?
  - At 361 LOC, is there enough for a genuine rule engine vs a simple if-then chain?
- `lib.rs`: What does the crate root expose?
  - Does it re-export graph + rules, or does it add its own API surface?
  - Does it define a GraphReasoner struct that composes graph + rules?
  - Are there integration tests or example usage patterns?
  - Does it document the crate's purpose and relation to the psycho-symbolic parent crate?

---

### Cluster C: Temporal-Compare ML Models (2 files, ~408 LOC)

The ML model implementations in the temporal-compare crate. R37 found sibling crate temporal-tensor HIGHEST QUALITY (93%). The temporal-compare crate has 14+ files covering MLP, ensemble, attention, quantization, reservoir computing. ensemble.rs and mlp.rs are the core model architectures.

| # | File ID | File | LOC | Domain | Package |
|---|---------|------|-----|--------|---------|
| 9 | 14148 | `crates/temporal-compare/src/ensemble.rs` | 300 | memory-and-learning | sublinear-rust |
| 10 | 14153 | `crates/temporal-compare/src/mlp.rs` | 108 | memory-and-learning | sublinear-rust |

**Full paths**:
9. `~/repos/sublinear-time-solver/crates/temporal-compare/src/ensemble.rs`
10. `~/repos/sublinear-time-solver/crates/temporal-compare/src/mlp.rs`

**Key questions**:
- `ensemble.rs`: What ensemble learning methods are implemented?
  - Does it implement genuine ensemble methods (bagging, boosting, stacking, random forest)?
  - Does it compose the MLP models from mlp.rs into an ensemble?
  - R23 found neural-network-implementation BEST IN ECOSYSTEM (90-98%) -- does the Rust implementation match?
  - Does it handle temporal data (time series) specifically or generic tabular data?
  - Is there proper train/predict separation with cross-validation?
  - Does it use SIMD (AVX2/NEON) like the rest of the high-quality Rust code?
- `mlp.rs`: How is the MLP (multi-layer perceptron) implemented?
  - Does it implement proper forward pass with activation functions (ReLU, sigmoid, tanh)?
  - Does it include backpropagation for training or is it inference-only?
  - R40/R45 found JS neural = inference-only facades -- does the Rust version include training?
  - At 108 LOC, this is compact -- is it a clean minimal MLP or a stub?
  - Does it use the same weight representation as temporal-tensor or define its own?
  - Does it connect to the sibling mlp_avx512.rs, mlp_optimized.rs, mlp_quantized.rs variants?

---

## Expected Outcomes

1. **ruv-swarm npm architecture verdict**: Does the JS layer connect to genuine Rust code or reimplement/facade?
2. **graph_reasoner quality assessment**: Is the symbolic graph reasoning genuine (matching R55's 3-4x quality)?
3. **temporal-compare ML verdict**: Does this crate match temporal-tensor's 93% quality?
4. **Claude integration pattern**: Does ruv-swarm's claude-integration/ differ from the setup-only pattern in R43?
5. **Graph system count**: Is graph_reasoner yet another independent graph representation (8th+)?

## DB Update Template

```javascript
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const sessionId = 66;
const today = new Date().toISOString().slice(0, 10);

// File IDs for this session:
// 9631: ruv-swarm/npm/src/index.js
// 9672: ruv-swarm/npm/src/utils.ts
// 9634: ruv-swarm/npm/src/logging-config.js
// 9671: ruv-swarm/npm/src/types.ts
// 9611: ruv-swarm/npm/src/claude-integration/core.js
// 14010: crates/psycho-symbolic-reasoner/graph_reasoner/src/graph.rs
// 14014: crates/psycho-symbolic-reasoner/graph_reasoner/src/rules.rs
// 14012: crates/psycho-symbolic-reasoner/graph_reasoner/src/lib.rs
// 14148: crates/temporal-compare/src/ensemble.rs
// 14153: crates/temporal-compare/src/mlp.rs
```

## Domain Tags

- All ruv-swarm files → `swarm-coordination` domain (already tagged)
- All graph_reasoner + temporal-compare files → `memory-and-learning` domain (already tagged)
