# R64 Execution Plan: Sublinear-Rust Architecture Completion + Temporal-Lead Core

**Date**: 2026-02-16
**Session ID**: 64
**Focus**: Complete the sublinear-time-solver src/ directory exploration started in R62 — graph adjacency, matrix CSR, core types, WASM integration, MCP temporal tool, ReasonGraph index, pattern detection engine, solver worker — plus temporal-lead-solver core and physics modules
**Parallel with**: R65 (no file overlap — R64 = sublinear-rust src/ + crates/temporal-lead-solver ONLY; R65 = custom-src agentdb-integration + agentic-flow-rust reasoningbank + psycho-symbolic MCP + ruv-swarm neural-network)

## IMPORTANT: Parallel Execution Notice

This plan runs IN PARALLEL with R65. The file lists are strictly non-overlapping:
- **R64 covers**: sublinear-rust `src/` architecture gaps (graph/adjacency, matrix/optimized_csr, core/, mcp/tools/temporal, reasongraph/index, neural-pattern-recognition/pattern-detection-engine, server/solver-worker) + `crates/temporal-lead-solver/` (core.rs, physics.rs)
- **R65 covers**: custom-src agentdb-integration adapters (4 TS files), agentic-flow-rust reasoningbank (benchmark.ts, HybridBackend.ts), psycho-symbolic MCP integration (schemas, wasm/loader, extractors/sentiment.rs), ruv-swarm neural-network.ts
- **ZERO shared files** between R64 and R65
- Do NOT read or analyze any file from R65's list (see R65-plan.md for that list)

## Rationale

- **graph/adjacency.rs completes R62's graph module**: R62 read graph/mod.rs (75-80%) which defines CSR and Graph trait but doesn't implement Graph. adjacency.rs (326 LOC) is the trait implementer and graph construction logic
- **matrix/optimized_csr.rs extends the matrix system count**: R60 found 5+ independent matrix systems. This Rust CSR optimization module could be the 7th+ system or could actually integrate with graph/mod.rs
- **src/core/wasm-integration.ts bridges WASM and JS**: R60 found WASM scoreboard 6 genuine vs 4 theatrical. This 383 LOC integration file reveals how the TS core loads and uses WASM — genuine or theatrical?
- **src/mcp/tools/temporal.ts is the MCP temporal tool**: R62 found MCP solver.ts 82-86% GENUINE. Does the temporal tool match that quality?
- **src/reasongraph/index.ts is the ReasonGraph barrel**: R62 found advanced-reasoning-engine.ts 0-5% THEATRICAL. Is the barrel file just re-exports or does it add integration logic?
- **crates/temporal-lead-solver/src/core.rs extends R58**: R58 found predictor.rs 92-95% (2nd genuine sublinear). core.rs (294 LOC) is the central module — does it match predictor quality?
- **crates/temporal-lead-solver/src/physics.rs adds the physics layer**: 266 LOC of physics for a lead-time solver. Genuine physics or theatrical?
- **pattern-detection-engine.js is the NPR core**: R19 found neural-pattern-recognition 80-90% facade. The detection engine (257 LOC) is the actual core — does it match the facade verdict?
- **src/core/types.ts defines the type system**: 189 LOC of core types reveals the solver's TypeScript API surface
- **server/solver-worker.js is the worker thread**: Connected to server/index.js (R44: 72% BIFURCATED). Does the worker actually call solvers?

## Target: 10 files, ~2,801 LOC

---

### Cluster A: Graph + Matrix Architecture (2 files, ~577 LOC)

Completes the graph module from R62 and examines another matrix system. R62 found graph/mod.rs defines CSR + WorkQueue but the Graph trait has NO implementor there.

| # | File ID | File | LOC | Domain | Package |
|---|---------|------|-----|--------|---------|
| 1 | 14367 | `src/graph/adjacency.rs` | 326 | memory-and-learning | sublinear-rust |
| 2 | 14375 | `src/matrix/optimized_csr.rs` | 251 | memory-and-learning | sublinear-rust |

**Full paths**:
1. `~/repos/sublinear-time-solver/src/graph/adjacency.rs`
2. `~/repos/sublinear-time-solver/src/matrix/optimized_csr.rs`

**Key questions**:
- `adjacency.rs`: Does it implement the Graph trait from mod.rs? Does it define AdjacencyList with edge list construction? Does it convert to/from CSR?
- `optimized_csr.rs`: Is this yet another independent CSR (7th+ matrix system) or does it extend graph/mod.rs CSR? Does it add SIMD-optimized SpMV? Does it use simd_ops.rs (R62: 82-86%)?

**Follow-up context**:
- R62: graph/mod.rs (75-80%) — CSR + WorkQueue + VisitedTracker. Graph trait defined but not implemented
- R34: ruvector-mincut BEST algorithmic, 2 incompatible matrix systems discovered
- R60: 5+ independent matrix systems confirmed
- R62: simd_ops.rs (82-86%) — genuine wide crate SIMD, orphaned from fully_optimized

---

### Cluster B: Core Infrastructure (3 files, ~920 LOC)

The core module types, WASM integration, and ReasonGraph barrel. These reveal how the TypeScript layer is organized.

| # | File ID | File | LOC | Domain | Package |
|---|---------|------|-----|--------|---------|
| 3 | 14355 | `src/core/wasm-integration.ts` | 383 | memory-and-learning | sublinear-rust |
| 4 | 14440 | `src/reasongraph/index.ts` | 301 | memory-and-learning | sublinear-rust |
| 5 | 14352 | `src/core/types.ts` | 189 | memory-and-learning | sublinear-rust |

**Full paths**:
3. `~/repos/sublinear-time-solver/src/core/wasm-integration.ts`
4. `~/repos/sublinear-time-solver/src/reasongraph/index.ts`
5. `~/repos/sublinear-time-solver/src/core/types.ts`

**Key questions**:
- `wasm-integration.ts`: Does it load the WASM solver genuinely (WebAssembly.instantiate) or theatrically? Does it bridge to wasm_iface.rs (R60: 90-93%)? Does it implement the fallback pattern seen in MCP solver.ts (R62: 82-86%)?
- `index.ts (ReasonGraph)`: Is this just re-exports or does it add orchestration? R62 found advanced-reasoning-engine.ts (0-5% theatrical) — does the barrel expose genuine modules alongside theatrical ones?
- `types.ts`: What types define the solver API? SolverConfig, SolverResult, MatrixFormat? Does it use discriminated unions or string enums?

**Follow-up context**:
- R60: WASM scoreboard 6 genuine vs 4 theatrical (60%). wasm_iface.rs (90-93%)
- R62: MCP solver.ts (82-86%) has 3-tier fallback cascade (WASM → Optimized → baseline)
- R62: advanced-reasoning-engine.ts (0-5%) COMPLETE THEATRICAL
- R61: ReasonGraph BIMODAL — infrastructure genuine, optimization theatrical

---

### Cluster C: MCP Tool + Neural Pattern Detection + Solver Worker (3 files, ~791 LOC)

The outer-layer tools: MCP temporal tool, the neural pattern detection engine core, and the server worker thread.

| # | File ID | File | LOC | Domain | Package |
|---|---------|------|-----|--------|---------|
| 6 | 14406 | `src/mcp/tools/temporal.ts` | 348 | memory-and-learning | sublinear-rust |
| 7 | 14429 | `src/neural-pattern-recognition/src/pattern-detection-engine.js` | 257 | memory-and-learning | sublinear-rust |
| 8 | 14298 | `server/solver-worker.js` | 186 | memory-and-learning | sublinear-rust |

**Full paths**:
6. `~/repos/sublinear-time-solver/src/mcp/tools/temporal.ts`
7. `~/repos/sublinear-time-solver/src/neural-pattern-recognition/src/pattern-detection-engine.js`
8. `~/repos/sublinear-time-solver/server/solver-worker.js`

**Key questions**:
- `temporal.ts`: R62 found solver.ts MCP (82-86%) GENUINE with 3-tier cascade. Does temporal.ts match quality? Does it expose temporal-tensor or temporal-nexus functionality via MCP?
- `pattern-detection-engine.js`: R19 found NPR subsystem 80-90% facade. R49 revised subsystem to ~72%. This is the actual detection engine — does it have genuine detection or Math.random()?
- `solver-worker.js`: R44 found server/index.js 72% BIFURCATED (infra 90%, solver 0%). Does the worker thread actually call solver code or is it a stub?

**Follow-up context**:
- R62: solver.ts MCP (82-86%) GENUINE, 3-tier fallback
- R19: neural-pattern-recognition 80-90% facade
- R49: NPR subsystem revised to ~72% (real-time-monitor, statistical-validator genuine)
- R44: server/index.js BIFURCATED — HTTP/WebSocket infra 90%, solver integration 0%

---

### Cluster D: Temporal-Lead-Solver Core (2 files, ~560 LOC)

The core and physics modules of the temporal-lead-solver crate, which produced the 2nd genuine sublinear algorithm (predictor.rs 92-95%).

| # | File ID | File | LOC | Domain | Package |
|---|---------|------|-----|--------|---------|
| 9 | 14171 | `crates/temporal-lead-solver/src/core.rs` | 294 | memory-and-learning | sublinear-rust |
| 10 | 14173 | `crates/temporal-lead-solver/src/physics.rs` | 266 | memory-and-learning | sublinear-rust |

**Full paths**:
9. `~/repos/sublinear-time-solver/crates/temporal-lead-solver/src/core.rs`
10. `~/repos/sublinear-time-solver/crates/temporal-lead-solver/src/physics.rs`

**Key questions**:
- `core.rs`: Does it define the TemporalLeadSolver struct? Does it integrate predictor.rs (R58: 92-95%)? Does it define the forward-backward push protocol? At 294 LOC, is this the actual solver coordinator or just types?
- `physics.rs`: What physics model underpins the temporal lead prediction? Is it genuine differential equations / dynamical systems or theatrical "physics" labels on basic math? Does it connect to temporal-tensor (R37: 93% PRODUCTION-READY)?

**Follow-up context**:
- R58: predictor.rs (92-95%) 2nd genuine sublinear O(√n). Forward-backward push, Kwok-Wei-Yang 2025
- R58: solver.rs (68-72%) BIMODAL — 5 algorithms but backward_push STUB
- R37: temporal-tensor 93% HIGHEST QUALITY CRATE — 213 tests, all files ≥88%

---

## Expected Outcomes

- **Graph module complete**: adjacency.rs reveals Graph trait implementation and graph construction
- **Matrix system count update**: optimized_csr.rs — 7th+ system or integration with graph/mod.rs CSR?
- **WASM integration verdict**: wasm-integration.ts — genuine loading or theatrical check?
- **ReasonGraph architecture**: index.ts reveals what ReasonGraph actually exports
- **Temporal-lead-solver quality**: core.rs + physics.rs — does the crate match predictor.rs quality (92-95%)?
- **MCP temporal quality**: Whether temporal.ts matches solver.ts (82-86%)
- **NPR engine verdict**: Whether pattern-detection-engine.js confirms or reverses the facade finding
- **Worker thread reality**: Whether solver-worker.js actually calls solvers or is a stub
- **Type system clarity**: core/types.ts reveals the TS API surface

## Stats Target

- ~10 file reads, ~2,801 LOC
- DEEP files: 1,066 → ~1,076
- Expected findings: 40-60

## Cross-Session Notes

- **ZERO overlap with R65**: R65 covers custom-src, agentic-flow-rust reasoningbank, psycho-symbolic MCP/extractors, ruv-swarm
- **Extends R62**: graph/adjacency.rs completes graph module; core/types.ts + wasm-integration.ts extend core/ exploration; MCP temporal.ts extends MCP tool layer; reasongraph/index.ts extends ReasonGraph
- **Extends R58**: temporal-lead-solver core.rs + physics.rs extend the crate that produced predictor.rs (92-95%)
- **Extends R19/R49**: pattern-detection-engine.js is the actual NPR engine core
- **Extends R44**: solver-worker.js extends server/ directory investigation
