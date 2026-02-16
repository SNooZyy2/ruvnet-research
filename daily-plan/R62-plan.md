# R62 Execution Plan: Sublinear-Rust Root Architecture + Core Algorithms

**Date**: 2026-02-16
**Session ID**: 62
**Focus**: The sublinear-time-solver library root architecture (lib.rs, error.rs), solver algorithms (forward_push, solver_core), math infrastructure (simd_ops, graph/mod.rs), JL dimensionality reduction, convergence detection, MCP solver tool, and ReasonGraph reasoning engine
**Parallel with**: R63 (no file overlap -- R62 = sublinear-rust src/ root files only; R63 = custom-src agentdb-integration + ruv-fann-rust ruv-swarm + sublinear-rust crates/ + agentic-flow-rust reasoningbank-mcp)

## IMPORTANT: Parallel Execution Notice

This plan runs IN PARALLEL with R63. The file lists are strictly non-overlapping:
- **R62 covers**: sublinear-rust `src/` root-level files only (lib.rs, error.rs, solver_core.rs, graph/mod.rs, simd_ops.rs, solver/forward_push.rs, sublinear/johnson_lindenstrauss.rs, convergence/convergence-detector.js, mcp/tools/solver.ts, reasongraph/advanced-reasoning-engine.ts)
- **R63 covers**: custom-src agentdb-integration (3 TS files), ruv-fann-rust ruv-swarm (3 files: 2 TS/JS + 1 Rust), sublinear-rust crates/ (3 files: psycho-symbolic MCP + extractors), agentic-flow-rust reasoningbank-mcp (1 Rust file)
- **ZERO shared files** between R62 and R63
- **R62 has NO agentdb-integration, ruv-swarm, crates/psycho-symbolic-reasoner, crates/temporal-compare, or reasoningbank files**
- **R63 has NO src/lib.rs, src/error.rs, src/solver/, src/solver_core.rs, src/graph/, src/simd_ops.rs, src/sublinear/, src/convergence/, src/mcp/, or src/reasongraph/ files**
- Do NOT read or analyze any file from R63's list (see R63-plan.md for that list)

## Rationale

- **lib.rs + error.rs = crate foundation**: These are the library root and error types for the entire sublinear-time-solver Rust crate. Understanding these reveals the public API surface, feature flags, and error taxonomy. R56 found fully_optimized.rs 96-99% and backward_push.rs 92-95% -- the crate root tells us how these excellent modules are organized and exported
- **forward_push.rs extends the sublinear algorithm arc**: R56 found backward_push.rs GENUINE O(1/epsilon). forward_push.rs (369 LOC) is the complementary algorithm -- are both directions implemented to the same quality? Together they should form a complete personalized PageRank solver
- **solver_core.rs is the algorithmic coordinator**: At 322 LOC, this is likely the central solver dispatch that connects backward_push, forward_push, and Neumann methods. Does it actually integrate the genuine algorithms?
- **graph/mod.rs defines the graph data structures**: At 289 LOC, this module root defines how the Rust crate represents graphs -- CSR adjacency, adjacency lists, or something novel. Critical for understanding all graph algorithms
- **simd_ops.rs extends the SIMD story**: R37 found ruvector-core has REAL SIMD (AVX-512/AVX2/NEON). simd_ops.rs (287 LOC) in sublinear-time-solver may show whether the solver also uses genuine SIMD or just claims to
- **johnson_lindenstrauss.rs = potential 3rd genuine sublinear algorithm**: JL transform is a mathematically rigorous technique for O(d*k) dimensionality reduction where k << n. This could be the 3rd genuine sublinear algorithm after backward_push O(1/epsilon) and predictor O(sqrt(n))
- **convergence-detector.js bridges Rust and JS**: At 316 LOC in JS (not Rust), this detects solver convergence. R60 found metrics-reporter.js 88-92% GENUINE -- does the convergence detector match?
- **MCP solver.ts is the external API**: R61 found MCP tools BIMODAL. solver.ts (360 LOC) exposes the solver to MCP consumers -- genuine integration or stub?
- **advanced-reasoning-engine.ts extends ReasonGraph**: R61 found ReasonGraph BIMODAL (infra genuine, optimization theatrical). The reasoning engine (297 LOC) is the core of ReasonGraph -- is it genuine inference or theatrical?

## Target: 10 files, ~3,309 LOC

---

### Cluster A: Crate Root + Error Types (2 files, ~781 LOC)

The foundation of the sublinear-time-solver Rust crate. lib.rs reveals the public API, module organization, and feature flags. error.rs defines the error taxonomy that all solver modules use.

| # | File ID | File | LOC | Domain | Package |
|---|---------|------|-----|--------|---------|
| 1 | 14370 | `src/lib.rs` | 377 | memory-and-learning | sublinear-rust |
| 2 | 14365 | `src/error.rs` | 404 | memory-and-learning | sublinear-rust |

**Full paths**:
1. `~/repos/sublinear-time-solver/src/lib.rs`
2. `~/repos/sublinear-time-solver/src/error.rs`

**Key questions**:
- `lib.rs`: What is the crate's public API?
  - What modules does it re-export? Does it expose backward_push, forward_push, Neumann, or just a unified solver API?
  - What feature flags are defined? (#[cfg(feature = "wasm")], #[cfg(feature = "simd")], etc.)
  - Does it import from sub-crates (wasm-solver, temporal-compare) or is it self-contained?
  - Is there a prelude module? What traits define the solver interface?
  - At 377 LOC, is there substantial logic or just module declarations + re-exports?
  - Does it expose graph construction + solver + sublinear algorithms as three separate APIs?
  - How does it relate to R56's fully_optimized.rs 96-99%? Is fully_optimized exported or internal?
- `error.rs`: How rich is the error taxonomy?
  - Does it use thiserror/anyhow or custom error types?
  - Does it define solver-specific errors (convergence failure, dimension mismatch, SIMD unavailable)?
  - At 404 LOC, this is a large error module -- does it indicate comprehensive error handling?
  - Does it support WASM errors (wasm-bindgen error conversion)?
  - Does it include error context for debugging (iteration count, residual, matrix dimensions)?

**Follow-up context**:
- R56: backward_push.rs 92-95%, fully_optimized.rs 96-99%
- R58: predictor.rs 92-95%
- R59: js/solver.js examined as entry point
- Crate root and error types have never been examined in any session

---

### Cluster B: Solver Algorithms (2 files, ~691 LOC)

The algorithmic core: forward_push complements backward_push (R56), and solver_core coordinates them. Does the solver integrate its genuine algorithms?

| # | File ID | File | LOC | Domain | Package |
|---|---------|------|-----|--------|---------|
| 3 | 14447 | `src/solver/forward_push.rs` | 369 | memory-and-learning | sublinear-rust |
| 4 | 14454 | `src/solver_core.rs` | 322 | memory-and-learning | sublinear-rust |

**Full paths**:
3. `~/repos/sublinear-time-solver/src/solver/forward_push.rs`
4. `~/repos/sublinear-time-solver/src/solver_core.rs`

**Key questions**:
- `forward_push.rs`: Is this the complementary algorithm to backward_push?
  - R56 found backward_push.rs GENUINE O(1/epsilon) sublinear. Forward push should be the "push from source" variant
  - Does it implement Andersen et al.'s forward local push with residual bounds?
  - Does it have the same quality markers as backward_push (proper convergence, alpha/epsilon params, sparse residual)?
  - At 369 LOC (vs backward_push 397 LOC), is there enough for a genuine implementation?
  - Does it use the same graph data structures as backward_push?
  - Is this the 3rd genuine sublinear algorithm? (if O(1/epsilon) or O(volume/epsilon))
- `solver_core.rs`: How does it coordinate the solver algorithms?
  - Does it dispatch to forward_push, backward_push, Neumann, and conjugate gradient?
  - Does it implement a solver selection heuristic (pick best algorithm based on matrix properties)?
  - R60 found sublinear_neumann.rs had CORRECT math but O(n^2) extraction negated sublinear claims -- does solver_core route around this?
  - Does it define the SolverConfig / SolverResult types?
  - Does it implement iteration limits, convergence criteria, and error handling?

**Follow-up context**:
- R56: backward_push.rs 92-95% GENUINE O(1/epsilon) sublinear
- R58: predictor.rs 92-95% 2nd genuine sublinear O(sqrt(n))
- R60: sublinear_neumann.rs correct math but O(n^2) extraction = FALSE sublinearity
- R56: optimized_solver.rs 72-76% (standard CG+SIMD, not a dispatcher)

---

### Cluster C: Math Infrastructure (3 files, ~864 LOC)

The mathematical foundation: graph data structures, SIMD operations, and Johnson-Lindenstrauss dimensionality reduction. These support all solver algorithms.

| # | File ID | File | LOC | Domain | Package |
|---|---------|------|-----|--------|---------|
| 5 | 14368 | `src/graph/mod.rs` | 289 | memory-and-learning | sublinear-rust |
| 6 | 14445 | `src/simd_ops.rs` | 287 | memory-and-learning | sublinear-rust |
| 7 | 14457 | `src/sublinear/johnson_lindenstrauss.rs` | 288 | memory-and-learning | sublinear-rust |

**Full paths**:
5. `~/repos/sublinear-time-solver/src/graph/mod.rs`
6. `~/repos/sublinear-time-solver/src/simd_ops.rs`
7. `~/repos/sublinear-time-solver/src/sublinear/johnson_lindenstrauss.rs`

**Key questions**:
- `graph/mod.rs`: How does the crate represent graphs?
  - Does it define CSR/CSC adjacency structures?
  - Does it support directed/undirected/weighted variants?
  - Does it implement graph construction from edge lists?
  - Does it export types used by forward_push, backward_push, and other algorithms?
  - At 289 LOC, is this a module root that re-exports sub-modules, or self-contained?
  - Does it align with ruvector-graph's graph types or define independent ones?
- `simd_ops.rs`: Does the solver have genuine SIMD?
  - R37 confirmed ruvector-core has REAL SIMD (AVX-512/AVX2/NEON). Does sublinear-time-solver match?
  - Does it use std::arch (x86_64, aarch64) or packed_simd/std::simd?
  - Does it implement vectorized dot product, matrix-vector multiply, or norm computation?
  - At 287 LOC, is there enough for real SIMD (vs just a feature flag wrapper)?
  - Does it have runtime CPU feature detection (#[cfg(target_feature = "avx2")])?
  - R56 found fully_optimized.rs uses INT8+AVX2 -- does simd_ops.rs provide those primitives?
- `johnson_lindenstrauss.rs`: Is this genuine JL dimensionality reduction?
  - JL lemma: for any epsilon, n points in R^d can be embedded into R^k where k = O(log(n)/epsilon^2)
  - Does it implement random projection via Gaussian or sparse Rademacher matrices?
  - Is the projection O(d*k) per point -- genuinely sublinear in n?
  - Does it provide JL guarantees (epsilon-distortion with high probability)?
  - This could be the 3rd genuine sublinear algorithm (after backward_push and predictor)
  - Does it integrate with the solver (used for dimensionality reduction of large graphs)?

**Follow-up context**:
- R37: ruvector-core REAL SIMD (AVX-512/AVX2/NEON)
- R56: fully_optimized.rs 96-99% with INT8+AVX2
- R34: ruvector-mincut BEST algorithmic
- R52: HNSW vendored 98-100%
- R39: FALSE sublinearity confirmed for most claims

---

### Cluster D: Convergence + MCP + Reasoning (3 files, ~973 LOC)

The outer layer: convergence detection (JS bridge), MCP solver tool (external API), and the reasoning engine (ReasonGraph core). Tests whether infrastructure matches the algorithmic quality.

| # | File ID | File | LOC | Domain | Package |
|---|---------|------|-----|--------|---------|
| 8 | 14343 | `src/convergence/convergence-detector.js` | 316 | memory-and-learning | sublinear-rust |
| 9 | 14403 | `src/mcp/tools/solver.ts` | 360 | memory-and-learning | sublinear-rust |
| 10 | 14439 | `src/reasongraph/advanced-reasoning-engine.ts` | 297 | memory-and-learning | sublinear-rust |

**Full paths**:
8. `~/repos/sublinear-time-solver/src/convergence/convergence-detector.js`
9. `~/repos/sublinear-time-solver/src/mcp/tools/solver.ts`
10. `~/repos/sublinear-time-solver/src/reasongraph/advanced-reasoning-engine.ts`

**Key questions**:
- `convergence-detector.js`: How does JS detect solver convergence?
  - R60 found metrics-reporter.js 88-92% GENUINE (zero Math.random). Does the detector match?
  - Does it implement real convergence criteria (residual norm < epsilon, relative change < threshold)?
  - Does it track iteration history and detect stalling?
  - Does it bridge to the Rust solver (reading convergence data from WASM) or run independently?
  - Does it expose events/callbacks for solver status updates?
- `solver.ts (MCP tool)`: How does the MCP tool expose solver functionality?
  - R61 found MCP tools BIMODAL (82-92% genuine vs 18-28% stub). Which side?
  - Does it define proper MCP tool schemas (inputSchema, outputSchema)?
  - Does it call the Rust solver via WASM or reimplement in TS?
  - Does it support multiple solver modes (forward_push, backward_push, CG, Neumann)?
  - Does it handle large inputs properly (streaming, pagination)?
  - Does it validate inputs (matrix dimensions, parameter ranges)?
- `advanced-reasoning-engine.ts`: What does the reasoning engine actually do?
  - R61 found ReasonGraph BIMODAL: caching/monitoring genuine (85-92%), optimization theater (0-15%)
  - Does the "advanced" reasoning engine implement genuine inference (rule chaining, graph traversal)?
  - Or is it theatrical (fabricated reasoning with hardcoded outputs)?
  - Does it import from the solver algorithms (using solver for graph reasoning)?
  - Does it connect to ReasoningBank (R57) or operate independently?
  - At 297 LOC, is there enough for real reasoning vs a stub?

**Follow-up context**:
- R60: metrics-reporter.js 88-92% genuine, zero Math.random()
- R61: MCP tools BIMODAL, ReasonGraph BIMODAL
- R53: performance-optimizer GENUINE (88-92%) in different subsystem
- R57: ReasoningBank = 4th disconnected data layer
- R43: ReasoningBank core APIs genuine, demo theater 35%

---

## Expected Outcomes

- **Crate architecture map**: First-ever examination of lib.rs reveals how the solver's best modules (backward_push 92-95%, fully_optimized 96-99%) are organized and exported
- **Forward push quality**: Whether forward_push.rs matches backward_push's quality (92-95%) and whether it's the 3rd genuine sublinear algorithm
- **Solver coordinator**: Whether solver_core.rs intelligently routes between algorithms or is a simple dispatcher
- **SIMD reality**: Whether sublinear-time-solver has genuine SIMD like ruvector-core or just claims
- **JL algorithm**: Whether johnson_lindenstrauss.rs implements genuine O(d*k) dimensionality reduction (potential 3rd sublinear algorithm)
- **Convergence quality**: Whether convergence-detector.js matches metrics-reporter.js quality (88-92%)
- **MCP solver quality**: Whether the solver MCP tool is genuine integration or stub
- **ReasonGraph core**: Whether the reasoning engine has substance or is theatrical
- **Architecture coherence**: Whether the crate root reveals a well-organized library or a collection of disconnected modules

## Stats Target

- ~10 file reads, ~3,309 LOC
- DEEP files: 1,049 -> ~1,059
- Expected findings: 50-70 (10 files, heavy algorithm + architecture focus)

## Cross-Session Notes

- **ZERO overlap with R63**: R63 covers custom-src, ruv-swarm, psycho-symbolic crates/, and reasoningbank-mcp. No shared files
- **Extends R56**: forward_push.rs complements backward_push.rs (same solver/ directory)
- **Extends R60**: convergence-detector.js in same directory as metrics-reporter.js
- **Extends R61**: MCP solver.ts and ReasonGraph reasoning engine extend MCP + ReasonGraph findings
- **Extends R37**: simd_ops.rs extends SIMD investigation from ruvector-core
- **NEW: lib.rs + error.rs** = first crate root examination for sublinear-time-solver
- **NEW: forward_push.rs** = complementary algorithm to backward_push
- **NEW: solver_core.rs** = solver coordination layer
- **NEW: graph/mod.rs** = graph data structure definitions
- **NEW: johnson_lindenstrauss.rs** = potential 3rd genuine sublinear algorithm
- Combined DEEP files from R62+R63: 1,049 -> ~1,069 (approximately +20)
