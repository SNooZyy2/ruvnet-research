# R60 Execution Plan: WASM Pipeline Complete + Sublinear Core Algorithms

**Date**: 2026-02-16
**Session ID**: 60
**Focus**: Complete WASM architecture mapping (wasm-solver crate, wasm_iface, wasm.rs, math_wasm), core solver TS layer (optimized-solver, memory-manager, matrix), sublinear algorithm investigation (sublinear_neumann), temporal sparse operations, and convergence metrics
**Parallel with**: R61 (no file overlap -- R60 = WASM + core/ + crates/ + src/sublinear/ + src/convergence/; R61 = MCP tools + ReasonGraph + agentic-flow + ruv-fann-rust)

## IMPORTANT: Parallel Execution Notice

This plan runs IN PARALLEL with R61. The file lists are strictly non-overlapping:
- **R60 covers**: sublinear-rust WASM files (4 Rust files), sublinear-rust src/core/ (3 TS files), sublinear-rust src/sublinear/ (1 Rust file), sublinear-rust crates/temporal-compare (1 Rust file), sublinear-rust src/convergence/ (1 JS file)
- **R61 covers**: sublinear-rust src/mcp/tools/ (3 TS files), sublinear-rust src/reasongraph/ (2 TS files), agentic-flow-rust controllers+routing (2 TS files), agentic-flow-rust reasoningbank MCP (1 Rust file), ruv-fann-rust ruv-swarm (2 files)
- **ZERO shared files** between R60 and R61
- **R60 has NO src/mcp/, src/reasongraph/, agentic-flow-rust, or ruv-fann-rust files**
- **R61 has NO crates/wasm-solver, src/wasm*, src/core/, src/sublinear/, src/convergence/, or crates/temporal-compare files**
- Do NOT read or analyze any file from R61's list (see R61-plan.md for that list)

## Rationale

- **WASM is the single most contested question in the project**: 2 genuine WASM instances (R49 ReasoningBank, R58 text-extractor) vs 4 theatrical ones (R47, R43 x2, R51 quic-coordinator). The wasm-solver crate (lib.rs 426 LOC), plus 3 WASM interface files (wasm_iface.rs, wasm.rs, math_wasm.rs = 1,181 LOC combined), form the complete WASM pipeline. This is the definitive WASM verdict for sublinear-time-solver
- **src/core/ is the TS solver heart**: optimized-solver.ts (462 LOC), memory-manager.ts (437 LOC), and matrix.ts (404 LOC) are the consumer-facing TS core. R59 examined js/ entry points; now we examine the core TS layer. Does it connect to the WASM/Rust backend or reimplement everything?
- **sublinear_neumann.rs extends the sublinear algorithm arc**: R56 found backward_push.rs GENUINE O(1/epsilon), R58 found predictor.rs GENUINE O(sqrt(n)). sublinear_neumann.rs (420 LOC) is named after the Neumann series -- a legitimate mathematical technique for solving linear systems. Is this the 3rd genuine sublinear algorithm?
- **temporal-compare/sparse.rs extends the sparse matrix arc**: R28 found sparse.rs 95% BEST, R34 found 2 incompatible matrix systems. temporal-compare/sparse.rs is a different sparse implementation -- does it align with or diverge from existing ones?
- **convergence/metrics-reporter.js extends the metrics pattern**: R56 found metrics_collector.rs 40-45% (random data), R57 found performance.js 25-30% (Math.random). Does metrics-reporter.js continue the theatrical metrics pattern?

## Target: 10 files, ~4,140 LOC

---

### Cluster A: WASM Pipeline (4 files, ~1,607 LOC)

The complete WASM stack for sublinear-time-solver. Previously: 2 genuine vs 4 theatrical WASM instances. These 4 files form a coherent pipeline that should reveal the definitive architecture.

| # | File ID | File | LOC | Domain | Package |
|---|---------|------|-----|--------|---------|
| 1 | 14181 | `crates/wasm-solver/src/lib.rs` | 426 | memory-and-learning | sublinear-rust |
| 2 | 14506 | `src/wasm_iface.rs` | 397 | memory-and-learning | sublinear-rust |
| 3 | 14504 | `src/wasm.rs` | 394 | memory-and-learning | sublinear-rust |
| 4 | 14372 | `src/math_wasm.rs` | 390 | memory-and-learning | sublinear-rust |

**Full paths**:
1. `~/repos/sublinear-time-solver/crates/wasm-solver/src/lib.rs`
2. `~/repos/sublinear-time-solver/src/wasm_iface.rs`
3. `~/repos/sublinear-time-solver/src/wasm.rs`
4. `~/repos/sublinear-time-solver/src/math_wasm.rs`

**Key questions**:
- `lib.rs (wasm-solver)`: Is this the genuine WASM solver core?
  - Does it use `wasm-bindgen` or `wasm-pack` for real WASM compilation targets?
  - Does it expose actual solver functions (conjugate gradient, backward push) via `#[wasm_bindgen]`?
  - Does it import from the Rust solver crates (backward_push, optimized_solver)?
  - R43 found TWO independent WASM facades -- is this a third, or the real one?
  - At 426 LOC in Rust, is there enough substance for a real solver vs a wrapper?
- `wasm_iface.rs`: How does the WASM interface layer work?
  - Does it define the FFI boundary (extern "C" functions, memory allocation)?
  - Does it handle serialization/deserialization for WASM (e.g., serde-wasm-bindgen)?
  - Is it the bridge between wasm-solver crate and the JS consumers?
  - Does it expose a typed API or raw pointer manipulation?
- `wasm.rs`: What WASM bindings does this provide?
  - How does this relate to wasm_iface.rs? Duplicate, wrapper, or different API surface?
  - R57 found ruv-swarm WASM API MISMATCH (types don't align) -- does this file show alignment or mismatch?
  - Does it use `js_sys` / `web_sys` for browser integration?
  - Does it compile to actual .wasm output (check for `#[wasm_bindgen]` annotations)?
- `math_wasm.rs`: Does WASM math use real numerical operations?
  - Does it implement matrix operations, sparse algebra, or linear solvers in WASM?
  - Does it use SIMD (wasm32 SIMD intrinsics) for performance?
  - R56 found fully_optimized.rs 96-99% with INT8+AVX2 -- does WASM math approach that quality?
  - Does it operate on LinearAlgebra types or have its own data structures?

**Follow-up context**:
- R43: TWO independent WASM facades (solver.ts "loaded but unused" + wasm-sublinear-complete.ts "checked but never loaded")
- R47: 3rd theatrical WASM
- R49: ReasoningBank WASM 100% GENUINE
- R57: ruv-swarm WASM API MISMATCH = 0% WASM
- R58: text-extractor.ts REVERSES theatrical pattern (first genuine WASM->Rust integration)
- WASM scoreboard: 2 genuine vs 4 theatrical = 33% genuine rate

---

### Cluster B: Core TS Solver Layer (3 files, ~1,303 LOC)

The TypeScript core of the solver. R59 examined the js/ entry points; these are the src/core/ internals. Do they bridge to WASM or reimplement?

| # | File ID | File | LOC | Domain | Package |
|---|---------|------|-----|--------|---------|
| 5 | 14349 | `src/core/optimized-solver.ts` | 462 | memory-and-learning | sublinear-rust |
| 6 | 14347 | `src/core/memory-manager.ts` | 437 | memory-and-learning | sublinear-rust |
| 7 | 14346 | `src/core/matrix.ts` | 404 | memory-and-learning | sublinear-rust |

**Full paths**:
5. `~/repos/sublinear-time-solver/src/core/optimized-solver.ts`
6. `~/repos/sublinear-time-solver/src/core/memory-manager.ts`
7. `~/repos/sublinear-time-solver/src/core/matrix.ts`

**Key questions**:
- `optimized-solver.ts`: How does the TS solver relate to the Rust solver?
  - Does it import from the WASM module (from Cluster A) or reimplement in TS?
  - R56 found optimized_solver.rs 72-76% (standard CG+SIMD) -- does the TS version match or differ?
  - Does it claim "optimized" via genuine algorithmic work or just naming?
  - Does it implement the same algorithms (conjugate gradient, backward push) as the Rust backend?
  - Does it fall back to pure JS when WASM is unavailable (like R57's ruv-swarm pattern)?
- `memory-manager.ts`: What memory does this manage?
  - R58 found psycho-symbolic memory-manager.ts is 5th MISLABELED FILE (zero WASM memory, TS object registry)
  - Does THIS memory-manager.ts follow the same pattern or is it genuine?
  - Does it manage WASM linear memory (ArrayBuffer, memory.grow())?
  - Or does it manage solver state (cached computations, intermediate results)?
  - Does it implement proper lifecycle (alloc, free, resize)?
- `matrix.ts`: Is this the 4th matrix system?
  - R34 found 2 incompatible matrix systems, R53 found 3rd
  - Does this implement sparse matrices (CSR/CSC) or dense?
  - Does it use TypedArrays (Float64Array) for performance?
  - Does it connect to the Rust matrix operations via WASM or reimplement?
  - Does it align with R59's matrix-utils.js or is it independent?

**Follow-up context**:
- R56: optimized_solver.rs 72-76% (standard CG+SIMD, NOT a dispatcher)
- R58: memory-manager.ts (psycho-symbolic) = 5th MISLABELED FILE
- R34: 2 incompatible matrix systems, R53: 3rd system
- R59: matrix-utils.js and js/solver.js examined (results pending)
- R55: Rust 3-4x better than TS pattern

---

### Cluster C: Algorithm Core + Sparse + Metrics (3 files, ~1,230 LOC)

The algorithmic backbone: a potentially genuine sublinear algorithm, a sparse temporal implementation, and convergence metrics.

| # | File ID | File | LOC | Domain | Package |
|---|---------|------|-----|--------|---------|
| 8 | 14461 | `src/sublinear/sublinear_neumann.rs` | 420 | memory-and-learning | sublinear-rust |
| 9 | 14163 | `crates/temporal-compare/src/sparse.rs` | 397 | memory-and-learning | sublinear-rust |
| 10 | 14344 | `src/convergence/metrics-reporter.js` | 413 | memory-and-learning | sublinear-rust |

**Full paths**:
8. `~/repos/sublinear-time-solver/src/sublinear/sublinear_neumann.rs`
9. `~/repos/sublinear-time-solver/crates/temporal-compare/src/sparse.rs`
10. `~/repos/sublinear-time-solver/src/convergence/metrics-reporter.js`

**Key questions**:
- `sublinear_neumann.rs`: Is this the 3rd genuine sublinear algorithm?
  - The Neumann series (I - A)^{-1} = sum(A^k) is a legitimate mathematical technique for approximate matrix inversion
  - Does it implement truncated Neumann series (genuine O(k * nnz) for k terms)?
  - R56 found backward_push.rs GENUINE O(1/epsilon), R58 found predictor.rs O(sqrt(n)) -- does this match that quality?
  - R39 found FALSE sublinearity across most files -- which side does this fall on?
  - Does it use sparse matrix operations from the crate ecosystem?
  - Does it have proper convergence guarantees (spectral radius check)?
- `sparse.rs (temporal-compare)`: How does this sparse implementation differ?
  - R28 found sparse.rs 95% BEST in project
  - Does temporal-compare's sparse.rs implement CSR/CSC or a different format?
  - Does it handle temporal data (time-varying sparse matrices)?
  - Does it use SIMD/AVX intrinsics like the best Rust files?
  - Does it import from or connect to the main sparse.rs in sublinear-time-solver?
- `metrics-reporter.js`: Does convergence reporting use real data?
  - R56 found metrics_collector.rs 40-45% (real framework + random() data)
  - R57 found performance.js 25-30% (ALL metrics Math.random())
  - Does metrics-reporter.js follow the theatrical metrics pattern?
  - Does it report actual solver convergence (residual norms, iteration counts)?
  - Does it connect to the solver pipeline or operate independently?

**Follow-up context**:
- R56: backward_push.rs 92-95% GENUINE sublinear, fully_optimized.rs 96-99%
- R58: predictor.rs 92-95% 2ND GENUINE SUBLINEAR
- R39: FALSE sublinearity across multiple files (all O(n^2)+)
- R28: sparse.rs 95% BEST
- R56: metrics_collector.rs 40-45% (random data)
- R57: performance.js 25-30% (Math.random theater)

---

## Expected Outcomes

- **WASM verdict for sublinear-time-solver**: 4 WASM files examined in one session = definitive answer on whether the project's WASM is genuine or theatrical. Updates the 2-genuine-vs-4-theatrical scoreboard
- **Rust-to-TS bridge architecture**: How src/core/ connects to crates/wasm-solver/ -- is there a real bridge or are they parallel implementations?
- **3rd sublinear algorithm?**: Whether sublinear_neumann.rs implements genuine truncated Neumann series (extending the backward_push + predictor.rs arc)
- **Matrix system count update**: Whether src/core/matrix.ts is the 4th independent matrix system or connects to existing ones
- **Memory manager verdict**: Whether src/core/memory-manager.ts follows R58's "mislabeled" pattern or is genuine
- **Sparse implementation comparison**: How temporal-compare/sparse.rs compares to R28's sparse.rs (95%)
- **Metrics pattern confirmation**: Whether metrics-reporter.js extends the theatrical metrics pattern (random data)
- **WASM-to-JS completeness**: Whether the full pipeline (Rust solver -> WASM compilation -> JS interface -> TS core) is functional end-to-end

## Stats Target

- ~10 file reads, ~4,140 LOC
- DEEP files: 1,020 -> ~1,030
- Expected findings: 60-80 (10 files, heavy WASM + algorithm focus)

## Cross-Session Notes

- **ZERO overlap with R61**: R61 covers MCP tools + ReasonGraph + agentic-flow + ruv-fann-rust. No shared files or directories.
- **Extends R43**: Investigates the WASM solver crate that R43's facades ("loaded but unused") reference
- **Extends R56**: sublinear_neumann.rs extends backward_push.rs sublinear algorithm arc
- **Extends R58**: WASM files test whether text-extractor's "genuine WASM" pattern extends to the solver
- **Extends R34/R53**: matrix.ts extends the incompatible matrix systems investigation
- **Extends R57**: metrics-reporter.js extends the theatrical metrics pattern
- **NEW: wasm-solver crate** is completely unexplored -- the actual Rust WASM compilation target
- **NEW: wasm_iface.rs + wasm.rs + math_wasm.rs** form the complete WASM interface layer
- **NEW: sublinear_neumann.rs** = potential 3rd genuine sublinear algorithm
- **NEW: temporal-compare/sparse.rs** = new sparse implementation in unexplored crate
- Combined DEEP files from R60+R61: 1,020 -> ~1,040 (approximately +20)
