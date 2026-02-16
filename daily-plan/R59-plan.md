# R59 Execution Plan: Benchmark Archaeology + JS Solver Layer

**Date**: 2026-02-16
**Session ID**: 59
**Focus**: Systemic benchmark deception investigation (Rust benches/ + TS benchmarks + neural benchmarks + ruv-fann-rust benchmarking), JavaScript solver entry points (solver.js, fast-solver.js, bmssp-solver.js), and matrix system analysis
**Parallel with**: R58 (no file overlap -- R59 = benchmarks + JS solvers + matrix; R58 = psycho-symbolic MCP + temporal solvers + goalie reasoning)

## IMPORTANT: Parallel Execution Notice

This plan runs IN PARALLEL with R58. The file lists are strictly non-overlapping:
- **R59 covers**: sublinear-rust benches/ (2 Rust files), neural-network-implementation benchmarks (1 Rust file), src/benchmarks/ (1 TS file), js/ solver layer (3 JS files), src/utils/ (1 JS file), ruv-fann-rust benchmarking (2 Rust files)
- **R58 covers**: psycho-symbolic-reasoner MCP integration (3 TS files), psycho-symbolic extractors (1 Rust file), temporal-lead-solver (2 Rust files), goalie advanced reasoning (3 TS files)
- **ZERO shared files** between R58 and R59
- **R59 has NO psycho-symbolic-reasoner, temporal-lead-solver, or goalie files**
- **R58 has NO benches/, js/, src/benchmarks/, src/utils/, or ruv-fann-rust files**
- Do NOT read or analyze any file from R58's list (see R58-plan.md for that list)

## Rationale

- **Benchmark deception is a confirmed systemic pattern**: R43 found rustc_optimization_benchmarks.rs "MOST DECEPTIVE" (15%, asymptotic mismatch), R56 found standalone_benchmark.rs "MOST DECEPTIVE" (8-12%, spin-loop to hardcoded latencies), R57 found benchmarking.rs (20-25%, simulate_execution = sleep). But NOT ALL benchmarks are fake -- R56 found latency_benchmark.rs at 72-78% with genuine criterion. This session systematically examines 4 benchmark files to map the deception boundary
- **JS solver layer is completely unexplored**: solver.js (397 LOC), fast-solver.js (416 LOC), and bmssp-solver.js (385 LOC) are the JavaScript entry points for the sublinear-time-solver engine. R56 examined the Rust solvers but the JS consumer-facing layer is unknown -- do these call Rust via WASM/NAPI or reimplement in JS?
- **matrix-utils.js (529 LOC)** is the largest untouched file in priority queue. R34 discovered 2 incompatible matrix systems, R53 found a 3rd -- matrix-utils.js may reveal a 4th or clarify the relationships
- **ruv-fann-rust benchmarking/** is a separate benchmarking subsystem in ruv-swarm: claude_executor.rs (387 LOC) and metrics.rs (383 LOC). These provide a cross-package benchmark comparison -- is ruv-swarm's benchmarking genuine where sublinear-rust's is theatrical?

## Target: 10 files, ~4,293 LOC

---

### Cluster A: Benchmark Investigation (4 files, ~1,796 LOC)

Systematic examination of 4 benchmark files across 2 packages to map the deception boundary. Previous sessions found benchmark quality ranges from 8% (standalone_benchmark.rs) to 78% (latency_benchmark.rs) -- where do these files fall?

| # | File ID | File | LOC | Domain | Package |
|---|---------|------|-----|--------|---------|
| 1 | 13846 | `benches/performance_benchmarks.rs` | 456 | memory-and-learning | sublinear-rust |
| 2 | 13847 | `benches/solver_benchmarks.rs` | 423 | memory-and-learning | sublinear-rust |
| 3 | 14301 | `src/benchmarks/performance-benchmark.ts` | 484 | memory-and-learning | sublinear-rust |
| 4 | 13881 | `crates/neural-network-implementation/benches/throughput_benchmark.rs` | 433 | memory-and-learning | sublinear-rust |

**Full paths**:
1. `~/repos/sublinear-time-solver/benches/performance_benchmarks.rs`
2. `~/repos/sublinear-time-solver/benches/solver_benchmarks.rs`
3. `~/repos/sublinear-time-solver/src/benchmarks/performance-benchmark.ts`
4. `~/repos/sublinear-time-solver/crates/neural-network-implementation/benches/throughput_benchmark.rs`

**Key questions**:
- `performance_benchmarks.rs`: Does this use genuine criterion benchmarks?
  - R56 found latency_benchmark.rs 72-78% with genuine criterion + theatrical thread::sleep -- does this follow the same pattern?
  - Does it import actual solver crates and benchmark real operations?
  - Does it use `#[bench]` or criterion groups?
  - Does it measure real matrix operations or spin-loop to hardcoded latencies (like standalone_benchmark.rs)?
- `solver_benchmarks.rs`: What solver operations does it benchmark?
  - R56 confirmed backward_push.rs 92-95% GENUINE sublinear -- does this benchmark actually call it?
  - Does it set up real problem instances (sparse matrices, graph structures)?
  - Does it compare different solver strategies or benchmark a single one?
  - Are timing results genuine (criterion) or theatrical (thread::sleep, hardcoded)?
- `performance-benchmark.ts`: How does the TS benchmark compare to Rust?
  - R43 found ruvector-benchmark.ts 92% REAL -- does this follow the genuine TS benchmark pattern?
  - Or does it follow R53's scheduler.ts 18-22% THEATRICAL pattern?
  - Does it benchmark WASM solver calls or pure TS operations?
  - Does it use performance.now() / process.hrtime() or fake timing?
- `throughput_benchmark.rs`: Is the neural network benchmark genuine?
  - R23 found neural-network-implementation BEST IN ECOSYSTEM (90-98%) -- do the benchmarks match quality?
  - Does it measure actual inference throughput (ops/sec) or fabricate metrics?
  - Does it use criterion with real model forward passes?
  - R56 found fully_optimized.rs 96-99% ANTI-FACADE -- is the benchmark consistent with that quality?

**Follow-up context**:
- R43: rustc_optimization_benchmarks.rs MOST DECEPTIVE (15%) -- asymptotic mismatch
- R56: standalone_benchmark.rs 8-12% MOST DECEPTIVE (spin-loop), latency_benchmark.rs 72-78% (genuine criterion)
- R57: benchmarking.rs 20-25% (simulate_execution = sleep(10ms))
- R23: neural-network-implementation 90-98% BEST IN ECOSYSTEM
- R43: ruvector-benchmark.ts 92% REAL -- genuine TS benchmarks DO exist

---

### Cluster B: JS Solver Layer (3 files, ~1,198 LOC)

The JavaScript entry points for the sublinear-time-solver engine. R56 examined the Rust solver core (backward_push.rs 92-95% genuine, optimized_solver.rs 72-76%). These JS files are the consumer-facing layer -- do they call Rust or reimplement?

| # | File ID | File | LOC | Domain | Package |
|---|---------|------|-----|--------|---------|
| 5 | 14186 | `js/fast-solver.js` | 416 | memory-and-learning | sublinear-rust |
| 6 | 14188 | `js/solver.js` | 397 | memory-and-learning | sublinear-rust |
| 7 | 14185 | `js/bmssp-solver.js` | 385 | memory-and-learning | sublinear-rust |

**Full paths**:
5. `~/repos/sublinear-time-solver/js/fast-solver.js`
6. `~/repos/sublinear-time-solver/js/solver.js`
7. `~/repos/sublinear-time-solver/js/bmssp-solver.js`

**Key questions**:
- `fast-solver.js`: Is this a fast JS implementation or WASM wrapper?
  - Does it import/require a WASM binary (the Rust solver compiled to WASM)?
  - Or does it reimplement solver algorithms in JavaScript (likely slower, possibly facade)?
  - "fast" prefix -- does it use typed arrays, SIMD.js (deprecated), or Web Workers for parallelism?
  - Does it connect to the WASM solver crate (`crates/wasm-solver/`) or bypass it?
- `solver.js`: Is this the main JS solver entry point?
  - Is this a WASM bridge (load .wasm, call exports) or pure JS implementation?
  - Does it implement the same algorithms as the Rust backend (backward push, conjugate gradient)?
  - R57 found ruv-swarm index.ts has "WASM API MISMATCH = 0% WASM" -- does solver.js have the same pattern?
  - Does it export a clean API that consumers can use?
- `bmssp-solver.js`: What is BMSSP?
  - BMSSP likely stands for "Balanced Min-Sum Shortest Path" or similar graph algorithm
  - Does it implement a genuine graph algorithm or is it a facade?
  - Does it use the project's sparse matrix infrastructure?
  - At 385 LOC, there's enough room for a real algorithm implementation
  - Is BMSSP referenced anywhere else in the codebase?

**Follow-up context**:
- R56: backward_push.rs 92-95% GENUINE sublinear, optimized_solver.rs 72-76% standard CG+SIMD
- R57: ruv-swarm index.ts WASM API MISMATCH = 0% WASM (pure-JS fallback always executes)
- R53: strange-loop NON-FUNCTIONAL -- JS runtime quality varies widely
- R40: JS neural models inference works, training facade -- JS implementations CAN be genuine for inference

---

### Cluster C: Matrix System + ruv-fann Benchmarking (3 files, ~1,299 LOC)

matrix-utils.js is the largest untouched priority file and may reveal another matrix system. The ruv-fann-rust benchmarking files provide cross-package comparison -- is ruv-swarm's benchmarking genuine where sublinear-rust's varies?

| # | File ID | File | LOC | Domain | Package |
|---|---------|------|-----|--------|---------|
| 8 | 14502 | `src/utils/matrix-utils.js` | 529 | memory-and-learning | sublinear-rust |
| 9 | 8869 | `ruv-swarm/benchmarking/src/claude_executor.rs` | 387 | swarm-coordination | ruv-fann-rust |
| 10 | 8872 | `ruv-swarm/benchmarking/src/metrics.rs` | 383 | swarm-coordination | ruv-fann-rust |

**Full paths**:
8. `~/repos/sublinear-time-solver/src/utils/matrix-utils.js`
9. `~/repos/ruv-FANN/ruv-swarm/benchmarking/src/claude_executor.rs`
10. `~/repos/ruv-FANN/ruv-swarm/benchmarking/src/metrics.rs`

**Key questions**:
- `matrix-utils.js`: Is this a 4th matrix system?
  - R34 discovered 2 incompatible matrix systems, R53 found a 3rd -- does this file introduce yet another?
  - Does it implement basic matrix operations (multiply, transpose, invert) or sparse matrix formats (CSR, CSC)?
  - Does it use typed arrays (Float64Array) for performance or plain JS arrays?
  - At 529 LOC, this is substantial -- does it implement genuine linear algebra or wrap a library?
  - Does it connect to any of the existing matrix systems or is it standalone?
- `claude_executor.rs`: Does ruv-swarm benchmark Claude execution?
  - Does it actually execute Claude API calls for benchmarking?
  - Or does it simulate Claude execution with mock responses?
  - R43 found claude-integration/ is setup toolkit NOT API -- does this follow the same pattern?
  - Does it measure real latency, token usage, and throughput?
  - This is in ruv-fann-rust, not sublinear-rust -- different quality patterns may apply
- `metrics.rs`: What swarm metrics does this collect?
  - R56 found metrics_collector.rs 40-45% (real framework + random() data) -- does ruv-fann's metrics.rs follow the same pattern?
  - Does it define proper metric types (counters, gauges, histograms)?
  - Does it use real timing infrastructure (Instant::now()) or fabricated values?
  - R55 found performance_monitor.rs 88-92% GENUINE -- Rust metrics CAN be genuine

**Follow-up context**:
- R34: 2 incompatible matrix systems discovered
- R53: 3rd matrix system found
- R43: claude-integration/ is setup toolkit NOT API (69%)
- R56: metrics_collector.rs 40-45% (random() data), but fully_optimized.rs 96-99% ANTI-FACADE
- R55: performance_monitor.rs 88-92% GENUINE -- Rust quality varies widely within same codebase

---

## Expected Outcomes

- **Benchmark deception map**: Clear classification of 4 benchmark files on the genuine (72-99%) vs theatrical (8-25%) spectrum, establishing which benchmark categories are real
- **JS solver architecture**: Whether solver.js/fast-solver.js/bmssp-solver.js call Rust via WASM or reimplement in JS (extending R57's WASM API MISMATCH finding)
- **BMSSP discovery**: What the BMSSP algorithm is and whether it's genuinely implemented
- **Matrix system count**: Whether matrix-utils.js is a 4th independent matrix system or connects to existing ones
- **Cross-package benchmark comparison**: Whether ruv-fann-rust's benchmarking (claude_executor.rs, metrics.rs) follows sublinear-rust's theatrical pattern or is more genuine
- **Neural benchmark vs crate quality**: Whether neural-network-implementation's benchmarks match R23's "BEST IN ECOSYSTEM" assessment of the crate itself
- **Benchmark quality predictors**: Can we identify patterns (criterion usage, real imports, problem instance setup) that predict genuine vs theatrical benchmarks?

## Stats Target

- ~10 file reads, ~4,293 LOC
- DEEP files: 1,010 -> ~1,020
- Expected findings: 60-80 (10 files across 2 packages, heavy benchmark focus)

## Cross-Session Notes

- **ZERO overlap with R58**: R58 covers psycho-symbolic MCP + temporal solvers + goalie reasoning. No shared files or directories.
- **Extends R43**: Directly extends benchmark deception investigation (rustc_optimization_benchmarks.rs 15%)
- **Extends R56**: Tests whether benches/ directory follows standalone_benchmark.rs (8-12%) or latency_benchmark.rs (72-78%) pattern
- **Extends R57**: Tests whether ruv-fann benchmarking.rs (20-25%) pattern extends to claude_executor.rs and metrics.rs
- **Extends R34**: matrix-utils.js extends the matrix system incompatibility investigation (2 systems -> 3 -> ?)
- **Extends R56**: JS solver layer extends R56's Rust solver assessment (do JS wrappers match Rust quality?)
- **Extends R23**: Neural throughput_benchmark.rs tests whether benchmarks match BEST IN ECOSYSTEM quality
- **NEW: JS solver entry points** (solver.js, fast-solver.js, bmssp-solver.js) never examined
- **NEW: BMSSP algorithm** is a completely unknown algorithm in the codebase
- **NEW: Cross-package benchmark comparison** -- first direct comparison of ruv-fann-rust vs sublinear-rust benchmark quality
- Combined DEEP files from R58+R59: 1,010 -> ~1,029 (approximately +19)
