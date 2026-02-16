# R52 Execution Plan: ruvector Algorithmic Infrastructure Deep-Dive

**Date**: 2026-02-16
**Session ID**: 52
**Focus**: HNSW implementation patches, SPARQL parser, subpolynomial mincut algorithms, edge SIMD compute, and postgres benchmarks
**Parallel with**: R53 (no file overlap -- R52 = ruvector-rust algorithmic core, R53 = sublinear-rust MCP tools + strange-loop runtime)

## IMPORTANT: Parallel Execution Notice

This plan runs IN PARALLEL with R53. The file lists are strictly non-overlapping:
- **R52 covers**: ruvector-rust ONLY -- HNSW patches, SPARQL parser, subpolynomial algorithms, edge SIMD, postgres benchmarks
- **R53 covers**: sublinear-rust ONLY -- strange-loop MCP/CLI/examples, MCP tools layer, core optimizers
- **ZERO shared files** between R52 and R53
- **ZERO shared packages** -- R52 is entirely ruvector-rust, R53 is entirely sublinear-rust
- Do NOT read or analyze any file from R53's list (see R53-plan.md for that list)

## Rationale

- R12-R15 covered ruvector-core and found REAL SIMD (AVX-512/AVX2/NEON) and HNSW wrapping hnsw_rs, but the **actual HNSW patches** applied to hnsw_rs have NEVER been read
- R38 confirmed rvlite has a working Cypher executor, but the **SPARQL parser** (2,496 LOC) -- the LARGEST untouched Rust source file in ruvector -- has NEVER been examined
- R34 found ruvector-mincut is BEST algorithmic with 2 incompatible matrix systems, but the **subpolynomial module** (1,385 LOC) was not in scope
- R36 found HNSW patches are genuine infrastructure, but only assessed them structurally -- never line-by-line
- These 6 files represent the **algorithmic foundation** that all higher-level ruvector features depend on

## Target: 6 files, ~10,271 LOC

---

### Cluster A: HNSW Implementation Patches (2 files, ~3,577 LOC)

These are patches to the upstream `hnsw_rs` library -- the core algorithm that ruvector-core wraps for approximate nearest neighbor search. Understanding these patches reveals what ruvector actually adds to the base HNSW algorithm.

| # | File ID | File | LOC | Domain |
|---|---------|------|-----|--------|
| 1 | 8254 | `scripts/patches/hnsw_rs/src/hnsw.rs` | 1,873 | ruvector |
| 2 | 8255 | `scripts/patches/hnsw_rs/src/hnswio.rs` | 1,704 | ruvector |

**Key questions**:
- `hnsw.rs`: What modifications were made to the upstream hnsw_rs library? The original hnsw_rs implements the HNSW algorithm (Malkov & Yashunin 2018) -- do these patches enhance the algorithm (better neighbor selection, improved layer construction), add features (batch insert, concurrent access, custom distance metrics), or fix bugs?
- Does it contain the actual HNSW graph construction with layer assignment, greedy search, and neighbor pruning? Or is it a thin shim?
- R12-R15 confirmed ruvector-core has REAL SIMD (AVX-512/AVX2/NEON) -- do these patches add SIMD distance calculations to the base hnsw_rs implementation?
- R36 found the HNSW patches are "genuine infrastructure" structurally -- does line-by-line examination confirm genuine algorithmic modifications?
- Are there any custom distance metrics beyond L2/cosine (e.g., Hamming, dot product with quantization)?
- `hnswio.rs`: How is the HNSW index serialized/deserialized? Key aspects:
  - Memory-mapped I/O for large indices?
  - Incremental saves (append-only log) or full rewrite on each save?
  - Format versioning for backward compatibility?
  - Does this connect to the ruvector-postgres persistence layer (R35 found architecture-complete, persistence-incomplete)?
  - Is there support for partial loading (load only specific layers of the HNSW graph)?
  - R44 found THREE disconnected AgentDB layers -- does HNSW I/O serialize to formats compatible with any of them?

**Follow-up context**:
- R12-R15: ruvector-core wraps hnsw_rs, REAL SIMD (AVX-512/AVX2/NEON)
- R36: HNSW patches assessed structurally as genuine infrastructure, but never line-by-line DEEP
- R37: hnsw_router.rs (90-93%) BEST ruvector-core integration -- uses HNSW for model routing
- R28: sparse.rs (95%) BEST algorithmic file overall

---

### Cluster B: Graph Query Layer (2 files, ~3,891 LOC)

The SPARQL parser is the LARGEST untouched Rust source file in ruvector. Combined with postgres index benchmarks, this cluster reveals whether ruvector has genuine graph query capabilities.

| # | File ID | File | LOC | Domain |
|---|---------|------|-----|--------|
| 3 | 3809 | `crates/ruvector-postgres/src/graph/sparql/parser.rs` | 2,496 | ruvector |
| 4 | 3666 | `crates/ruvector-postgres/benches/index_bench.rs` | 1,395 | ruvector |

**Key questions**:
- `parser.rs`: Is this a genuine SPARQL parser implementing the W3C specification? Key indicators:
  - Does it parse standard SPARQL syntax: SELECT, WHERE, OPTIONAL, FILTER, GROUP BY, ORDER BY, UNION?
  - Does it support basic graph patterns (BGPs), property paths, and named graphs?
  - Is there an AST representation with proper node types, or string concatenation?
  - At 2,496 LOC, there's room for a real recursive-descent or PEG parser -- is it one?
  - R38 confirmed rvlite has a working Cypher executor -- does the SPARQL parser connect to a similar executor?
  - ruvector-graph has a Cypher parser (R12-R15 DEEP) -- are SPARQL and Cypher query systems integrated, or completely separate?
  - Is the parser used by any production code path, or is it a research prototype?
- `index_bench.rs`: Are the postgres index benchmarks genuine? Key indicators:
  - Does it create real pgvector or HNSW indices in postgres?
  - Does it measure real kNN query latency, recall@k, throughput at varying dimensions?
  - Does it use criterion or similar benchmarking frameworks with proper warmup and statistical analysis?
  - R43 found rustc_benchmarks uses asymptotic mismatch deception -- does this benchmark exhibit the same pattern?
  - R43 found ruvector-benchmark.ts (92%) validates HNSWIndex genuinely -- do the Rust postgres benchmarks match that quality?

**Follow-up context**:
- R38: rvlite HAS working Cypher executor (corrects R13's "no executor" finding)
- R35: ruvector-postgres architecture-complete, persistence-incomplete
- R43: rustc_benchmarks MOST DECEPTIVE (15%) -- asymptotic mismatch deception
- R43: ruvector-benchmark.ts (92%) REAL performance testing

---

### Cluster C: Subpolynomial Algorithms & Edge SIMD (2 files, ~2,803 LOC)

Extends R34's finding that ruvector-mincut is BEST algorithmic, and validates whether edge computing SIMD claims are real.

| # | File ID | File | LOC | Domain |
|---|---------|------|-----|--------|
| 5 | 3451 | `crates/ruvector-mincut/src/subpolynomial/mod.rs` | 1,385 | ruvector |
| 6 | 5409 | `examples/edge-net/src/compute/simd.rs` | 1,418 | ruvector |

**Key questions**:
- `subpolynomial/mod.rs`: R34 found ruvector-mincut is BEST algorithmic. Does this subpolynomial module implement genuine subpolynomial-time graph algorithms?
  - Does it implement Karger's randomized mincut, Karger-Stein algorithm, or novel approaches?
  - Are the claimed complexity bounds (sub-polynomial) actually achieved in the implementation?
  - R39 found FALSE sublinearity in sublinear-time-solver (all O(n^2)+) -- does ruvector-mincut's subpolynomial module suffer the same issue?
  - R42 found dynamic_mincut EXCEEDS R34 -- is this module related to or independent of dynamic_mincut?
  - At 1,385 LOC, there's room for significant algorithmic content -- is it real or inflated with comments/boilerplate?
- `simd.rs`: Does edge-net's SIMD compute implement real SIMD operations?
  - Are there actual `#[target_feature(enable = "avx2")]` or `#[cfg(target_arch = "aarch64")]` annotations?
  - R12 confirmed ruvector-core has REAL SIMD (AVX-512/AVX2/NEON) -- does the edge computing layer share this code, or is it independent SIMD?
  - Does it implement distance calculations (L2, cosine, dot product) with SIMD intrinsics?
  - R38 found CUDA-WASM has 4 backends -- does edge SIMD connect to any GPU backend?
  - Is this intended for WebAssembly SIMD (wasm-simd128) for browser execution, or native SIMD for server-side edge computing?

**Follow-up context**:
- R34: ruvector-mincut BEST algorithmic, 2 incompatible matrix systems discovered
- R42: dynamic_mincut EXCEEDS R34's original assessment
- R39: FALSE sublinearity confirmed in sublinear-time-solver (all O(n^2)+)
- R12: ruvector-core REAL SIMD (AVX-512/AVX2/NEON)
- R28: sparse.rs (95%) BEST algorithmic file in entire project

---

## Expected Outcomes

- **HNSW quality verdict**: Whether the patches are genuine algorithmic improvements to hnsw_rs (performance, features, bug fixes) or cosmetic modifications
- **SPARQL reality check**: Whether ruvector has a genuine SPARQL parser with real W3C compliance, or a simplified subset/stub
- **Graph query integration**: Whether SPARQL and Cypher parsers are integrated or completely separate query systems
- **Benchmark integrity**: Whether postgres benchmarks are genuine (like ruvector-benchmark.ts 92%) or deceptive (like rustc_benchmarks 15%)
- **Subpolynomial validation**: Whether the claimed subpolynomial complexity bounds are real or fabricated (like R39's false sublinearity findings)
- **Edge SIMD assessment**: Whether edge computing SIMD matches ruvector-core's genuine SIMD quality

## Stats Target

- ~6 file reads, ~10,271 LOC
- DEEP files: 955 -> ~961
- Expected findings: 50-80 (dense algorithmic code yields more findings per file)

## Cross-Session Notes

- **ZERO overlap with R53**: R53 covers sublinear-rust MCP tools (domain-management, domain-registry, scheduler, psycho-symbolic-dynamic), strange-loop runtime (MCP server, CLI, examples), and core optimizers (optimized-matrix, performance-optimizer). Completely different package.
- **ZERO overlap with R51**: R51 covered agentic-flow runtime execution layer (MCP bridge, worker pipeline, coordination). No ruvector-rust files were read.
- **Extends R12-R15**: R12-R15 covered ruvector-core crate internals but never examined the HNSW patches or SPARQL parser
- **Extends R34**: R34 found ruvector-mincut BEST algorithmic but didn't cover subpolynomial/mod.rs
- **Extends R36**: R36 found HNSW patches are genuine infrastructure -- this session validates that with line-by-line reading
- If Cluster A confirms genuine HNSW patches, ruvector-core's quality assessment strengthens further
- If Cluster B finds real SPARQL parsing, ruvector has TWO genuine query languages (Cypher confirmed R38 + SPARQL)
- If Cluster C finds genuine subpolynomial algorithms, it validates and extends R34's BEST algorithmic finding
- Combined DEEP files from R52+R53: 955 -> ~970 (approximately +15)
