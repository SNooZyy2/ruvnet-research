# Rust Compilation Audit — Complete Results

**Date**: 2026-03-03 | **Session**: R140 (id=141) | **Toolchain**: rustc 1.93.1

## Overview

Audited **159 Rust crates** across 4 repositories using `cargo check` and `cargo test --lib`
to produce binary truth signals for cross-referencing against static realness scores.

| Repo | Crates | Check PASS | Check FAIL | Excluded | Tests Run | Tests Passing |
|------|--------|-----------|-----------|----------|-----------|---------------|
| **ruvector** | 115 | 100 (87%) | 10 | 5 | 42 crates | 3,984 tests |
| **ruv-FANN** | 24 | 4 (17%) | 19 | 1 | 1 crate | 165 tests |
| **agentic-flow** | 11 | 9 (82%) | 2 | 0 | 9 crates | 167 tests |
| **sublinear** | 9 | 6 (67%) | 3 | 0 | 3 crates | 7 tests |
| **TOTAL** | **159** | **119 (75%)** | **34** | **6** | **55 crates** | **4,323 tests** |

## Key Findings

### CRITICAL — Never Compiled

| Crate | Repo | LOC | Error Class | Evidence |
|-------|------|-----|-------------|----------|
| ruvllm | ruvector | 120,345 | openssl-sys build failure | Largest crate, cannot compile |
| neural-network-implementation | sublinear | 17,294 | 106 errors, dyn-incompatible trait | Architectural defect |
| sona | ruvector | 10,582 | Package not found in workspace | Broken workspace membership |
| ruv-swarm-* (14 crates) | ruv-FANN | ~55,000 | Version mismatch (0.1.5 vs 0.2.0) | Entire subtree disconnected |
| neuro-divergent-* (5 crates) | ruv-FANN | ~32,667 | Unparseable Cargo.toml | Structurally broken manifest |

### HIGH — Check Passes, Tests Broken

| Crate | Repo | LOC | Tests | Issue |
|-------|------|-----|-------|-------|
| prime-radiant | ruvector | 52,466 | CFAIL | Test binary won't compile |
| ruvector-mincut | ruvector | 42,157 | CFAIL | Test binary won't compile |
| ruvector-graph | ruvector | 16,840 | CFAIL | Test binary won't compile |
| sublinear (root) | sublinear | 35,136 | CFAIL (7 errors) | Missing methods on test types |
| agentic-jujutsu | agentic-flow | 9,138 | 5f (2 security) | ML-DSA accepts invalid sigs |
| agent-booster | agentic-flow | 2,292 | 6f | Strategy selection broken |

### Genuine — Heavily Tested Crates (>100 tests)

| Crate | Repo | LOC | Tests | Signal |
|-------|------|-----|-------|--------|
| ruqu-core | ruvector | 26,093 | 602p/0f | Quantum core, massively tested |
| ruvector-nervous-system | ruvector | 14,708 | 359p/0f | Heavily tested |
| ruvector-temporal-tensor | ruvector | 11,446 | 269p/0f | Heavily tested |
| ruvector-robotics | ruvector | 9,578 | 252p/0f | Heavily tested |
| rvf-types | ruvector/rvf | 8,484 | 230p/0f | Core format types |
| rvf-runtime | ruvector/rvf | 13,572 | 219p/0f | Core runtime |
| ruvector-gnn | ruvector | 8,083 | 198p/0f | GNN subsystem |
| ruv-fann (root) | ruv-FANN | 27,630 | 165p/0f | Root crate, well-tested |
| ruvector-math | ruvector | 13,166 | 148p/0f | Math subsystem |
| ruvector-attention | ruvector | 16,003 | 142p/0f | Flash attention |
| ruvector-solver | ruvector | 10,892 | 140p/0f | Solver subsystem |
| ruvector-core | ruvector | 12,658 | 122p/1f | Core DB (1 test failure) |

## Error Patterns

### 1. Version/Workspace Mismatches (19 crates)
ruv-FANN's `ruv-swarm-*` subtree depends on `ruv-fann = "^0.1.5"` but root is `0.2.0`.
Proves these crates were added without running `cargo check` against the workspace.

### 2. getrandom/WASM Feature Flags (5 crates)
WASM crates missing `getrandom` "js" feature. Common oversight when using `uuid`/`rand`
on `wasm32-unknown-unknown`. One-line Cargo.toml fix per crate.

### 3. System Dependency Failures (2 crates)
`ruvllm` and `ruvllm-cli` fail on `openssl-sys` build script. Needs `libssl-dev` or
a vendored openssl feature. Not a code quality issue per se, but blocks compilation.

### 4. Test Binary CFAIL Pattern (6 crates)
`cargo check` passes (library compiles) but `cargo test --lib` fails to build the test
binary. Tests reference types/methods that exist in test code but not in library. Proves
tests were written speculatively and never executed.

### 5. Package Not Found (3 crates)
`sona`, `rvf`, `ruQu` — crate names don't match workspace member declarations.
Likely renamed without updating the workspace Cargo.toml.

## Files

| File | Description |
|------|-------------|
| `results-ruvector.txt` | Per-crate results for 115 ruvector crates |
| `results-ruv-fann.txt` | Per-crate results for 24 ruv-FANN crates |
| `results-agentic-flow.txt` | Per-crate results for 11 agentic-flow crates |
| `results-sublinear.txt` | Per-crate results for 9 sublinear crates |
| `results-ruvector-errors.txt` | Error details for failing ruvector crates |
| `results-ruv-fann-errors.txt` | Error details for failing ruv-FANN crates |
| `run-ruv-fann.sh` | Audit script for ruv-FANN |
| `run-ruvector.sh` | Audit script for ruvector (sequential) |
| `batches/` | Crate lists ordered by research tier |

## DB Records

67 findings inserted into session 141 (`[COMPILATION AUDIT]` prefix).
31 crates skipped (rvf-* sub-workspace and newer crates not yet in files/crates tables).
Agentic-flow and sublinear findings were inserted by the remote server session.
