# R100 Execution Plan: Final Sweep — All Remaining Small Files

**Date**: {YYYY-MM-DD}
**Session ID**: 100
**Focus**: Clear the entire v4-priority queue. ~15 files, ~1,100 LOC — all under 300 LOC (entry points, mod.rs, error types).
**Strategic value**: Queue completion. Every file is a module root, error type, or tiny entry point. No surprises expected, but mod.rs files reveal what each crate actually exports publicly. Completing R100 means the v4-priority domain is fully characterized.

## Rationale

This is the final sweep. All remaining files are under 300 LOC — module roots (mod.rs, lib.rs), error type definitions, and small entry points. While individually low-value, they collectively complete the coverage map for MinCut-Gated-Transformer, SONA, ruvector-attention hyperbolic, ruvector-hyperbolic-hnsw, and ruvector-postgres. The mod.rs files are particularly useful for understanding what each crate publicly exports.

## Target: ~15 files, ~1,100 LOC

---

### Cluster A: ruvector-attention Hyperbolic (4 files, ~620 LOC)

Hyperbolic attention module — mixed curvature, Poincare attention, and module root.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 1 | — | `src/hyperbolic/mixed_curvature.rs` | 241 | Mixed curvature spaces — Poincare + Lorentz in one model? |
| 2 | — | `src/hyperbolic/poincare.rs` | 181 | Poincare ball attention — complements R92's lorentz_cascade? |
| 3 | — | `src/hyperbolic/hyperbolic_attention.rs` | 172 | Generic hyperbolic attention — base trait? |
| 4 | — | `src/hyperbolic/mod.rs` | 26 | Module root — what does it export? |

**Full paths** (all under `~/repos/ruvector/crates/ruvector-attention/`):
1. `src/hyperbolic/mixed_curvature.rs`
2. `src/hyperbolic/poincare.rs`
3. `src/hyperbolic/hyperbolic_attention.rs`
4. `src/hyperbolic/mod.rs`

---

### Cluster B: ruvector-hyperbolic-hnsw Small Files (2 files, ~254 LOC)

Crate root and error types.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 5 | — | `src/lib.rs` | 211 | Crate root — public API surface |
| 6 | — | `src/error.rs` | 43 | Error types — custom or thiserror? |

**Full paths** (all under `~/repos/ruvector/crates/ruvector-hyperbolic-hnsw/`):
5. `src/lib.rs`
6. `src/error.rs`

---

### Cluster C: MinCut-Gated-Transformer Small Files (5 files, ~319 LOC)

Module roots and error types for KV cache, attention, and kernel subsystems.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 7 | — | `src/kv_cache/mod.rs` | 98 | KV cache module root — which cache types are public? |
| 8 | — | `src/error.rs` | 95 | Error types — inference errors, cache errors? |
| 9 | — | `src/attention/linear.rs` | 67 | Linear attention — O(n) attention variant? |
| 10 | — | `src/attention/mod.rs` | 35 | Attention module root |
| 11 | — | `src/kernel/mod.rs` | 24 | Kernel module root |

**Full paths** (all under `~/repos/ruvector/crates/ruvector-mincut-gated-transformer/`):
7. `src/kv_cache/mod.rs`
8. `src/error.rs`
9. `src/attention/linear.rs`
10. `src/attention/mod.rs`
11. `src/kernel/mod.rs`

---

### Cluster D: SONA Small Files (3 files, ~110 LOC)

Training and loop module roots.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 12 | — | `src/training/mod.rs` | 71 | Training module root — what training modes are exported? |
| 13 | — | `src/loops/mod.rs` | 15 | Loops module root |
| 14 | — | `src/mod.rs` | 24 | Crate-level module |

**Full paths** (all under `~/repos/ruvector/crates/sona/`):
12. `src/training/mod.rs`
13. `src/loops/mod.rs`
14. `src/mod.rs`

---

### Cluster E: Remaining (2 files, ~65 LOC)

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 15 | — | `neural-net-impl/.../inference/memory_pool.rs` | 34 | Inference memory pool — relates to R85's memory_pool.rs? |
| 16 | — | `ruvector-postgres/src/hyperbolic/mod.rs` | 31 | SQL hyperbolic module root |

**Full paths**:
15. `~/repos/ruv-FANN/neural-net-impl/src/inference/memory_pool.rs`
16. `~/repos/ruvector/crates/ruvector-postgres/src/hyperbolic/mod.rs`

---

## Expected Outcomes

1. v4-priority queue CLEARED — all ~92 files characterized
2. Public API surface understood for all crates (via mod.rs/lib.rs reads)
3. Linear attention variant assessed (could be novel O(n) approach)
4. DEEP count: ~1,405-1,411 → ~1,420-1,427
5. New findings: ~20-30 expected across 1,100 LOC (mostly low-severity)

## DB Update Template

```javascript
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const sessionId = 100;
const today = new Date().toISOString().slice(0, 10);

// File IDs: resolve from DB before session — many are small files
// that may not have file IDs assigned yet. Check files table first.
```

## Domain Tags

- Cluster A → `ruvector`, `v4-priority`
- Cluster B → `ruvector`, `v4-priority`
- Cluster C → `ruvector`, `v4-priority`
- Cluster D → `memory-and-learning`, `v4-priority`
- Cluster E → `ruvector` (16), `v4-priority`

## Isolation Check

All files are in CONNECTED packages. No isolation concerns.

---

## Synthesis Doc Update Protocol (ADR-040)

**MANDATORY**: After all files are read and findings inserted into the DB, update the relevant `domains/*/analysis.md` files following the ADR-040 in-place protocol. Reference: `domains/memory-and-learning/analysis.md` for canonical structure.
