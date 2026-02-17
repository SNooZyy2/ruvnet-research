# R98 Execution Plan: SONA Remaining + Hyperbolic Postgres + Small Files

**Date**: {YYYY-MM-DD}
**Session ID**: 98
**Focus**: Deep-read 9 files (~1,989 LOC) — SQL hyperbolic functions, MinCut crate root + kernel norm, SONA loop variants, prime-radiant hyperbolic config/depth.
**Strategic value**: Fills remaining gaps across three areas. SQL hyperbolic functions (Poincare/Lorentz in Postgres) extend the R92/R97 hyperbolic arc into the database layer. SONA loops complete the training loop subsystem. MinCut lib.rs reveals what the crate actually exports.

## Rationale

This is a mixed session clearing mid-priority files across several arcs. All files are under 300 LOC, making this a high-throughput session. The Postgres hyperbolic files are the most strategically interesting — they determine whether hyperbolic geometry operations run inside the database (novel) or are just stub declarations.

## Target: 9 files, ~1,989 LOC

---

### Cluster A: SQL Hyperbolic Functions (2 files, ~526 LOC)

PostgreSQL extension implementations for Poincare and Lorentz hyperbolic geometry.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 1 | 3830 | `src/hyperbolic/poincare.rs` | 268 | SQL Poincare ball functions? |
| 2 | 3827 | `src/hyperbolic/lorentz.rs` | 258 | SQL Lorentz model functions? |

**Full paths** (all under `~/repos/ruvector/crates/ruvector-postgres/`):
1. `src/hyperbolic/poincare.rs`
2. `src/hyperbolic/lorentz.rs`

**Key questions**:
- `poincare.rs` (268 LOC): Real pgx functions for Poincare ball distance, exponential map? Or SQL string templates?
- `lorentz.rs` (258 LOC): Minkowski inner product as SQL function? Does it match R92's lorentz_cascade.rs math?

---

### Cluster B: MinCut Crate Root + Kernel (2 files, ~474 LOC)

Crate entry point and normalization kernel.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 3 | 3507 | `src/lib.rs` | 261 | Crate root — what does it export? |
| 4 | 3493 | `src/kernel/norm.rs` | 213 | Layer normalization kernel? RMSNorm? |

**Full paths** (all under `~/repos/ruvector/crates/ruvector-mincut-gated-transformer/`):
3. `src/lib.rs`
4. `src/kernel/norm.rs`

**Key questions**:
- `lib.rs` (261 LOC): What does the crate publicly export? Does it expose the full inference pipeline or just building blocks?
- `norm.rs` (213 LOC): RMSNorm (LLaMA style) or LayerNorm? Real SIMD optimization or scalar?

---

### Cluster C: SONA Loops + Time Compat (3 files, ~604 LOC)

Training loop variants and time compatibility.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 5 | 4461 | `src/loops/background.rs` | 235 | Background training loop? |
| 6 | 4462 | `src/loops/coordinator.rs` | 227 | Loop coordinator — orchestrates training stages? |
| 7 | 4470 | `src/time_compat.rs` | 142 | Time compatibility layer? |

**Full paths** (all under `~/repos/ruvector/crates/sona/`):
5. `src/loops/background.rs`
6. `src/loops/coordinator.rs`
7. `src/time_compat.rs`

**Key questions**:
- `background.rs` (235 LOC): Real async background training (tokio spawn) or blocking loop on separate thread?
- `coordinator.rs` (227 LOC): Orchestrates pretrain → finetune → export stages? Or just delegates to pipeline.rs?
- `time_compat.rs` (142 LOC): What time compatibility is needed? Chrono ↔ std::time? Or temporal-tensor integration?

---

### Cluster D: Prime-Radiant Hyperbolic Config (2 files, ~385 LOC)

Configuration and depth estimation for the hyperbolic knowledge substrate.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 8 | 2523 | `src/hyperbolic/depth.rs` | 215 | Hierarchy depth estimation? |
| 9 | 2522 | `src/hyperbolic/config.rs` | 170 | Hyperbolic geometry config? |

**Full paths** (all under `~/repos/ruvector/crates/prime-radiant/`):
8. `src/hyperbolic/depth.rs`
9. `src/hyperbolic/config.rs`

**Key questions**:
- `depth.rs` (215 LOC): Estimating hierarchy depth from hyperbolic embeddings? Uses Poincare norm as depth proxy?
- `config.rs` (170 LOC): Curvature parameters, dimension settings? Sensible defaults for different use cases?

---

## Expected Outcomes

1. SQL hyperbolic functions assessed — genuine pgx or stubs
2. MinCut crate public API understood (lib.rs)
3. SONA training loops completed
4. Prime-radiant hyperbolic module COMPLETE (all 5 files: mod, energy, adapter, depth, config)
5. DEEP count: ~1,393 → ~1,402
6. New findings: ~30-50 expected across 1,989 LOC

## DB Update Template

```javascript
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const sessionId = 98;
const today = new Date().toISOString().slice(0, 10);

// File IDs for this session:
// 3830: poincare.rs (268 LOC) — ruvector-rust, PROXIMATE
// 3827: lorentz.rs (258 LOC) — ruvector-rust, PROXIMATE
// 3507: lib.rs (261 LOC) — ruvector-rust, PROXIMATE
// 3493: norm.rs (213 LOC) — ruvector-rust, PROXIMATE
// 4461: background.rs (235 LOC) — ruvector-rust, PROXIMATE
// 4462: coordinator.rs (227 LOC) — ruvector-rust, PROXIMATE
// 4470: time_compat.rs (142 LOC) — ruvector-rust, PROXIMATE
// 2523: depth.rs (215 LOC) — ruvector-rust, DOMAIN_ONLY
// 2522: config.rs (170 LOC) — ruvector-rust, DOMAIN_ONLY
```

## Domain Tags

- Files 3830, 3827 → `ruvector`, `v4-priority`
- Files 3507, 3493 → `ruvector`, `v4-priority`
- Files 4461, 4462, 4470 → `memory-and-learning`, `v4-priority`
- Files 2523, 2522 → `ruvector`, `v4-priority`

## Isolation Check

All files are in ruvector-rust (36 total cross-deps, CONNECTED). No isolation concerns.

---

## Synthesis Doc Update Protocol (ADR-040)

**MANDATORY**: After all files are read and findings inserted into the DB, update the relevant `domains/*/analysis.md` files following the ADR-040 in-place protocol. Reference: `domains/memory-and-learning/analysis.md` for canonical structure.
