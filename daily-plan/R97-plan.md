# R97 Execution Plan: Hyperbolic Geometry + SQL-Attention Remaining

**Date**: {YYYY-MM-DD}
**Session ID**: 97
**Focus**: Deep-read 9 files (~3,055 LOC) — prime-radiant hyperbolic module, remaining Postgres SQL attention, MinCut spike/config.
**Strategic value**: Prime-radiant is 89% genuine (sheaf-theoretic). Its hyperbolic module is likely real but unverified. Remaining SQL attention files (multi_head, scaled_dot, mod) complete that arc. MinCut spike.rs and config.rs fill remaining gaps.

## Rationale

R92 established the hyperbolic HNSW crate as genuine (88-95%). Prime-radiant's hyperbolic module is a separate implementation — it uses hyperbolic geometry for knowledge representation rather than nearest-neighbor search. These files determine whether prime-radiant's hyperbolic features match its overall 89% quality. The remaining SQL attention files complete the ruvector-postgres attention arc started in R96.

## Target: 9 files, ~3,055 LOC

---

### Cluster A: Prime-Radiant Hyperbolic (3 files, ~1,048 LOC)

Hyperbolic geometry for knowledge representation — energy functions, adapters, and module root.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 1 | 2525 | `src/hyperbolic/mod.rs` | 363 | Hyperbolic module root — what algorithms? |
| 2 | 2524 | `src/hyperbolic/energy.rs` | 352 | Hyperbolic energy — curvature-aware optimization? |
| 3 | 2521 | `src/hyperbolic/adapter.rs` | 333 | Hyperbolic adapter — Poincare ↔ Lorentz conversion? |

**Full paths** (all under `~/repos/ruvector/crates/prime-radiant/`):
1. `src/hyperbolic/mod.rs`
2. `src/hyperbolic/energy.rs`
3. `src/hyperbolic/adapter.rs`

**Key questions**:
- `mod.rs` (363 LOC): What does the hyperbolic module export? Does it coordinate Poincare and Lorentz models?
- `energy.rs` (352 LOC): Curvature-aware energy function for optimization on the Poincare ball? Riemannian gradient descent?
- `adapter.rs` (333 LOC): Correct Poincare ↔ Lorentz model conversion (via the hyperboloid-to-disk map)? Numerically stable at boundaries?

---

### Cluster B: SQL Attention Remaining (3 files, ~967 LOC)

Completing the ruvector-postgres attention arc (R96 reads operators.rs and flash.rs).

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 4 | 3764 | `src/attention/multi_head.rs` | 368 | SQL multi-head attention — real parallel heads? |
| 5 | 3766 | `src/attention/scaled_dot.rs` | 308 | SQL scaled dot-product attention — fundamental op? |
| 6 | 3763 | `src/attention/mod.rs` | 291 | SQL attention module root — what does it export? |

**Full paths** (all under `~/repos/ruvector/crates/ruvector-postgres/`):
4. `src/attention/multi_head.rs`
5. `src/attention/scaled_dot.rs`
6. `src/attention/mod.rs`

**Key questions**:
- `multi_head.rs` (368 LOC): Real parallel head computation in SQL or sequential loop? Does it use the scaled_dot as building block?
- `scaled_dot.rs` (308 LOC): Correct QK^T/sqrt(d_k) computation as PostgreSQL function? Matrix operations on SQL arrays?
- `mod.rs` (291 LOC): Module root — just re-exports or orchestration logic?

---

### Cluster C: MinCut Remaining (3 files, ~1,040 LOC)

Spike networks, configuration, and tiered KV cache.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 7 | 3517 | `src/spike.rs` | 366 | Spike networks — neuromorphic integration? |
| 8 | 3484 | `src/config.rs` | 369 | MinCut config — all hyperparameters? |
| 9 | 3506 | `src/kv_cache/tier.rs` | 305 | Tiered KV cache — hot/warm/cold? |

**Full paths** (all under `~/repos/ruvector/crates/ruvector-mincut-gated-transformer/`):
7. `src/spike.rs`
8. `src/config.rs`
9. `src/kv_cache/tier.rs`

**Key questions**:
- `spike.rs` (366 LOC): Extends spike_driven.rs (R93)? Neuromorphic spike-timing or renamed binary attention?
- `config.rs` (369 LOC): Complete hyperparameter surface — does it expose all knobs for the gated transformer? Sensible defaults?
- `tier.rs` (305 LOC): Hot/warm/cold tiering with promotion/demotion policies? Or just multiple Vec<> pools?

---

## Expected Outcomes

1. Prime-radiant hyperbolic module quality characterized — matches 89% crate average?
2. SQL attention arc COMPLETE (all 5 ruvector-postgres/attention files read)
3. MinCut spike network and config quality established
4. KV cache tiering architecture understood
5. DEEP count: ~1,384 → ~1,393
6. New findings: ~40-60 expected across 3,055 LOC

## DB Update Template

```javascript
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const sessionId = 97;
const today = new Date().toISOString().slice(0, 10);

// File IDs for this session:
// 2525: mod.rs (363 LOC) — ruvector-rust, DOMAIN_ONLY
// 2524: energy.rs (352 LOC) — ruvector-rust, DOMAIN_ONLY
// 2521: adapter.rs (333 LOC) — ruvector-rust, DOMAIN_ONLY
// 3764: multi_head.rs (368 LOC) — ruvector-rust, PROXIMATE
// 3766: scaled_dot.rs (308 LOC) — ruvector-rust, PROXIMATE
// 3763: mod.rs (291 LOC) — ruvector-rust, PROXIMATE
// 3517: spike.rs (366 LOC) — ruvector-rust, PROXIMATE
// 3484: config.rs (369 LOC) — ruvector-rust, PROXIMATE
// 3506: tier.rs (305 LOC) — ruvector-rust, PROXIMATE
```

## Domain Tags

- Files 2525, 2524, 2521 → `ruvector`, `v4-priority`
- Files 3764, 3766, 3763 → `ruvector`, `v4-priority`
- Files 3517, 3484, 3506 → `ruvector`, `v4-priority`

## Isolation Check

All files are in ruvector-rust (36 total cross-deps, CONNECTED). No isolation concerns.

---

## Synthesis Doc Update Protocol (ADR-040)

**MANDATORY**: After all files are read and findings inserted into the DB, update the relevant `domains/*/analysis.md` files following the ADR-040 in-place protocol. Reference: `domains/memory-and-learning/analysis.md` for canonical structure.
