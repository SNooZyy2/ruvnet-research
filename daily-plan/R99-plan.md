# R99 Execution Plan: Hyperbolic HNSW Tangent + GNN Bindings

**Date**: {YYYY-MM-DD}
**Session ID**: 99
**Focus**: Deep-read 3 files (~1,192 LOC) — hyperbolic HNSW tangent space operations, GNN NAPI and WASM bindings.
**Strategic value**: R92 already read 6 of the original 9 files (hnsw.rs, poincare.rs, lorentz_cascade.rs, hyperbolic-hnsw-wasm/lib.rs, shard.rs, flash_attention.rs — all DEEP). The remaining 3 files complete the hyperbolic HNSW crate (tangent.rs) and determine whether ruvector-gnn has functional JS/WASM bindings or theatrical stubs (currently 17 genuine, 13 theatrical WASM).

## Rationale

R92 confirmed the hyperbolic HNSW crate as genuine (88-95%). tangent.rs is the last unread file in that crate — it likely provides tangent-space projections used as fast-path approximations for Poincare ball operations. The GNN NAPI and WASM bindings are the last unread GNN files and test the theatrical-vs-genuine pattern for this crate.

This is a small session (3 files) — consider merging with R100's final sweep if scheduling allows.

## Target: 3 files, ~1,192 LOC

---

### Cluster A: Hyperbolic HNSW Tangent (1 file, ~349 LOC)

Last unread file in the ruvector-hyperbolic-hnsw crate. R92 established 88-95% quality for the crate.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 1 | 3279 | `src/tangent.rs` | 349 | Tangent space operations — logarithmic/exponential maps, fast-path distance? |

**Full path**:
1. `~/repos/ruvector/crates/ruvector-hyperbolic-hnsw/src/tangent.rs`

**Key questions**:
- `tangent.rs` (349 LOC): Does it implement tangent space projection for approximate nearest-neighbor operations? Are the exponential/logarithmic maps consistent with poincare.rs (R92)? Does it provide the fast-path optimization for distance computation in HNSW traversal?

---

### Cluster B: GNN Bindings (2 files, ~843 LOC)

NAPI and WASM bindings for the GNN crate. R91 showed bimodal GNN quality (mmap.rs 88-92% vs compress.rs 55-65%). These bindings determine JS/browser interop quality.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 2 | 3177 | `ruvector-gnn-node/src/lib.rs` | 428 | GNN NAPI binding — does it compile? Real N-API or stub? |
| 3 | 3182 | `ruvector-gnn-wasm/src/lib.rs` | 415 | GNN WASM binding — genuine or theatrical? 18th genuine or 14th theatrical? |

**Full paths**:
2. `~/repos/ruvector/crates/ruvector-gnn-node/src/lib.rs`
3. `~/repos/ruvector/crates/ruvector-gnn-wasm/src/lib.rs`

**Key questions**:
- `lib.rs (gnn-node)` (428 LOC): Real napi-rs bindings with proper type conversion (JsBuffer ↔ Vec<f32>)? Does it expose training or just inference? Does it import from ruvector-gnn core?
- `lib.rs (gnn-wasm)` (415 LOC): Real wasm_bindgen with functional graph operations? Or another console.log theatrical stub? Does it expose GNN training APIs to the browser?

---

## Expected Outcomes

1. Hyperbolic HNSW crate COMPLETE — all files characterized
2. GNN NAPI/WASM binding quality assessed — genuine or theatrical
3. WASM count updated (if GNN-WASM is genuine: 18th, if theatrical: 14th)
4. DEEP count: ~1,402 → ~1,405
5. New findings: ~15-25 expected across 1,192 LOC

## DB Update Template

```javascript
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const sessionId = 99;
const today = new Date().toISOString().slice(0, 10);

// File IDs for this session:
// 3279: tangent.rs (349 LOC) — ruvector-rust, DOMAIN_ONLY
// 3177: gnn-node/lib.rs (428 LOC) — ruvector-rust, DOMAIN_ONLY
// 3182: gnn-wasm/lib.rs (415 LOC) — ruvector-rust, DOMAIN_ONLY
//
// REMOVED (already DEEP from R92):
// 3275: hnsw.rs, 3277: poincare.rs, 3278: shard.rs,
// 3283: hyperbolic-hnsw-wasm/lib.rs, 2738: lorentz_cascade.rs,
// 8530: flash_attention.rs
```

## Domain Tags

- All files → `ruvector`, `v4-priority`

## Isolation Check

All files are in ruvector-rust (36 total cross-deps, CONNECTED). No isolation concerns.

---

## Synthesis Doc Update Protocol (ADR-040)

**MANDATORY**: After all files are read and findings inserted into the DB, update the relevant `domains/*/analysis.md` files following the ADR-040 in-place protocol. Reference: `domains/memory-and-learning/analysis.md` for canonical structure.
