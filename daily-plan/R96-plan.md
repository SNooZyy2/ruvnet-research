# R96 Execution Plan: SQL-Attention + MinCut KV-Cache + AgentDB

**Date**: {YYYY-MM-DD}
**Session ID**: 96
**Focus**: Deep-read 9 files (~3,737 LOC) — Postgres SQL attention mechanisms, remaining KV cache variants, arena allocator, and AgentDB MemoryController.
**Strategic value**: R91 debunked "39 attention mechanisms in SQL" (AttentionService.ts db param is dead code). These Rust files are the actual PostgreSQL extension implementations — they determine whether SQL attention is genuine pgx/C-FFI code or another facade. The MinCut KV cache files complete that subsystem. AgentDB MemoryController is the last unread AgentDB file.

## Rationale

The SQL attention arc has been unresolved since R91 debunked the "39 SQL attention" claim. These Rust files in ruvector-postgres are the actual implementation layer — they either produce valid PostgreSQL extension functions or are design-doc-as-code (like R90's ruvector-graph distributed pattern). The MinCut KV cache files (squat, kivi, policy, hot_buffer) complete the cache subsystem started in R91 (legacy.rs) and R93 (manager.rs, kvquant.rs). The arena allocator may be the memory backing for the entire MinCut inference pipeline.

## Target: 9 files, ~3,737 LOC

---

### Cluster A: MinCut KV Cache + Arena (5 files, ~2,265 LOC)

Completing the KV cache subsystem — specific quantization algorithms, eviction policies, and hot buffer.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 1 | 3479 | `src/arena.rs` | 472 | Arena allocator — real bump alloc or Vec wrapper? |
| 2 | 3505 | `src/kv_cache/squat.rs` | 467 | SQUAT quantized attention — specific algorithm? |
| 3 | 3497 | `src/kv_cache/kivi.rs` | 458 | KIVI — Key-Value cache quantization method? |
| 4 | 3503 | `src/kv_cache/policy.rs` | 440 | Eviction policy — LRU/H2O/attention-sink? |
| 5 | 3496 | `src/kv_cache/hot_buffer.rs` | 419 | Hot buffer — fast path for recent tokens? |

**Full paths** (all under `~/repos/ruvector/crates/ruvector-mincut-gated-transformer/`):
1. `src/arena.rs`
2. `src/kv_cache/squat.rs`
3. `src/kv_cache/kivi.rs`
4. `src/kv_cache/policy.rs`
5. `src/kv_cache/hot_buffer.rs`

**Key questions**:
- `arena.rs` (472 LOC): Real bump allocator with reset semantics, or Vec::with_capacity wrapper? Does MinCut inference use this for all tensor allocations?
- `squat.rs` (467 LOC): SQUAT (Shared QUantized ATtention) — is this a known algorithm or custom? Does it reduce KV cache memory?
- `kivi.rs` (458 LOC): KIVI (Key-Value cache with Integer quantization) — does it implement the KIVI paper's per-channel quantization?
- `policy.rs` (440 LOC): Which eviction policies? H2O (Heavy-Hitter Oracle), attention-sink, StreamingLLM, or simpler LRU?
- `hot_buffer.rs` (419 LOC): Fast path for recently-used KV entries? Does it interact with the eviction policy?

---

### Cluster B: SQL Attention (2 files, ~814 LOC)

PostgreSQL extension implementations for attention operators and flash attention.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 6 | 3765 | `src/attention/operators.rs` | 426 | SQL attention operators — CREATE FUNCTION wrappers? |
| 7 | 3762 | `src/attention/flash.rs` | 388 | SQL Flash Attention — PostgreSQL C extension? |

**Full paths** (all under `~/repos/ruvector/crates/ruvector-postgres/`):
6. `src/attention/operators.rs`
7. `src/attention/flash.rs`

**Key questions**:
- `operators.rs` (426 LOC): Real pgx `#[pg_extern]` functions or raw SQL strings? Do they produce compilable PostgreSQL extensions?
- `flash.rs` (388 LOC): Flash Attention as a PostgreSQL function? Does it tile over SQL result sets or operate on in-memory tensors?

---

### Cluster C: MinCut Tracing + AgentDB (2 files, ~875 LOC)

Attention tracing/visualization and the last AgentDB file.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 8 | 3519 | `src/trace.rs` | 413 | Tracing — attention visualization? Debug tooling? |
| 9 | 12871 | `MemoryController.ts` | 462 | Memory controller — CRUD for agent memories? |

**Full paths**:
8. `~/repos/ruvector/crates/ruvector-mincut-gated-transformer/src/trace.rs`
9. `~/repos/agentic-flow/packages/agentdb/src/controllers/MemoryController.ts`

**Key questions**:
- `trace.rs` (413 LOC): Real attention weight capture and visualization output (JSON/SVG) or just logging?
- `MemoryController.ts` (462 LOC): Real CRUD controller with pagination, filtering, TTL? Or thin wrapper over raw SQL?

---

## Expected Outcomes

1. SQL attention verdict: genuine pgx extensions or design-doc-as-code
2. MinCut KV cache subsystem COMPLETE (6 files across R91/R93/R96)
3. Arena allocator role in MinCut inference pipeline clarified
4. AgentDB file coverage COMPLETE
5. DEEP count: ~1,375 → ~1,384
6. New findings: ~50-70 expected across 3,737 LOC

## DB Update Template

```javascript
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const sessionId = 96;
const today = new Date().toISOString().slice(0, 10);

// File IDs for this session:
// 3479: arena.rs (472 LOC) — ruvector-rust, PROXIMATE
// 3505: squat.rs (467 LOC) — ruvector-rust, PROXIMATE
// 3497: kivi.rs (458 LOC) — ruvector-rust, PROXIMATE
// 3503: policy.rs (440 LOC) — ruvector-rust, PROXIMATE
// 3496: hot_buffer.rs (419 LOC) — ruvector-rust, PROXIMATE
// 3765: operators.rs (426 LOC) — ruvector-rust, PROXIMATE
// 3762: flash.rs (388 LOC) — ruvector-rust, PROXIMATE
// 3519: trace.rs (413 LOC) — ruvector-rust, PROXIMATE
// 12871: MemoryController.ts (462 LOC) — agentic-flow-rust, PROXIMATE
```

## Domain Tags

- Files 3479, 3505, 3497, 3503, 3496, 3519 → `ruvector`, `v4-priority`
- Files 3765, 3762 → `ruvector`, `v4-priority`
- File 12871 → `agentdb-integration`, `memory-and-learning`

## Isolation Check

All files are in CONNECTED packages (ruvector-rust: 36 cross-deps, agentic-flow-rust: 27 cross-deps). No isolation concerns.

---

## Synthesis Doc Update Protocol (ADR-040)

**MANDATORY**: After all files are read and findings inserted into the DB, update the relevant `domains/*/analysis.md` files following the ADR-040 in-place protocol. Reference: `domains/memory-and-learning/analysis.md` for canonical structure.
