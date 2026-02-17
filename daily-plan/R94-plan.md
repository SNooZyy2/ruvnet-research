# R94 Execution Plan: MinCut-Transformer Extended + GNN

**Date**: {YYYY-MM-DD}
**Session ID**: 94
**Focus**: Deep-read 9 files (~4,570 LOC) — remaining larger MinCut-Transformer files + GNN scheduler/replay.
**Strategic value**: Extends R93's MinCut core reads with module routing (MoE), model state, KV cache variants, and packet abstraction. GNN scheduler and replay buffer are the key remaining unread files in the ruvector-gnn crate (bimodal quality: mmap 88-92% vs compress 55-65%).

## Rationale

R93 covers the core MinCut computational modules. R94 extends into the second tier — infrastructure and integration files that support the core. Module routing may reveal MoE (Mixture of Experts) or conditional computation patterns. The GNN files are strategically important: scheduler.rs and replay.rs determine whether the GNN crate's bimodal pattern (R91) resolves toward genuine or facade.

## Target: 9 files, ~4,570 LOC

---

### Cluster A: MinCut Infrastructure (7 files, ~3,535 LOC)

Module routing, model state, KV cache variants, and packet abstraction — the second tier of MinCut internals.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 1 | 3509 | `src/mod_routing.rs` | 537 | Module routing — MoE or conditional computation? |
| 2 | 3504 | `src/kv_cache/quantized_store.rs` | 523 | Quantized storage — production implementation? |
| 3 | 3495 | `src/kernel/quant4.rs` | 506 | 4-bit quantization kernel — GPTQ/AWQ style? |
| 4 | 3518 | `src/state.rs` | 501 | Model state management — checkpointing, serialization? |
| 5 | 3501 | `src/kv_cache/metrics.rs` | 495 | Cache metrics — real performance tracking? |
| 6 | 3511 | `src/packets.rs` | 492 | Packet abstraction — data movement, streaming? |
| 7 | 3483 | `src/attention/window.rs` | 481 | Window attention — sliding window or Longformer-style? |

**Full paths** (all under `~/repos/ruvector/crates/ruvector-mincut-gated-transformer/`):
1. `src/mod_routing.rs`
2. `src/kv_cache/quantized_store.rs`
3. `src/kernel/quant4.rs`
4. `src/state.rs`
5. `src/kv_cache/metrics.rs`
6. `src/packets.rs`
7. `src/attention/window.rs`

**Key questions**:
- `mod_routing.rs` (537 LOC): Real MoE routing with learned gating (top-k expert selection) or static dispatch? Load balancing loss?
- `quant4.rs` (506 LOC): GPTQ/AWQ-style 4-bit quantization with group-wise scaling? Or simple round-to-nearest?
- `state.rs` (501 LOC): Real checkpoint/resume with safetensors serialization or in-memory only?
- `window.rs` (481 LOC): Sliding window with global attention tokens (Longformer) or just truncated self-attention?

---

### Cluster B: GNN Training Components (2 files, ~1,035 LOC)

The two largest unread ruvector-gnn files. R91 showed bimodal quality (mmap.rs 88-92% vs compress.rs 55-65%). These determine the training subsystem quality.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 8 | 3163 | `src/scheduler.rs` | 532 | GNN training scheduler — real LR scheduling or placeholder? |
| 9 | 3162 | `src/replay.rs` | 503 | Experience replay — real prioritized replay buffer? |

**Full paths**:
8. `~/repos/ruvector/crates/ruvector-gnn/src/scheduler.rs`
9. `~/repos/ruvector/crates/ruvector-gnn/src/replay.rs`

**Key questions**:
- `scheduler.rs` (532 LOC): Real learning rate scheduling (cosine annealing, warmup, OneCycleLR) or constant LR? Does it integrate with the training loop?
- `replay.rs` (503 LOC): Prioritized experience replay (sum-tree, proportional sampling) or uniform random? TD-error priority updates?

---

## Expected Outcomes

1. MinCut infrastructure tier quality established (MoE routing, 4-bit quant, state management)
2. GNN bimodal quality resolved — scheduler and replay tilt genuine or facade
3. Window attention pattern identified (Longformer vs truncated)
4. DEEP count: ~1,357 → ~1,366
5. New findings: ~60-80 expected across 4,570 LOC

## DB Update Template

```javascript
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const sessionId = 94;
const today = new Date().toISOString().slice(0, 10);

// File IDs for this session:
// 3509: mod_routing.rs (537 LOC) — ruvector-rust, PROXIMATE
// 3504: quantized_store.rs (523 LOC) — ruvector-rust, PROXIMATE
// 3495: quant4.rs (506 LOC) — ruvector-rust, PROXIMATE
// 3518: state.rs (501 LOC) — ruvector-rust, PROXIMATE
// 3501: metrics.rs (495 LOC) — ruvector-rust, PROXIMATE
// 3511: packets.rs (492 LOC) — ruvector-rust, PROXIMATE
// 3483: window.rs (481 LOC) — ruvector-rust, PROXIMATE
// 3163: scheduler.rs (532 LOC) — ruvector-rust, PROXIMATE
// 3162: replay.rs (503 LOC) — ruvector-rust, PROXIMATE
```

## Domain Tags

- Files 3509, 3504, 3495, 3518, 3501, 3511, 3483 → `ruvector`, `v4-priority`
- Files 3163, 3162 → `ruvector`, `v4-priority`

## Isolation Check

All files are in ruvector-rust (36 total cross-deps, CONNECTED). No isolation concerns.

---

## Synthesis Doc Update Protocol (ADR-040)

**MANDATORY**: After all files are read and findings inserted into the DB, update the relevant `domains/*/analysis.md` files following the ADR-040 in-place protocol. Reference: `domains/memory-and-learning/analysis.md` for canonical structure.
