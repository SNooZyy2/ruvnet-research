# R93 Execution Plan: MinCut-Transformer Core

**Date**: {YYYY-MM-DD}
**Session ID**: 93
**Focus**: Deep-read 9 MinCut-Gated-Transformer core files (~5,421 LOC) — FFN, quantization, KV cache manager, attention variants.
**Strategic value**: MinCut-Gated-Transformer is rated "MOST NOVEL" crate (R34). R91 confirmed 85-92% quality for speculative.rs, rope.rs, kv_cache/legacy.rs. These 9 files are the highest-LOC unread internals — FFN gating, quantized GEMM, spike-driven attention, and energy-based gating. Completing these establishes the quality baseline for the remaining 24 MinCut files.

## Rationale

After R91 established the MinCut-Gated-Transformer quality range (speculative.rs 88-92%, rope.rs 88-92%, kv_cache/legacy.rs 82-88%), R93 extends into the core computational modules. These are all PROXIMATE tier with high nearby-DEEP counts from R91's reads.

The files span four functional areas: attention variants (spike-driven, sparse), quantization (Q1.15, quantized GEMM), KV cache management, and the core FFN + energy gate. The energy_gate.rs is particularly interesting — it may connect to the mincut lambda signal that makes this crate novel.

## Target: 9 files, ~5,421 LOC

---

### Cluster A: Attention Variants (2 files, ~1,156 LOC)

Novel attention mechanisms — spike-driven (neuromorphic) and sparse (Longformer/BigBird style). These extend the attention arc from R91's speculative.rs.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 1 | 3482 | `src/attention/spike_driven.rs` | 585 | Spike-driven attention — neuromorphic computing or cosmetic? |
| 2 | 3514 | `src/sparse_attention.rs` | 571 | Sparse attention — Longformer/BigBird style? Real block-sparse masking? |

**Full paths** (all under `~/repos/ruvector/crates/ruvector-mincut-gated-transformer/`):
1. `src/attention/spike_driven.rs`
2. `src/sparse_attention.rs`

**Key questions**:
- `spike_driven.rs` (585 LOC): Is this genuine neuromorphic spike-timing attention or renamed softmax? Does it use binary spike trains or continuous activations? Any connection to energy_gate.rs?
- `sparse_attention.rs` (571 LOC): Real block-sparse masking with local+global attention patterns? Or just masked softmax with zeros?

---

### Cluster B: Quantization (2 files, ~1,255 LOC)

Fixed-point and quantized matrix multiply — the performance-critical kernels.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 3 | 3512 | `src/q15.rs` | 634 | Q1.15 fixed-point quantization — matches rope.rs Q15 path? Full quantized inference? |
| 4 | 3494 | `src/kernel/qgemm.rs` | 621 | Quantized GEMM kernel — real matmul or scaffold? SIMD? |

**Full paths**:
3. `src/q15.rs`
4. `src/kernel/qgemm.rs`

**Key questions**:
- `q15.rs` (634 LOC): Does Q1.15 format match the Q15 quantization path in rope.rs (R91)? Full quantized inference pipeline or just format conversion?
- `qgemm.rs` (621 LOC): Real quantized matrix multiplication with SIMD (AVX2/NEON) or scalar loops? Does it support asymmetric quantization?

---

### Cluster C: KV Cache + FFN (3 files, ~1,784 LOC)

Core inference components — the feed-forward network and KV cache manager that R91's legacy.rs was missing eviction policies for.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 5 | 3500 | `src/kv_cache/manager.rs` | 596 | KV cache manager — eviction policies missing in legacy.rs, present here? |
| 6 | 3488 | `src/ffn.rs` | 628 | Feed-forward network — SwiGLU/GeGLU gating? Real activation functions? |
| 7 | 3498 | `src/kv_cache/kvquant.rs` | 565 | KV quantization — advanced vs legacy.rs baseline? Per-token vs per-head? |

**Full paths**:
5. `src/kv_cache/manager.rs`
6. `src/ffn.rs`
7. `src/kv_cache/kvquant.rs`

**Key questions**:
- `manager.rs` (596 LOC): Does it implement real eviction (H2O, attention-sink, StreamingLLM) or just append-only growth? How does it coordinate with legacy.rs?
- `ffn.rs` (628 LOC): SwiGLU/GeGLU gating with real activation functions? Or just ReLU? Does it support MoE routing?
- `kvquant.rs` (565 LOC): Per-token or per-head quantization? Does it implement KIVI or its own scheme?

---

### Cluster D: Energy Gate + Early Exit (2 files, ~1,221 LOC)

The novel components — energy-based gating may connect to the mincut lambda signal, and early exit enables adaptive computation.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 8 | 3486 | `src/energy_gate.rs` | 560 | Energy-based gating — mincut lambda connection? |
| 9 | 3485 | `src/early_exit.rs` | 661 | Early exit / adaptive computation — real confidence threshold or placeholder? |

**Full paths**:
8. `src/energy_gate.rs`
9. `src/early_exit.rs`

**Key questions**:
- `energy_gate.rs` (560 LOC): How does energy-based gating connect to the mincut lambda? Is this the mechanism that makes the crate "most novel"? Real energy function or renamed threshold?
- `early_exit.rs` (661 LOC): Real adaptive computation with learned confidence thresholds? Or just a fixed-layer cutoff?

---

## Expected Outcomes

1. MinCut-Gated-Transformer core quality characterized across 9 files
2. Energy gate → mincut lambda connection verified or debunked
3. Quantization pipeline quality assessed (Q15 + GEMM)
4. KV cache management completeness established (manager vs legacy)
5. DEEP count: ~1,348 → ~1,357
6. New findings: ~70-90 expected across 5,421 LOC

## DB Update Template

```javascript
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const sessionId = 93;
const today = new Date().toISOString().slice(0, 10);

// File IDs for this session:
// 3482: spike_driven.rs (585 LOC) — ruvector-rust, PROXIMATE
// 3514: sparse_attention.rs (571 LOC) — ruvector-rust, PROXIMATE
// 3512: q15.rs (634 LOC) — ruvector-rust, PROXIMATE
// 3494: qgemm.rs (621 LOC) — ruvector-rust, PROXIMATE
// 3500: manager.rs (596 LOC) — ruvector-rust, PROXIMATE
// 3488: ffn.rs (628 LOC) — ruvector-rust, PROXIMATE
// 3498: kvquant.rs (565 LOC) — ruvector-rust, PROXIMATE
// 3486: energy_gate.rs (560 LOC) — ruvector-rust, PROXIMATE
// 3485: early_exit.rs (661 LOC) — ruvector-rust, PROXIMATE
```

## Domain Tags

- All files → `ruvector`, `v4-priority` (already tagged)

## Isolation Check

All files are in ruvector-rust (36 total cross-deps, CONNECTED). No isolation concerns.

---

## Synthesis Doc Update Protocol (ADR-040)

**MANDATORY**: After all files are read and findings inserted into the DB, update the relevant `domains/*/analysis.md` files following the ADR-040 in-place protocol. Reference: `domains/memory-and-learning/analysis.md` for canonical structure.
