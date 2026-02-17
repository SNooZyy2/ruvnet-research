# R95 Execution Plan: SONA Crate Deep Dive

**Date**: {YYYY-MM-DD}
**Session ID**: 95
**Focus**: Deep-read 9 SONA crate files (~4,129 LOC) — export pipeline, training internals, NAPI binding, and loop variants.
**Strategic value**: SONA is 85% production-ready with only 4/156 files DEEP. The export pipeline (pretrain.rs, huggingface_hub.rs) is the biggest gap — these determine whether SONA can actually export trained models. R91 found pipeline.rs at 82-88% with a validation facade (echoes test labels). These files complete the training and export subsystems.

## Rationale

SONA has been characterized as 85% production-ready (MicroLoRA 92-95%, EWC++, federated, SafeTensors export), but only 4 of 156 files have DEEP reads. R91's pipeline.rs (82-88%) revealed a validation facade — the export pipeline files may have similar patterns. The training templates and factory files determine whether SONA's training system is configurable or hardcoded. The NAPI binding determines JS interop quality.

## Target: 9 files, ~4,129 LOC

---

### Cluster A: Export Pipeline (4 files, ~1,957 LOC)

The biggest gap in SONA characterization. These files handle model export to SafeTensors and HuggingFace Hub formats.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 1 | 4458 | `src/export/pretrain.rs` | 668 | Pretrain export — SafeTensors? HuggingFace format? |
| 2 | 4456 | `src/export/huggingface_hub.rs` | 486 | HuggingFace Hub export — real API client or scaffold? |
| 3 | 4455 | `src/export/dataset.rs` | 408 | Dataset export — data serialization format? |
| 4 | 4457 | `src/export/mod.rs` | 395 | Export module root — what does it coordinate? |

**Full paths** (all under `~/repos/ruvector/crates/sona/`):
1. `src/export/pretrain.rs`
2. `src/export/huggingface_hub.rs`
3. `src/export/dataset.rs`
4. `src/export/mod.rs`

**Key questions**:
- `pretrain.rs` (668 LOC): Does it produce valid SafeTensors files with correct metadata? Or just serializes to custom format?
- `huggingface_hub.rs` (486 LOC): Real HTTP client for Hub API (model upload, repo creation) or scaffold with todo!()?
- `dataset.rs` (408 LOC): Real dataset serialization (Arrow, Parquet, JSON) or custom binary format?
- `mod.rs` (395 LOC): Coordinates export stages or just re-exports? Does it enforce export ordering?

---

### Cluster B: Training Internals (3 files, ~1,637 LOC)

Training templates, factory, and metrics — the configuration layer above pipeline.rs.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 5 | 4476 | `src/training/templates.rs` | 662 | Training templates — predefined configs for model types? |
| 6 | 4471 | `src/training/factory.rs` | 508 | Training factory — builder pattern for pipeline? |
| 7 | 4473 | `src/training/metrics.rs` | 467 | Training metrics — loss tracking, learning curves? |

**Full paths**:
5. `src/training/templates.rs`
6. `src/training/factory.rs`
7. `src/training/metrics.rs`

**Key questions**:
- `templates.rs` (662 LOC): Real predefined training configs (LLaMA, GPT, BERT) with correct hyperparameters? Or generic placeholders?
- `factory.rs` (508 LOC): Builder pattern that correctly wires optimizer + scheduler + pipeline? Or just struct construction?
- `metrics.rs` (467 LOC): Real loss tracking with EMA smoothing, gradient norms, learning curves? Or just print statements?

---

### Cluster C: NAPI + Loop Variant (2 files, ~535 LOC)

JS interop and a loop variant.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 8 | 4468 | `src/napi_simple.rs` | 287 | NAPI binding — bridges Rust SONA to JS? |
| 9 | 4463 | `src/loops/instant.rs` | 248 | Instant loop — single-shot inference? |

**Full paths**:
8. `src/napi_simple.rs`
9. `src/loops/instant.rs`

**Key questions**:
- `napi_simple.rs` (287 LOC): Real NAPI bridge with proper type conversion (Buffer ↔ Vec<f32>) or stub? Does it expose training or just inference?
- `instant.rs` (248 LOC): Single-shot inference loop or degenerate training loop? How does it relate to the loop coordinator?

---

## Expected Outcomes

1. SONA export pipeline quality characterized — can it actually export models?
2. HuggingFace Hub integration assessed — real API client or scaffold
3. Training subsystem completeness established (templates + factory + metrics)
4. NAPI JS bridge quality assessed
5. DEEP count: ~1,366 → ~1,375
6. New findings: ~50-70 expected across 4,129 LOC

## DB Update Template

```javascript
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const sessionId = 95;
const today = new Date().toISOString().slice(0, 10);

// File IDs for this session:
// 4458: pretrain.rs (668 LOC) — ruvector-rust, PROXIMATE
// 4456: huggingface_hub.rs (486 LOC) — ruvector-rust, PROXIMATE
// 4455: dataset.rs (408 LOC) — ruvector-rust, PROXIMATE
// 4457: mod.rs (395 LOC) — ruvector-rust, PROXIMATE
// 4476: templates.rs (662 LOC) — ruvector-rust, PROXIMATE
// 4471: factory.rs (508 LOC) — ruvector-rust, PROXIMATE
// 4473: metrics.rs (467 LOC) — ruvector-rust, PROXIMATE
// 4468: napi_simple.rs (287 LOC) — ruvector-rust, PROXIMATE
// 4463: instant.rs (248 LOC) — ruvector-rust, PROXIMATE
```

## Domain Tags

- All files → `memory-and-learning`, `v4-priority`

## Isolation Check

All files are in ruvector-rust (36 total cross-deps, CONNECTED). No isolation concerns.

---

## Synthesis Doc Update Protocol (ADR-040)

**MANDATORY**: After all files are read and findings inserted into the DB, update the relevant `domains/*/analysis.md` files following the ADR-040 in-place protocol. Reference: `domains/memory-and-learning/analysis.md` for canonical structure.
