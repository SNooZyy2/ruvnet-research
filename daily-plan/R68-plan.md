# R68 Execution Plan: ReasoningBank Core Internals + Temporal-Compare MLP Family

**Date**: 2026-02-16
**Session ID**: 68
**Focus**: ReasoningBank Rust core implementation (pattern matching, similarity search, engine orchestration, async storage, DB migrations) + temporal-compare MLP variant family (classifier, ultra, optimized, AVX-512, crate entry point)
**Parallel with**: R69 (no file overlap -- R68 = agentic-flow-rust reasoningbank/ internals + sublinear-rust crates/temporal-compare/ MLP variants; R69 = sublinear-rust src/temporal_nexus/quantum/ + graph_reasoner/ internals + ruv-fann-rust ruv-swarm/ crates)

## IMPORTANT: Parallel Execution Notice

This plan runs IN PARALLEL with R69. The file lists are strictly non-overlapping:
- **R68 covers**: agentic-flow-rust reasoningbank/crates/ (5 Rust files: core pattern/similarity/engine + storage async_wrapper/migrations), sublinear-rust crates/temporal-compare/src/ (5 Rust files: mlp_classifier/mlp_ultra/mlp_optimized/mlp_avx512/main)
- **R69 covers**: sublinear-rust src/temporal_nexus/quantum/ (3 Rust files), sublinear-rust crates/psycho-symbolic-reasoner/graph_reasoner/src/ (3 files), ruv-fann-rust ruv-swarm/ crates (4 files)
- **ZERO shared files** between R68 and R69
- **R68 has NO src/temporal_nexus/, graph_reasoner/, or ruv-fann-rust files**
- **R69 has NO reasoningbank/ or temporal-compare/ files**
- Do NOT read or analyze any file from R69's list (see R69-plan.md for that list)

## Rationale

- **ReasoningBank core internals are the implementation behind R67's crate roots**: R67 read the 4 crate root lib.rs files (core 88-92%, storage 94%, learning 95-98%, mcp 93-95%) and found genuinely architected traits and types. pattern.rs (253 LOC), similarity.rs (241 LOC), and engine.rs (233 LOC) are the IMPLEMENTATION behind those traits. R67 found the Rust workspace is a 5th disconnected data layer with own types/MCP/network — do these internals reveal integration points missed by the crate roots?
- **ReasoningBank storage internals extend R67's 94% verdict**: async_wrapper.rs (175 LOC) wraps the storage in tokio async, migrations.rs (170 LOC) defines the schema evolution. R57 found queries.ts has a 7-table schema — do the Rust migrations define the same schema? Does async_wrapper use proper tokio patterns matching the genuine quinn QUIC from R67?
- **Temporal-compare MLP family extends R66's bimodal verdict**: R66 found mlp.rs 62% BIMODAL (inference genuine, training broken with arbitrary gradient scaling) and ensemble.rs 85-90% GENUINE. The crate has 4 more MLP variants: mlp_classifier.rs (325 LOC), mlp_ultra.rs (314 LOC), mlp_optimized.rs (246 LOC), mlp_avx512.rs (341 LOC). Do the variants fix the broken backprop, or replicate the same bug? Does mlp_avx512.rs have real AVX-512 SIMD like ruvector-core?
- **temporal-compare main.rs is the crate entry point**: At 299 LOC, main.rs orchestrates ensemble + MLP variants. Does it wire the MLP variants into the ensemble (AdaBoost/bagging)? Does it implement a training pipeline or inference-only demo?

## Target: 10 files, ~2,587 LOC

---

### Cluster A: ReasoningBank Core Implementation (3 files, ~724 LOC)

The implementation files behind the core crate's traits. R67 found core/lib.rs defines Trajectory, Verdict, Pattern types with serde + traits. These files implement the actual logic.

| # | File ID | File | LOC | Domain | Package |
|---|---------|------|-----|--------|---------|
| 1 | 13440 | `reasoningbank/crates/reasoningbank-core/src/pattern.rs` | 253 | memory-and-learning | agentic-flow-rust |
| 2 | 13441 | `reasoningbank/crates/reasoningbank-core/src/similarity.rs` | 241 | memory-and-learning | agentic-flow-rust |
| 3 | 13438 | `reasoningbank/crates/reasoningbank-core/src/engine.rs` | 233 | memory-and-learning | agentic-flow-rust |

**Full paths**:
1. `~/repos/agentic-flow/reasoningbank/crates/reasoningbank-core/src/pattern.rs`
2. `~/repos/agentic-flow/reasoningbank/crates/reasoningbank-core/src/similarity.rs`
3. `~/repos/agentic-flow/reasoningbank/crates/reasoningbank-core/src/engine.rs`

**Key questions**:
- `pattern.rs`: How does ReasoningBank extract and match patterns?
  - Does it implement the trajectory → verdict → pattern learning pipeline?
  - R67 found learning/lib.rs 95-98% with PatternExtractor and ReinforcementLearner traits — does pattern.rs implement these?
  - Does it use real ML techniques (clustering, embeddings) or heuristic matching?
  - Does it define Pattern types with confidence scores, frequency tracking, decay?
  - At 253 LOC, is there enough for genuine pattern extraction vs simple string matching?
- `similarity.rs`: How does ReasoningBank compute similarity?
  - R20 ROOT CAUSE: hash-based embeddings break similarity search. Does the Rust version use real embeddings?
  - R65 found embedding-adapter.ts is the smoking gun (SHA-256→hash). Does similarity.rs avoid this?
  - Does it implement cosine similarity, Jaccard, or more sophisticated measures?
  - Does it integrate with the HNSW infrastructure (ruvector-core)?
  - This is a CRITICAL file for the R20 root cause arc — does Rust solve what TS broke?
- `engine.rs`: How does the ReasoningBank engine orchestrate?
  - Does it compose pattern extraction + similarity search + storage into a coherent pipeline?
  - Does it implement the trajectory lifecycle (start → step → end → judge → distill)?
  - R67 found the crate roots define clean trait boundaries — does engine.rs implement them?
  - Does it use dependency injection (trait objects) or concrete implementations?
  - Is there a connection to the TS ReasoningBank (queries.ts at 85-90%) or completely independent?

---

### Cluster B: ReasoningBank Storage Internals (2 files, ~343 LOC)

The storage implementation files. R67 found storage/lib.rs 94% with SQLite via rusqlite, CRUD operations, and proper error handling. These files extend that with async wrappers and schema migrations.

| # | File ID | File | LOC | Domain | Package |
|---|---------|------|-----|--------|---------|
| 4 | 13476 | `reasoningbank/crates/reasoningbank-storage/src/async_wrapper.rs` | 175 | memory-and-learning | agentic-flow-rust |
| 5 | 13479 | `reasoningbank/crates/reasoningbank-storage/src/migrations.rs` | 170 | memory-and-learning | agentic-flow-rust |

**Full paths**:
4. `~/repos/agentic-flow/reasoningbank/crates/reasoningbank-storage/src/async_wrapper.rs`
5. `~/repos/agentic-flow/reasoningbank/crates/reasoningbank-storage/src/migrations.rs`

**Key questions**:
- `async_wrapper.rs`: How does the storage handle async?
  - Does it use tokio::task::spawn_blocking for SQLite operations (correct pattern)?
  - Or does it attempt async SQLite directly (incorrect — rusqlite is sync)?
  - R67 found genuine tokio usage in the network crate (quic.rs 96%) — does storage match?
  - Does it implement connection pooling or single-connection with mutex?
  - Does it expose the same CRUD API as the sync storage but with async signatures?
- `migrations.rs`: What schema does ReasoningBank Rust define?
  - R57 found queries.ts defines a 7-table schema (trajectories, verdicts, patterns, steps, metadata, configs, sessions)
  - Does the Rust migration define the same 7 tables, or a different schema?
  - If different: confirms the 5th disconnected data layer verdict from R67
  - If same: there may be more integration than R67 concluded
  - Does it use proper migration versioning (up/down, version numbers)?
  - Does it define vector columns for embeddings, or text-only storage?

---

### Cluster C: Temporal-Compare MLP Variant Family (5 files, ~1,520 LOC)

The MLP variant family in temporal-compare. R66 found mlp.rs 62% BIMODAL (inference genuine but training has arbitrary gradient scaling). R66 also found ensemble.rs 85-90% GENUINE (AdaBoost + bagging). These 5 files are the remaining MLP variants plus the crate entry point.

| # | File ID | File | LOC | Domain | Package |
|---|---------|------|-----|--------|---------|
| 6 | 14155 | `crates/temporal-compare/src/mlp_classifier.rs` | 325 | memory-and-learning | sublinear-rust |
| 7 | 14158 | `crates/temporal-compare/src/mlp_ultra.rs` | 314 | memory-and-learning | sublinear-rust |
| 8 | 14156 | `crates/temporal-compare/src/mlp_optimized.rs` | 246 | memory-and-learning | sublinear-rust |
| 9 | 14154 | `crates/temporal-compare/src/mlp_avx512.rs` | 341 | memory-and-learning | sublinear-rust |
| 10 | 14151 | `crates/temporal-compare/src/main.rs` | 299 | memory-and-learning | sublinear-rust |

**Full paths**:
6. `~/repos/sublinear-time-solver/crates/temporal-compare/src/mlp_classifier.rs`
7. `~/repos/sublinear-time-solver/crates/temporal-compare/src/mlp_ultra.rs`
8. `~/repos/sublinear-time-solver/crates/temporal-compare/src/mlp_optimized.rs`
9. `~/repos/sublinear-time-solver/crates/temporal-compare/src/mlp_avx512.rs`
10. `~/repos/sublinear-time-solver/crates/temporal-compare/src/main.rs`

**Key questions**:
- `mlp_classifier.rs`: Is this a classification-specific MLP?
  - Does it add softmax output + cross-entropy loss on top of the base MLP?
  - Does it fix the broken backpropagation from mlp.rs (arbitrary 0.01 gradient scaling)?
  - Does it handle multi-class classification (confusion matrix, precision/recall)?
  - R66 found the base mlp.rs has genuine forward pass (ReLU/sigmoid/tanh) — does the classifier reuse or reimplement?
- `mlp_ultra.rs`: What makes this "ultra"?
  - Is it a performance-optimized variant (SIMD, memory layout, batch processing)?
  - Or does it add more layers/features (dropout, batch normalization, residual connections)?
  - Does it replicate the broken training or implement genuine backpropagation?
  - Does it import from mlp.rs (shared base) or reimplement from scratch?
- `mlp_optimized.rs`: What optimization strategy?
  - Does it implement training optimizers (Adam, SGD with momentum, RMSProp)?
  - Or is it about inference optimization (quantization, pruning)?
  - R37 found temporal-tensor HIGHEST QUALITY (93%) — does the optimized MLP match?
  - How does "optimized" differ from "ultra"?
- `mlp_avx512.rs`: Is this real AVX-512 SIMD?
  - ruvector-core has REAL SIMD (AVX-512/AVX2/NEON) — does this file use the same approach?
  - Does it use `#[cfg(target_feature = "avx512f")]` and std::arch intrinsics?
  - Or is it "AVX-512" in name only (like the fake SIMD in simd-vector-ops.ts)?
  - R37 found micro_lora.rs 92-95% with genuine NEON SIMD — does mlp_avx512.rs match?
  - At 341 LOC, is there enough for real SIMD matmul + activation functions?
- `main.rs`: What does the temporal-compare binary do?
  - Does it wire ensemble + MLP variants into a training/evaluation pipeline?
  - Does it define benchmark datasets (synthetic or real)?
  - Does it implement temporal comparison logic (time series, before/after)?
  - Is it a runnable binary or mostly argument parsing + stubs?
  - Does it demonstrate the training bug from mlp.rs or work around it?

---

## Expected Outcomes

1. **ReasoningBank core quality**: Do the implementation files match the 88-95% quality of R67's crate roots?
2. **R20 root cause in Rust**: Does similarity.rs use real embeddings or hash-based? CRITICAL for the R20 arc
3. **5th data layer evidence**: Do migrations.rs and engine.rs confirm or refute the disconnected 5th layer?
4. **MLP training bug propagation**: Does the broken backprop from mlp.rs replicate across all variants?
5. **Real AVX-512**: Is mlp_avx512.rs genuine SIMD or theatrical (like simd-vector-ops.ts)?
6. **temporal-compare crate architecture**: Does main.rs show a coherent training pipeline or demo-only?

## DB Update Template

```javascript
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const sessionId = 68;
const today = new Date().toISOString().slice(0, 10);

// File IDs for this session:
// 13440: reasoningbank-core/pattern.rs
// 13441: reasoningbank-core/similarity.rs
// 13438: reasoningbank-core/engine.rs
// 13476: reasoningbank-storage/async_wrapper.rs
// 13479: reasoningbank-storage/migrations.rs
// 14155: temporal-compare/mlp_classifier.rs
// 14158: temporal-compare/mlp_ultra.rs
// 14156: temporal-compare/mlp_optimized.rs
// 14154: temporal-compare/mlp_avx512.rs
// 14151: temporal-compare/main.rs
```

## Domain Tags

- All ReasoningBank files → `memory-and-learning` domain (already tagged)
- All temporal-compare files → `memory-and-learning` domain (already tagged)
