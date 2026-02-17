# R91 Execution Plan: v4-Priority Phase 1 — Mixed High-Value Targets

**Date**: 2026-02-17
**Session ID**: 91
**Focus**: Deep-read 9 files (~8,443 LOC) from the v4-priority queue. First session of the post-R90 "uncovered areas" sweep targeting files identified in README-REALITY-CHECK.md.
**Strategic value**: These are the largest unread files across 5 uncovered areas: AgentDB attention/ruvector integration, ruvector-gnn compression/memory-mapping, ruvector-mincut-gated-transformer (speculative decoding, RoPE, KV cache), SONA training pipeline, and neural model presets. All are PROXIMATE tier with 5-78 nearby DEEP files.

## Rationale

After R90 closed the original priority queue (90 sessions, 1,332 DEEP files), README-REALITY-CHECK.md identified 7 areas with significant unread code that could contain genuine implementations. A new `v4-priority` domain (HIGH) was created and 92 files (~38,975 LOC) were tagged for systematic coverage across R91-R100.

R91 takes the top 9 by LOC × tier_rank, covering the highest-value targets first.

## Target: 9 files, ~8,443 LOC

---

### Cluster A: AgentDB Integration (2 files, ~2,494 LOC)

These two files are in the agentic-flow monorepo's AgentDB package. RuVectorBackend.ts was already identified as a GENUINE-ASSETS file (88-92%). AttentionService.ts is the largest unread file and could contain real attention mechanisms or be another facade.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 1 | 12925 | `packages/agentdb/src/services/AttentionService.ts` | 1523 | What attention mechanisms? Real multi-head/cross-attention or cosmetic naming? |
| 2 | 12814 | `packages/agentdb/src/backends/ruvector/RuVectorBackend.ts` | 971 | Already in GENUINE-ASSETS (88-92%). Re-read for v4 extraction assessment |

**Full paths**:
1. `~/repos/agentic-flow/packages/agentdb/src/services/AttentionService.ts`
2. `~/repos/agentic-flow/packages/agentdb/src/backends/ruvector/RuVectorBackend.ts`

---

### Cluster B: ruvector-gnn (2 files, ~1,597 LOC)

GNN module — ruvector-gnn has 130 files but only 3 DEEP reads. These are the two largest unread files. R90 showed ruvector-core advanced features at 85-93% — does GNN match?

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 3 | 3159 | `crates/ruvector-gnn/src/mmap.rs` | 918 | Memory-mapped graph storage? Real mmap or abstracted Vec? |
| 4 | 3154 | `crates/ruvector-gnn/src/compress.rs` | 679 | Graph compression — real sparse encoding or placeholder? |

**Full paths**:
3. `~/repos/ruvector/crates/ruvector-gnn/src/mmap.rs`
4. `~/repos/ruvector/crates/ruvector-gnn/src/compress.rs`

---

### Cluster C: ruvector-mincut-gated-transformer (3 files, ~2,338 LOC)

The MinCut-Gated-Transformer crate was rated "MOST NOVEL" in R34. Flash attention is in this crate. These 3 files cover speculative decoding, rotary position embeddings, and KV cache management.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 5 | 3516 | `src/speculative.rs` | 788 | Speculative decoding — real draft-verify or placeholder? |
| 6 | 3513 | `src/rope.rs` | 777 | RoPE — correct complex exponential rotation or simplified? |
| 7 | 3499 | `src/kv_cache/legacy.rs` | 773 | KV cache — real cache with eviction or growing Vec? |

**Full paths** (all under `~/repos/ruvector/crates/ruvector-mincut-gated-transformer/`):
5. `src/speculative.rs`
6. `src/rope.rs`
7. `src/kv_cache/legacy.rs`

---

### Cluster D: SONA Training Pipeline (1 file, 708 LOC)

SONA is 85% production-ready per memory. Only 3 DEEP reads out of 156 files. This is the training pipeline — likely the most integration-critical file in the crate.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 8 | 4475 | `crates/sona/src/training/pipeline.rs` | 708 | Training pipeline — full forward/backward/optimizer or configuration shell? |

**Full path**:
8. `~/repos/ruvector/crates/sona/src/training/pipeline.rs`

---

### Cluster E: Neural Presets (1 file, 1,306 LOC)

ruv-swarm neural model presets. R40 found JS neural models = inference works, training facade. This is the "complete" presets file — does it define real architectures or cosmetic configs?

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 9 | 9650 | `ruv-swarm/npm/src/neural-models/neural-presets-complete.js` | 1306 | Complete neural presets — real architecture definitions or config objects? |

**Full path**:
9. `~/repos/ruv-FANN/ruv-swarm/npm/src/neural-models/neural-presets-complete.js`

---

## Expected Outcomes

1. AgentDB attention quality assessed — genuine ML or naming facade?
2. RuVectorBackend re-read with v4 extraction lens
3. ruvector-gnn compression/mmap quality characterized
4. MinCut transformer modules (speculative/RoPE/KV-cache) quality assessed
5. SONA training pipeline quality characterized
6. Neural presets assessed for genuine architecture definitions
7. DEEP count: 1,332 → ~1,341
8. New findings: ~60-80 expected across 8,443 LOC

## DB Update Template

```javascript
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const sessionId = 91;
const today = new Date().toISOString().slice(0, 10);

// File IDs for this session:
// 12925: AttentionService.ts (1523 LOC) — agentic-flow-rust, MEDIUM→DEEP
// 12814: RuVectorBackend.ts (971 LOC) — agentic-flow-rust, MEDIUM→DEEP
// 3159: mmap.rs (918 LOC) — ruvector-rust, MEDIUM→DEEP
// 3154: compress.rs (679 LOC) — ruvector-rust, MEDIUM→DEEP
// 3516: speculative.rs (788 LOC) — ruvector-rust, NOT_TOUCHED→DEEP
// 3513: rope.rs (777 LOC) — ruvector-rust, NOT_TOUCHED→DEEP
// 3499: kv_cache/legacy.rs (773 LOC) — ruvector-rust, NOT_TOUCHED→DEEP
// 4475: pipeline.rs (708 LOC) — ruvector-rust, MEDIUM→DEEP
// 9650: neural-presets-complete.js (1306 LOC) — ruv-fann-rust, MEDIUM→DEEP
```

## Domain Tags

- Files 12925, 12814 → `agentdb-integration`, `memory-and-learning`, `v4-priority`
- Files 3159, 3154 → `ruvector`, `v4-priority`
- Files 3516, 3513, 3499 → `ruvector`, `v4-priority`
- File 4475 → `memory-and-learning`, `v4-priority`
- File 9650 → `swarm-coordination`, `v4-priority`

## Synthesis Doc Update Protocol (ADR-040)

**MANDATORY**: After all files are read and findings inserted into the DB, update the relevant `domains/*/analysis.md` files following the ADR-040 in-place protocol.
