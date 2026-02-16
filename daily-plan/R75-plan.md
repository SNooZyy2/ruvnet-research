# R75 Execution Plan: Clear CONNECTED + temporal-compare Deep Sweep + Crate Closures

**Date**: 2026-02-16
**Session ID**: 75
**Focus**: Clear the final 3 CONNECTED-tier ReasoningBank TS files, deep-sweep 5 of 11 remaining temporal-compare source files, complete temporal-lead-solver crate src, and read the sublinear crate-root WASM bindings.
**Strategic value**: CONNECTED tier CLEARED (milestone — 0 remaining after R73's reduction to 3). temporal-compare pushed to 13/17 source files DEEP (76%). temporal-lead-solver src 100% COMPLETE (8/8). WASM arc extended with crate-root wasm_bindings.rs.

## Rationale

R74 (currently executing) covers the reasoningbank-network Rust crate (7 files) and 3 ReasoningBank TS inner-ring files. It does NOT touch the 3 remaining CONNECTED-tier files (judge.ts, pii-scrubber.ts, retrieve.ts), which are all ReasoningBank TS source files whose `.js` counterparts were already deep-read in R73 (judge.js at 54%, pii-scrubber.js at 88%). The `.ts` source files likely contain type annotations, additional logic, or import patterns that the compiled `.js` files obscure. Clearing these 3 files achieves the CONNECTED tier milestone — every file with a direct dependency edge to a DEEP file will have been read.

With CONNECTED cleared, the session turns to two crate-completion opportunities. **temporal-compare** has 8 DEEP source files (R68 found TRIMODAL MLP: mlp_optimized 92-96%, mlp_avx512 REAL AVX-512, mlp 62% bimodal) but 11 untouched source files. The 5 selected here — quantization.rs, reservoir.rs, mlp_quantized.rs, attention.rs, fourier.rs — are the highest-LOC remaining and extend the ML/SIMD arc. **temporal-lead-solver** has 7/8 source files DEEP (R70-R71); only cli.rs remains, which would complete the crate's source directory. Finally, **wasm_bindings.rs** is the sublinear-time-solver crate-root WASM binding file — it exposes the entire crate to WASM consumers and extends the WASM scoreboard (currently 10:5 genuine:theatrical).

## Target: 10 files, ~2,342 LOC

---

### Cluster A: Clear CONNECTED Tier (3 files, ~431 LOC)

The last 3 CONNECTED-tier files. All are ReasoningBank TS source files in agentic-flow whose `.js` counterparts are already DEEP. Clearing these achieves a project milestone: every file with a direct dependency to a DEEP file has been deep-read.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 1 | 10813 | `agentic-flow/src/reasoningbank/core/judge.ts` | 178 | TS source of judge.js (R73: 54%). CONNECTED — dep from DEEP files |
| 2 | 10845 | `agentic-flow/src/reasoningbank/utils/pii-scrubber.ts` | 131 | TS source of pii-scrubber.js (R73: 88%). CONNECTED — dep from DEEP files |
| 3 | 10817 | `agentic-flow/src/reasoningbank/core/retrieve.ts` | 122 | ReasoningBank retrieval API. CONNECTED — dep from DEEP files |

**Full paths**:
1. `~/repos/agentic-flow/agentic-flow/src/reasoningbank/core/judge.ts`
2. `~/repos/agentic-flow/agentic-flow/src/reasoningbank/utils/pii-scrubber.ts`
3. `~/repos/agentic-flow/agentic-flow/src/reasoningbank/core/retrieve.ts`

**Key questions**:
- `judge.ts` (178 LOC): R73 found judge.js at 54% — genuine LLM-as-Judge but BYPASSES MCP store, creating parallel data flow. Does the TS source reveal additional logic, stricter types, or the MCP bypass more clearly? Is the LLM prompt template hardcoded or configurable? Does it import from the barrel (index.ts) or directly from queries.js?
- `pii-scrubber.ts` (131 LOC): R73 found pii-scrubber.js at 88% with 14 regex patterns. Does the TS version add type safety, additional patterns, or validation that the JS version lacks? Is it imported by the judge pipeline (PII scrub before storing verdicts)?
- `retrieve.ts` (122 LOC): Core retrieval operation — does it use vector search (embeddings) or key-value lookup? Does it go through the barrel, HybridBackend, or bypass to queries.js directly? Is there any caching or deduplication?

---

### Cluster B: temporal-compare Crate Sweep (5 files, ~1,280 LOC)

Extends the R68 MLP/SIMD arc. temporal-compare currently has 8/17 source files DEEP. These 5 are the highest-LOC remaining source files. After this sweep, the crate reaches 13/17 source files DEEP (76%), with only 6 tiny files remaining (250 LOC total).

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 4 | 14159 | `crates/temporal-compare/src/quantization.rs` | 323 | Vector quantization — PQ, scalar, or binary? Memory reduction for embeddings? |
| 5 | 14160 | `crates/temporal-compare/src/reservoir.rs` | 311 | Reservoir computing/sampling — online learning? Streaming data? |
| 6 | 14157 | `crates/temporal-compare/src/mlp_quantized.rs` | 267 | Quantized MLP — extends R68 TRIMODAL (mlp_optimized 92-96%, mlp_avx512 REAL) |
| 7 | 14145 | `crates/temporal-compare/src/attention.rs` | 198 | Attention mechanism — Flash? Multi-head? Compare to R51 attention-fallbacks |
| 8 | 14149 | `crates/temporal-compare/src/fourier.rs` | 181 | Fourier/spectral methods — FFT for temporal signal processing? |

**Full paths**:
4. `~/repos/sublinear-time-solver/crates/temporal-compare/src/quantization.rs`
5. `~/repos/sublinear-time-solver/crates/temporal-compare/src/reservoir.rs`
6. `~/repos/sublinear-time-solver/crates/temporal-compare/src/mlp_quantized.rs`
7. `~/repos/sublinear-time-solver/crates/temporal-compare/src/attention.rs`
8. `~/repos/sublinear-time-solver/crates/temporal-compare/src/fourier.rs`

**Key questions**:
- `quantization.rs` (323 LOC): Product quantization, scalar quantization, or binary? Does it quantize model weights (INT8/INT4) or data embeddings? How does it interact with mlp_quantized.rs? Is there SIMD acceleration (R68 found real AVX-512 in mlp_avx512)?
- `reservoir.rs` (311 LOC): Reservoir computing (echo state networks) or reservoir sampling (streaming statistics)? If computing, does it implement genuine ESN with random fixed weights? At 311 LOC either is plausible.
- `mlp_quantized.rs` (267 LOC): R68 found mlp_optimized at 92-96% (BEST MLP) and mlp_avx512 with REAL `_mm512_*` intrinsics. Where does the quantized variant fall on that spectrum? Does it use quantization.rs? INT8 matmul or simulated?
- `attention.rs` (198 LOC): Self-attention, cross-attention, or linear attention? At 198 LOC likely a focused implementation. How does it compare to R51's attention-fallbacks.ts (72-88%, 5 genuine implementations)?
- `fourier.rs` (181 LOC): FFT for temporal signal decomposition? Does it use a library (rustfft) or roll its own? Application to temporal comparison or standalone utility?

---

### Cluster C: Crate Completion + Sublinear WASM Root (2 files, ~631 LOC)

Two high-value files that close arcs: cli.rs completes the temporal-lead-solver crate source directory (8/8), and wasm_bindings.rs exposes the entire sublinear-time-solver crate to WASM consumers.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 9 | 14170 | `crates/temporal-lead-solver/src/bin/cli.rs` | 298 | CLI binary — COMPLETES crate src (7/8 → 8/8). Uses genuine O(√n) predictor? |
| 10 | 14505 | `src/wasm_bindings.rs` | 333 | Crate-root WASM bindings — what does the sublinear solver expose to WASM? |

**Full paths**:
9. `~/repos/sublinear-time-solver/crates/temporal-lead-solver/src/bin/cli.rs`
10. `~/repos/sublinear-time-solver/src/wasm_bindings.rs`

**Key questions**:
- `cli.rs` (298 LOC): R70 found temporal-solver.rs (82-86% BIMODAL) imports UltraFastTemporalSolver NOT the genuine temporal-lead predictor. Does cli.rs use the genuine O(√n) predictor from predictor.rs, or the UltraFast CG variant (R71: standard O(κn), NOT sublinear)? Does it use clap for arg parsing? What subcommands does it expose — solve, predict, validate?
- `wasm_bindings.rs` (333 LOC): This is the crate-ROOT wasm_bindings, not a subcrate. Does it expose the genuine sublinear algorithms (backward_push, forward_push, predictor) or the FALSE ones from sublinear/mod.rs? Does it use wasm-bindgen or wasm-pack? R70 found sublinear/mod.rs deliberately ORPHANS the genuine algos — does wasm_bindings.rs repeat that pattern or bypass it? WASM scoreboard currently 10:5 genuine:theatrical.

---

## Expected Outcomes

1. **CONNECTED tier CLEARED** — 0 remaining (milestone, from 3)
2. **temporal-compare crate at 76% source coverage** — 13/17 source files DEEP (from 8/17)
3. **temporal-lead-solver crate src 100% COMPLETE** — 8/8 source files (from 7/8)
4. **R68 TRIMODAL MLP arc extended** — quantized variant positioned on the quality spectrum
5. **WASM scoreboard updated** — wasm_bindings.rs classified as genuine or theatrical (currently 10:5)
6. **ReasoningBank TS layer FULLY COMPLETE** — all core, utils, and entry point files deep-read
7. **DEEP files**: ~1,184 → ~1,194 (post-R74 estimate + 10)
8. **Priority queue**: ~155 → ~145 (all PROXIMATE/NEARBY/DOMAIN_ONLY)

## DB Update Template

```javascript
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const sessionId = 75;
const today = new Date().toISOString().slice(0, 10);

// File IDs for this session:
// 10813: judge.ts (178 LOC) — agentic-flow-rust, ReasoningBank core, CONNECTED
// 10845: pii-scrubber.ts (131 LOC) — agentic-flow-rust, ReasoningBank utils, CONNECTED
// 10817: retrieve.ts (122 LOC) — agentic-flow-rust, ReasoningBank core, CONNECTED
// 14159: quantization.rs (323 LOC) — sublinear-rust, temporal-compare, PROXIMATE
// 14160: reservoir.rs (311 LOC) — sublinear-rust, temporal-compare, PROXIMATE
// 14157: mlp_quantized.rs (267 LOC) — sublinear-rust, temporal-compare, PROXIMATE
// 14145: attention.rs (198 LOC) — sublinear-rust, temporal-compare, PROXIMATE
// 14149: fourier.rs (181 LOC) — sublinear-rust, temporal-compare, PROXIMATE
// 14170: cli.rs (298 LOC) — sublinear-rust, temporal-lead-solver, PROXIMATE
// 14505: wasm_bindings.rs (333 LOC) — sublinear-rust, crate root, PROXIMATE
```

## Domain Tags

- Files 1-3 → `memory-and-learning` (already tagged)
- Files 4-8 → `neural-network-implementation` (temporal-compare — needs tagging for quantization/reservoir/attention/fourier)
- Files 9-10 → `sublinear-algorithms` (needs tagging — cli.rs and wasm_bindings.rs)

## Isolation Check

No selected files are in known-isolated subtrees. temporal-compare is in sublinear-rust (CONNECTED package, 8 DEEP files already). temporal-lead-solver is in sublinear-rust (7 DEEP files, same package). The 3 ReasoningBank TS files are in agentic-flow-rust (CONNECTED, 75 nearby DEEP). wasm_bindings.rs is at the sublinear-rust crate root.

Confirmed no overlap with known-isolated subtrees: neuro-divergent, cuda-wasm, ruvector/patches, ruvector/scripts, agentdb/simulation, goalie — none match.

---

## Synthesis Doc Update Protocol (ADR-040)

**MANDATORY**: After all files are read and findings inserted into the DB, update the relevant `domains/*/analysis.md` files following the ADR-040 in-place protocol. Reference: `domains/memory-and-learning/analysis.md` for canonical structure.

### Rules for Each Section

| Section | Action | NEVER Do |
|---------|--------|----------|
| **1. Current State Summary** | REWRITE in-place to reflect current state | Append session narrative |
| **2. File Registry** | ADD new rows to existing subsystem tables, UPDATE rows if re-read | Duplicate rows, create per-session file tables |
| **3. Findings Registry** | ADD new findings with next sequential ID (C{max+1}, H{max+1}) to 3a/3b | Create `### R75 Findings` blocks, re-list old findings, restart ID numbering |
| **4. Positives Registry** | ADD new positives with session tag | Re-list existing positives |
| **5. Subsystem Sections** | UPDATE existing sections, CREATE new ones by topic | Create per-session narrative blocks |
| **8. Session Log** | APPEND 2-5 line entry for this session | Put findings here, write full narratives |

### Finding ID Assignment

Before adding findings, check the current max ID in the target domain's analysis.md:
- Section 3a: find last `| C{N} |` row → new CRITICALs start at C{N+1}
- Section 3b: find last `| H{N} |` row → new HIGHs start at H{N+1}

**ID format**: `| {ID} | **{short title}** — {description} | {file(s)} | R{session} | Open |`

### Anti-Patterns (NEVER do these)

- **NEVER** create `### R75 Findings (Session date)` blocks outside Section 3
- **NEVER** append findings after Section 8
- **NEVER** create `### R75 Full Session Verdict` blocks
- **NEVER** use finding IDs that collide with existing ones (always check max first)
- **NEVER** re-list findings from previous sessions

### Synthesis Update Checklist

- [ ] Section 1 rewritten with updated state
- [ ] New file rows added to Section 2 (correct subsystem table)
- [ ] New findings added to Section 3a/3b with sequential IDs
- [ ] New positives added to Section 4 (if any)
- [ ] Relevant subsystem sections in Section 5 updated
- [ ] Session log entry appended to Section 8 (2-5 lines max)
- [ ] No per-session finding blocks created anywhere
