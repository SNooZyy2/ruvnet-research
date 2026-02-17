# R83 Execution Plan: ReasoningBank TS Integration, Neural Training Pipeline, ruv-swarm WASM+Infra

**Date**: 2026-02-16
**Session ID**: 83
**Focus**: First post-CONNECTED session — PROXIMATE tier files across ReasoningBank TS integration, neural-network training internals, and ruv-swarm WASM/persistence
**Strategic value**: Extends three major arcs: (1) ReasoningBank TS completeness (Rust+TS confirmed COMPLETE in R73-R78, but integration/test layer untouched), (2) neural-network-implementation training pipeline (R23 rated 90-98%, R82 qualified with theatrical integration layers — training files reveal if quality holds), (3) ruv-swarm WASM scoreboard (14 genuine vs 11 theatrical) and persistence layer (6+ disconnected layers)

## Rationale

R82 marked the exhaustion of the CONNECTED tier — all files with direct dependency links to DEEP-read files have been analyzed. From R83 onward, work shifts to PROXIMATE files: those co-located (2-level directory prefix) with high concentrations of DEEP files but lacking recorded dependency edges. PROXIMATE files are the most likely to contain integration glue, test infrastructure, and secondary implementation that confirms or contradicts prior assessments.

The three clusters selected represent the highest-value PROXIMATE targets. The ReasoningBank TS files (4 files in `agentic-flow/src/reasoningbank/`) sit alongside 85 DEEP files and may reveal whether the "Rust+TS COMPLETE" verdict extends to test infrastructure. The neural-network-implementation files (4 files across `real-implementation/`, `realistic-implementation/`, and `src/`) sit near 35 DEEP files and test R23's "BEST IN ECOSYSTEM" claim at the training pipeline level. The ruv-swarm files (2 files) extend the WASM cascade analysis and the persistence layer investigation.

No selected files are in RELIABLY isolated subtrees. The agentdb/simulation subtree IS isolated but those files were not selected. All files are in CONNECTED or WEAKLY_CONNECTED packages.

## Target: 10 files, ~1,456 LOC

---

### Cluster A: ReasoningBank TS Integration (4 files, ~581 LOC)

Extends the ReasoningBank arc (R67 GENUINE, R73-R78 Rust+TS COMPLETE). These are the remaining untouched integration and test files in the TS implementation. `consolidate.js` is the most architecturally significant — memory consolidation is a core ReasoningBank capability. The test files reveal whether the TS implementation has real test coverage or is test-facade.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 1 | 10808 | `agentic-flow/src/reasoningbank/core/consolidate.js` | 140 | CORE: memory consolidation logic, extends R67 ReasoningBank GENUINE |
| 2 | 10825 | `agentic-flow/src/reasoningbank/index-new.ts` | 132 | New entry point — may replace or extend existing index.ts |
| 3 | 10834 | `agentic-flow/src/reasoningbank/test-integration.ts` | 115 | Integration test — reveals end-to-end TS ReasoningBank coverage |
| 4 | 10836 | `agentic-flow/src/reasoningbank/test-validation.ts` | 194 | Validation test — largest test file, reveals quality thresholds |

**Full paths**:
1. `~/repos/agentic-flow/agentic-flow/src/reasoningbank/core/consolidate.js`
2. `~/repos/agentic-flow/agentic-flow/src/reasoningbank/index-new.ts`
3. `~/repos/agentic-flow/agentic-flow/src/reasoningbank/test-integration.ts`
4. `~/repos/agentic-flow/agentic-flow/src/reasoningbank/test-validation.ts`

**Key questions**:
- `consolidate.js` (140 LOC): Does it implement real memory consolidation (EWC++, importance weighting) or is it a facade? Does it connect to the Rust ReasoningBank or operate independently?
- `index-new.ts` (132 LOC): Why "new"? Does it replace/extend the existing index? What API surface does it expose?
- `test-integration.ts` (115 LOC): Are these real integration tests with assertions, or placeholder/todo tests? Do they exercise the full ReasoningBank pipeline?
- `test-validation.ts` (194 LOC): What validation criteria are tested? Do they match the statistical ranking architecture found in R67?

---

### Cluster B: Neural-Network Training Pipeline (4 files, ~580 LOC)

Extends R23 (neural-network-implementation rated 90-98% "BEST IN ECOSYSTEM") and R82 (solver_gate_simple.rs 0-5% qualifier — integration layers theatrical). These training pipeline files test whether the quality assessment holds for the training subsystem or only for inference. The `optimized_benchmark.rs` in `real-implementation/` is especially interesting — R23 found the "real-implementation" directory to be the genuine core.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 5 | 13911 | `crates/neural-network-implementation/real-implementation/optimized_benchmark.rs` | 209 | In "real-implementation" dir (R23 GENUINE core), benchmarking |
| 6 | 13923 | `crates/neural-network-implementation/realistic-implementation/simple_test.rs` | 135 | In "realistic-implementation" dir — R23 did not differentiate this from "real" |
| 7 | 13929 | `crates/neural-network-implementation/src/bin/train.rs` | 117 | Training binary entry point — reveals if training actually works |
| 8 | 13953 | `crates/neural-network-implementation/src/training/losses.rs` | 119 | Loss function implementations — core training component |

**Full paths**:
5. `~/repos/sublinear-time-solver/crates/neural-network-implementation/real-implementation/optimized_benchmark.rs`
6. `~/repos/sublinear-time-solver/crates/neural-network-implementation/realistic-implementation/simple_test.rs`
7. `~/repos/sublinear-time-solver/crates/neural-network-implementation/src/bin/train.rs`
8. `~/repos/sublinear-time-solver/crates/neural-network-implementation/src/training/losses.rs`

**Key questions**:
- `optimized_benchmark.rs` (209 LOC): Are benchmarks real (criterion/bencher) or theatrical? Do they measure actual neural-net operations?
- `simple_test.rs` (135 LOC): What's the difference between "realistic-implementation" and "real-implementation"? Is this genuine test code?
- `train.rs` (117 LOC): Does the training binary actually train a model, or is it a facade that imports but doesn't execute? Does it connect to losses.rs?
- `losses.rs` (119 LOC): Which loss functions are implemented? Are they mathematically correct (proper gradient computation)?

---

### Cluster C: ruv-swarm WASM Cascade + Persistence (2 files, ~295 LOC)

Extends two tracking systems: (1) WASM scoreboard (14 genuine vs 11 theatrical), and (2) disconnected persistence layers (6+). `cascade.rs` is in the ruv-swarm-wasm crate (R79: BIMODAL), and `sqlite-worker.js` is the npm package's persistence worker.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 9 | 9042 | `ruv-swarm/crates/ruv-swarm-wasm/src/cascade.rs` | 154 | WASM cascade logic, extends R79 BIMODAL assessment |
| 10 | 9670 | `ruv-swarm/npm/src/sqlite-worker.js` | 141 | SQLite persistence worker — extends 6+ disconnected layers |

**Full paths**:
9. `~/repos/ruv-FANN/ruv-swarm/crates/ruv-swarm-wasm/src/cascade.rs`
10. `~/repos/ruv-FANN/ruv-swarm/npm/src/sqlite-worker.js`

**Key questions**:
- `cascade.rs` (154 LOC): Is this genuine WASM cascade logic (multi-level decision trees, priority propagation) or theatrical? Does it import from ruv-swarm-wasm's real modules?
- `sqlite-worker.js` (141 LOC): Is this a real worker thread for SQLite operations, or a 7th disconnected persistence layer? Does it connect to the npm runtime's memory system?

---

## Expected Outcomes

1. ReasoningBank TS integration quality assessed — confirm or qualify "Rust+TS COMPLETE" verdict
2. Neural-network training pipeline quality scored — confirm or qualify R23's "90-98% BEST IN ECOSYSTEM"
3. WASM scoreboard updated (cascade.rs → genuine or theatrical, 15th or 12th)
4. Persistence layer count updated (sqlite-worker.js → 7th disconnected or connected)
5. DEEP count: 1,262 → ~1,272

## DB Update Template

```javascript
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const sessionId = 83;
const today = new Date().toISOString().slice(0, 10);

// File IDs for this session:
// 10808: consolidate.js (140 LOC) — agentic-flow-rust, PROXIMATE
// 10825: index-new.ts (132 LOC) — agentic-flow-rust, PROXIMATE
// 10834: test-integration.ts (115 LOC) — agentic-flow-rust, PROXIMATE
// 10836: test-validation.ts (194 LOC) — agentic-flow-rust, PROXIMATE
// 13911: optimized_benchmark.rs (209 LOC) — sublinear-rust, PROXIMATE
// 13923: simple_test.rs (135 LOC) — sublinear-rust, PROXIMATE
// 13929: train.rs (117 LOC) — sublinear-rust, PROXIMATE
// 13953: losses.rs (119 LOC) — sublinear-rust, PROXIMATE
// 9042: cascade.rs (154 LOC) — ruv-fann-rust, PROXIMATE
// 9670: sqlite-worker.js (141 LOC) — ruv-fann-rust, PROXIMATE
```

## Domain Tags

- Files 10808, 10825, 10834, 10836 → `memory-and-learning` (already tagged)
- Files 13911, 13923, 13929, 13953 → `memory-and-learning` (already tagged)
- Files 9042, 9670 → `swarm-coordination` (already tagged)

## Isolation Check

No selected files are in RELIABLY isolated subtrees. Checked:
- `neuro-divergent` (170 untouched, RELIABLE isolated) — no files selected
- `cuda-wasm` (203 untouched, RELIABLE isolated) — no files selected
- `ruvector-rust/patches` (22 untouched, RELIABLE isolated) — no files selected
- `ruvector-rust/scripts` (49 untouched, RELIABLE isolated) — no files selected
- `agentdb/simulation` (74 untouched, RELIABLE isolated) — no files selected (causal-reasoning.ts and reflexion-learning.ts appeared in priority queue but were excluded from this plan due to isolation)

Isolated packages: `claude-config` (ISOLATED, 744 files) and `@ruvector/core` (ISOLATED, 2 files) — no files selected from either.

---

## Synthesis Doc Update Protocol (ADR-040)

**MANDATORY**: After all files are read and findings inserted into the DB, update the relevant `domains/*/analysis.md` files following the ADR-040 in-place protocol. Reference: `domains/memory-and-learning/analysis.md` for canonical structure.

### Rules for Each Section

| Section | Action | NEVER Do |
|---------|--------|----------|
| **1. Current State Summary** | REWRITE in-place to reflect current state | Append session narrative |
| **2. File Registry** | ADD new rows to existing subsystem tables, UPDATE rows if re-read | Duplicate rows, create per-session file tables |
| **3. Findings Registry** | ADD new findings with next sequential ID (C{max+1}, H{max+1}) to 3a/3b | Create `### RXX Findings` blocks, re-list old findings, restart ID numbering |
| **4. Positives Registry** | ADD new positives with session tag | Re-list existing positives |
| **5. Subsystem Sections** | UPDATE existing sections, CREATE new ones by topic | Create per-session narrative blocks |
| **8. Session Log** | APPEND 2-5 line entry for this session | Put findings here, write full narratives |

### Finding ID Assignment

Before adding findings, check the current max ID in the target domain's analysis.md:
- Section 3a: find last `| C{N} |` row → new CRITICALs start at C{N+1}
- Section 3b: find last `| H{N} |` row → new HIGHs start at H{N+1}

**ID format**: `| {ID} | **{short title}** — {description} | {file(s)} | R{session} | Open |`

### Anti-Patterns (NEVER do these)

- **NEVER** create `### R{N} Findings (Session date)` blocks outside Section 3
- **NEVER** append findings after Section 8
- **NEVER** create `### R{N} Full Session Verdict` blocks
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
