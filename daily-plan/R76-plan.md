# R76 Execution Plan: MCP Tools + Sublinear Algorithms + ReasoningBank Core

**Date**: 2026-02-16
**Session ID**: 76
**Focus**: Clear remaining CONNECTED files, then sweep two high-density PROXIMATE clusters: sublinear-time-solver MCP tools and sublinear Rust algorithms
**Strategic value**: The 3 CONNECTED files close the last direct-dependency gaps in the priority system. The MCP tools cluster (10 files, ~2K LOC) is the largest untouched PROXIMATE group in sublinear-time-solver and likely reveals how the solver's Rust algorithms are exposed to end users. The sublinear Rust cluster extends the genuine-vs-false sublinearity arc (R39/R56/R58/R60).

## Rationale

R75 cleared the last temporal-compare internals and the temporal-lead-solver CLI. Three CONNECTED files remain — two ReasoningBank core JS files (`distill.js`, `retrieve.js`) and the tiny `temporal-compare/src/lib.rs` crate root. These are the highest-signal targets because they have direct dependency links to DEEP files.

After clearing CONNECTED, the largest untouched PROXIMATE cluster is `src/mcp/tools/` in sublinear-time-solver (10 files, ~2K LOC). These MCP tool wrappers are the **user-facing API surface** for the solver's algorithms — reading them will reveal which algorithms are actually wired up for production use vs. facade. This directly extends the MCP protocol arc (3 protocols found so far) and the WASM genuine-vs-theatrical arc.

The second cluster is `src/sublinear/` (3 files, 670 LOC) — `spectral_sparsification.rs`, `sketching.rs`, and `dimension_reduction.rs`. These extend the sublinearity arc where we've found 3 genuine and 4 false sublinear algorithms. These files sit next to already-DEEP files like `backward_push.rs` (92-95%) and `predictor.rs` (92-95%).

Finally, `src/bmssp.rs` (313 LOC) extends R59's discovery of the BMSSP "invented algorithm" — this is the full Rust implementation vs the benchmark we already read.

## Target: 10 files, ~2,776 LOC

---

### Cluster A: CONNECTED Tier Clearance (3 files, ~270 LOC)

The last 3 files with direct dependency edges to DEEP files. Clearing these means the entire CONNECTED tier is empty — a milestone for the priority system.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 1 | 10810 | `agentic-flow/src/reasoningbank/core/distill.js` | 160 | CONNECTED to DEEP ReasoningBank files. JS distill — compare with R72's TS distill.ts (78-82%) |
| 2 | 10816 | `agentic-flow/src/reasoningbank/core/retrieve.js` | 87 | CONNECTED to DEEP ReasoningBank files. Core retrieval — compare with R75's retrieve.ts |
| 3 | 14150 | `crates/temporal-compare/src/lib.rs` | 23 | CONNECTED to R75's temporal-compare internals. Tiny crate root — what's exported? |

**Full paths**:
1. `~/repos/agentic-flow/src/reasoningbank/core/distill.js`
2. `~/repos/agentic-flow/src/reasoningbank/core/retrieve.js`
3. `~/repos/sublinear-time-solver/crates/temporal-compare/src/lib.rs`

**Key questions**:
- `distill.js` (160 LOC): Is this the same LLM-based extraction as R72's distill.ts? Does it have the same 13th hash-based embedding? How does it relate to index.js (100% BEST JS)?
- `retrieve.js` (87 LOC): Does this implement actual vector retrieval or hash-based lookup? Does it connect to AgentDB or use its own backend?
- `lib.rs` (23 LOC): Which temporal-compare modules does it re-export? Does it expose R75's quantization/reservoir/attention/fourier?

---

### Cluster B: Sublinear Rust Algorithms (4 files, ~983 LOC)

Three files from `src/sublinear/` extending the sublinearity arc, plus the full BMSSP implementation. These sit next to `backward_push.rs` (92-95% GENUINE O(1/ε)) and `predictor.rs` (92-95% O(√n)).

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 4 | 14460 | `src/sublinear/spectral_sparsification.rs` | 312 | Spectral sparsification — likely graph algorithm. Genuine or false sublinear? |
| 5 | 14459 | `src/sublinear/sketching.rs` | 255 | Sketching algorithms — CountMin/CountSketch? Extends R62 JL findings |
| 6 | 14455 | `src/sublinear/dimension_reduction.rs` | 103 | Dimension reduction — JL transform or PCA? R62 found false JL sublinearity |
| 7 | 14303 | `src/bmssp.rs` | 313 | Full BMSSP implementation. R59 found this "invented algorithm" in benchmarks |

**Full paths**:
4. `~/repos/sublinear-time-solver/src/sublinear/spectral_sparsification.rs`
5. `~/repos/sublinear-time-solver/src/sublinear/sketching.rs`
6. `~/repos/sublinear-time-solver/src/sublinear/dimension_reduction.rs`
7. `~/repos/sublinear-time-solver/src/bmssp.rs`

**Key questions**:
- `spectral_sparsification.rs` (312 LOC): Does it implement Spielman-Srivastava or Benczúr-Karger? What's the actual complexity? Real effective resistance sampling?
- `sketching.rs` (255 LOC): CountMin Sketch, CountSketch, or AMS? Does it achieve genuine sublinear space?
- `dimension_reduction.rs` (103 LOC): JL transform — does it repeat R62's false O(1) claims or implement honest O(nd) projection?
- `bmssp.rs` (313 LOC): Is BMSSP genuinely novel or a rebrand of existing algorithm? What problem does it solve?

---

### Cluster C: MCP Tool Wrappers — Top 3 (3 files, ~938 LOC)

The 3 largest MCP tool files from `src/mcp/tools/`. These reveal which solver algorithms are exposed as MCP tools — the user-facing API surface.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 8 | 14404 | `src/mcp/tools/temporal-attractor-handlers.ts` | 353 | Handler implementations for temporal attractor MCP tools |
| 9 | 14401 | `src/mcp/tools/solver-optimized.ts` | 327 | Optimized solver MCP tool — wraps which Rust algorithms? |
| 10 | 14409 | `src/mcp/tools/wasm-sublinear-solver-simple.ts` | 269 | Simple WASM solver — genuine WASM bridge or theatrical? |

**Full paths**:
8. `~/repos/sublinear-time-solver/src/mcp/tools/temporal-attractor-handlers.ts`
9. `~/repos/sublinear-time-solver/src/mcp/tools/solver-optimized.ts`
10. `~/repos/sublinear-time-solver/src/mcp/tools/wasm-sublinear-solver-simple.ts`

**Key questions**:
- `temporal-attractor-handlers.ts` (353 LOC): Does it call actual Rust temporal-attractor code or stub? Is this a 4th MCP protocol or reuses existing?
- `solver-optimized.ts` (327 LOC): Which optimization algorithms does it wrap? Does it connect to the genuine sublinear Rust code (backward_push, predictor)?
- `wasm-sublinear-solver-simple.ts` (269 LOC): Genuine WASM bridge to Rust or theatrical facade? Extends WASM scoreboard (10 genuine vs 5 theatrical)

---

## Expected Outcomes

1. **CONNECTED tier CLEARED** — all 3 remaining files read, priority queue drops to PROXIMATE-only
2. **ReasoningBank JS layer assessed** — distill.js and retrieve.js compared with TS counterparts
3. **Sublinearity arc extended** — 3 new `src/sublinear/` algorithms classified as genuine or false
4. **BMSSP "invented algorithm" fully assessed** — R59 benchmark context now has implementation context
5. **MCP tool surface mapped** — how solver algorithms are exposed to users
6. **WASM scoreboard updated** — wasm-sublinear-solver-simple classified
7. **DEEP count**: 1,192 → ~1,202

## DB Update Template

```javascript
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const sessionId = 76;
const today = new Date().toISOString().slice(0, 10);

// File IDs for this session:
// 10810: distill.js (160 LOC) — agentic-flow-rust, CONNECTED
// 10816: retrieve.js (87 LOC) — agentic-flow-rust, CONNECTED
// 14150: lib.rs (23 LOC) — sublinear-rust, CONNECTED
// 14460: spectral_sparsification.rs (312 LOC) — sublinear-rust, PROXIMATE
// 14459: sketching.rs (255 LOC) — sublinear-rust, PROXIMATE
// 14455: dimension_reduction.rs (103 LOC) — sublinear-rust, PROXIMATE
// 14303: bmssp.rs (313 LOC) — sublinear-rust, PROXIMATE
// 14404: temporal-attractor-handlers.ts (353 LOC) — sublinear-rust, PROXIMATE
// 14401: solver-optimized.ts (327 LOC) — sublinear-rust, PROXIMATE
// 14409: wasm-sublinear-solver-simple.ts (269 LOC) — sublinear-rust, PROXIMATE
```

## Domain Tags

- Files 1-2 (distill.js, retrieve.js) → `memory-and-learning` (already tagged)
- Files 3-10 (all sublinear-rust) → `memory-and-learning` (already tagged)
- Files 8-10 may also need `claude-flow-cli` if they expose MCP tools in claude-flow's tool registry

## Isolation Check

No selected files are in known-isolated subtrees. The RELIABLE isolated subtrees are:
- `neuro-divergent` (ruv-fann) — not selected
- `cuda-wasm` (ruv-fann) — not selected
- `patches` (ruvector) — not selected
- `scripts` (ruvector) — not selected
- `simulation` (agentdb) — not selected

All selected files are in CONNECTED packages (`agentic-flow-rust`, `sublinear-rust`) with confirmed cross-package dependencies.

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
