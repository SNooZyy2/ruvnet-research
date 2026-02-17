# R101 Execution Plan: v4-priority Clearance + Postgres Arc Completions

**Date**: 2026-02-17
**Session ID**: 101
**Focus**: Clear the v4-priority domain (4 remaining files), complete postgres healing/ arc, and open the postgres GNN subsystem
**Strategic value**: Clears the last tagged priority queue entirely. Completes 2 near-done postgres arcs (hyperbolic 4/4, healing 7/7). Opens the postgres GNN layer for cross-crate comparison with already-DEEP ruvector-gnn files.

## Rationale

The `smart_priority_gaps` view now returns ONLY 4 files — all in `v4-priority`. No other domain has tier_rank <= 3 gaps. Clearing these 4 files empties the priority queue for the first time since R88 repopulated it. This is a project milestone.

Beyond the mandatory 4, the session targets two strategic clusters in ruvector-postgres. The `healing/` directory has 5/7 files at DEEP (from R36/R98) — the remaining `functions.rs` and `mod.rs` complete the arc. The postgres `gnn/` directory (6 files, ~1,505 LOC) is entirely untouched but directly parallels the ruvector-gnn crate (11/13 DEEP from R91/R94/R99). Reading the postgres GNN SQL implementation enables cross-crate comparison: is postgres GNN a genuine SQL port, or a facade mirroring the Rust crate's interface?

No selected files are in isolated subtrees. All are in `ruvector-rust` package (CONNECTED). The `subtree_connectivity` check confirms no isolation concerns for postgres or mincut dirs.

## Target: 9 files, ~2,618 LOC

---

### Cluster A: v4-priority Completion (4 files, ~1,032 LOC)

Clears the final 4 files from the tagged priority queue. Each completes or extends a near-done arc.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 1 | 3829 | `crates/ruvector-postgres/src/hyperbolic/operators.rs` | 395 | CONNECTED — completes postgres hyperbolic/ (3/4 DEEP: lorentz, mod, poincare) |
| 2 | 3491 | `crates/ruvector-mincut-gated-transformer/src/kernel/bench_utils.rs` | 442 | PROXIMATE — last non-DEEP in MinCut kernel/ (40 nearby DEEP files) |
| 3 | 3155 | `crates/ruvector-gnn/src/error.rs` | 112 | PROXIMATE — GNN error types, tagged MEDIUM but 0 lines read |
| 4 | 3160 | `crates/ruvector-gnn/src/mmap_fixed.rs` | 83 | PROXIMATE — GNN mmap fix, tagged MEDIUM but 0 lines read |

**Full paths**:
1. `~/repos/ruvector/crates/ruvector-postgres/src/hyperbolic/operators.rs`
2. `~/repos/ruvector/crates/ruvector-mincut-gated-transformer/src/kernel/bench_utils.rs`
3. `~/repos/ruvector/crates/ruvector-gnn/src/error.rs`
4. `~/repos/ruvector/crates/ruvector-gnn/src/mmap_fixed.rs`

**Key questions**:
- `operators.rs` (395 LOC): Does it implement poincare_distance/lorentz_distance as SQL functions? Does it reference the hyperbolic types from poincare.rs/lorentz.rs? Any manifold validation (R98 flagged lorentz missing it)?
- `bench_utils.rs` (442 LOC): Real criterion benchmarks with SIMD kernel invocations? Or boilerplate/stubs? Does it benchmark the kernels we rated 88-93% in R93?
- `error.rs` (112 LOC): Standard thiserror types or custom error hierarchy? Does it cover mmap failures?
- `mmap_fixed.rs` (83 LOC): What does "fixed" mean — is this a bugfix patch? Does it wrap mmap with safety checks?

---

### Cluster B: Postgres Healing Arc Completion (2 files, ~702 LOC)

Completes the healing/ directory. R36 analyzed the core healing infrastructure, R98 read additional healing files. 5/7 files are DEEP — these last 2 close the arc.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 5 | 3816 | `crates/ruvector-postgres/src/healing/functions.rs` | 468 | SQL-exposed healing functions (CREATE FUNCTION wrappers?) |
| 6 | 3818 | `crates/ruvector-postgres/src/healing/mod.rs` | 234 | Module root — re-exports, initialization, possibly healing loop setup |

**Full paths**:
5. `~/repos/ruvector/crates/ruvector-postgres/src/healing/functions.rs`
6. `~/repos/ruvector/crates/ruvector-postgres/src/healing/mod.rs`

**Key questions**:
- `functions.rs` (468 LOC): What SQL functions does it expose (heal_index, repair_corruption, check_integrity)? Does it use the healing strategies analyzed in R36?
- `mod.rs` (234 LOC): Does it orchestrate the healing pipeline (detect → diagnose → repair)? Any background worker registration?

---

### Cluster C: Postgres GNN Layer (3 files, ~884 LOC)

Opens the postgres GNN subsystem. ruvector-gnn crate is extensively analyzed (R91/R94/R99); this is the SQL-side implementation. Key question: genuine SQL port or interface mirror?

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 7 | 3797 | `crates/ruvector-postgres/src/gnn/operators.rs` | 426 | SQL GNN operators — parallels attention/operators.rs pattern (already DEEP) |
| 8 | 3793 | `crates/ruvector-postgres/src/gnn/gcn.rs` | 224 | Graph Convolutional Network in SQL — compare with ruvector-gnn's implementation |
| 9 | 3795 | `crates/ruvector-postgres/src/gnn/message_passing.rs` | 234 | GNN message passing — the core algorithm layer |

**Full paths**:
7. `~/repos/ruvector/crates/ruvector-postgres/src/gnn/operators.rs`
8. `~/repos/ruvector/crates/ruvector-postgres/src/gnn/gcn.rs`
9. `~/repos/ruvector/crates/ruvector-postgres/src/gnn/message_passing.rs`

**Key questions**:
- `operators.rs` (426 LOC): Does it register GNN SQL functions (gnn_train, gnn_predict, gnn_embed)? Does it call into ruvector-gnn crate or reimplement?
- `gcn.rs` (224 LOC): Real GCN math (adjacency normalization, feature propagation) or wrapper around ruvector-gnn? How does it handle graph structure in SQL tables?
- `message_passing.rs` (234 LOC): Genuine neighborhood aggregation? Does it support custom aggregation (sum/mean/max)?

---

## Expected Outcomes

1. **v4-priority domain CLEARED** — priority queue empty for all tagged domains
2. **Postgres hyperbolic/ arc COMPLETE** (4/4 DEEP)
3. **Postgres healing/ arc COMPLETE** (7/7 DEEP)
4. **MinCut kernel/ arc COMPLETE** (all files DEEP)
5. **ruvector-gnn crate near-COMPLETE** (13/13 or 11/13 DEEP pending depth classification)
6. **Postgres GNN layer OPENED** — 3/6 files analyzed, cross-crate comparison established
7. DEEP count: 1,421 → ~1,430

## DB Update Template

```javascript
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const sessionId = 101;
const today = new Date().toISOString().slice(0, 10);

// File IDs for this session:
// 3829: operators.rs (395 LOC) — ruvector-postgres/hyperbolic, CONNECTED
// 3491: bench_utils.rs (442 LOC) — mincut-gated-transformer/kernel, PROXIMATE
// 3155: error.rs (112 LOC) — ruvector-gnn, PROXIMATE
// 3160: mmap_fixed.rs (83 LOC) — ruvector-gnn, PROXIMATE
// 3816: functions.rs (468 LOC) — ruvector-postgres/healing
// 3818: mod.rs (234 LOC) — ruvector-postgres/healing
// 3797: operators.rs (426 LOC) — ruvector-postgres/gnn
// 3793: gcn.rs (224 LOC) — ruvector-postgres/gnn
// 3795: message_passing.rs (234 LOC) — ruvector-postgres/gnn
```

## Domain Tags

- Files 3829, 3816, 3818, 3797, 3793, 3795 → `ruvector` (needs tagging)
- Files 3155, 3160 → `ruvector` (needs tagging)
- Files 3491 → `ruvector` (needs tagging)
- Files 3829, 3491, 3155, 3160 → `v4-priority` (already tagged)

## Isolation Check

All 9 files are in `ruvector-rust` package (connectivity: CONNECTED, 570+ cross-deps). No files are in isolated subtrees. The `subtree_connectivity` check confirms no isolation for `crates/ruvector-postgres`, `crates/ruvector-mincut-gated-transformer`, or `crates/ruvector-gnn` directories.

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
