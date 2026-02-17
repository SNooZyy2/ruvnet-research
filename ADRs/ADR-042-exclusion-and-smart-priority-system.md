# ADR-042: Exclusion System and Smart Priority Gaps

**Status**: IMPLEMENTED
**Date**: 2026-02-16
**Depends on**: ADR-038 (Research Database System)

---

## Context

After 61 research sessions, `priority_gaps` was surfacing irrelevant files: quantum experiments, consciousness demos, Goalie (a standalone Perplexity wrapper), benchmark stubs, and release scripts. Root causes:

1. **Bulk domain tagging** — entire repos got blanket-tagged (e.g., all of `sublinear-rust` → `memory-and-learning`)
2. **No exclusion mechanism** — once tagged, files couldn't be deprioritized without re-tagging
3. **No relevance signal** — a file in `src/consciousness_demo.rs` ranked equally with `src/solver_core.rs`
4. **Sparse dependency data** — only 29 cross-package deps recorded across 1,049 DEEP files, so dependency proximity alone was unreliable

Of 421 files in `priority_gaps`, zero had a recorded dependency to claude-flow's runtime. 17% were clearly unrelated (demos, experiments, scripts).

---

## Decision

### 1. `exclude_paths` table

Dynamic path-pattern exclusion. Patterns are SQL LIKE expressions auto-filtered by `priority_gaps`.

```sql
CREATE TABLE exclude_paths (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern TEXT NOT NULL UNIQUE,
    reason TEXT NOT NULL,
    added_session_id INTEGER REFERENCES sessions(id),
    added_date TEXT NOT NULL
);
```

39 patterns added covering: goalie, docs, dist, validation, benchmarks, examples, demos, standalone experiments, release scripts, test infrastructure.

### 2. `smart_priority_gaps` view

Relevance-tiered priority queue using two signals: dependency connectivity and directory co-location with DEEP files.

| Tier | Rank | Signal |
|------|------|--------|
| CONNECTED | 1 | Has recorded dependency to/from a DEEP file |
| OWN_CODE | 1 | In `custom-src` package (our DDD implementation) |
| PROXIMATE | 2 | Same 2-level directory as 3+ DEEP files |
| NEARBY | 3 | Same 2-level directory as 1-2 DEEP files |
| DOMAIN_ONLY | 4 | Only HIGH-priority domain tag, no proximity signal |

Standard query: `SELECT * FROM smart_priority_gaps WHERE tier_rank <= 2 LIMIT 10`

### 3. `package_connectivity` and `subtree_connectivity` views

Detect isolated packages and directory subtrees before wasting research cycles. `subtree_connectivity` includes a `confidence` column (RELIABLE / LOW_CONFIDENCE / NO_DATA) based on what percentage of DEEP files have recorded dependencies.

### 4. EXCLUDED filter on all views

All 12 DB views now filter `depth != 'EXCLUDED'`. The `report.js` script excludes them from MASTER-INDEX.md stats.

---

## Changes Made

| Artifact | Change |
|----------|--------|
| Live DB | New table (`exclude_paths`), 4 new views, 8 existing views rebuilt with EXCLUDED filter |
| `db/schema.sql` | Synced with live DB — all 13 tables, 12 views now match |
| `scripts/report.js` | Uses `smart_priority_gaps` with tier column; excludes EXCLUDED from stats |
| `CLAUDE.md` | Session Protocol step 2b (mandatory island check); query recipes 1/1b/12-14; schema reminders |
| `agents/synthesizer.md` | Gap query now filters `exclude_paths` |
| Auto-memory (`MEMORY.md`) | Exclusion system documented |

---

## Result

- Priority queue: 421 → 298 files (29% noise removed)
- Top 10 now shows genuinely connected files (`forward_push.rs`, `solver_core.rs`, custom-src DDD adapters)
- Zero manual re-tagging needed — purely additive SQL computation
- Agents automatically get filtered results via views; CLAUDE.md enforces the island check step

---

## Alternatives Considered

- **Re-tag `file_domains`**: Correct in principle but poor ROI — hundreds of rows to audit, and the smart view achieves the same practical effect
- **Dependency-only ranking**: Blocked by sparse dep data (29 cross-package deps). Co-location is a stronger signal with current data
- **Per-session manual exclusion**: Doesn't scale and agents forget between sessions
