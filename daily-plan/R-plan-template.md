# R{N} Execution Plan: {Title}

**Date**: {YYYY-MM-DD}
**Session ID**: {N}
**Focus**: {1-2 sentence summary of what this session covers}
**Strategic value**: {Why these files matter — what dependency gaps they close, what arcs they extend}

## Rationale

{2-3 paragraphs explaining file selection logic. Reference smart_priority_gaps tiers, connectivity, and which arcs/clusters this extends.}

## Target: {count} files, ~{total_LOC} LOC

---

### Cluster A: {Cluster Name} ({count} files, ~{LOC} LOC)

{1-2 sentences of context for this cluster.}

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 1 | {id} | `{relative_path}` | {loc} | {why selected — connected from X, extends arc Y} |

**Full paths**:
1. `~/repos/{repo}/{path}`

**Key questions**:
- `{filename}` ({LOC} LOC): {2-3 specific questions to answer during deep-read}

---

{Repeat for Clusters B, C, etc.}

---

## Expected Outcomes

1. {Outcome 1 — e.g., "CONNECTED tier CLEARED"}
2. {Outcome 2}

## DB Update Template

```javascript
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const sessionId = {N};
const today = new Date().toISOString().slice(0, 10);

// File IDs for this session:
// {id}: {filename} ({LOC} LOC) — {package}, {tier}
```

## Domain Tags

- Files {list} → `{domain}` (already tagged / needs tagging)

## Isolation Check

{Confirm no selected files are in known-isolated subtrees. Reference subtree_connectivity check.}

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
