# Codebase Coverage Ledger

> **Purpose:** This document drives systematic, exhaustive codebase exploration. It tracks exactly what has been read, at what depth, and what remains. Every research session consumes this ledger, picks up where the last left off, and updates it with findings.
>
> **How to use:** Copy this template per project. Run Phase 0 first to auto-populate the file inventory. Then execute research sessions against the queue, updating depth status and writing module briefs as you go.

---

## Project Metadata

| Field | Value |
|-------|-------|
| **Repository** | `[org/repo-name]` |
| **Cloned at** | `[commit hash + date]` |
| **Total files** | `[auto-populated by Phase 0]` |
| **Total LOC** | `[auto-populated by Phase 0]` |
| **Ledger version** | `1` |
| **Last session** | `[date + session number]` |

---

## Coverage Summary

Updated after each session. This is the first thing Claude reads to know the state of exploration.

| Depth Level | Definition | File Count | LOC Count | % of Total LOC |
|-------------|-----------|------------|-----------|-----------------|
| **DEEP** | Every line read. Algorithms traced. Exact code extracted. Data flow verified. | 0 | 0 | 0% |
| **MEDIUM** | Key sections read. Architecture understood. Some paths untraced. | 0 | 0 | 0% |
| **SURFACE** | Glob/grep only. Categorized by filename/directory. 0-2 lines read. | 0 | 0 | 0% |
| **UNTOUCHED** | Not opened. Not grepped. Exists in inventory only. | 0 | 0 | 0% |

**Effective deep coverage:** `0%` (DEEP LOC / Total LOC)

---

## Phase 0: Repository Inventory

> Run this once at project start. Produces the raw file list that populates the ledger.

### Phase 0 Script

```bash
# Run from repo root. Generates the initial inventory.
# Excludes: node_modules, .git, dist, build, coverage, lock files, images, binaries

find . -type f \
  ! -path '*/node_modules/*' \
  ! -path '*/.git/*' \
  ! -path '*/dist/*' \
  ! -path '*/build/*' \
  ! -path '*/coverage/*' \
  ! -path '*/.next/*' \
  ! -name '*.lock' \
  ! -name 'package-lock.json' \
  ! -name 'yarn.lock' \
  ! -name '*.png' ! -name '*.jpg' ! -name '*.gif' ! -name '*.ico' \
  ! -name '*.woff' ! -name '*.woff2' ! -name '*.ttf' ! -name '*.eot' \
  ! -name '*.wasm' \
  | while read f; do
    lines=$(wc -l < "$f" 2>/dev/null || echo "0")
    echo "$lines|$f"
  done | sort -t'|' -k1 -rn
```

### Subsystem Map

> After running Phase 0, group files into logical subsystems. This becomes the exploration order.

| # | Subsystem | Directory/Pattern | File Count | Total LOC | Priority | Status |
|---|-----------|-------------------|------------|-----------|----------|--------|
| 1 | | | | | | ⬜ NOT STARTED |
| 2 | | | | | | ⬜ NOT STARTED |
| 3 | | | | | | ⬜ NOT STARTED |

**Priority key:** P0 = core architecture, read first. P1 = major functionality. P2 = supporting/utility. P3 = config/generated/tests.

**Status key:** ⬜ NOT STARTED · 🔶 IN PROGRESS · ✅ COMPLETE (all files DEEP or MEDIUM with justification)

---

## File-Level Coverage Registry

> The ground truth. One row per source file. Updated after every read operation.
> Sort by subsystem, then by priority within subsystem.

### How to update this registry

After reading a file, update its row:
- **Depth:** UNTOUCHED → SURFACE → MEDIUM → DEEP
- **Lines read:** Actual line ranges examined (e.g., `1-50, 200-350, 490-520`)
- **Session:** Which session performed the read
- **Key findings:** One-line summary. Details go in the Module Brief below.

### Registry

<!-- SUBSYSTEM: [Name] -->

| File | LOC | Depth | Lines Read | Session | Key Findings |
|------|-----|-------|------------|---------|--------------|
| `path/to/file.js` | 0 | UNTOUCHED | — | — | — |

<!-- Copy this block per subsystem -->

---

## Research Session Log

> Each session is one Claude conversation or one execution of a research.md script.
> Sessions have a clear scope (which files/subsystems to cover) and produce updates to the registry + module briefs.

### Session Template

```markdown
### Session [N] — [Date]

**Scope:** [Which subsystem(s) / files targeted]
**Goal:** [What questions to answer]
**Time budget:** [How many files / lines to cover]

**Files read this session:**
- `file.js` (lines 1-500) → DEEP. [one-line finding]
- `other.js` (lines 1-100, 300-400) → MEDIUM. [one-line finding]

**Registry updates:** [list of depth changes]
**Module brief updates:** [which briefs were created/updated]
**Open questions for next session:** [what to investigate next]
**Next session scope:** [specific files/subsystems to tackle]
```

### Session History

<!-- Append sessions here -->

---

## Module Briefs

> Dense knowledge documents per subsystem. This is where the actual understanding lives.
> Each brief should give Claude enough context to work on that subsystem competently.

### Module Brief Template

```markdown
## Brief: [Subsystem Name]

### Purpose
[What this subsystem does in 1-2 sentences]

### Entry Points
- [Main file(s) where execution starts]
- [Exported API surface]

### Architecture
[How it's structured internally. Key classes/functions and their relationships.]

### Data Flow
[What goes in, what transformations happen, what comes out. Include exact function names.]

### Key Algorithms & Logic
[Non-obvious logic. Scoring formulas, state machines, retry strategies, etc. Include exact thresholds/values.]

### Dependencies
- **Depends on:** [other subsystems/modules it imports]
- **Depended on by:** [what imports this]

### Configuration
[How behavior is configured. Env vars, config files, feature flags.]

### Patterns & Conventions
[Coding patterns specific to this subsystem. Error handling approach, naming conventions, etc.]

### Gotchas & Non-Obvious Behavior
[Things that would trip up someone working on this code for the first time.]

### Open Questions
[What remains unclear even after deep reading.]
```

### Briefs

<!-- Append module briefs here -->

---

## Exploration Queue

> The prioritized backlog of what to read next. Consumed by each session.
> Regenerate after each session based on updated coverage gaps.

### Current Queue

| Priority | File / Area | Current Depth | Target Depth | Why | Blocked By |
|----------|-------------|---------------|--------------|-----|------------|
| | | | | | |

### Queue Generation Rules

1. **P0 — Architectural spine first:** Entry points, main orchestrators, config files, type definitions.
2. **P1 — Trace call paths:** Once spine is known, follow actual execution paths through the code.
3. **P2 — Fill coverage gaps:** Files in SURFACE/UNTOUCHED that are imported by DEEP files.
4. **P3 — Long tail:** Utility files, tests, generated code. Read selectively.

### Depth Escalation Triggers

Move a file from SURFACE/MEDIUM to DEEP when:
- It's imported by 3+ other files (high fan-in = architectural importance)
- It contains >200 LOC of non-trivial logic (not just config/wiring)
- Understanding it is blocking understanding of a DEEP file
- It contains the implementation of a pattern used across the codebase

---

## Cross-Cutting Findings

> Patterns, conventions, and architectural decisions that span multiple subsystems.
> These emerge during exploration and inform how to read the rest of the codebase.

### Codebase-Wide Patterns
<!-- e.g., "All async operations use a custom retry wrapper in utils/retry.js" -->

### Architectural Decisions Discovered
<!-- e.g., "Event-driven via EventEmitter, not request/response between subsystems" -->

### Naming Conventions
<!-- e.g., "Files ending in -tools.js export MCP tool arrays. Files ending in -hooks.sh are lifecycle scripts." -->

### Tech Debt / Code Smells Noted
<!-- Track these as you encounter them, they inform where the codebase is fragile -->

---

## Verification Checklist

> Run this periodically to ensure the ledger is honest.

- [ ] Every file from Phase 0 inventory has a row in the registry
- [ ] No file is marked DEEP unless actual line ranges are recorded
- [ ] Coverage Summary percentages match registry counts
- [ ] Every DEEP/MEDIUM file has findings recorded (not just "read")
- [ ] Module briefs exist for every subsystem marked ✅ COMPLETE
- [ ] Exploration queue is regenerated from current gaps
- [ ] Cross-cutting findings are updated with latest session discoveries
