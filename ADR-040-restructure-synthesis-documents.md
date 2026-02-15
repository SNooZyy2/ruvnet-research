# ADR-040: Restructure Existing Synthesis Documents

**Status**: PROPOSED
**Date**: 2026-02-15
**Relates to**: ADR-038 (Research Database System), ADR-041 (Workflow Changes)

---

## Context

The 14 domain synthesis files (`domains/*/analysis.md`) have grown to 7,035 total lines. They were designed as synthesis documents (ADR-038 Section 2) but have devolved into **append-only chronological session logs** with severe structural problems.

### Measured Problems

1. **Cumulative finding repetition**: Each session appends an "Updated CRITICAL Findings (+N = M total)" section that re-lists ALL previous findings in full text. For example, `memory-and-learning/analysis.md` has CRITICAL findings numbered 1-30 with every prior finding repeated at each checkpoint (lines 194, 763, 829). Finding #1 ("Three fragmented ReasoningBanks") appears in 12+ locations across the file.

2. **Same pattern for HIGH and Positive lists**: "Updated HIGH Findings" grows from 8 to 23 entries with full repetition at each session boundary. "Updated Positive" sections do the same.

3. **Chronological not topical**: To understand "what do we know about consciousness?", you must read R21, R25, R33, AND R41 sections scattered across 997 lines of `memory-and-learning/analysis.md`. There is no current-state summary.

4. **File sizes**: 6 files exceed 300 lines; 4 exceed 700 lines. The largest (`ruvector/analysis.md`) is 1,319 lines.

| File | Lines | Sessions Covered |
|------|-------|-----------------|
| ruvector | 1,319 | R4-R42 |
| claude-flow-cli | 1,045 | R3-R36 |
| swarm-coordination | 1,089 | R5-R42 |
| memory-and-learning | 997 | R8-R41 |
| agentdb-integration | 716 | R8-R41 |
| agentic-flow | 714 | R10-R40 |
| hook-pipeline | 304 | R7-R19 |
| model-routing | 285 | R7-R19 |
| agent-lifecycle | 279 | R5-R16 |
| init-and-codegen | 120 | R3-R10 |
| production-infra | 69 | (stub) |
| process-spawning | 68 | (stub) |
| plugin-system | 15 | (stub) |
| transfer-system | 15 | (stub) |

### Estimated Redundancy

~25-30% of the content in the 6 largest files is redundant repetition from the cumulative counting pattern. This accounts for an estimated 1,200-1,500 wasted lines across all files.

### What is NOT Redundant

- Per-file analysis tables (each file's LOC/Real%/Verdict is unique data)
- Component breakdowns within files (specific line references, algorithm details)
- Session-specific architectural insights (connection maps, architecture diagrams)
- Cross-session corrections (R38 correcting R13, R42 correcting R28, etc.)
- Subsystem-level verdicts ("emergence is 51% fabricated", "consciousness is 79% genuine")

---

## Decision

Restructure each domain synthesis file from chronological session logs into **topical synthesis documents** with the following canonical structure:

### New Document Structure

```
# {Domain Name} Domain Analysis

> **Priority**: ... | **Coverage**: ... | **Status**: ...
> **Last updated**: {date} (Session R{n})

## 1. Current State Summary
20-30 lines. What we know NOW. Key verdicts, top risks, overall realness %.
Written in present tense. Updated in-place each session — NOT appended.

## 2. File Registry
ONE consolidated table of ALL deep-read files in this domain.
Columns: File | Package | LOC | Real% | Depth | Key Verdict | Session
Sorted by package then file path. Updated in-place (rows added/modified).

## 3. Findings Registry
ALL findings in ONE deduplicated list, grouped by severity.
Each finding has: ID, description, file(s), session discovered, status (open/resolved/superseded).
When a finding is corrected by a later session, mark it as SUPERSEDED with reference.
NEVER re-list findings — update this single list.

### 3a. CRITICAL Findings
### 3b. HIGH Findings
### 3c. MEDIUM Findings (if tracked in synthesis)

## 4. Positives Registry
Same structure as findings — one deduplicated list.
Each positive has: description, file(s), session discovered.

## 5. Subsystem Sections
Organized by TOPIC (not by session date). Each subsystem gets a section:
- What it does, how it works
- Key files involved (references to File Registry by name)
- Architecture/data flow diagrams
- Quality assessment
Updated in-place when new information arrives.

Examples for memory-and-learning:
  - 5a. ReasoningBank Implementations
  - 5b. Embedding Fallback Chain
  - 5c. SONA Crate
  - 5d. Neural Pattern Recognition
  - 5e. Consciousness + Strange Loop
  - 5f. Emergence Subsystem

## 6. Cross-Domain Dependencies
Links to other domains. Updated in-place.

## 7. Knowledge Gaps
What remains unread or uncertain. Items removed when resolved.

## 8. Session Log (Appendix)
Collapsed/brief per-session notes for PROVENANCE ONLY.
Format: `### R{n} ({date}): {1-2 sentence summary}`
NO findings re-listed here — just session scope and file counts.
This section is append-only but each entry is short (2-5 lines max).
```

### Migration Rules

1. **File Registry**: Consolidate all per-session file tables into one master table. Deduplicate files that appear in multiple sessions (keep highest depth, latest Real%).

2. **Findings Registry**: Extract all findings from cumulative lists. Assign sequential IDs within each domain. Mark superseded findings (e.g., R13 "no Cypher executor" → SUPERSEDED by R38). Remove all "Updated CRITICAL Findings (+N = M total)" sections.

3. **Subsystem Sections**: Group related per-session analyses into topical sections. For example, R8 AgentDB core + R16 CLI surface + R40 intelligence + R41 simulations → four subsections under "AgentDB Architecture".

4. **Session Log**: Compress each session's narrative to 2-5 lines of provenance. Example:
   ```
   ### R8 (2026-02-09): AgentDB core deep-read
   7 files, 8,594 LOC. Established vector-quantization as production-grade,
   LearningSystem RL as cosmetic, CausalMemoryGraph statistics as broken.
   ```

5. **Corrections**: When R38 corrects R13, the original finding in the Findings Registry is marked `SUPERSEDED by R38: rvlite has working executor`. The correction lives in the finding itself, not scattered across session logs.

### Expected Outcome

| Metric | Before | After (est.) |
|--------|--------|-------------|
| Total lines across 14 files | 7,035 | ~4,000-4,500 |
| Lines to find "all CRITICAL findings" | Scan full file | Jump to Section 3a |
| Lines to find "all files in domain" | Scan full file | Jump to Section 2 |
| Duplicate finding text | 1,200-1,500 lines | 0 |
| Time to answer "what do we know about X?" | Read multiple session blocks | Read one subsystem section |

### Migration Scope

Only the 6 files >300 lines need full restructuring:
- `memory-and-learning/analysis.md` (997 lines)
- `ruvector/analysis.md` (1,319 lines)
- `claude-flow-cli/analysis.md` (1,045 lines)
- `swarm-coordination/analysis.md` (1,089 lines)
- `agentdb-integration/analysis.md` (716 lines)
- `agentic-flow/analysis.md` (714 lines)

The 4 files between 120-304 lines can be restructured opportunistically (they're small enough to not be painful). The 4 stub files (<120 lines) need no migration — they'll adopt the new structure when content is first added.

### What This ADR Does NOT Cover

- Changes to the database schema (no schema changes needed)
- Changes to `report.js` or `MASTER-INDEX.md` generation
- Changes to agent prompts or session workflows (covered by ADR-041)
- Changes to MEMORY.md or CLAUDE.md

---

## Risks

1. **Information loss during migration**: Mitigated by doing migration in a session that can be reviewed. The DB retains all findings independently — synthesis docs are secondary.

2. **Subsystem grouping is subjective**: Different analysts might group topics differently. Mitigated by using domain expertise accumulated over R1-R42 to make informed groupings.

3. **Large diff**: Each file will be substantially rewritten. Mitigated by committing the migration as a single atomic commit with clear message.

---

## Implementation Plan

1. Migrate one file first (`memory-and-learning/analysis.md`) as the template/proof of concept
2. Review the result before proceeding to the other 5 files
3. Migrate remaining 5 files in one batch
4. Update any cross-references between domain files
5. Run `node scripts/report.js` to verify MASTER-INDEX.md still generates correctly
