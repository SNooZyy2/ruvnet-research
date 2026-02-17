# ADR-041: Research Workflow Changes for Topical Synthesis

**Status**: IMPLEMENTED
**Date**: 2026-02-15
**Depends on**: ADR-040 (Restructure Synthesis Documents)
**Relates to**: ADR-038 (Research Database System)

---

## Context

ADR-040 restructures the existing synthesis documents from chronological session logs into topical synthesis documents. This ADR addresses the **process changes** needed so that future research sessions produce topical updates instead of reverting to the old append-only pattern.

The current workflow (defined in `CLAUDE.md` Session Protocol and agent prompts) instructs agents to:
1. Append a new session block with file tables and narrative
2. Append "Updated CRITICAL Findings (+N = M total)" cumulative lists
3. Append "Updated Positive" cumulative lists

This is what created the structural problems ADR-040 addresses. Without workflow changes, the first session after migration would re-introduce the old pattern.

---

## Decision

### 1. Synthesis Update Protocol (replaces "Write/Update Domain Synthesis" in CLAUDE.md)

When a session produces findings for a domain, the synthesizer agent MUST update the existing synthesis document **in-place** rather than appending a new session block:

#### Section 2 (File Registry)
- **ADD** new rows for newly deep-read files
- **UPDATE** existing rows if a file was re-read at deeper depth or has a revised Real%
- Do NOT duplicate rows

#### Section 3 (Findings Registry)
- **ADD** new findings with the next sequential ID in the appropriate severity subsection
- **MARK** findings as `SUPERSEDED by R{n}: {reason}` if a later session contradicts them
- **MARK** findings as `RESOLVED in R{n}` if the issue has been fixed upstream
- NEVER re-list or re-number existing findings
- NEVER create "Updated CRITICAL Findings (+N = M total)" sections

#### Section 4 (Positives Registry)
- **ADD** new positives with session tag
- NEVER re-list existing positives

#### Section 5 (Subsystem Sections)
- **UPDATE** existing subsystem sections with new information
- **CREATE** new subsystem sections for newly discovered subsystems
- Write in present tense ("X uses Y" not "R41 revealed that X uses Y")
- Include session references inline for provenance: `(R41)` or `(confirmed R38, updated R41)`

#### Section 8 (Session Log)
- **APPEND** a 2-5 line entry for the current session:
  ```
  ### R{n} ({date}): {1-sentence focus}
  {file_count} files, {loc} LOC, {finding_count} findings.
  {1-sentence key discovery or correction.}
  ```

#### Section 1 (Current State Summary)
- **REWRITE** this section to reflect the domain's current state after all updates
- This is the only section that gets fully rewritten each session (it's only 20-30 lines)

### 2. Agent Prompt Changes

The following agent prompts need modification:

#### `agents/synthesizer.md`

Remove instructions to:
- Create new session header sections (`## R{n}: Title (Session N)`)
- Append cumulative findings lists
- Append cumulative positives lists

Add instructions to:
- Read the existing synthesis document structure BEFORE writing
- Update the File Registry table (add/modify rows)
- Add findings to the Findings Registry (append to severity subsections)
- Update or create subsystem sections with new content
- Append a brief entry to the Session Log
- Rewrite the Current State Summary

#### `agents/reader.md`

No structural changes needed. Reader agents produce findings and DB updates — they don't write synthesis documents directly. Their output is consumed by the synthesizer.

#### `agents/facade-detector.md`

No structural changes. Findings go to DB. If the facade-detector writes directly to synthesis docs, add the same in-place update instructions as synthesizer.

#### `agents/realness-scorer.md`

When updating domain scores, update:
- Section 1 (Current State Summary) with new realness %
- Section 2 (File Registry) Real% column for re-scored files
- Do NOT append a new scoring section

### 3. CLAUDE.md Changes

#### Session Protocol Section 4 ("Write/Update Domain Synthesis")

Replace the current text:
```
Create or update `domains/{domain-name}/analysis.md` with:
- Overview of domain purpose
- Key files and their roles
- Architecture patterns
- Findings (grouped by severity)
- Cross-domain dependencies
- Knowledge gaps requiring deeper analysis
```

With:
```
Update `domains/{domain-name}/analysis.md` IN-PLACE following the
canonical structure (ADR-040):
- Section 1: Rewrite Current State Summary (20-30 lines, present tense)
- Section 2: Add/update rows in File Registry table
- Section 3: Add new findings to Findings Registry (never re-list old ones)
- Section 4: Add new positives (never re-list old ones)
- Section 5: Update or create subsystem sections with new content
- Section 8: Append 2-5 line session log entry
NEVER create "Updated CRITICAL Findings (+N = M total)" sections.
NEVER append a new chronological session block.
```

#### Query Recipe #9 ("Add finding")

No change — DB insert is unaffected. But add a note:
```
-- After inserting to DB, update the domain synthesis doc's
-- Findings Registry (Section 3) with the new finding.
-- Do NOT create cumulative finding lists.
```

### 4. Swarm Composition Changes

#### Domain Synthesis Swarm

Current trigger: "synthesize domain", "write analysis", "update synthesis"

The synthesizer agent in this swarm needs the updated prompt. No change to swarm composition (synthesizer + mapper + realness-scorer), only to the synthesizer's instructions.

#### Full Session Swarm

Phase 5 (Synthesize) uses the synthesizer agent. Same prompt update applies.

### 5. Validation Checks

After each synthesis update, verify:

1. **No cumulative lists**: Grep for "Updated CRITICAL Findings" or "Updated HIGH Findings" — should return 0 matches in the updated file
2. **No session header blocks**: Grep for `## R\d+:` outside Section 8 — should return 0 matches
3. **File Registry is sorted**: Rows in Section 2 should be sorted by package then path
4. **Finding IDs are sequential**: IDs in each severity subsection should be monotonically increasing
5. **Session Log is chronological**: Entries in Section 8 should be in ascending R{n} order

These checks can be added to `report.js` or run as a post-synthesis hook.

---

## What This ADR Does NOT Cover

- The actual migration of existing files (ADR-040)
- Database schema changes (none needed)
- Changes to `report.js` output format
- Changes to MEMORY.md structure (separate concern)

---

## Risks

1. **Agent drift**: Without enforcement, agents may revert to appending session blocks. Mitigated by:
   - Updated prompt templates with explicit anti-patterns ("NEVER create cumulative lists")
   - Optional post-synthesis validation hook
   - Reviewer agent can check synthesis doc structure

2. **Merge conflicts**: In-place edits to tables and lists are more conflict-prone than appending. Mitigated by:
   - Only one synthesizer per domain per session (existing rule)
   - Sequential synthesis after parallel reads (existing rule)

3. **Provenance degradation**: With topical structure, it's harder to reconstruct "what exactly happened in R37". Mitigated by:
   - Session Log appendix preserves high-level provenance
   - Inline `(R{n})` tags on all content
   - DB retains full session-level detail via `file_reads` and `findings` tables

---

## Implementation Order

1. ADR-040 migration completes first (restructure existing content)
2. Update `agents/synthesizer.md` prompt template
3. Update `CLAUDE.md` Session Protocol section
4. Optionally add validation checks to `report.js`
5. First session after migration validates the new workflow produces correct structure
