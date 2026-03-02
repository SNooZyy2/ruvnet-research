# Option C: Lead Preprocessor — Build Plan

## Goal

Create a preprocessing step that sits between the leads doc and the existing research pipeline. It cross-references leads against the research DB to produce a **verification plan** — a prioritized, deduplicated list of files to read, organized as a normal research session plan.

## Why This Is Needed

The leads doc (`leads/live-february-26-leads.md`) contains 22 leads with claims like "ruvector uses CRDT consensus." But:
- We may already have findings covering this from R44 (libp2p), R38 (distributed graph)
- The leads reference modules/crates, not exact file paths
- Some leads may contradict existing findings (which makes them HIGH priority)
- Without preprocessing, reader agents would waste tokens re-reading files already at DEEP depth

The preprocessor resolves file paths, checks existing coverage, and outputs a clean plan.

## Files to Create/Edit

### 1. NEW: `ruv-vods/agents/lead-verifier.md`

**Type:** Agent template (like reader.md, transcript-analyzer.md)
**subagent_type:** `general-purpose` (needs Read, Bash, Grep, Glob)
**model:** `opus`

**What it does:**
- Receives: path to a leads file, the research DB path
- For EACH lead in the file:

  **Step A — Resolve references to actual files:**
  - Extract file/module/crate names from the lead's `Referenced:` field
  - Query the DB: `SELECT id, relative_path, depth, loc, package_id FROM files WHERE relative_path LIKE '%{term}%'`
  - Also grep the repos for the referenced terms if DB lookup fails
  - Record: which files exist, which are NOT_TOUCHED vs DEEP, which don't exist at all

  **Step B — Check existing coverage:**
  - For each resolved file, query: `SELECT severity, category, description FROM findings WHERE file_id = ? ORDER BY severity`
  - Check if existing findings already cover the lead's claim
  - Flag contradictions (lead says X, finding says NOT X) as CRITICAL

  **Step C — Classify the lead:**
  - `ALREADY_COVERED` — existing DEEP file + findings fully address this claim
  - `PARTIALLY_COVERED` — file exists at DEEP/MEDIUM but findings don't address this specific claim
  - `NEW` — referenced files are NOT_TOUCHED/SURFACE or don't exist in DB
  - `CONTRADICTION` — lead contradicts existing findings (highest priority)

  **Step D — Build verification entry:**
  For non-ALREADY_COVERED leads, output:
  ```
  ## LEAD-{NNN}: {short description}
  Classification: NEW | PARTIALLY_COVERED | CONTRADICTION
  Original claim: "{claim from leads doc}"

  ### Files to read:
  | File | Package | Current Depth | LOC | Action |
  |------|---------|--------------|-----|--------|
  | path/to/file.rs | ruvector-rust | NOT_TOUCHED | 450 | DEEP read, check for {specific thing} |
  | path/to/other.rs | ruvector-rust | MEDIUM | 200 | Re-read lines X-Y for {claim} |

  ### Existing findings that relate:
  - Finding #{id}: "{description}" (file: {path}, severity: {sev})
  - ...or "None found"

  ### What to verify:
  - Specific question 1 (e.g., "Does delta_consensus.rs implement vector clock sharding?")
  - Specific question 2

  ### Suggested research agent: reader | facade-detector | cross-repo-tracer
  ```

**Output:** Writes the full verification plan to `ruv-vods/leads/{name}-verification-plan.md`

**Key DB queries the agent will run:**

```sql
-- Find files matching a term
SELECT f.id, f.relative_path, f.depth, f.loc, p.name as package
FROM files f JOIN packages p ON f.package_id = p.id
WHERE f.relative_path LIKE '%{term}%'
AND f.depth != 'EXCLUDED';

-- Check findings for a file
SELECT severity, category, description
FROM findings WHERE file_id = ?
ORDER BY CASE severity WHEN 'CRITICAL' THEN 1 WHEN 'HIGH' THEN 2 WHEN 'MEDIUM' THEN 3 ELSE 4 END;

-- Check if a crate/module directory has been analyzed
SELECT f.relative_path, f.depth, f.loc
FROM files f JOIN packages p ON f.package_id = p.id
WHERE f.relative_path LIKE '%{crate_or_module}%'
ORDER BY f.depth, f.loc DESC;

-- Find related findings by keyword
SELECT f2.relative_path, fi.severity, fi.description
FROM findings fi
JOIN files f2 ON fi.file_id = f2.id
WHERE fi.description LIKE '%{keyword}%';
```

**Template structure (sections):**
1. Purpose (2-3 lines)
2. Receive assignment (inputs)
3. Parse leads file (extract structured data from each LEAD block)
4. For each lead: resolve → check coverage → classify → build entry
5. Write verification plan
6. Write summary stats
7. Schema constraints reminder (copy from reader.md)
8. Success criteria

### 2. EDIT: `ruv-vods/CLAUDE.md`

Add between Step 6 (Consolidate) and Step 7 (Verify):

```
### Step 6b: Preprocess leads (cross-reference with research DB)

Spawn a lead-verifier agent (opus) with the leads file. It queries the research
DB to resolve file paths, check existing coverage, and classify each lead.

Output: `leads/{name}-verification-plan.md`

Update `index.json` status to `preprocessed`.
```

Update the Agent Registry table to add:

```
| PREPROCESS | `agents/lead-verifier.md` | `general-purpose` | opus | DB cross-reference + plan generation |
```

Update the lifecycle diagram:

```
inbox → chunked → scanned → analyzed → preprocessed → verified
```

### 3. EDIT: `ruv-vods/README.md`

Add a "Step 4b: Preprocess" section between Consolidate and Verify in the Quick Start:

```
### Step 4b: Preprocess leads

Before verification, cross-reference leads against the research DB:

1. Read `agents/lead-verifier.md`
2. Spawn one opus agent with the leads file path
3. Agent queries research DB for file matches, existing findings, depth coverage
4. Outputs `leads/{name}-verification-plan.md`

The verification plan categorizes each lead as:
- ALREADY_COVERED — skip, existing research handles it
- PARTIALLY_COVERED — re-read specific sections of known files
- NEW — file needs first read or doesn't exist in DB
- CONTRADICTION — lead contradicts existing finding (highest priority)

Take the verification plan to ruvnet-research/ and run a normal research session.
```

Update the File Index table:

```
| `lead-verifier.md` | Cross-references leads against research DB | `ruv-vods/agents/lead-verifier.md` |
```

### 4. EDIT: `ruv-vods/index.json` schema

The `status` field gains a new valid value: `preprocessed`

No schema change needed (it's just a string), but document it.

### 5. NO CHANGES to parent project

The verification plan output is just a markdown file that a human reads and uses to plan a normal research session in `ruvnet-research/`. No changes needed to the parent CLAUDE.md, reader.md, or any research infrastructure.

## Execution Order

1. Create `agents/lead-verifier.md` (the bulk of the work, ~150-200 lines)
2. Edit `CLAUDE.md` (add Step 6b, update registry, update lifecycle)
3. Edit `README.md` (add Step 4b, update file index)
4. Test: run the lead-verifier on `leads/live-february-26-leads.md`

## How to Spawn

From `ruv-vods/`:

```
1. Read agents/lead-verifier.md
2. Task(
     subagent_type="general-purpose",
     model="opus",
     prompt="<full contents of agents/lead-verifier.md>

     Assignment:
     - Leads file: /home/snoozyy/ruvnet-research/ruv-vods/leads/live-february-26-leads.md
     - Research DB: /home/snoozyy/ruvnet-research/db/research.db
     - Output: /home/snoozyy/ruvnet-research/ruv-vods/leads/live-february-26-verification-plan.md"
   )
```

## Expected Output Example

```markdown
# Verification Plan: live-february-26

Generated: 2026-03-01
Source: leads/live-february-26-leads.md (22 leads)

## Summary
- ALREADY_COVERED: 4 leads (skip)
- PARTIALLY_COVERED: 7 leads (targeted re-reads)
- NEW: 8 leads (full reads needed)
- CONTRADICTION: 3 leads (highest priority)

## Total files to read: ~18
## Estimated session: 1 deep-read swarm (5-9 reader agents)

---

## CONTRADICTION LEADS (verify first)

### LEAD-009: AgentDB real vector operations
Classification: CONTRADICTION
Original claim: "AgentDB performs real semantic vector search via ruvector integration"
Contradicts: R20 finding — EmbeddingService never initialized in claude-flow bridge
Files to read:
| File | Package | Depth | LOC | Action |
|------|---------|-------|-----|--------|
| src/vector/search.ts | agentdb | MEDIUM | 340 | Re-read, check if EmbeddingService now initialized |
...

## NEW LEADS (full reads needed)

### LEAD-003: RVF binary format
Classification: NEW
Original claim: "Copy-on-write binary with 7-bit quantization..."
Files to read:
| File | Package | Depth | LOC | Action |
|------|---------|-------|-----|--------|
| crates/rvf/src/format.rs | ruvector-rust | NOT_TOUCHED | 612 | DEEP read |
| crates/rvf/src/quantize.rs | ruvector-rust | NOT_TOUCHED | 287 | DEEP read |
...
```

## Design Notes

- The lead-verifier agent does NOT write to the research DB — it only reads
- The verification plan is a human-reviewed document, not an automated pipeline
- CONTRADICTION leads go first because they may change existing realness scores
- File resolution uses fuzzy matching (LIKE '%term%') because transcripts rarely give exact paths
- The agent should also check `exclude_paths` to avoid recommending reads on excluded files
- If a lead references something that doesn't exist in any repo, mark it as `UNRESOLVABLE` with a note
