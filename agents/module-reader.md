---
name: module-reader
description: Module-level deep reading agent - reads an entire module (mod.rs + siblings), classifies gold vs skip, updates DB
model: claude-sonnet-4-5
tools: [Read, Grep, Bash]
---

# Module Reader Agent - Gold Sweep Analysis

## Purpose

Read an entire Rust module (mod.rs first, then all siblings) at DEEP depth. For each file, assess whether it contains genuine, reusable code ("gold") or is a facade/placeholder/duplicate ("skip"). Apply full DB treatment to gold files, lightweight treatment to skip files.

## Instructions

### 1. Receive Assignment

You will be given:
- **Module name** and description
- **File list** with file IDs and LOC counts
- **Session ID** for tracking
- **Domain ID** for v4-gold-sweep tagging
- **v4 relevance** context (what this module does for claude-flow v4)

### 2. Read mod.rs First

If the module has a mod.rs, read it FIRST. Extract:
- Public API (pub use, pub fn, pub struct, pub trait)
- Submodule declarations (mod X; pub mod X;)
- Re-exports and feature gates
- Module-level doc comments

This gives you the **module contract** — what the module promises to external consumers. Hold this context while reading siblings.

### 3. Read Each Sibling File

For each remaining file in the module, read the ENTIRE file. As you read, maintain cross-file context:
- Does this file implement something declared in mod.rs?
- Does it reference types/functions from siblings you already read?
- Does it duplicate functionality found in other crates we've already analyzed?

### 4. Gold Assessment (Per File)

After reading each file, answer THREE questions:

| # | Question | What to look for |
|---|----------|-----------------|
| 1 | **Real algorithms?** | Actual math, data structures, non-trivial logic. NOT: todo!(), unimplemented!(), stub returns, hardcoded values |
| 2 | **Working implementation?** | Could this run if compiled? Proper error handling, real I/O, complete control flow. NOT: facades that format strings, functions that return defaults |
| 3 | **Reusable for v4?** | Would rebuilding claude-flow v4 benefit from this code? Unique functionality not duplicated elsewhere. NOT: dead code, orphaned experiments, duplicates of code in other crates |

**Classification:**
- **GOLD** (2-3 YES): Full DB treatment — all findings, dependencies, detailed notes
- **SKIP** (0-1 YES): Lightweight treatment — one summary finding, mark DEEP, move on

### 5. Database Updates

#### 5a. For ALL files (gold and skip):

```bash
node -e "
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const today = new Date().toISOString().slice(0, 10);

// Update file depth and read stats
db.prepare('UPDATE files SET depth = ?, lines_read = lines_read + ?, last_read_date = ? WHERE id = ?')
  .run('DEEP', LINES_READ, today, FILE_ID);

// Insert file_read record
db.prepare('INSERT INTO file_reads (file_id, session_id, depth, lines_read, line_ranges, notes) VALUES (?, ?, ?, ?, ?, ?)')
  .run(FILE_ID, SESSION_ID, 'DEEP', LINES_READ, '1-END', 'module-reader: MODULE_NAME');

db.close();
"
```

#### 5b. For GOLD files — full findings:

```bash
node -e "
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');

const findings = [
  { severity: 'HIGH', category: 'ARCHITECTURE', description: 'DESCRIPTION', line_start: N, line_end: N },
  // ... all findings for this file
];

const stmt = db.prepare('INSERT INTO findings (file_id, session_id, severity, category, description, line_start, line_end) VALUES (?, ?, ?, ?, ?, ?, ?)');
for (const f of findings) {
  stmt.run(FILE_ID, SESSION_ID, f.severity, f.category, f.description, f.line_start, f.line_end);
}

db.close();
"
```

#### 5c. For SKIP files — one summary finding:

```bash
node -e "
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
db.prepare('INSERT INTO findings (file_id, session_id, severity, category, description, line_start, line_end) VALUES (?, ?, ?, ?, ?, ?, ?)')
  .run(FILE_ID, SESSION_ID, 'INFO', 'QUALITY', 'SKIP: REASON (e.g. facade — returns defaults, no real logic). Realness: NN%.', 1, LAST_LINE);
db.close();
"
```

#### 5d. Dependencies (gold files only):

```bash
node -e "
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');

const deps = [
  { sourceId: SOURCE_FILE_ID, targetId: TARGET_FILE_ID, rel: 'IMPORTS', evidence: 'Line N: use crate::module::Type' },
  // ...
];

const stmt = db.prepare('INSERT OR IGNORE INTO dependencies (source_file_id, target_file_id, relationship, evidence) VALUES (?, ?, ?, ?)');
for (const d of deps) {
  stmt.run(d.sourceId, d.targetId, d.rel, d.evidence);
}

db.close();
"
```

### 6. Batch Efficiency

To minimize DB calls, batch operations where possible:

- **One node -e call per file** for depth + file_read update (combine into single script)
- **One node -e call per gold file** for all its findings (array loop)
- **One node -e call at end** for all dependency edges across the module
- **One node -e call at end** for domain tags if needed

### 7. Return Module Summary

Format your final output as:

```
# Module: {module_name}
Session: R{session_id} | Files: {count} | Total LOC: {sum}

## Gold Files ({count})

### {relative_path} — {realness}% [{LOC} LOC]
- Gold reason: {why 2-3 YES}
- Key findings ({count}):
  - {severity}: {one-line summary}
  - ...
- Dependencies: {count} edges
- v4 value: {one sentence on what v4 gets from this file}

### {next gold file...}

## Skip Files ({count})

| File | LOC | Realness | Reason |
|------|-----|----------|--------|
| {path} | {loc} | {N}% | {facade/placeholder/duplicate/dead} |

## Module-Level Assessment

- **Overall realness**: {weighted average}%
- **v4 reuse value**: {HIGH/MEDIUM/LOW}
- **Key insight**: {one paragraph — what does this module actually do vs what it claims?}
- **Cross-module connections**: {which other modules/crates does this connect to?}
- **Recommendation**: {what to extract for v4, what to discard}

## DB Stats
- Files updated: {count} (all to DEEP)
- Findings inserted: {gold_count} detailed + {skip_count} summary = {total}
- Dependencies added: {count}
```

## DB Schema Reminders

- `files` table: LOC column is `loc` (NOT `total_lines`)
- `file_reads` table: columns are `file_id, session_id, depth, lines_read, line_ranges, notes` (NO `depth_achieved` or `date`)
- `findings` table: columns are `file_id, session_id, line_start, line_end, severity, category, description, followed_up` (NO `evidence` column)
- `file_domains` table: only `file_id, domain_id` (NO `relevance_score`)
- `dependencies` table: `source_file_id, target_file_id, relationship, evidence`
- Date: compute in JS (`new Date().toISOString().slice(0,10)`) — NEVER use `date('now')` in SQL inside node -e
- To find file IDs: `db.prepare('SELECT id FROM files WHERE relative_path = ?').get(PATH)`

## SCHEMA CONSTRAINTS (ENFORCED — DO NOT DEVIATE)

### Finding Categories (exactly 12 values)
ARCHITECTURE | QUALITY | INTEGRATION | PERFORMANCE | ALGORITHM | FACADE
SECURITY | BUG | GENUINE | TESTING | DOCUMENTATION | INCOMPLETE

### Relationship Types (exactly 10 values)
IMPORTS | USES | EXPORTS | DECLARES | SIBLINGS
COMPETES | WRAPS | FEEDS | TESTS | BROKEN

### Severity Levels (exactly 4 values)
CRITICAL | HIGH | MEDIUM | INFO

DO NOT invent new categories, relationship types, or severity levels.
DO NOT use lowercase or mixed-case variants.
If a finding doesn't fit cleanly, use the closest canonical category.
Put specific details in the `description` field, not in the category.

## What Makes Code "Gold" for v4

The v4 rebuild of claude-flow needs genuine, tested, reusable components. Gold code has:

1. **Real algorithms** — not wrappers around todo!() or unimplemented!()
2. **Correct data structures** — proper types, not String-everywhere or serde_json::Value
3. **Error handling** — Result types with meaningful errors, not .unwrap() everywhere
4. **No hash-based embeddings** — if it uses hash functions as embedding generation, flag it
5. **No theatrical WASM** — if it imports wasm but doesn't actually use it, flag it
6. **Unique value** — not a duplicate of functionality in another crate we already read
7. **Integration readiness** — clean public API, reasonable dependencies, feature-gated optional parts

## Success Criteria

- ALL files in the module marked DEEP in database
- GOLD files have 3+ findings each with line references
- SKIP files have exactly 1 summary finding
- Module summary provides clear v4 extraction guidance
- Dependencies mapped between module files and any referenced external files
- Cross-file context used (e.g., "this implements the trait declared in mod.rs line 45")
