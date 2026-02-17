# Research Mode - Claude Flow V3 Analysis Project

## Project Overview

This is a research project analyzing the ruvnet multi-repo universe (claude-flow, agentic-flow, agentdb, ruvector).

**Source of Truth**: `research/db/research.db` (SQLite via better-sqlite3)
**Synthesis Documents**: `research/domains/*/analysis.md`
**Auto-Generated Index**: `MASTER-INDEX.md` (never edit manually)
**Reference**: `research/ADR-038-research-database-system.md`

## Behavioral Rules

- **CRITICAL: NEVER spawn a research Task agent without first reading its template from `agents/`** — see Agent Registry below. The template MUST be injected as the prompt prefix. Bare prompts are forbidden.
- ALWAYS update the database when reading files (depth, lines_read, last_read_date)
- ALWAYS insert findings with appropriate severity (CRITICAL/HIGH/MEDIUM/INFO) and category
- ALWAYS tag files with domains via file_domains junction table
- ALWAYS insert dependency edges when discovering cross-file relationships
- NEVER edit MASTER-INDEX.md directly — run `node scripts/report.js` instead
- New synthesis docs go in `domains/{domain-name}/analysis.md`
- Use `better-sqlite3` for all DB operations (available at ~/node_modules/)
- Follow depth classification system strictly

## Database Interaction

All database operations use better-sqlite3. Use inline Node.js via Bash tool:

```bash
node -e "
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const rows = db.prepare('SELECT * FROM smart_priority_gaps WHERE tier_rank <= 2 LIMIT 10').all();
console.log(JSON.stringify(rows, null, 2));
db.close();
"
```

## Query Recipes

### 1. What should I read next? (smart, relevance-tiered)
```sql
SELECT * FROM smart_priority_gaps WHERE tier_rank <= 2 LIMIT 10;
```
Tiers: CONNECTED (has dep to DEEP file), OWN_CODE (custom-src), PROXIMATE (3+ DEEP in same dir), NEARBY (1-2 DEEP), DOMAIN_ONLY (tag only).
Use `tier_rank <= 2` for high-signal files, or `LIMIT 10` without filter for all tiers.

### 1b. What should I read next? (raw, unranked)
```sql
SELECT * FROM priority_gaps LIMIT 10;
```

### 2. Coverage by domain
```sql
SELECT * FROM domain_coverage;
```

### 3. Coverage by package
```sql
SELECT * FROM package_coverage;
```

### 4. Files in domain X
```sql
SELECT f.* FROM files f
JOIN file_domains fd ON f.id = fd.file_id
JOIN domains d ON fd.domain_id = d.id
WHERE d.name = 'domain-name';
```

### 5. All CRITICAL findings
```sql
SELECT * FROM open_findings WHERE severity = 'CRITICAL';
```

### 6. Integration hotspots
```sql
SELECT * FROM integration_hotspots;
```

### 7. Dependency chain from file X
```sql
WITH RECURSIVE chain AS (
  SELECT d.*, 0 as depth FROM dependencies d
  JOIN files f ON d.source_file_id = f.id
  WHERE f.relative_path = 'target/file.js'
  UNION ALL
  SELECT d.*, c.depth + 1 FROM dependencies d
  JOIN chain c ON d.source_file_id = c.target_file_id
  WHERE c.depth < 5
)
SELECT sf.relative_path as source, tf.relative_path as target, chain.relationship
FROM chain
JOIN files sf ON chain.source_file_id = sf.id
JOIN files tf ON chain.target_file_id = tf.id;
```

### 8. Update file after reading
```js
// Compute date in JS to avoid quoting issues in node -e
const today = new Date().toISOString().slice(0, 10);

// Insert read record
db.prepare('INSERT INTO file_reads (file_id, session_id, depth, lines_read, line_ranges, notes) VALUES (?, ?, ?, ?, ?, ?)').run(fileId, sessionId, depth, linesRead, lineRanges, notes);

// Update file metadata
db.prepare('UPDATE files SET depth = ?, lines_read = lines_read + ?, last_read_date = ? WHERE id = ?').run(depth, linesRead, today, fileId);
```

### 9. Add finding
```sql
INSERT INTO findings (file_id, session_id, severity, category, description, evidence, line_ref)
VALUES (?, ?, ?, ?, ?, ?, ?);
-- After inserting to DB, update the domain synthesis doc's
-- Findings Registry (Section 3) with the new finding.
-- Do NOT create cumulative finding lists.
```

### 10. Tag file with domain
```sql
INSERT OR IGNORE INTO file_domains (file_id, domain_id)
VALUES (?, ?);
```

### 11. Add dependency
```sql
INSERT INTO dependencies (source_file_id, target_file_id, relationship, evidence)
VALUES (?, ?, ?, ?);
```

### 12. Check package connectivity
```sql
SELECT * FROM package_connectivity ORDER BY total_cross_deps ASC;
```

### 13. Find isolated subtrees (Goalie-like islands)
```sql
SELECT * FROM subtree_connectivity
WHERE outbound_cross_deps = 0 AND inbound_cross_deps = 0 AND untouched >= 10
ORDER BY total_loc DESC;
```

### 14. List/add exclusion patterns
```sql
-- List current exclusions
SELECT * FROM exclude_paths;
-- Add new exclusion (priority_gaps view auto-filters these)
INSERT OR IGNORE INTO exclude_paths (pattern, reason, added_date)
VALUES ('%pattern%', 'reason why excluded', '2026-02-16');
```

## Session Protocol

### 1. Start Session
```bash
node -e "
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const today = new Date().toISOString().slice(0, 10);
const result = db.prepare('INSERT INTO sessions (name, date, focus) VALUES (?, ?, ?)').run('session-2026-02-14', today, 'focus description');
console.log('Session ID:', result.lastInsertRowid);
db.close();
"
```

### 2. Check Priorities
```bash
node -e "
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const gaps = db.prepare('SELECT * FROM smart_priority_gaps WHERE tier_rank <= 2 LIMIT 10').all();
console.log(JSON.stringify(gaps, null, 2));
db.close();
"
```

### 2b. Check for Isolated Subtrees (MANDATORY before planning)
Before selecting files from smart_priority_gaps, check for disconnected islands that should be excluded.
Files in isolated subtrees have zero cross-package dependencies and waste research time.

Only trust RELIABLE confidence — LOW_CONFIDENCE and NO_DATA mean we haven't recorded enough
dependencies to judge. When confidence is low, do NOT auto-exclude; instead investigate first.
```bash
node -e "
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
// RELIABLE islands are safe to exclude without investigation
const reliable = db.prepare(\`
  SELECT * FROM subtree_connectivity
  WHERE outbound_cross_deps = 0 AND inbound_cross_deps = 0
    AND untouched >= 10 AND confidence = 'RELIABLE'
  ORDER BY total_loc DESC LIMIT 10
\`).all();
console.log('RELIABLE isolated (safe to exclude):');
console.log(JSON.stringify(reliable, null, 2));
// LOW_CONFIDENCE/NO_DATA need investigation before excluding
const suspect = db.prepare(\`
  SELECT * FROM subtree_connectivity
  WHERE outbound_cross_deps = 0 AND inbound_cross_deps = 0
    AND untouched >= 10 AND confidence != 'RELIABLE'
  ORDER BY total_loc DESC LIMIT 10
\`).all();
console.log('SUSPECT (need investigation before excluding):');
console.log(JSON.stringify(suspect, null, 2));
db.close();
"
```
If a RELIABLE subtree is isolated AND not core to claude-flow's runtime, add it to `exclude_paths`:
```bash
node -e "
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const today = new Date().toISOString().slice(0, 10);
db.prepare('INSERT OR IGNORE INTO exclude_paths (pattern, reason, added_date) VALUES (?, ?, ?)').run('%pattern%', 'reason', today);
db.close();
"
```
Also check that selected files from smart_priority_gaps are not in known-isolated packages:
```sql
SELECT * FROM package_connectivity WHERE connectivity = 'ISOLATED';
```

### 3. Research Loop
- Read files from priority queue
- Update database (depth, lines_read, findings, domains, dependencies)
- Write discoveries to domain synthesis documents
- Insert file_reads records for session history

### 4. Write/Update Domain Synthesis
Update `domains/{domain-name}/analysis.md` IN-PLACE following the
canonical structure (ADR-040). See `domains/memory-and-learning/analysis.md` for the reference example.
- Section 1: Rewrite Current State Summary (20-30 lines, present tense)
- Section 2: Add/update rows in File Registry table
- Section 3: Add new findings to Findings Registry (never re-list old ones)
- Section 4: Add new positives (never re-list old ones)
- Section 5: Update or create subsystem sections with new content
- Section 8: Append 2-5 line session log entry
NEVER create "Updated CRITICAL Findings (+N = M total)" sections.
NEVER append a new chronological session block.

### 5. End Session
```bash
node /home/snoozyy/ruvnet-research/scripts/report.js
```

This regenerates MASTER-INDEX.md with updated statistics.

## Depth Classification

### DEEP (50%+ read)
- Algorithms traced line-by-line
- Exact code snippets and line references extracted
- Data flow verified through the entire pipeline
- Edge cases and error handling documented
- Performance characteristics understood

### MEDIUM (20-50% read)
- Architecture and design patterns understood
- Key functions and classes mapped
- Some execution paths untraced
- Major data structures identified
- Cross-file interactions noted

### SURFACE (0-20% read)
- Glob/grep only
- Categorized by filename and directory structure
- Minimal content reading (0-2 lines)
- Basic purpose inferred from naming

### MENTIONED
- Referenced in another file's analysis
- Never directly opened or read
- Known only through cross-references

### NOT_TOUCHED
- Discovered in Phase 0 filesystem scan
- Zero analysis performed
- Waiting in priority queue

## Domain Management

### List Domains
```sql
SELECT * FROM domains ORDER BY priority, name;
```

### Create Domain
```bash
node -e "
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
db.prepare('INSERT INTO domains (name, priority, description) VALUES (?, ?, ?)').run('domain-name', 1, 'description');
db.close();
"
mkdir -p /home/snoozyy/ruvnet-research/domains/domain-name
```

### Tag File with Domain
```bash
node -e "
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const fileId = db.prepare('SELECT id FROM files WHERE relative_path = ?').get('path/to/file.js').id;
const domainId = db.prepare('SELECT id FROM domains WHERE name = ?').get('domain-name').id;
db.prepare('INSERT OR IGNORE INTO file_domains (file_id, domain_id, relevance_score) VALUES (?, ?, ?)').run(fileId, domainId, 0.9);
db.close();
"
```

## File Organization

```
research/
├── db/                      # Database files
│   ├── research.db          # Main SQLite database
│   └── schema.sql           # Schema definition
├── scripts/                 # Automation scripts
│   ├── report.js            # Generate MASTER-INDEX.md
│   └── backup.sh            # Database backup
├── domains/                 # Synthesis documents
│   ├── {domain-name}/
│   │   └── analysis.md      # Domain analysis
├── agents/                  # Agent prompt templates
│   ├── scanner.md           # Phase 0 inventory agent
│   ├── reader.md            # Deep reading agent
│   ├── synthesizer.md       # Cross-domain synthesis agent
│   ├── mapper.md            # Dependency mapping agent
│   ├── facade-detector.md   # Stub/facade/placeholder detection
│   ├── cross-repo-tracer.md # Systemic pattern tracing across repos
│   └── realness-scorer.md   # Weighted realness % computation
└── ADR-038-*.md             # Architecture decision record

```

## Research Swarm Protocol

### OVERRIDE: Research sessions ignore generic swarm rules

When executing a research plan (any file in `daily-plan/`), the following rules from the main `~/CLAUDE.md` are **overridden**:

1. **Do NOT run `claude-flow swarm init`** — research agents are spawned directly via Task tool, not through CLI swarm coordination. The swarm init step primes the orchestrator toward generic agent types (`v3-coder`, `v3-researcher`) which are WRONG for research.
2. **Do NOT use generic agent types** — never use `v3-coder`, `v3-researcher`, `v3-tester`, or any other generic Task subagent_type for research file reads. ALWAYS use the Agent Registry table below.
3. **The `"spawn swarm"` trigger phrase** in research context means: read the plan, create a session, read the `reader.md` template, and spawn one `Bash` agent per file. It does NOT mean `claude-flow swarm init`.
4. **Spawn per-file, not per-cluster** — each file in the plan gets its own agent. Never group an entire cluster into a single agent.
5. **Always inject the agent template** — read `agents/reader.md` and include its full contents in the Task prompt, followed by the file-specific assignment.

### Agent Registry

Each research agent `.md` file is a prompt template. To use it in a swarm, read its contents and inject them into a Task tool call with the appropriate `subagent_type`.

| Research Agent | Task subagent_type | Model | Why |
|---------------|-------------------|-------|-----|
| `reader.md` | `Bash` | sonnet | Needs Bash for DB writes via `node -e` |
| `scanner.md` | `Bash` | haiku | Simple filesystem walk + DB inserts |
| `synthesizer.md` | `general-purpose` | sonnet | Needs Read + Write + Bash for docs + DB |
| `mapper.md` | `Bash` | sonnet | Needs Read + Grep + Bash for tracing |
| `facade-detector.md` | `Bash` | sonnet | Needs Read + Bash for analysis + DB |
| `cross-repo-tracer.md` | `Bash` | sonnet | Needs Grep across repos + Bash for DB |
| `realness-scorer.md` | `Bash` | haiku | Mostly DB queries, math, reporting |

### Spawning a Research Agent

To invoke a research agent in a swarm:

1. Read the agent `.md` file to get its prompt template
2. Spawn a Task with the mapped `subagent_type` and `model`
3. Pass file-specific parameters (file path, session ID, etc.) in the prompt
4. Use `run_in_background: true` for swarm parallelism

Example:
```
Task(subagent_type="Bash", model="sonnet", run_in_background=true,
     prompt="<contents of reader.md>\n\nAssignment:\n- File: {path}\n- File ID: {id}\n- Session ID: {sid}\n- Target depth: DEEP")
```

### Research Swarm Compositions

These are standard agent combinations for common research workflows. When a trigger phrase is detected, spawn ALL listed agents in ONE message with `run_in_background: true`.

#### 1. Deep-Read Swarm (trigger: "deep read", "read files", "analyze files")

For reading a batch of files from the priority queue:

| Agent | Count | Role |
|-------|-------|------|
| reader | 3-5 | One per file, reads + inserts findings + updates DB |
| facade-detector | 1-2 | Runs on files suspected of being facades |
| mapper | 1 | Traces dependencies from all read files |

Spawn readers in parallel (one per file), then mapper after readers complete.

#### 2. Pattern Trace Swarm (trigger: "trace pattern", "find systemic", "cross-repo")

For tracing a specific pattern across all repositories:

| Agent | Count | Role |
|-------|-------|------|
| cross-repo-tracer | 1 | Searches all 4 repos, classifies instances |
| facade-detector | 1-2 | Verifies suspect files found by tracer |
| reader | 1-2 | Deep-reads newly discovered files |

#### 3. Scoring Swarm (trigger: "score realness", "compute scores", "quality report")

For computing realness scores across crates or domains:

| Agent | Count | Role |
|-------|-------|------|
| realness-scorer | 1 | Queries DB, computes scores, generates report |
| facade-detector | 1-2 | Re-checks low-scoring files for new facade evidence |

#### 4. Domain Synthesis Swarm (trigger: "synthesize domain", "write analysis", "update synthesis")

For writing or updating domain analysis documents:

| Agent | Count | Role |
|-------|-------|------|
| synthesizer | 1 | Writes/updates domain analysis.md |
| mapper | 1 | Refreshes cross-domain dependency data |
| realness-scorer | 1 | Computes domain-level realness for the doc |

#### 5. Full Session Swarm (trigger: "full session", "research session")

Complete research session workflow:

| Phase | Agents | Parallel? |
|-------|--------|-----------|
| 1. Plan | Query `smart_priority_gaps` (tier_rank <= 2), select files | Sequential |
| 2. Read | reader ×3-5 + facade-detector ×2 | Parallel |
| 3. Map | mapper ×1 | After phase 2 |
| 4. Score | realness-scorer ×1 | After phase 2 |
| 5. Synthesize | synthesizer ×1 | After phases 3-4 |

### Research Trigger Phrases

Add these to the existing trigger phrase table:

| Phrase | Swarm Composition | Action |
|--------|-------------------|--------|
| "deep read" | Deep-Read Swarm | Spawn readers + facade-detector + mapper |
| "trace pattern" | Pattern Trace Swarm | Spawn cross-repo-tracer + readers |
| "score realness" | Scoring Swarm | Spawn realness-scorer + facade-detector |
| "synthesize domain" | Domain Synthesis Swarm | Spawn synthesizer + mapper + scorer |
| "full session" | Full Session Swarm | All 5 phases sequentially |
| "facade check" | facade-detector ×1 | Single agent on specific file(s) |
| "cross-repo trace" | cross-repo-tracer ×1 | Single agent for pattern search |

### Swarm Rules for Research

- ALL research agents MUST receive `session_id` in their prompt
- ALL agents that write to DB MUST use `subagent_type: Bash` (not Explore/researcher — those lack Bash)
- ALWAYS spawn readers in parallel (one per file) — they don't conflict on DB writes
- NEVER spawn two synthesizers for the same domain — they'll overwrite each other
- Mapper should run AFTER readers complete (needs their findings for context)
- Realness-scorer should run AFTER readers complete (needs their findings data)
- When spawning 5+ agents, use `model: sonnet` for readers and `model: sonnet` for facade-detector/synthesizer

### DB Schema Reminders for Agents

These gotchas MUST be included in agent prompts to avoid errors:

- `files` table: LOC column is `loc` (NOT `total_lines`)
- `file_reads` table: columns are `file_id, session_id, depth, lines_read, line_ranges, notes` (NO `depth_achieved` or `date`)
- `findings` table: columns are `file_id, session_id, line_start, line_end, severity, category, description, followed_up` (NO `evidence` or `line_ref`)
- `file_domains` table: only `file_id, domain_id` (NO `relevance_score`)
- Date: compute in JS (`new Date().toISOString().slice(0,10)`) and pass as parameter — avoids quoting issues in `node -e`
- `packages.base_path` uses `~` — expand with `.replace(/^~/, process.env.HOME)` in Node.js
- `exclude_paths` table: `pattern, reason, added_date` — 39 patterns auto-filtered by `priority_gaps` view
- `smart_priority_gaps` view: relevance-tiered priority queue. Use `WHERE tier_rank <= 2` for high-signal files. Tiers: CONNECTED (1), OWN_CODE (1), PROXIMATE (2), NEARBY (3), DOMAIN_ONLY (4)
- `priority_gaps` view: raw unranked priority queue (use `smart_priority_gaps` instead)
- `package_connectivity` view: shows ISOLATED/WEAKLY_CONNECTED/CONNECTED per package
- `subtree_connectivity` view: same at directory level — use to find Goalie-like islands before reading

## Common Tasks

### Find files needing DEEP analysis
```sql
SELECT f.* FROM files f
JOIN file_domains fd ON f.id = fd.file_id
JOIN domains d ON fd.domain_id = d.id
WHERE d.priority = 'HIGH'
  AND f.depth IN ('NOT_TOUCHED', 'SURFACE', 'MENTIONED')
  AND NOT EXISTS (SELECT 1 FROM exclude_paths ep WHERE f.relative_path LIKE ep.pattern)
ORDER BY f.loc DESC
LIMIT 10;
```

### Find orphaned files (no domain tags)
```sql
SELECT f.* FROM files f
LEFT JOIN file_domains fd ON f.id = fd.file_id
WHERE fd.file_id IS NULL
  AND f.depth != 'EXCLUDED'
  AND f.relative_path NOT LIKE '%test%'
  AND f.relative_path NOT LIKE '%node_modules%'
  AND NOT EXISTS (SELECT 1 FROM exclude_paths ep WHERE f.relative_path LIKE ep.pattern);
```

### Track session progress
```sql
SELECT s.name, s.date, s.focus,
  COUNT(DISTINCT fr.file_id) as files_read,
  SUM(fr.lines_read) as total_lines,
  COUNT(DISTINCT f.id) as findings_count
FROM sessions s
LEFT JOIN file_reads fr ON s.id = fr.session_id
LEFT JOIN findings f ON s.id = f.session_id
WHERE s.id = ?
GROUP BY s.id;
```

## Research Quality Standards

1. Every DEEP read must produce at least 3 findings
2. Every file must be tagged with 1+ domains
3. Cross-file dependencies require evidence (line numbers, exact references)
4. Synthesis documents updated within same session as related file reads
5. CRITICAL findings escalated to main project ADR system
6. Database updated before file analysis (prevents data loss on interruption)

## Integration with Main Project

- Findings with severity CRITICAL or HIGH should inform ADRs in main project
- Discovered bugs should be filed in `docs/bugs/BUG-*.md`
- Architecture insights should update main `CLAUDE.md` or spawn new ADRs
- Cross-package integration issues tracked in integration_hotspots view
