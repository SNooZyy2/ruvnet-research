# ADR-038: Research Ledger Database System

**Status**: PROPOSED
**Date**: 2026-02-14
**Supersedes**: Manual markdown ledgers (6 directories, 5 unpopulated templates)

---

## Context

We are conducting systematic analysis of the ruvnet multi-repo universe — multiple npm packages, thousands of files, complex cross-package dependencies. The current approach uses per-topic markdown ledgers:

- `swarm-functionality-analysis/` — 418-line ledger (the only populated one), 1377-line analysis
- `agentdb-analysis/` — blank template, 386-line analysis
- `agentic-flow-analysis/` — blank template, 511-line analysis
- `claude-flow-analysis/` — blank template, 962-line analysis
- `model-router-analysis/` — blank template, 434-line analysis
- `ruvector-analysis/` — blank template, 662-line analysis

### Problems with the current approach

1. **5 of 6 ledgers are empty templates** — the inventory system was never bootstrapped for most packages
2. **The filled ledger mixes sources** — the swarm ledger tracks files from 6 different base paths (CLI dist/, helpers, agents, skills, source TS, config) in one flat table with no relational structure
3. **No cross-file relationships** — the dependency chain `settings.json → format-routing-directive.sh → hooks-tools.js → model-router.js` is discoverable only by reading prose, not queryable
4. **Many-to-many domains impossible** — `memory-initializer.js` belongs to domains: memory, embeddings, vector-search, HNSW. A flat ledger forces it into one category
5. **No session history** — a file read at SURFACE in R1 then DEEP in R3 only shows the latest state
6. **Findings disconnected from files** — "CRITICAL: Line 218 spawns claude CLI" is embedded in a markdown cell, not queryable by severity or category
7. **Scale ceiling** — 479 files tracked in one package. Across the full ruvnet universe, thousands of files will make markdown tables unmanageable
8. **Per-repo structure doesn't match how research works** — analysis is cross-cutting by functional domain, not siloed by package

---

## Decision

Replace the per-topic markdown ledger system with:

1. **SQLite database** (`research/db/research.db`) for inventory, coverage tracking, dependency graphs, and findings
2. **Domain-based synthesis documents** (`research/domains/*/analysis.md`) for narrative analysis — organized by functional concern, not by repository
3. **Auto-generated master index** (`research/MASTER-INDEX.md`) from database views
4. **Self-contained research project** with its own `CLAUDE.md`, agent prompts, and initialization scripts

### What is NOT included

- No automated database export/dump system. Manual `.db` file copies for backups when desired.
- No modification to existing runtime databases (`~/.swarm/memory.db`, `~/.swarm/agentdb.db`). The research DB is fully separate.
- No auto-discovered Claude Code agents (would require placement in `~/.claude/agents/`). Research agent protocols live in `research/agents/` as reference documents and Task tool prompt templates.

---

## Architecture

### Folder Structure

```
research/
  CLAUDE.md                              # Research-mode project instructions
  MASTER-INDEX.md                        # Auto-generated from DB views
  ADR-038-research-database-system.md    # This document
  .gitignore                             # Ignores research.db only

  db/
    schema.sql                           # DDL — git tracked, source of truth for DB structure
    research.db                          # Working SQLite database — git IGNORED

  scripts/
    init-db.js                           # Creates research.db from schema.sql
    populate.js                          # Walks package dirs → populates files table
    report.js                            # Generates MASTER-INDEX.md from DB views

  domains/                               # Synthesis documents — organized by functional concern
    swarm-coordination/
      analysis.md                        # Migrated from swarm-functionality-analysis/
    model-routing/
      analysis.md                        # Migrated from model-router-analysis/
    agentdb-integration/
      analysis.md                        # Migrated from agentdb-analysis/
    agentic-flow/
      analysis.md                        # Migrated from agentic-flow-analysis/
    ruvector/
      analysis.md                        # Migrated from ruvector-analysis/
    claude-flow-cli/
      analysis.md                        # Migrated from claude-flow-analysis/
    (new domains created as discovered)

  agents/                                # Research agent prompt templates (NOT auto-discovered)
    scanner.md                           # Phase 0: filesystem inventory population
    reader.md                            # Phase 1: deep file reading protocol
    synthesizer.md                       # Phase 2: cross-domain narrative synthesis
    mapper.md                            # Phase 3: dependency graph construction
```

### Migration from existing structure

The 6 existing `*-analysis/` directories each contain an `*-analysis.md` (real research content) and an `*-analysis-ledger.md` (5 blank templates + 1 populated). Migration:

| Current | Becomes |
|---------|---------|
| `swarm-functionality-analysis/swarm-functionality-analysis.md` | `domains/swarm-coordination/analysis.md` |
| `swarm-functionality-analysis/swarm-functionality-analysis-ledger.md` | Data migrated into `research.db`, file retired |
| `model-router-analysis/model-router-analysis.md` | `domains/model-routing/analysis.md` |
| `model-router-analysis/model-router-analysis-ledger.md` | Blank template, deleted |
| `agentdb-analysis/agentdb-analysis.md` | `domains/agentdb-integration/analysis.md` |
| `agentic-flow-analysis/agentic-flow-analysis.md` | `domains/agentic-flow/analysis.md` |
| `ruvector-analysis/ruvector-analysis.md` | `domains/ruvector/analysis.md` |
| `claude-flow-analysis/claude-flow-analysis.md` | `domains/claude-flow-cli/analysis.md` |

Old directories removed after migration is verified.

---

## Database Schema

### Tables

```sql
-- ============================================================
-- PACKAGES: Top-level repositories or npm packages being analyzed
-- ============================================================
CREATE TABLE packages (
  id INTEGER PRIMARY KEY,
  name TEXT UNIQUE NOT NULL,
  base_path TEXT NOT NULL,
  repo_url TEXT,
  description TEXT,
  total_files INTEGER DEFAULT 0,
  total_loc INTEGER DEFAULT 0,
  populated_at TEXT                     -- ISO 8601 timestamp of last Phase 0 scan
);

-- Known packages for initial seed:
-- 'claude-flow-cli'    → ~/.npm-global/lib/node_modules/@claude-flow/cli/
-- 'agentdb'            → ~/node_modules/agentdb/
-- 'agentic-flow'       → ~/node_modules/agentic-flow/
-- '@ruvector/core'     → ~/node_modules/@ruvector/core/
-- 'custom-src'         → ~/claude-flow-self-implemented/src/
-- 'claude-config'      → ~/.claude/
-- '@claude-flow/guidance' → ~/node_modules/@claude-flow/guidance/

-- ============================================================
-- FILES: Every file across all packages
-- ============================================================
CREATE TABLE files (
  id INTEGER PRIMARY KEY,
  package_id INTEGER NOT NULL REFERENCES packages(id),
  relative_path TEXT NOT NULL,           -- e.g. 'dist/src/ruvector/model-router.js'
  loc INTEGER,                           -- lines of code (NULL = not yet counted)
  depth TEXT NOT NULL DEFAULT 'NOT_TOUCHED',
    -- DEEP:        50%+ of file read, algorithms traced, exact code extracted
    -- MEDIUM:      Key sections read (20-50%), architecture understood
    -- SURFACE:     Glob/grep only, categorized by filename
    -- MENTIONED:   Referenced in research but not directly read
    -- NOT_TOUCHED: Known to exist, zero analysis
  lines_read INTEGER DEFAULT 0,
  last_read_date TEXT,                   -- ISO 8601
  notes TEXT,                            -- free-form per-file notes
  UNIQUE(package_id, relative_path)
);

CREATE INDEX idx_files_depth ON files(depth);
CREATE INDEX idx_files_package ON files(package_id);

-- ============================================================
-- DOMAINS: Functional concerns that span multiple packages
-- ============================================================
CREATE TABLE domains (
  id INTEGER PRIMARY KEY,
  name TEXT UNIQUE NOT NULL,             -- kebab-case: 'swarm-coordination'
  description TEXT,
  priority TEXT NOT NULL DEFAULT 'MEDIUM',
    -- HIGH:   Directly affects understanding of core system behavior
    -- MEDIUM: Important for complete picture
    -- LOW:    Completeness only, not blocking other analysis
  synthesis_path TEXT                    -- relative path to analysis.md, e.g. 'domains/swarm-coordination/analysis.md'
);

-- ============================================================
-- FILE_DOMAINS: Many-to-many junction — a file can belong to multiple domains
-- ============================================================
CREATE TABLE file_domains (
  file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  domain_id INTEGER NOT NULL REFERENCES domains(id) ON DELETE CASCADE,
  PRIMARY KEY (file_id, domain_id)
);

CREATE INDEX idx_file_domains_domain ON file_domains(domain_id);

-- ============================================================
-- SESSIONS: Research sessions with metadata
-- ============================================================
CREATE TABLE sessions (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,                    -- human label: 'R1', 'R3', 'ADR-037', 'swarm-deep-dive-2'
  date TEXT NOT NULL,                    -- ISO 8601
  focus TEXT,                            -- what this session aimed to investigate
  agent_count INTEGER DEFAULT 1,         -- how many agents were used
  notes TEXT
);

-- ============================================================
-- FILE_READS: Per-session read history (a file can be read across multiple sessions)
-- ============================================================
CREATE TABLE file_reads (
  id INTEGER PRIMARY KEY,
  file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  depth TEXT NOT NULL,                   -- depth achieved THIS session
  lines_read INTEGER,
  line_ranges TEXT,                      -- e.g. '1-50,170-320,800-810'
  notes TEXT                             -- findings from this specific read
);

CREATE INDEX idx_file_reads_file ON file_reads(file_id);
CREATE INDEX idx_file_reads_session ON file_reads(session_id);

-- ============================================================
-- FINDINGS: Discrete discoveries linked to files + line ranges
-- ============================================================
CREATE TABLE findings (
  id INTEGER PRIMARY KEY,
  file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  session_id INTEGER REFERENCES sessions(id),
  line_start INTEGER,
  line_end INTEGER,
  severity TEXT NOT NULL DEFAULT 'INFO',
    -- CRITICAL: Fundamentally changes understanding (e.g. "spawns real process")
    -- HIGH:     Important architectural insight
    -- MEDIUM:   Notable behavior or pattern
    -- INFO:     Catalogued for completeness
  category TEXT,
    -- 'real-functionality'  — code that actually does something meaningful
    -- 'stub'                — presentational/fake/hardcoded
    -- 'security'            — security-relevant behavior
    -- 'architecture'        — structural pattern or anti-pattern
    -- 'integration-point'   — where packages connect
    -- 'bug'                 — defect or inconsistency
    -- 'unknown'             — needs further investigation
  description TEXT NOT NULL,
  followed_up INTEGER DEFAULT 0          -- has this finding been investigated further?
);

CREATE INDEX idx_findings_file ON findings(file_id);
CREATE INDEX idx_findings_severity ON findings(severity);
CREATE INDEX idx_findings_category ON findings(category);

-- ============================================================
-- DEPENDENCIES: Directed graph edges between files
-- ============================================================
CREATE TABLE dependencies (
  id INTEGER PRIMARY KEY,
  source_file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  target_file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  relationship TEXT NOT NULL,
    -- 'imports'     — ES/CJS import/require
    -- 'calls'       — runtime function call or CLI invocation
    -- 'triggers'    — hook/event trigger (e.g. settings.json triggers a helper)
    -- 'bridges'     — adapter/bridge pattern connecting two systems
    -- 'bundles'     — file is a bundled/copied version of another
    -- 'generates'   — one file generates/creates the other (e.g. init generators)
    -- 'reads'       — reads data from (e.g. JSON state files)
    -- 'writes'      — writes data to
  evidence TEXT,                          -- e.g. 'line 218: spawn("claude", [...])'
  verified INTEGER NOT NULL DEFAULT 0,    -- 1 = confirmed by reading both files
  UNIQUE(source_file_id, target_file_id, relationship)
);

CREATE INDEX idx_deps_source ON dependencies(source_file_id);
CREATE INDEX idx_deps_target ON dependencies(target_file_id);
CREATE INDEX idx_deps_relationship ON dependencies(relationship);

-- ============================================================
-- PACKAGE_DEPENDENCIES: High-level inter-package relationships
-- ============================================================
CREATE TABLE package_dependencies (
  id INTEGER PRIMARY KEY,
  source_package_id INTEGER NOT NULL REFERENCES packages(id),
  target_package_id INTEGER NOT NULL REFERENCES packages(id),
  relationship TEXT NOT NULL,
    -- 'npm-dependency'  — listed in package.json
    -- 'bundles-copy'    — contains copied/vendored code from target
    -- 'bridges-to'      — has adapter/bridge code connecting to target
    -- 'patches'         — applies runtime patches to target
    -- 'optional'        — optional dependency, may not be installed
  notes TEXT,
  UNIQUE(source_package_id, target_package_id, relationship)
);
```

### Views

```sql
-- Coverage dashboard by domain
CREATE VIEW domain_coverage AS
SELECT
  d.name,
  d.priority,
  d.synthesis_path,
  COUNT(f.id) AS total_files,
  SUM(f.loc) AS total_loc,
  SUM(CASE WHEN f.depth = 'DEEP' THEN 1 ELSE 0 END) AS deep,
  SUM(CASE WHEN f.depth = 'MEDIUM' THEN 1 ELSE 0 END) AS medium,
  SUM(CASE WHEN f.depth = 'SURFACE' THEN 1 ELSE 0 END) AS surface,
  SUM(CASE WHEN f.depth = 'NOT_TOUCHED' THEN 1 ELSE 0 END) AS not_touched,
  ROUND(100.0 * SUM(CASE WHEN f.depth IN ('DEEP', 'MEDIUM') THEN f.loc ELSE 0 END)
    / MAX(SUM(f.loc), 1), 1) AS pct_loc_covered
FROM domains d
JOIN file_domains fd ON d.id = fd.domain_id
JOIN files f ON fd.file_id = f.id
GROUP BY d.id
ORDER BY d.priority, d.name;

-- Coverage dashboard by package
CREATE VIEW package_coverage AS
SELECT
  p.name,
  p.base_path,
  COUNT(f.id) AS total_files,
  SUM(f.loc) AS total_loc,
  SUM(CASE WHEN f.depth = 'DEEP' THEN 1 ELSE 0 END) AS deep,
  SUM(CASE WHEN f.depth = 'MEDIUM' THEN 1 ELSE 0 END) AS medium,
  SUM(CASE WHEN f.depth = 'SURFACE' THEN 1 ELSE 0 END) AS surface,
  SUM(CASE WHEN f.depth = 'NOT_TOUCHED' THEN 1 ELSE 0 END) AS not_touched,
  ROUND(100.0 * SUM(CASE WHEN f.depth IN ('DEEP', 'MEDIUM') THEN f.loc ELSE 0 END)
    / MAX(SUM(f.loc), 1), 1) AS pct_loc_covered
FROM packages p
JOIN files f ON p.id = f.package_id
GROUP BY p.id
ORDER BY p.name;

-- Files that span 3+ domains — integration hotspots
CREATE VIEW integration_hotspots AS
SELECT
  f.relative_path,
  p.name AS package,
  f.loc,
  f.depth,
  GROUP_CONCAT(d.name, ', ') AS domains,
  COUNT(d.id) AS domain_count
FROM files f
JOIN packages p ON f.package_id = p.id
JOIN file_domains fd ON f.id = fd.file_id
JOIN domains d ON fd.domain_id = d.id
GROUP BY f.id
HAVING COUNT(d.id) >= 3
ORDER BY COUNT(d.id) DESC, f.loc DESC;

-- Priority gaps: NOT_TOUCHED files in HIGH priority domains
CREATE VIEW priority_gaps AS
SELECT
  d.name AS domain,
  p.name AS package,
  f.relative_path,
  f.loc
FROM files f
JOIN packages p ON f.package_id = p.id
JOIN file_domains fd ON f.id = fd.file_id
JOIN domains d ON fd.domain_id = d.id
WHERE f.depth = 'NOT_TOUCHED'
  AND d.priority = 'HIGH'
ORDER BY f.loc DESC;

-- Unverified dependency claims
CREATE VIEW unverified_deps AS
SELECT
  sf.relative_path AS source,
  sp.name AS source_pkg,
  tf.relative_path AS target,
  tp.name AS target_pkg,
  dep.relationship,
  dep.evidence
FROM dependencies dep
JOIN files sf ON dep.source_file_id = sf.id
JOIN packages sp ON sf.package_id = sp.id
JOIN files tf ON dep.target_file_id = tf.id
JOIN packages tp ON tf.package_id = tp.id
WHERE dep.verified = 0
ORDER BY dep.relationship, sf.relative_path;

-- All CRITICAL/HIGH findings not yet followed up
CREATE VIEW open_findings AS
SELECT
  fi.severity,
  fi.category,
  f.relative_path,
  p.name AS package,
  fi.line_start,
  fi.line_end,
  fi.description,
  s.name AS session
FROM findings fi
JOIN files f ON fi.file_id = f.id
JOIN packages p ON f.package_id = p.id
LEFT JOIN sessions s ON fi.session_id = s.id
WHERE fi.followed_up = 0
  AND fi.severity IN ('CRITICAL', 'HIGH')
ORDER BY
  CASE fi.severity WHEN 'CRITICAL' THEN 0 WHEN 'HIGH' THEN 1 END,
  f.relative_path;
```

### Seed Data

```sql
-- Packages
INSERT INTO packages (name, base_path, description) VALUES
  ('claude-flow-cli', '~/.npm-global/lib/node_modules/@claude-flow/cli/', 'Main CLI package — commands, MCP tools, services, bundled ruvector'),
  ('agentdb', '~/node_modules/agentdb/', 'AgentDB npm package — vector DB, episodes, skills'),
  ('agentic-flow', '~/node_modules/agentic-flow/', 'Agentic-flow npm package — ReasoningBank, embeddings, health server'),
  ('@ruvector/core', '~/node_modules/@ruvector/core/', 'Rust NAPI HNSW index'),
  ('custom-src', '~/claude-flow-self-implemented/src/', 'Our custom DDD TypeScript implementation'),
  ('claude-config', '~/.claude/', 'Claude Code project config — helpers, agents, skills, commands, settings'),
  ('@claude-flow/guidance', '~/node_modules/@claude-flow/guidance/', 'Guidance package — WASM kernel, constitutional AI');

-- Package dependencies
INSERT INTO package_dependencies (source_package_id, target_package_id, relationship, notes) VALUES
  (1, 2, 'optional', 'agentdb npm used via bridge in agentdb-tools.js'),
  (1, 3, 'optional', 'agentic-flow used for embeddings in memory-initializer.js'),
  (1, 4, 'optional', '@ruvector/core used for HNSW in memory-initializer.js'),
  (1, 4, 'bundles-copy', 'dist/src/ruvector/ contains bundled code, NOT the npm package'),
  (1, 7, 'npm-dependency', 'guidance package for WASM kernel'),
  (5, 2, 'bridges-to', 'custom src builds DDD services on top of agentdb tables'),
  (5, 4, 'bridges-to', 'custom src uses @ruvector/core for vector backend'),
  (6, 1, 'patches', 'helpers and hooks patch/extend CLI behavior at runtime');

-- Initial domains
INSERT INTO domains (name, description, priority, synthesis_path) VALUES
  ('swarm-coordination', 'How multi-agent swarm init, coordination, and topology work across CLI, MCP, helpers, and agents', 'HIGH', 'domains/swarm-coordination/analysis.md'),
  ('model-routing', 'Model selection, complexity scoring, routing curves, and the 3-tier routing pipeline', 'HIGH', 'domains/model-routing/analysis.md'),
  ('memory-and-learning', 'Memory storage, HNSW search, embeddings, ReasoningBank, neural patterns, SONA/EWC++', 'HIGH', 'domains/memory-and-learning/analysis.md'),
  ('agent-lifecycle', 'Agent spawn, template loading, execution, termination — what actually happens', 'HIGH', 'domains/agent-lifecycle/analysis.md'),
  ('hook-pipeline', 'settings.json hook config → helper scripts → CLI commands → side effects', 'HIGH', 'domains/hook-pipeline/analysis.md'),
  ('agentdb-integration', 'AgentDB DDD services, episodes, skills, reflexion, vector search bridge', 'MEDIUM', 'domains/agentdb-integration/analysis.md'),
  ('agentic-flow', 'Agentic-flow package internals — ReasoningBank, health server, auto-execute patch', 'MEDIUM', 'domains/agentic-flow/analysis.md'),
  ('ruvector', '@ruvector/core NAPI bindings, HNSW implementation, CLI-bundled ruvector modules', 'MEDIUM', 'domains/ruvector/analysis.md'),
  ('init-and-codegen', 'What claude-flow init generates — executor, helpers-generator, statusline-generator, settings-generator', 'MEDIUM', 'domains/init-and-codegen/analysis.md'),
  ('process-spawning', 'Real process spawning — hive-mind claude CLI, headless workers, container pool, daemon', 'HIGH', 'domains/process-spawning/analysis.md'),
  ('plugin-system', 'Plugin manager, IPFS discovery, marketplace, install/uninstall lifecycle', 'LOW', 'domains/plugin-system/analysis.md'),
  ('transfer-system', 'IPFS, PII detection, Seraphine model, pattern store, GCS storage', 'LOW', 'domains/transfer-system/analysis.md'),
  ('production-infra', 'Circuit breaker, rate limiter, retry, monitoring, error handler', 'LOW', 'domains/production-infra/analysis.md'),
  ('claude-flow-cli', 'CLI entry point, command registration, MCP server scaffold — the outer shell', 'MEDIUM', 'domains/claude-flow-cli/analysis.md');

-- Research sessions (historical, from existing ledger)
INSERT INTO sessions (name, date, focus, agent_count, notes) VALUES
  ('R1', '2026-02-13', 'Initial swarm functionality analysis — swarm commands, MCP tools, coordination', 5, 'First research round. Discovered swarm-tools.js is stub, coordination uses Math.random()'),
  ('R2', '2026-02-13', 'Worker system, headless execution, communication', 4, 'Discovered headless-worker-executor spawns real claude CLI processes'),
  ('R3', '2026-02-14', 'Deep dive + gap discovery — init system, production, transfer, hooks-tools internals', 5, 'Discovered 53 new files totaling ~14,800 lines. bin/cli.js=4368 lines, bin/mcp-server.js=5116 lines'),
  ('R3-verify', '2026-02-14', 'Ledger verification and line count corrections', 4, 'Corrected bin/cli.js from ~200 to 4,368 lines');
```

---

## Scripts

### `scripts/init-db.js`

Creates `research.db` from `schema.sql`. Idempotent — drops and recreates if DB already exists. Also applies seed data.

**Input**: `db/schema.sql`
**Output**: `db/research.db`
**Usage**: `node research/scripts/init-db.js`

### `scripts/populate.js`

Phase 0 scanner. For each package in the `packages` table:
1. Resolves `base_path` (expands `~`)
2. Walks the directory tree (respecting common ignores: `node_modules/`, `.git/`, `__pycache__/`)
3. Counts LOC per file via `wc -l`
4. Inserts rows into `files` table with `depth = 'NOT_TOUCHED'`
5. Updates `packages.total_files` and `packages.total_loc`
6. Skips files already in the table (idempotent)

**File type filters**: `.js`, `.mjs`, `.cjs`, `.ts`, `.sh`, `.json`, `.md` (in agents/skills/commands dirs), `.wasm`, `.sql`
**Ignore patterns**: `node_modules/`, `.git/`, `*.map`, `*.d.ts` (unless in custom-src), `package-lock.json`

**Usage**: `node research/scripts/populate.js` (run after init-db, or anytime to pick up new files)

### `scripts/report.js`

Generates `MASTER-INDEX.md` from database views:
1. Runs `domain_coverage` view → domain coverage table
2. Runs `package_coverage` view → package coverage table
3. Runs `integration_hotspots` view → hotspot list
4. Runs `open_findings` view → priority findings
5. Counts total files, total LOC, overall coverage percentage
6. Writes formatted markdown to `MASTER-INDEX.md`

**Usage**: `node research/scripts/report.js` (run after any research session)

---

## Research CLAUDE.md

The file at `research/CLAUDE.md` provides research-mode instructions when Claude is invoked from the `research/` directory. Key sections:

### Purpose & Orientation
- This is a research project analyzing the ruvnet multi-repo universe
- The source of truth for inventory and coverage is `research/db/research.db`
- Synthesis documents live in `research/domains/*/analysis.md`
- MASTER-INDEX.md is auto-generated — never edit manually

### Database Interaction Protocol
- Use `better-sqlite3` via inline Node.js scripts (Bash tool) to query/update the DB
- After reading any file, update the corresponding row in `files`: set `depth`, `lines_read`, `last_read_date`
- After discovering a cross-file dependency, insert into `dependencies` with `verified = 1`
- After a significant finding, insert into `findings` with appropriate severity and category
- When a file belongs to a domain not yet tagged, insert into `file_domains`

### Query Recipes
Common queries provided as copy-paste SQL for quick session startup:
- "What should I read next?" → priority_gaps view
- "What's our coverage?" → domain_coverage view
- "Show dependency chain for file X" → recursive CTE on dependencies
- "All CRITICAL findings" → open_findings view
- "Files in domain X not yet read" → join files, file_domains, domains with depth filter

### Domain Management
- New domains: `INSERT INTO domains (name, description, priority, synthesis_path) VALUES (...)`
- Create matching directory under `research/domains/`
- Tag files: `INSERT INTO file_domains (file_id, domain_id) VALUES (...)`

### Session Protocol
1. Start: `INSERT INTO sessions (name, date, focus) VALUES (...)`
2. Work: read files, update DB, write synthesis
3. End: run `node research/scripts/report.js` to regenerate MASTER-INDEX.md

### Depth Classification Rules
- **DEEP**: 50%+ of file read. Algorithms traced. Exact code snippets or line references extracted. Data flow verified against connected files.
- **MEDIUM**: 20-50% read. Architecture and purpose understood. Some execution paths untraced.
- **SURFACE**: Glob/grep only. File exists, categorized by name/directory. 0-2 lines actually read.
- **MENTIONED**: Referenced in another file's analysis but never directly opened.
- **NOT_TOUCHED**: Known to exist from Phase 0 scan. Zero analysis.

---

## Agent Prompt Templates

Located in `research/agents/`. These are NOT auto-discovered Claude Code agents. They are reference documents providing structured prompts for use with the Task tool.

### `scanner.md` — Phase 0 Inventory Agent
- Given a package name and base path, walk the filesystem
- Insert all discovered files into the `files` table
- Count LOC, set depth to NOT_TOUCHED
- Report: total files found, total LOC, file type distribution

### `reader.md` — Deep Reading Agent
- Given a file path and target depth (MEDIUM or DEEP)
- Read the file, extract key findings
- Update `files.depth`, `files.lines_read`, `files.last_read_date`
- Insert `file_reads` row for session history
- Insert `findings` rows for any discoveries
- Tag file with relevant domains via `file_domains`
- Note any dependencies to other files (insert into `dependencies` with evidence)

### `synthesizer.md` — Cross-Domain Synthesis Agent
- Given a domain name, query all files tagged to that domain
- Read the DEEP and MEDIUM files
- Write or update the domain's `analysis.md` with narrative synthesis
- Identify cross-domain interaction points
- Flag files that need deeper reading

### `mapper.md` — Dependency Mapping Agent
- Given a starting file or domain
- Trace import/require chains, hook triggers, CLI delegations
- Insert verified edges into `dependencies` table
- Build visual dependency chain (text-based) for the synthesis doc
- Flag unverified dependencies for follow-up

---

## Workflow

### First-Time Setup

```bash
cd ~/research
node scripts/init-db.js              # Creates research.db from schema.sql + seed data
node scripts/populate.js             # Walks all 7 packages, populates files table
node scripts/report.js               # Generates initial MASTER-INDEX.md
```

### Research Session

```bash
cd ~/research
claude                               # Picks up research/CLAUDE.md

# Inside Claude session:
# 1. Read MASTER-INDEX.md to see current state
# 2. Query priority_gaps view to decide what to read
# 3. Read files, update DB, write findings
# 4. Write/update domain synthesis docs
# 5. Run: node research/scripts/report.js
```

### Adding a New Package

```sql
-- In the DB:
INSERT INTO packages (name, base_path, description) VALUES
  ('new-package', '/path/to/package/', 'Description');
```
Then run `node research/scripts/populate.js` to scan it.

### Adding a New Domain

```sql
INSERT INTO domains (name, description, priority, synthesis_path) VALUES
  ('new-domain', 'What this domain covers', 'HIGH', 'domains/new-domain/analysis.md');
```
Then create `research/domains/new-domain/analysis.md`.

### Manual Backup

```bash
cp research/db/research.db research/db/research-backup-$(date +%Y%m%d).db
```

---

## Consequences

### Positive
- **Single source of truth** for "have we read this file?" across all packages
- **Cross-cutting queries** that were impossible with flat markdown (multi-domain files, dependency chains, coverage by domain)
- **Session history preserved** — know when and how deeply each file was analyzed
- **Findings are queryable** by severity, category, file — not buried in markdown cells
- **Dependency graph** enables tracing execution chains across package boundaries
- **Self-contained** — the research/ directory is a complete project with its own instructions and tooling
- **Scalable** — SQLite handles the full ruvnet universe without structural changes

### Negative
- **DB is not human-readable** — must use SQL queries or the report script to inspect state
- **Requires discipline** — researchers must update the DB during sessions, not just read files
- **Binary file** — .db doesn't git-diff; state history relies on manual backups
- **Script maintenance** — init-db.js, populate.js, report.js must be kept working

### Risks
- **Schema evolution** — if we need new columns/tables later, must write migration SQL and re-init (or ALTER TABLE)
- **Data loss** — if research.db is deleted without backup, all coverage/findings data is lost (synthesis docs in domains/ survive since they're separate markdown files)
- **Stale inventory** — if a package updates (npm update), populate.js must be re-run to catch new/removed files

### Mitigations
- Schema changes: always modify `schema.sql` first, then ALTER TABLE on the live DB or re-init
- Data loss: manual backups before major changes; synthesis docs are the irreplaceable output, DB is rebuildable (re-run populate, re-do depth tracking)
- Staleness: run `populate.js` at the start of each research campaign against a package

---

## Open Questions

1. **Should we track GitHub repos not yet cloned locally?** — e.g. other repos under `ruvnet/` org. Could add a `packages` row with `base_path = NULL` and a `repo_url`, then populate when cloned.
2. **Should findings auto-propagate to synthesis docs?** — Currently manual. Could build a script that appends new CRITICAL findings to the relevant domain's analysis.md.
3. **FTS5 for findings search?** — If the findings table grows large, adding `CREATE VIRTUAL TABLE findings_fts USING fts5(description, content=findings)` would enable full-text search across all findings.
