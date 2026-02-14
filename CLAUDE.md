# Research Mode - Claude Flow V3 Analysis Project

## Project Overview

This is a research project analyzing the ruvnet multi-repo universe (claude-flow, agentic-flow, agentdb, ruvector).

**Source of Truth**: `research/db/research.db` (SQLite via better-sqlite3)
**Synthesis Documents**: `research/domains/*/analysis.md`
**Auto-Generated Index**: `MASTER-INDEX.md` (never edit manually)
**Reference**: `research/ADR-038-research-database-system.md`

## Behavioral Rules

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
const rows = db.prepare('SELECT * FROM priority_gaps LIMIT 10').all();
console.log(JSON.stringify(rows, null, 2));
db.close();
"
```

## Query Recipes

### 1. What should I read next?
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
```sql
-- Insert read record
INSERT INTO file_reads (file_id, session_id, lines_read, depth_achieved, date)
VALUES (?, ?, ?, ?, date('now'));

-- Update file metadata
UPDATE files SET
  depth = ?,
  lines_read = lines_read + ?,
  last_read_date = date('now')
WHERE id = ?;
```

### 9. Add finding
```sql
INSERT INTO findings (file_id, session_id, severity, category, description, evidence, line_ref)
VALUES (?, ?, ?, ?, ?, ?, ?);
```

### 10. Tag file with domain
```sql
INSERT OR IGNORE INTO file_domains (file_id, domain_id, relevance_score)
VALUES (?, ?, ?);
```

### 11. Add dependency
```sql
INSERT INTO dependencies (source_file_id, target_file_id, relationship, evidence)
VALUES (?, ?, ?, ?);
```

## Session Protocol

### 1. Start Session
```bash
node -e "
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const result = db.prepare('INSERT INTO sessions (name, date, focus) VALUES (?, date(\"now\"), ?)').run('session-2026-02-14', 'focus description');
console.log('Session ID:', result.lastInsertRowid);
db.close();
"
```

### 2. Check Priorities
```bash
node -e "
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const gaps = db.prepare('SELECT * FROM priority_gaps LIMIT 10').all();
console.log(JSON.stringify(gaps, null, 2));
db.close();
"
```

### 3. Research Loop
- Read files from priority queue
- Update database (depth, lines_read, findings, domains, dependencies)
- Write discoveries to domain synthesis documents
- Insert file_reads records for session history

### 4. Write/Update Domain Synthesis
Create or update `domains/{domain-name}/analysis.md` with:
- Overview of domain purpose
- Key files and their roles
- Architecture patterns
- Findings (grouped by severity)
- Cross-domain dependencies
- Knowledge gaps requiring deeper analysis

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
│   └── mapper.md            # Dependency mapping agent
└── ADR-038-*.md             # Architecture decision record

```

## Common Tasks

### Find files needing DEEP analysis
```sql
SELECT f.* FROM files f
JOIN file_domains fd ON f.id = fd.file_id
JOIN domains d ON fd.domain_id = d.id
WHERE d.priority = 1
  AND f.depth IN ('NOT_TOUCHED', 'SURFACE', 'MENTIONED')
ORDER BY f.total_lines DESC
LIMIT 10;
```

### Find orphaned files (no domain tags)
```sql
SELECT f.* FROM files f
LEFT JOIN file_domains fd ON f.id = fd.file_id
WHERE fd.file_id IS NULL
  AND f.relative_path NOT LIKE '%test%'
  AND f.relative_path NOT LIKE '%node_modules%';
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
