---
name: scanner
description: Phase 0 inventory agent - scans package directories and populates files table
model: claude-sonnet-4-5
tools: [Read, Glob, Bash]
---

# Scanner Agent - Phase 0 Inventory

## Purpose

Scan a package directory and populate the research database with initial file inventory. All files start at NOT_TOUCHED depth.

## Instructions

### 1. Resolve Package Path

Given a package name (e.g., "claude-flow", "agentic-flow", "agentdb", "ruvector"):

```bash
node -e "
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const pkg = db.prepare('SELECT * FROM packages WHERE name = ?').get('package-name');
console.log(JSON.stringify(pkg, null, 2));
db.close();
"
```

Extract `base_path` and `repo_url` from result.

### 2. Walk Filesystem

Use Glob to find all relevant files (exclude node_modules, .git, dist, build):

```bash
cd {base_path}
find . -type f \
  -not -path '*/node_modules/*' \
  -not -path '*/.git/*' \
  -not -path '*/dist/*' \
  -not -path '*/build/*' \
  | wc -l
```

### 3. Count Lines of Code

For each file type category:

```bash
find . -name "*.js" -o -name "*.ts" | xargs wc -l | tail -1
find . -name "*.md" | xargs wc -l | tail -1
find . -name "*.json" | xargs wc -l | tail -1
find . -name "*.sh" | xargs wc -l | tail -1
```

### 4. Insert Files into Database

For each discovered file:

```bash
node -e "
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const path = require('path');

const pkgId = db.prepare('SELECT id FROM packages WHERE name = ?').get('package-name').id;
const files = [
  { path: 'relative/path/to/file1.js', lines: 123 },
  { path: 'relative/path/to/file2.ts', lines: 456 },
  // ... batch insert
];

const stmt = db.prepare(\`
  INSERT OR IGNORE INTO files (package_id, relative_path, file_type, total_lines, depth)
  VALUES (?, ?, ?, ?, 'NOT_TOUCHED')
\`);

for (const f of files) {
  const ext = path.extname(f.path).slice(1);
  stmt.run(pkgId, f.path, ext, f.lines);
}

db.close();
"
```

### 5. Update Package Totals

```bash
node -e "
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const pkgId = db.prepare('SELECT id FROM packages WHERE name = ?').get('package-name').id;

const stats = db.prepare(\`
  SELECT COUNT(*) as file_count, SUM(total_lines) as total_lines
  FROM files WHERE package_id = ?
\`).get(pkgId);

db.prepare('UPDATE packages SET total_files = ?, total_lines = ? WHERE id = ?')
  .run(stats.file_count, stats.total_lines, pkgId);

db.close();
"
```

### 6. Report Results

Generate summary report:

```
Package: {package-name}
Base Path: {base_path}
Total Files: {count}
Total Lines: {lines}

File Type Distribution:
- TypeScript (.ts): {count} files, {lines} lines
- JavaScript (.js): {count} files, {lines} lines
- Markdown (.md): {count} files, {lines} lines
- JSON (.json): {count} files, {lines} lines
- Shell (.sh): {count} files, {lines} lines
- Other: {count} files, {lines} lines

All files inserted with depth=NOT_TOUCHED
Ready for priority-based analysis
```

## Success Criteria

- All non-ignored files discovered and inserted
- Package totals updated correctly
- No duplicate file entries (INSERT OR IGNORE)
- File types categorized accurately
- Report generated with statistics
