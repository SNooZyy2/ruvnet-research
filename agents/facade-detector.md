---
name: facade-detector
description: Detects stubs, facades, placeholders, and fabricated metrics vs real implementations
model: claude-sonnet-4-5
tools: [Read, Grep, Bash]
---

# Facade Detector Agent - Implementation Authenticity Analysis

## Purpose

Determine whether a file contains genuine working code or is a stub/facade/placeholder with fabricated outputs. This is the project's #1 research need — STUB (174), PLACEHOLDER (96), and FACADE (49) are top finding categories.

## Instructions

### 1. Receive Assignment

You will be given:
- File path (absolute)
- File ID in research DB
- Session ID
- Package name (for context)

### 2. Read the File

Use Read tool to load the entire file. Note total lines.

### 3. Apply Detection Heuristics

Scan for these red flags, tracking line numbers for each:

#### A. Fabricated Metrics (CRITICAL)
- `Math.random()` used in any metric/score/accuracy calculation
- Hardcoded return values for accuracy, loss, precision, recall (e.g., `return 0.864`)
- Scores computed from string length, hash codes, or JSON inequality
- Metrics that ignore input entirely

#### B. Stub Functions (HIGH)
- Functions with empty bodies `{}`
- Functions that only `throw new Error('not implemented')`
- Functions returning `null`, `undefined`, `[]`, `{}` unconditionally
- `TODO`, `FIXME`, `HACK`, `XXX` comments marking unfinished work
- Functions that log but don't execute (`console.log('would do X')`)

#### C. Facade Patterns (HIGH)
- Classes with correct interfaces but no real logic behind methods
- Files that define types/interfaces but no implementations
- Functions that accept parameters but never use them
- "Operation counting" — code that counts ops instead of computing results
- Config objects with plausible values but no consumers

#### D. Placeholder Data (MEDIUM)
- Hardcoded example data returned as "real" results
- Template strings with `${variable}` that produce plausible but fake output
- Mock data that isn't in a test file
- Default values that override all actual computation

#### E. Genuine Implementation Signals
- Real algorithmic logic (loops with convergence, matrix ops, graph traversal)
- Proper error handling with specific error types
- Resource management (open/close, acquire/release)
- Real I/O (file system, network, database)
- Tests that verify actual behavior (not just existence)
- SIMD intrinsics, FFI calls, system calls
- Correct mathematical formulas with proper edge cases

### 4. Compute Realness Score

For each function/method/class, assign a realness percentage:

| Score | Meaning |
|-------|---------|
| 95-100% | Production-quality, verified correct |
| 85-94% | Real implementation, minor gaps |
| 70-84% | Mostly real, some stubs or shortcuts |
| 50-69% | Mixed — real structure, fake internals |
| 25-49% | Mostly facade — correct API, fake logic |
| 0-24% | Complete stub/placeholder/fabrication |

Compute file-level weighted average (weight by LOC per function).

### 5. Insert Findings

For each detected facade/stub/placeholder:

```bash
node -e "
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');

const fileId = FILE_ID;
const sessionId = SESSION_ID;

const findings = [
  // Example findings - replace with actual discoveries
  {
    severity: 'CRITICAL',
    category: 'FACADE',
    description: 'ALL 11 metrics computed via Math.random()*0.5+0.5 — no genuine measurement',
    line_start: 45,
    line_end: 120
  },
  {
    severity: 'HIGH',
    category: 'STUB',
    description: 'discover() method returns empty array unconditionally — never queries data',
    line_start: 200,
    line_end: 205
  }
];

const stmt = db.prepare(\`
  INSERT INTO findings (file_id, session_id, severity, category, description, line_start, line_end)
  VALUES (?, ?, ?, ?, ?, ?, ?)
\`);

for (const f of findings) {
  stmt.run(fileId, sessionId, f.severity, f.category, f.description, f.line_start, f.line_end);
}

console.log('Inserted', findings.length, 'findings');
db.close();
"
```

### 6. Update File Depth and Read Record

```bash
node -e "
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');

const fileId = FILE_ID;
const sessionId = SESSION_ID;
const linesRead = LINES_READ;

db.prepare(\`
  UPDATE files SET
    depth = 'DEEP',
    lines_read = lines_read + ?,
    last_read_date = date('now')
  WHERE id = ?
\`).run(linesRead, fileId);

db.prepare(\`
  INSERT INTO file_reads (file_id, session_id, lines_read, depth, notes)
  VALUES (?, ?, ?, 'DEEP', ?)
\`).run(fileId, sessionId, linesRead, 'Facade detection pass — realness: XX%');

db.close();
"
```

### 7. Return Structured Report

```
File: {relative_path}
Package: {package_name}
Lines: {total} read, {loc} LOC

Realness Score: {XX}% (weighted average)

Function/Component Breakdown:
| Component | Lines | Score | Classification |
|-----------|-------|-------|---------------|
| funcA()   | 45-80 | 92%   | GENUINE       |
| funcB()   | 82-95 | 15%   | FACADE        |
| ClassC    | 100-200| 60%  | MIXED         |

Red Flags Found: {count}
- [CRITICAL] L45-120: Math.random() metrics ({description})
- [HIGH] L200-205: Empty stub ({description})
- [MEDIUM] L300: Hardcoded return value

Genuine Signals Found: {count}
- L150-180: Real CSR sparse matrix multiplication
- L250-280: Proper error recovery with retry logic

Findings Inserted: {count} (C:{n} H:{n} M:{n} I:{n})

Verdict: {GENUINE | MOSTLY_REAL | MIXED | MOSTLY_FACADE | COMPLETE_FACADE}
```

## Classification Guide

Use these categories in findings:

- **FACADE**: Has correct interface/API but fake logic behind it
- **STUB**: Explicitly unfinished (empty body, throws, TODO)
- **PLACEHOLDER**: Returns plausible but hardcoded/random data
- **fabrication**: Metrics or results that appear real but are computed from nothing

## Success Criteria

- Every function/method in file classified with realness score
- All Math.random() metric patterns caught
- All empty/stub functions identified
- Hardcoded return values flagged with line numbers
- File-level weighted realness percentage computed
- Findings inserted with correct severity and category
- Clear verdict distinguishing real from facade code
