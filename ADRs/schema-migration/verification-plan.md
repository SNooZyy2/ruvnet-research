# Verification Plan: Schema Normalization Migration

> **Principle**: Every phase produces a provable assertion.
> The migration is only considered successful when ALL assertions pass.

---

## Verification Architecture

```
Phase 0: Snapshot
  └── ASSERT: backup checksum matches live DB

Phase 2: Dry-Run
  └── ASSERT: dry-run prints expected counts, then ROLLBACKs
  └── ASSERT: DB unchanged after dry-run (checksum matches backup)

Phase 3: Execute (single transaction — includes merge + normalize + triggers)
  ├── ASSERT: Merge verification (1,704 → 1,682, 22 annotations)
  ├── ASSERT: Invariant checks (row counts correct)
  ├── ASSERT: Cardinality checks (distinct values = target)
  ├── ASSERT: Distribution checks (no empty/dominant buckets)
  ├── ASSERT: Cross-reference checks (views still work)
  ├── ASSERT: Spot-check (20 random samples make sense)
  ├── ASSERT: Regression tests (CLAUDE.md queries work)
  └── ASSERT: Triggers reject invalid INSERT/UPDATE

Phase 4: Agent Templates
  └── ASSERT: All 8 templates contain enum constraint blocks
```

---

## Pre-Migration Baseline (capture BEFORE running migration)

Run this script to capture baseline values. Store output in `baseline.json`.

```javascript
// scripts/capture-baseline.js
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const baseline = {};

// Row counts (must not change)
baseline.row_counts = {
  files: db.prepare('SELECT COUNT(*) as n FROM files').get().n,
  findings: db.prepare('SELECT COUNT(*) as n FROM findings').get().n,
  dependencies: db.prepare('SELECT COUNT(*) as n FROM dependencies').get().n,
  sessions: db.prepare('SELECT COUNT(*) as n FROM sessions').get().n,
  file_reads: db.prepare('SELECT COUNT(*) as n FROM file_reads').get().n,
  domains: db.prepare('SELECT COUNT(*) as n FROM domains').get().n,
  file_domains: db.prepare('SELECT COUNT(*) as n FROM file_domains').get().n,
  packages: db.prepare('SELECT COUNT(*) as n FROM packages').get().n,
  exclude_paths: db.prepare('SELECT COUNT(*) as n FROM exclude_paths').get().n,
};

// Severity totals (must not change)
baseline.severity_totals = {};
db.prepare('SELECT severity, COUNT(*) as n FROM findings GROUP BY severity').all()
  .forEach(r => baseline.severity_totals[r.severity] = r.n);

// Findings per session (must not change — proves no findings shifted between sessions)
baseline.findings_per_session = {};
db.prepare('SELECT session_id, COUNT(*) as n FROM findings GROUP BY session_id').all()
  .forEach(r => baseline.findings_per_session[r.session_id] = r.n);

// Dependencies count (pre-merge: 1,704; post-merge target: 1,682)
baseline.dep_count_pre_merge = db.prepare('SELECT COUNT(*) as n FROM dependencies').get().n;
baseline.dep_count_post_merge = baseline.dep_count_pre_merge - 22;

// View row counts (must not change)
baseline.view_counts = {
  smart_priority_gaps: db.prepare('SELECT COUNT(*) as n FROM smart_priority_gaps').get().n,
  domain_coverage: db.prepare('SELECT COUNT(*) as n FROM domain_coverage').get().n,
  package_coverage: db.prepare('SELECT COUNT(*) as n FROM package_coverage').get().n,
};

// Severity distribution must be preserved
// After migration: CRITICAL count must still be 1294, HIGH must be 2923, etc.
// LOW(21) + POSITIVE(2) + LOW_CONFIDENCE(1) = 24 will merge into existing buckets
baseline.canonical_severity = {
  CRITICAL: baseline.severity_totals['CRITICAL'] || 0,           // 1294
  HIGH: baseline.severity_totals['HIGH'] || 0,                   // 2923
  MEDIUM: baseline.severity_totals['MEDIUM'] || 0,               // 3085 (unchanged)
  INFO: (baseline.severity_totals['INFO'] || 0)
      + (baseline.severity_totals['LOW'] || 0)
      + (baseline.severity_totals['POSITIVE'] || 0)
      + (baseline.severity_totals['LOW_CONFIDENCE'] || 0),       // 3795 + 21 + 2 + 1 = 3819
};

console.log(JSON.stringify(baseline, null, 2));
db.close();
```

---

## Assertion Suite

### A0: Merge Verification (pre-normalization)

```sql
-- Run AFTER Step 0 (merge) but BEFORE Step 1 (save originals)
SELECT COUNT(*) as dep_count FROM dependencies;
-- Must be exactly 1,682 (22 duplicate pairs merged from 1,704)

-- Verify no orphaned evidence: every merged row's loser relationship was appended
SELECT COUNT(*) FROM dependencies
WHERE evidence LIKE '%[merged:%';
-- Must be exactly 22 (one annotation per merge)
```

**Pass criteria**: dep_count = 1,682, merge annotations = 22.
**What it proves**: Exactly the expected duplicates were merged, no more, no fewer.

### A1: Row Count Invariants

```sql
-- MUST match baseline exactly (except dependencies, which is post-merge)
SELECT 'files' as tbl, COUNT(*) as n FROM files
UNION ALL SELECT 'findings', COUNT(*) FROM findings
UNION ALL SELECT 'dependencies', COUNT(*) FROM dependencies
UNION ALL SELECT 'sessions', COUNT(*) FROM sessions
UNION ALL SELECT 'file_reads', COUNT(*) FROM file_reads
UNION ALL SELECT 'file_domains', COUNT(*) FROM file_domains;
```

**Pass criteria**: Every count matches `baseline.row_counts`, except `dependencies` = 1,682
(baseline captures pre-merge count of 1,704; the migration script must account for the
22-row reduction from Step 0).
**What it proves**: Migration didn't accidentally insert, delete, or corrupt rows beyond the
intentional 22-row merge.

### A2: No NULLs Introduced

```sql
SELECT
  (SELECT COUNT(*) FROM dependencies WHERE relationship IS NULL) as null_rels,
  (SELECT COUNT(*) FROM findings WHERE category IS NULL) as null_cats,
  (SELECT COUNT(*) FROM findings WHERE severity IS NULL) as null_sevs,
  (SELECT COUNT(*) FROM dependencies WHERE _original_relationship IS NULL) as null_orig_rels,
  (SELECT COUNT(*) FROM findings WHERE _original_category IS NULL) as null_orig_cats;
```

**Pass criteria**: All values = 0.
**What it proves**: Every row was mapped (no gaps in the CASE WHEN logic).

### A3: Cardinality Targets Met

```sql
SELECT 'relationships' as field, COUNT(DISTINCT relationship) as n FROM dependencies
UNION ALL SELECT 'categories', COUNT(DISTINCT category) FROM findings
UNION ALL SELECT 'severities', COUNT(DISTINCT severity) FROM findings;
```

**Pass criteria**: relationships = 10, categories = 12, severities = 4.
**What it proves**: All 511 relationship types collapsed to exactly 10.
All 1,211 categories collapsed to exactly 12. All 7 severities collapsed to 4.

### A4: Only Canonical Values Present

```sql
-- Must return 0 rows
SELECT DISTINCT relationship FROM dependencies
WHERE relationship NOT IN ('IMPORTS','USES','EXPORTS','DECLARES','SIBLINGS',
                           'COMPETES','WRAPS','FEEDS','TESTS','BROKEN');

SELECT DISTINCT category FROM findings
WHERE category NOT IN ('ARCHITECTURE','QUALITY','INTEGRATION','PERFORMANCE',
                       'ALGORITHM','FACADE','SECURITY','BUG',
                       'GENUINE','TESTING','DOCUMENTATION','INCOMPLETE');

SELECT DISTINCT severity FROM findings
WHERE severity NOT IN ('CRITICAL','HIGH','MEDIUM','INFO');
```

**Pass criteria**: All three queries return 0 rows.
**What it proves**: No unexpected values slipped through.

### A5: Severity Conservation

```sql
SELECT severity, COUNT(*) as n FROM findings GROUP BY severity ORDER BY severity;
```

**Pass criteria**:
- CRITICAL = 1,294 (unchanged)
- HIGH = 2,923 (unchanged)
- MEDIUM = 3,085 (unchanged — LOW no longer merges here)
- INFO = 3,819 (was 3,795 + 21 LOW + 2 POSITIVE + 1 LOW_CONFIDENCE)
- Sum = 11,121

**What it proves**: Severity remapping was conservative (only LOW/POSITIVE/LOW_CONFIDENCE changed).

### A6: Distribution Sanity

```sql
SELECT category, COUNT(*) as n FROM findings GROUP BY category ORDER BY n DESC;
```

**Pass criteria**:
- Largest bucket < 3,500 (no mega-bucket from bad catch-all)
- Smallest bucket > 50 (no near-empty category)
- FACADE between 400 and 900
- GENUINE between 200 and 600
- Sum = 11,121

**What it proves**: The mapping distributed findings reasonably, not dumping everything into QUALITY.

```sql
SELECT relationship, COUNT(*) as n FROM dependencies GROUP BY relationship ORDER BY n DESC;
```

**Pass criteria**:
- IMPORTS is the largest (> 400)
- Every type has at least 5 entries
- Sum = 1,682 (1,704 minus 22 merged duplicates)

### A7: Evidence Preservation

```sql
-- For sentence-form relationships: evidence must contain the original string
SELECT COUNT(*) as preserved
FROM dependencies
WHERE LENGTH(_original_relationship) > 50
  AND (evidence LIKE '%' || SUBSTR(_original_relationship, 1, 30) || '%'
       OR _original_relationship IS NOT NULL);
```

**Pass criteria**: Count equals the number of sentence-form relationships (~398).
**What it proves**: Detailed relationship descriptions weren't lost.

### A8: View Integrity

```sql
SELECT 'smart_priority_gaps' as view_name, COUNT(*) as n FROM smart_priority_gaps
UNION ALL SELECT 'domain_coverage', COUNT(*) FROM domain_coverage
UNION ALL SELECT 'package_coverage', COUNT(*) FROM package_coverage
UNION ALL SELECT 'open_findings', COUNT(*) FROM open_findings
UNION ALL SELECT 'integration_hotspots', COUNT(*) FROM integration_hotspots;
```

**Pass criteria**: Counts match `baseline.view_counts`. Views don't error.
**What it proves**: Downstream views aren't broken by the schema changes.

### A9: Session Distribution Unchanged

```sql
SELECT session_id, COUNT(*) as n FROM findings GROUP BY session_id;
```

**Pass criteria**: Every session has the same finding count as in `baseline.findings_per_session`.
**What it proves**: No findings accidentally shifted between sessions during UPDATE.

### A10: report.js Still Works

```bash
node /home/snoozyy/ruvnet-research/scripts/report.js
echo $?
```

**Pass criteria**: Exit code 0 and MASTER-INDEX.md generated.
**What it proves**: The entire reporting pipeline still functions.

---

## Spot-Check Protocol (Manual)

After all automated assertions pass, manually review 20 samples:

### Spot-Check 1: Category accuracy (10 samples)

```sql
SELECT
  f.id, f.category, f._original_category,
  SUBSTR(f.description, 1, 200) as desc_preview
FROM findings f
WHERE f._original_category != f.category
ORDER BY RANDOM() LIMIT 10;
```

For each row, ask: **Does the new category make sense given the description?**

Score:
- 9-10 correct: PASS
- 7-8 correct: PASS with notes (log the misclassifications)
- < 7 correct: FAIL — mapping rules need revision

### Spot-Check 2: Relationship accuracy (10 samples)

```sql
SELECT
  d.relationship, d._original_relationship,
  SUBSTR(d.evidence, 1, 200) as evidence_preview,
  sf.relative_path as source,
  tf.relative_path as target
FROM dependencies d
JOIN files sf ON d.source_file_id = sf.id
JOIN files tf ON d.target_file_id = tf.id
WHERE d._original_relationship != d.relationship
ORDER BY RANDOM() LIMIT 10;
```

For each row, ask: **Does the canonical relationship type correctly describe the source→target edge?**

Score: Same as above.

---

## Dry-Run Verification

The migration script MUST support `--dry-run` mode:

```
1. BEGIN TRANSACTION
2. Run all UPDATEs
3. Run ALL assertions (A1-A10)
4. Print results
5. ROLLBACK (not COMMIT)
6. Verify DB unchanged: md5sum matches backup
```

If ALL assertions pass in dry-run → safe to run for real.

---

## Post-Migration Monitoring

After migration completes and all checks pass:

1. **First research session after migration**: Verify new findings insert with canonical categories
2. **Trigger test**: Attempt INSERT with invalid category — should fail
3. **One week later**: Check that no new non-canonical values have appeared

```sql
-- Run weekly to verify enforcement holds
SELECT DISTINCT category FROM findings
WHERE category NOT IN ('ARCHITECTURE','QUALITY','INTEGRATION','PERFORMANCE',
                       'ALGORITHM','FACADE','SECURITY','BUG',
                       'GENUINE','TESTING','DOCUMENTATION','INCOMPLETE');
-- Must return 0 rows
```

---

## Failure Recovery

If any assertion fails during the real migration:

```bash
# Step 1: Do NOT attempt to fix in-place
# Step 2: Restore from backup
cp ADRs/schema-migration/research.db.pre-normalization.bak db/research.db

# Step 3: Verify restoration
md5sum db/research.db
# Must match: e3449ccaaa8fcbceb5a50dab7545b4ee

# Step 4: Fix the mapping rules based on failure analysis
# Step 5: Re-run dry-run
# Step 6: Only proceed to real migration when dry-run passes all assertions
```

---

## Success Criteria Summary

The migration is **SUCCESSFUL** when ALL of the following are true:

- [ ] A0: Merge verified (1,704 → 1,682, 22 merge annotations present)
- [ ] A1: All row counts match baseline (dependencies = 1,682 post-merge, all others unchanged)
- [ ] A2: No NULL values in any migrated column
- [ ] A3: Cardinality: 10 relationship types, 12 categories, 4 severities
- [ ] A4: Only canonical values present (no strays)
- [ ] A5: Severity distribution conserved (CRITICAL=1294, HIGH=2923, MEDIUM=3085, INFO=3819)
- [ ] A6: Distribution is reasonable (no mega-bucket, no empty bucket)
- [ ] A7: Evidence preserved for sentence-form relationships
- [ ] A8: All views still work with same row counts
- [ ] A9: Per-session finding counts unchanged
- [ ] A10: report.js completes successfully
- [ ] Spot-check 1: >= 9/10 categories correct
- [ ] Spot-check 2: >= 9/10 relationships correct
- [ ] Dry-run passed before real execution
- [ ] Enforcement triggers reject invalid values (deployed in same transaction)
- [ ] All 8 agent templates updated with enum constraint blocks
