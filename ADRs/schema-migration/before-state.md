# Before-State Snapshot

> **Date**: 2026-02-20
> **DB Checksum (MD5)**: `e3449ccaaa8fcbceb5a50dab7545b4ee`
> **Backup Location**: `research.db.pre-normalization.bak` (same directory)

## Row Counts (Invariant — must NOT change during migration)

| Table | Count |
|-------|-------|
| files | 14,633 |
| findings | 11,121 |
| dependencies | 1,704 |
| sessions | 114 |
| file_reads | 1,850 |
| domains | 17 |
| file_domains | 15,939 |
| packages | 11 |
| exclude_paths | 64 |

**Verification**: After migration, ALL row counts must remain identical. The migration changes column values, never adds or deletes rows.

## Column Cardinality (Before → After targets)

| Column | Distinct Values (Before) | Target (After) |
|--------|-------------------------|----------------|
| `dependencies.relationship` | 511 | ~10 |
| `findings.category` | 1,211 | ~12 |
| `findings.severity` | 7 | 4 |

## Severity Distribution (Before)

| Severity | Count | Note |
|----------|-------|------|
| INFO | 3,795 | Clean |
| MEDIUM | 3,085 | Clean |
| HIGH | 2,923 | Clean |
| CRITICAL | 1,294 | Clean |
| LOW | 21 | Remap → INFO |
| POSITIVE | 2 | Remap → INFO |
| LOW_CONFIDENCE | 1 | Remap → INFO |

## Exported Artifacts

All in `ADRs/schema-migration/`:

| File | Purpose |
|------|---------|
| `research.db.pre-normalization.bak` | Full DB backup (7.7MB) |
| `before-relationships.csv` | All 511 relationship types + counts |
| `before-categories.csv` | All 1,211 category types + counts |
| `before-severities.csv` | All 7 severity values + counts |
| `before-row-counts.json` | Table row counts (machine-readable) |
| `before-schema.sql` | Full schema snapshot (14 tables, 12 views) |
| `relationships-full.json` | All 511 relationships with counts |
| `categories-full.json` | All 1,211 categories with severity distribution |

## Rollback Procedure

If migration fails or produces incorrect results:

```bash
cp ADRs/schema-migration/research.db.pre-normalization.bak db/research.db
# Verify:
md5sum db/research.db  # Must match: e3449ccaaa8fcbceb5a50dab7545b4ee
```
