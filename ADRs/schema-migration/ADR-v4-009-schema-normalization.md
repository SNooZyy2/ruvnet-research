# ADR-v4-009: Research Database Schema Normalization

> **Status**: Implemented (2026-02-24)
> **Date**: 2026-02-20
> **Deciders**: Research project lead
> **Related**: ADR-v4-007 (Subsystem Graph), ADR-038 (Research Database System)

---

## Context

After 114 research sessions, the `findings` and `dependencies` tables have accumulated
severe data quality problems due to unconstrained free-text fields:

| Problem | Scope | Impact |
|---------|-------|--------|
| 511 distinct `relationship` types (78% singletons) | 1,704 dependency rows | Cannot aggregate or query by type |
| 1,211 distinct `category` values | 11,121 finding rows | Case drift, synonym explosion |
| 7 `severity` values (should be 4) | 11,121 finding rows | 3 stray values leak into queries |
| Evidence jammed into `relationship` field | ~398 rows | Sentences where enum values should be |
| 22 duplicate dependency pairs | 44 rows → 22 | Case/synonym variants for same file pair violate UNIQUE after normalization |

This makes the data unsuitable for the subsystem graph (ADR-v4-007), any future vector
embedding layer, or basic aggregate queries like "how many FACADE findings exist?"

### Root Cause

Research agent templates (`agents/reader.md` etc.) insert free-text strings into fields
that should be constrained enums. No validation exists at write-time.

## Decision

Normalize the three affected columns via a **3-phase migration**:

1. **Backfill**: Normalize existing data in-place using deterministic mapping rules
2. **Schema enforcement**: Add CHECK constraints to prevent future drift
3. **Agent template update**: Add enum constraints to all agent templates + DB triggers

---

## Phase 1: Canonical Enums

### 1.1 Relationship Types (511 → 10)

| Canonical | Definition | Absorbs |
|-----------|-----------|---------|
| `IMPORTS` | A uses B's exports at runtime | imports, IMPORTS, import, imports_from, imports_type, imports_types, imports_class, direct-import, type-imports, type_import, imports_trait, imports_config, imports_error_types, imports_persistence, imports_solver, imports_simd_ops, imports_wasm_bridge, imports_wasm_loader, imports_csr_storage, imports_gnn_service, imports_precision_type, imports_spawns, imports_broken, imports_error_type, lazy_import, optional_import, mod_use_import, pub_mod_import, transitive_import, dynamic-import, test_imports, imports-types, imports-and-calls, imports-facade, imports-unused, imports-extensively, imports-for-initialization, imports-uses, imports-domain-model, import-Pattern-StorageStats, imported-by, imported-but-unused, imported_by, IMPORTED_BY, IMPORTS_AND_REEXPORTS, IMPORTS_TYPE, MODULE_IMPORT, **all 100+ sentence-form "imports ..." strings** |
| `USES` | A calls/instantiates B's API | uses, USES, used_by, USED_BY, used-by, calls, calls_method, implements, extends, EXTENDS, instantiates, requires, depends_on, DEPENDS_ON, depends-on, depends, DEPENDS_ON_TYPES, consumed_by, CONSUMED_BY, consumes, uses_types, uses_type, uses_struct, uses_api, uses_constant, uses_wrapper, uses_prompt_templates, uses-types, uses-impl, uses-concept, uses-for-embeddings, type_consumer, **all sentence-form "uses ..." strings** |
| `EXPORTS` | A exposes API/types for others | exports, EXPORTS, re-exports, re-export, BARREL_EXPORT, pub_use, pub_mod_reexport, module-reexport, INDIRECT_EXPORT, CONDITIONAL_EXPORT, exported_by, EXPORTED_FROM, EXPORTS_AS_NAMESPACE, EXPORTS_TO, exports_from, exports_to, exports-to, exported-by, reexported_by, pub_re_export, pub_reexport_indirect, RE_EXPORTS_MODULE, RE_EXPORTS, RE_EXPORTS_ALL, RE_EXPORTS_TYPE, module-export, module_export, MODULE_EXPORT, module:exported, **all sentence-form "exports ..." / "re-exports ..." / "pub use ..." strings** |
| `DECLARES` | A declares/defines module B | pub_module_declaration, pub_mod, pub mod, declares_module, declares, mod-declaration, declares-submodule, module-declares, module-definition, module-parent, module_declaration, defines_module, cfg_module, submodule, always-on submodule, feature-gated submodule, DEFINES, declared-in, declared_in, member-of, member_of, part-of, mod-wasm, **all sentence-form "pub mod ..." strings** |
| `SIBLINGS` | A and B are co-modules at same level | module-sibling, sibling, sibling-module, SIBLING_MODULE, SIBLING, SIBLING_SCENARIO, co-module, peer, PEER, sibling-kernel, sibling_impl, sibling_module, sibling-cli-mcp, cohomology-sibling, same-package, co-located, related, RELATED_IMPLEMENTATION |
| `COMPETES` | A and B independently implement same thing | parallel-implementation, parallel_implementation, PARALLEL_IMPLEMENTATION, parallel-impl, parallel_impl, parallel-system, parallel_system, parallel_to, parallel, parallel_api, parallel_simulation_pattern, parallel implementation, reimplements, REIMPLEMENTS, alternative-implementation, alternative_to, ALTERNATIVE_LOADER, mirrors, mirrors_architecture, signature_mirror, duplicate_of, duplicates_logic, COPY_OF_REFACTORED, algorithmic-parallel, conceptual-parallel, NAME_COLLISION, naming-conflict, ARCHITECTURAL_CONTRAST, CONTRASTS_WITH, contrasts_with, compared-to, comparison |
| `WRAPS` | A provides higher-level API over B | wraps, WRAPS, wraps_native, wraps_deprecated, wrapped_by, delegates_to, delegates-to, delegates, delegates-storage, delegates_to_via_npx, facade-of, FACADE_OF, **all sentence-form "wraps ..." / "delegates ..." strings** |
| `FEEDS` | A produces data consumed by B | feeds, GENERATES, produces, produces_for, RECEIVES_DATA_FROM, reads, reads-writes, queries, configures, registers_tools_from, orchestrates, ORCHESTRATED_BY, spawns-process, spawns_mcp_subprocess, loads, loads-config-from, loads_wasm_module, loads compiled WASM, LOADS_WASM_INTERFACE, embeds_schema, **all sentence-form data-flow strings** |
| `TESTS` | A tests/validates B | tests, TESTS, TESTED_BY, tested_by, tested-by integration tests, TESTS_VARIANT_OF, TEST_SUITE, tests_via_spawn, tests_performance, benchmarks, related-benchmark, VALIDATES_AGAINST, internal-validation, test-dependency |
| `BROKEN` | Intended dependency that doesn't work | bypasses, MISSING_INTEGRATION, missing_integration, SHOULD_INTEGRATE, should_import, should-import-but-reimplements, should_call_but_doesnt, should_integrate_but_orphaned, expects_integration, potential-integration, ISOLATED_FROM, CONTRADICTS, VIOLATES, HONEST_ALTERNATIVE_TO, declares_dependency_but_never_uses, calls_broken, imports_broken, supports_but_facades, orphaned-module, orphaned_from, incomplete_port, js_fallback_for, replaced-by, replaces, supersedes, patched_version_of, improved_version_of |

**Unmapped catch-all**: Any relationship string not matching the above → normalize to `USES` (most are detailed import descriptions that are functionally imports/uses). The original string always preserved in evidence.

#### Sentence-form relationship handling

The 398 singleton relationship strings like `"imports ChatMessage, ChatTemplate, RuvTokenizer (line 53)..."` are handled as follows:

1. Scan the string for keyword signals: starts with "imports" → `IMPORTS`, contains "delegates" → `WRAPS`, contains "parallel" → `COMPETES`, etc.
2. Move the **full original string** to the `evidence` column (appended, not replaced)
3. Set `relationship` to the detected canonical type

Priority order for keyword detection:
```
BROKEN:   "should", "missing", "broken", "bypass", "violat", "orphan", "isolated"
COMPETES: "parallel", "alternative", "reimpl", "mirror", "duplicate", "compet"
WRAPS:    "wraps", "delegat", "facade"
FEEDS:    "produces", "feeds", "loads", "consumes", "spawns", "generates", "orchestrat"
TESTS:    "test", "benchmark", "validat"
EXPORTS:  "export", "re-export", "pub use", "pub mod.*re-export"
DECLARES: "pub mod", "declares", "module"
IMPORTS:  "imports", "import"
USES:     (default fallback)
```

### 1.2 Finding Categories (1,211 → 12)

| Canonical | Count (est.) | Absorbs |
|-----------|-------------|---------|
| `ARCHITECTURE` | ~2,800 | architecture, ARCHITECTURE, Architecture, design, DESIGN, Design, api-design, API_DESIGN, api_design, interface-design, INTERFACE_DESIGN, interface_design, data-model, DATA_MODEL, configuration, CONFIGURATION, Configuration, config, CONFIG, infrastructure, INFRASTRUCTURE, protocol, PROTOCOL, protocols, PROTOCOLS, data-structures, DATA_STRUCTURES, data-structure, DATA_STRUCTURE, data_structure, API_SURFACE, api-surface, INTERFACE, interface, DATA_MODEL |
| `QUALITY` | ~2,400 | quality, QUALITY, Quality, code-quality, CODE_QUALITY, Code Quality, code_quality, implementation, IMPLEMENTATION, Implementation, implementation-quality, naming, NAMING, SIMPLIFICATION, simplification, best-practice, ANTI_PATTERN, anti-pattern, DESIGN_PATTERN, design-pattern, ASSESSMENT, types, TYPES, HELPER, FEATURE, feature, Feature, PORTABILITY, portability, FUNCTIONALITY, functionality, api, API |
| `INTEGRATION` | ~750 | integration, INTEGRATION, Integration, dependency, DEPENDENCY, dependencies, DEPENDENCIES, missing-integration, MISSING_INTEGRATION, Missing Integration, disconnected, DISCONNECTED, cross-reference, CROSS_REFERENCE, data-flow, DATA_FLOW, api-mismatch, API_MISMATCH, missing-feature, MISSING_FEATURE, broken-dep, BROKEN_DEP, broken, BROKEN, ORPHANED, orphaned, Orphaned |
| `PERFORMANCE` | ~530 | performance, PERFORMANCE, Performance, optimization, OPTIMIZATION, concurrency, CONCURRENCY, memory, MEMORY, Memory, SCALABILITY |
| `ALGORITHM` | ~430 | algorithm, ALGORITHM, Algorithm, algorithm-quality, ALGORITHM_QUALITY, algorithm-correctness, ALGORITHM_CORRECTNESS, numerical-stability, NUMERICAL_STABILITY, logic, LOGIC, LEGITIMATE_ALGORITHM, algorithm-real |
> **Note**: `ACCURACY` (8 rows) maps to BUG, not ALGORITHM. Most accuracy findings describe
> correctness errors (wrong values, misleading docs), not algorithmic design choices.
| `FACADE` | ~590 | facade, FACADE, Facade, stub, STUB, Stub, placeholder, PLACEHOLDER, fabrication, FABRICATION, fabricated, FABRICATED, deception, DECEPTION, misleading, MISLEADING, theatrical, THEATRICAL, PHANTOM_WRAPPER, FACADE_PATTERN, facade-pattern, facade_pattern, FACADE_DECEPTION, facade-deception, QUANTUM_FACADE, ARCHITECTURE_FACADE, FALSE_COMPLEXITY, fabricated-data, FABRICATED_DATA, fabricated-metrics, FABRICATED_METRICS, data-generation, Demonstration, demonstration, DEMONSTRATION |
| `SECURITY` | ~171 | security, SECURITY |
| `BUG` | ~378 | bug, BUG, correctness, CORRECTNESS, error-handling, ERROR_HANDLING, Error Handling, Error_Handling, error_handling, ACCURACY, accuracy, FALLBACK, fallback, ROBUSTNESS, robustness |
| `GENUINE` | ~360 | genuine, GENUINE, Genuine, GENUINE_IMPLEMENTATION, genuine-implementation, GENUINE_SIGNAL, GENUINE_INFRASTRUCTURE, genuine-infrastructure, GENUINE_ALGORITHM, GENUINE_ALGORITHMS, genuine-algorithms, REAL, real, REAL_CODE, REAL_IMPLEMENTATION, real-implementation, REAL_WASM, real-wasm, INNOVATION, realness, positive, POSITIVE, CORRECT_IMPLEMENTATION, correct-implementation |
| `TESTING` | ~190 | testing, TESTING, Testing, test-coverage, TEST_COVERAGE, test_coverage, test-quality, TEST_QUALITY, comparison, COMPARISON, benchmark-quality, ANALYSIS_QUALITY, analysis-quality, validation, VALIDATION |
| `DOCUMENTATION` | ~310 | documentation, DOCUMENTATION, Documentation, completeness, COMPLETENESS, observability, OBSERVABILITY, Observability, metrics, METRICS, Metrics, reporting, REPORTING, ROADMAP |
| `INCOMPLETE` | ~320 | incomplete, INCOMPLETE, Incomplete, incomplete-implementation, INCOMPLETE_IMPLEMENTATION, INCOMPLETE_IMPL, missing-implementation, MISSING_IMPLEMENTATION, Missing Implementation, gap, GAP, Gap, limitation, LIMITATION, dead-code, DEAD_CODE, CONFIG_GAP, TYPE_SAFETY, type-safety, TYPE_SAFETY, RELIABILITY, reliability |

**Unmapped catch-all**: Pattern match strategy for the ~900 tiny categories.
Rules are evaluated **in priority order** — first match wins. This means keywords
appearing in multiple rules always resolve to the higher-priority bucket.

1. If it contains "facade", "stub", "placeholder", "fabricat", "decep", "mislead", "theatrical" → `FACADE`
2. If it contains "genuine", "real", "innovat", "positive", "correct-impl" → `GENUINE`
3. If it contains "integrat", "depend", "disconnect", "orphan", "broken", "missing" → `INTEGRATION`
4. If it contains "algorithm", "numeric", "math", "correctness" → `ALGORITHM`
5. If it contains "test", "benchmark", "coverage", "validation" → `TESTING`
6. If it contains "secur", "vulnerab", "crypto" → `SECURITY`
7. If it contains "perform", "optim", "memory", "concurr", "latency" → `PERFORMANCE`
8. If it contains "doc", "comment", "readme", "completeness" → `DOCUMENTATION`
9. If it contains "incomplet", "gap", "dead", "limit", "missing" → `INCOMPLETE`
10. If it contains "bug", "error", "broken", "fault", "crash" → `BUG`
11. If it contains "architect", "design", "pattern", "config", "struct" → `ARCHITECTURE`
12. Default → `QUALITY`

> **Keyword overlap notes** (conscious tradeoffs in priority order):
>
> These only apply to the ~900 unmapped catch-all categories. Exact absorb-list matches
> (section 1.2 table) are resolved BEFORE pattern matching runs. For example, bare
> `correctness` → BUG via absorb list, even though the pattern rule below would send it
> to ALGORITHM. The overlap only matters for strings like `"numerical-correctness-issue"`
> that don't appear in any absorb list.
>
> - `"missing"` → INTEGRATION (step 3) beats INCOMPLETE (step 9). Rationale: most "missing-X"
>   findings describe missing integrations between subsystems, not incomplete implementations.
>   Exception: absorb list entries like "missing-implementation" → INCOMPLETE bypass this.
> - `"broken"` → INTEGRATION (step 3) beats BUG (step 10). Rationale: "broken" in this codebase
>   almost always means broken cross-module wiring, not runtime bugs.
>   Exception: absorb list entries like "broken" → INTEGRATION, "broken-dep" → INTEGRATION.
> - `"correctness"` → ALGORITHM (step 4) beats BUG (step 10) **in pattern matching only**.
>   Note: bare `correctness` and `CORRECTNESS` are in BUG's absorb list and go to BUG.
>   This rule catches compound strings like `"numerical-correctness-gap"`.
> - `"validation"` → TESTING (step 5) beats BUG (step 10). Rationale: validation findings are
>   about test/validation coverage, not error-handling bugs.

### 1.3 Severity (7 → 4)

| Before | Count | Maps To | Rationale |
|--------|-------|---------|-----------|
| CRITICAL | 1,294 | CRITICAL | Clean |
| HIGH | 2,923 | HIGH | Clean |
| MEDIUM | 3,085 | MEDIUM | Clean |
| INFO | 3,795 | INFO | Clean |
| LOW | 21 | INFO | LOW never defined in protocol, semantically = INFO |
| POSITIVE | 2 | INFO | Positive observations, not actionable findings |
| LOW_CONFIDENCE | 1 | INFO | Confidence qualifier, not severity |

---

## Phase 2: Migration Script Design

### 2.1 Schema Changes

```sql
-- Preserve originals for audit trail
ALTER TABLE dependencies ADD COLUMN _original_relationship TEXT;
ALTER TABLE findings ADD COLUMN _original_category TEXT;
ALTER TABLE findings ADD COLUMN _original_severity TEXT;

-- After backfill, add constraints
-- (Only after verifying all values are canonical)
```

### 2.2 Dry-Run Mode

The migration script supports `--dry-run` which:
1. Opens the DB in WAL mode (normal)
2. Runs the full migration inside a transaction
3. Prints all verification results (Phase 3 checks)
4. **Rolls back the transaction** — no changes persisted
5. Exits with code 0 if all checks passed, 1 otherwise

This allows validating the migration without risk. Run `node scripts/migrate-enums.js --dry-run`
before `node scripts/migrate-enums.js --execute`.

### 2.3 Rollback Procedure

> **Note**: Rollback must also account for the 22 merged dependency rows.
> The pre-migration snapshot (`research.db.pre-migration`) contains all 1,704 rows.
> Restoring from snapshot is the cleanest rollback path.

If the migration fails or produces unexpected results:

```bash
# Option A: Transaction was never committed (script aborted mid-run)
# → SQLite auto-rolls back. DB is unchanged. Re-run after fixing the script.

# Option B: Migration committed but verification fails post-hoc
# → Restore from pre-migration snapshot:
cp db/research.db.pre-migration db/research.db

# Option C: Migration committed, want to undo just the normalization
# → Reverse from _original_* columns.
# NOTE: Must drop enforcement triggers first (they reject old values like 'LOW').
node -e "
const db = require('better-sqlite3')('db/research.db');
db.exec('BEGIN');
// Drop triggers that would block restoring non-canonical values
db.exec('DROP TRIGGER IF EXISTS enforce_relationship_enum_insert');
db.exec('DROP TRIGGER IF EXISTS enforce_relationship_enum_update');
db.exec('DROP TRIGGER IF EXISTS enforce_category_enum_insert');
db.exec('DROP TRIGGER IF EXISTS enforce_category_enum_update');
db.exec('DROP TRIGGER IF EXISTS enforce_severity_enum_insert');
db.exec('DROP TRIGGER IF EXISTS enforce_severity_enum_update');
// Restore original values
db.exec('UPDATE dependencies SET relationship = _original_relationship WHERE _original_relationship IS NOT NULL');
db.exec('UPDATE findings SET category = _original_category WHERE _original_category IS NOT NULL');
db.exec('UPDATE findings SET severity = _original_severity WHERE _original_severity IS NOT NULL');
db.exec('COMMIT');
console.log('Rollback complete (triggers removed, original values restored)');
db.close();
"
```

The pre-migration snapshot **must** be created before running the migration:
```bash
cp db/research.db db/research.db.pre-migration
```

### 2.4 Pre-Normalization Deduplication

The `dependencies` table has a `UNIQUE(source_file_id, target_file_id, relationship)` constraint.
After normalization, 22 file pairs would have duplicate canonical relationship types (e.g.,
`imports` + `IMPORTS` both become `IMPORTS` for the same file pair), violating this constraint.

These must be merged **before** the normalization UPDATE:

1. Detect all 22 collision pairs (pairs where 2+ relationships map to the same canonical type)
2. For each collision: keep the row with the longer `evidence` field
3. Append the loser's original relationship to the survivor's evidence:
   `" [merged: <loser_relationship>]"`
4. DELETE the loser row

This reduces `dependencies` from 1,704 → 1,682 rows (22 genuine duplicates removed).

All 22 collisions are:
- Case variants: `imports` vs `IMPORTS`, `implements` vs `IMPLEMENTS` (pure duplicates)
- Near-identical sentences: slightly different wording of the same relationship
- Semantic duplicates: `uses` + `uses_types` for the same file pair

No information is lost — the merged row retains the longer evidence and the loser's
original relationship is appended.

### 2.5 Migration Steps (single transaction)

```
BEGIN TRANSACTION;

-- Step 0: Merge 22 duplicate dependency pairs (before normalization)
-- For each collision pair: keep row with longer evidence, append loser's relationship
-- to survivor's evidence, DELETE loser row.
-- Verify: SELECT COUNT(*) FROM dependencies → must go from 1,704 to 1,682

-- Step 1: Save originals
UPDATE dependencies SET _original_relationship = relationship;
UPDATE findings SET _original_category = category;
UPDATE findings SET _original_severity = severity WHERE severity NOT IN ('CRITICAL','HIGH','MEDIUM','INFO');

-- Step 2: Normalize severities (smallest, safest change)
-- Verify: SELECT COUNT(*) WHERE severity NOT IN (4 canonical) → must be 24 before, 0 after

-- Step 3: Normalize categories (large, pattern-based)
-- Verify: SELECT COUNT(DISTINCT category) → must go from 1211 to 12

-- Step 4: Normalize relationships (most complex — sentence extraction)
-- Verify: SELECT COUNT(DISTINCT relationship) → must go from 511 to 10

-- Step 5: Run all verification queries (see Phase 3)

-- Step 6: Create enforcement triggers (6 triggers: INSERT + UPDATE for each column)
-- See Phase 4 section 4.1 for trigger definitions
-- Deploying in same transaction ensures zero window for drift

COMMIT;  -- only if all checks pass
```

### 2.6 Evidence Preservation

For sentence-form relationships, the migration:
1. Copies the original string to `_original_relationship` (already done in Step 1)
2. **Appends** to `evidence` column: `" [migrated from relationship: <original>]"`
3. Overwrites `relationship` with the canonical type

This means NO information is lost. The original can always be recovered from either `_original_relationship` or `evidence`.

---

## Phase 3: Verification Plan

Every step has a built-in proof of correctness. The migration script prints all of these.

### 3.1 Invariant Checks (must ALL pass)

| Check | Query | Expected |
|-------|-------|----------|
| Total findings unchanged | `SELECT COUNT(*) FROM findings` | 11,121 (findings are never merged or deleted) |
| Total dependencies (post-merge) | `SELECT COUNT(*) FROM dependencies` | 1,682 (was 1,704 — 22 duplicate pairs merged) |
| Total sessions unchanged | `SELECT COUNT(*) FROM sessions` | 114 |
| Total files unchanged | `SELECT COUNT(*) FROM files` | 14,633 |
| Total file_reads unchanged | `SELECT COUNT(*) FROM file_reads` | 1,850 |
| No NULL relationships | `SELECT COUNT(*) FROM dependencies WHERE relationship IS NULL` | 0 |
| No NULL categories | `SELECT COUNT(*) FROM findings WHERE category IS NULL` | 0 |
| No NULL severities | `SELECT COUNT(*) FROM findings WHERE severity IS NULL` | 0 |
| Originals preserved | `SELECT COUNT(*) FROM dependencies WHERE _original_relationship IS NULL` | 0 |
| Originals preserved | `SELECT COUNT(*) FROM findings WHERE _original_category IS NULL` | 0 |

| Stray severities preserved | `SELECT COUNT(*) FROM findings WHERE _original_severity IN ('LOW','POSITIVE','LOW_CONFIDENCE')` | 24 (all stray-severity originals captured) |

### 3.2 Cardinality Checks

| Check | Query | Expected |
|-------|-------|----------|
| Relationship cardinality | `SELECT COUNT(DISTINCT relationship) FROM dependencies` | 10 |
| Category cardinality | `SELECT COUNT(DISTINCT category) FROM findings` | 12 |
| Severity cardinality | `SELECT COUNT(DISTINCT severity) FROM findings` | 4 |
| All relationships canonical | `SELECT DISTINCT relationship FROM dependencies` | Exactly the 10 defined values |
| All categories canonical | `SELECT DISTINCT category FROM findings` | Exactly the 12 defined values |
| All severities canonical | `SELECT DISTINCT severity FROM findings` | CRITICAL, HIGH, MEDIUM, INFO |

### 3.3 Distribution Sanity Checks

These verify the migration didn't accidentally lump everything into one bucket:

| Check | Query | Constraint |
|-------|-------|-----------|
| Largest category | `SELECT category, COUNT(*) ... ORDER BY cnt DESC LIMIT 1` | Must be < 35% of total (< 3,892) |
| Smallest category | `SELECT category, COUNT(*) ... ORDER BY cnt ASC LIMIT 1` | Must be > 0 |
| FACADE count | `SELECT COUNT(*) FROM findings WHERE category = 'FACADE'` | Must be 400-800 (was ~590 across synonyms) |
| GENUINE count | `SELECT COUNT(*) FROM findings WHERE category = 'GENUINE'` | Must be 200-500 (was ~360 across synonyms) |
| IMPORTS count | `SELECT COUNT(*) FROM dependencies WHERE relationship = 'IMPORTS'` | Must be 490-690 (was 534+ across variants, minus ~8 merged duplicates) |
| Singletons gone | `SELECT COUNT(*) FROM (SELECT relationship, COUNT(*) c FROM dependencies GROUP BY relationship HAVING c = 1)` | 0 (no more singletons) |
| Dep sum correct | `SELECT SUM(cnt) FROM (SELECT COUNT(*) cnt FROM dependencies GROUP BY relationship)` | 1,682 (1,704 minus 22 merged) |

### 3.4 View Audit (pre-migration, completed)

All 10 views were audited for hardcoded category/relationship/severity string literals.

| View | References enum columns? | Hardcoded values? | Risk |
|------|------------------------|-------------------|------|
| `open_findings` | severity, category | `severity IN ('CRITICAL','HIGH')` | **None** — uses canonical uppercase |
| `unverified_deps` | relationship | No filter on value | None |
| `analysis_coverage` | — | depth only | None |
| `domain_coverage` | — | depth only | None |
| `package_coverage` | — | depth only | None |
| `package_connectivity` | — | No enum refs | None |
| `subtree_connectivity` | — | depth only | None |
| `smart_priority_gaps` | — | depth only | None |
| `priority_gaps` | — | depth only | None |
| `integration_hotspots` | — | depth only | None |

**Result**: No views use old/non-canonical enum values. Migration is safe for all views.

### 3.5 Cross-Reference Checks (post-migration)

Verify that data the rest of the system depends on is intact:

| Check | Query | Expected |
|-------|-------|----------|
| smart_priority_gaps view works | `SELECT COUNT(*) FROM smart_priority_gaps` | Same as before migration |
| domain_coverage view works | `SELECT COUNT(*) FROM domain_coverage` | Same as before (17) |
| package_coverage view works | `SELECT COUNT(*) FROM package_coverage` | Same as before |
| open_findings view works | `SELECT COUNT(*) FROM open_findings` | Same as before |
| integration_hotspots view works | `SELECT COUNT(*) FROM integration_hotspots` | Same as before |
| report.js succeeds | `node scripts/report.js` | Exit code 0, generates MASTER-INDEX.md |

### 3.6 Spot-Check Protocol

After automated checks pass, manually inspect 20 random samples:

```sql
-- 10 random findings: does the new category make sense given the description?
SELECT category, _original_category, SUBSTR(description, 1, 150)
FROM findings ORDER BY RANDOM() LIMIT 10;

-- 10 random dependencies: was evidence preserved?
SELECT relationship, _original_relationship, SUBSTR(evidence, -100)
FROM dependencies WHERE _original_relationship != relationship
ORDER BY RANDOM() LIMIT 10;
```

### 3.7 Regression Test (post-migration)

Run the exact queries from CLAUDE.md "Query Recipes" section and verify they still work:

1. `SELECT * FROM smart_priority_gaps WHERE tier_rank <= 2 LIMIT 10` — returns rows
2. `SELECT * FROM domain_coverage` — returns 17 rows
3. `SELECT * FROM package_coverage` — returns rows
4. `SELECT * FROM open_findings WHERE severity = 'CRITICAL'` — count matches pre-migration CRITICAL count (1,294)
5. `SELECT * FROM integration_hotspots` — returns rows
6. `SELECT * FROM subtree_connectivity WHERE confidence = 'RELIABLE'` — returns rows

---

## Phase 4: Schema Enforcement

> **Note**: The 6 enforcement triggers below are now created inside the migration
> transaction (Step 6 in section 2.5). This ensures zero window for drift between
> data normalization and constraint enforcement. Phase 4 post-migration work is
> limited to the inline validation helper (4.2) and agent template updates (4.3).

### 4.1 Enforcement Triggers (deployed in migration transaction)

```sql
-- Created in Step 6 of the migration transaction (section 2.5)
-- INSERT triggers
CREATE TRIGGER IF NOT EXISTS enforce_relationship_enum_insert
BEFORE INSERT ON dependencies
BEGIN
  SELECT CASE
    WHEN NEW.relationship NOT IN ('IMPORTS','USES','EXPORTS','DECLARES','SIBLINGS',
         'COMPETES','WRAPS','FEEDS','TESTS','BROKEN')
    THEN RAISE(ABORT, 'Invalid relationship type. Must be one of: IMPORTS, USES, EXPORTS, DECLARES, SIBLINGS, COMPETES, WRAPS, FEEDS, TESTS, BROKEN')
  END;
END;

CREATE TRIGGER IF NOT EXISTS enforce_category_enum_insert
BEFORE INSERT ON findings
BEGIN
  SELECT CASE
    WHEN NEW.category NOT IN ('ARCHITECTURE','QUALITY','INTEGRATION','PERFORMANCE',
         'ALGORITHM','FACADE','SECURITY','BUG','GENUINE','TESTING','DOCUMENTATION','INCOMPLETE')
    THEN RAISE(ABORT, 'Invalid category. Must be one of: ARCHITECTURE, QUALITY, INTEGRATION, PERFORMANCE, ALGORITHM, FACADE, SECURITY, BUG, GENUINE, TESTING, DOCUMENTATION, INCOMPLETE')
  END;
END;

CREATE TRIGGER IF NOT EXISTS enforce_severity_enum_insert
BEFORE INSERT ON findings
BEGIN
  SELECT CASE
    WHEN NEW.severity NOT IN ('CRITICAL','HIGH','MEDIUM','INFO')
    THEN RAISE(ABORT, 'Invalid severity. Must be one of: CRITICAL, HIGH, MEDIUM, INFO')
  END;
END;

-- UPDATE triggers (without these, UPDATE statements bypass validation)
CREATE TRIGGER IF NOT EXISTS enforce_relationship_enum_update
BEFORE UPDATE OF relationship ON dependencies
BEGIN
  SELECT CASE
    WHEN NEW.relationship NOT IN ('IMPORTS','USES','EXPORTS','DECLARES','SIBLINGS',
         'COMPETES','WRAPS','FEEDS','TESTS','BROKEN')
    THEN RAISE(ABORT, 'Invalid relationship type. Must be one of: IMPORTS, USES, EXPORTS, DECLARES, SIBLINGS, COMPETES, WRAPS, FEEDS, TESTS, BROKEN')
  END;
END;

CREATE TRIGGER IF NOT EXISTS enforce_category_enum_update
BEFORE UPDATE OF category ON findings
BEGIN
  SELECT CASE
    WHEN NEW.category NOT IN ('ARCHITECTURE','QUALITY','INTEGRATION','PERFORMANCE',
         'ALGORITHM','FACADE','SECURITY','BUG','GENUINE','TESTING','DOCUMENTATION','INCOMPLETE')
    THEN RAISE(ABORT, 'Invalid category. Must be one of: ARCHITECTURE, QUALITY, INTEGRATION, PERFORMANCE, ALGORITHM, FACADE, SECURITY, BUG, GENUINE, TESTING, DOCUMENTATION, INCOMPLETE')
  END;
END;

CREATE TRIGGER IF NOT EXISTS enforce_severity_enum_update
BEFORE UPDATE OF severity ON findings
BEGIN
  SELECT CASE
    WHEN NEW.severity NOT IN ('CRITICAL','HIGH','MEDIUM','INFO')
    THEN RAISE(ABORT, 'Invalid severity. Must be one of: CRITICAL, HIGH, MEDIUM, INFO')
  END;
END;
```

### 4.2 Inline Validation Helper (for agent `node -e` scripts)

> **Why not Zod?** Agents write inline `node -e "..."` scripts — they never `require()` modules.
> A Zod schema would never actually execute. Instead, we provide a zero-dependency validation
> snippet that agents can paste directly into their inline scripts. The DB triggers (4.1) are
> the real enforcement layer; this helper provides fail-fast feedback before the INSERT hits SQLite.

```javascript
// scripts/validate-enums.js — copy-paste into node -e scripts, or require() if using a script file
// Zero dependencies. Works in any node -e context.

const VALID = {
  relationship: new Set(['IMPORTS','USES','EXPORTS','DECLARES','SIBLINGS','COMPETES','WRAPS','FEEDS','TESTS','BROKEN']),
  category: new Set(['ARCHITECTURE','QUALITY','INTEGRATION','PERFORMANCE','ALGORITHM','FACADE','SECURITY','BUG','GENUINE','TESTING','DOCUMENTATION','INCOMPLETE']),
  severity: new Set(['CRITICAL','HIGH','MEDIUM','INFO']),
};

function assertEnum(field, value) {
  if (!VALID[field].has(value)) {
    throw new Error(`Invalid ${field}: "${value}". Must be one of: ${[...VALID[field]].join(', ')}`);
  }
}

// Usage in agent inline scripts:
// assertEnum('severity', 'CRITICAL');    // passes
// assertEnum('category', 'facade');      // throws: must be FACADE
// assertEnum('relationship', 'imports'); // throws: must be IMPORTS

module.exports = { VALID, assertEnum };
```

Agents should include this snippet at the top of their `node -e` blocks. The `assertEnum` call
goes before the `db.prepare(...).run(...)` call, providing an immediate clear error message
instead of the generic SQLite trigger abort.

### 4.3 Agent Template Updates

Add to **all 8 agent templates** that insert findings or dependencies:
- `agents/reader.md`
- `agents/module-reader.md`
- `agents/scanner.md`
- `agents/synthesizer.md`
- `agents/mapper.md`
- `agents/facade-detector.md`
- `agents/cross-repo-tracer.md`
- `agents/realness-scorer.md`

```
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
```

---

## Effort Estimate

| Phase | Work | Time |
|-------|------|------|
| 0. Snapshot | Backup DB, export before-state | **Done** |
| 1. Design enums + mappings | This document + view audit | **Done** |
| 1b. Gap analysis | Collision detection, mapping conflicts, decision log | **Done** |
| 2. Write migration script | Node.js script with --dry-run and --execute (includes merge + triggers) | ~2 hours |
| 3. Execute + verify | Dry-run → review → execute → all checks + spot-checks | ~1 hour |
| 4. Agent templates + validator | Inline validator helper + constraint blocks in all 8 agent templates | ~30 min |
| **Total** | | **~3.5 hours remaining** |

## Consequences

**Positive**:
- Aggregate queries now work: `WHERE category = 'FACADE'` returns all facades
- Subsystem graph (ADR-v4-007) can aggregate by relationship type
- Future vector embeddings built on clean categorical data
- New findings/dependencies constrained by triggers — no more drift
- All original data preserved in `_original_*` columns

**Negative**:
- ~900 tiny categories force-merged into 12 buckets — some precision lost
- Sentence-form relationships lose their inline-evidence convenience
- Agent templates must be updated (breaking change for existing prompts)
- `_original_*` columns add ~2-3% DB size overhead
- 22 dependency rows merged (1,704 → 1,682) — original data preserved in evidence + backup

**Scope exclusion**:
- `package_dependencies.relationship` (8 rows, 5 types: `optional`, `bundles-copy`,
  `npm-dependency`, `bridges-to`, `patches`) is **not** normalized. These are a separate
  domain (inter-package relationships, not inter-file) and already use a clean enum.

**Risks**:
- Pattern-matching for the long tail (~900 categories) may misclassify some
  - **Mitigation**: spot-check protocol (3.6) catches these
  - **Mitigation**: `_original_category` preserved for post-hoc correction
- Existing synthesis docs reference old category names
  - **Mitigation**: synthesis docs use prose, not exact DB queries
