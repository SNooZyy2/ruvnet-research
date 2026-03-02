#!/usr/bin/env node
'use strict';
// scripts/migrate-enums.js — ADR-v4-009 Schema Normalization Migration
// Usage: node scripts/migrate-enums.js --dry-run   (validates without persisting)
//        node scripts/migrate-enums.js --execute    (persists changes)

const Database = require('better-sqlite3');
const path = require('path');
const fs = require('fs');

const DB_PATH = path.join(__dirname, '..', 'db', 'research.db');
const BACKUP_PATH = DB_PATH + '.pre-migration';
const mode = process.argv[2];
if (mode !== '--dry-run' && mode !== '--execute') {
  console.error('Usage: node scripts/migrate-enums.js [--dry-run | --execute]');
  process.exit(1);
}
const dryRun = mode === '--dry-run';

// ════════════════════════════════════════════════════════════════════════════
// RELATIONSHIP ABSORB MAP (511 → 10 canonical types)
// ════════════════════════════════════════════════════════════════════════════
const REL_ABSORBS = {
  IMPORTS: [
    'imports', 'IMPORTS', 'import', 'imports_from', 'imports_type', 'imports_types',
    'imports_class', 'direct-import', 'type-imports', 'type_import', 'imports_trait',
    'imports_config', 'imports_error_types', 'imports_persistence', 'imports_solver',
    'imports_simd_ops', 'imports_wasm_bridge', 'imports_wasm_loader', 'imports_csr_storage',
    'imports_gnn_service', 'imports_precision_type', 'imports_spawns',
    'imports_error_type', 'lazy_import', 'optional_import', 'mod_use_import',
    'pub_mod_import', 'transitive_import', 'dynamic-import', 'test_imports',
    'imports-types', 'imports-and-calls', 'imports-facade', 'imports-unused',
    'imports-extensively', 'imports-for-initialization', 'imports-uses',
    'imports-domain-model', 'import-Pattern-StorageStats', 'imported-by',
    'imported-but-unused', 'imported_by', 'IMPORTED_BY', 'IMPORTS_AND_REEXPORTS',
    'IMPORTS_TYPE', 'MODULE_IMPORT',
  ],
  USES: [
    'uses', 'USES', 'used_by', 'USED_BY', 'used-by', 'calls', 'calls_method',
    'implements', 'IMPLEMENTS', 'IMPLEMENTS_FOR', 'extends', 'EXTENDS', 'instantiates',
    'requires', 'depends_on', 'DEPENDS_ON', 'depends-on', 'depends', 'DEPENDS_ON_TYPES',
    'consumed_by', 'CONSUMED_BY', 'consumes', 'uses_types', 'uses_type', 'uses_struct',
    'uses_api', 'uses_constant', 'uses_wrapper', 'uses_prompt_templates', 'uses-types',
    'uses-impl', 'uses-concept', 'uses-for-embeddings', 'type_consumer',
  ],
  EXPORTS: [
    'exports', 'EXPORTS', 're-exports', 're-export', 'BARREL_EXPORT', 'pub_use',
    'pub_mod_reexport', 'module-reexport', 'INDIRECT_EXPORT', 'CONDITIONAL_EXPORT',
    'exported_by', 'EXPORTED_FROM', 'EXPORTS_AS_NAMESPACE', 'EXPORTS_TO', 'exports_from',
    'exports_to', 'exports-to', 'exported-by', 'reexported_by', 'pub_re_export',
    'pub_reexport_indirect', 'RE_EXPORTS_MODULE', 'RE_EXPORTS', 'RE_EXPORTS_ALL',
    'RE_EXPORTS_TYPE', 'module-export', 'module_export', 'MODULE_EXPORT', 'module:exported',
  ],
  DECLARES: [
    'pub_module_declaration', 'pub_mod', 'pub mod', 'declares_module', 'declares',
    'mod-declaration', 'declares-submodule', 'module-declares', 'module-definition',
    'module-parent', 'module_declaration', 'defines_module', 'cfg_module', 'submodule',
    'always-on submodule', 'feature-gated submodule', 'DEFINES', 'declared-in',
    'declared_in', 'member-of', 'member_of', 'part-of', 'mod-wasm',
  ],
  SIBLINGS: [
    'module-sibling', 'sibling', 'sibling-module', 'SIBLING_MODULE', 'SIBLING',
    'SIBLING_SCENARIO', 'co-module', 'peer', 'PEER', 'sibling-kernel', 'sibling_impl',
    'sibling_module', 'sibling-cli-mcp', 'cohomology-sibling', 'same-package',
    'co-located', 'related', 'RELATED_IMPLEMENTATION',
  ],
  COMPETES: [
    'parallel-implementation', 'parallel_implementation', 'PARALLEL_IMPLEMENTATION',
    'parallel-impl', 'parallel_impl', 'parallel-system', 'parallel_system', 'parallel_to',
    'parallel', 'parallel_api', 'parallel_simulation_pattern', 'parallel implementation',
    'reimplements', 'REIMPLEMENTS', 'alternative-implementation', 'alternative_to',
    'ALTERNATIVE_LOADER', 'mirrors', 'mirrors_architecture', 'signature_mirror',
    'duplicate_of', 'duplicates_logic', 'COPY_OF_REFACTORED', 'algorithmic-parallel',
    'conceptual-parallel', 'NAME_COLLISION', 'naming-conflict', 'ARCHITECTURAL_CONTRAST',
    'CONTRASTS_WITH', 'contrasts_with', 'compared-to', 'comparison',
  ],
  WRAPS: [
    'wraps', 'WRAPS', 'wraps_native', 'wraps_deprecated', 'wrapped_by', 'delegates_to',
    'delegates-to', 'delegates', 'delegates-storage', 'delegates_to_via_npx', 'facade-of',
    'FACADE_OF',
  ],
  FEEDS: [
    'feeds', 'GENERATES', 'produces', 'produces_for', 'RECEIVES_DATA_FROM', 'reads',
    'reads-writes', 'queries', 'configures', 'registers_tools_from', 'orchestrates',
    'ORCHESTRATED_BY', 'spawns-process', 'spawns_mcp_subprocess', 'loads',
    'loads-config-from', 'loads_wasm_module', 'loads compiled WASM', 'LOADS_WASM_INTERFACE',
    'embeds_schema',
  ],
  TESTS: [
    'tests', 'TESTS', 'TESTED_BY', 'tested_by', 'tested-by integration tests',
    'TESTS_VARIANT_OF', 'TEST_SUITE', 'tests_via_spawn', 'tests_performance', 'benchmarks',
    'related-benchmark', 'VALIDATES_AGAINST', 'internal-validation', 'test-dependency',
  ],
  BROKEN: [
    'bypasses', 'MISSING_INTEGRATION', 'missing_integration', 'SHOULD_INTEGRATE',
    'should_import', 'should-import-but-reimplements', 'should_call_but_doesnt',
    'should_integrate_but_orphaned', 'expects_integration', 'potential-integration',
    'ISOLATED_FROM', 'CONTRADICTS', 'VIOLATES', 'HONEST_ALTERNATIVE_TO',
    'declares_dependency_but_never_uses', 'calls_broken', 'imports_broken',
    'supports_but_facades', 'orphaned-module', 'orphaned_from', 'incomplete_port',
    'js_fallback_for', 'replaced-by', 'replaces', 'supersedes', 'patched_version_of',
    'improved_version_of',
  ],
};

// Build reverse lookup
const REL_MAP = new Map();
for (const [canonical, variants] of Object.entries(REL_ABSORBS)) {
  for (const v of variants) REL_MAP.set(v, canonical);
}

// Keyword priority chain for unmapped relationships (ADR section 1.1)
const REL_KEYWORDS = [
  [/should|missing|broken|bypass|violat|orphan|isolated/i, 'BROKEN'],
  [/parallel|alternative|reimpl|mirror|duplicate|compet/i, 'COMPETES'],
  [/wraps|delegat|facade/i, 'WRAPS'],
  [/produces|feeds|loads|consumes|spawns|generates|orchestrat/i, 'FEEDS'],
  [/test|benchmark|validat/i, 'TESTS'],
  [/export|re-export|pub use/i, 'EXPORTS'],
  [/pub mod|declares|module/i, 'DECLARES'],
  [/imports|import/i, 'IMPORTS'],
];

function canonicalRelationship(rel) {
  if (!rel) return 'USES';
  if (REL_ABSORBS[rel]) return rel; // Already canonical
  if (REL_MAP.has(rel)) return REL_MAP.get(rel);
  for (const [regex, canonical] of REL_KEYWORDS) {
    if (regex.test(rel)) return canonical;
  }
  return 'USES';
}

// ════════════════════════════════════════════════════════════════════════════
// CATEGORY ABSORB MAP (1,211 → 12 canonical types)
// ════════════════════════════════════════════════════════════════════════════
const CAT_ABSORBS = {
  ARCHITECTURE: [
    'architecture', 'ARCHITECTURE', 'Architecture', 'design', 'DESIGN', 'Design',
    'api-design', 'API_DESIGN', 'api_design', 'interface-design', 'INTERFACE_DESIGN',
    'interface_design', 'data-model', 'DATA_MODEL', 'configuration', 'CONFIGURATION',
    'Configuration', 'config', 'CONFIG', 'infrastructure', 'INFRASTRUCTURE', 'protocol',
    'PROTOCOL', 'protocols', 'PROTOCOLS', 'data-structures', 'DATA_STRUCTURES',
    'data-structure', 'DATA_STRUCTURE', 'data_structure', 'API_SURFACE', 'api-surface',
    'INTERFACE', 'interface',
  ],
  QUALITY: [
    'quality', 'QUALITY', 'Quality', 'code-quality', 'CODE_QUALITY', 'Code Quality',
    'code_quality', 'implementation', 'IMPLEMENTATION', 'Implementation',
    'implementation-quality', 'naming', 'NAMING', 'SIMPLIFICATION', 'simplification',
    'best-practice', 'ANTI_PATTERN', 'anti-pattern', 'DESIGN_PATTERN', 'design-pattern',
    'ASSESSMENT', 'types', 'TYPES', 'HELPER', 'FEATURE', 'feature', 'Feature',
    'PORTABILITY', 'portability', 'FUNCTIONALITY', 'functionality', 'api', 'API',
  ],
  INTEGRATION: [
    'integration', 'INTEGRATION', 'Integration', 'dependency', 'DEPENDENCY',
    'dependencies', 'DEPENDENCIES', 'missing-integration', 'MISSING_INTEGRATION',
    'Missing Integration', 'disconnected', 'DISCONNECTED', 'cross-reference',
    'CROSS_REFERENCE', 'data-flow', 'DATA_FLOW', 'api-mismatch', 'API_MISMATCH',
    'missing-feature', 'MISSING_FEATURE', 'broken-dep', 'BROKEN_DEP', 'broken', 'BROKEN',
    'ORPHANED', 'orphaned', 'Orphaned',
  ],
  PERFORMANCE: [
    'performance', 'PERFORMANCE', 'Performance', 'optimization', 'OPTIMIZATION',
    'concurrency', 'CONCURRENCY', 'memory', 'MEMORY', 'Memory', 'SCALABILITY',
  ],
  ALGORITHM: [
    'algorithm', 'ALGORITHM', 'Algorithm', 'algorithm-quality', 'ALGORITHM_QUALITY',
    'algorithm-correctness', 'ALGORITHM_CORRECTNESS', 'numerical-stability',
    'NUMERICAL_STABILITY', 'logic', 'LOGIC', 'LEGITIMATE_ALGORITHM', 'algorithm-real',
  ],
  FACADE: [
    'facade', 'FACADE', 'Facade', 'stub', 'STUB', 'Stub', 'placeholder', 'PLACEHOLDER',
    'fabrication', 'FABRICATION', 'fabricated', 'FABRICATED', 'deception', 'DECEPTION',
    'misleading', 'MISLEADING', 'theatrical', 'THEATRICAL', 'PHANTOM_WRAPPER',
    'FACADE_PATTERN', 'facade-pattern', 'facade_pattern', 'FACADE_DECEPTION',
    'facade-deception', 'QUANTUM_FACADE', 'ARCHITECTURE_FACADE', 'FALSE_COMPLEXITY',
    'fabricated-data', 'FABRICATED_DATA', 'fabricated-metrics', 'FABRICATED_METRICS',
    'data-generation', 'Demonstration', 'demonstration', 'DEMONSTRATION',
  ],
  SECURITY: ['security', 'SECURITY'],
  BUG: [
    'bug', 'BUG', 'correctness', 'CORRECTNESS', 'error-handling', 'ERROR_HANDLING',
    'Error Handling', 'Error_Handling', 'error_handling', 'ACCURACY', 'accuracy',
    'FALLBACK', 'fallback', 'ROBUSTNESS', 'robustness',
  ],
  GENUINE: [
    'genuine', 'GENUINE', 'Genuine', 'GENUINE_IMPLEMENTATION', 'genuine-implementation',
    'GENUINE_SIGNAL', 'GENUINE_INFRASTRUCTURE', 'genuine-infrastructure',
    'GENUINE_ALGORITHM', 'GENUINE_ALGORITHMS', 'genuine-algorithms', 'REAL', 'real',
    'REAL_CODE', 'REAL_IMPLEMENTATION', 'real-implementation', 'REAL_WASM', 'real-wasm',
    'INNOVATION', 'realness', 'positive', 'POSITIVE', 'CORRECT_IMPLEMENTATION',
    'correct-implementation',
  ],
  TESTING: [
    'testing', 'TESTING', 'Testing', 'test-coverage', 'TEST_COVERAGE', 'test_coverage',
    'test-quality', 'TEST_QUALITY', 'comparison', 'COMPARISON', 'benchmark-quality',
    'ANALYSIS_QUALITY', 'analysis-quality', 'validation', 'VALIDATION',
  ],
  DOCUMENTATION: [
    'documentation', 'DOCUMENTATION', 'Documentation', 'completeness', 'COMPLETENESS',
    'observability', 'OBSERVABILITY', 'Observability', 'metrics', 'METRICS', 'Metrics',
    'reporting', 'REPORTING', 'ROADMAP',
  ],
  INCOMPLETE: [
    'incomplete', 'INCOMPLETE', 'Incomplete', 'incomplete-implementation',
    'INCOMPLETE_IMPLEMENTATION', 'INCOMPLETE_IMPL', 'missing-implementation',
    'MISSING_IMPLEMENTATION', 'Missing Implementation', 'gap', 'GAP', 'Gap', 'limitation',
    'LIMITATION', 'dead-code', 'DEAD_CODE', 'CONFIG_GAP', 'TYPE_SAFETY', 'type-safety',
    'RELIABILITY', 'reliability',
  ],
};

const CAT_MAP = new Map();
for (const [canonical, variants] of Object.entries(CAT_ABSORBS)) {
  for (const v of variants) CAT_MAP.set(v, canonical);
}

// Keyword priority chain for unmapped categories (ADR section 1.2)
const CAT_KEYWORDS = [
  [/facade|stub|placeholder|fabricat|decep|mislead|theatrical/i, 'FACADE'],
  [/genuine|real|innovat|positive|correct-impl/i, 'GENUINE'],
  [/integrat|depend|disconnect|orphan|broken|missing/i, 'INTEGRATION'],
  [/algorithm|numeric|math|correctness/i, 'ALGORITHM'],
  [/test|benchmark|coverage|validation/i, 'TESTING'],
  [/secur|vulnerab|crypto/i, 'SECURITY'],
  [/perform|optim|memory|concurr|latency/i, 'PERFORMANCE'],
  [/doc|comment|readme|completeness/i, 'DOCUMENTATION'],
  [/incomplet|gap|dead|limit|missing/i, 'INCOMPLETE'],
  [/bug|error|broken|fault|crash/i, 'BUG'],
  [/architect|design|pattern|config|struct/i, 'ARCHITECTURE'],
];

function canonicalCategory(cat) {
  if (!cat) return 'QUALITY';
  if (CAT_ABSORBS[cat]) return cat; // Already canonical
  if (CAT_MAP.has(cat)) return CAT_MAP.get(cat);
  for (const [regex, canonical] of CAT_KEYWORDS) {
    if (regex.test(cat)) return canonical;
  }
  return 'QUALITY';
}

// ════════════════════════════════════════════════════════════════════════════
// SEVERITY (7 → 4)
// ════════════════════════════════════════════════════════════════════════════
const VALID_SEVERITIES = new Set(['CRITICAL', 'HIGH', 'MEDIUM', 'INFO']);

// ════════════════════════════════════════════════════════════════════════════
// MIGRATION
// ════════════════════════════════════════════════════════════════════════════
function main() {
  console.log(`\n${'═'.repeat(70)}`);
  console.log(`  ADR-v4-009 Schema Normalization Migration`);
  console.log(`  Mode: ${dryRun ? 'DRY RUN (no changes persisted)' : 'EXECUTE'}`);
  console.log(`${'═'.repeat(70)}\n`);

  // ── Pre-migration baselines ──
  const db = new Database(DB_PATH);
  db.pragma('journal_mode = WAL');
  const baselines = {
    findings: db.prepare('SELECT COUNT(*) as c FROM findings').get().c,
    dependencies: db.prepare('SELECT COUNT(*) as c FROM dependencies').get().c,
    sessions: db.prepare('SELECT COUNT(*) as c FROM sessions').get().c,
    files: db.prepare('SELECT COUNT(*) as c FROM files').get().c,
    file_reads: db.prepare('SELECT COUNT(*) as c FROM file_reads').get().c,
    critical: db.prepare("SELECT COUNT(*) as c FROM findings WHERE severity='CRITICAL'").get().c,
    smart_gaps: db.prepare('SELECT COUNT(*) as c FROM smart_priority_gaps').get().c,
    domain_cov: db.prepare('SELECT COUNT(*) as c FROM domain_coverage').get().c,
    package_cov: db.prepare('SELECT COUNT(*) as c FROM package_coverage').get().c,
    open_findings: db.prepare('SELECT COUNT(*) as c FROM open_findings').get().c,
    hotspots: db.prepare('SELECT COUNT(*) as c FROM integration_hotspots').get().c,
  };
  console.log('Pre-migration baselines captured.');

  // ── Backup ──
  if (!dryRun) {
    console.log(`Creating backup: ${BACKUP_PATH}`);
    fs.copyFileSync(DB_PATH, BACKUP_PATH);
  }

  // ── Begin transaction ──
  db.exec('BEGIN');
  let allPassed = true;
  try {
    // Step 0: Add _original_* columns
    const depCols = db.pragma('table_info(dependencies)').map(c => c.name);
    if (!depCols.includes('_original_relationship')) {
      db.exec('ALTER TABLE dependencies ADD COLUMN _original_relationship TEXT');
    }
    const findCols = db.pragma('table_info(findings)').map(c => c.name);
    if (!findCols.includes('_original_category')) {
      db.exec('ALTER TABLE findings ADD COLUMN _original_category TEXT');
    }
    if (!findCols.includes('_original_severity')) {
      db.exec('ALTER TABLE findings ADD COLUMN _original_severity TEXT');
    }
    console.log('Step 0: _original_* columns added.');

    // Step 1: Save originals
    db.exec('UPDATE dependencies SET _original_relationship = relationship');
    db.exec("UPDATE findings SET _original_category = category");
    db.exec("UPDATE findings SET _original_severity = severity WHERE severity NOT IN ('CRITICAL','HIGH','MEDIUM','INFO')");
    console.log('Step 1: Originals saved.');

    // Step 2: Dedup collision pairs
    const mergedCount = dedupDependencies(db);
    console.log(`Step 2: Merged ${mergedCount} duplicate dependency rows.`);

    // Step 3: Normalize severities
    const sevChanged = normalizeSeverities(db);
    console.log(`Step 3: Severity — ${sevChanged} rows normalized.`);

    // Step 4: Normalize categories
    const catChanged = normalizeCategories(db);
    console.log(`Step 4: Categories — ${catChanged} rows normalized.`);

    // Step 5: Normalize relationships (with evidence preservation)
    const relChanged = normalizeRelationships(db);
    console.log(`Step 5: Relationships — ${relChanged} rows normalized.`);

    // Step 6: Create enforcement triggers
    createTriggers(db);
    console.log('Step 6: Enforcement triggers created.');

    // Step 7: Verify
    allPassed = verify(db, baselines, mergedCount);

    // Commit or rollback
    if (dryRun) {
      db.exec('ROLLBACK');
      console.log(`\n${'─'.repeat(70)}`);
      console.log('  DRY RUN COMPLETE — all changes rolled back.');
      console.log(`${'─'.repeat(70)}`);
    } else if (allPassed) {
      db.exec('COMMIT');
      console.log(`\n${'═'.repeat(70)}`);
      console.log('  MIGRATION COMMITTED SUCCESSFULLY');
      console.log(`${'═'.repeat(70)}`);
    } else {
      db.exec('ROLLBACK');
      console.error('\nVerification FAILED — all changes rolled back.');
      process.exit(1);
    }
  } catch (e) {
    try { db.exec('ROLLBACK'); } catch (_) {}
    console.error('\nMigration ERROR — rolled back:', e.message);
    console.error(e.stack);
    process.exit(1);
  }
  db.close();
  process.exit(allPassed ? 0 : 1);
}

// ════════════════════════════════════════════════════════════════════════════
// STEP 2: DEDUP COLLISION PAIRS
// ════════════════════════════════════════════════════════════════════════════
function dedupDependencies(db) {
  const allDeps = db.prepare(
    'SELECT id, source_file_id, target_file_id, relationship, evidence FROM dependencies'
  ).all();

  // Compute canonical for each row and group by (source, target, canonical)
  const groups = new Map();
  for (const d of allDeps) {
    const canonical = canonicalRelationship(d.relationship);
    const key = `${d.source_file_id}:${d.target_file_id}:${canonical}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push({ ...d, canonical });
  }

  // Find collision groups (2+ rows mapping to same canonical for same file pair)
  const collisions = [...groups.values()].filter(g => g.length > 1);
  console.log(`  Found ${collisions.length} collision groups.`);

  const deleteStmt = db.prepare('DELETE FROM dependencies WHERE id = ?');
  const updateEvStmt = db.prepare('UPDATE dependencies SET evidence = ? WHERE id = ?');
  let mergedCount = 0;

  for (const group of collisions) {
    // Keep row with longest evidence
    group.sort((a, b) => (b.evidence?.length || 0) - (a.evidence?.length || 0));
    const survivor = group[0];
    let newEvidence = survivor.evidence || '';
    for (let i = 1; i < group.length; i++) {
      const loser = group[i];
      newEvidence += ` [merged: ${loser.relationship}`;
      if (loser.evidence) newEvidence += ` | ${loser.evidence}`;
      newEvidence += ']';
      deleteStmt.run(loser.id);
      mergedCount++;
    }
    if (newEvidence !== (survivor.evidence || '')) {
      updateEvStmt.run(newEvidence, survivor.id);
    }
  }
  return mergedCount;
}

// ════════════════════════════════════════════════════════════════════════════
// STEP 3: NORMALIZE SEVERITIES
// ════════════════════════════════════════════════════════════════════════════
function normalizeSeverities(db) {
  const result = db.prepare(
    "UPDATE findings SET severity = 'INFO' WHERE severity NOT IN ('CRITICAL','HIGH','MEDIUM','INFO')"
  ).run();
  return result.changes;
}

// ════════════════════════════════════════════════════════════════════════════
// STEP 4: NORMALIZE CATEGORIES
// ════════════════════════════════════════════════════════════════════════════
function normalizeCategories(db) {
  // Handle NULLs first
  db.prepare("UPDATE findings SET category = 'QUALITY' WHERE category IS NULL").run();

  // Get all distinct categories and compute canonical
  const distinctCats = db.prepare('SELECT DISTINCT category FROM findings').all();
  const mapping = new Map(); // old → canonical
  for (const { category } of distinctCats) {
    mapping.set(category, canonicalCategory(category));
  }

  // Group by canonical target
  const groups = new Map(); // canonical → [old1, old2, ...]
  for (const [old, canonical] of mapping) {
    if (!groups.has(canonical)) groups.set(canonical, []);
    groups.get(canonical).push(old);
  }

  // Bulk update per canonical
  let totalChanged = 0;
  for (const [canonical, oldCats] of groups) {
    // Batch in groups of 900 to stay under SQLite parameter limit
    for (let i = 0; i < oldCats.length; i += 900) {
      const batch = oldCats.slice(i, i + 900);
      const placeholders = batch.map(() => '?').join(',');
      const result = db.prepare(
        `UPDATE findings SET category = ? WHERE category IN (${placeholders})`
      ).run(canonical, ...batch);
      totalChanged += result.changes;
    }
  }
  return totalChanged;
}

// ════════════════════════════════════════════════════════════════════════════
// STEP 5: NORMALIZE RELATIONSHIPS
// ════════════════════════════════════════════════════════════════════════════
function normalizeRelationships(db) {
  const distinctRels = db.prepare('SELECT DISTINCT relationship FROM dependencies').all();
  const mapping = new Map();
  for (const { relationship } of distinctRels) {
    mapping.set(relationship, canonicalRelationship(relationship));
  }

  // Separate short-form and sentence-form originals
  // Sentence-form (>30 chars): preserve original in evidence column
  const shortGroups = new Map(); // canonical → [old1, old2, ...]
  const longEntries = [];        // [{old, canonical}]

  for (const [old, canonical] of mapping) {
    if (old === canonical) continue; // Already canonical — skip
    if (old.length > 30) {
      longEntries.push({ old, canonical });
    } else {
      if (!shortGroups.has(canonical)) shortGroups.set(canonical, []);
      shortGroups.get(canonical).push(old);
    }
  }

  let totalChanged = 0;

  // Bulk update short-form
  for (const [canonical, oldRels] of shortGroups) {
    for (let i = 0; i < oldRels.length; i += 900) {
      const batch = oldRels.slice(i, i + 900);
      const placeholders = batch.map(() => '?').join(',');
      const result = db.prepare(
        `UPDATE dependencies SET relationship = ? WHERE relationship IN (${placeholders})`
      ).run(canonical, ...batch);
      totalChanged += result.changes;
    }
  }

  // Per-value update for sentence-form (append to evidence)
  const longStmt = db.prepare(
    `UPDATE dependencies
     SET relationship = ?,
         evidence = COALESCE(evidence, '') || ?
     WHERE relationship = ?`
  );
  for (const { old, canonical } of longEntries) {
    const result = longStmt.run(canonical, ` [migrated from relationship: ${old}]`, old);
    totalChanged += result.changes;
  }

  return totalChanged;
}

// ════════════════════════════════════════════════════════════════════════════
// STEP 6: ENFORCEMENT TRIGGERS
// ════════════════════════════════════════════════════════════════════════════
function createTriggers(db) {
  db.exec(`
    CREATE TRIGGER IF NOT EXISTS enforce_relationship_enum_insert
    BEFORE INSERT ON dependencies
    BEGIN
      SELECT CASE
        WHEN NEW.relationship NOT IN ('IMPORTS','USES','EXPORTS','DECLARES','SIBLINGS',
             'COMPETES','WRAPS','FEEDS','TESTS','BROKEN')
        THEN RAISE(ABORT, 'Invalid relationship type. Must be one of: IMPORTS, USES, EXPORTS, DECLARES, SIBLINGS, COMPETES, WRAPS, FEEDS, TESTS, BROKEN')
      END;
    END;

    CREATE TRIGGER IF NOT EXISTS enforce_relationship_enum_update
    BEFORE UPDATE OF relationship ON dependencies
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

    CREATE TRIGGER IF NOT EXISTS enforce_category_enum_update
    BEFORE UPDATE OF category ON findings
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

    CREATE TRIGGER IF NOT EXISTS enforce_severity_enum_update
    BEFORE UPDATE OF severity ON findings
    BEGIN
      SELECT CASE
        WHEN NEW.severity NOT IN ('CRITICAL','HIGH','MEDIUM','INFO')
        THEN RAISE(ABORT, 'Invalid severity. Must be one of: CRITICAL, HIGH, MEDIUM, INFO')
      END;
    END;
  `);
}

// ════════════════════════════════════════════════════════════════════════════
// STEP 7: VERIFICATION
// ════════════════════════════════════════════════════════════════════════════
function verify(db, baselines, mergedCount) {
  console.log(`\n${'─'.repeat(70)}`);
  console.log('  VERIFICATION');
  console.log(`${'─'.repeat(70)}`);
  let allPassed = true;
  const expectedDeps = baselines.dependencies - mergedCount;

  function check(name, actual, expected) {
    const pass = actual === expected;
    const icon = pass ? 'PASS' : 'FAIL';
    console.log(`  [${icon}] ${name}: ${actual} (expected ${expected})`);
    if (!pass) allPassed = false;
    return pass;
  }

  function checkRange(name, actual, lo, hi) {
    const pass = actual >= lo && actual <= hi;
    const icon = pass ? 'PASS' : 'FAIL';
    console.log(`  [${icon}] ${name}: ${actual} (expected ${lo}-${hi})`);
    if (!pass) allPassed = false;
    return pass;
  }

  // ── 3.1 Invariant checks ──
  console.log('\n  --- Invariant Checks ---');
  const cnt = (sql) => db.prepare(sql).get().c;
  check('Total findings unchanged', cnt('SELECT COUNT(*) as c FROM findings'), baselines.findings);
  check('Total dependencies (post-merge)', cnt('SELECT COUNT(*) as c FROM dependencies'), expectedDeps);
  check('Total sessions unchanged', cnt('SELECT COUNT(*) as c FROM sessions'), baselines.sessions);
  check('Total files unchanged', cnt('SELECT COUNT(*) as c FROM files'), baselines.files);
  check('Total file_reads unchanged', cnt('SELECT COUNT(*) as c FROM file_reads'), baselines.file_reads);
  check('No NULL relationships', cnt('SELECT COUNT(*) as c FROM dependencies WHERE relationship IS NULL'), 0);
  check('No NULL categories', cnt('SELECT COUNT(*) as c FROM findings WHERE category IS NULL'), 0);
  check('No NULL severities', cnt('SELECT COUNT(*) as c FROM findings WHERE severity IS NULL'), 0);
  check('Dep originals preserved', cnt('SELECT COUNT(*) as c FROM dependencies WHERE _original_relationship IS NULL'), 0);
  check('Cat originals preserved', cnt('SELECT COUNT(*) as c FROM findings WHERE _original_category IS NULL'), 0);
  check('Stray sev originals preserved',
    cnt("SELECT COUNT(*) as c FROM findings WHERE _original_severity IN ('LOW','POSITIVE','LOW_CONFIDENCE')"), 24);

  // ── 3.2 Cardinality checks ──
  console.log('\n  --- Cardinality Checks ---');
  check('Distinct relationships', cnt('SELECT COUNT(DISTINCT relationship) as c FROM dependencies'), 10);
  check('Distinct categories', cnt('SELECT COUNT(DISTINCT category) as c FROM findings'), 12);
  check('Distinct severities', cnt('SELECT COUNT(DISTINCT severity) as c FROM findings'), 4);

  // Verify exact canonical values
  const actualRels = db.prepare('SELECT DISTINCT relationship FROM dependencies ORDER BY relationship').all().map(r => r.relationship);
  const expectedRels = ['BROKEN','COMPETES','DECLARES','EXPORTS','FEEDS','IMPORTS','SIBLINGS','TESTS','USES','WRAPS'];
  check('All relationships canonical', JSON.stringify(actualRels), JSON.stringify(expectedRels));

  const actualCats = db.prepare('SELECT DISTINCT category FROM findings ORDER BY category').all().map(r => r.category);
  const expectedCats = ['ALGORITHM','ARCHITECTURE','BUG','DOCUMENTATION','FACADE','GENUINE','INCOMPLETE','INTEGRATION','PERFORMANCE','QUALITY','SECURITY','TESTING'];
  check('All categories canonical', JSON.stringify(actualCats), JSON.stringify(expectedCats));

  const actualSevs = db.prepare('SELECT DISTINCT severity FROM findings ORDER BY severity').all().map(r => r.severity);
  const expectedSevs = ['CRITICAL','HIGH','INFO','MEDIUM'];
  check('All severities canonical', JSON.stringify(actualSevs), JSON.stringify(expectedSevs));

  // ── 3.3 Distribution sanity checks ──
  console.log('\n  --- Distribution Checks ---');
  const catDist = db.prepare('SELECT category, COUNT(*) as c FROM findings GROUP BY category ORDER BY c DESC').all();
  console.log('  Category distribution:');
  for (const row of catDist) console.log(`    ${row.category.padEnd(15)} ${row.c}`);
  check('Largest category < 35%', catDist[0].c < baselines.findings * 0.35 ? 1 : 0, 1);
  check('Smallest category > 0', catDist[catDist.length - 1].c > 0 ? 1 : 0, 1);

  const facadeCount = catDist.find(r => r.category === 'FACADE')?.c || 0;
  checkRange('FACADE count', facadeCount, 400, 1200);
  const genuineCount = catDist.find(r => r.category === 'GENUINE')?.c || 0;
  checkRange('GENUINE count', genuineCount, 200, 800);

  const relDist = db.prepare('SELECT relationship, COUNT(*) as c FROM dependencies GROUP BY relationship ORDER BY c DESC').all();
  console.log('\n  Relationship distribution:');
  for (const row of relDist) console.log(`    ${row.relationship.padEnd(10)} ${row.c}`);
  const importsCount = relDist.find(r => r.relationship === 'IMPORTS')?.c || 0;
  checkRange('IMPORTS count', importsCount, 400, 800);

  const singletons = db.prepare('SELECT COUNT(*) as c FROM (SELECT relationship, COUNT(*) cc FROM dependencies GROUP BY relationship HAVING cc = 1)').get().c;
  check('No singleton relationships', singletons, 0);

  const depSum = relDist.reduce((s, r) => s + r.c, 0);
  check('Dep sum correct', depSum, expectedDeps);

  // ── 3.5 Cross-reference checks ──
  console.log('\n  --- Cross-Reference Checks ---');
  check('smart_priority_gaps works', cnt('SELECT COUNT(*) as c FROM smart_priority_gaps'), baselines.smart_gaps);
  check('domain_coverage works', cnt('SELECT COUNT(*) as c FROM domain_coverage'), baselines.domain_cov);
  check('package_coverage works', cnt('SELECT COUNT(*) as c FROM package_coverage'), baselines.package_cov);
  check('open_findings works', cnt('SELECT COUNT(*) as c FROM open_findings'), baselines.open_findings);
  check('integration_hotspots works', cnt('SELECT COUNT(*) as c FROM integration_hotspots'), baselines.hotspots);

  // ── 3.7 Regression: CRITICAL count preserved ──
  check('CRITICAL count preserved', cnt("SELECT COUNT(*) as c FROM findings WHERE severity='CRITICAL'"), baselines.critical);

  // ── 3.6 Spot-check samples ──
  console.log('\n  --- Spot-Check Samples (10 findings) ---');
  const fSamples = db.prepare(
    "SELECT category, _original_category, SUBSTR(description, 1, 120) as desc FROM findings ORDER BY RANDOM() LIMIT 10"
  ).all();
  for (const s of fSamples) {
    const changed = s.category !== s._original_category ? '*' : ' ';
    console.log(`  ${changed} [${s.category}] ← "${s._original_category}" | ${s.desc}`);
  }

  console.log('\n  --- Spot-Check Samples (10 dependencies) ---');
  const dSamples = db.prepare(
    `SELECT relationship, _original_relationship, SUBSTR(evidence, -120) as ev
     FROM dependencies WHERE _original_relationship != relationship
     ORDER BY RANDOM() LIMIT 10`
  ).all();
  for (const s of dSamples) {
    console.log(`  [${s.relationship}] ← "${s._original_relationship}" | ${s.ev || '(no evidence)'}`);
  }

  console.log(`\n  --- Overall: ${allPassed ? 'ALL CHECKS PASSED' : 'SOME CHECKS FAILED'} ---`);
  return allPassed;
}

main();
