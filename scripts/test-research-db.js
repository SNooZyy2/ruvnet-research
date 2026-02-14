#!/usr/bin/env node
/**
 * Research Ledger System — Validation Suite
 * Validates database schema, seed data, population, views, and file system
 */

const Database = require('better-sqlite3');
const fs = require('fs');
const path = require('path');

// Configuration
const DB_PATH = '/home/snoozyy/ruvnet-research/db/research.db';
const RESEARCH_ROOT = '/home/snoozyy/ruvnet-research';

// Test results
const results = {
  passed: 0,
  failed: 0,
  tests: []
};

// Helper functions
function test(num, description, fn) {
  try {
    fn();
    results.passed++;
    results.tests.push({ num, description, status: 'PASS', error: null });
    console.log(`[PASS]  ${num}. ${description}`);
  } catch (error) {
    results.failed++;
    results.tests.push({ num, description, status: 'FAIL', error: error.message });
    console.log(`[FAIL]  ${num}. ${description} — ${error.message}`);
  }
}

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

function assertEquals(actual, expected, message) {
  if (actual !== expected) {
    throw new Error(`${message} (expected: ${expected}, got: ${actual})`);
  }
}

function assertGreaterThan(actual, min, message) {
  if (actual <= min) {
    throw new Error(`${message} (expected > ${min}, got: ${actual})`);
  }
}

function assertContains(array, value, message) {
  if (!array.includes(value)) {
    throw new Error(`${message} (value: ${value} not found)`);
  }
}

// Main validation
console.log('Research Ledger System — Validation Suite');
console.log('==========================================\n');

let db = null;

// Test 1: Database file exists
test(1, 'Database file exists', () => {
  assert(fs.existsSync(DB_PATH), `Database not found at ${DB_PATH}`);
  db = new Database(DB_PATH, { readonly: true });
});

// If DB doesn't exist, skip remaining DB tests
if (!db) {
  console.log('\n⚠️  Database not found — skipping remaining DB tests\n');
} else {
  // Schema validation (tests 2-9)

  test(2, 'All 9 tables exist', () => {
    const tables = db.prepare(`
      SELECT name FROM sqlite_master
      WHERE type='table' AND name NOT LIKE 'sqlite_%'
      ORDER BY name
    `).all().map(r => r.name);

    const expected = [
      'dependencies',
      'domains',
      'file_domains',
      'file_reads',
      'files',
      'findings',
      'package_dependencies',
      'packages',
      'sessions'
    ];

    assertEquals(tables.length, 9, 'Expected 9 tables');
    expected.forEach(table => {
      assertContains(tables, table, `Missing table: ${table}`);
    });
  });

  test(3, 'All 6 views exist', () => {
    const views = db.prepare(`
      SELECT name FROM sqlite_master
      WHERE type='view'
      ORDER BY name
    `).all().map(r => r.name);

    const expected = [
      'domain_coverage',
      'integration_hotspots',
      'open_findings',
      'package_coverage',
      'priority_gaps',
      'unverified_deps'
    ];

    assertEquals(views.length, 6, 'Expected 6 views');
    expected.forEach(view => {
      assertContains(views, view, `Missing view: ${view}`);
    });
  });

  test(4, 'files table has correct columns', () => {
    const columns = db.prepare(`PRAGMA table_info(files)`).all();
    const columnNames = columns.map(c => c.name);

    const expected = [
      'id', 'package_id', 'relative_path', 'loc',
      'depth', 'lines_read', 'last_read_date', 'notes'
    ];

    expected.forEach(col => {
      assertContains(columnNames, col, `Missing column: ${col}`);
    });
  });

  test(5, 'findings table has correct columns', () => {
    const columns = db.prepare(`PRAGMA table_info(findings)`).all();
    const columnNames = columns.map(c => c.name);

    const expected = ['id', 'file_id', 'session_id',
                     'line_start', 'line_end', 'severity', 'category',
                     'description', 'followed_up'];

    expected.forEach(col => {
      assertContains(columnNames, col, `Missing column: ${col}`);
    });
  });

  test(6, 'dependencies table has correct columns', () => {
    const columns = db.prepare(`PRAGMA table_info(dependencies)`).all();
    const columnNames = columns.map(c => c.name);

    const expected = ['id', 'source_file_id', 'target_file_id',
                     'relationship', 'evidence', 'verified'];

    expected.forEach(col => {
      assertContains(columnNames, col, `Missing column: ${col}`);
    });
  });

  test(7, 'Foreign key enforcement works', () => {
    // Need a writable connection to test constraints
    const wdb = new Database(DB_PATH);
    wdb.pragma('foreign_keys = ON');

    try {
      wdb.prepare(`
        INSERT INTO files (package_id, relative_path, loc, depth)
        VALUES (99999, 'test.js', 100, 'NOT_TOUCHED')
      `).run();
      wdb.close();
      throw new Error('Foreign key constraint should have failed');
    } catch (error) {
      try { wdb.close(); } catch(_) {}
      assert(
        error.message.includes('FOREIGN KEY constraint failed'),
        'Expected FOREIGN KEY constraint error'
      );
    }
  });

  test(8, 'UNIQUE constraint on files(package_id, relative_path) works', () => {
    // Get an existing file
    const existing = db.prepare('SELECT package_id, relative_path FROM files LIMIT 1').get();

    if (existing) {
      const wdb = new Database(DB_PATH);
      wdb.pragma('foreign_keys = ON');
      try {
        wdb.prepare(`
          INSERT INTO files (package_id, relative_path, loc, depth)
          VALUES (?, ?, 100, 'NOT_TOUCHED')
        `).run(existing.package_id, existing.relative_path);
        wdb.close();
        throw new Error('UNIQUE constraint should have failed');
      } catch (error) {
        try { wdb.close(); } catch(_) {}
        assert(
          error.message.includes('UNIQUE constraint failed'),
          'Expected UNIQUE constraint error'
        );
      }
    } else {
      assert(true, 'No files to test uniqueness');
    }
  });

  test(9, 'UNIQUE constraint on dependencies works', () => {
    // Get an existing dependency
    const existing = db.prepare(`
      SELECT source_file_id, target_file_id, relationship
      FROM dependencies LIMIT 1
    `).get();

    if (existing) {
      try {
        db.prepare(`
          INSERT INTO dependencies (source_file_id, target_file_id, relationship)
          VALUES (?, ?, ?)
        `).run(existing.source_file_id, existing.target_file_id, existing.relationship);
        throw new Error('UNIQUE constraint should have failed');
      } catch (error) {
        assert(
          error.message.includes('UNIQUE constraint failed'),
          'Expected UNIQUE constraint error'
        );
      }
    } else {
      // No dependencies yet — this is expected in fresh DB
      assert(true, 'No dependencies to test uniqueness');
    }
  });

  // Seed data validation (tests 10-16)

  test(10, 'packages table has exactly 7 rows', () => {
    const count = db.prepare('SELECT COUNT(*) as cnt FROM packages').get().cnt;
    assertEquals(count, 7, 'Expected 7 packages');
  });

  test(11, 'Package names match expected', () => {
    const names = db.prepare('SELECT name FROM packages ORDER BY name').all()
      .map(r => r.name);

    const expected = [
      '@claude-flow/guidance',
      '@ruvector/core',
      'agentdb',
      'agentic-flow',
      'claude-config',
      'claude-flow-cli',
      'custom-src'
    ];

    assertEquals(names.length, 7, 'Expected 7 package names');
    names.forEach((name, i) => {
      assertEquals(name, expected[i], `Package ${i} mismatch`);
    });
  });

  test(12, 'domains table has exactly 14 rows', () => {
    const count = db.prepare('SELECT COUNT(*) as cnt FROM domains').get().cnt;
    assertEquals(count, 14, 'Expected 14 domains');
  });

  test(13, 'Domain priorities: HIGH=6, MEDIUM=5, LOW=3', () => {
    const high = db.prepare("SELECT COUNT(*) as cnt FROM domains WHERE priority='HIGH'").get().cnt;
    const medium = db.prepare("SELECT COUNT(*) as cnt FROM domains WHERE priority='MEDIUM'").get().cnt;
    const low = db.prepare("SELECT COUNT(*) as cnt FROM domains WHERE priority='LOW'").get().cnt;

    assertEquals(high, 6, 'Expected 6 HIGH priority domains');
    assertEquals(medium, 5, 'Expected 5 MEDIUM priority domains');
    assertEquals(low, 3, 'Expected 3 LOW priority domains');
  });

  test(14, 'sessions table has exactly 4 rows', () => {
    const count = db.prepare('SELECT COUNT(*) as cnt FROM sessions').get().cnt;
    assertEquals(count, 4, 'Expected 4 sessions (R1, R2, R3, R3-verify)');
  });

  test(15, 'package_dependencies table has exactly 8 rows', () => {
    const count = db.prepare('SELECT COUNT(*) as cnt FROM package_dependencies').get().cnt;
    assertEquals(count, 8, 'Expected 8 package dependencies');
  });

  test(16, 'All domain synthesis_path values follow pattern', () => {
    const paths = db.prepare('SELECT name, synthesis_path FROM domains').all();

    paths.forEach(domain => {
      const expected = `domains/${domain.name}/analysis.md`;
      assertEquals(
        domain.synthesis_path,
        expected,
        `synthesis_path mismatch for domain ${domain.name}`
      );
    });
  });

  // Population validation (tests 17-22)

  test(17, 'files table has more than 0 rows', () => {
    const count = db.prepare('SELECT COUNT(*) as cnt FROM files').get().cnt;
    assertGreaterThan(count, 0, 'Expected files to be populated');
  });

  test(18, 'At least one package has total_files > 0 and populated_at IS NOT NULL', () => {
    const populated = db.prepare(`
      SELECT COUNT(*) as cnt FROM packages
      WHERE total_files > 0 AND populated_at IS NOT NULL
    `).get().cnt;

    assertGreaterThan(populated, 0, 'Expected at least one populated package');
  });

  test(19, 'All files have depth = NOT_TOUCHED (initial state)', () => {
    const nonTouched = db.prepare(`
      SELECT COUNT(*) as cnt FROM files
      WHERE depth != 'NOT_TOUCHED'
    `).get().cnt;

    assertEquals(nonTouched, 0, 'All files should have depth=NOT_TOUCHED initially');
  });

  test(20, 'No files have NULL loc', () => {
    const nullLoc = db.prepare('SELECT COUNT(*) as cnt FROM files WHERE loc IS NULL').get().cnt;
    assertEquals(nullLoc, 0, 'No files should have NULL loc');
  });

  test(21, 'Total LOC across all files is > 0', () => {
    const totalLoc = db.prepare('SELECT SUM(loc) as total FROM files').get().total || 0;
    assertGreaterThan(totalLoc, 0, 'Expected total LOC > 0');
  });

  test(22, 'Files from claude-flow-cli package exist', () => {
    const count = db.prepare(`
      SELECT COUNT(*) as cnt FROM files f
      JOIN packages p ON f.package_id = p.id
      WHERE p.name = 'claude-flow-cli'
    `).get().cnt;

    assertGreaterThan(count, 0, 'Expected files from claude-flow-cli package');
  });

  // View validation (tests 23-28)

  test(23, 'domain_coverage view executes without error', () => {
    const rows = db.prepare('SELECT * FROM domain_coverage').all();
    assert(Array.isArray(rows), 'Expected array from view');
  });

  test(24, 'package_coverage view returns rows matching packages count', () => {
    const rows = db.prepare('SELECT * FROM package_coverage').all();
    assertEquals(rows.length, 7, 'Expected 7 rows (one per package)');
  });

  test(25, 'integration_hotspots view executes without error', () => {
    const rows = db.prepare('SELECT * FROM integration_hotspots').all();
    assert(Array.isArray(rows), 'Expected array from view');
  });

  test(26, 'priority_gaps view executes without error', () => {
    const rows = db.prepare('SELECT * FROM priority_gaps').all();
    assert(Array.isArray(rows), 'Expected array from view');
  });

  test(27, 'unverified_deps view executes without error', () => {
    const rows = db.prepare('SELECT * FROM unverified_deps').all();
    assert(Array.isArray(rows), 'Expected array from view');
  });

  test(28, 'open_findings view executes without error', () => {
    const rows = db.prepare('SELECT * FROM open_findings').all();
    assert(Array.isArray(rows), 'Expected array from view');
  });

  db.close();
}

// File system validation (tests 29-35)

test(29, 'MASTER-INDEX.md exists', () => {
  const filePath = path.join(RESEARCH_ROOT, 'MASTER-INDEX.md');
  assert(fs.existsSync(filePath), `File not found: ${filePath}`);
});

test(30, 'CLAUDE.md exists', () => {
  const filePath = path.join(RESEARCH_ROOT, 'CLAUDE.md');
  assert(fs.existsSync(filePath), `File not found: ${filePath}`);
});

test(31, '.gitignore exists', () => {
  const filePath = path.join(RESEARCH_ROOT, '.gitignore');
  assert(fs.existsSync(filePath), `File not found: ${filePath}`);
});

test(32, 'At least 6 domain directories exist', () => {
  const domainsDir = path.join(RESEARCH_ROOT, 'domains');
  if (!fs.existsSync(domainsDir)) {
    throw new Error(`Domains directory not found: ${domainsDir}`);
  }

  const dirs = fs.readdirSync(domainsDir, { withFileTypes: true })
    .filter(dirent => dirent.isDirectory())
    .map(dirent => dirent.name);

  assertGreaterThan(dirs.length, 5, `Expected at least 6 domain directories`);
});

test(33, 'At least 6 analysis.md files exist in domain directories', () => {
  const domainsDir = path.join(RESEARCH_ROOT, 'domains');
  if (!fs.existsSync(domainsDir)) {
    throw new Error(`Domains directory not found: ${domainsDir}`);
  }

  const dirs = fs.readdirSync(domainsDir, { withFileTypes: true })
    .filter(dirent => dirent.isDirectory())
    .map(dirent => dirent.name);

  const analysisCounts = dirs.filter(dir => {
    const analysisPath = path.join(domainsDir, dir, 'analysis.md');
    return fs.existsSync(analysisPath);
  }).length;

  assertGreaterThan(analysisCounts, 5, `Expected at least 6 analysis.md files`);
});

test(34, 'All 4 agent templates exist', () => {
  const agentsDir = path.join(RESEARCH_ROOT, 'agents');
  const expected = ['scanner.md', 'reader.md', 'synthesizer.md', 'mapper.md'];

  expected.forEach(agent => {
    const agentPath = path.join(agentsDir, agent);
    assert(fs.existsSync(agentPath), `Agent template not found: ${agent}`);
  });
});

test(35, 'schema.sql exists', () => {
  const schemaPath = path.join(RESEARCH_ROOT, 'db', 'schema.sql');
  assert(fs.existsSync(schemaPath), `File not found: ${schemaPath}`);
});

// Summary
console.log('\n==========================================');
console.log(`Results: ${results.passed}/${results.passed + results.failed} PASS, ${results.failed} FAIL`);
console.log('==========================================\n');

// Exit with appropriate code
process.exit(results.failed > 0 ? 1 : 0);
