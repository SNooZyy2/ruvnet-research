#!/usr/bin/env node

/**
 * populate-rust-files.js
 *
 * Indexes all files from the 4 cloned Rust repositories into the research database.
 * Recursively walks each repo, determines file metadata, and inserts into files table.
 */

const fs = require('fs');
const path = require('path');
const Database = require(path.join(process.env.HOME, 'node_modules', 'better-sqlite3'));

// Configuration
const DB_PATH = '/home/snoozyy/ruvnet-research/db/research.db';
const SKIP_DIRS = new Set(['.git', 'node_modules', 'target', '.claude', 'dist']);

// Package mappings: package_name -> repo_path
const REPO_MAPPINGS = {
  'ruvector-rust': '/home/snoozyy/repos/ruvector',
  'ruv-fann-rust': '/home/snoozyy/repos/ruv-FANN',
  'agentic-flow-rust': '/home/snoozyy/repos/agentic-flow',
  'sublinear-rust': '/home/snoozyy/repos/sublinear-time-solver'
};

// Binary file extensions
const BINARY_EXTENSIONS = new Set(['.wasm', '.node', '.png', '.jpg', '.jpeg', '.gif', '.ico', '.so', '.dylib', '.dll']);

/**
 * Determine file type from extension
 */
function getFileType(filename) {
  const ext = path.extname(filename).toLowerCase();
  const typeMap = {
    '.rs': 'rs',
    '.js': 'js',
    '.ts': 'ts',
    '.tsx': 'ts',
    '.json': 'json',
    '.md': 'md',
    '.wasm': 'wasm',
    '.node': 'node',
    '.toml': 'toml',
    '.sh': 'sh',
    '.bash': 'sh',
    '.sql': 'sql',
    '.yaml': 'yaml',
    '.yml': 'yaml'
  };
  return typeMap[ext] || 'other';
}

/**
 * Determine knowability based on file type
 */
function getKnowability(fileType, filename) {
  const ext = path.extname(filename).toLowerCase();

  // Binary files are UNKNOWN
  if (BINARY_EXTENSIONS.has(ext)) {
    return 'UNKNOWN';
  }

  // Source files are SOURCE_AUDITABLE
  const auditableTypes = new Set(['rs', 'js', 'ts', 'json', 'md', 'toml', 'sh', 'sql', 'yaml']);
  if (auditableTypes.has(fileType)) {
    return 'SOURCE_AUDITABLE';
  }

  return 'UNKNOWN';
}

/**
 * Count lines in file or get file size for binary files
 */
function getFileMetrics(filePath, fileType) {
  try {
    const ext = path.extname(filePath).toLowerCase();

    // For binary files, return 0 lines and file size
    if (BINARY_EXTENSIONS.has(ext)) {
      const stats = fs.statSync(filePath);
      return { loc: 0, fileSize: stats.size };
    }

    // For text files, count lines
    const content = fs.readFileSync(filePath, 'utf8');
    const lines = content.split('\n').length;
    return { loc: lines, fileSize: null };
  } catch (error) {
    console.error(`Error reading file ${filePath}: ${error.message}`);
    return { loc: 0, fileSize: null };
  }
}

/**
 * Recursively walk directory and collect file paths
 */
function walkDirectory(dirPath, baseDir, files = []) {
  try {
    const entries = fs.readdirSync(dirPath, { withFileTypes: true });

    for (const entry of entries) {
      const fullPath = path.join(dirPath, entry.name);

      if (entry.isDirectory()) {
        // Skip excluded directories
        if (SKIP_DIRS.has(entry.name)) {
          continue;
        }
        // Recursively walk subdirectory
        walkDirectory(fullPath, baseDir, files);
      } else if (entry.isFile()) {
        const relativePath = path.relative(baseDir, fullPath);
        files.push({ fullPath, relativePath });
      }
    }
  } catch (error) {
    console.error(`Error walking directory ${dirPath}: ${error.message}`);
  }

  return files;
}

/**
 * Index files for a single package
 */
function indexPackage(db, packageName, repoPath) {
  console.log(`\nIndexing package: ${packageName}`);
  console.log(`Repository path: ${repoPath}`);

  // Verify repo exists
  if (!fs.existsSync(repoPath)) {
    console.error(`ERROR: Repository path does not exist: ${repoPath}`);
    return { total: 0, byType: {} };
  }

  // Get package_id
  const packageRow = db.prepare('SELECT id FROM packages WHERE name = ?').get(packageName);
  if (!packageRow) {
    console.error(`ERROR: Package not found in database: ${packageName}`);
    return { total: 0, byType: {} };
  }
  const packageId = packageRow.id;

  // Walk directory and collect files
  const files = walkDirectory(repoPath, repoPath);
  console.log(`Found ${files.length} files to index`);

  // Prepare insert statement
  const insertStmt = db.prepare(`
    INSERT OR IGNORE INTO files (
      package_id, relative_path, loc, depth, file_type, knowability
    ) VALUES (?, ?, ?, 'NOT_TOUCHED', ?, ?)
  `);

  // Statistics
  const stats = { total: 0, byType: {} };

  // Begin transaction for batch insert
  const insertMany = db.transaction((fileList) => {
    for (const { fullPath, relativePath } of fileList) {
      const fileType = getFileType(relativePath);
      const knowability = getKnowability(fileType, relativePath);
      const { loc } = getFileMetrics(fullPath, fileType);

      const result = insertStmt.run(packageId, relativePath, loc, fileType, knowability);

      if (result.changes > 0) {
        stats.total++;
        stats.byType[fileType] = (stats.byType[fileType] || 0) + 1;
      }
    }
  });

  // Execute transaction
  insertMany(files);

  console.log(`Indexed ${stats.total} files for ${packageName}`);
  return stats;
}

/**
 * Main execution
 */
function main() {
  console.log('Starting Rust repository file indexing...');
  console.log(`Database: ${DB_PATH}\n`);

  // Connect to database
  const db = new Database(DB_PATH);

  try {
    // Overall statistics
    const overallStats = {
      totalFiles: 0,
      byPackage: {},
      byType: {}
    };

    // Index each package
    for (const [packageName, repoPath] of Object.entries(REPO_MAPPINGS)) {
      const stats = indexPackage(db, packageName, repoPath);

      overallStats.totalFiles += stats.total;
      overallStats.byPackage[packageName] = stats.total;

      // Aggregate by type
      for (const [type, count] of Object.entries(stats.byType)) {
        overallStats.byType[type] = (overallStats.byType[type] || 0) + count;
      }
    }

    // Print summary
    console.log('\n' + '='.repeat(60));
    console.log('INDEXING COMPLETE');
    console.log('='.repeat(60));
    console.log(`\nTotal files indexed: ${overallStats.totalFiles}`);

    console.log('\nFiles by package:');
    for (const [pkg, count] of Object.entries(overallStats.byPackage)) {
      console.log(`  ${pkg}: ${count}`);
    }

    console.log('\nFiles by type:');
    const sortedTypes = Object.entries(overallStats.byType)
      .sort((a, b) => b[1] - a[1]);
    for (const [type, count] of sortedTypes) {
      console.log(`  ${type}: ${count}`);
    }

  } catch (error) {
    console.error('ERROR during indexing:', error);
    process.exit(1);
  } finally {
    db.close();
  }

  console.log('\nDatabase connection closed.');
}

// Execute
main();
