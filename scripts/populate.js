#!/usr/bin/env node
/**
 * populate.js
 * Walks package directories and populates the files table
 *
 * Usage: node research/scripts/populate.js
 *
 * For each package in the packages table:
 * - Expands ~ in base_path to actual home directory
 * - Walks the directory tree recursively
 * - Filters by file extension and directory exclusions
 * - Counts lines of code
 * - Inserts into files table with depth='NOT_TOUCHED'
 * - Updates package totals
 */

const Database = require('better-sqlite3');
const fs = require('fs');
const path = require('path');
const os = require('os');

const DB_PATH = path.join(__dirname, '..', 'db', 'research.db');

// File extensions to include
const INCLUDE_EXTENSIONS = new Set([
  '.js', '.mjs', '.cjs', '.ts',
  '.sh', '.json', '.wasm', '.sql',
  '.yaml', '.yml'
]);

// Special case: .md files only if in specific directories
const MD_ALLOWED_DIRS = new Set(['agents', 'skills', 'commands']);

// Directory patterns to exclude
const EXCLUDE_DIRS = new Set([
  'node_modules', '.git', '__pycache__',
  '.cache', 'coverage'
]);

// File patterns to exclude
const EXCLUDE_FILES = new Set(['package-lock.json']);

/**
 * Expand ~ in path to actual home directory
 */
function expandHome(filepath) {
  if (filepath.startsWith('~/')) {
    return path.join(os.homedir(), filepath.slice(2));
  }
  return filepath;
}

/**
 * Check if a directory should be excluded
 */
function shouldExcludeDir(dirName) {
  return EXCLUDE_DIRS.has(dirName);
}

/**
 * Check if a file should be included based on extension and directory
 */
function shouldIncludeFile(filePath, relativePath, packageName) {
  const fileName = path.basename(filePath);
  const ext = path.extname(filePath);

  // Exclude specific files
  if (EXCLUDE_FILES.has(fileName)) {
    return false;
  }

  // Exclude .map files
  if (ext === '.map') {
    return false;
  }

  // Exclude .d.ts files UNLESS package is 'custom-src'
  if (fileName.endsWith('.d.ts') && packageName !== 'custom-src') {
    return false;
  }

  // Check for .md files - only include if in specific directories
  if (ext === '.md') {
    const pathParts = relativePath.split(path.sep);
    const hasAllowedDir = pathParts.some(part => MD_ALLOWED_DIRS.has(part));
    return hasAllowedDir;
  }

  // Check if extension is in allowed list
  return INCLUDE_EXTENSIONS.has(ext);
}

/**
 * Count lines in a file
 * Returns null if file is binary or unreadable
 */
function countLines(filePath) {
  try {
    const content = fs.readFileSync(filePath, 'utf8');
    return content.split('\n').length;
  } catch (error) {
    // File is binary or unreadable
    return null;
  }
}

/**
 * Recursively walk directory tree
 */
function* walkDir(dirPath, basePath = dirPath) {
  try {
    const entries = fs.readdirSync(dirPath, { withFileTypes: true });

    for (const entry of entries) {
      const fullPath = path.join(dirPath, entry.name);
      const relativePath = path.relative(basePath, fullPath);

      if (entry.isDirectory()) {
        // Skip excluded directories
        if (!shouldExcludeDir(entry.name)) {
          yield* walkDir(fullPath, basePath);
        }
      } else if (entry.isFile()) {
        yield { fullPath, relativePath };
      }
    }
  } catch (error) {
    console.error(`  Warning: Cannot read directory ${dirPath}: ${error.message}`);
  }
}

/**
 * Process a single package
 */
function processPackage(db, pkg) {
  const { id, name, base_path } = pkg;
  const expandedPath = expandHome(base_path);

  console.log(`\nProcessing package: ${name}`);
  console.log(`  Base path: ${base_path}`);
  console.log(`  Expanded: ${expandedPath}`);

  // Check if path exists
  if (!fs.existsSync(expandedPath)) {
    console.log(`  SKIP: Path does not exist on disk\n`);
    return;
  }

  // Prepare statements
  const insertStmt = db.prepare(`
    INSERT OR IGNORE INTO files (package_id, relative_path, loc, depth)
    VALUES (?, ?, ?, 'NOT_TOUCHED')
  `);

  const updatePkgStmt = db.prepare(`
    UPDATE packages
    SET total_files = ?,
        total_loc = ?,
        populated_at = ?
    WHERE id = ?
  `);

  // Walk directory and collect files
  let filesFound = 0;
  let totalLoc = 0;

  // Use transaction for performance
  const transaction = db.transaction(() => {
    for (const { fullPath, relativePath } of walkDir(expandedPath)) {
      // Check if file should be included
      if (!shouldIncludeFile(fullPath, relativePath, name)) {
        continue;
      }

      // Count lines
      const loc = countLines(fullPath);

      // Insert into database
      const result = insertStmt.run(id, relativePath, loc);

      // Track stats (only if actually inserted, not if already existed)
      if (result.changes > 0) {
        filesFound++;
        if (loc !== null) {
          totalLoc += loc;
        }
      }
    }

    // Update package totals
    // Get actual counts from database (in case some files already existed)
    const stats = db.prepare(`
      SELECT COUNT(*) as file_count, COALESCE(SUM(loc), 0) as total_loc
      FROM files
      WHERE package_id = ?
    `).get(id);

    updatePkgStmt.run(
      stats.file_count,
      stats.total_loc,
      new Date().toISOString(),
      id
    );

    return { filesFound, totalLoc, actualTotal: stats.file_count, actualLoc: stats.total_loc };
  });

  const stats = transaction();

  console.log(`  Files found: ${stats.filesFound} new (${stats.actualTotal} total in DB)`);
  console.log(`  Lines of code: ${stats.totalLoc.toLocaleString()} new (${stats.actualLoc.toLocaleString()} total in DB)`);
}

function main() {
  try {
    console.log('Research Ledger Database Population');
    console.log('====================================\n');

    // Open database
    if (!fs.existsSync(DB_PATH)) {
      console.error(`ERROR: Database not found at ${DB_PATH}`);
      console.error('Run init-db.js first to create the database.');
      process.exit(1);
    }

    const db = new Database(DB_PATH);
    db.pragma('foreign_keys = ON');

    // Get all packages
    const packages = db.prepare('SELECT id, name, base_path FROM packages ORDER BY name').all();

    console.log(`Found ${packages.length} packages to process\n`);

    // Process each package
    for (const pkg of packages) {
      processPackage(db, pkg);
    }

    // Print overall summary
    console.log('\n====================================');
    console.log('Population Summary');
    console.log('====================================\n');

    const totalStats = db.prepare(`
      SELECT
        COUNT(*) as total_files,
        COALESCE(SUM(loc), 0) as total_loc
      FROM files
    `).get();

    const packageStats = db.prepare(`
      SELECT
        name,
        total_files,
        total_loc
      FROM packages
      WHERE total_files > 0
      ORDER BY name
    `).all();

    console.log('Files by package:');
    for (const pkg of packageStats) {
      console.log(`  ${pkg.name}: ${pkg.total_files} files, ${pkg.total_loc.toLocaleString()} LOC`);
    }

    console.log(`\nTotal across all packages:`);
    console.log(`  Files: ${totalStats.total_files}`);
    console.log(`  Lines of code: ${totalStats.total_loc.toLocaleString()}\n`);

    // Close database
    db.close();

    console.log('====================================');
    console.log('Population complete!');
    console.log('====================================\n');

    process.exit(0);

  } catch (error) {
    console.error('\nERROR during population:');
    console.error(error.message);
    console.error(error.stack);
    process.exit(1);
  }
}

main();
