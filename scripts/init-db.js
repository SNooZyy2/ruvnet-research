#!/usr/bin/env node
/**
 * init-db.js
 * Creates research.db from schema.sql
 *
 * Usage: node research/scripts/init-db.js
 *
 * - Reads schema.sql
 * - Deletes existing research.db if present
 * - Creates new database with schema + seed data
 * - Sets pragmas: journal_mode=WAL, foreign_keys=ON
 * - Verifies tables and views
 * - Prints summary
 */

const Database = require('better-sqlite3');
const fs = require('fs');
const path = require('path');

const DB_DIR = path.join(__dirname, '..', 'db');
const SCHEMA_PATH = path.join(DB_DIR, 'schema.sql');
const DB_PATH = path.join(DB_DIR, 'research.db');

function main() {
  try {
    console.log('Research Ledger Database Initialization');
    console.log('========================================\n');

    // Step 1: Read schema.sql
    console.log(`Reading schema from: ${SCHEMA_PATH}`);
    if (!fs.existsSync(SCHEMA_PATH)) {
      console.error(`ERROR: Schema file not found at ${SCHEMA_PATH}`);
      process.exit(1);
    }
    const schema = fs.readFileSync(SCHEMA_PATH, 'utf8');
    console.log(`Schema loaded (${schema.length} bytes)\n`);

    // Step 2: Delete existing DB if present
    if (fs.existsSync(DB_PATH)) {
      console.log(`Deleting existing database: ${DB_PATH}`);
      fs.unlinkSync(DB_PATH);
      // Also delete WAL files if they exist
      if (fs.existsSync(DB_PATH + '-wal')) fs.unlinkSync(DB_PATH + '-wal');
      if (fs.existsSync(DB_PATH + '-shm')) fs.unlinkSync(DB_PATH + '-shm');
      console.log('Existing database removed\n');
    }

    // Step 3: Create new database
    console.log(`Creating new database: ${DB_PATH}`);
    const db = new Database(DB_PATH);

    // Step 4: Set pragmas
    console.log('Setting pragmas...');
    db.pragma('journal_mode = WAL');
    db.pragma('foreign_keys = ON');
    const journalMode = db.pragma('journal_mode', { simple: true });
    const foreignKeys = db.pragma('foreign_keys', { simple: true });
    console.log(`  journal_mode: ${journalMode}`);
    console.log(`  foreign_keys: ${foreignKeys}\n`);

    // Step 5: Execute schema SQL
    console.log('Executing schema SQL...');
    db.exec(schema);
    console.log('Schema execution complete\n');

    // Step 6: Verify tables and views
    console.log('Verifying database structure...');
    const tables = db.prepare(`
      SELECT name, type
      FROM sqlite_master
      WHERE type IN ('table', 'view')
      ORDER BY type, name
    `).all();

    const tableCount = tables.filter(t => t.type === 'table').length;
    const viewCount = tables.filter(t => t.type === 'view').length;

    console.log(`\nTables (${tableCount}):`);
    tables.filter(t => t.type === 'table').forEach(t => {
      console.log(`  - ${t.name}`);
    });

    console.log(`\nViews (${viewCount}):`);
    tables.filter(t => t.type === 'view').forEach(t => {
      console.log(`  - ${t.name}`);
    });

    // Step 7: Check seed data counts
    console.log('\nSeed data summary:');
    const packageCount = db.prepare('SELECT COUNT(*) as count FROM packages').get().count;
    const domainCount = db.prepare('SELECT COUNT(*) as count FROM domains').get().count;
    const sessionCount = db.prepare('SELECT COUNT(*) as count FROM sessions').get().count;
    const packageDepCount = db.prepare('SELECT COUNT(*) as count FROM package_dependencies').get().count;

    console.log(`  Packages: ${packageCount}`);
    console.log(`  Domains: ${domainCount}`);
    console.log(`  Sessions: ${sessionCount}`);
    console.log(`  Package dependencies: ${packageDepCount}`);

    // Step 8: Close database
    db.close();

    console.log('\n========================================');
    console.log('Database initialization successful!');
    console.log(`Database created at: ${DB_PATH}`);
    console.log(`Tables: ${tableCount}, Views: ${viewCount}`);
    console.log('========================================\n');

    process.exit(0);

  } catch (error) {
    console.error('\nERROR during database initialization:');
    console.error(error.message);
    console.error(error.stack);
    process.exit(1);
  }
}

main();
