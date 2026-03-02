const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const PKG_ID = 8; // ruvector-rust
const BASE = '/home/snoozyy/repos/ruvector';

// Get all new/modified source files since 1c1678c (our old HEAD)
const raw = execSync(
  'cd /home/snoozyy/repos/ruvector && git diff --name-only 1c1678c..HEAD -- "*.rs" "*.ts" "*.js" | grep -v node_modules | grep -v ".d.ts" | grep -v ".js.map"',
  { encoding: 'utf8' }
);
const newFiles = raw.trim().split('\n').filter(Boolean);
console.log('Total new/modified source files:', newFiles.length);

const stmt = db.prepare(`
  INSERT OR IGNORE INTO files (package_id, relative_path, file_type, loc, depth)
  VALUES (?, ?, ?, ?, 'NOT_TOUCHED')
`);

const updateStmt = db.prepare(`
  UPDATE files SET loc = ? WHERE package_id = ? AND relative_path = ?
`);

let inserted = 0;
let updated = 0;
let errors = 0;

const insertMany = db.transaction(() => {
  for (const relPath of newFiles) {
    const fullPath = path.join(BASE, relPath);
    try {
      if (!fs.existsSync(fullPath)) continue;
      const content = fs.readFileSync(fullPath, 'utf8');
      const loc = content.split('\n').length;
      const ext = path.extname(relPath).slice(1);

      const result = stmt.run(PKG_ID, relPath, ext, loc);
      if (result.changes > 0) {
        inserted++;
      } else {
        // File already exists - update LOC
        updateStmt.run(loc, PKG_ID, relPath);
        updated++;
      }
    } catch (e) {
      errors++;
    }
  }
});

insertMany();

// Update package totals
const stats = db.prepare('SELECT COUNT(*) as cnt, COALESCE(SUM(loc),0) as total_loc FROM files WHERE package_id = ?').get(PKG_ID);
db.prepare('UPDATE packages SET total_files = ?, total_loc = ? WHERE id = ?').run(stats.cnt, stats.total_loc, PKG_ID);

console.log('Inserted:', inserted);
console.log('Updated:', updated);
console.log('Errors:', errors);
console.log('Package total files:', stats.cnt, 'LOC:', stats.total_loc);
db.close();
