const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const fs = require('fs');
const path = require('path');
const base = '/home/snoozyy/repos/ruvector';

const tsFiles = [
  'npm/packages/rvf/src/backend.ts',
  'npm/packages/rvf/src/index.ts',
  'npm/packages/rvf/src/types.ts',
  'npm/packages/rvf/src/errors.ts',
  'npm/packages/rvf/src/database.ts',
];

for (const relPath of tsFiles) {
  const full = path.join(base, relPath);
  if (fs.existsSync(full)) {
    const loc = fs.readFileSync(full, 'utf8').split('\n').length;
    db.prepare("INSERT OR IGNORE INTO files (package_id, relative_path, file_type, loc, depth) VALUES (8, ?, ?, ?, 'NOT_TOUCHED')").run(relPath, 'ts', loc);
    const fid = db.prepare('SELECT id FROM files WHERE package_id = 8 AND relative_path = ?').get(relPath);
    if (fid) {
      db.prepare('INSERT OR IGNORE INTO file_domains (file_id, domain_id) VALUES (?, 9)').run(fid.id);
    }
    console.log(relPath, loc, 'LOC');
  }
}
db.close();
