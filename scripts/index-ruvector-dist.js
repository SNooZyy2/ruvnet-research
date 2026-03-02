#!/usr/bin/env node
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const fs = require('fs');
const path = require('path');

// 1. Create package for ruvector umbrella
const existingPkg = db.prepare("SELECT id FROM packages WHERE name = 'ruvector-umbrella'").get();
let pkgId;
if (existingPkg) {
  pkgId = existingPkg.id;
  console.log('Package ruvector-umbrella already exists, id:', pkgId);
} else {
  const result = db.prepare("INSERT INTO packages (name, base_path, description) VALUES (?, ?, ?)").run(
    'ruvector-umbrella',
    '~/node_modules/ruvector/',
    'ruvector npm umbrella package (0.1.99) — VectorDBWrapper, 25+ JS modules in dist/core/'
  );
  pkgId = result.lastInsertRowid;
  console.log('Created package ruvector-umbrella, id:', pkgId);
}

// 2. Walk the dist/ directory and index all .js and .d.ts files
const basePath = '/home/snoozyy/node_modules/ruvector/';
const distPath = path.join(basePath, 'dist');

function walkDir(dir) {
  const results = [];
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      results.push(...walkDir(fullPath));
    } else if (entry.isFile()) {
      // Only .js and .d.ts files, skip .map files
      const name = entry.name;
      if (name.endsWith('.d.ts.map') || name.endsWith('.js.map')) continue;
      if (name.endsWith('.js') || name.endsWith('.d.ts')) {
        const content = fs.readFileSync(fullPath, 'utf-8');
        const loc = content.split('\n').length;
        const relPath = path.relative(basePath, fullPath);
        results.push({ relPath, loc, fullPath });
      }
    }
  }
  return results;
}

const files = walkDir(distPath);
console.log('Found', files.length, 'files to index');

// 3. Insert files, skip if already exists
const insertFile = db.prepare("INSERT OR IGNORE INTO files (package_id, relative_path, loc, depth) VALUES (?, ?, ?, 'NOT_TOUCHED')");
const checkFile = db.prepare("SELECT id FROM files WHERE relative_path = ?");

let inserted = 0;
let skipped = 0;
for (const f of files) {
  const dbPath = 'ruvector-umbrella/' + f.relPath;
  const existing = checkFile.get(dbPath);
  if (!existing) {
    insertFile.run(pkgId, dbPath, f.loc);
    inserted++;
  } else {
    skipped++;
  }
}

console.log('Inserted:', inserted, 'Skipped (already existed):', skipped);

// 4. Verify
const count = db.prepare("SELECT COUNT(*) as cnt FROM files WHERE package_id = ?").get(pkgId);
console.log('Total files in ruvector-umbrella package:', count.cnt);

const top10 = db.prepare("SELECT id, relative_path, loc, depth FROM files WHERE package_id = ? ORDER BY loc DESC LIMIT 10").all(pkgId);
console.log('\nTop 10 by LOC:');
console.log(JSON.stringify(top10, null, 2));

db.close();
