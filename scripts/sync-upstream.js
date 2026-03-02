#!/usr/bin/env node
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const fs = require('fs');
const path = require('path');
const today = new Date().toISOString().slice(0, 10);
const sessionId = 130;

// ---- Package IDs ----
const pkgRuvector = db.prepare("SELECT id FROM packages WHERE name = 'ruvector-rust'").get().id;
const pkgAgenticFlow = db.prepare("SELECT id FROM packages WHERE name = 'agentic-flow-rust'").get().id;
const pkgClaudeFlow = db.prepare("SELECT id FROM packages WHERE name = 'claude-flow-cli'").get().id;

// ---- 1. ADDON CRATE EXCLUSION PATTERNS ----
const addonPatterns = [
  ['%agentic-robotics-%', 'Addon: robotics middleware crates (R115 sync)'],
  ['%ruvector-robotics%', 'Addon: robotics domain expansion (R115 sync)'],
  ['%ruvector-graph-transformer%', 'Addon: proof-gated graph transformer (R115 sync)'],
  ['%ruvector-dither%', 'Addon: dithering/quantization crate (R115 sync)'],
  ['%ruvector-verified%', 'Addon: formal verification layer (R115 sync)'],
  ['%thermorust%', 'Addon: thermodynamic neural-motif crate (R115 sync)'],
  ['%rvf-federation%', 'Addon: federated transfer learning (R115 sync)'],
  ['%exo-ai-%', 'Addon: EXO-AI multi-paradigm integration (R115 sync)'],
  ['%/examples/robotics/%', 'Addon: robotics examples (R115 sync)'],
  ['%/examples/verified-applications/%', 'Addon: verified-applications examples (R115 sync)'],
  ['%/examples/rvf-kernel-optimized/%', 'Addon: rvf-kernel-optimized example (R115 sync)'],
];

let excludeCount = 0;
const insertExclude = db.prepare('INSERT OR IGNORE INTO exclude_paths (pattern, reason, added_date) VALUES (?, ?, ?)');
for (const [pat, reason] of addonPatterns) {
  const r = insertExclude.run(pat, reason, today);
  if (r.changes > 0) excludeCount++;
}
console.log('Exclusion patterns added:', excludeCount);

// ---- 2. REGISTER NEW FILES ----
function walkDir(dir) {
  const results = [];
  try {
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    for (const e of entries) {
      const full = path.join(dir, e.name);
      if (e.isDirectory() && e.name !== 'target' && e.name !== 'node_modules' && e.name !== '.git' && e.name !== 'pkg') {
        results.push(...walkDir(full));
      } else if (e.isFile() && /\.(rs|ts|js)$/.test(e.name) && !e.name.endsWith('.d.ts')) {
        results.push(full);
      }
    }
  } catch(e) {}
  return results;
}

const insertFile = db.prepare('INSERT OR IGNORE INTO files (package_id, relative_path, loc, depth, lines_read, last_read_date) VALUES (?, ?, ?, ?, 0, NULL)');

// --- 2a. Ruvector addon crates ---
const repoRuvector = process.env.HOME + '/repos/ruvector';
const addonCrateDirs = [
  'crates/agentic-robotics-benchmarks',
  'crates/agentic-robotics-core',
  'crates/agentic-robotics-embedded',
  'crates/agentic-robotics-mcp',
  'crates/agentic-robotics-node',
  'crates/agentic-robotics-rt',
  'crates/ruvector-dither',
  'crates/ruvector-graph-transformer',
  'crates/ruvector-graph-transformer-node',
  'crates/ruvector-graph-transformer-wasm',
  'crates/ruvector-robotics',
  'crates/ruvector-verified',
  'crates/ruvector-verified-wasm',
  'crates/rvf/rvf-federation',
  'crates/thermorust',
];

let newAddonCount = 0;
let addonLoc = 0;
for (const crateDir of addonCrateDirs) {
  const fullDir = path.join(repoRuvector, crateDir);
  const files = walkDir(fullDir);
  for (const f of files) {
    const relPath = f.replace(repoRuvector + '/', '');
    let loc = 0;
    try { loc = fs.readFileSync(f, 'utf8').split('\n').length; } catch(e) {}
    const r = insertFile.run(pkgRuvector, relPath, loc, 'NOT_TOUCHED');
    if (r.changes > 0) {
      newAddonCount++;
      addonLoc += loc;
    }
  }
}
console.log('Ruvector addon files registered:', newAddonCount, '(' + addonLoc + ' LOC)');

// --- 2b. Agentic-flow new orchestration files ---
const repoAgenticFlow = process.env.HOME + '/repos/agentic-flow';
const afNewDirs = ['agentic-flow/src/orchestration', 'agentic-flow/tests/orchestration'];
let newAfCount = 0;
let afLoc = 0;
for (const dir of afNewDirs) {
  const fullDir = path.join(repoAgenticFlow, dir);
  const files = walkDir(fullDir);
  for (const f of files) {
    const relPath = f.replace(repoAgenticFlow + '/', '');
    let loc = 0;
    try { loc = fs.readFileSync(f, 'utf8').split('\n').length; } catch(e) {}
    const r = insertFile.run(pkgAgenticFlow, relPath, loc, 'NOT_TOUCHED');
    if (r.changes > 0) {
      newAfCount++;
      afLoc += loc;
    }
  }
}
console.log('Agentic-flow new files registered:', newAfCount, '(' + afLoc + ' LOC)');

// --- 2c. Claude-flow new files ---
const repoClaudeFlow = process.env.HOME + '/repos/claude-flow';
const cfBase = db.prepare("SELECT base_path FROM packages WHERE name = 'claude-flow-cli'").get().base_path;
const cfBasePath = cfBase.replace(/^~/, process.env.HOME);
const cfNewDirs = [
  'v3/@claude-flow/cli/src/appliance',
  'v3/@claude-flow/cli/src/commands',
  'v3/@claude-flow/cli/src/mcp-tools',
  'v3/@claude-flow/cli/src/memory',
  'v3/@claude-flow/cli/src/services',
  'v3/@claude-flow/embeddings/src',
  'v3/@claude-flow/memory/src',
  'v3/@claude-flow/shared/src/events',
];
let newCfCount = 0;
let cfLoc = 0;
for (const dir of cfNewDirs) {
  const fullDir = path.join(repoClaudeFlow, dir);
  const files = walkDir(fullDir);
  for (const f of files) {
    const relPath = f.replace(repoClaudeFlow + '/', '');
    let loc = 0;
    try { loc = fs.readFileSync(f, 'utf8').split('\n').length; } catch(e) {}
    const r = insertFile.run(pkgClaudeFlow, relPath, loc, 'NOT_TOUCHED');
    if (r.changes > 0) {
      newCfCount++;
      cfLoc += loc;
    }
  }
}
console.log('Claude-flow new files registered:', newCfCount, '(' + cfLoc + ' LOC)');

// ---- 3. UPDATE MODIFIED CORE FILES (LOC changes) ----
const modifiedCore = [
  'crates/cognitum-gate-kernel/src/canonical_witness.rs',
  'crates/cognitum-gate-kernel/src/lib.rs',
  'crates/ruvector-attention-unified-wasm/src/graph.rs',
  'crates/ruvector-cli/src/config.rs',
  'crates/ruvector-cli/src/mcp/gnn_cache.rs',
  'crates/ruvector-cli/src/mcp/handlers.rs',
  'crates/ruvector-cli/src/mcp/transport.rs',
  'crates/ruvector-cognitive-container/src/container.rs',
  'crates/ruvector-cognitive-container/src/memory.rs',
  'crates/ruvector-cognitive-container/src/witness.rs',
  'crates/ruvector-coherence/src/spectral.rs',
  'crates/ruvector-core/src/vector_db.rs',
  'crates/ruvector-crv/src/stage_iii.rs',
  'crates/ruvector-gnn-node/src/lib.rs',
  'crates/ruvector-gnn-wasm/src/lib.rs',
  'crates/ruvector-gnn/src/cold_tier.rs',
  'crates/ruvector-gnn/src/layer.rs',
  'crates/ruvector-gnn/src/mmap.rs',
  'crates/ruvector-gnn/src/search.rs',
  'crates/ruvector-mincut/src/canonical/mod.rs',
  'crates/ruvector-mincut/src/canonical/tests.rs',
  'crates/ruvector-mincut/src/lib.rs',
  'crates/rvf/rvf-types/src/segment_type.rs',
];

let locUpdated = 0;
const missingFromDb = [];
const locChanged = [];
for (const relPath of modifiedCore) {
  const fullPath = path.join(repoRuvector, relPath);
  let newLoc = 0;
  try { newLoc = fs.readFileSync(fullPath, 'utf8').split('\n').length; } catch(e) { continue; }

  const row = db.prepare('SELECT id, loc, depth FROM files WHERE relative_path = ? AND package_id = ?').get(relPath, pkgRuvector);
  if (!row) {
    insertFile.run(pkgRuvector, relPath, newLoc, 'NOT_TOUCHED');
    missingFromDb.push(relPath + ' (' + newLoc + ' LOC)');
  } else if (row.loc !== newLoc) {
    const delta = newLoc - row.loc;
    db.prepare('UPDATE files SET loc = ? WHERE id = ?').run(newLoc, row.id);
    locChanged.push({ path: relPath, old: row.loc, new: newLoc, delta, depth: row.depth });
    locUpdated++;
  }
}

if (missingFromDb.length) {
  console.log('\nNewly registered core files:', missingFromDb.length);
  missingFromDb.forEach(f => console.log('  NEW:', f));
}
if (locChanged.length) {
  console.log('\nCore files with LOC changes:', locChanged.length);
  locChanged.forEach(f => console.log('  MOD:', f.path + ': ' + f.old + ' -> ' + f.new + ' (' + (f.delta > 0 ? '+' : '') + f.delta + ') depth=' + f.depth));
}

// ---- 4. SUMMARY ----
const totalFiles = db.prepare('SELECT COUNT(*) as c FROM files').get().c;
const totalExcl = db.prepare('SELECT COUNT(*) as c FROM exclude_paths').get().c;
console.log('\n=== SUMMARY ===');
console.log('Total files in DB:', totalFiles);
console.log('Total exclusion patterns:', totalExcl);
console.log('New addon files (excluded from priority):', newAddonCount);
console.log('New agentic-flow files:', newAfCount);
console.log('New claude-flow files:', newCfCount);
console.log('Core files with LOC updates:', locUpdated);
console.log('Core files newly registered:', missingFromDb.length);

db.close();
