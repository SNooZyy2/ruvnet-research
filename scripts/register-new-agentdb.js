const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');

const pkg = db.prepare("SELECT id FROM packages WHERE name = 'agentic-flow'").get();
if (!pkg) { console.error('Package not found'); process.exit(1); }
const pkgId = pkg.id;

const dom = db.prepare("SELECT id FROM domains WHERE name = 'agentdb-integration'").get();
const domId = dom ? dom.id : null;

const files = [
  ['packages/agentdb/LICENSE', 21],
  ['packages/agentdb/docs/adrs/ADR-002-ruvector-wasm-integration.md', 648],
  ['packages/agentdb/docs/adrs/ADR-003-rvf-native-format-integration.md', 1400],
  ['packages/agentdb/docs/adrs/ADR-004-agi-capabilities-integration.md', 171],
  ['packages/agentdb/docs/adrs/ADR-005-self-learning-pipeline-integration.md', 165],
  ['packages/agentdb/docs/adrs/ADR-006-unified-self-learning-rvf-integration.md', 670],
  ['packages/agentdb/docs/adrs/ADR-007-ruvector-full-capability-integration.md', 872],
  ['packages/agentdb/docs/adrs/ADR-008-chat-ui-rvf-kernel-embedding.md', 743],
  ['packages/agentdb/docs/adrs/ADR-009-causal-atlas-rvf-runtime.md', 0],
  ['packages/agentdb/docs/adrs/ADR-010-rvf-solver-v014-deep-integration.md', 305],
  ['packages/agentdb/src/backends/rvf/AdaptiveIndexTuner.ts', 631],
  ['packages/agentdb/src/backends/rvf/ContrastiveTrainer.ts', 559],
  ['packages/agentdb/src/backends/rvf/FederatedSessionManager.ts', 526],
  ['packages/agentdb/src/backends/rvf/FilterBuilder.ts', 209],
  ['packages/agentdb/src/backends/rvf/NativeAccelerator.ts', 489],
  ['packages/agentdb/src/backends/rvf/RvfBackend.ts', 749],
  ['packages/agentdb/src/backends/rvf/RvfSolver.ts', 312],
  ['packages/agentdb/src/backends/rvf/SelfLearningRvfBackend.ts', 487],
  ['packages/agentdb/src/backends/rvf/SemanticQueryRouter.ts', 456],
  ['packages/agentdb/src/backends/rvf/SimdFallbacks.ts', 254],
  ['packages/agentdb/src/backends/rvf/SolverBandit.ts', 270],
  ['packages/agentdb/src/backends/rvf/SonaLearningBackend.ts', 357],
  ['packages/agentdb/src/backends/rvf/SqlJsRvfBackend.ts', 457],
  ['packages/agentdb/src/backends/rvf/WasmStoreBridge.ts', 83],
  ['packages/agentdb/src/backends/rvf/validation.ts', 82],
  ['packages/agentdb/src/cli/commands/rvf.ts', 501],
  ['packages/agentdb/src/model/ModelCacheLoader.ts', 144],
  ['packages/agentdb/src/utils/chalk-fallback.ts', 20],
  ['packages/agentdb/src/utils/similarity.ts', 28],
  ['packages/agentdb/scripts/build-model-rvf.mjs', 159],
];

const insert = db.prepare('INSERT INTO files (package_id, relative_path, loc, depth) VALUES (?, ?, ?, ?)');
const tagDomain = db.prepare('INSERT OR IGNORE INTO file_domains (file_id, domain_id) VALUES (?, ?)');

let inserted = 0;
let skipped = 0;
let tagged = 0;

const txn = db.transaction(() => {
  for (const [path, loc] of files) {
    const existing = db.prepare('SELECT id FROM files WHERE relative_path = ?').get(path);
    if (existing) {
      skipped++;
      // Still tag existing files with domain if not tagged
      if (domId && (path.endsWith('.ts') || path.endsWith('.mjs'))) {
        tagDomain.run(existing.id, domId);
      }
      continue;
    }
    const result = insert.run(pkgId, path, loc, 'NOT_TOUCHED');
    inserted++;
    if (domId && (path.endsWith('.ts') || path.endsWith('.mjs'))) {
      tagDomain.run(result.lastInsertRowid, domId);
      tagged++;
    }
  }
});

txn();

console.log('Inserted:', inserted, '| Skipped (already exists):', skipped, '| Domain-tagged:', tagged);
console.log('Total new LOC registered:', files.reduce((s, f) => s + f[1], 0));

const count = db.prepare("SELECT COUNT(*) as c FROM files WHERE relative_path LIKE '%packages/agentdb/%'").get();
console.log('Total agentdb files in DB now:', count.c);

db.close();
