#!/usr/bin/env node
/**
 * R-Plan Generator for v4-gold-sweep
 *
 * Usage:
 *   node scripts/gen-rplan.js                  # Next batch (9 files, module-grouped)
 *   node scripts/gen-rplan.js --batch 2        # Second batch
 *   node scripts/gen-rplan.js --tier connected  # Only CONNECTED tier
 *   node scripts/gen-rplan.js --module ruvllm-rb # Specific module group
 *   node scripts/gen-rplan.js --list            # List all modules and their status
 */

const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');

const args = process.argv.slice(2);
const flags = {};
for (let i = 0; i < args.length; i++) {
  if (args[i].startsWith('--')) {
    flags[args[i].slice(2)] = args[i + 1] || true;
    i++;
  }
}

// Module groupings (for batch reading — read mod.rs first, then siblings)
const MODULE_GROUPS = {
  'ruvllm-cf': {
    name: 'ruvllm claude_flow integration',
    modFile: 'crates/ruvllm/src/claude_flow/mod.rs',
    dir: 'crates/ruvllm/src/claude_flow/',
    tier: 'CRITICAL',
    v4relevance: 'Direct claude-flow integration — hooks, task generation, routing',
  },
  'ruvllm-rb': {
    name: 'ruvllm reasoning_bank',
    modFile: 'crates/ruvllm/src/reasoning_bank/mod.rs',
    dir: 'crates/ruvllm/src/reasoning_bank/',
    tier: 'CRITICAL',
    v4relevance: 'ReasoningBank Rust implementation — verdicts, patterns, distillation',
  },
  'ruvllm-ctx': {
    name: 'ruvllm context (agent memory)',
    modFile: 'crates/ruvllm/src/context/mod.rs',
    dir: 'crates/ruvllm/src/context/',
    tier: 'CRITICAL',
    v4relevance: 'Agent memory architecture — episodic, working, semantic cache',
  },
  'ruvllm-qual': {
    name: 'ruvllm quality',
    modFile: 'crates/ruvllm/src/quality/mod.rs',
    dir: 'crates/ruvllm/src/quality/',
    tier: 'HIGH',
    v4relevance: 'Output quality validation — validators, diversity, coherence',
  },
  'ruvllm-lora': {
    name: 'ruvllm LoRA adapters',
    modFile: 'crates/ruvllm/src/lora/mod.rs',
    dir: 'crates/ruvllm/src/lora/',
    includeSubdirs: true,
    tier: 'HIGH',
    v4relevance: 'LoRA adapter management — training, merge, adapter lifecycle',
  },
  'ruvllm-rv': {
    name: 'ruvllm-ruvector bridge',
    modFile: null,
    dir: null,
    explicitFiles: ['crates/ruvllm/src/ruvector_integration.rs'],
    tier: 'HIGH',
    v4relevance: 'How ruvllm talks to ruvector-core — bridge code',
  },
  'ruvllm-serve': {
    name: 'ruvllm serving',
    modFile: 'crates/ruvllm/src/serving/mod.rs',
    dir: 'crates/ruvllm/src/serving/',
    tier: 'HIGH',
    v4relevance: 'Model serving — KV cache, batching, request handling',
  },
  'sheaf': {
    name: 'ruvector-attention sheaf',
    modFile: 'crates/ruvector-attention/src/sheaf/mod.rs',
    dir: 'crates/ruvector-attention/src/sheaf/',
    tier: 'HIGH',
    v4relevance: 'Sheaf-theoretic attention — novel math, early exit, sparse',
  },
  'rv-transport': {
    name: 'ruvector-attention transport',
    modFile: null,
    dir: 'crates/ruvector-attention/src/transport/',
    tier: 'MEDIUM',
    v4relevance: 'Optimal transport attention — Wasserstein, centroid OT',
  },
  'rv-training': {
    name: 'ruvector-attention training',
    modFile: null,
    dir: 'crates/ruvector-attention/src/training/',
    tier: 'MEDIUM',
    v4relevance: 'Attention training — optimizer, loss, curriculum, mining',
  },
  'pr-coher': {
    name: 'prime-radiant coherence',
    modFile: 'crates/prime-radiant/src/coherence/mod.rs',
    dir: 'crates/prime-radiant/src/coherence/',
    tier: 'MEDIUM',
    v4relevance: 'Knowledge consistency — energy, spectral, incremental',
  },
  'pr-cohom': {
    name: 'prime-radiant cohomology',
    modFile: 'crates/prime-radiant/src/cohomology/mod.rs',
    dir: 'crates/prime-radiant/src/cohomology/',
    tier: 'MEDIUM',
    v4relevance: 'Algebraic topology — cocycles, sheaves, laplacians',
  },
  'pr-gov': {
    name: 'prime-radiant governance',
    modFile: 'crates/prime-radiant/src/governance/mod.rs',
    dir: 'crates/prime-radiant/src/governance/',
    tier: 'MEDIUM',
    v4relevance: 'Policy, lineage, witness — provenance tracking',
  },
  'pr-store': {
    name: 'prime-radiant storage',
    modFile: 'crates/prime-radiant/src/storage/mod.rs',
    dir: 'crates/prime-radiant/src/storage/',
    tier: 'MEDIUM',
    v4relevance: 'Persistence — postgres, file, memory backends',
  },
};

// Helper: get files for a module
function getModuleFiles(mod) {
  if (mod.explicitFiles) {
    return mod.explicitFiles.map(path => {
      return db.prepare('SELECT id, relative_path, loc, depth FROM files WHERE relative_path = ?').get(path);
    }).filter(Boolean);
  }
  if (mod.includeSubdirs) {
    return db.prepare(`
      SELECT id, relative_path, loc, depth FROM files
      WHERE relative_path LIKE ? || '%'
        AND relative_path LIKE '%.rs'
      ORDER BY loc DESC
    `).all(mod.dir);
  }
  return db.prepare(`
    SELECT id, relative_path, loc, depth FROM files
    WHERE relative_path LIKE ? || '%'
      AND relative_path NOT LIKE ? || '%/%'
      AND relative_path LIKE '%.rs'
    ORDER BY loc DESC
  `).all(mod.dir, mod.dir);
}

if (flags.list) {
  console.log('=== V4-GOLD-SWEEP MODULE GROUPS ===\n');
  for (const [key, mod] of Object.entries(MODULE_GROUPS)) {
    const files = getModuleFiles(mod);
    const unread = files.filter(f => f.depth === 'NOT_TOUCHED');
    const deep = files.filter(f => f.depth === 'DEEP');
    const totalLoc = unread.reduce((s, f) => s + f.loc, 0);

    const status = unread.length === 0 ? 'DONE' : `${unread.length} unread (${totalLoc} LOC)`;
    console.log(`  [${mod.tier}] ${key}: ${mod.name}`);
    console.log(`    ${deep.length} DEEP, ${status}`);
    console.log(`    v4: ${mod.v4relevance}`);
    console.log('');
  }
  db.close();
  process.exit(0);
}

// Generate next batch
const batchNum = parseInt(flags.batch || '1');
const batchSize = parseInt(flags.size || '9');
const tierFilter = flags.tier;
const moduleFilter = flags.module;

// Get files ordered by tier priority, then by LOC
let allFiles = [];
const tierOrder = { CRITICAL: 0, HIGH: 1, MEDIUM: 2 };

for (const [key, mod] of Object.entries(MODULE_GROUPS)) {
  if (moduleFilter && key !== moduleFilter) continue;
  if (tierFilter && mod.tier.toLowerCase() !== tierFilter.toLowerCase()) continue;

  const files = getModuleFiles(mod).filter(f => f.depth === 'NOT_TOUCHED');

  for (const f of files) {
    allFiles.push({ ...f, module: key, moduleName: mod.name, tier: mod.tier, v4relevance: mod.v4relevance });
  }
}

// Sort: CRITICAL first, then HIGH, then MEDIUM. Within tier, mod.rs first, then by LOC
allFiles.sort((a, b) => {
  const aTier = tierOrder[a.tier] !== undefined ? tierOrder[a.tier] : 99;
  const bTier = tierOrder[b.tier] !== undefined ? tierOrder[b.tier] : 99;
  if (aTier !== bTier) return aTier - bTier;
  // mod.rs first within same module
  if (a.module === b.module) {
    const aIsMod = a.relative_path.endsWith('/mod.rs') ? 0 : 1;
    const bIsMod = b.relative_path.endsWith('/mod.rs') ? 0 : 1;
    if (aIsMod !== bIsMod) return aIsMod - bIsMod;
  }
  return b.loc - a.loc;
});

// Slice for this batch
const offset = (batchNum - 1) * batchSize;
const batch = allFiles.slice(offset, offset + batchSize);

if (batch.length === 0) {
  console.log('No files remaining for this batch.');
  console.log(`Total unread: ${allFiles.length} files`);
  db.close();
  process.exit(0);
}

// Output the plan
const totalBatches = Math.ceil(allFiles.length / batchSize);
console.log(`# R-Plan: v4-gold-sweep Batch ${batchNum}/${totalBatches}`);
console.log(`# ${batch.length} files, ${batch.reduce((s, f) => s + f.loc, 0)} LOC`);
console.log(`# Remaining after this batch: ${allFiles.length - offset - batch.length} files\n`);

// Group by module for the plan
const byModule = {};
for (const f of batch) {
  if (!byModule[f.module]) {
    byModule[f.module] = { name: f.moduleName, tier: f.tier, v4relevance: f.v4relevance, files: [] };
  }
  byModule[f.module].files.push(f);
}

for (const [key, mod] of Object.entries(byModule)) {
  console.log(`## ${mod.name} [${mod.tier}]`);
  console.log(`v4 relevance: ${mod.v4relevance}\n`);
  for (const f of mod.files) {
    console.log(`- [ ] ${f.relative_path} (${f.loc} LOC) [file_id: ${f.id}]`);
  }
  console.log('');
}

console.log('---');
console.log(`# Agent instructions:`);
console.log(`# - Read mod.rs FIRST in each module to understand structure`);
console.log(`# - For each file: assess realness %, insert findings, update depth`);
console.log(`# - Key question: is this code GENUINE and REUSABLE for v4?`);
console.log(`# - Look for: real algorithms, working implementations, reusable patterns`);
console.log(`# - Flag: facades, placeholders, duplicates of code we already have`);

db.close();
