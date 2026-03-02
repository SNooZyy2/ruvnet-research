#!/usr/bin/env node
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const today = new Date().toISOString().slice(0, 10);

// 1. Create domain
db.prepare('INSERT OR IGNORE INTO domains (name, priority, description) VALUES (?, ?, ?)').run(
  'v4-gold-sweep', 'HIGH',
  'Unread .rs files in gold crates (85%+ siblings) — potential high-value code for v4 rebuild. 70 files, ~43K LOC across ruvllm, ruvector-attention, prime-radiant.'
);
const sweepDomain = db.prepare("SELECT id FROM domains WHERE name = 'v4-gold-sweep'").get();
console.log('Domain v4-gold-sweep ID:', sweepDomain.id);

// 2. Define modules by priority
const modules = {
  CRITICAL: [
    ['crates/ruvllm/src/claude_flow/hooks_integration.rs', 'ruvllm-cf'],
    ['crates/ruvllm/src/claude_flow/task_generator.rs', 'ruvllm-cf'],
    ['crates/ruvllm/src/claude_flow/mod.rs', 'ruvllm-cf'],
    ['crates/ruvllm/src/claude_flow/task_classifier.rs', 'ruvllm-cf'],
    ['crates/ruvllm/src/claude_flow/flow_optimizer.rs', 'ruvllm-cf'],
    ['crates/ruvllm/src/claude_flow/agent_router.rs', 'ruvllm-cf'],
    ['crates/ruvllm/src/reasoning_bank/verdicts.rs', 'ruvllm-rb'],
    ['crates/ruvllm/src/reasoning_bank/pattern_store.rs', 'ruvllm-rb'],
    ['crates/ruvllm/src/reasoning_bank/distillation.rs', 'ruvllm-rb'],
    ['crates/ruvllm/src/reasoning_bank/consolidation.rs', 'ruvllm-rb'],
    ['crates/ruvllm/src/reasoning_bank/trajectory.rs', 'ruvllm-rb'],
    ['crates/ruvllm/src/reasoning_bank/mod.rs', 'ruvllm-rb'],
    ['crates/ruvllm/src/context/agentic_memory.rs', 'ruvllm-ctx'],
    ['crates/ruvllm/src/context/context_manager.rs', 'ruvllm-ctx'],
    ['crates/ruvllm/src/context/episodic_memory.rs', 'ruvllm-ctx'],
    ['crates/ruvllm/src/context/working_memory.rs', 'ruvllm-ctx'],
    ['crates/ruvllm/src/context/claude_flow_bridge.rs', 'ruvllm-ctx'],
    ['crates/ruvllm/src/context/semantic_cache.rs', 'ruvllm-ctx'],
    ['crates/ruvllm/src/context/mod.rs', 'ruvllm-ctx'],
  ],
  HIGH: [
    ['crates/ruvllm/src/quality/validators.rs', 'ruvllm-qual'],
    ['crates/ruvllm/src/quality/diversity.rs', 'ruvllm-qual'],
    ['crates/ruvllm/src/quality/coherence.rs', 'ruvllm-qual'],
    ['crates/ruvllm/src/quality/metrics.rs', 'ruvllm-qual'],
    ['crates/ruvllm/src/quality/mod.rs', 'ruvllm-qual'],
    ['crates/ruvllm/src/lora/training.rs', 'ruvllm-lora'],
    ['crates/ruvllm/src/lora/adapter.rs', 'ruvllm-lora'],
    ['crates/ruvllm/src/lora/adapters/merge.rs', 'ruvllm-lora'],
    ['crates/ruvllm/src/lora/adapters/trainer.rs', 'ruvllm-lora'],
    ['crates/ruvllm/src/lora/adapters/mod.rs', 'ruvllm-lora'],
    ['crates/ruvllm/src/lora/mod.rs', 'ruvllm-lora'],
    ['crates/ruvllm/src/ruvector_integration.rs', 'ruvllm-rv'],
    ['crates/ruvector-attention/src/sheaf/attention.rs', 'sheaf'],
    ['crates/ruvector-attention/src/sheaf/sparse.rs', 'sheaf'],
    ['crates/ruvector-attention/src/sheaf/router.rs', 'sheaf'],
    ['crates/ruvector-attention/src/sheaf/early_exit.rs', 'sheaf'],
    ['crates/ruvector-attention/src/sheaf/restriction.rs', 'sheaf'],
    ['crates/ruvllm/src/serving/kv_cache_manager.rs', 'ruvllm-serve'],
    ['crates/ruvllm/src/serving/batch.rs', 'ruvllm-serve'],
    ['crates/ruvllm/src/serving/request.rs', 'ruvllm-serve'],
    ['crates/ruvllm/src/serving/mod.rs', 'ruvllm-serve'],
  ],
  MEDIUM: [
    ['crates/prime-radiant/src/cohomology/neural.rs', 'pr-cohom'],
    ['crates/prime-radiant/src/cohomology/cohomology_group.rs', 'pr-cohom'],
    ['crates/prime-radiant/src/cohomology/simplex.rs', 'pr-cohom'],
    ['crates/prime-radiant/src/cohomology/laplacian.rs', 'pr-cohom'],
    ['crates/prime-radiant/src/cohomology/obstruction.rs', 'pr-cohom'],
    ['crates/prime-radiant/src/cohomology/diffusion.rs', 'pr-cohom'],
    ['crates/prime-radiant/src/cohomology/cocycle.rs', 'pr-cohom'],
    ['crates/prime-radiant/src/cohomology/sheaf.rs', 'pr-cohom'],
    ['crates/prime-radiant/src/cohomology/mod.rs', 'pr-cohom'],
    ['crates/prime-radiant/src/coherence/energy.rs', 'pr-coher'],
    ['crates/prime-radiant/src/coherence/spectral.rs', 'pr-coher'],
    ['crates/prime-radiant/src/coherence/incremental.rs', 'pr-coher'],
    ['crates/prime-radiant/src/coherence/history.rs', 'pr-coher'],
    ['crates/prime-radiant/src/coherence/mod.rs', 'pr-coher'],
    ['crates/prime-radiant/src/governance/repository.rs', 'pr-gov'],
    ['crates/prime-radiant/src/governance/policy.rs', 'pr-gov'],
    ['crates/prime-radiant/src/governance/lineage.rs', 'pr-gov'],
    ['crates/prime-radiant/src/governance/witness.rs', 'pr-gov'],
    ['crates/prime-radiant/src/governance/mod.rs', 'pr-gov'],
    ['crates/prime-radiant/src/storage/postgres.rs', 'pr-store'],
    ['crates/prime-radiant/src/storage/file.rs', 'pr-store'],
    ['crates/prime-radiant/src/storage/memory.rs', 'pr-store'],
    ['crates/prime-radiant/src/storage/mod.rs', 'pr-store'],
    ['crates/ruvector-attention/src/transport/centroid_ot.rs', 'rv-transport'],
    ['crates/ruvector-attention/src/transport/sliced_wasserstein.rs', 'rv-transport'],
    ['crates/ruvector-attention/src/transport/cached_projections.rs', 'rv-transport'],
    ['crates/ruvector-attention/src/training/optimizer.rs', 'rv-training'],
    ['crates/ruvector-attention/src/training/loss.rs', 'rv-training'],
    ['crates/ruvector-attention/src/training/curriculum.rs', 'rv-training'],
    ['crates/ruvector-attention/src/training/mining.rs', 'rv-training'],
  ],
};

// Cross-domain mapping
const domainMap = {
  'ruvllm-cf': [3, 11],  // memory-and-learning, claude-flow-cli
  'ruvllm-rb': [3],       // memory-and-learning
  'ruvllm-ctx': [3],      // memory-and-learning
  'ruvllm-qual': [3],     // memory-and-learning
  'ruvllm-lora': [3],     // memory-and-learning
  'ruvllm-rv': [9],       // ruvector
  'sheaf': [9],           // ruvector
  'ruvllm-serve': [4],    // agent-lifecycle
  'pr-cohom': [3],        // memory-and-learning
  'pr-coher': [3],        // memory-and-learning
  'pr-gov': [14],         // production-infra
  'pr-store': [7],        // agentdb-integration
  'rv-transport': [9],    // ruvector
  'rv-training': [3],     // memory-and-learning
};

const addTag = db.prepare('INSERT OR IGNORE INTO file_domains (file_id, domain_id) VALUES (?, ?)');
const addDep = db.prepare('INSERT OR IGNORE INTO dependencies (source_file_id, target_file_id, relationship, evidence) VALUES (?, ?, ?, ?)');

let tagged = 0;
let notFound = [];
let depsAdded = 0;

for (const [tier, files] of Object.entries(modules)) {
  for (const [path, group] of files) {
    const file = db.prepare('SELECT id FROM files WHERE relative_path = ?').get(path);
    if (!file) {
      notFound.push(path);
      continue;
    }

    // Tag with v4-gold-sweep
    addTag.run(file.id, sweepDomain.id);

    // Cross-tag with existing domains
    const crossDomains = domainMap[group] || [];
    for (const dId of crossDomains) {
      addTag.run(file.id, dId);
    }
    tagged++;

    // Add dependency edges from DEEP siblings in same directory
    const dir = path.substring(0, path.lastIndexOf('/') + 1);
    const deepSiblings = db.prepare(`
      SELECT id, relative_path FROM files
      WHERE relative_path LIKE ? || '%'
        AND relative_path NOT LIKE ? || '%/%'
        AND depth = 'DEEP'
        AND id != ?
    `).all(dir, dir, file.id);

    for (const sib of deepSiblings) {
      addDep.run(sib.id, file.id, 'module-sibling', `Same module: ${dir} [v4-gold-sweep ${tier}]`);
      depsAdded++;
    }
  }
}

console.log(`\nTagged ${tagged} files (${notFound.length} not found)`);
if (notFound.length > 0) {
  console.log('Not found:', notFound);
}
console.log(`Added ${depsAdded} dependency edges`);

// Verify priority queue
const gaps = db.prepare(`
  SELECT tier, tier_rank, COUNT(*) as cnt, SUM(loc) as total_loc
  FROM smart_priority_gaps
  GROUP BY tier, tier_rank
  ORDER BY tier_rank
`).all();
console.log('\n=== SMART PRIORITY GAPS NOW ===');
console.log(JSON.stringify(gaps, null, 2));

const topGaps = db.prepare('SELECT relative_path, loc, tier FROM smart_priority_gaps ORDER BY tier_rank, loc DESC LIMIT 15').all();
console.log('\nTop 15:');
topGaps.forEach(g => console.log(`  [${g.tier}] ${g.loc} LOC  ${g.relative_path}`));

db.close();
