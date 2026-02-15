#!/usr/bin/env node
/**
 * Import prior analysis from the 6 analysis directories into the research DB.
 * Maps analyzed files to their DB entries, updates depth, and inserts findings.
 */
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');

// Create import session
const session = db.prepare(
  "INSERT INTO sessions (name, date, focus, agent_count) VALUES (?, date('now'), ?, ?)"
).run('R6-import-prior', 'Import findings from 6 prior analysis documents', 1);
const SESSION_ID = session.lastInsertRowid;
console.log(`Created session ${SESSION_ID}: R6-import-prior`);

// Helper: find file by relative_path pattern across packages
function findFile(pkg, pathPattern) {
  const row = db.prepare(
    `SELECT f.id, f.relative_path, f.loc FROM files f
     JOIN packages p ON f.package_id = p.id
     WHERE p.name = ? AND f.relative_path LIKE ?
     LIMIT 1`
  ).get(pkg, pathPattern);
  return row;
}

// Helper: find file by exact path
function findFileExact(pkg, path) {
  return db.prepare(
    `SELECT f.id, f.relative_path, f.loc FROM files f
     JOIN packages p ON f.package_id = p.id
     WHERE p.name = ? AND f.relative_path = ?`
  ).get(pkg, path);
}

// Helper: update file depth
const updateDepth = db.prepare(
  `UPDATE files SET depth = ?, lines_read = COALESCE(lines_read, 0) + ?, last_read_date = date('now') WHERE id = ?`
);

// Helper: insert file_read record
const insertRead = db.prepare(
  `INSERT INTO file_reads (file_id, session_id, depth, lines_read) VALUES (?, ?, ?, ?)`
);

// Helper: insert finding
const insertFinding = db.prepare(
  `INSERT INTO findings (file_id, session_id, severity, category, description) VALUES (?, ?, ?, ?, ?)`
);

let stats = { matched: 0, notFound: 0, findings: 0, alreadyUpdated: 0 };

function setDepth(pkg, path, depth, linesRead) {
  const file = findFileExact(pkg, path) || findFile(pkg, `%${path}%`);
  if (!file) {
    stats.notFound++;
    return null;
  }
  updateDepth.run(depth, linesRead, file.id);
  insertRead.run(file.id, SESSION_ID, depth, linesRead);
  stats.matched++;
  return file.id;
}

function addFinding(fileId, severity, category, description) {
  if (!fileId) return;
  insertFinding.run(fileId, SESSION_ID, severity, category, description);
  stats.findings++;
}

// Wrap in transaction
const importAll = db.transaction(() => {

  // ============================================================
  // 1. SWARM FUNCTIONALITY ANALYSIS
  //    Source: swarm-functionality-analysis/swarm-functionality-analysis.md
  // ============================================================
  console.log('\n--- Importing: swarm-functionality-analysis ---');

  // CLI Commands (claude-flow-cli package)
  setDepth('claude-flow-cli', 'dist/src/commands/swarm.js', 'DEEP', 748);
  setDepth('claude-flow-cli', 'dist/src/commands/agent.js', 'MEDIUM', 200);
  setDepth('claude-flow-cli', 'dist/src/commands/hive-mind.js', 'DEEP', 500);
  setDepth('claude-flow-cli', 'dist/src/commands/hooks.js', 'SURFACE', 280);
  setDepth('claude-flow-cli', 'dist/src/commands/guidance.js', 'MEDIUM', 280);
  setDepth('claude-flow-cli', 'dist/src/commands/plugins.js', 'MEDIUM', 410);
  setDepth('claude-flow-cli', 'dist/src/commands/init.js', 'SURFACE', 60);
  setDepth('claude-flow-cli', 'dist/src/commands/doctor.js', 'SURFACE', 40);

  // MCP Tools (claude-flow-cli)
  setDepth('claude-flow-cli', 'dist/src/mcp-tools/hooks-tools.js', 'DEEP', 2976);
  setDepth('claude-flow-cli', 'dist/src/mcp-tools/swarm-tools.js', 'DEEP', 101);
  setDepth('claude-flow-cli', 'dist/src/mcp-tools/hive-mind-tools.js', 'DEEP', 500);
  setDepth('claude-flow-cli', 'dist/src/mcp-tools/coordination-tools.js', 'DEEP', 486);
  setDepth('claude-flow-cli', 'dist/src/mcp-tools/agent-tools.js', 'DEEP', 400);
  setDepth('claude-flow-cli', 'dist/src/mcp-tools/memory-tools.js', 'MEDIUM', 150);
  setDepth('claude-flow-cli', 'dist/src/mcp-tools/claims-tools.js', 'MEDIUM', 150);
  setDepth('claude-flow-cli', 'dist/src/mcp-tools/agentdb-tools.js', 'DEEP', 685);
  setDepth('claude-flow-cli', 'dist/src/mcp-tools/session-tools.js', 'SURFACE', 30);
  setDepth('claude-flow-cli', 'dist/src/mcp-tools/task-tools.js', 'SURFACE', 30);
  setDepth('claude-flow-cli', 'dist/src/mcp-tools/workflow-tools.js', 'SURFACE', 30);
  setDepth('claude-flow-cli', 'dist/src/mcp-tools/terminal-tools.js', 'SURFACE', 30);
  setDepth('claude-flow-cli', 'dist/src/mcp-tools/daa-tools.js', 'SURFACE', 30);
  setDepth('claude-flow-cli', 'dist/src/mcp-tools/browser-tools.js', 'SURFACE', 30);
  setDepth('claude-flow-cli', 'dist/src/mcp-tools/config-tools.js', 'SURFACE', 30);
  setDepth('claude-flow-cli', 'dist/src/mcp-tools/system-tools.js', 'SURFACE', 30);
  setDepth('claude-flow-cli', 'dist/src/mcp-tools/neural-tools.js', 'SURFACE', 30);
  setDepth('claude-flow-cli', 'dist/src/mcp-tools/performance-tools.js', 'SURFACE', 30);
  setDepth('claude-flow-cli', 'dist/src/mcp-tools/embeddings-tools.js', 'SURFACE', 30);
  setDepth('claude-flow-cli', 'dist/src/mcp-tools/github-tools.js', 'SURFACE', 30);
  setDepth('claude-flow-cli', 'dist/src/mcp-tools/transfer-tools.js', 'SURFACE', 30);
  setDepth('claude-flow-cli', 'dist/src/mcp-tools/security-tools.js', 'SURFACE', 30);
  setDepth('claude-flow-cli', 'dist/src/mcp-tools/analyze-tools.js', 'SURFACE', 30);
  setDepth('claude-flow-cli', 'dist/src/mcp-tools/progress-tools.js', 'SURFACE', 30);

  // Services (claude-flow-cli)
  setDepth('claude-flow-cli', 'dist/src/services/headless-worker-executor.js', 'DEEP', 810);
  setDepth('claude-flow-cli', 'dist/src/services/container-worker-pool.js', 'DEEP', 407);
  setDepth('claude-flow-cli', 'dist/src/services/worker-daemon.js', 'DEEP', 756);
  setDepth('claude-flow-cli', 'dist/src/services/worker-queue.js', 'MEDIUM', 255);

  // Memory
  setDepth('claude-flow-cli', 'dist/src/memory/memory-initializer.js', 'DEEP', 1929);

  // RuVector Intelligence (claude-flow-cli bundled)
  setDepth('claude-flow-cli', 'dist/src/ruvector/model-router.js', 'DEEP', 493);
  setDepth('claude-flow-cli', 'dist/src/ruvector/enhanced-model-router.js', 'DEEP', 300);
  setDepth('claude-flow-cli', 'dist/src/ruvector/flash-attention.js', 'DEEP', 367);
  setDepth('claude-flow-cli', 'dist/src/ruvector/lora-adapter.js', 'MEDIUM', 200);
  setDepth('claude-flow-cli', 'dist/src/ruvector/moe-router.js', 'SURFACE', 50);

  // MCP Infrastructure
  setDepth('claude-flow-cli', 'bin/mcp-server.js', 'MEDIUM', 1000);
  setDepth('claude-flow-cli', 'bin/cli.js', 'SURFACE', 200);
  setDepth('claude-flow-cli', 'dist/src/mcp-client.js', 'DEEP', 235);

  // Helper Scripts (claude-config package)
  setDepth('claude-config', 'helpers/learning-hooks.sh', 'DEEP', 330);
  setDepth('claude-config', 'helpers/swarm-hooks.sh', 'MEDIUM', 200);
  setDepth('claude-config', 'helpers/learning-service.mjs', 'SURFACE', 100);
  // claude-flow-cli copies
  setDepth('claude-flow-cli', '.claude/helpers/learning-hooks.sh', 'DEEP', 330);
  setDepth('claude-flow-cli', '.claude/helpers/swarm-hooks.sh', 'MEDIUM', 200);
  setDepth('claude-flow-cli', '.claude/helpers/learning-service.mjs', 'SURFACE', 100);

  // Agent Templates (agentic-flow)
  setDepth('agentic-flow', '.claude/agents/swarm/adaptive-coordinator.md', 'DEEP', 1133);
  setDepth('agentic-flow', '.claude/agents/swarm/hierarchical-coordinator.md', 'DEEP', 710);
  setDepth('agentic-flow', '.claude/agents/swarm/mesh-coordinator.md', 'DEEP', 963);

  // Configuration
  setDepth('claude-flow-cli', '.claude/settings.json', 'DEEP', 533);

  // --- SWARM FINDINGS ---
  let fid;

  fid = (findFileExact('claude-flow-cli', 'dist/src/mcp-tools/swarm-tools.js') || {}).id;
  addFinding(fid, 'CRITICAL', 'fabricated-data', 'swarm_status returns hardcoded zeros; swarm_health always returns "ok"; no connection to actual agent state');

  fid = (findFileExact('claude-flow-cli', 'dist/src/mcp-tools/coordination-tools.js') || {}).id;
  addFinding(fid, 'CRITICAL', 'fabricated-data', 'coordination metrics use Math.random() — latency, throughput, availability are random numbers');

  fid = (findFileExact('claude-flow-cli', 'dist/src/services/headless-worker-executor.js') || {}).id;
  addFinding(fid, 'HIGH', 'architecture', 'Three real process execution modes: Headless Worker Executor spawns claude CLI, Container Worker Pool manages Docker, Hive-Mind spawns interactive Claude with Byzantine prompt');

  fid = (findFileExact('claude-flow-cli', 'dist/src/mcp-tools/hooks-tools.js') || {}).id;
  addFinding(fid, 'HIGH', 'misleading-marketing', 'Agent Booster marketed as "<1ms WASM" but actually calls npx agent-booster@0.2.2 as subprocess with 5s timeout');

  fid = (findFileExact('claude-flow-cli', 'dist/src/memory/memory-initializer.js') || {}).id;
  addFinding(fid, 'HIGH', 'architecture', 'Real 4-layer learning pipeline: episodic memory via HNSW vectors (384-dim MiniLM-L6-v2), composite skill scoring, circuit breaker after 5 failures');
  addFinding(fid, 'MEDIUM', 'fabricated-data', '--train-patterns flag is display-only with fabricated training values');


  // ============================================================
  // 2. RUVECTOR ANALYSIS
  //    Source: ruvector-analysis/ruvector-analysis.md
  //    Note: ruvector repo files are NOT in the DB (only @ruvector/core npm package is)
  // ============================================================
  console.log('\n--- Importing: ruvector-analysis ---');

  // Only 2 files in @ruvector/core package — mark as analyzed
  setDepth('@ruvector/core', 'index.js', 'MEDIUM', 50);
  setDepth('@ruvector/core', 'index.d.ts', 'MEDIUM', 50);

  // Findings about ruvector (attach to the core package files)
  fid = (findFileExact('@ruvector/core', 'index.js') || {}).id;
  addFinding(fid, 'CRITICAL', 'inflated-metrics', '"2 million lines of Rust" claim inflated 4-5x; actual ~400K-600K LOC. GitHub API shows 365K-438K Rust');
  addFinding(fid, 'CRITICAL', 'unreliable-benchmarks', 'Benchmarks show 100% recall (impossible), 0 MB memory (broken profiler), simulated competitors. 16,400 QPS claimed vs 3,597 measured');
  addFinding(fid, 'HIGH', 'architecture', '76 Rust crates confirmed with 550+ transitive deps. Core HNSW wraps hnsw_rs, not novel. 3 distinct HNSW implementations with varying novelty');
  addFinding(fid, 'HIGH', 'distribution', 'npm ecosystem: 50+ published packages, 80K+ monthly downloads, 15-20 MB native binaries per platform');
  addFinding(fid, 'MEDIUM', 'process', 'AI co-authored (explicit Claude credits); 10.3 commits/day is 6-20x faster than sustainable human dev');


  // ============================================================
  // 3. MODEL ROUTER ANALYSIS
  //    Source: model-router-analysis/model-router-analysis.md
  // ============================================================
  console.log('\n--- Importing: model-router-analysis ---');

  // agentic-flow router files
  setDepth('agentic-flow', 'dist/router/router.js', 'MEDIUM', 200);
  setDepth('agentic-flow', 'dist/cli-proxy.js', 'MEDIUM', 350);

  // Findings
  fid = (findFile('agentic-flow', '%router/router.js') || {}).id;
  addFinding(fid, 'HIGH', 'architecture-gap', 'Two disconnected routing systems: claude-flow 3-tier hooks (advisory text only) and agentic-flow ModelRouter (actual execution). Zero integration between them');
  addFinding(fid, 'MEDIUM', 'incomplete', 'Complexity scoring marked TODO in ModelRouter — selectByCost picks cheapest without measuring actual complexity');

  fid = (findFile('agentic-flow', '%cli-proxy.js') || {}).id;
  addFinding(fid, 'MEDIUM', 'architecture-gap', 'claude-flow hooks model-route output is text-only [TASK_MODEL_RECOMMENDATION: haiku] with no programmatic consumption by routing systems');


  // ============================================================
  // 4. CLAUDE-FLOW ANALYSIS
  //    Source: claude-flow-analysis/claude-flow-analysis.md
  // ============================================================
  console.log('\n--- Importing: claude-flow-analysis ---');

  // These are mostly v3 source paths that may not match DB paths exactly
  // Map to claude-flow-cli dist equivalents
  setDepth('claude-flow-cli', 'dist/src/commands/embeddings.js', 'SURFACE', 80);
  setDepth('claude-flow-cli', 'dist/src/commands/neural.js', 'SURFACE', 80);
  setDepth('claude-flow-cli', 'dist/src/commands/memory.js', 'SURFACE', 80);

  // Helpers already covered in swarm analysis, but add claude-flow-specific ones
  setDepth('claude-flow-cli', '.claude/helpers/statusline.cjs', 'SURFACE', 100);

  // Findings
  fid = (findFile('claude-flow-cli', '%mcp-tools/hooks-tools.js') || {}).id;
  addFinding(fid, 'CRITICAL', 'graceful-degradation', 'MCP tools gracefully degrade to no-ops when optional deps missing — memory_store returns {stored: true} without actually storing');

  fid = (findFile('claude-flow-cli', '%bin/mcp-server.js') || {}).id;
  addFinding(fid, 'CRITICAL', 'distribution', 'npm-published package excludes key modules: memory, neural, hooks, claims, embeddings, swarm, browser, aidefence, mcp server itself');

  fid = (findFile('claude-flow-cli', '%memory/memory-initializer.js') || {}).id;
  addFinding(fid, 'HIGH', 'misleading-marketing', 'HNSW "150x-12,500x faster" is theoretical vs brute-force, not benchmarked. Pure TS implementation significantly slower than C++/Rust');
  addFinding(fid, 'HIGH', 'reliability', 'Without agentdb, agentic-flow, @ruvector/core installed, all learning/memory features are non-functional (silent failure)');

  fid = (findFile('claude-flow-cli', '%ruvector/model-router.js') || {}).id;
  addFinding(fid, 'MEDIUM', 'terminology', '"60+ specialized agents" are markdown prompt templates, not persistent processes');

  // Three separate ReasoningBank implementations
  addFinding(fid, 'CRITICAL', 'fragmentation', 'Three separate ReasoningBank implementations (claude-flow, agentic-flow, agentdb) with zero code sharing');


  // ============================================================
  // 5. AGENTIC-FLOW ANALYSIS
  //    Source: agentic-flow-analysis/agentic-flow-analysis.md
  // ============================================================
  console.log('\n--- Importing: agentic-flow-analysis ---');

  // Core agentic-flow files
  setDepth('agentic-flow', 'dist/index.js', 'MEDIUM', 200);
  setDepth('agentic-flow', 'dist/llm/RuvLLMOrchestrator.js', 'MEDIUM', 300);
  setDepth('agentic-flow', 'dist/mcp/claudeFlowSdkServer.js', 'MEDIUM', 200);

  // ReasoningBank subsystem
  setDepth('agentic-flow', 'dist/reasoningbank/core/retrieve.js', 'DEEP', 87);
  setDepth('agentic-flow', 'dist/reasoningbank/core/judge.js', 'MEDIUM', 80);
  setDepth('agentic-flow', 'dist/reasoningbank/core/distill.js', 'MEDIUM', 80);
  setDepth('agentic-flow', 'dist/reasoningbank/core/consolidate.js', 'MEDIUM', 80);
  setDepth('agentic-flow', 'dist/reasoningbank/utils/pii-scrubber.js', 'DEEP', 100);

  // Intelligence
  setDepth('agentic-flow', 'dist/intelligence/RuVectorIntelligence.js', 'MEDIUM', 200);
  setDepth('agentic-flow', 'dist/intelligence/EmbeddingService.js', 'MEDIUM', 300);

  // Swarm
  setDepth('agentic-flow', 'dist/swarm/p2p-swarm-v2.js', 'MEDIUM', 350);
  setDepth('agentic-flow', 'dist/swarm/quic-coordinator.js', 'MEDIUM', 200);
  setDepth('agentic-flow', 'dist/coordination/attention-coordinator.js', 'MEDIUM', 200);

  // Agent runners
  setDepth('agentic-flow', 'dist/agents/claudeAgent.js', 'DEEP', 335);
  setDepth('agentic-flow', 'dist/agents/claudeAgentDirect.js', 'MEDIUM', 200);
  setDepth('agentic-flow', 'dist/agents/directApiAgent.js', 'MEDIUM', 250);

  // Proxy/routing
  setDepth('agentic-flow', 'dist/proxy/requesty-proxy.js', 'MEDIUM', 100);

  // Findings
  fid = (findFile('agentic-flow', '%mcp/claudeFlowSdkServer.js') || {}).id;
  addFinding(fid, 'CRITICAL', 'inflated-metrics', 'MCP tool count inflated: "213 tools" counts external packages; agentic-flow only defines 9-11 tools that shell out to npx claude-flow@alpha');
  addFinding(fid, 'CRITICAL', 'circular-dependency', 'agentic-flow MCP tools shell out to npx claude-flow@alpha, creating circular import when dependencies resolved');

  fid = (findFile('agentic-flow', '%swarm/quic-coordinator.js') || {}).id;
  addFinding(fid, 'CRITICAL', 'stub-code', 'QUIC transport loads WASM returning {}, Federation Hub returns empty arrays, HTTP/3 proxy returns empty Uint8Array — stubs presented as functional');

  fid = (findFile('agentic-flow', '%reasoningbank/core/retrieve.js') || {}).id;
  addFinding(fid, 'HIGH', 'unused-code', 'Sophisticated ReasoningBank implements real DeepMind algorithms (retrieve, judge, distill, consolidate, MaTTS) but claude-flow never calls judge/distill/consolidate');

  fid = (findFile('agentic-flow', '%intelligence/EmbeddingService.js') || {}).id;
  addFinding(fid, 'HIGH', 'broken-dependency', 'ONNX model download fails (downloadModel is not a function), so hash-based embeddings with no semantic meaning run instead');

  fid = (findFile('agentic-flow', '%router/router.js') || {}).id;
  addFinding(fid, 'HIGH', 'positive', 'Real multi-provider routing (OpenRouter, Gemini, Anthropic) with proper API translation and fallback chains actually works — unique value proposition');


  // ============================================================
  // 6. AGENTDB ANALYSIS
  //    Source: agentdb-analysis/agentdb-analysis.md
  // ============================================================
  console.log('\n--- Importing: agentdb-analysis ---');

  // AgentDB controllers
  setDepth('agentdb', 'src/controllers/ReflexionMemory.js', 'MEDIUM', 400);
  setDepth('agentdb', 'src/controllers/SkillLibrary.js', 'MEDIUM', 350);
  setDepth('agentdb', 'src/controllers/CausalMemoryGraph.js', 'MEDIUM', 300);
  setDepth('agentdb', 'src/controllers/NightlyLearner.js', 'MEDIUM', 265);
  setDepth('agentdb', 'src/controllers/AttentionService.js', 'MEDIUM', 258);
  setDepth('agentdb', 'src/controllers/ExplainableRecall.js', 'MEDIUM', 258);
  setDepth('agentdb', 'src/controllers/ReasoningBank.js', 'MEDIUM', 247);
  setDepth('agentdb', 'src/controllers/CausalRecall.js', 'MEDIUM', 177);
  setDepth('agentdb', 'src/controllers/SelfAttentionController.js', 'MEDIUM', 102);
  setDepth('agentdb', 'src/controllers/CrossAttentionController.js', 'MEDIUM', 162);
  setDepth('agentdb', 'src/controllers/MultiHeadAttentionController.js', 'MEDIUM', 167);
  setDepth('agentdb', 'src/controllers/MemoryController.js', 'MEDIUM', 151);
  setDepth('agentdb', 'src/controllers/HNSWIndex.js', 'MEDIUM', 218);
  setDepth('agentdb', 'src/controllers/LearningSystem.js', 'MEDIUM', 464);
  setDepth('agentdb', 'src/controllers/MMRDiversityRanker.js', 'MEDIUM', 64);
  setDepth('agentdb', 'src/controllers/MetadataFilter.js', 'MEDIUM', 121);
  setDepth('agentdb', 'src/controllers/ContextSynthesizer.js', 'MEDIUM', 103);
  setDepth('agentdb', 'src/controllers/EmbeddingService.js', 'MEDIUM', 70);
  setDepth('agentdb', 'src/controllers/QUICServer.js', 'MEDIUM', 191);
  setDepth('agentdb', 'src/controllers/QUICClient.js', 'MEDIUM', 244);
  setDepth('agentdb', 'src/controllers/WASMVectorSearch.js', 'MEDIUM', 167);
  setDepth('agentdb', 'src/controllers/SyncCoordinator.js', 'MEDIUM', 276);
  setDepth('agentdb', 'src/controllers/EnhancedEmbeddingService.js', 'MEDIUM', 59);

  // Core
  setDepth('agentdb', 'src/core/AgentDB.ts', 'MEDIUM', 92);

  // Wrappers
  setDepth('agentdb', 'wrappers/attention-fallbacks.js', 'MEDIUM', 742);

  // Findings
  fid = (findFile('agentdb', '%core/AgentDB.ts') || {}).id;
  addFinding(fid, 'CRITICAL', 'unused-integration', 'Claude-flow uses NONE of AgentDB functionality despite transitive dependency. AgentDB loads and prints warning but is never called');

  fid = (findFile('agentdb', '%WASMVectorSearch.js') || {}).id;
  addFinding(fid, 'CRITICAL', 'missing-module', 'reasoningbank_wasm.js does not exist; WASMVectorSearch falls back to brute-force JS implementation');

  fid = (findFile('agentdb', '%wrappers/attention-fallbacks.js') || {}).id;
  addFinding(fid, 'CRITICAL', 'broken-dependency', '@ruvector/gnn and @ruvector/attention have broken native APIs requiring 1,484+ lines of JS fallback code');

  fid = (findFile('agentdb', '%QUICServer.js') || {}).id;
  addFinding(fid, 'HIGH', 'misleading-naming', 'QUIC implementation is actually TCP — QUICServer and QUICClient are TCP implementations, not RFC 9000 QUIC protocol');

  fid = (findFile('agentdb', '%LearningSystem.js') || {}).id;
  addFinding(fid, 'HIGH', 'misleading-naming', 'Claims 9 RL algorithms but DQN, PPO, Actor-Critic, DecisionTransformer all reduce to tabular RL with cosmetic naming');

  fid = (findFile('agentdb', '%ReflexionMemory.js') || {}).id;
  addFinding(fid, 'INFO', 'positive', '18 of 23 controllers genuinely implemented with real paper references (Reflexion arXiv:2303.11366, Voyager arXiv:2305.16291, Flash Attention, MMR)');

});

// Execute
importAll();

// Print summary
console.log('\n=== IMPORT SUMMARY ===');
console.log(`Files matched & updated: ${stats.matched}`);
console.log(`Files not found in DB:   ${stats.notFound}`);
console.log(`Findings inserted:       ${stats.findings}`);

// Print new coverage
const coverage = db.prepare(`
  SELECT depth, COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM files), 1) as pct
  FROM files GROUP BY depth ORDER BY
    CASE depth
      WHEN 'DEEP' THEN 1
      WHEN 'MEDIUM' THEN 2
      WHEN 'SURFACE' THEN 3
      WHEN 'MENTIONED' THEN 4
      WHEN 'NOT_TOUCHED' THEN 5
    END
`).all();

console.log('\n=== UPDATED COVERAGE ===');
coverage.forEach(r => console.log(`  ${r.depth}: ${r.count} files (${r.pct}%)`));

const totalFiles = db.prepare('SELECT COUNT(*) as c FROM files').get().c;
const touched = db.prepare("SELECT COUNT(*) as c FROM files WHERE depth != 'NOT_TOUCHED'").get().c;
const totalLoc = db.prepare('SELECT SUM(loc) as s FROM files').get().s;
const touchedLoc = db.prepare("SELECT SUM(loc) as s FROM files WHERE depth != 'NOT_TOUCHED'").get().s;

console.log(`\n=== DISCOVERY PROGRESS ===`);
console.log(`  Files: ${touched}/${totalFiles} (${(touched/totalFiles*100).toFixed(1)}%)`);
console.log(`  LOC:   ${touchedLoc?.toLocaleString() || 0}/${totalLoc?.toLocaleString()} (${((touchedLoc||0)/totalLoc*100).toFixed(1)}%)`);

db.close();
