const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const sessionId = 8; // R8-deepen-high
const today = '2026-02-14';

const findFile = db.prepare('SELECT id FROM files WHERE package_id = (SELECT id FROM packages WHERE name = ?) AND relative_path = ?');
const insertRead = db.prepare('INSERT INTO file_reads (file_id, session_id, lines_read, depth) VALUES (?, ?, ?, ?)');
const updateFile = db.prepare('UPDATE files SET depth = ?, lines_read = lines_read + ?, last_read_date = ? WHERE id = ?');
const insertFinding = db.prepare('INSERT INTO findings (file_id, session_id, severity, category, description, line_start, line_end) VALUES (?, ?, ?, ?, ?, ?, ?)');

let filesMatched = 0, findingsInserted = 0;

function record(pkg, path, linesRead, depth, findings) {
  const row = findFile.get(pkg, path);
  if (!row) { console.log('  MISS:', pkg, path); return; }
  insertRead.run(row.id, sessionId, linesRead, depth);
  updateFile.run(depth, linesRead, today, row.id);
  filesMatched++;
  for (const f of findings) {
    let ls = null, le = null;
    if (f.line_ref) {
      const m = f.line_ref.match(/L(\d+)/);
      if (m) ls = parseInt(m[1]);
      const m2 = f.line_ref.match(/L\d+-(\d+)/);
      if (m2) le = parseInt(m2[1]);
    }
    insertFinding.run(row.id, sessionId, f.severity, f.category, f.description + (f.evidence ? ' | ' + f.evidence : ''), ls, le);
    findingsInserted++;
  }
}

const tx = db.transaction(() => {

  // === MODEL-ROUTING (8 files) ===

  record('agentdb', 'src/services/LLMRouter.ts', 660, 'DEEP', [
    { severity: 'HIGH', category: 'architecture', description: 'LLMRouter is independent of hook-based routing - two disconnected routing systems confirmed at provider level', evidence: 'No references to ADR-021 or hook system', line_ref: 'L1-660' },
    { severity: 'MEDIUM', category: 'misleading', description: 'generateLocalFallback() returns hardcoded template strings, not real LLM inference', evidence: 'Lines 568-583: canned responses about voter preferences, market analysis', line_ref: 'L568-583' },
    { severity: 'MEDIUM', category: 'bug', description: 'ruvllmAvailable race condition - checked in selectDefaultProvider() before initialize() sets it', evidence: 'Line 147 set false, line 188 set in init, line 139 checked in selectDefaultProvider', line_ref: 'L139-188' },
    { severity: 'MEDIUM', category: 'stub', description: 'RuvLLM feedback hardcoded to quality=0.8 instead of actual feedback', evidence: 'engine.feedback({ quality: 0.8, useful: true })', line_ref: 'L318-323' },
    { severity: 'INFO', category: 'maintenance', description: 'Anthropic pricing hardcoded at $0.000003/$0.000015 per token - will drift', evidence: 'Lines 553-556', line_ref: 'L553-556' },
  ]);

  record('agentic-flow', 'validation/test-provider-fallback.ts', 286, 'DEEP', [
    { severity: 'MEDIUM', category: 'testing', description: 'Test suite uses fallback test-key strings masking real auth failures', evidence: 'Lines 20,29: process.env.KEY || "test-key"', line_ref: 'L20-29' },
    { severity: 'MEDIUM', category: 'testing', description: 'Budget overflow test missing - sets $1.00 budget but never tests exceeding it', evidence: 'Line 212: costBudget: 1.00 but no overflow test', line_ref: 'L212' },
  ]);

  record('agentic-flow', 'scripts/test-router-docker.sh', 106, 'DEEP', [
    { severity: 'INFO', category: 'portability', description: 'Hardcoded Docker volume paths - only works in specific container setup', evidence: '/workspaces/flow-cloud/docker/claude-agent-sdk/dist/router/router.js', line_ref: 'L25' },
  ]);

  record('claude-flow-cli', '.claude/helpers/router.js', 67, 'DEEP', [
    { severity: 'MEDIUM', category: 'misleading', description: 'router.js routes to AGENT TYPES (coder/tester/reviewer), NOT LLM models - naming misleading', evidence: 'Pattern matching assigns agent types, not haiku/sonnet/opus', line_ref: 'L20-27' },
    { severity: 'INFO', category: 'quality', description: 'All pattern matches return hardcoded confidence 0.8 with no learning', evidence: 'Line 41: confidence: 0.8', line_ref: 'L41' },
  ]);

  record('claude-config', 'helpers/router.js', 67, 'DEEP', [
    { severity: 'INFO', category: 'duplication', description: 'Identical copy of claude-flow-cli router.js - unclear which is source of truth', evidence: 'Both files are byte-identical', line_ref: 'L1-67' },
  ]);

  record('agentic-flow', 'scripts/validate-providers.sh', 51, 'DEEP', [
    { severity: 'MEDIUM', category: 'testing', description: 'Provider validation only checks for "Completed" string - weak assertion', evidence: 'grep -q "Completed" with no quality validation', line_ref: 'L18-42' },
  ]);

  record('claude-config', 'helpers/format-routing-directive.sh', 50, 'DEEP', [
    { severity: 'HIGH', category: 'architecture', description: 'format-routing-directive.sh is THE SOURCE of [ROUTING DIRECTIVE] output - confirmed real', evidence: 'Line 43: echo "**[ROUTING DIRECTIVE]** Model: ${MODEL}"', line_ref: 'L43' },
    { severity: 'MEDIUM', category: 'resilience', description: 'Falls back to hardcoded Model=sonnet if jq missing or input invalid', evidence: 'Line 48: hardcoded sonnet fallback', line_ref: 'L47-49' },
  ]);

  record('claude-config', 'hooks/route-wrapper.sh', 23, 'DEEP', [
    { severity: 'HIGH', category: 'architecture', description: 'route-wrapper.sh is intentionally non-blocking - routing failures never prevent Claude response', evidence: 'Line 20-22: || { exit 0 } always succeeds', line_ref: 'L16-22' },
    { severity: 'MEDIUM', category: 'observability', description: 'Routing errors silently swallowed - user never knows if routing failed', evidence: '>/dev/null 2>&1 suppresses all output', line_ref: 'L16' },
  ]);

  // === MEMORY-AND-LEARNING: AgentDB Core ML (4 files) ===

  record('agentdb', 'src/quantization/vector-quantization.ts', 1529, 'DEEP', [
    { severity: 'INFO', category: 'quality', description: 'vector-quantization.ts is PRODUCTION-GRADE: real 8-bit/4-bit scalar quantization, Product Quantization with K-means++, asymmetric distance computation', evidence: 'Lines 200-240 (8-bit), 251-298 (4-bit), 505-709 (PQ), 794-810 (asymmetric)', line_ref: 'L200-810' },
    { severity: 'INFO', category: 'quality', description: 'Memory reductions accurate: 8-bit=4x, 4-bit=8x. Input validation and numerical stability correct', evidence: 'Lines 25-50 bounds, 319 pre-computed inverses, 458-464 reduction claims', line_ref: 'L25-464' },
  ]);

  record('agentdb', 'src/services/enhanced-embeddings.ts', 1436, 'DEEP', [
    { severity: 'HIGH', category: 'degradation', description: 'enhanced-embeddings.ts silently falls back to mock embeddings (hash-based, no semantic meaning) when @xenova/transformers fails', evidence: 'Line 1109: fallback to mockEmbedding(); Lines 1381-1415: Math.sin(seed) * Math.cos(seed*0.5)', line_ref: 'L1105-1415' },
    { severity: 'INFO', category: 'quality', description: 'LRU cache is real O(1) doubly-linked list implementation; semaphore for concurrency is legitimate', evidence: 'Lines 299-472 (LRU), 481-525 (semaphore)', line_ref: 'L299-525' },
  ]);

  record('agentdb', 'src/controllers/LearningSystem.ts', 1288, 'DEEP', [
    { severity: 'CRITICAL', category: 'fabricated', description: 'LearningSystem claims 9 RL algorithms but all reduce to identical tabular Q-value updates. DQN has no neural network. PPO/Actor-Critic/Policy Gradient are indistinguishable', evidence: 'Lines 494-527: algorithm selection is cosmetic. Lines 552-587: updatePolicyIncremental() identical for all', line_ref: 'L494-587' },
    { severity: 'HIGH', category: 'misleading', description: 'Decision Transformer and Model-Based RL are STUB implementations that fall back to average rewards', evidence: 'Lines 509-521, 695-711', line_ref: 'L509-711' },
    { severity: 'INFO', category: 'quality', description: 'Database schema (lines 85-139) and experience recording (lines 1156-1177) are real SQLite operations', evidence: 'learning_experiences and learning_policies tables with actual inserts', line_ref: 'L85-1177' },
  ]);

  record('agentdb', 'src/simd/simd-vector-ops.ts', 1287, 'DEEP', [
    { severity: 'HIGH', category: 'misleading', description: 'simd-vector-ops.ts labeled "SIMD" but uses only scalar 8x loop unrolling (ILP). NO WebAssembly SIMD instructions executed', evidence: 'Lines 4-9 claim WASM SIMD; Lines 218-379: cosineSimilaritySIMD() is pure JS; Lines 135-174: SIMD detection exists but never used', line_ref: 'L4-379' },
    { severity: 'MEDIUM', category: 'misleading', description: 'WASM SIMD detection module creates test bytes (lines 146-161) but never loads actual WASM vector operations', evidence: 'v128.const detection runs but result unused for computation', line_ref: 'L135-174' },
    { severity: 'INFO', category: 'quality', description: 'Buffer pool and 8x loop unrolling are real performance optimizations (ILP, not SIMD)', evidence: 'Lines 266-316 separate accumulators; Lines 822-893 buffer pool', line_ref: 'L266-893' },
  ]);

  // === MEMORY-AND-LEARNING: AgentDB Controllers (3 files) ===

  record('agentdb', 'src/controllers/ReflexionMemory.ts', 1115, 'DEEP', [
    { severity: 'HIGH', category: 'incomplete', description: 'ReflexionMemory breaks arXiv:2303.11366 paper - missing judge/feedback loop. Critique is write-only, never synthesized', evidence: 'Line 28: critique field exists. Lines 76-217: no judge function implementation. Critique user-provided at line 102', line_ref: 'L28-217' },
    { severity: 'MEDIUM', category: 'quality', description: 'GNN enhancement is syntactic sugar - calls opaque backend.enhance() with no local attention computation', evidence: 'Lines 950-1000: enhanceQueryWithGNN() delegates everything', line_ref: 'L950-1000' },
    { severity: 'INFO', category: 'quality', description: 'Storage, retrieval, and episode filtering are real. Cosine similarity correct. Cache invalidation overbroad', evidence: 'Lines 82-83 blanket invalidation; Lines 850-862 correct cosine', line_ref: 'L82-862' },
  ]);

  record('agentdb', 'src/controllers/CausalMemoryGraph.ts', 876, 'DEEP', [
    { severity: 'CRITICAL', category: 'fabricated', description: 'CausalMemoryGraph claims Pearl do-calculus but implements NONE. No d-separation, no backdoor criterion, no instrumental variables. Just observational correlation', evidence: 'Lines 1-17 claim do-calculus. No Rule 1/2/3 implementation. Line 334: uplift = treatmentMean - controlMean (simple A/B)', line_ref: 'L1-376' },
    { severity: 'CRITICAL', category: 'bug', description: 't-distribution CDF is mathematically WRONG. tInverse hardcoded to 1.96 regardless of df. All p-values and confidence intervals are unreliable', evidence: 'Lines 851-855: incorrect tCDF formula. Line 860: return 1.96 (ignores df). For df=5 should be 2.57', line_ref: 'L851-860' },
    { severity: 'MEDIUM', category: 'quality', description: 'Confounder detection is superficial - counts shared sessions, not actual confounder criterion', evidence: 'Lines 761-813: correlation based on session co-occurrence. Line 792: arbitrary sqrt formula', line_ref: 'L761-813' },
  ]);

  record('agentdb', 'src/backends/ruvector/RuVectorBackend.ts', 971, 'DEEP', [
    { severity: 'INFO', category: 'quality', description: 'RuVectorBackend is PRODUCTION-READY: excellent error handling, security validation (path traversal, prototype pollution), proper distance-to-similarity conversion', evidence: 'Lines 46-50 path checks; 858-863 proto pollution; 908-919 correct distance conversion', line_ref: 'L46-919' },
    { severity: 'MEDIUM', category: 'bug', description: 'Search early termination may return fewer than k results. Metadata filtering is post-hoc, may also return <k', evidence: 'Lines 625-673: earlyTermThreshold=0.9999 breaks early. Lines 645-657: post-search filter', line_ref: 'L625-673' },
    { severity: 'INFO', category: 'quality', description: 'Adaptive HNSW parameters (M, ef) scale well: <1K M=8, 1K-100K M=16, >100K M=32', evidence: 'Lines 284-295', line_ref: 'L284-295' },
  ]);

  // === HOOK-PIPELINE: Helper Scripts (5 files) ===

  record('claude-flow-cli', '.claude/helpers/auto-memory-hook.mjs', 351, 'DEEP', [
    { severity: 'INFO', category: 'quality', description: 'auto-memory-hook.mjs is REAL: JsonFileBackend persists to disk via writeFileSync(), package resolution has 3-strategy fallback', evidence: 'Lines 122-126: _persist() writes; Lines 133-156: 3 import strategies', line_ref: 'L41-156' },
    { severity: 'MEDIUM', category: 'resilience', description: 'Depends on @claude-flow/memory exports (AutoMemoryBridge, LearningBridge) that may not exist. YAML parser is fragile', evidence: 'Lines 199-250: doImport() calls bridge methods. Lines 162-193: regex YAML parsing', line_ref: 'L162-250' },
  ]);

  record('claude-flow-cli', '.claude/helpers/hook-handler.cjs', 233, 'DEEP', [
    { severity: 'HIGH', category: 'fabricated', description: 'hook-handler.cjs route handler IGNORES actual routing result and outputs 100% HARDCODED fake tables with Math.random() latency', evidence: 'Lines 67-100: hardcoded output ignoring router.routeTask() result from line 66', line_ref: 'L57-105' },
    { severity: 'MEDIUM', category: 'fabricated', description: 'session-restore handler outputs hardcoded metrics (0 tasks, 0 agents) regardless of actual state', evidence: 'Lines 150-157: hardcoded table', line_ref: 'L135-168' },
    { severity: 'INFO', category: 'architecture', description: 'Central dispatcher for 8 hook events. Module loading (safeRequire) and dispatch are real', evidence: 'Lines 24-43: safeRequire(). Lines 220-232: command dispatch', line_ref: 'L24-232' },
  ]);

  record('claude-flow-cli', '.claude/helpers/intelligence.cjs', 917, 'DEEP', [
    { severity: 'HIGH', category: 'quality', description: 'intelligence.cjs is GENUINELY REAL: implements actual PageRank with power iteration, edge building via trigram Jaccard similarity, and consolidation with confidence decay', evidence: 'Lines 109-157: real PageRank. Lines 161-213: real edge building. Lines 515-653: real consolidation', line_ref: 'L109-653' },
    { severity: 'INFO', category: 'quality', description: 'Context retrieval scores entries via Jaccard, boosts previously matched patterns as implicit feedback', evidence: 'Lines 399-454: getContext(). Lines 431-441: pattern boosting', line_ref: 'L399-454' },
    { severity: 'INFO', category: 'limitation', description: 'Trigram Jaccard is simplistic (no semantic understanding). Stop words hardcoded. No actual neural learning', evidence: 'Lines 30-40: hardcoded stop words', line_ref: 'L30-40' },
  ]);

  record('claude-config', 'helpers/learning-service.mjs', 1145, 'DEEP', [
    { severity: 'HIGH', category: 'quality', description: 'learning-service.mjs is GENUINELY REAL: HNSW index with greedy search, SQLite persistence, pattern promotion (short-term to long-term), ONNX embedding with hash fallback', evidence: 'Lines 171-444: HNSW. Lines 81-165: SQLite schema. Lines 733-782: promotion. Lines 450-564: embeddings', line_ref: 'L81-782' },
    { severity: 'MEDIUM', category: 'degradation', description: 'HNSW is simplified single-layer (not full multi-layer). Hash fallback embeddings are deterministic but not semantic', evidence: 'Lines 537-563: hash-based fallback with Math.sin/cos', line_ref: 'L171-563' },
    { severity: 'INFO', category: 'quality', description: 'Benchmark command is real performance test: stores 5 patterns, runs 100 searches, measures avg/p95 latency', evidence: 'Lines 1073-1110', line_ref: 'L1073-1110' },
  ]);

  record('claude-flow-cli', '.claude/helpers/standard-checkpoint-hooks.sh', 190, 'DEEP', [
    { severity: 'INFO', category: 'quality', description: 'standard-checkpoint-hooks.sh is ALL REAL: creates git stashes, branches, tags, commits, and JSON metadata at lifecycle events', evidence: 'Lines 5-40: pre-edit stash+branch. Lines 43-99: post-edit tag+commit. Lines 129-169: session summary', line_ref: 'L5-169' },
    { severity: 'MEDIUM', category: 'maintenance', description: 'No cleanup of old checkpoints - infinite growth in .claude/checkpoints/ directory', evidence: 'No prune/cleanup function exists', line_ref: '' },
  ]);

});

tx();
console.log(`R8 import complete: ${filesMatched} files matched, ${findingsInserted} findings inserted`);
db.close();
