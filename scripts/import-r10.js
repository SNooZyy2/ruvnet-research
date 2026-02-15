const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');

// Create R10 session
const sessionResult = db.prepare("INSERT INTO sessions (name, date, focus) VALUES (?, '2026-02-14', ?)").run('R10-swarm-coverage', 'Broad swarm coverage: coordinator templates, GitHub templates, SKILL.md files, dist/ implementation, command files, hive-mind stubs');
const sessionId = sessionResult.lastInsertRowid;
console.log('Session ID:', sessionId);

const findFile = db.prepare('SELECT id FROM files WHERE package_id = (SELECT id FROM packages WHERE name = ?) AND relative_path = ?');
const insertRead = db.prepare('INSERT INTO file_reads (file_id, session_id, lines_read, depth) VALUES (?, ?, ?, ?)');
const updateFile = db.prepare('UPDATE files SET depth = ?, lines_read = lines_read + ?, last_read_date = ? WHERE id = ?');
const insertFinding = db.prepare('INSERT INTO findings (file_id, session_id, severity, category, description, line_start, line_end) VALUES (?, ?, ?, ?, ?, ?, ?)');

let filesMatched = 0, findingsInserted = 0, misses = [];

function record(pkg, path, linesRead, depth, findings) {
  const row = findFile.get(pkg, path);
  if (!row) { misses.push(`${pkg}:${path}`); return; }
  insertRead.run(row.id, sessionId, linesRead, depth);
  updateFile.run(depth, linesRead, '2026-02-14', row.id);
  filesMatched++;
  for (const f of findings) {
    insertFinding.run(row.id, sessionId, f.sev, f.cat, f.desc, f.ls || null, f.le || null);
    findingsInserted++;
  }
}

// Try recording in multiple packages (many files are copies)
function recordMulti(pkgs, path, linesRead, depth, findings) {
  for (const pkg of pkgs) {
    record(pkg, path, linesRead, depth, findings);
  }
}

const tx = db.transaction(() => {

  // ============================================================
  // AGENT 1: Coordinator Templates (claude-config + agentic-flow)
  // ============================================================

  const coordPkgs = ['claude-config', 'agentic-flow'];

  recordMulti(coordPkgs, 'agents/swarm/mesh-coordinator.md', 971, 'DEEP', [
    { sev: 'INFO', cat: 'quality', desc: 'mesh-coordinator.md: EXCELLENT. Real algorithms: gossip with fanout, work-stealing with load detection, auction-based task assignment, GraphRoPE topology-aware position embeddings, BFS shortest-path, Byzantine detection via attention outliers', ls: 82, le: 522 },
    { sev: 'INFO', cat: 'quality', desc: 'Multi-head attention for peer coordination correctly implemented. Betweenness centrality computation proper', ls: 199, le: 482 },
  ]);

  recordMulti(coordPkgs, 'agents/swarm/hierarchical-coordinator.md', 718, 'DEEP', [
    { sev: 'INFO', cat: 'quality', desc: 'hierarchical-coordinator.md: EXCELLENT. Hyperbolic attention for hierarchies, GraphRoPE with depth/sibling encoding, weighted consensus generation. Queen 1.5x influence multiplier', ls: 99, le: 398 },
  ]);

  recordMulti(coordPkgs, 'agents/swarm/adaptive-coordinator.md', 1133, 'DEEP', []); // Already covered in R9

  recordMulti(coordPkgs, 'agents/consensus/consensus-coordinator.md', 346, 'DEEP', [
    { sev: 'MEDIUM', cat: 'incomplete', desc: 'consensus-coordinator.md: PageRank voting and matrix consensus are conceptually sound but heavily rely on non-existent sublinear-time-solver MCP tool', ls: 40, le: 153 },
  ]);

  recordMulti(coordPkgs, 'agents/consensus/performance-benchmarker.md', 859, 'DEEP', [
    { sev: 'INFO', cat: 'quality', desc: 'performance-benchmarker.md: EXCELLENT. Real benchmarking: throughput ramp with adaptive rate, latency percentiles (p50-p99.99), phase breakdown, CPU/memory/network profiling, bottleneck detection, adaptive optimizer with rollback', ls: 154, le: 798 },
  ]);

  recordMulti(coordPkgs, 'agents/consensus/byzantine-coordinator.md', 71, 'DEEP', [
    { sev: 'MEDIUM', cat: 'stub', desc: 'byzantine-coordinator.md: STUB (71 LOC). Mentions PBFT and threshold signatures but provides NO implementation or algorithm detail. Insufficient to guide implementation', ls: 33, le: 71 },
  ]);

  recordMulti(coordPkgs, 'agents/consensus/gossip-coordinator.md', 71, 'DEEP', [
    { sev: 'MEDIUM', cat: 'stub', desc: 'gossip-coordinator.md: STUB (71 LOC). Lists push/pull gossip, anti-entropy, Merkle trees, vector clocks but zero algorithmic detail or code', ls: 33, le: 71 },
  ]);

  recordMulti(coordPkgs, 'agents/consensus/raft-manager.md', 71, 'DEEP', [
    { sev: 'MEDIUM', cat: 'stub', desc: 'raft-manager.md: STUB (71 LOC). Mentions leader election, log replication, snapshotting but no timing, pseudocode, or substantive guidance', ls: 33, le: 71 },
  ]);

  // Topology optimizer
  record('claude-config', 'agents/swarm/topology-optimizer.md', 816, 'DEEP', [
    { sev: 'INFO', cat: 'quality', desc: 'topology-optimizer.md: Good. GA (pop=100, mutation=0.1, crossover=0.8, 500 gen), simulated annealing with add/remove/modify/relocate neighbors, METIS-like graph partitioning. "AI-generated novel topologies" is vague', ls: 238, le: 729 },
  ]);

  // ============================================================
  // AGENT 2: GitHub Swarm Agent Templates
  // ============================================================

  const ghPkgs = ['claude-config', 'agentic-flow'];

  recordMulti(ghPkgs, 'agents/github/release-swarm.md', 573, 'DEEP', [
    { sev: 'MEDIUM', cat: 'quality', desc: 'release-swarm.md: Multi-stage release orchestration (changelog, versioning, builds, deploy, rollback). gh CLI mostly correct. 3 issues: date filtering fragile, gh repo clone non-standard, references non-existent claude-flow hook flags', ls: 46, le: 573 },
  ]);

  recordMulti(ghPkgs, 'agents/github/swarm-issue.md', 559, 'DEEP', [
    { sev: 'INFO', cat: 'quality', desc: 'swarm-issue.md: VERY GOOD. Issue-to-swarm conversion, comment commands, label automation. gh CLI excellent with 1 edge case: date -d GNU-ism fails on macOS (L335). Production-ready issue triage logic', ls: 65, le: 482 },
    { sev: 'MEDIUM', cat: 'portability', desc: 'date -d "30 days ago" is GNU-specific, breaks on macOS BSD date', ls: 335, le: 335 },
  ]);

  recordMulti(ghPkgs, 'agents/github/multi-repo-swarm.md', 537, 'DEEP', [
    { sev: 'MEDIUM', cat: 'quality', desc: 'multi-repo-swarm.md: Cross-repo coordination. Sound dependency discovery but fragile: base64 -d vs -D portability, gh repo clone non-standard syntax. Kafka/webhook sections are architectural suggestions only', ls: 56, le: 296 },
  ]);

  recordMulti(ghPkgs, 'agents/github/code-review-swarm.md', 323, 'DEEP', [
    { sev: 'INFO', cat: 'quality', desc: 'code-review-swarm.md: EXCELLENT reasoning blueprint. Self-learning protocol with ReasoningBank pattern retrieval, GNN-enhanced code search, attention-based consensus scoring. Not a shell script guide—a reasoning template', ls: 35, le: 320 },
  ]);

  recordMulti(ghPkgs, 'agents/github/swarm-pr.md', 412, 'DEEP', [
    { sev: 'INFO', cat: 'quality', desc: 'swarm-pr.md: PR lifecycle management. Label-to-agent mapping, comment commands, topology by complexity. gh CLI good with 1 concern: --auto flag version compatibility. Pre/post hook lifecycle clear', ls: 83, le: 412 },
  ]);

  // ============================================================
  // AGENT 3: SKILL.md Files + Misc Templates
  // ============================================================

  record('claude-config', 'skills/swarm-advanced/SKILL.md', 974, 'DEEP', [
    { sev: 'HIGH', cat: 'incomplete', desc: 'swarm-advanced SKILL: 4 swarm patterns (Research, Dev, Testing, Analysis) with full workflows. ~30% of code references non-existent MCP functions. Aspirational but useful architecture', ls: 1, le: 974 },
  ]);

  record('claude-config', 'skills/hive-mind-advanced/SKILL.md', 713, 'DEEP', [
    { sev: 'INFO', cat: 'quality', desc: 'hive-mind-advanced SKILL: High-quality reference. Queen-led architecture, 3 consensus algorithms, REAL CLI tools documented. Memory types with TTL classification. Benchmarks cited lack sources', ls: 1, le: 713 },
  ]);

  record('claude-config', 'skills/flow-nexus-swarm/SKILL.md', 611, 'DEEP', [
    { sev: 'HIGH', cat: 'incomplete', desc: 'flow-nexus-swarm SKILL: Requires external flow-nexus MCP server that may not exist. ~60% references cloud APIs without proof. MCP functions use mcp__flow-nexus__ prefix (not claude-flow)', ls: 1, le: 611 },
  ]);

  record('claude-config', 'skills/v3-swarm-coordination/SKILL.md', 340, 'DEEP', [
    { sev: 'INFO', cat: 'quality', desc: 'v3-swarm-coordination SKILL: BEST in set. Concrete 15-agent blueprint tied to actual v3 ADRs. Named agents, 4-phase timeline, dependency graph. Measurable KPIs (2.49x-7.47x, 84.8%)', ls: 1, le: 340 },
  ]);

  record('claude-config', 'skills/swarm-orchestration/SKILL.md', 180, 'DEEP', [
    { sev: 'MEDIUM', cat: 'stub', desc: 'swarm-orchestration SKILL: SKELETON (180 LOC). Most sections 3-5 lines placeholder. References agentic-flow@1.5.11+ (outdated). Needs 3-4x expansion', ls: 1, le: 180 },
  ]);

  // Misc agent templates
  recordMulti(['claude-config', 'agentic-flow'], 'agents/github/scout-explorer.md', 251, 'DEEP', [
    { sev: 'MEDIUM', cat: 'incomplete', desc: 'scout-explorer.md: 5 exploration patterns, 3 scouting strategies. Orchestrates other tools (Read, Grep, Glob) but does no core work itself. Memory structures are pseudo-code', ls: 1, le: 251 },
  ]);

  recordMulti(['claude-config', 'agentic-flow'], 'agents/swarm/swarm-memory-manager.md', 165, 'DEEP', [
    { sev: 'MEDIUM', cat: 'incomplete', desc: 'swarm-memory-manager.md: Architecture diagram sound (CRDT engine, backends). References non-existent MCP functions (memory_namespace, memory_sync). Conflict resolution strategies not implemented', ls: 1, le: 165 },
  ]);

  recordMulti(['claude-config', 'agentic-flow'], 'agents/testing/tdd-london-swarm.md', 254, 'DEEP', [
    { sev: 'INFO', cat: 'quality', desc: 'tdd-london-swarm.md: Good testing methodology. Real TypeScript test examples. London School (outside-in, mock-first) well-explained. Swarm coordination uses pseudo-APIs', ls: 37, le: 254 },
  ]);

  recordMulti(['claude-config', 'agentic-flow'], 'agents/templates/coordinator-swarm-init.md', 98, 'DEEP', [
    { sev: 'MEDIUM', cat: 'stub', desc: 'coordinator-swarm-init.md: PLACEHOLDER (98 LOC). No actual code or examples. Usage examples are just quoted descriptions. Needs 5-10x expansion', ls: 1, le: 98 },
  ]);

  // ============================================================
  // AGENT 4: dist/ Implementation Files
  // ============================================================

  record('agentic-flow', 'dist/federation/integrations/supabase-adapter-debug.js', 401, 'DEEP', [
    { sev: 'INFO', cat: 'quality', desc: 'supabase-adapter-debug.js: 95% REAL production-grade. Actual Supabase SDK, real table existence checking, pgvector extension check, real CRUD operations, error handling for PGRST116', ls: 7, le: 384 },
    { sev: 'MEDIUM', cat: 'security', desc: 'Depends on actual Supabase credentials (env vars). pgvector extension required', ls: 24, le: 99 },
  ]);

  record('agentic-flow', 'dist/swarm/transport-router.js', 375, 'DEEP', [
    { sev: 'CRITICAL', cat: 'fabricated', desc: 'transport-router.js QUIC layer FABRICATED: sendViaQuic() sends nothing. QuicClient imported but stubbed. HTTP/2 fallback is solid and uses real http2.connect()', ls: 106, le: 211 },
    { sev: 'HIGH', cat: 'quality', desc: 'HTTP/2 session management real using Node.js built-in http2 module. EMA statistics calculation correct. Health check timer infrastructure functional', ls: 215, le: 358 },
  ]);

  record('agentic-flow', 'dist/sdk/e2b-swarm.js', 366, 'DEEP', [
    { sev: 'INFO', cat: 'quality', desc: 'e2b-swarm.js: 90% REAL. E2B sandbox creation, Python/JS/shell execution, file operations, task batching by priority, capability matching. Requires E2B_API_KEY', ls: 39, le: 355 },
    { sev: 'MEDIUM', cat: 'incomplete', desc: 'Template mapping hardcoded to "base" for all capabilities. Should vary by E2B template type', ls: 231, le: 243 },
  ]);

  record('agentic-flow', 'dist/hooks/swarm-learning-optimizer.js', 351, 'DEEP', [
    { sev: 'CRITICAL', cat: 'fabricated', desc: 'swarm-learning-optimizer.js: Reward calculations INVENTED. Base reward 0.5 unjustified. Speedup predictions (2.5x mesh, 3.5x hierarchical, 4.0x star) have NO empirical grounding', ls: 32, le: 209 },
    { sev: 'HIGH', cat: 'fabricated', desc: 'O(n^2) claim for mesh topology unproven. Default recommendations hardcoded with arbitrary thresholds', ls: 268, le: 339 },
  ]);

  record('agentic-flow', 'dist/cli/commands/swarm.js', 325, 'DEEP', [
    { sev: 'CRITICAL', cat: 'broken', desc: 'swarm.js CLI: Imports createP2PSwarmV2 from p2p-swarm-v2.js which may not exist at this path. All 11 CLI commands will crash on missing backend. GunDB relay coordination mentioned but not implemented', ls: 13, le: 270 },
  ]);

  // ============================================================
  // AGENT 5: Swarm Command Files
  // ============================================================

  // Key command files
  record('claude-config', 'commands/swarm/swarm.md', 88, 'DEEP', [
    { sev: 'INFO', cat: 'quality', desc: 'swarm.md: CRITICAL OPERATIONAL GUIDE. Mandates ALL agents run with run_in_background:true in ONE message. Core directive against polling/status checking', ls: 1, le: 88 },
  ]);

  record('claude-config', 'commands/claude-flow-swarm.md', 206, 'DEEP', [
    { sev: 'INFO', cat: 'quality', desc: 'claude-flow-swarm.md: SUBSTANTIVE main command. 7 swarm strategies (auto, dev, research, analysis, testing, optimization, maintenance), 5 coordination modes, monitoring, memory integration', ls: 1, le: 206 },
  ]);

  // Command stubs
  const cmdStubs = [
    ['commands/swarm/analysis.md', 96],
    ['commands/swarm/development.md', 97],
    ['commands/swarm/maintenance.md', 103],
    ['commands/swarm/optimization.md', 118],
    ['commands/swarm/testing.md', 132],
    ['commands/swarm/research.md', 137],
    ['commands/swarm/examples.md', 169],
  ];
  for (const [path, loc] of cmdStubs) {
    record('claude-config', path, loc, 'DEEP', [
      { sev: 'INFO', cat: 'stub', desc: `${path.split('/').pop()}: Stub template defining swarm pattern with agent roles and strategy. Low implementation depth`, ls: 1, le: loc },
    ]);
  }

  // GitHub command files (agents/ versions already covered; these are commands/ copies)
  record('claude-config', 'commands/github/swarm-pr.md', 285, 'DEEP', []);
  record('claude-config', 'commands/github/github-swarm.md', 122, 'DEEP', [
    { sev: 'INFO', cat: 'stub', desc: 'github-swarm.md: Generic GitHub swarm stub. 5 agent types (triager, reviewer, documenter, tester, security), 3 workflows', ls: 1, le: 122 },
  ]);
  record('claude-config', 'commands/github/swarm-issue.md', 482, 'DEEP', []);
  record('claude-config', 'commands/github/code-review-swarm.md', 514, 'DEEP', []);
  record('claude-config', 'commands/github/multi-repo-swarm.md', 519, 'DEEP', []);
  record('claude-config', 'commands/github/release-swarm.md', 544, 'DEEP', []);

  // ============================================================
  // AGENT 6: Hive-Mind + Misc Files
  // ============================================================

  // Hive-mind stubs
  const hiveMindStubs = [
    'commands/hive-mind/hive-mind.md',
    'commands/hive-mind/hive-mind-init.md',
    'commands/hive-mind/hive-mind-consensus.md',
    'commands/hive-mind/hive-mind-memory.md',
    'commands/hive-mind/hive-mind-metrics.md',
    'commands/hive-mind/hive-mind-resume.md',
    'commands/hive-mind/hive-mind-sessions.md',
    'commands/hive-mind/hive-mind-status.md',
    'commands/hive-mind/hive-mind-stop.md',
    'commands/hive-mind/hive-mind-wizard.md',
    'commands/hive-mind/README.md',
  ];
  for (const path of hiveMindStubs) {
    const loc = path.includes('hive-mind.md') && !path.includes('-') ? 28 : (path === 'commands/hive-mind/hive-mind-init.md' ? 19 : (path === 'commands/hive-mind/README.md' ? 18 : 9));
    record('claude-config', path, loc, 'DEEP', [
      { sev: 'INFO', cat: 'stub', desc: `${path.split('/').pop()}: Hive-mind command stub/placeholder. 8 of 11 subcommands are 9-LOC placeholders with no implementation`, ls: 1, le: loc },
    ]);
  }

  // Flow-nexus swarm files
  record('claude-config', 'commands/flow-nexus/swarm.md', 87, 'DEEP', [
    { sev: 'INFO', cat: 'quality', desc: 'flow-nexus swarm command: MCP API for cloud swarm deploy with init, spawn, orchestrate, monitor', ls: 1, le: 87 },
  ]);
  record('claude-config', 'agents/flow-nexus/swarm.md', 84, 'DEEP', [
    { sev: 'INFO', cat: 'quality', desc: 'flow-nexus swarm agent: 4 topologies (hierarchical, mesh, ring, star), 5 specialist types', ls: 1, le: 84 },
  ]);

  // Coordination docs
  record('claude-config', 'commands/coordination/init.md', 45, 'DEEP', [
    { sev: 'INFO', cat: 'quality', desc: 'Coordination init: Clarifies swarm-init tool does NOT write code—creates coordination patterns for Claude Code to follow', ls: 1, le: 45 },
  ]);
  record('claude-config', 'commands/coordination/orchestrate.md', 44, 'DEEP', []);
  record('claude-config', 'commands/coordination/task-orchestrate.md', 26, 'DEEP', []);
  record('claude-config', 'commands/coordination/README.md', 10, 'DEEP', []);
  record('claude-config', 'commands/coordination/agent-coordination.md', 29, 'DEEP', []);

  // Monitoring/optimization commands
  record('claude-config', 'commands/monitoring/swarm-monitor.md', 26, 'DEEP', []);
  record('claude-config', 'commands/optimization/auto-topology.md', 62, 'DEEP', [
    { sev: 'INFO', cat: 'quality', desc: 'auto-topology: Automatic topology selection analyzing task complexity to choose star/mesh/hierarchical/ring', ls: 1, le: 62 },
  ]);
  record('claude-config', 'commands/optimization/topology-optimize.md', 26, 'DEEP', []);
  record('claude-config', 'commands/swarm/swarm-init.md', 86, 'DEEP', []);

  // SPARC swarm coordinator
  record('claude-config', 'commands/sparc/swarm-coordinator.md', 55, 'DEEP', [
    { sev: 'INFO', cat: 'stub', desc: 'SPARC swarm-coordinator: Mode activation via MCP tool or CLI. Lists capabilities but no implementation detail', ls: 1, le: 55 },
  ]);

  // Agentic-flow README files
  record('agentic-flow', '.claude/agents/swarm/README.md', 190, 'DEEP', [
    { sev: 'INFO', cat: 'quality', desc: 'Swarm agents README: Comprehensive guide. 3 topologies compared (hierarchical/mesh/adaptive), MCP tool integration, architecture decision framework, performance characteristics', ls: 1, le: 190 },
  ]);
  record('agentic-flow', '.claude/agents/consensus/README.md', 253, 'DEEP', [
    { sev: 'INFO', cat: 'quality', desc: 'Consensus agents README: 7 agents documented (Byzantine, Raft, Gossip, Security, CRDT, Benchmarker, Quorum). Usage patterns, security, scalability, testing considerations', ls: 1, le: 253 },
  ]);

});

tx();

console.log('\nR10 import complete:');
console.log(`  Files matched: ${filesMatched}`);
console.log(`  Findings inserted: ${findingsInserted}`);
console.log(`  Misses: ${misses.length}`);
if (misses.length > 0) {
  console.log('  Missed files:');
  for (const m of misses) console.log(`    ${m}`);
}
db.close();
