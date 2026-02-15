const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const sessionId = 10; // R10

const findFile = db.prepare('SELECT id FROM files WHERE package_id = (SELECT id FROM packages WHERE name = ?) AND relative_path = ?');
const insertRead = db.prepare('INSERT INTO file_reads (file_id, session_id, lines_read, depth) VALUES (?, ?, ?, ?)');
const updateFile = db.prepare('UPDATE files SET depth = ?, lines_read = lines_read + ?, last_read_date = ? WHERE id = ?');
const insertFinding = db.prepare('INSERT INTO findings (file_id, session_id, severity, category, description, line_start, line_end) VALUES (?, ?, ?, ?, ?, ?, ?)');

let matched = 0, findings = 0;

function rec(pkg, path, loc, depth, finds) {
  const row = findFile.get(pkg, path);
  if (!row) { console.log('  still miss:', pkg, path); return; }
  insertRead.run(row.id, sessionId, loc, depth);
  updateFile.run(depth, loc, '2026-02-14', row.id);
  matched++;
  for (const f of finds) {
    insertFinding.run(row.id, sessionId, f.sev, f.cat, f.desc, f.ls || null, f.le || null);
    findings++;
  }
}

const tx = db.transaction(() => {
  // Fix agentic-flow paths (need .claude/ prefix)
  rec('agentic-flow', '.claude/agents/swarm/mesh-coordinator.md', 971, 'DEEP', []);
  rec('agentic-flow', '.claude/agents/swarm/hierarchical-coordinator.md', 718, 'DEEP', []);
  rec('agentic-flow', '.claude/agents/swarm/adaptive-coordinator.md', 1133, 'DEEP', []);
  rec('agentic-flow', '.claude/agents/consensus/byzantine-coordinator.md', 71, 'DEEP', []);
  rec('agentic-flow', '.claude/agents/consensus/gossip-coordinator.md', 71, 'DEEP', []);
  rec('agentic-flow', '.claude/agents/consensus/raft-manager.md', 71, 'DEEP', []);
  rec('agentic-flow', '.claude/agents/consensus/performance-benchmarker.md', 859, 'DEEP', []);
  rec('agentic-flow', '.claude/agents/github/release-swarm.md', 573, 'DEEP', []);
  rec('agentic-flow', '.claude/agents/github/swarm-issue.md', 559, 'DEEP', []);
  rec('agentic-flow', '.claude/agents/github/multi-repo-swarm.md', 537, 'DEEP', []);
  rec('agentic-flow', '.claude/agents/github/code-review-swarm.md', 323, 'DEEP', []);
  rec('agentic-flow', '.claude/agents/github/swarm-pr.md', 412, 'DEEP', []);
  rec('agentic-flow', '.claude/agents/testing/tdd-london-swarm.md', 254, 'DEEP', []);
  rec('agentic-flow', '.claude/agents/templates/coordinator-swarm-init.md', 98, 'DEEP', []);

  // Fix claude-config path corrections
  rec('claude-config', 'agents/sublinear/consensus-coordinator.md', 346, 'DEEP', [
    { sev: 'MEDIUM', cat: 'incomplete', desc: 'consensus-coordinator.md: PageRank voting and matrix consensus sound but rely on non-existent sublinear-time-solver MCP tool', ls: 40, le: 153 },
  ]);
  rec('claude-config', 'agents/optimization/topology-optimizer.md', 816, 'DEEP', [
    { sev: 'INFO', cat: 'quality', desc: 'topology-optimizer.md: GA, simulated annealing, METIS-like partitioning. AI-generated novel topologies is vague', ls: 238, le: 729 },
  ]);
  rec('claude-config', 'agents/v3/swarm-memory-manager.md', 165, 'DEEP', [
    { sev: 'MEDIUM', cat: 'incomplete', desc: 'swarm-memory-manager.md: CRDT engine architecture sound. References non-existent MCP functions. Conflict resolution not implemented', ls: 1, le: 165 },
  ]);
  rec('claude-config', 'agents/hive-mind/scout-explorer.md', 251, 'DEEP', [
    { sev: 'MEDIUM', cat: 'incomplete', desc: 'scout-explorer.md: 5 exploration patterns, 3 scouting strategies. Orchestrates tools but does no core work itself', ls: 1, le: 251 },
  ]);

  // Optimization commands that DO exist in DB
  rec('claude-config', 'commands/optimization/auto-topology.md', 62, 'DEEP', [
    { sev: 'INFO', cat: 'quality', desc: 'auto-topology: Automatic topology selection by task complexity analysis', ls: 1, le: 62 },
  ]);
  rec('claude-config', 'commands/optimization/topology-optimize.md', 26, 'DEEP', []);
});

tx();
console.log('Fix complete:', matched, 'additional files,', findings, 'additional findings');
db.close();
