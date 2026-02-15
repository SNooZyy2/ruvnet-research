#!/usr/bin/env node
const db = require("better-sqlite3")("/home/snoozyy/ruvnet-research/db/research.db");
const SESSION_ID = 7;
const today = "2026-02-14";

const updateDepth = db.prepare("UPDATE files SET depth = ?, lines_read = COALESCE(lines_read, 0) + ?, last_read_date = ? WHERE id = ?");
const insertRead = db.prepare("INSERT INTO file_reads (file_id, session_id, depth, lines_read) VALUES (?, ?, ?, ?)");
const insertFinding = db.prepare("INSERT INTO findings (file_id, session_id, severity, category, description) VALUES (?, ?, ?, ?, ?)");

function findFile(pkg, path) {
  return db.prepare("SELECT f.id FROM files f JOIN packages p ON f.package_id = p.id WHERE p.name = ? AND f.relative_path = ?").get(pkg, path);
}

let stats = { matched: 0, notFound: 0, findings: 0 };

function set(pkg, path, depth, lines) {
  const f = findFile(pkg, path);
  if (!f) { stats.notFound++; return null; }
  updateDepth.run(depth, lines, today, f.id);
  insertRead.run(f.id, SESSION_ID, depth, lines);
  stats.matched++;
  return f.id;
}

function finding(fid, sev, cat, desc) {
  if (!fid) return;
  insertFinding.run(fid, SESSION_ID, sev, cat, desc);
  stats.findings++;
}

const importAll = db.transaction(() => {
  let fid;

  // AGENT 1: process.js + daemon.js
  fid = set("claude-flow-cli", "dist/src/commands/process.js", "DEEP", 641);
  finding(fid, "CRITICAL", "fabricated-data", "All monitoring metrics use Math.random() for CPU, memory, agent counts, vectors, network (L244-272)");
  finding(fid, "CRITICAL", "stub", "Worker list is hardcoded static array with fake Jan 2024 timestamps (L383-388)");
  finding(fid, "HIGH", "stub", "Daemon start only writes PID file, no actual daemon process spawned (L111-129)");
  finding(fid, "HIGH", "fabricated-data", "Log viewing generates random fake entries from predefined arrays (L562-593)");
  finding(fid, "HIGH", "stub", "Worker spawn prints success but does not actually spawn processes (L408-412)");

  fid = set("claude-flow-cli", "dist/src/commands/daemon.js", "DEEP", 593);
  finding(fid, "HIGH", "positive-security", "Production-grade path validation: null byte, shell metachar, traversal prevention (L133-152)");
  finding(fid, "HIGH", "positive-security", "Secure process spawning uses argument array not shell string (L184-192)");
  finding(fid, "MEDIUM", "positive", "Real service integration: worker-daemon.js getDaemon/startDaemon/stopDaemon");
  finding(fid, "INFO", "architecture", "12 worker types: map(5m), audit(10m), optimize(15m), consolidate(30m), testgaps(20m), predict(2m), document(60m) + manual");

  // AGENT 2: headless + agentBooster
  fid = set("@claude-flow/guidance", "dist/headless.js", "DEEP", 342);
  finding(fid, "CRITICAL", "reliability", "No validation claude binary exists or supports --output-format json before executing (L164-168)");
  finding(fid, "HIGH", "security", "Command injection vulnerability in parseCommand() regex fallback (L44-53)");
  finding(fid, "MEDIUM", "testing", "Compliance suite has only 3 tests with brittle string-literal assertions (L282-335)");

  fid = set("claude-flow-cli", "dist/src/runtime/headless.js", "DEEP", 284);
  finding(fid, "CRITICAL", "fabricated-data", "Benchmark claims based on synthetic random vectors, not real workloads (L158-187)");
  finding(fid, "HIGH", "architecture-gap", "All workers hardcoded to model=sonnet contradicting ADR-008 3-tier routing (L107-109)");

  fid = set("agentic-flow", "dist/utils/agentBoosterPreprocessor.js", "DEEP", 271);
  finding(fid, "CRITICAL", "security", "npx --yes agent-booster@0.2.2 auto-downloads unverified code from npm on every execution (L95)");
  finding(fid, "HIGH", "stub", "5 of 7 intent extractors return null, 70% of features non-functional (L225-257)");
  finding(fid, "MEDIUM", "data-loss", "writeFileSync with no backup: agent-booster corruption loses original code (L121)");

  // AGENT 3: shell scripts
  fid = set("claude-config", "helpers/daemon-manager.sh", "DEEP", 253);
  finding(fid, "CRITICAL", "reliability", "References swarm-monitor.sh and metrics-db.mjs without existence checks (L70,92)");
  finding(fid, "HIGH", "reliability", "MCP detection uses ps aux|grep giving false positives (L177-182)");
  set("claude-flow-cli", ".claude/helpers/daemon-manager.sh", "DEEP", 253);

  fid = set("claude-config", "helpers/worker-manager.sh", "DEEP", 206);
  finding(fid, "CRITICAL", "reliability", "All worker output suppressed to /dev/null, failures undetectable (L37)");
  finding(fid, "CRITICAL", "security", "pgrep -f matches ANY process with script name, could kill unrelated processes (L123-129)");
  set("claude-flow-cli", ".claude/helpers/worker-manager.sh", "DEEP", 206);

  fid = set("claude-config", "helpers/perf-worker.sh", "DEEP", 170);
  finding(fid, "CRITICAL", "architecture", "ADR-016 workaround: agentic-flow require() patched to skip main() (L71-74)");
  finding(fid, "HIGH", "stub", "run_deep_benchmark() only checks package.json availability (L124-129)");
  set("claude-flow-cli", ".claude/helpers/perf-worker.sh", "DEEP", 170);

  fid = set("claude-config", "agents/hive-mind/worker-specialist.md", "DEEP", 225);
  finding(fid, "MEDIUM", "aspirational", "JS code examples in markdown template not executable (L25-37)");

  // AGENT 4: spawn commands + billing
  set("claude-flow-cli", ".claude/commands/agents/spawn.md", "DEEP", 141);
  set("claude-flow-cli", ".claude/skills/worker-integration/skill.md", "DEEP", 155);
  set("claude-flow-cli", ".claude/skills/worker-benchmarks/skill.md", "DEEP", 136);
  set("claude-flow-cli", ".claude/commands/coordination/spawn.md", "DEEP", 46);

  fid = set("agentic-flow", "dist/mcp/fastmcp/tools/swarm/spawn.js", "DEEP", 41);
  finding(fid, "HIGH", "positive", "REAL implementation: executes npx claude-flow@alpha agent spawn via execSync (L21-25)");
  finding(fid, "MEDIUM", "discrepancy", "Zod schema allows 5 agent types but docs list 87 (L8-10)");

  fid = set("agentic-flow", "dist/billing/payments/processor.js", "DEEP", 158);
  finding(fid, "CRITICAL", "stub", "Entire payment processing SIMULATED: fake Stripe/PayPal/Crypto IDs, 1% random failure (L36-61)");
  finding(fid, "CRITICAL", "security", "Webhook signature verification accepts ANY non-empty string as valid (L119-122)");

  set("claude-flow-cli", ".claude/commands/agents/agent-spawning.md", "DEEP", 29);
  set("agentic-flow", ".claude/commands/agents/agent-spawning.md", "DEEP", 29);
  set("claude-flow-cli", ".claude/commands/automation/smart-spawn.md", "DEEP", 26);
  set("agentic-flow", ".claude/commands/automation/smart-spawn.md", "DEEP", 26);
  set("claude-config", "commands/automation/smart-spawn.md", "DEEP", 26);
  set("claude-flow-cli", ".claude/commands/coordination/agent-spawn.md", "DEEP", 26);
  set("claude-flow-cli", ".claude/commands/hive-mind/hive-mind-spawn.md", "DEEP", 22);
  set("agentic-flow", ".claude/commands/hive-mind/hive-mind-spawn.md", "DEEP", 22);
  set("claude-flow-cli", ".claude/commands/swarm/swarm-spawn.md", "DEEP", 20);
  set("agentic-flow", ".claude/commands/swarm/swarm-spawn.md", "DEEP", 20);

  // AGENT 5: hooks.js
  fid = set("claude-flow-cli", "dist/src/commands/hooks.js", "DEEP", 2528);
  finding(fid, "CRITICAL", "fabricated-data", "Token stats fabricated: totalTokensSaved += 200, cacheHits = 2, cacheMisses = 1 (L3236-3239)");
  finding(fid, "CRITICAL", "fabricated-data", "Pattern count from DB file size: patterns = Math.floor(sizeKB / 2) (L2651-2656)");
  finding(fid, "CRITICAL", "fabricated-data", "Anti-drift config hardcoded not learned (batchSize:4, topology:hierarchical) (L3202-3214)");
  finding(fid, "HIGH", "fabricated-data", "Domain completion from arbitrary pattern thresholds (L2679-2692)");
  finding(fid, "HIGH", "fabricated-data", "Intelligence % from directory existence checks not real intelligence (L2777-2803)");
  finding(fid, "HIGH", "stub", "Fake progress animations: setTimeout 500-800ms simulating processing (L694-697, L1631-1638)");
  finding(fid, "MEDIUM", "architecture", "Pure CLI presentation layer, ZERO business logic. 35 subcommands via callMCPTool()");
  finding(fid, "INFO", "architecture", "12 workers: ultralearn, optimize, consolidate, predict, audit, map, preload, deepdive, document, refactor, benchmark, testgaps");
});

importAll();

console.log("Files matched:", stats.matched);
console.log("Not found:", stats.notFound);
console.log("Findings:", stats.findings);
db.close();
