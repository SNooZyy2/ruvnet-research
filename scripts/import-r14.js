// R14 Hook Pipeline Deep-Read Import
// Session 14: 33 files, ~40 findings
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const sessionId = 14;
const today = '2026-02-14';

const insertRead = db.prepare('INSERT INTO file_reads (file_id, session_id, lines_read, depth) VALUES (?, ?, ?, ?)');
const updateFile = db.prepare('UPDATE files SET depth = ?, lines_read = lines_read + ?, last_read_date = ? WHERE id = ?');
const insertFinding = db.prepare('INSERT INTO findings (file_id, session_id, severity, category, description, line_start, line_end) VALUES (?, ?, ?, ?, ?, ?, ?)');

let filesOk = 0, findingsOk = 0;

function rec(fileId, linesRead, depth, findings) {
  insertRead.run(fileId, sessionId, linesRead, depth);
  updateFile.run(depth, linesRead, today, fileId);
  filesOk++;
  for (const f of findings) {
    insertFinding.run(fileId, sessionId, f.sev, f.cat, f.desc, f.ls || null, f.le || null);
    findingsOk++;
  }
}

db.transaction(() => {
  // ============================================================
  // AGENT 1: Intelligence Layer (agentic-flow, pkg 3)
  // ============================================================

  // intelligence-bridge.js (1038 LOC) — CRITICAL RESOLUTION: EXISTS AND IS REAL
  rec(819, 1038, 'DEEP', [
    { sev: 'CRITICAL', cat: 'resolution', desc: 'intelligence-bridge.js EXISTS and is REAL (1038 LOC). Previously catalogued as non-existent by model-routing analysis. routeTaskIntelligent() at L382 and findSimilarPatterns() at L542 both exist and work.', ls: 1, le: 1038 },
    { sev: 'HIGH', cat: 'architecture', desc: 'Six-layer intelligence architecture: TensorCompress tiering (L55-204), Multi-Algorithm Learning with 9 RL algorithms (L206-320), SONA+MoE+HNSW routing (L374-431), Trajectory tracking (L441-504), Pattern storage/search (L509-569), Parallel execution with worker pool (L621-1037).', ls: 55, le: 1037 },
    { sev: 'HIGH', cat: 'fabrication', desc: 'HNSW explicitly disabled: enableHnsw: false at L347 with comment "API compatibility issue". 150x faster claims undermined — falls back to brute-force.', ls: 347, le: 347 },
    { sev: 'MEDIUM', cat: 'architecture', desc: 'TensorCompress tiered compression (hot/warm/cool/cold/archive) with 50%-96.9% savings claims. Lazy-loaded from ruvector with graceful fallback to uncompressed if unavailable.', ls: 55, le: 204 },
    { sev: 'MEDIUM', cat: 'architecture', desc: 'Multi-Algorithm Learning routes 9 task types to specialized RL algorithms (double-q, sarsa, actor-critic, ppo, decision-transformer, td-lambda, q-learning, reinforce, a2c). Optional ruvector dependency with fallback.', ls: 206, le: 320 },
    { sev: 'MEDIUM', cat: 'security', desc: 'Shell injection potential in pretrain/worker functions using execSync. Extended worker pool (L824-1037) includes AST analysis, security scanning, git blame/churn operations.', ls: 824, le: 1037 },
  ]);

  // intelligence-tools.js (425 LOC) — REAL MCP tools
  rec(820, 425, 'DEEP', [
    { sev: 'HIGH', cat: 'architecture', desc: '9 MCP intelligence tools registered: route, trajectory-start/step/end, pattern-store/search, stats, learn, attention. All import from intelligence-bridge.js — all 10 imports confirmed to exist.', ls: 12, le: 425 },
    { sev: 'INFO', cat: 'quality', desc: 'All tools use Zod schema validation on parameters with proper error handling and latency tracking.', ls: 17, le: 425 },
  ]);

  // pretrain.js (171 LOC) — REAL repository analysis
  rec(826, 171, 'DEEP', [
    { sev: 'INFO', cat: 'architecture', desc: 'Four-phase pretrain: file structure analysis via git ls-files (L25-56), git history co-edit patterns (L57-100), key file memory creation (L101-124), directory-agent mappings (L125-149). Uses shared.js utilities.', ls: 25, le: 149 },
    { sev: 'MEDIUM', cat: 'security', desc: 'Uses execSync with potential shell injection via git commands. No input sanitization on repository paths.', ls: 25, le: 56 },
  ]);

  // explain.js (164 LOC) — REAL routing explainer
  rec(817, 164, 'DEEP', [
    { sev: 'INFO', cat: 'architecture', desc: 'Five-factor routing explanation: file pattern analysis (L22-52), task keyword matching with 1.5x weight (L54-78), memory similarity with cosine threshold 0.3 (L79-104), error pattern analysis (L105-121), weighted scoring returning top 5 agents (L122-148).', ls: 22, le: 148 },
  ]);

  // ============================================================
  // AGENT 2: Hook Infrastructure (agentic-flow, pkg 3)
  // ============================================================

  // hooks.js CLI (923 LOC) — REAL, production-grade
  rec(749, 923, 'DEEP', [
    { sev: 'HIGH', cat: 'architecture', desc: 'CLI hooks command registers 19 subcommands: 10 hook tools (pre/post-edit, pre/post-command, route, explain, pretrain, build-agents, metrics, transfer) + 9 intelligence commands (route, trajectory-start/step/end, pattern-store/search, stats, learn, attention). Init command generates .claude/settings.json and statusline.mjs (539 LOC embedded).', ls: 38, le: 919 },
    { sev: 'INFO', cat: 'quality', desc: 'Full error handling with try/catch on all commands. JSON output support throughout. Uses Commander.js framework. Zero fabrication detected.', ls: 1, le: 923 },
  ]);

  // hooks-bridge.js SDK (235 LOC) — REAL, production-grade
  rec(913, 235, 'DEEP', [
    { sev: 'HIGH', cat: 'architecture', desc: 'SDK bridge registers 7 hook types with Claude Agent SDK: PreToolUse, PostToolUse, PostToolUseFailure, SessionStart, SessionEnd, SubagentStart, SubagentStop. Manages trajectory lifecycle with TTL-based cleanup (5-min TTL, 2-min cleanup interval). Lazy-loads intelligence-bridge.js.', ls: 22, le: 234 },
    { sev: 'INFO', cat: 'quality', desc: 'Map-based state management for active trajectories. .unref() on cleanup timer prevents blocking. Silent failure on errors (returns empty object).', ls: 22, le: 65 },
  ]);

  // shared.js (159 LOC) — REAL utilities
  rec(828, 159, 'DEEP', [
    { sev: 'HIGH', cat: 'architecture', desc: 'Core hook utilities: 18-type agent mapping by file extension (L50-91), hash-based 64-dim embedding via character frequency (L94-111, deterministic not neural), cosine similarity (L113-126), 10 dangerous command patterns with risk scoring 0-0.9 (L128-157).', ls: 50, le: 157 },
    { sev: 'INFO', cat: 'quality', desc: 'simpleEmbed() is hash-based not ML — same input always produces same output. Suitable for caching. Not semantic understanding.', ls: 94, le: 111 },
  ]);

  // hooks-server.js (63 LOC) — REAL MCP server
  rec(805, 63, 'DEEP', [
    { sev: 'INFO', cat: 'architecture', desc: 'FastMCP stdio server registering all 19 hook tools (10 hooks + 9 intelligence). Standalone server mode for Claude Code MCP integration.', ls: 19, le: 62 },
  ]);

  // index.js (59 LOC) — REAL registry
  rec(818, 59, 'DEEP', [
    { sev: 'INFO', cat: 'architecture', desc: 'Hook tools registry: exports 10 hook tools + 10 intelligence bridge functions + 9 intelligence MCP tools. hookTools array (10) and allHookTools array (19).', ls: 14, le: 58 },
  ]);

  // parallel-validation.js (167 LOC) — REAL validation
  rec(790, 167, 'DEEP', [
    { sev: 'MEDIUM', cat: 'architecture', desc: 'Validates parallel execution with 6 checks: sequential subprocess spawning (-0.3), missing ReasoningBank (-0.2), small batch size (-0.1), no QUIC transport (-0.15), no result synthesis (-0.15), no pattern storage (recommendation only). Grades A-F.', ls: 7, le: 165 },
    { sev: 'INFO', cat: 'quality', desc: 'All checks use regex pattern matching. Deterministic scoring. No fabrication.', ls: 66, le: 98 },
  ]);

  // reasoningbank/hooks/post-task.js (110 LOC) — REAL learning
  rec(870, 110, 'DEEP', [
    { sev: 'INFO', cat: 'architecture', desc: 'ReasoningBank post-task hook: judges trajectory via judgeTrajectory() returning {label, confidence, reasons}, distills memories via distillMemories(), checks consolidation threshold for memory compaction. Three-step learning pipeline.', ls: 71, le: 96 },
  ]);

  // reasoningbank/hooks/pre-task.js (69 LOC) — REAL retrieval
  rec(871, 69, 'DEEP', [
    { sev: 'INFO', cat: 'architecture', desc: 'ReasoningBank pre-task hook: retrieves relevant memories via retrieveMemories(query, options), formats for system prompt injection via formatMemoriesForPrompt(). Supports domain and agent filtering.', ls: 44, le: 60 },
  ]);

  // ============================================================
  // AGENT 3: MCP Hook Tools (agentic-flow, pkg 3)
  // ============================================================

  // build-agents.js (276 LOC) — REAL
  rec(816, 276, 'DEEP', [
    { sev: 'INFO', cat: 'architecture', desc: 'Generates language-specific agent configs (YAML/JSON) from detected patterns. Detects Rust, TS, Python, Go, React, Vue via substring matching. 5 focus modes (quality/speed/security/testing/fullstack) with different temperatures and prompts.', ls: 1, le: 276 },
  ]);

  // transfer.js (151 LOC) — REAL
  rec(829, 151, 'DEEP', [
    { sev: 'INFO', cat: 'architecture', desc: 'Cross-project pattern transfer with 3 modes: replace (score*0.7 penalty), additive (score*0.5 conservative), merge (0.6*current + 0.4*source weighted average). Memory deduplication via 50-char prefix comparison.', ls: 1, le: 151 },
  ]);

  // post-edit.js (146 LOC) — REAL Q-learning
  rec(823, 146, 'DEEP', [
    { sev: 'HIGH', cat: 'architecture', desc: 'Post-edit hook implements Q-learning: Q(s,a) += 0.1 * (reward - Q(s,a)) with reward=1.0 (success) or -0.3 (failure). Integrates with RuVector trajectory tracking via beginTaskTrajectory/recordTrajectoryStep/endTaskTrajectory. Memory limits: 200 memories, 50 error patterns.', ls: 30, le: 146 },
  ]);

  // metrics.js (137 LOC) — REAL dashboard
  rec(821, 137, 'DEEP', [
    { sev: 'INFO', cat: 'architecture', desc: 'Learning dashboard: routing accuracy = successCount/totalCount, learning velocity = secondHalf_rate - firstHalf_rate, health thresholds at 0.7 (healthy), 0.5 (learning), <0.5 (needs-data). Temporal filtering (1h/24h/7d/30d). All stats computed from actual data.', ls: 1, le: 137 },
  ]);

  // pre-edit.js (121 LOC) — REAL context retrieval
  rec(825, 121, 'DEEP', [
    { sev: 'INFO', cat: 'architecture', desc: 'Pre-edit hook: begins trajectory tracking (L36-42), selects agent via 3-tier priority: getAgentForFile() default, Q-table pattern override if score>0, directory pattern override. Memory retrieval via cosine similarity with 0.3 threshold, returns top 3.', ls: 36, le: 121 },
  ]);

  // benchmark.js (110 LOC) — REAL performance test
  rec(815, 110, 'DEEP', [
    { sev: 'INFO', cat: 'architecture', desc: 'Latency benchmark harness: 5 warmup + 100 benchmark iterations, computes P50/P95/P99 percentiles. Targets: pre-edit <20ms, post-edit <50ms, pre-command <10ms, route <30ms, metrics <50ms.', ls: 1, le: 110 },
  ]);

  // post-command.js (91 LOC) — REAL
  rec(822, 91, 'DEEP', [
    { sev: 'INFO', cat: 'architecture', desc: 'Post-command hook: extracts error types via regex (TypeError, ENOENT, EACCES, EPERM), Q-learning update with same formula as post-edit. Stores successful commands with 100-char output snippets. Error deduplication by (errorType, command) tuple.', ls: 35, le: 91 },
  ]);

  // pre-command.js (70 LOC) — REAL
  rec(824, 70, 'DEEP', [
    { sev: 'INFO', cat: 'architecture', desc: 'Pre-command risk assessment: uses 10 dangerous patterns from shared.js (fork bomb, rm -rf /, curl|bash, etc.). Risk scale 0-1: 0.9=blocked, 0.5-0.9=dangerous, 0.3-0.5=caution, <0.3=safe. Suggests safer alternatives.', ls: 1, le: 70 },
  ]);

  // ============================================================
  // AGENT 4: Shell Scripts, Templates, Commands (claude-config pkg 6, guidance pkg 7)
  // ============================================================

  // hooks-automation SKILL.md (1202 LOC) — DOCUMENTATION
  rec(1473, 1202, 'DEEP', [
    { sev: 'MEDIUM', cat: 'documentation', desc: 'Comprehensive hooks automation skill spec: 20+ hook type reference with examples, JSON configuration patterns for PreToolUse/PostToolUse matchers, MCP tool integration pseudo-code. Specification/documentation only, not working implementation. References npx claude-flow hook CLI commands that may not exist.', ls: 1, le: 1202 },
  ]);

  // standard-checkpoint-hooks.sh (190 LOC) — REAL
  rec(1221, 190, 'DEEP', [
    { sev: 'HIGH', cat: 'quality', desc: 'Production-grade git checkpoint system: pre_edit_checkpoint creates branches/stashes (L4-40), post_edit_checkpoint creates tags+metadata JSON (L42-99), task_checkpoint creates commits (L101-126), session_end_checkpoint generates markdown summaries (L128-169). Case statement routing (L172-189).', ls: 4, le: 189 },
    { sev: 'MEDIUM', cat: 'bug', desc: 'Edge case: L81 uses git diff HEAD~1 --stat which assumes prior commit exists. Will fail on first commit in repository.', ls: 81, le: 81 },
  ]);

  // guidance-hooks.sh (109 LOC) — REAL
  rec(1201, 109, 'DEEP', [
    { sev: 'INFO', cat: 'architecture', desc: 'Guidance context router: 4 hook handlers (pre-edit detects security/V3 files L20-36, post-edit logs to edit-history.log L38-42, pre-command detects dangerous commands with ADR-016 shell safety L44-56, route does keyword-based agent routing L58-71) + session-context returns hardcoded V3 dev context L73-98.', ls: 20, le: 98 },
  ]);

  // statusline-hook.sh (23 LOC) — DEPRECATED
  rec(1222, 23, 'DEEP', [
    { sev: 'INFO', cat: 'architecture', desc: 'DEPRECATED per ADR-023. References absent statusline.cjs script. Non-functional.', ls: 1, le: 23 },
  ]);

  // guidance-hook.sh (14 LOC) — REAL thin wrapper
  rec(1200, 14, 'DEEP', [
    { sev: 'INFO', cat: 'architecture', desc: 'Thin delegation wrapper: routes to npx agentic-flow@alpha hooks route. Writes output to .claude-flow/last-guidance.txt for Claude visibility.', ls: 8, le: 12 },
  ]);

  // commands/hooks/session-end.md (119 LOC) — DOCUMENTATION
  rec(1146, 119, 'DEEP', [
    { sev: 'INFO', cat: 'documentation', desc: 'Session end hook command spec: CLI usage, options, JSON output format. Documentation/specification only.', ls: 1, le: 119 },
  ]);

  // commands/hooks/post-edit.md (118 LOC) — DOCUMENTATION
  rec(1142, 118, 'DEEP', [
    { sev: 'INFO', cat: 'documentation', desc: 'Post-edit hook command spec: CLI usage, options, JSON output format. Documentation/specification only.', ls: 1, le: 118 },
  ]);

  // commands/hooks/pre-edit.md (114 LOC) — DOCUMENTATION
  rec(1144, 114, 'DEEP', [
    { sev: 'INFO', cat: 'documentation', desc: 'Pre-edit hook command spec: CLI usage, options, JSON output format. Documentation/specification only.', ls: 1, le: 114 },
  ]);

  // commands/hooks/post-task.md (113 LOC) — DOCUMENTATION
  rec(1143, 113, 'DEEP', [
    { sev: 'INFO', cat: 'documentation', desc: 'Post-task hook command spec. Documentation only.', ls: 1, le: 113 },
  ]);

  // commands/hooks/pre-task.md (112 LOC) — DOCUMENTATION
  rec(1145, 112, 'DEEP', [
    { sev: 'INFO', cat: 'documentation', desc: 'Pre-task hook command spec. Documentation only.', ls: 1, le: 112 },
  ]);

  // commands/hooks/setup.md (103 LOC) — DOCUMENTATION
  rec(1147, 103, 'DEEP', [
    { sev: 'INFO', cat: 'documentation', desc: 'Hook setup/initialization command spec. Creates .claude/settings.json with hook configurations.', ls: 1, le: 103 },
  ]);

  // commands/hooks/overview.md (58 LOC) — DOCUMENTATION
  rec(1141, 58, 'DEEP', [
    { sev: 'INFO', cat: 'documentation', desc: 'Hook events summary: 13 hook types across Pre-Operation, Post-Operation, MCP Integration, and Session categories.', ls: 1, le: 58 },
  ]);

  // @claude-flow/guidance hooks.js (347 LOC) — REAL TypeScript
  rec(16, 347, 'DEEP', [
    { sev: 'HIGH', cat: 'architecture', desc: 'GuidanceHookProvider class registers 5 hooks: PreCommand (Critical priority), PreToolUse (Critical), PreEdit (High), PreTask (Normal), PostTask (Normal). Severity ordering: block > require-confirmation > warn > allow. Converts EnforcementGate decisions to hook results. handlePreTask retrieves guidance shards based on classified intent.', ls: 109, le: 328 },
    { sev: 'INFO', cat: 'quality', desc: 'Production TypeScript compiled to JS. Factory function createGuidanceHooks(). Proper async/await with intent classification.', ls: 27, le: 98 },
  ]);

  // hook-development SKILL.md (713 LOC) — DOCUMENTATION
  rec(1365, 713, 'DEEP', [
    { sev: 'MEDIUM', cat: 'documentation', desc: 'Comprehensive hook development guide: prompt-based hooks (LLM-driven) vs command hooks (deterministic bash). Plugin hooks.json format (wrapped) vs settings format (direct). 9 hook event types. SessionStart can persist env via $CLAUDE_ENV_FILE. Lifecycle limitation: hooks load at session start, require restart to reload (L572-599).', ls: 26, le: 599 },
  ]);

  // hookify writing-rules SKILL.md (375 LOC) — DOCUMENTATION
  rec(1331, 375, 'DEEP', [
    { sev: 'INFO', cat: 'documentation', desc: 'Hookify rule format: YAML frontmatter with name/enabled/event/pattern/action fields. 5 event types (bash, file, stop, prompt, all). 2 action types (warn, block). 6 operators (regex_match, contains, equals, not_contains, starts_with, ends_with). Files stored as .claude/hookify.{name}.local.md.', ls: 17, le: 328 },
  ]);

  // ralph-loop stop-hook.sh (178 LOC) — REAL, production quality
  rec(1413, 178, 'DEEP', [
    { sev: 'HIGH', cat: 'quality', desc: 'Production-grade stop hook: reads state from .claude/ralph-loop.local.md, robust numeric validation (L28-47) with corrupted state cleanup, max iteration checking (L50-55), JSONL transcript parsing for last assistant message (L69-112), promise tag extraction via Perl regex (L119), atomic file update via temp file (L152-156). Excellent error handling.', ls: 5, le: 174 },
  ]);

})();

console.log(`R14 import complete: ${filesOk} files recorded, ${findingsOk} findings inserted`);
db.close();
