// R15 Agent-Lifecycle Deep-Read Import
// Session 16 (R15): 31 files, ~50 findings
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const sessionId = 16; // R15 session
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
  // AGENT 1: Core Code (agentic-flow, pkg 3)
  // ============================================================

  // agent-manager.js (452 LOC) — REAL
  rec(747, 452, 'DEEP', [
    { sev: 'HIGH', cat: 'architecture', desc: 'AgentManager: 9 public methods for agent lifecycle. Scans package + local .claude/agents with deduplication (local overrides package). Interactive creation via readline. Frontmatter parsing for .md agent files. Conflict detection between package and local agents.', ls: 1, le: 452 },
    { sev: 'INFO', cat: 'quality', desc: 'All filesystem operations use proper error handling. Deduplication via Map with relative path keys. No external dependencies (Node.js built-ins only).', ls: 45, le: 56 },
  ]);

  // sona-agent-training.js (382 LOC) — REAL with FABRICATED mock embeddings
  rec(922, 382, 'DEEP', [
    { sev: 'HIGH', cat: 'architecture', desc: 'AgentFactory (EventEmitter): creates specialized SONA engines, trains on examples with quality scoring, pattern search via HNSW. CodebaseTrainer: indexes code by chunking functions/classes, creates training data. 6 pre-configured AgentTemplates (codeAssistant, chatBot, dataAnalyst, ragAgent, taskPlanner, domainExpert).', ls: 1, le: 382 },
    { sev: 'HIGH', cat: 'fabrication', desc: 'CodebaseTrainer.mockEmbedding() at L290-297 uses sine-based pseudorandom generation (Math.sin(seed)*10000), NOT real embeddings. Comment: "replace with actual embedding service in production". 3072-dim output is deterministic but meaningless.', ls: 290, le: 297 },
    { sev: 'MEDIUM', cat: 'architecture', desc: 'Depends on @ruvector/sona SonaEngine for learning. Code chunking uses regex for functions/classes with brace-counting block detection.', ls: 234, le: 283 },
  ]);

  // long-running-agent.js (220 LOC) — REAL
  rec(772, 220, 'DEEP', [
    { sev: 'HIGH', cat: 'architecture', desc: 'LongRunningAgent: budget enforcement (cost + runtime limits), periodic checkpointing with state snapshots, provider failover via ProviderManager, graceful shutdown with timer cleanup. Progress = completed/(completed+failed). Three lifecycle phases: start→execute→stop.', ls: 1, le: 220 },
    { sev: 'INFO', cat: 'quality', desc: 'Proper timeout/interval cleanup on stop (L205-217). Error handling with logging. Checkpoint compression. Budget constraint validation before task execution.', ls: 52, le: 63 },
  ]);

  // EphemeralAgent.js (259 LOC) — REAL with architectural placeholders
  rec(777, 259, 'DEEP', [
    { sev: 'HIGH', cat: 'architecture', desc: 'EphemeralAgent: spawn→execute→learn→destroy lifecycle with auto-expiry timer. JWT tenant isolation. Pre/post sync with federation hub. Learning episodes stored with quality threshold 0.7. Agent ID format: eph-{tenantId}-{timestamp}. Default lifetime 300s.', ls: 33, le: 241 },
    { sev: 'MEDIUM', cat: 'architecture', desc: 'FederationHub.connect() is placeholder ("simulate connection with WebSocket fallback"). Memory path defaults to :memory: (in-memory only via better-sqlite3). Abstraction allows future QUIC implementation.', ls: 32, le: 52 },
  ]);

  // agent-converter.js (188 LOC) — REAL
  rec(909, 188, 'DEEP', [
    { sev: 'HIGH', cat: 'architecture', desc: 'Converts agentic-flow agent definitions to Claude SDK format. 8-type inference (researcher→coder→reviewer→tester→doc→architect→coordinator→analyst) via substring matching. Tool assignment per type. Model selection: Opus for complex/architect/security, Haiku for simple/quick, inherit default. 60s TTL cache. 5 essential agents: researcher, coder, reviewer, tester, documenter.', ls: 12, le: 157 },
  ]);

  // agentLoader.js (162 LOC) — REAL
  rec(936, 162, 'DEEP', [
    { sev: 'INFO', cat: 'architecture', desc: 'Agent template loader: recursive scan of .claude/agents (package + local), YAML frontmatter parsing, tool comma-split. Deduplication: local agents override package agents with same relative path. Logs override decisions.', ls: 14, le: 146 },
  ]);

  // ============================================================
  // AGENT 2: MCP Tools & Booster (agentic-flow, pkg 3)
  // ============================================================

  // agent-booster-migration.js (343 LOC) — FABRICATED
  rec(840, 343, 'DEEP', [
    { sev: 'CRITICAL', cat: 'fabrication', desc: 'AgentBoosterMigration performance metrics are FABRICATED. L121: hardcoded traditionalTime=352ms. L198: avgSpeedupFactor=352 (constant). L150: editTraditional() uses sleep(352) as artificial delay. L187: isWasmAvailable() returns hardcoded true. L265: migrateCodebase() returns estimated results without actual migration.', ls: 107, le: 274 },
    { sev: 'HIGH', cat: 'broken-dep', desc: 'Module "agent-booster" does not exist in dependencies (L18). Lazy loading with fallback (L14-24) prevents crash but ALL booster operations fail silently.', ls: 14, le: 24 },
  ]);

  // agent-booster-tools.js (274 LOC) — BROKEN
  rec(836, 274, 'DEEP', [
    { sev: 'CRITICAL', cat: 'broken-dep', desc: 'Three MCP tools (agent_booster_edit_file, batch_edit, parse_markdown) are non-functional. getBooster() at L13 imports missing "agent-booster" module. booster.apply() called without null check at L122 and L172 — runtime crash. Markdown regex parser (L225) is REAL but delegates to broken handler.', ls: 9, le: 248 },
    { sev: 'HIGH', cat: 'fabrication', desc: 'Speedup calculation at L216: Math.round(352/avgLatency) uses hardcoded 352 denominator. Empty error responses returned on failure (L143-152).', ls: 216, le: 216 },
  ]);

  // add-command.js (118 LOC) — REAL
  rec(811, 118, 'DEEP', [
    { sev: 'INFO', cat: 'architecture', desc: 'MCP tool command_add: creates .claude/commands/{name}.md files. Kebab-case validation, markdown template generation, real filesystem write. Fully functional and production-ready.', ls: 1, le: 118 },
  ]);

  // add-agent.js (108 LOC) — REAL
  rec(810, 108, 'DEEP', [
    { sev: 'INFO', cat: 'architecture', desc: 'MCP tool agent_add: creates .claude/agents/{category}/{name}.md files. Kebab-case validation, recursive mkdir, markdown template with frontmatter. Production-ready.', ls: 1, le: 108 },
  ]);

  // list.js (83 LOC) — BROKEN
  rec(813, 83, 'DEEP', [
    { sev: 'HIGH', cat: 'broken-dep', desc: 'MCP tool agent_list: uses execSync("npx agentic-flow --list") at L15-19. CLI wrapper, not MCP-native. Fails if npx or package not in PATH. Output parsing via regex is real but depends on CLI format.', ls: 15, le: 39 },
  ]);

  // parallel.js (64 LOC) — BROKEN
  rec(814, 64, 'DEEP', [
    { sev: 'HIGH', cat: 'broken-dep', desc: 'MCP tool agent_parallel: shell wrapper around execSync("npx agentic-flow") at L26. Claims to run 3 agents but just executes generic command. No actual parallelization logic. Environment variables passed but command ignores input parameters.', ls: 13, le: 62 },
  ]);

  // execute.js (57 LOC) — BROKEN
  rec(812, 57, 'DEEP', [
    { sev: 'HIGH', cat: 'broken-dep', desc: 'MCP tool agent_execute: CLI wrapper using execSync at L16. Constructs "npx agentic-flow --agent X --task Y" — not MCP-native. Fails in isolated environments.', ls: 12, le: 55 },
  ]);

  // claudeFlowAgent.js (116 LOC) — REAL
  rec(732, 116, 'DEEP', [
    { sev: 'HIGH', cat: 'architecture', desc: 'REAL Claude Agent SDK integration. claudeFlowAgent() uses @anthropic-ai/claude-agent-sdk query() method with streaming. Memory context construction, coordination context, stream processing loop. 4 exported agent functions: claudeFlowAgent, memoryResearchAgent, orchestratorAgent, hybridAgent. withRetry() wrapper for resilience.', ls: 10, le: 114 },
  ]);

  // multi-agent-orchestration.js (46 LOC) — REAL example
  rec(774, 46, 'DEEP', [
    { sev: 'INFO', cat: 'architecture', desc: 'Example code: loads agents from .claude/agents/, demonstrates sequential multi-agent workflow using goal-planner and code-analyzer agents. Works if agents available.', ls: 7, le: 44 },
  ]);

  // agentic_flow_quic.js (780 LOC) — REAL WASM bindings
  rec(994, 780, 'DEEP', [
    { sev: 'HIGH', cat: 'architecture', desc: 'REAL wasm-bindgen generated WASM binding layer for QUIC transport. Memory management, UTF-8 string marshalling, closure management with FinalizationRegistry. Exports: createQuicMessage(), defaultConfig(), WasmQuicClient class (sendMessage, recvMessage, poolStats, close). Loads .wasm binary via fs.readFileSync + WebAssembly.Module.', ls: 249, le: 778 },
  ]);

  // ============================================================
  // AGENT 3: V3 Agent Templates
  // ============================================================

  // performance-engineer.md (1234 LOC) — ASPIRATIONAL
  rec(1836, 1234, 'DEEP', [
    { sev: 'HIGH', cat: 'architecture', desc: 'V3 Performance Engineer template: Flash Attention optimization (2.49x-7.47x claimed), WASM SIMD acceleration, token usage optimization (50-75% claimed), SONA integration, HNSW optimization, quantization analysis. Pre/post hooks with memory store and baseline CPU collection. Opus-level agent with 21 capabilities listed.', ls: 1, le: 50 },
    { sev: 'MEDIUM', cat: 'fabrication', desc: 'Performance claims (2.49x-7.47x Flash Attention, 150x-12500x HNSW, 50-75% memory reduction) are aspirational design targets stated as facts. No benchmarks provided. References MCP tools that may not fully implement claimed features.', ls: 22, le: 27 },
  ]);

  // memory-specialist.md (996 LOC) — ASPIRATIONAL
  rec(1835, 996, 'DEEP', [
    { sev: 'HIGH', cat: 'architecture', desc: 'V3 Memory Specialist: HNSW indexing, hybrid SQLite+AgentDB backend (ADR-006, ADR-009), vector quantization (4-32x reduction), EWC++ for preventing catastrophic forgetting, cross-session persistence, namespace management. Pre/post hooks init namespace and compress.', ls: 1, le: 50 },
    { sev: 'MEDIUM', cat: 'architecture', desc: 'References ADR-006 (Unified Memory Service) and ADR-009 (Hybrid Memory Backend). Well-designed architecture but dependent on MCP tools for HNSW, compression, analytics that may be stubs.', ls: 19, le: 40 },
  ]);

  // collective-intelligence-coordinator.md (1002 LOC) — ASPIRATIONAL
  rec(1078, 1002, 'DEEP', [
    { sev: 'HIGH', cat: 'architecture', desc: 'Collective Intelligence Coordinator: hive-mind consensus with Byzantine fault tolerance (2/3+1 threshold), CRDT synchronization, attention-based coordination, distributed cognition, multi-agent voting. Uses Opus model. Pre hooks init hierarchical-mesh topology, CRDT sync, Byzantine consensus, neural pattern training.', ls: 1, le: 50 },
    { sev: 'MEDIUM', cat: 'fabrication', desc: 'Heavy MCP tool dependencies: swarm_init, daa_consensus, neural_patterns, neural_train, swarm_monitor — many of these return fabricated metrics (confirmed in swarm-coordination domain analysis).', ls: 27, le: 50 },
  ]);

  // security-architect.md (868 LOC) — ASPIRATIONAL with REAL patterns
  rec(1840, 868, 'DEEP', [
    { sev: 'HIGH', cat: 'architecture', desc: 'V3 Security Architect: STRIDE/DREAD threat modeling, CVE tracking, claims-based authorization (ADR-010), zero-trust patterns. Pre hooks search HNSW for threat patterns, learn from past failures, check CVE database, init trajectory tracking. References ReasoningBank pattern learning.', ls: 1, le: 50 },
    { sev: 'INFO', cat: 'quality', desc: 'Security methodology (STRIDE, DREAD) is textbook-correct. CVE check triggered by keyword matching (auth, session, inject) is practical. Template quality good for guiding security-focused agents.', ls: 38, le: 48 },
  ]);

  // security-auditor.md (708 LOC) — ASPIRATIONAL with REAL patterns
  rec(1084, 708, 'DEEP', [
    { sev: 'INFO', cat: 'architecture', desc: 'Security Auditor template: vulnerability detection, code scanning, OWASP compliance, dependency analysis. Complements security-architect with hands-on auditing focus. Uses ReasoningBank for pattern learning. Well-structured for audit workflows.', ls: 1, le: 50 },
  ]);

  // goal/agent.md (824 LOC) — ASPIRATIONAL with REAL algorithms
  rec(1038, 824, 'DEEP', [
    { sev: 'HIGH', cat: 'architecture', desc: 'Goal-Oriented Action Planning (GOAP) agent: A* search with priority queue (L207-255 textbook-correct), PageRank goal prioritization via sublinear-time-solver (L152-176), utility-based action selection with 4-factor weighted scoring (L544-595), behavior tree implementation (L490-541), OODA loop replanning (L358-431). Well-designed planning framework.', ls: 68, le: 595 },
    { sev: 'MEDIUM', cat: 'architecture', desc: 'Depends on mcp__sublinear-time-solver MCP tools (solve, pageRank, analyzeMatrix, predictWithTemporalAdvantage) and mcp__flow-nexus tools (swarm_init, agent_spawn). Algorithms are correct but external tool availability uncertain.', ls: 51, le: 65 },
  ]);

  // ============================================================
  // AGENT 4: SPARC and Specialized Templates
  // ============================================================

  // sparc/refinement.md (742 LOC) — ASPIRATIONAL with REAL TDD
  rec(1052, 742, 'DEEP', [
    { sev: 'HIGH', cat: 'architecture', desc: 'SPARC Refinement agent: Red-Green-Refactor TDD cycle with pattern learning. References ReasoningBank searchPatterns, GNN-enhanced code search (+12.4% claimed), Flash Attention for test suites. TDD examples (L260-468) are REAL, concrete TypeScript/Auth service code. Performance optimization patterns (L471-533) are practical.', ls: 45, le: 533 },
    { sev: 'MEDIUM', cat: 'fabrication', desc: 'GNN-enhanced search and Flash Attention references are ASPIRATIONAL — depend on agentDB functions that may not exist. Pattern storage reward system (0.5-1.0 range) well-designed but untested.', ls: 82, le: 188 },
  ]);

  // sparc/architecture.md (699 LOC) — ASPIRATIONAL with REAL patterns
  rec(1803, 699, 'DEEP', [
    { sev: 'HIGH', cat: 'architecture', desc: 'SPARC Architecture agent (Opus model): system architecture design with microservices, K8s, security patterns. Architecture examples (L220-533) are REAL production patterns. Flash Attention (4-7x claimed) and GNN search ASPIRATIONAL. Reward calculation includes scalability and maintainability scores.', ls: 45, le: 630 },
  ]);

  // templates/migration-plan.md (750 LOC) — REAL planning framework
  rec(1069, 750, 'DEEP', [
    { sev: 'HIGH', cat: 'quality', desc: 'Migration planning template: highest quality (80% REAL). Clear YAML agent schema (L38-59), 8-category migration taxonomy (Coordination, GitHub, SPARC, Analysis, Memory, Automation, Optimization, Monitoring), concrete agent definitions with tool access control, 5-step migration process with backwards compatibility. Production-ready.', ls: 38, le: 750 },
  ]);

  // optimization/resource-allocator.md (682 LOC) — ASPIRATIONAL
  rec(1046, 682, 'DEEP', [
    { sev: 'MEDIUM', cat: 'architecture', desc: 'Resource allocator: multi-objective genetic algorithm, LSTM predictive scaling, DQN reinforcement learning, adaptive circuit breaker (REAL pattern), bulkhead pattern (REAL). Heavy MCP dependency for metrics_collect, daa_resource_alloc. ~30% real, ~70% aspirational.', ls: 29, le: 497 },
  ]);

  // optimization/performance-monitor.md (680 LOC) — ASPIRATIONAL with REAL metrics
  rec(1045, 680, 'DEEP', [
    { sev: 'MEDIUM', cat: 'architecture', desc: 'Performance monitor: multi-dimensional metrics (system, agents, coordination, tasks, resources, network), SLA monitoring with threshold evaluation (REAL), ensemble anomaly detection (ASPIRATIONAL — LSTM, statistical, behavioral models), bottleneck analysis. ~50% real collection framework, ~50% aspirational ML.', ls: 29, le: 512 },
  ]);

  // optimization/benchmark-suite.md (673 LOC) — MOSTLY REAL
  rec(1043, 673, 'DEEP', [
    { sev: 'HIGH', cat: 'quality', desc: 'Benchmark suite: highest quality among optimization templates (70% REAL). CUSUM change-point detection for regression (REAL algorithm), load testing with ramp-up phases (industry-standard), stress testing to breaking point, SLA validation, scalability linearity analysis. Mostly implementable without external dependencies.', ls: 134, le: 509 },
  ]);

  // swarm/adaptive-coordinator.md (1127 LOC) — ASPIRATIONAL with REAL topology
  rec(1814, 1127, 'DEEP', [
    { sev: 'HIGH', cat: 'architecture', desc: 'Adaptive coordinator (1127 LOC, largest template): dynamic topology switching (hierarchical/mesh/ring/hybrid), 5 attention mechanisms (Flash, Multi-head, Linear, Hyperbolic, MoE). MoE routing is REAL and implementable (L276-338): capabilityScore*0.5 + performanceScore*0.3 + availabilityScore*0.2. GraphRoPE position encoding for topologies. Performance mechanism selection from history.', ls: 85, le: 565 },
    { sev: 'MEDIUM', cat: 'fabrication', desc: 'Flash Attention (2.49x-7.47x) and HNSW (150x-12500x) claims repeated without benchmarks. References attentionService functions that may not exist. ~40% real topology/routing, ~60% aspirational ML/attention.', ls: 143, le: 240 },
  ]);

  // github/workflow-automation.md (835 LOC) — ASPIRATIONAL with REAL workflows
  rec(1037, 835, 'DEEP', [
    { sev: 'HIGH', cat: 'architecture', desc: 'GitHub workflow automation: pattern learning from past workflows, GNN-enhanced optimization (+12.4% claimed), GitHub Actions YAML templates (REAL, L246-289), security scanning workflow (REAL, L293-347), gh CLI integration (REAL, L330-346). Mix of aspirational AI features and production-ready workflow patterns.', ls: 36, le: 377 },
  ]);

  // validate-agent.sh (218 LOC) — REAL
  rec(1353, 218, 'DEEP', [
    { sev: 'HIGH', cat: 'quality', desc: 'Agent validation script (95% REAL): YAML frontmatter validation, required field checking (name, description, model, color), name regex /^[a-zA-Z0-9][a-zA-Z0-9-]*[a-zA-Z0-9]$/ (3-50 chars), description length (10-5000), example block detection, trigger pattern detection, model enum (inherit/sonnet/opus/haiku), color enum, system prompt length/structure checks. Production-ready.', ls: 26, le: 203 },
  ]);

})();

console.log(`R15 import complete: ${filesOk} files recorded, ${findingsOk} findings inserted`);
db.close();
