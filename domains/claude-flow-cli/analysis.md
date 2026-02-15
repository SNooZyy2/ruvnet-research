# Claude Flow CLI Domain Analysis

> **Priority**: HIGH | **Coverage**: ~13.8% (45/326 DEEP) | **Status**: In Progress
> **Last updated**: 2026-02-14 (Session R17)

## 1. Current State Summary

The claude-flow CLI domain spans 326 files across the v3/@claude-flow/cli package, comprising the published npm entrypoint, MCP server, and command implementations. Quality is bifurcated — core infrastructure (doctor 95%, embeddings 95%, neural 90%, agentdb 90%) is production-ready, while governance commands (config 10%, deployment 10%, migrate 20%, providers 30%) are UI shells with simulated persistence.

**Top-level verdicts:**

- **CLI is a thin presentation layer** delegating to MCP tools via `callMCPTool()`. Real work happens in dist/src/mcp-tools/ (170+ tools).
- **MCP server has two implementations** — full v3/mcp/server.ts (repo-only) vs CLI embedded dist/src/mcp-client.js (published). They differ in behavior.
- **Optional dependencies create two experiences** — without agentdb/agentic-flow/@ruvector, memory degrades to no-ops while appearing to work.
- **RuVector PostgreSQL bridge has systemic extension confusion** — setup.js creates `ruvector`, init.js creates `vector`, hardcoded dimension mismatches (384/1536/configurable).
- **Config command has zero persistence** — init/get/set/export/import are entirely UI shells, never write to disk.
- **Three fragmented ReasoningBanks** — claude-flow (Map+JSON), agentic-flow (SQLite+DeepMind), agentdb (SQLite+embeddings). Only claude-flow's runs.
- **HNSW is pure TypeScript** — real implementation following Malkov & Yashunin but ~60x slower than native. "150x-12,500x" claim is vs brute-force, misleading.
- **Learning system is architecturally sound but operationally limited** — RL algorithms are tabular (Map-based), not neural. "LoRA" operates on routing tables, not LLM weights.
- **Graceful degradation everywhere** — dynamic imports with null fallback for all optional deps. Silent failures hide missing functionality.
- **Best security** — neural.js export has Ed25519 signing, PII stripping, secret detection. Production-grade.

## 2. File Registry

### CLI Commands

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| analyze.js | cli | 1,823 | 70% | DEEP | Real AST/graph. code/deps subcommands stubbed | R17 |
| embeddings.js | cli | 1,576 | 95% | DEEP | BEST command. 14 subcommands, real HNSW, sql.js | R17 |
| neural.js | cli | 1,448 | 90% | DEEP | Real WASM training, Ed25519 signing, PII strip | R17 |
| memory.js | cli | 1,268 | 85% | DEEP | 9-table sql.js schema. cleanup/compress→MCP | R17 |
| init.js | cli | 964 | 85% | DEEP | Wizard + upgrade + codex. Windows execSync bug | R17 |
| session.js | cli | 760 | 80% | DEEP | Real save/restore. YAML import broken (JSON.parse) | R17 |
| mcp.js | cli | 700 | 50% | DEEP | toggle broken, logs hardcoded, tool list stale | R17 |
| task.js | cli | 671 | 90% | DEEP | 10 types, 4 priorities, dependencies | R17 |
| agentdb.js | cli | 625 | 90% | DEEP | Real reflexion: episodes, skills, hybrid search | R17 |
| workflow.js | cli | 617 | 60% | DEEP | Template create never saves. Metadata hardcoded | R17 |
| status.js | cli | 584 | 70% | DEEP | Real MCP aggregation. Perf metrics hardcoded | R17 |
| performance.js | cli | 579 | 75% | DEEP | Real benchmarks. optimize/bottleneck stubbed | R17 |
| security.js | cli | 575 | 45% | DEEP | Real npm audit + secret scan. CVE/STRIDE static | R17 |
| doctor.js | cli | 571 | 95% | DEEP | Parallel health checks, npx cache detect, auto-fix | R17 |
| issues.js | cli | 567 | 95% | DEEP | Real ADR-016 claims with kanban | R17 |
| completions.js | cli | 539 | 100% | DEEP | 4 shells. Hardcoded command lists | R17 |
| benchmark.js | cli | 459 | 80% | DEEP | Real neural/memory benchmarks with fallbacks | R17 |
| start.js | cli | 418 | 85% | DEEP | MCP integration, daemon mode, PID management | R17 |
| config.js | cli | 406 | 10% | DEEP | STUB. No persistence anywhere | R17 |
| migrate.js | cli | 410 | 20% | DEEP | V2→V3 steps hardcoded. Good breaking changes docs | R17 |
| claims.js | cli | 373 | 40% | DEEP | Real wildcard eval. grant/revoke don't persist | R17 |
| index.js | cli | 366 | 100% | DEEP | Lazy loading saves ~200ms | R17 |
| deployment.js | cli | 289 | 10% | DEEP | All steps simulated with setTimeout() | R17 |
| update.js | cli | 276 | 95% | DEEP | Real npm check with rate limiting | R17 |
| progress.js | cli | 259 | 100% | DEEP | Real MCP integration | R17 |
| providers.js | cli | 232 | 30% | DEEP | Hardcoded provider data, simulated config | R17 |
| categories.js | cli | 178 | 100% | DEEP | Clean DDD taxonomy | R17 |

### RuVector PostgreSQL Bridge

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| setup.js | cli/ruvector-pg | 765 | 70% | DEEP | Real SQL gen. Creates `ruvector` extension | R17 |
| backup.js | cli/ruvector-pg | 746 | 85% | DEEP | Real backup: SQL/JSON/CSV + compression | R17 |
| optimize.js | cli/ruvector-pg | 503 | 80% | DEEP | Real pg health + tuning recommendations | R17 |
| migrate.js | cli/ruvector-pg | 481 | 75% | DEEP | 6 migrations, checksum system | R17 |
| benchmark.js | cli/ruvector-pg | 480 | 85% | DEEP | Real perf with percentiles | R17 |
| status.js | cli/ruvector-pg | 456 | 80% | DEEP | Real connection + schema health | R17 |
| init.js | cli/ruvector-pg | 431 | 70% | DEEP | Creates `vector` extension (conflicts setup) | R17 |
| import.js | cli/ruvector-pg | 349 | 80% | DEEP | sql.js→PostgreSQL. Uses `ruvector(384)` | R17 |
| index.js | cli/ruvector-pg | 129 | 100% | DEEP | Clean coordinator | R17 |

### Memory & HNSW

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| hnsw-index.ts | memory | 27,799 | 85% | SURFACE | Pure TS HNSW. Real but slow vs native | R1 |
| sqlite-backend.ts | memory | 20,000 | 80% | SURFACE | SQLite persistence | R1 |
| hybrid-backend.ts | memory | 19,253 | 80% | SURFACE | SQLite+HNSW integration | R1 |
| agentdb-adapter.ts | memory | 27,278 | 75% | SURFACE | AgentDB integration with fallback | R1 |

### Neural & Learning

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| reasoning-bank.ts | neural | 38,604 | 75% | SURFACE | Trajectory learning. RETRIEVE/JUDGE/DISTILL/CONSOLIDATE | R1 |
| sona-manager.ts | neural | 22,661 | 70% | SURFACE | 5 modes: real-time/balanced/research/edge/batch | R1 |
| pattern-learner.ts | neural | 22,312 | 70% | SURFACE | Pattern extraction, clustering, quality tracking | R1 |

### Init System

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| templates.js | cli/init | 1,346 | 90% | DEEP | 7 architecture templates with good defaults | R17 |
| dependency-installer.js | cli/init | 924 | 85% | DEEP | Real npm/pnpm/yarn detection + parallel install | R17 |
| workspace-setup.js | cli/init | 854 | 80% | DEEP | Real workspace creation with .gitignore gen | R17 |
| upgrade-handler.js | cli/init | 778 | 75% | DEEP | V2→V3 migration orchestration | R17 |
| config-manager.js | cli/init | 692 | 70% | DEEP | Config merge + validation | R17 |
| cli-integration.js | cli/init | 592 | 65% | DEEP | Claude Code settings.json generation | R17 |
| codex-integration.js | cli/init | 583 | 60% | DEEP | Codex ingestion setup (depends on external tool) | R17 |
| index.js | cli/init | 451 | 85% | DEEP | Clean init orchestrator with wizard flow | R17 |

## 3. Findings Registry

### 3a. CRITICAL Findings

| ID | Description | File(s) | Session | Status |
|----|-------------|---------|---------|--------|
| C1 | **Three fragmented ReasoningBanks** — Zero code sharing across packages | claude-flow, agentic-flow, agentdb | R1 | Open |
| C2 | **RuVector extension confusion** — setup creates `ruvector`, init creates `vector`, hardcoded dimension mismatches (384/1536/configurable) | ruvector-pg/*.js | R17 | Open |
| C3 | **config.js has zero persistence** — init/get/set/export/import never write to disk | config.js | R17 | Open |
| C4 | **MCP server behavior divergence** — Full server (v3/mcp/server.ts) vs CLI embedded (dist/src/mcp-client.js) may behave differently | server.ts, mcp-client.js | R1 | Open |
| C5 | **Optional deps create silent degradation** — Memory/learning features appear to work but do nothing without agentdb/agentic-flow | memory-tools.ts, hooks-tools.ts | R1 | Open |
| C6 | **ONNX embeddings broken** — downloadModel is not a function, falls back to hash-based | enhanced-embeddings.ts | R1 | Open |
| C7 | **Agent Booster WASM doesn't exist** — dist/agent-booster/ directory missing in published package | agentic-flow | R1 | Open |
| C8 | **YAML session import broken** — Uses JSON.parse instead of YAML parser | session.js | R17 | Open |
| C9 | **workflow template create never saves** — Hardcoded metadata in CLI, not persisted | workflow.js | R17 | Open |
| C10 | **deployment command entirely simulated** — All steps use setTimeout(), no real deployment | deployment.js | R17 | Open |

### 3b. HIGH Findings

| ID | Description | File(s) | Session | Status |
|----|-------------|---------|---------|--------|
| H1 | **HNSW speed claims misleading** — "150x-12,500x" is vs brute-force, not vs native libs | hnsw-index.ts | R1 | Open |
| H2 | **Performance metrics hardcoded** — "<0.05ms adaptation", "352x transforms", "34,798 routes/s" are marketing strings | status.js, various | R1 | Open |
| H3 | **MCP toggle non-functional** — mcp.js toggle command doesn't actually enable/disable | mcp.js | R17 | Open |
| H4 | **MCP logs hardcoded** — Doesn't read real log files | mcp.js | R17 | Open |
| H5 | **MCP tool list stale** — Fallback list doesn't match actual tools | mcp.js | R17 | Open |
| H6 | **security CVE/STRIDE/audit are static examples** — Not real vulnerability scanning | security.js | R17 | Open |
| H7 | **performance optimize/bottleneck are stubs** — Placeholder implementations | performance.js | R17 | Open |
| H8 | **analyze code/deps subcommands stubbed** — Main functionality unimplemented | analyze.js | R17 | Open |
| H9 | **providers command simulates config updates** — Doesn't actually persist provider changes | providers.js | R17 | Open |
| H10 | **migrate V2→V3 steps hardcoded** — Not real migration, just simulation | migrate.js | R17 | Open |
| H11 | **claims grant/revoke don't persist** — Only in-memory, lost on restart | claims.js | R17 | Open |
| H12 | **Windows execSync bug in init** — Uses shell mode without proper escaping | init.js | R17 | Open |
| H13 | **completions hardcoded command lists** — Maintenance burden, will drift | completions.js | R17 | Open |
| H14 | **codex integration depends on external tool** — Not self-contained | codex-integration.js | R17 | Open |
| H15 | **ruvector-pg migrate dimensions hardcoded** — Uses vector(1536) regardless of config | migrate.js (pg) | R17 | Open |
| H16 | **ruvector-pg import dimensions hardcoded** — Uses ruvector(384) regardless of config | import.js (pg) | R17 | Open |

## 4. Positives Registry

| Description | File(s) | Session |
|-------------|---------|---------|
| **embeddings.js is BEST command** — 14 subcommands, real HNSW, sql.js search, 95% complete | embeddings.js | R17 |
| **neural.js has production-grade security** — Ed25519 signing, PII stripping, secret detection for exports | neural.js | R17 |
| **doctor.js excellent health checks** — Parallel checks, npx cache detection, auto-fix capability | doctor.js | R17 |
| **agentdb.js real reflexion memory** — Episodes, skills, hybrid search from arxiv papers | agentdb.js | R17 |
| **issues.js real claims system** — ADR-016 implementation with kanban board view | issues.js | R17 |
| **task.js rich task model** — 10 types, 4 priorities, dependency tracking | task.js | R17 |
| **memory.js real 9-table schema** — Structured sql.js with proper indexing | memory.js | R17 |
| **completions 100% complete** — Works for bash/zsh/fish/powershell | completions.js | R17 |
| **lazy loading optimization** — index.js saves ~200ms startup time | index.js | R17 |
| **graceful degradation everywhere** — All optional deps have null fallbacks | All commands | R17 |
| **MCP-first architecture** — Clean separation: CLI is presentation, MCP does work | All commands | R17 |
| **backup.js comprehensive formats** — SQL/JSON/CSV + compression support | backup.js (pg) | R17 |
| **optimize.js real pg tuning** — Health analysis + tuning recommendations | optimize.js (pg) | R17 |
| **benchmark.js real percentiles** — P50/P90/P95/P99 performance measurement | benchmark.js (pg) | R17 |
| **templates.js 7 architecture templates** — Good defaults for different project types | templates.js | R17 |
| **dependency-installer.js multi-manager** — Real detection of npm/pnpm/yarn | dependency-installer.js | R17 |
| **hnsw-index.ts real HNSW implementation** — Follows Malkov & Yashunin paper correctly | hnsw-index.ts | R1 |
| **reasoning-bank.ts 4-step learning** — RETRIEVE/JUDGE/DISTILL/CONSOLIDATE from DeepMind | reasoning-bank.ts | R1 |

## 5. Subsystem Sections

### 5a. CLI Architecture & MCP Integration

The CLI follows a **thin presentation layer** pattern where every command delegates to MCP tools via `callMCPTool()`. The architecture is MCP-first (R17). Commands format arguments, handle user interaction, and display results, but actual work happens in dist/src/mcp-tools/ (170+ tools).

**Two MCP implementations** exist with different behavior (R1): the full server at v3/mcp/server.ts (20,632 bytes, repo-only) supports multiple transports (stdio/HTTP/WebSocket/in-process), session management, connection pooling, and metrics tracking. The CLI embedded version at dist/src/mcp-client.js (published to npm) is simpler and auto-detected when stdin is piped.

**Lazy loading optimization** (R17): index.js uses on-demand command loading, saving ~200ms startup time. Commands are only `require()`'d when invoked.

**Graceful degradation** is systematic (R17) — every command has dynamic imports with null fallback for optional deps (ruvector, aidefence, codex). The system appears to work while doing nothing when deps are missing.

### 5b. ReasoningBank Fragmentation

**Three completely independent ReasoningBanks** exist with zero code sharing (R1):

1. **claude-flow's** (hooks-tools.js): In-memory Map + JSON file at ~/.claude-flow/neural/patterns.json. No judge, no distill, no consolidation. **This is the only one that runs.**
2. **agentic-flow's** (reasoning-bank.js): Most sophisticated — 5 algorithms from DeepMind paper (Retrieve, Judge, Distill, Consolidate, MaTTS). Requires LLM API key. SQLite at .swarm/memory.db.
3. **agentdb's** (ReasoningBank.js): Pattern store with optional GNN enhancement and VectorBackend integration.

claude-flow uses agentic-flow's ReasoningBank only for `retrieveMemories()` in token-optimize hook (read-only). The learning pipeline (judge/distill/consolidate) never runs.

### 5c. HNSW Vector Search

**hnsw-index.ts is a real, from-scratch HNSW implementation** in TypeScript (27,799 bytes) following the Malkov & Yashunin paper (R1). Key features: multi-layer graph with configurable M/efConstruction, pre-normalized Float32Array vectors for O(1) cosine similarity, BinaryMinHeap/BinaryMaxHeap for priority queues, int8 quantization support, layer-by-layer search from top to layer 0.

**Pure TypeScript means ~60x slower than native** — the "150x-12,500x faster" claim (R1) is vs brute-force linear scan (O(n) → O(log n)), not vs hnswlib/FAISS. For small datasets (hundreds to low thousands) it works fine. For large datasets, it lags native implementations.

**Optional native enhancement** via @ruvector/core (R1): if installed, provides Rust-based HNSW with genuine 60x speedup over JS implementation. But it's optional and most users won't have it.

### 5d. Learning System (SONA / Neural)

**Architecturally sound but operationally limited** (R1). The system tracks trajectories, stores patterns, and implements RL algorithms — this is real. But execution is constrained:

**RL algorithms are tabular** (R1): 7 real implementations (q-learning.ts 333 lines, sarsa.ts 383, dqn.ts 382, ppo.ts 429, a2c.ts 478, curiosity.ts 509, decision-transformer.ts 521) but all use in-memory Maps as Q-tables, not neural networks with weight matrices. DQN uses simple array-based "networks."

**"LoRA" is not LLM weight adaptation** (R1) — it operates on in-memory routing tables and pattern weights, not transformer model weights. "EWC++" is weight consolidation for the local pattern store, not for neural nets. The "<0.05ms adaptation" claim is plausible for updating an in-memory map, not training a neural network.

**SONA Manager** (sona-manager.ts, 22,661 bytes, R1) has 5 real operating modes: real-time (LoRA rank 2, <0.5ms), balanced (rank 4, <18ms), research (rank 16, <100ms), edge (rank 1, <1ms, <5MB), batch (rank 8, <50ms). Each mode configures learning rate, batch size, HNSW ef_search, pattern threshold, and optimizations.

**Pattern Learner** (pattern-learner.ts, 22,312 bytes, R1) implements pattern extraction, clustering, cosine similarity matching, quality tracking, and promotion. Uses configurable thresholds and learning rates.

**Without optional deps** (R1), everything falls back to in-memory Maps that don't persist across sessions. Learning metrics reset on restart.

### 5e. Memory & Persistence

**memory.js implements real 9-table sql.js schema** (R17): stores, sessions, embeddings, clusters, associations, queries, feedback, analytics, config. cleanup/compress commands delegate to MCP tools.

**Three backend options** (R1): (1) sqlite-backend.ts (20,000 bytes) — SQLite persistence with WAL mode, (2) hybrid-backend.ts (19,253 bytes) — SQLite + HNSW integration, (3) agentdb-adapter.ts (27,278 bytes) — AgentDB integration with fallback to in-memory.

**Memory tools** (memory-tools.ts, R1) attempt to use UnifiedMemoryService with HNSW, but fall back to simple in-memory storage that just returns `{ id, stored: true, storedAt }` without actual persistence when optional deps are missing. Silent degradation makes the system appear to work while doing nothing.

### 5f. Command Quality Spectrum

**Production-ready** (R17): embeddings (95%), doctor (95%), neural (90%), agentdb (90%), task (90%), update (95%), completions (100%), progress (100%), categories (100%), index (100%).

**Functional with gaps** (R17): memory (85%), init (85%), session (80%), analyze (70%), status (70%), performance (75%), benchmark (80%), start (85%).

**Partially stubbed** (R17): mcp (50%), workflow (60%), security (45%), claims (40%).

**UI shells with no persistence** (R17): config (10%), deployment (10%), migrate (20%), providers (30%).

**Best security**: neural.js export (R17) has production-grade Ed25519 signing with PII stripping and secret detection. Scans for credentials/API keys before export.

### 5g. RuVector PostgreSQL Bridge

**9 files, 4,211 LOC with systemic extension confusion** (R17). setup.js creates `ruvector(384)` types and `CREATE EXTENSION ruvector`. init.js creates `vector(${dimensions})` types and `CREATE EXTENSION vector`. import.js uses `ruvector(384)` hardcoded. migrate.js uses `vector(1536)` hardcoded. benchmark.js uses `vector(${dimensions})`.

**Impact**: Database initialization will fail or use the wrong extension. Dimension mismatches (384 vs 1536 vs configurable) cause import/migration failures.

**Otherwise high-quality code** (R17): backup.js has comprehensive SQL/JSON/CSV export with compression. optimize.js has real PostgreSQL health analysis + tuning recommendations. benchmark.js has real percentiles (P50/P90/P95/P99). status.js has connection + schema health checks.

### 5h. Init System

**8 files, 6,220 LOC, 80% weighted average** (R17). templates.js has 7 architecture templates (minimal/standard/advanced/enterprise/monorepo/plugin/custom) with good defaults. dependency-installer.js has real npm/pnpm/yarn detection + parallel install. workspace-setup.js creates workspace with real .gitignore generation.

**Upgrade handler** (upgrade-handler.js, 778 LOC, R17) orchestrates V2→V3 migration but actual steps are in migrate.js which is 20% real (hardcoded steps).

**Codex integration** (codex-integration.js, 583 LOC, R17) depends on external tool, not self-contained.

**Windows bug** (R17): init.js uses execSync with shell mode without proper escaping, risk of command injection on Windows paths with spaces.

### 5i. Package Distribution & Dependency Strategy

**What's published to npm** (R1): CLI bin + dist, shared types, guidance framework, .claude-plugin, .claude agents. NOT published: v3/@claude-flow/memory, neural, hooks, claims, embeddings, swarm, browser, aidefence, mcp (full server), providers.

**However**, the CLI's dist/ directory contains embedded compiled versions of many features (dist/src/mcp-tools/hooks-tools.js 112KB, dist/src/memory/, dist/src/ruvector/).

**Optional dependencies** all first-party (R1): agentic-flow v2.0.6 (574MB, 96% bundled deps), agentdb v2.0.0-alpha.3.4 (68MB), @ruvector/core v0.1.30 (5.2MB Rust binary), better-sqlite3 (native compilation).

**Two-tier experience** (R1): with optional deps installed, ~60x speedup for memory search, real semantic embeddings, persistent learning. Without them, graceful degradation to non-persistent in-memory Maps, hash-based embeddings, features that appear to work but do nothing.

## 6. Cross-Domain Dependencies

- **memory-and-learning domain**: ReasoningBank implementations, HNSW, embeddings, pattern storage
- **agentdb-integration domain**: AgentDB controllers, reflexion memory, skill library
- **agentic-flow domain**: EmbeddingService, multi-provider routing (unused)
- **ruvector domain**: Native HNSW, attention, SONA, router binaries

## 7. Knowledge Gaps

- MCP server full implementation at v3/mcp/server.ts (20,632 bytes) — how does it differ from CLI embedded?
- v3/mcp/tools/ directory — are these duplicates of dist/src/mcp-tools/ or different?
- Hooks system integration — how do .claude/helpers/ shell scripts wire into MCP?
- Agent system runtime — how does Task tool invocation work with .claude/agents/ templates?
- Claims system persistence — where is the real backend vs simulated?
- WASM modules — which exist, which are missing, which are stubs?
- Browser build — is this real or placeholder?

## 8. Session Log

### R1 (2026-02-06): Initial repository audit
Package structure, MCP server, HNSW, learning system, dependencies assessed. Three ReasoningBanks discovered. Optional deps two-tier experience identified.

### R17 (2026-02-14): CLI command deep-read
45 files, 33,929 LOC, 79 findings. All 37 command files + 8 init system files. Quality matrix established. RuVector PostgreSQL extension confusion discovered. Config command zero persistence confirmed.
