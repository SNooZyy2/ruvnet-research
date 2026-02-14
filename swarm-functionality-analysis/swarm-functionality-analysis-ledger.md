# Codebase Coverage Ledger

**Purpose**: Track every file analyzed across all research sessions, with exact depth, lines read, and findings.
**Last Updated**: 2026-02-14 (R3 discovery + ledger verification complete)
**Total Research Rounds**: 3 + verification (5 + 4 + 5 + 4 agents = 18 agents total)
**Research Doc**: `research/swarm-functionality-analysis.md` (1,377 lines)
**Sections**: 23 categories, ~479 files tracked

## Depth Legend

| Level | Meaning | Evidence |
|-------|---------|----------|
| **DEEP** | Read 50%+ of file, exact algorithms/line numbers extracted | Code snippets in research doc |
| **MEDIUM** | Read key sections (20-50%), architecture understood | Section summaries, some line refs |
| **SURFACE** | Glob/grep only, categorized by filename/directory | Counts and categories only |
| **MENTIONED** | Referenced in research but not directly read | Inferred from other files |
| **NOT TOUCHED** | Known to exist, zero analysis | Listed as gap |

## Coverage Summary

| Category | Files Known | Deep | Medium | Surface | Mentioned | Not Touched |
|----------|------------|------|--------|---------|-----------|-------------|
| CLI Commands | ~15 | 3 | 2 | 5 | 3 | 2 |
| MCP Tools | 24 | 6 | 2 | 16 | 0 | 0 |
| Services | 5 | 3 | 1 | 1 | 0 | 0 |
| Memory | 1 | 1 | 0 | 0 | 0 | 0 |
| RuVector | 14 | 3 | 1 | 3 | 0 | 7 |
| MCP Server/Client | 3 | 2 | 0 | 1 | 0 | 0 |
| Helpers | 45 | 12 | 2 | 6 | 0 | 25 |
| Agent Templates | 95 | 5 | 0 | 0 | 0 | 90 |
| Skills | 33 | 2 | 0 | 3 | 0 | 28 |
| Commands | 90+ | 0 | 0 | 10 | 0 | 80+ |
| Marketplace Plugins | 19 | 0 | 0 | 0 | 0 | 19 |
| Config | 3 | 2 | 0 | 1 | 0 | 0 |
| **Init System** | **8** | 0 | 0 | 0 | 0 | **8** |
| **Production** | **6** | 0 | 0 | 0 | 0 | **6** |
| **Transfer** | **19** | 0 | 0 | 0 | 0 | **19** |
| **Plugins Impl** | **8** | 0 | **1** | **1** | 0 | **6** |
| **Update System** | **5** | 0 | 0 | 0 | 0 | **5** |
| **Root src files** | **5** | 0 | 0 | 0 | 0 | **5** |
| **Infrastructure** | **1** | 0 | 0 | 0 | 0 | **1** |
| **Benchmarks** | **1** | 0 | 0 | 0 | 0 | **1** |
| Guidance Pkg | 3 | 0 | 2 | 1 | 0 | 0 |
| Source (TS) | 55 | 0 | 0 | 0 | 5 | 50 |
| Tests | 21 | 0 | 0 | 0 | 3 | 18 |
| **TOTAL** | **~479** | **40** | **11** | **47** | **11** | **~370** |

**Coverage rate**: ~109 files touched / ~479 known = **~23% analyzed** (deep+medium: ~11%)
**Lines discovered**: ~14,800 new lines across 53 newly inventoried files (R3 discovery agent)

---

## FILE-BY-FILE LEDGER

Base path for CLI: `/home/snoozyy/.npm-global/lib/node_modules/@claude-flow/cli/`
Base path for helpers: `/home/snoozyy/.claude/helpers/`
Base path for agents: `/home/snoozyy/.claude/agents/`
Base path for skills: `/home/snoozyy/.claude/skills/`
Base path for commands: `/home/snoozyy/.claude/commands/`
Base path for source: `/home/snoozyy/claude-flow-self-implemented/src/`

### 1. CLI Commands (`dist/src/commands/`)

| # | File | Lines | Depth | Lines Read | Round | Key Findings |
|---|------|-------|-------|------------|-------|-------------|
| 1 | `swarm.js` | 748 | DEEP | ~300 (multiple ranges) | R1 | Entirely presentational. `init` writes JSON, `status` counts files, `coordinate` displays hardcoded 15-agent table. Topologies = string labels, zero behavioral difference. |
| 2 | `agent.js` | ~400 | MEDIUM | ~100 | R1 | `spawn` creates JSON record in `.swarm/agents/`. No process spawning. Template loading from `.claude/agents/`. |
| 3 | `hive-mind.js` | ~500 | DEEP | ~200 (lines 170-320) | R3 | **CRITICAL**: Line 218 spawns `claude` CLI with full TTY. Line 183 checks `which claude`. Generates Byzantine consensus prompt. `--dangerously-skip-permissions`. SIGINT handler. |
| 4 | `daemon.js` | ~300 | DEEP | ~100 (lines 170-270) | R3 | Line 184: `spawn(process.execPath, [cliPath, 'daemon', 'start', '--foreground', '--quiet'])`. Detached, PID file, SIGTERM cleanup. |
| 5 | `hooks.js` | ~2800 | SURFACE | ~50 (grep) | R2 | Lines 2731, 2820-2821 use ps/git. Mostly CLI wrapper for hooks-tools.js MCP handlers. |
| 6 | `guidance.js` | 560 | MEDIUM | ~150 | R3 | 6 subcommands: compile, retrieve, gates, status, optimize, ab-test. Hash-shard retrieval. Constitutional AI. |
| 7 | `plugins.js` | 820 | MEDIUM | ~200 | R3 | 9 subcommands. IPFS registry, Ed25519 signatures, trust levels. Install/uninstall/upgrade. |
| 8 | `init.js` | ~600 | SURFACE | ~30 (grep) | R3 | Lines 286, 301, 325: starts daemon, swarm init, embeddings init via execSync. |
| 9 | `doctor.js` | ~400 | SURFACE | ~10 (grep) | R3 | Line 365: `npm install -g @anthropic-ai/claude-code`. Health checks. |
| 10 | `session.js` | ~300 | MENTIONED | 0 | R1 | Referenced via ADR-017 fix. `s.updatedAt` vs `s.savedAt` bug. Not directly read. |
| 11 | `ruvector/import.js` | ~300 | SURFACE | ~5 (grep) | R3 | Line 294: docker exec psql import. |
| 12 | `skills.js` | unknown | NOT TOUCHED | 0 | - | |
| 13 | `memory.js` | unknown | NOT TOUCHED | 0 | - | CLI wrapper for memory-tools.js. |

### 2. MCP Tool Modules (`dist/src/mcp-tools/`)

| # | File | Lines | Depth | Lines Read | Round | Key Findings |
|---|------|-------|-------|------------|-------|-------------|
| 1 | `hooks-tools.js` | 2,976 | DEEP | ~800 (10+ ranges) | R1,R3 | Core intelligence. Swarm-gate (2759-2893), model-route (2652-2691), worker dispatch (2442-2632), trajectory (1500-1680), patterns (1712-1767), SONA/EWC++ (1627-1680). Exact formulas. |
| 2 | `swarm-tools.js` | 101 | DEEP | Full | R1 | Stub handlers. `swarm_init` returns timestamp. `swarm_status` returns hardcoded zeros. `swarm_health` = "healthy". |
| 3 | `hive-mind-tools.js` | ~500 | DEEP | ~200 | R1 | JSON file state at `.claude-flow/hive-mind/state.json`. Consensus = majority vote in JSON. |
| 4 | `coordination-tools.js` | 486 | DEEP | ~200 | R1 | Header: "LOCAL STATE MANAGEMENT". `coordination_metrics` returns `Math.random()`. |
| 5 | `agent-tools.js` | ~400 | DEEP | ~150 | R1 | `agent_spawn` creates JSON record. No process. Template loading. |
| 6 | `memory-tools.js` | ~300 | MEDIUM | ~50 | R2 | Wrapper for memory-initializer.js. |
| 7 | `claims-tools.js` | ~300 | MEDIUM | ~80 | R1 | File-based claim state. No distributed locking. |
| 8 | `agentdb-tools.js` | 685 | DEEP | ~200 | R1 (ADR-037) | Service-first bridge to TS DDD. `createRequire(import.meta.url)` ESM-CJS. Lazy singleton. |
| 9 | `session-tools.js` | ~300 | SURFACE | ~20 | R1 | ADR-017 dual-schema handling. |
| 10 | `task-tools.js` | ~200 | SURFACE | ~10 | R1 | Basic task CRUD in JSON. |
| 11 | `workflow-tools.js` | ~300 | SURFACE | ~10 (export array) | R1 | 9 tools counted. |
| 12 | `terminal-tools.js` | ~200 | SURFACE | ~10 (export array) | R2 | 5 tools counted. |
| 13 | `daa-tools.js` | ~300 | SURFACE | ~10 (export array) | R3 | 8 tools counted. |
| 14 | `browser-tools.js` | ~400 | SURFACE | ~10 (line 16) | R3 | `execSync('agent-browser ...')`. 23 tools. |
| 15 | `config-tools.js` | ~200 | SURFACE | ~10 (export array) | R3 | 6 tools counted. |
| 16 | `system-tools.js` | ~200 | SURFACE | ~10 (export array) | R3 | 5 tools counted. |
| 17 | `neural-tools.js` | ~300 | SURFACE | ~10 (export array) | R3 | 6 tools counted. |
| 18 | `performance-tools.js` | ~300 | SURFACE | ~10 (export array) | R3 | 8 tools counted. |
| 19 | `embeddings-tools.js` | ~300 | SURFACE | ~10 (export array) | R3 | 7 tools counted. |
| 20 | `github-tools.js` | ~300 | SURFACE | ~10 (export array) | R3 | 5 tools counted. |
| 21 | `transfer-tools.js` | ~300 | SURFACE | ~10 (export array) | R3 | 12 tools counted. |
| 22 | `security-tools.js` | ~500 | SURFACE | ~10 (export array) | R3 | 6 tools counted. |
| 23 | `analyze-tools.js` | ~400 | SURFACE | ~10 (export array) | R3 | 6 tools counted. |
| 24 | `progress-tools.js` | 348 | SURFACE | ~20 (lines 338-348) | R3 | 4 tools. Export array. |

### 3. Services (`dist/src/services/`)

| # | File | Lines | Depth | Lines Read | Round | Key Findings |
|---|------|-------|-------|------------|-------|-------------|
| 1 | `headless-worker-executor.js` | 810+ | DEEP | ~200 | R2,R3 | **CRITICAL**: Line 810 `spawn('claude', ['--print', prompt])`. 8 AI worker types. Pool max 2. Timeout 5-15 min. Line 389 checks claude --version. |
| 2 | `container-worker-pool.js` | 407+ | DEEP | ~150 | R2,R3 | **CRITICAL**: Line 407 Docker spawn. Image `ghcr.io/ruvnet/claude-flow-headless:latest`. 2 CPUs, 4GB. Health checks. |
| 3 | `worker-daemon.js` | 756 | DEEP | ~200 | R2 | 12 scheduled workers. Resource gating. Staggered scheduling. Local fallback = JSON generators. |
| 4 | `worker-queue.js` | 511 | MEDIUM | ~100 | R2 | Production job queue. Priority scheduling. Exponential backoff. Dead letter queue. |
| 5 | `runtime/headless.js` | unknown | SURFACE | ~10 | R3 | CLI wrapper. Not a service. |

### 4. Memory (`dist/src/memory/`)

| # | File | Lines | Depth | Lines Read | Round | Key Findings |
|---|------|-------|-------|------------|-------|-------------|
| 1 | `memory-initializer.js` | 1,929 | DEEP | ~500 (5+ ranges) | R3 | 9-table SQLite. MiniLM-L6-v2 384-dim ONNX. HNSW @ruvector/core. Flash Attention batch ops. Int8 quantization. storeEntry/searchEntries/listEntries/getEntry/deleteEntry. |

### 5. RuVector (`dist/src/ruvector/`)

| # | File | Lines | Depth | Lines Read | Round | Key Findings |
|---|------|-------|-------|------------|-------|-------------|
| 1 | `model-router.js` | 493 | DEEP | ~300 | R1,R3 | ADR-021 curves. Opus=`score*2.0`, Haiku drops 0.25, Sonnet peaks 0.45. Circuit breaker (5 fails). Learning history (100). `.swarm/model-router-state.json`. |
| 2 | `enhanced-model-router.js` | 300+ | DEEP | ~200 | R3 | 3-tier routing. Agent Booster = `npx agent-booster@0.2.2` (NOT WASM). 19 Opus keywords. 6 intents. Threshold 0.7. |
| 3 | `flash-attention.js` | 367 | DEEP | ~200 | R3 | Block-wise tiling (32 elem). 8x unrolling. Top-K sparse (12%). 2.49x-7.47x speedup. |
| 4 | `lora-adapter.js` | 400+ | MEDIUM | ~100 | R3 | Rank 8, 384-dim, scaling=2.0. Xavier/Kaiming. Online GD. Auto-save/50 updates. `.swarm/lora-weights.json`. |
| 5 | `moe-router.js` | 500+ | SURFACE | ~50 | R3 | 8 experts, 2-layer MLP gating, top-2. `.swarm/moe-weights.json`. Training not traced. |
| 6 | `semantic-router.js` | unknown | NOT TOUCHED | 0 | - | |
| 7 | `q-learning-router.js` | unknown | NOT TOUCHED | 0 | - | |
| 8 | `coverage-router.js` | unknown | NOT TOUCHED | 0 | - | |
| 9 | `diff-classifier.js` | unknown | NOT TOUCHED | 0 | - | |
| 10 | `ast-analyzer.js` | unknown | NOT TOUCHED | 0 | - | |
| 11 | `graph-analyzer.js` | unknown | NOT TOUCHED | 0 | - | |
| 12 | `vector-db.js` | unknown | NOT TOUCHED | 0 | - | |
| 13 | `index.js` | unknown | NOT TOUCHED | 0 | - | |
| 14 | `types.js` | unknown | NOT TOUCHED | 0 | - | |

### 6. MCP Server/Client

| # | File | Lines | Depth | Lines Read | Round | Key Findings |
|---|------|-------|-------|------------|-------|-------------|
| 1 | `bin/mcp-server.js` | 5,116 | DEEP | ~190 (first section) | R3 | JSON-RPC 2.0 stdio. 5 methods. Protocol `2024-11-05`. Remaining 4,900+ lines NOT read. |
| 2 | `bin/cli.js` | 4,368 | SURFACE | ~10 | R3 | Commander.js entry point. Full CLI with 60+ commands. |
| 3 | `dist/src/mcp-client.js` | 235 | DEEP | Full | R3 | 24 tool module imports. TOOL_REGISTRY Map. Synchronous reg. No middleware. |

### 7. Helper Scripts (`.claude/helpers/`)

| # | File | Lines | Depth | Lines Read | Round | Key Findings |
|---|------|-------|-------|------------|-------|-------------|
| 1 | `guidance-hooks.sh` | 109 | DEEP | Full | R3 | pre-edit, post-edit, pre-command, route, **session-context (23 lines V3 context injection)**, user-prompt. |
| 2 | `learning-hooks.sh` | 330 | DEEP | Full | R3 | session_start, session_end, store_pattern, search_patterns, run_benchmark. Integrates learning-service.mjs. |
| 3 | `reflexion-pre-task.sh` | 60 | DEEP | Full | R3 | BUG-010-A stdin JSON. Episode retrieval via `claude-flow agentdb episode-retrieve -k 5`. additionalContext output. |
| 4 | `reflexion-post-task.sh` | 75 | DEEP | Full | R3 | BUG-010-A. Reward 0.8/0.2. Background `{ cmd } & disown`. |
| 5 | `skill-suggest.sh` | 59 | DEEP | Full | R3 | BUG-010-A. `claude-flow agentdb skill-suggest -k 3`. Composite scoring. |
| 6 | `skill-extract.sh` | 63 | DEEP | Full | R3 | BUG-010-A. Only success + output>100. Background. |
| 7 | `format-routing-directive.sh` | 49 | DEEP | Full | R1 | JSON to `[SWARM REQUIRED]`/`[ROUTING DIRECTIVE]`. |
| 8 | `verify-patches.sh` | 150+ | DEEP | 80 lines | R3 | ADR-037. NPX hash. Patches both cache + local. Copies agentdb-tools.js + symlinks. |
| 9 | `statusline-v2.sh` | 130 | DEEP | Full | ADR-018 | stdin JSON. git cache 5s TTL. 31-64ms. |
| 10 | `swarm-hooks.sh` | ~400 | MEDIUM | ~150 | R1 | Agent registration, stale reaping, message queue, handoff protocol, consensus voting. File-based. |
| 11 | `swarm-comms.sh` | 354 | MEDIUM | ~100 | R2 | Priority queue (4 levels). Batching. Connection pooling. Async ops. |
| 12 | `session.js` | ~200 | MEDIUM | ~50 | R1 | Dual-writes DB + JSON. ADR-017 fix. |
| 13 | `learning-service.mjs` | ~300 | SURFACE | ~20 | R3 | Node.js HNSW learning service. Referenced by learning-hooks.sh. |
| 14 | `swarm-monitor.sh` | unknown | SURFACE | 0 | R3 | settings.json hook ref. Health check. |
| 15 | `checkpoint-manager.sh` | unknown | SURFACE | 0 | R3 | settings.json hook ref. Checkpoint. |
| 16 | `pattern-consolidator.sh` | unknown | SURFACE | 0 | R3 | settings.json hook ref. Consolidation. |
| 17 | `auto-commit.sh` | unknown | SURFACE | 0 | R3 | settings.json hook ref. Batch commit. |
| 18 | `guidance-hook.sh` | unknown | SURFACE | 0 | R3 | Older, deprecated by guidance-hooks.sh. |
| 19 | `health-monitor.sh` | unknown | NOT TOUCHED | 0 | - | |
| 20 | `daemon-manager.sh` | unknown | NOT TOUCHED | 0 | - | |
| 21 | `ddd-tracker.sh` | unknown | NOT TOUCHED | 0 | - | |
| 22 | `memory.js` | unknown | NOT TOUCHED | 0 | - | |
| 23 | `router.js` | unknown | NOT TOUCHED | 0 | - | |
| 24 | `metrics-db.mjs` | unknown | NOT TOUCHED | 0 | - | |
| 25 | `quick-start.sh` | unknown | NOT TOUCHED | 0 | - | |
| 26 | `setup-mcp.sh` | unknown | NOT TOUCHED | 0 | - | |
| 27 | `github-safe.js` | unknown | NOT TOUCHED | 0 | - | |
| 28 | `github-setup.sh` | unknown | NOT TOUCHED | 0 | - | |
| 29 | `security-scanner.sh` | unknown | NOT TOUCHED | 0 | - | |
| 30 | `shell-safety-check.sh` | unknown | NOT TOUCHED | 0 | - | |
| 31 | `lib/safe-run.sh` | unknown | NOT TOUCHED | 0 | - | |
| 32 | `learning-optimizer.sh` | unknown | NOT TOUCHED | 0 | - | |
| 33 | `v3-quick-status.sh` | unknown | NOT TOUCHED | 0 | - | |
| 34 | `sync-v3-metrics.sh` | unknown | NOT TOUCHED | 0 | - | |
| 35 | `update-v3-progress.sh` | unknown | NOT TOUCHED | 0 | - | |
| 36 | `worker-manager.sh` | unknown | NOT TOUCHED | 0 | - | |
| 37 | `perf-worker.sh` | unknown | NOT TOUCHED | 0 | - | |
| 38 | `validate-v3-config.sh` | unknown | NOT TOUCHED | 0 | - | |
| 39 | `v3.sh` | unknown | NOT TOUCHED | 0 | - | |
| 40 | `adr-compliance.sh` | unknown | NOT TOUCHED | 0 | - | |
| 41 | `standard-checkpoint-hooks.sh` | unknown | NOT TOUCHED | 0 | - | |
| 42 | `statusline.cjs` | unknown | NOT TOUCHED | 0 | - | Deprecated (ADR-018). |
| 43 | `statusline.js` | unknown | NOT TOUCHED | 0 | - | Deprecated (ADR-018). |
| 44 | `statusline-hook.sh` | unknown | NOT TOUCHED | 0 | - | Deprecated (ADR-018). |
| 45 | `statusline.mjs` | unknown | NOT TOUCHED | 0 | - | Deprecated (ADR-018). |

### 8. Agent Templates (`.claude/agents/`)

| # | File | Lines | Depth | Round | Key Findings |
|---|------|-------|-------|-------|-------------|
| 1 | `v3-queen-coordinator.md` | 82 | DEEP | R1 | Queen persona. TS coordination code (NOT executed). Task decomposition prompt. |
| 2 | `hierarchical-coordinator.md` | 717 | DEEP | R1 | Hierarchical patterns. Worker delegation prompt engineering. |
| 3 | `mesh-coordinator.md` | 970 | DEEP | R1 | Peer-to-peer mesh. Distributed decision making. |
| 4 | `adaptive-coordinator.md` | 1,133 | DEEP | R1 | Dynamic topology switching. MoE attention concept. |
| 5 | `collective-intelligence-coordinator.md` | 1,002 | DEEP | R1 | Byzantine fault-tolerant consensus. Attention-based coordination. |
| 6-95 | (90 other agents) | varies | NOT TOUCHED | - | 95 total valid agents. Only coordinators analyzed. |

### 9. Configuration

| # | File | Lines | Depth | Lines Read | Round | Key Findings |
|---|------|-------|-------|------------|-------|-------------|
| 1 | `.claude/settings.json` | 533 | DEEP | Full | R3 | 44 hook scripts. 7 SessionStart, 5 SessionEnd, 6 PreToolUse[Task], 8 PostToolUse[Task], 3 UserPromptSubmit, 3 Stop. |
| 2 | `CLAUDE.md` | 239 | DEEP | Full | R1 | Project config, behavioral rules, swarm activation, model routing, triggers. |
| 3 | `.claude.json` | unknown | SURFACE | 0 | - | MCP config (ADR-012). |

### 10. Skills (`.claude/skills/`)

| # | Directory | Depth | Round | Notes |
|---|-----------|-------|-------|-------|
| 1 | `swarm-orchestration/` | DEEP | R3 | Full SKILL.md read. Topologies, task patterns, CLI examples. |
| 2 | `agentdb-vector-search/` | DEEP | R3 | HNSW search patterns. |
| 3 | `agentdb-advanced/` | SURFACE | R3 | Categorized by name. |
| 4 | `agentdb-learning/` | SURFACE | R3 | Categorized by name. |
| 5 | `agentdb-optimization/` | SURFACE | R3 | Categorized by name. |
| 6-33 | (28 others) | NOT TOUCHED | - | Listed by directory name only. |

### 11. Commands (`.claude/commands/`)

| # | Category | Count | Depth | Round | Notes |
|---|----------|-------|-------|-------|-------|
| 1 | `github/` | 20 | SURFACE | R3 | Directory listing. |
| 2 | `sparc/` | 30+ | SURFACE | R3 | Directory listing. |
| 3 | `analysis/` | 6 | SURFACE | R3 | Directory listing. |
| 4 | `automation/` | 7 | SURFACE | R3 | Directory listing. |
| 5 | `monitoring/` | 6 | SURFACE | R3 | Directory listing. |
| 6 | `optimization/` | 5 | SURFACE | R3 | Directory listing. |
| 7 | `hooks/` | 7 | SURFACE | R3 | Directory listing. |
| 8 | Root | 3 | SURFACE | R3 | help, memory, swarm. |

### 12. Marketplace Plugins (`.claude-flow/plugins/`)

| # | Item | Depth | Round | Notes |
|---|------|-------|-------|-------|
| 1-19 | (19 marketplace plugins) | NOT TOUCHED | - | Listed by directory. Runtime plugins separate from implementation (section 16). |

### 13. Init System (`dist/src/init/`)

| # | File | Lines | Depth | Lines Read | Round | Key Findings |
|---|------|-------|-------|------------|-------|-------------|
| 1 | `executor.js` | 1,762 | NOT TOUCHED | 0 | R3-disc | Main init orchestrator. Largest init file. |
| 2 | `statusline-generator.js` | 1,310 | NOT TOUCHED | 0 | R3-disc | Generates statusline helper scripts. |
| 3 | `helpers-generator.js` | 998 | NOT TOUCHED | 0 | R3-disc | Generates helper scripts during `claude-flow init`. |
| 4 | `claudemd-generator.js` | 485 | NOT TOUCHED | 0 | R3-disc | Generates CLAUDE.md project config. |
| 5 | `settings-generator.js` | 283 | NOT TOUCHED | 0 | R3-disc | Generates settings.json. |
| 6 | `types.js` | 257 | NOT TOUCHED | 0 | R3-disc | Type definitions for init system. |
| 7 | `mcp-generator.js` | 99 | NOT TOUCHED | 0 | R3-disc | Generates MCP configuration. |
| 8 | `index.js` | ~100 | NOT TOUCHED | 0 | R3-disc | Module barrel export. |

### 14. Production Infrastructure (`dist/src/production/`)

| # | File | Lines | Depth | Lines Read | Round | Key Findings |
|---|------|-------|-------|------------|-------|-------------|
| 1 | `monitoring.js` | 355 | NOT TOUCHED | 0 | R3-disc | Production monitoring/observability. |
| 2 | `error-handler.js` | 298 | NOT TOUCHED | 0 | R3-disc | Production error handling. |
| 3 | `circuit-breaker.js` | 240 | NOT TOUCHED | 0 | R3-disc | Circuit breaker pattern for resilience. |
| 4 | `rate-limiter.js` | 200 | NOT TOUCHED | 0 | R3-disc | Rate limiting for API/resource protection. |
| 5 | `retry.js` | 178 | NOT TOUCHED | 0 | R3-disc | Retry with backoff logic. |
| 6 | `index.js` | ~100 | NOT TOUCHED | 0 | R3-disc | Module barrel export. |

### 15. Transfer System (`dist/src/transfer/`)

| # | File | Lines | Depth | Lines Read | Round | Key Findings |
|---|------|-------|-------|------------|-------|-------------|
| 1 | `ipfs-client.js` | ~400 | NOT TOUCHED | 0 | R3-disc | IPFS integration client. |
| 2 | `ipfs-upload.js` | ~300 | NOT TOUCHED | 0 | R3-disc | IPFS file upload. |
| 3 | `seraphine-model.js` | 372 | NOT TOUCHED | 0 | R3-disc | **Seraphine federated learning model**. |
| 4 | `pii-anonymizer.js` | 174 | NOT TOUCHED | 0 | R3-disc | PII detection and anonymization. |
| 5 | `pii-detector.js` | ~200 | NOT TOUCHED | 0 | R3-disc | PII pattern detection. |
| 6 | `pattern-store/discovery.js` | ~300 | NOT TOUCHED | 0 | R3-disc | Pattern discovery from store. |
| 7 | `pattern-store/download.js` | ~200 | NOT TOUCHED | 0 | R3-disc | Pattern download. |
| 8 | `pattern-store/publish.js` | ~200 | NOT TOUCHED | 0 | R3-disc | Pattern publishing. |
| 9 | `pattern-store/registry.js` | ~200 | NOT TOUCHED | 0 | R3-disc | Pattern registry management. |
| 10 | `pattern-store/search.js` | ~200 | NOT TOUCHED | 0 | R3-disc | Pattern search. |
| 11 | `gcs-storage.js` | ~200 | NOT TOUCHED | 0 | R3-disc | Google Cloud Storage integration. |
| 12 | `cfp-serializer.js` | ~200 | NOT TOUCHED | 0 | R3-disc | CFP format serialization. |
| 13-19 | (7 other files) | ~1,400 | NOT TOUCHED | 0 | R3-disc | Remaining transfer utilities (index, types, etc.) |

### 16. Plugins Implementation (`dist/src/plugins/`)

| # | File | Lines | Depth | Lines Read | Round | Key Findings |
|---|------|-------|-------|------------|-------|-------------|
| 1 | `store/discovery.js` | 1,146 | NOT TOUCHED | 0 | R3-disc | **IPFS plugin discovery**. Largest plugin file. |
| 2 | `manager.js` | 382 | MEDIUM | ~100 | R3 | Install/uninstall/enable. Manifest at `.claude-flow/plugins/installed.json`. |
| 3 | `store/search.js` | 229 | NOT TOUCHED | 0 | R3-disc | Plugin search functionality. |
| 4 | `store/types.d.ts` | ~100 | SURFACE | ~10 | R3 | 8 plugin types. |
| 5-8 | (4 other files) | ~600 | NOT TOUCHED | 0 | R3-disc | Tests, index, utilities. |

### 17. Update System (`dist/src/update/`)

| # | File | Lines | Depth | Lines Read | Round | Key Findings |
|---|------|-------|-------|------------|-------|-------------|
| 1 | `checker.js` | ~200 | NOT TOUCHED | 0 | R3-disc | Version check logic. |
| 2 | `executor.js` | ~200 | NOT TOUCHED | 0 | R3-disc | Update execution. |
| 3 | `validator.js` | ~100 | NOT TOUCHED | 0 | R3-disc | Update validation. |
| 4 | `rate-limiter.js` | ~100 | NOT TOUCHED | 0 | R3-disc | Update rate limiting. |
| 5 | `index.js` | ~50 | NOT TOUCHED | 0 | R3-disc | Module barrel export. |

### 18. Root Source Files (`dist/src/`)

| # | File | Lines | Depth | Lines Read | Round | Key Findings |
|---|------|-------|-------|------------|-------|-------------|
| 1 | `prompt.js` | 500 | NOT TOUCHED | 0 | R3-disc | Prompt construction/templating. |
| 2 | `parser.js` | 376 | NOT TOUCHED | 0 | R3-disc | Command/input parsing. |
| 3 | `config-adapter.js` | 185 | NOT TOUCHED | 0 | R3-disc | Configuration adaptation layer. |
| 4 | `suggest.js` | 199 | NOT TOUCHED | 0 | R3-disc | Command/action suggestions. |
| 5 | `types.js` | 37 | NOT TOUCHED | 0 | R3-disc | Core type definitions. |

### 19. Infrastructure (`dist/src/infrastructure/`)

| # | File | Lines | Depth | Lines Read | Round | Key Findings |
|---|------|-------|-------|------------|-------|-------------|
| 1 | `in-memory-repositories.js` | 263 | NOT TOUCHED | 0 | R3-disc | In-memory repo implementations. |

### 20. Benchmarks (`dist/src/benchmarks/`)

| # | File | Lines | Depth | Lines Read | Round | Key Findings |
|---|------|-------|-------|------------|-------|-------------|
| 1 | `pretrain-benchmark.js` | 403 | NOT TOUCHED | 0 | R3-disc | Pretrain performance benchmarking. |

### 21. Guidance Package (`node_modules/@claude-flow/guidance/`)

| # | File | Depth | Round | Key Findings |
|---|------|-------|-------|-------------|
| 1 | `wasm-pkg/guidance_kernel_bg.wasm` | SURFACE | R3 | 94.3KB binary. Exports from JS wrapper. |
| 2 | `wasm-pkg/guidance_kernel.js` | MEDIUM | R3 | batch_process, content_hash, detect_destructive, hmac_sha256, scan_secrets, kernel_init. |
| 3 | `guidance-provider.js` | MEDIUM | R3 | 350 lines. Session context, pre-edit gates, routing guidance, security patterns. |

### 22. Source TypeScript (`src/agentdb-integration/`)

| # | Category | Files | Depth | Notes |
|---|----------|-------|-------|-------|
| 1 | All 55 .ts files | 55 | NOT TOUCHED | Analyzed in ADR-020/028/037, not in swarm research. |

### 23. Test Files (`tests/agentdb-integration/`)

| # | Category | Files | Depth | Notes |
|---|----------|-------|-------|-------|
| 1 | All 21 .ts test files | 21 | NOT TOUCHED | Tested via ADR work, not in swarm research. |

---

## GAPS: Priority Queue for Next Research

### HIGH PRIORITY (affect swarm/routing understanding)

| File | Why | Est. Lines |
|------|-----|-----------|
| `bin/cli.js` deep read | 4,368 lines, only ~10 read. Full command registration, argument parsing, plugin loading. | 4,368 |
| `bin/mcp-server.js` deep read | 5,116 lines, only ~190 read. Full MCP server internals beyond JSON-RPC scaffold. | 5,116 |
| `init/executor.js` | **1,762 lines**. Main init orchestrator — what does `claude-flow init` actually create? | 1,762 |
| 7 ruvector modules (semantic-router, q-learning, coverage, diff-classifier, ast-analyzer, graph-analyzer, vector-db) | Complete routing intelligence picture | ~2,000 |
| `worker-queue.js` full internals | Job scheduling, priority, DLQ details | 511 |
| `browser-tools.js` handlers | 23 tools, `agent-browser` subprocess | ~400 |
| `daa-tools.js` handlers | Adaptive agent creation/adaptation | ~300 |

### MEDIUM PRIORITY (broader architecture)

| File | Why | Est. Lines |
|------|-----|-----------|
| `init/statusline-generator.js` | 1,310 lines. How statusline helpers are generated. | 1,310 |
| `init/helpers-generator.js` | 998 lines. How helper scripts are generated during init. | 998 |
| `transfer/seraphine-model.js` | 372 lines. **Federated learning model** — unknown architecture. | 372 |
| `plugins/store/discovery.js` | 1,146 lines. IPFS plugin discovery — potential real distributed feature. | 1,146 |
| `production/` all 6 files | Circuit breaker, monitoring, rate limiter — production infrastructure. | 1,171 |
| `security-tools.js` handlers | AIDefence scan/analyze/learn | ~500 |
| `neural-tools.js` handlers | train/predict/compress | ~300 |
| `github-tools.js` handlers | GitHub integration | ~300 |
| `workflow-tools.js` handlers | Workflow engine | ~300 |
| `performance-tools.js` handlers | Performance profiling | ~300 |
| `learning-service.mjs` full | HNSW learning Node.js module | ~300 |
| 25 NOT TOUCHED helpers | Runtime library gaps | ~2,500 |

### LOW PRIORITY (completeness)

| Category | Count | Why |
|----------|-------|-----|
| 90 agent templates | 90 | Persona definitions, mostly prompt engineering. |
| 28 skills | 28 | Domain knowledge modules. |
| 80+ commands | 80+ | Prompt templates. |
| 19 marketplace plugins | 19 | External integrations. |
| `update/` 5 files | 5 | Version checking (low swarm relevance). |
| `transfer/` remaining 15 files | 15 | PII, IPFS, pattern store (low swarm relevance). |

---

## UNKNOWN FILES (not yet inventoried)

R3 discovery agent resolved most unknowns. Remaining gaps:

| Location | Status |
|----------|--------|
| `dist/src/guidance/` | **UNKNOWN** — may exist separately from npm package |
| `dist/src/` other subdirs | **PARTIALLY RESOLVED** — init/, production/, transfer/, plugins/, update/, infrastructure/, benchmarks/ now inventoried. Others may exist. |
| `bin/` other scripts | **RESOLVED** — only cli.js (4,368 lines) and mcp-server.js (5,116 lines) found |

## LINE COUNT CORRECTIONS (R3 Discovery)

| File | Old Estimate | Actual Lines | Delta |
|------|-------------|-------------|-------|
| `bin/cli.js` | ~200 | 4,368 | +4,168 |
| `bin/mcp-server.js` | 190 | 5,116 | +4,926 |

These two files alone account for 9,484 lines — 49x more than originally estimated.
