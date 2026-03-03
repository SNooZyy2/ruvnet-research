# Process Spawning Domain Analysis

> **Priority**: HIGH | **Coverage**: 100% (34/34 files) | **Status**: CLOSED
> **Last updated**: 2026-03-03 (Session R140) | Updated with V3 TypeScript headless executor deep-read

## Overview

The process-spawning domain covers how claude-flow creates, manages, and terminates background processes — workers, daemons, headless agents, and container pools. 34 files / ~9K LOC spanning 4 packages. The V3 TypeScript rewrite of headless-worker-executor.ts (1,342 LOC) is now deep-read (R140), confirming the execution mechanism: all workers run as `claude --print <prompt>` subprocesses with no MCP protocol, no AgentDB connection, and no bidirectional messaging.

## Architecture

Three-tier execution chain (V3 confirmed):
1. **ContainerWorkerPool** (`container-worker-pool.ts`) — outer Docker CLI layer; REAL integration but has CRITICAL BUG: prompt and contextPatterns silently dropped in buildWorkerCommand()
2. **worker-daemon.ts** — NOT a daemon; foreground class; setTimeout-based scheduling; 9/12 local worker types are FACADE stubs
3. **HeadlessWorkerExecutor** (`headless-worker-executor.ts`, 1,342 LOC) — real subprocess spawn via `child_process.spawn('claude', ['--print', prompt])`; 78-83% genuine

Four concrete execution modes:
1. **Headless Worker Executor** (V2: `headless-worker-executor.js` 999 LOC; V3: `.ts` 1,342 LOC) — spawns actual `claude` CLI processes; maxConcurrent=2; pending queue
2. **Container Worker Pool** (`container-worker-pool.js`/`.ts`, 407 LOC) — manages Docker containers
3. **Hive-Mind Launcher** — spawns interactive Claude with Byzantine consensus prompts
4. **claim-service.ts** — LOCAL-ONLY JSON file claims; incompatible with MCP claims-tools.ts (2-part vs 3-part claimant key formats)

Two-layer shell worker management (V2 only):
- **daemon-manager.sh** — high-level daemons (swarm monitor, metrics)
- **worker-manager.sh** — 7 domain-specific workers (perf, health, patterns, DDD, ADR, security, learning)

## Key Files

| File | Package | LOC | Depth | Role |
|------|---------|-----|-------|------|
| `v3/@claude-flow/cli/src/services/headless-worker-executor.ts` | claude-flow V3 | 1,342 | DEEP | V3 real process spawning (R140) |
| `dist/src/services/headless-worker-executor.js` | claude-flow-cli | 999 | DEEP | V2 real process spawning |
| `dist/src/services/worker-daemon.js` | claude-flow-cli | 756 | DEEP | Background daemon service |
| `dist/src/commands/process.js` | claude-flow-cli | 641 | DEEP | **STUB** — all metrics fabricated |
| `dist/src/commands/daemon.js` | claude-flow-cli | 593 | DEEP | Real daemon mgmt with security |
| `dist/src/runtime/headless.js` | claude-flow-cli | 284 | DEEP | CLI runtime for headless mode |
| `helpers/daemon-manager.sh` | claude-config | 253 | DEEP | Daemon lifecycle |
| `helpers/worker-manager.sh` | claude-config | 206 | DEEP | 7-worker orchestration |
| `helpers/perf-worker.sh` | claude-config | 170 | DEEP | Performance benchmarking |
| `dist/utils/agentBoosterPreprocessor.js` | agentic-flow | 271 | DEEP | Code transform preprocessor |

## CRITICAL Findings (11)

1. **Fabricated monitoring in process.js** — All metrics use `Math.random()`. Worker list hardcoded. Log viewer generates random entries.
2. **Benchmark fabrication** — SONA/Flash Attention claims based on synthetic random vectors.
3. **Security: unverified npm execution** — `npx --yes agent-booster@0.2.2` auto-downloads without verification.
4. **Security: process killing** — `pgrep -f` could kill unrelated user processes.
5. **Silent worker failures** — All output to `/dev/null`, failures undetectable.
6. **Missing dependencies** — daemon-manager.sh references scripts without existence checks.
7. **ADR-016 workaround** — agentic-flow `require()` patched to skip `main()`.
8. **Simulated billing** — Fake Stripe/PayPal/Crypto transaction IDs.
9. **Webhook security bypass** — Accepts ANY non-empty string as valid signature.
10. **No claude binary validation** — Executes without checking binary exists.
11. **Hardcoded model routing** — All workers use `model=sonnet`, ignoring ADR-008.

## HIGH Findings (8)

1. **Real security in daemon.js** — Path validation, argument-array spawning.
2. **Three real execution modes** are genuine implementations.
3. **process.js entirely stub** — PID-only start, fake spawn, fake logs.
4. **Agent Booster 70% non-functional** — 5/7 extractors return null.
5. **deep benchmark stub** — Only checks package.json availability.
6. **MCP detection unreliable** — `ps aux | grep` false positives.
7. **Real spawn.js** — MCP tool genuinely executes `npx claude-flow@alpha`.
8. **daemon.js security** — Production-grade path validation and injection prevention.

## R140 Findings: V3 HeadlessWorkerExecutor Deep-Read

**HIGH — SECURITY** (`headless-worker-executor.ts`): Prompt passed directly as a CLI argument `spawn("claude", ["--print", prompt])`. No sanitization of prompt content. If user-controlled content reaches the prompt parameter, CLI argument injection is possible. (lines: varies)

**HIGH — ARCHITECTURE** (`headless-worker-executor.ts`): HeadlessWorkerExecutor does NOT use MCP protocol to communicate with workers. Workers are invoked as dumb subprocess calls via `claude --print`. There is no bidirectional message passing, no tool call intercept, and no structured response parsing beyond raw stdout capture.

**HIGH — INTEGRATION** (`headless-worker-executor.ts`): Class has zero imports from memory, AgentDB, ruvector, or any vector search backend. No session context is passed to spawned workers. Workers cannot read or write AgentDB from within the execution path — the memory disconnect is architectural, not accidental.

**HIGH — BUG** (`headless-worker-executor.ts`): DOUBLE TIMEOUT BUG — `executeClaudeCode()` sets TWO independent timeouts for the same process: one at line 1133 (`timeoutHandle` via `setTimeout`) and a second at line 1209 (`timeoutMs + 5000`). The second fires 5 seconds after the first, sending SIGKILL to an already-SIGTERM'd process. No-op in practice but indicates untested code paths.

**MEDIUM — ARCHITECTURE** (`headless-worker-executor.ts`): `MODEL_IDS` maps to hardcoded model version strings (`sonnet -> claude-sonnet-4-5-20250929`, `opus -> claude-opus-4-6`, `haiku -> claude-haiku-4-5-20251001`). These will silently break when Anthropic deprecates model version identifiers.

**MEDIUM — QUALITY** (`headless-worker-executor.ts`): `simpleGlob()` is a hand-rolled recursive directory scanner. Handles only `*.ext`, `prefix*`, `*suffix`, and `**` patterns. No brace expansion `{a,b}`, no negation `!`, no complex globs. Used for context file collection but misses files when patterns are complex.

**MEDIUM — ARCHITECTURE** (`headless-worker-executor.ts`): Of 8 headless worker types, only 2 are `enabled=true` by default: `audit` (30-min interval, haiku, strict sandbox) and `optimize` (60-min interval, sonnet, permissive). Workers `document`, `ultralearn`, `refactor`, `analyze`, `test`, and `review` are all disabled by default.

**INFO — GENUINE** (`headless-worker-executor.ts`): `executeClaudeCode()` spawns the claude CLI as a real subprocess via `child_process.spawn` with args `["--print", prompt]`. This is the confirmed actual mechanism behind headless worker execution — real, not a mock.

**INFO — GENUINE** (`headless-worker-executor.ts`): Process pool (`Map<string, PoolEntry>`) with configurable `maxConcurrent` (default: 2), pending queue (`QueueEntry[]`), and graceful timeout handling (SIGTERM then SIGKILL after 5s). Real concurrency management.

**INFO — GENUINE** (`headless-worker-executor.ts`): `buildContext()` reads files matching glob patterns from `projectRoot` using `simpleGlob()`. Limited to `maxContextFiles` (default: 20) and `maxCharsPerFile` (default: 5,000 chars). Context injected into prompt string before subprocess spawn. Real implementation.

**INFO — SECURITY** (`headless-worker-executor.ts`): `audit` worker `contextPatterns` includes `**/.env*` — this sends .env file contents to Claude AI as context. While the intent is to detect hardcoded secrets, the actual .env files (with real secrets) are uploaded to the Anthropic API as part of the prompt.

**INFO — SECURITY** (`headless-worker-executor.ts`): `logExecution()` writes full prompt content (including injected codebase context) to `.claude-flow/logs/headless/{executionId}_prompt.log`. Potentially sensitive file contents logged to disk. No log rotation or retention policy configured.

## Terminal Tools (Added R22)

`terminal-tools.js` (246 LOC) — **STATE TRACKING ONLY**. 5 MCP tools (create, execute, list, close, history) that record commands to JSON but never execute them. `terminal_execute` returns fake random duration `Math.random()*100+10`. Explicitly documented as non-executing.

## Architecture Issues

1. **process.js vs daemon.js** — Two overlapping commands with vastly different quality. `process.js` is entirely fabricated; `daemon.js` is production-ready. Consolidation needed.
2. **terminal_execute misleading** — Tool name implies command execution but only does state tracking. Users calling this MCP tool expecting execution will get no results.
3. **No MCP protocol between orchestrator and workers** (confirmed R140) — Workers are invoked as blind `claude --print` subprocesses. No structured tool call flow, no bidirectional messaging, no memory/AgentDB injection. Every spawned worker starts cold with zero project context beyond the injected prompt string.
4. **Memory disconnect is architectural** (R140) — HeadlessWorkerExecutor has zero imports from memory, AgentDB, or ruvector. Workers cannot persist findings or retrieve prior context. The entire multi-session "learning" capability advertised in marketing is absent from the execution layer.

## Session Log

### R22 (2026-02-14) — Domain closed
Deep-read of terminal-tools.js. Confirmed state-tracking-only MCP tools. Domain marked CLOSED at 33 files / 7.6K LOC.

### R140 (2026-03-03) — V3 TypeScript executor deep-read (ML-F)
Deep-read of `v3/@claude-flow/cli/src/services/headless-worker-executor.ts` (1,342 LOC, DEEP). Added to file registry. 12 findings inserted (4 HIGH, 3 MEDIUM, 5 INFO). Key revelation: confirmed execution mechanism is `spawn('claude', ['--print', prompt])` — no MCP protocol, no AgentDB connection. Three-tier chain confirmed: ContainerWorkerPool → worker-daemon → HeadlessWorkerExecutor. Double timeout bug discovered. Audit worker sends .env contents to Anthropic API. Coverage updated to 34 files.
