# Process Spawning Domain Analysis

> **Priority**: HIGH | **Coverage**: 100% (33/33 files) | **Status**: CLOSED
> **Last updated**: 2026-02-14 (Session R22) | Fully closed with terminal-tools.js deep-read

## Overview

The process-spawning domain covers how claude-flow creates, manages, and terminates background processes — workers, daemons, headless agents, and container pools. 33 files / 7.6K LOC spanning 4 packages.

## Architecture

Three real execution modes exist:

1. **Headless Worker Executor** (`headless-worker-executor.js`, 999 LOC) — spawns actual `claude` CLI processes
2. **Container Worker Pool** (`container-worker-pool.js`, 407 LOC) — manages Docker containers
3. **Hive-Mind Launcher** — spawns interactive Claude with Byzantine consensus prompts

Two-layer worker management:
- **daemon-manager.sh** — high-level daemons (swarm monitor, metrics)
- **worker-manager.sh** — 7 domain-specific workers (perf, health, patterns, DDD, ADR, security, learning)

## Key Files

| File | Package | LOC | Depth | Role |
|------|---------|-----|-------|------|
| `dist/src/services/headless-worker-executor.js` | claude-flow-cli | 999 | DEEP | Real process spawning |
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

## Terminal Tools (Added R22)

`terminal-tools.js` (246 LOC) — **STATE TRACKING ONLY**. 5 MCP tools (create, execute, list, close, history) that record commands to JSON but never execute them. `terminal_execute` returns fake random duration `Math.random()*100+10`. Explicitly documented as non-executing.

## Architecture Issues

1. **process.js vs daemon.js** — Two overlapping commands with vastly different quality. `process.js` is entirely fabricated; `daemon.js` is production-ready. Consolidation needed.
2. **terminal_execute misleading** — Tool name implies command execution but only does state tracking. Users calling this MCP tool expecting execution will get no results.
