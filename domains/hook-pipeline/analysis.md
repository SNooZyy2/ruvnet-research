# Hook Pipeline Domain Analysis

> **Priority**: HIGH | **Coverage**: 98.1% DEEP (101/103 files) | **Status**: CLOSED
> **Last updated**: 2026-02-14 (Session R17 — Tier 2 Closeout)

## Overview

The hook pipeline is the **nervous system** of claude-flow — connecting CLI commands, MCP tools, learning, model routing, and worker dispatch. 103 files / 25.6K LOC. 101 files DEEP-read.

## Architecture (Updated R14)

Six layers, with the intelligence layer now confirmed REAL:

| Layer | Components | LOC | Status |
|-------|-----------|-----|--------|
| **CLI Presentation** | hooks.js (3,672 LOC main) | ~4.6K | REAL (delegates to MCP tools) |
| **CLI Commands** | dist/cli/commands/hooks.js (923 LOC) | ~1K | REAL (19 subcommands via Commander.js) |
| **MCP Tool Layer** | 17 hook tool files in dist/mcp/fastmcp/tools/hooks/ | ~3.5K | REAL (all tools functional) |
| **Intelligence Layer** | intelligence-bridge.js (1,038), intelligence-tools.js (425) | ~1.5K | REAL (was thought non-existent) |
| **Shell Layer** | checkpoint-hooks.sh, guidance-hooks.sh, ralph-loop stop-hook.sh | ~0.5K | REAL (production git integration) |
| **Helper Layer** | intelligence.cjs (917), learning-service.mjs (1,145) | ~2.5K | REAL (PageRank, HNSW, SQLite) |

## R14 CRITICAL RESOLUTION: intelligence-bridge.js EXISTS

The most important R14 finding: **intelligence-bridge.js (1,038 LOC) exists and is REAL**. The model-routing domain analysis (R11) had catalogued this as non-existent, creating a CRITICAL finding. R14 deep-read confirms:

- **routeTaskIntelligent()** at L382 — EXISTS and WORKING
- **findSimilarPatterns()** at L542 — EXISTS and WORKING
- All 10 functions imported by intelligence-tools.js are confirmed present

### intelligence-bridge.js Architecture (6 Layers)

```
┌─────────────────────────────────────────────────┐
│ Layer 1: TensorCompress Tiering (L55-204)       │
│ - hot/warm/cool/cold/archive compression       │
│ - 50% to 96.9% savings (claimed)               │
│ - Lazy-loaded from ruvector, graceful fallback  │
├─────────────────────────────────────────────────┤
│ Layer 2: Multi-Algorithm Learning (L206-320)    │
│ - 9 RL algorithms per task type                 │
│ - double-q, sarsa, actor-critic, ppo, etc.     │
│ - Optional ruvector dependency with fallback    │
├─────────────────────────────────────────────────┤
│ Layer 3: SONA + MoE + HNSW Routing (L374-431)  │
│ - routeTaskIntelligent() — O(log n) with HNSW  │
│ - Micro-LoRA enabled (~0.05ms claimed)         │
│ - HNSW DISABLED by default (L347)              │
├─────────────────────────────────────────────────┤
│ Layer 4: Trajectory Tracking (L441-504)         │
│ - beginTaskTrajectory, record steps, end       │
│ - In-memory Map + SQLite persistence           │
│ - EWC++ consolidation on trajectory end        │
├─────────────────────────────────────────────────┤
│ Layer 5: Pattern Storage & Search (L509-569)    │
│ - storePattern with tiered compression         │
│ - findSimilarPatterns via HNSW (when enabled)  │
├─────────────────────────────────────────────────┤
│ Layer 6: Parallel Execution (L621-1037)         │
│ - 7 worker threads via IntelligenceEngine      │
│ - Batch Q-learning (3-4x claimed speedup)      │
│ - Extended pool: AST, security, git, RAG       │
└─────────────────────────────────────────────────┘
```

**Key caveat**: HNSW explicitly disabled at L347 (`enableHnsw: false, // API compatibility issue`). The 150x faster claim is undermined — it falls back to brute-force similarity.

## R14: Hook Infrastructure Deep-Read

### SDK hooks-bridge.js (235 LOC) — REAL

Bridges agentic-flow intelligence with Claude Agent SDK native hooks:
- **7 hook types**: PreToolUse, PostToolUse, PostToolUseFailure, SessionStart, SessionEnd, SubagentStart, SubagentStop
- **Trajectory lifecycle**: Begins tracking on PreToolUse, ends on PostToolUse/Failure
- **TTL cleanup**: 5-minute max lifetime, cleanup runs every 2 minutes via setInterval().unref()
- **Lazy loading**: intelligence-bridge.js imported only on first use (circular dependency protection)

### shared.js (159 LOC) — REAL Utilities

Core utilities used by all hook tools:
- **Agent mapping**: 18 file types → agent types (L50-91)
- **simpleEmbed()**: Hash-based 64-dim embedding (deterministic, NOT neural) (L94-111)
- **cosineSimilarity()**: Standard dot-product implementation (L113-126)
- **dangerousPatterns**: 10 regex patterns for command safety (rm -rf /, fork bomb, curl|bash, etc.) (L128-139)
- **assessCommandRisk()**: Risk score 0-0.9 with graduated severity

### CLI hooks.js (923 LOC) — REAL

Commander.js-based CLI registering 19 subcommands:
- 10 hook tools: pre/post-edit, pre/post-command, route, explain, pretrain, build-agents, metrics, transfer
- 9 intelligence commands: route, trajectory-start/step/end, pattern-store/search, stats, learn, attention
- Init command generates .claude/settings.json and statusline.mjs (539 LOC embedded)

### hooks-server.js (63 LOC) — REAL MCP Server

FastMCP stdio server registering all 19 hook tools for Claude Code MCP integration.

### index.js (59 LOC) — REAL Registry

Exports hookTools (10) and allHookTools (19) arrays. Re-exports 10 intelligence bridge functions.

## R14: MCP Hook Tools Deep-Read (8 files)

All 8 MCP hook tools are **REAL, working implementations** with no fabrication:

| File | LOC | MCP Tool | Key Algorithm |
|------|-----|----------|---------------|
| **build-agents.js** | 276 | hook_build_agents | Language detection → YAML/JSON agent config generation. 5 focus modes. |
| **transfer.js** | 151 | hook_transfer | Cross-project pattern transfer: replace (0.7x), additive (0.5x), merge (0.6/0.4 weighted). |
| **post-edit.js** | 146 | hook_post_edit | Q-learning: Q(s,a) += 0.1*(reward - Q(s,a)). RuVector trajectory integration. |
| **metrics.js** | 137 | hook_metrics | Learning dashboard: accuracy, velocity (trend detection), health thresholds. |
| **pre-edit.js** | 121 | hook_pre_edit | 3-tier agent selection: file type → Q-table pattern → directory override. |
| **benchmark.js** | 110 | (harness) | 5 warmup + 100 iterations, P50/P95/P99 percentiles, latency targets. |
| **post-command.js** | 91 | hook_post_command | Error extraction (ENOENT, EACCES, TypeError), Q-learning update. |
| **pre-command.js** | 70 | hook_pre_command | 10-pattern risk assessment, 0-0.9 scale, safer alternatives. |

### Q-Learning Across Hook Tools

Three hooks implement the same Q-learning update:
- **pre-edit.js**: Agent selection based on learned Q-values
- **post-edit.js**: Updates Q-values with success (reward=1.0) or failure (reward=-0.3)
- **post-command.js**: Same formula for command outcomes

All use learning_rate=0.1 and the standard tabular Q-update. Not deep RL.

### intelligence-tools.js (425 LOC) — 9 MCP Tools

Wraps intelligence-bridge.js functions as MCP tools with Zod schema validation:
1. intelligenceRouteTool, 2. trajectoryStartTool, 3. trajectoryStepTool, 4. trajectoryEndTool,
5. patternStoreTool, 6. patternSearchTool, 7. statsTool, 8. learnTool, 9. attentionTool

### pretrain.js (171 LOC) — Repository Bootstrap

Four-phase pretrain: git ls-files structure analysis, git history co-edit patterns, key file memory creation, directory-agent mappings.

### explain.js (164 LOC) — Routing Explainer

Five-factor analysis: file patterns, task keywords (1.5x weight), memory similarity (cosine, threshold 0.3), error patterns, weighted scoring returning top 5 agents.

## R14: Shell Scripts & Templates Deep-Read

### Production-Grade Shell Scripts

| File | LOC | Status | Key Feature |
|------|-----|--------|-------------|
| **standard-checkpoint-hooks.sh** | 190 | REAL | Git checkpoint system: stash, tags, metadata JSON, session summaries |
| **ralph-loop stop-hook.sh** | 178 | REAL | Prevents premature session exit, JSONL parsing, atomic file updates |
| **guidance-hooks.sh** | 109 | REAL | 4 handlers: security detection, edit logging, dangerous command detection (ADR-016), keyword routing |
| **guidance-hook.sh** | 14 | REAL | Thin wrapper delegating to npx agentic-flow@alpha |
| **statusline-hook.sh** | 23 | DEPRECATED | Per ADR-023, references absent statusline.cjs |

### @claude-flow/guidance hooks.js (347 LOC) — REAL TypeScript

GuidanceHookProvider class registering 5 hooks with priority ordering:
1. PreCommand (Critical) — guidance gate evaluation
2. PreToolUse (Critical) — tool allowlist + secrets
3. PreEdit (High) — diff size + secrets
4. PreTask (Normal) — shard retrieval based on classified intent
5. PostTask (Normal) — ledger finalization

Severity ordering: block > require-confirmation > warn > allow.

### Documentation Files

7 hook command .md files (session-end, post-edit, pre-edit, post-task, pre-task, setup, overview) are specifications/documentation, not implementations. Total ~737 LOC of spec.

3 SKILL.md files provide comprehensive reference:
- **hooks-automation** (1,202 LOC): 20+ hook types with JSON config examples
- **hook-development** (713 LOC): Plugin vs settings format, lifecycle limitations (hooks require session restart to reload)
- **hookify writing-rules** (375 LOC): Rule markdown format with YAML frontmatter, 6 operators

## R14: Parallel Validation (167 LOC) — REAL

Validates parallel execution with 6 checks:
1. Sequential subprocess spawning (-0.3 penalty)
2. Missing ReasoningBank (-0.2)
3. Small batch size (-0.1)
4. No QUIC transport (-0.15)
5. No result synthesis (-0.15)
6. No pattern storage (recommendation only)

Grades A (0.9+) through F (<0.4). All regex-based, deterministic.

## R14: ReasoningBank Hooks (179 LOC) — REAL

- **pre-task.js** (69 LOC): Retrieves relevant memories for prompt injection via retrieveMemories() + formatMemoriesForPrompt()
- **post-task.js** (110 LOC): Three-step learning pipeline: judgeTrajectory() → distillMemories() → consolidate()

## CRITICAL Findings (5, updated R14)

1. **Token stats fabricated** (hooks.js L3236-3239) — `totalTokensSaved += 200`, `cacheHits = 2`. Users see fake optimization metrics.
2. **Pattern count from file size** (hooks.js L2651-2656) — `Math.floor(sizeKB / 2)`. Not actual patterns.
3. **Anti-drift config hardcoded** (hooks.js L3202-3214) — Not learned values.
4. **MCP tools degrade to no-ops** when dependencies missing.
5. **intelligence-bridge.js EXISTS** (RESOLVED) — Previously catalogued as non-existent. All functions present and working. This was a synchronization error in research, not a code deficiency.

## HIGH Findings (15, +8 from R14)

1. **Domain completion fabricated** — from arbitrary pattern count thresholds.
2. **Intelligence % is filesystem check** — directory existence, not real intelligence.
3. **Fake progress animations** — `setTimeout(500-800ms)` to simulate processing.
4. **Agent Booster marketing mismatch** — "<1ms WASM" is actually npx subprocess.
5. **model-route output unread** — `[ROUTING DIRECTIVE]` text has no programmatic consumer.
6. **hook-handler.cjs route output fabricated** — Ignores actual routing result, outputs hardcoded tables with Math.random() latency.
7. **intelligence.cjs is genuinely real** — PageRank, edge building, consolidation.
8. **intelligence-bridge.js is 6-layer real architecture** (R14) — TensorCompress, Multi-Algorithm Learning, SONA+MoE+HNSW, Trajectory tracking, Pattern storage, Parallel workers.
9. **HNSW disabled by default** (R14) — L347: `enableHnsw: false` due to "API compatibility issue". 150x faster claim undermined.
10. **SDK hooks-bridge.js fully functional** (R14) — 7 hook types registered, TTL-based trajectory cleanup.
11. **All 8 MCP hook tools are REAL** (R14) — Q-learning, risk assessment, benchmarking, transfer, metrics all functional with no fabrication.
12. **GuidanceHookProvider production TypeScript** (R14) — 5 hooks with priority ordering, enforcement gate integration.
13. **standard-checkpoint-hooks.sh production-grade** (R14) — Real git checkpoints with metadata JSON.
14. **ralph-loop stop-hook.sh excellent quality** (R14) — Robust numeric validation, atomic file updates, Perl regex extraction.
15. **19 CLI subcommands + 19 MCP tools confirmed** (R14) — Complete command/tool surface area mapped.

## MEDIUM Findings (5, +3 from R14)

1. **TensorCompress optional** — Lazy-loaded from ruvector with graceful fallback to uncompressed.
2. **Multi-Algorithm Learning optional** — 9 algorithms routed by task type, falls back if ruvector unavailable.
3. **Shell injection potential** — pretrain.js uses execSync with git commands, no input sanitization.
4. **standard-checkpoint-hooks.sh edge case** — L81 uses `git diff HEAD~1` which fails on first commit.
5. **Hook lifecycle limitation** — Hooks load at session start, require restart to reload (documented in hook-development SKILL.md L572-599 but not in overview).

## Positive

- **intelligence-bridge.js** is a sophisticated 6-layer intelligence system with graceful degradation
- **All MCP hook tools** are genuine implementations with Q-learning, risk assessment, benchmarks
- **intelligence.cjs** implements actual PageRank with power iteration
- **learning-service.mjs** has working HNSW search with SQLite persistence
- **SDK hooks-bridge.js** properly integrates with Claude Agent SDK hook lifecycle
- **shared.js** provides solid utility foundation (agent mapping, embedding, similarity, risk assessment)
- **standard-checkpoint-hooks.sh** creates real git state snapshots
- **ralph-loop stop-hook.sh** has exceptional error handling and robustness
- **GuidanceHookProvider** is production TypeScript with proper priority ordering
- **ReasoningBank hooks** implement real judge→distill→consolidate learning pipeline

## Key Insight (Updated R14)

The hook pipeline is **more real than initially assessed**. R8 focused on the CLI/MCP presentation layer which has fabricated metrics. R14 revealed that the **underlying intelligence and tool implementations are genuine**:

- intelligence-bridge.js EXISTS and implements real routing, trajectory tracking, and pattern learning
- All 8 MCP hook tools implement functional Q-learning and risk assessment
- The SDK bridge properly manages trajectory lifecycle with TTL cleanup
- The shell scripts are production-grade with real git integration

The split personality remains: **presentation metrics are fabricated** (token stats, pattern counts, progress animations) while **core algorithms are real** (Q-learning, PageRank, HNSW, trajectory tracking, risk assessment).

## R16: Guidance Kernel WASM Source Investigation

### Source Located — No Longer Opaque

The `guidance_kernel_bg.wasm` binary was the last remaining OPAQUE artifact in the project.
**Source found**: `https://github.com/ruvnet/claude-flow` at `v3/@claude-flow/guidance/wasm-kernel/`

Note: The source is NOT in the ruvector repo (confirmed by checking all 76 crates on GitHub).
The `@claude-flow/guidance` package.json points to the claude-flow repo.

### Rust Source (4 files, ~800 LOC estimated)

| File | Purpose | Key Implementation |
|------|---------|-------------------|
| **lib.rs** | Entry points + batch dispatch | `kernel_init()` returns version, `batch_process()` dispatches JSON array of ops to handlers |
| **gates.rs** | Security scanning | 8 secret regex patterns + 12 destructive command patterns. Redaction: >12 chars → first4****last4 |
| **proof.rs** | Cryptographic chain | SHA-256 (sha2 crate), HMAC-SHA256 (hmac crate), sorted-key content hashing, proof chain verification |
| **scoring.rs** | Shard scoring | ShardRetriever: base_weight + 0.3 domain + 0.2 risk_class + 0.1/keyword (capped 0.3) |

### Cargo.toml

- **Crate**: `guidance-kernel` v0.1.0, edition 2021
- **Type**: cdylib + rlib (for WASM + testing)
- **Dependencies**: wasm-bindgen 0.2, sha2 0.10, hmac 0.12, regex 1.x, serde/serde_json 1.x, hex 0.4, js-sys 0.3
- **Build**: WASM SIMD128 enabled (`.cargo/config.toml`: `target-feature=+simd128`), O2, LTO, single codegen unit, stripped

### 9 Exported Functions (confirmed matching WASM glue)

| Export | Rust Module | JS Fallback? |
|--------|-------------|-------------|
| `kernel_init` | lib.rs | N/A |
| `sha256` | proof.rs | Yes (node:crypto) |
| `hmac_sha256` | proof.rs | Yes (node:crypto) |
| `content_hash` | proof.rs | Yes (JSON sort + sha256) |
| `sign_envelope` | proof.rs (alias for hmac_sha256) | Yes (hmac fallback) |
| `verify_chain` | proof.rs | **NO — throws** (defers to ProofChain class) |
| `scan_secrets` | gates.rs | **NO — throws** (defers to EnforcementGates) |
| `detect_destructive` | gates.rs | **NO — throws** (defers to EnforcementGates) |
| `batch_process` | lib.rs (dispatcher) | **NO — throws** |

### Security Findings

1. **Default signing key hardcoded**: `"claude-flow-guidance-default-key"` used when no key provided in BatchOp. Attacker knowing this can forge envelope signatures.
2. **Secret patterns cover 8 categories**: API keys, passwords, bearer tokens, private key headers, OpenAI (sk-*), GitHub (ghp_*), npm (npm_*), AWS (AKIA*). Custom formats would be missed.
3. **Destructive patterns cover 12 categories**: rm -rf, del /s, SQL DROP/TRUNCATE/DELETE/ALTER DROP, git push --force/reset --hard/clean, format, kubectl delete, helm delete.
4. **JS fallback gap**: If WASM fails to load, `verifyChain`, `scanSecrets`, and `detectDestructive` throw errors — security-critical functions silently degrade.

### sign_envelope Aliasing

In the WASM glue (line 199), `sign_envelope` calls `wasm.hmac_sha256` — it's just an alias at the WASM boundary. The Rust source has a proper `envelope_signing_body()` function that strips the signature field before signing, but this logic runs via `batch_process`, not the direct `sign_envelope` export.

## Remaining Gaps

~51 files still NOT_TOUCHED in the hook-pipeline domain, mostly:
- Triplicated copies across packages (claude-flow-cli, agentic-flow, claude-config)
- Plugin hooks.json configuration files (small, <50 LOC each)
- Additional plugin hook scripts and examples
- hook-handler.cjs and its variants (partially covered in R8)
