# R140 Execution Plan: The Execution Engine — Services, Intelligence, and the Largest Command

**Date**: 2026-03-02
**Session ID**: 140
**Focus**: Read the V3 service layer (how agents actually execute), the memory intelligence/SONA layer (whether SONA is wired at V3 level), and the hooks command (4.5K LOC — either the biggest real feature or the biggest facade in V3)
**Strategic value**: This is the last ML session. After ML-A through ML-E, we know the front door, the wiring, the tools, the backends, and the ground truth. This session fills the one remaining gap: what happens inside the box when an agent executes? Is the V3 "intelligence" layer real, and is hooks.ts a 4.5K LOC implementation or a 4.5K LOC placeholder?

## Rationale

ML-A through ML-D traced the integration path from CLI → MCP → memory → backend. ML-E verifies it builds. But we've never read the **execution infrastructure** — the code that actually runs agents, manages workers, handles claims, and coordinates intelligence. These files sit between the command dispatch (ML-D) and the memory backend (ML-B).

This session targets 3 clusters chosen for **maximum surprise potential**:
1. The **service layer** — worker execution, daemon process, claims, container pool
2. The **memory intelligence layer** — SONA optimizer, intelligence engine (V3 level, different from the R137 dead-code `sona-agentdb-integration.ts`)
3. The **hooks command** — at 4,530 LOC, it's the single largest command implementation. If it's real, hooks are a major feature. If it's a facade, it's the most elaborate one in the codebase.

After this session, the middle layer is mapped. Further reading would be pattern-repetition across individual commands.

## Target: 7 files, ~10,542 LOC

---

### Cluster A: Service Layer — How Agents Execute (4 files, 4,185 LOC)

The agent execution infrastructure. When claude-flow spawns an agent, what code actually runs it?

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 1 | 15493 | `v3/@claude-flow/cli/src/services/headless-worker-executor.ts` | 1,342 | **Agent execution engine.** When an agent is spawned, this is (likely) what runs it headlessly. Does it shell out to `claude`? Does it use subprocess management? Does it connect to the memory layer? |
| 2 | 15497 | `v3/@claude-flow/cli/src/services/worker-daemon.ts` | 942 | **Long-running worker process.** How does claude-flow keep workers alive? Is this a PM2-style daemon? Does it manage multiple workers? Health checks? |
| 3 | 15491 | `v3/@claude-flow/cli/src/services/claim-service.ts` | 1,118 | **Work distribution.** How do agents claim tasks? Mutex-style locking? Distributed claims? Does this connect to the MCP claims tools (873 LOC)? |
| 4 | 15492 | `v3/@claude-flow/cli/src/services/container-worker-pool.ts` | 783 | **Container-based execution.** Does claude-flow actually manage Docker containers for agent execution? Or is "container" an abstraction for process isolation? |

**Full paths**:
1. `~/.npm-global/lib/node_modules/@claude-flow/cli/src/services/headless-worker-executor.ts`
2. `~/.npm-global/lib/node_modules/@claude-flow/cli/src/services/worker-daemon.ts`
3. `~/.npm-global/lib/node_modules/@claude-flow/cli/src/services/claim-service.ts`
4. `~/.npm-global/lib/node_modules/@claude-flow/cli/src/services/container-worker-pool.ts`

**Key questions**:
- `headless-worker-executor.ts` (1,342 LOC): Does it invoke `claude` CLI as subprocess? Does it manage stdin/stdout? Does it inject agent prompts? Does it use the MCP protocol to communicate with workers? This file likely reveals the **actual mechanism** behind `claude-flow agent spawn`.
- `worker-daemon.ts` (942 LOC): Is this a proper daemon (backgrounded, PID-file, signal handling)? Or a foreground loop? Does it restart crashed workers? How does it report status?
- `claim-service.ts` (1,118 LOC): File-based locks? SQLite-based? Redis? Does it handle the `claims_claim`, `claims_release`, `claims_handoff` MCP tools we saw registered in ML-D?
- `container-worker-pool.ts` (783 LOC): Real Docker SDK integration? Or abstracted "container" meaning an isolated process? Pool sizing? Resource limits?

---

### Cluster B: Memory Intelligence + SONA at V3 Level (2 files, 1,827 LOC)

We know from R137 that `sona-agentdb-integration.ts` (agentic-flow) is dead code. But V3 has its own intelligence/SONA layer. Is it different?

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 5 | 15486 | `v3/@claude-flow/cli/src/memory/intelligence.ts` | 985 | **V3 intelligence engine.** Does this connect SONA to memory at the V3 level? Is it a separate implementation from the dead-code `sona-agentdb-integration.ts`? Does it actually call into ruvector? |
| 6 | 15489 | `v3/@claude-flow/cli/src/memory/sona-optimizer.ts` | 842 | **SONA optimizer at V3 level.** ML-C found SONA dead at V1 level. Does V3 have a working SONA integration? Does it use the genuine `hnsw_router.rs` (90-93% real, ML-C) or the broken TS adapters? |

**Full paths**:
5. `~/.npm-global/lib/node_modules/@claude-flow/cli/src/memory/intelligence.ts`
6. `~/.npm-global/lib/node_modules/@claude-flow/cli/src/memory/sona-optimizer.ts`

**Key questions**:
- `intelligence.ts` (985 LOC): What is the "intelligence" abstraction? Does it compose HNSW + SONA + attention? Does it import from `@ruvector/sona` or implement its own? Is it used by any other V3 file, or is it another orphan?
- `sona-optimizer.ts` (842 LOC): Does it reference `SonaEngine`? Does it optimize memory access patterns? Does it connect to the `sona-tools.ts` MCP tools (1,002 LOC, also unread)? Is it an optimization layer over the HNSW index from ML-B's `hnsw-index.ts`?

**Why this matters**: If V3 has a WORKING SONA integration (even without native Rust), it changes the assessment. R137 found SONA dead at the agentic-flow level, but V3 might have reimplemented it over hnswlib-node.

---

### Cluster C: The Hooks Command — Largest Command Implementation (1 file, 4,530 LOC)

The single largest CLI command implementation in all of claude-flow V3.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 7 | 15426 | `v3/@claude-flow/cli/src/commands/hooks.ts` | 4,530 | **17 hook subcommands.** Pre/post task, pre/post edit, model routing, worker dispatch, intelligence, session management, swarm gate, and more. 4.5K LOC of command handlers. |

**Full path**:
7. `~/.npm-global/lib/node_modules/@claude-flow/cli/src/commands/hooks.ts`

**Key questions**:
- **Real or facade?** At 4,530 LOC, this is either a substantial feature implementation or the most elaborate placeholder in V3. The hooks system is central to claude-flow's value proposition (self-learning, routing, intelligence). Does the implementation actually execute hooks, or does it just format output?
- **17 subcommands**: `pre-edit`, `post-edit`, `pre-command`, `post-command`, `route`, `metrics`, `pre-task`, `post-task`, `session-start`, `session-end`, `session-restore`, `notify`, `intelligence`, `worker-list`, `worker-dispatch`, `worker-status`, `model-route`. How many have real handlers?
- **Intelligence integration**: Does the `intelligence` subcommand connect to `intelligence.ts` (Cluster B)? Does `model-route` implement the ADR-008 3-tier routing?
- **Worker dispatch**: Does `worker-dispatch` connect to the `headless-worker-executor.ts` (Cluster A)?
- **Self-learning**: Is there actual pattern recording, or just console output?

---

## Expected Outcomes

1. **Agent execution model**: How does claude-flow actually run agents — subprocess? Docker? Direct API calls?
2. **Claim/work distribution**: Is the claims system a real distributed coordination primitive or a local file lock?
3. **V3 SONA status**: Working integration at V3 level, or dead code like V1?
4. **Hooks reality**: The 4.5K LOC hooks command — real feature or elaborate facade?
5. **Intelligence layer**: Does V3 compose HNSW + SONA + attention into a working intelligence layer?
6. **Complete middle layer**: After this session, the full integration picture is mapped from CLI → dispatch → services → memory → intelligence → backend → algorithm

## DB Update Template

```javascript
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const sessionId = 140;
const today = new Date().toISOString().slice(0, 10);

// File IDs for this session:
// 15493: headless-worker-executor.ts (1,342 LOC) — claude-flow-cli, NOT_TOUCHED
// 15497: worker-daemon.ts (942 LOC) — claude-flow-cli, NOT_TOUCHED
// 15491: claim-service.ts (1,118 LOC) — claude-flow-cli, NOT_TOUCHED
// 15492: container-worker-pool.ts (783 LOC) — claude-flow-cli, NOT_TOUCHED
// 15486: intelligence.ts (985 LOC) — claude-flow-cli, NOT_TOUCHED
// 15489: sona-optimizer.ts (842 LOC) — claude-flow-cli, NOT_TOUCHED
// 15426: hooks.ts (4,530 LOC) — claude-flow-cli, NOT_TOUCHED
```

## Domain Tags

- Cluster A → `production-infra` + `swarm-coordination`
- Cluster B → `memory-and-learning` + `ruvector`
- Cluster C → `production-infra` + `model-routing`

## Isolation Check

All files are in the `claude-flow-cli` package (the globally-installed production CLI). By definition connected to the runtime.

---

## Why This Is The Last ML Session

After ML-F, the middle layer map is:

| Layer | ML Session | Status |
|-------|------------|--------|
| CLI entrypoints (front door) | ML-A (R135) | DONE |
| Memory bootstrap + AgentDB adapter | ML-B (R136) | DONE |
| Rust integration hubs + ghost DEEP | ML-C (R137) | DONE |
| MCP tool chain (tool → handler) | ML-D (R138) | DONE |
| CI / tests / deployment (ground truth) | ML-E (R139) | PLANNED |
| Execution engine + intelligence + hooks | ML-F (R140) | THIS SESSION |

**What remains unread**: ~30 individual CLI command implementations (`commands/memory.ts`, `commands/neural.ts`, `commands/embeddings.ts`, etc.) and ~10 MCP tool implementation files. These are **pattern-repetition** — each command calls into the controller-registry → adapter → backend chain we already understand from ML-B. Reading them would confirm the pattern but not discover new architecture.

The analogy: ML-A through ML-F mapped all the roads, intersections, and bridges. The remaining files are individual houses along roads we've already driven.

---

## Synthesis Doc Update Protocol (ADR-040)

**MANDATORY**: After all files are read and findings inserted, update relevant `domains/*/analysis.md` following ADR-040.

### Rules for Each Section

| Section | Action | NEVER Do |
|---------|--------|----------|
| **1. Current State Summary** | REWRITE in-place to reflect current state | Append session narrative |
| **2. File Registry** | ADD new rows to existing subsystem tables, UPDATE rows if re-read | Duplicate rows, create per-session file tables |
| **3. Findings Registry** | ADD new findings with next sequential ID (C{max+1}, H{max+1}) to 3a/3b | Create `### RXX Findings` blocks, re-list old findings, restart ID numbering |
| **4. Positives Registry** | ADD new positives with session tag | Re-list existing positives |
| **5. Subsystem Sections** | UPDATE existing sections, CREATE new ones by topic | Create per-session narrative blocks |
| **8. Session Log** | APPEND 2-5 line entry for this session | Put findings here, write full narratives |

### Finding ID Assignment

Before adding findings, check the current max ID in the target domain's analysis.md:
- Section 3a: find last `| C{N} |` row → new CRITICALs start at C{N+1}
- Section 3b: find last `| H{N} |` row → new HIGHs start at H{N+1}

**ID format**: `| {ID} | **{short title}** — {description} | {file(s)} | R{session} | Open |`

### Anti-Patterns (NEVER do these)

- **NEVER** create `### R{N} Findings (Session date)` blocks outside Section 3
- **NEVER** append findings after Section 8
- **NEVER** create `### R{N} Full Session Verdict` blocks
- **NEVER** use finding IDs that collide with existing ones (always check max first)
- **NEVER** re-list findings from previous sessions

### Synthesis Update Checklist

- [ ] Section 1 rewritten with updated state
- [ ] New file rows added to Section 2 (correct subsystem table)
- [ ] New findings added to Section 3a/3b with sequential IDs
- [ ] New positives added to Section 4 (if any)
- [ ] Relevant subsystem sections in Section 5 updated
- [ ] Session log entry appended to Section 8 (2-5 lines max)
- [ ] No per-session finding blocks created anywhere
