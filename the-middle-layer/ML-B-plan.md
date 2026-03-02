# R135 Execution Plan: The V3 Memory Layer — claude-flow's Brain

**Date**: 2026-03-01
**Session ID**: 135
**Focus**: Read the claude-flow V3 memory subsystem — the layer that bootstraps AgentDB, manages 29 controllers, and bridges TypeScript to ruvector's Rust backend
**Strategic value**: This is the most comprehensive integration layer in the entire codebase. The controller-registry alone manages 29 subsystems. If there's a working end-to-end pipeline from claude-flow to native HNSW, it runs through these files.

## Rationale

The V3 `@claude-flow/memory` package was completely missed by our priority system because it lives in a different directory tree (`v3/@claude-flow/`) than the algorithm files we've been reading. Yet it contains the AgentDB adapter (1,038 LOC), a controller-registry managing 29 subsystems (1,026 LOC), a memory bridge (1,773 LOC), and a memory initializer (2,564 LOC). These are the files that answer: "when claude-flow starts, how does it connect to AgentDB and ruvector?"

All files are in the `claude-flow-cli` package (462 NOT_TOUCHED files, 170K LOC total). This session targets the memory layer specifically.

## Target: 7 files, ~8,967 LOC

---

### Cluster A: Memory Bootstrap (2 files, ~4,337 LOC)

What happens when the memory system initializes. This is step zero of any claude-flow operation.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 1 | 15488 | `v3/@claude-flow/cli/src/memory/memory-initializer.ts` | 2,564 | Memory system bootstrap. What gets loaded, in what order? |
| 2 | 15487 | `v3/@claude-flow/cli/src/memory/memory-bridge.ts` | 1,773 | Bridge layer — could be the TS-to-Rust connection we've been looking for. |

**Full paths**:
1. `~/.npm-global/lib/node_modules/@claude-flow/cli/src/memory/memory-initializer.ts`
2. `~/.npm-global/lib/node_modules/@claude-flow/cli/src/memory/memory-bridge.ts`

**Key questions**:
- `memory-initializer.ts` (2,564 LOC): Does it initialize the EmbeddingService? (R20 root cause: never initialized.) Does it call `detectBackends()` from the factory? Does it prefer native ruvector or fall back to HNSWLib? What config drives the backend selection?
- `memory-bridge.ts` (1,773 LOC): What does "bridge" mean here — bridge to Rust NAPI? Bridge to AgentDB? Bridge between V2 and V3? Does it use VectorDBWrapper (which works) or RuVectorBackend (which is broken)?

---

### Cluster B: Controller Registry + AgentDB Adapter (3 files, ~2,659 LOC)

The runtime wiring — which controllers exist, how they're registered, and how they talk to AgentDB.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 3 | 15525 | `v3/@claude-flow/memory/src/controller-registry.ts` | 1,026 | 29 controllers lifecycle-managed. Level-ordered init. |
| 4 | 15512 | `v3/@claude-flow/memory/src/agentdb-adapter.ts` | 1,038 | How V3 memory talks to AgentDB. The adapter pattern. |
| 5 | 15536 | `v3/@claude-flow/memory/src/index.ts` | 595 | Module barrel — what's exported, what's internal? |

**Full paths**:
3. `~/.npm-global/lib/node_modules/@claude-flow/cli/v3/@claude-flow/memory/src/controller-registry.ts`
4. `~/.npm-global/lib/node_modules/@claude-flow/cli/v3/@claude-flow/memory/src/agentdb-adapter.ts`
5. `~/.npm-global/lib/node_modules/@claude-flow/cli/v3/@claude-flow/memory/src/index.ts`

**Key questions**:
- `controller-registry.ts` (1,026 LOC): Does it actually instantiate all 29 controllers? Which ones are real vs stubbed? Does level-ordered init actually enforce dependency ordering? Does it error or fallback if a controller fails to initialize?
- `agentdb-adapter.ts` (1,038 LOC): Does it adapt the AgentDB we've analyzed (broken embedding service, R20) or a V3 version? Does it handle the backend factory selection? Does it expose vector search?
- `index.ts` (595 LOC): What's the public API surface? Does it re-export the adapter/registry or hide them?

---

### Cluster C: HNSW + Auto-Memory (2 files, ~1,971 LOC)

The HNSW index wrapper and auto-memory bridge — how semantic search and automatic memory work in V3.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 6 | 15532 | `v3/@claude-flow/memory/src/hnsw-index.ts` | 1,014 | HNSW index wrapper — does it use native ruvector or hnswlib-node? |
| 7 | 15521 | `v3/@claude-flow/memory/src/auto-memory-bridge.ts` | 957 | Auto-memory — how claude-flow decides what to remember. |

**Full paths**:
6. `~/.npm-global/lib/node_modules/@claude-flow/cli/v3/@claude-flow/memory/src/hnsw-index.ts`
7. `~/.npm-global/lib/node_modules/@claude-flow/cli/v3/@claude-flow/memory/src/auto-memory-bridge.ts`

**Key questions**:
- `hnsw-index.ts` (1,014 LOC): Does it import `@ruvector/core` (native HNSW) or `hnswlib-node` (JS fallback)? Does it use the auto-detect factory pattern? Is this the file that ultimately decides whether you get Rust SIMD or JS arrays?
- `auto-memory-bridge.ts` (957 LOC): What triggers automatic memory storage? Does it use embeddings (which embedder?) or keyword matching? Does it connect to the HNSW index above?

---

## Expected Outcomes

1. **Memory bootstrap path**: Exact sequence from `claude-flow` startup → memory-initializer → controller-registry → AgentDB
2. **Backend selection**: Does V3 use the factory.ts auto-detect? Does it default to native or HNSWLib?
3. **EmbeddingService status**: Is the R20 root cause (never initialized) fixed in V3?
4. **29 controllers**: Which are real, which are stubs? How many actually initialize?
5. **HNSW backend**: Native ruvector or hnswlib-node at the V3 memory layer?

## DB Update Template

```javascript
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const sessionId = 135;
const today = new Date().toISOString().slice(0, 10);

// File IDs for this session:
// 15488: memory-initializer.ts (2,564 LOC) — claude-flow-cli, NOT_TOUCHED
// 15487: memory-bridge.ts (1,773 LOC) — claude-flow-cli, NOT_TOUCHED
// 15525: controller-registry.ts (1,026 LOC) — claude-flow-cli, NOT_TOUCHED
// 15512: agentdb-adapter.ts (1,038 LOC) — claude-flow-cli, NOT_TOUCHED
// 15536: index.ts (595 LOC) — claude-flow-cli, NOT_TOUCHED
// 15532: hnsw-index.ts (1,014 LOC) — claude-flow-cli, NOT_TOUCHED
// 15521: auto-memory-bridge.ts (957 LOC) — claude-flow-cli, NOT_TOUCHED
```

## Domain Tags

- All files → `memory-and-learning` + `agentdb-integration`
- Files 15525, 15512 → also `production-infra` (runtime lifecycle)

## Isolation Check

All files are in the `claude-flow-cli` package, which is the globally-installed CLI at `~/.npm-global/lib/node_modules/@claude-flow/cli/`. This is the production install — by definition not isolated.

---

## Synthesis Doc Update Protocol (ADR-040)

**MANDATORY**: After all files are read and findings inserted into the DB, update the relevant `domains/*/analysis.md` files following the ADR-040 in-place protocol. Reference: `domains/memory-and-learning/analysis.md` for canonical structure.

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
