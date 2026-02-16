# R72 Execution Plan: ruv-swarm Rust Crate Sweep + ReasoningBank TS Core

**Date**: 2026-02-16
**Session ID**: 72
**Focus**: ruv-swarm outer Rust crates (CLI, WASM, MCP, DAA) + ReasoningBank TypeScript core operations (matts, consolidate, distill, backend-selector, embeddings)
**Strategic value**: Completes the ruv-swarm Rust picture beyond the core crates already deep-read (memory.rs 95%, persistence lib.rs 88-92%, p2p.rs 92-95%). Answers whether the ReasoningBank TS layer is genuine integration or disconnected facade relative to the R67-R68 Rust workspace (core 88-92%, learning 95-98%).

## Rationale

With the CONNECTED tier cleared in R70-R71 (MILESTONE), the priority queue is now 100% PROXIMATE. The two highest-connectivity clusters dominate:

**ruv-swarm Rust crates** (78 nearby DEEP files) are the densest unread area. R50 established the bimodal split (memory.rs 95% vs spawn.rs 8%). R66 found the npm TWO-TIER SPLIT (TS layer has 0% Rust integration). R69 found DAA memory.rs 0-5% FACADE. R70 confirmed persistence lib.rs 88-92%. But the outer crates — CLI entry point, WASM bindings, MCP service, DAA WASM — remain untouched. These reveal whether ruv-swarm has a usable CLI, genuine WASM compilation, and a working MCP service layer.

**ReasoningBank TS core** (58 nearby DEEP) is the second-densest. R67 proved the Rust workspace genuinely architected. R57 found queries.ts 85-90% (production-ready 7-table schema). R71 explored queries.js and AdvancedMemory.ts. But the core TypeScript operations — MATTS (trajectory tracking), consolidate (memory consolidation), distill (knowledge distillation), backend-selector, and embeddings — remain unread. These are the functional heart of the TS layer and will determine if it's a genuine implementation or duplicative facade alongside the Rust workspace.

## Target: 10 files, ~2,698 LOC

---

### Cluster A: ruv-swarm Rust Crate Internals (5 files, ~1,467 LOC)

Extends R50/R66/R69/R70's ruv-swarm analysis into the outer crate ring: CLI, WASM, MCP, DAA. The core Rust crates (memory, persistence, p2p, transport) are already DEEP. These outer crates answer: can you actually USE ruv-swarm?

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 1 | 8900 | `ruv-swarm/crates/ruv-swarm-cli/src/config.rs` | 335 | CLI config — 78 nearby DEEP, extends R50 CLI findings |
| 2 | 8901 | `ruv-swarm/crates/ruv-swarm-cli/src/main.rs` | 308 | CLI entry point — clap? structopt? What commands? |
| 3 | 9055 | `ruv-swarm/crates/ruv-swarm-wasm/src/utils.rs` | 300 | WASM utilities — genuine wasm-bindgen or theatrical? |
| 4 | 8955 | `ruv-swarm/crates/ruv-swarm-daa/src/wasm_simple.rs` | 268 | DAA WASM — R69 found daa/memory.rs 0-5% FACADE. Is WASM too? |
| 5 | 8971 | `ruv-swarm/crates/ruv-swarm-mcp/src/service.rs` | 256 | MCP service — does it expose real tools? Connects to mcp-server.js (R51: 88-92%)? |

**Full paths**:
1. `~/repos/ruv-FANN/ruv-swarm/crates/ruv-swarm-cli/src/config.rs`
2. `~/repos/ruv-FANN/ruv-swarm/crates/ruv-swarm-cli/src/main.rs`
3. `~/repos/ruv-FANN/ruv-swarm/crates/ruv-swarm-wasm/src/utils.rs`
4. `~/repos/ruv-FANN/ruv-swarm/crates/ruv-swarm-daa/src/wasm_simple.rs`
5. `~/repos/ruv-FANN/ruv-swarm/crates/ruv-swarm-mcp/src/service.rs`

**Key questions**:
- `config.rs` (335 LOC): What config structures does the CLI use? Does it reference the 3 persistence backends (SQLite/IndexedDB/in-memory) from R70? Does it support swarm topology configuration? Are there hardcoded values or genuine config parsing?
- `main.rs` (308 LOC): What CLI framework (clap/structopt/custom)? What commands are registered — spawn, status, memory, swarm? Does it connect to the Rust core crates (memory.rs, p2p.rs) or is it a thin wrapper around shell commands like the TS CLI (R31)?
- `utils.rs` (300 LOC): WASM scoreboard is 8 genuine vs 5 theatrical (62%). Does this have real wasm-bindgen annotations? What Rust types does it expose to WASM? Is there SIMD (simd_tests.rs is a neighbor at 273 LOC)?
- `wasm_simple.rs` (268 LOC): R69 found DAA memory.rs 0-5% FACADE. Does the WASM binding layer inherit the facade pattern or wrap genuine functionality from other crates? Does it import from ruv-swarm-core or re-implement?
- `service.rs` (256 LOC): R51 found 256 MCP tools in mcp-client.js. Does the Rust MCP service expose equivalent tools? Is it JSON-RPC 2.0 or a different protocol? Does it delegate to the genuine core crates?

---

### Cluster B: ReasoningBank TS Core Operations (5 files, ~1,231 LOC)

The functional heart of the agentic-flow TypeScript ReasoningBank. R67-R68 proved the Rust workspace is genuine (core 88-92%, learning 95-98%). R57 found queries.ts 85-90%. R71 explored queries.js and AdvancedMemory.ts. These 5 files are the remaining core operations.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 6 | 10815 | `agentic-flow/src/reasoningbank/core/matts.ts` | 309 | MATTS = Multi-Agent Trajectory Tracking System? Core algorithm |
| 7 | 10809 | `agentic-flow/src/reasoningbank/core/consolidate.ts` | 259 | Memory consolidation — EWC++ integration? |
| 8 | 10811 | `agentic-flow/src/reasoningbank/core/distill.ts` | 230 | Knowledge distillation — connects to ReasoningBank learning? |
| 9 | 10804 | `agentic-flow/src/reasoningbank/backend-selector.ts` | 222 | Backend selection — bridges Rust workspace? Or more disconnection? |
| 10 | 10841 | `agentic-flow/src/reasoningbank/utils/embeddings.ts` | 211 | Embeddings utility — 13th hash-based embedding? Or real? |

**Full paths**:
6. `~/repos/agentic-flow/agentic-flow/src/reasoningbank/core/matts.ts`
7. `~/repos/agentic-flow/agentic-flow/src/reasoningbank/core/consolidate.ts`
8. `~/repos/agentic-flow/agentic-flow/src/reasoningbank/core/distill.ts`
9. `~/repos/agentic-flow/agentic-flow/src/reasoningbank/backend-selector.ts`
10. `~/repos/agentic-flow/agentic-flow/src/reasoningbank/utils/embeddings.ts`

**Key questions**:
- `matts.ts` (309 LOC): What is MATTS? If Multi-Agent Trajectory Tracking System, does it implement genuine trajectory storage and retrieval? Does it use real vector similarity or hash-based? Does it connect to ReasoningBank Rust core's trajectory types? R67 found type-algorithm gap in R68 — does the TS layer suffer the same?
- `consolidate.ts` (259 LOC): Does it implement genuine memory consolidation (episodic → semantic transfer, importance scoring, decay)? Does it integrate with EWC++ from sona (R37) or implement its own? Is it used by AdvancedMemory.ts (R71)?
- `distill.ts` (230 LOC): Knowledge distillation from trajectories to patterns. Does it implement real distillation algorithms (compression, pruning, selection)? Or is it a thin wrapper around array operations? Does it connect to reasoningbank-learning (R67: 95-98%)?
- `backend-selector.ts` (222 LOC): CRITICAL for the disconnection question. Does it select between Rust ReasoningBank backend and TS-native backend? Does it use vector-backend-adapter.interface.ts (R67: 100% = MISSING LINK)? This file could reveal whether the 5th disconnected data layer has a unification path.
- `embeddings.ts` (211 LOC): Hash-based embedding count is at 12+. Does this utility use hash-based embeddings (13th instance) or real embeddings? If hash-based, does it at least acknowledge the limitation? Does it integrate with embedding-service.ts (R51: 75-80%, R20 ROOT CAUSE)?

---

## Expected Outcomes

1. **ruv-swarm CLI characterized**: Is it a genuine clap/structopt CLI or a thin wrapper? What commands does it actually support?
2. **ruv-swarm WASM scoreboard update**: utils.rs + wasm_simple.rs — genuine (push to 10+) or theatrical (push to 7+)?
3. **ruv-swarm MCP Rust layer**: Does service.rs connect to the same MCP ecosystem as mcp-server.js (R51)?
4. **DAA WASM vs DAA facade**: Does wasm_simple.rs inherit the facade pattern from memory.rs (0-5%)?
5. **ReasoningBank TS genuineness**: Are matts/consolidate/distill genuine algorithms or thin wrappers?
6. **Disconnection diagnosis**: Does backend-selector.ts bridge the Rust workspace or confirm the 5th disconnected data layer?
7. **Hash-based embeddings**: Does embeddings.ts push the count to 13+?
8. **DEEP files**: 1,150 → ~1,160

## DB Update Template

```javascript
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const sessionId = 72;
const today = new Date().toISOString().slice(0, 10);

// File IDs for this session:
// 8900: config.rs (335 LOC) — ruv-swarm-cli, swarm-coordination
// 8901: main.rs (308 LOC) — ruv-swarm-cli, swarm-coordination
// 9055: utils.rs (300 LOC) — ruv-swarm-wasm, swarm-coordination
// 8955: wasm_simple.rs (268 LOC) — ruv-swarm-daa, swarm-coordination
// 8971: service.rs (256 LOC) — ruv-swarm-mcp, swarm-coordination
// 10815: matts.ts (309 LOC) — reasoningbank core, memory-and-learning
// 10809: consolidate.ts (259 LOC) — reasoningbank core, memory-and-learning
// 10811: distill.ts (230 LOC) — reasoningbank core, memory-and-learning
// 10804: backend-selector.ts (222 LOC) — reasoningbank, memory-and-learning
// 10841: embeddings.ts (211 LOC) — reasoningbank utils, memory-and-learning
```

## Domain Tags

- Files 1-5 → `swarm-coordination` (already tagged)
- Files 6-10 → `memory-and-learning` (already tagged)

## Isolation Check

No selected files are in known-isolated subtrees. The ruv-swarm crates have 78 nearby DEEP files (highest connectivity in queue). The ReasoningBank TS files have 58 nearby DEEP files. Both clusters are well-connected to existing deep-read files. Neither is in ISOLATED packages (only claude-config and @ruvector/core are ISOLATED). All files safe to read.

---

## Synthesis Doc Update Protocol (ADR-040)

**MANDATORY**: After all files are read and findings inserted into the DB, update the relevant `domains/*/analysis.md` files following the ADR-040 in-place protocol. Reference: `domains/memory-and-learning/analysis.md` for canonical structure.

### Current Max IDs (check before inserting)

**memory-and-learning/analysis.md**:
- Check current max C{N} and H{N} in Sections 3a/3b before adding
- R71 added findings — verify latest IDs before inserting R72 findings

**swarm-coordination/analysis.md**:
- Check current max IDs before adding findings for Files 1-5

### Rules

| Section | Action | NEVER Do |
|---------|--------|----------|
| **1. Current State Summary** | REWRITE in-place | Append session narrative |
| **2. File Registry** | ADD/UPDATE rows in subsystem tables | Create per-session tables |
| **3. Findings Registry** | ADD with next sequential ID to 3a/3b | Create `### R72 Findings` blocks |
| **4. Positives Registry** | ADD new positives with session tag | Re-list old positives |
| **5. Subsystem Sections** | UPDATE existing, CREATE new by topic | Per-session narrative blocks |
| **8. Session Log** | APPEND 2-5 line entry | Put findings here |

### Anti-Patterns

- **NEVER** create `### R72 Findings (Session date)` blocks
- **NEVER** append findings after Section 8
- **NEVER** re-list findings from previous sessions
- **NEVER** use finding IDs that collide with existing ones
