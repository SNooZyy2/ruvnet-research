# R84 Execution Plan: ruv-swarm MCP+Persistence Tests, Rust Crate Roots, Cross-Arc Completion

**Date**: 2026-02-17
**Session ID**: 84
**Focus**: ruv-swarm npm MCP+persistence test suite, Rust crate entry points (DAA, ML, WASM), AgentDB embedding CLI, ReasoningBank hooks
**Strategic value**: (1) The MCP test files directly validate whether the 6+ disconnected persistence layers actually integrate or remain isolated — this is the highest-signal test for a CRITICAL cumulative finding. (2) Rust crate lib.rs/coordination.rs files are entry points that complete crate-level assessments left partial by R50/R79/R82. (3) install-embeddings.ts may reveal the INTENDED embedding initialization path, closing the R20 arc at the CLI level. (4) pre-task.ts extends R83's ReasoningBank TS cluster.

## Rationale

With CONNECTED tier exhausted (R82) and R83 executing the first PROXIMATE batch (ReasoningBank TS integration, neural training pipeline, ruv-swarm WASM cascade), R84 continues into the densest PROXIMATE territory: the ruv-swarm npm test suite (61 nearby DEEP files) and ruv-swarm Rust crate entry points (96 nearby DEEP files).

The ruv-swarm test cluster is the highest-value target remaining in the queue. Five test files (967 LOC) directly exercise MCP-DB interactions, persistence, feature flags, and MCP fixes. These tests are the best evidence available for whether the "6+ disconnected persistence layers" finding holds at runtime or whether the npm layer actually bridges them. Past sessions found ruv-swarm npm BIMODAL (R81) with real infrastructure but theatrical integration — the test files will reveal what the tests actually validate.

The Rust crate roots (4 files, 257 LOC) complete assessments of ruv-swarm-daa (R82: types.rs 78-82%, neural.rs 0-5%, patterns.rs 15-20%), ruv-swarm-wasm (R79: BIMODAL), and introduce ruv-swarm-ml (untouched). Each lib.rs reveals what a crate actually exports versus what its internal files implement.

Two cross-arc files close gaps: install-embeddings.ts tests R20's root cause (EmbeddingService never initialized) at the CLI level, and pre-task.ts extends R83's ReasoningBank hooks.

## Target: 11 files, ~1,386 LOC

---

### Cluster A: ruv-swarm MCP+Persistence Test Suite (5 files, ~967 LOC)

The densest remaining PROXIMATE cluster. These files test the MCP server's DB interactions, persistence layer, feature flags, and bug fixes. They directly address three cumulative findings: (1) 6+ disconnected persistence layers, (2) 5 MCP protocols, (3) ruv-swarm npm BIMODAL (R81). If these tests pass with real assertions and exercise actual DB operations, it contradicts "disconnected". If they're facade tests with mocked/stubbed persistence, it confirms.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 1 | 9815 | `ruv-swarm/npm/test-flags.js` | 264 | Feature flag test — reveals conditional behavior in npm runtime |
| 2 | 9814 | `ruv-swarm/npm/test-db-updates.js` | 193 | DB update tests — do they exercise real SQLite operations? |
| 3 | 9816 | `ruv-swarm/npm/test-mcp-db.js` | 173 | MCP-DB interaction tests — key to persistence layer verdict |
| 4 | 9818 | `ruv-swarm/npm/test-mcp-persistence.js` | 169 | MCP persistence tests — validates cross-layer persistence |
| 5 | 9817 | `ruv-swarm/npm/test-mcp-fixes.js` | 168 | MCP fix tests — reveals past bugs and their resolution |

**Full paths**:
1. `~/repos/ruv-FANN/ruv-swarm/npm/test-flags.js`
2. `~/repos/ruv-FANN/ruv-swarm/npm/test-db-updates.js`
3. `~/repos/ruv-FANN/ruv-swarm/npm/test-mcp-db.js`
4. `~/repos/ruv-FANN/ruv-swarm/npm/test-mcp-persistence.js`
5. `~/repos/ruv-FANN/ruv-swarm/npm/test-mcp-fixes.js`

**Key questions**:
- `test-flags.js` (264 LOC): What feature flags exist? Are they runtime-evaluated or compile-time? Do they gate real behavior or theatrical features?
- `test-db-updates.js` (193 LOC): Does it test actual SQLite write/read cycles, or mock the DB layer? What schema does it expect?
- `test-mcp-db.js` (173 LOC): Which MCP protocol does it test (JSON-RPC, rmcp, fastmcp, custom, @modelcontextprotocol/sdk)? Does it exercise real DB operations through the MCP layer?
- `test-mcp-persistence.js` (169 LOC): Does it test cross-session persistence? Does it prove that MCP state survives restarts, or is it a single-session mock?
- `test-mcp-fixes.js` (168 LOC): What bugs were fixed? Do the fixes correspond to known issues (R20 embedding, R48 three AgentDB layers, R81 bimodal)?

---

### Cluster B: ruv-swarm Rust Crate Roots (4 files, ~257 LOC)

Entry points for four under-analyzed crates. Each lib.rs or build.rs reveals the crate's actual public API versus internal implementation. coordination.rs completes the DAA picture (R82 found types.rs 78-82%, neural.rs 0-5% FACADE, patterns.rs 15-20%). ruv-swarm-ml is completely untouched — a new crate to assess. The WASM files extend both the WASM scoreboard (14 genuine vs 11 theatrical) and the R79 BIMODAL verdict.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 6 | 8939 | `ruv-swarm/crates/ruv-swarm-daa/src/coordination.rs` | 75 | DAA coordination — extends R82 (types.rs 78%, neural.rs 0%, patterns.rs 15%) |
| 7 | 9036 | `ruv-swarm/crates/ruv-swarm-wasm/build.rs` | 75 | WASM build script — reveals actual compilation targets and features |
| 8 | 8986 | `ruv-swarm/crates/ruv-swarm-ml/src/lib.rs` | 54 | ML crate root — UNTOUCHED crate, first look at API surface |
| 9 | 8956 | `ruv-swarm/crates/ruv-swarm-daa/src-wasm/lib.rs` | 53 | DAA WASM entry — extends R69 GHOST WASM (27 models) and R82 daa |

**Full paths**:
6. `~/repos/ruv-FANN/ruv-swarm/crates/ruv-swarm-daa/src/coordination.rs`
7. `~/repos/ruv-FANN/ruv-swarm/crates/ruv-swarm-wasm/build.rs`
8. `~/repos/ruv-FANN/ruv-swarm/crates/ruv-swarm-ml/src/lib.rs`
9. `~/repos/ruv-FANN/ruv-swarm/crates/ruv-swarm-daa/src-wasm/lib.rs`

**Key questions**:
- `coordination.rs` (75 LOC): Does DAA coordination implement real distributed coordination (Raft, PBFT, gossip), or is it a facade like neural.rs (0-5%)? Does it use the types defined in types.rs?
- `build.rs` (75 LOC): What wasm-bindgen/wasm-pack targets does it configure? Does it compile real Rust → WASM, or is it a no-op build script?
- `ruv-swarm-ml/lib.rs` (54 LOC): What ML capabilities does this crate expose? Is it a real ML library or a re-export wrapper? Does it depend on external ML crates (tch, candle, burn)?
- `ruv-swarm-daa/src-wasm/lib.rs` (53 LOC): Does this WASM entry point export the DAA types (types.rs 78%) or the facade modules (neural.rs 0%)? Is it a 15th genuine or 12th theatrical WASM?

---

### Cluster C: Cross-Arc Completion (2 files, ~162 LOC)

Two files that close gaps in major arcs. install-embeddings.ts is directly relevant to R20 (AgentDB search broken because EmbeddingService never initialized) — if this CLI command is the INTENDED initialization mechanism, it proves embedding setup was designed as a manual step that was never wired into the runtime. pre-task.ts extends R83's ReasoningBank TS integration cluster.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 10 | 355 | `src/cli/commands/install-embeddings.ts` | 82 | R20 ARC: intended embedding initialization path? |
| 11 | 10824 | `agentic-flow/src/reasoningbank/hooks/pre-task.ts` | 80 | Extends R83 ReasoningBank TS cluster — hooks integration |

**Full paths**:
10. `~/node_modules/agentdb/src/cli/commands/install-embeddings.ts`
11. `~/repos/agentic-flow/agentic-flow/src/reasoningbank/hooks/pre-task.ts`

**Key questions**:
- `install-embeddings.ts` (82 LOC): Does this command actually download/install an embedding model? Does it configure EmbeddingService? If so, R20's root cause becomes "embeddings require manual CLI install that was never run" — a design flaw rather than a code bug.
- `pre-task.ts` (80 LOC): What does the pre-task hook do for ReasoningBank? Does it load trajectories, check memory, or initialize state? Does it connect to the statistical ranking system found in R67?

---

## Expected Outcomes

1. Persistence layer verdict: MCP tests either prove integration (reduce "6+ disconnected" count) or confirm isolation with test evidence
2. ruv-swarm-daa crate assessment complete: coordination.rs fills the gap between types.rs (78%) and neural.rs (0%)
3. WASM scoreboard updated: DAA WASM lib.rs and build.rs → 15th/16th genuine or 12th/13th theatrical
4. ruv-swarm-ml crate initial assessment: first look at an untouched crate
5. R20 arc CLI-level closure: install-embeddings.ts reveals intended vs actual embedding workflow
6. ReasoningBank hooks integration quality assessed
7. DEEP count: 1,262 → ~1,273

## DB Update Template

```javascript
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const sessionId = 84;
const today = new Date().toISOString().slice(0, 10);

// File IDs for this session:
// 9815: test-flags.js (264 LOC) — ruv-fann-rust, PROXIMATE
// 9814: test-db-updates.js (193 LOC) — ruv-fann-rust, PROXIMATE
// 9816: test-mcp-db.js (173 LOC) — ruv-fann-rust, PROXIMATE
// 9818: test-mcp-persistence.js (169 LOC) — ruv-fann-rust, PROXIMATE
// 9817: test-mcp-fixes.js (168 LOC) — ruv-fann-rust, PROXIMATE
// 8939: coordination.rs (75 LOC) — ruv-fann-rust, PROXIMATE
// 9036: build.rs (75 LOC) — ruv-fann-rust, PROXIMATE
// 8986: lib.rs (54 LOC) — ruv-fann-rust, PROXIMATE
// 8956: lib.rs (53 LOC) — ruv-fann-rust, PROXIMATE
// 355: install-embeddings.ts (82 LOC) — agentdb, PROXIMATE
// 10824: pre-task.ts (80 LOC) — agentic-flow-rust, PROXIMATE
```

## Domain Tags

- Files 9815, 9814, 9816, 9818, 9817, 8939, 9036, 8986, 8956 → `swarm-coordination` (already tagged)
- File 355 → `memory-and-learning`, `agentdb-integration` (already tagged)
- File 10824 → `memory-and-learning`, `agentic-flow` (already tagged)

## Isolation Check

No selected files are in RELIABLY isolated subtrees. Checked:
- `neuro-divergent` (170 untouched, RELIABLE isolated) — no files selected
- `cuda-wasm` (203 untouched, RELIABLE isolated) — no files selected
- `ruvector-rust/patches` (22 untouched, RELIABLE isolated) — no files selected
- `ruvector-rust/scripts` (49 untouched, RELIABLE isolated) — no files selected
- `agentdb/simulation` (74 untouched, RELIABLE isolated) — no files selected (causal-reasoning.ts and reflexion-learning.ts appeared in priority queue but excluded due to isolation)
- `sublinear-rust/validation` (22 untouched, RELIABLE isolated) — no files selected

Isolated packages: `claude-config` (ISOLATED, 744 files) and `@ruvector/core` (ISOLATED, 2 files) — no files selected from either. File 1207 (helpers/memory.js) excluded because claude-config is ISOLATED.

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
