# R86 Execution Plan: ruv-swarm Rust Crate Sweep + Sublinear Arc Closures

**Date**: 2026-02-17
**Session ID**: 86
**Focus**: Complete ruv-swarm Rust crate sweep (MCP, CLI, DAA, ML), close psycho-symbolic + temporal crate arcs, validate npm error handling and memory storage
**Strategic value**: Closes 5+ crate-level arcs with small residual files. 101 nearby DEEP files in ruv-swarm Rust crates = highest proximity signal in remaining queue. Extends R84 MCP analysis (ruv-swarm-mcp entry+stdio), R82/R84 DAA analysis (telemetry), R85 npm validation (error handling + memory), and R80/R82 psycho-symbolic arc (config-loader, time_utils).

## Rationale

The remaining priority queue is 53 files (44 PROXIMATE, 7 NEARBY, 2 DOMAIN_ONLY) totaling ~3,758 LOC. With CONNECTED fully exhausted since R82, all work is now in the PROXIMATE tier — files co-located with 3+ DEEP files in the same directory.

This session targets two high-efficiency clusters. First, 4 small ruv-swarm Rust crate files (195 LOC total) that sit alongside 101 DEEP files — these close residual gaps in ruv-swarm-cli (monitor command), ruv-swarm-mcp (entry + stdio transport), and ruv-swarm-daa (telemetry). The MCP files directly extend R84's discovery that @modelcontextprotocol/sdk is the ruv-swarm npm MCP protocol, now checking the Rust-side MCP server entry point for the 5th+ MCP protocol instance.

Second, 4 sublinear-rust files (394 LOC) that close the psycho-symbolic reasoner arc (config-loader.ts extends R80/R82, time_utils.rs extends graph_reasoner analysis) and the temporal crate family (temporal-lead-solver entry, temporal-compare data layer — both untouched crates adjacent to temporal-tensor, our HIGHEST QUALITY crate at 93%).

Two ruv-swarm npm test files (200 LOC) bridge clusters — validate-error-handling.js and test-memory-storage.js extend R84/R85's npm runtime analysis and may reveal whether the "8th disconnected persistence layer" (R85 ruv-swarm-memory.js) is tested.

**Pre-session action**: Exclude `agentdb/simulation/` — RELIABLY ISOLATED (83 files, 14,177 LOC, 0 cross-deps). This removes 2 files from the queue (causal-reasoning.ts, reflexion-learning.ts) that would waste research time.

## Target: 10 files, ~789 LOC

---

### Cluster A: ruv-swarm Rust Crate Sweep (4 files, ~195 LOC)

Completes residual gaps in 4 ruv-swarm Rust sub-crates. All files sit in directories with 101 nearby DEEP files — the highest proximity signal in the remaining queue.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 1 | 8896 | `ruv-swarm/crates/ruv-swarm-cli/src/commands/monitor.rs` | 68 | CLI monitor command — extends R31/R71 "CLI = demo framework" thesis |
| 2 | 8969 | `ruv-swarm/crates/ruv-swarm-mcp/src/main.rs` | 46 | MCP server entry point — Rust-side MCP, extends R84 @modelcontextprotocol/sdk discovery |
| 3 | 8951 | `ruv-swarm/crates/ruv-swarm-daa/src/telemetry.rs` | 45 | DAA telemetry — extends R82/R84 DAA BIMODAL analysis |
| 4 | 8964 | `ruv-swarm/crates/ruv-swarm-mcp/src/bin/stdio.rs` | 36 | MCP stdio transport — binary entry for Rust MCP server |

**Full paths**:
1. `~/repos/ruv-FANN/ruv-swarm/crates/ruv-swarm-cli/src/commands/monitor.rs`
2. `~/repos/ruv-FANN/ruv-swarm/crates/ruv-swarm-mcp/src/main.rs`
3. `~/repos/ruv-FANN/ruv-swarm/crates/ruv-swarm-daa/src/telemetry.rs`
4. `~/repos/ruv-FANN/ruv-swarm/crates/ruv-swarm-mcp/src/bin/stdio.rs`

**Key questions**:
- `monitor.rs` (68 LOC): Does the CLI monitor produce real metrics or fabricated output? Does it connect to any actual swarm runtime (agents, MCP) or just print demo data?
- `main.rs` (46 LOC): Is this a 6th MCP protocol (Rust-native) or does it bridge to the npm @modelcontextprotocol/sdk? What transport does it use (stdio, HTTP, SSE)?
- `telemetry.rs` (45 LOC): Real OpenTelemetry integration or stub? Does it feed into DAA's genuine types.rs (R82: 78-82%) or the facade neural.rs (R82: 0-5%)?
- `stdio.rs` (36 LOC): Thin binary wrapper or does it add transport-specific logic (buffering, framing)?

---

### Cluster B: ruv-swarm npm Validation Layer (2 files, ~200 LOC)

Test and validation scripts for ruv-swarm npm package. Extends R84 MCP test suite analysis and R85 npm runtime findings.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 5 | 9827 | `ruv-swarm/npm/validate-error-handling.js` | 143 | Error handling validation — how rigorous is npm error recovery? |
| 6 | 9819 | `ruv-swarm/npm/test-memory-storage.js` | 57 | Memory storage test — tests the "8th disconnected persistence" (R85)? |

**Full paths**:
5. `~/repos/ruv-FANN/ruv-swarm/npm/validate-error-handling.js`
6. `~/repos/ruv-FANN/ruv-swarm/npm/test-memory-storage.js`

**Key questions**:
- `validate-error-handling.js` (143 LOC): Does it test actual error recovery paths or just assert that errors are thrown? What error categories does it cover (MCP errors, agent failures, memory errors)?
- `test-memory-storage.js` (57 LOC): Does it test the genuine sqlite-pool (R45: 92%) or the facade ruv-swarm-memory.js (R85: 0%)? Cross-session persistence or single-session only (like R84 test-mcp-persistence)?

---

### Cluster C: Sublinear Crate Arc Closures (4 files, ~394 LOC)

Closes psycho-symbolic reasoner arc (extends R80/R82) and temporal crate family (adjacent to temporal-tensor, HIGHEST QUALITY at 93%).

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 7 | 14063 | `crates/psycho-symbolic-reasoner/src/typescript/utils/config-loader.ts` | 167 | Psycho-symbolic config — extends R80 "pure TS" vs R82 "zero WASM config" |
| 8 | 14168 | `crates/temporal-lead-solver/index.js` | 104 | Temporal lead solver entry — untouched crate, near temporal-tensor (93%) |
| 9 | 14147 | `crates/temporal-compare/src/data.rs` | 62 | Temporal comparison data layer — untouched crate, Rust data structures |
| 10 | 14015 | `crates/psycho-symbolic-reasoner/graph_reasoner/src/time_utils.rs` | 61 | Graph reasoner time utilities — extends R80 psycho-symbolic analysis |

**Full paths**:
7. `~/repos/sublinear-time-solver/crates/psycho-symbolic-reasoner/src/typescript/utils/config-loader.ts`
8. `~/repos/sublinear-time-solver/crates/temporal-lead-solver/index.js`
9. `~/repos/sublinear-time-solver/crates/temporal-compare/src/data.rs`
10. `~/repos/sublinear-time-solver/crates/psycho-symbolic-reasoner/graph_reasoner/src/time_utils.rs`

**Key questions**:
- `config-loader.ts` (167 LOC): What config does the psycho-symbolic reasoner load? Does it reference WASM paths (contradicting R82 "zero WASM config") or is it purely TS configuration?
- `temporal-lead-solver/index.js` (104 LOC): What's the "lead solver"? Is it genuine algorithmic content like temporal-tensor or a thin wrapper? Any connection to the sublinear algorithms?
- `temporal-compare/data.rs` (62 LOC): What data structures does the comparison crate use? Does it share types with temporal-tensor or define its own?
- `time_utils.rs` (61 LOC): Real time-series utilities for the graph reasoner or trivial timestamp formatting?

---

## Expected Outcomes

1. **ruv-swarm-mcp Rust characterization** — determine if this is a 6th MCP protocol (Rust-native) or bridges to the npm SDK
2. **DAA telemetry verdict** — genuine vs facade, which DAA layer it connects to
3. **CLI monitor characterization** — real metrics vs demo output, extends "CLI = demo framework" thesis
4. **Psycho-symbolic arc near-closure** — config-loader + time_utils extend R80/R82 coverage
5. **Temporal crate family expansion** — temporal-lead-solver and temporal-compare characterized (adjacent to 93% temporal-tensor)
6. **npm persistence layer tested** — does test-memory-storage validate genuine or facade persistence?
7. **agentdb/simulation EXCLUDED** — 83 files, 14,177 LOC removed from queue

## DB Update Template

```javascript
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const sessionId = 86;
const today = new Date().toISOString().slice(0, 10);

// File IDs for this session:
// 8896: monitor.rs (68 LOC) — ruv-fann-rust, PROXIMATE
// 8969: main.rs (46 LOC) — ruv-fann-rust, PROXIMATE
// 8951: telemetry.rs (45 LOC) — ruv-fann-rust, PROXIMATE
// 8964: stdio.rs (36 LOC) — ruv-fann-rust, PROXIMATE
// 9827: validate-error-handling.js (143 LOC) — ruv-fann-rust, PROXIMATE
// 9819: test-memory-storage.js (57 LOC) — ruv-fann-rust, PROXIMATE
// 14063: config-loader.ts (167 LOC) — sublinear-rust, PROXIMATE
// 14168: index.js (104 LOC) — sublinear-rust, PROXIMATE
// 14147: data.rs (62 LOC) — sublinear-rust, PROXIMATE
// 14015: time_utils.rs (61 LOC) — sublinear-rust, PROXIMATE
```

## Domain Tags

- Files 1-6 (ruv-swarm) → `swarm-coordination` (already tagged)
- Files 7, 10 (psycho-symbolic) → `memory-and-learning` (already tagged)
- File 8 (temporal-lead-solver) → `memory-and-learning` (already tagged)
- File 9 (temporal-compare) → `memory-and-learning` (already tagged)

## Isolation Check

All selected files are in PROXIMATE tier with 8-101 nearby DEEP files. None are in known-isolated subtrees:
- `agentdb/simulation` (RELIABLY ISOLATED) — not selected, will be EXCLUDED
- `claude-config` (ISOLATED package) — not selected
- `@ruvector/core` (ISOLATED package) — not selected
- `neuro-divergent`, `cuda-wasm`, `ruvector-rust/patches`, `ruvector-rust/scripts` (RELIABLY ISOLATED) — not selected

**Pre-session exclusion** (run before spawning agents):
```bash
node -e "
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const today = new Date().toISOString().slice(0, 10);
db.prepare('INSERT OR IGNORE INTO exclude_paths (pattern, reason, added_date) VALUES (?, ?, ?)').run('%simulation/%', 'agentdb/simulation: RELIABLY ISOLATED (83 files, 14177 LOC, 0 cross-deps)', today);
db.close();
"
```

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
