# R78 Execution Plan: CONNECTED Clear + ReasoningBank Rust Completion + Sublinear Core Layer

**Date**: 2026-02-17
**Session ID**: 78
**Focus**: Clear remaining CONNECTED tier (3 files), complete ReasoningBank Rust crate ecosystem (4 files), read sublinear-rust top-level Rust core (3 files)
**Strategic value**: CONNECTED tier cleared (recurring goal since R73). ReasoningBank Rust arc closed — all 4 remaining crates mapped (extends R67 genuinely-architected finding). Sublinear core Rust layer (fast_solver, temporal_nexus/mod, utils) completes understanding of how the top-level Rust solver connects to crate subsystems.

## Rationale

After R77, the CONNECTED tier has 3 files that emerged from R76's reads: `ruv_fann_impl.rs` and `ruv_fann_adapter.rs` in temporal-compare (connected to the 14 DEEP files in that crate), and `mmr.js` in agentic-flow's reasoningbank/utils (connected to 83 DEEP files in that directory tree). These are small (205 LOC total) but must be cleared first — they represent direct dependency connections to already-understood code.

The ReasoningBank Rust ecosystem has 4 untouched crates in `reasoningbank/crates/`: the MCP server (server.rs, 243 LOC), WASM storage adapter (wasm.rs, 224 LOC), learning optimizer (optimizer.rs, 211 LOC), and WASM bindings (lib.rs, 202 LOC). R67 found the ReasoningBank Rust architecture genuinely designed (trajectory tracking, verdict judgment, pattern distillation). R74 found the network layer BIMODAL (infra 85-95% vs gossip 18-22% fake). These 4 files complete the Rust side — is the MCP server genuine? Does the WASM adapter connect to the TS WASM-adapter (R73: 92-95%)? Is the learning optimizer real training or facade?

The sublinear-rust core Rust layer has 3 high-value files: `fast_solver.rs` (294 LOC) is the fast solver implementation at the crate root — how does it relate to the 9 DEEP files in src/core/? `temporal_nexus/mod.rs` (286 LOC) is the module root for temporal_nexus — R55 found genuine physics (80.75%), but we haven't read the Rust mod.rs that ties the subsystem together. `utils.rs` (218 LOC) is shared utilities — potential hash-based embedding code (would be 16th instance).

## Target: 10 files, ~1,883 LOC

---

### Cluster A: CONNECTED Tier Clear (3 files, ~205 LOC)

Last 3 CONNECTED-tier files from R76's emergence. Small but mandatory — clearing CONNECTED tier maintains the invariant that all direct-dependency files are understood before moving to proximity-only files.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 1 | 14162 | `crates/temporal-compare/src/ruv_fann_impl.rs` | 109 | CONNECTED to 14 DEEP temporal-compare files. ruv-FANN integration in sublinear solver |
| 2 | 10842 | `agentic-flow/src/reasoningbank/utils/mmr.js` | 65 | CONNECTED to 83 DEEP files. JS Maximal Marginal Relevance — compare with R73's mmr.ts (90%+) |
| 3 | 14161 | `crates/temporal-compare/src/ruv_fann_adapter.rs` | 31 | CONNECTED to 14 DEEP files. Adapter pattern for ruv-FANN <-> temporal-compare |

**Full paths**:
1. `~/repos/sublinear-time-solver/crates/temporal-compare/src/ruv_fann_impl.rs`
2. `~/repos/agentic-flow/src/reasoningbank/utils/mmr.js`
3. `~/repos/sublinear-time-solver/crates/temporal-compare/src/ruv_fann_adapter.rs`

**Key questions**:
- `ruv_fann_impl.rs` (109 LOC): What does ruv-FANN integration provide to temporal-compare? Real neural network ops or thin wrapper? Does it import from ruv-FANN crates or reimplement?
- `mmr.js` (65 LOC): JS port of mmr.ts or independent? R73 rated mmr.ts at 90%+ — does the .js version match quality? Is it used by the JS ReasoningBank layer?
- `ruv_fann_adapter.rs` (31 LOC): Trait adapter or data conversion? How does it bridge ruv-FANN types into temporal-compare's type system?

---

### Cluster B: ReasoningBank Rust Ecosystem Completion (4 files, ~880 LOC)

Four remaining PROXIMATE files across the `reasoningbank/crates/` directory. With 23 nearby DEEP files, this cluster completes the Rust side of the ReasoningBank arc (R67: genuinely architected, R74: network layer BIMODAL). After these, the ReasoningBank Rust crates are fully mapped.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 4 | 13452 | `reasoningbank/crates/reasoningbank-mcp/src/server.rs` | 243 | MCP server for ReasoningBank — 3rd MCP protocol (after R72's rmcp, R51's JSON-RPC)? |
| 5 | 13475 | `reasoningbank/crates/reasoningbank-storage/src/adapters/wasm.rs` | 224 | WASM storage adapter — connects to R73's wasm-adapter.ts (92-95%)? |
| 6 | 13448 | `reasoningbank/crates/reasoningbank-learning/src/optimizer.rs` | 211 | Learning optimizer — real gradient-based training or configuration facade? |
| 7 | 13483 | `reasoningbank/crates/reasoningbank-wasm/src/lib.rs` | 202 | WASM bindings crate root — genuine wasm-bindgen or theatrical? |

**Full paths**:
4. `~/repos/agentic-flow/reasoningbank/crates/reasoningbank-mcp/src/server.rs`
5. `~/repos/agentic-flow/reasoningbank/crates/reasoningbank-storage/src/adapters/wasm.rs`
6. `~/repos/agentic-flow/reasoningbank/crates/reasoningbank-learning/src/optimizer.rs`
7. `~/repos/agentic-flow/reasoningbank/crates/reasoningbank-wasm/src/lib.rs`

**Key questions**:
- `server.rs` (243 LOC): Which MCP SDK does it use? rmcp (R72), raw JSON-RPC (R51), or yet another? What tools does it expose — trajectory CRUD, pattern search, verdict judgment? Is it wired to the TS MCP layer or independent?
- `wasm.rs` (224 LOC): Does this provide Rust→WASM storage? How does it relate to the TS wasm-adapter.ts (R73: 92-95% genuine)? Is it a genuine adapter or facade?
- `optimizer.rs` (211 LOC): Does it implement real optimization (SGD, Adam)? Or is it a ReasoningBank-specific verdict/confidence optimizer? Connects to R67's trajectory tracking?
- `lib.rs` (202 LOC): Crate root for reasoningbank-wasm. Does it use wasm-bindgen/wasm-pack for genuine WASM compilation? What functions are exported? 11th WASM genuine or 8th theatrical?

---

### Cluster C: Sublinear Rust Core Layer (3 files, ~798 LOC)

Three top-level Rust files in the sublinear-time-solver crate. These connect to the 21 DEEP files in `src/` (including R76's solver-optimized, R60's metrics-reporter, R58's predictor). `fast_solver.rs` is particularly strategic — does a fast path exist that bypasses the elaborate WASM/MCP pipeline?

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 8 | 14366 | `src/fast_solver.rs` | 294 | Fast solver at crate root — shortcut past WASM pipeline? Extends R56 backward_push (92-95%) |
| 9 | 14492 | `src/temporal_nexus/mod.rs` | 286 | Module root for temporal_nexus subsystem — R55 found genuine physics (80.75%). Ties subsystem together |
| 10 | 14503 | `src/utils.rs` | 218 | Shared utilities — potential 16th hash-based embedding instance. What helpers exist at crate root? |

**Full paths**:
8. `~/repos/sublinear-time-solver/src/fast_solver.rs`
9. `~/repos/sublinear-time-solver/src/temporal_nexus/mod.rs`
10. `~/repos/sublinear-time-solver/src/utils.rs`

**Key questions**:
- `fast_solver.rs` (294 LOC): What algorithm does the "fast" solver use? Direct Rust computation bypassing WASM? Does it import from backward_push (R56: genuine O(1/epsilon) sublinear) or predictor (R58: genuine O(sqrt(n)))? Performance claims?
- `temporal_nexus/mod.rs` (286 LOC): How does it organize the temporal_nexus subsystem? Which sub-modules does it re-export? Does it expose a clean API or raw module dump (like temporal-compare/lib.rs at 35-40%)?
- `utils.rs` (218 LOC): Matrix helpers? Hash functions? Embedding utilities? Does it contain the Nth hash-based embedding fallback? Shared across how many other src/ files?

---

## Expected Outcomes

1. **CONNECTED tier CLEARED** — 0 remaining (goal maintained since R73)
2. **ReasoningBank Rust COMPLETE** — all crate files mapped, MCP/WASM/learning/storage assessed
3. **Sublinear core Rust layer extended** — fast_solver, temporal_nexus/mod, utils understood
4. **WASM arc extended** — reasoningbank-wasm/lib.rs + storage/wasm.rs classified (genuine vs theatrical)
5. **MCP protocol count updated** — server.rs may reveal 4th MCP SDK variant
6. **DEEP count**: ~1,222 (post-R77) → ~1,232

## DB Update Template

```javascript
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const sessionId = 78;
const today = new Date().toISOString().slice(0, 10);

// File IDs for this session:
// 14162: ruv_fann_impl.rs (109 LOC) — sublinear-rust, CONNECTED
// 10842: mmr.js (65 LOC) — agentic-flow-rust, CONNECTED
// 14161: ruv_fann_adapter.rs (31 LOC) — sublinear-rust, CONNECTED
// 13452: server.rs (243 LOC) — agentic-flow-rust, PROXIMATE
// 13475: wasm.rs (224 LOC) — agentic-flow-rust, PROXIMATE
// 13448: optimizer.rs (211 LOC) — agentic-flow-rust, PROXIMATE
// 13483: lib.rs (202 LOC) — agentic-flow-rust, PROXIMATE
// 14366: fast_solver.rs (294 LOC) — sublinear-rust, PROXIMATE
// 14492: temporal_nexus/mod.rs (286 LOC) — sublinear-rust, PROXIMATE
// 14503: utils.rs (218 LOC) — sublinear-rust, PROXIMATE
```

## Domain Tags

- Files 1, 3, 8-10 → `memory-and-learning` (already tagged)
- File 2 → `memory-and-learning` (already tagged)
- Files 4-7 → `memory-and-learning` (already tagged via reasoningbank)
- Files 4-7 may also need `agentdb-integration` if MCP/WASM bridges connect to AgentDB
- File 9 may need `neural-network-implementation` if temporal_nexus includes neural components

## Isolation Check

No selected files are in known-isolated subtrees. CONNECTED files are by definition connected. ReasoningBank crates are in `agentic-flow-rust` (CONNECTED package, 23 nearby DEEP files). Sublinear-rust `src/` directory has 21 nearby DEEP files. The RELIABLE isolated subtrees remain untouched:
- `neuro-divergent` (ruv-fann) — not selected
- `cuda-wasm` (ruv-fann) — not selected
- `patches` (ruvector) — not selected
- `scripts` (ruvector) — not selected
- `simulation` (agentdb) — not selected

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
