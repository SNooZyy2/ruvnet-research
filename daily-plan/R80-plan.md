# R80 Execution Plan: CONNECTED Clear + Psycho-Symbolic TS + Neural-Network Periphery

**Date**: 2026-02-16
**Session ID**: 80
**Focus**: Clear final 2 CONNECTED files (activation.rs, adapters/mod.rs), then deep-read psycho-symbolic TS layer and neural-network peripheral files
**Strategic value**: CONNECTED tier cleared to 0 (again). Extends R55 psycho-symbolic arc (Rust verified, now TS layer). Extends R23 neural-network arc (core BEST, now peripheral variants/benchmarks). Both clusters test the "core is genuine, periphery may be deceptive" pattern seen across the ecosystem.

## Rationale

The priority queue has 115 files remaining: 2 CONNECTED, 104 PROXIMATE, 7 NEARBY, 2 DOMAIN_ONLY. The 2 CONNECTED files are mandatory — activation.rs completes the ruv-swarm-wasm crate internals (extends R79's BIMODAL findings), and adapters/mod.rs completes the ReasoningBank Rust storage layer (extends R78's comprehensive coverage).

For the remaining 8 files, two coherent clusters emerge from PROXIMATE tier. The psycho-symbolic-reasoner TS layer (4 files, ~1,130 LOC) was partially mapped in R55 which showed "Rust 3-4x better than TS" — now we verify the TS CLI, build pipeline, MCP integration, and performance utilities. The neural-network-implementation periphery (4 files, ~948 LOC) orbits R23's BEST IN ECOSYSTEM core — these variant implementations and benchmarks will reveal whether the high quality extends to the full crate or follows the "core genuine, periphery deceptive" pattern (as seen with criterion vs standalone benchmarks in R59).

Both clusters are in PROXIMATE tier (35 and 27 nearby DEEP files respectively), ensuring high contextual value. No selected files are in isolated subtrees (confirmed via subtree_connectivity check — all isolated subtrees are in neuro-divergent, cuda-wasm, patches, scripts, simulation, validation directories).

## Target: 10 files, ~2,206 LOC

---

### Cluster A: CONNECTED Clear (2 files, ~148 LOC)

Both remaining CONNECTED-tier files. Clearing these brings CONNECTED to 0 for the 4th time (R75, R76, R78, R80).

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 1 | 9039 | `ruv-swarm/crates/ruv-swarm-wasm/src/activation.rs` | 82 | CONNECTED via ruv-swarm-wasm crate (R79: crate is BIMODAL — simd_tests 88-92% vs agent 28-32%) |
| 2 | 13473 | `reasoningbank/crates/reasoningbank-storage/src/adapters/mod.rs` | 66 | CONNECTED via reasoningbank-storage (R78: ReasoningBank Rust COMPLETE, but adapters/mod.rs emerged late) |

**Full paths**:
1. `~/repos/ruv-FANN/ruv-swarm/crates/ruv-swarm-wasm/src/activation.rs`
2. `~/repos/agentic-flow/reasoningbank/crates/reasoningbank-storage/src/adapters/mod.rs`

**Key questions**:
- `activation.rs` (82 LOC): Which activation functions are implemented? Are they real math or stubs? Does this connect to ruv-fann's FANN activation system?
- `adapters/mod.rs` (66 LOC): What storage adapters are declared? Does this route to the 3+ disconnected AgentDB persistence layers or introduce a new one?

---

### Cluster B: Psycho-Symbolic TS Layer (4 files, ~1,130 LOC)

R55 found psycho-symbolic Rust "3-4x better than TS". These 4 files map the TS outer ring: build tooling, CLI entry point, performance utilities, and MCP integration. The MCP integration is especially interesting — does it use a 5th MCP protocol or reuse one of the 4 known protocols?

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 3 | 14051 | `crates/psycho-symbolic-reasoner/scripts/build.js` | 342 | Build pipeline — does it compile Rust→WASM or just bundle TS? |
| 4 | 14059 | `crates/psycho-symbolic-reasoner/src/typescript/cli/index.ts` | 286 | CLI entry — genuine CLI or demonstration framework (like R72 main.rs)? |
| 5 | 14065 | `crates/psycho-symbolic-reasoner/src/typescript/utils/performance.ts` | 266 | Performance utils — real metrics or hardcoded claims? |
| 6 | 14024 | `crates/psycho-symbolic-reasoner/mcp-integration/src/index.ts` | 236 | MCP integration — which protocol? What tools? |

**Full paths**:
3. `~/repos/sublinear-time-solver/crates/psycho-symbolic-reasoner/scripts/build.js`
4. `~/repos/sublinear-time-solver/crates/psycho-symbolic-reasoner/src/typescript/cli/index.ts`
5. `~/repos/sublinear-time-solver/crates/psycho-symbolic-reasoner/src/typescript/utils/performance.ts`
6. `~/repos/sublinear-time-solver/crates/psycho-symbolic-reasoner/mcp-integration/src/index.ts`

**Key questions**:
- `build.js` (342 LOC): Does it invoke `wasm-pack` or `cargo build --target wasm32`? Or is it a pure TS bundler with no Rust compilation?
- `cli/index.ts` (286 LOC): Does this actually run reasoning? Or is it a thin wrapper over the Rust binary?
- `performance.ts` (266 LOC): Real benchmark instrumentation or hardcoded performance claims (like R43's rustc_benchmarks)?
- `mcp-integration/src/index.ts` (236 LOC): 5th MCP protocol? How many tools? Does it expose genuine reasoning capabilities?

---

### Cluster C: Neural-Network Periphery (4 files, ~948 LOC)

R23 marked neural-network-implementation as BEST IN ECOSYSTEM (90-98%). These 4 peripheral files test whether the quality extends to variants and benchmarks. R59 established the "criterion genuine (88-95%) vs standalone deceptive (8-25%)" benchmark boundary — quick_demo.rs is a standalone benchmark and may follow that pattern.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 7 | 13960 | `crates/neural-network-implementation/standalone_benchmark/src/quick_demo.rs` | 276 | Standalone benchmark — deceptive per R59 pattern? |
| 8 | 13955 | `crates/neural-network-implementation/src/training/optimizer.rs` | 237 | Training optimizer — real SGD/Adam or stubs? |
| 9 | 13924 | `crates/neural-network-implementation/realistic-implementation/src/lib.rs` | 219 | "Realistic" variant root — what makes it "realistic" vs the main impl? |
| 10 | 13918 | `crates/neural-network-implementation/real-implementation/src/solver_integration.rs` | 216 | Solver integration — does this connect to genuine sublinear algorithms? |

**Full paths**:
7. `~/repos/sublinear-time-solver/crates/neural-network-implementation/standalone_benchmark/src/quick_demo.rs`
8. `~/repos/sublinear-time-solver/crates/neural-network-implementation/src/training/optimizer.rs`
9. `~/repos/sublinear-time-solver/crates/neural-network-implementation/realistic-implementation/src/lib.rs`
10. `~/repos/sublinear-time-solver/crates/neural-network-implementation/real-implementation/src/solver_integration.rs`

**Key questions**:
- `quick_demo.rs` (276 LOC): Does it use criterion or hand-rolled timing? Real computations or hardcoded results?
- `optimizer.rs` (237 LOC): Which optimizers? Learning rate scheduling? Does this connect to the sona/training ecosystem?
- `realistic-implementation/lib.rs` (219 LOC): Why a separate "realistic" impl? Is it a dumbed-down version or a different approach entirely?
- `solver_integration.rs` (216 LOC): Does this actually call backward_push.rs or predictor.rs (the 2 genuine sublinear algorithms)?

---

## Expected Outcomes

1. **CONNECTED tier CLEARED** (→0, 4th time)
2. **Psycho-symbolic TS quality scored** — confirms or refines R55's "3-4x worse" assessment
3. **Neural-network periphery pattern** — confirms or refutes "core genuine, periphery deceptive" hypothesis
4. **MCP protocol count** — 5th protocol or reuse of existing?
5. **Benchmark deception boundary** — quick_demo.rs tests R59's criterion-vs-standalone pattern
6. DEEP files: 1,232 → ~1,242

## DB Update Template

```javascript
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const sessionId = 80;
const today = new Date().toISOString().slice(0, 10);

// File IDs for this session:
// 9039: activation.rs (82 LOC) — ruv-fann-rust, CONNECTED
// 13473: adapters/mod.rs (66 LOC) — agentic-flow-rust, CONNECTED
// 14051: build.js (342 LOC) — sublinear-rust, PROXIMATE
// 14059: cli/index.ts (286 LOC) — sublinear-rust, PROXIMATE
// 14065: performance.ts (266 LOC) — sublinear-rust, PROXIMATE
// 14024: mcp-integration/src/index.ts (236 LOC) — sublinear-rust, PROXIMATE
// 13960: quick_demo.rs (276 LOC) — sublinear-rust, PROXIMATE
// 13955: optimizer.rs (237 LOC) — sublinear-rust, PROXIMATE
// 13924: realistic-implementation/lib.rs (219 LOC) — sublinear-rust, PROXIMATE
// 13918: solver_integration.rs (216 LOC) — sublinear-rust, PROXIMATE
```

## Domain Tags

- Files 9039 → `swarm-coordination` (already tagged)
- Files 13473 → `memory-and-learning` (already tagged)
- Files 14051, 14059, 14065, 14024 → `memory-and-learning` (already tagged; may also need `mcp-and-tools` for 14024)
- Files 13960, 13955, 13924, 13918 → `memory-and-learning` (already tagged; may also need `neural-network-implementation`)

## Isolation Check

No selected files are in known-isolated subtrees. Confirmed via `subtree_connectivity` check:
- All RELIABLE isolated subtrees are in: neuro-divergent, cuda-wasm, patches, scripts, simulation, validation
- ISOLATED packages: claude-config, @ruvector/core — none of our files belong to these
- All 10 files are in packages with CONNECTED or WEAKLY_CONNECTED connectivity

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
