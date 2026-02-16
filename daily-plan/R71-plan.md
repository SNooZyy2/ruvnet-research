# R71 Execution Plan: Clear CONNECTED Tier + ReasoningBank Expansion

**Date**: 2026-02-16
**Session ID**: 71
**Focus**: Final 4 CONNECTED files (clears tier 1 entirely — MILESTONE) + 6 PROXIMATE files expanding the ReasoningBank and sublinear-time-solver arcs.
**Strategic value**: After R71, zero CONNECTED files remain in the priority queue. Every dependency edge connects two DEEP files. The 203 remaining PROXIMATE files become the sole frontier.

## Rationale

The smart priority queue has 207 files remaining. Only **4 are CONNECTED** — the last files with direct dependency edges to DEEP files that haven't been read. Clearing them is a research milestone: the entire dependency graph becomes fully analyzed.

The 6 PROXIMATE files extend two active arcs:
- **ReasoningBank**: R67 found the Rust workspace genuinely architected (core 88-92%, storage 94%, learning 95-98%). The agentic-flow JS/TS layer around it remains partially explored — queries.js, intent.rs, and adaptive.rs fill critical gaps.
- **Sublinear runtime**: R70 found TWO parallel temporal solvers (temporal-lead-solver O(√n) vs UltraFastTemporalSolver). ultra_fast.rs likely IS the UltraFast implementation. scheduler_tool.rs reveals MCP scheduling for the solver ecosystem.

## Target: 10 files, ~3,251 LOC

---

### Cluster A: Final CONNECTED Tier — MILESTONE (4 files, ~1,243 LOC)

The last 4 files in the priority queue with direct dependency edges to existing DEEP files. Reading these closes every blind link in the dependency graph.

| # | File ID | File | LOC | Connected From (DEEP) |
|---|---------|------|-----|-----------------------|
| 1 | 13917 | `crates/neural-network-implementation/real-implementation/src/optimized.rs` | 351 | ← temporal-solver.rs (DEEP, IMPORTS) |
| 2 | 9009 | `ruv-swarm/crates/ruv-swarm-persistence/src/models.rs` | 333 | ← lib.rs (DEEP, imports) |
| 3 | 14176 | `crates/temporal-lead-solver/src/validation.rs` | 314 | ← lib.rs (DEEP, pub_use) |
| 4 | 14177 | `crates/temporal-lead-solver/src/wasm.rs` | 245 | ← lib.rs (DEEP, module_declaration) |

**Full paths**:
1. `~/repos/sublinear-time-solver/crates/neural-network-implementation/real-implementation/src/optimized.rs`
2. `~/repos/ruv-FANN/ruv-swarm/crates/ruv-swarm-persistence/src/models.rs`
3. `~/repos/sublinear-time-solver/crates/temporal-lead-solver/src/validation.rs`
4. `~/repos/sublinear-time-solver/crates/temporal-lead-solver/src/wasm.rs`

**Key questions**:
- `optimized.rs` (351 LOC): R23 found neural-network-implementation BEST IN ECOSYSTEM (90-98%). Does optimized.rs contain SIMD/AVX optimizations for the neural net? temporal-solver.rs (DEEP) imports from it — what functions does it use? Is this a production-quality optimization layer or placeholder?
- `models.rs` (333 LOC): R70 found ruv-swarm-persistence lib.rs (88-92%) with 28 async CRUD methods, 3 backends, SQL injection prevention. models.rs defines the data structures — does it use serde? Does it define the 5 tables from R69 migrations (agents/tasks/events/messages/metrics)? How do the structs map to the CRUD API?
- `validation.rs` (314 LOC): temporal-lead-solver lib.rs (R70: 92-95%) uses pub_use to re-export validation. R70 found "proof-carrying validation" as a distinguishing feature. Does validation.rs implement actual proof verification (zero-knowledge? hash chains?) or just input validation? This could elevate temporal-lead-solver's quality score further.
- `wasm.rs` (245 LOC): temporal-lead-solver lib.rs declares this as a module. R70 found temporal-lead-solver is the "best crate root in project." Does wasm.rs expose the O(√n) predictor to WASM? Is this genuine wasm-bindgen or another theatrical WASM? The WASM scoreboard is 8 genuine vs 5 theatrical — which side does this land on?

---

### Cluster B: ReasoningBank Expansion (3 files, ~975 LOC)

Extends the ReasoningBank arc from R67-R68. The Rust workspace was genuinely architected (core 88-92%, storage 94%, learning 95-98%, MCP 93-95%). These files explore the neural_bus network layer and the agentic-flow JS integration layer (which R57 found as the 4th disconnected data layer).

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 5 | 13463 | `reasoningbank/crates/reasoningbank-network/src/neural_bus/intent.rs` | 322 | ReasoningBank network: neural bus intent classification |
| 6 | 13444 | `reasoningbank/crates/reasoningbank-learning/src/adaptive.rs` | 318 | ReasoningBank learning: adaptive learning algorithms |
| 7 | 10818 | `agentic-flow/src/reasoningbank/db/queries.js` | 335 | agentic-flow ReasoningBank: 4th disconnected data layer (R57) |

**Full paths**:
5. `~/repos/agentic-flow/reasoningbank/crates/reasoningbank-network/src/neural_bus/intent.rs`
6. `~/repos/agentic-flow/reasoningbank/crates/reasoningbank-learning/src/adaptive.rs`
7. `~/repos/agentic-flow/agentic-flow/src/reasoningbank/db/queries.js`

**Key questions**:
- `intent.rs` (322 LOC): The neural_bus in reasoningbank-network. R67 found the workspace genuinely architected. Does intent.rs classify intent for routing messages between agents? Does it use real embeddings or hash-based (12th instance)? Is this connected to SemanticRouter (R51: 88-92% "first genuinely intelligent routing") or independent?
- `adaptive.rs` (318 LOC): R67-R68 found reasoningbank-learning at 95-98%. Does adaptive.rs implement online learning (learning rate scheduling, curriculum learning)? Does it integrate with EWC++ from sona (R37)? Is this the feedback loop that makes ReasoningBank self-improving?
- `queries.js` (335 LOC): R57 found queries.ts at 85-90% (PRODUCTION-READY, 7-table schema). This is a DIFFERENT queries file — in agentic-flow's JS ReasoningBank layer. Does it duplicate the 7-table schema? Does it connect to the Rust ReasoningBank workspace or is it the 4th disconnected data layer? Does it use real SQL or fabricated queries?

---

### Cluster C: Sublinear Runtime + Memory Integration (3 files, ~1,033 LOC)

Extends the sublinear-time-solver arc from R70 and bridges into the agentic-flow ReasoningBank memory system.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 8 | 14501 | `src/ultra_fast.rs` | 361 | Likely UltraFastTemporalSolver from R70's TWO parallel solvers |
| 9 | 14379 | `src/mcp/scheduler_tool.rs` | 357 | MCP scheduler tool for sublinear-time-solver |
| 10 | 10800 | `agentic-flow/src/reasoningbank/AdvancedMemory.ts` | 315 | ReasoningBank advanced memory in agentic-flow |

**Full paths**:
8. `~/repos/sublinear-time-solver/src/ultra_fast.rs`
9. `~/repos/sublinear-time-solver/src/mcp/scheduler_tool.rs`
10. `~/repos/agentic-flow/agentic-flow/src/reasoningbank/AdvancedMemory.ts`

**Key questions**:
- `ultra_fast.rs` (361 LOC): R70 found temporal-solver.rs imports UltraFastTemporalSolver (NOT temporal-lead-solver's predictor). Is this the UltraFast implementation? Does it claim O(1) or O(√n)? Is it genuine or another false sublinearity (4 known instances)? Does it compete with temporal-lead-solver or complement it? This file could resolve the "two parallel temporal solvers" mystery.
- `scheduler_tool.rs` (357 LOC): An MCP tool in the solver's Rust src/mcp/ directory. R53 found MCP tools BIMODAL (82-92% vs 18-28%). Does scheduler_tool.rs expose genuine solver scheduling or is it a thin wrapper? Does it use the genuine sublinear algorithms (backward_push, forward_push) or the false ones? How does it interface with MCP — direct JSON-RPC or delegating to another layer?
- `AdvancedMemory.ts` (315 LOC): In agentic-flow's ReasoningBank JS layer (same directory as queries.js). Does it implement advanced memory patterns (episodic, semantic, procedural)? Does it connect to the Rust ReasoningBank workspace or is it completely independent? R65 found HybridBackend reverses disconnected AgentDB — does AdvancedMemory bridge anything?

---

## Expected Outcomes

1. **CONNECTED tier CLEARED** — all 4 files read, zero blind dependency edges remain
2. **DEEP files**: 1,140 → ~1,150
3. **temporal-lead-solver COMPLETE**: validation.rs + wasm.rs complete the crate (lib.rs already DEEP from R70)
4. **ruv-swarm-persistence COMPLETE**: models.rs completes the data model (lib.rs + migrations.rs already DEEP)
5. **ReasoningBank network layer**: intent.rs reveals neural_bus routing intelligence
6. **UltraFastTemporalSolver identity**: Confirm or deny it's the "second temporal solver" from R70
7. **ReasoningBank JS layer map**: queries.js + AdvancedMemory.ts complete the agentic-flow ReasoningBank picture
8. **WASM scoreboard update**: temporal-lead-solver/wasm.rs — genuine (9th) or theatrical (6th)?

## DB Update Template

```javascript
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const sessionId = 71;
const today = new Date().toISOString().slice(0, 10);

// File IDs for this session:
// 13917: optimized.rs (351 LOC) — neural-network-implementation, CONNECTED
// 9009: models.rs (333 LOC) — ruv-swarm-persistence, CONNECTED
// 14176: validation.rs (314 LOC) — temporal-lead-solver, CONNECTED
// 14177: wasm.rs (245 LOC) — temporal-lead-solver, CONNECTED
// 13463: intent.rs (322 LOC) — reasoningbank-network
// 13444: adaptive.rs (318 LOC) — reasoningbank-learning
// 10818: queries.js (335 LOC) — agentic-flow reasoningbank
// 14501: ultra_fast.rs (361 LOC) — sublinear-time-solver
// 14379: scheduler_tool.rs (357 LOC) — sublinear-time-solver MCP
// 10800: AdvancedMemory.ts (315 LOC) — agentic-flow reasoningbank
```

## Domain Tags

- Files 1, 3, 4, 5, 6, 7, 8, 9, 10 → `memory-and-learning` (already tagged)
- File 2 → `swarm-coordination` (already tagged)

## Isolation Check

No selected files are in known-isolated subtrees (neuro-divergent, cuda-wasm, patches, scripts, simulation, validation). The ruv-swarm-persistence subtree is NOT isolated — it has inbound deps from lib.rs. The reasoningbank workspace is CONNECTED via R67's findings. All files are safe to read.

---

## Synthesis Doc Update Protocol (ADR-040)

**MANDATORY**: After all files are read and findings inserted into the DB, update the relevant `domains/*/analysis.md` files following the ADR-040 in-place protocol. See `daily-plan/R-plan-template.md` for full instructions.

### Current Max IDs (check before inserting)

**memory-and-learning/analysis.md**:
- Section 3a CRITICAL: last ID = C116 (R70)
- Section 3b HIGH: last ID = H99 (R70)
- New CRITICALs start at C117, new HIGHs at H100

**swarm-coordination/analysis.md**:
- Check current max IDs before adding findings for File 2 (models.rs)

### Rules

| Section | Action | NEVER Do |
|---------|--------|----------|
| **1. Current State Summary** | REWRITE in-place | Append session narrative |
| **2. File Registry** | ADD/UPDATE rows in subsystem tables | Create per-session tables |
| **3. Findings Registry** | ADD with next sequential ID to 3a/3b | Create `### R71 Findings` blocks |
| **4. Positives Registry** | ADD new positives with session tag | Re-list old positives |
| **5. Subsystem Sections** | UPDATE existing, CREATE new by topic | Per-session narrative blocks |
| **8. Session Log** | APPEND 2-5 line entry | Put findings here |

### Anti-Patterns

- **NEVER** create `### R71 Findings (Session date)` blocks
- **NEVER** append findings after Section 8
- **NEVER** re-list findings from previous sessions
- **NEVER** use finding IDs that collide with existing ones
