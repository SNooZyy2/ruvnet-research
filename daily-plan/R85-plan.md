# R85 Execution Plan: ruv-swarm npm Runtime, Neural-Net Crate Completion, Cross-Arc Closures

**Date**: 2026-02-17
**Session ID**: 85
**Focus**: ruv-swarm npm runtime code (hooks, memory CLI, build), neural-network-implementation final 4 files, ReasoningBank async+types, sublinear simplified entry point
**Strategic value**: (1) Pairs with R84's test suite — R84 reads the TESTS, R85 reads the RUNTIME CODE those tests exercise, giving both sides of the persistence layer verdict. (2) Finishes the last 4 untouched neural-network-implementation files, closing the R23 assessment completely. (3) Closes ReasoningBank Rust async learning and TS types arcs. (4) lib_simple.rs extends the R70 sublinear smoking gun arc.

## Rationale

R84 is currently reading the ruv-swarm npm TEST files (test-flags.js, test-db-updates.js, test-mcp-db.js, test-mcp-persistence.js, test-mcp-fixes.js). R85 complements this by reading the RUNTIME infrastructure those tests validate: the build system, hooks integration, and memory CLI. Together, R84+R85 provide the definitive test-vs-implementation picture for the "6+ disconnected persistence layers" cumulative finding and the "ruv-swarm npm BIMODAL" verdict from R81.

The neural-network-implementation crate has been the subject of a multi-session qualification arc: R23 scored it 90-98% (BEST IN ECOSYSTEM), R82 found solver_gate_simple.rs at 0-5% (qualifying integration layers), R83 found train.rs at 92-95% (countering R82) but losses.rs at 68-75% (further qualifying at the math layer). Four files remain: callbacks.rs, preprocessing.rs, simd_ops.rs, and memory_pool.rs. These are the inference and training support modules — reading them closes the crate completely and provides the final R23 qualification verdict.

The cross-arc cluster closes three small but important gaps: async_learner.rs extends the ReasoningBank Rust arc (R73-R78: COMPLETE), reasoningbank-types.ts provides the foundational type definitions for the ReasoningBank TS layer (R83: test layer 92-95% GENUINE), and lib_simple.rs extends the R70 sublinear/mod.rs SMOKING GUN arc by revealing an alternative simplified entry point.

## Target: 11 files, ~951 LOC

---

### Cluster A: ruv-swarm npm Runtime + Hooks (4 files, ~530 LOC)

The runtime counterpart to R84's test suite. These files reveal what the npm package actually builds, how hooks integrate with GitHub coordination, and whether the memory CLI provides real persistence or another disconnected layer. R81 found ruv-swarm npm BIMODAL (real infrastructure but theatrical integration) — this cluster tests whether the runtime hooks and memory layers are the "real" or "theatrical" side.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 1 | 9600 | `ruv-swarm/npm/scripts/build.js` | 167 | Build script — what does npm package compile? Reveals real vs bundled artifacts |
| 2 | 9624 | `ruv-swarm/npm/src/github-coordinator/claude-hooks.js` | 162 | GitHub coordinator hooks — hook pipeline integration point |
| 3 | 9490 | `ruv-swarm/npm/bin/ruv-swarm-memory.js` | 119 | Memory CLI command — 7+ disconnected persistence layers: does this connect any? |
| 4 | 9626 | `ruv-swarm/npm/src/hooks/cli.js` | 82 | Hooks CLI layer — how hooks are exposed to CLI consumers |

**Full paths**:
1. `~/repos/ruv-FANN/ruv-swarm/npm/scripts/build.js`
2. `~/repos/ruv-FANN/ruv-swarm/npm/src/github-coordinator/claude-hooks.js`
3. `~/repos/ruv-FANN/ruv-swarm/npm/bin/ruv-swarm-memory.js`
4. `~/repos/ruv-FANN/ruv-swarm/npm/src/hooks/cli.js`

**Key questions**:
- `build.js` (167 LOC): Does it compile TypeScript/Rust→JS? Does it bundle WASM? Does it copy pre-built artifacts or actually run a build pipeline? What dependencies does it pull?
- `claude-hooks.js` (162 LOC): Does this implement real Claude Code hook integration (pre-edit, post-edit, pre-task, post-task)? Does it connect to the 5+ MCP protocols or operate independently? Is it the bridge between GitHub coordination and the hooks pipeline?
- `ruv-swarm-memory.js` (119 LOC): What persistence backend does it use (SQLite, file-based, in-memory)? Does it connect to AgentDB, the MCP memory system, or a standalone layer? Is this the 8th disconnected persistence layer?
- `cli.js` (82 LOC): What hooks does it expose? Does it use the same hook registration as claude-flow's hooks system, or is it a parallel implementation?

---

### Cluster B: Neural-Network-Implementation Final Files (4 files, ~240 LOC)

The LAST 4 untouched files in the neural-network-implementation crate. After this cluster, every file in the crate has been deep-read, enabling a definitive final assessment. The running score: R23 90-98% core, R82 solver_gate 0-5% (integration facade), R83 train.rs 92-95% + losses.rs 68-75%. These files cover training callbacks, data preprocessing, SIMD inference, and memory pooling — the support infrastructure that determines whether the crate is genuinely production-ready or has facade support layers.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 5 | 13952 | `crates/neural-network-implementation/src/training/callbacks.rs` | 99 | Training callbacks — real monitoring/early-stopping or stub? |
| 6 | 13934 | `crates/neural-network-implementation/src/data/preprocessing.rs` | 66 | Data preprocessing — normalization, augmentation, or placeholder? |
| 7 | 13939 | `crates/neural-network-implementation/src/inference/simd_ops.rs` | 41 | SIMD inference ops — connects to ruvector-core AVX-512/AVX2/NEON arc |
| 8 | 13936 | `crates/neural-network-implementation/src/inference/memory_pool.rs` | 34 | Inference memory management — arena allocation or basic alloc? |

**Full paths**:
5. `~/repos/sublinear-time-solver/crates/neural-network-implementation/src/training/callbacks.rs`
6. `~/repos/sublinear-time-solver/crates/neural-network-implementation/src/data/preprocessing.rs`
7. `~/repos/sublinear-time-solver/crates/neural-network-implementation/src/inference/simd_ops.rs`
8. `~/repos/sublinear-time-solver/crates/neural-network-implementation/src/inference/memory_pool.rs`

**Key questions**:
- `callbacks.rs` (99 LOC): Does it implement real training callbacks (EarlyStopping, ModelCheckpoint, LearningRateScheduler)? Does it integrate with the training loop in train.rs (R83: 92-95%)? Does it have state tracking or is it fire-and-forget?
- `preprocessing.rs` (66 LOC): What preprocessing operations are implemented (normalization, standardization, one-hot, tokenization)? Does it operate on real tensor types or generic arrays? Does it have batch processing support?
- `simd_ops.rs` (41 LOC): Does it use `std::arch` SIMD intrinsics (AVX2, NEON) like ruvector-core, or `packed_simd`/`portable_simd`? Is it genuine SIMD or scalar fallback? What operations does it accelerate (matmul, dot product, activation)?
- `memory_pool.rs` (34 LOC): Does it implement arena/slab/pool allocation, or is it a thin wrapper around `Vec`? Does it have pre-allocation for inference batches? Is it the same pattern as ruvector-core's memory management?

---

### Cluster C: Cross-Arc Completions (3 files, ~181 LOC)

Three files closing gaps in major arcs. async_learner.rs is the Rust async learning module for ReasoningBank (R73-R78 declared the arc COMPLETE — this file validates whether the Rust async layer supports that verdict). reasoningbank-types.ts provides the foundational TypeScript type definitions that the R83 test layer (92-95% GENUINE) builds on. lib_simple.rs is a simplified entry point for the sublinear-time-solver crate, extending the R70 SMOKING GUN arc that found genuine O(1/epsilon) algorithms.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 9 | 13445 | `reasoningbank/crates/reasoningbank-learning/src/async_learner.rs` | 70 | ReasoningBank Rust async learning — validates R73-R78 COMPLETE |
| 10 | 10806 | `agentic-flow/src/reasoningbank/config/reasoningbank-types.ts` | 58 | ReasoningBank TS type definitions — foundation for R83 test layer |
| 11 | 14371 | `src/lib_simple.rs` | 53 | Simplified sublinear entry point — extends R70 smoking gun |

**Full paths**:
9. `~/repos/agentic-flow/reasoningbank/crates/reasoningbank-learning/src/async_learner.rs`
10. `~/repos/agentic-flow/agentic-flow/src/reasoningbank/config/reasoningbank-types.ts`
11. `~/repos/sublinear-time-solver/src/lib_simple.rs`

**Key questions**:
- `async_learner.rs` (70 LOC): Does it implement genuine async learning (tokio tasks, async trait, channels)? Does it connect to the ReasoningBank trajectory/verdict system? Is this a duplicate of R81's async_learner_v2.rs or a separate implementation?
- `reasoningbank-types.ts` (58 LOC): What type interfaces does it define (Trajectory, Verdict, Pattern, Memory)? Do these types match the Rust struct definitions? Are they used by the test layer (R83) or orphaned?
- `lib_simple.rs` (53 LOC): Does it re-export the genuine sublinear algorithms (backward_push, LocalPush)? Or is it a simplified facade that bypasses the real implementations? Does it provide the "simple" API that the ORPHANED JS bridge (R81) was trying to call?

---

## Expected Outcomes

1. **ruv-swarm npm runtime verdict**: Paired with R84's test results, determine whether npm hooks/memory are genuine integration points or disconnected layers
2. **Neural-network-implementation COMPLETE**: All files deep-read — final crate-wide assessment replaces the R23→R82→R83 qualification chain
3. **SIMD arc extended**: simd_ops.rs reveals whether neural-net uses the same SIMD strategy as ruvector-core
4. **ReasoningBank arc validated**: async_learner.rs supports or qualifies the R73-R78 COMPLETE verdict
5. **ReasoningBank TS types mapped**: foundational types for the 92-95% genuine test layer
6. **Sublinear simple API assessed**: whether lib_simple.rs connects the genuine algorithms to a usable API
7. **DEEP count**: 1,272 → ~1,283 (assuming R84 adds ~11)

## DB Update Template

```javascript
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const sessionId = 85;
const today = new Date().toISOString().slice(0, 10);

// File IDs for this session:
// 9600: build.js (167 LOC) — ruv-fann-rust, PROXIMATE
// 9624: claude-hooks.js (162 LOC) — ruv-fann-rust, PROXIMATE
// 9490: ruv-swarm-memory.js (119 LOC) — ruv-fann-rust, PROXIMATE
// 9626: cli.js (82 LOC) — ruv-fann-rust, PROXIMATE
// 13952: callbacks.rs (99 LOC) — sublinear-rust, PROXIMATE
// 13934: preprocessing.rs (66 LOC) — sublinear-rust, PROXIMATE
// 13939: simd_ops.rs (41 LOC) — sublinear-rust, PROXIMATE
// 13936: memory_pool.rs (34 LOC) — sublinear-rust, PROXIMATE
// 13445: async_learner.rs (70 LOC) — agentic-flow-rust, PROXIMATE
// 10806: reasoningbank-types.ts (58 LOC) — agentic-flow-rust, PROXIMATE
// 14371: lib_simple.rs (53 LOC) — sublinear-rust, PROXIMATE
```

## Domain Tags

- Files 9600, 9624, 9490, 9626 → `swarm-coordination` (already tagged)
- Files 13952, 13934, 13939, 13936, 14371 → `memory-and-learning` (already tagged)
- Files 13445, 10806 → `memory-and-learning`, `agentic-flow` (already tagged)

## Isolation Check

No selected files are in RELIABLY isolated subtrees. Checked:
- `neuro-divergent` (170 untouched, RELIABLE isolated) — no files selected
- `cuda-wasm` (203 untouched, RELIABLE isolated) — no files selected
- `ruvector-rust/patches` (22 untouched, RELIABLE isolated) — no files selected
- `ruvector-rust/scripts` (49 untouched, RELIABLE isolated) — no files selected
- `agentdb/simulation` (74 untouched, RELIABLE isolated) — no files selected (causal-reasoning.ts and reflexion-learning.ts from queue EXCLUDED due to isolation)
- `sublinear-rust/validation` (22 untouched, RELIABLE isolated) — no files selected

Isolated packages: `claude-config` (ISOLATED, 744 files) — helpers/memory.js (file 1207) excluded. `@ruvector/core` (ISOLATED, 2 files) — no files selected.

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
