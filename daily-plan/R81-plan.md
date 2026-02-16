# R81 Execution Plan: sublinear JS Bridge + ruv-swarm npm Runtime + ReasoningBank Storage

**Date**: 2026-02-16
**Session ID**: 81
**Focus**: Map the JS/WASM bridge for sublinear-rust (2 files), assess ruv-swarm npm runtime modules (4 files), and extend ReasoningBank storage+learning coverage (4 files)
**Strategic value**: Tests three critical cross-layer boundaries: (1) whether genuine sublinear algorithms (backward_push R56, predictor R58) are callable from JS, (2) whether ruv-swarm npm runtime is genuine or follows the CLI demonstration-framework pattern (R72), (3) whether ReasoningBank storage backends and async learning match R78's "genuinely architected" assessment. All 10 files are PROXIMATE tier with 3-57 nearby DEEP files.

## Rationale

After R80 clears the 2 CONNECTED files and covers psycho-symbolic TS + neural-network periphery, the priority queue drops to ~105 PROXIMATE files. Three coherent clusters emerge from the top of that queue.

The sublinear-rust `js/` directory has 2 untouched files that form the JS-to-WASM bridge. R56 found backward_push.rs as GENUINE O(1/epsilon) sublinear and R58 found predictor.rs as GENUINE O(sqrt(n)) — but R70 showed sublinear/mod.rs ORPHANS these algorithms (2-5% SMOKING GUN). The JS bridge layer (`wasm-loader.js`, `mcp-dense-fix.js`) reveals whether the genuine algorithms are actually exposed to consumers or remain orphaned. This directly tests the "genuine but inaccessible" pattern.

The ruv-swarm npm runtime has 4 core modules remaining after R79's benchmark sweep. R72 established the CLI as a demonstration framework and R79 found the JS benchmarks deeply fabricated (benchmark.js 0-5%). But the *runtime* modules (neural-models barrel, GitHub coordinator, security, singleton container) haven't been assessed — they could follow the CLI facade pattern or contain genuine logic like the Rust crate internals (R79: simd_tests 88-92%, training ~80%).

The ReasoningBank extended files close the gap between R78's Rust-layer "COMPLETE" assessment and the actual storage backend implementation. The native adapter (`adapters/native.rs`) directly implements what `adapters/mod.rs` (R80) declares. The async_learner_v2 extends R67's "genuinely architected" learning system. The ReasoningBank benchmark tests R59's deception boundary in the AgentDB context. The test-retrieval.ts validates whether the TS ReasoningBank layer (R73) actually functions end-to-end.

## Target: 10 files, ~2,320 LOC

---

### Cluster A: sublinear-rust JS/WASM Bridge (2 files, ~579 LOC)

The JS-side bridge that loads sublinear Rust WASM modules. R56-R58 proved 2 genuine sublinear algorithms exist in Rust, but R70 showed they're orphaned from the module root. These 2 files reveal whether the JS consumer layer can actually reach them — or if the orphaning extends across the language boundary.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 1 | 14189 | `js/wasm-loader.js` | 356 | WASM loader for sublinear-rust. Does it load the genuine algorithms or the theatrical WASM? |
| 2 | 14187 | `js/mcp-dense-fix.js` | 223 | MCP "dense fix" — patch for MCP tool integration. What does it fix? Which MCP protocol? |

**Full paths**:
1. `~/repos/sublinear-time-solver/js/wasm-loader.js`
2. `~/repos/sublinear-time-solver/js/mcp-dense-fix.js`

**Key questions**:
- `wasm-loader.js` (356 LOC): Does it import from the genuine `backward_push`/`predictor` WASM modules or the theatrical ones? Uses `WebAssembly.instantiate` or a bundler? Feature detection for SIMD? Error handling for missing WASM?
- `mcp-dense-fix.js` (223 LOC): What MCP tools does this fix? Does it patch the ORPHANED MCP→Rust solver gap (R76)? Uses which MCP protocol of the 4 known?

---

### Cluster B: ruv-swarm npm Runtime Modules (4 files, ~934 LOC)

Four core runtime modules from `ruv-swarm/npm/src/`. R72 showed the CLI is a demonstration framework, R79 showed benchmarks are fabricated (0-5%). These runtime modules sit *between* the CLI and the Rust crates — they're either genuine runtime logic or thin wrappers over fabricated operations. The GitHub coordinator is particularly interesting given the ecosystem's 6 routing systems.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 3 | 9648 | `ruv-swarm/npm/src/neural-models/index.js` | 273 | Neural model barrel — which models? Connected to R40's inference-works/training-facade split? |
| 4 | 9625 | `ruv-swarm/npm/src/github-coordinator/gh-cli-coordinator.js` | 260 | GitHub CLI coordination — genuine `gh` CLI wrapper or stub? |
| 5 | 9667 | `ruv-swarm/npm/src/security.js` | 218 | Security module — real validation/crypto or placeholder? |
| 6 | 9668 | `ruv-swarm/npm/src/singleton-container.js` | 183 | Dependency injection — how does it wire the runtime together? |

**Full paths**:
3. `~/repos/ruv-FANN/ruv-swarm/npm/src/neural-models/index.js`
4. `~/repos/ruv-FANN/ruv-swarm/npm/src/github-coordinator/gh-cli-coordinator.js`
5. `~/repos/ruv-FANN/ruv-swarm/npm/src/security.js`
6. `~/repos/ruv-FANN/ruv-swarm/npm/src/singleton-container.js`

**Key questions**:
- `neural-models/index.js` (273 LOC): Barrel file or contains logic? Which neural models does it export? Does it bridge to the Rust ruv-swarm-ml crate (R79: neural_bridge.rs) or reimplement in JS?
- `gh-cli-coordinator.js` (260 LOC): Does it actually shell out to `gh` CLI? Manages PRs/issues? Is this the runtime backing for the R72 demonstration-framework CLI?
- `security.js` (218 LOC): Input validation, crypto operations, or access control? Uses native crypto modules? Connected to the ed25519 code found in R58?
- `singleton-container.js` (183 LOC): Classic DI container pattern? What services does it register? Does it wire up the 4+ routing systems or keep them parallel?

---

### Cluster C: ReasoningBank Storage + Learning Extended (4 files, ~807 LOC)

Extends R78's "ReasoningBank Rust COMPLETE" and R73's TS layer into the storage backend implementation, async learning v2, benchmark quality, and TS retrieval integration. The native.rs adapter directly implements what R80's adapters/mod.rs declares. The benchmark tests R59's deception boundary in the ReasoningBank context.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 7 | 12334 | `packages/agentdb/benchmarks/benchmark-reasoningbank.js` | 272 | ReasoningBank benchmark — R59 deception pattern test. Real perf measurement or hardcoded? |
| 8 | 13474 | `reasoningbank/crates/reasoningbank-storage/src/adapters/native.rs` | 181 | Native storage adapter — what backend? SQLite? Rocks? How does it implement the adapter trait? |
| 9 | 13446 | `reasoningbank/crates/reasoningbank-learning/src/async_learner_v2.rs` | 156 | Async learner v2 — extends R67's genuinely architected learning. What improved over v1? |
| 10 | 10835 | `agentic-flow/src/reasoningbank/test-retrieval.ts` | 198 | TS retrieval test — does the R73 TS ReasoningBank layer actually retrieve successfully? |

**Full paths**:
7. `~/repos/agentic-flow/packages/agentdb/benchmarks/benchmark-reasoningbank.js`
8. `~/repos/agentic-flow/reasoningbank/crates/reasoningbank-storage/src/adapters/native.rs`
9. `~/repos/agentic-flow/reasoningbank/crates/reasoningbank-learning/src/async_learner_v2.rs`
10. `~/repos/agentic-flow/agentic-flow/src/reasoningbank/test-retrieval.ts`

**Key questions**:
- `benchmark-reasoningbank.js` (272 LOC): Uses `perf_hooks` or wallclock? Tests store/retrieve/search operations? Pre-populated results or real measurements? Follows R59's criterion-vs-standalone pattern?
- `adapters/native.rs` (181 LOC): Which storage backend does "native" mean? How does it implement the trait from adapters/mod.rs? Async or sync? Error handling quality?
- `async_learner_v2.rs` (156 LOC): What's the v2 improvement? Streaming? Batch? Does it use the statistical strategy ranker from R78's optimizer.rs?
- `test-retrieval.ts` (198 LOC): End-to-end or unit test? Does it use real embeddings or hash-based (15th instance)? Does it connect to the 3+ disconnected AgentDB layers?

---

## Expected Outcomes

1. **Sublinear JS bridge assessed** — confirms or refutes whether genuine algorithms (R56/R58) are accessible from JS, or if R70's orphaning extends cross-language
2. **ruv-swarm npm runtime quality** — establishes whether runtime modules follow CLI facade pattern (R72: 0%) or Rust crate quality (R79: 88-92%)
3. **ReasoningBank storage implementation** — validates R78's "genuinely architected" with actual adapter code
4. **Benchmark deception test** — benchmark-reasoningbank.js tests R59 pattern in AgentDB/ReasoningBank context
5. **WASM scoreboard update** — wasm-loader.js reveals which WASM modules are actually loaded by consumers
6. **Hash-based embeddings** — test-retrieval.ts likely 15th+ instance if it uses hash fallback
7. **DEEP count**: ~1,242 (post-R80 estimate) → ~1,252

## DB Update Template

```javascript
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const sessionId = 81;
const today = new Date().toISOString().slice(0, 10);

// File IDs for this session:
// 14189: wasm-loader.js (356 LOC) — sublinear-rust, PROXIMATE
// 14187: mcp-dense-fix.js (223 LOC) — sublinear-rust, PROXIMATE
// 9648: neural-models/index.js (273 LOC) — ruv-fann-rust, PROXIMATE
// 9625: gh-cli-coordinator.js (260 LOC) — ruv-fann-rust, PROXIMATE
// 9667: security.js (218 LOC) — ruv-fann-rust, PROXIMATE
// 9668: singleton-container.js (183 LOC) — ruv-fann-rust, PROXIMATE
// 12334: benchmark-reasoningbank.js (272 LOC) — agentic-flow-rust, PROXIMATE
// 13474: adapters/native.rs (181 LOC) — agentic-flow-rust, PROXIMATE
// 13446: async_learner_v2.rs (156 LOC) — agentic-flow-rust, PROXIMATE
// 10835: test-retrieval.ts (198 LOC) — agentic-flow-rust, PROXIMATE
```

## Domain Tags

- Files 1-2 (wasm-loader.js, mcp-dense-fix.js) → `memory-and-learning` (already tagged); may also need `wasm-and-simd` if WASM loading is primary function
- Files 3-6 (ruv-swarm npm modules) → `swarm-coordination` (already tagged)
- Files 7-10 (ReasoningBank) → `memory-and-learning` (already tagged)
- File 4 (gh-cli-coordinator.js) may also need `mcp-and-tools` if it implements tool dispatch
- File 2 (mcp-dense-fix.js) may also need `mcp-and-tools` if it patches MCP protocol

## Isolation Check

No selected files are in known-isolated subtrees. Confirmed via `subtree_connectivity` check:
- RELIABLE isolated subtrees remain: neuro-divergent, cuda-wasm, patches, scripts, simulation, validation
- None of the 10 selected files belong to these subtrees
- sublinear-rust `js/` directory has 3 nearby DEEP files — low density but not isolated (it's a bridge layer)
- ruv-swarm `npm/src/` has 57 nearby DEEP files — densely connected
- agentic-flow-rust `reasoningbank/` and `packages/agentdb/` have 27-31 nearby DEEP files — well connected
- ISOLATED packages (claude-config, @ruvector/core) — none of our files belong to these

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
