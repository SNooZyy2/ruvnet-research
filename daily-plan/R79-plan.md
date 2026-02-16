# R79 Execution Plan: ruv-swarm Rust Crate Internals + JS Benchmarks

**Date**: 2026-02-17
**Session ID**: 79
**Focus**: Complete ruv-swarm-wasm/src/ core files (5), finish last source files in ruv-swarm-mcp + transport (2), bridge ML neural layer (1), assess JS benchmark trustworthiness (2)
**Strategic value**: Extends R50/R72 ruv-swarm analysis from CLI demonstration framework into actual crate internals. Completes source-level coverage of 3 ruv-swarm crates (WASM/MCP/transport). Updates WASM scoreboard (currently 10:9). Tests R59 benchmark deception pattern in JS context. 83 nearby DEEP files provide dense cross-reference context.

## Rationale

R72 established that the ruv-swarm Rust CLI is a demonstration framework (main.rs 82-86%: clap 90-95%, execution 0%). But the underlying crates tell a different story — R50 found the bimodal split (memory.rs 95% vs spawn.rs 8%), and the 83 DEEP files in ruv-swarm/crates/ include production-quality code (ruv-swarm-mcp/service.rs 88-92%, persistence/sqlite.rs DEEP, transport/websocket.rs DEEP). The 5 remaining ruv-swarm-wasm/src/ files (agent, swarm, training, memory_pool, simd_tests) are the non-neural WASM internals that bridge between the already-read neural/SIMD/orchestration files and the base agent/swarm primitives.

For ruv-swarm-mcp, error.rs (194 LOC) is the last untouched source file — all 8 other src/ files are DEEP. Completing it closes the source layer (only tests/examples remain). Similarly, ruv-swarm-transport/src/lib.rs (178 LOC) is the crate root — the 4 transport implementations (in_process, protocol, shared_memory, websocket) are all DEEP, but we haven't read how they're organized and exported.

The ML neural_bridge.rs (234 LOC) bridges ruv-swarm-ml's model/ensemble/forecasting layers (all DEEP) to the neural network ecosystem. The two JS benchmarks (mcp-tools-benchmarks.js 328 LOC, benchmark.js 267 LOC) test whether the R59 deception pattern (criterion 88-95% vs standalone 8-25%) applies to the JS benchmark layer too, and benchmark-mcp directly validates R51's 256 MCP tools finding.

## Target: 10 files, ~2,303 LOC

---

### Cluster A: ruv-swarm-wasm Core Source (5 files, ~1,102 LOC)

Five remaining source files in ruv-swarm-wasm/src/. The crate already has 9 DEEP src/ files (lib.rs, agent_neural.rs, cognitive_diversity_wasm.rs, cognitive_neural_architectures.rs, neural_swarm_coordinator.rs, simd_ops.rs, simd_optimizer.rs, swarm_orchestration_wasm.rs, utils.rs). These 5 are the base-level primitives — the "non-neural" WASM layer underneath the cognitive/neural files already read.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 1 | 9051 | `ruv-swarm/crates/ruv-swarm-wasm/src/simd_tests.rs` | 273 | SIMD test suite — validates simd_ops.rs (420 LOC, DEEP). Are these real WASM SIMD tests or facade? |
| 2 | 9054 | `ruv-swarm/crates/ruv-swarm-wasm/src/training.rs` | 253 | WASM training module — connects to ML crate? R69 GHOST WASM had 27 models never compiled — is this real? |
| 3 | 9040 | `ruv-swarm/crates/ruv-swarm-wasm/src/agent.rs` | 200 | Base WASM agent (vs agent_neural.rs 552 LOC DEEP). How do base and neural agents relate? |
| 4 | 9052 | `ruv-swarm/crates/ruv-swarm-wasm/src/swarm.rs` | 191 | Base WASM swarm (vs swarm_orchestration_wasm.rs 757 LOC DEEP). Thin wrapper or independent? |
| 5 | 9046 | `ruv-swarm/crates/ruv-swarm-wasm/src/memory_pool.rs` | 185 | WASM memory pool — arena allocator? Connected to persistence crate? |

**Full paths**:
1. `~/repos/ruv-FANN/ruv-swarm/crates/ruv-swarm-wasm/src/simd_tests.rs`
2. `~/repos/ruv-FANN/ruv-swarm/crates/ruv-swarm-wasm/src/training.rs`
3. `~/repos/ruv-FANN/ruv-swarm/crates/ruv-swarm-wasm/src/agent.rs`
4. `~/repos/ruv-FANN/ruv-swarm/crates/ruv-swarm-wasm/src/swarm.rs`
5. `~/repos/ruv-FANN/ruv-swarm/crates/ruv-swarm-wasm/src/memory_pool.rs`

**Key questions**:
- `simd_tests.rs` (273 LOC): Does it use `wasm_bindgen_test`? Tests actual WASM SIMD intrinsics (i32x4, f32x4) or just Rust SIMD? Covers simd_ops.rs or simd_optimizer.rs or both?
- `training.rs` (253 LOC): Real gradient computation or facade? Does it import from ruv-swarm-ml or reimplement? Connected to R69's GHOST WASM models?
- `agent.rs` (200 LOC): Trait impl for `#[wasm_bindgen]`? Does agent_neural.rs extend this with neural methods? State management in WASM context?
- `swarm.rs` (191 LOC): WASM swarm coordination primitives. Uses SharedArrayBuffer? Connected to swarm_orchestration_wasm.rs?
- `memory_pool.rs` (185 LOC): Arena allocator for WASM linear memory? Custom allocator or standard? Performance-critical for WASM execution?

---

### Cluster B: ruv-swarm Crate Roots + Error Types (3 files, ~606 LOC)

Last untouched source files in three crates: ruv-swarm-mcp (8/9 src DEEP → 9/9), ruv-swarm-transport (4/5 src DEEP → 5/5), ruv-swarm-ml neural bridge (5/6 models DEEP → 6/6). Each completes its crate's source-level coverage.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 6 | 8988 | `ruv-swarm/crates/ruv-swarm-ml/src/models/neural_bridge.rs` | 234 | Last untouched ML model file. Bridges swarm-ml to neural ecosystem. How does it connect to agent_forecasting (813 LOC DEEP)? |
| 7 | 8965 | `ruv-swarm/crates/ruv-swarm-mcp/src/error.rs` | 194 | Last untouched MCP source file. R72: service.rs uses rmcp SDK. Custom error types or thiserror? |
| 8 | 9025 | `ruv-swarm/crates/ruv-swarm-transport/src/lib.rs` | 178 | Crate root — organizes 4 DEEP transport impls (in_process, protocol, shared_memory, websocket). Clean barrel or re-export mess? |

**Full paths**:
6. `~/repos/ruv-FANN/ruv-swarm/crates/ruv-swarm-ml/src/models/neural_bridge.rs`
7. `~/repos/ruv-FANN/ruv-swarm/crates/ruv-swarm-mcp/src/error.rs`
8. `~/repos/ruv-FANN/ruv-swarm/crates/ruv-swarm-transport/src/lib.rs`

**Key questions**:
- `neural_bridge.rs` (234 LOC): What neural network does it bridge to? Does it use ndarray/nalgebra for actual computation? How does it relate to the ensemble system (1006 LOC DEEP)?
- `error.rs` (194 LOC): Custom error enum with `thiserror`? Covers MCP protocol errors, tool execution errors, transport errors? How granular vs R72's rmcp SDK errors?
- `lib.rs` (178 LOC): How does it organize 4 transport backends? Feature-gated? Does it provide a transport trait that all backends implement?

---

### Cluster C: ruv-swarm JS Benchmarks (2 files, ~595 LOC)

Two benchmark files from `ruv-swarm/npm/src/`. R59 established the benchmark deception boundary (criterion 88-95% vs standalone 8-25%). These JS benchmarks test whether the same pattern holds in the npm layer. `mcp-tools-benchmarks.js` directly tests R51's finding of 256 MCP tools.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 9 | 9636 | `ruv-swarm/npm/src/mcp-tools-benchmarks.js` | 328 | MCP tool benchmarking — tests R51's 256 tools. Real performance measurement or hardcoded results? |
| 10 | 9608 | `ruv-swarm/npm/src/benchmark.js` | 267 | General swarm benchmarking. R59 pattern: criterion-grade or standalone-deceptive? |

**Full paths**:
9. `~/repos/ruv-FANN/ruv-swarm/npm/src/mcp-tools-benchmarks.js`
10. `~/repos/ruv-FANN/ruv-swarm/npm/src/benchmark.js`

**Key questions**:
- `mcp-tools-benchmarks.js` (328 LOC): Does it actually invoke MCP tools and measure latency? Or pre-populated results? How many of the 256 tools does it cover? Uses `perf_hooks` or wallclock?
- `benchmark.js` (267 LOC): Benchmarks what — agent spawn, task throughput, memory? Real async measurement or synchronous facade? Connects to the Rust benchmark system or independent?

---

## Expected Outcomes

1. **ruv-swarm-wasm/src/ 82% DEEP** — 14/17 source files covered (9 existing + 5 new)
2. **ruv-swarm-mcp/src/ COMPLETE** — 9/9 source files DEEP (only tests/examples remain)
3. **ruv-swarm-transport/src/ COMPLETE** — 5/5 source files DEEP (only tests remain)
4. **WASM scoreboard updated** — training.rs + memory_pool.rs + agent.rs + swarm.rs classifications
5. **Benchmark deception pattern tested in JS** — extend R59's criterion-vs-standalone boundary
6. **Neural bridge arc** — how ML crate connects to broader neural ecosystem
7. **DEEP count**: ~1,222 (post-R78 estimate) → ~1,232

## DB Update Template

```javascript
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const sessionId = 79;
const today = new Date().toISOString().slice(0, 10);

// File IDs for this session:
// 9051: simd_tests.rs (273 LOC) — ruv-fann-rust, PROXIMATE
// 9054: training.rs (253 LOC) — ruv-fann-rust, PROXIMATE
// 9040: agent.rs (200 LOC) — ruv-fann-rust, PROXIMATE
// 9052: swarm.rs (191 LOC) — ruv-fann-rust, PROXIMATE
// 9046: memory_pool.rs (185 LOC) — ruv-fann-rust, PROXIMATE
// 8988: neural_bridge.rs (234 LOC) — ruv-fann-rust, PROXIMATE
// 8965: error.rs (194 LOC) — ruv-fann-rust, PROXIMATE
// 9025: lib.rs (178 LOC) — ruv-fann-rust, PROXIMATE
// 9636: mcp-tools-benchmarks.js (328 LOC) — ruv-fann-rust, PROXIMATE
// 9608: benchmark.js (267 LOC) — ruv-fann-rust, PROXIMATE
```

## Domain Tags

- Files 1-5 → `swarm-coordination` (already tagged)
- Files 6-8 → `swarm-coordination` (already tagged)
- Files 9-10 → `swarm-coordination` (already tagged)
- File 6 (neural_bridge.rs) may also need `neural-network-implementation` if it bridges to external neural crates
- File 9 (mcp-tools-benchmarks.js) may also need `mcp-integration` if it tests cross-system MCP tools

## Isolation Check

No selected files are in known-isolated subtrees. All 10 files are in `ruv-swarm/` within `ruv-fann-rust`, which has 83 nearby DEEP files in the crates/ directory and 55 in npm/src/. This is one of the most densely connected areas of the entire codebase. The RELIABLE isolated subtrees remain untouched:
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
