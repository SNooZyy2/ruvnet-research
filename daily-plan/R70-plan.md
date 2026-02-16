# R70 Execution Plan: Clear the CONNECTED Tier — Complete Every Dependency Chain

**Date**: 2026-02-16
**Session ID**: 70
**Focus**: All 10 remaining CONNECTED (tier 1) files — module roots, crate entry points, and bridge files that directly link to existing DEEP files. Clears the entire CONNECTED tier, leaving only PROXIMATE files in the priority queue.
**Strategic value**: After R70, every dependency edge in the knowledge graph connects two analyzed files. No more blind links.

## Rationale

The smart priority queue has 217 files remaining. Of these, only **10 are CONNECTED** — meaning they have a direct dependency edge to an already-DEEP file that has NOT been read. These are the highest-value files in the entire queue because they close gaps in the dependency graph.

Once these 10 are read:
- All 724+ dependency edges connect DEEP↔DEEP files (no blind spots)
- The 198 PROXIMATE files become the new frontier (co-located but not directly linked)
- Reasoning about any cross-file flow becomes reliable

This session is lighter on LOC (~1,175) but architecturally the most impactful remaining batch.

## Target: 10 files, ~1,175 LOC

---

### Cluster A: sublinear-time-solver Module Roots (5 files, ~447 LOC)

These are `mod.rs` / `lib.rs` / `core.rs` entry points that compose already-DEEP submodules into coherent APIs. Each one ties together 2-3 analyzed files and reveals the actual public interface of its subsystem.

| # | File ID | File | LOC | Connects To |
|---|---------|------|-----|-------------|
| 1 | 14357 | `src/core.rs` | 147 | → solver/random_walk.rs (DEEP) |
| 2 | 14458 | `src/sublinear/mod.rs` | 73 | → sublinear_neumann.rs, wasm.rs, lib.rs (all DEEP) |
| 3 | 14467 | `src/temporal_nexus/core/mod.rs` | 132 | → temporal_window.rs (DEEP) |
| 4 | 14490 | `src/temporal_nexus/dashboard/mod.rs` | 51 | → exporter.rs, metrics_collector.rs (DEEP) |
| 5 | 14172 | `crates/temporal-lead-solver/src/lib.rs` | 44 | → predictor.rs (92-95% O(√n)), solver.rs (DEEP) |

**Full paths**:
1. `~/repos/sublinear-time-solver/src/core.rs`
2. `~/repos/sublinear-time-solver/src/sublinear/mod.rs`
3. `~/repos/sublinear-time-solver/src/temporal_nexus/core/mod.rs`
4. `~/repos/sublinear-time-solver/src/temporal_nexus/dashboard/mod.rs`
5. `~/repos/sublinear-time-solver/crates/temporal-lead-solver/src/lib.rs`

**Key questions**:
- `core.rs` (147 LOC): Is this the crate's primary entry point? Does it re-export sublinear/, solver/, temporal_nexus/? R62 found 4 parallel APIs in the crate root — does core.rs unify them or add a 5th? Does it connect to random_walk.rs or just re-export it?
- `sublinear/mod.rs` (73 LOC): Does it expose the genuine sublinear algorithms (backward_push O(1/ε), forward_push O(vol/ε)) or also the FALSE ones (neumann, JL)? Does it gate the WASM bindings? R62 found the best algorithms orphaned — does mod.rs surface them?
- `temporal_nexus/core/mod.rs` (132 LOC): What does the temporal_nexus core expose? Does it compose quantum (R69: 92-95%) + dashboard + core into a coherent API? Does it define the TemporalState type?
- `temporal_nexus/dashboard/mod.rs` (51 LOC): At only 51 LOC, is this a thin barrel re-export or does it add composition logic? Does it wire exporter + metrics_collector together?
- `temporal-lead-solver/src/lib.rs` (44 LOC): The crate root for the solver that has the 2nd genuine sublinear algorithm (predictor.rs O(√n)). Does the lib.rs expose predictor properly? Is the public API usable or does it hide the best code?

---

### Cluster B: Psycho-Symbolic + Neural Bridges (3 files, ~411 LOC)

Bridge files that connect analyzed crates to their consumers. The temporal-solver binary connects neural-network-implementation to temporal-lead-solver. The extractors lib.rs connects to the MCP text-extractor (GENUINE WASM, R58). The wasm-loader-simple connects 4 DEEP MCP wrappers.

| # | File ID | File | LOC | Connects To |
|---|---------|------|-----|-------------|
| 6 | 13914 | `crates/neural-network-implementation/real-implementation/src/bin/temporal-solver.rs` | 192 | → temporal-lead-solver/core.rs (DEEP) |
| 7 | 14002 | `crates/psycho-symbolic-reasoner/extractors/src/lib.rs` | 75 | → text-extractor.ts (GENUINE WASM), sentiment.rs, loader.ts (DEEP) |
| 8 | 14028 | `crates/psycho-symbolic-reasoner/mcp-integration/src/wasm/wasm-loader-simple.ts` | 144 | → memory-manager.ts, text-extractor.ts, graph-reasoner.ts, loader.ts (all DEEP) |

**Full paths**:
6. `~/repos/sublinear-time-solver/crates/neural-network-implementation/real-implementation/src/bin/temporal-solver.rs`
7. `~/repos/sublinear-time-solver/crates/psycho-symbolic-reasoner/extractors/src/lib.rs`
8. `~/repos/sublinear-time-solver/crates/psycho-symbolic-reasoner/mcp-integration/src/wasm/wasm-loader-simple.ts`

**Key questions**:
- `temporal-solver.rs` (192 LOC): A binary crate — does it run the temporal-lead predictor as a standalone tool? Does it use the genuine O(√n) predictor or bypass it? R23 found neural-network-implementation BEST IN ECOSYSTEM (90-98%) — does this binary match that quality?
- `extractors/lib.rs` (75 LOC): The crate root for extractors. Does it compose sentiment.rs with the WASM text-extractor? R58 found text-extractor.ts GENUINE WASM — does lib.rs bridge Rust↔WASM? At 75 LOC, likely a thin barrel — but does it define the Extractor trait?
- `wasm-loader-simple.ts` (144 LOC): Connects to 4 DEEP files. R58 found memory-manager.ts MISLABELED (5th). R69 found graph-reasoner types genuine (88-92%). Does wasm-loader-simple actually load WASM or is it another theatrical loader? This is the key integration point for the psycho-symbolic MCP tools.

---

### Cluster C: ruv-swarm Persistence + Neural Pattern Recognition (2 files, ~317 LOC)

The remaining CONNECTED files from two separate subsystems: ruv-swarm's persistence crate root (which connects to migrations.rs from R69), and the neural-pattern-recognition barrel (which connects to real-time-monitor and zero-variance-detector).

| # | File ID | File | LOC | Connects To |
|---|---------|------|-----|-------------|
| 9 | 9006 | `ruv-swarm/crates/ruv-swarm-persistence/src/lib.rs` | 250 | → migrations.rs (R69 DEEP) |
| 10 | 14427 | `src/neural-pattern-recognition/src/index.js` | 67 | → real-time-monitor.js, zero-variance-detector.js (DEEP) |

**Full paths**:
9. `~/repos/ruv-FANN/ruv-swarm/crates/ruv-swarm-persistence/src/lib.rs`
10. `~/repos/sublinear-time-solver/src/neural-pattern-recognition/src/index.js`

**Key questions**:
- `ruv-swarm-persistence/lib.rs` (250 LOC): The crate root for ruv-swarm's 3rd persistence layer. R69 found migrations.rs defines 5 tables (agents/tasks/events/messages/metrics). Does lib.rs expose CRUD operations for them? Does it use rusqlite/sqlx/diesel? Does it overlap with ReasoningBank storage (94%)? At 250 LOC, this is the largest CONNECTED file — likely has real logic beyond re-exports.
- `neural-pattern-recognition/index.js` (67 LOC): Barrel for the JS neural pattern detection subsystem. Does it compose real-time-monitor + zero-variance-detector into a usable API? R40 found JS neural models inference-works-training-facade — does this follow the same pattern? At 67 LOC, likely a thin entry point.

---

## Expected Outcomes

1. **CONNECTED tier: CLEARED** — all 10 files read, every dependency edge in the DB connects two analyzed files
2. **Crate root architecture map**: 5 `mod.rs`/`lib.rs` files reveal what each subsystem actually exports vs hides
3. **sublinear-time-solver public API clarity**: Does core.rs surface the genuine algorithms or bury them?
4. **Psycho-symbolic MCP integration verdict**: Does wasm-loader-simple actually bridge Rust↔WASM↔MCP or is it theatrical?
5. **ruv-swarm persistence completeness**: With lib.rs + migrations.rs, is the 3rd persistence layer a real ORM?
6. **temporal-lead-solver usability**: Does the binary + lib.rs make predictor.rs (O(√n)) accessible?

## DB Update Template

```javascript
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const sessionId = 70;
const today = new Date().toISOString().slice(0, 10);

// File IDs for this session:
// 14357: src/core.rs (147 LOC)
// 14458: src/sublinear/mod.rs (73 LOC)
// 14467: src/temporal_nexus/core/mod.rs (132 LOC)
// 14490: src/temporal_nexus/dashboard/mod.rs (51 LOC)
// 14172: crates/temporal-lead-solver/src/lib.rs (44 LOC)
// 13914: crates/neural-network-implementation/real-implementation/src/bin/temporal-solver.rs (192 LOC)
// 14002: crates/psycho-symbolic-reasoner/extractors/src/lib.rs (75 LOC)
// 14028: crates/psycho-symbolic-reasoner/mcp-integration/src/wasm/wasm-loader-simple.ts (144 LOC)
// 9006: ruv-swarm/crates/ruv-swarm-persistence/src/lib.rs (250 LOC)
// 14427: src/neural-pattern-recognition/src/index.js (67 LOC)
```

## Domain Tags

- Files 1-8, 10 → `memory-and-learning` (already tagged)
- File 9 → `swarm-coordination` (already tagged)

## Isolation Check

No CONNECTED files are in known-isolated subtrees (neuro-divergent, cuda-wasm, patches, scripts, simulation all RELIABLE-isolated but none of these 10 files are in those directories).
