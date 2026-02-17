# R87 Execution Plan: PROXIMATE Sweep — CLI, WASM Crates, ruv-swarm Tests

**Date**: 2026-02-17
**Session ID**: 87
**Focus**: Clear remaining PROXIMATE-tier files across CLI infrastructure, WASM algorithm crates, and ruv-swarm npm test layer
**Strategic value**: This session attacks the PROXIMATE tier from three angles: (1) two untouched WASM crate entry points that extend the WASM genuine/theatrical scoreboard, (2) CLI scheduler and benchmark files that validate or further qualify the "CLI = demo framework" thesis from R31/R71, and (3) the ruv-swarm npm test+config layer that closes coverage on the swarm-coordination domain's remaining PROXIMATE files.

## Rationale

With CONNECTED tier exhausted since R82, all remaining work draws from PROXIMATE and lower tiers. The current `smart_priority_gaps` queue has ~41 files remaining, heavily concentrated in two packages: `ruv-fann-rust` (12 PROXIMATE files in ruv-swarm/npm) and `sublinear-rust` (17 PROXIMATE + 4 NEARBY + 2 DOMAIN_ONLY). Many are small config files (eslint, webpack, rollup, jest configs) with minimal research value — this plan selects the **substantive** files while skipping pure build configs.

Three high-value targets justify inclusion: `temporal-neural-solver-wasm/src/lib.rs` (275 LOC) is an untouched WASM crate entry point that extends the temporal crate family arc (R86: "tensor 93% vs lead-solver 5-10%"); `bit-parallel-search/src/lib.rs` (198 LOC) is a potential genuine algorithm crate; and `src/cli/scheduler.rs` (293 LOC) directly extends the CLI demo-framework characterization from R31/R71/R86.

The ruv-swarm npm files (tests + config) are individually small but collectively close out the swarm-coordination domain's PROXIMATE tier. The `claude-config` package is ISOLATED with zero cross-package deps, so we skip its duplicate `helpers/memory.js` (ID 1207) in favor of the claude-flow-cli version (ID 2037).

## Target: 14 files, ~1,492 LOC

---

### Cluster A: Sublinear CLI + Benchmarks (3 files, ~456 LOC)

CLI scheduler and benchmark files from sublinear-time-solver. Extends R31/R71/R86 "CLI = demo framework" thesis and closes the consciousness-simple CLI variant.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 1 | 14307 | `src/cli/scheduler.rs` | 293 | NEARBY tier. CLI scheduler — does it implement real task scheduling or is it another demo skeleton like R86 monitor.rs? |
| 2 | 13845 | `benches/performance_benchmark.rs` | 104 | NEARBY tier. Criterion benchmarks — genuine like ruvector-benchmark.ts (R43: 92%) or deceptive like rustc_benchmarks (R43: 15%)? |
| 3 | 14304 | `src/cli/consciousness-simple.ts` | 59 | NEARBY tier. Simplified consciousness CLI — extends consciousness bimodal arc (R47/R49: infra 75-95% vs theory 0-5%) |

**Full paths**:
1. `~/repos/sublinear-time-solver/src/cli/scheduler.rs`
2. `~/repos/sublinear-time-solver/benches/performance_benchmark.rs`
3. `~/repos/sublinear-time-solver/src/cli/consciousness-simple.ts`

**Key questions**:
- `scheduler.rs` (293 LOC): Does it implement a real task scheduler with queue management, priorities, and concurrency? Or is it a CLI skeleton with placeholder methods like R86's monitor.rs?
- `performance_benchmark.rs` (104 LOC): Does it benchmark genuine algorithms (backward_push, HNSW) or fabricated ones? Are the benchmark targets real crate exports?
- `consciousness-simple.ts` (59 LOC): Is this a stripped-down version of the consciousness framework, or an independent simplified implementation? Does it exhibit the bimodal split?

---

### Cluster B: WASM + Algorithm Crates (3 files, ~506 LOC)

Two untouched WASM crate entry points plus a small quantization extension. Extends the WASM scoreboard (currently 15 genuine vs 12 theatrical) and the temporal crate family arc.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 4 | 14179 | `crates/temporal-neural-solver-wasm/src/lib.rs` | 275 | DOMAIN_ONLY tier. WASM entry point for temporal-neural-solver. Temporal crate family is BIMODAL — will this align with tensor (93%) or lead-solver (5-10%)? |
| 5 | 13872 | `crates/bit-parallel-search/bit-parallel-search/src/lib.rs` | 198 | DOMAIN_ONLY tier. Bit-parallel search — potential genuine algorithm using SIMD bit operations for string matching or set intersection |
| 6 | 13938 | `crates/neural-network-implementation/src/inference/quantization.rs` | 33 | PROXIMATE tier. Small quantization file — extends R82 quantization.rs (75-78%). Different path suggests inference-specific quantization vs training |

**Full paths**:
4. `~/repos/sublinear-time-solver/crates/temporal-neural-solver-wasm/src/lib.rs`
5. `~/repos/sublinear-time-solver/crates/bit-parallel-search/bit-parallel-search/src/lib.rs`
6. `~/repos/sublinear-time-solver/crates/neural-network-implementation/src/inference/quantization.rs`

**Key questions**:
- `temporal-neural-solver-wasm/lib.rs` (275 LOC): Does it wrap genuine temporal-tensor algorithms or is it another theatrical WASM like R43/R47? What wasm_bindgen exports does it expose?
- `bit-parallel-search/lib.rs` (198 LOC): Is this a genuine bit-parallel algorithm (Shift-Or, BNDM, bitwise set operations)? Or a facade with the name but scalar implementation like R85's simd_ops.rs?
- `inference/quantization.rs` (33 LOC): INT8/INT4 quantization for inference? How does it relate to R82's training-side quantization.rs?

---

### Cluster C: ruv-swarm npm Tests + Config (5 files, ~305 LOC)

Test and configuration files from the ruv-swarm npm package. Closes out most remaining PROXIMATE-tier swarm-coordination files.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 7 | 9822 | `ruv-swarm/npm/test-pr34-local.js` | 119 | PROXIMATE tier. Tests for PR#34 — what feature did this PR add? Validates ruv-swarm npm test quality |
| 8 | 9828 | `ruv-swarm/npm/verify-db-updates.js` | 57 | PROXIMATE tier. DB update verification — relates to R86 test-memory-storage.js (88-92%)? |
| 9 | 9825 | `ruv-swarm/npm/test-wasm-loading.js` | 48 | PROXIMATE tier. WASM loading test — validates R84 build.rs (85%) WASM compilation path |
| 10 | 9638 | `ruv-swarm/npm/src/memory-config.js` | 42 | PROXIMATE tier. Memory configuration — extends 8th disconnected persistence layer investigation |
| 11 | 9613 | `ruv-swarm/npm/src/claude-integration/env-template.js` | 39 | PROXIMATE tier. Environment template — what secrets/config does claude-integration expect? |

**Full paths**:
7. `~/repos/ruv-FANN/ruv-swarm/npm/test-pr34-local.js`
8. `~/repos/ruv-FANN/ruv-swarm/npm/verify-db-updates.js`
9. `~/repos/ruv-FANN/ruv-swarm/npm/test-wasm-loading.js`
10. `~/repos/ruv-FANN/ruv-swarm/npm/src/memory-config.js`
11. `~/repos/ruv-FANN/ruv-swarm/npm/src/claude-integration/env-template.js`

**Key questions**:
- `test-pr34-local.js` (119 LOC): What PR#34 feature is being tested? Are assertions meaningful or smoke-test level?
- `verify-db-updates.js` (57 LOC): Is this a real DB verification tool or a demo script? Does it test better-sqlite3 integration?
- `test-wasm-loading.js` (48 LOC): Does it actually load and execute WASM binaries, or just check file existence?
- `memory-config.js` (42 LOC): Does this configure a 9th disconnected persistence layer or connect to an existing one?

---

### Cluster D: MCP + Memory Integration (3 files, ~225 LOC)

MCP integration test, neural-pattern MCP starter, and claude-flow memory helper. Extends psycho-symbolic MCP arc and memory-and-learning domain.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 12 | 14032 | `crates/psycho-symbolic-reasoner/mcp-integration/test-build.js` | 90 | PROXIMATE tier. MCP integration build test — extends R80/R86 psycho-symbolic arc. 7th MCP protocol? |
| 13 | 2037 | `.claude/helpers/memory.js` | 84 | PROXIMATE tier. Claude-flow CLI memory helper — what memory operations does it provide? Duplicate of claude-config ID 1207 (ISOLATED package, skipped) |
| 14 | 14423 | `src/neural-pattern-recognition/scripts/start-mcp.js` | 51 | PROXIMATE tier. MCP server starter for neural-pattern-recognition. Extends MCP protocol count |

**Full paths**:
12. `~/repos/sublinear-time-solver/crates/psycho-symbolic-reasoner/mcp-integration/test-build.js`
13. `~/.npm-global/lib/node_modules/@claude-flow/cli/.claude/helpers/memory.js`
14. `~/repos/sublinear-time-solver/src/neural-pattern-recognition/scripts/start-mcp.js`

**Key questions**:
- `test-build.js` (90 LOC): Does it test a real MCP build pipeline or just check file existence? Is this a 7th MCP protocol or reuses an existing one?
- `helpers/memory.js` (84 LOC): What memory operations does this helper expose? Is it a wrapper around claude-flow memory commands or standalone localStorage/file persistence?
- `start-mcp.js` (51 LOC): What MCP SDK does it use? Does it register real tools or is it a template/placeholder?

---

## Expected Outcomes

1. WASM scoreboard updated: temporal-neural-solver-wasm classified as genuine or theatrical (16th or 13th)
2. bit-parallel-search classified: genuine algorithm crate or facade
3. CLI demo-framework thesis further validated or qualified by scheduler.rs
4. Consciousness arc fully closed (consciousness-simple.ts is last remaining file)
5. ruv-swarm npm PROXIMATE tier largely cleared (5 of 12 remaining files)
6. MCP protocol count potentially updated (7th protocol from psycho-symbolic or neural-pattern)
7. DEEP count: ~1,303 → ~1,317

## DB Update Template

```javascript
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const sessionId = 87;
const today = new Date().toISOString().slice(0, 10);

// File IDs for this session:
// 14307: scheduler.rs (293 LOC) — sublinear-rust, NEARBY
// 13845: performance_benchmark.rs (104 LOC) — sublinear-rust, NEARBY
// 14304: consciousness-simple.ts (59 LOC) — sublinear-rust, NEARBY
// 14179: temporal-neural-solver-wasm/lib.rs (275 LOC) — sublinear-rust, DOMAIN_ONLY
// 13872: bit-parallel-search/lib.rs (198 LOC) — sublinear-rust, DOMAIN_ONLY
// 13938: inference/quantization.rs (33 LOC) — sublinear-rust, PROXIMATE
// 9822: test-pr34-local.js (119 LOC) — ruv-fann-rust, PROXIMATE
// 9828: verify-db-updates.js (57 LOC) — ruv-fann-rust, PROXIMATE
// 9825: test-wasm-loading.js (48 LOC) — ruv-fann-rust, PROXIMATE
// 9638: memory-config.js (42 LOC) — ruv-fann-rust, PROXIMATE
// 9613: env-template.js (39 LOC) — ruv-fann-rust, PROXIMATE
// 14032: test-build.js (90 LOC) — sublinear-rust, PROXIMATE
// 2037: helpers/memory.js (84 LOC) — claude-flow-cli, PROXIMATE
// 14423: start-mcp.js (51 LOC) — sublinear-rust, PROXIMATE
```

## Domain Tags

- Files 14307, 13845, 14304 → `memory-and-learning` (already tagged)
- Files 14179, 13872, 13938 → `memory-and-learning` (already tagged)
- Files 9822, 9828, 9825, 9638, 9613 → `swarm-coordination` (already tagged)
- Files 14032, 14423 → `memory-and-learning` (already tagged)
- File 2037 → `memory-and-learning` (already tagged)

## Isolation Check

No selected files are in known-isolated subtrees:
- `neuro-divergent` (RELIABLE isolated, 176 files) — no files selected from here
- `cuda-wasm` (RELIABLE isolated, 205 files) — no files selected from here
- `patches` (RELIABLE isolated, 31 files) — no files selected from here
- `agentdb/simulation` (excluded R86) — no files selected from here
- `claude-config` package is ISOLATED — skipped ID 1207 (duplicate of ID 2037 in claude-flow-cli)

All selected files are in active, connected packages (`sublinear-rust`, `ruv-fann-rust`, `claude-flow-cli`).

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
