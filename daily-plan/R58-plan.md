# R58 Execution Plan: Psycho-Symbolic MCP + Temporal Solvers + Goalie Reasoning

**Date**: 2026-02-16
**Session ID**: 58
**Focus**: Psycho-symbolic-reasoner MCP integration layer (TS + Rust), temporal-lead-solver prediction crate, WASM solver core, and Goalie advanced reasoning internals
**Parallel with**: R59 (no file overlap -- R58 = unexplored crate subsystems; R59 = benchmarks + JS solver layer)

## IMPORTANT: Parallel Execution Notice

This plan runs IN PARALLEL with R59. The file lists are strictly non-overlapping:
- **R58 covers**: psycho-symbolic-reasoner MCP integration (3 TS files), psycho-symbolic extractors (1 Rust file), temporal-lead-solver crate (2 Rust files), goalie advanced reasoning core (3 TS files)
- **R59 covers**: benches/ (2 Rust files), neural-network-implementation benchmarks (1 Rust file), src/benchmarks/ (1 TS file), js/ solver layer (3 JS files), matrix-utils (1 JS file), ruv-fann-rust benchmarking (2 Rust files)
- **ZERO shared files** between R58 and R59
- **R58 has NO benches/, js/, src/benchmarks/, src/utils/, or ruv-fann-rust files**
- **R59 has NO psycho-symbolic-reasoner, temporal-lead-solver, or goalie files**
- Do NOT read or analyze any file from R59's list (see R59-plan.md for that list)

## Rationale

- **psycho-symbolic-reasoner MCP integration is completely unexplored**: R55 read psycho-symbolic Rust internals (3-4x better than TS), but the MCP integration layer (text-extractor, memory-manager, server.ts) has never been examined. This is the bridge between the Rust reasoner and external MCP consumers
- **temporal-lead-solver is an entirely new crate**: predictor.rs (426 LOC) and solver.rs (418 LOC) have never been read. The crate name suggests temporal prediction -- potentially related to R55's Temporal Nexus or R56's temporal_window, but possibly disconnected
- **Goalie advanced reasoning never deep-read**: R46 REVERSED Goalie assessment (internal engines real, cli.ts 88-92%), but the actual reasoning engine (`advanced-reasoning-engine.ts` 396 LOC) and plugins (`self-consistency-plugin.ts` 455 LOC) have never been examined. The `ed25519-verifier-real.ts` (406 LOC) filename suffix "-real" is unusual and may indicate genuine cryptographic verification
- **patterns.rs** in psycho-symbolic extractors is the Rust pattern extraction core -- R55 showed Rust quality is 3-4x better than TS equivalents, making this a high-value target

## Target: 9 files, ~3,754 LOC

---

### Cluster A: Psycho-Symbolic MCP Integration (4 files, ~1,653 LOC)

The psycho-symbolic-reasoner crate has a full MCP integration subsystem that has never been examined. R55 found the Rust core is genuine (3-4x better than TS), but the MCP bridge is unknown -- is it a real integration or a shell?

| # | File ID | File | LOC | Domain | Package |
|---|---------|------|-----|--------|---------|
| 1 | 14031 | `crates/psycho-symbolic-reasoner/mcp-integration/src/wrappers/text-extractor.ts` | 439 | memory-and-learning | sublinear-rust |
| 2 | 14027 | `crates/psycho-symbolic-reasoner/mcp-integration/src/wasm/memory-manager.ts` | 393 | memory-and-learning | sublinear-rust |
| 3 | 14060 | `crates/psycho-symbolic-reasoner/src/typescript/mcp/server.ts` | 431 | memory-and-learning | sublinear-rust |
| 4 | 14003 | `crates/psycho-symbolic-reasoner/extractors/src/patterns.rs` | 390 | memory-and-learning | sublinear-rust |

**Full paths**:
1. `~/repos/sublinear-time-solver/crates/psycho-symbolic-reasoner/mcp-integration/src/wrappers/text-extractor.ts`
2. `~/repos/sublinear-time-solver/crates/psycho-symbolic-reasoner/mcp-integration/src/wasm/memory-manager.ts`
3. `~/repos/sublinear-time-solver/crates/psycho-symbolic-reasoner/src/typescript/mcp/server.ts`
4. `~/repos/sublinear-time-solver/crates/psycho-symbolic-reasoner/extractors/src/patterns.rs`

**Key questions**:
- `text-extractor.ts`: What does the MCP wrapper extract?
  - Does it genuinely extract text from documents (NLP pipeline) or wrap a stub?
  - Does it call into the Rust psycho-symbolic core via NAPI/WASM or is it pure TS?
  - R55 found Rust quality 3-4x better than TS -- does the TS wrapper add value or degrade quality?
  - Does it expose proper MCP tool definitions with input/output schemas?
- `memory-manager.ts`: How does WASM memory work in the MCP integration?
  - Does it manage real WASM linear memory (ArrayBuffer, grow operations)?
  - R47 found 3rd theatrical WASM, R49 found ReasoningBank WASM 100% GENUINE -- which pattern does this follow?
  - Does it handle memory lifecycle (alloc/dealloc) or is it a facade?
  - Is this connected to the Rust WASM solver (wasm-solver crate) or independent?
- `server.ts`: What MCP server does psycho-symbolic-reasoner expose?
  - R51 found the main MCP server is GENUINE (256 tools, raw JSON-RPC 2.0) -- does this follow the same pattern?
  - Does it expose psycho-symbolic reasoning as MCP tools?
  - Does it implement proper JSON-RPC or wrap execSync calls (like claudeFlowSdkServer.ts)?
  - How many tools does it register and what capabilities do they expose?
- `patterns.rs`: What patterns does the Rust extractor find?
  - R55 showed Rust code is 3-4x more genuine than TS -- is this consistent?
  - Does it implement real pattern matching (regex, AST-based, or ML-based)?
  - Does it use the crate's cognitive/symbolic reasoning infrastructure?
  - Does it connect to the TS MCP layer or operate independently?

**Follow-up context**:
- R55: psycho-symbolic Rust 3-4x better than TS
- R47: 3rd theatrical WASM
- R49: ReasoningBank WASM 100% GENUINE -- WASM quality varies widely
- R51: Main MCP server genuine (256 tools), but claudeFlowSdkServer.ts BIMODAL

---

### Cluster B: Temporal Lead Solver (2 files, ~844 LOC)

The temporal-lead-solver crate is entirely unexplored. It may relate to R55's Temporal Nexus (genuine physics) or R56's temporal_window (production sliding window), or it may be an independent temporal prediction system.

| # | File ID | File | LOC | Domain | Package |
|---|---------|------|-----|--------|---------|
| 5 | 14174 | `crates/temporal-lead-solver/src/predictor.rs` | 426 | memory-and-learning | sublinear-rust |
| 6 | 14175 | `crates/temporal-lead-solver/src/solver.rs` | 418 | memory-and-learning | sublinear-rust |

**Full paths**:
5. `~/repos/sublinear-time-solver/crates/temporal-lead-solver/src/predictor.rs`
6. `~/repos/sublinear-time-solver/crates/temporal-lead-solver/src/solver.rs`

**Key questions**:
- `predictor.rs`: What does the temporal predictor predict?
  - Does it implement genuine time-series prediction (ARIMA, exponential smoothing, neural)?
  - "Lead" in the crate name suggests prediction ahead of time -- financial trading, latency prediction, or solver convergence?
  - R56 found backward_push.rs is GENUINE O(1/epsilon) sublinear -- does this predictor use similar algorithmic rigor?
  - Does it use real numerical methods or hardcoded values?
- `solver.rs`: How does the temporal solver differ from the main solver?
  - R56 found optimized_solver.rs at 72-76% (standard CG+SIMD) -- is this a variant or different algorithm?
  - Does it solve temporal optimization problems (dynamic programming, optimal control)?
  - Does it integrate with predictor.rs or are they independent?
  - At 418 LOC in Rust, there's enough room for real algorithmic content

**Follow-up context**:
- R55: Temporal Nexus genuine physics (80.75%)
- R56: backward_push.rs 92-95% GENUINE sublinear, optimized_solver.rs 72-76% standard
- R56: temporal_window.rs 88-92% production sliding window -- orphaned from quantum physics

---

### Cluster C: Goalie Advanced Reasoning (3 files, ~1,257 LOC)

R46 REVERSED the Goalie assessment from R41 "COMPLETE FACADE" to "internal engines real (cli.ts 88-92%)". But the advanced reasoning engine itself and its plugins have never been deep-read. This cluster examines the actual reasoning logic.

| # | File ID | File | LOC | Domain | Package |
|---|---------|------|-----|--------|---------|
| 7 | 14211 | `npx/goalie/src/core/advanced-reasoning-engine.ts` | 396 | memory-and-learning | sublinear-rust |
| 8 | 14214 | `npx/goalie/src/core/ed25519-verifier-real.ts` | 406 | memory-and-learning | sublinear-rust |
| 9 | 14226 | `npx/goalie/src/plugins/advanced-reasoning/self-consistency-plugin.ts` | 455 | memory-and-learning | sublinear-rust |

**Full paths**:
7. `~/repos/sublinear-time-solver/npx/goalie/src/core/advanced-reasoning-engine.ts`
8. `~/repos/sublinear-time-solver/npx/goalie/src/core/ed25519-verifier-real.ts`
9. `~/repos/sublinear-time-solver/npx/goalie/src/plugins/advanced-reasoning/self-consistency-plugin.ts`

**Key questions**:
- `advanced-reasoning-engine.ts`: What reasoning does Goalie actually perform?
  - R46 found cli.ts 88-92% with real engines -- does this file contain the actual engine?
  - Does it implement chain-of-thought, tree-of-thought, or other LLM reasoning patterns?
  - Does it call external LLM APIs or perform local reasoning?
  - R50 confirmed Goalie genuine -- does the reasoning engine use real algorithmic techniques or prompt engineering?
- `ed25519-verifier-real.ts`: Why is "-real" in the filename?
  - Does it implement genuine Ed25519 signature verification (using tweetnacl, noble-ed25519, or crypto.subtle)?
  - The "-real" suffix suggests there may also be a fake/mock version -- does it exist?
  - Does Goalie use Ed25519 for authenticating reasoning results or verifying agent identity?
  - At 406 LOC, this is substantial for a verifier -- may include full key management
- `self-consistency-plugin.ts`: Does self-consistency sampling work?
  - Self-consistency is a known LLM technique (sample multiple reasoning paths, vote on answer)
  - Does it implement genuine multi-sample voting or is it a stub?
  - Does it integrate with the advanced-reasoning-engine?
  - R46/R50 confirmed Goalie has real internal engines -- does this plugin follow the pattern?

**Follow-up context**:
- R41: Goalie initially assessed as COMPLETE FACADE
- R46: REVERSAL -- internal engines real (cli.ts 88-92%)
- R50: Goalie genuine confirmed
- R43: claude-integration/ is setup toolkit NOT API -- Goalie's crypto verifier may serve a different purpose

---

## Expected Outcomes

- **Psycho-symbolic MCP truth**: Whether the TS MCP integration layer matches R55's genuine Rust core or degrades quality (like the systemic Rust-vs-JS split)
- **WASM memory verdict**: Whether memory-manager.ts follows R49's "100% GENUINE" WASM pattern or R47's "theatrical WASM" pattern
- **Temporal-lead-solver discovery**: Whether this is a genuine temporal prediction crate or another stub, and how it relates to the Temporal Nexus ecosystem
- **Goalie reasoning depth**: Whether the advanced reasoning engine contains real algorithmic techniques (confirming R46/R50 reversals) or is surface-level
- **Ed25519 verification**: Whether Goalie has genuine cryptographic verification (rare in this codebase) or is another mislabeled file
- **Self-consistency implementation**: Whether this is a genuine multi-sample reasoning plugin or a stub
- **Rust-vs-TS quality gap**: patterns.rs quality vs the 3 TS MCP files -- does the 3-4x gap from R55 hold?

## Stats Target

- ~9 file reads, ~3,754 LOC
- DEEP files: 1,010 -> ~1,019
- Expected findings: 50-70 (9 files across 3 subsystems, mixed Rust+TS)

## Cross-Session Notes

- **ZERO overlap with R59**: R59 covers benchmarks + JS solvers + matrix-utils + ruv-fann-rust. No shared files or directories.
- **Extends R55**: psycho-symbolic MCP layer extends R55's internals assessment (Rust 3-4x better than TS)
- **Extends R46/R50**: Goalie reasoning engine extends the reversal arc (R41 facade -> R46 real -> R50 confirmed)
- **Extends R56**: temporal-lead-solver extends R56's temporal_window and backward_push findings
- **Extends R49**: WASM memory-manager tests R49's "ReasoningBank WASM 100% GENUINE" pattern
- **NEW: temporal-lead-solver** is a completely unexplored crate
- **NEW: Goalie advanced reasoning** internals (reasoning engine + plugins) never deep-read
- **NEW: Ed25519 verification** is the first cryptographic verification file examined in Goalie
- Combined DEEP files from R58+R59: 1,010 -> ~1,029 (approximately +19)
