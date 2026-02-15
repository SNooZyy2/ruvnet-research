# R43 Execution Plan: ruv-swarm Claude Integration + AgentDB Benchmarks + Sublinear WASM

**Date**: 2026-02-15
**Session ID**: 43
**Focus**: ruv-swarm Claude integration module, AgentDB ruvector benchmarks + ReasoningBank, sublinear WASM tools + psycho-symbolic benchmarks

## Rationale

- swarm-coordination domain at 15.58% coverage — top 4 priority gaps are all `ruv-swarm/npm/` files, with `claude-integration/` being a complete untouched module
- R41 found goalie/tools.ts was COMPLETE FACADE — need to verify whether ruv-swarm's claude-integration follows the same pattern or contains real API integration
- R40 found AgentDB intelligence was mixed quality (67%) — ruvector-benchmark.ts (1,264 LOC) is the largest untouched benchmark file, could reveal real performance characteristics
- ReasoningBank in agentic-flow is completely unexplored — demo-comparison.ts may show whether it's genuine RL or another facade
- R39 confirmed "WASM loaded but unused" in solver.ts — wasm-sublinear-complete.ts (710 LOC) is the dedicated WASM tool layer, should resolve whether WASM integration is real
- psycho-symbolic-reasoner benchmarks follow up R41's finding that the knowledge graph is genuine (78%)

## Target: ~10 files, ~7,250 LOC

---

### Cluster A: ruv-swarm Claude Integration Module (4 files, ~2,726 LOC)

| # | File ID | File | LOC | Domain |
|---|---------|------|-----|--------|
| 1 | 9614 | `ruv-swarm/npm/src/claude-integration/index.js` | 209 | swarm-coordination |
| 2 | 9610 | `ruv-swarm/npm/src/claude-integration/advanced-commands.js` | 561 | swarm-coordination |
| 3 | 9615 | `ruv-swarm/npm/src/claude-integration/remote.js` | 408 | swarm-coordination |
| 4 | 9612 | `ruv-swarm/npm/src/claude-integration/docs.js` | 1548 | swarm-coordination |

**Key questions**:
- Is this a real Claude API integration (HTTP calls, auth, streaming) or template generation?
- Does `remote.js` implement actual network transport (WebSocket, SSE) or return stubs?
- Does `advanced-commands.js` contain real command execution or string templates like goalie/tools.ts?
- Is `docs.js` (1,548 LOC) mostly documentation strings or does it contain functional code?
- Does `index.js` wire the module together with real imports and initialization?

**Follow-up context**:
- R41 found goalie/tools.ts was COMPLETE FACADE (45%) — 10 CRITICAL findings, GoapPlanner imported but never called
- R31 concluded ruv-swarm CLI is a "demonstration framework" — does claude-integration reinforce or correct this?

---

### Cluster B: AgentDB Benchmarks + ReasoningBank (3 files, ~2,485 LOC)

| # | File ID | File | LOC | Domain |
|---|---------|------|-----|--------|
| 5 | 12349 | `packages/agentdb/benchmarks/ruvector-benchmark.ts` | 1264 | memory-and-learning |
| 6 | 10822 | `agentic-flow/src/reasoningbank/demo-comparison.ts` | 616 | memory-and-learning |
| 7 | 304 | `simulation/scenarios/latent-space/neural-augmentation.ts` | 605 | agentdb-integration |

**Key questions**:
- ruvector-benchmark.ts: Does it run real benchmarks with actual vector operations, or is it a benchmark template with hardcoded results?
- Does it test the real hnswlib-node C++ backend (R40: HNSWIndex.ts 85%) or a mock?
- ReasoningBank demo-comparison.ts: Is ReasoningBank a genuine RL/learning system or another facade?
- Does it connect to AgentDB's actual learning backends or operate standalone?
- neural-augmentation.ts: Follows R41's latent-space cluster (81% weighted) — does neural augmentation add real ML or decorative metrics?

**Follow-up context**:
- R40 found HNSWIndex.ts is REAL (85%, wraps hnswlib-node C++) but attention-tools-handlers.ts is FACADE (40%, all Math.random)
- R20 root cause: EmbeddingService never initialized — does benchmark bypass or expose this?
- R41 found latent-space sims are "standalone research testbeds, NOT connected to production"

---

### Cluster C: Sublinear WASM Tools + Benchmarks (3 files, ~2,036 LOC)

| # | File ID | File | LOC | Domain |
|---|---------|------|-----|--------|
| 8 | 14408 | `src/mcp/tools/wasm-sublinear-complete.ts` | 710 | memory-and-learning |
| 9 | 13980 | `crates/psycho-symbolic-reasoner/benchmarks/benches/baseline_comparison.rs` | 669 | memory-and-learning |
| 10 | 14444 | `src/rustc_optimization_benchmarks.rs` | 657 | memory-and-learning |

**Key questions**:
- wasm-sublinear-complete.ts: Does it actually load and call WASM modules, or is WASM still "loaded but unused" (R39)?
- Is this a complete MCP tool with real solve() paths, or another facade like goalie/tools.ts?
- baseline_comparison.rs: Does it benchmark the real psycho-symbolic knowledge graph (R41: 78%) against alternatives?
- Are benchmark results computed or hardcoded?
- rustc_optimization_benchmarks.rs: What is this benchmarking? Rust compiler optimizations or the sublinear solver?

**Follow-up context**:
- R39 confirmed solver.ts has "WASM loaded but unused" and high-performance-solver.ts is "ORPHANED"
- R41 found psycho-symbolic-enhanced.ts is BEST knowledge graph in sublinear-time-solver (78%, real BFS, 50+ triples)
- R41 found cli/index.ts is GENUINE (88%, real SublinearSolver import, real MCP server)

---

## Expected Outcomes

- **Claude integration verdict**: Real API integration or facade — resolves whether ruv-swarm has genuine Claude connectivity
- **Benchmark quality**: Whether AgentDB benchmarks use real vector operations or hardcoded results
- **ReasoningBank**: First look at agentic-flow's RL/learning subsystem — genuine or facade?
- **WASM resolution**: Definitive answer on whether sublinear-time-solver's WASM integration works end-to-end
- **Psycho-symbolic validation**: Benchmarks confirm or undermine R41's "genuine knowledge graph" finding

## Stats Target

- ~10 file reads, ~7,250 LOC
- DEEP files: 860 -> ~870
- Expected findings: 40-70

## Cross-Session Notes

- All 3 clusters touch different packages (ruv-fann-rust, agentic-flow-rust/agentdb, sublinear-rust) — no overlap risk if parallelized
- Cluster A is entirely swarm-coordination domain; Clusters B+C are memory-and-learning — no domain synthesis conflicts
- If Cluster A confirms claude-integration is facade, combine with R31+R41 for "ruv-swarm facade pattern" meta-finding
- If Cluster C confirms WASM works, it reverses R39's conclusion and elevates sublinear-time-solver's overall score
