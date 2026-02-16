# R57 Execution Plan: Cross-Package Integration + ReasoningBank + Swarm Infrastructure

**Date**: 2026-02-16
**Session ID**: 57
**Focus**: ruv-fann-rust ruv-swarm npm layer + swe-bench adapter, agentic-flow-rust ReasoningBank DB and optimization, agentdb simulation scenarios, sublinear-rust server infrastructure and emergence system
**Parallel with**: R56 (no file overlap -- R57 = multi-package JS/TS + ruv-fann Rust; R56 = sublinear-rust Rust ONLY)

## IMPORTANT: Parallel Execution Notice

This plan runs IN PARALLEL with R56. The file lists are strictly non-overlapping:
- **R57 covers**: ruv-fann-rust (4 files: ruv-swarm npm + swe-bench-adapter), agentic-flow-rust (2 files: ReasoningBank DB + optimization), agentdb (1 file: neural-augmentation simulation), sublinear-rust (3 files: server JS + emergence TS ONLY)
- **R56 covers**: sublinear-rust ONLY -- solver algorithms (Rust), neural benchmarks (Rust), temporal_nexus (Rust), core types (Rust)
- **ZERO shared files** between R56 and R57
- **R57 has NO sublinear-rust Rust files**; R56 has NO ruv-fann-rust, agentic-flow-rust, or agentdb files
- Do NOT read or analyze any file from R56's list (see R56-plan.md for that list)

## Rationale

- **ruv-fann-rust is severely underexplored**: 1,283 NOT_TOUCHED files, 0% LOC covered. ruv-swarm npm layer (`index.ts` 457 LOC, `performance.js` 458 LOC) is the JS entry point for the entire ruv-swarm ecosystem -- never read
- **swe-bench-adapter** is an entirely unexplored subsystem -- `stream_parser.rs` (439 LOC) and `benchmarking.rs` (430 LOC) may reveal whether ruv-swarm has genuine SWE-bench integration or is another facade
- **ReasoningBank DB layer** is critical infrastructure: R43 found demo-comparison.ts is "scripted marketing theater" (35%), but the actual DB queries (`queries.ts` 441 LOC) and optimization examples (`reasoningbank-optimize.js` 483 LOC) have never been read
- **agentdb neural-augmentation.js** (472 LOC) is in `dist/simulation/scenarios/latent-space/` -- a completely unexplored agentdb simulation system
- **sublinear-rust server** (`streaming.js` 520 LOC, `session-manager.js` 440 LOC) is the HTTP/streaming layer that serves the solver -- never examined
- **persistent-learning-system.ts** (452 LOC) in `emergence/` extends R39's emergence analysis (51% FABRICATED) -- does this file have genuine persistence?

## Target: 10 files, ~4,592 LOC

---

### Cluster A: ruv-fann-rust Infrastructure (4 files, ~1,784 LOC)

ruv-fann-rust (package 9) has 1,283 NOT_TOUCHED files. These 4 files span two critical subsystems: the ruv-swarm npm entry point (how JS consumers interact with swarm) and the swe-bench adapter (potential real-world benchmark integration).

| # | File ID | File | LOC | Domain | Package |
|---|---------|------|-----|--------|---------|
| 1 | 9632 | `ruv-swarm/npm/src/index.ts` | 457 | swarm-coordination | ruv-fann-rust |
| 2 | 9663 | `ruv-swarm/npm/src/performance.js` | 458 | swarm-coordination | ruv-fann-rust |
| 3 | 9089 | `ruv-swarm/crates/swe-bench-adapter/src/stream_parser.rs` | 439 | swarm-coordination | ruv-fann-rust |
| 4 | 9084 | `ruv-swarm/crates/swe-bench-adapter/src/benchmarking.rs` | 430 | swarm-coordination | ruv-fann-rust |

**Full paths**:
1. `~/repos/ruv-FANN/ruv-swarm/npm/src/index.ts`
2. `~/repos/ruv-FANN/ruv-swarm/npm/src/performance.js`
3. `~/repos/ruv-FANN/ruv-swarm/crates/swe-bench-adapter/src/stream_parser.rs`
4. `~/repos/ruv-FANN/ruv-swarm/crates/swe-bench-adapter/src/benchmarking.rs`

**Key questions**:
- `index.ts`: What does the ruv-swarm npm entry point expose?
  - Does it export a real client SDK for spawning/managing swarms?
  - Does it wrap the Rust binary via NAPI or call it as a subprocess?
  - R50 found ruv-swarm Rust has same split (memory.rs 95%, spawn.rs 8%) -- does the npm layer expose the genuine parts?
  - Does it import from persistence.js (R55 confirmed 95-98% production SQLite)?
  - Is this a functional npm package or a placeholder that re-exports stubs?
- `performance.js`: What performance does this measure/provide?
  - Is this genuine performance monitoring or another R53-style theatrical scheduler?
  - Does it measure real swarm metrics (agent throughput, task completion, memory usage)?
  - R55 found performance_monitor.rs (88-92%) GENUINE with Instant::now() -- does the JS version match?
  - Does it use process.hrtime() or Date.now() for timing?
- `stream_parser.rs`: What streams does the swe-bench adapter parse?
  - SWE-bench outputs JSON-lines format with patch diffs -- does this parse that?
  - Does it handle real SWE-bench task instances (repository, instance_id, patch)?
  - Is this a genuine integration with the SWE-bench benchmark suite or a stub?
  - At 439 LOC in Rust, there's enough room for real stream processing
- `benchmarking.rs`: What SWE-bench benchmarks does this run?
  - Does it actually invoke SWE-bench tasks (clone repos, apply patches, run tests)?
  - Does it compute pass@k metrics?
  - R43 found benchmark deception is systemic -- does this follow the pattern or is it genuine?
  - Does it connect to stream_parser.rs for result parsing?

**Follow-up context**:
- R50: ruv-swarm Rust split (memory.rs 95%, spawn.rs 8%), Goalie genuine, AgentDB RESCUED
- R55: persistence.js 95-98% PRODUCTION SQLite, performance_monitor.rs 88-92% GENUINE
- R53: scheduler.ts 18-22% THEATRICAL -- JS performance files need scrutiny

---

### Cluster B: ReasoningBank + AgentDB Simulation (3 files, ~1,396 LOC)

ReasoningBank spans agentic-flow-rust and agentdb. R43 found demo-comparison.ts is "scripted marketing theater" (35%), but the actual DB persistence layer and optimization examples have never been read. Plus agentdb's simulation scenarios represent an entirely unexplored subsystem.

| # | File ID | File | LOC | Domain | Package |
|---|---------|------|-----|--------|---------|
| 5 | 10819 | `agentic-flow/src/reasoningbank/db/queries.ts` | 441 | memory-and-learning | agentic-flow-rust |
| 6 | 12050 | `examples/reasoningbank-optimize.js` | 483 | memory-and-learning | agentic-flow-rust |
| 7 | 64 | `dist/simulation/scenarios/latent-space/neural-augmentation.js` | 472 | memory-and-learning | agentdb |

**Full paths**:
5. `~/repos/agentic-flow/agentic-flow/src/reasoningbank/db/queries.ts`
6. `~/repos/agentic-flow/examples/reasoningbank-optimize.js`
7. `~/node_modules/agentdb/dist/simulation/scenarios/latent-space/neural-augmentation.js`

**Key questions**:
- `queries.ts`: What SQL/queries does ReasoningBank's DB layer execute?
  - R43 found demo-comparison.ts is scripted theater -- but DB queries represent the actual data layer
  - Does it use proper parameterized queries or string concatenation (SQL injection risk)?
  - What tables does it access? Episodes, trajectories, skills, patterns?
  - Does it implement the reflexion retrieve/store pattern that AgentDB exposes via MCP?
  - R20 ROOT CAUSE was broken EmbeddingService -- does queries.ts depend on embeddings for similarity search?
- `reasoningbank-optimize.js`: What optimization does this demonstrate?
  - Is this a genuine example showing how to optimize ReasoningBank performance?
  - Does it demonstrate real optimization techniques (index creation, query tuning, batch operations)?
  - Or is it another demo-comparison-style marketing artifact?
  - Does it use agentdb's HNSW search or fall back to brute-force?
- `neural-augmentation.js`: What is the latent-space neural augmentation simulation?
  - This is in agentdb's `dist/simulation/scenarios/` -- an entirely unexplored subsystem
  - Does it simulate neural network augmentation of agent behavior?
  - Is it a genuine simulation with real neural computations or a visualization/demo?
  - R40 found JS neural models: inference works, training facade -- does this follow the same pattern?
  - Does it use agentdb's vector search infrastructure?

**Follow-up context**:
- R43: demo-comparison.ts 35% scripted marketing theater -- ReasoningBank demos need scrutiny
- R20: AgentDB search broken (EmbeddingService never initialized) -- DB queries may reveal the expected flow
- R40: JS neural models inference works, training facade
- R51: AgentDB has 3+ disconnected layers -- which layer do these files target?

---

### Cluster C: Server Infrastructure + Emergence (3 files, ~1,412 LOC)

sublinear-time-solver has an unexplored `server/` directory (HTTP serving layer) and `emergence/` directory. R39 found emergence 51% FABRICATED -- does persistent-learning-system.ts in emergence show genuine persistence or continue the pattern?

| # | File ID | File | LOC | Domain | Package |
|---|---------|------|-----|--------|---------|
| 8 | 14299 | `server/streaming.js` | 520 | memory-and-learning | sublinear-rust |
| 9 | 14297 | `server/session-manager.js` | 440 | memory-and-learning | sublinear-rust |
| 10 | 14362 | `src/emergence/persistent-learning-system.ts` | 452 | memory-and-learning | sublinear-rust |

**Full paths**:
8. `~/repos/sublinear-time-solver/server/streaming.js`
9. `~/repos/sublinear-time-solver/server/session-manager.js`
10. `~/repos/sublinear-time-solver/src/emergence/persistent-learning-system.ts`

**Key questions**:
- `streaming.js`: What does the server stream?
  - Does it implement SSE (Server-Sent Events), WebSocket, or HTTP chunked streaming?
  - Does it stream solver computation results in real-time?
  - R51 found http-streaming-updated.ts is COMPLETELY MISLABELED (zero HTTP, stdio-only, 3rd mislabeled file) -- does this streaming.js actually implement HTTP streaming?
  - Does it use Express, Fastify, or raw Node.js HTTP?
  - At 520 LOC, there's room for a real streaming server -- but filename mislabeling is a known pattern
- `session-manager.js`: How does the server manage sessions?
  - In-memory or persistent (SQLite, filesystem)?
  - R55 found persistence.js 95-98% PRODUCTION SQLite -- does session-manager.js use similar infrastructure?
  - Does it manage solver computation sessions or user authentication sessions?
  - Does it implement TTL, cleanup, and concurrent session limits?
  - R53 found domain-management.ts 82% but IN-MEMORY ONLY -- does session-manager follow this pattern?
- `persistent-learning-system.ts`: Does emergence actually persist learning?
  - R39 found emergence 51% FABRICATED -- persistent-learning-system.ts is a direct test
  - "Persistent" in the filename is a strong claim -- does it use SQLite, filesystem, or is persistence faked?
  - Does it implement genuine online learning (gradient updates, model checkpointing)?
  - Does it connect to ReasoningBank's reflexion pattern or is it standalone?
  - R53 found psycho-symbolic-dynamic.ts 28% with console.log stubs -- does this emergence file show the same quality?

**Follow-up context**:
- R39: Emergence 51% FABRICATED -- persistence claim needs verification
- R51: http-streaming-updated.ts COMPLETE MISLABELING (zero HTTP) -- streaming.js filename may be misleading
- R55: persistence.js 95-98% PRODUCTION SQLite -- real persistence infrastructure exists in the ecosystem
- R53: domain-management.ts 82% but IN-MEMORY ONLY -- server-side persistence patterns vary widely

---

## Expected Outcomes

- **ruv-swarm npm truth**: Whether the JS entry point exposes genuine swarm functionality or re-exports stubs
- **SWE-bench reality**: Whether stream_parser.rs and benchmarking.rs genuinely integrate with SWE-bench or are another facade
- **ReasoningBank DB ground truth**: Whether queries.ts shows genuine data access patterns (vs demo-comparison.ts theater)
- **AgentDB simulation discovery**: What the unexplored simulation/scenarios subsystem contains
- **Server layer assessment**: Whether streaming.js actually implements HTTP streaming (unlike R51's mislabeled file)
- **Emergence persistence verdict**: Whether persistent-learning-system.ts has genuine persistence (contradicting R39's 51% FABRICATED) or confirms the facade pattern
- **Cross-package integration**: Whether ReasoningBank queries.ts connects to AgentDB's vector search or bypasses it

## Stats Target

- ~10 file reads, ~4,592 LOC
- DEEP files: 990 -> ~1,000
- Expected findings: 60-90 (10 files across 4 packages, diverse subsystems)

## Cross-Session Notes

- **ZERO overlap with R56**: R56 covers sublinear-rust Rust files only (solver, neural benchmarks, temporal). No shared files.
- **Extends R50**: ruv-swarm npm layer extends R50's ruv-swarm assessment (memory.rs 95%, spawn.rs 8%)
- **Extends R43**: ReasoningBank DB queries extend R43's demo-comparison.ts (35%) theater finding
- **Extends R39**: persistent-learning-system.ts directly tests R39's emergence 51% FABRICATED finding
- **Extends R51**: streaming.js tests whether R51's HTTP mislabeling is isolated or systemic
- **Extends R55**: performance.js extends R55's performance_monitor.rs (88-92%) comparison between Rust and JS quality
- **NEW: swe-bench-adapter** is a completely unexplored subsystem -- may reveal real benchmark integration
- **NEW: agentdb simulation/scenarios** is an entirely unexplored subsystem -- first look at agentdb's simulation layer
- Combined DEEP files from R56+R57: 990 -> ~1,010 (approximately +20)
