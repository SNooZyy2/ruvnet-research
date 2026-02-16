# R61 Execution Plan: MCP Tool Layer + ReasonGraph + Cross-Package Integration

**Date**: 2026-02-16
**Session ID**: 61
**Focus**: Sublinear-rust MCP tools (reasoning-cache, matrix, graph), ReasonGraph subsystem (performance-optimizer, research-interface), agentic-flow AgentDB controllers (ExplainableRecall), routing investigation (TinyDancerRouter), ReasoningBank MCP legacy (tools_old.rs), ruv-swarm WASM loader and CLI output
**Parallel with**: R60 (no file overlap -- R61 = MCP tools + ReasonGraph + agentic-flow + ruv-fann-rust; R60 = WASM + core/ + crates/ + src/sublinear/ + src/convergence/)

## IMPORTANT: Parallel Execution Notice

This plan runs IN PARALLEL with R60. The file lists are strictly non-overlapping:
- **R61 covers**: sublinear-rust src/mcp/tools/ (3 TS files), sublinear-rust src/reasongraph/ (2 TS files), agentic-flow-rust controllers+routing (2 TS files), agentic-flow-rust reasoningbank MCP (1 Rust file), ruv-fann-rust ruv-swarm (2 files: JS + Rust)
- **R60 covers**: sublinear-rust WASM files (4 Rust), sublinear-rust src/core/ (3 TS), sublinear-rust src/sublinear/ (1 Rust), sublinear-rust crates/temporal-compare (1 Rust), sublinear-rust src/convergence/ (1 JS)
- **ZERO shared files** between R60 and R61
- **R61 has NO crates/wasm-solver, src/wasm*, src/core/, src/sublinear/, src/convergence/, or crates/temporal-compare files**
- **R60 has NO src/mcp/, src/reasongraph/, agentic-flow-rust, or ruv-fann-rust files**
- Do NOT read or analyze any file from R60's list (see R60-plan.md for that list)

## Rationale

- **MCP tools are the external API surface**: R51 found 256 MCP tools genuine, but claudeFlowSdkServer.ts was BIMODAL (7 tools = execSync wrappers at 0%). These 3 MCP tools in sublinear-rust (reasoning-cache, matrix, graph) expose solver functionality to MCP consumers. Are they genuine integrations or stubs?
- **ReasonGraph is completely unexplored**: performance-optimizer.ts (458 LOC) and research-interface.ts (406 LOC) form the ReasonGraph subsystem. No session has examined ReasonGraph. It may be related to ReasoningBank (R57) or may be an independent reasoning graph system
- **ExplainableRecall extends the AgentDB controller stack**: R48 found 3 disconnected AgentDB layers, R57 found a 4th (ReasoningBank). ExplainableRecall (578 LOC) is a higher-level controller that may bridge the disconnect
- **TinyDancerRouter = potential 5th routing system**: R51 found FOUR parallel routing systems (ADR-008, LLMRouter, RuvLLMOrchestrator, ProviderManager) with ZERO integration. TinyDancerRouter (534 LOC) may be a 5th, or it may implement one of the existing systems
- **tools_old.rs = ReasoningBank MCP evolution**: R57 found queries.ts genuine (85-90%) but ReasoningBank as 4th disconnected data layer. tools_old.rs (419 LOC) is the RUST MCP implementation for ReasoningBank -- does the Rust version match the TS quality?
- **ruv-swarm wasm-loader2.js extends WASM investigation**: R57 found ruv-swarm index.ts "WASM API MISMATCH = 0% WASM". wasm-loader2.js (404 LOC) is a second WASM loader -- does it fix the mismatch or show a different approach?
- **ruv-swarm-cli output.rs = CLI infrastructure**: R19 CLOSED CLI at 94.9%. output.rs (407 LOC) is the CLI output formatting -- does it follow the genuine CLI pattern?

## Target: 10 files, ~4,481 LOC

---

### Cluster A: Sublinear-Rust MCP Tool Layer (3 files, ~1,275 LOC)

The MCP tools expose solver functionality to external consumers. R51 found the main MCP server genuine (256 tools), but individual tool quality varies (claudeFlowSdkServer.ts BIMODAL). These 3 tools are in a different package (sublinear-rust), testing whether the genuine MCP pattern extends.

| # | File ID | File | LOC | Domain | Package |
|---|---------|------|-----|--------|---------|
| 1 | 14398 | `src/mcp/tools/reasoning-cache.ts` | 454 | memory-and-learning | sublinear-rust |
| 2 | 14389 | `src/mcp/tools/matrix.ts` | 419 | memory-and-learning | sublinear-rust |
| 3 | 14387 | `src/mcp/tools/graph.ts` | 402 | memory-and-learning | sublinear-rust |

**Full paths**:
1. `~/repos/sublinear-time-solver/src/mcp/tools/reasoning-cache.ts`
2. `~/repos/sublinear-time-solver/src/mcp/tools/matrix.ts`
3. `~/repos/sublinear-time-solver/src/mcp/tools/graph.ts`

**Key questions**:
- `reasoning-cache.ts`: How does the reasoning cache MCP tool work?
  - Does it implement a genuine caching layer for reasoning results (LRU, TTL, invalidation)?
  - Does it connect to ReasonGraph (Cluster B) or ReasoningBank (R57)?
  - Does it expose proper MCP tool definitions with JSON schemas?
  - R53 found performance-optimizer GENUINE (88-92%) -- does this MCP tool match that quality?
  - Does it cache actual solver outputs or fabricated data?
- `matrix.ts (MCP tool)`: Does this MCP tool expose matrix operations?
  - Does it wrap the Rust sparse matrix infrastructure via WASM/NAPI?
  - Or does it implement matrix operations in pure TS (like the pattern in src/core/)?
  - R34 found 2 incompatible matrix systems, R53 found 3rd -- does this tool unify or add another?
  - Does it define proper MCP tool input/output schemas for matrix operations?
  - How does it relate to src/core/matrix.ts (in R60)?
- `graph.ts (MCP tool)`: What graph operations does this expose?
  - Does it wrap the Rust graph infrastructure (ruvector-graph)?
  - Or does it implement its own graph algorithms?
  - Does it connect to the ReasonGraph subsystem (Cluster B)?
  - Does it support graph queries (Cypher-like, SPARQL-like, or custom)?
  - R52 found SPARQL 1.1 parser 93-95% -- does this tool use it?

**Follow-up context**:
- R51: 256 MCP tools genuine, claudeFlowSdkServer.ts BIMODAL (0% execSync, 90% WASM)
- R53: MCP tools BIMODAL (82-92% vs 18-28%)
- R58: server.ts 72-76% genuine MCP SDK with 5 tools
- R34: 2 incompatible matrix systems, R53: 3rd system
- R52: SPARQL 1.1 parser 93-95%

---

### Cluster B: ReasonGraph Subsystem (2 files, ~864 LOC)

ReasonGraph is completely unexplored. It may be a graph-based reasoning system (like knowledge graphs or inference chains) or related to ReasoningBank. These are the core files.

| # | File ID | File | LOC | Domain | Package |
|---|---------|------|-----|--------|---------|
| 4 | 14441 | `src/reasongraph/performance-optimizer.ts` | 458 | memory-and-learning | sublinear-rust |
| 5 | 14442 | `src/reasongraph/research-interface.ts` | 406 | memory-and-learning | sublinear-rust |

**Full paths**:
4. `~/repos/sublinear-time-solver/src/reasongraph/performance-optimizer.ts`
5. `~/repos/sublinear-time-solver/src/reasongraph/research-interface.ts`

**Key questions**:
- `performance-optimizer.ts`: What does ReasonGraph optimize?
  - R53 found a GENUINE performance-optimizer (88-92%) in a different subsystem -- does ReasonGraph's follow the same pattern?
  - Does it optimize graph traversal, query execution, or reasoning inference?
  - Does it use genuine optimization techniques (caching, pruning, indexing)?
  - Or does it fabricate performance metrics (like R57's performance.js at 25-30%)?
  - Does it connect to any solver algorithms (backward_push, conjugate gradient)?
- `research-interface.ts`: What does the research interface expose?
  - Is this a genuine API for interacting with the reasoning graph?
  - Does "research" mean academic research (experiment API) or code research (like this project)?
  - Does it expose graph construction, query, and visualization capabilities?
  - Does it import from other ReasonGraph files or from external packages?
  - At 406 LOC, is there enough for a real API or is it mostly type definitions?

**Follow-up context**:
- R53: performance-optimizer GENUINE (88-92%) -- different file, same name pattern
- R57: ReasoningBank = 4th disconnected data layer (uses .swarm/memory.db)
- R43: ReasoningBank core APIs genuine, demo-comparison.ts theater (35%)
- NEW: ReasonGraph has never been examined in any session

---

### Cluster C: Cross-Package Integration (3 files, ~1,531 LOC)

Three files from agentic-flow-rust that test whether the disconnected layers have hidden connections. ExplainableRecall is a higher-level AgentDB controller. TinyDancerRouter may be the 5th routing system. tools_old.rs is the Rust side of ReasoningBank MCP.

| # | File ID | File | LOC | Domain | Package |
|---|---------|------|-----|--------|---------|
| 6 | 10647 | `agentic-flow/src/agentdb/controllers/ExplainableRecall.ts` | 578 | agentic-flow | agentic-flow-rust |
| 7 | 10867 | `agentic-flow/src/routing/TinyDancerRouter.ts` | 534 | agentic-flow | agentic-flow-rust |
| 8 | 13454 | `reasoningbank/crates/reasoningbank-mcp/src/tools_old.rs` | 419 | memory-and-learning | agentic-flow-rust |

**Full paths**:
6. `~/repos/agentic-flow/agentic-flow/src/agentdb/controllers/ExplainableRecall.ts`
7. `~/repos/agentic-flow/agentic-flow/src/routing/TinyDancerRouter.ts`
8. `~/repos/agentic-flow/reasoningbank/crates/reasoningbank-mcp/src/tools_old.rs`

**Key questions**:
- `ExplainableRecall.ts`: Does this bridge AgentDB's disconnected layers?
  - R48 found 3 disconnected AgentDB layers, R57 found a 4th (ReasoningBank)
  - Does ExplainableRecall query across layers or only one?
  - Does it implement genuine explainability (attribution, provenance, confidence)?
  - Or is it a wrapper around basic vector similarity with "explainable" branding?
  - Does it use the real EmbeddingService or the broken mock one (R20 root cause)?
  - At 578 LOC, there's room for genuine explainability logic
- `TinyDancerRouter.ts`: Is this the 5th routing system?
  - R51 found FOUR parallel routing systems with ZERO integration
  - Is TinyDancerRouter independent (5th) or an implementation of one of the 4?
  - What routing strategy does it use? The name suggests lightweight/efficient
  - Does it route between models (like ADR-008), agents (like SemanticRouter), or services?
  - Does it import from any existing routing infrastructure?
  - Does it implement health checks, circuit breaking, or load balancing?
- `tools_old.rs`: How does the Rust MCP compare to TS MCP?
  - R57 found queries.ts (TS) at 85-90% with PRODUCTION-READY 7-table schema
  - Does the Rust tools_old.rs match that quality or was it deprecated for good reason?
  - Does "old" mean a previous version that was superseded, or abandoned code?
  - Does it implement the same MCP tool set as the current TS version?
  - Does it use proper Rust MCP SDK (rmcp, tower) or raw JSON handling?

**Follow-up context**:
- R48: 3 disconnected AgentDB layers
- R51: FOUR parallel routing systems (ADR-008, LLMRouter, RuvLLMOrchestrator, ProviderManager)
- R51: SemanticRouter 88-92% = intent classifier, complementary NOT 5th routing
- R57: queries.ts 85-90% PRODUCTION-READY, ReasoningBank = 4th disconnected layer
- R20: ROOT CAUSE = EmbeddingService never initialized -> broken search
- R42: RAC 92% HIGHEST QUALITY SINGLE-FILE RUST

---

### Cluster D: ruv-fann-rust Swarm Infrastructure (2 files, ~811 LOC)

Two files from the ruv-swarm subsystem in ruv-fann-rust. wasm-loader2.js extends the WASM investigation from a different package. output.rs is CLI formatting infrastructure.

| # | File ID | File | LOC | Domain | Package |
|---|---------|------|-----|--------|---------|
| 9 | 9674 | `ruv-swarm/npm/src/wasm-loader2.js` | 404 | swarm-coordination | ruv-fann-rust |
| 10 | 8902 | `ruv-swarm/crates/ruv-swarm-cli/src/output.rs` | 407 | swarm-coordination | ruv-fann-rust |

**Full paths**:
9. `~/repos/ruv-FANN/ruv-swarm/npm/src/wasm-loader2.js`
10. `~/repos/ruv-FANN/ruv-swarm/crates/ruv-swarm-cli/src/output.rs`

**Key questions**:
- `wasm-loader2.js`: Why is there a SECOND WASM loader?
  - R57 found ruv-swarm index.ts "WASM API MISMATCH = 0% WASM" (pure-JS fallback always executes)
  - The "2" suffix suggests this is a replacement/evolution -- does it fix the API mismatch?
  - Does it properly load .wasm files (fetch, instantiate, validate exports)?
  - Does it handle the WASM/JS fallback correctly (unlike the original)?
  - Does it expose the correct typed API that matches the Rust WASM exports?
  - Is this the loader that R60's WASM pipeline targets, or a separate ecosystem?
- `output.rs`: How does ruv-swarm-cli format output?
  - R19 CLOSED CLI at 94.9%, R50 found ruv-swarm Rust bimodal (memory.rs 95%, spawn.rs 8%)
  - Does output.rs implement proper terminal formatting (ANSI colors, tables, progress bars)?
  - Does it use established Rust crates (clap, indicatif, console) or custom formatting?
  - At 407 LOC, does it cover diverse output formats (JSON, table, human-readable)?
  - Does it match R19's genuine CLI quality or R50's bimodal pattern?

**Follow-up context**:
- R57: ruv-swarm index.ts WASM API MISMATCH = 0% WASM (pure-JS fallback always executes)
- R19: CLI 94.9% CLOSED
- R50: ruv-swarm Rust bimodal (memory.rs 95%, spawn.rs 8%)
- R31: CLI = demonstration framework

---

## Expected Outcomes

- **MCP tool quality across packages**: Whether sublinear-rust's MCP tools match R51's genuine pattern (256 tools) or claudeFlowSdkServer.ts BIMODAL pattern
- **ReasonGraph discovery**: First-ever examination of the ReasonGraph subsystem -- is it a 5th disconnected layer, or does it connect to ReasoningBank/AgentDB?
- **5th routing system?**: Whether TinyDancerRouter adds to the routing proliferation (FOUR systems with ZERO integration) or implements an existing one
- **AgentDB bridge**: Whether ExplainableRecall bridges the 3+ disconnected AgentDB layers or operates within a single one
- **Rust vs TS MCP quality**: tools_old.rs (Rust) vs queries.ts (TS at 85-90%) -- does the Rust quality advantage (3-4x) hold for MCP tools?
- **WASM loader fix**: Whether wasm-loader2.js fixes R57's WASM API MISMATCH or introduces new problems
- **CLI infrastructure quality**: Whether output.rs matches R19's genuine CLI quality (94.9%)
- **Cross-package integration map**: Updated understanding of how sublinear-rust, agentic-flow-rust, and ruv-fann-rust connect (or don't)

## Stats Target

- ~10 file reads, ~4,481 LOC
- DEEP files: 1,020 -> ~1,030
- Expected findings: 65-85 (10 files across 3 packages, integration-heavy)

## Cross-Session Notes

- **ZERO overlap with R60**: R60 covers WASM pipeline + core/ + crates/ + sublinear/ + convergence/. No shared files or directories.
- **Extends R51**: TinyDancerRouter extends the 4-routing-systems finding. MCP tools extend the 256-tool MCP investigation
- **Extends R48/R57**: ExplainableRecall extends the 3+1 disconnected AgentDB layers finding
- **Extends R57**: tools_old.rs extends the ReasoningBank investigation (queries.ts 85-90%)
- **Extends R57**: wasm-loader2.js extends the WASM API MISMATCH finding
- **Extends R19**: output.rs extends the CLI quality assessment (94.9%)
- **Extends R53**: MCP tools extend the BIMODAL MCP quality finding
- **NEW: ReasonGraph** is a completely unexplored subsystem
- **NEW: ExplainableRecall** is the first AgentDB controller examined at this level
- **NEW: TinyDancerRouter** is a novel routing implementation
- **NEW: wasm-loader2.js** suggests an iteration on the original WASM loader
- Combined DEEP files from R60+R61: 1,020 -> ~1,040 (approximately +20)
