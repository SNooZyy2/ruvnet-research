# R45 Execution Plan: ruv-swarm npm Package Reality Check

**Date**: 2026-02-15
**Session ID**: 45
**Focus**: ruv-swarm npm package infrastructure — MCP workflows, performance benchmarks, WASM loading, neural capabilities, SQLite pooling, Docker simulation, docs generation
**Parallel with**: R46 (no file overlap — verified: R45 = ruv-fann-rust only, R46 = sublinear-rust only)

## Rationale

- R31 found the ruv-swarm CLI is a "demonstration framework" (~78% avg), and R43 found claude-integration is a "setup toolkit NOT API" (69%) — the npm package's `src/` and `examples/` have NOT been examined
- R31 analyzed the Rust MCP crate (`ruv-swarm-mcp/`) and found `lib.rs` has CRITICAL stubs (WebSocket handler disabled, core modules commented out) — the npm/JS side may be more or less functional
- 7 large files (574-991 LOC each) sit entirely NOT_TOUCHED in the swarm-coordination domain, which is at only 16% LOC coverage
- These files represent the operational layer of ruv-swarm: what users actually run when they `npm install` the package
- Multiple files hint at real infrastructure (sqlite-pool, wasm-loader) vs potential facade patterns (performance-benchmarks, neural)

## Target: 7 files, ~5,209 LOC

---

### Cluster A: Core Runtime Infrastructure (3 files, ~1,763 LOC)

| # | File ID | File | LOC | Domain |
|---|---------|------|-----|--------|
| 1 | 9673 | `ruv-swarm/npm/src/wasm-loader.js` | 602 | swarm-coordination |
| 2 | 9669 | `ruv-swarm/npm/src/sqlite-pool.js` | 587 | swarm-coordination |
| 3 | 9661 | `ruv-swarm/npm/src/neural.js` | 574 | swarm-coordination |

**Key questions**:
- `wasm-loader.js`: Does this load REAL WASM modules (the Rust crate compiles to WASM), or is it another facade like sublinear-time-solver's TWO independent WASM facades (R43: "loaded but unused" + "checked but never loaded")?
- Does the WASM loader connect to the actual `ruv-swarm-mcp` Rust crate, or load a stub/placeholder module?
- `sqlite-pool.js`: Is this a real connection pool with proper lifecycle management (acquire/release, max connections, timeout), or a thin wrapper around a single connection?
- Does sqlite-pool back the swarm's state management (agent registry, task queue) with real persistence?
- `neural.js`: Does this implement real neural network inference (matching R23's neural-network-implementation at 90-98%), or is it facade metrics like R39's emergence (51% FABRICATED with Math.random)?
- Are these three files interconnected (neural uses WASM, WASM uses sqlite), or isolated modules?

**Follow-up context**:
- R31: Rust MCP crate has `lib.rs` with CRITICAL stubs — 200 lines of McpServer commented out
- R43: Two independent WASM facades in sublinear-time-solver — pattern may repeat here
- R23: neural-network-implementation is BEST IN ECOSYSTEM (90-98%) — does the JS neural match?

---

### Cluster B: MCP + Performance (2 files, ~1,890 LOC)

| # | File ID | File | LOC | Domain |
|---|---------|------|-----|--------|
| 4 | 9587 | `ruv-swarm/npm/examples/mcp-workflows.js` | 991 | swarm-coordination |
| 5 | 9662 | `ruv-swarm/npm/src/performance-benchmarks.js` | 899 | swarm-coordination |

**Key questions**:
- `mcp-workflows.js`: At 991 LOC this is the largest file — does it demonstrate REAL MCP tool invocations (agent spawn, task create, memory store), or template responses like R41's goalie/tools.ts (COMPLETE FACADE)?
- Does it actually connect to the MCP server, or construct fake request/response pairs?
- Are the workflow examples executable, or documentation-as-code?
- `performance-benchmarks.js`: Given R43 found rustc_optimization_benchmarks.rs is the "most deceptive file in project" (15%, asymptotic mismatch deception), does this JS benchmark have similar patterns?
- Does it run real benchmarks with measurable timings, or hardcode results?
- Does it benchmark the real sqlite-pool and wasm-loader, or synthetic operations?

**Follow-up context**:
- R43: rustc_benchmarks uses asymptotic mismatch deception — claims O(√n) but implements O(n²)
- R43: ruvector-benchmark.ts (92%) IS real — so benchmarks vary wildly in quality
- R31: MCP server in Rust has disabled handlers — do JS examples reference a working server?

---

### Cluster C: Tooling + Simulation (2 files, ~1,556 LOC)

| # | File ID | File | LOC | Domain |
|---|---------|------|-----|--------|
| 6 | 9603 | `ruv-swarm/npm/scripts/generate-docs.js` | 954 | swarm-coordination |
| 7 | 9522 | `ruv-swarm/npm/docker/claude-simulator.js` | 602 | swarm-coordination |

**Key questions**:
- `generate-docs.js`: At 954 LOC, this is substantial for a docs generator — does it parse real source code ASTs, or template-generate markdown from hardcoded structures?
- Does it extract actual API signatures and types from the codebase, or fabricate documentation?
- Does the generated documentation match the actual capabilities discovered in Clusters A and B?
- `claude-simulator.js`: Does this simulate Claude API responses for testing (legitimate test infrastructure), or is it a facade that pretends to be Claude?
- Does it implement realistic response patterns (streaming, tool use, function calling), or return static strings?
- Is the simulator used by the MCP workflow examples, creating a self-contained demo that never talks to real services?
- R43 found claude-integration is a "setup toolkit NOT API" — does the simulator reinforce the "demonstration framework" pattern?

**Follow-up context**:
- R31: CLI = demonstration framework — the Docker simulator could be the proof
- R43: claude-integration generates documentation, not API calls — does generate-docs.js do the same?

---

## Expected Outcomes

- **npm reality verdict**: Whether the npm package delivers real swarm infrastructure (WASM + SQLite + neural) or is a demonstration/documentation package
- **WASM connection**: Whether the JS WASM loader connects to the real Rust crate or loads stubs
- **Benchmark honesty**: Whether performance-benchmarks.js has real metrics or deceptive patterns like R43's rustc_benchmarks
- **Demo vs production**: Whether claude-simulator.js + mcp-workflows.js form a self-contained demo loop that never touches real services
- **R31 refinement**: Sharpen the "demonstration framework" characterization with concrete evidence from the npm layer

## Stats Target

- ~7 file reads, ~5,209 LOC
- DEEP files: 879 -> ~886
- Expected findings: 35-55

## Cross-Session Notes

- **Zero overlap with R46**: R46 covers sublinear-rust (goalie, benchmarks, consciousness, integration). No shared files or packages.
- **Zero overlap with R43-R44**: R43 covered claude-integration/ and AgentDB benchmarks. R44 covered ruvector P2P, agentic-flow bridges, sublinear servers. No file overlap.
- If Cluster A finds real infrastructure, it contradicts R31's "demonstration framework" — ruv-swarm may be more capable than the CLI layer suggests
- If Cluster B confirms deceptive benchmarks, it extends R43's finding from Rust to JS
- If Cluster C confirms self-contained demo loop, it definitively establishes ruv-swarm npm as a showcase package
