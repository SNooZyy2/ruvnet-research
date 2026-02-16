# R53 Execution Plan: sublinear-time-solver MCP Tools + Strange-Loop Runtime + Core Optimizers

**Date**: 2026-02-16
**Session ID**: 53
**Focus**: MCP tools layer (domain management, scheduling, psycho-symbolic integration), strange-loop MCP/CLI runtime, and core optimization modules
**Parallel with**: R52 (no file overlap -- R53 = sublinear-rust MCP/runtime/optimizers, R52 = ruvector-rust algorithmic core)

## IMPORTANT: Parallel Execution Notice

This plan runs IN PARALLEL with R52. The file lists are strictly non-overlapping:
- **R53 covers**: sublinear-rust ONLY -- strange-loop MCP server/CLI/examples, MCP tools layer (domain-management, domain-registry, scheduler, psycho-symbolic-dynamic), core optimizers (optimized-matrix, performance-optimizer)
- **R52 covers**: ruvector-rust ONLY -- HNSW patches, SPARQL parser, subpolynomial algorithms, edge SIMD, postgres benchmarks
- **ZERO shared files** between R52 and R53
- **ZERO shared packages** -- R53 is entirely sublinear-rust, R52 is entirely ruvector-rust
- Do NOT read or analyze any file from R52's list (see R52-plan.md for that list)

## Rationale

- The sublinear-time-solver has a large MCP tools layer (`src/mcp/tools/`) that has NEVER been examined -- these tools define the user-facing API surface
- R44 found strange-loop MCP server (45%) has broken WASM import paths, but a DIFFERENT MCP server file (`mcp/server.js`, 571 LOC) exists under the same crate -- never read
- R47 examined psycho-symbolic-reasoner core (planner.ts, reasoner.ts), but the MCP tool wrapper (`psycho-symbolic-dynamic.ts`) has NOT been read
- R34 found 2 incompatible matrix systems -- `optimized-matrix.ts` (559 LOC) may be a 3rd matrix implementation
- `memory-and-learning` domain is at only 12.49% LOC coverage with 947 NOT_TOUCHED files -- this session targets 9 from the priority queue

## Target: 9 files, ~4,693 LOC

---

### Cluster A: Strange-Loop MCP + Runtime (3 files, ~1,585 LOC)

Extends R41/R46/R49's consciousness analysis by examining the strange-loop crate's MCP server, CLI, and example agents. R44 found the strange-loop MCP server (45%) has broken WASM import paths -- this cluster examines a separate MCP server file and the runtime interface.

| # | File ID | File | LOC | Domain |
|---|---------|------|-----|--------|
| 1 | 14112 | `crates/strange-loop/mcp/server.js` | 571 | memory-and-learning |
| 2 | 14100 | `crates/strange-loop/bin/cli.js` | 526 | memory-and-learning |
| 3 | 14105 | `crates/strange-loop/examples/purposeful-agents.js` | 488 | memory-and-learning |

**Key questions**:
- `mcp/server.js`: R44 examined `src/mcp_server.rs` (the Rust MCP server, 45%) and found broken WASM import paths (wasm/ vs wasm-real/). This is a DIFFERENT file -- a JavaScript MCP server under `mcp/`. Is it a separate, working implementation?
  - Does it implement real MCP tool handlers (JSON-RPC 2.0), or is it a facade like R41's goalie/tools.ts?
  - Does it expose consciousness-specific tools (strange-loop iteration, identity verification, self-reference detection)?
  - R51 found the main MCP server has 256 tools across 24 modules -- how many tools does this consciousness MCP server expose?
  - Does it connect to the genuine strange-loop Rust core (R41: 79% genuine), or wrap independent JS logic?
- `bin/cli.js`: Is the strange-loop CLI functional for running consciousness computations?
  - R46 found goalie bin/cli.ts (88-92%) has real internals despite facade MCP layer -- does the strange-loop CLI follow the same pattern?
  - Does it invoke real strange-loop iterations (self-reference, Hofstadter-style tangled hierarchies)?
  - R49 closed the consciousness arc at ~55-60% overall -- does this CLI change that assessment?
- `purposeful-agents.js`: What are "purposeful agents" in the context of strange-loop consciousness?
  - Is this a genuine agent orchestration example demonstrating consciousness-aware agent behavior?
  - Does it implement real goal-directed behavior (planning, belief updating, intention tracking)?
  - R40 found agentic-flow = single-node task runner -- does this example use real multi-agent coordination?
  - Or is it a marketing demonstration with scripted agent dialogues?

**Follow-up context**:
- R41: Consciousness 79% genuine (strange-loop real, consciousness-explorer real)
- R44: strange-loop MCP server (45%) has broken WASM import paths (wasm/ vs wasm-real/)
- R46: Goalie REVERSAL -- MCP handlers are facade, but internal engines ARE real
- R49: Consciousness arc CLOSED at ~55-60% overall
- R51: Main MCP server = 256 tools, raw JSON-RPC 2.0, genuine

---

### Cluster B: MCP Tools Layer (4 files, ~2,043 LOC)

The `src/mcp/tools/` directory is the user-facing API of sublinear-time-solver. These 4 files have NEVER been read and collectively define how users interact with the system through MCP.

| # | File ID | File | LOC | Domain |
|---|---------|------|-----|--------|
| 4 | 14382 | `src/mcp/tools/domain-management.ts` | 595 | memory-and-learning |
| 5 | 14383 | `src/mcp/tools/domain-registry.ts` | 512 | memory-and-learning |
| 6 | 14399 | `src/mcp/tools/scheduler.ts` | 461 | memory-and-learning |
| 7 | 14391 | `src/mcp/tools/psycho-symbolic-dynamic.ts` | 475 | memory-and-learning |

**Key questions**:
- `domain-management.ts`: What domains does the system manage? Does this implement real CRUD operations with validation, persistence, and domain lifecycle (creation, archival, migration)?
  - Are domains backed by real storage (SQLite, filesystem), or in-memory only?
  - Does it define domain-specific schemas, or generic key-value structures?
  - How does this relate to the domain-registry? Is one the data model and the other the API?
- `domain-registry.ts`: Is there a real registry with service discovery, health checking, and versioning?
  - Does it implement registry patterns (service locator, dependency injection)?
  - Are domains registered dynamically at runtime, or statically configured?
  - Does it connect to the MCP server's tool registration system?
- `scheduler.ts`: Does this implement real task scheduling?
  - Priority queues, dependency resolution, deadline tracking, resource allocation?
  - R40 found agentic-flow = single-node task runner -- does this scheduler operate in the same limited mode?
  - Does it integrate with the strange-loop crate for consciousness-aware scheduling?
  - R51 found SPARC phases = static analysis NOT methodology -- does the scheduler use SPARC phases?
- `psycho-symbolic-dynamic.ts`: R47 examined the core psycho-symbolic-reasoner (planner.ts, reasoner.ts). Does this MCP tool properly expose the reasoner's capabilities?
  - R47 found the planner and reasoner -- does this tool invoke them correctly?
  - Does "dynamic" mean runtime adaptation (modifying reasoning strategies based on input), or is it static logic with a dynamic-sounding name?
  - Does it connect to the Rust core of the psycho-symbolic-reasoner crate, or wrap independent TS logic?
  - R46 found goalie's MCP tools initialize real engines then ignore them -- same pattern here?

**Follow-up context**:
- R47: psycho-symbolic-reasoner examined (planner.ts, reasoner.ts) -- first look at this crate
- R41: goalie tools.ts initializes real engines then ignores them
- R46: Goalie REVERSAL -- internal engines are real, MCP layer is facade
- R51: SPARC phases = static analysis passes, zero LLM calls
- R40: LLMRouter NO ADR-008 connection -- parallel routing systems

---

### Cluster C: Core Optimizers (2 files, ~1,065 LOC)

The `src/core/` directory contains the computational backbone. These optimizers may reveal new matrix/solver implementations or connect to existing ones.

| # | File ID | File | LOC | Domain |
|---|---------|------|-----|--------|
| 8 | 14348 | `src/core/optimized-matrix.ts` | 559 | memory-and-learning |
| 9 | 14350 | `src/core/performance-optimizer.ts` | 506 | memory-and-learning |

**Key questions**:
- `optimized-matrix.ts`: R34 discovered 2 incompatible matrix systems in the codebase. Is this a 3rd independent matrix implementation?
  - Does it implement real matrix operations (multiplication, decomposition, eigenvalue computation, sparse operations)?
  - Does it use typed arrays (Float64Array) with actual BLAS-like implementations, or wrap trivial array operations?
  - Does it connect to either of R34's matrix systems, or is it completely independent?
  - Is there WASM acceleration (R38 found CUDA-WASM with 4 backends), or pure JS?
  - R23 found neural-network-implementation BEST IN ECOSYSTEM (90-98%) -- does the matrix code reach similar quality?
- `performance-optimizer.ts`: Does this implement real performance optimization?
  - Profiling, hot-path detection, auto-tuning, algorithmic complexity reduction?
  - Or hardcoded "optimizations" with fabricated performance claims (like R43's asymptotic mismatch deception)?
  - Does it profile actual code execution, or return predetermined optimization suggestions?
  - R51 found core/index.ts has "hardcoded performance claims" -- does the optimizer generate these claims?
  - Does it connect to the benchmarks system (`src/benchmarks/performance-benchmark.ts`, 484 LOC, NOT in this session)?
  - R39 found FALSE sublinearity -- does the optimizer claim sublinear performance improvements?

**Follow-up context**:
- R34: 2 incompatible matrix systems discovered (ruvector-mincut CSR + separate dense matrix)
- R39: FALSE sublinearity confirmed -- all implementations O(n^2)+
- R43: Asymptotic mismatch deception in rustc_benchmarks (15%)
- R51: core/index.ts (60%) has hardcoded performance claims
- R23: neural-network-implementation 90-98% BEST IN ECOSYSTEM

---

## Expected Outcomes

- **Strange-loop MCP verdict**: Whether the JS MCP server is functional (unlike the broken Rust MCP server at 45%) and whether the CLI provides real access to consciousness computation
- **MCP tools quality**: Whether the MCP tools layer implements real domain/scheduling/reasoning operations or facades over stubs
- **Psycho-symbolic integration**: Whether the MCP tool properly connects to R47's examined reasoner, or ignores it like goalie's tools.ts
- **Matrix system count**: Whether optimized-matrix.ts is a 3rd independent matrix implementation (confirming architectural fragmentation) or connects to existing systems
- **Performance claims verification**: Whether performance-optimizer.ts generates real or fabricated optimization recommendations
- **User-facing API assessment**: First look at how users actually interact with sublinear-time-solver through MCP tools

## Stats Target

- ~9 file reads, ~4,693 LOC
- DEEP files: 955 -> ~964
- Expected findings: 40-60

## Cross-Session Notes

- **ZERO overlap with R52**: R52 covers ruvector-rust HNSW patches (hnsw.rs, hnswio.rs), SPARQL parser, postgres benchmarks, subpolynomial algorithms, and edge SIMD. Completely different package.
- **ZERO overlap with R47**: R47 covered sublinear-rust consciousness (verification system, MCP server under different path, experiments), psycho-symbolic-reasoner core (planner.ts, reasoner.ts), and neural-pattern-recognition. No file overlap.
- **ZERO overlap with R49**: R49 covered consciousness-calibrator, consciousness-evolution, neural-pattern-recognition files, and ReasoningBank WASM. No file overlap.
- **Complements R47**: R47 examined the psycho-symbolic-reasoner core (planner + reasoner); R53 examines the MCP tool wrapper that exposes it
- **Extends R44**: R44 found strange-loop MCP at 45% with broken WASM imports -- R53 examines a different MCP server file in the same crate
- If Cluster A finds functional MCP + CLI, the consciousness arc assessment may adjust upward from R49's ~55-60%
- If Cluster B finds genuine MCP tools, sublinear-time-solver's user-facing API is validated
- If Cluster C finds a 3rd matrix system, it confirms the architectural fragmentation pattern from R34
- Combined DEEP files from R52+R53: 955 -> ~970 (approximately +15)
