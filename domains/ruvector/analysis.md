# Ruvector Domain Analysis

> **Priority**: HIGH | **Coverage**: 14.5% by count (446 DEEP / 3,085 non-excluded files) | **Status**: In Progress
> **Last updated**: 2026-03-03 (sessions through R141)

## Document Index

This analysis has been split into sub-files for manageability. **Agents: follow links below to the relevant section.**

| Section | File | Lines | Description |
|---------|------|-------|-------------|
| 1. Current State | **this file** | — | Top-level verdicts, stats, key conclusions |
| 2. File Registry | [file-registry.md](file-registry.md) | ~458 | All deep-read file tables grouped by subsystem |
| 3. Findings Registry | [findings.md](findings.md) | ~325 | CRITICAL (C1-C112), HIGH (H1-H267), MEDIUM findings |
| 4. Positives Registry | [positives.md](positives.md) | ~135 | Confirmed good patterns and genuine code |
| 5a. Subsystems (Core) | [subsystems-core.md](subsystems-core.md) | ~262 | HNSW, embeddings, attention, postgres, ruvllm, temporal-tensor, ruQu, prime-radiant basics, graph DB, SONA, subpoly, edge-net, dev methodology, ruvector-core advanced |
| 5b. Subsystems (Extended) | [subsystems-extended.md](subsystems-extended.md) | ~336 | MinCut, LLM extensions, hyperbolic, GNN bindings, AIDefence, CUDA-WASM, ruvllm context/serving/LoRA, cohomology, training/transport, npm umbrella, RVF format |
| 6. Cross-Domain | [cross-domain.md](cross-domain.md) | ~41 | Dependencies, findings distribution, package overview |
| 7. Knowledge Gaps | [gaps.md](gaps.md) | ~50 | Crate-level coverage, largest remaining gaps |
| 8. Session Log | [session-log.md](session-log.md) | ~130 | All session entries (Initial through R141) |

## Agent Instructions

When updating this domain analysis:
- **Section 1** (below): Rewrite in-place in THIS file
- **Section 2**: Edit [file-registry.md](file-registry.md) — add rows, never duplicate
- **Section 3**: Edit [findings.md](findings.md) — add with next sequential ID, never re-number
- **Section 4**: Edit [positives.md](positives.md) — append new, never re-list
- **Section 5**: Edit the appropriate subsystems file — update existing topics or create new ones
- **Section 8**: Append to [session-log.md](session-log.md)
- Follow ADR-040 canonical structure and ADR-041 in-place update protocol
- NEVER create chronological session blocks outside session-log.md
- NEVER re-list all findings/positives at each update

## 1. Current State Summary

The ruvector domain spans 6,066 files (5,577 non-excluded) across 7 packages totaling 2,525,973 LOC. Research through R141 has produced ~3,785 findings (~280 CRITICAL, ~825 HIGH, ~962 MEDIUM, ~1,730 INFO), mapped 870+ cross-file dependencies, and deep-read 446 files (14.5% of non-excluded). The codebase represents 81 days of human-AI co-development at 10.3 commits/day.

**Top-level verdicts:**

- **Hash-based embeddings remain systemic — now confirmed at CLI layer (C104) and MCP layer (C105).** Rust crates default to hash. The ONNX path exists (R117) but intelligence-engine.ts sync `embed()` falls through to hash (C78). The ruvector CLI's Intelligence.embed() uses the engine only if already initialized — most hook invocations pass skipEngine:true, yielding 64-dim hash. The MCP server (mcp-server.js) is the 9th confirmed hash instance. Hash dominance is a wiring failure, not an absence of real embeddings.
- **Rust compilation audit (R141): 100/115 crates pass cargo check (87%), 42 crates have 3,984 passing tests.** The 42 genuinely tested crates confirm the bottom-of-stack algorithmic quality: ruvector-gnn (198 tests), temporal-tensor (269 tests), nervous-system (359 tests), math (148 tests). However: ruvllm (120,345 LOC — the LARGEST crate) fails cargo check entirely (C111), sona (10,582 LOC) fails check (C112), and ruvector-cli + ruvllm-cli (the entry points) both fail. 6 additional crates (mincut 42K LOC, prime-radiant 52K LOC, ruvector-graph 17K LOC) pass check but fail test binary compilation (H265-H267). The many algorithmic bugs documented for prime-radiant (C36-C49) and mincut (C68-C74) have no running test coverage.
- **V3 intelligence.ts is a CRITICAL FACADE (R140).** intelligence.ts claims O(log n) HNSW search in its header but implements O(n) brute-force cosine similarity (C from R140 = finding ID per memory). LocalSonaCoordinator implements LoRA/EWC config fields (loraLearningRate, loraRank, ewcLambda) that are stored but NEVER used. compactPatterns() runs O(n²) all-pairs cosine comparison with maxPatterns=5000 (12.5M operations). The file has 14+ consumers but zero @ruvector/* imports — the V3 intelligence layer is completely disconnected from the genuine native HNSW in ruvector-core and hnsw_router.rs.
- **Two independent SONA subsystems exist (R140).** sona-optimizer.ts (842 LOC) is a genuine agent-routing optimizer using Bayesian confidence updates with temporal decay, connected to hooks-tools.ts and the real hooks pipeline. It has zero connection to hnsw_router.rs, hnsw-index.ts, or sona-tools.ts. sona-tools.ts (the "150x-12,500x" file) is a fabricated speedup using `searchLatency * 1000`. These share only the name "SONA" — they implement completely different systems.
- **Best code:** temporal-tensor (95%, production-ready), ruQu QEC (89%), ruvllm kernels (90%, NEON SIMD), cognitum-gate-kernel (93%), postgres SIMD (95-98%), ruqu-core (noise 96-98%, mitigation 95-98%, transpiler 95-98%), ruvllm serving/ (90% MODULE COMPLETE).
- **Worst code:** MinCut algorithm/mod.rs (35-45% COMPLEXITY FRAUD), ruvector-graph distributed rpc.rs (15-20% stubs), subpoly_decoder.rs (35-40% 3rd FALSE subpolynomial), MinCut linear attention PLACEHOLDER (15-20%), HNSW connect_node_to_neighbors() EMPTY. Three broken Laplacian implementations are systemic (C32, C42/C43, C44). intelligence.ts O(n) brute-force under HNSW branding.
- **Integration testing is pervasively mock-only (R139).** Both "integration" test files (e2e_integration_test.rs 1,535 LOC, ruvllm_integration_tests.rs 1,393 LOC) totaling 2,928 LOC contain zero cross-crate imports and use inline reimplementations with APIs that diverge from production (C109). Individual algorithms ARE genuinely tested (softmax, sampling, KV cache, speculation trees), but no cross-crate integration path is validated. The distributed Raft test cluster (docker-compose.yml) runs shell-script nodes with netcat returning "200 OK" — never invoking any Rust binary (C110). The testing picture: unit-level algorithms well-verified, integration-level paths completely untested.
- **Barrel export leaks dead code (R139).** `npm/packages/ruvector/src/core/index.ts` (57 LOC) re-exports ALL 23 modules (GNN, SONA, ONNX×2, Router, Graph, Cluster, AST, Neural×3, RVF, etc.) with no selective gating. sona-wrapper (62-68% dead per R137) and other broken modules leak to consumers via `@ruvector/core` (H262). The NAPI CI infrastructure is surprisingly production-grade: build-native.yml (5 platforms) and sona-napi.yml (7 platforms + universal macOS binary + post-publish smoke test) are genuine cross-compilation pipelines, though both carry the continue-on-error:true pattern on tests/publish (H263, H264).
- **ruvector npm CLI (7,357 LOC, R135):** Largest single CLI file. 14 top-level commands, ~70% genuine. Uses VectorDB (working), NOT RuVectorBackend (broken). ONNX embed commands (text/adaptive/benchmark/optimized/neural) are real. 4 facade commands (graph/router/server/cluster = "Coming Soon"). ~4,000 lines of hooks/intelligence system. Hash embedding confirmed at this layer (C104, extends C1).
- **ruvector MCP server (3,007 LOC, R135):** 55 MCP tools in a single 3,060-line switch statement — largest single MCP server in the ecosystem. Separate from agentdb-mcp-server.ts. Does NOT initialize EmbeddingService (R20 pattern). Heterogeneous delegation: 14 tools via execSync("npx ruvector hooks ...") with 2-5s cold-start, 11 tools via "npx agentic-flow@alpha". Query sanitization destroys SQL/Cypher/SPARQL (C106).
- **rvlite CLI (1,686 LOC, R135):** INDEPENDENT vector DB with zero ruvector imports or shared code. O(n) brute-force flat search (no HNSW index). Genuine WASM integration (SONA + Attention modules). Hyperbolic geometry (Poincare + Lorentz) mathematically correct. Advertises SQL/Cypher/SPARQL but provides none in CLI (deferred to SDK). HuggingFace model download real.
- **ruvllm CLI (1,005 LOC, R135):** Native-binary dependent — without .node binary, query/generate/route/embed return hardcoded or hash values (C107). Training is SIMULATED (C108). SIMD benchmark is genuine. Zero ruvector RAG integration (no HNSW, no @ruvector imports).
- **RVF file format subsystem characterized (R121+R123):** 6 files deep-read, all 88-93% GENUINE. Custom binary vector DB format with SHAKE-256 witness chains, 64-byte aligned RVFS segments, XXH3-128 content hash, cluster-level COW with cryptographic tamper-evidence. NAPI bridge exposes 22 methods to Node.js. **CRITICAL**: NAPI `verify_witness()` bypasses real `verify_witness_chain()`.
- **Three distinct HNSW implementations:** ruvector-core (hnsw_rs wrapper, 98%), hyperbolic-hnsw (native Poincare, 88-93% CRATE COMPLETE), micro-hnsw-wasm (novel no_std, 60-70%).
- **TWO independent SIMD codebases** (ruvector-core for distance, edge-net for inference), **TWO parallel GNN ecosystems** (ruvector-gnn native vs postgres/gnn SQL-side), **DUAL query languages** (Cypher parser-only vs SPARQL full parser+executor).
- **MinCut-Gated-Transformer is MOST NOVEL crate** (~84% weighted avg, 42 DEEP files) with genuine energy-based gating and SIMD kernels, but dynamic-update claims DISQUALIFIED (R111).
- **Prime-radiant modules COMPLETE:** coherence (5/5, ~89%), governance (5/5, ~88%), storage (4/4, ~86%), cohomology (9/9, ~83%), execution triad (~89%), hyperbolic (5/5, ~81%). Sheaf Laplacian eigensolver bugs systemic across 3 files.
- **ruvector PostgreSQL setup command (R138):** `claude-flow ruvector setup` (setup.ts, 784 LOC, ~85-90%) is a pure scaffolding command generating docker-compose.yml + 476-line init-db.sql + README. The SQL schema is production-quality (8 tables, 6 HNSW indices, 7 stored functions) but the command does NOT install native ruvector, does NOT configure the backend factory (factory.ts 5-tier fallback), and does NOT switch the default backend from sql.js to PostgreSQL. Even after successful setup, claude-flow memory continues using in-process sql.js unless something else intervenes. Completely independent from memory-initializer.ts.
- **AI co-authored explicitly** — commits credit "Claude Opus 4.5/4.6". Scope (GNN, quantum, FPGA, distributed consensus, graph DB, 39 attention types, postgres ext) would take 2-3 years for an experienced team.
