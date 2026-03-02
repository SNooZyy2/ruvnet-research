# AgentDB Integration Domain Analysis

> **Priority**: MEDIUM | **Coverage**: 57.51% LOC (149 DEEP / 516 total) | **Status**: In Progress
> **Last updated**: 2026-03-02 (Session R139)

## Document Index

This analysis has been split into sub-files for manageability. **Agents: follow links below to the relevant section.**

| Section | File | Lines | Description |
|---------|------|-------|-------------|
| 1. Current State | **this file** | — | Top-level verdicts, stats, key conclusions |
| 2. File Registry | [file-registry.md](file-registry.md) | ~500 | All deep-read file tables grouped by subsystem |
| 3. Findings Registry | [findings.md](findings.md) | ~440 | CRITICAL (C1-C58), HIGH (H1-H116), MEDIUM findings |
| 4. Positives Registry | [positives.md](positives.md) | ~60 | Confirmed good patterns and genuine code |
| 5a. Subsystems (Core) | [subsystems-core.md](subsystems-core.md) | ~280 | Architecture, search, security, attention, quality spectrum, simulations, LLM routing, CRDT, CLI ops |
| 5b. Subsystems (Extended) | [subsystems-extended.md](subsystems-extended.md) | ~350 | Native/patched architecture, infrastructure, RuVectorBackend, AIDefence, prime-radiant, RVF backend, V3 memory |
| 6. Cross-Domain | [cross-domain.md](cross-domain.md) | ~15 | Dependencies, findings distribution, package overview |
| 7. Knowledge Gaps | [gaps.md](gaps.md) | ~10 | Remaining untouched areas |
| 8. Session Log | [session-log.md](session-log.md) | ~100 | All session entries (R8 through R138) |

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

AgentDB is a 514-file vector database with agent learning capabilities. Despite claude-flow listing it as an optional dependency, **none of its 23 controllers are called** — 140K+ LOC of genuinely sophisticated code sits unused. The codebase is **~85% authentic** with production-grade search algorithms, real security implementation, and working neural attention mechanisms. Post-R115 category breakdown across 940+ findings: ARCHITECTURE (structural issues dominate); INTEGRATION (15th disconnected subsystem, 12th parallel routing system); FACADE; BUG; SECURITY; PERFORMANCE; QUALITY; GENUINE; ALGORITHM; TESTING; DOCUMENTATION; INCOMPLETE.

**R138 V3 MCP Tool Chain verdict (3 files, ~2,371 LOC, 82-92%):** The R138 MCP analysis reveals a **4-layer integration gap** that creates a complete disconnect from MCP entry point through to backend. (1) `memory-initializer` (R136): AgentDBAdapter has zero connection to real AgentDB. (2) `v3/mcp/server.ts` (R138): MCP server bootstrap does not call memory-initializer, AgentDB, or EmbeddingService — boots entirely standalone. (3) `v3/mcp/tools/index.ts` (R138): memory-tools.ts claims ADR-006 (AgentDB integration) but has NO agentdb import. (4) MCP tool handlers (R138): SONA tools fabricate metrics (`estimatedBruteForce = searchLatency * 1000`, producing ~1000x "speedup"), SONAState uses in-memory Maps with zero persistence and zero AgentDB integration. The path `user -> MCP server -> tool handler -> no AgentDB anywhere` is now fully traced. This EXTENDS C57 (the "150x-12,500x" marketing claim) from the integration file level (R137) to the active TOOL HANDLER level — fabricated speedup is not only in dead code but also in the live MCP tool surface. The library MCP server (`v3/@claude-flow/mcp/src/server.ts`, 1,134 LOC, 88-92%) is the most sophisticated MCP implementation but has ZERO awareness of AgentDB across 14 methods and 9 sub-registries — it is a pure protocol shell.

**R136 V3 @claude-flow/memory verdict (7 files, ~8,967 LOC, 68-82%):** The V3 "AgentDB" memory layer is a **misnomer**. `AgentDBAdapter` (the default backend via `UnifiedMemoryService`) has ZERO connection to any AgentDB package — it is a plain `Map<string, MemoryEntry>` in-memory store with pure JS HNSW. The REAL AgentDB integration exists in `controller-registry.ts` (~19 controllers delegate to AgentDB via dynamic import) and `memory-bridge.ts` (intended to route through ControllerRegistry to AgentDB), but memory-bridge.ts is **NOT compiled into the npm dist** — dead at runtime. The default V3 path completely bypasses real AgentDB. This is the **15th disconnected persistence/memory layer** in the project. R20 root cause NOT fixed: embeddingGenerator remains optional in AgentDBAdapter with 2/4 factory functions omitting it. The cache manager (O(1) LRU, TTL, tiered L1/L2) and controller-registry's level-ordered init (Promise.allSettled) are production-quality bright spots in an otherwise broken integration.

**Top-level verdicts:**

- **Best-in-ecosystem search implementation**: HybridSearch (BM25 + HNSW + fusion strategies) is production-ready and surpasses all other ruvnet search code.
- **Production-grade quantization**: K-means++ PQ with 8/4-bit scalar quantization rivals standalone vector databases.
- **Genuine neural attention**: MultiHead and CrossAttention controllers implement real transformer-style attention from scratch (inference-only, random weights).
- **AttentionService.ts "39 mechanisms in SQL" claim is zero-backed**: The `db` parameter is dead code — zero SQL operations execute. 4 JS fallback math implementations are genuine (FlashAttention tiled online-softmax, HyperbolicAttention Poincaré ball, GraphRoPE, MoEAttention), but WASM/NAPI backends are never compiled. All mechanisms default to `enabled:false`. (R91)
- **Solid security model**: Argon2id hashing, SQL injection whitelists, JWT tokens, brute-force protection — comprehensive and correct.
- **Complete facade in MCP tools layer**: Goalie subsystem (856 LOC) imports GoapPlanner and reasoning engines but calls NONE of them. R138 extends this: 14 SONA MCP tools are full facades at handler level (LoRA handlers are no-ops, output = input).
- **Systemic embedding degradation**: Hash-based embedding fallback silently breaks all semantic search features.
- **Critical bugs in core controllers**: LearningSystem RL is cosmetic (9 algorithms → 1 implementation), CausalMemoryGraph statistics are mathematically wrong.
- **Five parallel AgentDB systems (R136-R138 update)**: Native standalone MCP server (R137: EmbeddingService confirmed initialized, Pipeline 1), agentic-flow wrapper, claude-flow patched bridge, V3 @claude-flow/memory, and V3 MCP tool layer — only native works correctly. V3 MCP server (R138) boots with zero AgentDB init. sona-agentdb-integration.ts is a 6th aspirational system that is fully dead code (C56/C57), and its fabricated speedup metric is now confirmed to also exist in the live MCP tool handlers (C58).
- **R20 root cause clarification (R88)**: RuVectorBackend accepts correct Float32Array vectors and performs genuine HNSW search. The R20 failure is upstream — EmbeddingService never initialized in claude-flow bridge means hash-based garbage vectors are fed in. The backend itself works correctly; it is the INPUT that is wrong.
- **R20 root cause NOT fixed in V3 (R136)**: AgentDBAdapter's embeddingGenerator is optional, and 2 of 4 factory functions in index.ts omit it entirely. The structural embedding gap from R20 persists in the V3 rewrite.
- **Two competing backend factories (R139)**: The agentdb root `src/backends/factory.ts` (ID 333, 235 LOC) has a 2-tier fallback (ruvector > hnswlib) with clean dynamic import detection. The packages/agentdb `factory.ts` (ID 12809) has a 5-tier fallback (+RVF+sql.js). Both export `createBackend()` with the same signature. No import resolution analysis determines which one claude-flow actually loads at runtime (H118).
- **RuVectorBackend.ts revised down to 72-78% (R91)**: `updateAdaptiveParams()` only adjusts `efSearch` (not M/efConstruction despite incrementing `indexRebuildCount`). `insertBatchParallel()` creates a local semaphore bypassing the instance-level one. mmap wired in constructor but not used in hot paths. L2 similarity formula (Math.exp(-distance)) is uncalibrated.
- **AIDefenceGuard is STANDALONE (R92)**: The `aidefence` npm package (^2.1.1) is listed in ruvbot's package.json but never imported anywhere in the codebase. AIDefenceGuard.ts is entirely self-contained with 28 hand-crafted regex patterns for injection/jailbreak/PII detection. It is NOT wired into AgentDB at runtime. The integration scenario (`aidefence-integration.ts`) is a simulation-only scaffold — hardcoded threat data, commented-out causal links, and `enablePolicyVerification` with zero implementation behind it.
- **MemoryController.ts revised down to 72% (R96)**: The controller is a pure in-memory Map store — NOT a SQL/persistent CRUD controller as the file registry previously indicated (95%). It is the **10th disconnected persistence layer** in the project. No EmbeddingService usage: callers must supply pre-computed embeddings, structurally embedding the R20 root cause. VectorBackend is OPTIONAL (defaults null); when absent, search falls back to O(n) pure-JS cosine similarity. delete() has a critical bug: removes from Map but NOT from VectorBackend — deleted memories resurface in ANN results. CrossAttentionController is initialized and populated but NEVER consulted during search(). THREE attention controllers initialized unconditionally regardless of `enableAttention` config. Attention score combination hardcoded at `0.5*base + 0.5*(attention/2)` with unexplained `/2`.
- **Prime-radiant storage module COMPLETE (4/4 DEEP, ~85%, R107-R108)**: 4 backend implementations (postgres, file, memory, mod.rs hub) all implement GraphStorage + GovernanceStorage traits. CRITICAL WAL commit bug in file.rs (never sets committed=true). memory.rs witness mapping bug (never writes witnesses_by_action). HybridStorage = FileStorage only despite declaring both backends. postgres backend has race condition in store_witness() and O(n) full-scan find_similar() with no HNSW.
- **CrossAttentionController.ts confirmed DEAD in production (R114)**: computeCrossAttention() and computeMultiContextAttention() have zero callers in production. VectorBackend insert is write-only (never queried). Revised down from 98% to 62-68%. Math is sound; integration is dead. Confirms R96 H60 finding.
- **DB enum normalization (ADR-v4-009)**: All 773 findings are now queryable by canonical category for the first time, enabling cross-domain analysis by category type.

**R118 AgentDB ↔ ruvector bridge verdict:** The relationship is a **5-plane bidirectional dependency**, not a simple wrapper. Plane 1: agentdb WRAPS ruvector as preferred vector engine (TS→TS dynamic optional imports). Plane 2: ruvector ships `agentdb-fast.ts` — a simplified agentdb-compatible API. Plane 3: Rust `rvf-adapter-agentdb` maps agent memory to RVF segments (genuine code, broken persistence). Plane 4: agentic-flow COMPOSES both as peer deps. Plane 5: ruvector CLI delegates workers to `npx agentic-flow@alpha`. The published AgentDB npm copy has ZERO ruvector imports; the dev copy (agentic-flow repo) has 4 new commands with optional `@ruvector/*` dynamic imports — copies are significantly diverged. "AgentDB is a simplified ruvector" is partially true only in Plane 2 (ruvector ships an agentdb shim), but inverted in Plane 1 (agentdb is the client, ruvector is the engine).

**R118 ruvbot verdict:** RuvBot is a standalone LLM chatbot scaffold with **ZERO ruvector integration** despite being co-located in the ruvector monorepo. No @ruvector imports, no WASM, no attention, no embeddings. Self-training metrics (84%/12%) completely absent. It is NOT the "optimizer" desktop app — it's a Node.js HTTP server + EventEmitter with Slack/Discord/Webhook stubs. The "RuVector's WASM vector operations" header comment is a facade.

**R118 rvf-mcp-server verdict:** 8th genuine MCP confirmed. MCP protocol mechanics real (10 tools, 2 prompts, dual transport). But RVF backend is pure in-memory JS Map — `@ruvector/rvf` declared but never imported. `rvf_compact` is a no-op, `rvf_query` is O(n), `rvf_open_store` never reads from disk. Confirms 14th disconnected subsystem.

**R118 RVF migration path (Feb26-LEAD-009):** 3-stage path confirmed: (1) legacy .db+.meta.json → (2) SqlJsRvfBackend .rvf (SQLite) → (3) native @ruvector/rvf .rvf (HNSW binary). CRITICAL: SqlJsRvfBackend uses `.rvf` extension but contains SQLite — format identity deception alongside C26/C33.

**R135 claude-flow CLI entry point verdict:** The claude-flow CLI (cli.js 156 LOC, mcp-server.js 189 LOC) is a **cold dispatcher** — zero subsystem initialization at boot (no memory, no AgentDB, no HNSW, no ruvector). AgentDB access is 3 layers deep: entry point -> mcp-client.ts TOOL_REGISTRY (22 modules, ~256 tools) -> agentdb-tools.ts -> memory-bridge.js. Two disjoint bootstrap paths (MCP vs CLI) share NO initialization code. The MCP server near-duplicates the CLI's MCP path and false-advertises resources capability ({subscribe: true, listChanged: true} with zero handlers). This confirms the "integration gap is organizational" verdict — AgentDB is available as a lazy-loaded tool module but has no first-class initialization at the application entry layer.

The integration gap is organizational, not technical. AgentDB quality exceeds the rest of the ruvnet ecosystem across search, quantization, security, and attention.
