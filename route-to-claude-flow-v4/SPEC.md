# claude-flow v4 Specification

> **Status**: Initial outline
> **Date**: 2026-02-17
> **Based on**: 90 research sessions, 1,332 DEEP files, 9,171 findings across 4 repos
> **Predecessor**: claude-flow-self-implemented (76 commits, abandoned — built on broken foundation)

## Executive Summary

The ruvnet multi-repo ecosystem contains ~50-80K LOC of genuine, production-quality Rust code (92-98% realness) buried under ~200K+ LOC of theatrical TypeScript scaffolding. The genuine algorithms (HNSW, PQ, sublinear PageRank, quantum error correction, temporal analysis) are unreachable through the public API because the integration layer was never completed. claude-flow v4 keeps the genuine Rust, discards the theatrical TS, and builds a thin new integration layer that actually works.

The single most impactful fix: **initialize the EmbeddingService** (R20 root cause). Once real embeddings flow into RuVectorBackend (which already works), semantic search is operational.

---

## 1. Genuine Assets Inventory

### Tier 1: Production-Ready Rust Crates (copy verbatim)

| Crate | Source Repo | Key Files | LOC | Realness | What It Does |
|-------|------------|-----------|-----|----------|-------------|
| ruvector-core HNSW | ruvector | `hnsw.rs`, `simd.rs` | ~2,000 | 92-98% | HNSW vector search with AVX-512/AVX2/NEON SIMD dispatch |
| ruvector-core PQ | ruvector | `product_quantization.rs` | 551 | 88-92% | Product quantization with k-means++, Lloyd's, ADC lookup tables |
| ruvector-core conformal | ruvector | `conformal_prediction.rs` | 505 | 88-93% | Split-conformal prediction with Vovk et al. quantile formula |
| ruvector-core hypergraph | ruvector | `hypergraph.rs` | 551 | 85-90% | Bipartite incidence hypergraph with k-hop BFS, NeurIPS 2025 design |
| temporal-tensor | sublinear-time-solver | entire crate | ~3,000 | 93% | Time series analysis, 213 tests, PRODUCTION-READY |
| ruQu quantum | sublinear-time-solver | `decoder.rs`, core files | ~2,500 | 91-95% | Quantum error correction: Union-Find, MWPM, surface codes, AVX2 |
| backward_push | sublinear-time-solver | `backward_push.rs` | ~500 | 95%+ | Genuine O(1/epsilon) sublinear PageRank |
| bit-parallel-search | sublinear-time-solver | `lib.rs` | 198 | 92-95% | Textbook Shift-Or O(n) string matching |
| micro_lora + EWC++ | sublinear-time-solver | `micro_lora.rs` | ~500 | 92-95% | NEON SIMD LoRA with Elastic Weight Consolidation |
| RAC consensus | sublinear-time-solver | `rac.rs`, `p2p.rs` | ~1,500 | 92% | Raft consensus + real libp2p transport |
| shard partitioner | ruvector | `shard.rs` | 596 | 70-80% | EdgeCutMinimizer (multilevel Kernighan-Lin), xxh3/blake3 hashing |

### Tier 2: Genuine TS/JS (adapt and keep)

| Module | Source | LOC | Realness | What It Does |
|--------|--------|-----|----------|-------------|
| RuVectorBackend.ts | agentdb | ~500 | 88-92% | HNSW integration with adaptive params, Semaphore, BufferPool, path security |
| ReasoningBank (TS) | claude-flow-cli | ~800 | 92-95% | Statistical ranking, decay coefficients, MMR search |
| ReasoningBank (Rust) | sublinear-time-solver | ~600 | 92-95% | WASM-compatible, v1->v2 migration complete |
| pre-task.ts hook | claude-flow-cli | ~200 | 88-92% | ReasoningBank hook with 4-factor scoring |

### Tier 3: Salvage from self-implemented repo

| Module | Path | LOC | Status |
|--------|------|-----|--------|
| DDD infrastructure adapters | `src/agentdb-integration/infrastructure/` | ~1,200 | Retarget imports |
| Episodic/reflexion services | `src/agentdb-integration/episodic/` | ~900 | Retarget imports |
| Skill library services | `src/agentdb-integration/skill/` | ~1,400 | Retarget imports |
| Hybrid search pipeline | `src/agentdb-integration/search/` | ~1,100 | Retarget imports |
| Input validator | `src/agentdb-integration/security/` | 270 | Keep as-is |
| Event bus | `src/agentdb-integration/events/` | ~400 | Keep as-is |
| MCP tools | `src/agentdb-integration/mcp-tools/` | ~500 | Retarget to new MCP server |
| Test suite | `tests/` | 9,075 | Adapt assertions to new interfaces |

---

## 2. What Gets Discarded

### Categories of dead code (DO NOT COPY)

| Category | Count | Examples |
|----------|-------|---------|
| Theatrical WASM stubs | 13 | DAA lib.rs (37 lines), test-build.js, activation.rs |
| Hash-based embedding placeholders | 16+ | ruvector-core, ruvllm/candle, sona_llm, training |
| Disconnected persistence layers | 9 | ruv-swarm-memory.js, sqlite-worker.js, memory-config.js |
| Parallel MCP protocols | 5 of 6 | Keep only @modelcontextprotocol/sdk |
| CLI demo skeletons | All | monitor.rs, todo!() stubs, "not yet implemented" |
| False sublinearity | 8 instances | temporal-lead-solver (O(n^2) as "sublinear") |
| Fabricated systems | Multiple | EmergenceSystem (51%), consciousness theory (0-5%) |
| Dead code | Multiple | callbacks.rs (defined but never invoked), performance_benchmark.rs (orphaned) |
| lib_simple.rs facade | 1 | Deliberately excludes genuine algorithms from WASM surface |
| Distributed transport stubs | 4 | rpc.rs (15-20%), coordinator.rs (30-35%), gossip transport, federation transport |

### Packages to remove entirely

- `@claude-flow/guidance` — 56% theatrical WASM, hash embeddings systemic
- `agentic-flow` npm — single-node task runner (R40), fabricated emergence
- All `SublinearSolver` TS wrappers — routes through theatrical facade

---

## 3. Architecture Decisions

### ADR-v4-001: Single MCP Protocol

**Decision**: Use `@modelcontextprotocol/sdk` exclusively.

**Context**: The original codebase has 6 parallel MCP protocols (@anthropic/sdk, @modelcontextprotocol/sdk, custom WS, JSON-RPC, custom TS, rmcp Rust). None interoperate. The ruv-swarm MCP tests (R84) confirmed @modelcontextprotocol/sdk is the most complete.

**Consequence**: All MCP tool registration goes through one SDK. No custom protocols.

### ADR-v4-002: Single Persistence Layer

**Decision**: SQLite via `better-sqlite3` as the only data store.

**Context**: 9 disconnected persistence layers discovered (R85-R87). The only ones that actually work are better-sqlite3 based (test-memory-storage.js R86: 88-92%, verify-db-updates.js R87: 88-92%).

**Consequence**: One database file, one schema, one migration system. The self-implemented `schema-migrator.ts` and `database-adapter.ts` are the right pattern.

### ADR-v4-003: Real Embeddings Mandatory

**Decision**: EmbeddingService must be initialized at startup. No hash-based fallback.

**Context**: R20 root cause (confirmed across R20, R52, R65, R84, R88): EmbeddingService was never initialized in the claude-flow bridge. Hash-based embeddings are systemic (16+ files). RuVectorBackend works correctly but receives garbage input.

**Implementation**:
```typescript
// Mandatory at startup — fail fast if unavailable
const embeddingService = new EmbeddingService();
await embeddingService.initialize(); // Uses @xenova/transformers
// If this fails, the system should NOT start with hash fallback
```

**Consequence**: Semantic search either works correctly or fails visibly. No silent degradation to hash-based non-search.

### ADR-v4-004: WASM Exports from lib.rs

**Decision**: All Rust-to-WASM exports come from `lib.rs` (the real API), never `lib_simple.rs` (the facade).

**Context**: R85 discovered `lib_simple.rs` deliberately excludes genuine sublinear algorithms. R88 confirmed `src/index.ts` routes the public API through this theatrical facade. R81's orphaned JS bridge targeted the wrong entry point.

**Consequence**: The WASM build step must compile `lib.rs` with proper `wasm_bindgen` annotations for all genuine algorithms.

### ADR-v4-005: DDD Bounded Contexts

**Decision**: Reuse the bounded context structure from claude-flow-self-implemented.

**Context**: The self-implemented DDD architecture is sound (episodic, skill, search, infrastructure, events). The problem was the foundation (upstream deps), not the domain model.

**Bounded contexts**:
- **Search** — hybrid BM25 + vector search via RuVectorBackend
- **Episodic** — reflexion/experience replay via ReasoningBank
- **Skill** — skill library with consolidation service
- **Infrastructure** — adapters, persistence, embedding service
- **Events** — domain event bus for cross-context communication

### ADR-v4-006: No Upstream Dependencies

**Decision**: Do not depend on `@claude-flow/guidance`, `agentic-flow`, or any npm package from the ruvnet ecosystem.

**Context**: These packages are the broken integration layer. The genuine code is in standalone Rust crates that can be compiled independently.

**Consequence**: All Rust functionality accessed via direct WASM imports or FFI. No npm intermediary packages.

---

## 4. Target Architecture

```
                    ┌──────────────────────────────────┐
                    │         Claude Code CLI           │
                    │    (consumer — not modified)      │
                    └──────────┬───────────────────────┘
                               │ MCP protocol
                    ┌──────────▼───────────────────────┐
                    │     MCP Server (NEW — ~500 LOC)   │
                    │   @modelcontextprotocol/sdk       │
                    │   Tool registry + request routing │
                    └──────────┬───────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼──────┐ ┌──────▼───────┐ ┌──────▼───────┐
    │  Search Context │ │ Episodic Ctx │ │  Skill Ctx   │
    │  (from self-impl)│ │(from self-impl)│ │(from self-impl)│
    │  BM25 + vector  │ │ reflexion    │ │ consolidation│
    └─────────┬──────┘ └──────┬───────┘ └──────┬───────┘
              │                │                │
    ┌─────────▼────────────────▼────────────────▼───────┐
    │           Infrastructure Layer (ADAPT)             │
    │  ┌──────────────┐ ┌─────────────┐ ┌─────────────┐ │
    │  │EmbeddingService│ │RuVectorBackend│ │ReasoningBank│ │
    │  │(NEW — fix R20)│ │ (COPY 88-92%)│ │(COPY 92-95%)│ │
    │  └──────┬───────┘ └──────┬──────┘ └──────┬──────┘ │
    │         │                │               │         │
    │  ┌──────▼────────────────▼───────────────▼──────┐ │
    │  │        SQLite (better-sqlite3)                │ │
    │  │        ONE database, ONE schema               │ │
    │  └───────────────────────────────────────────────┘ │
    └───────────────────────┬───────────────────────────┘
                            │ WASM / FFI
    ┌───────────────────────▼───────────────────────────┐
    │              Rust Crates (COPY)                     │
    │  ┌──────────┐ ┌──────────┐ ┌───────────┐          │
    │  │ruvector  │ │temporal  │ │backward   │          │
    │  │core HNSW │ │tensor    │ │push       │  ...     │
    │  │PQ,conformal│ │(213 tests)│ │O(1/eps)  │          │
    │  └──────────┘ └──────────┘ └───────────┘          │
    └───────────────────────────────────────────────────┘
```

---

## 5. Implementation Phases

### Phase 0: Extraction (~1-2 days)

1. Create fresh repo `claude-flow-v4/`
2. Copy Tier 1 Rust crates verbatim (preserve Cargo.toml, tests, benches)
3. Verify each crate compiles standalone: `cargo build`, `cargo test`
4. Copy Tier 2 TS files (RuVectorBackend.ts, ReasoningBank)
5. Copy Tier 3 self-implemented DDD services (strip upstream imports)

**Validation gate**: All Rust crates pass `cargo test` independently.

### Phase 1: Fix R20 Root Cause (~1 day)

1. Implement `EmbeddingService` with `@xenova/transformers`
2. Wire it into `RuVectorBackend` initialization
3. Verify: store 100 text documents, search returns semantically relevant results
4. No hash-based fallback — fail fast if model unavailable

**Validation gate**: `search("authentication")` returns auth-related documents, not random results.

### Phase 2: Single MCP + Single Persistence (~2-3 days)

1. Create MCP server using `@modelcontextprotocol/sdk`
2. Register tools wrapping each DDD service (search, episodic, skill)
3. Set up SQLite schema (adapt from self-impl `agentdb-schema.sql`)
4. Wire `database-adapter.ts` to the single SQLite instance
5. Connect event bus for cross-context communication

**Validation gate**: All MCP tools callable from Claude Code. Data persists across sessions.

### Phase 3: WASM Bridge (~2-3 days)

1. Add `wasm_bindgen` exports to genuine Rust crate entry points (`lib.rs`, NOT `lib_simple.rs`)
2. Build with `wasm-pack build --target nodejs`
3. Create TS wrapper that imports WASM and exposes to service layer
4. Wire HNSW search, backward_push, PQ compression through WASM to MCP tools

**Validation gate**: `backward_push` callable from MCP tool and returns correct PageRank approximation.

### Phase 4: Integration Tests (~1-2 days)

1. Adapt self-implemented test suite (9,075 LOC) to new interfaces
2. End-to-end: Claude Code → MCP → service → WASM → Rust crate → result
3. Verify no imports from `@claude-flow/guidance` or `agentic-flow`
4. Performance: HNSW search < 10ms for 100K vectors, embedding < 100ms

**Validation gate**: Full test suite passes. Zero upstream dependencies.

---

## 6. Estimated Effort

| Phase | New Code | Adapted Code | Copied Code | Time |
|-------|----------|-------------|-------------|------|
| Phase 0: Extraction | 0 | 0 | ~15K LOC Rust + ~2K TS | 1-2 days |
| Phase 1: R20 Fix | ~200 LOC | 0 | 0 | 1 day |
| Phase 2: MCP + Persistence | ~1,500 LOC | ~3,000 LOC (self-impl) | 0 | 2-3 days |
| Phase 3: WASM Bridge | ~1,000 LOC | 0 | 0 | 2-3 days |
| Phase 4: Tests | ~500 LOC | ~9,000 LOC (self-impl tests) | 0 | 1-2 days |
| **Total** | **~3,200 LOC new** | **~12,000 LOC adapted** | **~17,000 LOC copied** | **~7-11 days** |

---

## 7. Risk Registry

| Risk | Mitigation |
|------|-----------|
| Rust crates may not compile standalone (hidden deps) | Phase 0 validation gate: `cargo test` each crate |
| @xenova/transformers model too large for CLI | Use `all-MiniLM-L6-v2` (~80MB), lazy-load on first search |
| WASM build breaks with `wasm_bindgen` on complex types | Start with simple primitives (f32 arrays, u32 indices), avoid complex structs across boundary |
| Self-implemented DDD services have implicit upstream deps | Grep for all `@claude-flow` and `agentic-flow` imports, replace with direct adapters |
| ruvector-graph distributed module tempting to include | DO NOT include — it's 15-55% with no transport. Defer to v5 if needed |

---

## 8. Success Criteria

1. **Semantic search works**: Query returns contextually relevant results (not hash-based random)
2. **Single MCP protocol**: All tools registered via @modelcontextprotocol/sdk, callable from Claude Code
3. **Single persistence**: One SQLite file, data survives restarts
4. **Genuine algorithms accessible**: backward_push, HNSW, PQ callable through the public API
5. **Zero upstream deps**: No imports from @claude-flow/guidance, agentic-flow, or any ruvnet npm package
6. **Tests pass**: Adapted test suite from self-implemented repo validates all integration points
7. **No theatrical code**: Every file in the repo scores 80%+ by the research project's realness criteria

---

## Appendix A: Research Project Reference

- **Research DB**: `/home/snoozyy/ruvnet-research/db/research.db`
- **MASTER-INDEX**: `/home/snoozyy/ruvnet-research/MASTER-INDEX.md` (3,973 lines)
- **Domain analyses**: `/home/snoozyy/ruvnet-research/domains/*/analysis.md` (14 domains)
- **Session plans**: `/home/snoozyy/ruvnet-research/daily-plan/R*.md` (R01-R90)
- **Self-implemented repo**: `/home/snoozyy/claude-flow-self-implemented/`
- **Source repos**: `/home/snoozyy/repos/` (ruvector, ruv-FANN, agentic-flow, sublinear-time-solver)

## Appendix B: Key Research Sessions

| Session | Key Finding | Relevance to v4 |
|---------|------------|-----------------|
| R20 | AgentDB search broken — EmbeddingService never initialized | Phase 1 fix target |
| R23/R85 | Neural-net crate 75-85% (core genuine, infra theatrical) | Know what to copy vs skip |
| R37 | temporal-tensor 93%, 213 tests, PRODUCTION-READY | Copy verbatim |
| R39 | ruQu 91-95% HIGHEST QUALITY multi-file crate | Copy verbatim |
| R42 | RAC 92% HIGHEST QUALITY single-file Rust | Copy verbatim |
| R56 | backward_push.rs GENUINE O(1/epsilon) sublinear | Copy verbatim |
| R85 | lib_simple.rs deliberately excludes genuine algorithms | ADR-v4-004: export from lib.rs |
| R87 | bit-parallel-search 92-95% GENUINE Shift-Or | Copy verbatim |
| R88 | ruvector/index.ts 88-92% — R20 backend is genuine | Copy RuVectorBackend.ts |
| R90 | PQ 88-92%, conformal 88-93% — ruvector-core advanced features genuine | Copy to v4 crates |
| R90 | Distributed module 15-55% — design-doc-as-code, no transport | DO NOT include in v4 |
