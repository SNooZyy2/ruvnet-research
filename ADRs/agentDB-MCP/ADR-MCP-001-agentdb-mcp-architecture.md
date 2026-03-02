# ADR-MCP-001: AgentDB MCP Server — Architectural Recovery & Consolidation Strategy

**Status**: PROPOSED
**Date**: 2026-02-21
**Deciders**: Architecture Team
**Research Sessions**: R6, R18, R20, R25, R32, R40, R43, R51, R63, R65, R67, R72, R84, R91, R96, R112, R114
**Evidence Base**: 1,507 findings across 1,002 AgentDB files + 496 MCP files (114 sessions)

---

## 1. Context

### 1.1 The Problem

The ruvnet multi-repo universe contains **7 parallel MCP protocol implementations** that share zero inter-process communication. Each was built independently with different SDKs, transport layers, and tool registration patterns. The result is a fragmented service mesh where:

- **None of the 7 (possibly 8) MCP servers talk to each other.** Zero inter-MCP bridge protocol, zero shared state, zero tool routing. This is the central architectural failure — not "too many servers" but "zero composition primitives."
- **MCP Server #4** (AgentDB native) is the ONLY server with a working end-to-end vector pipeline
- **MCP Server #1** (claude-flow main, 256 tools) never initializes `EmbeddingService`, degrading to hash-based pseudoembeddings
- Only **4 of 7 servers are functional islands** (#1, #4, #5, #7) — the other 3 are wrappers or facades that add zero value
- **4+ independent HNSW indexes** exist across subsystems but never share or query across each other — index fragmentation is distinct from the embedding problem
- **Two parallel episodic memory systems** exist (R104) with `context_manager` composing only 2/5 siblings
- **13 disconnected persistence layers** store data independently with no reconciliation
- **14+ hash-based embedding instances** exist because no shared embedding service is available

### 1.2 Research Evidence Summary

| Metric | Value | Source |
|--------|-------|--------|
| Total AgentDB files inventoried | 1,002 | research.db `files` table |
| DEEP-read AgentDB files | 59 | 59 files with 50%+ LOC traced |
| AgentDB/MCP findings (all severity) | 1,507 | research.db `findings` table |
| CRITICAL findings (AgentDB+MCP+Embedding) | 208 | Filtered from 11,125 total |
| Cross-file dependencies mapped | 195 | AgentDB/MCP dependency edges |
| AgentDB package connectivity | 29 cross-deps (CONNECTED) | `package_connectivity` view |
| Sessions touching AgentDB/MCP | 63 of 114 | File-read session joins |
| AgentDB LOC (packages/agentdb/) | 140,908 | `packages` table |
| Domain: agentdb-integration coverage | 55.72% by LOC | `domain_coverage` view |

### 1.3 The 7 Parallel MCP Protocols

| # | Location | SDK | Tools | Working? | Notes |
|---|----------|-----|-------|----------|-------|
| 1 | `dist/src/mcp-server.js` | Raw JSON-RPC 2.0 | 256 | Degraded | Many handlers → no-ops without deps (R6) |
| 2 | `agentic-flow/src/mcp/claudeFlowSdkServer.ts` | @anthropic-ai/claude-agent-sdk | 7 | Wrapper | All 7 shell out via `execSync` (R51) |
| 3 | `agentic-flow/src/mcp/fastmcp/servers/http-streaming-updated.ts` | FastMCP TS | 11 | Wrapper | CLI wrappers via `execSync` (R51) |
| **4** | **`packages/agentdb/src/mcp/agentdb-mcp-server.ts`** | **@modelcontextprotocol/sdk + StdioServerTransport** | **27-34** | **YES** | **Standalone. Real EmbeddingService. THE ONE THAT WORKS (R20)** |
| 5 | `ruv-swarm/crates/ruv-swarm-mcp/src/service.rs` | rmcp v0.2.1 (Rust) | 11 | Genuine | Two-layer delegation. No TS integration (R72) |
| 6 | `reasoningbank/crates/reasoningbank-mcp/src/server.rs` | Custom Rust traits | 4+2 | Isolated | Hand-rolled. Zero integration (R78) |
| 7 | `crates/mcp-gate/src/server.rs` | Rust (cognitum-gate) | 3 | Complete | permit/receipt/replay. 91% quality (R114) |
| ~8 | `crates/psycho-symbolic-reasoner/mcp-integration/src/index.ts` | @modelcontextprotocol/sdk TS (official) | 16 | Near-miss | Same official SDK as #4. 16 tools. Not a wrapper (R80) |

### 1.4 Functional Islands vs Wrappers

Not all 7 servers are equal. The critical taxonomy:

**Functional Islands (4 — each does real, independent work):**
- **#1** (claude-flow main): 256 tools, many degraded but genuine registry
- **#4** (AgentDB native): Real vector search, standalone process
- **#5** (ruv-swarm Rust): Genuine two-layer delegation, Rust-native
- **#7** (mcp-gate): Coherence gating, complete at 91%

**Wrappers/Facades (3 — add zero value):**
- **#2** (claude-agent-sdk): All 7 tools shell out via `execSync`
- **#3** (FastMCP): All 11 tools shell out via `execSync`
- **#6** (ReasoningBank Rust): Hand-rolled traits, zero integration

Consolidation means **composing 4 islands**, not simplifying 7 servers. The near-miss #8 (psycho-symbolic, 16 tools) would be a 5th island if wired.

---

## 2. Decision

### 2.1 Designate AgentDB Native MCP Server (#4) as the Canonical Vector Search Service

**Rationale**: It is the ONLY implementation that:
1. Calls `EmbeddingService.initialize()` at startup (line 196-201)
2. Loads `@xenova/transformers` with `all-MiniLM-L6-v2` (384-dim ONNX)
3. Produces real `Float32Array` embeddings for every episode
4. Backs them with `hnswlib-node` (genuine HNSW cosine search)
5. Exposes 27-34 tools covering Core Vector, Frontier Memory, Learning System, Batch Ops, and Attention

### 2.2 Consolidation Strategy (Three Horizons)

#### Horizon 1: Immediate (Today)
1. Find `packages/agentdb/src/mcp/agentdb-mcp-server.ts` in the agentic-flow repo
2. Run it as a standalone MCP server process
3. Configure `.mcp.json` to point at it (see ADR-MCP-004 Section 5 for config)
4. It will call `EmbeddingService.initialize()` on startup — real 384-dim embeddings
5. Real HNSW search works immediately. Accept TypeScript performance.

#### Horizon 2: Short-Term (Weeks)
- Build `mcp-gate` (#7) from source alongside #4
- Route coherence-gating decisions through mcp-gate
- Bridge #4 output → #7 input for gated vector queries

#### Horizon 3: Long-Term (Months) — Prime-Radiant as Convergence Target

Prime-radiant is the most architecturally ambitious crate — a genuine sheaf-theoretic knowledge substrate with 143 internal dependencies and high-quality execution (~89%) and governance (~88%) layers. But today it is an **architectural vision document written in Rust**: no HNSW (brute-force O(n) everywhere), no embeddings, no working persistence, three broken Laplacian eigensolvers, and zero Cargo.toml workspace deps in either direction.

**"Prime-radiant is the cathedral; the TypeScript layer is the bazaar."**

The 5-step path to make it the convergence point:
1. Fix the 3 Laplacian eigensolvers (coherence/spectral.rs, laplacian.rs, cocycle.rs) + `is_coboundary()` always-false bug
2. Wire `ruvector-core` for HNSW (replace brute-force O(n) in storage/file.rs, hyperbolic/adapter.rs)
3. Add real embedding generation (ONNX via `ort` crate, or candle)
4. Implement the PostgresStorage backend (feature-gated, never wired — R107)
5. Connect it via Cargo.toml to ruvllm (ruvllm_integration/ references ruvllm types but Cargo.toml doesn't declare the dep — aspirational code that can't compile)

Additionally in H3:
- Replace TS vector pipeline with native Rust end-to-end
- Consolidate 4+ independent HNSW indexes into shared ruvector-core instance
- Retire wrappers (#2, #3) and isolated servers (#6)
- Reconcile two parallel episodic memory systems (R104)

### 2.3 Retirement Plan for Non-Canonical Servers

| Server | Action | Timeline |
|--------|--------|----------|
| #1 (claude-flow main) | Keep for non-vector tools (session, config, workflow) | Keep indefinitely |
| #2 (claude-agent-sdk) | Retire — pure CLI wrapper | H1 |
| #3 (FastMCP) | Retire — pure CLI wrapper | H1 |
| #4 (AgentDB native) | **PROMOTE to canonical vector service** | H1 |
| #5 (ruv-swarm Rust) | Keep for Rust-native swarm coordination | Keep, evolve |
| #6 (ReasoningBank Rust) | Retire or absorb into #5 | H2 |
| #7 (mcp-gate) | Keep for coherence gating | Keep, compose with #4 |
| ~#8 (psycho-symbolic) | Evaluate — 16 tools, official SDK, potential 5th island | H2 |

---

## 3. Technical Architecture

### 3.1 Current State: The Working Vector Pipeline

```
User text
  → EmbeddingService.initialize()           [agentdb-mcp-server.ts L196-201]
  → @xenova/transformers pipeline()          [EmbeddingService.ts]
  → all-MiniLM-L6-v2 (384-dim ONNX)
  → Float32Array
  → VectorBackend.insert()                   [HNSWIndex.ts]
  → hnswlib-node (real HNSW index)
  → VectorBackend.search()
  → cosine similarity
  → results
```

### 3.2 Why It's Broken in claude-flow Main MCP (#1)

The claude-flow bridge **never calls `EmbeddingService.initialize()`**.

Evidence chain:
- `src/cli/commands/install-embeddings.ts` (R84): Embeddings treated as **optional manual install** (`claude-flow install-embeddings`)
- Without this step, `EmbeddingService` falls back to `mockEmbedding()` — sin/cos hash producing deterministic but semantically meaningless vectors
- `claude_flow_bridge.rs` (R104): Imports `{chrono, parking_lot, serde, std::process::Command}` — zero ruvector-core, zero VectorDB, zero HNSW. Spawns subprocess.

### 3.3 Rust Side: Real HNSW, No Embeddings

| Rust Path | Real HNSW? | Real Embeddings? | End-to-End? |
|-----------|-----------|-----------------|-------------|
| `ruvllm/reasoning_bank/pattern_store.rs` | YES (ruvector-core VectorDB, R103) | **REAL VectorDB** (R102 correction) | CLOSEST to E2E |
| `ruvllm/context/agentic_memory.rs` | YES (HnswIndex) | Caller must supply | NO |
| `ruvllm/context/semantic_cache.rs` | YES (genuine HNSW, Cosine) | Caller must supply | NO |
| `ruvllm/ruvector_integration.rs` | YES (HNSW + SONA) | Caller must supply | NO |
| `ruvector-core` | YES (wraps hnsw_rs, real SIMD) | No embedding model | NO |

**Pattern**: Nearly every Rust path expects caller-supplied `Vec<f32>`. The **one exception** is `pattern_store.rs` which uses a real VectorDB (first non-hash semantic store in ruvllm, R102) — but it still lacks an integrated embedding model. Zero ONNX/candle embedding integration exists in Rust.

**HNSW Index Fragmentation**: These 4+ real HNSW indexes are **independent instances that never share data**. Even after embedding is fixed, queries against one index cannot find vectors stored in another. This is a distinct problem from the embedding gap — consolidation requires either a shared HNSW service or cross-index federation.

### 3.4 The 14+ Hash-Based Embedding Instances

| Location | Method | Quality |
|----------|--------|---------|
| `ruvllm/bitnet/rlm_embedder.rs` | FNV-1a hash + bigram | FAKE |
| `ruvllm/backends/candle_backend.rs` | bytes summed by modulo | FAKE (comment admits) |
| `reasoningbank-core/src/similarity.rs` | char→bytes→normalize | FAKE (comment admits) |
| `agentdb/src/mcp/attention-tools-handlers.ts` | char-code hashing | FAKE |
| `agentic-flow/src/optimizations/ruvector-backend.ts` | Map-based in-memory | NO persistence |
| `agentdb-service-fallback` (dist/) | SQL-only storage | NO embeddings at all |

---

## 4. Consequences

### 4.1 Positive
- Single canonical path for vector search eliminates ambiguity
- 27-34 MCP tools immediately usable without Rust compilation
- Real 384-dim embeddings + HNSW cosine = production-quality search
- Clear upgrade path from TS→Rust when embedding model available

### 4.2 Negative
- TypeScript performance ceiling (no SIMD, no AVX)
- Single-process bottleneck (StdioServerTransport is serial)
- 13 persistence layers remain disconnected until Horizon 3
- Teams using #1 (claude-flow main) must route vector queries to #4

### 4.3 Risks
- `@xenova/transformers` model download on cold start (~100MB, 10-30s)
- `hnswlib-node` native addon requires compatible Node.js + build tools
- Version drift between `agentdb-mcp-server.ts` copies (exists in both `packages/agentdb/` and `src/mcp/`)

---

## 5. Compliance

- **ADR-006 (Unified Memory Service)**: This ADR provides the first concrete pathway to fulfill ADR-006 by designating #4 as the canonical memory access point
- **ADR-009 (Hybrid Memory Backend)**: AgentDB native uses SQLite + HNSW hybrid, aligning with ADR-009
- **ADR-008 (3-Tier Model Routing)**: Vector search routing should integrate with model routing for query complexity assessment
- **R20 Root Cause**: Fully addresses the R20 finding that `EmbeddingService` was never initialized in claude-flow bridge
