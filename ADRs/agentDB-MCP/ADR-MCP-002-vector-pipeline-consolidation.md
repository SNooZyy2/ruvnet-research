# ADR-MCP-002: Vector Pipeline Consolidation — From 14 Hash Fallbacks to One Real Path

**Status**: PROPOSED
**Date**: 2026-02-21
**Supersedes**: Implicit per-subsystem embedding strategies
**Related**: ADR-MCP-001, ADR-006 (Unified Memory), ADR-009 (Hybrid Backend)

---

## 1. Context

### The Systemic Embedding Failure

Across the ruvnet multi-repo universe, **14+ independent subsystems** each implement their own embedding generation. Because no shared embedding service exists at the Rust level, and the TypeScript `EmbeddingService` is treated as optional, every subsystem independently falls back to hash-based pseudo-embeddings.

This is not 14 independent bugs. It is **one architectural gap** manifesting 14 times.

### The Parallel Problem: HNSW Index Fragmentation

Even where real HNSW exists, the indexes are **independent instances that never compose**:
- `ruvllm/reasoning_bank/pattern_store.rs` — ruvector-core VectorDB (R102: first non-hash store)
- `ruvllm/context/agentic_memory.rs` — HnswIndex
- `ruvllm/context/semantic_cache.rs` — HNSW with Cosine
- `ruvllm/ruvector_integration.rs` — HNSW + SONA ReasoningBank
- `packages/agentdb/` — hnswlib-node (TypeScript)
- prime-radiant — brute-force O(n) scan (no HNSW at all despite `ShardedHyperbolicHnsw` comment in adapter.rs)

Fixing embeddings without consolidating indexes means vectors stored in one HNSW instance remain invisible to queries against another. These are **two distinct architectural gaps** that must be addressed together.

### R102 Correction: One Rust Path Has Real VectorDB

`reasoning_bank/pattern_store.rs` (R102) uses a genuine ruvector-core `VectorDB` — the first and only non-hash semantic store in ruvllm. It still lacks an embedding model (callers supply `Vec<f32>`) but it breaks the "every Rust path is hash-based" generalization. This file is the closest Rust analog to AgentDB native's vector pipeline.

### Evidence: The Hash Epidemic

| # | Location | Hash Method | Dimension | Why It Exists |
|---|----------|-------------|-----------|---------------|
| 1 | `ruvllm/bitnet/rlm_embedder.rs` | FNV-1a + bigram features | Variable | No ONNX/candle model available |
| 2 | `ruvllm/backends/candle_backend.rs` | bytes summed by modulo | Variable | Comment: "placeholder" |
| 3 | `reasoningbank-core/src/similarity.rs` | char→bytes→normalize | Variable | Comment: "should use proper model" |
| 4 | `agentdb/src/mcp/attention-tools-handlers.ts` | char-code hashing | Variable | Standalone string templates |
| 5 | `agentic-flow/src/optimizations/ruvector-backend.ts` | Map-based (no embedding) | N/A | In-memory only |
| 6 | `src/cli/commands/install-embeddings.ts` | sin/cos mockEmbedding | 384 | Default when not installed |
| 7 | `agentdb-integration/bootstrap.ts` | SHA-256 hashEmbedder | 384 | Fallback when real unavailable |
| 8 | `sona_llm` training module | Token hash features | Variable | R22 identified |
| 9 | `agentic-flow/intelligence` module | Placeholder vectors | Variable | R40 identified |
| 10 | `agentdb-service-fallback` | SQL-only (no vectors) | 0 | Complete bypass |
| 11 | `ruvllm/context/agentic_memory.rs` | Caller-supplied only | None | Expects external embedder |
| 12 | `ruvllm/context/semantic_cache.rs` | Caller-supplied only | None | Expects external embedder |
| 13 | `ruvllm/reasoning_bank/pattern_store.rs` | **REAL VectorDB** (R102) | None | First non-hash in ruvllm, but no model |
| 14 | `ruvllm/ruvector_integration.rs` | Caller-supplied only | None | HNSW+SONA, no embedder |

### The One Working Path

```
AgentDB Native MCP Server (#4)
  → EmbeddingService.initialize()
  → @xenova/transformers pipeline('feature-extraction')
  → Xenova/all-MiniLM-L6-v2 (384-dim ONNX)
  → Float32Array(384)
  → hnswlib-node HNSW index
  → cosine similarity search
  → results
```

This path ONLY works in `packages/agentdb/src/mcp/agentdb-mcp-server.ts` because it is the ONLY entry point that calls `EmbeddingService.initialize()` at startup.

---

## 2. Decision

### 2.1 Establish Shared Embedding Service Contract

Define a cross-context embedding interface that ALL subsystems must use:

```typescript
// packages/agentdb/src/contracts/embedding-provider.ts
interface EmbeddingProvider {
  /** Generate embedding for text. MUST NOT return hash-based vectors. */
  embed(text: string): Promise<Float32Array>;

  /** Batch embed for efficiency */
  embedBatch(texts: string[]): Promise<Float32Array[]>;

  /** Verify service is initialized with real model */
  isReady(): boolean;

  /** Get embedding dimension */
  getDimension(): number;

  /** Health check — fails if using mock/hash */
  healthCheck(): Promise<{ real: boolean; model: string; dim: number }>;
}
```

### 2.2 Fail-Loud Policy

**REMOVE all silent fallbacks.** When `EmbeddingService` is not initialized:
- Do NOT fall back to `mockEmbedding()`
- Do NOT fall back to `hashEmbedder`
- THROW `EmbeddingServiceNotInitializedError` with clear instructions
- Log the error with the remediation command: `claude-flow install-embeddings`

### 2.3 Rust Embedding Strategy

For Horizon 3, provide a Rust-native embedding path:

| Option | Model | Performance | Dependency |
|--------|-------|-------------|------------|
| A. `candle` ONNX runtime | all-MiniLM-L6-v2 | ~5-10x TS | candle + ONNX model file |
| B. `ort` (ONNX Runtime) | all-MiniLM-L6-v2 | ~10-20x TS | ort crate + ONNX model file |
| C. Custom tokenizer + FFI | all-MiniLM-L6-v2 | ~15-30x TS | tokenizers + model weights |

**Recommendation**: Option B (`ort`) — best balance of performance and ecosystem maturity.

### 2.4 Dimension Alignment

Standardize on **384 dimensions** (all-MiniLM-L6-v2) as the canonical embedding dimension across all subsystems.

| Current Dimension | Subsystem | Migration Action |
|-------------------|-----------|-----------------|
| 384 | AgentDB native, integration layer | No change |
| 64 | prime-radiant memory_layer.rs | Re-embed with 384-dim |
| 768 | Various ruvllm contexts | Projection layer 768→384 |
| 1536 | Some ruvllm configs | Projection layer 1536→384 |
| Variable (hash) | All 14 hash instances | Replace with real 384-dim |

---

## 3. Consolidation Roadmap

### Phase 1: Fail-Loud (Week 1-2)
- Remove `mockEmbedding()` from `EmbeddingService`
- Remove `hashEmbedder` fallback from `bootstrap.ts`
- Add health check to AgentDB MCP server startup
- Emit `EmbeddingServiceDegraded` event (not silent)

### Phase 2: Shared Service (Week 3-4)
- Extract `EmbeddingProvider` interface
- AgentDB MCP server implements `EmbeddingProvider`
- Integration layer consumes via dependency injection
- claude-flow main MCP routes embedding requests to #4

### Phase 3: Rust Native + HNSW Consolidation (Month 2-3)
- Add `ort` ONNX runtime to `ruvector-core` Cargo.toml
- Implement `EmbeddingProvider` trait in Rust
- Wire to all Rust subsystems expecting `Vec<f32>` (start with `pattern_store.rs` — closest to E2E, R102)
- Benchmark against TypeScript path
- **Consolidate 4+ independent HNSW indexes** into shared ruvector-core instance or cross-index federation
- Replace prime-radiant brute-force O(n) scans (storage/file.rs, hyperbolic/adapter.rs) with ruvector-core HNSW

### Phase 4: Migration (Month 3-4)
- Re-embed all existing episodes with consistent 384-dim
- Migrate persistence layers 3-9 to use shared embedding
- Add dimension validation at vector insert boundary
- Reconcile two parallel episodic systems (R104: `context_manager` composes only 2/5 siblings)
- Retire hash-based implementations

---

## 4. Consequences

### Positive
- Single source of truth for embeddings eliminates 14 inconsistencies
- Real semantic search across ALL subsystems
- Clear contract enables Rust migration without breaking consumers
- Health checks make degradation visible

### Negative
- Cold-start penalty (~100MB model download, 10-30s init)
- Breaking change for consumers relying on hash determinism
- Memory cost of embedding model (~200MB resident)
- Batch embedding introduces latency for high-throughput paths

### Risks
- `@xenova/transformers` is unmaintained or breaks with Node.js updates
- ONNX model format changes in newer MiniLM versions
- Dimension mismatch during migration causes search quality regression
- Some subsystems may have hardcoded hash-based logic in tests

---

## 5. Metrics for Success

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Hash-based embedding instances | 14+ | 0 | Grep for hash/mock patterns |
| Real embedding coverage | 1/7 MCP servers | 7/7 | Health check endpoints |
| Search quality (MRR@10) | ~0.05 (hash) | >0.60 (real) | Evaluation suite |
| Embedding dimension consistency | 4+ dimensions | 1 (384) | Config audit |
| Silent fallback count | 14+ | 0 | Error monitoring |
