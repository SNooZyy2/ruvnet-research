# AgentDB MCP — Architecture Decision Records & DDD Analysis

**Generated**: 2026-02-21
**Evidence Base**: 114 research sessions, 1,507 findings, 195 dependencies, 59 DEEP files
**Scope**: AgentDB MCP Server ecosystem across the ruvnet multi-repo universe

---

## Documents

| Document | Purpose | Key Finding |
|----------|---------|-------------|
| [ADR-MCP-001](ADR-MCP-001-agentdb-mcp-architecture.md) | **Architecture Recovery** — The 7 parallel MCP protocols and why only #4 works | AgentDB native MCP is the ONLY end-to-end vector search path |
| [ADR-MCP-002](ADR-MCP-002-vector-pipeline-consolidation.md) | **Vector Pipeline** — From 14 hash fallbacks to one real embedding path | One architectural gap manifests 14 times |
| [ADR-MCP-003](ADR-MCP-003-persistence-layer-unification.md) | **Persistence Unification** — From 13 disconnected layers to tiered consolidation | 13 independent stores, zero reconciliation |
| [ADR-MCP-004](ADR-MCP-004-mcp-tool-inventory.md) | **Tool Inventory** — Complete capability matrix for 35 MCP tools | 29/35 work, 4 fabricated, 21 unreachable from claude-flow |
| [DDD Analysis](DDD-agentdb-mcp-bounded-contexts.md) | **Domain-Driven Design** — 9 bounded contexts, aggregates, invariants, ubiquitous language | Embedding Context is the root cause of systemic failure |

---

## Executive Summary

### What Works Today

```
AgentDB Native MCP Server (#4)
  → packages/agentdb/src/mcp/agentdb-mcp-server.ts
  → 29 working tools (of 35 total)
  → Real 384-dim embeddings via @xenova/transformers
  → Real HNSW via hnswlib-node
  → 9 RL algorithms
  → Episodic memory, skill library, causal graph
```

### What's Broken

1. **None of the 7 (possibly 8) MCP servers talk to each other** — zero composition primitives
2. **claude-flow main MCP (#1)** never initializes `EmbeddingService` → hash-based noise
3. **14+ hash-based embedding instances** across Rust and TypeScript (one R102 exception: `pattern_store.rs` has real VectorDB)
4. **4+ independent HNSW indexes** never share data — index fragmentation distinct from embedding problem
5. **Two parallel episodic memory systems** (R104) — `context_manager` composes only 2/5 siblings
6. **13 disconnected persistence layers** with zero reconciliation
7. **Prime-radiant** is the aspirational convergence target but has no HNSW, no embeddings, 3 broken eigensolvers
8. **4 attention tools** return fabricated `Math.random()` metrics
9. **21 tools** unreachable from claude-flow's MCP bridge

### Three Horizons

| Horizon | Timeline | Action |
|---------|----------|--------|
| **H1** | Today | Run AgentDB native MCP (#4) as standalone server |
| **H2** | Weeks | Add mcp-gate (#7) for coherence gating. Route vector tools #1→#4. Evaluate ~#8 (psycho-symbolic, 16 tools) |
| **H3** | Months | Prime-radiant convergence: fix eigensolvers, wire ruvector-core HNSW, add Rust ONNX embeddings, consolidate indexes, unify persistence |

---

## Data Sources

All findings are backed by the research database at `/home/snoozyy/ruvnet-research/db/research.db`.

### Key Queries

```sql
-- All AgentDB DEEP files (59)
SELECT * FROM files WHERE relative_path LIKE '%agentdb%' AND depth = 'DEEP';

-- CRITICAL findings for AgentDB/MCP (208)
SELECT fi.*, f.relative_path FROM findings fi
JOIN files f ON fi.file_id = f.id
WHERE fi.severity = 'CRITICAL'
  AND (f.relative_path LIKE '%agentdb%' OR f.relative_path LIKE '%mcp%');

-- AgentDB dependency graph (195 edges)
SELECT sf.relative_path as src, tf.relative_path as tgt, d.relationship
FROM dependencies d
JOIN files sf ON d.source_file_id = sf.id
JOIN files tf ON d.target_file_id = tf.id
WHERE sf.relative_path LIKE '%agentdb%' OR tf.relative_path LIKE '%agentdb%';

-- Package connectivity
SELECT * FROM package_connectivity WHERE package_name = 'agentdb';
-- Result: 12 outbound, 17 inbound, 29 total cross-deps, CONNECTED
```

---

## Cross-References

| ADR | Related Project ADRs |
|-----|---------------------|
| MCP-001 | ADR-006 (Unified Memory Service), ADR-008 (3-Tier Model Routing) |
| MCP-002 | ADR-009 (Hybrid Memory Backend), R20 Root Cause |
| MCP-003 | ADR-006 (Unified Memory), ADR-009 (Hybrid Backend) |
| MCP-004 | R91 (Attention Fabrication), R20 (Embedding Root Cause) |
