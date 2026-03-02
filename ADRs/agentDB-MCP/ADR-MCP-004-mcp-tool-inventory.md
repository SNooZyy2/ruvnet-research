# ADR-MCP-004: AgentDB MCP Tool Inventory & Capability Matrix

**Status**: PROPOSED
**Date**: 2026-02-21
**Related**: ADR-MCP-001, DDD Bounded Contexts

---

## 1. Complete Tool Inventory: AgentDB Native MCP Server (#4)

### 1.1 Core Vector DB (5 tools)

| Tool | Input | Output | Status | Quality |
|------|-------|--------|--------|---------|
| `agentdb_init` | config (dim, metric, backend) | success/error | WORKING | 85% |
| `agentdb_insert` | id, vector, metadata | success/error | WORKING | 88% |
| `agentdb_insert_batch` | vectors[], metadata[] | success/error, count | WORKING | 85% |
| `agentdb_search` | query_vector, k, filter | results[] with scores | WORKING | 88% |
| `agentdb_delete` | id | success/error | WORKING | 85% |

**Dependency**: `EmbeddingService` MUST be initialized. Without it, `agentdb_search` returns semantically meaningless results.

### 1.2 Frontier Memory (8 tools)

| Tool | Input | Output | Status | Quality |
|------|-------|--------|--------|---------|
| `reflexion_store` | task, output, reward, critique | episode_id | WORKING | 80% |
| `reflexion_retrieve` | query, k, minReward? | episodes[] | WORKING | 80% |
| `skill_create` | name, description, code | skill_id | WORKING | 78% |
| `skill_search` | query, k | skills[] | WORKING | 75% |
| `causal_add_edge` | source_id, target_id, relation | edge_id | WORKING | 78% |
| `causal_query` | node_id, direction, depth | subgraph | WORKING | 75% |
| `recall_with_certificate` | query, k | results[] + provenance | WORKING | 80% |
| `learner_discover` | context | discoveries[] | PARTIAL | 68% |
| `db_stats` | — | counts, sizes, health | WORKING | 85% |

### 1.3 Learning System (10 tools)

| Tool | Input | Output | Status | Quality |
|------|-------|--------|--------|---------|
| `learning_start_session` | algorithm, config | session_id | WORKING | 75% |
| `learning_end_session` | session_id | summary | WORKING | 75% |
| `learning_predict` | state, session_id | action, confidence | WORKING | 72% |
| `learning_feedback` | session_id, reward | ack | WORKING | 72% |
| `learning_train` | session_id, episodes | metrics | WORKING | 70% |
| `learning_metrics` | session_id | loss, reward_avg, etc. | WORKING | 68% |
| `learning_transfer` | source_session, target | ack | PARTIAL | 60% |
| `learning_explain` | session_id, decision_id | explanation | PARTIAL | 55% |
| `experience_record` | state, action, reward, next | ack | WORKING | 75% |
| `reward_signal` | session_id, signal | ack | WORKING | 72% |

**9 RL Algorithms Supported**: Q-learning, SARSA, DQN, Policy Gradient, Actor-Critic, PPO, Decision Transformer, MCTS, Model-Based.

**Note**: Learning transfer (`learning_transfer`) and explanation (`learning_explain`) are partially implemented — transfer copies weights but doesn't adapt, explain returns template strings.

### 1.4 AgentDB Core (5 tools)

| Tool | Input | Output | Status | Quality |
|------|-------|--------|--------|---------|
| `agentdb_stats` | — | comprehensive stats | WORKING | 85% |
| `pattern_store` | key, value, embedding | ack | WORKING | 80% |
| `pattern_search` | query, k, threshold | patterns[] | WORKING | 78% |
| `pattern_stats` | — | pattern counts, sizes | WORKING | 82% |
| `clear_cache` | scope? | ack | WORKING | 85% |

### 1.5 Batch Operations (3 tools)

| Tool | Input | Output | Status | Quality |
|------|-------|--------|--------|---------|
| `skill_batch` | skills[] | results[], failures[] | WORKING | 78% |
| `reflexion_batch` | episodes[] | results[], failures[] | WORKING | 78% |
| `pattern_batch` | patterns[] | results[], failures[] | WORKING | 78% |

All batch operations use SQLite transactions for atomicity.

### 1.6 Attention (4 tools) — FABRICATED

| Tool | Input | Output | Status | Quality |
|------|-------|--------|--------|---------|
| `attention_compute` | query, keys, mechanism | scores | **FABRICATED** | 15% |
| `attention_benchmark` | config | metrics | **FABRICATED** | 10% |
| `attention_configure` | settings | ack | **EPHEMERAL** | 20% |
| `attention_metrics` | — | stats | **FABRICATED** | 10% |

**WARNING**: ALL attention tools return fabricated data:
- `attention_compute`: flash/linear/performer all compute identical dot product
- `attention_benchmark`: generates random test data
- `attention_metrics`: `totalCalls = Math.floor(Math.random()*10000) + 1000`
- `attention_configure`: merges config but never persists — next call returns defaults
- Sparse attention uses `Math.random() > 0.9` — random dropout, not real sparsity
- Poincare distance produces `Infinity` for normalized vectors

**Recommendation**: Remove attention tools entirely or replace with honest stubs returning `{ status: "not_implemented" }`.

---

## 2. Near-Miss 8th MCP: Psycho-Symbolic Reasoner

`crates/psycho-symbolic-reasoner/mcp-integration/src/index.ts` (R80) uses the **same official @modelcontextprotocol/sdk** as #4, with 16 tools. It is NOT a wrapper — it has its own tool implementations.

| Tool Group | Count | Status |
|-----------|-------|--------|
| Reasoning tools | ~8 | Genuine symbolic reasoning |
| Integration tools | ~4 | Bridge to main reasoner |
| Query tools | ~4 | Graph/pattern queries |

This near-miss 8th server would be a **5th functional island** if wired. Its tools are not counted in the #4 inventory above but should be evaluated for H2 composition.

---

## 3. Functional Islands vs Wrappers

| Category | Servers | Total Tools | Value |
|----------|---------|-------------|-------|
| **Functional Islands** | #1 (256), #4 (27-34), #5 (11), #7 (3) | ~301 | Each does real, independent work |
| **Near-Miss Island** | ~#8 (16) | ~16 | Would be 5th island if wired |
| **Wrappers/Facades** | #2 (7), #3 (11), #6 (4+2) | ~24 | Zero added value — all CLI wrappers or hand-rolled stubs |

Consolidation targets the **4 functional islands**, not all 7 servers. The 3 wrappers should be retired, not composed.

---

## 4. Tool Comparison: Native (#4) vs claude-flow Bridge (#1)

| Capability | Native MCP #4 | claude-flow #1 | Gap |
|-----------|---------------|----------------|-----|
| Core vector ops | 5 tools, real HNSW | 5 tools, mockEmbedding | **BROKEN** in #1 |
| Frontier memory | 8 tools | 2 tools (store/retrieve) | 6 tools missing |
| Learning system | 10 tools, 9 algorithms | 0 tools | **COMPLETELY MISSING** |
| Batch operations | 3 tools, transactional | 0 tools | All missing |
| Attention | 4 tools (fabricated) | 4 tools (fabricated) | Both broken |
| Pattern store | 3 tools | 3 tools (via fallback) | Degraded in #1 |
| Stats/health | 2 tools | 1 tool | Missing health check |
| **Total** | **27-34 tools** | **6-15 tools** | **12-19 tools missing** |

### What claude-flow Users Don't Get

1. **Zero RL algorithms** — the entire learning system (10 tools) is unreachable
2. **No causal graph** — `causal_add_edge` and `causal_query` not exposed
3. **No skill library** — `skill_create`, `skill_search` not exposed
4. **No batch operations** — every operation is individual
5. **No recall certificates** — provenance tracking unavailable
6. **No real embeddings** — search returns hash-based noise

---

## 3. MCP Server #4 Startup Sequence

```
1. Parse CLI args / stdin config
2. Initialize better-sqlite3 database
3. Create controllers:
   a. HNSWIndex (hnswlib-node, cosine metric)
   b. MemoryController
   c. ReflexionMemory
   d. SkillLibrary
   e. CausalMemoryGraph
   f. ReasoningBank
   g. LearningSystem
   h. ExplainableRecall
4. ★ EmbeddingService.initialize()              ← THE CRITICAL STEP
   a. Load @xenova/transformers
   b. Download/cache all-MiniLM-L6-v2 (384d)
   c. Warm up with test embedding
5. Register 27-34 MCP tools with @modelcontextprotocol/sdk
6. Start StdioServerTransport
7. Begin accepting JSON-RPC requests
```

**Step 4 is what makes #4 unique.** No other server performs this initialization.

---

## 4. Tool Dependency Graph

```
agentdb_init ──────────────────────────────────┐
                                               │
agentdb_insert ──▶ EmbeddingService ──▶ HNSW   │
agentdb_insert_batch ──▶ EmbeddingService ──▶ HNSW
agentdb_search ──▶ EmbeddingService ──▶ HNSW   │
agentdb_delete ──▶ HNSW                        │
                                               │
reflexion_store ──▶ ReflexionMemory ──▶ EmbeddingService ──▶ HNSW
reflexion_retrieve ──▶ ReflexionMemory ──▶ EmbeddingService ──▶ HNSW
                                               │
skill_create ──▶ SkillLibrary ──▶ EmbeddingService ──▶ HNSW
skill_search ──▶ SkillLibrary ──▶ EmbeddingService ──▶ HNSW
                                               │
causal_add_edge ──▶ CausalMemoryGraph          │
causal_query ──▶ CausalMemoryGraph             │
                                               │
recall_with_certificate ──▶ ExplainableRecall ──▶ EmbeddingService
                                               │
learning_* ──▶ LearningSystem                  │
                                               │
pattern_store ──▶ ReasoningBank ──▶ EmbeddingService ──▶ HNSW
pattern_search ──▶ ReasoningBank ──▶ EmbeddingService ──▶ HNSW
                                               │
attention_* ──▶ (FABRICATED — no real deps) ───┘
```

**Key insight**: 19 of 27 tools depend on `EmbeddingService`. When it's not initialized, 70% of the tool surface degrades to hash-based noise.

---

## 5. Configuration Reference

### 5.1 MCP Server Config (`.mcp.json`)

```json
{
  "mcpServers": {
    "agentdb": {
      "command": "node",
      "args": ["packages/agentdb/src/mcp/agentdb-mcp-server.ts"],
      "env": {
        "AGENTDB_DB_PATH": "./data/agentdb.sqlite",
        "AGENTDB_HNSW_PATH": "./data/hnsw.index",
        "AGENTDB_EMBEDDING_MODEL": "Xenova/all-MiniLM-L6-v2",
        "AGENTDB_EMBEDDING_DIM": "384"
      }
    }
  }
}
```

### 5.2 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENTDB_DB_PATH` | `./agentdb.sqlite` | SQLite database location |
| `AGENTDB_HNSW_PATH` | `./hnsw.index` | HNSW index file location |
| `AGENTDB_EMBEDDING_MODEL` | `Xenova/all-MiniLM-L6-v2` | Hugging Face model ID |
| `AGENTDB_EMBEDDING_DIM` | `384` | Embedding dimension |
| `AGENTDB_HNSW_M` | `16` | HNSW M parameter |
| `AGENTDB_HNSW_EF` | `200` | HNSW ef_construction |
| `AGENTDB_MAX_ELEMENTS` | `100000` | Max vectors in HNSW |
| `AGENTDB_BATCH_SIZE` | `100` | Default batch size |

---

## 6. Quality Assessment Summary

| Tool Group | Tools | Working | Fabricated | Missing from #1 |
|-----------|-------|---------|-----------|-----------------|
| Core Vector | 5 | 5 | 0 | 0 (but degraded) |
| Frontier Memory | 8 | 8 | 0 | 6 |
| Learning | 10 | 8 | 0 | 10 |
| Core/Stats | 5 | 5 | 0 | 2 |
| Batch | 3 | 3 | 0 | 3 |
| Attention | 4 | 0 | **4** | 0 (both fabricated) |
| **Total** | **35** | **29** | **4** | **21** |

**29 of 35 tools work correctly when EmbeddingService is initialized.**
**4 attention tools are entirely fabricated and should be removed or reimplemented.**
**21 tools are unreachable from the claude-flow main MCP bridge.**
