# ADR-MCP-003: Persistence Layer Unification — From 13 Islands to One Archipelago

**Status**: PROPOSED
**Date**: 2026-02-21
**Related**: ADR-MCP-001, ADR-MCP-002, ADR-006 (Unified Memory), ADR-009 (Hybrid Backend)

---

## 1. Context

### The 13 Disconnected Persistence Layers

The ruvnet codebase contains **13 independently-operating persistence mechanisms** discovered across 114 research sessions. Each stores data in its own format, at its own path, with its own schema. No reconciliation, no event sourcing, no change data capture exists between them.

| # | Layer | Storage Mechanism | Path/Database | Connected To | Session |
|---|-------|-------------------|---------------|-------------|---------|
| 1 | AgentDB SQLite | better-sqlite3 | `episode_embeddings` table | MCP #4 only | R20 |
| 2 | AgentDB HNSW | hnswlib-node file | Index file on disk | MCP #4 only | R20 |
| 3 | claude-flow memory CLI | JSON filesystem | `.claude-flow/data/memory.json` | Nothing | R84 |
| 4 | ReasoningBank JS (queries.js) | better-sqlite3 | `.swarm/memory.db` | Nothing | R73 |
| 5 | ReasoningBank hooks (post-task) | better-sqlite3 | `patterns/pattern_embeddings` | Nothing | R73 |
| 6 | agentic-flow long-running-agent | In-memory array | Heap (no persistence) | Nothing | R41 |
| 7 | worker-agent-integration | In-memory Maps | Heap (no persistence) | Nothing | R25 |
| 8 | ReasoningBank Rust | Rust SQLite | Separate workspace DB | Nothing | R78 |
| 9 | .claude/helpers/memory.js | JSON filesystem | `.claude-flow/data/memory.json` | Nothing | R84 |
| 10 | prime-radiant FileStorage | WAL-backed file | Custom binary format | Nothing | R108 |
| 11 | prime-radiant InMemoryStorage | Heap | No persistence (test) | Nothing | R108 |
| 12 | prime-radiant PostgresStorage | feature-gated Postgres | Never wired at runtime | Nothing | R107 |
| 13 | policy_store.rs | Cache-only HashMap | Heap (broken `get`) | Nothing | R114 |

### Impact

- **Data loss on restart**: Layers 6, 7, 11, 13 lose all state
- **Data fragmentation**: Same conceptual entity (e.g., "agent memory") stored in 3+ incompatible formats
- **No cross-layer queries**: Cannot ask "what did agent X learn?" across layers 1, 4, 5, 8
- **Two parallel episodic systems** (R104): `context_manager` composes only 2/5 siblings — episodes stored in one system are invisible to the other
- **HNSW index fragmentation**: 4+ independent HNSW vector indexes never share data — distinct from persistence layer problem but compounding it
- **Debugging nightmare**: Symptoms in one layer caused by missing data in another
- **Storage waste**: Same patterns duplicated across layers
- **No inter-MCP reconciliation**: None of the 7 MCP servers share persistence state — a vector stored via #4 is invisible to #1, #5, #7

### Package Connectivity Context

| Package | Cross-deps | Connectivity | Persistence Layers |
|---------|-----------|-------------|-------------------|
| agentdb | 29 | CONNECTED | #1, #2 |
| agentic-flow | 36 | CONNECTED | #4, #5, #6, #7 |
| claude-flow-cli | 6 | CONNECTED | #3, #9 |
| agentic-flow-rust | 55 | CONNECTED | #8 |
| ruvector-rust (prime-radiant) | 38 | CONNECTED | #10, #11, #12 |
| sublinear-rust | 25 | CONNECTED | #13 |

---

## 2. Decision

### 2.1 Tiered Consolidation Strategy

NOT all 13 layers should merge into one database. Instead, organize into **3 tiers** based on durability requirements:

#### Tier 1: Durable (must survive restarts)
- Episodes, skills, causal graphs, patterns, learned models
- **Target**: AgentDB SQLite (#1) + HNSW (#2) via MCP #4
- **Migrate**: Layers 3, 4, 5, 8, 9

#### Tier 2: Session-Scoped (must survive within session)
- Agent working memory, checkpoint state, performance profiles
- **Target**: In-memory with optional SQLite WAL
- **Migrate**: Layers 6, 7 → add WAL snapshotting

#### Tier 3: Structural (compile-time or config-time)
- Policy definitions, storage backends, feature flags
- **Target**: Configuration files + build-time feature gates
- **Migrate**: Layers 10, 12, 13 → proper config management

#### Special Case: Prime-Radiant (Layers 10-12)

Prime-radiant's 3-tier storage (FileStorage/InMemory/Postgres) is architecturally intentional — a clean durability ladder implementing `GraphStorage + GovernanceStorage` traits. It should NOT be force-migrated to AgentDB.

Instead:
- **Layer 10** (FileStorage): Fix the WAL commit bug, then keep as domain-specific. Add event emission for CDC.
- **Layer 11** (InMemory): Test-only. No migration needed.
- **Layer 12** (PostgresStorage): Feature-gated, never wired. Wire it when Postgres is available. This becomes prime-radiant's production backend, NOT AgentDB — the two serve different domains (sheaf-theoretic knowledge vs episodic agent memory).

The bridge point is `ruvllm_integration/` which maps between prime-radiant's `StateVector` world and ruvllm's memory types. This should be the CDC consumer, not a persistence migration target.

#### Special Case: HNSW Index Consolidation

Independent from SQLite persistence, the 4+ HNSW indexes need their own consolidation:

| HNSW Instance | Location | Dimension | Consolidation |
|---------------|----------|-----------|---------------|
| AgentDB hnswlib-node | packages/agentdb/ | 384 | **Canonical** (H1) |
| pattern_store.rs VectorDB | ruvllm/reasoning_bank/ | Caller-supplied | Bridge to canonical via MCP |
| agentic_memory.rs HnswIndex | ruvllm/context/ | Caller-supplied | Bridge to canonical via MCP |
| semantic_cache.rs HNSW | ruvllm/context/ | Caller-supplied | Bridge to canonical via MCP |
| prime-radiant brute-force | storage/file.rs | 64 (default) | Replace with ruvector-core HNSW (H3) |

### 2.2 Canonical Storage Architecture

```
                     ┌─────────────────────┐
                     │  AgentDB MCP #4     │
                     │  (Canonical Entry)   │
                     └────────┬────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
     ┌────────▼──────┐ ┌─────▼─────┐ ┌───────▼───────┐
     │  SQLite Core  │ │ HNSW Index│ │ Event Journal  │
     │  (episodes,   │ │ (vectors, │ │ (CDC stream    │
     │   skills,     │ │  cosine   │ │  for external  │
     │   patterns,   │ │  search)  │ │  consumers)    │
     │   causal)     │ │           │ │                │
     └───────────────┘ └───────────┘ └────────────────┘
```

### 2.3 Migration Approach Per Layer

| Layer | Current | Migration Strategy | Priority |
|-------|---------|-------------------|----------|
| #3 (memory CLI JSON) | `.claude-flow/data/memory.json` | Route through AgentDB MCP `pattern_store` tool | HIGH |
| #4 (RB JS queries) | `.swarm/memory.db` | Migrate tables to AgentDB schema, use shared DB | HIGH |
| #5 (RB hooks) | `patterns/` tables | Merge with #4, then migrate to AgentDB | HIGH |
| #6 (long-running) | In-memory array | Add WAL checkpoint on interval | MEDIUM |
| #7 (worker profiles) | In-memory Maps | Add WAL checkpoint, merge with #6 | MEDIUM |
| #8 (RB Rust) | Separate workspace | Bridge via MCP tool calls to #4 | LOW |
| #9 (helpers/memory) | Duplicate of #3 | Delete, redirect to #3 path | HIGH |
| #10 (prime-radiant file) | WAL binary | Keep as domain-specific, add event emit | LOW |
| #11 (prime-radiant mem) | Test-only | No migration needed | NONE |
| #12 (prime-radiant pg) | Never wired | Wire if Postgres available, else skip | LOW |
| #13 (policy_store) | Broken cache | Fix `get` to read from backing store | MEDIUM |

---

## 3. Event Sourcing Design

### 3.1 Event Schema

All mutations across consolidated layers emit events to the Event Journal:

```typescript
interface PersistenceEvent {
  id: string;                    // ULID
  timestamp: number;             // Unix ms
  source_layer: string;          // Original layer identifier
  aggregate_type: 'episode' | 'skill' | 'pattern' | 'causal_edge' | 'policy';
  aggregate_id: string;          // Entity ID
  event_type: 'created' | 'updated' | 'deleted' | 'migrated';
  payload: Record<string, any>;  // Event-specific data
  embedding_status: 'real' | 'hash' | 'pending' | 'none';
}
```

### 3.2 CDC Stream

External consumers (Rust subsystems, monitoring, debugging) subscribe to the Event Journal:

```typescript
interface CDCSubscription {
  subscribe(filter: {
    aggregate_type?: string[];
    source_layer?: string[];
    event_type?: string[];
  }): AsyncIterator<PersistenceEvent>;
}
```

This enables Rust subsystems (#8, #10) to stay synchronized without direct SQLite access.

---

## 4. Specific Bug Fixes Required

### 4.1 policy_store.rs `get` (Layer #13)

```
Finding R114: get() reads from cache HashMap only.
If key was never set in current session, returns None
even if it exists in the backing PolicyBundle.
Ghost entries: set() writes to cache AND bundle,
but get() only reads cache.
```

**Fix**: `get()` must fall through to PolicyBundle on cache miss.

### 4.2 prime-radiant WAL Commit Bug (Layer #10)

```
Finding R108 (CRITICAL): storage/file.rs WAL commit flag
makes deletions non-durable. Delete operations are logged
but the commit flag is not set, so on recovery, deletes
are replayed as no-ops.
```

**Fix**: Set WAL commit flag for delete operations.

### 4.3 ReasoningBank Judge Persistence (Layer #4)

```
Finding R73: judge.ts calls db.logMetric() for rb.judge
metrics but NEVER calls db.store() to persist verdict.
Judgment results lost.
```

**Fix**: Add `db.store()` call after judgment completes.

### 4.4 Auto-Prune Data Loss (Integration Layer)

```
Finding R65: reflexion-service.ts auto-prune at 80%
threshold with zero persistence check. If AgentDB store
fails, episodes only in EpisodeRepository. Prune = permanent
data loss for failed stores.
```

**Fix**: Check AgentDB store success before allowing prune.

---

## 5. Consequences

### Positive
- Single query surface for "what has agent X learned?"
- Data survives restarts (Tier 1 guarantee)
- CDC enables loose coupling between TS and Rust
- Debugging: single place to inspect all persistence

### Negative
- Migration risk: data format differences between layers
- Increased SQLite contention under concurrent writes
- Event Journal storage grows unbounded without compaction
- Some layers (#10, #12) have legitimate domain-specific needs

### Risks
- Migration corrupts existing data if schema mapping is wrong
- CDC latency adds overhead to write path
- Rust subsystems expecting local SQLite may break with remote MCP
- ReasoningBank JS/Rust divergence makes unified schema difficult

---

## 6. Validation Criteria

| Criterion | Test |
|-----------|------|
| Layer #3, #9 merged | `ls .claude-flow/data/` shows single file |
| Layer #4, #5 merged | Single DB with both trajectories + patterns tables |
| Layer #6 survives restart | Kill process, restart, verify checkpoint restored |
| Layer #13 `get` fixed | `set("key", val)` then cold restart, `get("key")` returns val |
| CDC events emitted | Subscribe to journal, insert episode, receive event |
| No data loss on prune | Store 100 episodes, fail AgentDB, prune → episodes still in repository |
