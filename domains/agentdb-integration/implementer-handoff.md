# AgentDB Integration — Implementer Handoff

> For the team that picks up implementation after research phase.
> Cross-reference: `analysis.md` (same directory), R20/R65/R84/R88 session findings.

## The One Fix That Unblocks Everything

**Problem** (R20, confirmed across R65/R84/R88): `EmbeddingService` in the claude-flow bridge defaults to hash-based SHA-256 embeddings — deterministic but semantically meaningless. Every downstream consumer (hybrid search, vector migration, skill consolidation, reflexion retrieval) receives garbage vectors and silently degrades.

**Fix**:
1. `npm install @xenova/transformers` as a **real** dependency (not optional)
2. In `real-embedding-adapter.ts`: remove the `catch` fallback to `EmbeddingServiceAdapter` (the hash generator). Fail loudly instead.
3. Delete `embedding-adapter.ts` entirely — it is the hash generator, confirmed as R20 root cause.

The `RealEmbeddingAdapter` already wraps MiniLM-L6-v2 correctly. It just falls back to hash when the package isn't installed. That's the entire bug.

**Files**:
- Kill: `src/agentdb-integration/infrastructure/adapters/embedding-adapter.ts` (170 LOC)
- Fix: `src/agentdb-integration/infrastructure/adapters/real-embedding-adapter.ts` (153 LOC)
- Verify: `src/agentdb-integration/infrastructure/factories.ts` — `createEmbeddingAdapter()` still returns hash adapter as primary

**After fixing**: Tier 2 components (hybrid search, search pipeline, ruvector backend, vector migration) start working with real embeddings. No other code changes needed.

## Salvage Tiers from claude-flow-self-implemented

Repo: `/home/snoozyy/claude-flow-self-implemented/`
Assessed: 2026-02-17 (55 src files, ~8,700 LOC)

### TIER 1 — Keep as-is (~2,270 LOC, standalone, zero upstream coupling)

| File | LOC | Value |
|------|-----|-------|
| `search/services/bm25-index.ts` | 269 | Textbook BM25, inverted index, stopwords. Self-contained. |
| `security/input-validator.ts` | 270 | Zod + manual fallback, HTML sanitize, vector NaN checks. |
| `events/event-bus.ts` | 138 | Error-isolated async handlers, unsubscribe pattern. |
| `events/domain-events.ts` | 217 | Discriminated union event types. Standard DDD. |
| `infrastructure/schema/schema-migrator.ts` | 233 | WAL mode, idempotent, versioned migrations (1.0→1.2). |
| `search/adapters/mmr-adapter.ts` | 141 | Clean ACL for AgentDB MMRDiversityRanker. |
| `types/*.ts` (5 files) | ~700 | AgentDBConfig, Episode, Skill, SearchRequest interfaces. |
| Value objects (5 files) | ~300 | DiversityConfig, Critique, Reward, SkillSignature, SuccessRate. |

### TIER 2 — Good code, blocked by R20 fix (~1,383 LOC)

| File | LOC | Status |
|------|-----|--------|
| `search/services/hybrid-search-service.ts` | 366 | RRF/linear/max fusion correct. Needs real embeddings. |
| `search/aggregates/search-pipeline.ts` | 287 | Timeout, MMR re-ranking, event emission. Needs hybrid search working. |
| `infrastructure/adapters/ruvector-backend-adapter.ts` | 374 | 92-95% production-ready (Semaphore, path traversal, dimension probe). |
| `infrastructure/jobs/vector-migration-job.ts` | 203 | Correct batch migration. Currently migrates hash vectors. |
| `infrastructure/adapters/real-embedding-adapter.ts` | 153 | Correct MiniLM-L6-v2 wrapper. THE file to fix. |

### TIER 3 — Discard code, keep as design reference (~2,394 LOC)

| File | LOC | Issue |
|------|-----|-------|
| `episodic/services/reflexion-service.ts` | 330 | Pass-through orchestrator, no reflexion logic |
| `episodic/adapters/reflexion-memory-adapter.ts` | 328 | Hardcoded absolute paths, designed-to-fail-gracefully |
| `skill/services/skill-library-service.ts` | 483 | Untouched, likely same pass-through pattern |
| `skill/services/consolidation-service.ts` | 273 | Consolidates hash vectors = meaningless |
| `bootstrap.ts` | 435 | Well-structured composition root, wires broken components |
| Other skill/episodic repos/adapters | ~545 | SQLite CRUD, functional but low standalone value |

### TIER 4 — Delete (~1,360 LOC)

| File | LOC | Why |
|------|-----|-----|
| `patches/cli-dist-src/*` | ~700 | Patches compiled JS, breaks on any npm update |
| `infrastructure/adapters/vector-backend-adapter.ts` | 230 | Orphaned stub, confirmed replaced by ruvector-backend-adapter |
| `infrastructure/adapters/embedding-adapter.ts` | 170 | THE hash generator. R20 root cause. Kill it. |
| barrel index.ts files, agent YAMLs | ~260 | Trivially regenerated |

### Tests — Keep Tier 1+2 tests only

Worth keeping: `test-bm25-index.ts`, `test-input-validation.ts`, `test-event-bus.ts`, `test-schema-migrator.ts`, `test-ruvector-backend.ts`, `test-ruvector-security.ts`, `test-hybrid-search.ts`

Discard: integration/bootstrap tests (test broken wiring)

## Key Context for Implementers

- **3,650 LOC** (~42% of src) is production-quality
- The DDD architecture (bounded contexts, event bus, ACLs) is sound
- The upstream `agentic-flow@2.0.6` npm package has the same defects the research catalogued across 90 sessions and 14,633 files
- AgentDB's **native** CLI/MCP server (`agentdb-cli.js`, `agentdb-mcp-server.js`) correctly initializes EmbeddingService — the bug is only in the claude-flow bridge path
- The `patches/` approach (patching compiled dist/) is a dead end. Either fix upstream or bypass it entirely.
