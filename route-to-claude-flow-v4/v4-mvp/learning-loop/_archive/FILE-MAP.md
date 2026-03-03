# File Map: Exact Sources for v4 Learning Loop MVP

> Every file needed to close the feedback loop, with exact paths and action required.

---

## Source: ~/repos/agentic-flow/agentic-flow/src/reasoningbank/

The ReasoningBank pipeline. This is the core of the MVP.

### Core Pipeline (COPY or FIX)

| File | LOC | Action | Fix Description |
|------|-----|--------|----------------|
| `core/retrieve.ts` | 121 | COPY | 4-factor scoring + MMR. Works as-is. |
| `core/judge.ts` | 177 | FIX | (1) Make ModelRouter import dynamic with catch — static import crashes if router.ts absent. (2) Add exitCode check to heuristic — otherwise ALL tasks judged as Failure. |
| `core/distill.ts` | 229 | FIX | (1) **Fix import**: uses `../utils/embeddings.js` (hash) — change to `../utils/embeddings` (real). Without this, stored memories get hash vectors while retrieval uses real vectors. (2) Make `templateBasedDistill` async, call `storeMemories()` instead of returning []. |
| `core/consolidate.ts` | 258 | COPY | LSH dedup + contradiction detection + pruning. Works as-is. |

### Hooks (ADAPT)

| File | LOC | Action | Adaptation |
|------|-----|--------|-----------|
| `hooks/pre-task.ts` | 79 | ADAPT | Change from standalone CLI to importable function. Remove `process.exit()`. |
| `hooks/post-task.ts` | 127 | ADAPT | Same — make importable. Remove `process.exit()`. Add trajectory parameter. **Add explicit `db.storeTrajectory()` after judgeTrajectory() — judge.ts does NOT persist verdicts.** Pass real `agentId` and `domain` (not "unknown"/undefined). |

### Database (COPY or FIX)

| File | LOC | Action | Notes |
|------|-----|--------|-------|
| `db/queries.ts` | 441 | FIX | Complete CRUD. **Fix 3 UPSERTs**: change `INSERT OR REPLACE` to `INSERT ... ON CONFLICT DO UPDATE` to preserve `created_at`, `usage_count`, `last_used`. **Add `LIMIT 1000`** to `fetchMemoryCandidates`. **Add BLOB alignment guard**: `if (buf.length % 4 !== 0) continue`. |
| `db/schema.ts` | 89 | COPY | TypeScript types matching SQLite schema. |
| `migrations/000_base_schema.sql` | 49 | VERIFY | **NOT_TOUCHED.** May define schema that differs from `queries.ts:runMigrations()` inline DDL. Read before Phase 0 to confirm. |
| `migrations/001_reasoningbank_schema.sql` | 169 | VERIFY | **NOT_TOUCHED.** Same concern — inline vs file-based migration may diverge. **IMPORTANT**: `vector-migration-job.ts` queries a `vector_index_meta` table for `embedding_model` and `dimension` columns. If inline DDL in queries.ts does not create `vector_index_meta`, the migration job (needed for hash→real embedding re-indexing) cannot run. This is a functional dependency, not just a schema difference. |

### Types (COPY — PREVIOUSLY MISSING)

| File | LOC | Action | Notes |
|------|-----|--------|-------|
| `types/index.ts` | 134 | COPY | **Was not in prior FILE-MAP.** Imported by consolidate.ts, distill.ts, embeddings.ts, queries.js. Without it, 5 core files fail to load. |

### Utilities (COPY or FIX)

| File | LOC | Action | Notes |
|------|-----|--------|-------|
| `utils/embeddings.ts` | 211 | COPY | **Use .ts NOT .js.** The TS version already has `@xenova/transformers` with all-MiniLM-L6-v2 (384d). Remove/bypass NPX detection (L37-43) or set `FORCE_TRANSFORMERS=1`. **Both `retrieve.ts` and `distill.ts` must import this file** (distill.ts currently imports `.js` — Fix 9). |
| `utils/mmr.js` | ~60 | COPY | `mmrSelection()` + `cosineSimilarity()`. Pure math. |
| `utils/pii-scrubber.js` | ~50 | COPY | `scrubMemory()` + `containsPII()`. |
| `utils/config.js` | ~80 | ADAPT | Point `loadConfig()` to v4 config location. **Requires `yaml` npm package.** |

### Config & Prompts (COPY or FIX)

| File | LOC | Action | Notes |
|------|-----|--------|-------|
| `config/reasoningbank-types.ts` | ~100 | COPY | |
| `config/reasoningbank.yaml` | 146 | FIX | **PREVIOUSLY MISSING + NEVER READ (NOT_TOUCHED in DB).** Must set `embeddings.dims: 384` (DEFAULT_CONFIG hardcodes 1024 → dimension mismatch). Set `embeddings.provider: 'transformers'`. Read this file before copying to verify actual content. |
| `prompts/judge.json` | ~30 | COPY | |
| `prompts/distill-success.json` | ~30 | COPY | |
| `prompts/distill-failure.json` | ~30 | COPY | |

### Dependencies (from agentic-flow's package.json)

| Package | Why |
|---------|-----|
| `better-sqlite3` | SQLite driver for queries.ts |
| `ulid` | Memory ID generation in distill.ts + consolidate.ts |
| `yaml` | **PREVIOUSLY MISSING.** Required by config.js/ts to parse reasoningbank.yaml |

---

## Source: ~/repos/claude-flow/v3/@claude-flow/cli/

### SONA Optimizer (COPY)

| File | LOC | Action | Notes |
|------|-----|--------|-------|
| `src/memory/sona-optimizer.ts` | 842 | COPY | Bayesian agent routing. Self-contained, persists to `.swarm/sona-patterns.json`. |

### DO NOT COPY from claude-flow

| File | Why |
|------|-----|
| `src/memory/intelligence.ts` | O(n) facade claiming O(log n). 14+ consumers. BIGGEST active facade. |
| `.claude/helpers/hook-handler.cjs` | The stub we're replacing. |
| `src/commands/hooks.ts` | 4,530 LOC, 30 subcommands. Too large. Extract only what's needed. |
| `src/router/router.ts` (via agentic-flow) | 420 LOC, **completely unaudited** (NOT_TOUCHED in DB). judge.ts static-imports it, but use dynamic import with catch instead. |

### DO NOT COPY from reasoningbank

| File | Why |
|------|-----|
| `utils/embeddings.js` | 114 LOC. Superseded by `embeddings.ts` (211 LOC) which already has @xenova/transformers. |
| `core/distill.js` | JS version regresses vs TS — hardcoded Anthropic API instead of ModelRouter. |
| `core/retrieve.js` | JS duplicate of retrieve.ts with hash-based embedding import. |
| `core/judge.js` | JS version uses direct Anthropic API, not ModelRouter. |

---

## Source: ~/repos/ruvector/

### Real Embeddings (REFERENCE)

| File | LOC | Action | Notes |
|------|-----|--------|-------|
| `npm/packages/ruvector/src/core/onnx-embedder.ts` | ~400 | REFERENCE | If using ONNX/Tract path instead of @xenova/transformers. Not needed if using @xenova directly. |

---

## Source: ~/repos/agentic-flow/agentic-flow/src/router/

### Model Router (OPTIONAL)

| File | LOC | Action | Notes |
|------|-----|--------|-------|
| `router/router.ts` | 420 | OPTIONAL | Multi-provider LLM router (OpenRouter / Anthropic / Gemini). Only needed if using LLM-based judge + distill. **NOT_TOUCHED in DB — zero findings, zero reads. 420 LOC with unknown dependency chain.** Heuristic fallbacks work without it. |

---

## Reference Files (Not Copied, But Relevant for v4.1+)

Files discovered in domain research that the learning loop could benefit from but are not needed for MVP. Listed here so implementors know they exist.

| File | LOC | Realness | Why It Matters | Source |
|------|-----|----------|----------------|--------|
| `agentdb/HybridSearch.ts` | ~300 | 95% | BM25 + HNSW vector fusion with 3 strategies (RRF, Linear, Max). Best search code in the ecosystem. Could replace brute-force `fetchMemoryCandidates` in v4.1. | 03-agentdb extraction |
| `agentdb-integration/infrastructure/jobs/vector-migration-job.ts` | 203 | 85-90% | Batch re-embedding job for stale vectors. Queries `vector_index_meta` for dimension/model mismatches. **Needed if existing memories must be migrated from hash (1024-dim) to real (384-dim) embeddings.** | 11-combined extraction |
| `agentic-flow SDK hooks-bridge.js` | 235 | ~95% | Bridges agentic-flow intelligence with Claude Agent SDK native hook lifecycle. 7 hook types. Already captures trajectory start/stop events. Could be adapted instead of writing `trajectory-capture.ts` from scratch. | 09-hook-pipeline extraction |
| `ruv-swarm/sqlite-pool.js` | 587 | 92% | Production SQLite connection pool with WAL mode, worker threads, health monitoring, auto-recovery, prepared statement caching. Could wrap the ReasoningBank DB for production use. | 04-swarm extraction |
| `src/cli/lib/health-monitor.ts` | 514 | 99% | Linear regression memory leak detection, 4-tier self-healing escalation, EventEmitter coordination. Best monitoring code in the ecosystem. No loop monitoring exists in the MVP. | 05-production-infra extraction |
| `agentdb/ExplainableRecall.ts` | ~250 | 88% | Provenance tracking (Merkle tree) + greedy set-cover diversification for retrieved memories. Could improve judge step quality by explaining why a memory was retrieved. | 03-agentdb extraction |
| `ruv-swarm/daa-cognition.js` | 977 | 88% | Byzantine-tolerant consensus with weighted voting, distributed learning with pattern extraction (occurrence > 0.7, diversity > 0.5). Its pattern extraction is a model for the distillation step. | 04-swarm extraction |
| `reasoningbank/crates/reasoningbank-learning/` | 788 | 95-98% | **GENUINE Rust crate.** AdaptiveLearner, StrategyOptimizer, async_learner_v2. 7/7 tests pass. Highest quality learning code in the project. Listed in GENUINE-ASSETS.md but excluded from this file map. Relevant for v4.1 Rust integration. | 06-agentic-flow extraction |

---

## New Files to Create

| File | Est. LOC | Purpose |
|------|----------|---------|
| `v4/hooks/handler.ts` | ~150 | v4 hook handler: pre-task retrieval + post-task pipeline + SONA update |
| `v4/bootstrap.ts` | ~50 | Initialize DB (runMigrations) + embedding model (lazy load) |
| `v4/trajectory-capture.ts` | ~100 | Wrap execution to capture structured trajectory data |
| `v4/config/default.yaml` | ~40 | Default ReasoningBank config (retrieve k, distill temps, consolidation thresholds) |
| `package.json` | ~20 | Deps: better-sqlite3, ulid, @xenova/transformers |

---

## NPM Dependencies

### Required

| Package | Version | Size | Purpose |
|---------|---------|------|---------|
| `better-sqlite3` | ^11.x | ~8MB native | SQLite driver (patterns, embeddings, trajectories). **Note: `agentic-flow/package.json` (212 LOC) was never read — verify actual dep versions before Phase 0.** |
| `ulid` | ^2.x | ~10KB | Sortable unique IDs for memories |
| `@xenova/transformers` | ^2.x | ~2MB (+ ~80MB model on first use) | Real local embeddings (all-MiniLM-L6-v2, 384-dim) |
| `yaml` | ^2.8.x | ~100KB | **PREVIOUSLY MISSING.** YAML parser for config.js/ts |

### Optional (for LLM-based judge/distill)

| Package | Version | Purpose |
|---------|---------|---------|
| `openai` | ^4.x | OpenRouter / OpenAI API for LLM judge |
| `@anthropic-ai/sdk` | ^0.x | Anthropic API for LLM judge |

Without these, the pipeline uses heuristic judge + template distill. Functional but lower quality.

---

## Directory Structure (Target)

```
claude-flow-v4/
├── package.json
├── tsconfig.json
├── bootstrap.ts              # DB init + embedding model warm-up
├── hooks/
│   └── handler.ts            # v4 hook handler (replaces stub)
├── trajectory/
│   └── capture.ts            # Structured trajectory from execution
├── reasoningbank/
│   ├── core/
│   │   ├── retrieve.ts       # 4-factor + MMR (COPIED)
│   │   ├── judge.ts          # LLM/heuristic judge (FIXED: dynamic import, exitCode)
│   │   ├── distill.ts        # Knowledge extraction (FIXED: template storage)
│   │   └── consolidate.ts    # LSH dedup + prune (COPIED)
│   ├── db/
│   │   ├── queries.ts        # SQLite CRUD (FIXED: UPSERTs, LIMIT, BLOB guard)
│   │   └── schema.ts         # Types (COPIED)
│   ├── types/
│   │   └── index.ts          # Shared types (PREVIOUSLY MISSING — required by 5 files)
│   ├── utils/
│   │   ├── embeddings.ts     # Real embeddings (COPIED — already has @xenova/transformers)
│   │   ├── mmr.ts            # MMR + cosine sim (COPIED)
│   │   ├── pii-scrubber.ts   # PII detection (COPIED)
│   │   └── config.ts         # Config loader (ADAPTED — requires `yaml` npm pkg)
│   ├── config/
│   │   ├── reasoningbank-types.ts  # Config type defs (COPIED)
│   │   └── reasoningbank.yaml      # PREVIOUSLY MISSING — dims: 384
│   └── prompts/
│       ├── judge.json        # Judge prompt (COPIED)
│       ├── distill-success.json
│       └── distill-failure.json
├── routing/
│   └── sona-optimizer.ts     # Bayesian routing (COPIED)
└── config/
    └── default.yaml          # Default configuration
```

---

## Validation Checklist

After setup, verify each component independently:

```bash
# 1. DB initializes
node -e "require('./reasoningbank/db/queries').runMigrations(); console.log('OK')"

# 2. Real embeddings work (NOT hash fallback)
FORCE_TRANSFORMERS=1 node -e "
const { computeEmbedding } = require('./reasoningbank/utils/embeddings');
(async () => {
  const v1 = await computeEmbedding('user authentication login');
  const v2 = await computeEmbedding('auth session management');
  const v3 = await computeEmbedding('banana smoothie recipe');
  console.log('dims:', v1.length);         // Should be 384, NOT 1024
  const cos = (a, b) => {
    let d = 0, n1 = 0, n2 = 0;
    for (let i = 0; i < a.length; i++) { d += a[i]*b[i]; n1 += a[i]*a[i]; n2 += b[i]*b[i]; }
    return d / (Math.sqrt(n1) * Math.sqrt(n2));
  };
  console.log('auth-auth:', cos(v1, v2));  // Should be > 0.6
  console.log('auth-banana:', cos(v1, v3)); // Should be < 0.3
})()
"

# 3. Store + retrieve cycle works (check UPSERT preserves metadata)
node -e "
const db = require('./reasoningbank/db/queries');
db.runMigrations();
// Store a test memory, upsert same ID twice, verify usage_count increments
"

# 4. judge.ts loads without crashing (ModelRouter import is non-fatal)
node -e "const { judgeTrajectory } = require('./reasoningbank/core/judge'); console.log('OK')"

# 5. Heuristic judge recognizes exitCode=0 as success
node -e "
const { judgeTrajectory } = require('./reasoningbank/core/judge');
(async () => {
  const v = await judgeTrajectory({steps: [{action:'execute', exitCode:0, outputLength:1000}]}, 'test');
  console.log(v.label);      // Should be 'Success', NOT 'Failure'
  console.log(v.confidence);  // Should be > 0.5
})()
"

# 6. Full pipeline: judge → distill → consolidate
node -e "
const { judgeTrajectory } = require('./reasoningbank/core/judge');
const { distillMemories } = require('./reasoningbank/core/distill');
// ... exercise with test trajectory, verify patterns table has new rows
"

# 7. Verify distill uses SAME embedding path as retrieve (Fix 9)
# After distill stores a memory, check embedding dimensions match retrieve's:
node -e "
const db = require('better-sqlite3')('.swarm/memory.db');
const rows = db.prepare('SELECT dims FROM pattern_embeddings LIMIT 5').all();
console.log('Stored dims:', rows.map(r => r.dims));
// ALL should be 384 (real). If any are 1024, distill is still using hash path.
"
```
