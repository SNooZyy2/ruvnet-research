# Pre-Flight Resolution: v4 MVP Learning Loop

> Generated 2026-03-03 from reading ALL source files in the pipeline.
> Resolves 4 open gaps from the SPEC before implementation begins.

---

## Gap 1: reasoningbank.yaml (RESOLVED — GOOD NEWS)

**SPEC assumed**: `dims: 1024` default, needs fixing to 384.

**Actual content** (`~/repos/agentic-flow/agentic-flow/src/reasoningbank/config/reasoningbank.yaml`):
```yaml
embeddings:
  provider: "local"
  model: "Xenova/all-MiniLM-L6-v2"
  dimensions: 384
```

**Already correct.** Provider is `"local"`, model is the right one, dimensions are 384.

**BUT** — `config.ts` has a DEFAULT_CONFIG fallback with wrong values:
```typescript
// config.ts lines 111-116
embeddings: {
  provider: 'claude',        // ← WRONG default
  model: 'claude-sonnet-4-5-20250929',  // ← WRONG default
  dims: 1024,                // ← WRONG default
  dimensions: 1024,          // ← WRONG default
  cache_ttl_seconds: 3600
}
```

If the YAML file is not found at runtime (config search paths miss it), the system silently falls back to 1024-dim hash embeddings.

**Resolution**: Ensure the v4 directory structure places `reasoningbank.yaml` at one of the search paths:
- `../config/reasoningbank.yaml` (relative to `utils/config.ts`)
- `../../config/reasoningbank.yaml` (relative to compiled `dist/`)
- `.swarm/reasoningbank.yaml` (relative to cwd)

Also: fix the DEFAULT_CONFIG in `config.ts` to use `dims: 384, provider: 'local'` as the fallback. This is a 2-line change.

**Action**: COPY yaml as-is. Fix 2 lines in `config.ts` DEFAULT_CONFIG.

---

## Gap 2: migrations/*.sql (RESOLVED — INLINE DDL IS CANONICAL)

**SPEC assumed**: Migration SQL files might define different schema from `queries.ts:runMigrations()`.

**Actual comparison**:

| Table | Inline DDL (queries.ts) | SQL Files | Difference |
|-------|------------------------|-----------|------------|
| `patterns` | ✅ | ✅ (000_base) | Identical columns |
| `pattern_embeddings` | ✅ | ✅ (001) | Identical |
| `pattern_links` | ✅ | ✅ (001) | SQL has CHECK constraints, inline doesn't |
| `task_trajectories` | ✅ | ✅ (001) | SQL has CHECK on judge_label, inline doesn't |
| `matts_runs` | ✅ | ✅ (001) | SQL has CHECK constraints |
| `consolidation_runs` | ✅ | ✅ (001) | Identical |
| `metrics_log` | ✅ | ❌ | **Inline-only** — all queries.ts functions reference this |
| `performance_metrics` | ❌ | ✅ (000_base) | **SQL-only** — different table name |
| `memory_namespaces` | ❌ | ✅ (000_base) | SQL-only (multi-tenant, not needed for MVP) |
| `session_state` | ❌ | ✅ (000_base) | SQL-only (not needed for MVP) |
| Views (3) | ❌ | ✅ (001) | Nice monitoring views, not required |
| Triggers (2) | ❌ | ✅ (001) | Auto-cleanup + last_used update |
| `vector_index_meta` | ❌ | ❌ | **Missing everywhere** — vector-migration-job.ts can't run |

**Resolution**: Use `queries.ts:runMigrations()` inline DDL — it's what the pipeline code references. The SQL files are a parallel setup path that creates different table names (`performance_metrics` vs `metrics_log`).

The 2 triggers from the SQL are genuinely useful:
- `trg_cleanup_embeddings`: auto-deletes orphaned embeddings on pattern delete
- `trg_update_last_used`: auto-updates `last_used` on usage_count change

**Action**: Use inline DDL. Optionally add the 2 triggers from `001_reasoningbank_schema.sql` after migration. Do NOT use the SQL files as primary schema source.

---

## Gap 3: Import Graph (RESOLVED — 3 CORRECTIONS TO SPEC)

### Full transitive dependency closure

```
CORE PIPELINE (12 files):
retrieve.ts ─── embeddings.ts ─── @xenova/transformers
            ├── mmr.ts            config.ts ─── yaml
            ├── queries.ts ─── better-sqlite3
            │                └── schema.ts
            └── config.ts

judge.ts ─── config.ts
         ├── ../../router/router.js ⚠️ STATIC IMPORT (CRASHES)
         ├── schema.ts (type-only)
         └── queries.ts (late import, line 177)

distill.ts ─── config.ts
           ├── pii-scrubber.ts ─── config.ts
           ├── embeddings.ts (SAME path as retrieve!)
           ├── ../../router/router.js ⚠️ STATIC IMPORT (CRASHES)
           ├── queries.ts
           ├── schema.ts (type-only)
           └── judge.ts (type-only: Verdict)

consolidate.ts ─── config.ts
               ├── mmr.ts
               └── queries.ts

HOOKS (2 files):
pre-task.ts ─── retrieve.ts, config.ts
post-task.ts ─── judge.ts, distill.ts, consolidate.ts, config.ts, schema.ts
```

### Correction 1: distill.ts ALSO has static ModelRouter import

The SPEC identifies Fix 7 as "Make ModelRouter import non-fatal in **judge.ts**". But `distill.ts` line 13 has the **same static import**:

```typescript
// distill.ts:13
import { ModelRouter } from '../../router/router.js';
```

**Fix 7 must cover BOTH files**, not just judge.ts. Same fix (dynamic import with catch).

### Correction 2: Fix 9 (embeddings import mismatch) is WRONG

The SPEC says:
> `retrieve.ts` imports from `embeddings.ts`, `distill.ts` imports from `embeddings.js`

**Both files import from the same path**: `../utils/embeddings.js`

```typescript
// retrieve.ts:6
import { computeEmbedding } from '../utils/embeddings.js';
// distill.ts:12
import { computeEmbedding } from '../utils/embeddings.js';
```

In ESM TypeScript, `.js` extension is the runtime resolution path. Both resolve to `embeddings.ts` when:
- Running via `tsx` (maps `.js` → `.ts`)
- Running compiled `dist/` (tsc compiles `embeddings.ts` → `dist/utils/embeddings.js`)

The real risk is that both `embeddings.ts` AND `embeddings.js` exist in the same directory. Node.js ESM may pick the literal `.js` file.

**Fix 9 replacement**: Don't copy `embeddings.js` to v4. Only copy `embeddings.ts`. Then `import from '../utils/embeddings.js'` can only resolve to the `.ts` file. No import path changes needed in any file.

### Correction 3: types/index.ts and config/reasoningbank-types.ts are NOT imported

The SPEC says types/index.ts is "imported by consolidate, distill, embeddings, queries". **Zero files import it.** Grep confirms: no `from.*types/index` matches in the pipeline.

Similarly, `config/reasoningbank-types.ts` is never imported — `config.ts` defines its own `ReasoningBankConfig` interface inline.

**Action**: Do NOT copy `types/index.ts` or `config/reasoningbank-types.ts`. They are dead files. The actual type source is `db/schema.ts`.

### Final file list (corrected)

**Copy** (10 files, ~2,110 LOC):
| File | LOC | Notes |
|------|-----|-------|
| `core/retrieve.ts` | 122 | As-is |
| `core/judge.ts` | 177 | Fix: dynamic ModelRouter import |
| `core/distill.ts` | 230 | Fix: dynamic ModelRouter import, async templateBasedDistill |
| `core/consolidate.ts` | 259 | As-is |
| `db/queries.ts` | 441 | Fix: 3 UPSERTs, add LIMIT |
| `db/schema.ts` | 89 | As-is |
| `utils/embeddings.ts` | 211 | As-is (already has @xenova/transformers) |
| `utils/mmr.ts` | 80 | As-is |
| `utils/pii-scrubber.ts` | 131 | As-is |
| `utils/config.ts` | 241 | Fix: DEFAULT_CONFIG dims/provider |
| `config/reasoningbank.yaml` | 146 | As-is (already correct!) |

**Copy hooks** (2 files, adapt):
| File | LOC | Notes |
|------|-----|-------|
| `hooks/pre-task.ts` | 79 | Remove process.exit(), make importable |
| `hooks/post-task.ts` | 128 | Remove process.exit(), add storeTrajectory call |

**Copy prompts** (3 files):
| File | LOC |
|------|-----|
| `prompts/judge.json` | ~120 |
| `prompts/distill-success.json` | ~120 |
| `prompts/distill-failure.json` | ~150 |

**DO NOT copy** (removed from SPEC):
| File | Why |
|------|-----|
| `types/index.ts` | Not imported by anything |
| `config/reasoningbank-types.ts` | Not imported by anything |
| `utils/embeddings.js` | Competing hash-based file. Removing it IS the fix. |

---

## Gap 4: CJS/ESM Decision (RESOLVED)

**Finding**: `agentic-flow/package.json` has `"type": "module"`. All source files use ESM `import/export`. The v4 handler must be ESM.

**SPEC options**:
- **Option A (subprocess)**: `tsx post-task.ts` — **REJECTED**. `tsx` is a devDependency, not available in production.
- **Option B (in-process ESM)**: Import pipeline directly — **SELECTED**. The v4 handler is a `.ts` file in an ESM package. All pipeline files are ESM. Direct import works.
- **Option C (hybrid CJS→ESM)**: Dynamic import from CJS — **NOT NEEDED**. No CJS in the path.

**Resolution**: v4 handler is `hooks/handler.ts` (ESM). It directly imports from the pipeline modules:
```typescript
import { retrieveMemories, formatMemoriesForPrompt } from '../core/retrieve.js';
import { judgeTrajectory } from '../core/judge.js';
import { distillMemories } from '../core/distill.js';
import { consolidate, shouldConsolidate } from '../core/consolidate.js';
```

This is the same pattern `post-task.ts` already uses. No ESM/CJS bridging needed.

**package.json for v4**:
```json
{
  "type": "module",
  "dependencies": {
    "better-sqlite3": "^11.10.0",
    "ulid": "^3.0.1",
    "@xenova/transformers": "^2.17.2",
    "yaml": "^2.8.1"
  }
}
```

Note: `ulid` is v3 (SPEC assumed v2). No API change — `ulid()` function works the same in both.

Note: `better-sqlite3` is an `optionalDependency` in agentic-flow (may not install). In v4, make it a hard dependency.

---

## Gap 5: Competing Pipelines (Assessed)

| Pipeline | Fires on | Risk to MVP | Action |
|----------|----------|-------------|--------|
| Q-learning (post-edit.js / post-command.js) | Edit/command events | **NONE** — different event type than post-task | Ignore |
| GuidanceHookProvider PreTask | Task start (hooks.js) | **LOW** — only fires if `@claude-flow/guidance` is loaded | Guard: check if guidance hooks are active, skip retrieval if so |
| intelligence-bridge.js EWC++ | Trajectory end (SDK hooks) | **NONE** — R140 confirmed LoRA/EWC config stored but never used | Ignore |
| LocalSonaCoordinator (intelligence.ts) | Loaded by 14+ V3 paths | **MEDIUM** — runs O(n) search alongside ReasoningBank retrieval | v4 handler doesn't load intelligence.ts. Not in import graph. |

**Resolution**: The MVP handler imports only from the ReasoningBank pipeline (10 files above). It does NOT import intelligence.ts, hooks.js, or any competing system. As long as the v4 handler is the sole post-task hook, no double-firing occurs.

If v4 runs INSIDE the claude-flow V3 runtime (where intelligence.ts is already loaded), add a feature flag:
```typescript
if (process.env.V4_LEARNING_LOOP === '1') {
  // Use ReasoningBank pipeline
} else {
  // Fall through to legacy intelligence.feedback(true)
}
```

---

## Updated Fix List (Post Pre-Flight)

| Fix | Original SPEC | Correction | New Effort |
|-----|---------------|------------|------------|
| Fix 1 | Wire post-task | Use Option B (ESM in-process) | Same |
| Fix 2 | Activate real embeddings | YAML already correct. Fix DEFAULT_CONFIG in config.ts (2 lines) | **Smaller** |
| Fix 3 | templateBasedDistill stores | Same | Same |
| Fix 4 | Capture trajectory | Same | Same |
| Fix 5 | Persist verdict | Same | Same |
| Fix 6 | Heuristic exitCode | Same | Same |
| Fix 7 | Dynamic ModelRouter import | Must fix in **BOTH judge.ts AND distill.ts** | **Slightly larger** |
| Fix 8 | UPSERT metadata | Same | Same |
| Fix 9 | distill imports wrong embeddings | **WRONG.** Both files import same path. Real fix: don't copy `embeddings.js` | **Simpler** |

---

## Pre-Flight Checklist (All Green)

- [x] `reasoningbank.yaml` read — already has dims:384, provider:local
- [x] `migrations/*.sql` read — inline DDL is canonical, SQL files diverge on table names
- [x] `package.json` read — ESM, better-sqlite3 optional, ulid v3, all deps confirmed
- [x] Import graph traced — 12 internal files, 2 external (ModelRouter), 2 dead files removed
- [x] CJS/ESM decided — Option B (ESM in-process)
- [x] Competing pipelines assessed — none in the import graph, guard with env var if needed
- [x] SPEC corrections documented (3 corrections: ModelRouter in distill.ts, Fix 9 is wrong, types/index.ts unused)

**Status: Ready to implement.**
