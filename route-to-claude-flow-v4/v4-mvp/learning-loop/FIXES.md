# v4 MVP: Exact Code Changes

> Implementation agent's work order. Each fix: file path, before snippet, after snippet, line numbers.
> All 3 PREFLIGHT corrections applied.
> Cross-references to [SPEC.md](SPEC.md) sections. File inventory in [FILES.md](FILES.md).

---

## Fix 1: Wire Post-Task to ReasoningBank Pipeline

**SPEC ref**: S3, S5 Phase 3
**Decision**: Option B (ESM in-process) -- selected in PREFLIGHT Gap 4

**Current** (`~/repos/claude-flow/.claude/helpers/hook-handler.cjs:140-147`):

```javascript
'post-task': () => {
    if (intelligence && intelligence.feedback) {
      try { intelligence.feedback(true); } catch (e) {}
    }
    console.log('[OK] Task completed');
}
```

Seven lines. `intelligence.feedback(true)` passes a single boolean -- no trajectory, no judgment, no distillation. The `intelligence` object is the O(n) facade (R140 CRITICAL). Its `feedback()` method updates a single counter.

**Target**: Create `hooks/handler.ts` (~150 LOC) that:

1. Pre-task: call `retrieveMemories(query)` -> `formatMemoriesForPrompt()` -> inject into prompt
2. Post-task: collect trajectory (Fix 4) -> `judgeTrajectory()` -> `db.storeTrajectory()` (Fix 5) -> `distillMemories()` -> optionally `consolidate()`
3. Update SONA optimizer with outcome

Import pattern (ESM, same as post-task.ts already uses):

```typescript
// hooks/handler.ts -> reasoningbank/core/ (per FILES.md directory structure)
import { retrieveMemories, formatMemoriesForPrompt } from '../reasoningbank/core/retrieve.js';
import { judgeTrajectory } from '../reasoningbank/core/judge.js';
import { distillMemories } from '../reasoningbank/core/distill.js';
import { consolidate, shouldConsolidate } from '../reasoningbank/core/consolidate.js';
```

---

## Fix 2: Activate Real Embeddings

**SPEC ref**: S3
**PREFLIGHT correction**: YAML is already correct. Only fix DEFAULT_CONFIG in config.ts.

`reasoningbank.yaml` already has:

```yaml
embeddings:
  provider: "local"
  model: "Xenova/all-MiniLM-L6-v2"
  dimensions: 384
```

But `config.ts` DEFAULT_CONFIG has wrong fallback values.

**File**: `~/repos/agentic-flow/agentic-flow/src/reasoningbank/utils/config.ts` lines ~111-116

**Before**:

```typescript
embeddings: {
  provider: 'claude',
  model: 'claude-sonnet-4-5-20250929',
  dims: 1024,
  dimensions: 1024,
  cache_ttl_seconds: 3600
}
```

**After**:

```typescript
embeddings: {
  provider: 'local',
  model: 'Xenova/all-MiniLM-L6-v2',
  dims: 384,
  dimensions: 384,
  cache_ttl_seconds: 3600
}
```

**Also**: Remove or bypass NPX detection block in `embeddings.ts` lines 37-43, or set `FORCE_TRANSFORMERS=1` in bootstrap. The 4 heuristics (detect npx, detect npm exec, etc.) prevent @xenova/transformers from activating in development environments.

---

## Fix 3: templateBasedDistill Stores Memories

**SPEC ref**: S3

**File**: `~/repos/agentic-flow/agentic-flow/src/reasoningbank/core/distill.ts:210-229`

Three changes to the function (body unchanged):

```typescript
// Line 210: add async
async function templateBasedDistill(
    trajectory: Trajectory, verdict: Verdict, query: string, options: any
): Promise<string[]> {                    // Line 212: add Promise<>
    // ... memory object creation unchanged ...

    const confidencePrior = verdict.label === 'Success' ? 0.6 : 0.3;
    return storeMemories([memory], confidencePrior, verdict, options);
}                                          // Line 228-229: replace `return []`
```

The caller (`distillMemories()`) already awaits the result, so no caller changes needed.

---

## Fix 4: Capture Trajectory Data

**SPEC ref**: S3, S5 Phase 3

**File**: Create `trajectory/capture.ts` (~100 LOC)

The hook handler knows: task query (from pre-task routing), agent type (from SONA), start/end timestamps, exit code, stdout length.

**Minimal trajectory** (matches `db/schema.ts`):

```typescript
function captureTrajectory(taskQuery, agentType, routedModel, startTime, endTime, exitCode, outputLength): Trajectory {
    return {
        steps: [
            { action: 'spawn', agent: agentType, query: taskQuery, timestamp: startTime },
            { action: 'execute', exitCode, outputLength, timestamp: endTime }
        ],
        metadata: { duration: endTime - startTime, agent: agentType, model: routedModel }
    };
}
```

Coarse but sufficient. Gives judge exit code (Fix 6) and basic signal. `Trajectory` type: `{ steps: Array<{action: string, [k:string]: any}>, metadata?: Record<string,any> }`. Refine in v4.1.

---

## Fix 5: Persist Verdict to DB

**SPEC ref**: S3

**File**: `~/repos/agentic-flow/agentic-flow/src/reasoningbank/hooks/post-task.ts`

judge.ts calls `db.logMetric('rb.judge', ...)` but NEVER calls `db.storeTrajectory()`. The verdict is returned but `task_trajectories.judge_label`, `judge_conf`, `judge_reasons` columns stay NULL.

**Add after judgeTrajectory() returns**:

```typescript
const verdict = await judgeTrajectory(trajectory, query);

// FIX: persist verdict (judge.ts returns it but doesn't store it)
db.storeTrajectory(
    taskId,
    agentId,                                    // NOT "unknown"
    query,
    JSON.stringify(trajectory),
    verdict.label,
    verdict.confidence,
    JSON.stringify(verdict.reasons)
);
```

**Also**: Pass real `agentId` and `domain` to `distillMemories()` (not "unknown"/undefined). The agentId comes from SONA routing in pre-task. The domain can be inferred from the task query or set to a default.

---

## Fix 6: Heuristic Judge Checks Exit Codes

**SPEC ref**: S3

**File**: `~/repos/agentic-flow/agentic-flow/src/reasoningbank/core/judge.ts:153-173`

The heuristic JSON.stringifies each step and searches for `"error"` and `"complete"`. With the trajectory from Fix 4, neither string appears -- every task is judged Failure with 0.5 confidence.

**Before** (inside the step loop):

```typescript
for (const step of steps) {
    const s = JSON.stringify(step).toLowerCase();
    if (s.includes('error')) errorCount++;
    if (s.includes('complete')) successSignals++;
}
```

**After**:

```typescript
for (const step of steps) {
    const s = JSON.stringify(step).toLowerCase();
    if (s.includes('error')) errorCount++;
    if (s.includes('complete')) successSignals++;
    // FIX: check exitCode from trajectory capture
    if (step.exitCode === 0) successSignals++;
    if (step.exitCode != null && step.exitCode !== 0) errorCount++;
}
```

Two lines added. Now `{exitCode: 0}` counts as a success signal, and `{exitCode: 1}` counts as an error.

---

## Fix 7: Dynamic ModelRouter Import

**SPEC ref**: S3
**PREFLIGHT correction**: Must fix BOTH judge.ts AND distill.ts (not just judge.ts).

### judge.ts (line 12)

**Before**:

```typescript
import { ModelRouter } from '../../router/router.js';
```

**After**:

```typescript
let ModelRouter: any = null;
try {
    ({ ModelRouter } = await import('../../router/router.js'));
} catch {
    // ModelRouter unavailable -- heuristic fallback will be used
}
```

### distill.ts (line 13)

**Before**:

```typescript
import { ModelRouter } from '../../router/router.js';
```

**After** (same pattern):

```typescript
let ModelRouter: any = null;
try {
    ({ ModelRouter } = await import('../../router/router.js'));
} catch {
    // ModelRouter unavailable -- template-based distillation will be used
}
```

**Note**: Both files must guard their LLM code paths with `if (ModelRouter)` checks. Without ModelRouter, judge.ts uses heuristic fallback and distill.ts uses template-based distillation. Both paths are functional.

**Why not copy router.ts instead**: 420 LOC, zero research findings, zero reads, unknown dependency chain. The dynamic import with catch is safer and keeps the MVP dependency surface minimal.

---

## Fix 8: UPSERT Preserves Metadata

**SPEC ref**: S3

**File**: `~/repos/agentic-flow/agentic-flow/src/reasoningbank/db/queries.ts`

Three `INSERT OR REPLACE` statements that lose `created_at`, `usage_count`, `last_used` on re-insert. This breaks the convergence mechanism -- memories that should accumulate confidence instead reset to zero.

### Statement 1: patterns

**Before**:

```sql
INSERT OR REPLACE INTO patterns (id, type, pattern_data, confidence)
VALUES (?, ?, ?, ?)
```

**After**:

```sql
INSERT INTO patterns (id, type, pattern_data, confidence)
VALUES (?, ?, ?, ?)
ON CONFLICT(id) DO UPDATE SET
  pattern_data = excluded.pattern_data,
  confidence = excluded.confidence,
  usage_count = usage_count + 1,
  last_used = datetime('now')
```

### Statement 2: task_trajectories

**Before**:

```sql
INSERT OR REPLACE INTO task_trajectories (task_id, ...) VALUES (?, ...)
```

**After**: Two changes required. First, change the CREATE TABLE in `runMigrations()` — the current DDL has `task_id TEXT PRIMARY KEY` which enforces one row per task. Change to a composite key:

```sql
CREATE TABLE IF NOT EXISTS task_trajectories (
  task_id TEXT NOT NULL,
  agent_id TEXT,
  query TEXT,
  trajectory TEXT,
  judge_label TEXT,
  judge_conf REAL,
  judge_reasons TEXT,
  created_at TEXT DEFAULT (datetime('now')),
  PRIMARY KEY (task_id, agent_id, created_at)
);
```

Then change the INSERT to match:

```sql
INSERT INTO task_trajectories (task_id, agent_id, query, trajectory, judge_label, judge_conf, judge_reasons)
VALUES (?, ?, ?, ?, ?, ?, ?)
```

No ON CONFLICT needed — the composite key with `created_at` makes each run a unique row.

### Statement 3: pattern_links

**Before**:

```sql
INSERT OR REPLACE INTO pattern_links (src_id, dst_id, relation, weight)
VALUES (?, ?, ?, ?)
```

**After**:

```sql
INSERT INTO pattern_links (src_id, dst_id, relation, weight)
VALUES (?, ?, ?, ?)
ON CONFLICT(src_id, dst_id, relation) DO UPDATE SET
  weight = excluded.weight
```

### Additional queries.ts fixes

**Add LIMIT** to `fetchMemoryCandidates` (prevents returning entire DB at scale):

```sql
SELECT ... FROM patterns p JOIN pattern_embeddings pe ON ... LIMIT 1000
```

**Add BLOB alignment guard** in Float32Array conversion:

```typescript
if (buf.length % 4 !== 0) continue;  // skip corrupted embeddings
```

### Optional: Add triggers from migration SQL

Two useful triggers from `001_reasoningbank_schema.sql` (not in inline DDL):

```sql
-- Auto-delete orphaned embeddings when pattern deleted
CREATE TRIGGER IF NOT EXISTS trg_cleanup_embeddings
AFTER DELETE ON patterns
BEGIN DELETE FROM pattern_embeddings WHERE id = OLD.id; END;

-- Auto-update last_used when usage_count changes
CREATE TRIGGER IF NOT EXISTS trg_update_last_used
AFTER UPDATE OF usage_count ON patterns
BEGIN UPDATE patterns SET last_used = datetime('now') WHERE id = NEW.id; END;
```

---

## Fix 9: Eliminate Embeddings File Mismatch

**SPEC ref**: S3
**PREFLIGHT correction**: Original fix was WRONG. The import paths are identical.

**Original claim**: retrieve.ts imports `embeddings.ts`, distill.ts imports `embeddings.js` -- mismatch.

**Reality** (PREFLIGHT Gap 3, Correction 2): Both files import `../utils/embeddings.js`. In ESM TypeScript, `.js` extension is the runtime resolution path -- `tsc` compiles `embeddings.ts` to `dist/utils/embeddings.js`, and `tsx` maps `.js` back to `.ts`.

**The real risk**: Both `embeddings.ts` (211 LOC, real semantic embeddings) AND `embeddings.js` (114 LOC, hash-based) exist in the source directory. If both are copied to v4, Node.js ESM resolution picks the literal `.js` file, silently activating hash embeddings.

**Fix**: Do NOT copy `embeddings.js` to the v4 directory. Only copy `embeddings.ts`. Then `import from '../utils/embeddings.js'` can only resolve to the compiled `.ts` file. No import path changes needed in any source file.

This is listed in the DO NOT COPY table in [FILES.md](FILES.md#do-not-copy).

---

## Schema Decision: Inline DDL is Canonical

**PREFLIGHT Gap 2 resolution**.

Use `queries.ts:runMigrations()` inline DDL as the schema source. The SQL migration files (`000_base_schema.sql` + `001_reasoningbank_schema.sql`) diverge:

| Difference | Inline DDL | SQL Files |
|------------|-----------|-----------|
| Telemetry table name | `metrics_log` | `performance_metrics` |
| Multi-tenant tables | absent | `memory_namespaces`, `session_state` |
| CHECK constraints | absent | present on `judge_label`, link types |
| Triggers | absent | 2 useful triggers (see Fix 8) |
| `vector_index_meta` | absent | absent (neither defines it) |

The pipeline code references `metrics_log` (not `performance_metrics`), so inline DDL is correct. Optionally add the 2 triggers after migration.
