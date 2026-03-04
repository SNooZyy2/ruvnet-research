# v4 MVP: File Inventory

> Corrected file list. All PREFLIGHT corrections applied.
> Companion docs: [SPEC.md](SPEC.md) (what + why) | [FIXES.md](FIXES.md) (exact code changes)

---

## Copy from ~/repos/agentic-flow/agentic-flow/src/reasoningbank/

### Core Pipeline

| File | LOC | Action | Fix Ref |
|------|-----|--------|---------|
| `core/retrieve.ts` | 121 | COPY | -- |
| `core/judge.ts` | 177 | FIX | [Fix 6](FIXES.md#fix-6-heuristic-judge-checks-exit-codes), [Fix 7](FIXES.md#fix-7-dynamic-modelrouter-import) |
| `core/distill.ts` | 229 | FIX | [Fix 3](FIXES.md#fix-3-templatebaseddistill-stores-memories), [Fix 7](FIXES.md#fix-7-dynamic-modelrouter-import) |
| `core/consolidate.ts` | 258 | COPY | -- |

### Hooks

| File | LOC | Action | Fix Ref |
|------|-----|--------|---------|
| `hooks/pre-task.ts` | 79 | ADAPT | Remove `process.exit()`, make importable |
| `hooks/post-task.ts` | 127 | ADAPT | Remove `process.exit()`, [Fix 5](FIXES.md#fix-5-persist-verdict-to-db), pass real agentId/domain |

### Database

| File | LOC | Action | Fix Ref |
|------|-----|--------|---------|
| `db/queries.ts` | 441 | FIX | [Fix 8](FIXES.md#fix-8-upsert-preserves-metadata) |
| `db/schema.ts` | 89 | COPY | -- |

### Utilities

| File | LOC | Action | Fix Ref |
|------|-----|--------|---------|
| `utils/embeddings.ts` | 211 | COPY | Already has @xenova/transformers. **Use .ts NOT .js** |
| `utils/mmr.js` | ~60 | COPY | Pure math (mmrSelection + cosineSimilarity) |
| `utils/pii-scrubber.js` | ~50 | COPY | scrubMemory + containsPII |
| `utils/config.ts` | 241 | FIX | [Fix 2](FIXES.md#fix-2-activate-real-embeddings) (DEFAULT_CONFIG dims/provider) |

### Config and Prompts

| File | LOC | Action | Notes |
|------|-----|--------|-------|
| `config/reasoningbank.yaml` | 146 | COPY | Already correct: dims 384, provider local, model all-MiniLM-L6-v2 |
| `prompts/judge.json` | ~120 | COPY | LLM judge prompt |
| `prompts/distill-success.json` | ~120 | COPY | Success distillation prompt |
| `prompts/distill-failure.json` | ~150 | COPY | Failure distillation prompt |

**Subtotal**: 15 files, ~2,420 LOC

---

## Copy from ~/repos/claude-flow/v3/@claude-flow/cli/

| File | LOC | Action | Notes |
|------|-----|--------|-------|
| `src/memory/sona-optimizer.ts` | 842 | COPY | Bayesian routing. Self-contained, persists to .swarm/sona-patterns.json |

---

## DO NOT COPY

| File | Why |
|------|-----|
| `utils/embeddings.js` | 114 LOC hash-based. Superseded by embeddings.ts. **Removing it IS Fix 9.** |
| `types/index.ts` | NOT imported by anything (PREFLIGHT grep-confirmed). Dead file. |
| `config/reasoningbank-types.ts` | NOT imported by anything. config.ts defines types inline. Dead file. |
| `core/distill.js` | JS version regresses vs TS -- hardcoded Anthropic API |
| `core/retrieve.js` | JS duplicate with hash embedding import |
| `core/judge.js` | JS version uses direct Anthropic API |
| `intelligence.ts` | O(n) facade claiming O(log n). BIGGEST active facade. |
| `hook-handler.cjs` | The stub we're replacing |
| `router/router.ts` | 420 LOC, zero findings, zero reads. Use dynamic import with catch instead. |

---

## New Files to Create

| File | Est. LOC | Purpose |
|------|----------|---------|
| `hooks/handler.ts` | ~150 | v4 hook handler: pre-task retrieval + post-task pipeline + SONA update |
| `bootstrap.ts` | ~50 | Initialize DB (runMigrations) + embedding model (lazy load) |
| `trajectory/capture.ts` | ~100 | Structured trajectory from execution |
| `config/default.yaml` | ~40 | Default ReasoningBank config |
| `package.json` | ~20 | Deps and ESM config |

---

## npm Dependencies

### Required

| Package | Version | Size | Purpose |
|---------|---------|------|---------|
| `better-sqlite3` | ^11.x | ~8MB native | SQLite driver |
| `ulid` | ^3.x | ~10KB | Sortable unique IDs for memories |
| `@xenova/transformers` | ^2.x | ~2MB (+80MB model on first use) | Real local embeddings (all-MiniLM-L6-v2, 384-dim) |
| `yaml` | ^2.8.x | ~100KB | YAML parser for config |

### Optional (for LLM-based judge/distill)

| Package | Version | Purpose |
|---------|---------|---------|
| `openai` | ^4.x | OpenRouter / OpenAI API for LLM judge |
| `@anthropic-ai/sdk` | ^0.x | Anthropic API for LLM judge |

Without optional deps, the pipeline uses heuristic judge + template distill. Functional but lower quality.

### package.json

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

---

## Target Directory Structure

```
claude-flow-v4/
  package.json
  tsconfig.json
  bootstrap.ts
  hooks/
    handler.ts              # NEW (Fix 1)
  trajectory/
    capture.ts              # NEW (Fix 4)
  reasoningbank/
    core/
      retrieve.ts           # COPY
      judge.ts              # FIX (Fixes 6, 7)
      distill.ts            # FIX (Fixes 3, 7)
      consolidate.ts        # COPY
    db/
      queries.ts            # FIX (Fix 8)
      schema.ts             # COPY
    utils/
      embeddings.ts         # COPY (NO .js!)
      mmr.js                # COPY
      pii-scrubber.js       # COPY
      config.ts             # FIX (Fix 2)
    config/
      reasoningbank.yaml    # COPY (already correct)
    prompts/
      judge.json            # COPY
      distill-success.json  # COPY
      distill-failure.json  # COPY
  routing/
    sona-optimizer.ts       # COPY
  config/
    default.yaml            # NEW
```

---

## Validation Checklist

Run after each phase. Phase gates in [SPEC.md S5](SPEC.md#5-implementation-phases).

Note: Project is `"type": "module"`. All validation uses ESM `import()`, not `require()`.

```bash
# 1. DB initializes (Phase 0 gate)
node --input-type=module -e "
import { runMigrations } from './reasoningbank/db/queries.js';
runMigrations(); console.log('OK');
"

# 2. Real embeddings, NOT hash (Phase 1 gate)
FORCE_TRANSFORMERS=1 node --input-type=module -e "
import { computeEmbedding } from './reasoningbank/utils/embeddings.js';
const v1 = await computeEmbedding('user authentication login');
const v2 = await computeEmbedding('auth session management');
const v3 = await computeEmbedding('banana smoothie recipe');
console.log('dims:', v1.length);          // 384, NOT 1024
const cos = (a,b) => {
  let d=0,n1=0,n2=0;
  for (let i=0;i<a.length;i++) { d+=a[i]*b[i]; n1+=a[i]**2; n2+=b[i]**2; }
  return d / Math.sqrt(n1*n2);
};
console.log('auth-auth:', cos(v1, v2));   // > 0.6
console.log('auth-banana:', cos(v1, v3)); // < 0.3
"

# 3. UPSERT preserves metadata (Phase 0 gate)
# Upsert same pattern ID twice, verify usage_count increments to 2

# 4. judge.ts loads without crash (Phase 3 gate)
node --input-type=module -e "
import { judgeTrajectory } from './reasoningbank/core/judge.js';
console.log('OK');
"

# 5. Heuristic recognizes exitCode=0 as success (Phase 3 gate)
node --input-type=module -e "
import { judgeTrajectory } from './reasoningbank/core/judge.js';
const v = await judgeTrajectory(
  {steps: [{action:'execute', exitCode:0, outputLength:1000}]}, 'test');
console.log(v.label);       // 'Success', NOT 'Failure'
console.log(v.confidence);  // > 0.5
"

# 6. Stored embeddings are 384-dim (Phase 2 gate)
node --input-type=module -e "
import Database from 'better-sqlite3';
const db = new Database('.swarm/memory.db');
const rows = db.prepare('SELECT dims FROM pattern_embeddings LIMIT 5').all();
console.log('Stored dims:', rows.map(r => r.dims));  // ALL 384
"
```

---

## Reference Files (Not Copied -- v4.1+)

Files discovered in research that the loop could benefit from later.

| File | LOC | Realness | Why It Matters |
|------|-----|----------|----------------|
| `agentdb/HybridSearch.ts` | ~300 | 95% | BM25+HNSW vector fusion. Replace brute-force in v4.1. |
| `vector-migration-job.ts` | 203 | 85-90% | Batch re-embedding for hash->real migration. |
| `hooks-bridge.js` | 235 | ~95% | SDK hook lifecycle bridge. Could adapt for trajectory capture. |
| `sqlite-pool.js` | 587 | 92% | Production SQLite pool with WAL + auto-recovery. |
| `health-monitor.ts` | 514 | 99% | Memory leak detection + 4-tier self-healing. Best monitoring code. |
| `ExplainableRecall.ts` | ~250 | 88% | Provenance tracking + set-cover diversification. |
| `reasoningbank-learning/` (Rust) | 788 | 95-98% | GENUINE Rust crate. AdaptiveLearner, StrategyOptimizer. |
