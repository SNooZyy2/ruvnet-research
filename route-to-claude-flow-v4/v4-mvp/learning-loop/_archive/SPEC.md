# v4 MVP: Close the Learning Loop

> **Status**: Implementation-ready (PREFLIGHT corrections applied)
> **Date**: 2026-03-03
> **Scope**: Fix the severed feedback loop — the ONE thing that makes claude-flow learn
> **Based on**: 142 research sessions + verified source code reads of every file in the pipeline
> **Companion docs**: [FILES.md](FILES.md) (what to copy) | [FIXES.md](FIXES.md) (exact code changes)

---

## 1. The Problem

The claude-flow architecture was designed with a self-improving feedback loop:

```
pre-task (retrieve past patterns, route to best model)
    -> AI does work (trajectory recorded)
        -> post-task (judge outcome, distill knowledge, consolidate memory)
            -> next pre-task retrieves what was learned
```

**The loop is severed at post-task.** Eight breaks prevent it from closing:

| # | Break | File | Severity |
|---|-------|------|----------|
| 1 | hook-handler.cjs stubs out the entire pipeline | hook-handler.cjs:140-147 | CRITICAL |
| 2 | Embeddings fall back to hash (no semantic content) | config.ts DEFAULT_CONFIG | CRITICAL |
| 3 | Distill fallback creates memory then returns [] | distill.ts:210-229 | HIGH |
| 4 | No trajectory data flows into post-task | execution engine gap | HIGH |
| 5 | Heuristic judge always returns "Failure" | judge.ts:153-173 | CRITICAL |
| 6 | Judge doesn't persist verdict to DB | post-task.ts (missing storeTrajectory call) | HIGH |
| 7 | UPSERT loses metadata on re-insert | queries.ts (3 statements) | HIGH |
| 8 | embeddings.js coexists with embeddings.ts | Node ESM may resolve wrong file | CRITICAL |

Details with exact code snippets in [FIXES.md](FIXES.md).

---

## 2. What Already Works

These components are genuine and functional — they just aren't wired together.

**Retrieve** (`core/retrieve.ts`, 121 LOC, 92-95% genuine)
4-factor scoring: `alpha * similarity + beta * recency + gamma * reliability` + MMR diversity. Works as-is.

**Judge** (`core/judge.ts`, 177 LOC, genuine with 3 bugs)
LLM-as-judge via ModelRouter + heuristic fallback. Bugs: static ModelRouter import crashes without router.ts, heuristic ignores exit codes, verdict not persisted. See Fixes 5-7.

**Distill** (`core/distill.ts`, 229 LOC, genuine with fixes needed)
LLM or template knowledge extraction + PII scrubbing. Template fallback creates memory then discards it. Also has static ModelRouter import. See Fixes 3, 7.

**Consolidate** (`core/consolidate.ts`, 258 LOC, genuine)
LSH dedup + contradiction detection + age-based pruning in single SQLite transaction. Works as-is.

**Database** (`db/queries.ts`, 441 LOC, genuine with data-loss bugs)
better-sqlite3, WAL mode, foreign keys. Full CRUD. Three UPSERTs reset metadata on re-insert. No LIMIT on fetchMemoryCandidates. No BLOB alignment check. See Fix 8.

**SONA Optimizer** (`sona-optimizer.ts`, 842 LOC, 72-78% genuine)
Bayesian agent routing, Thompson sampling, temporal decay. Already wired into hooks. Lazy-loads `q-learning-router.js` -> `@ruvector/core` which may fail if native module absent.

**Hooks** (`pre-task.ts` 79 LOC, `post-task.ts` 127 LOC, genuine but orphaned)
Pre-task retrieves + injects memories. Post-task orchestrates judge -> distill -> consolidate. Post-task is never called.

---

## 3. The 9 Fixes

Each fix has exact before/after code in [FIXES.md](FIXES.md). File inventory in [FILES.md](FILES.md).

| Fix | What | Files | Effort | PREFLIGHT correction |
|-----|------|-------|--------|---------------------|
| 1 | Wire post-task to ReasoningBank pipeline | handler.ts (new) | S-M | Use Option B (ESM in-process) |
| 2 | Activate real embeddings | config.ts | S | YAML already correct. Fix DEFAULT_CONFIG only (2 lines) |
| 3 | templateBasedDistill stores memories | distill.ts | S | -- |
| 4 | Capture trajectory from execution | trajectory/capture.ts (new) | M | -- |
| 5 | Persist verdict to trajectory table | post-task.ts | S | -- |
| 6 | Heuristic judge checks exit codes | judge.ts | S | -- |
| 7 | Dynamic ModelRouter import | judge.ts AND distill.ts | S | Must fix BOTH files, not just judge.ts |
| 8 | UPSERT preserves metadata | queries.ts | S | -- |
| 9 | Eliminate embeddings file mismatch | Don't copy embeddings.js | S | Original fix was WRONG. Real fix: only copy .ts |

### PREFLIGHT corrections (applied throughout)

- **reasoningbank.yaml**: Already has `dims: 384, provider: local, model: Xenova/all-MiniLM-L6-v2`. Only fix the DEFAULT_CONFIG fallback in config.ts.
- **types/index.ts**: NOT imported by anything (grep-confirmed). Do NOT copy. Dead file.
- **config/reasoningbank-types.ts**: NOT imported by anything. Do NOT copy. Dead file.
- **CJS/ESM**: Option B selected. v4 handler is ESM `.ts`, directly imports pipeline modules.
- **Inline DDL canonical**: `queries.ts:runMigrations()` is the schema source. SQL migration files diverge on table names.

---

## 4. What's NOT in This MVP

| Dropped | Why | When |
|---------|-----|------|
| Rust crate extraction | Algorithms, not the feedback loop | v4.1 |
| NAPI/WASM bridge | Loop is pure TS + SQLite | v4.1 |
| DDD bounded contexts | The loop IS the episodic context | v4.1 |
| Cryptographic provenance (RVF) | Audit trail, not learning | v4.2 |
| Claims system | Coordination, not learning | v4.2 |
| MaTTS multi-rollout | Exotic. Defer. | v4.1 |
| 175+ MCP tools | MVP needs ~5 | v4.1 |
| intelligence.ts replacement | ReasoningBank bypasses it; 14+ V3 consumers remain | v4.1 |
| Full MCP server rewrite | Two competing V3 servers; MVP depends on neither | v4.1 |

### Competing Pipelines (Won't Block MVP)

| Pipeline | Fires on | Risk | Action |
|----------|----------|------|--------|
| Q-learning (post-edit/post-command) | Edit/command events | NONE | Ignore (different event type) |
| GuidanceHookProvider PreTask | Task start | LOW | Guard if @claude-flow/guidance loaded |
| intelligence-bridge.js EWC++ | Trajectory end | NONE | LoRA/EWC stored but never used |
| LocalSonaCoordinator | 14+ V3 paths | MEDIUM | v4 handler doesn't import it |

---

## 5. Implementation Phases

### Phase 0: Bootstrap

Create v4 directory. Copy files per [FILES.md](FILES.md). Install deps. Fix UPSERTs (Fix 8) BEFORE first migration. Run DB migrations. Write bootstrap.ts.

**Gate**: `runMigrations()` creates `.swarm/memory.db` with correct schema.

### Phase 1: Activate Real Embeddings

Copy `embeddings.ts` (NOT `.js`). Fix DEFAULT_CONFIG in config.ts (Fix 2). Install `@xenova/transformers`. Remove NPX detection or set `FORCE_TRANSFORMERS=1`.

**Gate**: `cosine(embed("login auth"), embed("user auth")) > 0.7` and `cosine(embed("login"), embed("banana")) < 0.3`. Dims = 384.

### Phase 2: Fix Distill Fallback

Don't copy `embeddings.js` (Fix 9). Make templateBasedDistill async + call storeMemories (Fix 3).

**Gate**: After distill with no API key, `patterns` table has new rows.

### Phase 3: Wire Hook Pipeline

Fix judge.ts + distill.ts: dynamic ModelRouter import (Fix 7). Fix judge.ts: exitCode heuristic (Fix 6). Write handler.ts (Fix 1). Implement trajectory capture (Fix 4). Add storeTrajectory call (Fix 5). Pass real agentId/domain.

**Gate**: Run task -> `task_trajectories` has row with judge_label. `patterns` has new memories. Successful tasks get `Success` verdict.

### Phase 4: End-to-End Validation

Run 7 tasks of the same type. Verify: memories accumulate, retrieval scores improve, SONA confidence increases, consolidation merges duplicates.

**Gate**: On run 7, pre-task retrieves memories from prior runs with score > 0.5.

---

## 6. Dependency Graph

```
Phase 0: Bootstrap
    |
    +-- Phase 1: Activate Embeddings
    |       |
    |       +-- Phase 2: Fix Distill
    |               |
    |               +-- Phase 3: Wire Pipeline
    |                       |
    |                       +-- Phase 4: E2E Validation
    |
    +-- (SONA optimizer -- already works, just copy)
```

All phases sequential. Each depends on the previous.

---

## 7. Effort

| Phase | New/Fixed LOC | Notes |
|-------|--------------|-------|
| 0. Bootstrap | ~65 | Copy files + npm init + Fix 8 UPSERTs |
| 1. Embeddings | ~10 | Fix 2 config + npm install |
| 2. Distill | ~11 | Fix 3 + Fix 9 |
| 3. Pipeline | ~280 | handler.ts + trajectory + Fixes 5-7 |
| 4. Validation | ~100 | Test protocol + fixes from testing |
| **Total** | **~466 LOC** | |

---

## 8. Risk Registry

| Risk | Mitigation |
|------|-----------|
| @xenova/transformers downloads 80MB on first load | Cache in `.swarm/models/`. Lazy-load on first search. |
| No API key = heuristic judge only | Sufficient after Fix 6 (exitCode). LLM judge optional. |
| Coarse trajectory from spawn() | Start coarse, refine in v4.1. |
| sona-optimizer.ts lazy-loads @ruvector/core | Verify graceful degradation to Thompson sampling. |
| Dimension mismatch if config not fixed | Fix DEFAULT_CONFIG BEFORE first embedding stored. |
| fetchMemoryCandidates no LIMIT | Add LIMIT 1000 in queries.ts (included in Fix 8). |
| helpers/memory.js competing persistence | Don't load in v4 handler. |
| Double consolidation from intelligence-bridge.js | v4 handler must be sole post-task hook. |
| Double retrieval from GuidanceHookProvider | Check if guidance hooks fire; disable one path. |
| Hooks require session restart to reload | Re-query DB each invocation, don't cache at startup. |
| DELETE doesn't propagate to vector index | MVP uses queries.ts directly (no AgentDB adapter). |
| Two competing V3 MCP servers | MVP depends on neither. |

---

## 9. What Closing the Loop Gives You

**Run 1**: Agent gets "fix TS import error" -> no memories -> works -> judged Success -> distills "check tsconfig paths" -> stored at 0.7 confidence

**Run 2**: Same task type -> retrieves "check tsconfig paths" (score 0.6) -> works faster -> Success -> reinforcing memory -> consolidation merges -> confidence 0.8

**Run 7**: Same type -> retrieves consolidated memory (confidence 0.9) -> SONA routes to optimal model -> full context from 6 prior successes

Convergent determinism through reinforcement. The system increasingly makes the same successful choices for similar tasks.

---

## 10. Expansion Path

| Phase | What | Effort |
|-------|------|--------|
| v4.1 | Real HNSW via ruvector-core NAPI | 1-2 weeks |
| v4.1 | MaTTS multi-rollout verification | 3-5 days |
| v4.1 | Full MCP server with DDD contexts | 1-2 weeks |
| v4.2 | Cryptographic witness chains (RVF) | 1 week |
| v4.2 | Claims system unification | 3-5 days |
| v4.3 | Rust crate library via NAPI | 2-3 weeks |

**The loop must work before the algorithms matter.**
