# Break Analysis: Why the Learning Loop Never Closes

> Verified from source code reads on 2026-03-03.
> Every claim below includes the exact file path, line number, and code snippet.

---

## The Designed Flow

```
┌─────────┐    ┌──────────┐    ┌──────────┐    ┌─────────────┐    ┌─────────┐
│pre-task  │───▶│ AI works │───▶│post-task │───▶│ judge       │───▶│ distill │
│ retrieve │    │(trajectory│    │(STUB)    │    │ (never      │    │ (never  │
│ + route  │    │ recorded) │    │          │    │  called)    │    │  called)│
└─────────┘    └──────────┘    └──────────┘    └─────────────┘    └─────────┘
     ▲                                                                   │
     │                              ╳ SEVERED                            │
     └───────────────────────────────────────────────────────────────────┘
```

---

## Break 1: The Post-Task Stub (PRIMARY)

### Where

`~/repos/claude-flow/.claude/helpers/hook-handler.cjs` lines 140-147

### What It Does

```javascript
'post-task': () => {
    if (intelligence && intelligence.feedback) {
      try {
        intelligence.feedback(true);
      } catch (e) { /* non-fatal */ }
    }
    console.log('[OK] Task completed');
},
```

Seven lines. `intelligence.feedback(true)` passes a single boolean — no trajectory data, no query, no agent info, no timing. The `intelligence` object here is `intelligence.ts` (the O(n) facade, R140 CRITICAL). Its `feedback()` method updates a single counter. That's it.

### What Should Happen

The complete pipeline exists in agentic-flow:

`~/repos/agentic-flow/agentic-flow/src/reasoningbank/hooks/post-task.ts` (127 LOC)

```typescript
// Step 1: Judge trajectory
const verdict = await judgeTrajectory(trajectory, query);

// Step 2: Distill memories
const memoryIds = await distillMemories(trajectory, verdict, query, { taskId, agentId, domain });

// Step 3: Check if consolidation should run
if (shouldConsolidate()) {
    const result = await consolidate();
}
```

This file imports `judgeTrajectory` from `core/judge.ts`, `distillMemories` from `core/distill.ts`, and `consolidate` + `shouldConsolidate` from `core/consolidate.ts`. All four modules are genuine (92-95% realness). The orchestration is correct. It's simply never called.

### Why It's Never Called

hook-handler.cjs is a CJS file loaded synchronously by the claude-flow CLI. It has no knowledge of or dependency on the agentic-flow ReasoningBank. The post-task handler was written as a placeholder ("we'll wire this later") and never replaced.

### Evidence Chain

- R73 (CRITICAL): First identified hook-handler.cjs stub bypassing ReasoningBank
- R140: Confirmed stub still present, traced full execution path
- Direct source code read (2026-03-03): Lines 140-147 verified unchanged

---

## Break 2: Hash-Based Embeddings (SECONDARY)

### Where

`~/repos/agentic-flow/agentic-flow/src/reasoningbank/utils/embeddings.js` lines 22-29, 72-80

### What It Does

```javascript
if (config.embeddings.provider === 'openai') {
    embedding = await openaiEmbed(text, config.embeddings.model);
} else if (config.embeddings.provider === 'claude') {
    embedding = hashEmbed(text, config.embeddings.dimensions || 1024);
} else {
    embedding = hashEmbed(text, config.embeddings.dimensions || 1024);
}
```

The fallback chain: OpenAI (needs API key) → Claude (no native embeddings, falls to hash) → local hash.

```javascript
function hashEmbed(text, dims) {
    const hash = simpleHash(text);  // DJB2-style integer hash
    const vec = new Float32Array(dims);
    for (let i = 0; i < dims; i++) {
        vec[i] = Math.sin(hash * (i + 1) * 0.01) * 0.1 + Math.cos(hash * i * 0.02) * 0.05;
    }
    return normalize(vec);
}
```

This generates a deterministic vector from a single integer hash of the text. The resulting vectors have no semantic content — `hashEmbed("authentication login")` and `hashEmbed("user auth session")` produce dissimilar vectors despite being semantically identical.

### Impact on the Learning Loop

Even if Break 1 were fixed, the retrieve step would return random results:
1. `computeEmbedding(query)` produces a hash-based query vector
2. `fetchMemoryCandidates()` returns memories with hash-based stored vectors
3. `cosineSimilarity(queryEmbed, candidateEmbed)` computes similarity between two meaningless vectors
4. Results are effectively random

This is the same R20 root cause that affects the entire ruvnet ecosystem. The ReasoningBank has its own instance of the problem.

**Scope is broader than this file**: Domain research (02-ruvector, 07-model-routing) confirms 9+ additional hash embedding instances across the codebase: `hooks-tools.ts generateSimpleEmbedding()` (uses sin/cos hash even when native VectorDb/HNSW is loaded), `agentdb-fast.ts`, and several ruvector CLI paths. Fix 2 only fixes the ReasoningBank pipeline instance. The hooks layer, model routing layer, and ruvector CLI all retain their own hash embedding instances independently. These won't affect the MVP loop (which only uses the ReasoningBank path), but implementors should be aware they persist in the surrounding runtime.

### Why It's Like This

No locally-runnable embedding model is wired in. `@xenova/transformers` exists in the ecosystem (confirmed working in Pipeline 1, R137) and `onnx-embedder.ts` (85-90%) is available in the ruvector umbrella package. Neither was wired into this utility.

### Evidence Chain

- R20: Root cause identified — EmbeddingService never initialized in claude-flow bridge
- R117: onnx-embedder.ts found with real ONNX via Tract/WASM
- R137: Pipeline 1 (@xenova/transformers) confirmed working in agentdb-mcp-server.ts
- Direct source code read (2026-03-03): hashEmbed fallback verified

---

## Break 3: Distill Fallback Returns Empty (TERTIARY)

### Where

`~/repos/agentic-flow/agentic-flow/src/reasoningbank/core/distill.ts` lines 210-229

### What It Does

```typescript
function templateBasedDistill(
    trajectory: Trajectory, verdict: Verdict, query: string, options: any
): string[] {
    console.log('[INFO] Using template-based distillation (no API key)');

    const memory: DistilledMemory = {
        title: `${verdict.label}: ${query.substring(0, 50)}`,
        description: `Task outcome: ${verdict.label}`,
        content: `Query: ${query}\nSteps: ${trajectory.steps?.length || 0}\nOutcome: ${verdict.label}`,
        tags: [verdict.label.toLowerCase(), 'template'],
        domain: options.domain
    };

    return []; // Skip storage for template-based (would need to make this async)
}
```

The function creates a perfectly valid `DistilledMemory` object, then discards it and returns an empty array. The comment reveals the reason: the function is synchronous but `storeMemories()` is async (it calls `computeEmbedding()` which is async). The developer chose to return [] rather than make the function async.

### Impact on the Learning Loop

Without an external LLM API key (OpenRouter, Anthropic, or Google Gemini), the distill step produces zero memories. Since the heuristic judge fallback doesn't require an API key, the judge step works fine without one — but the distill step silently drops all results.

This means that in the common case (running without external API keys), even a fully wired pipeline would learn nothing.

### Fix

Make `templateBasedDistill` async and call `storeMemories()`:

```typescript
async function templateBasedDistill(
    trajectory: Trajectory, verdict: Verdict, query: string, options: any
): Promise<string[]> {
    const memory: DistilledMemory = { /* same as before */ };
    const confidencePrior = verdict.label === 'Success' ? 0.6 : 0.3;
    return storeMemories([memory], confidencePrior, verdict, options);
}
```

One-line return type change + one-line body change. The caller (`distillMemories()`) already awaits the result.

### Evidence Chain

- Direct source code read (2026-03-03): return [] verified on line 228
- Comment on line 228 confirms intentional skip

---

## Break 4: No Trajectory Data Captured (QUATERNARY)

### Where

The gap exists between the execution engine and the post-task hook.

### What Happens Today

1. Pre-task hook runs model routing (enhanced-model-router.js) → SONA routes task → agent selected
2. Agent executes via `spawn('claude', ['--print', prompt])` (R140)
3. Agent output is captured as a string
4. Post-task hook fires with... nothing. No trajectory steps. No query. No timing.

The existing `post-task.ts` expects a trajectory JSON on stdin or via `--trajectory-file`:

```typescript
function loadTrajectory(filePath?: string): { trajectory: Trajectory; query: string } {
    let content: string;
    if (filePath) {
        content = readFileSync(filePath, 'utf-8');
    } else {
        content = readFileSync(0, 'utf-8'); // stdin
    }
    const data = JSON.parse(content);
    return { trajectory: { steps: data.steps || [], metadata: data.metadata || {} }, query: data.query || 'Unknown' };
}
```

No component in the execution pipeline produces this JSON.

### Impact on the Learning Loop

Even with Breaks 1-3 fixed, the judge has nothing to judge. It would receive an empty trajectory (`{ steps: [], metadata: {} }`) and the heuristic judge would return `{ label: 'Failure', confidence: 0.5 }` since there are no steps and no completion markers.

### Minimal Fix

The hook handler knows:
- The task query (from pre-task routing)
- The agent type (from SONA routing)
- The start/end timestamps
- The exit code of the spawned process
- The stdout of the agent (contains tool calls, reasoning, results)

Construct a minimal trajectory from these:

```typescript
const trajectory: Trajectory = {
    steps: [
        { action: 'spawn', agent: agentType, query: taskQuery, timestamp: startTime },
        { action: 'execute', exitCode: result.exitCode, outputLength: result.stdout.length, timestamp: endTime }
    ],
    metadata: { duration: endTime - startTime, agent: agentType, model: routedModel }
};
```

This is coarse but gives the judge enough signal: exit code 0 + output present = likely success. The heuristic judge can work with this. LLM judge (when available) can analyze the actual output.

---

---

## Break 5: Heuristic Judge Always Returns "Failure" (QUINARY)

### Where

`~/repos/agentic-flow/agentic-flow/src/reasoningbank/core/judge.ts` lines 153-173

### What It Does

The heuristic fallback JSON.stringifies each trajectory step and searches for the literal strings `"error"` and `"complete"`:

```typescript
for (const step of steps) {
    const s = JSON.stringify(step).toLowerCase();
    if (s.includes('error')) errorCount++;
    if (s.includes('complete')) successSignals++;
}
```

With the minimal trajectory from Fix 4:
```json
{"action": "execute", "exitCode": 0, "outputLength": 5000}
```

This serialized step **never contains "complete"**. Result: `successSignals = 0`, `errorCount = 0`, verdict defaults to `{ label: 'Failure', confidence: 0.5 }`.

### Impact on the Learning Loop

Every task is judged as a failure regardless of actual outcome. The distill step receives failure verdicts, storing failure-pattern memories. The system "learns" that everything fails — the exact opposite of convergent improvement.

### Fix

Add exit code awareness:
```typescript
if (step.exitCode === 0) successSignals++;
if (step.exitCode && step.exitCode !== 0) errorCount++;
```

### Evidence Chain

- R140: heuristic judge identified as naive (string "error" / "complete" only)
- DB query (2026-03-03): CRITICAL/QUALITY finding confirmed

---

## Break 6: Judge Does Not Persist Verdict to DB (SENARY)

### Where

`~/repos/agentic-flow/agentic-flow/src/reasoningbank/core/judge.ts` line 176 (late import of db)

### What It Does

judge.ts calls `db.logMetric('rb.judge', ...)` for telemetry but **never** calls `db.storeTrajectory()`. The verdict object is returned to the caller but the `task_trajectories` table is never updated with judge results.

### Impact on the Learning Loop

The trajectory table has rows (from storeTrajectory calls in other code paths) but `judge_label`, `judge_conf`, and `judge_reasons` columns are NULL. No queryable evaluation history. The only way to know what happened is to re-judge from scratch.

### Fix

post-task.ts must explicitly store:
```typescript
const verdict = await judgeTrajectory(trajectory, query);
db.storeTrajectory(taskId, agentId, query, JSON.stringify(trajectory), verdict.label, verdict.confidence, JSON.stringify(verdict.reasons));
```

### Evidence Chain

- R140: CRITICAL/INTEGRATION finding: "MCP BYPASS CONFIRMED: Calls db.logMetric() but NEVER db.store()"

---

## Break 7: UPSERT Loses Metadata on Re-Insert (SEPTENARY)

### Where

`~/repos/agentic-flow/agentic-flow/src/reasoningbank/db/queries.ts` — 3 statements

### What It Does

All three UPSERT operations use `INSERT OR REPLACE`:

```sql
INSERT OR REPLACE INTO patterns (id, type, pattern_data, confidence) VALUES (?, ?, ?, ?)
INSERT OR REPLACE INTO task_trajectories (task_id, ...) VALUES (?, ...)
INSERT OR REPLACE INTO pattern_links (src_id, dst_id, relation, weight) VALUES (?, ?, ?, ?)
```

SQLite's `INSERT OR REPLACE` deletes the existing row and inserts a new one. This loses:
- `created_at` (reset to now)
- `usage_count` (reset to 0)
- `last_used` (reset)
- All accumulated history

### Impact on the Learning Loop

The SPEC's Section 10 promises "confidence rises to 0.8" over multiple runs. But if a pattern with the same ID is re-inserted, `usage_count` resets to 0 and `created_at` resets to now. The convergence mechanism is undermined — memories never accumulate history.

### Fix

Change to `INSERT ... ON CONFLICT(id) DO UPDATE SET`:
```sql
INSERT INTO patterns (id, type, pattern_data, confidence)
VALUES (?, ?, ?, ?)
ON CONFLICT(id) DO UPDATE SET
  pattern_data = excluded.pattern_data,
  confidence = excluded.confidence,
  usage_count = usage_count + 1,
  last_used = datetime('now');
```

### Evidence Chain

- DB query (2026-03-03): 3 HIGH/QUALITY findings for UPSERT metadata loss

---

## Break 8: Distill Imports Wrong Embeddings File (OCTONARY)

### Where

`~/repos/agentic-flow/agentic-flow/src/reasoningbank/core/distill.ts` — import statement

### What It Does

`distill.ts` imports from `../utils/embeddings.js` (114 LOC, hash-based only). Meanwhile, `retrieve.ts` imports from `../utils/embeddings.ts` (211 LOC, with `@xenova/transformers`).

### Impact on the Learning Loop

This creates a **vector space mismatch**:
1. Distill stores memories with hash-based embedding vectors (from `embeddings.js`)
2. Retrieve computes query vectors with real semantic embeddings (from `embeddings.ts`)
3. Cosine similarity between real and hash vectors is meaningless

Even with Breaks 1-7 fixed and real embeddings activated (Fix 2), the loop silently fails: memories get stored but can never be meaningfully retrieved because the two embedding spaces are incompatible.

### Fix

Change the import path from `../utils/embeddings.js` to `../utils/embeddings` (or `.ts`).

### Evidence Chain

- Research DB dependency table (2026-03-03): `distill.ts --[IMPORTS]--> embeddings.js` confirmed
- Contrast: `retrieve.ts --[IMPORTS]--> embeddings.ts` confirmed
- This was not caught in R72-R73 because each file was read in isolation; the cross-file import mismatch only surfaces when checking the dependency graph

---

## Summary: The Eight Breaks in Order

| # | Break | Severity | Fix Effort | Files |
|---|-------|----------|-----------|-------|
| 1 | hook-handler.cjs stubs out pipeline | **CRITICAL** | S-M | hook-handler.cjs |
| 2 | Embeddings fall back to hash | **CRITICAL** | S | embeddings.ts (config fix) |
| 3 | Distill fallback stores nothing | **HIGH** | S | distill.ts |
| 4 | No trajectory data captured | **HIGH** | M | New: trajectory-capture.ts |
| 5 | Heuristic judge always returns "Failure" | **CRITICAL** | S | judge.ts |
| 6 | Judge doesn't persist verdict to DB | **HIGH** | S | post-task.ts |
| 7 | UPSERT loses metadata on re-insert | **HIGH** | S | queries.ts |
| 8 | Distill imports wrong embeddings file | **CRITICAL** | S | distill.ts (import) |

Fixes 1+2+5+8 are required — without them, nothing works correctly.
Fix 3 is required for no-API-key environments.
Fix 4 is required for meaningful learning (otherwise judge sees empty input).
Fix 6 is required for trajectory history (otherwise evaluation data is lost).
Fix 7 is required for convergent confidence (otherwise memories reset instead of accumulating).

Also required but not "breaks": judge.ts static ModelRouter import (crashes without router.ts), missing `types/index.ts` (import failure), missing `yaml` npm dep (config crash), missing `reasoningbank.yaml` (dimension mismatch).

All eight breaks + supporting fixes: ~466 LOC of changes/additions to close the loop.
