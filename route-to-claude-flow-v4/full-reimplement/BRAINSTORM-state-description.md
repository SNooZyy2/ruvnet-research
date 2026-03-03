# Brainstorm: How to Reason Over the Whole Project

> Date: 2026-03-03 (updated)
> Original: 2026-02-19 (112 sessions, 1,518 DEEP, 10,954 findings)
> Current: 142 sessions, 1,696 DEEP files, 12,877 findings — the index exceeds a single context window. How do we create a navigable "state description" of the ecosystem?

---

## The Core Problem

No single context window can hold the full state of a 14,633-file, 4-repo ecosystem. The MASTER-INDEX is already too big. The 14 synthesis docs together are too big. And even if they weren't, a flat document doesn't capture *connections* — which is what you actually need for building v4.

## Why "One Big Document" Is the Wrong Goal

The instinct is to build a comprehensive state description. But that's solving the wrong problem. When building v4, you never need to reason about the *whole project simultaneously*. You need to answer specific questions:

- "Can I reuse the HNSW implementation?" -> ruvector-core, score 85-93%, real SIMD, placeholder embeddings
- "Which persistence layer should v4 use?" -> There are 12 disconnected ones, none compose
- "Is the MCP server genuinely functional?" -> Yes, 256 real tools, but R20 root cause still lurking

The problem isn't *size* — it's *navigation*.

## What's Already Built

| Layer | Artifact | Status |
|-------|----------|--------|
| Raw data | DB (14,633 files, 10,954 findings, 1,662 deps) | Complete |
| Per-domain narrative | 14 synthesis docs | Complete |
| Statistical overview | MASTER-INDEX.md | Complete |
| Feature-by-feature verdict | README-REALITY-CHECK.md | Complete |
| Extraction manifest | GENUINE-ASSETS.md | Complete |
| Build specification | SPEC.md | Complete |
| Middle Layer trace | ML-A through ML-F (R135-R140) | Complete — CLI→MCP→tool→memory→backend fully traced |
| Compilation audit | R141 binary pass/fail for 115 crates | Complete — 100/115 pass cargo check |
| MEMORY.md navigation layer | Evolved across 142 sessions | Partially fulfills "Architectural Reality doc" role |

## What's Missing: Two Intermediate Layers

### Missing Layer 1: Subsystem Graph (~20-30 nodes)

Not file-level, not domain-level — *subsystem-level*. Something like:

```
ruvector-core/hnsw  --uses-->  hnsw_rs (vendored)
ruvector-core/hnsw  --feeds--> ruvllm/hnsw_router
ruvllm/serving      --owns-->  vLLM/Orca batch engine
ruvllm/serving      --ignores--> ruvllm/context_manager (only 2/5 siblings)
sona/algorithms     --real-->  MicroLoRA + EWC++
sona/orchestration  --broken--> background loop synchronous, Loop C missing
mincut/kernels      --real-->  SIMD, 88-93%
mincut/theory       --broken--> expander O(n^3), witness tautology
```

This is maybe 20-30 nodes with ~50-80 edges. It fits in one page. It tells you the *architecture as it actually exists*, not as claimed. And it's directly actionable for v4 decisions.

> **STATUS (2026-03-03)**: ADR-v4-007 was designed (see `ADR-v4-007-subsystem-graph.md`) but NEVER EXECUTED. The design remains valid. The Middle Layer sessions (R135-R140) would significantly enrich Phase 3's prose mining — ~500+ new findings with cross-subsystem signals are now available. R141's compilation audit adds a binary truth signal (pass/fail) that wasn't available when this was designed.

### Missing Layer 2: v4 Reuse Inventory

For each v4 promise/feature, a mapping:

```
v4 Feature           | Best Existing Impl        | Score  | Verdict
---------------------|---------------------------|--------|----------
HNSW vector search   | ruvector-core             | 85-93% | REUSE (fix embeddings)
Agent coordination   | MCP server (256 tools)    | ~88%   | REUSE (fix EmbeddingService init)
LoRA fine-tuning     | sona micro_lora.rs        | 92-95% | REUSE
Batch inference      | ruvllm/serving/           | ~90%   | REUSE (best subsystem)
Sublinear search     | backward_push.rs          | ~88%   | REUSE (only genuine O(1/eps))
Persistence          | 12 parallel layers        | varies | REWRITE (none compose)
WASM acceleration    | 60% genuine, 40% facade   | mixed  | CHERRY-PICK
Graph queries        | rvlite Cypher executor    | 82-86% | REUSE
```

This is the **bridge document** from research to implementation. It's small, fits in one window, and directly answers "what do I build vs. what do I reuse?"

> **STATUS (2026-03-03)**: ADR-v4-008 was designed and written (see `ADR-v4-008-reuse-inventory.md`). It needs revision: L2-01 action should change from BUILD to ADAPT (onnx-embedder.ts found in R117), NAPI should be primary bridge (R116), ruvllm marked as non-compilable (R141).

## Concrete Next Steps (in order)

### Step 1: Build the Subsystem Graph (from DB)

You already have ~1,662 dependency edges. Aggregate them to subsystem level:
- Group files into ~25-30 subsystems (by crate + directory)
- Count cross-subsystem edges
- Tag each subsystem with avg realness score
- Output: a single Mermaid diagram + adjacency list

This is a DB query + some JS post-processing. Feasible in one session.

### Step 2: Build the v4 Reuse Inventory

Read `route-to-claude-flow-v4/README-REALITY-CHECK.md`, extract each v4 feature/promise, then query the DB for the best existing implementation of each. Output: one table, ~30-50 rows.

### Step 3: Build a "Question Router"

Instead of one monolithic doc, build a small set of pre-built queries:
- "What subsystems connect to X?" -> SQL join on subsystem graph
- "What's the realest implementation of capability Y?" -> findings + scores query
- "What's broken in subsystem Z?" -> CRITICAL/HIGH findings filtered by subsystem

This gives any future Claude session the ability to *navigate* the research without loading all of it.

### Step 4: Write the "Architectural Reality" doc (~10 pages)

Only after steps 1-3. This is the executive summary that fits in one context window:
- Page 1-2: Subsystem graph + legend
- Page 3-4: What's genuinely real (top 15 subsystems by score)
- Page 5-6: What's broken/facade (systemic issues: hash embeddings, theatrical WASM, disconnected persistence)
- Page 7-8: v4 reuse inventory
- Page 9-10: Open questions and architectural risks

This replaces the need to ever load the full index or all 14 synthesis docs.

### Status Update (2026-03-03)

| Step | Status | Notes |
|------|--------|-------|
| Step 1: Subsystem Graph | DESIGNED, NOT EXECUTED | ADR-v4-007 complete, needs ~6-7 hours to build |
| Step 2: Reuse Inventory | BUILT, NEEDS REVISION | ADR-v4-008 exists, needs R115-R142 updates |
| Step 3: Question Router | NOT STARTED | MEMORY.md partially serves this role |
| Step 4: Architectural Reality doc | NOT STARTED | Conversation summaries partially serve this role |

The Middle Layer deep-dive (R135-R140) and compilation audit (R141) have provided the remaining data needed for Steps 1-2. The research is now complete enough to execute these steps with high confidence.

## The Key Insight

**You don't need a document that describes the whole project. You need a *navigation system* that can answer any question about the project by loading the right 5% of data.**

**Post-R142 addendum**: MEMORY.md has evolved into a partial navigation system over 142 sessions — it captures stable patterns, key corrections, and cross-subsystem signals in ~200 lines. Combined with the research DB's queryable views (`smart_priority_gaps`, `domain_coverage`, `integration_hotspots`), the "navigation system" is partially operational even without the formal Subsystem Graph. The graph would still be valuable for cold-start sessions that don't have MEMORY.md loaded.

The DB is already that system — it just lacks the intermediate aggregation layer (subsystems) and the decision-oriented views (reuse inventory). The synthesis docs are narratives for humans; what you need for v4 is a *queryable decision support system* with a thin summary layer on top.

The index was the right first step. The subsystem graph + reuse inventory is the right next step.
