# R73 Execution Plan: Clear CONNECTED Tier — ReasoningBank TS Outer Ring

**Date**: 2026-02-16
**Session ID**: 73
**Focus**: All 12 remaining CONNECTED files — the entire agentic-flow ReasoningBank TypeScript outer ring (entry points, utils, hooks, DB schema, WASM adapter). Clears the CONNECTED tier entirely (MILESTONE).
**Strategic value**: After R73, zero CONNECTED files remain. Every dependency edge connects two DEEP files. The ~165 remaining PROXIMATE files become the sole frontier. All 12 files are in the ReasoningBank TS layer, completing the picture started by R67 (Rust workspace), R71 (queries.js, AdvancedMemory.ts), and R72 (core operations: matts.ts, consolidate.ts, distill.ts, backend-selector.ts, embeddings.ts).

## Rationale

The smart priority queue has ~177 files remaining (post-R72 estimate). Of these, **12 are CONNECTED** — all in the agentic-flow ReasoningBank TypeScript layer. These files emerged as CONNECTED after R71 discovered dependency edges linking them to files that R71/R72 deep-read (consolidate.ts, distill.ts, matts.ts, backend-selector.ts, embeddings.ts).

These 12 files form the **outer ring** of the ReasoningBank TS implementation: the barrel exports (index.ts/index.js), configuration layer (config.ts/config.js), utility functions (embeddings.js, pii-scrubber.js, mmr.ts), DB schema definitions (schema.ts/schema.js), the WASM adapter bridge, a hooks integration point (post-task.ts), and a verdict judge (judge.js). Together they answer: is the ReasoningBank TS layer a coherent system with a usable API surface, or a collection of disconnected files around the genuine Rust core?

This is a lighter session (~1,462 LOC) but architecturally decisive — it completes the full ReasoningBank map across both Rust and TypeScript.

## Target: 12 files, ~1,462 LOC

---

### Cluster A: Entry Points & WASM Bridge (4 files, ~577 LOC)

The public API surface of the ReasoningBank TS layer. index.ts and index.js are the barrel exports — they determine what the ReasoningBank actually exposes to consumers. wasm-adapter.ts bridges to the Rust ReasoningBank workspace (R67: 88-92%). post-task.ts is the hooks integration that triggers consolidation after task completion.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 1 | 10827 | `agentic-flow/src/reasoningbank/index.ts` | 136 | ← bypasses backend-selector.ts (DEEP). Barrel export — what does ReasoningBank expose? |
| 2 | 10826 | `agentic-flow/src/reasoningbank/index.js` | 101 | ← imports backend-selector.ts (DEEP). JS barrel — duplicate of index.ts or different API? |
| 3 | 10846 | `agentic-flow/src/reasoningbank/wasm-adapter.ts` | 170 | ← imports backend-selector.ts (DEEP). WASM bridge to Rust workspace? |
| 4 | 10823 | `agentic-flow/src/reasoningbank/hooks/post-task.ts` | 128 | ← used-by consolidate.ts (DEEP). Hooks integration — triggers memory consolidation? |

**Full paths**:
1. `~/repos/agentic-flow/agentic-flow/src/reasoningbank/index.ts`
2. `~/repos/agentic-flow/agentic-flow/src/reasoningbank/index.js`
3. `~/repos/agentic-flow/agentic-flow/src/reasoningbank/wasm-adapter.ts`
4. `~/repos/agentic-flow/agentic-flow/src/reasoningbank/hooks/post-task.ts`

**Key questions**:
- `index.ts` (136 LOC): The dependency edge says "bypasses" backend-selector.ts — does it hardcode a backend instead of using the selector? What does the barrel export? Does it compose matts+consolidate+distill into a coherent API or just re-export them individually? Is this the intended entry point for consumers?
- `index.js` (101 LOC): Is this a compiled version of index.ts or a separate JS implementation? Does it import different modules? R71 found queries.js (100%) was a genuine independent JS layer — does index.js follow the same pattern?
- `wasm-adapter.ts` (170 LOC): WASM scoreboard is 9 genuine vs 5 theatrical (64%). Does this bridge to the Rust ReasoningBank workspace (R67) via wasm-bindgen/wasm-pack? Or is it another theatrical WASM that imports but never calls? Does it use backend-selector.ts to route between WASM and native JS backends?
- `post-task.ts` (128 LOC): R19 found hook-pipeline 100% CLOSED. R51 found hooks-integration.ts at 78-82%. Does post-task.ts trigger memory consolidation (consolidate.ts) after each task? Is it registered as a Claude Code hook or an internal hook? Does it store trajectories via matts.ts?

---

### Cluster B: Utils Layer (5 files, ~704 LOC)

The utility functions that support the ReasoningBank core operations. config.ts/config.js manage configuration. embeddings.js is a JS embedding utility (13th hash-based?). pii-scrubber.js handles data sanitization. mmr.ts implements Maximal Marginal Relevance for diverse retrieval.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 5 | 10839 | `agentic-flow/src/reasoningbank/utils/config.ts` | 241 | ← imports consolidate.ts (DEEP). Config layer — what settings? |
| 6 | 10838 | `agentic-flow/src/reasoningbank/utils/config.js` | 170 | ← imports distill.ts + embeddings.ts (DEEP). JS config — duplicate or different? |
| 7 | 10840 | `agentic-flow/src/reasoningbank/utils/embeddings.js` | 114 | ← imports distill.ts (DEEP). JS embeddings — 13th hash-based? Or real? |
| 8 | 10844 | `agentic-flow/src/reasoningbank/utils/pii-scrubber.js` | 99 | ← imports distill.ts (DEEP). PII scrubbing — genuine data sanitization? |
| 9 | 10843 | `agentic-flow/src/reasoningbank/utils/mmr.ts` | 80 | ← imports consolidate.ts (DEEP). Maximal Marginal Relevance — genuine diversity algorithm? |

**Full paths**:
5. `~/repos/agentic-flow/agentic-flow/src/reasoningbank/utils/config.ts`
6. `~/repos/agentic-flow/agentic-flow/src/reasoningbank/utils/config.js`
7. `~/repos/agentic-flow/agentic-flow/src/reasoningbank/utils/embeddings.js`
8. `~/repos/agentic-flow/agentic-flow/src/reasoningbank/utils/pii-scrubber.js`
9. `~/repos/agentic-flow/agentic-flow/src/reasoningbank/utils/mmr.ts`

**Key questions**:
- `config.ts` (241 LOC): What ReasoningBank settings are configurable? Does it define database paths, embedding dimensions, consolidation thresholds? Does it reference the .swarm/memory.db path from R71's queries.js (100%)? Is it the authoritative config or one of many?
- `config.js` (170 LOC): JS version — compiled from config.ts or independent? Imports both distill.ts AND embeddings.ts — does it configure the distillation pipeline? Are there hardcoded values or genuine environment-driven config?
- `embeddings.js` (114 LOC): Hash-based embedding count is at 12+. Does this use hash-based embeddings (13th instance) or real vector embeddings? R71 found queries.js uses REAL Float32Array embeddings — does embeddings.js share that approach or regress to hashing? Does it integrate with embedding-service.ts (R51: 75-80%)?
- `pii-scrubber.js` (99 LOC): Genuine PII detection (regex patterns for SSN, email, phone, credit card)? Or a stub that returns input unchanged? Does it integrate with the aidefence system? At 99 LOC, could be a genuine utility.
- `mmr.ts` (80 LOC): Maximal Marginal Relevance is a well-known algorithm for diverse retrieval. Does this implement real MMR (cosine similarity + lambda diversity tradeoff)? Or is it a placeholder? At 80 LOC, a genuine MMR implementation is feasible. Does consolidate.ts use it for selecting diverse memories?

---

### Cluster C: Core Judge + DB Schema (3 files, ~223 LOC)

The verdict judge from ReasoningBank core and the DB schema definitions. judge.js implements trajectory verdict classification. schema.ts/schema.js define the database tables — do they match the 8-table schema from R71's queries.js (100%)?

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 10 | 10812 | `agentic-flow/src/reasoningbank/core/judge.js` | 129 | ← imports distill.ts (DEEP). Verdict judgment — trajectory success/failure classification? |
| 11 | 10821 | `agentic-flow/src/reasoningbank/db/schema.ts` | 89 | ← imports matts.ts (DEEP). DB schema — matches queries.js 8-table schema? |
| 12 | 10820 | `agentic-flow/src/reasoningbank/db/schema.js` | 5 | ← imports distill.ts (DEEP). JS schema — at 5 LOC, likely a re-export |

**Full paths**:
10. `~/repos/agentic-flow/agentic-flow/src/reasoningbank/core/judge.js`
11. `~/repos/agentic-flow/agentic-flow/src/reasoningbank/db/schema.ts`
12. `~/repos/agentic-flow/agentic-flow/src/reasoningbank/db/schema.js`

**Key questions**:
- `judge.js` (129 LOC): R67 found ReasoningBank Rust core genuinely architected with verdict judgment. Does the TS judge.js implement the same verdict logic (success/failure/partial classification of trajectories)? Does it use distill.ts for knowledge extraction from judged trajectories? Is this connected to the ReasoningBank MCP "store" tool (R67: 72%)?
- `schema.ts` (89 LOC): R71 found queries.js has an 8-table schema with real SQL. Does schema.ts define those same tables as TypeScript types/interfaces? Does it use Zod, io-ts, or raw interfaces? Does it import matts.ts types for trajectory table definitions? This could be the type-safe layer over queries.js's raw SQL.
- `schema.js` (5 LOC): At 5 LOC, almost certainly a one-line re-export or compiled barrel. Does it just `module.exports = require('./schema.ts')` or does it export a subset?

---

## Expected Outcomes

1. **CONNECTED tier CLEARED** — all 12 files read, zero blind dependency edges remain. TRUE MILESTONE this time.
2. **ReasoningBank TS layer COMPLETE**: entry points + core + utils + DB + hooks — full picture of the TS implementation alongside the Rust workspace (R67-R68)
3. **WASM adapter verdict**: Does wasm-adapter.ts bridge to Rust ReasoningBank? (10th genuine or 6th theatrical)
4. **Hash-based embeddings**: Does embeddings.js push count to 13+?
5. **Public API surface**: Do index.ts/index.js expose a coherent API or a grab-bag of re-exports?
6. **PII scrubbing**: Genuine data sanitization or stub?
7. **MMR algorithm**: Real diverse retrieval or placeholder?
8. **DEEP files**: ~1,170 → ~1,182 (post-R72 estimate + 12)
9. **Priority queue**: ~177 → ~165, all PROXIMATE/NEARBY/DOMAIN_ONLY

## DB Update Template

```javascript
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const sessionId = 73;
const today = new Date().toISOString().slice(0, 10);

// File IDs for this session:
// 10827: index.ts (136 LOC) — reasoningbank entry point, CONNECTED
// 10826: index.js (101 LOC) — reasoningbank entry point, CONNECTED
// 10846: wasm-adapter.ts (170 LOC) — reasoningbank WASM bridge, CONNECTED
// 10823: post-task.ts (128 LOC) — reasoningbank hooks, CONNECTED
// 10839: config.ts (241 LOC) — reasoningbank utils, CONNECTED
// 10838: config.js (170 LOC) — reasoningbank utils, CONNECTED
// 10840: embeddings.js (114 LOC) — reasoningbank utils, CONNECTED
// 10844: pii-scrubber.js (99 LOC) — reasoningbank utils, CONNECTED
// 10843: mmr.ts (80 LOC) — reasoningbank utils, CONNECTED
// 10812: judge.js (129 LOC) — reasoningbank core, CONNECTED
// 10821: schema.ts (89 LOC) — reasoningbank db, CONNECTED
// 10820: schema.js (5 LOC) — reasoningbank db, CONNECTED
```

## Domain Tags

- Files 1-12 → `memory-and-learning` (already tagged)

## Isolation Check

No selected files are in known-isolated subtrees. All 12 files are in the agentic-flow ReasoningBank directory (`agentic-flow/src/reasoningbank/`) which has 63 nearby DEEP files — the second-densest cluster in the queue. The agentic-flow-rust package is CONNECTED (not ISOLATED). All files safe to read.

---

## Synthesis Doc Update Protocol (ADR-040)

**MANDATORY**: After all files are read and findings inserted into the DB, update the relevant `domains/*/analysis.md` files following the ADR-040 in-place protocol. Reference: `domains/memory-and-learning/analysis.md` for canonical structure.

### Rules for Each Section

| Section | Action | NEVER Do |
|---------|--------|----------|
| **1. Current State Summary** | REWRITE in-place to reflect current state | Append session narrative |
| **2. File Registry** | ADD new rows to existing subsystem tables, UPDATE rows if re-read | Duplicate rows, create per-session file tables |
| **3. Findings Registry** | ADD new findings with next sequential ID (C{max+1}, H{max+1}) to 3a/3b | Create `### R73 Findings` blocks, re-list old findings, restart ID numbering |
| **4. Positives Registry** | ADD new positives with session tag | Re-list existing positives |
| **5. Subsystem Sections** | UPDATE existing sections, CREATE new ones by topic | Create per-session narrative blocks |
| **8. Session Log** | APPEND 2-5 line entry for this session | Put findings here, write full narratives |

### Finding ID Assignment

Before adding findings, check the current max ID in the target domain's analysis.md:
- Section 3a: find last `| C{N} |` row → new CRITICALs start at C{N+1}
- Section 3b: find last `| H{N} |` row → new HIGHs start at H{N+1}

**ID format**: `| {ID} | **{short title}** — {description} | {file(s)} | R{session} | Open |`

### Anti-Patterns (NEVER do these)

- **NEVER** create `### R73 Findings (Session date)` blocks outside Section 3
- **NEVER** append findings after Section 8
- **NEVER** create `### R73 Full Session Verdict` blocks
- **NEVER** use finding IDs that collide with existing ones (always check max first)
- **NEVER** re-list findings from previous sessions

### Synthesis Update Checklist

- [ ] Section 1 rewritten with updated state
- [ ] New file rows added to Section 2 (correct subsystem table)
- [ ] New findings added to Section 3a/3b with sequential IDs
- [ ] New positives added to Section 4 (if any)
- [ ] Relevant subsystem sections in Section 5 updated
- [ ] Session log entry appended to Section 8 (2-5 lines max)
- [ ] No per-session finding blocks created anywhere
