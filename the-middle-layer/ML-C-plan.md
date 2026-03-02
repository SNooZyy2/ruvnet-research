# R136 Execution Plan: Ghost DEEP Files + Rust Integration Hubs

**Date**: 2026-03-01
**Session ID**: 136
**Focus**: Read the "ghost DEEP" files (marked DEEP, 0 lines read) plus the cross-repo integration bridges that wire ruvector Rust to claude-flow/agentdb TypeScript
**Strategic value**: Fixes metadata inflation in the DB (at least 6K phantom LOC at DEEP), and reads the Rust-side integration files that parallel the TS memory layer from ML-B. Together ML-B + ML-C give both sides of the Rust↔TS bridge.

## Rationale

Pass 3 of the connection-layer search discovered files marked DEEP in the research DB that have 0 lines actually read. These inflate our DEEP count and hide gaps in integration coverage. Three are in `ruvllm/src/claude_flow/` — the Rust-side integration with claude-flow that complements the TS memory layer.

Additionally, the `agentdb-mcp-server.ts` (2,367 LOC) is the single largest MCP registration hub — 10 controllers wired to MCP protocol — and has never actually been read despite being marked DEEP. The `factory.ts` (344 LOC) is the backend selection logic that decides whether AgentDB uses ruvector-native, RVF, or HNSWLib. The `sona-agentdb-integration.ts` (463 LOC) is the explicit cross-repo bridge between @ruvector/sona and agentdb.

**Dependency**: Best run AFTER ML-B (R135), so you can compare the TS memory layer with these Rust integration files.

## Target: 7 files, ~7,165 LOC

---

### Cluster A: Ghost DEEP — Rust claude_flow Integration (3 files, ~3,950 LOC)

These files are in `ruvllm/src/claude_flow/` — the Rust-side of the claude-flow integration. All marked DEEP with 0 lines read. Must be re-classified after actual reading.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 1 | — | `crates/ruvllm/src/claude_flow/hnsw_router.rs` | 1,287 | Semantic HNSW task router. Imports ruvector_core::hnsw + sona. Ghost DEEP. |
| 2 | — | `crates/ruvllm/src/claude_flow/claude_integration.rs` | 1,341 | Primary Rust-side claude-flow integration. Ghost DEEP. |
| 3 | — | `crates/ruvllm/src/claude_flow/model_router.rs` | 1,322 | Model routing — Rust side of ADR-008. Ghost DEEP. |

**Full paths**:
1. `~/repos/ruvector/crates/ruvllm/src/claude_flow/hnsw_router.rs`
2. `~/repos/ruvector/crates/ruvllm/src/claude_flow/claude_integration.rs`
3. `~/repos/ruvector/crates/ruvllm/src/claude_flow/model_router.rs`

**Key questions**:
- `hnsw_router.rs` (1,287 LOC): Does it actually instantiate HnswIndex from ruvector-core? Does it implement the "150x faster" routing claim? Does it integrate with SONA trajectories? Is there a working search + learn loop?
- `claude_integration.rs` (1,341 LOC): What does "claude integration" mean in Rust? Does this compile into the NAPI bridge? Does it expose functions to the TS layer? Is it consumed by the ruvector-node crate?
- `model_router.rs` (1,322 LOC): Is this the Rust implementation of the 3-tier model routing (ADR-008)? Does it connect to the TS `model-route` hook? Or is it a parallel implementation?

---

### Cluster B: Ghost DEEP — AgentDB MCP Server (1 file, 2,367 LOC)

The single largest MCP registration hub in the codebase. 10 AgentDB controllers wired to one MCP protocol server. Marked DEEP with 0 lines read.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 4 | — | `packages/agentdb/src/mcp/agentdb-mcp-server.ts` | 2,367 | 10 controllers: CausalMemory, Reflexion, Skills, NightlyLearner, EmbeddingService, ReasoningBank, BatchOps, security. Ghost DEEP. |

**Full paths**:
4. `~/repos/agentic-flow/packages/agentdb/src/mcp/agentdb-mcp-server.ts`

**Key questions**:
- Does it initialize the EmbeddingService? (R20 root cause)
- Does it use the backend factory for vector search?
- How many MCP tools does it actually register?
- Does it handle the "Pipeline 1" path (xenova/transformers → HNSWLib) or the "Pipeline 2" path (native ruvector)?
- Is this the MCP server that claude-flow's MCP tool calls actually hit?

---

### Cluster C: Backend Factory + Cross-Repo Bridge (2 files, ~807 LOC)

The factory that decides at runtime which backend AgentDB uses, and the explicit sona-agentdb bridge.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 5 | — | `packages/agentdb/src/backends/factory.ts` | 344 | Backend selector: RuVector native > RuVector WASM > RVF > HNSWLib. |
| 6 | — | `agentic-flow/src/services/sona-agentdb-integration.ts` | 463 | Explicit cross-repo bridge: @ruvector/sona + agentdb. |

**Full paths**:
5. `~/repos/agentic-flow/packages/agentdb/src/backends/factory.ts`
6. `~/repos/agentic-flow/agentic-flow/src/services/sona-agentdb-integration.ts`

**Key questions**:
- `factory.ts` (344 LOC): How does `detectBackends()` work? Does it dynamic-import `@ruvector/core`? What happens when native detection fails — does it gracefully fall back to HNSWLib? Is this the code path that ML-B's `memory-initializer.ts` calls?
- `sona-agentdb-integration.ts` (463 LOC): Does it import `SonaEngine` from `@ruvector/sona` (WASM)? Does SONA actually connect to AgentDB's vector store? Is the "150x-12,500x" performance claim sourced from here? Is this integration active in production or dead code?

---

### Cluster D: DB Cleanup — Ghost DEEP Correction (1 file, 0 LOC new)

After reading Clusters A-C, fix the DB to reflect actual read status.

| # | Action | Files Affected |
|---|--------|---------------|
| 7 | Correct DEEP→actual depth | hnsw_router.rs, claude_integration.rs, model_router.rs, agentdb-mcp-server.ts + any others found with 0 lines_read |

**Cleanup query**:
```javascript
// After reading, run this to find ALL ghost DEEP files
const ghosts = db.prepare(`
  SELECT f.id, f.relative_path, f.depth, f.lines_read, f.loc
  FROM files f
  WHERE f.depth = 'DEEP' AND (f.lines_read = 0 OR f.lines_read IS NULL)
`).all();
console.log('Ghost DEEP files:', JSON.stringify(ghosts, null, 2));
// Then correct each to actual depth based on this session's read
```

---

## Expected Outcomes

1. **Rust integration truth**: Do hnsw_router.rs and claude_integration.rs contain working integration, or are they the Rust equivalent of the broken TS adapters?
2. **MCP server anatomy**: What code path does AgentDB-MCP-Server actually execute — native or fallback?
3. **Factory behavior**: What's the actual runtime backend selection in production?
4. **SONA bridge**: Is SONA connected to AgentDB or orphaned?
5. **DB accuracy**: Ghost DEEP files corrected, true DEEP count recalculated
6. **Combined with ML-B**: Complete picture of both TS and Rust sides of the integration layer

## DB Update Template

```javascript
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const sessionId = 136;
const today = new Date().toISOString().slice(0, 10);

// File IDs for this session (look up actual IDs — some may need paths checked):
// Ghost DEEP files — update BOTH depth and lines_read after actual reading
// factory.ts — check if already in DB, may need path lookup
// sona-agentdb-integration.ts — check if in DB

// Ghost DEEP correction query (run after all reads):
// UPDATE files SET depth = ?, lines_read = ? WHERE id = ? AND lines_read = 0;
```

## Domain Tags

- Cluster A (Rust files) → `model-routing` + `ruvector`
- Cluster B (MCP server) → `agentdb-integration` + `production-infra`
- Cluster C (factory, sona bridge) → `agentdb-integration` + `memory-and-learning`

## Isolation Check

All files are in connected packages (ruvector-rust, agentic-flow-rust, agentic-flow). No isolation concerns.

---

## Synthesis Doc Update Protocol (ADR-040)

**MANDATORY**: After all files are read and findings inserted into the DB, update the relevant `domains/*/analysis.md` files following the ADR-040 in-place protocol. Reference: `domains/memory-and-learning/analysis.md` for canonical structure.

### Rules for Each Section

| Section | Action | NEVER Do |
|---------|--------|----------|
| **1. Current State Summary** | REWRITE in-place to reflect current state | Append session narrative |
| **2. File Registry** | ADD new rows to existing subsystem tables, UPDATE rows if re-read | Duplicate rows, create per-session file tables |
| **3. Findings Registry** | ADD new findings with next sequential ID (C{max+1}, H{max+1}) to 3a/3b | Create `### RXX Findings` blocks, re-list old findings, restart ID numbering |
| **4. Positives Registry** | ADD new positives with session tag | Re-list existing positives |
| **5. Subsystem Sections** | UPDATE existing sections, CREATE new ones by topic | Create per-session narrative blocks |
| **8. Session Log** | APPEND 2-5 line entry for this session | Put findings here, write full narratives |

### Finding ID Assignment

Before adding findings, check the current max ID in the target domain's analysis.md:
- Section 3a: find last `| C{N} |` row → new CRITICALs start at C{N+1}
- Section 3b: find last `| H{N} |` row → new HIGHs start at H{N+1}

**ID format**: `| {ID} | **{short title}** — {description} | {file(s)} | R{session} | Open |`

### Anti-Patterns (NEVER do these)

- **NEVER** create `### R{N} Findings (Session date)` blocks outside Section 3
- **NEVER** append findings after Section 8
- **NEVER** create `### R{N} Full Session Verdict` blocks
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
