# R134 Execution Plan: The Front Door — CLI Entrypoints

**Date**: 2026-03-01
**Session ID**: 134
**Focus**: Read the actual CLI entrypoints — the files that execute when a user types `ruvector`, `claude-flow`, `rvlite`, `ruvllm`, or `ruflo`
**Strategic value**: These files answer the most fundamental question: "what code path runs when someone uses these tools?" After 133 sessions reading library internals, we have never read the front door.

## Rationale

The `smart_priority_gaps` view systematically deprioritized CLI entrypoints because they live in `bin/` directories disconnected from the algorithm files that dominate our DEEP coverage. The ruvector CLI alone is 7,357 LOC — this is not a stub. It's a substantial application that wires together the subsystems we've individually analyzed.

CLI files are typically command-dispatch boilerplate, so effective LOC requiring deep analysis is lower than raw line count. This session is scannable in one sitting despite the high nominal LOC.

**NOTE**: This is a Middle Layer (ML) session series. These sessions use a different approach from standard research sessions — the goal is to trace integration paths, not score individual file quality. Findings focus on: does the wiring work? Which code path actually executes? Where does it connect to the Rust/TS libraries we've already analyzed?

## Target: 7 files, ~13,450 LOC

---

### Cluster A: ruvector npm CLI (2 files, ~10,364 LOC)

The ruvector npm package is the primary user-facing product. These two files define what `npx ruvector` and the ruvector MCP server actually do.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 1 | 7701 | `npm/packages/ruvector/bin/cli.js` | 7,357 | THE ruvector CLI. What `npx ruvector` executes. |
| 2 | 7702 | `npm/packages/ruvector/bin/mcp-server.js` | 3,007 | THE ruvector MCP server. How MCP tools reach ruvector. |

**Full paths**:
1. `~/repos/ruvector/npm/packages/ruvector/bin/cli.js`
2. `~/repos/ruvector/npm/packages/ruvector/bin/mcp-server.js`

**Key questions**:
- `cli.js` (7,357 LOC): What commands does it expose? Does it call into the native Rust NAPI layer or the broken TS adapters? Does it use `VectorDBWrapper` (which works) or `RuVectorBackend` (which is broken)? Does it import `@ruvector/core` or the umbrella package? Is there a `--native` flag? Does it actually invoke HNSW operations or is it a CRUD wrapper over simpler storage?
- `mcp-server.js` (3,007 LOC): How many MCP tools does it register? Do they call the same code paths as the CLI? Does it use the `agentdb-mcp-server.ts` we found (2,367 LOC) or a separate implementation? Does it initialize the EmbeddingService (R20 root cause: never initialized)?

---

### Cluster B: claude-flow CLI Entrypoints (3 files, ~395 LOC)

The claude-flow/ruflo entry layer. Small files — likely thin dispatchers into the V3 command system.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 3 | — | `v3/@claude-flow/cli/bin/cli.js` | 156 | THE claude-flow v3 entrypoint. |
| 4 | — | `v3/@claude-flow/cli/bin/mcp-server.js` | 189 | THE claude-flow MCP server entrypoint. |
| 5 | — | `ruflo/bin/ruflo.js` | 50 | The ruflo rebrand — does it forward to v3 or something else? |

**Full paths**:
3. `~/repos/claude-flow/v3/@claude-flow/cli/bin/cli.js`
4. `~/repos/claude-flow/v3/@claude-flow/cli/bin/mcp-server.js`
5. `~/repos/claude-flow/ruflo/bin/ruflo.js`

**Key questions**:
- `cli.js` (156 LOC): Does it bootstrap the V3 command registry (`commands/index.ts`, 398 LOC)? Does it initialize the memory system? Does it load ruvector?
- `mcp-server.js` (189 LOC): Does it use the V3 MCP server library or roll its own? How does it register the 256 tools we found?
- `ruflo.js` (50 LOC): Is this a symlink/redirect to claude-flow v3, or an independent entry? Does it add new functionality?

---

### Cluster C: Satellite CLIs (2 files, ~2,691 LOC)

Secondary CLI entrypoints for rvlite and ruvllm — may reveal alternative integration paths.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 6 | 8119 | `npm/packages/rvlite/bin/cli.js` | 1,686 | rvlite CLI — lightweight vector ops, may have working HNSW path. |
| 7 | 7980 | `npm/packages/ruvllm/bin/cli.js` | 1,005 | ruvllm CLI — LLM integration entrypoint. |

**Full paths**:
6. `~/repos/ruvector/npm/packages/rvlite/bin/cli.js`
7. `~/repos/ruvector/npm/packages/ruvllm/bin/cli.js`

**Key questions**:
- `rvlite/cli.js` (1,686 LOC): Does rvlite bypass the broken adapters and talk to a simpler backend? R38 found rvlite at 82-86% — does the CLI actually exercise those capabilities?
- `ruvllm/cli.js` (1,005 LOC): Does it expose model loading, inference, or training? Does it connect to the Rust ruvllm crate or is it a JS-only wrapper?

---

## Expected Outcomes

1. **Definitive answer**: Do the CLIs use the broken TS adapter layer (RuVectorBackend) or a working path (VectorDBWrapper, direct native calls)?
2. **MCP tool map**: What tools does the ruvector MCP server actually register, and what code do they call?
3. **claude-flow bootstrap**: How does `claude-flow` start up — does it initialize memory, load ruvector, connect to AgentDB?
4. **ruflo identity**: Is ruflo a rebrand or a fork?
5. **Alternative paths**: Do rvlite/ruvllm CLIs have their own working integration?

## DB Update Template

```javascript
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const sessionId = 134;
const today = new Date().toISOString().slice(0, 10);

// File IDs for this session:
// 7701: cli.js (7,357 LOC) — ruvector npm CLI
// 7702: mcp-server.js (3,007 LOC) — ruvector MCP server
// (no DB ID): claude-flow/v3/cli/bin/cli.js (156 LOC) — claude-flow-cli package
// (no DB ID): claude-flow/v3/cli/bin/mcp-server.js (189 LOC) — claude-flow-cli package
// (no DB ID): claude-flow/ruflo/bin/ruflo.js (50 LOC) — claude-flow package
// 8119: rvlite/bin/cli.js (1,686 LOC) — ruvector npm rvlite
// 7980: ruvllm/bin/cli.js (1,005 LOC) — ruvector npm ruvllm

// NOTE: claude-flow bin files may not have DB entries. Check:
// db.prepare('SELECT id FROM files WHERE relative_path LIKE ?').get('%cli/bin/cli.js%');
// If missing, insert them into files table before recording reads.
```

## Domain Tags

- Files 7701, 7702 → `production-infra` (CLI/MCP layer)
- Files 8119, 7980 → `production-infra`
- claude-flow files → `production-infra` + `agentdb-integration`

## Isolation Check

All files are in published npm packages — by definition not isolated. The ruvector npm packages are the user-facing surface of the project.

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
