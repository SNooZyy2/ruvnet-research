# R137 Execution Plan: MCP Tool Chain — From Tool Call to Backend

**Date**: 2026-03-01
**Session ID**: 137
**Focus**: Trace the MCP tool registration and dispatch chain — from where tools are defined, through the server bootstrap, to what code actually executes when a tool is called
**Strategic value**: MCP is the primary interface between claude-flow and the outside world. The 256 tools registered in V3 are the product's feature surface. Understanding this chain reveals whether advertised features actually connect to working backends.

## Rationale

We've never read the MCP server bootstraps or tool registration hubs. These files determine:
1. What tools users can call
2. What code those tools invoke
3. Whether tool handlers call into working backends or broken adapters

This session traces the full chain: tool definition → server registration → handler execution. By combining this with ML-A (CLI entrypoints) and ML-B (memory layer), we get the complete picture of how a user interacts with the system.

**Dependency**: Best run AFTER ML-A (R134), so you know how CLIs bootstrap the MCP servers.

## Target: 8 files, ~8,069 LOC

---

### Cluster A: claude-flow V3 MCP Stack (4 files, ~2,769 LOC)

The V3 MCP server and its tool registration. This is the MCP server that runs when you do `claude-flow mcp start`.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 1 | — | `v3/mcp/tools/index.ts` | 445 | 256 tools from 12 groups via `getAllTools()`. The deepest aggregation point. |
| 2 | — | `v3/mcp/server.ts` | 792 | V3 MCP server bootstrap — ToolRegistry + SessionManager + ConnectionPool. |
| 3 | — | `v3/@claude-flow/mcp/src/server.ts` | 1,134 | Library MCP server with 9 sub-registries including SamplingManager. |
| 4 | — | `v3/@claude-flow/cli/src/commands/index.ts` | 398 | 30+ CLI commands lazy-loaded. Full V3 command surface. |

**Full paths**:
1. `~/repos/claude-flow/v3/mcp/tools/index.ts`
2. `~/repos/claude-flow/v3/mcp/server.ts`
3. `~/repos/claude-flow/v3/@claude-flow/mcp/src/server.ts`
4. `~/repos/claude-flow/v3/@claude-flow/cli/src/commands/index.ts`

**Key questions**:
- `tools/index.ts` (445 LOC): All 12 tool groups listed — which ones import from real backends vs stubs? Do `sonaTools` and `federationTools` actually connect to SONA/federation or are they facades?
- `server.ts` (792 LOC): How does the server initialize? Does it call the memory-initializer from ML-B? Does it wait for backend detection before registering tools?
- `mcp/src/server.ts` (1,134 LOC): What's the SamplingManager? Does the LLMProvider interface connect to actual model routing? How does it differ from the CLI-level server?
- `commands/index.ts` (398 LOC): What's the full command tree? Does the `ruvectorCommand` (line 127) actually call into ruvector setup? Which commands are placeholders?

---

### Cluster B: agentic-flow MCP Server (1 file, 812 LOC)

The agentic-flow MCP server — the predecessor to claude-flow's V3 server. Understanding this reveals evolution.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 5 | — | `agentic-flow/src/mcp/standalone-stdio.ts` | 812 | FastMCP server with 66+ agent types registered. |

**Full paths**:
5. `~/repos/agentic-flow/agentic-flow/src/mcp/standalone-stdio.ts`

**Key questions**:
- Does it use FastMCP's `server.addTool()` pattern or a custom registration?
- Does it directly import ruvector, or does it shell out?
- How many tools are actually functional vs placeholder registrations?
- Does it share code with the V3 server or is it independent?

---

### Cluster C: claude-flow V2 MCP Server (1 file, 646 LOC)

The V2 MCP server — integrates 3 distinct tool sets. Useful for understanding evolutionary path.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 6 | — | `v2/src/mcp/server.ts` | 646 | V2 server with createClaudeFlowTools + createSwarmTools + createRuvSwarmTools. |

**Full paths**:
6. `~/repos/claude-flow/v2/src/mcp/server.ts`

**Key questions**:
- Does V2 server have working tool handlers that V3 lost?
- Do `createRuvSwarmTools` actually call ruv-swarm?
- Is V2 simpler but more functional than V3?

---

### Cluster D: ruvector MCP Server + ruvector Setup Command (2 files, ~3,791 LOC)

The ruvector-specific MCP server from the npm package, plus the claude-flow ruvector setup command.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 7 | 7702 | `npm/packages/ruvector/bin/mcp-server.js` | 3,007 | (Also in ML-A — if already read there, SKIP. If ML-D runs first, read here.) |
| 8 | 15447 | `v3/@claude-flow/cli/src/commands/ruvector/setup.ts` | 784 | How claude-flow sets up ruvector integration. |

**Full paths**:
7. `~/repos/ruvector/npm/packages/ruvector/bin/mcp-server.js`
8. `~/.npm-global/lib/node_modules/@claude-flow/cli/src/commands/ruvector/setup.ts`

**Key questions**:
- `mcp-server.js` (3,007 LOC): (See ML-A questions if not yet read)
- `setup.ts` (784 LOC): What does `claude-flow ruvector setup` actually do? Does it install the native binary? Does it configure the backend factory? Does it test the connection?

---

## Expected Outcomes

1. **Tool-to-backend map**: For key MCP tools (memory_store, memory_search, agent_spawn), what code path actually executes?
2. **Real vs facade tools**: Of 256 registered tools, how many have working handlers?
3. **Server evolution**: V2 → agentic-flow → V3 — did integration get better or worse?
4. **ruvector setup**: What does the setup command do — is there a one-command path to native HNSW?
5. **Combined with ML-A+B+C**: Complete user journey from CLI → MCP server → tool handler → memory system → backend → algorithm

## DB Update Template

```javascript
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const sessionId = 137;
const today = new Date().toISOString().slice(0, 10);

// File IDs for this session:
// Most claude-flow V3 files may not have DB entries — check and insert if needed
// 15447: setup.ts (784 LOC) — claude-flow-cli, NOT_TOUCHED
// 7702: mcp-server.js (3,007 LOC) — ruvector npm (skip if read in ML-A)
```

## Domain Tags

- Cluster A → `production-infra` + `agentdb-integration`
- Cluster B → `agentic-flow` + `production-infra`
- Cluster C → `production-infra`
- Cluster D → `ruvector` + `production-infra`

## Isolation Check

All files are in published packages or the main CLI. No isolation concerns.

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
