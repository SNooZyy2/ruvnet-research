# Claude-Flow Swarm Functionality: How It Actually Works

**Analysis Date**: 2026-02-14
**Scope**: Complete analysis of swarm coordination, queen coordinator, and agent-to-agent interaction in our local claude-flow implementation
**Method**: 14-agent research swarm (5 initial + 4 gap-finding + 5 deep-dive) analyzing CLI commands, MCP tools, agent templates, claims/coordination, end-to-end flows, worker systems, learning pipeline, v3 source packages, real-world usage evidence, hooks internals, MCP runtime, headless/container execution, settings.json wiring, and plugin/guidance/skills/commands systems
**Files Analyzed**: ~60 source files, ~8 agent templates, ~45 helper scripts, 33 skills, 90+ commands (~50,000+ lines)

---

## Executive Summary

Claude-flow's "swarm" system operates on **three distinct layers** that are commonly conflated:

| Layer | What It Is | What It Does |
|-------|-----------|--------------|
| **Layer 1: Claude Code Task Tool** | Built-in Claude Code feature | Actually spawns parallel agent instances. **This IS the real swarm.** |
| **Layer 2: claude-flow CLI/MCP** | JSON file state management | Writes metadata to `.swarm/` and `.claude-flow/`. No processes spawned. |
| **Layer 3: Agent Templates** | Prompt engineering | `.claude/agents/*.md` files that define agent personas. Educational TypeScript examples, not executed. |

**The critical insight**: The actual multi-agent parallelism comes entirely from Claude Code's native Task tool with `run_in_background: true`. Everything claude-flow adds on top is metadata, hooks, and prompt engineering — valuable for organization, but not for actual distributed coordination.

---

## Table of Contents

1. [The Two Agent Systems](#1-the-two-agent-systems)
2. [What swarm init Actually Does](#2-what-swarm-init-actually-does)
3. [The Queen Coordinator Myth](#3-the-queen-coordinator-myth)
4. [Agent-to-Agent Communication](#4-agent-to-agent-communication)
5. [The Claims System](#5-the-claims-system)
6. [Consensus Mechanisms](#6-consensus-mechanisms)
7. [End-to-End Flow Trace](#7-end-to-end-flow-trace)
8. [What Actually Provides Value](#8-what-actually-provides-value)
9. [Architecture Diagram](#9-architecture-diagram)
10. [Gap Analysis](#10-gap-analysis)
11. [Recommendations](#11-recommendations)

---

## 1. The Two Agent Systems

There are **two completely separate "agent" systems** that share no code and serve different purposes:

### System A: Claude Code Task Tool (THE REAL SWARM)

- Built into Claude Code (Opus 4.6)
- Spawns isolated Claude instances as background agents
- Parameters: `prompt`, `subagent_type` (matches `.claude/agents/*.md`), `model`, `run_in_background`
- Each agent gets full tool access (Read, Write, Bash, Grep, etc.)
- Agents execute truly in parallel when `run_in_background: true`
- Results collected and returned to parent when all complete
- **No connection to claude-flow CLI whatsoever**

### System B: claude-flow MCP `agent_spawn` Tool (METADATA ONLY)

- MCP tool in `agent-tools.js` (line 134)
- Creates a JSON entry in `.claude-flow/agents/store.json`
- **Does NOT spawn any process, thread, or Claude instance**
- Agent record structure:
  ```json
  {
    "agentId": "agent-1707900000-abc123",
    "agentType": "coder",
    "status": "idle",
    "health": 1.0,
    "taskCount": 0,
    "config": {},
    "createdAt": "2026-02-14T..."
  }
  ```
- Terminating an agent just sets `status: 'terminated'` in JSON

### The Confusion

CLAUDE.md conflates these: "MUST spawn agents matching the suggested roles via Task tool" (System A) alongside "use `claude-flow agent spawn`" (System B). In practice, only System A does real work. System B is bookkeeping.

---

## 2. What `swarm init` Actually Does

### The Command

```bash
claude-flow swarm init --topology hierarchical --max-agents 15 --strategy specialized
```

### What It Claims
- Initialize distributed swarm with topology selection
- Set up communication protocols and consensus mechanisms
- Deploy queen coordinator with worker agents

### What Actually Happens

**Step 1**: Calls MCP tool `swarm_init` (swarm-tools.js lines 19-37):
```javascript
handler: async (input) => {
    return {
        success: true,
        swarmId: `swarm-${Date.now()}`,     // Timestamp-based ID
        topology: input.topology || 'hierarchical-mesh',
        config: {
            currentAgents: 0,               // HARDCODED ZERO
            communicationProtocol: 'message-bus',  // LABEL ONLY
            consensusMechanism: 'majority',        // NOT IMPLEMENTED
        }
    };
}
```

**Step 2**: Writes `.swarm/state.json`:
```json
{
  "id": "swarm-1707900000",
  "topology": "hierarchical",
  "maxAgents": 15,
  "strategy": "specialized",
  "status": "ready"
}
```

**Step 3**: Displays spinner animations and formatted output.

**What is NOT done**:
- No processes spawned
- No network connections opened
- No agent pool initialized
- No communication channels created
- No coordinator started

### Topology Has Zero Behavioral Impact

Available topologies: `hierarchical`, `mesh`, `ring`, `star`, `hybrid`, `hierarchical-mesh`

All topologies are **string labels stored in JSON**. No code paths change based on topology selection. There is no routing, no graph structure, no communication pattern difference. From `coordination-tools.js` header:

```javascript
/**
 * ⚠️ IMPORTANT: These tools provide LOCAL STATE MANAGEMENT.
 * - Topology/consensus state is tracked locally
 * - No actual distributed coordination
 * - Useful for single-machine workflow orchestration
 */
```

---

## 3. The Queen Coordinator Myth

### What the Architecture Claims

```
    👑 QUEEN (Strategic Command)
        │
   ┌────┼────┬────┐
   │    │    │    │
   🔬   💻   📊   🛡️
  RSR  CODE  ANLY  SEC
```

### What Actually Exists

The "queen" is an **agent template** at `.claude/agents/v3/v3-queen-coordinator.md` — a markdown file with instructions for an LLM. It has:

- **Same tools as any other agent**: Read, Grep, Glob, Bash
- **No special privileges**: Cannot spawn, terminate, or redirect other agents
- **No supervisory capability**: Cannot monitor agent progress in real-time
- **No enforcement mechanism**: Cannot prevent agents from ignoring its "commands"

### How the "Hierarchy" Manifests

1. The queen template prompt says: "You are the strategic coordinator. Delegate tasks to workers."
2. Worker template prompts say: "You execute tasks from the coordinator."
3. **Compliance is voluntary** — based entirely on prompt engineering
4. If a worker agent ignores the queen's plan, nothing prevents it

### The TypeScript Code in Templates

Coordinator templates (hierarchical, mesh, adaptive, collective-intelligence) contain **300-800 lines of TypeScript code** showing:
- `AttentionService` with Flash/Multi-Head/Hyperbolic attention
- GraphRoPE position embeddings
- Byzantine node detection algorithms
- CRDT synchronization protocols

**This code is NEVER EXECUTED.** Agents only have Bash tool access, not a TypeScript runtime. The code serves as:
- Conceptual guidance for the LLM to understand coordination patterns
- Prompt engineering to prime sophisticated reasoning
- Implementation examples for what *could* be built

### Coordinator Comparison

| Coordinator | Template Size | TypeScript Example | Actual Capability |
|-------------|--------------|-------------------|-------------------|
| v3-queen-coordinator | 82 lines | None | Read/Grep/Glob/Bash |
| hierarchical-coordinator | 717 lines | 457 lines | Read/Grep/Glob/Bash |
| mesh-coordinator | 970 lines | 638 lines | Read/Grep/Glob/Bash |
| adaptive-coordinator | 1133 lines | 669 lines | Read/Grep/Glob/Bash |
| collective-intelligence | 1002 lines | 803 lines | Read/Grep/Glob/Bash |

All coordinators have **identical actual capabilities**. The differences are only in their prompt instructions.

---

## 4. Agent-to-Agent Communication

### Available Communication Mechanisms

#### Mechanism A: Shared Memory (SQLite) — WORKS
```bash
# Agent A stores finding
claude-flow memory store --key "auth-pattern" --value "Use JWT with refresh" --namespace patterns

# Agent B retrieves it
claude-flow memory search --query "authentication" --namespace patterns
```
- Backed by SQLite at `~/.swarm/memory.db` and `~/.swarm/agentdb.db`
- HNSW vector search for semantic retrieval (via @ruvector/core)
- Persistent across sessions
- **This is the REAL coordination mechanism**

#### Mechanism B: File-Based Messages — EXISTS BUT PASSIVE
```bash
# Agent A sends message (via swarm-hooks.sh)
.claude/helpers/swarm-hooks.sh send "agent_B" "Found schema conflict" "context" "high"
# Creates: .claude-flow/swarm/messages/msg_*.json

# Agent B must explicitly poll
.claude/helpers/swarm-hooks.sh messages 10
# Reads messages addressed to agent_B or broadcast ("*")
```
- **Polling-based only** — no push notifications
- Agents must explicitly call the helper to check messages
- No automatic polling mechanism
- Task agents don't know other agents exist unless told in their prompt

#### Mechanism C: Handoff Protocol — EXISTS BUT MANUAL
```bash
# Agent A initiates handoff
.claude/helpers/swarm-hooks.sh handoff "agent_B" "Complete auth" '{"filesModified":["auth.ts"]}'
# Creates: .claude-flow/swarm/handoffs/ho_*.json

# Agent B accepts
.claude/helpers/swarm-hooks.sh accept-handoff "ho_12345"
# Outputs formatted context
```
- Requires explicit orchestration by agents
- Not automatic — both sides must participate
- Creates JSON files that the accepting agent reads

#### Mechanism D: Pattern Broadcasting — EXISTS BUT OPT-IN
```bash
# Agent broadcasts pattern
.claude/helpers/swarm-hooks.sh broadcast '{"strategy":"TDD","domain":"testing","quality":0.85}'
# Creates: .claude-flow/swarm/patterns/bc_*.json

# Other agents poll for broadcasts
.claude/helpers/swarm-hooks.sh get-pattern-broadcasts
```

### The Communication Reality

**Can agents communicate mid-task?** Technically yes, practically no.

- Agents would need to explicitly call `swarm-hooks.sh` via Bash
- No agent template instructs agents to poll for messages during execution
- Task agents execute in isolation — they don't know about the messaging system unless their prompt tells them to use it
- **The most reliable "communication" is through the filesystem**: one agent writes a file, another reads it

---

## 5. The Claims System

### Purpose
Prevent duplicate work by tracking which agent "owns" which resource.

### Storage
`.claude-flow/claims/claims.json`:
```json
{
  "claims": {
    "issue-123": {
      "issueId": "issue-123",
      "claimant": {"type": "agent", "agentId": "coder-1", "agentType": "coder"},
      "status": "active",
      "progress": 45,
      "context": "Working on authentication"
    }
  },
  "stealable": {},
  "contests": {}
}
```

### Operations
| Operation | What It Does |
|-----------|-------------|
| `claim` | Write claim record to JSON |
| `release` | Delete claim from JSON |
| `handoff` | Set `handoffTo` field, status to `handoff-pending` |
| `steal` | Transfer ownership if marked `stealable` |
| `rebalance` | Suggestion engine — doesn't auto-execute |

### Limitations
- **No atomic operations** — multiple agents writing simultaneously causes race conditions (last write wins)
- **No integration with swarm system** — claims and swarm operate on different JSON files with no shared logic
- **No locking** — no distributed locks or semaphores
- **Separate from Task tool** — Claude Code Task agents don't automatically check claims

---

## 6. Consensus Mechanisms

### What's Claimed
- Byzantine Fault Tolerant consensus with 2/3 majority
- Raft leader election with log replication
- Gossip-based distributed consensus

### What's Implemented

**hive-mind consensus** (hive-mind-tools.js lines 302-421):
```javascript
// "Consensus" = simple majority vote in JSON object
if (action === 'vote') {
    proposal.votes[voterId] = input.vote;  // Just sets key-value
    const votesFor = Object.values(proposal.votes).filter(v => v).length;
    const majority = Math.ceil(state.workers.length / 2) + 1;
    if (votesFor >= majority) {
        proposal.status = 'approved';
    }
}
```

**swarm-hooks.sh consensus** (lines 507-600):
```bash
initiate_consensus "Which framework?" "React,Vue,Svelte" 30000
# Creates: .claude-flow/swarm/consensus/cons_*.json

vote_consensus "cons_123" "React"
# Adds vote to JSON: .votes["agent_A"] = "React"

resolve_consensus "cons_123"
# Counts votes, picks winner
```

### Reality
- Simple majority counting in a local JSON file
- No quorum enforcement, no timeout handling
- No Byzantine detection, no leader election, no log replication
- "Raft" and "Byzantine" are string labels, not implementations
- Workers don't automatically participate — must be explicitly asked to vote

---

## 7. End-to-End Flow Trace

### What Actually Happens When You Say "spawn swarm"

**Step 1: UserPromptSubmit Hook Fires**
```bash
claude-flow hooks swarm-gate --task "spawn swarm for API implementation"
# Output: **[SWARM REQUIRED]** Complexity: 0.62
# → This is informational context for Opus, NOT enforced
```

**Step 2: Opus Reads the Directive**
The swarm-gate output appears in Claude Code's context:
```
**[SWARM REQUIRED]** Complexity: 0.62 (threshold: 0.50)
  Action: You MUST initialize swarm before executing this task.
  - Run: claude-flow swarm init --topology hierarchical --max-agents 8
  - Spawn 3+ agents with roles: coder, tester, reviewer
```

**Step 3: Opus Decides What To Do**
Usually one of two things:
- **Option A** (common): Ignores directive, executes as solo agent
- **Option B** (intended): Follows CLAUDE.md rules and initiates swarm

**Step 4: If Swarm Is Initiated**
```bash
# Opus calls:
claude-flow swarm init --topology hierarchical --max-agents 5 --strategy specialized
# → Writes .swarm/state.json (metadata only)
```

**Step 5: Opus Spawns Task Agents**
```
Task(subagent_type="v3-coder", prompt="Implement auth", run_in_background=true)
Task(subagent_type="v3-tester", prompt="Write tests", run_in_background=true)
Task(subagent_type="v3-reviewer", prompt="Review code", run_in_background=true)
```
All three calls go in ONE message for parallel execution.

**Step 6: Claude Code Manages the Agents**
- Each Task spawns an isolated Claude instance
- Agents execute in parallel with no awareness of each other
- Each agent has full tool access (Read, Write, Bash, etc.)
- **No claude-flow CLI involvement in execution**

**Step 7: Agents Complete, Results Return**
- Claude Code collects all background agent results
- PostToolUse hooks fire for each completed agent:
  ```bash
  claude-flow hooks post-task --task-id "$AGENT_ID" --success true --train-patterns true
  bash .claude/helpers/reflexion-post-task.sh   # Store episode
  bash .claude/helpers/skill-extract.sh          # Extract skills
  .claude/helpers/swarm-hooks.sh post-task "$PROMPT" true  # Broadcast completion
  ```

**Step 8: Opus Reviews All Results**
- All agent outputs visible to parent Opus
- Opus synthesizes, resolves conflicts, creates final output

### The Key Realization

**The "swarm" is just Claude Code's Task tool with `run_in_background: true`.**

Everything else — swarm init, topology, queen coordinator, consensus — is organizational metadata and prompt engineering layered on top. It's valuable for structure but does not drive execution.

---

## 8. What Actually Provides Value

### Genuinely Useful Components

| Component | Value | How It Works |
|-----------|-------|-------------|
| **Task tool parallelism** | HIGH | `run_in_background: true` enables real parallel execution |
| **Agent templates** | HIGH | `.claude/agents/*.md` define specialized personas |
| **Memory persistence** | HIGH | SQLite + HNSW for cross-agent data sharing |
| **Model routing** | MEDIUM | Swarm-gate complexity scoring selects haiku/sonnet/opus |
| **Pattern learning** | MEDIUM | PostToolUse hooks store successful patterns |
| **Claims system** | LOW-MEDIUM | Work ownership tracking (no enforcement) |
| **Swarm hooks** | LOW | Informational context for Opus (not enforced) |
| **Swarm init** | LOW | Metadata ceremony (no functional impact) |
| **Topology selection** | NONE | String label with zero behavioral difference |
| **Consensus tools** | NONE | Simple vote counting, never used in practice |

### The Real Coordination Pattern

The most effective swarm pattern we've used (confirmed by ADR-020, ADR-028, ADR-037):

1. **Decompose** task into non-overlapping subtasks
2. **Spawn** parallel Task agents with detailed prompts including file ownership boundaries
3. **Share data** via `claude-flow memory store/search` (instructed in agent prompts)
4. **Collect** results when all agents complete
5. **Synthesize** final output from all agent results

This works because:
- Non-overlapping file ownership prevents conflicts
- Memory serves as shared context
- Detailed prompts replace queen coordination
- Claude Code handles lifecycle (no need for custom supervision)

---

## 9. Architecture Diagram

### What's Documented

```
┌──────────────────────────────────────────────────────────┐
│  👑 Queen Coordinator (Byzantine Consensus)              │
│  ├── Flash Attention 2.49x-7.47x                        │
│  ├── HNSW 150x-12,500x                                  │
│  └── CRDT Synchronization                                │
│       │                                                   │
│  ┌────┼────┬────┬────┐                                   │
│  │    │    │    │    │                                    │
│  🔬   💻   📊   🛡️   🧪                                 │
│  RSR  CODE ANLY  SEC  TST                                │
│       │              │                                    │
│  ┌────┴────────┐  ┌──┴──┐                                │
│  Message Bus    │  Raft  │                                │
│  (Byzantine)    │  Cons. │                                │
└──────────────────────────────────────────────────────────┘
```

### What's Real

```
┌─────────────────────────────────────────────────────────┐
│                     Claude Code (Opus 4.6)              │
│  [The ONLY entity that spawns and manages agents]       │
│                                                          │
│  Hooks fire → informational context → Opus decides      │
└──────────────────┬──────────────────────────────────────┘
                   │ Task tool (run_in_background: true)
       ┌───────────┼───────────┐
       │           │           │
  ┌────▼────┐ ┌───▼────┐ ┌───▼────┐
  │ Agent A │ │Agent B │ │Agent C │
  │ (v3-    │ │(v3-    │ │(v3-    │
  │  coder) │ │ tester)│ │ rev.)  │
  └────┬────┘ └───┬────┘ └───┬────┘
       │          │           │
       │    No direct comms   │
       │          │           │
       └──────────┼───────────┘
                  │
         ┌────────▼────────┐
         │  Shared State   │
         │                 │
         │ memory.db (SQL) │  ← claude-flow memory store/search
         │ agentdb.db      │  ← episodes, skills, reflexion
         │ .claude-flow/   │  ← claims, messages (JSON)
         │ .swarm/         │  ← state.json (metadata)
         └─────────────────┘
```

---

## 10. Gap Analysis

### Claims vs Reality

| Feature | Documented | Implemented | Gap |
|---------|-----------|-------------|-----|
| Agent spawning | "Spawn specialized workers" | JSON record in file | **CRITICAL** — no processes created |
| Queen coordination | "Strategic command & control" | Same tools as any agent | **HIGH** — no special privileges |
| Topology routing | "Hierarchical/mesh/ring patterns" | String label, zero behavioral change | **HIGH** — cosmetic only |
| Consensus | "Byzantine fault-tolerant" | Majority vote in JSON | **HIGH** — no BFT implementation |
| Message bus | "Distributed message passing" | JSON files with manual polling | **MEDIUM** — exists but passive |
| Load balancing | "Adaptive work distribution" | Picks agent with lowest counter | **MEDIUM** — basic heuristic |
| Health monitoring | "Real-time health checks" | Always returns "healthy" | **HIGH** — hardcoded response |
| CRDT sync | "Conflict-free replication" | Last-write-wins SQLite | **HIGH** — not implemented |
| Agent communication | "Peer-to-peer messaging" | Shared filesystem + polling | **MEDIUM** — works but clunky |
| Swarm metrics | "Real-time performance data" | `Math.random()` values | **CRITICAL** — fabricated |

### Honest Assessment of swarm-tools.js

```javascript
// swarm_status handler — ALWAYS returns zeros
handler: async (input) => {
    return {
        swarmId: input.swarmId,
        status: 'running',    // HARDCODED
        agentCount: 0,        // HARDCODED
        taskCount: 0,         // HARDCODED
    };
}

// swarm_health handler — ALWAYS returns healthy
handler: async (input) => {
    return {
        status: 'healthy',    // HARDCODED
        checks: [
            { name: 'coordinator', status: 'ok' },  // HARDCODED
            { name: 'agents', status: 'ok' },        // HARDCODED
            { name: 'memory', status: 'ok' },        // HARDCODED
            { name: 'messaging', status: 'ok' },     // HARDCODED
        ]
    };
}
```

### coordination-tools.js Metrics — Fabricated

```javascript
// Returns RANDOM NUMBERS as "metrics"
metrics: {
    latency: { avg: 25 + Math.random() * 20 },
    throughput: { current: Math.floor(Math.random() * 1000) + 500 },
    availability: { uptime: 99.9 + Math.random() * 0.09 },
}
```

---

## 11. Recommendations

### For Our Implementation

1. **Stop calling `swarm init`** unless you want the metadata for organizational clarity. It has zero functional impact.

2. **Focus on Task tool parallelism** — this is the real swarm:
   ```
   Task(subagent=coder, prompt=..., background=true)
   Task(subagent=tester, prompt=..., background=true)
   ```

3. **Use memory for inter-agent data sharing** — instruct agents in their prompts to store/retrieve via `claude-flow memory`.

4. **Design prompts for non-overlapping file ownership** — this prevents merge conflicts better than any coordination protocol.

5. **Trust the pattern learning** — PostToolUse hooks store successful patterns in AgentDB/ReasoningBank, which informs future routing.

### For Improving Swarm Coordination

If we wanted to build real coordination on top of what exists:

| Priority | Improvement | Effort |
|----------|------------|--------|
| P0 | Fix `swarm_status` to read actual `.claude-flow/agents/store.json` | Low |
| P0 | Fix `swarm_health` to check real SQLite connections | Low |
| P1 | Add active message polling to agent templates | Medium |
| P1 | Wire claims system to Task tool agent prompts | Medium |
| P2 | Implement real topology-based message routing | High |
| P3 | Build actual queen coordinator as persistent process | Very High |
| P3 | Implement Raft/BFT consensus | Very High |

### What We Should NOT Do

- Don't build a distributed consensus system — it's unnecessary for our use case
- Don't try to make agents into persistent processes — Claude Code manages lifecycle better
- Don't implement real message buses — shared memory (SQLite) is sufficient
- Don't add CRDT/vector clocks — last-write-wins is fine for non-overlapping file ownership

---

## Appendix A: File Inventory

### CLI Commands
- `dist/src/commands/swarm.js` (748 lines) — CLI facade, JSON file management
- `dist/src/commands/agent.js` (819 lines) — Agent CRUD on JSON
- `dist/src/commands/hive-mind.js` (~300 lines) — Hive-mind CLI, `--claude` flag

### MCP Tools
- `dist/src/mcp-tools/swarm-tools.js` (101 lines) — Stub handlers returning hardcoded values
- `dist/src/mcp-tools/hive-mind-tools.js` (~500 lines) — JSON file state management
- `dist/src/mcp-tools/agent-tools.js` (549 lines) — Agent JSON CRUD
- `dist/src/mcp-tools/coordination-tools.js` (486 lines) — "LOCAL STATE MANAGEMENT" (per header)
- `dist/src/mcp-tools/claims-tools.js` (731 lines) — Issue claims system
- `dist/src/mcp-tools/task-tools.js` — Task JSON CRUD

### Agent Templates
- `.claude/agents/v3/v3-queen-coordinator.md` (82 lines)
- `.claude/agents/swarm/hierarchical-coordinator.md` (717 lines, 457 TS example)
- `.claude/agents/swarm/mesh-coordinator.md` (970 lines, 638 TS example)
- `.claude/agents/swarm/adaptive-coordinator.md` (1133 lines, 669 TS example)
- `.claude/agents/v3/collective-intelligence-coordinator.md` (1002 lines, 803 TS example)
- `.claude/agents/v3/swarm-memory-manager.md` (165 lines)
- `.claude/agents/templates/coordinator-swarm-init.md` (98 lines)
- `.claude/agents/templates/memory-coordinator.md` (195 lines)

### Helper Scripts
- `.claude/helpers/swarm-hooks.sh` — File-based message passing, handoffs, consensus
- `.claude/helpers/format-routing-directive.sh` — Formats swarm-gate output
- `.claude/helpers/reflexion-pre-task.sh` — AgentDB episode retrieval
- `.claude/helpers/reflexion-post-task.sh` — AgentDB episode storage
- `.claude/helpers/skill-suggest.sh` — AgentDB skill suggestions
- `.claude/helpers/skill-extract.sh` — AgentDB skill extraction

### State Files
```
.claude-flow/
├── agents/store.json          # Agent registry (System B)
├── tasks/store.json           # Task queue
├── claims/claims.json         # Work claims
├── coordination/store.json    # Topology metadata
├── hive-mind/state.json       # Hive-mind workers, consensus, memory
└── swarm/
    ├── messages/msg_*.json    # Agent messages
    ├── patterns/bc_*.json     # Broadcast patterns
    ├── consensus/cons_*.json  # Consensus votes
    └── handoffs/ho_*.json     # Task handoffs

.swarm/
├── state.json                 # Swarm metadata (from CLI init)
├── memory.db                  # Shared SQLite memory
└── agentdb.db                 # AgentDB episodes/skills
```

---

## Appendix B: The One Real Feature in Hive-Mind

The `hive-mind spawn --claude` flag (hive-mind.js lines 149-305) is the **only thing that creates an actual process**:

```javascript
async function spawnClaudeCodeInstance(swarmId, swarmName, objective, workers, flags) {
    const hiveMindPrompt = generateHiveMindPrompt(swarmId, swarmName, objective, workers);
    await writeFile(promptFile, hiveMindPrompt, 'utf8');

    // Spawns actual Claude Code CLI process
    const claudeProcess = childSpawn('claude', claudeArgs, {
        stdio: 'inherit',
        shell: false,
    });
}
```

This literally:
1. Generates a text prompt telling the LLM to "act as a queen coordinator"
2. Spawns the `claude` CLI process with that prompt as input
3. All "coordination" happens in the LLM's context window

It's essentially the same as manually opening Claude Code and pasting a "you are a queen coordinator" prompt — but automated.

---

## Appendix C: What Our Successful Swarms Actually Used

Based on ADR-020 (8-agent swarm), ADR-028 (5-agent swarm), and ADR-037 (5-agent swarm):

### Pattern That Works
1. **Decomposition**: Split work by file ownership boundaries
2. **Parallel Task spawn**: All agents in ONE message with `run_in_background: true`
3. **Detailed prompts**: Each agent gets exact files to create/modify
4. **Non-overlapping ownership**: No two agents touch the same file
5. **Memory sharing**: Agents store/retrieve shared context via `claude-flow memory`
6. **Collect and review**: Parent reviews all results before proceeding

### Pattern That Does NOT Work
1. Expecting agents to communicate mid-task
2. Relying on queen coordinator to supervise workers
3. Using consensus for decision-making
4. Expecting topology to affect routing
5. Depending on claims system for conflict prevention

### Conclusion

**The swarm is the Task tool. Everything else is decoration.**

The decoration is useful — agent templates define specialized roles, hooks store learning patterns, memory enables data sharing — but the actual multi-agent execution is 100% Claude Code's native Task tool. Understanding this distinction is critical for using the system effectively.

---
---

# ADDENDUM: Gap-Finding Analysis (Round 2)

**Method**: 4 additional research agents investigating gaps in the initial analysis
**Focus Areas**: Worker systems, v3 source packages, learning pipeline, real-world swarm usage evidence

---

## Addendum 1: Worker Systems — Genuinely Functional (MISSED)

The initial analysis completely overlooked three **real, functional subsystems**:

### WorkerQueue (511 lines) — REAL

**Location**: `dist/src/services/worker-queue.js`

A **full-featured Redis-compatible task queue** with:
- Priority-based scheduling (critical/high/normal/low)
- Automatic retry with exponential backoff (`Math.min(30000, 1000 * 2^retryCount)`)
- Dead letter queue for permanent failures
- Worker heartbeat monitoring
- Concurrent task processing with configurable limits
- Result caching with TTL
- EventEmitter-based events (`enqueued`, `completed`, `failed`, `retrying`)

```javascript
class WorkerQueue extends EventEmitter {
    async enqueue(workerType, payload, options)   // Add to queue
    async dequeue(workerTypes)                     // Pull from queue
    async complete(taskId, result)                 // Mark done
    async fail(taskId, error, retryable = true)    // Mark failed + retry
    async registerWorker(workerTypes, options)      // Register consumer
    async start(workerTypes, handler, options)      // Processing loop
    async shutdown()                                // Graceful shutdown (30s timeout)
}
```

**Verdict**: Production-ready job queue. Currently uses in-memory store (fallback when Redis unavailable), but architecturally ready for distributed execution.

### WorkerDaemon (756 lines) — REAL BACKGROUND PROCESS

**Location**: `dist/src/services/worker-daemon.js`

A **genuine Node.js daemon** with 12 scheduled workers:

| Worker | Interval | Priority | Purpose |
|--------|----------|----------|---------|
| `map` | 15 min | normal | Codebase mapping |
| `audit` | 10 min | critical | Security analysis |
| `optimize` | 15 min | high | Performance optimization |
| `consolidate` | 30 min | low | Memory consolidation |
| `testgaps` | 20 min | normal | Test coverage analysis |
| `predict` | manual | high | Pattern prediction |
| `document` | manual | normal | Documentation generation |
| `ultralearn` | manual | critical | Deep learning analysis |
| `refactor` | manual | normal | Code refactoring |
| `deepdive` | manual | high | Deep code analysis |
| `benchmark` | manual | normal | Performance benchmarking |
| `preload` | manual | low | Predictive preloading |

Real features:
- **Resource gating**: Checks CPU load and free memory before executing workers
- **Concurrency limits**: Default 2 concurrent workers, queues excess
- **Staggered scheduling**: Prevents I/O spikes
- **Persistent state**: Saves to `.claude-flow/daemon-state.json`, restores on restart
- **Headless execution**: Integrates with Claude Code CLI for AI-powered workers
- **Metrics output**: Writes real JSON to `.claude-flow/metrics/` (codebase-map, security-audit, performance, consolidation)

### Daemon CLI (593 lines) — PRODUCTION-READY

```bash
claude-flow daemon start              # Detached background (PID file, log file)
claude-flow daemon start --foreground  # Blocks terminal
claude-flow daemon status              # PID, uptime, worker stats, success rates
claude-flow daemon stop                # SIGTERM → wait 1s → SIGKILL
claude-flow daemon trigger -w audit    # Manual worker execution
claude-flow daemon enable -w predict   # Enable disabled worker
```

Security hardened:
- Path validation (prevents null bytes, shell metacharacters)
- No shell string interpolation (uses spawn argument array)
- PID file management with stale detection

### Worker Dispatch Intelligence

**Location**: `hooks-tools.js` lines 2441-2636

12 worker types with **pattern-based auto-detection** from user prompts:
- Scans for keywords: "optimize", "test", "security", "document", etc.
- Returns confidence scores per detected worker
- Can auto-dispatch via `--auto-dispatch` flag
- Integrates with swarm-gate: complexity >0.3 triggers worker dispatch recommendations

### Workflow Engine (Framework)

**Location**: `dist/src/mcp-tools/workflow-tools.js`

Step types: `task`, `condition`, `parallel`, `loop`, `wait`
State machine: `ready → running → completed/failed`
Features: pause/resume, template system, variable injection

**Current status**: Framework real, but step execution is placeholder — designed for extension.

### Other Tools: State Tracking Only

- **DAA Tools** (`daa-tools.js`): Header says "LOCAL STATE MANAGEMENT". Tracks agent coordination metadata in `.claude-flow/daa/store.json`
- **Terminal Tools** (`terminal-tools.js`): Header says "does NOT actually execute commands". Records command history for coordination tracking.

---

## Addendum 2: V3 Source Packages — Infrastructure Shipped But Unwired

### Event Sourcing: EXISTS BUT UNUSED

`@claude-flow/shared` ships a **full EventStore implementation** (415 lines):
- Append-only event log with SQLite (sql.js)
- Event versioning per aggregate
- Snapshot support
- Event replay for projections
- Causation/correlation IDs

**But the CLI claims system uses simple JSON** (`readFileSync`/`writeFileSync` to `claims.json`). The event sourcing infrastructure exists in the shared package but is never imported by the CLI.

### swarm-comms.sh: More Sophisticated Than Initially Found

**Location**: `/home/snoozyy/.claude/helpers/swarm-comms.sh` (354 lines)

Beyond the basic `swarm-hooks.sh` messaging, this implements:

1. **Priority-based message queue** (4 levels: 0=critical → 3=low)
2. **Message batching** with configurable batch size (default 10, auto-flush)
3. **Connection pooling** with acquire/release/cleanup lifecycle
4. **Async pattern broadcasting** (fire-and-forget via background `&`)
5. **Async consensus** with configurable timeout and background auto-resolve
6. **Mailbox-based routing** (per-agent directories under `.claude-flow/swarm/mailbox/`)

**Key difference from swarm-hooks.sh**: swarm-comms.sh is designed for **non-blocking async operations** where every operation returns immediately and processing happens in background subshells.

### Hive-Mind `--claude` Flag: Rich Queen Prompt

The `hive-mind spawn --claude` command generates a **143-line queen coordination prompt** that:
- Lists ALL available MCP tools organized by category (5 sections)
- Provides a 4-phase execution protocol (Init → Execute → Monitor → Complete)
- Includes coordination tips for broadcast, consensus, and memory sharing
- Spawns actual `claude` CLI process with the prompt
- Handles SIGINT for session pause/resume with saved prompts
- Supports `--non-interactive` mode with JSON streaming

This is the **most functionally complete coordination feature** — it's the only thing that creates a real process and gives it a comprehensive coordination blueprint.

---

## Addendum 3: Learning Pipeline — Genuinely Feeds Back (With Limitations)

### The 4-Layer Learning Loop

```
TASK EXECUTION
    │
    ▼
POST-TASK HOOKS (fire after each Task agent completes)
    ├── reflexion-post-task.sh → Store episode (task, output, reward) to agentdb.db
    ├── skill-extract.sh → Auto-consolidate similar episodes into skills
    └── hooks post-task --train-patterns → Display-only (NOT real training)
    │
    ▼
STORAGE
    ├── agentdb.db: episodes table, skills table (DDD services)
    ├── model-router-state.json: routing history (last 100 outcomes)
    └── memory.db: sessions table, routing context
    │
    ▼
NEXT SESSION / NEXT TASK
    │
    ▼
PRE-TASK HOOKS (fire before each new Task agent)
    ├── session-routing-context.js → Load past 3 sessions' complexity, detect escalation (+0.1 boost)
    ├── reflexion-pre-task.sh → Retrieve k=5 similar episodes via HNSW → inject into agent prompt
    └── skill-suggest.sh → Retrieve k=3 skills via composite scoring → inject as suggestions
    │
    ▼
AGENT RECEIVES AUGMENTED PROMPT (original + past episodes + suggested skills)
```

### What's Genuinely Valuable

1. **Episodic memory injection** (`reflexion-pre-task.sh`): Retrieves 5 most similar past episodes via HNSW vector search (384-dim MiniLM-L6-v2 embeddings) and injects them as `additionalContext` in the agent prompt. Agents receive concrete past outcomes including successes, failures, and critiques. **This is the strongest learning feedback.**

2. **Composite skill scoring**: `0.4*similarity + 0.3*successRate + 0.1*(usage/1000) + 0.2*avgReward` — well-designed formula that balances relevance, proven success, usage frequency, and reward.

3. **Circuit breaker**: After 5 consecutive failures on a model, it gets penalized in routing. Prevents stuck loops.

4. **Cross-session escalation**: Tracks whether task complexity is increasing across sessions and boosts routing thresholds accordingly.

### What's Weak or Broken

1. **Binary reward scoring**: Fixed 0.8 (success) or 0.2 (failure). Cannot distinguish mediocre from exceptional.

2. **`--train-patterns` is display-only**: Returns fabricated values (`patternsUpdated: 2`, `newPatterns: 1`) with no backend storage. This flag does nothing.

3. **Skill extraction uses prefix matching**: Groups tasks by `substr(task, 1, 50)` — cannot recognize semantically similar tasks with different wording. Vector search exists for retrieval but not consolidation.

4. **Static routing thresholds**: Model routing thresholds (0.3 haiku, 0.6 sonnet) are never adjusted by outcome feedback. No gradient descent, no Bayesian updates.

5. **Learning history capped at 100 entries**: Older patterns are forgotten entirely (no EWC++ consolidation despite it being advertised).

---

## Addendum 4: Real-World Swarm Usage — The Evidence Gap

### What ADRs Claim

| ADR | Claim | Agents |
|-----|-------|--------|
| ADR-020 | "8-agent swarm, hierarchical" | coordinator + 2 architects + 2 coders + 2 reviewers + tester |
| ADR-028 | "5-agent swarm, hierarchical, opus" | schema + skill-library + reflexion + bootstrap + tester |
| ADR-037 | "5-agent swarm, hierarchical, opus" | Fixes A-D agents + Fix E bridge agent |

### What Evidence Exists

**Swarm state file** (`~/.swarm/state.json`):
```json
{
  "id": "swarm-1770735408334",
  "topology": "hierarchical",
  "maxAgents": 15,
  "strategy": "specialized",
  "initializedAt": "2026-02-10T14:56:48.334Z",
  "status": "ready"
}
```
Swarm was initialized once (Feb 10). Status is "ready" — never moved to "running" or "completed".

**Message files** (`~/.claude-flow/swarm/messages/`): 100+ JSON files, but ALL are **completion notifications**:
```json
{
  "from": "agent_1770407152_befa27e5",
  "to": "*",
  "type": "result",
  "content": "Completed: \n",
  "priority": "low"
}
```
No inter-agent coordination messages. No queries, handoffs, or conflict resolution.

**Handoff files** (`~/.claude-flow/swarm/handoffs/`): **ZERO files**. The handoff mechanism was never used.

**Git history**: Individual commits per ADR, not the chaotic multi-agent pattern expected from true parallel execution. Single commits suggest either:
- Agents coordinated before committing (single final commit)
- Work was done sequentially and attributed to "N-agent swarm" as a decomposition strategy

### What Actually Happened

Based on ADR-028 Agent 5 completion report (the most detailed evidence):
> "60s agent wait allowed other agents to complete"

This reveals **sequential execution with waits**, not true parallel coordination. The pattern was likely:
1. Decompose work into logical phases (Agent 1-N roles)
2. Execute via Claude Code Task tool (possibly `run_in_background: true` for some)
3. Agent 5 waited for prior agents to finish
4. Results aggregated and documented as "N-agent swarm"

### The Pattern That Actually Works

Based on real-world evidence, successful multi-agent work in this project used:

1. **Task decomposition by file ownership** — non-overlapping boundaries
2. **Parallel Task tool calls** — `run_in_background: true` for real parallelism
3. **Detailed prompts replacing coordination** — each agent told exactly what to do
4. **Memory as shared context** — `claude-flow memory store/search` for data sharing
5. **Parent agent as integrator** — reviews all results, resolves conflicts

What was NOT used:
- `swarm init` (state file exists but was ceremonial)
- Inter-agent messages (only completion notifications)
- Handoffs (zero files)
- Consensus (no voting evidence)
- Queen coordinator (no supervisor process)

---

## Revised Summary Table

After both analysis rounds, here's the complete picture:

| Component | First Analysis | Gap-Finding Revision | Actual Status |
|-----------|---------------|---------------------|---------------|
| **Task tool parallelism** | THE real swarm | Confirmed | **WORKS — primary execution engine** |
| **Agent templates** | Prompt engineering | Confirmed | **WORKS — defines agent personas** |
| **Memory persistence** | SQLite + HNSW | Confirmed | **WORKS — real coordination mechanism** |
| **WorkerQueue** | MISSED | 511 lines, production-ready | **WORKS — real job queue** |
| **WorkerDaemon** | MISSED | 756 lines, real background process | **WORKS — genuine daemon with resource gating** |
| **Worker Dispatch** | MISSED | Pattern detection + auto-dispatch | **WORKS — integrates with swarm-gate** |
| **Learning pipeline** | Mentioned briefly | Full 4-layer loop traced | **PARTIALLY WORKS — episodic retrieval real, pattern training fake** |
| **swarm-comms.sh** | MISSED | 354 lines async messaging | **EXISTS — priority queues, batching, connection pooling** |
| **hive-mind --claude** | Mentioned | 143-line queen prompt + process spawn | **WORKS — the most complete coordination feature** |
| **Event sourcing** | Not analyzed | Shipped in @claude-flow/shared | **EXISTS BUT UNUSED — claims still use JSON** |
| **Workflow engine** | MISSED | Real framework, placeholder execution | **PARTIAL — structure exists, steps not implemented** |
| **Model routing** | Mentioned | Learning loop traced | **WORKS — static thresholds, no adaptation** |
| **Swarm init** | Metadata only | Confirmed by real usage evidence | **CEREMONIAL — no functional impact** |
| **Topology** | Cosmetic label | Confirmed | **COSMETIC — zero behavioral difference** |
| **Consensus** | Simple voting | Confirmed by absence of evidence | **UNUSED IN PRACTICE** |
| **Inter-agent messaging** | File-based polling | 100+ messages are completion notifications only | **EXISTS BUT UNUSED FOR COORDINATION** |
| **Handoffs** | Manual protocol | Zero files found | **UNUSED** |
| **Queen coordinator** | No special powers | No supervisor process evidence | **PROMPT ENGINEERING ONLY** |
| **Swarm metrics** | `Math.random()` | Confirmed | **FABRICATED** |

---

## Final Revised Conclusion

**The initial conclusion stands but needs refinement:**

> "The swarm is the Task tool. Everything else is decoration."

**Revised**: "The swarm is the Task tool. The decoration is more substantial than initially found — the WorkerQueue, WorkerDaemon, learning pipeline, and swarm-comms.sh provide real infrastructure. But the decoration has never been fully exercised in practice. What actually coordinates our multi-agent work is: (1) Task tool parallelism, (2) detailed prompts with file ownership boundaries, (3) memory-based data sharing, and (4) episodic retrieval from past sessions."

The biggest gap is not missing features — it's the **gap between what's built and what's used**. The infrastructure for sophisticated swarm coordination exists (worker queues, async messaging, event sourcing, consensus voting). It's just not wired into the actual execution path that Claude Code follows when spawning Task agents.

---
---

# ADDENDUM 2: Deep Dive Analysis (Round 3)

**Method**: 5-agent deep-dive swarm focusing on implementation internals with exact code references
**Agents**: hooks-tools.js internals, MCP runtime path, headless/container execution, settings.json hook wiring, plugin/guidance/skills/commands systems

## CRITICAL CORRECTION: Real Process Execution Exists

**Previous conclusion was PARTIALLY WRONG.** Rounds 1-2 stated "no real processes" — this is incorrect. Claude-flow has **THREE real process execution modes** that were completely missed because earlier analysis focused on `agent-tools.js` (MCP metadata) and never examined the `services/` directory.

### Three Real Execution Modes

#### 1. Headless Worker Executor (`headless-worker-executor.js:810`)
```javascript
const child = spawn('claude', ['--print', prompt], {
    cwd: this.projectRoot,
    env: { ...process.env, CLAUDE_CODE_HEADLESS: 'true', ANTHROPIC_MODEL: MODEL_IDS[options.model] },
    stdio: ['pipe', 'pipe', 'pipe']
});
```
- Spawns REAL `claude` CLI processes in headless mode
- 8 AI worker types: `audit`, `optimize`, `testgaps`, `document`, `ultralearn`, `refactor`, `deepdive`, `predict`
- Process pool with max 2 concurrent, timeout protection (5-15 min)
- **Requires**: `@anthropic-ai/claude-code` CLI installed

#### 2. Container Worker Pool (`container-worker-pool.js:407`)
```javascript
const child = spawn('docker', args, { stdio: ['pipe', 'pipe', 'pipe'] });
```
- Manages pool of Docker containers (1-3 default, max 15)
- Image: `ghcr.io/ruvnet/claude-flow-headless:latest`
- Resource limits: 2 CPUs, 4GB memory per container
- Health checking, auto-recovery, idle termination
- **Requires**: Docker daemon running

#### 3. Hive-Mind Claude Code Launcher (`hive-mind.js:218`)
```javascript
const claudeProcess = childSpawn('claude', claudeArgs, { stdio: 'inherit', shell: false });
```
- Spawns `claude` CLI with full TTY (interactive, not headless)
- Generates coordination prompt with Byzantine consensus instructions
- `--dangerously-skip-permissions` for seamless execution
- **This is the ONLY mode that delegates to Claude Code's Task tool for sub-agent spawning**

#### 4. Worker Daemon Local Fallback (what previous analysis found)
- JSON file generators with no AI — the only mode previous research examined
- Falls back to this when Claude CLI and Docker are unavailable

### Execution Architecture Comparison

| Feature | Headless Executor | Container Pool | Hive-Mind | Daemon Local |
|---------|------------------|----------------|-----------|--------------|
| Process Type | Claude CLI | Docker | Claude CLI (TTY) | In-process Node.js |
| Isolation | Process | Container (Docker) | Process | None |
| Concurrency | 2 (default) | 1-3 (scalable) | 1 (interactive) | 2 (queued) |
| AI Execution | **REAL** (Claude API) | **REAL** (via container) | **REAL** (Task tool) | **FAKE** (JSON) |

### Critical Code Paths

```
Path 1: daemon.js:184 → worker-daemon.js:78 → headless-worker-executor.js:810 → REAL CLAUDE PROCESS
Path 2: container-worker-pool.js:299 → line 407 (docker exec) → REAL DOCKER CONTAINER
Path 3: hive-mind.js:218 → CLAUDE CODE INTERACTIVE → Claude Task tool sub-agents
Path 4: worker-daemon.js fallback → JSON generators (NO AI)
```

---

## Exact Complexity Scoring Algorithm (hooks-tools.js)

### Base Complexity (Lines 2747-2758)

```
complexity = min(1, max(0, 0.3 + (highCount * 0.2) - (lowCount * 0.15) + (length/200 * 0.2)))
```

- **High indicators** (+0.2 each): `architect`, `design`, `refactor`, `security`, `audit`, `complex`, `analyze`
- **Low indicators** (-0.15 each): `simple`, `typo`, `format`, `rename`, `comment`
- **Length factor**: task.length / 200, capped at 0.2

### Contextual Boosts (Lines 2810-2820)

| Boost | Condition | Amount |
|-------|-----------|--------|
| File count | `parseInt(fileCount) > 3` | +0.10 |
| Multi-step | `multiStep === 'true'` | +0.05 |
| Swarm intent | `/swarm\|coordinate\|orchestrat\|parallel/i` | +0.15 |
| Security intent | `/secur\|audit\|vuln\|cve/i` | +0.10 |
| Architecture intent | `/architect\|design\|refactor\|migrat/i` | +0.05 |

### Swarm Gate Thresholds (Lines 2821-2825)

| Score | Enforcement | Action |
|-------|------------|--------|
| < 0.30 | `optional` | Single agent OK |
| 0.30-0.49 | `recommended` | Swarm preferred |
| >= 0.50 | `required` | Swarm mandatory |

### Agent Role Detection (Lines 2827-2836)

Regex-based detection assigns suggested roles:
- `secur|audit|vuln|cve` → `security-architect`, `security-auditor`
- `test|validat|verif` → `v3-tester`
- `code|implement|build|fix|patch` → `v3-coder`
- `review|analyz` → `v3-reviewer`
- `architect|design|refactor|migrat` → `architecture`
- `plan|decompos|break down` → `v3-planner`
- `research|investigat|explor` → `v3-researcher`
- Default fallback: `v3-coder`

---

## Model Routing Scoring Curves (model-router.js, ADR-021)

### Base Complexity Features (Lines 198-227)

```
score = min(1, max(0, lexicalComplexity * 0.2 + semanticDepth * 0.35 + taskScope * 0.25 + uncertaintyLevel * 0.2))
```

### ADR-021 Adjusted Model Curves (Lines 292-304)

```javascript
haiku:  score < 0.25 ? 1 - score * 1.5 : 1 - score * 2.5   // Drops to 0 at ~0.4
sonnet: 1 - Math.abs(score - 0.45) * 2.2                     // Peaks at 0.45
opus:   Math.min(1, score * 2.0)                              // Reaches 100% at 0.5
```

**Key insight**: Opus reaches max score at just 0.5 complexity, meaning moderate-complexity tasks get routed to Opus. Haiku is only viable below 0.25.

### Learning & Circuit Breaker

- History bounded to last 100 outcomes, persisted to `.swarm/model-router-state.json`
- Circuit breaker: 5 consecutive failures trips protection for that model
- Escalation ladder: haiku → sonnet → opus (when uncertainty > threshold)

---

## MCP Runtime: 175 Tools Across 24 Modules

### Startup Sequence

1. `bin/mcp-server.js` — Pure JSON-RPC 2.0 stdio transport (5 methods: initialize, tools/list, tools/call, ping, notifications/initialized)
2. `dist/src/mcp-client.js` — Imports 24 tool modules, builds global `TOOL_REGISTRY` Map synchronously
3. `callMCPTool(name, input, context)` — Direct Map lookup → `tool.handler(input, context)`. No middleware, no retry, no auth.

### Tool Count by Category

| Category | Count | Category | Count |
|----------|-------|----------|-------|
| hooks | 37 | browser | 23 |
| claims | 12 | transfer | 12 |
| hive-mind | 9 | workflow | 9 |
| daa | 8 | performance | 8 |
| agent | 7 | memory | 7 |
| embeddings | 7 | coordination | 7 |
| config | 6 | security | 6 |
| task | 6 | analyze | 6 |
| neural | 6 | agentdb | 6 |
| system | 5 | terminal | 5 |
| session | 5 | github | 5 |
| swarm | 4 | progress | 4 |
| **TOTAL** | **175** | | |

### Memory Backend (memory-initializer.js, 1,929 lines)

- **9-table SQLite schema**: memory_entries, patterns, pattern_history, trajectories, trajectory_steps, migration_state, sessions, vector_indexes, metadata
- **Embeddings**: MiniLM-L6-v2 (384-dim ONNX) → agentic-flow fallback (768-dim) → hash fallback (128-dim)
- **HNSW**: @ruvector/core Rust NAPI, lazy-initialized, 150x-12,500x faster search
- **Flash Attention ops**: batchCosineSim (~50us/1000 vectors), softmaxAttention, topKIndices
- **Int8 quantization**: 4x memory reduction, `scale = max(|min|, |max|) / 127`

---

## Complete Hook Pipeline Per Task Tool Call

### Phase 1: Pre-Task Hooks (6 hooks, 22s max timeout)

| Order | Script | Timeout | Purpose |
|-------|--------|---------|---------|
| 1 | `swarm-gate \| format-routing-directive.sh` | 5s | **Swarm enforcement directive** |
| 2 | `guidance-hooks.sh route` | 3s | Agent role hints via keyword matching |
| 3 | `swarm-hooks.sh pre-task` | 3s | Agent registration + stale reaping |
| 4 | `claude-flow hooks pre-task` | 5s | Intelligence + coordination state |
| 5 | `reflexion-pre-task.sh` | 3s | **Episodic memory retrieval (ADR-020)** |
| 6 | `skill-suggest.sh` | 3s | **Skill library retrieval (ADR-020)** |

Hooks 5-6 inject `hookSpecificOutput.additionalContext` into the Task prompt (past episodes + proven skills).

### Phase 2: Task Execution

Task prompt now enriched with: swarm directive, 5 similar past episodes, 3 proven skills, routing hints.

### Phase 3: Post-Task Hooks (8 hooks, 22s max timeout)

| Order | Script | Timeout | Purpose |
|-------|--------|---------|---------|
| 1 | `claude-flow hooks post-task` | 5s | Pattern training + metrics |
| 2 | `reflexion-post-task.sh` | 3s | **Episode storage (BACKGROUND, non-blocking)** |
| 3 | `skill-extract.sh` | 3s | **Skill extraction (BACKGROUND, non-blocking)** |
| 4 | `swarm-hooks.sh post-task` | 3s | Pattern broadcast |
| 5 | `claude-flow hooks metrics` | 3s | Update metrics JSON |
| 6 | `swarm-monitor.sh check` | 3s | Health check |
| 7 | `checkpoint-manager.sh summary` | 3s | Checkpoint |
| 8 | `pattern-consolidator.sh check` | 5s | Pattern consolidation |

Hooks 2-3 use `{ command } & disown` pattern — fire-and-forget, non-blocking (ADR-016).

### Session Lifecycle Hooks

- **SessionStart** (7 hooks): agent registration, V3 context injection, learning init, session restore, agent config build, **verify-patches.sh** (ADR-037)
- **SessionEnd** (5 hooks): learning consolidation, session cleanup, memory export, metrics, checkpoint
- **UserPromptSubmit** (3 hooks): swarm-gate, worker status, guidance prompt analysis

---

## Agent Booster: External NPX Package, NOT WASM

**CLAUDE.md claims**: "Agent Booster (WASM)" with <1ms latency

**Actual implementation** (enhanced-model-router.js:415-461):
```javascript
const cmd = `npx --yes agent-booster@0.2.2 apply --language ${language}`;
const result = execSync(cmd, { input: JSON.stringify({ code, edit: instruction }), timeout: 5000 });
```

- Calls **external npm package** `agent-booster@0.2.2` as subprocess
- NOT WASM, NOT AST-based — it's a child process with 5s timeout
- 6 supported intents: `var-to-const`, `remove-console`, `async-await`, `add-logging`, `add-types`, `add-error-handling`
- Confidence threshold: 0.7 (below = falls through to LLM)

---

## Worker Dispatch: Simulated Execution

The 12-worker system (ultralearn, optimize, consolidate, predict, audit, map, preload, deepdive, document, refactor, benchmark, testgaps) has **real trigger detection** but **simulated execution**:

```javascript
// Lines 2605-2628: Worker "execution" is just setTimeout
setTimeout(() => {
    const w = activeWorkers.get(workerId);
    if (w) { w.progress = 100; w.phase = 'completed'; w.status = 'completed'; }
}, 1500);
```

Workers are created in an in-memory Map, progress is updated via setTimeout, no actual analysis runs. The trigger detection (regex patterns) is sophisticated, but dispatch is a no-op.

---

## Overlooked Subsystems (~40-50% of Architecture)

### 1. Guidance Framework (Constitutional AI)

- **WASM Kernel**: `guidance_kernel_bg.wasm` (94.3KB Rust binary)
  - `batch_process()`, `content_hash()`, `detect_destructive()`, `hmac_sha256()`, `scan_secrets()`
- **CLI**: `claude-flow guidance` with 6 subcommands (compile, retrieve, gates, status, optimize, ab-test)
- **GuidanceProvider** (350 lines): Pre-edit gates (blocks .env/.pem/.key), post-edit quality checks, pre-command risk assessment, routing recommendations
- **Current status**: Hooks use `guidance-hooks.sh` (bash keyword matching), NOT the full WASM semantic retrieval engine. WASM exists but is unwired.

### 2. Plugin System (IPFS Marketplace)

- **Architecture**: IPFS-based decentralized registry with Ed25519 signature verification
- **21 plugins**: 9 official (Anthropic) + 12 external (Asana, GitHub, Slack, Stripe, Firebase, etc.)
- **Plugin types**: agent, hook, command, provider, integration, theme, core, hybrid
- **10 permission types**: network, filesystem, execute, memory, agents, credentials, config, hooks, privileged
- **CLI**: `claude-flow plugins` with 9 subcommands (list, search, install, uninstall, upgrade, toggle, info, create, rate)
- **Trust levels**: unverified → community → verified → official

### 3. RuVector Intelligence Layer (14 modules, ~3,500 lines)

- **Flash Attention** (`flash-attention.js`, 367 lines): 2-7x speedup via block-wise tiling, 32-element blocks, top-K sparse attention
- **LoRA Adapter** (`lora-adapter.js`, 400+ lines): Rank 8, 384-dim, 24x parameter reduction, online gradient descent, persists to `.swarm/lora-weights.json`
- **MoE Router** (`moe-router.js`, 500+ lines): 8 experts, 2-layer MLP gating, top-2 selection, load balancing, persists to `.swarm/moe-weights.json`
- **Additional**: semantic-router, q-learning-router, coverage-router, diff-classifier, ast-analyzer, graph-analyzer, vector-db

### 4. Skills System (33 Auto-Invoked Skills)

Skills activate automatically when Claude's task matches their `description` frontmatter trigger:
- **AgentDB** (5): advanced, learning, optimization, memory-patterns, vector-search
- **GitHub** (5): code-review, project-management, release-management, multi-repo, workflow-automation
- **V3 Implementation** (9): core, DDD, CLI, integration, performance, MCP, security, memory, swarm
- **Swarm** (2): advanced, orchestration
- **ReasoningBank** (2): agentdb, intelligence
- **Other** (10): browser, hooks-automation, pair-programming, skill-builder, sparc, stream-chain, verification-quality, etc.

### 5. Commands System (90+ Slash Commands)

User-invoked via `/command-name`:
- **GitHub** (20): code-review, pr-manager, release-manager, issue-tracker, multi-repo-swarm, etc.
- **SPARC** (30+): architect, coder, tester, debugger, reviewer, sparc-modes, etc.
- **Analysis** (6): bottleneck-detect, performance-report, token-efficiency, etc.
- **Automation** (7): auto-agent, smart-agents, self-healing, session-memory, etc.
- **Monitoring** (6): agents, swarm-monitor, real-time-view, status, etc.
- **Optimization** (5): auto-topology, cache-manage, parallel-execute, etc.

### 6. Helpers Runtime Library (45 Scripts)

Categories: hook helpers (11), service helpers (5), management helpers (9), safety helpers (4), status helpers (5), setup helpers (5), tool helpers (6), git hooks (2).

---

## Trajectory & Intelligence Tracking

### What's Real

- **Trajectory recording**: In-memory Map, stores steps with action/result/quality timestamps
- **Pattern storage**: Real HNSW indexing via `memory-initializer.js storeEntry()` — generates embeddings, writes to SQLite
- **SONA learning**: Real modules called on trajectory completion (processes outcome, returns pattern key + confidence)
- **Model routing history**: Last 100 outcomes persisted, circuit breaker, learning from escalations

### What's Synthetic

- **EWC++ gradients**: `sin(i * 0.01) * (steps.length / 10)` — NOT real neural network gradients
- **Pretrain pipeline**: Returns multiplier-based fabricated numbers, not real training
- **Worker execution**: setTimeout progress updates only

---

## Revised Architecture: FOUR Layers, Not Three

| Layer | What | Execution | Status |
|-------|------|-----------|--------|
| **Layer 1: Claude Code Task Tool** | Built-in parallel agent spawning | **REAL** — this IS the swarm | Active, primary execution |
| **Layer 2: claude-flow Services** | Headless executor, container pool, worker daemon | **REAL** — spawns claude/docker processes | Available but requires CLI/Docker |
| **Layer 3: claude-flow CLI/MCP** | 175 MCP tools, hooks, memory, routing | **METADATA** — JSON state + complexity scoring | Active, enriches Layer 1 |
| **Layer 4: Agent Templates + Skills** | 95 agents, 33 skills, 90+ commands | **PROMPT ENGINEERING** — behavioral guidance | Active, shapes agent behavior |

**Key revision**: Layer 2 (services) was completely missed in Rounds 1-2. The headless-worker-executor and container-worker-pool are production-grade with process pooling, timeout protection, resource limits, and error recovery.

---

## Final Revised Conclusion (Updated)

> "The swarm is the Task tool. Everything else is decoration."

**Updated**: "The swarm PRIMARILY runs via the Task tool. But claude-flow has a SECOND real execution layer (headless executor + container pool + hive-mind launcher) that spawns actual Claude/Docker processes. This layer is production-grade but currently unused in our typical workflow because we use Claude Code's native Task tool instead. The third layer (175 MCP tools, hooks, complexity scoring, model routing) enriches the Task tool with episodic memory, skill suggestions, and swarm enforcement directives. The fourth layer (agents, skills, commands) shapes behavior through prompt engineering. Additionally, ~40-50% of the architecture was overlooked: a guidance framework (constitutional AI via WASM), a plugin marketplace (21 IPFS-based plugins), an ML intelligence layer (Flash Attention, LoRA, MoE), 33 auto-invoked skills, and 90+ slash commands."

The system is significantly more sophisticated than initially assessed. The gap is not between what's built and what exists — it's between what's built and what's **wired together and actively used**.
