# Claude-Flow Repository Analysis

**Repository**: https://github.com/ruvnet/claude-flow
**Published Package**: `claude-flow` on npm (tag: `v3alpha`)
**Analyzed Version**: 3.1.0-alpha.3 (npm) / 3.1.0-alpha.1 (repo package.json)
**Analysis Date**: 2026-02-06
**Total Files in Repo**: ~10,274

---

## Executive Summary

Claude-flow is a large, ambitious project that genuinely implements most of what it
advertises, but with significant caveats:

1. **The core MCP server, tools, and CLI are real and functional** -- there is actual
   TypeScript code implementing 170+ MCP tools, hooks, memory, swarms, etc.
2. **Advanced features (HNSW, neural, RL) are real code** -- not stubs, but pure
   TypeScript/JavaScript implementations, not production-grade native libraries.
3. **Performance claims are aspirational** -- the "150x-12,500x faster" HNSW claim
   refers to theoretical HNSW vs brute-force, not benchmarked real numbers.
4. **Critical dependencies (`agentic-flow`, `agentdb`, `@ruvector/core`) are all
   first-party packages** by the same author, listed as `optionalDependencies`.
5. **The system gracefully degrades** -- every component has fallback paths when
   optional dependencies are missing, which is the default state for most users.
6. **The npm-published package only ships CLI + guidance + shared** -- the MCP server,
   memory, neural, hooks, claims, embeddings, etc. from the `v3/` monorepo are NOT
   in the npm package. They exist only in the Git repo source.

---

## 1. Package Structure and Distribution

### What's Advertised
- "One-line install" via npx
- 170+ MCP tools
- 60+ specialized agents
- Full neural/learning/HNSW capabilities

### What's Actually Published to npm

The `files` field in `package.json` specifies only:
```
v3/@claude-flow/cli/bin/**
v3/@claude-flow/cli/dist/**/*.js
v3/@claude-flow/cli/dist/**/*.d.ts
v3/@claude-flow/shared/...
v3/@claude-flow/guidance/...
.claude-plugin/**
.claude/**
```

**Key finding**: The published npm package contains:
- `v3/@claude-flow/cli/` -- The CLI with `bin/cli.js` and `bin/mcp-server.js`
- `v3/@claude-flow/cli/dist/` -- Compiled JS with its own embedded MCP tools
- `v3/@claude-flow/guidance/` -- Guidance/governance framework
- `v3/@claude-flow/shared/` -- Shared types
- `.claude-plugin/` -- Plugin system files
- `.claude/` -- Agent definitions, commands, skills, helpers

**NOT published** (repo-only source code):
- `v3/@claude-flow/memory/` -- The HNSW + SQLite memory package
- `v3/@claude-flow/neural/` -- SONA, ReasoningBank, RL algorithms
- `v3/@claude-flow/hooks/` -- Hooks package
- `v3/@claude-flow/claims/` -- Claims/work-ownership system
- `v3/@claude-flow/embeddings/` -- Embedding generation
- `v3/@claude-flow/swarm/` -- Swarm coordination
- `v3/@claude-flow/browser/` -- Browser automation
- `v3/@claude-flow/aidefence/` -- Security/threat detection
- `v3/@claude-flow/mcp/` -- The full MCP server implementation
- `v3/@claude-flow/providers/` -- Multi-provider LLM support

However, the CLI's `dist/` directory contains its own compiled versions of many of
these features embedded directly (e.g., `dist/src/mcp-tools/hooks-tools.js` at 112KB,
`dist/src/memory/`, `dist/src/ruvector/`).

### Dependencies

**Hard dependency**: Only `zod` (schema validation)

**Optional dependencies** (all first-party):
```json
{
  "@claude-flow/plugin-gastown-bridge": "^0.1.0",
  "agentdb": "2.0.0-alpha.3.4",
  "bcrypt": "^5.1.1",
  "@ruvector/attention": "^0.1.3",
  "@ruvector/core": "^0.1.30",
  "@ruvector/router": "^0.1.27",
  "@ruvector/router-linux-x64-gnu": "^0.1.27",
  "@ruvector/sona": "^0.1.5",
  "agentic-flow": "^2.0.1-alpha.80"
}
```

These are ALL packages by the same author (ruvnet/RuvNet):
- `agentic-flow` v2.0.6 -- "Production-ready AI agent orchestration platform"
- `agentdb` v2.0.0-alpha.3.3 -- "RuVector-powered graph database"
- `@ruvector/core` v0.1.30 -- "High-performance vector database with HNSW indexing - 50k+ inserts/sec, built in Rust"

---

## 2. MCP Server Implementation

### What's Advertised
- 170+ MCP tools
- Native Claude Code integration
- <400ms startup, <10ms tool registration, <50ms tool execution overhead

### What the Source Code Shows

**Repo file**: `v3/mcp/server.ts` (20,632 bytes)

The MCP server is a **real, complete implementation** that:
- Extends `EventEmitter`
- Supports multiple transports (stdio, HTTP, WebSocket, in-process)
- Has session management with timeouts
- Has a tool registry with batch registration
- Implements the full MCP JSON-RPC protocol (initialize, tools/list, tools/call, ping)
- Has connection pooling
- Tracks request metrics (total, successful, failed, response time)

**Tool registration**: `v3/mcp/tools/index.ts` imports and re-exports 12 tool groups:
- `agentTools` -- agent/spawn, agent/list, agent/terminate, agent/status
- `swarmTools` -- swarm/init, swarm/status, swarm/scale
- `memoryTools` -- memory/store, memory/search, memory/list
- `configTools` -- config/load, config/save, config/validate
- `hooksTools` -- 9 tools (pre-edit, post-edit, pre-command, post-command, route, explain, pretrain, metrics, list)
- `taskTools` -- 8 tools
- `systemTools` -- 4 tools
- `sessionTools` -- 3 tools
- `workerTools` -- 8 tools
- `sonaTools` -- 14 tools (trajectory tracking, pattern search, LoRA, benchmarks)
- `federationTools` -- 8 tools
- `v2CompatTools` -- backward compatibility (17+ tools)

**However**, this full MCP server at `v3/mcp/server.ts` is NOT what runs when you do
`npx claude-flow mcp start`. The CLI entry point at `v3/@claude-flow/cli/bin/cli.js`
auto-detects MCP mode and uses `dist/src/mcp-client.js` -- a separate, simplified
MCP implementation embedded in the CLI package.

The CLI's `dist/src/mcp-tools/` directory contains its own tool implementations
(separate from the `v3/mcp/tools/` monorepo source).

### Discrepancies
- The "170+ tools" claim is plausible when counting V2 compat + all tool groups
- The two MCP implementations (full server vs CLI embedded) may behave differently
- Performance claims are theoretical targets, not verified benchmarks

---

## 3. HNSW Vector Search

### What's Advertised
- "150x-12,500x faster vector similarity search"
- Real HNSW index with quantization
- Pre-normalized vectors for O(1) cosine similarity

### What the Source Code Shows

**Repo file**: `v3/@claude-flow/memory/src/hnsw-index.ts` (27,799 bytes)

This is a **real, from-scratch HNSW implementation** in TypeScript:

```typescript
export class HNSWIndex extends EventEmitter {
  private nodes: Map<string, HNSWNode> = new Map();
  private entryPoint: string | null = null;
  private maxLevel: number = 0;
  private levelMult: number;
```

Key implementation details:
- `BinaryMinHeap` and `BinaryMaxHeap` classes for O(log n) priority queue operations
- Pre-normalized vectors (`Float32Array`) for O(1) cosine similarity
- Multi-layer graph with configurable M (max connections), efConstruction
- Quantization support (`Quantizer` class for int8)
- Search with post-retrieval filtering
- Performance tracking (search count, insert time)
- Proper layer-by-layer search from top layer down to layer 0

**Quality assessment**: This is a legitimate HNSW implementation following the
original paper's algorithm. It is NOT a wrapper around a native library -- it's
pure TypeScript. This means:
- It works without any native dependencies
- It will be significantly slower than C++/Rust implementations (hnswlib, faiss)
- The "150x faster" claim is vs brute-force linear scan, not vs other ANN libraries
- For small datasets (hundreds to low thousands), it works fine
- For large datasets, it would lag behind native implementations

### Where It's Used

The memory tools (`v3/mcp/tools/memory-tools.ts`) attempt to use the memory service
with HNSW, but fall back to simple in-memory storage:

```typescript
if (resourceManager?.memoryService) {
  try {
    const { UnifiedMemoryService } = await import('@claude-flow/memory');
    // ... use HNSW-backed memory
  } catch (error) {
    // Fall through to simple implementation
  }
}
// Simple implementation when no memory service is available
return { id, stored: true, storedAt };
```

**The fallback is just storing an ID and returning success** -- no actual
persistence, no search, no vectors.

### Optional Native Enhancement

`@ruvector/core` (v0.1.30) is described as "built in Rust for AI/ML similarity
search" and IS listed as an optional dependency. If installed, it presumably
provides a much faster native HNSW implementation. But it's optional and most
users won't have it.

---

## 4. Learning System (SONA / ReasoningBank)

### What's Advertised
- "Self-Optimizing Neural Architecture (SONA)"
- "<0.05ms adaptation"
- "Learns from every task execution"
- "EWC++ prevents catastrophic forgetting"
- "9 RL Algorithms (Q-Learning, SARSA, A2C, PPO, DQN, Decision Transformer, etc.)"
- "ReasoningBank with 4-step learning pipeline: RETRIEVE, JUDGE, DISTILL, CONSOLIDATE"
- "MicroLoRA for 128x compression"

### What the Source Code Shows

#### ReasoningBank (`v3/@claude-flow/neural/src/reasoning-bank.ts`, 38,604 bytes)

This is a **real implementation** of a trajectory-based learning system:
- Stores trajectories (sequences of action-observation-reward steps)
- Implements RETRIEVE with MMR (Maximal Marginal Relevance) diversity
- Implements JUDGE (scoring trajectories)
- Implements DISTILL (extracting strategy memories from trajectories)
- Implements CONSOLIDATE (dedup, contradiction detection, pruning)
- Attempts to use `agentdb` for persistent vector storage with fallback

```typescript
let AgentDB: any;
async function ensureAgentDBImport(): Promise<void> {
  try {
    const agentdbModule = await import('agentdb');
    AgentDB = agentdbModule.AgentDB || agentdbModule.default;
  } catch {
    AgentDB = undefined;  // Fallback to in-memory
  }
}
```

#### SONA Manager (`v3/@claude-flow/neural/src/sona-manager.ts`, 22,661 bytes)

Real implementation with 5 operating modes:
- `real-time` (LoRA rank 2, <0.5ms latency target)
- `balanced` (LoRA rank 4, <18ms)
- `research` (LoRA rank 16, <100ms)
- `edge` (LoRA rank 1, <1ms, <5MB memory)
- `batch` (LoRA rank 8, <50ms)

Each mode has different configuration for learning rate, batch size, HNSW ef_search,
pattern threshold, and optimizations (SIMD, microLoRA, gradient checkpointing, etc.).

#### RL Algorithms (`v3/@claude-flow/neural/src/algorithms/`)

**7 real algorithm implementations** (3,035 total lines):
- `q-learning.ts` (333 lines)
- `sarsa.ts` (383 lines)
- `dqn.ts` (382 lines)
- `ppo.ts` (429 lines)
- `a2c.ts` (478 lines)
- `curiosity.ts` (509 lines)
- `decision-transformer.ts` (521 lines)

These are genuine RL algorithm implementations, but they operate on **in-memory
state tables** (not neural networks with actual weight matrices). For example,
Q-Learning uses a `Map<string, Map<string, number>>` as its Q-table, DQN uses
simple array-based "networks."

#### Pattern Learner (`v3/@claude-flow/neural/src/pattern-learner.ts`, 22,312 bytes)

Implements pattern extraction, matching, and evolution:
- Pattern clustering for fast lookup
- Cosine similarity matching
- Pattern quality tracking and promotion
- Configurable thresholds and learning rates

#### Learning Service Helper (`.claude/helpers/learning-service.mjs`, 35,393 bytes)

This is a **persistent learning service** that:
- Uses `better-sqlite3` for pattern storage
- Has short-term and long-term pattern tables
- Implements promotion logic (short-term patterns promoted after N uses)
- Has a consolidation cycle (dedup, prune old patterns)
- Plans for ONNX embeddings via `agentic-flow@alpha` (MiniLM-L6-v2)
- Actually creates SQLite database at `.claude-flow/learning/patterns.db`

### Key Assessment

The learning system is **architecturally sound but operationally limited**:
- It tracks trajectories and stores patterns -- this is real
- The RL algorithms are simple table-based implementations, not deep RL
- "LoRA" in this context is NOT Low-Rank Adaptation of actual LLM weights --
  it operates on in-memory routing tables and pattern weights
- "EWC++" is implemented as a weight consolidation strategy for the local
  pattern store, not for transformer model weights
- The "<0.05ms adaptation" claim is plausible for updating an in-memory map
- Without `agentdb` or `agentic-flow` installed, everything falls back to
  in-memory Maps that don't persist across sessions

### SONA MCP Tools (`v3/mcp/tools/sona-tools.ts`)

The SONA tools use a `SONAState` singleton class with in-memory Maps:
```typescript
class SONAState {
  trajectories: Map<string, Trajectory> = new Map();
  patterns: Map<string, Pattern> = new Map();
  profiles: Map<string, SONAProfile> = new Map();
  enabled: boolean = true;
}
```

It attempts to load `agentic-flow/core` for HNSW support:
```typescript
async function loadAgenticFlow(): Promise<boolean> {
  try {
    agenticFlowCore = await import('agentic-flow/core');
    if (agenticFlowCore?.createFastAgentDB) {
      agentDBInstance = agenticFlowCore.createFastAgentDB({ dimensions: 768 });
    }
    return true;
  } catch {
    return false;  // Fallback to simple implementations
  }
}
```

---

## 5. Hooks System

### What's Advertised
- 17 hooks (pre-edit, post-edit, pre-command, post-command, route, explain, etc.)
- Self-learning from hook data
- Agent Booster (WASM) for simple transforms (<1ms)
- Intelligent 3-tier model routing

### What the Source Code Shows

**Repo file**: `v3/mcp/tools/hooks-tools.ts` (40,552 bytes)

The hooks system is **real and integrates with ReasoningBank**:
```typescript
import {
  ReasoningBank,
  createReasoningBank,
  type Trajectory,
  type TrajectoryStep,
} from '../../@claude-flow/neural/src/index.js';
```

Hook tools implemented:
- `hooks/pre-edit` -- Analyzes file before edit, provides suggestions
- `hooks/post-edit` -- Records edit outcome for learning
- `hooks/pre-command` -- Risk assessment before command execution
- `hooks/post-command` -- Records command outcome
- `hooks/route` -- Routes tasks to optimal agents using learned patterns
- `hooks/explain` -- Explains routing decisions
- `hooks/pretrain` -- Bootstraps intelligence from repository analysis
- `hooks/metrics` -- Returns learning metrics
- `hooks/list` -- Lists registered hooks

The routing system (`hooks/route`) uses the ReasoningBank to:
1. Retrieve similar past trajectories
2. Score task complexity
3. Match against agent capabilities
4. Return recommended agent with confidence score

### How Hooks Feed Into Memory

The flow is:
1. `hooks/pre-edit` starts a trajectory
2. Each edit/command adds steps to the trajectory
3. `hooks/post-edit` / `hooks/post-command` ends the trajectory with a verdict
4. If verdict is positive, patterns are distilled via ReasoningBank
5. Distilled patterns are stored (in-memory, or in AgentDB if available)
6. Future `hooks/route` calls retrieve relevant patterns to inform routing

### Hooks in the `.claude/settings.json`

The repo ships with a pre-configured `settings.json` that wires hooks into
Claude Code's lifecycle:
- `pre_tool_use` hooks for edit/command operations
- `post_tool_use` hooks for recording outcomes
- These call shell scripts in `.claude/helpers/` which invoke the MCP tools

---

## 6. Agent System

### What's Advertised
- 60+ specialized agents
- Queen/worker hierarchy
- Hive mind with consensus

### What the Source Code Shows

**Location**: `.claude/agents/` and `v3/@claude-flow/cli/.claude/agents/`

The "agents" are **markdown files** that serve as system prompts/instructions.
They are NOT runtime agent processes. Examples:
- `.claude/agents/core/coder.md` (6,502 bytes) -- Instructions for a coder agent
- `.claude/agents/core/tester.md` (8,488 bytes) -- Instructions for a tester agent
- `.claude/agents/hive-mind/queen-coordinator.md` -- Queen agent template

When you "spawn an agent," the system:
1. Reads the appropriate markdown template
2. Uses it as instructions for a Claude Code Task tool invocation
3. The Task tool spawns a new Claude session with those instructions

The swarm coordination (`v3/@claude-flow/claims/`) implements:
- Claim/release/handoff protocols for work ownership
- Load balancing across agents
- Work stealing for idle agents
- Event sourcing for claim state

This is real coordination code (the claims package alone has 45,907 bytes of tests),
but agents themselves are just prompt templates -- not persistent processes.

---

## 7. CLI vs MCP Server

### The Distinction

**CLI** (`v3/@claude-flow/cli/bin/cli.js`):
- Entry point for all user-facing commands
- Auto-detects MCP mode when stdin is piped
- Commands: init, agent, swarm, memory, task, session, hooks, hive-mind, etc.
- Ships as the `claude-flow` binary in the npm package
- Contains its own embedded MCP tool implementations in `dist/src/mcp-tools/`

**MCP Server** (`v3/mcp/server.ts`):
- Full MCP protocol implementation
- Used when `npx claude-flow mcp start` is run
- Registers tools from `v3/mcp/tools/`
- NOT published to npm as a separate package

When you run `claude mcp add claude-flow -- npx claude-flow mcp start`:
- The CLI's `mcp-server.js` entrypoint runs
- It uses the CLI's embedded MCP tools (from `dist/src/mcp-tools/`)
- These are different from but functionally equivalent to `v3/mcp/tools/`

---

## 8. Key File Paths in the Repository

### Core MCP Server
- `v3/mcp/server.ts` -- Full MCP server implementation (20,632 bytes)
- `v3/mcp/tools/index.ts` -- Tool registry, exports all tool groups
- `v3/mcp/tools/memory-tools.ts` -- Memory store/search/list tools
- `v3/mcp/tools/hooks-tools.ts` -- Hooks/learning tools (40,552 bytes)
- `v3/mcp/tools/sona-tools.ts` -- SONA/neural tools (28,136 bytes)
- `v3/mcp/tools/agent-tools.ts` -- Agent lifecycle tools
- `v3/mcp/tools/swarm-tools.ts` -- Swarm management tools
- `v3/mcp/tools/worker-tools.ts` -- Background worker tools
- `v3/mcp/tools/task-tools.ts` -- Task management tools
- `v3/mcp/tools/session-tools.ts` -- Session persistence tools
- `v3/mcp/tools/system-tools.ts` -- System status/health/metrics

### Memory/HNSW
- `v3/@claude-flow/memory/src/hnsw-index.ts` -- Pure TS HNSW implementation (27,799 bytes)
- `v3/@claude-flow/memory/src/sqlite-backend.ts` -- SQLite persistence (20,000 bytes)
- `v3/@claude-flow/memory/src/hybrid-backend.ts` -- Hybrid SQLite+HNSW (19,253 bytes)
- `v3/@claude-flow/memory/src/agentdb-adapter.ts` -- AgentDB integration (27,278 bytes)
- `v3/@claude-flow/memory/src/cache-manager.ts` -- LRU cache (11,413 bytes)
- `v3/@claude-flow/memory/src/types.ts` -- Type definitions

### Neural/Learning
- `v3/@claude-flow/neural/src/reasoning-bank.ts` -- ReasoningBank (38,604 bytes)
- `v3/@claude-flow/neural/src/sona-manager.ts` -- SONA manager (22,661 bytes)
- `v3/@claude-flow/neural/src/pattern-learner.ts` -- Pattern extraction (22,312 bytes)
- `v3/@claude-flow/neural/src/sona-integration.ts` -- SONA integration (11,852 bytes)
- `v3/@claude-flow/neural/src/algorithms/q-learning.ts` -- Q-Learning (333 lines)
- `v3/@claude-flow/neural/src/algorithms/ppo.ts` -- PPO (429 lines)
- `v3/@claude-flow/neural/src/algorithms/dqn.ts` -- DQN (382 lines)

### CLI (Published)
- `v3/@claude-flow/cli/bin/cli.js` -- CLI entry point
- `v3/@claude-flow/cli/bin/mcp-server.js` -- MCP server entry point
- `v3/@claude-flow/cli/dist/src/mcp-tools/hooks-tools.js` -- Embedded hooks (112KB)
- `v3/@claude-flow/cli/dist/src/memory/` -- Embedded memory system
- `v3/@claude-flow/cli/dist/src/ruvector/` -- Embedded routing/LoRA/attention

### Helpers (Deployed with package)
- `.claude/helpers/learning-service.mjs` -- Persistent learning with SQLite (35,393 bytes)
- `.claude/helpers/learning-hooks.sh` -- Shell hooks for Claude Code integration
- `.claude/helpers/statusline.cjs` -- Status bar for Claude Code (40,925 bytes)
- `.claude/helpers/swarm-hooks.sh` -- Swarm coordination hooks
- `.claude/settings.json` -- Pre-configured hook wiring for Claude Code

### Agent Templates (Deployed with package)
- `.claude/agents/core/` -- coder.md, planner.md, researcher.md, reviewer.md, tester.md
- `.claude/agents/hive-mind/` -- queen-coordinator.md, worker-specialist.md, etc.
- `.claude/agents/sparc/` -- architecture.md, pseudocode.md, refinement.md, etc.
- `.claude/agents/github/` -- pr-manager.md, code-review-swarm.md, etc.

---

## 9. Discrepancies Between Marketing and Implementation

### 1. "150x-12,500x Faster HNSW"
**Marketing**: Implies production-grade vector search performance
**Reality**: Pure TypeScript HNSW is real but slow compared to native implementations.
The speedup claim is vs brute-force linear scan (O(n) -> O(log n)), which is
mathematically correct but misleading. Users expecting performance comparable to
FAISS or hnswlib will be disappointed.

### 2. "Self-Learning Neural Architecture"
**Marketing**: Suggests deep learning with neural networks
**Reality**: Pattern matching with in-memory maps and simple RL tables. "LoRA" is
not LLM weight adaptation -- it's lightweight local state updates. "EWC++" is
weight consolidation for the pattern store, not transformer weights. The learning
IS real (patterns are extracted, stored, retrieved), but "neural" overstates it.

### 3. "60+ Specialized Agents"
**Marketing**: Suggests 60+ autonomous agent processes
**Reality**: 60+ markdown prompt templates. Agents are Claude Code sessions
initialized with specific instructions. They are powerful when used correctly, but
they are not persistent services.

### 4. "Production-Ready"
**Marketing**: "Enterprise-grade architecture"
**Reality**: The core package version is `3.1.0-alpha`. All dependencies are alpha
versions. The npm publish tag is `v3alpha`. No production deployments are documented.

### 5. "84.8% SWE-Bench Solve Rate"
**Marketing**: Headline performance claim
**Reality**: No source or methodology for this claim is provided in the repository.
SWE-Bench is a well-known benchmark, and this would be a remarkable result. It may
refer to internal testing, but it's unverifiable.

### 6. Performance Numbers
**Marketing**: "<0.05ms adaptation", "352x faster code transforms", "34,798 routes/s"
**Reality**: These may be micro-benchmark results on specific operations (updating a
single map entry IS sub-0.05ms). They don't represent end-to-end latency that users
experience. The Agent Booster WASM transforms are not implemented in the published
package (no WASM files are shipped).

### 7. Optional Dependencies Create Two Experiences
**Marketing**: Presents all features as available
**Reality**: Without `agentdb`, `agentic-flow`, and `@ruvector/*` installed:
- Memory falls back to non-persistent in-memory Maps
- HNSW is unavailable (no vectors are generated without embeddings)
- Pattern storage doesn't persist across sessions
- Learning metrics reset on restart
These optional deps are all first-party alpha packages that may have their own issues.

---

## 10. What Actually Works (For Our Setup)

Based on this analysis, here's what should work with the standard npm install:

### Works Out of the Box
1. **MCP tool registration** -- Claude Code sees the tools (confirmed by our setup)
2. **Agent templates** -- The `.claude/agents/` markdown files are shipped and usable
3. **CLI commands** -- init, agent, swarm, memory, task, hooks, etc.
4. **Hook wiring** -- `.claude/settings.json` hooks into Claude Code lifecycle
5. **Basic memory** -- In-memory store/search works within a session
6. **Task routing** -- The hooks/route tool works with heuristic scoring
7. **Claims system** -- Work ownership between agents

### Requires Optional Dependencies
1. **Persistent memory** -- Needs `agentdb` or `better-sqlite3`
2. **Vector search** -- Needs embeddings (from `agentic-flow` or ONNX)
3. **Cross-session learning** -- Needs the SQLite-backed learning service
4. **Native HNSW performance** -- Needs `@ruvector/core`
5. **Multi-provider LLM** -- Needs `agentic-flow`

### Doesn't Work / Not Implemented
1. **Agent Booster WASM** -- No WASM files are published
2. **ONNX embeddings** -- Requires downloading model files
3. **RuVector PostgreSQL** -- External database, not bundled
4. **Flow Nexus cloud platform** -- Separate service, not part of the package

---

## 11. Why Our Local Setup May Be Broken

Based on this analysis, likely issues with our local setup:

1. **The MCP server is the CLI's embedded version**, not the full `v3/mcp/server.ts`.
   The CLI's MCP tools at `dist/src/mcp-tools/` are a separate codebase that may
   have different behavior or bugs.

2. **All "intelligence" features gracefully degrade to no-ops**. Without optional
   dependencies installed, memory_store returns `{ stored: true }` without actually
   storing anything. This makes the system APPEAR to work while doing nothing.

3. **The learning-service.mjs helper requires `better-sqlite3`** which is a native
   module that needs compilation. If the npm install skips optional deps (common
   with npx), this service won't work.

4. **Hook shell scripts assume specific paths** relative to the project root.
   If the project structure doesn't match expectations, hooks silently fail.

5. **The `.claude/settings.json` hook configuration** may point to scripts that
   don't exist or can't execute in the current environment.

---

## 12. Recommendations

1. **Don't rely on "self-learning"** -- Treat claude-flow as a tool registry and
   agent template system. The learning/HNSW/neural features are bonus if they work.

2. **Install optional deps explicitly** if you want persistence:
   ```bash
   npm install better-sqlite3  # For learning-service.mjs
   npm install agentdb@alpha   # For AgentDB vector storage
   ```

3. **Verify the MCP tools actually work** by testing each category:
   ```bash
   npx claude-flow hooks metrics    # Does this return real data?
   npx claude-flow memory list      # Are memories persisted?
   npx claude-flow neural status    # Is SONA actually running?
   ```

4. **The real value of claude-flow is**:
   - Pre-built MCP tool definitions (saves writing your own)
   - Agent prompt templates (60+ well-crafted role definitions)
   - Hook wiring for Claude Code (pre/post edit/command lifecycle)
   - CLI for swarm/memory/task management
   - Claims system for multi-agent coordination

5. **For our Gecko Vision Hub project**, consider:
   - Using only the agent templates and MCP tools we actually need
   - Implementing our own persistent memory with Supabase (we already have it)
   - Skipping the SONA/neural features entirely
   - Using the hooks system for pre/post edit tracking
   - Writing our own simple routing instead of relying on learned patterns

---

## 13. RuVector Ecosystem Deep-Dive (2026-02-08)

For the full analysis of the ruvector repository and ecosystem, see
[ruvector-analysis.md](./ruvector-analysis.md).

### Key Findings

The entire ruvnet ecosystem is a **5-layer dependency stack** where claude-flow
sits at the top:

```
claude-flow v3.1.0     →  CLI + 170 MCP tools + 60 agent templates
agentic-flow v2.0.6    →  66 agents, Claude SDK, multi-provider orchestration
agentdb v2.0.0-alpha   →  Memory DB, ReasoningBank, attention, QUIC sync
ruvector v0.1.96       →  TypeScript wrapper (189 lines)
@ruvector/* binaries   →  15+ MB compiled Rust (HNSW, GNN, attention, SONA)
```

### What RuVector Actually Is

- A **72-crate Rust monorepo** with 45 npm packages
- The native performance foundation for vector search, attention, and learning
- Core HNSW is a **wrapper around `hnsw_rs`** (not a novel implementation)
- 40+ attention mechanisms compiled to native speed via NAPI-RS
- SONA (Self-Optimizing Neural Architecture) for continual learning
- Several genuinely interesting algorithms: hyperbolic HNSW, sheaf Laplacian
  coherence, no_std neuromorphic WASM

### Benchmark Reality Check

Documented benchmark results are unreliable:
- 100% recall reported everywhere (impossible for real HNSW)
- 0 MB memory reported everywhere (profiler broken)
- Simulated competitors (`"simulated": "true"`) presented alongside real results
- README claims 16,400 QPS / 61us latency; measured 3,597 QPS / 780us

### Integration with Claude-Flow

RuVector integrates at four levels:
1. **Direct CLI module** (`dist/src/ruvector/`) -- 15 files for routing, attention, vectors
2. **Memory subsystem** -- `@ruvector/core` VectorDb for HNSW-backed memory search
3. **Training service** -- Uses `@ruvector/attention`, `@ruvector/sona`, `@ruvector/learning-wasm`
4. **Via agentdb** -- VectorBackend, AttentionService, ReasoningBank

Every integration gracefully degrades when ruvector is unavailable. The performance
impact of having ruvector is ~60x for memory search (native HNSW vs JS brute-force)
and ~5-50x for attention operations.

### Our Local Installation Status

All native binaries are present and loadable:
- `@ruvector/core` v0.1.30 (HNSW, 5.2 MB binary) -- WORKING
- `@ruvector/attention` v0.1.4 (attention, 1.1 MB) -- PRESENT
- `@ruvector/sona` v0.1.5 (SONA, 564 KB) -- PRESENT (all 8 platforms bundled)
- `@ruvector/gnn` v0.1.22 (GNN, 744 KB) -- PRESENT
- `@ruvector/router` v0.1.28 (routing, 1.3 MB) -- PRESENT
- `@ruvector/graph-node` v0.1.26 (graph DB, 4.6 MB) -- PRESENT
- `@ruvector/ruvllm` v0.2.4 (LLM, 1.1 MB) -- PRESENT
- `@ruvector/learning-wasm` v0.1.29 (LoRA, WASM) -- PRESENT

**Still missing/broken:**
- Agent Booster WASM (`agentic-flow/dist/agent-booster/` doesn't exist)
- ONNX embeddings (`embeddings init` fails -- "downloadModel is not a function")
- Duplicate ReasoningBank implementations (agentdb vs agentic-flow -- different code, same name)
- Core uses placeholder hash embeddings, not real semantic embeddings

### Revised Assessment of Optional Dependencies

With this deeper understanding, our optional dependencies are more valuable than
initially assessed:

| Package | Value | Status |
|---------|-------|--------|
| `@ruvector/core` | HIGH -- ~60x faster memory search | Working |
| `@ruvector/attention` | MEDIUM -- native attention for training | Binary present |
| `@ruvector/sona` | MEDIUM -- SONA continual learning | Binary present |
| `agentdb` | HIGH -- structured memory DB with controllers | Working |
| `agentic-flow` | LOW-MEDIUM -- mostly used for embeddings/ReasoningBank | Patched, working |
| `better-sqlite3` | HIGH -- persistence for learning service | Working |

The ecosystem is ambitious but young (3 months old, all alpha). For our use case
of hundreds to low thousands of stored patterns, the native binaries provide
meaningful speedup over JS fallbacks, and the scale limitations are irrelevant.

---

## 14. Agentic-Flow Deep-Dive (2026-02-08)

For the full analysis, see [agentic-flow-analysis.md](./agentic-flow-analysis.md).

### Critical Discovery: Three ReasoningBanks

The ruvnet ecosystem contains **three separate ReasoningBank implementations that
share zero code**:

1. **agentic-flow's** (`dist/reasoningbank/`): Most sophisticated — 5 algorithms
   from DeepMind paper (Retrieve, Judge, Distill, Consolidate, MaTTS). Requires
   LLM API key for real learning. SQLite at `.swarm/memory.db`.
2. **agentdb's** (`controllers/ReasoningBank.js`): Pattern store with optional GNN
   enhancement and VectorBackend integration.
3. **claude-flow's** (`intelligence.js`): In-memory Map + JSON file at
   `~/.claude-flow/neural/patterns.json`. No judge, no distill, no consolidation.
   **This is the one that actually runs.**

### What agentic-flow Actually Provides to claude-flow

Claude-flow uses agentic-flow for surprisingly little:
- Embedding model access (fallback chain, but ONNX download is broken → hash-based)
- ReasoningBank `retrieveMemories()` in token-optimize hook (read-only)
- @ruvector native binaries as transitive dependencies

It does NOT use: learning pipeline, multi-provider routing, Claude Agent SDK,
AgentDB controllers, swarm coordination, or any of the 9 MCP tools.

### What's Actually Real vs Misleading

**Real**: Multi-provider routing (Anthropic/OpenRouter/Gemini/ONNX),
CircuitBreakerRouter, attention wrappers around @ruvector native, PII scrubber,
SONA service integration, 82 agent prompt templates.

**Stub**: QUIC transport (returns `{}`), Federation Hub (returns `[]`), HTTP/3
proxy (returns empty Uint8Array), Agent Booster (directory doesn't exist).

**Misleading**: "213 MCP tools" (9-11 that shell out to claude-flow), "66 agents"
(82 markdown files), "GNN-Enhanced" (single-layer perceptron), "9 RL algorithms"
(all Q-value updates), "Byzantine consensus" (not found).

### AgentDB: Genuinely Substantive

Despite the issues above, agentdb's controllers are real implementations:
- **ReflexionMemory** (815 lines): Episodic replay grounded in arxiv:2303.11366
- **SkillLibrary** (697 lines): Skill consolidation from arxiv:2305.16291 (Voyager)
- **CausalMemoryGraph** (602 lines): Pearl's do-calculus with A/B experiments
- **RuVectorBackend** (776 lines): Production-quality HNSW wrapper with security

These are NOT used by claude-flow. They represent unlockable value.

### The Usage Gap

The biggest finding across all three analyses (claude-flow, ruvector, agentic-flow)
is the gap between what exists and what runs:

| Sophisticated Implementation | Actually Used? |
|------------------------------|---------------|
| agentic-flow ReasoningBank (5 DeepMind algorithms) | Only `retrieveMemories()`, no learning |
| RuVectorIntelligence (SONA + attention + HNSW) | Never imported by claude-flow |
| AgentDB controllers (Reflexion, Voyager, do-calculus) | Zero imports from claude-flow |
| Multi-provider routing (4+ providers) | Unused — claude-flow routes haiku/sonnet/opus only |
| agentdb RL algorithms (Q-Learning, SARSA) | Never called |
| Semantic embeddings (MiniLM-L6-v2 384-dim) | Broken — hash-based fallback runs instead |

### Updated Dependency Value Assessment

| Package | Previous Assessment | Revised Assessment |
|---------|-------------------|-------------------|
| `@ruvector/core` | HIGH | HIGH — ~60x faster memory search (confirmed) |
| `@ruvector/attention` | MEDIUM | MEDIUM — used by training service |
| `@ruvector/sona` | MEDIUM | MEDIUM — used by training service |
| `agentdb` | HIGH | LOW (unused) — zero imports from claude-flow |
| `agentic-flow` | LOW-MEDIUM | LOW — only provides embedding fallback + read-only memory retrieval |
| `better-sqlite3` | HIGH | HIGH — persistence backbone |

---

## 15. Complete ruvnet Ecosystem Map (2026-02-08)

See also: [ruvector-analysis.md](./ruvector-analysis.md),
[agentic-flow-analysis.md](./agentic-flow-analysis.md)

```
┌─────────────────────────────────────────────────────────┐
│  claude-flow v3.1.0 (CLI + 170 MCP tools)              │
│  ACTUALLY USES: LocalReasoningBank, memory system,      │
│  hooks, agent templates, @ruvector/core HNSW            │
├─────────────────────────────────────────────────────────┤
│  agentic-flow v2.0.6 (574 MB, 96% bundled deps)        │
│  PROVIDES TO CLAUDE-FLOW: embedding fallback,           │
│  retrieveMemories() read-only, @ruvector transitive deps│
│  UNUSED: learning pipeline, multi-provider routing,      │
│  Claude Agent SDK, swarm coordination, MCP tools        │
├─────────────────────────────────────────────────────────┤
│  agentdb v2.0.0-alpha.3.4 (68 MB)                      │
│  UNUSED BY CLAUDE-FLOW: ReflexionMemory, SkillLibrary, │
│  CausalMemoryGraph, all 23 controllers, RL algorithms   │
├─────────────────────────────────────────────────────────┤
│  @ruvector/* native binaries (15+ MB compiled Rust)     │
│  USED: core (HNSW), attention (training), sona (training)│
│  ~60x speedup for memory search over JS brute-force     │
└─────────────────────────────────────────────────────────┘
```

---

## 16. AgentDB Deep-Dive (2026-02-08)

See also: [agentdb-analysis.md](./agentdb-analysis.md)

### What AgentDB Is

AgentDB (v2.0.0-alpha.3.4, 68 MB) is a **frontier memory database for AI agents**
with 23 controllers implementing algorithms from published papers. It is the most
genuinely implemented package in the ruvnet ecosystem — unlike agentic-flow's stubs
or ruvector's third-party wrappers.

### Key Findings

**23 Controllers Assessed**: 18 GENUINE, 3 PARTIAL, 0 STUBS (9,697 lines total)

Strongest subsystems:
- **Causal inference pipeline** (4 controllers): CausalMemoryGraph → NightlyLearner →
  CausalRecall → ExplainableRecall. Uses doubly robust estimation (Robins 1994),
  Merkle-proof provenance, utility-based reranking.
- **Reflexion + Voyager** (2 controllers): ReflexionMemory (815 lines, arxiv:2303.11366)
  and SkillLibrary (697 lines, arxiv:2305.16291). Functional episodic memory with
  self-critique and automated skill extraction.
- **Attention** (4 controllers): Textbook Scaled Dot-Product, Cross-Attention,
  Multi-Head Attention (Vaswani 2017), plus Flash Attention and GraphRoPE.
- **LearningSystem** (928 lines): Genuine tabular Q-Learning and SARSA, but its
  "DQN/PPO/Actor-Critic" modes are misleadingly named tabular approximations.

Partial implementations:
- QUIC server/client: TCP-based, not real QUIC
- WASMVectorSearch: WASM module doesn't exist
- EnhancedEmbeddingService: WASM acceleration unavailable

### Security Architecture

Defense-in-depth with 4 layers:
1. Input validation (whitelist SQL/XSS/null-byte prevention)
2. Vector validation (NaN/Inf checking, Cypher injection prevention)
3. Path security (traversal blocking, symlink protection, atomic writes)
4. Resource limits (16GB memory cap, token bucket rate limiting, circuit breaker)

Plus: Argon2id auth, JWT tokens, API keys, Express rate limiting, Helmet.js headers.

### Claude-Flow Integration Status

**Zero imports.** Claude-flow does not use any agentdb controller. AgentDB loads as a
transitive dependency (via agentic-flow), prints `[AgentDB Patch] Controller index not
found`, and sits idle. Claude-flow's own LocalReasoningBank, memory system, and hooks
handle all functionality that agentdb could theoretically provide.

### The Three ReasoningBanks (Updated)

| ReasoningBank | Location | Store | Used? |
|--------------|----------|-------|-------|
| claude-flow's | hooks-tools.js | In-memory Map + JSON | **YES** |
| agentic-flow's | reasoning-bank.js | SQLite (DeepMind paper) | Partially |
| agentdb's | ReasoningBank.js | SQLite + embeddings | **NO** |

### What AgentDB Offers If Integrated

If claude-flow V3 adopted agentdb's controllers, it would gain:
- **ReflexionMemory**: Agent self-critique with episodic replay (paper-backed)
- **SkillLibrary**: Automated skill extraction and consolidation
- **CausalMemoryGraph**: Causal inference for understanding action→outcome
- **HybridSearch**: BM25 + vector fusion with RRF/linear/max methods
- **LearningSystem**: Tabular RL (Q-Learning/SARSA) for policy optimization
- **Quantization**: 4-32x memory reduction for large vector stores
- **ExplainableRecall**: Merkle-proof provenance for auditable retrieval

### Ecosystem Position

```
agentdb provides → 23 controllers, 33 MCP tools, 24 simulations, browser build
agentdb depends on → better-sqlite3, @ruvector/core, @xenova/transformers
agentdb bundled by → agentic-flow (packages/agentdb)
agentdb used by → nobody (loaded but unused by claude-flow)
```

---

## 17. Complete Ecosystem Audit Summary (2026-02-08)

Four research files now cover the entire ruvnet stack:

| Analysis File | Package | Key Verdict |
|--------------|---------|-------------|
| [claude-flow-analysis.md](./claude-flow-analysis.md) | claude-flow v3.1.0 | The orchestrator. 170 MCP tools, hooks, routing. |
| [ruvector-analysis.md](./ruvector-analysis.md) | @ruvector/* | 72 Rust crates. Core HNSW wraps hnsw_rs. ~60x speedup. |
| [agentic-flow-analysis.md](./agentic-flow-analysis.md) | agentic-flow v2.0.6 | 574MB package. Mostly stubs. Provides embedding fallback. |
| [agentdb-analysis.md](./agentdb-analysis.md) | agentdb v2.0.0-alpha.3.4 | Most genuine code. 23 controllers, 18 GENUINE. Unused. |

### What Actually Runs vs What Exists

| Capability | EXISTS in ecosystem | RUNS in claude-flow |
|-----------|-------------------|-------------------|
| HNSW vector search | ruvector (Rust) + agentdb (HNSWLibBackend) + claude-flow (JS) | YES via @ruvector/core |
| Episodic memory | agentdb ReflexionMemory (815 lines, paper-backed) | NO — claude-flow has simpler memory.db |
| Pattern storage | 3 ReasoningBanks (agentdb + agentic-flow + claude-flow) | YES — only claude-flow's Map+JSON |
| Causal inference | agentdb CausalMemoryGraph + NightlyLearner | NO |
| Attention mechanisms | agentdb (4 controllers) + ruvector (Rust) + agentic-flow (stubs) | Partially — JS Flash Attention in claude-flow |
| Reinforcement learning | agentdb LearningSystem (Q-Learning/SARSA, 928 lines) | NO |
| Multi-provider LLM routing | agentic-flow (5 providers) + agentdb (LLMRouter) | NO — claude-flow uses own model-router |
| Semantic embeddings | agentdb EmbeddingService + agentic-flow embeddings | Broken — hash-based fallback |
| Security (auth, JWT, audit) | agentdb (4 layers, Argon2id, JWT, rate limiting) | NO — claude-flow has no auth layer |
| Hybrid search (BM25+vector) | agentdb HybridSearch with RRF fusion | NO |
| Vector quantization | agentdb (8-bit, 4-bit, Product Quantization) | NO |
| Browser build | agentdb (35KB UMD, WASM loader) | NO |
| 24 simulation scenarios | agentdb (stock market, swarm, consciousness) | NO |

### Revised Dependency Value Assessment

| Package | Assessment | Justification |
|---------|-----------|---------------|
| `@ruvector/core` | **HIGH** | Only native HNSW backend that works. ~60x speedup. |
| `better-sqlite3` | **HIGH** | Persistence backbone for memory.db |
| `agentdb` | **UNUSED but HIGH POTENTIAL** | 18 genuine controllers with paper-backed algorithms. Best V3 integration candidate. |
| `agentic-flow` | **LOW** | 574MB for embedding fallback. Most features unused/broken. |
| `@ruvector/attention` | **MEDIUM** | Used by training service, broken for direct use |
| `@ruvector/sona` | **MEDIUM** | Used by training service |
| `@ruvector/gnn` | **LOW** | Broken API, JS wrapper fallback |

---

## 18. CLI Command Deep-Read (R17, 2026-02-14)

**Session**: R17 | **Files**: 45 | **LOC**: 33,929 | **Findings**: 79 (10 CRITICAL, 17 HIGH, 37 MEDIUM, 15 INFO)

Deep-read of ALL 37 unread CLI command files + 8 init system files. This covers every `dist/src/commands/*.js` file and the complete `dist/src/init/` system.

### Command Implementation Quality Matrix

| Command | LOC | Real % | Key Insight |
|---------|-----|--------|-------------|
| **analyze.js** | 1,823 | 70% | Real AST/graph analysis via ruvector. `code` and `deps` subcommands are stubs. |
| **embeddings.js** | 1,576 | 95% | Most complete command. 14 subcommands, real HNSW via @ruvector/core, sql.js search. |
| **neural.js** | 1,448 | 90% | Real WASM training, Ed25519 signing for exports, PII stripping. Best security. |
| **memory.js** | 1,268 | 85% | Real sql.js with 9-table schema. cleanup/compress delegate to MCP. |
| **init.js** | 964 | 85% | Wizard + upgrade + codex modes. Windows execSync bugs. |
| **session.js** | 760 | 80% | Real save/restore/export. YAML import broken (uses JSON.parse). |
| **mcp.js** | 700 | 50% | toggle non-functional, logs hardcoded, fallback tool list stale. |
| **task.js** | 671 | 90% | Rich task model (10 types, 4 priorities, dependencies). |
| **agentdb.js** | 625 | 90% | Real reflexion memory: episodes, skills, hybrid search. |
| **workflow.js** | 617 | 60% | Template create never saves. Metadata hardcoded in CLI. |
| **status.js** | 584 | 70% | Real MCP aggregation. Performance metrics are hardcoded marketing strings. |
| **performance.js** | 579 | 75% | Real benchmarks with percentiles. optimize/bottleneck are stubs. |
| **security.js** | 575 | 45% | Real npm audit + secret scan. CVE/STRIDE/audit are static examples. |
| **doctor.js** | 571 | 95% | Excellent parallel health checks. npx cache detection. Auto-fix. |
| **issues.js** | 567 | 95% | Real ADR-016 claim system with kanban board view. |
| **completions.js** | 539 | 100% | Complete for 4 shells. Hardcoded command lists (maintenance burden). |
| **benchmark.js** | 459 | 80% | Real neural/memory benchmarks with fallbacks. |
| **start.js** | 418 | 85% | Full MCP integration, daemon mode, PID management. |
| **config.js** | 406 | 10% | ENTIRELY STUB. init/get/set/export/import all fake persistence. |
| **migrate.js** | 410 | 20% | V2→V3 migration steps are hardcoded stubs. Good breaking changes docs. |
| **claims.js** | 373 | 40% | Real wildcard evaluation from config. grant/revoke don't persist. |
| **index.js** | 366 | 100% | Lazy loading saves ~200ms startup. Clean architecture. |
| **deployment.js** | 289 | 10% | All deployment steps simulated with setTimeout(). |
| **update.js** | 276 | 95% | Real npm update checking with rate limiting. |
| **progress.js** | 259 | 100% | Real MCP integration. |
| **providers.js** | 232 | 30% | Hardcoded provider data, simulated config updates. |
| **categories.js** | 178 | 100% | Clean DDD-aligned command taxonomy. |

### RuVector PostgreSQL Bridge (9 files, 4,211 LOC)

| Command | LOC | Quality |
|---------|-----|---------|
| setup.js | 765 | Real SQL generation but version/type mismatches |
| backup.js | 746 | Real backup with SQL/JSON/CSV + compression |
| optimize.js | 503 | Real pg health analysis + tuning recommendations |
| migrate.js | 481 | 6 predefined migrations, checksum system |
| benchmark.js | 480 | Real perf benchmarking with percentiles |
| status.js | 456 | Real connection + schema health checks |
| init.js | 431 | Real schema creation with interactive prompts |
| import.js | 349 | Real sql.js→PostgreSQL import |
| index.js | 129 | Clean command coordinator |

### CRITICAL Cross-File Bug: RuVector Extension Confusion

The ruvector PostgreSQL bridge has a **systemic inconsistency** across all 9 command files:

- `setup.js` creates `ruvector(384)` types and `CREATE EXTENSION ruvector`
- `init.js` creates `vector(${dimensions})` types and `CREATE EXTENSION vector`
- `import.js` uses `ruvector(384)` hardcoded
- `migrate.js` uses `vector(1536)` hardcoded
- `benchmark.js` uses `vector(${dimensions})`

**Impact**: Database initialization will fail or use the wrong extension. Dimension mismatches (384 vs 1536 vs configurable) will cause import/migration failures.

### CRITICAL: config.js Has Zero Persistence

The `config` command (`init`, `get`, `set`, `export`, `import`) is entirely a UI shell:
- `init` creates a config object but never writes to disk
- `get` returns hardcoded values, doesn't read any file
- `set` prints success but doesn't persist
- `export`/`import` don't read or write files

### Key Architectural Observations

1. **MCP-first architecture**: All core commands delegate to MCP tools via `callMCPTool()`. The CLI is a thin presentation layer.
2. **Graceful degradation everywhere**: Dynamic imports with null fallback for ruvector, aidefence, codex.
3. **Lazy loading optimization**: Command index uses on-demand loading, saving ~200ms startup time.
4. **Real implementations exceed expectations**: embeddings (95%), neural (90%), agentdb (90%), doctor (95%) are genuinely functional.
5. **Stub pattern**: Commands that manage external state (config, deployment, migration, providers) are mostly UI shells.
6. **Security highlight**: neural.js export has production-grade Ed25519 signing with PII stripping and secret detection.
