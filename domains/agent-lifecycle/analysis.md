# Agent Lifecycle Domain Analysis

> **Priority**: HIGH | **Coverage**: 32% (LOC deep) | **Status**: In Progress
> **Last updated**: 2026-02-15 (Session R37 — Rust agent training, routing, and lifecycle data)

## Overview

Manages agent creation, spawning, execution, training, conversion, and template loading. 629 files / 120K LOC. 81 files DEEP-read. The largest domain by file count (tied with claude-flow-cli at 629 files) but inflated by hundreds of duplicate agent templates across packages.

## Architecture

Three distinct agent patterns coexist:

| Pattern | Component | Lifecycle | Key Feature |
|---------|-----------|-----------|-------------|
| **CLI-based** | AgentManager (452 LOC) | create→list→describe→delete | Filesystem scan, frontmatter parsing, deduplication |
| **Long-running** | LongRunningAgent (220 LOC) | start→execute→checkpoint→stop | Budget enforcement, provider failover, state snapshots |
| **Ephemeral** | EphemeralAgent (259 LOC) | spawn→execute→learn→destroy | Auto-expiry (300s), JWT tenant isolation, federation sync |

### Agent Loading Pipeline

```
agentLoader.js
  → Recursive scan: ~/.npm-global/.../agents/ (package)
  → Recursive scan: .claude/agents/ (local)
  → YAML frontmatter parsing for .md files
  → Deduplication: local overrides package (same relative path)
  → Returns: Map<name, AgentDefinition>
```

### Agent Conversion (Claude SDK)

```
agent-converter.js
  → 8-type inference (substring matching):
    researcher → coder → reviewer → tester → doc → architect → coordinator → analyst
  → Tool assignment per type
  → Model selection:
    - Opus: complex/architect/security keywords
    - Haiku: simple/quick keywords
    - Default: inherit
  → 60s TTL cache
```

## R15 Deep-Read: Core Agent Code (6 files)

### agent-manager.js (452 LOC) — REAL

- 9 public methods for full agent lifecycle
- Scans package + local `.claude/agents/` directories
- Deduplication: local agents override package agents via Map keys
- Interactive creation via Node.js `readline`
- Frontmatter parsing for `.md` agent files
- No external dependencies (Node.js built-ins only)

### sona-agent-training.js (382 LOC) — REAL with FABRICATED embeddings

- **AgentFactory** (EventEmitter): creates specialized SONA engines, trains on examples with quality scoring, pattern search via HNSW
- **CodebaseTrainer**: indexes code by chunking functions/classes, creates training data
- 6 pre-configured AgentTemplates: codeAssistant, chatBot, dataAnalyst, ragAgent, taskPlanner, domainExpert
- **FABRICATED**: `mockEmbedding()` at L290-297 uses `Math.sin(seed)*10000` — deterministic but meaningless 3072-dim output. Comment: "replace with actual embedding service in production"

### long-running-agent.js (220 LOC) — REAL

- Budget enforcement (cost + runtime limits)
- Periodic checkpointing with state snapshots
- Provider failover via ProviderManager
- Graceful shutdown with timer cleanup (L205-217)
- Progress = completed/(completed+failed)
- Three lifecycle phases: start → execute → stop

### EphemeralAgent.js (259 LOC) — REAL with placeholders

- Auto-expiry timer (default 300s lifetime)
- JWT tenant isolation with agent ID format: `eph-{tenantId}-{timestamp}`
- Pre/post sync with federation hub
- Learning episodes stored with quality threshold 0.7
- **Placeholder**: `FederationHub.connect()` simulates connection ("simulate connection with WebSocket fallback")
- Memory defaults to `:memory:` (in-memory only via better-sqlite3)

### agent-converter.js (188 LOC) — REAL

- Converts agentic-flow agent definitions to Claude SDK format
- 8-type inference via substring matching
- Tool assignment per agent type
- Model selection: Opus (complex/architect/security), Haiku (simple/quick), inherit
- 60s TTL cache for converted agents
- 5 essential agents: researcher, coder, reviewer, tester, documenter

### agentLoader.js (162 LOC) — REAL

- Recursive directory scan with `readdirSync({recursive: true})`
- YAML frontmatter parsing with tool comma-split
- Deduplication: local agents override package agents with same relative path
- Logs override decisions for debugging

## R15 Deep-Read: MCP Tools & Booster (10 files)

### claudeFlowAgent.js (116 LOC) — REAL Claude Agent SDK

The most important integration file: actual Claude Agent SDK usage.

- Uses `@anthropic-ai/claude-agent-sdk` `query()` method with streaming
- Memory context construction from stored memories
- Coordination context for multi-agent workflows
- Stream processing loop for incremental results
- 4 exported agent functions: `claudeFlowAgent`, `memoryResearchAgent`, `orchestratorAgent`, `hybridAgent`
- `withRetry()` wrapper for resilience

### agentic_flow_quic.js (780 LOC) — REAL WASM bindings

- wasm-bindgen generated binding layer for QUIC transport
- Memory management with TextEncoder/TextDecoder UTF-8 marshalling
- Closure management with FinalizationRegistry
- Exports: `createQuicMessage()`, `defaultConfig()`, `WasmQuicClient` class
- Loads `.wasm` binary via `fs.readFileSync` + `WebAssembly.Module`

### agent-booster-migration.js (343 LOC) — FABRICATED

Performance metrics are entirely fabricated:
- L121: `traditionalTime = 352` (hardcoded milliseconds)
- L150: `editTraditional()` uses `sleep(352)` as artificial delay
- L187: `isWasmAvailable()` returns hardcoded `true`
- L198: `avgSpeedupFactor = 352` (constant)
- L265: `migrateCodebase()` returns estimated results without actual migration
- Module "agent-booster" does not exist in dependencies (L18)

### agent-booster-tools.js (274 LOC) — BROKEN

- Three MCP tools (`agent_booster_edit_file`, `batch_edit`, `parse_markdown`) are non-functional
- `getBooster()` at L13 imports missing "agent-booster" module
- `booster.apply()` called without null check at L122 and L172 — runtime crash
- Markdown regex parser (L225) is REAL but delegates to broken handler
- Speedup calculation at L216: `Math.round(352/avgLatency)` uses hardcoded 352

### MCP Agent Tools (5 files, mixed quality)

| File | LOC | Status | Key Issue |
|------|-----|--------|-----------|
| **add-command.js** | 118 | REAL | Creates `.claude/commands/{name}.md`. Production-ready. |
| **add-agent.js** | 108 | REAL | Creates `.claude/agents/{category}/{name}.md`. Production-ready. |
| **list.js** | 83 | BROKEN | CLI wrapper: `execSync("npx agentic-flow --list")`. Not MCP-native. |
| **parallel.js** | 64 | BROKEN | Claims 3-agent parallelization but just runs generic `npx` command. |
| **execute.js** | 57 | BROKEN | CLI wrapper via `execSync`. Fails in isolated environments. |

### multi-agent-orchestration.js (46 LOC) — Example code

Demonstration of sequential multi-agent workflow using goal-planner and code-analyzer agents. Works if agents are available.

## R15 Deep-Read: V3 Agent Templates (6 templates)

All v3 templates follow a common pattern:
- YAML frontmatter (model, color, description)
- Capabilities list and tool requirements
- Pre/post hooks referencing MCP tools
- Examples with structured workflows

### Reality Assessment

| Template | LOC | Model | % Real | Key Algorithms |
|----------|-----|-------|--------|----------------|
| **performance-engineer** | 1,234 | Opus | ~30% | Flash Attention (2.49x-7.47x claimed), WASM SIMD, token optimization |
| **memory-specialist** | 996 | Opus | ~35% | HNSW indexing, EWC++, vector quantization, hybrid SQLite+AgentDB |
| **collective-intelligence** | 1,002 | Opus | ~25% | Byzantine consensus (2/3+1 threshold), CRDT sync, neural patterns |
| **security-architect** | 868 | Opus | ~60% | STRIDE/DREAD (textbook-correct), CVE tracking, zero-trust |
| **security-auditor** | 708 | Opus | ~55% | OWASP compliance, vulnerability detection, code scanning |
| **goal/agent** | 824 | Sonnet | ~65% | A* search (textbook-correct), PageRank, OODA loop, behavior trees |

**Key observation**: Security templates are the most grounded (real methodologies). ML/attention templates are the most aspirational (unvalidated performance claims). Goal planner has genuinely correct A* and PageRank algorithms.

## R15 Deep-Read: SPARC & Specialized Templates (9 templates)

| Template | LOC | % Real | Key Feature |
|----------|-----|--------|-------------|
| **validate-agent.sh** | 218 | 95% | Production-ready agent validation. YAML frontmatter checks, field validation, regex name enforcement. |
| **migration-plan** | 750 | 80% | Highest quality template. 8-category migration taxonomy, concrete agent definitions, backwards compatibility. |
| **benchmark-suite** | 673 | 70% | CUSUM change-point detection (REAL algorithm), load/stress testing, SLA validation. |
| **refinement** | 742 | ~45% | Red-Green-Refactor TDD (REAL examples), GNN search (ASPIRATIONAL). |
| **architecture** | 699 | ~40% | Microservices/K8s patterns (REAL), Flash Attention (ASPIRATIONAL). |
| **adaptive-coordinator** | 1,127 | ~40% | Dynamic topology switching (REAL), 5 attention mechanisms (ASPIRATIONAL). MoE routing scoring is implementable. |
| **workflow-automation** | 835 | ~35% | GitHub Actions YAML (REAL), gh CLI (REAL), GNN optimization (ASPIRATIONAL). |
| **performance-monitor** | 680 | ~30% | SLA thresholds (REAL), ensemble anomaly detection (ASPIRATIONAL). |
| **resource-allocator** | 682 | ~30% | Circuit breaker/bulkhead (REAL), genetic algorithm/LSTM/DQN (ASPIRATIONAL). |

**Pattern**: Templates that reference concrete tools (git, gh CLI, shell commands, standard patterns) are real. Templates that reference ML/attention/HNSW features are aspirational — they describe desired behavior that depends on MCP tools that may return fabricated results.

## CRITICAL Findings (2)

1. **agent-booster-migration.js entirely fabricated** — Hardcoded 352ms timing, `sleep(352)` as delay, `isWasmAvailable()` returns true unconditionally, migrateCodebase() returns estimates without migration. The "WASM Agent Booster" is a marketing facade.
2. **agent-booster-tools.js broken at runtime** — Three MCP tools crash on null reference. Missing "agent-booster" module. Any user calling these tools gets a runtime error.

## HIGH Findings (12)

1. **AgentManager is solid production code** — 9 methods, deduplication, frontmatter parsing, no external deps.
2. **Three agent lifecycle patterns coexist** — CLI-based, long-running, and ephemeral agents with no common interface.
3. **claudeFlowAgent.js is REAL Claude SDK integration** — Actual streaming, memory context, retry wrapper.
4. **WASM QUIC bindings are real generated code** — 780 LOC wasm-bindgen output with proper memory management.
5. **sona-agent-training mockEmbedding() is fabricated** — Math.sin-based 3072-dim output masquerading as real embeddings.
6. **LongRunningAgent has proper budget enforcement** — Cost limits, runtime limits, checkpointing, provider failover.
7. **EphemeralAgent federation sync is placeholder** — "Simulate connection" comment. In-memory only.
8. **3 MCP agent tools are broken CLI wrappers** — list.js, parallel.js, execute.js all use execSync("npx ...").
9. **2 MCP agent tools are production-ready** — add-command.js and add-agent.js create real files.
10. **V3 templates split 40/60 real/aspirational** — Security and planning templates are most grounded.
11. **validate-agent.sh is production-ready** — 95% real, proper YAML validation, regex enforcement.
12. **migration-plan.md is highest-quality template** — 80% real, concrete taxonomy, backwards compatibility.

## MEDIUM Findings (6)

1. sona-agent-training depends on @ruvector/sona SonaEngine. Code chunking uses regex with brace-counting.
2. EphemeralAgent federation is a placeholder ("simulate connection with WebSocket fallback").
3. agent-booster "agent-booster" module missing. Lazy loading prevents crash but all operations fail silently.
4. Performance claims in templates (2.49x-7.47x Flash Attention, 150x-12500x HNSW) are aspirational design targets stated as facts.
5. MoE routing in adaptive-coordinator is implementable (capability*0.5 + performance*0.3 + availability*0.2) but unvalidated.
6. resource-allocator and performance-monitor templates are ~70% aspirational ML features.

## Positive

- **AgentManager** is clean, well-structured production code with proper deduplication
- **LongRunningAgent** demonstrates real budget enforcement and graceful shutdown
- **claudeFlowAgent.js** is genuine Claude Agent SDK integration with streaming
- **WASM QUIC bindings** are real wasm-bindgen generated code
- **validate-agent.sh** is production-grade validation with comprehensive checks
- **migration-plan.md** provides a practical, concrete migration framework
- **benchmark-suite.md** has real CUSUM change-point detection algorithm
- **goal/agent.md** has textbook-correct A* search and PageRank algorithms
- **Security templates** use real STRIDE/DREAD/OWASP methodologies

## The Agent Template Problem

The agent-lifecycle domain is dominated by ~500+ agent template `.md` files duplicated across 3 packages:
- `~/.npm-global/lib/node_modules/@claude-flow/cli/.claude/agents/` (canonical)
- `~/node_modules/agentic-flow/.claude/agents/` (copy)
- `~/.claude/agents/` (local overrides)

Most unread files are these duplicates. The canonical set of ~60 unique templates has been partially analyzed. Further deepening should focus on:
1. **Core code files** not yet read (agent pools, health checks, scaling)
2. **The deduplication question** — how many truly unique templates exist vs copies
3. **Integration points** between the three agent patterns

## R37: Rust Agent Training + Lifecycle Data (Session 37)

### Overview

R37 deep-read of ruvllm crate files reveals the Rust-side agent lifecycle components: training data generators for agent specialization, model routing for agent selection, and workflow execution for agent task handling.

### Relevant Files (4 files, 5,365 LOC)

| File | LOC | Real% | Relevance to Agent Lifecycle |
|------|-----|-------|-------------|
| **claude_dataset.rs** | 1,209 | **75-80%** | Claude task dataset: 5 categories of agent task templates (coding, debugging, architecture, testing, documentation). 60+ templates for training agent specialization. Weak augmentation (5 word pairs only). |
| **model_router.rs** | 1,292 | **88-92%** | 7-factor complexity analyzer for routing tasks to agent types. Feedback tracking stores last 1000 predictions. Maps task characteristics → model selection (cost/latency constraints). |
| **reasoning_bank.rs** | 1,520 | **92-95%** | Pattern storage for agent decision-making. K-means clustering categorizes agent trajectories. EWC++ consolidation preserves learned patterns across agent sessions. |
| **claude_integration.rs** | 1,344 | **70-75%** | ClaudeModel enum with real API pricing and context windows. Agent workflow execution infrastructure. **CRITICAL**: execute_workflow SIMULATION — hardcoded 500 tokens, no real Claude API. |

### Cross-Domain Insights

1. **Agent training data**: claude_dataset.rs generates training examples for 5 agent categories. These templates mirror the agent types defined in claude-flow CLI (coder, reviewer, tester, architect, documenter). The Rust training pipeline could theoretically fine-tune agent behavior but depends on hash-based embeddings.

2. **Agent selection via model_router**: The 7-factor complexity analyzer in model_router.rs performs the same function as agent-converter.js (R15) but with ML-based analysis vs substring matching. Both select Opus for complex/architect tasks and Haiku for simple/quick tasks.

3. **Agent memory via reasoning_bank**: Each agent's trajectory (action → outcome) is stored and clustered via K-means. EWC++ prevents catastrophic forgetting across agent sessions — theoretically enabling agents to learn from past interactions.

### R37 Updated Findings

**HIGH** (+1):
13. **Rust agent training datasets are static templates** — claude_dataset.rs has 60+ templates but no mechanism to incorporate real agent execution outcomes. Training data is synthetic, not experiential. (R37)

**Positive** (+2):
- **reasoning_bank.rs** provides genuine trajectory-based learning for agents (K-means + EWC++) — the theoretical foundation for self-improving agents (R37)
- **model_router.rs** is a significantly more sophisticated agent routing system than the JS agent-converter.js substring matching (R37)

## Remaining Gaps

~545 files still NOT_TOUCHED, but most are:
- Duplicate agent templates across packages (~400+ estimated)
- Test files and examples (~50+)
- Build artifacts and type definitions
- Agent pool/scaling infrastructure (genuinely unread)
- Agent health monitoring and metrics (genuinely unread)
