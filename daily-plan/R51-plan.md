# R51 Execution Plan: MCP Bridge + Agent Lifecycle + Worker Pipeline + Coordination

**Date**: 2026-02-15
**Session ID**: 51
**Focus**: Deep-read the runtime execution layer — how claude-flow actually spawns agents, manages MCP connections, executes worker phases, and coordinates across swarm topologies. This is the "unlock" session for understanding how to use claude-flow to its max potential.

## Rationale

After 50 sessions (932 DEEP files, 658K LOC), we have strong coverage of the **crate-level libraries** (ruvector-core, SONA, temporal-tensor, ruQu, ReasoningBank) and the **CLI command surface** (hooks 98.1%, model-routing 94.9%). But we have a critical blind spot: **the actual runtime execution pipeline** — what happens between "user runs a command" and "agent produces output."

Specifically:
- `agent-manager.ts` (563 LOC) — how agents are CRUD-managed — **NEVER READ**
- `claude-code-wrapper.ts` (405 LOC) — how Claude Code itself is invoked — **NEVER READ**
- `embedding-service.ts` (456 LOC) — R20 identified this as ROOT CAUSE of broken AgentDB search (never initialized) — **NEVER READ THE SOURCE**
- `mcp-server.js` (610 LOC) — the actual entry point for `claude-flow mcp start` — **NEVER READ**
- Worker phase pipeline (consolidated-phases + phase-executors + custom-worker-factory = 1,661 LOC) — **ALL UNTOUCHED**
- 3 routing systems discovered (ADR-008 + LLMRouter + RuvLLMOrchestrator) but we've never read the actual router implementations

This session closes the loop from "what does the code say?" to "how do I actually use this?"

## Structure: 3 Parallel Clusters (25 files, ~11,687 LOC)

These clusters have **ZERO file overlap** and can be executed as parallel sub-sessions.

---

### Cluster A: Agent Lifecycle + MCP Entry Points (9 files, ~4,027 LOC)

The most critical cluster. Answers: "What happens when you spawn an agent? How does the MCP server start? How is Claude Code wrapped?"

| # | File ID | File | LOC | Package | Priority |
|---|---------|------|-----|---------|----------|
| 1 | 2166 | `dist/src/mcp-server.js` | 610 | claude-flow-cli | P0 |
| 2 | 10680 | `agentic-flow/src/cli/agent-manager.ts` | 563 | agentic-flow-rust | P0 |
| 3 | 10681 | `agentic-flow/src/cli/claude-code-wrapper.ts` | 405 | agentic-flow-rust | P0 |
| 4 | 10691 | `agentic-flow/src/cli/mcp-manager.ts` | 581 | agentic-flow-rust | P0 |
| 5 | 10710 | `agentic-flow/src/core/provider-manager.ts` | 579 | agentic-flow-rust | P1 |
| 6 | 10706 | `agentic-flow/src/core/embedding-service.ts` | 456 | agentic-flow-rust | P1 |
| 7 | 10708 | `agentic-flow/src/core/index.ts` | 152 | agentic-flow-rust | P1 |
| 8 | 10709 | `agentic-flow/src/core/long-running-agent.ts` | 299 | agentic-flow-rust | P2 |
| 9 | 10745 | `agentic-flow/src/mcp/claudeFlowSdkServer.ts` | 382 | agentic-flow-rust | P2 |

**Key questions**:
- `mcp-server.js`: What is the ACTUAL tool registry? How many tools are registered? Does it use fastmcp or raw JSON-RPC? How does `claude-flow mcp start` map to the MCP tools we see in Claude Code?
- `agent-manager.ts`: How are agents created, tracked, and terminated? Is there real process management (child_process, tokio::process) or is it in-memory state only? Does it connect to the worker registry (R31 DEEP)?
- `claude-code-wrapper.ts`: THIS IS THE KEY FILE. How does claude-flow invoke Claude Code? Is it `claude --prompt`? Does it pipe MCP? Does it manage multiple Claude Code instances? R31 said CLI = demonstration framework — does this wrapper confirm or reverse that?
- `mcp-manager.ts`: How are MCP servers discovered, connected, and managed? Does it handle the `claude mcp add` flow? Is there real connection pooling?
- `provider-manager.ts`: How are LLM providers selected? Does this implement the 3-tier model routing from ADR-008, or is it a separate system (making it the 4th parallel routing system)?
- `embedding-service.ts`: R20 found AgentDB search broken because EmbeddingService was never initialized. Does the SOURCE show why? Is there a real embedding model, or does it fall back to hash-based embeddings (9th occurrence)?
- `core/index.ts`: What does the core module actually export? This tells us the public API surface.
- `long-running-agent.ts`: How are agents that persist across multiple interactions managed? Is there real state persistence?
- `claudeFlowSdkServer.ts`: What SDK does the server expose? Does it differ from the standalone-stdio.ts (R37 DEEP)?

**Follow-up context**:
- R20: AgentDB search broken — ROOT CAUSE: EmbeddingService never initialized in claude-flow bridge
- R31: CLI = demonstration framework, swarm-coordination ~78%
- R40: LLMRouter NO ADR-008 connection — parallel routing system
- R44: RuvLLMOrchestrator is THIRD parallel routing system, zero ruvllm connection
- R50: spawn.rs (8-12%) COMPLETE FACADE — "In a real implementation" at L366
- R48: health-monitor.ts (99%), config-manager.ts (78%) — runtime services CAN be genuine

---

### Cluster B: Worker Execution Pipeline + MCP Servers (8 files, ~3,764 LOC)

How tasks flow from assignment to completion. Answers: "What phases does a task go through? How are workers created and dispatched? What do the MCP server transports actually implement?"

| # | File ID | File | LOC | Package | Priority |
|---|---------|------|-----|---------|----------|
| 10 | 10926 | `agentic-flow/src/workers/consolidated-phases.ts` | 600 | agentic-flow-rust | P0 |
| 11 | 10933 | `agentic-flow/src/workers/phase-executors.ts` | 558 | agentic-flow-rust | P0 |
| 12 | 10928 | `agentic-flow/src/workers/custom-worker-factory.ts` | 503 | agentic-flow-rust | P1 |
| 13 | 10932 | `agentic-flow/src/workers/mcp-tools.ts` | 429 | agentic-flow-rust | P1 |
| 14 | 10930 | `agentic-flow/src/workers/hooks-integration.ts` | 376 | agentic-flow-rust | P1 |
| 15 | 10752 | `agentic-flow/src/mcp/fastmcp/servers/stdio-full.ts` | 445 | agentic-flow-rust | P2 |
| 16 | 10750 | `agentic-flow/src/mcp/fastmcp/servers/http-streaming-updated.ts` | 445 | agentic-flow-rust | P2 |
| 17 | 10866 | `agentic-flow/src/routing/SemanticRouter.ts` | 408 | agentic-flow-rust | P2 |

**Key questions**:
- `consolidated-phases.ts`: What SPARC phases exist (Specification, Pseudocode, Architecture, Refinement, Completion)? Are they real execution stages with distinct logic, or a single pass-through? How does phase ordering work?
- `phase-executors.ts`: What does each phase executor DO? Does it call LLM APIs? Does it read/write files? Or are they stubs?
- `custom-worker-factory.ts`: How are custom workers created? Is there real process isolation (containers, subprocesses) or is everything in-memory? Does it use the worker-daemon (R31 DEEP) or container-worker-pool (R31 DEEP)?
- `mcp-tools.ts`: What MCP tools do workers expose? Is this the bridge between worker execution and MCP tool calls?
- `hooks-integration.ts`: How do hooks (pre-task, post-task, pre-edit, etc.) integrate with the worker pipeline? Are hooks executed synchronously in the pipeline, or fire-and-forget?
- `stdio-full.ts` vs `http-streaming-updated.ts`: Two MCP transport implementations. Are they both real? We already have standalone-stdio.ts DEEP (813 LOC) — how does stdio-full compare?
- `SemanticRouter.ts`: Does it implement real semantic routing (embedding similarity, intent classification) or keyword matching?

**Follow-up context**:
- Already DEEP: dispatch-service.ts (1,212 LOC), worker-registry.ts (662 LOC), worker-agent-integration.ts (613 LOC)
- Already DEEP: standalone-stdio.ts (813 LOC) — baseline for MCP server quality
- R31: workers/dispatch-service does real task queuing but limited by CLI framework
- R19: hook-pipeline CLOSED at 98.1% — but hooks-integration.ts was never read
- R40: agentic-flow = single-node task runner — do phases confirm this?

---

### Cluster C: Coordination + Core Subsystems (8 files, ~3,896 LOC)

How agents coordinate and what core wrappers they use. Answers: "Is there real distributed coordination? What do the attention mechanisms actually do? How many AgentDB wrapper layers exist?"

| # | File ID | File | LOC | Package | Priority |
|---|---------|------|-----|---------|----------|
| 18 | 10698 | `agentic-flow/src/coordination/attention-coordinator.ts` | 541 | agentic-flow-rust | P1 |
| 19 | 10889 | `agentic-flow/src/swarm/quic-coordinator.ts` | 584 | agentic-flow-rust | P1 |
| 20 | 10890 | `agentic-flow/src/swarm/transport-router.ts` | 472 | agentic-flow-rust | P1 |
| 21 | 10865 | `agentic-flow/src/routing/CircuitBreakerRouter.ts` | 575 | agentic-flow-rust | P1 |
| 22 | 10703 | `agentic-flow/src/core/agentdb-wrapper.ts` | 491 | agentic-flow-rust | P2 |
| 23 | 10701 | `agentic-flow/src/core/agentdb-fast.ts` | 444 | agentic-flow-rust | P2 |
| 24 | 10705 | `agentic-flow/src/core/attention-native.ts` | 359 | agentic-flow-rust | P2 |
| 25 | 10704 | `agentic-flow/src/core/attention-fallbacks.ts` | 430 | agentic-flow-rust | P2 |

**Key questions**:
- `attention-coordinator.ts`: Does this implement real multi-head attention for agent coordination (as claimed in ADRs), or is it threshold-based heuristics? Does it connect to the attention-native.ts implementation?
- `quic-coordinator.ts`: R48 found quic.ts (95%) has full reconciliation + Merkle + JWT — does the agentic-flow coordinator actually USE the QUIC layer, or is it another disconnected implementation?
- `transport-router.ts`: How does it route between QUIC, WebSocket, and in-memory transports? Is there real transport selection logic?
- `CircuitBreakerRouter.ts`: Does it implement real circuit breaker patterns (half-open state, failure counting, exponential backoff)? This is critical for production resilience.
- `agentdb-wrapper.ts` vs `agentdb-wrapper-enhanced.ts` (DEEP): What's the base wrapper? Does enhanced extend it, or is it a complete rewrite? How many wrapper layers exist between user code and actual AgentDB?
- `agentdb-fast.ts`: Is this a performance-optimized path that bypasses the wrapper? Does it use direct HNSW access?
- `attention-native.ts` + `attention-fallbacks.ts`: These should be the actual attention computation. Is native = WASM/SIMD and fallbacks = pure JS? Or is native = a stub pointing to non-existent native modules?

**Follow-up context**:
- R48: THREE disconnected AgentDB distributed layers (QUIC + libp2p + embeddings) — does coordination unify them?
- R44: p2p.rs (92-95%) REAL libp2p, swarm.rs (72%) mixed — is TS coordination connected to Rust P2P?
- R42: RAC (92%) highest quality single-file Rust — does attention-coordinator use RAC?
- R38: mincut-gated-transformer most novel — does attention-native implement it?
- R23: neural-network-implementation 90-98% — do fallbacks connect to this?

---

## Expected Outcomes

1. **Agent lifecycle map**: Complete picture of create → configure → execute → terminate flow
2. **MCP architecture diagram**: How `claude-flow mcp start` → tool registration → tool execution works
3. **Worker phase pipeline**: What SPARC phases exist and whether they're real execution stages
4. **Routing reality check**: How many routing systems actually exist and which one is active
5. **Coordination assessment**: Whether distributed coordination is real or in-memory-only
6. **Embedding service verdict**: Whether R20's root cause has been fixed or remains broken
7. **"Demonstration framework" final verdict**: R31's conclusion tested against the actual runtime code

## Domains to Update

| Domain | Files | Expected Updates |
|--------|-------|-----------------|
| `swarm-coordination` | 18-25 | Worker pipeline, coordination, transport routing |
| `memory-and-learning` | 22-25 | AgentDB wrappers, attention mechanisms |
| `cli-and-hooks` | 1, 4, 14 | MCP server, MCP manager, hooks-integration |

## Execution Notes

- **Parallel execution**: All 3 clusters can run simultaneously with zero file overlap
- **Model**: Sonnet for all clusters (standard deep-read complexity)
- **Estimated LOC**: ~11,687 across 25 files
- **Priority order if sequential**: Cluster A first (highest impact for "max potential"), then B, then C
- **Session DB**: Use session ID 51, create at start
