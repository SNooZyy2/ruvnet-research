# Agentic-Flow Repository Analysis

**Repository**: https://github.com/ruvnet/agentic-flow
**Published Package**: `agentic-flow` on npm
**Analyzed Version**: 2.0.6 (npm) / 2.0.2-alpha (repo root)
**Analysis Date**: 2026-02-08
**GitHub Stars**: 422 | **Forks**: 99 | **Contributors**: 1 (ruvnet, 342 commits)
**Created**: September 2, 2024
**License**: MIT

---

## Executive Summary

Agentic-flow is a TypeScript-based AI agent orchestration platform built on
Anthropic's Claude Agent SDK. It lives in a monorepo containing 5 sub-packages
(including AgentDB). Key findings:

1. **Agents are prompt templates, not code** -- 82 markdown files with system
   prompts, loaded by thin JS wrappers around the Claude Agent SDK.
2. **MCP tools shell out to `npx claude-flow@alpha`** -- circular dependency,
   not in-process implementations. The "213 tools" claim counts external packages.
3. **Three separate ReasoningBanks exist that share zero code** -- agentic-flow
   has the most sophisticated one (5 algorithms from DeepMind paper), but
   claude-flow never calls its learning functions.
4. **Real multi-provider routing** (Anthropic/OpenRouter/Gemini/ONNX) with
   proper API translation and fallback chains -- this is genuine value.
5. **QUIC transport and Federation are complete stubs** -- WASM binary exists
   but is never loaded; network calls return empty arrays.
6. **574 MB package is 96% bundled deps** -- onnxruntime-node alone is 513 MB.
7. **AgentDB is genuinely substantive** -- real controllers grounded in published
   papers (Reflexion, Voyager, Pearl's do-calculus), not thin abstractions.
8. **The gap between what EXISTS and what RUNS is significant** -- sophisticated
   learning algorithms exist in the codebase but claude-flow only uses the
   simplest local implementations.

---

## 1. Package Structure

### Monorepo Layout

```
agentic-flow/
  agentic-flow/         Main workspace package
  packages/
    agentdb/            Memory/vector database (npm: agentdb)
    agent-booster/      WASM code editing (npm: agent-booster)
    agentdb-onnx/       ONNX embeddings (not published)
    agentic-jujutsu/    Jujutsu VCS integration (npm: agentic-jujutsu)
    agentic-llm/        LLM routing layer
  src/                  Main source (TypeScript)
  crates/               Rust crates for @ruvector bindings
  docker/               Docker infrastructure
  bench/                Performance benchmarking
  docs/                 Documentation
```

### npm Package Metadata

| Package | Version | Size | Monthly Downloads |
|---------|---------|------|-------------------|
| `agentic-flow` | 2.0.6 | 574 MB installed (2.2 MB unpacked) | 84,541 |
| `agentdb` | 2.0.0-alpha.3.4 | 68 MB installed (909 KB unpacked) | 111,686 |
| `agent-booster` | 0.2.2 | Small | — |
| `agentic-jujutsu` | 2.3.6 | Small | — |

### Size Breakdown

| Component | Size | % of Total |
|-----------|------|------------|
| node_modules/ (bundled) | 553 MB | 96.3% |
| — onnxruntime-node | 513 MB | 89.4% |
| — better-sqlite3 | 27 MB | 4.7% |
| — fastmcp | 11 MB | 1.9% |
| dist/ (application code) | 13 MB | 2.3% |
| Agent markdown files | ~2 MB | 0.3% |

### Dependencies

**Runtime**: @anthropic-ai/sdk, @anthropic-ai/claude-agent-sdk, claude-flow,
fastmcp, express, tiktoken, zod, axios, onnxruntime-node, better-sqlite3,
@google/genai

**Optional**: @ruvector/core, @ruvector/sona, @ruvector/attention, ruvector,
@ruvector/edge-full, @ruvector/router, @ruvector/ruvllm, @ruvector/tiny-dancer

---

## 2. Agent System

### What's Advertised
- "66 specialized self-learning agents"
- Autonomous multi-agent swarms
- Queen/worker hierarchy

### What the Code Shows

**82 markdown agent definition files** in `.claude/agents/` organized across
20 categories (analysis, architecture, consensus, core, development, etc.).

Each file has YAML frontmatter with name, description, capabilities, hooks,
followed by a system prompt body. Example structure:

```yaml
---
name: security-architect
type: specialized
capabilities: [threat-modeling, security-review, CVE-detection]
hooks:
  pre: "claude-flow hooks pre-task"
  post: "claude-flow hooks post-task"
---
You are a security architecture specialist...
```

**7 JavaScript agent runners** that load these templates:

| Runner | Lines | What It Does |
|--------|-------|-------------|
| `claudeAgent.js` | 335 | Main runner via Claude Agent SDK `query()` |
| `claudeAgentDirect.js` | ~200 | Raw Anthropic SDK with streaming, no tools |
| `directApiAgent.js` | ~250 | Anthropic SDK with 7 custom tools that shell out to `npx claude-flow@alpha` |
| `codeReviewAgent.js` | ~50 | System prompt: "You review diffs and point out risks" |
| `webResearchAgent.js` | ~50 | System prompt: "You perform fast web-style reconnaissance" |
| `dataAgent.js` | ~50 | System prompt: "You analyze tabular data and produce a short brief" |
| `claudeFlowAgent.js` | ~100 | Agent using claude-flow MCP tools |

**Assessment**: Agents are entirely prompt-driven. The "self-learning" and
"GNN-enhanced" references in system prompts are aspirational documentation,
not executed code. The 77-82 markdown files are useful as well-crafted system
prompts, but they are not programmatic agent implementations.

---

## 3. Orchestration Logic

### RuvLLMOrchestrator (`dist/llm/RuvLLMOrchestrator.js`)

The most sophisticated orchestration code:
- `selectAgent()`: Embeds task, searches ReasoningBank, applies SONA weighting,
  routes via "FastGRNN"
- `decomposeTask()`: Estimates complexity, splits by sentences/conjunctions
- `recordOutcome()`: Updates performance metrics

**Reality check**:
- "TRM" (Tiny Recursive Models) = word count + keyword heuristics
- "SONA adaptation" = `weights[i] += 0.01` (uniform across 384 dimensions)
- "FastGRNN" = sort-by-score-and-pick-top
- Task decomposition splits by periods and "and/then/after" keywords
- No LLM-based decomposition, no result synthesis between agents

### Main Entry (`dist/index.js`)

`runParallelMode()` runs 3 agents via `Promise.all()` then concatenates outputs
with section headers. No synthesis, reconciliation, or feedback loops.

---

## 4. Multi-Provider Routing (REAL)

The `ModelRouter` at `dist/router/router.js` with 4+ providers:

| Provider | File | Real? | Tool Support |
|----------|------|-------|-------------|
| Anthropic | `providers/anthropic.js` | YES | YES |
| OpenRouter | `providers/openrouter.js` | YES (converts Anthropic→OpenAI format) | YES |
| Gemini | `providers/gemini.js` | YES (via @google/genai) | NO |
| ONNX Local | Referenced | Partial | N/A |
| Requesty | `proxy/requesty-proxy.js` | YES (proxy) | Via proxy |

**Routing modes**: manual, rule-based, cost-optimized, performance-optimized.
Fallback chain with automatic retry. Metrics tracking per provider.

**Assessment**: This is real, functional code. The Anthropic-to-OpenAI format
translation for OpenRouter is properly implemented with streaming support.
This is the most unique capability agentic-flow provides.

---

## 5. MCP Tools

### What's Advertised
- "213 MCP tools"

### What the Code Shows

**In-SDK MCP server** (`dist/mcp/claudeFlowSdkServer.js`): 9 tools, ALL of
which shell out to `npx claude-flow@alpha` via `execSync`:
- memory_store, memory_retrieve, memory_search
- swarm_init, agent_spawn, task_orchestrate, swarm_status
- agent_booster_edit_file, agent_booster_batch_edit (stubs)

**FastMCP stdio server** (`dist/mcp/fastmcp/servers/stdio-full.js`): 11 tools,
same `execSync` pattern.

The "213 tools" count comes from combining tools from external packages
(`claude-flow@alpha` provides 170+, `flow-nexus@latest` adds more). Agentic-flow
itself defines 9-11 tools, all of which are CLI command wrappers.

---

## 6. Three ReasoningBanks (That Share Zero Code)

This is the most important architectural finding.

### A. agentic-flow's ReasoningBank (`dist/reasoningbank/`)

**Based on**: arXiv:2509.25140 (Google DeepMind)

Implements 5 algorithms from the paper:

| Algorithm | File | What It Does | Fallback Without API Key |
|-----------|------|-------------|------------------------|
| **Retrieve** | `core/retrieve.js` | MMR diversity-aware retrieval with 4-factor scoring (similarity, recency, reliability, diversity) | Still works (local SQLite) |
| **Judge** | `core/judge.js` | LLM-as-Judge trajectory evaluation via ModelRouter | Returns heuristic confidence=0.5 |
| **Distill** | `core/distill.js` | LLM-based memory extraction from trajectories, PII scrubbing | Returns empty array `[]` |
| **Consolidate** | `core/consolidate.js` | Dedup (cosine >= 0.95), contradiction detection, pruning (180 days) | Works (pure computation) |
| **MaTTS** | `core/matts.js` | Memory-aware Test-Time Scaling (parallel rollouts or sequential refinement) | Requires LLM |

**Database**: SQLite at `.swarm/memory.db`. Tables: `patterns`, `pattern_embeddings`,
`pattern_links`, `task_trajectories`, `matts_runs`.

**Key limitation**: Without an API key (OPENROUTER, ANTHROPIC, or GEMINI), Judge
falls back to heuristic and Distill returns nothing. Real learning requires LLM access.

### B. agentdb's ReasoningBank (`agentdb/dist/src/controllers/ReasoningBank.js`)

**Completely different design**:
- v1: SQLite `reasoning_patterns` table (different schema from agentic-flow)
- v2: Uses abstract `VectorBackend` for 8x faster search
- Optional GNN enhancement via `LearningBackend`
- Stores task type, approach text, success rate, average reward
- `trainGNN()` triggers actual GNN training

**Not used by claude-flow** (grep for agentdb imports returned zero matches).

### C. claude-flow's LocalReasoningBank (`intelligence.js`)

**The simplest and the one actually used**:
- In-memory Map + Array with JSON file persistence at `~/.claude-flow/neural/patterns.json`
- O(1) store, O(n) linear scan search
- No SQLite, no LLM calls, no consolidation, no judge, no distill
- Achieves "<0.05ms" because it barely does anything
- This is what `claude-flow hooks intelligence *` commands invoke
- The 50 patterns in MEMORY.md are from THIS system

---

## 7. Learning and Intelligence Systems

### agentdb's "9 RL Algorithms"

| Claimed Algorithm | Actual Implementation |
|-------------------|----------------------|
| Q-Learning | Real: standard Q(s,a) with alpha, gamma |
| SARSA | Simplified: uses current Q-value as next-state |
| DQN | Fake: just reads Q-values, no neural network |
| Policy Gradient | Average reward tracking |
| Actor-Critic | Same as Policy Gradient |
| PPO | Same as Policy Gradient |
| Decision Transformer | Returns avg_reward (no transformer) |
| MCTS | UCB1 formula only (no tree search) |
| Model-Based | Returns avg_reward (no model) |

All use TD-error Q-value updates regardless of name. The algorithm names are cosmetic.

### RuVectorIntelligence (`dist/intelligence/RuVectorIntelligence.js`)

The most sophisticated implementation, integrating real Rust/WASM packages:
- @ruvector/sona SonaEngine (MicroLoRA, BaseLoRA, EWC++)
- @ruvector/attention (MultiHead, Flash, Hyperbolic, MoE, GraphRoPE)
- ruvector core (HNSW)
- Background learning: `setInterval(sona.tick, 60000)`
- Quality-gated adaptations: microLora always, baseLora >= 0.7, EWC++ >= 0.8

**BUT claude-flow never imports this module.** It only lives in agentic-flow.

### Embedding Fallback Chain

Four separate embedding implementations:

| System | Model | Dimensions | Used When |
|--------|-------|------------|-----------|
| ONNX via @xenova/transformers | MiniLM-L6-v2 | 384 | First choice (broken: "downloadModel is not a function") |
| agentic-flow via ruvector ONNX | Various | 384-768 | Second choice |
| Hash-based (sin/cos) | N/A | 128-384 | **What actually runs** |
| Simple hash (different algo) | N/A | 256 | EmbeddingService fallback |

**In practice**: ONNX model download fails, so hash-based embeddings run.
These produce vectors in the right dimension but with **no semantic meaning** --
similar text will NOT have similar embeddings.

---

## 8. Swarm Coordination

### QUIC Coordinator (`dist/swarm/quic-coordinator.js`) — 460 lines

Real coordination logic with topology-aware routing (mesh/hierarchical/ring/star),
agent registration, message queuing, heartbeat. **BUT** depends on QUIC transport
that is a complete stub.

### P2P Swarm v2 (`dist/swarm/p2p-swarm-v2.js`) — 1786 lines

The largest file. Includes Ed25519 identity, X25519 session keys, message replay
protection, Gun-based WebRTC signaling, IPFS CID pointers. Sophisticated
cryptographic code. **Requires Gun/WebRTC infrastructure not available locally.**

### Attention Coordinator (`dist/coordination/attention-coordinator.js`) — 361 lines

Attention-based agent consensus, MoE routing, topology-aware coordination.
**Requires external attentionService** for actual computation.

### Assessment

All three swarm implementations are architecturally sophisticated but
non-functional in our environment due to missing dependencies (QUIC client,
Gun/WebRTC, attention service).

---

## 9. What's Stub vs Real

### REAL and Functional

| Component | Lines | Evidence |
|-----------|-------|---------|
| Multi-provider routing | ~800 | Real API translation, streaming, fallback |
| Agent prompt templates | 82 files | Well-crafted system prompts |
| Claude Agent SDK wrapper | 335 | Real integration with `query()` |
| ReasoningBank (retrieve) | 87 | Genuine MMR algorithm |
| CircuitBreakerRouter | 459 | Full state machine (CLOSED/OPEN/HALF_OPEN) |
| SemanticRouter | 291 | Real cosine similarity (but brute-force, not HNSW as claimed) |
| PII Scrubber | ~100 | 12 regex patterns for credentials/PII |
| Proxy layer | 9 files | Real HTTP proxies for Anthropic→OpenRouter/Gemini |
| Attention wrappers | 268 | Proper integration with @ruvector/attention |
| SONA service | 447 | Real integration with @ruvector/sona |

### STUB / PLACEHOLDER

| Component | Evidence |
|-----------|---------|
| QUIC transport | `loadWasmModule()` returns `{}`; comment: "will be implemented" |
| Federation Hub | `sendSyncMessage()` returns `[]`; connection is a boolean flag |
| HTTP/3 proxy | `encodeHttp3Request()` returns empty `Uint8Array` |
| Agent Booster | `dist/agent-booster/` directory doesn't exist |
| WASM ReasoningBank | Import path points to non-existent files |
| Billing system | Always uses in-memory storage; no real payment integration |

### MISLEADING

| Claim | Reality |
|-------|---------|
| "213 MCP tools" | 9-11 tools that shell out to `npx claude-flow@alpha` |
| "HNSW indexing" | SemanticRouter admits brute-force in code comments |
| "66 specialized agents" | 82 markdown prompt templates |
| "Self-learning agents" | Learning code exists but claude-flow doesn't call it |
| "GNN-Enhanced" | `RuvectorLayer` is a single-layer perceptron, not GNN |
| "9 RL algorithms" | All are tabular Q-value updates with different names |
| "Byzantine fault-tolerant consensus" | No Raft/PBFT found in agentic-flow dist |

---

## 10. AgentDB Deep-Dive

AgentDB is the most genuinely substantive component of the agentic-flow ecosystem.

### Database Schema

12 tables, 5 views, 4 triggers implementing 5 memory patterns:
1. **Reflexion Episodic Replay**: `episodes`, `episode_embeddings`
2. **Skill Library**: `skills`, `skill_links`, `skill_embeddings`
3. **Structured Mixed Memory**: `facts` (SPO triples), `notes`, `note_embeddings`
4. **Episodic Segmentation**: `events`, `consolidated_memories`
5. **Graph-Aware Recall**: `exp_nodes`, `exp_edges`, `exp_node_embeddings`

### Controllers (23 total)

| Controller | Lines | Based On | Assessment |
|------------|-------|----------|------------|
| **ReflexionMemory** | 815 | arxiv 2303.11366 | **GENUINE** -- episodic replay, 4 retrieval strategies, GNN enhancement, graph node creation |
| **SkillLibrary** | 697 | arxiv 2305.16291 (Voyager) | **GENUINE** -- pattern extraction, learning trends, composite scoring |
| **CausalMemoryGraph** | 602 | Pearl's do-calculus | **GENUINE** -- uplift modeling, A/B experiments, t-stats, p-values, recursive CTE chains |
| **AttentionService** | 517 | Transformer attention | PARTIAL -- JS fallback works; Flash/MoE require @ruvector/attention NAPI |
| **HNSWIndex** | 437 | HNSW | Real wrapper around hnswlib-node (C++) |
| **QUICServer** | 383 | QUIC | **STUB** -- comment says "Actual QUIC would use a library" |
| **QUICClient** | 489 | QUIC | **STUB** -- `sendRequest()` does `sleep(100)` and returns `{success: true}` |
| **SyncCoordinator** | 553 | — | PARTIAL -- real logic, but built on stub QUIC transport |
| ReasoningBank | ~400 | — | Real pattern store with optional GNN (see Section 6) |
| Other (14) | Varies | — | Mix of real and partial implementations |

### Vector Backend

Priority chain: RuVector (Rust NAPI) > RuVector (WASM) > hnswlib-node (C++) > None

**RuVectorBackend** (776 lines): Real implementation with semaphore concurrency,
BufferPool memory reuse, adaptive HNSW parameters, mmap support, path traversal
security, prototype pollution protection.

### Browser Build

Real but minimal: `agentdb.browser.js` (48 KB) and `.min.js` (23 KB) with
sql.js WASM SQLite fallback.

---

## 11. What Claude-Flow Actually Uses

| Component | Used? | How? |
|-----------|-------|------|
| agentic-flow ReasoningBank (retrieve) | Partially | `hooks.js` imports for `retrieveMemories()` in token-optimize hook |
| agentic-flow ReasoningBank (judge/distill/consolidate) | **NO** | Never called by claude-flow |
| agentdb (any controller) | **NO** | Zero imports found |
| RuVectorIntelligence | **NO** | Never imported |
| LocalReasoningBank (intelligence.js) | **YES** | What `claude-flow hooks intelligence *` uses |
| EmbeddingService | Indirectly | memory-initializer falls back to agentic-flow embeddings |
| Multi-provider routing | **NO** | claude-flow routes haiku/sonnet/opus only |
| Claude Agent SDK integration | **NO** | claude-flow uses its own Task tool |
| PII scrubber | Only if distill runs | Requires API key + explicit call |
| WASM ruvector-edge | At import time | Falls back to JS |
| @ruvector/core HNSW | Via memory system | memory-initializer uses VectorDb |
| @ruvector/attention | Via training service | ruvector-training.js uses FlashAttention, MoE |
| @ruvector/sona | Via training service | ruvector-training.js uses SonaEngine |

**Bottom line**: Claude-flow uses agentic-flow primarily for:
1. Embedding model access (fallback chain)
2. ReasoningBank `retrieveMemories()` (partial, read-only)
3. The @ruvector native binaries that come as transitive dependencies

It does NOT use the sophisticated learning pipeline, multi-provider routing,
Claude Agent SDK integration, or any AgentDB controllers.

---

## 12. Comparison: agentic-flow vs claude-flow

| Capability | claude-flow (CLI) | agentic-flow (npm) |
|---|---|---|
| Agent execution | Task tool → Claude Code subprocess | Claude Agent SDK `query()` |
| Agent definitions | 60+ CLI templates | 82 markdown prompt templates |
| Model routing | 3-tier (haiku/sonnet/opus) | 4+ providers (Anthropic/OpenRouter/Gemini/ONNX) |
| MCP tools | 170+ native in-process | 9-11 that shell out to claude-flow |
| Memory | SQLite + HNSW, persistent | Shells out to claude-flow memory |
| Learning | LocalReasoningBank (patterns.json) | Full pipeline (unused by claude-flow) |
| Swarm | In-process swarm init/lifecycle | QUIC/P2P/Attention (non-functional) |
| Unique value | Working CLI + hooks + tools | Multi-provider routing + @ruvector binaries |

---

## 13. PII Scrubbing

Located at `dist/reasoningbank/utils/pii-scrubber.js`. Regex-based with 12 patterns:

| Category | Replacement |
|----------|-------------|
| Email addresses | `[EMAIL]` |
| SSN | `[SSN]` |
| Anthropic/GitHub/Slack API keys | `[API_KEY]` |
| AWS keys | `[AWS_KEY]` |
| Credit card numbers | `[CREDIT_CARD]` |
| US phone numbers | `[PHONE]` |
| IP addresses | `[IP]` |
| URL tokens/keys | `[REDACTED]` |
| JWT tokens | `[JWT]` |

**Limitations**: SSN pattern `\d{9}` matches any 9-digit number. Phone pattern
can match arbitrary number sequences. No NER or contextual analysis.

---

## 14. Performance Claims vs Reality

| Claim | Reality |
|-------|---------|
| "Flash Attention 2.49x-7.47x speedup" | Refers to JS implementation vs its own naive path, not Rust native |
| "HNSW 150x-12,500x faster" | Theoretical HNSW vs brute-force; SemanticRouter admits using brute-force |
| "SONA <0.05ms adaptation" | LocalReasoningBank circular buffer, not real SONA |
| "GNN +12.4% recall" | No GNN actually runs in claude-flow context |
| "52x faster Agent Booster" | Package directory doesn't exist |
| "2,211 ops/sec SIMD" | From @ruvector native, not agentic-flow JS |

---

## 15. Recommendations

### What to Actually Use from agentic-flow

1. **Multi-provider routing** -- If you need OpenRouter/Gemini access, this is
   real working code. Currently unused by claude-flow.
2. **Agent prompt templates** -- 82 well-crafted system prompts for various roles.
3. **@ruvector native binaries** (transitive deps) -- The real performance value.

### What to Skip

1. **MCP tools** -- They just shell out to claude-flow, adding latency.
2. **Swarm coordination** -- Non-functional without QUIC/WebRTC infrastructure.
3. **The "213 tools" claim** -- It's 9-11 actual tools.
4. **RL algorithms in agentdb** -- Mostly cosmetic names over Q-value updates.

### What Could Be Unlocked

1. **agentic-flow ReasoningBank learning pipeline** -- Would need an API key
   (OPENROUTER_API_KEY) and explicit integration to call `runTask()`, `judge()`,
   `distill()`, `consolidate()`.
2. **RuVectorIntelligence** -- Has real SONA/attention integration but needs
   explicit import from claude-flow.
3. **AgentDB controllers** -- ReflexionMemory, SkillLibrary, CausalMemoryGraph
   are genuinely sophisticated but completely unused.
4. **Real embeddings** -- Fix ONNX model download to replace hash-based fallback
   with actual semantic embeddings (MiniLM-L6-v2 384-dim).
