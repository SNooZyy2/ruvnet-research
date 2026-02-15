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

---

## R22: TypeScript Source Deep-Read (Session 27 — agentic-flow-rust)

> 54 files read, ~59K LOC, 150 findings across 5 agent clusters. Weighted average 83% real — highest quality batch analyzed.

### Intelligence Layer Architecture (10 files, 12,100 LOC) — 82% REAL

The agentic-flow intelligence subsystem has a well-layered architecture:

```
cli-proxy.ts (95%) ──> agent-booster-enhanced.ts (75%) ──> EmbeddingService.ts (80%) ──> EmbeddingCache.ts (90%)
                                                                ^
dispatch-service.ts (80%) ──> intelligence-bridge.ts (70%) ──> RuVectorIntelligence.ts (80%) ──> IntelligenceStore.ts (90%)
```

| File | LOC | Real % | Key Finding |
|------|-----|--------|-------------|
| **p2p-swarm-v2.ts** | 2,280 | 85% | Production Ed25519/X25519/AES-256-GCM crypto. Task execution STUB. Fake IPFS CIDs. |
| **EmbeddingService.ts** | 1,810 | 80% | Real ONNX embedding, K-means clustering, pretrain system. Hash-based simpleEmbed fallback. |
| **cli-proxy.ts** | 1,432 | 95% | Main CLI entry. Multi-provider proxy (OpenRouter, Gemini, ONNX, Requesty). QUIC transport. |
| **agent-booster-enhanced.ts** | 1,428 | 75% | Pattern caching (exact+fuzzy), 5-tier compression, co-edit graph. External npx dep. |
| **intelligence-bridge.ts** | 1,371 | 70% | Bridge to RuVectorIntelligence. 9 RL algorithms config-only. Math.random()*0.1 fabricated activations. |
| **RuVectorIntelligence.ts** | 1,200 | 80% | Core: SONA Micro-LoRA, 6 attention types, HNSW retrieval, LRU eviction (10K trajectories). |
| **dispatch-service.ts** | 1,212 | 80% | 12 worker types with real file analysis. Secret detection, dependency scanning. |
| **edge-full.ts** | 943 | 75% | WASM integration for 6 ruvector sub-modules. JS fallback for 5/6. Cypher throws without WASM. |
| **EmbeddingCache.ts** | 726 | 90% | 3-tier cache (native SQLite > WASM SQLite > Memory). SHA-256 keys, LRU eviction. |
| **IntelligenceStore.ts** | 698 | 90% | SQLite dual backend (sql.js > better-sqlite3 > no-op). **SQL injection risk** in incrementStat. |

### Proxy Layer (2 files, 1,655 LOC) — 91% REAL

Both proxy files are genuine HTTP proxies with streaming:

| File | LOC | Real % | Key Finding |
|------|-----|--------|-------------|
| **anthropic-to-requesty.ts** | 880 | 93% | Real Anthropic→OpenAI→Requesty proxy. API key prefix leaked in logs. |
| **anthropic-to-openrouter.ts** | 775 | 90% | ~95% identical to Requesty. Missing sanitizeJsonSchema(). No request timeout. |

### Agentic-Jujutsu Crate (10 files, ~11,300 LOC) — GENUINELY FUNCTIONAL

The agentic-jujutsu Rust crate is one of the most genuine packages in the ecosystem:

- **wrapper.rs** (1,300 LOC, 90%): `include_bytes!` embeds real jj binary at compile time. Build system downloads actual Jujutsu binary. All 15 JJ operations are real subprocess invocations.
- **operations.rs** (1,449 LOC, 95%): 30 genuine JJ operation variants with 15 real unit tests.
- **reasoning_bank.rs** (731 LOC, 85%): EMA-based pattern extraction, trajectory tracking with reward scoring. Same concept as claude-flow ReasoningBank, adapted for VCS operations.
- **types.rs** (816 LOC, 98%): Clean type definitions with napi(object) and builder patterns.
- **pkg/{web,node,bundler,deno}/*.js**: ALL wasm-bindgen auto-generated. Should never be manually edited.

**Supply chain risk**: Build system downloads jj binary from internet at compile time.

### CLI, Workers, MCP (12 files, ~10,500 LOC) — 88% REAL

| File | LOC | Real % | Key Finding |
|------|-----|--------|-------------|
| **hooks.ts** | 1,149 | 100% | Pure CLI delegation layer. 10+ hook tools as Commander.js subcommands. |
| **workers.ts** | 1,082 | 95% | 15+ subcommands. `dispatch-prompt` silently swallows ALL errors. |
| **optimized-embedder.ts** | 917 | 90% | Real O(1) LRU + FNV-1a hash. `simpleTokenize` is hash-to-ID, not real wordpiece. |
| **agentdb-cli.ts** | 862 | 95% | CONFIRMS R18: standalone CLI DOES initialize EmbeddingService with ONNX WASM. |
| **standalone-stdio.ts** | 813 | 85% | Real FastMCP server, 15 tools. **SHELL INJECTION** via unsanitized execSync. |
| **sona-tools.ts** | 676 | 90% | 15 tool definitions delegating to sonaService singletons. Real LoRA/trajectory handlers. |
| **agentdb-wrapper-enhanced.ts** | 899 | 80% | Does NOT fix R18. AttentionService falls back to stub. `calculateRecall` returns wrong metric. |
| **neural-substrate.ts** | 817 | 92% | REAL: SemanticDriftDetector, MemoryPhysics (hippocampal model), EmbeddingStateMachine. |
| **ruvector-integration.ts** | 718 | 75% | 5-priority embedding fallback chain. `generateActivations` is hash-based placeholder. |

### Systemic Pattern: Hash-Based Embedding Fallback

4+ files silently degrade semantic search to character-frequency matching when ONNX unavailable:
- `optimized-embedder.ts` — simpleTokenize uses hash-to-token-ID
- `ruvector-integration.ts` — simpleEmbedding produces hash vectors as final fallback
- `edge-full.ts` — simpleEmbed uses charCode mapping
- `agentdb-wrapper-enhanced.ts` — inherits the problem through dependencies

### Security Findings (R22)

| Issue | File | Severity |
|-------|------|----------|
| Shell injection via unsanitized execSync | standalone-stdio.ts | CRITICAL |
| SQL injection in incrementStat (string interpolation for column name) | IntelligenceStore.ts | HIGH |
| API key prefix leaked in logs | anthropic-to-requesty.ts | HIGH |
| Missing request timeout (can hang indefinitely) | anthropic-to-openrouter.ts | HIGH |

### Updated Summary Statistics

| Metric | Before R22 | After R22 |
|--------|-----------|-----------|
| DEEP files (agentic-flow-rust) | 4 | 57 |
| NOT_TOUCHED (agentic-flow-rust) | 3,685 | 3,632 |

## R40: Worker System & QUIC Transport Deep-Read (Session 40)

### Overview

3 files from agentic-flow's worker and transport layers. Key question: Are workers the real execution layer beneath the "demonstration framework" CLI (R31)? **Weighted average: 56% real.**

### File Analysis

| File | LOC | Real% | Verdict | Execution Model |
|------|-----|-------|---------|-----------------|
| **worker-registry.ts** (10941) | 662 | 80% | REAL persistence | In-memory objects + SQLite |
| **worker-agent-integration.ts** (10939) | 613 | 68% | PARTIAL — advisory only | Configuration + selection |
| **quic.ts** (10893) | 599 | 24% | COMPLETE FACADE | All stubs |

### R31 Extension: "Functional Single-Node Task Runner"

R31 characterized the CLI as a "demonstration framework." R40 refines this to a more nuanced picture:

| Aspect | Assessment |
|--------|------------|
| Worker execution | **REAL** — dispatch-service performs genuine file I/O, pattern extraction, analysis |
| Worker persistence | **REAL** — SQLite-backed with WAL mode, ULID IDs, 3-tier DB backend |
| Agent selection | **REAL logic, advisory-only** — EMA performance tracking, multi-factor scoring, but no actual bridging |
| Transport | **FACADE** — QUIC layer is pure scaffolding. Real QUIC exists only in Rust crate |

**Corrected characterization**: agentic-flow is a **functional single-node task runner** with aspirational distributed transport. Workers execute real analysis in-process (same Node.js event loop). Distributed coordination is structurally defined but non-functional.

### worker-registry.ts — Real SQLite Persistence (80%)

Production-quality worker metadata store:
- **Three-tier DB backend**: better-sqlite3 > sql.js > in-memory Map (cross-platform)
- **WAL mode** + NORMAL synchronous (correct high-performance SQLite config)
- **ULID-based** worker IDs (8-char truncated)
- **Worker lifecycle**: create, updateStatus (with started_at/completed_at timestamps), cleanup
- **json_insert()** for atomic result key appending

**Issues**:
- Race condition: sql.js async init in synchronous constructor — data loss window (lines 159-186)
- Performance: sql.js wrapper calls writeFileSync on every mutation (lines 53-58)
- Workers are database rows, NOT real processes or threads

### worker-agent-integration.ts — Advisory Selection Only (68%)

Provides recommendation logic for WHICH agent should handle a trigger, but does NOT actually bridge to agents:
- 6 hardcoded agent types (researcher, coder, tester, security-analyst, performance-analyzer, documenter)
- 12 trigger-to-agent mappings with fallback chains
- EMA-based performance tracking (alpha=0.2) — good pattern
- Multi-factor agent scoring: `quality * success_rate * (1/latency_factor)`

**Issues**:
- No process spawning, IPC, or agent lifecycle management — the "integration" is purely declarative
- Performance data stored in-memory Maps — lost on restart
- p95 latency is decaying high-water mark, not real percentile
- `getWorkerRegistry` imported but never used

### quic.ts — Complete Facade (24%)

Zero QUIC protocol implementation:
- `loadWasmModule()` returns `{}` (line 288)
- `send()` writes nothing (lines 189-193)
- `receive()` returns empty `Uint8Array` (lines 194-198)
- `sendRequest()` returns hardcoded 200 with empty body (lines 315-319)
- `getStats()` returns all zeros (lines 274-278)

**Critical context**: Real QUIC exists in Rust crate `agentic-flow-quic` using **quinn 0.11** (production QUIC library) with **rustls 0.23** (real TLS 1.3). WASM bindings exist (`wasm.rs`) but are never connected to this TypeScript file. The TypeScript → WASM → Rust/quinn bridge was never completed.

### Worker-Transport Architecture Gap

```
Transport Layer (QUIC) ────[FACADE]──── swarm/quic-coordinator.ts
                                              │
                                              ▼
                                    SwarmAgent coordination
                                    (depends on stub transport)
                                              │
                                              ▼
Agent Integration ────[ADVISORY ONLY]─── dispatch-service.ts
  (recommends agent)                    (executes in-process)
        │                                      │
        ▼                                      ▼
  AgentCapabilities                    worker-registry.ts (SQLite)
  (static config)                      (real persistence)
```

The critical gap is between the middle (agent selection) and top (transport) layers. Workers work locally, but distributed transport is absent.

### Findings

**CRITICAL** (2):
- quic.ts has zero implementation — all stubs, `loadWasmModule()` returns `{}` (quic.ts:284-320)
- `sendRequest()` returns hardcoded 200 with empty body regardless of input (quic.ts:213-235)

**HIGH** (4):
- Placeholder never connected to real quinn-based QUIC in Rust crate (quic.ts)
- `getStats()` returns zeros with "From WASM" comments — misleading (quic.ts:274-278)
- worker-agent-integration does NOT bridge agents to workers — advisory only, no IPC (worker-agent-integration.ts)
- Performance profiles in-memory only — self-learning never persists (worker-agent-integration.ts:221-264)

**MEDIUM** (6):
- sql.js async init race condition in synchronous constructor (worker-registry.ts:159-186)
- sql.js wrapper calls writeFileSync on every mutation (worker-registry.ts:53-58)
- Workers are NOT real processes — database rows only (worker-registry.ts)
- p95 latency is decaying high-water mark, not real percentile (worker-agent-integration.ts:239)
- Incomplete benchmark compliance — only 3 of 6+ metrics checked (worker-agent-integration.ts:500-511)
- All QUIC config options stored but never read (quic.ts:69-89)

### Updated Summary

| Metric | Before R40 | After R40 |
|--------|-----------|-----------|
| DEEP files (agentic-flow) | 57 | 60 |
| Worker system verdict | Unknown | Functional single-node task runner |
| QUIC transport verdict | Suspected stub (R22b) | Confirmed complete facade |
| Real QUIC location | Unknown | Rust crate agentic-flow-quic (quinn 0.11) |
