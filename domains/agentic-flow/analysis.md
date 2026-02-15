# Agentic Flow Domain Analysis

> **Priority**: HIGH | **Coverage**: ~8.5% (60/714 DEEP) | **Status**: In Progress
> **Last updated**: 2026-02-08 (Session R40)

## 1. Current State Summary

The agentic-flow domain spans 714 files / 574 MB (96% bundled deps) across a TypeScript monorepo containing 5 sub-packages. Published on npm as `agentic-flow` (2.0.6) with 84,541 monthly downloads. Quality bifurcates dramatically — from 100% (hooks.ts pure delegation) to 24% (quic.ts complete facade).

**Top-level verdicts:**

- **Agents are prompt templates, not code** — 82 markdown files with YAML frontmatter, loaded by thin SDK wrappers.
- **MCP tools shell out to `npx claude-flow@alpha`** — circular dependency, not in-process. "213 tools" counts external packages.
- **Three separate ReasoningBanks** (claude-flow, agentic-flow, agentdb) share zero code. Agentic-flow has the most sophisticated (5 DeepMind algorithms) but claude-flow never calls it.
- **Multi-provider routing is the genuine value** — real Anthropic/OpenRouter/Gemini/ONNX translation with fallback chains (91-95%).
- **QUIC and Federation are complete stubs** — WASM exists but never loads, all network calls return empty arrays.
- **AgentDB is substantive** — 23 controllers grounded in published papers (Reflexion, Voyager, Pearl's do-calculus), not thin abstractions.
- **Hash-based embeddings are systemic** — 4+ files (optimized-embedder, ruvector-integration, edge-full, agentdb-wrapper) silently degrade to character-frequency matching.
- **Gap between EXISTS and RUNS is vast** — sophisticated learning algorithms exist but claude-flow only uses LocalReasoningBank (patterns.json).
- **Worker system is functional single-node** — real SQLite persistence, real file I/O, but distributed transport is facade.
- **Best code:** cli-proxy.ts (95%), hooks.ts (100%), TypeScript sources (80-95%), agentdb controllers (82-95%).
- **Worst code:** quic.ts (24%), enhanced-consciousness.js (15-20%), neural-coordination-protocol.js (10-15%).

## 2. File Registry

### Agentic-Flow Core

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| hooks.ts | agentic-flow | 1,149 | 100% | DEEP | Pure CLI delegation. 10+ hook tools via Commander.js | R22 |
| cli-proxy.ts | agentic-flow | 1,432 | 95% | DEEP | Multi-provider proxy (OpenRouter/Gemini/ONNX/Requesty). QUIC transport | R22 |
| workers.ts | agentic-flow | 1,082 | 95% | DEEP | 15+ subcommands. dispatch-prompt swallows all errors | R22 |
| types.rs (agentic-jujutsu) | agentic-flow | 816 | 98% | DEEP | Clean type definitions with napi(object), builder patterns | R22 |
| wrapper.rs (agentic-jujutsu) | agentic-flow | 1,300 | 90% | DEEP | Embeds real jj binary. 15 genuine JJ operations | R22 |
| operations.rs (agentic-jujutsu) | agentic-flow | 1,449 | 95% | DEEP | 30 genuine JJ variants with 15 real unit tests | R22 |
| reasoning_bank.rs (agentic-jujutsu) | agentic-flow | 731 | 85% | DEEP | EMA-based pattern extraction adapted for VCS | R22 |
| anthropic-to-requesty.ts | agentic-flow | 880 | 93% | DEEP | Real proxy with streaming. API key prefix leaked in logs | R22 |
| anthropic-to-openrouter.ts | agentic-flow | 775 | 90% | DEEP | ~95% identical to Requesty. No request timeout | R22 |
| optimized-embedder.ts | agentic-flow | 917 | 90% | DEEP | Real O(1) LRU + FNV-1a. simpleTokenize is hash fallback | R22 |
| neural-substrate.ts | agentic-flow | 817 | 92% | DEEP | Real SemanticDriftDetector, MemoryPhysics (hippocampal) | R22 |
| agentdb-cli.ts | agentic-flow | 862 | 95% | DEEP | Standalone CLI initializes EmbeddingService with ONNX | R22 |
| EmbeddingCache.ts | agentic-flow | 726 | 90% | DEEP | 3-tier cache (native SQLite > WASM > Memory), SHA-256 keys | R22 |
| IntelligenceStore.ts | agentic-flow | 698 | 90% | DEEP | SQLite dual backend. SQL injection risk in incrementStat | R22 |
| sona-tools.ts | agentic-flow | 676 | 90% | DEEP | 15 tools delegating to sonaService singletons | R22 |
| EmbeddingService.ts | agentic-flow | 1,810 | 80% | DEEP | ONNX, K-means clustering. simpleEmbed = hash fallback | R22 |
| worker-registry.ts | agentic-flow | 662 | 80% | DEEP | SQLite WAL persistence. sql.js race condition | R40 |
| RuVectorIntelligence.ts | agentic-flow | 1,200 | 80% | DEEP | SONA Micro-LoRA, 6 attention types, HNSW, LRU | R22 |
| dispatch-service.ts | agentic-flow | 1,212 | 80% | DEEP | 12 worker types, secret detection, dependency scanning | R22 |
| agentdb-wrapper-enhanced.ts | agentic-flow | 899 | 80% | DEEP | AttentionService stub fallback. calculateRecall wrong | R22 |
| edge-full.ts | agentic-flow | 943 | 75% | DEEP | 6 ruvector WASM modules. JS fallback for 5/6 | R22 |
| agent-booster-enhanced.ts | agentic-flow | 1,428 | 75% | DEEP | Pattern caching, 5-tier compression. External npx dep | R22 |
| ruvector-integration.ts | agentic-flow | 718 | 75% | DEEP | 5-priority embedding fallback. Hash placeholders | R22 |
| intelligence-bridge.ts | agentic-flow | 1,371 | 70% | DEEP | Bridge to RuVectorIntelligence. 9 RL config-only | R22 |
| worker-agent-integration.ts | agentic-flow | 613 | 68% | DEEP | Advisory agent selection. No IPC or lifecycle | R40 |
| standalone-stdio.ts | agentic-flow | 813 | 85% | DEEP | FastMCP server, 15 tools. SHELL INJECTION risk | R22 |
| p2p-swarm-v2.ts | agentic-flow | 2,280 | 85% | DEEP | Production crypto. Task execution stub. Fake IPFS CIDs | R22 |
| quic.ts | agentic-flow | 599 | 24% | DEEP | COMPLETE FACADE. loadWasmModule returns {}, all stubs | R40 |

### Core Integration Bridges (R44)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| RuvLLMOrchestrator.ts | agentic-flow | 635 | 35-40% | DEEP | FACADE — "FastGRNN"=sort-by-score, "TRM"=sentence splitting, "SONA"=uniform weights. THIRD parallel routing system. Zero ruvllm connection. Orphaned (never imported by execution code) | R44 |
| ruvector-backend.ts | agentic-flow | 626 | 12% | DEEP | COMPLETE FACADE — zero ruvector imports, isRustAvailable()=always true, searchRuVector()=sleep+brute-force, "125x speedup"=hardcoded constant, NEVER imported anywhere | R44 |
| sona-service.ts | agentic-flow | 592 | 78% | DEEP | GENUINE wrapper around @ruvector/sona SonaEngine. 5 vibecast profiles. API mismatch with ruvector-integration.ts (beginTrajectory vs startTrajectory). Parallel incompatible SONA paths | R44 |

### AgentDB Controllers

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| ReflexionMemory | agentdb | 815 | 82% | SURFACE | Episodic replay, 4 retrieval strategies, GNN enhancement | Initial |
| SkillLibrary | agentdb | 697 | 82% | SURFACE | Voyager-based pattern extraction, composite scoring | Initial |
| CausalMemoryGraph | agentdb | 602 | 82% | SURFACE | Pearl's do-calculus, uplift modeling, t-stats | Initial |
| AttentionService | agentdb | 517 | 70% | SURFACE | JS fallback works. Flash/MoE require @ruvector/attention | Initial |
| RuVectorBackend | agentdb | 776 | 90% | SURFACE | Semaphore concurrency, security, adaptive HNSW params | Initial |
| HNSWIndex | agentdb | 437 | 88% | SURFACE | Real wrapper around hnswlib-node (C++) | Initial |
| QUICServer | agentdb | 383 | 15% | SURFACE | STUB — "Actual QUIC would use a library" | Initial |
| QUICClient | agentdb | 489 | 15% | SURFACE | sleep(100) + {success: true} | Initial |
| SyncCoordinator | agentdb | 553 | 40% | SURFACE | Real logic on stub QUIC transport | Initial |
| ReasoningBank | agentdb | ~400 | 82% | SURFACE | Real pattern store with optional GNN | Initial |

### ReasoningBank Implementations

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| retrieve.js | agentic-flow | 87 | 95% | SURFACE | MMR diversity-aware with 4-factor scoring | Initial |
| judge.js | agentic-flow | ~150 | 70% | SURFACE | LLM-as-Judge via ModelRouter. Heuristic fallback | Initial |
| distill.js | agentic-flow | ~100 | 60% | SURFACE | LLM-based extraction + PII scrub. Returns [] without API | Initial |
| consolidate.js | agentic-flow | ~120 | 90% | SURFACE | Dedup (cosine ≥0.95), contradiction, pruning (180d) | Initial |
| matts.js | agentic-flow | ~80 | 75% | SURFACE | Memory-aware test-time scaling. Requires LLM | Initial |
| intelligence.js | claude-flow | ~200 | 30% | SURFACE | In-memory Map + JSON. O(n) linear scan | Initial |

### Orchestration & Routing

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| RuvLLMOrchestrator.js | agentic-flow | ~400 | 60% | SURFACE | Real embedding search. "TRM" = word count heuristics | Initial |
| router.js | agentic-flow | ~600 | 92% | SURFACE | 4 providers with real translation, fallback, metrics | Initial |
| SemanticRouter | agentic-flow | 291 | 65% | SURFACE | Real cosine similarity. Admits brute-force in comments | Initial |
| CircuitBreakerRouter | agentic-flow | 459 | 90% | SURFACE | Full state machine (CLOSED/OPEN/HALF_OPEN) | Initial |
| attention-coordinator.js | agentic-flow | 361 | 50% | SURFACE | Attention consensus, MoE. Requires external service | Initial |

### MCP & Tool Layer

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| claudeFlowSdkServer.js | agentic-flow | ~300 | 25% | SURFACE | 9 tools via execSync to npx claude-flow@alpha | Initial |
| stdio-full.js | agentic-flow | ~400 | 25% | SURFACE | 11 tools, same execSync pattern | Initial |
| pii-scrubber.js | agentic-flow | ~100 | 80% | SURFACE | 12 regex patterns for credentials/PII | Initial |

### Agent Runners

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| claudeAgent.js | agentic-flow | 335 | 85% | SURFACE | Claude Agent SDK query() wrapper | Initial |
| claudeAgentDirect.js | agentic-flow | ~200 | 90% | SURFACE | Raw Anthropic SDK with streaming, no tools | Initial |
| directApiAgent.js | agentic-flow | ~250 | 80% | SURFACE | Anthropic SDK with 7 custom tools (shell out) | Initial |
| codeReviewAgent.js | agentic-flow | ~50 | 95% | SURFACE | Thin prompt wrapper | Initial |
| webResearchAgent.js | agentic-flow | ~50 | 95% | SURFACE | Thin prompt wrapper | Initial |

## 3. Findings Registry

### 3a. CRITICAL Findings

| ID | Description | File(s) | Session | Status |
|----|-------------|---------|---------|--------|
| C1 | **Three fragmented ReasoningBanks** — Zero code sharing between claude-flow, agentic-flow, agentdb | All 3 packages | Initial | Open |
| C2 | **MCP tools are circular dependency** — Shell out to `npx claude-flow@alpha`, not in-process | claudeFlowSdkServer.js, stdio-full.js | Initial | Open |
| C3 | **QUIC transport complete facade** — loadWasmModule() returns {}, all methods stubbed | quic.ts | R40 | Open |
| C4 | **sendRequest hardcoded response** — Returns 200 + empty body regardless of input | quic.ts | R40 | Open |
| C5 | **Hash-based embeddings systemic** — 4+ files silently degrade to character-frequency matching | optimized-embedder.ts, ruvector-integration.ts, edge-full.ts, agentdb-wrapper-enhanced.ts | R22 | Open |
| C6 | **Shell injection via unsanitized execSync** — User input directly in shell command | standalone-stdio.ts | R22 | Open |
| C7 | **Agent Booster directory doesn't exist** — dist/agent-booster/ missing from npm package | Package structure | Initial | Open |
| C8 | **WASM ReasoningBank paths point to non-existent files** — Import references dead code | Multiple | Initial | Open |
| C9 | **ONNX model download broken** — Falls back to hash embeddings without warning | EmbeddingService.ts | Initial | Open |

### 3b. HIGH Findings

| ID | Description | File(s) | Session | Status |
|----|-------------|---------|---------|--------|
| H1 | **Agentic-flow learning pipeline unused** — Claude-flow never calls judge/distill/consolidate | Multiple | Initial | Open |
| H2 | **RuVectorIntelligence never imported** — Sophisticated SONA integration orphaned | RuVectorIntelligence.ts | R22 | Open |
| H3 | **AgentDB controllers zero usage** — No imports found in claude-flow | All 23 controllers | Initial | Open |
| H4 | **QUIC placeholder never connected to quinn** — Real Rust crate exists but WASM bridge incomplete | quic.ts | R40 | Open |
| H5 | **Worker-agent integration advisory only** — No IPC, process spawning, or lifecycle management | worker-agent-integration.ts | R40 | Open |
| H6 | **Performance profiles in-memory only** — Self-learning never persists across restarts | worker-agent-integration.ts | R40 | Open |
| H7 | **SQL injection in incrementStat** — String interpolation for column name | IntelligenceStore.ts | R22 | Open |
| H8 | **API key prefix leaked in logs** — Sensitive data exposure | anthropic-to-requesty.ts | R22 | Open |
| H9 | **Missing request timeout** — Can hang indefinitely | anthropic-to-openrouter.ts | R22 | Open |
| H10 | **HNSW speed claims misleading** — "150x-12,500x" is theoretical vs brute-force | Documentation | Initial | Open |
| H11 | **"213 MCP tools" counts external packages** — Only 9-11 actual tools in agentic-flow | MCP layer | Initial | Open |
| H12 | **Silent dependency failure** — Without optional deps, all learning features become no-ops | Package dependencies | Initial | Open |
| H13 | **Federation Hub returns empty arrays** — sendSyncMessage() is stub | Multiple | Initial | Open |
| H14 | **HTTP/3 proxy returns empty Uint8Array** — encodeHttp3Request() unimplemented | Multiple | Initial | Open |
| H15 | **"66 specialized agents" are markdown prompts** — Not programmatic implementations | .claude/agents/ | Initial | Open |
| H16 | **"9 RL algorithms" are Q-value updates** — All share identical implementation with different names | LearningSystem | Initial | Open |
| H17 | **SemanticRouter admits brute-force** — Code comments contradict HNSW marketing | SemanticRouter | Initial | Open |
| H18 | **sql.js async init race condition** — Synchronous constructor calls async init | worker-registry.ts | R40 | Open |
| H19 | **sql.js writeFileSync on every mutation** — Performance issue for high-frequency writes | worker-registry.ts | R40 | Open |
| H20 | **Workers are database rows, not processes** — Misleading "worker" terminology | worker-registry.ts | R40 | Open |
| H21 | **p95 latency is high-water mark** — Not real percentile calculation | worker-agent-integration.ts | R40 | Open |
| H22 | **Incomplete benchmark compliance** — Only 3 of 6+ metrics checked | worker-agent-integration.ts | R40 | Open |
| H23 | **All QUIC config stored but never read** — Dead configuration code | quic.ts | R40 | Open |
| C10 | **RuvLLMOrchestrator is THIRD parallel routing system** — No connection to ADR-008 or ruvllm backend. Imports only agentdb (ReasoningBank) | RuvLLMOrchestrator.ts | R44 | Open |
| C11 | **ruvector-backend.ts COMPLETE FACADE** — Zero ruvector imports, hardcoded "125x speedup" constant, never imported anywhere. isRustAvailable()=always true | ruvector-backend.ts | R44 | Open |
| C12 | **"FastGRNN" is NOT a neural network** — Just sorts patterns by weight and picks top. Marketing terminology for simple heuristics | RuvLLMOrchestrator.ts | R44 | Open |
| C13 | **Unguarded optional dependency import** — sona-service.ts will crash if @ruvector/sona not installed | sona-service.ts | R44 | Open |
| H24 | **Parallel incompatible SONA integrations** — sona-service.ts uses beginTrajectory() but ruvector-integration.ts expects startTrajectory() | sona-service.ts, ruvector-integration.ts | R44 | Open |
| H25 | **sona-service.ts unbounded memory** — trajectories Map grows without cleanup | sona-service.ts | R44 | Open |
| H26 | **ruvector-backend.ts fabricated metrics** — All performance numbers are formula-based constants, not measurements | ruvector-backend.ts | R44 | Open |
| H27 | **RuvLLMOrchestrator.ts orphaned** — Only imported by llm/index.ts and tests, NOT by execution code | RuvLLMOrchestrator.ts | R44 | Open |

## 4. Positives Registry

| Description | File(s) | Session |
|-------------|---------|---------|
| **hooks.ts is 100% real** — Pure CLI delegation layer with clean Commander.js subcommands | hooks.ts | R22 |
| **cli-proxy.ts excellent multi-provider routing** — Real OpenRouter/Gemini/ONNX/Requesty integration (95%) | cli-proxy.ts | R22 |
| **agentic-jujutsu is genuinely functional** — Real Jujutsu VCS operations with embedded binary (90-98%) | 3 Rust files | R22 |
| **Proxy layer is production-quality** — Real HTTP proxies with streaming support (91-93%) | anthropic-to-requesty.ts, anthropic-to-openrouter.ts | R22 |
| **EmbeddingCache well-architected** — 3-tier cache with cross-platform support (90%) | EmbeddingCache.ts | R22 |
| **IntelligenceStore clean dual backend** — SQLite with debounced saves (90%) | IntelligenceStore.ts | R22 |
| **neural-substrate.ts real neuroscience models** — SemanticDriftDetector, hippocampal MemoryPhysics (92%) | neural-substrate.ts | R22 |
| **worker-registry.ts production SQLite** — WAL mode, ULID IDs, 3-tier backend (80%) | worker-registry.ts | R40 |
| **Multi-provider routing is unique value** — Best feature in agentic-flow, well-implemented | router.js + proxies | Initial |
| **AgentDB controllers are substantive** — 18/23 implement real paper-referenced algorithms | 23 files | Initial |
| **ReasoningBank retrieve is genuine** — MMR algorithm with 4-factor scoring (95%) | retrieve.js | Initial |
| **CircuitBreakerRouter complete state machine** — Proper CLOSED/OPEN/HALF_OPEN transitions (90%) | CircuitBreakerRouter | Initial |
| **PII scrubber is real** — 12 regex patterns for credentials/PII (80%) | pii-scrubber.js | Initial |
| **RuVectorBackend production-ready** — Excellent security, adaptive HNSW, semaphore (90%) | RuVectorBackend | Initial |
| **82 agent prompt templates** — Well-crafted system prompts for various roles | .claude/agents/ | Initial |
| **EMA-based performance tracking** — Good pattern in worker-agent-integration (alpha=0.2) | worker-agent-integration.ts | R40 |
| **sona-service.ts genuine SONA integration** — Real SonaEngine wrapper with 5 vibecast profiles, real trajectory/LoRA delegation, proper EventEmitter lifecycle | sona-service.ts | R44 |

## 5. Subsystem Sections

### 5a. Three ReasoningBank Fragmentation

Four completely independent ReasoningBank implementations exist (discovered 4th in ruvllm R37, memory-and-learning domain). Agentic-flow vs claude-flow vs agentdb:

| Implementation | Package | Storage | Algorithms | Status |
|---|---|---|---|---|
| agentic-flow | agentic-flow | SQLite (memory.db) | 5 (Retrieve, Judge, Distill, Consolidate, MaTTS) | **Sophisticated, unused** |
| agentdb | agentdb | SQLite + VectorBackend | Pattern store with optional GNN | Never imported by claude-flow |
| claude-flow | claude-flow | JSON (patterns.json) | In-memory Map, O(n) scan | **Only one that runs** |
| ruvllm | ruvllm | Rust (K-means + EWC++) | Best math, separate repo | Fourth implementation (R37) |

**Agentic-flow ReasoningBank** implements arXiv:2509.25140 (Google DeepMind) with real MMR retrieval (4-factor scoring: similarity, recency, reliability, diversity), LLM-as-Judge via ModelRouter (falls back to heuristic confidence=0.5 without API key), LLM-based distill with PII scrubbing (returns [] without API key), consolidate dedup (cosine ≥0.95) + contradiction detection + pruning (180 days), and MaTTS test-time scaling (Initial).

**Critical limitation**: Without OPENROUTER_API_KEY/ANTHROPIC_API_KEY/GEMINI_API_KEY, Judge and Distill are non-functional. The sophisticated learning pipeline exists but requires API access and explicit integration to unlock (Initial).

**Claude-flow uses LocalReasoningBank** — in-memory Map + Array with JSON file persistence. O(1) store, O(n) linear scan search. No SQLite, no LLM calls, no consolidation. This is what `claude-flow hooks intelligence *` invokes. The 50 patterns in MEMORY.md are from this system (Initial).

### 5b. Embedding Fallback Chain & Hash Problem

The intended embedding pipeline (Initial):

1. `@ruvector/core` (Rust NAPI) → Usually missing
2. ONNX via `@xenova/transformers` → downloadModel fails
3. **Hash-based embeddings** → THIS IS WHAT RUNS

Confirmed systemic across 4+ files in agentic-flow (R22):

| File | Mechanism |
|------|-----------|
| optimized-embedder.ts | simpleTokenize: hash-to-token-ID, not real wordpiece |
| ruvector-integration.ts | simpleEmbedding: hash vectors as final fallback |
| edge-full.ts | simpleEmbed: charCode mapping |
| agentdb-wrapper-enhanced.ts | Inherits degradation through dependencies |

In practice, all "semantic search" using defaults is character-frequency matching. HNSW indices are structurally valid but search results are meaningless without plugging in a real embedding provider (R22).

EmbeddingService.ts (1,810 LOC, 80%) has real ONNX embedding, K-means clustering, pretrain system, but simpleEmbed = hash fallback (R22).

### 5c. Multi-Provider Routing — The Genuine Value

**router.js** (~600 LOC, 92%) implements 4+ providers with real API translation (Initial):

| Provider | File | Real? | Tool Support | Key Features |
|----------|------|-------|-------------|--------------|
| Anthropic | providers/anthropic.js | YES | YES | Native integration |
| OpenRouter | providers/openrouter.js | YES | YES | Anthropic→OpenAI format conversion |
| Gemini | providers/gemini.js | YES | NO | Via @google/genai |
| ONNX Local | Referenced | Partial | N/A | Local inference |
| Requesty | proxy/requesty-proxy.js | YES | Via proxy | HTTP proxy layer |

**Routing modes**: manual, rule-based, cost-optimized, performance-optimized. Fallback chain with automatic retry. Metrics tracking per provider (Initial).

**Proxy layer** (R22): anthropic-to-requesty.ts (880 LOC, 93%) and anthropic-to-openrouter.ts (775 LOC, 90%) are genuine HTTP proxies with streaming support. ~95% identical implementations. API key prefix leaked in logs (HIGH). Missing request timeout (HIGH).

**Assessment**: This is the most unique capability agentic-flow provides. The Anthropic-to-OpenAI format translation for OpenRouter is properly implemented with streaming support (Initial).

### 5d. Agent System — Prompt Templates, Not Code

**82 agent markdown files** in `.claude/agents/` organized across 20 categories (Initial). Each file has YAML frontmatter with name, description, capabilities, hooks, followed by system prompt body.

**7 JavaScript agent runners** load these templates (Initial):

| Runner | Lines | Implementation |
|--------|-------|----------------|
| claudeAgent.js | 335 | Claude Agent SDK query() wrapper (85%) |
| claudeAgentDirect.js | ~200 | Raw Anthropic SDK with streaming (90%) |
| directApiAgent.js | ~250 | Anthropic SDK + 7 tools that shell out (80%) |
| codeReviewAgent.js | ~50 | System prompt: "review diffs" (95%) |
| webResearchAgent.js | ~50 | System prompt: "web reconnaissance" (95%) |
| dataAgent.js | ~50 | System prompt: "analyze tabular data" (95%) |
| claudeFlowAgent.js | ~100 | Uses claude-flow MCP tools |

**RuvLLMOrchestrator** (Initial): selectAgent() embeds task, searches ReasoningBank, applies SONA weighting, routes via "FastGRNN". Reality: "TRM" = word count + keyword heuristics, "SONA adaptation" = uniform weights[i] += 0.01, "FastGRNN" = sort-by-score-and-pick-top, task decomposition splits by periods and "and/then/after" keywords (60%).

**Assessment**: Agents are entirely prompt-driven. "Self-learning" and "GNN-enhanced" references in system prompts are aspirational documentation, not executed code. 77-82 markdown files are useful as well-crafted system prompts but not programmatic agent implementations (Initial).

### 5e. Worker System — Functional Single-Node Task Runner

**worker-registry.ts** (662 LOC, 80%, R40): Production SQLite WAL persistence with ULID IDs, 3-tier DB backend (better-sqlite3 > sql.js > in-memory Map). json_insert() for atomic result appending. Issues: sql.js async init race in synchronous constructor (data loss window), writeFileSync on every mutation (performance), workers are database rows NOT real processes.

**worker-agent-integration.ts** (613 LOC, 68%, R40): Advisory agent selection only — 6 hardcoded agent types, 12 trigger-to-agent mappings, EMA performance tracking (alpha=0.2), multi-factor scoring `quality * success_rate * (1/latency_factor)`. Issues: no process spawning/IPC/lifecycle management, performance data in-memory only (lost on restart), p95 latency is decaying high-water mark not real percentile.

**dispatch-service.ts** (1,212 LOC, 80%, R22): 12 worker types with real file analysis — secret detection, dependency scanning. Workers execute real analysis in-process (same Node.js event loop).

**Architecture** (R40): Workers are functional single-node task runner with real SQLite persistence and genuine file I/O. Distributed coordination is structurally defined but non-functional (QUIC stub). Agent integration is advisory-only, not executable.

### 5f. QUIC Transport — Complete Facade

**quic.ts** (599 LOC, 24%, R40): Zero QUIC protocol implementation. loadWasmModule() returns {} (L288), send() writes nothing (L189-193), receive() returns empty Uint8Array (L194-198), sendRequest() returns hardcoded 200 + empty body (L315-319), getStats() returns all zeros (L274-278).

**Critical context**: Real QUIC exists in Rust crate `agentic-flow-quic` using quinn 0.11 (production QUIC library) with rustls 0.23 (real TLS 1.3). WASM bindings exist (wasm.rs) but are never connected to TypeScript. The TypeScript → WASM → Rust/quinn bridge was never completed (R40).

**Impact**: All swarm coordination (quic-coordinator.ts, p2p-swarm-v2.js) is architecturally sophisticated but non-functional without transport. SyncCoordinator (553 LOC, 40%) has real logic built on stub QUIC (Initial).

### 5g. AgentDB Controllers — Substantive but Unused

**Database schema** (Initial): 12 tables, 5 views, 4 triggers implementing 5 memory patterns — Reflexion episodic replay (episodes, episode_embeddings), Skill Library (skills, skill_links, skill_embeddings), Structured mixed memory (facts SPO triples, notes, note_embeddings), Episodic segmentation (events, consolidated_memories), Graph-aware recall (exp_nodes, exp_edges, exp_node_embeddings).

**Top controllers** (Initial):

| Controller | LOC | Based On | Real% | Key Features |
|------------|-----|----------|-------|--------------|
| ReflexionMemory | 815 | arXiv 2303.11366 | 82% | Episodic replay, 4 retrieval strategies, GNN enhancement |
| SkillLibrary | 697 | arXiv 2305.16291 (Voyager) | 82% | Pattern extraction, learning trends, composite scoring |
| CausalMemoryGraph | 602 | Pearl's do-calculus | 82% | Uplift modeling, A/B experiments, t-stats, recursive CTEs |
| RuVectorBackend | 776 | — | 90% | Semaphore concurrency, BufferPool, security, adaptive HNSW |
| AttentionService | 517 | Transformer | 70% | JS fallback works; Flash/MoE require @ruvector/attention |
| HNSWIndex | 437 | HNSW | 88% | Wrapper around hnswlib-node (C++) |

**Critical finding**: Zero imports found in claude-flow. AgentDB controllers are genuinely sophisticated but completely orphaned (H3, Initial).

**Browser build** (Initial): Real but minimal — agentdb.browser.js (48 KB) and .min.js (23 KB) with sql.js WASM SQLite fallback.

### 5h. Intelligence Layer Architecture

**RuVectorIntelligence.ts** (1,200 LOC, 80%, R22): Core integration of @ruvector/sona (MicroLoRA, BaseLoRA, EWC++), @ruvector/attention (MultiHead, Flash, Hyperbolic, MoE, GraphRoPE), ruvector core HNSW retrieval, LRU eviction (10K trajectories). Background learning via setInterval(sona.tick, 60000). Quality-gated adaptations: microLora always, baseLora ≥0.7, EWC++ ≥0.8.

**intelligence-bridge.ts** (1,371 LOC, 70%, R22): Bridge to RuVectorIntelligence. 9 RL algorithms config-only (DQN/PPO/SARSA/etc. all reduce to Q-value updates). Math.random()*0.1 fabricated activations.

**Critical gap**: Claude-flow never imports RuVectorIntelligence. The sophisticated SONA/attention integration is orphaned (H2, R22).

### 5i. MCP Tool Layer — Circular Dependency

**claudeFlowSdkServer.js** (~300 LOC, 25%, Initial): 9 tools ALL shell out to `npx claude-flow@alpha` via execSync — memory_store, memory_retrieve, memory_search, swarm_init, agent_spawn, task_orchestrate, swarm_status, agent_booster_edit_file, agent_booster_batch_edit (stubs).

**stdio-full.js** (~400 LOC, 25%, Initial): 11 tools, same execSync pattern. FastMCP stdio server.

**standalone-stdio.ts** (813 LOC, 85%, R22): Real FastMCP server with 15 tools. **SHELL INJECTION** via unsanitized execSync (C6).

**Assessment**: The "213 tools" count combines external packages (claude-flow@alpha 170+, flow-nexus@latest). Agentic-flow itself defines 9-11 tools that are CLI command wrappers, creating circular dependency (C2, Initial).

### 5k. Security Findings

| Severity | Issue | File | Session |
|----------|-------|------|---------|
| CRITICAL | Shell injection via unsanitized execSync | standalone-stdio.ts | R22 |
| HIGH | SQL injection in incrementStat (string interpolation for column name) | IntelligenceStore.ts | R22 |
| HIGH | API key prefix leaked in logs | anthropic-to-requesty.ts | R22 |
| HIGH | Missing request timeout (can hang indefinitely) | anthropic-to-openrouter.ts | R22 |

### 5l. Package Metadata

| Metric | Value |
|--------|-------|
| Total size | 574 MB installed (2.2 MB unpacked) |
| node_modules (bundled) | 553 MB (96.3%) |
| — onnxruntime-node | 513 MB (89.4%) |
| — better-sqlite3 | 27 MB (4.7%) |
| — fastmcp | 11 MB (1.9%) |
| dist/ (application code) | 13 MB (2.3%) |
| Agent markdown files | ~2 MB (0.3%) |
| Monthly downloads | 84,541 (agentic-flow), 111,686 (agentdb) |

### 5m. What Claude-Flow Actually Uses

| Component | Used? | How? |
|-----------|-------|------|
| agentic-flow ReasoningBank (retrieve) | Partially | hooks.js imports for retrieveMemories() in token-optimize hook |
| agentic-flow ReasoningBank (judge/distill/consolidate) | **NO** | Never called |
| agentdb (any controller) | **NO** | Zero imports found |
| RuVectorIntelligence | **NO** | Never imported |
| LocalReasoningBank (intelligence.js) | **YES** | What claude-flow hooks intelligence * uses |
| EmbeddingService | Indirectly | memory-initializer fallback to agentic-flow embeddings |
| Multi-provider routing | **NO** | claude-flow routes haiku/sonnet/opus only |
| Claude Agent SDK integration | **NO** | claude-flow uses Task tool |
| PII scrubber | Conditional | Only if distill runs (requires API key) |
| @ruvector native binaries | Via transitive deps | memory-initializer, ruvector-training.js |

**Bottom line**: Claude-flow uses agentic-flow for (1) embedding model access (fallback chain), (2) ReasoningBank retrieveMemories() (partial, read-only), (3) @ruvector native binaries as transitive dependencies. Does NOT use sophisticated learning pipeline, multi-provider routing, Claude Agent SDK integration, or any AgentDB controllers (Initial).

## 6. Cross-Domain Dependencies

- **memory-and-learning domain**: ReasoningBank (4 implementations), embeddings (hash-based fallback), AgentDB controllers, LearningSystem RL
- **claude-flow-cli domain**: LocalReasoningBank (only one that runs), hooks.js imports
- **ruvector domain**: @ruvector/core, @ruvector/sona, @ruvector/attention (transitive deps)
- **agentdb-integration domain**: AgentDB controllers overlap

## 7. Knowledge Gaps

- ~650+ files still NOT_TOUCHED (mostly dist/ JavaScript, node_modules, tests)
- Main entry dist/index.js orchestration logic
- Agent markdown prompt templates (82 files) content analysis
- MCP tool implementations beyond stdio servers
- P2P swarm crypto implementation details
- Federation Hub architecture
- HTTP/3 proxy layer
- Billing system implementation
- Browser build internals
- Rust agentic-flow-quic crate internals
- Full agentic-jujutsu WASM bindings
- Remaining TypeScript sources in src/

## 8. Session Log

### Initial (2026-02-08): Repository analysis
Package structure, agent system, MCP tools, ReasoningBank fragmentation, multi-provider routing, swarm coordination, AgentDB deep-dive, comparison with claude-flow. 714 total files discovered.

### R22 (2026-02-08): TypeScript source deep-read
54 files, ~59K LOC, 150 findings. Intelligence layer architecture, proxy layer, agentic-jujutsu crate, CLI/workers/MCP, hash-based embedding fallback confirmed systemic, security findings (shell injection, SQL injection).

### R40 (2026-02-08): Worker system & QUIC transport
3 files, 1,874 LOC, 12 findings. Characterized as functional single-node task runner. QUIC confirmed complete facade (24%). Worker-registry real SQLite persistence, worker-agent-integration advisory-only.

### R44 (2026-02-15): Core integration bridges (LLM, ruvector, SONA)
3 files, 1,853 LOC, ~68 findings. Integration bridges are mostly facades. RuvLLMOrchestrator.ts (35-40%) is a THIRD parallel routing system — "FastGRNN/TRM/SONA" marketing names hide simple heuristics, zero ruvllm connection, orphaned. ruvector-backend.ts (12%) is COMPLETE FACADE — zero ruvector imports, hardcoded "125x speedup", never imported anywhere. sona-service.ts (78%) is the ONLY genuine bridge — real @ruvector/sona wrapper, but has parallel incompatible API with ruvector-integration.ts (beginTrajectory vs startTrajectory). Confirms R40's "single-node task runner" characterization — bridges don't add real cross-system connectivity.
