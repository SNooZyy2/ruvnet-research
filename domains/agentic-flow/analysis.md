# Agentic Flow Domain Analysis

> **Priority**: HIGH | **Coverage**: ~7.7% (330/4264 DEEP) | **Status**: In Progress
> **Last updated**: 2026-03-03 (Session R142)

## 1. Current State Summary

The agentic-flow domain spans 4,264 files across a TypeScript monorepo with 5 sub-packages, 574 MB installed (96% bundled deps, dominated by onnxruntime-node at 513 MB). Published on npm as `agentic-flow` (2.0.6) with 84,541 monthly downloads. Quality bifurcates dramatically — from 100% (hooks.ts pure delegation) to 12% (ruvector-backend.ts complete facade). Coverage: 330 DEEP / 4,264 total (~7.7%).

**Top-level verdicts:**

- **Agents are prompt templates, not code** — 82 markdown files with YAML frontmatter, loaded by thin SDK wrappers.
- **MCP tools shell out to `npx claude-flow@alpha`** — circular dependency, not in-process. "213 tools" counts external packages.
- **Four separate ReasoningBanks** (claude-flow, agentic-flow, agentdb, ruvllm Rust) share zero code. Agentic-flow has the most sophisticated (5 DeepMind algorithms) but claude-flow never calls it.
- **Multi-provider routing is the genuine value** — real Anthropic/OpenRouter/Gemini/ONNX translation with fallback chains (91-95%).
- **QUIC TypeScript is a complete stub; QUIC Rust compiles clean** — agentic-flow-quic Rust crate (999 LOC) passes all 8 tests, but the TypeScript → WASM → Rust bridge was never completed. loadWasmModule() returns {}.
- **AgentDB MCP server IS substantive and initialized** — R136 confirms EmbeddingService is initialized with Xenova/all-MiniLM-L6-v2, all 8 controllers instantiated with proper DI, 28 real tools (not 32 as claimed). This CORRECTS prior R20 root-cause claim that EmbeddingService is never initialized in the MCP path.
- **Hash-based embeddings are systemic** — 4+ files (optimized-embedder, ruvector-integration, edge-full, agentdb-wrapper) silently degrade to character-frequency matching.
- **sona-agentdb-integration.ts is DEAD CODE** — SONAAgentDBTrainer never imported by any production file. Both critical imports (@ruvector/sona, agentdb) are NOT installed in workspace. Source of the "150x-12,500x" hardcoded marketing claim (line 45).
- **Gap between EXISTS and RUNS is vast** — sophisticated learning algorithms exist but claude-flow only uses LocalReasoningBank (patterns.json).
- **Worker system is functional single-node** — real SQLite persistence, real file I/O, but distributed transport is facade.
- **agentic-jujutsu Rust compiles, has security failure** — 83/88 tests pass, but 2 CRITICAL crypto failures: ML-DSA verify() accepts invalid signatures and wrong public keys. Cryptographic rejection property broken.
- **agent-booster Rust core logic broken** — 6/25 tests fail, strategy selection and similarity-matching assertions fail. Token optimization (Tier 1) logic incorrect.
- **reasoningbank workspace mixed** — core/learning/storage/network all PASS (46/46 tests total). reasoningbank-mcp FAILS compile (StorageConfig type mismatch). reasoningbank-wasm FAILS native compile (cfg-gated WASM module), PASSES wasm32 cross-compile.
- **Deployment is demo-only** — docker-compose.yml is a 9-line single-service demo. docker-compose.agent.yml defines 7 hardcoded demo tasks, not production infrastructure.
- **Best code:** cli-proxy.ts (95%), hooks.ts (100%), TypeScript sources (80-95%), agentdb controllers (82-95%), agentic-flow-quic Rust (clean).
- **Worst code:** ruvector-backend.ts (12%), quic.ts TypeScript (24%), sona-agentdb-integration.ts (dead code).

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
| EmbeddingService.ts | agentic-flow | 1,810 | 80% | DEEP | ONNX, K-means clustering. simpleEmbed = hash fallback. ONNX path IS reachable when ruvectorModule loads (corrects prior "always hash" claim). semanticSearch() is O(n) brute-force, no HNSW | R22, R119 |
| worker-registry.ts | agentic-flow | 662 | 80% | DEEP | SQLite WAL persistence. sql.js race condition | R40 |
| RuVectorIntelligence.ts | agentic-flow | 1,200 | 80% | DEEP | **GENUINE 2-plane pipeline**: routeTask() chains HNSW→SONA→attention (0.3/0.7 weighted). 6 attention types. "Graph" = GraphRoPeAttention (scoring), NOT graph traversal. MoE expertWeights bug. First confirmed cross-subsystem composition | R22, R119 |
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

### Integration Hubs & Dead Code (R136)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| sona-agentdb-integration.ts | agentic-flow | ~600 | 62-68% | DEEP | **DEAD CODE** — SONAAgentDBTrainer never imported in production. Both @ruvector/sona + agentdb NOT installed. Source of "150x-12,500x" hardcoded claim (line 45). query() genuinely combines HNSW + SONA paths. 4 well-differentiated config presets. export() stub (saves JSON not LoRA weights). | R136 |

### Deployment (R139)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| docker-compose.yml | agentic-flow | 9 | DEMO | DEEP | 9-line demo config. Single agents service, replicas=1, hardcoded TOPIC. No volumes/networking/healthcheck. | R139 |
| docker-compose.agent.yml | agentic-flow | ~80 | DEMO | DEEP | 7 agent types as Docker profiles with hardcoded example tasks. Not operational infrastructure. | R139 |

### Rust Crates — Compilation Audit (R141)

| Crate | LOC | Check | Tests | Verdict | Session |
|-------|-----|-------|-------|---------|---------|
| crates/agentic-flow-quic | 999 | PASS | 8p/0f | GENUINE — 8/8 clean. QUIC Rust compiles; TS bridge never completed. | R141 |
| packages/agent-booster | 2,292 | PASS | 19p/6f | BROKEN logic — strategy selection, similarity matching, template detection all fail | R141 |
| packages/agent-booster-native | 187 | PASS | 0p/0f | PASS — NAPI wrapper, no unit tests | R141 |
| packages/agent-booster-wasm | 470 | PASS | CFAIL | PASS check, WASM cross-compile PASS, test compile fails (missing .unwrap()) | R141 |
| packages/agentic-jujutsu | 9,138 | PASS | 83p/5f | SECURITY FAILURE — ML-DSA verify() accepts invalid signatures + wrong public keys | R141 |
| reasoningbank/reasoningbank-core | 773 | PASS | 12p/0f | GENUINE — fully clean | R141 |
| reasoningbank/reasoningbank-learning | 788 | PASS | 7p/0f | GENUINE — 8 deprecation warnings (AsyncLearner) | R141 |
| reasoningbank/reasoningbank-mcp | 1,037 | FAIL | NOT RUN | BROKEN — StorageConfig type mismatch (E0308), 6 errors. Unusable. | R141 |
| reasoningbank/reasoningbank-network | 2,647 | PASS | 18p/0f | GENUINE — QUIC, NeuralBus gossip all green | R141 |
| reasoningbank/reasoningbank-storage | 1,403 | PASS | 9p/0f | GENUINE — SQLite/async/migrations pass | R141 |
| reasoningbank/reasoningbank-wasm | 201 | CFAIL | — | FAIL native (cfg-gated WASM), PASS wasm32 cross-compile | R141 |

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
| MultiHeadAttentionController | agentdb | 494 | 55-65% | DEEP | Genuine 8-head structure (INVERTS Rust single-head R108). CRITICAL: random projections (not learned) — results non-reproducible. VectorBackend populated but NEVER queried for search (dead on read path). Sequential despite async signature (claims "parallel"). Average aggregation divides signal by 8. Max aggregation zero-biased. Disconnected from main AgentDB pipeline | R112 |

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
| C14 | **sona-agentdb-integration.ts is DEAD CODE with missing deps** — SONAAgentDBTrainer never imported by any production file. @ruvector/sona and agentdb NOT installed in workspace. Will crash with ModuleNotFoundError at any import. Source of "150x-12,500x" hardcoded marketing string (line 45) | sona-agentdb-integration.ts | R136 | Open |
| C15 | **agentic-jujutsu: ML-DSA cryptographic security failure** — verify() accepts invalid signatures AND wrong public keys (crypto.rs:341, 354). Cryptographic rejection property broken. 2 failing tests prove violation | packages/agentic-jujutsu | R141 | Open |
| C16 | **reasoningbank-mcp compile failure** — 6 errors from StorageConfig type mismatch (E0308). StorageConfig missing serde::Deserialize. Server.rs:65 uses wrong type. Crate is unusable. | reasoningbank/crates/reasoningbank-mcp | R141 | Open |
| H24 | **Parallel incompatible SONA integrations** — sona-service.ts uses beginTrajectory() but ruvector-integration.ts expects startTrajectory() | sona-service.ts, ruvector-integration.ts | R44 | Open |
| H25 | **sona-service.ts unbounded memory** — trajectories Map grows without cleanup | sona-service.ts | R44 | Open |
| H26 | **ruvector-backend.ts fabricated metrics** — All performance numbers are formula-based constants, not measurements | ruvector-backend.ts | R44 | Open |
| H27 | **RuvLLMOrchestrator.ts orphaned** — Only imported by llm/index.ts and tests, NOT by execution code | RuvLLMOrchestrator.ts | R44 | Open |
| H28 | **agentdb-mcp-server.ts: causal graph data model broken** — causal_add_edge hardcodes fromMemoryId=0 / toMemoryId=0. causal_query hardcodes interventionMemoryId=0. All causal edges share same source/target IDs regardless of input. Causal graph is non-functional | packages/agentdb/src/mcp/agentdb-mcp-server.ts | R136 | Open |
| H29 | **agentdb factory BackendType union incomplete** — BackendType is "auto|ruvector|rvf|hnswlib" but BackendDetection.available can be "sqljsrvf". If only sql.js backend is available, createBackend() will throw on unknown type | packages/agentdb/src/backends/factory.ts | R136 | Open |
| H30 | **agent-booster Rust core logic broken** — 6/25 tests fail. Strategy selection (FuzzyReplace vs InsertAfter), normalized-match similarity assertions, async try-catch template detection, and 3 integration tests all fail. Tier 1 token-optimization logic is incorrect | packages/agent-booster/crates/agent-booster | R141 | Open |
| H31 | **agentdb_init uses wrong db handle** — agentdb_init handler applies initializeSchema(db) to the globally-initialized db regardless of the db_path parameter provided. Different db_path requests silently operate on the wrong database | packages/agentdb/src/mcp/agentdb-mcp-server.ts | R136 | Open |
| H32 | **sona-agentdb-integration.ts export() is a stub** — Saves JSON config+stats to file, NOT actual LoRA weights. Comment says "future: use HuggingFaceExporter" (line 328). Trained model state cannot be persisted or reloaded. | sona-agentdb-integration.ts | R136 | Open |
| H33 | **agentdb_search session_id filter silently ignored** — Filter parameter accepted in schema, TODO comment in handler: "Session ID filter would require custom query" (line 1014). Using session_id filter returns unfiltered results | packages/agentdb/src/mcp/agentdb-mcp-server.ts | R136 | Open |
| H34 | **agentdb-mcp-server tool count mismatch** — Server claims 32 tools (5+9+10+5+3) but tools array contains only 28 entries. Documentation overcounts | packages/agentdb/src/mcp/agentdb-mcp-server.ts | R136 | Open |
| H35 | **sona-agentdb-integration.ts dimension hardcoded** — validateEmbedding enforces exactly 3072 dimensions regardless of config.vectorDimensions. Configuring any other dimension is silently rejected | sona-agentdb-integration.ts | R136 | Open |
| H36 | **sona-agentdb-integration.ts batchTrain is serial** — Processes patterns with await-in-loop (O(n) serial latency). No parallelism, no chunking, no batch insert to AgentDB | sona-agentdb-integration.ts | R136 | Open |

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
| **agentdb-mcp-server.ts: EmbeddingService properly initialized** — Xenova/all-MiniLM-L6-v2 (384-dim, transformers provider) awaited at startup. Corrects prior R20 claim. All 8 controllers instantiated with proper dependency injection. 28 real MCP tools with full input validation | packages/agentdb/src/mcp/agentdb-mcp-server.ts | R136 |
| **agentdb factory 5-tier resilient fallback** — Auto-mode: RuVector → RVF SDK → HNSWLib → sql.js RVF → none. sql.js built-in zero-dependency last resort. Genuine fallback pattern (confirmed R136, corrects prior 4-tier count) | packages/agentdb/src/backends/factory.ts | R136 |
| **agentdb-mcp-server production lifecycle** — keepAlive setInterval, auto-save every 5 minutes, graceful shutdown on SIGINT/SIGTERM, uncaughtException handler, WAL mode + 64MB cache, shebang CLI entrypoint | packages/agentdb/src/mcp/agentdb-mcp-server.ts | R136 |
| **agentic-flow-quic Rust crate is clean** — 999 LOC, cargo check PASS, all 8 tests green (error, client, server, config, message types). Prior SIGABRT (R44) was environment, not code. Real quinn QUIC Rust impl compiles correctly. | crates/agentic-flow-quic | R141 |
| **sona-agentdb-integration.ts query() is genuinely dual-path** — Combines HNSW nearest-neighbor (via AgentDB) + SONA pattern matching with real weighted merge. Correct trajectory lifecycle sequencing (beginTrajectory → addContext → addStep → endTrajectory) | sona-agentdb-integration.ts | R136 |
| **reasoningbank Rust workspace largely genuine** — core (12/12 tests), learning (7/7 tests), storage (9/9 tests), network (18/18 tests) all pass. Total 46/46 across 4 crates. NeuralBus gossip, QUIC transport, SQLite migrations all green | reasoningbank/ | R141 |
| **containerized agent execution proven** — docker-compose.agent.yml defines 7 agent types (goal-planner, coder, reviewer, tester, researcher, flow-nexus-swarm, parallel) with ANTHROPIC_API_KEY and claude-agent-sdk. Proves containerized agents are architecturally real even if demo-only | agentic-flow/deployment/ | R139 |

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

**Critical context**: Real QUIC exists in Rust crate `agentic-flow-quic` using quinn 0.11 (production QUIC library) with rustls 0.23 (real TLS 1.3). WASM bindings exist (wasm.rs) but are never connected to TypeScript. The TypeScript → WASM → Rust/quinn bridge was never completed (R40). R141 CONFIRMS the Rust crate compiles cleanly: all 8 tests pass. The fault is entirely in the TypeScript stub layer, not the Rust implementation.

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

### 5j. sona-agentdb-integration.ts — Dead Code with Genuine Core (R136)

**sona-agentdb-integration.ts** (~600 LOC, 62-68%) is the most misleading file in the agentic-flow repo:

**Dead / broken:**
- SONAAgentDBTrainer is NEVER imported by any production source. Only consumer is `tests/sona/sona-training.test.ts`.
- Both critical dependencies (@ruvector/sona, agentdb) are NOT installed in the workspace — will throw ModuleNotFoundError at any import.
- `export()` method saves JSON config+stats only, NOT LoRA weights. Comment acknowledges "future: use HuggingFaceExporter".
- `getStats().combined` returns hardcoded strings: avgQueryLatency="~1.25ms (HNSW + SONA)", storageEfficiency="~3KB per pattern". These are not measurements.
- validateEmbedding hardcodes 3072 dimensions regardless of config.vectorDimensions.
- The "150x-12,500x" performance claim on line 45 is a hardcoded marketing string. No benchmark exists anywhere in the codebase.

**Genuine internals:**
- `query()` method genuinely combines two retrieval paths: HNSW nearest-neighbor (via AgentDB) and SONA pattern matching, with weighted merge.
- SONA trajectory lifecycle is correctly sequenced: beginTrajectory → addTrajectoryContext → addTrajectoryStep → endTrajectory.
- 4 well-differentiated config presets (realtime/balanced/quality/largescale) with sensible HNSW parameter gradients.
- `close()` properly cleans up: removes listeners, closes DB, nullifies sonaEngine reference.

**Assessment**: The architecture is sound and the query path is genuine. But the file cannot run — missing deps make it permanently broken as-is. The "150x-12,500x" claim it contains is the same fabricated figure that appears in sona-tools.ts MCP tool handlers (R138), both contributing to the systemic marketing inflation pattern.

### 5k. Security Findings

| Severity | Issue | File | Session |
|----------|-------|------|---------|
| CRITICAL | Shell injection via unsanitized execSync | standalone-stdio.ts | R22 |
| CRITICAL | ML-DSA verify() accepts invalid signatures and wrong public keys — cryptographic rejection broken | packages/agentic-jujutsu/crypto.rs:341,354 | R141 |
| HIGH | SQL injection in incrementStat (string interpolation for column name) | IntelligenceStore.ts | R22 |
| HIGH | API key prefix leaked in logs | anthropic-to-requesty.ts | R22 |
| HIGH | Missing request timeout (can hang indefinitely) | anthropic-to-openrouter.ts | R22 |
| HIGH | agentdb_init applies schema to global db regardless of db_path parameter | agentdb-mcp-server.ts | R136 |
| MEDIUM | ANTHROPIC_API_KEY passed via env var correctly but TOPIC hardcoded in demo compose | docker-compose.yml | R139 |

### 5n. Rust Compilation Audit Results (R141)

Systematic cargo check + cargo test --lib across the agentic-flow workspace crates:

**PASS (clean or warnings only):**
- `crates/agentic-flow-quic` — 999 LOC, 8/8 tests green. Prior SIGABRT was environment issue, not code bug. QUIC Rust implementation is genuinely functional.
- `packages/agent-booster-native` — 187 LOC, 0 tests. NAPI wrapper compiles clean.
- `reasoningbank/reasoningbank-core` — 773 LOC, 12/12 tests green.
- `reasoningbank/reasoningbank-learning` — 788 LOC, 7/7 tests green. 8 deprecation warnings for AsyncLearner.
- `reasoningbank/reasoningbank-storage` — 1,403 LOC, 9/9 tests green.
- `reasoningbank/reasoningbank-network` — 2,647 LOC, 18/18 tests green. NeuralBus gossip, priority queues, QUIC streams all pass.

**FAIL (test failures):**
- `packages/agent-booster` — 2,292 LOC, 19p/6f. Strategy selection (FuzzyReplace vs InsertAfter), normalized similarity, async template detection, 3 integration tests fail. Tier 1 token-optimization is broken (H30).
- `packages/agentic-jujutsu` — 9,138 LOC, 83p/5f. 2 CRITICAL crypto failures: verify() accepts invalid signatures + wrong public keys at crypto.rs:341,354 (C15).

**FAIL (compile failures):**
- `packages/agent-booster-wasm` — 470 LOC. check PASS, WASM cross-compile PASS, test compile FAIL (missing .unwrap() on Result in lib.rs:435).
- `reasoningbank/reasoningbank-mcp` — 1,037 LOC. 6 compile errors — StorageConfig type mismatch (E0308), missing serde::Deserialize (C16). Crate unusable.
- `reasoningbank/reasoningbank-wasm` — 201 LOC. FAIL native check (wasm module cfg-gated), PASS wasm32 cross-compile. Expected for WASM-only crate.

**Summary**: 6/11 agentic-flow Rust crates fully clean. 2/11 have test failures (1 security-critical). 3/11 have compile issues (1 BlockingBug, 2 expected WASM patterns).

### 5o. AgentDB MCP Server — Corrected Assessment (R136)

**agentdb-mcp-server.ts** (2,368 LOC, ~82%) is significantly more genuine than initial assessment suggested:

- **EmbeddingService IS initialized** — Xenova/all-MiniLM-L6-v2 (384-dim) awaited at module startup. This CORRECTS the R20 root cause claim. R20's "never initialized" finding applies to the claude-flow bridge (which uses a different MCP server path), NOT to this standalone server.
- **All 8 controllers with proper DI** — CausalMemoryGraph(db), ReflexionMemory(db, embeddingService), SkillLibrary(db, embeddingService), etc.
- **28 real tools** (not 32 as documented) — batch operations have full input validation, parameterized queries, security error handling.
- **Production lifecycle** — WAL mode + 64MB cache, auto-save every 5 min, graceful SIGINT/SIGTERM, keepAlive.
- **Broken subsystems**: causal graph data model (hardcoded IDs=0, C14-equivalent), agentdb_init db_path ignored (H31), session_id filter silently ignored (H33), tool count overclaim (H34).
- **Architecture gotcha**: Top-level await means importing as library triggers DB creation + ONNX model download at require() time. Designed as CLI executable, not importable library.

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

- ~3,865 files still NOT_TOUCHED (mostly dist/ JavaScript, node_modules, tests across all packages)
- Main entry dist/index.js orchestration logic
- Agent markdown prompt templates (82 files) content analysis
- MCP tool implementations beyond stdio servers
- P2P swarm crypto implementation details
- Federation Hub architecture
- HTTP/3 proxy layer
- Billing system implementation
- Browser build internals
- Full agentic-jujutsu WASM bindings and specific failing test details (crypto.rs:341,354)
- Remaining TypeScript sources in src/
- agentic-flow-quic WASM bridge wasm.rs — what exists, why never connected
- agent-booster failing test root causes (strategy selection logic, similarity normalization)
- agentdb factory full runtime behavior with each backend tier
- CI configuration for agentic-flow repo (separate from claude-flow CI)

## 8. Session Log

### Initial (2026-02-08): Repository analysis
Package structure, agent system, MCP tools, ReasoningBank fragmentation, multi-provider routing, swarm coordination, AgentDB deep-dive, comparison with claude-flow. 714 total files discovered.

### R22 (2026-02-08): TypeScript source deep-read
54 files, ~59K LOC, 150 findings. Intelligence layer architecture, proxy layer, agentic-jujutsu crate, CLI/workers/MCP, hash-based embedding fallback confirmed systemic, security findings (shell injection, SQL injection).

### R40 (2026-02-08): Worker system & QUIC transport
3 files, 1,874 LOC, 12 findings. Characterized as functional single-node task runner. QUIC confirmed complete facade (24%). Worker-registry real SQLite persistence, worker-agent-integration advisory-only.

### R44 (2026-02-15): Core integration bridges (LLM, ruvector, SONA)
3 files, 1,853 LOC, ~68 findings. Integration bridges are mostly facades. RuvLLMOrchestrator.ts (35-40%) is a THIRD parallel routing system — "FastGRNN/TRM/SONA" marketing names hide simple heuristics, zero ruvllm connection, orphaned. ruvector-backend.ts (12%) is COMPLETE FACADE — zero ruvector imports, hardcoded "125x speedup", never imported anywhere. sona-service.ts (78%) is the ONLY genuine bridge — real @ruvector/sona wrapper, but has parallel incompatible API with ruvector-integration.ts (beginTrajectory vs startTrajectory). Confirms R40's "single-node task runner" characterization — bridges don't add real cross-system connectivity.

### R136 (2026-03-01): Ghost DEEP files + Rust integration hubs (ML-C)
3 files, ~3,200 LOC, 45 findings. sona-agentdb-integration.ts confirmed DEAD CODE — both critical deps not installed, never imported in production. Source of "150x-12,500x" hardcoded claim. agentdb-mcp-server.ts reassessed upward: EmbeddingService IS initialized, all 8 controllers with DI, 28 real tools, production lifecycle. CORRECTS R20 root-cause claim (applies to claude-flow bridge, not this standalone server). agentdb factory: 5-tier fallback (not 4), BackendType union bug (missing sqljsrvf). New CRITICAL findings: C14 (dead code + missing deps), H28 (causal graph broken), H29 (BackendType union incomplete).

### R139 (2026-03-02): CI/Tests/Deployment ground truth (ML-E)
2 files, ~90 LOC, 7 findings. agentic-flow deployment is demo-only: docker-compose.yml is 9-line single-service demo, docker-compose.agent.yml defines 7 agent types with hardcoded tasks (not operational). Positive: containerized agent execution is architecturally proven — ANTHROPIC_API_KEY pattern correct, 7 agent types defined.

### R141 (2026-03-02): Rust compilation audit
11 crates across agentic-flow workspace, ~19,700 LOC total, 22 findings. 6/11 crates clean. CRITICAL findings: agentic-jujutsu ML-DSA verify() accepts invalid signatures + wrong public keys (C15); reasoningbank-mcp compile failure (C16). agent-booster 6/25 tests fail (H30). agentic-flow-quic Rust confirmed GENUINE: 8/8 tests pass — fault is entirely in TypeScript bridge. reasoningbank workspace largely genuine: 46/46 tests across core/learning/storage/network.

### R142 (2026-03-03): Synthesis update
Incorporated findings from R136-R141 (ML-C through ML-F + Rust audit) into this document. Updated coverage stats, corrected R20 agentdb-mcp-server claim, added 3 new CRITICAL findings (C14-C16), 13 new HIGH findings (H28-H36), new subsystem sections 5n-5o, Rust compilation audit table, deployment section. Total domain findings: 1,998 all-time (262 CRITICAL, 575 HIGH).
