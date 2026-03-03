# Model Routing Domain Analysis

> **Priority**: HIGH | **Coverage**: ~96% DEEP (~92/96 files) | **Status**: ACTIVE
> **Last updated**: 2026-03-03 (Sessions R134, R136, R140)

## Section 1: Current State Summary (as of R140, 2026-03-02)

Model routing is the most fragmented domain in the ruvnet ecosystem. At least **seven distinct routing surfaces** have been confirmed across three languages (Shell, JavaScript, Rust) and four packages (claude-flow-cli, agentic-flow, agentdb, ruvllm). None of these surfaces compose at runtime — each accumulates independently learned patterns in separate state stores.

The JS layer (R11/R17) is what actually runs in production: Q-Learning TD(0) and MoE 2-layer network in claude-flow CLI, plus a seven-file API proxy layer in agentic-flow. The Rust layer (R37, R107, R137) is more sophisticated (HNSW semantic routing, 7-factor complexity analysis, real K-means clustering) but is never called by the CLI.

**R134 (ML-A)** added two CLI entry-point files to the domain. ruvector/bin/cli.js (7,357 LOC) is the user-facing surface for all ruvector commands including the hooks subsystem that wires PreToolUse/PostToolUse routing into Claude. Its Intelligence class is self-contained Q-learning with lazy-loaded IntelligenceEngine, but the ROUTER command prints "Status: Coming Soon" and SERVER/CLUSTER commands are full facades. Hash-based embedding confirmed at the CLI layer: Intelligence.embed() only uses engine embedding if already initialized, otherwise falls back to 64-dim charCode hash. ruvllm/bin/cli.js (1,005 LOC) exposes the ruvllm benchmark/query/models surface — query and generate commands are fully non-functional if the native .node binary is absent, and the training pipeline is simulated (ContrastiveTrainer.train() exports JSONL but does not actually train).

**R136 (ML-C)** re-confirmed the three core ruvllm/claude_flow/ files and added factory.ts:
- **hnsw_router.rs (90-93%)**: BEST ruvector-core integration CONFIRMED. Real HnswIndex instantiation, SONA trajectory recording, HybridRouter blending keyword+semantic. Bugs: pattern metadata inserted before index add_batch (race-condition corruption), SONA trajectory uses same embedding for query and response (degenerate delta), routing_features is 2-element sparse vector.
- **claude_integration.rs (68-73%)**: Setup toolkit, NOT API client. execute_workflow() hardcodes mock results. ClaudeClient struct referenced in architecture doc but does not exist. Genuine: topological workflow dependency resolution, context compression, cost/latency tracking, AgentCoordinator with parking_lot RwLock.
- **model_router.rs (88-92%)**: PARALLEL to TS ADR-008 with zero bridge. Thresholds differ (<0.35 vs <0.30 for haiku, <0.70 vs <0.50 for sonnet). hooks_integration.rs always passes None,None for agent/task overrides — the entire override map system is unreachable dead code. calibration_bias() computed but never called.
- **factory.ts (88-92%)**: 5-tier fallback confirmed (not 4): RuVector > RVF SDK > HNSWLib > sql.js RVF > none. BackendType union omits "sqljsrvf", causing a type error if only sql.js is available. detectBackends() called on every createBackend() (6 dynamic imports each time — no caching).

**R140 (ML-F)** read hooks.ts (4,530 LOC), the V3 CLI hooks command. The pre-task command integrates enhanced-model-router.js for real ADR-008 3-tier routing — the ONLY place in V3 where tier routing is wired to a user-facing command. model-route/outcome/stats connect to model-router.js (tiny-dancer-neural) via hooks-tools.ts. However, hooks-tools.ts semantic routing uses sin/cos character hash (NOT ONNX), so hash-based non-semantic routing extends to the hooks layer. token-optimize unconditionally adds +200 saved tokens and 2 cache hits regardless of actual activity. statusline vector count uses a dbSizeKB/2 file-size heuristic, not a real DB query.

The hash-based embedding problem (R37) is now confirmed systemic: present in pretrain_pipeline.rs (Rust), hnsw_router.rs (Rust, degenerate trajectory), ruvector/bin/cli.js (JS CLI fallback), ruvllm/bin/cli.js (benchmark fallback), and hooks-tools.ts (sin/cos hash). No routing surface uses real learned embeddings end-to-end.

## Overview

Determines which LLM handles each task, routes tasks to agent types, and provides multi-provider API proxying. ~96 files / ~45K LOC. ~92 files DEEP-read.

### File Registry (recent additions — sessions R134, R136, R140)

| File | LOC | Real% | Session | Key Finding |
|------|-----|-------|---------|-------------|
| **npm/packages/ruvector/bin/cli.js** | 7,357 | **55-65%** | R134 | Intelligence class is real Q-learning. ROUTER/SERVER/CLUSTER commands are facades ("Coming Soon"). Hash fallback 64-dim charCode embedding confirmed. HOOKS init (692 lines) generates .claude/settings.json wiring. Export command broken (vectors: []). |
| **npm/packages/ruvllm/bin/cli.js** | 1,005 | **60-70%** | R134 | query/generate/route non-functional without native .node binary. SIMD benchmark genuine. Model download genuine. Training pipeline simulated. Hash embedding in benchmark fallback. ZERO ruvector RAG integration. |
| **crates/ruvllm/src/claude_flow/hnsw_router.rs** | 1,288 | **90-93%** | R136 | Best ruvector-core integration confirmed. HybridRouter + SONA trajectory. Bug: metadata inserted before index batch (race corruption). Degenerate SONA trajectory (same embedding for query+response). |
| **crates/ruvllm/src/claude_flow/claude_integration.rs** | 1,341 | **68-73%** | R136 | Setup toolkit NOT API client. execute_workflow() mocks results. ClaudeClient struct missing. Genuine: topological workflow resolution, cost/latency, AgentCoordinator with parking_lot. |
| **crates/ruvllm/src/claude_flow/model_router.rs** | 1,322 | **88-92%** | R136 | 7-factor complexity, 45 routing patterns. Override maps dead (hooks_integration passes None,None). calibration_bias() computed but never called. Different thresholds from TS ADR-008. |
| **packages/agentdb/src/backends/factory.ts** | 235 | **88-92%** | R136 | 5-tier fallback (not 4). BackendType union omits "sqljsrvf" causing type gap. detectBackends() on every createBackend() (no caching, 6 dynamic imports). GNN/graph assumed present for monolithic ruvector package. |
| **v3/@claude-flow/cli/src/commands/hooks.ts** | 4,530 | **72-78%** | R140 | 30 subcommands (not 17 documented). Real ADR-008 pre-task routing via enhanced-model-router.js. hooks-tools.ts uses sin/cos hash (NOT ONNX). token-optimize adds +200 hardcoded savings. statusline vector count = dbSizeKB/2 heuristic. |

## The Big Picture

Model routing has **seven routing surfaces** across multiple subsystems:

| Subsystem | Components | Status | Evidence |
|-----------|-----------|--------|----------|
| **Hook-based Routing** | route-wrapper.sh, format-routing-directive.sh | **REAL** (advisory) | Outputs [ROUTING DIRECTIVE] text for Claude to read |
| **Q-Learning Router** | q-learning-router.js, route.js (CLI) | **REAL** | TD(0) algorithm, experience replay, MurmurHash3 features |
| **MoE Router** | moe-router.js | **REAL** | 2-layer gating network, REINFORCE gradients, Xavier init |
| **API Proxy Layer** | 7 proxy files (Requesty, OpenRouter, Gemini, WS, H2) | **REAL** | Live API calls, SSE streaming, format conversion |
| **Intelligence Layer** | agent-booster-enhanced.js, SemanticRouter, route.js (MCP hook) | **FABRICATED** | Non-existent functions, fake compression, brute-force claims HNSW |
| **Rust Routing (Unused)** | hnsw_router.rs, model_router.rs, agent_router.rs, task_classifier.rs | **REAL but unused** | Best algorithms, never called by claude-flow CLI. R119: model_router.rs routes by AgentType/ClaudeFlowTask enum overrides + 7-factor complexity scoring. No "skill"-based routing. Adaptive learning is in-memory-only facade (no persistence, no weight updates) |
| **Sheaf Attention Routing** | sheaf/router.rs (ruvector-attention) | **REAL** (token-level) | 4-lane energy dispatch, correct but theta_deep dead, confidence=1.0 |

## How [ROUTING DIRECTIVE] Is Produced (Confirmed R8)

```
UserPromptSubmit Hook
  → route-wrapper.sh (intentionally non-blocking, 5s timeout)
    → npx claude-flow hooks route --task "${prompt}"
      → swarm-gate analysis (JSON output)
        → format-routing-directive.sh (parses JSON via jq)
          → Outputs: [ROUTING DIRECTIVE] Model: ${MODEL} | Complexity: ${COMPLEXITY}
```

- `format-routing-directive.sh` (L43) is THE source of `[ROUTING DIRECTIVE]`
- `route-wrapper.sh` always exits 0 — routing failures never block Claude
- If jq is missing or input invalid, falls back to `Model: sonnet`

## R11 Deep-Read: Routing Algorithms

### Q-Learning Router (681 LOC) — REAL RL

**Correctly implemented algorithms:**
- **TD(0) update** (L339-359): `Q(s,a) += lr * (reward + γ * max(Q(s')) - Q(s,a))`
- **Prioritized experience replay** (L362-412): Circular buffer, priority = |TD error| + 0.01
- **Three epsilon decay strategies** (L416-432): Linear, exponential, cosine
- **MurmurHash3** (L609-628): Proper 32-bit hash for n-gram features
- **Softmax confidence** (L660-665): Numerically stable (subtract max before exp)
- **Feature extraction** (L543-581): 64-dim: 32 keyword binary + 8 length + 8 word count + 8 extension + 8 n-gram hash, L2 normalized

**Limitations:**
- Tabular Q-Learning only, NOT deep RL. No gradient backpropagation
- State space 64-dim with 10K max Q-table entries — risk of state explosion
- Feature extraction is deterministic keyword-based, not semantic
- avgTDError not normalized, can explode with large rewards

### MoE Router (626 LOC) — REAL Neural Network

**Correctly implemented algorithms:**
- **Forward pass** (L240-296): `hidden = ReLU(W1 @ input + b1)`, `probs = softmax(W2 @ hidden + b2 + noise)`
- **Xavier initialization** (L64-74): Box-Muller normal distribution, std = sqrt(2/(fan_in+fan_out))
- **REINFORCE gradients** (L318-388): Full backprop through W2→hidden→W1 with ReLU mask
- **Load balance loss** (L556-572): From Switch Transformer: `L = N * Σ(f_i * P_i)`
- **Gini coefficient** (L577-591): Correct formula for load distribution
- **Top-k expert selection** (L545-553): Selects 2 experts per routing

**Critical gap:** Load balance loss computed but NEVER backpropagated. Gradients are isolated per update (no momentum/batch accumulation). Embedding source (384-dim) not shown.

### Coverage Router (529 LOC) — REAL Parsing, FABRICATED Routing

- **REAL**: LCOV, Istanbul, Cobertura, JSON coverage parsing (L56-168)
- **REAL**: Path traversal protection (L380-401): rejects `..`, null bytes, length limit
- **FABRICATED**: Priority scoring is hardcoded regex guessing (auth=+3, core=+2)
- **FABRICATED**: coverageRoute/Suggest/Gaps return empty if no coverage file exists
- `useNative` metric always false in practice

### CircuitBreakerRouter (460 LOC) — REAL State Machine

- **REAL**: CLOSED→OPEN→HALF_OPEN→CLOSED state transitions (L277-330)
- **REAL**: Timer cleanup prevents leaks, destroy() clears all (L450-457)
- **REAL**: Rate limiter 100 req/min, 5-min block, DJB2 hash keying
- **REAL**: Input validation 10KB max description, timeout bounds 100-60000ms
- **Heuristic**: Uncertainty = failureRate*0.5 + sampleSize*0.3 + recency*0.2 (arbitrary weights)

## R11 Deep-Read: API Proxy Layer (7 files, 3,275 LOC)

All proxy files are **REAL, working code** with live API calls.

| File | LOC | Target API | Auth | Quality |
|------|-----|-----------|------|---------|
| **anthropic-to-requesty.js** | 708 | Requesty (router.requesty.ai) | Bearer token | Best: 60s timeout, schema sanitization, 10-tool limit |
| **anthropic-to-openrouter.js** | 619 | OpenRouter | Bearer token | Good but NO timeout (can hang) |
| **anthropic-to-gemini.js** | 446 | Google Gemini | **INSECURE** query param | API key in URL, XML tool injection |
| **websocket-proxy.js** | 407 | Gemini via WS | Query param | DoS protection (1000 conn), ping/pong heartbeat |
| **http2-proxy.js** | 382 | Gemini via H2 | Header/none | TLS cert validation, 1MB body limit, rate limiting |
| **tool-emulation.js** | 366 | Local | None | ReAct pattern, 5 iterations, confidence scoring |
| **provider-instructions.js** | 347 | Local | None | 8 provider templates, file ops detection |

**Key security findings:**
- **CRITICAL**: Gemini API key in query parameter (?key=...) — exposed in logs, URLs, referrer headers
- **HIGH**: API key prefix leaked in debug logs (requesty L141, openrouter L138)
- **HIGH**: OpenRouter calls have NO timeout — requests can hang indefinitely

## R11 Deep-Read: Advanced Routers (5 files)

| File | LOC | Status | Key Finding |
|------|-----|--------|-------------|
| **TinyDancerRouter.js** | 407 | **Partially REAL** | Native @ruvector/tiny-dancer is real compiled binary. JS fallback = cosine similarity + softmax |
| **SemanticRouter.js** | 290 | **FABRICATED** | Claims HNSW but implements brute-force. Code comments admit it |
| **onnx-local.js** | 294 | **REAL** | True ONNX inference: onnxruntime-node, Phi-4 INT4, KV cache, auto-download from HuggingFace |
| **onnx.js** | 264 | **REAL** | @xenova/transformers, Phi-3-mini quantized. Streaming is simulated (word chunking with 10ms delays) |
| **openrouter.js** | 245 | **REAL** | OpenRouter API client, real SSE streaming, full tool use. Cost hardcoded (inaccurate) |

### ONNX Inference is Genuinely Real

Two independent ONNX implementations:
1. **onnx-local.js**: Direct onnxruntime-node, Phi-4-mini INT4 (~4.9GB), greedy decoding, KV cache for 32-layer transformer, tiktoken cl100k_base
2. **onnx.js**: @xenova/transformers wrapper, Phi-3-mini quantized, top-p sampling (0.9), platform-based provider detection

Both have cost=0 (local inference). onnx-local.js has REAL streaming; onnx.js simulates it.

## R11 Deep-Read: Provider Management & Intelligence (4 files)

### provider-manager.js (435 LOC) — 85% REAL

- **REAL**: Circuit breaker, exponential backoff (1-30s), fallback chains
- **REAL**: Performance scoring = successRate*0.7 - normalizedLatency*0.3
- **REAL**: Round-robin and cost optimization strategies
- **INCOMPLETE**: Health check is TODO (L86). No circuit breaker recovery test. Flat-rate cost only

### LLMRouter.js (AgentDB, 570 LOC) — 60% REAL

Provider chain: RuvLLM → OpenRouter → Gemini → Anthropic → ONNX. External API calls genuine. RuvLLM integration speculative (dynamic import may not exist). Local fallback returns hardcoded template strings for keywords.

### route.js (MCP Hook, 267 LOC) — 40% REAL

- ~~**FABRICATED**: RuVector intelligence facade (L54-96) references non-existent functions~~ **RESOLVED R14**: intelligence-bridge.js EXISTS (1,038 LOC). routeTaskIntelligent() and findSimilarPatterns() confirmed present. The route.js references are REAL, not fabricated — they import from a different package path than initially checked.
- **REAL**: Q-learning fallback (L98-264) with epsilon-greedy (10%), file patterns, keyword scoring, memory similarity, error patterns
- Q-learning state oversimplified as string `edit:${ext}`

### agent-booster-enhanced.js (1,122 LOC) — 25% REAL

- **FABRICATED**: Compression tier system completely fake. TensorCompress not a real ruvector export. Claims 87.5-96.9% savings but nothing compressed
- **FABRICATED**: GNN differentiableSearch() calls non-existent function. "WASM Agent Booster" loads no WASM
- **REAL**: Exact cache matching (hash-based). Fuzzy matching (cosine similarity, threshold 0.85). Error pattern learning. Pattern persistence. 24 pretrain code edit patterns

## THREE Disconnected Routing Systems (Confirmed R8+R11)

| System | Package | Function | Status |
|--------|---------|----------|--------|
| 3-Tier Hook Routing | claude-flow-cli | Outputs `[ROUTING DIRECTIVE]` text | Advisory only |
| LLMRouter | agentdb | Provider routing (5 providers) | Real but unused by claude-flow |
| Agent Task Router | claude-flow-cli + claude-config | Pattern-matches tasks to agent types | Real but NOT model routing |
| Q-Learning/MoE Routers | claude-flow-cli | RL-based task→agent routing | Real algorithms, CLI-only |
| API Proxy Layer | agentic-flow | Format conversion to 3rd-party APIs | Real, production-quality |

These systems **do not coordinate**. Provider-manager selects LLM providers. Route.js selects agent types. Agent-booster caches code edits. No orchestration layer connects them.

## CRITICAL Findings (11)

1. **Gemini API key in query parameter** — Exposed in HTTP logs, URLs, referrer headers across 3 files (gemini proxy, websocket proxy, http2 proxy).
2. **SemanticRouter HNSW is fabricated** — Claims HNSW-powered routing but code implements brute-force cosine similarity. Comments acknowledge this.
3. **agent-booster compression fabricated** — TensorCompress not real. Claims 87.5-96.9% savings but embeddings stored uncompressed. GNN search calls non-existent function.
4. ~~**RuVector intelligence facade**~~ **RESOLVED R14** — intelligence-bridge.js EXISTS (1,038 LOC). routeTaskIntelligent() at L382 and findSimilarPatterns() at L542 both confirmed working. Research synchronization error, not code deficiency.
5. **Three fragmented ReasoningBanks** — Learning system informing decisions broken across 3 packages.
6. **Hash-based embeddings SYSTEMIC across all routing layers** (R134) — Confirmed in ruvector/bin/cli.js (Intelligence.embed() 64-dim charCode fallback), ruvllm/bin/cli.js (Math.sin() benchmark fallback), hooks-tools.ts (sin/cos character hash for semantic routing), hnsw_router.rs (degenerate SONA trajectory using same embedding for query and response), and pretrain_pipeline.rs (character sum % dim). No routing surface uses real learned embeddings end-to-end. All HNSW semantic routing is non-semantic.
7. **ruvllm CLI non-functional without native binary** (R134) — ruvllm/bin/cli.js query(), generate(), route() fallback returns hardcoded "[Fallback] Response to: ..." with confidence 0.5 (engine.ts L121-131). Without the native .node binary, ALL query/generate/route/embed commands are decoration. Training is simulated (ContrastiveTrainer.train() exports JSONL for external GPU use but does not train).
8. **execute_workflow() hardcodes mock results** (R136) — claude_integration.rs execute_workflow() always returns success=true, tokens_used=500, cost=0.001. No Claude API call. The entire workflow orchestration layer is simulation.
9. **ClaudeClient struct missing** (R136) — claude_integration.rs architecture doc references ClaudeClient as the core API interface but the struct does not exist anywhere in the file or crate. The Rust API integration layer is entirely absent.
10. **ROUTER/SERVER/CLUSTER commands are facades** (R134) — ruvector/bin/cli.js ROUTER prints "Status: Coming Soon", SERVER prints "Status: Coming Soon" with planned features list, CLUSTER prints "Status: Coming Soon" with Raft/failover features list. Zero implementation in all three. These are the most user-visible commands in the ruvector CLI.
11. **hooks-tools.ts semantic routing uses sin/cos hash** (R140) — The hooks model-route command's semantic embedding uses generateSimpleEmbedding() (sin/cos character hash) even when native VectorDb/HNSW is loaded. The HNSW index contains hash-embedded routing patterns — extends hash-embedding failure to the V3 hooks layer that surfaces ADR-008 routing to users.

## HIGH Findings (34)

1. **Q-Learning router is REAL** — TD(0), experience replay, 3 epsilon decay, MurmurHash3 features all correct.
2. **MoE router is REAL** — Xavier init, REINFORCE gradients, forward pass all correct. Load balance loss computed but not backpropagated.
3. **CircuitBreakerRouter is REAL** — Proper state machine, timer cleanup, rate limiting.
4. **All 7 proxy files are REAL** — Live API calls to Requesty, OpenRouter, Gemini with format conversion.
5. **ONNX inference is REAL** — Two implementations (onnxruntime-node + transformers.js) with real models.
6. **OpenRouter proxy has NO timeout** — Can hang indefinitely unlike Requesty's 60s timeout.
7. **API key prefixes leaked in debug logs** — Requesty and OpenRouter proxies expose first 10 chars.
8. **Coverage router is fabricated facade** — Parsing works but routing returns empty without coverage files on disk.
9. **Provider-manager health check not implemented** — TODO comment, no actual health checking.
10. **THREE disconnected routing systems** — No integration between hook advisory, provider selection, and agent task routing.
11. **TinyDancerRouter native binary is REAL** — @ruvector/tiny-dancer compiled FastGRNN. JS fallback is cosine similarity.
12. **agent-booster REAL components** — Exact cache, fuzzy matching, error patterns, persistence all genuinely work.
13. **Rust routing is BEST but unused** — hnsw_router.rs at 90-93% real is the most sophisticated routing in the ecosystem, but claude-flow uses JS Q-learning instead. (R37)
14. **Four distinct ReasoningBanks** — Rust reasoning_bank.rs is the fourth independent implementation with zero code sharing. (R37)
15. **6th routing surface: AgentRouter runs parallel with no composition** — AgentRouter (keyword+SONA), HnswRouter (HNSW semantic), ModelRouter (token-count complexity), LLMRouter, RuvLLMOrchestrator, and claude_flow_bridge CLI each hold separate SonaIntegration instances, accumulating different learned patterns independently. (R107)
16. **theta_deep config field dead in sheaf/router.rs** — route_by_energy() 4-lane dispatch skips theta_deep threshold entirely. Config field validated and documented but never read. (R107)
17. **RoutingDecision.confidence hardcoded to 1.0** — All routing decisions from sheaf/router.rs report confidence=1.0 regardless of energy proximity to thresholds. (R107)
18. **SONA adaptive interface in sheaf/router.rs is passive** — tune_thresholds() provides API for feedback-driven threshold tuning but TokenRouter has no internal feedback loop. Caller must externally invoke manually. (R107)
19. **stop_sequences silently ignored in serving/request.rs** — should_stop() delegates to is_complete() (token count only). Stop finish_reason never populated. (R107)
20. **AgentRouter tests never exercise SONA path** — All tests use SonaConfig::default() (cold-start). CRITICAL model-index mismatch is completely untested. (R107)
21. **claude_integration.rs dead imports** — ClaudeFlowAgent and ClaudeFlowTask imported but never referenced. (R137)
22. **claude_integration.rs ResponseStreamer has no data source** — Token processing with mpsc channels is real but nothing generates tokens. No HTTP client, SSE parser, or stream reader. (R137)
23. **model_router.rs override maps dead code** — hooks_integration.rs always passes None,None for agent/task overrides, making the AgentType/TaskType override map system unreachable. (R137)
24. **model_router.rs parallel to TS ADR-008 with different thresholds** — Rust <0.35 haiku / <0.70 sonnet vs TS <0.30 / <0.50. No FFI bridge, no WASM interface, no mechanism connecting the two. (R137)
25. **hooks.ts pre-task embeds real ADR-008 3-tier model routing** — enhanced-model-router.js integration at L1529-1576 provides legitimate [TASK_MODEL_RECOMMENDATION] directives. The ONLY place in V3 where ADR-008 tier routing is wired to a user-facing command. (R140)
26. **hooks-tools.ts semantic routing uses sin/cos hash embeddings** — route command's generateSimpleEmbedding() used even when native VectorDb/HNSW is loaded. Hash-embedding failure propagated to hooks routing layer. (R140)
27. **ruvector/bin/cli.js Intelligence class genuine** — Self-contained Q-learning + vector memory + trajectory tracking + co-edit patterns + error learning. Lazy-loads IntelligenceEngine. PRETRAIN phase (549 lines, 11 phases) is real: git co-edit patterns, AST patterns, dir-agent mappings. (R134)
28. **ruvector/bin/cli.js export command broken** — Line 1858 exports {vectors: []} regardless of DB content. Vector data never serialized. Export/import pipeline is broken by design. (R134)
29. **ruvllm/bin/cli.js SIMD benchmark genuine** — Creates test vectors, measures real Date.now() timing for dotProduct, cosineSimilarity, l2Distance, softmax. Reports ops/sec. (R134)
30. **hnsw_router.rs metadata-before-index race** — add_patterns() inserts metadata into DashMaps BEFORE calling index.add_batch(). If add_batch() fails, metadata exists without index entries creating orphaned entries. (R136)
31. **factory.ts BackendType union gap** — BackendType omits "sqljsrvf" but BackendDetection.available CAN be "sqljsrvf". If only sql.js available, createBackend() throws an unhandled type error. (R136)
32. **factory.ts detectBackends() called on every create** — 6 dynamic imports on every createBackend() invocation with no caching. Any application creating backends repeatedly pays full detection cost each time. (R136)
33. **hooks.ts token-optimize fabricates savings** — Unconditionally adds +200 totalTokensSaved and sets cacheHits=2 regardless of optimization activity. Cumulative display is inflation not measurement. (R140)
34. **hooks.ts statusline vector count is heuristic** — agentdbStats.vectorCount = Math.floor(agentdbStats.dbSizeKB / 2). Size-based proxy, not actual DB query. Intelligence stats output is systematically wrong. (R140)

## MEDIUM Findings (selected new additions — R134/R136/R140)

Existing findings 1-11 unchanged (see prior session entries above). New additions:

12. **ruvector/bin/cli.js monolithic structure risk** — 7,357 LOC single file with 14+ top-level commands. No modular separation. All routing, intelligence, and hook commands co-located. (R134)
13. **ruvllm/bin/cli.js ZERO ruvector RAG integration** — No imports from ruvector-core, no HNSW index creation, no vector store connection. Memory commands use RuvLLM.addMemory/searchMemory (native binary or in-memory map). (R134)
14. **ruvector/bin/cli.js intelligence path resolution has no sanitization** — getIntelPath() checks .ruvector/.claude directories then falls back to HOME/.ruvector/. No path sanitization against directory traversal if cwd() is manipulated. (R134)
15. **hnsw_router.rs distance-to-similarity conversion incorrect for non-cosine metrics** — 1.0 - score.max(0.0).min(2.0) is only correct for cosine distance (range 0-2). For Euclidean/DotProduct produces incorrect similarity values. (R136)
16. **hnsw_router.rs SONA trajectory sparse features** — routing_features vector is only 2 elements: [agent_type_as_f32/10, success_rate]. Extremely sparse for a learning system — no task complexity, latency, or embedding characteristics. (R136)
17. **hnsw_router.rs SIMD comment misleading** — normalize_embedding() uses scalar loop, not SIMD despite comment "SIMD-friendly". Actual SIMD is in ruvector-core distance calculations. (R136)
18. **model_router.rs calibration_bias() never called** — Lines 660-688 computes signed mean error but no consumer invokes it. Accuracy calibration dead code. (R136)
19. **model_router.rs dual TaskComplexityAnalyzer** — hooks_integration.rs instantiates BOTH ModelRouter (containing its own TaskComplexityAnalyzer) AND a separate standalone TaskComplexityAnalyzer. Two independent analyzers running in parallel, no shared state. (R136)
20. **model_router.rs token estimation heuristic** — estimate_tokens() uses len()/4 as base multiplied by keyword-derived factors (1.2x-3.0x). "Fix typo" (8 chars) gets 2 base tokens * 1.2 = 2 tokens, severely underestimating. (R136)
21. **claude_integration.rs model IDs hardcoded as dated versions** — claude-3-5-haiku-20241022, claude-sonnet-4-20250514, claude-opus-4-20250514. Will become stale as Anthropic releases new versions. No dynamic model resolution. (R136)
22. **hooks.ts pretrain command cosmetic delay** — Calls await new Promise(resolve => setTimeout(resolve, 800)) BEFORE each real MCP call. 6-step progress animation is theatrical delay not actual work. (R140)
23. **hooks.ts exports 30 subcommands vs 17 documented** — Includes 4 v2 backward-compat aliases and 3 coverage-aware commands absent from all documentation. Undocumented routing surface. (R140)
24. **factory.ts GNN/graph capability falsely assumed for monolithic ruvector** — When "ruvector" package detected, GNN and graph capabilities set true unconditionally ("Main package includes GNN/Graph" comment). Actual capability depends on which submodules compiled. (R136)

## Positive

- **Q-Learning and MoE routers** implement real, correct RL/ML algorithms
- **CircuitBreakerRouter** is a solid, well-implemented fault tolerance pattern
- **API proxy layer** is production-quality with real API integrations
- **ONNX local inference** genuinely works with Phi-4/Phi-3 models
- **TinyDancerRouter** has real compiled native binary (FastGRNN in Rust)
- **provider-manager** has real exponential backoff, circuit breaking, and cost optimization
- **agent-booster** has genuine pattern caching, fuzzy matching, and persistence
- **hnsw_router.rs** has real HNSW with M/ef configuration and pattern consolidation — production-quality semantic routing (R37)
- **model_router.rs** has genuine 7-factor complexity analysis with feedback tracking (R37)
- **reasoning_bank.rs** has real K-means + EWC++ — best mathematical foundation of all 4 ReasoningBank implementations (R37)
- **ruvector/bin/cli.js PRETRAIN** is genuinely implemented (549 lines, 11 phases): git co-edit patterns, key-file vector memories, dir-agent mappings, AST patterns — real data from real project artifacts (R134)
- **ruvllm/bin/cli.js model management** is real infrastructure — ModelDownloader handles HuggingFace downloads with progress bars, force re-download, per-model status tracking. SIMD benchmark is genuine timing measurement. (R134)
- **claude_integration.rs AgentCoordinator** implements genuine topological workflow dependency resolution with circular dependency detection and parking_lot RwLock concurrency. 7 unit tests cover cost, compression, token estimation. (R136)
- **hooks.ts pre-task ADR-008 routing** is genuinely wired to enhanced-model-router.js — the only place in V3 where 3-tier model routing reaches a user-facing CLI command (R140)
- **sona-optimizer.ts** is GENUINELY FUNCTIONAL Bayesian agent-routing with temporal decay, wired into hooks pipeline (R140, from ML-F session)

## R17 Closeout (2026-02-14)

All 24 remaining files deep-read. Domain now at **94.9% DEEP** (56/59 files, 3 MEDIUM).

### Proxy Layer Summary
| File | LOC | Real % | Key Finding |
|------|-----|--------|-------------|
| quic-proxy.js | 228 | 50% | Depends on unverified transport/quic.js. QUIC feature-flagged off by default. |
| adaptive-proxy.js | 225 | 70% | Multi-protocol fallback (H3→H2→H1→WS). 4 unverified proxy dependencies. |
| anthropic-to-onnx.js | 214 | 90% | Real Express proxy. Converts Anthropic→ONNX format. No streaming. |
| cli-standalone-proxy.js | 198 | 95% | Working CLI for Gemini/OpenRouter proxying. |
| http2-proxy-optimized.js | 192 | 40% | 5 optimization utils unverified. Performance claims unverified. |
| http3-proxy.js | 52 | 10% | **STUB** — TODO comment, always falls back to HTTP/2. |

### Provider Layer Summary
| File | LOC | Real % | Key Finding |
|------|-----|--------|-------------|
| anthropic.js | 98 | 100% | Full MCP + tool support via @anthropic-ai/sdk. |
| gemini.js | 103 | 95% | Uses @google/genai. No tool/MCP support. |
| onnx-phi4.js | 191 | 70% | API fallback works. **Local ONNX NOT implemented** — throws error. |
| onnx-local-optimized.js | 168 | 80% | Naive sliding window context pruning (2048 tokens). |
| model-mapping.js | 132 | 90% | Maps Claude models across Anthropic/OpenRouter/Bedrock. |

### Semantic Router (claude-flow-cli)
- semantic-router.js (178 LOC): Pure JS fallback for @ruvector/router. Brute-force cosine similarity — no HNSW.

### CRITICAL: ONNX Local Inference Chain
`onnx-phi4.js` → `onnx-local.js` → `model-downloader.js` (unverified). Local ONNX throws "not yet implemented". Only API mode works via HuggingFace.

## R37: Rust Model Routing Deep-Read (Session 37)

### Overview

R37 deep-read of 4 ruvllm/claude_flow files reveals the **Rust equivalent** of the JS routing system analyzed in R11/R17. The Rust routing is significantly more sophisticated than JS, with real ML algorithms, but shares the same critical gap: hash-based embeddings.

### Rust Routing Files (4 files, 5,494 LOC) — 88% REAL

| File | LOC | Real% | Key Finding |
|------|-----|-------|-------------|
| **hnsw_router.rs** | 1,288 | **90-93%** | **BEST ruvector-core integration in project**. HybridRouter blends HNSW semantic + keyword routing with confidence weighting. Real HnswIndex with M/ef config, batch adds, genuine search. Pattern consolidation merges similar patterns by agent_type. Ghost DEEP corrected R137. |
| **claude_integration.rs** | 1,341 | **68-73%** | Setup toolkit NOT API client. Claude API types + context compression + cost tracking genuine. execute_workflow() hardcodes mock results. ClaudeClient referenced but doesn't exist. Dead imports (ClaudeFlowAgent, ClaudeFlowTask). ResponseStreamer has no data source. Ghost DEEP corrected R137. |
| **model_router.rs** | 1,322 | **88-92%** | 7-factor complexity analyzer (code length, test presence, multi-file, security keywords, etc.), model selector with cost/latency constraints. 45 routing patterns. record_feedback tracks last 1000 predictions with accuracy stats. Parallel to TS ADR-008 with different thresholds. Override maps dead (hooks_integration passes None,None). Re-read R137. |
| **pretrain_pipeline.rs** | 1,394 | **85-88%** | Multi-phase pretraining: Bootstrap → Synthetic → Reinforce → Consolidate. Curriculum learning with difficulty progression. **CRITICAL**: generate_embedding is HASH-BASED (character sum % dim). Quality scores simulated with rand_simple(). |
| **reasoning_bank.rs** | 1,520 | **92-95%** | Production ReasoningBank: real K-means clustering (10 iterations, convergence check), EWC++ consolidation, pattern distillation. 16 tests. Informs routing decisions via pattern retrieval. |
| **agent_router.rs** (R107) | 311 | **72-77%** | **6th routing surface**. Dual-path: SONA + keyword fallback. CRITICAL: sona_to_routing_decision maps quality-tier indices to agent types — semantic mismatch. Degenerate trajectory (response_embedding=query_embedding). 8+ AgentType → 4-bucket routing with CicdEngineer→Coder collapse. |
| **sheaf/router.rs** (R107) | 666 | **85-90%** | 4-lane energy dispatch (Reflex/Standard/Deep/Escalate). Genuine SheafAttention composition via route_token(). theta_deep config field dead. confidence hardcoded 1.0. SONA adaptive interface passive (caller-driven). 12 tests. |
| **serving/request.rs** (R107) | 473 | **88-92%** | Priority enum (Low/Normal/High/Critical) feeds scheduler. 6-state RequestState including Preempted. stop_sequences silently ignored (functional bug). RunningRequest has dual kv_cache_slot + block_table (potential dead field). Full timing breakdown for P50/P99. |

### How Rust Routing Relates to JS Routing

| Aspect | JS (claude-flow CLI) | Rust (ruvllm crate) |
|--------|---------------------|---------------------|
| Algorithm | Q-Learning TD(0), MoE 2-layer | 7-factor analyzer, HNSW semantic, HybridRouter |
| Sophistication | Tabular RL (10K state limit) | K-means clustering, EWC++ consolidation |
| Embedding quality | Hash-based (JS hooks) | Hash-based (same pattern!) |
| Integration | Advisory [ROUTING DIRECTIVE] text | Compiled Rust binary via NAPI |
| Training data | 24 pretrain patterns | 140+ tool templates, 60+ Claude task templates |
| Status | Active in CLI | Not integrated into claude-flow |

**Key finding**: The Rust routing system is 3-4x more sophisticated than the JS version but **NEVER USED** by claude-flow. The JS Q-learning/MoE routers (R11) are what actually runs, while the Rust routing sits in the ruvllm crate unused.

### Updated Assessment (R107)

The model-routing domain now has **SEVEN routing surfaces** across four packages:

| Surface # | System | Package | Language | Runtime Status |
|-----------|--------|---------|----------|----------------|
| 1 | Hook-based Routing | claude-flow-cli | Shell/JS | **Active** (advisory text only) |
| 2 | Q-Learning/MoE | claude-flow-cli | JS | **Active** (CLI-only) |
| 3 | API Proxy Layer | agentic-flow | JS | **Active** (production) |
| 4 | LLMRouter | agentdb | JS/TS | Available but unused |
| 5 | HnswRouter + ModelRouter + pretrain_pipeline | ruvllm | Rust | Available but unused |
| 6 | AgentRouter | ruvllm/claude_flow | Rust | Available but SONA broken (keyword only works) |
| 7 | task_classifier.rs | ruvllm | Rust | Keyword-only, parallel to AgentRouter |
| — | sheaf/router.rs | ruvector-attention | Rust | Token-level energy dispatch (different layer) |

No system in surfaces 4–7 is called by the claude-flow CLI. Surfaces 5, 6, and 7 each hold independent SonaIntegration instances with no shared state.

### R37 Findings

**CRITICAL** (+2):
6. **Hash-based embeddings in Rust routing** — pretrain_pipeline.rs generate_embedding uses character sum % dim. All HNSW routing patterns stored with fake embeddings, making semantic search non-semantic. Same pattern as ruvector-core. (R37)
7. **SONA model-index semantic mismatch in AgentRouter** — sona_to_routing_decision maps suggested_model (a quality-tier index: 0=best, 1=medium, 2=low from SONA avg_quality thresholds) to AgentType (0=Coder, 1=Researcher, 2=Tester, 3=Reviewer). High quality produces Coder, medium produces Researcher, low produces Tester — completely disconnected from which agent was used in training. Additionally, model_index in record_feedback is set to agent_used as usize (AgentType has 8 variants, SONA model_index expects ModelSize 0-3), poisoning the pattern store with out-of-range indices. Degenerate trajectory: response_embedding = query_embedding.to_vec() collapses SONA's delta-learning to zero. Cold-start keyword path is the ONLY working routing. (R107, agent_router.rs)

8. **claude_integration.rs execute_workflow() hardcodes mock results** — Always returns success, 500 tokens, $0.001 cost. No Claude API call. Entire workflow orchestration is simulation. (R137, claude_integration.rs)
9. **claude_integration.rs missing ClaudeClient** — Architecture doc (lines 16-27) references `ClaudeClient` as core component but the struct does not exist anywhere in the file or crate. API integration layer completely absent. (R137, claude_integration.rs)

**HIGH** (+6):
13. **Rust routing is BEST but unused** — hnsw_router.rs at 90-93% real is the most sophisticated routing in the ecosystem, but claude-flow uses JS Q-learning instead. (R37)
14. **Four distinct ReasoningBanks** — Rust reasoning_bank.rs is the fourth independent implementation with zero code sharing. (R37)
15. **6th routing surface: AgentRouter runs parallel with no composition** — AgentRouter (keyword+SONA), HnswRouter (HNSW semantic), ModelRouter (token-count complexity), LLMRouter, RuvLLMOrchestrator, and claude_flow_bridge CLI each hold separate SonaIntegration instances, accumulating different learned patterns independently with no runtime wiring. (R107, agent_router.rs)
16. **theta_deep config field dead in sheaf/router.rs** — route_by_energy() has four lanes (Reflex/Standard/Deep/Escalate) but the routing logic skips the theta_deep threshold entirely: E < theta_reflex → Reflex, E < theta_standard → Standard, E < theta_escalate → Deep, else Escalate. theta_deep is validated and documented but never read during routing. (R107, sheaf/router.rs)
17. **RoutingDecision.confidence hardcoded to 1.0** — All routing decisions from sheaf/router.rs report confidence=1.0 regardless of energy proximity to thresholds. Any consumer of RoutingDecision.confidence receives stale maximum confidence. (R107, sheaf/router.rs)
18. **SONA adaptive interface in sheaf/router.rs is passive** — tune_thresholds() and config_mut() provide the external API for SONA feedback-driven threshold tuning, but TokenRouter has no internal feedback loop. Caller must externally collect LaneStatistics and invoke tune_thresholds() manually. Not autonomous. (R107, sheaf/router.rs)
19. **stop_sequences silently ignored in serving/request.rs** — should_stop() takes _decoded_text but never inspects it, delegating to is_complete() (token count only). Any caller passing stop=[...] will never see a Stop finish_reason — functional bug for stop-sequence use cases. (R107, serving/request.rs)
20. **AgentRouter tests never exercise SONA path** — All tests use SonaConfig::default() (cold-start), so the SONA routing path is never tested. No test for sona_to_routing_decision, record_feedback, accuracy(), or SONA-path routing. The CRITICAL model-index mismatch (Finding #7) is completely untested. (R107, agent_router.rs)
21. **claude_integration.rs dead imports** — `ClaudeFlowAgent` and `ClaudeFlowTask` imported from parent module but never referenced. Indicates unfinished integration with the broader claude_flow module. (R137, claude_integration.rs)
22. **claude_integration.rs ResponseStreamer has no data source** — Token processing logic via mpsc channels is real but nothing generates the tokens. No HTTP client, no SSE parser, no stream reader. (R137, claude_integration.rs)
23. **model_router.rs override maps dead code** — hooks_integration.rs (the sole consumer) always passes `None, None` for agent_type and task_type overrides in `route()`, making the entire AgentType/TaskType override map system (lines 1104-1123) unreachable. (R137, model_router.rs)
24. **model_router.rs parallel to TS ADR-008 with different thresholds** — Rust uses <0.35 for Haiku, <0.70 for Sonnet; TS ADR-008 uses <0.30/<0.50. No FFI bridge, no WASM interface, no mechanism connecting the two. (R137, model_router.rs)

**Positive** (+6):
- **hnsw_router.rs** has real HNSW with M/ef configuration and pattern consolidation — production-quality semantic routing (R37)
- **model_router.rs** has genuine 7-factor complexity analysis with feedback tracking (R37)
- **reasoning_bank.rs** has real K-means + EWC++ — best mathematical foundation of all 4 ReasoningBank implementations (R37)
- **agent_router.rs dual-path architecture is structurally sound** — primary SONA path with ReasoningBank pattern store and 3-loop learning; deterministic keyword fallback. Bugs are in trajectory encoding only, not in dispatch logic. (R107)
- **sheaf/router.rs 4-lane energy dispatch is genuinely correct** — Reflex/Standard/Deep/Escalate thresholds gate compute depth in semantic alignment with sheaf energy semantics. route_token() genuinely composes with SheafAttention.average_token_energy(). 12 tests including lane ordering, config validation, batch routing. (R107)
- **serving/request.rs InferenceRequest/RunningRequest/CompletedRequest tripartite lifecycle** — clean state-machine with chunked prefill (advance_prefill/complete_prefill), paged-attention block_table, full timing breakdown (prefill_time_ms, decode_time_ms) for P50/P99 latency. (R107)

## R107 Deep-Read: AgentRouter, Sheaf Router, Serving Request (3 files, ~1,450 LOC)

### agent_router.rs (311 LOC) — 72-77% REAL

**AgentRouter** is the sixth confirmed routing surface in ruvllm, providing both a SONA-learned path and a deterministic keyword fallback for routing tasks to agent types (Coder, Researcher, Tester, Reviewer, etc.).

**Working components:**
- Keyword scoring across 8+ agent buckets with ClaudeFlowAgent→AgentType From conversion
- Dual-path dispatch: SONA path activates only when embedding is Some AND based_on_patterns > 0 AND confidence > 0.6
- RoutingDecision struct with confidence, alternatives (Vec<AgentType>), task_type string, reasoning string

**CRITICAL BROKEN components:**
- sona_to_routing_decision maps SONA suggested_model (quality-tier: 0=best, 1=medium, 2=low) to AgentType (0=Coder, 1=Researcher, 2=Tester, 3=Reviewer). Semantic mismatch is total — high quality routing always returns Coder, medium always returns Researcher.
- record_feedback sets model_index = agent_used as usize. AgentType has 8 variants (0-7); SONA model_index expects ModelSize (0=Tiny, 1=Small, 2=Medium, 3=Large). Agent Security (index 5) or Reviewer (index 3) produce out-of-range or wrong-semantic model_index values.
- response_embedding = query_embedding.to_vec() — degenerate trajectory means SONA 3-loop learning receives zero delta signal. SONA cannot distinguish request from response context.
- Cold-start always falls through to keyword routing (SONA confidence gate: based_on_patterns > 0 required). SONA path never activates from scratch.
- CicdEngineer maps to Coder (no CicdEngineer AgentType variant), collapsing CI/CD specialty.
- 5 tests cover only keyword paths. SONA path completely untested.

**Relation to other surfaces:** AgentRouter and HnswRouter both hold separate SonaIntegration instances. Neither shares state with the JS Q-learning router in claude-flow CLI. Surfaces 5, 6, and 7 accumulate independently.

### sheaf/router.rs (666 LOC) — 85-90% REAL

**TokenRouter** is an energy-gated compute dispatch router operating at the token level within ruvector-attention. It is architecturally distinct from model-selection routing — it determines which compute lane (Reflex/Standard/Deep/Escalate) processes each token, not which model to call.

**Working components:**
- route_token(): calls SheafAttention.average_token_energy() and token_energy(), gates to lane by energy thresholds. Genuine attention composition.
- route_by_energy(): 4-lane dispatch with ordered thresholds (theta_reflex < theta_standard < theta_escalate)
- LaneStatistics: mean_energy, token_count per lane — basis for adaptive threshold tuning
- tune_thresholds(): adjusts thresholds from LaneStatistics (caller-driven, not autonomous)
- TokenRouterConfig: 6 with_* builder methods, validate() with ordered threshold check
- Small-context guard: context.len() < min_context_size → Standard lane default
- 12 tests: lane ordering, config validation, route_by_energy, route_token, route_batch, group_by_lane, LaneStatistics, builder pattern, small-context default

**Gaps:**
- theta_deep: second high threshold config field, NEVER READ in route_by_energy(). Dead field despite documentation and validation.
- confidence hardcoded to 1.0 for all decisions (placeholder comment present)
- route_batch() is sequential map over route_token() — no batch-level optimization
- estimate_latency_ms() uses magic numbers (0.1/1/5ms per lane), not measured values
- SONA interface: structurally ready (tune_thresholds + config_mut) but passive — no autonomous adaptation

### serving/request.rs (473 LOC) — 88-92% REAL

Routing-adjacent infrastructure for the vLLM-style serving layer. Provides the scheduling primitives that feed into scheduler.rs and batch.rs (R106).

**Working components:**
- Priority enum (Low/Normal/High/Critical) with Ord derivation and value() → u8 accessor. Feeds scheduler priority queues.
- RequestState: 6 states including Preempted — validates R106 batch.rs preemption claims
- InferenceRequest builder pattern: with_priority, with_session, with_metadata. arrival_time: Instant at construction (correct for wait-time SLA).
- RunningRequest: kv_cache_slot + block_table (paged attention), chunked prefill (advance_prefill/complete_prefill/get_prefill_tokens), add_token() with decode_steps tracking, tokens_per_second() genuine throughput
- CompletedRequest: full timing breakdown (processing, waiting, prefill, decode), success()/failure()/cancelled() factory methods consuming RunningRequest
- FinishReason: Length/Stop/EndOfSequence/Cancelled/Error. Length vs EndOfSequence correctly assigned.
- TokenOutput: request_id, token_id, token_text (Option), logprob (Option<f32>), is_final, finish_reason (Option) — OpenAI-compatible SSE chunk primitive

**Gaps:**
- stop_sequences in GenerateParams silently ignored: should_stop() never inspects _decoded_text, delegates to is_complete() (token count only). Stop finish_reason never populated from success() factory.
- kv_cache_slot vs block_table: two parallel indexing schemes present — kv_cache_slot may be legacy pre-paged design; block_table is correct paged mechanism.
- max_seq_len computed at construction without context window validation — oversized requests can be queued silently.
- EOS token check explicitly deferred (comment: tokenizer not wired here) — is_complete() = max-token check only.

## R140: Hooks CLI Command Layer (1 file, 4,530 LOC)

### hooks.ts (4,530 LOC) — 72-78% REAL

**The claude-flow hooks CLI command** is a 4,530-line TypeScript file implementing 30 subcommands as callMCPTool() wrappers delegating to hooks-tools.ts (3,281 LOC). It is the user-facing surface for the ADR-005 MCP-first hooks architecture.

**Confirmed genuine (model-routing relevant):**
- **pre-task** integrates enhanced-model-router.js for ADR-008 3-tier routing (L1529-1576) — produces [AGENT_BOOSTER_AVAILABLE] or [TASK_MODEL_RECOMMENDATION] directives
- **model-route** connects to model-router.js (tiny-dancer-neural) with real complexity scoring and pattern learning via recordOutcome()
- **model-outcome** and **model-stats** feed the learning loop for adaptive routing decisions
- ADR-008 thresholds in hooks.ts: <0.3 haiku, 0.3-0.5 sonnet, >0.5 opus — parallel to but NOT bridged to model_router.rs Rust layer (<0.35/<0.70)

**Bugs (model-routing relevant):**
- **statusline vector count** = dbSizeKB/2 file-size heuristic, NOT a real AgentDB query
- **token-optimize** unconditionally adds stats.totalTokensSaved += 200 and cacheHits = 2 regardless of actual activity — fabricated efficiency display

**File registry row:**

| File | LOC | Real% | Key Finding |
|------|-----|-------|-------------|
| **v3/@claude-flow/cli/src/commands/hooks.ts** | 4,530 | **72-78%** | 30 subcommands (not 17 documented). Genuine ADR-008 pre-task routing via enhanced-model-router.js. model-route/outcome/stats connect to model-router.js (tiny-dancer-neural). hooks-tools.ts semantic routing uses sin/cos hash (NOT ONNX). statusline vector count = file size heuristic. token-optimize hardcodes +200 saved tokens unconditionally. 4 v2 backward-compat aliases. R140. |

**R140 findings (model-routing scope):**

**HIGH** (+2):
25. **hooks.ts pre-task embeds real ADR-008 3-tier model routing** — enhanced-model-router.js integration (L1529-1576) provides legitimate [TASK_MODEL_RECOMMENDATION] directives with complexity scoring. This is the ONLY place in V3 where ADR-008 tier routing (Tier 1 Agent Booster / Tier 2 Haiku / Tier 3 Sonnet/Opus) is wired to a user-facing command. (R140, hooks.ts)
26. **hooks-tools.ts semantic routing uses sin/cos hash embeddings** — The route command's semantic embedding for pattern matching uses generateSimpleEmbedding() (sin/cos character hash) even when native VectorDb/HNSW is loaded. The HNSW index contains hash-embedded patterns, making semantic routing non-semantic end-to-end. Extends the hash-embedding pattern to the hooks routing layer. (R140, hooks-tools.ts)

**MEDIUM** (+1):
27. **token-optimize fabricates savings statistics** — hooks.ts unconditionally adds stats.totalTokensSaved += 200 and cacheHits = 2 per call regardless of actual optimization activity. Cumulative display is inflation, not measurement. (R140, hooks.ts)

## Section 8: Session Log

| Session | Date | Files | LOC | Key Result |
|---------|------|-------|-----|------------|
| R8 | 2026-01-xx | — | — | [ROUTING DIRECTIVE] pipeline confirmed |
| R11 | 2026-01-xx | 22 | ~8K | JS routing algorithms deep-read |
| R14 | 2026-01-xx | — | — | intelligence-bridge.js confirmed real (CRITICAL #4 resolved) |
| R17 | 2026-02-14 | 24 | ~5K | Domain closeout — 94.9% DEEP |
| R37 | 2026-02-xx | 4 | 5,494 | Rust routing deep-read — FIVE systems confirmed, hash embeddings CRITICAL |
| R104 | 2026-02-xx | — | — | claude_flow_bridge (5th surface), task_classifier (parallel surface) |
| R107 | 2026-02-18 | 3 | ~1,450 | **SEVEN routing surfaces fully mapped**. AgentRouter CRITICAL SONA mismatch. sheaf/router.rs 85-90% real but theta_deep dead. request.rs stop_sequences bug. Domain re-opened. |
| R114 | 2026-02-19 | 3 | ~1,108 | mcp-gate server.rs (90-93%) confirms **7th parallel MCP protocol**. SNN mod.rs (65-70%) tagged model-routing (RL reward signal wrong). prime-radiant error.rs (90-93%) maps to 5 ADRs including confidence routing. |
| R137 | 2026-03-01 | 3 | ~3,951 | **Ghost DEEP corrections + ML-C integration review**. hnsw_router.rs R37 CONFIRMED (90-93%). claude_integration.rs 68-73% setup toolkit (2 CRITICAL: mock workflow, missing ClaudeClient). model_router.rs 88-92% PARALLEL to TS ADR-008 (override maps dead). |
| R140 | 2026-03-02 | 1 | 4,530 | **hooks.ts CLI hooks command** — model-route/model-outcome/model-stats commands confirmed to connect to model-router.js (tiny-dancer-neural) via hooks-tools.ts lazy-loading. pre-task command integrates enhanced-model-router.js for real ADR-008 tier routing (Tier 1 Agent Booster / Tier 2 Haiku / Tier 3 Sonnet/Opus). hooks-tools.ts semantic routing uses sin/cos hash embeddings (NOT ONNX) even when native HNSW VectorDb is loaded. ADR-008 thresholds in hooks.ts (<0.3 haiku, 0.3-0.5 sonnet, >0.5 opus) parallel but do NOT bridge to model_router.rs Rust layer. |
| R134+R136+R140 | 2026-03-03 | 7 | ~17K | **Synthesis update**: Section 1 rewritten to include ML-A/ML-C/ML-F findings. File Registry extended with 7 new files. CRITICAL findings expanded to 11 (hash-based embedding systemic across ALL layers confirmed, ruvllm CLI non-functional without native binary, ROUTER/SERVER/CLUSTER CLI facades, hooks-tools.ts sin/cos hash routing). HIGH findings expanded to 34. MEDIUM findings extended to 24. Positives extended with 8 new entries. Coverage updated to ~96% DEEP (~92/96 files). |
