# Model Routing Domain Analysis

> **Priority**: HIGH | **Coverage**: 94.9% DEEP (56/59 files) | **Status**: CLOSED
> **Last updated**: 2026-02-14 (Session R17)

## Overview

Determines which LLM handles each task, routes tasks to agent types, and provides multi-provider API proxying. 59 files / 17K LOC. 31 files DEEP-read (11.9K LOC).

## The Big Picture

Model routing has **five subsystems**, each serving a different purpose:

| Subsystem | Components | Status | Evidence |
|-----------|-----------|--------|----------|
| **Hook-based Routing** | route-wrapper.sh, format-routing-directive.sh | **REAL** (advisory) | Outputs [ROUTING DIRECTIVE] text for Claude to read |
| **Q-Learning Router** | q-learning-router.js, route.js (CLI) | **REAL** | TD(0) algorithm, experience replay, MurmurHash3 features |
| **MoE Router** | moe-router.js | **REAL** | 2-layer gating network, REINFORCE gradients, Xavier init |
| **API Proxy Layer** | 7 proxy files (Requesty, OpenRouter, Gemini, WS, H2) | **REAL** | Live API calls, SSE streaming, format conversion |
| **Intelligence Layer** | agent-booster-enhanced.js, SemanticRouter, route.js (MCP hook) | **FABRICATED** | Non-existent functions, fake compression, brute-force claims HNSW |

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

## CRITICAL Findings (5)

1. **Gemini API key in query parameter** — Exposed in HTTP logs, URLs, referrer headers across 3 files (gemini proxy, websocket proxy, http2 proxy).
2. **SemanticRouter HNSW is fabricated** — Claims HNSW-powered routing but code implements brute-force cosine similarity. Comments acknowledge this.
3. **agent-booster compression fabricated** — TensorCompress not real. Claims 87.5-96.9% savings but embeddings stored uncompressed. GNN search calls non-existent function.
4. ~~**RuVector intelligence facade**~~ **RESOLVED R14** — intelligence-bridge.js EXISTS (1,038 LOC). routeTaskIntelligent() at L382 and findSimilarPatterns() at L542 both confirmed working. Research synchronization error, not code deficiency.
5. **Three fragmented ReasoningBanks** — Learning system informing decisions broken across 3 packages.

## HIGH Findings (12)

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

## MEDIUM Findings (11)

1. route.js CLI: No input validation on task descriptions.
2. q-learning-router: Tabular only, not deep RL. State space can explode.
3. moe-router: Gradients isolated per update, no momentum.
4. coverage-router: Cache TTL 60s with no invalidation.
5. CircuitBreakerRouter: Uncertainty weights arbitrary (0.5/0.3/0.2).
6. websocket-proxy: Gemini API key in URL, hardcoded model.
7. onnx-local: Greedy decoding only, streaming throws error.
8. onnx.js: Streaming simulated (word chunking + 10ms delays). Token count = length/4.
9. openrouter.js: Cost hardcoded at $0.00001/$0.00003, inaccurate.
10. LLMRouter: Circular fallback (RuvLLM→RuvLLM→local).
11. route.js MCP hook: Q-learning state oversimplified as `edit:${ext}`.

## Positive

- **Q-Learning and MoE routers** implement real, correct RL/ML algorithms
- **CircuitBreakerRouter** is a solid, well-implemented fault tolerance pattern
- **API proxy layer** is production-quality with real API integrations
- **ONNX local inference** genuinely works with Phi-4/Phi-3 models
- **TinyDancerRouter** has real compiled native binary (FastGRNN in Rust)
- **provider-manager** has real exponential backoff, circuit breaking, and cost optimization
- **agent-booster** has genuine pattern caching, fuzzy matching, and persistence

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
| **hnsw_router.rs** | 1,288 | **90-93%** | **BEST ruvector-core integration in project**. HybridRouter blends HNSW semantic + keyword routing with confidence weighting. Real HnswIndex with M/ef config, batch adds, genuine search. Pattern consolidation merges similar patterns by agent_type. |
| **model_router.rs** | 1,292 | **88-92%** | 7-factor complexity analyzer (code length, test presence, multi-file, security keywords, etc.), model selector with cost/latency constraints. 45 routing patterns. record_feedback tracks last 1000 predictions with accuracy stats. |
| **pretrain_pipeline.rs** | 1,394 | **85-88%** | Multi-phase pretraining: Bootstrap → Synthetic → Reinforce → Consolidate. Curriculum learning with difficulty progression. **CRITICAL**: generate_embedding is HASH-BASED (character sum % dim). Quality scores simulated with rand_simple(). |
| **reasoning_bank.rs** | 1,520 | **92-95%** | Production ReasoningBank: real K-means clustering (10 iterations, convergence check), EWC++ consolidation, pattern distillation. 16 tests. Informs routing decisions via pattern retrieval. |

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

### Updated Assessment

The model-routing domain now has **FIVE routing systems** (adding Rust layer to the four identified in R11/R17):

| System | Package | Language | Status |
|--------|---------|----------|--------|
| Hook-based Routing | claude-flow-cli | Shell/JS | **Active** (advisory) |
| Q-Learning/MoE | claude-flow-cli | JS | **Active** (CLI-only) |
| API Proxy Layer | agentic-flow | JS | **Active** (production) |
| LLMRouter | agentdb | JS/TS | **Available but unused** |
| **HNSW/Complexity Router** | **ruvllm** | **Rust** | **Available but unused** |

### R37 Findings

**CRITICAL** (+1):
6. **Hash-based embeddings in Rust routing** — pretrain_pipeline.rs generate_embedding uses character sum % dim. All HNSW routing patterns stored with fake embeddings, making semantic search non-semantic. Same pattern as ruvector-core. (R37)

**HIGH** (+2):
13. **Rust routing is BEST but unused** — hnsw_router.rs at 90-93% real is the most sophisticated routing in the ecosystem, but claude-flow uses JS Q-learning instead. (R37)
14. **Four distinct ReasoningBanks** — Rust reasoning_bank.rs is the fourth independent implementation with zero code sharing. (R37)

**Positive** (+3):
- **hnsw_router.rs** has real HNSW with M/ef configuration and pattern consolidation — production-quality semantic routing (R37)
- **model_router.rs** has genuine 7-factor complexity analysis with feedback tracking (R37)
- **reasoning_bank.rs** has real K-means + EWC++ — best mathematical foundation of all 4 ReasoningBank implementations (R37)
