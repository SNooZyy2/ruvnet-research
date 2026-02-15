const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const sessionId = 11; // R11

const findFile = db.prepare('SELECT id FROM files WHERE package_id = (SELECT id FROM packages WHERE name = ?) AND relative_path = ?');
const insertRead = db.prepare('INSERT INTO file_reads (file_id, session_id, lines_read, depth) VALUES (?, ?, ?, ?)');
const updateFile = db.prepare('UPDATE files SET depth = ?, lines_read = lines_read + ?, last_read_date = ? WHERE id = ?');
const insertFinding = db.prepare('INSERT INTO findings (file_id, session_id, severity, category, description, line_start, line_end) VALUES (?, ?, ?, ?, ?, ?, ?)');

let matched = 0, findings = 0;

function rec(pkg, path, loc, depth, finds) {
  const row = findFile.get(pkg, path);
  if (!row) { console.log('  MISS:', pkg, path); return; }
  insertRead.run(row.id, sessionId, loc, depth);
  updateFile.run(depth, loc, '2026-02-14', row.id);
  matched++;
  for (const f of finds) {
    insertFinding.run(row.id, sessionId, f.sev, f.cat, f.desc, f.ls || null, f.le || null);
    findings++;
  }
}

const tx = db.transaction(() => {

  // === AGENT 1: ROUTING CORE (5 files) ===

  rec('claude-flow-cli', 'dist/src/commands/route.js', 813, 'DEEP', [
    { sev: 'MEDIUM', cat: 'quality', desc: 'route.js: CLI facade over Q-Learning router. No input validation on task descriptions (L93). Routes hard-coded to 8 agent types', ls: 18, le: 141 },
    { sev: 'MEDIUM', cat: 'incomplete', desc: 'route.js: Coverage routing claims ADR-017 support (L524) but coverage module optional; fallback returns empty gaps', ls: 521, le: 726 },
  ]);

  rec('claude-flow-cli', 'dist/src/ruvector/q-learning-router.js', 681, 'DEEP', [
    { sev: 'HIGH', cat: 'quality', desc: 'q-learning-router.js: REAL TD(0) Q-Learning with correct update rule (L339-359). Prioritized experience replay (L362-412). Three epsilon decay strategies (linear/exponential/cosine). MurmurHash3 for n-gram features (L609-628)', ls: 339, le: 628 },
    { sev: 'MEDIUM', cat: 'architecture', desc: 'q-learning-router.js: Tabular Q-Learning only, NOT deep RL. No gradient backpropagation despite comments. State space 64-dim with 10K max Q-table entries. avgTDError not normalized, can explode', ls: 325, le: 581 },
    { sev: 'MEDIUM', cat: 'quality', desc: 'q-learning-router.js: Feature extraction is deterministic keyword-based (32 binary features), not semantic. LRU cache TTL=300s only checked during exploit phase', ls: 543, le: 581 },
  ]);

  rec('claude-flow-cli', 'dist/src/ruvector/moe-router.js', 626, 'DEEP', [
    { sev: 'HIGH', cat: 'quality', desc: 'moe-router.js: REAL Mixture of Experts with 2-layer gating (384->128->8). Xavier init via Box-Muller (L64-74). Forward pass with ReLU and softmax (L240-296). REINFORCE gradients with backprop (L318-388)', ls: 64, le: 388 },
    { sev: 'HIGH', cat: 'fabricated', desc: 'moe-router.js: Load balance loss from Switch Transformer paper computed (L556-572) but NEVER backpropagated. Loss is purely informational, not used in gradient updates', ls: 556, le: 572 },
    { sev: 'MEDIUM', cat: 'architecture', desc: 'moe-router.js: REINFORCE gradients isolated per update, no momentum/batch accumulation. Expects 384-dim embedding input but source of embeddings not shown', ls: 313, le: 388 },
  ]);

  rec('claude-flow-cli', 'dist/src/ruvector/coverage-router.js', 529, 'DEEP', [
    { sev: 'HIGH', cat: 'fabricated', desc: 'coverage-router.js: Coverage parsing REAL (LCOV/Istanbul/Cobertura/JSON, L56-168). But coverageRoute/Suggest/Gaps return empty if no coverage file. Priority scoring is hardcoded regex guessing, not learned', ls: 56, le: 487 },
    { sev: 'MEDIUM', cat: 'quality', desc: 'coverage-router.js: Path traversal protection solid (L380-401). Cache TTL 60s with no invalidation on file update. useNative metric always false in practice', ls: 380, le: 460 },
  ]);

  rec('agentic-flow', 'dist/routing/CircuitBreakerRouter.js', 460, 'DEEP', [
    { sev: 'HIGH', cat: 'quality', desc: 'CircuitBreakerRouter.js: REAL circuit breaker state machine (CLOSED->OPEN->HALF_OPEN->CLOSED). Proper timer cleanup on destroy(). Rate limiter 100 req/min. Input validation (10KB max, timeout bounds)', ls: 22, le: 459 },
    { sev: 'MEDIUM', cat: 'architecture', desc: 'CircuitBreakerRouter.js: Uncertainty estimation heuristic (failure rate*0.5 + sample size*0.3 + recency*0.2) with arbitrary weights. avgRoutingTimeMs tracked but never used for routing decisions', ls: 398, le: 434 },
  ]);

  // === AGENT 2: PROXY/PROVIDER (7 files) ===

  rec('agentic-flow', 'dist/proxy/anthropic-to-requesty.js', 708, 'DEEP', [
    { sev: 'HIGH', cat: 'quality', desc: 'anthropic-to-requesty.js: REAL proxy to Requesty API. Live HTTP fetch (L144-153). 60s timeout. JSON Schema sanitization for OpenAI compat (L308-319). Max 10 tools limit (L453)', ls: 144, le: 659 },
    { sev: 'HIGH', cat: 'security', desc: 'anthropic-to-requesty.js: API key prefix (first 10 chars) leaked in debug logs (L141)', ls: 141, le: 141 },
  ]);

  rec('agentic-flow', 'dist/proxy/anthropic-to-openrouter.js', 619, 'DEEP', [
    { sev: 'HIGH', cat: 'quality', desc: 'anthropic-to-openrouter.js: REAL proxy to OpenRouter API. Similar to Requesty but simpler tool handling. No schema sanitization', ls: 141, le: 542 },
    { sev: 'HIGH', cat: 'bug', desc: 'anthropic-to-openrouter.js: NO TIMEOUT on OpenRouter API calls (L254) unlike Requesty 60s timeout. Requests can hang indefinitely', ls: 253, le: 268 },
    { sev: 'HIGH', cat: 'security', desc: 'anthropic-to-openrouter.js: API key prefix leaked in debug logs (L138)', ls: 138, le: 138 },
  ]);

  rec('agentic-flow', 'dist/proxy/anthropic-to-gemini.js', 446, 'DEEP', [
    { sev: 'CRITICAL', cat: 'security', desc: 'anthropic-to-gemini.js: API key exposed in query parameter ?key=... (L54). Visible in HTTP logs, URLs, referrer headers. Should use Authorization header instead', ls: 54, le: 54 },
    { sev: 'HIGH', cat: 'architecture', desc: 'anthropic-to-gemini.js: Tool instructions injected as XML comments in system prompt (L152-175). Relies on Gemini parsing XML from text. Structured command parsing via regex (L260-298)', ls: 152, le: 300 },
  ]);

  rec('agentic-flow', 'dist/proxy/websocket-proxy.js', 407, 'DEEP', [
    { sev: 'HIGH', cat: 'quality', desc: 'websocket-proxy.js: REAL WebSocket proxy with ping/pong heartbeat (30s interval), DoS protection (1000 connection limit), 5-min timeout. Uses ws library', ls: 35, le: 274 },
    { sev: 'MEDIUM', cat: 'security', desc: 'websocket-proxy.js: Gemini API key in query parameter (L178,250). Hardcoded model gemini-2.0-flash-exp. No request size limiting', ls: 178, le: 250 },
  ]);

  rec('agentic-flow', 'dist/proxy/http2-proxy.js', 382, 'DEEP', [
    { sev: 'HIGH', cat: 'quality', desc: 'http2-proxy.js: REAL HTTP/2 proxy with TLS cert validation (L24-51). Rate limiting per client IP. Request body max 1MB. h2c cleartext for dev', ls: 24, le: 240 },
    { sev: 'MEDIUM', cat: 'incomplete', desc: 'http2-proxy.js: Stream prioritization mentioned in comments but not implemented. Endpoint hardcoded to gemini-2.0-flash-exp (L171)', ls: 159, le: 227 },
  ]);

  rec('agentic-flow', 'dist/proxy/tool-emulation.js', 366, 'DEEP', [
    { sev: 'MEDIUM', cat: 'quality', desc: 'tool-emulation.js: REAL ReAct pattern (Thought/Action/Observation markers). Two strategies: ReAct and Prompt-Based. Max 5 iterations. Confidence base 0.5 + 0.3 valid tool + 0.1 reasoning', ls: 14, le: 364 },
    { sev: 'MEDIUM', cat: 'incomplete', desc: 'tool-emulation.js: Schema validation basic (type checking only, no constraints). No nested object/array validation', ls: 211, le: 251 },
  ]);

  rec('agentic-flow', 'dist/proxy/provider-instructions.js', 347, 'DEEP', [
    { sev: 'INFO', cat: 'quality', desc: 'provider-instructions.js: 8 provider instruction templates (Anthropic=native, others=XML tags). Model capability detection. File ops regex detection (13 patterns). Parallel execution instructions with ReasoningBank', ls: 1, le: 347 },
  ]);

  // === AGENT 3: ADVANCED ROUTERS (5 files) ===

  rec('agentic-flow', 'dist/routing/TinyDancerRouter.js', 407, 'DEEP', [
    { sev: 'HIGH', cat: 'quality', desc: 'TinyDancerRouter.js: Native @ruvector/tiny-dancer is REAL (compiled .node binary). FastGRNN neural routing with circuit breaker wrapper. JS fallback uses cosine similarity with softmax temperature scaling', ls: 29, le: 377 },
    { sev: 'MEDIUM', cat: 'architecture', desc: 'TinyDancerRouter.js: Adaptive weight learning (lr=0.1, bounds [0.1,10.0]). Batch routing delegates to native. Uncertainty from score variance. Metrics memory-only (last 100 routes)', ls: 195, le: 361 },
  ]);

  rec('agentic-flow', 'dist/routing/SemanticRouter.js', 290, 'DEEP', [
    { sev: 'CRITICAL', cat: 'fabricated', desc: 'SemanticRouter.js: Claims HNSW-powered semantic routing but implements brute-force cosine similarity (L206-214). Comments acknowledge: "For this implementation, we use brute-force cosine similarity"', ls: 76, le: 214 },
    { sev: 'MEDIUM', cat: 'incomplete', desc: 'SemanticRouter.js: Multi-intent detection via simple text splitting on sentences/conjunctions. Requires external embedder not provided. Execution order inferred from text position only', ls: 147, le: 279 },
  ]);

  rec('agentic-flow', 'dist/router/providers/onnx-local.js', 294, 'DEEP', [
    { sev: 'HIGH', cat: 'quality', desc: 'onnx-local.js: REAL local ONNX inference with onnxruntime-node. Phi-4-mini INT4 quantized (~4.9GB). Auto-download from HuggingFace. KV cache for 32-layer transformer. tiktoken cl100k_base tokenization', ls: 30, le: 217 },
    { sev: 'MEDIUM', cat: 'incomplete', desc: 'onnx-local.js: Greedy decoding only (not sampling). Streaming not implemented (throws error). Graph optimization enabled. Cost always 0 (local)', ls: 189, le: 260 },
  ]);

  rec('agentic-flow', 'dist/router/providers/onnx.js', 264, 'DEEP', [
    { sev: 'HIGH', cat: 'quality', desc: 'onnx.js: REAL @xenova/transformers wrapper for ONNX. Phi-3-mini-4k-instruct quantized. Top-p nucleus sampling (0.9). Platform-based CUDA/DirectML/CPU detection', ls: 52, le: 184 },
    { sev: 'MEDIUM', cat: 'fabricated', desc: 'onnx.js: Streaming is SIMULATED - generates full response then chunks by words with 10ms delays (L192,207-218). Token counting rough approximation (length/4)', ls: 188, le: 233 },
  ]);

  rec('agentic-flow', 'dist/router/providers/openrouter.js', 245, 'DEEP', [
    { sev: 'HIGH', cat: 'quality', desc: 'openrouter.js: REAL OpenRouter API client. Bearer auth (L20). Streaming via SSE (L42-74). Full tool use support. Model mapping via mapModelId()', ls: 20, le: 185 },
    { sev: 'MEDIUM', cat: 'fabricated', desc: 'openrouter.js: Cost hardcoded at $0.00001/input, $0.00003/output (L234-235). Actual OpenRouter pricing varies widely by model', ls: 229, le: 237 },
  ]);

  // === AGENT 4: REMAINING (4 files) ===

  rec('agentdb', 'dist/src/services/LLMRouter.js', 570, 'DEEP', [
    { sev: 'HIGH', cat: 'quality', desc: 'LLMRouter.js: Multi-provider abstraction (RuvLLM>OpenRouter>Gemini>Anthropic>ONNX). Real API calls for external providers. Optimization scoring based on priority levels (quality/balanced/cost/speed/privacy)', ls: 105, le: 529 },
    { sev: 'MEDIUM', cat: 'fabricated', desc: 'LLMRouter.js: RuvLLM integration speculative - dynamic import may not exist. Circular fallback (RuvLLM->RuvLLM->local). Local fallback uses hardcoded template responses for keywords', ls: 23, le: 495 },
  ]);

  rec('agentic-flow', 'dist/core/provider-manager.js', 435, 'DEEP', [
    { sev: 'HIGH', cat: 'quality', desc: 'provider-manager.js: REAL provider lifecycle. Circuit breaker (N consecutive failures). Exponential backoff (1-30s). Performance scoring = successRate*0.7 - normalizedLatency*0.3. Round-robin and cost optimization', ls: 20, le: 389 },
    { sev: 'HIGH', cat: 'incomplete', desc: 'provider-manager.js: Health check NOT IMPLEMENTED (L86 TODO comment). Cost calc assumes flat rate, no tiered pricing. No circuit breaker recovery test', ls: 67, le: 200 },
  ]);

  rec('agentic-flow', 'dist/mcp/fastmcp/tools/hooks/route.js', 267, 'DEEP', [
    { sev: 'CRITICAL', cat: 'fabricated', desc: 'route.js (MCP hook): RuVector intelligence facade (L54-96) references non-existent routeTaskIntelligent/findSimilarPatterns/getIntelligenceStats from intelligence-bridge.js', ls: 54, le: 96 },
    { sev: 'HIGH', cat: 'quality', desc: 'route.js (MCP hook): Q-learning fallback is REAL (L98-264). Epsilon-greedy (10%). File pattern scoring + keyword matching + memory similarity + error pattern lookup. 18 agent types', ls: 98, le: 264 },
    { sev: 'MEDIUM', cat: 'architecture', desc: 'route.js (MCP hook): Q-learning state oversimplified as string "edit:${ext}". Memory similarity uses hash-based embeddings, not real semantic embeddings', ls: 119, le: 185 },
  ]);

  rec('agentic-flow', 'dist/intelligence/agent-booster-enhanced.js', 1122, 'DEEP', [
    { sev: 'CRITICAL', cat: 'fabricated', desc: 'agent-booster-enhanced.js: Compression tier system COMPLETELY FABRICATED. TensorCompress (L31) not a real ruvector export. No actual compression happens; embeddings stored as-is (L236). Claims 87.5-96.9% savings but nothing compressed', ls: 193, le: 297 },
    { sev: 'HIGH', cat: 'fabricated', desc: 'agent-booster-enhanced.js: GNN differentiableSearch() (L554) calls non-existent function. Token optimization claim FALSE - this is a code edit pattern cache, not token optimizer. No WASM loaded despite name', ls: 548, le: 575 },
    { sev: 'HIGH', cat: 'quality', desc: 'agent-booster-enhanced.js: REAL exact cache matching (hash-based, L462-476). REAL fuzzy matching with cosine similarity (L481-527, threshold 0.85). REAL error pattern learning (L579-608). REAL persistence (L919-938)', ls: 301, le: 938 },
    { sev: 'MEDIUM', cat: 'quality', desc: 'agent-booster-enhanced.js: Access frequency formula (accessRatio*0.5 + recencyBoost*0.3 + successBoost*0.2). 24 pretrain patterns for common code edits. Co-edit graph tracks file paths. ONNX init may fail silently', ls: 202, le: 1000 },
  ]);

});

tx();
console.log('R11 import complete:', matched, 'files matched,', findings, 'findings inserted');
db.close();
