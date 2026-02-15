// R16 AgentDB Deep-Read Import
// Session 18 (R16): 25 files, ~60 findings
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const sessionId = 18;
const today = '2026-02-14';

const insertRead = db.prepare('INSERT INTO file_reads (file_id, session_id, lines_read, depth) VALUES (?, ?, ?, ?)');
const updateFile = db.prepare('UPDATE files SET depth = ?, lines_read = lines_read + ?, last_read_date = ? WHERE id = ?');
const insertFinding = db.prepare('INSERT INTO findings (file_id, session_id, severity, category, description, line_start, line_end) VALUES (?, ?, ?, ?, ?, ?, ?)');

let filesOk = 0, findingsOk = 0;

function rec(fileId, linesRead, depth, findings) {
  insertRead.run(fileId, sessionId, linesRead, depth);
  updateFile.run(depth, linesRead, today, fileId);
  filesOk++;
  for (const f of findings) {
    insertFinding.run(fileId, sessionId, f.sev, f.cat, f.desc, f.ls || null, f.le || null);
    findingsOk++;
  }
}

db.transaction(() => {
  // ============================================================
  // AGENT 1: CLI & MCP Surface (agentdb, pkg 2)
  // ============================================================

  // agentdb-cli.ts (3,422 LOC) — 98% REAL
  rec(350, 3422, 'DEEP', [
    { sev: 'HIGH', cat: 'architecture', desc: 'AgentDB CLI: 35+ subcommands across 8 command groups. 60+ public methods on AgentDBCLI class. Groups: setup (init/status/doctor/migrate), vector-search, import/export, causal (add-edge/experiment/query), recall (with-certificate), learner (run/prune), reflexion (store/retrieve/critique/prune), skills (create/search/consolidate/prune), QUIC sync (start-server/connect/push/pull). All fully implemented with real database operations.', ls: 126, le: 3421 },
    { sev: 'HIGH', cat: 'architecture', desc: 'Controller initialization chain at L199-219: CausalMemoryGraph, CausalRecall, ExplainableRecall, NightlyLearner, ReflexionMemory, SkillLibrary, QUICServer/Client, SyncCoordinator. Database uses better-sqlite3 with WAL mode, synchronous=NORMAL, 64MB cache (L146-148).', ls: 199, le: 219 },
    { sev: 'MEDIUM', cat: 'architecture', desc: 'CLI spawns MCP server as subprocess (L2108-2111). Both CLI and MCP server initialize identical controllers independently. Schema loading has multiple fallback paths (L152-159), warns if not found.', ls: 2108, le: 2111 },
    { sev: 'INFO', cat: 'quality', desc: 'Simulate command uses dynamic import with fallback (L1933-1937). Session ID uses "experiment-placeholder" for all experiments — collision risk.', ls: 1933, le: 1937 },
  ]);

  // agentdb-mcp-server.ts (2,368 LOC) — 85% REAL
  rec(420, 2368, 'DEEP', [
    { sev: 'HIGH', cat: 'architecture', desc: 'AgentDB MCP Server registers 34 tools in 6 groups: Core Vector DB (5: init/insert/insert_batch/search/delete), Frontier Memory (8: reflexion_store/retrieve, skill_create/search, causal_add_edge/query, recall_with_certificate, learner_discover, db_stats), Learning System (10: start/end_session, predict, feedback, train, metrics, transfer, explain, experience_record, reward_signal), AgentDB Core (5: stats/pattern_store/search/stats/clear_cache), Batch Ops (3: skill/reflexion/pattern batch), Attention (4: compute/benchmark/configure/metrics).', ls: 295, le: 883 },
    { sev: 'HIGH', cat: 'architecture', desc: 'MCP uses @modelcontextprotocol/sdk/server with StdioServerTransport. All tool handlers include input validation. Error handling wraps SecurityError. Database transaction support for batch operations.', ls: 8, le: 275 },
    { sev: 'MEDIUM', cat: 'fabrication', desc: 'Attention benchmark tool generates random test data. Attention metrics fabricated (Math.random for totalCalls, latencies, memory). Attention config not persisted — returns defaults without saving.', ls: 900, le: 1150 },
  ]);

  // attention-tools-handlers.ts (587 LOC) — 40% REAL
  rec(422, 587, 'DEEP', [
    { sev: 'CRITICAL', cat: 'fabrication', desc: 'Attention metrics handler at L293-299: totalCalls=Math.floor(Math.random()*10000)+1000, avgLatency=Math.random()*10+1, avgMemory=Math.random()*50+10, successRate=0.95+Math.random()*0.05, cacheHitRate=0.6+Math.random()*0.3. 100% fabricated metrics returned to MCP consumers.', ls: 293, le: 299 },
    { sev: 'HIGH', cat: 'fabrication', desc: 'Attention benchmark tool generates random test keys and times iterations. Speedup comparison is synthetic. Query encoding at L338 uses charCode hashing instead of real embeddings. Config not persisted (L247-257).', ls: 81, le: 338 },
    { sev: 'INFO', cat: 'architecture', desc: 'Supports 5 attention mechanisms: flash, hyperbolic, sparse, linear, performer. dotProductMCP() and poincareDistanceMCP() helper functions are mathematically correct (L412-426).', ls: 348, le: 426 },
  ]);

  // learning-tools-handlers.ts (107 LOC) — 95% REAL
  rec(423, 107, 'DEEP', [
    { sev: 'INFO', cat: 'architecture', desc: 'Learning metrics handler (L7-92): queries episodes from database, calculates success_rate/avg_reward/avg_latency from actual data. Supports time windows and grouping by task/session/skill. 7-day trend analysis. Version 1.4.0.', ls: 7, le: 106 },
  ]);

  // attention-mcp-integration.ts (146 LOC) — REAL
  rec(421, 146, 'DEEP', [
    { sev: 'INFO', cat: 'architecture', desc: 'MCP integration glue: registers attention tools with MCP server. Delegates to attention-tools-handlers.ts. Small but functional bridge.', ls: 1, le: 146 },
  ]);

  // ============================================================
  // AGENT 2: Core Controllers (agentdb, pkg 2)
  // ============================================================

  // ReasoningBank.ts (676 LOC) — 98% REAL
  rec(396, 676, 'DEEP', [
    { sev: 'HIGH', cat: 'architecture', desc: 'CANONICAL ReasoningBank implementation. v1/v2 dual-mode: v1 stores embeddings in SQLite, v2 delegates to VectorBackend. Cosine similarity at L657-674 is standard correct implementation. SQLite schema with proper indices (L128-155). Search pipeline: Query → Embed → VectorBackend.search() → Hydrate from SQLite → Return ReasoningPattern[]. GNN enhancement optional via learningBackend.enhance() interface delegation.', ls: 101, le: 674 },
    { sev: 'HIGH', cat: 'architecture', desc: 'This is the SAME CLASS used in three contexts across AgentDB: direct usage, via NightlyLearner for pattern storage, and via AgentDB factory for unified initialization. The "three separate ReasoningBanks" finding from SYNTHESIS.md refers to three PACKAGES (claude-flow, agentic-flow, agentdb) with separate implementations — this is the agentdb one.', ls: 1, le: 676 },
    { sev: 'MEDIUM', cat: 'architecture', desc: 'Pattern stats (L451-512) use simplistic thresholds: "high performing" = success_rate >= 0.8. GNN enhancement is interface-only — actual GNN implementation is external.', ls: 451, le: 512 },
  ]);

  // HNSWIndex.ts (582 LOC) — 96% REAL
  rec(388, 582, 'DEEP', [
    { sev: 'HIGH', cat: 'architecture', desc: 'Real HNSW via hnswlib-node C++ library wrapper. L213: this.index = new HierarchicalNSW(...) with M and efConstruction parameters. Label mapping (L129-131, L239-244) maps pattern IDs to HNSW integer labels. Distance-to-similarity conversion (L473-492) correct: cosine=1-distance, L2=exp(-distance), IP=-distance.', ls: 143, le: 492 },
    { sev: 'MEDIUM', cat: 'architecture', desc: 'Delete is STUB: L379 comment says "hnswlib doesn\'t support deletion". Code tracks updates but rebuild logic (L354-362) logs warning without actually rebuilding. Persistent index (L403-430) depends on hnswlib-node having writeIndex/readIndex methods.', ls: 354, le: 430 },
  ]);

  // SkillLibrary.ts (925 LOC) — 90% REAL
  rec(398, 925, 'DEEP', [
    { sev: 'HIGH', cat: 'architecture', desc: 'Skill CRUD (L80-176), search via VectorBackend (L181-308), cache via QueryCache (L194-204), skill links/relationships (L399-456). Composite scoring: similarity*0.4 + success_rate*0.3 + (uses/1000)*0.1 + avg_reward*0.2 (L912-923). Weights hardcoded, not learned.', ls: 80, le: 923 },
    { sev: 'MEDIUM', cat: 'fabrication', desc: 'Pattern extraction (L590-661) claims "ML-inspired analysis" but is basic TF word counting with stop words. Learning trend analysis at L654-658 uses hardcoded 10% improvement threshold. Confidence at L808-814: arbitrary sampleSize/10 saturation. NOT machine learning.', ls: 590, le: 661 },
  ]);

  // ExplainableRecall.ts (747 LOC) — 85% REAL
  rec(387, 747, 'DEEP', [
    { sev: 'HIGH', cat: 'architecture', desc: 'Genuine algorithms: Greedy minimal hitting set for set cover (L498-541). Merkle tree provenance with SHA-256 (L638-662). Certificate creation (L130-212) and verification (L218-266) with hash checking. Justification paths with necessity scoring (L700-732). All mathematically correct.', ls: 130, le: 732 },
    { sev: 'MEDIUM', cat: 'architecture', desc: 'GraphRoPE feature flag (L31-34, L113-121) advertised but not implemented — delegates to AttentionService. Content hash calculated once (L151) but never re-verified on change. Labeled "v2.0.0-alpha.3 Features" but features are aspirational.', ls: 31, le: 121 },
  ]);

  // NightlyLearner.ts (665 LOC) — 80% REAL
  rec(393, 665, 'DEEP', [
    { sev: 'HIGH', cat: 'architecture', desc: 'Doubly-robust causal inference formula quoted correctly (L165-173). Implementation at L385: (mu1-mu0) + (a*(y-mu1)/propensity). Propensity score via frequency estimation (L422-438) with clipping [0.01, 0.99]. A/B experiment tracking (L488-566) with real lifecycle management.', ls: 165, le: 566 },
    { sev: 'HIGH', cat: 'fabrication', desc: 'DR formula INCOMPLETE — missing second adjustment term for control group. FlashAttention consolidation (L204-320) calls this.attentionService.flashAttention() but actual Flash Attention is external. Causal discovery at L263-312 computes similarities and calls causalGraph.addCausalEdge() without proper temporal ordering (L289). Confidence at L475-483 uses arbitrary heuristics (sampleSize/100, effectSize/0.5).', ls: 204, le: 483 },
  ]);

  // MemoryController.ts (462 LOC) — 95% REAL
  rec(391, 462, 'DEEP', [
    { sev: 'HIGH', cat: 'architecture', desc: 'Memory orchestration wrapper: CRUD operations (L130-217), attention-enhanced retrieval (L275-359) combining self-attention + multi-head attention. Temporal decay weighting (L247-251): exp(-weight*age/(24*60*60*1000)). Score combination: 0.5*base + 0.5*(attention/2). Delegates to SelfAttentionController, CrossAttentionController, MultiHeadAttentionController.', ls: 130, le: 383 },
    { sev: 'INFO', cat: 'quality', desc: 'Cosine similarity at L364-383 is standard correct implementation. No fabricated metrics. All major operations are real. Weights (0.5/0.5) are arbitrary but functional.', ls: 364, le: 383 },
  ]);

  // ============================================================
  // AGENT 3: Search & Optimization (agentdb, pkg 2)
  // ============================================================

  // HybridSearch.ts (1,062 LOC) — 95% REAL
  rec(439, 1062, 'DEEP', [
    { sev: 'HIGH', cat: 'architecture', desc: 'Production-grade hybrid search. BM25 ranking (L316-356): IDF formula correct, k1=1.2, b=0.75 standard defaults, proper document length normalization. Three fusion strategies: RRF (L677-759) with 1/(k+rank), Linear (L767-838) with weighted α*vector + β*keyword, Max (L846-912) element-wise maximum. BM25 complexity O(n*m). Tokenization with stopword filtering.', ls: 316, le: 912 },
    { sev: 'INFO', cat: 'quality', desc: 'All three fusion strategies are mathematically sound. Score normalization to [0,1] correct. This is genuine production search code.', ls: 677, le: 912 },
  ]);

  // Quantization.ts (996 LOC) — 98% REAL
  rec(431, 996, 'DEEP', [
    { sev: 'HIGH', cat: 'architecture', desc: 'Real vector quantization. Scalar 8-bit (L95-210): min/max normalization to [0,255], correct dequantization formula. Product Quantization with K-means++ init (L356-458): first centroid random, remaining probability proportional to D(x)², standard k-means iteration with convergence tracking. Asymmetric Distance Computation (L804-853): precomputes d(query_sub, centroid) for O(1) lookup per subvector.', ls: 95, le: 853 },
    { sev: 'MEDIUM', cat: 'architecture', desc: 'L807 references codebooks field without null guard in some paths — potential runtime error. Overall K-means++ and ADC are mathematically correct and production-quality.', ls: 804, le: 853 },
  ]);

  // BatchOperations.ts (809 LOC) — 92% REAL
  rec(430, 809, 'DEEP', [
    { sev: 'HIGH', cat: 'architecture', desc: 'Real batch operations: transaction management (L78-111), parallel batch insert with concurrency control (L293-427), SQL injection prevention via parameterized queries (L500-555). Performance claims (L269-273) cite 4.5-5.5x speedup on parallel inserts — plausible estimates but not benchmark-verified.', ls: 78, le: 555 },
  ]);

  // WASMVectorSearch.ts (458 LOC) — 70% REAL
  rec(400, 458, 'DEEP', [
    { sev: 'HIGH', cat: 'fabrication', desc: 'WASM module "reasoningbank_wasm.js" does NOT exist. getWASMSearchPaths() lists 9 search paths (L111-163) but module is never found. initializeWASM() (L188-222) logs fallback to JS. SIMD detection code (L238-258) is real but WASM never loads.', ls: 111, le: 258 },
    { sev: 'HIGH', cat: 'architecture', desc: 'JS fallback is genuine: cosine similarity (L263-293) with loop unrolling (4 elements per iteration) is correct optimization. Honest about fallback behavior but WASM acceleration claims are misleading.', ls: 263, le: 293 },
  ]);

  // CausalRecall.ts (506 LOC) — 75% REAL
  rec(383, 506, 'DEEP', [
    { sev: 'HIGH', cat: 'architecture', desc: 'Reranking formula at L284: U = 0.7*similarity + 0.2*uplift - 0.1*latencyCost. Mathematically sound with proper bounds (similarity in [0,1], latencyCost = min(ms/1000, 1.0)).', ls: 261, le: 308 },
    { sev: 'HIGH', cat: 'fabrication', desc: 'Depends on CausalMemoryGraph which has FABRICATED statistics: hardcoded tInverse=1.96, wrong t-distribution CDF. Causal edges loaded at L235-250 carry broken confidence values. The reranking logic is sound but the causal data feeding it is not.', ls: 235, le: 250 },
  ]);

  // BenchmarkSuite.ts (1,361 LOC) — 85% REAL
  rec(341, 1361, 'DEEP', [
    { sev: 'HIGH', cat: 'architecture', desc: 'Real benchmarking framework: VectorInsertBenchmark (L283-413) generates random vectors, measures performance.now(), calculates ops/sec. Latency statistics (L226-249) with correct percentile calculation: sorted[(p/100)*n-1] for P50/P95/P99.', ls: 226, le: 413 },
    { sev: 'HIGH', cat: 'fabrication', desc: 'QuantizationBenchmark (L765-950) would CRASH at runtime: L809 creates QuantizedVectorStore with wrong field names (quantizationType instead of correct field, metric not in interface). Interface mismatch means benchmark produces fabricated results via exception handling.', ls: 765, le: 950 },
  ]);

  // ============================================================
  // AGENT 4: Security & Infrastructure (agentdb, pkg 2)
  // ============================================================

  // validation.ts (557 LOC) — 95% REAL
  rec(444, 557, 'DEEP', [
    { sev: 'HIGH', cat: 'security', desc: 'Comprehensive security validation: NaN/Infinity prevention (L81-87), path traversal blocking (L130-155), control character filtering (L139-145), 13 regex patterns for sensitive field detection (L349-364: password, secret, token, api_key, credit_card, cvv, ssn). SECURITY_LIMITS const (L17-39) with 21 limits (10M vector limit, 30s timeout). Metadata size limit 64KB (L367-374). Cypher injection prevention (L148-155). safeLog() removes vectors.', ls: 17, le: 556 },
  ]);

  // input-validation.ts (544 LOC) — 98% REAL
  rec(441, 544, 'DEEP', [
    { sev: 'HIGH', cat: 'security', desc: 'Whitelist-based SQL injection prevention: ALLOWED_TABLES (13 tables, L14-29), ALLOWED_COLUMNS per-table (L34-48), ALLOWED_PRAGMAS (11 pragmas, L53-64). Table/column names validated via Set lookup O(1). PRAGMA parsing prevents injection. parameterized SQL WHERE/SET clause builders (L449-500). Task string validation blocks <script>, javascript:, onload=, null bytes (L91-125). Custom ValidationError with safe messages (L69-86).', ls: 14, le: 500 },
  ]);

  // auth.service.ts (668 LOC) — 92% REAL
  rec(448, 668, 'DEEP', [
    { sev: 'HIGH', cat: 'security', desc: 'Argon2id password hashing (L181, L621). Brute force protection: 5 failed attempts → 15min lockout (L268-290). Returns "Invalid credentials" for both wrong user and wrong password (L240) — prevents username enumeration. API key management: hashed before storage (L411), returned once only (L427). Environment tags (live/test) for key separation. Token rotation (L540-568).', ls: 134, le: 568 },
    { sev: 'MEDIUM', cat: 'architecture', desc: 'IN-MEMORY storage for users, API keys, sessions (L118-129) with comments "for production, use a database". No IP-based rate limiting. No password complexity requirements beyond 8-char minimum (L172). Prototype-quality auth suitable for single-instance.', ls: 117, le: 129 },
  ]);

  // token.service.ts (492 LOC) — 96% REAL
  rec(451, 492, 'DEEP', [
    { sev: 'HIGH', cat: 'security', desc: 'JWT HS256 implementation via jsonwebtoken npm package. Access token 15min, refresh 7 days (L26-27). JWT_SECRET requires minimum 32 chars (L94), errors if not set (L88-91). Token verification: checks revocation list first (L192), validates algorithm/issuer/audience (L203-205), type verification prevents using refresh as access (L209). JTI field for unique token identification (L130). Session-to-token mapping for bulk revocation (L309-331).', ls: 25, le: 452 },
    { sev: 'MEDIUM', cat: 'architecture', desc: 'Revocation list in-memory (L68) with auto-cleanup after 7 days via setTimeout (L301-303). Comment at L66: "for production, use Redis". Suitable for single-instance.', ls: 66, le: 303 },
  ]);

  // telemetry.ts (545 LOC) — 85% REAL
  rec(429, 545, 'DEEP', [
    { sev: 'HIGH', cat: 'architecture', desc: 'OpenTelemetry integration: TelemetryManager singleton (L78-338). Dynamic imports of @opentelemetry packages with graceful degradation (L125-129). Metric instruments: histogram (query latency), counters (cache hits/misses/errors/ops/throughput), gauge (cache hit rate %). @traced decorator for automatic span wrapping (L396-475). Real metric recording logic.', ls: 78, le: 475 },
    { sev: 'MEDIUM', cat: 'fabrication', desc: 'OTel SDK initialization is stubbed: instrumentations array is EMPTY (L138-141). No OTLP exporter actually connected. Cache stats are manually tracked (L96-98) rather than connected to actual cache. Version hardcoded as "2.0.0-alpha" (L62). Framework exists but not wired to production observability.', ls: 62, le: 157 },
  ]);

  // MultiHeadAttentionController.ts (494 LOC) — 98% REAL
  rec(402, 494, 'DEEP', [
    { sev: 'HIGH', cat: 'architecture', desc: 'GENUINE multi-head attention implementation. Xavier initialization (L114-118): scale=sqrt(2/(fan_in+fan_out)). Scaled dot-product attention (L303-310): score = dotProduct * (1/sqrt(d_k)) / temperature. Numerically stable softmax (L345-370): subtracts max before exp, handles NaN/Infinity. Multi-head aggregation (L375-446): 4 strategies (average, max, concat, weighted). Per-head reconstruction from subspace. Complexity O(H*M*D). NOT a facade — real neural attention.', ls: 103, le: 446 },
    { sev: 'MEDIUM', cat: 'architecture', desc: 'Projection matrices are random (not learned) — no gradient tracking for training. Head output reconstruction simplified (L451-468). Implements inference-only attention, not trainable.', ls: 114, le: 468 },
  ]);

  // CrossAttentionController.ts (467 LOC) — 98% REAL
  rec(401, 467, 'DEEP', [
    { sev: 'HIGH', cat: 'architecture', desc: 'GENUINE cross-attention implementation. Scaled dot-product with temperature (L197-210). Numerically stable softmax (L332-361). Attended output via weighted sum (L366-385). Multi-context aggregation (L246-315): computes attention per context independently then aggregates. Context strategies: average, max, weighted. Namespace support via contextStores Map (L74). No external neural libraries — implements attention from scratch. NOT a facade.', ls: 165, le: 430 },
  ]);

})();

console.log(`R16 import complete: ${filesOk} files recorded, ${findingsOk} findings inserted`);
db.close();
