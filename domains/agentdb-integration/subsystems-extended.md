# AgentDB Integration — Section 5b: Subsystems (Extended)

> Part of [AgentDB Integration Domain Analysis](analysis.md). See index for full document map.

### 5g. R18 Native vs Patched Architecture

R18 deep-read revealed the root cause of broken claude-flow integration:

**Native architecture** (FUNCTIONAL):
```
agentdb-mcp-server.ts  ← #!/usr/bin/env node (STANDALONE process)
  ├── EmbeddingService(Xenova/all-MiniLM-L6-v2, 384d)  ← INITIALIZED at startup (L219-224)
  ├── ReflexionMemory(db, embedder, ...)
  ├── SkillLibrary(db, embedder)
  └── All 27+ tools with proper embeddings
```

**Patched architecture** (BROKEN):
```
claude-flow mcp-server.js
  ├── agentdb-tools.js (598 LOC)  ← bridges via createRequire()
  │     └── agentdb-service-fallback  ← silent degradation path
  └── 6 tools exposed (vs 27 native)
      → NO EmbeddingService initialized
      → Episodes stored WITHOUT embeddings
      → episode_embeddings stays EMPTY
      → SQL JOIN returns zero rows
      → Search returns empty array
```

Tool coverage gap: User exposes 6/27 native tools, 2 work (stats), 3 are broken (search/retrieve/suggest), 1 is custom (R18).

**agentic-flow resolution** (R32): agentdb-wrapper-enhanced.ts properly chains reflexionController→embedder→vectorBackend. EnhancedAgentDBWrapper fixes R18 for agentic-flow users (H24).

### 5h. Infrastructure & Telemetry

**telemetry.ts** (545 LOC, 85%, R16): OpenTelemetry integration with proper metric instruments (histogram, counters, gauge), @traced decorator. BUT: SDK initialization is stubbed (empty instrumentations array), no OTLP exporter connected (H16).

**BenchmarkSuite** has two implementations:
- **BenchmarkSuite.ts** (1,361 LOC, 95%, R16, R22): Best-quality file in AgentDB. Complete framework with percentile latency, 5% regression threshold. Quantization benchmark would crash due to interface mismatch at L809 (C7).
- **BenchmarkSuite.js** (984 LOC, 100%, R32): Compiled version with performance.now ×28, zero fake benchmarks across all 5 classes (H22).

### 5j. RuVectorBackend — Entry Point & Implementation (R88)

**src/backends/ruvector/index.ts** (10 LOC) is a minimal barrel re-export: `export { RuVectorBackend, RuVectorLearning } from './RuVectorBackend'`. It is the public entry point for the ruvector backend package within AgentDB.

**RuVectorBackend.ts** (971 LOC, **72-78% — REVISED DOWN from 88-92% in R88**) is a pure adapter: zero own HNSW code, all vector logic delegated to external npm `ruvector`/`@ruvector/core`. R91 deep-read reveals both genuine strengths and significant implementation gaps:

**Genuine strengths:**
- **Dynamic import with graceful degradation**: Tries `import('ruvector')` first, then falls back to `import('@ruvector/core')`. Runtime optional dependency (H43). No user warning on failure (H46).
- **Semaphore (FIFO promise queue)**: Instance-level concurrency control for batch insert ops.
- **BufferPool (size-keyed pools, zeroed)**: Float32Array reuse to reduce GC pressure.
- **validatePath + FORBIDDEN_PATH_PATTERNS**: Blocks `/etc`, `/proc`, `/sys`, `/dev`, traversal sequences. `MAX_METADATA_ENTRIES=10M`, `MAX_VECTOR_DIMENSION=4096` guard resource limits.
- **Prototype pollution defense**: `Object.keys` guard on metadata ingestion.
- **R20 exoneration (R88, H44)**: Accepts `Float32Array`, performs genuine HNSW insert/search/remove via VectorDB. Failure is upstream (EmbeddingService).

**Implementation gaps discovered in R91:**
- **`updateAdaptiveParams()` only adjusts `efSearch`** (H51): Despite incrementing `indexRebuildCount`, M and efConstruction are never changed and the index is never rebuilt. Adaptive tuning is partial.
- **`insertBatchParallel()` creates local semaphore** (H52): Bypasses instance-level semaphore entirely. Concurrent batch insert calls can exceed intended concurrency limits.
- **mmap scaffolded but not wired to hot paths**: Constructor tries mmap-io import and stores the handle, but read/write paths do not use it.
- **L2 similarity uncalibrated** (H53): `Math.exp(-distance)` maps L2 to (0,1] without normalization. Scores not comparable to cosine similarity or across datasets.

The compiled `.js` (R50, 88-92%) holds up better than the `.ts` source, since the `.js` assessment focused on VectorDB integration correctness rather than adaptive parameter completeness.

### 5k. AIDefence Security Guard — Standalone Module (R92)

**AIDefenceGuard.ts** (763 LOC, 82-88%) is a self-contained rule-based security module in the `ruvbot` package. It is NOT integrated with AgentDB at runtime despite being architecturally adjacent.

**What is genuine:**
- **28 INJECTION_PATTERNS** covering direct override, role manipulation, system prompt extraction, jailbreak, code injection, data exfiltration — hand-crafted but well-designed.
- **PII detection** via named-capture regexes for email, phone, SSN, credit card, IP address, and API key patterns.
- **HOMOGLYPH_MAP** with 8 Cyrillic-to-ASCII substitutions — real unicode normalization attack prevention.
- **Input sanitization**: control character removal, homoglyph normalization, length limiting, null byte stripping.
- **Middleware factory pattern** (`createAIDefenceMiddleware`): clean validateInput + validateOutput pipeline matching real dual-validation production patterns.
- **Config factory helpers**: `createStrictConfig()` (blockThreshold=low) and `createPermissiveConfig()` (blockThreshold=critical) are correct and useful.

**What is missing or broken:**
- **`aidefence` npm dependency (^2.1.1) is never imported** (H56, INFO finding 9316) — the entire implementation is self-contained. The npm package is dead weight in package.json.
- **`enablePolicyVerification` flag has NO implementation** (H55): zero policy verification code in the file; setting it true has no runtime effect.
- **`behaviorBaseline` is stateless across sessions** (H54): in-memory `Map` with no persistence. `analyzeBehavior()` uses only 4 naive statistical features (length, punctuation count, caps ratio, digit ratio) with a hardcoded deviation threshold of 2.0 — no ML model, no trained classifier, easily gameable.
- **PII regex `lastIndex` inconsistency** (MEDIUM 9310): `detectPII()` resets `pattern.lastIndex` before each call, but `sanitize()` does not — risks missed matches on repeated calls with the same `/g` flag patterns.
- **AuditLog is in-memory only** (INFO 9318): capped at 1000 entries via `slice(-1000)`, survives only for process lifetime.

**Integration gap with AgentDB:**
The `aidefence-integration.ts` simulation scenario (166 LOC, 25%) documents the *intent* to wire AIDefence into AgentDB but is not a runtime integration:
- All 5 threat patterns (sql_injection, xss_attack, etc.) are hardcoded with fixed severity scores 0.88-0.98, not detected from real system activity.
- Causal links between threats and defenses are commented out (H57).
- `CausalMemoryGraph` receives `graph DB` twice (copy-paste from another scenario) — never creates real causal links between threats and defenses.
- `EmbeddingService` initialized (Xenova/all-MiniLM-L6-v2, dim=384) but never used in threat analysis or defense selection.
- The scenario file is in `simulation/scenarios/` — it is a demonstration scaffold, not production code.

**Test file** (aidefence-guard.test.ts, 235 LOC): 37 tests using vitest with clean beforeEach isolation. All tests verify regex-based outputs; none exercise the behavioral analysis path (`enableBehavioralAnalysis=true`) or `analyzeBehavior()`. Config tests create strict/permissive configs but never verify their effect on detection behavior. Tests give a misleadingly high quality impression for the untested subsystems.

**AIDefence security classification**: The module provides genuine regex-based defense at the input/output boundary. It would function correctly if integrated — the code quality is real. The problem is organizational: it is architecturally isolated (ruvbot package), not called from AgentDB controllers, and the `aidefence` npm package it was intended to wrap is unused.

### 5l. Prime-Radiant Storage Module (R107, R108)

The prime-radiant storage module comprises 4 backend implementations totaling 3,193 LOC across postgres.rs (1,082), file.rs (804), memory.rs (731), and mod.rs (576). All 4 files are DEEP and average ~85% real. Each backend implements the `GraphStorage` + `GovernanceStorage` trait pair, giving the module a consistent interface despite fragmented internals.

**postgres.rs** (78-82%): Feature-gated via `sqlx` postgres feature flag. Has a race condition in `store_witness()` — sequence number is computed with `SELECT COALESCE(MAX(sequence),0)+1` outside a transaction lock, so concurrent calls can assign duplicates (H64). `find_similar()` fetches ALL nodes with matching dimension and computes cosine similarity in Rust iterators with no SIMD and no HNSW — O(n) scan at scale (H65). The file lacks any sheaf-theoretic structure despite being in the prime-radiant crate.

**file.rs** (85-90%): Uses a WAL (write-ahead log) with Blake3 checksums for integrity verification — a genuine persistence design. CRITICAL bug: `commit_wal()` does not set `committed=true` on `WalEntry`. As a result, `recover_from_wal()` replays ALL entries including previously committed ones on crash recovery, causing duplicate operations (C24). `find_similar()` is also O(n) sequential scan.

**memory.rs** (88-92%): Uses `parking_lot::RwLock` for concurrency. Has 9 unit tests. `store_witness()` inserts the witness into `self.witnesses` but NEVER writes the `action→witness` mapping into `witnesses_by_action` — `get_witnesses_for_action()` therefore returns empty for all stored witnesses (H67).

**mod.rs** (82-86%): The hub declares both `HybridStorage` and `StorageFactory`, but `HybridStorage` has only a `file_storage:FileStorage` field with no postgres field — the "hybrid" name is misleading (H66). `StorageFactory` only creates `InMemoryStorage`; no factory method produces postgres or hybrid backends.

The module follows the broader ruvnet pattern: architecturally complete with well-designed trait interfaces, but implementation gaps and bugs prevent production use.

### 5m. RVF Backend Subsystem (R115)

The RVF backend subsystem is a 20-file, ~5,921 LOC self-learning vector storage layer (ADR-004 + ADR-006 + ADR-007 + ADR-010). It spans factory.ts, RvfBackend.ts, SelfLearningRvfBackend.ts, NativeAccelerator.ts, ContrastiveTrainer.ts, SonaLearningBackend.ts, FederatedSessionManager.ts, SemanticQueryRouter.ts, AdaptiveIndexTuner.ts, SolverBandit.ts, SimdFallbacks.ts, RvfSolver.ts, FilterBuilder.ts, SqlJsRvfBackend.ts, WasmStoreBridge.ts, validation.ts, db-fallback.ts, rvf.ts, ModelCacheLoader.ts, and wasm-loader.ts. Weighted average: **~77%**, with a bimodal distribution — infrastructure files 88-95% genuine, self-learning orchestration 55-72% broken.

**4-tier backend chain (ADR-004):**

```
factory.ts auto-detect:
  1. RuVectorBackend (ruvector npm) — fastest, HNSW native
  2. RvfBackend (@ruvector/rvf) — native HNSW .rvf binary format
  3. HNSWLibBackend (hnswlib-node C++) — fallback ANN
  4. SqlJsRvfBackend (sql.js WASM) — last resort, O(n), SQLite .rvf format
```

Key problem: RvfBackend .rvf (HNSW binary) and SqlJsRvfBackend .rvf (SQLite export) are incompatible despite sharing the same file extension and capability name. db-fallback.ts has zero knowledge of the native format. An .rvf file from Tier 2 cannot be loaded by Tier 4 and vice versa (C26, C33, C45).

**Self-learning pipeline — SONA loop real, everything else broken:**

ADR-006 describes a 3-layer self-learning system: SONA delegation (Loop A/B/C), contrastive learning (ContrastiveTrainer), and bandit-guided compression (SolverBandit + AdaptiveIndexTuner).

| Component | Status | Verdict |
|-----------|--------|---------|
| SONA delegation (SonaLearningBackend) | REAL | N-API to @ruvector/sona. All 3 loops present. sonaApplyBaseLora 1-arg vs 2-arg bug (C42). |
| Contrastive learning (ContrastiveTrainer) | TRAINS but NEVER APPLIED | trainBatch() updates projection weights but trainer.project() never called in SelfLearningRvfBackend (C34). |
| Negative mining (SelfLearningRvfBackend) | PERMANENTLY BROKEN | recentSearches entries pushed with quality:undefined, so condition r.quality !== undefined is always false — no negatives ever produced (C35). |
| Bandit tier selection (SolverBandit) | RANDOM FOREVER | selectArm() called but recordReward() never called anywhere — Beta(1,1) uniform forever (C40). |
| Learning rate adaptation (_learningRate) | DEAD VARIABLE | Updated by runAcceptanceCheck() but never passed to any component (C36). |

**Self-learning ~40% justified:** The SONA delegation layer (SonaLearningBackend.ts) is real N-API code that genuinely invokes Rust SONA kernels. The contrastive and bandit layers have genuine algorithmic implementations (ContrastiveTrainer 87-90%, SolverBandit Thompson Sampling correct) but both are dead in production due to integration bugs, not algorithmic failure.

**NativeAccelerator API mismatch epidemic (ADR-007 Phase 1):**

NativeAccelerator.ts claims to wire 11 @ruvector package capabilities. Of these, 7 are permanently broken:

| Capability | Status | Root Cause |
|------------|--------|------------|
| TensorCompress | BROKEN | Static property check on instance method (C27) |
| AdamWOptimizer | BROKEN | 7-arg call vs 2-arg API (C28) |
| InfoNceLoss | BROKEN | Single positive + 4th arg vs array positives + constructor temp (C29) |
| SemanticRouter load | BROKEN | Static method call on instance method (H74) |
| Graph transactions | BROKEN | beginTransaction not in @ruvector/graph-node API (H75) |
| Graph Cypher | BROKEN | .cypher not in API, actual method is .query (H75) |
| RVF WASM | BROKEN | @ruvector/rvf-wasm not installed in any known deployment (H76) |
| SIMD via @ruvector/ruvllm | REAL | N-API binary works if ruvllm binary present |
| @ruvector/gnn (graph ops) | REAL | Installed, batch insert works |
| @ruvector/attention (MatMul) | REAL | Installed, matMul/softmax work |
| @ruvector/sona (via SonaLearningBackend) | REAL | N-API delegation works |

**12th parallel routing system (SemanticQueryRouter):**

SemanticQueryRouter routes SEARCH STRATEGY (which memory store/handler to use) via embedding cosine similarity. It is not connected to ADR-008 model routing (haiku/sonnet/opus), not connected to agentic-flow's SemanticRouter.ts (which independently imports EmbeddingService), and not connected to LLMRouter.ts. The @ruvector/router dependency is not installed — the HNSW-backed path is permanently unavailable; all routing runs brute-force fallback (C39).

**14th disconnected subsystem:**

SelfLearningRvfBackend is exported from neither index.ts nor backends/index.ts. It is reachable only via wasm-loader.ts (a secondary export point), making it a private implementation detail. ADR-006's primary class is architecturally isolated from the AgentDB public API surface (C46, H92).

### 5o. V3 MCP Tool Layer (R138)

R138 traced the MCP tool chain end-to-end to determine where AgentDB integration is (or is not) present. The result is a **4-layer integration gap** — a complete disconnect from user-facing MCP entry point through to backend storage.

**Layer 1 — Memory Initializer (R136):** `AgentDBAdapter` (the default backend for `UnifiedMemoryService`) is a plain `Map<string, MemoryEntry>` with zero connection to any AgentDB package (C50).

**Layer 2 — MCP Server Bootstrap (R138):** `v3/mcp/server.ts` (792 LOC, 88-92%) calls `getAllTools()` to register all 82 MCP tools but performs zero backend initialization. No calls to `memory-initializer`, `AgentDB`, or `EmbeddingService`. The server boots completely standalone.

**Layer 3 — MCP Tool Registration (R138):** `v3/mcp/tools/index.ts` (445 LOC, 82%) is the central tool hub aggregating 82 tools from 12 groups. The `memory-tools.ts` group claims ADR-006 (AgentDB integration) but contains zero agentdb imports — it is self-contained with zod-only validation (H116).

**Layer 4 — MCP Tool Handlers (R138):** The 14 SONA tools are full facades (H114). Their handler implementations:
- **Fabricated speedup**: `estimatedBruteForce = searchLatency * 1000`, then `speedup = estimatedBruteForce / searchLatency` — always produces ~1000x "speedup" (C58). This extends C57 from dead code to the LIVE tool surface.
- **LoRA no-ops**: LoRA tool handlers return `output = input.input` — the "fine-tuning" operation is an identity function.
- **SONAState = Maps only**: All trajectory, pattern, and metric state uses plain JavaScript Maps. Zero persistence, zero AgentDB, all state lost on restart (H115).
- **agentic-flow/core NOT installed**: The intended SONA engine import fails, causing all handlers to fall to their Map-based stubs.

**4 groups with REAL backends:**

| Group | Backend | Status |
|-------|---------|--------|
| hooks | ReasoningBank | Connected, functional |
| worker | WorkerDispatch | Connected, functional |
| federation | FederationHub | Connected, functional |
| agent | SecureLogger | Connected, functional |

These 4 groups (~24 tools) are genuinely wired. The remaining ~58 tools (SONA, memory, analytics, etc.) operate on in-memory stubs or no-op handlers.

**Library MCP Server (`v3/@claude-flow/mcp/src/server.ts`):**

The library MCP server (1,134 LOC, 88-92%) is the most sophisticated MCP implementation in the project — 14 methods, 9 sub-registries, TypedEventEmitter, proper lifecycle management. It registers only 4 built-in tools; all domain tools (including any AgentDB-related ones) must be externally registered via its `registerTool()` API. The file has zero references to memory, AgentDB, or embeddings anywhere in its imports or methods. It is architecturally a pure protocol shell — the cleanest MCP code in the project, but with no awareness of the vector database layer (H113).

**End-to-end disconnect confirmed:**

```
User → MCP server (zero init) → getAllTools() (82 tools) → tool handler
  ├── 4 groups → real backends (ReasoningBank, WorkerDispatch, etc.)
  ├── 14 SONA tools → in-memory Maps, fabricated metrics, no-op LoRA
  ├── memory tools → zod-only, zero agentdb import (claims ADR-006)
  └── remaining tools → mixed, no AgentDB anywhere
```

AgentDB is not just unused at the entry point (R135) or the memory layer (R136) — it is absent at EVERY layer of the V3 MCP stack.

### 5n. V3 @claude-flow/memory Integration (R136)

The V3 memory layer (`@claude-flow/memory` package, 7 files, ~8,967 LOC) is claude-flow's rewritten "brain" — the subsystem responsible for all agent memory operations. R136 reveals a deeply misleading architecture where the naming suggests AgentDB integration but the default runtime path contains none.

**Architecture: Three disconnected paths to memory**

```
Path A (DEFAULT — used at runtime):
  UnifiedMemoryService → AgentDBAdapter → Map<string, MemoryEntry> + pure JS HNSW
  Result: In-memory only, zero AgentDB, zero persistence, broken HNSW (p=0.5)

Path B (INTENDED — dead at runtime):
  memory-bridge.ts → ControllerRegistry → ~19 AgentDB controllers via dynamic import
  Result: NOT COMPILED into npm dist. All 28 bridge functions return null.

Path C (ALTERNATIVE — not wired to default service):
  agentdb-backend.ts → native HNSW fallback chain (ruvector > hnswlib > sql.js)
  Result: Exists but inaccessible from UnifiedMemoryService. Orphaned alternative.
```

**Why Path B is dead:** `memory-bridge.ts` imports from `./controller-registry` which itself dynamically imports `agentdb`. The bridge file (1,773 LOC of correct BM25, hybrid scoring, and 28 route functions) is excluded from the TypeScript compilation that produces the npm-distributed JavaScript. At runtime, the import fails silently, every bridge function returns null, and the system falls back to sql.js direct access.

**AgentDBAdapter vs actual AgentDB:**

| Feature | AgentDBAdapter (V3 default) | AgentDB (native package) |
|---------|----------------------------|--------------------------|
| Storage | Map<string, MemoryEntry> | SQLite + EmbeddingService |
| HNSW | Pure JS (broken p=0.5) | hnswlib-node (C++) |
| Persistence | Empty stubs | SQLite WAL |
| Embeddings | Optional (often omitted) | @xenova/transformers |
| Controllers | 0 | 23+ |
| Search | JS cosine O(n) | BM25 + HNSW fusion |

The naming `AgentDBAdapter` for a plain in-memory Map store is the single most misleading identifier in the V3 memory layer. It causes readers (and likely the original developers in later sessions) to assume AgentDB is wired in when it is not.

**controller-registry.ts — the bridge that almost works:**

The controller registry is the most sophisticated component in the V3 memory layer. It declares 29 controllers, initializes 28 across 5 ordered levels via Promise.allSettled, and delegates ~19 of them to AgentDB via `import('agentdb')`. The level-ordered init and reverse-order shutdown are genuinely production-quality patterns. However, the registry is only reachable from memory-bridge.ts (dead) and direct import (not the default path). Four controllers are explicit null placeholders (`hybridSearch`, `agentMemoryScope`, `federatedSession`, `sonaTrajectory`), and `causalRecall` is declared but absent from all init levels.

**memory-initializer.ts — the third embedding pipeline:**

The memory initializer (2,564 LOC) has its OWN 3-tier embedding pipeline completely separate from both AgentDB's EmbeddingService and the controller-registry's createEmbeddingService(). The `options.backend` configuration is written to metadata but NEVER influences actual backend selection — the initializer always uses its own internal path. HNSW is lazy-loaded via `@ruvector/core` VectorDb with silent degradation to SQLite when unavailable.

**15th disconnected persistence/memory layer:**

The V3 memory layer adds another disconnected persistence system to the project's growing collection. The default path (AgentDBAdapter) has no persistence at all — `loadFromDisk()` and `saveToDisk()` are empty stubs. Combined with the 14 previously identified disconnected subsystems (MemoryController.ts, rvf-mcp-server, prime-radiant storage backends, etc.), this represents a systemic architectural pattern where each rewrite/layer introduces its own memory system without composing with existing ones.
