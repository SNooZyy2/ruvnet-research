# Production Infrastructure Domain Analysis

> **Priority**: MEDIUM | **Coverage**: ~19% (78/~400 DEEP) | **Status**: In Progress
> **Last updated**: 2026-03-03 (Sessions R134-R140, full middle-layer sweep)

## Section 1: Current State Summary

Production infrastructure spans deployment, operations, self-healing, monitoring, database management, governance, CLI entrypoints, MCP tool chains, CI/CD pipelines, integration testing, and deployment configurations across the ruvnet ecosystem. R139 analyzed the CI/testing/deployment ground truth — 14 files totaling ~5,757 LOC across CI pipelines (6 files including NAPI build workflows), Rust integration tests (2 files), deployment configs (4 files), and backend/barrel exports (2 files) — revealing that the entire CI and testing infrastructure provides near-zero quality assurance, while the NAPI cross-compilation infrastructure is surprisingly production-grade.

**CI pipelines are facades.** Both claude-flow pipelines (ci.yml 228 LOC, v3-ci.yml 157 LOC) use `continue-on-error: true` on all test, typecheck, and security audit steps — only lint can fail the build. V3 has 11 specific test scripts in package.json (test:integration:memory, test:integration:swarm, test:integration:mcp, etc.) but NONE run in CI. V2 and V3 are completely separate pipelines (npm/Jest vs pnpm/Vitest) with zero cross-version regression testing. Green CI provides near-zero assurance about code correctness. The **ruvector release pipeline** (release.yml, 621 LOC) is the most genuine CI in the ecosystem — a 7-job DAG publishing 25 Rust crates in topological sort with rate limiting — but has a `skip_tests` input that bypasses ALL validation, `npm run test:unit || true` that silently swallows failures, and tests on ubuntu-22.04 only before publishing 5-platform native binaries. The **publish-all pipeline** (552 LOC) is a misnomer covering only math+attention families, with all 12 publish steps using `continue-on-error: true` so the pipeline always reports success even if every publish fails.

**"Integration tests" are 100% mock-only.** Both Rust integration test files (2,928 LOC combined) test zero cross-crate integration. e2e_integration_test.rs (ruvllm, 1,535 LOC) uses MockLlmBackend with deterministic hash-based token generation — it genuinely tests production algorithms (softmax, top_k, KV cache, SpeculativeStats) but not hnsw_router, NAPI bridge, SIMD, or cross-crate composition. ruvllm_integration_tests.rs (prime-radiant, 1,393 LOC) has zero imports from either `prime_radiant::` or `ruvllm::` despite `#![cfg(feature = "ruvllm")]` — all 25 tests use inline reimplementations of 5 subsystems whose mock APIs do not match production APIs at all.

**NAPI cross-compilation is production-grade.** The build-native.yml (242 LOC) builds ruvector-core NAPI binaries for 5 platforms (linux-x64/arm64, darwin-x64/arm64, win32-x64). The sona-napi.yml (299 LOC) is the most comprehensive NAPI CI — 7 platforms (adds musl + win32-arm64), creates a universal macOS binary via `lipo`, publishes per-platform @ruvector/sona-{platform} packages to npm, and runs a post-publish smoke test on 3 OSes. However, both workflows have the `continue-on-error: true` pattern on tests/publish steps, and the graph-node NAPI build is DISABLED due to compilation issues (PR #15).

**Deployment: V3 Docker is production-grade, distributed test is facade.** The V3 docker-compose.yml (117 LOC) is the most production-ready deployment in the ecosystem — 3 profiles (lite/full/workers), non-root user, dumb-init, monitoring via Prometheus+Grafana. The ruvector distributed docker-compose.yml (198 LOC) configures a 5-node Raft cluster where nodes run shell scripts with netcat responding "200 OK" to healthchecks — they NEVER invoke any Rust binary, and .rlib files (static libraries, not executables) are copied to runtime images. Agentic-flow has a demo config (docker-compose.agent.yml: 7 agent services with Docker profiles and hardcoded example tasks, not production).

**R140 traced the V3 execution engine** — 7 files, ~10,542 LOC. Cluster A (Service Layer): headless-worker-executor.ts (1,342 LOC) spawns `claude --print <prompt>` as a one-shot subprocess with no MCP protocol and no AgentDB/memory connection; worker-daemon.ts (942 LOC) is a foreground Node.js class (not a daemon) where 9 of 12 local workers are facade stubs; claim-service.ts (1,118 LOC) uses JSON file persistence but its claimant format (2-part `agentId:taskId`) is incompatible with MCP claims-tools.ts (3-part `agentId:taskId:timestamp`), making them two parallel systems that cannot interoperate; container-worker-pool.ts (783 LOC) implements real Docker CLI integration but silently drops `prompt` and `contextPatterns` fields in `buildWorkerCommand()` — every containerized task runs without its instructions. Cluster B (Memory Intelligence): intelligence.ts (985 LOC) advertises O(log n) HNSW routing but is actually O(n) brute-force, and its LocalSonaCoordinator is a circular buffer with no HNSW; sona-optimizer.ts (842 LOC) implements real Bayesian confidence updates for agent routing but is not a vector search optimizer. Cluster C (Hooks CLI): hooks.ts (4,530 LOC) exposes 30 subcommands (vs 17 documented) with genuine ADR-008 pre-task model routing via enhanced-model-router.js, but token-optimize hardcodes +200 saved tokens unconditionally regardless of actual compression.

The V3 MCP tool chain, CLI entrypoints, V3 memory layer, AgentDB MCP server, and prior subsystem summaries from R138/R134-R136 remain valid. Systemic patterns: (1) real primitives disconnected from semantic layers; (2) governance with unverified crypto; (3) strong in-memory, no production backends; (4) algebraic-topology correctness bugs; (5) hash embedding end-to-end from CLI through engine to storage; (6) V2→V3 tool regression — more infrastructure, fewer working tools, zero memory bootstrap in any MCP server; (7) CI/testing facade pattern — pipelines and "integration tests" exist structurally but provide near-zero quality gate; (8) worker execution is subprocess-only with zero memory integration — the actual agent spawning mechanism (`claude --print`) is a one-shot subprocess with no MCP, no AgentDB, no session continuity, extending the zero-memory-bootstrap pattern from MCP servers all the way through the worker execution chain; (9) V3 memory is a misnamed facade — AgentDBAdapter uses a plain JS Map with no AgentDB connection, loadFromDisk/saveToDisk are empty stubs, and the path traversal check in controller-registry is a no-op; (10) claim coordination is bifurcated — ClaimService uses a local JSON file while MCP claims-tools operate inline with incompatible 2-part vs 3-part claimant formats, making the two systems non-interoperable.

---

## Section 2: File Registry

| File | LOC | Real% | Session | Key Feature |
|------|-----|-------|---------|-------------|
| **crates/ruvvector-postgres/src/healing/learning.rs** | 670 | **92-95%** | R36 | Adaptive weight formula: success_rate*(1+improvement/100). Confidence scoring asymptotic. Human feedback integration. |
| **crates/ruvector-postgres/src/healing/detector.rs** | 826 | **85-90%** | R36 | 8 problem types, severity classification. ALL 8 metric collection methods return empty/zero. |
| **crates/ruvector-postgres/src/healing/engine.rs** | 789 | **75-80%** | R36 | Cooldown enforcement, rate limiting, rollback logic. CRITICAL: execute_with_safeguards() does not enforce timeout. |
| **crates/ruvector-postgres/src/healing/worker.rs** | 619 | **70-75%** | R36 | Health check loop, check_health() production-ready. CRITICAL: register_healing_worker() COMMENTED OUT. |
| **crates/ruvector-postgres/src/healing/strategies.rs** | 1,166 | **60-65%** | R36 | StrategyRegistry + adaptive learning real. ALL 5 execution methods are log-only stubs. |
| **src/cli/lib/health-monitor.ts** | 514 | **99%** | R48 | BEST monitoring in ecosystem. Real OS/V8 metrics, linear regression leak detection, MPC self-healing. |
| **src/security/path-security.ts** | 437 | **88-92%** | R48 | OWASP-compliant path traversal prevention. CRITICAL: ORPHANED — zero imports in entire AgentDB codebase. |
| **crates/ruvllm/src/serving/request.rs** | 473 | **88-92%** | R107 | vLLM/Orca request lifecycle. Chunked prefill, paged attention block_table, priority/preemption, OpenAI SSE TokenOutput. HIGH: stop_sequences silently ignored. |
| **crates/ruvllm/src/serving/mod.rs** | 348 | **92%** | R107 | Pure orchestration + 4 integration tests. NoopBackend decoupling confirmed. serving MODULE COMPLETE (6/6). |
| **crates/prime-radiant/src/storage/postgres.rs** | 1,082 | **82-87%** | R107 | REAL async sqlx/PgPool. JSONB governance. BUT: zero sheaf integration, brute-force similarity, witness race condition, dead AsyncGraphStorageAdapter, no migration versioning. |
| **crates/prime-radiant/src/governance/repository.rs** | 1,062 | **62%** | R107 | TRAITS ONLY — no production backends. "Async-First" claim FALSE. Good in-memory (parking_lot, Kahn topo sort, chain integrity). 12th persistence stub. |
| **crates/prime-radiant/src/storage/file.rs** | 804 | **85-90%** | R108 | Dual-format (bincode/JSON), blake3 WAL integrity, parking_lot concurrency. CRITICAL WAL commit-flag bug — deletions non-durable across restarts. |
| **crates/prime-radiant/src/storage/memory.rs** | 731 | **88-92%** | R108 | Genuine volatile backend. witnesses_by_action never populated (HIGH). |
| **crates/prime-radiant/src/storage/mod.rs** | 576 | **82-86%** | R108 | GraphStorage + GovernanceStorage traits clean. HybridStorage = FileStorage ONLY despite config. StorageFactory ignores postgres_url. |
| **crates/prime-radiant/src/governance/policy.rs** | 970 | **82-87%** | R108 | Genuine state machine + Blake3. CRITICAL: signatures never verified. CRITICAL: zero-byte placeholder() activates bundles. EscalationCondition DSL has no evaluator. GOVERNANCE MODULE COMPLETE (5/5). |
| **crates/prime-radiant/src/governance/mod.rs** | 439 | **88-92%** | R108 | Pure re-export. Hash/Timestamp/Version shared types. Blake3 hex inline module. |
| **crates/prime-radiant/src/governance/witness.rs** | 723 | **85-90%** | R108 | GENUINE Blake3 hash chain. ComputeLane enum. Single-witness model only (no quorum). |
| **crates/prime-radiant/src/governance/lineage.rs** | 873 | **85-90%** | R108 | Genuine tamper-detection (Blake3 hash-of-hashes). Kahn topo sort in repository.rs. EntityLineageTracker no cycle detection. |
| **crates/prime-radiant/src/cohomology/cocycle.rs** | 471 | **75-80%** | R110 | Cocycle/Coboundary/SheafCocycle/SheafCoboundary. CRITICAL: is_coboundary() always false for degree>0; apply_adjoint() dimension bug. Doubly-broken Hodge Laplacian. 4 tests. |
| **crates/ruvector-attention/src/transport/cached_projections.rs** | 242 | **88-92%** | R110 | Genuine sliced-Wasserstein OT utility. ProjectionCache (unit-norm directions), WindowCache (sorted per projection), CDF histograms. 3 tests. dot_product_simd is scalar unroll only. |

### CLI Entrypoints (R134)

| File | LOC | Real% | Session | Key Feature |
|------|-----|-------|---------|-------------|
| **npm/packages/ruvector/bin/cli.js** | 7,357 | **72-78%** | R134 | Monolith CLI, ~70% genuine. Uses VectorDB (working). Hash embedding at CLI layer. 4 facade commands (graph/router/server/cluster). |
| **npm/packages/ruvector/bin/mcp-server.js** | 3,007 | **78-82%** | R134 | 55-tool MCP server (largest). Heterogeneous delegation (library, execSync, agentic-flow). sanitizeShellArg() destroys SQL/Cypher/SPARQL. |
| **v3/@claude-flow/cli/bin/cli.js** | 156 | **N/A** | R134 | Cold dispatcher. Zero subsystem init at boot. Everything lazy-loaded. |
| **v3/@claude-flow/cli/bin/mcp-server.js** | 189 | **72-78%** | R134 | Near-duplicate MCP. Async ordering race. False resources capability. |
| **ruflo/bin/ruflo.js** | 50 | **70-75%** | R134 | Pure claude-flow rebrand. Resolves @claude-flow/cli, fails silently if missing. |
| **npm/packages/rvlite/bin/cli.js** | 1,686 | **80-85%** | R134 | Independent alternative. Own vector store, O(n) search, genuine WASM. Zero code sharing with ruvector CLI. |
| **npm/packages/ruvllm/bin/cli.js** | 1,005 | **72-78%** | R134 | Native-binary-dependent. JS fallback = all facades. Training simulated. Benchmark uses hash embeddings. |

### V3 Memory Layer (R135) and AgentDB MCP Server (R136)

| File | LOC | Real% | Session | Key Feature |
|------|-----|-------|---------|-------------|
| **v3/@claude-flow/memory/src/agentdb-adapter.ts** | 1,038 | **35-45%** | R135 | MISNAMED: stores data in plain JS Map, zero AgentDB imports. loadFromDisk/saveToDisk are EMPTY STUBS. R20 root cause NOT fixed. |
| **v3/@claude-flow/memory/src/controller-registry.ts** | 1,026 | **55-65%** | R135 | Path traversal validation is a no-op. createEmbeddingService() fallback returns zero-filled Float32Array. causalRecall stub never reached. |
| **packages/agentdb/src/mcp/agentdb-mcp-server.ts** | 2,368 | **75-80%** | R136 | 28 actual tools (not 32 claimed). EmbeddingService IS initialized (Pipeline 1 only). causal_add_edge hardcodes fromMemoryId=0. |

### V3 MCP Tool Chain (R138)

| File | LOC | Real% | Session | Key Feature |
|------|-----|-------|---------|-------------|
| **v3/mcp/tools/index.ts** | 445 | **82-87%** | R138 | Central tool export hub. 82 tools (67 V3 + 15 V2 compat) from 12 groups. SONA tools fabricate speedup metrics. |
| **v3/mcp/server.ts** | 792 | **88-92%** | R138 | V3 MCP server bootstrap. JSON-RPC 2.0, AJV schema validation. Zero memory/AgentDB init. |
| **v3/@claude-flow/mcp/src/server.ts** | 1,134 | **88-92%** | R138 | Library-level MCP 2025-11-25 server. 14 methods, 9 sub-registries, 3,040 LOC modules. Only 4 built-in tools. |
| **v3/@claude-flow/cli/src/commands/index.ts** | 399 | **82-87%** | R138 | CLI command registry. 31 commands. Lazy loading nullified. 2 orphan files unreachable. |
| **v2/src/mcp/server.ts** | 646 | **85-90%** | R138 | V2 MCP server. 3 tool factories (~64 tools). AuthManager + LoadBalancer with circuit breaker — ALL LOST in V3. |
| **v3/@claude-flow/cli/src/commands/ruvector/setup.ts** | 784 | **85-90%** | R138 | PostgreSQL scaffolding. docker-compose + init-db.sql (8 tables, 6 HNSW indices). Zero backend factory bridge. |

### CI/CD Pipelines (R139)

| File | LOC | Real% | Session | Key Feature |
|------|-----|-------|---------|-------------|
| **.github/workflows/release.yml** (ruvector) | 621 | **82-87%** | R139 | 7-job DAG publishing 25 Rust crates in topo sort. skip_tests bypasses ALL validation. npm test silently swallowed. |
| **.github/workflows/publish-all.yml** (ruvector) | 552 | **70-75%** | R139 | MISNOMER — only math+attention families. All 12 publish steps continue-on-error: true. Hardcoded 0.1.31 fallback. |
| **.github/workflows/ci.yml** (claude-flow) | 228 | **40-45%** | R139 | CI FACADE — continue-on-error on all test/typecheck/audit. Only lint can fail. Deploy job is placeholder. |
| **.github/workflows/v3-ci.yml** (claude-flow) | 157 | **35-40%** | R139 | 11 test scripts exist but NONE run in CI. No security audit. Zero cross-version V2/V3 regression testing. |
| **.github/workflows/build-native.yml** (ruvector) | 242 | **82-87%** | R139 | 5-platform NAPI cross-compile (linux-x64/arm64, darwin-x64/arm64, win32-x64). graph-node DISABLED (PR #15). Test continue-on-error:true. Manual publish only. Binaries auto-committed to repo. |
| **.github/workflows/sona-napi.yml** (ruvector) | 299 | **85-90%** | R139 | 7-platform SONA NAPI build (+ musl + win32-arm64) with universal macOS binary via lipo. Full publish pipeline to @ruvector/sona-{platform}. Post-publish smoke test. Most comprehensive NAPI CI. |

### Rust Integration Tests (R139)

| File | LOC | Real% | Session | Key Feature |
|------|-----|-------|---------|-------------|
| **crates/ruvllm/tests/e2e_integration_test.rs** | 1,535 | **65-70%** | R139 | "E2E" misnomer — mock-backend unit tests. Genuinely tests softmax/top_k/KV cache. Unsafe UB at L840. |
| **crates/prime-radiant/tests/ruvllm_integration_tests.rs** | 1,393 | **45-50%** | R139 | 100% mock-only. Zero ruvllm imports despite cfg(feature="ruvllm"). Mock APIs don't match production. |

### Deployment Configs (R139)

| File | LOC | Real% | Session | Key Feature |
|------|-----|-------|---------|-------------|
| **distributed/docker-compose.yml** (ruvector) | 198 | **30-35%** | R139 | 5-node Raft FACADE — nodes run netcat shell scripts, never invoke Rust binaries. .rlib copied but not executable. |
| **docker-compose.yml** (agentic-flow) | 9 | **60-65%** | R139 | 9-line minimal demo. Hardcoded TOPIC. Backing Dockerfile IS functional (node:20-slim). |
| **v3/docker-compose.yml** (claude-flow) | 117 | **88-92%** | R139 | MOST production-ready deployment. 3 profiles (lite/full/workers). Non-root, dumb-init, Prometheus+Grafana. |
| **docker-compose.agent.yml** (agentic-flow) | 114 | **55-60%** | R139 | Agent demo compose — 7 services (goal-planner, coder, reviewer, tester, researcher, swarm, parallel) all using Docker profiles with hardcoded example tasks. Not production deployment. |

### Backend & Barrel Exports (R139)

| File | LOC | Real% | Session | Key Feature |
|------|-----|-------|---------|-------------|
| **src/backends/factory.ts** (agentdb root) | 235 | **85-88%** | R139 | 2-tier backend factory (ruvector > hnswlib). SIMPLER than 5-tier (ID 12809). Clean dynamic imports, lazy loading. COMPETES with packages/agentdb factory. |
| **npm/packages/ruvector/src/core/index.ts** | 57 | **85-88%** | R139 | Master barrel export — 23 modules. GNN, SONA, ONNX×2, Router, Graph, Cluster, AST, Neural×3, RVF, etc. No selective gating — dead code leaks to consumers. |

### Execution Engine (R140)

| File | LOC | Real% | Session | Key Feature |
|------|-----|-------|---------|-------------|
| **v3/@claude-flow/cli/src/services/headless-worker-executor.ts** | 1,342 | **78-83%** | R140 | GENUINE subprocess executor. Spawns `claude --print <prompt>` via child_process.spawn. Process pool (maxConcurrent=2) + pending queue. Zero MCP, zero memory layer. 8 worker types (2 enabled by default). Prompt injected as raw CLI argument — potential shell injection. Double timeout bug. |
| **v3/@claude-flow/cli/src/services/worker-daemon.ts** | 942 | **55-65%** | R140 | NOT a daemon — foreground EventEmitter class. 9 of 12 local workers are FACADE stubs. Integrates with HeadlessWorkerExecutor for AI-powered workers only. Signal handlers present but stop() bug saves state as running=false. |
| **v3/@claude-flow/cli/src/services/container-worker-pool.ts** | 783 | **72-78%** | R140 | REAL Docker CLI integration (docker --version, docker run). CRITICAL BUG: prompt + contextPatterns silently dropped in buildWorkerCommand(). ANTHROPIC_API_KEY passed via -e flag. Hard-coded image ghcr.io/ruvnet/claude-flow-headless:latest. |
| **v3/@claude-flow/cli/src/services/claim-service.ts** | 1,118 | **68-75%** | R140 | LOCAL-ONLY JSON file persistence (.claude-flow/claims/claims.json). Format incompatible with MCP claims-tools.ts (2-part vs 3-part claimant). rebalance() generates suggestions only, never moves claims. getAvailableIssues() stub — always returns []. |

### Hooks CLI Command Layer (R140)

| File | LOC | Real% | Session | Key Feature |
|------|-----|-------|---------|-------------|
| **v3/@claude-flow/cli/src/commands/hooks.ts** | 4,530 | **72-78%** | R140 | Genuine ADR-005 MCP-first CLI. 30 subcommands (not 17 as documented). All callMCPTool() wrappers. REAL: pre-task enhanced-model-router integration, model-route/outcome/stats, intelligence status (honest lazy-loading). BUGS: statusline vector count from file size heuristic; token-optimize hardcodes +200 savings. |

---

## Section 3: Findings Registry

### CRITICAL Findings

**F-PRODINFRA-001** (R36): ALL 5 healing strategies are log-only stubs — reindex, promote replicas, evict problematic, block queries, repair edges perform no actual database operations.

**F-PRODINFRA-002** (R36): execute_with_safeguards() does NOT enforce timeout — comment says it should catch panics and enforce timeout, but implementation does neither.

**F-PRODINFRA-003** (R36): register_healing_worker() is COMMENTED OUT — background worker cannot be registered with PostgreSQL, making the entire healing system non-functional in production.

**F-PRODINFRA-004** (R48): path-security.ts ORPHANED — 437 LOC of OWASP-compliant security with ZERO imports. RuVectorBackend.ts implements its own validatePath() instead.

**F-PRODINFRA-005** (R48): path-security.ts missing Unicode normalization — vulnerable to Unicode equivalence attacks (/\u002e\u002e/).

**F-PRODINFRA-006** (R108): Approval signatures NEVER cryptographically verified in policy.rs add_approval() — checks approver_id list membership and deduplication only. A zero-byte [0u8; 64] placeholder() signature is accepted identically to a real ed25519 signature. The entire multi-signature governance mechanism provides zero cryptographic assurance.

**F-PRODINFRA-007** (R108): ApprovalSignature::placeholder() is a production-visible constructor injecting zero-byte signatures (vec![0u8; 64]) with algorithm="placeholder". Since verification never runs, placeholder signatures activate real policy bundles. Method doc says "NOT for production" but there is no runtime guard.

**F-PRODINFRA-008** (R108): WAL commit-flag bug in file.rs — WAL entries are written with committed=false and the commit step is either missing or incorrectly wired. Deletions written to WAL are non-durable across restarts: the record exists in WAL but the committed flag gates replay, so deletes are lost on crash recovery.

**F-PRODINFRA-009** (R110): `is_coboundary()` for degree > 0 in cocycle.rs always returns false without solving the linear system — determining coboundary status requires solving delta(g) = f. `cohomology_group.rs` relies on this to separate Z^n from B^n and compute H^n. All non-zero H^n for n >= 1 is therefore wrong: every cocycle reports as non-coboundary, inflating all higher cohomology groups. CORRECTNESS BUG in the entire cohomology pipeline.

**F-PRODINFRA-010** (R110): `apply_adjoint()` in cocycle.rs has a dimension indexing bug — to compute delta^*(f) at degree n, must iterate (n+1)-simplices and distribute to n-faces. The method instead iterates simplices at `cochain.dimension` (n) and applies their boundary to (n-1)-faces, computing delta^* using wrong source dimension. The Hodge Laplacian composed via `laplacian()` is therefore incorrect. Combined with laplacian.rs eigensolver bugs from R109, the Hodge decomposition is doubly corrupted.

**F-PRODINFRA-011** (R134): Hash embedding fallback at CLI layer — ruvector cli.js:2784 Intelligence.embed() uses IntelligenceEngine only if ALREADY initialized; most hook invocations pass skipEngine:true, falling through to 64-dim hash embedding. Confirms R20/R117 systemic hash-embedding pattern at the user-facing CLI layer. No user-visible warning that embeddings are hash-based.

**F-PRODINFRA-012** (R134): Hash embedding fallback in MCP server — mcp-server.js:230 is the 9th confirmed instance of the systemic hash-embedding pattern. IntelligenceEngine unavailable is the common case since no MCP startup path initializes it.

**F-PRODINFRA-013** (R134): SQL/Cypher/SPARQL query destruction — mcp-server.js:2888 sanitizeShellArg() strips (), ;, ', ", $, {}, | from query strings before passing to rvlite. This destroys most legitimate SQL (`SELECT * FROM t WHERE x IN (1,2)`), Cypher (`MATCH (n)-[:R]->(m)`), and SPARQL queries. The sanitizer also fails to prevent actual SQL injection since it strips shell metacharacters, not SQL injection vectors.

**F-PRODINFRA-014** (R134): ruvllm CLI all-facades without native binary — ruvllm cli.js:121 query/generate/route/embed ALL return hardcoded or hash values in the JS fallback path. Without the .node native addon the entire CLI is non-functional for its stated purpose.

**F-PRODINFRA-015** (R134): ruvllm training SIMULATED — ruvllm cli.js:624 ContrastiveTrainer.train() outputs "Training loss (simulated)". No actual gradient computation or model weight updates occur. Training command exists in help text with no disclaimer.

**F-PRODINFRA-015b** (R135): AgentDBAdapter is a MISNAMED facade — v3/@claude-flow/memory/src/agentdb-adapter.ts stores all data in a plain JS Map<string, MemoryEntry> (line 97). Zero imports from any AgentDB package. loadFromDisk() and saveToDisk() are empty stubs (lines 872-881) that emit events and return without reading/writing anything. createHybridService() claims to create SQLite+AgentDB hybrid but creates a plain UnifiedMemoryService. The R20 root cause (EmbeddingService never initialized in the claude-flow bridge) is NOT fixed in V3 — embeddingGenerator is optional and defaults to undefined.

**F-PRODINFRA-015c** (R135): Path traversal validation is a no-op in controller-registry.ts — path.resolve() normalizes ".." away BEFORE the includes("..") check. Any input including `../../etc/passwd` resolves to an absolute path, which then passes the test. The validation provides no security for path traversal attacks.

**F-PRODINFRA-015d** (R136): agentdb-mcp-server.ts tool count mismatch — the server documentation claims 32 tools (5 core + 9 frontier + 10 learning + 5 AgentDB + 3 batch ops = 32) but the actual tools array contains only 28 entries. 4 tools are documented but not registered.

**F-PRODINFRA-015e** (R136): causal_add_edge and causal_query handlers hardcode memory IDs to 0 — causal_add_edge sets fromMemoryId=0 and toMemoryId=0 (lines 1201-1204), meaning all causal edges point from episode 0 to episode 0 regardless of input. causal_query sets interventionMemoryId=0 (line 1229). The causal graph data model is entirely broken.

**F-PRODINFRA-016** (R138): SONA tools fabricate HNSW speedup metrics — v3/mcp/tools/index.ts computes `estimatedBruteForce = searchLatency * 1000` then `speedup = estimatedBruteForce / searchLatency`, which algebraically always yields ~1000x regardless of actual search method. This is the source of the "150x-12,500x" marketing claim. The fabrication is not a bug — it is a deliberate formula that manufactures impressive-looking numbers from a tautological multiplication.

**F-PRODINFRA-017** (R139): CI facade — claude-flow ci.yml (228 LOC) uses `continue-on-error: true` on BOTH test steps, type checking, and security audit. Only `npm run lint` can actually fail the build. Type checking is disabled due to "TypeScript compiler crash". Deploy job is a placeholder that downloads artifacts, prints echo, and exits. Green CI provides near-zero assurance about code correctness.

**F-PRODINFRA-018** (R139): Distributed Raft test is FACADE — ruvector distributed/docker-compose.yml configures a 5-node Raft cluster where each node runs a shell script with netcat responding "200 OK" to healthchecks. Nodes NEVER invoke any Rust binary. The Dockerfile copies .rlib files (static libraries, NOT executables) into the runtime image. Distributed deployment has NEVER been validated as real multi-process/multi-node.

**F-PRODINFRA-019** (R139): Both "integration" test files (2,928 LOC combined) are 100% mock-only — e2e_integration_test.rs (1,535 LOC) uses MockLlmBackend with deterministic hash-based token generation and zero cross-crate imports; ruvllm_integration_tests.rs (1,393 LOC) reimplements 5 subsystems inline with mock APIs that don't match production code at all (SheafCoherenceValidator mock uses cosine similarity vs real CoherenceEngine + SheafGraph; PatternToRestrictionBridge mock uses Vec<LearnedPattern> vs real ReasoningBank + rho maps; UnifiedWitnessLog mock is simple hash chain vs real GenerationWitness). Neither file tests actual cross-crate integration.

### HIGH Findings

**F-PRODINFRA-H-001** (R36): ALL 8 metric collection methods return empty/zero — detector.rs cannot query PostgreSQL catalog tables, so no problems are ever detected.

**F-PRODINFRA-H-002** (R36): worker.rs uses thread::sleep instead of PostgreSQL WaitLatch — not interruptible, cannot respond to shutdown signals.

**F-PRODINFRA-H-003** (R36): Cooldown/rate-limiting is production-ready — engine.rs tracks attempts per window, enforces min_healing_interval. Genuine but undercut by non-functional strategy execution. (Positive framing — documented as notable finding)

**F-PRODINFRA-H-004** (R48): health-monitor.ts linear regression leak detection is IMPRESSIVE — mathematically sound slope via sum(x-meanX)*(y-meanY)/sum(x-meanX)^2, checks slope>10MB AND 80% consistent growth. Best leak detection across ruvnet.

**F-PRODINFRA-H-005** (R48): health-monitor.ts MPC self-healing production-ready — canRecoverWithGC() checks v8.getHeapStatistics(), healByGarbageCollection() invokes global.gc() if available. 4-tier escalation strategy.

**F-PRODINFRA-H-006** (R107): stop_sequences in GenerateParams silently ignored — should_stop() takes _decoded_text but never inspects it and delegates to is_complete() (token count only). Functional bug for any stop-sequence use case.

**F-PRODINFRA-H-007** (R107): serving/mod.rs integration tests are the ONLY end-to-end validation in the serving module — NoopBackend confirms decoupling but also means no GPU backend is tested in CI. Four tests cover concurrent requests, continuous batching join, priority scheduling, and KV cache allocation.

**F-PRODINFRA-H-008** (R107): Repository layer in governance/repository.rs is TRAITS ONLY with InMemory implementations — no PostgreSQL, SQLite, or ruvector backends. Doc comments promise "Hybrid (PostgreSQL + ruvector)" but zero concrete production backends are implemented. 12th persistence layer architecture.

**F-PRODINFRA-H-009** (R107): governance/repository.rs module doc claims "Async-First: All operations are async for I/O-bound persistence" but ALL trait methods and implementations are synchronous. Significant gap between stated design and reality.

**F-PRODINFRA-H-010** (R107): Race condition in postgres.rs store_witness() — uses SELECT COALESCE(MAX(sequence),0)+1 without transaction or lock. Under concurrent writes two callers can get the same sequence and one INSERT fails with unique constraint violation.

**F-PRODINFRA-H-011** (R107): AsyncGraphStorageAdapter in postgres.rs wraps PostgresStorage but does NOT implement GraphStorage or GovernanceStorage traits. The adapter is useless for trait-dispatch code.

**F-PRODINFRA-H-012** (R107): postgres.rs find_similar() fetches ALL nodes with matching dimension (full table scan), computes cosine similarity in Rust, sorts in memory — O(n) memory + O(n log n) CPU. pgvector mentioned in comment but not implemented.

**F-PRODINFRA-H-013** (R107): postgres.rs stores plain REAL[] vectors and JSONB blobs with no sheaf-theoretic structure, no Blake3 hash chains, no DashMap, no restriction maps. The postgres backend is a generic governance+graph store with zero connection to the coherence subsystem (energy.rs, spectral.rs, incremental.rs).

**F-PRODINFRA-H-014** (R108): storage/mod.rs HybridStorage = FileStorage ONLY despite StorageConfig.postgres_url field — StorageFactory reads config but never wires postgres. PostgresStorage is the only genuine async backend and it is systematically bypassed.

**F-PRODINFRA-H-015** (R108): memory.rs witnesses_by_action HashMap is declared and never populated — by_action lookup always returns empty Vec. Any caller querying witnesses by action_id gets no results.

**F-PRODINFRA-H-016** (R108): governance/policy.rs EscalationCondition enum defines a DSL (EnergyAbove, PersistentEnergy, SpectralDrift, ConsecutiveRejections, All, Any) but there is NO evaluator anywhere in the file. The escalation rule system is declaration-only with zero runtime enforcement.

**F-PRODINFRA-H-017** (R108): policy.rs add_approval() error-handling bug — when allowed_approvers is non-empty and an unauthorized approver attempts to sign, error returned is DuplicateApprover(approver_id) instead of an UnauthorizedApprover variant. Misclassification misleads callers.

**F-PRODINFRA-H-018** (R108): policy.rs content_hash() omits several fields — description, escalation_rules conditions (only hashes name+target_lane+priority, not the EscalationCondition variant/params), allowed_approvers list, custom_thresholds, and supersedes reference. Hash collision risk for governance integrity.

**F-PRODINFRA-H-019** (R108): policy.rs supersede() accepts successor_id parameter but does nothing with it — variable is unused, function only sets status=Superseded. The successor link parameter is silently dropped.

**F-PRODINFRA-H-020** (R108): policy.rs is a pure data aggregate — zero integration with prime-radiant coherence subsystem (energy.rs, spectral.rs, incremental.rs). ThresholdConfig defines energy thresholds conceptually mirroring the coherence lane system but no code reads CoherenceEngine or calls energy queries.

**F-PRODINFRA-H-021** (R108): governance/witness.rs verify_chain_link() error semantics bug — ID check returns PreviousNotFound when self.previous_witness != Some(previous.id), but PreviousNotFound should mean the ID is absent, not wrong. A chain referencing the wrong predecessor returns a misleading error type.

**F-PRODINFRA-H-022** (R108): governance/witness.rs fragile hash recomputation — with_actor() and with_correlation_id() recompute content_hash after mutation, but a caller who clones the returned record from add_witness() and calls with_actor() gets a different hash than what is in the builder head. Post-construction mutation creates hash/identity splits silently.

**F-PRODINFRA-H-023** (R110): cocycle.rs `is_coboundary()` degree-0 path checks global constancy — a 0-cochain is treated as a coboundary iff it is globally constant. On a disconnected complex this is incorrect: a locally-constant function (one value per connected component) is not a coboundary in H^0 but passes the constancy check if all components share the same value. False positives on disconnected simplicial complexes.

**F-PRODINFRA-H-024** (R110): cocycle.rs `SheafCoboundary.apply()` assigns edge results via `SimplexId::new(i as u64)` using loop index `i` rather than a stable edge identifier from the simplicial complex. If edges are reordered between construction and application, cocycle values are assigned to wrong simplex IDs, silently corrupting all sheaf cohomology results. No validation that edge list ordering matches the sheaf restriction map indexing.

**F-PRODINFRA-H-025** (R110): cocycle.rs `Coboundary.laplacian()` computes the Hodge Laplacian L = delta^* delta + delta delta^* correctly by formula, but `apply_adjoint()` has the dimension indexing bug (F-PRODINFRA-010). This is a SECOND parallel Laplacian implementation (alongside laplacian.rs matrix assembly) and both are incorrect via different bugs. No test validates `laplacian()` output against known eigenvalues.

**F-PRODINFRA-H-026** (R110): cocycle.rs `Cocycle.add()` calls `set()` after summing values, and `set()` drops values below 1e-10 for sparsity. Near-cancellation between two cocycles silently removes simplex entries, breaking linearity: (c1 + c2) loses entries where values nearly cancel. The result does not satisfy delta(c1 + c2) = delta(c1) + delta(c2) at those entries. The cochain group addition is not mathematically well-defined with this truncation.

**F-PRODINFRA-H-027** (R134): ruvector CLI 4 facade commands — cli.js:1625-1824 graph, router, server, and cluster commands all print "Coming Soon" or delegate to missing packages. These appear in `--help` output with no indication they are non-functional.

**F-PRODINFRA-H-028** (R134): Export vectors returns EMPTY — cli.js:1853 export command always exports an empty array regardless of database content. Data export is a critical operational capability.

**F-PRODINFRA-H-029** (R134): MCP 55-tool monolith — mcp-server.js is a 3,060-line single file with all 55 tool handlers in one switch statement. No modularization, no per-tool error isolation.

**F-PRODINFRA-H-030** (R134): CLI delegation via execSync with cold-start penalty — mcp-server.js:1348 delegates 14 tools via `npx ruvector hooks ...` adding 2-5s cold-start per invocation. Another 11 tools depend on `npx agentic-flow@alpha` with similar latency.

**F-PRODINFRA-H-031** (R134): Claude-flow CLI cold entry point — cli.js:1-156 performs zero subsystem initialization at boot. No memory, AgentDB, HNSW, or ruvector is loaded until first use. Combined with execSync delegation in MCP, first-use latency can exceed 5-10s.

**F-PRODINFRA-H-032** (R134): Claude-flow MCP resources false advertisement — mcp-server.js:109 declares `resources` capability in the MCP handshake but implements zero resource handlers. Clients may request resource listings and receive empty or error responses.

**F-PRODINFRA-H-033** (R134): Claude-flow MCP async message ordering race — mcp-server.js:35 uses async/await inside a synchronous readline callback. Multiple rapid incoming messages can interleave, producing out-of-order JSON-RPC responses.

**F-PRODINFRA-H-034** (R134): rvlite O(n) brute-force search — rvlite cli.js:283 searches all vectors via brute-force cosine similarity. Despite ruvector-core having a genuine HNSW implementation, rvlite does not use it. At scale this is unusable.

**F-PRODINFRA-H-035** (R134): rvlite ZERO integration with ruvector CLI — rvlite cli.js shares zero code, zero imports, and zero protocol compatibility with the main ruvector CLI or MCP server. Two completely independent vector stores with no migration or interop path.

**F-PRODINFRA-H-036** (R134): rvlite advertises SQL/Cypher/SPARQL but provides none — cli.js:957 query interface references structured query languages in help text but the implementation only supports vector similarity search. No query parser exists.

**F-PRODINFRA-H-037** (R134): ruvllm embedding benchmark uses hash — ruvllm cli.js:470 benchmark command computes embeddings via Math.sin of character hash codes. Benchmark results are meaningless without the native binary and do not reflect actual embedding quality or latency.

**F-PRODINFRA-H-038** (R134): ruflo missing error on unresolved @claude-flow/cli — ruflo.js:27 attempts to require('@claude-flow/cli') but produces no user-friendly error message if the package is not installed, yielding a raw Node.js MODULE_NOT_FOUND stack trace.

**F-PRODINFRA-H-038b** (R135): createEmbeddingService() fallback in controller-registry.ts returns zero-filled Float32Array for ALL embed() calls — any controller (including the critical embedding controller) silently gets useless zero vectors when the actual embedding service fails to load. No error is raised; callers receive plausible-looking Float32Array results with no quality indicator.

**F-PRODINFRA-H-038c** (R135): causalRecall controller is registered in the AgentDBControllerName type and has a createController case (line 760-768) but is never added to the controllers Map and never initialized. The controller is effectively unreachable at runtime despite appearing in the public API surface.

**F-PRODINFRA-H-038d** (R136): agentdb-mcp-server.ts agentdb_init handler re-uses the globally-initialized db object (line 908) regardless of what database path the caller specifies. Any attempt to reinitialize AgentDB against a different path silently re-uses the first-opened database.

**F-PRODINFRA-H-038e** (R136): agentdb_search session_id filter has a TODO comment — "Session ID filter would require custom query" (line 1014). The filter parameter is accepted but never applied, so session-scoped searches return results from all sessions.

**F-PRODINFRA-H-039** (R138): SONA tools (14 tools) are full facades — v3/mcp/tools/index.ts registers 14 SONA-prefixed tools but `agentic-flow/core` is NOT installed. All handlers fall to in-memory Maps with no persistence. LoRA handlers are no-ops. SONAState singleton stores all data in Maps, lost on process restart.

**F-PRODINFRA-H-040** (R138): V3 MCP server zero memory bootstrap — v3/mcp/server.ts makes zero calls to memory-initializer, AgentDB, or EmbeddingService. The ML layer is not bootstrapped by the MCP server, meaning memory-related tools are registered but have no initialized backend.

**F-PRODINFRA-H-041** (R138): V3 library MCP server SamplingManager has no auto-registered provider — v3/@claude-flow/mcp/src/server.ts SamplingManager contains a real Anthropic provider implementation but no provider is registered at startup. Sampling requests fail until a client explicitly registers a provider.

**F-PRODINFRA-H-042** (R138): V3 library MCP server zero memory/AgentDB integration — v3/@claude-flow/mcp/src/server.ts is a pure protocol layer with 14 MCP methods and 9 sub-registries but zero domain tool registration. Ships with only 4 built-in tools vs V2's 64.

**F-PRODINFRA-H-043** (R138): V2→V3 tool regression — V2 MCP server (v2/src/mcp/server.ts, 646 LOC) had 3 tool factories producing ~64 working tools with real context injection, AuthManager, and LoadBalancer with circuit breaker. V3 LOST all tool factories, AuthManager, and circuit breaker. These were removed, not refactored — no replacement exists.

**F-PRODINFRA-H-044** (R138): CLI lazy loading nullified — v3/@claude-flow/cli/src/commands/index.ts declares lazy loading but synchronously imports all 31 commands at module load time. The lazy loading infrastructure is dead code.

**F-PRODINFRA-H-045** (R138): 2 orphan CLI command files unreachable — transfer-store.ts and appliance-advanced.ts exist in the commands directory but are not imported by index.ts. These commands cannot be invoked from the CLI.

**F-PRODINFRA-H-046** (R138): ruvector setup has zero backend factory bridge — v3/@claude-flow/cli/src/commands/ruvector/setup.ts generates docker-compose.yml and init-db.sql but does NOT configure the backend factory. Setup and runtime initialization are independent subsystems with no connection — running setup does not switch the default backend from HNSWLib/sql.js to PostgreSQL.

**F-PRODINFRA-H-047** (R138): ruvector setup SQL depends on C extension functions — setup.ts init-db.sql calls ruvector_exp_map and ruvector_poincare_distance, which are PostgreSQL C extension functions. If the extension version does not match, these calls fail at init time with no fallback.

**F-PRODINFRA-H-048** (R138): ruvector setup zero relationship to memory-initializer — setup.ts and memory-initializer.ts are completely independent subsystems. The PostgreSQL schema generated by setup has no connection to the memory initialization path, meaning the database can be provisioned but never used by the runtime.

**F-PRODINFRA-H-049** (R139): release.yml `skip_tests` input bypasses ALL validation — when set to true, untested code can be published directly to crates.io. No guardrails or warnings around this escape hatch.

**F-PRODINFRA-H-050** (R139): publish-all.yml all 12 publish steps use `continue-on-error: true` — the pipeline always reports success even if every single publish step fails. A completely broken release appears green in CI.

**F-PRODINFRA-H-051** (R139): publish-all.yml version drift bug — hardcoded `0.1.31` fallback used when triggered by tag-push (inputs.version is undefined on tag-push triggers). Crates and npm packages may be published with stale version numbers that don't match the triggering git tag.

**F-PRODINFRA-H-052** (R139): release.yml tests run on ubuntu-22.04 ONLY before publishing 5-platform native binaries (linux-x64, linux-arm64, darwin-x64, darwin-arm64, win32-x64). Platform-specific bugs in macOS and Windows are never caught before release.

**F-PRODINFRA-H-053** (R139): release.yml npm test silently swallowed — `npm run test:unit || true` ensures npm test failures never block publishing. Combined with skip_tests for Rust, both Rust and JS tests can be bypassed.

**F-PRODINFRA-H-054** (R139): v3-ci.yml has 11 specific test scripts in V3 package.json (test:integration:memory, test:integration:swarm, test:integration:mcp, test:unit, test:e2e, etc.) but NONE are executed in the CI pipeline. Only a generic `pnpm test` runs, which may not invoke these scripts.

**F-PRODINFRA-H-055** (R139): V2 and V3 CI pipelines have zero cross-version regression testing — V2 uses npm/Jest, V3 uses pnpm/Vitest. No test ensures that V3 changes don't break V2 compatibility or vice versa.

**F-PRODINFRA-H-056** (R139): v3-ci.yml typecheck is non-blocking — `continue-on-error: true` on the TypeScript compilation step means type errors never fail the build. Type safety is advisory only in V3 CI.

**F-PRODINFRA-H-057** (R139): v3-ci.yml has zero security audit — unlike the V2 ci.yml which at least runs `npm audit` (albeit non-blocking), V3 CI has no security audit step at all.

**F-PRODINFRA-H-058** (R139): ruvllm_integration_tests.rs `#![cfg(feature = "ruvllm")]` gate is misleading — the feature flag gates compilation but the file has zero imports from `ruvllm::` or `prime_radiant::`. The cfg gate implies ruvllm dependency but provides zero actual ruvllm integration testing.

**F-PRODINFRA-H-059** (R139): e2e_integration_test.rs unsafe pointer cast UB at L840 — `Arc::as_ptr() as *mut MockLlmBackend` violates Rust aliasing rules by casting a shared Arc pointer to a mutable raw pointer. This is undefined behavior under Rust's memory model and may cause incorrect optimization or data races.

**F-PRODINFRA-H-060** (R139): build-native.yml test step uses `continue-on-error: true` — native module smoke test (load .node, check Object.keys()) failures cannot block the build. 5-platform binaries are published/committed to repo regardless of whether the module actually loads. Pattern consistent with F-PRODINFRA-017 CI facade.

**F-PRODINFRA-H-061** (R139): sona-napi.yml publish failures silently swallowed — `npm publish --access public || echo "Warning: $name may already exist"` makes real publish errors indistinguishable from already-published packages. All 7 platform packages + main @ruvector/sona use this pattern.

**F-PRODINFRA-H-062** (R139): Two competing backend factories coexist — agentdb root factory.ts (ID 333) has 2-tier fallback (ruvector > hnswlib), packages/agentdb factory.ts (ID 12809) has 5-tier (+RVF+sql.js). Both export `createBackend()`. No clear import resolution determines which one claude-flow loads at runtime.

**F-PRODINFRA-H-063** (R140): headless-worker-executor.ts prompt injected as raw CLI argument — spawn('claude', ['--print', prompt]) where prompt is the full worker-specific template plus injected codebase context. If prompt contains user-controlled content or filenames with shell metacharacters, there is potential for argument injection into the claude process. No sanitization of the prompt string before passing as argv.

**F-PRODINFRA-H-064** (R140): headless-worker-executor.ts does NOT use MCP protocol for worker communication. Workers are invoked as dumb one-shot subprocess calls (claude --print). There is no bidirectional message passing, no tool invocation protocol, no streaming, no session continuity across executions. Each execution is fully stateless from the subprocess perspective. This contradicts any documentation claiming MCP-based worker orchestration.

**F-PRODINFRA-H-065** (R140): headless-worker-executor.ts has ZERO imports from the memory layer — no AgentDB, no ruvector, no EmbeddingService, no memory-initializer. Worker executions cannot read from or write to AgentDB. The memory layer is completely absent from the headless execution pipeline. This extends the R20/R138 zero-bootstrap pattern: not only is memory never initialized in MCP servers, it is also never connected to the worker executor.

**F-PRODINFRA-H-066** (R140): headless-worker-executor.ts DOUBLE TIMEOUT BUG — executeClaudeCode() sets two independent setTimeout calls for the same process. The first (line 1133) sends SIGTERM then SIGKILL after 5s. The second (line 1209) fires at timeoutMs+100ms and also sends SIGTERM and calls cleanup(). The resolved flag prevents double-resolve, but the second kill attempt races with the first SIGKILL. Both timeouts check processPool.has() but the first clears the pool entry before the second fires, so the second timeout's processPool.has() check passes false and the second SIGTERM is skipped — yet the second timeout still resolves the promise even though cleanup was already called by the child 'close' event. The interaction creates non-deterministic control flow.

### MEDIUM Findings

**F-PRODINFRA-M-001** (R110): cocycle.rs `Cocycle.is_coboundary` field is always false — set at construction and never mutated by anything in the codebase. `Coboundary.is_coboundary()` does not update the struct field. Dead metadata field misleads consumers who check it directly instead of calling the operator method.

**F-PRODINFRA-M-002** (R110): cocycle.rs `Cocycle.scale()` updates cached norm arithmetically (`norm *= factor.abs()`) rather than recomputing. After repeated add/set operations with 1e-10 truncation, cached norm can diverge from actual L2 norm of stored values. The arithmetic shortcut is only exact when no truncation has occurred since the last `update_norm()` call.

**F-PRODINFRA-M-003** (R110): cocycle.rs `SheafCocycle` has no sparsity threshold in `set()` — stores all Array1<f64> values including zero vectors. Unlike scalar `Cocycle` (1e-10 threshold), all edges are accumulated. Asymmetry: `is_global_section()` uses norm-based tolerance but the data structure may include zero-residual edges that inflate `norm_squared()` via floating-point arithmetic.

**F-PRODINFRA-M-004** (R110): cocycle.rs `Coboundary.laplacian()` has zero test coverage. The Hodge Laplacian is the most mathematically significant operation in this module but given the `apply_adjoint()` dimensional bug, output is incorrect and no test will detect it.

**F-PRODINFRA-M-005** (R110): cocycle.rs `test_coboundary_on_path()` only asserts `delta_f` dimension and non-zeroness. Does not verify actual signed difference values. A coboundary computing wrong differences with correct sign pattern would pass.

**F-PRODINFRA-M-006** (R110): cocycle.rs `SheafCoboundary` constructor takes edge list independently from `Sheaf` with no validation that edges match the restriction map structure. If an edge lacks a restriction map, `sheaf.edge_residual()` returns None and the cocycle entry is silently omitted, potentially producing misleading `is_global_section()` = true for sheaves with incomplete restriction maps.

**F-PRODINFRA-M-007** (R110): cached_projections.rs `dot_product_simd` is 4-way scalar unrolling only — no SIMD intrinsics (no std::arch::x86_64, no packed_simd). Named _simd but is portable scalar loop unroll. Named contract violated; compiler auto-vectorization not guaranteed.

**F-PRODINFRA-M-008** (R134): Claude-flow MCP near-duplicates CLI MCP path — mcp-server.js contains significant code duplication with the CLI's own MCP serving code. Two separate implementations for the same protocol with divergent behavior.

**F-PRODINFRA-M-009** (R134): Zero cross-CLI code sharing — all four CLI tools (ruvector, claude-flow, rvlite, ruvllm) implement independent vector store init, independent embedding paths, and independent MCP servers with no shared library extraction. Maintenance burden scales linearly with CLI count.

**F-PRODINFRA-M-010** (R138): V3 MCP server ConnectionPool pools lightweight state wrappers — v3/mcp/server.ts ConnectionPool manages in-memory state objects, not real I/O connections (no socket pools, no database handles, no HTTP keep-alive).

**F-PRODINFRA-M-011** (R138): V3 MCP server SessionManager purely in-memory — v3/mcp/server.ts sessions have no persistence. A single `currentSession` field creates a concurrency bug for multi-client transports.

**F-PRODINFRA-M-012** (R138): V3 library MCP server completion/complete prompt-argument branch is an empty stub — v3/@claude-flow/mcp/src/server.ts handles completion requests for resource references but the prompt-argument branch returns empty results.

**F-PRODINFRA-M-013** (R138): V3 library MCP server resourceSubscriptions leak — v3/@claude-flow/mcp/src/server.ts unsubscribe handler does not call registry.unsubscribe(), leaving stale subscription callbacks.

**F-PRODINFRA-M-014** (R138): CLI commands[] array has only 19/31 commands — v3/@claude-flow/cli/src/commands/index.ts setupCommands() registers all 31 but the exported commands[] array misses 12, affecting any consumer that iterates commands[] directly.

**F-PRODINFRA-M-015** (R138): CLI silent load failures — v3/@claude-flow/cli/src/commands/index.ts swallows import errors unless DEBUG=1 is set. Failed command loads produce no user-visible warning, masking real bugs.

**F-PRODINFRA-M-016** (R138): ruvector setup hardcoded test credentials — setup.ts docker-compose.yml contains POSTGRES_PASSWORD=claude-flow-test. Users who deploy without changing this have a default password in their database.

**F-PRODINFRA-M-017** (R138): ruvector setup repeats "150x-12,500x" unsubstantiated claim — setup.ts output references the fabricated speedup number (see F-PRODINFRA-016) in generated README.md.

**F-PRODINFRA-M-018** (R138): Two competing V3 MCP servers — v3/mcp/server.ts and v3/@claude-flow/mcp/src/server.ts both implement MCPServer with the same interface but different protocol versions (2024-11-05 vs 2025-11-25), different registry counts (3 vs 9), and zero code sharing.

**F-PRODINFRA-M-019** (R139): "publish-all" misnomer — publish-all.yml only publishes math and attention crate families (8 npm packages, 4 crates.io crates). It does NOT publish ruvector-core, hnsw, ruvllm, or agentdb. The name implies comprehensive publishing but delivers partial coverage.

**F-PRODINFRA-M-020** (R139): release.yml allows `cargo publish --allow-dirty` — permits publishing crates with unexpected uncommitted file content. Modified or untracked files in the workspace are silently included in the published crate.

**F-PRODINFRA-M-021** (R139): release.yml uses static release notes template — "What's New" section content never changes between releases. Release notes are decorative rather than informative.

**F-PRODINFRA-M-022** (R139): v3-ci.yml publish job re-builds instead of downloading artifacts — the publish job runs a full build rather than consuming build artifacts from the preceding matrix build job, wasting CI compute and risking build non-determinism between test and publish.

**F-PRODINFRA-M-023** (R139): e2e_integration_test.rs suppresses 12 lint/warning categories via `#[allow]` — including unused variables, dead code, and unused mut. The broad suppression masks potential real issues in the test code.

**F-PRODINFRA-M-024** (R139): agentic-flow docker-compose is a 9-line demo config — single `agents` service, 1 replica, hardcoded TOPIC="upgrade checkout flow", no volumes, networking, healthcheck, or resource limits. Only suitable for quick demos.

**F-PRODINFRA-M-025** (R139): build-native.yml graph-node build DISABLED — ruvector-graph-node has compilation issues (noted as "see PR #15"). The graph NAPI bridge cannot be built. Only ruvector-core NAPI bridge is operational.

**F-PRODINFRA-M-026** (R139): build-native.yml manual npm publish only — publish step explicitly removed with comment "packages are published manually". No automated npm publish pipeline for ruvector-core native binaries.

**F-PRODINFRA-M-027** (R139): sona-napi.yml `sleep 30` for npm propagation — post-publish test waits a fixed 30 seconds for npm registry propagation. No retry logic. Fragile timing assumption.

**F-PRODINFRA-M-028** (R139): docker-compose.agent.yml is demo not deployment — all 7 services use Docker profiles with hardcoded example tasks ("Create a 3-step plan...", "Implement retry logic..."). No volume mounts, health checks, or resource limits. Suitable for demos only.

**F-PRODINFRA-M-029** (R139): ruvector core barrel exports ALL 23 modules unselectively — npm/packages/ruvector/src/core/index.ts re-exports everything including dead code (sona-wrapper 62-68% dead per R137). Broken modules leak to consumers via `@ruvector/core`.

**F-PRODINFRA-M-030** (R140): headless-worker-executor.ts simpleGlob() is a homegrown recursive directory scanner with known limitations — only handles *.ext, prefix*, *suffix, and ** patterns. Does not support brace expansion {a,b}, negation !, or complex multi-wildcard patterns. The ** handling silently drops files when remainingParts.length == 1 and the current entry is a file (lines 1028-1033), causing legitimate matches to be missed.

**F-PRODINFRA-M-031** (R140): headless-worker-executor.ts MODEL_IDS maps to hardcoded version strings — sonnet → claude-sonnet-4-5-20250929, haiku → claude-haiku-4-5-20251001, opus → claude-opus-4-6. No dynamic model resolution or latest-alias support. These strings will break when Anthropic deprecates those version identifiers.

**F-PRODINFRA-M-032** (R140): headless-worker-executor.ts logExecution() writes the full prompt (including injected codebase context) to disk at .claude-flow/logs/headless/{executionId}_prompt.log. The audit worker includes **/.env* in its contextPatterns, so .env file contents are written to both the Anthropic API and local log files. No log rotation or cleanup is implemented.

**F-PRODINFRA-M-033** (R140): headless-worker-executor.ts 6 of 8 worker types are enabled=false by default (document, ultralearn, refactor, deepdive, predict, and partially benchmark/preload). The enabled flag is static with no runtime toggle mechanism in this file. The worker daemon must wire a separate enable/disable command path.

---

## Section 4: Positives

- **learning.rs** (R36) is genuinely innovative — adaptive weight formula rewards both reliability AND effectiveness
- **Confidence scoring** (1-1/(1+n/10)) is mathematically sound — asymptotic to 1.0 with more data
- **StrategyRegistry** weight-based selection is production-ready architecture
- **check_health()** pipeline is well-designed: collect → detect → filter → remediate
- **Rollback logic** is genuine: checks reversibility, calls rollback, logs failures
- **health-monitor.ts** (99%) is BEST monitoring code — linear regression, MPC self-healing, EventEmitter coordination
- **path-security.ts** (88-92%) implements real OWASP path traversal prevention — just needs integration
- **serving/request.rs** (88-92%): clean tripartite lifecycle (InferenceRequest/RunningRequest/CompletedRequest) matching vLLM/Orca pattern. CompletedRequest captures full latency breakdown (prefill/decode/wait/total) — correct data for P50/P99 SLA reporting
- **serving/mod.rs** (92%): 4 integration tests verify all major sub-systems compose correctly. NoopBackend decoupling is real and clean
- **serving/**: MODULE COMPLETE (6/6 DEEP, ~90% avg) — BEST ruvllm subsystem. Full pipeline from request admission through scheduling, KV cache management, continuous batching, and completion
- **prime-radiant/storage/postgres.rs**: REAL async sqlx/PgPool — genuine production postgres, not a stub. Only genuinely async storage backend in the crate
- **prime-radiant/governance/repository.rs**: Kahn topological sort for lineage cycle detection is textbook correct. parking_lot::RwLock use throughout shows performance awareness. Dual HashMap for O(1) witness chain head lookup is efficient
- **prime-radiant/governance/**: MODULE COMPLETE (5/5 DEEP, ~88% avg). Policy state machine (Draft→Pending→Active→Superseded/Revoked), Blake3 hash-chained witnesses, DAG lineage with tamper detection — architectural ambition is sound
- **prime-radiant/storage/**: MODULE COMPLETE (4/4 DEEP, ~86% avg). Three storage backends (file, memory, postgres) with clean trait boundaries
- **governance/policy.rs**: PolicyBundleBuilder consuming builder pattern with compile-time validation deferred to build() is idiomatic Rust. Blake3 sorted-key hash for deterministic HashMap hashing is correct
- **governance/witness.rs**: GENUINE Blake3 hash chain covering 10+ fields with tamper detection test. ComputeLane enum (Reflex/Retrieval/Heavy/Human) is a sound architectural signal aligned with ADR-014
- **governance/lineage.rs**: hash-of-hashes Merkle-like structure (EntityRef.content_hash() nested in LineageRecord.compute_content_hash()) is consistent with coherence module fingerprinting. Dependency sorting before hashing is correct
- **cached_projections.rs** (R110): GENUINE, non-facade OT utility — zero hash shortcuts, zero imports from ruvector-core, 3 substantive tests. WindowCache pre-sorts key projections per direction enabling O(1) sorted-order access for sliced Wasserstein. `project_into()` writes into caller-supplied buffer (zero-allocation hot path) with ergonomic `project()` counterpart — good design for hot-loop usage in sliced_wasserstein.rs
- **cocycle.rs** (R110): `Coboundary.apply()` correctly implements signed simplicial coboundary — delta(f)(sigma) = sum_{i=0}^{n+1} (-1)^i f(d_i sigma) — using `boundary()` face/sign pairs from simplex.rs. `SheafCoboundary.apply()` correctly implements graph sheaf coboundary per Hansen & Ghrist 2021. `Cocycle.inner_product()` correctly computes sparse L2 inner product on n-cochains. Integration with SimplicialComplex infrastructure is genuine and exercised in tests.
- **ruvector CLI uses VectorDB** (R134): The main ruvector CLI correctly wires VectorDB (the working vector store path) rather than RuVectorBackend (which is broken). Vector operations actually function when used via CLI.
- **ONNX embed commands are REAL** (R134): ruvector CLI embed commands using the ONNX path provide genuine neural embeddings when the onnx-embedder is available, not hash fallbacks.
- **55-tool MCP server is genuine and functional** (R134): Despite being a monolith, the ruvector MCP server's 55 tools are real implementations with working delegation chains. This is the largest functional MCP server in the ecosystem.
- **rvlite hyperbolic geometry correct** (R134): rvlite implements Poincare disk and Lorentz models with mathematically correct distance/embedding operations. Hyperbolic search is genuine.
- **rvlite genuine WASM integration** (R134): rvlite has real WASM bindings to SONA and Attention modules — not stubs or facades.
- **agentdb-mcp-server.ts EmbeddingService initialized** (R136): DIRECTLY CONTRADICTS R20 — EmbeddingService IS initialized with Xenova/all-MiniLM-L6-v2 (384-dim) via top-level await. This is the one path in the ecosystem that successfully loads a real embedding model at startup.
- **agentdb-mcp-server.ts production-grade lifecycle** (R136): keepAlive setInterval, auto-save every 5 minutes, SIGTERM/SIGINT handlers, proper shutdown sequence — well-engineered operational lifecycle for a long-running MCP server process.
- **agentdb-mcp-server.ts batch operations** (R136): skill_create_batch, reflexion_store_batch, agentdb_pattern_store_batch all handle arrays with proper iteration and error aggregation. Rare example of genuine batch API in the ecosystem.
- **ControllerRegistry dependency injection** (R135): All 8 controllers are instantiated with proper DI patterns — each controller type receives only the dependencies it needs, enabling clean testing and substitution.
- **ruvllm model downloading is REAL** (R134): The ruvllm CLI's HuggingFace model download command performs actual HTTP fetches with progress bars and file validation. One of the few fully functional paths without the native binary.
- **V3 MCP server AJV schema validation** (R138): v3/mcp/server.ts ToolRegistry uses AJV for JSON Schema validation of tool inputs — production-grade input validation, correctly rejecting malformed requests before tool execution.
- **V3 library MCP server 9 sub-registries** (R138): v3/@claude-flow/mcp/src/server.ts has 9 dedicated sub-registry modules (ToolRegistry, ResourceRegistry, PromptRegistry, SamplingManager, RootsManager, LoggingManager, CompletionHandler, ProgressManager, NotificationManager) totaling 3,040 LOC — clean separation of concerns with real implementations.
- **V3 library MCP server Anthropic sampling provider** (R138): SamplingManager contains a real Anthropic API provider implementation with proper request/response mapping — genuinely functional once a provider is registered.
- **V2 MCP server real tool context injection** (R138): v2/src/mcp/server.ts passes orchestrator, swarmCoordinator, and other real runtime objects into tool factories — tools get genuine backend access, not stubs.
- **V2 ruv-swarm integration is REAL** (R138): V2 MCP server's ruv-swarm tool bridge uses `npx ruv-swarm` CLI — a genuine external process delegation pattern that actually invokes the Rust swarm binary.
- **ruvector setup PostgreSQL schema production-quality** (R138): setup.ts generates 8 tables, 6 HNSW indices, and 7 SQL functions with proper indexing and constraint definitions. The schema itself is well-designed even though the bridge to runtime is missing.
- **V3 tool registry 6 self-contained groups** (R138): Swarm, memory, config, task, system, and session tool groups have no external backend dependencies — they operate on in-process state and are genuinely functional.
- **V3 tool registry 4 backend-connected groups** (R138): Hooks (ReasoningBank), worker (WorkerDispatch), federation (FederationHub), and agent (SecureLogger) tool groups connect to real backend implementations.
- **Session tools genuine file-based persistence** (R138): Among all V3 tool groups, session tools have real file-system persistence — sessions survive process restarts.
- **CLI ruvectorCommand confirmed real** (R138): v3/@claude-flow/cli/src/commands/index.ts ruvectorCommand has 8 genuine subcommands connecting to the ruvector subsystem.
- **release.yml 25-crate topological sort** (R139): ruvector release pipeline publishes 25 Rust crates in a well-ordered topological sort with rate limiting between publishes. The dependency ordering ensures crates are available on crates.io before their dependents attempt to build. This is the most genuine CI infrastructure in the ecosystem.
- **publish-all.yml 5-platform NAPI-RS build matrix** (R139): Builds native bindings for 5 platforms (linux-x64, linux-arm64, darwin-x64, darwin-arm64, win32-x64) via NAPI-RS. The cross-compilation matrix is production-quality with proper target configuration per platform.
- **V3 Docker deployment most production-ready** (R139): v3/docker-compose.yml implements 3 profiles (lite <100MB / full ~800MB / workers) with non-root user, dumb-init signal handling, aggressive dependency pruning in lite mode, Redis queue for workers, and Prometheus+Grafana monitoring. Branded "Ruflo v3.5" with `ruflo` CLI binary. ruvector is included as an npm dependency confirming the NAPI/npm linking model.
- **e2e_integration_test.rs genuinely tests production algorithms** (R139): Despite being mock-only for backends, the test file genuinely exercises production functions: softmax, log_softmax, top_k_filter, top_p_filter, sample_from_probs, TwoTierKvCache numerical operations, SpeculativeStats, and ServingEngine. These are real production codepaths being validated.
- **ruvllm_integration_tests.rs mock algorithms are genuine** (R139): Mock implementations contain real algorithmic content — cosine similarity computation, cryptographic hash chains for witness logs, sigmoid mapping for pattern-to-restriction bridges, and negation detection. The algorithms are sound even though they don't match the production APIs they model.
- **ruvector distributed test architecture well-designed** (R139): The 5-node Raft cluster configuration is architecturally sound: 3 ports per node (7000 Raft, 8000 cluster, 9000 replication), 64 shards, replication factor 3. The test-runner service runs REAL `cargo test -p ruvector-raft -p ruvector-cluster -p ruvector-replication` against the actual crate test suites.
- **Proper secret management in crates-io** (R139): release.yml uses GitHub environment protection for the crates-io secret, requiring environment approval before Rust crate publishing. This is correct operational security practice.
- **sona-napi.yml 7-platform build with universal macOS binary** (R139): Most comprehensive NAPI CI in the repo — builds for linux-x64-gnu/musl, linux-arm64-gnu, darwin-x64/arm64, win32-x64/arm64-msvc, then creates universal macOS binary via `lipo`. Production-grade cross-platform distribution.
- **sona-napi.yml post-publish smoke test** (R139): Tests `npm install @ruvector/sona@latest` + `require()` + check `SonaEngine` export on 3 platforms (ubuntu, macos, windows). Confirms the published npm package is actually installable and loadable.
- **build-native.yml genuine cross-compilation infrastructure** (R139): 5-platform NAPI build with proper cross-compilation tools (aarch64-linux-gnu-gcc for ARM64), Rust cache per target, and NAPI-RS CLI. The Rust→Node bridge is real CI infrastructure, not a stub.
- **agentdb root factory.ts clean detection pattern** (R139): Dynamic import detection with graceful scoped-package fallback (try `ruvector`, then `@ruvector/core`), optional GNN/Graph detection, lazy HNSWLib loading to avoid build-tool failures, and helpful error messages with install commands. Well-engineered factory pattern.
- **docker-compose.agent.yml clean profile organization** (R139): Each of 7 agent types is independently runnable via Docker profiles. Clean separation allows testing individual agents without spinning up the full stack.
- **headless-worker-executor.ts real subprocess execution** (R140): executeClaudeCode() is a genuine, working subprocess executor — not a mock or stub. spawn('claude', ['--print', prompt]) with configurable env (CLAUDE_CODE_HEADLESS, CLAUDE_CODE_SANDBOX_MODE, ANTHROPIC_MODEL), stdout/stderr capture, graceful SIGTERM→SIGKILL termination, and event emission for monitoring. The process pool (maxConcurrent=2), pending queue, and context caching with TTL are all genuine functional implementations.
- **headless-worker-executor.ts typed public API** (R140): Well-structured TypeScript with clean separation of 12 exported types/interfaces, a rich worker config registry (HEADLESS_WORKER_CONFIGS, LOCAL_WORKER_CONFIGS), utility functions (isHeadlessWorker, isLocalWorker, getModelId, getWorkerConfig), and a concrete class extending EventEmitter. All worker configs include explicit timeout, sandbox mode, model, and context pattern settings — good operational configuration structure.

---

## Section 5: Subsystem Analyses

### 5.1 ruvvector-postgres Healing (R36) — 5 files, 4,070 LOC, ~76% avg

Self-healing database infrastructure. Real learning layer (92-95%), stub detection (metric queries return empty), stub execution (all 5 strategies log-only). Architecture is sound — learning.rs would genuinely improve healing decisions if metric collection and strategy execution were implemented.

```
detector.rs  →  engine.rs  →  strategies.rs
(find problems)  (coordinate)   (fix problems — LOG ONLY)
     ↑               ↓              ↓
learning.rs  ←  worker.rs   ←  (results)
(GENUINE)      (schedule/run)
```

**Key Pattern**: Functioning brain (learning.rs), no eyes (detector.rs returns empty), no hands (strategies.rs logs but doesn't act).

### 5.2 AgentDB Health Monitoring (R48) — 2 files, 951 LOC, ~94% avg

health-monitor.ts (99%) is the ONLY fully-integrated production-infra file in AgentDB CLI. Real OS/V8 metrics, linear regression memory leak detection (slope via least-squares on last 10 samples), MPC self-healing with 4 strategies (GC, workload reduction, component restart, abort), EventEmitter coordination. path-security.ts (88-92%) is OWASP-compliant but completely orphaned — zero imports found.

### 5.3 ruvllm Serving (R106-R107) — MODULE COMPLETE (6/6 DEEP, ~90% avg)

The serving subsystem is the BEST-validated subsystem in ruvllm and the strongest production-infra component overall. All 6 files are DEEP:

| File | LOC | Real% | Role |
|------|-----|-------|------|
| **engine.rs** | 1,302 | ~90% | Main serving loop, ServingEngine orchestrator |
| **scheduler.rs** | 840 | ~88% | ContinuousBatchScheduler, priority scheduling |
| **batch.rs** | 501 | **90-95%** | vLLM/Orca batch management, preemption (R106 BEST) |
| **kv_cache_manager.rs** | 606 | **88-92%** | Paged KV cache, block allocation (R106) |
| **request.rs** | 473 | **88-92%** | Request lifecycle, chunked prefill, TokenOutput (R107) |
| **mod.rs** | 348 | **92%** | Orchestration + 4 integration tests (R107) |

Architecture mirrors Kwon et al. 2023 (vLLM continuous batching). NoopBackend decoupling confirmed — serving layer can run without GPU backend. Priority enum (Low/Normal/High/Critical) with Ord derivation feeds priority-queue scheduling. RequestState enum covers 6 states including Preempted — preemption is architecturally modeled.

One functional gap: stop_sequences are silently ignored (should_stop() reduces to is_complete() — token count only). EOS token check explicitly deferred (tokenizer not wired). Two legacy fields: kv_cache_slot (flat index, pre-paged) coexists with block_table (paged attention, correct), suggesting transition in progress.

### 5.4 prime-radiant Storage (R107-R108) — MODULE COMPLETE (4/4 DEEP, ~86% avg)

| File | LOC | Real% | Role |
|------|-----|-------|------|
| **postgres.rs** | 1,082 | **82-87%** | REAL async sqlx/PgPool — only async backend |
| **file.rs** | 804 | **85-90%** | WAL + dual-format (bincode/JSON) + blake3 |
| **memory.rs** | 731 | **88-92%** | Volatile backend, witnesses_by_action unpopulated |
| **mod.rs** | 576 | **82-86%** | Traits + HybridStorage = FileStorage ONLY |

**Systemic failure**: StorageFactory ignores postgres_url. HybridStorage wires only FileStorage despite the name and configuration. The only genuinely async backend (postgres.rs) is systematically bypassed at the factory level.

postgres.rs is the sole genuine production backend — real async PgPool, JSONB governance data, feature-gated (sqlx 0.8 + runtime-tokio). But it has zero sheaf-theoretic integration, no blake3 hash chains, no pgvector (brute-force cosine similarity in Rust), a race condition in store_witness() sequence generation, and a dead AsyncGraphStorageAdapter that implements neither GraphStorage nor GovernanceStorage traits.

file.rs CRITICAL WAL bug: the committed=false flag written at WAL-write time is not flipped to committed=true on successful flush. Deletions written to WAL are non-durable — the commit step is missing or incorrectly wired, so crash recovery replays only WAL entries where committed=true, silently losing delete operations.

### 5.5 prime-radiant Governance (R108) — MODULE COMPLETE (5/5 DEEP, ~88% avg)

| File | LOC | Real% | Role |
|------|-----|-------|------|
| **policy.rs** | 970 | **82-87%** | PolicyBundle state machine, Blake3, CRITICAL security gaps |
| **repository.rs** | 1,062 | **62%** | Traits only — no production backends, no async |
| **mod.rs** | 439 | **88-92%** | Pure re-export, Hash/Timestamp/Version shared types |
| **witness.rs** | 723 | **85-90%** | Blake3 hash chain, GateDecision, ComputeLane |
| **lineage.rs** | 873 | **85-90%** | DAG provenance, hash-of-hashes tamper detection |

**CRITICAL security gap (policy.rs)**: The multi-signature governance mechanism provides zero cryptographic assurance. add_approval() checks approver_id list membership and deduplication only — it never verifies the ApprovalSignature.signature bytes against any public key or hash. The production-visible ApprovalSignature::placeholder() constructor (zero-byte [0u8; 64]) can activate real policy bundles identically to valid ed25519 signatures.

**EscalationCondition DSL (policy.rs)**: Defines a rich DSL (EnergyAbove, PersistentEnergy, SpectralDrift, ConsecutiveRejections, All, Any) with no evaluator anywhere in the file. Rules sort by priority but are never tested against live system state. Declaration-only with zero runtime enforcement.

**Coherence isolation**: Neither policy.rs nor any governance sibling reads CoherenceEngine, calls energy queries, or references the sheaf substrate from the coherence module (energy.rs, spectral.rs). ThresholdConfig defines energy thresholds conceptually aligned with the coherence lane system but no code wires the two together.

**What works**: PolicyBundle Draft→Pending→Active→Superseded/Revoked state machine with immutability invariant for Active bundles. Blake3 content hashing with sorted HashMap keys for determinism. WitnessRecord Blake3 hash chain covering 10+ fields with tamper detection. LineageRecord hash-of-hashes Merkle-like provenance. Kahn topological sort for lineage DAG cycle detection in repository.rs. 8-9 unit tests per file with meaningful behaviorally-targeted coverage.

### 5.6 prime-radiant Cohomology + ruvector-attention OT Utilities (R110) — 2 files, 713 LOC

Two files from adjacent mathematical infrastructure tagged production-infra due to their role in the sheaf-theoretic reasoning layer.

**cocycle.rs** (471 LOC, DEEP, ~75-80%): Implements four algebraic topology structures — `Cocycle` (sparse n-cochain on simplicial complex), `Coboundary` (coboundary operator + Hodge Laplacian), `SheafCocycle` (vector-valued cochain on graph), `SheafCoboundary` (sheaf coboundary operator). The classical coboundary operator (`Coboundary.apply()`) is mathematically correct and exercises simplex.rs boundary infrastructure. `SheafCoboundary.apply()` correctly implements the Hansen-Ghrist graph sheaf coboundary formula. The file has two CRITICAL bugs: (1) `is_coboundary()` for degree > 0 always returns false, corrupting all H^n for n >= 1; (2) `apply_adjoint()` iterates the wrong dimension, making the Hodge Laplacian incorrect — compounded by R109 laplacian.rs eigensolver bugs.

**cached_projections.rs** (242 LOC, DEEP, ~88-92%): A pure sliced-Wasserstein optimal transport utility. `ProjectionCache` generates L2-normalized random directions; `WindowCache` pre-sorts key vectors per projection for O(1) sorted-order access during sliced Wasserstein computation. CDF histogram approximation avoids full sort on query. Three substantive tests (unit-norm directions, shape correctness, CDF sum-to-1). Genuine, self-contained, no hash shortcuts. The only weakness: `dot_product_simd` is 4-way scalar unrolling, not actual SIMD.

### 5.7 CLI Entrypoints (R134) — 7 files, ~13,450 LOC

R134 analyzed all four CLI front doors in the ruvnet ecosystem for the first time, revealing the user-facing layer through which all external consumers interact with the underlying Rust crates and JS libraries.

**Architecture Overview:**

```
User Commands
    │
    ├── ruvector CLI (7,357 LOC) ──→ VectorDB (working) + Intelligence (hash fallback)
    │       └── ruvector MCP (3,007 LOC) ──→ 55 tools via library/execSync/agentic-flow
    │
    ├── claude-flow CLI (156 LOC) ──→ cold dispatcher, lazy-loads everything
    │       └── claude-flow MCP (189 LOC) ──→ near-duplicate, async race
    │       └── ruflo (50 LOC) ──→ pure rebrand
    │
    ├── rvlite CLI (1,686 LOC) ──→ own vector store, O(n) search, WASM (independent)
    │
    └── ruvllm CLI (1,005 LOC) ──→ native .node required; JS = facades
```

**ruvector CLI** (7,357 LOC, ~72-78%): The largest and most functional CLI. Genuine commands include: vector CRUD (add/search/delete/info), ONNX embedding, hooks pipeline, batch operations, stats, and database management. Uses VectorDB directly (not the broken RuVectorBackend). Hash embedding confirmed at CLI layer: Intelligence.embed() checks if the engine is already initialized; since most paths pass skipEngine:true, the 64-dim hash fallback is the common case. Four commands are facades: graph ("Coming Soon"), router, server, cluster. Export always returns empty array.

**ruvector MCP server** (3,007 LOC, ~78-82%): Registers 55 tools — the largest MCP server in the ecosystem. Three delegation strategies: (1) direct library calls for vector operations, (2) execSync to `npx ruvector hooks` for 14 hook tools (2-5s cold-start per call), (3) execSync to `npx agentic-flow@alpha` for 11 agentic tools. The sanitizeShellArg() function strips shell metacharacters from query strings, destroying legitimate SQL/Cypher/SPARQL syntax while failing to prevent actual SQL injection. 9th instance of the hash-embedding pattern.

**claude-flow CLI** (156 LOC, N/A): A pure dispatcher that requires('@claude-flow/cli') and passes through. Zero subsystem initialization at boot — no memory, no AgentDB, no HNSW, no ruvector loaded until first use. This is architecturally intentional (fast startup) but means every first operation incurs full cold-start latency.

**claude-flow MCP server** (189 LOC, ~72-78%): Near-duplicate of the CLI's MCP code path. Declares `resources` capability but implements zero resource handlers. Uses async/await inside a synchronous readline callback, creating an ordering race under rapid message delivery.

**ruflo** (50 LOC, ~70-75%): Pure rebrand of claude-flow. Resolves @claude-flow/cli and delegates. If the package is not installed, produces a raw MODULE_NOT_FOUND error with no user-friendly message.

**rvlite CLI** (1,686 LOC, ~80-85%): Fully independent alternative to ruvector. Has its own vector store (SQLite-backed), genuine WASM integration (SONA + Attention modules), and correct hyperbolic geometry (Poincare disk + Lorentz model). However: uses O(n) brute-force search (no HNSW), advertises SQL/Cypher/SPARQL without implementing any query parser, and shares zero code with the main ruvector CLI. No migration or interop path between rvlite and ruvector stores.

**ruvllm CLI** (1,005 LOC, ~72-78%): Entirely dependent on a native .node binary addon. Without it, all core commands (query, generate, route, embed) return hardcoded or hash-based values. Training is explicitly simulated ("Training loss (simulated)"). The embedding benchmark computes Math.sin of character hash codes, making results meaningless. Only HuggingFace model downloading works without native support.

**Key Verdict**: The CLI layer confirms the systemic hash-embedding pattern end-to-end: from CLI user input, through Intelligence.embed(), through IntelligenceEngine (if initialized, which it usually is not), down to the 64-dim hash fallback. The user-facing tools are more functional than the Rust subsystem analysis might suggest (VectorDB works, ONNX works when available), but the four CLIs share zero code and each maintains independent vector stores, embedding paths, and MCP implementations.

### 5.7b V3 Memory Layer (R135) — 2 files, ~2,064 LOC

R135 analyzed the V3 @claude-flow/memory package — the nominal "brain" of claude-flow V3 that was supposed to connect AgentDB, HNSW, and the MCP toolchain.

**AgentDBAdapter (1,038 LOC, ~35-45%)**: The most misleadingly named file in the V3 codebase. Despite its name and doc comments, it does NOT adapt AgentDB. Storage is a plain `Map<string, MemoryEntry>` (line 97). Zero imports from any AgentDB package. The three factory functions (`createInMemoryService`, `createPersistentService`, `createHybridService`) all return a `UnifiedMemoryService` with the same in-memory Map backend. `loadFromDisk()` and `saveToDisk()` (lines 872-881) emit events and immediately return — no disk I/O. The R20 root cause (EmbeddingService not initialized) persists in V3: `embeddingGenerator` is an optional field defaulting to undefined, and `createInMemoryService()` + `createPersistentService()` create services WITHOUT embedding generators. A binary quantizer (line 956) packs 32 floats into 1 Float32Array element via bitwise OR on float values, but the distance function uses standard float comparison — the quantized values are never decoded correctly.

**ControllerRegistry (1,026 LOC, ~55-65%)**: Manages 8 controller types (episodic, semantic, procedural, working, causal, meta, temporal, contextual) plus the embedding service. The path traversal validation for storage paths is a no-op: `path.resolve()` normalizes ".." before the `includes("..")` check, so `../../etc/passwd` passes validation. `createEmbeddingService()` fallback returns a zero-filled Float32Array stub with no error — callers silently receive useless vectors. `causalRecall` is declared in the type but never added to the controllers Map. `healthCheck()` only inspects the init-time error flags, never probing actual controller liveness.

### 5.7c AgentDB MCP Server (R136) — 1 file, 2,368 LOC

**agentdb-mcp-server.ts (2,368 LOC, ~75-80%)**: The standalone AgentDB MCP server that DIRECTLY CONTRADICTS the R20 finding that EmbeddingService is never initialized. In this file, EmbeddingService IS initialized with Xenova/all-MiniLM-L6-v2 (384-dim, transformers provider) at module load time via top-level await. This is Pipeline 1 (all-JS, no Rust bridge). Key positives: all 8 controllers instantiated with proper DI, batch operations with full array handling, production-grade lifecycle management (keepAlive setInterval, 5-min auto-save, SIGTERM/SIGINT handlers). Key failures: (1) tool count discrepancy — documentation claims 32, actual array has 28; (2) causal_add_edge hardcodes fromMemoryId=0 and toMemoryId=0 for ALL edges — the causal graph data model is broken; (3) agentdb_init re-uses the global db regardless of caller-specified path; (4) agentdb_search session_id filter has a TODO and is never applied.

### 5.8 V3 MCP Tool Chain (R138) — 6 files, ~4,200 LOC

R138 traced the V3 MCP tool chain from tool registration through server bootstrap to CLI command dispatch, and compared it against the V2 predecessor. This is the first end-to-end analysis of the MCP infrastructure evolution.

**Architecture Overview:**

```
V3 Tool Chain (two competing servers)
    │
    ├── v3/mcp/tools/index.ts (445 LOC) ──→ getAllTools(): 82 tools from 12 groups
    │       ├── 6 self-contained groups (swarm, memory, config, task, system, session)
    │       ├── 4 backend-connected groups (hooks, worker, federation, agent)
    │       └── 2 SONA groups (14 tools) ──→ ALL FACADES, fabricated metrics
    │
    ├── v3/mcp/server.ts (792 LOC) ──→ JSON-RPC 2.0, protocol 2024-11-05
    │       └── dynamic import('./tools/index.js') → getAllTools()
    │       └── AJV schema validation (production-grade)
    │       └── ConnectionPool (in-memory state, not real I/O)
    │       └── SessionManager (in-memory, single currentSession)
    │
    └── v3/@claude-flow/mcp/src/server.ts (1,134 LOC) ──→ MCP 2025-11-25
            └── 14 MCP methods, 9 sub-registries (3,040 LOC)
            └── Only 4 built-in tools (protocol shell)
            └── SamplingManager (real Anthropic provider, not auto-registered)
            └── COMPETES with v3/mcp/server.ts (zero code sharing)

V2 Predecessor (SIMPLER BUT MORE COMPLETE)
    │
    └── v2/src/mcp/server.ts (646 LOC) ──→ protocol 2024-11-05
            └── 3 tool factories: createClaudeFlowTools(30) + createSwarmTools(19) + createRuvSwarmTools(15)
            └── AuthManager + LoadBalancer with circuit breaker ──→ ALL LOST IN V3
            └── Real tool context injection (orchestrator, swarmCoordinator)
            └── ruv-swarm: REAL CLI bridge via `npx ruv-swarm`

CLI Command Registry
    │
    └── v3/@claude-flow/cli/src/commands/index.ts (399 LOC) ──→ 31 commands
            └── Lazy loading DEAD (synchronous imports)
            └── 2 orphans: transfer-store.ts, appliance-advanced.ts
            └── ruvectorCommand: 8 genuine subcommands

PostgreSQL Scaffolding (DISCONNECTED)
    │
    └── v3/@claude-flow/cli/src/commands/ruvector/setup.ts (784 LOC)
            └── Generates docker-compose.yml + init-db.sql + README.md
            └── 8 tables, 6 HNSW indices, 7 SQL functions
            └── ZERO bridge to backend factory or memory-initializer
```

**V2→V3 Evolution Summary:**

| Dimension | V2 | V3 (internal) | V3 (library) |
|-----------|----|----|-----|
| LOC | 646 | 792 | 1,134 (+3,040 registries) |
| Protocol version | 2024-11-05 | 2024-11-05 | 2025-11-25 |
| Tool count | ~64 (3 factories) | 82 (12 groups) | 4 (built-in only) |
| Tool factories | 3 (real context injection) | 0 (direct registration) | 0 (protocol shell) |
| Auth | AuthManager | None | None |
| Load balancing | LoadBalancer + circuit breaker | None | None |
| Memory init | None | None | None |
| Schema validation | None | AJV (production-grade) | Per-method validation |

**Key Finding — Zero Memory Bootstrap**: Neither V2 nor any V3 MCP server bootstraps memory, AgentDB, or EmbeddingService. Memory tools exist as registered handlers in the V3 internal server but the backend they depend on is never initialized. This extends the R20/R137 finding (EmbeddingService never initialized in the claude-flow bridge) to the entire MCP server layer: no MCP entry point in the ecosystem initializes the ML subsystem.

**SONA Facade Analysis**: The 14 SONA tools in v3/mcp/tools/index.ts form the largest single facade group in the ecosystem. They register tool schemas with proper JSON-RPC metadata but: (1) `agentic-flow/core` is NOT installed so all imports fail; (2) handlers fall to in-memory Maps with no persistence; (3) LoRA handlers are no-ops; (4) the speedup calculation is a tautological multiplication (F-PRODINFRA-016). The SONA tools inflate the V3 tool count from 68 genuine to 82 total.

**ruvector Setup Island**: setup.ts generates production-quality PostgreSQL schema (8 tables with proper indexing, 6 HNSW indices, 7 SQL functions including hyperbolic geometry operations) but exists as a completely disconnected island. Running `claude-flow ruvector setup` provisions infrastructure that no runtime path consumes — the backend factory defaults to HNSWLib/sql.js regardless of whether PostgreSQL is available.

### 5.9 CI/CD Pipeline Architecture (R139) — 4 files, ~1,558 LOC

Four CI/CD pipelines serve three packages with fundamentally different quality gates:

```
ruvector CI (GENUINE but bypassable)
    │
    ├── release.yml (621 LOC) ──→ 7-job DAG: validate → build → publish
    │       ├── 25 Rust crates in topological sort (GENUINE)
    │       ├── 4 WASM packages (ruvector-wasm, gnn-wasm, graph-wasm, tiny-dancer-wasm)
    │       ├── 4 npm packages (MANUAL publish)
    │       └── BYPASS: skip_tests=true, `|| true` on npm tests
    │
    └── publish-all.yml (552 LOC) ──→ 6-phase DAG: validate → build → publish
            ├── MISNOMER: only math+attention families (8 npm, 4 crates)
            ├── 5-platform NAPI-RS matrix (GENUINE)
            └── ALL 12 steps: continue-on-error: true (always green)

claude-flow CI (FACADE — tests never block)
    │
    ├── ci.yml (228 LOC) ──→ V2 pipeline (npm/Jest)
    │       ├── continue-on-error on tests + typecheck + audit
    │       ├── Only lint can fail
    │       └── Deploy job: placeholder (echo + exit)
    │
    └── v3-ci.yml (157 LOC) ──→ V3 pipeline (pnpm/Vitest)
            ├── 11 test scripts exist, NONE in CI
            ├── No security audit
            └── Publish re-builds instead of downloading artifacts
```

**Key Finding**: No CI pipeline in the ecosystem functions as a genuine quality gate. The ruvector release pipeline comes closest with real topological crate ordering and environment-protected secrets, but `skip_tests` and `|| true` undermine the entire validation chain. Claude-flow CI is structurally a facade — it exists solely for the green badge, providing zero evidence of code correctness. V2 and V3 CI pipelines share no test infrastructure and have no cross-version regression testing.

### 5.10 Rust Integration Testing (R139) — 2 files, 2,928 LOC

Both Rust "integration test" files are mock-only and test zero cross-crate composition:

| File | LOC | Tests | Genuine Imports | Mock Coverage |
|------|-----|-------|-----------------|---------------|
| **e2e_integration_test.rs** (ruvllm) | 1,535 | 36 (3 #[ignore]) | `ruvllm::serving::*`, `ruvllm::sampling::*` | MockTokenizer (20 tokens) + MockLlmBackend (hash gen) |
| **ruvllm_integration_tests.rs** (prime-radiant) | 1,393 | 25 (0 #[ignore]) | ZERO from ruvllm or prime_radiant | 5 inline subsystem reimplementations |

**e2e_integration_test.rs** is the more valuable file despite its "E2E" misnomer. It imports and exercises real production functions (softmax, log_softmax, top_k_filter, top_p_filter, sample_from_probs) and real data structures (TwoTierKvCache, SpeculativeStats, SpeculationTree, ServingEngine). What it does NOT test: hnsw_router, claude_flow integration, NAPI bridge, SIMD paths, any cross-crate composition. The 3 `#[ignore]` tests only check GGUF magic bytes or env vars — they don't run actual inference even with a model present. An unsafe pointer cast at L840 (`Arc::as_ptr() as *mut MockLlmBackend`) is undefined behavior under Rust aliasing rules.

**ruvllm_integration_tests.rs** is the weaker file. Despite `#![cfg(feature = "ruvllm")]`, it imports nothing from either ruvllm or prime-radiant. All 25 tests operate on inline reimplementations of 5 subsystems: SheafCoherenceValidator (cosine similarity vs real CoherenceEngine), PatternToRestrictionBridge (Vec<LearnedPattern> vs real ReasoningBank), UnifiedWitnessLog (hash chain vs real GenerationWitness), SpeculativeDecoder (mock beam search), and CausalGraph (mock DAG). The tests validate the CONCEPT of coherence-gated LLM validation, not the actual implementation. The mock algorithms are individually sound (cosine similarity, hash chains, sigmoid mapping, negation detection) but prove nothing about the production subsystems they model.

### 5.11 Deployment Architecture (R139) — 3 files, 324 LOC

Three deployment models span the spectrum from production-grade to facade:

| Deployment | LOC | Target | Production Readiness |
|-----------|-----|--------|---------------------|
| **V3 Docker** (claude-flow) | 117 | Claude-flow V3 (Ruflo) | HIGH — 3 profiles, monitoring, non-root |
| **Distributed Raft** (ruvector) | 198 | ruvector cluster | FACADE — netcat only, never invokes Rust |
| **Agentic-flow** | 9 | Demo | MINIMAL — single service, hardcoded topic |

**V3 Docker** is the most production-ready deployment in the entire ruvnet ecosystem. Three profiles serve different use cases: **lite** (<100MB, aggressive pruning removes ONNX 158MB + agentic-flow 238MB) for minimal deployments; **full** (~800MB, all deps including ONNX+ruvector, `CLAUDE_FLOW_MEMORY_BACKEND=hybrid`) for complete installations; **workers** (headless Claude Code instances, Redis queue, Prometheus+Grafana monitoring) for scaled-out operation via separate docker-compose.workers.yml. Non-root user and dumb-init signal handling follow container best practices. ruvector is included as an npm dependency (not a separate container), confirming the NAPI/npm linking model rather than microservice architecture.

**Distributed Raft** test is architecturally well-designed but operationally a facade. The 5-node cluster configuration specifies 3 ports per node (7000 Raft consensus, 8000 cluster membership, 9000 data replication), 64 shards, and replication factor 3. However, the nodes themselves are shell scripts: each runs a `while true; sleep 1; echo heartbeat` loop with netcat listening on healthcheck ports responding "200 OK". The Dockerfile copies `.rlib` files from the build stage into runtime — but `.rlib` files are Rust static libraries, not executables, so no binary can actually run. The one genuine component is the test-runner service which runs `cargo test -p ruvector-raft -p ruvector-cluster -p ruvector-replication` against the actual Rust test suites. Multi-node distributed behavior has never been validated.

**Agentic-flow** Docker config is a 9-line minimal demo: single `agents` service, 1 replica, ANTHROPIC_API_KEY from environment (proper pattern), hardcoded `TOPIC="upgrade checkout flow"`. The backing Dockerfile IS functional (node:20-slim, npm ci + build, installs @anthropic-ai/claude-code), but the compose file has no volumes, networking, healthcheck, or resource limits.

---

## Section 6: Architecture Patterns

### Pattern 1: Real Primitives, Disconnected from Semantic Layer

Seen across all production-infra subsystems:
- postgres.rs: REAL async sqlx but zero sheaf integration
- policy.rs: Real state machine but EscalationCondition has no evaluator and no coherence wiring
- path-security.ts: Real OWASP security but ORPHANED (zero imports)
- strategies.rs: Real StrategyRegistry + adaptive learning but LOG-ONLY execution

The primitives are often 85-95% real. The integration with the semantic/intelligence layers they nominally serve is 0%. R110 adds a variant: cohomology operators that are individually correct (classical coboundary) but composed into incorrect higher-order structures (Hodge Laplacian doubly broken).

### Pattern 2: Persistence Layer Proliferation (12th+ instance)

governance/repository.rs is the 12th persistence layer stub identified in the ruvnet ecosystem. It follows the established pattern: clean traits, good in-memory implementations, zero production backends. See also: AgentDB THREE-LAYER persistence (R48), MinCut KV cache (R96), serving KV cache (R106), consolidation.rs EWC++ vs agentic_memory stub (R104).

### Pattern 3: Module-Complete Subsystems with Integration Gaps

Three subsystems reached MODULE COMPLETE status in R107-R108. In each case, the internal composition is coherent and well-tested (integration tests confirm sub-system APIs compose). But each subsystem is isolated from the adjacent semantic layer: serving/ from actual GPU backends, storage/ from sheaf coherence, governance/ from coherence engine.

### Pattern 4: CLI Island Architecture (R134)

All four CLIs (ruvector, claude-flow, rvlite, ruvllm) are completely independent: zero shared code, zero shared vector stores, zero shared MCP protocol implementations. Each reinvents embedding, storage initialization, and tool registration. This parallels the Rust-side persistence proliferation (12+ layers) but at the user-facing layer. The pattern suggests independent development timelines rather than coordinated architecture.

### Pattern 5: Hash Embedding End-to-End Confirmation (R135)

R20 identified the root cause (EmbeddingService never initialized in AgentDB bridge). R117 traced it through onnx-embedder.ts vs intelligence-engine.ts sync path. R134 closes the loop at the CLI layer: ruvector CLI, ruvector MCP, and ruvllm CLI all fall through to hash embeddings as the common case. The user-facing embedding quality is 64-dim hash unless ONNX or native binaries are explicitly available and initialized — which no standard startup path ensures.

### Pattern 6: V2→V3 Regression with Infrastructure Inflation (R138)

V3 adds vastly more protocol infrastructure (9 sub-registries, 3,040 LOC of dedicated modules, MCP 2025-11-25 compliance) but regresses on operational features. V2 had AuthManager, LoadBalancer with circuit breaker, and 3 real tool factories with context injection — all removed in V3 with no replacement. V3 compensates with more tools (82 vs 64) but 14 are facades. Net genuine tool count is comparable (~68 V3 vs ~64 V2) with significantly more code to maintain. This "more infrastructure, fewer features" anti-pattern suggests the V3 rewrite prioritized protocol compliance over operational completeness.

### Pattern 7: Competing Implementations with Zero Convergence (R138)

Two V3 MCP server implementations (v3/mcp/server.ts and v3/@claude-flow/mcp/src/server.ts) implement the same MCPServer interface with different protocol versions, different registry counts, and zero code sharing. This parallels the CLI Island Architecture (Pattern 4) at the server layer. The ecosystem now has 8+ MCP server implementations across Rust and TypeScript with no consolidation path.

### Pattern 8: CI/Testing Facade Pattern (R139)

CI pipelines and "integration tests" exist structurally but provide near-zero quality gate. Both claude-flow CI pipelines use `continue-on-error: true` on all test/typecheck/audit steps so only lint blocks. ruvector pipelines have `skip_tests` and `|| true` bypasses. Both Rust "integration" test suites (2,928 LOC) are 100% mock-only with zero cross-crate imports. The distributed Raft test deploys shell scripts with netcat instead of Rust binaries. The ecosystem has accumulated ~5,000 LOC of CI/test infrastructure that provides the appearance of quality assurance without the substance — the green badge pattern. V3 has 11 named test scripts that exist in package.json but none are wired into CI.

### Pattern 9: Worker Execution Zero-Memory Chain (R134-R140)

The zero-memory-bootstrap pattern identified in R20 (EmbeddingService never initialized in the AgentDB bridge) extends all the way through the complete worker execution chain:

```
CLI entry (cli.js) → zero memory init
MCP server (mcp-server.js, v3/mcp/server.ts) → zero memory/AgentDB init
Memory layer (agentdb-adapter.ts) → plain JS Map, zero AgentDB imports
Tool handlers (v3/mcp/tools/index.ts) → SONA facade, in-memory Maps only
Execution layer (headless-worker-executor.ts) → spawn('claude --print') → zero memory layer
Container layer (container-worker-pool.ts) → Docker CLI → zero memory integration
```

Not a single step in the end-to-end chain from user CLI invocation to agent subprocess execution reads from or writes to AgentDB, HNSW, or any vector store. The memory system is architecturally absent from the operational path despite being the stated core value proposition of the platform.

### Pattern 10: Bifurcated Coordination with Incompatible Formats (R140)

Multiple pairs of supposedly coordinating subsystems use incompatible data formats:
- **ClaimService vs MCP claims-tools**: ClaimService formats claimants as `agentId:taskId` (2-part); claims-tools.ts formatClaimant() produces `userId:name` or `agentId:taskId:timestamp` (3-part). Neither can parse the other's claims.
- **Two V3 MCP servers** (v3/mcp/server.ts vs v3/@claude-flow/mcp/src/server.ts): Different protocol versions (2024-11-05 vs 2025-11-25), different registry counts, zero code sharing.
- **Two backend factories** (agentdb root factory.ts vs packages/agentdb factory.ts): 2-tier vs 5-tier fallback chains with different priority orders, both exporting `createBackend()`.
- **Two AgentDB init paths** (agentdb-mcp-server.ts vs agentdb-adapter.ts): One initializes EmbeddingService (Pipeline 1), the other never does. They cannot be interchanged.

The pattern suggests parallel development without cross-team coordination, producing systems that look compatible from their interface names but are structurally incompatible at runtime.

---

## Section 7: Knowledge Gaps

- PostgreSQL bgworker API integration (how to properly register healing worker)
- Actual metric collection queries for ruvector-postgres (pg_stat_user_tables, pg_stat_activity)
- Strategy implementation for each of the 5 healing actions
- Whether path-security.ts was intentionally orphaned or accidentally overlooked
- How (or whether) governance/policy.rs ThresholdConfig is intended to wire to coherence/energy.rs lane system
- Whether postgres.rs AsyncGraphStorageAdapter is intended for future trait impl or is dead code
- WAL recovery code path in file.rs (where the committed=true flip is supposed to happen)
- Ed25519 signature verification — whether there is an external verifier intended to call add_approval() after verification
- Whether ruvector CLI's VectorDB path has a startup sequence that initializes IntelligenceEngine before first embed call
- Whether the 55-tool MCP monolith is intended to be split into per-domain tool servers
- Whether rvlite is a prototype for ruvector replacement or a parallel product
- Whether ruvllm's native .node binary is distributed in any published npm package
- Whether the two V3 MCP servers are intended to coexist or if one should replace the other
- Whether V2's AuthManager and LoadBalancer features are planned for V3 re-implementation
- Which V3 MCP server is the canonical entry point for production deployments
- Whether the ruvector setup PostgreSQL schema is tested against the current ruvector C extension version
- What the intended bootstrap path is for memory/AgentDB in MCP server contexts
- **ruflo#1207** (Henrik Pettersen, 2026-02-23): External community has mapped a 16-op fix across 8 files in 4 packages to upgrade AgentDB v2→v3 (WM-008). Fix addresses `vectorBackend: 'auto'` hardcoding, `.db→.rvf` migration, and the integration gap. Status: OPEN, not merged
- Whether `skip_tests` in release.yml is ever used in production releases or only for hotfixes
- Whether the 11 V3 test scripts (test:integration:memory, etc.) are ever run manually or are aspirational
- Whether the distributed Raft docker-compose was ever used with real Rust binaries or was always shell-script-only
- Whether ruvllm_integration_tests.rs cfg(feature="ruvllm") was originally wired to real imports that were later removed
- What the intended relationship is between the test-runner cargo tests and the shell-script node services in the distributed setup
- Whether the V3 Docker lite profile (<100MB) has been validated against all claude-flow features
- Whether agentdb-mcp-server.ts Pipeline 1 (Xenova/all-MiniLM-L6-v2) is actually integrated into claude-flow's runtime, or is a standalone server never used by the main claude-flow CLI
- What the correct claimant format should be — whether ClaimService (2-part) or claims-tools.ts (3-part) represents the canonical spec
- Whether the binary quantizer in agentdb-adapter.ts was intended to be used (broken math) or is dead experimental code
- Whether causal_add_edge hardcoding of memoryId=0 in agentdb-mcp-server.ts is a known bug or intended placeholder
- Whether the ghost DEEP files from R136 were previously analyzed under different session IDs or are genuinely unexplored
- What initializes the agentdb-mcp-server.ts — whether it is invoked by the main claude-flow CLI or must be run as a separate process

---

## Section 8: Session Log

### R36 (2026-02-15): ruvector-postgres healing subsystem
5 files, 4,070 LOC. Self-healing database infrastructure with real learning (92-95%) but stub detection and execution. Functioning brain, no eyes or hands.

### R48 (2026-02-15): AgentDB health monitoring + path security
2 files, 951 LOC. health-monitor.ts (99%) is BEST monitoring in AgentDB — real OS/V8 metrics, linear regression leak detection, MPC self-healing. path-security.ts (88-92%) is OWASP-compliant but ORPHANED — zero imports found.

### R107 (2026-02-18): ruvllm serving completion + prime-radiant storage start
4 files, ~2,965 LOC. serving/request.rs (88-92%) + serving/mod.rs (92%) complete the serving MODULE (6/6 DEEP, ~90% avg, BEST ruvllm subsystem). NoopBackend decoupling confirmed via integration tests. stop_sequences silently ignored (functional gap). prime-radiant/storage/postgres.rs (82-87%) is REAL async sqlx/PgPool but disconnected from sheaf coherence, with race condition and dead AsyncGraphStorageAdapter. governance/repository.rs (62%) is 12th persistence stub — traits only, no async despite "Async-First" claim.

### R108 (2026-02-18): prime-radiant storage + governance module completion
5 files, ~3,903 LOC. storage MODULE COMPLETE (4/4 DEEP, ~86% avg): file.rs CRITICAL WAL commit-flag bug (deletions non-durable), memory.rs witnesses_by_action never populated, mod.rs HybridStorage = FileStorage ONLY (postgres never wired). governance MODULE COMPLETE (5/5 DEEP, ~88% avg): TWO CRITICAL security findings in policy.rs — signatures never verified, zero-byte placeholder() activates real bundles. EscalationCondition DSL has no evaluator. Governance completely isolated from coherence engine it governs.

### R110 (2026-02-18): prime-radiant cohomology + ruvector-attention OT
2 files, 713 LOC, 24 findings (2 CRITICAL, 4 HIGH, 7 MEDIUM, 11 INFO). cocycle.rs (471 LOC, ~75-80%): TWO CRITICAL correctness bugs — is_coboundary() always returns false for degree>0 (all H^n for n>=1 wrong), apply_adjoint() iterates wrong source dimension (Hodge Laplacian doubly broken combined with R109 laplacian.rs bugs). Classical coboundary operator and sheaf coboundary individually correct; composed Hodge decomposition pipeline is corrupted end-to-end. cached_projections.rs (242 LOC, ~88-92%): GENUINE non-facade OT utility — ProjectionCache + WindowCache for sliced Wasserstein. Zero hash shortcuts. dot_product_simd is scalar unrolling only (misleading name).

### R114 (2026-02-19): mcp-gate crate + ruvllm policy_store + hnsw_rs patches + prime-radiant error boundary
5 files in prod-infra scope, ~2,080 LOC. **mcp-gate CRATE EFFECTIVELY COMPLETE (3/3 DEEP, ~91% avg)**: types.rs (92%) clean MCP DTO re-exporting 8 cognitum-gate-tilezero types + 12 envelope types; server.rs (90-93%) genuine stdio JSON-RPC 2.0 MCP server, 3 tools only, env-var config, Arc injection; tools.rs (88-92% from R113). **CONFIRMS 7th parallel MCP protocol implementation** — distinct from rmcp, strange-loop, psycho-symbolic, claudeFlowSdk, bin/mcp-server.js, ruvector-cli/mcp_server.rs. PermitToken crypto stripped to base64 at API boundary (extends unverified-crypto to 5 files). **ruvllm/policy_store.rs (72%) FUNCTIONALLY BROKEN persistence** — get() cache-only (data lost on restart), delete() ghost entries, search_by_type() cache-only. Uses REAL AgenticDB (not hash). Two non-interoperating policy systems (PolicyStore runtime vs PolicyBundle governance). 13th persistence layer. **hnsw_rs/datamap.rs (88-92%) vendored upstream copy** — zero algorithmic changes, process::exit() in library code, rand 0.8/0.9 mismatch (stated patch purpose contradicted). **prime-radiant error.rs (90-93%) high-quality thiserror** — 26 variants → 5 ADRs, WitnessId/String boundary asymmetry, pure error taxonomy.

### R134 (2026-03-01): CLI Entrypoints — The Front Door (ML-A)
7 files, ~13,450 LOC, 6 CRITICAL + 28 HIGH + 47 MEDIUM findings. First analysis of all four CLI front doors (ruvector, claude-flow, rvlite, ruvllm). ruvector CLI (7,357 LOC, ~72-78%) is a substantial monolith using VectorDB (working path); hash embedding CONFIRMED at CLI layer (extends R20/R117 end-to-end). ruvector MCP registers 55 tools (largest in ecosystem) but sanitizeShellArg() destroys SQL/Cypher/SPARQL queries. claude-flow CLI is a cold 156-LOC dispatcher; ruflo is a pure rebrand. rvlite (1,686 LOC, ~80-85%) is a fully independent alternative with genuine WASM and correct hyperbolic geometry but O(n) search and zero ruvector integration. ruvllm CLI (1,005 LOC) is entirely native-binary-dependent — JS fallback is all facades, training is simulated. Zero cross-CLI code sharing across all four tools.

### R135 (2026-03-01): V3 Memory Layer — AgentDB Adapter and Controller Registry
2 files, ~2,064 LOC, 4 CRITICAL + 8 HIGH + 9 MEDIUM findings. First analysis of the V3 @claude-flow/memory package. AgentDBAdapter (1,038 LOC) is the most misleadingly named file in the codebase — stores data in a plain JS Map with zero AgentDB imports. loadFromDisk/saveToDisk are empty stubs. R20 root cause NOT fixed in V3. createHybridService() creates a plain UnifiedMemoryService. Binary quantizer logic is mathematically broken (packs bits but distance function never decodes). ControllerRegistry (1,026 LOC) — path traversal validation is a no-op, createEmbeddingService() fallback emits zero vectors silently, causalRecall controller is never added to the controllers Map.

### R136 (2026-03-01): Ghost DEEP Files and AgentDB MCP Server (ML-C partial)
3 files in this domain, 2,368 LOC for agentdb-mcp-server.ts, 2 CRITICAL + 6 HIGH + 7 MEDIUM findings. agentdb-mcp-server.ts DIRECTLY CONTRADICTS R20 — EmbeddingService IS initialized (Pipeline 1: Xenova/all-MiniLM-L6-v2 via top-level await). Tool count mismatch: 28 actual vs 32 documented. CRITICAL: causal_add_edge/causal_query hardcode memory IDs to 0, making the entire causal graph data model broken. agentdb_init re-uses global db regardless of caller path. agentdb_search session_id filter has TODO and is never applied.

### R138 (2026-03-02): V3 MCP Tool Chain — ML-D
6 files, ~4,200 LOC, 1 CRITICAL + 10 HIGH + 9 MEDIUM findings. End-to-end analysis of V3 MCP tool chain from tool registry through two competing servers to CLI dispatch, plus V2 predecessor comparison. CRITICAL: SONA tools fabricate HNSW speedup via tautological multiplication (source of "150x-12,500x" claim). V2→V3 regression confirmed — V2 had AuthManager, LoadBalancer with circuit breaker, and 3 real tool factories; all LOST in V3. Two competing V3 MCP servers (792 LOC internal + 1,134 LOC library) share zero code. Zero memory/AgentDB/EmbeddingService bootstrap in ANY MCP server (V2 or V3). CLI command registry has 31 commands with dead lazy loading and 2 orphans. ruvector setup generates production-quality PostgreSQL schema but has zero bridge to backend factory or memory-initializer. **External validation**: ruflo#1207 (Henrik Pettersen) independently maps the V2→V3 AgentDB integration gap with a 16-op fix across 8 files in 4 packages (WM-008, status: OPEN).

### R139 (2026-03-02): ML-E — CI, Tests, and Deployment Ground Truth
14 files, ~5,757 LOC, 99 findings (3 CRITICAL, 14 HIGH, 11 MEDIUM). CI pipelines are facades — both claude-flow pipelines use continue-on-error:true on tests. Both Rust "integration" test files (2,928 LOC) are 100% mock-only. V3 Docker is the most production-ready deployment. ruvector releases 25 genuine crates but has skip_tests bypass. **v2 supplement**: build-native.yml (5-platform NAPI cross-compile, graph-node DISABLED), sona-napi.yml (7-platform build with universal macOS binary + full npm publish pipeline — most comprehensive NAPI CI), docker-compose.agent.yml (demo not deployment — 7 agent services with hardcoded tasks), agentdb root factory.ts (2-tier, COMPETES with 5-tier factory), ruvector core barrel (23 modules re-exported unselectively).

### R140 (2026-03-02): ML-F — headless-worker-executor.ts (Execution Engine)
1 file, 1,342 LOC, 12 findings (0 CRITICAL, 4 HIGH, 4 MEDIUM, 4 INFO). **GENUINE execution engine** — the actual mechanism behind `claude-flow agent spawn` for headless workers. executeClaudeCode() spawns `claude --print <prompt>` as a real subprocess via child_process.spawn. CONFIRMED: no MCP protocol, no stdin injection, no memory layer, no AgentDB. Prompt is the full template + filesystem context, passed as a raw CLI argument. Process pool (maxConcurrent=2), pending queue, context caching with TTL, and EventEmitter monitoring are all genuine. Key bugs: double timeout race (two independent setTimeout calls on same process), simpleGlob() misses files when ** is the terminal pattern segment. Key security: audit worker ships .env* in contextPatterns (secrets sent to Anthropic API + written to local logs), prompt injected as raw argv (potential argument injection). 6 of 8 worker types are disabled=false by default. Zero memory layer connection extends the R20/R138/R139 systemic pattern: no startup path in the entire MCP→worker execution chain initializes AgentDB.

### R140 (2026-03-02): ML-G — hooks.ts CLI Command Layer (Self-Learning Hooks System)
1 file, 4,530 LOC, 12 findings (0 CRITICAL, 3 HIGH BUG, 5 HIGH ARCHITECTURE/GENUINE/INTEGRATION, 3 MEDIUM, 1 INFO). **Genuine ADR-005 MCP-first architecture** — all 30 subcommands are callMCPTool() wrappers delegating to hooks-tools.ts (3,281 LOC). Key results: (1) REAL: pre-task embeds enhanced-model-router.js for ADR-008 3-tier routing (L1529-1576), producing [AGENT_BOOSTER_AVAILABLE] or [TASK_MODEL_RECOMMENDATION] directives. (2) REAL: model-route/model-outcome/model-stats connect to model-router.js (tiny-dancer-neural) with learning via recordOutcome(). (3) REAL: intelligence command accurately reports SONA/MoE/HNSW availability via lazy loading — shows "loading" when modules absent, not fake "active". (4) REAL: Worker system (12 named workers: ultralearn, optimize, consolidate, predict, audit, map, preload, deepdive, document, refactor, benchmark, testgaps). (5) BUG: statusline agentDB vector count = dbSizeKB/2 (heuristic, NOT real query). Intelligence score inferred from file presence checks. (6) BUG: token-optimize unconditionally adds stats.totalTokensSaved += 200, cacheHits = 2 regardless of real activity — inflated display. (7) ARCHITECTURE: 30 actual subcommands (not 17 as documented) including 4 v2 backward-compat aliases, 3 coverage-aware, 3 model-routing, 2 Agent Teams, statusline, progress, token-optimize. (8) hooks-tools.ts semantic router uses generateSimpleEmbedding() (sin/cos hash — NOT ONNX) for route patterns even when native VectorDb/HNSW is active.
