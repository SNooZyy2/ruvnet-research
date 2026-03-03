# claude-flow README vs Reality — Feature-by-Feature Verdict

> Cross-referenced against 142 research sessions, 1,696 DEEP files, ~12,877 findings.
> Each feature classified as: GENUINE / PARTIALLY REAL / FABRICATED / UNCOVERED
> Date: 2026-03-03 (updated with Middle Layer R135-R140 + Compilation Audit R141)

---

## Self-Learning & Intelligence

The headline feature. Can we deliver it?

| README Claim | Verdict | Evidence | Can We Rebuild? |
|-------------|---------|----------|----------------|
| **ReasoningBank** (RETRIEVE→JUDGE→DISTILL) | **GENUINE** | TS+Rust both 92-95%. 160 findings across 101 sessions. Real DeepMind-style algorithms (MaTTS search, MMR). Mature v1→v2 migration. | YES — copy both TS and Rust implementations |
| **EWC++** (prevents catastrophic forgetting) | **GENUINE** | micro_lora.rs 92-95%. Real online Fisher, adaptive lambda, task boundary detection. NEON SIMD. R106: training.rs orchestration has correct EWC penalty math (λ*F*(w-w*)²/2) but Fisher information is never updated during training — making it static EWC, not truly "++". The micro_lora.rs core is genuine; the training loop's continual adaptation is incomplete. | YES — copy from Rust crate; wire Fisher updates for true EWC++ |
| **SONA** (<0.05ms adaptation) | **PARTIALLY REAL** | sona crate ~75% production-ready (R98 revised down from 85%). Algorithms 85-90%, orchestration 60-70%. LoRA+EWC+++federated+SafeTensors present. Loop C missing from exports (R100). <0.05ms unverified. **R140 UPDATE**: TWO independent "SONA" systems discovered: `sona-optimizer.ts` (842 LOC, genuinely functional Bayesian agent-routing wired into hooks) vs `sona-tools.ts` (fabricates "1000x speedup" via `estimatedBruteForce = searchLatency * 1000`). Same branding, completely different realities. **R141**: sona Rust crate (10,582 LOC) fails `cargo check` — broken integration with workspace. | PARTIALLY — sona-optimizer.ts is the real asset; sona-tools.ts and Rust crate are not usable |
| **MoE** (8 expert networks) | **PARTIALLY REAL** | moe-router.js has real 2-layer gating (384→128→8) with Xavier init. But it's a JS file, not integrated into any training pipeline. | MAYBE — the routing math is real, needs wiring |
| **LearningBridge** (0.12ms/insight) | **FABRICATED** | Zero files found in DB. No code exists. | NO — write from scratch if needed |
| **9 RL Algorithms** (PPO, A2C, DQN, etc.) | **FABRICATED** | All 9 reduce to identical tabular Q-value updates. DQN has no neural network. PPO has no clipping. Decision Transformer has no transformer. Cosmetic naming only. | NO — would need real RL implementations |
| **intelligence.ts** (V3 learning engine) | **FABRICATED** | R140 CRITICAL: Claims O(log n) HNSW search, actual is O(n) brute-force linear scan. LoRA/EWC config stored but NEVER used. compactPatterns() is O(n²), blocking event loop at scale. Has 14+ consumers — NOT dead code, actively misleading the runtime. This is the BIGGEST active facade in V3. | NO — replace with real HNSW call or honest brute-force |

**Self-learning verdict**: YES, we can rebuild real self-learning. ReasoningBank + EWC++ + hooks gives you:
- Pattern storage and retrieval from past decisions
- Verdict judgment on whether past patterns apply
- Memory consolidation without forgetting old patterns
- Automatic pattern application via pre-task hooks

What you WON'T get: the inflated performance claims (<0.05ms), the 9 RL algorithms, or the "LearningBridge."

**R140 WARNING**: The self-learning pipeline is severed at the post-task boundary. `hook-handler.cjs` stubs out the ReasoningBank judge->distill->consolidate chain (R73 CRITICAL, confirmed R140). Data collection works (IntelligenceStore 98% real, SQLite-backed) but the learning-from-data path never executes. The most critical fix for v4 is reconnecting this pipeline.

---

## Memory & Vector Operations

| README Claim | Verdict | Evidence | Can We Rebuild? |
|-------------|---------|----------|----------------|
| **HNSW Vector Search** | **GENUINE** | ruvector-core 92-98%. Real SIMD dispatch (AVX-512/AVX2/NEON). 3 distinct HNSW implementations found. | YES — copy ruvector-core |
| **"150x-12,500x faster"** retrieval | **INFLATED** | HNSW is genuinely fast but no benchmark proves these specific multipliers. The numbers appear to be theoretical comparisons vs brute-force, not measured. | Partially — HNSW IS fast, specific numbers are marketing |
| **Product Quantization** | **GENUINE** | R90: 88-92%. Real k-means++, Lloyd's, ADC with lookup tables. | YES — copy from ruvector-core |
| **Conformal Prediction** | **GENUINE** | R90: 88-93%. Valid Vovk et al. quantile formula. | YES — copy from ruvector-core |
| **RuVector PostgreSQL Bridge** (77+ SQL functions) | **PARTIALLY REAL** | R35: architecture-complete, persistence-incomplete. SQL functions exist but R35 found persistence layer not fully wired. ~61µs search claim unverified. | MAYBE — needs completion work |
| **39 Attention Mechanisms in SQL** | **PARTIALLY REAL** | R96-R97: 5 DEEP files in `ruvector-postgres/src/attention/` (flash.rs, mod.rs, multi_head.rs, operators.rs, scaled_dot.rs). SQL attention arc COMPLETE (R97). Real pg_extern functions implementing attention as SQL operators. 16 total postgres attention files. | PARTIALLY — genuine SQL operator implementations, "39 mechanisms" count inflated |
| **15 GNN Layer Types** | **PARTIALLY REAL** | 33 files in ruvector-gnn crate, 15 DEEP. Real training.rs (1,368 LOC), genuine Kipf & Welling GCN (R101). TWO parallel GNN ecosystems: ruvector-gnn crate + ruvector-postgres/gnn (self-contained reimplementation, zero cross-crate composition). GNN bindings genuine: gnn-node 88-92% (napi-rs), gnn-wasm 90-94% (R99). Both inference-only. Deterministic weights throughout (no training). | PARTIALLY — real GNN math, two disconnected ecosystems, no training pipeline |
| **MemoryGraph with PageRank** | **PARTIALLY REAL** | backward_push.rs is genuine O(1/ε) (95%+). intelligence.cjs has real PageRank with power iteration. But "MemoryGraph" as an integrated product doesn't exist. | YES for PageRank algorithm, NO for integrated MemoryGraph |
| **Hyperbolic Geometry** (Poincaré ball) | **GENUINE** | 63 files, 21 DEEP. R92: hyperbolic HNSW GENUINE (88-95%). R99: hyperbolic-hnsw crate COMPLETE. R100: ruvector-attention hyperbolic module COMPLETE (4/4, 90-93%), poincare.rs 93-96% STRONGEST, mixed_curvature.rs MOST NOVEL. R98: SQL hyperbolic GENUINE (poincare 88-92%, lorentz 87-92%). R101: postgres hyperbolic COMPLETE. CRITICAL: zero manifold validation at any layer. | YES — genuine hyperbolic math across 4 crates, needs manifold validation |
| **AgentMemoryScope** (3-scope system) | **PARTIALLY REAL** | Memory system works at basic level (better-sqlite3). But 11 disconnected persistence layers (R106: serving KV cache = 11th) means the "3-scope" claim hides architectural chaos. R104: ruvllm context module COMPLETE (7/7 DEEP) has 4+ genuine HNSW stores that NEVER compose at runtime. **R135-R136 UPDATE**: V3 `agentdb-adapter.ts` is MISNAMED — storage is a plain `Map<string, MemoryEntry>`, zero connection to real AgentDB. `memory-bridge.ts` (1,773 LOC genuine) is NOT compiled into npm dist. R20 root cause (EmbeddingService never initialized) is NOT fixed in V3. Persistence layers now counted at 13+. | Rebuild with single persistence (ADR-v4-002) |

---

## LLM Serving Infrastructure

Not prominently claimed in README but genuinely present.

| Component | Verdict | Evidence | Can We Rebuild? |
|-----------|---------|----------|----------------|
| **Continuous Batching** (vLLM/Orca) | **GENUINE** | R35: scheduler.rs 90-92% (vLLM-style with preemption, chunked prefill). R106: batch.rs 90-95% — production data structures with correct merge_prefill_decode(), TokenBudget dual-gate, PagedAttention block_table threading. 4 tests. | YES — strongest serving module in ruvllm |
| **PagedAttention KV Cache** | **GENUINE** | R106: kv_cache_manager.rs 88-92%. Genuine Kwon et al. 2023 paged allocation. Real bugs: deadlock risk (double RwLock), memory estimate 2x too low (claims f16, stores f32), broken swap_out accounting. 11th parallel subsystem vs MinCut KV cache. 7 tests. | YES — fix the 3 bugs, use as single KV cache layer |
| **LoRA Adapter Management** | **GENUINE** | R106: adapter.rs 85-88%. AdapterRegistry (DashMap+LRU eviction), AdapterPool, AdapterComposer (6 strategies). forward_sequential() has math bug (double-removes input). All math delegates to micro_lora.rs (92-95%). | YES — fix forward_sequential(), wire to micro_lora.rs |
| **LoRA Training Loop** | **PARTIALLY REAL** | R106: training.rs 82-87%. Correct EWC penalty math, 7 LR schedules (Cosine, OneCycle, etc.), async feedback queue. But GradientAccumulator is dead code in main path, Fisher updates never called, EwcState cloned per step. | MOSTLY — orchestration works, dead code needs cleanup |
| **Two-Tier KV Cache** | **GENUINE** | R34: kv_cache.rs 90%. NEON SIMD quantize/dequantize, hot/cold tiering. | YES — wire to kv_cache_manager.rs |

---

## Performance Claims

| README Claim | Verdict | Evidence | Can We Rebuild? |
|-------------|---------|----------|----------------|
| **Agent Booster (WASM)** "352x faster" | **FABRICATED** | agent-booster-enhanced.ts (1,428 LOC DEEP) exists, but 13 theatrical WASM stubs found. "352x" has no benchmark. The WASM stubs we found are console.log facades. | NO — the 352x claim is baseless |
| **Flash Attention** "2.49x-7.47x" | **PARTIALLY REAL** | 3 DEEP files: flash_attention.rs in ruvector-mincut-gated-transformer (997 LOC), cuda-wasm flash_attention.rs (528 LOC, R92), JS fallback (643 LOC). R34/R93: MinCut crate "MOST NOVEL," kernels 88-93% with SIMD. Performance numbers unverified. | MAYBE — the algorithm exists across Rust+CUDA+JS, speedup claims unverified |
| **Int8 Quantization** "3.92x memory reduction" | **FABRICATED** | R82: quantization.rs 75-78%, R87: inference/quantization.rs 0-5% PLACEHOLDER (returns empty Vec). No working quantization pipeline. **R141**: neural-network-implementation crate (17,294 LOC, previously rated 75-85%) produces 106 compilation errors — UNCOMPILABLE. Downgraded from "partially real" to FABRICATED. | NO — placeholder code, crate doesn't compile |
| **LoRA Compression** "128x" | **PARTIALLY REAL** | micro_lora.rs IS genuine (92-95%). But "128x compression" is a theoretical maximum, not demonstrated. | YES for LoRA, NO for 128x claim |
| **Token Optimizer** "30-50% reduction" | **PARTIALLY REAL** | Hook system is genuine (98.1% R19). Token optimization hooks exist. But 30-50% reduction claim is unverified. | MAYBE — hooks work, savings unproven |
| **SemanticRouter** "34,798 routes/s" | **UNCOVERED** | 1 DEEP file (semantic-router.js, 178 LOC). Not deeply assessed. Performance claim unverified. | Unknown |
| **84.8% SWE-Bench** | **FABRICATED** | SWE-bench evaluator exists (991 LOC DEEP) but findings show `build_command` generates English prompts, not actual CLI flags. Benchmark cannot execute. | NO — benchmark scaffolding exists but cannot run |

---

## Swarm Coordination

| README Claim | Verdict | Evidence | Can We Rebuild? |
|-------------|---------|----------|----------------|
| **6 Topology Patterns** | **PARTIALLY REAL** | CLI defines hierarchical, mesh, ring, star, hybrid, adaptive. R31/R71: CLI = demo framework. But Claude Code's Task tool actually does the execution — topologies are labels, not protocol implementations. | YES — topology is a coordination pattern, not complex code |
| **Byzantine Consensus** (2/3 majority) | **FABRICATED** | R84: coordination.rs 15-25% FACADE. No actual Byzantine fault detection or 2/3 majority voting. Vote files written but no voting logic. **R129**: ruvector-raft and delta-consensus crates exist but are separate from the claimed Byzantine consensus. **R141**: All 14 ruv-swarm crates blocked by `ruv-fann ^0.1.5` vs `0.2.0` version pin — none compile. | NO — would need real implementation |
| **Raft Consensus** | **GENUINE** | RAC 92%. Real Raft with leader election + real libp2p (R44). | YES — copy RAC crate |
| **Gossip Protocol** | **PARTIALLY REAL** | R90: gossip.rs 45-55%. Correct SWIM state machine, but transport = log statements. The protocol design is real, network I/O is absent. | MAYBE — needs transport layer (~200-300 LOC) |
| **CRDT** | **UNCOVERED** | Mentioned in docs but no specific DEEP reads on CRDT implementation. Findings note "LWW timestamps, no vector clocks, no CRDTs." | FABRICATED at system level — no real CRDTs found |
| **Queen-Led Hive Mind** | **PARTIALLY REAL** | CLI orchestration exists. In practice, Claude Code Task tool does the actual multi-agent work. The "queen" concept is a coordination label. | YES — it's already how we use claude-flow |
| **Claims System** | **PARTIALLY REAL** | Claims code exists in claude-flow-cli. Simple file-based ownership. Works for basic cases. | YES — keep and improve |

---

## Security

| README Claim | Verdict | Evidence | Can We Rebuild? |
|-------------|---------|----------|----------------|
| **AIDefence** (<10ms threat detection) | **PARTIALLY REAL** | R92: 3 DEEP files (AIDefenceGuard.ts 763 LOC, test 235 LOC, integration 166 LOC). Overall 82-88%. 16 files exist, npm-published package excludes aidefence module. Guard class has real pattern matching and threat classification. <10ms claim unverified. | PARTIALLY — real threat detection logic exists, performance claims unverified, excluded from npm publish |
| **Input Validation** (Zod) | **GENUINE** | config-loader.ts 92-95% with Zod. input-validator.ts exists in self-impl (270 LOC). | YES — already have this |
| **Path Traversal Prevention** | **GENUINE** | R88: RuVectorBackend has FORBIDDEN_PATH_PATTERNS, validatePath() on every op. | YES — copy from RuVectorBackend |
| **HMAC-SHA256 Proof Chain** | **PARTIALLY REAL** | guidance_kernel has HMAC but with HARDCODED key (security concern from early sessions). Concept real, implementation has a critical flaw. | MAYBE — fix the hardcoded key |
| **Command Sandboxing** | **PARTIALLY REAL** | Hook system has pre-command hooks. But the "allowlisted commands" claim depends on configuration, not enforcement in code. | PARTIALLY — hooks provide the mechanism |

---

## Model Routing

| README Claim | Verdict | Evidence | Can We Rebuild? |
|-------------|---------|----------|----------------|
| **3-Tier Routing** (WASM/Haiku/Opus) | **PARTIALLY REAL** | 6 PARALLEL routing systems found — none properly connected to each other. R106 confirms 6th: ruvector_integration.rs (82-87%) implements SONA→HNSW→keyword three-tier fusion with REAL ruvector-core HnswIndex (not hash). But it creates two independent UnifiedIndex instances that never synchronize, and is completely parallel to hnsw_router.rs (R37 BEST at 90-93%). The concept works through claude-flow hooks. | YES — already works via hooks, just needs consolidation. Pick hnsw_router.rs as the single routing surface |
| **75% Cost Reduction** | **INFLATED** | Model routing does save money by using cheaper models for simple tasks. 75% is a theoretical maximum. | PARTIALLY — real savings, inflated number |
| **Multi-LLM** (GPT-5.2, o3, Gemini 3, Grok 4.1, Llama 4) | **SPECULATIVE** | README lists models that may not exist. Provider config files exist but actual multi-provider testing unverified. | Depends on actual model availability |

---

## Development Features

| README Claim | Verdict | Evidence | Can We Rebuild? |
|-------------|---------|----------|----------------|
| **60+ Specialized Agents** | **GENUINE** | The agent types are defined and work via Claude Code Task tool. We use them daily. | YES — already working |
| **175+ MCP Tools** | **GENUINE** | R51: MCP server confirmed with 256 tools. | YES — already working |
| **42+ Pre-Built Skills** | **GENUINE** | Skills are YAML/MD templates that expand into prompts. They work. | YES — already working |
| **33 Lifecycle Hooks** | **GENUINE** | R19: hook-pipeline 98.1%. Hooks are one of the most genuine subsystems. | YES — already working |
| **Pair Programming** | **GENUINE** | It's a skill/mode, not complex code. Works through prompting patterns. | YES |
| **London School TDD** | **GENUINE** | Also a prompting pattern/skill. Works. | YES |
| **Event Sourcing** | **PARTIALLY REAL** | Event bus exists in self-impl. But "complete audit trail with replay" is aspirational — no replay mechanism found. | PARTIALLY — event bus yes, replay no |

---

## Orchestration & Execution Engine (NEW — R135-R140)

The Middle Layer deep-dive (R135-R140) traced the complete path from CLI entry to agent execution. This was previously uncovered.

| Component | Verdict | Evidence | Can We Rebuild? |
|-----------|---------|----------|----------------|
| **Agent Execution** (spawn subprocess) | **GENUINE but PRIMITIVE** | R140: Agent execution = `spawn('claude', ['--print', prompt])` subprocess. Zero MCP protocol between orchestrator and workers. Three-tier chain: ContainerWorkerPool (Docker) → worker-daemon (setTimeout) → HeadlessWorkerExecutor (spawn claude). HeadlessWorkerExecutor 78-83% genuine: real process pool, context caching, output parsing. | YES — HeadlessWorkerExecutor is reusable with fixes |
| **worker-daemon** | **MOSTLY FACADE** | R140: NOT a daemon — foreground class, no fork/PID. 9/12 local workers are FACADE stubs. | PARTIALLY — 3/12 workers real, rest need implementation |
| **container-worker-pool** | **GENUINE** | R140: REAL Docker CLI integration. CRITICAL BUG: prompt+contextPatterns silently dropped in `buildWorkerCommand()`. | YES — fix buildWorkerCommand() context dropping |
| **claim-service** | **PARTIALLY REAL** | R140: LOCAL-ONLY JSON file, no distributed coordination. COMPETES with `claims-tools.ts` (incompatible 2-part vs 3-part claimant formats). | YES — pick one format, add distributed support later |
| **V3 Memory Bootstrap** | **BROKEN** | R135-R136: `memory-bridge.ts` (1,773 LOC genuine) NOT compiled into npm dist. `agentdb-adapter.ts` is plain Map. `memory-initializer.ts` has SQL injection, dimension mismatch (768 vs 384), and writes invalid SQLite headers as fallback. | NO for V3 — rebuild from scratch per ADR-v4-002 |
| **MCP Tool Chain** | **PARTIALLY REAL** | R138: V3 has 82 tools (67 V3 + 15 V2-compat). 14 SONA tools are FACADES fabricating speedups. V2→V3 REGRESSION: V2 had 3 tool factories (64 tools), V3 lost all factories. TWO competing V3 MCP servers exist. ZERO memory/AgentDB initialization in ANY MCP server bootstrap. | PARTIALLY — tool registration works, memory integration broken |
| **CI/CD Pipelines** | **FACADE** | R139: CI pipelines use `continue-on-error: true` on tests/typecheck/audit. Only lint can actually fail the build. V3 has 11 test scripts but NONE run in CI. Release publishes 25 Rust crates in genuine topological sort but `skip_tests` bypasses ALL validation. | NO — CI gives false green. Cannot trust as validation. |
| **hooks.ts** (4,530 LOC) | **GENUINE with caveats** | R140: 30 real MCP wrapper subcommands (not 17 documented). 72-78% genuine. `token-optimize` has hardcoded +200 fake savings. `pre-task` has REAL ADR-008 3-tier routing via `enhanced-model-router.js`. Statusline uses `dbSizeKB/2` heuristic for vector count. | YES — copy, prune fake savings, keep routing |
| **SONA Bayesian Optimizer** | **GENUINE** | R140: `sona-optimizer.ts` (842 LOC) implements real Bayesian agent-routing with temporal decay and Thompson sampling. ONLY V3 memory subsystem wired into hooks. Zero HNSW/ruvector connection. | YES — the one real learning component in V3 runtime |

---

## Synthesis-Sourced Bugs (Not Previously in V4 Plans)

> Critical bugs surfaced from 14 domain synthesis docs (~10,800 LOC of analysis) that v4 must address. Finding IDs are domain-local (e.g., ruvector C38 ≠ swarm C38).

| ID | Domain | Bug | Impact | Evidence |
|----|--------|-----|--------|----------|
| C38 | ruvector | `storage/file.rs` WAL commit flag never set — `commit_wal()` never marks `WalEntry.committed=true`. `recover_from_wal()` filters `!committed`, so on EVERY startup ALL WAL entries replay. | Deletions non-durable across restarts | R108 |
| C52 | memory-and-learning | 5 command injection vulnerabilities in `independent_verification_system.ts` — `execSync` with unvalidated input in `verifyHashExternally()`, `countFilesMethod1/2/3()` | Remote code execution via crafted input | R47 |
| C36/C66 | ruvector | Ed25519 hardcoded example keys + unencrypted private key storage in `key_management.rs` | Cryptographic security void | R108/R111 |
| C53 | agentdb-integration | Path traversal validation in `controller-registry.ts` uses `path.resolve()` to normalize `..` BEFORE comparing against allowed paths — making the check a no-op | Directory traversal attacks bypass validation | R136 |
| — | claude-flow-cli | `intelligence.ts` `compactPatterns()` does O(n²) cosine comparisons at `maxPatterns=5000` (12.5M operations), blocking event loop | Performance degradation at scale | R140 |
| — | claude-flow-cli | RuVector extension confusion: `setup.js` creates `ruvector(384)`, `init.js` creates `vector(${dim})`, `migrate.js` hardcodes `vector(1536)` — dimension mismatch across initialization paths | Silent data corruption on vector operations | R35 |
| — | process-spawning | `HeadlessWorkerExecutor` double timeout: two kill signals (SIGTERM + SIGKILL 5s later) fire even if first already killed process | Zombie process cleanup race condition | R140 |
| — | claude-flow-cli | Lazy loading nullified: `commands/index.ts` defines 31 `CommandLoader`s but synchronously imports all 31 at module parse time | Startup latency — all commands loaded regardless | R138 |
| — | claude-flow-cli | `config.js` zero persistence: all `init/get/set/export/import` are UI shells, config lost on restart | Configuration not durable | R135 |

---

## Entirely Fabricated Claims (High Confidence)

These features have zero genuine code backing them:

1. **LearningBridge** (0.12ms/insight) — No code exists. Zero files in DB.
2. **9 RL Algorithms** — All reduce to identical tabular Q-value updates. Cosmetic naming.
3. **IPFS Marketplace** — IPFS CID generation is FAKE (creates "Qm" + hash, not real IPFS CID). Cannot interoperate with actual IPFS.
4. **84.8% SWE-Bench Solve Rate** — Benchmark evaluator generates English prompts, cannot execute.
5. **Byzantine Consensus** — coordination.rs 15-25% FACADE. Vote files written, no voting logic.
6. **CRDT Synchronization** — "LWW timestamps, no vector clocks, no CRDTs" per findings.
7. **Int8 Quantization** "3.92x" — Returns empty Vec, ignores input.
8. **Agent Booster** "352x faster" — WASM stubs are console.log facades.
9. **Multi-Agent Collusion Detection** — No code found.
10. **"Eliminates 10,000+ duplicate lines"** via agentic-flow — agentic-flow is a single-node task runner (R40).
11. **V3 AgentDB Integration** — `agentdb-adapter.ts` is a plain `Map<string, MemoryEntry>` (R135). `memory-bridge.ts` (1,773 LOC of genuine code) is not compiled into npm dist (R135). Zero connection to real AgentDB.

---

## Coverage Progress on Previously Uncovered Areas

These areas were identified as uncovered at R89. R90-R101 addressed most of them:

| Area | At R89 | At R101 | Sessions | Verdict |
|------|--------|---------|----------|---------|
| **AIDefence** | 16 files, 0 DEEP | 16 files, 3 DEEP | R92 | 82-88%, real pattern matching |
| **Hyperbolic Geometry** | 63 files, 3 DEEP | 63 files, 21 DEEP | R92,R97-R101 | GENUINE across 4 crates, zero manifold validation |
| **SQL Attention** | 16 files, 0 DEEP | 16 files, 5 DEEP | R96-R97 | Arc COMPLETE, real pg_extern operators |
| **SONA crate** | 156 files, 3 DEEP | 156 files, 35 DEEP | R95,R98,R100 | ~75% (revised down from 85%), Loop C missing |
| **ruvector-gnn** | 33 files, 3 DEEP | 33 files, 15 DEEP | R91,R94,R99,R101 | Two parallel ecosystems, inference-only |
| **SWE-Bench adapter** | 21 files, 7 DEEP | 21 files, 7 DEEP | (pre-R89) | Cannot execute, no new coverage needed |
| **Flash Attention Rust** | 5 files, 2 DEEP | 5 files, 3 DEEP | R92 | CUDA-WASM flash_attention added |

### Remaining Uncovered Areas

| Area | Files | DEEP | Location | Priority |
|------|-------|------|----------|----------|
| **AIDefence** | 16 files | 3 DEEP (13 remaining) | npm/packages/ruvbot/src/security/ | LOW — core guard reviewed, remaining are docs/helpers |
| **SONA crate** | 156 files | 35 DEEP (116 NOT_TOUCHED) | crates/sona/ | MEDIUM — algorithms confirmed, remaining are internals |
| **ruvector-gnn** | 33 files | 15 DEEP (18 remaining) | crates/ruvector-gnn/ | LOW — inference-only, no training pipeline |
| **Hyperbolic** | 63 files | 21 DEEP (37 NOT_TOUCHED) | Various crates | LOW — all major modules COMPLETE |

### Middle Layer Coverage (R135-R140)

The Middle Layer deep-dive was the most architecturally clarifying work post-R112:

| Area | Sessions | Files Read | Key Discovery |
|------|----------|-----------|---------------|
| CLI Entry Points | R135 | 6 | ruvector/claude-flow/rvlite/ruvllm entry points traced |
| V3 Memory Layer | R136 | 7 | AgentDB adapter = Map, memory-bridge not in dist |
| Rust Integration Hubs | R137 | 5 | hnsw_router.rs BEST (90-93%), model_router.rs parallel to TS |
| MCP Tool Chain | R138 | 4 | SONA speedup fabricated, V2→V3 regression, dual MCP servers |
| CI/Tests/Deployment | R139 | 5 | CI facades (continue-on-error), all integration tests mock-only |
| Execution Engine | R140 | 7 | spawn('claude') primitive, intelligence.ts biggest facade |
| Compilation Audit | R141 | 115 crates | 87% compile, 3,984 tests pass |

---

## Summary: What claude-flow v4 Can Actually Deliver

### Genuinely deliverable (backed by real code):
- Self-learning via ReasoningBank + EWC++ + hooks
- HNSW vector search with PQ compression and conformal prediction
- Hyperbolic geometry (Poincare ball, Lorentz, mixed curvature — 21 DEEP files across 4 crates)
- Multi-agent coordination via Claude Code Task tool + MCP
- 175+ MCP tools, 42+ skills, 33 lifecycle hooks
- Raft consensus (from RAC crate)
- Path traversal security, Zod validation, AIDefence threat detection (82-88%)
- Model routing (3-tier via hooks)
- Sublinear PageRank, bit-parallel search, temporal analysis
- SQL attention operators (5 DEEP postgres functions)
- Bayesian agent-routing via sona-optimizer.ts (the one real V3 learning component)
- Cryptographic provenance via RVF witness chains (SHAKE-256, R122-R124)
- NAPI bridge path (R116-R117 proven working)

### Deliverable with moderate effort (real design, needs wiring):
- Gossip protocol (state machine correct, needs ~300 LOC transport)
- MoE routing (math is real, needs integration)
- Flash Attention (Rust implementation exists, needs benchmarking)
- Event sourcing (event bus exists, replay needs building)
- GNN layers (real Kipf & Welling math, needs unified ecosystem + training pipeline)
- Hyperbolic manifold validation (math exists, zero validation enforcement — needs ~100 LOC guard layer)
- Execution engine (HeadlessWorkerExecutor 78-83%, fix buildWorkerCommand context dropping)
- Hook pipeline pruning (remove fake token savings, keep real ADR-008 routing)

### NOT deliverable (fabricated, would need from-scratch implementation):
- 9 RL algorithms (all fake)
- IPFS marketplace (fake CID generation)
- Byzantine consensus (facade)
- CRDT synchronization (doesn't exist)
- SWE-Bench benchmarking (cannot execute)
- Agent Booster 352x speedup (theatrical WASM)
- Int8 quantization pipeline (placeholder)
- Multi-agent collusion detection (no code)
- LearningBridge (no code)
- V3 AgentDB integration (plain Map, bridge not compiled)
- V3 CI/CD as validation (all continue-on-error, cannot trust green builds)
- intelligence.ts as learning engine (O(n) facade with 14+ consumers)
