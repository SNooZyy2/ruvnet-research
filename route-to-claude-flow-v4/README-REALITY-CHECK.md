# claude-flow README vs Reality — Feature-by-Feature Verdict

> Cross-referenced against 106 research sessions, 1,466 DEEP files, ~10,400 findings.
> Each feature classified as: GENUINE / PARTIALLY REAL / FABRICATED / UNCOVERED
> Date: 2026-02-18

---

## Self-Learning & Intelligence

The headline feature. Can we deliver it?

| README Claim | Verdict | Evidence | Can We Rebuild? |
|-------------|---------|----------|----------------|
| **ReasoningBank** (RETRIEVE→JUDGE→DISTILL) | **GENUINE** | TS+Rust both 92-95%. 160 findings across 101 sessions. Real DeepMind-style algorithms (MaTTS search, MMR). Mature v1→v2 migration. | YES — copy both TS and Rust implementations |
| **EWC++** (prevents catastrophic forgetting) | **GENUINE** | micro_lora.rs 92-95%. Real online Fisher, adaptive lambda, task boundary detection. NEON SIMD. R106: training.rs orchestration has correct EWC penalty math (λ*F*(w-w*)²/2) but Fisher information is never updated during training — making it static EWC, not truly "++". The micro_lora.rs core is genuine; the training loop's continual adaptation is incomplete. | YES — copy from Rust crate; wire Fisher updates for true EWC++ |
| **SONA** (<0.05ms adaptation) | **PARTIALLY REAL** | sona crate ~75% production-ready (R98 revised down from 85%). Algorithms 85-90%, orchestration 60-70%. LoRA+EWC+++federated+SafeTensors present. Loop C missing from exports (R100). <0.05ms unverified. | PARTIALLY — algorithms work, orchestration incomplete, performance claims inflated |
| **MoE** (8 expert networks) | **PARTIALLY REAL** | moe-router.js has real 2-layer gating (384→128→8) with Xavier init. But it's a JS file, not integrated into any training pipeline. | MAYBE — the routing math is real, needs wiring |
| **LearningBridge** (0.12ms/insight) | **FABRICATED** | Zero files found in DB. No code exists. | NO — write from scratch if needed |
| **9 RL Algorithms** (PPO, A2C, DQN, etc.) | **FABRICATED** | All 9 reduce to identical tabular Q-value updates. DQN has no neural network. PPO has no clipping. Decision Transformer has no transformer. Cosmetic naming only. | NO — would need real RL implementations |

**Self-learning verdict**: YES, we can rebuild real self-learning. ReasoningBank + EWC++ + hooks gives you:
- Pattern storage and retrieval from past decisions
- Verdict judgment on whether past patterns apply
- Memory consolidation without forgetting old patterns
- Automatic pattern application via pre-task hooks

What you WON'T get: the inflated performance claims (<0.05ms), the 9 RL algorithms, or the "LearningBridge."

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
| **AgentMemoryScope** (3-scope system) | **PARTIALLY REAL** | Memory system works at basic level (better-sqlite3). But 11 disconnected persistence layers (R106: serving KV cache = 11th) means the "3-scope" claim hides architectural chaos. R104: ruvllm context module COMPLETE (7/7 DEEP) has 4+ genuine HNSW stores that NEVER compose at runtime — each sibling (semantic_cache, episodic_memory, agentic_memory, working_memory) maintains its own independent vector index. | Rebuild with single persistence (ADR-v4-002) |

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
| **Int8 Quantization** "3.92x memory reduction" | **FABRICATED** | R82: quantization.rs 75-78%, R87: inference/quantization.rs 0-5% PLACEHOLDER (returns empty Vec). No working quantization pipeline. | NO — placeholder code |
| **LoRA Compression** "128x" | **PARTIALLY REAL** | micro_lora.rs IS genuine (92-95%). But "128x compression" is a theoretical maximum, not demonstrated. | YES for LoRA, NO for 128x claim |
| **Token Optimizer** "30-50% reduction" | **PARTIALLY REAL** | Hook system is genuine (98.1% R19). Token optimization hooks exist. But 30-50% reduction claim is unverified. | MAYBE — hooks work, savings unproven |
| **SemanticRouter** "34,798 routes/s" | **UNCOVERED** | 1 DEEP file (semantic-router.js, 178 LOC). Not deeply assessed. Performance claim unverified. | Unknown |
| **84.8% SWE-Bench** | **FABRICATED** | SWE-bench evaluator exists (991 LOC DEEP) but findings show `build_command` generates English prompts, not actual CLI flags. Benchmark cannot execute. | NO — benchmark scaffolding exists but cannot run |

---

## Swarm Coordination

| README Claim | Verdict | Evidence | Can We Rebuild? |
|-------------|---------|----------|----------------|
| **6 Topology Patterns** | **PARTIALLY REAL** | CLI defines hierarchical, mesh, ring, star, hybrid, adaptive. R31/R71: CLI = demo framework. But Claude Code's Task tool actually does the execution — topologies are labels, not protocol implementations. | YES — topology is a coordination pattern, not complex code |
| **Byzantine Consensus** (2/3 majority) | **FABRICATED** | R84: coordination.rs 15-25% FACADE. No actual Byzantine fault detection or 2/3 majority voting. Vote files written but no voting logic. | NO — would need real implementation |
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

### Deliverable with moderate effort (real design, needs wiring):
- Gossip protocol (state machine correct, needs ~300 LOC transport)
- MoE routing (math is real, needs integration)
- Flash Attention (Rust implementation exists, needs benchmarking)
- Event sourcing (event bus exists, replay needs building)
- GNN layers (real Kipf & Welling math, needs unified ecosystem + training pipeline)
- Hyperbolic manifold validation (math exists, zero validation enforcement — needs ~100 LOC guard layer)

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
