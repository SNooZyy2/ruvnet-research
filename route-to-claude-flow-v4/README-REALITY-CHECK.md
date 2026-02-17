# claude-flow README vs Reality — Feature-by-Feature Verdict

> Cross-referenced against 90 research sessions, 1,332 DEEP files, 9,171 findings.
> Each feature classified as: GENUINE / PARTIALLY REAL / FABRICATED / UNCOVERED
> Date: 2026-02-17

---

## Self-Learning & Intelligence

The headline feature. Can we deliver it?

| README Claim | Verdict | Evidence | Can We Rebuild? |
|-------------|---------|----------|----------------|
| **ReasoningBank** (RETRIEVE→JUDGE→DISTILL) | **GENUINE** | TS+Rust both 92-95%. 160 findings across 90 sessions. Real DeepMind-style algorithms (MaTTS search, MMR). Mature v1→v2 migration. | YES — copy both TS and Rust implementations |
| **EWC++** (prevents catastrophic forgetting) | **GENUINE** | micro_lora.rs 92-95%. Real online Fisher, adaptive lambda, task boundary detection. NEON SIMD. | YES — copy from Rust crate |
| **SONA** (<0.05ms adaptation) | **PARTIALLY REAL** | sona crate is 85% production-ready (~4,500 LOC). LoRA+EWC+++federated+SafeTensors all present. But <0.05ms claim is unverified — no benchmark proves this. | PARTIALLY — the crate works, performance claims are inflated |
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
| **39 Attention Mechanisms in SQL** | **UNCOVERED** | 0 DEEP files on SQL attention specifically. 3 files exist in DB. Likely PostgreSQL CREATE FUNCTION statements — may be syntactically valid SQL but untested. | Unknown — needs investigation |
| **15 GNN Layer Types** | **PARTIALLY REAL** | ruvector-gnn has real training.rs (1,368 LOC DEEP), tensor.rs, query.rs. But findings show "GNN enhancement is syntactic sugar — calls opaque backend.enhance()". JS fallback needed for broken native APIs. | PARTIALLY — some real code, integration broken |
| **MemoryGraph with PageRank** | **PARTIALLY REAL** | backward_push.rs is genuine O(1/ε) (95%+). intelligence.cjs has real PageRank with power iteration. But "MemoryGraph" as an integrated product doesn't exist. | YES for PageRank algorithm, NO for integrated MemoryGraph |
| **Hyperbolic Geometry** (Poincaré ball) | **PARTIALLY REAL** | 63 files exist, 0 DEEP. Coordinator docs describe real hyperbolic attention but findings note it's in documentation/design, not deployed code. | Unknown — needs deep reads |
| **AgentMemoryScope** (3-scope system) | **PARTIALLY REAL** | Memory system works at basic level (better-sqlite3). But 9 disconnected persistence layers means the "3-scope" claim hides architectural chaos. | Rebuild with single persistence (ADR-v4-002) |

---

## Performance Claims

| README Claim | Verdict | Evidence | Can We Rebuild? |
|-------------|---------|----------|----------------|
| **Agent Booster (WASM)** "352x faster" | **FABRICATED** | agent-booster-enhanced.ts (1,428 LOC DEEP) exists, but 13 theatrical WASM stubs found. "352x" has no benchmark. The WASM stubs we found are console.log facades. | NO — the 352x claim is baseless |
| **Flash Attention** "2.49x-7.47x" | **PARTIALLY REAL** | flash_attention.rs in ruvector-mincut-gated-transformer (997 LOC DEEP) and JS fallback (643 LOC). R34 found this crate is "MOST NOVEL." Performance numbers unverified. | MAYBE — the algorithm exists, speedup claims unverified |
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
| **AIDefence** (<10ms threat detection) | **UNCOVERED** | 0 DEEP files. 16 files exist. 1 finding: "npm-published package excludes aidefence module." The module exists but was excluded from the published package. | Unknown — needs investigation |
| **Input Validation** (Zod) | **GENUINE** | config-loader.ts 92-95% with Zod. input-validator.ts exists in self-impl (270 LOC). | YES — already have this |
| **Path Traversal Prevention** | **GENUINE** | R88: RuVectorBackend has FORBIDDEN_PATH_PATTERNS, validatePath() on every op. | YES — copy from RuVectorBackend |
| **HMAC-SHA256 Proof Chain** | **PARTIALLY REAL** | guidance_kernel has HMAC but with HARDCODED key (security concern from early sessions). Concept real, implementation has a critical flaw. | MAYBE — fix the hardcoded key |
| **Command Sandboxing** | **PARTIALLY REAL** | Hook system has pre-command hooks. But the "allowlisted commands" claim depends on configuration, not enforcement in code. | PARTIALLY — hooks provide the mechanism |

---

## Model Routing

| README Claim | Verdict | Evidence | Can We Rebuild? |
|-------------|---------|----------|----------------|
| **3-Tier Routing** (WASM/Haiku/Opus) | **PARTIALLY REAL** | 6 PARALLEL routing systems found — none properly connected to each other. The concept works through claude-flow hooks. | YES — already works via hooks, just needs consolidation |
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

## Where to Find More Genuine Code (Uncovered Areas)

These areas have code in the repos but weren't DEEP-read. They could contain additional genuine implementations:

| Area | Files | Location | Priority |
|------|-------|----------|----------|
| **AIDefence** | 16 files, 0 DEEP | Excluded from npm publish but code exists | HIGH — if genuine, adds security layer |
| **Hyperbolic Geometry** | 63 files, 0 DEEP | Various locations | MEDIUM — could be real math like PQ/conformal |
| **SQL Attention Mechanisms** | 3 files, 0 DEEP | PostgreSQL functions | LOW — likely CREATE FUNCTION statements |
| **SONA crate internals** | 156 files, 3 DEEP | crates/sona/ | MEDIUM — 85% production-ready per findings |
| **ruvector-gnn** | 130 files, 3 DEEP | crates/ruvector-gnn/ | MEDIUM — training.rs is DEEP, but integration broken |
| **SWE-Bench adapter** | 21 files, 3 DEEP | ruv-swarm/crates/swe-bench-adapter/ | LOW — cannot execute benchmarks |
| **Flash Attention Rust** | 5 files, 2 DEEP | ruvector-mincut-gated-transformer | MEDIUM — could provide real speedups |

### Recommended Deep-Read Priority for v4

If you want to expand v4's feature set beyond the current spec, read these first:
1. **AIDefence** (16 files) — most impactful if genuine
2. **SONA crate** (156 files, only 3 DEEP) — already 85% production-ready
3. **Flash Attention Rust** (5 files) — direct performance benefit
4. **ruvector-gnn** (130 files) — if the GNN layers work, adds ML capabilities
5. **Hyperbolic geometry** (63 files) — if genuine, adds hierarchical code understanding

---

## Summary: What claude-flow v4 Can Actually Deliver

### Genuinely deliverable (backed by real code):
- Self-learning via ReasoningBank + EWC++ + hooks
- HNSW vector search with PQ compression and conformal prediction
- Multi-agent coordination via Claude Code Task tool + MCP
- 175+ MCP tools, 42+ skills, 33 lifecycle hooks
- Raft consensus (from RAC crate)
- Path traversal security, Zod validation
- Model routing (3-tier via hooks)
- Sublinear PageRank, bit-parallel search, temporal analysis

### Deliverable with moderate effort (real design, needs wiring):
- Gossip protocol (state machine correct, needs ~300 LOC transport)
- MoE routing (math is real, needs integration)
- Flash Attention (Rust implementation exists, needs benchmarking)
- Event sourcing (event bus exists, replay needs building)

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
