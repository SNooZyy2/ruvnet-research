# Cross-Validation Plan

Generated: 2026-03-01
Sources: live-february-26-verification-plan.md (22 leads), ai-hackerspace-march-01-verification-plan.md (22 leads)

## Summary

| Metric | live-february-26 | ai-hackerspace-march-01 | Combined | After Merge |
|--------|------------------|-------------------------|----------|-------------|
| Total leads | 22 | 22 | 44 | 44 |
| ALREADY_COVERED | 4 | 4 | 8 | 8 (skip) |
| PARTIALLY_COVERED | 7 | 5 | 12 | 12 |
| NEW | 8 | 7 | 15 | 15 |
| CONTRADICTION | 1 | 3 | 4 | 4 |
| UNRESOLVABLE | 2 | 3 | 5 | 5 (skip) |
| Actionable leads | 16 | 15 | 31 | 31 |
| Files to read (naive) | ~42 | ~18-25 | ~60-67 | — |
| Files to read (deduped) | — | — | — | ~50 |
| Estimated sessions (naive) | 4-5 | 2-3 | 6-8 | **5** |

## File Deduplication

Zero direct file overlap between the two plans. They target entirely different file sets:

- **Feb26** targets Rust infrastructure: ruvector-raft, ruvector-delta-consensus, ruvector-replication, RVF internals (qr_seed, agi_container, rvf-ebpf), ruvector-solver, rvf-adapters/agentdb, causal atlas examples, npm/ruvector CLI
- **March01** targets JS/application layer: ruvbot, edge-net pkg/ JS files, WASM pkg output files, markov.rs, plus re-reads of DEEP files (micro_lora.rs, EmbeddingService.ts, daemon.js, hive-mind.js)

Savings come from **thematic clustering**, not file deduplication.

## Thematic Clusters

### Cluster A: Consensus & Distribution

Leads from both transcripts that address distributed coordination.

| Lead | Source | Claim Summary |
|------|--------|---------------|
| Feb26-LEAD-005 | live-february-26 | Raft, DAG, CRDT, vector clocks, 64-shard hash ring |
| Feb26-LEAD-014 | live-february-26 | Hierarchical grid swarm with gossip + Raft |
| Feb26-LEAD-019 | live-february-26 | QUIC for delta sync + Raft for consensus |
| Mar01-LEAD-016 | ai-hackerspace-march-01 | Three-tier agents: 15 primary + headless + daemon |
| Mar01-LEAD-020 | ai-hackerspace-march-01 | Persistent daemon monitoring sessions |

### Cluster B: WASM Ecosystem

Both transcripts claim broad WASM capabilities; existing research shows 6:4 genuine/theatrical split.

| Lead | Source | Claim Summary |
|------|--------|---------------|
| Feb26-LEAD-007 | live-february-26 | Self-learning solver with sublinear attention + SIMD in WASM |
| Mar01-LEAD-013 | ai-hackerspace-march-01 | ALL ruvector packages compile to WASM |
| Mar01-LEAD-003 | ai-hackerspace-march-01 | WASM plugin system with ed25519 signing |

### Cluster C: AgentDB-ruvector Relationship

Two leads addressing the same bridge from opposite directions.

| Lead | Source | Claim Summary |
|------|--------|---------------|
| Feb26-LEAD-009 | live-february-26 | AgentDB RVF migration path (JSON → SQLite → sql.js → RVF) |
| Mar01-LEAD-008 | ai-hackerspace-march-01 | "AgentDB is a simplified ruvector" (creator statement) |

### Cluster D: Optimizer / ruvbot / NPX

Multiple leads that likely resolve from reading one package (ruvbot).

| Lead | Source | Claim Summary |
|------|--------|---------------|
| Mar01-LEAD-001 | ai-hackerspace-march-01 | Optimizer uses ruvector attention/hyperbolics |
| Mar01-LEAD-002 | ai-hackerspace-march-01 | Ruvector memory in optimizer startup |
| Mar01-LEAD-003 | ai-hackerspace-march-01 | WASM plugin with ed25519 (ruvbot ADR) |
| Mar01-LEAD-006 | ai-hackerspace-march-01 | Self-training metrics 84%/12% — hardcoded or real? |
| Feb26-LEAD-022 | live-february-26 | NPX ruvector with RVF runtime |

### Cluster E: Edge-net Browser Compute

Edge-net distributed compute claims spanning both transcripts.

| Lead | Source | Claim Summary |
|------|--------|---------------|
| Mar01-LEAD-010 | ai-hackerspace-march-01 | Browser swarm shared compute via edge-net |
| Feb26-LEAD-002 | live-february-26 | Sync API with vector deltas (shares edge-net context) |

### Standalone Leads (no cross-transcript clustering)

These appear in only one plan and don't overlap thematically:

| Lead | Source | Cluster | Claim Summary |
|------|--------|---------|---------------|
| Feb26-LEAD-003 | live-february-26 | RVF internals | RVF binary with COW, layers, 7-bit quantization |
| Feb26-LEAD-004 | live-february-26 | RVF internals | QR code seed format (2000 bytes) |
| Feb26-LEAD-010 | live-february-26 | Transformers | New graph transformer crates (sublinear, proof-gated) |
| Feb26-LEAD-018 | live-february-26 | RVF internals | Causal Atlas blind test with witness chains |
| Feb26-LEAD-008 | live-february-26 | MinCut | MinCut used for RVF/system boundaries |
| Feb26-LEAD-011 | live-february-26 | Graph query | Algebraic graph traversal via sparse matrices |
| Feb26-LEAD-012 | live-february-26 | Replication | PostgreSQL clustering and replication |
| Mar01-LEAD-005 | ai-hackerspace-march-01 | ML | Markov chain-based app launch prediction |
| Mar01-LEAD-014 | ai-hackerspace-march-01 | Routing | Dynamic model routing per skill |
| Mar01-LEAD-015 | ai-hackerspace-march-01 | Contradiction | Unified vector-graph-attention pipeline |
| Mar01-LEAD-017 | ai-hackerspace-march-01 | Contradiction | Micro LoRAs for embedding model training |
| Mar01-LEAD-018 | ai-hackerspace-march-01 | Swarm | Hive queen with dynamic skill discovery |

## Corroborating Leads

Claims that appear in both transcripts, strengthening each other:

| Theme | Feb26 Lead | Mar01 Lead | Agreement |
|-------|-----------|-----------|-----------|
| Delta sync | LEAD-002 (sync API with vector deltas) | LEAD-010 (shared compute via deltas) | Both describe delta-based peer communication |
| Raft consensus | LEAD-005 (Raft + CRDT crates) | LEAD-016 (three-tier with daemon) | Both describe Raft as coordination layer |
| RVF deployment | LEAD-022 (NPX runtime) | LEAD-003 (WASM plugin delivery) | Both describe RVF as deployable binary format |
| AgentDB wraps ruvector | LEAD-009 (migration path) | LEAD-008 (simplified ruvector) | Consistent: AgentDB wraps/simplifies ruvector |

## Inter-Plan Contradictions

| Issue | Feb26 | Mar01 | Resolution |
|-------|-------|-------|------------|
| WASM scope | LEAD-007 targets specific solver WASM (4 files) | LEAD-013 claims ALL packages compile to WASM | Mar01 is the stronger claim — a single WASM audit session settles both |

No other inter-plan contradictions found. The two transcripts are complementary (Rust layer vs JS/application layer) rather than contradictory.

## Research Contradictions (vs existing DB findings)

These are leads that contradict established research findings. Highest priority for verification.

| Lead | Source | Claim | Contradicts |
|------|--------|-------|-------------|
| Mar01-LEAD-015 | ai-hackerspace-march-01 | Unified vector-graph-attention pipeline | 4+ HNSW stores never compose (R104), 12+ parallel subsystems |
| Mar01-LEAD-013 | ai-hackerspace-march-01 | ALL packages compile to WASM | WASM 6:4 genuine/theatrical (R51-R60) |
| Mar01-LEAD-017 | ai-hackerspace-march-01 | Micro LoRAs train embedding models | Hash-based embeddings SYSTEMIC, micro_lora targets LM weights not embedding weights |
| Feb26-LEAD-017 | live-february-26 | LOC + AI mischaracterization of capabilities | Nuanced: postgres real alongside facades, "millions of LOC" CONFIRMED |

---

## Merged Session Plan

### R115: Consensus & Distribution (Cluster A + standalone replication)

**Leads resolved:** Feb26-005, Feb26-012, Feb26-014, Feb26-019, Mar01-016, Mar01-020

**New files to read:**

| # | File | Package | LOC | Lead(s) |
|---|------|---------|-----|---------|
| 1 | crates/ruvector-raft/src/node.rs | ruvector-rust | 632 | Feb26-005, Feb26-014 |
| 2 | crates/ruvector-raft/src/election.rs | ruvector-rust | 360 | Feb26-005 |
| 3 | crates/ruvector-raft/src/log.rs | ruvector-rust | 351 | Feb26-005 |
| 4 | crates/ruvector-raft/src/rpc.rs | ruvector-rust | 443 | Feb26-005 |
| 5 | crates/ruvector-delta-consensus/src/lib.rs | ruvector-rust | 489 | Feb26-002, Feb26-005 |
| 6 | crates/ruvector-delta-consensus/src/crdt.rs | ruvector-rust | 486 | Feb26-005 |
| 7 | crates/ruvector-delta-consensus/src/causal.rs | ruvector-rust | 320 | Feb26-005 |
| 8 | crates/ruvector-delta-consensus/src/conflict.rs | ruvector-rust | 286 | Feb26-002 |
| 9 | crates/ruvector-replication/src/replica.rs | ruvector-rust | 379 | Feb26-012 |
| 10 | crates/ruvector-replication/src/sync.rs | ruvector-rust | 375 | Feb26-012 |
| 11 | crates/ruvector-replication/src/stream.rs | ruvector-rust | 404 | Feb26-012 |

**Re-reads (DEEP files, targeted sections):**

| # | File | Package | Focus |
|---|------|---------|-------|
| 1 | dist/src/services/worker-daemon.js | claude-flow-cli | Daemon monitoring + session hooks (Mar01-016, 020) |
| 2 | dist/src/commands/daemon.js | claude-flow-cli | Daemon lifecycle (Mar01-020) |

**Additional tasks:**
- npm/packages/replication/src/vector-clock.ts (149 LOC) — Feb26-005
- tests/integration/distributed/sharding_tests.rs (398 LOC) — Feb26-005

**Total: ~13 new reads + 2 re-reads, ~4,900 LOC**

**What we're answering:**
- Are ruvector-raft and ruvector-delta-consensus genuine or facades?
- Does Raft integrate with CRDT layer?
- Do vector clocks exist in runtime code?
- Is 64-shard consistent hash ring implemented?
- Does the JS daemon actually spawn persistent workers?
- Does replication implement real Postgres logical replication?

---

### R116: WASM Audit + Edge-net Browser (Clusters B + E)

**Leads resolved:** Feb26-007, Mar01-003, Mar01-010, Mar01-013

**New files to read:**

| # | File | Package | LOC | Lead(s) |
|---|------|---------|-----|---------|
| 1 | crates/ruvector-solver/src/true_solver.rs | ruvector-rust | 933 | Feb26-007 |
| 2 | crates/ruvector-solver/src/router.rs | ruvector-rust | 1615 | Feb26-007 |
| 3 | crates/ruvector-solver/src/simd.rs | ruvector-rust | 282 | Feb26-007 |
| 4 | crates/rvf/rvf-solver-wasm/src/engine.rs | ruvector-rust | 784 | Feb26-007 |
| 5 | crates/ruvector-attention-unified-wasm/pkg/ruvector_attention_unified_wasm.js | ruvector-rust | 2752 | Mar01-013 |
| 6 | crates/ruvector-nervous-system-wasm/pkg/ruvector_nervous_system_wasm.js | ruvector-rust | 1648 | Mar01-013 |
| 7 | examples/edge-full/pkg/rvlite/rvlite.js | ruvector-rust | 2367 | Mar01-013 |
| 8 | examples/edge-net/pkg/join.js | ruvector-rust | 1334 | Mar01-010 |
| 9 | examples/edge-net/pkg/contribute-daemon.js | ruvector-rust | 740 | Mar01-003, Mar01-010 |
| 10 | examples/edge-net/pkg/real-agents.js | ruvector-rust | 1289 | Mar01-010 |
| 11 | examples/edge-net/pkg/monitor.js | ruvector-rust | 676 | Mar01-010 |

**Re-reads:**

| # | File | Package | Focus |
|---|------|---------|-------|
| 1 | crates/rvlite/src/lib.rs | ruvector-rust | Plugin signing references (Mar01-003) |

**Additional tasks:**
- Cargo workspace scan: `grep -r 'wasm32-unknown-unknown' */Cargo.toml` — count genuine WASM targets (Mar01-013)

**Total: ~11 new reads + 1 re-read + 1 scan, ~14,400 LOC**

**What we're answering:**
- Is ruvector-solver genuine self-reinforcement via puzzle generation?
- Is solver SIMD real (like ruvector-core AVX) or decorative?
- Does true_solver.rs achieve sublinear complexity?
- How many crates have actual wasm32 targets in Cargo.toml?
- Are WASM pkg/ outputs genuine wasm-bindgen or stubs?
- Does edge-net implement WebRTC/WebSocket compute pooling for browsers?
- Is contribute-daemon.js a compute contribution daemon?
- Is there a WASM plugin store with ed25519 signing?

---

### R117: RVF Deep Dive (standalone RVF leads)

**Leads resolved:** Feb26-003, Feb26-004, Feb26-008, Feb26-010, Feb26-018

**New files to read:**

| # | File | Package | LOC | Lead(s) |
|---|------|---------|-----|---------|
| 1 | crates/rvf/rvf-runtime/src/qr_encode.rs | ruvector-rust | 1135 | Feb26-004 |
| 2 | crates/rvf/rvf-runtime/src/qr_seed.rs | ruvector-rust | 1095 | Feb26-004 |
| 3 | crates/rvf/rvf-runtime/src/seed_crypto.rs | ruvector-rust | 236 | Feb26-004 |
| 4 | crates/rvf/rvf-types/src/qr_seed.rs | ruvector-rust | 379 | Feb26-004 |
| 5 | crates/rvf/rvf-types/src/agi_container.rs | ruvector-rust | 963 | Feb26-003 |
| 6 | crates/rvf/rvf-ebpf/src/lib.rs | ruvector-rust | 1083 | Feb26-003 |
| 7 | examples/rvf/examples/causal_atlas.rs | ruvector-rust | 479 | Feb26-018 |
| 8 | examples/rvf/examples/causal_atlas_dashboard.rs | ruvector-rust | 585 | Feb26-018 |
| 9 | examples/delta-behavior/applications/04-self-stabilizing-world-model.rs | ruvector-rust | 554 | Feb26-010 |

**Re-reads:**

| # | File | Package | Focus |
|---|------|---------|-------|
| 1 | crates/rvf/rvf-runtime/src/cow.rs | ruvector-rust | COW mechanism details (Feb26-003) |

**Additional tasks:**
- Mapper pass: trace MinCut → RVF crate dependencies (Feb26-008)
- Cross-repo-tracer: search for `graph_transformer`, `proof_gated`, `sublinear_attention` crate names (Feb26-010)

**Total: ~9 new reads + 1 re-read + 2 tracer tasks, ~6,500 LOC**

**What we're answering:**
- Does qr_encode.rs produce a QR-encodable 2000-byte seed?
- Does qr_seed.rs implement bootstrap/expansion from seed?
- Does the eBPF subsystem execute embedded programs?
- Does agi_container.rs implement the layered binary structure?
- Does 7-bit quantization exist in the Rust RVF runtime?
- Does any code connect MinCut to RVF boundary detection?
- Do "sublinear attention" and "proof-gated attention" crates exist?
- Does the causal atlas use real crypto or the fake simple_shake256_256?

---

### R118: AgentDB Bridge + ruvbot/NPX (Clusters C + D)

**Leads resolved:** Feb26-009, Feb26-022, Mar01-001, Mar01-002, Mar01-006, Mar01-008

**New files to read:**

| # | File | Package | LOC | Lead(s) |
|---|------|---------|-----|---------|
| 1 | npm/packages/ruvbot/src/RuvBot.ts | ruvector-rust | 781 | Mar01-001, 002, 006 |
| 2 | npm/packages/ruvbot/README.md | ruvector-rust | 1400 | Mar01-001 |
| 3 | npm/packages/ruvbot/docs/adr/ADR-006-wasm-integration.md | ruvector-rust | 776 | Mar01-003 |
| 4 | npm/packages/ruvector/bin/cli.js | ruvector-rust | 7357 | Feb26-022 |
| 5 | npm/packages/rvf-mcp-server/src/server.ts | ruvector-rust | 569 | Feb26-022 |
| 6 | packages/agentdb/docs/adrs/ADR-003-rvf-native-format-integration.md | agentic-flow | 1400 | Feb26-009 |
| 7 | crates/rvf/rvf-adapters/agentdb/src/pattern_store.rs | ruvector-rust | 457 | Feb26-009 |
| 8 | crates/rvf/rvf-adapters/agentdb/src/vector_store.rs | ruvector-rust | 327 | Feb26-009 |

**Re-reads:**

| # | File | Package | Focus |
|---|------|---------|-------|
| 1 | packages/agentdb/src/cli/agentdb-cli.ts | agentic-flow | Imports from ruvector (Mar01-008) |
| 2 | src/cli/agentdb-cli.ts | agentdb | Compare ruvector imports (Mar01-008) |

**Additional tasks:**
- Cross-repo-tracer: AgentDB → ruvector import chain (Mar01-008)

**Total: ~8 new reads + 2 re-reads + 1 tracer, ~13,000 LOC**

**What we're answering:**
- Is ruvbot the "optimizer" desktop app?
- Does ruvbot import ruvector attention/hyperbolic modules?
- Are self-training metrics (84%/12%) hardcoded or computed?
- Does the NPX CLI load and execute RVF files?
- Is the rvf-mcp-server functional?
- Does the Rust rvf-adapters/agentdb bridge to TS AgentDB?
- Is AgentDB a subset (simpler API) or a fork (diverged codebase)?

---

### R119: Contradictions + Remaining Leads

**Leads resolved:** Mar01-015, Mar01-017, Mar01-005, Mar01-014, Mar01-018, Feb26-011

**New files to read:**

| # | File | Package | LOC | Lead(s) |
|---|------|---------|-----|---------|
| 1 | examples/vibecast-7sense/crates/sevensense-analysis/src/infrastructure/markov.rs | ruvector-rust | 525 | Mar01-005 |

**Re-reads (DEEP files, targeted sections):**

| # | File | Package | Focus |
|---|------|---------|-------|
| 1 | crates/ruvllm/src/lora/micro_lora.rs | ruvector-rust | Does it target embedding layers? (Mar01-017) |
| 2 | crates/ruvllm/src/bitnet/rlm_embedder.rs | ruvector-rust | Is LoRA applied to embedding weights? (Mar01-017) |
| 3 | crates/ruvllm/src/lora/training.rs | ruvector-rust | Training targets (Mar01-017) |
| 4 | agentic-flow/src/intelligence/RuVectorIntelligence.ts | agentic-flow | Vector-graph-attention composition (Mar01-015) |
| 5 | agentic-flow/src/intelligence/EmbeddingService.ts | agentic-flow | ONNX + HNSW + graph composition (Mar01-015) |
| 6 | crates/ruvllm/src/claude_flow/model_router.rs | ruvector-rust | Skill-based routing (Mar01-014) |
| 7 | dist/src/commands/hive-mind.js | claude-flow-cli | Skill discovery in hive-mind (Mar01-018) |

**Additional tasks:**
- Cross-repo-tracer: search for ANY code path composing vector search + graph traversal + attention scoring (Mar01-015)

**Total: ~1 new read + 7 re-reads + 1 tracer, ~9,000 LOC**

**What we're answering:**
- Does micro_lora.rs have any code path targeting embedding model parameters?
- Is there ANY composing pipeline for vector + graph + attention?
- Is the Markov chain implementation generic or app-launch specific?
- Are skills defined with explicit model associations?
- Does hive-mind.js enumerate and select skills dynamically?

---

## Leads NOT Requiring Sessions

### ALREADY_COVERED (8 leads — skip entirely)

| Lead | Source | Why Covered |
|------|--------|-------------|
| Feb26-LEAD-013 | live-february-26 | Personal 6GB memory store — private data, federated infra documented |
| Feb26-LEAD-015 | live-february-26 | Dev pipeline ruvector → AgentDB → claude-flow — extensively documented |
| Feb26-LEAD-016 | live-february-26 | Transfer learning — spread across 3 implementations, all DEEP |
| Feb26-LEAD-021 | live-february-26 | MCP device on Raspberry Pi — 7+ MCP implementations documented |
| Mar01-LEAD-004 | ai-hackerspace-march-01 | AIDefence PII detection — R92 confirmed 82-88% genuine |
| Mar01-LEAD-009 | ai-hackerspace-march-01 | Edge-net in-browser agents — RAC, P2P, SIMD all DEEP |
| Mar01-LEAD-019 | ai-hackerspace-march-01 | Time-travel checkpoints — temporal-tensor 93%, git checkpoints real |
| Mar01-LEAD-022 | ai-hackerspace-march-01 | rvlite plugin discovery — rvlite DEEP, plugin is claude-flow feature |

### UNRESOLVABLE (5 leads — cannot verify from available codebases)

| Lead | Source | Why Unresolvable |
|------|--------|-----------------|
| Feb26-LEAD-006 | live-february-26 | Raspberry Pi Zero appliance — private/commercial repo |
| Feb26-LEAD-020 | live-february-26 | Stuart's "ask-roof-net" — external community project |
| Mar01-LEAD-007 | ai-hackerspace-march-01 | "Roof bot" = agentic-flow — optimizer app not in tracked repos |
| Mar01-LEAD-011 | ai-hackerspace-march-01 | 150,000 active users — business metric |
| Mar01-LEAD-021 | ai-hackerspace-march-01 | Robert's unicorn-scan integration — external project |

### CAN BE ANSWERED NOW (1 lead)

| Lead | Source | Answer |
|------|--------|--------|
| Mar01-LEAD-012 | ai-hackerspace-march-01 | ruvector-rust: 2,473,230 LOC / 6,035 files. All packages: 5,280,944 LOC / 14,010 files. "Close to two million" is ACCURATE for ruvector-rust alone. |

---

## Session Totals

| Session | New Reads | Re-reads | Tracer Tasks | LOC | Leads Resolved |
|---------|-----------|----------|--------------|-----|----------------|
| R115 | 13 | 2 | 0 | ~4,900 | 6 |
| R116 | 11 | 1 | 1 (cargo scan) | ~14,400 | 4 |
| R117 | 9 | 1 | 2 (mapper, tracer) | ~6,500 | 5 |
| R118 | 8 | 2 | 1 (tracer) | ~13,000 | 6 |
| R119 | 1 | 7 | 1 (tracer) | ~9,000 | 6 |
| **Total** | **42** | **13** | **5** | **~47,800** | **27 + 1 answered + 8 skipped + 5 unresolvable = 41** |

3 leads answered by existing coverage without dedicated sessions (Feb26-LEAD-011 algebraic graph, Feb26-LEAD-017 LOC count, Mar01-LEAD-012 LOC count).
