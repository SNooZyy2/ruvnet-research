# R50 Execution Plan: ruv-swarm Rust Crates + Goalie Security + Backend Reality Check + SWE-Bench

**Date**: 2026-02-15
**Session ID**: 50
**Focus**: First examination of ruv-swarm Rust crates (persistence, transport, WASM SIMD, CLI, SWE-Bench adapter), goalie crypto and anti-hallucination verification, AgentDB RuVectorBackend reality, ReasoningBank Rust SQLite storage
**Parallel with**: R49 (no file overlap -- R50 = ruv-swarm Rust/goalie security/AgentDB backend/ReasoningBank Rust, R49 = consciousness final/neural-pattern-recognition completion/ReasoningBank WASM/consciousness-explorer validators)

## IMPORTANT: Parallel Execution Notice

This plan runs IN PARALLEL with R49. The file lists are strictly non-overlapping:
- **R50 covers**: ruv-fann-rust ruv-swarm Rust crates (persistence memory, transport protocol, WASM SIMD, CLI spawn, SWE-Bench evaluation), sublinear-rust goalie (ed25519-verifier, anti-hallucination-plugin, perplexity-actions), agentdb RuVectorBackend.js, agentic-flow-rust ReasoningBank SQLite storage
- **R49 covers**: sublinear-rust consciousness files (genuine detector, temporal validator, MCP integration, optimization masterplan), neural-pattern-recognition (real-time-monitor, statistical-validator, signal-analyzer), consciousness-explorer validators, agentic-flow ReasoningBank WASM
- **ZERO shared files** between R49 and R50
- Do NOT read or analyze any file from R49's list (see R49-plan.md for that list)

## Rationale

- ruv-swarm Rust crates have NEVER been examined -- R45/R48 covered npm JS files (8:1 genuine:facade), R31 covered JS CLI, but the actual Rust implementation is untouched
- R31 concluded CLI = "demonstration framework" and R45 confirmed via claude-simulator + docker-compose -- but does the Rust layer tell a different story? Historically Rust > JS in quality
- R46 found goalie REVERSAL (MCP facade but internal engines ARE real) -- ed25519-verifier and anti-hallucination-plugin are core goalie internals that may strengthen or weaken this reversal
- `RuVectorBackend.js` (776 LOC) is AgentDB's LARGEST untouched priority file -- R44 found agentic-flow's ruvector-backend.ts (12%) is a COMPLETE FACADE. Does AgentDB's own RuVector backend fare better?
- `reasoningbank-storage/src/sqlite.rs` examines ReasoningBank's Rust persistence -- R49 takes the WASM layer (JS), R50 takes the storage layer (Rust). Together they complete the ReasoningBank source picture
- SWE-Bench adapter is an evaluation framework for coding benchmarks -- is ruv-swarm actually evaluated against real benchmarks, or is it marketing?

## Target: 10 files, ~4,903 LOC

---

### Cluster A: ruv-swarm Rust Crates - First Look (5 files, ~2,117 LOC)

| # | File ID | File | LOC | Package | Domain |
|---|---------|------|-----|---------|--------|
| 1 | 9007 | `ruv-swarm/crates/ruv-swarm-persistence/src/memory.rs` | 435 | ruv-fann-rust | swarm-coordination |
| 2 | 9026 | `ruv-swarm/crates/ruv-swarm-transport/src/protocol.rs` | 379 | ruv-fann-rust | swarm-coordination |
| 3 | 9049 | `ruv-swarm/crates/ruv-swarm-wasm/src/simd_ops.rs` | 420 | ruv-fann-rust | swarm-coordination |
| 4 | 8898 | `ruv-swarm/crates/ruv-swarm-cli/src/commands/spawn.rs` | 413 | ruv-fann-rust | swarm-coordination |
| 5 | 9085 | `ruv-swarm/crates/swe-bench-adapter/src/evaluation.rs` | 470 | ruv-fann-rust | swarm-coordination |

**Key questions**:
- `memory.rs`: Does ruv-swarm-persistence implement real in-memory data structures for agent state (hashmaps, LRU caches, ring buffers), or minimal stubs?
- Does it integrate with the SQLite pool (R45: sqlite-pool.js 92% GENUINE), or is it a separate memory system?
- R48 found THREE disconnected AgentDB distributed layers -- does ruv-swarm persistence add a fourth?
- `protocol.rs`: Does ruv-swarm-transport define real wire protocols (framing, serialization, handshaking, error recovery)?
- R40 found real QUIC in Rust (quinn) but TS = stub -- does this protocol implementation use real networking?
- Does it define message types for swarm coordination (task dispatch, heartbeat, result aggregation)?
- `simd_ops.rs`: R45 found performance-benchmarks.js has real SIMD/WASM (78-95%) -- does the Rust SIMD implementation match?
- Does it implement real SIMD intrinsics (AVX2, SSE, NEON) for vector operations, or wrappers?
- ruvector-core has REAL SIMD (AVX-512/AVX2/NEON) -- does ruv-swarm-wasm use the same approach?
- `spawn.rs`: This is the Rust CLI's spawn command -- does it implement real process spawning (fork/exec, tokio::process, stdio piping)?
- R31 found CLI = demonstration framework in JS -- does the Rust CLI tell a different story?
- Does it spawn real Claude/LLM processes, or simulate agent creation?
- `evaluation.rs`: Does the SWE-Bench adapter implement real benchmark evaluation (test harness, result parsing, scoring)?
- Or is it a mock evaluation that generates predetermined scores?
- Is there real SWE-Bench dataset integration (patch application, test execution, pass/fail detection)?

**Follow-up context**:
- R31: swarm-coordination 25 files (~78%) -- CLI = demonstration framework
- R45: sqlite-pool.js (92%) GENUINE, neural.js (28%) FACADE -- infrastructure vs intelligence split
- R48: diagnostics.js (87%) + errors.js (90%) complete ruv-swarm npm picture (8:1 genuine:facade)
- R44: p2p.rs (92-95%) REAL libp2p -- Rust networking CAN be genuine
- ruvector-core: REAL SIMD (AVX-512/AVX2/NEON) -- sets quality bar for SIMD implementations

---

### Cluster B: Goalie Security & Anti-Hallucination (3 files, ~1,518 LOC)

| # | File ID | File | LOC | Package | Domain |
|---|---------|------|-----|---------|--------|
| 6 | 14215 | `npx/goalie/src/core/ed25519-verifier.ts` | 515 | sublinear-rust | memory-and-learning |
| 7 | 14223 | `npx/goalie/src/plugins/advanced-reasoning/anti-hallucination-plugin.ts` | 516 | sublinear-rust | memory-and-learning |
| 8 | 14209 | `npx/goalie/src/actions/perplexity-actions.ts` | 487 | sublinear-rust | memory-and-learning |

**Key questions**:
- `ed25519-verifier.ts`: Does goalie implement REAL Ed25519 signature verification (key generation, signing, verification using tweetnacl or noble-ed25519)?
- R44 found swarm.rs has production crypto (Ed25519+AES-256-GCM 88%) -- does the goalie TypeScript match the Rust quality?
- Is the verifier used for actual content signing (LLM response verification, plugin authentication), or is it dead crypto code?
- Does it implement proper key management (key derivation, rotation, storage)?
- `anti-hallucination-plugin.ts`: This is a high-value target -- does goalie implement real hallucination detection?
- Real approaches: semantic consistency checking, source attribution, factual grounding, confidence calibration
- Facade approaches: regex keyword matching, hardcoded thresholds, random sampling
- R46 found goalie REVERSAL -- internal engines ARE real behind MCP facade. Is the anti-hallucination plugin part of the real engine?
- Does it integrate with the ReasoningBank (trajectory tracking could enable hallucination detection)?
- `perplexity-actions.ts`: Does this implement real Perplexity API integration (search, citations, follow-up queries)?
- Or is it a mock action that returns simulated search results?
- Does it handle real API authentication, rate limiting, error handling?
- R46 found goalie's plugin registry is genuinely invoked via CLI -- are perplexity actions registered and callable?

**Follow-up context**:
- R46: Goalie REVERSAL -- MCP handlers are facade, but internal engines ARE real (GoapPlanner, PluginRegistry invoked via CLI)
- R46: goalie cli.ts (88-92%) GENUINE
- R41: tools.ts initializes real engines then ignores them
- R44: swarm.rs has production crypto (Ed25519+AES-256-GCM 88%) but 0% GUN transport
- R47: psycho-symbolic-reasoner.ts (38-42%) -- "psycho" = NOTHING, keyword matching reasoning

---

### Cluster C: Backend Reality Check (2 files, ~1,268 LOC)

| # | File ID | File | LOC | Package | Domain |
|---|---------|------|-----|---------|--------|
| 9 | 90 | `dist/src/backends/ruvector/RuVectorBackend.js` | 776 | agentdb | agentdb-integration |
| 10 | 13480 | `reasoningbank/crates/reasoningbank-storage/src/sqlite.rs` | 492 | agentic-flow-rust | memory-and-learning |

**Key questions**:
- `RuVectorBackend.js`: This is AgentDB's OWN RuVector backend -- at 776 LOC it's the largest file in the priority queue. R44 found agentic-flow's ruvector-backend.ts (12%) is a COMPLETE FACADE with zero ruvector imports and hardcoded "125x speedup". Does AgentDB's version fare better?
- Does it import and use real RuVector/HNSW operations, or simulate them?
- Does it implement real vector storage (add, search, delete, update operations with actual HNSW)?
- R20 found AgentDB search broken due to EmbeddingService never initialized -- does this backend have the same initialization problem?
- R43 found ruvector-benchmark.ts (92%) validates HNSWIndex genuinely -- does the backend use the same HNSWIndex?
- Is this `dist/` compiled output from a TypeScript source, or handwritten JS?
- `reasoningbank-storage/src/sqlite.rs`: Does ReasoningBank implement real SQLite storage in Rust?
- Real: rusqlite/sqlx integration, proper schema migrations, WAL mode, prepared statements, transaction management
- Facade: in-memory HashMap with SQLite naming
- R49 takes ReasoningBank WASM (JS), R50 takes storage (Rust) -- together they reveal whether ReasoningBank has genuine persistence
- Does it store trajectories, verdicts, and patterns as the ReasoningBank API promises?
- R48 found quic.ts (95%) has correct CRDTs -- does the storage layer support CRDT state?

**Follow-up context**:
- R44: ruvector-backend.ts (12%) COMPLETE FACADE -- zero ruvector imports, hardcoded "125x speedup", never imported
- R20: AgentDB search broken -- EmbeddingService never initialized in claude-flow bridge
- R43: ruvector-benchmark.ts (92%) = REAL performance testing with genuine HNSWIndex
- R48: THREE disconnected AgentDB distributed layers (QUIC + libp2p + embeddings)
- R43: ReasoningBank core APIs genuine, demo-comparison.ts (35%) scripted theater
- R45: sqlite-pool.js (92%) GENUINE -- sets quality bar for SQLite implementations

---

## Expected Outcomes

- **ruv-swarm Rust reality**: Whether the Rust crates match the JS infrastructure quality (8:1 genuine:facade per R48), or reveal the demonstration framework extends to Rust too. Historically Rust has been higher quality -- this tests that hypothesis on ruv-swarm specifically
- **Goalie security depth**: Whether ed25519-verifier and anti-hallucination-plugin strengthen the R46 reversal (genuine internals behind facade MCP). If the crypto is real and hallucination detection is genuine, goalie becomes a significant genuine subsystem
- **RuVectorBackend verdict**: Whether AgentDB's own RuVector integration succeeds where agentic-flow's failed (12%). This is the single most important file for AgentDB's vector search credibility
- **ReasoningBank persistence**: Whether the Rust SQLite storage is genuine (completing the ReasoningBank picture with R49's WASM findings)
- **SWE-Bench reality**: Whether ruv-swarm has real benchmark evaluation infrastructure or marketing-grade evaluation claims
- **SIMD cross-verification**: Whether ruv-swarm's SIMD matches ruvector-core's quality (REAL AVX-512/AVX2/NEON)

## Stats Target

- ~10 file reads, ~4,903 LOC
- DEEP files: 913 -> ~923
- Expected findings: 45-65

## Cross-Session Notes

- **ZERO overlap with R49**: R49 covers consciousness final sweep (detector, temporal validator, MCP integration, optimization masterplan), neural-pattern-recognition completion (monitor, statistical-validator, signal-analyzer), consciousness-explorer validators, and ReasoningBank WASM. Completely different packages and files.
- **ZERO overlap with R47-R48**: R47 covered consciousness verification/experiments, psycho-symbolic-reasoner, neural-pattern-recognition (logger, CLI, detector). R48 covered agentdb QUIC/CLI/security, ruv-swarm diagnostics/errors. No file overlap.
- **Complements R49**: R49 takes ReasoningBank WASM (JS), R50 takes ReasoningBank storage (Rust) -- together they cover both layers of ReasoningBank's implementation
- **Complements R45/R48**: R45+R48 covered ruv-swarm npm JS files (10 files, 8:1 genuine:facade). R50 covers the Rust crates for the first time
- **Complements R46**: R46 found goalie reversal -- R50 examines goalie's crypto/plugin internals to verify the reversal
- If RuVectorBackend.js is genuine, AgentDB's vector search story strengthens significantly despite R20's broken bridge
- If RuVectorBackend.js is a facade (like ruvector-backend.ts 12%), it confirms the vector search layer is consistently broken across packages
- If ruv-swarm Rust crates are high quality, it suggests the "demonstration framework" characterization applies primarily to the JS/CLI layer, not the Rust core
- Combined DEEP files from R49+R50: 913 -> ~932 (approximately +19)
