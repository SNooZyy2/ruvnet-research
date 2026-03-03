# Swarm Coordination Domain Analysis

> **Priority**: HIGH | **Coverage**: ~21.6% (313/1428 DEEP) | **Status**: In Progress
> **Last updated**: 2026-03-02 (Session R141)

## Document Index

This analysis has been split into sub-files for manageability. **Agents: follow links below to the relevant section.**

| Section | File | Description |
|---------|------|-------------|
| 1. Current State | **this file** | Top-level verdicts, stats, key conclusions |
| 2. File Registry | [file-registry.md](file-registry.md) | All deep-read file tables grouped by subsystem |
| 3. Findings Registry | [findings.md](findings.md) | CRITICAL (C1-C62), HIGH (H1-H187) findings |
| 4. Positives Registry | [positives.md](positives.md) | Confirmed good patterns and genuine code |
| 5a. Subsystems (Core) | [subsystems-core.md](subsystems-core.md) | Architecture, P2P, federation, coordination, templates, SKILL files, Rust crates, npm JS, neural, Python ML, workflow, benchmarking |
| 5b. Subsystems (Extended) | [subsystems-extended.md](subsystems-extended.md) | (Reserved for future subsystem sections) |
| 6. Cross-Domain | [cross-domain.md](cross-domain.md) | Dependencies on other domains |
| 7. Knowledge Gaps | [gaps.md](gaps.md) | Remaining coverage gaps |
| 8. Session Log | [session-log.md](session-log.md) | All session entries (R9 through R141) |

## Agent Instructions

When updating this domain analysis:
- **Section 1** (below): Rewrite in-place in THIS file
- **Section 2**: Edit [file-registry.md](file-registry.md) — add rows, never duplicate
- **Section 3**: Edit [findings.md](findings.md) — add with next sequential ID, never re-number
- **Section 4**: Edit [positives.md](positives.md) — append new, never re-list
- **Section 5**: Edit the appropriate subsystems file — update existing topics or create new ones
- **Section 8**: Append to [session-log.md](session-log.md)
- Follow ADR-040 canonical structure and ADR-041 in-place update protocol
- NEVER create chronological session blocks outside session-log.md
- NEVER re-list all findings/positives at each update

## 1. Current State Summary

The swarm-coordination domain spans 267+ DEEP files across multi-agent lifecycle, topology, consensus, health monitoring, and inter-agent communication. R83-R84 added the ruv-swarm WASM cascade layer and persistence analysis. **R85 adds 4 ruv-swarm npm runtime files (530 LOC)** exposing a split personality in the npm package: genuine build orchestration alongside fabricated runtime layers.

**R91 key results:**

- **neural-presets-complete.js (52-58% BIMODAL, 1,306 LOC)** — ruv-swarm neural model presets "complete" version. Lines 1-774 (35-40%): `COMPLETE_NEURAL_PRESETS` object — 27 architecture families as nested config objects. Hyperparameter values are textbook-accurate (BERT 768/12/12/3072, EfficientNet-B0 scaling, DDPM timesteps:1000/cosine) but are LOOKUP TABLES ONLY — no forward pass, no weights, no tensors. Performance metadata (`expectedAccuracy`, `inferenceTime`, `memoryUsage`) are hardcoded paper benchmarks, not measured. `calculatePresetScore()` uses `parseInt`/`parseFloat` which returns NaN for range strings like "5ms/step". Lines 780-1305 (65-70%): `CognitivePatternSelector` + `NeuralAdaptationEngine` contain real algorithmic logic — branching for creativity/precision/adaptation contexts, diversity enforcement, top-5 scored recommendations. `NeuralAdaptationEngine` records adaptation histories, computes accuracy deltas, and suggests hyperparameters from successful past runs.
- **9th disconnected persistence** — `crossSessionMemory` in `CognitivePatternSelector` is an in-memory Map that resets on every instantiation. Despite the name implying cross-session storage, there is zero DB write, zero file I/O, zero serialization.
- **Dead imports confirmed** — `CognitivePatternEvolution` and `MetaLearningFramework` imported and instantiated but never called anywhere in the file.
- **Integration is genuine**: neural-network-manager.js has 4 active call sites for this file (not orphaned).
- **Confirms R69**: This file is the SOURCE of the "27 ghost models" count — 27 architectures defined as lookup-table configs, few actually implemented in the WASM backend.
- **WASM scoreboard**: 15 genuine + 1 GHOST vs 12 theatrical (56% genuine). **Persistence layer fragmentation**: 9 disconnected layers confirmed.

**R87 key results:**

- **test-pr34-local.js (0% COMPLETE FACADE)** — 118 LOC. PR#34 test for "Comprehensive Onboarding Integration" feature. Imports non-existent src/onboarding/index.js module. Declares 4 phantom classes (DefaultClaudeDetector, DefaultMCPConfigurator, MCPServerConfig, MCPConfig). File does not execute (import error). Test suite structured as 5 incremental checks but all assertions are smoke tests: class instantiation attempts (undefined), private method calls (_generateRuvSwarmConfig), unconditional pass if no errors thrown. Summary claims "All classes instantiate correctly" but instantiation is against undefined. **9th complete test facade** after benchmark.js deception. PR#34 intended onboarding framework (Claude detector + MCP config generation) but implementation was deferred/abandoned.

**R85 key results:**

- **build.js (90-95% GENUINE)** — 167 LOC WASM build orchestrator. Dual compilation (standard + SIMD-optimized), wasm-opt optimization pass, TypeScript definition generation. Real wasm-pack + rustc dependency validation at startup. Confirms pre-built WASM artifacts in npm/wasm/ are genuinely compiled from Rust. FIRST npm build script confirmed as genuine infrastructure.
- **claude-hooks.js (71% BIMODAL)** — 162 LOC GitHub coordination hook bridge. 5 real GitHub hook types (pre-task, post-edit, post-task, check-conflicts, get-dashboard) with genuine GitHub issue integration. BUT: no connection to claude-flow hooks system, no MCP/ADR-008 routing, in-memory state lost on restart (activeTask), autoClose disabled, placeholder conflict detection (count-based proxy only). Operates independently of all 6 parallel routing systems.
- **ruv-swarm-memory.js (0% PURE DEMO)** — 119 LOC CLI with fabricated metrics. Hardcoded claims (40% reduction, 2.8x faster, 84% less fragmentation). No SQLite, no AgentDB, no MCP. Confirms **8th disconnected persistence layer** (first confirmed R84, now formally classified R85).
- **hooks/cli.js (75-80%)** — 82 LOC thin CLI wrapper delegating ALL business logic to index.js (1,899 LOC). 15+ hook types via JSON stdout protocol. Custom arg parser. Exit codes 0/1/2 signal result type to parent caller. Does not implement hook logic itself — pure routing layer.

**WASM scoreboard**: 16 genuine vs 13 theatrical (55% genuine — R87 adds temporal-neural-solver-wasm 88-92% genuine, psycho-symbolic test-build.js 13th theatrical). **Persistence layer fragmentation**: 9 disconnected layers (R87: memory-config.js adds 9th). ruv-swarm npm build pipeline is REAL; npm runtime layer remains BIMODAL (genuine infra, fabricated intelligence). **test-wasm-loading.js (95-98%) VALIDATES R84 build.rs** — core WASM binary loads and executes real functions. **verify-db-updates.js (88-92%) GENUINE** — real DB queries confirm persistence layer works despite theatrical CLI metrics.

**R141 key results (Rust compilation audit):**

- **ENTIRE ruv-swarm Rust workspace fails `cargo check`** — All 14 Cargo.toml manifests declare `ruv-fann = "^0.1.5"` but the workspace root provides ruv-fann 0.2.0. The `^0.1.5` semver range does NOT include 0.2.0 (breaking change boundary). This workspace-wide version mismatch causes `cargo check` to fail for every crate: ruv-swarm-core, ruv-swarm-agents, ruv-swarm-cli, ruv-swarm-daa, ruv-swarm-mcp, ruv-swarm-ml, ruv-swarm-persistence, ruv-swarm-transport, ruv-swarm-wasm, ruv-swarm-wasm-unified, claude-parser, swe-bench-adapter, benchmarking, and ml-training.
- **Binary truth signal** — This single ruv-fann version pin proves that NO crate in the ruv-swarm Rust layer has been integration-tested against the current workspace. Prior sessions (R50, R70-R72, R79-R86) assessed individual files as 85-95% genuine based on source code quality; R141 confirms that quality code cannot reach production because the workspace won't compile.
- **Impact on earlier findings** — Source-level quality assessments (memory.rs 95-98%, protocol.rs 92-95%, models.rs 92-95%, service.rs 88-92%) remain valid for the code that was written, but all runtime claims about those crates are invalidated. The persistence crate (93% weighted average) and MCP service (rmcp bindings) are non-functional in the current workspace state. Finding C65 added.
- **NOT affected**: ruv-fann itself (a separate workspace), ruvector-core, and the JS/TS swarm layer (npm package). Only the ruv-swarm Rust workspace is blocked.

**Top verdicts:**
- **Best infrastructure**: sqlite-pool.js (92%), storage.rs (95-98%), in_process.rs (92%), service.rs (88-92%), config.rs (88-92%), models.rs (92-95%), build.js (90-95%).
- **Worst gaps**: neural.js (28%), wasm_simple.rs (22-28%), neural-coordination-protocol.js (10-15%), QUIC empty everywhere, GPU operations zero.
- **Compilation blocker (R141)**: ENTIRE ruv-swarm Rust workspace fails cargo check — ruv-fann ^0.1.5 vs workspace 0.2.0 version mismatch across all 14 Cargo.toml files (C65).

**R82 key results (4 files, ~462 LOC):**

- **neural.rs (wasm) 85-90% GENUINE** — 155 LOC REVERSES R80 activation.rs (35-40% broken). Correct ruv-fann API: NetworkBuilder, run(), get_weights/set_weights, 17-variant activation parser. All verified against ruv-fann 0.1.5 source. Inference-only (no train()). Metrics struct is facade (never populated). WASM scorecard: +1 genuine → 14 genuine + 1 GHOST vs 11 theatrical (56% genuine).
- **types.rs (daa) 78-82% genuine type architecture** — 132 LOC RESOLVES R69 ghost model mismatch. AgentType enum defines 5 agent ROLES (Researcher, Coder, Analyst, Coordinator, Specialist), NOT 27 neural models. Two separate type hierarchies that never connected. DecisionContext production-quality (5 fields), AutonomousCapability 11 well-defined variants. NeuralNetworkManager is STUB (only initialized: bool).
- **neural.rs (daa) 0-5% PURE METADATA FACADE** — 94 LOC. NeuralManager stores HashMap<String, NeuralNetworkInfo> but ZERO neural computation (no forward/backward/train/run). Zero ruv-fann imports. Compare: ruv-swarm-wasm neural.rs wraps real Network<f32>. Confirms R69 GHOST WASM pattern.
- **patterns.rs (daa) 15-20% skeletal** — 81 LOC. PatternManager pure data structure. Defines 6 cognitive thinking styles (Convergent, Divergent, Lateral, Systems, Critical, Adaptive) but only 2 have metadata (67% undefined). Zero behavior: no selection, effectiveness tracking, or evolution logic.

**R79 key results (carried forward):**

- **ruv-swarm-wasm BIMODAL within single crate** — simd_tests.rs (88-92% GENUINE wasm_bindgen_test), training.rs (~80% GENUINE, 4/5 real ruv-fann algorithms, SARPROP silently falls back to RPROP), memory_pool.rs (78-82% GENUINE 3-tier pool). BUT agent.rs (28-32% FACADE, mock setTimeout execute), swarm.rs (0% ORPHANED, imports non-existent enums, cannot compile).
- **ruv-swarm-mcp/src/ COMPLETE** (9/9 DEEP) — error.rs (88-92%) completes source layer with protocol-agnostic error bridge. Triple representation (message/JSON-RPC code/HTTP status). rmcp-decoupled by design.
- **ruv-swarm-transport/src/ COMPLETE** (5/5 DEEP) — lib.rs (90%+) clean barrel with polymorphic async Transport trait (8 methods), DashMap registry for runtime backend registration. No feature gates.
- **neural_bridge.rs 87% GENUINE** — Real ruv-fann bridge (Adam optimizer, sliding window time series), but predict() returns zeros and load_parameters() is no-op.
- **JS benchmarks BOTH DECEPTIVE** — benchmark.js (0-5%) is 100% setTimeout fabrication, DEEPEST in research corpus. mcp-tools-benchmarks.js is 8th MISLABELED file — generic JS micro-benchmarks, not MCP tool testing. Extends R59 benchmark deception to JS layer.
- **WASM scoreboard update**: +3 genuine (simd_tests, training, memory_pool), +2 theatrical (agent, swarm). Running total: 13 genuine + 1 GHOST vs 11 theatrical (54% genuine).
