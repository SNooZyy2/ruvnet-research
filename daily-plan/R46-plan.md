# R46 Execution Plan: Goalie Deep-Dive + Sublinear Benchmarks + Integration Points

**Date**: 2026-02-15
**Session ID**: 46
**Focus**: Goalie CLI + plugins (expanding R41's facade finding), neural-network benchmarks (post-R43 deception detection), sublinear-time-solver integration and verification layer
**Parallel with**: R45 (no file overlap — verified: R46 = sublinear-rust only, R45 = ruv-fann-rust only)

## Rationale

- R41 found goalie/tools.ts is a "COMPLETE FACADE" (8 CRITICAL findings, every handler returns hardcoded templates) — but the CLI (758 LOC) and two large plugins (673 + 626 LOC) are untouched. Are the plugins also facades, or does the real logic live there?
- R43 found rustc_optimization_benchmarks.rs is the "most deceptive file in project" (15%) — two more benchmark files from neural-network-implementation (601 + 600 LOC) are untouched and may reveal more deceptive patterns
- R41 found consciousness at 79% genuine and cli/index.ts at 88% genuine — the consciousness-verifier.js and integration layer (flow-nexus.js, bin/cli.js) haven't been checked
- memory-and-learning domain is at only 12.19% LOC coverage with 956 NOT_TOUCHED files — this session targets 9 of the largest

## Target: 9 files, ~5,700 LOC

---

### Cluster A: Goalie Deep-Dive (3 files, ~2,057 LOC)

| # | File ID | File | LOC | Domain |
|---|---------|------|-----|--------|
| 1 | 14210 | `npx/goalie/src/cli.ts` | 758 | memory-and-learning |
| 2 | 14222 | `npx/goalie/src/plugins/advanced-reasoning/agentic-research-flow-plugin.ts` | 673 | memory-and-learning |
| 3 | 14231 | `npx/goalie/src/plugins/state-of-art-anti-hallucination.ts` | 626 | memory-and-learning |

**Key questions**:
- `cli.ts`: R41 found tools.ts facades call GoapPlanner but never invoke it — does the CLI bypass tools.ts and call the planner directly?
- Does cli.ts implement real command parsing, file I/O, and solver invocation, or is it another template response layer?
- Is the CLI the "real" goalie interface while MCP tools.ts is the facade layer?
- `agentic-research-flow-plugin.ts`: At 673 LOC this is a substantial plugin — does it implement genuine GOAP-based research planning (goal decomposition, action selection, state tracking)?
- Does it connect to real external services (Perplexity, web search) or return templates like tools.ts's handlePerplexitySearch?
- `state-of-art-anti-hallucination.ts`: Does this implement real hallucination detection (cross-reference checking, confidence calibration, source verification)?
- R41 found tools.ts calls `antiHallucinationVerifier.verify()` but throws away the result — does the plugin itself work when called directly?
- Are the plugins loaded by cli.ts, tools.ts, both, or neither?

**Follow-up context**:
- R41: goalie/tools.ts = COMPLETE FACADE — GoapPlanner imported but NEVER called, every handler returns hardcoded templates
- R41: tools.ts initializes real GoapPlanner, PluginRegistry, AdvancedReasoningEngine, Ed25519Verifier — then ignores them all
- R41: `handleReasoningAnalysis` has real `await reasoningEngine.analyze()` but result thrown away, returns template instead

---

### Cluster B: Neural-Network Benchmarks (3 files, ~1,798 LOC)

| # | File ID | File | LOC | Domain |
|---|---------|------|-----|--------|
| 4 | 13959 | `crates/neural-network-implementation/standalone_benchmark/src/main.rs` | 601 | memory-and-learning |
| 5 | 13880 | `crates/neural-network-implementation/benches/system_comparison.rs` | 600 | memory-and-learning |
| 6 | 13848 | `benchmarks/strange-loops-benchmark.js` | 597 | memory-and-learning |

**Key questions**:
- `standalone_benchmark/main.rs`: R23 found neural-network-implementation is BEST IN ECOSYSTEM (90-98%) — does the standalone benchmark genuinely exercise it?
- Does it use real neural network operations (forward pass, backprop, inference timing) or fabricate metrics like R43's rustc_benchmarks?
- Does it exhibit asymptotic mismatch deception (claims O(f(n)) complexity but implements O(g(n)))?
- `system_comparison.rs`: Does this compare the neural-network crate against real external baselines (PyTorch, ONNX, candle), or against hardcoded target values?
- Are the comparison results reproducible (uses Criterion/bencher framework) or scripted?
- `strange-loops-benchmark.js`: Does this benchmark the real strange-loop crate (R41: consciousness 79% genuine)?
- R44 found strange-loop MCP server (45%) has broken WASM import paths — does the benchmark use the same broken paths?
- Is this a real performance measurement or marketing collateral like R43's demo-comparison.ts (35%)?

**Follow-up context**:
- R23: neural-network-implementation is BEST IN ECOSYSTEM — benchmarks should be correspondingly genuine
- R43: rustc_optimization_benchmarks.rs (15%) = most deceptive file — asymptotic mismatch deception
- R43: baseline_comparison.rs (0%) = NON-COMPILABLE
- R43: ruvector-benchmark.ts (92%) = genuinely real — benchmark quality varies wildly

---

### Cluster C: Integration + Verification Layer (3 files, ~1,845 LOC)

| # | File ID | File | LOC | Domain |
|---|---------|------|-----|--------|
| 7 | 14184 | `integrations/flow-nexus.js` | 620 | memory-and-learning |
| 8 | 13849 | `bin/cli.js` | 618 | memory-and-learning |
| 9 | 14321 | `src/consciousness-explorer/lib/consciousness-verifier.js` | 607 | memory-and-learning |

**Key questions**:
- `integrations/flow-nexus.js`: R44 found server/index.js has "FlowNexus 0% (fake endpoint)" — is this integration file the client that calls those endpoints, or another independent implementation?
- Does it connect sublinear-time-solver to the FlowNexus platform with real API calls, or stub responses?
- Are there real authentication flows (API keys, OAuth), or hardcoded tokens?
- `bin/cli.js`: R41 found cli/index.ts (88%) is GENUINE with real SublinearSolver import — is bin/cli.js the compiled output of that file, or a separate CLI entry point?
- If separate: does it duplicate, wrap, or extend cli/index.ts functionality?
- If compiled output: skip deep analysis (focus on source, not dist)
- `consciousness-verifier.js`: R41 found consciousness at 79% genuine — does the verifier implement real verification (mathematical proofs, consistency checks, output validation)?
- Does it connect to the consciousness subsystem's real components (strange-loop, consciousness-explorer)?
- Is this verification meaningful (catches real errors) or theatrical (always returns "verified")?

**Follow-up context**:
- R44: server/index.js has real HTTP infra (90%) but FlowNexus endpoint is 0% (fake)
- R41: cli/index.ts (88%) is GENUINE — verify if bin/cli.js is compiled output or separate
- R41: consciousness cluster is 79% genuine — verifier could raise or lower this
- R44: All 3 sublinear servers are isolated from each other — does flow-nexus bridge them?

---

## Expected Outcomes

- **Goalie verdict**: Whether the facade pattern extends beyond tools.ts to CLI and plugins, or if real GOAP logic lives in the plugins
- **Plugin quality**: Whether agentic-research-flow and anti-hallucination plugins implement genuine algorithms or more template responses
- **Benchmark trustworthiness**: Whether neural-network benchmarks match the crate's high quality (90-98%) or follow the deceptive pattern from R43
- **Integration reality**: Whether flow-nexus.js provides real cross-system connectivity or is another isolated stub
- **Consciousness confidence**: Whether the verifier raises or lowers R41's 79% genuine assessment

## Stats Target

- ~9 file reads, ~5,700 LOC
- DEEP files: 879 -> ~888
- Expected findings: 40-60

## Cross-Session Notes

- **Zero overlap with R45**: R45 covers ruv-swarm npm package (ruv-fann-rust). No shared files or packages.
- **Zero overlap with R43-R44**: R43 covered WASM tools, AgentDB benchmarks, rustc_benchmarks. R44 covered P2P, agentic-flow bridges, servers. No file overlap.
- If Cluster A finds real GOAP logic in plugins, R41's "COMPLETE FACADE" may need revision to "facade MCP layer over genuine plugins"
- If Cluster B confirms genuine benchmarks, it validates R23's 90-98% assessment of neural-network-implementation
- If Cluster B finds deceptive benchmarks, it contradicts R23 and suggests the crate's quality was overstated
- If Cluster C finds bin/cli.js is compiled output of cli/index.ts, skip deep analysis and substitute another file from priority queue
