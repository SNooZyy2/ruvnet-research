# R40: Neural Model Zoo + AgentDB Intelligence + Worker System

## Context

818 DEEP files achieved through R38. R39 running in parallel covers sublinear-time-solver core (solver.ts, high-performance-solver.ts), emergence subsystem (5 files), and ruQu quantum completion (tile.rs, planner.rs) — all in sublinear-time-solver and ruvector repos.

R40 targets **completely different repos** (ruv-FANN, agentic-flow) to avoid any overlap. Three under-covered domains addressed:
- **swarm-coordination** at 15.1% (HIGH) — ruv-swarm neural models + agentic-flow workers
- **agentic-flow** at 7.5% (MEDIUM) — AgentDB intelligence layer + worker system
- **agentdb-integration** at 45% (MEDIUM) — AgentDB's HNSW, learning, routing internals

## No-Overlap Guarantee

| R39 Repos | R40 Repos |
|-----------|-----------|
| ~/repos/sublinear-time-solver/ | ~/repos/ruv-FANN/ |
| ~/repos/ruvector/ | ~/repos/agentic-flow/ |

R39 file IDs: 14351, 14345, 14361, 14360, 14358, 14359, 14364, 2643, 2680
R40 file IDs: 9644, 9649, 9657, 9646, 12868, 12873, 12926, 12902, 10941, 10939, 10893

Zero intersection.

## Targets: 3 Clusters, 11 Files, ~6,150 LOC

### Cluster A: ruv-swarm Neural Model Architectures (4 files, 1,782 LOC)

**Question:** Does ruv-swarm ship real neural network implementations or facade/stub code? R23 found neural-network-implementation crate (Rust) was BEST IN ECOSYSTEM (90-98%). These are the JS-side equivalents in the npm package. Are they genuine or wrappers?

| # | File ID | Path | LOC |
|---|---------|------|-----|
| 1 | 9644 | ruv-swarm/npm/src/neural-models/base.js | 269 |
| 2 | 9649 | ruv-swarm/npm/src/neural-models/lstm.js | 551 |
| 3 | 9657 | ruv-swarm/npm/src/neural-models/transformer.js | 515 |
| 4 | 9646 | ruv-swarm/npm/src/neural-models/gnn.js | 447 |

**Key questions to answer:**
- Does base.js define a real training API (forward/backward pass) or just config objects?
- Is lstm.js implementing genuine LSTM gates (forget, input, output, cell) with matrix math?
- Does transformer.js have real multi-head attention and positional encoding?
- Is gnn.js doing real message passing / neighborhood aggregation?
- Do any of these use WASM bindings to the Rust neural-network-implementation crate?

**Repo path:** `~/repos/ruv-FANN/ruv-swarm/npm/src/neural-models/`

### Cluster B: AgentDB Intelligence Layer (4 files, 2,494 LOC)

**Question:** How does AgentDB implement its learning and intelligence features? We know AgentDB's RuVectorBackend.ts (DEEP) and the bridge has broken embeddings (R20). What about the higher-level intelligence: HNSW indexing, nightly learning, LLM routing, attention tools?

| # | File ID | Path | LOC |
|---|---------|------|-----|
| 5 | 12868 | packages/agentdb/src/controllers/HNSWIndex.ts | 582 |
| 6 | 12873 | packages/agentdb/src/controllers/NightlyLearner.ts | 665 |
| 7 | 12926 | packages/agentdb/src/services/LLMRouter.ts | 660 |
| 8 | 12902 | packages/agentdb/src/mcp/attention-tools-handlers.ts | 587 |

**Key questions to answer:**
- Does HNSWIndex.ts implement HNSW from scratch or wrap ruvector-core?
- Is NightlyLearner.ts running real batch learning jobs or event logging?
- Does LLMRouter.ts do intelligent model selection or simple round-robin?
- Are attention-tools-handlers.ts exposing genuine attention analysis via MCP?
- How do these connect to the broken EmbeddingService (R20 finding)?

**Repo path:** `~/repos/agentic-flow/packages/agentdb/src/`

### Cluster C: agentic-flow Worker & Transport System (3 files, 1,874 LOC)

**Question:** How does agentic-flow manage its worker agents and transport layer? R31 found swarm-coordination at ~78% with CLI as demonstration framework. These are the actual execution-layer files that workers use.

| # | File ID | Path | LOC |
|---|---------|------|-----|
| 9 | 10941 | agentic-flow/src/workers/worker-registry.ts | 662 |
| 10 | 10939 | agentic-flow/src/workers/worker-agent-integration.ts | 613 |
| 11 | 10893 | agentic-flow/src/transport/quic.ts | 599 |

**Key questions to answer:**
- Does worker-registry.ts manage real worker lifecycle (spawn, health check, teardown)?
- Is worker-agent-integration.ts bridging claude-code agents to internal workers?
- Does quic.ts implement real QUIC protocol (UDP, streams, 0-RTT) or simulate it?
- How do workers communicate — shared memory, message passing, or IPC?

**Repo path:** `~/repos/agentic-flow/agentic-flow/src/`

## Execution Plan

### Step 1: Create Session

```sql
INSERT INTO sessions (name, date, focus)
VALUES ('R40', date('now'), 'Neural model zoo + AgentDB intelligence + worker system');
```
Session ID will be 40.

### Step 2: Spawn 3 Parallel Research Agents

**Agent 1 — Neural Model Zoo** (v3-researcher, files 1-4)
- Read base.js → lstm.js → transformer.js → gnn.js
- Assess each: real neural net implementation vs stub/facade
- Check for WASM/native bindings to Rust crates
- Report: per-file quality %, architecture summary, findings by severity

**Agent 2 — AgentDB Intelligence** (v3-researcher, files 5-8)
- Read HNSWIndex.ts → NightlyLearner.ts → LLMRouter.ts → attention-tools-handlers.ts
- Map the intelligence architecture: how do these pieces compose?
- Check for R20 EmbeddingService dependency
- Report: per-file quality %, real vs facade assessment, findings by severity

**Agent 3 — Worker & Transport** (v3-researcher, files 9-11)
- Read worker-registry.ts → worker-agent-integration.ts → quic.ts
- Assess: real worker management or demonstration code?
- Check QUIC implementation authenticity
- Report: per-file quality %, transport layer assessment, findings by severity

### Step 3: Collect Results & Record to DB

For each file read by agents:
1. Insert `file_reads` record (file_id, session_id=40, depth, lines_read, notes)
2. Update `files` table (depth, lines_read, last_read_date)
3. Insert `findings` (severity, category, description)
4. Tag with domains via `file_domains` (add swarm-coordination, memory-and-learning, agentdb-integration, agent-lifecycle as appropriate)
5. Insert `dependencies` where cross-file relationships discovered

### Step 4: Update Synthesis Documents

- Update `domains/swarm-coordination/analysis.md` with neural models + worker findings
- Update `domains/agentdb-integration/analysis.md` with AgentDB intelligence findings
- Update `domains/agentic-flow/analysis.md` with worker/transport findings

### Step 5: Regenerate Index

```bash
node /home/snoozyy/ruvnet-research/scripts/report.js
```

## Expected Outcomes

| Metric | Target |
|--------|--------|
| Files read | 11 |
| LOC analyzed | ~6,150 |
| DEEP count | 818 → 829 (or R39's count + 11) |
| Findings | 40-60 expected |
| Domains updated | swarm-coordination, agentdb-integration, agentic-flow |

## Key Risks

- Neural model files may be thin wrappers generating config objects (like emergence-tools.ts was facade)
- AgentDB intelligence files may depend on the broken EmbeddingService — could all be non-functional
- QUIC transport may be simulated over TCP/WebSocket (common pattern in Node.js)
- Worker system may be demonstration code matching R31's "CLI = demonstration framework" finding

## Cross-Session Knowledge Targets

- **Confirm or deny:** Does ruv-swarm have usable JS neural networks, or only Rust (R23)?
- **Extend R20:** How far does the broken EmbeddingService ripple through AgentDB's intelligence layer?
- **Extend R31:** Are workers the real execution layer beneath the "demonstration framework" CLI?
