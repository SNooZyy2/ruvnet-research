# ruvnet-research

## The Research Question

> **How much of the ruvnet multi-repo universe is real, working code — and how much is facade?**

The [claude-flow](https://github.com/ruvnet/claude-flow) ecosystem spans 4 repositories, 12 packages, 14,318 files, and 5.5 million lines of code across Rust and TypeScript. It claims self-learning AI agents, HNSW vector search with "150x-12,500x" speedups, Byzantine consensus, 9 reinforcement learning algorithms, and dozens of other advanced capabilities.

This project is a systematic, evidence-based audit of every claim — reading source code line by line, tracing data flows, classifying what's genuine, what's partially real, and what's fabricated.

---

## Methodology

Every file in the ecosystem is inventoried in a SQLite research database. Files are read at classified depths:

| Depth | Meaning |
|-------|---------|
| **DEEP** | 50%+ read, algorithms traced line-by-line, data flow verified |
| **MEDIUM** | 20-50% read, architecture and key functions mapped |
| **SURFACE** | Categorized by name/structure only |
| **NOT_TOUCHED** | Inventoried, zero analysis |

Findings are recorded with severity (CRITICAL / HIGH / MEDIUM / INFO), evidence (line numbers, code snippets), and cross-referenced across domains. Dependencies between files are tracked to identify integration hotspots and isolated subtrees.

Research agents (readers, facade detectors, cross-repo tracers, realness scorers) are deployed in parallel swarms to scale analysis across the codebase.

## Coverage

After 140 research sessions:

- **1,696 files read DEEP** out of 14,318 total (11.8%)
- **12,734 findings** recorded with evidence
- **2,247 cross-file dependencies** mapped
- **17 domains** analyzed across 4 repositories
- **76 exclusion patterns** filtering low-signal subtrees

### By Domain

| Domain | Files | DEEP | Coverage |
|--------|-------|------|----------|
| hook-pipeline | 105 | 103 | 100% |
| process-spawning | 36 | 35 | 100% |
| model-routing | 96 | 92 | 97.5% |
| v4-gold-sweep | 70 | 70 | 100% |
| memory-and-learning | 1,315 | 696 | 73.1% |
| agentdb-integration | 567 | 164 | 60.7% |
| swarm-coordination | 1,108 | 347 | 46.7% |
| ruvector | 6,012 | 528 | 17.4% |
| agentic-flow | 4,212 | 330 | 10.6% |

---

## Key Findings

### What's Genuine

- **ReasoningBank** (92-95%) — Real DeepMind-style RETRIEVE-JUDGE-DISTILL with MaTTS search and MMR
- **EWC++** (92-95%) — Real online Fisher information, adaptive lambda, NEON SIMD
- **HNSW Vector Search** (92-98%) — Real SIMD dispatch (AVX-512/AVX2/NEON), 3 distinct implementations
- **Product Quantization** (88-92%) — Real k-means++, Lloyd's algorithm, ADC with lookup tables
- **Hyperbolic Geometry** (88-95%) — Genuine Poincare ball across 4 crates, 21 DEEP files
- **Raft Consensus** (92%) — Real leader election + libp2p networking
- **Hook Pipeline** (98.1%) — One of the most genuine subsystems in the entire codebase
- **Continuous Batching** (90-95%) — Production-grade vLLM/Orca-style scheduler
- **60+ Agent Types, 175+ MCP Tools, 42+ Skills** — All working

### What's Partially Real

- **SONA** (~75%) — Algorithms work, orchestration incomplete, performance claims inflated
- **Flash Attention** — Algorithm exists in Rust+CUDA+JS, speedup claims unverified
- **GNN Layers** — Real Kipf & Welling GCN math, but two disconnected ecosystems, inference-only
- **Gossip Protocol** (45-55%) — Correct SWIM state machine, transport layer = log statements
- **"150x-12,500x" search speedup** — HNSW is genuinely fast, specific multipliers are marketing

### What's Fabricated

- **9 RL Algorithms** — All reduce to identical tabular Q-value updates. Cosmetic naming only
- **Byzantine Consensus** — coordination.rs is 15-25% FACADE. Vote files written, no voting logic
- **CRDT Synchronization** — Does not exist. "LWW timestamps, no vector clocks, no CRDTs"
- **Agent Booster "352x faster"** — WASM stubs are console.log facades
- **Int8 Quantization "3.92x"** — Returns empty Vec, ignores input
- **84.8% SWE-Bench** — Evaluator generates English prompts, cannot execute
- **LearningBridge** — Zero code exists
- **IPFS Marketplace** — Fake CID generation ("Qm" + hash, not real IPFS)

---

## The Middle Layer Discovery

After 131 sessions, we identified a systematic blind spot: the priority queue optimized for algorithmic component files and deprioritized the integration/wiring layer — the files that answer "how does this actually connect and run?"

Sessions R135-R140 (ML-A through ML-F) specifically targeted this layer:

| Layer | What We Found |
|-------|---------------|
| **CLI Entrypoints** | 31 commands, lazy loading is dead code |
| **Memory Bootstrap** | Zero AgentDB initialization in any MCP server bootstrap |
| **MCP Tool Chain** | 82 tools registered, 14 SONA tools are facades |
| **CI/CD** | Pipelines are facades — `continue-on-error:true` on tests/typecheck |
| **Execution Engine** | Agents = `spawn('claude', ['--print', prompt])`. Zero MCP protocol between orchestrator and workers |
| **Intelligence** | Claims O(log n) HNSW search, actual is O(n) brute force |

---

## Project Structure

```
ruvnet-research/
├── db/research.db              # SQLite source of truth (better-sqlite3)
├── MASTER-INDEX.md             # Auto-generated statistics (never edit)
├── domains/                    # Domain analysis documents
│   ├── ruvector/               # 9-file split structure
│   ├── memory-and-learning/
│   ├── swarm-coordination/
│   ├── agentdb-integration/
│   ├── model-routing/
│   ├── production-infra/
│   └── ...
├── agents/                     # Research agent prompt templates
│   ├── reader.md               # Deep file reading
│   ├── facade-detector.md      # Stub/facade detection
│   ├── cross-repo-tracer.md    # Cross-repo pattern tracing
│   ├── realness-scorer.md      # Weighted realness scoring
│   ├── mapper.md               # Dependency mapping
│   ├── synthesizer.md          # Domain synthesis writing
│   └── scanner.md              # Filesystem inventory
├── the-middle-layer/           # ML-A through ML-F analysis plans
├── route-to-claude-flow-v4/    # V4 planning based on findings
│   └── README-REALITY-CHECK.md # Feature-by-feature verdict
├── ADRs/                       # Architecture decision records
├── scripts/                    # DB migration & reporting scripts
└── ruv-vods/                   # Transcript analysis subproject
```

## Repositories Analyzed

| Repository | Language | Description |
|------------|----------|-------------|
| [claude-flow](https://github.com/ruvnet/claude-flow) | TypeScript | CLI orchestrator, MCP server, agent framework |
| [agentic-flow](https://github.com/ruvnet/agentic-flow) | TypeScript + Rust | AgentDB, MCP tools, RL algorithms |
| [ruvector](https://github.com/ruvnet/ruvector) | Rust | HNSW, SIMD, quantization, GNN, LLM serving |
| [sublinear-time-solver](https://github.com/ruvnet/sublinear-time-solver) | Rust | PageRank, MinCut, graph algorithms |

---

## Bottom Line

Of 5.5 million lines of code, the genuine core is substantial but narrower than advertised. The strongest components — ReasoningBank, HNSW, EWC++, continuous batching, hooks — are production-quality. But the integration layer is systematically broken: components that individually work are never wired together at runtime. The "middle layer" connecting algorithms to user-facing features is where most facades live.

A realistic v4 can be built from the genuine components. It just can't claim everything the current README does.
