# Verification Plan: live-february-26-leads

Generated: 2026-03-01
Source: /home/snoozyy/ruvnet-research/ruv-vods/leads/live-february-26-leads.md (22 leads)

## Summary
- ALREADY_COVERED: 4 leads (skip)
- PARTIALLY_COVERED: 7 leads (targeted re-reads)
- NEW: 8 leads (full reads needed)
- CONTRADICTION: 1 lead (highest priority)
- UNRESOLVABLE: 2 leads (cannot verify)

## Total files to read: ~42
## Estimated session: 4-5 research sessions (R115-R119), prioritizing CONTRADICTION and HIGH-priority NEW leads first

---

## CONTRADICTION LEADS (verify first)

### LEAD-017: LOC count and AI mischaracterization of ruvector capabilities
Classification: CONTRADICTION
Original claim: "Ruv states ruvector has 'millions of lines of code' in the repo. Claude/AI tools frequently mischaracterize ruvector's capabilities -- for example, claiming it has no PostgreSQL persistence, no graph-aware RAG."

Contradicts: Research DB shows ruvector-rust has 2,473,230 LOC across 6,035 non-excluded files. The total project (all packages) has 5,280,944 LOC across 14,010 files. The "millions of lines" claim is CONFIRMED for the full project. However, the specific claim about PostgreSQL and graph-aware RAG needs nuance -- research HAS found postgres persistence (R101 COMPLETE) and Cypher executors (R38), but also found many facades and incomplete integrations. The claim that ruvector "has no PostgreSQL persistence" is contradicted by existing DEEP findings showing real postgres code (healing, HNSW AM, SPARQL) alongside facades (detector returns zeros, worker uses sleep not WaitLatch).

#### Files to read:
| File | Package | Current Depth | LOC | Action |
|------|---------|--------------|-----|--------|
| No new files needed | - | - | - | Cross-reference existing findings |

#### Existing findings that relate:
- Finding: "PRODUCTION-QUALITY SPARQL 1.1 PARSER" (ruvector-postgres/graph/sparql/parser.rs, CRITICAL/ARCHITECTURE)
- Finding: "ALL 8 metric collection methods return empty/zero" (ruvector-postgres/healing/detector.rs, HIGH/FACADE)
- Finding: "Cypher executor: 70-75% REAL - WORKING EXECUTOR" (rvlite/cypher/executor.rs, CRITICAL/QUALITY)
- Finding: "DUAL QUERY LANGUAGE SYSTEM: Cypher + SPARQL" (ruvector-postgres, INFO/ARCHITECTURE)
- Finding: "COMPLETE MISLABELING: File in ruvector-postgres/benches/ contains ZERO postgres integration" (CRITICAL/FACADE)
- Finding: "ruvector-rust LOC: 2,473,230" (DB query)

#### What to verify:
- The "millions of LOC" claim: CONFIRMED (2.47M for ruvector-rust alone, 5.28M total project)
- Postgres persistence: PARTIALLY TRUE -- real code exists alongside facades
- Graph-aware RAG: No dedicated "graph RAG" files found (search returned 0). Cypher executors exist but are query engines, not RAG pipelines

#### Suggested research agent: N/A -- existing coverage sufficient for this claim

---

## NEW LEADS (full reads needed)

### LEAD-005: Multiple consensus strategies (Raft, DAG, event sourcing, CRDTs, vector clocks, sharding)
Classification: NEW
Original claim: "Ruvector uses multiple consensus strategies: Raft consensus, DAG consensus, event sourcing with delta stream/checkpoints/relay/compaction, causal delivery with vector clocks and sharding (64 shards via consistent hash ring), and CRDTs for conflict-free merging."
Priority: HIGH

#### Files to read:
| File | Package | Current Depth | LOC | Action |
|------|---------|--------------|-----|--------|
| crates/ruvector-raft/src/node.rs | ruvector-rust | NOT_TOUCHED | 632 | Full DEEP read - Raft state machine |
| crates/ruvector-raft/src/election.rs | ruvector-rust | NOT_TOUCHED | 360 | Full DEEP read - leader election |
| crates/ruvector-raft/src/log.rs | ruvector-rust | NOT_TOUCHED | 351 | Full DEEP read - log replication |
| crates/ruvector-raft/src/rpc.rs | ruvector-rust | NOT_TOUCHED | 443 | Full DEEP read - RPC layer |
| crates/ruvector-delta-consensus/src/lib.rs | ruvector-rust | NOT_TOUCHED | 489 | Full DEEP read - delta consensus core |
| crates/ruvector-delta-consensus/src/crdt.rs | ruvector-rust | NOT_TOUCHED | 486 | Full DEEP read - CRDT implementation |
| crates/ruvector-delta-consensus/src/causal.rs | ruvector-rust | NOT_TOUCHED | 320 | Full DEEP read - causal delivery |
| npm/packages/replication/src/vector-clock.ts | ruvector-rust | NOT_TOUCHED | 149 | Full DEEP read - vector clocks |
| tests/integration/distributed/sharding_tests.rs | ruvector-rust | NOT_TOUCHED | 398 | Full DEEP read - sharding tests |

#### Existing findings that relate:
- Finding: "CRDT algorithms ACCURATE: G-Counter, PN-Counter, OR-Set, LWW-Register" (agents/consensus/crdt-synchronizer.md, INFO/QUALITY) -- but this is an AGENT TEMPLATE not runtime code
- Finding: "REAL distributed protocol types: VectorClock, CRDTs" (src/types/quic.ts, CRITICAL/ARCHITECTURE) -- TYPE DEFINITIONS only, not runtime
- Finding: "PRODUCTION-GRADE CRDT operations" (src/types/quic.ts, CRITICAL/QUALITY) -- helper functions, tested correct
- Finding: "Raft election logic correct, follows paper exactly, 95% correct" (examples/edge/src/p2p/advanced.rs, INFO/QUALITY) -- but in examples/ dir

#### What to verify:
- Are the ruvector-raft and ruvector-delta-consensus crates genuine implementations or facades?
- Does Raft integrate with the delta-consensus CRDT layer?
- Do vector clocks exist in runtime code (not just type definitions)?
- Is the 64-shard consistent hash ring implemented?

#### Suggested research agent: reader (5 files), mapper (1 -- trace cross-crate deps)

---

### LEAD-002: Sync API with vector deltas and sparse monitoring
Classification: NEW
Original claim: "Ruv describes a 'sync API' endpoint that synchronizes state between nodes purely via vector deltas -- not text. The system uses delta-based push/pull with sparse monitoring that wakes up only when integrity bounds are exceeded."
Priority: HIGH

#### Files to read:
| File | Package | Current Depth | LOC | Action |
|------|---------|--------------|-----|--------|
| crates/ruvector-delta-consensus/src/lib.rs | ruvector-rust | NOT_TOUCHED | 489 | Full DEEP read (shared with LEAD-005) |
| crates/ruvector-delta-consensus/src/conflict.rs | ruvector-rust | NOT_TOUCHED | 286 | Full DEEP read - conflict resolution |
| examples/edge-net/pkg/sync.js | ruvector-rust | NOT_TOUCHED | 800 | Full DEEP read - sync package |

#### Existing findings that relate:
- Finding: "SyncCoordinator backend FAKE: QUICClient.sendRequest() returns hardcoded {success:true}" (src/controllers/SyncCoordinator.ts, CRITICAL/FACADE) -- AgentDB sync is FAKE
- Finding: "No distributed consensus: uses simple LWW timestamps" (src/coordination/MultiDatabaseCoordinator.ts, HIGH/ARCHITECTURE)

#### What to verify:
- Does ruvector-delta-consensus implement actual delta-based vector synchronization?
- Is there a sparse monitoring system with integrity-bound triggers?
- Distinguish between the known-fake AgentDB/JS sync layer and the Rust delta-consensus crate

#### Suggested research agent: cross-repo-tracer (delta sync across repos), reader (3 files)

---

### LEAD-003: RVF binary format with copy-on-write, layers, 7-bit quantization
Classification: PARTIALLY_COVERED (some RVF files are DEEP, but specific COW/quantization/layer claims need verification)
Original claim: "The RVF format is a self-contained binary using copy-on-write structure that includes a Linux runtime kernel, ViteJS dashboard, 3JS visualization, a small language model, and a self-contained vector store. Uses layered binary structure. Achieves 80-90% data reduction via 7-bit integer quantization and delta-only storage with cryptographic hashing."
Priority: HIGH

#### Files to read:
| File | Package | Current Depth | LOC | Action |
|------|---------|--------------|-----|--------|
| crates/rvf/rvf-runtime/src/cow.rs | ruvector-rust | DEEP | 503 | Re-read targeted sections on COW mechanism |
| crates/rvf/rvf-types/src/agi_container.rs | ruvector-rust | NOT_TOUCHED | 963 | Full DEEP read - AGI container format |
| crates/rvf/rvf-ebpf/src/lib.rs | ruvector-rust | NOT_TOUCHED | 1083 | Full DEEP read - eBPF subsystem |
| examples/rvf/examples/causal_atlas_dashboard.rs | ruvector-rust | NOT_TOUCHED | 585 | Full DEEP read - dashboard example |

#### Existing findings that relate:
- Finding: "derive() implements Copy-On-Write branching at .rvf file level" (RvfBackend.ts, INFO/GENUINE)
- Finding: "GENUINE NAPI bridge: 22 methods on RvfDatabase class" (rvf-node/lib.rs, INFO/GENUINE)
- Finding: "VectorData stores all vectors in-memory. boot() loads every VEC_SEG payload" (rvf-runtime/store.rs, HIGH/ARCHITECTURE)
- Finding: "query() is brute-force linear scan O(n) over ALL vectors" (rvf-runtime/store.rs, CRITICAL/PERFORMANCE)
- Finding: "rvf-kernel is a kernel IMAGE BUILDER not a microVM/hypervisor" (rvf-kernel/lib.rs, CRITICAL/ARCHITECTURE)
- Finding: "ZERO callers outside its own crate" for rvf-kernel (CRITICAL/INTEGRATION)

#### What to verify:
- Does the 7-bit quantization exist in the Rust RVF runtime? (store.rs is DEEP but no quantization findings)
- Is the layered binary structure (kernel, GNN, self-learning loop, POSIX storage) actually implemented?
- Does the eBPF subsystem actually execute embedded programs?
- Does the dashboard example demonstrate the claimed capabilities?

#### Suggested research agent: reader (3 files), facade-detector (1 -- rvf-ebpf)

---

### LEAD-010: New graph transformer crates (sublinear attention, proof-gated attention)
Classification: NEW
Original claim: "New graph transformer crates were created including sublinear attention, proof-gated attention (validates each element in graph state before making reasoning decisions), and world model physics simulation using simulated quantum state with attention."
Priority: HIGH

#### Files to read:
| File | Package | Current Depth | LOC | Action |
|------|---------|--------------|-----|--------|
| examples/delta-behavior/applications/04-self-stabilizing-world-model.rs | ruvector-rust | NOT_TOUCHED | 554 | Full DEEP read |

DB search returned ZERO files for `graph_transformer`, `proof_gated`, `proof-gated`, `sublinear_attention`. These crates may not exist in the analyzed repos, or may be too new for the DB scan.

#### Existing findings that relate:
- Finding: "mincut-gated-transformer MOST NOVEL" (R34, R38) -- known existing transformer
- Finding: "estimate_partition_boundaries() is a simplified uniform-stride placeholder" (sparse_attention.rs, HIGH/ARCHITECTURE)
- Finding: "Novel design thesis: mincut partition boundaries as semantic transitions" (sparse_attention.rs, INFO/ARCHITECTURE)

#### What to verify:
- Do "sublinear attention" and "proof-gated attention" crates exist in the repos?
- If they exist, are they in the DB (may need scanner pass)?
- If they don't exist, this is aspirational/planned, not implemented
- The world model file exists at 554 LOC -- check if it implements physics simulation

#### Suggested research agent: cross-repo-tracer (search for new crates), reader (1 file)

---

### LEAD-004: QR code seed format (2000 bytes)
Classification: NEW
Original claim: "The RVF format can be compressed to a 2000-byte seed that fits in a QR code. The seed expands by downloading additional data."
Priority: MEDIUM

#### Files to read:
| File | Package | Current Depth | LOC | Action |
|------|---------|--------------|-----|--------|
| crates/rvf/rvf-runtime/src/qr_encode.rs | ruvector-rust | NOT_TOUCHED | 1135 | Full DEEP read |
| crates/rvf/rvf-runtime/src/qr_seed.rs | ruvector-rust | NOT_TOUCHED | 1095 | Full DEEP read |
| crates/rvf/rvf-runtime/src/seed_crypto.rs | ruvector-rust | NOT_TOUCHED | 236 | Full DEEP read |
| crates/rvf/rvf-types/src/qr_seed.rs | ruvector-rust | NOT_TOUCHED | 379 | Full DEEP read |

#### Existing findings that relate:
- Finding: "simple_shake256_256 is NOT SHAKE-256, is a homebrew XOR-fold hash" (rvf-runtime/store.rs, CRITICAL/SECURITY) -- crypto concern affects seed integrity
- Finding: "seed_crypto::sign_seed uses REAL HMAC-SHA256" (rvf-runtime/witness.rs, CRITICAL/SECURITY) -- seed_crypto module has real crypto

#### What to verify:
- Does qr_encode.rs actually produce a QR-encodable 2000-byte seed?
- Does qr_seed.rs implement bootstrap/expansion from seed?
- Is the seed functional or aspirational?

#### Suggested research agent: reader (4 files), facade-detector (1 -- check if expansion works)

---

### LEAD-007: Self-learning solver with sublinear attention and SIMD
Classification: PARTIALLY_COVERED
Original claim: "The self-learning solver creates self-reinforcement learning through auto-generated puzzles. Uses 'sublinear attention' in tightly constrained microsecond loops. Creates parallel tracks tested via SIMD."
Priority: HIGH

#### Files to read:
| File | Package | Current Depth | LOC | Action |
|------|---------|--------------|-----|--------|
| crates/ruvector-solver/src/router.rs | ruvector-rust | NOT_TOUCHED | 1615 | Full DEEP read - solver routing |
| crates/ruvector-solver/src/true_solver.rs | ruvector-rust | NOT_TOUCHED | 933 | Full DEEP read - true solver impl |
| crates/ruvector-solver/src/simd.rs | ruvector-rust | NOT_TOUCHED | 282 | Full DEEP read - SIMD operations |
| crates/rvf/rvf-solver-wasm/src/engine.rs | ruvector-rust | NOT_TOUCHED | 784 | Full DEEP read - WASM solver |

#### Existing findings that relate:
- Finding: "Fake sublinear timing: sublinearTimeMs = sqrt(matrixSize) * 0.001" (strange-loop/lib/sublinear-integration.js, CRITICAL/FACADE) -- JS facade
- Finding: "FALSE sublinearity confirmed (all O(n^2)+)" (R39 summary)
- Finding: "backward_push.rs O(1/epsilon) sublinear" (R51-R60) -- 1 genuine sublinear algorithm found
- Finding: "GENUINE 3-tier solver" (npx/goalie/src/mcp/server.ts, INFO/GENUINE)

#### What to verify:
- ruvector-solver is a DIFFERENT crate from the sublinear-time-solver analyzed in R39
- Does this crate implement genuine self-reinforcement via puzzle generation?
- Is the SIMD in simd.rs real (like ruvector-core AVX) or decorative?
- Does true_solver.rs actually achieve sublinear complexity?

#### Suggested research agent: reader (4 files), facade-detector (1)

---

### LEAD-009: AgentDB RVF migration path
Classification: PARTIALLY_COVERED
Original claim: "AgentDB has been updated to use the RVF format. Claude flow is moving from SQLite to RVF format. The migration path was: JSON -> SQLite -> sql.js (WASM) -> binary DB files -> RVF multi-layered binary."
Priority: HIGH

#### Files to read:
| File | Package | Current Depth | LOC | Action |
|------|---------|--------------|-----|--------|
| packages/agentdb/docs/adrs/ADR-003-rvf-native-format-integration.md | agentic-flow | NOT_TOUCHED | 1400 | Full read - RVF integration ADR |
| crates/rvf/rvf-adapters/agentdb/src/pattern_store.rs | ruvector-rust | NOT_TOUCHED | 457 | Full DEEP read - Rust-side adapter |
| crates/rvf/rvf-adapters/agentdb/src/vector_store.rs | ruvector-rust | NOT_TOUCHED | 327 | Full DEEP read - Rust-side vector store |

#### Existing findings that relate:
- Finding: "SqlJsRvfBackend .rvf files are raw SQLite databases, NOT file-format compatible with native RvfBackend" (SqlJsRvfBackend.ts, HIGH/ARCHITECTURE)
- Finding: "derive() implements Copy-On-Write branching at .rvf file level" (RvfBackend.ts, INFO/GENUINE)
- Finding: "factory.ts auto-detect chain: ruvector > native-rvf > hnswlib > sqljsRvf" (SqlJsRvfBackend.ts, INFO/ARCHITECTURE)
- Finding: 15 AgentDB RVF backend files at DEEP (RvfBackend.ts, AdaptiveIndexTuner.ts, ContrastiveTrainer.ts, etc.) -- many CRITICAL bugs found
- Finding: "@ruvector/router is NOT installed, SemanticQueryRouter always runs brute-force fallback" (CRITICAL/ARCHITECTURE)
- Finding: "NativeAccelerator TensorCompress static check always fails" (CRITICAL/BUG)

#### What to verify:
- Does the Rust-side rvf-adapters/agentdb actually bridge to the TS AgentDB?
- Is there a working migration path from SQLite to native RVF?
- The TS RVF backend has many CRITICAL bugs -- does the Rust adapter avoid them?

#### Suggested research agent: reader (3 files)

---

### LEAD-019: QUIC protocol for delta sync
Classification: PARTIALLY_COVERED
Original claim: "Ruv uses QUIC protocol for networking (eventual consistency) and Raft for consensus. When resources are constrained, exports only mathematical deltas."
Priority: HIGH

#### Files to read:
| File | Package | Current Depth | LOC | Action |
|------|---------|--------------|-----|--------|
| No new files needed -- QUIC extensively covered | - | - | - | - |

#### Existing findings that relate:
- Finding: "QUIC transport loads WASM returning {}" (quic-coordinator.js, CRITICAL/FACADE)
- Finding: "QUICClient is ENTIRELY STUB" (QUICClient.ts, CRITICAL/FACADE)
- Finding: "REAL wasm-bindgen WASM binding layer for QUIC" (wasm/quic/agentic_flow_quic.js, HIGH/ARCHITECTURE)
- Finding: "Real QUIC in Rust (quinn), TS = stub" (R40 key finding)
- Finding: "Complete libp2p integration: Gossipsub, Kademlia DHT" (edge-net/network/p2p.rs, CRITICAL/ARCHITECTURE)
- Finding: "REAL distributed protocol types: VectorClock, CRDTs" (quic.ts, CRITICAL/ARCHITECTURE)

#### What to verify:
- The delta export mechanism specifically -- does it exist beyond the known QUIC/p2p findings?
- The Rust-side QUIC via quinn is confirmed real (R40). TS side is confirmed fake.
- The delta-consensus crate (LEAD-005) is where the "mathematical deltas" mechanism likely lives -- read there

#### Suggested research agent: N/A -- covered by LEAD-005 reads

---

## PARTIALLY_COVERED LEADS (targeted re-reads)

### LEAD-008: Min cut used for system bounds/RVF boundaries
Classification: PARTIALLY_COVERED
Original claim: "Min cut is used to define the boundaries/bounds of the system. Referenced in context of the memory system and the RVF format bounds."
Priority: MEDIUM

#### Files to read:
| File | Package | Current Depth | LOC | Action |
|------|---------|--------------|-----|--------|
| No new files needed | - | - | - | - |

#### Existing findings that relate:
- Finding: "MinCut 27+ DEEP files" (R93-R100 comprehensive coverage)
- Finding: "Novel design thesis: mincut partition boundaries as semantic transitions" (sparse_attention.rs)
- Finding: "Global min-cut is approximate: upper bounds, not true min-cut" (hierarchy.rs, HIGH/QUALITY)
- Finding: "MinCut linear attention PLACEHOLDER" (R100)
- Finding: "MinCut BIMODAL: graph 90-95% vs fragmentation 70-78%" (R112)

#### What to verify:
- Research has 27+ DEEP MinCut files but findings focus on the graph transformer, NOT on RVF boundary detection
- No existing finding connects MinCut to RVF or memory system bounds
- This is a NOVEL claim about MinCut's application that existing coverage does not address
- Check if ruvector-mincut is imported by any RVF crate (mapper task)

#### Suggested research agent: mapper (trace mincut -> rvf dependency)

---

### LEAD-011: Algebraic graph traversal via sparse matrices (Falcor comparison)
Classification: PARTIALLY_COVERED
Original claim: "Ruvector uses 'sublinear algebra for queries' while Falcor uses 'linear.' Sparse matrix multiplications for graph traversal."
Priority: MEDIUM

#### Files to read:
| File | Package | Current Depth | LOC | Action |
|------|---------|--------------|-----|--------|
| No new files needed | - | - | - | - |

#### Existing findings that relate:
- Finding: "Cypher parser is production-quality: 1,296 LOC recursive descent" (ruvector-graph/cypher/parser.rs, INFO/ARCHITECTURE)
- Finding: "Cypher executor: 70-75% REAL - WORKING EXECUTOR" (rvlite/cypher/executor.rs, CRITICAL/QUALITY)
- Finding: "Query planner is naive string search -- not a real planner" (ruvector-graph/distributed/coordinator.rs, HIGH/ARCHITECTURE)
- Finding: "sparse.rs 95% BEST" (R28)
- Finding: "sparse.rs residual-sparse GENUINELY NOVEL" (R105)

#### What to verify:
- Cypher and SPARQL parsers/executors exist (confirmed REAL by R38)
- But graph traversal is via Cypher execution on property graphs, NOT via GraphBLAS-style sparse matrix multiplication
- The "sublinear algebra for queries" claim needs verification against the actual query execution path
- No GraphBLAS dependency found anywhere in ruvector

#### Suggested research agent: N/A -- existing findings adequately characterize the gap between claim and reality

---

### LEAD-012: PostgreSQL clustering and replication
Classification: PARTIALLY_COVERED
Original claim: "Ruvector can run entirely in PostgreSQL or outside it (SQLite). Supports backup, replication, consensus."
Priority: MEDIUM

#### Files to read:
| File | Package | Current Depth | LOC | Action |
|------|---------|--------------|-----|--------|
| crates/ruvector-replication/src/replica.rs | ruvector-rust | NOT_TOUCHED | 379 | Full DEEP read |
| crates/ruvector-replication/src/sync.rs | ruvector-rust | NOT_TOUCHED | 375 | Full DEEP read |
| crates/ruvector-replication/src/stream.rs | ruvector-rust | NOT_TOUCHED | 404 | Full DEEP read |

#### Existing findings that relate:
- Finding: "PRODUCTION-QUALITY SPARQL 1.1 PARSER" (ruvector-postgres, CRITICAL/ARCHITECTURE)
- Finding: "ALL 8 metric collection methods return empty/zero" (healing/detector.rs, HIGH/FACADE)
- Finding: "Postgres hyperbolic+healing COMPLETE" (R101)
- Finding: "COMPLETE MISLABELING: postgres benches contain ZERO postgres integration" (CRITICAL/FACADE)

#### What to verify:
- The ruvector-replication crate is entirely NOT_TOUCHED -- needs full reads
- Does replication implement actual Postgres logical replication or is it standalone?
- Is SQLite mode an alternative to Postgres (as claimed) or are they separate subsystems?

#### Suggested research agent: reader (3 files)

---

### LEAD-014: Hierarchical grid swarm with gossip discovery + Raft
Classification: PARTIALLY_COVERED
Original claim: "The swarm structure uses a hierarchical grid for node discovery. Nodes start knowing only a few seed peers, then gossip to discover the network before running Raft elections."
Priority: MEDIUM

#### Files to read:
| File | Package | Current Depth | LOC | Action |
|------|---------|--------------|-----|--------|
| crates/ruvector-raft/src/node.rs | ruvector-rust | NOT_TOUCHED | 632 | Shared with LEAD-005 |

#### Existing findings that relate:
- Finding: "Complete libp2p integration: Gossipsub, Kademlia DHT" (p2p.rs, CRITICAL/ARCHITECTURE) -- REAL gossip
- Finding: "Gossipsub configuration: mesh_n=6 optimal" (p2p.rs, HIGH/QUALITY)
- Finding: "Raft election logic correct, follows paper exactly, 95% correct" (advanced.rs, INFO/QUALITY)
- Finding: "CLI = demonstration framework" (R31)
- Finding: "ALL consensus protocol executions return hardcoded consensus_reached" (neural-coordination-protocol.js, CRITICAL/FACADE)

#### What to verify:
- The Rust-side gossip (libp2p in edge-net) and Raft (in examples/edge) are confirmed REAL
- But the JS/CLI swarm layer is confirmed FACADE
- The ruvector-raft crate is NOT_TOUCHED -- is it real or a duplicate of the edge examples?
- Does any code implement the "hierarchical grid" topology specifically?

#### Suggested research agent: reader (shared with LEAD-005 reads)

---

### LEAD-018: Causal Atlas blind test with witness chains
Classification: PARTIALLY_COVERED
Original claim: "The causal Atlas dashboard demonstrates blind-test methodology using vector space to find known planets, achieving 94% accuracy. Each step is cryptographically verified as a 'proof of work' witness chain. Deterministic execution."
Priority: MEDIUM

#### Files to read:
| File | Package | Current Depth | LOC | Action |
|------|---------|--------------|-----|--------|
| examples/rvf/examples/causal_atlas.rs | ruvector-rust | NOT_TOUCHED | 479 | Full DEEP read |
| examples/rvf/examples/causal_atlas_dashboard.rs | ruvector-rust | NOT_TOUCHED | 585 | Full DEEP read |

#### Existing findings that relate:
- Finding: "TWO completely different crypto systems for witness signing vs chain-linking" (rvf-runtime, CRITICAL/SECURITY)
- Finding: "simple_shake256_256 is NOT SHAKE-256 -- trivially collisible" (store.rs, CRITICAL/SECURITY)
- Finding: "WitnessBuilder produces ISOLATED bundles -- no inter-bundle chaining" (witness.rs, HIGH/ARCHITECTURE)
- Finding: "verify_witness() returns valid=true for ANY non-zero terminal hash" (rvf-node, CRITICAL/SECURITY)
- Finding: "Genuine Blake3 hash chain" (governance/witness.rs, INFO/ARCHITECTURE)

#### What to verify:
- Does causal_atlas.rs implement the blind-test planet detection?
- Is the witness chain in the example using real crypto (rvf-crypto) or the fake simple_shake256_256?
- Is execution truly deterministic?

#### Suggested research agent: reader (2 files)

---

### LEAD-022: NPX ruvector with RVF runtime
Classification: PARTIALLY_COVERED
Original claim: "Ruvector NPX runtime includes the RVF runtime, allowing execution without compilation. Published as both NPM package and Rust crate."
Priority: MEDIUM

#### Files to read:
| File | Package | Current Depth | LOC | Action |
|------|---------|--------------|-----|--------|
| npm/packages/ruvector/bin/cli.js | ruvector-rust | NOT_TOUCHED | 7357 | Full DEEP read -- main CLI entry |
| npm/packages/rvf-mcp-server/src/server.ts | ruvector-rust | NOT_TOUCHED | 569 | Full DEEP read -- RVF MCP server |

#### Existing findings that relate:
- Finding: "intelligence-engine.ts sync embed() NEVER produces ONNX embeddings" (CRITICAL/BUG)
- Finding: "hashEmbed() is the de facto embedding strategy" (intelligence-engine.ts, HIGH/ARCHITECTURE)
- Finding: "npm/packages/rvf/src/backend.ts" is DEEP (791 LOC)

#### What to verify:
- Does the NPX CLI actually load and execute RVF files?
- Is the rvf-mcp-server functional (new, NOT_TOUCHED)?
- Does the NPM ruvector package include native bindings or pure JS fallback?

#### Suggested research agent: reader (2 files)

---

## ALREADY_COVERED LEADS (skip)

### LEAD-015: Development pipeline (ruvector -> AgentDB + agentic-flow -> Claude flow)
Classification: ALREADY_COVERED
Original claim: "The development pipeline flows: ruvector (lab/experimentation) -> AgentDB + agentic-flow (commercial packaging, bug-tested) -> Claude flow (final product)."
Covered by: R40 finding "agentic-flow = single-node task runner" + extensive AgentDB RVF backend analysis showing 15+ DEEP files. The pipeline description matches observed architecture: ruvector has raw implementations, agentic-flow packages them (often with quality loss), claude-flow is the CLI product. The "lab to product" pattern is thoroughly documented.

---

### LEAD-016: Transfer learning in Claude flow / APF plugin
Classification: ALREADY_COVERED
Original claim: "Claude flow's transfer learning component allows exporting learnings between different environments."
Covered by: CLI hooks command findings show `transfer` as one of 19 subcommands. The ruvector-domain-expansion/transfer.rs (DEEP, 584 LOC) implements domain transfer. Sona federated.rs (DEEP, 682 LOC) implements federated learning. The claude-flow dist/src/commands/neural.js finding confirms "Export has EXCELLENT security: ephemeral Ed25519 keys, PII stripping." Transfer learning EXISTS but is spread across multiple disconnected implementations (Rust crate, JS CLI, TS hooks).

---

### LEAD-021: MCP device on Raspberry Pi with endpoints
Classification: ALREADY_COVERED
Original claim: "The MCP device on the Raspberry Pi exposes endpoints for loading kernels, RVFs, upgrading hardware, GPIO sensor interaction."
Covered by: Research has documented 7+ parallel MCP implementations. The npm/packages/rvf-mcp-server/ (NOT_TOUCHED, 4 files, 802 LOC) is the RVF-specific MCP server. The known mcp-gate crate is COMPLETE at ~91% (R114). The claim of an 8th MCP for IoT/appliance aligns with the known proliferation pattern. However, the specific Raspberry Pi firmware MCP is likely in a private/commercial repo not available for verification.

Note: The rvf-mcp-server files ARE NOT_TOUCHED and could be read as part of LEAD-022 work, but the core claim about appliance-specific MCP is already characterized by the known pattern.

---

### LEAD-013: Personal 6GB memory store with federated export
Classification: ALREADY_COVERED
Original claim: "Ruv has a personal 6GB vector/graph memory structure. Plans to share a federated version with PII stripped."
Covered by: This is a private/personal data store, not public code. Research has documented: (1) federated learning in sona/training/federated.rs (DEEP), (2) FederatedSessionManager.ts (DEEP, CRITICAL bug: does NOT implement FedAvg), (3) PII stripping in neural.js export (confirmed REAL). The infrastructure for federated RVF export EXISTS but has quality issues. The 6GB personal store is a runtime artifact, not verifiable from source code.

---

## UNRESOLVABLE LEADS

### LEAD-006: Raspberry Pi Zero appliance (ARM, mDNS, GPIO, thermal management)
Classification: UNRESOLVABLE
Original claim: "Ruvector appliance runs on Raspberry Pi Zero (512MB RAM, 16GB SD, 4-core ARM). Uses mDNS, WiFi, overclocking, GPIO sensor integration."
Reason: DB search for ARM, mDNS, GPIO, thermal, and appliance patterns returns NO relevant files in any analyzed repo. The appliance firmware is likely in a private/commercial repository not part of the 4 public repos analyzed. The cognitum-gate-kernel crate exists but is about shard management and evidence gating, not ARM firmware. No verification possible from available codebases.

---

### LEAD-020: Stuart's "ask-roof-net" external project
Classification: UNRESOLVABLE
Original claim: "Stuart built an automated system that sweeps 148 Ruv repos nightly, stores content in ruvector."
Reason: This is an external community project not part of ruvector's codebase. It validates the research project's own findings about feature discoverability challenges but cannot be verified from the analyzed repos. The mention of 148 repos is notable -- our research analyzes 4 public repos.

---

## Verification Statistics

| Metric | Value |
|--------|-------|
| Total leads processed | 22 |
| Unique files resolved | ~60+ across all leads |
| Files already at DEEP | 14 RVF files, 27+ MinCut files, multiple AgentDB RVF backend files |
| Files needing first read | ~35 (primarily ruvector-raft, ruvector-delta-consensus, ruvector-solver, ruvector-replication, QR seed, causal atlas examples) |
| Findings cross-referenced | ~120 (across all lead resolution queries) |
| Contradictions found | 1 (LEAD-017: LOC count CONFIRMED, postgres/graph claims are nuanced) |
| Leads skippable | 6 (4 ALREADY_COVERED + 2 UNRESOLVABLE) |
| Leads actionable | 16 (1 CONTRADICTION + 8 NEW + 7 PARTIALLY_COVERED) |

## Prioritized Reading Order

### Session 1 (R115): Consensus & Distribution (~10 files)
1. crates/ruvector-raft/src/node.rs (632 LOC) -- LEAD-005, LEAD-014
2. crates/ruvector-raft/src/election.rs (360 LOC) -- LEAD-005, LEAD-014
3. crates/ruvector-raft/src/log.rs (351 LOC) -- LEAD-005
4. crates/ruvector-delta-consensus/src/lib.rs (489 LOC) -- LEAD-002, LEAD-005
5. crates/ruvector-delta-consensus/src/crdt.rs (486 LOC) -- LEAD-002, LEAD-005
6. crates/ruvector-delta-consensus/src/causal.rs (320 LOC) -- LEAD-002, LEAD-005
7. crates/ruvector-delta-consensus/src/conflict.rs (286 LOC) -- LEAD-002
8. crates/ruvector-replication/src/replica.rs (379 LOC) -- LEAD-012
9. crates/ruvector-replication/src/sync.rs (375 LOC) -- LEAD-012

### Session 2 (R116): RVF Deep Dive (~8 files)
1. crates/rvf/rvf-runtime/src/qr_encode.rs (1135 LOC) -- LEAD-004
2. crates/rvf/rvf-runtime/src/qr_seed.rs (1095 LOC) -- LEAD-004
3. crates/rvf/rvf-runtime/src/seed_crypto.rs (236 LOC) -- LEAD-004
4. crates/rvf/rvf-types/src/agi_container.rs (963 LOC) -- LEAD-003
5. crates/rvf/rvf-ebpf/src/lib.rs (1083 LOC) -- LEAD-003
6. examples/rvf/examples/causal_atlas.rs (479 LOC) -- LEAD-018
7. examples/rvf/examples/causal_atlas_dashboard.rs (585 LOC) -- LEAD-018

### Session 3 (R117): Solver & Transformers (~6 files)
1. crates/ruvector-solver/src/true_solver.rs (933 LOC) -- LEAD-007
2. crates/ruvector-solver/src/router.rs (1615 LOC) -- LEAD-007
3. crates/ruvector-solver/src/simd.rs (282 LOC) -- LEAD-007
4. crates/rvf/rvf-solver-wasm/src/engine.rs (784 LOC) -- LEAD-007
5. examples/delta-behavior/applications/04-self-stabilizing-world-model.rs (554 LOC) -- LEAD-010
6. crates/rvf/rvf-adapters/agentdb/src/pattern_store.rs (457 LOC) -- LEAD-009

### Session 4 (R118): NPM/CLI & Remaining (~4 files)
1. npm/packages/ruvector/bin/cli.js (7357 LOC) -- LEAD-022
2. npm/packages/rvf-mcp-server/src/server.ts (569 LOC) -- LEAD-022
3. crates/rvf/rvf-adapters/agentdb/src/vector_store.rs (327 LOC) -- LEAD-009
4. Mapper pass: trace MinCut -> RVF dependencies -- LEAD-008
