# Ruvector Domain Analysis

> **Priority**: HIGH | **Coverage**: ~9.2% (196/~2,150 DEEP est.) | **Status**: In Progress
> **Last updated**: 2026-02-17 (Session R91)

## 1. Current State Summary

The ruvector domain is a 76-crate Rust monorepo with 50+ npm packages providing distributed vector database infrastructure with self-learning capabilities. Actual codebase is ~400K LOC Rust (not the claimed 2M), representing 81 days of human-AI co-development at 10.3 commits/day — 6-20x faster than sustainable human-only velocity.

**Top-level verdicts:**

- **Hash-based embeddings are systemic across Rust + JS.** 7+ files default to character-sum or FNV-1a hash, not semantic embeddings. All "semantic search" using defaults is character-frequency matching.
- **Best code:** temporal-tensor (95%, production-ready), ruQu QEC (91→89% revised with subpoly_decoder drag), ruvllm kernels (90%, NEON SIMD), cognitum-gate-kernel (93%, rivals neural-network-impl), postgres SIMD (95-98%), ruqu-core (noise 96-98%, mitigation 95-98%, transpiler 95-98%).
- **Worst code:** ruvector-graph distributed module shows a new "transport-absent distributed protocol" pattern — algorithm state machines are correctly designed but no socket I/O exists (15-80% range). ruvector-graph Cypher parser has NO executor (30-35% overall). Postgres HNSW `connect_node_to_neighbors()` completely empty. Speculative decoding 2x SLOWER than vanilla. index_bench.rs (42%) theatrical benchmarking. subpolynomial/mod.rs (45-50%) false complexity claims. **subpoly_decoder.rs (35-40%) 3rd FALSE subpolynomial** — O(n²) greedy under "provable" claims.
- **Edge AI confirmed production-grade** — lora.rs (90-95%) real dual-SIMD LoRA with Q4/Q8 quantization, federated.rs (95-98%) BEST federated learning in project (Byzantine-robust, differential privacy, TopK compression).
- **ruvector-core advanced features confirmed genuine (85-93%):** product_quantization.rs (88-92%) real k-means++ and Lloyd's with ADC. conformal_prediction.rs (88-93%) valid split-conformal with Vovk et al. quantile. hypergraph.rs (85-90%) genuine bipartite incidence and k-hop BFS citing HyperGraphRAG (NeurIPS 2025). tda.rs (60-70%) MISLABELED — implements graph metrics, not persistent homology (11th mislabeled file).
- **ruvector-graph distributed module (15-80% gradient):** shard.rs (70-80%) has real EdgeCutMinimizer multilevel Kernighan-Lin and xxh3/blake3 hashing. gossip.rs (45-55%) correct SWIM state machine, no transport. federation.rs (40-50%) real merge logic, execute_on_cluster always empty. coordinator.rs (30-35%) 2PC types defined, state machine frozen. rpc.rs (15-20%) all 4 RPC methods stubs, gRPC feature-gated out.
- **TWO independent LoRA implementations**: micro_lora.rs (training-focused, EWC++, sona) vs edge-net lora.rs (inference-focused, quantized, WASM).
- **Core HNSW wraps hnsw_rs** — vendored upstream v0.3.3 (NOT patched), but adds real value (SIMD, REDB, concurrency).
- **Attention crate is real** — 18+ implementations (Flash, Hyperbolic, MoE, Graph, Sheaf, OT, PDE) across 66 files, ~9,200 LOC. SIMD/Rayon features are no-ops.
- **Three distinct HNSW implementations:** ruvector-core (wrapper), micro-hnsw-wasm (novel `#![no_std]` <12KB), hyperbolic-hnsw (Poincare geometry).
- **TWO independent SIMD codebases:** ruvector-core (distance metrics for HNSW) and edge-net (NN inference — matmul, activations, quantization). Zero code sharing.
- **DUAL query languages confirmed:** Cypher (parser only, no executor) AND SPARQL (93-95% parser + 92% executor). Both property graphs and RDF triple stores supported.
- **AI co-authored explicitly** — commits credit "Claude Opus 4.5/4.6". Scope (GNN, quantum, FPGA, Raft, graph DB, 39 attention types, postgres ext) would take 2-3 years for experienced team.
- **R91 additions (5 files, ~3,935 LOC):** mmap.rs (88-92%) genuine memmap2 file-backed mmap + AtomicBitmap, for GNN training only (no HNSW integration). compress.rs (55-65%) 12TH MISLABELED FILE — named "graph compression" but implements embedding/tensor quantization with fake IEEE f16. speculative.rs (88-92%) EAGLE-style tree speculative decoding with novel lambda-guided confidence; sequential path verification (not true parallel). rope.rs (88-92%) correct RoPE with NTK-aware scaling and partial YaRN. kv_cache/legacy.rs (82-88%) RotateKV with FWHT; per-head scale stomping bug; no eviction policy.

## 2. File Registry

### ruvector-core & HNSW

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| simd_intrinsics.rs | ruvector-core | 1,605 | 90% | DEEP | Real AVX-512/AVX2/NEON runtime detection. PQ incomplete (RESOLVED by product_quantization.rs) | C |
| agenticdb.rs | ruvector-core | 1,447 | 70% | DEEP | Metadata filtering integration. Hash embeddings CRITICAL | C |
| lockfree.rs | ruvector-core | 591 | 85% | DEEP | Real lock-free structures via crossbeam | C |
| hnsw.rs | hnsw_rs vendored | 1,873 | 98-100% | DEEP | NOT a patch — vendored upstream v0.3.3. Zero modifications. Complete Malkov & Yashunin | R52 |
| hnswio.rs | hnsw_rs vendored | 1,704 | 95-98% | DEEP | Dual-file persistence, 4 format versions, hybrid mmap. No postgres/AgentDB connection | R52 |
| libext.rs | hnsw_rs fork | 1,241 | 75-85% | DEEP | Julia FFI. CRIT: no bounds checking, std::mem::forget | R36 |
| datamap.rs | hnsw_rs fork | 458 | 85-90% | DEEP | Zero-copy mmap. CRIT: use-after-free risk | R36 |
| product_quantization.rs | ruvector-core | 551 | 88-92% | DEEP | Real k-means++ + Lloyd's + ADC with LUT. RESOLVES H1 | R90 |
| conformal_prediction.rs | ruvector-core | 505 | 88-93% | DEEP | Valid split-conformal, Vovk et al. quantile, 3 nonconformity measures. 7 tests | R90 |
| hypergraph.rs | ruvector-core | 551 | 85-90% | DEEP | Genuine bipartite incidence, k-hop BFS, causal memory utility fn. Cites HyperGraphRAG (NeurIPS 2025) | R90 |
| tda.rs | ruvector-core | 497 | 60-70% | DEEP | MISLABELED — graph metrics only, no persistent homology. 11th mislabeled file | R90 |

### Attention & Neural

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| ruvector-attention (66 files) | ruvector-attention | ~9,200 | 80% | DEEP | 18+ real implementations. SIMD/Rayon no-ops | B |
| ruvector-gnn (~40 files) | ruvector-gnn | ~6,000 | 80% | DEEP | Custom hybrid GAT+GRU+edge, full EWC | C |
| micro-hnsw-wasm | ruvector | 1,263 | 60-70% | DEEP | Novel `#![no_std]` HNSW. 6 neuromorphic features UNTESTED | R36 |
| mmap.rs | ruvector-gnn (or ruvector) | 918 | 88-92% | DEEP | Real memmap2 file-backed mmap, AtomicBitmap lock-free, Linux madvise(MADV_WILLNEED), 17 tests. GNN training only — no HNSW integration. Pin count unused (no eviction) | R91 |

### ruvector LLM Extensions

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| compress.rs | ruvector (graph compression module) | 679 | 55-65% | DEEP | **12th MISLABELED FILE** — named "graph compression" implements embedding/tensor quantization. 5-tier access-frequency tiers. Fake IEEE f16 (fixed-point ×1000). Trivial PQ codebook (linear interpolation). Binary quantization correct. Zero GNN graph types | R91 |
| speculative.rs | ruvector | 788 | 88-92% | DEEP | EAGLE-style tree speculative decoding. Textbook rejection sampling. Novel lambda-guided confidence from mincut signal. Correct tree attention mask. Sequential path verification — not true parallel tree forward. Logit-processing only, no model objects | R91 |
| rope.rs | ruvector | 777 | 88-92% | DEEP | Correct RoPE (Su et al. 2021). NTK-aware scaling (CodeLlama/Qwen formula). Partial YaRN (base+bands, missing attention scale factor). Q15 quantized path. 11 substantive tests. No false SIMD claims. Independent from ruvllm/kernels/rope.rs | R91 |
| kv_cache/legacy.rs | ruvector | 773 | 82-88% | DEEP | RotateKV (IJCAI 2025) with Fast Walsh-Hadamard Transform. 2-bit/4-bit quantization with correct bit-packing. Per-head scale stomping bug (overwrites min/max on each new token). No eviction policy. 15 genuine tests | R91 |

### Temporal Tensor (Production-Grade)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| store.rs | temporal-tensor | ~2,500 | 92-95% | DEEP | BEST FILE. 74.7KB. CRC32, SVD reconstruction, 4-tier quant | R22 |
| store_ffi.rs | temporal-tensor | 889 | 90-92% | DEEP | 11 extern "C" FFI functions for WASM/C | R37 |
| agentdb.rs | temporal-tensor | 843 | 88-92% | DEEP | Pattern-aware tiering, 4-dim PatternVector. 36 tests | R37 |
| quantizer.rs | temporal-tensor | 1,430 | 93-95% | DEEP | K-means PQ with asymmetric distance | R37 |
| compressor.rs | temporal-tensor | 1,568 | 95-98% | DEEP | Delta+run-length+Huffman pipeline, CRC32 | R37 |
| tiering.rs | temporal-tensor | 1,613 | 93-95% | DEEP | 4-tier Hot→Warm→Cold→Archive with LRU | R37 |

### ruvllm LLM Inference

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| bitnet/backend.rs | ruvllm | 4,559 | 92-95% | DEEP | Complete BitNet 1-bit LLM. MLA 17.8x memory reduction | R22 |
| kernels/attention.rs | ruvllm | 2,215 | 88-92% | DEEP | Flash Attention 2, NEON dot product, paged KV cache | R22 |
| kernels/matmul.rs | ruvllm | 2,050 | 85-90% | DEEP | 12x4 GEMM micro-kernel, Apple Accelerate, Metal GPU | R22 |
| memory_pool.rs | ruvllm | 1,704 | 95% | DEEP | Lock-free bump allocator, RAII buffer pool, 12 tests | R34 |
| autodetect.rs | ruvllm | 1,945 | 92% | DEEP | Hardware detection (Metal, CPU features). 27 tests | R34 |
| kv_cache.rs | ruvllm | 1,528 | 90% | DEEP | Two-tier KV cache, NEON SIMD quantize/dequantize | R34 |
| norm.rs | ruvllm/kernels | 652 | 95% | DEEP | BEST quality — 4x unrolled FMA, correct variance | R35 |
| rope.rs | ruvllm/kernels | 660 | 95% | DEEP | Real RoPE, NEON interleaved ops, NTK-aware scaling | R35 |
| quantized.rs | ruvllm/kernels | 1,219 | 92% | DEEP | Real NEON int8/int4/q4k kernels, llama.cpp-compatible | R35 |
| activations.rs | ruvllm/kernels | 1,041 | 92% | DEEP | Vectorized exp/sigmoid/tanh, polynomial approx | R35 |
| ane_ops.rs | ruvllm/kernels | 1,758 | 70% | DEEP | MISLEADING: gelu_ane/silu_ane are SCALAR FALLBACKS | R35 |
| scheduler.rs | ruvllm/serving | 840 | 90-92% | DEEP | vLLM-style continuous batching, preemption, chunked prefill | R35 |
| engine.rs | ruvllm/serving | 1,302 | 80-85% | DEEP | Real continuous batching. Fallback: hash%32000 when no model | R35 |
| speculative.rs | ruvllm/bitnet | 1,392 | 55-60% | DEEP | CRITICAL: 2K forward passes for K tokens = SLOWER | R35 |
| reasoning_bank.rs | ruvllm/claude_flow | 1,520 | 92-95% | DEEP | Fourth ReasoningBank. Real K-means, EWC++. 16 tests | R37 |
| hnsw_router.rs | ruvllm/claude_flow | 1,288 | 90-93% | DEEP | BEST ruvector-core integration. Hybrid semantic+keyword | R37 |
| model_router.rs | ruvllm/claude_flow | 1,292 | 88-92% | DEEP | 7-factor complexity, feedback tracking 1000 predictions | R37 |
| pretrain_pipeline.rs | ruvllm/claude_flow | 1,394 | 85-88% | DEEP | Multi-phase pretraining. CRIT: hash-based embeddings | R37 |
| claude_integration.rs | ruvllm/claude_flow | 1,344 | 70-75% | DEEP | CRIT: execute_workflow SIMULATION, hardcoded 500 tokens | R37 |
| micro_lora.rs | ruvllm/training | 1,261 | 92-95% | DEEP | BEST learning code. NEON 8x unroll, EWC++. <1ms forward | R37 |
| grpo.rs | ruvllm/training | 898 | 90-92% | DEEP | Textbook GRPO: GAE, PPO clipping, adaptive KL. 16 tests | R37 |
| real_trainer.rs | ruvllm/training | 1,000 | 70-75% | DEEP | Triplet loss + InfoNCE. CRIT: hash-based embeddings | R37 |
| tool_dataset.rs | ruvllm/training | 2,147 | 88-92% | DEEP | 140+ MCP tool-call templates, 19 categories | R37 |

### Postgres Extension

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| distance/simd.rs | ruvector-postgres | 2,129 | 95-98% | DEEP | BEST SIMD IN ECOSYSTEM. AVX-512/AVX2/NEON. 23 tests | R22 |
| index/hnsw_am.rs | ruvector-postgres | 1,997 | 75-80% | DEEP | CRIT: connect_node_to_neighbors() EMPTY | R22 |
| index/ivfflat_am.rs | ruvector-postgres | 2,165 | 80-85% | DEEP | Real k-means++, Lloyd. STUBS: insert, delete, retrain | R22 |
| sparql/parser.rs | ruvector-postgres | 2,496 | 93-95% | DEEP | PRODUCTION W3C SPARQL 1.1 parser. All 4 query forms, property paths, 33+ functions | R52 |
| sparql_executor.rs | ruvector-postgres | 1,885 | 92% | DEEP | COMPLETE SPARQL 1.1 query engine. BGP, property paths, 7 aggs | R34 |
| index_bench.rs | ruvector-postgres | 1,395 | 42% | DEEP | THEATRICAL: HNSW search is brute-force O(n). Zero postgres integration despite location | R52 |
| operators.rs | ruvector-postgres | ~1,200 | 85% | DEEP | 54 verified SQL functions | Initial |
| healing/learning.rs | ruvector-postgres | 670 | 92-95% | DEEP | Genuine adaptive weight formula, confidence scoring | R36 |
| healing/detector.rs | ruvector-postgres | 826 | 85-90% | DEEP | 8 problem types. All 8 metric collection methods EMPTY | R36 |
| healing/engine.rs | ruvector-postgres | 789 | 75-80% | DEEP | Cooldown/rate-limiting real. CRIT: no timeout enforcement | R36 |
| healing/strategies.rs | ruvector-postgres | 1,166 | 60-65% | DEEP | StrategyRegistry 95%. ALL 5 executions log-only stubs | R36 |

### ruQu Quantum Error Correction

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| decoder.rs | ruQu | 2,400 | 95-98% | DEEP | BEST FILE. Union-Find O(α(n)) + MWPM. Top 15% quality | R37 |
| syndrome.rs | ruQu | 1,640 | 90-92% | DEEP | Real AVX2 SIMD vpshufb lookup popcount | R37 |
| surface_code.rs | ruQu | 1,820 | 88-92% | DEEP | Complete surface code, weight-2 stabilizers | R37 |
| qec_scheduler.rs | ruQu | 1,505 | 88-92% | DEEP | Critical path learning. All remote providers stub | R37 |
| noise_model.rs | ruQu | 1,330 | 82-85% | DEEP | 7 noise channels, Kraus operator validation | R37 |
| tile.rs | ruQu | 2,125 | 92% | DEEP | Coherence gate, Union-Find, Ed25519 signatures. 27 tests | R39 |
| planner.rs | ruQu | 1,478 | 88% | DEEP | 4 backend cost models, entanglement estimation. 33 tests | R39 |
| filters.rs | ruQu | 1,357 | 82-86% | DEEP | MISNAMED — coherence quality gate, NOT quantum filtering. Three-filter pipeline (structural/shift/evidence). 14 tests | R54 |
| fabric.rs | ruQu | 1,280 | 93-96% | DEEP | Production 256-tile WASM fabric orchestrator. Blake3 audit trails, surface code generator. 23 tests | R54 |

### ruqu-core Extended

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| mitigation.rs | ruqu-core | 1,276 | 95-98% | DEEP | Three NISQ-era strategies (ZNE, Measurement Error, CDR). Richardson extrapolation, tensor-product calibration. 40+ tests | R54 |
| transpiler.rs | ruqu-core | 1,211 | 95-98% | DEEP | Complete 3-phase circuit transpiler. 3 hardware backends (IBM/IonQ/Rigetti). 44 tests. BEST-IN-CLASS | R54 |
| subpoly_decoder.rs | ruqu-core | 1,208 | 35-40% | DEEP | **FALSE SUBPOLYNOMIAL** (3rd instance). O(n²) greedy under "provable O(d^{2-ε})" claims. Zero citations | R54 |
| noise.rs | ruqu-core | 1,175 | 96-98% | DEEP | Production Kraus operator formalism. 4 noise channels + hardware calibration pipeline. 498 test lines. BEST-IN-CLASS | R54 |

### Prime-Radiant Coherence

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| restriction.rs | prime-radiant | 1,489 | 90-92% | DEEP | BEST sparse matrix — CSR 6 formats, 4x SIMD unroll | R37 |
| memory_layer.rs | prime-radiant | 1,260 | 92-95% | DEEP | Triple memory, real cosine similarity. 19 tests | R37 |
| witness_log.rs | prime-radiant | 1,130 | 88-92% | DEEP | blake3 hash chains with tamper evidence. 16 tests | R37 |
| coherence.rs | prime-radiant | 1,500 | 88-90% | DEEP | Sheaf Laplacian, spectral gap computation | R37 |
| knowledge_graph.rs | prime-radiant | 1,190 | 85-88% | DEEP | DashMap concurrent graph, topological sort | R37 |

### Cognitum Gate & Other Specialized

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| lib.rs | cognitum-gate-kernel | 713 | 95% | DEEP | Custom bump allocator, no_std WASM. 6 FFI exports | R36 |
| report.rs | cognitum-gate-kernel | 491 | 98% | DEEP | 64-byte cache-line aligned, compile-time assertions | R36 |
| delta.rs | cognitum-gate-kernel | 465 | 98% | DEEP | Tagged union 7 operation types, fixed-size FFI-safe | R36 |
| shard.rs | cognitum-gate-kernel | 983 | 92% | DEEP | Optimal union-find, iterative path compression | R36 |
| evidence.rs | cognitum-gate-kernel | 852 | 88% | DEEP | Fixed-point log-space, anytime-valid e-process | R36 |
| sparse.rs | sublinear-solver | 964 | 95% | DEEP | 4 sparse formats (CSR/CSC/COO/Graph), no_std | R28 |
| wrapper/mod.rs | ruvector-mincut | 1,505 | 90% | DEEP | Bounded-range decomposition from arXiv:2512.13105. 22 tests | R34 |
| hierarchy.rs | ruvector-mincut | 1,489 | 88% | DEEP | 3-level hierarchy. check_and_split_expander incomplete | R34 |
| subpolynomial/mod.rs | ruvector-mincut | 1,385 | 45-50% | DEEP | FALSE subpolynomial complexity. Invalid arXiv citation. Same R39 pattern | R52 |
| graph Cypher parser | ruvector-graph | 1,296 | 95% | DEEP | Production parser. CRIT: NO EXECUTOR | C |

### ruvector-graph Distributed

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| distributed/shard.rs | ruvector-graph | 596 | 70-80% | DEEP | BEST distributed file. EdgeCutMinimizer multilevel KL, real xxh3/blake3. In-memory only | R90 |
| distributed/gossip.rs | ruvector-graph | 624 | 45-55% | DEEP | Correct SWIM state machine + failure detector, no network transport (log-only) | R90 |
| distributed/federation.rs | ruvector-graph | 583 | 40-50% | DEEP | Real merge/dedup logic + FederationStrategy dispatch, execute_on_cluster always returns empty Vec | R90 |
| distributed/coordinator.rs | ruvector-graph | 536 | 30-35% | DEEP | 2PC types defined, state machine frozen at Active, no network layer, naive string-based query planner | R90 |
| distributed/rpc.rs | ruvector-graph | 516 | 15-20% | DEEP | All 4 RPC methods stubs. gRPC (tonic) feature-gated out of default build | R90 |

### Edge-Net P2P Transport

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| simd.rs | edge-net (ruvector) | 1,418 | 92-95% | DEEP | COMPLETE independent SIMD for NN inference. Real AVX2/WASM/SSE4.1, Q4/Q8 quantization, numerically stable | R52 |
| lora.rs | edge-net (ruvector) | 1,355 | 90-95% | DEEP | Complete edge LoRA. Dual SIMD (AVX2+WASM128), Q4/Q8 quantization, LRU adapter pool, WASM bindings. 15 tests. Independent from micro_lora.rs | R54 |
| federated.rs | edge-net (ruvector) | 1,218 | 95-98% | DEEP | BEST federated learning in project. Byzantine-robust (MAD+median), differential privacy (Gaussian), TopK compression with error feedback, reputation-weighted FedAvg. 13 tests | R54 |
| p2p.rs | edge-net (ruvector) | 845 | 92-95% | DEEP | **REVERSES R42**: Real libp2p (Gossipsub/Kademlia/RequestResponse/Identify), NOISE+yamux+TCP, direct RAC integration via broadcast_rac_event(), 6 gossipsub topics, production P2P | R44 |
| advanced.rs | edge (ruvector) | 2041 | 72% | DEEP | MISNOMER — zero networking. ML primitives: Raft 85%, SNN 95% (STDP), HDC 93%, HNSW reimpl 88%, hash embeddings (8th occurrence), quantization 92% | R44 |
| swarm.rs | edge (ruvector) | 612 | 72% | DEEP | Production crypto protocol (Ed25519+AES-256-GCM 88%, identity registry 85%, task claiming 80%) but 0% GUN network transport — all publish = stubs | R44 |

### SONA & Learning

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| sona (27 files) | ruvector | ~4,500 | 85% | DEEP | MicroLoRA, EWC++, federated, SafeTensors. Production-ready | R13 |

## 3. Findings Registry

### 3a. CRITICAL Findings

| ID | Description | File(s) | Session | Status |
|----|-------------|---------|---------|--------|
| C1 | **Hash-based embeddings systemic** — 7+ files (Rust + JS) default to character-sum, FNV-1a, or position hash. All semantic search using defaults is character-frequency matching | ruvector-core agenticdb.rs, ruvector-cli hooks.rs, pretrain_pipeline.rs, real_trainer.rs, rlm_embedder.rs, learning-service.mjs, enhanced-embeddings.ts | C, R22, R37 | Open |
| C2 | **HNSW deletions broken** — hnsw_rs does not support vector deletion. Delete method silently fails or panics | ruvector-core | C | Open |
| C3 | **Postgres HNSW neighbor connections empty** — `connect_node_to_neighbors()` is COMPLETELY EMPTY. Graph never actually built during insertion | index/hnsw_am.rs | R22 | Open |
| C4 | **Postgres IVFFlat mutations stubbed** — `aminsert()`, `ambulkdelete()`, `retrain()` all stubs. Index can be built/searched but not updated | index/ivfflat_am.rs | R22 | Open |
| C5 | **ruvector-graph has NO query executor** — Cypher parser generates full AST but nothing executes queries. "Working Cypher" claim is false | ruvector-graph | C | Open |
| C6 | **execute_workflow SIMULATION** — claude_integration.rs hardcodes 500 tokens, no real Claude API calls despite complete type system | claude_integration.rs | R37 | Open |
| C7 | **Speculative decoding SLOWER than vanilla** — 2K sequential forward passes for K tokens vs K vanilla passes. Current implementation is anti-optimization | speculative.rs | R35 | Open |
| C8 | **Fourth ReasoningBank with zero code sharing** — reasoning_bank.rs joins claude-flow, agentic-flow, agentdb implementations. All implement same algorithms independently | reasoning_bank.rs | R37 | Open |
| C9 | **HNSW patches FFI unsafe** — libext.rs has no bounds checking on C pointer dereferences, std::mem::forget on vectors returned to C (memory leak risk) | libext.rs | R36 | Open |
| C10 | **HNSW patches no integrity validation** — hnswio.rs has no checksum/hash on serialized dumps. Corrupted data silently loads | hnswio.rs | R36 | Open |
| C11 | **Postgres healing strategies are stubs** — All 5 healing strategies (reindex, promote, evict, block queries, repair edges) are log-only | healing/strategies.rs | R36 | Open |
| C12 | **Postgres healing metrics empty** — All 8 metric collection methods in detector.rs return empty/zero. Self-healing system cannot detect problems | healing/detector.rs | R36 | Open |
| C13 | **micro-hnsw-wasm neuromorphic unvalidated** — 6 novel features (spike encoding, homeostatic plasticity, 40Hz resonance, WTA, dendritic computation, temporal patterns) have ZERO tests | micro-hnsw-wasm | R36 | Open |
| C14 | **ruQu remote quantum providers ALL stub** — IBM/IonQ/Rigetti/Braket all return AuthenticationFailed. Only LocalSimulator works | qec_scheduler.rs | R37 | Open |
| C15 | **HNSW "patches" are vendored upstream** — hnsw.rs is unmodified hnsw_rs v0.3.3 from crates.io. Directory name `scripts/patches/` is misleading — zero ruvector-specific modifications | hnsw.rs | R52 | Open |
| C16 | **index_bench.rs HNSW is brute-force O(n)** — search_with_ef() iterates ALL nodes linearly. Benchmarks measure brute-force kNN, NOT real HNSW graph traversal | index_bench.rs | R52 | Open |
| C17 | **index_bench.rs zero postgres** — Located in ruvector-postgres/benches/ but contains zero PostgreSQL code (no pgrx, sqlx, tokio_postgres). Complete mislabeling | index_bench.rs | R52 | Open |
| C18 | **subpolynomial/mod.rs invalid paper citation** — Claims arXiv:2512.13105 (Dec 2024) but arXiv format 2512.NNNNN means Dec 2025, not 2024. Theoretical foundation suspect | subpolynomial/mod.rs | R52 | Open |
| C19 | **subpolynomial/mod.rs FALSE complexity claims** — Claims O(n^{o(1)}) subpolynomial but implements O(log n) levels × O(recourse). Same R39 false sublinearity pattern | subpolynomial/mod.rs | R52 | Open |
| C20 | **subpolynomial/mod.rs deterministic contradiction** — Claims "deterministic" subpolynomial mincut, which is an OPEN PROBLEM in graph algorithms. No randomization in code either | subpolynomial/mod.rs | R52 | Open |
| C21 | **subpoly_decoder.rs FALSE subpolynomial** — 3rd instance of false complexity. Claims "provable O(d^{2-ε} polylog d)" but ALL 3 decoders use O(n²) greedy_pair_and_correct. Zero citations, zero empirical validation | subpoly_decoder.rs | R54 | Open |
| C22 | **filters.rs COMPLETE domain mislabeling** — Named "filters" in quantum crate but implements classical coherence quality gate. Zero connection to decoder.rs. Zero quantum error filtering | filters.rs | R54 | Open |
| C23 | **ruQu contains two unrelated systems** — QEC (decoder/syndrome/surface_code) and coherence gate (filters/fabric/tile) share a crate name but have ZERO cross-references. "Qu" may mean "Quality" not "Quantum" | ruQu crate | R54 | Open |
| C24 | **"Transport-absent distributed protocol" — new pattern class.** All 5 files in ruvector-graph/src/distributed/ share the same defect: algorithm logic and state machines are correctly designed, but every network send is replaced with a debug log comment "In production, send actual network message". Zero socket I/O anywhere in the module. The distributed graph system is a design doc rendered as code | distributed/ (5 files) | R90 | Open |
| C25 | **tda.rs MISLABELED — 11th mislabeled file.** Named "Topological Data Analysis" but implements zero canonical TDA. No Vietoris-Rips complex, no boundary operators, no Betti numbers, no persistence diagrams. Implements graph metrics (connected components, clustering coefficient, diagonal-covariance degeneracy, multi-scale component counting). Misleading to any consumer expecting homology | tda.rs | R90 | Open |
| C26 | **coordinator.rs 2PC never transitions from Active.** TransactionState enum has Active/Preparing/Committed/Aborted. commit_transaction() removes the entry and logs — no prepare phase, no participant coordination, no WAL, no rollback. 2PC is type-system scaffolding only; the state machine is frozen at creation | coordinator.rs | R90 | Open |
| C27 | **rpc.rs all 4 RPC methods are stubs, gRPC feature-gated out.** RpcClient.execute_query() returns empty QueryResult with all-zero stats. RpcServer.start() logs a string. GraphRpcService and tonic::async_trait are gated behind cfg(feature="federation") absent from default Cargo.toml. In the standard build, zero gRPC code compiles | rpc.rs | R90 | Open |
| C28 | **compress.rs 12th MISLABELED FILE — "graph compression" is actually embedding/tensor quantization.** Zero GNN graph types (GraphEdge, NodeFeature, etc.) are imported or referenced. The file implements 5-tier access-frequency compression (hot/warm/cool/cold/archive) for embedding vectors and tensors. No compression of graph topology, adjacency, or edge weights whatsoever. Consumers expecting graph compression will find tensor quantization routines | compress.rs | R91 | Open |
| C29 | **compress.rs fake IEEE f16 — fixed-point *1000.0 substituted for real half-precision float.** WarmCompressor::compress_to_f16() multiplies by 1000.0 and stores as i16. This is lossy fixed-point with range ±32.767 (values ≥32.768 overflow silently). Real IEEE 754 binary16 has 5-bit exponent and full float semantics. Any consumer expecting f16 precision, exponent range, or NaN/Inf handling will get incorrect results | compress.rs | R91 | Open |

### 3b. HIGH Findings

| ID | Description | File(s) | Session | Status |
|----|-------------|---------|---------|--------|
| H1 | **PQ incomplete** — codebook training partial, missing PQ distance computation | ruvector-core simd_intrinsics.rs | C | Open |
| H2 | **ID translation overhead** — u64 internal ↔ string external mapping adds indirection at every operation | ruvector-core | C | Open |
| H3 | **Attention SIMD/Rayon no-ops** — Features declared but zero actual usage across 66 files | ruvector-attention (66 files) | B | Open |
| H4 | **Core HNSW wraps hnsw_rs** — Not novel implementation, but adds real value (SIMD, storage, concurrency) | ruvector-core | C | Open |
| H5 | **FP16 path not SIMD** — matmul.rs FP16 GEMV uses scalar `half` crate, NOT NEON FP16 intrinsics despite comments claiming it | kernels/matmul.rs | R22 | Open |
| H6 | **ANE naming misleading** — Functions named `*_ane` are SCALAR FALLBACKS. BNNS API limitations acknowledged in comments | ane_ops.rs | R35 | Open |
| H7 | **Architecture-complete, persistence-incomplete** — Model backends have correct math but incomplete weight loading. Only Candle backend can load models | Gemma2, Phi3, CoreML, Mistral backends | R35 | Open |
| H8 | **TL1 kernel LUT generation wrong** — Lookup table generation has incorrect mapping but NEVER CALLED in practice | tl1_kernel.rs | R35 | Open |
| H9 | **rlm_embedder HashEmbedder FAKE** — FNV-1a hash, not semantic. NO BitNet integration | rlm_embedder.rs | R35 | Open |
| H10 | **datamap.rs use-after-free risk** — unsafe slice::from_raw_parts with mmap lifetime issues | datamap.rs | R36 | Open |
| H11 | **Postgres healing timeout not enforced** — execute_with_safeguards() does NOT enforce timeout despite comment | healing/engine.rs | R36 | Open |
| H12 | **Postgres healing bgworker registration commented out** — worker.rs register_healing_worker() has bgworker registration COMMENTED OUT | healing/worker.rs | R36 | Open |
| H13 | **prime-radiant SIMD not enabled by default** — wide::f32x8 cfg-gated behind `simd` feature which is not in default features | prime-radiant | R37 | Open |
| H14 | **Training data augmentation simplistic** — tool_dataset and claude_dataset have weak paraphrasing (5 word pairs, literal replacement) | tool_dataset.rs, claude_dataset.rs | R37 | Open |
| H15 | **tile.rs witness hash only processes 6/255 worker reports** — Significant undersampling | tile.rs | R39 | Open |
| H16 | **planner.rs CliffordT overflow** — Silently becomes u64::MAX at t_count>40 | planner.rs | R39 | Open |
| H17 | **Two independent SIMD codebases** — ruvector-core (distance metrics) and edge-net (NN inference). Zero code reuse despite overlapping math (dot product in both) | simd_intrinsics.rs, simd.rs | R52 | Open |
| H18 | **subpolynomial expander splitting incomplete** — check_and_split_expander() has TODO "A full split would require more complex logic". Just marks invalid | subpolynomial/mod.rs | R52 | Open |
| H19 | **index_bench.rs code duplication** — Reimplements HNSW (280 LOC) and IVFFlat (200 LOC) internally instead of importing ruvector-core production code | index_bench.rs | R52 | Open |
| H20 | **subpolynomial fragmentation + witness imports unused** — Two supporting modules imported but never called. Incomplete integration | subpolynomial/mod.rs | R52 | Open |
| H21 | **transpiler.rs "noise-aware" claim FALSE** — Module comment claims noise-aware transpilation but ZERO noise modeling. No error rates, gate fidelities, or noise-adaptive compilation | transpiler.rs | R54 | Open |
| H22 | **TWO independent LoRA implementations** — micro_lora.rs (training, EWC++, sona) and lora.rs (inference, quantization, WASM, edge-net). Zero code sharing despite same algorithm | micro_lora.rs, lora.rs | R54 | Open |
| H23 | **THREE independent noise/noise_model files** — ruQu noise_model.rs (R37), ruqu-core noise.rs (R54), plus surface_code.rs noise handling. No single noise source of truth | noise.rs, noise_model.rs | R54 | Open |
| H24 | **subpoly_decoder.rs zero citations** — Unlike mod.rs (invalid arXiv), subpoly_decoder.rs has NO references, DOIs, or paper citations. Theoretical claims unsupported | subpoly_decoder.rs | R54 | Open |
| H25 | **federation.rs execute_on_cluster always empty.** The scatter-gather framework (FederationStrategy: Parallel/Sequential/Fallback) dispatches tokio::spawn correctly but every work unit returns an empty stub QueryResult. No real cross-cluster data flows | federation.rs | R90 | Open |
| H26 | **gossip.rs SWIM failure detector has no network transport.** join(), send_ping(), handle_ping(), handle_ack() correctly model SWIM with incarnation numbers and suspicion timeouts, but no socket write is ever made. Failure detection cannot fire across processes; protocol only works for in-process state tracking | gossip.rs | R90 | Open |
| H27 | **H1 RESOLVED — PQ now complete in product_quantization.rs.** H1 ("PQ incomplete — codebook training partial, missing PQ distance computation") was recorded against simd_intrinsics.rs. R90 confirms product_quantization.rs (88-92%) has complete k-means++ codebook training, encode(), and ADC LookupTable. The capability exists; it is in a different module than H1 assumed | product_quantization.rs | R90 | Resolved |
| H28 | **kv_cache/legacy.rs per-head scale stomping bug.** In the 4-bit quantization path, the per-head min and max scale values are recomputed and overwritten on every new-token append. Prior scale values are discarded, making it impossible to correctly dequantize previously quantized tokens after even one update. Dequantization of the full KV history is silently corrupted | kv_cache/legacy.rs | R91 | Open |
| H29 | **kv_cache/legacy.rs no eviction policy.** The KV cache grows unboundedly with no LRU, sliding window, or capacity limit. For long-context inference this will exhaust memory without graceful degradation. Contrast with ruvllm/kv_cache.rs which has a hot/cold tiering strategy | kv_cache/legacy.rs | R91 | Open |
| H30 | **mmap.rs pin count allocated but unused — no page eviction mechanism.** PinnedPage and pin_count fields are defined but pin_count is never incremented, checked, or used in any eviction guard. The mmap region is always available for unmapping regardless of whether callers have outstanding references | mmap.rs | R91 | Open |
| H31 | **speculative.rs sequential path verification negates tree parallelism benefit.** EAGLE tree decoding produces a draft tree of candidate continuations intended for parallel verification. This implementation verifies accepted tokens sequentially along the accepted path only. No batch forward pass over tree branches occurs. The novelty is in tree construction (lambda-guided confidence) but the parallel verification that makes EAGLE fast is not implemented | speculative.rs | R91 | Open |
| H32 | **compress.rs PQ codebook uses trivial linear interpolation, not k-means.** ColdCompressor::encode_pq() selects centroids by linear interpolation between min and max values (linspace). This produces uniform quantization, NOT vector quantization optimized to data distribution. Product quantization requires k-means or k-means++ training on representative data. The "PQ" labeling is misleading — this is uniform scalar quantization applied per subvector dimension | compress.rs | R91 | Open |

## 4. Positives Registry

| Description | File(s) | Session |
|-------------|---------|---------|
| **temporal-tensor production-ready** — 95% complete, 125+ tests, 4-tier quantization, CRC32 integrity, SVD reconstruction. BEST crate in repo | temporal-tensor (6 files) | Initial, R22, R37 |
| **ruvector-postgres SIMD** — BEST SIMD in ecosystem. Real AVX-512/AVX2/NEON intrinsics with 4x unrolling, simsimd integration, 23 tests | distance/simd.rs | R22 |
| **ruvllm BitNet backend** — Complete 1-bit LLM inference. MLA 17.8x memory reduction (genuine innovation) | bitnet/backend.rs | R22 |
| **ruvllm kernels** — Production BLAS-level GEMM micro-kernels, textbook Flash Attention 2, Apple Accelerate + Metal GPU integration | kernels/attention.rs, matmul.rs | R22 |
| **ruQu QEC genuine** — Union-Find O(α(n)) decoder, real AVX2 SIMD, complete surface code. Top 15% quality in ecosystem | ruQu (7 files) | R37, R39 |
| **cognitum-gate-kernel exceptional** — 93%, rivals neural-network-implementation. Anytime-valid e-process, custom bump allocator, 64-byte cache-line aligned | cognitum-gate-kernel (5 files) | R36 |
| **prime-radiant coherence substrate** — Sheaf-theoretic memory, BEST sparse matrix (CSR 6 formats), blake3 hash chains | prime-radiant (5 files) | R37 |
| **ruvector-attention real** — 18+ implementations (Flash, Hyperbolic, MoE, Graph, Sheaf, OT, PDE) across 66 files | ruvector-attention | B |
| **sona production-ready** — 85% complete. MicroLoRA + EWC++ + federated learning + SafeTensors export | sona | R13 |
| **ruvector-gnn custom hybrid** — GAT+GRU+edge, full EWC, not a wrapper. ~6,000 LOC | ruvector-gnn | C |
| **postgres extension substantial** — 290+ SQL functions, 3 vector types, rivals pgvector in feature scope | ruvector-postgres | Initial |
| **Raft complete** — Pre-vote, snapshots, dynamic membership, linearizable reads | ruvector-raft | Initial |
| **graph Cypher parser production-quality** — 1,296-line recursive descent, correct lexer | ruvector-graph | C |
| **micro-hnsw-wasm novel** — Genuine `#![no_std]` HNSW in <12KB with neuromorphic extensions | micro-hnsw-wasm | Initial, R36 |
| **hyperbolic HNSW** — Tangent space pruning for Poincare ball geometry | hyperbolic-hnsw | Initial |
| **NAPI bindings well-structured** — Proper async, multi-platform CI, 7+ targets | ruvector-node, gnn-node, sona | Initial |
| **HNSW patches correct algorithm** — Malkov & Yashunin with Rayon parallel insertion, 4 I/O format versions | hnsw_rs fork | R36 |
| **ruvllm memory_pool** — BEST systems code. Lock-free bump allocator, RAII buffer pool, per-thread scratch. 95% real | memory_pool.rs | R34 |
| **postgres SPARQL executor** — COMPLETE SPARQL 1.1 query engine. Property paths, all 7 aggregates, 30+ expression types | sparql_executor.rs | R34 |
| **mincut wrapper** — Genuine bounded-range decomposition from arXiv:2512.13105 (Dec 2024). Among best algorithmic code | wrapper/mod.rs | R34 |
| **edge-net P2P production libp2p** — REVERSES R42's "NO P2P transport". Real Gossipsub/Kademlia/RequestResponse, NOISE encryption, 6 topics, direct RAC integration via broadcast_rac_event() | p2p.rs (edge-net) | R44 |
| **advanced.rs SNN excellence** — 95% real LIF neuron model with STDP Hebbian learning, refractory periods. Highest quality neural component in file | advanced.rs (edge) | R44 |
| **swarm.rs cryptographic envelope** — 88% real Ed25519+AES-256-GCM, nonce replay protection, counter ordering, registry-based identity (never trusts envelope keys) | swarm.rs (edge) | R44 |
| **ruvllm autodetect** — Real hardware feature detection with platform-specific probes. 92% real with 27 tests | autodetect.rs | R34 |
| **ruvllm scheduler** — vLLM-style continuous batching, preemption (recompute+swap), chunked prefill. BEST serving code | scheduler.rs | R35 |
| **ruvllm kernel extensions exceptional** — norm.rs, rope.rs, quantized.rs, activations.rs all 92-95% with real NEON | 4 files | R35 |
| **hnsw_router.rs BEST ruvector-core integration** — Real HnswIndex, HybridRouter blends semantic+keyword | hnsw_router.rs | R37 |
| **micro_lora.rs BEST learning code** — NEON 8x unroll, EWC++ Fisher penalty, <1ms forward. 92-95% real | micro_lora.rs | R37 |
| **sparse.rs** — 95%, 4 sparse formats (CSR/CSC/COO/Graph), no_std compatible. BEST matrix code | sparse.rs | R28 |
| **SPARQL parser production W3C** — 93-95%, 2,496 LOC recursive-descent. All 4 query forms, property paths, aggregates, 33+ built-in functions, UPDATE operations. ruvector has BOTH parser AND executor for SPARQL | sparql/parser.rs | R52 |
| **edge-net SIMD production-quality** — 92-95%, real AVX2/WASM/SSE4.1 intrinsics. Numerically stable softmax (log-sum-exp), Welford layer norm, Q4/Q8 quantization, 19 tests. Independent from ruvector-core SIMD | simd.rs (edge-net) | R52 |
| **hnswio.rs BEST-IN-CLASS persistence** — 95-98%, dual-file format (graph+data), hybrid mmap strategy, 4 backward-compatible versions, zero-copy serialization, concurrent safety | hnswio.rs | R52 |
| **hnsw.rs vendored upstream quality** — 98-100%, complete Malkov & Yashunin 2018. Rayon parallel insert, FilterT filtered search, LayerGenerator exponential sampling. Battle-tested upstream code | hnsw.rs | R52 |
| **ruqu-core noise.rs BEST-IN-CLASS** — 96-98%, production Kraus operator formalism. 4 noise channels (depolarizing, amplitude damping, phase damping, thermal relaxation). Hardware calibration pipeline (T1/T2→γ/λ). 498 test lines. Comparable to Qiskit Aer | noise.rs | R54 |
| **ruqu-core mitigation.rs publication-quality** — 95-98%, three NISQ-era strategies (ZNE, Measurement Error, CDR). Richardson exact extrapolation, tensor-product calibration optimization, Clifford data regression. 40+ tests at 1e-12 precision | mitigation.rs | R54 |
| **ruqu-core transpiler.rs BEST-IN-CLASS** — 95-98%, complete 3-phase quantum circuit transpiler. 3 hardware backends (IBM Eagle, IonQ Aria, Rigetti Aspen), BFS qubit routing, 2-level optimization, 44 tests. Mid-tier Qiskit equivalent | transpiler.rs | R54 |
| **fabric.rs production orchestration** — 93-96%, 256-tile WASM fabric. Blake3 cryptographic audit trails, surface code topology generator, 2μs latency target, 23 tests | fabric.rs | R54 |
| **edge-net federated.rs BEST federated learning** — 95-98%, Byzantine-robust MAD+median, (ε,δ)-DP Gaussian mechanism, TopK compression with error feedback (arXiv:1712.01887), reputation-weighted FedAvg with superlinear weighting, WASM cross-platform. Exceeds sona (85%) | federated.rs | R54 |
| **edge-net lora.rs production edge LoRA** — 90-95%, complete Low-Rank Adaptation with dual SIMD (AVX2+WASM128), Q4/Q8 quantization (4-8× memory reduction), LRU adapter pool with task routing, online gradient accumulation, P2P serialization, 15 tests | lora.rs | R54 |
| **product_quantization.rs genuine PQ** — 88-92%, real k-means++ initialization (D² weighting), Lloyd's algorithm (assignment + centroid update), asymmetric distance computation (ADC) with LookupTable per subspace. RESOLVES H1 | product_quantization.rs | R90 |
| **conformal_prediction.rs valid statistical guarantees** — 88-93%, textbook split-conformal procedure (Vovk et al. 2005). Correct calibration/inference separation, Bonferroni-corrected quantile, 3 nonconformity measures (distance threshold, inverse rank, normalized distance), 7 tests | conformal_prediction.rs | R90 |
| **hypergraph.rs genuine bipartite hypergraph** — 85-90%, correct incidence representation (entity_to_hyperedges + hyperedge_to_entities HashMaps), real k-hop BFS over hyperedge-mediated paths, causal utility function with log1p uplift, temporal expiry. Cites HyperGraphRAG (NeurIPS 2025) | hypergraph.rs | R90 |
| **shard.rs EdgeCutMinimizer multilevel KL** — 70-80%, genuine 3-phase multilevel k-way partitioning: heavy-edge coarsening, greedy initial partition, Kernighan-Lin local search (10 iterations). Real xxh3_64 + blake3 hashing, RangePartitioner with binary search | shard.rs (distributed) | R90 |
| **mmap.rs production memmap2 with lock-free AtomicBitmap** — 88-92%, real memmap2 file-backed memory mapping, Linux madvise(MADV_WILLNEED) prefetch, AtomicBitmap with CAS-based lock-free bit set/clear, RwLock granularity for MmapGradientAccumulator. 17 genuine tests. Clean `#![cfg(not(wasm32))]` gating | mmap.rs | R91 |
| **speculative.rs novel lambda-guided confidence from mincut signal** — 88-92%, EAGLE-style tree speculative decoding with textbook rejection sampling (min(1, target/draft)), adaptive tree width controlled by confidence threshold, correct tree attention mask generation. Unique integration of mincut boundary signal as lambda-guided confidence weight not seen in reference EAGLE implementations | speculative.rs | R91 |
| **rope.rs correct NTK-aware scaling with Q15 quantized path** — 88-92%, faithful implementation of Su et al. 2021 RoPE with CodeLlama/Qwen NTK-aware scaling formula, Q15 fixed-point quantized inference path for edge deployment, 11 substantive tests. No inflated SIMD claims | rope.rs | R91 |
| **kv_cache/legacy.rs RotateKV with Fast Walsh-Hadamard Transform** — 82-88%, genuine RotateKV rotation (IJCAI 2025 paper), correct 2-bit and 4-bit quantization with proper bit-packing, FWHT rotation for key diversity. 15 genuine tests | kv_cache/legacy.rs | R91 |

## 5. Subsystem Sections

### 5a. HNSW Implementations

Three distinct HNSW implementations exist, each serving different use cases (confirmed Phases B+C):

**ruvector-core (Primary)** wraps the third-party `hnsw_rs` crate, NOT a from-scratch implementation. Adds real value: SIMD intrinsics (AVX-512/AVX2/NEON with runtime CPU detection, 1,605 LOC), REDB persistent storage, lock-free concurrency (parking_lot, DashMap, crossbeam). **CRITICAL issues**: placeholder embeddings (sums character bytes, not semantic), HNSW deletions broken (hnsw_rs limitation), ID translation overhead (u64↔string). **PQ NOTE (H1 RESOLVED by R90)**: simd_intrinsics.rs had partial PQ; product_quantization.rs (advanced_features/) has complete k-means++ codebook training + ADC. PQ capability exists; it is in the advanced_features module, not simd_intrinsics.

**micro-hnsw-wasm** is genuinely novel, from-scratch HNSW for ultra-constrained WASM: `#![no_std]`, fixed capacity (32 vectors/core, 16 dims max, 6 neighbors/node), static memory (all in `static mut` arrays, no heap), 256-core sharding (8K total vectors), Quake III fast inverse sqrt, SNN integration (LIF neurons, STDP learning), target <12KB binary. R36 deep-read revealed 6 novel neuromorphic features (spike encoding, homeostatic plasticity, 40Hz resonance, WTA, dendritic computation, temporal patterns) ALL UNVALIDATED with ZERO tests (CRITICAL).

**hyperbolic-hnsw** adapts HNSW for Poincare ball geometry: Mobius addition, exp/log maps, parallel transport, tangent space pruning (cheap Euclidean check, exact Poincare for top-N), per-shard curvature, dual-space index.

**HNSW "patches" (R36, CORRECTED R52)**: R52 line-by-line DEEP read reveals these are **NOT patches** — `scripts/patches/hnsw_rs/` contains a **vendored copy** of upstream hnsw_rs v0.3.3 with ZERO ruvector-specific modifications. Directory naming is misleading. hnsw.rs (98-100%) is complete Malkov & Yashunin with Rayon parallel insert, FilterT search, LayerGenerator exponential sampling — all upstream features. hnswio.rs (95-98%) is BEST-IN-CLASS HNSW persistence: dual-file format (graph+data), hybrid mmap (upper layers in memory, lower mmapped), 4 backward-compatible format versions, zero-copy raw serialization, concurrent safety via unique basename generation. **No postgres or AgentDB integration** — file-based persistence only. **CRITICAL**: libext.rs (75-85%) Julia FFI has no bounds checking on C pointers, std::mem::forget leaks. datamap.rs (85-90%) use-after-free risk with mmap lifetimes. All distance calculations delegated to external `anndists` crate — SIMD is NOT in hnsw.rs itself.

### 5b. Hash-Based Embeddings (Systemic Weakness)

The most pervasive architectural weakness across the entire ruvnet ecosystem. Confirmed in 7+ files across 5 packages in both Rust and JavaScript (C, R13, R22, R37):

| File | Package | Mechanism |
|------|---------|-----------|
| agenticdb.rs | ruvector-core | Sums character bytes of input text (R13, C) |
| hooks.rs | ruvector-cli | Position-based hash (R22) |
| pretrain_pipeline.rs | ruvllm/claude_flow | character sum % dim (R37) |
| real_trainer.rs | ruvllm/training | text_to_embedding_batch deterministic hash (R37) |
| rlm_embedder.rs | ruvllm/bitnet | FNV-1a hash (R35) |
| learning-service.mjs | claude-flow | Math.sin(seed) mock (ecosystem) |
| enhanced-embeddings.ts | agentdb | Math.sin(seed) fallback (ecosystem) |

In practice, all "semantic search" using defaults is character-frequency matching. HNSW indices are structurally valid but search results are meaningless without plugging in a real embedding provider.

### 5c. Attention Mechanisms (Corrected R37)

**Initial assessment was WRONG.** Phase B deep read of Rust source (66 files, ~9,200 LOC across 19 modules) reveals **18+ real implementations** with algorithmic substance. Earlier analysis examined npm-packaged `.js` files only — actual implementations live in `crates/ruvector-attention/src/`.

**Real implementations**: Scaled Dot-Product (standard softmax), Multi-Head, FlashAttention (tiled + online softmax), LinearAttention (FAVOR+), LocalGlobal (Longformer-style), HyperbolicAttention (Poincare + Frechet mean), LorentzCascade (novel), MixedCurvature (E^e × H^h × S^s), EdgeFeatured (GATv2), DualSpace (Euclidean+Hyperbolic), GraphRoPE, MoEAttention (Top-K routing), SlicedWasserstein, CentroidOT, TopologyGated, SheafAttention, DiffusionAttention, NaturalGradient.

**Concerns**: SIMD feature flag is no-op (zero `#[target_feature]` usage in 66 files), Rayon parallelism unused (zero `par_iter()` — multi-head processes heads serially), zero unsafe code (positive for safety but means no hand-tuned SIMD), novel algorithms unvalidated (LorentzCascade, SheafAttention, TopologyGated — no benchmarks vs baselines).

**NAPI bindings** (5 files, ~2,548 LOC): 24 classes, 7 async functions, 9 utilities, 3 enums. Zero unsafe blocks, all errors via `Error::from_reason()`. Async uses `tokio::task::spawn_blocking()` with proper thread safety. Rust native is 10-40x faster than JS fallbacks.

### 5d. Postgres Extension

Substantial PostgreSQL extension rivaling pgvector in feature scope. 290+ SQL functions claimed, 54 verified in operators.rs (R22), 20 module directories. Three vector types (ruvector, halfvec, sparsevec).

**SIMD (95-98%, BEST IN ECOSYSTEM, R22)**: distance/simd.rs has real AVX-512 (16 floats/iter), AVX2 (8 floats/iter with 4x unrolling = 32/iter), ARM NEON (4 floats/iter), simsimd 5.9 integration, runtime feature detection, zero-copy PostgreSQL pointer APIs, 23 test functions. Dimension-specialized dispatch (384/768/1536/3072).

**Index implementations**: hnsw_am.rs (75-80%) has real beam search + greedy descent, insertion logic. **CRITICAL**: `connect_node_to_neighbors()` COMPLETELY EMPTY — graph never actually linked. ivfflat_am.rs (80-85%) has real k-means++ initialization (D² weighting, ChaCha8Rng), Lloyd clustering, adaptive probes. **CRITICAL**: insert, delete, retrain all STUBS.

**SPARQL system (93% weighted, R34+R52)**: COMPLETE SPARQL 1.1 with both parser AND executor. parser.rs (93-95%, 2,496 LOC, R52) is a production W3C SPARQL 1.1 recursive-descent parser: all 4 query forms (SELECT/CONSTRUCT/ASK/DESCRIBE), full UPDATE operations (INSERT/DELETE/LOAD/CLEAR/CREATE/DROP), property paths (sequence/alternative/inverse/transitive), graph patterns (OPTIONAL/UNION/MINUS/GRAPH/FILTER/BIND/VALUES/SERVICE/subqueries), aggregates (COUNT/SUM/AVG/MIN/MAX/GROUP_CONCAT/SAMPLE), 33+ built-in functions (string/numeric/datetime/hash/UUID), proper AST representation (907 LOC ast.rs). Total SPARQL module: 7,421 LOC across 7 files. executor.rs (92%, 1,884 LOC, R34) full algebra execution, property paths (BFS), all 7 aggregates. DELETE is no-op. In-memory TripleStore. **Two SPARQL implementations exist**: ruvector-postgres (this) and rvlite (embedded).

**Benchmark system (42%, R52)**: index_bench.rs is **theatrical benchmarking** — uses production-quality criterion framework (proper warmup, statistical analysis, recall@10, p50/p95/p99 latency percentiles) but measures **wrong implementations**. HNSW search_with_ef() is brute-force O(n) linear scan, NOT real HNSW graph traversal. IVFFlat K-means is genuine. Reimplements HNSW (280 LOC) and IVFFlat (200 LOC) internally instead of importing ruvector-core. Located in ruvector-postgres/benches/ but contains ZERO postgres code. Different category of facade than R43's rustc_benchmarks (15%): not asymptotic deception, but algorithmic mislabeling.

**Healing subsystem (76% weighted, R36)**: Real learning but stub execution. learning.rs (92-95%) BEST — genuine adaptive weight formula, confidence scoring, human feedback. detector.rs (85-90%) 8 problem types but all 8 metric collection methods return empty (CRITICAL). engine.rs (75-80%) cooldown/rate-limiting real but timeout enforcement missing (CRITICAL). strategies.rs (60-65%) StrategyRegistry 95% real but ALL 5 execution methods (reindex, promote, evict, block, repair) are log-only stubs (CRITICAL). worker.rs (70-75%) health check loop works but bgworker registration COMMENTED OUT.

**Verdict**: EXCELLENT read-path foundations but incomplete write paths. SIMD production-ready. Index builds functional. Index searches real. Index mutations 40-60% incomplete. Healing system can learn which strategies work but cannot execute any of them or detect problems.

### 5e. ruvllm LLM Inference

Complete BitNet 1-bit LLM inference backend optimized for Apple Silicon M4 Pro. Three-tier architecture: Metal GPU → Apple Accelerate → NEON SIMD → Scalar fallback. Three deep-read sessions (R22, R34, R35, R37) covering 39 files, ~58K LOC, weighted avg 86% real.

**bitnet/backend.rs (4,559 LOC, 92-95%, R22)**: TL1 ternary lookup tables (2-bit decode), GQA attention (4-wide unrolling), MLA Multi-Head Latent (17.8x memory reduction — genuine innovation, stores latents only), expert predictor (Laplace-smoothed transitions), GGUF model loading, ScratchPool (zero-allocation), AVX2 SIMD dispatch.

**kernels (90.3% weighted, R22+R35)**: attention.rs (88-92%) Flash Attention 2 matching Tri Dao paper, NEON dot product (8x unroll, dual accumulators), PagedKvCache (zero-alloc), GQA parallel (rayon), paged attention, softmax NEON (60% vectorized — exp falls back to scalar). matmul.rs (85-90%) 12x4 GEMM micro-kernel (production BLAS-level), Accelerate integration, Metal GPU offload, 8-accumulator dot for ILP, FP16 path uses scalar `half` crate NOT NEON FP16 (40% real). norm.rs (95%) BEST quality — 4x unrolled FMA, correct variance. rope.rs (95%) real RoPE, NEON interleaved ops, NTK-aware scaling. quantized.rs (92%) real NEON int8/int4/q4k kernels, llama.cpp-compatible. activations.rs (92%) vectorized exp/sigmoid/tanh with polynomial approx. ane_ops.rs (70%) MISLEADING — gelu_ane/silu_ane are SCALAR FALLBACKS, not real ANE ops.

**Infrastructure (92% weighted, R34)**: memory_pool.rs (95%) BEST systems code — lock-free bump allocator (atomic CAS), RAII buffer pool (5 size classes), per-thread scratch with WASM variant, 12 tests. autodetect.rs (92%) real hardware detection (platform, CPU features NEON/AVX, Metal probe), 27 tests, CUDA/WebGPU stub. kv_cache.rs (90%) two-tier KV cache (hot FP16 + cold quantized), real NEON SIMD quantize/dequantize, f32 storage gap (simulated compression), potential deadlock in lock ordering.

**Backends (82% weighted, R35)**: Architecture-complete, persistence-incomplete (SYSTEMIC). All backends have correct math but incomplete weight loading. CoreML (88-92%) real objc2-core-ml bindings, ANE detection, expects pre-converted .mlmodel. Candle (80-85%) ONLY FUNCTIONAL BACKEND — real GGUF + safetensors loading. Mistral (70-75%) real via mistral-rs, X-LoRA manager 90% (learned MLP routing). Gemma2 (88-92%) real soft-capping, alternating local/global attention, from_gguf stub. Phi3 (85-90%) real SuRoPE (128K context), sliding window, from_gguf stub. HybridPipeline (70-75%) generate/stream ALL return NotImplemented.

**Serving (86% weighted, R35)**: scheduler.rs (90-92%) BEST scheduler in ecosystem — vLLM-style continuous batching, preemption (recompute+swap), chunked prefill, priority queues. engine.rs (80-85%) real continuous batching + speculative integration, fallback hash%32000 when no model. paged_attention.rs (75-80%) real page table + block allocator, kernel simplified.

**BitNet extensions (75% weighted, R35)**: expert_cache.rs (88-92%) real LRU/LFU/Adaptive eviction, batch scheduling. tl1_kernel.rs (80-85%) real NEON GEMV (i8→i16→i32 widening), LUT generation wrong but never called. rlm_embedder.rs (75-80%) real recursive refinement, NO BitNet integration, HashEmbedder FAKE (FNV-1a). speculative.rs (55-60%) **CRITICAL perf bug**: 2K sequential forward passes for K tokens vs K vanilla passes — SLOWER than vanilla.

**Training + LoRA (83% weighted, R37)**: micro_lora.rs (92-95%) BEST learning code — NEON SIMD 8x unrolling, EWC++ Fisher penalty, <1ms forward, 18 tests. grpo.rs (90-92%) textbook GRPO (GAE, PPO clipping, adaptive KL), 16 tests. real_trainer.rs (70-75%) triplet loss + InfoNCE, hash-based embeddings (CRITICAL). tool_dataset.rs (88-92%) 140+ templates, 19 categories. claude_dataset.rs (75-80%) 60+ templates, weak augmentation.

**Claude Flow bridge (87% weighted, R37)**: reasoning_bank.rs (92-95%) FOURTH ReasoningBank — real K-means (10 iterations), EWC++ consolidation, 16 tests. hnsw_router.rs (90-93%) BEST ruvector-core integration — HybridRouter blends HNSW semantic + keyword. model_router.rs (88-92%) 7-factor complexity, feedback tracking 1000 predictions. pretrain_pipeline.rs (85-88%) multi-phase pretraining, hash-based embeddings (CRITICAL). claude_integration.rs (70-75%) execute_workflow SIMULATION — hardcoded 500 tokens, no real API (CRITICAL).

**ruvllm coverage after R37**: Total .rs files 338, DEEP 39, LOC read ~58K, weighted avg 86%, files remaining 309 (197K LOC).

### 5f. Temporal Tensor (Production-Ready)

**HIGHEST QUALITY CRATE** — 93% weighted avg, 213 tests total, production-ready. All files ≥88%. Deep-read across R22 and R37.

store.rs (~2,500 LOC, 92-95%, R22) BEST FILE — 74.7KB. Real 4-tier quantization (3-8 bit), CRC32 integrity, SVD frame reconstruction. store_ffi.rs (889 LOC, 90-92%, R37) 11 extern "C" FFI functions for WASM/C, real quantization via crate::quantizer. agentdb.rs (843 LOC, 88-92%, R37) pattern-aware tiering with 4-dim PatternVector, cosine similarity, weighted neighbor voting, 36 tests. quantizer.rs (1,430 LOC, 93-95%, R37) K-means PQ with configurable subvectors, asymmetric distance computation. compressor.rs (1,568 LOC, 95-98%, R37) Delta + run-length + Huffman pipeline, CRC32 integrity. tiering.rs (1,613 LOC, 93-95%, R37) 4-tier storage (Hot→Warm→Cold→Archive) with LRU tracking, promotion/demotion with hysteresis.

### 5g. ruQu + ruqu-core Quantum Computing

**GENUINE QEC + COMPLETE QC PIPELINE** — not a facade. Now 15 files across ruQu + ruqu-core, ~18,500 LOC. Revised weighted avg ~89% (subpoly_decoder drags from 91.3%). Deep-read R37, R39, R54.

**R54 CRITICAL DISCOVERY**: ruQu contains TWO unrelated systems under one crate:
- **QEC system**: decoder.rs, syndrome.rs, surface_code.rs, noise_model.rs, qec_scheduler.rs — genuine quantum error correction
- **Coherence gate system**: filters.rs, fabric.rs, tile.rs, planner.rs — classical statistical decision pipeline for gate quality

These systems have ZERO cross-references. "Qu" may mean "Quality" not "Quantum" for the coherence gate subsystem.

**ruQu QEC (unchanged from R37/R39)**: decoder.rs (2,400 LOC, 95-98%) BEST FILE — Union-Find O(α(n)) + MWPM. syndrome.rs (1,640 LOC, 90-92%) real AVX2 SIMD. surface_code.rs (1,820 LOC, 88-92%) complete surface code. qec_scheduler.rs (1,505 LOC, 88-92%) critical path learning, remote providers stub. noise_model.rs (1,330 LOC, 82-85%) 7 noise channels.

**ruQu Coherence Gate (R39+R54)**: tile.rs (2,125 LOC, 92%) coherence gate architecture, Union-Find, Ed25519, 27 tests. planner.rs (1,478 LOC, 88%) 4 backend cost models, 33 tests. filters.rs (1,357 LOC, 82-86%, R54) MISNAMED — three-filter coherence pipeline (structural min-cut + shift drift + evidence e-value), production statistical methods, 14 tests, ZERO quantum filtering. fabric.rs (1,280 LOC, 93-96%, R54) production 256-tile WASM orchestrator, Blake3 audit trails, surface code topology generator, 23 tests.

**ruqu-core Foundation (R54)**: mitigation.rs (1,276 LOC, 95-98%) THREE NISQ-era strategies — ZNE (Richardson exact extrapolation, polynomial least-squares), Measurement Error (tensor-product calibration, O(n·2^n) scalable inversion), CDR (Clifford data regression). 40+ tests at 1e-12 precision. transpiler.rs (1,211 LOC, 95-98%) BEST-IN-CLASS — complete 3-phase transpiler (decompose/route/optimize), 3 real hardware backends (IBM Eagle, IonQ Aria, Rigetti Aspen), BFS qubit routing with SWAP insertion, 2-level optimization (inverse cancellation + Rz merging), 44 tests. noise.rs (1,175 LOC, 96-98%) BEST-IN-CLASS — production Kraus operator formalism, 4 channels (depolarizing, amplitude damping, phase damping, thermal relaxation), hardware calibration pipeline (T1/T2→γ/λ derivation), confusion matrix inversion for readout, 498 test lines. Comparable to Qiskit Aer. subpoly_decoder.rs (1,208 LOC, 35-40%) **FALSE SUBPOLYNOMIAL** — 3rd instance of false complexity pattern (R39, R52, R54). Claims "provable O(d^{2-ε} polylog d)" but ALL 3 decoders (Hierarchical, Renormalization, SlidingWindow) use O(n²) greedy_pair_and_correct. Zero citations. Implementation is CORRECT but conventional — use decoder.rs's Union-Find instead.

**Combined ruQu+ruqu-core pipeline**: noise.rs (noise models) → mitigation.rs (error mitigation) → transpiler.rs (circuit compilation) → surface_code.rs (QEC layout) → decoder.rs (error correction). This is a **near-complete quantum computing stack** from noise characterization to error-corrected execution.

### 5h. Prime-Radiant & Cognitum-Gate

**prime-radiant (89% weighted, R37)**: Sheaf-theoretic knowledge substrate for AI memory governance. restriction.rs (1,489 LOC, 90-92%) BEST sparse matrix in ecosystem — complete CSR with 6 formats, 4x SIMD unrolling, zero-alloc hot paths. memory_layer.rs (1,260 LOC, 92-95%) triple memory (Agentic/Working/Episodic) with real cosine similarity, genuine temporal/semantic/hierarchical edge creation, 19 tests. witness_log.rs (1,130 LOC, 88-92%) blake3 hash chains with tamper evidence, chain verification (genesis, content hashes, linkage), 16 tests. coherence.rs (1,500 LOC, 88-90%) global/local coherence via sheaf Laplacian, real spectral gap computation. knowledge_graph.rs (1,190 LOC, 85-88%) DashMap concurrent graph, blake3 hashing, topological sort. **Issue**: SIMD not enabled by default — wide::f32x8 cfg-gated behind `simd` feature (HIGH).

**cognitum-gate-kernel (93% weighted, 5 files, 3,504 LOC, R36)**: EXCEPTIONAL CODE — rivals neural-network-implementation as best in ecosystem. 256-tile distributed coherence verification via anytime-valid sequential testing (e-values). lib.rs (713 LOC, 95%) custom bump allocator for no_std WASM, complete tick loop, 6 WASM FFI exports. report.rs (491 LOC, 98%) TileReport exactly 64 bytes with cache-line alignment, compile-time size assertions, correct aggregation. delta.rs (465 LOC, 98%) tagged union 7 operation types, fixed-size FFI-safe layout. shard.rs (983 LOC, 92%) optimal union-find with iterative path compression and union by rank, cache-line alignment for hot fields. evidence.rs (852 LOC, 88%) fixed-point log-space arithmetic with pre-computed thresholds (eliminates libm), genuine sequential testing via e-process.

### 5i. Graph Database

**ruvector-graph (30-35% complete, C)**: Cypher parser is production-quality (1,296-line recursive descent, correct lexer). **CRITICAL**: NO query executor — AST generated but never executed. MVCC incomplete (no conflict detection/GC), ALL optimizations 0% stubs, hybrid features type-defs only. Hyperedge support unique but partial. "Working Cypher queries" claim is FALSE.

**Distributed module (R90, ~2,855 LOC, 5 files) — "transport-absent distributed protocol" (new pattern class):** Every file in ruvector-graph/src/distributed/ shares the same defect: algorithm logic and state machines are correctly designed, but actual network sends are replaced with debug log comments ("In production, send actual network message"). No socket I/O exists anywhere in the module. Quality gradient spans from 15-80%:

- **shard.rs (70-80%, BEST):** Three genuine partitioners — HashPartitioner (xxh3_64 + blake3 dual hashing), RangePartitioner (binary search with dynamic repartitioning), EdgeCutMinimizer (3-phase multilevel k-way: heavy-edge coarsening, greedy initial partition, Kernighan-Lin local search over 10 iterations). GraphShard data container is in-memory DashMap only; no persistence, replication, or split/merge.
- **gossip.rs (45-55%):** Complete SWIM state machine (GossipMessage, MembershipEvent, NodeHealth, incarnation numbers, suspicion timeout). join(), send_ping(), handle_ping(), handle_ack() model SWIM correctly. All send operations are debug logs only. emit_event() never calls registered listeners. Failure detection cannot fire across processes.
- **federation.rs (40-50%):** FederationStrategy dispatch (Parallel/Sequential/Fallback via tokio::spawn) is real. merge_results() performs real node/edge deduplication and stats aggregation. execute_on_cluster() always returns empty QueryResult stub. health_check() hardcodes Healthy. discover_clusters() always returns empty Vec (DNS-SD, Consul, etcd all TODO).
- **coordinator.rs (30-35%):** DashMap concurrency and UUID generation solid. ShardCoordinator fan-out is in-process only (Arc<GraphShard>, no inter-node routing). Query planner is naive string search (contains "match"/"count"/"limit"), not AST-based. 2PC state machine frozen at Active — commit_transaction() removes HashMap entry and logs; no prepare phase, no WAL, no rollback (CRITICAL C26). execute_query() is O(steps × shards) sequential, defeating sharding.
- **rpc.rs (15-20%):** All 4 RPC methods (execute_query, broadcast, health_check, get_shard_info) return hardcoded stubs. RpcServer.start() logs a debug message. GraphRpcService (tonic) feature-gated behind cfg(feature="federation") absent from Cargo.toml defaults — zero gRPC compiles in standard builds (CRITICAL C27). RpcConnectionPool infrastructure (DashMap, get_client()) is correct but connects only to stub clients.

**Contrast**: ruvector-postgres has COMPLETE SPARQL 1.1 system (R34+R52) — BOTH production parser (93-95%, 2,496 LOC) AND executor (92%, 1,884 LOC). Total 7,421 LOC across 7 files. Property paths, all 7 aggregates, full algebra execution, 33+ built-in functions. Cypher has parser only, SPARQL has parser AND executor. ruvector is a **multi-model database** supporting both property graph (Cypher) and RDF triple store (SPARQL) paradigms, though only SPARQL has end-to-end query capability.

### 5j. SONA & Learning

**sona (85%, ~4,500 LOC, R13)**: Production-ready. Complete MicroLoRA (2,211 ops/sec) + EWC++ (online Fisher, adaptive lambda) + ReasoningBank (K-means++) + federated learning + SafeTensors export. Lock-free trajectory recording. ~21 MB memory.

**ruvector-gnn (80%, ~6,000 LOC, C)**: Custom hybrid GNN (GAT+GRU+edge-weighted), not a wrapper. EWC fully implemented. Reads HNSW structure, refines embeddings. 3 unsafe blocks (mmap, properly audited).

### 5k. Subpolynomial Algorithms (R52)

**subpolynomial/mod.rs (45-50%, 1,385 LOC, R52)**: Theatrical claims with partial implementation. Claims to implement a "December 2024 breakthrough" from arXiv:2512.13105, but the arXiv ID format is invalid (2512 = Dec 2025, not 2024). The theoretical foundation is suspect.

**FALSE complexity**: Claims O(n^{o(1)}) subpolynomial update time but implements O(log n) levels × O(recourse). Same pattern as R39's false sublinearity in sublinear-time-solver. Claims "Deterministic" subpolynomial mincut, which is an OPEN PROBLEM in graph algorithms — neither randomization nor subpolynomial bounds are achieved.

**Partial implementation**: Multi-level hierarchy and incremental API (insert_edge/delete_edge) are real. Core algorithmic primitive (expander splitting) has TODO "A full split would require more complex logic". Falls back to full recomputation on deletions (NOT truly incremental). Two supporting modules (fragmentation, witness) imported but NEVER CALLED. 12 tests cover API behavior but not complexity bounds.

**Comparison to R42 dynamic_mincut**: More ambitious claims but less complete implementation. dynamic_mincut EXCEEDS R34 with working algorithms; subpolynomial/mod.rs has grander documentation but incomplete primitives.

### 5l. Edge-Net AI Layer (R52+R54)

**PRODUCTION-GRADE edge computing stack** — SIMD compute (R52) + LoRA inference (R54) + Federated learning (R54). Three files, combined 93-96% weighted avg.

**simd.rs (92-95%, R52)**: Complete SIMD for NN inference (see 5l-simd below).

**lora.rs (90-95%, 1,355 LOC, R54)**: Complete edge LoRA implementation. True low-rank adaptation W' = W + (A·B) * (alpha/rank) with Kaiming init for A, zero-init for B. Dual SIMD targets (AVX2 + WASM128) with automatic detection — same architectural pattern as simd.rs. Q4/Q8 quantization for edge devices (4-8× memory reduction). LRU adapter pool with configurable slots, task-based cosine similarity routing, usage tracking. Online gradient accumulation (SGD update on B matrix). P2P serialization via bincode. 9 WASM exports. 15 tests. **Independent from micro_lora.rs** — this is inference-focused (quantized, WASM, adapter pool) while micro_lora.rs is training-focused (EWC++, federated, sona). ComputeOps trait defined but UNUSED (over-engineering). Task embeddings hardcoded, not learned.

**federated.rs (95-98%, 1,218 LOC, R54)**: **BEST federated learning in entire project** — exceeds sona (85%) by 10-13 points. Five major components: (1) TopK Sparsifier with stateful error feedback (Deep Gradient Compression, arXiv:1712.01887, 90% compression ratio), (2) Byzantine detection via coordinate-wise median + MAD with Z-score threshold (1.4826 scaling), (3) Differential privacy with (ε,δ)-DP Gaussian mechanism (Box-Muller transform, WASM-compatible PRNG), (4) Reputation-weighted FedAvg with superlinear weighting (rep^1.5), (5) Gossipsub gradient sharing protocol with multi-stage validation (model hash, staleness, reputation, magnitude). SGD with momentum for model updates. Cross-platform WASM support. 13 tests. **Missing**: No actual libp2p networking code (architectural separation — networking in p2p.rs per R44). No signature implementation (field exists, unused).

**Edge-net stack verdict**: The complete edge-net AI pipeline (SIMD→LoRA→Federated) is **production-grade** at 93-96% weighted. Combined with p2p.rs (92-95%, R44), this is a real distributed edge AI system — inference, adaptation, and collaborative learning all functional.

### 5l-simd. Edge-Net SIMD Compute (R52)

**simd.rs (92-95%, 1,418 LOC, R52)**: Complete, independent SIMD compute library for neural network inference. **CRITICAL finding: completely independent from ruvector-core**. The ruvector ecosystem has TWO independent SIMD codebases with zero code sharing:
- **ruvector-core**: Distance metrics (L2, cosine, dot product) for HNSW indexing
- **edge-net**: NN layer operations (matmul, activations, normalization, quantization) for inference

Real SIMD intrinsics: AVX2 (`#[target_feature(enable = "avx2")]`), WASM simd128, SSE4.1, with runtime dispatch via `is_x86_feature_detected!()`. Numerically stable: softmax uses log-sum-exp trick (tested with [1000, 1001, 1002]), layer norm uses Welford's algorithm with f64 accumulation. Production Q4/Q8 quantization: block-wise with per-block scales, on-the-fly dequantization in matvec, <15% Q4 error, <2% Q8 error.

Activation functions: GELU (fast tanh via Padé approximation), ReLU (SIMD), SiLU (scalar only — missed optimization). Tiled matrix multiplication with TILE_SIZE=64 but suboptimal B column gathering (strided access, per-iteration Vec allocation). 19 tests validate correctness. CPU-only (no GPU backend integration despite R38 CUDA-WASM findings).

### 5m. Development Methodology

RuVector is **explicitly AI co-authored**. Commits credit "Claude Opus 4.5/4.6". Velocity: 834 commits in 81 days (10.3/day), ~600 LOC/commit, 76 crates (~0.94/day), v0.1.0 "Production Ready" 1 day after repo creation. This is 6-20x faster than sustainable human-only development.

Scope (GNN, quantum, FPGA, distributed consensus, graph DB, 39 attention types, PostgreSQL extension) would typically require 2-3 years for an experienced team. Real achievement is demonstrating human-AI collaboration at scale, not creating battle-tested production system.

**Bulk feature pattern**: Feb 8, 2026 — temporal tensor store ~4,000 lines, 170+ tests. Feb 8 — quantum simulation 306 tests, 11 improvements. Feb 6 — exotic quantum-classical 8 modules, 99 tests.

### 5n. ruvector-core Advanced Features (R90)

**R90 deep-read confirmed a quality gradient within ruvector-core**: core algorithms (HNSW, SIMD) 90-98%, advanced_features/ module 85-93%, advanced/ module 60-90%. Total 4 files, ~2,104 LOC, avg 80-87% real.

**product_quantization.rs (88-92%, 551 LOC)**: Complete Product Quantization implementation resolving H1. k-means++ initialization (distance-weighted D² random sampling), Lloyd's algorithm (assignment step + centroid update), Asymmetric Distance Computation (ADC) with LookupTable (query-to-centroid distances computed once at creation, distance() sums via table lookup). encode() finds nearest centroid via exhaustive scan — O(k × subspace_dim) per subspace, no SIMD acceleration. Minor logic bug in k-means++ fallback (tautological condition line 384, harmless). Test suite covers creation, training, encoding, lookup table accuracy, compression ratio.

**conformal_prediction.rs (88-93%, 505 LOC)**: Valid split-conformal prediction (Vovk et al. 2005). calibrate() computes nonconformity scores from held-out calibration set, compute_threshold() sets (1-alpha) quantile with correct finite-sample Bonferroni-style correction — ceil((1-alpha)*(n+1)/n) formula. Three nonconformity measures: distance threshold, inverse rank (1/(rank+1)), normalized distance with per-query average normalization. predict() implements all three and returns sets of candidates exceeding threshold. adaptive_top_k() delegates to predict().results.len() (pragmatic). 7 tests using mock search functions.

**hypergraph.rs (85-90%, 551 LOC)**: Genuine bipartite hypergraph index. Correct incidence representation via entity_to_hyperedges (HashMap<VectorId, HashSet<String>>) and hyperedge_to_entities (HashMap<String, HashSet<VectorId>>). k_hop_neighbors() implements BFS over hyperedge-mediated paths correctly (node→hyperedge→all other nodes in hyperedge, not pairwise edges). CausalMemory computes utility function U = alpha*similarity + beta*causal_uplift - gamma*latency_penalty, with causal_uplift using log1p of co-occurrence counts (prevents outlier domination). Temporal index with four granularities (hourly/daily/monthly/yearly) via floor division. Cites HyperGraphRAG (NeurIPS 2025). Tests verify 2-hop reachability and causal utility queries.

**tda.rs (60-70%, 497 LOC) — MISLABELED (CRITICAL C25)**: Named "Topological Data Analysis" but implements ZERO canonical TDA algorithms. No Vietoris-Rips complex, no boundary operators, no Betti numbers, no persistence diagrams. Implements: kNN graph construction (all-pairs O(n²) epsilon-neighborhood), connected components (recursive DFS), clustering coefficient (triangle counting via shared neighbors), degeneracy detection (covariance matrix then diagonal-element singular value approximation — NOT a real SVD, invalid for non-axis-aligned manifolds), persistence approximation (component count at 5 fixed scales [0.1, 0.5, 1.0, 2.0, 5.0] — not birth/death pairs). mode_collapse detection (coefficient of variation of pairwise distances) is a reasonable heuristic. This is an embedding quality analyzer, not TDA.

**Quality gradient confirmed (R90)**:
```
ruvector-core algorithms (HNSW, SIMD): 92-98% — production-ready
ruvector-core advanced_features/ (PQ, conformal): 88-93% — production-ready
ruvector-core advanced/ (hypergraph): 85-90% — production-ready
ruvector-core advanced/ (tda.rs mislabeled): 60-70% — functional but misleading
ruvector-graph distributed protocols: 40-55% — correct design, no transport
ruvector-graph distributed transport: 15-20% — stubs only
```

### 5o. ruvector LLM Extensions (R91)

Five files (3,935 LOC) from ruvector that implement LLM inference primitives as standalone modules independent of ruvllm. These are lower-level building blocks (speculative decoding, RoPE, KV cache, mmap, compression) that can be composed into inference pipelines.

**Quality gradient (R91, weighted avg ~83%):**
```
rope.rs:             88-92% — production-ready RoPE (independent from ruvllm/kernels/rope.rs)
speculative.rs:      88-92% — EAGLE tree decoding (sequential verification, not parallel)
mmap.rs:             88-92% — production memmap2 for GNN training (not HNSW)
kv_cache/legacy.rs:  82-88% — RotateKV with scale stomping bug; no eviction
compress.rs:         55-65% — MISLABELED + fake f16; binary quantization correct
```

**mmap.rs (918 LOC, 88-92%)**: Production memmap2 file-backed memory mapping for gradient accumulation in GNN training. AtomicBitmap with CAS-based lock-free bit set/clear operations (dirty page tracking). Linux madvise(MADV_WILLNEED) for prefetching. MmapGradientAccumulator wraps the mmap region with RwLock granularity for concurrent readers. `#![cfg(not(wasm32), feature = "mmap")]` compile-time gating is clean. Seventeen tests cover bitmap ops, mmap creation, and gradient accumulation. **No HNSW integration** — only GNN training callsites. Pin count fields allocated but pin_count never used (no eviction guard, H30).

**compress.rs (679 LOC, 55-65%) — CRITICAL C28+C29**: Named "graph compression" but implements embedding/tensor quantization across 5 temperature tiers. HotCompressor (identity pass-through), WarmCompressor (fake f16: val*1000 → i16, CRITICAL C29), CoolCompressor (Q8 scalar quantization, correct), ColdCompressor (labelled PQ but uses trivial linear interpolation centroids, not k-means, HIGH H32), ArchiveCompressor (binary quantization by sign — correct). PQ4 outlier handling is genuine: z-score detection (> 3σ), sparse storage for outliers vs packed storage for inliers. 12 unit tests, none covering overflow or precision edge cases for fake f16.

**speculative.rs (788 LOC, 88-92%)**: EAGLE-style tree speculative decoding. Builds speculative draft trees with branching factor controlled by lambda-guided confidence (novel: lambda derived from mincut boundary signal, not standard confidence scoring). Standard textbook rejection sampling: accept token i if u ≤ min(1, p_target(x_i) / p_draft(x_i)). Tree attention mask is correctly computed. **Limitation (H31)**: path verification is sequential along the accepted prefix — no batch forward pass over all tree branches simultaneously. The architectural innovation is in tree construction; EAGLE's parallel verification speedup is not realized. Logit-processing layer only — no model weights or KV cache objects embedded; composable with external model runners. 9 genuine tests.

**rope.rs (777 LOC, 88-92%)**: Correct RoPE rotary embeddings per Su et al. 2021. NTK-aware scaling uses the CodeLlama/Qwen formula (scale = (max_seq_len / base_seq_len)^(d/(d-2)) per frequency pair). Partial YaRN: base frequency scaling and frequency bands (ramp_up) implemented; the attention scale factor (√(1 + 0.1·log(scale))) from the YaRN paper is absent. Q15 fixed-point quantized path for edge inference. 11 substantive tests covering rotation correctness, NTK scaling, and quantization round-trips. No false SIMD claims. Independent from ruvllm/kernels/rope.rs (R35, 95%) — that file targets Apple Silicon NEON; this one is platform-agnostic.

**kv_cache/legacy.rs (773 LOC, 82-88%)**: Implements RotateKV rotation (IJCAI 2025) using Fast Walsh-Hadamard Transform for key diversity before caching. 2-bit quantization uses correct bit-packing (4 values per byte). 4-bit quantization uses correct 2-nibble packing. **Scale stomping bug (H28)**: per-head min/max scales recomputed on each append, discarding prior scale; dequantization of full history silently corrupted after first update. No eviction or capacity limit (H29) — unbounded growth. 15 genuine tests, but none cover multi-token dequantization correctness across appends (would expose H28). Labeled "legacy" but no replacement file observed.

**R91 ruvector LLM extensions verdict**: rope.rs and speculative.rs are production-quality standalone modules. mmap.rs is solid infrastructure for GNN. kv_cache/legacy.rs has a correctness bug and lacks eviction. compress.rs is mislabeled, has fake f16, and should not be used for real graph or tensor compression without significant fixes.

## 6. Cross-Domain Dependencies

- **memory-and-learning domain**: ReasoningBank implementations (4 distinct), SONA, EWC++, embeddings, attention mechanisms
- **agentdb-integration domain**: AgentDB controllers, vector-quantization, LearningSystem, AttentionService
- **agentic-flow domain**: ReasoningBank, EmbeddingService, IntelligenceStore, learning-service
- **claude-flow-cli domain**: LocalReasoningBank (only one that runs), ruvector/ modules, model-router, semantic-router
- **sublinear-time-solver domain**: sparse.rs (BEST matrix code), consciousness integration

## 7. Knowledge Gaps

- **76 crates total** — only ~23 crates deep-read across 196 files (R91: +5 files)
- **ruvllm remaining** — 309 files, ~197K LOC unread
- **ruvector-graph distributed** — ADDRESSED by R90 (all 5 distributed files DEEP). Verdict: transport-absent protocol. Remaining gap: MVCC, optimizer, hybrid features.
- **ruvector-core advanced_features/** — ADDRESSED by R90 (PQ + conformal confirmed genuine). Remaining: other files in module.
- **ruvector-core advanced/** — ADDRESSED by R90 (hypergraph + tda.rs). Remaining: other files.
- **ruvector LLM extensions** — PARTIALLY ADDRESSED by R91 (mmap, compress, speculative, rope, kv_cache/legacy). Other files in same cluster (e.g., kv_cache non-legacy, additional compression tiers) unread.
- **npm packages** — 50+ packages, most unread
- **WASM targets** — 15+ WASM crates, only micro-hnsw-wasm deep-read
- **router crate** — 4 crates (router-core, router-cli, router-ffi, router-wasm)
- **delta framework** — 5 crates (delta-core, delta-wasm, delta-index, delta-graph, delta-consensus)
- **cluster/replication** — distributed system crates
- **examples** — 34+ example projects
- **benchmark validation** — no standard ANN-Benchmark results (SIFT1M, GIST1M, Deep1M)

## 8. Session Log

### Initial (2026-02-09): Repository overview + deep dive of 11 crates
11 largest/most complex crates analyzed. Established completeness spectrum from 95% (temporal-tensor) to 30-35% (ruvector-graph). "2 million lines" claim debunked — actual ~400K Rust.

### C (2026-02-14): ruvector-core + ruvector-gnn deep-reads
Phase C Rust source examination. CRITICAL placeholder embeddings discovered. HNSW deletions broken. Real SIMD in simd_intrinsics.rs. ruvector-gnn custom hybrid confirmed.

### B (2026-02-14): ruvector-attention correction
66 files, ~9,200 LOC. Initial 45% completeness was WRONG — 18+ real implementations found. SIMD/Rayon features are no-ops but algorithms are genuine.

### R13 (2026-02-14): SONA + ruvector-core phase
~40 files across SONA, ruvector-gnn, ruvector-core. SONA 85% production-ready. Hash-based embeddings confirmed in Rust source.

### R22 (2026-02-15): ruvllm LLM inference + postgres SIMD/indexes
3 ruvllm files (8,824 LOC), 3 postgres files (6,291 LOC). BitNet backend 92-95%, SIMD 95-98% BEST IN ECOSYSTEM. HNSW neighbor connections EMPTY (CRITICAL). IVFFlat mutations stubbed.

### R28 (2026-02-15): sparse.rs sublinear matrix code
1 file, 964 LOC. 95% real — 4 sparse formats (CSR/CSC/COO/Graph), no_std. BEST matrix code in ecosystem.

### R34 (2026-02-15): ruvllm infrastructure + postgres SPARQL + mincut
10 files, ~8,300 LOC. memory_pool.rs 95% BEST systems code. SPARQL executor COMPLETE (corrects "no executor" verdict for graph domain). mincut wrapper 90% genuine arXiv:2512.13105.

### R35 (2026-02-15): ruvllm backends cluster
23 files, 26,454 LOC, 5-agent swarm. Architecture-complete, persistence-incomplete (SYSTEMIC). Kernel extensions 90.3% exceptional. Speculative decoding SLOWER than vanilla (CRITICAL). Scheduler 90-92% BEST serving code.

### R36 (2026-02-15): HNSW patches + nervous-system + neuro-divergent + cognitum + postgres healing
28 files, 26,569 LOC, 5-agent swarm. cognitum-gate-kernel 93% EXCEPTIONAL. HNSW patches 87% with CRITICAL FFI unsafe + no integrity validation. micro-hnsw-wasm neuromorphic features UNVALIDATED. Postgres healing 76% — real learning, stub execution.

### R37 (2026-02-15): ruvllm claude_flow bridge + training + ruQu + prime-radiant + temporal-tensor
25 files, 30,960 LOC, 5-agent swarm. Fourth ReasoningBank discovered. micro_lora.rs BEST learning code. ruQu QEC 89% genuine. prime-radiant 89% sheaf-theoretic coherence. temporal-tensor updated to 93% production-ready. Hash-based embeddings in Rust training confirmed systemic.

### R39 (2026-02-15): ruQu completion (tile.rs + planner.rs)
2 files, 3,603 LOC. Completes ruQu picture — 91.3% weighted real across all 7 files. HIGHEST QUALITY MULTI-FILE CRATE in ecosystem. Genuine quantum error correction throughout.

### R44 (2026-02-15): Edge-Net P2P Transport
3 files, 3,498 LOC, ~63 findings. **MAJOR REVERSAL**: p2p.rs (92-95%) has production libp2p with Gossipsub/Kademlia/RequestResponse, NOISE encryption, direct RAC integration via broadcast_rac_event() — edge-net IS a distributed system. advanced.rs (72%) is a MISNOMER — zero networking, actually ML primitives (SNN 95%, Raft 85%, HDC 93%, hash embeddings 8th occurrence). swarm.rs (72%) has excellent cryptographic protocol (Ed25519+AES-256-GCM) but 0% GUN network transport (all publish = stubs). Two parallel P2P architectures: edge-net uses libp2p (production), edge uses GUN relays (stubs).

### R54 (2026-02-16): ruQu Quantum Extended + Edge AI
8 files, ~10,080 LOC, ~170 findings. DEEP: 970→978. **Cluster A (ruQu Extended)**: filters.rs (82-86%) MISNAMED — coherence quality gate NOT quantum filtering, three-filter pipeline (structural/shift/evidence), 14 tests, zero connection to decoder.rs. fabric.rs (93-96%) production 256-tile WASM fabric orchestrator, Blake3 audit trails, surface code topology generator, 23 tests. **Cluster B (ruqu-core Extended)**: mitigation.rs (95-98%) three complete NISQ-era strategies (ZNE, Measurement Error, CDR), Richardson exact extrapolation, tensor-product calibration, 40+ tests at 1e-12. transpiler.rs (95-98%) complete 3-phase quantum circuit transpiler, 3 real hardware backends (IBM/IonQ/Rigetti), 44 tests, BEST-IN-CLASS. subpoly_decoder.rs (35-40%) **FALSE SUBPOLYNOMIAL** — 3rd instance. O(n²) greedy under "provable O(d^{2-ε})" claims. Zero citations. noise.rs (96-98%) BEST-IN-CLASS Kraus operator formalism, 4 noise channels + hardware calibration, 498 test lines. **Cluster C (Edge AI)**: lora.rs (90-95%) complete edge LoRA with dual SIMD (AVX2+WASM128), Q4/Q8 quantization, LRU adapter pool, independent from micro_lora.rs. federated.rs (95-98%) BEST federated learning — Byzantine-robust, differential privacy, TopK compression, reputation-weighted FedAvg. **Key findings**: ruQu contains TWO unrelated systems (QEC + coherence gate). Near-complete QC pipeline confirmed (noise→mitigation→transpiler→surface_code→decoder). Edge AI stack production-grade at 93-96% weighted. FALSE subpolynomial pattern now confirmed in 3 files across project.

### R52 (2026-02-16): Algorithmic Infrastructure Deep-Dive
6 files, ~10,271 LOC, 118 findings (26C, 30H, 31M, 31I). DEEP: 955→970. **Cluster A (HNSW patches)**: hnsw.rs (98-100%) is vendored upstream hnsw_rs v0.3.3, NOT a patch — zero ruvector modifications. CORRECTS R36 "fork" assessment. hnswio.rs (95-98%) BEST-IN-CLASS persistence — dual-file format, hybrid mmap, 4 versions, zero-copy. No postgres/AgentDB integration. All SIMD delegated to anndists crate. **Cluster B (Graph query)**: SPARQL parser.rs (93-95%) PRODUCTION W3C SPARQL 1.1 — all 4 query forms, property paths, 33+ functions, proper AST. Total SPARQL module 7,421 LOC. ruvector confirmed as multi-model DB (Cypher parser + SPARQL parser+executor). index_bench.rs (42%) THEATRICAL — HNSW search is brute-force O(n), zero postgres despite location. New facade category: "algorithmic mislabeling". **Cluster C (Subpoly+SIMD)**: subpolynomial/mod.rs (45-50%) FALSE complexity claims — invalid arXiv citation, O(log n) not O(n^{o(1)}), deterministic open problem. Same R39 false sublinearity pattern. simd.rs (92-95%) COMPLETE independent SIMD for NN inference — real AVX2/WASM/SSE4.1, Q4/Q8 quantization, numerically stable. TWO independent SIMD codebases in ruvector (core=distance, edge-net=inference).

### R90 (2026-02-17): ruvector Blind Spot — Distributed Graph + Core Advanced Features
9 files, ~4,959 LOC, 50 findings. DEEP: 1,323→1,332. Two clusters addressed long-standing gaps. **Cluster A (ruvector-graph distributed, 5 files, ~2,855 LOC)**: Establishes new "transport-absent distributed protocol" pattern class — algorithm state machines are correctly designed throughout, but all network sends are replaced with debug log comments. shard.rs (70-80%, BEST) has genuine EdgeCutMinimizer (multilevel Kernighan-Lin) and real xxh3/blake3 hashing. gossip.rs (45-55%) has correct SWIM type system and state tracking but no socket I/O. federation.rs (40-50%) real scatter-gather and merge logic but execute_on_cluster always returns empty. coordinator.rs (30-35%) 2PC state machine frozen at Active, naive string-based query planner. rpc.rs (15-20%) all 4 methods stubs, gRPC feature-gated out of default build. **Cluster B (ruvector-core advanced features, 4 files, ~2,104 LOC)**: product_quantization.rs (88-92%) RESOLVES H1 — complete k-means++ + Lloyd's + ADC LookupTable. conformal_prediction.rs (88-93%) valid split-conformal with Vovk et al. quantile, 3 nonconformity measures. hypergraph.rs (85-90%) genuine bipartite incidence, k-hop BFS, causal utility function, cites HyperGraphRAG (NeurIPS 2025). tda.rs (60-70%) MISLABELED (C25, 11th mislabeled file) — graph metrics masquerading as persistent homology. Quality gradient confirmed: core algorithms 92-98% > advanced_features 88-93% > graph distributed protocols 40-55% > graph transport 15-20%.

### R91 (2026-02-17): ruvector LLM Extensions — mmap, compress, speculative, rope, kv_cache/legacy
5 files, ~3,935 LOC, 9 findings (2C, 5H, positives). DEEP: 1,332→1,337. Cluster covers standalone LLM inference primitives in ruvector (distinct from ruvllm crate). mmap.rs (88-92%) genuine memmap2 + lock-free AtomicBitmap for GNN training; no HNSW integration; pin count unused (no eviction, H30). compress.rs (55-65%) is the 12th MISLABELED FILE (C28) — "graph compression" is embedding/tensor quantization; fake IEEE f16 by ×1000 fixed-point (C29); PQ codebook is trivial linear interpolation not k-means (H32); binary quantization correct. speculative.rs (88-92%) EAGLE-style tree decoding with novel lambda-guided confidence from mincut signal; textbook rejection sampling correct; sequential path verification negates parallel tree benefit (H31). rope.rs (88-92%) faithful RoPE + NTK-aware scaling + partial YaRN; Q15 path; independent from ruvllm/kernels/rope.rs. kv_cache/legacy.rs (82-88%) RotateKV + FWHT + correct 2/4-bit quantization; per-head scale stomping bug corrupts multi-token dequantization (H28); no eviction policy (H29). Weighted avg ~83%. Cumulative mislabeled files: 12.
