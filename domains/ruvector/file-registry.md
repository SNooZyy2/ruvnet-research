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
| scheduler.rs | ruvector-gnn | 532 | 82-88% | DEEP | 5 correct LR algorithms (StepDecay, Exponential, CosineAnnealing SGDR, WarmupLinear, ReduceOnPlateau). Zero GNN integration — no imports from crate. Falls genuine side of bimodal | R94 |
| replay.rs | ruvector-gnn | 503 | 88-92% | DEEP | Correct Vitter/Knuth Algorithm R reservoir sampling, Welford online stats, partial Fisher-Yates. No prioritized replay (uniform only). False KL divergence claim (actually Cohen's d). 12 tests | R94 |
| error.rs | ruvector-gnn | 112 | 95%+ | DEEP | Standard thiserror flat enum (GnnError, 11 variants). #[from] for io::Error and ruvector_core::error::RuvectorError. Mmap variant #[cfg(not(target_arch = "wasm32"))] gated — correct WASM exclusion. Training variant vestigial (GNN inference-only per R99). All variants have named constructors | R101 |
| mmap_fixed.rs | ruvector-gnn | 83 | 90-93% | DEEP | AtomicBitmap GENUINE (lock-free, Acquire/Release ordering, efficient bit iteration via trailing_zeros). MmapManager + MmapGradientAccumulator ABSENT despite module doc advertisement (H86). memmap2/UnsafeCell/RwLock/File all imported but orphaned. "fixed" = planned bugfix never completed | R101 |

### SQL Attention (ruvector-postgres) (R96+R97)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| attention/operators.rs | ruvector-postgres | 426 | 88-92% | DEEP | GENUINE pgx extension — 5 #[pg_extern] functions. CORRECTS R91 AttentionService.ts skepticism. Only 3/10 attention types SQL-dispatchable; 7 fall to ScaledDot default. JsonB matrix boundary. 6 genuine pg_test tests | R96 |
| attention/flash.rs | ruvector-postgres | 388 | 72-78% | DEEP | GENUINE algorithm (KV-tiling + correct online softmax). NOT a pgx function — zero #[pg_extern]. False O(sqrt(N)) space claim (actual O(N)). block_size_q dead_code (query tiling abandoned) | R96 |
| attention/multi_head.rs | ruvector-postgres | 368 | 88-92% | DEEP | Genuine Rayon parallel MHA, delegates to ScaledDotAttention. No SQL generation, no W_Q/W_K/W_V projections (retrieval-focused) | R97 |
| attention/scaled_dot.rs | ruvector-postgres | 308 | 90-93% | DEEP | Correct QK^T/sqrt(d_k), numerically stable softmax, real simsimd SIMD with fallback. Dead dropout field | R97 |
| attention/mod.rs | ruvector-postgres | 291 | 82-87% | DEEP | Orchestration hub with real pgrx PostgresEnum. Inflated "39 mechanisms" claim — only 10 enum variants defined | R97 |
| hyperbolic/poincare.rs | ruvector-postgres | 268 | 88-92% | DEEP | Real pgx Poincare ball math: distance (acosh), Mobius addition, exp_map, log_map. Pure Rust library; SQL via operators.rs. Correct curvature scaling, 13 unit tests. simsimd imported but unused | R98 |
| hyperbolic/lorentz.rs | ruvector-postgres | 258 | 87-92% | DEEP | Correct Lorentz/Minkowski model: inner product, hyperboloid constraint, acosh distance. Bidirectional Poincare↔Lorentz transforms. CRIT: no manifold validation, no pgx annotations (pure Rust). 13 tests with cross-model validation | R98 |
| hyperbolic/mod.rs | ruvector-postgres | 31 | ~92% | DEEP | Clean module root: lorentz, poincare, operators (internal, not pub use-d). DEFAULT_CURVATURE=-1.0, EPSILON=1e-8, MAX_NORM=1.0-1e-5 — correct constants. Completes SQL hyperbolic arc (R98). operators.rs is internal shared math only | R100 |
| hyperbolic/operators.rs | ruvector-postgres | 395 | 92-95% | DEEP | 8 pg_extern SQL functions: poincare_distance, lorentz_distance, mobius_add, exp_map, log_map, poincare_to_lorentz, lorentz_to_poincare, minkowski_dot. All immutable+parallel_safe. 11 pg_tests verifying symmetry, identity, exp/log roundtrip, coordinate conversion. No manifold validation (confirms R98 H70). COMPLETES postgres hyperbolic/ (4/4 DEEP) | R101 |

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
| ruvector_integration.rs | ruvllm | 1,100 | 82-87% | DEEP | **REAL HNSW** (not hash!) via ruvector_core::HnswIndex + ReasoningBank. 6th routing surface (SONA>HNSW>keyword). Parallel to hnsw_router.rs. Two independent UnifiedIndex never synced. response_embedding copy bug | R106 |
| quality/coherence.rs | ruvllm | 836 | 65-72% | DEEP | 19th hash pseudo-embedding. 3-subsystem coherence. 4/6 ContradictionType dead. Silent degradation without external embeddings | R106 |
| quality/diversity.rs | ruvllm | 894 | 82-87% | DEEP | Good lexical diversity (TTR, hapax, n-gram). 19th hash embedding on 60%-weighted semantic path. Mode collapse genuine. 10 tests | R106 |
| quality/validators.rs | ruvllm | 995 | 88-92% | DEEP | Pure JSON schema validation — zero AI/embedding despite quality/ placement. 4 validators + combinator. 10 tests | R106 |
| quality/metrics.rs | ruvllm | 578 | 92-95% | DEEP | Clean 5-dimension quality framework. Test bug: asserts grade B but composite=0.76 (grade C). 11 tests | R106 |
| quality/mod.rs | ruvllm | 110 | 95% | DEEP | Pure re-export of 5 quality submodules. HIGH: docstring example configures semantic_coherence=0.25+diversity=0.20 — toggles degraded hash paths at 45% combined weight, undocumented. **QUALITY MODULE COMPLETE (6/6 DEEP)** | R107 |
| lora/adapter.rs | ruvllm | 726 | 85-88% | DEEP | Management layer over micro_lora.rs. forward_sequential() MATH BUG. TOCTOU in ensure_capacity(). 9 tests | R106 |
| lora/training.rs | ruvllm | 799 | 82-87% | DEEP | Training orchestration. Correct EWC penalty. 7 LR schedules. GradientAccumulator DEAD. Fisher updates never called. 6 tests | R106 |
| lora/mod.rs | ruvllm | 123 | 90% | DEEP | Pure re-export of adapter.rs + training.rs. Propagates R106 forward_sequential() math bug and dead GradientAccumulator through public API surface | R107 |
| lora/adapters/merge.rs | ruvllm | 631 | 72-78% | DEEP | CRITICAL: SLERP=LERP. TaskArithmetic delegates to WeightedSum. DARE seed=42 deterministic. Independent A+B matrix merge wrong for mixed-rank adapters. No LoRA alpha scaling | R107 |
| claude_flow/agent_router.rs | ruvllm | 311 | 72-77% | DEEP | 6th routing surface. CRITICAL: SONA model-index mismatch (agent type 0-7 poisoning model quality-tier 0-2 field). Degenerate response_embedding=query_embedding. Pattern store poisoned | R107 |
| serving/batch.rs | ruvllm | 501 | 90-95% | DEEP | Production continuous batching (vLLM/Orca). merge_prefill_decode() correct. TokenBudget dual-gate. 4 tests | R106 |
| serving/kv_cache_manager.rs | ruvllm | 606 | 88-92% | DEEP | Genuine PagedAttention. **11th parallel subsystem** vs MinCut KV. Deadlock risk. Memory estimate 2x low. 7 tests | R106 |
| serving/request.rs | ruvllm | 473 | 88-92% | DEEP | Genuine vLLM request lifecycle — chunked prefill, KV block allocation, preemption states. HIGH: stop_sequences silently ignored in should_stop(). **SERVING MODULE COMPLETE (6/6 DEEP)** | R107 |
| serving/mod.rs | ruvllm | 348 | 92% | DEEP | Pure orchestration hub. 4 integration tests directly compose ServingEngine+ContinuousBatchScheduler+KvCacheManager+RequestQueue (H in test scope). NoopBackend confirms clean decoupling | R107 |
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
| sparql/ast.rs | ruvector-postgres | 908 | 88-92% | DEEP | Complete SPARQL 1.1 AST: all 4 query forms, 12 GraphPattern variants (BGP/Join/LeftJoin/Union/Filter/Graph/Service/Minus/Exists/Bind/Group/SubSelect), full PropertyPath algebra, 9 Update ops, 7 aggregates. Real parser+executor siblings. DashMap in-memory store, NOT Postgres-backed despite crate name. #![allow(dead_code)] suppression. 4 tests | R111 |
| sparql/triple_store.rs | ruvector-postgres | 740 | 82-87% | DEEP | In-memory DashMap triple store (zero Postgres). 3 of 6 TripleIndex variants implemented. Genuine RDF 1.1: named graphs, lang-tagged literals, typed literals, blank nodes. Real link to ast.rs (Iri+RdfTerm). 7 passing tests. Missing SOP/PSO/OPS indexes causes O(|predicates|) mixed-bound queries. remove() O(|graphs|) | R113 |
| sparql/functions.rs | ruvector-postgres | 704 | 62-68% | DEEP | SPARQL function dispatch. CRIT: DefaultHasher for crypto hashes. CRIT: fn_replace() string not regex. Missing BOUND/IF/COALESCE/isIRI/isBlank. NOW() format broken. RuVector cosine/L2 genuine. 9 tests | R113 |
| sparql/results.rs | ruvector-postgres | 567 | 88-92% | DEEP | 6-format serializer (JSON/XML/CSV/TSV/N-Triples/Turtle). CSV double-escaping bug. CONSTRUCT as SELECT bindings (non-standard). 12 tests | R113 |
| sparql_executor.rs | ruvector-postgres | 1,885 | 92% | DEEP | COMPLETE SPARQL 1.1 query engine. BGP, property paths, 7 aggs | R34 |
| index_bench.rs | ruvector-postgres | 1,395 | 42% | DEEP | THEATRICAL: HNSW search is brute-force O(n). Zero postgres integration despite location | R52 |
| operators.rs | ruvector-postgres | ~1,200 | 85% | DEEP | 54 verified SQL functions | Initial |
| healing/learning.rs | ruvector-postgres | 670 | 92-95% | DEEP | Genuine adaptive weight formula, confidence scoring | R36 |
| healing/detector.rs | ruvector-postgres | 826 | 85-90% | DEEP | 8 problem types. All 8 metric collection methods EMPTY | R36 |
| healing/engine.rs | ruvector-postgres | 789 | 75-80% | DEEP | Cooldown/rate-limiting real. CRIT: no timeout enforcement | R36 |
| healing/strategies.rs | ruvector-postgres | 1,166 | 60-65% | DEEP | StrategyRegistry 95%. ALL 5 executions log-only stubs | R36 |
| healing/functions.rs | ruvector-postgres | 468 | 88-92% | DEEP | 17 pg_extern SQL functions across 5 groups (health status/history/triggers/config/strategy). dry_run genuine. read/write lock inconsistency in set_thresholds (engine.read() for mutation). Empty test placeholder. COMPLETES healing/ (7/7 DEEP) | R101 |
| healing/mod.rs | ruvector-postgres | 234 | 90-93% | DEEP | Module root + orchestration hub. OnceLock global singleton HEALING_ENGINE limits per-connection isolation. 4-stage pipeline (detect→diagnose→repair→learn). OutcomeTracker cloned into RemediationEngine AND kept as self.tracker — diverging state risk. HealingWorkerState initialized but worker not spawned here. 3 tests | R101 |

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
| types.rs | ruQu | 855 | 88-92% | DEEP | **NOT quantum types** — coherence gate control plane. GateDecision (Safe/Cautious/Unsafe→Permit/Defer/Deny), RegionMask 256-bit bitvector (4×u64, PRODUCTION quality), FilterResults (structural+shift+evidence), PermitToken (blake3+ed25519 fields but NO verification — matches R108 C36). StructuralSignal curvature field stored but ignored in time_to_threshold(). 9 tests | R111 |

### ruqu-core Extended

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| mitigation.rs | ruqu-core | 1,276 | 95-98% | DEEP | Three NISQ-era strategies (ZNE, Measurement Error, CDR). Richardson extrapolation, tensor-product calibration. 40+ tests | R54 |
| transpiler.rs | ruqu-core | 1,211 | 95-98% | DEEP | Complete 3-phase circuit transpiler. 3 hardware backends (IBM/IonQ/Rigetti). 44 tests. BEST-IN-CLASS | R54 |
| subpoly_decoder.rs | ruqu-core | 1,208 | 35-40% | DEEP | **FALSE SUBPOLYNOMIAL** (3rd instance). O(n²) greedy under "provable O(d^{2-ε})" claims. Zero citations | R54 |
| noise.rs | ruqu-core | 1,175 | 96-98% | DEEP | Production Kraus operator formalism. 4 noise channels + hardware calibration pipeline. 498 test lines. BEST-IN-CLASS | R54 |

### Prime-Radiant Hyperbolic (R97)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| hyperbolic/mod.rs | prime-radiant | 363 | 88-92% | DEEP | Genuine Poincare geometry (correct formulas), brute-force O(n) search (no real HNSW back-end), disconnected from sheaf substrate, 17 tests | R97 |
| hyperbolic/energy.rs | prime-radiant | 352 | 82-87% | DEEP | Pure data container — all Riemannian math in adapter.rs. merge() has silent curvature compatibility bug. curvature stored but unused in aggregation | R97 |
| hyperbolic/adapter.rs | prime-radiant | 333 | 78-83% | DEEP | Poincare-only (no Lorentz conversion despite name), HNSW is documented brute-force stub, exp_map lacks Mobius addition (approximate geodesic stepping) | R97 |
| hyperbolic/depth.rs | prime-radiant | 215 | 78-82% | DEEP | Correct Poincare depth formula (2*arctanh(|x|)/sqrt(-c)). Curvature stored but NEVER modulates calculations (hardcoded -1.0). Hardcoded level thresholds. Dead weight_multiplier(). 6 tests with curvature=-1.0 only | R98 |
| hyperbolic/config.rs | prime-radiant | 170 | 75-82% | DEEP | Well-structured serde config with curvature, dimension, Frechet, HNSW params. HNSW params are DEAD CONFIG (never used, R97). Euclidean defaults (M=16, ef_construction=200) uncorrected for Poincare geometry. Weak validation | R98 |

### Prime-Radiant Substrate (R111)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| substrate/edge.rs | prime-radiant | 820 | 93-96% | DEEP | **BEST PRIME-RADIANT FILE.** Genuine sheaf constraint edges with rho_source/rho_target restriction maps, residual E_e=w*‖r_e‖², dual API (allocating + zero-alloc EdgeScratch), SIMD chunks_exact(4) auto-vectorization, 18 tests (best coverage in substrate). Only MEDIUM: content_hash() excludes restriction maps | R111 |
| substrate/node.rs | prime-radiant | 588 | 90-93% | DEEP | Genuine sheaf stalk — StateVector f32 with SIMD-hinted arithmetic (4-element chunk accumulator), SheafNode with UUID NodeId + optimistic concurrency version counter, SheafNodeBuilder fluent API. HIGH: content_hash() mixes version (breaks content-addressed caching). DefaultHasher not blake3 (cross-module inconsistency). debug_assert dimension checks compiled out in release. 12 tests | R112 |

### Prime-Radiant Execution (R111)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| execution/gate.rs | prime-radiant | 859 | 88-92% | DEEP | Genuine CoherenceGate — ADR-014 "Gate = refusal mechanism with witness". 4-lane dispatch (Reflex/Retrieval/Heavy/Human). Blake3 witness chain. HIGH: fast-path Reflex skips witnesses (audit gap). PolicyBundleRef.content_hash never verified (matches R108 C36). Parallel WitnessRecord to governance/witness.rs. 9 tests | R111 |
| execution/action.rs | prime-radiant | 600 | 88-92% | DEEP | Genuine governance-gated action boundary: ActionId (UUID), ScopeId (hierarchical path), ActionImpact (4-dim risk scoring), Action trait (scope+impact+execute+content_hash+optional rollback), ActionError (7 variants). BoxedAction erases Output to () (H). ScopeId is_parent_of() prefix bug — "users" matches "users_admin" (H). 5 tests | R112 |
| execution/ladder.rs | prime-radiant | 578 | 88-92% | DEEP | ADR-014 compute escalation ladder: 4 ComputeLane levels (Reflex 1ms / Retrieval 10ms / Heavy 100ms / Human unlimited). Branchless lane_for_energy() via boolean summation (CMOVcc-friendly). LaneThresholds with conservative/aggressive presets. EscalationReason 5-variant union. LaneTransition audit trail. Dead lane_for_energy_lookup() duplicate. 10 tests | R112 |
| execution/executor.rs | prime-radiant | 861 | 90-93% | DEEP | **BEST PRIME-RADIANT** — genuinely composes action+ladder+gate in ADR-014 order. Mandatory witness creation via gate.evaluate_with_witness(). Lane routing, retry with exponential backoff, witness ring buffer. HIGHs: gate() accessor exposes no-witness fast path (audit bypass), Human lane error ambiguity. Witness store in-memory VecDeque 10K cap (no persistence). 8 tests with real CoherenceGate | R113 |

### Prime-Radiant ruvllm Integration (R111)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| ruvllm_integration/coherence_validator.rs | prime-radiant | 1,020 | 82% | DEEP | Real full-stack sheaf pipeline (SheafGraph→CoherenceGate→EnergySnapshot). Blake3 witness audit trail. CRITICAL: sigmoid inversion in compute_confidence() — higher energy → lower confidence (backwards). ValidationAction::execute() is no-op (gate-LLM disconnect). identity_restrictions only. 8 tests | R111 |

### Prime-Radiant Coherence

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| restriction.rs | prime-radiant | 1,489 | 90-92% | DEEP | BEST sparse matrix — CSR 6 formats, 4x SIMD unroll | R37 |
| memory_layer.rs | prime-radiant | 1,260 | 92-95% | DEEP | Triple memory, real cosine similarity. 19 tests | R37 |
| witness_log.rs | prime-radiant | 1,130 | 88-92% | DEEP | blake3 hash chains with tamper evidence. 16 tests | R37 |
| coherence.rs | prime-radiant | 1,500 | 88-90% | DEEP | Sheaf Laplacian, spectral gap computation | R37 |
| knowledge_graph.rs | prime-radiant | 1,190 | 85-88% | DEEP | DashMap concurrent graph, topological sort | R37 |
| coherence/energy.rs | prime-radiant | 760 | 90-93% | DEEP | Sheaf Laplacian E(S)=sum(w_e*|r_e|^2). SIMD via wide::f32x8, zero-alloc hot path, Blake3 fingerprint snapshots. 5 tests | R105 |
| coherence/spectral.rs | prime-radiant | 738 | 85-88% | DEEP | Eigenvalue drift detection via SpectralAnalyzer (EMA, windowed drift_trend). **CRIT: deflation uses λ*I instead of rank-1 Hotelling** — fallback eigenvalues wrong for k>1. 8 tests | R105 |
| coherence/incremental.rs | prime-radiant | 691 | 88-92% | DEEP | O(deg(v)) incremental coherence update with dirty-edge HashSet + adaptive 30% threshold. Slope inversion bug in energy_trend(). 6 tests | R105 |
| coherence/history.rs | prime-radiant | 617 | 88-92% | DEEP | Rolling-window time-series (VecDeque). f64 running-sum accumulators, OLS regression trend, z-score anomaly detection, persistence thresholds. 12 tests | R105 |
| coherence/mod.rs | prime-radiant | 79 | 95%+ | DEEP | Pure re-export facade. 24+ symbols from 5 submodules (engine, energy, history, incremental, spectral). ResidualCache→IncrementalCache compat alias. **COHERENCE MODULE COMPLETE (5/5 DEEP)** | R105 |
### Prime-Radiant Governance (R107+R108)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|--------|
| governance/repository.rs | prime-radiant | 1,062 | 62% | DEEP | LOWEST quality — traits only, InMemory stubs, no PostgreSQL/SQLite/ruvector backends. "Async-First" claim FALSE (all methods sync). verify_chain() calls methods that don't exist on trait (H from R107). **12th persistence stub** | R107 |
| governance/policy.rs | prime-radiant | 970 | 82-87% | DEEP | 2 CRITICAL: signatures never verified (zero-byte placeholder activates bundles), placeholder() production-reachable. Escalation DSL declared but NO evaluator. State machine + blake3 hashing genuine | R108 |
| governance/lineage.rs | prime-radiant | 873 | 88-93% | DEEP | Genuine Blake3 hash chain on every record. DAG provenance with WitnessId attestation. 9 substantive tests. No cycle detection — deferred to repository.rs Kahn sort | R108 |
| governance/witness.rs | prime-radiant | 723 | 88-94% | DEEP | Genuine Blake3 attestation chain. 10+ fields hashed per record. tamper_detection test confirms. Single-witness only (no quorum). 4-lane compute enum (ADR-014) | R108 |
| governance/mod.rs | prime-radiant | 439 | 93-96% | DEEP | Clean re-export + Hash/Timestamp/Version shared types. No orchestrator — composition at crate level. **GOVERNANCE MODULE COMPLETE (5/5 DEEP incl. repository.rs R107)** | R108 |

### Prime-Radiant Storage (R107+R108)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|--------|
| storage/postgres.rs | prime-radiant | 1,082 | 82-87% | DEEP | REAL async sqlx/PgPool. Zero sheaf integration. Brute-force similarity (no pgvector). Witness race condition in store_witness(). AsyncGraphStorageAdapter does NOT implement GraphStorage/GovernanceStorage traits | R107 |
| storage/file.rs | prime-radiant | 804 | 85-90% | DEEP | CRITICAL: WAL commit flag never set — deletions non-durable across restarts. Dual-format (bincode/JSON), blake3 WAL integrity, parking_lot concurrency | R108 |
| storage/memory.rs | prime-radiant | 731 | 88-92% | DEEP | Genuine volatile backend. parking_lot::RwLock on all 8 fields. HIGH: witnesses_by_action never populated (trait design gap). 9 tests. IndexedInMemoryStorage adds tag+name indexes | R108 |
| storage/mod.rs | prime-radiant | 576 | 82-86% | DEEP | GraphStorage + GovernanceStorage traits genuine. HybridStorage wraps FileStorage ONLY — postgres never wired despite StorageConfig carrying postgres_url. **STORAGE MODULE COMPLETE (4/4 DEEP incl. postgres.rs R107)** | R108 |

### Prime-Radiant Cohomology (R109+R110) — MODULE COMPLETE (9/9 DEEP, ~83%)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|--------|
| cohomology/neural.rs | prime-radiant | 650 | 88-93% | DEEP | SheafNeuralLayer (Xavier init, Laplacian diffusion, residual, LayerNorm), SheafConvolution (GraphSAGE-style, identity restriction maps only), CohomologyPooling (5 methods). HIGH: restriction maps never applied — GCN not true sheaf conv. spectral_weighting declared but dead. Dropout silently absent. Laplacian recomputed every diffusion step. 4 tests | R109 |
| cohomology/cohomology_group.rs | prime-radiant | 601 | 85-90% | DEEP | GENUINE H^n(K,F) computation: coboundary as boundary transpose, RREF-based kernel, Gram-Schmidt quotient space. R-coefficient only (no Smith Normal Form for integer cohomology). `use_sparse` config is FACADE (no sparse path). scale(1.0) dead call in normalize(). ndarray real. 5 canonical tests (point, edge, circle H^1=R, filled triangle). Betti numbers correct | R109 |
| cohomology/simplex.rs | prime-radiant | 598 | 90-94% | DEEP | Foundational algebraic topology: Simplex (BTreeSet canonical ordering, boundary operator), SimplicialComplex (Bron-Kerbosch clique enumeration with pivot), Chain (R-coefficient formal sum), Cochain (inner product). HIGH: naive pivot selection — exponential for dense graphs. DefaultHasher SimplexId collision risk. No d^2=0 verification. 5 tests | R109 |
| cohomology/laplacian.rs | prime-radiant | 545 | 65-70% | DEEP | 2 CRITICAL: (1) power_iteration uniform init converges to eigenvalue 0 trivially; (2) finds LARGEST eigenvalues but Laplacian needs SMALLEST (Fiedler). HIGH: build_matrix ignores restriction maps — non-trivial sheaves reduced to graph Laplacian. Same deflation bug class as coherence/spectral.rs. energy() correct. Tests pass accidentally | R109 |
| cohomology/obstruction.rs | prime-radiant | 525 | 85-90% | DEEP | Genuine H^1 obstruction detection: sheaf Laplacian energy, per-edge hotspot attribution, 5-level severity, MinCut-aware remediation. HIGH: betti_numbers always single-element. Only 1 Obstruction per detect(). compute_cocycles dead. 4 tests | R109 |
| cohomology/cocycle.rs | prime-radiant | 471 | ~80% | DEEP | Scalar coboundary apply() correct (signed simplicial). CRITICAL: apply_adjoint() dimension bug (iterates n-simplices not n+1) — 2nd broken Laplacian. is_coboundary() always false for n≥1 (inflates all H^n). Cochain add() truncates near-cancellation (1e-10). SheafCoboundary edge indexing fragile. 16 findings | R110 |
| cohomology/sheaf.rs | prime-radiant | 464 | 72% | DEEP | Genuine sheaf structure (Stalk, LocalSection, SheafBuilder). CRITICAL: gluing axiom semantically broken (no shared edge stalk). CRITICAL: architecturally ISOLATED — zero consumers in cohomology pipeline (SheafLaplacian/CohomologyGroup/SheafNeuralLayer build own structures). is_global dead state. from_graph() bidirectional maps mathematically unrelated. 14 findings | R110 |
| cohomology/diffusion.rs | prime-radiant | 486 | 78-82% | DEEP | SheafDiffusion heat kernel on complexes. CRITICAL: update_section() clamping breaks gradient flow monotonicity. CRITICAL: diffuse_adaptive() reports converged=true at f64::MAX on full divergence. Laplacian eigensolver bugs NOT inherited (uses apply()/energy() not compute_spectrum()). No CFL stability check. Residual obstruction is heuristic not cohomological. 10 findings | R110 |
| cohomology/mod.rs | prime-radiant | 61 | 95% | DEEP | Clean module root: all 8 submodules declared and re-exported (35 types). Dual obstruction pathways (Laplacian+diffusion). Precise mathematical docs with coboundary formula and 3 academic refs. Known systemic risk: HarmonicRepresentative propagates laplacian.rs eigensolver bugs | R110 |

### ruvector-attention Training (R109)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|--------|
| training/optimizer.rs | ruvector-attention | 401 | 88-92% | DEEP | SGD/Adam/AdamW + LR scheduler (warmup + cosine decay). Nesterov formula double-counts momentum. Adam weight_decay field dead. AdamW duplicates Adam inner loop. All scalar. No autograd, no checkpoint. 6 tests | R109 |
| training/loss.rs | ruvector-attention | 360 | 88-92% | DEEP | InfoNCE (stable log-sum-exp, analytic gradient), LocalContrastiveLoss (triplet margin), SpectralRegularization (misnamed: VICReg-style, NOT spectral norm). HIGH: SpectralReg gradient returns ZERO vector. Two divergent code paths for InfoNCE. Send+Sync | R109 |
| training/curriculum.rs | ruvector-attention | 357 | 90-95% | DEEP | CurriculumScheduler (4-stage easy→hard). TemperatureAnnealing (Linear/Exponential/Cosine SGDR/Step). HIGH: Exponential decay NaN on final_temp=0. Clean self-contained. 6 tests | R109 |
| training/mining.rs | ruvector-attention | 352 | 88-92% | DEEP | 4 mining strategies (Random/Hard/SemiHard/DistanceWeighted) + InBatchMiner. HIGH: fixed seed=42 makes random deterministic. SemiHard silently degrades. Mixed cosine/euclidean metrics. 5 tests | R109 |

### ruvector-attention Transport (R110) — MODULE EFFECTIVELY COMPLETE (3/4 DEEP, ~83%)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|--------|
| transport/sliced_wasserstein.rs | ruvector-attention | 444 | 72% | DEEP | CRITICAL: core compute_1d_ot_distance() is projected-L2/L1 NOT Sliced Wasserstein — computes pointwise |q_p-k_p|^p. Correct distributional OT (compute_distributional_ot) EXISTS but is dead code. Histogram/CDF infrastructure from cached_projections.rs built but never consumed. Two-stage prefilter has metric mismatch (dot-product selection → OT scoring). Zero training pipeline integration. 10 findings | R110 |
| transport/centroid_ot.rs | ruvector-attention | 444 | 88-93% | DEEP | Genuine k-means clustering + softmax attention. Sinkhorn (namesake algorithm) is dead code with logic bug (log_u never updated in iteration). O(n²) cluster-size recount from discarded counts. 4-wide loop-unrolled squared_distance. Full Attention trait. Deterministic seed=42. 3 substantive tests. 8 findings | R110 |
| transport/cached_projections.rs | ruvector-attention | 242 | 88-92% | DEEP | ProjectionCache (P random unit-directions, seed-reproducible) + WindowCache (pre-sorted projections per window). dot_product_simd is scalar 4-way unrolling (no intrinsics despite name). Uniform-hypercube sampling (not Marsaglia — minor directional bias). Histogram CDFs genuine approximation speedup. Clone+Send, caller-responsible thread safety. 8 findings | R110 |

### ruvllm LoRA Adapters (R108)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|--------|
| lora/adapters/trainer.rs | ruvllm | 593 | 72-78% | DEEP | CRITICAL: validation loss is fake (precomputed quality, not model forward pass). 3-layer LoRA arch real but broken at seams (EWC per-epoch misuse). SyntheticDataGenerator semantically meaningless | R108 |
| lora/adapters/mod.rs | ruvllm | 508 | 88-92% | DEEP | Genuine 5-preset config registry (coder/researcher/security/architect/reviewer). Square matrix assumption wrong for real transformers. Parallel abstraction to parent adapter.rs (zero cross-reference) | R108 |


### Cognitum Gate & Other Specialized

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| lib.rs | cognitum-gate-kernel | 713 | 95% | DEEP | Custom bump allocator, no_std WASM. 6 FFI exports | R36 |
| report.rs | cognitum-gate-kernel | 491 | 98% | DEEP | 64-byte cache-line aligned, compile-time assertions | R36 |
| delta.rs | cognitum-gate-kernel | 465 | 98% | DEEP | Tagged union 7 operation types, fixed-size FFI-safe | R36 |
| shard.rs | cognitum-gate-kernel | 983 | 92% | DEEP | Optimal union-find, iterative path compression | R36 |
| evidence.rs | cognitum-gate-kernel | 852 | 88% | DEEP | Fixed-point log-space, anytime-valid e-process | R36 |
| decision.rs | cognitum-gate-tilezero | 539 | 88-93% | DEEP | Genuine ThreeFilterDecision for TileZero 256-tile WASM arbiter (structural/shift/evidence). Pre-computed reciprocals, #[inline(always)]. CRIT: zero-then-sign PermitToken panic path. 12th parallel subsystem (vs prime-radiant gate). History/baseline dead code. 8 tests | R113 |
| tools.rs | mcp-gate | 458 | 88-92% | DEEP | Thin 3-tool MCP adapter wrapping cognitum-gate-tilezero (permit_action/get_receipt/replay_decision). NOT a 7th MCP protocol. PermitToken never verified (systemic crypto pattern). replay_decision genuinely correct. 4 async tests | R113 |
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

### Hyperbolic HNSW & Attention (R92)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| hnsw.rs | ruvector-hyperbolic-hnsw | 651 | 88-93% | DEEP | NATIVE HNSW (not hnsw_rs). Poincaré ball distance, tangent-space pruning, DualSpaceIndex RRF | R92 |
| poincare.rs | ruvector-hyperbolic-hnsw | 628 | 90-95% | DEEP | ALL formulas correct (Ganea 2018). Möbius ops, exp/log maps, Fréchet mean. Comprehensive stability | R92 |
| lorentz_cascade.rs | ruvector-attention | 580 | 90-93% | DEEP | Genuine Lorentz model attention. Minkowski metric, Busemann functions, multi-curvature cascade, Einstein midpoint | R92 |
| lib.rs | ruvector-hyperbolic-hnsw-wasm | 633 | 88-92% | DEEP | **17th GENUINE WASM**. Production wasm_bindgen, 7 real math ops, ShardedIndex | R92 |
| shard.rs | ruvector-hyperbolic-hnsw | 576 | 82-85% | DEEP | Hyperbolic-aware radius partitioning. Canary deployment. Transport-absent (confirms R90) | R92 |
| tangent.rs | ruvector-hyperbolic-hnsw | 349 | 88-92% | DEEP | Two-phase Poincare pruning: Euclidean filter in tangent space → exact Poincare re-rank. TangentCache Fréchet centroid. O(N) linear scan by design (receives HNSW candidates from caller). Dead import: norm_squared. 2 genuine tests. Completes crate | R99 |

### GNN Bindings (napi-rs & WASM)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| lib.rs | ruvector-gnn-node | 428 | 88-92% | DEEP | Real napi-rs #[napi] macros. Inference-only FFI: forward/compress/decompress/differentiable_search/hierarchical_forward. Solid error handling (Status::InvalidArg, .map_err). hierarchical_forward() deserializes layer configs from JSON strings per-call (serde overhead). 5-tier compression enum | R99 |
| lib.rs | ruvector-gnn-wasm | 415 | 90-94% | DEEP | **18th GENUINE WASM**. All exports delegate to real ruvector_gnn core via serde_wasm_bindgen. Inline cosineSimilarity() correct (95%). console_error_panic_hook initialized. Same inference-only surface as gnn-node | R99 |

### Attention — Sheaf (R105+R107)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| sheaf/attention.rs | ruvector-attention | 712 | 82-87% | DEEP | Genuine sheaf attention A_ij=exp(-β*E_ij)/Z. RestrictionMap replaces QKV. HIGH: no backward pass, fixed seed=42, multi-head structurally single-head (1 set of restriction maps). 11 tests | R105 |
| sheaf/early_exit.rs | ruvector-attention | 651 | 90-94% | DEEP | Energy-convergence early exit (not confidence-threshold). Patience counter, EMA smoothing, PerfectCoherence shortcut. 13 tests — highest density in sheaf module | R105 |
| sheaf/sparse.rs | ruvector-attention | 712 | 88-92% | DEEP | **GENUINELY NOVEL**: residual-sparse attention uses sheaf restriction map energy as sparsity criterion. CSR present but unreachable from compute path. No SIMD. 10 tests | R105 |
| sheaf/restriction.rs | ruvector-attention | 430 | ~85% | DEEP | ρ(x)=Ax+b restriction maps. Xavier-scaled init with fixed seed=42. identity(), from_weights(), residual(), energy(), energy_matrix(), apply_batch(). 11 tests | R105 |
| sheaf/router.rs | ruvector-attention | 666 | 85-90% | DEEP | Genuine SheafAttention composition as token router. theta_deep config field dead (comment: "if needed"). confidence=1.0 hardcoded for all decisions. tune_thresholds() structurally decoupled — no internal feedback loop. SONA adaptive interface passive | R107 |

### Attention — Hyperbolic (R100)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| hyperbolic_attention.rs | ruvector-attention | 172 | 88-92% | DEEP | Fully working Poincare-ball attention, Fréchet mean aggregation. HIGH: adaptive_curvature is dead field — fixed at construction, never updated despite config flag. Unconditional project_to_ball on every compute() wastes cycles | R100 |
| mixed_curvature.rs | ruvector-attention | 241 | 90-94% | DEEP | MOST ARCHITECTURALLY NOVEL — splits embeddings into [euclidean \|\| hyperbolic] halves, independent attention, blend. HIGH: linear blend of two softmax distributions is ad-hoc (not proper mixture model); renormalization rescues correctness | R100 |
| poincare.rs | ruvector-attention | 181 | 93-96% | DEEP | Shared mathematical foundation for entire hyperbolic cluster. All Gyrovector operations correct: poincare_distance, mobius_add, mobius_scalar_mult, exp_map, log_map. EPS clamping, acosh clamping, projection after every update. MEDIUM: frechet_mean fixed learning rate (0.1) may oscillate at high curvature | R100 |
| mod.rs (attention hyperbolic) | ruvector-attention | 26 | 100% | DEEP | Pure wiring — re-exports hyperbolic_attention, lorentz_cascade, mixed_curvature, poincare. Flat pub re-export confirms lorentz_cascade (R92) is first-class alongside Poincare | R100 |

### Hyperbolic HNSW — Library Entry Point (R100)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| lib.rs | ruvector-hyperbolic-hnsw | 211 | 90-93% | DEEP | Crate entry point exposing 5 submodules: error, hnsw, poincare, shard, tangent. Tangent-space pruning: log_c(x) at shard centroid, cheap Euclidean prune before exact Poincare. Per-shard curvature (ShardedHyperbolicHnsw). DualSpaceIndex: synchronized Euclidean fallback for ranking fusion near ball boundary. HIGH: no bounds check on shard index in tangent pruning; DualSpaceIndex sync overhead not benchmarked | R100 |
| error.rs | ruvector-hyperbolic-hnsw | 43 | 95%+ | DEEP | 8-variant HyperbolicError enum via thiserror. OutsideBall carries norm + curvature for programmatic recovery. HyperbolicResult type alias exported. Clone-derived for retry/backoff. Clean: no std::io::Error wrapping | R100 |

### Postgres GNN (ruvector-postgres) (R101)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| gnn/operators.rs | ruvector-postgres | 426 | 82-87% | DEEP | 4 genuine SQL operators + 1 FACADE (ruvector_message_pass returns format string, H90). Self-contained — imports from super::gcn, super::graphsage, super::aggregators, NOT ruvector-gnn crate (H91). Deterministic weights. 8 pg_test cases | R101 |
| gnn/gcn.rs | ruvector-postgres | 224 | 82-88% | DEEP | Independent Kipf & Welling GCN — Xavier init, 1/sqrt(degree) normalization, ReLU. Missing self-loops: uses A not A+I (pure neighbor aggregation, not canonical GCN, H93). Zero SQL — purely in-memory Vec<Vec<f32>> (H92). Rayon par_iter. Deterministic pseudo-random weights (H94). 6 tests with hand-computed expected values | R101 |
| gnn/message_passing.rs | ruvector-postgres | 234 | 88-92% | DEEP | Core MessagePassing trait (message/aggregate/update). Genuine build_adjacency_list (inbound adjacency), propagate() with rayon par_iter, propagate_weighted() with per-edge f32 weights. Zero Postgres-specific code — pure Rust graph algorithm. Only SUM aggregation. 3 tests | R101 |

### AIDefence Security (R92)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| AIDefenceGuard.ts | ruvbot (npm) | 763 | 82-88% | DEEP | PARTIALLY REAL — 28 genuine regex patterns, zero ML, behavioral stub, policy ghost | R92 |
| aidefence-guard.test.ts | ruvbot (npm) | 280 | — | DEEP | Tests MOCKED — exercise regex only, behavioral analysis untested, PII false positives unverified | R92 |
| aidefence-integration.ts | agentdb | 166 | — | DEEP | SIMULATION-ONLY — hardcoded threat data, correct AgentDB pattern, no real AIDefence | R92 |

### CUDA-WASM Flash Attention (R92)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| flash_attention.rs | cuda-wasm (ruv-FANN) | 528 | 88-92% | DEEP | MISPLACED — CPU Flash Attention v2 reference, zero CUDA/GPU code. Textbook online softmax + causal masking. 7 tests | R92 |

### MinCut-Gated-Transformer Core (R93)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| attention/spike_driven.rs | ruvector-mincut-gated-transformer | 585 | 88-93% | DEEP | GENUINE neuromorphic LIF + STDP coincidence attention. saturating_mul violates "multiplication-free" claim. Zero energy_gate.rs integration | R93 |
| sparse_attention.rs | ruvector-mincut-gated-transformer | 571 | 72-80% | DEEP | Novel mincut-aware design thesis, but partition boundaries are PLACEHOLDER (uniform stride). _target_density and _gate dead params. BTreeSet dedup correct | R93 |
| q15.rs | ruvector-mincut-gated-transformer | 634 | 88-92% | DEEP | Correct Q15 fixed-point (u16). CRITICAL: u16/i16 type mismatch with rope.rs. Newtype exported but unused within crate. 11 tests | R93 |
| kernel/qgemm.rs | ruvector-mincut-gated-transformer | 621 | 88-93% | DEEP | GENUINE quantized GEMM. Real AVX2+NEON widening multiply-accumulate. Correct asymmetric quantization. Compile-time-only SIMD dispatch (no runtime detection) | R93 |
| kv_cache/manager.rs | ruvector-mincut-gated-transformer | 596 | 72-78% | DEEP | 3-tier AdaptiveKVCache (Hot/Warm/Archive). Pure FIFO eviction, no H2O/StreamingLLM. SQuat/KVQuant quantizers dead code. Rematerialization wired but never triggered | R93 |
| ffn.rs | ruvector-mincut-gated-transformer | 628 | 88-92% | DEEP | Genuine INT8 FFN with real SIMD GELU (AVX2 Padé tanh + NEON Newton-Raphson). Vanilla GELU, not SwiGLU/GeGLU. Allocation-free forward. No MoE | R93 |
| kv_cache/kvquant.rs | ruvector-mincut-gated-transformer | 565 | 78-85% | DEEP | Genuine pre-RoPE KVQuant (Hooper et al. 2024). 3-bit packing silently broken (falls back to 4-bit). O(n²) outlier scan. calibrate() per-layer API lies (flattens to single pair) | R93 |
| energy_gate.rs | ruvector-mincut-gated-transformer | 560 | 88-93% | DEEP | **R34 "MOST NOVEL" CONFIRMED.** Genuine 3-component EBT: sigmoid lambda + boundary penalty + partition entropy. Central-difference gradient. System-2 gradient descent refinement. Hybrid GatePolicy fallback | R93 |
| early_exit.rs | ruvector-mincut-gated-transformer | 661 | 88-92% | DEEP | Novel lambda-stability-driven early exit (LayerSkip alternative). 40/40/20 confidence weighting. abs() overflow risk. Unnormalized logit verification. Adaptive/config threshold inconsistency | R93 |
| arena.rs | ruvector-mincut-gated-transformer | 472 | 92% | DEEP | GENUINE bump allocator with 64B cache-line alignment. QKV+FFN weight layout in calculate_arena_size. Aliasing UB hazard (multiple &mut slices from same Vec). calculate_arena_size ignores _heads. 10 tests | R96 |
| kv_cache/squat.rs | ruvector-mincut-gated-transformer | 467 | 78-84% | DEEP | GENUINE SQuAt 2024 paper. Hadamard basis + Gram-Schmidt correct. 2/4-bit packing complete. CRIT: calibrate() STUB — ignores calibration_data, no data-driven basis learning | R96 |
| kv_cache/kivi.rs | ruvector-mincut-gated-transformer | 458 | 72-78% | DEEP | GENUINE FWHT (butterfly recursion). Per-channel quantization claim FALSE — global min/max regardless of QuantScheme. PerGroup stub. SIMD dequantize TODO. Low-level primitive for quantized_store.rs | R96 |
| kv_cache/policy.rs | ruvector-mincut-gated-transformer | 440 | 82-87% | DEEP | Age-based tier graduation only. NO H2O/attention-sink/LRU. Three-struct architecture genuine. Evaluate/tracker pressure divergence. Cost model inversion. Zero hot_buffer.rs integration | R96 |
| kv_cache/hot_buffer.rs | ruvector-mincut-gated-transformer | 419 | 82-87% | DEEP | FIFO ring buffer for "FP16" (actually f32 — test acknowledges). pop_oldest() BUG (no read cursor for sequential pops). Dual push API trap (push vs push_head+advance). Zero policy.rs integration | R96 |
| trace.rs | ruvector-mincut-gated-transformer | 413 | 88-92% | DEEP | Gate-decision diagnostics with stack-allocated [T;64] circular buffer. Records GateDecision/lambda/tier from packets.rs. Feature gate claim documentation-only — compiles unconditionally. 6 genuine tests | R96 |
| packets.rs | ruvector-mincut-gated-transformer | 492 | 90-95% | DEEP | Pure type-definition coherence interface. GatePacket/GateDecision/SpikePacket/InferInput/Witness. Novel QuarantineUpdates isolation mode. Consistent Q15. repr(C) throughout | R94 |
| state.rs | ruvector-mincut-gated-transformer | 500 | 88-92% | DEEP | Zero-alloc buffer layout (64-byte aligned). Inference-only — no checkpoint/safetensors. KV ring buffer correct. Unchecked layer index UB risk. No version tagging | R94 |
| quantized_store.rs | ruvector-mincut-gated-transformer | 523 | 88-92% | DEEP | Two-tier KIVI (4-bit warm, 2-bit archive). Correct byte packing. Per-channel keys / per-token values (KIVI paper). Silent warm overflow. Orthogonal to kvquant.rs | R94 |
| mod_routing.rs | ruvector-mincut-gated-transformer | 537 | 72-78% | DEEP | MoD (Mixture-of-Depths) NOT MoE. Deterministic lambda-delta routing. route_unstable/stable are byte-identical duplicates. Boundary detection = stride heuristic, not actual mincut edges | R94 |
| window.rs | ruvector-mincut-gated-transformer | 481 | 82-87% | DEEP | Pure causal sliding window (NOT Longformer despite docstring). Correct numerical-stable softmax. Scalar attention kernels. Feature-gated SparseMask bridge. 5 tests | R94 |
| metrics.rs | ruvector-mincut-gated-transformer | 495 | 78-82% | DEEP | PPL/accuracy quality tracker, NOT cache hit-rates despite docstring. tier_metrics() mostly hardcoded. Dead code paths (should_adapt, boundary_adjustment_factor unused by manager.rs). Rolling window correct | R94 |
| quant4.rs | ruvector-mincut-gated-transformer | 506 | 72-78% | DEEP | RTN (round-to-nearest) NOT GPTQ/AWQ. Fully scalar, no SIMD. BlockInt4Weights write-only (no dequantize). Wastes 12.5% INT4 range. 9 tests including exhaustive round-trip | R94 |
| spike.rs | ruvector-mincut-gated-transformer | 366 | 88-92% | DEEP | Scheduling layer (NOT neuron dynamics), Q15 rate-tiering, FNV-1a novelty hashing, full coherence with packets.rs (R94). no_std compatible | R97 |
| config.rs | ruvector-mincut-gated-transformer | 369 | 88-92% | DEEP | Well-structured serde config, covers only ~40% subsystem surface (missing energy gate, spike, quantization, KV cache tier, and MoD routing configs) | R97 |
| kv_cache/tier.rs | ruvector-mincut-gated-transformer | 305 | 90-93% | DEEP | Clean 3-tier definitions (Hot/Warm/Archive = FP16/4-bit/2-bit), age-based not access-frequency tier assignment. Aligns with R94 KIVI findings. 9 unit tests | R97 |
| lib.rs | ruvector-mincut-gated-transformer | 261 | 90-95% | DEEP | EXCELLENT crate root: 23 public modules, 60+ exported items, 10 feature flags, 43-item prelude, dual KV cache re-export. Rustdoc with academic citations + working example. All DEEP modules accessible | R98 |
| norm.rs | ruvector-mincut-gated-transformer | 213 | 55-65% | DEEP | WEAKEST MinCut kernel. LayerNorm + RMSNorm math correct but ALL pure scalar — no SIMD despite SIMD siblings (qgemm.rs, ffn.rs). RMSNorm feature-flag body duplication is dead code. 6 tests | R98 |
| kv_cache/mod.rs | ruvector-mincut-gated-transformer | 98 | ~90% | DEEP | ADR-004 Three-Tier Adaptive KV Cache module root: FP16 hot → 4-bit KIVI warm → 2-bit archive. 9 submodules publicly re-exported. Legacy backward-compat preserved. Pure module root — all logic in children | R100 |

### MinCut Dynamic Algorithms (R111)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| expander/mod.rs | ruvector-mincut | 954 | 62-68% | DEEP | Correct conductance formula φ(S)=cut(S,V\S)/min(vol(S),vol(V\S)). 3 CRITICAL: single-level hierarchy (O(log n) claimed, never built), delete_edge incomplete (disconnection unhandled), O(n³) actual vs O(m log n) claimed. NOT Saranurak-Wang — greedy BFS heuristic. 22 tests. **WEAKEST MinCut algorithmic link** | R111 |
| witness/mod.rs | ruvector-mincut | 921 | 55-65% | DEEP | WitnessTree+LazyWitnessTree citing Jin-Sun-Thorup SODA 2024. **JST claim is ARCHITECTURAL FICTION** — O(n*m) BFS, not subpolynomial. 3 CRITICAL: dead existence check (tautology), incorrect LCT cut (wrong parent severed), dual source-of-truth corruption (tree_edges vs lct diverge). recompute_min_cut misses non-tree edges. 22 tests | R111 |
| graph/mod.rs | ruvector-mincut | 735 | 90-95% | DEEP | **BEST MinCut infrastructure file.** DynamicGraph with 3-index DashMap layout (adjacency, edges, edge_index). O(1) has_edge, O(deg) neighbors. BFS connectivity + connected_components. AtomicU64 edge IDs. TOCTOU race on concurrent insert_edge (M). Clone remaps EdgeIds (M). 20 tests | R112 |
| lib.rs | ruvector-mincut | 732 | 88% | DEEP | Genuine orchestration root: 23 public modules, 60+ exported items, 10 feature flags. 2 CRITICAL: ForestPacking naming collision (localkcut vs jtree shadows on glob import), SNN 6-layer neuromorphic unverified. WASM types without cfg(wasm). Two parallel Fragment namespaces. 17 tests | R112 |
| fragmentation/mod.rs | ruvector-mincut | 640 | 70-75% | DEEP | Recursive decomposition hierarchy citing arXiv:2512.13105. HIGH: Trim is greedy heuristic NOT spectral/flow-based. Boundary underestimates (ignores cross-fragment edges). is_expander() threshold inconsistency. No SIMD. 7 thin tests | R112 |
| localkcut/deterministic.rs | ruvector-mincut | 515 | 70-78% | DEEP | Cites real arXiv:2512.13105. 4-color scheme correct structure. 2 CRITICAL: coloring family STUB (empty HashMaps), color assignment is arithmetic parity NOT paper's (a,b)-family. O(m) delete vs paper's O(log n). 600K DFS per query. Boundary double-counting. 5 tests | R112 |
| algorithm/mod.rs | ruvector-mincut | 1,009 | 35-45% | DEEP | **COMPLEXITY FRAUD** — claims O(n^{o(1)}) but does O(n log n) rebuild + O(n*m) BFS recompute per mutation. Cycle-edge handler EMPTY STUB. LCT cut() wrong (severs parent not target). find_replacement_edge() tree-edges-only. 22 tests pass via brute-force correctness. **WORST MinCut file** — makes SIMD kernels architecturally unreachable | R113 |
| linkcut/mod.rs | ruvector-mincut | 963 | 70-78% | DEEP | Genuine Sleator-Tarjan skeleton (preferred paths, zig-zig/zig-zag, path_parent). 3 CRITICAL: is_root() conflates NodeId (u64) with index (usize) — only correct for sequential IDs from 0 (all 13 tests masked). link() semantics inverted vs docstring. verify_root_cache() complete stub. bulk_update() stale aggregates | R113 |
| localkcut/mod.rs | ruvector-mincut | 929 | 72-78% | DEEP | Genuine BFS structure, cites arXiv:2512.13105 correctly. find_cut() uses 16 flat color masks not paper's 4^d tuples. Trivial mod-4 coloring not derandomized family. ForestPacking builds identical forests. witnesses_cut() AND/OR inverted. UnionFind vertex ID collision. 21 tests | R113 |
| error.rs | ruvector-mincut-gated-transformer | 95 | ~92% | DEEP | 5-variant Error: BadConfig, BadWeights, BadInput, OutputTooSmall, UnsupportedMode. Hard no-panic contract documented. is_recoverable() + is_config_error() for caller retry logic. 3 tests, no_std compatible. HIGH QUALITY | R100 |
| attention/linear.rs | ruvector-mincut-gated-transformer | 67 | 15-20% | DEEP | **LINEAR ATTENTION IS A PLACEHOLDER** — struct exists with config, NO forward pass, NO ELU+1 approximation, NO kernel math. Docstring cites Katharopoulos 2020 but implementation EMPTY. Hidden behind `linear_attention` feature gate (not in default build). Deflates MinCut O(n) attention novelty claims | R100 |
| attention/mod.rs | ruvector-mincut-gated-transformer | 35 | ~88% | DEEP | Feature-gated attention architecture: linear/spike/sparse all opt-in, default = SlidingWindowAttention only. MinCut sparse path (apply_mincut_sparse_mask) real but non-default. 4 academic citations: MInference, Spike-driven Transformer, Spectral Attention | R100 |
| kernel/mod.rs | ruvector-mincut-gated-transformer | 24 | ~90% | DEEP | Pure re-export root for bench_utils, norm, qgemm, quant4. Dual-precision: INT8 GEMM + INT4 pipeline. Benchmark utilities built into kernel module | R100 |
| kernel/bench_utils.rs | ruvector-mincut-gated-transformer | 442 | 80-85% | DEEP | PURE MEASUREMENT SCAFFOLDING — zero MinCut SIMD kernel invocations. Timer/BenchStats/BenchConfig harness library. Three timer bugs: non-serialized RDTSC (can produce negative cycles), hardcoded 3 GHz divisor, no black_box (optimizer may eliminate benchmarked code). Wrong SSE2 gate on RDTSC (should be unconditional on x86_64). Genuine no_std with AArch64 CNTVCT_EL0 path. 5 unit tests | R101 |
| lib.rs | ruvector-mincut-gated-transformer-wasm | 489 | 88-92% | DEEP | **19th GENUINE WASM.** Real wasm_bindgen, wraps MincutGatedTransformer, imports 9 types from parent crate. Full gating mechanism preserved (5 GateDecision + 8 GateReason). 3 constructors. WasmInferResult 12 witness fields. MEDIUM: QuantizedWeights::empty() in all constructors — no weight-loading API. SpikePacket top-k zeroed at boundary. 2 wasm_bindgen_tests | R112 |

### ruvllm Context Module (R104)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| context/semantic_cache.rs | ruvllm | 675 | 88-93% | DEEP | Dual-path lookup: MD5 hash for exact + HNSW cosine for semantic (threshold 0.85). REAL ruvector-core HnswIndex. LRU eviction + TTL. 7 tests. NOT hash-based | R104 |
| context/episodic_memory.rs | ruvllm | 743 | 82-88% | DEEP | REAL HNSW-indexed episode storage. MemoryCompressor genuine (top-K by reward, vector mean, frequency patterns). Standalone design — not composed by AgenticMemory (TWO parallel episodic systems). Truncation-not-PCA in compress_embedding | R104 |
| context/working_memory.rs | ruvllm | 697 | 88-92% | DEEP | Time-decay attention (exp, rate=0.1/min). VecDeque scratchpad + tool cache + variable store. CRITICAL: eviction is O(n) full scan on every overflow. Attention weights computed but IGNORED by FIFO eviction. 8 tests | R104 |
| context/context_manager.rs | ruvllm | 794 | 82-87% | DEEP | IntelligentContextManager genuinely composes AgenticMemory + SemanticToolCache. prepare_context() real pipeline. EpisodicMemory + WorkingMemory + ClaudeFlowBridge NOT directly composed. PriorityScorer + MemorySummarizer real. 6 tests | R104 |
| context/claude_flow_bridge.rs | ruvllm | 688 | 60-68% | DEEP | CRITICAL: CLI-subprocess adapter — ALL calls via std::process::Command spawning npx. No EmbeddingService, no vector search, no Rust API. TTL cache, input validation, hive sync stubs. 5th routing surface (get_routing_suggestion via CLI hooks). 7 tests | R104 |
| reasoning_bank/trajectory.rs | ruvllm | 630 | 88-92% | DEEP | TrajectoryRecorder builder pattern. 5-variant StepOutcome with hardcoded quality scores (Success=1.0, Failure=0.0). compute_quality() heuristic verdict-weighted average (not RL discounted return). 8 tests. No VerdictAnalyzer hookup | R104 |
| reasoning_bank/consolidation.rs | ruvllm | 736 | 90-94% | DEEP | GENUINE EWC++: FisherInformation EMA-of-squared-gradients (Schwarz et al. 2018). apply_constraint() Fisher-weighted gradient damping. regularization_loss() textbook Kirkpatrick eq. 3. consolidate_fisher() destroys per-pattern importance (lossy trade-off). consolidation_count never incremented (bug: &self not &mut self) | R104 |
| claude_flow/task_classifier.rs | ruvllm | 383 | 72-78% | DEEP | Pure keyword matcher — two Vec<(String, Vec<&str>)> pattern lists. Zero ML/embedding. Claims "RuvLTRA embeddings" in docstring. Dead-end: outputs not consumed by model_router/hooks/hnsw_router. 2 tests | R104 |
| claude_flow/flow_optimizer.rs | ruvllm | 319 | 70-75% | DEEP | Orchestration wrapper around AgentRouter + TaskClassifier. Hardcoded latency/memory improvement constants (10/25/40%, 20/40/60%). 18th pseudo-embedding (sinusoidal sweep, content-independent). SONA delegation real | R104 |

### SONA & Learning

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| sona (27 files) | ruvector | ~4,500 | 85% | DEEP | MicroLoRA, EWC++, federated, SafeTensors. Production-ready | R13 |
| sona_instant.rs | ruvllm-wasm | 846 | 78-83% | DEEP | Genuine WASM EMA/pattern-buffer/cosine machinery. 2 CRITICAL: EWC-lite is NOT EWC (index heuristic, no Fisher matrix), HNSW claim FALSE (O(n) linear scan, code admits "future integration point"). current_rank has no operational effect. Pattern buffer not serialized. 13 tests. Confirms SONA WASM bimodal | R111 |
| sona/integration.rs | ruvllm | 571 | 72-78% | DEEP | Bridge between ruvllm inference and ruvector_sona. All API conformance real. 3 CRITICAL: dual-instance state divergence (EWC++ updates siloed from inference path), background loop synchronous on request thread (stalls inference), compute_pseudo_gradients() voided EWC++ (query embeddings not parameter gradients). EWC param_count 256 vs coordinator's 4096. Reinforces R98 SONA downgrade | R111 |

### ruvector npm Umbrella Package (R117)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| src/core/onnx-embedder.ts | ruvector (npm) | 458 | 85-90% | DEEP | **REAL ONNX** — Tract WASM (7.4MB), all-MiniLM-L6-v2, genuine 384-dim Float32Array. 6 models, parallel batch. BREAKS hash-based finding for TS layer | R117 |
| src/core/learning-engine.ts | ruvector (npm) | 752 | 78-82% | DEEP | 9 RL algorithms: 6 REAL tabular (Q-Learning, SARSA, Double-Q, Monte Carlo, TD(lambda), Actor-Critic), PPO BROKEN (wrong ratio), Decision Transformer FACADE, DQN simplified. All Map-based. CLI+MCP consumed | R117 |
| src/core/intelligence-engine.ts | ruvector (npm) | 1233 | 72-78% | DEEP | Master orchestrator. 3 CRITICAL: sync embed() ALWAYS hash even with ONNX, VectorDB/Map desync after import, constructor race conditions. Hash dominates despite ONNX availability | R117 |
| src/index.ts | ruvector (npm) | 195 | 85-88% | DEEP | VectorDBWrapper — **CONFIRMED WORKING** by live test. Synchronous native detection, proper await. Kitchen-sink re-exports 22+ core modules. NativeVectorDb bypass skips JSON conversion | R117 |
| ruvector.test.js | agentic-synth | 326 | 72-78% | DEEP | 22 vitest tests — ALL run against JS Map fallback, NEVER native HNSW. @ruvector/core absent from package.json. Good error-path coverage, wrong integration target | R117 |
| npm/packages/ruvector/src/core/index.ts | ruvector (npm) | 57 | 85-88% | DEEP | Master barrel export — 23 modules re-exported (GNN, SONA, ONNX×2, Router, Graph, Cluster, AST, AgentDB-fast, Intelligence, Parallel×2, Neural×3, RVF, Attention, Diff, Coverage, Tensor, Learning, Adaptive, Analysis). 14 named defaults. No selective gating — dead code leaks to consumers | R139 |

### CI/CD Pipelines (R139)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| .github/workflows/build-native.yml | ruvector | 242 | 82-87% | DEEP | 5-platform NAPI cross-compile (linux-x64/arm64, darwin-x64/arm64, win32-x64). graph-node DISABLED (PR #15). Test continue-on-error:true. Manual publish only. Binaries auto-committed to repo | R139 |
| .github/workflows/sona-napi.yml | ruvector | 299 | 85-90% | DEEP | 7-platform SONA NAPI build (+ musl + win32-arm64) with universal macOS binary via lipo. Full publish pipeline to @ruvector/sona-{platform} packages. Post-publish smoke test on 3 platforms. Most comprehensive NAPI CI in repo | R139 |

### RVF File Format (R121+R123)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| rvf-node/src/lib.rs | rvf-node | 987 | 88-92% | DEEP | GENUINE 22-method NAPI bridge. CRITICAL: verify_witness bypasses real chain verifier | R121 |
| rvf-runtime/src/read_path.rs | rvf-runtime | 520 | 88-92% | DEEP | GENUINE binary read path. Backward-scan manifest, CRC32 integrity | R123 |
| rvf-runtime/src/cow.rs | rvf-runtime | 503 | 88-93% | DEEP | GENUINE cluster-level COW. SHAKE-256 witness trail. Write coalescing | R123 |
| rvf-crypto/src/witness.rs | rvf-crypto | 190 | 88-92% | DEEP | ROOT-CAUSES R121 CRITICAL. Real SHAKE-256 chain verifier exists. NAPI bypasses it | R123 |
| rvf-wire/src/writer.rs | rvf-wire | 167 | 88-92% | DEEP | GENUINE serializer. 64-byte RVFS header, XXH3-128, 64-byte alignment | R123 |
| rvf-wire/src/reader.rs | rvf-wire | 175 | 90-93% | DEEP | GENUINE deserializer. Symmetric with writer. SEALED bypass prevention | R123 |
| rvf-crypto/src/attestation.rs | rvf-crypto | 840 | 75-80% | DEEP | SHAKE-256 genuine. TEE verification FACADE (zero QuoteVerifier impls). Unverified sealed_key | R124 |
| rvf-kernel/src/lib.rs | rvf-kernel | 862 | 82-87% | DEEP | IMAGE BUILDER not VM. Docker+prebuilt bzImage. Zero callers. ORPHANED from runtime | R124 |
| rvf-launch/src/lib.rs | rvf-launch | 718 | 85% | DEEP | Genuine QEMU spawner. microvm+KVM/TCG, virtio-blk/net. Zero host isolation | R124 |
| rvf-manifest/src/level0.rs | rvf-manifest | 549 | 88% | DEEP | 4096-byte EOF root manifest. CRC32C, 6 segment pointers, COW metadata. Signature NOT verified | R124 |
| npm/packages/rvf/src/backend.ts | rvf (npm) | 791 | 78-84% | DEEP | NodeBackend genuine NAPI delegation. WasmBackend PARTIAL FACADE (9/20 methods throw) | R124 |
| npm/packages/rvf/src/types.ts | rvf (npm) | 238 | 80% | DEEP | 18 type exports. 8 type mismatches vs Rust NAPI. RvfIndexStats/WitnessResult dead | R124 |
| npm/packages/rvf/src/index.ts | rvf (npm) | 69 | 85% | DEEP | Barrel re-export. Hard rvf-solver import crashes if absent | R124 |
| npm/packages/rvf/src/database.ts | rvf (npm) | 292 | 85% | DEEP | Clean delegation wrapper. ensureOpen() no health check. Kernel/eBPF API confirms general container | R124 |
| npm/packages/rvf/src/errors.ts | rvf (npm) | 143 | 90% | DEEP | 30 error codes in 7 categories. 1:1 Rust mapping. Tile WASM category confirms microkernel | R124 |
| domain-expansion-wasm/src/lib.rs | ruvector-domain-expansion-wasm | 504 | 85-88% | DEEP | 19th genuine WASM module. Full engine + Thompson + population + RVF bridge via wasm_bindgen | R115 |
| domain-expansion/src/cost_curve.rs | ruvector-domain-expansion | 483 | 85-88% | DEEP | AUC trapezoidal integration, convergence thresholds, progressive_acceleration() IQ growth test | R115 |
| domain-expansion/src/domain.rs | ruvector-domain-expansion | 213 | 85% | DEEP | Domain trait (Send+Sync): generate_tasks, evaluate, embed, reference_solution. Extensibility point | R115 |
| domain-expansion/src/error.rs | ruvector-domain-expansion | 40 | 90% | DEEP | Standard error enum. Clean | R115 |
| domain-expansion/src/lib.rs | ruvector-domain-expansion | 592 | 85-90% | DEEP | DomainExpansionEngine composes 5 subsystems. FIRST working Thompson Sampling feedback loop | R115 |
| domain-expansion/src/meta_learning.rs | ruvector-domain-expansion | 1399 | 82-87% | DEEP | Regret+plateau+pareto+curiosity meta-learning. Largest file in crate | R115 |
| domain-expansion/src/planning.rs | ruvector-domain-expansion | 648 | 82-85% | DEEP | Planning domain. Goal matching exact string only (M9) | R115 |
| domain-expansion/src/policy_kernel.rs | ruvector-domain-expansion | 469 | 85-88% | DEEP | 8 policy kernels, elitism 25%, tournament mutation 30%. Standard evolutionary algorithm | R115 |
| domain-expansion/src/rust_synthesis.rs | ruvector-domain-expansion | 604 | 75-80% | DEEP | Feature-counting embeddings (H218), heuristic-only evaluation (H219). 60-70% tasks lack reference | R115 |
| domain-expansion/src/rvf_bridge.rs | ruvector-domain-expansion | 716 | 88-92% | DEEP | STANDOUT: SHAKE-256 witness, SolverPriorExchange (could fix C40), AGI container TLV, 10 tests | R115 |
| domain-expansion/src/tool_orchestration.rs | ruvector-domain-expansion | 728 | 85-88% | DEEP | Most sophisticated evaluation: type chain validation, latency estimation, cost+retry, error coverage | R115 |
| domain-expansion/src/transfer.rs | ruvector-domain-expansion | 584 | 85-88% | DEEP | sqrt-dampened priors, Beta approx Thompson Sampling, pessimistic 1.5x cost EMAs | R115 |
| postgres/src/domain_expansion/mod.rs | ruvector-postgres | 22 | 85% | DEEP | Global engine state via DashMap<Arc<RwLock>>. Correct pgrx concurrent access | R115 |
| postgres/src/domain_expansion/operators.rs | ruvector-postgres | 52 | 72-78% | DEEP | pg_extern ruvector_domain_transfer(). Thin entry point — no evolution loop (M12) | R115 |

### npm CLI Entrypoints (R135)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| npm/packages/ruvector/bin/cli.js | ruvector (npm) | 7,357 | 72-78% | DEEP | Largest single CLI. 14 top-level commands. Uses VectorDB (working). Hash embedding confirmed (C104). 4 facade commands. ~4K lines hooks/intelligence | R135 |
| npm/packages/ruvector/bin/mcp-server.js | ruvector (npm) | 3,007 | 78-82% | DEEP | 55 MCP tools in one switch statement. 9th hash instance (C105). Query sanitization destroys SQL/Cypher/SPARQL (C106). 14 tools via execSync, 11 via agentic-flow | R135 |
| npm/packages/rvlite/bin/cli.js | rvlite (npm) | 1,686 | 80-85% | DEEP | INDEPENDENT vector DB. O(n) flat search. Genuine WASM (SONA+Attention). Poincare+Lorentz correct. Zero ruvector imports. Advertises SQL/Cypher/SPARQL, provides none | R135 |
| npm/packages/ruvllm/bin/cli.js | ruvllm (npm) | 1,005 | 72-78% | DEEP | All facades without native binary (C107). Training SIMULATED (C108). SIMD benchmark genuine. Zero ruvector RAG integration (H253) | R135 |

### claude-flow CLI ruvector Commands (R138)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| v3/@claude-flow/cli/src/commands/ruvector/setup.ts | claude-flow (v3) | 784 | 85-90% | DEEP | Pure scaffolding: generates docker-compose.yml + init-db.sql (476 LOC) + README. SQL schema production-quality (8 tables, 6 HNSW indices, 7 functions). Does NOT configure backend factory, does NOT switch default from sql.js. Hardcoded test credentials. Extension version mismatch risk (0.1.0 vs 2.0.0). Zero relationship to memory-initializer.ts | R138 |

### Integration Tests & Deployment (R139)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| crates/ruvllm/tests/e2e_integration_test.rs | ruvllm | 1,535 | 60-65% | DEEP | Mock-backend unit tests, genuine softmax/sampling math, UB pointer cast at L840 | R139 |
| crates/prime-radiant/tests/ruvllm_integration_tests.rs | prime-radiant | 1,393 | 40-45% | DEEP | 100% mock, zero cross-crate imports, API mismatch with production | R139 |
| tests/integration/distributed/docker-compose.yml | ruvector-root | 198 | 50-55% | DEEP | 5-node Raft cluster with dummy shell-script nodes, real cargo test runner | R139 |


### V3 Intelligence Layer (R140)

| File | Package | LOC | Real% | Depth | Key Verdict | Session |
|------|---------|-----|-------|-------|-------------|---------|
| v3/@claude-flow/cli/src/memory/intelligence.ts | claude-flow (v3) | 985 | 55-60% | DEEP | CRITICAL FACADE: claims O(log n) HNSW, actual is O(n) brute-force cosine. LocalSonaCoordinator: LoRA/EWC config fields stored but NEVER used. compactPatterns() O(n²) with maxPatterns=5000. 14+ consumers, zero @ruvector/* imports. NOT dead code — just algorithmically false | R140 |
| v3/@claude-flow/cli/src/memory/sona-optimizer.ts | claude-flow (v3) | 842 | 72-78% | DEEP | GENUINE agent-routing optimizer. Bayesian confidence update, temporal decay, pattern pruning. Connected to hooks pipeline. Zero HNSW/ruvector connection. Q-learning lazy-load degrades to JS-only silently | R140 |
