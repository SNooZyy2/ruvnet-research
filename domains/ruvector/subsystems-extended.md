### 5o. MinCut-Gated-Transformer (R91+R93+R94+R96)

**MOST NOVEL CRATE confirmed — ~84% weighted avg across 27 DEEP files (R91+R93+R94+R96).** The ruvector-mincut-gated-transformer crate implements a quantized INT8 transformer inference engine with a novel energy-based gating mechanism derived from minimum-cut graph theory.

**Core novelty mechanism — energy_gate.rs (88-93%, R93):** The "MOST NOVEL" rating (R34) traces to this file. Implements a genuine 3-component Energy-Based Transformer (EBT) energy function: sigmoid lambda decay (1/(1+lambda/150)), boundary edge penalty, and partition entropy (ln(k)/ln(10)). Central-difference finite gradient computation drives System-2 iterative refinement via gradient descent on the controllable lambda variable. Dominant gradient component selects intervention type. Low-confidence states fall back to rule-based GatePolicy — a pragmatic hybrid. Cites Gladstone 2025. The scale-sensitivity of hardcoded normalization constants (lambda/150, boundary/100, max 10 partitions) is the main engineering gap.

**Early exit — early_exit.rs (88-92%, R93):** Implements CoherenceEarlyExit, an alternative to LayerSkip (Elhoushi et al. 2024). Uses precomputed mincut lambda signals from GatePacket to decide when to terminate early and begin speculative token generation. Novel 40/40/20 confidence weighting: lambda-strength + stability + boundary dispersion. Adaptive exit layer adjusts based on lambda stability. Does not directly import energy_gate.rs — dependency is mediated by transformer forward pass.

**Computational kernels (88-93% avg, R93):**
- **qgemm.rs (88-93%):** Production quantized GEMM. Real AVX2 (correct widening madd_epi16, prefetch pipelining) and NEON (dual accumulators, vaddvq horizontal reduction). Asymmetric per-row scales matching llama.cpp convention. No cache-blocking/tiling limits throughput on large matrices. SIMD dispatch is compile-time only (no runtime detection — H43).
- **ffn.rs (88-92%):** Standard GELU FFN (Vaswani 2017), NOT SwiGLU/GeGLU despite crate name. Genuine SIMD GELU: AVX2 Padé tanh (27+x²)/(27+9x²), NEON Newton-Raphson reciprocal. Allocation-free forward pass. Single-scale approximation bug (H47).
- **spike_driven.rs (88-93%):** Genuine neuromorphic spike-driven attention following Yao et al. 2023. Real LIF neuron rate coding with membrane accumulate-and-reset, correct STDP temporal coincidence detection, refractory period enforcement. 12 tests. However, spike_value_contribution() uses saturating_mul, contradicting "multiplication-free" claim (H37). Zero integration with energy_gate.rs.
- **q15.rs (88-92%):** Correct Q15 fixed-point newtype (transparent u16). Widening multiply, direction-aware LERP, 11 tests. Exported but UNUSED within crate — rope.rs duplicates the constant as raw i16 (type mismatch H41).

**Attention variants:**
- **sparse_attention.rs (72-80%, R93):** Novel design thesis — mincut lambda values control attention density (cites MInference NeurIPS 2024). GatePacket API with lambda, boundary_concentration_q15, partition_count. Well-reasoned adaptive density formula. **BUT**: partition boundaries are PLACEHOLDER (uniform stride, H39) and density parameter is dead code (H40). Functionally equivalent to fixed-stride BigBird local attention.
- **speculative.rs (88-92%, R91):** EAGLE-style tree speculative decoding with novel lambda-guided confidence. Sequential path verification (not parallel tree forward).

**KV cache management (72-87% avg, R91+R93+R96):**
- **manager.rs (72-78%, R93):** AdaptiveKVCache with 3-tier Hot/Warm/Archive. PARALLEL to legacy.rs (zero imports between them). Pure FIFO eviction (H44). SQuat/KVQuant quantizers declared but dead (H45). Rematerialization policy complete but never triggered (H46). Presets match LLaMA-2/3 and GPT-4 parameters.
- **kvquant.rs (78-85%, R93):** Pre-RoPE KVQuant (Hooper et al. 2024) — quantize keys BEFORE RoPE for lower dynamic range. Correct deferred RoPE formula. 3-bit packing broken (falls back to 4-bit, H48). Per-layer calibration API lies (flattens to single pair).
- **kivi.rs (72-78%, R96):** KIVI paper's bit-packing primitive. GENUINE FWHT (butterfly, self-inverse, power-of-2 enforced). CORE DEFECT: per-channel quantization claim FALSE — global min/max regardless of QuantScheme (H60c). PerGroup discards group_size (H60c). SIMD dequantize TODO (H60d). Complementary to quantized_store.rs (tier/graduation logic) — provides low-level bit-packing.
- **squat.rs (78-84%, R96):** SQuAt 2024 paper. Hadamard basis + Gram-Schmidt math CORRECT. 2/4-bit packing COMPLETE. calibrate() STUB — ignores calibration data (H60b). No integration with sibling kv_cache files.
- **policy.rs (82-87%, R96):** Three-struct architecture (MemoryTracker/TierPolicy/RematerializationPolicy) genuine. Age-based graduation only — NO H2O/attention-sink/StreamingLLM (H60j). Evaluate/tracker pressure divergence (H60k). Cost model inversion (older tokens appear cheaper to recompute). No direct hot_buffer.rs integration.
- **hot_buffer.rs (82-87%, R96):** FIFO ring buffer for recent tokens in "FP16" (actually f32 — test acknowledges). Separate high-precision tier. pop_oldest() BUG — no read cursor (H60e). Dual-API trap (push vs push_head+advance — H60e). Zero policy.rs integration. keys()/values() correctly reconstruct chronological order for wrapped ring buffer.
- **legacy.rs (82-88%, R91):** RotateKV (IJCAI 2025) with FWHT. Per-head scale stomping bug (H28). No eviction (H29).
- **tier.rs (90-93%, R97):** Clean 3-tier definitions (Hot/Warm/Archive = FP16/4-bit/2-bit). Age-based not access-frequency tier assignment (H68). 9 unit tests.

**R96 additions — arena + trace:**

**arena.rs (92%, R96):** GENUINE bump allocator built specifically for MinCut inference weight layout. Single Vec<u8> backing store with manual offset tracking. 64-byte cache-line alignment via bit-mask arithmetic. WeightRef stores u32 offset+size (serialization-safe). calculate_arena_size encodes QKV+FFN layout (ignores _heads param — GQA/MQA unsupported). SAFETY comments on all unsafe blocks are accurate. **CRITICAL**: multiple &mut slices from same Vec trigger aliasing UB (H60a). reset() leaves stale weights readable via new alloc slices. 10 unit tests.

**trace.rs (88-92%, R96):** Gate-decision diagnostics with zero heap allocation. Fixed-size [T;64] stack arrays as circular rolling buffer. Records GateDecision/GateReason/lambda/tier from packets.rs. Tier 0-3 classification consistent with packets.rs semantics. lambda_trend() splits window in half and compares means (simple monotone detection, not regression). Feature gate claim documentation-only — compiles unconditionally (H60i). No serialization output path. 6 genuine unit tests.

**Quantization pipeline:** q15.rs (u16) and rope.rs (i16) use the same Q15_SCALE constant (32768.0) but are **architecturally disconnected** due to signed/unsigned divergence. The quantized inference path would flow: q15.rs format → qgemm.rs INT8 matmul → ffn.rs GELU → output, but q15.rs newtype is never imported by any sibling.

**R94 additions — infrastructure + routing (7 files, ~3,534 LOC, ~83% weighted avg):**

**packets.rs (90-95%, BEST in session):** Pure type-definition coherence interface connecting energy gate → spike scheduler → transformer kernel. GatePacket carries lambda, boundary_edges, partition_count. GateDecision is a 5-level hierarchy: Allow → ReduceScope → FlushKv → FreezeWrites → QuarantineUpdates. QuarantineUpdates is architecturally novel — compute runs but all state changes discarded, enabling speculative inference. SpikePacket bridges LIF+STDP output to sparse attention. InferInput supports dual modes (tokens or quantized embeddings). InferStats tracks qgemm_calls, tokens_skipped (MoD), early_exit_layer — confirming subsystem integration. All repr(C) with consistent Q15 fixed-point for cross-language ABI.

**state.rs (88-92%):** Zero-alloc RuntimeState manages all inference buffers as a single contiguous Vec with computed BufferLayout offsets: Q → K → V → attn_scores → FFN → residual → norm → K_cache → V_cache, all 64-byte aligned. KV ring buffer with per-layer write indices and valid lengths. Inference-only — no checkpoint, no safetensors, no version tagging. Unsafe accessor inconsistency: 6 of 9 buffer accessors use unbounded `[start..]` slicing (UB risk if layout is wrong, H57).

**quantized_store.rs (88-92%):** Two-tier KIVI quantized KV cache — 4-bit warm tier, 2-bit archive tier. Correct asymmetric quantization (PerChannel keys, PerToken values per KIVI paper). Warm→archive graduation via dequantize+requantize (cascaded quantization error, no error diffusion). Orthogonal to kvquant.rs — imports kivi.rs only. Silent no-op on warm overflow (H54).

**mod_routing.rs (72-78%):** MoD (Mixture-of-Depths) routing, NOT MoE. Deterministic lambda-delta signal from GatePacket drives per-token skip/compute decisions. |λ_delta| < threshold → stable → more skipping; |λ_delta| >= threshold → unstable → more compute. 50% FLOPs hard cap via layer_capacity_ratio. BUT: route_unstable_tokens() and route_stable_tokens() are functionally identical (H52), and boundary detection uses regular stride intervals, not actual mincut edge IDs (H53).

**window.rs (82-87%):** Causal sliding window attention — each token attends to W previous tokens. NOT Longformer despite docstring claim (no global/CLS tokens, no two-pattern attention). Scalar attention kernels (no SIMD in Q@K^T or V weighted sum — only qkv_projection delegates to qgemm_i8). Feature-gated SparseMask bridge to sparse_attention.rs under `cfg(feature = "sparse_attention")`. Correct numerical-stable softmax (max subtraction + inv_sum). Asymmetric Q vs KV layout fragility.

**metrics.rs (78-82%):** QualityTracker tracks PPL-degradation and task-accuracy scores in a rolling 1,000-entry VecDeque. Correct sliding-window subtract-on-evict pattern (floating-point drift risk is acceptable at window size 1000). Docstring falsely claims "cache hit rates per tier" but has zero hit-rate tracking (H59). tier_metrics() returns hardcoded min/max/std_dev. boundary_adjustment_factor() and should_adapt() are dead code from manager.rs's perspective.

**quant4.rs (72-78%):** Plain RTN (round-to-nearest) 4-bit quantization with absmax scaling. NOT GPTQ/AWQ — no Hessian-based error minimization, no activation-aware reordering. Fully scalar (no SIMD) despite being in kernel/ directory alongside SIMD-heavy qgemm.rs (H55). BlockInt4Weights is write-only (no dequantize_row, no GEMV — H56). Wastes 12.5% INT4 range (never uses -8 slot).

**R97 additions — config.rs, spike.rs, kv_cache/tier.rs (3 files, ~1,040 LOC, ~90% weighted avg):**

**config.rs (88-92%, 369 LOC):** Well-structured serde config for transformer inference. TransformerConfig covers: num_layers, hidden_size, num_heads, head_dim, seq_len_max, window_normal/degraded/critical (3-tier degraded mode hierarchy). GatePolicy with Q15 fixed-point ratios for allow/reduce/flush/freeze thresholds and conservative/permissive presets. Both derive Serialize/Deserialize (fully TOML-compatible). Correct kv_cache_bytes() formula uses layers×seq_len_max×hidden but omits head dimension multiplier for multi-head (H67 adjacent). validate() checks window_normal > 0 but misses window_degraded > 0. No from_file() or from_toml() constructor (only serde_json::from_str). Tests: 3 covering baseline_config, micro_config, invalid_config. **Coverage gap (H67)**: NO fields for energy gate EBT weights, spike neuron parameters (tau_m, tau_s, threshold), quantization bit-width selection, KV cache tier boundaries (Hot/Warm/Archive capacities), or MoD routing thresholds — five out of nine DEEP subsystems are unconfigurable through this file.

**spike.rs (88-92%, 366 LOC):** SpikeScheduler is a TOKEN DISPATCH LAYER, not a spiking neuron implementation (H69). Reads SpikePacket from packets.rs (coherence with R94). Q15 rate-tiering: rate_q15 above high_threshold → Tier 0 (compute), above medium_threshold → Tier 1 (reduced), above skip_threshold → Tier 2 (skip), else → inactive. FNV-1a novelty hashing (offset_basis=0xcbf29ce484222325) for compute_input_signature() and compute_embedding_signature(). build_sparse_mask() populates boolean mask Vec<bool> from SpikePacket.top_indices() with bounds-checking. no_std compatible (extern crate alloc). 7 tests covering all scheduling branches. Zero spiking neuron dynamics — no membrane_potential, no spike threshold, no STDP weight updates. Contrast with spike_driven.rs (R93, 88-93%) which has genuine LIF+STDP. The two files are complementary: spike_driven.rs generates spikes, spike.rs dispatches based on them.

**kv_cache/tier.rs (90-93%, 305 LOC):** Pure definitions file — enums, structs, and pure functions only (no state, no mutable operations). CacheTier enum: Hot (FP16, 16-bit), Warm (4-bit KIVI), Archive (2-bit KIVI/SQuat/KVQuant). TierConfig fields: hot_capacity, warm_capacity, adaptive bool, quality_threshold (0.95 default). Three presets: default (hot=64, warm=512), long_context (hot=64, warm=1024), extreme_context (hot=32, warm=2048). TierBoundary.tier_for_position() computes age = current_len - position - 1 and delegates to tier_for_age() — **age-based (positional), NOT access-frequency** (H68). requires_dequantization() returns true for Warm and Archive — every attention step for non-hot tokens requires dequant (performance cost). TierCounts.memory_bytes() correct per-tier byte math (Hot=2B/elem, Warm=0.5B/elem+4B scale, Archive=0.25B/elem+4B scale). 9 unit tests covering tier_bits, compression ratios, boundary defaults, tier_for_age edge cases. Perfectly consistent with quantized_store.rs two-tier KIVI (R94, 88-92%) — tier.rs provides the type system, quantized_store.rs implements the logic.

**R98 additions — lib.rs (crate root) and norm.rs (kernel):**

**lib.rs (90-95% EXCELLENT, 261 LOC, R98):** Crate root serving as the public interface for the entire MinCut-Gated-Transformer crate. 23 public module declarations: 18 unconditional (kernel/qgemm, kernel/ffn, energy_gate, early_exit, packets, state, mod_routing, window, metrics, quant4, arena, trace, spike, config, speculative, quantized_store, norm, transformer) + 5 feature-gated (spike_attention, spectral_pe, sparse_attention, energy_gate_extra, kv_cache/*). 60+ exported items via explicit re-exports: MincutGatedTransformer, TransformerConfig, GatePolicy, GatePacket, GateDecision, SpikePacket, InferInput, InferStats, CacheTier, TierConfig, plus all KV cache types and ArenaAllocator. 10 feature flags with semantic groupings: default=[sliding_window], full=[simd,trace,spike_attention,spectral_pe,sparse_attention,energy_gate,kv_cache], novel=[spike_attention,spectral_pe,sparse_attention,energy_gate]. Prelude module re-exports 43 items. Dual KV cache re-export: legacy manager::AdaptiveKVCache (backward compat) + new kv_cache::* types (three-tier). Excellent rustdoc: opening doc comment cites Gladstone 2025 + energy gate concept, includes working code example demonstrating TransformerConfig → MincutGatedTransformer flow. **ARCHITECTURAL CONFIRMATION**: ALL 22+ DEEP-analyzed modules from R93-R97 are publicly accessible — no hidden silos, no dead modules (unlike lib_simple.rs in other crates which excluded algorithms). 3 HIGH findings: (a) "novel" feature flag is non-additive (resets full features), (b) transformer.rs module declared but NOT in file registry (unread DEEP target), (c) feature combinations create combinatorial build matrix (untested combinations).

**norm.rs (55-65%, 213 LOC, R98):** LayerNorm and RMSNorm implementations for the MinCut transformer. Math is correct: LayerNorm uses standard two-pass algorithm (mean pass, then variance pass) with epsilon=1e-5; RMSNorm uses single-pass root-mean-square normalization. In-place variants (layernorm_inplace, rmsnorm_inplace) modify output slice directly. f32_to_i8() and i8_to_f32() quantization helpers for the INT8 inference path. ALL implementations are pure scalar — no SIMD intrinsics anywhere in the file, no `#[target_feature]`, no unsafe blocks, no AVX2/NEON. This contrasts sharply with sibling kernel files: qgemm.rs has real AVX2+NEON widening multiply-accumulate, ffn.rs has Padé tanh + Newton-Raphson GELU. **Dead code anti-pattern (H73)**: `#[cfg(feature = "rmsnorm")]` block contains byte-identical body to the unconditional block — the feature provides no behavioral difference. 6 passing tests. WEAKEST MinCut kernel by far.

**Quality gradient (consistent with broader ruvector pattern, updated R98):**
```
Crate interface (lib.rs):                                   90-95%
Novel algorithms (energy gate, early exit, speculative):    88-93%
Computational kernels (qgemm, ffn, spike_driven, q15):      88-93%
Coherence types (packets, tier.rs):                         90-95%
Runtime state (state, quantized_store, spike.rs, config):   88-92%
Attention variants (window):                                 82-87%
Infrastructure (manager, kvquant, sparse_attention,
               metrics, mod_routing, quant4):                72-82%
Normalization kernel (norm.rs):                              55-65%
```

MinCut crate now has 24 DEEP files with ~84% weighted average.

### 5p. ruvector Non-MinCut LLM Extensions (R91)

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

### 5p. Hyperbolic Geometry (R92+R99)

**FIRST DEEP READS on 63-file crate family — confirmed GENUINE at 88-95%.** Three distinct hyperbolic geometry implementations discovered.

**ruvector-hyperbolic-hnsw (5 files DEEP, ~2,837 LOC, 87-93% weighted — CRATE COMPLETE as of R99):** Complete Poincaré ball HNSW implementation. hnsw.rs (88-93%) is a NATIVE implementation distinct from ruvector-core's hnsw_rs wrapper. Uses fused_norms() for 3x memory bandwidth reduction, tangent-space two-phase search (prune via Euclidean in tangent space, then exact Poincaré for top-N), DualSpaceIndex (Poincaré + Euclidean with RRF fusion). poincare.rs (90-95%) is the mathematical foundation — all Möbius operations verified correct (Ganea et al. 2018), comprehensive numerical stability (EPS guards, clamping, fast_acosh). shard.rs (82-85%) implements hyperbolic-aware radius partitioning (bins by ||x|| in Poincaré ball, which correlates with hierarchy depth), canary deployment pattern, Spearman correlation for hierarchy validation. Transport-absent by design (no sockets, designed for separate coordination layer). WASM wrapper (88-92%) is the 17th GENUINE WASM — 7 real math operations, ShardedIndex, WasmTangentCache, zero stubs.

**R99 — tangent.rs (88-92%, 349 LOC):** Completes the ruvector-hyperbolic-hnsw crate. TangentPruner implements the two-phase pruning strategy referenced in hnsw.rs: TangentCache precomputes Fréchet mean centroid (correct iterative Riemannian GD via poincare.rs log_map/exp_map), then TangentPruner.prune() (1) logs all candidates into tangent space at that centroid, (2) filters by cheap Euclidean distance, (3) re-ranks survivors with exact Poincaré geodesic distance, returning top-N. Design note: search() is O(N) linear scan over the full vector store — this is by design, as the pruner is meant to receive candidates from the HNSW layer at the caller level. However no caller currently wires HNSW traversal output into TangentPruner.search() (H77). Dead import: norm_squared from poincare.rs (H78). Two genuine tests covering full pipeline construction and single round-trip. All imports from poincare.rs and error.rs are consistent — crate is internally coherent throughout.

**ruvector-hyperbolic-hnsw crate quality summary (5 files, all DEEP):**
```
poincare.rs    90-95%  — math foundation, all formulas correct
hnsw.rs        88-93%  — native HNSW, tangent-space two-phase pruning
lib.rs (wasm)  88-92%  — 17th GENUINE WASM
tangent.rs     88-92%  — Fréchet centroid cache, two-phase pruner
shard.rs       82-85%  — radius partitioning, transport-absent
Weighted avg:  ~88%
```

**ruvector-attention hyperbolic module (1 file DEEP + prior coverage, ~580 LOC):** lorentz_cascade.rs (90-93%) is a genuine Lorentz model (hyperboloid) attention mechanism — one of the most mathematically rigorous files in the ecosystem. Uses Minkowski inner product ⟨x,y⟩_L = -x₀y₀ + Σxᵢyᵢ (correct signature), Busemann function scoring B_ξ(x) = log(-⟨x,ξ⟩_L) for O(d) hierarchical attention, multi-curvature cascade (each head at different log-spaced curvature for multi-scale hierarchy capture), Einstein midpoint centroid, correct log/exp maps on the hyperboloid. Inference-only (curvatures/focal directions fixed, no learning mechanism — H35).

**ruvector-postgres hyperbolic module (2 files DEEP, R98):** poincare.rs (88-92%, 268 LOC) and lorentz.rs (87-92%, 258 LOC) are now DEEP-read. Both are pure Rust library files — NOT direct pgx entry points. SQL exposure is indirect via operators.rs `#[pg_extern]` re-exports (DEEP R96). poincare.rs implements correct Poincare ball math (Mobius addition, exp/log maps, curvature scaling). lorentz.rs implements correct Lorentz/Minkowski model with bidirectional coordinate transforms. CRITICAL H70: lorentz.rs distance() accepts off-hyperboloid points silently. See Section 5d-ii for full SQL hyperbolic analysis.

**Three distinct HNSW implementations now all DEEP-confirmed:** ruvector-core (98-100% hnsw_rs wrapper), micro-hnsw-wasm (60-70% novel no_std), hyperbolic-hnsw (88-93% native Poincaré, CRATE COMPLETE R99). Quality ordering: hyperbolic-hnsw > ruvector-core (wrapping) > micro-hnsw-wasm (novel but untested).

### 5p-ii. GNN Bindings — napi-rs and WASM (R99)

**Verdict: GENUINE inference-only bindings at 88-94% — 18th GENUINE WASM confirmed.** Two files complete the FFI surface for ruvector-gnn: a Node.js binding (napi-rs) and a WASM binding (wasm_bindgen). Both share the same inference-only API surface and both genuinely delegate to ruvector_gnn core crate algorithms.

**gnn-node/lib.rs (88-92%, 428 LOC, R99):** Production napi-rs bindings using all four attribute macros: `#[napi]` (function export), `#[napi(constructor)]` (class init), `#[napi(factory)]` (named constructor), `#[napi(object)]` (data-transfer object). GnnGraph class exposes: forward(), compress(), decompress(), differentiable_search(), hierarchical_forward(). CompressionTier enum (None/Half/PQ8/PQ4/Binary) with u8 discriminant, correct defaults. Error handling is solid: Status::InvalidArg returned for range violations (e.g., k parameter validation), .map_err() with Status::GenericFailure for all Rust error conversions — zero unwrap()/panic!() in error paths. Imports 6 real symbols from ruvector-gnn core crate (compress, layer, search modules). **Performance note**: hierarchical_forward() deserializes layer configs from JSON strings on every call (H79) — serde overhead per invocation. No training APIs exported (H80).

**gnn-wasm/lib.rs (90-94%, 415 LOC, R99) — 18th GENUINE WASM:** All exports via `#[wasm_bindgen]` delegate to real ruvector_gnn core algorithms via serde_wasm_bindgen JsValue deserialization. Same inference API as gnn-node: forward(), compress(), decompress(), differentiable_search(), hierarchical_forward(). Additional: inline cosineSimilarity() JavaScript-callable helper with correct numerics — dot(a,b) / (‖a‖·‖b‖) with 1e-8 epsilon guard (95% correct). console_error_panic_hook::init() properly called via Once on first construction (correct initialization pattern). Tests are browser-targeted (#[cfg(target_arch = "wasm32")]) — construction-only, no round-trip coverage (M2). No training APIs exported (H80).

**GNN bindings verdict**: Both FFI layers are genuine wrappers around ruvector_gnn core. The "inference-only" gap (H80) is an architectural limitation — training code exists in the core crate (scheduler.rs, replay.rs, confirmed R94) but is deliberately excluded from the public FFI surface. This is a design choice (deploy-only edge inference), not a facade. The 18th GENUINE WASM brings the ecosystem WASM ratio to 18 genuine / 13 theatrical = 58% genuine.

### 5q. AIDefence Security Module (R92)

**Verdict: PARTIALLY REAL (82-88%) — genuine regex-based security, not AI.** README-REALITY-CHECK previously classified this as UNCOVERED (16 files, 0 DEEP). R92 resolves: the "AI" in AIDefence is misleading.

**AIDefenceGuard.ts (763 LOC, 82-88%):** 28 hand-crafted regex patterns for prompt injection (direct override, role manipulation, system prompt extraction, jailbreak keywords, code injection, data exfiltration), 8 jailbreak-specific patterns (DAN, unlimited mode, GPT-4 jailbreak), 6 PII categories (email, phone, SSN, credit card, IP, API keys), Unicode homoglyph normalization (8 Cyrillic lookalikes — genuine attack defense), control character sanitization, response-side injection compliance (6 response patterns). Clean middleware factory with Zod schema validation. **Not AI**: zero ML models, zero classifiers, zero embeddings. behaviorAnalysis() is a 4-feature stub (length/punct/caps/digits) with hardcoded 2.0 threshold and ephemeral in-memory baseline. enablePolicyVerification config flag has ZERO implementation (ghost feature). The declared `aidefence@^2.1.1` npm dependency is never imported (dead dependency).

**Test suite (280 LOC):** 37 tests across 10 describe blocks exercise regex patterns correctly but miss behavioral analysis entirely (0 test cases), PII false positive rates untested, config impact on blocking behavior untested. Tests reveal the module works as designed — the design is just rule-based, not AI.

**Integration: STANDALONE.** Zero imports from AgentDB, hooks, MCP, claude-flow, or any ruvector internal package. Only external dependency: Zod. Not excluded from npm publish (contrary to earlier speculation) — exported via root barrel, just missing `./security` subpath in exports map (design omission).

### 5r. CUDA-WASM Flash Attention (R92)

**MISPLACED CPU code — 88-92% algorithm quality, 0% CUDA.** flash_attention.rs (528 LOC) in cuda-wasm/src/runtime/ is a pure CPU Flash Attention v2 reference implementation. Zero cuBLAS, zero device memory allocation, zero GPU kernels, zero WASM bindgen. All computation on heap Vec<f32>.

Textbook-correct algorithm: outer Q-tile loop, inner KV-tile loop, online softmax (Milakov & Gimelshein 2018) with running m_i/l_i and correction factor rescaling, correct final normalization with logsumexp, causal masking via kv_limit + inner guard, SRAM budget formula matching paper Section 3.1: (bq×d + 2×bkv×d + bq×bkv)×4. FLOPs counter: 4×seq_q×seq_kv×d (verified by test). forward_multi_head is serial CPU loop over (batch × heads).

7 comprehensive tests: correctness vs naive reference (tolerance 0.1), causal NaN check, memory savings monotonicity, multi-head shapes, FLOPs exact match, dimension error, single-token identity.

**Role**: Almost certainly the algorithmic ground truth / reference implementation used to validate the 997-LOC GPU version in ruvector-mincut-gated-transformer. The "pure math" layer before GPU dispatch is wired.

### 5s. ruvllm Context Module (R104)

**Nine files, ~5,669 LOC, split quality: context/ uses REAL HNSW (88-93%), claude_flow/ uses pseudo-embeddings (70-78%).** R104 characterizes the ruvllm context and reasoning_bank modules tagged to ruvector.

**Architectural finding: ruvllm context module uses GENUINE vector similarity.** semantic_cache.rs and episodic_memory.rs both import ruvector_core::index::hnsw::HnswIndex with DistanceMetric::Cosine. Callers supply Vec<f32> embeddings directly. This CONTRASTS with the claude_flow/ training files (pretrain_pipeline.rs, real_trainer.rs: hash-based) and claude_flow_bridge.rs (CLI-delegated, no vector math).

**context/ module (5 files, ~3,597 LOC):**

- **semantic_cache.rs (88-93%, 675 LOC):** Dual-path lookup: MD5 hash for exact match (O(1)), HnswIndex cosine for semantic match (similarity_threshold=0.85). LRU eviction + TTL. 7 tests. NOT hash-based.

- **episodic_memory.rs (82-88%, 743 LOC):** STANDALONE episodic store with HnswIndex. MemoryCompressor: top-K reward selection + vector mean. compress_embedding() uses truncation not PCA (acknowledged placeholder). Parallel to AgenticMemory.episodic (thin wrapper) — two episodic systems with no composition bridge (H107).

- **working_memory.rs (88-92%, 697 LOC):** VecDeque scratchpad, tool cache, variable store. Time-decay attention exp(-rate*elapsed) but CRITICAL: eviction ignores attention weights (FIFO only, C35). O(n) eviction on overflow. 8 tests.

- **context_manager.rs (82-87%, 794 LOC):** IntelligentContextManager owns AgenticMemory + SemanticToolCache. prepare_context() real pipeline. EpisodicMemory and ClaudeFlowBridge NOT directly wired — accessible only indirectly. 6 tests.

- **claude_flow_bridge.rs (60-68%, 688 LOC):** CRITICAL shell adapter: ALL calls via std::process::Command spawning npx. No EmbeddingService, no vector search, no Rust API. 5th routing surface via get_routing_suggestion() CLI hook (C33, C34). Stats inconsistency on error (H110). Fabricated retrieval timestamps (H112). 7 tests.

**reasoning_bank/ additions (2 files, ~1,366 LOC):**

- **consolidation.rs (90-94%, 736 LOC):** GENUINE EWC++: FisherInformation EMA-of-squared-gradients (Schwarz 2018), apply_constraint() Fisher-weighted damping, regularization_loss() Kirkpatrick eq. 3. Bug: consolidation_count never incremented (H108). consolidate_fisher() lossy trade-off: destroys per-pattern importance (H109). 13 tests.

- **trajectory.rs (88-92%, 630 LOC):** TrajectoryRecorder, 5-variant StepOutcome (hardcoded quality floats). Heuristic verdict-weighted quality scoring (not RL return). 8 tests.

**claude_flow/ additions (2 files, ~702 LOC):**

- **task_classifier.rs (72-78%, 383 LOC):** Pure keyword matcher. Docstring claims "RuvLTRA embeddings" but zero ML. Outputs never consumed by routing infrastructure (H115). Dead-end.

- **flow_optimizer.rs (70-75%, 319 LOC):** Hardcoded latency/memory improvement constants. 18th pseudo-embedding (sinusoidal sweep). SONA delegation real.

**Verdict**: ruvllm context/ uses genuine HNSW-backed semantics (88-93%). The claude_flow/ integration layer falls short: bridge is a subprocess wrapper, task classifier is keyword-only, optimizer has fabricated metrics (60-78%). The runtime context quality substantially exceeds the integration layer quality.

### 5t. Prime-Radiant Coherence Module Complete (R105)

**Five files, ~2,885 LOC, ~89% weighted avg. MODULE COMPLETE (5/5 DEEP).** Implements continuous sheaf Laplacian coherence monitoring for the knowledge graph.

**Architecture:** energy.rs computes E(S)=sum(w_e*|r_e|^2) per coherence state. incremental.rs maintains a dirty-edge HashSet for O(deg(v)) updates. spectral.rs monitors drift via eigenvalue tracking. history.rs maintains rolling time-series with OLS regression and z-score anomaly detection. mod.rs re-exports 24+ symbols.

- **energy.rs (90-93%, 760 LOC, HIGHEST):** Sheaf Laplacian formula correct. SIMD via wide::f32x8 (cfg-gated). Zero Vec allocations in hot path. Blake3 fingerprint snapshots. 5 tests.

- **spectral.rs (85-88%, 738 LOC):** Two code paths: nalgebra feature (correct) vs fallback (BROKEN: C32, H116-H118). CRITICAL: deflate_matrix() subtracts lambda*I not rank-1 Hotelling. Power iteration targets largest eigenvalue, not Fiedler value. detect_drift() noisy single-step delta (H116). spectral_distance() asymmetric (H95). No warning when using broken fallback (H96). 8 tests.

- **incremental.rs (88-92%, 691 LOC):** O(deg(v)) incremental vs O(|E|) full recompute. Adaptive 30% threshold. energy_trend() slope sign inverted (H119). removed_energies dead code. 6 tests.

- **history.rs (88-92%, 617 LOC):** VecDeque rolling window, f64 accumulators, OLS regression, z-score anomaly (3-sigma default). 12 tests. clear() does not reset anomaly counters.

- **mod.rs (95%+, 79 LOC):** Clean re-export, ResidualCache->IncrementalCache alias.

**Critical gap**: spectral.rs broken fallback is silent — any deployment without the `nalgebra` feature computes wrong eigenvalues for all k>1. The nalgebra feature path is correct.

### 5u. ruvector-attention Sheaf Module (R105)

**Four files DEEP (of ~6 total), ~2,505 LOC, ~87% weighted avg.** Attention grounded in sheaf theory: restriction maps (rho(x) = Ax + b) replace W_Q/W_K/W_V. Energy-based coherence attention: A_ij = exp(-beta*E_ij)/Z where E_ij is restriction map residual energy.

- **attention.rs (82-87%, 712 LOC):** Genuine sheaf attention. Three architecture gaps: no backward pass (H98), fixed seed=42 for all instances (H99, shared with restriction.rs), structurally single-head despite num_heads config (H100). compute_energy_matrix() batch method unreachable from forward() (H101). 11 tests.

- **early_exit.rs (90-94%, 651 LOC, HIGHEST test density):** Energy-convergence early stopping (not confidence threshold). EMA smoothing + patience + PerfectCoherence shortcut. process_with_early_exit() generic fn. No type-level binding between energy_fn and sheaf Laplacian (H102). 13 tests.

- **sparse.rs (88-92%, 712 LOC, GENUINELY NOVEL):** Residual-sparse attention using restriction map energy as sparsity criterion. CSR format built but unreachable from compute_sparse() path (H103) -- compute path uses O(n_q*nnz) linear scan. No SIMD. 10 tests.

- **restriction.rs (~85%, 430/519 LOC read):** Linear restriction maps rho(x)=Ax+b. Xavier init with seed=42 (shared seed causes identical initialization across all SheafAttention instances). energy(), residual(), energy_matrix(), apply_batch(), update_weights/update_bias hooks. 11 tests.

**Key distinction from other attention types:** sheaf module grounds attention in algebraic topology (restriction maps, coherence energy) rather than Riemannian geometry (lorentz_cascade, hyperbolic_attention) or graph adjacency (GAT, GraphAttention). Most architecturally novel attention mechanism in the crate after mixed_curvature.rs.

**Current limitations:** No backward pass, identical initialization across all instances, multi-head structurally single-head. Two of approximately six sheaf module files remain unread.

**R107 addition — sheaf/router.rs (85-90%, 666 LOC):** TokenRouter uses SheafAttention as the routing mechanism — restriction map energy scores determine which processing tier a token receives. Three threshold-based tiers: Reflex (E < theta_reflex), Standard (E < theta_standard), Escalation (E < theta_escalation), plus a Deep Contemplation tier with dead theta_deep field (H150). Confidence hardcoded 1.0 (H151). tune_thresholds() exists but structurally decoupled — no internal feedback accumulation (H152). SONA adaptive interface: update_sona_model() implemented but never auto-called on routing decisions. **Sheaf module now 5/6 DEEP (R107 completes the actionable coverage).**

### 5v. ruvllm Serving Module Complete (R107)

**Six files DEEP, ~4,570 LOC, ~90% weighted avg. SERVING MODULE COMPLETE.** The ruvllm serving/ module implements a production LLM inference serving pipeline.

**serving/request.rs (88-92%, 473 LOC, R107):** Genuine vLLM request lifecycle — chunked prefill with block allocation, KV block acquisition/release, preemption state tracking (Running/Preempted/Finished). GenerateParams carries all inference hyperparameters. **HIGH (H163)**: stop_sequences silently ignored — should_stop() accepts _decoded_text but never inspects it. Only EOS token and max_new_tokens checked. Requests with stop_sequences generate to max limit.

**serving/mod.rs (92%, 348 LOC, R107):** Orchestration hub exposing ServingEngine. 4 integration tests at lines 168-348 directly compose ServingEngine + ContinuousBatchScheduler + KvCacheManager + RequestQueue — production system-level integration confirmed. NoopBackend implements BackendTrait returning zeroed logits — confirms clean decoupling between serving infrastructure and model weights.

**Module completeness:** scheduler.rs (R35, 90-92%) + engine.rs (R35, 80-85%) + paged_attention.rs (R35, 75-80%) + batch.rs (R106, 90-95%) + kv_cache_manager.rs (R106, 88-92%) + request.rs + mod.rs (R107). All six files DEEP.

**Serving verdict**: Production-quality continuous batching and KV management. Integration gap: serving infrastructure is well-designed but stop_sequences and several preemption modes are incomplete. The system is production-viable for max_new_tokens-bounded generation but not for instruction-following generation that relies on stop tokens.

### 5w. ruvllm LoRA Adapters and Merge Module (R107)

**Three files DEEP across lora/adapters/, ~1,501 LOC combined, ~78% weighted avg.** R107 completes the ruvllm LoRA subsystem characterization.

**lora/adapters/merge.rs (72-78%, 631 LOC, R107):** Four merge strategies (Average, Weighted, TaskArithmetic, SLERP, DARE) at the API surface, but THREE are incorrect: (1) SLERP=LERP (C41) — plain (1-t)*A + t*B, not geodesic interpolation. (2) TaskArithmetic unconditionally delegates to WeightedSum (H154) — forfeits the task vector construction step. (3) DARE seed=42 fixed (H155) — same pruning pattern every call defeats Monte Carlo guarantees. (4) All strategies merge A and B matrices independently (H156) — wrong for mixed-rank adapters. (5) No LoRA alpha scaling applied (H157) — merged weights off by rank/alpha ratio throughout.

**lora/mod.rs (90%, 123 LOC, R107):** Pure re-export of adapter.rs and training.rs. Propagates R106 forward_sequential() math bug and dead GradientAccumulator through public API — users importing lora:: namespace inherit both broken integration patterns.

**LoRA subsystem verdict**: micro_lora.rs (R37, 92-95%) is the BEST training code in the ecosystem. The adapter management layer (adapter.rs R106, training.rs R106) has correct EWC math but dead accumulator paths. The merge module (merge.rs R107) has five correctness issues and one CRITICAL SLERP misimplementation. The LoRA subsystem is strongest at the algorithm level and weakest at the composition/merge level.

### 5x. ruvllm Agent Router and claude_flow Integration (R107)

**agent_router.rs (72-77%, 311 LOC, R107):** The 6th routing surface in ruvllm's claude_flow module. AgentRouter combines keyword matching (task type → AgentType enum, 8 variants) with SONA-guided routing. **CRITICAL (C40)**: SONA feedback is broken at the type level — record_feedback() writes AgentType (enum index 0-7) into SONA's model_index field, which semantically expects 0=high-quality, 1=medium, 2=low quality tier. AgentType::Coder=0 is treated as "high quality tier" — wrong. Additionally, response_embedding is a copy of query_embedding (comment: "Simplified") — SONA receives identical query/response embeddings, making it impossible to learn query→response representation transitions. Every SONA pattern record carries the wrong model quality label and zero response signal.

**Integration positioning**: AgentRouter sits parallel to HnswRouter (R37, semantic HNSW), ModelRouter (R37, 7-factor complexity), ruvector_integration.rs (R106, SONA>HNSW>keyword three-tier), and claude_flow_bridge.rs (R104, CLI-subprocess). No composition bridge between any pair. **This is the 6th independent routing surface.**

### 5y. Prime-Radiant Cohomology Module Complete (R109+R110)

**Module status: 9/9 DEEP, ~83% weighted average.** The prime-radiant cohomology module (cocycle, cohomology_group, diffusion, laplacian, mod, neural, obstruction, sheaf, simplex) is now fully deep-read across all nine submodules.

**Algorithmic quality**: The core algebraic topology machinery is genuine. simplex.rs (90-94%) implements Bron-Kerbosch clique enumeration and the boundary operator correctly. cohomology_group.rs (85-90%) computes genuine H^n(K,F): boundary matrix, RREF kernel, Gram-Schmidt quotient, 5 canonical tests. obstruction.rs (85-90%) detects H^1 obstruction via sheaf Laplacian energy with MinCut-aware remediation. neural.rs (88-93%) implements a valid 5-step SheafNeuralLayer pipeline (Xavier init, diffusion, activation, residual, LayerNorm). diffusion.rs (78-82%) SheafDiffusion heat kernel is correct for the identity-restriction case.

**Systemic CRITICAL: THREE broken Laplacian implementations**. All three share the same root-cause class — eigenvalue power iteration defects:
- laplacian.rs (C42): uniform-init power_iteration trivially converges to eigenvalue 0.
- laplacian.rs (C43): finds LARGEST eigenvalue instead of smallest (Fiedler value) — sign flip absent.
- cocycle.rs adjoint (C44): apply_adjoint() iterates over wrong degree (n not n+1), composing a wrong Hodge Laplacian.
- coherence/spectral.rs (C32, R105): fallback deflation subtracts λ*I instead of Hotelling rank-1.

All three Laplacian objects (laplacian.rs HodgeLaplacian, cocycle.rs Hodge composition, coherence/spectral.rs fallback) are independently broken with no shared code, indicating the mathematical error was independently replicated rather than inherited. Any code path depending on spectrum computation produces wrong eigenvalues silently; tests pass accidentally because energy() is correct while spectrum() is not.

**Identity-restriction limitation (systemic)**: All three files that accept RestrictionMap objects at runtime (laplacian.rs H172, neural.rs H166, diffusion.rs H189) ignore the actual restriction maps and fall back to identity operations. The RestrictionMap API exists and is structurally consistent, but none of the three users invoke it in the hot path. This means the algebraic topology is R-coefficient only — functorial sheaf structure is not exercised.

**sheaf.rs architectural isolation (C46+C47)**: The SheafComplex implementation has a semantically broken gluing axiom (edge stalks are reconstructed per-call rather than shared) and is architecturally isolated — SheafLaplacian, CohomologyGroup, and SheafNeuralLayer all build their own internal structures rather than consuming SheafComplex. Zero pipeline consumers confirmed. The module's central abstraction is a dead end.

**cocycle.rs coboundary chain bug (C45)**: is_coboundary() always returns false for degree n≥1 because the image-in-kernel test is inverted. This inflates all H^n computations beyond H^0 throughout the module.

**Module verdict**: The cohomology module is architecturally genuine (all nine submodules present with correct type signatures and mathematical structure) but contains interlocking algorithmic bugs: three broken Laplacians, broken is_coboundary(), identity-only restriction maps, and an isolated central SheafComplex. Any production use of spectrum analysis, coboundary checking, or functor composition would produce incorrect results. The module is ~83% real by code volume but significantly less reliable for its stated algebraic topology purpose.

### 5z. ruvector-attention Training and Transport Modules (R109+R110)

**Training module: 4 files, ~1,470 LOC, ~89% weighted average.**

optimizer.rs (88-92%), loss.rs (88-92%), curriculum.rs (90-95% HIGHEST), mining.rs (88-92%) form a complete metric-learning training pipeline. All core algorithms are textbook-correct: SGD/Adam/AdamW optimizers, InfoNCE loss with stable log-sum-exp, curriculum learning with 4 temperature schedules (SGDR cosine correct), and 4 hard-negative mining strategies (FaceNet semi-hard correct). All files are pure scalar — no SIMD or Rayon parallelism despite the training context.

Key defects: Nesterov momentum formula double-counts momentum (optimizer.rs); Adam weight_decay path is dead code. SpectralReg loss computes zero gradient every step (H176, loss.rs). Exponential temperature schedule produces NaN when final_temp=0.0 (H177, curriculum.rs). seed=42 is hardcoded across mining.rs — "random" sampling is deterministic in all runs (H178). InBatch and HardNeg miners operate on disjoint data structures with no shared negative pool (H179).

Despite these bugs the training module is the most complete learning-pipeline implementation in ruvector — four files covering optimizer, loss, curriculum, and mining with correct mathematical foundations and test coverage. The bugs are correctness issues (wrong loss gradient, NaN edge case, deterministic randomness) rather than structural gaps.

**Transport module: 3 files, ~1,130 LOC, ~83% weighted average (effectively complete).**

Both namesake algorithms are present as dead code:
- sliced_wasserstein.rs (72%): The `compute_distance()` method computes projected-L2 not Sliced Wasserstein (C50). The correct distributional OT implementation exists in the file as unreachable dead code (H190). Histogram and CDF infrastructure is built and populated but never consumed in the active distance path (H192).
- centroid_ot.rs (88-93%): The Sinkhorn iteration is dead code with a logic bug (H193). The active path uses a correct k-means + softmax approximation but not optimal transport.

cached_projections.rs (88-92%) is the strongest transport file — genuine projection caching, pre-sorted windows, histogram CDFs — and its infrastructure is imported by both sibling files but the data it produces flows into the wrong distance computations.

Both transport files ship the infrastructure for their namesake algorithms (OT histograms, Sinkhorn skeleton, random projections) but execute cheaper approximations in the active code paths while keeping the correct algorithms as dead code. Zero integration with the training pipeline — transport distances are not used as loss functions or mining criteria.

**Combined assessment**: The ruvector-attention training + transport modules are the most complete standalone ML training subsystem in the codebase, with no dependency on any other ruvector module (pure Rust, scalar math, no SIMD, no HNSW). The quality gradient is training (89%) > transport (83%). Both modules could be connected to the attention training loop with targeted fixes to the dead-code transport paths.

### 5aa. ruvector npm Umbrella Package (R117) — ONNX + RL + Intelligence Orchestration

**5 files, ~2,964 LOC, ~79% weighted average. R116 correction confirmed + extended.**

The ruvector npm umbrella package (`npm/packages/ruvector/`) is the JavaScript-facing layer of the ecosystem. R116 live testing confirmed VectorDBWrapper works with native HNSW. R117 deep-reads reveal the full architecture:

**VectorDBWrapper (index.ts, 85-88%)**: The entry point that WORKS. Synchronous `require('@ruvector/core')` at module load — no async races, no deferred init. Proper await on all methods. Complete metadata JSON round-trip. Kitchen-sink re-exports 22+ core modules (GNN, SONA, ONNX, attention, learning, graph, tensor, AST, etc.).

**ONNX Embedder (onnx-embedder.ts, 85-90%)**: **MAJOR CORRECTION** — this is a REAL neural embedding path using Tract inference engine compiled to WASM (7.4MB binary). Downloads all-MiniLM-L6-v2 from HuggingFace, tokenizes via WASM tokenizer, produces genuine 384-dim Float32Array embeddings. 6 pre-configured models (384d-768d), 5 pooling strategies, L2 normalization, SIMD detection, parallel batch support. 4 confirmed consumers.

**Intelligence Engine (intelligence-engine.ts, 72-78%)**: Master orchestrator composing FastAgentDB, SonaEngine, OnnxEmbedder, ParallelIntelligence, VectorDB, and Attention. Provides 5 pipelines: memory (remember/recall), routing, trajectory learning, episode learning, pattern learning. **CRITICAL**: sync `embed()` ALWAYS falls through to `hashEmbed()` even when ONNX is configured — only `embedAsync()` reaches real ONNX. This means route(), recordEpisode(), beginTrajectory() all operate on hash embeddings. VectorDB/Map desync after import (C79). Constructor race conditions (C80).

**Learning Engine (learning-engine.ts, 78-82%)**: 9 RL algorithms, all tabular (Map-based Q-tables). Quality is trimodal: textbook (Q-Learning, SARSA, Double-Q, Monte Carlo, TD-lambda: 88-95%), partial (Actor-Critic: 72-78%), broken/facade (PPO: 40-50% wrong ratio, Decision Transformer: 25-35% zero attention, DQN: 55-65% no neural net). Actively consumed by CLI train/benchmark/simulate commands + MCP server. Corrects "dead RL" from R20.

**Test Suite (ruvector.test.js, 72-78%)**: 22 vitest tests all run against JS Map + cosine fallback. `@ruvector/core` not in package.json — structurally cannot reach native HNSW. Good error-path coverage but validates the wrong integration target.

**The integration gap**: ONNX embeddings exist and are real, but the orchestrator's sync path always falls through to hash. The test suite never exercises native HNSW. VectorDBWrapper works perfectly but its consumers (intelligence-engine) have race conditions and hash fallbacks. Net result: the pieces are genuine but the wiring has 3 CRITICAL bugs preventing the full pipeline from functioning as intended.

### 5ab. RVF File Format Subsystem (R121+R123+R124)

**15 files, ~7,044 LOC, ~156 findings. Rust Crates 88-93% GENUINE; TS SDK 78-90% with type mismatches.**

The RVF (RuVector Format) subsystem spans 8 crates (rvf-node, rvf-runtime, rvf-crypto, rvf-wire, rvf-types, rvf-kernel, rvf-launch, rvf-manifest) plus an npm SDK package (npm/packages/rvf/), implementing a custom binary vector database format with cryptographic tamper-evidence, VM execution, and a TypeScript consumer API.

**Wire format (rvf-wire, writer.rs + reader.rs)**: 64-byte fixed header containing magic (`0x52564653` = "RVFS"), version (currently 1), segment type, flags (12 bits), segment ID, payload length, timestamp, checksum algorithm, compression algorithm, content hash (XXH3-128), uncompressed length, and alignment padding. Payload follows immediately, zero-padded to next 64-byte boundary for AVX-512/cache-line alignment. The format is self-describing at the container level but requires SegmentType knowledge for payload interpretation. Writer and reader are symmetric (confirmed byte-by-byte). `#[repr(C)]` struct with compile-time size assertion (ABI stability across FFI). CRC32C deprecated with silent upgrade to XXH3-128. SEALED flag bypass attack explicitly prevented (security-aware anti-bypass comment in reader).

**Storage engine (rvf-runtime, read_path.rs)**: Segment-based with manifest discovery via backward scan from EOF (64KB tail chunk). Binary protocol parser: 22-byte header + 25-byte directory entries + deletion bitmap + FileIdentity trailer. Vector segment deserializer: dim/count header + id/f32-vector pairs with size validation. Progressive read boot: L0 manifest -> hotset mmap -> background L1 fill -> on-demand cold read. Unix-optimized with `pread` (avoids seek overhead for concurrent reads). MEDIUM: CRC32 rotation "expansion" to 16 bytes is cosmetic, not real 128-bit hash. Bit-at-a-time CRC32 (~8x slower than table lookup). Compression field parsed but never acted upon.

**COW snapshots (rvf-runtime, cow.rs)**: Genuine cluster-level copy-on-write with three-state CowMap (LocalOffset/ParentRef/Unallocated). Write coalescing buffers mutations and copies parent cluster ONCE on flush. SHAKE-256-256 witness trail on every COW copy (event type 0x0E). Unix `pread` optimization (FileExt::read_exact_at). Freeze semantics prevent data loss (rejects unflushed writes, requires empty write buffer). Not a full clone — operates at cluster granularity. MEDIUM: L0 cache unbounded growth (no eviction), L0 cache not populated on LocalOffset read path. 6 well-structured tests.

**Witness chain (rvf-crypto, witness.rs)**: SHAKE-256-256 (SHA-3 family, NIST-standardized, quantum-resistant to 128-bit) hash chain for tamper-evidence. Each witness entry chains to the previous via `prev_hash = SHAKE-256(previous entry bytes)`. Creation and full-chain verification both implemented and tested. `verify_witness_chain()` IS a correct verifier — it recomputes every hash in the chain. 6 tests covering empty/single/multi/tamper/truncation/links. HIGH: `is_multiple_of()` is nightly-only unstable API — fails on stable Rust. MEDIUM: No timestamp monotonicity check — backdated witnesses undetectable.

**NAPI bridge (rvf-node, lib.rs)**: 22-method napi-rs bridge exposing `RvfDatabase` to Node.js. All methods delegate to real `rvf_runtime::RvfStore`. Thread safety via `Mutex<Option<RvfStore>>`, zero unsafe in public API. Includes complete recursive JSON filter parser (9 operators: $eq/$ne/$gt/$gte/$lt/$lte/$in/$nin/$exists, typed values, $and/$or/$not boolean combinators) and kernel/eBPF embedding passthrough. **CRITICAL gap (C84)**: `verify_witness()` does NOT call the real `verify_witness_chain()` from rvf-crypto. It only checks for a non-zero terminal hash — giving Node.js consumers false security assurance about tamper-evidence. The CLI, WASM, and claude-flow adapters all call the real verifier correctly. This is a one-line fix. Confirmed by R123 reading the actual rvf-crypto source. HIGH: `index_stats().layers` hardcoded to 0. MEDIUM: dimension u32->u16 truncation risk, RvfQuantConfig defined but unused, needs_rebuild heuristic arbitrary.

**Attestation (rvf-crypto, attestation.rs, 840 LOC, 75-80%, R124)**: SHAKE-256 hash layer is GENUINE (sha3 crate, XOF interface, verified in Cargo.toml). Witness chain integration correctly applies shake256_256 for action_hash binding. However, TEE attestation verification is a STRUCTURAL FACADE: `QuoteVerifier` trait defines the interface (SGX, SEV-SNP, TDX, ARM CCA) but zero concrete implementations exist anywhere in ruvector. `verify_key_binding()` performs structural checks only (platform enum match + measurement byte equality) — no cryptographic verification of sealed_key_material, no unsealing operation, no signature check. Same unverified-crypto pattern as policy.rs (C36) and mcp-gate/types.rs (R114). 16 high-quality unit tests. No hardcoded secrets.

**Kernel image builder (rvf-kernel, lib.rs, 862 LOC, 82-87%, R124)**: KEY CORRECTION — "micro-Linux VM" is MISLEADING. This is a kernel image build toolkit, not a VM runtime. Three build paths: (1) `from_prebuilt()` reads any bzImage/ELF/PE from disk, (2) `build_docker()` orchestrates real Alpine Docker + `make bzImage` with 416-directive expert-level .config (VirtIO, eBPF+JIT, KASLR, namespaces, io_uring), (3) `from_builtin_minimal()` generates 4096-byte valid boot protocol 2.15 halt stub. `KernelVerifier` uses real SHA3-256 against embedded image_hash. `initramfs.rs` produces genuine cpio/newc archives. CRITICAL: crate has ZERO callers — neither rvf-launch nor rvf-cli import it. Architecturally orphaned. Silent fallback to halt stub (H201), zero UUID/timestamp (H202), unsigned source download (H203).

**VM launcher (rvf-launch, lib.rs, 718 LOC, 85%, R124)**: Genuine QEMU process spawner. Constructs complete QEMU command: `-machine microvm,accel=kvm|tcg`, `-kernel vmlinuz -initrd initramfs`, virtio-blk (RVF readonly) + virtio-net (SLIRP+port forwarding), QMP management socket. Multi-arch: x86_64/aarch64/riscv64. Real multi-step shutdown (QMP powerdown → quit → SIGTERM → SIGKILL). KVM check is path-exists only (`/dev/kvm`), not ioctl. NOT Firecracker-style — delegates all VM execution to QEMU. CRITICAL GAP: zero host-side isolation (no seccomp, no namespaces, no capability dropping). RAII lifetime management for temp files; Drop kills process.

**Root manifest (rvf-manifest, level0.rs, 549 LOC, 88%, R124)**: Fixed 4096-byte footer at EOF of every RVF container — the structural anchor. Contains 4-byte magic + version + flags, Level1 pointer (offset+length), 6 typed segment pointer-triples (entrypoint, toplayer, centroid, quantdict, hot_cache, prefetch_map), FileIdentity lineage block (UUID + parent_hash[32] + lineage_depth), COW extension block (cow_map_offset, membership, snapshot_epoch, double_root). CRC32C integrity over 4092 bytes. Not OCI-compatible, not Merkle tree — flat CRC. Signature buffer stored but NOT VERIFIED (H200). 7 comprehensive tests including round-trip, corruption detection, COW pointers.

**TypeScript SDK (npm/packages/rvf/, 5 files ~1,533 LOC, 78-90%, R124)**: User-facing API with dual-backend strategy. `NodeBackend` (genuine NAPI delegation with bidirectional string⟷numeric id mapping via `.idmap.json` sidecar) and `WasmBackend` (partial facade — 9/20 interface methods throw, BigInt(UUID) crashes C88). `RvfDatabase` is a clean delegation wrapper. `errors.ts` defines 30 error codes in 7 categories with 1:1 Rust mapping — Tile WASM category (0x04) corroborates microkernel architecture. `types.ts` has 18 exports with 8 type mismatches vs Rust NAPI (C91/C92, H212-H215). `index.ts` barrel re-export has hard rvf-solver dependency that crashes if absent (H216). `resolveBackend('auto')` picks NodeBackend first with no WASM fallback (H210).

**Incomplete areas**: Compression (LZ4/ZSTD/custom) header fields exist in wire format but no implementation in runtime. CRC32 path is bit-at-a-time (~8x slower than table lookup). L0 cache has unbounded growth (no eviction). No timestamp monotonicity in witness chain. rvf-types crate not yet deep-read. rvf-kernel architecturally disconnected. WasmBackend ~55% functional. 8 TS/Rust type mismatches need reconciliation.

### 5ac. Domain Expansion Crate (R115)

**14 files, ~6,396 LOC, all DEEP. Crate avg ~84%. Standout: rvf_bridge.rs 88-92%.**

The ruvector-domain-expansion crate implements an automated domain mastery system: given a new problem domain (planning, Rust synthesis, tool orchestration), it generates tasks, evolves solution populations via Thompson Sampling + evolutionary search, and measures acceleration (does each new domain converge faster than the last?). This is the "IQ growth" mechanism — the system's ability to learn new domains progressively faster.

**Architecture — lib.rs (85-90%):** DomainExpansionEngine composes 5 subsystems cleanly: Domain trait objects (extensibility), MetaThompsonEngine (Bayesian arm selection for transfer), PopulationSearch (evolutionary optimization), AccelerationScoreboard (convergence metrics), and MetaLearningEngine (regret+plateau+pareto+curiosity). evaluate_and_record() calls thompson.record_outcome() on every evaluation — the FIRST working Thompson Sampling feedback loop in ruvector. AgentDB's SolverBandit (C40) implements the same algorithm but recordReward() is never called in production.

**Domain trait — domain.rs (85%):** Send + Sync trait requiring generate_tasks(), evaluate(), embed(), embedding_dim(), reference_solution(). Three concrete implementations: RustSynthesisDomain, PlanningDomain, ToolOrchestrationDomain. Each defines its own task generation, evaluation metrics, and embedding strategy.

**Meta-learning — meta_learning.rs (82-87%, 1,399 LOC):** Largest file in crate. Implements 4 meta-learning signals: cumulative regret tracking across domains, plateau detection (stalled improvement), Pareto frontier management (multi-objective trade-offs), and curiosity-driven exploration (preference for under-explored domains). These signals feed into MetaThompsonEngine arm selection.

**Transfer learning — transfer.rs (85-88%):** init_domain_with_transfer() uses sqrt-dampened priors — alpha_new = 1 + sqrt(alpha_old - 1). Principled: preserves prior mean but reduces confidence, preventing over-commitment to cross-domain evidence. Cost EMAs pessimistically scaled 1.5x. Beta distribution sampling uses Abramowitz-and-Stegun normal approximation (adequate but inexact — M11).

**Policy kernel search — policy_kernel.rs (85-88%):** Population-based evolutionary optimization with 8 tunable kernel parameters. Standard GA: 25% elitism, tournament selection, 30% crossover rate, Gaussian mutation. Correctly implemented but architecturally simple.

**Cost curve tracking — cost_curve.rs (85-88%):** AccelerationScoreboard computing AUC via trapezoidal integration, compression ratios, and convergence thresholds (accuracy >= 0.95, robustness >= 0.90, zero violations). progressive_acceleration() is the core IQ test — checks if each successive domain converges faster than the last.

**Concrete domains:**
- **rust_synthesis.rs (75-80%):** LOWEST in crate. Embeddings are 64-dim feature-counting vectors (pattern occurrence counts for .map(, for, unsafe, etc.) — the hash-based pattern but more structured with 8 feature groups (H218). Evaluation is heuristic-only: no compilation, execution, or testing (H219). reference_solution() only covers Transform category — 60-70% of task types lack baselines (M10).
- **planning.rs (82-85%):** Evaluation checks goal coverage and dependency ordering. Goal matching is exact string — no partial satisfaction (M9). Pipeline validation genuine.
- **tool_orchestration.rs (85-88%):** Most sophisticated evaluation: validates type chain (output_type matches next input_type), estimates latency (parallel groups take max, sequential sum), computes cost with retry multiplier, checks error scenario coverage.

**RVF bridge — rvf_bridge.rs (88-92%, STANDOUT):** Three key capabilities: (1) SHAKE-256 witness hashing via rvf_crypto — first verified cryptographic hashing in domain-expansion, using the same chain verified in R123. (2) SolverPriorExchange bidirectional bridge — extract_solver_priors() converts domain-expansion bucket priors to solver-compatible flat keys, import_solver_priors() reverses. This could theoretically fix the broken AgentDB SolverBandit feedback loop (C40). (3) AGI container TLV binary packaging with encode_tlv_entries() using [tag:u16 LE][length:u32 LE][value:N bytes] wire format. 10 tests including witness chain verification and multi-segment assembly with 64-byte alignment.

**WASM — domain-expansion-wasm/lib.rs (85-88%):** 19th genuine WASM module in ruvector. Full WasmDomainExpansionEngine via wasm_bindgen exposing: generateTasks, evaluateAndRecord, selectArm, initiateTransfer, verifyTransfer, evaluatePopulation, evolvePopulation. Uses serde_wasm_bindgen for JSON exchange. WasmRvfBridge (behind rvf feature) exports RVF serialization + SHAKE-256 witness to browser/edge.

**Postgres integration — mod.rs (85%) + operators.rs (72-78%):** Thin but real: pg_extern ruvector_domain_transfer() with global engine state via DashMap<Arc<RwLock<DomainExpansionEngine>>> (correct concurrent access for pgrx). However, the operator only initiates transfer and returns status JSON — no population evolution, no cost curve tracking (M12). PostgreSQL is an entry point, not a training loop.

**Key pattern:** domain-expansion is the first crate where Thompson Sampling actually works end-to-end. The Bayesian machinery exists in 3+ other locations (AgentDB SolverBandit, reasoning_bank, RL algorithms) but none have the feedback loop wired correctly. rvf_bridge.rs provides the missing integration point (SolverPriorExchange) that could unify these parallel systems.

### 5ad. npm CLI Entrypoints (R135)

**4 CLI files, ~13,055 LOC total, avg ~75%. The "front door" to the ruvector ecosystem.**

The npm CLI layer is the primary user-facing interface for the ruvector ecosystem. R135 characterizes all four CLI entrypoints across three npm packages (ruvector, rvlite, ruvllm), revealing that the CLI tier inherits and amplifies the hash-embedding wiring failure documented since C1, while also exposing genuine capabilities.

**ruvector CLI — cli.js (7,357 LOC, 72-78%):** The largest single file in the npm layer. 14 top-level commands: store/search/delete/info/benchmark/embed/graph/router/server/cluster/intelligence/hooks/mcp/onnx. Critically, the CLI uses VectorDB (confirmed working R117), NOT RuVectorBackend.ts (broken). The ONNX embed subcommands (text/adaptive/benchmark/optimized/neural) are genuine and produce real 384-dim vectors when the ONNX model is available. However, Intelligence.embed() — used pervasively by hooks and intelligence commands — only invokes the embedding engine if it is ALREADY initialized; most hook invocations pass skipEngine:true, falling through to 64-dim hash (C104). Four commands (graph, router, server, cluster) are facades printing "Coming Soon" (H244). Export vectors returns an empty array (H245). The ~4,000-line hooks/intelligence subsystem implements pattern detection, learning, and trajectory tracking — all functional but operating on hash embeddings.

**ruvector MCP server — mcp-server.js (3,007 LOC, 78-82%):** The largest single MCP server in the ruvector ecosystem with 55 tools in a monolithic 3,060-line switch statement (H246). This is separate from the agentdb-mcp-server.ts (R20) and mcp-gate crate (R114). The server does NOT initialize EmbeddingService, repeating the R20 pattern (C105 — 9th hash instance). Delegation is heterogeneous: 14 tools delegate via execSync("npx ruvector hooks ...") incurring 2-5s cold-start per call (H247), 11 tools delegate to "npx agentic-flow@alpha" (H248), and the remaining tools use direct SDK calls. Query sanitization (sanitizeShellArg) strips (), ;, ', ", $, {} — destroying valid SQL/Cypher/SPARQL queries before passing to rvlite (C106). The sanitization is also not a real injection defense since it strips shell metacharacters, not SQL injection vectors.

**rvlite CLI — cli.js (1,686 LOC, 80-85%):** A completely independent vector database CLI with ZERO shared code with the main ruvector CLI (H250). Implements flat brute-force O(n) search (H249) — no HNSW index whatsoever. Genuine WASM integration delegates to SONA and Attention modules via real wasm_bindgen. Hyperbolic geometry (Poincare ball and Lorentz model) is mathematically correct with proper curvature scaling. Advertises SQL, Cypher, and SPARQL query support but the CLI provides none — all deferred to the SDK layer (H251). HuggingFace model download commands are real.

**ruvllm CLI — cli.js (1,005 LOC, 72-78%):** Entirely dependent on the native .node binary. Without it, query/generate/route/embed all return hardcoded or hash values (C107) — the entire CLI is non-functional in JS-only mode. Training is explicitly simulated with "Training loss (simulated)" printed to console, with no actual gradient descent or model update (C108). The SIMD benchmark is genuine, and HuggingFace downloading is real. Zero ruvector RAG integration — no HNSW, no vector store, no @ruvector imports (H253). The embedding benchmark uses hash fallback, making all benchmark results meaningless without native (H252).

**Cross-CLI patterns:** (1) Hash embedding penetration is now confirmed at ALL layers — Rust core (C1), TypeScript engine (C78), CLI hooks (C104), MCP server (C105). (2) The rvlite CLI is architecturally isolated — it could be extracted to its own repo with zero impact. (3) The ruvllm CLI's native dependency makes it the most fragile entrypoint — a missing .node file renders it entirely non-functional. (4) The MCP server's heterogeneous delegation (SDK + execSync + agentic-flow) creates three distinct failure modes and latency profiles.

### 5ae. ruvector PostgreSQL Setup Command (R138)

**1 file, 784 LOC, ~85-90%. Pure scaffolding — no runtime integration.**

The `claude-flow ruvector setup` command (setup.ts, file ID 15447) is a code-generation command that writes three files to the user's project directory: a Docker Compose configuration, a PostgreSQL initialization script, and a README quickstart guide. It does NOT install a native ruvector binary, does NOT configure the backend factory, and does NOT modify any runtime configuration.

**Docker Compose — docker-compose.yml (53 lines generated):** Defines a single `ruvector-postgres` service using the `ruvnet/ruvector-postgres:latest` Docker image with a companion pgAdmin container. Hardcoded test credentials (POSTGRES_PASSWORD=claude-flow-test) in the template (M13). Exposes ports 5432 (Postgres) and 8080 (pgAdmin). Volume-mounts init-db.sql for automatic schema initialization on first container start.

**SQL Schema — init-db.sql (476 lines generated):** Production-quality PostgreSQL schema. 8 tables (vectors, documents, memories, sessions, graphs, hyperbolic_spaces, temporal_data, embeddings) with proper column types including vector(384) and jsonb. 6 HNSW indices using pgvector operators (vector_l2_ops, vector_cosine_ops, vector_ip_ops) with tuned m=16, ef_construction=64 parameters. 7 stored functions for similarity search, hybrid search (combining vector similarity with metadata filtering), memory consolidation, temporal queries, and graph traversal. The schema calls extension C functions (ruvector_exp_map, ruvector_poincare_distance) that require the ruvector PostgreSQL extension to be installed in the Docker image — version mismatch between control file (2.0.0) and SQL references (0.1.0) causes function resolution failure (H255).

**Integration gap:** The setup command creates the infrastructure for a PostgreSQL-backed ruvector deployment, but the 3-step manual process (setup files -> docker-compose up -> run init-db.sql) STILL does not switch the default backend from sql.js to PostgreSQL. The backend factory (factory.ts, R137) has a 5-tier fallback that defaults to sql.js when no native backend is detected. Setup.ts does not write any configuration file, environment variable, or backend selection that factory.ts would read. Even after a fully successful setup + init, `claude-flow memory` commands continue using in-process sql.js unless something else intervenes.

**Zero relationship to memory-initializer.ts (H256):** These are completely independent subsystems. Memory-initializer.ts (deep-read in memory-and-learning domain) configures in-process backends for the V3 memory layer. Setup.ts generates Docker scaffolding for an external PostgreSQL deployment. Neither references the other. This is the pattern of parallel non-composing subsystems seen throughout ruvector (C24, C8, etc.).

**File composition:** Of the 784 LOC, approximately 660 lines are embedded template strings (SQL, YAML, Markdown). Only ~120 lines are actual TypeScript logic (commander option parsing, fs.writeFileSync calls, directory creation). The command repeats the "150x-12,500x" performance claim (M14) traced to sona-agentdb-integration.ts in R137.

### 5af. Integration Testing & Deployment (R139)

**3 files, ~3,126 LOC, avg ~52%. The integration testing layer is a facade over individually genuine algorithms.**

R139 deep-reads the test files that should validate cross-crate integration and distributed deployment. The central finding is that BOTH "integration" test suites are 100% mock-only, and the distributed deployment infrastructure uses shell-script facades instead of real Rust binaries. Individual algorithms ARE genuinely tested, but no cross-crate composition path has ever been validated end-to-end.

**ruvllm E2E integration test — e2e_integration_test.rs (1,535 LOC, 60-65%):** The "E2E" label is a MISNOMER. This is a mock-backend unit test suite for ruvllm-internal modules. 36 test functions use MockTokenizer (20 hardcoded tokens) and MockLlmBackend (deterministic hash-based generation). What IS genuinely tested: softmax, log_softmax, top_k_filter, top_p_filter, sample_from_probs (REAL production functions with mathematical property verification — sum=1.0, ordering preservation, seeded determinism). TwoTierKvCache with concurrent 4-thread access and migration triggers. SpeculativeDecoding pipeline with SpeculativeConfig, SpeculativeStats, AtomicSpeculativeStats, SpeculationTree, TreeNode — all real data structures. GGUF v3 parser format confirmed at byte level via manual header construction. ServingEngine uses poll-based processing (run_iteration loop). What is NOT tested: hnsw_router, claude_flow integration, NAPI bridge, SIMD paths — zero cross-crate imports. 3 #[ignore] tests gated on a real GGUF model file still don't perform actual inference (reads magic bytes, checks env var, does memory math). 12 lint categories suppressed via #[allow]. Unsafe pointer cast UB at L840: Arc::as_ptr() as *mut MockLlmBackend (H259).

**prime-radiant ruvllm integration tests — ruvllm_integration_tests.rs (1,393 LOC, 40-45%):** The MOST misleading file name in the test suite. `#![cfg(feature = "ruvllm")]` gate controls compilation but the file has ZERO imports from either `prime_radiant::` or `ruvllm::` — the gate creates a false appearance of cross-crate testing (H257). 25 tests across 7 modules all use inline reimplementations of 5 subsystems: SheafCoherenceValidator (mock: pairwise cosine similarity; real: CoherenceEngine + SheafGraph + PolicyBundleRef + CoherenceGate), PatternToRestrictionBridge (mock: Vec<LearnedPattern>; real: ReasoningBank + SheafGraph rho maps, 973 LOC pattern_bridge.rs), UnifiedWitnessLog (mock: simple hash chain; real: GenerationWitness linking inference+coherence), MemoryCoherenceLayer (mock: linear scan + negation detection; real: likely complex), CoherenceConfidence (mock: sigmoid mapping; real: closest match). Tests validate the CONCEPT of coherence-gated validation, not actual production code (H258). The algorithmic content (cosine similarity, Blake3 hash chains, sigmoid mapping, negation detection) is genuine — but tests prove concepts, not integration.

**Distributed Raft test cluster — docker-compose.yml (198 LOC, 50-55%):** A 5-node Raft cluster with well-specified parameters: static IPs (172.28.0.10-14), 3 ports per node (7000 Raft consensus, 8000 cluster, 9000 replication), 64 shards, replication factor 3, election timeout 150-300ms, heartbeat 50ms. The ARCHITECTURE is genuine and the parameters are reasonable for a Raft cluster. However, the nodes are FACADES: each runs a shell script that uses netcat to respond "200 OK" on all three ports — no Rust binary is ever invoked (C110). The Dockerfile copies .rlib files (static libraries), which cannot be directly executed as binaries. Critically, the test-runner container DOES run real `cargo test -p ruvector-raft -p ruvector-cluster -p ruvector-replication` — so the individual crate tests are real, but distributed multi-process deployment has NEVER been validated. The 5-node cluster exists only as infrastructure theater for an observer who doesn't inspect the entrypoint scripts.

**Overall testing gap:** The ruvector ecosystem has strong algorithm-level unit testing (softmax, sampling, KV cache, speculation, reservoir sampling, LR scheduling, etc.) but a complete absence of cross-crate integration testing. The two files labeled "integration" (2,928 LOC combined) test ZERO cross-module boundaries. This is consistent with the broader pattern documented since C24 (transport-absent distributed protocol) — subsystems are designed and tested in isolation, then never wired together.


### 5ag. V3 Intelligence Layer & SONA Dual-System (R140)

**2 files, 1,827 LOC, avg ~65%. Intelligence.ts is a CRITICAL FACADE; sona-optimizer.ts is GENUINE.**

R140 characterizes the V3 CLI intelligence and SONA subsystems, confirming the "genuine Rust algorithms orphaned from TS consumers" pattern: the native HNSW in hnsw_router.rs (90-93% genuine) is completely unreachable from the V3 intelligence layer.

**intelligence.ts (985 LOC, 55-60% — CRITICAL FACADE):** The file header claims O(log n) pattern search via HNSW, but `LocalReasoningBank.findSimilar()` (lines 357-385) performs brute-force linear scan with cosine similarity — O(n), no HNSW index, no @ruvector import, no hnswlib-node. `LocalSonaCoordinator` (lines 150-234) uses SONA branding but implements only a pre-allocated circular buffer for signal recording — no LoRA, no EWC, no gradient updates despite SonaConfig exposing `loraLearningRate` (0.001), `loraRank` (8), `ewcLambda` (0.4) (all stored in `this.config` but never referenced by any computation). `compactPatterns()` (lines 863-928) runs an O(n²) all-pairs cosine similarity with no indexing: maxPatterns=5000 means 12.5M operations. The file has 14+ consumers (headless.ts, benchmarks, CLI commands) but ZERO @ruvector/* imports. `benchmarkAdaptation()` (lines 762-799) runs 10,000 iterations and targets <0.05ms — achievable only because the "adaptation" is a trivial circular buffer write, not real ML. The intelligence layer IS NOT dead code (it runs and is consumed), but its core algorithmic claims are false.

**sona-optimizer.ts (842 LOC, 72-78% — GENUINE agent-routing optimizer):** Despite the "SONA" name, this is an AGENT-ROUTING optimizer (learns which agent type to dispatch), not a vector-search optimizer. It implements a genuine Bayesian confidence update loop: success increments via `CONFIDENCE_INCREMENT*(1-conf)`, failure decrements via `CONFIDENCE_DECREMENT*conf`, with temporal decay `exp(-DECAY_RATE*days)` and pattern pruning by `score=confidence*recency`. This is a real ML feedback loop with JSON-file persistence to `.swarm/sona-patterns.json`. Connected to hooks-tools.ts via lazy import — it IS wired into the production hooks pipeline. Lazy-loads `q-learning-router.js` from `../ruvector/q-learning-router.js` which attempts `@ruvector/core` native import with JS fallback — the Q-learning integration degrades silently to pure-JS in production (H228 equivalent). Zero imports from hnsw-index.ts, sona-tools.ts, or hnsw_router.rs — the "150x-12,500x via HNSW" claim in sona-tools.ts is completely disconnected from this module.

**SONA dual-system architecture:** Two independent SONA subsystems exist in V3 with zero cross-reference:
1. `sona-optimizer.ts` — agent dispatch routing, Bayesian learning, trajectory-based, persists to `.swarm/sona-patterns.json`. GENUINE ML feedback loop.
2. `sona-tools.ts` (MCP tool handlers) — fabricated HNSW speedup via `estimatedBruteForce = searchLatency * 1000` producing always-~1000x "improvement". FAKE.

Both import different dependencies, serve different purposes, and share only the "SONA" brand name. `intelligence.ts` is the third member of this family — LocalSonaCoordinator with circular buffer — equally disconnected from the other two.

**Cross-subsystem orphaning confirmed:** The V3 intelligence layer (intelligence.ts → LocalReasoningBank → brute-force cosine) sits adjacent to but completely disconnected from hnsw_router.rs (90-93% genuine, real HnswIndex from ruvector-core, SONA trajectory recording, complete online learning pipeline). A one-import change could wire intelligence.ts into the genuine HNSW backend, but no such bridge exists.

### 5ah. Rust Compilation Audit (R141)

**115 crates audited via cargo check + cargo test --lib across 4 workspaces. Binary truth signal: 87% check pass, 42 crates with 3,984 passing tests.**

R141 provides ground-truth compilation and test status for the entire ruvector Rust codebase. Prior realness scores were based on code reading; this session adds binary compilation evidence.

**Package-level summary:**

| Status | Count | Key crates | LOC impact |
|--------|-------|-----------|------------|
| PASS check + test | 42 | temporal-tensor (269p), nervous-system (359p), GNN (198p), math (148p) | ~250K LOC confirmed genuine |
| PASS check, no tests | 52 | dag, crv, metrics, replication, raft, many delta-* | ~300K LOC compiles |
| FAIL check (CRITICAL) | 2 | ruvllm (120K LOC), sona (10K LOC) | 130K LOC broken |
| FAIL check (HIGH) | 5 | ruvector-cli, ruvllm-cli, ruQu root, + 2 others | ~35K LOC broken entry points |
| CFAIL (test binary) | 6 | prime-radiant (52K), mincut (42K), ruvector-graph (17K), delta-index, + 2 | ~115K LOC untestable |

**CRITICAL compilation failures:**
- **ruvllm (120,345 LOC)**: The largest crate in the repo cannot compile. All downstream crates depending on ruvllm (ruvllm-cli, ruvllm integration) fail as consequences. 120K LOC of LLM inference — the BitNet backend, Flash Attention, GGUF loading, serving engine — is currently unintegrable into the workspace.
- **sona (10,582 LOC)**: The Rust SONA crate (distinct from npm sona-napi) fails check. Consistent with C59 (dual-instance state divergence) and C60 (synchronous background loop).

**Strong genuine signal in core subsystems:**
- ruvector-gnn: 198 tests passing, 8,083 LOC
- ruvector-temporal-tensor: 269 tests passing, 11,446 LOC
- ruvector-nervous-system: 359 tests passing, 14,708 LOC (MOST tests in repo)
- ruvector-math: 148 tests passing, 13,166 LOC
- ruvector-sparse-inference: 88 tests passing, 8,248 LOC
- ruvector-dag: 77 tests, 8,188 LOC

**CFAIL pattern (pass check, fail test binary):** prime-radiant (52,466 LOC), ruvector-mincut (42,157 LOC), and ruvector-graph (16,840 LOC) compile but cannot produce runnable test binaries. This means: the algorithmic bugs documented for these crates (C36-C49 for prime-radiant, C52-C74 for mincut, C5 for graph) have ZERO test coverage and cannot be caught by running `cargo test`. The test binary compilation failures likely stem from feature flag conflicts, missing test dependencies, or proc-macro issues — distinct from the logic bugs in source.

**ruQu anomaly:** The ruQu workspace root (14,251 LOC, C14) fails check, but individual sub-crates pass AND have tests: ruqu-core (602 tests), ruqu-algorithms (23 tests), ruqu-exotic (57 tests). This is a workspace dependency aggregation issue, not a per-crate compilation failure.

**Compilation audit vs. realness scores:** The audit confirms the asymmetry between algorithmic quality (what code reading reveals) and integration quality (what compiling reveals). A crate can be 90%+ genuine algorithmically but fail to compile due to workspace dependency issues. Conversely, crates that compile successfully may still contain logical bugs (C68-C74). Both signals are necessary and neither is sufficient alone.
