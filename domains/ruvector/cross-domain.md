## 6. Cross-Domain Dependencies

**862 dependency edges** mapped across the ruvector domain (DB query). Key cross-domain connections:

- **memory-and-learning domain**: ReasoningBank implementations (4 distinct), SONA, EWC++, embeddings, attention mechanisms. consolidation.rs (genuine EWC++ in ruvllm reasoning_bank), episodic_memory.rs + semantic_cache.rs (HNSW-backed context memory)
- **agentdb-integration domain**: AgentDB controllers, vector-quantization, LearningSystem, AttentionService
- **agentic-flow domain**: ReasoningBank, EmbeddingService, IntelligenceStore, learning-service
- **claude-flow-cli domain**: LocalReasoningBank (only one that runs), ruvector/ modules, model-router, semantic-router. claude_flow_bridge.rs is a REVERSE dependency — ruvllm calls back to claude-flow CLI as a subprocess for memory storage
- **sublinear-time-solver domain**: sparse.rs (BEST matrix code), consciousness integration
- **prime-radiant cross-domain**: coherence/energy.rs confirmed SIMD-accelerated sheaf Laplacian. coherence/spectral.rs drift detection feeds into ruvector-attention sheaf module (energy_fn linkage — but type-level coupling is absent, H102)
- **ruvector-attention sheaf**: sheaf/attention.rs uses RestrictionMap from sheaf/restriction.rs. early_exit.rs decoupled from sheaf Laplacian by opaque energy_fn. sparse.rs restriction_map.energy() recalculated per-connection (discards stored residuals). Zero integration with prime-radiant coherence module despite shared mathematical foundation

### Findings Distribution by Category (top 12)

| Category | Count | % of Total |
|----------|-------|------------|
| QUALITY | 1,140 | 32.3% |
| ARCHITECTURE | 1,129 | 32.0% |
| ALGORITHM | 326 | 9.2% |
| PERFORMANCE | 203 | 5.8% |
| INTEGRATION | 170 | 4.8% |
| TESTING | 119 | 3.4% |
| BUG | 110 | 3.1% |
| FACADE | 100 | 2.8% |
| DOCUMENTATION | 75 | 2.1% |
| GENUINE | 62 | 1.8% |
| INCOMPLETE | 54 | 1.5% |
| SECURITY | 41 | 1.2% |

### Package-Level Overview

| Package | Files | LOC | DEEP | Findings |
|---------|-------|-----|------|----------|
| ruvector-rust | 5,991 | 2,480,670 | 375 | 3,151 |
| @claude-flow/guidance | 36 | 25,219 | 8 | 16 |
| agentdb | 15 | 12,287 | 14 | 48 |
| sublinear-rust | 16 | 5,490 | 16 | 261 |
| agentic-flow-rust | 3 | 1,268 | 3 | 28 |
| ruv-fann-rust | 3 | 924 | 3 | 20 |
| @ruvector/core | 2 | 115 | 0 | 5 |

