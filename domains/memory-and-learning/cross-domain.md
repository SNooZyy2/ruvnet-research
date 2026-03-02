## 6. Cross-Domain Dependencies

- **ruvector domain**: SONA, ruvector-gnn, nervous-system, cognitum-gate, HNSW patches all live in ruvector repo but have strong memory/learning relevance
- **agentdb-integration domain**: AgentDB core components overlap heavily — vector-quantization, LearningSystem, etc. exist in both domains
- **agentic-flow domain**: ReasoningBank, EmbeddingService, IntelligenceStore are shared
- **claude-flow-cli domain**: LocalReasoningBank (the only one that runs) lives there
- **ruvllm**: reasoning_bank.rs, micro_lora.rs, training pipeline


### Detailed Cross-Domain Architecture

(Updated R73: ReasoningBank outer ring exposes 4-tier layering + 3 independent embedding systems)

**ReasoningBank TS Architecture (R73)**:
- `index.ts` →[hardcodes]→ HybridReasoningBank (BYPASSES backend-selector.ts)
- `index.ts` →[patches]→ window/global.AgentDB (runtime timing risk)
- `index.js` ←[targeted_by]→ backend-selector.ts (NOT HybridBackend)
- `wasm-adapter.ts` →[bridges]→ Rust ReasoningBank workspace (R67: core/storage/learning/mcp)
- `embeddings.js` →[3rd_system]→ NO integration with embedding-service.ts (R51) or embeddings.ts (R72)
- `config.ts` →[orphaned_from]→ queries.ts 7-table schema (no db_path)
- `schema.ts` →[mirrors]→ queries.ts schema exactly

**Algorithm Pipeline (R72+R73)**:
- `post-task.ts` →[imports]→ judge.js, distill.ts, consolidate.ts
- `post-task.ts` →[missing]→ matts.ts (should integrate Test-Time Scaling)
- `post-task.ts` →[registered_in]→ hook-handler.cjs (which has STUB implementation)
- `judge.js` →[bypasses]→ MCP ReasoningBank store (creates parallel data flow)
- `judge.js` →[calls]→ Anthropic API directly (security risk)
- `distill.ts` →[uses]→ embeddings.ts (13th hash fallback)
- `consolidate.ts` →[operates_on]→ patterns/pattern_embeddings tables (5th persistence layer)
- `consolidate.ts` →[uses]→ mmr.ts for diversity selection
- `matts.ts` →[uses]→ ModelRouter (6th routing system)

