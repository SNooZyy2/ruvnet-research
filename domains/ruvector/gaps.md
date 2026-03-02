## 7. Knowledge Gaps

### Crate-Level Coverage (DEEP / non-excluded files, by DB query)

| Crate | DEEP | Total | LOC | Coverage |
|-------|------|-------|-----|----------|
| ruvllm | 88 | 244 | 174,319 | 36% by count |
| ruvector-mincut-gated-transformer | 42 | 69 | 30,002 | 61% by count |
| prime-radiant | 41 | 151 | 73,613 | 27% by count |
| ruvector-postgres | 29 | 243 | 105,078 | 12% by count |
| sona | 27 | 39 | 13,270 | 69% by count |
| ruvector-attention | 23 | 130 | 32,626 | 18% by count |
| ruQu | 15 | 109 | 66,985 | 14% by count |
| ruvector-core | 15 | 55 | 21,745 | 27% by count |
| ruvector-gnn | 15 | 32 | 9,667 | 47% by count |
| ruvector-graph | 12 | 84 | 27,133 | 14% by count |
| ruvector-hyperbolic-hnsw | 7 | 12 | 4,139 | 58% by count |
| ruvector-nervous-system | 7 | 95 | 34,235 | 7% by count |
| ruvector-temporal-tensor | 6 | 27 | 15,307 | 22% by count |
| rvlite | 5 | 96 | 36,736 | 5% by count |
| cognitum-gate-kernel | 5 | 14 | 8,840 | 36% by count |
| hnsw_rs (patches) | 4 | 25 | 7,616 | 16% by count |
| mcp-gate | 3 | 6 | 1,377 | 50% by count |
| edge-net | 6 | 272 | 143,423 | 2% by count |
| cuda-wasm | 1 | 1 | 528 | 100% by count |

### Largest Remaining Gaps

- **ruvllm** — 156 unread files (~100K+ LOC). Largest gap by absolute count. Backends, models, optimization modules partially covered
- **edge-net** — 266 unread files (143K LOC). Only 6 DEEP reads (p2p, simd, lora, federated, rac, attention_unified). Massive crate at 2% coverage
- **ruvector-postgres** — 214 unread files (105K LOC). Index, graph, operator modules have large untouched sections
- **prime-radiant** — 110 unread files (73K LOC). Substrate + execution + coherence + cohomology modules complete; many peripheral files remain
- **ruvector-attention** — 107 unread files (32K LOC). Hyperbolic + sheaf + training + transport characterized; many mechanism files remain
- **ruQu** — 94 unread files (67K LOC). QEC + coherence gate core characterized; wrapper files untouched
- **ruvector-nervous-system** — 88 unread files (34K LOC). Only 7% coverage despite 7 DEEP files
- **ruvector-graph** — 72 unread files (27K LOC). Distributed module characterized; MVCC, optimizer, hybrid features remain
- **rvlite** — 91 unread files (37K LOC). SPARQL parser + Cypher executor deep-read; 95% untouched
- **npm packages** — 1,148 files in npm/, most unread (2 DEEP only)
- **examples** — 1,407 files, only 2 DEEP. 34+ example projects
- **docs** — 341 files (250K LOC), zero DEEP

### Structural Gaps

- **benchmark validation** — no standard ANN-Benchmark results (SIFT1M, GIST1M, Deep1M)
- **spectral.rs without nalgebra feature** — broken fallback path (C32) untested in all 8 existing tests
- **MinCut transformer.rs** — declared in lib.rs but unread (the main inference entry point)
- **router crate** — 4 crates (router-core, router-cli, router-ffi, router-wasm) completely untouched
- **delta framework** — 5 crates (delta-core, delta-wasm, delta-index, delta-graph, delta-consensus) untouched
- **cluster/replication** — distributed system crates untouched

