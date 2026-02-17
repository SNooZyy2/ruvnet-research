# R90 Execution Plan: ruvector Blind Spot — Distributed Graph + Core Advanced Features

**Date**: 2026-02-17
**Session ID**: 90
**Focus**: Deep-read 9 SURFACE-depth files from ruvector's two biggest blind spots: the distributed graph module (5 files, ~2,855 LOC) and core advanced features (4 files, ~2,104 LOC)
**Strategic value**: After R88 cleared the priority queue and R89 closed the research phase, post-mortem analysis identified two significant blind spots — both in the ruvector package which has only 4% DEEP coverage by LOC despite containing production-quality HNSW and SIMD code. These 9 files (all 500-624 LOC) were never queued because they were tagged SURFACE during Phase 0 and lacked cross-deps to DEEP files. Given ruvector-core's 92-98% quality, these could represent either the same production standard or another bimodal split.

## Rationale

The priority queue system (smart_priority_gaps) ranks files by connectivity to existing DEEP files. These 9 files fell through because:
1. ruvector-graph's distributed module has zero recorded dependencies to any DEEP file
2. ruvector-core's advanced features are in subdirectories (`advanced/`, `advanced_features/`) that share no 2-level prefix with existing DEEP reads
3. All 9 files were tagged SURFACE depth during the Phase 0 filesystem scan and never promoted

This is a systematic gap in the priority system: high-quality code in packages with low connectivity gets permanently deprioritized. The manual override for R90 tests whether the blind spot contains genuine implementations or more theatrical stubs.

## Target: 9 files, ~4,959 LOC

---

### Cluster A: ruvector-graph Distributed Module (5 files, ~2,855 LOC)

The distributed layer of ruvector-graph. ruvector-core (HNSW, SIMD) is 92-98% genuine. Does the distributed layer match, or does the bimodal quality pattern (core genuine, integration theatrical) repeat?

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 1 | 3204 | `crates/ruvector-graph/src/distributed/gossip.rs` | 624 | Gossip protocol — SWIM, epidemic, or push-pull? Real failure detector or stub? |
| 2 | 3208 | `crates/ruvector-graph/src/distributed/shard.rs` | 596 | Graph sharding — consistent hashing, range-based, or placeholder? Rebalancing? |
| 3 | 3203 | `crates/ruvector-graph/src/distributed/federation.rs` | 583 | Cross-shard federation — scatter-gather, distributed joins, or empty routing? |
| 4 | 3202 | `crates/ruvector-graph/src/distributed/coordinator.rs` | 536 | Distributed coordinator — leader election, 2PC, or central dispatcher? |
| 5 | 3207 | `crates/ruvector-graph/src/distributed/rpc.rs` | 516 | RPC layer — tonic/gRPC, tarpc, custom protocol, or stubs? |

**Full paths**:
1. `~/repos/ruvector/crates/ruvector-graph/src/distributed/gossip.rs`
2. `~/repos/ruvector/crates/ruvector-graph/src/distributed/shard.rs`
3. `~/repos/ruvector/crates/ruvector-graph/src/distributed/federation.rs`
4. `~/repos/ruvector/crates/ruvector-graph/src/distributed/coordinator.rs`
5. `~/repos/ruvector/crates/ruvector-graph/src/distributed/rpc.rs`

**Key questions**:
- Do these implement real distributed protocols or are they design-docs-as-code?
- Is there actual network transport (sockets, gRPC channels, QUIC) or only in-process DashMap fan-out?
- Does the SWIM gossip protocol have a real failure detector with configurable timeouts?
- Does the RPC layer use tonic, tarpc, or any real RPC framework?

---

### Cluster B: ruvector-core Advanced Features (4 files, ~2,104 LOC)

Advanced mathematical modules in ruvector-core. The core HNSW is 92-98% genuine. Do these advanced features maintain that quality?

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 6 | 2920 | `crates/ruvector-core/src/advanced_features/product_quantization.rs` | 551 | PQ — codebook training (k-means), subspace partitioning, ADC? |
| 7 | 2916 | `crates/ruvector-core/src/advanced_features/conformal_prediction.rs` | 505 | Conformal prediction — nonconformity measures, p-values, confidence sets? |
| 8 | 2911 | `crates/ruvector-core/src/advanced/hypergraph.rs` | 551 | Hypergraph — real bipartite incidence or relabeled regular graph? |
| 9 | 2915 | `crates/ruvector-core/src/advanced/tda.rs` | 497 | TDA — persistent homology, Betti numbers, persistence diagrams? |

**Full paths**:
6. `~/repos/ruvector/crates/ruvector-core/src/advanced_features/product_quantization.rs`
7. `~/repos/ruvector/crates/ruvector-core/src/advanced_features/conformal_prediction.rs`
8. `~/repos/ruvector/crates/ruvector-core/src/advanced/hypergraph.rs`
9. `~/repos/ruvector/crates/ruvector-core/src/advanced/tda.rs`

**Key questions**:
- Does PQ implement real k-means++ initialization and Lloyd's algorithm for codebook training?
- Does conformal prediction use the correct Vovk et al. finite-sample quantile formula?
- Is the hypergraph a real bipartite incidence structure or a regular graph with "hyper" in the name?
- Does TDA implement actual persistent homology (Vietoris-Rips, boundary operators, Betti numbers)?

---

## Expected Outcomes

1. ruvector quality gradient characterized: core (92-98%) vs advanced features vs distributed layer
2. Determine if ruvector-graph distributed is genuine or another "design-doc-as-code" layer
3. Identify if the bimodal quality split (core genuine, integration theatrical) extends to ruvector
4. DEEP count: 1,323 → ~1,332
5. New finding class: "transport-absent distributed protocol" if confirmed

## DB Update Template

```javascript
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const sessionId = 90;
const today = new Date().toISOString().slice(0, 10);

// File IDs for this session:
// 3204: gossip.rs (624 LOC) — ruvector-rust, SURFACE→DEEP
// 3208: shard.rs (596 LOC) — ruvector-rust, SURFACE→DEEP
// 3203: federation.rs (583 LOC) — ruvector-rust, SURFACE→DEEP
// 3202: coordinator.rs (536 LOC) — ruvector-rust, SURFACE→DEEP
// 3207: rpc.rs (516 LOC) — ruvector-rust, SURFACE→DEEP
// 2920: product_quantization.rs (551 LOC) — ruvector-rust, SURFACE→DEEP
// 2916: conformal_prediction.rs (505 LOC) — ruvector-rust, SURFACE→DEEP
// 2911: hypergraph.rs (551 LOC) — ruvector-rust, SURFACE→DEEP
// 2915: tda.rs (497 LOC) — ruvector-rust, SURFACE→DEEP
```

## Domain Tags

- All 9 files → `ruvector` (already tagged)

## Isolation Check

All files are in `ruvector-rust` package which is active with 211 existing DEEP files. No isolation concerns.

---

## Results (retroactive — session executed before plan written)

### Cluster A Verdict: Design-Doc-as-Code (15-80%)

| File | LOC | Realness | Key Finding |
|------|-----|----------|-------------|
| `shard.rs` | 596 | 70-80% | **BEST** — real EdgeCutMinimizer (multilevel Kernighan-Lin), real xxh3/blake3 hashing |
| `gossip.rs` | 624 | 45-55% | Correct SWIM state machine + failure detector, transport = log statements only |
| `federation.rs` | 583 | 40-50% | Real merge/dedup logic, `execute_on_cluster()` returns empty Vec always |
| `coordinator.rs` | 536 | 30-35% | 2PC type system defined, state machine never transitions past Active |
| `rpc.rs` | 516 | 15-20% | All 4 RPC methods are stubs. gRPC feature-gated out of default Cargo build |

**New pattern class**: "transport-absent distributed protocol" — algorithm and state machine are correctly designed, but no socket I/O exists. Every file contains `// In production, send actual network message` comments.

### Cluster B Verdict: Genuine (85-93%)

| File | LOC | Realness | Key Finding |
|------|-----|----------|-------------|
| `conformal_prediction.rs` | 505 | 88-93% | Valid split-conformal, correct Vovk et al. quantile formula |
| `product_quantization.rs` | 551 | 88-92% | Real k-means++, Lloyd's algorithm, asymmetric distance computation with LUT |
| `hypergraph.rs` | 551 | 85-90% | Genuine bipartite incidence, k-hop BFS, cites HyperGraphRAG (NeurIPS 2025) |
| `tda.rs` | 497 | 60-70% | **MISLABELED** — graph metrics real, but no persistent homology. 11th mislabeled file |

### Quality Gradient Confirmed

```
ruvector-core algorithms:     92-98%  (HNSW, SIMD — production-ready)
ruvector-core advanced:       85-93%  (PQ, conformal, hypergraph — production-ready)
ruvector-graph partitioning:  70-80%  (EdgeCutMinimizer — working algorithm)
ruvector-graph protocols:     40-55%  (SWIM, federation — correct design, no transport)
ruvector-graph transport:      0-20%  (RPC, 2PC — stubs)
```

### Session Totals
- 9 files, ~4,959 LOC, 50 findings
- DEEP: 1,323 → 1,332
- Findings: 9,121 → 9,171

---

## Synthesis Doc Update Protocol (ADR-040)

**MANDATORY**: After all files are read and findings inserted into the DB, update the relevant `domains/*/analysis.md` files following the ADR-040 in-place protocol. Reference: `domains/memory-and-learning/analysis.md` for canonical structure.

### Rules for Each Section

| Section | Action | NEVER Do |
|---------|--------|----------|
| **1. Current State Summary** | REWRITE in-place to reflect current state | Append session narrative |
| **2. File Registry** | ADD new rows to existing subsystem tables, UPDATE rows if re-read | Duplicate rows, create per-session file tables |
| **3. Findings Registry** | ADD new findings with next sequential ID (C{max+1}, H{max+1}) to 3a/3b | Create `### RXX Findings` blocks, re-list old findings, restart ID numbering |
| **4. Positives Registry** | ADD new positives with session tag | Re-list existing positives |
| **5. Subsystem Sections** | UPDATE existing sections, CREATE new ones by topic | Create per-session narrative blocks |
| **8. Session Log** | APPEND 2-5 line entry for this session | Put findings here, write full narratives |

### Finding ID Assignment

Before adding findings, check the current max ID in the target domain's analysis.md:
- Section 3a: find last `| C{N} |` row → new CRITICALs start at C{N+1}
- Section 3b: find last `| H{N} |` row → new HIGHs start at H{N+1}

**ID format**: `| {ID} | **{short title}** — {description} | {file(s)} | R{session} | Open |`

### Anti-Patterns (NEVER do these)

- **NEVER** create `### R{N} Findings (Session date)` blocks outside Section 3
- **NEVER** append findings after Section 8
- **NEVER** create `### R{N} Full Session Verdict` blocks
- **NEVER** use finding IDs that collide with existing ones (always check max first)
- **NEVER** re-list findings from previous sessions

### Synthesis Update Checklist

- [ ] Section 1 rewritten with updated state
- [ ] New file rows added to Section 2 (correct subsystem table)
- [ ] New findings added to Section 3a/3b with sequential IDs
- [ ] New positives added to Section 4 (if any)
- [ ] Relevant subsystem sections in Section 5 updated
- [ ] Session log entry appended to Section 8 (2-5 lines max)
- [ ] No per-session finding blocks created anywhere
