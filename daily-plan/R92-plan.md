# R92 Execution Plan: Reality-Check Uncovered Areas

**Date**: 2026-02-17
**Session ID**: 92
**Focus**: Deep-read files from README-REALITY-CHECK.md "Uncovered Areas" — AIDefence, Hyperbolic HNSW, CUDA Flash Attention
**Strategic value**: Resolves 3 UNCOVERED verdicts in reality check. AIDefence (HIGH priority) could add genuine security layer to v4. Hyperbolic HNSW (63 files, 0 DEEP) is the largest completely uncharted territory. CUDA flash_attention fills a gap in the Flash Attention arc (R34/R38).

## Rationale

The README-REALITY-CHECK.md identifies 7 uncovered areas with code in the repos but zero DEEP reads. Three of these have the highest strategic value for v4:

1. **AIDefence** (16 files, 0 DEEP, HIGH priority): The reality check notes it was "excluded from npm publish but code exists." If genuine, this adds a real security layer. AIDefenceGuard.ts (763 LOC) is the main implementation — one read determines the verdict.

2. **Hyperbolic HNSW** (63 files, 0 DEEP, MEDIUM priority): The ruvector-hyperbolic-hnsw crate contains Poincaré ball math, Lorentz model attention, and distributed sharding. Given that ruvector-core advanced features maintain 85-93% quality (R90), these could be equally genuine. The WASM version tests the "genuine vs theatrical WASM" pattern (currently 16 genuine, 13 theatrical).

3. **Flash Attention CUDA** (1 file, 0 DEEP): cuda-wasm/src/runtime/flash_attention.rs extends the R38 finding that CUDA-WASM "REVERSES 'no GPU'" with 4 backends. The main flash_attention.rs (997 LOC) is already DEEP at 92%+ — this CUDA variant could be equally genuine.

All files are in CONNECTED packages (ruvector-rust: 36 cross-deps, ruv-fann-rust: 19 cross-deps). No isolation concerns.

## Target: 9 files, ~4,760 LOC

---

### Cluster A: AIDefence (3 files, ~1,164 LOC)

Reality-check verdict: UNCOVERED. "0 DEEP files. 16 files exist. Excluded from npm publish." This cluster determines whether AIDefence is genuine code or another facade.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 1 | 7647 | `npm/packages/ruvbot/src/security/AIDefenceGuard.ts` | 763 | Main implementation — DOMAIN_ONLY tier, highest-LOC AIDefence file |
| 2 | 7690 | `npm/packages/ruvbot/tests/unit/security/aidefence-guard.test.ts` | 235 | Test suite — reveals what's actually tested vs mocked |
| 3 | 283 | `simulation/scenarios/aidefence-integration.ts` | 166 | AgentDB simulation — shows integration surface |

**Full paths**:
1. `~/repos/ruvector/npm/packages/ruvbot/src/security/AIDefenceGuard.ts`
2. `~/repos/ruvector/npm/packages/ruvbot/tests/unit/security/aidefence-guard.test.ts`
3. `~/node_modules/agentdb/simulation/scenarios/aidefence-integration.ts`

**Key questions**:
- `AIDefenceGuard.ts` (763 LOC): Does it implement real threat detection (pattern matching, anomaly scoring) or is it a config-driven stub? Does it reference any ML models or is it rule-based? Why was it excluded from npm publish — size, security, or incompleteness?
- `aidefence-guard.test.ts` (235 LOC): Do tests exercise real detection logic or just mock everything? What attack patterns are tested?
- `aidefence-integration.ts` (166 LOC): Does this actually wire AIDefence into AgentDB's runtime or is it another simulation-only scenario?

---

### Cluster B: Hyperbolic HNSW Core (3 files, ~1,859 LOC)

The ruvector-hyperbolic-hnsw crate is the largest completely uncharted territory (63 files, 0 DEEP). Given ruvector-core's 92-98% quality and advanced features at 85-93% (R90), these Poincaré/Lorentz implementations could be genuine mathematical code.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 4 | 3275 | `crates/ruvector-hyperbolic-hnsw/src/hnsw.rs` | 651 | Core HNSW in hyperbolic space — adapts Euclidean HNSW to Poincaré ball |
| 5 | 3277 | `crates/ruvector-hyperbolic-hnsw/src/poincare.rs` | 628 | Poincaré ball model — exponential/logarithmic maps, geodesic distance |
| 6 | 2738 | `crates/ruvector-attention/src/hyperbolic/lorentz_cascade.rs` | 580 | Lorentz model attention — alternative hyperbolic geometry for cascaded attention |

**Full paths**:
4. `~/repos/ruvector/crates/ruvector-hyperbolic-hnsw/src/hnsw.rs`
5. `~/repos/ruvector/crates/ruvector-hyperbolic-hnsw/src/poincare.rs`
6. `~/repos/ruvector/crates/ruvector-attention/src/hyperbolic/lorentz_cascade.rs`

**Key questions**:
- `hnsw.rs` (651 LOC): Does it implement genuine hyperbolic distance functions in the HNSW graph traversal? Does it use the Poincaré ball or Lorentz model? Are the layer selection and neighbor pruning adapted for non-Euclidean geometry?
- `poincare.rs` (628 LOC): Are the exponential/logarithmic maps mathematically correct? Does it handle the boundary (|x| → 1) correctly? Is there numerical stability handling?
- `lorentz_cascade.rs` (580 LOC): Is this genuine Lorentz model attention (Minkowski inner product) or renamed Euclidean? Does the "cascade" refer to multi-scale attention?

---

### Cluster C: WASM + CUDA Flash Attention (3 files, ~1,737 LOC)

Extends two running arcs: the genuine-vs-theatrical WASM count (16:13) and the Flash Attention quality assessment (Rust main: DEEP 92%+, JS fallback: DEEP).

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 7 | 3283 | `crates/ruvector-hyperbolic-hnsw-wasm/src/lib.rs` | 633 | WASM wrapper for hyperbolic HNSW — 17th genuine or 14th theatrical? |
| 8 | 3278 | `crates/ruvector-hyperbolic-hnsw/src/shard.rs` | 576 | Distributed sharding — R90 found "transport-absent" pattern in ruvector-graph |
| 9 | 8530 | `cuda-wasm/src/runtime/flash_attention.rs` | 528 | CUDA Flash Attention — extends R38 finding (4 GPU backends) |

**Full paths**:
7. `~/repos/ruvector/crates/ruvector-hyperbolic-hnsw-wasm/src/lib.rs`
8. `~/repos/ruvector/crates/ruvector-hyperbolic-hnsw/src/shard.rs`
9. `~/repos/ruv-FANN/cuda-wasm/src/runtime/flash_attention.rs`

**Key questions**:
- `lib.rs` (633 LOC): Does the WASM wrapper expose real hyperbolic HNSW operations (insert, search, distance) or is it another wasm_bindgen stub with console.log?
- `shard.rs` (576 LOC): Does it follow the R90 "transport-absent" pattern (algorithm correct, no socket I/O)? Does it reuse the EdgeCutMinimizer from R90's shard.rs or have its own partitioning?
- `flash_attention.rs` (528 LOC): Does this implement Flash Attention with CUDA kernels (cuBLAS, tiling) or wrap the Rust version? How does it compare to the main 997-LOC version?

---

## Expected Outcomes

1. AIDefence verdict resolved: GENUINE / PARTIALLY REAL / FABRICATED (updates README-REALITY-CHECK.md)
2. Hyperbolic HNSW quality baseline established — determines if 63-file crate is worth further reading
3. WASM count updated: 17th genuine or 14th theatrical
4. Flash Attention arc extended with CUDA variant quality assessment
5. DEEP count: 1,339 → 1,348 (+9)

## DB Update Template

```javascript
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const sessionId = 92;
const today = new Date().toISOString().slice(0, 10);

// File IDs for this session:
// 7647: AIDefenceGuard.ts (763 LOC) — ruvector-rust, DOMAIN_ONLY
// 7690: aidefence-guard.test.ts (235 LOC) — ruvector-rust, NOT IN QUEUE
// 283: aidefence-integration.ts (166 LOC) — agentdb, NOT IN QUEUE
// 3275: hnsw.rs (651 LOC) — ruvector-rust, DOMAIN_ONLY
// 3277: poincare.rs (628 LOC) — ruvector-rust, DOMAIN_ONLY
// 2738: lorentz_cascade.rs (580 LOC) — ruvector-rust, DOMAIN_ONLY
// 3283: hyperbolic-hnsw-wasm/lib.rs (633 LOC) — ruvector-rust, DOMAIN_ONLY
// 3278: shard.rs (576 LOC) — ruvector-rust, DOMAIN_ONLY
// 8530: cuda-wasm/flash_attention.rs (528 LOC) — ruv-fann-rust, NOT IN QUEUE (newly tagged)
```

## Domain Tags

- Files 7647, 7690 → `v4-priority` (already tagged)
- File 283 → `agentdb-integration` (already tagged), add `v4-priority`
- Files 3275, 3277, 3278, 3283, 2738 → `v4-priority` (already tagged), `ruvector` (already tagged)
- File 8530 → `v4-priority` (newly tagged), needs `ruvector` tag

## Isolation Check

All files are in CONNECTED packages:
- ruvector-rust: 36 total cross-deps (5 outbound, 31 inbound) — CONNECTED
- ruv-fann-rust: 19 total cross-deps (16 outbound, 3 inbound) — CONNECTED
- agentdb: 27 total cross-deps (11 outbound, 16 inbound) — CONNECTED

No isolation concerns. No files in excluded subtrees.

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
