# Systematic Rust Compilation Audit Plan

> **Session**: R140 (or next available)
> **Goal**: Binary truth signal — does the code compile? Do tests pass?
> **Method**: `cargo check -p {crate}` + `cargo test -p {crate} --lib` per crate
> **Scope**: ~140+ Rust crates across 4 workspaces
> **Estimated wall-clock**: ~45-60 min (bottleneck: ruvector workspace)

## Research Question

Our static analysis scores "realness" on a 0-100% scale. Compilation/testing provides a hard binary signal: code either compiles or it doesn't. The interesting findings are **disagreements** between our static scores and compilation reality — code we rated high that fails, or code we rated low that passes clean.

## Recording Model (Option B: Findings-Based)

Each crate gets findings anchored to its **Cargo.toml** file_id in the DB.

### Per-Crate Recording

1. **Ensure Cargo.toml has a `files` entry** — insert if missing. Set `loc` to the crate's total source LOC (sum of `src/**/*.rs` files only — not `tests/`), NOT the Cargo.toml line count itself. Compute programmatically:
   ```bash
   find crate_dir/src -name '*.rs' -exec cat {} + | wc -l
   ```
   Record `tests/` LOC separately in `file_reads.notes` if desired, but do not mix into `files.loc`.
2. **Record a `file_reads` entry** on the Cargo.toml with:
   - `depth`: Do NOT update depth. Compilation-testing is not source-reading. Leave existing depth unchanged (or SURFACE if newly inserted).
   - `lines_read`: 0 (compilation test, not source read)
   - `notes`: `"COMPILATION-TESTED | cargo check: PASS|FAIL | cargo test --lib: X passed, Y failed, Z ignored | warnings: N | crate LOC: M"`
3. **Insert findings** per crate:
   - **CRITICAL** if `cargo check` fails (code does not compile)
   - **HIGH** if `cargo test` has failures (tests compile but fail at runtime)
   - **MEDIUM** if test code fails to compile (distinct from check — means tests were never run)
   - **INFO** for clean pass with test counts
   - Category: `COMPILATION` for check failures, `TESTING` for test failures, `QUALITY` for warning-level issues

### Per-Workspace Recording

After individual crates, record one summary finding per workspace:
- `cargo check --workspace --exclude {blocked}` result
- Total crates checked / total in workspace
- Cross-crate compilation: does the full workspace link?

### DB Queries for Analysis

```sql
-- All compilation results from this session
SELECT f.relative_path, fi.severity, fi.category, fi.description
FROM findings fi
JOIN files f ON fi.file_id = f.id
WHERE fi.session_id = ? AND fi.category IN ('COMPILATION', 'TESTING', 'QUALITY')
ORDER BY fi.severity;

-- Compilation failures in crates we rated highly
-- Requires manual cross-reference via the realness lookup table (see Phase 5)
```

## Limitations & Caveats

Record these at session start:

- **`cargo check` vs `cargo build`**: We use `check` (type-checking only, no codegen). This is faster but misses linker errors. A crate can pass `check` but fail `build` if it has unresolvable link-time dependencies. Note this in session findings.
- **Default features only**: `cargo check` tests only default features. Crates with gated code paths behind `#[cfg(feature = "...")]` have untested branches. We do NOT exhaustively test feature combinations.
- **Rust toolchain version**: Record `rustc --version` at session start. Some crates may require nightly features (`#![feature(...)]`). If a crate fails only because of nightly gating, record as MEDIUM not CRITICAL.
- **`cargo check` passes but `cargo test` fails to compile**: This is a distinct category. Test code has its own `#[cfg(test)]` module that may have errors invisible to `check`. The sublinear-time-solver `{:.2f}` bug is exactly this — it proves the test code was never run. Record as MEDIUM (test code broken) not CRITICAL (lib code broken).
- **`--lib` scope only**: We use `cargo test --lib` which tests unit tests in `src/`. Integration tests in `tests/` directories and doctests are NOT covered. This is intentional — integration tests often need network, Docker, or running services. The agentic-flow-quic SIGABRT is an integration test failure, not a `--lib` failure. Record scope in session caveats.
- **WASM binding layer untested**: `cargo check --target wasm32-unknown-unknown` verifies Rust type-correctness only. `wasm-bindgen` JS interface generation and runtime behavior are untested. Testing the full WASM-to-JS pipeline would require `wasm-pack` + npm toolchain, which is out of scope for this session.

## Exclusion Rules

Exclude upfront and record as `"EXCLUDED: {reason}"` in file_reads.notes:

| Exclusion Reason | Affected Crates | Detection |
|-----------------|-----------------|-----------|
| Needs `pgrx` (PostgreSQL extensions) | ruvector-postgres | `Cargo.toml` deps contain `pgrx` |
| Needs `fontconfig` system lib | ruvector-bench (via plotters) | `Cargo.toml` deps contain `plotters` |
| Needs CUDA toolkit | cuda-wasm (ruv-FANN) | `Cargo.toml` deps contain `cuda`/`cublas` |
| Needs running PostgreSQL | docker-integration tests | Test requires network service |
| Needs clang BPF target | rvf-ebpf | BPF C compilation |

**NAPI `-node` crates**: Do NOT pre-exclude. Try `cargo check` first — NAPI may only be needed for `cargo build` (native addon linking), not `check` (type-checking). If check fails with NAPI error, THEN record as `"EXCLUDED: NAPI build toolchain required"`. If check passes, that's a data point.

**WASM crates**: NOT excluded — they need `--target wasm32-unknown-unknown` flag instead of default target. Test separately in Phase 4.

**`examples/` Cargo.toml**: Excluded. Not core code, often reference missing external crates.

**Vendored copies** (`patches/hnsw_rs`, `scripts/patches/hnsw_rs`): Excluded. Dead duplicates of upstream.

## Priority Tiers

### Tier 1 — Core to Research Question (DO FIRST)

Crates we've deeply analyzed and rated. Compilation validates or contradicts our scores.

| Crate | Workspace | Our Rating | Why Important |
|-------|-----------|------------|---------------|
| `ruvector-core` | ruvector | 88-92% | THE vector engine. HNSW, SIMD, real algorithms |
| `rvf` | ruvector | 88-93% | Binary vector format. R123-R124 deep-read |
| `rvf-crypto` | ruvector | 88-93% | SHAKE-256 witness chain. R123 confirmed genuine |
| `rvf-types` | ruvector | 88-92% | Wire format, QR seed headers |
| `sona` | ruvector | ~75% | SONA engine. Central to "150x" claim investigation |
| `ruvector-attention` | ruvector | ~80% | Flash attention, multi-head, hyperbolic |
| `ruvector-router-core` | ruvector | ~88-92% | Model routing (parallel to TS ADR-008). Key to "parallel subsystem" thesis |
| `agent-booster` | agentic-flow | unrated | Token optimization (Tier 1 routing claim). Core crate only, not -wasm/-native |

### Tier 2 — Validated Subsystems + Unrated Core

Crates we've deep-read OR that are structurally important but unrated. Compilation is either validation (rated) or first signal (unrated).

| Crate | Workspace | Our Rating | Why Important |
|-------|-----------|------------|---------------|
| `prime-radiant` | ruvector | ~85-88% | Governance/storage. R107-R108 deep-read |
| `ruvector-mincut` | ruvector | bimodal 35-95% | Graph algorithms. R112-R113 deep-read |
| `ruvector-gnn` | ruvector | ~85-90% | GNN. R99 deep-read |
| `ruvector-graph` | ruvector | ~80% | Graph layer |
| `ruvector-graph-transformer` | ruvector | ~85-90% | PPR implementation. R132 confirmed |
| `ruvector-hyperbolic-hnsw` | ruvector | ~88% | Novel HNSW variant. R99 crate complete |
| `ruvllm` | ruvector | ~80-85% | LLM integration |
| `ruvector-raft` | ruvector | genuine but no transport | Consensus. R115 confirmed |
| `ruvector-cluster` | ruvector | genuine but no transport | Clustering |
| `ruvector-replication` | ruvector | genuine but no transport | Replication |
| `reasoningbank-core` | agentic-flow | unrated (first signal) | Learning pipeline core. Compilation = first data point |
| `reasoningbank-storage` | agentic-flow | unrated (first signal) | Storage backend |
| `reasoningbank-learning` | agentic-flow | unrated (first signal) | RL algorithms |
| `reasoningbank-network` | agentic-flow | unrated (first signal) | P2P networking |
| `ruvector-cli` | ruvector | unrated | CLI binary |
| `ruvector-filter` | ruvector | unrated | Query filtering |
| `ruvector-collections` | ruvector | unrated | Collection management |
| `ruvector-snapshot` | ruvector | unrated | Snapshot/backup |
| `ruvector-server` | ruvector | unrated | Server binary |
| `ruvector-metrics` | ruvector | unrated | Metrics/monitoring |
| `ruv-fann` (lib) | ruv-FANN | unrated | Neural network library |
| `ruv-swarm` (workspace) | ruv-FANN | unrated | Swarm orchestration |
| `sublinear` (lib) | sublinear | unrated | Core algorithms |
| `agentic-flow-quic` | agentic-flow | unrated | QUIC protocol |

### Tier 3 — Peripheral / Exotic

| Crate | Workspace | Notes |
|-------|-----------|-------|
| `ruvector-solver` | ruvector | Sublinear algorithms |
| `ruvector-coherence` | ruvector | Coherence checking |
| `ruvector-cognitive-container` | ruvector | AGI container spec |
| `ruvector-delta-*` (5 crates) | ruvector | Delta/CRDT system |
| `ruvector-domain-expansion` | ruvector | Domain expansion |
| `ruvector-economy-wasm` | ruvector | Token economics |
| `ruvector-dither` | ruvector | Dithering |
| `ruvector-crv` | ruvector | CRV format |
| `ruvector-profiler` | ruvector | Performance profiling |
| `ruvector-sparse-inference` | ruvector | Sparse inference |
| `ruvector-temporal-tensor` | ruvector | Temporal tensors |
| `ruvector-verified` | ruvector | Verified computation |
| `ruvector-nervous-system` | ruvector | SNN system (R114: structurally real, algorithmically broken) |
| `ruvector-learning-wasm` | ruvector | Learning WASM |
| `ruvector-math` | ruvector | Math utilities |
| `thermorust` | ruvector | Thermodynamic computing |
| `rvlite` | ruvector | Lightweight RVF |
| `cognitum-gate-*` (2 crates) | ruvector | Cognitive gating |
| `mcp-gate` | ruvector | MCP gating (R113: 7th MCP) |
| `ruQu` + `ruqu-*` (4 crates) | ruvector | Quantum error correction |
| `agentic-robotics-*` (6 crates) | ruvector | Robotics |
| neuro-divergent (5 crates) | ruv-FANN | Neuro-divergent ML |
| opencv-rust (4 crates) | ruv-FANN | OpenCV bindings |
| psycho-symbolic-reasoner (5 crates) | sublinear | Symbolic reasoning |
| temporal-* (3 crates) | sublinear | Temporal solvers |

### Tier 4 — WASM Verification (Separate Phase)

All `-wasm` crates compiled with `--target wasm32-unknown-unknown`:

| Crate | Workspace |
|-------|-----------|
| `ruvector-wasm` | ruvector |
| `ruvector-attention-wasm` | ruvector |
| `ruvector-attention-unified-wasm` | ruvector |
| `ruvector-gnn-wasm` | ruvector |
| `ruvector-graph-wasm` | ruvector |
| `ruvector-graph-transformer-wasm` | ruvector |
| `ruvector-mincut-wasm` | ruvector |
| `ruvector-hyperbolic-hnsw-wasm` | ruvector (if exists) |
| `ruvector-dag-wasm` | ruvector |
| `ruvector-domain-expansion-wasm` | ruvector |
| `ruvector-solver-wasm` | ruvector |
| `ruvector-verified-wasm` | ruvector |
| `ruvector-router-wasm` | ruvector |
| `ruvector-tiny-dancer-wasm` | ruvector |
| `micro-hnsw-wasm` | ruvector |
| `rvf-wasm` | ruvector |
| `rvf-solver-wasm` | ruvector |
| `reasoningbank-wasm` | agentic-flow |
| `agent-booster-wasm` | agentic-flow |
| `ruv-swarm-wasm` | ruv-FANN |
| `ruv-swarm-wasm-unified` | ruv-FANN |

### NAPI `-node` Crates (try before excluding)

Try `cargo check` on these. If check passes, record as data. If it fails with NAPI-specific errors, record as EXCLUDED.

| Crate | Workspace |
|-------|-----------|
| `ruvector-node` | ruvector |
| `ruvector-gnn-node` | ruvector |
| `ruvector-attention-node` | ruvector |
| `ruvector-graph-node` | ruvector |
| `ruvector-graph-transformer-node` | ruvector |
| `ruvector-mincut-node` | ruvector |
| `ruvector-solver-node` | ruvector |
| `ruvector-tiny-dancer-node` | ruvector |
| `rvf-node` | ruvector |

### Hard-Excluded (record but do not attempt)

| Crate | Reason |
|-------|--------|
| `ruvector-bench` | Needs `fontconfig` system library (plotters dep) |
| `ruvector-postgres` | Needs `pgrx` + PostgreSQL |
| `ruvector-fpga-transformer` | Likely needs FPGA toolchain |
| `ruvector-fpga-transformer-wasm` | Same |
| `cuda-wasm` (ruv-FANN) | Needs CUDA toolkit |
| `ruvector-attn-mincut` | Likely needs both attention + mincut native deps — try, exclude if fails on system deps |
| docker-integration | Needs Docker + running services |
| `rvf-ebpf` | Needs clang BPF target |
| All `examples/` Cargo.toml | Not core code, often reference missing crates |
| `patches/hnsw_rs` | Vendored upstream copy, dead duplicate |
| `scripts/patches/hnsw_rs` | Same |

## Execution Model

### Parallelization Strategy

Do NOT use per-agent `CARGO_TARGET_DIR` — it duplicates dep compilation (serde, tokio, etc.), wasting disk and time. Instead, exploit the shared `target/` cache: the first `cargo check` per workspace pays the dep-compile cost; subsequent crates are incremental (seconds).

**1 agent per workspace, serialize within, parallelize across:**

| Workspace | Agent | Crate count | Role |
|-----------|-------|-------------|------|
| ruvector | Agent 1 (long pole) | ~100+ | Tiers 1-3 ruvector crates + NAPI + WASM |
| agentic-flow | Agent 2 | ~10 | Tiers 1-2 agentic-flow + WASM |
| ruv-FANN | Agent 3 | ~15 | Tier 2-3 ruv-FANN + WASM |
| sublinear | Agent 4 | ~10 | Tier 2-3 sublinear |

All 4 agents spawn in ONE message with `run_in_background: true`.

### Tier 3 Strategy: Workspace-Wide Check (Free Absorption)

Do NOT run 60+ individual `cargo check -p` commands for Tier 3. Instead, after Tier 1-2 per-crate testing, run one workspace-wide command:

```bash
cargo check --workspace --exclude ${DYNAMIC_EXCLUDES} 2>&1
```

This tests ALL crates (including Tier 3) in one pass. Individual failures show up in the error output. Parse the output to attribute failures to specific crates. Only run individual `cargo test -p {crate} --lib` on Tier 3 crates that:
1. Passed the workspace check, AND
2. Show disagreement with static scores (rated low but compiled clean)

### Dynamic Exclude List

Do NOT hardcode excludes. Build them programmatically per workspace:

```bash
EXCLUDES=""
for toml in $(find . -name Cargo.toml -not -path '*/examples/*' -not -path '*/patches/*'); do
  crate=$(grep '^name' "$toml" | head -1 | sed 's/.*"\(.*\)"/\1/')
  if grep -qE 'pgrx|plotters|fontconfig|cuda|cublas|ebpf' "$toml"; then
    EXCLUDES="$EXCLUDES --exclude $crate"
    echo "EXCLUDED: $crate (system dep detected)"
  fi
done
```

Store the final exclude list in `file_reads.notes` for the workspace summary finding.

## Execution Phases

### Phase 0: Environment Record (~2 min)

```bash
rustc --version
cargo --version
rustup target list --installed
rustup target add wasm32-unknown-unknown  # ensure available for Phase 4
# Record in session notes
```

### Phase 1: Inventory + Dynamic Excludes (~10 min)

**4 agents in parallel (one per workspace)**. Each agent:

```bash
# 1. Walk all Cargo.toml files in the workspace
# 2. For each crate:
#    a. Compute src LOC: find crate_dir/src -name '*.rs' -exec cat {} + | wc -l
#    b. Check if Cargo.toml exists in DB (SELECT id FROM files WHERE relative_path LIKE '%{crate}/Cargo.toml')
#    c. Insert if missing (package_id, relative_path, loc = computed src LOC)
#    d. Tag with appropriate domain(s)
# 3. Build dynamic exclude list by grepping Cargo.toml for system deps
# 4. Import pre-existing results from "Already Known" table (if applicable to this workspace)
# 5. Output: crate list + exclude list for Phase 2
```

### Phase 2: Tier 1 + Tier 2 Per-Crate Compilation (~20 min)

**Same 4 agents continue** (or spawn fresh with Phase 1 output). Sequential within workspace:

```bash
# For each Tier 1 crate, then Tier 2 crate in this workspace:
# 1. cargo check -p {crate} 2>&1
#    Record: PASS/FAIL, error message if fail, warning count
# 2. If check passes: cargo test -p {crate} --lib 2>&1
#    Record: tests passed, failed, ignored
#    NOTE: if test CODE fails to compile, that is MEDIUM severity (distinct from check failure)
#    NOTE: capture full output — cargo includes crate name per test binary for attribution
# 3. Insert finding(s) on Cargo.toml file_id
# 4. Insert file_reads entry (depth unchanged, notes = "COMPILATION-TESTED | ...")
```

Skip crates with pre-existing results from "Already Known" table (import those in Phase 1).

### Phase 3: Workspace-Wide Check + Tier 3 Absorption (~15 min)

**Same agents continue**. After per-crate Tier 1-2 testing:

```bash
# Run full workspace check (absorbs all Tier 3 crates for free):
cargo check --workspace ${DYNAMIC_EXCLUDES} 2>&1

# Parse output:
# - Crates that fail → insert CRITICAL findings
# - Crates that pass (not individually tested) → insert INFO finding "WORKSPACE-CHECK-PASS"
# - Record workspace-level summary finding
```

If the full workspace links, that proves cross-crate type compatibility. If it fails, the error identifies which crate boundary is broken — high-value data.

### Phase 4: WASM Verification (~10 min)

```bash
# For each -wasm crate in this workspace:
cargo check -p {crate} --target wasm32-unknown-unknown 2>&1

# NOTE: cargo test is NOT meaningful for wasm32 target (no test runner).
# Only check compilation.
# NOTE: This tests Rust type-correctness only, NOT wasm-bindgen JS interface.
```

### Phase 5: NAPI Probe (ruvector agent only, ~5 min)

```bash
# For each -node crate (all in ruvector workspace):
cargo check -p {crate} 2>&1
# If fails with NAPI-specific error → record as EXCLUDED with error details
# If passes → record as data point (NAPI types resolve even without native build)
```

Folded into the ruvector workspace agent. No separate phase needed for other workspaces.

### Phase 6: Synthesis (~15 min)

1. Query all findings from this session grouped by severity
2. Compute **compilation rate by tier**: `pass / (pass + fail)` excluding EXCLUDED
3. **Build realness lookup table**: For each tested crate, map to our static realness scores from MEMORY.md and session logs. Format:

```
| Crate | Static Score | Check | Test | Agreement? |
```

4. Identify **disagreements** (high-rated but fails, low-rated but passes) — these are the most valuable findings
5. Analyze **error types**: cross-crate interface mismatches vs internal bugs vs missing deps vs test-never-run
6. Compute **compilation by LOC** (not just crate count) — a 50-line crate passing is less meaningful than a 5,000-line crate passing
7. Update `domains/production-infra/analysis.md` with compilation audit section
8. Run `node scripts/report.js`

## Expected Outputs

### Key Metrics
- **Tier 1 compilation rate**: X/8 pass check, Y/8 pass tests
- **Tier 2 compilation rate**: X/24 pass check, Y/24 pass tests
- **Tier 3 compilation rate** (from workspace-wide check): X/N pass
- **WASM compilation rate**: X/21 pass check
- **NAPI probe rate**: X/9 pass check (vs expected 0)
- **Cross-crate linkage**: Does full workspace build? Which crate boundaries fail?
- **Compilation by LOC**: % of total Rust LOC that compiles (weighted, not just crate count)

### Most Valuable Findings (what to look for)
1. **Cross-crate type mismatches** — proof of disconnected development
2. **Test failures in "genuine" code** — runtime bugs our static analysis missed
3. **Clean compilation of code we rated as facade** — maybe we were wrong
4. **Entire subsystems that fail** — validates our "parallel subsystem" thesis
5. **Warning patterns** — dead_code warnings confirm our unused-code findings
6. **check passes but test-compile fails** — proves tests were never run (like sublinear `{:.2f}`)

### Conclusion Framework
After all phases, answer:
- What % of the Rust codebase compiles? (by LOC, by crate count)
- Does compilation rate correlate with our realness scores?
- Which cross-crate boundaries are broken?
- Are the "15+ parallel subsystems" truly independent (no shared compilation)?
- How many test suites have never been run? (check passes, test compile fails)

## Already Known (from accidental partial runs)

These results were obtained before this plan was formalized. Record in Phase 1 as pre-existing data.

| Workspace/Crate | cargo check | cargo test --lib | Notes |
|----------------|-------------|-----------------|-------|
| ruv-FANN (workspace) | PASS | 165 pass, 0 fail (`--lib` only) | Full `cargo test` (with examples) fails: `cuda_wasm_neural_integration` example references missing `cuda_rust_wasm` crate. NOTE: per-crate test breakdown unknown — re-run with full output to attribute 165 tests to individual crates. |
| sublinear-time-solver | PASS (check) | FAIL (7 errors, test code won't compile) | `unknown format trait 'f'` — test code uses C-style `{:.2f}` instead of Rust `{:.2}`. Proves tests were NEVER RUN. |
| agent-booster (workspace check) | PASS | — | Check passes but test fails: agent-booster-wasm test has unwrapped Result (`booster.get_config()` on `Result<T,E>`) |
| agent-booster (workspace test) | — | FAIL (1 compile error) | agent-booster-wasm test code doesn't compile. Core crate untested separately. |
| agentic-flow-quic (check) | PASS | — | Check passes |
| agentic-flow-quic (test) | — | FAIL (SIGABRT) | Integration test panics in destructor cleanup. `test_echo_basic` thread panic → abort. |
| reasoningbank-core | PASS | 9 pass, 0 fail | Clean. Only crate with working tests in reasoningbank workspace. |
| reasoningbank-storage | PASS | 0 tests | Compiles but has no test code |
| reasoningbank-learning | PASS | 0 tests | Compiles but has no test code |
| reasoningbank-network | PASS | 0 tests | Compiles but has no test code |
| reasoningbank-mcp | FAIL (6 errors) | — | Missing `wasm` module in `reasoningbank_storage::adapters`, `StorageConfig` missing Serialize/Deserialize |
| reasoningbank-wasm | FAIL (6 errors) | — | Same missing `wasm` adapters module, type inference failures on storage constructors |
| ruvector (full workspace) | FAIL | — | fontconfig (bench via plotters) + pgrx (postgres) system deps. Excluding those → pgrx still blocks. Need to exclude BOTH bench + postgres. |

## Amendments Log

### v2 (2026-03-02) — Post-Review Amendments

1. **LOC computation scripted**: `find src/ -name '*.rs' | wc -l` baked into Phase 1 agent. `src/` only, `tests/` recorded separately in notes. Eliminates manual counting inconsistency.
2. **Parallelization: 4 workspace agents, not per-crate**: One agent per workspace exploits shared `target/` cache (first check pays dep cost, rest incremental). No `CARGO_TARGET_DIR` tricks — they duplicate deps and waste disk on a resource-limited server.
3. **Tier 3 absorbed into workspace-wide check**: Instead of 60+ individual commands, one `cargo check --workspace --exclude ...` per workspace catches all Tier 3 crates for free. Individual `cargo test` only for interesting Tier 3 disagreements.
4. **Dynamic exclude list**: Built programmatically by grepping Cargo.toml for `pgrx|plotters|cuda|ebpf` instead of hardcoded. Prevents missed excludes that waste entire workspace runs.
5. **`--lib` scope caveat added**: Explicit documentation that `tests/` integration tests and doctests are out of scope. agentic-flow-quic SIGABRT noted as integration test, not `--lib`.
6. **WASM binding layer caveat added**: `cargo check --target wasm32` = Rust type-check only, not `wasm-bindgen` JS interface. Full WASM-to-JS pipeline requires `wasm-pack` + npm, out of scope.
7. **ruv-FANN per-crate attribution gap noted**: 165 tests need re-run with full output to attribute to individual crates. Flagged in Already Known table.
8. **LOC-weighted compilation metric added**: Phase 6 now computes compilation rate by LOC (not just crate count) since a 50-line passing crate is less meaningful than a 5,000-line one.
9. **NAPI probe folded into ruvector agent**: All `-node` crates are in ruvector workspace. No separate agent needed.
10. **Phases 2+3 merged conceptually**: Same agents handle Tier 1 → Tier 2 → workspace-wide sequentially. Avoids agent respawn overhead.
