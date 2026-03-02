# R139 Execution Plan: Ground Truth — CI, Tests, Deployment + Missed Files

**Date**: 2026-03-02
**Session ID**: 139
**Focus**: Read CI pipelines, integration tests, deployment configs, and 2 missed files from earlier ML sessions. Answer: "does this actually build, test, and deploy?"
**Strategic value**: CI is the ultimate ground truth. If a package builds in CI, it's real. If it's tested in CI, the integration path works. If it's published, users have it. This is the cheapest high-confidence evidence available. The 2 missed files close small gaps from ML-C/ML-D.

## Rationale

After ML-A through ML-D, we understand the code paths from CLI → MCP → memory → backend. But we don't know: does any of this actually run? CI pipelines answer definitively. R117 (cargo compilation) was our most productive session ever — this extends that approach.

Additionally, 2 files from the original ML plan were missed:
- `src/backends/factory.ts` (agentic-flow package, ID 333) — the non-AgentDB factory. ML-C read the AgentDB one (ID 12809) but not this one.
- `npm/packages/ruvector/src/core/index.ts` (ID 7777) — the barrel export wiring 20+ subsystems. 57 LOC but shows what's exposed.

## Target: 13 files, ~7,632 LOC + 4 CI/Docker files not in DB

---

### Cluster A: Missed Files from ML-C/ML-D (2 files, 292 LOC)

These fell through the cracks. Small but close gaps in the integration picture.

| # | File ID | File | LOC | Why Missed |
|---|---------|------|-----|------------|
| 1 | 333 | `src/backends/factory.ts` | 235 | ML-C read the AgentDB copy (12809) but not the agentic-flow root copy |
| 2 | 7777 | `npm/packages/ruvector/src/core/index.ts` | 57 | Deprioritized due to low LOC — but it's the master barrel export |

**Full paths**:
1. `~/repos/agentic-flow/src/backends/factory.ts`
2. `~/repos/ruvector/npm/packages/ruvector/src/core/index.ts`

**Key questions**:
- `factory.ts` (235 LOC): Is this the same as the AgentDB `packages/agentdb/src/backends/factory.ts` (ID 12809, already DEEP), or a separate factory? Does the agentic-flow root import from AgentDB's factory or maintain its own? If identical, just note it as a re-export; if different, trace its backend selection logic.
- `index.ts` (57 LOC): Which of the 20+ ruvector subsystems does it actually export? Does it re-export GNN, AgentDB, SONA, ONNX, Router, etc.? Or is it a selective subset? This shows what users actually get when they `import from '@ruvector/core'`.

---

### Cluster B: CI Pipelines (6 files, ~2,333 LOC)

What GitHub Actions actually builds and tests. The ruvector repo has 35+ workflows — we read the 4 most important for integration truth, plus 2 that verify native builds.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 3 | NOT_IN_DB | `ruvector/.github/workflows/release.yml` | 621 | What gets built and published to npm/crates.io |
| 4 | NOT_IN_DB | `ruvector/.github/workflows/publish-all.yml` | 552 | Multi-package publish — which packages are publishable |
| 5 | NOT_IN_DB | `claude-flow/.github/workflows/ci.yml` | 228 | What tests run on claude-flow PRs |
| 6 | NOT_IN_DB | `claude-flow/.github/workflows/v3-ci.yml` | 157 | V3-specific CI — does V3 have its own test suite? |
| 7 | NOT_IN_DB | `ruvector/.github/workflows/build-native.yml` | 242 | Native NAPI binary build — proves Rust→Node bridge compiles |
| 8 | NOT_IN_DB | `ruvector/.github/workflows/sona-napi.yml` | 298 | SONA NAPI build — proves SONA Rust→Node bridge compiles |

**Full paths**:
3. `~/repos/ruvector/.github/workflows/release.yml`
4. `~/repos/ruvector/.github/workflows/publish-all.yml`
5. `~/repos/claude-flow/.github/workflows/ci.yml`
6. `~/repos/claude-flow/.github/workflows/v3-ci.yml`
7. `~/repos/ruvector/.github/workflows/build-native.yml`
8. `~/repos/ruvector/.github/workflows/sona-napi.yml`

**Key questions**:
- `release.yml` (621 LOC): Which packages does the release pipeline build? Does it build native NAPI binaries for multiple platforms? Does it run integration tests before publishing? What npm packages does it publish?
- `publish-all.yml` (552 LOC): How many packages get published? Does it include ruvector, ruvllm, rvlite, agentdb? Does it build Rust NAPI bridges?
- `ci.yml` (228 LOC): Does claude-flow CI run memory tests? Does it test MCP tools? Does it require ruvector?
- `v3-ci.yml` (157 LOC): Is V3 tested separately? Does it have different deps?
- `build-native.yml` (242 LOC): What target triples? Does the NAPI binary actually build cross-platform?
- `sona-napi.yml` (298 LOC): Same questions for SONA. This validates whether the SONA Rust crate has a working Node bridge.

**NOTE**: CI files are not in the research DB. After reading, consider inserting them into `files` table under the appropriate package, or just record findings directly. The findings matter more than the DB bookkeeping for YAML files.

---

### Cluster C: Rust Integration Tests (2 files, 2,930 LOC)

End-to-end Rust tests that exercise cross-crate integration. These are the most rigorous integration tests in the codebase.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 9 | 4290 | `crates/ruvllm/tests/e2e_integration_test.rs` | 1,536 | ruvllm end-to-end test — full Rust pipeline |
| 10 | 2598 | `crates/prime-radiant/tests/ruvllm_integration_tests.rs` | 1,394 | prime-radiant ↔ ruvllm cross-crate verification |

**Full paths**:
9. `~/repos/ruvector/crates/ruvllm/tests/e2e_integration_test.rs`
10. `~/repos/ruvector/crates/prime-radiant/tests/ruvllm_integration_tests.rs`

**Key questions**:
- `e2e_integration_test.rs` (1,536 LOC): What does "end to end" cover? Does it test HNSW insert → search → learn? Does it test the `claude_flow` module we read in ML-C? Does it exercise `ruvector_integration.rs`?
- `ruvllm_integration_tests.rs` (1,394 LOC): Does prime-radiant actually call into ruvllm? Does it test `pattern_bridge.rs` (973 LOC, NOT_TOUCHED)? Real integration or mocked units?

---

### Cluster D: Deployment Configs (3 files, ~435 LOC)

Docker-compose files reveal the actual service architecture. How many containers? What network topology?

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 11 | 8306 | `tests/integration/distributed/docker-compose.yml` | 199 | Distributed integration test — multi-node setup |
| 12 | 10148 | `agentic-flow/deployment/docker-compose.agent.yml` | 114 | Agent deployment config |
| 13 | NOT_IN_DB | `v3/@claude-flow/cli/docker/docker-compose.yml` | 117 | V3 claude-flow Docker setup |

**Full paths**:
11. `~/repos/ruvector/tests/integration/distributed/docker-compose.yml`
12. `~/repos/agentic-flow/agentic-flow/deployment/docker-compose.agent.yml`
13. `~/repos/claude-flow/v3/@claude-flow/cli/docker/docker-compose.yml`

**Key questions**:
- `distributed/docker-compose.yml` (199 LOC): What services compose the distributed test? Separate Rust backend from Node frontend? gRPC/HTTP or NAPI?
- `docker-compose.agent.yml` (114 LOC): What does agent deployment look like? Standalone container or sidecar?
- `docker-compose.yml` (117 LOC): Does V3 bundle ruvector or reference it externally? Required vs optional containers?

---

### Optional Cluster E: Live Verification (0 LOC — runtime checks)

If time permits after reading, run verification commands:

```bash
# Check for recent successful CI runs
cd ~/repos/ruvector && gh run list --workflow=release.yml --limit 5 2>/dev/null
cd ~/repos/ruvector && gh run list --workflow=build-native.yml --limit 5 2>/dev/null

# Check published npm packages
npm view ruvector versions --json 2>/dev/null | tail -5
npm view @claude-flow/cli versions --json 2>/dev/null | tail -5

# Run the ruvllm e2e tests locally
cd ~/repos/ruvector && cargo test -p ruvllm --test e2e_integration_test 2>&1 | tail -20
```

---

## Expected Outcomes

1. **Published packages**: Definitive list of what's published to npm/crates.io
2. **CI test coverage**: Which integration paths are verified on every PR
3. **Native build status**: Does the NAPI binary (ruvector-node, sona-napi) actually build cross-platform?
4. **Deployment architecture**: Single-process or multi-service? NAPI or network protocol?
5. **Integration test reality**: What do the Rust e2e tests exercise? Do they pass?
6. **Missed file gaps closed**: agentic-flow factory.ts identity, ruvector barrel export surface

## DB Update Template

```javascript
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const sessionId = 139;
const today = new Date().toISOString().slice(0, 10);

// File IDs for this session:
// 333: src/backends/factory.ts (235 LOC) — agentic-flow, NOT_TOUCHED
// 7777: npm/packages/ruvector/src/core/index.ts (57 LOC) — ruvector-rust, NOT_TOUCHED
// 4290: e2e_integration_test.rs (1,536 LOC) — ruvector-rust, NOT_TOUCHED
// 2598: ruvllm_integration_tests.rs (1,394 LOC) — ruvector-rust, NOT_TOUCHED
// 8306: distributed/docker-compose.yml (199 LOC) — ruvector-rust, NOT_TOUCHED
// 10148: docker-compose.agent.yml (114 LOC) — agentic-flow-rust, NOT_TOUCHED
// CI YAML files NOT in DB — insert findings directly or add to files table first
```

## Domain Tags

- File 333 → `agentdb-integration` + `production-infra`
- File 7777 → `ruvector` + `production-infra`
- Cluster B (CI) → `production-infra`
- Cluster C (tests) → `ruvector` + `production-infra`
- Cluster D (deployment) → `production-infra`

## Isolation Check

All files are in published packages or CI infrastructure. No isolation concerns.

---

## Synthesis Doc Update Protocol (ADR-040)

**MANDATORY**: After all files are read and findings inserted, update relevant `domains/*/analysis.md` following ADR-040.

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
