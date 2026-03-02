# R138 Execution Plan: Ground Truth — CI, Tests, and Deployment

**Date**: 2026-03-01
**Session ID**: 138
**Focus**: Read the CI pipelines, integration tests, and deployment configs to determine what actually builds, tests, and deploys in production
**Strategic value**: CI pipelines are the ultimate ground truth — they reveal which packages are real enough to build and publish. Integration tests show which cross-crate paths are exercised. Docker configs show what the production deployment actually looks like.

## Rationale

After ML-A through ML-D, we'll understand the code paths from CLI → MCP → memory → backend. But we still won't know: does any of this actually run? CI pipelines answer that question definitively.

R117 (cargo compilation verification) was one of the most productive sessions ever — proving that 8 Rust crates compile and 729 tests pass in minutes rather than sessions. This session extends that approach to the CI/CD and deployment layer.

**Dependency**: Run AFTER ML-A through ML-D. The CI/test files reference code paths that will make much more sense after reading the integration layer.

## Target: 9 files, ~6,895 LOC

---

### Cluster A: CI Pipelines (4 files, ~1,558 LOC)

What GitHub Actions actually builds and tests. These reveal which packages are real enough for CI.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 1 | — | `ruvector/.github/workflows/release.yml` | 621 | What gets built and published to npm/crates.io. |
| 2 | — | `ruvector/.github/workflows/publish-all.yml` | 552 | Multi-package publish — shows which packages are publishable. |
| 3 | — | `claude-flow/.github/workflows/ci.yml` | 228 | What tests run on claude-flow PRs. |
| 4 | — | `claude-flow/.github/workflows/v3-ci.yml` | 157 | V3-specific CI — does V3 have its own test suite? |

**Full paths**:
1. `~/repos/ruvector/.github/workflows/release.yml`
2. `~/repos/ruvector/.github/workflows/publish-all.yml`
3. `~/repos/claude-flow/.github/workflows/ci.yml`
4. `~/repos/claude-flow/.github/workflows/v3-ci.yml`

**Key questions**:
- `release.yml` (621 LOC): Which packages does the release pipeline build? Does it build native NAPI binaries for multiple platforms? Does it run integration tests before publishing? What npm packages does it actually publish?
- `publish-all.yml` (552 LOC): How many packages get published? Does it include ruvector, ruvllm, rvlite, agentdb? Does it build the Rust NAPI bridges?
- `ci.yml` (228 LOC): Does claude-flow CI run memory tests? Does it test MCP tools? Does it require ruvector as a dependency?
- `v3-ci.yml` (157 LOC): Is V3 tested separately? Does it have different requirements?

---

### Cluster B: Rust Integration Tests (2 files, ~2,930 LOC)

End-to-end Rust tests that exercise cross-crate integration. These are the most rigorous integration tests in the codebase.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 5 | 4290 | `crates/ruvllm/tests/e2e_integration_test.rs` | 1,536 | ruvllm end-to-end test — exercises the full Rust pipeline. |
| 6 | 2598 | `crates/prime-radiant/tests/ruvllm_integration_tests.rs` | 1,394 | prime-radiant ↔ ruvllm integration — cross-crate verification. |

**Full paths**:
5. `~/repos/ruvector/crates/ruvllm/tests/e2e_integration_test.rs`
6. `~/repos/ruvector/crates/prime-radiant/tests/ruvllm_integration_tests.rs`

**Key questions**:
- `e2e_integration_test.rs` (1,536 LOC): What does "end to end" cover? Does it test HNSW insert → search → learn? Does it test the claude_flow integration module we're reading in ML-C? Do these tests actually pass? (Can verify with `cargo test` if time permits.)
- `ruvllm_integration_tests.rs` (1,394 LOC): Does prime-radiant actually call into ruvllm, or are these unit tests with mocks? Does it test the pattern_bridge.rs (973 LOC, NOT_TOUCHED)?

---

### Cluster C: Deployment Configs (3 files, ~2,407 LOC estimated)

How the system is actually deployed. Docker-compose files reveal service architecture.

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 7 | — | `ruvector/tests/integration/distributed/docker-compose.yml` | ~200 | Distributed integration test — multi-node setup. |
| 8 | — | `agentic-flow/agentic-flow/deployment/docker-compose.yml` | ~200 | Production deployment config for agentic-flow. |
| 9 | — | `claude-flow/v3/@claude-flow/cli/docker/docker-compose.yml` | ~200 | V3 claude-flow Docker setup. |

**Full paths**:
7. `~/repos/ruvector/tests/integration/distributed/docker-compose.yml`
8. `~/repos/agentic-flow/agentic-flow/deployment/docker-compose.yml`
9. `~/repos/claude-flow/v3/@claude-flow/cli/docker/docker-compose.yml`

**Key questions**:
- `distributed/docker-compose.yml`: What services does the distributed test run? Is there a Rust backend service separate from the Node.js frontend? Do they communicate over gRPC/HTTP rather than NAPI?
- `deployment/docker-compose.yml`: What's the production architecture? How many containers? Which ones are required vs optional?
- `cli/docker/docker-compose.yml`: Does claude-flow V3 deploy differently from V2? Does it bundle ruvector or reference it externally?

---

### Optional Cluster D: Live Verification (0 LOC — runtime tests)

If time permits after reading, run actual verification commands to compare with what the CI configs describe.

```bash
# Check if ruvector CI workflows have recent successful runs
cd ~/repos/ruvector && gh run list --workflow=release.yml --limit 5

# Check if published npm packages exist
npm view ruvector versions --json | tail -5
npm view @claude-flow/cli versions --json | tail -5

# Run the ruvllm e2e tests locally (if deps available)
cd ~/repos/ruvector && cargo test -p ruvllm --test e2e_integration_test 2>&1 | tail -20
```

---

## Expected Outcomes

1. **Published packages**: Definitive list of what's actually published to npm/crates.io
2. **CI test coverage**: What tests run on every PR — which integration paths are verified
3. **Deployment architecture**: Single-process or multi-service? NAPI bridge or network protocol?
4. **Integration test results**: Do the Rust e2e tests pass? What do they exercise?
5. **Combined with ML-A through ML-D**: Complete picture from user command → CLI → MCP → memory → backend → algorithm → tested in CI → deployed in production

## DB Update Template

```javascript
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const sessionId = 138;
const today = new Date().toISOString().slice(0, 10);

// File IDs for this session:
// 4290: e2e_integration_test.rs (1,536 LOC) — ruvector-rust, NOT_TOUCHED
// 2598: ruvllm_integration_tests.rs (1,394 LOC) — ruvector-rust, NOT_TOUCHED
// CI/Docker files likely not in DB — these are infrastructure, not source code
// Consider adding them to files table if they contain significant logic
```

## Domain Tags

- Cluster A (CI) → `production-infra`
- Cluster B (integration tests) → `ruvector` + `production-infra`
- Cluster C (deployment) → `production-infra`

## Isolation Check

CI workflows and deployment configs are by definition connected to the published project. Integration test files are in the ruvector-rust package (CONNECTED). No isolation concerns.

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
