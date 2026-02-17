# ADR-039: Rust and Compiled Code Research Methodology

**Status:** Proposed
**Date:** 2026-02-14
**Author:** Claude Flow Research
**Supersedes:** None
**Related:** ADR-038 (Research Ledger Database System)

## Context

The current research methodology (ADR-038) was designed to analyze JavaScript and TypeScript source files across the ruvnet multi-repo universe. It tracks files, classifies read depth, records findings, and maps dependencies — all assuming the code under study is human-readable source.

This assumption is fundamentally broken. The ruvnet ecosystem sits on a substantial Rust substrate:

- **207 published Rust crates** on crates.io under user `ruvnet` (user ID 339999)
- **3 compiled WASM blobs** shipped inside npm packages with no Rust source alongside them
- **1 native Node.js addon** (`@ruvector/attention`) compiled from Rust via napi-rs
- **~12 source repositories** containing the Rust code that produces these artifacts

### The Three Mystery WASM Blobs

Our research database contains 3 WASM binaries at `NOT_TOUCHED` depth with no path to understanding their internals:

| WASM Blob | npm Package | Size (LOC*) | Source Found? |
|-----------|-------------|-------------|---------------|
| `guidance_kernel_bg.wasm` | @claude-flow/guidance | 8,943 | **No** — not on crates.io, source repo unknown |
| `reasoningbank_wasm_bg.wasm` | agentic-flow | 1,137 | **Yes** — `ruvnet/agentic-flow` → `reasoningbank/crates/reasoningbank-wasm/` |
| `agentic_flow_quic_bg.wasm` | agentic-flow | 674 | **Yes** — `ruvnet/agentic-flow` → `crates/agentic-flow-quic/` |

*LOC counts for WASM are meaningless binary line counts, not code lines.

### WASM Glue Interface Catalog

Analysis of the JS bridge files (`*_bg.js`) produced by `wasm-bindgen` reveals the full API surface:

**guidance_kernel** (9 functions): `kernel_init`, `batch_process`, `content_hash`, `detect_destructive`, `hmac_sha256`, `sha256`, `scan_secrets`, `sign_envelope`, `verify_chain`. Purpose: cryptographic operations, security scanning, proof chain validation.

**ReasoningBankWasm** (1 class, 6 methods): `storePattern`, `findSimilar`, `getPattern`, `searchByCategory`, `getStats` + constructor. Purpose: vector-based pattern storage/retrieval via IndexedDB.

**WasmQuicClient** (1 class, 4 methods + 2 functions): `sendMessage`, `recvMessage`, `poolStats`, `close` + `createQuicMessage`, `defaultConfig`. Purpose: agent-to-agent QUIC communication.

**@ruvector/attention** (native addon, 14 classes, 15 functions): `DotProductAttention`, `MultiHeadAttention`, `FlashAttention`, `HyperbolicAttention`, `MoEAttention`, `LinearAttention`, etc. Purpose: high-performance attention mechanisms with SIMD, used by agentdb.

### Source Repository Survey

| Repository | Total Files | .rs Files | Cargo.toml | Key Crates |
|-----------|----------:|----------:|-----------:|------------|
| `ruvnet/ruvector` | 6,394 | 2,543 | 139 | ruvector-core, ruvector-attention, ruvector-gnn, ruvector-graph, ruvector-raft, sona, ruvllm |
| `ruvnet/ruv-FANN` | 2,289 | 514 | 28 | ruv-fann, ruv-swarm-core, cuda-wasm, neuro-divergent, opencv-* |
| `ruvnet/agentic-flow` | 4,344 | 239 | 28 | agentic-flow-quic, reasoningbank-*, agent-booster, agentic-jujutsu |
| `ruvnet/sublinear-time-solver` | 1,190 | 231 | 17 | sublinear, bit-parallel-search, temporal-*, neural-network-impl |
| **Total** | **14,217** | **3,527** | **212** | |

### Current Research Gaps

1. **No language classification** — `.rs` and `.js` files are indistinguishable in the DB
2. **No source-to-artifact mapping** — cannot trace which `.rs` files produce which `.wasm` blob
3. **No knowability tracking** — "haven't read it" conflated with "can't read it"
4. **No crate metadata** — Rust workspace structure, crate names, crates.io links absent
5. **No cross-repo dependency model** — crate A in repo X depends on crate B in repo Y
6. **Zero Rust source files inventoried** — 0 of 3,527 `.rs` files are in the database

## Decision

### 1. Clone and Index Rust Source Repositories

Clone the four primary Rust repos into `/home/snoozyy/repos/` and register them as new packages in the research database:

```bash
mkdir -p /home/snoozyy/repos
cd /home/snoozyy/repos
git clone --depth 1 https://github.com/ruvnet/ruvector.git
git clone --depth 1 https://github.com/ruvnet/ruv-FANN.git
git clone --depth 1 https://github.com/ruvnet/agentic-flow.git
git clone --depth 1 https://github.com/ruvnet/sublinear-time-solver.git
```

Register as packages:
```sql
INSERT INTO packages (name, base_path) VALUES
  ('ruvector-rust', '~/repos/ruvector'),
  ('ruv-fann-rust', '~/repos/ruv-FANN'),
  ('agentic-flow-rust', '~/repos/agentic-flow'),
  ('sublinear-rust', '~/repos/sublinear-time-solver');
```

### 2. Schema Extensions

#### 2a. Add `file_type` column to `files`

```sql
ALTER TABLE files ADD COLUMN file_type TEXT;

UPDATE files SET file_type =
  CASE
    WHEN relative_path LIKE '%.rs' THEN 'rs'
    WHEN relative_path LIKE '%.js' THEN 'js'
    WHEN relative_path LIKE '%.mjs' THEN 'js'
    WHEN relative_path LIKE '%.ts' THEN 'ts'
    WHEN relative_path LIKE '%.json' THEN 'json'
    WHEN relative_path LIKE '%.md' THEN 'md'
    WHEN relative_path LIKE '%.wasm' THEN 'wasm'
    WHEN relative_path LIKE '%.node' THEN 'node'
    WHEN relative_path LIKE '%.toml' THEN 'toml'
    WHEN relative_path LIKE '%.sh' THEN 'sh'
    WHEN relative_path LIKE '%.sql' THEN 'sql'
    WHEN relative_path LIKE '%.yaml' OR relative_path LIKE '%.yml' THEN 'yaml'
    ELSE 'other'
  END;

CREATE INDEX idx_files_file_type ON files(file_type);
```

#### 2b. Add `knowability` column to `files`

```sql
ALTER TABLE files ADD COLUMN knowability TEXT
  CHECK(knowability IN ('SOURCE_AUDITABLE','INTERFACE_ONLY','OPAQUE','UNKNOWN'));

UPDATE files SET knowability =
  CASE
    WHEN file_type IN ('rs','js','ts','sh','sql','json','toml','md','yaml') THEN 'SOURCE_AUDITABLE'
    WHEN file_type IN ('wasm','node') THEN 'UNKNOWN'
    ELSE 'UNKNOWN'
  END;
```

Knowability levels:
- **SOURCE_AUDITABLE** — readable source code, line-by-line analysis possible
- **INTERFACE_ONLY** — compiled artifact whose source exists elsewhere in inventory
- **OPAQUE** — compiled artifact with no known source (security audit gap)
- **UNKNOWN** — not yet classified

#### 2c. Create `crates` table

```sql
CREATE TABLE crates (
  id INTEGER PRIMARY KEY,
  name TEXT UNIQUE NOT NULL,
  version TEXT,
  package_id INTEGER REFERENCES packages(id),
  crate_type TEXT,  -- 'lib', 'cdylib', 'bin', 'proc-macro'
  cargo_toml_file_id INTEGER REFERENCES files(id),
  crates_io_url TEXT,
  source_repo TEXT,  -- GitHub repo URL
  workspace_root_id INTEGER REFERENCES crates(id),
  is_napi_wrapper INTEGER DEFAULT 0,
  notes TEXT
);
CREATE INDEX idx_crates_package ON crates(package_id);
```

#### 2d. Create `artifacts` table

Maps compiled outputs to their source crates:

```sql
CREATE TABLE artifacts (
  id INTEGER PRIMARY KEY,
  artifact_file_id INTEGER NOT NULL REFERENCES files(id),
  source_crate_id INTEGER REFERENCES crates(id),
  target_triple TEXT,  -- 'x86_64-unknown-linux-gnu', 'wasm32-unknown-unknown'
  artifact_type TEXT,  -- 'napi-module', 'wasm-bindgen', 'cdylib'
  size_bytes INTEGER,
  notes TEXT,
  UNIQUE(artifact_file_id)
);
CREATE INDEX idx_artifacts_crate ON artifacts(source_crate_id);
```

#### 2e. Create `crate_dependencies` table

```sql
CREATE TABLE crate_dependencies (
  id INTEGER PRIMARY KEY,
  source_crate_id INTEGER NOT NULL REFERENCES crates(id),
  target_crate_name TEXT NOT NULL,
  target_crate_id INTEGER REFERENCES crates(id),  -- NULL if external
  version_req TEXT,
  path TEXT,  -- local path dependency
  is_dev INTEGER DEFAULT 0,
  UNIQUE(source_crate_id, target_crate_name)
);
CREATE INDEX idx_crate_deps_source ON crate_dependencies(source_crate_id);
```

#### 2f. Create views

```sql
-- What Rust code is unauditable?
CREATE VIEW opaque_inventory AS
SELECT p.name AS package, f.relative_path, f.file_type, f.knowability,
  a.target_triple, a.artifact_type, a.size_bytes
FROM files f
JOIN packages p ON f.package_id = p.id
LEFT JOIN artifacts a ON f.id = a.artifact_file_id
WHERE f.knowability IN ('OPAQUE','UNKNOWN') AND f.file_type IN ('wasm','node')
ORDER BY a.size_bytes DESC NULLS LAST;

-- Crate analysis coverage
CREATE VIEW crate_coverage AS
SELECT c.name AS crate_name, c.version, c.crate_type,
  COUNT(DISTINCT CASE WHEN f.file_type = 'rs' THEN f.id END) AS rs_files,
  SUM(CASE WHEN f.file_type = 'rs' THEN f.loc ELSE 0 END) AS rs_loc,
  COUNT(DISTINCT CASE WHEN f.depth IN ('DEEP','MEDIUM') AND f.file_type = 'rs' THEN f.id END) AS analyzed_rs_files,
  COUNT(DISTINCT a.id) AS compiled_artifacts
FROM crates c
LEFT JOIN files f ON f.package_id = c.package_id
LEFT JOIN artifacts a ON a.source_crate_id = c.id
GROUP BY c.id;
```

### 3. Targeted Analysis Strategy (Not Boil-the-Ocean)

We will NOT attempt to read all 3,527 Rust files. The strategy is **surgical tracing** from known integration points:

#### Phase A: Trace the WASM Blobs (Priority 1)

Start from the JS glue → find the Rust source → DEEP read:

| WASM Blob | Rust Source Location | Target Files | Est. Effort |
|-----------|---------------------|--------------|-------------|
| `reasoningbank_wasm` | `agentic-flow/reasoningbank/crates/reasoningbank-wasm/src/lib.rs` | + `reasoningbank-core/src/*.rs` (4 files), `reasoningbank-storage/src/*.rs` (5 files) | ~15 files |
| `agentic_flow_quic` | `agentic-flow/crates/agentic-flow-quic/src/*.rs` | `lib.rs`, `client.rs`, `server.rs`, `types.rs`, `wasm.rs`, `error.rs` | ~7 files |
| `guidance_kernel` | **UNKNOWN** — not found in any surveyed repo | Investigate `@claude-flow/guidance` npm package provenance | TBD |

#### Phase B: Trace the Native Addon (Priority 2)

The `@ruvector/attention` native addon has 14 Rust-backed classes. Source is at:
- `ruvnet/ruvector/crates/ruvector-attention/src/*.rs`
- `ruvnet/ruvector/crates/ruvector-attention-node/src/*.rs` (napi bindings)

Cross-reference with `agentdb/src/wrappers/attention-native.ts` (already partially analyzed at MEDIUM depth) and `agentdb/src/wrappers/attention-fallbacks.ts` (1,953 lines — the JS fallback implementation).

#### Phase C: Core Crate Deep-Reads (Priority 3)

Read the `lib.rs` + key modules for foundational crates:
- `ruvector-core` — vector database core
- `ruvector-gnn` — graph neural network
- `ruvector-graph` — graph operations
- `sona` — SONA learning optimizer
- `ruv-swarm-core` — swarm coordination in Rust

#### Phase D: Cross-Reference JS Fallbacks vs Rust Implementations (Priority 4)

For each JS fallback in agentdb (e.g., `attention-fallbacks.ts` at 1,953 LOC), determine:
- Does the fallback implement the same algorithm as the Rust code?
- Are there behavioral differences?
- Which path is actually taken at runtime?

### 4. File Inventory Protocol for Rust Repos

When indexing a cloned Rust repo:

1. **Scan all files** — insert into `files` table with `depth = 'NOT_TOUCHED'`, populate `file_type` and `knowability`
2. **Parse all Cargo.toml** — insert into `crates` table with version, type, workspace root
3. **Parse crate dependencies** — insert into `crate_dependencies` from `[dependencies]` sections
4. **Map artifacts** — link `.wasm` and `.node` files in the npm packages to their source crates
5. **Update knowability** — mark WASM/node files as `INTERFACE_ONLY` when source is found, `OPAQUE` when not
6. **Tag domains** — all ruvector crates → domain `ruvector`, reasoningbank crates → `memory-and-learning`, etc.
7. **Skip non-essential files** — exclude `.claude/` config dirs, `node_modules/`, `.git/`, `target/` from inventory

### 5. Guidance Kernel Investigation

The `guidance_kernel` WASM blob remains the biggest black box:
- Not on crates.io
- Not found in any of the 4 surveyed repos
- Published as `@claude-flow/guidance` on npm
- Contains security-critical functions (HMAC, chain verification, secret scanning)

Investigation steps:
1. Check npm package metadata for build scripts or source references
2. Search for `guidance-kernel` or `guidance_kernel` across all ruvnet GitHub repos
3. Check if the WASM was hand-compiled or generated by an unreleased build system
4. If source is truly unavailable, classify as `OPAQUE` and document the security implications

### 6. Depth Classification Amendments for Compiled Code

Add to the existing depth classification system:

**For `.wasm` / `.node` files (compiled artifacts):**
- **DEEP**: Full disassembly analysis (wasm-objdump), all exports mapped, behavior verified against JS fallback
- **MEDIUM**: All exports cataloged from JS glue, source crate identified, purpose documented
- **SURFACE**: File size and type noted, artifact type classified
- **INTERFACE_ONLY**: Source exists elsewhere — redirect analysis to the `.rs` files

**For `.rs` files (Rust source):**
- Same classification as JS/TS files (per ADR-038)
- Additionally: document `unsafe` blocks, FFI boundaries, and `#[wasm_bindgen]` exports

## Consequences

### Positive
- Closes the "what does the Rust code do?" gap for ~80% of compiled artifacts
- Enables security auditing of cryptographic code (guidance_kernel) and networking code (QUIC)
- Cross-referencing JS fallbacks vs Rust implementations reveals behavioral differences
- Schema extensions are purely additive — zero risk to existing data

### Negative
- Adds ~14,200 files to inventory (6x increase), though most will remain at NOT_TOUCHED
- Cloning 4 repos requires ~544 MB disk (within the 55 GB available)
- Cargo.toml parsing requires a populate script (est. 4 hours to build)
- `guidance_kernel` source may be permanently unavailable

### Risks
- Rust repos may diverge from the compiled WASM blobs shipped in npm packages (version mismatch)
- The 139 crates in ruvector alone could create analysis paralysis — must maintain surgical focus
- Some crates may be stubs or examples rather than production code

### Mitigations
- Phase A (WASM blob tracing) is scoped to ~22 Rust files — achievable in 1-2 sessions
- Priority ordering ensures highest-value analysis happens first
- Knowability classification prevents wasting time on files that can't be analyzed
- `--depth 1` clone keeps disk usage minimal

## Open Questions

1. Where is the `guidance_kernel` Rust source? Is it in a private repo?
2. Do the WASM blobs in the npm packages match the HEAD of the source repos, or are they from a specific tag/release?
3. Are there additional Rust repos beyond the 4 surveyed that contribute compiled artifacts?
4. Should we also index `ruvnet/QuDAG`, `ruvnet/daa`, `ruvnet/neural-trader`, `ruvnet/midstream`?

## References

- ADR-038: Research Ledger Database System
- crates.io user profile: https://crates.io/users/ruvnet (207 crates)
- ruv.io catalog repo: https://github.com/ruvnet/ruv.io (110 npm package specs, no source)
- Source repos: `ruvnet/ruvector`, `ruvnet/ruv-FANN`, `ruvnet/agentic-flow`, `ruvnet/sublinear-time-solver`
