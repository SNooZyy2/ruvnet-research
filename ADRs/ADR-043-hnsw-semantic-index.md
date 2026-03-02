# ADR-043: HNSW Semantic Index via @ruvector/core

**Status**: PROPOSED
**Date**: 2026-02-27
**Author**: Research team
**Depends on**: ADR-038 (Research Database System), ADR-042 (Exclusion & Smart Priority)

---

## Context

The research database (SQLite, 15,273 files, 11,753 findings, 1,924 dependencies across 128 sessions) has no semantic search capability. Current limitations:

1. **Finding search is keyword-only** — `LIKE '%keyword%'` misses semantic equivalents (e.g., searching "hash-based embeddings" won't find "placeholder vector generation" even though they describe the same pattern)
2. **Near-duplicate findings are invisible** — across 11,753 findings from 128 sessions, many describe the same underlying issue in different wording. No deduplication mechanism exists
3. **Priority ranking is structural, not semantic** — `smart_priority_gaps` (ADR-042) uses directory co-location and dependency edges. A file about "HNSW indexing" gets no boost from being semantically related to other HNSW files already read at DEEP depth
4. **Cross-domain pattern discovery is manual** — finding that `ruvector-core`, `agentdb`, and `sona` all share the same "hash fallback" antipattern required human reading across all three domains

### What we have

All dependencies are already installed — no new packages required:

| Package | Version | Role |
|---------|---------|------|
| `@ruvector/core` | 0.1.30 | HNSW vector index (Rust/NAPI-RS binding) with automatic disk persistence |
| `@xenova/transformers` | 2.17.2 | Local embedding model (`all-MiniLM-L6-v2`, 384-dim) |
| `better-sqlite3` | latest | Existing SQLite driver for research.db |

The `all-MiniLM-L6-v2` model is already cached locally (~23 MB). Embeddings run entirely on CPU with no API calls and zero cost.

---

## Decision

### D1: Architecture — SQLite Metadata + VectorDb Persistent Index

```
┌────────────────────────┐      ┌──────────────────────────────┐
│   research.db (SQLite) │      │  @ruvector/core VectorDb     │
│                        │      │  ruvector.db (auto-persisted) │
│  findings (source text)│      │                              │
│  embedding_registry    │─────▶│  384-dim HNSW index          │
│   (metadata + backup)  │      │  cosine metric               │
│  semantic_clusters     │◀────│  sub-ms search               │
│                        │      │  survives process restarts   │
└────────────────────────┘      └──────────────────────────────┘
       ▲                                    ▲
       │ source of truth                    │ primary search index
       │ for metadata + text                │ auto-persisted to CWD
       │                                    │
       └──── backup vectors (BLOBs) ────────┘
             for disaster recovery
```

**Key discovery**: `@ruvector/core` VectorDb **automatically persists to `ruvector.db` in the current working directory**. Vectors inserted in one Node.js process survive and are available in subsequent processes. This was verified empirically — the ADR originally assumed in-memory-only behavior, but testing revealed full cross-process persistence.

**Principle**: SQLite is the source of truth for finding text, metadata, and session provenance. `embedding_registry` stores raw embedding BLOBs as a backup/recovery mechanism. VectorDb's `ruvector.db` is the primary search index — scripts read from it directly without rebuilding. If `ruvector.db` is deleted, it can be reconstructed from `embedding_registry`. If `embedding_registry` is also lost, embeddings can be regenerated from finding text.

**Recovery hierarchy**:
1. Normal operation: search VectorDb directly (sub-ms)
2. `ruvector.db` corrupted/deleted: rebuild from `embedding_registry` BLOBs (~2 sec for 12K vectors)
3. Both lost: re-embed from finding text via `embed-findings.js` (~4 min)

### D2: Embedding Strategy — all-MiniLM-L6-v2 (384-dim)

| Property | Value |
|----------|-------|
| Model | `Xenova/all-MiniLM-L6-v2` |
| Dimensions | 384 (hardcoded in @ruvector/core v0.1.30 — see Gotchas) |
| Max tokens | 256 |
| Output | Float32Array (1536 bytes/vector) |
| Speed | ~50 embeddings/sec on CPU |
| Cost | $0 (local inference) |
| Quality | Good for short text similarity; sufficient for finding descriptions (avg ~30 words) |

**Embedding pipeline** (verified working):

```js
const { pipeline } = require('@xenova/transformers');
const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

async function embed(text) {
  const output = await extractor(text, { pooling: 'mean', normalize: true });
  return new Float32Array(output.data);  // [384]
}
```

### D3: Schema Additions

Three new tables added to `research.db`. All are additive — no ALTER TABLE on existing tables. Also update `db/schema.sql` to keep the schema documentation in sync.

```sql
-- Stores embedding metadata and raw vectors as backup/recovery
-- Primary search uses VectorDb (ruvector.db); this is the fallback
CREATE TABLE IF NOT EXISTS embedding_registry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_table TEXT NOT NULL,       -- 'findings' or 'files'
    source_id INTEGER NOT NULL,       -- FK to findings.id or files.id
    embedding BLOB NOT NULL,          -- Float32Array as raw bytes (1536 bytes for 384-dim)
    model TEXT NOT NULL DEFAULT 'all-MiniLM-L6-v2',
    created_date TEXT NOT NULL,
    UNIQUE(source_table, source_id, model)
);

-- Groups of semantically similar findings (populated by dedup script)
CREATE TABLE IF NOT EXISTS semantic_clusters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    label TEXT,                        -- Human-readable cluster name (optional)
    centroid BLOB,                     -- Average embedding of members
    threshold REAL NOT NULL DEFAULT 0.25,  -- Max cosine distance for membership
    created_date TEXT NOT NULL,
    finding_count INTEGER DEFAULT 0
);

-- Junction table: findings → clusters
CREATE TABLE IF NOT EXISTS semantic_cluster_members (
    cluster_id INTEGER NOT NULL REFERENCES semantic_clusters(id) ON DELETE CASCADE,
    finding_id INTEGER NOT NULL,       -- FK to findings.id
    distance REAL NOT NULL,            -- Cosine distance from centroid
    PRIMARY KEY (cluster_id, finding_id)
);

CREATE INDEX IF NOT EXISTS idx_embedding_source
    ON embedding_registry(source_table, source_id);
CREATE INDEX IF NOT EXISTS idx_cluster_members_finding
    ON semantic_cluster_members(finding_id);
```

### D4: Use Cases

#### Use Case A: Semantic Finding Search

**Script**: `scripts/semantic-search.js`

Search findings by natural language query instead of keyword matching.

```js
// scripts/semantic-search.js
// Usage: node scripts/semantic-search.js "hash-based embedding fallback"

const { pipeline } = require('@xenova/transformers');
const { VectorDb } = require('@ruvector/core');
const Database = require('better-sqlite3');

const DB_PATH = '/home/snoozyy/ruvnet-research/db/research.db';

async function main() {
  const query = process.argv[2];
  if (!query) { console.error('Usage: node semantic-search.js "<query>"'); process.exit(1); }

  const sqlite = new Database(DB_PATH);

  // 1. Open persistent HNSW index (auto-loaded from ruvector.db in CWD)
  const vdb = new VectorDb({ dimensions: 384, metric: 'cosine' });

  // 2. Embed query
  const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
  const output = await extractor(query, { pooling: 'mean', normalize: true });
  const qvec = new Float32Array(output.data);

  // 3. Search persistent index directly (no rebuild needed)
  const results = await vdb.search({ vector: qvec, k: 10 });

  // 4. Hydrate from SQLite — filter out internal HNSW nodes (non-numeric ids)
  const findingStmt = sqlite.prepare(`
    SELECT f.id, f.severity, f.category, f.description,
           fi.relative_path, s.name as session_name
    FROM findings f
    JOIN files fi ON f.file_id = fi.id
    JOIN sessions s ON f.session_id = s.id
    WHERE f.id = ?
  `);

  console.log(`\nTop 10 results for: "${query}"\n`);
  for (const r of results) {
    const findingId = parseInt(r.id);
    if (isNaN(findingId)) continue;  // skip internal HNSW nodes
    const finding = findingStmt.get(findingId);
    if (!finding) continue;
    const dist = r.score.toFixed(4);
    console.log(`[${dist}] ${finding.severity} | ${finding.category}`);
    console.log(`  ${finding.description.slice(0, 120)}`);
    console.log(`  File: ${finding.relative_path} | Session: ${finding.session_name}\n`);
  }

  sqlite.close();
}

main().catch(console.error);
```

#### Use Case B: Near-Duplicate Detection

**Script**: `scripts/dedup-findings.js`

Identify findings that describe the same issue across different sessions or files. Populates `semantic_clusters` and `semantic_cluster_members` tables.

```js
// scripts/dedup-findings.js
// Usage: node scripts/dedup-findings.js [--threshold 0.15] [--severity CRITICAL] [--persist]
//
// Score semantics: cosine distance on [0, 2]
//   0.00 = identical vectors
//   0.10 = very similar (~0.90 cosine similarity)
//   0.15 = similar (~0.85 cosine similarity) — default threshold
//   0.50 = loosely related
//   1.00 = orthogonal (unrelated)
//   2.00 = opposite

const { VectorDb } = require('@ruvector/core');
const Database = require('better-sqlite3');

const DB_PATH = '/home/snoozyy/ruvnet-research/db/research.db';

async function main() {
  const threshold = parseFloat(process.argv.find((_, i, a) => a[i-1] === '--threshold') || '0.15');
  const severity = process.argv.find((_, i, a) => a[i-1] === '--severity');
  const persist = process.argv.includes('--persist');

  const sqlite = new Database(DB_PATH);

  // Load embeddings from SQLite backup (not VectorDb, to control which subset)
  let query = 'SELECT er.source_id, er.embedding FROM embedding_registry er WHERE er.source_table = ?';
  const params = ['findings'];
  if (severity) {
    query = `SELECT er.source_id, er.embedding FROM embedding_registry er
             JOIN findings f ON er.source_id = f.id
             WHERE er.source_table = ? AND f.severity = ?`;
    params.push(severity);
  }
  const rows = sqlite.prepare(query).all(...params);

  // Build a temporary in-process index for controlled dedup
  // (We don't use the persistent VectorDb because it may contain
  //  vectors from other source_tables or stale entries)
  const vdb = new VectorDb({ dimensions: 384, metric: 'cosine' });
  for (const row of rows) {
    const vec = blobToFloat32(row.embedding);
    await vdb.insert({ id: String(row.source_id), vector: vec });
  }

  // For each finding, search for near-duplicates
  const clusters = [];   // Array of Sets
  const assigned = new Set();

  for (const row of rows) {
    if (assigned.has(row.source_id)) continue;
    const vec = blobToFloat32(row.embedding);
    const neighbors = await vdb.search({ vector: vec, k: 20 });

    const cluster = new Set();
    cluster.add(row.source_id);
    for (const n of neighbors) {
      const nid = parseInt(n.id);
      if (isNaN(nid)) continue;  // skip internal HNSW nodes
      if (n.score <= threshold && nid !== row.source_id) {
        cluster.add(nid);
        assigned.add(nid);
      }
    }
    assigned.add(row.source_id);
    if (cluster.size > 1) clusters.push(cluster);
  }

  // Report
  const findingStmt = sqlite.prepare(
    'SELECT f.id, f.severity, f.description, fi.relative_path FROM findings f JOIN files fi ON f.file_id = fi.id WHERE f.id = ?'
  );

  console.log(`Found ${clusters.length} duplicate clusters (threshold: ${threshold}, cosine similarity >= ${(1 - threshold).toFixed(2)}):\n`);
  for (const [i, cluster] of clusters.entries()) {
    console.log(`--- Cluster ${i + 1} (${cluster.size} findings) ---`);
    for (const fid of cluster) {
      const f = findingStmt.get(fid);
      if (f) console.log(`  [${f.id}] ${f.severity} | ${f.description.slice(0, 100)} | ${f.relative_path}`);
    }
    console.log();
  }

  // Persist clusters to DB if --persist flag
  if (persist && clusters.length > 0) {
    const today = new Date().toISOString().slice(0, 10);

    // Clear old clusters
    sqlite.exec('DELETE FROM semantic_cluster_members');
    sqlite.exec('DELETE FROM semantic_clusters');

    const insertCluster = sqlite.prepare(
      'INSERT INTO semantic_clusters (label, threshold, created_date, finding_count) VALUES (?, ?, ?, ?)'
    );
    const insertMember = sqlite.prepare(
      'INSERT INTO semantic_cluster_members (cluster_id, finding_id, distance) VALUES (?, ?, ?)'
    );

    const persistAll = sqlite.transaction(() => {
      for (const [i, cluster] of clusters.entries()) {
        const label = `cluster-${i + 1}`;
        const result = insertCluster.run(label, threshold, today, cluster.size);
        const clusterId = result.lastInsertRowid;
        for (const fid of cluster) {
          // Use 0 as placeholder distance (exact per-member distance would
          // require re-searching, which is expensive for a persistence step)
          insertMember.run(clusterId, fid, 0);
        }
      }
    });
    persistAll();
    console.log(`Persisted ${clusters.length} clusters to semantic_clusters/semantic_cluster_members`);
  }

  sqlite.close();
}

// Safe Buffer → Float32Array conversion
// better-sqlite3 returns Buffers that MAY share an underlying ArrayBuffer pool.
// Slicing ensures we get only the bytes for this BLOB.
function blobToFloat32(buf) {
  return new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
}

main().catch(console.error);
```

#### Use Case C: Semantic Priority Reranking

**Script**: `scripts/semantic-priority.js`

Rerank `smart_priority_gaps` results by semantic similarity to findings from already-read DEEP files. For each candidate file, finds the most semantically similar DEEP-file findings, surfacing files whose research themes overlap with productive past reads.

```js
// scripts/semantic-priority.js
// Usage: node scripts/semantic-priority.js [--limit 20]
//
// Unlike the search script, this embeds finding descriptions (not file paths)
// because MiniLM-L6 was trained on English sentences, not filesystem paths.
// For each candidate file, we use its sibling findings (same directory/package)
// to generate a semantic fingerprint.

const { pipeline } = require('@xenova/transformers');
const { VectorDb } = require('@ruvector/core');
const Database = require('better-sqlite3');

const DB_PATH = '/home/snoozyy/ruvnet-research/db/research.db';

async function main() {
  const limit = parseInt(process.argv.find((_, i, a) => a[i-1] === '--limit') || '20');
  const sqlite = new Database(DB_PATH);
  const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

  // 1. Build index of DEEP file finding embeddings
  const vdb = new VectorDb({ dimensions: 384, metric: 'cosine' });
  const deepFindings = sqlite.prepare(`
    SELECT er.source_id, er.embedding FROM embedding_registry er
    JOIN findings f ON er.source_id = f.id
    JOIN files fi ON f.file_id = fi.id
    WHERE er.source_table = 'findings' AND fi.depth = 'DEEP'
  `).all();

  for (const row of deepFindings) {
    const vec = blobToFloat32(row.embedding);
    await vdb.insert({ id: String(row.source_id), vector: vec });
  }
  console.log(`Index: ${deepFindings.length} DEEP-file finding embeddings\n`);

  // 2. Get priority gaps
  const gaps = sqlite.prepare('SELECT * FROM smart_priority_gaps LIMIT ?').all(limit * 3);

  // 3. For each gap file, find sibling findings from same package/directory
  //    and embed those descriptions as the semantic fingerprint.
  //    If no sibling findings exist, fall back to embedding the file's
  //    package name + directory context as a short descriptive phrase.
  const scored = [];
  for (const gap of gaps) {
    // Find findings from files in the same 2-level directory prefix
    const dirPrefix = gap.relative_path.split('/').slice(0, 3).join('/');
    const siblingFindings = sqlite.prepare(`
      SELECT f.description FROM findings f
      JOIN files fi ON f.file_id = fi.id
      WHERE fi.relative_path LIKE ? AND fi.depth = 'DEEP'
      LIMIT 5
    `).all(dirPrefix + '%');

    let queryText;
    if (siblingFindings.length > 0) {
      // Use sibling finding descriptions — these are proper English sentences
      queryText = siblingFindings.map(f => f.description).join('. ');
    } else {
      // Fallback: construct a descriptive phrase from metadata
      // e.g., "HNSW vector indexing in ruvector core package"
      const parts = gap.relative_path.split('/').filter(p => !p.match(/\.(rs|ts|js)$/));
      queryText = parts.join(' ') + ' ' + (gap.domain || '') + ' ' + (gap.package_name || '');
    }

    // Truncate to ~200 chars to stay within MiniLM-L6 sweet spot
    queryText = queryText.slice(0, 200);

    const output = await extractor(queryText, { pooling: 'mean', normalize: true });
    const qvec = new Float32Array(output.data);
    const results = await vdb.search({ vector: qvec, k: 3 });

    // Filter out internal HNSW nodes and compute average
    const validResults = results.filter(r => !isNaN(parseInt(r.id)));
    const avgScore = validResults.length > 0
      ? validResults.reduce((s, r) => s + r.score, 0) / validResults.length
      : 2.0;  // max cosine distance = unrelated
    scored.push({ ...gap, semantic_distance: avgScore });
  }

  // 4. Sort by combined rank: tier_rank (structural) + semantic_distance
  scored.sort((a, b) => {
    const aCombo = (a.tier_rank || 4) + a.semantic_distance;
    const bCombo = (b.tier_rank || 4) + b.semantic_distance;
    return aCombo - bCombo;
  });

  console.log('Semantically reranked priority queue:\n');
  console.log('Rank | Tier      | Sem.Dist | LOC  | Path');
  console.log('-----|-----------|----------|------|-----');
  for (const [i, s] of scored.slice(0, limit).entries()) {
    const tier = (s.tier || 'UNKNOWN').padEnd(9);
    const dist = s.semantic_distance.toFixed(3).padStart(8);
    const loc = String(s.loc || '?').padStart(4);
    console.log(`${String(i + 1).padStart(4)} | ${tier} | ${dist} | ${loc} | ${s.relative_path}`);
  }

  sqlite.close();
}

// Safe Buffer → Float32Array conversion
function blobToFloat32(buf) {
  return new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
}

main().catch(console.error);
```

### D5: Migration Path — Initial Embedding Population

**Script**: `scripts/embed-findings.js`

Populates both `embedding_registry` (SQLite backup) and VectorDb (persistent HNSW index) for all findings.

```js
// scripts/embed-findings.js
// Usage: node scripts/embed-findings.js [--tx-size 50] [--skip-existing]
//
// --tx-size controls SQLite transaction batch size (default 50).
//   Embeddings are generated one-at-a-time (sequential). The tx-size
//   only groups SQLite INSERTs into transactions for performance.
//   See Open Question #1 about potential batched embedding support.

const { pipeline } = require('@xenova/transformers');
const { VectorDb } = require('@ruvector/core');
const Database = require('better-sqlite3');

const DB_PATH = '/home/snoozyy/ruvnet-research/db/research.db';

async function main() {
  const txSize = parseInt(process.argv.find((_, i, a) => a[i-1] === '--tx-size') || '50');
  const skipExisting = process.argv.includes('--skip-existing');

  const sqlite = new Database(DB_PATH);
  const today = new Date().toISOString().slice(0, 10);

  // Create tables if not exist
  sqlite.exec(`
    CREATE TABLE IF NOT EXISTS embedding_registry (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_table TEXT NOT NULL,
        source_id INTEGER NOT NULL,
        embedding BLOB NOT NULL,
        model TEXT NOT NULL DEFAULT 'all-MiniLM-L6-v2',
        created_date TEXT NOT NULL,
        UNIQUE(source_table, source_id, model)
    );
    CREATE TABLE IF NOT EXISTS semantic_clusters (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        label TEXT,
        centroid BLOB,
        threshold REAL NOT NULL DEFAULT 0.25,
        created_date TEXT NOT NULL,
        finding_count INTEGER DEFAULT 0
    );
    CREATE TABLE IF NOT EXISTS semantic_cluster_members (
        cluster_id INTEGER NOT NULL REFERENCES semantic_clusters(id) ON DELETE CASCADE,
        finding_id INTEGER NOT NULL,
        distance REAL NOT NULL,
        PRIMARY KEY (cluster_id, finding_id)
    );
    CREATE INDEX IF NOT EXISTS idx_embedding_source
        ON embedding_registry(source_table, source_id);
    CREATE INDEX IF NOT EXISTS idx_cluster_members_finding
        ON semantic_cluster_members(finding_id);
  `);

  // Get findings to embed (use f.id, not rowid)
  let findingsQuery = 'SELECT id, description, category, severity FROM findings';
  if (skipExisting) {
    findingsQuery += ` WHERE id NOT IN (
      SELECT source_id FROM embedding_registry WHERE source_table = 'findings'
    )`;
  }
  const findings = sqlite.prepare(findingsQuery).all();
  console.log(`Findings to embed: ${findings.length}`);

  if (findings.length === 0) {
    console.log('Nothing to do.');
    sqlite.close();
    return;
  }

  // Load model
  console.log('Loading embedding model...');
  const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
  console.log('Model loaded.');

  // Open persistent VectorDb (writes to ruvector.db in CWD)
  const vdb = new VectorDb({ dimensions: 384, metric: 'cosine' });

  const insertStmt = sqlite.prepare(`
    INSERT OR REPLACE INTO embedding_registry (source_table, source_id, embedding, model, created_date)
    VALUES ('findings', ?, ?, 'all-MiniLM-L6-v2', ?)
  `);

  const insertMany = sqlite.transaction((batch) => {
    for (const item of batch) {
      // Safe BLOB: use byteOffset/byteLength in case Float32Array is a view
      const buf = Buffer.from(
        item.embedding.buffer,
        item.embedding.byteOffset,
        item.embedding.byteLength
      );
      insertStmt.run(item.id, buf, today);
    }
  });

  let processed = 0;
  for (let i = 0; i < findings.length; i += txSize) {
    const batch = findings.slice(i, i + txSize);
    const texts = batch.map(f => `${f.severity} ${f.category}: ${f.description}`);

    // Embed sequentially (one text at a time)
    const embedded = [];
    for (let j = 0; j < texts.length; j++) {
      const output = await extractor(texts[j], { pooling: 'mean', normalize: true });
      const vec = new Float32Array(output.data);
      embedded.push({ id: batch[j].id, embedding: vec });

      // Insert into persistent VectorDb
      // Use 'f-{id}' prefix to namespace finding vectors
      await vdb.insert({ id: `f-${batch[j].id}`, vector: vec });
    }

    // Batch-insert into SQLite
    insertMany(embedded);
    processed += batch.length;

    if (processed % 500 === 0 || processed === findings.length) {
      console.log(`Progress: ${processed}/${findings.length} (${(processed/findings.length*100).toFixed(1)}%)`);
    }
  }

  console.log(`\nDone. ${processed} embeddings stored.`);
  console.log(`  SQLite (embedding_registry): ~${(processed * 1536 / 1024 / 1024).toFixed(1)} MB`);
  console.log(`  VectorDb (ruvector.db): persistent HNSW index updated`);

  sqlite.close();
}

main().catch(console.error);
```

**Estimated time**: 11,753 findings ÷ ~50/sec = ~4 minutes on CPU.

### D6: What NOT to Use from the ruvector Ecosystem

The ruvector monorepo contains many crates and packages. This ADR deliberately scopes to **only** `@ruvector/core` (the NAPI-RS binding to `hnsw_rs`). Specifically excluded:

| Package | Why excluded |
|---------|-------------|
| `ruvector-graph` | Has Cypher parser and executor (corrected in R38), but not needed for vector search |
| `ruvector-core` (Rust crate) | We use the JS binding, not the Rust crate directly |
| `ruvector-mincut` | Graph partitioning algorithms — not relevant to embedding search |
| `@ruvector/embeddings` | Uses hash-based embeddings (CRITICAL finding from R12). We use `@xenova/transformers` instead |
| `@ruvector/sona` | ML orchestration — overkill for single-model embedding |
| `ruvector-benchmark` | Benchmarking infrastructure — not needed for integration |

**Key research finding**: The ruvector ecosystem has a SYSTEMIC hash-based embedding pattern where multiple crates fall back to deterministic hashing instead of learned embeddings (findings from R12, R22, R117). By using `@xenova/transformers` for real neural embeddings and only `@ruvector/core` for the HNSW index, we get genuine semantic search without inheriting any hash fallback paths.

### D7: Incremental Embedding Integration

New findings created during research sessions must be embedded before the next semantic search. Add a post-session step:

**After `scripts/report.js`** in the Session Protocol (step 5), run:

```bash
node scripts/embed-findings.js --skip-existing
```

This embeds only newly-inserted findings (typically 50–200 per session, ~2–4 seconds). The `--skip-existing` flag queries `embedding_registry` to skip already-embedded findings.

The step should be added to the Session Protocol in CLAUDE.md:

```
### 5. End Session
node /home/snoozyy/ruvnet-research/scripts/report.js
node /home/snoozyy/ruvnet-research/scripts/embed-findings.js --skip-existing
```

---

## Implementation Plan

### Phase 1: Schema + Migration (1 hour)

1. Create the 3 new tables (`embedding_registry`, `semantic_clusters`, `semantic_cluster_members`) in research.db
2. **Update `db/schema.sql`** to include the new tables and indexes (keep schema documentation in sync per ADR-038)
3. Run `scripts/embed-findings.js` to populate both `embedding_registry` and VectorDb (~4 min)
4. Verify: `SELECT COUNT(*) FROM embedding_registry WHERE source_table = 'findings'` should equal findings count
5. Spot-check: run `semantic-search.js` with a known query, manually verify top results make sense

### Phase 2: Semantic Search Script (1 hour)

1. Implement `scripts/semantic-search.js`
2. Test queries:
   - `"hash-based embedding fallback"` — should find findings across ruvector-core, agentdb, sona
   - `"broken Laplacian eigensolver"` — should find the 3 systemic Laplacian bugs (C32, C36/C37, C42)
   - `"WASM compilation"` — should find both genuine (18 confirmed) and facade WASM findings
3. Add `--json` flag for programmatic consumption

### Phase 3: Dedup + Clustering (2 hours)

1. Implement `scripts/dedup-findings.js`
2. Run initial dedup pass at threshold 0.15 (cosine similarity ≥ 0.85), review clusters
3. Calibrate threshold — too low (0.10) = false positives, too high (0.25) = missed dupes
4. Run with `--persist` to populate `semantic_clusters` and `semantic_cluster_members`
5. Generate dedup report for human review
6. Estimated runtime: ~12K findings × HNSW search ≈ 2–5 minutes (HNSW search is sub-ms per query; bottleneck is iteration overhead)

### Phase 4: Priority Reranking (1 hour)

1. Implement `scripts/semantic-priority.js`
2. Compare output against current `smart_priority_gaps` ordering
3. Evaluate whether semantic distance adds signal beyond tier_rank
4. If valuable, create a `semantic_priority_gaps` view combining both signals

### Phase 5: Workflow Integration (30 min)

1. Add `embed-findings.js --skip-existing` to Session Protocol end-of-session step
2. Update CLAUDE.md Session Protocol
3. Update agent templates that insert findings to note the embedding step

**Total estimated effort**: 5.5 hours across 5 phases.

---

## Consequences

### Positive

- **Semantic search unlocked** — natural language queries over 11,753 findings with sub-ms latency
- **Duplicate detection** — surface redundant findings across 128 sessions, reducing noise for synthesis
- **Zero infrastructure cost** — all-local computation, no API keys, no external services
- **Persistent index** — VectorDb auto-persists to `ruvector.db`; no rebuild needed between sessions
- **Disaster-recoverable** — SQLite `embedding_registry` BLOBs can reconstruct the HNSW index if `ruvector.db` is lost; finding text can reconstruct embeddings if both are lost
- **Additive schema** — no changes to existing tables; new tables only. Safe to deploy alongside ongoing research
- **Compatible with existing tools** — `report.js`, research agents, and all existing queries continue to work unchanged

### Negative

- **~17 MB storage overhead** — 11,753 embeddings × 1,536 bytes = ~18 MB in SQLite BLOBs, plus ~1.5 MB in `ruvector.db`. Acceptable for a research DB
- **Dual storage** — embeddings exist in both SQLite and `ruvector.db`. The redundancy is intentional (backup + fast search), but means inserts must write to both
- **Single-model lock-in** — embeddings are model-specific. Switching models requires re-embedding everything (~4 min, so not a real concern)
- **256-token limit** — `all-MiniLM-L6-v2` truncates at 256 tokens. Finding descriptions average ~30 words so this is fine, but file-content embeddings would need chunking
- **CWD-dependent persistence** — `ruvector.db` is written to the current working directory. Scripts must be run from the project root (`/home/snoozyy/ruvnet-research/`) to share the same index

### Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Embedding quality too low for short finding text | Low | MiniLM-L6 is well-suited for sentence similarity; findings are sentence-length |
| Dedup threshold hard to calibrate | Medium | Start conservative (0.15 = similarity ≥ 0.85), tune with human review; expose as CLI arg |
| VectorDb persistence path collision | Medium | Always run scripts from project root; document in CLAUDE.md |
| @ruvector/core API changes in future versions | Low | Pin to 0.1.30; API is stable Rust NAPI binding |
| `ruvector.db` accumulates stale vectors | Low | Use `f-{id}` namespaced keys; add a garbage-collection script if needed |

---

## @ruvector/core API Reference (Verified)

Constructor and all methods verified against `@ruvector/core` v0.1.30:

```js
const { VectorDb } = require('@ruvector/core');

// Constructor — note: field is "dimensions" (plural)
const vdb = new VectorDb({ dimensions: 384, metric: 'cosine' });

// All methods are async (return Promises)

// Insert single vector — id must be string, vector must be Float32Array
await vdb.insert({ id: 'f-42', vector: new Float32Array(384) });

// Insert batch
await vdb.insertBatch([
  { id: 'f-1', vector: new Float32Array(384) },
  { id: 'f-2', vector: new Float32Array(384) },
]);

// Search — returns [{ id: string, score: number, vector: Float32Array }]
// score is cosine distance: 0 = identical, 1 = orthogonal, 2 = opposite
const results = await vdb.search({ vector: queryVec, k: 10 });

// Get single vector by id
const entry = await vdb.get('f-42');  // { id, vector }

// Delete
await vdb.delete('f-42');

// Utilities (NOTE: len() includes internal HNSW sentinel nodes)
const count = await vdb.len();      // NOT user vector count — includes ~5-7 internal nodes
const empty = await vdb.isEmpty();   // unreliable — returns false even for "empty" index
```

**Gotchas discovered during verification**:
- `dimensions` not `dimension` — constructor throws `Missing field 'dimensions'` otherwise
- **`dimensions` is hardcoded to 384 in v0.1.30** — constructor accepts other values without error, but insert throws `Dimension mismatch: expected 384, got N`. This happens to match MiniLM-L6 output, but is not configurable
- `id` must be `String` — throws `StringExpected` if passed a number
- `vector` must be `Float32Array` — plain `Array` throws `Get TypedArray info failed`
- `search` takes `{ vector, k }` object — not positional arguments
- All methods return Promises (async) — must use `await`
- **VectorDb auto-persists to `ruvector.db` in the current working directory** — vectors survive across Node.js processes. This is not documented in the package but confirmed empirically
- **`len()` includes internal HNSW sentinel/navigation nodes** — an empty index reports `len()` ≈ 5–7. Do not use `len()` to count user vectors; use `SELECT COUNT(*) FROM embedding_registry` instead
- **`path` and `name` constructor options are accepted but appear to be silently ignored** — all instances share the same `ruvector.db` in CWD regardless of constructor options. Do not rely on these for isolation
- **Search results may include internal node IDs** — filter results to only numeric IDs matching your naming convention (e.g., `f-{finding_id}`)

### Score Semantics

The `score` field in search results is **cosine distance**, not cosine similarity:

| Score | Meaning | Cosine Similarity |
|-------|---------|-------------------|
| 0.00 | Identical direction | 1.00 |
| 0.05 | Extremely similar | 0.95 |
| 0.15 | Similar (dedup threshold) | 0.85 |
| 0.50 | Loosely related | 0.50 |
| 1.00 | Orthogonal (unrelated) | 0.00 |
| 2.00 | Opposite direction | -1.00 |

Formula: `score = 1 - cosine_similarity(a, b)`. Range: [0, 2].

---

## @xenova/transformers Embedding Reference (Verified)

```js
const { pipeline } = require('@xenova/transformers');

// Load model (cached locally at ~/.cache/huggingface/)
const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

// Embed text — returns Tensor with dims [1, 384]
const output = await extractor('query text', { pooling: 'mean', normalize: true });

// Extract Float32Array — output.data is already Float32Array at byteOffset 0
const vec = new Float32Array(output.data);  // length 384

// Safe Buffer conversion for SQLite storage:
// Always use byteOffset/byteLength in case the Float32Array is a view
const buf = Buffer.from(vec.buffer, vec.byteOffset, vec.byteLength);

// Safe BLOB → Float32Array restoration from better-sqlite3:
// Use .slice() to copy bytes out of the potentially-shared Buffer pool
function blobToFloat32(buf) {
  return new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
}
```

---

## Open Questions

1. **Batch embedding** — `@xenova/transformers` may support batched embedding (`extractor(['text1', 'text2'], ...)`) with output `dims: [N, 384]`. If so, embedding speed could increase 2–5× during initial population. Should be tested before Phase 1 to potentially halve migration time
2. **efSearch calibration** — `@ruvector/core` may expose HNSW build parameters (`efConstruction`, `efSearch`, `M`). Default values are likely fine for ~12K vectors but should be documented if configurable
3. **Threshold tuning for dedup** — cosine distance 0.15 (similarity ≥ 0.85) is a starting point. Need empirical calibration across severity levels: CRITICAL findings tend to be verbose and specific (may need tighter threshold ~0.10), while INFO findings are shorter and more formulaic (may tolerate looser threshold ~0.20)
4. **Upgrade path to API embeddings** — if a future phase needs higher-quality embeddings (e.g., Claude API or OpenAI), the `model` column in `embedding_registry` supports multi-model coexistence. Old embeddings can be retained while new ones are generated
5. **File-content embeddings** — this ADR covers finding-text embeddings only. Embedding actual file contents would require chunking strategy for files >256 tokens and a separate population script
6. **VectorDb persistence isolation** — the `path` constructor option appears silently ignored in v0.1.30. If a future version supports custom paths, we should use a dedicated path like `db/ruvector-findings.db` to avoid collision with other tools that might create their own VectorDb in the same directory
7. **Vector ID namespace collisions** — this ADR uses `f-{finding_id}` as the VectorDb key. If file-content embeddings are added later, they should use a different prefix (e.g., `file-{file_id}`) to coexist in the same index

---

## Alternatives Considered

### A: Use hash-based embeddings from @ruvector/embeddings

Rejected. Our own research (R12, R22, R117) identified hash-based embeddings as a SYSTEMIC antipattern in the ruvector ecosystem. Using them would undermine the semantic quality that makes this feature valuable.

### B: Use an external embedding API (OpenAI, Claude)

Rejected for Phase 1. Adds cost, latency, API key management, and network dependency. `all-MiniLM-L6-v2` is good enough for finding-level similarity. The `model` column in `embedding_registry` enables future migration without schema changes.

### C: Use a standalone vector DB (ChromaDB, Qdrant, Weaviate)

Rejected. Adds infrastructure complexity disproportionate to scale. At 12K vectors, `@ruvector/core` with its automatic persistence is simpler, faster to set up, and sufficient for all three use cases.

### D: Add full-text search (FTS5) instead

Complementary, not alternative. FTS5 handles keyword search well but can't find semantic similarity ("hash-based embeddings" ≈ "placeholder vector generation"). Could add FTS5 as a separate enhancement later.

### E: Use only VectorDb persistence, skip SQLite embedding_registry

Considered but rejected. VectorDb's persistence is an undocumented implementation detail that could change. SQLite `embedding_registry` provides a documented, queryable, standard backup. The ~18 MB overhead is negligible. Additionally, `embedding_registry` enables SQL JOINs for analysis (e.g., "which sessions have the most embeddings?") that VectorDb cannot answer.

### F: Rebuild HNSW from SQLite on every script run (original ADR design)

Rejected after discovering VectorDb's automatic persistence. Rebuilding 12K vectors on every invocation wastes 2–5 seconds and was based on the incorrect assumption that VectorDb was in-memory-only.
