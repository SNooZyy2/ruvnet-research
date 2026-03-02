#!/usr/bin/env node
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const today = new Date().toISOString().slice(0, 10);

// 1. Update file metadata
db.prepare('UPDATE files SET depth = ?, lines_read = lines_read + ?, last_read_date = ? WHERE id = ?').run('DEEP', 1928, today, 15488);

// 2. Insert file_read record
db.prepare('INSERT INTO file_reads (file_id, session_id, depth, lines_read, line_ranges, notes) VALUES (?, ?, ?, ?, ?, ?)').run(
  15488, 136, 'DEEP', 1928, '1-1928',
  'FULL read of compiled JS (TS source unavailable). Memory bootstrap: sql.js WASM SQLite, HNSW via @ruvector/core VectorDb, embeddings via @xenova/transformers or hash fallback. NO EmbeddingService. NO detectBackends. Directly answers R20 root cause.'
);

// 3. Insert findings
const insertFinding = db.prepare('INSERT INTO findings (file_id, session_id, line_start, line_end, severity, category, description, followed_up) VALUES (?, ?, ?, ?, ?, ?, ?, ?)');

// C49: EmbeddingService NEVER called — R20 ROOT CAUSE CONFIRMED
insertFinding.run(15488, 136, 1145, 1325, 'CRITICAL', 'ARCHITECTURE',
  `C49: EmbeddingService (from @claude-flow/embeddings) is NEVER imported or used in memory-initializer. Embedding generation uses its own 3-tier fallback: (1) @xenova/transformers pipeline for ONNX MiniLM-L6-v2 384-dim, (2) agentic-flow.embeddings 768-dim, (3) hash-based fallback 128-dim. This is a COMPLETELY INDEPENDENT embedding pipeline from EmbeddingService/onnx-embedder.ts. The R20 root cause (EmbeddingService never initialized in claude-flow bridge) is confirmed but goes DEEPER: memory-initializer does not even KNOW about EmbeddingService — it has its own parallel implementation.`, 0);

// C50: HNSW initialization is lazy, never pre-warmed
insertFinding.run(15488, 136, 310, 434, 'CRITICAL', 'ARCHITECTURE',
  `C50: HNSW index via @ruvector/core VectorDb is initialized LAZILY on first search/add call (getHNSWIndex), NOT during initializeMemoryDatabase(). If @ruvector/core import fails (.catch(() => null)), HNSW silently degrades to null and all searchHNSWIndex calls return null, falling through to O(n) brute-force SQLite scan limited to 1000 entries. No error, no warning, no metric — silent degradation.`, 0);

// C51: SQL injection in multiple functions
insertFinding.run(15488, 136, 1620, 1622, 'CRITICAL', 'SECURITY',
  `C51: SQL injection vulnerability in searchEntries, listEntries, getEntry, deleteEntry. Namespace values are interpolated via template literal with only single-quote escaping. String interpolation into SQL bypasses parameterized queries. Example at L1620: namespace inserted directly into SQL WHERE clause. If namespace contains crafted sequences, escaping is bypassed. All 4 CRUD functions affected.`, 0);

// H findings
insertFinding.run(15488, 136, 877, 1029, 'HIGH', 'ARCHITECTURE',
  `initializeMemoryDatabase() creates a sql.js WASM SQLite database with schema, metadata, and vector_indexes config — but does NOT initialize HNSW, does NOT load embedding model, does NOT connect to @ruvector/core. All three are deferred to first use. This means the "init" command gives false confidence: features reported as enabled but nothing is actually loaded.`, 0);

insertFinding.run(15488, 136, 749, 765, 'HIGH', 'BUG',
  `getInitialMetadata() inserts vector_indexes with dimensions=768 (lines 763-764), but loadEmbeddingModel returns 384 dims for the primary ONNX model (MiniLM-L6-v2) and 128 for hash fallback. The 768-dim metadata matches ONLY the agentic-flow fallback that almost never loads. Dimension mismatch between index config and actual embeddings.`, 0);

insertFinding.run(15488, 136, 1110, 1121, 'HIGH', 'ALGORITHM',
  `Temporal decay formula is LINEAR approximation (1.0 - decay_rate * days) instead of the claimed exponential exp(-decay_rate * days). For large time gaps this goes NEGATIVE, making confidence < 0. The WHERE clause (confidence > 0.1) prevents the worst case but any gap > 1/decay_rate days produces negative confidence before the floor catches it. For default decay_rate=0.01, gaps > 100 days produce negative values.`, 0);

insertFinding.run(15488, 136, 317, 323, 'HIGH', 'BUG',
  `HNSW initialization has a busy-wait spin loop: while(hnswInitializing) { await setTimeout(10ms) }. In high-concurrency scenarios this blocks the event loop with repeated 10ms polls. No timeout, no backoff, no maximum wait — could spin indefinitely if initialization hangs.`, 0);

insertFinding.run(15488, 136, 992, 1003, 'HIGH', 'BUG',
  `When sql.js is unavailable, falls back to writing a hand-crafted 4096-byte SQLite header. This creates an invalid SQLite file — it has a page-size header but NO actual schema. Any subsequent read will either fail silently or corrupt. The schema is written as a .sql text file alongside, never actually applied.`, 0);

insertFinding.run(15488, 136, 1158, 1198, 'HIGH', 'ARCHITECTURE',
  `Embedding model loading has 3 independent fallback paths: @xenova/transformers (384d), agentic-flow.embeddings (768d), hash-fallback (128d). Each produces DIFFERENT dimensions. Existing embeddings in the DB may be one dimension and new ones another if the available libraries change. No dimension validation, no migration path — cosine similarity between mismatched dimensions silently truncates via Math.min(a.length, b.length).`, 0);

// MEDIUM findings
insertFinding.run(15488, 136, 14, 303, 'MEDIUM', 'QUALITY',
  `MEMORY_SCHEMA_V3 is a 290-line SQL string constant embedded directly in the JS. 9 tables, 13 indexes. Schema includes SONA trajectory tables and vector_indexes metadata. Well-structured but not versioned — no ALTER TABLE migration path from future versions.`, 0);

insertFinding.run(15488, 136, 438, 450, 'MEDIUM', 'QUALITY',
  `saveHNSWMetadata serializes the entire entries Map to JSON on every insert (addToHNSWIndex). No debouncing. For high-volume inserts this writes the full metadata file on each call. Comment at L465 acknowledges this: "debounced would be better for high-volume".`, 0);

insertFinding.run(15488, 136, 543, 607, 'MEDIUM', 'GENUINE',
  `Int8 symmetric quantization (quantizeInt8/dequantizeInt8) is GENUINE and correctly implemented. Scale = max(|min|,|max|)/127, clamped to [-127,127]. quantizedCosineSim correctly computes similarity directly on Int8 without dequantizing. 4x memory reduction claim is accurate.`, 0);

insertFinding.run(15488, 136, 616, 742, 'MEDIUM', 'GENUINE',
  `Flash attention-style batch operations are GENUINE: batchCosineSim (typed-array optimized), softmaxAttention (numerically stable with max subtraction), topKIndices (min-heap O(n+k log k)), flashAttentionSearch (composed pipeline). Correct implementations, V8-optimized with typed arrays. topK heap is textbook.`, 0);

insertFinding.run(15488, 136, 676, 720, 'MEDIUM', 'ALGORITHM',
  `topKIndices min-heap implementation is correct: O(n + k log k) complexity. Builds heap of size k, replaces min when a larger element found, heapifies down. This is a genuine improvement over Array.sort() for small k vs large n.`, 0);

insertFinding.run(15488, 136, 1310, 1325, 'MEDIUM', 'ALGORITHM',
  `Hash-based embedding fallback: word-by-word, char-by-char positional hashing with sin() modulation. Deterministic but NOT semantic — similar texts do not produce similar vectors. Useful only for exact dedup/testing. The embedding is normalized to unit vector (correct for cosine sim).`, 0);

insertFinding.run(15488, 136, 1500, 1580, 'MEDIUM', 'INTEGRATION',
  `storeEntry() is the main write path: generates embedding (lazy-loads model), writes to SQLite via sql.js, then adds to HNSW index. Read-modify-write pattern: reads entire DB into memory, inserts, exports back to file. NOT concurrent-safe — two simultaneous stores will lose one.`, 0);

insertFinding.run(15488, 136, 1585, 1674, 'MEDIUM', 'INTEGRATION',
  `searchEntries() tries HNSW first (fast path), falls back to brute-force SQLite scan. Brute-force is limited to 1000 entries. Also has keyword matching fallback when embedding similarity is below threshold. Three-tier search: HNSW -> brute-force cosine -> keyword matching.`, 0);

insertFinding.run(15488, 136, 1878, 1885, 'MEDIUM', 'QUALITY',
  `deleteEntry uses soft delete (status=deleted) but uses string interpolation SQL instead of parameterized queries, same SQL injection risk as C51. Also, soft-deleted entries still consume storage and slow queries since no cleanup/vacuum mechanism exists.`, 0);

// INFO findings
insertFinding.run(15488, 136, 8, 9, 'INFO', 'ARCHITECTURE',
  `Only imports: fs and path. Everything else is dynamic import: sql.js, @ruvector/core, @xenova/transformers, agentic-flow. This means the module is maximally lazy-loaded — the import statement alone has zero side effects.`, 0);

insertFinding.run(15488, 136, 1912, 1928, 'INFO', 'ARCHITECTURE',
  `Default export bundles 13 functions + 2 constants. This is the complete public API surface of memory-initializer. Functions: initializeMemoryDatabase, checkMemoryInitialization, checkAndMigrateLegacy, ensureSchemaColumns, applyTemporalDecay, loadEmbeddingModel, generateEmbedding, verifyMemoryInit, storeEntry, searchEntries, listEntries, getEntry, deleteEntry.`, 0);

insertFinding.run(15488, 136, 829, 873, 'INFO', 'QUALITY',
  `checkAndMigrateLegacy scans 4 legacy paths for old databases but only DETECTS migration need — does not actually migrate. Returns {needsMigration: true} for the caller to handle. The actual migration happens in memory-tools.js ensureInitialized() for legacy JSON stores.`, 0);

insertFinding.run(15488, 136, 771, 828, 'INFO', 'QUALITY',
  `ensureSchemaColumns performs ALTER TABLE ADD COLUMN for 12 required columns. Defensive migration for older DBs. Issue #977 referenced: type column was missing. Correctly handles the case where column already exists (catches error and continues).`, 0);

console.log('Findings inserted:', db.prepare('SELECT COUNT(*) as c FROM findings WHERE file_id = 15488 AND session_id = 136').get().c);

// 4. Tag with domains
const domainIds = db.prepare("SELECT id, name FROM domains WHERE name IN ('memory-and-learning', 'agentdb-integration')").all();
for (const d of domainIds) {
  db.prepare('INSERT OR IGNORE INTO file_domains (file_id, domain_id) VALUES (?, ?)').run(15488, d.id);
  console.log('Tagged domain:', d.name, d.id);
}

// 5. Map dependencies
// Find target file IDs for dependencies
const deps = [
  { target_path: 'v3/@claude-flow/cli/dist/src/mcp-tools/memory-tools.js', rel: 'FEEDS', evidence: 'memory-tools.js imports from memory-initializer.js via getMemoryFunctions()' },
  { target_path: 'v3/@claude-flow/cli/dist/src/mcp-tools/neural-tools.js', rel: 'SIBLINGS', evidence: 'neural-tools.js has SEPARATE EmbeddingService from @claude-flow/embeddings — parallel embedding pipeline' },
  { target_path: 'v3/@claude-flow/cli/dist/src/commands/memory.js', rel: 'FEEDS', evidence: 'memory command imports initializeMemoryDatabase, loadEmbeddingModel, verifyMemoryInit' },
  { target_path: 'v3/@claude-flow/cli/dist/src/commands/embeddings.js', rel: 'FEEDS', evidence: 'embeddings command imports generateEmbedding, loadEmbeddingModel' },
];

const findFile = db.prepare('SELECT id FROM files WHERE relative_path LIKE ?');
const insertDep = db.prepare('INSERT OR IGNORE INTO dependencies (source_file_id, target_file_id, relationship, evidence) VALUES (?, ?, ?, ?)');

for (const dep of deps) {
  const target = findFile.get('%' + dep.target_path.split('/').pop());
  if (target) {
    insertDep.run(15488, target.id, dep.rel, dep.evidence);
    console.log('Dep:', dep.target_path.split('/').pop(), '->', target.id);
  } else {
    console.log('Dep target not found:', dep.target_path);
  }
}

// Also add ruvector/core as external dep (by searching for known ruvector files)
const ruvectorFiles = db.prepare("SELECT id, relative_path FROM files WHERE relative_path LIKE '%ruvector%core%' AND relative_path LIKE '%lib.rs%' LIMIT 1").all();
if (ruvectorFiles.length > 0) {
  insertDep.run(15488, ruvectorFiles[0].id, 'USES', 'Dynamic import: @ruvector/core VectorDb for HNSW indexing');
  console.log('Dep: ruvector-core lib.rs ->', ruvectorFiles[0].id);
}

console.log('\nDone. Verifying...');
const file = db.prepare('SELECT depth, lines_read, last_read_date FROM files WHERE id = 15488').get();
console.log('File state:', JSON.stringify(file));
db.close();
