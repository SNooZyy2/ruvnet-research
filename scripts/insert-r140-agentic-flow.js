// R140 Compilation Audit: agentic-flow — DB insertion script
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const today = new Date().toISOString().slice(0, 10);
const SESSION_ID = 141;

function ensureFile(relPath, loc) {
  let row = db.prepare('SELECT id FROM files WHERE relative_path = ? AND package_id = 10').get(relPath);
  if (!row) {
    const r = db.prepare("INSERT INTO files (package_id, relative_path, loc, depth) VALUES (10, ?, ?, 'SURFACE')").run(relPath, loc);
    row = { id: r.lastInsertRowid };
  }
  return row.id;
}

const crates = [
  {
    relPath: 'packages/agent-booster/crates/agent-booster/Cargo.toml',
    loc: 2292,
    notes: 'COMPILATION-TESTED | cargo check: PASS | cargo test --lib: 19 passed, 6 failed, 0 ignored | warnings: 1 | crate LOC: 2292 | FAILURES: merge::test_select_strategy, similarity::test_normalized_match, templates::test_detect_try_catch_async, tests::test_empty_file, tests::test_class_method_edit, tests::test_typescript_interface',
    severity: 'HIGH',
    category: 'TESTING',
    description: 'agent-booster: 6/25 lib tests fail. Strategy selection logic wrong (FuzzyReplace vs InsertAfter), similarity normalized-match assertion fails, async try-catch template detection regression, and 3 integration tests fail for empty file / TypeScript interface / class method edits. Core Tier 1 token-optimization logic has broken test paths.',
  },
  {
    relPath: 'packages/agent-booster/crates/agent-booster-native/Cargo.toml',
    loc: 187,
    notes: 'COMPILATION-TESTED | cargo check: PASS | cargo test --lib: 0 passed, 0 failed, 0 ignored | warnings: 0 | crate LOC: 187',
    severity: 'INFO',
    category: 'QUALITY',
    description: 'agent-booster-native: check PASS, 0 tests (no unit tests defined). Thin NAPI wrapper with no test coverage.',
  },
  {
    relPath: 'packages/agent-booster/crates/agent-booster-wasm/Cargo.toml',
    loc: 470,
    notes: 'COMPILATION-TESTED | cargo check: PASS | cargo test --lib: COMPILE FAIL (1 error) | WASM cross-compile: PASS | warnings: 2 | crate LOC: 470 | ERROR: E0599 get_config() called on Result<AgentBoosterWasm> at lib.rs:435 — missing .unwrap()',
    severity: 'MEDIUM',
    category: 'TESTING',
    description: 'agent-booster-wasm: cargo check PASS, wasm32-unknown-unknown cross-compile PASS. Test compilation fails: lib.rs:435 calls .get_config() on Result<AgentBoosterWasm, JsValue> without unwrapping. Only test code is broken; WASM bindings themselves are valid.',
  },
  {
    relPath: 'reasoningbank/crates/reasoningbank-core/Cargo.toml',
    loc: 773,
    notes: 'COMPILATION-TESTED | cargo check: PASS | cargo test --lib: 12 passed, 0 failed, 0 ignored | warnings: 0 | crate LOC: 773',
    severity: 'INFO',
    category: 'QUALITY',
    description: 'reasoningbank-core: fully clean. 12/12 tests pass — engine, pattern, similarity, error all green.',
  },
  {
    relPath: 'reasoningbank/crates/reasoningbank-storage/Cargo.toml',
    loc: 1403,
    notes: 'COMPILATION-TESTED | cargo check: PASS | cargo test --lib: 9 passed, 0 failed, 0 ignored | warnings: 2 | crate LOC: 1403',
    severity: 'INFO',
    category: 'QUALITY',
    description: 'reasoningbank-storage: 9/9 tests pass. 2 dead-code warnings (db_path, config fields never read). SQLite/async/migrations all green.',
  },
  {
    relPath: 'reasoningbank/crates/reasoningbank-learning/Cargo.toml',
    loc: 788,
    notes: 'COMPILATION-TESTED | cargo check: PASS | cargo test --lib: 7 passed, 0 failed, 0 ignored | warnings: 8 (deprecated AsyncLearner self-references) | crate LOC: 788',
    severity: 'INFO',
    category: 'QUALITY',
    description: 'reasoningbank-learning: 7/7 tests pass. 8 deprecation warnings: AsyncLearner struct deprecated in favour of AsyncLearnerV2 but still exported and self-referenced.',
  },
  {
    relPath: 'reasoningbank/crates/reasoningbank-network/Cargo.toml',
    loc: 2647,
    notes: 'COMPILATION-TESTED | cargo check: PASS | cargo test --lib: 18 passed, 0 failed, 0 ignored | warnings: 2 | crate LOC: 2647',
    severity: 'INFO',
    category: 'QUALITY',
    description: 'reasoningbank-network: 18/18 tests pass. QUIC, NeuralBus gossip/priority/snapshot/streams all green. 2 dead-code warnings (config fields).',
  },
  {
    relPath: 'reasoningbank/crates/reasoningbank-mcp/Cargo.toml',
    loc: 1037,
    notes: 'COMPILATION-TESTED | cargo check: FAIL (6 errors) | cargo test --lib: NOT RUN | warnings: 4 | crate LOC: 1037 | ROOT CAUSE: mismatched StorageConfig types + missing serde::Deserialize derive on reasoningbank_storage::StorageConfig | server.rs:65',
    severity: 'CRITICAL',
    category: 'BUG',
    description: 'reasoningbank-mcp: cargo check FAILS with 6 errors. Root cause: mismatched StorageConfig types — server.rs:65 passes reasoningbank_storage::StorageConfig where local StorageConfig expected (E0308). StorageConfig lacks serde Deserialize derive (3x E0277). Workspace type aliasing bug: two StorageConfig structs with same name in different crates.',
  },
  {
    relPath: 'reasoningbank/crates/reasoningbank-wasm/Cargo.toml',
    loc: 201,
    notes: 'COMPILATION-TESTED | cargo check: FAIL on native (6 errors) | WASM cross-compile: PASS | warnings: 2 native / 2 WASM | crate LOC: 201 | NATIVE ERRORS: E0432 adapters::wasm module cfg-gated to wasm target + E0282 type inference for Storage::new()',
    severity: 'MEDIUM',
    category: 'ARCHITECTURE',
    description: 'reasoningbank-wasm: native cargo check FAILS (6 errors) — imports reasoningbank_storage::adapters::wasm which is cfg(target_family=wasm)-gated, not available on native. wasm32-unknown-unknown cross-compile PASSES cleanly. Build-target configuration issue, not a logic bug.',
  },
  {
    relPath: 'crates/agentic-flow-quic/Cargo.toml',
    loc: 999,
    notes: 'COMPILATION-TESTED | cargo check: PASS | cargo test --lib: 8 passed, 0 failed, 0 ignored | warnings: 0 | crate LOC: 999',
    severity: 'INFO',
    category: 'QUALITY',
    description: 'agentic-flow-quic: fully clean. 8/8 lib tests pass — error, client, server, config, message types all green. Prior SIGABRT was from integration test harness; --lib passes completely.',
  },
  {
    relPath: 'packages/agentic-jujutsu/Cargo.toml',
    loc: 9138,
    notes: 'COMPILATION-TESTED | cargo check: PASS | cargo test --lib: 83 passed, 5 failed, 0 ignored | warnings: 16 | crate LOC: 9138 | FAILURES: agent_coordination::test_conflict_detection, config::test_default_config, crypto::test_verify_invalid_signature, crypto::test_verify_wrong_public_key, operations::test_operation_creation',
    severity: 'HIGH',
    category: 'TESTING',
    description: 'agentic-jujutsu: 83/88 tests pass, 5 fail. CRITICAL: ML-DSA signature verification does not reject invalid signatures or wrong public keys (crypto.rs:341, 354) — security property violations. Config test fails because jj binary is cached to ~/.cache path. Operations case mismatch (describe vs Describe). Agent coordination conflict severity threshold not met.',
  },
];

const insertRead = db.prepare("INSERT INTO file_reads (file_id, session_id, depth, lines_read, notes) VALUES (?, ?, 'SURFACE', 0, ?)");
const insertFinding = db.prepare('INSERT INTO findings (file_id, session_id, severity, category, description) VALUES (?, ?, ?, ?, ?)');

for (const c of crates) {
  const fid = ensureFile(c.relPath, c.loc);
  insertRead.run(fid, SESSION_ID, c.notes);
  insertFinding.run(fid, SESSION_ID, c.severity, c.category, c.description);
  console.log('Inserted:', c.relPath, '-> file_id', fid);
}

db.close();
console.log('Done. Session:', SESSION_ID);
