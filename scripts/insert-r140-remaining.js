// R140 — insert remaining 4 crate records
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const SESSION_ID = 141;

function ensureFile(relPath, loc) {
  let row = db.prepare('SELECT id FROM files WHERE relative_path = ? AND package_id = 10').get(relPath);
  if (row) { return row.id; }
  const r = db.prepare("INSERT INTO files (package_id, relative_path, loc, depth) VALUES (10, ?, ?, 'SURFACE')").run(relPath, loc);
  return r.lastInsertRowid;
}

const insertRead = db.prepare("INSERT INTO file_reads (file_id, session_id, depth, lines_read, notes) VALUES (?, ?, 'SURFACE', 0, ?)");
const insertFinding = db.prepare('INSERT INTO findings (file_id, session_id, severity, category, description) VALUES (?, ?, ?, ?, ?)');

// 1. Insert missing finding for reasoningbank-mcp (file_read already inserted in prior run)
const mcpRow = db.prepare('SELECT id FROM files WHERE relative_path = ? AND package_id = 10').get('reasoningbank/crates/reasoningbank-mcp/Cargo.toml');
const mcpId = mcpRow.id;
insertFinding.run(mcpId, SESSION_ID, 'CRITICAL', 'BUG',
  'reasoningbank-mcp: cargo check FAILS with 6 errors. Root cause: mismatched StorageConfig types — server.rs:65 passes reasoningbank_storage::StorageConfig where local StorageConfig expected (E0308). StorageConfig lacks serde Deserialize derive (3x E0277). Workspace type aliasing bug: two StorageConfig structs with same name in different crates.');
console.log('Finding inserted for reasoningbank-mcp, file_id', mcpId);

// 2. reasoningbank-wasm
const wasmId = ensureFile('reasoningbank/crates/reasoningbank-wasm/Cargo.toml', 201);
insertRead.run(wasmId, SESSION_ID,
  'COMPILATION-TESTED | cargo check: FAIL on native (6 errors) | WASM cross-compile: PASS | warnings: 2 native / 2 WASM | crate LOC: 201 | NATIVE ERRORS: E0432 adapters::wasm module cfg-gated to wasm target + E0282 type inference for Storage::new()');
insertFinding.run(wasmId, SESSION_ID, 'MEDIUM', 'ARCHITECTURE',
  'reasoningbank-wasm: native cargo check FAILS (6 errors) — imports reasoningbank_storage::adapters::wasm which is cfg(target_family=wasm)-gated, not available on native. wasm32-unknown-unknown cross-compile PASSES cleanly. Build-target configuration issue, not a logic bug.');
console.log('Inserted reasoningbank-wasm, file_id', wasmId);

// 3. agentic-flow-quic
const quicId = ensureFile('crates/agentic-flow-quic/Cargo.toml', 999);
insertRead.run(quicId, SESSION_ID,
  'COMPILATION-TESTED | cargo check: PASS | cargo test --lib: 8 passed, 0 failed, 0 ignored | warnings: 0 | crate LOC: 999');
insertFinding.run(quicId, SESSION_ID, 'INFO', 'QUALITY',
  'agentic-flow-quic: fully clean. 8/8 lib tests pass — error, client, server, config, message types all green. Prior SIGABRT was from integration test harness; --lib passes completely.');
console.log('Inserted agentic-flow-quic, file_id', quicId);

// 4. agentic-jujutsu
const jjId = ensureFile('packages/agentic-jujutsu/Cargo.toml', 9138);
insertRead.run(jjId, SESSION_ID,
  'COMPILATION-TESTED | cargo check: PASS | cargo test --lib: 83 passed, 5 failed, 0 ignored | warnings: 16 | crate LOC: 9138 | FAILURES: agent_coordination::test_conflict_detection, config::test_default_config, crypto::test_verify_invalid_signature, crypto::test_verify_wrong_public_key, operations::test_operation_creation');
insertFinding.run(jjId, SESSION_ID, 'HIGH', 'SECURITY',
  'agentic-jujutsu: 83/88 tests pass, 5 fail. CRITICAL security: ML-DSA signature verification does not reject invalid signatures or wrong public keys (crypto.rs:341, 354) — security property violations. Config test fails because jj binary cached to ~/.cache path. Operations case mismatch (describe vs Describe). Agent coordination conflict severity threshold not met.');
console.log('Inserted agentic-jujutsu, file_id', jjId);

db.close();
console.log('All done.');
