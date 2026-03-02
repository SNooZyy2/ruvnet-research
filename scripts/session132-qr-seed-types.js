'use strict';
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const today = new Date().toISOString().slice(0, 10);
const fileId = 14994;
const sessionId = 132;

// 1. Insert file_read record
db.prepare('INSERT INTO file_reads (file_id, session_id, depth, lines_read, line_ranges, notes) VALUES (?, ?, ?, ?, ?, ?)').run(
  fileId, sessionId, 'DEEP', 379, '1-379',
  'Full read. RVQS binary format: 64-byte header, 8 flag bits, TLV download manifest, 7 progressive layer IDs. Compile-time size assertions. Ed25519 + ML-DSA-65 sig algos declared. content_hash only 8 bytes (SHAKE-256-64 truncation). QR limit is 2,953 bytes (Version 40 Low EC), not 2000.'
);

// 2. Update file metadata to DEEP
db.prepare('UPDATE files SET depth = ?, lines_read = lines_read + ?, last_read_date = ? WHERE id = ?').run(
  'DEEP', 379, today, fileId
);

// 3. Tag with ruvector domain (id=9)
db.prepare('INSERT OR IGNORE INTO file_domains (file_id, domain_id) VALUES (?, ?)').run(fileId, 9);

// 4. Findings

const insertFinding = db.prepare(
  'INSERT INTO findings (file_id, session_id, line_start, line_end, severity, category, description, followed_up) VALUES (?, ?, ?, ?, ?, ?, ?, ?)'
);

// F1: CRITICAL — content_hash truncation to 64 bits
insertFinding.run(
  fileId, sessionId, 76, 76, 'CRITICAL', 'SECURITY',
  'content_hash is only 8 bytes (SHAKE-256-64, 64-bit truncation). Header doc says "SHAKE-256-64 of the complete expanded RVF file". 64 bits is well below the 128-bit minimum for collision resistance. A separate DL_TAG_CONTENT_HASH TLV (line 193) uses SHAKE-256-256 for the full file hash — two inconsistent hash widths for overlapping semantic purposes. The header-level hash provides insufficient tamper detection for a QR-bootstrapped trust anchor.',
  0
);

// F2: HIGH — sig_algo values not range-checked in types layer
insertFinding.run(
  fileId, sessionId, 69, 73, 'HIGH', 'SECURITY',
  'sig_algo field encodes Ed25519 (0) or ML-DSA-65 (1) but from_bytes() performs zero validation — unknown values 2-65535 are silently accepted. No enum or range guard exists in the types layer. Runtime counterpart must enforce this; if it does not, malformed seeds with invalid sig_algo would be accepted without error.',
  0
);

// F3: HIGH — host_key_hash only 16 bytes (SHAKE-256-128)
insertFinding.run(
  fileId, sessionId, 217, 219, 'HIGH', 'SECURITY',
  'HostEntry.host_key_hash is 16 bytes (SHAKE-256-128). This is the public key fingerprint for authenticating download hosts. At 128 bits this is at the minimum for collision resistance (2^64 birthday bound). For a pinning mechanism preventing MITM on RVF downloads, 256-bit (32-byte) hashes would be more appropriate. The separate TLS cert pin (DL_TAG_CERT_PIN) uses SHA-256 of SPKI which is stronger, creating an inconsistency.',
  0
);

// F4: MEDIUM — file_id only 8 bytes, no UUID alignment or generation contract
insertFinding.run(
  fileId, sessionId, 50, 50, 'MEDIUM', 'ARCHITECTURE',
  'file_id is [u8; 8] (64-bit). No specification of ID generation strategy (UUID v4 = 128-bit). If seeds are published widely and IDs collide, seed caching or deduplication logic could confuse distinct seeds. The types layer provides no generation helper or uniqueness contract.',
  0
);

// F5: MEDIUM — HostEntry.to_bytes() wire order differs from struct definition order
insertFinding.run(
  fileId, sessionId, 228, 237, 'MEDIUM', 'BUG',
  'HostEntry::to_bytes() encodes url_length first (bytes 0-1), then url (bytes 2-130), priority (130-132), region (132-134), host_key_hash (134-150). The struct definition orders fields as: url first, then url_length, priority, region, host_key_hash. The wire layout differs from the in-memory field order. HostEntry lacks #[repr(C)] (unlike SeedHeader and LayerEntry) so this is intentional, but undocumented. Makes the struct definition misleading for readers expecting field order = wire order.',
  0
);

// F6: MEDIUM — no offset/size bounds checks in from_bytes()
insertFinding.run(
  fileId, sessionId, 62, 68, 'MEDIUM', 'BUG',
  'from_bytes() deserializes microkernel_offset, microkernel_size, download_manifest_offset, download_manifest_size, and total_seed_size but performs zero internal consistency checks. A malformed seed could specify microkernel_offset + microkernel_size > total_seed_size, or overlapping regions for the two sections. The types constructor silently accepts such internally inconsistent headers.',
  0
);

// F7: INFO — GENUINE progressive layer architecture
insertFinding.run(
  fileId, sessionId, 263, 278, 'INFO', 'GENUINE',
  'layer_id module defines 7 progressive download tiers: LEVEL0 (4KB manifest), HOT_CACHE (centroids+entry points), HNSW_LAYER_A (recall>=0.70), QUANT_DICT, HNSW_LAYER_B (recall>=0.85), FULL_VECTORS, HNSW_LAYER_C (recall>=0.95). Tiered recall-vs-bandwidth tradeoff mirrors production ANN retrieval systems. Recall thresholds (0.70/0.85/0.95) are specific and meaningful, not placeholder values.',
  0
);

// F8: INFO — post-quantum ML-DSA-65 explicitly supported
insertFinding.run(
  fileId, sessionId, 69, 70, 'INFO', 'ARCHITECTURE',
  'sig_algo=1 maps to ML-DSA-65 (CRYSTALS-Dilithium NIST PQC standard). RVQS format is designed for post-quantum readiness alongside Ed25519. sig_length field is variable-width (u16) to accommodate algorithm output size differences (Ed25519=64 bytes, ML-DSA-65=3293 bytes). Forward-compatible design.',
  0
);

// F9: INFO — compile-time size assertions enforce ABI stability
insertFinding.run(
  fileId, sessionId, 79, 79, 'INFO', 'QUALITY',
  'Two compile-time const assertions enforce exact struct sizes: SeedHeader == 64 bytes (line 79) and LayerEntry == 28 bytes (line 260). Prevents accidental ABI breakage from field additions. Test suite also validates sizes at runtime (header_size_is_64, layer_entry_size_is_28). Good format stability practice.',
  0
);

// F10: INFO — QR limit is 2,953 bytes not 2,000 (lead context correction)
insertFinding.run(
  fileId, sessionId, 10, 11, 'INFO', 'DOCUMENTATION',
  'Lead context (Feb26-004) referenced a "2000-byte constraint" — this is incorrect. Actual limit is QR_MAX_BYTES = 2,953 (QR Version 40, Low EC binary capacity). Test confirms exactly 2953 passes and 2954 fails. The 64-byte header leaves 2,889 bytes for microkernel + manifest + signature combined.',
  0
);

// 5. Dependency edges
const runtimeFile = db.prepare("SELECT id FROM files WHERE relative_path LIKE '%rvf-runtime%qr_seed%'").get();
if (runtimeFile) {
  db.prepare('INSERT OR IGNORE INTO dependencies (source_file_id, target_file_id, relationship, evidence) VALUES (?, ?, ?, ?)').run(
    fileId, runtimeFile.id, 'SIBLINGS',
    'rvf-types/src/qr_seed.rs provides type definitions; rvf-runtime/src/qr_seed.rs implements runtime logic (sig verification, bounds checks, WASM execution) using these types'
  );
  console.log('Runtime sibling dep inserted, id:', runtimeFile.id);
} else {
  console.log('Runtime qr_seed not in DB yet');
}

const errorFile = db.prepare("SELECT id FROM files WHERE relative_path LIKE '%rvf-types%error%'").get();
if (errorFile) {
  db.prepare('INSERT OR IGNORE INTO dependencies (source_file_id, target_file_id, relationship, evidence) VALUES (?, ?, ?, ?)').run(
    fileId, errorFile.id, 'USES',
    'from_bytes() returns crate::error::RvfError::SizeMismatch and BadMagic variants (lines 138-150)'
  );
  console.log('Error dep inserted, id:', errorFile.id);
} else {
  console.log('rvf-types error module not in DB yet');
}

const findingCount = db.prepare('SELECT COUNT(*) as n FROM findings WHERE file_id = ? AND session_id = ?').get(fileId, sessionId);
console.log('Findings inserted this session:', findingCount.n);

const fileCheck = db.prepare('SELECT depth, lines_read, last_read_date FROM files WHERE id = ?').get(fileId);
console.log('File updated:', JSON.stringify(fileCheck));

console.log('All DB updates complete.');
db.close();
