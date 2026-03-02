const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const fileId = 4162;

// Find claude-flow CLI files for dependency edges
// This bridge calls: memory store/search/retrieve/delete, hive-mind memory, task list, hooks route, hooks post-task
// These are CLI commands — target is the claude-flow CLI package
// Look for any claude-flow cli or hooks files

const ins = db.prepare(
  'INSERT OR IGNORE INTO dependencies (source_file_id, target_file_id, relationship, evidence) VALUES (?, ?, ?, ?)'
);

// Look for hooks_integration.rs (already known, file 4162 is in same module)
// hooks_integration.rs was read in R102 — find its file id
const hooksFile = db.prepare("SELECT id, relative_path FROM files WHERE relative_path LIKE '%hooks_integration%' LIMIT 5").all();
console.log('hooks files:', JSON.stringify(hooksFile, null, 2));

// Look for ruvllm error.rs (imported as crate::error)
const errorFile = db.prepare("SELECT id, relative_path FROM files WHERE relative_path LIKE '%ruvllm%error%' LIMIT 5").all();
console.log('error files:', JSON.stringify(errorFile, null, 2));

// Look for mod.rs in context module
const modFile = db.prepare("SELECT id, relative_path FROM files WHERE relative_path LIKE '%ruvllm%context%mod%' LIMIT 5").all();
console.log('context mod files:', JSON.stringify(modFile, null, 2));

db.close();
