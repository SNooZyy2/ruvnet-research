#!/usr/bin/env node
/**
 * chunk-transcript.js — Split a large transcript into overlapping chunks
 *
 * Usage (run from repo root or ruv-vods/):
 *   node ruv-vods/scripts/chunk-transcript.js <transcript.txt> [options]
 *
 * Output:
 *   - Creates chunk files in ruv-vods/chunks/{transcript-name}/
 *   - Creates manifest.json with chunk metadata
 *   - Prints spawn instructions for transcript-analyzer agents
 */

const fs = require('fs');
const path = require('path');

const RUV_VODS_ROOT = path.resolve(__dirname, '..');

// Parse args
const args = process.argv.slice(2);
if (args.length === 0 || args[0] === '--help') {
  console.log(`Usage: node ruv-vods/scripts/chunk-transcript.js <transcript.txt> [options]

Options:
  --chunk-size N   Lines per chunk (default: 3000)
  --overlap N      Overlap lines between chunks (default: 200)
  --name NAME      Transcript name (default: filename without ext)

Input file can be:
  - Absolute path
  - Relative to CWD
  - Just a filename (looked up in ruv-vods/inbox/)

Output goes to ruv-vods/chunks/{name}/`);
  process.exit(0);
}

let inputFile = args[0];
let chunkSize = 3000;
let overlap = 200;
let transcriptName = null;

for (let i = 1; i < args.length; i += 2) {
  switch (args[i]) {
    case '--chunk-size': chunkSize = parseInt(args[i + 1]); break;
    case '--overlap': overlap = parseInt(args[i + 1]); break;
    case '--name': transcriptName = args[i + 1]; break;
  }
}

// Resolve input file — check inbox if bare filename
if (!fs.existsSync(inputFile)) {
  const inboxPath = path.join(RUV_VODS_ROOT, 'inbox', inputFile);
  if (fs.existsSync(inboxPath)) {
    inputFile = inboxPath;
  } else {
    console.error(`Error: File not found: ${inputFile}`);
    console.error(`  Also checked: ${inboxPath}`);
    process.exit(1);
  }
}

if (!transcriptName) {
  transcriptName = path.basename(inputFile, path.extname(inputFile));
}

const outDir = path.join(RUV_VODS_ROOT, 'chunks', transcriptName);

// Read transcript
const content = fs.readFileSync(inputFile, 'utf-8');
const lines = content.split('\n');
const totalLines = lines.length;

console.log(`Transcript: ${inputFile}`);
console.log(`Name: ${transcriptName}`);
console.log(`Total lines: ${totalLines}`);
console.log(`Chunk size: ${chunkSize}, Overlap: ${overlap}`);
console.log(`Output: ${outDir}/`);

// Create output directory
fs.mkdirSync(outDir, { recursive: true });

// Split into chunks
const chunks = [];
let start = 0;
let chunkNum = 1;

while (start < totalLines) {
  const end = Math.min(start + chunkSize, totalLines);
  const chunkLines = lines.slice(start, end);
  const chunkFile = `chunk-${String(chunkNum).padStart(3, '0')}.txt`;
  const chunkPath = path.join(outDir, chunkFile);

  fs.writeFileSync(chunkPath, chunkLines.join('\n'));

  chunks.push({
    id: chunkNum,
    file: chunkFile,
    path: chunkPath,
    startLine: start + 1,
    endLine: end,
    lineCount: end - start
  });

  start = end - overlap;
  if (start >= totalLines) break;
  if (end === totalLines) break;
  chunkNum++;
}

// Write manifest
const manifest = {
  transcript: transcriptName,
  sourceFile: path.resolve(inputFile),
  totalLines,
  chunkSize,
  overlap,
  chunkCount: chunks.length,
  createdAt: new Date().toISOString(),
  chunks
};

const manifestPath = path.join(outDir, 'manifest.json');
fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));

// Update master index
const indexPath = path.join(RUV_VODS_ROOT, 'index.json');
let index = { transcripts: [] };
if (fs.existsSync(indexPath)) {
  index = JSON.parse(fs.readFileSync(indexPath, 'utf-8'));
}

// Upsert entry
const existing = index.transcripts.findIndex(t => t.name === transcriptName);
const entry = {
  name: transcriptName,
  sourceFile: path.resolve(inputFile),
  totalLines,
  chunkCount: chunks.length,
  status: 'chunked',
  chunkedAt: new Date().toISOString(),
  scannedAt: null,
  analyzedAt: null,
  verifiedAt: null,
  leadsFile: null,
  leadsCount: 0,
  verifiedCount: 0
};

if (existing >= 0) {
  index.transcripts[existing] = { ...index.transcripts[existing], ...entry };
} else {
  index.transcripts.push(entry);
}

fs.writeFileSync(indexPath, JSON.stringify(index, null, 2));

console.log(`\nCreated ${chunks.length} chunks in ${outDir}/`);
console.log(`Manifest: ${manifestPath}`);
console.log(`Index updated: ${indexPath}`);

// Print summary
console.log('\nChunk summary:');
for (const c of chunks) {
  console.log(`  ${c.file}: lines ${c.startLine}-${c.endLine} (${c.lineCount} lines)`);
}

console.log(`\n--- NEXT STEPS ---`);
console.log(`1. Read ruv-vods/agents/transcript-analyzer.md`);
console.log(`2. SCAN PASS (haiku): Spawn one agent per chunk in SCAN mode`);
console.log(`3. Review scan results, identify relevant chunks`);
console.log(`4. ANALYZE PASS (sonnet): Spawn agents for relevant chunks in ANALYZE mode`);
console.log(`5. Consolidate leads into ruv-vods/leads/${transcriptName}-leads.md`);
console.log(`6. Update index.json status → 'analyzed'`);
console.log(`7. Verify leads against source code (separate step)`);
