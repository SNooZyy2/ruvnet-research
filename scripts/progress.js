#!/usr/bin/env node
/**
 * Research progress indicator — shows discovery % by domain and overall.
 * Usage: node scripts/progress.js
 */
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');

const totalFiles = db.prepare('SELECT COUNT(*) as c FROM files').get().c;
const totalLoc = db.prepare('SELECT SUM(loc) as s FROM files').get().s;

const touched = db.prepare("SELECT COUNT(*) as c FROM files WHERE depth != 'NOT_TOUCHED'").get().c;
const touchedLoc = db.prepare("SELECT SUM(loc) as s FROM files WHERE depth != 'NOT_TOUCHED'").get().s || 0;

const deepFiles = db.prepare("SELECT COUNT(*) as c FROM files WHERE depth = 'DEEP'").get().c;
const deepLoc = db.prepare("SELECT SUM(loc) as s FROM files WHERE depth = 'DEEP'").get().s || 0;
const medFiles = db.prepare("SELECT COUNT(*) as c FROM files WHERE depth = 'MEDIUM'").get().c;
const surfFiles = db.prepare("SELECT COUNT(*) as c FROM files WHERE depth = 'SURFACE'").get().c;

const findings = db.prepare('SELECT COUNT(*) as c FROM findings').get().c;
const critFindings = db.prepare("SELECT COUNT(*) as c FROM findings WHERE severity = 'CRITICAL'").get().c;
const highFindings = db.prepare("SELECT COUNT(*) as c FROM findings WHERE severity = 'HIGH'").get().c;

// Weighted score: DEEP=1.0, MEDIUM=0.5, SURFACE=0.2
const weightedProgress = db.prepare(`
  SELECT ROUND(
    SUM(CASE depth
      WHEN 'DEEP' THEN loc * 1.0
      WHEN 'MEDIUM' THEN loc * 0.5
      WHEN 'SURFACE' THEN loc * 0.2
      ELSE 0
    END) * 100.0 / SUM(loc), 1
  ) as pct FROM files
`).get().pct || 0;

// Bar helper
function bar(pct, width = 30) {
  const filled = Math.round(pct / 100 * width);
  return '[' + '█'.repeat(filled) + '░'.repeat(width - filled) + ']';
}

// Overall
console.log('');
console.log('╔══════════════════════════════════════════════════════════╗');
console.log('║           RESEARCH DISCOVERY PROGRESS                   ║');
console.log('╠══════════════════════════════════════════════════════════╣');
console.log('║                                                         ║');

const filePct = (touched / totalFiles * 100).toFixed(1);
const locPct = (touchedLoc / totalLoc * 100).toFixed(1);
console.log(`║  Files Touched:  ${bar(filePct)} ${filePct.padStart(5)}%  ║`);
console.log(`║  LOC Coverage:   ${bar(locPct)} ${locPct.padStart(5)}%  ║`);
console.log(`║  Weighted Score: ${bar(weightedProgress)} ${String(weightedProgress).padStart(5)}%  ║`);
console.log('║                                                         ║');
console.log(`║  ${String(touched).padStart(4)} / ${totalFiles} files   │  ${touchedLoc.toLocaleString().padStart(7)} / ${totalLoc.toLocaleString()} LOC  ║`);
console.log(`║  DEEP: ${String(deepFiles).padStart(3)}  MEDIUM: ${String(medFiles).padStart(3)}  SURFACE: ${String(surfFiles).padStart(3)}              ║`);
console.log(`║  Findings: ${findings} total (${critFindings} CRITICAL, ${highFindings} HIGH)        ║`);
console.log('║                                                         ║');
console.log('╠══════════════════════════════════════════════════════════╣');
console.log('║  BY DOMAIN (HIGH priority first)                        ║');
console.log('╠══════════════════════════════════════════════════════════╣');

const domains = db.prepare(`
  SELECT d.name, d.priority,
    COUNT(f.id) as total,
    SUM(f.loc) as total_loc,
    SUM(CASE WHEN f.depth != 'NOT_TOUCHED' THEN 1 ELSE 0 END) as touched,
    SUM(CASE WHEN f.depth != 'NOT_TOUCHED' THEN f.loc ELSE 0 END) as touched_loc,
    ROUND(SUM(CASE WHEN f.depth != 'NOT_TOUCHED' THEN f.loc ELSE 0 END) * 100.0 / SUM(f.loc), 1) as pct
  FROM domains d
  JOIN file_domains fd ON d.id = fd.domain_id
  JOIN files f ON fd.file_id = f.id
  GROUP BY d.id
  ORDER BY
    CASE d.priority WHEN 'HIGH' THEN 1 WHEN 'MEDIUM' THEN 2 ELSE 3 END,
    d.name
`).all();

for (const d of domains) {
  const pct = d.pct || 0;
  const prio = d.priority === 'HIGH' ? '!' : d.priority === 'MEDIUM' ? '~' : ' ';
  const nameStr = d.name.padEnd(22);
  const miniBar = bar(pct, 18);
  console.log(`║ ${prio} ${nameStr} ${miniBar} ${String(pct).padStart(5)}%  ║`);
}

console.log('║                                                         ║');
console.log('╠══════════════════════════════════════════════════════════╣');
console.log('║  BY PACKAGE                                             ║');
console.log('╠══════════════════════════════════════════════════════════╣');

const packages = db.prepare(`
  SELECT p.name,
    COUNT(f.id) as total,
    SUM(f.loc) as total_loc,
    SUM(CASE WHEN f.depth != 'NOT_TOUCHED' THEN 1 ELSE 0 END) as touched,
    ROUND(SUM(CASE WHEN f.depth != 'NOT_TOUCHED' THEN f.loc ELSE 0 END) * 100.0 / SUM(f.loc), 1) as pct
  FROM packages p
  JOIN files f ON f.package_id = p.id
  GROUP BY p.id
  ORDER BY pct DESC
`).all();

for (const p of packages) {
  const pct = p.pct || 0;
  const nameStr = p.name.padEnd(22);
  const miniBar = bar(pct, 18);
  console.log(`║   ${nameStr} ${miniBar} ${String(pct).padStart(5)}%  ║`);
}

console.log('║                                                         ║');
console.log('╚══════════════════════════════════════════════════════════╝');
console.log('');

// Sessions summary
const sessions = db.prepare(`
  SELECT s.id, s.name, s.date, s.focus,
    COUNT(DISTINCT fr.file_id) as files_read,
    SUM(fr.lines_read) as lines,
    (SELECT COUNT(*) FROM findings WHERE session_id = s.id) as findings
  FROM sessions s
  LEFT JOIN file_reads fr ON s.id = fr.session_id
  GROUP BY s.id ORDER BY s.id
`).all();

console.log('Sessions:');
for (const s of sessions) {
  console.log(`  ${s.name.padEnd(20)} ${s.date}  ${String(s.files_read || 0).padStart(3)} files  ${String(s.findings || 0).padStart(2)} findings  ${s.focus}`);
}
console.log('');

db.close();
