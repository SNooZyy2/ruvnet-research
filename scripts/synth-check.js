#!/usr/bin/env node
/**
 * Synthesis Staleness Checker
 *
 * Shows which domain synthesis files need updating after a research session.
 * Usage:
 *   node scripts/synth-check.js          # check against latest session
 *   node scripts/synth-check.js 38       # check against session R38
 *   node scripts/synth-check.js --all    # show staleness for all domains
 */

const Database = require('better-sqlite3');
const path = require('path');
const fs = require('fs');

const DB_PATH = path.join(__dirname, '../db/research.db');
const DOMAINS_DIR = path.join(__dirname, '../domains');

const db = new Database(DB_PATH, { readonly: true });

// --- Queries ---

const Q_LATEST_SESSION = `SELECT id, name, date, focus FROM sessions ORDER BY id DESC LIMIT 1`;

const Q_SESSION_BY_ID = `SELECT id, name, date, focus FROM sessions WHERE id = ?`;

// Domains that got new findings in a given session
const Q_DOMAINS_WITH_FINDINGS = `
  SELECT d.id, d.name, d.priority,
    COUNT(*) as finding_count,
    SUM(CASE WHEN f.severity = 'CRITICAL' THEN 1 ELSE 0 END) as critical,
    SUM(CASE WHEN f.severity = 'HIGH' THEN 1 ELSE 0 END) as high,
    SUM(CASE WHEN f.severity = 'MEDIUM' THEN 1 ELSE 0 END) as medium,
    SUM(CASE WHEN f.severity = 'INFO' THEN 1 ELSE 0 END) as info
  FROM findings f
  JOIN file_domains fd ON f.file_id = fd.file_id
  JOIN domains d ON fd.domain_id = d.id
  WHERE f.session_id = ?
  GROUP BY d.id
  ORDER BY critical DESC, high DESC, finding_count DESC
`;

// Domains that got new file reads in a given session
const Q_DOMAINS_WITH_READS = `
  SELECT d.id, d.name, d.priority,
    COUNT(DISTINCT fr.file_id) as files_read,
    SUM(fr.lines_read) as lines_read
  FROM file_reads fr
  JOIN file_domains fd ON fr.file_id = fd.file_id
  JOIN domains d ON fd.domain_id = d.id
  WHERE fr.session_id = ?
  GROUP BY d.id
  ORDER BY files_read DESC
`;

// For each domain: last session that touched it, total findings, coverage
const Q_DOMAIN_STALENESS = `
  SELECT
    d.id,
    d.name,
    d.priority,
    (SELECT MAX(f2.session_id) FROM findings f2
     JOIN file_domains fd2 ON f2.file_id = fd2.file_id
     WHERE fd2.domain_id = d.id) as last_finding_session,
    (SELECT MAX(fr2.session_id) FROM file_reads fr2
     JOIN file_domains fd3 ON fr2.file_id = fd3.file_id
     WHERE fd3.domain_id = d.id) as last_read_session,
    (SELECT COUNT(*) FROM findings f3
     JOIN file_domains fd4 ON f3.file_id = fd4.file_id
     WHERE fd4.domain_id = d.id) as total_findings,
    (SELECT COUNT(*) FROM file_domains fd5 WHERE fd5.domain_id = d.id) as total_files,
    (SELECT COUNT(*) FROM file_domains fd6
     JOIN files fi ON fd6.file_id = fi.id
     WHERE fd6.domain_id = d.id AND fi.depth = 'DEEP') as deep_files
  FROM domains d
  ORDER BY d.priority, d.name
`;

// --- Helpers ---

function pad(str, len) {
  str = String(str);
  return str.length >= len ? str.substring(0, len) : str + ' '.repeat(len - str.length);
}

function rpad(str, len) {
  str = String(str);
  return str.length >= len ? str : ' '.repeat(len - str.length) + str;
}

function synthFileAge(domainName) {
  const filePath = path.join(DOMAINS_DIR, domainName, 'analysis.md');
  if (!fs.existsSync(filePath)) return 'MISSING';
  const stat = fs.statSync(filePath);
  const days = Math.floor((Date.now() - stat.mtimeMs) / 86400000);
  if (days === 0) return 'today';
  if (days === 1) return '1 day';
  return `${days} days`;
}

// --- Main ---

const arg = process.argv[2];

if (arg === '--all') {
  // Show staleness for all domains
  const latestSession = db.prepare(Q_LATEST_SESSION).get();
  const rows = db.prepare(Q_DOMAIN_STALENESS).all();

  console.log(`\n  Domain Synthesis Staleness (latest session: R${latestSession.id})\n`);
  console.log(
    '  ' + pad('Domain', 28) +
    rpad('Pri', 4) +
    rpad('Files', 6) +
    rpad('Deep', 6) +
    rpad('Finds', 7) +
    rpad('LastR', 6) +
    '  ' + pad('File Age', 10) +
    '  Status'
  );
  console.log('  ' + '-'.repeat(90));

  for (const r of rows) {
    const lastR = Math.max(r.last_finding_session || 0, r.last_read_session || 0);
    const gap = lastR ? latestSession.id - lastR : null;
    const fileAge = synthFileAge(r.name);

    let status;
    if (gap === null) status = 'NO DATA';
    else if (gap === 0) status = 'NEEDS UPDATE';
    else if (gap <= 2) status = 'recent';
    else if (gap <= 5) status = 'stale';
    else status = 'very stale';

    console.log(
      '  ' + pad(r.name, 28) +
      rpad(r.priority, 4) +
      rpad(r.total_files, 6) +
      rpad(r.deep_files, 6) +
      rpad(r.total_findings, 7) +
      rpad(lastR ? 'R' + lastR : '-', 6) +
      '  ' + pad(fileAge, 10) +
      '  ' + status
    );
  }

  console.log();

} else {
  // Check specific session (or latest)
  let session;
  if (arg) {
    session = db.prepare(Q_SESSION_BY_ID).get(parseInt(arg, 10));
    if (!session) {
      console.error(`Session ${arg} not found`);
      process.exit(1);
    }
  } else {
    session = db.prepare(Q_LATEST_SESSION).get();
  }

  console.log(`\n  Session: R${session.id} — ${session.focus}`);
  console.log(`  Date: ${session.date}\n`);

  // Domains with new findings
  const findings = db.prepare(Q_DOMAINS_WITH_FINDINGS).all(session.id);
  const reads = db.prepare(Q_DOMAINS_WITH_READS).all(session.id);

  // Merge into a single map
  const domainMap = new Map();
  for (const f of findings) {
    domainMap.set(f.id, { ...f, files_read: 0, lines_read: 0 });
  }
  for (const r of reads) {
    if (domainMap.has(r.id)) {
      domainMap.get(r.id).files_read = r.files_read;
      domainMap.get(r.id).lines_read = r.lines_read;
    } else {
      domainMap.set(r.id, {
        ...r,
        finding_count: 0, critical: 0, high: 0, medium: 0, info: 0
      });
    }
  }

  const affected = [...domainMap.values()];

  if (affected.length === 0) {
    console.log('  No domains affected by this session.\n');
  } else {
    console.log(`  Domains needing synthesis update: ${affected.length}\n`);
    console.log(
      '  ' + pad('Domain', 28) +
      rpad('Files', 6) +
      rpad('Lines', 8) +
      rpad('Finds', 7) +
      rpad('CRIT', 5) +
      rpad('HIGH', 5) +
      '  ' + pad('File Age', 10)
    );
    console.log('  ' + '-'.repeat(75));

    for (const d of affected) {
      console.log(
        '  ' + pad(d.name, 28) +
        rpad(d.files_read, 6) +
        rpad(d.lines_read, 8) +
        rpad(d.finding_count, 7) +
        rpad(d.critical, 5) +
        rpad(d.high, 5) +
        '  ' + synthFileAge(d.name)
      );
    }

    console.log('\n  Suggested update commands:');
    for (const d of affected) {
      console.log(`    - Update domains/${d.name}/analysis.md`);
    }
    console.log();
  }
}

db.close();
