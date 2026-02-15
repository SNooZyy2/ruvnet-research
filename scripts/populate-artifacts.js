#!/usr/bin/env node

/**
 * populate-artifacts.js
 *
 * Maps compiled WASM and native addon files to their source Rust crates,
 * updates knowability classifications, and tags files with domain associations.
 *
 * Usage: node scripts/populate-artifacts.js
 */

const Database = require('better-sqlite3');
const fs = require('fs');
const path = require('path');

const DB_PATH = '/home/snoozyy/ruvnet-research/db/research.db';
const REPO_ROOT = '/home/snoozyy/ruvnet-research';

// Hardcoded mappings from research findings
const ARTIFACT_MAPPINGS = [
  // WASM blobs → Rust crates
  {
    artifact_query: "relative_path LIKE '%guidance_kernel_bg.wasm'",
    source_crate_name: null,  // Source not found - OPAQUE
    target_triple: 'wasm32-unknown-unknown',
    artifact_type: 'wasm-bindgen',
    notes: 'Source repo unknown. Not on crates.io. Security-critical: HMAC, chain verification, secret scanning.'
  },
  {
    artifact_query: "relative_path LIKE '%reasoningbank_wasm_bg.wasm'",
    source_crate_name: 'reasoningbank-wasm',  // from agentic-flow repo
    target_triple: 'wasm32-unknown-unknown',
    artifact_type: 'wasm-bindgen',
    notes: 'Source: agentic-flow/reasoningbank/crates/reasoningbank-wasm/'
  },
  {
    artifact_query: "relative_path LIKE '%agentic_flow_quic_bg.wasm'",
    source_crate_name: 'agentic-flow-quic',  // from agentic-flow repo
    target_triple: 'wasm32-unknown-unknown',
    artifact_type: 'wasm-bindgen',
    notes: 'Source: agentic-flow/crates/agentic-flow-quic/'
  }
];

// Domain associations for Rust packages
const DOMAIN_TAGS = [
  {
    package_pattern: 'ruvector-rust',
    domain_name: 'ruvector',
    relevance: 1.0
  },
  {
    package_pattern: 'agentic-flow-rust',
    domain_name: 'agentic-flow',
    relevance: 0.9
  },
  {
    package_pattern: 'agentic-flow-rust',
    domain_name: 'memory-and-learning',
    relevance: 0.8,
    file_pattern: '%reasoningbank%'
  },
  {
    package_pattern: 'ruv-fann-rust',
    domain_name: 'swarm-coordination',
    relevance: 0.7,
    file_pattern: '%swarm%'
  },
  {
    package_pattern: 'sublinear-rust',
    domain_name: 'memory-and-learning',
    relevance: 0.6
  },
];

class ArtifactPopulator {
  constructor(dbPath) {
    this.db = new Database(dbPath);
    this.stats = {
      artifacts_mapped: 0,
      artifacts_with_source: 0,
      artifacts_without_source: 0,
      knowability_updated: 0,
      domain_tags_added: 0,
      opaque_files: []
    };
  }

  close() {
    this.db.close();
  }

  /**
   * Main execution flow
   */
  run() {
    console.log('=== Artifact Mapping and Knowability Update ===\n');

    try {
      this.db.prepare('BEGIN TRANSACTION').run();

      this.mapArtifacts();
      this.updateKnowability();
      this.tagDomains();
      this.printSummary();

      this.db.prepare('COMMIT').run();
      console.log('\n✓ Transaction committed successfully');
    } catch (error) {
      this.db.prepare('ROLLBACK').run();
      console.error('\n✗ Error occurred, rolling back:', error.message);
      throw error;
    }
  }

  /**
   * Map artifact files to their source crates
   */
  mapArtifacts() {
    console.log('1. Mapping artifacts to source crates...\n');

    const insertArtifact = this.db.prepare(`
      INSERT OR REPLACE INTO artifacts (
        artifact_file_id,
        source_crate_id,
        target_triple,
        artifact_type,
        size_bytes,
        notes
      ) VALUES (?, ?, ?, ?, ?, ?)
    `);

    for (const mapping of ARTIFACT_MAPPINGS) {
      const files = this.db.prepare(`
        SELECT id, relative_path FROM files WHERE ${mapping.artifact_query}
      `).all();

      if (files.length === 0) {
        console.log(`   ⚠ No files found for: ${mapping.artifact_query}`);
        continue;
      }

      let sourceCrateId = null;
      if (mapping.source_crate_name) {
        const crate = this.db.prepare(`
          SELECT id FROM crates WHERE name = ?
        `).get(mapping.source_crate_name);

        if (crate) {
          sourceCrateId = crate.id;
        } else {
          console.log(`   ⚠ Source crate not found: ${mapping.source_crate_name}`);
        }
      }

      for (const file of files) {
        const fullPath = path.join(REPO_ROOT, file.relative_path);
        let fileSize = null;

        try {
          const stats = fs.statSync(fullPath);
          fileSize = stats.size;
        } catch (error) {
          console.log(`   ⚠ Cannot stat file: ${file.relative_path}`);
        }

        insertArtifact.run(
          file.id,
          sourceCrateId,
          mapping.target_triple,
          mapping.artifact_type,
          fileSize,
          mapping.notes
        );

        this.stats.artifacts_mapped++;
        if (sourceCrateId) {
          this.stats.artifacts_with_source++;
          console.log(`   ✓ Mapped: ${file.relative_path} → ${mapping.source_crate_name}`);
        } else {
          this.stats.artifacts_without_source++;
          console.log(`   ✗ OPAQUE: ${file.relative_path} (no source found)`);
          this.stats.opaque_files.push(file.relative_path);
        }
      }
    }

    console.log(`\n   Total artifacts mapped: ${this.stats.artifacts_mapped}`);
    console.log(`   With source: ${this.stats.artifacts_with_source}`);
    console.log(`   Without source (OPAQUE): ${this.stats.artifacts_without_source}\n`);
  }

  /**
   * Update knowability for all wasm/node files
   */
  updateKnowability() {
    console.log('2. Updating knowability classifications...\n');

    // Files with mapped source crate are INTERFACE_ONLY
    const interfaceCount = this.db.prepare(`
      UPDATE files SET knowability = 'INTERFACE_ONLY'
      WHERE id IN (
        SELECT artifact_file_id FROM artifacts
        WHERE source_crate_id IS NOT NULL
      )
    `).run();
    console.log(`   ✓ Set INTERFACE_ONLY: ${interfaceCount.changes} files`);

    // Files without mapped source crate are OPAQUE
    const opaqueWithArtifact = this.db.prepare(`
      UPDATE files SET knowability = 'OPAQUE'
      WHERE file_type IN ('wasm', 'node')
        AND knowability = 'UNKNOWN'
        AND id IN (
          SELECT artifact_file_id FROM artifacts
          WHERE source_crate_id IS NULL
        )
    `).run();
    console.log(`   ✗ Set OPAQUE (in artifacts, no source): ${opaqueWithArtifact.changes} files`);

    // Remaining wasm/node files not in artifacts table are also OPAQUE
    const opaqueUnmapped = this.db.prepare(`
      UPDATE files SET knowability = 'OPAQUE'
      WHERE file_type IN ('wasm', 'node')
        AND knowability = 'UNKNOWN'
        AND id NOT IN (SELECT artifact_file_id FROM artifacts)
    `).run();
    console.log(`   ✗ Set OPAQUE (not in artifacts): ${opaqueUnmapped.changes} files`);

    this.stats.knowability_updated =
      interfaceCount.changes + opaqueWithArtifact.changes + opaqueUnmapped.changes;

    // Collect all OPAQUE files for security review
    const opaqueFiles = this.db.prepare(`
      SELECT relative_path FROM files
      WHERE knowability = 'OPAQUE'
      ORDER BY relative_path
    `).all();

    this.stats.opaque_files = opaqueFiles.map(f => f.relative_path);

    console.log(`\n   Total knowability updates: ${this.stats.knowability_updated}\n`);
  }

  /**
   * Tag files with domain associations
   */
  tagDomains() {
    console.log('3. Tagging files with domain associations...\n');

    const insertDomain = this.db.prepare(`
      INSERT OR IGNORE INTO file_domains (file_id, domain_id)
      VALUES (?, ?)
    `);

    for (const tag of DOMAIN_TAGS) {
      // Get domain ID
      const domain = this.db.prepare(`
        SELECT id FROM domains WHERE name = ?
      `).get(tag.domain_name);

      if (!domain) {
        console.log(`   ⚠ Domain not found: ${tag.domain_name}`);
        continue;
      }

      // Build query to find matching files
      let fileQuery = `
        SELECT DISTINCT f.id, f.relative_path
        FROM files f
        JOIN packages p ON f.package_id = p.id
        WHERE p.name LIKE ?
      `;
      const params = [`%${tag.package_pattern}%`];

      if (tag.file_pattern) {
        fileQuery += ` AND f.relative_path LIKE ?`;
        params.push(tag.file_pattern);
      }

      const files = this.db.prepare(fileQuery).all(...params);

      if (files.length === 0) {
        console.log(`   ⚠ No files found for: ${tag.package_pattern}${tag.file_pattern ? ' + ' + tag.file_pattern : ''}`);
        continue;
      }

      let tagged = 0;
      for (const file of files) {
        const result = insertDomain.run(file.id, domain.id);
        if (result.changes > 0) {
          tagged++;
        }
      }

      this.stats.domain_tags_added += tagged;
      console.log(`   ✓ Tagged ${tagged} files: ${tag.package_pattern} → ${tag.domain_name} (relevance: ${tag.relevance})`);
    }

    console.log(`\n   Total domain tags added: ${this.stats.domain_tags_added}\n`);
  }

  /**
   * Print comprehensive summary
   */
  printSummary() {
    console.log('=== Summary ===\n');

    // Artifact mapping summary
    console.log('Artifacts Mapped:');
    console.log(`  Total: ${this.stats.artifacts_mapped}`);
    console.log(`  With source: ${this.stats.artifacts_with_source}`);
    console.log(`  Without source (OPAQUE): ${this.stats.artifacts_without_source}\n`);

    // Knowability distribution
    console.log('Knowability Distribution:');
    const knowabilityStats = this.db.prepare(`
      SELECT knowability, COUNT(*) as count
      FROM files
      GROUP BY knowability
      ORDER BY
        CASE knowability
          WHEN 'READABLE' THEN 1
          WHEN 'INTERFACE_ONLY' THEN 2
          WHEN 'OPAQUE' THEN 3
          WHEN 'UNKNOWN' THEN 4
          ELSE 5
        END
    `).all();

    for (const stat of knowabilityStats) {
      console.log(`  ${stat.knowability}: ${stat.count}`);
    }
    console.log('');

    // Domain coverage
    console.log('Domain Tags Added:');
    console.log(`  Total: ${this.stats.domain_tags_added}\n`);

    const domainCoverage = this.db.prepare(`
      SELECT d.name, COUNT(DISTINCT fd.file_id) as file_count
      FROM domains d
      LEFT JOIN file_domains fd ON d.id = fd.domain_id
      GROUP BY d.id, d.name
      HAVING file_count > 0
      ORDER BY file_count DESC
    `).all();

    for (const domain of domainCoverage) {
      console.log(`  ${domain.name}: ${domain.file_count} files`);
    }
    console.log('');

    // Security concerns: OPAQUE files
    if (this.stats.opaque_files.length > 0) {
      console.log('⚠ SECURITY CONCERN: OPAQUE Files (no source code available)');
      console.log(`  Count: ${this.stats.opaque_files.length}\n`);

      console.log('  Files:');
      for (const file of this.stats.opaque_files.slice(0, 20)) {
        console.log(`    - ${file}`);
      }

      if (this.stats.opaque_files.length > 20) {
        console.log(`    ... and ${this.stats.opaque_files.length - 20} more`);
      }
      console.log('');
      console.log('  Recommendation: Perform security audit on these binaries or request source code.');
    } else {
      console.log('✓ No OPAQUE files found - all artifacts have mapped sources\n');
    }

    // Files still with UNKNOWN knowability
    const unknownCount = this.db.prepare(`
      SELECT COUNT(*) as count FROM files WHERE knowability = 'UNKNOWN'
    `).get();

    if (unknownCount.count > 0) {
      console.log(`⚠ ${unknownCount.count} files still have UNKNOWN knowability`);
      console.log('  These may need manual classification or additional artifact mappings.\n');
    }
  }
}

// Main execution
if (require.main === module) {
  const populator = new ArtifactPopulator(DB_PATH);

  try {
    populator.run();
  } catch (error) {
    console.error('\nFATAL ERROR:', error);
    process.exit(1);
  } finally {
    populator.close();
  }
}

module.exports = ArtifactPopulator;
