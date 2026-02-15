#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const Database = require(path.join(process.env.HOME, 'node_modules', 'better-sqlite3'));

const DB_PATH = '/home/snoozyy/ruvnet-research/db/research.db';

// Simple TOML parser for Cargo.toml format
class CargoTomlParser {
  constructor(content) {
    this.lines = content.split('\n').map(line => {
      // Remove comments
      const commentIdx = line.indexOf('#');
      return commentIdx >= 0 ? line.substring(0, commentIdx) : line;
    }).map(line => line.trim()).filter(line => line.length > 0);
    this.currentSection = null;
    this.data = {};
  }

  parse() {
    for (let i = 0; i < this.lines.length; i++) {
      const line = this.lines[i];

      // Section header
      if (line.startsWith('[') && line.endsWith(']')) {
        this.currentSection = line.slice(1, -1);
        if (!this.data[this.currentSection]) {
          this.data[this.currentSection] = {};
        }
        continue;
      }

      // Key-value pair
      const eqIdx = line.indexOf('=');
      if (eqIdx > 0 && this.currentSection) {
        const key = line.substring(0, eqIdx).trim();
        const valueStr = line.substring(eqIdx + 1).trim();
        const value = this.parseValue(valueStr);
        this.data[this.currentSection][key] = value;
      }
    }
    return this.data;
  }

  parseValue(str) {
    str = str.trim();

    // String value
    if (str.startsWith('"') && str.endsWith('"')) {
      return str.slice(1, -1);
    }

    // Array value: ["a", "b", "c"]
    if (str.startsWith('[') && str.endsWith(']')) {
      const inner = str.slice(1, -1).trim();
      if (!inner) return [];
      return inner.split(',').map(item => {
        item = item.trim();
        if (item.startsWith('"') && item.endsWith('"')) {
          return item.slice(1, -1);
        }
        return item;
      });
    }

    // Inline table: { version = "1.0", path = "../foo" }
    if (str.startsWith('{') && str.endsWith('}')) {
      const inner = str.slice(1, -1).trim();
      const table = {};
      let currentKey = '';
      let currentValue = '';
      let inString = false;
      let depth = 0;

      for (let i = 0; i < inner.length; i++) {
        const char = inner[i];

        if (char === '"') {
          inString = !inString;
          currentValue += char;
        } else if (char === '[' && !inString) {
          depth++;
          currentValue += char;
        } else if (char === ']' && !inString) {
          depth--;
          currentValue += char;
        } else if (char === '=' && !inString && depth === 0) {
          currentKey = currentValue.trim();
          currentValue = '';
        } else if (char === ',' && !inString && depth === 0) {
          if (currentKey) {
            table[currentKey] = this.parseValue(currentValue.trim());
            currentKey = '';
            currentValue = '';
          }
        } else {
          currentValue += char;
        }
      }

      // Handle last key-value pair
      if (currentKey && currentValue) {
        table[currentKey] = this.parseValue(currentValue.trim());
      }

      return table;
    }

    // Number
    if (/^\d+$/.test(str)) {
      return parseInt(str, 10);
    }

    // Boolean
    if (str === 'true') return true;
    if (str === 'false') return false;

    // Default: return as string
    return str;
  }
}

// Map package names to GitHub source repos
const PACKAGE_TO_REPO = {
  'ruvector-rust': 'https://github.com/ruvnet/ruvector',
  'ruv-fann-rust': 'https://github.com/ruvnet/ruv-FANN',
  'agentic-flow-rust': 'https://github.com/ruvnet/agentic-flow',
  'sublinear-rust': 'https://github.com/ruvnet/sublinear-time-solver'
};

function main() {
  console.log('Opening database:', DB_PATH);
  const db = new Database(DB_PATH);

  try {
    // Find all Cargo.toml files from Rust packages
    const cargoFiles = db.prepare(`
      SELECT f.id, f.relative_path, f.loc, p.id as package_id, p.name as package_name, p.base_path
      FROM files f
      JOIN packages p ON f.package_id = p.id
      WHERE f.file_type = 'toml'
        AND f.relative_path LIKE '%Cargo.toml'
        AND p.name IN ('ruvector-rust', 'ruv-fann-rust', 'agentic-flow-rust', 'sublinear-rust')
      ORDER BY p.name, f.relative_path
    `).all();

    console.log(`Found ${cargoFiles.length} Cargo.toml files\n`);

    const stats = {
      totalCrates: 0,
      cratesByPackage: {},
      totalDeps: 0,
      internalDeps: 0,
      externalDeps: 0,
      devDeps: 0
    };

    const insertCrate = db.prepare(`
      INSERT OR IGNORE INTO crates (name, version, package_id, crate_type, cargo_toml_file_id, source_repo)
      VALUES (?, ?, ?, ?, ?, ?)
    `);

    const insertDep = db.prepare(`
      INSERT OR IGNORE INTO crate_dependencies (source_crate_id, target_crate_name, version_req, path, is_dev)
      VALUES (?, ?, ?, ?, ?)
    `);

    const crateIds = {}; // Map crate name -> id for dependency linking

    // First pass: insert all crates
    for (const file of cargoFiles) {
      const basePath = file.base_path.replace(/^~/, process.env.HOME || '/home/snoozyy');
      const fullPath = path.join(basePath, file.relative_path);

      if (!fs.existsSync(fullPath)) {
        console.warn(`WARNING: File not found: ${fullPath}`);
        continue;
      }

      console.log(`Parsing ${file.package_name}/${file.relative_path}`);

      const content = fs.readFileSync(fullPath, 'utf8');
      const parser = new CargoTomlParser(content);
      const toml = parser.parse();

      // Extract crate info
      const pkg = toml.package || {};
      const lib = toml.lib || {};

      const crateName = pkg.name;
      if (!crateName) {
        console.log(`  Skipping (no package.name)`);
        continue;
      }

      const version = pkg.version || '0.0.0';
      const crateTypes = lib['crate-type'] || [];
      const crateType = Array.isArray(crateTypes) ? crateTypes.join(',') : (crateTypes || 'rlib');
      const sourceRepo = PACKAGE_TO_REPO[file.package_name] || null;

      // Insert crate
      const result = insertCrate.run(
        crateName,
        version,
        file.package_id,
        crateType,
        file.id,
        sourceRepo
      );

      let crateId = result.lastInsertRowid;
      if (crateId === 0) {
        // Duplicate name — fetch existing id
        const existing = db.prepare('SELECT id FROM crates WHERE name = ?').get(crateName);
        crateId = existing ? existing.id : 0;
        console.log(`  Skipped duplicate crate: ${crateName} (existing id=${crateId})`);
      } else {
        console.log(`  Created crate: ${crateName} v${version} (id=${crateId})`);
        stats.totalCrates++;
        stats.cratesByPackage[file.package_name] = (stats.cratesByPackage[file.package_name] || 0) + 1;
      }
      crateIds[crateName] = crateId;
      if (crateId === 0) continue;  // Skip deps for unresolved duplicates

      // Process dependencies
      const deps = toml.dependencies || {};
      const devDeps = toml['dev-dependencies'] || {};

      for (const [depName, depSpec] of Object.entries(deps)) {
        let versionReq = null;
        let depPath = null;
        let features = null;

        if (typeof depSpec === 'string') {
          versionReq = depSpec;
        } else if (typeof depSpec === 'object') {
          versionReq = depSpec.version || null;
          depPath = depSpec.path || null;
          features = depSpec.features ? JSON.stringify(depSpec.features) : null;
        }

        try {
          insertDep.run(crateId, depName, versionReq, depPath, 0);
          stats.totalDeps++;
        } catch(e) {
          console.warn(`    WARN: Failed to insert dep ${depName}: ${e.message}`);
        }

        if (depPath) {
          stats.internalDeps++;
        } else {
          stats.externalDeps++;
        }
      }

      for (const [depName, depSpec] of Object.entries(devDeps)) {
        let versionReq = null;
        let depPath = null;
        let features = null;

        if (typeof depSpec === 'string') {
          versionReq = depSpec;
        } else if (typeof depSpec === 'object') {
          versionReq = depSpec.version || null;
          depPath = depSpec.path || null;
          features = depSpec.features ? JSON.stringify(depSpec.features) : null;
        }

        try {
          insertDep.run(crateId, depName, versionReq, depPath, 1);
          stats.totalDeps++;
          stats.devDeps++;
        } catch(e) {
          console.warn(`    WARN: Failed to insert dev-dep ${depName}: ${e.message}`);
        }

        if (depPath) {
          stats.internalDeps++;
        } else {
          stats.externalDeps++;
        }
      }
    }

    // Second pass: link target_crate_id for internal dependencies
    console.log('\nLinking internal crate dependencies...');
    const updateResult = db.prepare(`
      UPDATE crate_dependencies
      SET target_crate_id = (
        SELECT id FROM crates WHERE name = crate_dependencies.target_crate_name
      )
      WHERE target_crate_id IS NULL
    `).run();

    console.log(`Linked ${updateResult.changes} internal dependencies\n`);

    // Print summary
    console.log('=== CRATE POPULATION SUMMARY ===');
    console.log(`Total crates found: ${stats.totalCrates}`);
    console.log('\nCrates by package:');
    for (const [pkg, count] of Object.entries(stats.cratesByPackage)) {
      console.log(`  ${pkg}: ${count}`);
    }
    console.log(`\nTotal dependencies: ${stats.totalDeps}`);
    console.log(`  Internal (path-based): ${stats.internalDeps}`);
    console.log(`  External (crates.io): ${stats.externalDeps}`);
    console.log(`  Dev dependencies: ${stats.devDeps}`);

    // Verify internal dependency linking
    const linkedCount = db.prepare(`
      SELECT COUNT(*) as count FROM crate_dependencies WHERE target_crate_id IS NOT NULL
    `).get().count;
    console.log(`\nLinked internal dependencies: ${linkedCount}`);

  } catch (error) {
    console.error('ERROR:', error.message);
    console.error(error.stack);
    process.exit(1);
  } finally {
    db.close();
  }
}

if (require.main === module) {
  main();
}

module.exports = { CargoTomlParser };
