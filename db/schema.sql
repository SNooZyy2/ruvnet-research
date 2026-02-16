-- ============================================================================
-- Research Ledger Database Schema
-- ============================================================================
-- Tracks codebase analysis coverage across multiple npm packages/repos
-- Created: 2026-02-14
-- ============================================================================

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

-- ============================================================================
-- CORE ENTITIES
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Table: packages
-- Repos/npm packages being analyzed
-- ----------------------------------------------------------------------------
CREATE TABLE packages (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    base_path TEXT NOT NULL,
    repo_url TEXT,
    description TEXT,
    total_files INTEGER DEFAULT 0,
    total_loc INTEGER DEFAULT 0,
    populated_at TEXT -- ISO 8601 timestamp
);

-- ----------------------------------------------------------------------------
-- Table: files
-- Every file across all packages
-- ----------------------------------------------------------------------------
CREATE TABLE files (
    id INTEGER PRIMARY KEY,
    package_id INTEGER NOT NULL REFERENCES packages(id),
    relative_path TEXT NOT NULL,
    loc INTEGER,
    depth TEXT NOT NULL DEFAULT 'NOT_TOUCHED', -- DEEP, MEDIUM, SURFACE, MENTIONED, NOT_TOUCHED
    lines_read INTEGER DEFAULT 0,
    last_read_date TEXT, -- ISO 8601 timestamp
    notes TEXT,
    UNIQUE(package_id, relative_path)
);

CREATE INDEX idx_files_depth ON files(depth);
CREATE INDEX idx_files_package_id ON files(package_id);

-- ----------------------------------------------------------------------------
-- Table: domains
-- Functional concerns spanning multiple packages
-- ----------------------------------------------------------------------------
CREATE TABLE domains (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL, -- kebab-case
    description TEXT,
    priority TEXT NOT NULL DEFAULT 'MEDIUM', -- HIGH, MEDIUM, LOW
    synthesis_path TEXT
);

-- ----------------------------------------------------------------------------
-- Table: file_domains
-- Many-to-many junction between files and domains
-- ----------------------------------------------------------------------------
CREATE TABLE file_domains (
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    domain_id INTEGER NOT NULL REFERENCES domains(id) ON DELETE CASCADE,
    PRIMARY KEY (file_id, domain_id)
);

CREATE INDEX idx_file_domains_domain_id ON file_domains(domain_id);

-- ============================================================================
-- SESSION TRACKING
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Table: sessions
-- Research sessions
-- ----------------------------------------------------------------------------
CREATE TABLE sessions (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    date TEXT NOT NULL, -- ISO 8601 date
    focus TEXT,
    agent_count INTEGER DEFAULT 1,
    notes TEXT
);

-- ----------------------------------------------------------------------------
-- Table: file_reads
-- Per-session read history
-- ----------------------------------------------------------------------------
CREATE TABLE file_reads (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    depth TEXT NOT NULL, -- DEEP, MEDIUM, SURFACE, MENTIONED
    lines_read INTEGER,
    line_ranges TEXT, -- e.g. '1-50,170-320'
    notes TEXT
);

CREATE INDEX idx_file_reads_file_id ON file_reads(file_id);
CREATE INDEX idx_file_reads_session_id ON file_reads(session_id);

-- ============================================================================
-- FINDINGS & DEPENDENCIES
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Table: findings
-- Discoveries linked to files
-- ----------------------------------------------------------------------------
CREATE TABLE findings (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    session_id INTEGER REFERENCES sessions(id),
    line_start INTEGER,
    line_end INTEGER,
    severity TEXT NOT NULL DEFAULT 'INFO', -- CRITICAL, HIGH, MEDIUM, INFO
    category TEXT, -- real-functionality, stub, security, architecture, integration-point, bug, unknown
    description TEXT NOT NULL,
    followed_up INTEGER DEFAULT 0 -- 0 = not followed up, 1 = followed up
);

CREATE INDEX idx_findings_file_id ON findings(file_id);
CREATE INDEX idx_findings_severity ON findings(severity);
CREATE INDEX idx_findings_category ON findings(category);

-- ----------------------------------------------------------------------------
-- Table: dependencies
-- Directed graph edges between files
-- ----------------------------------------------------------------------------
CREATE TABLE dependencies (
    id INTEGER PRIMARY KEY,
    source_file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    target_file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    relationship TEXT NOT NULL, -- imports, calls, triggers, bridges, bundles, generates, reads, writes
    evidence TEXT,
    verified INTEGER NOT NULL DEFAULT 0, -- 0 = not verified, 1 = verified
    UNIQUE(source_file_id, target_file_id, relationship)
);

CREATE INDEX idx_dependencies_source ON dependencies(source_file_id);
CREATE INDEX idx_dependencies_target ON dependencies(target_file_id);
CREATE INDEX idx_dependencies_relationship ON dependencies(relationship);

-- ----------------------------------------------------------------------------
-- Table: crates
-- Rust crates discovered in Cargo.toml files
-- ----------------------------------------------------------------------------
CREATE TABLE crates (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    version TEXT,
    package_id INTEGER REFERENCES packages(id),
    crate_type TEXT,
    cargo_toml_file_id INTEGER REFERENCES files(id),
    crates_io_url TEXT,
    source_repo TEXT,
    workspace_root_id INTEGER REFERENCES crates(id),
    is_napi_wrapper INTEGER DEFAULT 0,
    notes TEXT
);

CREATE INDEX idx_crates_package ON crates(package_id);

-- ----------------------------------------------------------------------------
-- Table: crate_dependencies
-- Cargo dependency edges between crates
-- ----------------------------------------------------------------------------
CREATE TABLE crate_dependencies (
    id INTEGER PRIMARY KEY,
    source_crate_id INTEGER NOT NULL REFERENCES crates(id),
    target_crate_name TEXT NOT NULL,
    target_crate_id INTEGER REFERENCES crates(id),
    version_req TEXT,
    path TEXT,
    is_dev INTEGER DEFAULT 0,
    UNIQUE(source_crate_id, target_crate_name)
);

CREATE INDEX idx_crate_deps_source ON crate_dependencies(source_crate_id);

-- ----------------------------------------------------------------------------
-- Table: artifacts
-- Compiled binary artifacts (WASM, NAPI .node, etc.)
-- ----------------------------------------------------------------------------
CREATE TABLE artifacts (
    id INTEGER PRIMARY KEY,
    artifact_file_id INTEGER NOT NULL REFERENCES files(id),
    source_crate_id INTEGER REFERENCES crates(id),
    target_triple TEXT,
    artifact_type TEXT,
    size_bytes INTEGER,
    notes TEXT,
    UNIQUE(artifact_file_id)
);

CREATE INDEX idx_artifacts_crate ON artifacts(source_crate_id);

-- ----------------------------------------------------------------------------
-- Table: package_dependencies
-- High-level inter-package relationships
-- ----------------------------------------------------------------------------
CREATE TABLE package_dependencies (
    id INTEGER PRIMARY KEY,
    source_package_id INTEGER NOT NULL REFERENCES packages(id),
    target_package_id INTEGER NOT NULL REFERENCES packages(id),
    relationship TEXT NOT NULL, -- npm-dependency, bundles-copy, bridges-to, patches, optional
    notes TEXT,
    UNIQUE(source_package_id, target_package_id, relationship)
);

-- ----------------------------------------------------------------------------
-- Table: exclude_paths
-- Path patterns auto-filtered by priority_gaps and other views
-- Used to de-prioritize isolated subtrees (e.g. goalie, docs, dist)
-- ----------------------------------------------------------------------------
CREATE TABLE exclude_paths (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern TEXT NOT NULL UNIQUE,
    reason TEXT NOT NULL,
    added_session_id INTEGER REFERENCES sessions(id),
    added_date TEXT NOT NULL
);

-- ============================================================================
-- ANALYTICAL VIEWS
-- ============================================================================

-- ----------------------------------------------------------------------------
-- View: analysis_coverage
-- Source file coverage stats per package (excludes EXCLUDED, tests, .d.ts)
-- ----------------------------------------------------------------------------
CREATE VIEW analysis_coverage AS
SELECT
    p.name AS package_name,
    COUNT(CASE WHEN f.depth = 'DEEP' THEN 1 END) AS deep,
    COUNT(CASE WHEN f.depth = 'MEDIUM' THEN 1 END) AS medium,
    COUNT(CASE WHEN f.depth = 'SURFACE' THEN 1 END) AS surface,
    COUNT(CASE WHEN f.depth = 'NOT_TOUCHED' THEN 1 END) AS untouched,
    COUNT(*) AS total_src_files,
    SUM(CASE WHEN f.depth IN ('DEEP','MEDIUM') THEN f.loc ELSE 0 END) AS analyzed_loc,
    SUM(f.loc) AS total_loc,
    ROUND(
      CASE WHEN SUM(f.loc) = 0 THEN 0
      ELSE SUM(CASE WHEN f.depth IN ('DEEP','MEDIUM') THEN f.loc ELSE 0 END) * 100.0 / SUM(f.loc)
      END, 1
    ) AS pct_covered
FROM files f
JOIN packages p ON f.package_id = p.id
WHERE (f.relative_path LIKE '%.rs'
    OR f.relative_path LIKE '%.ts'
    OR f.relative_path LIKE '%.js')
  AND f.relative_path NOT LIKE '%.test.%'
  AND f.relative_path NOT LIKE '%.spec.%'
  AND f.relative_path NOT LIKE '%test/%'
  AND f.relative_path NOT LIKE '%tests/%'
  AND f.relative_path NOT LIKE '%__test__%'
  AND f.relative_path NOT LIKE '%.d.ts'
  AND f.depth != 'EXCLUDED'
GROUP BY p.name
ORDER BY pct_covered ASC;

-- ----------------------------------------------------------------------------
-- View: crate_coverage
-- Rust crate coverage stats (excludes EXCLUDED files)
-- ----------------------------------------------------------------------------
CREATE VIEW crate_coverage AS
SELECT c.name AS crate_name, c.version, c.crate_type,
  COUNT(DISTINCT CASE WHEN f.file_type = 'rs' THEN f.id END) AS rs_files,
  SUM(CASE WHEN f.file_type = 'rs' THEN f.loc ELSE 0 END) AS rs_loc,
  COUNT(DISTINCT CASE WHEN f.depth IN ('DEEP','MEDIUM') AND f.file_type = 'rs' THEN f.id END) AS analyzed_rs_files,
  COUNT(DISTINCT a.id) AS compiled_artifacts
FROM crates c
LEFT JOIN files f ON f.package_id = c.package_id AND f.depth != 'EXCLUDED'
LEFT JOIN artifacts a ON a.source_crate_id = c.id
GROUP BY c.id;

-- ----------------------------------------------------------------------------
-- View: opaque_inventory
-- WASM/node binary files (excludes EXCLUDED)
-- ----------------------------------------------------------------------------
CREATE VIEW opaque_inventory AS
SELECT p.name AS package, f.relative_path, f.file_type, f.knowability,
  a.target_triple, a.artifact_type, a.size_bytes
FROM files f
JOIN packages p ON f.package_id = p.id
LEFT JOIN artifacts a ON f.id = a.artifact_file_id
WHERE f.knowability IN ('OPAQUE','UNKNOWN') AND f.file_type IN ('wasm','node')
  AND f.depth != 'EXCLUDED'
ORDER BY a.size_bytes DESC NULLS LAST;

-- ----------------------------------------------------------------------------
-- View: domain_coverage
-- Coverage stats per domain (excludes EXCLUDED files)
-- ----------------------------------------------------------------------------
CREATE VIEW domain_coverage AS
SELECT
    d.id,
    d.name,
    d.priority,
    COUNT(DISTINCT f.id) AS total_files,
    COALESCE(SUM(f.loc), 0) AS total_loc,
    COUNT(DISTINCT CASE WHEN f.depth = 'DEEP' THEN f.id END) AS deep_files,
    COUNT(DISTINCT CASE WHEN f.depth = 'MEDIUM' THEN f.id END) AS medium_files,
    COUNT(DISTINCT CASE WHEN f.depth = 'SURFACE' THEN f.id END) AS surface_files,
    COUNT(DISTINCT CASE WHEN f.depth = 'NOT_TOUCHED' THEN f.id END) AS not_touched_files,
    ROUND(
        CASE
            WHEN COALESCE(SUM(f.loc), 0) = 0 THEN 0
            ELSE (
                COALESCE(SUM(CASE WHEN f.depth IN ('DEEP', 'MEDIUM', 'SURFACE') THEN f.loc ELSE 0 END), 0) * 100.0
                / SUM(f.loc)
            )
        END,
        2
    ) AS pct_loc_covered
FROM domains d
LEFT JOIN file_domains fd ON d.id = fd.domain_id
LEFT JOIN files f ON fd.file_id = f.id AND f.depth != 'EXCLUDED'
GROUP BY d.id, d.name, d.priority;

-- ----------------------------------------------------------------------------
-- View: package_coverage
-- Coverage stats per package (excludes EXCLUDED files)
-- ----------------------------------------------------------------------------
CREATE VIEW package_coverage AS
SELECT
    p.id,
    p.name,
    p.total_files,
    p.total_loc,
    COUNT(DISTINCT CASE WHEN f.depth = 'DEEP' THEN f.id END) AS deep_files,
    COUNT(DISTINCT CASE WHEN f.depth = 'MEDIUM' THEN f.id END) AS medium_files,
    COUNT(DISTINCT CASE WHEN f.depth = 'SURFACE' THEN f.id END) AS surface_files,
    COUNT(DISTINCT CASE WHEN f.depth = 'NOT_TOUCHED' THEN f.id END) AS not_touched_files,
    ROUND(
        CASE
            WHEN p.total_loc = 0 THEN 0
            ELSE (
                COALESCE(SUM(CASE WHEN f.depth IN ('DEEP', 'MEDIUM', 'SURFACE') THEN f.loc ELSE 0 END), 0) * 100.0
                / p.total_loc
            )
        END,
        2
    ) AS pct_loc_covered
FROM packages p
LEFT JOIN files f ON p.id = f.package_id AND f.depth != 'EXCLUDED'
GROUP BY p.id, p.name, p.total_files, p.total_loc;

-- ----------------------------------------------------------------------------
-- View: integration_hotspots
-- Files in 3+ domains (excludes EXCLUDED files)
-- ----------------------------------------------------------------------------
CREATE VIEW integration_hotspots AS
SELECT
    f.id,
    p.name AS package_name,
    f.relative_path,
    f.depth,
    f.loc,
    COUNT(fd.domain_id) AS domain_count,
    GROUP_CONCAT(d.name, ', ') AS domains
FROM files f
JOIN packages p ON f.package_id = p.id
JOIN file_domains fd ON f.id = fd.file_id
JOIN domains d ON fd.domain_id = d.id
WHERE f.depth != 'EXCLUDED'
GROUP BY f.id, p.name, f.relative_path, f.depth, f.loc
HAVING COUNT(fd.domain_id) >= 3
ORDER BY domain_count DESC, f.loc DESC;

-- ----------------------------------------------------------------------------
-- View: priority_gaps
-- NOT_TOUCHED files in HIGH priority domains, filtered by exclude_paths
-- ----------------------------------------------------------------------------
CREATE VIEW priority_gaps AS
SELECT
    f.id AS file_id,
    p.name AS package_name,
    f.relative_path,
    f.loc,
    d.name AS domain_name,
    d.priority
FROM files f
JOIN packages p ON f.package_id = p.id
JOIN file_domains fd ON f.id = fd.file_id
JOIN domains d ON fd.domain_id = d.id
WHERE f.depth = 'NOT_TOUCHED'
  AND d.priority = 'HIGH'
  AND (f.relative_path LIKE '%.rs'
    OR f.relative_path LIKE '%.ts'
    OR f.relative_path LIKE '%.js')
  AND f.relative_path NOT LIKE '%.test.%'
  AND f.relative_path NOT LIKE '%.spec.%'
  AND f.relative_path NOT LIKE '%test/%'
  AND f.relative_path NOT LIKE '%tests/%'
  AND f.relative_path NOT LIKE '__test__%'
  AND f.relative_path NOT LIKE '%.d.ts'
  AND f.relative_path NOT LIKE '%/plans/%'
  AND f.relative_path NOT LIKE '%/docs/%'
  AND f.relative_path NOT LIKE '%README%'
  AND NOT EXISTS (
    SELECT 1 FROM exclude_paths ep
    WHERE f.relative_path LIKE ep.pattern
  )
ORDER BY f.loc DESC;

-- ----------------------------------------------------------------------------
-- View: smart_priority_gaps
-- Relevance-tiered priority queue using directory co-location with DEEP files
-- and dependency connectivity. Tiers:
--   CONNECTED  (rank 1): Has recorded dependency to/from a DEEP file
--   OWN_CODE   (rank 1): In custom-src package (our own DDD implementation)
--   PROXIMATE  (rank 2): In same 2-level directory as 3+ DEEP files
--   NEARBY     (rank 3): In same 2-level directory as 1-2 DEEP files
--   DOMAIN_ONLY(rank 4): Only connection is HIGH-priority domain tag
-- ----------------------------------------------------------------------------
CREATE VIEW smart_priority_gaps AS
WITH
file_dirs AS (
  SELECT
    f.id,
    f.package_id,
    f.relative_path,
    CASE
      WHEN INSTR(SUBSTR(f.relative_path, INSTR(f.relative_path, '/') + 1), '/') > 0
      THEN SUBSTR(f.relative_path, 1,
        INSTR(f.relative_path, '/') + INSTR(SUBSTR(f.relative_path, INSTR(f.relative_path, '/') + 1), '/') - 1)
      WHEN INSTR(f.relative_path, '/') > 0
      THEN SUBSTR(f.relative_path, 1, INSTR(f.relative_path, '/') - 1)
      ELSE ''
    END AS dir2
  FROM files f
),
deep_dirs AS (
  SELECT
    fd.package_id,
    fd.dir2,
    COUNT(*) AS deep_count
  FROM file_dirs fd
  JOIN files f ON fd.id = f.id
  WHERE f.depth = 'DEEP'
  GROUP BY fd.package_id, fd.dir2
),
dep_connected AS (
  SELECT DISTINCT f.id AS file_id
  FROM files f
  JOIN dependencies d ON d.source_file_id = f.id OR d.target_file_id = f.id
  JOIN files peer ON (
    (d.source_file_id = f.id AND d.target_file_id = peer.id)
    OR (d.target_file_id = f.id AND d.source_file_id = peer.id)
  )
  WHERE peer.depth = 'DEEP'
    AND f.depth = 'NOT_TOUCHED'
)
SELECT
  pg.file_id,
  pg.package_name,
  pg.relative_path,
  pg.loc,
  pg.domain_name,
  pg.priority,
  CASE
    WHEN dc.file_id IS NOT NULL THEN 'CONNECTED'
    WHEN pg.package_name = 'custom-src' THEN 'OWN_CODE'
    WHEN dd.deep_count >= 3 THEN 'PROXIMATE'
    WHEN dd.deep_count >= 1 THEN 'NEARBY'
    ELSE 'DOMAIN_ONLY'
  END AS relevance_tier,
  COALESCE(dd.deep_count, 0) AS nearby_deep_files,
  CASE
    WHEN dc.file_id IS NOT NULL THEN 1
    WHEN pg.package_name = 'custom-src' THEN 1
    WHEN dd.deep_count >= 3 THEN 2
    WHEN dd.deep_count >= 1 THEN 3
    ELSE 4
  END AS tier_rank
FROM priority_gaps pg
JOIN files f ON pg.file_id = f.id
JOIN file_dirs fd ON fd.id = f.id
LEFT JOIN deep_dirs dd ON dd.package_id = f.package_id AND dd.dir2 = fd.dir2
LEFT JOIN dep_connected dc ON dc.file_id = f.id
ORDER BY tier_rank ASC, pg.loc DESC;

-- ----------------------------------------------------------------------------
-- View: unverified_deps
-- Dependency edges where verified=0
-- ----------------------------------------------------------------------------
CREATE VIEW unverified_deps AS
SELECT
    dep.id,
    ps.name AS source_package,
    fs.relative_path AS source_file,
    dep.relationship,
    pt.name AS target_package,
    ft.relative_path AS target_file,
    dep.evidence
FROM dependencies dep
JOIN files fs ON dep.source_file_id = fs.id
JOIN files ft ON dep.target_file_id = ft.id
JOIN packages ps ON fs.package_id = ps.id
JOIN packages pt ON ft.package_id = pt.id
WHERE dep.verified = 0;

-- ----------------------------------------------------------------------------
-- View: open_findings
-- CRITICAL/HIGH findings where followed_up=0 (excludes EXCLUDED files)
-- ----------------------------------------------------------------------------
CREATE VIEW open_findings AS
SELECT
    f.id,
    p.name AS package_name,
    fi.relative_path AS file_path,
    f.severity,
    f.category,
    f.description,
    f.line_start,
    f.line_end,
    s.name AS session_name,
    s.date AS session_date
FROM findings f
JOIN files fi ON f.file_id = fi.id
JOIN packages p ON fi.package_id = p.id
LEFT JOIN sessions s ON f.session_id = s.id
WHERE f.severity IN ('CRITICAL', 'HIGH')
  AND f.followed_up = 0
  AND fi.depth != 'EXCLUDED'
ORDER BY
    CASE f.severity
        WHEN 'CRITICAL' THEN 1
        WHEN 'HIGH' THEN 2
        ELSE 3
    END,
    f.id;

-- ----------------------------------------------------------------------------
-- View: package_connectivity
-- Cross-package dependency analysis per package
-- ----------------------------------------------------------------------------
CREATE VIEW package_connectivity AS
SELECT
    p.name AS package_name,
    p.total_files,
    p.total_loc,
    COALESCE(outbound.out_deps, 0) AS outbound_deps,
    COALESCE(outbound.out_packages, 0) AS out_package_count,
    COALESCE(inbound.in_deps, 0) AS inbound_deps,
    COALESCE(inbound.in_packages, 0) AS in_package_count,
    COALESCE(outbound.out_deps, 0) + COALESCE(inbound.in_deps, 0) AS total_cross_deps,
    COALESCE(analyzed.deep_cnt, 0) AS deep_files,
    COALESCE(analyzed.total_analyzed, 0) AS analyzed_files,
    CASE
      WHEN COALESCE(outbound.out_deps, 0) + COALESCE(inbound.in_deps, 0) = 0
      THEN 'ISOLATED'
      WHEN COALESCE(outbound.out_deps, 0) + COALESCE(inbound.in_deps, 0) <= 2
      THEN 'WEAKLY_CONNECTED'
      ELSE 'CONNECTED'
    END AS connectivity
FROM packages p
LEFT JOIN (
    SELECT sp.id AS pkg_id, COUNT(*) AS out_deps, COUNT(DISTINCT tp.id) AS out_packages
    FROM dependencies d
    JOIN files sf ON d.source_file_id = sf.id
    JOIN files tf ON d.target_file_id = tf.id
    JOIN packages sp ON sf.package_id = sp.id
    JOIN packages tp ON tf.package_id = tp.id
    WHERE sp.id != tp.id
    GROUP BY sp.id
) outbound ON p.id = outbound.pkg_id
LEFT JOIN (
    SELECT tp.id AS pkg_id, COUNT(*) AS in_deps, COUNT(DISTINCT sp.id) AS in_packages
    FROM dependencies d
    JOIN files sf ON d.source_file_id = sf.id
    JOIN files tf ON d.target_file_id = tf.id
    JOIN packages sp ON sf.package_id = sp.id
    JOIN packages tp ON tf.package_id = tp.id
    WHERE sp.id != tp.id
    GROUP BY tp.id
) inbound ON p.id = inbound.pkg_id
LEFT JOIN (
    SELECT package_id,
      SUM(CASE WHEN depth = 'DEEP' THEN 1 ELSE 0 END) AS deep_cnt,
      SUM(CASE WHEN depth IN ('DEEP','MEDIUM') THEN 1 ELSE 0 END) AS total_analyzed
    FROM files GROUP BY package_id
) analyzed ON p.id = analyzed.package_id
ORDER BY total_cross_deps ASC, p.total_files DESC;

-- ----------------------------------------------------------------------------
-- View: subtree_connectivity
-- Directory-level connectivity with confidence signal
-- Used to detect isolated subtrees (Goalie-like islands)
-- ----------------------------------------------------------------------------
CREATE VIEW subtree_connectivity AS
SELECT
    p.name AS package_name,
    CASE
      WHEN INSTR(f.relative_path, '/') > 0
      THEN SUBSTR(f.relative_path, 1, INSTR(f.relative_path, '/') - 1)
      ELSE f.relative_path
    END AS top_dir,
    COUNT(f.id) AS file_count,
    SUM(f.loc) AS total_loc,
    SUM(CASE WHEN f.depth = 'DEEP' THEN 1 ELSE 0 END) AS deep_files,
    SUM(CASE WHEN f.depth = 'NOT_TOUCHED' THEN 1 ELSE 0 END) AS untouched,
    COUNT(DISTINCT CASE
      WHEN d_out.target_file_id IS NOT NULL AND tf_out.package_id != f.package_id
      THEN d_out.id
    END) AS outbound_cross_deps,
    COUNT(DISTINCT CASE
      WHEN d_in.source_file_id IS NOT NULL AND sf_in.package_id != f.package_id
      THEN d_in.id
    END) AS inbound_cross_deps,
    COUNT(DISTINCT CASE
      WHEN d_out.source_file_id IS NOT NULL OR d_in.target_file_id IS NOT NULL
      THEN f.id
    END) AS files_with_deps,
    CASE
      WHEN SUM(CASE WHEN f.depth = 'DEEP' THEN 1 ELSE 0 END) = 0 THEN 'NO_DATA'
      WHEN COUNT(DISTINCT CASE
        WHEN (d_out.source_file_id IS NOT NULL OR d_in.target_file_id IS NOT NULL)
          AND f.depth = 'DEEP'
        THEN f.id END) * 1.0
        / MAX(SUM(CASE WHEN f.depth = 'DEEP' THEN 1 ELSE 0 END), 1) < 0.2
      THEN 'LOW_CONFIDENCE'
      ELSE 'RELIABLE'
    END AS confidence
FROM files f
JOIN packages p ON f.package_id = p.id
LEFT JOIN dependencies d_out ON f.id = d_out.source_file_id
LEFT JOIN files tf_out ON d_out.target_file_id = tf_out.id
LEFT JOIN dependencies d_in ON f.id = d_in.target_file_id
LEFT JOIN files sf_in ON d_in.source_file_id = sf_in.id
WHERE f.depth != 'EXCLUDED'
GROUP BY p.name, top_dir
HAVING file_count >= 5
ORDER BY p.name, outbound_cross_deps + inbound_cross_deps ASC, file_count DESC;

-- ============================================================================
-- SEED DATA
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Packages
-- ----------------------------------------------------------------------------
INSERT INTO packages (id, name, base_path, description) VALUES
(1, 'claude-flow-cli', '~/.npm-global/lib/node_modules/@claude-flow/cli/', 'Main CLI package — commands, MCP tools, services, bundled ruvector'),
(2, 'agentdb', '~/node_modules/agentdb/', 'AgentDB npm package — vector DB, episodes, skills'),
(3, 'agentic-flow', '~/node_modules/agentic-flow/', 'Agentic-flow npm package — ReasoningBank, embeddings, health server'),
(4, '@ruvector/core', '~/node_modules/@ruvector/core/', 'Rust NAPI HNSW index'),
(5, 'custom-src', '~/claude-flow-self-implemented/src/', 'Our custom DDD TypeScript implementation'),
(6, 'claude-config', '~/.claude/', 'Claude Code project config — helpers, agents, skills, commands, settings'),
(7, '@claude-flow/guidance', '~/node_modules/@claude-flow/guidance/', 'Guidance package — WASM kernel, constitutional AI');

-- ----------------------------------------------------------------------------
-- Package Dependencies
-- ----------------------------------------------------------------------------
INSERT INTO package_dependencies (source_package_id, target_package_id, relationship, notes) VALUES
(1, 2, 'optional', NULL),
(1, 3, 'optional', NULL),
(1, 4, 'optional', NULL),
(1, 4, 'bundles-copy', 'dist/src/ruvector/ contains bundled code, NOT the npm package'),
(1, 7, 'npm-dependency', NULL),
(5, 2, 'bridges-to', NULL),
(5, 4, 'bridges-to', NULL),
(6, 1, 'patches', NULL);

-- ----------------------------------------------------------------------------
-- Domains
-- ----------------------------------------------------------------------------
INSERT INTO domains (id, name, description, priority, synthesis_path) VALUES
(1, 'swarm-coordination', 'Multi-agent swarm lifecycle, topology, health monitoring', 'HIGH', 'domains/swarm-coordination/analysis.md'),
(2, 'model-routing', 'Complexity scoring, tier selection, routing hooks', 'HIGH', 'domains/model-routing/analysis.md'),
(3, 'memory-and-learning', 'AgentDB, ReasoningBank, vector search, pattern storage', 'HIGH', 'domains/memory-and-learning/analysis.md'),
(4, 'agent-lifecycle', 'Agent spawning, templates, state management', 'HIGH', 'domains/agent-lifecycle/analysis.md'),
(5, 'hook-pipeline', 'Hook execution, validators, formatters, event flow', 'HIGH', 'domains/hook-pipeline/analysis.md'),
(6, 'process-spawning', 'Worker processes, IPC, headless execution', 'HIGH', 'domains/process-spawning/analysis.md'),
(7, 'agentdb-integration', 'DDD services, episodes, skills, vector search', 'MEDIUM', 'domains/agentdb-integration/analysis.md'),
(8, 'agentic-flow', 'ReasoningBank, embeddings, flash attention', 'MEDIUM', 'domains/agentic-flow/analysis.md'),
(9, 'ruvector', 'HNSW index, vector operations, Rust NAPI bridge', 'MEDIUM', 'domains/ruvector/analysis.md'),
(10, 'init-and-codegen', 'Project init, templates, agent scaffolding', 'MEDIUM', 'domains/init-and-codegen/analysis.md'),
(11, 'claude-flow-cli', 'CLI commands, argument parsing, output formatting', 'MEDIUM', 'domains/claude-flow-cli/analysis.md'),
(12, 'plugin-system', 'Extension points, plugin loading, hooks', 'LOW', 'domains/plugin-system/analysis.md'),
(13, 'transfer-system', 'Import/export, migration, data portability', 'LOW', 'domains/transfer-system/analysis.md'),
(14, 'production-infra', 'Monitoring, error tracking, performance optimization', 'LOW', 'domains/production-infra/analysis.md');

-- ----------------------------------------------------------------------------
-- Sessions
-- ----------------------------------------------------------------------------
INSERT INTO sessions (id, name, date, focus, agent_count, notes) VALUES
(1, 'R1', '2026-02-13', 'Initial swarm functionality analysis', 5, NULL),
(2, 'R2', '2026-02-13', 'Worker system, headless execution, communication', 4, NULL),
(3, 'R3', '2026-02-14', 'Deep dive + gap discovery', 5, NULL),
(4, 'R3-verify', '2026-02-14', 'Ledger verification and line count corrections', 4, NULL);

-- ============================================================================
-- END OF SCHEMA
-- ============================================================================
