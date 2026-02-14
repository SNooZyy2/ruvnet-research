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

-- ============================================================================
-- ANALYTICAL VIEWS
-- ============================================================================

-- ----------------------------------------------------------------------------
-- View: domain_coverage
-- Coverage stats per domain
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
LEFT JOIN files f ON fd.file_id = f.id
GROUP BY d.id, d.name, d.priority;

-- ----------------------------------------------------------------------------
-- View: package_coverage
-- Coverage stats per package
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
LEFT JOIN files f ON p.id = f.package_id
GROUP BY p.id, p.name, p.total_files, p.total_loc;

-- ----------------------------------------------------------------------------
-- View: integration_hotspots
-- Files in 3+ domains (cross-cutting concerns)
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
GROUP BY f.id, p.name, f.relative_path, f.depth, f.loc
HAVING COUNT(fd.domain_id) >= 3
ORDER BY domain_count DESC, f.loc DESC;

-- ----------------------------------------------------------------------------
-- View: priority_gaps
-- NOT_TOUCHED files in HIGH priority domains
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
ORDER BY f.loc DESC;

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
-- CRITICAL/HIGH findings where followed_up=0
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
ORDER BY
    CASE f.severity
        WHEN 'CRITICAL' THEN 1
        WHEN 'HIGH' THEN 2
        ELSE 3
    END,
    f.id;

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
