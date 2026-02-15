# Init and Codegen Domain Analysis

**Session**: R17 (2026-02-14)
**Files Read**: 9 files, 7,180 LOC (100% complete read)
**Depth**: DEEP on all files
**Findings**: 19 (1 CRITICAL, 5 HIGH, 10 MEDIUM, 3 INFO)

## Overview

The init-and-codegen system handles project scaffolding for claude-flow. It generates configuration files, helper scripts, agent templates, and skill definitions from a multi-strategy template system. The architecture is a **string-based code generator** — all generators return code as template strings, with no AST or syntax validation.

## Architecture

```
CLI Entry Point
  dist/src/commands/init.js (964 LOC)
    ├── wizard mode (interactive prompts)
    ├── upgrade mode (safe asset replacement)
    └── codex mode (dual init via @claude-flow/codex)
         │
         ▼
Orchestration Engine
  dist/src/init/executor.js (1,763 LOC)
    ├── executeInit() → creates project structure
    ├── executeUpgrade() → safe upgrade preserving customizations
    ├── mergeSettingsForUpgrade() → 11-step settings merge
    └── findSourceHelpersDir() → 4-strategy source resolution
         │
    Calls all generators:
    ├── claudemd-generator.js (486 LOC) → CLAUDE.md templates (6 variants)
    ├── settings-generator.js (284 LOC) → .claude/settings.json + hooks
    ├── mcp-generator.js (100 LOC) → .mcp.json for MCP server config
    ├── statusline-generator.js (1,311 LOC) → statusline.cjs (10+ data sources)
    ├── helpers-generator.js (999 LOC) → git hooks, session, routing, intelligence
    └── types.js (258 LOC) → platform detection + option presets
```

## Key Files

| File | LOC | Role | Quality |
|------|-----|------|---------|
| `dist/src/init/executor.js` | 1,763 | Orchestration engine | GOOD - strategy pattern, builder pattern |
| `dist/src/init/statusline-generator.js` | 1,311 | Status display generator | FAIR - reads 4+ files per refresh |
| `dist/src/init/helpers-generator.js` | 999 | Helper script generator | FAIR - string array construction fragile |
| `dist/src/commands/init.js` | 964 | CLI entry point | GOOD - wizard, upgrade, codex modes |
| `dist/src/init/claudemd-generator.js` | 486 | CLAUDE.md template system | GOOD - 6 template variants, pure functions |
| `dist/src/init/settings-generator.js` | 284 | Settings generator | FAIR - Agent Teams experimental enabled by default |
| `dist/src/init/types.js` | 258 | Type definitions + presets | GOOD - 3 option presets (default/minimal/full) |
| `dist/src/init/mcp-generator.js` | 100 | MCP config generator | OK - Windows injection risk |
| `dist/src/init/index.js` | 15 | Barrel export | Trivial |

## Critical Findings

### CRITICAL: Intelligence Stub Degradation
**File**: `helpers-generator.js:538-739`
`generateIntelligenceStub()` lacks PageRank, Jaccard similarity, and graph algorithms that the full version provides. When source copy fails, users get a significantly degraded intelligence layer with no indication of the capability gap.

### HIGH: Silent Source Resolution Failures
**File**: `executor.js:854-899`
`findSourceHelpersDir()` tries 4 resolution strategies (require.resolve, __dirname, cwd, package root) but has NO logging when fallback occurs. Silent failures cause incomplete installs with no diagnostic trail.

### HIGH: Windows Compatibility Bugs
**Files**: `executor.js:222-242`, `init.js:271-305`
PowerShell null redirect syntax (`2>$null`) is incorrect for CMD. Unix-specific `2>/dev/null` used in several execSync calls. Will fail silently on Windows.

### HIGH: Statusline Performance
**File**: `statusline-generator.js:160-250`
`getLearningStats()` reads 4+ JSON files sequentially on EVERY statusline refresh (default 5s interval). Should cache or debounce to avoid filesystem hammering.

## Architecture Patterns

1. **String template generation**: All generators return code as strings (fragile, no syntax validation)
2. **Multi-strategy source resolution**: 4-tier fallback for finding package source (complex, hard to debug)
3. **Cross-platform abstraction**: Platform-specific wrappers (incomplete on Windows)
4. **Graceful degradation**: Stubs/fallbacks when full implementations unavailable
5. **Composition templates**: CLAUDE.md templates composed from arrays of section functions
6. **Builder pattern**: InitResult tracks created/skipped/error counts

## Cross-Domain Dependencies

- **executor.js** → all `*-generator.js` files (orchestration)
- **init.js** → `executor.js` (CLI → engine)
- **settings-generator.js** → references hook-handler.cjs paths
- **statusline-generator.js** → reads 10+ runtime JSON files at display time
- **init.js** → `@claude-flow/codex` (optional dynamic import)

## Knowledge Gaps

- 48 files still unread in this domain (mostly agent templates and SKILL.md files)
- Template duplicate analysis needed: `.claude/agents/templates/` vs `agents/templates/`
- `@claude-flow/codex` package not yet analyzed

## R37: Rust Training Data Generation for Init/Codegen (Session 37)

### Overview

R37 deep-read of ruvllm training crate files reveals Rust-side code generation training data that directly relates to the init-and-codegen domain. These files generate synthetic training examples for code-editing, tool-calling, and task-routing agents.

### Relevant Files (4 files, 5,639 LOC)

| File | LOC | Real% | Relevance to Init/Codegen |
|------|-----|-------|-------------|
| **tool_dataset.rs** | 2,147 | **88-92%** | MCP tool-call training dataset: 140+ tool definitions across 19 categories (file_ops, code_analysis, testing, deployment, monitoring, etc.) with quality scoring. Generates synthetic tool-call examples for training agents to use MCP tools correctly. |
| **claude_dataset.rs** | 1,209 | **75-80%** | Claude task templates: 5 categories (coding, debugging, architecture, testing, documentation) with 60+ templates. Generates training data for code generation tasks. |
| **pretrain_pipeline.rs** | 1,394 | **85-88%** | Multi-phase pretraining pipeline that consumes tool_dataset and claude_dataset outputs. Bootstrap → Synthetic → Reinforce → Consolidate phases generate progressively harder training examples. |
| **store_ffi.rs** | 889 | **90-92%** | WASM/C FFI for temporal-tensor storage: 11 extern "C" functions enabling code generation tools to persist state across WASM boundaries. Real quantization via crate::quantizer. |

### Cross-Domain Insights

1. **Tool-call training data**: tool_dataset.rs generates synthetic examples of correct MCP tool usage — 140+ tools matching the categories used in claude-flow's init system (file operations, code analysis, testing). This is the training data counterpart to the tool schemas generated by init.js.

2. **Code generation templates**: claude_dataset.rs mirrors the agent template structure from init-and-codegen — coding/debugging/architecture/testing/documentation categories match the 5 essential agent types in agent-converter.js.

3. **WASM persistence**: store_ffi.rs provides the FFI boundary for WASM-based code generation tools to persist quantized data. This bridges the Rust temporal-tensor crate with WASM deployment targets used in browser-based code generation.

### Quality Concerns

- **tool_dataset.rs paraphrasing is SIMPLISTIC** — literal word replacement ("create"→"make", etc.), lowercases entire input. Augmented examples may not represent realistic tool-call variation.
- **claude_dataset.rs augmentation weak** — Only 5 word pairs for paraphrasing. Complexity variation changes metadata (difficulty level) but NOT the actual task content.
- **pretrain_pipeline.rs hash-based embeddings** — All training routing depends on character sum % dim embeddings, making curriculum learning based on non-semantic similarity.
