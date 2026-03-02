# Relationship Type Mapping Rules

> **Reference**: ADR-v4-009 Section 1.1
> **Source data**: `relationships-full.json` (511 entries, 1,704 rows)

## Strategy

The migration uses a 3-tier approach:

1. **Exact match** (case-insensitive) for known short-form values
2. **Keyword scan** (priority-ordered) for sentence-form values
3. **Default fallback** to `USES` for anything unmatched

Original value always preserved in `_original_relationship` column.
Sentence-form values also appended to `evidence` column.

---

## Tier 1: Exact Match Rules

These cover the non-singleton relationships (113 types, 1,306 rows = 77% of data).

### → IMPORTS (target: ~600 rows)

```
imports, IMPORTS, import, imports_from, imports_type, imports_types, imports_class,
direct-import, type-imports, type_import, imports_trait, imports_config,
imports_error_types, imports_error_type, imports_persistence, imports_solver,
imports_simd_ops, imports_wasm_bridge, imports_wasm_loader, imports_csr_storage,
imports_gnn_service, imports_precision_type, imports_spawns, imports_broken,
lazy_import, optional_import, mod_use_import, pub_mod_import, transitive_import,
dynamic-import, test_imports, imports-types, imports-and-calls, imports-facade,
imports-unused, imports-extensively, imports-for-initialization, imports-uses,
imports-domain-model, import-Pattern-StorageStats, imported-by, imported-but-unused,
imported_by, IMPORTED_BY, IMPORTS_AND_REEXPORTS, IMPORTS_TYPE, MODULE_IMPORT,
imports (type), imports_all, imports_config
```

### → USES (target: ~250 rows)

```
uses, USES, used_by, USED_BY, used-by, used_by_conversion, calls, calls_method,
CALLS_FROM, implements, IMPLEMENTS, IMPLEMENTS_FOR, implements-api, implements-for,
implements-interface, implements-trait, extends, EXTENDS, extends_learning_from,
instantiates, instantiated_by, instantiation_bridge, requires,
depends_on, DEPENDS_ON, depends-on, depends, depends-on-fallback, depends_via_bootstrap,
DEPENDS_ON_TYPES, consumed_by, CONSUMED_BY, consumes, uses_types, uses_type,
uses_struct, uses_api, uses_constant, uses_wrapper, uses_prompt_templates,
uses-types, uses-impl, uses-concept, uses-for-embeddings, type_consumer,
type_alignment, type_provider, type-definitions, type_definition,
internal_usage, crate_dependency, cross_crate_dependency, cross_crate_counterpart,
cross-lang-api, cross-lang-impl, native_dependency, npm-module, npm_dependency,
data_dependency, initialization_dependency, worker_thread_dependency,
error type dependency, error_handling, defines-backends-for, defines-interface-for,
defines_types_for, complementary-types, COMPLEMENTARY, COMPLEMENTS, complements,
ALGORITHM_COMPLEMENT, BASELINE_FOR, PROVIDES_NEURAL_API, INTEGRATES, integrates,
feature-gated-uses, optional-dep, optional-dependency, optional-integration,
optional_integration, cli_indirect, indirect, indirect_via_server,
client-server, ffi-interface, architectural, wasm_binding, wasm_variant,
USES_EXTERNAL_CRATE, USES_LOADER, USES_TYPE, SHARED_TYPE, REPORTS_FOR,
OPTIMIZES, RECEIVES_DATA_FROM, cross_repo_comparable, related_to,
variant_of, equivalent_algorithm, solver_family, systemic_pattern
```

### → EXPORTS (target: ~120 rows)

```
exports, EXPORTS, re-exports, re-export, BARREL_EXPORT, pub_use,
pub_mod_reexport, module-reexport, INDIRECT_EXPORT, CONDITIONAL_EXPORT,
exported_by, exported-by, EXPORTED_FROM, EXPORTS_AS_NAMESPACE, EXPORTS_TO,
exports_from, exports_to, exports-to, reexported_by, pub_re_export,
pub_reexport_indirect, RE_EXPORTS_MODULE, RE_EXPORTS, RE_EXPORTS_ALL,
RE_EXPORTS_TYPE, module-export, module_export, MODULE_EXPORT, module:exported,
exposed_by_sql_wrapper
```

### → DECLARES (target: ~90 rows)

```
pub_module_declaration, pub_mod, pub mod, declares_module, declares,
mod-declaration, declares-submodule, module-declares, module-definition,
module-parent, module_declaration, defines_module, cfg_module, submodule,
always-on submodule, feature-gated submodule, DEFINES, declared-in,
declared_in, member-of, member_of, part-of, mod-wasm, module-member,
CONTAINS, contains
```

### → SIBLINGS (target: ~100 rows)

```
module-sibling, sibling, sibling-module, SIBLING_MODULE, SIBLING,
SIBLING_SCENARIO, co-module, peer, PEER, sibling-kernel, sibling_impl,
sibling_module, sibling-cli-mcp, cohomology-sibling, same-package,
co-located, related, RELATED_IMPLEMENTATION
```

### → COMPETES (target: ~30 rows)

```
parallel-implementation, parallel_implementation, PARALLEL_IMPLEMENTATION,
parallel-impl, parallel_impl, parallel-system, parallel_system, parallel_to,
parallel, parallel_api, parallel_simulation_pattern, parallel implementation,
reimplements, REIMPLEMENTS, alternative-implementation, alternative_to,
ALTERNATIVE_LOADER, mirrors, mirrors_architecture, signature_mirror,
duplicate_of, duplicates_logic, COPY_OF_REFACTORED, algorithmic-parallel,
conceptual-parallel, NAME_COLLISION, naming-conflict,
ARCHITECTURAL_CONTRAST, CONTRASTS_WITH, contrasts_with, compared-to, comparison,
competes_with
```

### → WRAPS (target: ~20 rows)

```
wraps, WRAPS, wraps_native, wraps_deprecated, wrapped_by,
delegates_to, delegates-to, delegates, delegates-storage, delegates_to_via_npx,
facade-of, FACADE_OF
```

### → FEEDS (target: ~30 rows)

```
feeds, GENERATES, produces, produces_for, reads, reads-writes, queries,
configures, registers_tools_from, orchestrates, ORCHESTRATED_BY,
spawns-process, spawns_mcp_subprocess, loads, loads-config-from,
loads_wasm_module, loads compiled WASM, LOADS_WASM_INTERFACE,
embeds_schema, database-schema, creates-via, compiles, integration-target,
transpiled-from, Configuration + data structures, Initialization + read/write cycles
```

### → TESTS (target: ~20 rows)

```
tests, TESTS, TESTED_BY, tested_by, tested-by integration tests,
TESTS_VARIANT_OF, TEST_SUITE, tests_via_spawn, tests_performance,
benchmarks, related-benchmark, VALIDATES_AGAINST, internal-validation,
test-dependency
```

### → BROKEN (target: ~20 rows)

```
bypasses, MISSING_INTEGRATION, missing_integration, SHOULD_INTEGRATE,
should_import, should-import-but-reimplements, should_call_but_doesnt,
should_integrate_but_orphaned, expects_integration, potential-integration,
ISOLATED_FROM, CONTRADICTS, VIOLATES, HONEST_ALTERNATIVE_TO,
declares_dependency_but_never_uses, calls_broken, imports_broken,
supports_but_facades, orphaned-module, orphaned_from, incomplete_port,
js_fallback_for, replaced-by, replaces, supersedes, patched_version_of,
improved_version_of, missing_config_for, architectural-mismatch
```

---

## Tier 2: Keyword Scan (for sentence-form strings)

Applied to the ~398 singleton strings that didn't match Tier 1.
Keywords checked in **priority order** (first match wins):

| Priority | Keywords (case-insensitive) | Maps To | Rationale |
|----------|---------------------------|---------|-----------|
| 1 | `should`, `missing`, `broken`, `bypass`, `violat`, `orphan`, `isolated`, `never` | BROKEN | Strongest architectural signal |
| 2 | `parallel`, `alternative`, `reimpl`, `mirror`, `duplicate`, `compet` | COMPETES | Parallel implementation pattern |
| 3 | `wraps`, `delegat`, `facade` | WRAPS | Wrapper/delegation pattern |
| 4 | `produces`, `feeds`, `loads`, `consumes`, `spawns`, `generates`, `orchestrat`, `drains` | FEEDS | Data flow pattern |
| 5 | `test`, `benchmark`, `validat` | TESTS | Testing relationship |
| 6 | `export`, `re-export`, `pub use` | EXPORTS | Export/re-export pattern |
| 7 | `pub mod`, `declares`, `module` | DECLARES | Module declaration |
| 8 | `import` | IMPORTS | Import pattern |
| 9 | (default) | USES | Catch-all |

---

## Tier 3: Default Fallback

Any string not matched by Tier 1 or Tier 2 → `USES`.

Rationale: Most unmatched strings describe specific usage patterns
("uses EmbeddingCache for 3-tier caching...") which are semantically USES relationships.

---

## Expected Distribution (After Migration)

| Canonical Type | Estimated Count | % of 1,704 (pre-merge; post-merge total = 1,682) |
|---------------|----------------|------------|
| IMPORTS | ~600 | 35% |
| USES | ~350 | 21% |
| EXPORTS | ~120 | 7% |
| SIBLINGS | ~100 | 6% |
| DECLARES | ~90 | 5% |
| FEEDS | ~150 | 9% |
| COMPETES | ~100 | 6% |
| WRAPS | ~50 | 3% |
| TESTS | ~50 | 3% |
| BROKEN | ~90 | 5% |

These are estimates. The migration script will print actual counts for verification.

## Verification

After migration, run:
```sql
SELECT relationship, COUNT(*) as cnt
FROM dependencies
GROUP BY relationship
ORDER BY cnt DESC;
```
Must return exactly 10 rows. No row should have 0 count.
Sum must equal 1,682 (1,704 minus 22 merged duplicate pairs).

### Pre-Normalization Merge

Before normalization, 22 file pairs have multiple relationships that map to the same
canonical type (e.g., `imports` + `IMPORTS` for the same source→target). These must be
merged first to avoid `UNIQUE(source_file_id, target_file_id, relationship)` violations.

Merge rule: keep the row with longer `evidence`, append `" [merged: <loser_rel>]"` to
the survivor's evidence, DELETE the loser. See ADR section 2.4 for details.
