# Finding Category Mapping Rules

> **Reference**: ADR-v4-009 Section 1.2
> **Source data**: `categories-full.json` (1,211 entries, 11,121 rows)

## Strategy

The migration uses a 3-tier approach:

1. **Case-normalized exact match** for the top ~50 buckets (covering ~90% of rows)
2. **Keyword scan** (priority-ordered) for the remaining ~960 tiny categories
3. **Default fallback** to `QUALITY` for anything unmatched

Original value always preserved in `_original_category` column.

---

## Category Definitions (for disambiguation)

Before mapping, clear definitions prevent ambiguity:

| Category | Use When | Do NOT Use When |
|----------|----------|-----------------|
| ARCHITECTURE | System design: module structure, API design, config, data models, protocols, type systems | Function-level code quality (→ QUALITY) |
| QUALITY | Code quality: readability, naming, patterns, implementation details, helpers, features, APIs | System-level design (→ ARCHITECTURE) |
| INTEGRATION | Cross-module connections: dependencies, disconnects, orphans, data flow, API mismatches | Internal module quality (→ QUALITY) |
| PERFORMANCE | Speed, memory, scalability, concurrency, latency, optimization | Algorithmic complexity analysis (→ ALGORITHM) |
| ALGORITHM | Algorithmic correctness, math, numerical stability, complexity analysis, logic errors | Code-level bugs without math (→ BUG) |
| FACADE | Code that *pretends* to work: stubs, placeholders, fabrications, theatrical code, deception | Honestly incomplete code (→ INCOMPLETE) |
| SECURITY | Vulnerabilities, crypto, auth, injection, secrets, permissions | General code bugs (→ BUG) |
| BUG | Bugs that *should* work: correctness errors, error handling, crashes, faults | Intentional stubs (→ FACADE) |
| GENUINE | Positive findings: real working code, innovation, novel implementations, production-ready | Just "not broken" (→ QUALITY) |
| TESTING | Test coverage, test quality, validation, benchmarks, comparison methodology | Test bugs (→ BUG) |
| DOCUMENTATION | Docs quality, completeness, observability, metrics, reporting, roadmap items | Missing features (→ INCOMPLETE) |
| INCOMPLETE | Honestly unfinished: gaps, dead code, missing features, limitations, type safety holes | Deceptive stubs (→ FACADE) |

---

## Tier 1: Case-Normalized Exact Match

Normalize to uppercase, replace non-alpha with `_`, strip leading/trailing `_`.
Then match against these buckets.

### → ARCHITECTURE (~2,800 rows)

```
ARCHITECTURE, DESIGN, API_DESIGN, API_SURFACE, INTERFACE_DESIGN,
INTERFACE, DATA_MODEL, CONFIGURATION, CONFIG, INFRASTRUCTURE,
PROTOCOL, PROTOCOLS, DATA_STRUCTURES, DATA_STRUCTURE, TYPES,
DESIGN_PATTERN, ORCHESTRATION, CONCURRENCY_DESIGN
```

### → QUALITY (~2,400 rows)

```
QUALITY, CODE_QUALITY, IMPLEMENTATION, IMPLEMENTATION_QUALITY,
NAMING, SIMPLIFICATION, BEST_PRACTICE, ANTI_PATTERN, ASSESSMENT,
HELPER, FEATURE, PORTABILITY, FUNCTIONALITY, API, PATTERN
```

### → INTEGRATION (~750 rows)

```
INTEGRATION, DEPENDENCY, DEPENDENCIES, MISSING_INTEGRATION,
DISCONNECTED, CROSS_REFERENCE, API_MISMATCH, MISSING_FEATURE,
BROKEN_DEP, BROKEN, ORPHANED, DATA_FLOW
```

Note: DATA_FLOW mapped to INTEGRATION (most DATA_FLOW findings in this project describe
disconnects between modules, not structural data flow design). Removed from ARCHITECTURE
Tier 1 list per gap analysis recommendation.

### → PERFORMANCE (~530 rows)

```
PERFORMANCE, OPTIMIZATION, CONCURRENCY, MEMORY, SCALABILITY
```

### → ALGORITHM (~430 rows)

```
ALGORITHM, ALGORITHM_QUALITY, ALGORITHM_CORRECTNESS, ALGORITHM_REAL,
NUMERICAL_STABILITY, LOGIC, LEGITIMATE_ALGORITHM
```

### → FACADE (~590 rows)

```
FACADE, STUB, PLACEHOLDER, FABRICATION, FABRICATED, DECEPTION,
MISLEADING, THEATRICAL, PHANTOM_WRAPPER, FACADE_PATTERN,
FACADE_DECEPTION, QUANTUM_FACADE, ARCHITECTURE_FACADE,
FALSE_COMPLEXITY, FABRICATED_DATA, FABRICATED_METRICS,
DATA_GENERATION, DEMONSTRATION
```

### → SECURITY (~171 rows)

```
SECURITY
```

### → BUG (~370 rows)

```
BUG, CORRECTNESS, ERROR_HANDLING, ACCURACY, FALLBACK, ROBUSTNESS
```

### → GENUINE (~360 rows)

```
GENUINE, GENUINE_IMPLEMENTATION, GENUINE_SIGNAL, GENUINE_INFRASTRUCTURE,
GENUINE_ALGORITHM, GENUINE_ALGORITHMS, REAL, REAL_CODE,
REAL_IMPLEMENTATION, REAL_WASM, INNOVATION, REALNESS,
POSITIVE, CORRECT_IMPLEMENTATION
```

### → TESTING (~190 rows)

```
TESTING, TEST_COVERAGE, TEST_QUALITY, COMPARISON,
BENCHMARK_QUALITY, ANALYSIS_QUALITY, VALIDATION
```

### → DOCUMENTATION (~310 rows)

```
DOCUMENTATION, COMPLETENESS, OBSERVABILITY, METRICS, REPORTING, ROADMAP
```

### → INCOMPLETE (~320 rows)

```
INCOMPLETE, INCOMPLETE_IMPLEMENTATION, INCOMPLETE_IMPL,
MISSING_IMPLEMENTATION, GAP, LIMITATION, DEAD_CODE, CONFIG_GAP,
TYPE_SAFETY, RELIABILITY
```

---

## Tier 2: Keyword Scan (for unmatched categories)

Applied to the ~960 categories not matched by Tier 1.
Keywords checked in **priority order** (first match wins):

| Priority | Keywords (case-insensitive) | Maps To | Rationale |
|----------|---------------------------|---------|-----------|
| 1 | `facade`, `stub`, `placeholder`, `fabricat`, `decep`, `mislead`, `theatrical`, `phantom`, `false` | FACADE | Deception detection is highest priority |
| 2 | `genuine`, `real`, `innovat`, `positive`, `correct.impl`, `legitimate` | GENUINE | Real code detection |
| 3 | `secur`, `vulnerab`, `crypto`, `auth`, `inject`, `secret` | SECURITY | Security findings |
| 4 | `integrat`, `depend`, `disconnect`, `orphan`, `cross.ref`, `mismatch` | INTEGRATION | Cross-module |
| 5 | `algorithm`, `numeric`, `math`, `eigenval`, `convergence` | ALGORITHM | Algorithmic |
| 6 | `test`, `benchmark`, `coverage`, `validat` | TESTING | Testing |
| 7 | `perform`, `optim`, `latency`, `throughput`, `cache`, `concurr` | PERFORMANCE | Performance |
| 8 | `doc`, `comment`, `readme`, `completeness`, `observab`, `metric` | DOCUMENTATION | Documentation |
| 9 | `incomplet`, `gap`, `dead`, `limit`, `missing`, `unfinish` | INCOMPLETE | Incomplete |
| 10 | `bug`, `error`, `broken`, `fault`, `crash`, `incorrect`, `wrong` | BUG | Bugs |
| 11 | `architect`, `design`, `pattern`, `config`, `struct`, `schema`, `protocol` | ARCHITECTURE | Architecture |
| 12 | (default) | QUALITY | Catch-all |

---

## Tier 3: Default Fallback

Any category not matched by Tier 1 or Tier 2 → `QUALITY`.

Rationale: The 900+ tiny categories are mostly fine-grained quality observations.
QUALITY is the safest catch-all — it's the largest bucket and least actionable
for v4 decision-making. The `_original_category` preserves full specificity.

---

## Special Cases

### "POSITIVE" severity vs "positive" category

- `severity = 'POSITIVE'` (2 rows) → remap severity to INFO
- `category = 'positive'` (34 rows) → remap category to GENUINE

These are different columns with the same string. The migration handles them separately.

### Compound categories

Some categories are compound: `REAL_IMPLEMENTATION`, `GENUINE_ALGORITHM`, etc.
These get matched by Tier 1 exact match (most are in the top 50 buckets).
For compounds not in Tier 1, the keyword scan picks up the most significant word.

### Category + Severity interaction

The migration does NOT change which findings are CRITICAL/HIGH/etc.
A FACADE finding that was CRITICAL before stays CRITICAL after.
Only the category label changes, not the severity.

---

## Expected Distribution (After Migration)

| Category | Estimated Count | % of 11,121 |
|----------|----------------|-------------|
| ARCHITECTURE | ~2,800 | 25% |
| QUALITY | ~2,400 | 22% |
| INTEGRATION | ~750 | 7% |
| FACADE | ~590 | 5% |
| PERFORMANCE | ~530 | 5% |
| ALGORITHM | ~430 | 4% |
| BUG | ~370 | 3% |
| GENUINE | ~360 | 3% |
| INCOMPLETE | ~320 | 3% |
| DOCUMENTATION | ~310 | 3% |
| TESTING | ~190 | 2% |
| SECURITY | ~171 | 2% |
| *unaccounted (from keyword scan)* | ~900 | 8% |

The "unaccounted" rows will distribute across all 12 categories via keyword scan.
Total must equal 11,121.

## Verification

After migration, run:
```sql
SELECT category, COUNT(*) as cnt,
  SUM(CASE WHEN severity = 'CRITICAL' THEN 1 ELSE 0 END) as critical,
  SUM(CASE WHEN severity = 'HIGH' THEN 1 ELSE 0 END) as high
FROM findings
GROUP BY category
ORDER BY cnt DESC;
```

Must return exactly 12 rows. No row should have 0 count.
Sum of all cnt must equal 11,121.
Sum of all critical must equal 1,294.
Sum of all high must equal 2,923.
