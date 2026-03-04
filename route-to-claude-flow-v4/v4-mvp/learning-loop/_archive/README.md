# v4 MVP: The Learning Loop

> Fix the 8 breaks that prevent claude-flow from learning from its own experience.

## Documents

| Doc | Lines | What | Read When |
|-----|-------|------|-----------|
| [SPEC.md](SPEC.md) | 226 | What + why. Problem, 9 fixes (summary), phases, risks, expansion | Planning the build |
| [FILES.md](FILES.md) | 236 | What to copy. 15 files, npm deps, directory structure, validation | Starting implementation |
| [FIXES.md](FIXES.md) | 397 | Exact code changes. Before/after snippets with line numbers | Implementing each fix |

Zero overlap between files. Each fact lives in exactly one place.
Cross-references link between them (e.g., FILES.md fix column links to FIXES.md sections).

## The Loop

```
pre-task --> AI works --> post-task --> judge --> distill --> consolidate
   ^                                                             |
   +--------------- retrieve learned patterns <------------------+
```

Today this loop is severed at post-task. 8 breaks, 9 fixes, ~466 LOC of changes.

## Quick Reference

| Fix | What | File |
|-----|------|------|
| 1 | Wire post-task to pipeline | handler.ts (new) |
| 2 | Activate real embeddings | config.ts DEFAULT_CONFIG |
| 3 | Distill fallback stores memories | distill.ts |
| 4 | Capture trajectory data | trajectory/capture.ts (new) |
| 5 | Persist verdict to DB | post-task.ts |
| 6 | Judge checks exit codes | judge.ts |
| 7 | Dynamic ModelRouter import | judge.ts + distill.ts |
| 8 | UPSERT preserves metadata | queries.ts |
| 9 | Remove embeddings.js (don't copy) | -- |

## Archived

`_archive/` contains the original 4-file set (BREAK-ANALYSIS.md, FILE-MAP.md, PREFLIGHT.md) that was consolidated into the 3 files above. All content preserved; contradictions resolved.
