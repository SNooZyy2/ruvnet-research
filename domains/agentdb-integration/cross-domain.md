# AgentDB Integration — Section 6: Cross-Domain Dependencies

> Part of [AgentDB Integration Domain Analysis](analysis.md). See index for full document map.

## 6. Cross-Domain Dependencies

```
agentdb-integration → memory-and-learning: 20 USES, 14 IMPORTS, 5 WRAPS, 3 BROKEN
agentdb-integration → agentic-flow: 14 IMPORTS, 6 USES, 2 BROKEN, 1 COMPETES
agentdb-integration → ruvector: 6 USES, 5 IMPORTS, 4 WRAPS, 3 EXPORTS, 2 SIBLINGS, 2 COMPETES, 1 BROKEN
agentdb-integration → v4-gold-sweep: 3 EXPORTS, 3 USES, 2 SIBLINGS, 1 COMPETES
agentdb-integration → v4-priority: 3 IMPORTS
agentdb-integration → production-infra: 2 COMPETES, 1 BROKEN, 1 EXPORTS, 1 SIBLINGS, 1 USES
agentdb-integration → swarm-coordination: 1 IMPORTS
agentdb-integration → transfer-system: 1 USES
```

Note: 8+ total BROKEN edges represent integration debt across 5+ target domains. R136 adds the V3 @claude-flow/memory layer as a new BROKEN integration point (memory-bridge.ts dead, AgentDBAdapter misnamed).
