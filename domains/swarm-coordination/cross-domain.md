# Swarm Coordination — Section 6: Cross-Domain Dependencies

> Part of [Swarm Coordination Domain Analysis](analysis.md)

## 6. Cross-Domain Dependencies

- **memory-and-learning domain**: Agent templates reference neural coordination, learning systems, ReasoningBank
- **agentdb-integration domain**: Simulations use AgentDB persistence, voting systems
- **agentic-flow domain**: P2P swarm, federation, QUICClient, SyncCoordinator, attention-fallbacks all live there
- **claude-flow-cli domain**: Shell coordination (swarm-comms.sh, swarm-monitor.sh), agent templates, SKILL files
- **ruvllm domain**: claude_integration.rs workflow execution
