# Verification Plan: ai-hackerspace-march-01-leads.md

Generated: 2026-03-01
Source: /home/snoozyy/ruvnet-research/ruv-vods/leads/ai-hackerspace-march-01-leads.md (22 leads)

## Summary
- ALREADY_COVERED: 4 leads (skip)
- PARTIALLY_COVERED: 5 leads (targeted re-reads)
- NEW: 7 leads (full reads needed)
- CONTRADICTION: 3 leads (highest priority)
- UNRESOLVABLE: 3 leads (cannot verify)

## Total files to read: ~18-25 (across NEW + PARTIALLY_COVERED + CONTRADICTION)
## Estimated session: 2-3 research sessions (R115-R117 scope)

---

## CONTRADICTION LEADS (verify first)

### LEAD-015: Unified vector-graph-attention pipeline operates independently of LLMs
Classification: CONTRADICTION
Original claim: "most of the learning is in vector space... most of the grounding capabilities and attention mechanisms are all in that interplay between the vector and the graph and attention and none of that's all mathematical"
Contradicts: Multiple findings documenting disconnected subsystems — 4+ HNSW stores never compose (R104), EmbeddingService never initialized (R20), MultiHeadAttentionController disconnected from AgentDB HNSW, CrossAttentionController never invoked in production pipeline, 12+ parallel subsystems, hash-based embeddings SYSTEMIC

#### Files to read:
| File | Package | Current Depth | LOC | Action |
|------|---------|--------------|-----|--------|
| agentic-flow/src/intelligence/RuVectorIntelligence.ts | agentic-flow-rust | DEEP (partial) | 853 | Re-read for composition evidence between vector/graph/attention |
| agentic-flow/src/intelligence/EmbeddingService.ts | agentic-flow-rust | DEEP | 1810 | Check if ONNX path actually composes with HNSW+graph |
| crates/ruvllm/src/ruvector_integration.rs | ruvector-rust | DEEP | varies | Already DEEP — check if capability flags compose |

#### Existing findings that contradict:
- Finding: "4+ HNSW stores never compose" (R104, multiple files)
- Finding: "EmbeddingService never initialized in claude-flow bridge" (R20, multiple files)
- Finding: "MultiHeadAttentionController is self-contained with its own in-memory Map store — does NOT plug into AgentDB HNSW" (packages/agentdb/src/controllers/attention/MultiHeadAttentionController.ts, INFO)
- Finding: "CrossAttentionController NEVER invoked in production MemoryController.retrieveWithAttention() pipeline" (packages/agentdb/src/controllers/attention/CrossAttentionController.ts, CRITICAL)
- Finding: "ruvector_integration.rs — HNSW_AVAILABLE and SONA_AVAILABLE imported but never referenced in any conditional guard" (DEEP, HIGH)
- Finding: "8th independent graph/matrix system — zero integration with ruvector-graph Cypher executor" (psycho-symbolic-reasoner, CRITICAL)
- Finding: "rvf-runtime and rvf-index HNSW worlds entirely separate — never composed" (rvf, HIGH)

#### What to verify:
- Is there ANY code path where vector search, graph traversal, and attention scoring compose into a single pipeline?
- Does the "optimizer" app (not in our repos) possibly implement this composition externally?
- The claim says learning is "independent" of LLMs — but SYSTEMIC hash-based embeddings mean vector space has no semantic meaning without an LLM-derived embedding model

#### Suggested research agent: cross-repo-tracer

---

### LEAD-013: All ruvector packages compile to WASM for cross-platform use
Classification: CONTRADICTION
Original claim: "because those packages I've built all to be WASMs I instantaneously can use them in the browser or across different operating environments without having to recompile them"
Contradicts: R51-R60 found WASM genuine/theatrical ratio of 6:4. Multiple WASM facades documented. Not ALL packages compile to WASM.

#### Files to read:
| File | Package | Current Depth | LOC | Action |
|------|---------|--------------|-----|--------|
| crates/ruvector-attention-unified-wasm/pkg/ruvector_attention_unified_wasm.js | ruvector-rust | NOT_TOUCHED | 2752 | Check if genuine WASM output or stub |
| crates/ruvector-nervous-system-wasm/pkg/ruvector_nervous_system_wasm.js | ruvector-rust | NOT_TOUCHED | 1648 | Check if genuine WASM output |
| examples/edge-full/pkg/rvlite/rvlite.js | ruvector-rust | NOT_TOUCHED | 2367 | Verify rvlite WASM bindings |

#### Existing findings that contradict:
- Finding: "WASM neural_train() call has NO RETURN VALUE processing — results ignored, facade metrics used" (ruv-swarm/npm/src/neural.js, CRITICAL)
- Finding: "QUIC transport loads WASM returning {}, Federation Hub returns empty arrays" (dist/swarm/quic-coordinator.js, CRITICAL)
- Finding: "loadWasmModule returns {}. send writes nothing. receive returns empty" (agentic-flow/src/transport/quic.ts, CRITICAL)
- Finding: "EnhancedEmbeddingService initializeWASM creates mockDb with stub prepare/exec/run" (src/controllers/EnhancedEmbeddingService.ts, MEDIUM)
- Finding: "Only core module (ruv_swarm_wasm_bg.wasm) marked as exists:true in manifest — all other modules are optional stubs" (ruv-swarm/npm/src/wasm-loader.js, MEDIUM)

#### What to verify:
- How many ruvector crates have actual `wasm32-unknown-unknown` targets in their Cargo.toml?
- Distinguish between: (a) crates that compile to WASM, (b) crates with WASM wrappers that work, (c) stub WASM bindings
- "All" is a strong claim — need to check Cargo workspace members for wasm targets

#### Suggested research agent: cross-repo-tracer

---

### LEAD-017: Micro LoRAs used for real-time training of embedding models
Classification: CONTRADICTION
Original claim: "we're using micro LoRAs and a few other approaches to do the real-time training on embedding models language models various forms of action and world models"
Contradicts: Hash-based embeddings are SYSTEMIC across the codebase. micro_lora.rs (92-95%) targets language model weights, not embedding model weights. No evidence of LoRA applied to embedding models.

#### Files to read:
| File | Package | Current Depth | LOC | Action |
|------|---------|--------------|-----|--------|
| crates/ruvllm/src/lora/micro_lora.rs | ruvector-rust | DEEP | 1261 | Re-read: does it target embedding layers or only LM layers? |
| crates/ruvllm/src/bitnet/rlm_embedder.rs | ruvector-rust | DEEP | 1832 | Check if LoRA is applied to embedding weights |
| crates/ruvllm/src/lora/training.rs | ruvector-rust | DEEP | 799 | Check training targets |

#### Existing findings that relate:
- Finding: "CRITICAL — Default embedding provider uses hash-based embeddings: sums character bytes" (crates/ruvector-core/src/embeddings.rs, CRITICAL)
- Finding: "MicroLoRA rank 1-2 for instant adaptation (2,211 ops/sec). BaseLoRA rank 4-16 for background" (crates/sona/src/lora.rs, INFO)
- Finding: "instant_adapt() applies MicroLoRA with instant_lr, queues sample — REAL <1ms adaptation" (crates/ruvllm/src/optimization/sona_llm.rs, HIGH)
- Finding: "COMPLETE LOW-RANK ADAPTATION: Real LoRA W + (A*B) * (alpha/rank)" (examples/edge-net/src/ai/lora.rs, CRITICAL)
- Finding: "EmbeddingService.initialize() tries @xenova/transformers for local all-MiniLM-L6-v2. Falls back to mockEmbedding on failure. Mock uses hash-based sin(seed)*cos(seed*0.5)" (packages/agentdb/src/controllers/EmbeddingService.ts, HIGH)

#### What to verify:
- Does micro_lora.rs have any code path targeting embedding model parameters specifically?
- Are "action models" and "world models" implemented anywhere, or is this aspirational?
- The claim of "real-time training on embedding models" directly contradicts the SYSTEMIC hash-based embedding finding — if true, would be a major correction

#### Suggested research agent: reader (targeted re-read of micro_lora.rs embedding layer targeting)

---

## NEW LEADS (full reads needed)

### LEAD-001: Optimizer desktop app uses ruvector attention/hyperbolics
Classification: NEW
Original claim: "This is using all the ruvector self-learning capabilities... attention the hyperbolics... how can I understand the signals and everything that's happening inside of my computer in real time and then optimize for it"
Reason: The "optimizer" desktop app is NOT in any of our 12 tracked packages. No files match "optimizer" + "desktop" pattern. This appears to be a separate, untracked application.

#### Files to read:
| File | Package | Current Depth | LOC | Action |
|------|---------|--------------|-----|--------|
| npm/packages/ruvbot/src/RuvBot.ts | ruvector-rust | NOT_TOUCHED | 781 | Closest candidate — check if ruvbot IS the optimizer |
| npm/packages/ruvbot/README.md | ruvector-rust | NOT_TOUCHED | 1400 | Check purpose description |

#### Existing findings that relate:
- Attention modules exist at DEEP with genuine quality (crates/ruvllm/src/kernels/attention.rs DEEP, edge-net attention_unified.rs DEEP)
- Hyperbolic HNSW confirmed genuine 88-95% from R92 (crates/ruvector-hyperbolic-hnsw/ DEEP)
- The underlying modules exist; question is whether the optimizer app genuinely imports them

#### What to verify:
- Is the "optimizer" app the same as ruvbot (npm/packages/ruvbot)?
- If separate, is it in a private repo not indexed in our research DB?
- Check ruvbot for imports from ruvector-core attention and hyperbolic modules

#### Suggested research agent: reader

---

### LEAD-002: Ruvector memory system in optimizer startup management
Classification: NEW
Original claim: "ruvector memory system" running as part of the optimizer's startup management
Reason: Same as LEAD-001 — optimizer app not in tracked packages. Cannot verify without locating the optimizer codebase.

#### Files to read:
Same as LEAD-001 (ruvbot investigation)

#### What to verify:
- Is "ruvector memory system" a reference to temporal-tensor (93% quality, DEEP) or to HNSW/vector stores?
- Verify if ruvbot has startup management features

#### Suggested research agent: reader

---

### LEAD-003: WASM plugin system with ed25519 signing
Classification: NEW
Original claim: "what I ended up doing is using a WASM so the WASM self-contained no external components... I'm securing that with ed25519"
Reason: No specific plugin store/delivery mechanism with ed25519 signing found in the tracked codebase. Ed25519 exists in p2p-swarm-v2.js (genuine crypto), in RAC (edge-net), and in ruQu, but not in a plugin signing context.

#### Files to read:
| File | Package | Current Depth | LOC | Action |
|------|---------|--------------|-----|--------|
| npm/packages/ruvbot/docs/adr/ADR-006-wasm-integration.md | ruvector-rust | NOT_TOUCHED | 776 | Check if ruvbot describes WASM plugin delivery |
| examples/edge-net/pkg/contribute-daemon.js | ruvector-rust | NOT_TOUCHED | 740 | Could be plugin distribution daemon |
| crates/rvlite/src/lib.rs | ruvector-rust | DEEP | 889 | Already DEEP — check if it mentions plugin signing |

#### Existing findings that relate:
- Finding: "REAL crypto: Ed25519 signing, X25519 ECDH, AES-256-GCM, replay protection" (dist/swarm/p2p-swarm-v2.js, HIGH)
- Finding: "Ed25519 signatures via ed25519_dalek with constant-time comparison" (crates/ruQu/src/tile.rs, INFO)
- R43 finding: "WASM theatrical (2nd facade)" — but this could be a THIRD, separate WASM deployment

#### What to verify:
- Is there a plugin store/registry that signs WASM bundles with ed25519?
- Is this the same WASM system as the known theatrical ones, or a new genuine one?
- Check if ruvbot has plugin management

#### Suggested research agent: facade-detector

---

### LEAD-005: Markov chain-based application launch prediction
Classification: NEW
Original claim: "training a model directly on something a Markov chain-based application launch prediction"
Reason: Only one Markov chain file found (examples/vibecast-7sense/crates/sevensense-analysis/src/infrastructure/markov.rs, NOT_TOUCHED, 525 LOC). Not in optimizer context.

#### Files to read:
| File | Package | Current Depth | LOC | Action |
|------|---------|--------------|-----|--------|
| examples/vibecast-7sense/crates/sevensense-analysis/src/infrastructure/markov.rs | ruvector-rust | NOT_TOUCHED | 525 | Check if this is app-launch related or generic |

#### Existing findings that relate:
- No findings about Markov chains in the research DB

#### What to verify:
- Is the Markov chain implementation generic enough to be used for app launch prediction?
- Could this be in the untracked optimizer codebase?

#### Suggested research agent: reader

---

### LEAD-006: Self-training metrics (84% success rate, 12% improvement) — hardcoded or real?
Classification: NEW
Original claim: "self-training metrics success rate 84... improvement over baseline 12"
Reason: Need to find where these specific numbers appear. Known facade pattern: swarm_monitor() metrics are ALL Math.random(). Training metrics could be similarly fabricated.

#### Files to read:
| File | Package | Current Depth | LOC | Action |
|------|---------|--------------|-----|--------|
| npm/packages/ruvbot/src/RuvBot.ts | ruvector-rust | NOT_TOUCHED | 781 | Check for hardcoded metrics in ruvbot |

#### Existing findings that relate:
- Finding: "swarm_monitor() metrics are ALL Math.random(): health_score (0.7-1.0), cpu_usage (20-80%)" (ruv-swarm/npm/src/mcp-tools-enhanced.js, HIGH)
- Finding: "Validation is a partial facade: run_validation() averages expected quality labels from test data. Comment admits 'In a real scenario, you would evaluate the model output'" (crates/sona/src/training/pipeline.rs, HIGH)

#### What to verify:
- Are 84% and 12% hardcoded constants or computed from actual training runs?
- If in the optimizer app, cannot verify without that codebase

#### Suggested research agent: facade-detector

---

### LEAD-010: Edge-net browser swarm shared compute
Classification: NEW
Original claim: "edge net... shares the capacity so it creates a global shared capacity... swarms of people's browsers that essentially share their compute capacity globally"
Reason: edge-net exists at DEEP for some files (RAC, P2P, SIMD, LoRA, attention) but browser-based distributed compute sharing specifically has not been investigated.

#### Files to read:
| File | Package | Current Depth | LOC | Action |
|------|---------|--------------|-----|--------|
| examples/edge-net/pkg/join.js | ruvector-rust | NOT_TOUCHED | 1334 | "join" suggests peer joining — check for compute sharing |
| examples/edge-net/pkg/contribute-daemon.js | ruvector-rust | NOT_TOUCHED | 740 | "contribute" suggests resource sharing |
| examples/edge-net/pkg/real-agents.js | ruvector-rust | NOT_TOUCHED | 1289 | Check for browser agent coordination |
| examples/edge-net/pkg/monitor.js | ruvector-rust | NOT_TOUCHED | 676 | Check for shared compute monitoring |

#### Existing findings that relate:
- Finding: "R42 REVERSAL: edge-net HAS production P2P transport via libp2p" (examples/edge-net/src/network/p2p.rs, CRITICAL)
- Finding: "COMPLETE INDEPENDENT SIMD IMPLEMENTATION: edge-net implements its own SIMD compute layer (1417 LOC)" (examples/edge-net/src/compute/simd.rs, CRITICAL)
- Finding: "P2P SERIALIZATION: Bincode-based adapter export/import. Full adapter sharing across edge network" (examples/edge-net/src/ai/lora.rs, HIGH)

#### What to verify:
- Does edge-net implement WebRTC/WebSocket-based compute pooling for browsers?
- Is "contribute-daemon.js" a compute contribution daemon that shares browser resources?
- Distinguish between: P2P communication (confirmed real) vs actual compute sharing/distribution

#### Suggested research agent: reader

---

## PARTIALLY_COVERED LEADS (targeted re-reads)

### LEAD-008: AgentDB = simplified ruvector (creator statement)
Classification: PARTIALLY_COVERED
Original claim: "AgentDB... is the same basic stuff [as ruvector] I just made it a little easier to understand... it's a DB for agents... I don't upgrade it as often so it's more stable"
Coverage: AgentDB and ruvector both extensively researched. R20 established the broken bridge. Multiple findings document the relationship. But the specific claim "AgentDB is a simplified ruvector subset" has not been directly verified via import chain analysis.

#### Files to read:
| File | Package | Current Depth | LOC | Action |
|------|---------|--------------|-----|--------|
| packages/agentdb/src/cli/agentdb-cli.ts | agentic-flow-rust | MEDIUM | 3505 | Check imports from ruvector |
| src/cli/agentdb-cli.ts | agentdb | DEEP | 3422 | Compare with ruvector imports |

#### Existing findings that partially address:
- Finding: "AgentDB CLI DOES initialize EmbeddingService with ONNX WASM — this is the standalone native CLI that WORKS" (agentic-flow/src/agentdb/cli/agentdb-cli.ts, CRITICAL)
- Finding: "Native AgentDB MCP server is STANDALONE process with 27 tools. User patched only 6 tools into claude-flow MCP" (packages/agentdb/src/mcp/agentdb-mcp-server.ts, HIGH)
- Finding: "agentdb uses synchronous import of @ruvector/attention — crashes entire module if unavailable" (src/wrappers/attention-native.ts, CRITICAL)
- R20: "AgentDB search broken — ROOT CAUSE: EmbeddingService never initialized"

#### What to verify:
- Does AgentDB's package.json list ruvector as a dependency?
- Is AgentDB a subset (simpler API wrapping ruvector) or a fork (diverged codebase)?
- The "more stable" claim suggests less frequent updates — check git history if available

#### Suggested research agent: cross-repo-tracer

---

### LEAD-014: Dynamic model routing per skill
Classification: PARTIALLY_COVERED
Original claim: "skills have different models... I determine which is the optimal skill... the skill then has an associated model"
Coverage: Model routing extensively researched. 5-6 parallel routing systems documented. But skill-to-model mapping specifically has not been checked.

#### Files to read:
| File | Package | Current Depth | LOC | Action |
|------|---------|--------------|-----|--------|
| crates/ruvllm/src/claude_flow/model_router.rs | ruvector-rust | DEEP | 1322 | Re-read for skill-based routing |
| dist/src/services/headless-worker-executor.js | claude-flow-cli | DEEP | 999 | Check if worker executor routes models per skill |

#### Existing findings that partially address:
- Finding: "LLMRouter is independent of hook-based routing — two disconnected routing systems" (src/services/LLMRouter.ts, HIGH)
- Finding: "5TH ROUTING SYSTEM CONFIRMED: TinyDancerRouter routes tasks->agents via task embeddings, NOT model selection" (agentic-flow/src/routing/TinyDancerRouter.ts, CRITICAL)
- Finding: "All workers hardcoded to model=sonnet contradicting ADR-008 3-tier routing" (dist/src/runtime/headless.js, HIGH)
- Finding: "agent-converter.js: Model selection: Opus for complex/architect/security, Haiku for simple/quick" (dist/sdk/agent-converter.js, HIGH)

#### What to verify:
- The agent-converter.js already does type-based model selection — does this extend to skills?
- Are skills defined with explicit model associations in configuration?
- Is "dynamic" routing happening or is it static configuration?

#### Suggested research agent: reader

---

### LEAD-016: Three-tier agent architecture (15 primary + headless sub-agents + daemon)
Classification: PARTIALLY_COVERED
Original claim: "within the primary cloud code system you can run a maximum of 15 agents... for the headless agents... either use open router or some cheaper models"
Coverage: Headless worker executor (DEEP), OpenRouter proxy (DEEP), worker-daemon (DEEP but 0 findings). Architecture partially documented but the three-tier composition not traced end-to-end.

#### Files to read:
| File | Package | Current Depth | LOC | Action |
|------|---------|--------------|-----|--------|
| dist/src/services/worker-daemon.js | claude-flow-cli | DEEP | 756 | Re-read for daemon capabilities (0 findings currently) |
| dist/src/commands/daemon.js | claude-flow-cli | varies | varies | Check daemon command implementation |

#### Existing findings that partially address:
- Finding: "Three real process execution modes: Headless Worker Executor spawns claude CLI, Container Worker Pool manages Docker, Hive-Mind spawns interactive Claude with Byzantine prompt" (dist/src/services/headless-worker-executor.js, HIGH)
- Finding: "Daemon start only writes PID file, no actual daemon process spawned" (dist/src/commands/process.js, HIGH)
- Finding: "Real service integration: worker-daemon.js getDaemon/startDaemon/stopDaemon" (dist/src/commands/daemon.js, MEDIUM)

#### What to verify:
- Does worker-daemon.js spawn headless Claude Code instances?
- Is the 15-agent limit a Claude Code platform limit or claude-flow configuration?
- Does OpenRouter integration actually work for headless agents?

#### Suggested research agent: reader

---

### LEAD-018: Hive queen with dynamic skill discovery and activation
Classification: PARTIALLY_COVERED
Original claim: "the hive queen the orchestrator... she's able to understand all the various skills available to her and she can pick and choose which skill she needs"
Coverage: Hive-mind command at DEEP. Swarm coordination at ~78% (R31). But skill discovery specifically not traced.

#### Files to read:
| File | Package | Current Depth | LOC | Action |
|------|---------|--------------|-----|--------|
| dist/src/commands/hive-mind.js | claude-flow-cli | DEEP | 1230 | Re-read for skill routing in hive-mind |

#### Existing findings that partially address:
- Finding: "v3-swarm-coordination SKILL: BEST in set. Concrete 15-agent blueprint" (skills/v3-swarm-coordination/SKILL.md, INFO)
- Finding: "Collective Intelligence Coordinator: hive-mind consensus with Byzantine fault tolerance" (agents/v3/collective-intelligence-coordinator.md, HIGH)
- Finding: "Heavy MCP tool dependencies: swarm_init, daa_consensus — many return fabricated metrics" (agents/v3/collective-intelligence-coordinator.md, MEDIUM)
- R31: "CLI = demonstration framework"

#### What to verify:
- Does hive-mind.js enumerate available skills?
- Is there a skill registry that the orchestrator queries?
- Is "picking skills" done via the routing systems or is it prompt-driven?

#### Suggested research agent: reader

---

### LEAD-020: Persistent daemon monitoring Claude Code sessions
Classification: PARTIALLY_COVERED
Original claim: "the demon is always running always monitoring what Claude's up to and saying how can I secure it how can I fix it how can I optimize it"
Coverage: worker-daemon.js at DEEP (0 findings). daemon.js at MEDIUM. Known that process.js daemon start "only writes PID file."

#### Files to read:
| File | Package | Current Depth | LOC | Action |
|------|---------|--------------|-----|--------|
| dist/src/services/worker-daemon.js | claude-flow-cli | DEEP | 756 | Re-read specifically for monitoring/security hooks |
| dist/src/commands/daemon.js | claude-flow-cli | varies | varies | Check daemon lifecycle |

#### Existing findings that partially address:
- Finding: "Daemon start only writes PID file, no actual daemon process spawned" (dist/src/commands/process.js, HIGH)
- Finding: "Real service integration: worker-daemon.js getDaemon/startDaemon/stopDaemon" (dist/src/commands/daemon.js, MEDIUM)

#### What to verify:
- Does worker-daemon.js implement continuous session monitoring?
- Is the "always running" claim accurate, or is it event-driven?
- Does it integrate with hook-pipeline for security/optimization?

#### Suggested research agent: reader

---

## ALREADY_COVERED LEADS (skip)

### LEAD-004: AIDefence sub-millisecond PII detection as kernel preprocessor
Classification: ALREADY_COVERED
Original claim: "AI defense system built right in... personally identifiable information... sub one millisecond... acts as a kind of preprocessor before that information ever hits the OS kernel itself"
Covered by: R92 found AIDefence at 82-88% genuine. AIDefenceGuard.ts (DEEP, 763 LOC) is application-level PII scanning, not kernel-level. PII scrubber found in agentic-flow/src/reasoningbank/utils/pii-scrubber.js (88% quality, HIGH). The sub-millisecond claim is plausible for regex-based PII detection but "kernel preprocessor" is an exaggeration — it operates at application level.

Existing coverage:
- AIDefenceGuard.ts (DEEP, ruvector-rust)
- pii-scrubber.js (DEEP, agentic-flow) — "14 comprehensive regex patterns for email, SSN, API keys, credit cards"
- pii-scrubber.ts (DEEP, agentic-flow) — "Config-gated scrubbing: if config loads fail, ALL PII stored unredacted"
- aidefence-integration.ts (DEEP, agentdb) — simulation scenario

Assessment: AIDefence and PII detection are genuine at the application level. "Kernel preprocessor" claim is aspirational/marketing. Sub-1ms plausible for regex but not verified with benchmarks.

---

### LEAD-009: Ruvector-edge in-browser agents with real-time ML
Classification: ALREADY_COVERED
Original claim: "ruvector edge... in browser and on device agents that chat with each other and exchange information... it's dropped some of my clients' AI bills by a massive amount"
Covered by: edge-net extensively researched. RAC (DEEP, 3326 LOC, 92% genuine), P2P transport via libp2p (DEEP, R42 REVERSAL confirming real P2P), SIMD compute (DEEP, 1418 LOC, complete independent implementation), LoRA adapter sharing (DEEP). The technical substrate for in-browser ML is real. Cost reduction claim is a business metric.

Existing coverage:
- examples/edge-net/src/rac/mod.rs (DEEP) — "RAC = genuine adversarial coherence framework. 3300 LOC production system"
- examples/edge-net/src/network/p2p.rs (DEEP) — "R42 REVERSAL: edge-net HAS production P2P transport via libp2p"
- examples/edge-net/src/compute/simd.rs (DEEP) — "COMPLETE INDEPENDENT SIMD IMPLEMENTATION"
- examples/edge-net/src/ai/lora.rs (DEEP) — "P2P SERIALIZATION: Full adapter sharing across edge network"
- examples/edge-net/src/ai/attention_unified.rs (DEEP) — attention module

Assessment: Technical capabilities are genuine. The edge computing substrate exists and is production-quality. Cost savings claim cannot be verified from source code.

---

### LEAD-019: Time-travel checkpoints for configuration rollback
Classification: ALREADY_COVERED
Original claim: "the time traveling system allows me to go back in time... I can create a checkpoint"
Covered by: temporal-tensor crate DEEP at 93% (HIGHEST QUALITY CRATE from R37). Checkpoint system in standard-checkpoint-hooks.sh (DEEP, genuine git-based). LongRunningAgent has checkpoint/restore but checkpoints stored in-memory only (facade).

Existing coverage:
- crates/ruvector-temporal-tensor/src/store.rs (DEEP, 2284 LOC) — "93% HIGHEST QUALITY"
- standard-checkpoint-hooks.sh (DEEP) — "REAL working git checkpoint system. Creates branches, tags, commits"
- agentic-flow/src/core/long-running-agent.ts (DEEP) — "saveCheckpoint() stores to IN-MEMORY ARRAY, NOT disk"

Assessment: Temporal storage exists and is high quality. Git-based checkpoints are real. The "time-travel" metaphor maps to temporal-tensor's versioned storage plus git checkpointing. Genuine feature.

---

### LEAD-022: rvlite plugin discovery
Classification: ALREADY_COVERED
Original claim: "rv light library... if you want to use gas town in cloud flow you just ask cloud flow to use gas town and it'll find the plugin"
Covered by: R38 found rvlite at 82-86%. Plugin discovery is a claude-flow CLI feature (skills system), not rvlite itself. rvlite is a WASM-based vector DB wrapper with Cypher/SPARQL/SQL query languages.

Existing coverage:
- crates/rvlite/src/lib.rs (DEEP) — "75-80% REAL — WASM wrapper integrating ruvector-core + 3 query languages"
- crates/rvlite/src/cypher/parser.rs (DEEP) — Cypher parser
- crates/rvlite/src/cypher/executor.rs (DEEP) — "PropertyGraph executor exists in rvlite — missing piece from R13"
- crates/rvlite/src/sparql/parser.rs (DEEP) — SPARQL parser
- crates/rvlite/src/sparql/executor.rs (DEEP) — SPARQL executor

Assessment: rvlite is genuine as a lightweight graph/vector DB. Plugin discovery is a separate claude-flow feature. The claim conflates two systems.

---

## UNRESOLVABLE LEADS

### LEAD-007: "Roof bot" = agentic-flow
Classification: UNRESOLVABLE
Original claim: "robot is essentially agentic flow just just be clear I put that out ages ago... the world went all nuts over claude bot which is a very very rudimentary version"
Reason: "Roof bot" / the optimizer desktop app is not in any tracked repository. R40 found agentic-flow = "single-node task runner." If the optimizer uses agentic-flow, it could be an application layer on top of it, but without the optimizer source code, we cannot verify the relationship.

Note: ruvbot (npm/packages/ruvbot) could be the "roof bot" — but it is NOT_TOUCHED in our research DB (0 files at DEEP). Would need a dedicated read session to investigate.

---

### LEAD-011: 150,000 active claude-flow users
Classification: UNRESOLVABLE
Original claim: "cloud flow is hitting I think 150,000 active users"
Reason: Business/usage metric that cannot be verified from source code. npm download stats could provide partial validation but are not part of this research scope.

---

### LEAD-021: External user (Robert) integrating ruvector into unicorn-scan
Classification: UNRESOLVABLE
Original claim: "Robert... I heavily used obviously all your cloud flow stuff I used the browser skill and then I also integrated ruvector into it"
Reason: unicorn-scan is an external project not in our tracked repositories. External integration claims cannot be verified without access to that codebase.

---

## Verification Statistics

| Metric | Value |
|--------|-------|
| Total leads processed | 22 |
| Unique files resolved | ~45 |
| Files already at DEEP | ~25 |
| Files needing first read | ~15 |
| Findings cross-referenced | ~85 |
| Contradictions found | 3 |
| Leads skippable | 7 (4 ALREADY_COVERED + 3 UNRESOLVABLE) |
| Leads actionable | 15 (3 CONTRADICTION + 7 NEW + 5 PARTIALLY_COVERED) |

## Recommended Verification Session Plan

### Phase 1: Contradictions (highest priority, ~1 session)
1. **LEAD-015** — Trace vector-graph-attention composition across all packages. If ANY composing pipeline exists, major correction needed.
2. **LEAD-017** — Targeted re-read of micro_lora.rs for embedding model targeting. Quick verification.
3. **LEAD-013** — Cross-repo scan of Cargo.toml files for wasm32 targets. Count genuine vs stub.

### Phase 2: NEW leads — ruvbot investigation (~1 session)
4. **LEAD-001/002/003/006** — Read ruvbot package (currently 0 DEEP files, ~20 source files). This single package may resolve 4 leads simultaneously if ruvbot IS the "optimizer."
5. **LEAD-010** — Read edge-net pkg/ files (join.js, contribute-daemon.js) for browser compute sharing.
6. **LEAD-005** — Quick read of markov.rs (525 LOC, single file).

### Phase 3: Partial coverage gap-fills (~1 session)
7. **LEAD-008** — AgentDB-ruvector import chain analysis via cross-repo-tracer.
8. **LEAD-014/016/018/020** — Targeted re-reads of DEEP files for skill routing, daemon, hive-mind.

### LEAD-012: Ruvector ~2 million LOC (quick answer)
Classification: CAN BE ANSWERED NOW
ruvector-rust package: 2,473,230 LOC across 6,035 files.
Total across all packages: 5,280,944 LOC across 14,010 files.
The "close to two million" claim is ACCURATE for ruvector-rust alone (2.47M). When including all packages, total is 5.28M.
