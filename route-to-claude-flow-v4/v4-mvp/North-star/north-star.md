● This is the right question. Let me trace the exact mechanisms the system uses to try to make probabilistic AI behavior deterministic.

  The Designed Architecture: 5 Layers of "Determinism"

  The system was designed with a multi-layer feedback architecture meant to converge toward deterministic behavior over time. Here's what
  the codebase reveals:

  Layer 1: The Hooks Pipeline (the control backbone)

  This is the core orchestration layer. Every AI agent action passes through hooks:

  pre-task → [AI does work] → post-task
  pre-edit → [AI edits file] → post-edit

  What's real:
  - pre-task hook runs ADR-008 3-tier model routing via enhanced-model-router.js — this genuinely selects haiku/sonnet/opus based on task
  complexity scoring. Confirmed working (R140).
  - pre-edit hook begins trajectory tracking and selects agent via Q-table pattern matching — if a file was previously edited successfully
  by a specific agent type, that agent is preferred next time.
  - post-edit hook implements actual Q-learning: Q(s,a) += 0.1 * (reward - Q(s,a)) with reward=1.0 (success) or -0.3 (failure). This is
  real reinforcement learning at the file-agent assignment level.

  What's broken:
  - post-task hook is ORPHANED (R73 CRITICAL). hook-handler.cjs line 140-147 has a stub that bypasses the entire ReasoningBank pipeline.
  The judge→distill→consolidate chain never executes.

  Layer 2: ReasoningBank (learning from experience)

  This is the most architecturally ambitious component — 5 algorithms from what's presented as an internal paper:

  ┌──────────────┬───────────────────────────┬───────────────────────────────────────────┬────────────────────────────────────────────┐
  │  Algorithm   │          Purpose          │              Implementation               │                   Status                   │
  ├──────────────┼───────────────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────┤
  │ 1. Retrieve  │ Find relevant past        │ 4-factor scoring (similarity + recency +  │ WORKS in TS                                │
  │              │ experiences               │ reliability + MMR diversity)              │                                            │
  ├──────────────┼───────────────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────┤
  │ 2. Judge     │ Evaluate trajectory       │ LLM-based or heuristic verdict with       │ VERDICTS LOST — judge.ts calls logMetric() │
  │              │ outcomes                  │ confidence score                          │  but never store() (R75 CRITICAL)          │
  ├──────────────┼───────────────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────┤
  │ 3. Distill   │ Extract reusable          │ LLM-based knowledge extraction with       │ ORPHANED — never called from hooks         │
  │              │ knowledge                 │ prompts                                   │                                            │
  ├──────────────┼───────────────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────┤
  │ 4.           │ Dedup/contradict/prune    │ LSH bucketing, batch queries              │ ORPHANED — lacks EWC++ Fisher weighting    │
  │ Consolidate  │ memory                    │                                           │                                            │
  ├──────────────┼───────────────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────┤
  │ 5. MaTTS     │ Multi-rollout             │ k parallel rollouts with self-contrast    │ ORPHANED — never wired to post-task        │
  │              │ verification              │ aggregation                               │                                            │
  └──────────────┴───────────────────────────┴───────────────────────────────────────────┴────────────────────────────────────────────┘

  The intent is a self-improving loop: AI acts → trajectory recorded → outcome judged → knowledge distilled → future decisions informed by
  past outcomes. This would create convergence — the system would increasingly choose strategies that worked before.

  The reality: the pipeline is broken at the post-task boundary. Data collection works (IntelligenceStore 98% real, SQLite-backed). But the
   learning-from-data path never executes because hook-handler.cjs stubs it out.

  Layer 3: Cryptographic Provenance (tamper-evident audit trail)

  Multiple genuine cryptographic mechanisms exist:

  - RVF witness chains (witness.rs, 953 LOC): SHAKE-256 hashing creates cryptographic tamper-evidence on vector DB entries. Each entry
  links to its predecessor — you can verify the entire history hasn't been altered. Genuine in Rust.
  - RAC (Adversarial Coherence) (rac/mod.rs, 3,326 LOC): Merkle event log (RFC 6962), Ed25519 signatures, Byzantine fault tolerance. This
  is a real adversarial verification framework — agents can challenge each other's claims, and resolutions are cryptographically signed.
  - Ed25519 anti-hallucination (ed25519-verifier-real.ts, 406 LOC): Genuine @noble/ed25519 crypto for signing AI outputs. The idea: AI
  generates output → output is signed → signature can verify the output hasn't been modified.
  - Prime-radiant governance witnesses (witness.rs, 723 LOC): Governance actions (proposal, vote, ratification) are logged with
  cryptographic witness chains.

  What's broken:
  - RVF NAPI bridge's verify_witness() bypasses the real verify_witness_chain() — the Rust verification exists but the JS bridge
  short-circuits it.
  - Ed25519 verifier uses hardcoded example root keys — not wired to a real key management system.
  - RAC exists only in examples/edge-net/, not wired into the main orchestration layer.
  - Anti-hallucination plugin (state-of-art-anti-hallucination.ts) is never registered, never loaded (R46 CRITICAL).

  Layer 4: SONA Optimizer (Bayesian convergence)

  This is the only provenance mechanism that actually works end-to-end:

  sona-optimizer.ts (842 LOC, 72-78% genuine) implements Bayesian agent-routing with temporal decay. It:
  1. Maintains confidence distributions per agent type per task category
  2. Updates beliefs based on observed outcomes (genuine Bayesian update)
  3. Uses Thompson sampling for exploration/exploitation balance
  4. Applies temporal decay so recent outcomes weigh more

  It's wired into the hooks pipeline and genuinely influences which model/agent handles each task. Over time, this converges — the system
  learns which agents perform best for which tasks.

  But: it operates on routing decisions only. It doesn't verify the content of AI outputs.

  Layer 5: Claims System (coordination, not verification)

  claim-service.ts provides resource ownership claims — when Agent A claims a file, Agent B can't edit it simultaneously. This prevents
  conflicting actions but doesn't verify correctness.

  Broken: LOCAL-ONLY JSON file. Competes with claims-tools.ts using incompatible formats (2-part vs 3-part claimant IDs).

  ---
  The Core Answer: How Was It Designed to Be Deterministic?

  The design philosophy is convergent determinism through feedback loops, not mathematical determinism. The architecture says:

                      ┌─────────────────────────────────┐
                      │     CRYPTOGRAPHIC PROVENANCE     │
                      │  (witness chains, Ed25519 sigs)  │
                      │  "You can VERIFY what happened"  │
                      └──────────────┬──────────────────┘
                                     │
      ┌──────────┐    ┌──────────────▼──────────────┐    ┌──────────┐
      │ pre-task  │───▶│       AI DOES WORK          │───▶│post-task │
      │ routing   │    │  (probabilistic, yes)       │    │ judgment │
      └─────▲─────┘    └────────────────────────────┘    └────┬─────┘
            │                                                  │
            │         ┌────────────────────────────┐          │
            │         │     REASONINGBANK           │          │
            └─────────│  judge → distill → store    │◀─────────┘
                      │  "Learn from outcomes"       │
                      └────────────────────────────┘

  The idea: AI is probabilistic, but:
  1. Pre-task: Route to the right model/agent based on learned patterns (SONA Bayesian)
  2. During task: Record trajectory (every step logged)
  3. Post-task: Judge outcome, distill knowledge, consolidate memory
  4. Cryptographic trail: Every action witness-chained so you can audit
  5. Next invocation: Retrieve relevant past experiences, choose strategies that worked

  Over many iterations, this should converge — the system would increasingly make the same (successful) choices for similar tasks,
  approaching determinism through empirical reinforcement rather than mathematical guarantee.

  The Brutal Reality

  The feedback loop is severed at the post-task boundary. The system collects data (trajectories, patterns, routing decisions) but the
  learning pipeline that would create convergence is orphaned. Specifically:

  - hook-handler.cjs stubs out the ReasoningBank pipeline — judge/distill/consolidate never run
  - Judge verdicts are lost — persisted to metrics log, not to retrievable memory
  - MaTTS (multi-rollout verification) is orphaned — the most sophisticated verification mechanism never executes
  - Cryptographic witnesses are bypassed at the JS bridge layer
  - Anti-hallucination plugins are never loaded

  What does work:
  - SONA Bayesian model routing (converges on model selection)
  - Q-learning at CLI level (converges on agent-file assignment)
  - IntelligenceStore trajectory storage (data accumulates, but isn't fed back effectively)
  - intelligence.ts stores patterns to disk — but uses O(n) brute-force to retrieve them, and the LoRA/EWC config that would enable actual
  neural adaptation is stored but never referenced

  Bottom line: The system was designed with a genuine, sophisticated approach to the determinism problem — learn from outcomes, verify
  cryptographically, converge through reinforcement. About 70% of this architecture is genuinely implemented. But the critical post-task
  feedback loop is broken, which means the learning never happens. The system makes probabilistic decisions, records them, but doesn't
  learn from them. The cryptographic provenance layer exists in Rust but is bypassed at every JS bridge point.

  Ruv's "provenance guarantees outcomes" claim maps to real architectural intent backed by real code — but the wiring between the layers is
   incomplete. The algorithms exist; the pipeline doesn't flow.