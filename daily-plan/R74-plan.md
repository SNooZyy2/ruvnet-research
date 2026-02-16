# R74 Execution Plan: ReasoningBank-Network Rust Crate + TS Inner Ring

**Date**: 2026-02-16
**Session ID**: 74
**Focus**: Complete reasoningbank-network Rust crate (neural_bus module — gossip, frames, streams, snapshots, priority) + 3 high-value ReasoningBank TS PROXIMATE files (CLI commands, JS matts mirror, type definitions).
**Strategic value**: First purely PROXIMATE session (R73 clears CONNECTED tier). Extends the R67 "genuinely architected" Rust workspace arc into its networking layer. The neural_bus is the only ReasoningBank crate with distributed communication primitives — determines whether cross-agent memory sharing is real or theatrical. The 3 TS files close the type/CLI gap around the R72-R73 TS core.

## Rationale

After R73 clears the CONNECTED tier (milestone), 163 PROXIMATE files remain as the sole frontier. Of these, 29 are in agentic-flow-rust (16 nearby DEEP each in reasoningbank-network), 76 in sublinear-rust, and 52 in ruv-fann-rust.

The **reasoningbank-network** crate is a complete 7-file Rust crate that has never been read. R67 established the ReasoningBank Rust workspace as genuinely architected (core 88-92%, storage 94%, learning 95-98%, mcp 93-95%). The network crate is the missing piece — it contains a `neural_bus` module with gossip protocols, frame serialization, stream management, snapshot/state transfer, and priority scheduling. If genuine, this is the distributed communication backbone for cross-agent ReasoningBank synchronization. If theatrical, it joins the long list of networking facades (swarm_transport.rs 28-32%, remote.js 15%).

The 3 TS files fill gaps in the R72-R73 TS picture: `reasoningbankCommands.ts` is the CLI entry point for ReasoningBank (does it use the TS API or bypass it?), `matts.js` is the JS mirror of R72's matts.ts (85-88% — compiled or independent?), and `types/index.ts` defines the shared type layer (does it match the Rust types from R67?).

## Target: 10 files, ~2,234 LOC

---

### Cluster A: ReasoningBank-Network Rust Crate (7 files, ~1,636 LOC)

Complete neural_bus module — the distributed communication layer for ReasoningBank. R67 found the Rust workspace genuinely architected; this crate handles inter-agent memory sharing. Key question: is the networking real (like quic.rs 96%) or theatrical (like swarm_transport.rs 28%)?

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 1 | 13461 | `reasoningbank/crates/reasoningbank-network/src/neural_bus/frame.rs` | 313 | Frame serialization — binary protocol for neural bus messages? |
| 2 | 13467 | `reasoningbank/crates/reasoningbank-network/src/neural_bus/streams.rs` | 307 | Stream management — multiplexed channels? Connection handling? |
| 3 | 13466 | `reasoningbank/crates/reasoningbank-network/src/neural_bus/snapshot.rs` | 291 | State snapshots — consistent state transfer for new nodes? |
| 4 | 13462 | `reasoningbank/crates/reasoningbank-network/src/neural_bus/gossip.rs` | 238 | Gossip protocol — epidemic dissemination? CRDT-based? |
| 5 | 13464 | `reasoningbank/crates/reasoningbank-network/src/neural_bus/mod.rs` | 214 | Module root — composes the neural bus subsystem |
| 6 | 13465 | `reasoningbank/crates/reasoningbank-network/src/neural_bus/priority.rs` | 211 | Priority scheduling — QoS for different message types? |
| 7 | 13460 | `reasoningbank/crates/reasoningbank-network/src/lib.rs` | 62 | Crate root — what does it export? Does it compose neural_bus into a usable API? |

**Full paths**:
1. `~/repos/agentic-flow/reasoningbank/crates/reasoningbank-network/src/neural_bus/frame.rs`
2. `~/repos/agentic-flow/reasoningbank/crates/reasoningbank-network/src/neural_bus/streams.rs`
3. `~/repos/agentic-flow/reasoningbank/crates/reasoningbank-network/src/neural_bus/snapshot.rs`
4. `~/repos/agentic-flow/reasoningbank/crates/reasoningbank-network/src/neural_bus/gossip.rs`
5. `~/repos/agentic-flow/reasoningbank/crates/reasoningbank-network/src/neural_bus/mod.rs`
6. `~/repos/agentic-flow/reasoningbank/crates/reasoningbank-network/src/neural_bus/priority.rs`
7. `~/repos/agentic-flow/reasoningbank/crates/reasoningbank-network/src/lib.rs`

**Key questions**:
- `frame.rs` (313 LOC): Does it define a real binary wire protocol (magic bytes, version, length-prefixed)? Or just Rust structs with serde? Does it support different frame types (data, control, heartbeat)? Is there proper error handling for malformed frames?
- `streams.rs` (307 LOC): Multiplexed streams over a single connection? Does it use tokio channels, or implement its own stream abstraction? Connection lifecycle (open, half-close, reset)?
- `snapshot.rs` (291 LOC): Full state snapshot for new joining nodes? Incremental/delta snapshots? Does it serialize the ReasoningBank memory state (trajectories, patterns) or just metadata?
- `gossip.rs` (238 LOC): Real gossip protocol (SWIM, HyParView, epidemic broadcast)? Or just a broadcast wrapper? Does it handle membership, failure detection, and convergence? At 238 LOC a genuine gossip impl is tight but feasible.
- `mod.rs` (214 LOC): How does it compose the 5 submodules? Does it define a NeuralBus trait/struct? Is there a runtime that ties frame+streams+gossip+snapshot+priority together?
- `priority.rs` (211 LOC): Priority queue for message scheduling? Does it implement deadline-based, weight-based, or strict priority? Does it interact with the gossip protocol for priority dissemination?
- `lib.rs` (62 LOC): At 62 LOC, likely a thin barrel. Does it re-export neural_bus or add crate-level configuration? Are there feature flags?

---

### Cluster B: ReasoningBank TS Inner Ring (3 files, ~598 LOC)

The CLI command layer, JS matts mirror, and shared type definitions. These fill the remaining gaps around R72's TS core (matts.ts 85-88%, consolidate.ts 82-86%, distill.ts 78-82%) and R73's outer ring (index.ts, config.ts, wasm-adapter.ts).

| # | File ID | File | LOC | Context |
|---|---------|------|-----|---------|
| 8 | 10917 | `agentic-flow/src/utils/reasoningbankCommands.ts` | 238 | CLI entry point — does it use the TS API (index.ts) or bypass it? |
| 9 | 10814 | `agentic-flow/src/reasoningbank/core/matts.js` | 226 | JS mirror of matts.ts (R72: 85-88%) — compiled or independent impl? |
| 10 | 10837 | `agentic-flow/src/reasoningbank/types/index.ts` | 134 | Shared type definitions — match Rust types from R67? |

**Full paths**:
8. `~/repos/agentic-flow/agentic-flow/src/utils/reasoningbankCommands.ts`
9. `~/repos/agentic-flow/agentic-flow/src/reasoningbank/core/matts.js`
10. `~/repos/agentic-flow/agentic-flow/src/reasoningbank/types/index.ts`

**Key questions**:
- `reasoningbankCommands.ts` (238 LOC): Is this the `claude-flow reasoningbank` CLI handler? Does it import from the ReasoningBank TS barrel (index.ts) or directly call queries.js/backend-selector.ts? Does it support store/retrieve/search/consolidate commands? How does it compare to the MCP tools (R67: 93-95%)?
- `matts.js` (226 LOC): R72 found matts.ts (85-88%) implements genuine Memory-Aware Test-Time Scaling (Algorithm 5). Is matts.js a compiled version of matts.ts, or an independent earlier JS implementation? Does it have the same Algorithm 5 or a different approach? If independent, is quality comparable?
- `types/index.ts` (134 LOC): What interfaces does it define — Trajectory, Pattern, Verdict, Memory? Do they align with the Rust types from R67's core crate? Does it use Zod/io-ts validation or plain TypeScript interfaces? Is this imported by the core TS files (matts.ts, consolidate.ts, distill.ts)?

---

## Expected Outcomes

1. **ReasoningBank-Network crate COMPLETE**: 7/7 files → full neural_bus assessment
2. **Networking verdict**: Genuine distributed communication (like quic.rs 96%) or theatrical (like swarm_transport.rs 28%)?
3. **Gossip protocol assessment**: Real membership + failure detection, or broadcast wrapper?
4. **ReasoningBank TS type layer mapped**: Do TS types match Rust types?
5. **CLI→API pathway traced**: Does reasoningbankCommands.ts use the barrel or bypass it?
6. **matts.js independence**: Compiled from matts.ts or independent JS implementation?
7. **DEEP files**: ~1,184 → ~1,194 (post-R73 estimate + 10)
8. **Priority queue**: ~165 → ~155 (all PROXIMATE/NEARBY/DOMAIN_ONLY)

## DB Update Template

```javascript
const db = require('better-sqlite3')('/home/snoozyy/ruvnet-research/db/research.db');
const sessionId = 74;
const today = new Date().toISOString().slice(0, 10);

// File IDs for this session:
// 13461: frame.rs (313 LOC) — reasoningbank-network, neural_bus, PROXIMATE
// 13467: streams.rs (307 LOC) — reasoningbank-network, neural_bus, PROXIMATE
// 13466: snapshot.rs (291 LOC) — reasoningbank-network, neural_bus, PROXIMATE
// 13462: gossip.rs (238 LOC) — reasoningbank-network, neural_bus, PROXIMATE
// 13464: mod.rs (214 LOC) — reasoningbank-network, neural_bus, PROXIMATE
// 13465: priority.rs (211 LOC) — reasoningbank-network, neural_bus, PROXIMATE
// 13460: lib.rs (62 LOC) — reasoningbank-network, crate root, PROXIMATE
// 10917: reasoningbankCommands.ts (238 LOC) — agentic-flow utils, PROXIMATE
// 10814: matts.js (226 LOC) — reasoningbank core, PROXIMATE
// 10837: types/index.ts (134 LOC) — reasoningbank types, PROXIMATE
```

## Domain Tags

- Files 1-7 → `memory-and-learning` (needs tagging — new crate not yet domain-tagged)
- Files 8-10 → `memory-and-learning` (already tagged via agentic-flow/src/reasoningbank)

## Isolation Check

No selected files are in known-isolated subtrees. The reasoningbank-network crate is inside the agentic-flow-rust package which is CONNECTED (not ISOLATED). It has 16 nearby DEEP files in the broader reasoningbank workspace. The 3 TS files are in agentic-flow/src/reasoningbank/ with 75 nearby DEEP files — the densest cluster in the queue. All files safe to read.

Reliable isolated subtrees (confirmed excluded): neuro-divergent, cuda-wasm, ruvector/patches, ruvector/scripts, agentdb/simulation — none overlap with selected files.

---

## Synthesis Doc Update Protocol (ADR-040)

**MANDATORY**: After all files are read and findings inserted into the DB, update the relevant `domains/*/analysis.md` files following the ADR-040 in-place protocol. Reference: `domains/memory-and-learning/analysis.md` for canonical structure.

### Rules for Each Section

| Section | Action | NEVER Do |
|---------|--------|----------|
| **1. Current State Summary** | REWRITE in-place to reflect current state | Append session narrative |
| **2. File Registry** | ADD new rows to existing subsystem tables, UPDATE rows if re-read | Duplicate rows, create per-session file tables |
| **3. Findings Registry** | ADD new findings with next sequential ID (C{max+1}, H{max+1}) to 3a/3b | Create `### R74 Findings` blocks, re-list old findings, restart ID numbering |
| **4. Positives Registry** | ADD new positives with session tag | Re-list existing positives |
| **5. Subsystem Sections** | UPDATE existing sections, CREATE new ones by topic | Create per-session narrative blocks |
| **8. Session Log** | APPEND 2-5 line entry for this session | Put findings here, write full narratives |

### Finding ID Assignment

Before adding findings, check the current max ID in the target domain's analysis.md:
- Section 3a: find last `| C{N} |` row → new CRITICALs start at C{N+1}
- Section 3b: find last `| H{N} |` row → new HIGHs start at H{N+1}

**ID format**: `| {ID} | **{short title}** — {description} | {file(s)} | R{session} | Open |`

### Anti-Patterns (NEVER do these)

- **NEVER** create `### R74 Findings (Session date)` blocks outside Section 3
- **NEVER** append findings after Section 8
- **NEVER** create `### R74 Full Session Verdict` blocks
- **NEVER** use finding IDs that collide with existing ones (always check max first)
- **NEVER** re-list findings from previous sessions

### Synthesis Update Checklist

- [ ] Section 1 rewritten with updated state
- [ ] New file rows added to Section 2 (correct subsystem table)
- [ ] New findings added to Section 3a/3b with sequential IDs
- [ ] New positives added to Section 4 (if any)
- [ ] Relevant subsystem sections in Section 5 updated
- [ ] Session log entry appended to Section 8 (2-5 lines max)
- [ ] No per-session finding blocks created anywhere
