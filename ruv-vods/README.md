# ruv-vods — Transcript Intelligence Extraction

Self-contained subproject for extracting research leads from video transcripts (VODs).

**Key principle:** Transcripts are **signal sources, not truth sources**. They produce *leads* — unverified claims that must be verified against actual source code before entering the research database.

## Directory Structure

```
ruv-vods/
├── README.md              # This file — tutorial and reference
├── index.json             # Master tracker — status of every transcript
├── inbox/                 # Drop raw .txt transcripts here
│   └── live-february-26.txt
├── chunks/                # Auto-generated chunks (per-transcript subdirs)
│   └── {name}/
│       ├── chunk-001.txt
│       ├── chunk-002.txt
│       └── manifest.json
├── leads/                 # Analysis output — verified leads docs
│   └── {name}-leads.md
├── scripts/               # Automation scripts
│   └── chunk-transcript.js
└── agents/                # Agent prompt templates
    └── transcript-analyzer.md
```

## File Index

| File | Purpose | Location |
|------|---------|----------|
| `index.json` | Master tracker with status per transcript | `ruv-vods/index.json` |
| `chunk-transcript.js` | Splits transcripts into overlapping chunks | `ruv-vods/scripts/chunk-transcript.js` |
| `transcript-analyzer.md` | Two-mode agent template (SCAN + ANALYZE) | `ruv-vods/agents/transcript-analyzer.md` |
| `lead-verifier.md` | Cross-references leads against research DB | `ruv-vods/agents/lead-verifier.md` |

## Lifecycle of a Transcript

Each transcript moves through these statuses (tracked in `index.json`):

```
inbox → chunked → scanned → analyzed → preprocessed → cross-validated → verified
```

| Status | Meaning |
|--------|---------|
| `inbox` | Raw .txt dropped in `inbox/`, not yet processed |
| `chunked` | Split into chunks by `chunk-transcript.js` |
| `scanned` | SCAN pass complete, relevant chunks identified |
| `analyzed` | ANALYZE pass complete, leads extracted |
| `preprocessed` | Leads cross-referenced against research DB, verification plan generated |
| `cross-validated` | Verification plan merged with others, deduplicated, R-plans generated |
| `verified` | Leads verified against source code, findings in research DB |

## Quick Start

### Step 0: Drop a transcript

Place a `.txt` transcript file in `ruv-vods/inbox/`:

```bash
cp ~/Downloads/stream-2026-03-01.txt ruv-vods/inbox/
```

### Step 1: Chunk it

```bash
node ruv-vods/scripts/chunk-transcript.js ruv-vods/inbox/stream-2026-03-01.txt
```

This creates `ruv-vods/chunks/stream-2026-03-01/chunk-NNN.txt` files and updates `index.json`.

Options:
- `--chunk-size 3000` — lines per chunk (default: 3000)
- `--overlap 200` — overlap between chunks (default: 200)
- `--name custom-name` — override the transcript name

### Step 2: SCAN pass (fast triage)

Tell Claude: **"analyze transcript stream-2026-03-01"**

Or manually: read `ruv-vods/agents/transcript-analyzer.md`, then spawn one haiku agent per chunk in SCAN mode. Each agent reads its chunk and outputs a JSON relevance assessment.

The SCAN pass filters out irrelevant chunks (off-topic discussion, greetings, setup talk) so the expensive ANALYZE pass only runs on content that matters.

### Step 3: ANALYZE pass (deep extraction)

For each chunk where SCAN returned `recommendAnalysis: true`, spawn a sonnet agent in ANALYZE mode. Each agent outputs structured leads.

### Step 4: Consolidate leads

Merge all per-chunk leads into a single file: `ruv-vods/leads/{name}-leads.md`

Update `index.json` status to `analyzed`.

### Step 4b: Preprocess leads

Before verification, cross-reference leads against the research DB:

1. Read `agents/lead-verifier.md`
2. Spawn one opus agent with the leads file path
3. Agent queries research DB for file matches, existing findings, depth coverage
4. Outputs `leads/{name}-verification-plan.md`

The verification plan categorizes each lead as:
- **ALREADY_COVERED** — skip, existing research handles it
- **PARTIALLY_COVERED** — re-read specific sections of known files
- **NEW** — file needs first read or doesn't exist in DB
- **CONTRADICTION** — lead contradicts existing finding (highest priority)
- **UNRESOLVABLE** — referenced files don't exist in any repo

### Step 4c: Cross-validate verification plans

**Do NOT create R-plans from a single verification plan.** Wait until ALL pending transcripts are preprocessed, then cross-validate:

1. Collect all `leads/{name}-verification-plan.md` files
2. Deduplicate files — the same file may appear in multiple leads across transcripts
3. Merge contradictions — if two transcripts make opposing claims about the same file, flag it
4. Consolidate priorities — a file referenced by 3 leads is higher priority than one referenced by 1
5. Output: `leads/cross-validated-plan.md` with a single deduplicated file list

**Rules:**
- A file appears ONCE in the cross-validated plan, even if referenced by multiple leads
- Each entry lists ALL leads that reference it (from any transcript)
- Priority = max(individual lead priorities) + boost for multi-lead references
- Contradictions between transcripts get their own section

### Step 4d: Generate R-plans

From the cross-validated plan, create session plans in `ruvnet-research/daily-plan/`:

1. Group files into sessions of 7-9 files each (~5,000-8,000 LOC)
2. Keep related files together (same crate/module in one session)
3. Name as `R{N}-{topic}.md` following existing convention
4. Each R-plan is a standard research session — spawn readers from `daily-plan/`

### Step 5: Verify leads (manual / reader agents)

Execute the R-plans as normal research sessions. For each file read:
1. Use research reader/facade-detector agents
2. If a lead's claim checks out → insert a finding into the research DB
3. If the claim is wrong → annotate in the leads doc as REJECTED
4. Update the verification plan with results

Update `ruvnet-research/ruv-vods/index.json` status to `verified`.

## Agent Template

The `transcript-analyzer.md` agent operates in two modes:

### SCAN mode (haiku)
- Fast triage of a single chunk
- Outputs JSON with domain relevance scores
- Sets `recommendAnalysis: true/false`
- Catches: file mentions, crate names, architecture decisions, bug references

### ANALYZE mode (sonnet)
- Deep extraction from relevant chunks only
- Outputs structured LEAD entries with:
  - Domain classification (17 research domains)
  - Lead type (IMPLEMENTATION, ARCHITECTURE, CORRECTION, etc.)
  - Verification action + difficulty + suggested agent
  - Priority ranking (HIGH/MEDIUM/LOW)
- Knows all 14 key research patterns (hash embeddings, HNSW, facades, etc.)

### Spawning agents

```
Task(
  subagent_type="general-purpose",
  model="haiku",           # or "sonnet" for ANALYZE
  run_in_background=true,
  prompt="<contents of ruv-vods/agents/transcript-analyzer.md>

Assignment:
- Mode: SCAN              # or ANALYZE
- Transcript: {name}
- Chunk file: ruv-vods/chunks/{name}/chunk-001.txt
- Chunk ID: 1
- Line range: 1-3000"
)
```

## Master Index (`index.json`)

The index tracks every transcript and its processing status. The chunking script auto-updates it. Manual updates for scan/analyze/verify status:

```bash
node -e "
const fs = require('fs');
const idx = JSON.parse(fs.readFileSync('ruv-vods/index.json', 'utf-8'));
const t = idx.transcripts.find(t => t.name === 'stream-name');
t.status = 'analyzed';
t.analyzedAt = new Date().toISOString();
t.leadsFile = 'leads/stream-name-leads.md';
t.leadsCount = 47;
fs.writeFileSync('ruv-vods/index.json', JSON.stringify(idx, null, 2));
"
```

## Design Decisions

**Why leads, not findings?**
Transcripts contain claims that may be outdated, aspirational, or wrong. The research DB has strict quality standards (1,500+ DEEP files, 11,686+ findings). Auto-indexing transcript claims would contaminate the dataset.

**Why two passes?**
A 50K-line transcript produces ~17 chunks. Not all are relevant. The SCAN pass (haiku, cheap) filters to maybe 8-10 relevant chunks. The ANALYZE pass (sonnet, expensive) only runs on those. This saves ~40-50% of tokens.

**Why self-contained in `ruv-vods/`?**
The transcript system is experimental and separate from the core research pipeline. Keeping it isolated prevents file pollution in the main repo structure and makes it easy to iterate without affecting the research DB.

**Why not a database table?**
Transcripts are transient inputs, not source-of-truth data. A simple JSON index is sufficient for tracking status. Only verified leads graduate to the research DB as proper findings.
