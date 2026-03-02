# Transcript Intelligence Extraction — ruv-vods

## What This Is

Subproject for extracting research leads from video transcript VODs. Self-contained — all files, scripts, agents, and output live in this directory.

**Key rule:** Transcripts are **signal sources, not truth sources**. They produce **leads** that require verification against actual source code before entering the research DB.

## Directory Layout

```
ruv-vods/
├── CLAUDE.md              # This file
├── README.md              # Tutorial for future users
├── index.json             # Master tracker (status per transcript)
├── inbox/                 # Drop raw .txt transcripts here
├── chunks/                # Auto-generated chunks (per-transcript subdirs)
├── leads/                 # Analysis output (leads docs)
├── scripts/
│   └── chunk-transcript.js
└── agents/
    └── transcript-analyzer.md
```

## Behavioral Rules

- NEVER auto-index transcript claims into the research database
- NEVER insert findings from transcripts without verification against source code
- ALL transcript analysis output goes to `leads/` as markdown docs
- The `index.json` file tracks processing status — always update it
- Chunk files in `chunks/` are ephemeral — safe to delete after analysis
- When verifying leads, use the parent project's reader/facade-detector agents

## Pipeline

### Step 1: Ingest

Drop a `.txt` transcript file into `inbox/`.

### Step 2: Chunk

```bash
node scripts/chunk-transcript.js inbox/<file.txt>
```

Options: `--chunk-size 3000` (default), `--overlap 200` (default), `--name <custom-name>`

This creates `chunks/{name}/chunk-NNN.txt` files + `manifest.json`, and updates `index.json` status to `chunked`.

### Step 3: SCAN pass (fast triage)

Spawn one agent per chunk in SCAN mode. Use sonnet for research-aware triage.

1. Read `agents/transcript-analyzer.md`
2. For each chunk, spawn:

```
Task(
  subagent_type="general-purpose", model="sonnet", run_in_background=true,
  prompt="<full contents of agents/transcript-analyzer.md>

Assignment:
- Mode: SCAN
- Transcript: {name}
- Chunk file: /home/snoozyy/ruvnet-research/ruv-vods/chunks/{name}/chunk-{NNN}.txt
- Chunk ID: {N}
- Line range: {start}-{end}"
)
```

3. Collect JSON results from each agent
4. Update `index.json` status to `scanned`

### Step 4: Filter

Review scan results. Only chunks with `"recommendAnalysis": true` proceed to the next step.

### Step 5: ANALYZE pass (deep extraction)

For relevant chunks only, spawn sonnet agents in ANALYZE mode:

```
Task(
  subagent_type="general-purpose", model="opus", run_in_background=true,
  prompt="<full contents of agents/transcript-analyzer.md>

Assignment:
- Mode: ANALYZE
- Transcript: {name}
- Chunk file: /home/snoozyy/ruvnet-research/ruv-vods/chunks/{name}/chunk-{NNN}.txt
- Chunk ID: {N}
- Line range: {start}-{end}
- Relevant domains from scan: {domain list}"
)
```

### Step 6: Consolidate

Merge all per-chunk lead outputs into a single file: `leads/{name}-leads.md`

Update `index.json`:
```bash
node -e "
const fs = require('fs');
const idx = JSON.parse(fs.readFileSync('index.json', 'utf-8'));
const t = idx.transcripts.find(t => t.name === '{name}');
t.status = 'analyzed';
t.analyzedAt = new Date().toISOString();
t.leadsFile = 'leads/{name}-leads.md';
t.leadsCount = {N};
fs.writeFileSync('index.json', JSON.stringify(idx, null, 2));
"
```

### Step 6b: Preprocess leads (cross-reference with research DB)

Spawn a lead-verifier agent (opus) with the leads file. It queries the research
DB to resolve file paths, check existing coverage, and classify each lead.

1. Read `agents/lead-verifier.md`
2. Spawn one opus agent with the leads file path
3. Agent queries research DB for file matches, existing findings, depth coverage
4. Outputs `leads/{name}-verification-plan.md`

Update `index.json` status to `preprocessed`.

### Step 7: Verify (separate step, may be another session)

For each high-priority lead:
1. Identify the referenced source files in the ruvnet repos
2. Use the parent project's reader/facade-detector agents to read those files
3. If verified → insert finding into research DB with the parent project's session protocol
4. If rejected → annotate in the leads doc

Update `index.json` status to `verified`.

## Agent Registry

| Mode | Template | subagent_type | model | Purpose |
|------|----------|---------------|-------|---------|
| SCAN | `agents/transcript-analyzer.md` | `general-purpose` | sonnet | Domain-relevance triage with research context awareness |
| ANALYZE | `agents/transcript-analyzer.md` | `general-purpose` | opus | Deep lead extraction |
| PREPROCESS | `agents/lead-verifier.md` | `general-purpose` | opus | DB cross-reference + verification plan generation |

**CRITICAL:** Always read `agents/transcript-analyzer.md` and inject its full contents into the Task prompt. Never spawn with a bare prompt.

## Master Index (`index.json`)

Statuses: `inbox` → `chunked` → `scanned` → `analyzed` → `preprocessed` → `cross-validated` → `verified`

Fields per transcript:
- `name` — identifier (derived from filename)
- `sourceFile` — path to raw transcript
- `totalLines` — line count
- `chunkCount` — number of chunks created
- `status` — current lifecycle stage
- `leadsFile` — path to consolidated leads doc
- `leadsCount` — total leads extracted
- `verifiedCount` — leads verified against source code

## Research Context (inherited from parent)

This subproject operates within the ruvnet research project. The parent `CLAUDE.md` provides:
- Research database schema and query recipes
- 17 research domains and 12 packages
- Session protocol for DB writes
- Agent templates for verification (reader, facade-detector, etc.)

The transcript-analyzer agent template (`agents/transcript-analyzer.md`) contains embedded context about all domains, packages, and the 14 key research patterns to watch for.

## What NOT To Do

- Do NOT run `claude-flow swarm init` for transcript analysis — spawn agents directly via Task tool
- Do NOT insert transcript claims into the research DB without reading the actual source files
- Do NOT use generic agent types (v3-coder, v3-researcher) — use the transcript-analyzer template
- Do NOT edit parent project files from this subproject — leads stay in `leads/`, findings go through the parent's session protocol
