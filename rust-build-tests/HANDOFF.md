# Rust Compilation Audit — Remote Machine Handoff

## What this is

We're auditing ~140 Rust crates across 4 repos to answer: **does the code actually compile? Do tests pass?** This produces hard binary truth signals to cross-reference against our static "realness" scores.

Two repos are already done (agentic-flow: 11 crates, sublinear: 9 crates). Results are in `rust-build-tests/results-*.txt`. The remaining two repos need to run on this machine:

| Repo | Crates | Script | Estimated time |
|------|--------|--------|---------------|
| ruv-FANN | 24 | `run-ruv-fann.sh` | ~10-15 min |
| ruvector | 115 | `run-ruvector.sh` | ~30-60 min |

## File layout on this machine

```
~/personal-projects/
├── repos/
│   ├── ruvector/          # https://github.com/ruvnet/ruvector
│   └── ruv-FANN/          # https://github.com/ruvnet/ruv-FANN
└── ruvnet-research/       # https://github.com/SNooZyy2/ruvnet-research
    └── rust-build-tests/
        ├── run-ruvector.sh        # Main script for ruvector
        ├── run-ruv-fann.sh        # Main script for ruv-FANN
        ├── batches/
        │   ├── ruvector-all.txt   # 115 crates ordered by priority tier
        │   └── ruv-fann-all.txt   # 24 crates across 4 sub-workspaces
        ├── results-agentic-flow.txt   # DONE (from remote server)
        └── results-sublinear.txt      # DONE (from remote server)
```

## Prerequisites

```bash
# 1. Rust toolchain (need stable >= 1.80)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# 2. WASM target (needed for ~25 WASM crates)
rustup target add wasm32-unknown-unknown

# 3. Verify
rustc --version   # should be 1.8x+
cargo --version
rustup target list --installed  # should include wasm32-unknown-unknown
```

No other dependencies needed — the scripts only use `cargo check` and `cargo test` which pull crate deps automatically.

## Running the audit

The scripts use env vars to find the repos. Set them to match this machine's layout:

```bash
cd ~/personal-projects/ruvnet-research

# Set paths (repos are siblings, not under ~/repos/)
export RUVECTOR_REPO=~/personal-projects/repos/ruvector
export RUVFANN_REPO=~/personal-projects/repos/ruv-FANN

# Run ruv-FANN first (smaller)
bash rust-build-tests/run-ruv-fann.sh

# Then ruvector (the big one)
bash rust-build-tests/run-ruvector.sh
```

### If something goes wrong

- **Scripts are crash-safe.** Just re-run the same command. Completed crates are tracked in `*-done.txt` files and automatically skipped.
- **To re-test a specific crate:** Remove its line from the relevant `*-done.txt` file, then re-run.
- **To reset entirely:** Delete the `*-done.txt`, `results-*.txt`, and `*-errors.txt` files in `rust-build-tests/`.
- **OOM or too slow:** Set `export CARGO_BUILD_JOBS=2` (or 1) before running.

## What the scripts produce

Per script run, 2-3 files appear in `rust-build-tests/`:

| File | Content |
|------|---------|
| `results-ruvector.txt` | One-line-per-crate summary table (check/test/warnings/LOC) |
| `results-ruvector-errors.txt` | First 20 error lines per failing crate |
| `ruvector-done.txt` | List of completed crate names (for resume) |

Same pattern for `ruv-fann`.

## After completion — push results back

```bash
cd ~/personal-projects/ruvnet-research
git add rust-build-tests/results-*.txt rust-build-tests/*-done.txt rust-build-tests/*-errors.txt
git commit -m "Rust compilation audit: ruvector + ruv-FANN results"
git push
```

The remote server will `git pull` and ingest the results into the research database.

## What we're looking for

The most valuable findings are **disagreements** between our static analysis scores and compilation reality:
- Code we rated 85%+ genuine that fails to compile → our scores were too generous
- Code we rated as facade that compiles clean → maybe we were wrong
- `cargo check` passes but `cargo test --lib` fails to compile → proves tests were never run
- Cross-crate type mismatches → proof of disconnected development
- `dead_code` warnings → confirms our unused-code findings
