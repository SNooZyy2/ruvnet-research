#!/bin/bash
# Ruvector compilation audit — sequential, crash-safe, portable
# Results appended per-crate so partial runs are preserved
#
# ENV VARS (set before running, or edit defaults below):
#   RUVECTOR_REPO  — path to ruvector repo clone
#   RESEARCH_DIR   — path to ruvnet-research checkout
#   CARGO_BUILD_JOBS — parallel rustc jobs (default: auto)
#
# Handles 3 workspace contexts:
#   1. Main ruvector workspace (most crates)
#   2. crates/rvf sub-workspace (rvf-* crates)
#   3. Standalone excluded crates (micro-hnsw-wasm, ruvector-hyperbolic-hnsw*)

set -euo pipefail

# --- CONFIGURE THESE ---
REPO="${RUVECTOR_REPO:-$HOME/repos/ruvector}"
RESEARCH="${RESEARCH_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
# On a beefy machine, leave CARGO_BUILD_JOBS unset (auto = num CPUs).
# On constrained machines, set to 1-2.
# Only export if user explicitly set a value; empty breaks cargo
if [ -z "${CARGO_BUILD_JOBS:-}" ]; then unset CARGO_BUILD_JOBS; fi
export RUSTFLAGS="${RUSTFLAGS:--C debuginfo=0}"
# -----------------------

RESULTS="$RESEARCH/rust-build-tests/results-ruvector.txt"
ERRORS="$RESEARCH/rust-build-tests/results-ruvector-errors.txt"
BATCH_FILE="${1:-$RESEARCH/rust-build-tests/batches/ruvector-all.txt}"
DONE_FILE="$RESEARCH/rust-build-tests/ruvector-done.txt"

echo "Repo:    $REPO"
echo "Results: $RESULTS"
echo "Batch:   $BATCH_FILE"
echo "Jobs:    ${CARGO_BUILD_JOBS:-auto}"
echo ""

# Verify repo exists
if [ ! -f "$REPO/Cargo.toml" ]; then
    echo "ERROR: ruvector repo not found at $REPO"
    echo "Set RUVECTOR_REPO env var or clone to ~/repos/ruvector"
    exit 1
fi

touch "$DONE_FILE"

# Write header only if results file is empty/missing
if [ ! -s "$RESULTS" ]; then
    toolchain=$(rustc --version 2>/dev/null || echo "unknown")
    cat > "$RESULTS" << HEADER
================================================================================
RUVECTOR COMPILATION AUDIT
Toolchain: $toolchain | CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-auto}
================================================================================

CRATE                                    | CHECK    | TEST             | WARN | LOC
-----------------------------------------|----------|------------------|------|------
HEADER
fi

# Determine which workspace directory to use for a given crate
get_workspace_dir() {
    local crate="$1"
    case "$crate" in
        rvf|rvf-types|rvf-wire|rvf-manifest|rvf-index|rvf-quant|rvf-crypto|\
        rvf-runtime|rvf-kernel|rvf-wasm|rvf-solver-wasm|rvf-node|rvf-server|\
        rvf-import|rvf-cli|rvf-launch|rvf-ebpf|rvf-federation|\
        rvf-adapter-*|rvf-adapters-*)
            echo "$REPO/crates/rvf"
            return
            ;;
    esac
    case "$crate" in
        micro-hnsw-wasm|ruvector-hyperbolic-hnsw|ruvector-hyperbolic-hnsw-wasm)
            echo "$REPO/crates/$crate"
            return
            ;;
    esac
    echo "$REPO"
}

find_crate_src() {
    local crate="$1"
    local ws_dir="$2"
    if [ -d "$REPO/crates/$crate/src" ]; then
        echo "$REPO/crates/$crate/src"; return
    fi
    if [ -d "$REPO/crates/rvf/$crate/src" ]; then
        echo "$REPO/crates/rvf/$crate/src"; return
    fi
    local found=$(find "$ws_dir" -maxdepth 4 -name Cargo.toml -not -path '*/target/*' \
        -exec grep -l "name = \"$crate\"" {} \; 2>/dev/null | head -1)
    if [ -n "$found" ]; then
        local dir=$(dirname "$found")
        if [ -d "$dir/src" ]; then echo "$dir/src"; return; fi
    fi
    echo ""
}

total=0
skipped=0
tested=0
start_time=$(date +%s)

while IFS= read -r line; do
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    [[ -z "${line// }" ]] && continue

    crate=$(echo "$line" | awk '{print $1}')
    flags=$(echo "$line" | cut -d' ' -f2-)
    total=$((total + 1))

    if grep -qxF "$crate" "$DONE_FILE" 2>/dev/null; then
        skipped=$((skipped + 1))
        continue
    fi

    if echo "$flags" | grep -q -- '--exclude'; then
        reason=$(echo "$flags" | sed 's/.*--exclude //')
        printf "%-40s | EXCLUDED | %-16s | -    | -\n" "$crate" "$reason" >> "$RESULTS"
        echo "$crate" >> "$DONE_FILE"
        echo "EXCLUDED: $crate ($reason)"
        continue
    fi

    ws_dir=$(get_workspace_dir "$crate")
    elapsed=$(( $(date +%s) - start_time ))
    echo "=== [$tested/$total ${elapsed}s] $crate ($(basename $ws_dir)) ==="
    cd "$ws_dir"

    src_dir=$(find_crate_src "$crate" "$ws_dir")
    if [ -n "$src_dir" ] && [ -d "$src_dir" ]; then
        loc=$(find "$src_dir" -name '*.rs' -exec cat {} + 2>/dev/null | wc -l)
    else
        loc="?"
    fi

    target=""
    if echo "$flags" | grep -q -- '--wasm'; then
        target="--target wasm32-unknown-unknown"
    fi

    # cargo check
    if [ "$ws_dir" != "$REPO" ] && [ "$ws_dir" != "$REPO/crates/rvf" ]; then
        check_out=$(cargo check $target 2>&1) && check_ok=true || check_ok=false
    else
        check_out=$(cargo check -p "$crate" $target 2>&1) && check_ok=true || check_ok=false
    fi
    warn_count=$(echo "$check_out" | grep -c "warning\[" || true)

    if $check_ok; then
        check_result="PASS"
    else
        err_count=$(echo "$check_out" | grep -c "^error\[" || true)
        check_result="FAIL($err_count)"
        echo "--- $crate CHECK ERRORS ---" >> "$ERRORS"
        echo "$check_out" | grep "^error" | head -20 >> "$ERRORS"
        echo "" >> "$ERRORS"
    fi

    # cargo test
    test_result="N/A"
    if $check_ok && ! echo "$flags" | grep -q -- '--skip-test' && ! echo "$flags" | grep -q -- '--wasm'; then
        if [ "$ws_dir" != "$REPO" ] && [ "$ws_dir" != "$REPO/crates/rvf" ]; then
            test_out=$(cargo test --lib -- 2>&1) && test_ok=true || test_ok=false
        else
            test_out=$(cargo test -p "$crate" --lib -- 2>&1) && test_ok=true || test_ok=false
        fi
        summary=$(echo "$test_out" | grep "^test result:" | tail -1)
        if [ -n "$summary" ]; then
            passed=$(echo "$summary" | grep -oP '\d+ passed' | grep -oP '\d+')
            failed=$(echo "$summary" | grep -oP '\d+ failed' | grep -oP '\d+')
            ignored=$(echo "$summary" | grep -oP '\d+ ignored' | grep -oP '\d+')
            test_result="${passed:-0}p/${failed:-0}f/${ignored:-0}i"
        elif ! $test_ok; then
            terr_count=$(echo "$test_out" | grep -c "^error\[" || true)
            test_result="CFAIL($terr_count)"
            echo "--- $crate TEST ERRORS ---" >> "$ERRORS"
            echo "$test_out" | grep "^error" | head -10 >> "$ERRORS"
            echo "" >> "$ERRORS"
        else
            test_result="0p/0f/0i"
        fi
    fi

    printf "%-40s | %-8s | %-16s | %-4s | %s\n" "$crate" "$check_result" "$test_result" "$warn_count" "$loc" >> "$RESULTS"
    echo "$crate" >> "$DONE_FILE"
    tested=$((tested + 1))
    echo "  -> check=$check_result test=$test_result warn=$warn_count loc=$loc"

done < "$BATCH_FILE"

echo ""
total_time=$(( $(date +%s) - start_time ))
echo "=== DONE: $tested tested, $skipped skipped, $total total in ${total_time}s ==="
echo "" >> "$RESULTS"
echo "Run completed: $(date) | ${total_time}s elapsed" >> "$RESULTS"
