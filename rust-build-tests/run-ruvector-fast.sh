#!/bin/bash
# Two-phase ruvector audit: check ALL first (fast), then test passing crates
# Phase 1: cargo check only (~2-5s per crate with cached deps)
# Phase 2: cargo test only on crates that passed check

set -uo pipefail

REPO="${RUVECTOR_REPO:-$HOME/repos/ruvector}"
RESEARCH="${RESEARCH_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
export RUSTFLAGS="${RUSTFLAGS:--C debuginfo=0}"

RESULTS="$RESEARCH/rust-build-tests/results-ruvector.txt"
ERRORS="$RESEARCH/rust-build-tests/results-ruvector-errors.txt"
BATCH_FILE="$RESEARCH/rust-build-tests/batches/ruvector-all.txt"
DONE_FILE="$RESEARCH/rust-build-tests/ruvector-done.txt"
CHECK_PASS_FILE="$RESEARCH/rust-build-tests/ruvector-check-passed.txt"

touch "$DONE_FILE" "$CHECK_PASS_FILE"

get_ws_dir() {
    local crate="$1"
    case "$crate" in
        rvf|rvf-types|rvf-wire|rvf-manifest|rvf-index|rvf-quant|rvf-crypto|\
        rvf-runtime|rvf-kernel|rvf-wasm|rvf-solver-wasm|rvf-node|rvf-server|\
        rvf-import|rvf-cli|rvf-launch|rvf-ebpf|rvf-federation|rvf-adapter-*)
            echo "$REPO/crates/rvf" ;;
        micro-hnsw-wasm|ruvector-hyperbolic-hnsw|ruvector-hyperbolic-hnsw-wasm)
            echo "$REPO/crates/$crate" ;;
        *) echo "$REPO" ;;
    esac
}

find_loc() {
    local crate="$1" ws_dir="$2"
    for d in "$REPO/crates/$crate/src" "$REPO/crates/rvf/$crate/src"; do
        [ -d "$d" ] && find "$d" -name '*.rs' -exec cat {} + 2>/dev/null | wc -l && return
    done
    local found=$(find "$ws_dir" -maxdepth 4 -name Cargo.toml -not -path '*/target/*' \
        -exec grep -l "name = \"$crate\"" {} \; 2>/dev/null | head -1)
    if [ -n "$found" ] && [ -d "$(dirname "$found")/src" ]; then
        find "$(dirname "$found")/src" -name '*.rs' -exec cat {} + 2>/dev/null | wc -l && return
    fi
    echo "?"
}

# ===== PHASE 1: cargo check all remaining =====
echo "===== PHASE 1: cargo check (fast) ====="
phase1_start=$(date +%s)
checked=0

while IFS= read -r line; do
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    [[ -z "${line// }" ]] && continue

    crate=$(echo "$line" | awk '{print $1}')
    flags=$(echo "$line" | cut -d' ' -f2-)

    grep -qxF "$crate" "$DONE_FILE" 2>/dev/null && continue

    if echo "$flags" | grep -q -- '--exclude'; then
        reason=$(echo "$flags" | sed 's/.*--exclude //')
        printf '%-40s | EXCLUDED | %-16s | -    | -\n' "$crate" "$reason" >> "$RESULTS"
        echo "$crate" >> "$DONE_FILE"
        echo "EXCLUDED: $crate ($reason)"
        continue
    fi

    ws_dir=$(get_ws_dir "$crate")
    cd "$ws_dir"
    checked=$((checked + 1))

    target=""
    echo "$flags" | grep -q -- '--wasm' && target="--target wasm32-unknown-unknown"

    elapsed=$(( $(date +%s) - phase1_start ))
    echo -n "[$checked ${elapsed}s] $crate ... "

    if [ "$ws_dir" != "$REPO" ] && [ "$ws_dir" != "$REPO/crates/rvf" ]; then
        check_out=$(timeout 60 cargo check $target 2>&1) && check_ok=true || check_ok=false
    else
        check_out=$(timeout 60 cargo check -p "$crate" $target 2>&1) && check_ok=true || check_ok=false
    fi

    warn_count=$(echo "$check_out" | grep -c "warning\[" || true)

    if $check_ok; then
        echo "PASS (${warn_count}w)"
        # Record for phase 2 test, with flags
        echo "$crate $flags" >> "$CHECK_PASS_FILE"
    else
        loc=$(find_loc "$crate" "$ws_dir")
        err_count=$(echo "$check_out" | grep -c "^error\[" || true)
        if echo "$check_out" | grep -q "TIMEOUT\|^Terminated"; then
            check_result="TIMEOUT"
        else
            check_result="FAIL($err_count)"
        fi
        echo "$check_result"
        printf '%-40s | %-8s | %-16s | %-4s | %s\n' "$crate" "$check_result" "N/A" "$warn_count" "$loc" >> "$RESULTS"
        echo "--- $crate CHECK ERRORS ---" >> "$ERRORS"
        echo "$check_out" | grep "^error" | head -20 >> "$ERRORS"
        echo "" >> "$ERRORS"
        echo "$crate" >> "$DONE_FILE"
    fi

done < "$BATCH_FILE"

phase1_time=$(( $(date +%s) - phase1_start ))
pass_count=$(wc -l < "$CHECK_PASS_FILE")
echo ""
echo "===== PHASE 1 DONE: $checked checked in ${phase1_time}s, $pass_count passed ====="
echo ""

# ===== PHASE 2: cargo test on passing crates =====
echo "===== PHASE 2: cargo test (passing crates) ====="
phase2_start=$(date +%s)
tested=0

while IFS= read -r line; do
    [ -z "$line" ] && continue
    crate=$(echo "$line" | awk '{print $1}')
    flags=$(echo "$line" | cut -d' ' -f2-)

    grep -qxF "$crate" "$DONE_FILE" 2>/dev/null && continue

    ws_dir=$(get_ws_dir "$crate")
    cd "$ws_dir"
    tested=$((tested + 1))
    loc=$(find_loc "$crate" "$ws_dir")
    warn_count=0  # already counted in phase 1

    # Skip test for wasm and skip-test flagged
    if echo "$flags" | grep -q -- '--skip-test' || echo "$flags" | grep -q -- '--wasm'; then
        printf '%-40s | %-8s | %-16s | %-4s | %s\n' "$crate" "PASS" "N/A" "$warn_count" "$loc" >> "$RESULTS"
        echo "$crate" >> "$DONE_FILE"
        elapsed=$(( $(date +%s) - phase2_start ))
        echo "[$tested ${elapsed}s] $crate -> PASS / N/A (skipped)"
        continue
    fi

    elapsed=$(( $(date +%s) - phase2_start ))
    echo -n "[$tested ${elapsed}s] $crate test ... "

    if [ "$ws_dir" != "$REPO" ] && [ "$ws_dir" != "$REPO/crates/rvf" ]; then
        test_out=$(timeout 120 cargo test --lib -- 2>&1) && test_ok=true || test_ok=false
    else
        test_out=$(timeout 120 cargo test -p "$crate" --lib -- 2>&1) && test_ok=true || test_ok=false
    fi

    exit_code=$?
    test_result="N/A"

    if [ $exit_code -eq 124 ]; then
        test_result="TIMEOUT"
        echo "TIMEOUT"
    else
        summary=$(echo "$test_out" | grep "^test result:" | tail -1)
        if [ -n "$summary" ]; then
            passed=$(echo "$summary" | grep -oP '\d+ passed' | grep -oP '\d+')
            failed=$(echo "$summary" | grep -oP '\d+ failed' | grep -oP '\d+')
            ignored=$(echo "$summary" | grep -oP '\d+ ignored' | grep -oP '\d+')
            test_result="${passed:-0}p/${failed:-0}f/${ignored:-0}i"
            echo "$test_result"
        elif ! $test_ok; then
            terr_count=$(echo "$test_out" | grep -c "^error\[" || true)
            test_result="CFAIL($terr_count)"
            echo "$test_result"
            echo "--- $crate TEST ERRORS ---" >> "$ERRORS"
            echo "$test_out" | grep "^error" | head -10 >> "$ERRORS"
            echo "" >> "$ERRORS"
        else
            test_result="0p/0f/0i"
            echo "$test_result"
        fi
    fi

    printf '%-40s | %-8s | %-16s | %-4s | %s\n' "$crate" "PASS" "$test_result" "$warn_count" "$loc" >> "$RESULTS"
    echo "$crate" >> "$DONE_FILE"

done < "$CHECK_PASS_FILE"

total_time=$(( $(date +%s) - phase1_start ))
done_count=$(wc -l < "$DONE_FILE")
echo ""
echo "===== ALL DONE: $done_count total in ${total_time}s (check: ${phase1_time}s + test: $(( $(date +%s) - phase2_start ))s) ====="
echo "" >> "$RESULTS"
echo "Run completed: $(date) | ${total_time}s elapsed (2-phase)" >> "$RESULTS"
