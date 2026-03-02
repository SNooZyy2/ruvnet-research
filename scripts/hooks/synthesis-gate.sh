#!/usr/bin/env bash
# synthesis-gate.sh — PostToolUse hook for Task completions
# Tracks reader agent completions and emits [SYNTHESIS REQUIRED] when all done.
#
# Reads/updates manifest at /tmp/ruvnet-research-session.json
# Expects TOOL_RESULT in environment (piped by Claude Code hook system)

set -euo pipefail

MANIFEST="/tmp/ruvnet-research-session.json"

# Exit silently if no manifest (not a research session)
[[ -f "$MANIFEST" ]] || exit 0

# Read the tool result from stdin (Claude Code pipes it)
TOOL_RESULT="$(cat)"

# Check for reader-completion markers
if ! echo "$TOOL_RESULT" | grep -qE '(Depth Achieved: DEEP|Realness:|DEEP read complete|findings inserted)'; then
    exit 0
fi

# Parse manifest with portable tools (no jq dependency)
completed=$(python3 -c "
import json, sys
with open('$MANIFEST') as f:
    m = json.load(f)
if m.get('synthesis_fired', False):
    sys.exit(1)
m['completed_readers'] = m.get('completed_readers', 0) + 1
print(json.dumps(m))
" 2>/dev/null) || exit 0

# Write updated manifest
echo "$completed" > "$MANIFEST"

# Check if all readers are done
python3 -c "
import json, sys
with open('$MANIFEST') as f:
    m = json.load(f)
if m['completed_readers'] >= m['expected_readers'] and not m.get('synthesis_fired', False):
    m['synthesis_fired'] = True
    with open('$MANIFEST', 'w') as f:
        json.dump(m, f)
    domains = ', '.join(m.get('domains', []))
    sid = m.get('session_id', '?')
    n = m['expected_readers']
    print(f'[SYNTHESIS REQUIRED] All {n} readers complete. Domains: {domains}. Session: {sid}')
" 2>/dev/null || true
