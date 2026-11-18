#!/bin/bash
# Run a replay command and fail unless its output contains every given pattern.
# Used as a trace binary_cmd, so that a capture whose replay drifted from the
# recording (a missed request, a different tool result) stops the trace job
# instead of being registered.
#
# Usage: replay_and_check.sh <pattern>... -- <command> [args...]
# Example (code-node): replay_and_check.sh replay_misses=0 match=yes -- \
#   env CLAUDE_BUILD=2.1.112 /bin/bash <task>/replay.sh /tmp/cn_tools_out
patterns=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do patterns+=("$1"); shift; done
[ "$1" = "--" ] || { echo "usage: $0 <pattern>... -- <command>" >&2; exit 2; }
shift
log=$(mktemp)
"$@" 2>&1 | tee "$log"
rc=${PIPESTATUS[0]}
status=0
for p in "${patterns[@]}"; do
  if ! grep -qF -- "$p" "$log"; then
    echo "replay_and_check: '$p' not found in the replay output" >&2
    status=1
  fi
done
rm -f "$log"
[ "$rc" -ne 0 ] && { echo "replay_and_check: command exited $rc" >&2; exit "$rc"; }
exit $status
