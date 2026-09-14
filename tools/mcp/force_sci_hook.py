#!/usr/bin/env python3
"""PreToolUse hook: deny Bash that reimplements ./sci, naming the tool to use instead."""

import json
import re
import sys

SCRIPTED = r"(?:python3?|awk|perl|ruby)\b"
# Paths that are sci's output, not yours to parse.
STATS = r"(?:\bstats\.out\b|/simulations/|\bcollected_stats\.csv\b)"

RULES = [
    # (regex, replacement tool). Aggregation over sim output is sci's job; reading a
    # single file with cat/grep while debugging is not blocked.
    (r"\bsqueue\b(?=.*\bexp_)",                        "sci_status"),
    (r"\bsacct\b(?=.*\bexp_)",                         "sci_status"),
    (SCRIPTED + r".*" + STATS,                          "sci_collect_stats"),
    (STATS + r".*" + SCRIPTED,                          "sci_collect_stats"),
    (r"\*[^\s]*\b(?:stats\.out)\b|/simulations/[^\s]*\*", "sci_collect_stats"),
    (r"\bfor\b.*\bdo\b.*" + STATS,                     "sci_collect_stats"),
    (r"\bmake\b[^|;]*\bscarab\b",                      "sci_build_scarab"),
    # Command position, and only a real binary: naming one (md5sum, ls, stat) is not
    # running it, and a bare "scarab" in prose or a markdown table is not either.
    (r"(?:^|[|;&]\s*|\bsudo\s+|\bsrun\s+)\s*(?:[^\s|;&='\"]*/)?(?:scarab_[0-9a-f]{7}[^\s]*\.opt|\./scarab)\b", "sci_sim"),
]
# `./sci ...` itself is always fine, and so is anything that merely mentions sci.
ALLOW = re.compile(r"(?:^|[|;&]\s*|\s)\./sci\b")


def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        sys.exit(0)

    if payload.get("tool_name") != "Bash":
        sys.exit(0)
    cmd = (payload.get("tool_input") or {}).get("command", "")
    if not cmd or ALLOW.search(cmd):
        sys.exit(0)

    for pat, tool in RULES:
        if re.search(pat, cmd, re.IGNORECASE):
            print(json.dumps({"hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": (
                    f"Blocked: this reimplements scarab-infra. Use the MCP tool `{tool}` "
                    f"(scarab-infra server) instead of hand-rolling it in Bash. "
                    f"If the tool genuinely cannot express what you need, say so to the user "
                    f"and ask before working around it."),
            }}))
            sys.exit(0)
    sys.exit(0)


if __name__ == "__main__":
    main()
