#!/usr/bin/env python3
"""PreToolUse hook: deny Bash that reimplements ./sci, naming the tool to use instead."""

import json
import re
import sys

SCRIPTED = r"(?:python3?|awk|perl|ruby)\b"
# Slurm job listings. How far along a sweep is, is sci's question, not squeue's.
QUEUE = r"\b(?:squeue|sacct)\b"
# Raw per-simulation output: collect it into a CSV, don't parse it by hand.
RAW_STATS = r"(?:\bstats\.out\b|/simulations/|\.stat\.\d+\.(?:out|csv)\b)"
# Already-collected results: aggregate / SimPoint-weight / plot with visualize, don't re-parse.
COLLECTED = r"(?:\bcollected_stats\.csv\b|\baggregates\.json\b)"

RULES = [
    # (regex, replacement tool). Aggregation over sim output is sci's job; reading a
    # single file with cat/grep while debugging is not blocked. Matched with re.DOTALL so
    # multi-line commands (heredocs, `cd ...; python3 ...`) cannot split the tokens across
    # lines to slip past the `.*`.
    # Sweep progress, however it is phrased. Naming the experiment is only one way to
    # ask: counting your own jobs, or polling in a loop, asks the same thing and used to
    # slip through because neither mentions exp_. A per-user count is the worst of them,
    # since it also counts held stragglers from other experiments and so reports a
    # finished sweep as stuck. Asking about the cluster -- which nodes are up, who else
    # is queued -- is a different question and stays allowed, as does sinfo.
    (QUEUE + r"(?=.*\bexp_)",                          "sci_status"),
    (QUEUE + r"[^\n]*\s-{1,2}u(?:ser)?[= ]",           "sci_status"),
    (QUEUE + r"[\s\S]*\|[\s\S]*\b(?:wc|uniq)\b",       "sci_status"),
    (QUEUE + r"[\s\S]*\|[\s\S]*\bgrep\b[^|\n]*-[a-z]*c", "sci_status"),
    (r"\b(?:for|while)\b[\s\S]*\bdo\b[\s\S]*" + QUEUE, "sci_status"),
    # already-collected results -> visualize (SimPoint-weighted tables/plots); checked
    # before RAW_STATS since collected_stats.csv also lives under /simulations/.
    (SCRIPTED + r".*" + COLLECTED,                     "sci_visualize"),
    (COLLECTED + r".*" + SCRIPTED,                     "sci_visualize"),
    (SCRIPTED + r".*" + RAW_STATS,                     "sci_collect_stats"),
    (RAW_STATS + r".*" + SCRIPTED,                     "sci_collect_stats"),
    (r"\*[^\s]*\b(?:stats\.out)\b|/simulations/[^\s]*\*", "sci_collect_stats"),
    (r"\bfor\b.*\bdo\b.*(?:" + RAW_STATS + r"|" + COLLECTED + r")", "sci_collect_stats"),
    (r"\bmake\b[^|;]*\bscarab\b",                      "sci_build_scarab"),
    # Command position, and only a real binary: naming one (md5sum, ls, stat) is not
    # running it, and a bare "scarab" in prose or a markdown table is not either.
    (r"(?:^|[|;&]\s*|\bsudo\s+|\bsrun\s+)\s*(?:[^\s|;&='\"]*/)?(?:scarab_[0-9a-f]{7}[^\s]*\.opt|\./scarab)\b", "sci_sim"),
]
# `./sci ...` itself is always fine, and so is anything that merely mentions sci.
ALLOW = re.compile(r"(?:^|[|;&]\s*|\s)\./sci\b")

# Checked before ALLOW: these call sci but drive it by hand, so the allow-list would
# wave them through. A descriptor that names scarab_<githash> binaries already gets
# them checked out and built by --sim.
CHECKOUT = r"\bgit\b[^\n;|&]*\bcheckout\b"
BUILD = r"--build-scarab\b"
DENY_FIRST = [
    (CHECKOUT + r"[\s\S]*" + BUILD, "sci_sim"),
    (BUILD + r"[\s\S]*" + CHECKOUT, "sci_sim"),
]
DENY_FIRST_REASON = (
    "Blocked: do not drive a scarab build with your own checkout loop. Use the MCP "
    "tool `{tool}` (scarab-infra server): name each binary scarab_<githash> in the "
    "descriptor's configurations and launch once. sci checks out and builds every "
    "hash that is not cached, restoring your branch and working tree afterwards.")


def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        sys.exit(0)

    if payload.get("tool_name") != "Bash":
        sys.exit(0)
    cmd = (payload.get("tool_input") or {}).get("command", "")
    if not cmd:
        sys.exit(0)

    for pat, tool in DENY_FIRST:
        if re.search(pat, cmd, re.IGNORECASE | re.DOTALL):
            deny(tool, DENY_FIRST_REASON.format(tool=tool))

    if ALLOW.search(cmd):
        sys.exit(0)

    for pat, tool in RULES:
        if re.search(pat, cmd, re.IGNORECASE | re.DOTALL):
            deny(tool, (
                f"Blocked: this reimplements scarab-infra. Use the MCP tool `{tool}` "
                f"(scarab-infra server) instead of hand-rolling it in Bash. "
                f"If the tool genuinely cannot express what you need, say so to the user "
                f"and ask before working around it."))
    sys.exit(0)


def deny(tool, reason):
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": reason,
    }}))
    sys.exit(0)


if __name__ == "__main__":
    main()
