#!/usr/bin/env python3
"""Cases for force_sci_hook. Vectors live here, not on a command line the hook reads."""

import json
import subprocess
import sys
from pathlib import Path

HOOK = Path(__file__).with_name("force_sci_hook.py")

# (command, expected tool or None for allow)
CASES = [
    # status
    ('squeue -u me -o "%j" | grep exp_sw621b | wc -l', "sci_status"),
    ("sacct -n --format=JobName | grep exp_sw620", "sci_status"),
    # stat aggregation
    ("grep IPC /home/me/simulations/exp_sw620/*/stats.out", "sci_collect_stats"),
    ("for f in /home/me/simulations/exp/*/*/stats.out; do awk '/IPC/{print $2}' $f; done", "sci_collect_stats"),
    ("python3 /tmp/myparse.py /home/me/simulations/exp_sw619/base_stock", "sci_collect_stats"),
    ('python3 -c "import csv; csv.reader(open(\'collected_stats.csv\'))"', "sci_collect_stats"),
    # build
    ("make -j32 scarab", "sci_build_scarab"),
    # running the simulator
    ("/home/me/simulations/scarab_7827a3c_1.opt --param_file x", "sci_sim"),
    ("srun /home/me/sim/scarab_7827a3c_1.opt", "sci_sim"),
    ("./scarab --param_file PARAMS.in", "sci_sim"),
    # allowed: sci itself
    ("./sci --status sw621b", None),
    ("cd /home/me/git/scarab-infra && ./sci --collect-stats sw620 && ./sci --visualize sw620", None),
    # allowed: inspecting one file while debugging
    ("grep -n PREF_DRAMQ_STALL /home/me/simulations/exp/feat/x/stats.out", None),
    # allowed: naming a binary is not running it
    ("md5sum /home/me/simulations/scarab_7827a3c_1.opt", None),
    ("ls -la /home/me/simulations/scarab_bc9352d_0.opt", None),
    ("stat -c%s /home/me/sim/scarab_b777a1a_1.opt", None),
    # allowed: a shell assignment is not a command invocation
    ("S=/home/me/builds/scarab_bc9352d_0.opt; md5sum $S", None),
    ("ls -la scarab_bc9352d_*.opt", None),
    # driving a build with your own checkout loop: --sim builds missing hashes itself
    ("for c in aaa1111 bbb2222; do git -C /r checkout -q $c; ./sci --build-scarab b; done", "sci_sim"),
    ("git checkout 8672066 && ./sci --build-scarab build_bisect", "sci_sim"),
    ("./sci --build-scarab build_bisect; git checkout mshr-on-miss", "sci_sim"),
    # allowed: building the tree you are on, then launching
    ("./sci --build-scarab build_x && ./sci --sim sw720", None),
    ("git checkout mshr-on-miss && ./sci --sim sw720", None),
    # allowed: unrelated work
    ("git log --oneline -5", None),
    ("gh pr view 621 --comments", None),
    ("cd /home/me/git/scarab-infra && git diff --stat", None),
]


def verdict(command):
    payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": command}})
    out = subprocess.run(
        [sys.executable, str(HOOK)], input=payload, capture_output=True, text=True
    ).stdout.strip()
    if not out:
        return None
    reason = json.loads(out)["hookSpecificOutput"]["permissionDecisionReason"]
    return reason.split("Use the MCP tool `")[1].split("`")[0]


def main():
    failures = 0
    for command, expected in CASES:
        got = verdict(command)
        ok = got == expected
        failures += not ok
        label = "ok  " if ok else "FAIL"
        want = expected or "allow"
        print(f"[{label}] {want:<18} {'' if ok else f'(got {got or 'allow'}) '}{command[:64]}")
    print(f"\n{len(CASES) - failures}/{len(CASES)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
