#!/bin/bash
# code-py requests_5414 replay for tracing its tool processes (agentcpu image).
#
# The same replay as the benchmark's bundles/code-py/replay_local.sh (SWE-agent
# run-replay with SWE-ReX's local deployment, tools as child processes against
# /testbed), with two differences that tracing the tools needs:
#   * the tool command timeouts are raised (CODEPY_TOOL_TIMEOUT seconds per
#     command, default 1800), because under DynamoRIO a pytest run is about 30
#     times slower and SWE-agent's 30 s default would cut the tool result;
#   * the content hash of the produced trajectory is printed at the end, so
#     replay_and_check.sh can compare it with the untraced replay.
# Used with SCARAB_TRACE_SHELL_VAR=PATH: the orchestrator runs natively and the
# `bash` that SWE-ReX starts (`/usr/bin/env bash`) is the drrun wrapper.
#
# Usage: codepy_tools_replay.sh [output_dir]
set -uo pipefail
B=${AGENTCPU:-/tmp_home/agentcpu-bench}/bundles/code-py
OUT=${1:-/tmp/codepy_tools_out}
cd "$B"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 LITELLM_LOCAL_MODEL_COST_MAP=True
export PATH="$PWD/testbed_venv/bin:$PATH"
mkdir -p "$OUT"
./rebase_traj.py trajectories/requests_5414/psf__requests-5414.traj "$OUT/input.traj"
# run-replay merges --update_config at the top level, so the whole agent
# section of the recorded config is written back with only the timeouts changed.
venv/bin/python - "$OUT/input.traj" "$OUT/tool_timeouts.yaml" "${CODEPY_TOOL_TIMEOUT:-1800}" <<'PY'
import json, sys, yaml
t = json.load(open(sys.argv[1]))
c = t["replay_config"]
c = json.loads(c) if isinstance(c, str) else c
agent = c["agent"]
tools = agent.setdefault("tools", {})
n = int(sys.argv[3])
tools["execution_timeout"] = n
tools["total_execution_timeout"] = max(20 * n, int(tools.get("total_execution_timeout", 1800)))
yaml.safe_dump({"agent": agent}, open(sys.argv[2], "w"))
PY
venv/bin/python -m sweagent.run.run run-replay --traj_path "$OUT/input.traj" --deployment.type local \
  --update_config "$OUT/tool_timeouts.yaml" --output_dir "$OUT"
venv/bin/python content_hash.py "$OUT"/*/*.traj
# Under DynamoRIO a pytest run takes minutes, and pytest then appends the
# duration as "(H:MM:SS)", which content_hash.py does not normalize; object
# addresses in pytest's failure reprs depend on the address layout (ASLR is off
# in the trace container). This hash drops both; per-step hashes show where a
# replay differs.
venv/bin/python - "$OUT"/*/*.traj <<'PY'
import hashlib, json, re, sys
def norm(s):
    s = re.sub(r"\b\d+\.\d+s\b", "Xs", s)
    s = re.sub(r" \(\d+:\d\d:\d\d\)", "", s)
    s = re.sub(r"\b0x[0-9a-f]{6,}\b", "0x<addr>", s)   # object addresses in pytest reprs (layout-dependent)
    s = re.sub(r"\b[0-9a-f]{40}\b", "<sha>", s)
    s = re.sub(r"\[\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+\]", "[ts]", s)
    s = re.sub(r"\[\d\d/[A-Z][a-z]{2}/\d{4} \d\d:\d\d:\d\d\]", "[ts]", s)
    return s
t = json.load(open(sys.argv[1]))
m = hashlib.sha256()
steps = []
for step in t.get("trajectory", []):
    x = (str(step.get("action", "")) + "|" + norm(str(step.get("observation", "")))).encode()
    m.update(x)
    steps.append(hashlib.sha256(x).hexdigest()[:8])
m.update(str(t.get("info", {}).get("submission", "")).encode())
print(f"step_hashes={','.join(steps)}")
print(f"timing_normalized_hash={m.hexdigest()[:16]}")
PY
if [ -n "${REPLAY_KEEP_DIR:-}" ]; then mkdir -p "$REPLAY_KEEP_DIR" && cp -r "$OUT"/. "$REPLAY_KEEP_DIR"/; fi
