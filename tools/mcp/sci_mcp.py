#!/usr/bin/env python3
"""Minimal stdio MCP server exposing scarab-infra's ./sci as tools.

No third-party deps: speaks MCP JSON-RPC over stdin/stdout directly.
"""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(os.environ.get("SCI_REPO") or Path(__file__).resolve().parents[2]).resolve()
SCI = REPO / "sci"
PROTOCOL = "2024-11-05"

# name -> (sci flag, help, needs descriptor, extra timeout)
TOOLS = [
    ("sci_status",        "--status",        "Run/node status for a sweep. Call this right after a launch and on every check-in.", True,  600),
    ("sci_sim",           "--sim",           "Launch the simulations defined in json/<descriptor>.json. A configuration whose binary is named scarab_<githash> is checked out and built automatically when it is not already cached, so pin the hashes you want in the descriptor rather than building them yourself.", True,  1800),
    ("sci_build_scarab",  "--build-scarab",  "Build scarab from the current working tree for json/<descriptor>.json. Never invoke the compiler directly, and never loop this over commits: to run several commits, name each one scarab_<githash> in the descriptor and let sci_sim build them.", True,  3600),
    ("sci_collect_stats", "--collect-stats", "Collect stats into collected_stats.csv. Never parse stats.out by hand.",             True,  1800),
    ("sci_visualize",     "--visualize",     "Plot IPC/speedup from collected stats.",                                             True,  900),
    ("sci_perf_analyze",  "--perf-analyze",  "Analyze IPC drift from collected stats.",                                            True,  900),
    ("sci_kill",          "--kill",          "Kill active simulations for a descriptor.",                                          True,  600),
    ("sci_list",          "--list",          "List workload group names.",                                                         False, 300),
]

DESC_SCHEMA = {
    "type": "object",
    "properties": {
        "descriptor": {"type": "string", "description": "Descriptor name, i.e. json/<name>.json without the path or extension."},
        "tail": {"type": "integer", "description": "Return only the last N lines of output (default 200, 0 = all)."},
    },
    "required": ["descriptor"],
}
NOARG_SCHEMA = {"type": "object", "properties": {"tail": {"type": "integer"}}, "required": []}


def tool_defs():
    out = []
    for name, flag, desc, needs_desc, _ in TOOLS:
        out.append({
            "name": name,
            "description": f"{desc} (runs `./sci {flag}`)",
            "inputSchema": DESC_SCHEMA if needs_desc else NOARG_SCHEMA,
        })
    return out


def run_tool(name, args):
    entry = next((t for t in TOOLS if t[0] == name), None)
    if entry is None:
        return True, f"unknown tool: {name}"
    _, flag, _, needs_desc, timeout = entry

    cmd = [str(SCI), flag]
    if needs_desc:
        d = (args or {}).get("descriptor")
        if not d:
            return True, "missing required argument: descriptor"
        d = str(d).strip()
        if d.endswith(".json"):
            d = d[:-5]
        d = Path(d).name
        if not (REPO / "json" / f"{d}.json").exists():
            avail = sorted(p.stem for p in (REPO / "json").glob("*.json"))
            return True, f"no descriptor json/{d}.json. available: {', '.join(avail[:40])}"
        cmd.append(d)

    try:
        p = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return True, f"`{shlex.join(cmd)}` timed out after {timeout}s"

    out = (p.stdout or "") + (("\n[stderr]\n" + p.stderr) if p.stderr.strip() else "")
    n = (args or {}).get("tail", 200)
    if isinstance(n, int) and n > 0:
        lines = out.splitlines()
        if len(lines) > n:
            out = f"[... {len(lines) - n} earlier lines omitted ...]\n" + "\n".join(lines[-n:])
    return p.returncode != 0, out or "(no output)"


def send(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue

        method, rid = req.get("method"), req.get("id")
        if method == "initialize":
            send({"jsonrpc": "2.0", "id": rid, "result": {
                "protocolVersion": PROTOCOL,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "scarab-infra", "version": "1.0.0"},
            }})
        elif method == "tools/list":
            send({"jsonrpc": "2.0", "id": rid, "result": {"tools": tool_defs()}})
        elif method == "tools/call":
            params = req.get("params") or {}
            is_err, text = run_tool(params.get("name"), params.get("arguments") or {})
            send({"jsonrpc": "2.0", "id": rid, "result": {
                "content": [{"type": "text", "text": text}], "isError": is_err}})
        elif method in ("notifications/initialized", "notifications/cancelled"):
            continue
        elif rid is not None:
            send({"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"method not found: {method}"}})


if __name__ == "__main__":
    main()
