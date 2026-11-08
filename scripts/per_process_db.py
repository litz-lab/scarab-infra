#!/usr/bin/env python3
"""Turn a per-process trace manifest into workloads_db.json entries.

run_simpoint_trace.py with SCARAB_PER_PROCESS=1 converts every process of a
capture that followed child processes and writes per_process_manifest.json in
the workload home (see the comment block above per_process_trace there). This
module makes one workload per process from it:

  <suite>/<subsuite>/<prefix>_<exe>_<pid>
    simulation.memtrace: trace_then_cluster, allbench_traces image,
      segment_size 10 M and warmup 50 M for a process that went through SimPoint;
      segment_size = its instruction count rounded up to whole 10 M chunks and
      warmup 0 for a short process simulated whole (one SimPoint, segment 0,
      weight 1, so run_memtrace_single_simpoint.sh runs it from the first
      instruction to the end of the trace)
    simpoints: phase "tools", tool_type = the process kind (python, bash, git, other)
    per_process: exe, pid, kind, instruction count, share of the tool-side
      instructions of the capture, and the session prefix, so results can be
      weighted by instruction share and grouped by kind. A manifest made with
      SCARAB_PER_PROCESS_THREADS=all has one workload per thread; those also
      carry tid, largest_thread and the thread's share of its process's raw
      bytes (shard_fraction). Combining threads by instruction share adds up
      the cycles of threads simulated alone: a CPU-time account that ignores
      contention between threads of one process.

finish_trace (scripts/utilities.py) calls register_manifest when it finds a
manifest. The command line does the same for a manifest made outside sci:

  python -m scripts.per_process_db --manifest <home>/per_process_manifest.json \
      --suite agentcpu --subsuite code-node-tools                 # dry run: print
  ... --traces-dir /home/$USER/traces --copy                      # copy the traces
  ... --db workloads/workloads_db.json --write                    # back up, then merge

Writing refuses to replace an existing workload unless --replace is given, and
always copies the database to <db>.bak.<time> first.
"""

import argparse
import json
import os
import shutil
import sys
import time

MANIFEST_NAME = "per_process_manifest.json"


def load_manifest(path):
    with open(path) as f:
        return json.load(f)


def ready_processes(manifest):
    return [p for p in manifest["processes"] if p.get("status") == "ready"]


def build_entries(manifest, manifest_path=None):
    """{workload: entry} for every process whose SimPoints are ready."""
    entries = {}
    for p in ready_processes(manifest):
        simpoints = [dict(sp, phase="tools", tool_type=p["kind"]) for sp in p["simpoints"]]
        entries[p["workload"]] = {
            "trace": {
                "dynamorio_args": manifest.get("dynamorio_args"),
                "clustering_k": manifest.get("clustering_k"),
            },
            "simulation": {
                "prioritized_mode": "memtrace",
                "memtrace": {
                    "image_name": "allbench_traces",
                    "segment_size": p["segment_size"],
                    "warmup": p["warmup"],
                    "whole_trace_file": os.path.basename(p["trace"]),
                    "trace_type": "trace_then_cluster",
                },
            },
            "simpoints": simpoints,
            "phase_shares": {"tools": round(sum(sp["weight"] for sp in simpoints), 6)},
            "per_process": {
                "session": manifest["prefix"],
                "source_dir": p["dir"],
                "exe": p["exe"],
                "pid": p["pid"],
                "kind": p["kind"],
                "instrs": p["instrs"],
                "share_of_tools": p["share"],
                "mode": p["mode"],
                "threads": p["threads"],
                "main_shard_fraction": p["main_shard_fraction"],
                "manifest": manifest_path,
            },
        }
        if "shard" in p:
            entries[p["workload"]]["per_process"].update({
                "tid": p["tid"],
                "largest_thread": p["largest_thread"],
                "shard_fraction": p["shard_fraction"],
            })
    return entries


def _link_or_copy(src, dst):
    """Hard link when src and dst are on one file system (a capture of several
    hundred GB of converted traces is then not stored twice), else copy."""
    if os.path.isdir(dst):
        dst = os.path.join(dst, os.path.basename(src))
    if os.path.exists(dst):
        if os.path.samefile(src, dst):
            return
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def copy_traces(manifest, manifest_path, traces_dir, suite, subsuite):
    """Copy (hard link where possible) each ready process into
    <traces_dir>/<suite>/<subsuite>/<workload>/traces/{whole,simp}."""
    home_root = os.path.dirname(os.path.abspath(manifest_path))
    for p in ready_processes(manifest):
        home = os.path.join(home_root, p["home"])
        dest = os.path.join(traces_dir, suite, subsuite, p["workload"], "traces")
        os.makedirs(os.path.join(dest, "whole"), exist_ok=True)
        os.makedirs(os.path.join(dest, "simp"), exist_ok=True)
        _link_or_copy(os.path.join(home_root, p["trace"]), os.path.join(dest, "whole"))
        simp = os.path.join(home, "traces_simp")
        for sp in p["simpoints"]:
            name = f"{sp['cluster_id']}.zip"
            _link_or_copy(os.path.join(simp, name), os.path.join(dest, "simp", name))
        os.system(f"chmod a+w -R {os.path.join(traces_dir, suite, subsuite, p['workload'])}")


def register_manifest(manifest_path, workload_db_data, suite, subsuite, traces_dir=None, replace=True):
    """Merge the manifest's workloads into workload_db_data (in place); copy traces if traces_dir."""
    manifest = load_manifest(manifest_path)
    entries = build_entries(manifest, os.path.abspath(manifest_path))
    not_ready = [p["workload"] for p in manifest["processes"] if p.get("status") != "ready"]
    if not_ready:
        print(f"per-process: {len(not_ready)} process(es) not ready and not registered: {not_ready}")
    if traces_dir:
        copy_traces(manifest, manifest_path, traces_dir, suite, subsuite)
    target = workload_db_data.setdefault(suite, {}).setdefault(subsuite, {})
    clash = [w for w in entries if w in target]
    if clash and not replace:
        raise RuntimeError(f"workloads already in {suite}/{subsuite}: {clash} (use --replace)")
    for w, e in entries.items():
        preserved = {k: v for k, v in target.get(w, {}).items() if k not in e}
        target[w] = e | preserved
    return entries


def summary(manifest):
    lines = []
    total = manifest.get("tool_instrs_total", 0)
    units = manifest["processes"]
    threads = manifest.get("threads_mode") == "all"
    lines.append(f"session {manifest['prefix']}: {len({p['dir'] for p in units})} processes"
                 + (f" ({len(units)} thread shards, {sum(1 for p in units if p.get('status') == 'empty')} empty)"
                    if threads else "")
                 + f", {len(manifest['skipped'])} dirs skipped, {total} instructions")
    for kind, k in sorted(manifest.get("kinds", {}).items(), key=lambda kv: -kv[1]["instrs"]):
        lines.append(f"  {kind:12s} {k['processes']:4d} processes"
                     + (f" {k['threads']:4d} threads" if threads else "")
                     + f" {k['instrs']:>14d} instr  share {k['share']:.4f}"
                     f"  ({k['simpoint_processes']} SimPoint, {k['whole_processes']} whole)")
    return "\n".join(lines)


def write_db(db_path, data):
    """Back up, then write in the format the checked-in database uses."""
    backup = f"{db_path}.bak.{time.strftime('%Y%m%d-%H%M%S')}"
    shutil.copy2(db_path, backup)
    tmp = db_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, separators=(",", ":"))
        f.write("\n")
    os.replace(tmp, db_path)
    return backup


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--subsuite", required=True, help="e.g. code-node-tools, code-py-tools")
    ap.add_argument("--db", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                 "workloads", "workloads_db.json"))
    ap.add_argument("--traces-dir", default=None)
    ap.add_argument("--copy", action="store_true", help="copy the traces into --traces-dir")
    ap.add_argument("--write", action="store_true", help="back up and update --db (default: print only)")
    ap.add_argument("--replace", action="store_true", help="allow replacing workloads already in the subsuite")
    ap.add_argument("--out", default=None, help="also write the proposed entries to this JSON file")
    args = ap.parse_args(argv)

    manifest = load_manifest(args.manifest)
    print(summary(manifest))
    entries = build_entries(manifest, os.path.abspath(args.manifest))
    print(f"{len(entries)} workload entries for {args.suite}/{args.subsuite}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({args.suite: {args.subsuite: entries}}, f, indent=2, separators=(",", ":"))
            f.write("\n")
        print(f"proposed entries written to {args.out}")
    if args.copy:
        if not args.traces_dir:
            ap.error("--copy needs --traces-dir")
        copy_traces(manifest, args.manifest, args.traces_dir, args.suite, args.subsuite)
        print(f"traces copied under {args.traces_dir}/{args.suite}/{args.subsuite}")
    if args.write:
        with open(args.db) as f:
            data = json.load(f)
        register_manifest(args.manifest, data, args.suite, args.subsuite, replace=args.replace)
        backup = write_db(args.db, data)
        print(f"{args.db} updated (backup {backup})")
    elif not args.out:
        json.dump(entries, sys.stdout, indent=1)
        print()


if __name__ == "__main__":
    main()
