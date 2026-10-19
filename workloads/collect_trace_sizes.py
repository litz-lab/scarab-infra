#!/usr/bin/env python3
"""Record trace file sizes in workloads_db.json from the local traces directory.

memtrace: each simpoint's ``size_bytes`` is <traces_dir>/<suite>/<subsuite>/<workload>/traces/simp/<cluster_id>.zip.
pt: ``simulation.pt.size_bytes`` is <traces_dir>/<suite>/<subsuite>/<workload>/traces/pt/trace.gz.
"""

import argparse
import json
import os
import sys
from typing import Iterable, Optional, Tuple


def _file_size(path: str) -> Optional[int]:
    try:
        return os.path.getsize(path)
    except OSError:
        return None


def update_trace_sizes(workloads_db: dict, traces_dir: str,
                       only: Optional[Iterable[Tuple[str, str, str]]] = None) -> int:
    """Set sizes for every workload, or only the (suite, subsuite, workload) keys in ``only``.

    Returns the number of size fields changed. A trace missing on disk leaves its field as is.
    """
    only = set(only) if only is not None else None
    changed = 0
    for suite, subsuites in workloads_db.items():
        if not isinstance(subsuites, dict):
            continue
        for subsuite, workloads in subsuites.items():
            if not isinstance(workloads, dict):
                continue
            for workload, entry in workloads.items():
                if not isinstance(entry, dict) or (only is not None and (suite, subsuite, workload) not in only):
                    continue
                traces = os.path.join(traces_dir, suite, subsuite, workload, "traces")
                simulation = entry.get("simulation") or {}

                if "memtrace" in simulation:
                    for simpoint in entry.get("simpoints") or []:
                        size = _file_size(os.path.join(traces, "simp", f"{simpoint.get('cluster_id')}.zip"))
                        if size is not None and simpoint.get("size_bytes") != size:
                            simpoint["size_bytes"] = size
                            changed += 1

                pt = simulation.get("pt")
                if isinstance(pt, dict):
                    size = _file_size(os.path.join(traces, "pt", "trace.gz"))
                    if size is not None and pt.get("size_bytes") != size:
                        pt["size_bytes"] = size
                        changed += 1
    return changed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--traces-dir", required=True, help="Root of the local traces, e.g. /soe/hlitz/lab/traces")
    parser.add_argument("--workloads", nargs="+",
                        default=["workloads/workloads_db.json", "workloads/workloads_top_simp.json"],
                        help="workloads DB JSON files to update in place")
    args = parser.parse_args(argv)

    for path in args.workloads:
        with open(path) as f:
            data = json.load(f)
        changed = update_trace_sizes(data, args.traces_dir)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, separators=(",", ":"))
            f.write("\n")
        print(f"{path}: {changed} size field(s) updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
