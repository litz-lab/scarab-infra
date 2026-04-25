#!/usr/bin/env python3
"""Aggregate per-phase perf stat outputs from run_with_perf.sh into a table.

Reads <outdir>/perf_<phase>.csv and <outdir>/agent_phase_log_<phase>.txt for
each phase, computes IPC and MPKI per phase, prints a CSV-friendly table.

Usage:
    aggregate_perf.py <outdir>
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def parse_perf_csv(path: Path) -> dict[str, float]:
    """Parse `perf stat -x ','` output into {event_name: count}.

    Lines are: <count>,<unit>,<event>,<runtime_ns>,<percent>,...
    Comments (starting with #) and blank lines are skipped.
    """
    out: dict[str, float] = {}
    if not path.is_file():
        return out
    with path.open(encoding="utf-8", errors="replace") as f:
        for row in csv.reader(f):
            if not row or not row[0] or row[0].startswith("#"):
                continue
            if len(row) < 3:
                continue
            try:
                val = float(row[0])
            except ValueError:
                continue
            event = row[2].strip()
            if event:
                out[event] = val
    return out


def count_invocations(path: Path, phase: str) -> int:
    if not path.is_file():
        return 0
    n = 0
    with path.open() as f:
        for line in f:
            if line.strip() == f"begin {phase}":
                n += 1
    return n


PHASES = [
    "build_prompt", "tokenize", "api_build", "api_parse", "http_parse",
    "tool_dispatch", "tool_exec", "context_update", "retrieve",
    "file_grep", "patch_write", "multi_agent",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outdir", type=Path)
    args = ap.parse_args()

    if not args.outdir.is_dir():
        print(f"error: {args.outdir} not a directory", file=sys.stderr)
        return 1

    rows = []
    for phase in PHASES:
        perf = parse_perf_csv(args.outdir / f"perf_{phase}.csv")
        if not perf:
            continue
        invocations = count_invocations(args.outdir / f"agent_phase_log_{phase}.txt", phase)
        cycles = perf.get("cycles", 0.0)
        insts  = perf.get("instructions", 0.0)
        if insts == 0:
            continue
        ipc = insts / cycles if cycles else 0.0
        l1d = perf.get("L1-dcache-load-misses", 0.0)
        llc = perf.get("LLC-load-misses", 0.0)
        bm  = perf.get("branch-misses", 0.0)
        br  = perf.get("branches", 0.0)
        rows.append({
            "phase": phase,
            "invocations": invocations,
            "insts": int(insts),
            "cycles": int(cycles),
            "IPC": ipc,
            "L1d_MPKI": 1000.0 * l1d / insts,
            "LLC_MPKI": 1000.0 * llc / insts,
            "br_mispred_pct": 100.0 * bm / br if br else 0.0,
        })

    if not rows:
        print(f"error: no perf_*.csv files in {args.outdir}", file=sys.stderr)
        return 1

    # Print table sorted by total instructions (the heaviest phases first)
    rows.sort(key=lambda r: -r["insts"])
    total_insts = sum(r["insts"] for r in rows)
    print(f"# {len(rows)} phases, total measured insts = {total_insts:,}")
    print(f"{'phase':<16} {'inv':>4} {'insts':>14} {'cycles':>14} "
          f"{'IPC':>5} {'L1d_MPKI':>9} {'LLC_MPKI':>9} {'BrMis%':>7}  pct_insts")
    for r in rows:
        pct = 100.0 * r["insts"] / total_insts
        print(f"{r['phase']:<16} {r['invocations']:>4} {r['insts']:>14,} "
              f"{r['cycles']:>14,} {r['IPC']:>5.3f} "
              f"{r['L1d_MPKI']:>9.2f} {r['LLC_MPKI']:>9.2f} "
              f"{r['br_mispred_pct']:>7.3f}  {pct:>5.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
