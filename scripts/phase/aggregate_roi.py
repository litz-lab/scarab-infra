#!/usr/bin/env python3
"""Aggregate Scarab per-ROI stat CSVs into per-phase tables.

Scarab emits one numbered `*.roi.<N>.csv` per begin/end marker pair it sees
during simulation. The AgentCPU workload (via agent_markers.roi) writes a
sidecar phase log so we can align each N with a phase name.

Usage:
    aggregate_roi.py <sim_output_dir> [--phase-log PATH] [--stats-file FILE]

<sim_output_dir>   Directory containing Scarab output (e.g., a simpoint run dir
                   with core.stat.0.roi.*.csv, inst.stat.0.roi.*.csv, etc.)
--phase-log        Path to agent_markers sidecar log (default:
                   <sim_output_dir>/agent_phase_log.txt; falls back to
                   /tmp/agent_phase_log.txt).
--stats-file       Basename of the stat file family to aggregate (default
                   core.stat.0). Try inst.stat.0 for the topdown breakdown,
                   memory.stat.0 for MPKI.

Output (stdout):
    One row per unique phase name with weighted-mean IPC + total insts/cycles
    + count of ROI occurrences contributing to that phase.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path


ROI_FILE_RE = re.compile(r"^(?P<base>.+?)\.roi\.(?P<n>\d+)\.csv$")


def read_phase_log(path: Path) -> list[str]:
    """Return phase names in begin order (one entry per begin event)."""
    if not path.is_file():
        return []
    names: list[str] = []
    with path.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0] == "begin":
                names.append(parts[1])
    return names


def read_stat_csv(path: Path) -> dict[str, float]:
    """Parse a Scarab stat CSV into {name: value}.

    Scarab stat CSVs are produced by statistics.c's dump_stats; column layout
    is forgiving — we scan for lines that look like "<name>,<value>" or whitespace
    variants and coerce values to float when possible.
    """
    out: dict[str, float] = {}
    with path.open(encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            name = row[0].strip()
            val_str = row[1].strip()
            if not name or not val_str:
                continue
            try:
                out[name] = float(val_str.split()[0])
            except (ValueError, IndexError):
                continue
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sim_dir", type=Path,
                    help="Scarab simulation output directory")
    ap.add_argument("--phase-log", type=Path, default=None,
                    help="agent_markers sidecar log path")
    ap.add_argument("--stats-file", default="core.stat.0",
                    help="Stat file basename to aggregate (default core.stat.0)")
    args = ap.parse_args()

    if not args.sim_dir.is_dir():
        print(f"error: {args.sim_dir} is not a directory", file=sys.stderr)
        return 1

    # Locate phase log
    candidates = [args.phase_log] if args.phase_log else []
    candidates += [args.sim_dir / "agent_phase_log.txt",
                   Path("/tmp/agent_phase_log.txt")]
    phase_log = next((p for p in candidates if p and p.is_file()), None)
    if phase_log is None:
        print(f"error: phase log not found in {candidates}", file=sys.stderr)
        return 1
    phase_names = read_phase_log(phase_log)
    print(f"# phase log: {phase_log} ({len(phase_names)} begin events)",
          file=sys.stderr)

    # Discover ROI stat files
    roi_files: list[tuple[int, Path]] = []
    for p in sorted(args.sim_dir.iterdir()):
        m = ROI_FILE_RE.match(p.name)
        if m and m.group("base") == args.stats_file:
            roi_files.append((int(m.group("n")), p))
    roi_files.sort()
    if not roi_files:
        print(f"error: no {args.stats_file}.roi.*.csv files in {args.sim_dir}",
              file=sys.stderr)
        return 1
    print(f"# found {len(roi_files)} roi files for {args.stats_file}",
          file=sys.stderr)

    # Key stats to extract (these exist in core.stat.0 for scarab_ll)
    KEY_STATS = [
        "NODE_CYCLE",             # cycles
        "NODE_INST_COUNT",        # retired insts
        "NODE_IPC",               # IPC (optional — we recompute)
        "INST_COUNT",             # alt inst count
        "CYCLE_COUNT",            # alt cycle count
    ]

    # Accumulate per-phase totals
    per_phase_cycles: dict[str, float] = defaultdict(float)
    per_phase_insts:  dict[str, float] = defaultdict(float)
    per_phase_occurrences: dict[str, int] = defaultdict(int)
    unmatched = 0

    for n, path in roi_files:
        if n >= len(phase_names):
            unmatched += 1
            continue
        phase = phase_names[n]
        stats = read_stat_csv(path)
        cycles = (stats.get("NODE_CYCLE")
                  or stats.get("CYCLE_COUNT") or 0.0)
        insts  = (stats.get("NODE_INST_COUNT")
                  or stats.get("INST_COUNT") or 0.0)
        per_phase_cycles[phase] += cycles
        per_phase_insts[phase]  += insts
        per_phase_occurrences[phase] += 1

    if unmatched:
        print(f"# warning: {unmatched} roi.N files have no corresponding "
              f"phase-log entry (N >= {len(phase_names)})", file=sys.stderr)

    # Print table sorted by total insts
    print(f"{'phase':<20} {'occurrences':>11} {'insts':>14} "
          f"{'cycles':>14} {'IPC':>6}")
    phases_sorted = sorted(per_phase_insts.items(), key=lambda kv: -kv[1])
    total_insts = sum(per_phase_insts.values())
    for phase, insts in phases_sorted:
        cycles = per_phase_cycles[phase]
        ipc = insts / cycles if cycles else 0.0
        pct = 100.0 * insts / total_insts if total_insts else 0.0
        print(f"{phase:<20} {per_phase_occurrences[phase]:>11} "
              f"{int(insts):>14} {int(cycles):>14} {ipc:>6.3f}  ({pct:4.1f}% insts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
