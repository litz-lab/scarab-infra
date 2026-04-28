#!/usr/bin/env python3
"""Parse scarab's power.stat.0.csv into a dict consumable by converter.py.

Scarab dumps both per-period (`*_count`) and accumulated (`*_total_count`)
columns in power.stat.0.csv. PR 178's converter expects flat
`POWER_*_count` keys; for simpoint-aggregated power analysis we want the
accumulated totals. This module reads the CSV, picks the `*_total_count`
rows, strips the `_total_` infix, and returns a dict the converter can
consume directly (no manual power.pkl required).

Usage:
    from parse_power_stat import scarab_csv_to_power_dict
    d = scarab_csv_to_power_dict("path/to/power.stat.0.csv")
    # d["POWER_CYCLE_count"] == accumulated cycles for the simpoint
"""
from __future__ import annotations

import csv
from pathlib import Path


def scarab_csv_to_power_dict(csv_path: str | Path) -> dict[str, float]:
    """Read a scarab power.stat.*.csv and return the dict converter.py wants.

    The CSV rows look like:
        POWER_CYCLE_count, 0,       3091316
        POWER_CYCLE_total_count, 0,      18535776
    We pick the *_total_count rows (whole-simpoint accumulation) and key
    them under the *_count name converter.py expects. The few non-POWER
    rows at the top of the file (Core, Cumulative_*, Periodic_*) are
    ignored.
    """
    out: dict[str, float] = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            name = row[0].strip()
            if not name.endswith("_total_count"):
                continue
            try:
                val = float(row[-1].strip())
            except ValueError:
                continue
            # POWER_CYCLE_total_count → POWER_CYCLE_count
            key = name[: -len("_total_count")] + "_count"
            out[key] = val
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        sys.exit("usage: parse_power_stat.py <power.stat.0.csv>")
    d = scarab_csv_to_power_dict(sys.argv[1])
    print(f"# {len(d)} keys parsed from {sys.argv[1]}")
    for k, v in sorted(d.items())[:10]:
        print(f"  {k} = {v}")
