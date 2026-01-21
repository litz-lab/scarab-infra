#!/usr/bin/env python3
"""stat_collector.py

Purpose:
  This is a thin command-line wrapper around `scarab_stats.stat_aggregator.write_experiment_csv_numpy()`.

Why this script exists:
  - It provides a stable CLI for users and for Slurm jobs.
  - All heavy lifting (descriptor expansion, parsing, CSV writing, postprocess) lives in scarab_stats.py.

Behavior:
  - Fast path only: we always use the NumPy-based collector.
  - If --postprocess is provided, derived IPC + distribution statistics are appended to the SAME outfile.

Usage examples:
  # Minimal
  python3 stat_collector.py -d /path/to/experiment_descriptor.json -o /path/to/experiment.csv

  # Common (recommended): include derived IPC + distribution stats and use more workers
  python3 stat_collector.py -d /path/to/experiment_descriptor.json -o /path/to/experiment.csv \
    --postprocess --skip-incomplete --jobs 16
"""


from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ensure_import_paths() -> None:
    """Allow running from either the project root or a scripts/ directory."""
    here = Path(__file__).resolve().parent
    parent = here.parent
    for p in (here, parent):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)


def main() -> None:
    # Ensure this script can import `scarab_stats` whether run from repo root or scripts/.
    _ensure_import_paths()
    # Import after path fix to avoid ImportError when run from different working directories.
    import scarab_stats  # noqa: E402

    # Define the CLI. We intentionally expose only fast-path knobs.
    parser = argparse.ArgumentParser(description="Collect Scarab stats into a CSV (fast path only).")
    parser.add_argument("-d", "--descriptor_name", required=True, help="Path to experiment descriptor JSON.")
    parser.add_argument("-o", "--outfile", required=True, help="Output CSV path.")
    parser.add_argument(
        "--postprocess",
        action="store_true",
        help="Append derived IPC and distribution stats to the CSV.",
    )
    parser.add_argument(
        "--skip-incomplete",
        action="store_true",
        help="Skip incomplete simpoints (warnings will be printed).",
    )
    parser.add_argument("--jobs", type=int, default=8, help="Worker processes to use.")

    # Parse CLI arguments.
    args = parser.parse_args()

    # Normalize paths and ensure the output directory exists.
    descriptor_path = Path(args.descriptor_name)
    out_path = Path(args.outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Instantiate the aggregator and run the collector.
    da = scarab_stats.stat_aggregator()
    # Generate the CSV. All descriptor expansion + parsing happens inside scarab_stats.
    written = da.write_experiment_csv_numpy(
        str(descriptor_path),
        str(out_path),
        slurm=True,
        postprocess=bool(args.postprocess),
        skip_incomplete=bool(args.skip_incomplete),
        jobs=int(args.jobs) if args.jobs is not None else 8,
    )
    # Print the final output path for logs/automation.
    print(f"Wrote CSV: {written}")


if __name__ == "__main__":
    main()
