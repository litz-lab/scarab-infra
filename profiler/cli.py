#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from config import Config
from loader import Loader
from tabulator import Tabulator
from plotter import Plotter

PROFILER_DIR = Path(__file__).resolve().parent
JSON_DIR = PROFILER_DIR / "json"
INPUT_DIR = PROFILER_DIR / "input"
OUTPUT_DIR = PROFILER_DIR / "output"
DEFAULT_NAME = "config"


def main() -> None:
    parser = argparse.ArgumentParser(description="Profiler CLI")
    parser.add_argument(
        "--name",
        default=DEFAULT_NAME,
        help=(
            "Base name: uses json/<name>.json and input/<name>.csv (when it "
            "exists), and writes results to output/<name>/"
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Override: JSON config file under json/ (or a path) to use instead of json/<name>.json",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Override: CSV filename under input/ to use instead of input/<name>.csv",
    )
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else JSON_DIR / f"{args.name}.json"
    if not config_path.is_absolute() and not config_path.exists():
        config_path = JSON_DIR / config_path

    csv_filename = args.csv
    if csv_filename is None:
        named_csv = INPUT_DIR / f"{args.name}.csv"
        if named_csv.exists():
            csv_filename = named_csv.name

    config = Config(config_path, csv_filename=csv_filename)
    output_dir = OUTPUT_DIR / args.name

    loader = Loader(config.csv_path)
    tabulator = Tabulator(loader)
    plotter = Plotter(tabulator, output_dir=output_dir)
    plotter.plot(config.outputs)


if __name__ == "__main__":
    main()
