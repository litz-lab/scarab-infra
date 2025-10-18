#!/usr/bin/env python3

import argparse
import json
import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import scarab_stats

print("START stat collector")

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--descriptor_name', required=True, help='Experiment descriptor name. Usage: -d exp.json')
parser.add_argument('-o', '--outfile', required=True, help='Output csv path. Usage: -o out.csv')
args = parser.parse_args()

descriptor_path = Path(args.descriptor_name)
outfile = args.outfile

try:
    with descriptor_path.open("r", encoding="utf-8") as handle:
        descriptor = json.load(handle)
except FileNotFoundError:
    print(f"Error: descriptor '{descriptor_path}' not found.")
    print("DONE")
    exit(1)
except json.JSONDecodeError as exc:
    print(f"Error decoding descriptor '{descriptor_path}': {exc}")
    print("DONE")
    exit(1)

root_directory = Path(descriptor["root_dir"]) / "simulations" / descriptor["experiment"] / "logs"

if not root_directory.exists():
    print(f"Error: log directory '{root_directory}' not found.")
    print("DONE")
    exit(1)

error_runs = []

for file in root_directory.iterdir():
    if not file.is_file():
        continue
    contents = file.read_text()
    if "Error" in contents:
        error_runs.append(file.name)

if error_runs:
    print(f"Not running due to {len(error_runs)} errors")
    print("DONE")
    exit(1)

try:
    da = scarab_stats.stat_aggregator()
    print(f"Loading {descriptor_path}")
    experiment = da.load_experiment_json(str(descriptor_path), True)
    experiment.to_csv(outfile)

    if experiment is not None:
        print(f"Statistics collected for {descriptor['experiment']}")
except Exception as exc:
    print(f"Stat collector failed: {exc}")
finally:
    print("DONE")
