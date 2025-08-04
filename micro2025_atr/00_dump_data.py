#!/usr/bin/env python3

# ----------------------------------------------------------------------------
# File: 00_dump_data.py
# Description:
#   This script reads a simulation experiment configuration from a JSON file,
#   which includes the setup details and path to Scarab SimPoint output data.
#   It then parses and aggregates the statistics, and exports the results
#   to a CSV file for analysis or visualization.
# ----------------------------------------------------------------------------

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "scarab_stats"))
from scarab_stats import stat_aggregator

JSON_PATH = Path("../json/atr.json").resolve()
DATA_PATH = Path("./data/atr.csv").resolve()

def main():
    if not JSON_PATH.is_file():
        print(f"[ERROR] Input JSON file not found: {JSON_PATH}")
        raise FileNotFoundError(JSON_PATH)

    print(f"[INFO] Loading experiment from: {JSON_PATH}")
    aggregator = stat_aggregator()
    experiment = aggregator.load_experiment_json(str(JSON_PATH), False)

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    experiment.to_csv(DATA_PATH)
    print(f"[INFO] CSV successfully saved to: {DATA_PATH}")

if __name__ == "__main__":
    main()
