#!/usr/bin/env python3

import argparse

import os
import sys

from scripts.utilities import read_descriptor_from_json
import scarab_stats

print("START stat collector")

parser = argparse.ArgumentParser()
parser.add_argument('-d','--descriptor_name', required=True, help='Experiment descriptor name. Usage: -d exp.json')
parser.add_argument('-o','--outfile', required=True, help='Experiment descriptor name. Usage: -d out.csv')
args = parser.parse_args()

descriptor_name = args.descriptor_name
outfile = args.outfile

json = read_descriptor_from_json(descriptor_name)
if json is None:
    print("Error: JSON file not found or invalid.")
    exit(1)
    
root_directory = json["root_dir"] + "/simulations/" + json["experiment"] + "/logs/"

log_files = os.listdir(root_directory)

error_runs = []

for file in log_files:
    with open(root_directory+file, 'r') as f:
        contents = f.read()
        if 'Error' in contents:
            error_runs += [file]

if len(error_runs) > 0:
    print(f"Not running due to {len(error_runs)} errors")
    print("DONE")
    exit(1)

try:
    da = scarab_stats.stat_aggregator()
    print(f"Loading {descriptor_name}")
    E = da.load_experiment_json(descriptor_name, True)
    E.to_csv(f"{outfile}")

    # This is the 'success' message
    if E is not None:
        print(f"Statistics collected for {json['experiment']}")
except:
    pass
finally:
    print("DONE")