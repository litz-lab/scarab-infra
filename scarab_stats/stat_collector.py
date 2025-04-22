#!/usr/bin/env python3

import argparse
import os
os.chdir("scarab_stats")

import scarab_stats

parser = argparse.ArgumentParser()
parser.add_argument('-d','--descriptor_name', required=True, help='Experiment descriptor name. Usage: -d exp.json')
parser.add_argument('-o','--outfile', required=True, help='Experiment descriptor name. Usage: -d exp.json')
args = parser.parse_args()

descriptor_name = args.descriptor_name
outfile = args.outfile

da = scarab_stats.stat_aggregator()
print(f"Loading {descriptor_name}")
E = da.load_experiment_json(descriptor_name, True)
E.to_csv(f"{outfile}")