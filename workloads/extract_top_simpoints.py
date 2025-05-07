#!/usr/bin/python3

# 04/23/2025 | Surim Oh | extract_top_simpoints.py
# A script that extracts top simpoints from the source workloads_db.json and write the recomputed weights of top simpoints to the destination workloads_top_simp.json

import json
import argparse
import copy

def normalize_weights(simpoints):
    total = sum(sp['weight'] for sp in simpoints)
    if total == 0:  # Avoid division by zero
        return simpoints
    for sp in simpoints:
        sp['weight'] = sp['weight'] / total
    return simpoints

def process_simpoints(simpoints):
    # Remove simpoints with null weight
    simpoints = [sp for sp in simpoints if sp.get('weight') is not None]
    # Sort and take top 3 by weight
    simpoints = sorted(simpoints, key=lambda x: x['weight'], reverse=True)[:3]
    return normalize_weights(simpoints)

def modify_simpoints_in_place(d):
    if isinstance(d, dict):
        if 'simpoints' in d and isinstance(d['simpoints'], list):
            d['simpoints'] = process_simpoints(d['simpoints'])
        for v in d.values():
            modify_simpoints_in_place(v)
    elif isinstance(d, list):
        for item in d:
            modify_simpoints_in_place(item)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract the top simpoints and recompute their weights')

    parser.add_argument('-s', '--src_descriptor_name', required=True, help='Source workloads_db.json descriptor name.')
    parser.add_argument('-d', '--dest_descriptor_name', required=True, help='Destination workloads_top_simp.json descriptor name.')
    args = parser.parse_args()

    with open(args.src_descriptor_name) as f_in:
        data = json.load(f_in)
        # Deep copy to avoid modifying input if needed elsewhere
        modified_data = copy.deepcopy(data)
        modify_simpoints_in_place(modified_data)

        with open(args.dest_descriptor_name, 'w') as f_out:
            json.dump(modified_data, f_out, indent=2, separators=(",", ":"))

