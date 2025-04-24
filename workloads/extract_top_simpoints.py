#!/usr/bin/python3

# 04/23/2025 | Surim Oh | extract_top_simpoints.py
# A script that extracts top simpoints from the source workloads_db.json and write the recomputed weights of top simpoints to the destination workloads_top_simp.json

import json
import argparse

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

def extract_workloads_with_simpoints(data):
    result = {}

    def recurse(d, path):
        if isinstance(d, dict):
            if 'simpoints' in d and isinstance(d['simpoints'], list):
                # Process this workload
                processed = d.copy()
                processed['simpoints'] = process_simpoints(d['simpoints'])
                # Set in result
                sub_result = result
                for p in path[:-1]:
                    sub_result = sub_result.setdefault(p, {})
                sub_result[path[-1]] = processed
            else:
                for k, v in d.items():
                    recurse(v, path + [k])

    recurse(data, [])
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract the top simpoints and recompute their weights')

    # Add arguments
    parser.add_argument('-s','--src_descriptor_name', required=True, help='Source workloads_db.json descriptor name. Usage: -s workloads_db.json')
    parser.add_argument('-d','--dest_descriptor_name', required=True, help='Destination workloads_top_simp.json descriptor name. Usage: -d workloads_top_simp.json')
    args = parser.parse_args()

    with open(args.src_descriptor_name) as f_in:
        data = json.load(f_in)
        top_simp = extract_workloads_with_simpoints(data)
        with open(args.dest_descriptor_name, 'w') as f_out:
            json.dump(top_simp, f_out, indent=2, separators=(",", ":"))
