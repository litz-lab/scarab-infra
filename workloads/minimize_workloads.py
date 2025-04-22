import json

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

# Load the JSON input
with open('workloads_db.json') as f:
    data = json.load(f)

# Process it
minimized = extract_workloads_with_simpoints(data)

# Write to a new JSON file
with open('workloads_top_simp.json', 'w') as f:
    json.dump(minimized, f, indent=2, separators=(",", ":"))
