#!/usr/bin/env python3

import argparse
import json
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

def load_workloads_data(descriptor: dict) -> dict:
    top = descriptor.get("top_simpoint")
    filename = "workloads_top_simp.json" if top else "workloads_db.json"
    workloads_path = project_root / "workloads" / filename
    try:
        with workloads_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        raise RuntimeError(f"Workloads file not found: {workloads_path}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse workloads file {workloads_path}: {exc}")


def resolve_simpoints(workload_entry: dict, sim_mode: str) -> list:
    if sim_mode == "memtrace":
        simpoints = workload_entry.get("simpoints") or []
        ids = []
        for entry in simpoints:
            cluster_id = entry.get("cluster_id")
            if cluster_id is not None:
                ids.append(str(cluster_id))
        return ids
    return ["0"]


def expected_run_paths(descriptor: dict, workloads_data: dict) -> list:
    configs = descriptor.get("configurations") or {}
    simulations = descriptor.get("simulations") or []
    experiment_root = Path(descriptor["root_dir"]) / "simulations" / descriptor["experiment"]

    metadata_errors = set()
    expected = set()

    for simulation in simulations:
        suite = simulation.get("suite")
        subsuite = simulation.get("subsuite")
        workload = simulation.get("workload")
        cluster_id = simulation.get("cluster_id")
        sim_mode = simulation.get("simulation_type")

        if suite not in workloads_data:
            metadata_errors.add(f"Suite '{suite}' not found in workloads database.")
            continue
        suite_data = workloads_data[suite]

        subsuites = [subsuite] if subsuite else list(suite_data.keys())
        for subsuite_name in subsuites:
            if subsuite_name not in suite_data:
                metadata_errors.add(f"Subsuite '{suite}/{subsuite_name}' not found in workloads database.")
                continue
            subsuite_data = suite_data[subsuite_name]

            workloads = [workload] if workload else list(subsuite_data.keys())
            for workload_name in workloads:
                if workload_name not in subsuite_data:
                    metadata_errors.add(f"Workload '{suite}/{subsuite_name}/{workload_name}' not found in workloads database.")
                    continue
                workload_entry = subsuite_data[workload_name]

                resolved_mode = sim_mode
                if not resolved_mode:
                    resolved_mode = workload_entry.get("simulation", {}).get("prioritized_mode")
                if not resolved_mode:
                    metadata_errors.add(f"No simulation mode available for '{suite}/{subsuite_name}/{workload_name}'.")
                    continue

                if cluster_id is None:
                    cluster_ids = resolve_simpoints(workload_entry, resolved_mode)
                    if not cluster_ids:
                        metadata_errors.add(f"No simpoints available for '{suite}/{subsuite_name}/{workload_name}' in mode '{resolved_mode}'.")
                        continue
                else:
                    cluster_ids = [str(cluster_id)]

                for cluster in cluster_ids:
                    for config_name in configs.keys():
                        run_dir = experiment_root / config_name / suite / subsuite_name / workload_name / str(cluster)
                        expected.add(run_dir)

    if metadata_errors:
        for message in sorted(metadata_errors):
            print(f"Error: {message}")
        raise RuntimeError("Descriptor references missing entries in workloads database.")

    return sorted(expected)


def verify_simulation_outputs(descriptor: dict) -> None:
    try:
        workloads_data = load_workloads_data(descriptor)
    except RuntimeError as exc:
        print(exc)
        print("DONE")
        exit(1)

    run_dirs = expected_run_paths(descriptor, workloads_data)
    if not run_dirs:
        print("No simulation runs described in descriptor; nothing to collect.")
        print("DONE")
        exit(0)

    missing = []
    for run_dir in run_dirs:
        inst_path = run_dir / "inst.stat.0.csv"
        if not inst_path.is_file():
            missing.append(str(run_dir))

    if missing:
        print("Stat collection aborted: some simulation outputs are incomplete.")
        for entry in missing[:20]:
            print(f"Missing inst.stat.0.csv in {entry}")
        if len(missing) > 20:
            print(f"... {len(missing) - 20} additional paths omitted ...")
        print("DONE")
        exit(1)


verify_simulation_outputs(descriptor)

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
