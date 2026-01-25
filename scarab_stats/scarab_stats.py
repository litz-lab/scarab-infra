"""scarab_stats.py

This module contains two major pieces of functionality:

1) CSV generation (FAST PATH ONLY)
   - `stat_aggregator.write_experiment_csv_numpy()` expands an experiment descriptor JSON into a concrete
     list of simpoint runs, parses each run's stat outputs, and writes a consolidated CSV.
   - If `postprocess=True`, the generator appends:
       * derived IPC rows, and
       * distribution statistics rows,
     to the *same* CSV file.

2) CSV analysis utilities (KEPT)
   - `Experiment` is a read-only wrapper around the generated CSV (loaded via pandas).
   - Plotting and diff helpers use `Experiment` to visualize/compare results.

The legacy “slow path” (incremental pandas DataFrame building) has been removed. Generation is NumPy-based.

CSV layout (conceptually):
  rows    = stats (+ some metadata rows)
  columns = simpoints (config + workload + cluster_id)
  fields  = [stats, write_protect, groups, <one column per simpoint>]
"""

import argparse
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
import numpy as np
import csv
import matplotlib
import matplotlib.patches as mpatches
import json
import os
import math
import multiprocessing
import atexit
from multiprocessing import shared_memory
from pathlib import Path

from scripts import utilities  # Descriptor parsing + simpoint metadata helpers

def _cleanup_shared_memory(name: str) -> None:
    """Best-effort cleanup for SharedMemory segments (avoids resource_tracker warnings on crashes)."""
    try:
        try:
            shm = shared_memory.SharedMemory(name=name, track=False)
        except TypeError:
            # Python < 3.13
            shm = shared_memory.SharedMemory(name=name)
    except FileNotFoundError:
        return
    except Exception:
        return
    try:
        try:
            shm.close()
        finally:
            shm.unlink()
    except FileNotFoundError:
        # Already unlinked
        pass
    except Exception:
        # Best-effort
        try:
            shm.close()
        except Exception:
            pass


def _infra_root() -> Path:
    """Return the repository's 'infra' root directory.

    We use this to locate companion JSON files (e.g., workloads/workloads_db.json) relative to this script.
    """
    return Path(__file__).resolve().parent.parent

def get_elem(l, i):
    """Return the i-th element from each entry in a list.

    A small helper used by some plotting code where inputs are often lists of tuples.
    """
    return list(map(lambda x: x[i], l))
class Experiment:
    """Read-only wrapper around a generated experiment CSV.

    This class exists to make plotting/diff code simpler:

      - The generator writes a flat CSV (one row per stat, one column per simpoint).
      - For analysis we load that CSV into a pandas DataFrame once.
      - Helper methods then validate selections and perform common aggregations.

    NOTE: The legacy slow-path builder methods have been removed; this class does not *construct*
    experiments, it only *reads* them.
    """

    def __init__(self, csv_path):
        self.data = pd.read_csv(str(csv_path), low_memory=False)

    def retrieve_stats(self, config: List[str], stats: List[str], workload: List[str], 
        aggregation_level:str = "Workload", simpoints: List[str] = None):
        results = {}
        # Get available workloads and configs from the dataframe
        available_workloads = set(self.data[self.data["stats"] == "Workload"].iloc[0][3:])
        available_configs = set(self.data[self.data["stats"] == "Configuration"].iloc[0][3:])

        # Validate workloads
        missing_workloads = set(workload) - available_workloads

        if missing_workloads:
            print(f"ERROR: The following workloads do not exist in the dataframe: {missing_workloads}")
            return None

        # Validate configs
        missing_configs = set(config) - available_configs
        if missing_configs:
            print(f"ERROR: The following configurations do not exist in the dataframe: {missing_configs}")
            return None

        if aggregation_level == "Workload":
            for c in config:
                for w in workload:
                    selected_simpoints = [col for col in self.data.columns if col.startswith(f"{c} {w}")]

                    for stat in stats:
                        values = list(self.data[selected_simpoints][self.data["stats"] == stat].iloc[0])
                        weights = list(self.data[selected_simpoints][self.data["stats"] == "Weight"].iloc[0])
                        values = list(map(float, values))
                        weights = list(map(float, weights))
                        results[f"{c} {w} {stat}"] = sum([v*w for v, w in zip(values, weights)])

        elif aggregation_level == "Simpoint":
            for c in config:
                for w in workload:

                    # Set selected simpoints to all possible if not provided
                    if simpoints == None:
                        selected_simpoints = [col.split(" ")[-1] for col in self.data.columns if col.startswith(f"{c} {w}")]
                    else: selected_simpoints = simpoints

                    for sp in selected_simpoints:
                        for stat in stats:
                            col = f"{c} {w} {sp}"
                            results[f"{c} {w} {sp} {stat}"] = self.data[col][self.data["stats"] == stat].iloc[0]

        elif aggregation_level == "Config":
            for c in config:
                config_data = {stat:[] for stat in stats}
                for w in workload:
                    selected_simpoints = [col for col in self.data.columns if col.startswith(f"{c} {w}")]

                    for stat in stats:
                        values = list(self.data[selected_simpoints][self.data["stats"] == stat].iloc[0])
                        weights = list(self.data[selected_simpoints][self.data["stats"] == "Weight"].iloc[0])
                        values = list(map(float, values))
                        weights = list(map(float, weights))
                        config_data[stat].append(sum([v*w for v, w in zip(values, weights)]))

                #print(config_data)
                for stat, val in config_data.items():
                    results[F"{c} {stat}"] = reduce(lambda x,y: x*y, val) ** (1/len(val))

        else:
            print(f"ERROR: Invalid aggreagation level {aggregation_level}.")
            print("Must be 'Workload' 'Simpoint' or 'Config'")
            return None

        return results

    def return_raw_data(self, must_contain: list = None, keep_weight: bool = False):
        """Return stat rows as a DataFrame (dropping metadata rows by default).

        The generated CSV includes metadata rows such as Experiment/Workload/Weight.
        Most raw analysis wants only true stat rows, so those are dropped.

        Args:
            must_contain: Keep only stats whose name contains this substring.
            keep_weight: If True, keep the "Weight" row (otherwise treated like metadata).
        """
        # Extra rows added by stat program as metadata
        metadata = ["Experiment",
                    "Architecture",
                    "Configuration",
                    "Workload",
                    "Segment Id",
                    "Cluster Id"]

        if not keep_weight: metadata.append("Weight")

        rows_to_drop = [self.data.index[self.data["stats"] == stat][0] for stat in metadata]

        if must_contain != None:
            bit_map = list(map(lambda x: not must_contain in x, list(self.data["stats"])))
            rows_to_drop += list(self.data.index[bit_map])
            #for row in list(self.data["stats"]):
            #    if must_contain not in row:
            #        rows_to_drop.append(self.data.index[self.data["stats"] == row][0])

        return self.data.copy().drop(rows_to_drop)

    def get_experiments(self):
        return list(set(list(self.data[self.data["stats"] == "Experiment"].iloc[0])[3:]))

    def get_configurations(self):
        return list(set(list(self.data[self.data["stats"] == "Configuration"].iloc[0])[3:]))

    def get_workloads(self):
        return list(set(list(self.data[self.data["stats"] == "Workload"].iloc[0])[3:]))

    def get_stats(self):
        return list(set(self.data["stats"]))

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"{', '.join(list(self.data.columns))}"

# Component stat files produced for each simpoint directory.
# Each file is a 3-column CSV: <stat_name>, <group_id>, <value>.
# The fast path parses these directly (without pandas) for performance.
stat_files = ["bp.stat.0.csv",
              "core.stat.0.csv",
              "fetch.stat.0.csv",
              "inst.stat.0.csv",
              "l2l1pref.stat.0.csv",
              "memory.stat.0.csv",
              "power.stat.0.csv",
              "pref.stat.0.csv",
              "stream.stat.0.csv"]

class stat_aggregator:
    """CSV generator + analysis helper.

    Generation:
      - Expand experiment descriptor -> list of simpoint tasks
      - Parse each simpoint directory -> numeric matrix (NumPy)
      - Write consolidated CSV (+ optional postprocess rows)

    Analysis:
      - Plotting routines (speedups, stacked bars, per-simpoint plots, etc.)
      - Diff routines to compare two experiments.
    """

    def __init__(self) -> None:
        self.experiments = {}
        self.simpoint_info = {}

    def colorwheel(self, x):
        print ((math.cos(2*math.pi*x)+1.5)/2.5, (math.cos(2*math.pi*x+(math.pi/1.5))+1.5)/2.5, (math.cos(2*math.pi*x+2*(math.pi/4))+1.5)/2.5)
        return ((math.cos(2*math.pi*x)+1.5)/2.5, (math.cos(2*math.pi*x+(math.pi/1.5))+1.5)/2.5, (math.cos(2*math.pi*x+2*(math.pi/4))+1.5)/2.5)

    def get_all_stats(self, path, load_ramulator=True, ignore_duplicates = True):
        all_stats = []

        for file in stat_files:
            filename = f"{path}{file}"
            df = pd.read_csv(filename).T
            df.columns = df.iloc[0]
            df = df.drop(df.index[0])
            to_add = list(df.columns)

            if ignore_duplicates:
                duplicates = set(to_add) & set(all_stats)

                for duplicate in duplicates:
                    to_add.remove(duplicate)

            all_stats += to_add

        if load_ramulator:
            f = open(f"{path}ramulator.stat.out")
            lines = f.readlines()

            for line in lines:
                if not "ramulator." in line:
                    continue
                all_stats.append(line.split()[0])

            f.close()

        return all_stats

    # Load simpoint from csv file as pandas dataframe
    def load_simpoint(self, path, load_ramulator=True, ignore_duplicates = True, return_stats = False, order = None, return_stats_and_data: bool = False):
        """Load stats for a single simpoint directory.

        PERFORMANCE: This method previously parsed each *.stat.0.csv using pandas and
        transposed the DataFrame. That was extremely expensive. We now parse these
        3-column CSVs using the built-in `csv` module.

        Behavior is preserved:
          - The first line of each *.stat.0.csv is treated as a header and skipped
            (e.g., 'Core, 0, 0'), matching the old pandas header behavior.
          - Duplicate stat names within a single CSV are permitted only if their values
            are identical (same as the previous 'resolvable duplicate' check).
          - When `ignore_duplicates` is True, stats that appear in earlier component
            files are skipped (same as before).
          - `return_stats=True` returns the ordered stat-name list.
          - `return_stats_and_data=True` returns (stat-names, data, group) from a single parse.
          - `order` reorders/fills the returned data to match `order`, with missing
            stats filled with NaN (same as previous reindex(fill_value='nan')).
        """
        import csv
        import math

        all_stats = []

        # Fast lookup tables
        values_by_stat = {}
        groups_by_stat = {}

        def _to_float(x: str) -> float:
            try:
                return float(x)
            except Exception:
                return float("nan")

        for file in stat_files:
            filename = f"{path}{file}"

            # Read (stat, group, value) lines. Preserve first-seen order.
            per_file = {}
            per_file_order = []
            dup_stats = set()

            with open(filename, newline="") as f:
                reader = csv.reader(f)

                # Old pandas path used the first line as a header; skip it.
                try:
                    next(reader)
                except StopIteration:
                    continue

                for row in reader:
                    if not row:
                        continue
                    if all((c is None) or (str(c).strip() == "") for c in row):
                        continue
                    if len(row) < 3:
                        continue

                    stat = str(row[0]).strip()
                    grp_raw = str(row[1]).strip()
                    val_raw = str(row[2]).strip()

                    # Match old behavior: drop duplicates against previously-collected stats
                    # when ignore_duplicates=True.
                    if ignore_duplicates and stat in values_by_stat:
                        continue

                    try:
                        grp = int(grp_raw)
                    except Exception:
                        try:
                            grp = int(float(grp_raw))
                        except Exception:
                            grp = 0

                    val = _to_float(val_raw)

                    if stat in per_file:
                        dup_stats.add(stat)
                        prev_grp, prev_val = per_file[stat]
                        # Resolvable only if duplicates are equivalent.
                        if (prev_grp != grp) or (prev_val != val and not (math.isnan(prev_val) and math.isnan(val))):
                            print(f"ERR: Unable to resolve duplicates. Duplicate columns have unique values. File:{filename}")
                            exit(1)
                        continue

                    per_file[stat] = (grp, val)
                    per_file_order.append(stat)

            if dup_stats:
                print("WARN: CSV file contains duplicates")
                print("Duplicates are:", dup_stats)
                print("Checking if issue is resolvable...")
                print("Duplicates equivalent! Resolved")

            for stat in per_file_order:
                grp, val = per_file[stat]

                # Old code errors on cross-file duplicates when ignore_duplicates=False.
                if (not ignore_duplicates) and (stat in values_by_stat):
                    print("ERR: Duplication prevention logic failed")
                    exit(1)

                all_stats.append(stat)
                values_by_stat[stat] = val
                groups_by_stat[stat] = grp

        # NOTE: Ramulator will not be in distribution, group should be 0
        if load_ramulator:
            with open(f"{path}ramulator.stat.out", "r") as f:
                for line in f:
                    if "ramulator." not in line:
                        continue
                    parts = line.split()
                    if len(parts) < 2:
                        continue

                    stat = parts[0]
                    val = _to_float(parts[1])

                    if ignore_duplicates and stat in values_by_stat:
                        continue
                    if (not ignore_duplicates) and (stat in values_by_stat):
                        print("ERR: Duplication prevention logic failed")
                        exit(1)

                    all_stats.append(stat)
                    values_by_stat[stat] = val
                    groups_by_stat[stat] = 0

        if return_stats and not return_stats_and_data:
            return all_stats

        ordered_stats = all_stats if order is None else order

        data = [values_by_stat.get(stat, float("nan")) for stat in ordered_stats]
        group = [groups_by_stat.get(stat, 0) for stat in ordered_stats]

        if return_stats_and_data:
            return all_stats, data, group

        if return_stats:
            return all_stats

        return data, group


    def _build_component_csv_schemas(self, sim_dir0: str, stat_to_row: dict, verify_k: int = 3):
        """Build per-component CSV schemas from simpoint0.

        A schema maps each non-empty data line in a component CSV (excluding the first pseudo-header line)
        to a row index in the output matrix (or -1 if the stat is not in `stat_to_row`).

        We also record a few leading stat names to cheaply validate that ordering hasn't changed.
        If a schema looks unsafe (e.g., duplicates within the file), we mark it so we can fall back
        to the robust parser for that file to preserve behavior and warnings.
        """
        schemas = {}
        for file in stat_files:
            filename = f"{sim_dir0}{file}"
            try:
                with open(filename, "r", buffering=512 * 512, newline="") as f:
                    _ = f.readline()  # skip pseudo-header
                    row_indices = []
                    check_stats = []
                    seen_stats = set()
                    has_dups = False

                    for line in f:
                        if not line:
                            continue
                        line = line.strip()
                        if not line:
                            continue

                        parts = line.split(",", 2)
                        if len(parts) < 3:
                            continue
                        stat = parts[0].strip()
                        if not stat:
                            continue

                        if len(check_stats) < verify_k:
                            check_stats.append(stat)

                        if stat in seen_stats:
                            has_dups = True
                        else:
                            seen_stats.add(stat)

                        row_idx = stat_to_row.get(stat)
                        row_indices.append(row_idx if row_idx is not None else -1)

                schemas[file] = {
                    "row_indices": row_indices,
                    "check_stats": check_stats,
                    "has_dups": has_dups,
                }
            except FileNotFoundError:
                # Some components may be absent; match old behavior by skipping missing files.
                continue

        return schemas

    def load_simpoint_into(
        self,
        path: str,
        stat_to_row: dict,
        out_col,
        *,
        load_ramulator: bool = True,
        ignore_duplicates: bool = True,
        clear_column: bool = True,
        schema_cache: dict = None,
    ) -> None:
        """Fill a preallocated column vector with stats from a simpoint directory.

        Performance-oriented variant used by `write_experiment_csv_numpy()`.
        It fills `out_col` in-place (preinitialized to NaN) using `stat_to_row` mapping.

        Semantics match `load_simpoint(..., order=known_stats)`:
          - out_col is set to NaN for missing stats.
          - Cross-file duplicates honor `ignore_duplicates` (earlier component wins).
          - Unknown stats (not present in stat_to_row) are ignored (same as ordering would drop them).

        When `schema_cache` is provided (built from simpoint0), component CSVs can be parsed
        without per-line stat-name parsing, with correctness preserved via cheap validation
        and robust fallback.
        """

        # We fill one output column vector representing *one simpoint*.
        #
        # The vector is preallocated by the caller (possibly in SharedMemory). Filling in-place:
        #   - avoids per-simpoint allocations,
        #   - enables multiprocessing where each worker writes a distinct column,
        #   - preserves the older semantics (missing stats -> NaN).


        nan = float("nan")
        if clear_column:
            out_col[:] = nan  # Ensure missing stats are NaN by default

        import math as _math

        # Mapping produced by write_experiment_csv_numpy() is dense: 0..N-1.
        # Use a bytearray for fast duplicate tracking by row index.
        n_rows = len(stat_to_row)
        seen = bytearray(n_rows)

        # Used only to distinguish within-file duplicates from cross-file duplicates in schema-fast mode.
        file_seen_tag = [0] * n_rows

        s2r_get = stat_to_row.get
        out = out_col
        ign = ignore_duplicates
        isnan = _math.isnan

        # Parse group field only when needed (within-file duplicates).
        def _parse_group(_s: str) -> int:
            _s = _s.strip()
            if _s == "":
                return 0
            try:
                return int(float(_s))
            except Exception:
                return 0

        parse_group = _parse_group

        def _assign_row(row_idx: int, val: float) -> None:
            # Cross-file duplicates (earlier file wins)
            if seen[row_idx]:
                if ign:
                    return
                print("ERR: Duplication prevention logic failed")
                exit(1)
            out[row_idx] = val
            seen[row_idx] = 1

        def _parse_component_file_robust(filename: str) -> None:
            # Track within-file duplicates so we can keep the same warnings/errors as the original.
            per_file = {}  # row_idx -> (grp_s, val)
            dup_stats = []

            with open(filename, "r", buffering=1024 * 1024, newline="") as f:
                _ = f.readline()  # Skip pseudo-header line (e.g., 'Core, 0, 0')

                for line in f:
                    if not line:
                        continue
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split(",", 2)
                    if len(parts) < 3:
                        continue

                    stat = parts[0].strip()
                    if not stat:
                        continue

                    row_idx = s2r_get(stat)
                    if row_idx is None:
                        continue

                    grp_s = parts[1]
                    try:
                        val = float(parts[2])
                    except Exception:
                        val = nan

                    if row_idx in per_file:
                        dup_stats.append(stat)
                        prev_grp_s, prev_val = per_file[row_idx]

                        # Compare values (NaN-safe)
                        if prev_val != val and not (isnan(prev_val) and isnan(val)):
                            print(f"ERR: Unable to resolve duplicates. Duplicate columns have unique values. File:{filename}")
                            exit(1)

                        # Compare groups (deferred parse)
                        if parse_group(prev_grp_s) != parse_group(grp_s):
                            print(f"ERR: Unable to resolve duplicates. Duplicate columns have unique values. File:{filename}")
                            exit(1)
                        continue

                    per_file[row_idx] = (grp_s, val)
                    _assign_row(row_idx, val)

            if dup_stats:
                print("WARN: CSV file contains duplicates")
                print("Duplicates are:", dup_stats)
                print("Checking if issue is resolvable...")
                print("Duplicates equivalent! Resolved")

        def _parse_component_file_schema_fast(filename: str, schema: dict, tag: int) -> None:
            row_indices = schema["row_indices"]
            check_stats = schema.get("check_stats", [])
            schema_len = len(row_indices)

            # Avoid per-line Python list appends; on mismatch we roll back by scanning row_indices.
            line_i = 0
            mismatch = False

            with open(filename, "r", buffering=1024 * 1024, newline="") as f:
                _ = f.readline()  # skip pseudo-header

                for raw in f:
                    if not raw:
                        continue
                    line = raw.strip()
                    if not line:
                        continue

                    if line_i >= schema_len:
                        mismatch = True
                        break

                    # Cheap order validation using the first few stat names
                    if line_i < len(check_stats):
                        parts = line.split(",", 2)
                        if len(parts) < 3:
                            mismatch = True
                            break
                        stat = parts[0].strip()
                        if stat != check_stats[line_i]:
                            mismatch = True
                            break

                    row_idx = row_indices[line_i]
                    line_i += 1

                    if row_idx < 0:
                        continue

                    if seen[row_idx]:
                        # Distinguish within-file duplicates (must be validated) from cross-file duplicates.
                        if file_seen_tag[row_idx] == tag:
                            mismatch = True
                            break
                        if ign:
                            continue
                        print("ERR: Duplication prevention logic failed")
                        exit(1)

                    # Parse value from the last comma onward (faster than split)
                    last = line.rfind(",")
                    if last == -1:
                        val = nan
                    else:
                        try:
                            val = float(line[last + 1 :].strip())
                        except Exception:
                            val = nan

                    out[row_idx] = val
                    seen[row_idx] = 1
                    file_seen_tag[row_idx] = tag


            # Detect shorter file (fewer data lines than schema)
            if not mismatch and line_i != schema_len:
                mismatch = True

            if mismatch:
                # Roll back and fall back to robust parsing for this file.
                for ridx in row_indices:
                    if ridx >= 0 and file_seen_tag[ridx] == tag:
                        out[ridx] = nan
                        seen[ridx] = 0
                        file_seen_tag[ridx] = 0
                _parse_component_file_robust(filename)

        # Parse component CSVs
        tag = 1
        for file in stat_files:
            filename = f"{path}{file}"
            try:
                if schema_cache is not None:
                    schema = schema_cache.get(file)
                else:
                    schema = None

                if schema is not None and not schema.get("has_dups", False):
                    _parse_component_file_schema_fast(filename, schema, tag)
                else:
                    _parse_component_file_robust(filename)
                tag += 1

            except FileNotFoundError:
                # Some components may be absent; match old behavior by skipping missing files.
                tag += 1
                continue

        # Ramulator stats are plain text; group is always 0
        if load_ramulator:
            with open(f"{path}ramulator.stat.out", "r", buffering=1024 * 1024) as f:
                for line in f:
                    if "ramulator." not in line:
                        continue
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    stat = parts[0]
                    row_idx = s2r_get(stat)
                    if row_idx is None:
                        continue
                    if seen[row_idx]:
                        if ign:
                            continue
                        print("ERR: Duplication prevention logic failed")
                        exit(1)
                    try:
                        out[row_idx] = float(parts[1])
                    except Exception:
                        out[row_idx] = nan
                    seen[row_idx] = 1
    def sp_is_complete(self, suite, subsuite, workload, config, cluster_id, experiment_name, simulations_path):
        directory = f"{simulations_path}{experiment_name}/{config}/{suite}/{subsuite}/{workload}/{str(cluster_id)}/"

        if not os.path.exists(directory):
            print(f"WARN: Simpoint directory <{directory}> not found")
            return False

        files = os.listdir(directory)

        for stat_file in stat_files:
            if stat_file not in files:
                print(f"WARN: Couldn't find csv file {directory}/{stat_file}")
                return False

        return True

    # Load experiment from saved file
    def load_experiment_csv(self, path):
        return Experiment(path)

    # Load experiment form json file


    def write_experiment_csv_numpy(
        self,
        experiment_file: str,
        outfile: str,
        slurm: bool = False,
        postprocess: bool = False,
        skip_incomplete: bool = False,
        jobs: int = 8,
    ) -> str:
        """Fast path: collect stats and write a base CSV without building a large pandas DataFrame incrementally.

        The descriptor expansion order matches the legacy pipeline so column ordering is consistent
        with the legacy pipeline. If `postprocess` is True, derived IPC stats and distribution stats are computed
        *after* the base CSV is written.
        """
        # ---------------------------------------------------------------------
        # Step 1: Read the experiment descriptor JSON and normalize paths.
        # ---------------------------------------------------------------------

        # Load json data from experiment descriptor file
        try:
            with open(experiment_file, "r") as file:
                json_data = json.loads(file.read())
        except Exception as exc:
            raise RuntimeError(f"Failed to read experiment descriptor {experiment_file}: {exc}")

        simulations_path = json_data["root_dir"]
        simpoints_path = "/soe/hlitz/lab/traces/" if json_data["traces_dir"] == None else json_data["traces_dir"]

        if simulations_path[-1] != '/': simulations_path += "/"
        if simpoints_path[-1] != '/': simpoints_path += "/"

        if not simulations_path.endswith("simulations/"):
            simulations_path += "simulations/"

        experiment_name = json_data["experiment"]
        architecture = json_data["architecture"]

        infra_dir = _infra_root()
        top_simpoint_only = bool(json_data["top_simpoint"])
        workload_db_path = str(infra_dir / "workloads" / ("workloads_top_simp.json" if top_simpoint_only else "workloads_db.json"))
        workloads_data = utilities.read_descriptor_from_json(workload_db_path)
        # ---------------------------------------------------------------------
        # Step 2: Expand the descriptor into a concrete list of simpoint tasks.
        #
        # Each task becomes exactly one output CSV column and has the shape:
        #   (config, suite, subsuite, workload, cluster_id)
        #
        # The descriptor can specify tasks at different granularities:
        #   - explicit cluster_id
        #   - a workload (all clusters for that workload)
        #   - a subsuite (all workloads in that subsuite)
        #   - a suite (all subsuites/workloads)
        # ---------------------------------------------------------------------


        tasks = []
        for config in json_data["configurations"]:
            config = config.replace("/", "-")
            for simulation in json_data["simulations"]:
                found_suite = simulation["suite"]
                found_subsuite = simulation["subsuite"]
                found_workload = simulation["workload"]
                found_cluster_id = simulation["cluster_id"]

                if found_cluster_id != None:
                    assert found_workload is not None, "Workload cannot be None when cluster_id is specified"
                    assert found_suite is not None, "Suite cannot be None when cluster_id is specified"
                    assert found_subsuite is not None, "Subsuite cannot be None when cluster_id is specified"
                    tasks.append((config, found_suite, found_subsuite, found_workload, str(found_cluster_id)))

                elif found_workload != None:
                    assert found_suite is not None, "Suite cannot be None when workload is specified"
                    assert found_subsuite is not None, "Subsuite cannot be None when workload is specified"
                    for cid in self.get_cluster_ids(found_workload, found_suite, found_subsuite, top_simpoint_only):
                        tasks.append((config, found_suite, found_subsuite, found_workload, str(cid)))

                elif found_subsuite != None:
                    assert found_suite is not None, "Suite cannot be None when subsuite is specified"
                    for workload in list(workloads_data[found_suite][found_subsuite].keys()):
                        for cid in self.get_cluster_ids(workload, found_suite, found_subsuite, top_simpoint_only, workloads_data=workloads_data):
                            tasks.append((config, found_suite, found_subsuite, workload, str(cid)))

                else:
                    for subsuite in list(workloads_data[found_suite].keys()):
                        for workload in list(workloads_data[found_suite][subsuite].keys()):
                            for cid in self.get_cluster_ids(workload, found_suite, subsuite, top_simpoint_only, workloads_data=workloads_data):
                                tasks.append((config, found_suite, subsuite, workload, str(cid)))

        # ... after building tasks ...
        # ---------------------------------------------------------------------
        # Step 3 (optional): Drop incomplete simpoints.
        #
        # If `skip_incomplete` is enabled and a simpoint directory is missing/partial,
        # we remove that simpoint for ALL configs so the CSV column set stays aligned.
        # ---------------------------------------------------------------------


        incomplete_simpoints = set()
        if skip_incomplete:
            for (conf, suite, subsuite, workload, cid) in tasks:
                if not self.sp_is_complete(suite, subsuite, workload, conf, cid, experiment_name, simulations_path):
                    incomplete_simpoints.add((suite, subsuite, workload, str(cid)))

            if incomplete_simpoints:
                print("WARN: The following simpoints were not found, and will be ignored for ALL configs:")
                pretty = [f"{s}/{ss}/{w}:{cid}" for (s, ss, w, cid) in sorted(incomplete_simpoints)]
                print(f"WARN: {', '.join(pretty)}")

            tasks = [t for t in tasks if (t[1], t[2], t[3], str(t[4])) not in incomplete_simpoints]

        if not tasks:
            raise RuntimeError("Descriptor expands to zero simulation runs; nothing to collect.")
        # ---------------------------------------------------------------------
        # Step 4: Parse the first simpoint to establish:
        #   - `known_stats`: the canonical stat row order for the entire CSV
        #   - `groups0`: a per-stat group id (used for distribution postprocessing)
        # ---------------------------------------------------------------------


        # Read first simpoint to determine stat order + groups
        config0, suite0, subsuite0, workload0, cid0 = tasks[0]
        sim_dir0 = f"{simulations_path}{experiment_name}/{config0}/{suite0}/{subsuite0}/{workload0}/{cid0}/"
        known_stats, vals0, groups0 = self.load_simpoint(sim_dir0, return_stats_and_data=True)

        n_stats = len(known_stats)
        n_cols = len(tasks)
        # ---------------------------------------------------------------------
        # Step 5: Allocate the main (n_stats x n_tasks) numeric matrix.
        #
        # - rows: stat index in `known_stats`
        # - cols: task index in `tasks`
        #
        # When jobs>1 we allocate this in SharedMemory so workers can fill columns in-place.
        # ---------------------------------------------------------------------


        # Allocate the main stats matrix.
        # If collecting in parallel, back it with shared memory so worker processes can fill columns in-place.
        shm = None
        nan = float("nan")
        jobs_int = 1
        try:
            jobs_int = int(jobs) if jobs is not None else 1
        except Exception:
            jobs_int = 1

        if jobs_int > 1 and n_cols > 1:
            try:
                shm = shared_memory.SharedMemory(
                    create=True,
                    size=n_stats * n_cols * np.dtype(np.float64).itemsize,
                    track=False,
                )
            except TypeError:
                # Python < 3.13
                shm = shared_memory.SharedMemory(
                    create=True,
                    size=n_stats * n_cols * np.dtype(np.float64).itemsize,
                )

            # Ensure shared memory is cleaned up even if we crash before normal cleanup.
            atexit.register(_cleanup_shared_memory, shm.name)
            values = np.ndarray((n_stats, n_cols), dtype=np.float64, buffer=shm.buf)
            values.fill(nan)
        else:
            values = np.full((n_stats, n_cols), nan, dtype=np.float64)

        values[:, 0] = vals0
        # Build a stat-name -> row-index mapping for O(1) placement while parsing each simpoint.


        stat_to_row = {s: i for i, s in enumerate(known_stats)}

        # Build per-component schemas from simpoint0 to accelerate subsequent simpoint parsing
        component_schemas = self._build_component_csv_schemas(sim_dir0, stat_to_row)
        # ---------------------------------------------------------------------
        # Step 6: Prepare metadata rows (written after the stat rows).
        #
        # These rows let downstream tools know, for every column, which experiment/config/workload
        # it corresponds to and what simpoint weight to use for aggregation.
        # ---------------------------------------------------------------------



        meta_names = ["Experiment", "Architecture", "Configuration", "Workload", "Segment Id", "Cluster Id", "Weight"]
        meta = {k: [None] * n_cols for k in meta_names}

        # Cache simpoint (weight, segment_id) lookup per (suite, subsuite, workload)
        sp_cache = {}

        w0, s0 = self.get_simpoint_info(cid0, workload0, subsuite0, suite0, top_simpoint_only, workloads_data=workloads_data, sp_cache=sp_cache)
        meta["Experiment"][0] = experiment_name
        meta["Architecture"][0] = architecture
        meta["Configuration"][0] = config0
        meta["Workload"][0] = f"{suite0}/{subsuite0}/{workload0}"
        meta["Segment Id"][0] = s0
        meta["Cluster Id"][0] = cid0
        meta["Weight"][0] = w0

        # Precompute simpoint directory paths for all columns and fill metadata (cheap, keep in main process).
        sim_dirs = [None] * n_cols
        sim_dirs[0] = sim_dir0

        for j, (conf, suite, subsuite, workload, cid) in enumerate(tasks[1:], start=1):
            sim_dirs[j] = f"{simulations_path}{experiment_name}/{conf}/{suite}/{subsuite}/{workload}/{cid}/"

            w, s = self.get_simpoint_info(
                cid,
                workload,
                subsuite,
                suite,
                top_simpoint_only,
                workloads_data=workloads_data,
                sp_cache=sp_cache,
            )
            meta["Experiment"][j] = experiment_name
            meta["Architecture"][j] = architecture
            meta["Configuration"][j] = conf
            meta["Workload"][j] = f"{suite}/{subsuite}/{workload}"
            meta["Segment Id"][j] = s
            meta["Cluster Id"][j] = cid
            meta["Weight"][j] = w
        # ---------------------------------------------------------------------
        # Step 7: Fill the matrix with all remaining simpoint columns.
        #
        # Serial mode: parse each simpoint in the main process.
        # Parallel mode: worker processes attach to SharedMemory and fill columns concurrently.
        # ---------------------------------------------------------------------


        # Fill values matrix: serial or parallel
        if jobs_int > 1 and n_cols > 1:
            work_items = [(j, sim_dirs[j]) for j in range(1, n_cols)]
            n_workers = min(jobs_int, len(work_items))
            if n_workers < 1:
                n_workers = 1

            # Prefer fork on Linux for lower overhead; fall back otherwise.
            try:
                ctx = multiprocessing.get_context("fork")
            except Exception:
                ctx = multiprocessing.get_context()

            chunksize = max(1, len(work_items) // (n_workers * 4))

            with ctx.Pool(
                processes=n_workers,
                initializer=_mp_init_shared_collector,
                initargs=(
                    shm.name,
                    values.shape,
                    values.dtype.str,
                    stat_to_row,
                    component_schemas,
                    True,   # load_ramulator
                    True,   # ignore_duplicates
                ),
            ) as pool:
                errors = []
                for res in pool.imap_unordered(_mp_fill_simpoint_column, work_items, chunksize=chunksize):
                    if res is not None:
                        errors.append(res)

            if errors:
                # Avoid leaking shared memory if we error out before writing the CSV.
                if shm is not None:
                    try:
                        shm.close()
                        shm.unlink()
                    except Exception:
                        pass

                col_idx, sim_dir, err = errors[0]
                raise RuntimeError(f"Failed to collect simpoint column {col_idx} from {sim_dir}: {err}")
        else:
            for j in range(1, n_cols):
                self.load_simpoint_into(
                    sim_dirs[j],
                    stat_to_row,
                    values[:, j],
                    schema_cache=component_schemas,
                    clear_column=False,
                )

        colnames = [f"{conf} {suite}/{subsuite}/{workload} {cid}" for (conf, suite, subsuite, workload, cid) in tasks]


        Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        # ---------------------------------------------------------------------
        # Step 8: Write the base CSV (stats + metadata).
        #
        # For large experiments the write itself can be expensive. If we already have SharedMemory,
        # we can write the stat rows in parallel by row-chunking and then concatenating part files.
        # ---------------------------------------------------------------------


        # Parallel chunked base CSV writing (Option 1):
        # - Workers write stat rows into ordered part files.
        # - Main process writes header once, concatenates parts in order, then writes metadata rows.
        #
        # This avoids contended locking on a single writer while preserving row order and file format.
        _use_parallel_write = (jobs_int > 1 and n_cols > 1 and shm is not None and n_stats > 0)
        if _use_parallel_write:
            import tempfile
            import shutil
            import os

            tmp_base = os.environ.get("TMPDIR") or None
            parts_dir = tempfile.mkdtemp(prefix="scarab_csv_parts_", dir=tmp_base)
            part_paths = []
            try:
                # Choose worker count and chunking.
                max_procs = os.cpu_count() or jobs_int
                n_workers_w = max(1, min(jobs_int, max_procs, n_stats))

                # Target ~8 chunks per worker for better load balance.
                target_chunks = max(1, n_workers_w * 8)
                chunk_size = max(1, (n_stats + target_chunks - 1) // target_chunks)

                work = []
                part_idx = 0
                for start_i in range(0, n_stats, chunk_size):
                    end_i = min(n_stats, start_i + chunk_size)
                    part_path = os.path.join(parts_dir, f"part_{part_idx:05d}.csv")
                    part_paths.append(part_path)
                    work.append((part_path, start_i, end_i))
                    part_idx += 1

                try:
                    ctx_w = multiprocessing.get_context("fork")
                except Exception:
                    ctx_w = multiprocessing.get_context()

                errors = []
                with ctx_w.Pool(
                    processes=min(n_workers_w, len(work)),
                    initializer=_mp_init_csv_part_writer,
                    initargs=(shm.name, values.shape, values.dtype.str, known_stats, groups0),
                ) as pool:
                    for res in pool.imap_unordered(_mp_write_csv_part, work, chunksize=1):
                        if res is not None:
                            errors.append(res)

                if errors:
                    raise RuntimeError(f"Failed to write CSV part: {errors[0]}")

                # Assemble the final file in-order.
                with open(outfile, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["stats", "write_protect", "groups", *colnames])

                with open(outfile, "a", newline="") as f:
                    for p in part_paths:
                        with open(p, "r", newline="") as pf:
                            shutil.copyfileobj(pf, f, length=1024 * 1024)

                    w = csv.writer(f)
                    for meta_name in meta_names:
                        w.writerow([meta_name, True, 0, *meta[meta_name]])

            finally:
                try:
                    shutil.rmtree(parts_dir)
                except Exception:
                    pass
        else:
            with open(outfile, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["stats", "write_protect", "groups", *colnames])

                for i, stat_name in enumerate(known_stats):
                    w.writerow([stat_name, True, groups0[i], *values[i, :].tolist()])

                for meta_name in meta_names:
                    w.writerow([meta_name, True, 0, *meta[meta_name]])


        # Shared-memory backing (if any) is kept alive while postprocessing uses `values`.
        def _cleanup_shm():
            if shm is not None:
                try:
                    shm.close()
                    shm.unlink()
                except Exception:
                    pass
        # ---------------------------------------------------------------------
        # Step 9 (optional): Postprocess.
        #
        # If disabled, we are done after the base CSV write.
        # If enabled, we append:
        #   - derived IPC rows
        #   - distribution statistics rows
        # directly to the same output file.
        # ---------------------------------------------------------------------


        if not postprocess:
            _cleanup_shm()
            return outfile

        full_path = outfile

        # Fast postprocess path: avoid re-reading the base CSV into pandas.
        # We already have all base stats (`known_stats`, `groups0`, `values`) and metadata (`meta`) in memory.
        # When enabled, derived/distribution stats are computed directly on NumPy arrays and the postprocessed
        # CSV is written to `full_path`.
        #
        # IMPORTANT: We match the legacy pandas `to_csv(index=False)` NaN representation by emitting empty
        # fields (not the string 'nan') for NaNs in the postprocessed CSV.

        def _fmt_any(x):
            if x is None:
                return ""
            # NaN check that works for Python floats and numpy scalars
            try:
                if isinstance(x, (float, np.floating)) and x != x:
                    return ""
            except Exception:
                pass
            return x

        def _fmt_num(x):
            # NaN -> empty string, else return numeric as-is
            try:
                return "" if x != x else x
            except Exception:
                return x

        header = ["stats", "write_protect", "groups", *colnames]

        # Precompute weights once for derived stats (same as legacy derive_stat uses Weight row).
        weights = np.asarray(meta["Weight"], dtype=float)

        # Precompute distribution row membership once (group -> row indices), matching legacy suffix logic.
        # Note: distribution stats only apply to base stats (groups != 0). Metadata/derived rows are group 0.
        group_to_total_rows = {}
        group_to_count_rows = {}
        for i, (g, stat_name) in enumerate(zip(groups0, known_stats)):
            if g == 0:
                continue
            if stat_name.endswith("_total_count"):
                group_to_total_rows.setdefault(int(g), []).append(i)
            elif stat_name.endswith("_count") and not stat_name.endswith("_total_count"):
                group_to_count_rows.setdefault(int(g), []).append(i)

        import time
        start_post = time.time()

        Path(full_path).parent.mkdir(parents=True, exist_ok=True)

        # and append derived/distribution rows (this saves ~one full base CSV write).
        with open(full_path, "a", newline="", buffering=512 * 512) as f_out:
            w_out = csv.writer(f_out)

            # 3) Derived IPC rows (matching legacy `derive_stat(..., pre_agg=True)` behavior)
            def _write_weighted_ratio_row(out_stat: str, num_stat: str, den_stat: str) -> None:
                if num_stat not in stat_to_row or den_stat not in stat_to_row:
                    return

                num = values[stat_to_row[num_stat], :] * weights
                den = values[stat_to_row[den_stat], :] * weights
                # Aggregate weighted sums per "setup" (exact prefix without last token).
                # This avoids substring collisions (e.g., "gcc" vs "531.gcc") and is O(n) via bincount.
                columns = colnames
                prefixes = np.array([" ".join(lbl.split(" ")[:-1]) for lbl in columns], dtype=object)
                _, inv = np.unique(prefixes, return_inverse=True)

                num_w = np.nan_to_num(num, nan=0.0)
                den_w = np.nan_to_num(den, nan=0.0)
                num_sum = np.bincount(inv, weights=num_w)
                den_sum = np.bincount(inv, weights=den_w)

                n = num_sum[inv]
                d = den_sum[inv]
                out = np.full(n_cols, np.nan, dtype=float)
                good = d != 0
                out[good] = n[good] / d[good]

                w_out.writerow([out_stat, True, 0, *(_fmt_num(x) for x in out)])

            _write_weighted_ratio_row("IPC_total", "Cumulative_Instructions", "Cumulative_Cycles")
            _write_weighted_ratio_row("IPC", "Periodic_Instructions", "Periodic_Cycles")

            # 4) Distribution stats (vectorized NumPy implementation of legacy calculate_distribution_stats)
            groups = set(int(g) for g in set(groups0))
            if 0 in groups:
                groups.remove(0)

            errs = 0
            num_groups = len(groups)
            print(f"INFO: Calculate distribution stats for {num_groups} groups")
            start_dist = time.time()

            for group in groups:
                # Legacy prints "group {group}/{num_groups}" (group is id, not an index); preserve.
                #print(f"INFO: group {group}/{num_groups}")

                total_rows = group_to_total_rows.get(group, [])
                count_rows = group_to_count_rows.get(group, [])

                # Legacy expects count and total_count sizes to match; otherwise warn and continue.
                if len(total_rows) != len(count_rows):
                    errs += 1
                    continue

                # Nothing to do for empty groups
                if len(total_rows) == 0 and len(count_rows) == 0:
                    continue

                A_total = values[total_rows, :]
                A_count = values[count_rows, :]

                total_sums = np.nansum(A_total, axis=0)
                count_sums = np.nansum(A_count, axis=0)

                # Legacy intent: if any column sum is zero, emit NaN rows for this group and skip.
                # (Original code had a use-before-definition bug here; this matches the intended behavior.)
                if (total_sums == 0).any() or (count_sums == 0).any():
                    new_stats = [
                        f"group_{group}_total_mean",
                        f"group_{group}_mean",
                        f"group_{group}_total_stddev",
                        f"group_{group}_stddev",
                    ]
                    for ridx in total_rows:
                        new_stats.append(f"{known_stats[ridx]}_pct")
                    for ridx in count_rows:
                        new_stats.append(f"{known_stats[ridx]}_pct")

                    empty_vals = [""] * n_cols
                    for stat in new_stats:
                        w_out.writerow([stat, True, 0, *empty_vals])
                    continue

                # Means: pandas divides by number of rows (not number of non-NaNs)
                total_means = total_sums / len(total_rows)
                count_means = count_sums / len(count_rows)

                # Stddev: match legacy ((df-mean)^2).sum(skipna) / N, then sqrt
                total_stddev = np.sqrt(np.nansum((A_total - total_means[None, :]) ** 2, axis=0) / len(total_rows))
                count_stddev = np.sqrt(np.nansum((A_count - count_means[None, :]) ** 2, axis=0) / len(count_rows))

                w_out.writerow([f"group_{group}_total_mean", True, 0, *(_fmt_num(x) for x in total_means)])
                w_out.writerow([f"group_{group}_mean", True, 0, *(_fmt_num(x) for x in count_means)])
                w_out.writerow([f"group_{group}_total_stddev", True, 0, *(_fmt_num(x) for x in total_stddev)])
                w_out.writerow([f"group_{group}_stddev", True, 0, *(_fmt_num(x) for x in count_stddev)])

                # Percentages (df / sums), preserving stat row order within each group
                total_pct = A_total / total_sums[None, :]
                for k, ridx in enumerate(total_rows):
                    w_out.writerow([f"{known_stats[ridx]}_pct", True, 0, *(_fmt_num(x) for x in total_pct[k, :])])

                count_pct = A_count / count_sums[None, :]
                for k, ridx in enumerate(count_rows):
                    w_out.writerow([f"{known_stats[ridx]}_pct", True, 0, *(_fmt_num(x) for x in count_pct[k, :])])

            print(f"Took: {time.time() - start_dist} seconds")
            if errs != 0:
                print("WARN: Distribution size and number of x_count + x_total_count stats is not equal.")

        print(f"INFO: Postprocess took: {time.time() - start_post} seconds")
        _cleanup_shm()
        return full_path

    # Plot one stat across multiple workloads

    def plot_workloads_speedup (self, experiment: Experiment, stats: List[str], workloads: List[str],
                                configs: List[str], speedup_baseline: str = None, title: str = "", x_label: str = "",
                                y_label: str= "", logscale: bool = False, bar_width:float = 0.2,
                                bar_spacing:float = 0.05, workload_spacing:float = 0.3, average: bool = False,
                                colors = None, plot_name = None, ylim = None):
        """Plot percentage speedup over a baseline across workloads.

        Steps:
          1) Aggregate each requested stat for each (config, workload) using simpoint weights.
          2) Compute speedup relative to `speedup_baseline` (baseline value / config value).
          3) Plot a grouped bar chart, optionally with an average bar.
        """
        if len(stats) > 1:
            print("WARN: This API is for only one stats.")
            print("INFO: Only plot the first stat, ignoring the rest from the provided list")

        stat = stats[0]
        # Get all data with structure all_data[f"{config} {wl} {stat}"]
        configs_to_load = configs + [speedup_baseline]
        all_data = experiment.retrieve_stats(configs_to_load, stats, workloads)
        workloads_to_plot = workloads.copy()

        mean_type = 1 # geomean
        stat_tokens = stat.split('_')
        if stat_tokens[-1] != 'pct':
            mean_type = 0 # arithmetic mean

        data_to_plot = {}
        for conf in configs:
            if mean_type == 1:
                avg_config = 1.0
            else:
                avg_config = 0.0
            data_config = []
            for wl_number, wl in enumerate(workloads_to_plot):
                data = all_data[f"{conf} {wl} {stat}"]/all_data[f"{speedup_baseline} {wl} {stat}"]
                data_config.append(100.0*data - 100.0)
                if mean_type == 1:
                    avg_config *= data
                else:
                    avg_config += data

            num_workloads = len(workloads_to_plot)
            if mean_type == 1:
                avg_config = avg_config**(num_workloads**-1)
            else:
                avg_config = avg_config/num_workloads
            if average:
                data_config.append(100.0*avg_config - 100.0)
            data_to_plot[conf] = data_config

        if average:
            benchmarks_to_plot = workloads_to_plot + ["Avg"]
        else:
            benchmarks_to_plot = workloads_to_plot

        fig, ax = plt.subplots(figsize=(6+len(benchmarks_to_plot)*((bar_spacing+bar_width)*len(configs)), 5))

        if colors == None:
            colors = ['#8ec1da', '#cde1ec', '#ededed', '#f6d6c2', '#d47264', '#800000', '#911eb4', '#4363d8', '#f58231', '#3cb44b', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#e6beff', '#e6194b', '#000075', '#800000', '#9a6324', '#808080', '#ffffff', '#000000']

        num_configs = len(configs_to_load)
        ind = np.arange(len(benchmarks_to_plot))
        start_id = -int(num_configs/2)
        for conf_number, config in enumerate(configs):
            ax.bar(ind + (start_id+conf_number)*bar_width, data_to_plot[config], width=bar_width, fill=True, color=colors[conf_number], edgecolor='black', label=config)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xticks(ind)
        ax.set_xticklabels(benchmarks_to_plot, rotation = 27, ha='right')
        ax.grid('x');
        ax.grid('y')
        if ylim != None:
            ax.set_ylim(ylim)
        legend = ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.title(title)
        plt.tight_layout()

        if plot_name == None:
            plt.show()
        else:
            plt.savefig(
                plot_name,
                bbox_inches="tight",
                bbox_extra_artists=(legend,),
            )


    # Plot graph comparing different configs
    # Aggregate simpoints
    # Params:
    # experiment file
    # List of stats you are interested in
    # List of configs you are interested in
    # Workloads to plot
    # Baseline config
    # Should plots be each stat individually, or proportion of each stat (Overlayed bar graphs, for percentages that sum to 1)
    # Plot on logarithmic scale

    # Plot one stat across multiple workloads
    def plot_workloads (self, experiment: Experiment, stats: List[str], workloads: List[str], 
                        configs: List[str], title: str = "", x_label: str = "",
                        y_label: str= "", logscale: bool = False, bar_width:float = 0.2,
                        bar_spacing:float = 0.05, workload_spacing:float = 0.3, average: bool = False,
                        colors = None, plot_name = None, ylim = None):
        """Plot one or more stats across multiple workloads (grouped bars).

        Steps:
          1) Use `Experiment.retrieve_stats(..., aggregation_level="Workload")` to get weighted values.
          2) Reformat into a per-config series.
          3) Render a grouped bar chart with optional log scale / colors / y-limits.
        """
        if len(stats) > 1:
            print("WARN: This API is for only one stats.")
            print("INFO: Only plot the first stat, ignoring the rest from the provided list")

        stat = stats[0]
        # Get all data with structure all_data[f"{config} {wl} {stat}"]

        configs_to_load = configs
        all_data = experiment.retrieve_stats(configs_to_load, stats, workloads)
        print(all_data)
        if all_data is None:
            print("ERROR: retrieve_stats returned None. This means either:")
            print("1. The requested workloads don't exist in the dataframe")
            print("2. The requested configurations don't exist in the dataframe")
            print("3. The dataframe doesn't contain the expected 'Workload' or 'Configuration' rows")
            return

        workloads_to_plot = workloads.copy()

        mean_type = 1 # geomean
        stat_tokens = stat.split('_')
        if stat_tokens[-1] != 'pct':
            mean_type = 0 # arithmetic mean

        data_to_plot = {}
        for conf in configs:
            if mean_type == 1:
                avg_config = 1.0
            else:
                avg_config = 0.0
            data_config = []
            for wl_number, wl in enumerate(workloads_to_plot):
                data = all_data[f"{conf} {wl} {stat}"]
                data_config.append(data)
                if mean_type == 1:
                    avg_config *= data
                else:
                    avg_config += data

            num_workloads = len(workloads_to_plot)
            if mean_type == 1:
                avg_config = avg_config**(num_workloads**-1)
            else:
                avg_config = avg_config/num_workloads
            if average:
                data_config.append(avg_config)
            data_to_plot[conf] = data_config

        if average:
            benchmarks_to_plot = workloads_to_plot + ["Avg"]
        else:
            benchmarks_to_plot = workloads_to_plot

        fig, ax = plt.subplots(figsize=(6+len(benchmarks_to_plot)*((bar_spacing+bar_width)*len(configs)), 5))

        if colors == None:
            colors = ['#8ec1da', '#cde1ec', '#ededed', '#f6d6c2', '#d47264', '#800000', '#911eb4', '#4363d8', '#f58231', '#3cb44b', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#e6beff', '#e6194b', '#000075', '#800000', '#9a6324', '#808080', '#ffffff', '#000000']

        num_configs = len(configs_to_load)
        ind = np.arange(len(benchmarks_to_plot))
        start_id = -int(num_configs/2)
        for conf_number, config in enumerate(configs):
            ax.bar(ind + (start_id+conf_number)*bar_width, data_to_plot[config], width=bar_width, fill=True, color=colors[conf_number], edgecolor='black', label=config)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xticks(ind)
        ax.set_xticklabels(benchmarks_to_plot, rotation = 27, ha='right')
        ax.grid('x');
        ax.grid('y')
        if ylim != None:
            ax.set_ylim(ylim)
        legend = ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.title(title)
        plt.tight_layout()

        if plot_name == None:
            plt.show()
        else:
            plt.savefig(
                plot_name,
                bbox_inches="tight",
                bbox_extra_artists=(legend,),
            )


    def plot_speedups (self, experiment: Experiment, stats: List[str], workloads: List[str], 
                        configs: List[str], speedup_baseline: str = None, title: str = "Default Title", x_label: str = "", 
                        y_label: str = "", logscale: bool = False, bar_width:float = 0.35, 
                        bar_spacing:float = 0.05, workload_spacing:float = 0.3, average: bool = False, 
                        colors = None, plot_name = None):
        """Plot speedups for a set of configs vs a baseline across workloads.

        This is similar to `plot_workloads_speedup`, but supports multiple stats and a slightly different
        layout/labeling for common “speedup vs baseline” reporting.
        """

        # Get all data with structure all_data[f"{config} {wl} {stat}"]
        configs_to_load = configs + [speedup_baseline]
        all_data = experiment.retrieve_stats(configs_to_load, stats, workloads)
        if all_data is None:
            print("ERROR: retrieve_stats returned None. This means either:")
            print("1. The requested workloads don't exist in the dataframe")
            print("2. The requested configurations don't exist in the dataframe")
            print("3. The dataframe doesn't contain the expected 'Workload' or 'Configuration' rows")
            return

        workloads_to_plot = workloads.copy()

        averages = {}
        if average and speedup_baseline == None:
            for stat in stats:
                all_conf_wl_data = []

                for conf in configs:
                    for wl in workloads:
                        all_conf_wl_data.append(all_data[f"{conf} {wl} {stat}"])

                averages[stat] = reduce(lambda x,y: x*y, all_conf_wl_data) ** (1/len(all_conf_wl_data)) 

            workloads_to_plot.append("average")

        workload_locations = []

        plt.figure(figsize=(6+len(workloads_to_plot), 8))
        ax = plt.axes()

        hatches = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']

        bar_offset = 0

        # TODO: Refactor to remove a loop, and ask how average should work
        if average and speedup_baseline != None:
            print("WARN: Average and a speedup baseline is currently unsupported.")
            print("INFO: Ignoring average parameter")

        # For each stat
        for wl_number, wl in enumerate(workloads_to_plot):
            workload_locations.append(bar_offset)
            for stat_number, stat in enumerate(stats):
                for conf_number, config in enumerate(configs):
                    if wl == "average" and conf_number != 0:
                        continue

                    # Plot each workload's stat as bar graph
                    if wl != "average":
                        data = all_data[f"{config} {wl} {stat}"]
                    else:
                        data = averages[stat]

                    if speedup_baseline != None:
                        baseline_data = all_data[f"{speedup_baseline} {wl} {stat}"]
                        data = data/baseline_data


                    if colors == None:
                        color_map = plt.get_cmap("Paired")
                        color = color_map((stat_number*(1/11))%1)
                    else:
                        color = colors[stat_number%len(colors)]

                    b = ax.bar(bar_offset, data, [bar_width], color=color, hatch=hatches[conf_number])

                    if stat_number == 0 and wl_number == 0:
                        plt.text(1.02, conf_number*0.05, f"{config}: {hatches[conf_number]}", transform=ax.transAxes)

                    if wl_number == 0 and conf_number == 0: b.set_label(stat)

                    bar_offset += bar_width + bar_spacing

            bar_offset += workload_spacing - bar_spacing



        if average:
            plt.text(1.02, len(configs)*0.05, f"average sums across all configs", transform=ax.transAxes)

        x_ticks = [f"{wl}" for wl in workloads_to_plot]
        if len(configs) == 1: x_ticks = workloads_to_plot
        ax.set_xticks(workload_locations, x_ticks)

        plt.legend(loc="center left", bbox_to_anchor=(1,0.5))

        if logscale: plt.yscale("log")

        if y_label == "":
            y_label = "Speedup" if speedup_baseline != None else "Count"

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        if plot_name == None:
            plt.show()
        else: plt.savefig(plot_name, bbox_inches="tight")

    # Plot multiple stats across simpoints
    def plot_simpoints (self, experiment: Experiment, stats: List[str], workload: str, 
                        configs: List[str], simpoints: List[str] = None, speedup_baseline: str = None, 
                        title: str = "Default Title", x_label: str = "", y_label: str = "", 
                        logscale: bool = False, bar_width:float = 0.35, bar_spacing:float = 0.05, workload_spacing:float = 0.3, 
                        average: bool = False, colors = None, plot_name = None, label_fontsize = "medium",
                        label_rotation = 0):
        """Plot raw per-simpoint values for a single workload.

        Steps:
          1) Select the simpoint columns for the given (config, workload).
          2) Pull raw (unweighted) values for each simpoint.
          3) Plot them so outliers/variance across simpoints are visible.
        """

        if not workload in experiment.get_workloads():
            print(f"ERR: {workload} not found in experiment workload")
            print(experiment.get_workloads())
            return

        # Get all data with structure all_data[f"{config} {wl} {simpoint} {stat}"]
        configs_to_load = configs + [speedup_baseline]
        all_data = experiment.retrieve_stats(configs_to_load, stats, [workload], aggregation_level="Simpoint", simpoints=simpoints)

        plt.figure(figsize=(6+len(stat_files),8))
        ax = plt.axes()

        xticks = []

        # For each stat
        for x_offset, stat in enumerate(stats):
            # Plot each workload's stat as bar graph

            # TODO: Doesn't work with configs, workloads, or simpoints with spaces in the names
            all = [(float(val), key) for key, val in all_data.items() if " ".join(key.split(" ")[3:]) == stat and key.split(" ")[0] != speedup_baseline]
            data = list(map(lambda x:x[0], all))
            keys = list(map(lambda x:x[1], all))

            num_workloads = len(data)
            if average: num_workloads += 1
            workload_locations = np.arange(num_workloads) * ((bar_width * len(stats) + bar_spacing * (len(stats) - 1)) + workload_spacing)

            if speedup_baseline != None:
                baseline_data = [val for key, val in all_data.items() if key.split(" ")[-1] == stat and key.split(" ")[0] == speedup_baseline]
                if 0 in baseline_data:
                    print("ERR: Found 0 in baseline data. Bar will be set to 0")
                    errors = [key for key, val in all_data.items() if key.split(" ")[-1] == stat and key.split(" ")[0] == speedup_baseline and val == 0]
                    print("Erroneous stat in baseline:", ", ".join(errors))
                    plt.clf()
                    return

                data = [test/baseline if baseline != 0 else 0 for test, baseline in zip(data, baseline_data)]

            if average:
                data.append(reduce(lambda x,y: x*y, data) ** (1/len(data)))

            if x_offset != -1:
                for i, loc in enumerate(workload_locations):
                    if average and i == len(workload_locations) - 1:
                        xticks.append((loc, "avg"))
                        continue
                    xticks.append((loc, keys[i].split(" ")[2]))

            if colors == None:
                color_map = plt.get_cmap("Paired")
                color = color_map((x_offset*(1/12))%1)
            else:
                color = colors[x_offset%len(colors)]

            b = plt.bar(workload_locations + x_offset*(bar_width + bar_spacing), data, [bar_width] * num_workloads, 
                        color=color)
            b.set_label(stat)

            for i, loc in enumerate(workload_locations + x_offset*(bar_width + bar_spacing)):
                length = len(f"{data[i]:3.3}")
                plt.text(loc-(bar_width*(length*0.25-0.25)), data[i], f"{data[i]:3.3}", transform=ax.transData, fontsize=label_fontsize, rotation=label_rotation)

            if x_offset == 0:
                locations = x_offset*(bar_width + bar_spacing)
                simpoints = int(len(all_data)/((len(configs) if not average else len(configs) + 1) * len(stats)))
                for i in range(len(configs)):
                    loc = workload_locations[i*simpoints]
                    plt.text(loc-bar_width*0.5, -0.15, f"{configs[i]}", transform=ax.transData)

        plt.legend(loc="center left", bbox_to_anchor=(1,0.5))

        plt.xticks(get_elem(xticks, 0), get_elem(xticks, 1))

        if logscale: plt.yscale("log")

        if y_label == "":
            y_label = "Speedup" if speedup_baseline != None else "Count"

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        if plot_name == None:
            plt.show()
        else: plt.savefig(plot_name, bbox_inches="tight")

    # Plot multiple stats across simpoints
    def plot_configs (self, experiment: Experiment, stats: List[str], workloads: List[str], 
                        configs: List[str], speedup_baseline: str = None, 
                        title: str = "Default Title", x_label: str = "", y_label: str = "", 
                        logscale: bool = False, bar_width:float = 0.35, bar_spacing:float = 0.05, workload_spacing:float = 0.3, 
                        average: bool = False, colors = None, plot_name = None):
        """Plot a stat aggregated at the config level (geomean across workloads).

        Steps:
          1) Aggregate each (config, workload) with weights.
          2) Combine workloads into a single value per config (geomean by default).
          3) Plot one bar per config.
        """

        all_data = experiment.retrieve_stats(configs, stats, workloads, aggregation_level="Config")
        if speedup_baseline != None: baseline_data = experiment.retrieve_stats([speedup_baseline], stats, workloads, aggregation_level="Config")

        plt.figure(figsize=(6+num_workloads,8))

        # For each stat
        for x_offset, stat in enumerate(stats):
            # Plot each workload's stat as bar graph
            data = [val for key, val in all_data.items() if stat in key]
            num_workloads = len(configs)
            if average: num_workloads += 1
            workload_locations = np.arange(num_workloads) * ((bar_width * len(stats) + bar_spacing * (len(stats) - 1)) + workload_spacing)

            if speedup_baseline != None:
                baseline_data = [val for key, val in baseline_data.items() if stat in key]
                if 0 in baseline_data:
                    print("ERR: Found 0 in baseline data. Bar will be set to 0")
                    errors = [key for key, val in all_data.items() if key.split(" ")[-1] == stat and key.split(" ")[0] == speedup_baseline and val == 0]
                    print("Erroneous stat in baseline:", ", ".join(errors))
                    plt.clf()
                    return

                data = [test/baseline if baseline != 0 else 0 for test, baseline in zip(data, baseline_data)]

            if average:
                data.append(reduce(lambda x,y: x*y, data) ** (1/len(data)))

            print(workload_locations, data)

            if colors == None:
                color_map = plt.get_cmap("Paired")
                color = color_map((x_offset*(1/12))%1)
            else:
                color = colors[x_offset%len(colors)]

            b = plt.bar(workload_locations + x_offset*(bar_width + bar_spacing), data, [bar_width] * num_workloads, 
                        color=color)
            b.set_label(stat)

        if average: workloads.append("Average")

        plt.legend(loc="center left", bbox_to_anchor=(1,0.5))

        if logscale: plt.yscale("log")

        if y_label == "":
            y_label = "Speedup" if speedup_baseline != None else "Count"

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        if plot_name == None:
            plt.show()
        else: plt.savefig(plot_name, bbox_inches="tight")


    # Plot simpoints within workload
    # Don't agregate
    # def plot_simpoints (self, experiment, stats, configs, workload)

    # Plot stacked bars. List of
    def plot_stacked (self, experiment: Experiment, stats: List[str], workloads: List[str],
                      configs: List[str], title: str = "", y_label: str= "",
                      bar_width:float = 0.2, bar_spacing:float = 0.05, workload_spacing:float = 0.3,
                      colors = None, plot_name = None):
        """Plot stacked bars for a set of stats per workload/config.

        Typical use: break a total into components (e.g., cycle breakdown).
        """

        # Get all data with structure all_data[stat][config][workload]
        all_data = experiment.retrieve_stats(configs, stats, workloads)
        #all_data = {stat:experiment.get_stat(stat, aggregate = True) for stat in stats}

        num_workloads = len(workloads)
        workload_locations = np.arange(num_workloads) * ((bar_width * len(configs) + bar_spacing * (len(configs) - 1)) + workload_spacing)

        fig, ax = plt.subplots(figsize=(6 + num_workloads, 8))
        fig.subplots_adjust(right=0.78)  # leave room for legends anchored outside the axes

        hatches = ['/', '\\', '.', '-', '+', 'x', 'o', 'O', '*']
        patch_hatches = []
        if colors == None:
            colors = ['#cecece', '#cde1ec', '#8ec1da', '#2066a8', '#a559aa', '#59a89c', '#f0c571', '#e02b35', '#082a54', '#ededed', '#f6d6c2', '#d47264', '#800000', '#911eb4', '#4363d8', '#f58231', '#3cb44b', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#e6beff', '#e6194b', '#000075', '#9a6324', '#808080', '#ffffff', '#000000']

        # For each stat
        for x_offset, config in enumerate(configs):

            offsets = np.array([0.0] * len(workloads))
            hatch = hatches[x_offset % len(hatches)]
            patch_hatch = mpatches.Patch(facecolor='beige', hatch=hatch, edgecolor="darkgrey", label=config)
            patch_hatches.append(patch_hatch)

            for i, stat in enumerate(stats):
                # Plot each workload's stat as bar graph
                #data = np.array([all_data[stat][config][wl]/totals[wl] for wl in workloads])
                data = np.array([all_data[f"{config} {wl} {stat}"] for wl in workloads])
                color = colors[i%len(colors)]

                if x_offset > len(hatches):
                    print("WARN: Too many configs for unique configuration labels")
                b = ax.bar(workload_locations + x_offset*(bar_width + bar_spacing), data, [bar_width] * num_workloads,
                        bottom=offsets, edgecolor="darkgrey", color = color, hatch=hatch)

                if x_offset == 0: b.set_label(f"{stat}")

                offsets += data

        plt.title(title)
        plt.ylabel(y_label)
        plt.xlabel("Workloads")
        plt.xticks(workload_locations, workloads)
        legend_1 = plt.legend(loc='center left', bbox_to_anchor=(1,0.8))
        legend_2 = plt.legend(handles=patch_hatches, bbox_to_anchor=(1,0.4))
        fig.add_artist(legend_1)

        if plot_name == None:
            plt.show()
        else:
            plt.savefig(
                plot_name,
                bbox_inches="tight",
                bbox_extra_artists=(legend_1, legend_2),
            )


    # Plot stacked bars. List of
    def plot_stacked_fraction (self, experiment: Experiment, stats: List[str], workloads: List[str],
                      configs: List[str], title: str = "",
                      bar_width:float = 0.2, bar_spacing:float = 0.05, workload_spacing:float = 0.3,
                      colors = None, plot_name = None):
        """Plot stacked fractions (normalized stacked bars) for a set of stats.

        Each stacked bar sums to 1.0 (or 100%) to show composition rather than absolute magnitude.
        """

        # Get all data with structure all_data[stat][config][workload]
        all_data = experiment.retrieve_stats(configs, stats, workloads)
        #all_data = {stat:experiment.get_stat(stat, aggregate = True) for stat in stats}

        num_workloads = len(workloads)
        workload_locations = np.arange(num_workloads) * ((bar_width * len(configs) + bar_spacing * (len(configs) - 1)) + workload_spacing)

        fig, ax = plt.subplots(figsize=(6 + num_workloads, 8))
        fig.subplots_adjust(right=0.78)  # leave room for legends anchored outside the axes

        hatches = ['/', '\\', '.', '-', '+', 'x', 'o', 'O', '*']
        patch_hatches = []
        if colors == None:
            colors = ['#cecece', '#cde1ec', '#8ec1da', '#2066a8', '#a559aa', '#59a89c', '#f0c571', '#e02b35', '#082a54', '#ededed', '#f6d6c2', '#d47264', '#800000', '#911eb4', '#4363d8', '#f58231', '#3cb44b', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#e6beff', '#e6194b', '#000075', '#9a6324', '#808080', '#ffffff', '#000000']

        # For each stat
        for x_offset, config in enumerate(configs):

            offsets = np.array([0.0] * len(workloads))
            totals = {wl: sum([all_data[f"{config} {wl} {stat}"] for stat in stats]) for wl in workloads}
            hatch = hatches[x_offset % len(hatches)]
            patch_hatch = mpatches.Patch(facecolor='beige', hatch=hatch, edgecolor="darkgrey", label=config)
            patch_hatches.append(patch_hatch)

            for i, stat in enumerate(stats):
                # Plot each workload's stat as bar graph
                #data = np.array([all_data[stat][config][wl]/totals[wl] for wl in workloads])
                data = np.array([all_data[f"{config} {wl} {stat}"]/totals[wl] for wl in workloads])
                color = colors[i%len(colors)]

                if x_offset > len(hatches):
                    print("WARN: Too many configs for unique configuration labels")
                b = ax.bar(workload_locations + x_offset*(bar_width + bar_spacing), data, [bar_width] * num_workloads,
                        bottom=offsets, edgecolor="darkgrey", color = color, hatch=hatch)

                if x_offset == 0: b.set_label(f"{stat}")

                offsets += data

        plt.title(title)
        plt.ylabel("Fraction of total")
        plt.xlabel("Workloads")
        plt.xticks(workload_locations, workloads)
        legend_1 = plt.legend(loc='center left', bbox_to_anchor=(1,0.8))
        legend_2 = plt.legend(handles=patch_hatches, bbox_to_anchor=(1,0.4))
        fig.add_artist(legend_1)

        if plot_name == None:
            plt.show()
        else:
            plt.savefig(
                plot_name,
                bbox_inches="tight",
                bbox_extra_artists=(legend_1, legend_2),
            )


    # Add basline for each experiment
    # Plot multiple stats across simpoints
    def plot_speedups_multi_stats (self, experiment: Experiment, experiment_baseline: Experiment, speedup_metric: str, 
                        title: str = None, x_label: str = "", y_label: str = "", baseline_conf = None,
                        bar_width:float = 0.35, bar_spacing:float = 0.05, workload_spacing:float = 0.3, 
                        colors = None, plot_name = None, relative_lbls = True, label_fontsize = "small",
                        label_rotation = 0):
        """Plot speedups when multiple related stats should be shown together.

        This helper is used when you want a speedup view (new vs baseline) but also want to plot
        several metrics side-by-side.
        """

        # Check experiments are similar
        configs = set(experiment.get_configurations())
        workloads = set(experiment.get_workloads())
        stats = set(experiment.get_stats())

        baseline_configs = set(experiment_baseline.get_configurations())
        baseline_worklaods = set(experiment_baseline.get_workloads())
        baseline_stats = set(experiment_baseline.get_stats())

        if configs != baseline_configs:
            print("ERR: Configs not the same")
            return

        if workloads != baseline_worklaods:
            print("ERR: Workloads not the same")
            return

        if not speedup_metric in stats or not speedup_metric in baseline_stats:
            print("ERR: Stats not the same")
            return

        if not baseline_conf is None and not baseline_conf in baseline_configs:
            print("ERR: baseline_conf not found in experiments")
            return

        num_workloads = len(workloads)
        workload_locations = np.arange(num_workloads) * ((bar_width * len(configs) + bar_spacing * (len(configs) - 1)) + workload_spacing)

        all_data = experiment.retrieve_stats(configs, [speedup_metric], workloads)
        baseline_data = experiment_baseline.retrieve_stats(configs, [speedup_metric], workloads)

        if all_data.keys() != baseline_data.keys():
            print("ERR: Keys don't match")
            return

        plt.figure(figsize=(6+num_workloads,8))

        key_order = None

        selected_configs = configs - {baseline_conf} if not baseline_conf is None else configs

        # For each Config
        for x_offset, config in enumerate(selected_configs):
            # Plot each workload's stat as bar graph

            # Determine keys (all workloads for this config) ordering consistently
            selected_keys = [key for key in all_data.keys() if config in key]
            selected_keys = sorted(selected_keys)

            # Verify consistent ordering
            if key_order == None: key_order = list(map(lambda x:x.split(" ")[1], selected_keys))
            else:
                for i in range(len(key_order)):
                    if key_order[i] != selected_keys[i].split(" ")[1]:
                        print("ERR: Ordering")
                        return

            #for key in selected_keys:
            #    print(key, baseline_data[key], all_data[key])

            data = []

            # Find speedup of all data
            for key in selected_keys:
                if baseline_data[key] == 0:
                    print("WARN: Baseline data is 0, setting column to 0")
                    data.append(0)
                    continue

                new_test_data = all_data[key]
                baseline_test_data = baseline_data[key]

                if baseline_conf != None:
                    baseline_key = " ".join([baseline_conf] + key.split(" ")[1:])
                    new_test_data /= all_data[baseline_key]
                    baseline_test_data /= baseline_data[baseline_key]

                data.append(new_test_data/baseline_test_data)

            if colors == None:
                color_map = plt.get_cmap("Paired")
                color = color_map((x_offset*(1/11))%1)
            else:
                color = colors[x_offset%len(colors)]

            # Graph
            b = plt.bar(workload_locations + x_offset*(bar_width) + (x_offset-1)*(bar_spacing), data, [bar_width] * num_workloads, 
                        color=color)
            b.set_label(config)

            for loc, dat in zip(workload_locations, data):
                if not relative_lbls:
                    lbl = f"{dat*100:3.4}%"
                else:
                    lbl = f"{'+' if dat >= 1 else '-'}{abs((1-dat)*100):3.2}%"
                plt.text(loc + x_offset*(bar_width + bar_spacing) - 2*bar_width/3, dat, lbl, fontsize = label_fontsize, rotation=label_rotation)


        plt.legend(loc="center left", bbox_to_anchor=(1,0.5))

        if y_label == "":
            y_label = f"Speedup as measured by {speedup_metric}"

        # Use saved order to label workloads
        plt.xticks(workload_locations, key_order)

        if title == None:
            title = f"Speedup of {experiment.get_experiments()[0]} over {experiment_baseline.get_experiments()[0]}"
            if baseline_conf != None: title += f" normalized by {baseline_conf} configuration"

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        if plot_name == None:
            plt.show()
        else: plt.savefig(plot_name)

    # Find diff of all numerical stats to investigate when performance differs
    def diff_stats_all (self, experiment_baseline: Experiment, experiment_new: Experiment, diff_thresh: float = 50,
                    must_contain: str = None):
        """Diff two experiments across all workloads/configs and print large changes.

        Steps:
          1) Compute weighted per-workload stats for each experiment.
          2) Compute percent difference for each stat.
          3) Print any stat whose absolute percent difference exceeds `diff_thresh`.
        """
        # Check experiments are similar
        configs1 = set(experiment_baseline.get_configurations())
        workloads1 = set(experiment_baseline.get_workloads())
        stats1 = set(experiment_baseline.get_stats())

        configs2 = set(experiment_new.get_configurations())
        workloads2 = set(experiment_new.get_workloads())
        stats2 = set(experiment_new.get_stats())

        if configs1 != configs2:
            print("ERR: Configs not the same")
            return

        if workloads1 != workloads2:
            print("ERR: Workloads not the same")
            return

        if stats1 != stats2:
            print("ERR: Stats not the same")
            return

        stats1 = stats1 - set(["Experiment", "Architecture", "Configuration", "Workload"])

        # Filter for stats that contain required phrase
        ex_baseline_df = experiment_baseline.return_raw_data(must_contain=must_contain).drop_duplicates()
        ex_new_df = experiment_new.return_raw_data(must_contain=must_contain).drop_duplicates()

        if list(ex_baseline_df["stats"]) != list(ex_new_df["stats"]):
            print("ERR: Stats not same after geting data")
            print("This error should not occur")
            # Stats were checked earlier...
            return

        ex_baseline_df = ex_baseline_df.set_index("stats").astype("float")
        ex_new_df = ex_new_df.set_index("stats").astype("float")

        differences =  ex_new_df - ex_baseline_df
        differences.drop_duplicates(inplace=True)
        diff_bit_vector = (differences.abs() >= diff_thresh).any(axis=1)
        differences = differences[diff_bit_vector]


        different_stats = list(differences.index)

        # TODO: Maybe process more? Format and return
        #print(diff_bit_vector)
        #print(different_stats)

        stat_averages = differences.sum(axis=1)/differences.count(axis=1)
        stat_variances = pd.DataFrame()
        differences.to_csv("dbg.csv")

        dropped = []

        for stat in differences.index:
            if list(differences.index).count(stat) > 1:
                print("WARN: Duplicate stat found:", stat)
                dropped.append(stat)
                continue

            xminusmean = differences.loc[stat] - stat_averages[stat]
            stat_variances = pd.concat([stat_variances, xminusmean], axis=1)

        stat_variances = stat_variances.pow(2).sum(axis=0)/stat_variances.count(axis=0)
        stat_stddev = stat_variances.pow(0.5)

        stat_stddev.to_csv("dev_dbg.csv")
        stat_averages.to_csv("avg_dbg.csv")

        # TODO: Make them have the same stats

        print(stat_averages, stat_stddev, list(stat_stddev.index))
        print(stat_averages[pd.Series(stat_averages.index) in stat_stddev.index])

        #print("Averages: \n", averages)
        #print("Variance: \n", variance)
        #print(differences)

    def diff_stats (self, experiment_baseline: Experiment, experiment_new: Experiment, workload: str, 
                    config: str, diff_thresh: float = 0.05, must_contain: str = None, baseline_config: str = None,
                    diff_type: str = "differential"):
        """Diff two experiments for a single workload and report notable stat deltas."""
        # Check experiments are similar
        configs1 = set(experiment_baseline.get_configurations())
        workloads1 = set(experiment_baseline.get_workloads())
        stats1 = set(experiment_baseline.get_stats())

        configs2 = set(experiment_new.get_configurations())
        workloads2 = set(experiment_new.get_workloads())
        stats2 = set(experiment_new.get_stats())

        if not config in configs1 or not config in configs2:
            print("ERR: Configs not the same")
            return

        if not workload in workloads1 or not workload in workloads2:
            print("ERR: Workloads not the same")
            return

        if stats1 != stats2:
            print("ERR: Stats not the same")
            return

        baseline_raw_data = experiment_baseline.return_raw_data(keep_weight=True).set_index("stats").astype("float")
        new_raw_data = experiment_new.return_raw_data(keep_weight=True).set_index("stats").astype("float")

        to_drop_for_data = []
        for col in baseline_raw_data.columns:
            if f"{config} {workload}" not in col:
                to_drop_for_data.append(col)

        baseline_data = baseline_raw_data.drop(columns=to_drop_for_data)
        new_data = new_raw_data.drop(columns=to_drop_for_data)

        def weighted_avg(df: pd.DataFrame):
            df = df*df.loc["Weight"]
            df = df.sum(axis=1)
            df = df.drop("Weight")
            return df

        baseline_data = weighted_avg(baseline_data)
        new_data = weighted_avg(new_data)

        if baseline_config != None:
            to_drop_for_baseline_config = []
            for col in baseline_raw_data.columns:
                if f"{baseline_config} {workload}" not in col:
                    to_drop_for_baseline_config.append(col)

            baseline_data_baseline_config = baseline_raw_data.drop(columns=to_drop_for_baseline_config)
            new_data_baseline_config = new_raw_data.drop(columns=to_drop_for_baseline_config)

            baseline_data_baseline_config = weighted_avg(baseline_data_baseline_config)
            new_data_baseline_config = weighted_avg(new_data_baseline_config)

            if diff_type == "differential":
                baseline_data /= baseline_data_baseline_config
                new_data /= new_data_baseline_config
            elif diff_type == "difference":
                baseline_data -= baseline_data_baseline_config
                new_data -= new_data_baseline_config
            else:
                print("diff_type must be differential or difference")
                return

        if diff_type == "differential":
            speedups = new_data/baseline_data
            speedups = speedups[speedups != math.inf]
            speedups = speedups.apply(lambda x: -1/x if x<1 and x != 0 else x)
        elif diff_type == "difference":
            speedups = new_data-baseline_data

        speedups = speedups.drop_duplicates()

        if must_contain != None:
            for col in speedups.index:
                if must_contain not in col and col in speedups.index:
                    speedups = speedups.drop(col)

        diff_vector = speedups.abs() > diff_thresh

        print("NOTE: Speedups are positive if new is faster, negative if baseline is faster")

        print()

        # TODO: SHow absolute numbers, figure out how to display well
        print("Differences sorted:", speedups[diff_vector].sort_values())
        print("\n30 biggest absolute differences (- if baseline is greater):\n", "\n".join([f"{i}: {speedups[diff_vector][i]}" for i in speedups[diff_vector].abs().sort_values().index[:-31:-1]]), sep='')
        #print(sorted(speedups[diff_vector], key=lambda x:abs(x), reverse=True))

# TODO: Make accessible

    def get_simpoint_info(self, cluster_id, workload, subsuite, suite, top_simpoint_only, workloads_data=None, sp_cache=None):
        """Get weight and segment_id for a given cluster_id from workloads_db.json or workloads_top_simp.json.

        Args:
            cluster_id: The cluster_id to look up
            workload: The workload name for error reporting
            subsuite: The subsuite name for error reporting
            suite: The suite name for error reporting
            top_simpoint_only: Top simpoint only True/False

        Returns:
            tuple: (weight, segment_id) if found, None if not found
        """
        infra_dir = _infra_root()
        if top_simpoint_only:
            workload_db_path = str(infra_dir / "workloads" / "workloads_top_simp.json")
        else:
            workload_db_path = str(infra_dir / "workloads" / "workloads_db.json")
        if workloads_data is None:
            workloads_data = utilities.read_descriptor_from_json(workload_db_path)

        try:
            key = (suite, subsuite, workload)

            # If a cache dict is provided, build (or reuse) a mapping:
            #   cluster_id -> (weight, segment_id)
            if sp_cache is not None:
                m = sp_cache.get(key)
                if m is None:
                    simpoints = workloads_data[suite][subsuite][workload].get("simpoints")
                    # If simpoints not present, treat as one simpoint with 100% weighting
                    if simpoints is None:
                        m = {}
                    else:
                        m = {int(sp["cluster_id"]): (sp["weight"], sp["segment_id"]) for sp in simpoints}
                    sp_cache[key] = m

                cid_i = int(cluster_id)
                if not m:
                    return 1, 0
                if cid_i in m:
                    return m[cid_i]
                raise StopIteration

            # No cache provided: fall back to linear scan.
            simpoints = workloads_data[suite][subsuite][workload].get("simpoints")
            if simpoints is None:
                return 1, 0
            cid_i = int(cluster_id)
            sp = next(sp for sp in simpoints if int(sp["cluster_id"]) == cid_i)
            return sp["weight"], sp["segment_id"]
        except StopIteration:

            print(f"ERROR: Could not find cluster_id {cluster_id} in simpoints for workload {workload} from suite {suite}, subsuite {subsuite}")
            return None
        except KeyError:
            if top_simpoint_only:
                print(f"ERROR: Could not find workload {workload} in workloads_top_simpoint.json")
            else:
                print(f"ERROR: Could not find workload {workload} in workloads_db.json")
            return None

    def get_cluster_ids(self, workload, suite, subsuite, top_simpoint_only, workloads_data=None):
        """Get list of cluster_ids for a given workload from workloads_db.json or workloads_top_simp.json.

        Args:
            workload: The workload name
            suite: The suite name
            subsuite: The subsuite name
            top_simpoint_only: Top simpoint only True/False

        Returns:
            list: List of cluster_ids if found, None if workload not found
        """
        infra_dir = _infra_root()
        if workloads_data is None:
            if top_simpoint_only:
                workload_db_path = str(infra_dir / "workloads" / "workloads_top_simp.json")
            else:
                workload_db_path = str(infra_dir / "workloads" / "workloads_db.json")
            workloads_data = utilities.read_descriptor_from_json(workload_db_path)

        try:
            sim_mode = workloads_data[suite][subsuite][workload]["simulation"]["prioritized_mode"]
            simpoints = utilities.get_simpoints(workloads_data[suite][subsuite][workload], sim_mode)
            return list(simpoints.keys())
        except KeyError:
            if top_simpoint_only:
                print(f"ERROR: Could not find workload {workload} in workloads_top_simp.json")
            else:
                print(f"ERROR: Could not find workload {workload} in workloads_db.json")
            return None

    # Print markdown table to easily report to github
    def print_markdown_table (self, experiment: Experiment, stats: List[str], workloads: List[str],
                             configs: List[str]):
        """Print a GitHub-friendly markdown table of selected stats.

        Useful for quickly pasting results into PRs/issues.
        """
        configs_to_load = configs
        all_data = experiment.retrieve_stats(configs_to_load, stats, workloads)
        data_to_plot = {}
        for stat in stats:
            print(f"|{stat}", end="")
            for conf in configs:
                print(f"|{conf}", end="")
            print("|")
            for wl_number, wl in enumerate(workloads):
                print(f"|{wl}", end="")
                for conf in configs:
                    data = all_data[f"{conf} {wl} {stat}"]
                    print(f"|{data}", end="")
                print("|")



# -----------------------------
# Multiprocessing helpers (for write_experiment_csv_numpy)
# -----------------------------
#
# `multiprocessing.Pool` can only execute *module-level* callables (they must be picklable).
# These helpers coordinate two parallel phases:
#   1) collecting simpoint columns into SharedMemory,
#   2) (optionally) writing the base CSV stat rows in parallel as ordered part files.

# These must be module-level functions so `multiprocessing.Pool` can pickle them.

_MP_SHM = None
_MP_VALUES = None
_MP_AGG = None
_MP_STAT_TO_ROW = None
_MP_SCHEMAS = None
_MP_LOAD_RAMULATOR = True
_MP_IGNORE_DUPS = True



def _mp_init_csv_part_writer(
    shm_name: str,
    shape,
    dtype_str: str,
    known_stats,
    groups0,
) -> None:
    """Pool initializer for parallel CSV part writing.

    Attaches to the shared `values` matrix and stores stat/group metadata in globals.
    """
    global _CSV_SHM, _CSV_VALUES, _CSV_KNOWN_STATS, _CSV_GROUPS0
    try:
        _CSV_SHM = shared_memory.SharedMemory(name=shm_name, track=False)
    except TypeError:
        _CSV_SHM = shared_memory.SharedMemory(name=shm_name)
        try:
            from multiprocessing import resource_tracker
            # resource_tracker.unregister(_CSV_SHM._name, "shared_memory")
        except Exception:
            pass

    _CSV_VALUES = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=_CSV_SHM.buf)
    _CSV_KNOWN_STATS = known_stats
    _CSV_GROUPS0 = groups0


def _mp_write_csv_part(arg):
    """Worker: write a contiguous slice of stat rows to a part CSV file.

    Args:
        arg: (part_path, start_idx, end_idx)
    Returns:
        None on success, or (part_path, start_idx, end_idx, error_repr) on failure.
    """
    part_path, start_idx, end_idx = arg
    try:
        with open(part_path, "w", newline="") as f:
            w = csv.writer(f)
            vals = _CSV_VALUES
            stats = _CSV_KNOWN_STATS
            groups = _CSV_GROUPS0
            for i in range(start_idx, end_idx):
                w.writerow([stats[i], True, groups[i], *vals[i, :].tolist()])
        return None
    except Exception as exc:
        return (part_path, start_idx, end_idx, repr(exc))


def _mp_init_shared_collector(
    shm_name: str,
    shape,
    dtype_str: str,
    stat_to_row: dict,
    component_schemas: dict,
    load_ramulator: bool,
    ignore_duplicates: bool,
) -> None:
    """Pool initializer: attach to shared memory and stash read-only metadata in globals."""
    global _MP_SHM, _MP_VALUES, _MP_AGG, _MP_STAT_TO_ROW, _MP_SCHEMAS, _MP_LOAD_RAMULATOR, _MP_IGNORE_DUPS
    try:
        # Python 3.13+: opt out of resource_tracker bookkeeping.
        _MP_SHM = shared_memory.SharedMemory(name=shm_name, track=False)
    except TypeError:
        # Python < 3.13: attach normally, then unregister (commented for now) so the worker won't try to unlink at shutdown.
        _MP_SHM = shared_memory.SharedMemory(name=shm_name)
        try:
            from multiprocessing import resource_tracker
            # resource_tracker.unregister(_MP_SHM._name, "shared_memory")
        except Exception:
            pass

    _MP_VALUES = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=_MP_SHM.buf)

    # Create a local aggregator instance per worker.
    _MP_AGG = stat_aggregator()

    _MP_STAT_TO_ROW = stat_to_row
    _MP_SCHEMAS = component_schemas
    _MP_LOAD_RAMULATOR = bool(load_ramulator)
    _MP_IGNORE_DUPS = bool(ignore_duplicates)


def _mp_fill_simpoint_column(arg):
    """Worker: fill one column in the shared `values` matrix."""
    col_idx, sim_dir = arg
    try:
        _MP_AGG.load_simpoint_into(
            sim_dir,
            _MP_STAT_TO_ROW,
            _MP_VALUES[:, col_idx],
            schema_cache=_MP_SCHEMAS,
            clear_column=False,
            load_ramulator=_MP_LOAD_RAMULATOR,
            ignore_duplicates=_MP_IGNORE_DUPS,
        )
        return None
    except Exception as exc:
        return (col_idx, sim_dir, repr(exc))
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a Scarab experiment CSV (fast path only).")
    parser.add_argument("-d", "--descriptor_name", required=True, help="Path to the experiment descriptor JSON.")
    parser.add_argument("-o", "--outfile", required=True, help="Output CSV path.")
    parser.add_argument("--postprocess", action="store_true", help="Append derived IPC and distribution stats to the CSV.")
    parser.add_argument("--skip-incomplete", action="store_true", help="Skip incomplete simpoints (warn).")
    parser.add_argument("--jobs", type=int, default=8, help="Worker processes for collection/writing.")
    args = parser.parse_args()

    da = stat_aggregator()
    out_written = da.write_experiment_csv_numpy(
        args.descriptor_name,
        args.outfile,
        slurm=True,
        postprocess=bool(args.postprocess),
        skip_incomplete=bool(args.skip_incomplete),
        jobs=int(args.jobs) if args.jobs is not None else 8,
    )
    print(f"Wrote CSV: {out_written}")
