import argparse
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
import numpy as np
import matplotlib
import matplotlib.patches as mpatches
import json
import os
import math
from pathlib import Path

from scripts import utilities

def _infra_root() -> Path:
    return Path(__file__).resolve().parent.parent

def get_elem(l, i):
    return list(map(lambda x:x[i], l))

load_simpoint_data_count = 0

class Experiment:
    def __init__(self, stats):
        '''Stats is either a path to saved experiment or list of stats'''
        if type(stats) == str:
            self.data = pd.read_csv(stats, low_memory=False)

        else:
            self.data = pd.DataFrame()
            rows = stats.copy()
            rows.append("Experiment")
            rows.append("Architecture")
            rows.append("Configuration")
            rows.append("Workload")
            rows.append("Segment Id")
            rows.append("Cluster Id")
            rows.append("Weight")

            self.data["stats"] = rows

            # TODO: Add Groups here. Issues with write protect will apply to the group col too. Getters and derived stats affected

            # Enable write protect for all base stats
            self.data["write_protect"] = [True for _ in rows]

    def has_group_data(self):
        return "groups" in self.data.columns

    def set_groups(self, groups):
        # NOTE: Add 7 0s at end for all the appends in add_simpoint
        self.data["groups"] = list(groups) + [0]*7

    def get_groups(self):
        # NOTE: Add 7 0s at end for all the appends in add_simpoint
        return set(self.data["groups"])

    def add_simpoint(self, simpoint_data, experiment, arch, config, workload, seg_id, c_id, weight):
        column = simpoint_data
        column.append(experiment)
        column.append(arch)
        column.append(config)
        column.append(workload)
        column.append(seg_id)
        column.append(c_id)
        column.append(weight)

        self.data = self.data.copy()
        self.data[f"{config} {workload} {c_id}"] = column

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

    def defragment(self):
        self.data = self.data.copy()

    def derive_stat(self, equation:str, overwrite:bool=True, pre_agg:bool=True,
                    write_prot:bool=False):
        # TODO: Doesn't work for stats with spaces in the names
        # Make sure tokens have space padding
        single_char_tokens = ["+", "-", "*", "/", "(", ")", "="]
        equation = "".join([f" {c} " if c in single_char_tokens else c for c in equation])

        # Tokenize
        tokens = list(filter(None, equation.split(" ")))

        insert_index = len(self.data)

        #self.data.loc[total_cols] = list(self.data.loc[self.data["stats"] == "Weight"])

        values = []
        panda_fy = lambda name: f'lookup["{name}"]'

        lookup_cols = {old:new for old, new in zip(self.data.T.columns, self.data.T.iloc[0])}
        str_rows = [list(self.data["stats"]).index(row) for row in ["Experiment","Architecture","Configuration","Workload"]]

        # NOTE: Drop metadata here, don't do operations on them
        lookup = self.data.T.rename(columns=lookup_cols).drop("stats").drop("write_protect").drop("groups")
        # print(lookup["Weight"])
        # print(panda_fy_agg(""))
        # exit()

        # Aggregates before returning row
        # Takes in a stat name, returns a Pandas series
        def panda_fy_agg(name):
            # Get the stat, pre-weight values
            data = lookup[name]*lookup["Weight"]
            data_weighted = {}
            columns = list(data.index)

            # Sum the weighted values
            for setup in set(map(lambda x:" ".join(x.split(" ")[:-1]), columns)):
                data_weighted[setup] = sum([data[lbl] for lbl in columns if setup in lbl])

            # Duplicate values to fill out all simpoints in the dataframe
            # Ex: mysql baseline 4, mysql baseline 10 should both have values (the same value)
            data_weighted_duplicated = {}
            for col in columns:
                prefix = " ".join(col.split(" ")[:-1])
                data_weighted_duplicated[col] = data_weighted[prefix]

            # Make a sereies so equaation works
            return pd.Series(data_weighted_duplicated.values(), data_weighted_duplicated.keys())

        # If pre_agg, then aggregate *while* retrieving stat
        if pre_agg:
            panda_fy = lambda name: f'panda_fy_agg("{name}")'

        str_rows = [lookup.columns[i] for i in str_rows]
        lookup = lookup.drop(columns=str_rows).astype("float")

        dependant_stats = []
        for i, tok in enumerate(tokens):
            if i == 0:
                if tok.isnumeric() or tok in single_char_tokens:
                    print("ERR: Equation should be '<result_name> = ...'")
                    return

                tokens[i] = "values.append(list("
                values.append(tok)
                continue

            if i == 1:
                if tok != "=":
                    print("ERR: Equation should be '<result_name> = ...'")
                    return
                tokens[i] = ""

            if not tok.isnumeric() and not tok in single_char_tokens:
                tokens[i] = panda_fy(tok)
                dependant_stats.append(tok)

            if tok.isnumeric():
                tokens[i] = str(tok)

        tokens.append("))")
        to_eval = " ".join(tokens)

        stat_name = values[0]

        # dependant_stats = list(set(dependant_stats))
        # print("Dependencies:", list(set(dependant_stats)))

        # print(lookup)

        # print(to_eval)

        # print("agg test",panda_fy_agg("ICACHE_EVICT_HIT_ONPATH_BY_FDIP_count"))

        # print(stat_name, stat_name in set(self.data["stats"]))
        if stat_name in set(self.data["stats"]):
            wr_prot = self.data[self.data["stats"] == stat_name]["write_protect"].item()
            if wr_prot:
                print(f"ERR: Tried to overwrite stat '{stat_name}' with write protect set. Cannot overwrite scarab generated stats.")
                return
            elif not overwrite:
                print(f"ERR: Tried to overwrite stat '{stat_name}' with overwrite set to False.")
                return
            else:
                print(f"INFO: Overwriting value(s) of stat '{stat_name}'")
                insert_index = self.data[self.data["stats"] == stat_name].index[0]

        # TODO: Unsafe!
        # print("Evalling:", to_eval)
        eval(to_eval)

        # NOTE: Add metadata columns here.
        # [name, write_prot, group]
        # print(f"Values:", values)
        row = [stat_name, write_prot, 0] + values[1]

        self.data.loc[insert_index] = row

        if self.data.loc[insert_index].isna().any():
            print(f"ERR: 'NaN' calculated in result of the equation '{equation}'")
            print(f"Likely division by zero. Found {self.data.loc[insert_index].isna().count()} NaNs",
                  "across all simpoints.")
            return

        return

    def to_csv(self, path:str):
        '''Turns selected stats from selected workloads/configs into a pandas dataframe'''

        self.data.to_csv(path, index=False)

    def return_raw_data(self, must_contain: list = None, keep_weight: bool = False):
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

    def calculate_distribution_stats(self):
        groups = self.get_groups()
        groups.remove(0)

        errs = 0

        # TODO: It takes more than 40 minutes for 523 groups for 15 workloads x 5 configurations
        # Performance improvement required
        num_groups = len(groups)
        print(f"INFO: Calculate distribution stats for {num_groups} groups")
        remove_columns = ["write_protect", "groups"]

        # Initialize numpy array to store all data
        all_data = []
        stat_columns = self.data.columns

        import time
        start = time.time()

        # Do calculations for each group
        for group in groups:
            print(f"INFO: group {group}/{num_groups}..")
            group_df = self.data[(self.data["groups"] == group)].drop(columns=remove_columns)

            count_data_stats = list(filter(lambda x: x.endswith("_count") and not x.endswith("_total_count"), group_df["stats"]))
            total_count_data_stats = list(filter(lambda x: x.endswith("_total_count"), group_df["stats"]))

            errs += 1 if len(group_df) != len(count_data_stats) + len(total_count_data_stats) else 0

            count_data_df = group_df[group_df["stats"].isin(count_data_stats)].set_index("stats")
            total_count_data_df = group_df[group_df["stats"].isin(total_count_data_stats)].set_index("stats")

            count_sums = count_data_df.sum(axis=0)
            total_count_sums = total_count_data_df.sum(axis=0)

            # Replace NaN values where all values are zero
            if float(0) in list(count_sums) or float(0) in list(total_count_sums):
                new_stats = [f"group_{group}_total_mean", f"group_{group}_mean",
                             f"group_{group}_total_stddev", f"group_{group}_stddev"]

                for stat in total_count_percentages.index:
                    new_stats.append(f"{stat}_pct")

                for stat in count_percentages.index:
                    new_stats.append(f"{stat}_pct")

                # Add NaN rows for this group
                for stat in new_stats:
                    all_data.append([stat, True, 0] + [np.nan] * len(count_sums))
                continue

            # Get mean and standard deviation of WHOLE distribution, then percent that each sample makes up of distribution (data_df/sums)
            total_count_means = total_count_sums / len(total_count_data_df)
            count_means = count_sums / len(count_data_df)

            total_count_stddev = (((total_count_data_df - total_count_means) ** 2).sum() / (len(total_count_data_df))).pow(0.5)
            count_stddev = (((count_data_df - count_means) ** 2).sum() / (len(count_data_df))).pow(0.5)

            total_count_percentages = total_count_data_df / total_count_sums
            count_percentages = count_data_df / count_sums

            # Add group statistics to numpy array
            all_data.append([f"group_{group}_total_mean", True, 0] + list(total_count_means))
            all_data.append([f"group_{group}_mean", True, 0] + list(count_means))
            all_data.append([f"group_{group}_total_stddev", True, 0] + list(total_count_stddev))
            all_data.append([f"group_{group}_stddev", True, 0] + list(count_stddev))

            # Add percentage statistics
            for stat in total_count_percentages.index:
                all_data.append([f"{stat}_pct", True, 0] + list(total_count_percentages.loc[stat]))
            for stat in count_percentages.index:
                all_data.append([f"{stat}_pct", True, 0] + list(count_percentages.loc[stat]))

        # Create single DataFrame from numpy array and concatenate with self.data
        group_calculations = pd.DataFrame(all_data, columns=stat_columns)
        self.data = pd.concat([self.data, group_calculations]).reset_index(drop=True)

        print(f"Took: {time.time() - start} seconds")

        if errs != 0:
            print("WARN: Distribution size and number of x_count + x_total_count stats is not equal.")
            return

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

# Files for pandas to read, does not like the per line data
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
    def __init__(self) -> None:
        self.experiments = {}
        self.simpoint_info = {}
        self.experiment = None  # Initialize experiment as class variable

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
    def load_simpoint(self, path, load_ramulator=True, ignore_duplicates = True, return_stats = False, order = None):
        data = pd.Series(dtype="float64")
        group = pd.Series(dtype="float64")
        all_stats = []

        for file in stat_files:
            filename = f"{path}{file}"
            df = pd.read_csv(filename).T
            df.columns = df.iloc[0]
            df = df.drop(df.index[0])

            if ignore_duplicates:
                duplicates = set(df.columns) & set(data.index)

                if duplicates != set():
                    df = df.drop(columns=list(duplicates))

            # Check duplicates in df.columns
            if True in df.columns.duplicated():
                print("WARN: CSV file contains duplicates")
                print("Duplicates are:", set(df.columns[df.columns.duplicated(False)]))
                print("Checking if issue is resolvable...")

                duplicated = set(df.columns[df.columns.duplicated(False)])
                equals = [(df[lbl].iloc[0] == df[lbl].iloc[0][0]).all() for lbl in duplicated]
                if not all(equals):
                    print(f"ERR: Unable to resolve duplicates. Duplicate columns have unique values. File:{filename}")
                    exit(1)

                print("Duplicates equivalent! Resolved")
                df = df.drop_duplicates()

            # Check for elements in both
            if set(df.columns) & set(all_stats) != set():
                print("ERR: Duplication prevention logic failed")
                exit(1)

            all_stats += list(df.columns)
            data = pd.concat([data, df.iloc[1]])
            group = pd.concat([group, df.iloc[0]])

        # NOTE: Ramulator will not be in distribution, group should be 0
        if load_ramulator:
            f = open(f"{path}ramulator.stat.out")
            lines = f.readlines()

            tmp = []
            tmp_lbl = []
            for line in lines:
                if not "ramulator." in line:
                    continue

                all_stats.append(line.split()[0])
                tmp_lbl.append(line.split()[0])
                tmp.append(line.split()[1])

            data = pd.concat([data, pd.Series(tmp, index=tmp_lbl)])
            group = pd.concat([group, pd.Series([0 for _ in range(len(tmp))], index=tmp_lbl)])

            f.close()


        #if order != None:
        #    tmp = []
        #    for lbl in order:
        #        if lbl in list(data.index):
        #            tmp.append(data[lbl])
        #        else: tmp.append("nan")

        # TODO: REmove duplicates. Ramulator duplicated??

        if order != None: 
            data.drop_duplicates()

        if order != None:
            data = data.reindex(order, fill_value="nan")

        data = list(map(float, list(data)))
        #if order != None: print(len(data), len(order))

        if not return_stats: return data, group
        else: return all_stats

    # Load experiment from saved file
    def load_experiment_csv(self, path):
        return Experiment(path)

    # Load experiment form json file
    def load_experiment_json(self, experiment_file: str, slurm: bool = False):
        # Load json data from experiment file
        json_data = None
        try:
            with open(experiment_file, "r") as file:
                json_data = json.loads(file.read())
        except:
            return None

        simulations_path = json_data["root_dir"]
        simpoints_path = "/soe/hlitz/lab/traces/" if json_data["traces_dir"] == None else json_data["traces_dir"]

        # Make sure simulations and simpoints path has known format
        if simulations_path[-1] != '/': simulations_path += "/"
        if simpoints_path[-1] != '/': simpoints_path += "/"

        # Asking user can provide simulations directory or docker home folder
        if not simulations_path.endswith("simulations/"):
            simulations_path += "simulations/"

        experiment_name = json_data["experiment"]
        architecture = json_data["architecture"]

        if (experiment_name in simulations_path) and slurm:
            print(f"WARN: simulations_path should only point to root of docker home. If this fails, please remove {experiment_name} from simulations_path")

        known_stats = None

        infra_dir = _infra_root()
        top_simpoint_only = False
        if json_data["top_simpoint"]:
            top_simpoint_only = True
            workload_db_path = str(infra_dir / "workloads" / "workloads_top_simp.json")
        else:
            workload_db_path = str(infra_dir / "workloads" / "workloads_db.json")
        workloads_data = utilities.read_descriptor_from_json(workload_db_path)

        experiment = None
        # Set set of all stats. Should only differ by config
        for config in json_data["configurations"]:
            config = config.replace("/", "-")
            # Use first workload
            for simulation in json_data["simulations"]:
                found_suite = simulation["suite"]
                found_subsuite = simulation["subsuite"]
                found_workload = simulation["workload"]
                found_cluster_id = simulation["cluster_id"]

                if found_cluster_id != None:
                    assert found_workload is not None, "Workload cannot be None when cluster_id is specified"
                    assert found_suite is not None, "Suite cannot be None when cluster_id is specified"
                    assert found_subsuite is not None, "Subsuite cannot be None when cluster_id is specified"
                    experiment, known_stats = self.load_simpoint_data(found_cluster_id, found_workload, found_subsuite, found_suite,
                                                                      config, experiment_name, architecture, simulations_path, top_simpoint_only)
                elif found_workload != None:
                    assert found_suite is not None, "Suite cannot be None when workload is specified"
                    assert found_subsuite is not None, "Subsuite cannot be None when workload is specified"
                    cluster_ids = self.get_cluster_ids(found_workload, found_suite, found_subsuite, top_simpoint_only)
                    for cluster_id in cluster_ids:
                        experiment, known_stats = self.load_simpoint_data(cluster_id, found_workload, found_subsuite, found_suite,
                                                                          config, experiment_name, architecture, simulations_path, top_simpoint_only)
                elif found_subsuite != None:
                    assert found_suite is not None, "Suite cannot be None when subsuite is specified"
                    workloads = list(workloads_data[found_suite][found_subsuite].keys())
                    for workload in workloads:
                        cluster_ids = self.get_cluster_ids(workload, found_suite, found_subsuite, top_simpoint_only)
                        for cluster_id in cluster_ids:
                            experiment, known_stats = self.load_simpoint_data(cluster_id, workload, found_subsuite, found_suite,
                                                                              config, experiment_name, architecture, simulations_path, top_simpoint_only)
                else:
                    subsuites = list(workloads_data[found_suite].keys())
                    for subsuite in subsuites:
                        workloads = list(workloads_data[found_suite][subsuite].keys())
                        for workload in workloads:
                            cluster_ids = self.get_cluster_ids(workload, found_suite, subsuite, top_simpoint_only)
                            for cluster_id in cluster_ids:
                                experiment, known_stats = self.load_simpoint_data(cluster_id, workload, subsuite, found_suite,
                                                                                  config, experiment_name, architecture, simulations_path, top_simpoint_only)

        print(f"load_simpoint_data was called {load_simpoint_data_count} times")
        experiment.defragment()
        # print("\n\n", experiment)

        print("INFO: calculating derived stats...")

        # Derive IPC
        experiment.derive_stat("Cumulative_IPC = Cumulative_Instructions / Cumulative_Cycles", write_prot = True)
        print("INFO: Cumulative_IPC calculated.")
        experiment.derive_stat("Periodic_IPC = Periodic_Instructions / Periodic_Cycles", write_prot = True)
        print("INFO: Periodic_IPC calculated.")

        # Derive distribution stats for each group
        # 'Group' 0 is no group
        experiment.calculate_distribution_stats()

        return experiment

    # Plot one stat across multiple workloads
    def plot_workloads_speedup (self, experiment: Experiment, stats: List[str], workloads: List[str],
                                configs: List[str], speedup_baseline: str = None, title: str = "", x_label: str = "",
                                y_label: str= "", logscale: bool = False, bar_width:float = 0.2,
                                bar_spacing:float = 0.05, workload_spacing:float = 0.3, average: bool = False,
                                colors = None, plot_name = None, ylim = None):
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

    def get_simpoint_info(self, cluster_id, workload, subsuite, suite, top_simpoint_only):
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
        workloads_data = utilities.read_descriptor_from_json(workload_db_path)

        try:
            simpoints = workloads_data[suite][subsuite][workload].get("simpoints")
            # If simpoints not present, treat as one simpoint with 100% weighting
            if simpoints == None:
                return 1, 0
            sp = next(sp for sp in simpoints if sp["cluster_id"] == int(cluster_id))
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

    def get_cluster_ids(self, workload, suite, subsuite, top_simpoint_only):
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

    def load_simpoint_data(self, cluster_id, workload, subsuite, suite, config, experiment_name, architecture, simulations_path, top_simpoint_only):
        """Helper function to load simpoint data and add it to the experiment.

        Args:
            cluster_id: The cluster ID to load
            workload: The workload name
            subsuite: The subsuite name
            suite: The suite name
            config: The configuration name
            experiment_name: Name of the experiment
            architecture: The architecture name
            simulations_path: Path to simulations directory
            top_simpoint_only: Top simpoint only True/False

        Returns:
            tuple: (experiment, known_stats) if successful, (None, None) if failed
        """
        global load_simpoint_data_count
        load_simpoint_data_count += 1

        weight, seg_id = self.get_simpoint_info(cluster_id, workload, subsuite, suite, top_simpoint_only)
        if weight is None or seg_id is None:
            return None, None

        directory = f"{simulations_path}{experiment_name}/{config}/{suite}/{subsuite}/{workload}/{str(cluster_id)}/"
        known_stats = self.load_simpoint(directory, return_stats=True)

        if self.experiment is None:
            self.experiment = Experiment(known_stats)

        data, groups = self.load_simpoint(directory, order=known_stats)
        if not self.experiment.has_group_data():
            print("INFO: Added group data")
            self.experiment.set_groups(groups)

        self.experiment.add_simpoint(data, experiment_name, architecture, config, workload, seg_id, cluster_id, weight)

        return self.experiment, known_stats

    # Print markdown table to easily report to github
    def print_markdown_table (self, experiment: Experiment, stats: List[str], workloads: List[str],
                             configs: List[str]):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-d','--descriptor_name', required=True, help='Experiment descriptor name. Usage: -d exp.json')

    args = parser.parse_args()

    da = stat_aggregator()
    E = da.load_experiment_json(args.descriptor_name, True)
    E.to_csv("fast.csv")
    # exit(1)
    EL = da.load_experiment_csv("fast.csv")
    print(E.data)
    print(EL.data)
    print(E.data == EL.data)
    print(E.get_experiments())

    # Create equation that sums all of the stats
    stats_to_plot = ['ICACHE_EVICT_MISS_ONPATH_BY_FDIP_count','ICACHE_EVICT_HIT_ONPATH_BY_FDIP_count']
    equation = f"UNUSEFUL_pct = ICACHE_EVICT_MISS_ONPATH_BY_FDIP_count / (ICACHE_EVICT_MISS_ONPATH_BY_FDIP_count + ICACHE_EVICT_HIT_ONPATH_BY_FDIP_count)"
    equation2 = f"UNUSEFUL_agg_pct = ICACHE_EVICT_MISS_ONPATH_BY_FDIP_count / (ICACHE_EVICT_MISS_ONPATH_BY_FDIP_count + ICACHE_EVICT_HIT_ONPATH_BY_FDIP_count)"
    print("Equation:", equation)

    # Add stat as new entry
    E.derive_stat(equation)
    E.derive_stat(equation2, pre_agg=True)
    E.derive_stat("Test1 = BP_ON_PATH_MISFETCH_total_count + 1")
    E.derive_stat("Test2 = Weight + 1", pre_agg=False)

    cfs = E.get_configurations()
    wls = E.get_workloads()
    stats_to_plot = ['UNUSEFUL_pct']

    # print(E.retrieve_stats(cfs, ["UNUSEFUL_pct", "UNUSEFUL_agg_pct"], wls))

    alls = ["UNUSEFUL_pct","UNUSEFUL_agg_pct", "ICACHE_EVICT_MISS_ONPATH_BY_FDIP_count", "ICACHE_EVICT_MISS_ONPATH_BY_FDIP_count", "ICACHE_EVICT_HIT_ONPATH_BY_FDIP_count"]

    # a = E.retrieve_stats(cfs, alls, wls)
    # for k,v in a.items():
    #     print(f"{k}: {v}")

    # Call the plot function
    # E.to_csv("fast.csv")
    da.plot_workloads(E, stats_to_plot, wls, cfs, title="", average=True, x_label="Benchmarks", y_label="UNUSEFUL_pct", bar_width=0.10, plot_name="a.png")

    stats_to_plot = ['Periodic_IPC']
    da.print_markdown_table(E, stats_to_plot, wls, cfs)

    #E = Experiment("panda3.csv")
    #E2 = Experiment("panda3.csv")
    #da.plot_speedups(E, E2, "Cumulative Cycles", plot_name="a.png")
    #da.diff_stats(E, E2)
    #print(E.data)
    #print(E.data)
    # E.to_csv("panda.csv")
    # E = Experiment("panda.csv")
    # ipc = instruction / cycles => surrount all column names with df[%s] and then eval()
    #da.plot_stacked(E, ['BTB_OFF_PATH_MISS_count', 'BTB_OFF_PATH_HIT_count'], ["mysql", "verilator", "xgboost"], ["fe_ftq_block_num.8", "fe_ftq_block_num.16"], plot_name="output.png")
    #da.plot_workloads(E, ["BTB_ON_PATH_MISS_count", "BTB_ON_PATH_HIT_count", "BTB_OFF_PATH_MISS_count", "BTB_ON_PATH_WRITE_count", "BTB_OFF_PATH_WRITE_count"], ["mysql", "verilator", "xgboost"], ["fe_ftq_block_num.8"], speedup_baseline="fe_ftq_block_num.16", logscale=False, average=True, plot_name="output.png")
    #print(E.retrieve_stats("exp2", ["fe_ftq_block_num.16"], ['BTB_OFF_PATH_MISS_count', 'BTB_OFF_PATH_HIT_count'], ["mysql"], "Simpoint"))
    # k = 1
    # E.derive_stat(f"test=(BTB_OFF_PATH_MISS_count + {k})") 
    # k = 100
    # E.derive_stat(f"test=(test * 2)") 
    #da.plot_workloads(E, "exp2", ["test"], ["mysql", "verilator", "xgboost"], ["fe_ftq_block_num.8"], speedup_baseline="fe_ftq_block_num.16", logscale=False, average=True)
    #print(E.get_stats())
    # E.to_csv("test.csv")
    #print(E.retrieve_stats("exp2", ["fe_ftq_block_num.16"], ['BTB_OFF_PATH_MISS_count', 'BTB_OFF_PATH_HIT_count'], ["mysql", "xgboost"], aggregation_level="Config"))
    #da.plot_simpoints(E, "exp2", ["BTB_ON_PATH_MISS_total_count"], ["mysql", "verilator", "xgboost"], ["fe_ftq_block_num.8"], speedup_baseline="fe_ftq_block_num.16", title="Simpoint")
    #da.plot_configs(E, "exp2", ['BTB_OFF_PATH_MISS_count', 'BTB_OFF_PATH_HIT_count'], ["mysql", "xgboost"], ["fe_ftq_block_num.16", "fe_ftq_block_num.8"])

load_simpoint_data_count = 0
