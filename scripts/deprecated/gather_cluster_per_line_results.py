import csv
import os, sys
import json
import argparse

def read_descriptor_from_json(descriptor_filename):
    # Read the descriptor data from a JSON file
    try:
        with open(descriptor_filename, 'r') as json_file:
            descriptor_data = json.load(json_file)
        return descriptor_data
    except FileNotFoundError:
        print(f"Error: File '{descriptor_filename}' not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in file '{descriptor_filename}': {e}")
        return None

class Simpoint:
    def __init__(self, seg_id, weight, sim_dir, c_id):
        self.seg_id = seg_id
        self.weight = weight
        self.sim_dir = sim_dir
        # some times the cluster ids by simpoint are not consecutive
        self.c_id = c_id
        # paralell with stat groups
        self.w_stat_vals = {}

def read_simpoints(sp_dir, sim_root_dir, whole_sim = False):
    total_weight = 0
    simpoints = []
    with open(sp_dir + "/opt.p.lpt0.99", "r") as f1, open(sp_dir + "/opt.w.lpt0.99", "r") as f2:
        for line1, line2 in zip(f1, f2):
            seg_id = int(line1.split()[0])
            weight = float(line2.split()[0])
            c_id = int(line1.split()[1])
            assert(int(line1.split()[1]) == int(line2.split()[1]))
            total_weight += weight
            if whole_sim == False:
                simpoints.append(Simpoint(seg_id, weight, sim_root_dir + "/" + str(seg_id), c_id))
            else:
                simpoints.append(Simpoint(seg_id, weight, sim_root_dir, c_id))

    if total_weight - 1 > 1e-5:
        print("total weight of SimPoint does not add up to 1? {}".format(total_weight))
        exit

    return simpoints

def read_simpoint_csv_stats(simpoints):
    for simp in simpoints:
        with open(simp.sim_dir + "/per_line_icache_line_info.csv", 'r') as file:
            csv_reader = csv.DictReader(file)
            simp.w_stat_vals = {row.pop('cl_addr'): row for row in csv_reader}
            for cl_addr in simp.w_stat_vals.keys():
                for key in simp.w_stat_vals[cl_addr].keys():
                    simp.w_stat_vals[cl_addr][key] = simp.weight * float(simp.w_stat_vals[cl_addr][key])

def calculate_weighted_average(simpoints):
    weighted_avg_stats = {}
    for simp in simpoints:
        for cl_addr in simp.w_stat_vals.keys():
            if cl_addr in weighted_avg_stats.keys():
                for key in simp.w_stat_vals[cl_addr].keys():
                    weighted_avg_stats[cl_addr][key] += simp.w_stat_vals[cl_addr][key]
            else:
                weighted_avg_stats[cl_addr] = simp.w_stat_vals[cl_addr]
    return weighted_avg_stats

def report(weighted_avg_stats, sim_path):
    unique_useful_lines = 0
    unique_unuseful_lines = 0
    with open(sim_path + "/per_line_icache_line_info.csv", 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["cl_addr", "useful_cnt", "unuseful_cnt", "prefetch_cnt", "new_prefetch_cnt", "icache_hit", "icache_miss"])
        for cl_addr in weighted_avg_stats.keys():
            writer.writerow([cl_addr, weighted_avg_stats[cl_addr]["useful_cnt"], weighted_avg_stats[cl_addr]["unuseful_cnt"], weighted_avg_stats[cl_addr]["prefetch_cnt"], weighted_avg_stats[cl_addr]["new_prefetch_cnt"], weighted_avg_stats[cl_addr]["icache_hit"], weighted_avg_stats[cl_addr]["icache_miss"]])
            if weighted_avg_stats[cl_addr]["useful_cnt"] != 0:
                unique_useful_lines += 1
            if weighted_avg_stats[cl_addr]["unuseful_cnt"] != 0:
                unique_unuseful_lines += 1

    with open(sim_path + "/unique_learned_cache_lines.csv", 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["unique_useful_lines", "unique_unuseful_lines", "unique_leanred_lines"])
        writer.writerow([unique_useful_lines, unique_unuseful_lines, len(weighted_avg_stats.keys())])

if __name__ == "__main__":
    # Create a parser for command-line arguments
    parser = argparse.ArgumentParser(description='Read descriptor file name')
    parser.add_argument('-d','--descriptor_name', required=True, help='Experiment descriptor name. Usage: -d exp2.json')
    parser.add_argument('-p','--sim_path', required=True, help='Path to the simulation directory. Usage: -p /soe/<USER>/allbench_home/simpoint_flow/simulations/')
    parser.add_argument('-t','--trace_path', required=True, help='Path to the trace directory for reading simpoints. Usage: -t /soe/hlitz/lab/traces/')

    args = parser.parse_args()
    descriptor_filename = args.descriptor_name
    descriptor_data = read_descriptor_from_json(descriptor_filename)
    print(descriptor_data)
    benchmarks = descriptor_data["workloads_list"]
    for benchmark in benchmarks:
        for config_key in descriptor_data["configurations"].keys():
            simp_path = args.trace_path + '/' + benchmark + '/simpoints/'
            sim_path = args.sim_path + '/' + benchmark + '/' + descriptor_data["experiment"] + '/' + config_key
            simpoints = read_simpoints(simp_path, sim_path)
            read_simpoint_csv_stats(simpoints)
            weighted_avg_stats = calculate_weighted_average(simpoints)
            report(weighted_avg_stats, sim_path)
