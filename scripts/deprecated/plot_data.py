import os
import json
import argparse
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

matplotlib.rc('font', size=14)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

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

def get_IPC(descriptor_data, baseline_name, sim_path):
  benchmarks_org = descriptor_data["workloads_list"].copy()
  benchmarks = descriptor_data["workloads_list"].copy()
  print("nrows: " + str(len(benchmarks)/3))
  ipc_speedup = {}
  mpki = {}
  imiss_cycle = {}

  try:
    for config_key in descriptor_data["configurations"].keys():
      print(config_key)
      ipc_speedups_config = []
      mpki_config = []
      imiss_cycle_config = []
      avg_IPC_speedup_config = 1
      # avg_IPC_speedup_config_wo_outlier = 1
      avg_MPKI_config = 1
      # avg_MPKI_config_wo_outlier = 1
      avg_cyc_imiss_config = 1
      # avg_cyc_imiss_config_wo_outlier = 1
      cnt_benchmarks = 0
      for benchmark in benchmarks_org:
        print(benchmark)
        exp_path = sim_path+benchmark+'/'+descriptor_data["experiment"]+'/'
        print(exp_path+baseline_name+'/ipc.csv')
        df_ipc = pd.read_csv(exp_path+baseline_name+'/ipc.csv')
        IPC_baseline = df_ipc['IPC'][0]

        cycles = 0
        insts = 0
        df_ipc = pd.read_csv(exp_path+config_key+'/ipc.csv')
        df_imiss = pd.read_csv(exp_path+config_key+'/icache_access.csv', index_col='Simpoints')
        df_imiss_cyc = pd.read_csv(exp_path+config_key+'/inst_lost_wait_for_icache_miss.csv', index_col='Simpoints')
        cycles = df_ipc['cycles'][0]
        insts = df_ipc['instructions'][0]
        IPC = df_ipc['IPC'][0]
        # IPC_speedup = IPC/IPC_baseline
        IPC_speedup = 100.0*IPC/IPC_baseline - 100.0
        KI = float(insts)/1000.0
        imiss = df_imiss['ICACHE_MISS_w_val']['weighted_avg']
        imiss_cyc = df_imiss_cyc['INST_LOST_WAIT_FOR_ICACHE_MISS_w_val']['weighted_avg']
        MPKI = imiss/KI
        cyc_imiss = imiss_cyc/KI
        avg_IPC_speedup_config *= IPC_speedup
        avg_MPKI_config *= MPKI
        avg_cyc_imiss_config *=cyc_imiss

        cnt_benchmarks = cnt_benchmarks + 1

        imiss_cycle_config.append(cyc_imiss)
        ipc_speedups_config.append(IPC_speedup)
        mpki_config.append(MPKI)

      num = len(benchmarks)
      if config_key != baseline_name:
        # ipc_speedups_config.append(avg_IPC_speedup_config_wo_outlier**((num-2)**-1))
        ipc_speedups_config.append(avg_IPC_speedup_config**(num**-1))
      # mpki_config.append(avg_MPKI_config_wo_outlier**((num-2)**-1))
      mpki_config.append(avg_MPKI_config**(num**-1))
      # imiss_cycle_config.append(avg_cyc_imiss_config_wo_outlier**((num-2)**-1))
      imiss_cycle_config.append(avg_cyc_imiss_config**(num**-1))

      print(benchmarks)
      if config_key != baseline_name:
        print(config_key + " IPC speedups")
        print(ipc_speedups_config)
        ipc_speedup[config_key] = ipc_speedups_config
      print(config_key + " MPKI")
      print(mpki_config)
      mpki[config_key] = mpki_config
      print(config_key + " imiss cyc")
      print(imiss_cycle_config)
      imiss_cycle[config_key] = imiss_cycle_config

    # benchmarks.append('Avg no outliers')
    benchmarks.append('Avg')
    # plot_data(benchmarks, ipc_speedup, 'IPC Speedup (%)', [0,10])
    # plot_data(benchmarks, ipc_speedup, 'IPC Speedup (%)', [0,50])
    plot_data(benchmarks, ipc_speedup, 'IPC Speedup (%)')
    plot_data(benchmarks, mpki, 'MPKI')
    plot_data(benchmarks, imiss_cycle, 'Icache Miss Cycles Per KI')

  except Exception as e:
    print(e)

def plot_data(benchmarks, data, ylabel_name, ylim=None):
  print(data)
  # colors = ['#800000', '#4363d8', '#f58231', '#3cb44b', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#e6beff', '#e6194b', '#000075', '#800000', '#9a6324', '#808080', '#ffffff', '#000000']
  colors = ['#4363d8', '#800000', '#f58231', '#3cb44b', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#e6beff', '#e6194b', '#000075', '#800000', '#9a6324', '#808080', '#ffffff', '#000000']
  ind = np.arange(len(benchmarks))
  width = 0.08
  fig, ax = plt.subplots(figsize=(14, 5.4), dpi=80)
  num_keys = len(data.keys())

  idx = 0
  start_id = -int(num_keys/2)
  for key in data.keys():
    hatch=''
    if idx % 2:
      hatch='\\\\'
    else:
      hatch='///'
    ax.bar(ind + (start_id+idx)*width, data[key], width=width, fill=False, hatch=hatch, color=colors[idx], edgecolor=colors[idx], label=key)
    idx += 1
  ax.set_xlabel("Benchmarks")
  ax.set_ylabel(ylabel_name)
  ax.set_xticks(ind)
  ax.set_xticklabels(benchmarks, rotation = 27, ha='right')
  if ylim != None:
    ax.set_ylim(ylim)
  # ax.legend(loc="upper left", ncols=2)
  ax.legend()


if __name__ == "__main__":
    # Create a parser for command-line arguments
    parser = argparse.ArgumentParser(description='Read descriptor file name')
    parser.add_argument('-d','--descriptor_name', required=True, help='Experiment descriptor name. Usage: -d exp2.json')
    parser.add_argument('-b','--baseline_name', required=True, help='Baseline config name. Usage: -b baseline')
    parser.add_argument('-s','--simulation_path', required=True, help='Simulation result path. Usage: -s /soe/$USER/allbench_home/simpoint_flow/simulations/')

    args = parser.parse_args()
    descriptor_filename = args.descriptor_name

    descriptor_data = read_descriptor_from_json(descriptor_filename)
    get_IPC(descriptor_data, args.baseline_name, args.simulation_path)
    plt.grid('x')
    plt.tight_layout()
    plt.show()
