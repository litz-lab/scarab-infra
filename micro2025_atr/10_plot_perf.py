#!/usr/bin/env python3

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# === Setup path for scarab_stats ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scarab_stats/')))
from scarab_stats import Experiment

# === Load experiment ===
experiment = Experiment("./data/atr.csv")

# === Workload groups ===
int_workloads = [
    "perlbench", "gcc", "mcf", "omnetpp", "xalancbmk",
    "x264", "deepsjeng", "leela", "exchange2", "xz"
]
fp_workloads = [
    "bwaves", "cactuBSSN", "namd", "parest", "povray",
    "lbm", "wrf", "blender", "cam4", "imagick",
    "nab", "fotonik3d", "roms"
]

all_workloads = int_workloads + fp_workloads
stat = 'Periodic_IPC'

# === Plot settings ===
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

colors = ['#6baed6', '#feb24c', '#31a354']
labels = ['atomic', 'nonspec', 'combined']


# === Retrieve speedup array ===
def get_speedup_array(data, workloads, cfg_suffix):
    categories = ['atomic', 'nonspec', 'combine']
    baseline = f"realistic_{cfg_suffix}"
    speedup_arr = []

    for cat in categories:
        cfg = f"{cat}_{cfg_suffix}_d0"
        row = []
        for wl in workloads:
            base_ipc = data.get(f"{baseline} {wl} {stat}", 1e-6)
            cur_ipc = data.get(f"{cfg} {wl} {stat}", 0.0)
            delta = (cur_ipc / base_ipc - 1.0) if base_ipc > 0 else 0.0
            row.append(delta)
        avg = sum(row) / len(row)
        row.append(avg)
        speedup_arr.append(row)
    return np.array(speedup_arr) * 100  # Convert to %


# === Plotting function ===
def plot_grouped_bar(ax, data, workloads, title, ylim):
    num_configs, num_workloads = data.shape
    x = np.arange(num_workloads)
    bar_width = 0.2
    offset = -(num_configs - 1) / 2 * bar_width

    for i in range(num_configs):
        ax.bar(x + offset + i * bar_width,
               data[i],
               width=bar_width,
               color=colors[i],
               label=labels[i],
               edgecolor='black',
               linewidth=1)

    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(workloads, rotation=45, ha='right')
    ax.set_ylabel("IPC Speedup (%)")
    ax.set_ylim(*ylim)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    for y in ax.get_yticks():
        ax.axhline(y, linestyle='--', color='gray', linewidth=0.5, alpha=0.5)


# === Main plot routine for given config ===
def generate_plot(cfg_suffix, ylim, label_tag):
    configs = [
        f"realistic_{cfg_suffix}",
        f"atomic_{cfg_suffix}_d0",
        f"nonspec_{cfg_suffix}_d0",
        f"combine_{cfg_suffix}_d0",
    ]
    data = experiment.retrieve_stats(configs, [stat], all_workloads)

    int_speedup = get_speedup_array(data, int_workloads, cfg_suffix)
    fp_speedup = get_speedup_array(data, fp_workloads, cfg_suffix)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 6.5), sharey=True)
    plot_grouped_bar(ax1, int_speedup, int_workloads + ["avg"], f"SPEC2017int@{label_tag}", ylim)
    plot_grouped_bar(ax2, fp_speedup, fp_workloads + ["avg"], f"SPEC2017fp@{label_tag}", ylim)

    legend_handles = [Patch(facecolor=colors[i], edgecolor='black', label=labels[i]) for i in range(len(labels))]
    fig.legend(handles=legend_handles, loc='lower center', ncol=3, frameon=False)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    os.makedirs("fig", exist_ok=True)
    plt.savefig(f"fig/fig10_ipc_speedup@{label_tag}.png", dpi=300)
    plt.close()


# === Run for both configs ===
generate_plot('p64_v64', (0, 50), '64')
generate_plot('p224_v224', (-2, 8), '224')
