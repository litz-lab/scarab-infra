import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# === Setup ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scarab_stats/')))
from scarab_stats import Experiment

experiment = Experiment("./data/atr.csv")
config = "stat_p280_v332"

# === Workloads ===
int_workloads = [
    "perlbench", "gcc", "mcf", "omnetpp", "xalancbmk",
    "x264", "deepsjeng", "leela", "exchange2", "xz"
]
fp_workloads = [
    "bwaves", "cactuBSSN", "namd", "parest", "povray",
    "lbm", "wrf", "blender", "cam4", "imagick",
    "nab", "fotonik3d", "roms"
]

# === Stats ===
int_stats = [
    'MAP_STAGE_ONPATH_INT_REG_ATOMIC_USE_COUNT_1_count',
    'MAP_STAGE_ONPATH_INT_REG_ATOMIC_USE_COUNT_2_count',
    'MAP_STAGE_ONPATH_INT_REG_ATOMIC_USE_COUNT_3_count',
    'MAP_STAGE_ONPATH_INT_REG_ATOMIC_USE_COUNT_4_count',
    'MAP_STAGE_ONPATH_INT_REG_ATOMIC_USE_COUNT_5_count',
    'MAP_STAGE_ONPATH_INT_REG_ATOMIC_USE_COUNT_6_count',
    'MAP_STAGE_ONPATH_INT_REG_ATOMIC_USE_COUNT_7_PLUS_count',
]
fp_stats = [s.replace("INT", "VEC") for s in int_stats]
all_stats = list(set(int_stats + fp_stats))
all_workloads = int_workloads + fp_workloads

# === Retrieve Data ===
data = experiment.retrieve_stats([config], all_stats, all_workloads)

int_values = [[data.get(f"{config} {wl} {stat}", 0.0) for wl in int_workloads] for stat in int_stats]
fp_values = [[data.get(f"{config} {wl} {stat}", 0.0) for wl in fp_workloads] for stat in fp_stats]


# === Plotting ===
int_array = np.array(int_values)
fp_array = np.array(fp_values)
int_pct = int_array / int_array.sum(axis=0, keepdims=True) * 100
fp_pct = fp_array / fp_array.sum(axis=0, keepdims=True) * 100
int_pct = np.concatenate([int_pct, np.mean(int_pct, axis=1, keepdims=True)], axis=1)
fp_pct = np.concatenate([fp_pct, np.mean(fp_pct, axis=1, keepdims=True)], axis=1)
int_workloads.append("avg")
fp_workloads.append("avg")

# === Plot settings ===
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
})

colors = [
    '#a6cee3', '#b2df8a', '#1f78b4', '#33a02c',
    '#fb9a99', '#fdbf6f', '#e31a1c',
]
labels = ['1', '2', '3', '4', '5', '6', '7+']

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.5, 6.5), sharey=True)

# INT Plot
x_int = np.arange(len(int_workloads))
bottom = np.zeros(len(int_workloads))
for i in range(len(int_stats)):
    ax1.bar(x_int, int_pct[i], width=0.6, label=labels[i], color=colors[i],
            bottom=bottom, edgecolor='black', linewidth=1.2)
    bottom += int_pct[i]
ax1.set_title("SPEC2017int")
ax1.set_xticks(x_int)
ax1.set_xticklabels(int_workloads, rotation=45, ha='right')

# FP Plot
x_fp = np.arange(len(fp_workloads))
bottom = np.zeros(len(fp_workloads))
for i in range(len(fp_stats)):
    ax2.bar(x_fp, fp_pct[i], width=0.6, label=labels[i], color=colors[i],
            bottom=bottom, edgecolor='black', linewidth=1.2)
    bottom += fp_pct[i]
ax2.set_title("SPEC2017fp")
ax2.set_xticks(x_fp)
ax2.set_xticklabels(fp_workloads, rotation=45, ha='right')

for ax in [ax1, ax2]:
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)

fig.legend(labels, loc='lower center', ncol=4, frameon=False)
plt.tight_layout(rect=[0, 0.04, 1, 1])
os.makedirs("fig", exist_ok=True)
plt.savefig("fig/12_consumer_sensitivity.png", dpi=300)
plt.close()
