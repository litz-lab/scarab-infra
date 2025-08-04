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
    'MAP_STAGE_ONPATH_INT_REG_NUM_NONBRANCH_count',
    'MAP_STAGE_ONPATH_INT_REG_NUM_NONEXCEPT_count',
    'MAP_STAGE_ONPATH_INT_REG_NUM_ATOMIC_count',
]
fp_stats = [s.replace("INT", "VEC") for s in int_stats]
all_stats = list(set(int_stats + fp_stats))
all_workloads = int_workloads + fp_workloads

# === Retrieve Data ===
data = experiment.retrieve_stats([config], all_stats, all_workloads)

int_values = np.array([[data.get(f"{config} {wl} {stat}", 0.0) for wl in int_workloads] for stat in int_stats])
fp_values = np.array([[data.get(f"{config} {wl} {stat}", 0.0) for wl in fp_workloads] for stat in fp_stats])

# === Compute Percentages ===
def compute_percentage(values):
    col_sum = values.sum(axis=0, keepdims=True)
    pct = values / np.where(col_sum == 0, 1, col_sum) * 100
    avg = np.mean(pct, axis=1, keepdims=True)
    return np.concatenate([pct, avg], axis=1)

int_pct = compute_percentage(int_values)
fp_pct = compute_percentage(fp_values)

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

colors = ['#fdae6b', '#ffd92f', '#9ecae1']
labels = ['nonbranch', 'nonexcept', 'atomic']

# === Plot Function ===
def plot_grouped_bar(ax, data, workloads, title):
    num_groups, num_workloads = data.shape
    x = np.arange(num_workloads)
    bar_width = 0.25
    offset = -(num_groups - 1) / 2 * bar_width

    for i in range(num_groups):
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
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 100)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    for y in ax.get_yticks():
        ax.axhline(y, linestyle='--', color='gray', linewidth=0.5, alpha=0.5)

# === Plotting ===
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 6.5), sharey=True)

plot_grouped_bar(ax1, int_pct, int_workloads, "SPEC2017int")
plot_grouped_bar(ax2, fp_pct, fp_workloads, "SPEC2017fp")

fig.legend(labels, loc='lower center', ncol=3, frameon=False)
plt.tight_layout(rect=[0, 0.04, 1, 1])
os.makedirs("fig", exist_ok=True)
plt.savefig("fig/06_ratio.png", dpi=300)
plt.close()
