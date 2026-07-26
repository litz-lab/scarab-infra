# Collecting Performance Metrics with `./sci --perf`

This document describes how to collect microarchitectural performance metrics
(top-down analysis, execution time, peak RSS) for workloads running inside
Docker containers.

## Overview

`./sci --perf <descriptor>` runs each workload configuration defined in a JSON
descriptor, collects Intel top-down L1 metrics via `pmu-tools/toplev`, measures
execution time via `perf stat`, and records peak RSS. Results are written to
`workloads/workloads_db.json` and consumed downstream by `./sci --trace` for
Slurm memory sizing.

`./sci --perf-interactive <descriptor>` opens an interactive shell in a
perf-ready container for manual debugging or ad-hoc measurement.

## Requirements

PMU counters must be readable on the host:
```
sudo sh -c 'echo -1 > /proc/sys/kernel/perf_event_paranoid'
```

The Docker image should include `perf` and `pmu-tools`. When using
`--perf-interactive`, the entrypoint installs them automatically; for
pre-built images, include:
```dockerfile
USER root
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    linux-tools-common linux-tools-generic
RUN cd $tmpdir && git clone https://github.com/andikleen/pmu-tools.git
```

## Host preparation (the measurement core)

For clean, low-noise counts, reduce the host activity **before** a collection run and
restore it **after**. This frees enough general-purpose PMU counters for the
full event set (see [Counter multiplexing](#counter-multiplexing)) and keeps
other work off the measurement core. `run_perf.py` also probes for
multiplexing at startup and warns if the host still needs attention.

> **Core numbers are platform-dependent.**
>
> The commands below isolate a single **measurement core**, which must match
> `perf_core` in the descriptor (default `10`; see [Descriptor JSON](#descriptor-json)).  
>
> The examples target an 8-core box (cores `0-7`) and isolate **core 7**,
> so the "other cores" are `0-6` and the CPU affinity mask is `0x7f`.
> On a different platform, set the isolated core to your `perf_core`,
> list the remaining cores as `0-(N-1)` excluding it, and size the hex
> `smp_affinity` mask to all cores **except** the isolated one.
>
> For example, a 32-core box (cores `0-31`) isolating **core 10**:
> - `AllowedCPUs` for the fenced units → `0-9,11-31` (note the gap at 10);
>   keep `docker.service` on the full `0-31`.
> - `smp_affinity` mask → `fffffbff` (all 32 bits set except bit 10).

### Before `--perf`

This example assumes the host has 8 cores and uses core 7.

```bash
# 1. Free the fixed counter so the full event set doesn't multiplex
cat /proc/sys/kernel/nmi_watchdog  # If already 0 you can skip the next line (and no need to re-enable afterward)
sudo sysctl kernel.nmi_watchdog=0

# 2. Verify SMT/HT is off (it halves the per-thread GP counters).
cat /sys/devices/system/cpu/smt/active

# 3. Quiet the measurement core: stop IRQ balancing and steer IRQs to the
#    other cores (mask 0x7f = cores 0-6, i.e. everything except core 7)
sudo systemctl stop irqbalance
for f in /proc/irq/*/smp_affinity; do echo 7f | sudo tee "$f" >/dev/null 2>&1 || true; done

# 4. Fence non-Docker tasks off the measurement core
sudo systemctl set-property --runtime system.slice   AllowedCPUs=0-6   # daemons (sshd, cron, journald, …)
sudo systemctl set-property --runtime user.slice     AllowedCPUs=0-6   # login sessions / user@.service
sudo systemctl set-property --runtime init.scope     AllowedCPUs=0-6   # systemd (PID 1)
sudo systemctl set-property --runtime machine.slice  AllowedCPUs=0-6   # VMs / nspawn (harmless if empty)
sudo systemctl set-property --runtime docker.service AllowedCPUs=0-7   # keep Docker on the measurement core

# 5. Confirm the measurement core is idle before starting
mpstat -P 7 2 3
```

### After `--perf`

```bash
sudo sysctl kernel.nmi_watchdog=1            # re-enable watchdog
sudo systemctl start irqbalance              # restore IRQ balancing
for f in /proc/irq/*/smp_affinity; do echo ff | sudo tee "$f" >/dev/null 2>&1 || true; done

# undo the cpuset fences from step 4 (or just reboot)
for u in system.slice user.slice init.scope machine.slice docker.service; do
  sudo systemctl set-property --runtime "$u" AllowedCPUs=0-7
done
```

### Counter multiplexing

If more events are requested than there are usable general-purpose counters,
`perf` time-shares them and extrapolates, making the ratio metrics noisier.
The two most common causes are **SMT/HT** (halves the per-thread GP counters)
and the **NMI watchdog** (occupies one counter), the two things steps 1-2
above address. 
`run_perf.py` reports the minimum counter coverage as `min_coverage_pct` in
`hw_metrics`; a value below `100.0` indicates multiplexing occurred.

> **Reclaiming the HT counters may require a reboot, not a runtime toggle.**  
> You *can* offline the sibling threads at runtime with
> `echo off | sudo tee /sys/devices/system/cpu/smt/control`,
> but that only takes the logical CPUs offline;
> it does **not** give the surviving thread its counters back.
> Intel partitions the general-purpose counters between the two threads at boot
> based on the firmware HT setting, so the full per-core counter budget is
> only reclaimed by disabling Hyper-Threading in the BIOS/UEFI and rebooting. 
> If you can't change firmware / reboot the host, prefer a machine that
> already has HT disabled, or a CPU without HT, for perf collection.
> (For example, the 8-core "Bronze" has no HT and avoids multiplexing on the full event set.)

## Descriptor JSON

Create a perf descriptor (e.g., `json/perf.json`):

```json
{
  "descriptor_type": "perf",
  "user": "root",
  "root_dir": "/home/$USER",
  "image_name": "dcperf",
  "perf_configurations": [
    {
      "workload": "fibers_benchmark",
      "suite": "dcperf",
      "subsuite": "wdlbench",
      "binary_cmd": "$tmpdir/DCPerf/benchmarks/wdl_bench/fibers_benchmark",
      "env_vars": null
    }
  ]
}
```

| Field | Description |
|-------|-------------|
| `user` | Container user (`root` or a regular username) |
| `root_dir` | Host directory bind-mounted as the container home |
| `image_name` | Docker image name (built via `./sci --build-image`) |
| `perf_core` | (Optional) Logical CPU to pin all measurements to. |
| `max_repeats` | (Optional) Cap on repeats per workload. <br>You can force exactly this many repeats by setting `target_seconds` large enough. |
| `target_seconds` | (Optional) Target wall time per measurement. |
| `perf_configurations[]` | Array of workload configs to measure |

Each `perf_configurations` entry specifies:
- `workload`, `suite`, `subsuite`: keys for `workloads_db.json`
- `binary_cmd`: the command to execute inside the container
- `env_vars`: optional space-separated `KEY=VALUE` string

## Pipeline

### 1. Container setup
A privileged Docker container is created from the workload image. Support
scripts (`root_entrypoint.sh`, `perf_entrypoint.sh`, `user_entrypoint.sh`,
`utilities.sh`, plus optional per-image `workload_*_entrypoint.sh`) are
`docker cp`'d in on every run so iterating on host scripts does not require
an image rebuild. A `/tmp_home/.scarab_perf_ready` sentinel ensures the root
+ perf entrypoints execute exactly once per container lifetime.

### 2. Warmup run
The binary command runs once via `taskset -c 10` (pinned to a single core).
Peak RSS of the process tree is captured via Python's
`resource.RUSAGE_CHILDREN` and persisted to `workloads_db.json`.

### 3. Adaptive repeat scaling
The number of measurement repeats is computed from the warmup wall time so
total measurement targets ~600 seconds:
```
repeat_count = max(1, min(15, int((600 / single_run_seconds * 2 + 1) // 2)))
```
Fast workloads get more repeats for noise reduction; slow workloads run
fewer times to keep wall clock bounded.

### 4. Top-down L1 analysis
```
taskset -c 10 python3 /tmp_home/pmu-tools/toplev.py -l1 --single-thread -- <binary_cmd>
```
The output is parsed for the four L1 categories — **Frontend Bound**,
**Backend Bound**, **Bad Speculation**, **Retiring** — and stored under
`performance.topdown`.

### 5. Execution time
```
perf stat -C 10 -- taskset -c 10 <repeat_script>
```
Elapsed time per run is extracted from `perf stat` stderr and stored under
`performance.execution_time` as `{min, sec}`.

### 6. Results storage
Results are written to `workloads/workloads_db.json`, indexed by
`suite/subsuite/workload`:
```json
{
  "dcperf": {
    "wdlbench": {
      "fibers_benchmark": {
        "performance": {
          "topdown": {
            "retiring": 19.3,
            "bad_speculation": 4.2,
            "frontend_bound": 26.9,
            "backend_bound": 47.7
          },
          "execution_time": { "min": 0, "sec": 5.12 },
          "peak_rss_mb": 1240
        }
      }
    }
  }
}
```

## Usage

```bash
# Run automated perf collection for every entry in perf_configurations
./sci --perf perf

# Open an interactive shell in a perf-ready container
./sci --perf-interactive perf
```

## Multi-threaded workloads

All measurements pin the workload to a single core (`taskset -c 10`). For
multi-threaded workloads (e.g., Python with torch/numpy worker threads), set
`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and `OPENBLAS_NUM_THREADS=1` in the
Docker image or via `env_vars` to keep execution single-threaded for accurate
per-core measurement.

Peak RSS captures the entire process tree's memory, including helper
threads. This value is consumed by `./sci --trace` to size Slurm memory
requests for the trace pipeline.

## Single-user container reuse

The container is named `<image_name>_perf_collect_<user>`. Re-running
`./sci --perf` against the same descriptor reuses the container and skips
the root/perf bootstrap (the `.scarab_perf_ready` sentinel signals it has
already been initialised), so subsequent iterations are fast.

To force a fresh container, remove it manually:
```
docker rm -f <image_name>_perf_collect_<user>
```
