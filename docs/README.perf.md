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
repeat_count = max(1, min(15, int(600 / single_run_seconds)))
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
