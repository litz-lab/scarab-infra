# Collecting Performance Metrics with `./sci --perf`

This document describes how to collect microarchitectural performance metrics
(top-down analysis, execution time, peak RSS) for workloads running inside
Docker containers.

## Overview

`./sci --perf <descriptor>` runs each workload configuration defined in a JSON
descriptor, collects Intel top-down L1 metrics via `pmu-tools/toplev`, measures
execution time via `perf stat`, and records peak RSS. Results are stored in
`workloads/workloads_db.json` and optionally auto-generate LaTeX tables for
paper integration.

## Requirements

PMU counters must be readable on the host:
```
sudo sh -c 'echo -1 > /proc/sys/kernel/perf_event_paranoid'
```

The Docker image should include `perf` and `pmu-tools`. When using
`--perf-interactive`, scarab-infra installs them automatically. For
pre-built images, include:
```dockerfile
USER root
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    linux-tools-common linux-tools-generic
RUN cd $tmpdir && git clone https://github.com/andikleen/pmu-tools.git
```

## Descriptor JSON

Create a perf descriptor (e.g., `json/agent_perf.json`):

```json
{
  "descriptor_type": "perf",
  "user": "root",
  "root_dir": "/path/to/docker/home",
  "image_name": "agent",
  "paper_output_dir": "/optional/path/to/paper/dir",
  "perf_configurations": [
    {
      "workload": "langchain_short_12iter",
      "suite": "agent",
      "subsuite": "langchain",
      "binary_cmd": "python3 /tmp_home/AgentCPU/workloads/run.py langchain ...",
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
| `paper_output_dir` | If set, auto-generate LaTeX macros/tables in `gen/` subdirectory |
| `perf_configurations[]` | Array of workload configs to measure |

Each configuration specifies:
- `workload`, `suite`, `subsuite`: keys for `workloads_db.json`
- `binary_cmd`: the command to execute (run inside the container)
- `env_vars`: optional environment variables (space-separated `KEY=VALUE` string)

## Pipeline

### 1. Container setup
A privileged Docker container is created with the workload image. Support
scripts (`root_entrypoint.sh`, `perf_entrypoint.sh`) are copied in and
executed to configure the environment.

### 2. Warmup run
The binary command runs once via `taskset -c 10` (pinned to a single core).
Peak RSS is captured through Python's `resource.RUSAGE_CHILDREN`.

### 3. Adaptive repeat scaling
The number of measurement repeats is computed to target ~600 seconds of total
measurement time:
```
repeat_count = max(1, min(15, int(600 / single_run_seconds)))
```

### 4. Top-down L1 analysis
```
taskset -c 10 python3 /tmp_home/pmu-tools/toplev.py -l1 --single-thread -- <binary_cmd>
```
Parses output for four L1 categories: **Frontend Bound**, **Backend Bound**,
**Bad Speculation**, **Retiring**.

### 5. Execution time
```
perf stat -C 10 -- taskset -c 10 <repeat_script>
```
Elapsed time per run is extracted from `perf stat` stderr.

### 6. Results storage
Results are written to `workloads/workloads_db.json`:
```json
{
  "agent": {
    "langchain": {
      "langchain_short_12iter": {
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

### 7. Paper table generation (optional)
If `paper_output_dir` is set, the pipeline auto-generates:
- `gen/paper_numbers.tex`: LaTeX macros for inline use (`\avgFE`, `\avgBE`, etc.)
- `gen/tab_topdown_data.tex`: data rows for the top-down table

## Usage

```bash
# Collect perf data
./sci --perf agent_perf

# Launch an interactive shell for manual measurement
./sci --perf-interactive agent_perf
```

## Multi-threaded workloads

All measurements pin the workload to a single core (`taskset -c 10`). For
multi-threaded workloads (e.g., Python with torch/numpy worker threads), the
environment variables `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and
`OPENBLAS_NUM_THREADS=1` should be set in the Docker image or via `env_vars`
to ensure single-threaded execution for accurate per-core measurement.

Peak RSS captures the entire process tree's memory, including helper threads.
This value is used by `--trace` to estimate Slurm memory requests.
