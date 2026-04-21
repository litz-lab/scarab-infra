# Collecting DynamoRIO Traces with `./sci --trace`

This document describes how to collect DynamoRIO memory traces using SimPoint
methodology for trace-driven, cycle-based simulation with Scarab.

## Overview

`./sci --trace <descriptor>` runs each workload configuration defined in a JSON
descriptor through a multi-stage pipeline: fingerprinting, SimPoint clustering,
per-segment tracing, raw-to-trace conversion, and trace minimization. Each
workload runs inside a Docker container scheduled as a Slurm job. Results are
stored under `traces_dir` and registered in `workloads/workloads_db.json`.

## Requirements

1. **Docker** — installed and accessible without sudo:
   ```
   sudo chmod 666 /var/run/docker.sock
   ```

2. **Slurm** — for workload scheduling (set `"workload_manager": "slurm"` in
   the descriptor). Manual mode is also supported.

3. **Docker image** — built via `./sci --build-image <image_name>`. The image
   must include DynamoRIO, SimPoint, and pmu-tools (provided by
   `common/Dockerfile.common`).

4. **Python environment** — activate the conda env:
   ```
   conda activate scarabinfra
   ```

5. **SSH key** — required if the Docker image clones private repos at build
   time. Add the machine's SSH key to your GitHub account.

## Descriptor JSON

Create a trace descriptor (e.g., `json/agent_trace.json`):

```json
{
  "descriptor_type": "trace",
  "workload_manager": "slurm",
  "root_dir": "/path/to/host/home",
  "application_dir": null,
  "scarab_path": "/path/to/scarab",
  "scarab_build": "opt",
  "traces_dir": "/path/to/output/traces",
  "trace_name": "trace_agent",
  "trace_parallel": 1,
  "raw2trace_parallel": 2,
  "trace_configurations": [
    {
      "workload": "langchain_short_12iter",
      "image_name": "agent",
      "suite": "agent",
      "subsuite": "langchain",
      "env_vars": null,
      "binary_cmd": "python3 $tmpdir/AgentCPU/workloads/run.py langchain ...",
      "client_bincmd": null,
      "trace_type": "cluster_then_trace",
      "dynamorio_args": null,
      "clustering_k": null,
      "slurm_options": ""
    }
  ]
}
```

| Field | Description |
|-------|-------------|
| `workload_manager` | `"slurm"` or `"manual"` |
| `root_dir` | Host directory bind-mounted as the container home |
| `application_dir` | Optional host dir mounted at `/tmp_home/application` (null if workload is in the image) |
| `scarab_path` | Path to the Scarab repository (used for post-processing) |
| `traces_dir` | Destination directory for final minimized traces |
| `trace_name` | Name for this tracing run (used in Slurm job names and directory layout) |
| `trace_parallel` | Max concurrent segment traces inside the container (default 1) |
| `raw2trace_parallel` | Max concurrent raw2trace processes inside the container (default 1) |
| `parallel_segments` | If true, each simpoint segment runs as an independent Slurm job (3-phase pipeline). Only for `cluster_then_trace`. Default false. |

Each configuration specifies:
- `workload`, `suite`, `subsuite`: keys for `workloads_db.json`
- `image_name`: Docker image name (built via `./sci --build-image`)
- `binary_cmd`: command to execute inside the container
- `client_bincmd`: optional client command (for client-server workloads)
- `trace_type`: `"cluster_then_trace"`, `"trace_then_cluster"`, or `"iterative_trace"`
- `dynamorio_args`: extra DynamoRIO flags (e.g., `-disable_rseq`)
- `clustering_k`: override SimPoint cluster count (null = auto)
- `slurm_options`: extra sbatch flags
- `trace_mem_mb`: optional per-workload memory override (MB)

## Trace Types

### `cluster_then_trace` (recommended)
1. Run the full workload under DynamoRIO with `libfpg.so` to produce a basic
   block fingerprint (BBV).
2. Run SimPoint clustering on the BBV to identify representative segments.
3. Trace only the representative segments (not the full execution).

This is the most memory-efficient approach for long-running workloads because
only selected segments are traced individually.

### `trace_then_cluster`
1. Trace the full workload under DynamoRIO.
2. Run SimPoint clustering on the resulting trace.
3. Extract representative segments from the full trace.

### `iterative_trace`
For workloads with discrete timesteps (e.g., server request loops). Each
timestep is traced separately with a timeout.

## Pipeline Stages

The pipeline runs inside a Docker container via `run_simpoint_trace.py`. Each
stage has resume support — if a stage's output already exists, it is skipped on
resubmission.

### 1. Fingerprinting

DynamoRIO runs the workload with the `libfpg.so` client to produce a basic
block vector (BBV) file. The segment size is 10M instructions by default.

For multi-threaded workloads, multiple BBV files are produced (one per thread).
The pipeline selects the main thread's BBV (the file with the most segments),
since helper threads contribute negligible instructions.

**Resume**: skipped if `bbv_file` already exists in `{workload_home}/`.

### 2. Clustering

SimPoint analyzes the BBV to identify representative segments. The output is
stored in `{workload_home}/simpoints/opt.p.lpt0.99` (99% coverage — segments
contributing <1% of execution are dropped).

Oversized simpoints are automatically adjusted via
`replace_oversized_simpoints.py`.

**Resume**: skipped if `simpoints/opt.p.lpt0.99` already exists.

### 3. Segment tracing

Each selected segment is traced individually by running the full workload under
DynamoRIO with `-trace_after_instrs` and `-trace_for_instrs` to capture only
the region of interest plus warmup.

```
drrun -opt_cleancall 2 -t drcachesim -jobs <DR_JOBS> -outdir <seg_dir> \
  -offline -count_fetched_instrs \
  -trace_after_instrs <roi_start> -trace_for_instrs <roi_length> \
  -- <binary_cmd>
```

Segments are traced sequentially by default (`TRACE_PARALLEL=1`) to stay within
Slurm memory limits. Each drrun process runs the full workload (~500MB–1GB RSS).
Set `"trace_parallel"` in the descriptor JSON to increase parallelism when
memory allows.

**Resume**: skipped if `<seg_dir>/raw/` contains `.raw.lz4` files.

### 4. Post-processing (portabilize)

For each segment, `portabilize_trace.py` rewrites `modules.log` paths so
traces are portable across machines. The original `modules.log` is backed up
to `modules.log.bak` so portabilize can be safely re-run.

**Resume**: skipped if `bin/` already contains >10 `.so` files.

### 5. Thread filtering

For multi-threaded workloads, DynamoRIO produces one `.raw.lz4` file per
thread inside `raw/window.0000/`. Helper threads (torch/numpy/sentence-
transformers worker threads) produce tiny raw files (~60KB each) but each
costs ~1GB+ during raw2trace decompression.

The pipeline keeps only the main thread's raw file (the largest one) and
removes all helper thread files before raw2trace. This is safe for Python
workloads because the GIL serializes bytecode execution — only the main
thread's trace contains meaningful instruction data.

### 6. Raw-to-trace conversion (raw2trace)

DynamoRIO's `drraw2trace` converts raw trace data into the final trace format:

```
drraw2trace -jobs 1 -indir <raw_path> -chunk_instr_count 10000000
```

The `-jobs 1` setting limits internal parallelism to control memory usage. A
single 500MB compressed raw trace can decompress to 100GB+ with `-jobs 2`.

Raw2trace runs sequentially by default (`RAW2TRACE_PARALLEL=1`). Set
`"raw2trace_parallel"` in the descriptor JSON to increase parallelism. After
thread filtering, each raw2trace processes a single ~500MB file, so 2–3 in
parallel is safe on machines with 32GB+ memory.

**Resume**: skipped if `trace/` contains `.trace.zip` files. The check
specifically looks for `drmemtrace.*.trace.zip` (not `cpu_schedule.bin.zip`,
which persists even after minimize deletes actual trace data).

### 7. Minimize

Traces are minimized and packaged into per-segment `.zip` files in
`traces_simp/trace/`. Each zip contains only the chunks needed for warmup +
simulation.

**Resume**: skipped if `<segment_id>.zip` already exists and is non-empty.

### 8. Finalization

When all trace jobs complete, `finish_trace` copies minimized traces to
`traces_dir` and updates `workloads/workloads_db.json` with trace paths and
SimPoint weights.

If Slurm is used, a dependent finalization job is automatically submitted
with `--dependency=afterany:<all_trace_job_ids>`.

## Parallel Segment Tracing

For workloads with many simpoints (50-76), the sequential trace pipeline can
take 15+ hours (raw2trace alone dominates). Setting `"parallel_segments": true`
in the descriptor enables a 3-phase Slurm pipeline that traces each segment as
an independent job:

```
Phase 1 (1 job):  Fingerprint + Cluster → produces opt.p.lpt0.99
    ↓ (Phase 1 script reads simpoints and submits Phase 2 jobs)
Phase 2 (N jobs): Trace + raw2trace + minimize for 1 segment each
    ↓ (--dependency=afterany)
Phase 3 (1 job):  Finalization (finish_trace)
```

This is only applicable to `cluster_then_trace` workloads. Other trace types
use the default single-job mode regardless of this setting.

### How it works

1. The host submits Phase 1 as a single Slurm job running mode 4 (`cluster_only`).
2. After Phase 1 completes inside the job, appended bash code reads
   `opt.p.lpt0.99`, skips already-complete segments, and submits one `sbatch`
   per remaining segment using a pre-generated Phase 2 template script (mode 5).
3. Each Phase 2 job runs `trace_single_segment()` — drrun, raw2trace, minimize
   for exactly one segment.
4. A Phase 3 finalization job is submitted with `--dependency=afterany` on all
   Phase 2 job IDs.

### Per-segment memory

Each segment job runs one drrun then one raw2trace sequentially, so memory
requirements are much lower than the full pipeline:

```
segment_mem_mb = max(SEGMENT_MEM_MIN_MB, peak_rss_mb * 1.5 + 1500)
```

| Constant | Default | Description |
|----------|---------|-------------|
| `SEGMENT_MEM_MIN_MB` | 10000 | Minimum memory per segment job |
| `SEGMENT_MEM_OVERHEAD_MB` | 1500 | Fixed overhead for DR runtime |
| `SEGMENT_MEM_FALLBACK_MB` | 10000 | Used if `--perf` was never run |
| `SEGMENT_MEM_FACTOR` | 1.5 | Multiplier on peak RSS |

### Resume support

Kill mid-Phase-2 and resubmit: Phase 1 will skip (fingerprint and simpoints
already exist), and Phase 2 submission will skip segments where
`traces_simp/trace/{segment_id}.zip` already exists. Only incomplete segments
are re-submitted.

### Descriptor option

```json
{
  "parallel_segments": true
}
```

Default is `false`, preserving the existing single-job behavior.

## Completion Detection

A trace workload is considered complete when minimized `.zip` files exist in
`traces_simp/trace/` for all simpoints listed in `opt.p.lpt0.99`. The count
of actual zips is compared against the expected simpoint count.

## Memory Management

Slurm memory requests are computed from `workloads_db.json`:

```
trace_mem_mb = max(TRACE_MEM_MIN_MB, peak_rss_mb * factor + TRACE_MEM_OVERHEAD_MB)
```

| Constant | Default | Description |
|----------|---------|-------------|
| `TRACE_MEM_MIN_MB` | 32000 | Minimum memory request |
| `TRACE_MEM_OVERHEAD_MB` | 2000 | Fixed overhead for DR runtime |
| `TRACE_MEM_FALLBACK_MB` | 32000 | Used if `--perf` was never run |
| Factor (`cluster_then_trace`) | 2.5 | Multiplier on peak RSS |

If a job OOMs, the pipeline queries `sacct` for the prior failure and bumps
memory by 1.5x on resubmission.

Per-workload overrides can be set via `"trace_mem_mb"` in the descriptor.

## Multi-threaded Workloads

Python workloads using torch, numpy, or sentence-transformers spawn helper
threads for BLAS operations, model loading, and tokenization. These threads
are problematic for tracing:

1. **Fingerprinting**: Multiple BBV files are produced. The pipeline selects
   the main thread's BBV (most segments) since helper threads contribute
   negligible instructions under the GIL.

2. **Segment tracing**: Helper threads produce additional `.raw.lz4` files.
   Each is small (~60KB) but costs ~1GB+ during raw2trace decompression.
   The pipeline removes all but the largest raw file per segment before
   raw2trace.

3. **Thread pinning**: The Docker image should set `OMP_NUM_THREADS=1`,
   `MKL_NUM_THREADS=1`, and `OPENBLAS_NUM_THREADS=1` to minimize helper
   thread activity. This does not eliminate all threads (Python's import
   machinery, signal handler, and GC threads still run) but reduces them
   significantly.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DR_JOBS` | auto (2–40 based on CPU count) | DynamoRIO `-jobs` fanout for fingerprinting |
| `TRACE_PARALLEL` | 1 | Max concurrent segment traces (set via descriptor `trace_parallel`) |
| `RAW2TRACE_PARALLEL` | 1 | Max concurrent raw2trace processes (set via descriptor `raw2trace_parallel`) |
| `OMP_NUM_THREADS` | 1 (set in Dockerfile) | OpenMP thread count |
| `MKL_NUM_THREADS` | 1 (set in Dockerfile) | MKL thread count |
| `OPENBLAS_NUM_THREADS` | 1 (set in Dockerfile) | OpenBLAS thread count |

## Usage

```bash
# Submit all trace jobs
./sci --trace agent_trace

# Check status of running/completed/failed jobs
./sci --status agent_trace

# Kill all running trace jobs
./sci --kill agent_trace

# Launch interactive shell for manual debugging
./sci --interactive agent_trace
```

## On-disk Layout

```
{trace_name}/{workload}/
├── bbv_file                    # Basic block vector (fingerprint)
├── simpoints/
│   └── opt.p.lpt0.99          # SimPoint output (segment_id cluster_id per line)
├── traces_simp/
│   ├── {segment_id}/           # Per-segment directory
│   │   ├── raw/
│   │   │   ├── window.0000/    # DynamoRIO raw output
│   │   │   │   └── *.raw.lz4  # Raw trace (main thread only after filtering)
│   │   │   ├── modules.log     # Portabilized module list
│   │   │   └── modules.log.bak # Original module list backup
│   │   ├── bin/                # Portabilized binaries
│   │   └── trace/              # raw2trace output
│   │       └── window.0000/
│   │           └── *.trace.zip
│   └── trace/                  # Minimized traces (final output)
│       ├── {segment_id}.zip
│       └── ...
└── logs/
    └── job_*.out               # Slurm job logs
```

## Troubleshooting

### OOM during raw2trace
Reduce parallelism: `TRACE_PARALLEL=1`, `RAW2TRACE_PARALLEL=1`. If still
OOMing, check for helper thread raw files in `raw/window.0000/` — there
should be only one `.raw.lz4` file (the main thread). Remove extras manually
and resubmit.

### Segment produces no raw data
Near-end-of-execution segments may fail to produce raw data for non-deterministic
workloads (instruction count varies between runs). Remove the problematic
segment from `opt.p.lpt0.99` if its weight is negligible (<1%).

### Resume not working
Ensure on-disk data is preserved between resubmissions. The pipeline checks
for existing outputs at each stage and skips completed work. If data was
deleted, the full pipeline runs from scratch.

### Stale raw/ directory
If a segment was re-traced, fresh raw data lands in `drmemtrace.*.dir/raw/`
while the old (empty) `raw/` directory persists. The pipeline detects this
and replaces the stale `raw/` with fresh data automatically.
