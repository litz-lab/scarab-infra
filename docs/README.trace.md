# Collecting DynamoRIO Traces with `./sci --trace`

This document describes how to collect DynamoRIO memory traces using SimPoint
methodology for trace-driven, cycle-based simulation with Scarab.

## Overview

`./sci --trace <descriptor>` runs each workload configuration defined in a JSON
descriptor through a multi-stage pipeline: fingerprinting, SimPoint clustering,
per-segment tracing, raw-to-trace conversion, and trace minimization. Each
workload runs inside a Docker container scheduled as a Slurm job (or in
manual mode). Results are stored under `traces_dir` and registered in
`workloads/workloads_db.json`.

## Requirements

1. **Docker** — installed and accessible without sudo:
   ```
   sudo chmod 666 /var/run/docker.sock
   ```

2. **Slurm** (optional) — for cluster scheduling, set
   `"workload_manager": "slurm"` in the descriptor. Manual mode is also
   supported with `"workload_manager": "manual"`.

3. **Docker image** — built via `./sci --build-image <image_name>`. The image
   must include DynamoRIO, SimPoint, and pmu-tools (provided by
   `common/Dockerfile.common`).

4. **Python environment** — activate the conda env:
   ```
   conda env create --file quickstart_env.yaml   # one-time
   conda activate scarabinfra
   ```

5. **SSH key** — required if the Docker image clones private repos at build
   time. Add the machine's SSH key to your GitHub account.

## Descriptor JSON

Create a trace descriptor (e.g., `json/trace.json`). A minimal example using
the DCPerf `fibers_benchmark` workload:

```json
{
  "descriptor_type": "trace",
  "workload_manager": "manual",
  "root_dir": "/home/$USER",
  "application_dir": "/home/$USER/applications",
  "scarab_path": "/home/$USER/allbench_home/scarab",
  "scarab_build": "opt",
  "traces_dir": "/home/$USER/lab/traces",
  "trace_name": "trace_dcperf",
  "trace_configurations": [
    {
      "workload": "fibers_benchmark",
      "image_name": "dcperf",
      "suite": "dcperf",
      "subsuite": "wdlbench",
      "env_vars": null,
      "binary_cmd": "$tmpdir/DCPerf/benchmarks/wdl_bench/fibers_benchmark",
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
| `application_dir` | Optional host dir mounted at `/tmp_home/application` (null if the workload is baked into the image) |
| `scarab_path` | Path to the Scarab repository (used for post-processing) |
| `traces_dir` | Destination directory for final minimized traces |
| `trace_name` | Name for this tracing run (used in Slurm job names and directory layout) |
| `trace_parallel` | Max concurrent segment traces inside the container (default 1) |
| `raw2trace_parallel` | Max concurrent raw2trace processes inside the container (default 1) |

Each configuration entry specifies:
- `workload`, `suite`, `subsuite`: keys for `workloads_db.json`
- `image_name`: Docker image name (built via `./sci --build-image`)
- `binary_cmd`: command to execute inside the container
- `client_bincmd`: optional client command (for client-server workloads)
- `trace_type`: `"cluster_then_trace"`, `"trace_then_cluster"`, or `"iterative_trace"`
- `dynamorio_args`: extra DynamoRIO flags
- `clustering_k`: override SimPoint cluster count (null = auto)
- `slurm_options`: extra sbatch flags (ignored in manual mode)
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

**Debugging tip:** `trace_then_cluster` is easier to debug because
everything runs off one fixed trace — any issue can be reproduced exactly.
`cluster_then_trace` uses two separate executions (fingerprint run, then
trace run), so non-deterministic failures may not reproduce across runs.

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
since helper threads contribute negligible instructions under the GIL.

**Resume**: skipped if `fingerprint/bbfp.*` and `fingerprint/segment_size`
already exist.

### 2. Clustering

SimPoint analyzes the BBV to identify representative segments. The output is
stored in `{workload_home}/simpoints/opt.p.lpt0.99` (99% coverage).
Oversized simpoints are automatically adjusted via
`replace_oversized_simpoints.py`.

**Resume**: skipped if `simpoints/opt.p` already exists.

### 3. Segment tracing

Each selected segment is traced individually by running the full workload under
DynamoRIO with `-trace_after_instrs` and `-trace_for_instrs` to capture only
the region of interest plus warmup:

```
drrun -opt_cleancall 2 -disable_rseq -t drcachesim -jobs <DR_JOBS> \
  -outdir <seg_dir> -offline -count_fetched_instrs \
  -trace_after_instrs <roi_start> -trace_for_instrs <roi_length> \
  -- <binary_cmd>
```

Segments are traced sequentially by default (`TRACE_PARALLEL=1`) to stay within
container memory limits. Each drrun process runs the full workload
(~500 MB – 1 GB RSS). Set `"trace_parallel"` in the descriptor JSON to
increase parallelism when memory allows.

**Resume**: skipped if `<seg_dir>/raw/` already contains `.raw.lz4` files.

### 4. Post-processing (portabilize)

For each segment, `portabilize_trace.py` rewrites `modules.log` paths so
traces are portable across machines. The original `modules.log` is backed up
to `modules.log.bak` so portabilize can be safely re-run.

**Resume**: skipped if `bin/` already contains more than 10 `.so` files.

### 5. Thread filtering

For multi-threaded workloads, DynamoRIO produces one `.raw.lz4` file per
thread inside `raw/window.0000/`. Helper threads (torch / numpy /
sentence-transformers worker threads) produce tiny raw files but each costs
~1 GB+ during raw2trace decompression.

The pipeline keeps only the main thread's raw file (the largest one) and
removes all helper-thread files before raw2trace. This is safe for Python
workloads because the GIL serializes bytecode execution: only the main
thread's trace contains meaningful instruction data.

### 6. Raw-to-trace conversion (raw2trace)

DynamoRIO's `drraw2trace` converts raw trace data into the final trace format:

```
drraw2trace -jobs 1 -indir <raw_path> -chunk_instr_count 10000000
```

`-jobs 1` is used per-segment to bound memory: a single 500 MB compressed raw
trace can decompress to 100 GB+ with higher fan-out.

**Resume**: skipped if `trace/` contains `.trace.zip` files. The check
specifically looks for `drmemtrace.*.trace.zip` (not `cpu_schedule.bin.zip`,
which persists even after minimize deletes actual trace data).

### 7. Minimize

Traces are minimized and packaged into per-segment `.zip` files in
`traces_simp/trace/`. Each zip contains only the chunks needed for warmup +
simulation. If multiple `.trace.zip` files exist for a segment (multi-thread
workloads), the largest is kept and the rest deleted.

If raw2trace produces no trace for a segment, the pipeline fails hard by
default so the user notices that the workload is non-deterministic (the
selected simpoint fell past end-of-execution on this trace run). To opt in
to silent recovery for sub-1%-weight segments, set
`EMPTY_SEGMENT_ACTION=remove` on the trace configuration's `env_vars`. With
that flag, eligible segments are removed from `opt.p.lpt0.99` and
`opt.w.lpt0.99` (originals backed up to `.bak`) and the remaining weights
renormalised. Segments at or above 1% still fail hard even with the flag.
Silent removal biases the simulated workload toward earlier phases, so the
conservative default is recommended unless you have characterised the
workload and accepted the bias.

**Resume**: skipped if `<segment_id>.zip` already exists and is non-empty.

### 8. Finalization

When all trace jobs complete, `finish_trace` copies minimized traces to
`traces_dir` and updates `workloads/workloads_db.json` with trace paths and
SimPoint weights.

## Parallel Segment Tracing

For workloads with many simpoints (50+), the default single-job pipeline
spends most of its time serially tracing one segment after another inside a
single Slurm allocation. Setting `"parallel_segments": true` in the
descriptor switches to a 3-phase Slurm pipeline that traces each segment as
an independent job:

```
Phase 1 (1 job):   Fingerprint + Cluster (mode 4)
    │             produces opt.p.lpt0.99, opt.w.lpt0.99
    ▼ (Phase 1 script reads opt.p.lpt0.99 and submits Phase 2 jobs)
Phase 2 (N jobs):  Trace + raw2trace + minimize one segment each (mode 5)
    │
    ▼ (--dependency=afterany:<all Phase 2 job ids>)
Phase 3 (1 job):   finish_trace (consolidates traces + updates workloads_db)
```

Only valid with `workload_manager: "slurm"` and `trace_type:
"cluster_then_trace"`. The descriptor validator rejects mixed
configurations.

**Important: cross-node ISA consistency.** When fingerprinting (Phase 1)
and segment tracing (Phase 2) land on nodes with different CPU generations,
runtime CPU dispatch (e.g., glibc `ifunc` resolvers) can change the
executed instruction stream, silently misaligning the BBV and traced
segments. The workload binary itself is fixed inside the Docker image, so
this only affects ISA-dependent runtime dispatchers. To avoid this, pin
all tracing jobs to the same node type via `slurm_options` (e.g.,
`"--constraint=icelake"` or `"--nodelist=<node>"`).

### How it works

1. The host submits Phase 1 as a single Slurm job that runs
   `run_simpoint_trace.py` in mode 4 (`cluster_only`): fingerprint +
   clustering only, no tracing.
2. The Phase 1 sbatch script has appended bash that, after mode 4 returns,
   reads `opt.p.lpt0.99`, skips segments where
   `traces_simp/trace/{segment_id}.zip` already exists, and `sbatch`'s one
   Phase 2 job per remaining segment using a pre-generated template script.
3. Each Phase 2 job runs `run_simpoint_trace.py` in mode 5
   (`trace_single_segment`) for one `(segment_id, cluster_id)` — drrun,
   raw2trace, minimize.
4. A Phase 3 `finish_trace` job is submitted with
   `--dependency=afterany:<all Phase 2 job IDs>` and runs the same
   finalisation that single-job mode runs synchronously on the host.

### Per-segment memory

A Phase 2 segment job runs one drrun then one raw2trace sequentially, so its
memory footprint is much smaller than a full single-job trace. PR δ ships
fixed defaults; a follow-up can wire `peak_rss_mb` from `workloads_db.json`:

| Constant | Default | Description |
|----------|---------|-------------|
| `PHASE1_MEM_DEFAULT_MB` | 32000 | Phase 1 (fingerprint + cluster) |
| `SEGMENT_MEM_MIN_MB` | 10000 | Per-segment Phase 2 job |
| `PHASE3_FINALIZE_MEM_MB` | 16384 | Phase 3 finalisation |

### Resume

Kill the Phase 1 job mid-flight and resubmit `./sci --trace <descriptor>`:
Phase 1 re-runs fingerprinting and clustering only if `opt.p.lpt0.99` is
missing (resume from inside `run_simpoint_trace.py`). After Phase 1
completes, the appended bash tail skips any segment whose minimized zip
already exists, so Phase 2 only re-submits incomplete segments.

### Status

`./sci --status <descriptor>` currently shows only Phase 1 jobs. To see
Phase 2 / Phase 3 progress, use `squeue -u <user>` directly. Full status
integration is a planned follow-up.

### Descriptor

```json
{
  "workload_manager": "slurm",
  "parallel_segments": true,
  "trace_configurations": [
    { "trace_type": "cluster_then_trace", ... }
  ]
}
```

Default is `false`, preserving the existing single-job behaviour.

## Completion Detection

A trace workload is considered complete when minimized `.zip` files exist in
`traces_simp/trace/` for all simpoints listed in `opt.p.lpt0.99`. The count
of actual zips is compared against the expected simpoint count.

## Memory Management

For Slurm runs, memory requests are computed from `workloads_db.json`:

```
trace_mem_mb = max(TRACE_MEM_MIN_MB, peak_rss_mb * factor + TRACE_MEM_OVERHEAD_MB)
```

| Constant | Default | Description |
|----------|---------|-------------|
| `TRACE_MEM_MIN_MB` | 8000 | Minimum memory request |
| `TRACE_MEM_OVERHEAD_MB` | 2000 | Fixed overhead for DR runtime |
| Factor (`cluster_then_trace`) | 2.5 | Multiplier on peak RSS |

Per-workload overrides can be set via `"trace_mem_mb"` in the descriptor.

## Multi-threaded Workloads

Python workloads using torch, numpy, or sentence-transformers spawn helper
threads for BLAS operations, model loading, and tokenization. These threads
are problematic for tracing in three ways, all handled automatically by the
pipeline:

1. **Fingerprinting**: Multiple BBV files are produced. The pipeline selects
   the main thread's BBV (most segments) since helper threads contribute
   negligible instructions under the GIL.

2. **Segment tracing**: Helper threads produce additional `.raw.lz4` files.
   Each is small (~60 KB) but costs ~1 GB+ during raw2trace decompression.
   The pipeline removes all but the largest raw file per segment before
   raw2trace.

3. **Thread pinning**: The Docker image should set `OMP_NUM_THREADS=1`,
   `MKL_NUM_THREADS=1`, and `OPENBLAS_NUM_THREADS=1` to minimize helper
   thread activity. The pipeline also sets these (plus `NUMEXPR_NUM_THREADS=1`
   and `TOKENIZERS_PARALLELISM=false`) on the fingerprint command line.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DR_JOBS` | auto (2–40, clamped by CPU count) | DynamoRIO `-jobs` fanout |
| `TRACE_PARALLEL` | 1 | Max concurrent segment traces (set via descriptor `trace_parallel`) |
| `RAW2TRACE_PARALLEL` | 1 | Max concurrent raw2trace processes (set via descriptor `raw2trace_parallel`) |
| `OMP_NUM_THREADS` | 1 (set in Dockerfile) | OpenMP thread count |
| `MKL_NUM_THREADS` | 1 (set in Dockerfile) | MKL thread count |
| `OPENBLAS_NUM_THREADS` | 1 (set in Dockerfile) | OpenBLAS thread count |

## Usage

```bash
# Submit all trace jobs
./sci --trace trace

# Check status of running/completed/failed jobs
./sci --status trace

# Kill all running trace jobs
./sci --kill trace

# Launch interactive shell for manual debugging
./sci --interactive trace
```

## On-disk Layout

```
{trace_name}/{workload}/
├── fingerprint/
│   ├── bbfp.*                  # Per-thread basic block vectors
│   ├── pcmap.*                 # Per-thread bb_id -> PC map
│   └── segment_size            # Segment size in instructions
├── simpoints/
│   ├── opt.p.lpt0.99           # SimPoint output (segment_id cluster_id per line)
│   └── opt.w.lpt0.99           # SimPoint weights (one per cluster, sums to 1.0)
├── traces_simp/
│   ├── {segment_id}/           # Per-segment directory
│   │   ├── raw/
│   │   │   ├── window.0000/    # DynamoRIO raw output
│   │   │   │   └── *.raw.lz4   # Raw trace (main thread only after filtering)
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
Reduce parallelism: `trace_parallel=1`, `raw2trace_parallel=1` in the
descriptor. If still OOMing, check for helper thread raw files in
`raw/window.0000/` — there should be only one `.raw.lz4` file (the main
thread). Remove extras manually and resubmit.

### Segment produces no raw data
Near-end-of-execution segments may fail to produce raw data for
non-deterministic workloads (instruction count varies between runs). The
pipeline fails hard by default. Two ways to recover:
- **Preferred**: investigate the source of non-determinism (Python GC,
  thread scheduling, randomised data structures) or switch this workload
  to `trace_then_cluster`, which is instruction-count consistent by
  construction.
- **Opt-in auto-removal**: set `EMPTY_SEGMENT_ACTION=remove` on the trace
  configuration's `env_vars`. Segments with simpoint weight below 1% will
  be removed from `opt.p.lpt0.99` / `opt.w.lpt0.99` (with `.bak` backups)
  and remaining weights renormalised. Note that this biases the simulated
  workload toward earlier phases.
- **Manual**: edit `opt.p.lpt0.99` and `opt.w.lpt0.99` to drop the
  segment and renormalise.

### Resume not working
Ensure on-disk data is preserved between resubmissions. The pipeline checks
for existing outputs at each stage and skips completed work. If data was
deleted, the full pipeline runs from scratch.

### Stale raw/ directory
If a segment was re-traced, fresh raw data lands in `drmemtrace.*.dir/raw/`
while the old (empty) `raw/` directory persists. The pipeline detects this
and replaces the stale `raw/` with fresh data automatically.

### DynamoRIO crash on multi-threaded workloads
Workloads that spawn many short-lived threads (Python with torch, faiss,
autogen, etc.) need `-disable_rseq` on `drrun` to avoid glibc rseq crashes.
The pipeline appends this flag automatically; if you call `drrun` directly
in custom scripts, include it.
