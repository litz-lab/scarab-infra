# Adding a New Workload Suite

This guide walks through introducing a new workload suite (e.g., SPEC2017,
SPEC CPU2026, an internal benchmark) end-to-end: image build → `--perf`
characterisation → `--trace` collection → `--sim` execution.

For the mechanics of each pipeline see:
- [`README.perf.md`](README.perf.md) — descriptor schema, top-down collection
- [`README.trace.md`](README.trace.md) — descriptor schema, SimPoint pipeline
- The top-level [`README.md`](../README.md) — simulation flow (`--sim`, `--status`, `--visualize`, etc.)

## Concepts

A **workload suite** in scarab-infra is identified by a single Docker
image. One image can host multiple individual workloads, organised as
`suite/subsuite/workload` keys in `workloads/workloads_db.json`. For
example, the upstream `dcperf` image contains a `wdlbench` subsuite with
workloads `fibers_benchmark`, `random_benchmark`, etc.

`workloads_db.json` is the canonical workload registry; the pipelines
read it for memory sizing and write top-down / execution-time / trace
metadata into it.

## Layout

```
workloads/<suite>/
├── Dockerfile                          # required
├── workload_root_entrypoint.sh         # optional, runs as root at start
└── workload_user_entrypoint.sh         # optional, runs as user at start
```

The directory name `<suite>` is the image name passed to
`./sci --build-image <suite>` and the `image_name` field in JSON
descriptors.

## Step 1 — Dockerfile

Start from [`workloads/example/Dockerfile`](../workloads/example/Dockerfile)
or copy an existing workload that resembles yours
(`workloads/dcperf/Dockerfile` for benchmark suites,
`workloads/sysbench/Dockerfile` for server-client patterns).

Minimal skeleton:

```dockerfile
# syntax = edrevo/dockerfile-plus

FROM ubuntu:22.04

INCLUDE+ ./common/Dockerfile.common

USER root
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    <packages your workload needs>

WORKDIR $tmpdir
RUN git clone <workload repo> && cd <workload> && git checkout <pinned-sha>
RUN <build / install steps>

# Optional: register per-workload entrypoints (see Step 2)
COPY ./workloads/<suite>/workload_root_entrypoint.sh /usr/local/bin/workload_root_entrypoint.sh
COPY ./workloads/<suite>/workload_user_entrypoint.sh /usr/local/bin/workload_user_entrypoint.sh

CMD ["/bin/bash"]
```

Key rules:
1. **First line must be `# syntax = edrevo/dockerfile-plus`.** This
   enables the `INCLUDE+` directive used on line 4.
2. **`INCLUDE+ ./common/Dockerfile.common` after the `FROM` line.** This
   pulls in DynamoRIO, SimPoint, Scarab dependencies, pmu-tools, and
   the standard tool chain. Without it none of the pipelines will work.
3. **Pin upstream sources to a specific commit.** Use
   `git clone ... && git checkout <sha>` rather than `git clone --depth 1`
   so the image is reproducible.
4. **Install everything into `$tmpdir`** (= `/tmp_home` inside the
   container). The pipeline scripts and other infra assume this layout.
5. **Use `USER root` for installs.** `Dockerfile.common` ends with the
   user switched away.

## Step 2 — Workload entrypoints (optional)

Two optional hook scripts let a workload customise container startup.
They are sourced by the always-on `root_entrypoint.sh` and
`user_entrypoint.sh` (provided by `common/scripts/`).

### `workload_root_entrypoint.sh`

Runs **as root** after user/group creation. Receives the workload's
`APPNAME` environment variable (set per-configuration in the descriptor)
as `$1` so a single script can branch across multiple workloads in the
same image.

Use it for setup that needs root: starting daemons, creating database
users, running `prepare` / install steps that touch system state.

Example (`workloads/sysbench/workload_root_entrypoint.sh`):
```bash
#!/bin/bash
APPNAME="$1"

if [ "$APPNAME" == "mysql" ]; then
  service mysql start
  mysql --execute="CREATE DATABASE sbtest;"
elif [ "$APPNAME" == "postgres" ]; then
  service postgresql start
  ...
fi
```

### `workload_user_entrypoint.sh`

Runs **as the descriptor user** after `root_entrypoint.sh` completes.
Use it to set workload-specific environment variables, activate Python
virtualenvs, or `cd` into the workload directory.

Example:
```bash
#!/bin/bash
cd /tmp_home/<workload-dir>
export LD_LIBRARY_PATH=$tmpdir/<workload-dir>/lib:$LD_LIBRARY_PATH
```

Both scripts are sourced only if present at `/usr/local/bin/`, so
copying them in the Dockerfile is opt-in.

## Step 3 — Build the image

```bash
./sci --build-image <suite>
```

This reads `workloads/<suite>/Dockerfile`, applies
`common/Dockerfile.common`, and tags the image with the current git hash
(`last_built_tag.txt` records the tag). Subsequent `--perf` / `--trace`
runs against descriptors that reference `"image_name": "<suite>"` will
pick up this image.

## Step 4 — Characterise with `--perf`

Run the workload under `pmu-tools/toplev` to capture top-down L1
metrics, execution time, and peak RSS. The peak RSS is consumed by
`--trace` to size Slurm memory requests.

Create `json/<suite>_perf.json`:

```json
{
  "descriptor_type": "perf",
  "user": "root",
  "root_dir": "/home/$USER",
  "image_name": "<suite>",
  "perf_configurations": [
    {
      "workload": "<workload>",
      "suite": "<suite>",
      "subsuite": "<subsuite>",
      "binary_cmd": "$tmpdir/<path-to-binary> <args>",
      "env_vars": null
    }
  ]
}
```

Run:
```bash
./sci --perf <suite>_perf
```

Results land in `workloads/workloads_db.json` under
`<suite>/<subsuite>/<workload>/performance`. See
[`README.perf.md`](README.perf.md) for the full pipeline.

For manual debugging in a perf-ready container:
```bash
./sci --perf-interactive <suite>_perf
```

## Step 5 — Trace with `--trace`

Collect SimPoint-based memory traces for cycle-accurate simulation in
Scarab.

Create `json/<suite>_trace.json`:

```json
{
  "descriptor_type": "trace",
  "workload_manager": "slurm",
  "root_dir": "/home/$USER",
  "scarab_path": "/home/$USER/scarab",
  "scarab_build": "opt",
  "traces_dir": "/home/$USER/traces",
  "application_dir": "/home/$USER/applications",
  "trace_name": "<suite>_traces",
  "trace_configurations": [
    {
      "workload": "<workload>",
      "image_name": "<suite>",
      "suite": "<suite>",
      "subsuite": "<subsuite>",
      "env_vars": null,
      "binary_cmd": "$tmpdir/<path-to-binary> <args>",
      "client_bincmd": null,
      "trace_type": "cluster_then_trace",
      "dynamorio_args": null,
      "clustering_k": null,
      "slurm_options": ""
    }
  ]
}
```

Run:
```bash
./sci --trace <suite>_trace
./sci --status <suite>_trace          # progress
```

`application_dir` is the path to a local benchmark application directory
(e.g., SPEC CPU installed tree). It is bind-mounted into the container at
`/tmp_home/application` during image build or tracing, replacing ISO-based
setups. Set it to `"."` if the application is already installed inside the
Docker image.

See [`README.trace.md`](README.trace.md) for `trace_type` modes,
resume semantics, multi-threaded workload handling, and the on-disk
layout of `traces_dir`.

## Step 6 — Simulate with `--sim`

Once traces are registered in `workloads_db.json`, point a simulation
descriptor at them and run Scarab.

Start from [`json/exp.json`](../json/exp.json) and edit the
`simulations` block to reference your `suite/subsuite/workload` keys
and the Scarab knobs you want to sweep.

```bash
./sci --build-scarab <suite>_sim
./sci --sim <suite>_sim
./sci --status <suite>_sim
```

## Worked example: hypothetical `spec2026`

Suppose you want to add a new SPEC CPU2026 suite with two workloads
(`502.gcc_r`, `505.mcf_r`). The flow is:

1. **Create `workloads/spec2026/Dockerfile`** that installs the SPEC
   tools (from a host-mounted ISO or tarball) and runs `runcpu
   --action build` for each target.

2. **Create `workloads/spec2026/workload_root_entrypoint.sh`** that, on
   `APPNAME=502.gcc_r`, exports the SPEC env file and `cd`s into the
   benchmark's run directory.

3. **Build**: `./sci --build-image spec2026`.

4. **Perf descriptor `json/spec2026_perf.json`** with one
   `perf_configurations` entry per workload, each `binary_cmd` set to
   the runcpu-emitted binary + reference input. Run
   `./sci --perf spec2026_perf` to populate `workloads_db.json`.

5. **Trace descriptor `json/spec2026_trace.json`** mirroring the perf
   descriptor's `binary_cmd`s. Run `./sci --trace spec2026_trace`
   under Slurm.

6. **Sim descriptor `json/spec2026_sim.json`** with a `simulations`
   block listing the two workloads and the Scarab configurations.
   `./sci --build-scarab spec2026_sim && ./sci --sim spec2026_sim`.

## Common gotchas

**Server-client workloads.** Put the *server* command in `binary_cmd`
(that's what gets traced) and the *client* load generator in
`client_bincmd`. The pipeline starts the client first, then traces the
server. See `workloads/sysbench/` for the canonical pattern.

**Multi-threaded Python workloads.** Set `OMP_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` in the Dockerfile or
`env_vars`. The trace pipeline handles helper-thread raw files
automatically (see `README.trace.md` § Multi-threaded Workloads) but
keeping the workload single-threaded gives cleaner measurements.

**Root vs regular user.** Some workloads (DCPerf, anything binding
privileged ports) need `"user": "root"` in the descriptor. The
entrypoints have been hardened to handle both paths since recent
releases.

**Pinning upstream sources.** Always `git checkout <sha>` after
`git clone` in the Dockerfile. Builds that float on `main` will
silently drift and break the `workloads_db.json` reproducibility
contract.

**Binary command discovery.** The binary command goes through
`drrun -- <binary_cmd>`. Wrapper scripts that fork (e.g., benchpress
CLIs, `runcpu`, shell wrappers) often hide the actual binary from
DynamoRIO. Trace the wrapper once interactively, find the final
`execve`, and use that path in `binary_cmd`.
