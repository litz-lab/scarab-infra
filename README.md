# Scarab-infra

scarab-infra is a set of tools that automate the execution of Scarab simulations. It utilizes [Docker](https://www.docker.com/) and [Slurm](https://slurm.schedmd.com/documentation.html) to effectively simulate applications according to the [SimPoint](https://cseweb.ucsd.edu/~calder/simpoint/) methodology. Furthermore, scarab-infra provides tools to analyze generated simulation statistics and to obtain simpoints and execution traces from binary applications.

## Quickstart (Most Common Flow)

1. **Bootstrap the environment**
   ```
   ./sci --init
   ```
   This installs Docker when possible, configures socket permissions, installs Miniconda if needed, creates/updates the `scarabinfra` conda environment, validates activation, ensures you have an SSH key, and optionally fetches SimPoint traces, Slurm, and ghcr.io credentials.

2. **Prepare (or update) your descriptor**
   ```
   cp json/exp.json json/<descriptor>.json
   # edit json/<descriptor>.json to match your workloads, configs, and paths
   ```
   Each descriptor specifies `root_dir`, `scarab_path`, workloads, and build mode. Adjust these fields before building or running. When you modify Scarab source codes that matter to this descriptor, make sure that you are modifying the right Scarab repo residing in `scarab_path`.

3. **Build Scarab for your descriptor**
   ```
   ./sci --build-scarab <descriptor>
   ```
   Provide the JSON filename (without extension) from `json/`. The build runs inside the correct Docker image and respects the `scarab_build` mode in the descriptor (defaults to `opt`).

4. **Run simulations**
   ```
   ./sci --sim <descriptor>
   ```
Launches the simulations defined in `json/<descriptor>.json`. Scarab runs in parallel across simpoints and reports status/logs under `<root_dir>/simulations/<descriptor>/` (with `root_dir` taken from the descriptor).

You only need additional steps if you want to inspect workloads, collect traces, or manage jobs manually. The sections below cover those workflows in more detail.

## Additional Workflows

### Monitor and clean up
- Check status (queued/running jobs, logs, and errors):
  ```
  ./sci --status <descriptor>
  ```
- Kill active simulations:
  ```
  ./sci --kill <descriptor>
  ```
- Remove containers and temporary state:
  ```
  ./sci --clean <descriptor>
  ```

### Visualize collected stats
```
./sci --visualize <descriptor>
```
Generates bar charts (value and speedup) for each counter listed in the descriptor’s `visualize_counters` field and saves them next to `collected_stats.csv` under `<root_dir>/simulations/<descriptor>/`.

Each entry in `visualize_counters` can be either:
- a single counter name (e.g. `"Periodic_IPC"`) to produce the existing bar and speedup plots, or
- a list of multiple counters (e.g. `["BTB_OFF_PATH_MISS_count", "BTB_OFF_PATH_HIT_count"]`) which will emit a stacked plot (`*_stacked.png`) combining those counters across workloads/configs.

For additional control you may instead supply objects such as:
```json
{
  "type": "stacked",
  "name": "btb_miss_hit",
  "title": "BTB Miss/Hit Breakdown",
  "y_label": "Events",
  "stats": ["BTB_OFF_PATH_MISS_count", "BTB_OFF_PATH_HIT_count"]
}
```
The `name` (optional) governs the output filename stem, while `title` and `y_label` adjust plot annotations.

Set `visualize_baseline` in the descriptor to force the speedup plots to use a specific configuration as their reference (defaults to the first configuration present in the stats file).

### List workloads and simulation modes
```
./sci --list
```
Shows the workload group hierarchy and the docker image each mode uses.

### Debug Scarab inside the Docker container

#### Build Scarab with the debug mode
Make sure to edit your json/<descriptor>.json to have `scarab_build`'s value `dbg`, and rebuild it with `./sci --build-scarab <descriptor>`

```
./sci --interactive <descriptor>
```

Then, go to the simulation directory under `~/simulations` inside the container where it is mounted to the descriptor’s `root_dir`, for example
```
cd ~/simulations/<exp_name>/baseline/<workload>/<simpoint>
```
Create a debug directory and copy the original PARAMS.out file as a new PARAMS.in, then cut the lines following after `--- Cut out everything below to use this file as PARAMS.in ---`
```
mkdir debug && cd debug
cp ../PARAMS.out ./PARAMS.in
```
Now, you can attach gdb with the same scarab parameters where you want to debug.
```
gdb /scarab/src/scarab
```

Cached images and containers are handled automatically by the commands above; use `./sci --clean <descriptor>` when you want to force a reset.

### Collect traces instead of running simulations
```
./sci --trace your_trace_descriptor
```
Uses `json/<descriptor>.json` with `descriptor_type: "trace"` to launch the trace pipeline (see [docs/README.trace.md](docs/README.trace.md) for details).

## Docker Images

`./sci --build-scarab <descriptor>` or `./sci --sim <descriptor>` automatically pulls or rebuilds the docker image it needs, but these commands are handy when you want to inspect or pre-stage images manually.

### Download a pre-built image for the current commit
```
export GIT_HASH=$(git rev-parse --short HEAD)
docker pull ghcr.io/litz-lab/scarab-infra/allbench_traces:$GIT_HASH
docker tag ghcr.io/litz-lab/scarab-infra/allbench_traces:$GIT_HASH allbench_traces:$GIT_HASH
```

### Build or retag a workload image yourself
```
./sci --build-image <workload_group>
```
Manual alternative:
```
export GIT_HASH=$(git rev-parse --short HEAD)
docker build . -f ./workloads/<workload_group>/Dockerfile --no-cache -t <workload_group>:$GIT_HASH
```

# Publications

```
@inproceedings{oh2024udp,
  author = {Oh, Surim and Xu, Mingsheng and Khan, Tanvir Ahmed and Kasikci, Baris and Litz, Heiner},
  title = {UDP: Utility-Driven Fetch Directed Instruction Prefetching},
  booktitle = {Proceedings of the 51st International Symposium on Computer Architecture (ISCA)},
  series = {ISCA 2024},
  year = {2024},
  month = jun,
}
```

## Requirements
> All of these checks are automated by `./sci --init`; follow them manually only if you need to diagnose issues locally.
1. Install Docker ([docs](https://docs.docker.com/engine/install/)).
2. Configure the Docker socket for non-root use ([ref](https://stackoverflow.com/questions/48957195/how-to-fix-docker-got-permission-denied-issue)):
   ```
   sudo chmod 666 /var/run/docker.sock
   ```
3. Install Miniconda (or Anaconda) so you have a writable Conda installation (see the [Miniconda docs](https://docs.conda.io/projects/miniconda/en/latest/)). `./sci --init` installs Miniconda to `~/miniconda3` if none is available.
4. Create or update the `scarabinfra` Conda environment from `quickstart_env.yaml`:
   ```
   conda env create --file quickstart_env.yaml
   ```
   The helper keeps this environment in sync (including `gdown` and other pip dependencies).
5. Activate or validate the environment as needed:
   ```
   conda activate scarabinfra
   ```
6. Add an SSH key for the machine running Docker to your GitHub account ([guide](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent?platform=linux)).
7. Place SimPoint traces under `$trace_home` (defaults to `~/traces`). A pre-packaged archive is available:
   ```
   cd ~/traces
   gdown https://drive.google.com/uc?id=1tfKL7wYK1mUqpCH8yPaPVvxk2UIAJrOX
   tar -xzvf simpoint_traces.tar.gz
   ```
8. Optional: Install [Slurm](docs/slurm_install_guide.md) if you plan to run simulations on a Slurm cluster.
9. Optional: Log in to ghcr.io so you can pull prebuilt images (requires a token with `read:packages`):
   ```
   echo <YOUR_GITHUB_TOKEN> | docker login ghcr.io -u <YOUR_GITHUB_USERNAME> --password-stdin
   ```
