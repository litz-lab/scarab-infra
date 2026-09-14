# Scarab-infra

scarab-infra is a set of tools that automate the execution of Scarab simulations. It utilizes [Docker](https://www.docker.com/) and [Slurm](https://slurm.schedmd.com/documentation.html) to effectively simulate applications according to the [SimPoint](https://cseweb.ucsd.edu/~calder/simpoint/) methodology. Furthermore, scarab-infra provides tools to analyze generated simulation statistics and to obtain simpoints and execution traces from binary applications.

## Quickstart (Most Common Flow)

1. **Bootstrap the environment**
   ```
   ./sci --init
   ```
  This installs Docker when possible, configures socket permissions, installs Miniconda if needed, creates/updates the `scarabinfra` conda environment, validates activation, ensures you have an SSH key, and optionally fetches SimPoint traces, Slurm, ghcr.io credentials, and verifies AI CLI auth status (Codex/Gemini/Claude), and installs the agent guardrails described below.

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

## Using scarab-infra with a coding agent

Coding agents reliably reinvent what `sci` already does: counting `squeue` lines for run status, parsing `stats.out` with throwaway scripts, calling `make` or the Scarab binary directly. Those answers quietly disagree with sci's own — wrong stat column, missed failed cells, stale binaries — and the wasted sweeps are expensive. Two pieces ship in `tools/mcp/` to prevent it.

**`sci_mcp.py`** exposes sci over the [Model Context Protocol](https://modelcontextprotocol.io): `sci_status`, `sci_sim`, `sci_build_scarab`, `sci_collect_stats`, `sci_visualize`, `sci_perf_analyze`, `sci_kill`, `sci_list`. It is a stdio server with no third-party dependencies, so it runs wherever sci runs. There is nothing to start and no port to open: your agent spawns it as a child process and reaps it when the session ends.

**`force_sci_hook.py`** is what actually enforces. MCP only *adds* tools; it does not take Bash away, so a server alone changes nothing. This is a Claude Code `PreToolUse` hook that denies the hand-rolled equivalents and names the tool to use instead:

| denied | use instead |
| --- | --- |
| `squeue` / `sacct` filtered on `exp_*` | `sci_status` |
| python/awk/perl or glob aggregation over `stats.out`, `/simulations/`, `collected_stats.csv` | `sci_collect_stats` |
| `make ... scarab` | `sci_build_scarab` |
| direct `scarab_<hash>.opt` launch | `sci_sim` |

Anything beginning `./sci` is always allowed, and so is `cat` or `grep` on a **single** `stats.out` — blocking that would stop you debugging one simpoint. Aggregation is what gets caught, not inspection.

### Install

`./sci --init` installs both at **user scope**, so they apply from whatever directory you start your agent in, not only from this repo:

```
./sci --init
```

The step is idempotent and backs up `~/.claude/settings.json` before touching it. To install without a full bootstrap, run the two commands it runs:

```
claude mcp add --scope user scarab-infra -- python3 "$PWD/tools/mcp/sci_mcp.py"
```

and add to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "python3 \"/abs/path/to/scarab-infra/tools/mcp/force_sci_hook.py\" 2>/dev/null || true"
          }
        ]
      }
    ]
  }
}
```

The `|| true` matters: the hook then fires on every Bash call you make anywhere, so a moved checkout should cost you enforcement, not your shell.

Restart your agent afterwards — both files are read at session start. `/mcp` should then list `scarab-infra` as connected.

If you would rather scope this to the repo than to your account, `.mcp.json` and `.claude/settings.json` in this repo carry the same configuration via `$CLAUDE_PROJECT_DIR`, and apply when your agent's project root *is* scarab-infra.

### Other agents

The MCP server is protocol-standard, so Codex and Gemini CLI can both consume it — register `python3 <repo>/tools/mcp/sci_mcp.py` as an stdio server in their own config. The hook is Claude Code specific; those tools have no equivalent pre-tool veto, so with them the MCP tools are an option the agent has rather than a rule it must follow.

### Limits

A hook denial is a strong nudge, not a sandbox. The agent is told "use `sci_status`" and could in principle rephrase around the pattern, so the deny message instructs it to raise the gap with you rather than work around it. Making it airtight would need `deny` permission rules on `Bash(python3:*)` and friends, which blocks far too much legitimate work.

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
Generates bar charts (value and speedup) for each counter listed in `visualize.counters` and saves them next to `collected_stats.csv` under `<root_dir>/simulations/<descriptor>/`.

Use the descriptor structure:
```json
"visualize": {
  "baseline": "baseline",
  "counters": ["IPC"]
}
```

Each entry in `visualize.counters` can be either:
- a single counter name (e.g. `"IPC"`) to produce the existing bar and speedup plots, or
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

Set `visualize.baseline` to force the speedup plots to use a specific configuration as their reference (defaults to the first configuration present in the stats file).

### Analyze performance drift
```
./sci --perf-analyze <descriptor>
```
Diffs collected stats against a baseline configuration, writes a deterministic drift report, and optionally invokes an analyzer CLI (for example Codex, Gemini, or Claude) for root-cause hypotheses.

Use:
```json
"perf_analyze": {
  "baseline": "baseline",
  "counters": ["IPC", "ICACHE_MISS", "BRANCH_MISPRED"],
  "stat_groups": ["bp", "fetch", "core"],
  "compare_all_stats": false,
  "drift_top_workloads": 5,
  "drift_top_simpoints": 5,
  "prompt_budget_tokens": 12000,
  "threshold_pct": 2.0,
  "analyzer_cli_cmd": "codex"
}
```

Notes:
- `perf_analyze.counters[0]` is used as the trigger counter for drift detection.
- `stat_groups` optionally restricts compared stats to selected Scarab groups: `bp`, `core`, `fetch`, `inst`, `l2l1pref`, `memory`, `power`, `pref`, `stream`.
- Set `compare_all_stats: true` to compare every stat present in `collected_stats.csv`; `counters[0]` remains the drift trigger.
- `drift_top_workloads` controls how many highest-drift workloads (by trigger counter abs delta) are expanded in the report/prompt.
- `drift_top_simpoints` controls how many highest-impact simpoints per selected workload are expanded.
- `prompt_budget_tokens` limits prompt size (approximate token budgeting) before invoking the analyzer CLI.
- `threshold_pct` is an absolute percent-delta threshold.
- `analyzer_cli_cmd` supports `{prompt_file}`, `{summary_file}`, and `{report_file}` placeholders. If `{prompt_file}` is omitted, the prompt path is appended as the last argument.
- Example commands: `codex` (auto-converted to non-interactive `codex exec -`), `codex exec -`, `gemini -p "@{prompt_file}"`, `claude` (auto-converted to non-interactive `claude -p` and prompt content over stdin).
- During `./sci --init`, Codex/Gemini/Claude checks are non-interactive: init only verifies whether each CLI is installed and logged in, then prints manual login instructions when needed.
- Account-login commands:
  - Codex: `codex login` then `codex login status`
  - Gemini: `gemini`, then `/auth`, then `gemini auth status`
  - Claude: `claude login` (or `claude` then `/login`), then `claude auth status`
- If compared configurations use different Scarab binary hashes, `--perf-analyze` runs `git diff` in `scarab_path` and includes changed files/commit summaries in the report and AI prompt.
- Outputs are written beside `collected_stats.csv`: `perf_diff_summary.json`, `perf_drift_report.md`, `perf_drift_prompt.md`, and optionally `perf_ai_report.md`.

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

### Collect perf metrics for a workload
```
./sci --perf your_perf_descriptor
```
Uses `json/<descriptor>.json` with `descriptor_type: "perf"` to run automated top-down / execution-time / peak-RSS collection across every entry in `perf_configurations` (see [docs/README.perf.md](docs/README.perf.md)). For interactive debugging in a perf-ready container, use `./sci --perf-interactive <descriptor>`.

### Add a new workload suite
Introducing a new benchmark suite (Dockerfile, descriptors, end-to-end flow through `--perf` / `--trace` / `--sim`) is documented in [docs/README.add_workload.md](docs/README.add_workload.md).

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
