#!/usr/bin/python3

# 10/7/2024 | Alexander Symons | run_slurm.py
# 01/27/2025 | Surim Oh | slurm_runner.py

import os
import shlex
import shutil
import subprocess
import re
import sys
import traceback
import json
from pathlib import Path

# Per-simpoint memory scheduling constants
DEFAULT_MEM_MB = 4096    # fallback when no base_memory_mb data exists in workloads_db
MEM_HEADROOM_MB = 1024   # fixed +1 GiB on top of workloads_db base_memory_mb for Slurm --mem

# Trace-job memory scheduling constants.
# Formula: trace_mem_mb = max(TRACE_MEM_MIN_MB, peak_rss_mb * factor + TRACE_MEM_OVERHEAD_MB)
# where peak_rss_mb comes from workloads_db.json performance.peak_rss_mb
# (written by run_perf.py during the warmup run via /usr/bin/time -v).
# `factor` absorbs the DynamoRIO + libfpg/drraw2trace inflation over the bare
# workload RSS; `overhead` covers the fixed DR runtime + per-cluster raw2trace
# buffers. See discussion in run_perf / cluster_then_trace design notes.
TRACE_MEM_OVERHEAD_MB = 2000
TRACE_MEM_MIN_MB = 16000
TRACE_MEM_FALLBACK_MB = 16000  # used only if perf never ran for this workload
TRACE_MEM_FACTOR = {
    "cluster_then_trace": 2.5,   # parallel drraw2trace across clusters
    "trace_then_cluster": 1.8,   # sequential raw2trace post-process
    "iterative_trace":    2.0,
}


def estimate_trace_mem_mb(wl_entry, trace_type, descriptor_data=None):
    """Return the slurm --mem (MB) to request for a single trace job.

    Reads `performance.peak_rss_mb` written by run_perf.py from the workload
    entry in workloads_db.json and applies a trace-type-specific multiplier +
    fixed overhead. Falls back to TRACE_MEM_FALLBACK_MB if perf never ran.

    Descriptor-level overrides (optional):
        "mem": {
            "trace_fallback_mb": 20000,
            "trace_factor_cluster_then_trace": 3.0,
            "trace_factor_trace_then_cluster": 2.0,
            "trace_overhead_mb": 3000,
            "trace_min_mb": 6000,
        }
    """
    mem_cfg = (descriptor_data or {}).get("mem") or {}
    overhead = int(mem_cfg.get("trace_overhead_mb", TRACE_MEM_OVERHEAD_MB))
    min_mb = int(mem_cfg.get("trace_min_mb", TRACE_MEM_MIN_MB))
    fallback = int(mem_cfg.get("trace_fallback_mb", TRACE_MEM_FALLBACK_MB))
    factor = float(mem_cfg.get(
        f"trace_factor_{trace_type}",
        TRACE_MEM_FACTOR.get(trace_type, 2.0),
    ))

    peak_rss_mb = None
    try:
        peak_rss_mb = ((wl_entry or {}).get("performance") or {}).get("peak_rss_mb")
    except (AttributeError, TypeError):
        peak_rss_mb = None

    if peak_rss_mb is None:
        return fallback, None  # (mem_mb, measured_rss_mb)

    est = int(round(peak_rss_mb * factor)) + overhead
    return max(est, min_mb), peak_rss_mb


OOM_HEADROOM_FACTOR = 1.5  # request 1.5x the peak RSS of the last OOM-killed job

def _parse_maxrss_kb(maxrss_str):
    """Parse a sacct MaxRSS string like '9489220K', '4036M', '8G' to KB."""
    if not maxrss_str:
        return 0
    try:
        if maxrss_str.endswith("K"):
            return int(maxrss_str[:-1])
        elif maxrss_str.endswith("M"):
            return int(maxrss_str[:-1]) * 1024
        elif maxrss_str.endswith("G"):
            return int(maxrss_str[:-1]) * 1024 * 1024
        else:
            return int(maxrss_str)
    except ValueError:
        return 0

def get_oom_mem_mb(user, job_name_pattern):
    """Query sacct for the most recent OOM-killed job matching the pattern.

    Returns (mem_mb_to_request, actual_maxrss_mb) or (None, None) if no OOM found.
    The returned mem_mb includes headroom so the resubmission won't OOM again.

    sacct reports MaxRSS only on the .batch step (not the job-level row), so we
    first collect job IDs whose name matches and state is FAILED/CANCELLED, then
    look up the .batch step's MaxRSS for each.
    """
    try:
        result = subprocess.run(
            ["sacct", "-u", user,
             "--format=JobID%12,JobName%120,MaxRSS,State%20",
             "--parsable2", "--noheader", "-S", "now-2days"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            return None, None

        # Two passes: 1) collect failed job IDs by name, 2) find their .batch MaxRSS
        failed_job_ids = set()
        batch_rss = {}  # job_id -> maxrss_kb

        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 4:
                continue
            job_id, jname, maxrss_str, state = (
                parts[0].strip(), parts[1].strip(),
                parts[2].strip(), parts[3].strip(),
            )

            # Job-level row (no dot in job_id): collect failed IDs by name
            if "." not in job_id and job_name_pattern in jname:
                if "CANCEL" in state or "FAILED" in state or "OUT_OF_ME" in state:
                    failed_job_ids.add(job_id)

            # .batch step row: record MaxRSS
            if ".batch" in job_id:
                base_id = job_id.split(".")[0]
                rss_kb = _parse_maxrss_kb(maxrss_str)
                if rss_kb > 0:
                    batch_rss[base_id] = max(batch_rss.get(base_id, 0), rss_kb)

        # Find the highest MaxRSS among failed jobs for this workload
        best_rss_kb = 0
        for jid in failed_job_ids:
            rss = batch_rss.get(jid, 0)
            if rss > best_rss_kb:
                best_rss_kb = rss

        if best_rss_kb == 0:
            return None, None

        actual_mb = best_rss_kb // 1024
        request_mb = int(actual_mb * OOM_HEADROOM_FACTOR)
        return request_mb, actual_mb
    except Exception:
        return None, None


from .utilities import (
        err,
        warn,
        info,
        get_simpoints,
        get_mode_specific_base_memory,
        write_docker_command_to_file,
        prepare_simulation,
        finish_simulation,
        get_image_list,
        get_docker_prefix,
        prepare_trace,
        finish_trace,
        write_trace_docker_command_to_file,
        get_weight_by_cluster_id,
        image_exist,
        check_can_skip,
        remove_old_job_logs,
        print_simulation_status_summary,
        run_on_node,
        normalize_simulations,
        )

# Check if the docker image exists on available slurm nodes
# Inputs: list of available slurm nodes
# Output: list of nodes where the docker image is ready
def check_docker_image(nodes, docker_prefix, githash, slurm_options="", dbg_lvl = 1):
    try:
        available_nodes = []
        for node in nodes:
            # Check if the image exists
            # NOTE: This is deprecated, but if it is recycled, we need to incorporate slurm_options.
            # subprocess fails when there are extra spaces in the commands. Be careful when slurm_options=""
            image = subprocess.check_output(["srun", f"--nodelist={node}", "docker", "images", "-q", f"{docker_prefix}:{githash}"])
            info(f"{image}", dbg_lvl)
            if image == [] or image == b'':
                info(f"Couldn't find image {docker_prefix}:{githash} on {node}", dbg_lvl)
                continue

            available_nodes.append(node)
        return available_nodes
    except Exception as e:
        raise e


# Check if a container is running on the provided nodes, return those that are
# Inputs: list of nodes, docker_prefix, job_name, user
# Output: dictionary of node-containers
def check_docker_container_running(nodes, docker_prefix_list, job_name, user, dbg_lvl = 1):
    try:
        running_nodes_dockers = {}
        for node in nodes:
            running_nodes_dockers[node] = []

        for docker_prefix in docker_prefix_list:
            pattern = re.compile(fr"^{docker_prefix}_.*_{job_name}.*_.*_{user}$")
            for node in nodes:
                # Check container is running and no errors
                try:
                    dockers = subprocess.run(["srun", f"--nodelist={node}", "docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True, check=True)
                    lines = dockers.stdout.strip().split("\n") if dockers.stdout else []
                    for line in lines:
                        if pattern.match(line):
                            running_nodes_dockers[node].append(line)
                except:
                    err(f"Error while checking a running docker container named {docker_prefix}_.*_{job_name}_.*_.*_{user} on node {node}", dbg_lvl)

                    continue
        return running_nodes_dockers
    except Exception as e:
        raise e
    
# Check to see what tasks are currently queued or running on the slurm cluster
# Inputs: The docker container prefix(s), job name, and user
# Outputs: a dictionary of node names and the tasks running on them
def check_slurm_task_queued_or_running (docker_prefix_list, job_name, user, dbg_lvl = 1):
    output = subprocess.run(["squeue", "-u", user, "--format='%N %o'"], capture_output=True, text=True, check=True)
    output = output.stdout.split("\n")[1:-1] if output.stdout else []

    tasks_per_node = {}

    for docker_prefix in docker_prefix_list:
        # Naming scheme for run commands contains the docker prefix, experiment name, and user
        pattern = re.compile(fr"^{docker_prefix}_.*_{job_name}.*_.*_{user}_tmp_run.sh$")

        for line in output:
            # Get the node, and the command
            node, command = line.split(" ")
            node = node.strip("'")
            command = command.strip("'")

            # Remove the path from the command
            command = command[command.rfind("/")+1:]

            if pattern.match(command):
                command = command[:-11] # Remove the _tmp_run.sh part

                # Check if the command is already in the list
                if node not in tasks_per_node.keys():
                    tasks_per_node[node] = []

                # Add this command to the list of tasks for this node
                tasks_per_node[node].append(command)
    
    return tasks_per_node

        

# Check if a container is running on the provided nodes, return those that are
# Inputs: list of nodes, docker container name, path to container mount
# Output: list of nodes where the docker container was found running
# NOTE: Possible race condition where node was available but become full before srun,
# in which case this code will hang.
def check_docker_container_running_by_mount_path(nodes, container_name, mount_path, dbg_lvl = 1):
    try:
        running_nodes = []
        for node in nodes:
            # Check container is running and no errors
            try:
                mounts = subprocess.check_output(["srun", f"--nodelist={node}", "docker", "inspect", "-f", "'{{ .Mounts }}'", container_name])
            except:
                info(f"Couldn't find container {container_name} on {node}", dbg_lvl)
                continue

            mounts = mounts.decode("utf-8")

            # Check mount matches
            if mount_path not in mounts:
                warn(f"Couldn't find {mount_path} mounted on {node}.\nFound {mounts}", dbg_lvl)
                continue

            running_nodes.append(node)

        # NOTE: Could figure out mount here if all of them agree. Then it wouldn't need to be provided

        return running_nodes
    except Exception as e:
        raise e

# Check what containers are running in the slurm cluster
# Inputs: None
# Outputs: a list containing all node names that are currently available or None
def check_available_nodes(dbg_lvl = 1):
    try:
        # Query sinfo to get all lines with status information for all nodes
        # Ex: [['LocalQ*', 'up', 'infinite', '2', 'idle', 'bohr[3,5]']]
        try:
            response = subprocess.check_output(["sinfo", "-N"], timeout=10).decode("utf-8")
        except subprocess.TimeoutExpired as e:
            err("sinfo command timed out. Please check slurm control node.", dbg_lvl)
            raise e
        
        lines = [r.split() for r in response.split('\n') if r != ''][1:]

        # Check each node is up and available
        available = []
        all_nodes = []
        for line in lines:
            node = line[0]
            all_nodes.append(node)

            # Index -1 is STATE. Skip if not partially available
            if line[-1] != 'idle' and line[-1] != 'mix' and line[-1] != 'alloc':
                info(f"{node} is not available. It is '{line[-1]}'", dbg_lvl)
                continue

            # Now append node(s) to available list. May be single (bohr3) or multiple (bohr[3,5])
            if '[' in node:
                err(f"'sinfo -N' should not produce lists (such as bohr[2-5]).", dbg_lvl)
                print(f"     problematic entry was '{node}'")
                return None

            available.append(node)

        return available, all_nodes
    except Exception as e:
        raise e

def list_cluster_nodes(dbg_lvl = 1):
    try:
        response = subprocess.check_output(["sinfo", "-h", "-N", "-o", "%N %T"]).decode("utf-8")
    except subprocess.CalledProcessError as exc:
        warn(f"Unable to query slurm nodes: {exc}", dbg_lvl)
        return []
    except Exception as exc:
        warn(f"Unexpected error querying slurm nodes: {exc}", dbg_lvl)
        return []

    nodes = []
    seen = set()
    for line in response.splitlines():
        parts = line.split()
        if not parts:
            continue
        node = parts[0]
        state = parts[1] if len(parts) > 1 else ""
        if "[" in node:
            warn(f"Skipping aggregated sinfo entry '{node}'", dbg_lvl)
            continue
        if node in seen:
            continue
        seen.add(node)
        nodes.append((node, state))
    return nodes

# Get command to sbatch scarab runs. 1 core each, exclude nodes where container isn't running
# mem_mb: when provided, sets --mem explicitly; any --mem present in slurm_options is stripped
#         (deprecated) and a warning is emitted. When None, slurm_options is passed through
#         unchanged (used for trace jobs where memory is not auto-managed).
def generate_sbatch_command(experiment_dir, slurm_options="", mem_mb=None):
    if mem_mb is not None:
        cleaned = re.sub(r'--mem\s+\S+', '', slurm_options or '').strip()
        if cleaned != (slurm_options or '').strip():
            warn(f"'--mem' in slurm_options is deprecated; memory is now scheduled "
                 f"per-simpoint by the infra (computed value: {mem_mb}M). "
                 f"Remove '--mem' from slurm_options to suppress this warning.", dbg_lvl=2)
        slurm_options = (" " + cleaned if cleaned else "") + f" --mem {mem_mb}M"
    elif slurm_options:
        slurm_options = " " + slurm_options
    else:
        slurm_options = ""

    return f"sbatch -c 1{slurm_options} -o {experiment_dir}/logs/job_%j.out "
#return f"sbatch -c 1 --ntasks-per-core=2 --oversubscribe -o {experiment_dir}/logs/job_%j.out "


def _print_sbatch_output(result: subprocess.CompletedProcess) -> None:
    if result.returncode == 0:
        return
    out = (result.stdout or "").rstrip()
    err_out = (result.stderr or "").rstrip()
    if out:
        print(out)
    if err_out:
        print(err_out, file=sys.stderr)


# Print info of docker/slurm nodes and running experiment
def print_status(user, job_name, docker_prefix_list, descriptor_data, workloads_data, dbg_lvl = 1):
    # Get GitHash
    try:
        githash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("utf-8").strip()
        info(f"Git hash: {githash}", dbg_lvl)
    except FileNotFoundError:
        err("Error: 'git' command not found. Make sure Git is installed and in your PATH.", dbg_lvl)
    except subprocess.CalledProcessError:
        err("Error: Not in a Git repository or unable to retrieve Git hash.", dbg_lvl)

    info(f"Getting information about all nodes", dbg_lvl)

    try:
        available_slurm_nodes, all_nodes = check_available_nodes(dbg_lvl)
    except:
        exit(1)

    # General info, not helpful for --status <job>?
    print(f"Checking resource availability of slurm nodes:")
    for node in all_nodes:
        if node in available_slurm_nodes:
            print(f"\033[92mAVAILABLE:   {node}\033[0m")
        else:
            print(f"\033[31mUNAVAILABLE: {node}\033[0m")

    cmd = f"squeue --state=PENDING | grep -B 10000 '{user}' | wc -l"
    output = subprocess.check_output(cmd, shell=True, timeout=5).decode("utf-8").strip()
    jobs_ahead = int(output) - 2 if output.isnumeric() else -1 # subtract 2 to exclude the header line and job itself
    if jobs_ahead >= 0: print(f"Jobs ahead in queue: {jobs_ahead}")

    # Get dictionary of {node: [processes]}
    # NOTE: This is a list of run commands, not the actual containers. Container name will be same miunus tmp_run.sh
    slurm_running_sims = check_slurm_task_queued_or_running(docker_prefix_list, job_name, user, dbg_lvl)

    print(f"\nChecking what nodes currently have a running job with the following name(s):")
    for docker_prefix in docker_prefix_list:
        print(f"{docker_prefix}_*_{job_name}_*_*_{user}")

    print()

    running_sims = []
    queued_sims = []

    print(f"Summary of running simulations (by node): ")
    for key, val in slurm_running_sims.items():
        if key == '':
            continue
        print(f"{key}: {len(val)} Jobs")
        running_sims += val

    # Print queued jobs last
    if '' in slurm_running_sims.keys():
        print(f"Queued:     {len(slurm_running_sims[''])}")
        queued_sims += slurm_running_sims['']

    if slurm_running_sims == dict():
        print("No simulation jobs currently running")

    print()
    print_simulation_status_summary(
        descriptor_data,
        workloads_data,
        docker_prefix_list,
        user,
        running_sims,
        queued_sims,
        dbg_lvl=dbg_lvl,
        all_nodes=all_nodes,
        log_file_count_buffer=1,
        strict_log_count=True,
        log_count_offset=1,
        prep_failed_label="Failed - Slurm",
    )

def _get_trace_stage(trace_dir, workload):
    """Read the job log to determine the current stage of a trace job."""
    logs_dir = os.path.join(trace_dir, "logs")
    if not os.path.isdir(logs_dir):
        return None
    # Find the log file containing this workload (search newest first)
    import glob as _glob
    log_files = sorted(_glob.glob(os.path.join(logs_dir, "job_*.out")), reverse=True)
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                first_line = f.readline()
                if workload not in first_line:
                    continue
                # Found the right log — scan for stage markers
                content = first_line + f.read()
                # Ordered from latest to earliest stage
                stages = [
                    ("minimize traces done", "done"),
                    ("minimize traces..", "minimizing traces"),
                    ("clustered traces raw2trace done", "done"),
                    ("clustered traces raw2trace..", "raw2trace"),
                    ("cluster tracing done", "tracing done"),
                    ("clustering tracing..", "tracing simpoints"),
                    ("adjusting oversized simpoints..", "adjusting simpoints"),
                    ("clustering..", "clustering"),
                    ("generate fingerprint done", "fingerprint done"),
                    ("generate fingerprint..", "fingerprinting"),
                    ("running run_simpoint_trace.py", "starting"),
                    ("END prepare_docker_image", "image ready"),
                    ("Invoking", "building image"),
                    ("BEGIN prepare_docker_image", "waiting for image build"),
                ]
                last_stage = None
                for marker, label in stages:
                    if marker in content:
                        last_stage = label
                        break
                # Also extract runtime from the last "Runtime:" line
                import re
                runtimes = re.findall(r'Runtime: (\d+:\d+:\d+)', content)
                if last_stage and last_stage == "done" and runtimes:
                    return f"completed (last step: {runtimes[-1]})"
                return last_stage
        except (OSError, UnicodeDecodeError):
            continue
    return None


def print_trace_status(user, job_name, docker_prefix_list, dbg_lvl = 1):
    """Print status of slurm trace jobs: node availability, running/queued/completed counts."""
    try:
        githash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("utf-8").strip()
        info(f"Git hash: {githash}", dbg_lvl)
    except (FileNotFoundError, subprocess.CalledProcessError):
        githash = "unknown"

    try:
        available_slurm_nodes, all_nodes = check_available_nodes(dbg_lvl)
    except:
        exit(1)

    print(f"Checking resource availability of slurm nodes:")
    for node in all_nodes:
        if node in available_slurm_nodes:
            print(f"\033[92mAVAILABLE:   {node}\033[0m")
        else:
            print(f"\033[31mUNAVAILABLE: {node}\033[0m")

    # Deduplicate prefixes (all configs may share the same image)
    unique_prefixes = list(dict.fromkeys(docker_prefix_list))
    slurm_tasks = check_slurm_task_queued_or_running(unique_prefixes, job_name, user, dbg_lvl)

    running = []
    queued = []
    by_node = {}
    for node, tasks in slurm_tasks.items():
        if node == '':
            queued += tasks
        else:
            running += tasks
            if tasks:
                by_node[node] = tasks

    # Check for completed/in-progress traces on disk
    trace_dir = f"{os.path.expanduser('~')}/simpoint_flow/{job_name}"
    completed = []
    failed = []
    in_progress = []

    # Extract workload names from slurm job names so we can count jobs
    # that haven't created their trace directory yet as in-progress.
    # Container name format: {image}_{workload}_{trace_name}_{simpoint_mode}_{user}
    # e.g. "agent_langchain_short_250iter_trace_agent_cluster_then_trace_surim"
    all_slurm_workloads = set()
    for task_name in running + queued:
        for mode in ("cluster_then_trace", "trace_then_post_process", "iterative_trace"):
            suffix = f"_{job_name}_{mode}_{user}"
            for prefix in unique_prefixes:
                prefix_str = f"{prefix}_"
                if task_name.startswith(prefix_str) and task_name.endswith(suffix):
                    wl_name = task_name[len(prefix_str):-len(suffix)]
                    all_slurm_workloads.add(wl_name)

    # Scan trace directories on disk
    disk_workloads = set()
    if os.path.isdir(trace_dir):
        for wl_dir in sorted(os.listdir(trace_dir)):
            wl_path = os.path.join(trace_dir, wl_dir)
            if not os.path.isdir(wl_path) or wl_dir in ("logs", "scarab"):
                continue
            disk_workloads.add(wl_dir)
            fp_file = os.path.join(wl_path, "fingerprint", "segment_size")
            if os.path.isfile(fp_file):
                completed.append(wl_dir)
            else:
                is_running = any(wl_dir in t for t in running)
                is_queued = any(wl_dir in t for t in queued)
                if is_running:
                    in_progress.append(wl_dir)
                elif is_queued:
                    in_progress.append(wl_dir)
                else:
                    failed.append(wl_dir)

    # Count running/queued slurm jobs that haven't created a trace dir yet
    pending = []
    for wl_name in sorted(all_slurm_workloads):
        if wl_name not in disk_workloads:
            is_queued = any(wl_name in t for t in queued)
            if is_queued:
                pending.append(wl_name)
            else:
                in_progress.append(wl_name)

    total = len(completed) + len(in_progress) + len(pending) + len(failed)
    print(f"\nTrace job: {job_name}")
    print(f"Slurm jobs — Running: {len(running)}  |  Queued: {len(queued)}")
    print(f"Traces    — Completed: {len(completed)}  |  In-progress: {len(in_progress)}  |  Pending: {len(pending)}  |  Failed: {len(failed)}  |  Total: {total}")

    if by_node:
        print(f"\nRunning jobs by node:")
        for node, tasks in by_node.items():
            print(f"  {node}: {len(tasks)} jobs")
            for t in tasks:
                print(f"    {t}")

    if completed:
        print(f"\nCompleted ({len(completed)}):")
        for wl in completed:
            print(f"  \033[92m{wl}\033[0m")
    if in_progress:
        print(f"\nIn-progress ({len(in_progress)}):")
        for wl in in_progress:
            stage = _get_trace_stage(trace_dir, wl)
            if stage:
                print(f"  \033[93m{wl}\033[0m  — {stage}")
            else:
                print(f"  \033[93m{wl}\033[0m")
    if pending:
        print(f"\nPending ({len(pending)}):")
        for wl in pending:
            print(f"  \033[36m{wl}\033[0m")
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for wl in failed:
            print(f"  \033[31m{wl}\033[0m")


# Kills all jobs for job_name, if associated with user
def kill_jobs(user, job_name, docker_prefix_list, dbg_lvl = 2):
    # Kill and exit if killing jobs
    info(f"Killing all slurm jobs associated with {job_name}", dbg_lvl)

    # Format is JobID Name
    response = subprocess.check_output(["squeue", "-u", user, "--Format=JobID,Name:90"]).decode("utf-8")
    lines = [r.split() for r in response.split('\n') if r != ''][1:]

    # Filter to entries assocaited with this job (experiment or trace), and get job ids
    lines = list(filter(lambda x:job_name in x[1], lines))
    job_ids = list(map(lambda x:int(x[0]), lines))

    if lines:
        print("Found jobs: ")
        print(lines)

        confirm = input("Do you want to kill these jobs? (y/n): ").lower()
        if confirm == 'y':
            # Kill each job
            info(f"Killing jobs with slurm job ids: {', '.join(map(str, job_ids))}", dbg_lvl)
            for id in job_ids:
                try:
                    subprocess.check_call(["scancel", "-u", user, str(id)])
                except subprocess.CalledProcessError as e:
                    err(f"Couldn't cancel job with id {id}. Return code: {e.returncode}", dbg_lvl)

            # Wait for all cancelled jobs to actually finish
            print("Waiting for jobs to terminate...", end="", flush=True)
            for _ in range(60):  # up to 60 seconds
                try:
                    response = subprocess.check_output(
                        ["squeue", "-u", user, "--Format=JobID,Name:90"],
                    ).decode("utf-8")
                    remaining = [l.split() for l in response.strip().split('\n')[1:] if l.strip()]
                    remaining = [r for r in remaining if job_name in r[1]]
                    if not remaining:
                        break
                    print(".", end="", flush=True)
                    import time as _time
                    _time.sleep(1)
                except subprocess.CalledProcessError:
                    break
            print(" done.")
            return True
        else:
            print("Operation canceled.")
            return False
    else:
        print("No job found.")
        return True

def clean_containers(user, job_name, docker_prefix_list, dbg_lvl = 2):
    if not docker_prefix_list:
        info("No docker images associated with descriptor; nothing to clean.", dbg_lvl)
        return

    node_entries = list_cluster_nodes(dbg_lvl)
    if not node_entries:
        warn("No slurm nodes discovered; skipping remote container cleanup.", dbg_lvl)
        return

    all_nodes = []
    for node, state in node_entries:
        normalized_state = state.lower()
        if any(token in normalized_state for token in ("down", "drain", "fail", "maint", "unk")):
            info(f"Skipping node {node} in state '{state}'.", dbg_lvl)
            continue
        all_nodes.append(node)

    if not all_nodes:
        warn("All discovered slurm nodes are unavailable; skipping remote container cleanup.", dbg_lvl)
        return

    patterns = [re.compile(fr"^{docker_prefix}_.*_{job_name}.*_.*_{user}$") for docker_prefix in docker_prefix_list]
    removed_any = False

    for node in all_nodes:
        try:
            result = run_on_node(
                ["docker", "ps", "-a", "--format", "{{.Names}}"],
                node=node,
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired as exc:
            warn(f"Timed out while listing containers on {node}: {exc}", dbg_lvl)
            continue
        except subprocess.CalledProcessError as exc:
            warn(f"Failed to list containers on {node}: {exc}", dbg_lvl)
            continue
        except Exception as exc:
            warn(f"Error while listing containers on {node}: {exc}", dbg_lvl)
            continue

        container_names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        matching = []
        for name in container_names:
            if any(pattern.match(name) for pattern in patterns):
                matching.append(name)

        if not matching:
            info(f"No containers to remove on {node}.", dbg_lvl)
            continue

        for container in matching:
            try:
                run_on_node(["docker", "rm", "-f", container], node=node, check=True, timeout=30)
                print(f"Removed container {container} on {node}")
                removed_any = True
            except subprocess.TimeoutExpired as exc:
                err(f"Timed out removing container {container} on {node}: {exc}", dbg_lvl)
            except subprocess.CalledProcessError as exc:
                err(f"Failed to remove container {container} on {node}: {exc}", dbg_lvl)
            except Exception as exc:
                err(f"Unexpected error removing container {container} on {node}: {exc}", dbg_lvl)

    if not removed_any:
        print("No matching containers found on any slurm node.")

    clean_tmp_run_scripts(all_nodes, job_name, user, dbg_lvl)
    clean_docker_images(all_nodes, docker_prefix_list, dbg_lvl)

def clean_docker_images(nodes, docker_prefix_list, dbg_lvl = 2):
    """Remove all Docker images matching the workload prefixes on every node."""
    for node in nodes:
        for prefix in docker_prefix_list:
            try:
                result = run_on_node(
                    ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}", prefix],
                    node=node,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=30,
                )
            except Exception as exc:
                warn(f"Failed to list images on {node}: {exc}", dbg_lvl)
                continue

            images = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            for img in images:
                try:
                    run_on_node(["docker", "rmi", "-f", img], node=node, check=True, timeout=30)
                    print(f"Removed image {img} on {node}")
                except Exception as exc:
                    warn(f"Failed to remove image {img} on {node}: {exc}", dbg_lvl)

def clean_tmp_run_scripts(nodes, job_name, user, dbg_lvl = 2):
    pattern = re.compile(rf".*_{re.escape(job_name)}_.*_{re.escape(user)}_tmp_run\.sh$")
    for node in nodes:
        try:
            result = run_on_node(
                ["bash", "-lc", "find . -maxdepth 1 -type f -name '*_tmp_run.sh' -print"],
                node=node,
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
        except subprocess.TimeoutExpired as exc:
            warn(f"Timed out while listing temporary scripts on {node}: {exc}", dbg_lvl)
            continue
        except Exception as exc:
            warn(f"Failed to list temporary scripts on {node}: {exc}", dbg_lvl)
            continue
        files = []
        for line in result.stdout.splitlines():
            candidate = line.strip()
            if not candidate:
                continue
            name = os.path.basename(candidate)
            if pattern.match(name):
                files.append(candidate)
        if not files:
            info(f"No temporary run scripts on {node}.", dbg_lvl)
            continue
        for script in files:
            try:
                run_on_node(["rm", "-f", script], node=node, check=True, timeout=30)
                print(f"Removed {script} on {node}")
            except subprocess.TimeoutExpired as exc:
                err(f"Timed out removing {script} on {node}: {exc}", dbg_lvl)
            except subprocess.CalledProcessError as exc:
                err(f"Failed to remove {script} on {node}: {exc}", dbg_lvl)
            except Exception as exc:
                err(f"Unexpected error removing {script} on {node}: {exc}", dbg_lvl)

def run_simulation(user, descriptor_data, workloads_data, infra_dir, descriptor_path, dbg_lvl = 1):
    architecture = descriptor_data["architecture"]
    experiment_name = descriptor_data["experiment"]
    docker_home = descriptor_data["root_dir"]
    scarab_path = descriptor_data["scarab_path"]
    scarab_build = descriptor_data["scarab_build"]
    traces_dir = descriptor_data["traces_dir"]
    configs = descriptor_data["configurations"]
    simulations = descriptor_data["simulations"]
    application_dir = descriptor_data.get("application_dir", ".")
    total_sims = 0
    docker_prefix_list = get_image_list(simulations, workloads_data)

    def run_single_workload(suite, subsuite, workload, exp_cluster_id, sim_mode, warmup):
        try:
            docker_prefix = get_docker_prefix(sim_mode, workloads_data[suite][subsuite][workload]["simulation"])
            if not hasattr(run_single_workload, "submitted"):
                run_single_workload.submitted = 0   # initialize once

            info(f"Using docker image with name {docker_prefix}:{githash}", dbg_lvl)
            trace_warmup = None
            trace_type = ""
            trace_file = None
            env_vars = ""
            bincmd = ""
            client_bincmd = ""
            seg_size = None
            simulation_data = workloads_data[suite][subsuite][workload]["simulation"][sim_mode]
            if sim_mode == "memtrace":
                trace_warmup = simulation_data["warmup"]
                trace_type = simulation_data["trace_type"]
                trace_file = simulation_data["whole_trace_file"]
                seg_size = simulation_data["segment_size"]
            if sim_mode == "pt":
                trace_warmup = simulation_data["warmup"]
            if sim_mode == "exec":
                env_vars = simulation_data["env_vars"]
                bincmd = simulation_data["binary_cmd"]
                client_bincmd = simulation_data["client_bincmd"]
                seg_size = simulation_data["segment_size"]

            if "simpoints" not in workloads_data[suite][subsuite][workload].keys():
                weight = 1
                simpoints = {}
                simpoints["0"] = weight
            elif exp_cluster_id == None:
                simpoints = get_simpoints(workloads_data[suite][subsuite][workload], sim_mode, dbg_lvl)
            elif exp_cluster_id >= 0:
                weight = get_weight_by_cluster_id(exp_cluster_id, workloads_data[suite][subsuite][workload]["simpoints"])
                simpoints = {}
                simpoints[exp_cluster_id] = weight

            slurm_ids = []
            for config_key in configs:
                config = configs[config_key]["params"]
                binary_name = configs[config_key]["binary"]
                slurm_options = configs[config_key].get("slurm_options") or ""
                overhead_mb = configs[config_key].get("memory_overhead_mb") or 0
                if config == "":
                    config = None

                for cluster_id, weight in simpoints.items():
                    info(f"cluster_id: {cluster_id}, weight: {weight}", dbg_lvl)

                    docker_container_name = f"{docker_prefix}_{suite}_{subsuite}_{workload}_{experiment_name}_{config_key.replace("/", "-")}_{cluster_id}_{sim_mode}_{user}"

                    # TODO: Notification when a run fails, point to output file and command that caused failure
                    # Add help (?)
                    # Look into squeue -o https://slurm.schedmd.com/squeue.html
                    # Look into resource allocation

                    # TODO: Rewrite with sbatch arrays

                    # Compute per-simpoint memory request from workloads_db base_memory_mb_by_mode for this simulation mode.
                    base_memory_mb = None
                    try:
                        wl_entry = memory_db[suite][subsuite][workload]
                        lookup_mode = sim_mode
                        if str(cluster_id) == "0":
                            base_memory_mb = get_mode_specific_base_memory(wl_entry, lookup_mode, wl_entry)
                        else:
                            for sp in wl_entry["simpoints"]:
                                if str(sp["cluster_id"]) == str(cluster_id):
                                    base_memory_mb = get_mode_specific_base_memory(sp, lookup_mode, wl_entry)
                                    break
                    except (KeyError, TypeError):
                        pass
                    if base_memory_mb is not None:
                        mem_mb = int(base_memory_mb) + MEM_HEADROOM_MB + overhead_mb
                        fallback_mb = None
                    else:
                        fallback_mb = (descriptor_data.get("mem") or {}).get("fallback_mb") or DEFAULT_MEM_MB
                        mem_mb = fallback_mb + overhead_mb

                    sbatch_cmd = generate_sbatch_command(experiment_dir, slurm_options=slurm_options, mem_mb=mem_mb)

                    # Create temp file with run command and run it
                    filename = f"{docker_container_name}_tmp_run.sh"
                    slurm_running_sims = check_slurm_task_queued_or_running(docker_prefix_list, experiment_name, user, dbg_lvl)
                    running_sims = []
                    for node_list in slurm_running_sims.values():
                        running_sims += node_list

                    if check_can_skip(descriptor_data, config_key, suite, subsuite, workload, cluster_id, filename, sim_mode, user, slurm_queue=running_sims, dbg_lvl=dbg_lvl):
                        info(f"Skipping {workload} with config {config_key} and cluster id {cluster_id}", dbg_lvl)
                        continue

                    if fallback_mb is not None:
                        print(f"WARN: no base_memory_mb_by_mode entry for {suite}/{subsuite}/{workload} cluster={cluster_id} sim_mode={sim_mode} config={config_key}, using fallback={fallback_mb}MB + overhead={overhead_mb}MB = {mem_mb}MB")

                    workload_home = f"{suite}/{subsuite}/{workload}"
                    write_docker_command_to_file(user, local_uid, local_gid, workload, workload_home, experiment_name,
                                                 docker_prefix, docker_container_name, traces_dir,
                                                 docker_home, githash, config_key, config, sim_mode, binary_name,
                                                 seg_size, architecture, cluster_id, warmup, trace_warmup, trace_type,
                                                 trace_file, env_vars, bincmd, client_bincmd, filename, infra_dir, application_dir, slurm=True)
                    tmp_files.append(filename)

                    remove_old_job_logs(f"{experiment_dir}/logs", config_key, suite, subsuite, workload, cluster_id)

                    result = subprocess.run(["touch", f"{experiment_dir}/logs/job_%j.out"], capture_output=True, text=True, check=True)
                    result = subprocess.run((sbatch_cmd + filename).split(" "), capture_output=True, text=True)
                    _print_sbatch_output(result)
                    job_id = result.stdout.split(" ")[-1].strip()
                    slurm_ids.append(job_id)
                    run_single_workload.submitted += 1
                print("\rSubmitting jobs: "+str(run_single_workload.submitted), end=' ', flush=True)
            return slurm_ids
        except Exception as e:
            print(f"Error running workload {workload}: {e}")
            raise e

    tmp_files = []
    try:
        # Get user for commands
        user = subprocess.check_output("whoami").decode('utf-8')[:-1]
        info(f"User detected as {user}", dbg_lvl)

        # Get a local user/group ids
        local_uid = os.getuid()
        local_gid = os.getgid()

        # Get GitHash
        try:
            githash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("utf-8").strip()
            info(f"Git hash: {githash}", dbg_lvl)
        except FileNotFoundError:
            err("Error: 'git' command not found. Make sure Git is installed and in your PATH.", dbg_lvl)
        except subprocess.CalledProcessError:
            err("Error: Not in a Git repository or unable to retrieve Git hash.", dbg_lvl)

        scarab_binaries = []
        for sim in simulations:
            for config_key in configs:
                scarab_binary = configs[config_key]["binary"]
                if scarab_binary not in scarab_binaries:
                    scarab_binaries.append(scarab_binary)

        # Generate commands for executing in users docker and sbatching to nodes with containers
        experiment_dir = f"{descriptor_data['root_dir']}/simulations/{experiment_name}"

        try:
            scarab_githash, image_tag_list = prepare_simulation(user, scarab_path, scarab_build, descriptor_data['root_dir'], experiment_name, architecture, docker_prefix_list, githash, infra_dir, scarab_binaries, False, dbg_lvl)
        except RuntimeError as e:
            # This error prints a message. Now stop execution
            return

        # Load workloads_db.json for base_memory_mb_by_mode lookups.
        # When top_simpoint=true, workloads_data comes from workloads_top_simp.json which lacks
        # base_memory_mb_by_mode; always read from the full DB instead.
        try:
            with open(Path(infra_dir) / "workloads" / "workloads_db.json") as _fh:
                memory_db = json.load(_fh)
        except Exception:
            memory_db = workloads_data

        print("Submitting jobs...")
        # Iterate over each workload and config combo
        simulations = normalize_simulations(simulations)
        tmp_files = []
        for simulation in simulations:
            suite = simulation["suite"]
            subsuite = simulation["subsuite"]
            workload = simulation["workload"]
            exp_cluster_id = simulation["cluster_id"]
            sim_mode = simulation["simulation_type"]
            sim_warmup = simulation["warmup"]

            slurm_ids = []

            # Run all the workloads within suite
            if workload == None and subsuite == None:
                for subsuite_ in workloads_data[suite].keys():
                    for workload_ in workloads_data[suite][subsuite_].keys():
                        if not isinstance(workloads_data[suite][subsuite_][workload_], dict):
                            continue
                        sim_mode_ = sim_mode
                        if sim_mode_ == None:
                            sim_mode_ = workloads_data[suite][subsuite_][workload_]["simulation"]["prioritized_mode"]
                        if sim_warmup == None:  # Use the whole warmup available in the trace if not specified
                            sim_warmup = workloads_data[suite][subsuite_][workload_]["simulation"][sim_mode_]["warmup"]
                        slurm_ids += run_single_workload(suite, subsuite_, workload_, exp_cluster_id, sim_mode_, sim_warmup)
            elif workload == None and subsuite != None:
                for workload_ in workloads_data[suite][subsuite].keys():
                    if not isinstance(workloads_data[suite][subsuite][workload_], dict):
                        continue
                    sim_mode_ = sim_mode
                    if sim_mode_ == None:
                        sim_mode_ = workloads_data[suite][subsuite][workload_]["simulation"]["prioritized_mode"]
                    if sim_warmup == None:  # Use the whole warmup available in the trace if not specified
                        sim_warmup = workloads_data[suite][subsuite][workload_]["simulation"][sim_mode_]["warmup"]
                    slurm_ids += run_single_workload(suite, subsuite, workload_, exp_cluster_id, sim_mode_, sim_warmup)
            elif workload != None and subsuite == None:
                found = False
                for subsuite_ in workloads_data[suite].keys():
                    if not workload in workloads_data[suite][subsuite_].keys():
                        continue
                    found = True
                    _subsuite = subsuite_
                assert found, f"Workload {workload} could not be found for any subsuite of {suite}. Check descriptor validation code"
                sim_mode_ = sim_mode
                if sim_mode_ == None:
                    sim_mode_ = workloads_data[suite][_subsuite][workload]["simulation"]["prioritized_mode"]
                if sim_warmup == None:  # Use the whole warmup available in the trace if not specified
                    sim_warmup = workloads_data[suite][_subsuite][workload]["simulation"][sim_mode_]["warmup"]
                slurm_ids += run_single_workload(suite, _subsuite, workload, exp_cluster_id, sim_mode_, sim_warmup)
            else:
                sim_mode_ = sim_mode
                if sim_mode_ == None:
                    sim_mode_ = workloads_data[suite][subsuite][workload]["simulation"]["prioritized_mode"]
                if sim_warmup == None:  # Use the whole warmup available in the trace if not specified
                    sim_warmup = workloads_data[suite][subsuite][workload]["simulation"][sim_mode_]["warmup"]
                slurm_ids += run_single_workload(suite, subsuite, workload, exp_cluster_id, sim_mode_, sim_warmup)
        print("\nSubmitted all jobs")
        # Clean up temp files
        for tmp in tmp_files:
            info(f"Removing temporary run script {tmp}", dbg_lvl)
            os.remove(tmp)

        #finish_simulation(user, docker_home, descriptor_path, descriptor_data['root_dir'], experiment_name, image_tag_list, slurm_ids)

        # TODO: check resource capping policies, add kill/info options

        # TODO: (long term) add evalstats to json descriptor to run stats library with PMU counters
    except Exception as e:
        print("An exception occurred:", e)
        traceback.print_exc()  # Print the full stack trace

        # Clean up temp files
        for tmp in tmp_files:
            info(f"Removing temporary run script {tmp}", dbg_lvl)
            os.remove(tmp)

        kill_jobs(user, experiment_name, docker_prefix_list, dbg_lvl)

def run_tracing(user, descriptor_data, workload_db_path, infra_dir, dbg_lvl = 2, descriptor_path = None):
    trace_name = descriptor_data["trace_name"]
    docker_home = descriptor_data["root_dir"]
    scarab_path = descriptor_data["scarab_path"]
    scarab_build = descriptor_data["scarab_build"]
    traces_dir = descriptor_data["traces_dir"]
    trace_configs = descriptor_data["trace_configurations"]
    application_dir = descriptor_data["application_dir"]

    docker_prefix_list = []
    for config in trace_configs:
        image_name = config["image_name"]
        if image_name not in docker_prefix_list:
            docker_prefix_list.append(image_name)

    tmp_files = []
    slurm_ids = []

    def run_single_trace(workload, suite, subsuite, image_name, trace_name, env_vars, binary_cmd, client_bincmd, trace_type, drio_args, clustering_k, application_dir, slurm_options, config_mem_override):
        try:
            # Look up this workload's peak RSS (written by run_perf.py during
            # the previous --perf run) and compute the slurm --mem request.
            # Per-config override wins over the computed value.
            wl_entry = None
            try:
                wl_entry = workloads_db_data[suite][subsuite][workload]
            except (KeyError, TypeError):
                wl_entry = None
            est_mem_mb, measured_rss_mb = estimate_trace_mem_mb(
                wl_entry, trace_type, descriptor_data=descriptor_data,
            )
            if config_mem_override is not None:
                trace_mem_mb = int(config_mem_override)
                mem_source = f"descriptor override trace_mem_mb={trace_mem_mb}MB"
            else:
                trace_mem_mb = est_mem_mb
                if measured_rss_mb is not None:
                    mem_source = (f"estimated from peak_rss={measured_rss_mb}MB "
                                  f"(trace_type={trace_type})")
                else:
                    mem_source = (f"fallback (no peak_rss_mb in workloads_db for "
                                  f"{suite}/{subsuite}/{workload}; run --perf first)")

                # Check if a prior run OOM'd — if so, use the actual peak + headroom
                oom_mem_mb, oom_actual_mb = get_oom_mem_mb(user, workload)
                if oom_mem_mb is not None and oom_mem_mb > trace_mem_mb:
                    info(f"Prior OOM detected for {workload}: actual peak was {oom_actual_mb}MB, "
                         f"bumping request from {trace_mem_mb}MB to {oom_mem_mb}MB", dbg_lvl)
                    trace_mem_mb = oom_mem_mb
                    mem_source = f"auto-scaled from prior OOM (peak={oom_actual_mb}MB × {OOM_HEADROOM_FACTOR})"
            info(f"Trace mem for {workload}: --mem {trace_mem_mb}M ({mem_source})", dbg_lvl)

            sbatch_cmd = generate_sbatch_command(
                trace_dir, slurm_options=slurm_options, mem_mb=trace_mem_mb,
            )

            if trace_type == "cluster_then_trace":
                simpoint_mode = "cluster_then_trace"
            elif trace_type == "trace_then_cluster":
                simpoint_mode = "trace_then_post_process"
            elif trace_type == "iterative_trace":
                simpoint_mode = "iterative_trace"
            else:
                raise Exception(f"Invalid trace type: {trace_type}")
            info(f"Using docker image with name {image_name}:{githash}", dbg_lvl)
            docker_container_name = f"{image_name}_{workload}_{trace_name}_{simpoint_mode}_{user}"
            filename = f"{docker_container_name}_tmp_run.sh"
            write_trace_docker_command_to_file(user, local_uid, local_gid, docker_container_name, githash,
                                               workload, image_name, trace_name, traces_dir, docker_home,
                                               env_vars, binary_cmd, client_bincmd, simpoint_mode, drio_args,
                                               clustering_k, filename, infra_dir, application_dir, slurm=True)
            tmp_files.append(filename)

            result = subprocess.run((sbatch_cmd + filename).split(" "), capture_output=True, text=True)
            _print_sbatch_output(result)
            if result.returncode == 0:
                job_id = result.stdout.strip().split()[-1]
                slurm_ids.append(job_id)
        except Exception as e:
            raise e

    try:
        # Get user for commands
        user = subprocess.check_output("whoami").decode('utf-8')[:-1]
        info(f"User detected as {user}", dbg_lvl)

        # Get a local user/group ids
        local_uid = os.getuid()
        local_gid = os.getgid()

        # Get GitHash
        try:
            githash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("utf-8").strip()
            info(f"Git hash: {githash}", dbg_lvl)
        except FileNotFoundError:
            err("Error: 'git' command not found. Make sure Git is installed and in your PATH.", dbg_lvl)
        except subprocess.CalledProcessError:
            err("Error: Not in a Git repository or unable to retrieve Git hash.", dbg_lvl)


        # Get avlailable nodes. Error if none available
        available_slurm_nodes, all_nodes = check_available_nodes(dbg_lvl)
        info(f"Available nodes: {', '.join(available_slurm_nodes)}", dbg_lvl)

        if available_slurm_nodes == []:
            err("Cannot find any running slurm nodes", dbg_lvl)
            exit(1)

        trace_dir = f"{descriptor_data['root_dir']}/simpoint_flow/{trace_name}"
        prepare_trace(user, scarab_path, scarab_build, docker_home, trace_name, infra_dir, docker_prefix_list, githash, False, available_slurm_nodes, dbg_lvl=dbg_lvl)

        # Load workloads_db.json once so run_single_trace can look up
        # per-workload peak_rss_mb for trace memory estimation.
        workloads_db_data = {}
        try:
            with open(workload_db_path) as f:
                workloads_db_data = json.load(f) or {}
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            warn(f"Could not read {workload_db_path}: {exc}. "
                 f"Trace memory requests will use fallback.", dbg_lvl)

        # Build skip sets: completed traces and currently running/queued slurm jobs
        completed_traces = set()
        if os.path.isdir(trace_dir):
            for wl_dir in os.listdir(trace_dir):
                fp_file = os.path.join(trace_dir, wl_dir, "fingerprint", "segment_size")
                if os.path.isfile(fp_file):
                    completed_traces.add(wl_dir)

        unique_prefixes = list(dict.fromkeys(docker_prefix_list))
        slurm_tasks = check_slurm_task_queued_or_running(unique_prefixes, trace_name, user, dbg_lvl)
        running_workloads = set()
        for tasks in slurm_tasks.values():
            for t in tasks:
                # Container name format: {image}_{workload}_{trace_name}_...
                # Extract workload name between first and second underscore-group
                parts = t.split(f"_{trace_name}_")
                if parts:
                    wl = parts[0].split(f"{unique_prefixes[0]}_", 1)[-1]
                    running_workloads.add(wl)

        skipped = 0
        submitted = 0
        # Iterate over each trace configuration
        for config in trace_configs:
            workload = config["workload"]
            suite = config.get("suite")
            subsuite = config.get("subsuite")
            image_name = config["image_name"]

            # Skip completed or already-running traces
            if workload in completed_traces:
                info(f"Skipping {workload}: already completed", dbg_lvl)
                skipped += 1
                continue
            if workload in running_workloads:
                info(f"Skipping {workload}: slurm job already running", dbg_lvl)
                skipped += 1
                continue

            # Clean stale partial state from prior failed runs
            wl_trace_dir = os.path.join(trace_dir, workload)
            if os.path.isdir(wl_trace_dir):
                info(f"Cleaning stale state for {workload}", dbg_lvl)
                shutil.rmtree(wl_trace_dir)

            if config["env_vars"] != None:
                env_vars = config["env_vars"].split()
            else:
                env_vars = config["env_vars"]
            binary_cmd = config["binary_cmd"]
            client_bincmd = config["client_bincmd"]
            trace_type = config["trace_type"]
            drio_args = config["dynamorio_args"]
            clustering_k = config["clustering_k"]
            slurm_options = config.get("slurm_options", "")
            config_mem_override = config.get("trace_mem_mb")

            run_single_trace(workload, suite, subsuite, image_name, trace_name, env_vars, binary_cmd,
                             client_bincmd, trace_type, drio_args, clustering_k, application_dir,
                             slurm_options, config_mem_override)
            submitted += 1

        info(f"Submitted: {submitted}, Skipped: {skipped} (completed: {len(completed_traces)}, running: {len(running_workloads)})", dbg_lvl)

        # Clean up temp files
        for tmp in tmp_files:
            if os.path.exists(tmp):
                info(f"Removing temporary run script {tmp}", dbg_lvl)
                os.remove(tmp)

        # If all traces are completed (nothing submitted, nothing running),
        # automatically run finish_trace to copy traces and update workloads_db.
        if submitted == 0 and len(running_workloads) == 0 and len(completed_traces) > 0:
            info("All traces completed. Running finish_trace to post-process...", dbg_lvl)
            finish_trace(user, descriptor_data, workload_db_path, infra_dir, dbg_lvl)
        elif submitted > 0 and slurm_ids and descriptor_path:
            # Submit a dependent finalization job that runs after all trace jobs complete
            finalize_cmd = f"cd {infra_dir} && python -m scripts.run_trace -d {descriptor_path} -f -si {infra_dir}"
            dep_str = ':'.join(slurm_ids)
            log_path = os.path.join(trace_dir, "logs", "finalize_job_%j.out")
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            sbatch_finalize = (
                f"sbatch --dependency=afterany:{dep_str} "
                f"-o {log_path} "
                f"--wrap={shlex.quote(finalize_cmd)}"
            )
            result = subprocess.run(sbatch_finalize, shell=True, capture_output=True, text=True)
            _print_sbatch_output(result)
            if result.returncode == 0:
                info(f"Finalization job submitted (dependency on {len(slurm_ids)} trace jobs). "
                     f"Will run automatically when all jobs complete.", dbg_lvl)
            else:
                warn(f"Failed to submit finalization job. Run './sci --trace {trace_name.replace('trace_', '')}' "
                     f"manually after all jobs complete.", dbg_lvl)
        elif submitted > 0:
            info(f"Slurm trace jobs submitted. Run './sci --trace {trace_name.replace('trace_', '')}' again after all jobs complete to finalize.", dbg_lvl)
    except Exception as e:
        print("An exception occurred:", e)
        traceback.print_exc()  # Print the full stack trace

        # Clean up temp files (idempotent)
        for tmp in tmp_files:
            if os.path.exists(tmp):
                info(f"Removing temporary run script {tmp}", dbg_lvl)
                os.remove(tmp)

        kill_jobs(user, trace_name, docker_prefix_list, dbg_lvl)
