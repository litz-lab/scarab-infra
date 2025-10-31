#!/usr/bin/python3

# 10/7/2024 | Alexander Symons | run_slurm.py
# 01/27/2025 | Surim Oh | slurm_runner.py

import os
import subprocess
import re
import traceback
import json
from pathlib import Path
from .utilities import (
        err,
        warn,
        info,
        get_simpoints,
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
        get_image_name,
        generate_table,
        run_on_node
        )

# Check if the docker image exists on available slurm nodes
# Inputs: list of available slurm nodes
# Output: list of nodes where the docker image is ready
def check_docker_image(nodes, docker_prefix, githash, dbg_lvl = 1):
    try:
        available_nodes = []
        for node in nodes:
            # Check if the image exists
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
            if line[-1] != 'idle' and line[-1] != 'mix':
                info(f"{node} is not available. It is '{line[-1]}'", dbg_lvl)
                continue

            # If docker is not installed, skip
            try:
                docker_installed = subprocess.check_output(["srun", f"--nodelist={node}", "docker", "--version"])
            except Exception as e:
                info(f"docker is not installed on {node}", dbg_lvl)
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
def generate_sbatch_command(excludes, experiment_dir):
    # If all nodes are usable, no need to exclude
    if not excludes == set():
        return f"sbatch --exclude {','.join(excludes)} -c 1 -o {experiment_dir}/logs/job_%j.out "

    return f"sbatch -c 1 -o {experiment_dir}/logs/job_%j.out "

def get_simulation_jobs(descriptor_data, workloads_data, docker_prefix, user, dbg_lvl = 1):
    architecture = descriptor_data["architecture"]
    experiment_name = descriptor_data["experiment"]
    docker_home = descriptor_data["root_dir"]
    scarab_path = descriptor_data["scarab_path"]
    scarab_build = descriptor_data["scarab_build"]
    traces_dir = descriptor_data["traces_dir"]
    configs = descriptor_data["configurations"]
    simulations = descriptor_data["simulations"]

    # Returns list of simpoints for a given workload
    # Inputs: suite, subsuite, workload, exp_cluster_id
    # Outputs: list of simpoints
    def get_simpoints_wrapper(suite, subsuite, workload, exp_cluster_id, sim_mode):
        if "simpoints" not in workloads_data[suite][subsuite][workload].keys():
            return [0]
        elif exp_cluster_id == None:
            return list(map(int, get_simpoints(workloads_data[suite][subsuite][workload], sim_mode, dbg_lvl).keys()))
        elif exp_cluster_id > 0:
            assert isinstance(exp_cluster_id, int), f"exp_cluster_id must be of type int, but got {type(exp_cluster_id)}"
            return [exp_cluster_id]
        
    all_jobs = []

    def docker_container_name(workload, config, cluster, sim_mode, img_name): 
        return f"{img_name}_{workload}_{experiment_name}_{config.replace("/", "-")}_{cluster}_{sim_mode}_{user}"

    for simulation in simulations:
        suite = simulation["suite"]
        subsuite = simulation["subsuite"]
        workload = simulation["workload"]
        exp_cluster_id = simulation["cluster_id"]
        sim_mode = simulation["simulation_type"]
        sim_warmup = simulation["warmup"]

        image_name = get_image_name(workloads_data, simulation)

        if image_name not in docker_prefix:
            print(f"suite {image_name} not in docker_prefix")
            exit()
        
        # Run all the workloads within suite
        if workload == None and subsuite == None:
            for subsuite_ in workloads_data[suite].keys():
                for workload_ in workloads_data[suite][subsuite_].keys():
                    # For each workload...

                    sim_mode_ = sim_mode
                    if sim_mode_ == None:
                        sim_mode_ = workloads_data[suite][subsuite_][workload_]["simulation"]["prioritized_mode"]

                    # Number of running jobs is the number of simpoints * number of configs
                    simpoint_ids = get_simpoints_wrapper(suite, subsuite_, workload_, exp_cluster_id, sim_mode_)*len(configs)

                    # Use docker_container_name to get the job names of all jobs
                    all_jobs += [docker_container_name(workload_, config, cluster_id, sim_mode_, image_name) for config in configs.keys() for cluster_id in simpoint_ids]
                    

        elif workload == None and subsuite != None:
            for workload_ in workloads_data[suite][subsuite].keys():
                sim_mode_ = sim_mode
                if sim_mode_ == None:
                    sim_mode_ = workloads_data[suite][subsuite][workload_]["simulation"]["prioritized_mode"]

                # Number of running jobs is the number of simpoints * number of configs
                simpoint_ids = get_simpoints_wrapper(suite, subsuite, workload_, exp_cluster_id, sim_mode_)*len(configs)

                # Use docker_container_name to get the job names of all jobs
                all_jobs += [docker_container_name(workload_, config, cluster_id, sim_mode_, image_name) for config in configs.keys() for cluster_id in simpoint_ids]

        else:
            sim_mode_ = sim_mode
            if sim_mode_ == None:
                sim_mode_ = workloads_data[suite][subsuite][workload]["simulation"]["prioritized_mode"]
                
            # Number of running jobs is the number of simpoints * number of configs
            simpoint_ids = get_simpoints_wrapper(suite, subsuite, workload, exp_cluster_id, sim_mode_)*len(configs)

            # Use docker_container_name to get the job names of all jobs
            all_jobs += [docker_container_name(workload, config, cluster_id, sim_mode_, image_name) for config in configs.keys() for cluster_id in simpoint_ids]
            
    return set(all_jobs)

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

    # Again general info. Not helpful for --status <job>? 
    # for docker_prefix in docker_prefix_list:
    #     print(f"\nChecking what nodes have {docker_prefix}:{githash} image:")
    #     docker_available_slurm_nodes = check_docker_image(all_nodes, docker_prefix, githash, dbg_lvl)
    #     for node in all_nodes:
    #         if node in docker_available_slurm_nodes:
    #             print(f"\033[92mAVAILABLE:   {node}\033[0m")
    #         else:
    #             print(f"\033[31mUNAVAILABLE: {node}\033[0m")

    # Get dictionary of {node: [processes]}
    # NOTE: This is a list of run commands, not the actual containers. Container name will be same miunus tmp_run.sh
    slurm_running_sims = check_slurm_task_queued_or_running(docker_prefix_list, job_name, user, dbg_lvl)

    print(f"\nChecking what nodes currently have a running job with the following name(s):")
    for docker_prefix in docker_prefix_list:
        print(f"{docker_prefix}_*_{job_name}_*_*_{user}")

    print()

    # Print out every running job
    # for key, val in running_sims.items():
    #     if key == '':
    #         print("Fount not running")
    #         continue

    #     if len(val) > 0:
    #         print(f"\033[92mRUNNING:     {key}\033[0m")
    #         for docker in val:
    #             print(f"\033[92m    COMMAND: {docker}\033[0m")
    #     else:
    #         print(f"\033[31mNOT RUNNING: {key}\033[0m")

    # if '' in running_sims.keys():
    #     for val in running_sims['']:
    #         if len(val) > 0:
    #             print(f"\033[92mQUEUED:     {val}\033[0m")
    # else:
    #     print(f"\033[31mNO COMMANDS IN QUEUE\033[0m")

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

    not_complete = running_sims + queued_sims

    all_jobs = get_simulation_jobs(descriptor_data, workloads_data, docker_prefix_list, user, dbg_lvl)
    # print(f"Completed Jobs: {len(all_jobs) - len(not_complete)}")

    # completed_jobs = list(set(all_jobs) - set(not_complete))
 
    print()

    root_directory = f"{descriptor_data["root_dir"]}/simulations/{descriptor_data["experiment"]}/logs/"

    # Check that experiment exists
    if not os.path.exists(root_directory):
        print("Log file directory does not exist")
        print("The current experiment does not seem to have been run yet")
        return
        
    log_files = os.listdir(root_directory)

    # Running sims have log files

    # Ignore stat collector. If log file found, ignore it
    # We actually dont need to care about counts. Just the status reported in the logs
    # COmpletely independently, read all logs and figure out error rates.

    # TODO: Check if log files are actually from this experiment - Shouldn't be necessary anymore
    if len(log_files) > len(all_jobs) + 1:
        print("More log files than total runs. Maybe same experiment name was run multiple times?")
        print("Any errors from a previous run with the same experiment name will be re-reported now")

    error_runs = set()
    skipped = 0
    stats_generating = False

    confs = list(descriptor_data["configurations"].keys())

    completed = {conf:0 for conf in confs}
    failed = {conf:0 for conf in confs}
    slurm_failed = {conf:0 for conf in confs}
    running = {conf:0 for conf in confs}
    pending = {conf:0 for conf in confs}    

    # NOTE: Potential Subset issue again. conf2 and conf sims will be added to conf2
    # Tried to create such a scenario but was unable
    for sim in queued_sims:
        print(sim)
        for conf in confs:
            if conf in sim:
                pending[conf] += 1
                break

    # Check each log file for errors
    for file in log_files:
        with open(root_directory+file, 'r') as f:
            contents = f.read()
            split = contents.split(" ")

            # Cannot get config if file isn't complete
            if len(split) < 2:
                continue

            config = split[1] 

            # Check if currently running, skip if so. Running simulations will not contain
            # the completion message
            is_stat_job = "stat_collection_job" in file

            if not is_stat_job:
                # Non-stat jobs will have 4 lines in their log file until they complete.
                # Fifth line completion message indicates completion
                if len(contents.split("\n")) < 5:
                    skipped += 1
                    if config in running:
                        running[config] += 1
                    continue
            else:
                # Stat jobs were modified to print DONE as a final message
                if "DONE" not in contents:
                    # skipped += 1
                    stats_generating = True
                    continue

            # Slurm error messages have 'node: error:' in them
            for node in all_nodes:
                if f"{node}: error:" in contents:
                    error_runs.add(root_directory+file)
                    if config in slurm_failed:
                        slurm_failed[config] += 1

            # Some failures will have a line with "Segmentation falut" in them if they fail
            if 'Segmentation fault' in contents:
                error_runs.add(root_directory+file)
                if config in failed:
                    failed[config] += 1
                continue

            # Most scarab runs and all stat runs will have a line with "Error" in them if they fail
            if 'Error' in contents:
                error_runs.add(root_directory+file)
                if config in failed:
                    failed[config] += 1
                continue

            # To be sure, check scarab runs with for final success line
            if config != 'stat' and descriptor_data["experiment"] not in contents:
                error_runs.add(root_directory+file)
                if config in failed:
                    failed[config] += 1
                continue

            inst_stat_missing = False
            if config != "stat":
                lines = contents.splitlines()
                if lines:
                    first_line = lines[0].strip()
                    tokens = first_line.split()
                    if len(tokens) >= 4 and tokens[0] == "Running":
                        workload_token = tokens[2]
                        cluster_token = tokens[3]
                        if cluster_token.lower() != "none":
                            workload_parts = workload_token.split("/")
                            if workload_parts:
                                sim_dir = Path(descriptor_data["root_dir"]) / "simulations" / descriptor_data["experiment"] / config
                                sim_dir = sim_dir.joinpath(*workload_parts, cluster_token)
                                inst_stat_path = sim_dir / "inst.stat.0.csv"
                                if not inst_stat_path.is_file():
                                    inst_stat_missing = True
            if inst_stat_missing:
                error_runs.add(root_directory+file)
                if config in failed:
                    failed[config] += 1
                continue

            if config != 'stat' and config in completed:
                completed[config] += 1
    

    print(f"Currently running {len(running_sims)} simulations (from logs: {skipped})")
    if stats_generating:
        print("Stat collector is running")

    # print(f"\033[92mSuccessfully Completed Jobs: {len(all_jobs) - len(not_complete) - len(error_runs)}\033[0m")
    
    data = {"Configuration":[],"Completed":[],"Failed":[],"Failed - Slurm":[],"Running":[],"Pending":[],"Non-existant":[],"Total":[]}
    for conf in confs:  
        data["Configuration"].append(conf)
        data["Completed"].append(completed[conf])
        data["Failed"].append(failed[conf])
        data["Failed - Slurm"].append(slurm_failed[conf])
        data["Running"].append(running[conf])
        data["Pending"].append(pending[conf])

        # Calculated, number of simpoints that should exist in every config
        total_per_conf = int(len(all_jobs)/len(confs))

        # Number of simpoints accounted for
        total_found = completed[conf] + failed[conf] + running[conf] + pending[conf] + slurm_failed[conf]

        data["Total"].append(total_per_conf)
        data["Non-existant"].append(total_per_conf - total_found) # Unaccounted for simpoints


    print(generate_table(data))

    if skipped != len(running_sims):
        print("\033[33mWARN: Number of log files skipped due to being 'in progress' does not match number of running simulations.")
        print("This could indicate the file format has changed in a way where the 'is running' checks need to be modified.")
        if skipped > len(running_sims):
            print("Completed jobs' log files were skipped. This could also be caused by running same experiment multiple times (check for prev. err).")
        else:
            print("Running jobs' log files were evaluated for success/failure")

        print("\033[0m")
    
    # Print up to five of the full paths
    if error_runs:
        error_list = sorted(error_runs)
        print(f"\033[31mErroneous Jobs: {len(error_list)}\033[0m")
        print(f"\033[31mErrors found in {len(error_list)}/{len(log_files)} log files.")
        print("First 5 error runs:\n", "\n".join(error_list[:5]), "\033[0m", sep='')
    else:
        print(f"\033[92mNo errors found in log files\033[0m")

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
        else:
            print("Operation canceled.")
    else:
        print("No job found.")

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
            )
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
                run_on_node(["docker", "rm", "-f", container], node=node, check=True)
                print(f"Removed container {container} on {node}")
                removed_any = True
            except subprocess.CalledProcessError as exc:
                err(f"Failed to remove container {container} on {node}: {exc}", dbg_lvl)
            except Exception as exc:
                err(f"Unexpected error removing container {container} on {node}: {exc}", dbg_lvl)

    if not removed_any:
        print("No matching containers found on any slurm node.")

    clean_tmp_run_scripts(all_nodes, job_name, user, dbg_lvl)

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
            )
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
                run_on_node(["rm", "-f", script], node=node, check=True)
                print(f"Removed {script} on {node}")
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

    docker_prefix_list = get_image_list(simulations, workloads_data)

    def run_single_workload(suite, subsuite, workload, exp_cluster_id, sim_mode, warmup, all_nodes):
        try:
            docker_prefix = get_docker_prefix(sim_mode, workloads_data[suite][subsuite][workload]["simulation"])
            info(f"Using docker image with name {docker_prefix}:{githash}", dbg_lvl)
            docker_running = check_docker_image(available_slurm_nodes, docker_prefix, githash, dbg_lvl)
            excludes = set(all_nodes) - set(docker_running)
            info(f"Excluding following nodes: {', '.join(excludes)}", dbg_lvl)
            sbatch_cmd = generate_sbatch_command(excludes, experiment_dir)
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

                    # Create temp file with run command and run it
                    filename = f"{docker_container_name}_tmp_run.sh"

                    slurm_running_sims = check_slurm_task_queued_or_running(docker_prefix_list, experiment_name, user, dbg_lvl)
                    running_sims = []
                    for node_list in slurm_running_sims.values():
                        running_sims += node_list

                    if check_can_skip(descriptor_data, config_key, suite, subsuite, workload, cluster_id, filename, running_sims, sim_mode, user, dbg_lvl):
                        info(f"Skipping {workload} with config {config_key} and cluster id {cluster_id}", dbg_lvl)
                        continue

                    workload_home = f"{suite}/{subsuite}/{workload}"
                    write_docker_command_to_file(user, local_uid, local_gid, workload, workload_home, experiment_name,
                                                 docker_prefix, docker_container_name, traces_dir,
                                                 docker_home, githash, config_key, config, sim_mode, binary_name,
                                                 seg_size, architecture, cluster_id, warmup, trace_warmup, trace_type,
                                                 trace_file, env_vars, bincmd, client_bincmd, filename, infra_dir)
                    tmp_files.append(filename)

                    info(f"Running sbatch command '{sbatch_cmd + filename}'", dbg_lvl)
                    result = subprocess.run((sbatch_cmd + filename).split(" "), capture_output=True, text=True)
                    print(result.stdout.split(" ")[-1].strip())
                    slurm_ids.append(result.stdout.split(" ")[-1].strip())

            return slurm_ids
        except Exception as e:
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


        # Get avlailable nodes. Error if none available
        available_slurm_nodes, all_nodes = check_available_nodes(dbg_lvl)
        info(f"Available nodes: {', '.join(available_slurm_nodes)}", dbg_lvl)

        if available_slurm_nodes == []:
            err("Cannot find any running slurm nodes", dbg_lvl)
            exit(1)

        scarab_binaries = []
        for sim in simulations:
            for config_key in configs:
                scarab_binary = configs[config_key]["binary"]
                if scarab_binary not in scarab_binaries:
                    scarab_binaries.append(scarab_binary)

        # Generate commands for executing in users docker and sbatching to nodes with containers
        experiment_dir = f"{descriptor_data['root_dir']}/simulations/{experiment_name}"
        scarab_githash, image_tag_list = prepare_simulation(user, scarab_path, scarab_build, descriptor_data['root_dir'], experiment_name, architecture, docker_prefix_list, githash, infra_dir, scarab_binaries, False, available_slurm_nodes, dbg_lvl)

        # Iterate over each workload and config combo
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
                        sim_mode_ = sim_mode
                        if sim_mode_ == None:
                            sim_mode_ = workloads_data[suite][subsuite_][workload_]["simulation"]["prioritized_mode"]
                        if sim_warmup == None:  # Use the whole warmup available in the trace if not specified
                            sim_warmup = workloads_data[suite][subsuite_][workload_]["simulation"][sim_mode_]["warmup"]
                        slurm_ids += run_single_workload(suite, subsuite_, workload_, exp_cluster_id, sim_mode_, sim_warmup, all_nodes)
            elif workload == None and subsuite != None:
                for workload_ in workloads_data[suite][subsuite].keys():
                    sim_mode_ = sim_mode
                    if sim_mode_ == None:
                        sim_mode_ = workloads_data[suite][subsuite][workload_]["simulation"]["prioritized_mode"]
                    if sim_warmup == None:  # Use the whole warmup available in the trace if not specified
                        sim_warmup = workloads_data[suite][subsuite][workload_]["simulation"][sim_mode_]["warmup"]
                    slurm_ids += run_single_workload(suite, subsuite, workload_, exp_cluster_id, sim_mode_, sim_warmup, all_nodes)
            else:
                sim_mode_ = sim_mode
                if sim_mode_ == None:
                    sim_mode_ = workloads_data[suite][subsuite][workload]["simulation"]["prioritized_mode"]
                if sim_warmup == None:  # Use the whole warmup available in the trace if not specified
                    sim_warmup = workloads_data[suite][subsuite][workload]["simulation"][sim_mode_]["warmup"]
                slurm_ids += run_single_workload(suite, subsuite, workload, exp_cluster_id, sim_mode_, sim_warmup, all_nodes)

        # Clean up temp files
        for tmp in tmp_files:
            info(f"Removing temporary run script {tmp}", dbg_lvl)
            os.remove(tmp)

        finish_simulation(user, docker_home, descriptor_path, descriptor_data['root_dir'], experiment_name, image_tag_list, available_slurm_nodes, slurm_ids)

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

def run_tracing(user, descriptor_data, workload_db_path, infra_dir, dbg_lvl = 2):
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

    def run_single_trace(workload, image_name, trace_name, env_vars, binary_cmd, client_bincmd, trace_type, drio_args, clustering_k, application_dir):
        try:
            docker_running = check_docker_image(available_slurm_nodes, image_name, githash, dbg_lvl)
            excludes = set(all_nodes) - set(docker_running)
            info(f"Excluding following nodes: {', '.join(excludes)}", dbg_lvl)
            sbatch_cmd = generate_sbatch_command(excludes, trace_dir)

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
                                               clustering_k, filename, infra_dir, application_dir)
            tmp_files.append(filename)

            os.system(sbatch_cmd + filename)
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

        # Iterate over each trace configuration
        for config in trace_configs:
            workload = config["workload"]
            image_name = config["image_name"]
            if config["env_vars"] != None:
                env_vars = config["env_vars"].split()
            else:
                env_vars = config["env_vars"]
            binary_cmd = config["binary_cmd"]
            client_bincmd = config["client_bincmd"]
            trace_type = config["trace_type"]
            drio_args = config["dynamorio_args"]
            clustering_k = config["clustering_k"]

            run_single_trace(workload, image_name, trace_name, env_vars, binary_cmd, client_bincmd,
                             trace_type, drio_args, clustering_k, application_dir)

        # Clean up temp files
        for tmp in tmp_files:
            info(f"Removing temporary run script {tmp}", dbg_lvl)
            os.remove(tmp)

        finish_trace(user, descriptor_data, workload_db_path, infra_dir, dbg_lvl)
    except Exception as e:
        print("An exception occurred:", e)
        traceback.print_exc()  # Print the full stack trace

        # Clean up temp files
        for tmp in tmp_files:
            info(f"Removing temporary run script {tmp}", dbg_lvl)
            os.remove(tmp)

        kill_jobs(user, trace_name, docker_prefix_list, dbg_lvl)

        print("Recover the ASLR setting with sudo. Provide password..")
        os.system("echo 2 | sudo tee /proc/sys/kernel/randomize_va_space")
