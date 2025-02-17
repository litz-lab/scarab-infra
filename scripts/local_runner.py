#!/usr/bin/python3

# 01/27/2025 | Surim Oh | local_runner.py

import os
import subprocess
import psutil
import signal
import re
import traceback
from utilities import (
        err,
        warn,
        info,
        get_simpoints,
        write_docker_command_to_file,
        remove_docker_containers,
        prepare_simulation,
        finish_simulation
        )

# Check if a container is running on local
# Inputs: docker_prefix, experiment_name, user,
# Output: list of containers
def check_docker_container_running(docker_prefix, experiment_name, user, dbg_lvl):
    pattern = re.compile(fr"^{docker_prefix}_.*_{experiment_name}.*_.*_{user}$")
    try:
        dockers = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True, check=True)
        lines = dockers.stdout.strip().split("\n") if dockers.stdout else []
        matching_containers = [line for line in lines if pattern.match(line)]
        return matching_containers
    except Exception as e:
        err(f"Error while checking a running docker container named {docker_prefix}_.*_{experiment_name}_.*_.*_{user} on node {node}", dbg_lvl)
        raise e

# Print info/status of running experiment and available cores
def print_status(user, experiment_name, docker_prefix, dbg_lvl = 2):
    docker_running = check_docker_container_running(docker_prefix, experiment_name, user, dbg_lvl)
    if len(docker_running) > 0:
        print(f"\033[92mRUNNING:     local\033[0m")
        for docker in docker_running:
            print(f"\033[92m    CONTAINER: {docker}\033[0m")
    else:
        print(f"\033[31mNOT RUNNING: local\033[0m")

def kill_jobs(user, experiment_name, docker_prefix, infra_dir, dbg_lvl):
    # Define the process name pattern
    pattern = re.compile(f"python3 {infra_dir}/scripts/run_simulation.py -dbg 3 -d {infra_dir}/json/{experiment_name}.json")

    # Iterate over all processes
    found_process = []
    for proc in psutil.process_iter(attrs=['pid', 'name', 'username', 'cmdline']):
        try:
            # Check if the process name matches the pattern and is run by the specified user
            cmdline = " ".join(proc.info['cmdline'])
            if pattern.match(cmdline) and proc.info['username'] == user:
                found_process.append(proc)
                print(f"Found process {proc.info['name']} with PID {proc.info['pid']} running by {user}")
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            # Handle processes that may have been terminated or we don't have permission to access
            continue

    print(str(len(found_process)) + "found.")
    confirm = input("Do you want to kill these processes? (y/n): ").lower()
    if confirm == 'y':
        # Iterate over found processes
        for proc in found_process:
            try:
                # Kill the process and all its child processes
                for child in proc.children(recursive=True):
                    try:
                        child.kill()
                        print(f"Killed child process with PID {child.pid}")
                    except (psutil.NoSuchProcess):
                        print(f"No such process {child.pid}")
                        continue
                    except (psutil.AccessDenied):
                        print(f"Access denied to {child.pid}")
                        continue
                proc.kill()
                print(f"Terminated process {proc.info['name']} and its children.")
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                # Handle processes that may have been terminated or we don't have permission to access
                continue
    else:
        print("Operation canceled.")

    info(f"Clean up docker containers..", dbg_lvl)
    remove_docker_containers(docker_prefix, experiment_name, user, dbg_lvl)

    info(f"Removing temporary run scripts..", dbg_lvl)
    os.system(f"rm {experiment_name}_*_tmp_run.sh")

def run_simulation(user, descriptor_data, dbg_lvl = 1):
    architecture = descriptor_data["architecture"]
    docker_prefix = descriptor_data["workload_group"]
    workloads = descriptor_data["workloads_list"]
    experiment_name = descriptor_data["experiment"]
    scarab_mode = descriptor_data["simulation_mode"]
    docker_home = descriptor_data["root_dir"]
    scarab_path = descriptor_data["scarab_path"]
    simpoint_traces_dir = descriptor_data["simpoint_traces_dir"]
    configs = descriptor_data["configurations"]

    available_cores = os.cpu_count()
    max_processes = int(available_cores * 0.9)
    processes = set()

    try:
        # Get a local user/group ids
        local_uid = os.getuid()
        local_gid = os.getgid()

        # Get GitHash
        try:
            githash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("utf-8").strip()
            info(f"Git hash: {githash}", dbg_lvl)
        except FileNotFoundError:
            err("Error: 'git' command not found. Make sure Git is installed and in your PATH.")
        except subprocess.CalledProcessError:
            err("Error: Not in a Git repository or unable to retrieve Git hash.")

        image = subprocess.check_output(["docker", "images", "-q", f"{docker_prefix}:{githash}"])
        if image == []:
            info(f"Couldn't find image {docker_prefix}:{githash}", dbg_lvl)
            subprocess.check_output(["./run.sh", "-b", docker_prefix])
            image = subprocess.check_output(["docker", "images", "-q", f"{docker_prefix}:{githash}"])
            if image == []:
                info(f"Still couldn't find image {docker_prefix}:{githash} after trying to build one", dbg_lvl)
                exit(1)

        scarab_githash = prepare_simulation(user, scarab_path, descriptor_data['root_dir'], experiment_name, architecture, dbg_lvl)

        # Iterate over each workload and config combo
        tmp_files = []
        for workload in workloads:
            simpoints = get_simpoints(simpoint_traces_dir, workload, dbg_lvl)
            for config_key in configs:
                config = configs[config_key]

                for simpoint, weight in simpoints.items():
                    print(simpoint, weight)

                    docker_container_name = f"{docker_prefix}_{workload}_{experiment_name}_{config_key}_{simpoint}_{user}"
                    # Create temp file with run command and run it
                    filename = f"{experiment_name}_{workload}_{config_key.replace("/", "-")}_{simpoint}_tmp_run.sh"
                    write_docker_command_to_file(user, local_uid, local_gid, workload, experiment_name,
                                                 docker_prefix, docker_container_name, simpoint_traces_dir,
                                                 docker_home, githash, config_key, config, scarab_mode, scarab_githash,
                                                 architecture, simpoint, filename)
                    tmp_files.append(filename)
                    command = '/bin/bash ' + filename
                    process = subprocess.Popen("exec " + command, stdout=subprocess.PIPE, shell=True)
                    processes.add(process)
                    info(f"Running command '{command}'", dbg_lvl)
                    while len(processes) >= max_processes:
                        # Loop through the processes and wait for one to finish
                        for p in processes.copy():
                            if p.poll() is not None: # This process has finished
                                p.wait() # Make sure it's really finished
                                processes.remove(p) # Remove from set of active processes
                                break # Exit the loop after removing one process

        print("Wait processes...")
        for p in processes:
            p.wait()

        # Clean up temp files
        for tmp in tmp_files:
            info(f"Removing temporary run script {tmp}", dbg_lvl)
            os.remove(tmp)

        finish_simulation(user, f"{docker_home}/simulations/{experiment_name}")

    except Exception as e:
        print("An exception occurred:", e)
        traceback.print_exec()  # Print the full stack trace
        for p in processes:
            p.kill()
