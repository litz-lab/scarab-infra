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
        finish_simulation,
        get_image_list,
        get_docker_prefix,
        prepare_trace,
        finish_trace,
        write_trace_docker_command_to_file
        )

# Check if a container is running on local
# Inputs: docker_prefix, job_name, user,
# Output: list of containers
def check_docker_container_running(docker_prefix_list, job_name, user, dbg_lvl):
    try:
        matching_containers = []
        for docker_prefix in docker_prefix_list:
            pattern = re.compile(fr"^{docker_prefix}_.*_{job_name}.*_.*_{user}$")
            dockers = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True, check=True)
            lines = dockers.stdout.strip().split("\n") if dockers.stdout else []
            for line in lines:
                if pattern.match(line):
                    matching_containers.append(line)
        return matching_containers
    except Exception as e:
        err(f"Error while checking a running docker container named {docker_prefix}_.*_{job_name}_.*_.*_{user} on node {node}", dbg_lvl)
        raise e

# Print info/status of running experiment and available cores
def print_status(user, job_name, docker_prefix_list, dbg_lvl = 2):
    docker_running = check_docker_container_running(docker_prefix_list, job_name, user, dbg_lvl)
    if len(docker_running) > 0:
        print(f"\033[92mRUNNING:     local\033[0m")
        for docker in docker_running:
            print(f"\033[92m    CONTAINER: {docker}\033[0m")
    else:
        print(f"\033[31mNOT RUNNING: local\033[0m")

def kill_jobs(user, job_type, job_name, docker_prefix_list, infra_dir, dbg_lvl):
    # Define the process name pattern
    if job_type == "simulation":
        pattern = re.compile(f"python3 {infra_dir}/scripts/run_simulation.py -dbg 3 -d {infra_dir}/json/{job_name}.json")
    elif job_type == "trace":
        pattern = re.compile(f"python3 {infra_dir}/scripts/run_trace.py -dbg 3 -d {infra_dir}/json/{job_name}.json")

    # Iterate over all processes
    found_process = []
    for proc in psutil.process_iter(attrs=['pid', 'name', 'username', 'cmdline']):
        try:
            # Ensure 'cmdline' exists and is iterable
            cmdline = proc.info.get('cmdline', [])
            # Check if the process name matches the pattern and is run by the specified user
            if cmdline and isinstance(cmdline, list):
                cmdline_str = " ".join(cmdline)
                if pattern.match(cmdline_str) and proc.info.get('username') == user:
                    found_process.append(proc)
                    print(f"Found process {proc.info.get('name')} with PID {proc.info.get('pid')} running by {user}")
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
    remove_docker_containers(docker_prefix_list, job_name, user, dbg_lvl)

    info(f"Removing temporary run scripts..", dbg_lvl)
    os.system(f"rm *_{job_name}_*_{user}_tmp_run.sh")

def run_simulation(user, descriptor_data, workloads_data, suite_data, infra_dir, dbg_lvl = 1):
    architecture = descriptor_data["architecture"]
    experiment_name = descriptor_data["experiment"]
    docker_home = descriptor_data["root_dir"]
    scarab_path = descriptor_data["scarab_path"]
    simpoint_traces_dir = descriptor_data["simpoint_traces_dir"]
    configs = descriptor_data["configurations"]
    simulations = descriptor_data["simulations"]

    docker_prefix_list = get_image_list(simulations, workloads_data, suite_data)

    available_cores = os.cpu_count()
    max_processes = int(available_cores * 0.9)
    processes = set()
    tmp_files = []

    def run_single_workload(workload, exp_cluster_id, sim_mode):
        try:
            docker_prefix = get_docker_prefix(sim_mode, workloads_data[workload]["simulation"])
            info(f"Using docker image with name {docker_prefix}:{githash}", dbg_lvl)
            trim_type = None
            modules_dir = ""
            trace_file = ""
            env_vars = ""
            bincmd = ""
            client_bincmd = ""
            simulation_data = workloads_data[workload]["simulation"][sim_mode]
            if sim_mode == "memtrace":
                trim_type = simulation_data["trim_type"]
                modules_dir = simulation_data["modules_dir"]
                trace_file = simulation_data["trace_file"]
            if sim_mode == "exec":
                env_vars = simulation_data["env_vars"]
                bincmd = simulation_data["binary_cmd"]
                client_bincmd = simulation_data["client_bincmd"]

            if "simpoints" not in workloads_data[workload].keys():
                weight = 1
                simpoints = {}
                simpoints["0"] = weight
            elif exp_cluster_id == None:
                simpoints = get_simpoints(workloads_data[workload], dbg_lvl)
            elif exp_cluster_id > 0:
                weight = get_weight_by_cluster_id(exp_cluster_id, workloads_data[workload]["simpoints"])
                simpoints = {}
                simpoints[f"{exp_cluster_id}"] = weight

            for config_key in configs:
                config = configs[config_key]

                for cluster_id, weight in simpoints.items():
                    print(cluster_id, weight)

                    docker_container_name = f"{docker_prefix}_{workload}_{experiment_name}_{config_key.replace("/", "-")}_{cluster_id}_{sim_mode}_{user}"
                    # Create temp file with run command and run it
                    filename = f"{docker_container_name}_tmp_run.sh"
                    write_docker_command_to_file(user, local_uid, local_gid, workload, experiment_name,
                                                 docker_prefix, docker_container_name, simpoint_traces_dir,
                                                 docker_home, githash, config_key, config, sim_mode, scarab_githash,
                                                 architecture, cluster_id, trim_type, modules_dir, trace_file,
                                                 env_vars, bincmd, client_bincmd, filename, infra_dir)
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
        except Exception as e:
            raise e

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

        for docker_prefix in docker_prefix_list:
            image = subprocess.check_output(["docker", "images", "-q", f"{docker_prefix}:{githash}"])
            if image == []:
                info(f"Couldn't find image {docker_prefix}:{githash}", dbg_lvl)
                subprocess.check_output(["docker", "pull", f"ghcr.io/litz-lab/scarab-infra/{docker_prefix}:{githash}"])
                image = subprocess.check_output(["docker", "images", "-q", f"ghcr.io/litz-lab/scarab-infra/{docker_prefix}:{githash}"])
                if image != []:
                    subprocess.check_output(["docker", "tag", f"ghcr.io/litz-lab/scarab-infra/{docker_prefix}:{githash}", f"{docker_prefix}:{githash}"])
                    subprocess.check_output(["docker", "rmi", f"ghcr.io/litz-lab/scarab-infra/{docker_prefix}:{githash}"])
                    continue
                subprocess.check_output(["./run.sh", "-b", docker_prefix])
                image = subprocess.check_output(["docker", "images", "-q", f"{docker_prefix}:{githash}"])
                if image == []:
                    info(f"Still couldn't find image {docker_prefix}:{githash} after trying to build one", dbg_lvl)
                    exit(1)

        scarab_githash = prepare_simulation(user, scarab_path, descriptor_data['root_dir'], experiment_name, architecture, dbg_lvl)

        # Iterate over each workload and config combo
        for simulation in simulations:
            suite = simulation["suite"]
            subsuite = simulation["subsuite"]
            workload = simulation["workload"]
            exp_cluster_id = simulation["cluster_id"]
            sim_mode = simulation["simulation_type"]

            # Run all the workloads within suite
            if workload == None and subsuite == None:
                for subsuite in suite_data[suite].keys():
                    for workload in suite_data[suite][subsuite]["predefined_simulation_mode"].keys():
                        sim_mode = suite_data[suite][subsuite]["predefined_simulation_mode"][workload]
                        run_single_workload(workload, exp_cluster_id, sim_mode)
            elif workload == None and subsuite != None:
                for workload in suite_data[suite][subsuite]["predefined_simulation_mode"].keys():
                    sim_mode = suite_data[suite][subsuite]["predefined_simulation_mode"][workload]
                    run_single_workload(workload, exp_cluster_id, sim_mode)
            else:
                run_single_workload(workload, exp_cluster_id, sim_mode)

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
        traceback.print_exc()  # Print the full stack trace
        for p in processes:
            p.kill()

        # Clean up temp files
        for tmp in tmp_files:
            info(f"Removing temporary run script {tmp}", dbg_lvl)
            os.remove(tmp)

        infra_dir = subprocess.check_output(["pwd"]).decode("utf-8").split("\n")[0]
        print(infra_dir)
        kill_jobs(user, "simulation", experiment_name, docker_prefix_list, infra_dir, dbg_lvl)

def run_tracing(user, descriptor_data, workload_db_path, suite_db_path, infra_dir, dbg_lvl = 2):
    trace_name = descriptor_data["trace_name"]
    docker_home = descriptor_data["root_dir"]
    scarab_path = descriptor_data["scarab_path"]
    simpoint_traces_dir = descriptor_data["simpoint_traces_dir"]
    trace_configs = descriptor_data["trace_configurations"]

    docker_prefix_list = []
    for config in trace_configs:
        image_name = config["image_name"]
        if image_name not in docker_prefix_list:
            docker_prefix_list.append(image_name)

    available_cores = os.cpu_count()
    max_processes = int(available_cores * 0.9)
    processes = set()
    tmp_files = []

    def run_single_trace(workload, image_name, trace_name, env_vars, binary_cmd, client_bincmd, post_processing, drio_args, clustering_k):
        try:
            simpoint_mode = "cluster_then_trace"
            if post_processing:
                simpoint_mode = "trace_then_post_process"
            info(f"Using docker image with name {image_name}:{githash}", dbg_lvl)
            docker_container_name = f"{image_name}_{workload}_{trace_name}_{simpoint_mode}_{user}"
            filename = f"{docker_container_name}_tmp_run.sh"
            write_trace_docker_command_to_file(user, local_uid, local_gid, docker_container_name, githash,
                                               workload, image_name, trace_name, simpoint_traces_dir, docker_home,
                                               env_vars, binary_cmd, client_bincmd, simpoint_mode, drio_args,
                                               clustering_k, filename, infra_dir)
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
        except Exception as e:
            raise e

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

        for docker_prefix in docker_prefix_list:
            image = subprocess.check_output(["docker", "images", "-q", f"{docker_prefix}:{githash}"])
            if image == []:
                info(f"Couldn't find image {docker_prefix}:{githash}", dbg_lvl)
                subprocess.check_output(["docker", "pull", f"ghcr.io/litz-lab/scarab-infra/{docker_prefix}:{githash}"])
                image = subprocess.check_output(["docker", "images", "-q", f"ghcr.io/litz-lab/scarab-infra/{docker_prefix}:{githash}"])
                if image != []:
                    subprocess.check_output(["docker", "tag", f"ghcr.io/litz-lab/scarab-infra/{docker_prefix}:{githash}", f"{docker_prefix}:{githash}"])
                    subprocess.check_output(["docker", "rmi", f"ghcr.io/litz-lab/scarab-infra/{docker_prefix}:{githash}"])
                    continue
                subprocess.check_output(["./run.sh", "-b", docker_prefix])
                image = subprocess.check_output(["docker", "images", "-q", f"{docker_prefix}:{githash}"])
                if image == []:
                    info(f"Still couldn't find image {docker_prefix}:{githash} after trying to build one", dbg_lvl)
                    exit(1)

        prepare_trace(user, scarab_path, docker_home, trace_name, dbg_lvl)

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
            post_processing = config["post_processing"]
            drio_args = config["dynamorio_args"]
            clustering_k = config["clustering_k"]

            run_single_trace(workload, image_name, trace_name, env_vars, binary_cmd, client_bincmd,
                             post_processing, drio_args, clustering_k)

        print("Wait processes...")
        for p in processes:
            p.wait()

        # Clean up temp files
        for tmp in tmp_files:
            info(f"Removing temporary run script {tmp}", dbg_lvl)
            os.remove(tmp)

        print("Recover the ASLR setting with sudo. Provide password..")
        os.system("echo 2 | sudo tee /proc/sys/kernel/randomize_va_space")
        finish_trace(user, descriptor_data, workload_db_path, suite_db_path, dbg_lvl)
    except Exception as e:
        print("An exception occurred:", e)
        traceback.print_exc()  # Print the full stack trace
        for p in processes:
            p.kill()

        # Clean up temp files
        for tmp in tmp_files:
            info(f"Removing temporary run script {tmp}", dbg_lvl)
            os.remove(tmp)

        print("Recover the ASLR setting with sudo. Provide password..")
        os.system("echo 2 | sudo tee /proc/sys/kernel/randomize_va_space")
