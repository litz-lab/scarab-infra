#!/usr/bin/python3

# 01/27/2025 | Surim Oh | local_runner.py

import os
import subprocess
import psutil
import signal
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
        remove_docker_containers,
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
        generate_table
        )
from .slurm_runner import get_simulation_jobs

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
def print_status(user, job_name, docker_prefix_list, descriptor_data=None, workloads_data=None, dbg_lvl = 2):
    docker_running = check_docker_container_running(docker_prefix_list, job_name, user, dbg_lvl)
    if len(docker_running) > 0:
        print(f"\033[92mRUNNING:     local\033[0m")
        for docker in docker_running:
            print(f"\033[92m    CONTAINER: {docker}\033[0m")
    else:
        print(f"\033[31mNOT RUNNING: local\033[0m")

    if descriptor_data is None or workloads_data is None:
        return

    running_sims = docker_running
    queued_sims = []
    all_jobs = get_simulation_jobs(descriptor_data, workloads_data, docker_prefix_list, user, dbg_lvl)

    root_directory = os.path.join(
        descriptor_data["root_dir"],
        "simulations",
        descriptor_data["experiment"],
    )
    root_logfile_directory = os.path.join(root_directory, "logs")
    os.system(f"ls -R {root_directory} > /dev/null")

    try:
        log_files = os.listdir(root_logfile_directory)
    except Exception:
        print("Log file directory does not exist")
        print("The current experiment does not seem to have been run yet")
        return

    if len(log_files) > len(all_jobs):
        print("More log files than total runs. Maybe same experiment name was run multiple times?")
        print("Any errors from a previous run with the same experiment name will be re-reported now")

    error_runs = set()
    skipped = 0

    confs = list(descriptor_data["configurations"].keys())

    completed = {conf: 0 for conf in confs}
    failed = {conf: 0 for conf in confs}
    prep_failed = {conf: 0 for conf in confs}
    running = {conf: 0 for conf in confs}
    pending = {conf: 0 for conf in confs}

    experiment_name = descriptor_data["experiment"]
    for sim in queued_sims:
        matches = [conf for conf in confs if f"{experiment_name}_{conf}" in sim]
        if not matches:
            info(f"'{experiment_name}_{conf}' not found in any queued sim names", dbg_lvl)
            continue
        conf = max(matches, key=len)
        pending[conf] += 1

    not_in_experiment = []
    for file in log_files:
        log_path = os.path.join(root_logfile_directory, file)
        with open(log_path, 'r') as f:
            contents = f.read()
            contents_after_docker = contents
            if len(contents.split("\n")) < 2:
                continue

            first_line = contents.split("\n")[0]
            config = first_line.split(" ")[1]
            cluster_id = first_line.split(" ")[3]
            workload_path = first_line.split(" ")[2]
            scarab_logfile_path = os.path.join(
                root_directory,
                config,
                workload_path,
                cluster_id,
                "sim.log",
            )

            if config not in confs:
                if config not in not_in_experiment:
                    print(f"WARN: Log files for config {config}, which is not in the experiment file")
                not_in_experiment.append(config)
                continue

            pattern = r"Script name: (\S*)"
            match = re.search(pattern, contents)
            if match:
                script_name = match.group(1)
                is_running = any(sim in script_name for sim in running_sims)
                if is_running:
                    skipped += 1
                    running[config] += 1
                    continue

            if "BEGIN prepare_docker_image" in contents:
                if "FAILED prepare_docker_image" in contents:
                    prep_failed[config] += 1
                    error_runs.add(log_path)
                    print("Docker image preparation failed, Simulation is not running (Error message in log file)")
                    continue

                if "END prepare_docker_image" in contents:
                    contents_after_docker = contents.split("END prepare_docker_image\n")[1]
                else:
                    prep_failed[config] += 1
                    error_runs.add(log_path)
                    print("Docker image preparation failed, Simulation is not running (Image prep never completed; no failure message)")
                    continue
            else:
                print("Docker image preparation failed (Image prep never started)")
                prep_failed[config] += 1
                error_runs.add(log_path)
                continue

            error = 0
            if 'Segmentation fault' in contents_after_docker:
                error = 1

            if 'error' in contents_after_docker.lower():
                error = 1

            if "Completed Simulation" in contents_after_docker and not error:
                workload_parts = workload_path.split("/")
                sim_dir = Path(descriptor_data["root_dir"]) / "simulations" / descriptor_data["experiment"] / config
                sim_dir = sim_dir.joinpath(*workload_parts, cluster_id)
                if any(list(map(lambda x: x.endswith(".csv"), os.listdir(sim_dir)))):
                    completed[config] += 1
                    continue
                err("Stat files not generated, despite being completed with no errors.", 1)

            error_runs.add(scarab_logfile_path)
            failed[config] += 1

    print(f"Currently running {len(running_sims)} simulations (from logs: {skipped})")

    calculated_logfile_count = 0
    data = {
        "Configuration": [],
        "Completed": [],
        "Failed": [],
        "Failed - Prep": [],
        "Running": [],
        "Pending": [],
        "Non-existant": [],
        "Total": [],
    }
    for conf in confs:
        data["Configuration"].append(conf)
        data["Completed"].append(completed[conf])
        data["Failed"].append(failed[conf])
        data["Failed - Prep"].append(prep_failed[conf])
        data["Running"].append(running[conf])
        data["Pending"].append(pending[conf])

        total_per_conf = int(len(all_jobs) / len(confs))
        total_found = completed[conf] + failed[conf] + running[conf] + pending[conf] + prep_failed[conf]
        calculated_logfile_count += total_found - pending[conf]

        assert total_per_conf >= total_found, "ERR: Assert Failed: More jobs found than should exist"

        data["Total"].append(total_per_conf)
        data["Non-existant"].append(total_per_conf - total_found)

    if len(not_in_experiment) == 0 and calculated_logfile_count != len(log_files):
        warn("Log file count doesn't match number of accounted jobs.", dbg_lvl)

    print(generate_table(data))

    if error_runs:
        error_list = sorted(error_runs)
        print(f"\033[31mErroneous Jobs: {len(error_list)}\033[0m")
        print(f"\033[31mErrors found in {len(error_list)}/{len(log_files)} log files.")
        print("First 5 error runs:\n", "\n".join(error_list[:5]), "\033[0m", sep='')
    else:
        print(f"\033[92mNo errors found in log files\033[0m")

def kill_jobs(user, job_type, job_name, docker_prefix_list, infra_dir, dbg_lvl):
    # Define the process name pattern
    if job_type == "simulation":
        pattern = re.compile(f"python3 -m scripts.run_simulation -dbg 3 -d {infra_dir}/json/{job_name}.json")
    elif job_type == "trace":
        pattern = re.compile(f"python3 -m scripts.run_trace -dbg 3 -d {infra_dir}/json/{job_name}.json")

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

    available_cores = os.cpu_count()
    max_processes = int(available_cores * 0.9)
    processes = set()
    process_logs = {}
    tmp_files = []
    log_files = []
    log_index = 0

    dont_collect = True

    def run_single_workload(suite, subsuite, workload, exp_cluster_id, sim_mode, warmup):
        nonlocal dont_collect
        nonlocal log_index
        try:
            docker_prefix = get_docker_prefix(sim_mode, workloads_data[suite][subsuite][workload]["simulation"])
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
                simpoints[f"{exp_cluster_id}"] = weight

            for config_key in configs:
                config = configs[config_key]["params"]
                scarab_binary = configs[config_key]["binary"]
                if config == "":
                    config = None

                for cluster_id, weight in simpoints.items():
                    info(f"cluster_id: {cluster_id}, weight: {weight}", dbg_lvl)
                    
                    dont_collect = False

                    docker_container_name = f"{docker_prefix}_{suite}_{subsuite}_{workload}_{experiment_name}_{config_key.replace("/", "-")}_{cluster_id}_{sim_mode}_{user}"
                    # Create temp file with run command and run it
                    filename = f"{docker_container_name}_tmp_run.sh"

                    if check_can_skip(descriptor_data, config_key, suite, subsuite, workload, cluster_id, filename, sim_mode, user, dbg_lvl=dbg_lvl):
                        info(f"Skipping {workload} with config {config_key} and cluster id {cluster_id}", dbg_lvl)
                        continue

                    workload_home = f"{suite}/{subsuite}/{workload}"
                    write_docker_command_to_file(user, local_uid, local_gid, workload, workload_home, experiment_name,
                                                 docker_prefix, docker_container_name, traces_dir,
                                                 docker_home, githash, config_key, config, sim_mode, scarab_binary,
                                                 seg_size, architecture, cluster_id, warmup, trace_warmup, trace_type,
                                                 trace_file, env_vars, bincmd, client_bincmd, filename, infra_dir)
                    tmp_files.append(filename)
                    command = '/bin/bash ' + filename
                    log_path = os.path.join(
                        docker_home,
                        "simulations",
                        experiment_name,
                        "logs",
                        f"local_job_{log_index}.out",
                    )
                    log_index += 1
                    log_handle = open(log_path, "w")
                    log_files.append(log_handle)
                    process = subprocess.Popen(
                        "exec " + command,
                        stdout=log_handle,
                        stderr=log_handle,
                        shell=True,
                    )
                    processes.add(process)
                    process_logs[process] = log_handle
                    info(f"Running command '{command}'", dbg_lvl)
                    while len(processes) >= max_processes:
                        # Loop through the processes and wait for one to finish
                        for p in processes.copy():
                            if p.poll() is not None: # This process has finished
                                p.wait() # Make sure it's really finished
                                processes.remove(p) # Remove from set of active processes
                                handle = process_logs.pop(p, None)
                                if handle:
                                    handle.close()
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

        scarab_binaries = []
        for sim in simulations:
            for config_key in configs:
                scarab_binary = configs[config_key]["binary"]
                if scarab_binary not in scarab_binaries:
                    scarab_binaries.append(scarab_binary)

        try:
            scarab_githash, image_tag_list = prepare_simulation(user, scarab_path, scarab_build, descriptor_data['root_dir'], experiment_name, architecture, docker_prefix_list, githash, infra_dir, scarab_binaries, interactive_shell=False, dbg_lvl=dbg_lvl)
        except RuntimeError as e:
            # This error prints a message. Now stop execution
            return
        os.makedirs(os.path.join(docker_home, "simulations", experiment_name, "logs"), exist_ok=True)

        print("Submitting jobs...")
        # Iterate over each workload and config combo
        for simulation in simulations:
            suite = simulation["suite"]
            subsuite = simulation["subsuite"]
            workload = simulation["workload"]
            exp_cluster_id = simulation["cluster_id"]
            sim_mode = simulation["simulation_type"]
            sim_warmup = simulation["warmup"]

            # Run all the workloads within suite
            if workload == None and subsuite == None:
                for subsuite_ in workloads_data[suite].keys():
                    for workload_ in workloads_data[suite][subsuite_].keys():
                        sim_mode_ = sim_mode
                        if sim_mode_ == None:
                            sim_mode_ = workloads_data[suite][subsuite_][workload_]["simulation"]["prioritized_mode"]
                        if sim_warmup == None:  # Use the whole warmup available in the trace if not specified
                            sim_warmup = workloads_data[suite][subsuite_][workload_]["simulation"][sim_mode_]["warmup"]
                        run_single_workload(suite, subsuite_, workload_, exp_cluster_id, sim_mode_, sim_warmup)
            elif workload == None and subsuite != None:
                for workload_ in workloads_data[suite][subsuite].keys():
                    sim_mode_ = sim_mode
                    if sim_mode_ == None:
                        sim_mode_ = workloads_data[suite][subsuite][workload_]["simulation"]["prioritized_mode"]
                    if sim_warmup == None:  # Use the whole warmup available in the trace if not specified
                        sim_warmup = workloads_data[suite][subsuite][workload_]["simulation"][sim_mode_]["warmup"]
                    run_single_workload(suite, subsuite, workload_, exp_cluster_id, sim_mode_, sim_warmup)
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
                run_single_workload(suite, _subsuite, workload, exp_cluster_id, sim_mode_, sim_warmup)
            else:
                sim_mode_ = sim_mode
                if sim_mode_ == None:
                    sim_mode_ = workloads_data[suite][subsuite][workload]["simulation"]["prioritized_mode"]
                if sim_warmup == None:  # Use the whole warmup available in the trace if not specified
                    sim_warmup = workloads_data[suite][subsuite][workload]["simulation"][sim_mode_]["warmup"]
                run_single_workload(suite, subsuite, workload, exp_cluster_id, sim_mode_, sim_warmup)

        print("Wait processes...")
        for p in processes:
            p.wait()

        for p in processes:
            log_handle = process_logs.get(p)
            if log_handle:
                log_handle.close()

        # Clean up temp files
        for tmp in tmp_files:
            info(f"Removing temporary run script {tmp}", dbg_lvl)
            os.remove(tmp)

        finish_simulation(user, docker_home, descriptor_path, descriptor_data['root_dir'], experiment_name, image_tag_list, [], dont_collect=dont_collect)

    except Exception as e:
        print("An exception occurred:", e)
        traceback.print_exc()  # Print the full stack trace
        for p in processes:
            p.kill()

        for handle in log_files:
            handle.close()

        # Clean up temp files
        for tmp in tmp_files:
            info(f"Removing temporary run script {tmp}", dbg_lvl)
            os.remove(tmp)

        infra_dir = subprocess.check_output(["pwd"]).decode("utf-8").split("\n")[0]
        print(infra_dir)
        kill_jobs(user, "simulation", experiment_name, docker_prefix_list, infra_dir, dbg_lvl)

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

    available_cores = os.cpu_count()
    max_processes = int(available_cores * 0.9)
    processes = set()
    tmp_files = []
    log_files = []

    def run_single_trace(workload, image_name, trace_name, env_vars, binary_cmd, client_bincmd, trace_type, drio_args, clustering_k, infra_dir, application_dir):
        try:
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
            command = '/bin/bash ' + filename
            subprocess.run(["mkdir", "-p", f"{docker_home}/simpoint_flow/{trace_name}/{workload}"], check=True, capture_output=True, text=True)
            log_out = f"{docker_home}/simpoint_flow/{trace_name}/{workload}/log.out"
            log_err = f"{docker_home}/simpoint_flow/{trace_name}/{workload}/log.err"
            out = open(log_out, "w")
            err = open(log_err, "w")
            log_files.append((out, err))
            process = subprocess.Popen(command, stdout=out, stderr=err, shell=True, preexec_fn=os.setsid)
            processes.add(process)
            info(f"Running command '{command}'", dbg_lvl)
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


        prepare_trace(user, scarab_path, scarab_build, docker_home, trace_name, infra_dir, docker_prefix_list, githash, False, [], dbg_lvl=dbg_lvl)

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
                             trace_type, drio_args, clustering_k, infra_dir, application_dir)

        print("Wait processes...")
        for p in processes:
            p.wait()

        for out, err in log_files:
            out.close()
            err.close()

        # Clean up temp files
        for tmp in tmp_files:
            info(f"Removing temporary run script {tmp}", dbg_lvl)
            os.remove(tmp)

        finish_trace(user, descriptor_data, workload_db_path, infra_dir, dbg_lvl)
    except Exception as e:
        print("An exception occurred:", e)
        traceback.print_exc()  # Print the full stack trace
        for p in processes:
            p.kill()

        for out, err in log_files:
            out.close()
            err.close()

        # Clean up temp files
        for tmp in tmp_files:
            info(f"Removing temporary run script {tmp}", dbg_lvl)
            os.remove(tmp)

        kill_jobs(user, "trace", trace_name, docker_prefix_list, infra_dir, dbg_lvl)

        print("Recover the ASLR setting with sudo. Provide password..")
        os.system("echo 2 | sudo tee /proc/sys/kernel/randomize_va_space")
