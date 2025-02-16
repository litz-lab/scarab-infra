#!/usr/bin/python3

# 01/27/2025 | Surim Oh | run_simulation.py
# An entrypoint script that performs Scarab runs on Slurm clusters using docker containers or on local

import subprocess
import argparse
import os
from utilities import (
        info,
        read_descriptor_from_json,
        verify_descriptor,
        open_interactive_shell,
        remove_docker_containers,
        get_image_list
        )
import slurm_runner
import local_runner


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Runs scarab on local or a slurm network')

    # Add arguments
    parser.add_argument('-d','--descriptor_name', required=True, help='Experiment descriptor name. Usage: -d exp.json')
    parser.add_argument('-k','--kill', required=False, default=False, action=argparse.BooleanOptionalAction, help='Don\'t launch jobs from descriptor, kill running jobs as described in descriptor')
    parser.add_argument('-i','--info', required=False, default=False, action=argparse.BooleanOptionalAction, help='Get info about all nodes and if they have containers for slurm workloads')
    parser.add_argument('-l','--launch', required=False, default=False, action=argparse.BooleanOptionalAction, help='Launch a docker container on a node for the purpose of development/debugging where the environment is for the experiment described in a descriptor.')
    parser.add_argument('-c','--clean', required=False, default=False, action=argparse.BooleanOptionalAction, help='Clean up all the docker containers related to an experiment')
    parser.add_argument('-dbg','--debug', required=False, type=int, default=2, help='1 for errors, 2 for warnings, 3 for info')
    parser.add_argument('-si','--scarab_infra', required=False, default=None, help='Path to scarab infra repo to launch new containers')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Assign clear names to arguments
    descriptor_path = args.descriptor_name
    dbg_lvl = args.debug
    infra_dir = args.scarab_infra

    if infra_dir == None:
        infra_dir = subprocess.check_output(["pwd"]).decode("utf-8").split("\n")[0]

    workload_db_path = f"{infra_dir}/workloads/workloads_db.json"
    suite_db_path = f"{infra_dir}/workloads/suite_db.json"

    # Get user for commands
    user = subprocess.check_output("whoami").decode('utf-8')[:-1]
    info(f"User detected as {user}", dbg_lvl)

    # Read descriptor json and extract important data
    descriptor_data = read_descriptor_from_json(descriptor_path, dbg_lvl)
    workloads_data = read_descriptor_from_json(workload_db_path, dbg_lvl)
    suite_data = read_descriptor_from_json(suite_db_path, dbg_lvl)
    workload_manager = descriptor_data["workload_manager"]
    experiment_name = descriptor_data["experiment"]
    simulations = descriptor_data["simulations"]
    docker_image_list = get_image_list(simulations, workloads_data, suite_data)

    if args.kill:
        if workload_manager == "manual":
            local_runner.kill_jobs(user, experiment_name, docker_image_list, infra_dir, dbg_lvl)
        else:
            slurm_runner.kill_jobs(user, experiment_name, docker_image_list, dbg_lvl)
        exit(0)

    if args.info:
        if workload_manager == "manual":
            local_runner.print_status(user, experiment_name, docker_image_list, dbg_lvl)
        else:
            slurm_runner.print_status(user, experiment_name, docker_image_list, dbg_lvl)
        exit(0)

    if args.launch:
        verify_descriptor(descriptor_data, workloads_data, suite_data, True, dbg_lvl)
        open_interactive_shell(user, descriptor_data, workloads_data, suite_data, dbg_lvl)
        exit(0)

    if args.clean:
        remove_docker_containers(docker_image_list, experiment_name, user, dbg_lvl)
        exit(0)

    verify_descriptor(descriptor_data, workloads_data, suite_data, False, dbg_lvl)
    if workload_manager == "manual":
        local_runner.run_simulation(user, descriptor_data, workloads_data, suite_data, dbg_lvl)
    else:
        slurm_runner.run_simulation(user, descriptor_data, workloads_data, suite_data, dbg_lvl)