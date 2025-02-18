#!/usr/bin/python3

# 01/27/2025 | Surim Oh | run_simulation.py
# An entrypoint script that performs Scarab runs on Slurm clusters using docker containers or on local

import subprocess
import argparse
import os
import docker

from utilities import (
        info,
        read_descriptor_from_json,
        remove_docker_containers,
        get_image_list,
        prepare_simulation,
        get_image_name
        )
import slurm_runner
import local_runner

client = docker.from_env()

# Verify the given descriptor file
def verify_descriptor(descriptor_data, workloads_data, suite_data, open_shell = False, dbg_lvl = 2):
    ## Check if the provided json describes all the valid data

    # Check the descriptor type
    if not descriptor_data["descriptor_type"]:
        err("Descriptor type must be 'simulation' for a simulation descriptor", dbg_lvl)
        exit(1)

    # Check the scarab path
    if descriptor_data["scarab_path"] == None:
        err("Need path to scarab path. Set in descriptor file under 'scarab_path'", dbg_lvl)
        exit(1)

    # Check if a correct architecture spec is provided
    if descriptor_data["architecture"] == None:
        err("Need an architecture spec to simulate. Set in descriptor file under 'architecture'. Available architectures are found from PARAMS.<architecture> in scarab repository. e.g) sunny_cove", dbg_lvl)
        exit(1)
    elif not os.path.exists(f"{descriptor_data['scarab_path']}/src/PARAMS.{descriptor_data['architecture']}"):
        err(f"PARAMS.{descriptor_data['architecture']} does not exist. Please provide an available architecture for scarab simulation", dbg_lvl)
        exit(1)

    # Check experiment doesn't already exists
    experiment_dir = f"{descriptor_data['root_dir']}/simulations/{descriptor_data['experiment']}"
    if os.path.exists(experiment_dir) and not open_shell:
        err(f"Experiment '{experiment_dir}' already exists. Please try a different name or remove the directory if not needed", dbg_lvl)
        exit(1)

    # Check if each simulation type is valid
    validate_simulation(workloads_data, suite_data, descriptor_data['simulations'])

    # Check the workload manager
    if descriptor_data["workload_manager"] != "manual" and descriptor_data["workload_manager"] != "slurm":
        err("Workload manager options: 'manual' or 'slurm'.", dbg_lvl)
        exit(1)

    # Check if docker home path is provided
    if descriptor_data["root_dir"] == None:
        err("Need path to docker home directory. Set in descriptor file under 'root_dir'", dbg_lvl)
        exit(1)

    # Check if the provided scarab path exists
    if descriptor_data["scarab_path"] == None:
        err("Need path to scarab directory. Set in descriptor file under 'scarab_path'", dbg_lvl)
        exit(1)
    elif not os.path.exists(descriptor_data["scarab_path"]):
        err(f"{descriptor_data['scarab_path']} does not exist.", dbg_lvl)
        exit(1)

    # Check if trace dir exists
    if descriptor_data["simpoint_traces_dir"] == None:
        err("Need path to simpoints and traces. Set in descriptor file under 'simpoint_traces_dir'", dbg_lvl)
        exit(1)
    elif not os.path.exists(descriptor_data["simpoint_traces_dir"]):
        err(f"{descriptor_data['simpoint_traces_dir']} does not exist.", dbg_lvl)
        exit(1)

    # Check if configurations are provided
    if descriptor_data["configurations"] == None:
        error("Need configurations to simulate. Set in descriptor file under 'configurations'", dbg_lvl)
        exit(1)

def open_interactive_shell(user, descriptor_data, workloads_data, suite_data, infra_dir, dbg_lvl = 1):
    experiment_name = descriptor_data["experiment"]
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
            err("Error: 'git' command not found. Make sure Git is installed and in your PATH.")
        except subprocess.CalledProcessError:
            err("Error: Not in a Git repository or unable to retrieve Git hash.")

        # TODO: always make sure to open the interactive shell on a development node (not worker nodes) if slurm mode
        # need to maintain the list of nodes for development
        # currently open it on local

        # Generate commands for executing in users docker and sbatching to nodes with containers
        scarab_githash = prepare_simulation(user,
                                            descriptor_data['scarab_path'],
                                            descriptor_data['root_dir'],
                                            experiment_name,
                                            descriptor_data['architecture'],
                                            dbg_lvl)
        workload = descriptor_data['simulations'][0]['workload']
        mode = descriptor_data['simulations'][0]['simulation_type']
        docker_prefix = get_image_name(workloads_data, suite_data, descriptor_data['simulations'][0])

        docker_container_name = f"{docker_prefix}_{experiment_name}_scarab_{scarab_githash}_{user}"
        simpoint_traces_dir = descriptor_data["simpoint_traces_dir"]
        docker_home = descriptor_data["root_dir"]
        try:
            os.system(f"docker run \
                -e user_id={local_uid} \
                -e group_id={local_gid} \
                -e username={user} \
                -e HOME=/home/{user} \
                -e APP_GROUPNAME={docker_prefix} \
                -e APPNAME={workload} \
                -dit \
                --name {docker_container_name} \
                --mount type=bind,source={simpoint_traces_dir},target=/simpoint_traces,readonly \
                --mount type=bind,source={docker_home},target=/home/{user} \
                {docker_prefix}:{githash} \
                /bin/bash")
            os.system(f"docker cp {infra_dir}/scripts/utilities.sh {docker_container_name}:/usr/local/bin")
            os.system(f"docker cp {infra_dir}/common/scripts/common_entrypoint.sh {docker_container_name}:/usr/local/bin")
            if mode == "memtrace":
                os.system(f"docker cp {infra_dir}/common/scripts/run_memtrace_single_simpoint.sh {docker_container_name}:/usr/local/bin")
            elif mode == "pt":
                os.system(f"docker cp {infra_dir}/common/scripts/run_pt_single_simpoint.sh {docker_container_name}:/usr/local/bin")
            elif mode == "exec":
                os.system(f"docker cp {infra_dir}/common/scripts/run_exec_single_simpoint.sh {docker_container_name}:/usr/local/bin")
            os.system(f"docker exec {docker_container_name} /bin/bash -c '/usr/local/bin/common_entrypoint.sh'")
            subprocess.run(["docker", "exec", "-it", f"--user={user}", f"--workdir=/home/{user}", docker_container_name, "/bin/bash"])
        except KeyboardInterrupt:
            os.system(f"docker rm -f {docker_container_name}")
            exit(0)
        finally:
            try:
                client.containers.get(docker_container_name).remove(force=True)
                print(f"Container {docker_container_name} removed.")
            except docker.errors.NotFound:
                print(f"Container {docker_container_name} not found.")
    except Exception as e:
        raise e

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
            local_runner.kill_jobs(user, "simulation", experiment_name, docker_image_list, infra_dir, dbg_lvl)
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
        open_interactive_shell(user, descriptor_data, workloads_data, suite_data, infra_dir, dbg_lvl)
        exit(0)

    if args.clean:
        remove_docker_containers(docker_image_list, experiment_name, user, dbg_lvl)
        exit(0)

    verify_descriptor(descriptor_data, workloads_data, suite_data, False, dbg_lvl)
    if workload_manager == "manual":
        local_runner.run_simulation(user, descriptor_data, workloads_data, suite_data, infra_dir, dbg_lvl)
    else:
        slurm_runner.run_simulation(user, descriptor_data, workloads_data, suite_data, infra_dir, dbg_lvl)