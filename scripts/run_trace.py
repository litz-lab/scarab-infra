#!/usr/bin/python3

# 02/17/2025 | Surim Oh | run_trace.py
# An entrypoint script that run clustering and tracing on Slurm clusters using docker containers or on local

import subprocess
import argparse
import os
import docker

from utilities import (
        info,
        err,
        read_descriptor_from_json,
        remove_docker_containers,
        prepare_trace
        )

import slurm_runner
import local_runner

client = docker.from_env()

def validate_tracing(trace_data, workload_db_path, dbg_lvl = 2):
    workload_data = read_descriptor_from_json(workload_db_path, dbg_lvl)
    for trace in trace_data:
        if trace["workload"] == None:
            err(f"A workload name must be provided.", dbg_lvl)
            exit(1)

        if trace["workload"] in workload_data.keys():
            if "trace" in workload_data[trace['workload']].keys():
                err(f"{trace['workload']} already exists in workload database (workloads/workloads_db.json). Choose a different workload name.", dbg_lvl)
                exit(1)

        if trace["suite"] == None:
            err(f"A suite name must be provided.", dbg_lvl)
            exit(1)

        if trace["image_name"] == None:
            err(f"An image name must be provided.", dbg_lvl)
            exit(1)

        if trace["binary_cmd"] == None:
            err(f"A binary command must be provided.", dbg_lvl)
            exit(1)

        if trace["post_processing"] == None:
            err(f"true or false must be set for post_processing.", dbg_lvl)
            exit(1)

def verify_descriptor(descriptor_data, workload_db_path, dbg_lvl = 2):
    # Check the descriptor type
    if not descriptor_data["descriptor_type"]:
        err("Descriptor type must be 'trace' for a clustering/tracing descriptor", dbg_lvl)
        exit(1)

    # Check the scarab path
    if descriptor_data["scarab_path"] == None:
        err("Need path to scarab path. Set in descriptor file under 'scarab_path'", dbg_lvl)
        exit(1)

    # Check trace doesn't already exists
    trace_dir = f"{descriptor_data['root_dir']}/traces/{descriptor_data['trace_name']}"
    if os.path.exists(trace_dir) and not open_shell:
        err(f"Trace '{trace_dir}' already exists. Please try a different name or remote the directory if not needed.", dbg_lvl)
        exit(1)

    # Check if each trace scenario is valid
    validate_tracing(descriptor_data["trace_configurations"], workload_db_path, dbg_lvl)

    # Check the workload manager
    if descriptor_data["workload_manager"] != "manual" and descriptor_data["workload_manager"] != "slurm":
        err("Workload manager options: 'manual' or 'slurm'.", dbg_lvl)
        exit(1)

    # Check if docker home path is provided
    if descriptor_data["root_dir"] == None:
        err("Need path to docker home directory. Set in descriptor file under 'root_dir'", dbg_lvl)
        exit(1)

    # Check if trace dir exists
    if descriptor_data["simpoint_traces_dir"] == None:
        err("Need path to write the newly collected simpoints and traces. Set in descriptor file under 'simpoint_traces_dir'", dbg_lvl)
        exit(1)

def get_image_list(traces):
    image_list = []
    for trace in traces:
        image_list.append(trace["image_name"])

    return image_list

def open_interactive_shell(user, descriptor_data, infra_dir, dbg_lvl = 1):
    trace_name = descriptor_data["trace_name"]
    try:
        # Get user for commands
        user = subprocess.check_output("whoami").decode('utf-8')[:-1]
        info(f"User detected as {user}", dbg_lvl)

        # Get a local user/group ids
        local_uid = os.getuid()
        local_gid = os.getgid()
        print(local_uid)
        print(local_gid)

        # Get GitHash
        try:
            githash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("utf-8").strip()
            info(f"Git hash: {githash}", dbg_lvl)
        except FileNotFoundError:
            err("Error: 'git' command not found. Make sure Git is installed and in your PATH.")
        except subprocess.CalledProcessError:
            err("Error: Not in a Git repository or unable to retrieve Git hash.")

        docker_home = descriptor_data["root_dir"]
        prepare_trace(user, descriptor_data["scarab_path"], docker_home, trace_name, dbg_lvl)
        trace_scenario = descriptor_data["trace_configurations"][0]
        workload = trace_scenario["workload"]
        docker_prefix = trace_scenario["image_name"]
        if trace_scenario["env_vars"] != None:
            env_vars = trace_scenario["env_vars"].split()
        else:
            env_vars = trace_scenario["env_vars"]

        scarab_githash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=descriptor_data['scarab_path']).decode("utf-8").strip()
        info(f"Scarab git hash: {scarab_githash}", dbg_lvl)

        docker_container_name = f"{docker_prefix}_{trace_name}_scarab_{scarab_githash}_{user}"
        trace_dir = f"{docker_home}/simpoint_flow/{trace_name}"
        # os.system(f"chmod -R 777 {trace_dir}")
        try:
            command = f"docker run --privileged \
                    -e user_id={local_uid} \
                    -e group_id={local_gid} \
                    -e username={user} \
                    -e HOME=/home/{user} \
                    -e APP_GROUPNAME={docker_prefix} \
                    -e APPNAME={workload} "
            if env_vars:
                for env in env_vars:
                    command = command + f"-e {env} "
            command = command + f"-dit \
                    --name {docker_container_name} \
                    --mount type=bind,source={docker_home},target=/home/{user} \
                    {docker_prefix}:{githash} \
                    /bin/bash"
            print(command)
            os.system(command)
            os.system(f"docker cp {infra_dir}/scripts/utilities.sh {docker_container_name}:/usr/local/bin")
            os.system(f"docker cp {infra_dir}/common/scripts/common_entrypoint.sh {docker_container_name}:/usr/local/bin")
            os.system(f"docker cp {infra_dir}/common/scripts/run_clustering.sh {docker_container_name}:/usr/local/bin")
            os.system(f"docker cp {infra_dir}/common/scripts/run_simpoint_trace.sh {docker_container_name}:/usr/local/bin")
            os.system(f"docker cp {infra_dir}/common/scripts/run_trace_post_processing.sh {docker_container_name}:/usr/local/bin")
            os.system(f"docker exec --privileged {docker_container_name} /bin/bash -c '/usr/local/bin/common_entrypoint.sh'")
            os.system(f"docker exec --privileged {docker_container_name} /bin/bash -c \"echo 0 | sudo tee /proc/sys/kernel/randomize_va_space\"")
            subprocess.run(["docker", "exec", "-it", f"--user={user}", f"--workdir=/home/{user}", docker_container_name, "/bin/bash"])
            # subprocess.run(["docker", "exec", "-it", f"--workdir=/home/{user}", docker_container_name, "/bin/bash"])
        except KeyboardInterrupt:
            os.system(f"docker rm -f {docker_container_name}")
            print("Recover the ASLR setting with sudo. Provide password..")
            os.system("echo 2 | sudo tee /proc/sys/kernel/randomize_va_space")
            exit(0)
        finally:
            try:
                client.containers.get(docker_container_name).remove(force=True)
                print(f"Container {docker_container_name} removed.")
            except docker.error.NotFound:
                print(f"Container {docker_container_name} not found.")
    except Exception as e:
        raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Runs clustering/tracing on local or a slurm network')

    # Add arguments
    parser.add_argument('-d','--descriptor_name', required=True, help='Tracing descriptor name. Usage: -d trace.json')
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
    workload_manager = descriptor_data["workload_manager"]
    trace_name = descriptor_data["trace_name"]
    traces = descriptor_data["trace_configurations"]
    docker_image_list = get_image_list(traces)

    if args.kill:
        if workload_manager == "manual":
            local_runner.kill_jobs(user, "trace", trace_name, docker_image_list, infra_dir, dbg_lvl)
        else:
            slurm_runner.kill_jobs(user, trace_name, docker_image_list, dbg_lvl)
        exit(0)

    if args.info:
        if workload_manager == "manual":
            local_runner.print_status(user, trace_name, docker_image_list, dbg_lvl)
        else:
            slurm_runner.print_status(user, trace_name, docker_image_list, dbg_lvl)
        exit(0)

    if args.launch:
        verify_descriptor(descriptor_data, workload_db_path, dbg_lvl)
        open_interactive_shell(user, descriptor_data, infra_dir, dbg_lvl)
        exit(0)

    if args.clean:
        remove_docker_containers(docker_image_list, trace_name, user, dbg_lvl)
        exit(0)

    verify_descriptor(descriptor_data, workload_db_path, dbg_lvl)
    if workload_manager == "manual":
        local_runner.run_tracing(user, descriptor_data, workload_db_path, suite_db_path, infra_dir, dbg_lvl)
    else:
        slurm_runner.run_tracing(user, descriptor_data, workload_db_path, suite_db_path, infra_dir, dbg_lvl)
