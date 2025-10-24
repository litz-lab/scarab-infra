#!/usr/bin/python3

# 02/17/2025 | Surim Oh | run_trace.py
# An entrypoint script that run clustering and tracing on Slurm clusters using docker containers or on local

import subprocess
import argparse
import os
import sys
import docker

from .utilities import (
    info,
    err,
    read_descriptor_from_json,
    remove_docker_containers,
    prepare_trace,
    is_container_running,
    count_interactive_shells
)
from . import slurm_runner, local_runner

client = docker.from_env()

def validate_tracing(trace_data, workload_db_path, dbg_lvl = 2):
    workload_data = read_descriptor_from_json(workload_db_path, dbg_lvl)
    for trace in trace_data:
        if trace["workload"] == None:
            err(f"A workload name must be provided.", dbg_lvl)
            exit(1)

        if trace["suite"] in workload_data.keys() and trace["subsuite"] in workload_data[trace["suite"]].keys() and trace["workload"] in workload_data[trace["suite"]][trace["subsuite"]].keys():
            if "trace" in workload_data[trace["suite"]][trace["subsuite"]][trace["workload"]].keys():
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

        if trace["trace_type"] == None:
            err(f"A type of trace must be set for trace_type.", dbg_lvl)
            exit(1)

def verify_descriptor(descriptor_data, workload_db_path, open_shell=False, dbg_lvl = 2):
    # Check the descriptor type
    if not descriptor_data["descriptor_type"]:
        err("Descriptor type must be 'trace' for a clustering/tracing descriptor", dbg_lvl)
        exit(1)

    # Check the scarab path
    if descriptor_data["scarab_path"] == None:
        err("Need path to scarab path. Set in descriptor file under 'scarab_path'", dbg_lvl)
        exit(1)

    # Check the scarab build mode
    if descriptor_data["scarab_build"] != None and descriptor_data["scarab_build"] != 'opt' and descriptor_data["scarab_build"] != 'dbg':
        err("Need a valid scarab build mode (\'opt\' or \'dbg\' or null). Set in descriptor file under 'scarab_build'", dbg_lvl)
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
    if descriptor_data["traces_dir"] == None:
        err("Need path to write the newly collected simpoints and traces. Set in descriptor file under 'traces_dir'", dbg_lvl)
        exit(1)

def get_image_list(traces):
    image_list = []
    for trace in traces:
        image_list.append(trace["image_name"])

    return image_list

def open_interactive_shell(user, descriptor_data, infra_dir, dbg_lvl = 1):
    trace_name = descriptor_data["trace_name"]
    scarab_path = descriptor_data["scarab_path"]
    scarab_build = descriptor_data["scarab_build"]
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
        application = descriptor_data["application_dir"]
        trace_scenario = descriptor_data["trace_configurations"][0]
        docker_prefix = trace_scenario["image_name"]
        workload = trace_scenario["workload"]

        # Set the env for simulation again (already set in Dockerfile.common) in case user's bashrc overwrite the existing ones when the home directory is mounted
        bashrc_path = f"{docker_home}/.bashrc"
        entry = "source /usr/local/bin/user_entrypoint.sh"
        with open(bashrc_path, "a+") as f:
            f.seek(0)
            if entry not in f.read():
                f.write(f"\n{entry}\n")

        prepare_trace(user, scarab_path, scarab_build, docker_home, trace_name, infra_dir, [docker_prefix], githash, True, [], dbg_lvl=dbg_lvl)
        if trace_scenario["env_vars"] != None:
            env_vars = trace_scenario["env_vars"].split()
        else:
            env_vars = trace_scenario["env_vars"]

        scarab_githash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=scarab_path).decode("utf-8").strip()
        info(f"Scarab git hash: {scarab_githash}", dbg_lvl)

        docker_container_name = f"{docker_prefix}_{trace_name}_scarab_{scarab_githash}_{user}"
        trace_dir = f"{docker_home}/simpoint_flow/{trace_name}"
        try:
            # If the container is already running, log into it by openning another interactive shell
            if is_container_running(docker_container_name, dbg_lvl):
                subprocess.run(["docker", "exec", "--privileged", "-it", f"--user={user}", f"--workdir=/home/{user}", docker_container_name, "/bin/bash"])
            else:
                info(f"Create a new container for the interactive mode", dbg_lvl)
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
                        --mount type=bind,source={docker_home},target=/home/{user},readonly=false \
                        --mount type=bind,source={scarab_path},target=/scarab,readonly=false \
                        --mount type=bind,source={application},target=/tmp_home/application,readonly=false \
                        {docker_prefix}:{githash} \
                        /bin/bash"
                print(command)
                os.system(command)
                os.system(f"docker cp {infra_dir}/scripts/utilities.sh {docker_container_name}:/usr/local/bin")
                os.system(f"docker cp {infra_dir}/common/scripts/root_entrypoint.sh {docker_container_name}:/usr/local/bin")
                os.system(f"docker cp {infra_dir}/common/scripts/user_entrypoint.sh {docker_container_name}:/usr/local/bin")
                if os.path.exists(f"{infra_dir}/workloads/{docker_prefix}/workload_root_entrypoint.sh"):
                    os.system(f"docker cp {infra_dir}/workloads/{docker_prefix}/workload_root_entrypoint.sh {docker_container_name}:/usr/local/bin")
                if os.path.exists(f"{infra_dir}/workloads/{docker_prefix}/workload_user_entrypoint.sh"):
                    os.system(f"docker cp {infra_dir}/workloads/{docker_prefix}/workload_user_entrypoint.sh {docker_container_name}:/usr/local/bin")
                os.system(f"docker cp {infra_dir}/common/scripts/run_clustering.sh {docker_container_name}:/usr/local/bin")
                os.system(f"docker cp {infra_dir}/common/scripts/run_simpoint_trace.py {docker_container_name}:/usr/local/bin")
                os.system(f"docker cp {infra_dir}/common/scripts/minimize_trace.sh {docker_container_name}:/usr/local/bin")
                os.system(f"docker cp {infra_dir}/common/scripts/run_trace_post_processing.sh {docker_container_name}:/usr/local/bin")
                os.system(f"docker cp {infra_dir}/common/scripts/gather_fp_pieces.py {docker_container_name}:/usr/local/bin")
                os.system(f"docker exec --privileged {docker_container_name} /bin/bash -c '/usr/local/bin/root_entrypoint.sh'")
                os.system(f"docker exec --privileged {docker_container_name} /bin/bash -c \"echo 0 | sudo tee /proc/sys/kernel/randomize_va_space\"")
                subprocess.run(["docker", "exec", "--privileged", "-it", f"--user={user}", f"--workdir=/home/{user}", docker_container_name, "/bin/bash"])
        except KeyboardInterrupt:
            if count_interactive_shells(docker_container_name, dbg_lvl) == 1:
                subprocess.run(["docker", "exec", "--privileged", f"--user={user}", f"--workdir=/home/{user}", docker_container_name,
                                "sed", "-i", "/source \\/usr\\/local\\/bin\\/user_entrypoint.sh/d", f"/home/{user}/.bashrc"], check=True, capture_output=True, text=True)
                os.system(f"docker rm -f {docker_container_name}")
                print("Recover the ASLR setting with sudo. Provide password..")
                os.system("echo 2 | sudo tee /proc/sys/kernel/randomize_va_space")
            return
        finally:
            try:
                if count_interactive_shells(docker_container_name, dbg_lvl) == 1:
                    subprocess.run(["docker", "exec", "--privileged", f"--user={user}", f"--workdir=/home/{user}", docker_container_name,
                                    "sed", "-i", "/source \\/usr\\/local\\/bin\\/user_entrypoint.sh/d", f"/home/{user}/.bashrc"], check=True, capture_output=True, text=True)
                    os.system("echo 2 | sudo tee /proc/sys/kernel/randomize_va_space")
                    client.containers.get(docker_container_name).remove(force=True)
                    print(f"Container {docker_container_name} removed.")
            except docker.errors.NotFound:
                print(f"Container {docker_container_name} not found.")
                raise
            except Exception as e:
                raise e
    except Exception as e:
        raise e


def run_trace_command(descriptor_path, action, dbg_lvl=2, infra_dir=None):
    if infra_dir is None:
        infra_dir = subprocess.check_output(["pwd"]).decode("utf-8").split("\n")[0]

    workload_db_path = f"{infra_dir}/workloads/workloads_db.json"

    user = subprocess.check_output("whoami").decode('utf-8').strip()
    info(f"User detected as {user}", dbg_lvl)

    descriptor_data = read_descriptor_from_json(descriptor_path, dbg_lvl)
    if descriptor_data is None:
        raise RuntimeError(f"Failed to read descriptor {descriptor_path}")

    workload_manager = descriptor_data.get("workload_manager")
    trace_name = descriptor_data.get("trace_name")
    traces = descriptor_data.get("trace_configurations") or []
    docker_image_list = get_image_list(traces)

    try:
        if action == "kill":
            if workload_manager == "manual":
                local_runner.kill_jobs(user, "trace", trace_name, docker_image_list, infra_dir, dbg_lvl)
            else:
                slurm_runner.kill_jobs(user, trace_name, docker_image_list, dbg_lvl)
            return 0

        if action == "info":
            if workload_manager == "manual":
                local_runner.print_status(user, trace_name, docker_image_list, dbg_lvl)
            else:
                slurm_runner.print_status(user, trace_name, docker_image_list, dbg_lvl)
            return 0

        if action == "launch":
            try:
                verify_descriptor(descriptor_data, workload_db_path, open_shell=True, dbg_lvl=dbg_lvl)
            except SystemExit as exc:
                raise RuntimeError("Descriptor verification failed") from exc
            open_interactive_shell(user, descriptor_data, infra_dir, dbg_lvl)
            return 0

        if action == "clean":
            remove_docker_containers(docker_image_list, trace_name, user, dbg_lvl)
            return 0

        try:
            verify_descriptor(descriptor_data, workload_db_path, open_shell=False, dbg_lvl=dbg_lvl)
        except SystemExit as exc:
            raise RuntimeError("Descriptor verification failed") from exc

        if workload_manager == "manual":
            local_runner.run_tracing(user, descriptor_data, workload_db_path, infra_dir, dbg_lvl)
        else:
            slurm_runner.run_tracing(user, descriptor_data, workload_db_path, infra_dir, dbg_lvl)
        return 0
    except Exception as exc:
        raise exc


def main():
    parser = argparse.ArgumentParser(description='Runs clustering/tracing on local or a slurm network')

    parser.add_argument('-d','--descriptor_name', required=True, help='Tracing descriptor name. Usage: -d trace.json')
    parser.add_argument('-k','--kill', required=False, default=False, action=argparse.BooleanOptionalAction, help='Don\'t launch jobs from descriptor, kill running jobs as described in descriptor')
    parser.add_argument('-i','--info', required=False, default=False, action=argparse.BooleanOptionalAction, help='Get info about all nodes and if they have containers for slurm workloads')
    parser.add_argument('-l','--launch', required=False, default=False, action=argparse.BooleanOptionalAction, help='Launch a docker container on a node for the purpose of development/debugging where the environment is for the experiment described in a descriptor.')
    parser.add_argument('-c','--clean', required=False, default=False, action=argparse.BooleanOptionalAction, help='Clean up all the docker containers related to an experiment')
    parser.add_argument('-dbg','--debug', required=False, type=int, default=2, help='1 for errors, 2 for warnings, 3 for info')
    parser.add_argument('-si','--scarab_infra', required=False, default=None, help='Path to scarab infra repo to launch new containers')

    args = parser.parse_args()
    descriptor_path = args.descriptor_name
    dbg_lvl = args.debug
    infra_dir = args.scarab_infra

    action = "trace"
    if args.kill:
        action = "kill"
    elif args.info:
        action = "info"
    elif args.launch:
        action = "launch"
    elif args.clean:
        action = "clean"

    return run_trace_command(descriptor_path, action, dbg_lvl=dbg_lvl, infra_dir=infra_dir)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as exc:
        err(str(exc), 1)
        sys.exit(1)
