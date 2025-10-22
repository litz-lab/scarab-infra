#!/usr/bin/python3

# 01/27/2025 | Surim Oh | run_simulation.py
# An entrypoint script that performs Scarab runs on Slurm clusters using docker containers or on local

import subprocess
import argparse
import os
import sys
from pathlib import Path
import docker

from .utilities import (
    info,
    err,
    read_descriptor_from_json,
    remove_docker_containers,
    remove_tmp_run_scripts,
    get_image_list,
    prepare_simulation,
    get_image_name,
    validate_simulation,
    is_container_running,
    count_interactive_shells
)
from . import slurm_runner, local_runner

client = docker.from_env()

# Verify the given descriptor file
def verify_descriptor(descriptor_data, workloads_data, open_shell = False, dbg_lvl = 2):
    ## Check if the provided json describes all the valid data

    # Check the descriptor type
    if not descriptor_data["descriptor_type"]:
        err("Descriptor type must be 'simulation' for a simulation descriptor", dbg_lvl)
        exit(1)

    # Check the scarab path
    if descriptor_data["scarab_path"] == None:
        err("Need path to scarab path. Set in descriptor file under 'scarab_path'", dbg_lvl)
        exit(1)

    # Check the scarab build mode
    if descriptor_data["scarab_build"] != None and descriptor_data["scarab_build"] != 'opt' and descriptor_data["scarab_build"] != 'dbg':
        err("Need a valid scarab build mode ('opt' or 'dbg' or null). Set in descriptor file under 'scarab_build'", dbg_lvl)
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
        print(f"Experiment '{experiment_dir}' already exists. It will overwrite the existing simulation results!")

    # Check if each simulation type is valid
    validate_simulation(workloads_data, descriptor_data['simulations'])

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
    if descriptor_data["traces_dir"] == None:
        err("Need path to simpoints and traces. Set in descriptor file under 'traces_dir'", dbg_lvl)
        exit(1)
    elif not os.path.exists(descriptor_data["traces_dir"]):
        err(f"{descriptor_data['traces_dir']} does not exist.", dbg_lvl)
        exit(1)

    # Check if configurations are provided
    if descriptor_data["configurations"] == None:
        err("Need configurations to simulate. Set in descriptor file under 'configurations'", dbg_lvl)
        exit(1)

def open_interactive_shell(user, descriptor_data, workloads_data, infra_dir, dbg_lvl = 1):
    experiment_name = descriptor_data["experiment"]
    scarab_path = descriptor_data["scarab_path"]
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

        docker_prefix = get_image_name(workloads_data, descriptor_data['simulations'][0])
        docker_prefix_list = [docker_prefix]
        docker_home = descriptor_data['root_dir']
        experiment_workdir = None
        experiment_workdir_host = Path(docker_home) / "simulations" / experiment_name
        try:
            experiment_workdir_host.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        if experiment_workdir_host.is_dir():
            experiment_workdir = f"/home/{user}/simulations/{experiment_name}"
        default_workdir = f"/home/{user}"

        # Set the env for simulation again (already set in Dockerfile.common) in case user's bashrc overwrite the existing ones when the home directory is mounted
        bashrc_path = f"{docker_home}/.bashrc"
        entry = "source /usr/local/bin/user_entrypoint.sh"
        with open(bashrc_path, "a+") as f:
            f.seek(0)
            if entry not in f.read():
                f.write(f"\n{entry}\n")

        # Generate commands for executing in users docker and sbatching to nodes with containers
        scarab_githash, image_tag_list = prepare_simulation(user,
                                            scarab_path,
                                            descriptor_data['scarab_build'],
                                            docker_home,
                                            experiment_name,
                                            descriptor_data['architecture'],
                                            docker_prefix_list,
                                            githash,
                                            infra_dir,
                                            ["scarab_current"],
                                            interactive_shell=True,
                                            available_slurm_nodes=[],
                                            dbg_lvl=dbg_lvl)
        workload = descriptor_data['simulations'][0]['workload']
        mode = descriptor_data['simulations'][0]['simulation_type']

        docker_container_name = f"{docker_prefix}_{experiment_name}_scarab_{scarab_githash}_{user}"
        traces_dir = descriptor_data["traces_dir"]
        docker_home = descriptor_data["root_dir"]
        try:
            # If the container is already running, log into it by openning another interactive shell
            if is_container_running(docker_container_name, dbg_lvl):
                workdir = experiment_workdir if experiment_workdir else default_workdir
                subprocess.run(["docker", "exec", "--privileged", "-it", f"--user={user}", f"--workdir={workdir}", docker_container_name, "/bin/bash"])
            else:
                info(f"Create a new container for the interactive mode", dbg_lvl)
                subprocess.run(["docker", "run",
                                "-e", f"user_id={local_uid}",
                                "-e", f"group_id={local_gid}",
                                "-e", f"username={user}",
                                "-e", f"HOME=/home/{user}",
                                "-e", f"APP_GROUPNAME={docker_prefix}",
                                "-e", f"APPNAME={workload}",
                                "-dit", "--name", f"{docker_container_name}",
                                "--mount", f"type=bind,source={traces_dir},target=/simpoint_traces,readonly=true",
                                "--mount", f"type=bind,source={docker_home},target=/home/{user},readonly=false",
                                "--mount", f"type=bind,source={scarab_path},target=/scarab,readonly=false",
                                f"{docker_prefix}:{githash}", "/bin/bash"], check=True, capture_output=True, text=True)
                subprocess.run(["docker", "cp", f"{infra_dir}/scripts/utilities.sh", f"{docker_container_name}:/usr/local/bin"],
                               check=True, capture_output=True, text=True)
                subprocess.run(["docker", "cp", f"{infra_dir}/common/scripts/root_entrypoint.sh", f"{docker_container_name}:/usr/local/bin"],
                               check=True, capture_output=True, text=True)
                subprocess.run(["docker", "cp", f"{infra_dir}/common/scripts/user_entrypoint.sh", f"{docker_container_name}:/usr/local/bin"],
                               check=True, capture_output=True, text=True)
                if os.path.exists(f"{infra_dir}/workloads/{docker_prefix}/workload_root_entrypoint.sh"):
                    subprocess.run(["docker", "cp", f"{infra_dir}/workloads/{docker_prefix}/workload_root_entrypoint.sh", f"{docker_container_name}:/usr/local/bin"],
                                   check=True, capture_output=True, text=True)
                if os.path.exists(f"{infra_dir}/workloads/{docker_prefix}/workload_user_entrypoint.sh"):
                    subprocess.run(["docker", "cp", f"{infra_dir}/workloads/{docker_prefix}/workload_user_entrypoint.sh", f"{docker_container_name}:/usr/local/bin"],
                                   check=True, capture_output=True, text=True)
                if mode == "memtrace":
                    subprocess.run(["docker", "cp", f"{infra_dir}/common/scripts/run_memtrace_single_simpoint.sh", f"{docker_container_name}:/usr/local_bin"],
                                   check=True, capture_output=True, text=True)
                elif mode == "pt":
                    subprocess.run(["docker", "cp", f"{infra_dir}/common/scripts/run_pt_single_simpoint.sh", f"{docker_container_name}:/usr/local/bin"],
                                   check=True, capture_output=True, text=True)
                elif mode == "exec":
                    subprocess.run(["docker", "cp", f"{infra_dir}/common/scripts/run_exec_single_simpoint.sh", f"{docker_container_name}:/usr/local/bin"],
                                   check=True, capture_output=True, text=True)
                subprocess.run(["docker", "exec", "--privileged", f"{docker_container_name}", "/bin/bash", "-c", "/usr/local/bin/root_entrypoint.sh"],
                               check=True, capture_output=True, text=True)
                workdir = experiment_workdir if experiment_workdir else default_workdir
                subprocess.run(["docker", "exec", "--privileged", "-it", f"--user={user}", f"--workdir={workdir}", docker_container_name, "/bin/bash"])
        except KeyboardInterrupt:
            if count_interactive_shells(docker_container_name, dbg_lvl) == 1:
                subprocess.run(["docker", "exec", "--privileged", f"--user={user}", f"--workdir=/home/{user}", docker_container_name,
                                "sed", "-i", "/source \\/usr\\/local\\/bin\\/user_entrypoint.sh/d", f"/home/{user}/.bashrc"], check=True, capture_output=True, text=True)
                subprocess.run(["docker", "rm", "-f", f"{docker_container_name}"], check=True, capture_output=True, text=True)
            return
        finally:
            try:
                if count_interactive_shells(docker_container_name, dbg_lvl) == 1:
                    subprocess.run(["docker", "exec", "--privileged", f"--user={user}", f"--workdir=/home/{user}", docker_container_name,
                                    "sed", "-i", "/source \\/usr\\/local\\/bin\\/user_entrypoint.sh/d", f"/home/{user}/.bashrc"], check=True, capture_output=True, text=True)
                    client.containers.get(docker_container_name).remove(force=True)
                    print(f"Container {docker_container_name} removed.")
            except docker.errors.NotFound as e:
                print(f"Container {docker_container_name} not found.")
                raise e
            except Exception as e:
                raise e
    except Exception as e:
        raise e


def run_simulation_command(descriptor_path, action, dbg_lvl=2, infra_dir=None):
    if infra_dir is None:
        infra_dir = subprocess.check_output(["pwd"]).decode("utf-8").split("\n")[0]

    user = subprocess.check_output("whoami").decode('utf-8').strip()
    info(f"User detected as {user}", dbg_lvl)

    descriptor_data = read_descriptor_from_json(descriptor_path, dbg_lvl)
    if descriptor_data is None:
        raise RuntimeError(f"Failed to read descriptor {descriptor_path}")

    workload_db_path = f"{infra_dir}/workloads/workloads_top_simp.json" if descriptor_data.get("top_simpoint") else f"{infra_dir}/workloads/workloads_db.json"
    workloads_data = read_descriptor_from_json(workload_db_path, dbg_lvl)
    if workloads_data is None:
        raise RuntimeError("Failed to read workloads database")

    workload_manager = descriptor_data.get("workload_manager")
    experiment_name = descriptor_data.get("experiment")
    simulations = descriptor_data.get("simulations") or []
    docker_image_list = get_image_list(simulations, workloads_data)

    try:
        if action == "kill":
            if workload_manager == "manual":
                local_runner.kill_jobs(user, "simulation", experiment_name, docker_image_list, infra_dir, dbg_lvl)
            else:
                slurm_runner.kill_jobs(user, experiment_name, docker_image_list, dbg_lvl)
            return 0

        if action == "info":
            if workload_manager == "manual":
                local_runner.print_status(user, experiment_name, docker_image_list, dbg_lvl)
            else:
                slurm_runner.print_status(user, experiment_name, docker_image_list, descriptor_data, workloads_data, dbg_lvl)
            return 0

        if action == "launch":
            try:
                verify_descriptor(descriptor_data, workloads_data, True, dbg_lvl)
            except SystemExit as exc:
                raise RuntimeError("Descriptor verification failed") from exc
            open_interactive_shell(user, descriptor_data, workloads_data, infra_dir, dbg_lvl)
            return 0

        if action == "clean":
            if workload_manager == "slurm":
                slurm_runner.clean_containers(user, experiment_name, docker_image_list, dbg_lvl)
                descriptor_root = Path(descriptor_data["root_dir"])
                experiment_dir = descriptor_root / "simulations" / experiment_name
                remove_tmp_run_scripts(Path(infra_dir), experiment_name, user, dbg_lvl)
                remove_tmp_run_scripts(experiment_dir, experiment_name, user, dbg_lvl)
            remove_docker_containers(docker_image_list, experiment_name, user, dbg_lvl)
            return 0

        # default: run simulation
        try:
            verify_descriptor(descriptor_data, workloads_data, False, dbg_lvl)
        except SystemExit as exc:
            raise RuntimeError("Descriptor verification failed") from exc

        if workload_manager == "manual":
            local_runner.run_simulation(user, descriptor_data, workloads_data, infra_dir, descriptor_path, dbg_lvl)
        else:
            slurm_runner.run_simulation(user, descriptor_data, workloads_data, infra_dir, descriptor_path, dbg_lvl)
        return 0
    except Exception as exc:
        raise exc


def main():
    parser = argparse.ArgumentParser(description='Runs scarab on local or a slurm network')

    parser.add_argument('-d','--descriptor_name', required=True, help='Experiment descriptor name. Usage: -d exp.json')
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

    action = "simulate"
    if args.kill:
        action = "kill"
    elif args.info:
        action = "info"
    elif args.launch:
        action = "launch"
    elif args.clean:
        action = "clean"

    return run_simulation_command(descriptor_path, action, dbg_lvl=dbg_lvl, infra_dir=infra_dir)

if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as exc:
        err(str(exc), 1)
        sys.exit(1)
