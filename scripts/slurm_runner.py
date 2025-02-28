#!/usr/bin/python3

# 10/7/2024 | Alexander Symons | run_slurm.py
# 01/27/2025 | Surim Oh | slurm_runner.py

import os
import random
import subprocess
import re
import traceback
from utilities import (
        err,
        warn,
        info,
        get_simpoints,
        write_docker_command_to_file,
        prepare_simulation,
        finish_simulation
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
            if image == []:
                info(f"Couldn't find image {docker_prefix}:{githash} on {node}", dbg_lvl)
                continue

            available_nodes.append(node)

        return available_nodes
    except Exception as e:
        raise e


# Prepare the docker image on each slurm node
# Inputs: list of available slurm nodes
# Output: list of nodes where the docker image is ready
def prepare_docker_image(nodes, docker_prefix, githash, dbg_lvl = 1):
    try:
        available_nodes = []
        for node in nodes:
            # Check if the image exists
            image = subprocess.check_output(["srun", f"--nodelist={node}", "docker", "images", "-q", f"{docker_prefix}:{githash}"])
            info(f"{image}", dbg_lvl)
            if image == []:
                info(f"Couldn't find image {docker_prefix}:{githash} on {node}", dbg_lvl)
                # TODO: user prebuilt image
                # build the image for now
                subprocess.check_output(["srun", f"--nodelist={node}", "./run.sh", "-b", docker_prefix])

                image = subprocess.check_output(["srun", f"--nodelist={node}", "docker", "images", "-q", f"{docker_prefix}:{githash}"])
                info(f"{image}", dbg_lvl)
                if image == []:
                    info(f"Still couldn't find image {docker_prefix}:{githash} on {node} after trying to build one", dbg_lvl)

                    continue

            available_nodes.append(node)

        return available_nodes
    except Exception as e:
        raise e

# Check if a container is running on the provided nodes, return those that are
# Inputs: list of nodes, docker_prefix, experiment_name, user
# Output: dictionary of node-containers
def check_docker_container_running(nodes, docker_prefix, experiment_name, user, dbg_lvl = 1):
    pattern = re.compile(fr"^{docker_prefix}_.*_{experiment_name}.*_.*_{user}$")
    try:
        running_nodes_dockers = {}
        for node in nodes:
            # Check container is running and no errors
            try:
                dockers = subprocess.run(["srun", f"--nodelist={node}", "docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True, check=True)
                lines = dockers.stdout.strip().split("\n") if dockers.stdout else []
                matching_containers = [line for line in lines if pattern.match(line)]
            except:
                err(f"Error while checking a running docker container named {docker_prefix}_.*_{experiment_name}_.*_.*_{user} on node {node}", dbg_lvl)

                continue

            running_nodes_dockers[node] = matching_containers
        return running_nodes_dockers
    except Exception as e:
        raise e

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
        response = subprocess.check_output(["sinfo", "-N"]).decode("utf-8")
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

# Get command to sbatch scarab runs. 1 core each, exclude nodes where container isn't running
def generate_sbatch_command(excludes, experiment_dir):
    # If all nodes are usable, no need to exclude
    if not excludes == set():
        return f"sbatch --exclude {','.join(excludes)} -c 1 -o {experiment_dir}/logs/job_%j.out "

    return f"sbatch -c 1 -o {experiment_dir}/logs/job_%j.out "

# Launch a docker container on one of the available nodes
# deprecated
def launch_docker(infra_dir, docker_home, available_nodes, node=None, dbg_lvl=1):
    try:
        # Get the path to the run script
        if infra_dir == ".": run_script = ""
        elif infra_dir[-1] == '/': run_script = infra_dir
        else: run_script = infra_dir + '/'

        # Check if run.sh script exists
        if not os.path.isfile(run_script + "run.sh"):
            err(f"Couldn't find file scarab infra run.sh at {run_script + 'run.sh'}. Check scarab_infra option", dbg_lvl)
            exit(1)

        # Get name of slurm node to spin up
        if node == None:
            spin_up_index = random.randint(0, len(available_nodes)-1)
            spin_up_node = available_nodes[spin_up_index]
        else:
            spin_up_node = node

        # Spin up docker container on that node
        print(f"Spinning up node {spin_up_node}")
        os.system(f"srun --nodelist={spin_up_node} -c 1 {run_script}run.sh -o {docker_home} -b 2")
    except Exception as e:
        raise

# Print info of docker/slurm nodes and running experiment
def print_status(user, experiment_name, docker_prefix, dbg_lvl = 1):
    # Get GitHash
    try:
        githash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("utf-8").strip()
        info(f"Git hash: {githash}", dbg_lvl)
    except FileNotFoundError:
        err("Error: 'git' command not found. Make sure Git is installed and in your PATH.")
    except subprocess.CalledProcessError:
        err("Error: Not in a Git repository or unable to retrieve Git hash.")

    info(f"Getting information about all nodes", dbg_lvl)
    available_slurm_nodes, all_nodes = check_available_nodes(dbg_lvl)

    print(f"Checking resource availability of slurm nodes:")
    for node in all_nodes:
        if node in available_slurm_nodes:
            print(f"\033[92mAVAILABLE:   {node}\033[0m")
        else:
            print(f"\033[31mUNAVAILABLE: {node}\033[0m")

    print(f"\nChecking what nodes have the corresponding image:")
    available_slurm_nodes = check_docker_image(available_slurm_nodes, docker_prefix, githash, dbg_lvl)
    for node in all_nodes:
        if node in available_slurm_nodes:
            print(f"\033[92mAVAILABLE:   {node}\033[0m")
        else:
            print(f"\033[31mUNAVAILABLE: {node}\033[0m")

    print(f"\nChecking what nodes have a running container with name {docker_prefix}_*_{experiment_name}_*_*_{user}")
    node_docker_running = check_docker_container_running(available_slurm_nodes, docker_prefix, experiment_name, user, dbg_lvl)

    for node in all_nodes:
        if node in node_docker_running.keys():
            print(f"\033[92mRUNNING:     {node}\033[0m")
            for docker in node_docker_running[node]:
                print(f"\033[92m    CONTAINER: {docker}\033[0m")
        else:
            print(f"\033[31mNOT RUNNING: {node}\033[0m")


# Kills all jobs for experiment_name, if associated with user
def kill_jobs(user, experiment_name, docker_prefix, dbg_lvl = 2):
    # Kill and exit if killing jobs
    info(f"Killing all slurm jobs associated with {experiment_name}", dbg_lvl)

    # Format is JobID Name
    response = subprocess.check_output(["squeue", "-u", user, "--Format=JobID,Name:90"]).decode("utf-8")
    lines = [r.split() for r in response.split('\n') if r != ''][1:]

    # Filter to entries assocaited with this experiment, and get job ids
    lines = list(filter(lambda x:experiment_name in x[1], lines))
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


        # Get avlailable nodes. Error if none available
        available_slurm_nodes, all_nodes = check_available_nodes(dbg_lvl)
        info(f"Available nodes: {', '.join(available_slurm_nodes)}", dbg_lvl)

        if available_slurm_nodes == []:
            err("Cannot find any running slurm nodes", dbg_lvl)
            exit(1)

        docker_running = prepare_docker_image(available_slurm_nodes, docker_prefix, githash, dbg_lvl)
        # If docker image still does not exist, exit
        if docker_running == []:
            err("Error with launched container. Could not detect it after launching", dbg_lvl)
            exit(1)

        info(f"Using docker image with name {docker_prefix}:{githash}", dbg_lvl)

        # Determine nodes without running containers. Launch a container if none found
        excludes = set(all_nodes) - set(docker_running)
        info(f"Excluding following nodes: {', '.join(excludes)}", dbg_lvl)

        # Generate commands for executing in users docker and sbatching to nodes with containers
        experiment_dir = f"{descriptor_data['root_dir']}/simulations/{experiment_name}"
        scarab_githash = prepare_simulation(user, scarab_path, descriptor_data['root_dir'], experiment_name, architecture, dbg_lvl)
        sbatch_cmd = generate_sbatch_command(excludes, experiment_dir)

        # Iterate over each workload and config combo
        tmp_files = []
        for workload in workloads:
            # Only needed for modes 3 and 4
            simpoints = get_simpoints(simpoint_traces_dir, workload, dbg_lvl)
            for config_key in configs:
                config = configs[config_key]

                for simpoint, weight in simpoints.items():
                    print(simpoint, weight)

                    docker_container_name = f"{docker_prefix}_{workload}_{experiment_name}_{config_key}_{simpoint}_{user}"

                    # TODO: Notification when a run fails, point to output file and command that caused failure
                    # Add help (?)
                    # Look into squeue -o https://slurm.schedmd.com/squeue.html
                    # Look into resource allocation

                    # TODO: Rewrite with sbatch arrays

                    # Create temp file with run command and run it
                    filename = f"{experiment_name}_{workload}_{config_key.replace("/", "-")}_{simpoint}_tmp_run.sh"
                    write_docker_command_to_file(user, local_uid, local_gid, workload, experiment_name,
                                                 docker_prefix, docker_container_name, simpoint_traces_dir,
                                                 docker_home, githash, config_key, config, scarab_mode, scarab_githash,
                                                 architecture, simpoint, filename)
                    tmp_files.append(filename)

                    os.system(sbatch_cmd + filename)
                    info(f"Running sbatch command '{sbatch_cmd + filename}'", dbg_lvl)

        # Clean up temp files
        for tmp in tmp_files:
            info(f"Removing temporary run script {tmp}", dbg_lvl)
            os.remove(tmp)

        finish_simulation(user, f"{docker_home}/simulations/{experiment_name}")

        # TODO: check resource capping policies, add kill/info options

        # TODO: (long term) add evalstats to json descriptor to run stats library with PMU counters
    except Exception as e:
        print("An exception occurred:", e)
        traceback.print_exec()  # Print the full stack trace
