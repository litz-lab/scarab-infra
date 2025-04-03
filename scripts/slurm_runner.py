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
        finish_simulation,
        get_image_list,
        get_docker_prefix,
        prepare_trace,
        finish_trace,
        write_trace_docker_command_to_file,
        get_weight_by_cluster_id
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
            if image == [] or image == b'':
                info(f"Couldn't find image {docker_prefix}:{githash} on {node}", dbg_lvl)
                subprocess.check_output(["srun", f"--nodelist={node}", "docker", "pull", f"ghcr.io/litz-lab/scarab-infra/{docker_prefix}:{githash}"])
                image = subprocess.check_output(["srun", f"--nodelist={node}", "docker", "images", "-q", f"ghcr.io/litz-lab/scarab-infra/{docker_prefix}:{githash}"])
                if image != []:
                    subprocess.check_output(["srun", f"--nodelist={node}", "docker", "tag", f"ghcr.io/litz-lab/scarab-infra/{docker_prefix}:{githash}", f"{docker_prefix}:{githash}"])
                    subprocess.check_output(["srun", f"--nodelist={node}", "docker", "rmi", f"ghcr.io/litz-lab/scarab-infra/{docker_prefix}:{githash}"])
                    available_nodes.append(node)
                    continue

                # build the image if a pre-built image is not found
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
def print_status(user, job_name, docker_prefix_list, dbg_lvl = 1):
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

    for docker_prefix in docker_prefix_list:
        print(f"\nChecking what nodes have {docker_prefix}:{githash} image:")
        available_slurm_nodes = check_docker_image(available_slurm_nodes, docker_prefix, githash, dbg_lvl)
        for node in all_nodes:
            if node in available_slurm_nodes:
                print(f"\033[92mAVAILABLE:   {node}\033[0m")
            else:
                print(f"\033[31mUNAVAILABLE: {node}\033[0m")

    print(f"\nChecking what nodes have a running container with name {docker_prefix}_*_{job_name}_*_*_{user}")
    node_docker_running = check_docker_container_running(available_slurm_nodes, docker_prefix_list, job_name, user, dbg_lvl)

    for node in all_nodes:
        if node in node_docker_running.keys() and len(node_docker_running.get(node)) > 0:
            print(f"\033[92mRUNNING:     {node}\033[0m")
            for docker in node_docker_running[node]:
                print(f"\033[92m    CONTAINER: {docker}\033[0m")
        else:
            print(f"\033[31mNOT RUNNING: {node}\033[0m")


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

def run_simulation(user, descriptor_data, workloads_data, suite_data, infra_dir, dbg_lvl = 1):
    architecture = descriptor_data["architecture"]
    experiment_name = descriptor_data["experiment"]
    docker_home = descriptor_data["root_dir"]
    scarab_path = descriptor_data["scarab_path"]
    scarab_build = descriptor_data["scarab_build"]
    traces_dir = descriptor_data["traces_dir"]
    configs = descriptor_data["configurations"]
    simulations = descriptor_data["simulations"]

    docker_prefix_list = get_image_list(simulations, workloads_data, suite_data)

    def run_single_workload(workload, exp_cluster_id, sim_mode):
        try:
            docker_prefix = get_docker_prefix(sim_mode, workloads_data[workload]["simulation"])
            info(f"Using docker image with name {docker_prefix}:{githash}", dbg_lvl)
            docker_running = check_docker_image(all_nodes, docker_prefix, githash, dbg_lvl)
            excludes = set(all_nodes) - set(docker_running)
            info(f"Excluding following nodes: {', '.join(excludes)}", dbg_lvl)
            sbatch_cmd = generate_sbatch_command(excludes, experiment_dir)
            trim_type = None
            trace_file = None
            env_vars = ""
            bincmd = ""
            client_bincmd = ""
            seg_size = None
            simulation_data = workloads_data[workload]["simulation"][sim_mode]
            if sim_mode == "memtrace":
                trim_type = simulation_data["trim_type"]
                if trim_type == 0:
                    trace_file = simulation_data["whole_trace_file"]
                seg_size = simulation_data["segment_size"]
            if sim_mode == "exec":
                env_vars = simulation_data["env_vars"]
                bincmd = simulation_data["binary_cmd"]
                client_bincmd = simulation_data["client_bincmd"]
                seg_size = simulation_data["segment_size"]

            if "simpoints" not in workloads_data[workload].keys():
                weight = 1
                simpoints = {}
                simpoints["0"] = weight
            elif exp_cluster_id == None:
                simpoints = get_simpoints(workloads_data[workload], sim_mode, dbg_lvl)
            elif exp_cluster_id > 0:
                weight = get_weight_by_cluster_id(exp_cluster_id, workloads_data[workload]["simpoints"])
                simpoints = {}
                simpoints[exp_cluster_id] = weight

            for config_key in configs:
                config = configs[config_key]

                for cluster_id, weight in simpoints.items():
                    info(f"cluster_id: {cluster_id}, weight: {weight}", dbg_lvl)

                    docker_container_name = f"{docker_prefix}_{workload}_{experiment_name}_{config_key.replace("/", "-")}_{cluster_id}_{sim_mode}_{user}"

                    # TODO: Notification when a run fails, point to output file and command that caused failure
                    # Add help (?)
                    # Look into squeue -o https://slurm.schedmd.com/squeue.html
                    # Look into resource allocation

                    # TODO: Rewrite with sbatch arrays

                    # Create temp file with run command and run it
                    filename = f"{docker_container_name}_tmp_run.sh"
                    write_docker_command_to_file(user, local_uid, local_gid, workload, experiment_name,
                                                 docker_prefix, docker_container_name, traces_dir,
                                                 docker_home, githash, config_key, config, sim_mode, scarab_githash,
                                                 seg_size, architecture, cluster_id, trim_type, trace_file,
                                                 env_vars, bincmd, client_bincmd, filename, infra_dir)
                    tmp_files.append(filename)

                    os.system(sbatch_cmd + filename)
                    info(f"Running sbatch command '{sbatch_cmd + filename}'", dbg_lvl)
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
            err("Error: 'git' command not found. Make sure Git is installed and in your PATH.")
        except subprocess.CalledProcessError:
            err("Error: Not in a Git repository or unable to retrieve Git hash.")


        # Get avlailable nodes. Error if none available
        available_slurm_nodes, all_nodes = check_available_nodes(dbg_lvl)
        info(f"Available nodes: {', '.join(available_slurm_nodes)}", dbg_lvl)

        if available_slurm_nodes == []:
            err("Cannot find any running slurm nodes", dbg_lvl)
            exit(1)

        for docker_prefix in docker_prefix_list:
            docker_running = prepare_docker_image(available_slurm_nodes, docker_prefix, githash, dbg_lvl)
            # If docker image still does not exist, exit
            if docker_running == []:
                err(f"Error with preparing docker image for {docker_prefix}:{githash}", dbg_lvl)
                exit(1)


        # Generate commands for executing in users docker and sbatching to nodes with containers
        experiment_dir = f"{descriptor_data['root_dir']}/simulations/{experiment_name}"
        docker_prefix = docker_prefix_list[0]
        scarab_githash = prepare_simulation(user, scarab_path, scarab_build, descriptor_data['root_dir'], experiment_name, architecture, docker_prefix, githash, infra_dir, False, dbg_lvl)

        # Iterate over each workload and config combo
        tmp_files = []
        for simulation in simulations:
            suite = simulation["suite"]
            subsuite = simulation["subsuite"]
            workload = simulation["workload"]
            exp_cluster_id = simulation["cluster_id"]
            sim_mode = simulation["simulation_type"]

            # Run all the workloads within suite
            if workload == None and subsuite == None:
                for subsuite_ in suite_data[suite].keys():
                    for workload_ in suite_data[suite][subsuite_]["predefined_simulation_mode"].keys():
                        sim_mode_ = sim_mode
                        if sim_mode_ == None:
                            sim_mode_ = suite_data[suite][subsuite_]["predefined_simulation_mode"][workload_]
                        run_single_workload(workload_, exp_cluster_id, sim_mode_)
            elif workload == None and subsuite != None:
                for workload_ in suite_data[suite][subsuite]["predefined_simulation_mode"].keys():
                    sim_mode_ = sim_mode
                    if sim_mode_ == None:
                        sim_mode_ = suite_data[suite][subsuite]["predefined_simulation_mode"][workload_]
                    run_single_workload(workload_, exp_cluster_id, sim_mode_)
            else:
                sim_mode_ = sim_mode
                if sim_mode_ == None:
                    sim_mode_ = suite_data[suite][subsuite]["predefined_simulation_mode"][workload]
                run_single_workload(workload, exp_cluster_id, sim_mode_)

        # Clean up temp files
        for tmp in tmp_files:
            info(f"Removing temporary run script {tmp}", dbg_lvl)
            os.remove(tmp)

        finish_simulation(user, docker_home)

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

def run_tracing(user, descriptor_data, workload_db_path, suite_db_path, infra_dir, dbg_lvl = 2):
    trace_name = descriptor_data["trace_name"]
    docker_home = descriptor_data["root_dir"]
    scarab_path = descriptor_data["scarab_path"]
    scarab_build = descriptor_data["scarab_build"]
    traces_dir = descriptor_data["traces_dir"]
    trace_configs = descriptor_data["trace_configurations"]

    docker_prefix_list = []
    for config in trace_configs:
        image_name = config["image_name"]
        if image_name not in docker_prefix_list:
            docker_prefix_list.append(image_name)

    tmp_files = []

    def run_single_trace(workload, image_name, trace_name, env_vars, binary_cmd, client_bincmd, trace_type, drio_args, clustering_k):
        try:
            docker_running = check_docker_image(all_nodes, image_name, githash, dbg_lvl)
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
                                               clustering_k, filename, infra_dir)
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
            err("Error: 'git' command not found. Make sure Git is installed and in your PATH.")
        except subprocess.CalledProcessError:
            err("Error: Not in a Git repository or unable to retrieve Git hash.")


        # Get avlailable nodes. Error if none available
        available_slurm_nodes, all_nodes = check_available_nodes(dbg_lvl)
        info(f"Available nodes: {', '.join(available_slurm_nodes)}", dbg_lvl)

        if available_slurm_nodes == []:
            err("Cannot find any running slurm nodes", dbg_lvl)
            exit(1)

        for docker_prefix in docker_prefix_list:
            docker_running = prepare_docker_image(available_slurm_nodes, docker_prefix, githash, dbg_lvl)
            # If docker image still does not exist, exit
            if docker_running == []:
                err(f"Error with preparing docker image for {docker_prefix}:{githash}", dbg_lvl)
                exit(1)

        trace_dir = f"{descriptor_data['root_dir']}/simpoint_flow/{trace_name}"
        docker_prefix = docker_prefix_list[0]
        prepare_trace(user, scarab_path, scarab_build, docker_home, trace_name, infra_dir, docker_prefix, githash, False, dbg_lvl)

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
                             trace_type, drio_args, clustering_k)

        # Clean up temp files
        for tmp in tmp_files:
            info(f"Removing temporary run script {tmp}", dbg_lvl)
            os.remove(tmp)

        finish_trace(user, descriptor_data, workload_db_path, suite_db_path, dbg_lvl)
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
