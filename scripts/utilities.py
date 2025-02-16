#!/usr/bin/python3

# 01/27/2025 Surim Oh | utilities.py

import json
import os
import subprocess
import re
import docker

client = docker.from_env()

# Print an error message if on right debugging level
def err(msg: str, level: int):
    if level >= 1:
        print("ERR:", msg)

# Print warning message if on right debugging level
def warn(msg: str, level: int):
    if level >= 2:
        print("WARN:", msg)

# Print info message if on right debugging level
def info(msg: str, level: int):
    if level >= 3:
        print("INFO:", msg)

# json descriptor reader
def read_descriptor_from_json(filename="experiment.json", dbg_lvl = 1):
    # Read the descriptor data from a JSON file
    try:
        with open(filename, 'r') as json_file:
            descriptor_data = json.load(json_file)
        return descriptor_data
    except FileNotFoundError:
        err(f"File '{filename}' not found.", dbg_lvl)
        return None
    except json.JSONDecodeError as e:
        err(f"Error decoding JSON in file '{filename}': {e}", dbg_lvl)
        return None

def validate_simulation(workloads_data, suite_data, simulations, dbg_lvl = 2):
    for simulation in simulations:
        suite = simulation["suite"]
        subsuite = simulation["subsuite"]
        workload = simulation["workload"]
        cluster_id = simulation["cluster_id"]
        sim_mode = simulation["simulation_type"]

        if suite == None:
            err(f"Suite field cannot be null.", dbg_lvl)
            exit(1)

        if suite not in suite_data.keys():
            err(f"Suite '{suite}' is not valid.", dbg_lvl)
            exit(1)

        if subsuite != None and subsuite not in suite_data[suite].keys():
            err(f"Subsuite '{subsuite}' is not valid in Suite '{suite}'.", dbg_lvl);
            exit(1)

        if workload == None and (cluster_id != None or sim_mode != None):
            err(f"If you want to run all the workloads within '{suite}', empty all 'workload', 'cluster_id', 'simulation_type'.", dbg_lvl)
            exit(1)

        if workload != None and workload not in workloads_data.keys():
            err(f"Workload '{workload}' is not valid.", dbg_lvl)
            exit(1)

        if workload != None and sim_mode not in workloads_data[workload]["simulation"].keys():
            err(f"Simulation mode '{sim_mode}' is not an valid option for workload '{workload}'.", dbg_lvl)
            exit(1)

        if workload != None and cluster_id == None and "simpoints" not in workloads_data[workload].keys():
            err(f"Simpoints are not available. Choose '0' for cluster id.", dbg_lvl)
            exit(1)

        if workload != None and cluster_id != None and cluster_id > 0:
            found = False
            for simpoint in workloads_data[workload]["simpoints"]:
                if cluster_id == simpoint["cluster_id"]:
                    found = True
                    break
            if not found:
                err(f"Cluster ID {cluster_id} is not valid for workload '{workload}'.", dbg_lvl)
                exit(1)
        print(f"[{suite}, {subsuite}, {workload}, {cluster_id}, {sim_mode}] is a valid simulation option.")


# Verify the given descriptor file
def verify_descriptor(descriptor_data, workloads_data, suite_data, open_shell = False, dbg_lvl = 2):
    ## Check if the provided json describes all the valid data

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
        err("Need path to simpoints/traces. Set in descriptor file under 'simpoint_traces_dir'", dbg_lvl)
        exit(1)
    elif not os.path.exists(descriptor_data["simpoint_traces_dir"]):
        err(f"{descriptor_data['simpoint_traces_dir']} does not exist.", dbg_lvl)
        exit(1)

    # Check if configurations are provided
    if descriptor_data["configurations"] == None:
        error("Need configurations to simulate. Set in descriptor file under 'configurations'", dbg_lvl)
        exit(1)

# copy_scarab deprecated
# new API prepare_simulation
# Copies specified scarab binary, parameters, and launch scripts
# Inputs:   user        - username
#           scarab_path - Path to the scarab repository on host
#           docker_home - Path to the directory on host to be mount to the docker container home
#           experiment_name - Name of the current experiment
#           architecture - Architecture name
#
# Outputs:  scarab githash
def prepare_simulation(user, scarab_path, docker_home, experiment_name, architecture, dbg_lvl=1):
    ## Copy required scarab files into the experiment folder
    try:
        local_uid = os.getuid()
        local_gid = os.getgid()

        scarab_githash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=scarab_path).decode("utf-8").strip()
        info(f"Scarab git hash: {scarab_githash}", dbg_lvl)

        # If scarab binary does not exist in the provided scarab path, build the binary first.
        scarab_bin = f"{scarab_path}/src/build/opt/scarab"
        if not os.path.isfile(scarab_bin):
            info(f"Scarab binary not found at '{scarab_bin}', build it first...", dbg_lvl)
            os.system(f"docker run --rm \
                    --mount type=bind,source={scarab_path}:/scarab \
                    /bin/bash -c \"cd /scarab/src && make clean && make && chown -R {local_uid}:{local_gid} /scarab\"")

        experiment_dir = f"{docker_home}/simulations/{experiment_name}"
        os.system(f"mkdir -p {experiment_dir}/logs/")

        # Copy binary and architectural params to scarab/src
        arch_params = f"{scarab_path}/src/PARAMS.{architecture}"
        os.system(f"mkdir -p {experiment_dir}/scarab/src/")
        os.system(f"cp {scarab_bin} {experiment_dir}/scarab/src/scarab")
        try:
            os.symlink(f"{experiment_dir}/scarab/src/scarab", f"{experiment_dir}/scarab/src/scarab_{scarab_githash}")
        except FileExistsError:
            pass
        os.system(f"cp {arch_params} {experiment_dir}/scarab/src")

        # Required for non mode 4. Copy launch scripts from the docker container's scarab repo.
        # NOTE: Could cause issues if a copied version of scarab is incompatible with the version of
        # the launch scripts in the docker container's repo
        os.system(f"mkdir -p {experiment_dir}/scarab/bin/scarab_globals")
        os.system(f"cp {scarab_path}/bin/scarab_launch.py  {experiment_dir}/scarab/bin/scarab_launch.py ")
        os.system(f"cp {scarab_path}/bin/scarab_globals/*  {experiment_dir}/scarab/bin/scarab_globals/ ")

        # os.system(f"chmod -R 777 {experiment_dir}")

        return scarab_githash
    except Exception as e:
        raise e

def finish_simulation(user, experiment_dir):
    try:
        print("Finish simulation..")
        # TODO: do some cleanup or sanity check
        # os.system(f"chmod -R 755 {experiment_dir}")
    except Exception as e:
        raise e

# Generate command to do a single run of scarab
def generate_single_scarab_run_command(user, workload, group, experiment, config_key, config,
                                       mode, arch, scarab_githash, cluster_id,
                                       trim_type, modules_dir, trace_file,
                                       env_vars, bincmd, client_bincmd):
    if mode == "memtrace":
        command = f"run_memtrace_single_simpoint.sh \"{workload}\" \"{group}\" \"/home/{user}/simulations/{experiment}/{config_key}\" \"{config}\" \"{arch}\" \"{trim_type}\" /home/{user}/simulations/{experiment}/scarab {cluster_id} {modules_dir} {trace_file}"
    elif mode == "pt":
        command = f"run_pt_single_simpoint.sh \"{workload}\" \"{group}\" \"/home/{user}/simulations/{experiment}/{config_key}\" \"{config}\" \"{arch}\" \"{trim_type}\" /home/{user}/simulations/{experiment}/scarab {cluster_id}"
    elif mode == "exec":
        command = f"run_exec_single_simpoint.sh \"{workload}\" \"{group}\" \"/home/{user}/simulations/{experiment}/{config_key}\" \"{config}\" \"{arch}\" /home/{user}/simulations/{experiment}/scarab {env_vars} {bincmd} {client_bincmd}"
    else:
        command = ""

    return command

def write_docker_command_to_file_run_by_root(user, local_uid, local_gid, workload, experiment_name,
                                             docker_prefix, docker_container_name, simpoint_traces_dir,
                                             docker_home, githash, config_key, config, scarab_mode, scarab_githash,
                                             architecture, cluster_id, trim_type, modules_dir, trace_file,
                                             env_vars, bincmd, client_bincmd, filename):
    try:
        scarab_cmd = generate_single_scarab_run_command(user, workload, docker_prefix, experiment_name, config_key, config,
                                                        scarab_mode, architecture, scarab_githash, cluster_id,
                                                        trim_type, modules_dir, trace_file, env_vars, bincmd, client_bincmd)
        with open(filename, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(f"echo \"Running {config_key} {workload} {cluster_id}\"\n")
            f.write("echo \"Running on $(uname -n)\"\n")
            f.write(f"docker run --rm \
            -e user_id={local_uid} \
            -e group_id={local_gid} \
            -e username={user} \
            -e HOME=/home/{user} \
            --name {docker_container_name} \
            --mount type=bind,source={simpoint_traces_dir},target=/simpoint_traces,readonly \
            --mount type=bind,source={docker_home},target=/home/{user} \
            {docker_prefix}:{githash} \
            /bin/bash {scarab_cmd}\n")
    except Exception as e:
        raise e

def write_docker_command_to_file(user, local_uid, local_gid, workload, experiment_name,
                                 docker_prefix, docker_container_name, simpoint_traces_dir,
                                 docker_home, githash, config_key, config, scarab_mode, scarab_githash,
                                 architecture, cluster_id, trim_type, modules_dir, trace_file,
                                 env_vars, bincmd, client_bincmd, filename):
    try:
        scarab_cmd = generate_single_scarab_run_command(user, workload, docker_prefix, experiment_name, config_key, config,
                                                        scarab_mode, architecture, scarab_githash, cluster_id,
                                                        trim_type, modules_dir, trace_file, env_vars, bincmd, client_bincmd)
        with open(filename, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(f"echo \"Running {config_key} {workload} {cluster_id}\"\n")
            f.write("echo \"Running on $(uname -n)\"\n")
            f.write(f"docker run \
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
            /bin/bash\n")
            f.write(f"docker exec {docker_container_name} /bin/bash -c '/usr/local/bin/common_entrypoint.sh'\n")
            f.write(f"docker exec --user={user} {docker_container_name} /bin/bash {scarab_cmd}\n")
            f.write(f"docker rm -f {docker_container_name}\n")
    except Exception as e:
        raise e

def get_simpoints (workload_data, dbg_lvl = 2):
    simpoints = {}
    for simpoint in workload_data["simpoints"]:
        simpoints[f"{simpoint['cluster_id']}"] = simpoint["weight"]

    return simpoints

# Get workload simpoint ids and their associated weights
def get_simpoints_from_simpoints_file (simpoint_traces_dir, workload, dbg_lvl = 2):
    read_simp_weight_command = f"cat /{simpoint_traces_dir}/{workload}/simpoints/opt.w.lpt0.99"
    read_simp_simpid_command = f"cat /{simpoint_traces_dir}/{workload}/simpoints/opt.p.lpt0.99"

    info(f"Executing '{read_simp_weight_command}'", dbg_lvl)
    weight_out = subprocess.check_output(read_simp_weight_command.split(" ")).decode("utf-8").split("\n")[:-1]

    info(f"Executing '{read_simp_simpid_command}'", dbg_lvl)
    simpid_out = subprocess.check_output(read_simp_simpid_command.split(" ")).decode("utf-8").split("\n")[:-1]

    # Make lut for the weight for each 'index' id
    weights = {}
    for weight_id in weight_out:
        weight, id = weight_id.split(" ")
        weights[id] = float(weight)

    # Make final dictionary associated each simpoint id to its weight
    simpoints = {}
    for simpid_id in simpid_out:
        simpid, id = simpid_id.split(" ")
        simpoints[int(simpid)] = weights[id]

    return simpoints

def get_image_name(workloads_data, suite_data, simulation):
    suite = simulation["suite"]
    subsuite = simulation["subsuite"]
    workload = simulation["workload"]
    cluster_id = simulation["cluster_id"]
    sim_mode = simulation["simulation_type"]

    if workload != None:
        return workloads_data[workload]["simulation"][sim_mode]["image_name"]

    if subsuite != None:
        workload = next(iter(suite_data[suite][subsuite]["predefined_simulation_mode"]))
        sim_mode = suite_data[suite][subsuite]["predefined_simulation_mode"][workload]
    else:
        subsuite = next(iter(suite_data[suite]))
        workload = next(iter(suite_data[suite][subsuite]["predefined_simulation_mode"]))
        sim_mode = suite_data[suite][subsuite]["predefined_simulation_mode"][workload]

    return workloads_data[workload]["simulation"][sim_mode]["image_name"]

def open_interactive_shell(user, descriptor_data, workloads_data, suite_data, dbg_lvl = 1):
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
        docker_prefix = get_image_name(workloads_data, suite_data, descriptor_data['simulations'], dbg_lvl)

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
                # f.write(f"docker start {docker_container_name}\n")
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

def remove_docker_containers(docker_prefix_list, experiment_name, user, dbg_lvl):
    try:
        for docker_prefix in docker_prefix_list:
            pattern = re.compile(fr"^{docker_prefix}_.*_{experiment_name}.*_.*_{user}$")
            dockers = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True, check=True)
            lines = dockers.stdout.strip().split("\n") if dockers.stdout else []
            matching_containers = [line for line in lines if pattern.match(line)]

            if matching_containers:
                for container in matching_containers:
                    subprocess.run(["docker", "rm", "-f", container], check=True)
                    info(f"Removed container: {container}", dbg_lvl)
            else:
                info("No containers found.", dbg_lvl)
    except subprocess.CalledProcessError as e:
        err(f"Error while removing containers: {e}")
        raise e

def get_image_list(simulations, workloads_data, suite_data):
    image_list = []
    for simulation in simulations:
        suite = simulation["suite"]
        subsuite = simulation["subsuite"]
        workload = simulation["workload"]
        exp_cluster_id = simulation["cluster_id"]
        mode = simulation["simulation_type"]

        if workload == None and exp_cluster_id == None and mode == None:
            if subsuite == None:
                for subsuite in suite_data[suite].keys():
                    for workload in suite_data[suite][subsuite]["predefined_simulation_mode"].keys():
                        mode = suite_data[suite][subsuite]["predefined_simulation_mode"][workload]
                        if mode in workloads_data[workload]["simulation"].keys() and workloads_data[workload]["simulation"][mode]["image_name"] not in image_list:
                            image_list.append(workloads_data[workload]["simulation"][mode]["image_name"])
            else:
                for workload in suite_data[suite][subsuite]["predefined_simulation_mode"].keys():
                    mode = suite_data[suite][subsuite]["predefined_simulation_mode"][workload]
                    if mode in workloads_data[workload]["simulation"].keys() and workloads_data[workload]["simulation"][mode]["image_name"] not in image_list:
                        image_list.append(workloads_data[workload]["simulation"][mode]["image_name"])
        else:
            if mode in workloads_data[workload]["simulation"].keys() and workloads_data[workload]["simulation"][mode]["image_name"] not in image_list:
                image_list.append(workloads_data[workload]["simulation"][mode]["image_name"])

    return image_list

def get_docker_prefix(sim_mode, simulation_data):
    if sim_mode not in simulation_data.keys():
        err(f"{sim_mode} is not a valid simulation type.")
        exit(1)
    return simulation_data[sim_mode]["image_name"]

def get_weight_by_cluster_id(exp_cluster_id, simpoints):
    for simpoint in simpoints:
        if simpoint["cluster_id"] == exp_cluster_id:
            return simpoint["weight"]
