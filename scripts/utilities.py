#!/usr/bin/python3

# 01/27/2025 Surim Oh | utilities.py

import json
import os
import subprocess
import re

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

# json descriptor writer
def write_json_descriptor(filename, descriptor_data, dbg_lvl = 1):
    # Write the descriptor data to a JSON file
    try:
        with open(filename, 'w') as json_file:
            json.dump(descriptor_data, file, indent=2, separators=(",", ":"))
    except TypeError as e:
            print(f"TypeError: {e}")
    except UnicodeEncodeError as e:
            print(f"UnicodeEncodeError: {e}")
    except OverflowError as e:
            print(f"OverflowError: {e}")
    except ValueError as e:
            print(f"ValueError: {e}")
    except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")

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
                                 env_vars, bincmd, client_bincmd, filename, infra_dir):
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
            f.write(f"docker cp {infra_dir}/scripts/utilities.sh {docker_container_name}:/usr/local/bin\n")
            f.write(f"docker cp {infra_dir}/common/scripts/common_entrypoint.sh {docker_container_name}:/usr/local/bin\n")
            if scarab_mode == "memtrace":
                f.write(f"docker cp {infra_dir}/common/scripts/run_memtrace_single_simpoint.sh {docker_container_name}:/usr/local/bin\n")
            elif scarab_mode == "pt":
                f.write(f"docker cp {infra_dir}/common/scripts/run_pt_single_simpoint.sh {docker_container_name}:/usr/local/bin\n")
            elif scarab_mode == "exec":
                f.write(f"docker cp {infra_dir}/common/scripts/run_exec_single_simpoint.sh {docker_container_name}:/usr/local/bin\n")
            f.write(f"docker exec {docker_container_name} /bin/bash -c '/usr/local/bin/common_entrypoint.sh'\n")
            f.write(f"docker exec --user={user} {docker_container_name} /bin/bash {scarab_cmd}\n")
            f.write(f"docker rm -f {docker_container_name}\n")
    except Exception as e:
        raise e

def generate_single_trace_run_command(user, workload, image_name, trace_name, binary_cmd, client_bincmd, simpoint_mode, drio_args, clustering_k):
    command = ""
    if simpoint_mode == "cluster_then_trace":
        mode = 1
    elif simpoint_mode == "trace_then_post_process":
        mode = 2
    command = f"run_simpoint_trace.sh \"{workload}\" \"{image_name}\" \"/home/{user}/simpoint_flow/{trace_name}\" \"{binary_cmd}\" \"{mode}\" \"{drio_args}\" \"{clustering_k}\""
    return command

def write_trace_docker_command_to_file(user, local_uid, local_gid, docker_container_name, githash,
                                       workload, image_name, trace_name, simpoint_traces_dir, docker_home,
                                       env_vars, binary_cmd, client_bincmd, simpoint_mode, drio_args,
                                       clustering_k, filename, infra_dir):
    try:
        trace_cmd = generate_single_trace_run_command(user, workload, image_name, trace_name, binary_cmd, client_bincmd,
                                                      simpoint_mode, drio_args, clustering_k)
        with open(filename, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(f"echo \"Tracing {workload}\"\n")
            f.write("echo \"Running on $(uname -n)\"\n")
            command = f"docker run --privileged \
                    -e user_id={local_uid} \
                    -e group_id={local_gid} \
                    -e username={user} \
                    -e HOME=/home/{user} \
                    -e APP_GROUPNAME={image_name} \
                    -e APPNAME={workload} "
            if env_vars:
                for env in env_vars:
                    command = command + f"-e {env} "
            command = command + f"-dit \
                    --name {docker_container_name} \
                    --mount type=bind,source={docker_home},target=/home/{user} \
                    {image_name}:{githash} \
                    /bin/bash\n"
            f.write(f"{command}")
            f.write(f"docker cp {infra_dir}/scripts/utilities.sh {docker_container_name}:/usr/local/bin\n")
            f.write(f"docker cp {infra_dir}/common/scripts/common_entrypoint.sh {docker_container_name}:/usr/local/bin\n")
            f.write(f"docker cp {infra_dir}/common/scripts/run_clustering.sh {docker_container_name}:/usr/local/bin\n")
            f.write(f"docker cp {infra_dir}/common/scripts/run_simpoint_trace.sh {docker_container_name}:/usr/local/bin\n")
            f.write(f"docker cp {infra_dir}/common/scripts/run_trace_post_processing.sh {docker_container_name}:/usr/local/bin\n")
            f.write(f"docker exec --privileged {docker_container_name} /bin/bash -c '/usr/local/bin/common_entrypoint.sh'\n")
            f.write(f"docker exec --privileged {docker_container_name} /bin/bash -c \"echo 0 | sudo tee /proc/sys/kernel/randomize_va_space\"\n")
            f.write(f"docker exec --privileged --user={user} {docker_container_name} /bin/bash {trace_cmd}\n")
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

def remove_docker_containers(docker_prefix_list, job_name, user, dbg_lvl):
    try:
        for docker_prefix in docker_prefix_list:
            pattern = re.compile(fr"^{docker_prefix}_.*_{job_name}.*_.*_{user}$")
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

def prepare_trace(user, scarab_path, docker_home, job_name, dbg_lvl=1):
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

        trace_dir = f"{docker_home}/simpoint_flow/{job_name}"
        os.system(f"mkdir -p {trace_dir}/scarab/src/")
        os.system(f"cp {scarab_bin} {trace_dir}/scarab/src/scarab")

        try:
            os.symlink(f"{trace_dir}/scarab/src/scarab", f"{trace_dir}/scarab/src/scarab_{scarab_githash}")
        except FileExistsError:
            pass

        os.system(f"mkdir -p {trace_dir}/scarab/bin/scarab_globals")
        os.system(f"cp {scarab_path}/bin/scarab_launch.py  {trace_dir}/scarab/bin/scarab_launch.py ")
        os.system(f"cp {scarab_path}/bin/scarab_globals/*  {trace_dir}/scarab/bin/scarab_globals/ ")
        os.system(f"mkdir -p {trace_dir}/scarab/utils/memtrace")
        os.system(f"cp {scarab_path}/utils/memtrace/portabilize_trace.py  {trace_dir}/scarab/utils/memtrace/portabilize_trace.py ")

        os.system(f"chmod -R 777 {trace_dir}")
        os.system(f"setfacl -m \"o:rwx\" {trace_dir}")
    except Exception as e:
        raise e

def finish_trace(user, descriptor_data, workload_db_path, suite_db_path, dbg_lvl):
    def read_weight_file(file_path):
        weights = {}
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.split()
                weight = float(parts[0])
                segment_id = int(parts[1])
                weights[segment_id] = weight
        return weights

    def read_cluster_file(file_path):
        clusters = {}
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.split()
                cluster_id = int(parts[0])
                segment_id = int(parts[1])
                clusters[segment_id] = cluster_id
        return clusters

    def get_modules_dir_and_trace_file(trim_type, workload):
        modules_dir = ""
        trace_file = ""
        if trim_type == 2:
            modules_dir = f"/simpoint_traces/{workload}/traces_simp/raw/"
            trace_file = f"/simpoint_traces/{workload}/traces_simp/trace/"
        elif trim_type == 3:
            modules_dir = f"/simpoint_traces/{workload}/traces_simp/"
            trace_file = f"/simpoint_traces/{workload}/traces_simp/"
        return modules_dir, trace_file

    try:
        workload_db_data = read_descriptor_from_json(workload_db_path, dbg_lvl)
        suite_db_data = read_descriptor_from_json(suite_db_path, dbg_lvl)
        trace_configs = descriptor_data["trace_configurations"]
        job_name = descriptor_data["trace_name"]
        trace_dir = f"{descriptor_data['root_dir']}/simpoint_flow/{job_name}"
        for config in trace_configs:
            workload = config['workload']

            # Update workload_db_data
            trace_dict = {}
            trace_dict['dynamorio_args'] = config['dynamorio_args']
            trace_dict['clustering_k'] = config['clustering_k']

            simulation_dict = {}
            exec_dict = {}
            exec_dict['image_name'] = config['image_name']
            exec_dict['env_vars'] = config['env_vars']
            exec_dict['binary_cmd'] = config['binary_cmd']
            exec_dict['client_bincmd'] = config['client_bincmd']
            memtrace_dict = {}
            memtrace_dict['image_name'] = config['image_name']
            memtrace_dict['trim_type'] = 2
            memtrace_dict['modules_dir'], memtrace_dict['trace_file'] = get_modules_dir_and_trace_file(2, workload)
            simulation_dict['exec'] = exec_dict
            simulation_dict['memtrace'] = memtrace_dict

            weight_file = os.path.join(trace_dir, workload, "simpoints", "opt.w.lpt0.99")
            cluster_file = os.path.join(trace_dir, workload, "simpoints", "opt.p.lpt0.99")
            weights = read_weight_file(weight_file)
            clusters = read_cluster_file(cluster_file)
            simpoints = []
            # Match segment IDs between weight and cluster files
            for segment_id, weight in weights.items():
                if segment_id in clusters:
                    simpoints.append({
                        'cluster_id': clusters[segment_id],
                        'segment_id': segment_id,
                        'weight': weight
                    })

            workload_db_data[workload] = {
                "trace":trace_dict,
                "simulation":simulation_dict,
                "simpoints":simpoints
            }

            # Update suite_db_data
            suite = config['suite']
            subsuite = config['subsuite'] if config['subsuite'] else suite
            if suite in suite_db_data.keys() and subsuite in suite_db_data[suite].keys():
                suite_db_data[suite][subsuite]['predefined_simulation_mode'][workload] = "memtrace"
            else:
                simulation_mode_dict = {}
                simulation_mode_dict[workload] = "memtrace"
                subsuite_dict = {'predefined_simulation_mode':simulation_mode_dict}
                suite_db_data[suite] = subsuite_dict

        write_json_descriptor(workload_db_path, workload_db_data, dbg_lvl)
        write_json_descriptor(suite_db_path, suite_db_data, dbg_lvl)

        # TODO: copy successfully collected simpoints and traces to simpoint_traces_dir
        simpoint_traces_dir = descriptor_data["simpoint_traces_dir"]
    except Exception as e:
        raise e
