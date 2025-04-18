#!/usr/bin/python3

# 01/27/2025 Surim Oh | utilities.py

import json
import os
import subprocess
import re
import docker

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
            json.dump(descriptor_data, json_file, indent=2, separators=(",", ":"))
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

        if workload == None and cluster_id != None:
            err(f"If you want to run all the workloads within '{suite}', empty 'workload' and 'cluster_id'.", dbg_lvl)
            exit(1)

        if workload == None:
            if subsuite == None:
                for subsuite_ in suite_data[suite].keys():
                    for workload_ in suite_data[suite][subsuite_]["predefined_simulation_mode"].keys():
                        predef_mode = suite_data[suite][subsuite_]["predefined_simulation_mode"][workload_]
                        sim_mode_ = sim_mode
                        if sim_mode_ == None:
                            sim_mode_ = predef_mode
                        if sim_mode_ not in workloads_data[workload_]["simulation"].keys():
                            err(f"{sim_mode_} is not a valid simulation mode for workload {workload_}.", dbg_lvl)
                            exit(1)
            else:
                for workload_ in suite_data[suite][subsuite]["predefined_simulation_mode"].keys():
                    predef_mode = suite_data[suite][subsuite]["predefined_simulation_mode"][workload_]
                    sim_mode_ = sim_mode
                    if sim_mode_ == None:
                        sim_mode_ = predef_mode
                    if sim_mode_ not in workloads_data[workload_]["simulation"].keys():
                        err(f"{sim_mode_} is not a valid simulation mode for workload {workload_}.", dbg_lvl)
                        exit(1)
        else:
            if workload not in workloads_data.keys():
                err(f"Workload '{workload}' is not valid.", dbg_lvl)
                exit(1)

            if subsuite == None:
                for subsuite_ in suite_data[suite].keys():
                    predef_mode = suite_data[suite][subsuite_]["predefined_simulation_mode"][workload]
                    sim_mode_ = sim_mode
                    if sim_mode_ == None:
                        sim_mode_ = predef_mode
                    if sim_mode_ not in workloads_data[workload]["simulation"].keys():
                        err(f"{sim_mode_} is not a valid simulation mode for workload {workload}.", dbg_lvl)
                        exit(1)
            else:
                predef_mode = suite_data[suite][subsuite]["predefined_simulation_mode"][workload]
                sim_mode_ = sim_mode
                if sim_mode_ == None:
                    sim_mode_ = predef_mode
                if sim_mode_ not in workloads_data[workload]["simulation"].keys():
                    err(f"{sim_mode_} is not a valid simulation mode for workload {workload}.", dbg_lvl)
                    exit(1)

            if cluster_id != None:
                if "simpoints" not in workloads_data[workload].keys():
                    err(f"Simpoints are not available for workload {workload}. Choose 'null' for cluster id.", dbg_lvl)
                    exit(1)
                if cluster_id > 0:
                    found = False
                    for simpoint in workloads_data[workload]["simpoints"]:
                        if cluster_id == simpoint["cluster_id"]:
                            found = True
                            break
                    if not found:
                        err(f"Cluster ID {cluster_id} is not valid for workload '{workload}'.", dbg_lvl)
                        exit(1)
                elif cluster_id < 0:
                    err(f"Cluster ID should be greater than 0. {cluster_id} is not valid.", dbg_lvl)
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
def prepare_simulation(user, scarab_path, scarab_build, docker_home, experiment_name, architecture, docker_prefix, githash, infra_dir, interactive_shell=False, dbg_lvl=1):
    ## Copy required scarab files into the experiment folder
    try:
        local_uid = os.getuid()
        local_gid = os.getgid()

        scarab_githash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=scarab_path).decode("utf-8").strip()
        info(f"Scarab git hash: {scarab_githash}", dbg_lvl)

        # (Re)build the scarab binary first.
        if not interactive_shell and scarab_build == None:
            scarab_bin = f"{scarab_path}/src/build/opt/scarab"
            if not os.path.isfile(scarab_bin):
                scarab_build = 'opt'
                info(F"Scarab binary not found at '{scarab_bin}', build with {scarab_build}", dbg_lvl)

        if scarab_build != None:
            scarab_bin = f"{scarab_path}/src/build/{scarab_build}/scarab"
            info(f"Scarab binary at '{scarab_bin}', building it first, please wait...", dbg_lvl)
            docker_container_name = f"{docker_prefix}_{user}_scarab_build"
            subprocess.run(
                    ["docker", "run", "-e", f"user_id={local_uid}",
                     "-e", f"group_id={local_gid}",
                     "-e", f"username={user}",
                     "-dit", "--name", f"{docker_container_name}",
                     "--mount", f"type=bind,source={docker_home},target=/home/{user},readonly=false",
                     "--mount", f"type=bind,source={scarab_path},target=/scarab,readonly=false",
                     f"{docker_prefix}:{githash}", "/bin/bash"], check=True, capture_output=True, text=True)
            subprocess.run(
                    ["docker", "cp", f"{infra_dir}/common/scripts/root_entrypoint.sh", f"{docker_container_name}:/usr/local/bin"],
                    check=True, capture_output=True, text=True)
            subprocess.run(
                    ["docker", "cp", f"{infra_dir}/common/scripts/user_entrypoint.sh", f"{docker_container_name}:/usr/local/bin"],
                    check=True, capture_output=True, text=True)
            if os.path.isfile(f"{infra_dir}/workloads/{docker_prefix}/workload_root_entrypoint.sh"):
                subprocess.run(
                        ["docker", "cp", f"{infra_dir}/workloads/{docker_prefix}/workload_root_entrypoint.sh", f"{docker_container_name}:/usr/local/bin"],
                        check=True, capture_output=True, text=True)
            if os.path.isfile(f"{infra_dir}/workloads/{docker_prefix}/workload_user_entrypoint.sh"):
                subprocess.run(
                        ["docker", "cp", f"{infra_dir}/workloads/{docker_prefix}/workload_user_entrypoint.sh", f"{docker_container_name}:/usr/local/bin"],
                        check=True, capture_output=True, text=True)

            subprocess.run(
                    ["docker", "exec", "--privileged", f"{docker_container_name}", "/bin/bash", "-c", "\'/usr/local/bin/root_entrypoint.sh\'"],
                    check=True, capture_output=True, text=True)
            subprocess.run(
                    ["docker", "exec", f"--user={user}", f"--workdir=/home/{user}", f"{docker_container_name}", "/bin/bash", "-c", f"cd /scarab/src && make clean && make {scarab_build}"],
                    check=True, capture_output=True, text=True)
            subprocess.run(["docker", "rm", "-f", f"{docker_container_name}"], check=True, capture_output=True, text=True)

        experiment_dir = f"{docker_home}/simulations/{experiment_name}"
        os.system(f"mkdir -p {experiment_dir}/logs/")

        # Copy binary and architectural params to scarab/src
        arch_params = f"{scarab_path}/src/PARAMS.{architecture}"
        os.system(f"mkdir -p {experiment_dir}/scarab/src/")
        if not interactive_shell:
            if scarab_build:
                scarab_bin = f"{scarab_path}/src/build/{scarab_build}/scarab"
            else:
                scarab_bin = f"{scarab_path}/src/build/opt/scarab"
            dest_scarab_bin = f"{experiment_dir}/scarab/src/scarab"
            try:
                result = subprocess.run(['diff', '-q', scarab_bin, dest_scarab_bin], capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError as e:
                if e.returncode == 1 or e.returncode == 2:
                    info("scarab binaries differ or the destination binary does not exist. Will copy.", dbg_lvl)
                    result = subprocess.run(["cp", scarab_bin, dest_scarab_bin], capture_output=True, text=True)
                    if result.returncode != 0:
                        err(f"Failed to copy scarab binary: {result.stderr}", dbg_lvl)
                        raise RuntimeError(f"Failed to copy scarab binary. Existing binary is in use and differs from the new binary: {result.stderr}")
                else:
                    raise e
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

        return scarab_githash
    except Exception as e:
        subprocess.run(["docker", "rm", "-f", docker_container_name], check=True)
        info(f"Removed container: {docker_container_name}", dbg_lvl)
        raise e

def finish_simulation(user, docker_home):
    try:
        print("Finish simulation..")
    except Exception as e:
        raise e

# Generate command to do a single run of scarab
def generate_single_scarab_run_command(user, workload, group, experiment, config_key, config,
                                       mode, seg_size, arch, scarab_githash, cluster_id,
                                       trim_type, trace_file,
                                       env_vars, bincmd, client_bincmd):
    if mode == "memtrace":
        command = f"run_memtrace_single_simpoint.sh \\\"{workload}\\\" \\\"{group}\\\" \\\"/home/{user}/simulations/{experiment}/{config_key}\\\" \\\"{config}\\\" \\\"{seg_size}\\\" \\\"{arch}\\\" \\\"{trim_type}\\\" /home/{user}/simulations/{experiment}/scarab {cluster_id} {trace_file}"
    elif mode == "pt":
        command = f"run_pt_single_simpoint.sh \\\"{workload}\\\" \\\"{group}\\\" \\\"/home/{user}/simulations/{experiment}/{config_key}\\\" \\\"{config}\\\" \\\"{arch}\\\" \\\"{trim_type}\\\" /home/{user}/simulations/{experiment}/scarab"
    elif mode == "exec":
        command = f"run_exec_single_simpoint.sh \\\"{workload}\\\" \\\"{group}\\\" \\\"/home/{user}/simulations/{experiment}/{config_key}\\\" \\\"{config}\\\" \\\"{arch}\\\" /home/{user}/simulations/{experiment}/scarab {env_vars} {bincmd} {client_bincmd}"
    else:
        command = ""

    return command

def write_docker_command_to_file_run_by_root(user, local_uid, local_gid, workload, experiment_name,
                                             docker_prefix, docker_container_name, traces_dir,
                                             docker_home, githash, config_key, config, scarab_mode, seg_size, scarab_githash,
                                             architecture, cluster_id, trim_type, trace_file,
                                             env_vars, bincmd, client_bincmd, filename):
    try:
        scarab_cmd = generate_single_scarab_run_command(user, workload, docker_prefix, experiment_name, config_key, config,
                                                        scarab_mode, seg_size, architecture, scarab_githash, cluster_id,
                                                        trim_type, trace_file, env_vars, bincmd, client_bincmd)
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
            --mount type=bind,source={traces_dir},target=/simpoint_traces,readonly=true \
            --mount type=bind,source={docker_home},target=/home/{user},readonly=false \
            {docker_prefix}:{githash} \
            /bin/bash {scarab_cmd}\n")
    except Exception as e:
        raise e

def write_docker_command_to_file(user, local_uid, local_gid, workload, experiment_name,
                                 docker_prefix, docker_container_name, traces_dir,
                                 docker_home, githash, config_key, config, scarab_mode, scarab_githash,
                                 seg_size, architecture, cluster_id, trim_type, trace_file,
                                 env_vars, bincmd, client_bincmd, filename, infra_dir):
    try:
        scarab_cmd = generate_single_scarab_run_command(user, workload, docker_prefix, experiment_name, config_key, config,
                                                        scarab_mode, seg_size, architecture, scarab_githash, cluster_id,
                                                        trim_type, trace_file, env_vars, bincmd, client_bincmd)
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
            --mount type=bind,source={traces_dir},target=/simpoint_traces,readonly=true \
            --mount type=bind,source={docker_home},target=/home/{user},readonly=false \
            {docker_prefix}:{githash} \
            /bin/bash\n")
            f.write(f"docker cp {infra_dir}/scripts/utilities.sh {docker_container_name}:/usr/local/bin\n")
            f.write(f"docker cp {infra_dir}/common/scripts/root_entrypoint.sh {docker_container_name}:/usr/local/bin\n")
            f.write(f"docker cp {infra_dir}/common/scripts/user_entrypoint.sh {docker_container_name}:/usr/local/bin\n")
            if os.path.exists(f"{infra_dir}/workloads/{docker_prefix}/workload_root_entrypoint.sh"):
                f.write(f"docker cp {infra_dir}/workloads/{docker_prefix}/workload_root_entrypoint.sh {docker_container_name}:/usr/local/bin\n")
            if os.path.exists(f"{infra_dir}/workloads/{docker_prefix}/workload_user_entrypoint.sh"):
                f.write(f"docker cp {infra_dir}/workloads/{docker_prefix}/workload_user_entrypoint.sh {docker_container_name}:/usr/local/bin\n")
            if scarab_mode == "memtrace":
                f.write(f"docker cp {infra_dir}/common/scripts/run_memtrace_single_simpoint.sh {docker_container_name}:/usr/local/bin\n")
            elif scarab_mode == "pt":
                f.write(f"docker cp {infra_dir}/common/scripts/run_pt_single_simpoint.sh {docker_container_name}:/usr/local/bin\n")
            elif scarab_mode == "exec":
                f.write(f"docker cp {infra_dir}/common/scripts/run_exec_single_simpoint.sh {docker_container_name}:/usr/local/bin\n")
            f.write(f"docker exec --privileged {docker_container_name} /bin/bash -c '/usr/local/bin/root_entrypoint.sh'\n")
            f.write(f"docker exec --user={user} {docker_container_name} /bin/bash -c \"source /usr/local/bin/user_entrypoint.sh && {scarab_cmd}\"\n")
            f.write(f"docker rm -f {docker_container_name}\n")
    except Exception as e:
        raise e

def generate_single_trace_run_command(user, workload, image_name, trace_name, binary_cmd, client_bincmd, simpoint_mode, drio_args, clustering_k):
    command = ""
    if simpoint_mode == "cluster_then_trace":
        mode = 1
    elif simpoint_mode == "trace_then_post_process":
        mode = 2
    elif simpoint_mode == "iterative_trace":
        mode = 3
    command = f"python3 -u /usr/local/bin/run_simpoint_trace.py --workload {workload} --suite {image_name} --simpoint_mode {mode} --simpoint_home \\\"/home/{user}/simpoint_flow/{trace_name}\\\" --bincmd \\\"{binary_cmd}\\\""
    if client_bincmd != None:
        command = f"{command} --client_bincmd \\\"{client_bincmd}\\\""
    if drio_args != None:
        command = f"{command} --drio_args {drio_args}"
    if clustering_k != None:
        command = f"{command} -userk {clustering_k}"
    return command

def write_trace_docker_command_to_file(user, local_uid, local_gid, docker_container_name, githash,
                                       workload, image_name, trace_name, traces_dir, docker_home,
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
                    --mount type=bind,source={docker_home},target=/home/{user},readonly=false \
                    {image_name}:{githash} \
                    /bin/bash\n"
            f.write(f"{command}")
            f.write(f"docker cp {infra_dir}/scripts/utilities.sh {docker_container_name}:/usr/local/bin\n")
            f.write(f"docker cp {infra_dir}/common/scripts/root_entrypoint.sh {docker_container_name}:/usr/local/bin\n")
            f.write(f"docker cp {infra_dir}/common/scripts/user_entrypoint.sh {docker_container_name}:/usr/local/bin\n")
            if os.path.exists(f"{infra_dir}/workloads/{image_name}/workload_root_entrypoint.sh"):
                f.write(f"docker cp {infra_dir}/workloads/{image_name}/workload_root_entrypoint.sh {docker_container_name}:/usr/local/bin\n")
            if os.path.exists(f"{infra_dir}/workloads/{image_name}/workload_user_entrypoint.sh"):
                f.write(f"docker cp {infra_dir}/workloads/{image_name}/workload_user_entrypoint.sh {docker_container_name}:/usr/local/bin\n")
            f.write(f"docker cp {infra_dir}/common/scripts/run_clustering.sh {docker_container_name}:/usr/local/bin\n")
            f.write(f"docker cp {infra_dir}/common/scripts/run_simpoint_trace.py {docker_container_name}:/usr/local/bin\n")
            f.write(f"docker cp {infra_dir}/common/scripts/minimize_trace.sh {docker_container_name}:/usr/local/bin\n")
            f.write(f"docker cp {infra_dir}/common/scripts/run_trace_post_processing.sh {docker_container_name}:/usr/local/bin\n")
            f.write(f"docker cp {infra_dir}/common/scripts/gather_fp_pieces.py {docker_container_name}:/usr/local/bin\n")
            f.write(f"docker exec --privileged {docker_container_name} /bin/bash -c '/usr/local/bin/root_entrypoint.sh'\n")
            f.write(f"docker exec --privileged {docker_container_name} /bin/bash -c \"echo 0 | sudo tee /proc/sys/kernel/randomize_va_space\"\n")
            f.write(f"docker exec --privileged --user={user} --workdir=/home/{user} {docker_container_name} /bin/bash -c \"source /usr/local/bin/user_entrypoint.sh && {trace_cmd}\"\n")
            f.write(f"docker rm -f {docker_container_name}\n")
    except Exception as e:
        raise e

def get_simpoints (workload_data, sim_mode, dbg_lvl = 2):
    simpoints = {}
    if sim_mode == "memtrace":
        for simpoint in workload_data["simpoints"]:
            simpoints[f"{simpoint['cluster_id']}"] = simpoint["weight"]
    else:
        simpoints["0"] = 1.0

    return simpoints

def get_image_name(workloads_data, suite_data, simulation):
    suite = simulation["suite"]
    subsuite = simulation["subsuite"]
    workload = simulation["workload"]
    cluster_id = simulation["cluster_id"]
    sim_mode = simulation["simulation_type"]

    if workload != None:
        if subsuite == None:
            subsuite = next(iter(suite_data[suite]))
        predef_sim_mode = suite_data[suite][subsuite]["predefined_simulation_mode"][workload]
        if sim_mode == None:
            sim_mode = predef_sim_mode
        return workloads_data[workload]["simulation"][sim_mode]["image_name"]

    if subsuite != None:
        workload = next(iter(suite_data[suite][subsuite]["predefined_simulation_mode"]))
        predef_sim_mode = suite_data[suite][subsuite]["predefined_simulation_mode"][workload]
        if sim_mode == None:
            sim_mode = predef_sim_mode
    else:
        subsuite = next(iter(suite_data[suite]))
        workload = next(iter(suite_data[suite][subsuite]["predefined_simulation_mode"]))
        predef_sim_mode = suite_data[suite][subsuite]["predefined_simulation_mode"][workload]
        if sim_mode == None:
            sim_mode = predef_sim_mode

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
        sim_mode = simulation["simulation_type"]

        if workload == None:
            if subsuite == None:
                for subsuite_ in suite_data[suite].keys():
                    for workload_ in suite_data[suite][subsuite_]["predefined_simulation_mode"].keys():
                        predef_mode = suite_data[suite][subsuite_]["predefined_simulation_mode"][workload_]
                        sim_mode_ = sim_mode
                        if sim_mode_ == None:
                            sim_mode_ = predef_mode
                        if sim_mode_ in workloads_data[workload_]["simulation"].keys() and workloads_data[workload_]["simulation"][sim_mode_]["image_name"] not in image_list:
                            image_list.append(workloads_data[workload_]["simulation"][sim_mode_]["image_name"])
            else:
                for workload_ in suite_data[suite][subsuite]["predefined_simulation_mode"].keys():
                    predef_mode = suite_data[suite][subsuite]["predefined_simulation_mode"][workload_]
                    sim_mode_ = sim_mode
                    if sim_mode_ == None:
                        sim_mode_ = predef_mode
                    if sim_mode_ in workloads_data[workload_]["simulation"].keys() and workloads_data[workload_]["simulation"][sim_mode_]["image_name"] not in image_list:
                        image_list.append(workloads_data[workload_]["simulation"][sim_mode_]["image_name"])
        else:
            if subsuite == None:
                for subsuite_ in suite_data[suite].keys():
                    predef_mode = suite_data[suite][subsuite_]["predefined_simulation_mode"][workload]
                    sim_mode_ = sim_mode
                    if sim_mode_ == None:
                        sim_mode_ = predef_mode
                    if sim_mode_ in workloads_data[workload]["simulation"].keys() and workloads_data[workload]["simulation"][sim_mode_]["image_name"] not in image_list:
                        image_list.append(workloads_data[workload]["simulation"][sim_mode_]["image_name"])
            else:
                predef_mode = suite_data[suite][subsuite]["predefined_simulation_mode"][workload]
                sim_mode_ = sim_mode
                if sim_mode_ == None:
                    sim_mode_ = predef_mode
                if sim_mode_ in workloads_data[workload]["simulation"].keys() and workloads_data[workload]["simulation"][sim_mode_]["image_name"] not in image_list:
                    image_list.append(workloads_data[workload]["simulation"][sim_mode_]["image_name"])

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

def prepare_trace(user, scarab_path, scarab_build, docker_home, job_name, infra_dir, docker_prefix, githash, interactive_shell=False, dbg_lvl=1):
    try:
        local_uid = os.getuid()
        local_gid = os.getgid()

        scarab_githash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=scarab_path).decode("utf-8").strip()
        info(f"Scarab git hash: {scarab_githash}", dbg_lvl)

        # (Re)build the scarab binary first.
        if not interactive_shell and scarab_build == None:
            scarab_bin = f"{scarab_path}/src/build/opt/scarab"
            if not os.path.isfile(scarab_bin):
                scarab_build = 'opt'
                info(F"Scarab binary not found at '{scarab_bin}', build with {scarab_build}", dbg_lvl)

        if scarab_build != None:
            scarab_bin = f"{scarab_path}/src/build/{scarab_build}/scarab"
            info(f"Scarab binary at '{scarab_bin}', building it first, please wait...", dbg_lvl)
            docker_container_name = f"{docker_prefix}_{user}_scarab_build"
            subprocess.run(
                    ["docker", "run", "-e", f"user_id={local_uid}",
                     "-e", f"group_id={local_gid}",
                     "-e", f"username={user}",
                     "-dit", "--name", f"{docker_container_name}",
                     "--mount", f"type=bind,source={docker_home},target=/home/{user},readonly=false",
                     "--mount", f"type=bind,source={scarab_path},target=/scarab,readonly=false",
                     f"{docker_prefix}:{githash}", "/bin/bash"], check=True, capture_output=True, text=True)
            subprocess.run(
                    ["docker", "cp", f"{infra_dir}/common/scripts/root_entrypoint.sh", f"{docker_container_name}:/usr/local/bin"],
                    check=True, capture_output=True, text=True)
            subprocess.run(
                    ["docker", "cp", f"{infra_dir}/common/scripts/user_entrypoint.sh", f"{docker_container_name}:/usr/local/bin"],
                    check=True, capture_output=True, text=True)
            if os.path.exists(f"{infra_dir}/workloads/{docker_prefix}/workload_root_entrypoint.sh"):
                subprocess.run(
                        ["docker", "cp", f"{infra_dir}/workloads/{docker_prefix}/workload_root_entrypoint.sh", f"{docker_container_name}:/usr/local/bin"],
                        check=True, capture_output=True, text=True)
            if os.path.exists(f"{infra_dir}/workloads/{docker_prefix}/workload_user_entrypoint.sh"):
                subprocess.run(
                        ["docker", "cp", f"{infra_dir}/workloads/{docker_prefix}/workload_user_entrypoint.sh", f"{docker_container_name}:/usr/local/bin"],
                        check=True, capture_output=True, text=True)
            subprocess.run(
                    ["docker", "exec", "--privileged", f"{docker_container_name}", "/bin/bash", "-c", "\'/usr/local/bin/root_entrypoint.sh\'"],
                    check=True, capture_output=True, text=True)
            subprocess.run(
                    ["docker", "exec", f"--user={user}", f"--workdir=/home/{user}", f"{docker_container_name}", "/bin/bash", "-c", f"cd /scarab/src && make clean && make {scarab_build}"],
                    check=True, capture_output=True, text=True)
            subprocess.run(["docker", "rm", "-f", f"{docker_container_name}"], check=True, capture_output=True, text=True)

        trace_dir = f"{docker_home}/simpoint_flow/{job_name}"
        os.system(f"mkdir -p {trace_dir}/scarab/src/")
        if not interactive_shell:
            if scarab_build:
                scarab_bin = f"{scarab_path}/src/build/{scarab_build}/scarab"
            else:
                scarab_bin = f"{scarab_path}/src/build/opt/scarab"
            result = subprocess.run(["cp", scarab_bin, f"{trace_dir}/scarab/src/scarab"],
                                   capture_output=True,
                                   text=True)
            if result.returncode != 0:
                err(f"Failed to copy scarab binary: {result.stderr}", dbg_lvl)
                raise RuntimeError(f"Failed to copy scarab binary: {result.stderr}")

        try:
            os.symlink(f"{trace_dir}/scarab/src/scarab", f"{trace_dir}/scarab/src/scarab_{scarab_githash}")
        except FileExistsError:
            pass

        os.system(f"mkdir -p {trace_dir}/scarab/bin/scarab_globals")
        os.system(f"cp {scarab_path}/bin/scarab_launch.py  {trace_dir}/scarab/bin/scarab_launch.py ")
        os.system(f"cp {scarab_path}/bin/scarab_globals/*  {trace_dir}/scarab/bin/scarab_globals/ ")
        os.system(f"mkdir -p {trace_dir}/scarab/utils/memtrace")
        os.system(f"cp {scarab_path}/utils/memtrace/* {trace_dir}/scarab/utils/memtrace/ ")
    except Exception as e:
        subprocess.run(["docker", "rm", "-f", docker_container_name], check=True)
        info(f"Removed container: {docker_container_name}", dbg_lvl)
        raise e

def finish_trace(user, descriptor_data, workload_db_path, suite_db_path, dbg_lvl):
    def read_first_line(file_path):
        with open(file_path, 'r') as f:
            value = f.readline().rstrip('\n')
        return value

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

    try:
        workload_db_data = read_descriptor_from_json(workload_db_path, dbg_lvl)
        suite_db_data = read_descriptor_from_json(suite_db_path, dbg_lvl)
        trace_configs = descriptor_data["trace_configurations"]
        job_name = descriptor_data["trace_name"]
        trace_dir = f"{descriptor_data['root_dir']}/simpoint_flow/{job_name}"
        target_traces_dir = descriptor_data["traces_dir"]
        docker_home = descriptor_data["root_dir"]

        print("Copying the successfully collected traces and update workloads_db.json/suite_db.json...")

        for config in trace_configs:
            workload = config['workload']

            # Update workload_db_data
            trace_dict = {}
            trace_dict['dynamorio_args'] = config['dynamorio_args']
            trace_dict['clustering_k'] = config['clustering_k']

            simulation_dict = {}
            exec_dict = {}
            exec_dict['image_name'] = config['image_name']
            segment_size_file = os.path.join(trace_dir, workload, "fingerprint", "segment_size")
            exec_dict['segment_size'] = int(read_first_line(segment_size_file))
            exec_dict['env_vars'] = config['env_vars']
            exec_dict['binary_cmd'] = config['binary_cmd']
            exec_dict['client_bincmd'] = config['client_bincmd']
            memtrace_dict = {}
            memtrace_dict['image_name'] = "allbench_traces"
            if config['trace_type'] == "trace_then_cluster":
                trim_type = 1
            elif config['trace_type'] == "cluster_then_trace":
                trim_type = 2
            elif config['trace_type'] == "iterative_trace":
                trim_type = 3
            else:
                raise Exception(f"Invalid trace type: {config['trace_type']}")
            memtrace_dict['trim_type'] = trim_type
            memtrace_dict['segment_size'] = int(read_first_line(segment_size_file))
            trace_clustering_info = read_descriptor_from_json(f"{docker_home}/simpoint_flow/{job_name}/{workload}/trace_clustering_info.json", dbg_lvl)
            memtrace_dict['whole_trace_file'] = trace_clustering_info['trace_file']
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
                subsuite_dict = {}
                subsuite_dict['predefined_simulation_mode'] = simulation_mode_dict
                suite_dict = {}
                suite_dict[subsuite] = subsuite_dict
                suite_db_data[suite] = suite_dict

            # TODO: switch to a hierarchical path
            # target_traces_path = f"{target_traces_dir}/{suite}/{subsuite}/{workload}"
            target_traces_path = f"{target_traces_dir}/{workload}"
            # Copy successfully collected simpoints and traces to target_traces_dir
            os.system(f"mkdir -p {target_traces_path}/simpoints")
            os.system(f"mkdir -p {target_traces_path}/traces_simp")
            os.system(f"cp -r {trace_dir}/{workload}/simpoints/* {target_traces_path}/simpoints/")
            if trim_type != 3:
                os.system(f"cp -r {trace_dir}/{workload}/traces_simp/* {target_traces_path}/traces_simp/")
                os.system(f"mkdir -p {target_traces_path}/traces/whole/trace")
                os.system(f"mkdir -p {target_traces_path}/traces/whole/raw")
                os.system(f"mkdir -p {target_traces_path}/traces/whole/bin")
                os.system(f"mkdir -p {target_traces_path}/traces_simp/bin")
                trace_clustering_info = read_descriptor_from_json(os.path.join(trace_dir, workload, "trace_clustering_info.json"), dbg_lvl)
                whole_trace_dir = trace_clustering_info['dr_folder']
                trace_file = trace_clustering_info['trace_file']
                subprocess.run([f"cp {trace_dir}/{workload}/traces/whole/{whole_trace_dir}/trace/{trace_file} {target_traces_path}/traces/whole/trace"], check=True, shell=True)
                subprocess.run([f"cp {trace_dir}/{workload}/traces/whole/{whole_trace_dir}/raw/modules.log {target_traces_path}/traces/whole/raw/modules.log"], check=True, shell=True)
                subprocess.run([f"cp {trace_dir}/{workload}/traces/whole/{whole_trace_dir}/raw/modules.log {target_traces_path}/traces_simp/raw/modules.log"], check=True, shell=True)
                subprocess.run([f"cp {trace_dir}/{workload}/traces/whole/{whole_trace_dir}/bin/* {target_traces_path}/traces/whole/bin"], check=True, shell=True)
                subprocess.run([f"cp {trace_dir}/{workload}/traces/whole/{whole_trace_dir}/bin/* {target_traces_path}/traces_simp/bin"], check=True, shell=True)
            else:
                trace_clustering_info = read_descriptor_from_json(os.path.join(trace_dir, workload, "trace_clustering_info.json"), dbg_lvl)
                largest_traces = trace_clustering_info['trace_file']

                for trace_path in largest_traces:
                    print("Processing trace:", trace_path)
                    prefix = "traces_simp/"
                    if prefix in trace_path:
                        relative_part = trace_path.split(prefix, 1)[1]
                        trace_source = os.path.join(trace_dir, workload, "traces_simp", relative_part)
                        trace_dest = os.path.join(target_traces_path, "traces_simp", relative_part)

                        os.makedirs(os.path.dirname(trace_dest), exist_ok=True)
                        os.system(f"cp -r {trace_source} {trace_dest}")

                        parts = relative_part.split(os.sep)
                        dr_folder_rel = os.path.join(parts[0], parts[1])
                        source_dr_folder = os.path.join(trace_dir, workload, "traces_simp", dr_folder_rel)
                        dest_dr_folder = os.path.join(target_traces_path, "traces_simp", dr_folder_rel)

                        os.makedirs(os.path.join(dest_dr_folder, "raw"), exist_ok=True)
                        os.makedirs(os.path.join(dest_dr_folder, "bin"), exist_ok=True)

                        os.system(f"cp {source_dr_folder}/raw/modules.log {dest_dr_folder}/raw/modules.log")
                        os.system(f"cp -r {source_dr_folder}/bin/* {dest_dr_folder}/bin/")

                simulation_dict['memtrace']['whole_trace_file'] = None

        write_json_descriptor(workload_db_path, workload_db_data, dbg_lvl)
        write_json_descriptor(suite_db_path, suite_db_data, dbg_lvl)

        print("Recover the ASLR setting with sudo. Provide password..")
        os.system("echo 2 | sudo tee /proc/sys/kernel/randomize_va_space")

    except Exception as e:
        raise e

def is_container_running(container_name, dbg_lvl):
    client = docker.from_env()
    try:
        container = client.containers.get(container_name)
        info(f"container {container_name} is already running.", dbg_lvl)
        return container.status == "running"
    except docker.errors.NotFound:
        return False

def count_interactive_shells(container_name, dbg_lvl):
    client = docker.APIClient()
    try:
        container = client.inspect_container(container_name)
        container_id = container['Id']
        processes = client.top(container_id)

        shell_count = 0
        for process in processes['Processes']:
            cmd = ' '.join(process)
            # Check for common interactive shells
            if any(shell in cmd for shell in ['bash', 'sh', 'zsh']):
                shell_count += 1
        info(f"{shell_count} shells are running for {container_name}.", dbg_lvl)
        return shell_count
    except docker.errors.NotFound:
        print(f"Container '{container_name}' not found.")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 0

def image_exist(image_tag):
    try:
        output = subprocess.check_output(["docker", "images", "-q", image_tag])
        return bool(output.strip())
    except subprocess.CalledProcessError:
        return False
