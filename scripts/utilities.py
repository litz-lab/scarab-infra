#!/usr/bin/python3

# 01/27/2025 Surim Oh | utilities.py

import json
import os
import subprocess
import re
import shlex
import shutil
from pathlib import Path
import importlib
import sys

try:
    import docker
except ImportError:  # pragma: no cover - docker is optional for some commands
    docker = None

# Add the project root to sys.path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import workloads.extract_top_simpoints as extract_top_simpoints
importlib.reload(extract_top_simpoints)

DEFAULT_CONDA_ENV = "scarabinfra"
_docker_client = None


def get_docker_client():
    global _docker_client
    if docker is None:
        return None
    if _docker_client is None:
        try:
            _docker_client = docker.from_env()
        except Exception:
            _docker_client = None
    return _docker_client

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

def run_on_node(cmd, node=None, **kwargs):
    if node != None:
        cmd = ["srun", f"--nodelist={node}"] + cmd
    return subprocess.run(cmd, **kwargs)

def validate_simulation(workloads_data, simulations, dbg_lvl = 2):
    for simulation in simulations:
        suite = simulation["suite"]
        subsuite = simulation["subsuite"]
        workload = simulation["workload"]
        cluster_id = simulation["cluster_id"]
        sim_mode = simulation["simulation_type"]
        sim_warmup = simulation["warmup"]

        if suite == None:
            err(f"Suite field cannot be null.", dbg_lvl)
            exit(1)

        if suite not in workloads_data.keys():
            err(f"Suite '{suite}' is not valid.", dbg_lvl)
            exit(1)

        if subsuite != None and subsuite not in workloads_data[suite].keys():
            err(f"Subsuite '{subsuite}' is not valid in Suite '{suite}'.", dbg_lvl);
            exit(1)

        if workload == None and cluster_id != None:
            err(f"If you want to run all the workloads within '{suite}', empty 'workload' and 'cluster_id'.", dbg_lvl)
            exit(1)

        if workload == None:
            if subsuite == None:
                for subsuite_ in workloads_data[suite].keys():
                    for workload_ in workloads_data[suite][subsuite_].keys():
                        predef_mode = workloads_data[suite][subsuite_][workload_]["simulation"]["prioritized_mode"]
                        sim_mode_ = sim_mode
                        if sim_mode_ == None:
                            sim_mode_ = predef_mode
                        if sim_mode_ not in workloads_data[suite][subsuite_][workload_]["simulation"].keys():
                            err(f"{sim_mode_} is not a valid simulation mode for workload {workload_}.", dbg_lvl)
                            exit(1)
                        if sim_warmup is not None and sim_mode_ == "memtrace" and sim_warmup > workloads_data[suite][subsuite_][workload_]["simulation"]["memtrace"]["warmup"]:
                            err(f"{sim_warmup} is not a valid warmup for workload {workload_} and {sim_mode_}.", dbg_lvl)
                            exit(1)

            else:
                for workload_ in workloads_data[suite][subsuite].keys():
                    predef_mode = workloads_data[suite][subsuite][workload_]["simulation"]["prioritized_mode"]
                    sim_mode_ = sim_mode
                    if sim_mode_ == None:
                        sim_mode_ = predef_mode
                    if sim_mode_ not in workloads_data[suite][subsuite][workload_]["simulation"].keys():
                        err(f"{sim_mode_} is not a valid simulation mode for workload {workload_}.", dbg_lvl)
                        exit(1)
                    if sim_warmup is not None and sim_mode_ == "memtrace" and sim_warmup > workloads_data[suite][subsuite][workload_]["simulation"]["memtrace"]["warmup"]:
                        err(f"{sim_warmup} is not a valid warmup for workload {workload_} and {sim_mode_}.", dbg_lvl)
                        exit(1)
        else:
            if subsuite == None:
                found = False
                for subsuite_ in workloads_data[suite].keys():
                    if workload not in workloads_data[suite][subsuite_].keys():
                        continue
                    found = True
                    predef_mode = workloads_data[suite][subsuite_][workload]["simulation"]["prioritized_mode"]
                    sim_mode_ = sim_mode
                    if sim_mode_ == None:
                        sim_mode_ = predef_mode
                    if sim_mode_ not in workloads_data[suite][subsuite_][workload]["simulation"].keys():
                        err(f"{sim_mode_} is not a valid simulation mode for workload {workload}.", dbg_lvl)
                        exit(1)
                    if sim_warmup is not None and sim_mode_ == "memtrace" and sim_warmup > workloads_data[suite][subsuite_][workload]["simulation"]["memtrace"]["warmup"]:
                        err(f"{sim_warmup} is not a valid warmup for workload {workload} and {sim_mode_}.", dbg_lvl)
                        exit(1)
                if not found:
                    err(f"Workload '{workload}' is not valid in suite {suite}", dbg_lvl)
                    exit(1)
            else:
                if workload not in workloads_data[suite][subsuite].keys():
                    err(f"Workload '{workload}' is not valid in suite {suite} and subsuite {subsuite}.", dbg_lvl)
                predef_mode = workloads_data[suite][subsuite][workload]["simulation"]["prioritized_mode"]
                sim_mode_ = sim_mode
                if sim_mode_ == None:
                    sim_mode_ = predef_mode
                if sim_mode_ not in workloads_data[suite][subsuite][workload]["simulation"].keys():
                    err(f"{sim_mode_} is not a valid simulation mode for workload {workload}.", dbg_lvl)
                    exit(1)
                if sim_warmup is not None and sim_mode_ == "memtrace" and sim_warmup > workloads_data[suite][subsuite][workload]["simulation"]["memtrace"]["warmup"]:
                    err(f"{sim_warmup} is not a valid warmup for workload {workload} and {sim_mode_}.", dbg_lvl)
                    exit(1)

            if cluster_id != None:
                if "simpoints" not in workloads_data[suite][subsuite][workload].keys():
                    err(f"Simpoints are not available for workload {workload}. Choose 'null' for cluster id.", dbg_lvl)
                    exit(1)
                if cluster_id > 0:
                    found = False
                    for simpoint in workloads_data[suite][subsuite][workload]["simpoints"]:
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

# Prepare the docker image on each node
# Inputs:   docker_prefix - docker image name
#           image_tag - image name with the tag to build
#           latest_image_tag - the latest pre-built image name with the tag
#           nodes - list of available nodes (local runner when it's none)
#           diff_output - diff of two git hashes in the directories that change the docker image
# Output: list of nodes where the docker image is ready
def prepare_docker_image(docker_prefix, image_tag, nodes=None, dbg_lvl=1):
    if nodes is None:
        nodes = []
    # build the image also locally
    nodes = [None] + nodes
    available_nodes = []
    for node in nodes:
        if not image_exist(image_tag, node):
            print(f"Couldn't find image {image_tag} on {node}")
            try:
                sci_path = os.path.join(project_root, "sci")
                print(f"Invoking {sci_path} --build-image {docker_prefix} on {node if node else 'local host'}")
                # Ensure stdout is streamed for visibility when running locally.
                run_on_node([sci_path, "--build-image", docker_prefix], node, check=True)
            except subprocess.CalledProcessError as e:
                err(f"sci --build-image failed with return code {e.returncode}", dbg_lvl)
                failure_stdout = getattr(e, "stdout", None)
                if failure_stdout:
                    err(failure_stdout.decode(), dbg_lvl)
                if not image_exist(image_tag, node):
                    err(f"Still couldn't find image {image_tag} after attempting to build.", dbg_lvl)
                    exit(1)
        available_nodes.append(node)
    # If docker image still does not exist anywhere, exit
    if available_nodes == []:
        err(f"Error with preparing docker image for {image_tag}", dbg_lvl)
        exit(1)

# Locally builds scarab using docker. No caching or skipping logic
def build_scarab_binary(user, scarab_path, scarab_build, docker_home, docker_prefix, githash, infra_dir, dbg_lvl=1, stream_build=False):
    local_uid = os.getuid()
    local_gid = os.getgid()

    exception = None

    scarab_bin = f"{scarab_path}/src/build/{scarab_build}/scarab"
    info(f"Scarab binary at '{scarab_bin}', building it first, please wait...", dbg_lvl)
    docker_container_name = f"{docker_prefix}_{user}_scarab_build"
    try:
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

        info(f"Building scarab with image {githash}...", dbg_lvl)
        build_cmd = [
                "docker",
                    "exec",
                    f"--user={user}",
                    f"--workdir=/home/{user}",
                    f"{docker_container_name}",
                    "/bin/bash",
                    "-c",
                    f"cd /scarab/src && make {scarab_build}"
            ]

        if stream_build:
            build_result = subprocess.run(build_cmd, text=True)
        else:
            build_result = subprocess.run(build_cmd, capture_output=True, text=True)

        if build_result.returncode != 0:
            if stream_build:
                err("Scarab build failed. See output above for details.", dbg_lvl)
            else:
                err(f"Build stdout: {build_result.stdout}", dbg_lvl)
                err(f"Build stderr: {build_result.stderr}", dbg_lvl)

    except Exception as e:
        exception = e
    finally:
        # Always clean up build container
        subprocess.run(["docker", "rm", "-f", f"{docker_container_name}"], check=True, capture_output=True, text=True)

    if exception != None:
        raise exception

# Wrapper function that handles rebuilding scarab if needed, and caching
def rebuild_scarab(infra_dir, scarab_path, user, docker_home, docker_prefix, githash, scarab_githash, scarab_build, stream_build=False, dbg_lvl=1):
    current_scarab_bin = f"{infra_dir}/scarab_builds/scarab_current"
    build_mode = scarab_build if scarab_build else "opt"

    # Check for suitable current binary in cache
    if not os.path.isfile(current_scarab_bin):
        warn(f"Scarab binary for current hash not found in cache. Will build it", dbg_lvl)
        build_mode = "opt"

    scarab_bin = f"{scarab_path}/src/build/{build_mode}/scarab"

    # Build and copy to cache
    try:
        build_scarab_binary(
            user,
            scarab_path,
            build_mode,
            docker_home,
            docker_prefix,
            githash,
            infra_dir,
            dbg_lvl=dbg_lvl,
            stream_build=stream_build,
        )

        if not os.path.isfile(scarab_bin):
            err("Scarab not found after building", dbg_lvl)
            raise RuntimeError(f"Scarab binary not found at {scarab_bin} after build!")

        # Name with git hash, with index for different iterations
        build_differs = False
        try:
            info(f"diff {current_scarab_bin} {scarab_bin}", dbg_lvl)
            subprocess.check_output(f"diff {current_scarab_bin} {scarab_bin}", shell=True, text=True)
        except subprocess.CalledProcessError:
            info("Caught exception caused by difference between cached current binary and build result", dbg_lvl)
            build_differs = True
        except FileNotFoundError:
            build_differs = True

        if not build_differs:
            info(f"Current scarab binary is the same as the cached version. Not updating cache.", dbg_lvl)
        else:
            info(f"Current scarab binary differs from cached version. Updating cache...", dbg_lvl)

            # Figure out index for binary. Order is _0 _1, _2, ...
            # Find all existing binaries with the githash
            scarab_binaries = os.listdir(f"{infra_dir}/scarab_builds")
            pattern = re.compile(f"scarab_{scarab_githash}(_|$)")
            current_githash_binaries = list(filter(pattern.match, scarab_binaries))

            print("Binaries matching current githash:", current_githash_binaries, "out of:", scarab_binaries, pattern)

            # If none exist, put it without index. Otherwise, add postfix index
            if current_githash_binaries == []:
                info(f"No binaries with hash {scarab_githash} exist. Creating version 0...", dbg_lvl)
                githash_scarab_bin = f"{infra_dir}/scarab_builds/scarab_{scarab_githash}_0"
            else:
                idx_pattern = re.compile(f"scarab_{scarab_githash}_")
                current_githash_versions = list(filter(idx_pattern.match, current_githash_binaries))

                print("Versions matching current githash:", current_githash_binaries)

                # If they do exist, take max index and increment it
                current_indicies = list(map(lambda x: int(x.split("_")[-1]), current_githash_versions))

                print("New index:", max(current_indicies)+1)
                githash_scarab_bin = f"{infra_dir}/scarab_builds/scarab_{scarab_githash}_{max(current_indicies)+1}"

            info(f"Copying scarab binary for {githash_scarab_bin} to cache", dbg_lvl)
            os.system(f"cp {scarab_bin} {githash_scarab_bin}")
            os.system(f"cp {scarab_bin} {current_scarab_bin}")

    except Exception as e:
        err(f"Scarab build failed! {str(e)}", dbg_lvl)
        raise e

    # Check for suitable current binary in cache after build
    if not os.path.isfile(current_scarab_bin):
        err(f"Scarab binary for current hash not found in cache after building", dbg_lvl)
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
def prepare_simulation(user, scarab_path, scarab_build, docker_home, experiment_name, architecture, docker_prefix_list, githash, infra_dir, scarab_binaries, interactive_shell=False, available_slurm_nodes=[], dbg_lvl=1, stream_build=False):
    # prepare docker images
    image_tag_list = []
    try:
        for docker_prefix in docker_prefix_list:
            image_tag = f"{docker_prefix}:{githash}"
            image_tag_list.append(image_tag)
            prepare_docker_image(docker_prefix, image_tag, available_slurm_nodes, dbg_lvl)
    except subprocess.CalledProcessError as e:
        info(f"Docker image preparation failed: {e.stderr if isinstance(e.stderr, str) else e.stderr.decode() if e.stderr else str(e)}", dbg_lvl)
        raise e
    except Exception as e:
        info(f"Unexpected error during docker image preparation: {str(e)}", dbg_lvl)
        raise e

    try:
        scarab_githash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=scarab_path).decode("utf-8").strip()
    except Exception as e:
        err(f"Could not get scarab githash: {str(e)}", dbg_lvl)
        raise e

    info(f"Current scarab git hash: {scarab_githash}", dbg_lvl)

    if not os.path.exists(f"{infra_dir}/scarab_builds"):
        os.system(f"mkdir -p {infra_dir}/scarab_builds")

    ## Copy required scarab files into the experiment folder
    docker_prefix = docker_prefix_list[0]
    docker_container_name = None
    try:
        local_uid = os.getuid()
        local_gid = os.getgid()

        experiment_dir = f"{docker_home}/simulations/{experiment_name}"
        os.system(f"mkdir -p {experiment_dir}/logs/")
        dest_scarab_bin = f"{experiment_dir}/scarab/src/scarab"

        # Make sure each git hash is present in the cache
        for bin_name in scarab_binaries:
            # Current will build if not present
            if bin_name == "scarab_current":
                continue

            scarab_ver = f"{infra_dir}/scarab_builds/{bin_name}"
            if os.path.isfile(scarab_ver):
                info(f"Scarab binary named {bin_name} found in cache!", dbg_lvl)
                continue

            err(f"Scarab binary named {bin_name} not found in cache. Please check out that version and build it", dbg_lvl)
            exit()

        # (Re)build the scarab binary first
        rebuild_scarab(infra_dir, scarab_path, user, docker_home, docker_prefix, githash, scarab_githash, scarab_build, stream_build=stream_build, dbg_lvl=dbg_lvl)

        # Copy architectural params to scarab/src
        arch_params = f"{scarab_path}/src/PARAMS.{architecture}"
        os.system(f"mkdir -p {experiment_dir}/scarab/src/")

        # Copy from cache all required scarab binaries
        for bin_name in scarab_binaries:
            scarab_ver = f"{infra_dir}/scarab_builds/{bin_name}"
            os.system(f"cp {scarab_ver} {experiment_dir}/scarab/src/")

        os.system(f"cp {arch_params} {experiment_dir}/scarab/src")

        # Required for non mode 4. Copy launch scripts from the docker container's scarab repo.
        # NOTE: Could cause issues if a copied version of scarab is incompatible with the version of
        # the launch scripts in the docker container's repo
        os.system(f"mkdir -p {experiment_dir}/scarab/bin/scarab_globals")
        os.system(f"cp {scarab_path}/bin/scarab_launch.py  {experiment_dir}/scarab/bin/scarab_launch.py ")
        os.system(f"cp {scarab_path}/bin/scarab_globals/*  {experiment_dir}/scarab/bin/scarab_globals/ ")

        return scarab_githash, image_tag_list
    except subprocess.CalledProcessError as e:
        if docker_container_name:
            try:
                subprocess.run(["docker", "rm", "-f", docker_container_name], check=True)
                info(f"Removed container: {docker_container_name}", dbg_lvl)
            except subprocess.CalledProcessError:
                info(f"Could not remove container: {docker_container_name}", dbg_lvl)

        info(f"Scarab build failed: {e.stderr if isinstance(e.stderr, str) else e.stderr.decode() if e.stderr else str(e)}", dbg_lvl)
        raise e
    except Exception as e:
        if docker_container_name:
            try:
                subprocess.run(["docker", "rm", "-f", docker_container_name], check=True)
                info(f"Removed container: {docker_container_name}", dbg_lvl)
            except subprocess.CalledProcessError:
                info(f"Could not remove container: {docker_container_name}", dbg_lvl)
        info(f"Unexpected error during scarab build: {str(e)}", dbg_lvl)

        raise e

def finish_simulation(user, docker_home, descriptor_path, root_dir, experiment_name, image_tag_list, available_nodes, slurm_ids = None, dont_collect = False):
    experiment_dir = f"{root_dir}/simulations/{experiment_name}"
    # clean up docker images only when no container is running on top of the image (the other user may be using it)
    # ignore the exception to ignore the rmi failure due to existing containers
    nodes = ' '.join(available_nodes)
    images = ' '.join(image_tag_list)
    clean_cmd = f"scripts/docker_cleaner.py --images {images}"
    if nodes:
        clean_cmd = clean_cmd + f" --nodes {nodes}"
    if slurm_ids:
        sbatch_cmd = f"sbatch --dependency=afterany:{','.join(slurm_ids)} -o {experiment_dir}/logs/stat_collection_job_%j.out "
        clean_cmd = sbatch_cmd + clean_cmd
    print(clean_cmd)
    os.system(clean_cmd)

    if dont_collect:
        return

    descriptor_abs = os.path.abspath(descriptor_path)
    stats_output = os.path.join(experiment_dir, "collected_stats.csv")
    stat_script = os.path.join(project_root, "scarab_stats", "stat_collector.py")

    conda_cmd = shutil.which("conda")
    python_executable = sys.executable
    env_python = None
    if conda_cmd:
        conda_path = Path(conda_cmd).resolve()
        base_dir = conda_path.parent
        if base_dir.name in {"bin", "condabin"}:
            base_prefix = base_dir.parent
        else:
            base_prefix = base_dir
        candidate = base_prefix / "envs" / DEFAULT_CONDA_ENV / "bin" / "python"
        if candidate.exists():
            env_python = str(candidate)

    tmp_dir = os.environ.get("TMPDIR")
    if not tmp_dir or not os.path.isdir(tmp_dir):
        tmp_dir = os.path.join(experiment_dir, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)

    if env_python:
        stat_runner_parts = [env_python, stat_script]
    elif conda_cmd:
        stat_runner_parts = [
            conda_cmd,
            "run",
            "-n",
            DEFAULT_CONDA_ENV,
            "python",
            stat_script,
        ]
    else:
        stat_runner_parts = [python_executable, stat_script]

    collect_parts = ["env", f"TMPDIR={tmp_dir}"] + stat_runner_parts + ["-d", descriptor_abs, "-o", stats_output]
    collect_stats_cmd = shlex.join(collect_parts)

    if slurm_ids:
        # afterok will not run if jobs fail. afterany used with stat_collector's error checking
        log_path = os.path.join(experiment_dir, "logs", "stat_collection_job_%j.out")
        sbatch_cmd = (
            f"sbatch --dependency=afterany:{','.join(slurm_ids)} "
            f"-o {shlex.quote(log_path)} --wrap={shlex.quote(collect_stats_cmd)}"
        )
        collect_stats_cmd = sbatch_cmd

    os.system(collect_stats_cmd)

    try:
        print("Finish simulation..")
    except Exception as e:
        raise e

# Generate command to do a single run of scarab
def generate_single_scarab_run_command(user, workload_home, experiment, config_key, config,
                                       mode, seg_size, arch, scarab_binary, cluster_id,
                                       warmup, trace_warmup, trace_type, trace_file,
                                       env_vars, bincmd, client_bincmd):

    if mode == "memtrace":
        command = f"run_memtrace_single_simpoint.sh \\\"{workload_home}\\\" \\\"/home/{user}/simulations/{experiment}/{config_key}\\\" \\\"{config}\\\" \\\"{seg_size}\\\" \\\"{arch}\\\" \\\"{warmup}\\\" \\\"{trace_warmup}\\\" \\\"{trace_type}\\\" /home/{user}/simulations/{experiment}/scarab {cluster_id} {trace_file} {scarab_binary}"
    elif mode == "pt":
        command = f"run_pt_single_simpoint.sh \\\"{workload_home}\\\" \\\"/home/{user}/simulations/{experiment}/{config_key}\\\" \\\"{config}\\\" \\\"{arch}\\\" \\\"{warmup}\\\" /home/{user}/simulations/{experiment}/scarab {scarab_binary}"

    elif mode == "exec":
        command = f"run_exec_single_simpoint.sh \\\"{workload_home}\\\" \\\"/home/{user}/simulations/{experiment}/{config_key}\\\" \\\"{config}\\\" \\\"{arch}\\\" /home/{user}/simulations/{experiment}/scarab {env_vars} {bincmd} {client_bincmd} {scarab_binary}"
    else:
        command = ""

    return command

def write_docker_command_to_file_run_by_root(user, local_uid, local_gid, workload, workload_home, experiment_name,
                                             docker_prefix, docker_container_name, traces_dir,
                                             docker_home, githash, config_key, config, scarab_mode, seg_size, scarab_githash,
                                             architecture, cluster_id, warmup, trace_warmup, trace_type, trace_file,
                                             env_vars, bincmd, client_bincmd, filename):
    try:
        scarab_cmd = generate_single_scarab_run_command(user, workload_home, experiment_name, config_key, config,
                                                        scarab_mode, seg_size, architecture, scarab_githash, cluster_id,
                                                        warmup, trace_warmup, trace_type, trace_file, env_vars, bincmd, client_bincmd)
        with open(filename, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(f"echo \"Running {config_key} {workload_home} {cluster_id}\"\n")
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

def write_docker_command_to_file(user, local_uid, local_gid, workload, workload_home, experiment_name,
                                 docker_prefix, docker_container_name, traces_dir,
                                 docker_home, githash, config_key, config, scarab_mode, scarab_binary,
                                 seg_size, architecture, cluster_id, warmup, trace_warmup, trace_type,
                                 trace_file, env_vars, bincmd, client_bincmd, filename, infra_dir):
    try:
        scarab_cmd = generate_single_scarab_run_command(user, workload_home, experiment_name, config_key, config,
                                                        scarab_mode, seg_size, architecture, scarab_binary, cluster_id,
                                                        warmup, trace_warmup, trace_type, trace_file, env_vars, bincmd, client_bincmd)
        with open(filename, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(f"echo \"Running {config_key} {workload_home} {cluster_id}\"\n")
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
                                       clustering_k, filename, infra_dir, application_dir):
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
                    --mount type=bind,source={application_dir},target=/tmp_home/application,readonly=false \
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

def get_image_name(workloads_data, simulation):
    suite = simulation["suite"]
    subsuite = simulation["subsuite"]
    workload = simulation["workload"]
    cluster_id = simulation["cluster_id"]
    sim_mode = simulation["simulation_type"]

    if workload != None:
        if subsuite == None:
            subsuite = next(iter(workloads_data[suite]))
        predef_sim_mode = workloads_data[suite][subsuite][workload]["simulation"]["prioritized_mode"]
        if sim_mode == None:
            sim_mode = predef_sim_mode
        return workloads_data[suite][subsuite][workload]["simulation"][sim_mode]["image_name"]

    if subsuite != None:
        workload = next(iter(workloads_data[suite][subsuite]))
        predef_sim_mode = workloads_data[suite][subsuite][workload]["simulation"]["prioritized_mode"]
        if sim_mode == None:
            sim_mode = predef_sim_mode
    else:
        subsuite = next(iter(workloads_data[suite]))
        workload = next(iter(workloads_data[suite][subsuite]))
        predef_sim_mode = workloads_data[suite][subsuite][workload]["simulation"]["prioritized_mode"]
        if sim_mode == None:
            sim_mode = predef_sim_mode

    return workloads_data[suite][subsuite][workload]["simulation"][sim_mode]["image_name"]

def remove_docker_containers(docker_prefix_list, job_name, user, dbg_lvl):
    try:
        for docker_prefix in docker_prefix_list:
            pattern = re.compile(fr"^{docker_prefix}_.*_{job_name}.*_.*_{user}$")
            dockers = subprocess.run(["docker", "ps", "-a", "--format", "{{.Names}}"], capture_output=True, text=True, check=True)
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

def remove_tmp_run_scripts(base_path, job_name, user, dbg_lvl):
    pattern = re.compile(rf".*_{re.escape(job_name)}_.*_{re.escape(user)}_tmp_run\.sh$")
    base = Path(base_path)
    if not base.is_dir():
        return

    removed_any = False
    for script_path in base.glob("*_tmp_run.sh"):
        if not pattern.match(script_path.name):
            continue
        try:
            script_path.unlink()
            info(f"Removed temporary run script {script_path}", dbg_lvl)
            removed_any = True
        except OSError as exc:
            warn(f"Failed to remove temporary run script {script_path}: {exc}", dbg_lvl)

    if not removed_any and dbg_lvl >= 3:
        info(f"No temporary run scripts found in {base}", dbg_lvl)

def get_image_list(simulations, workloads_data):
    image_list = []
    for simulation in simulations:
        suite = simulation["suite"]
        subsuite = simulation["subsuite"]
        workload = simulation["workload"]
        exp_cluster_id = simulation["cluster_id"]
        sim_mode = simulation["simulation_type"]

        if workload == None:
            if subsuite == None:
                for subsuite_ in workloads_data[suite].keys():
                    for workload_ in workloads_data[suite][subsuite_].keys():
                        predef_mode = workloads_data[suite][subsuite_][workload_]["simulation"]["prioritized_mode"]
                        sim_mode_ = sim_mode
                        if sim_mode_ == None:
                            sim_mode_ = predef_mode
                        if sim_mode_ in workloads_data[suite][subsuite_][workload_]["simulation"].keys() and workloads_data[suite][subsuite_][workload_]["simulation"][sim_mode_]["image_name"] not in image_list:
                            image_list.append(workloads_data[suite][subsuite_][workload_]["simulation"][sim_mode_]["image_name"])
            else:
                for workload_ in workloads_data[suite][subsuite].keys():
                    predef_mode = workloads_data[suite][subsuite][workload_]["simulation"]["prioritized_mode"]
                    sim_mode_ = sim_mode
                    if sim_mode_ == None:
                        sim_mode_ = predef_mode
                    if sim_mode_ in workloads_data[suite][subsuite][workload_]["simulation"].keys() and workloads_data[suite][subsuite][workload_]["simulation"][sim_mode_]["image_name"] not in image_list:
                        image_list.append(workloads_data[suite][subsuite][workload_]["simulation"][sim_mode_]["image_name"])
        else:
            if subsuite == None:
                for subsuite_ in workloads_data[suite].keys():
                    predef_mode = workloads_data[suite][subsuite_][workload]["simulation"]["prioritized_mode"]
                    sim_mode_ = sim_mode
                    if sim_mode_ == None:
                        sim_mode_ = predef_mode
                    if sim_mode_ in workloads_data[suite][subsuite_][workload]["simulation"].keys() and workloads_data[suite][subsuite_][workload]["simulation"][sim_mode_]["image_name"] not in image_list:
                        image_list.append(workloads_data[suite][subsuite_][workload]["simulation"][sim_mode_]["image_name"])
            else:
                predef_mode = workloads_data[suite][subsuite][workload]["simulation"]["prioritized_mode"]
                sim_mode_ = sim_mode
                if sim_mode_ == None:
                    sim_mode_ = predef_mode
                if sim_mode_ in workloads_data[suite][subsuite][workload]["simulation"].keys() and workloads_data[suite][subsuite][workload]["simulation"][sim_mode_]["image_name"] not in image_list:
                    image_list.append(workloads_data[suite][subsuite][workload]["simulation"][sim_mode_]["image_name"])

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

def prepare_trace(user, scarab_path, scarab_build, docker_home, job_name, infra_dir, docker_prefix_list, githash, interactive_shell=False, available_slurm_nodes=[], dbg_lvl=1):
    # prepare docker images
    try:
        for docker_prefix in docker_prefix_list:
            image_tag = f"{docker_prefix}:{githash}"
            prepare_docker_image(docker_prefix, image_tag, available_slurm_nodes, dbg_lvl)
    except subprocess.CalledProcessError as e:
        info(f"Docker image preparation failed: {e.stderr if isinstance(e.stderr, str) else e.stderr.decode() if e.stderr else str(e)}", dbg_lvl)
        raise e
    except Exception as e:
        info(f"Unexpected error during docker image preparation: {str(e)}", dbg_lvl)
        raise e

    try:
        scarab_githash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=scarab_path).decode("utf-8").strip()
    except Exception as e:
        err(f"Could not get scarab githash: {str(e)}", dbg_lvl)
        raise e

    info(f"Current scarab git hash: {scarab_githash}", dbg_lvl)

    if not os.path.exists(f"{infra_dir}/scarab_builds"):
        os.system(f"mkdir -p {infra_dir}/scarab_builds")

    docker_prefix = docker_prefix_list[0]
    docker_container_name = None
    try:
        local_uid = os.getuid()
        local_gid = os.getgid()

        trace_dir = f"{docker_home}/simpoint_flow/{job_name}"
        os.system(f"mkdir -p {trace_dir}/scarab/src/")

        # (Re)build the scarab binary first.
        rebuild_scarab(infra_dir, scarab_path, user, docker_home, docker_prefix, githash, scarab_githash, scarab_build, stream_build=False, dbg_lvl=dbg_lvl)

        # Copy current scarab binary to trace dir
        scarab_ver = f"{infra_dir}/scarab_builds/scarab_current"
        os.system(f"cp {scarab_ver} {trace_dir}/scarab/src/scarab")

        os.system(f"mkdir -p {trace_dir}/scarab/bin/scarab_globals")
        os.system(f"cp {scarab_path}/bin/scarab_launch.py  {trace_dir}/scarab/bin/scarab_launch.py ")
        os.system(f"cp {scarab_path}/bin/scarab_globals/*  {trace_dir}/scarab/bin/scarab_globals/ ")
        os.system(f"mkdir -p {trace_dir}/scarab/utils/memtrace")
        os.system(f"cp {scarab_path}/utils/memtrace/* {trace_dir}/scarab/utils/memtrace/ ")
    except subprocess.CalledProcessError as e:
        if docker_container_name:
            try:
                subprocess.run(["docker", "rm", "-f", docker_container_name], check=True)
                info(f"Removed container: {docker_container_name}", dbg_lvl)
            except subprocess.CalledProcessError:
                info(f"Could not remove container: {docker_container_name}", dbg_lvl)
        info(f"Scarab build failed: {e.stderr if isinstance(e.stderr, str) else e.stderr.decode() if e.stderr else str(e)}", dbg_lvl)
        raise e
    except Exception as e:
        if docker_container_name:
            try:
                subprocess.run(["docker", "rm", "-f", docker_container_name], check=True)
                info(f"Removed container: {docker_container_name}", dbg_lvl)
            except subprocess.CalledProcessError:
                info(f"Could not remove container: {docker_container_name}", dbg_lvl)
        info(f"Unexpected error during scarab build: {str(e)}", dbg_lvl)
        raise e

def finish_trace(user, descriptor_data, workload_db_path, infra_dir, dbg_lvl):
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
        trace_configs = descriptor_data["trace_configurations"]
        job_name = descriptor_data["trace_name"]
        trace_dir = f"{descriptor_data['root_dir']}/simpoint_flow/{job_name}"
        target_traces_dir = descriptor_data["traces_dir"]
        docker_home = descriptor_data["root_dir"]

        print("Copying the successfully collected traces and update workloads_db.json...")

        for config in trace_configs:
            workload = config['workload']
            suite = config['suite']
            subsuite = config['subsuite']

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
            memtrace_dict['segment_size'] = int(read_first_line(segment_size_file))

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

            target_traces_path = f"{target_traces_dir}/{suite}/{subsuite}/{workload}"
            # Copy successfully collected traces to target_traces_dir (simpoints are recorded in workloads_db.json)
            os.system(f"mkdir -p {target_traces_path}/traces/whole")
            os.system(f"mkdir -p {target_traces_path}/traces/simp")
            trace_clustering_info = read_descriptor_from_json(os.path.join(trace_dir, workload, "trace_clustering_info.json"), dbg_lvl)
            if config['trace_type'] == "trace_then_cluster":
                os.system(f"cp -r {trace_dir}/{workload}/traces_simp/* {target_traces_path}/traces/simp/")
                os.system(f"mkdir -p {target_traces_path}/traces/whole/")
                whole_trace_dir = trace_clustering_info['dr_folder']
                trace_file = trace_clustering_info['trace_file']
                subprocess.run([f"cp {trace_dir}/{workload}/traces/whole/{whole_trace_dir}/trace/{trace_file} {target_traces_path}/traces/whole/"], check=True, shell=True)
                memtrace_dict['warmup'] = 10000000
                memtrace_dict['whole_trace_file'] = trace_clustering_info['trace_file']
            elif config['trace_type'] == "cluster_then_trace":
                os.system(f"cp -r {trace_dir}/{workload}/traces_simp/trace/* {target_traces_path}/traces/simp/")
                memtrace_dict['warmup'] = 10000000
                memtrace_dict['whole_trace_file'] = None
                print("cluster_then_trace doesn't have a whole trace file.")
            else: # iterative_trace
                largest_traces = trace_clustering_info['trace_file']
                for trace_path in largest_traces:
                    print("Processing trace:", trace_path)
                    prefix = "traces_simp/"
                    if prefix in trace_path:
                        relative_part = trace_path.split(prefix, 1)[1]
                        timestep = trace_path.split("Timestep_")[1].split("/")[0]
                        trace_source = os.path.join(trace_dir, workload, "traces_simp", relative_part)
                        trace_dest_dir = os.path.join(target_traces_path, "traces/simp")
                        trace_dest = os.path.join(target_traces_path, "traces/simp", f"{timestep}.zip")

                        os.makedirs(os.path.dirname(trace_dest_dir), exist_ok=True)
                        os.system(f"cp -r {trace_source} {trace_dest}")
                memtrace_dict['warmup'] = 0
                memtrace_dict['whole_trace_file'] = None
            memtrace_dict['trace_type'] = config['trace_type']

            os.system(f"chmod a+w -R {target_traces_path}")
            simulation_dict['prioritized_mode'] = "memtrace"
            simulation_dict['exec'] = exec_dict
            simulation_dict['memtrace'] = memtrace_dict
            suite = config['suite']
            subsuite = config['subsuite'] if config['subsuite'] else suite
            workload_dict = {
                "trace":trace_dict,
                "simulation":simulation_dict,
                "simpoints":simpoints
            }

            if suite in workload_db_data.keys() and subsuite in workload_db_data[suite].keys() and workload in workload_db_data[suite][subsuite].keys():
                print("WARNING: workload name should be unique within a subsuite. db will be overwritten!")
            workload_db_data[suite][subsuite][workload] = workload_dict

        write_json_descriptor(workload_db_path, workload_db_data, dbg_lvl)
        extract_top_simpoints.modify_simpoints_in_place(workload_db_data)
        write_json_descriptor(f"{infra_dir}/workloads/workloads_top_simp.json", workload_db_data, dbg_lvl)

        print("Recover the ASLR setting with sudo. Provide password..")
        os.system("echo 2 | sudo tee /proc/sys/kernel/randomize_va_space")

    except Exception as e:
        raise e

def is_container_running(container_name, dbg_lvl):
    client = get_docker_client()
    if client is None:
        warn("Docker client unavailable; cannot inspect containers.", dbg_lvl)
        return False
    try:
        container = client.containers.get(container_name)
        info(f"container {container_name} is already running.", dbg_lvl)
        return container.status == "running"
    except Exception as exc:
        info(f"Failed to query container {container_name}: {exc}", dbg_lvl)
        return False

def count_interactive_shells(container_name, dbg_lvl):
    if docker is None:
        warn("Docker client unavailable; assuming 0 interactive shells.", dbg_lvl)
        return 0
    try:
        client_api = docker.APIClient()
    except Exception as exc:
        warn(f"Docker API client unavailable: {exc}", dbg_lvl)
        return 0
    try:
        container = client_api.inspect_container(container_name)
        container_id = container['Id']
        processes = client_api.top(container_id)

        shell_count = 0
        for process in processes['Processes']:
            cmd = ' '.join(process)
            # Check for common interactive shells
            if any(shell in cmd for shell in ['bash', 'sh', 'zsh']):
                shell_count += 1
        info(f"{shell_count} shells are running for {container_name}.", dbg_lvl)
        return shell_count
    except Exception as exc:
        print(f"Error checking shells for '{container_name}': {exc}")
        return 0

def image_exist(image_tag, node=None):
    try:
        output = run_on_node(["docker", "images", "-q", image_tag], node, capture_output=True, text=True)
        return bool(output.stdout.strip())
    except subprocess.CalledProcessError:
        return False

# Returns true if experiment exists
def check_sp_exist (descriptor_data, config_key, suite, subsuite, workload, exp_cluster_id):
    # Check if simpoint exists
    experiment_dir =  f"{descriptor_data['root_dir']}/simulations/{descriptor_data['experiment']}/"
    experiment_dir += f"{config_key}/{suite}/{subsuite}/{workload}/{exp_cluster_id}"

    print(experiment_dir)

    inst_stat_path = Path(experiment_dir) / "inst.stat.0.csv"
    if inst_stat_path.is_file():
        return True

    # Previous runs without a completed inst.stat.0.csv should be retried
    return False

# Returns true if experiment failed
def check_sp_failed (descriptor_data, config_key, suite, subsuite, workload, exp_cluster_id):
    # Check if simpoint exists
    experiment_dir =  f"{descriptor_data['root_dir']}/simulations/{descriptor_data['experiment']}/"
    experiment_dir += f"{config_key}/{suite}/{subsuite}/{workload}/{exp_cluster_id}"
    
    # Failed case; CSV files not generated. Ignoring .csv.warmup files.
    if len(list(filter(lambda x: x.endswith('.csv'), os.listdir(experiment_dir)))) == 0:
        return True
    
    # Success case
    return False

# Clean up failed run
def clean_failed_run (descriptor_data, config_key, suite, subsuite, workload, exp_cluster_id):
    # Remove failed run artifacts while preserving directory structure
    experiment_dir =  f"{descriptor_data['root_dir']}/simulations/{descriptor_data['experiment']}/"
    experiment_dir += f"{config_key}/{suite}/{subsuite}/{workload}/{exp_cluster_id}"

    experiment_path = Path(experiment_dir)
    patterns_to_clean = ["*.csv", "*.out", "*.in", "*.out.warmup", "sim.log"]

    try:
        if experiment_path.exists():
            for pattern in patterns_to_clean:
                for target in experiment_path.glob(pattern):
                    if target.is_file() or target.is_symlink():
                        target.unlink()
    except Exception as e:
        err(f"Error cleaning files in {experiment_dir}: {e}", 1)

    # Wipe log file
    log_dir =  f"{descriptor_data['root_dir']}/simulations/{descriptor_data['experiment']}/logs/"
    log_files = os.listdir(log_dir)
    for file in log_files:
        with open(os.path.join(log_dir, file), 'r') as f:
            lines = f.readlines()

            # Logfile will have {config} {suite}/{subsuite}/{workload} {simpoint} as header
            header = f"{config_key} {suite}/{subsuite}/{workload} {exp_cluster_id}"
            if header in lines:
                os.remove(os.path.join(log_dir, file))

# Check if run was already successful, and thus skippable
# Please use as follows:
# if check_can_skip(...):
#     continue
def check_can_skip (descriptor_data, config_key, suite, subsuite, workload, cluster_id, filename, sim_mode, user, slurm_queue=None, debug_lvl=1):
    # Check (re)run conditions 
    if check_sp_exist(descriptor_data, config_key, suite, subsuite, workload, cluster_id):
        # Previous run exists, check if it failed
        if not check_sp_failed(descriptor_data, config_key, suite, subsuite, workload, cluster_id):
            # Previous run exists and was successful
            info(f"Successful simulation with config {config_key} for workload {workload} already exists.", debug_lvl)
            return True
        
        # Previous run exists but failed
        info(f"Previous run with config {config_key} for workload {workload} failed. Cleaning directory and Re-running.", debug_lvl)
        
        clean_failed_run(descriptor_data, config_key, suite, subsuite, workload, cluster_id)
    else:
        # No previous run exists

        # Check if it is about to be run 
        if os.path.exists(filename):
            # Run script has generated run file, it will be run shortly.
            info(f"Run script for {config_key} for workload {workload} exists. Other script will run it.", debug_lvl)
            return True

        # If using slurm, check queue too
        if not slurm_queue is None:
            # Check each entry
            for entry in slurm_queue:
                # Check for following identifier. Should be of form <docker_prefix>_...as below..._<sim_mode>_<user>
                # Docker prefix and username checked in slurm_runner
                identifier = (
                    f"{suite}_{subsuite}_{workload}_{descriptor_data['experiment']}"
                    f"_{config_key.replace('/', '-')}_{cluster_id}_{sim_mode}_{user}"
                )
                if identifier in entry:
                    # Job is in the queue, it will be run shortly.
                    info(f"Job for {config_key} for workload {workload} is in the queue. Other script will run it.", debug_lvl)
                    return True
        
        info(f"Running simulation with config {config_key} for workload {workload}", debug_lvl)
        
    return False
    
def generate_table(data, title=""):
    """
    Generates a formatted table as a string, handling potential formatting issues
    with varying integer sizes.

    Args:
        data (dict): A dictionary containing the table data.  The keys of the
            dictionary are the column headers, and the values are lists
            representing the data for each column.  It is assumed that all
            lists have the same length.
        title (str, optional): An optional title for the table. Defaults to "".

    Returns:
        str: A string representing the formatted table.
    """
    
    if not data:
        return "No data provided."

    headers = list(data.keys())
    num_cols = len(headers)
    num_rows = len(data[headers[0]])

    # Calculate maximum width for each column based on header and data lengths
    column_widths = [len(header) for header in headers]
    for i in range(num_cols):
        for j in range(num_rows):
            column_widths[i] = max(column_widths[i], len(str(data[headers[i]][j])))

    # Create the table format string
    format_string = " | ".join(f"{{:<{width}}}" for width in column_widths)
    separator = "-" * (sum(column_widths) + 3 * (num_cols - 1))

    table_string = ""
    if title:
        table_string += f"{title.center(len(separator))}\n"

    # Add the header row
    table_string += format_string.format(*headers) + "\n"
    table_string += separator + "\n"

    # Add the data rows
    for j in range(num_rows):
        row_data = [data[header][j] for header in headers]
        table_string += format_string.format(*row_data) + "\n"

    return table_string
