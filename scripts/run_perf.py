#!/usr/bin/python3

# 02/17/2025 | Surim Oh | run_perf.py
# An entrypoint script that run clustering and tracing on Slurm clusters using docker containers or on local

import subprocess
import argparse
import os
import docker

from utilities import (
        info,
        err,
        read_descriptor_from_json
        )

client = docker.from_env()

def open_interactive_shell(user, docker_home, image_name, infra_dir, dbg_lvl = 1):
    try:
        # Get GitHash
        try:
            githash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("utf-8").strip()
            info(f"Git hash: {githash}", dbg_lvl)
        except FileNotFoundError:
            err("Error: 'git' command not found. Make sure Git is installed and in your PATH.")
        except subprocess.CalledProcessError:
            err("Error: Not in a Git repository or unable to retrieve Git hash.")

        docker_container_name = f"{image_name}_perf_{user}"
        try:
            command = f"docker run --privileged \
                    -dit \
                    --name {docker_container_name} \
                    --mount type=bind,source={docker_home},target=/home/{user} \
                    {image_name}:{githash} \
                    /bin/bash"
            print(command)
            os.system(command)
            os.system(f"docker cp {infra_dir}/scripts/utilities.sh {docker_container_name}:/usr/local/bin")
            os.system(f"docker cp {infra_dir}/common/scripts/common_entrypoint.sh {docker_container_name}:/usr/local/bin")
            os.system(f"docker cp {infra_dir}/common/scripts/perf_entrypoint.sh {docker_container_name}:/usr/local/bin")
            os.system(f"docker exec --privileged {docker_container_name} /bin/bash -c '/usr/local/bin/common_entrypoint.sh'")
            os.system(f"docker exec --privileged {docker_container_name} /bin/bash -c '/usr/local/bin/perf_entrypoint.sh'")
            subprocess.run(["docker", "exec", "-it", f"--user={user}", f"--workdir=/tmp_home", docker_container_name, "/bin/bash"])
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
    parser = argparse.ArgumentParser(description='Run perf on local')

    # Add arguments
    parser.add_argument('-d','--descriptor_name', required=True, help='Perf descriptor name. Usage: -d perf.json')
    parser.add_argument('-l','--launch', required=False, default=False, action=argparse.BooleanOptionalAction, help='Launch a docker container on a node for the purpose of running perf where the environment is for the experiment described in a descriptor.')
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

   # Read descriptor json and extract important data
    descriptor_data = read_descriptor_from_json(descriptor_path, dbg_lvl)
    user = descriptor_data["user"]
    root_dir = descriptor_data["root_dir"]
    image_name = descriptor_data["image_name"]

    if args.launch:
        open_interactive_shell(user, root_dir, image_name, infra_dir, dbg_lvl)
        exit(0)
