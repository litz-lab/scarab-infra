#!/usr/bin/python3

# 02/17/2025 | Surim Oh | run_perf.py
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
    is_container_running,
    count_interactive_shells
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
            if is_container_running(docker_container_name, dbg_lvl):
                subprocess.run(["docker", "exec", "-it", f"--user={user}", f"--workdir=/tmp_home", docker_container_name, "/bin/bash"])
            else:
                info(f"Create a new container for the interactive mode", dbg_lvl)
                command = f"docker run --privileged \
                        -dit \
                        --name {docker_container_name} \
                        --mount type=bind,source={docker_home},target=/home/{user},readonly=false \
                        {image_name}:{githash} \
                        /bin/bash"
                print(command)
                os.system(command)
                os.system(f"docker cp {infra_dir}/scripts/utilities.sh {docker_container_name}:/usr/local/bin")
                os.system(f"docker cp {infra_dir}/common/scripts/root_entrypoint.sh {docker_container_name}:/usr/local/bin")
                os.system(f"docker cp {infra_dir}/common/scripts/perf_entrypoint.sh {docker_container_name}:/usr/local/bin")
                os.system(f"docker exec --privileged {docker_container_name} /bin/bash -c '/usr/local/bin/root_entrypoint.sh'")
                os.system(f"docker exec --privileged {docker_container_name} /bin/bash -c '/usr/local/bin/perf_entrypoint.sh'")
                subprocess.run(["docker", "exec", "-it", f"--user={user}", f"--workdir=/tmp_home", docker_container_name, "/bin/bash"])
        except KeyboardInterrupt:
            if count_interactive_shells(docker_container_name, dbg_lvl) == 1:
                os.system(f"docker rm -f {docker_container_name}")
                print("Recover the ASLR setting with sudo. Provide password..")
                os.system("echo 2 | sudo tee /proc/sys/kernel/randomize_va_space")
            return
        finally:
            try:
                if count_interactive_shells(docker_container_name, dbg_lvl) == 1:
                    client.containers.get(docker_container_name).remove(force=True)
                    print(f"Container {docker_container_name} removed.")
            except docker.errors.NotFound:
                print(f"Container {docker_container_name} not found.")
    except Exception as e:
        raise e


def run_perf_command(descriptor_path, action, dbg_lvl=2, infra_dir=None):
    if infra_dir is None:
        infra_dir = subprocess.check_output(["pwd"]).decode("utf-8").split("\n")[0]

    descriptor_data = read_descriptor_from_json(descriptor_path, dbg_lvl)
    if descriptor_data is None:
        raise RuntimeError(f"Failed to read descriptor {descriptor_path}")

    user = descriptor_data.get("user")
    root_dir = descriptor_data.get("root_dir")
    image_name = descriptor_data.get("image_name")

    if action == "launch":
        open_interactive_shell(user, root_dir, image_name, infra_dir, dbg_lvl)
        return 0

    raise RuntimeError(f"Unsupported perf action: {action}")


def main():
    parser = argparse.ArgumentParser(description='Run perf on local')

    parser.add_argument('-d','--descriptor_name', required=True, help='Perf descriptor name. Usage: -d perf.json')
    parser.add_argument('-l','--launch', required=False, default=False, action=argparse.BooleanOptionalAction, help='Launch a docker container for perf descriptor.')
    parser.add_argument('-dbg','--debug', required=False, type=int, default=2, help='1 for errors, 2 for warnings, 3 for info')
    parser.add_argument('-si','--scarab_infra', required=False, default=None, help='Path to scarab infra repo to launch new containers')

    args = parser.parse_args()
    descriptor_path = args.descriptor_name
    dbg_lvl = args.debug
    infra_dir = args.scarab_infra

    action = "launch" if args.launch else "launch"
    return run_perf_command(descriptor_path, action, dbg_lvl=dbg_lvl, infra_dir=infra_dir)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as exc:
        err(str(exc), 1)
        sys.exit(1)
