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

def sync_perf_support_files(docker_container_name, image_name, infra_dir):
    common_files = [
        (f"{infra_dir}/scripts/utilities.sh", "/usr/local/bin"),
        (f"{infra_dir}/common/scripts/root_entrypoint.sh", "/usr/local/bin"),
        (f"{infra_dir}/common/scripts/user_entrypoint.sh", "/usr/local/bin"),
        (f"{infra_dir}/common/scripts/perf_entrypoint.sh", "/usr/local/bin"),
    ]
    for src, dst in common_files:
        subprocess.run(["docker", "cp", src, f"{docker_container_name}:{dst}"], check=True)

    for script in ("workload_root_entrypoint.sh", "workload_user_entrypoint.sh"):
        src = f"{infra_dir}/workloads/{image_name}/{script}"
        if os.path.exists(src):
            subprocess.run(["docker", "cp", src, f"{docker_container_name}:/usr/local/bin"], check=True)


def perf_container_initialized(docker_container_name):
    result = subprocess.run(
        ["docker", "exec", docker_container_name, "test", "-f", "/tmp_home/.scarab_perf_ready"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def bootstrap_perf_container(docker_container_name):
    subprocess.run(
        ["docker", "exec", "--privileged", docker_container_name, "/bin/bash", "-c", "/usr/local/bin/root_entrypoint.sh"],
        check=True,
    )
    subprocess.run(
        ["docker", "exec", "--privileged", docker_container_name, "/bin/bash", "-c", "/usr/local/bin/perf_entrypoint.sh"],
        check=True,
    )
    subprocess.run(
        ["docker", "exec", "--privileged", docker_container_name, "/bin/bash", "-c", "touch /tmp_home/.scarab_perf_ready"],
        check=True,
    )


def open_interactive_shell(user, docker_home, image_name, infra_dir, dbg_lvl = 1):
    try:
        local_uid = os.getuid()
        local_gid = os.getgid()
        container_home = "/tmp_home" if user == "root" else f"/home/{user}"

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
            shell_cmd = f"export HOME={container_home}; source /usr/local/bin/user_entrypoint.sh >/dev/null 2>&1 || true; exec /bin/bash"
            needs_bootstrap = False
            if is_container_running(docker_container_name, dbg_lvl):
                sync_perf_support_files(docker_container_name, image_name, infra_dir)
                needs_bootstrap = not perf_container_initialized(docker_container_name)
            else:
                info(f"Create a new container for the interactive mode", dbg_lvl)
                command = f"docker run --privileged \
                        -e user_id={local_uid} \
                        -e group_id={local_gid} \
                        -e username={user} \
                        -e HOME={container_home} \
                        -e APP_GROUPNAME={image_name} \
                        -e APPNAME={image_name} \
                        -dit \
                        --name {docker_container_name} \
                        --mount type=bind,source={docker_home},target=/home/{user},readonly=false \
                        {image_name}:{githash} \
                        /bin/bash"
                print(command)
                os.system(command)
                sync_perf_support_files(docker_container_name, image_name, infra_dir)
                needs_bootstrap = True
            if needs_bootstrap:
                bootstrap_perf_container(docker_container_name)
            subprocess.run([
                "docker", "exec", "-it", f"--user={user}", f"--workdir=/tmp_home",
                docker_container_name, "/bin/bash", "-c", shell_cmd
            ])
        except KeyboardInterrupt:
            if count_interactive_shells(docker_container_name, dbg_lvl) == 1:
                os.system(f"docker rm -f {docker_container_name}")
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
