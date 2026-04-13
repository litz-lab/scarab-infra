#!/usr/bin/python3

# 02/17/2025 | Surim Oh | run_perf.py
# An entrypoint script that run clustering and tracing on Slurm clusters using docker containers or on local

import json
import re
import subprocess
import argparse
import os
import sys
import docker

from .utilities import (
    info,
    err,
    read_descriptor_from_json,
    write_json_descriptor,
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


# ---------------------------------------------------------------------------
# Automated perf collection helpers
# ---------------------------------------------------------------------------

def parse_toplev_output(stdout, stderr):
    """Parse toplev.py -l1 human-readable output for top-down L1 percentages.

    The --single-thread stderr format has lines like:
        FE               Frontend_Bound   % Slots                       35.1   [50.0%]<==
        BAD              Bad_Speculation  % Slots                       16.6   [50.0%]
        RET              Retiring         % Slots                       28.5   [50.0%]
        BE               Backend_Bound    % Slots                       19.8   [50.0%]

    Only categories exceeding their threshold are printed; missing ones get None.
    """
    key_map = {
        "frontend_bound": "frontend_bound",
        "backend_bound": "backend_bound",
        "bad_speculation": "bad_speculation",
        "retiring": "retiring",
    }

    combined = (stdout or "") + "\n" + (stderr or "")

    # Match: MetricName <whitespace> % Slots <whitespace> VALUE
    pattern = re.compile(
        r'(Frontend_Bound|Backend_Bound|Bad_Speculation|Retiring)\s+%\s*Slots\s+([\d.]+)',
        re.IGNORECASE,
    )

    topdown = {}
    for m in pattern.finditer(combined):
        metric = m.group(1).lower()
        value = float(m.group(2))
        db_key = key_map.get(metric)
        if db_key:
            topdown[db_key] = round(value, 1)

    # The four L1 categories sum to ~100%. Infer missing ones from the rest.
    all_keys = ["retiring", "bad_speculation", "frontend_bound", "backend_bound"]
    present = {k: v for k, v in topdown.items() if k in all_keys}
    missing = [k for k in all_keys if k not in present]
    if 1 <= len(missing) <= 2 and len(present) >= 2:
        remainder = round(100.0 - sum(present.values()), 1)
        if len(missing) == 1:
            topdown[missing[0]] = remainder
        # If 2 are missing we can't split them; leave as None

    return topdown


def parse_perf_stat_time(stderr_text):
    """Parse perf stat stderr for elapsed seconds.

    Looks for a line like:
        1.234567890 seconds time elapsed
    """
    if not stderr_text:
        return None
    match = re.search(r"([\d]+\.[\d]+)\s+seconds\s+time\s+elapsed", stderr_text)
    if match:
        return float(match.group(1))
    return None


def _parse_peak_rss_mb(stderr_text):
    """Parse warmup stderr for peak RSS (MB int) or None.

    Two formats are supported:
      1. Our python3 RUSAGE_CHILDREN wrapper (default; see run_perf_for_descriptor):
             PEAK_RSS_KB=<n>
      2. GNU `/usr/bin/time -v` (legacy; only if `time` package is installed in
         the container):
             Maximum resident set size (kbytes): <n>
    Both report kB from getrusage(RUSAGE_CHILDREN).ru_maxrss, i.e. the
    high-water mark of the workload's RSS (VmHWM).
    """
    if not stderr_text:
        return None
    match = re.search(r"PEAK_RSS_KB=(\d+)", stderr_text)
    if not match:
        match = re.search(r"Maximum resident set size \(kbytes\):\s+(\d+)", stderr_text)
    if not match:
        return None
    kb = int(match.group(1))
    return int(round(kb / 1024))


TOPLEV_PATH = "/tmp_home/pmu-tools/toplev.py"
PERF_CORE = 10  # Pin workloads to a high-numbered core to reduce interference
PERF_REPEAT = 15  # Re-run the entire workload binary this many times per measurement


PERF_SCRIPT = "/tmp/_perf_repeat.sh"


def _write_repeat_script(container_name, binary_cmd):
    """Write a repeat-loop script into the container to avoid quoting issues."""
    script = f"#!/bin/bash\nfor i in $(seq 1 {PERF_REPEAT}); do\n  {binary_cmd}\ndone\n"
    subprocess.run(
        ["docker", "exec", container_name, "/bin/bash", "-c",
         f"cat > {PERF_SCRIPT} << 'PERF_EOF'\n{script}PERF_EOF\nchmod +x {PERF_SCRIPT}"],
        check=True, capture_output=True,
    )


def run_toplev_for_config(container_name, config):
    """Run toplev.py -l1 --single-thread on a pinned core via docker exec --privileged."""
    binary_cmd = config["binary_cmd"]
    _write_repeat_script(container_name, binary_cmd)
    env_str = ""
    if config.get("env_vars"):
        env_str = " ".join(f"{k}={v}" for k, v in config["env_vars"].items()) + " "
    cmd = [
        "docker", "exec", "--privileged", container_name,
        "/bin/bash", "-c",
        f"{env_str}taskset -c {PERF_CORE} python3 {TOPLEV_PATH} -l1 --single-thread --core C{PERF_CORE}"
        f" --no-desc --verbose"
        f" -- taskset -c {PERF_CORE} {PERF_SCRIPT}"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"    [toplev] returncode={result.returncode}")
        if result.stderr:
            for line in result.stderr.strip().splitlines()[-5:]:
                print(f"    [toplev stderr] {line}")
    topdown = parse_toplev_output(result.stdout, result.stderr)
    if not topdown:
        print(f"    [toplev] WARNING: no top-down metrics parsed")
    return topdown


def run_perf_stat_for_config(container_name, config):
    """Run perf stat on a pinned core via docker exec --privileged."""
    binary_cmd = config["binary_cmd"]
    _write_repeat_script(container_name, binary_cmd)
    env_str = ""
    if config.get("env_vars"):
        env_str = " ".join(f"{k}={v}" for k, v in config["env_vars"].items()) + " "
    cmd = [
        "docker", "exec", "--privileged", container_name,
        "/bin/bash", "-c",
        f"{env_str}perf stat -C {PERF_CORE} -- taskset -c {PERF_CORE} {PERF_SCRIPT}"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"    [perf stat] returncode={result.returncode}")
        if result.stderr:
            for line in result.stderr.strip().splitlines()[-5:]:
                print(f"    [perf stat stderr] {line}")
    elapsed = parse_perf_stat_time(result.stderr)
    if elapsed is None:
        print(f"    [perf stat] WARNING: could not parse elapsed time")
    return elapsed


def collect_perf_data(user, root_dir, image_name, infra_dir, perf_configs, dbg_lvl=2):
    """Orchestrator: create/reuse container, run toplev + perf stat for each config,
    write results to workloads_db.json."""
    local_uid = os.getuid()
    local_gid = os.getgid()
    container_home = "/tmp_home" if user == "root" else f"/home/{user}"

    try:
        githash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("utf-8").strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        raise RuntimeError("Cannot determine git hash for docker image tag")

    docker_container_name = f"{image_name}_perf_collect_{user}"
    created_container = False

    try:
        # Create or reuse container
        if is_container_running(docker_container_name, dbg_lvl):
            sync_perf_support_files(docker_container_name, image_name, infra_dir)
            if not perf_container_initialized(docker_container_name):
                bootstrap_perf_container(docker_container_name)
        else:
            info(f"Creating container for perf collection: {docker_container_name}", dbg_lvl)
            command = (
                f"docker run --privileged "
                f"-e user_id={local_uid} "
                f"-e group_id={local_gid} "
                f"-e username={user} "
                f"-e HOME={container_home} "
                f"-e APP_GROUPNAME={image_name} "
                f"-e APPNAME={image_name} "
                f"-dit "
                f"--name {docker_container_name} "
                f"--mount type=bind,source={root_dir},target=/home/{user},readonly=false "
                f"{image_name}:{githash} "
                f"/bin/bash"
            )
            info(command, dbg_lvl)
            os.system(command)
            created_container = True
            sync_perf_support_files(docker_container_name, image_name, infra_dir)
            bootstrap_perf_container(docker_container_name)

        # Validate that the pinned core exists inside the container
        core_check = subprocess.run(
            ["docker", "exec", docker_container_name, "/bin/bash", "-c",
             f"taskset -c {PERF_CORE} true"],
            capture_output=True, text=True,
        )
        if core_check.returncode != 0:
            raise RuntimeError(
                f"Core {PERF_CORE} is not available in container. "
                f"Check PERF_CORE setting or container --cpuset-cpus. "
                f"stderr: {core_check.stderr.strip()}"
            )
        print(f"  Pinning all workloads to core {PERF_CORE}, repeating {PERF_REPEAT}x per measurement")

        # Load workloads_db.json
        workload_db_path = os.path.join(infra_dir, "workloads", "workloads_db.json")
        workload_db = read_descriptor_from_json(workload_db_path, dbg_lvl)
        if workload_db is None:
            workload_db = {}

        # Run toplev + perf stat for each config
        for config in perf_configs:
            workload = config["workload"]
            suite = config["suite"]
            subsuite = config["subsuite"]
            print(f"  Collecting perf data for {workload}...")

            # Warmup: run once to populate FS cache, shared libs, bytecode
            # caches. We also capture the workload's peak RSS (ru_maxrss) via
            # a small python3 wrapper that execs taskset+binary_cmd as a child
            # and then reads RUSAGE_CHILDREN. Prints a PEAK_RSS_KB=<n> sentinel
            # on its own line to stderr. We avoid GNU time (`/usr/bin/time -v`)
            # so we don't depend on the `time` apt package being present in
            # every workload image. The captured value is stored in
            # workloads_db.json under performance.peak_rss_mb and later
            # consumed by slurm_runner.estimate_trace_mem_mb() to size `--mem`
            # for sbatch trace jobs. Effectively free: this run already happens.
            binary_cmd = config["binary_cmd"]
            rss_wrapper = (
                "python3 -c '"
                "import os,sys,subprocess,resource;"
                "cmd=sys.argv[1];"
                "r=subprocess.run([\"/bin/bash\",\"-c\",cmd],"
                "stdout=subprocess.DEVNULL);"
                "u=resource.getrusage(resource.RUSAGE_CHILDREN);"
                "sys.stderr.write(f\"PEAK_RSS_KB={u.ru_maxrss}\\n\");"
                "sys.exit(r.returncode)' "
                f'"taskset -c {PERF_CORE} {binary_cmd}"'
            )
            warmup_res = subprocess.run(
                ["docker", "exec", "--privileged", docker_container_name,
                 "/bin/bash", "-c", rss_wrapper],
                capture_output=True, text=True, timeout=300,
            )
            peak_rss_mb = _parse_peak_rss_mb(warmup_res.stderr)

            topdown = run_toplev_for_config(docker_container_name, config)
            elapsed = run_perf_stat_for_config(docker_container_name, config)

            elapsed_min = 0
            elapsed_sec = elapsed
            if elapsed is not None and elapsed >= 60:
                elapsed_min = int(elapsed // 60)
                elapsed_sec = round(elapsed % 60, 2)

            performance = {
                "topdown": {
                    "retiring": topdown.get("retiring"),
                    "bad_speculation": topdown.get("bad_speculation"),
                    "frontend_bound": topdown.get("frontend_bound"),
                    "backend_bound": topdown.get("backend_bound"),
                },
                "execution_time": {
                    "min": elapsed_min,
                    "sec": elapsed_sec,
                },
                "peak_rss_mb": peak_rss_mb,
            }

            # Write to workloads_db at [suite][subsuite][workload]["performance"]
            if suite not in workload_db:
                workload_db[suite] = {}
            if subsuite not in workload_db[suite]:
                workload_db[suite][subsuite] = {}
            if workload not in workload_db[suite][subsuite]:
                workload_db[suite][subsuite][workload] = {}
            workload_db[suite][subsuite][workload]["performance"] = performance

            print(f"    topdown: {topdown}")
            print(f"    elapsed: {elapsed}s")
            if peak_rss_mb is not None:
                print(f"    peak_rss: {peak_rss_mb} MB")
            else:
                print(f"    peak_rss: (not captured)")

        write_json_descriptor(workload_db_path, workload_db, dbg_lvl)
        print(f"Results written to {workload_db_path}")

    finally:
        # Cleanup container if we created it
        if created_container:
            try:
                client.containers.get(docker_container_name).remove(force=True)
                print(f"Container {docker_container_name} removed.")
            except docker.errors.NotFound:
                pass


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

    if action == "collect":
        perf_configs = descriptor_data.get("perf_configurations")
        if not perf_configs:
            raise RuntimeError("Descriptor has no perf_configurations for collection")
        collect_perf_data(user, root_dir, image_name, infra_dir, perf_configs, dbg_lvl)
        return 0

    raise RuntimeError(f"Unsupported perf action: {action}")


def main():
    parser = argparse.ArgumentParser(description='Run perf on local')

    parser.add_argument('-d','--descriptor_name', required=True, help='Perf descriptor name. Usage: -d perf.json')
    parser.add_argument('-l','--launch', required=False, default=False, action=argparse.BooleanOptionalAction, help='Launch a docker container for perf descriptor.')
    parser.add_argument('-c','--collect', required=False, default=False, action=argparse.BooleanOptionalAction, help='Run automated perf collection for all configurations.')
    parser.add_argument('-dbg','--debug', required=False, type=int, default=2, help='1 for errors, 2 for warnings, 3 for info')
    parser.add_argument('-si','--scarab_infra', required=False, default=None, help='Path to scarab infra repo to launch new containers')

    args = parser.parse_args()
    descriptor_path = args.descriptor_name
    dbg_lvl = args.debug
    infra_dir = args.scarab_infra

    if args.collect:
        action = "collect"
    else:
        action = "launch"
    return run_perf_command(descriptor_path, action, dbg_lvl=dbg_lvl, infra_dir=infra_dir)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as exc:
        err(str(exc), 1)
        sys.exit(1)
