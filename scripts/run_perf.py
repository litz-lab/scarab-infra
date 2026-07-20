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


def parse_perf_counters(stderr_text):
    """Parse `perf stat` default output for raw event counts.

    Lines look like (numbers may carry locale thousands separators):
         1,688,230,287      instructions              #    0.33  insn per cycle
         5,073,271,674      cycles
           100,375,188      L1-icache-load-misses
    Events reported as <not counted>/<not supported> are skipped.
    Returns {event_name: int_count} dictionary.
    """
    counts = {}
    if not stderr_text:
        return counts
    for line in stderr_text.splitlines():
        m = re.match(r"\s*([\d,]+)\s+([A-Za-z0-9._:-]+)", line)
        if m:
            counts[m.group(2)] = int(m.group(1).replace(",", ""))
    return counts


def parse_perf_min_coverage(text):
    """Return the minimum per-event counter coverage % in perf-stat output.

    Coverage is a fraction of the measurement window a counter was actually running
    (under multiplexing the rest is time-shared away and perf extrapolates it).
    perf stat omits the "(NN.NN%)" token when an event ran at 100% (no multiplexing).
    """
    if not text:
        return None
    # perf marks a multiplexed event with a trailing "(NN.NN%)" coverage token.
    pcts = [float(m) for m in re.findall(r"\((\d+\.\d+)%\)", text)]
    if pcts:
        return min(pcts)
    # If there are no markers but real count lines are present, coverage is full.
    if re.search(r"^\s*[\d,]+\s+\S", text, re.MULTILINE):
        return 100.0
    return None


def compute_hw_metrics(counts, repeat_count, min_coverage_pct=None):
    """
    Derive standard counter metrics from raw counts. The ratio metrics are
    independent of the repeat count (deterministic workload), so no per-run
    division is needed. `instructions` is the per-run dynamic instruction count.
    Any metric whose source event is missing/<not counted> comes back None.
    `min_coverage_pct` is a data-quality flag for the whole set.
    """
    # get raw total values
    ins  = counts.get("instructions")
    cyc  = counts.get("cycles")
    br   = counts.get("branches")
    brm  = counts.get("branch-misses")
    l1i  = counts.get("L1-icache-load-misses")
    l1d  = counts.get("L1-dcache-load-misses")
    l2   = counts.get("l2_rqsts.miss")
    l3   = counts.get("LLC-load-misses")
    itlb = counts.get("iTLB-load-misses")
    dtlb = counts.get("dTLB-load-misses")

    mpki = lambda x: round(x / ins * 1000, 3) if (x is not None and ins) else None
    mpmi = lambda x: round(x / ins * 1e6, 1) if (x is not None and ins) else None
    return {
        "instructions": int(ins / repeat_count) if (ins and repeat_count) else None,
        "ipc": round(ins / cyc, 3) if (ins and cyc) else None,
        "branch_pct": round(br / ins * 100, 2) if (br is not None and ins) else None,
        "branch_mispred_rate": round(brm / br * 100, 3) if (brm is not None and br) else None,
        "branch_mpki": mpki(brm),
        "l1i_mpki": mpki(l1i),
        "l1d_mpki": mpki(l1d),
        "l2_mpki": mpki(l2),
        "l3_mpki": mpki(l3),
        "itlb_mpmi": mpmi(itlb),
        "dtlb_mpmi": mpmi(dtlb),
        "min_coverage_pct": min_coverage_pct,
    }


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

# The descriptor json can overwrite these values.
# Setting PERF_TARGET_SECONDS huge will force exactly PERF_REPEAT_DEFAULT repeats regardless of runtime.
PERF_CORE = 10  # Pin workloads to a high-numbered core to reduce interference
PERF_REPEAT_DEFAULT = 15  # Re-run the entire workload binary this many times per measurement
PERF_TARGET_SECONDS = 600  # Target total measurement time; scale repeats to stay near this

# Full HW-counter set collected in the same perf-stat pass as elapsed time, yielding
# IPC, branch %, branch mispredict rate, branch MPKI, L1I/L1D/L2/L3 MPKI, and
# iTLB/dTLB MPMI; metrics the TMA top-down percentages do not capture.
PERF_STAT_EVENTS = ','.join([
    'instructions',
    'cycles',
    'branches',
    'branch-misses',
    'L1-icache-load-misses',
    'L1-dcache-load-misses',
    'l2_rqsts.miss',
    'LLC-load-misses',
    'iTLB-load-misses',
    'dTLB-load-misses',
])


PERF_SCRIPT = "/tmp/_perf_repeat.sh"



def _write_repeat_script(container_name, binary_cmd, repeat_count):
    """Write a repeat-loop script into the container to avoid quoting issues."""
    assert isinstance(repeat_count, int) and repeat_count > 0, \
        f"repeat_count must be a positive int, got {repeat_count!r}"
    n = repeat_count
    script = (
        f"#!/bin/bash\n"
        f"source /usr/local/bin/user_entrypoint.sh\n"
        f"for i in $(seq 1 {n}); do\n  {binary_cmd}\ndone\n"
    )
    subprocess.run(
        ["docker", "exec", container_name, "/bin/bash", "-c",
         f"cat > {PERF_SCRIPT} << 'PERF_EOF'\n{script}PERF_EOF\nchmod +x {PERF_SCRIPT}"],
        check=True, capture_output=True,
    )


# ===== toplev =====

def run_toplev_for_config(container_name, config, repeat_count):
    """Run toplev.py -l1 --single-thread on a pinned core via docker exec --privileged."""
    binary_cmd = config["binary_cmd"]
    _write_repeat_script(container_name, binary_cmd, repeat_count)
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
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        print(f"    [toplev] returncode={result.returncode}")
        if result.stderr:
            for line in result.stderr.strip().splitlines()[-5:]:
                print(f"    [toplev stderr] {line}")
    topdown = parse_toplev_output(result.stdout, result.stderr)
    if not topdown:
        print(f"    [toplev] WARNING: no top-down metrics parsed")
    return topdown


# ===== Helper functions for perf stat =====

def _read_nmi_watchdog(container_name):
    """
    Return the host's kernel.nmi_watchdog value (0/1) as seen from the container,
    or None if it can't be read. The watchdog occupies a fixed counter, which
    pushes `cycles` onto a GP counter and can force multiplexing.
    """
    result = subprocess.run(
        ["docker", "exec", "--privileged", container_name,
         "cat", "/proc/sys/kernel/nmi_watchdog"],
        capture_output=True, text=True,
    )
    try:
        return int((result.stdout or "").strip())
    except ValueError:
        return None


def _read_smt_active(container_name):
    """
    Return the host's SMT-active flag (0/1) as seen from the container, or None.
    SMT/HT halves the per-thread GP counters, the primary cause of multiplexing.
    """
    result = subprocess.run(
        ["docker", "exec", "--privileged", container_name,
         "cat", "/sys/devices/system/cpu/smt/active"],
        capture_output=True, text=True,
    )
    try:
        return int((result.stdout or "").strip())
    except ValueError:
        return None


def _probe_counter_fit(container_name, events):
    """
    Short perf-stat probe on a trivial workload. Returns the minimum counter
    coverage percentage across the events (100.0 == all fit, no multiplexing),
    or None if it could not be determined (e.g. perf/events unavailable).
    """
    cmd = [
        "docker", "exec", "--privileged", container_name, "/bin/bash", "-c",
        f"perf stat -C {PERF_CORE} -e {events} "
        f"-- taskset -c {PERF_CORE} bash -c 'timeout 2 yes > /dev/null'",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return None
    text = (result.stderr or "") + "\n" + (result.stdout or "")
    return parse_perf_min_coverage(text)


def _prompt_continue_or_abort(question):
    """Interactive y/N prompt. Aborts (returns False) on non-interactive stdin."""
    try:
        resp = input(f"  {question} [y/N] ").strip().lower()
    except EOFError:
        print("  (non-interactive stdin; aborting)")
        return False
    return resp in ("y", "yes")


def check_counter_budget(container_name):
    """
    Run once before collection. If the PERF_STAT_EVENTS set fits without multiplexing,
    the counter budget is fine regardless of SMT/watchdog, so we skip those checks.
    Only when it multiplexes do we investigate the cause; SMT first, which halves the
    GP counters, then the NMI watchdog, and prompt the user to continue or abort.

    Returns True to proceed, False to abort.
    """
    n_events = len(PERF_STAT_EVENTS.split(","))

    min_pct = _probe_counter_fit(container_name, PERF_STAT_EVENTS)
    if min_pct is None:
        # Probe inconclusive (perf missing, or none of the events resolved). The real
        # collection would likely also fail, leaving hw_metrics null. Let the user
        # decide whether that's worth a full run or better to abort and fix perf first.
        print(f"  [perf] WARNING: could not probe counter fit for the {n_events}-event set "
              f"(perf or events unavailable?); hw_metrics will likely be null.")
        return _prompt_continue_or_abort("Continue without verified HW counters anyway?")
    if min_pct >= 99.5:
        # Fits -> counter budget is sufficient; SMT/watchdog are irrelevant, skip them.
        print(f"  [perf] counter probe OK: all {n_events} events fit at ~{min_pct:.0f}%.")
        return True

    # Multiplexing happend: more events than usable counters, so perf time-shares and
    # scales them. The ratios stay roughly right but noisier. Investigate the cause (SMT
    # is the primary lever, then the watchdog) and let the user continue or abort.
    print(f"  [perf] WARNING: the {n_events}-event set MULTIPLEXES on core {PERF_CORE} "
          f"(min coverage {min_pct:.1f}%). hw_metrics ratios will be noisier.")
    print("         Cause: fewer than 8 usable GP counters. Fixes:")
    if _read_smt_active(container_name) == 1:
        print("           * SMT/HT is ON (halves per-thread GP counters).")
        print("             Disable it in the BIOS or boot with 'nosmt', then re-run.")
        # Note: runtime 'smt/control=off' does NOT reclaim the counters.
    if _read_nmi_watchdog(container_name) == 1:
        print("           * NMI watchdog is ON (occupies a counter). Free it on the HOST:")
        print("             $ sudo sysctl kernel.nmi_watchdog=0")
    return _prompt_continue_or_abort("Continue with multiplexed counters anyway?")


# ===== perf stat =====

def run_perf_stat_for_config(container_name, config, repeat_count):
    """Run perf stat on a pinned core via docker exec --privileged.

    Collects elapsed time AND the raw HW counters (PERF_STAT_EVENTS) in a single
    pass, so no extra workload run is needed. Returns (elapsed, hw_metrics);
    hw_metrics includes per-run `instructions` for absolute reconstruction.
    """
    binary_cmd = config["binary_cmd"]
    _write_repeat_script(container_name, binary_cmd, repeat_count)
    env_str = ""
    if config.get("env_vars"):
        env_str = " ".join(f"{k}={v}" for k, v in config["env_vars"].items()) + " "
    cmd = [
        "docker", "exec", "--privileged", container_name,
        "/bin/bash", "-c",
        f"{env_str}perf stat -C {PERF_CORE} -e {PERF_STAT_EVENTS} -- taskset -c {PERF_CORE} {PERF_SCRIPT}"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        print(f"    [perf stat] returncode={result.returncode}")
        if result.stderr:
            for line in result.stderr.strip().splitlines()[-5:]:
                print(f"    [perf stat stderr] {line}")
    elapsed = parse_perf_stat_time(result.stderr)
    if elapsed is None:
        print(f"    [perf stat] WARNING: could not parse elapsed time")
    counts = parse_perf_counters(result.stderr)
    coverage = parse_perf_min_coverage(result.stderr)
    hw_metrics = compute_hw_metrics(counts, repeat_count, coverage)
    if all(v is None for k, v in hw_metrics.items() if k != "min_coverage_pct"):
        print(f"    [perf stat] WARNING: no HW counters parsed")
    return elapsed, hw_metrics


# ===== Main part =====

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
        print(f"  Pinning all workloads to core {PERF_CORE}, target ~{PERF_TARGET_SECONDS}s per measurement")

        # Probe whether the HW-counter set fits without multiplexing on this core,
        # and warn/prompt if not (SMT-enabled core or NMI watchdog occupying a counter).
        if not check_counter_budget(docker_container_name):
            raise RuntimeError("perf collection aborted by user (counter multiplexing).")

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
            import time as _time
            warmup_t0 = _time.monotonic()
            warmup_res = subprocess.run(
                ["docker", "exec", "--privileged", docker_container_name,
                 "/bin/bash", "-c", f"source /usr/local/bin/user_entrypoint.sh; {rss_wrapper}"],
                capture_output=True, text=True, timeout=3600,
            )
            warmup_secs = _time.monotonic() - warmup_t0
            peak_rss_mb = _parse_peak_rss_mb(warmup_res.stderr)

            # Scale repeat count so total measurement time stays near PERF_TARGET_SECONDS
            single_run = max(1.0, warmup_secs)
            repeat_count = max(1, min(PERF_REPEAT_DEFAULT, int((PERF_TARGET_SECONDS / single_run * 2 + 1) // 2)))
            print(f"    single-run ~{single_run:.0f}s, using {repeat_count}x repeats")

            topdown = run_toplev_for_config(docker_container_name, config, repeat_count)
            elapsed, hw_metrics = run_perf_stat_for_config(docker_container_name, config, repeat_count)

            # Convert total elapsed to per-run average
            if elapsed is not None:
                elapsed = round(elapsed / repeat_count, 2)
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
                    # per-run average
                    "min": elapsed_min,
                    "sec": elapsed_sec,
                },
                "repeat_count": repeat_count,
                "peak_rss_mb": peak_rss_mb,
                "hw_metrics": hw_metrics,
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
            print(f"    hw_metrics: {hw_metrics}")
            print(f"    elapsed: {elapsed}s")
            if peak_rss_mb is not None:
                print(f"    peak_rss: {peak_rss_mb} MB")
            else:
                print(f"    peak_rss: (not captured)")

            # Save after each workload so completed results survive any interruptions
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

    # Let the descriptor set the values; otherwise keep the default.
    if descriptor_data.get("perf_core") is not None:
        global PERF_CORE
        PERF_CORE = int(descriptor_data["perf_core"])
    if descriptor_data.get("max_repeats") is not None:
        global PERF_REPEAT_DEFAULT
        PERF_REPEAT_DEFAULT = int(descriptor_data["max_repeats"])
    if descriptor_data.get("target_seconds") is not None:
        global PERF_TARGET_SECONDS
        PERF_TARGET_SECONDS = int(descriptor_data["target_seconds"])

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
