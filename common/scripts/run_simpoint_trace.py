#!/usr/bin/python3

# 02/26/2025 | Surim Oh | run_simpoint_trace.py
# A script to run tracing and clustering inside a docker container

import subprocess
import argparse
import os
import traceback
import time
import re
import glob
import json
import shlex
import shutil
import zipfile

def get_dr_jobs():
    """Decide DynamoRIO -jobs fanout based on env var DR_JOBS or auto-detect.

    Precedence:
      1. DR_JOBS env var (set by the outer local_runner based on host cores /
         memory headroom and the number of parallel trace variants).
      2. Auto-detect: max(2, min(40, cpu_count())). Assumes this drrun is the
         only DR process on the box.
    """
    env_val = os.environ.get("DR_JOBS")
    if env_val:
        try:
            jobs = int(env_val)
            if jobs >= 1:
                return jobs
        except ValueError:
            pass
    cores = os.cpu_count() or 1
    return max(2, min(40, cores))

DR_JOBS = get_dr_jobs()
print(f"run_simpoint_trace.py: DR_JOBS={DR_JOBS} (cpu_count={os.cpu_count()})", flush=True)

def find_trace_files(base_path):
    trace_files = glob.glob(os.path.join(base_path, "drmemtrace.*.dir/trace/dr*.trace.zip"))
    dr_folders = glob.glob(os.path.join(base_path, "drmemtrace.*.dir"))
    return trace_files, dr_folders

def get_largest_trace(base_path, simpoint_mode):
    traces = []
    if simpoint_mode == "3":
        pattern = os.path.join(base_path, "trace/dr*.trace.zip")
    else:
        pattern = os.path.join(base_path, "*/trace/dr*.trace.zip")

    for trace in glob.glob(os.path.join(base_path, pattern)):
        size = os.path.getsize(trace)
        traces.append((size, trace))

    if traces:
        return max(traces, key=lambda x: x[0])[1]
    return None

def get_largest_raw_trace(base_path):
    traces = []
    for trace in glob.glob(os.path.join(base_path, "*/raw/dr*.raw.lz4")):
        size = os.path.getsize(trace)
        traces.append((size, trace))

    if traces:
        return max(traces, key=lambda x: x[0])[1]
    return None

def report_time(procedure, start, end):
    elapsed_time = end - start
    hours = int(elapsed_time / 3600)
    minutes = int((elapsed_time % 3600) / 60)
    seconds = int((elapsed_time % 3600) % 60)
    print(f"{procedure}.. Runtime: {hours}:{minutes}:{seconds} (hh:mm:ss)")

# Get workload simpoint ids and their associated weights
def get_cluster_map(workload_home):
    read_simpoint_command = f"cat /{workload_home}/simpoints/opt.p.lpt0.99"

    simp_out = subprocess.check_output(read_simpoint_command.split(" ")).decode("utf-8").split("\n")[:-1]

    # Make final dictionary associated each simpoint id to its weight
    cluster_map = {}
    for cluster in simp_out:
        simp_id, cluster_id = cluster.split(" ")
        cluster_map[int(cluster_id)] = int(simp_id)

    return cluster_map

EMPTY_TRACE_WEIGHT_THRESHOLD = 0.01  # auto-remove segments with weight < 1%

def _remove_segment_from_simpoints(workload_home, cluster_id, segment_id, reason=""):
    """Remove a segment from opt.p.lpt0.99 and opt.w.lpt0.99 by cluster_id.

    Backs up originals to .bak (only on first call), removes the line,
    and renormalizes opt.w.lpt0.99 so weights still sum to 1.0.
    """
    simpoints_dir = os.path.join(workload_home, "simpoints")
    cid_str = str(cluster_id)

    # --- opt.p.lpt0.99: just remove the line ---
    for fname in ("opt.p.lpt0.99",):
        fpath = os.path.join(simpoints_dir, fname)
        bak = fpath + ".bak"
        if not os.path.isfile(bak):
            shutil.copy2(fpath, bak)
        with open(fpath) as f:
            lines = f.readlines()
        kept = [l for l in lines if l.strip() and l.split()[1] != cid_str]
        with open(fpath, "w") as f:
            f.writelines(kept)

    # --- opt.w.lpt0.99: remove the line and renormalize ---
    wpath = os.path.join(simpoints_dir, "opt.w.lpt0.99")
    bak = wpath + ".bak"
    if not os.path.isfile(bak):
        shutil.copy2(wpath, bak)
    with open(wpath) as f:
        lines = f.readlines()
    kept = []
    for l in lines:
        parts = l.split()
        if len(parts) == 2 and parts[1] != cid_str:
            kept.append((float(parts[0]), parts[1]))
    total = sum(w for w, _ in kept)
    with open(wpath, "w") as f:
        for w, cid in kept:
            f.write(f"{w / total:.6f} {cid}\n")

    tag = f" ({reason})" if reason else ""
    print(f"  AUTO-REMOVED segment {segment_id} (cluster {cluster_id}) from simpoints{tag}")
    print(f"  WARNING: workload may be non-deterministic (instruction count varies between runs). "
          f"Consider using trace_then_cluster for guaranteed correctness.")


def _get_weight_for_cluster(workload_home, cluster_id):
    """Look up the weight of a cluster_id from opt.w.lpt0.99."""
    wpath = os.path.join(workload_home, "simpoints", "opt.w.lpt0.99")
    if not os.path.isfile(wpath):
        return None
    with open(wpath) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2 and parts[1] == str(cluster_id):
                try:
                    return float(parts[0])
                except ValueError:
                    return None
    return None


def minimize_simpoint_traces(cluster_map, workload_home, warmup_chunks):
    ################################################################
    # minimize traces, rename traces
    # it is possible that SimPoint picks interval zero,
    # in that case the simulation would only need one chunk,
    # but we always keep two regardlessly
    try:
        dest_trace_dir = os.path.join(workload_home, "traces_simp", "trace")
        subprocess.run(["mkdir", "-p", dest_trace_dir], check=True, capture_output=True, text=True)
        for cluster_id, segment_id in cluster_map.items():
            # Skip segments already minimized
            dest_zip = os.path.join(dest_trace_dir, f"{segment_id}.zip")
            if os.path.isfile(dest_zip) and os.path.getsize(dest_zip) > 0:
                continue
            trace_dir = os.path.join(workload_home, "traces_simp", str(segment_id))
            trace_files = subprocess.getoutput(f"find {trace_dir} -name 'dr*.trace.zip' | grep 'drmemtrace.*.trace.zip'").splitlines()
            trace_files = [f for f in trace_files if f.strip()]
            num_traces = len(trace_files)
            if num_traces == 0:
                # raw2trace ran but produced no usable trace (e.g. segment past
                # end-of-execution in non-deterministic workloads). Auto-remove
                # if the weight is negligible; otherwise fail hard.
                weight = _get_weight_for_cluster(workload_home, cluster_id)
                if weight is not None and weight < EMPTY_TRACE_WEIGHT_THRESHOLD:
                    _remove_segment_from_simpoints(
                        workload_home, cluster_id, segment_id,
                        reason=f"empty trace, weight={weight:.4f} < {EMPTY_TRACE_WEIGHT_THRESHOLD}",
                    )
                    continue
                trace_clustering_info = {}
                trace_clustering_info["err"] = f"No trace.zip files found in {trace_dir}."
                with open(os.path.join(workload_home, "trace_clustering_info.json"), "w") as json_file:
                    json.dump(trace_clustering_info, json_file, indent=2, separators=(",", ":"))
                exit(1)
            if num_traces > 1:
                # Multi-threaded: pick the largest trace (dominant thread)
                trace_file = max(trace_files, key=lambda f: os.path.getsize(f))
                print(f"  Multiple trace.zip files ({num_traces}) for segment {segment_id}, using largest: {os.path.basename(trace_file)}")
                for f in trace_files:
                    if f != trace_file:
                        os.remove(f)
            else:
                trace_file = trace_files[0]
            big_zip_file = os.path.join(trace_dir, "trace", f"{segment_id}.big.zip")
            subprocess.run(f"mv {trace_file} {big_zip_file}", check=True, shell=True)
            unzip_output = subprocess.getoutput(f"unzip -l {big_zip_file}")
            num_chunk = len([line for line in unzip_output.splitlines() if 'chunk.' in line])

            if num_chunk < 2:
                print(f"WARN: the big trace {segment_id} contains less than 2 chunks: {num_chunk} !")

            chunk_list = ' '.join([f'chunk.{i:04d}' for i in range(warmup_chunks)])
            subprocess.run(f"zip {big_zip_file} --copy {chunk_list} --out {os.path.join(dest_trace_dir, f'{segment_id}.zip')}", shell=True)

            # Remove the big zip file
            os.remove(big_zip_file)
    except Exception as e:
        raise e

def trace_then_cluster(workload, suite, simpoint_home, bincmd, client_bincmd, simpoint_mode, drio_args, clustering_userk):
    # 1. trace the whole application
    # 2. drraw2trace
    # 3. post-process the trace in parallel
    # 4. aggregate the fingerprint
    # 5. clustering
    chunk_size = 10000000
    seg_size = 10000000
    warmup_chunks = 5
    try:
        start_time = time.perf_counter()
        subprocess.run(["mkdir", "-p", f"{simpoint_home}/{workload}/traces/whole"], check=True, capture_output=True, text=True)
        workload_home = f"{simpoint_home}/{workload}"
        dynamorio_home = os.environ.get('DYNAMORIO_HOME')
        trace_cmd = f"{dynamorio_home}/bin64/drrun -t drcachesim -jobs {DR_JOBS} -outdir {workload_home}/traces/whole -offline"
        if drio_args != None:
            trace_cmd = f"{trace_cmd} {drio_args}"
        trace_cmd = f"{trace_cmd} -- {bincmd}"
        trace_cmd_list = shlex.split(trace_cmd)

        whole_trace_path = f"{workload_home}/traces/whole"
        trace_clustering_info = {}
        print("whole app tracing..")
        if client_bincmd:
            subprocess.Popen("exec " + client_bincmd, stdout=subprocess.PIPE, shell=True)
        subprocess.run(trace_cmd_list, check=True, capture_output=True, text=True)
        end_time = time.perf_counter()
        report_time("whole app tracing done", start_time, end_time)

        trace_files, dr_folders = find_trace_files(whole_trace_path)
        num_traces = len(trace_files)
        num_dr_folder = len(dr_folders)

        # Single-threaded multi-process workload
        if num_traces == num_dr_folder and num_dr_folder > 1:
            whole_raw_trace = get_largest_raw_trace(whole_trace_path)
            dr_folder = os.path.dirname(os.path.dirname(whole_trace))
            modules_dir = os.path.dirname(os.path.join(dr_folder, "raw/modules.log"))
            dr_folder = os.path.basename(dr_folder)
            trace_clustering_info["modules_dir"] = modules_dir
            trace_clustering_info["dr_folder"] = dr_folder

        print("whole app raw2trace..")
        start_time = time.perf_counter()
        raw2trace_processes = set()

        if trace_clustering_info:
            dr_path = os.path.join(whole_trace_path, dr_folder)
            bin_path = os.path.join(whole_trace_path, dr_folder, "bin")
            raw_path = os.path.join(whole_trace_path, dr_folder, "raw")
            os.makedirs(bin_path, exist_ok=True)
            os.chmod(bin_path, 0o777)
            os.chmod(raw_path, 0o777)
            subprocess.run(f"cp {raw_path}/modules.log {bin_path}/modules.log", check=True, shell=True)
            subprocess.run(f"cp {raw_path}/modules.log {raw_path}/modules.log.bak", check=True, shell=True)
            subprocess.run(["python2", f"{simpoint_home}/scarab/utils/memtrace/portabilize_trace.py", f"{dr_path}"], capture_output=True, text=True, check=True)
            subprocess.run(f"cp {bin_path}/modules.log {raw_path}/modules.log", check=True, shell=True)
            raw2trace_cmd = f"{dynamorio_home}/tools/bin64/drraw2trace -jobs {DR_JOBS} -indir {raw_path} -chunk_instr_count {chunk_size}"
            process = subprocess.Popen("exec " + raw2trace_cmd, stdout=subprocess.PIPE, shell=True)
            raw2trace_processes.add(process)
        else:
            for root, dirs, files in os.walk(whole_trace_path):
                for directory in dirs:
                    if re.match(r"^dr.*", directory):
                        dr_path = os.path.join(root, directory)
                        bin_path = os.path.join(root, directory, "bin")
                        raw_path = os.path.join(root, directory, "raw")
                        os.makedirs(bin_path, exist_ok=True)
                        os.chmod(bin_path, 0o777)
                        os.chmod(raw_path, 0o777)
                        subprocess.run(f"cp {raw_path}/modules.log {bin_path}/modules.log", check=True, shell=True)
                        subprocess.run(f"cp {raw_path}/modules.log {raw_path}/modules.log.bak", check=True, shell=True)
                        subprocess.run(["python2", f"{simpoint_home}/scarab/utils/memtrace/portabilize_trace.py", f"{dr_path}"], capture_output=True, text=True, check=True)
                        subprocess.run(f"cp {bin_path}/modules.log {raw_path}/modules.log", check=True, shell=True)
                        raw2trace_cmd = f"{dynamorio_home}/tools/bin64/drraw2trace -jobs {DR_JOBS} -indir {raw_path} -chunk_instr_count {chunk_size}"
                        process = subprocess.Popen("exec " + raw2trace_cmd, stdout=subprocess.PIPE, shell=True)
                        raw2trace_processes.add(process)

        for p in raw2trace_processes:
            p.wait()
        end_time = time.perf_counter()
        report_time("whole app raw2trace done", start_time, end_time)

        print("post processing..")
        start_time = time.perf_counter()
        trace_files, dr_folders = find_trace_files(whole_trace_path)
        num_traces = len(trace_files)
        num_dr_folder = len(dr_folders)

        if not trace_clustering_info:
            if num_traces == 1 and num_dr_folder == 1:
                modules_dir = os.path.dirname(glob.glob(os.path.join(whole_trace_path, "drmemtrace.*.dir/raw/modules.log"))[0])
                whole_trace = glob.glob(os.path.join(whole_trace_path, "drmemtrace.*.dir/trace/dr*.zip"))[0]
                dr_folder = os.path.dirname(os.path.dirname(whole_trace))
            else:
                whole_trace = get_largest_trace(whole_trace_path, simpoint_mode)
                dr_folder = os.path.dirname(os.path.dirname(whole_trace))
                modules_dir = os.path.dirname(os.path.join(dr_folder, "raw/modules.log"))
            dr_folder = os.path.basename(dr_folder)
            trace_clustering_info["modules_dir"] = modules_dir
            trace_clustering_info["whole_trace"] = whole_trace
            trace_clustering_info["dr_folder"] = dr_folder
        else:
            if num_traces > 1:
                trace_clustering_info["err"] = "Tried to convert the largest raw trace, but found more than one converted traces."
                with open(os.path.join(workload_home, "trace_clustering_info.json"), "w") as json_file:
                    json.dump(trace_clustering_info, json_file, indent=2, separators=(",", ":"))
                exit(1)
            whole_trace = get_largest_trace(whole_trace_path, simpoint_mode)
            trace_clustering_info["whole_trace"] = whole_trace

        post_processing_cmd = f"/bin/bash /usr/local/bin/run_trace_post_processing.sh {workload_home} {modules_dir} {whole_trace} {chunk_size} {seg_size} {simpoint_home}"
        subprocess.run([post_processing_cmd], check=True, shell=True, stdin=open(os.devnull, 'r'))
        trace_file = os.path.basename(whole_trace)
        trace_clustering_info["trace_file"] = trace_file
        with open(os.path.join(workload_home, "trace_clustering_info.json"), "w") as json_file:
            json.dump(trace_clustering_info, json_file, indent=2, separators=(",", ":"))

        end_time = time.perf_counter()
        report_time("post processing done", start_time, end_time)

        print("clustering..")
        start_time = time.perf_counter()
        clustering_cmd = f"/bin/bash /usr/local/bin/run_clustering.sh {workload_home}/fingerprint/bbfp {workload_home}"
        if clustering_userk != None:
            clustering_cmd = f"{clustering_cmd} {clustering_userk}"
        subprocess.run([clustering_cmd], check=True, shell=True, stdin=open(os.devnull, 'r'))
        end_time = time.perf_counter()
        report_time("clustering done", start_time, end_time)

        print("minimizing traces..")
        start_time = time.perf_counter()
        os.makedirs(f"{workload_home}/traces_simp", exist_ok=True)
        minimize_trace_cmd = f"/bin/bash /usr/local/bin/minimize_trace.sh {dr_folder}/bin {whole_trace} {workload_home}/simpoints {warmup_chunks} {workload_home}/traces_simp"
        subprocess.run([minimize_trace_cmd], check=True, shell=True, stdin=open(os.devnull, 'r'))
        end_time = time.perf_counter()
        report_time("minimizing traces done", start_time, end_time)
    except Exception as e:
        raise e

def cluster_only(workload, suite, simpoint_home, bincmd, client_bincmd, simpoint_mode, drio_args, clustering_userk):
    """Mode 4: fingerprint + cluster only (Phase 1 of parallel segment pipeline)."""
    chunk_size = 10000000
    seg_size = 10000000
    try:
        os.makedirs(os.path.join(simpoint_home, workload, "fingerprint"), exist_ok=True)
        os.makedirs(os.path.join(simpoint_home, workload, "traces_simp"), exist_ok=True)
        workload_home = f"{simpoint_home}/{workload}"
        dynamorio_home = os.environ.get('DYNAMORIO_HOME')
        dr_home = workload_home

        # --- Fingerprinting (same as cluster_then_trace) ---
        fingerprint_dir = os.path.join(workload_home, "fingerprint")
        segment_size_path = os.path.join(fingerprint_dir, "segment_size")
        if os.path.isfile(segment_size_path) and glob.glob(os.path.join(fingerprint_dir, "bbfp.*")):
            print("generate fingerprint.. skipped (already exists)")
        else:
            print("generate fingerprint..")
            if client_bincmd:
                subprocess.Popen("exec " + client_bincmd, stdout=subprocess.PIPE, shell=True)
            start_time = time.perf_counter()
            drio_extra = drio_args if drio_args else ""
            drio_extra += " -disable_rseq"
            thread_limit_prefix = "PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false"
            fp_cmd = f"HOME={dr_home} {thread_limit_prefix} {dynamorio_home}/bin64/drrun -max_bb_instrs 4095 -opt_cleancall 2 {drio_extra} -c $tmpdir/libfpg.so -no_use_bb_pc -segment_size {seg_size} -output {workload_home}/fingerprint/bbfp -pcmap_output {workload_home}/fingerprint/pcmap -- {bincmd}"
            max_retries = 3
            for attempt in range(1, max_retries + 1):
                if attempt > 1:
                    import glob as _glob
                    for stale in _glob.glob(os.path.join(workload_home, "fingerprint", "bbfp.*")):
                        os.remove(stale)
                    for stale in _glob.glob(os.path.join(workload_home, "fingerprint", "pcmap.*")):
                        os.remove(stale)
                result = subprocess.run([fp_cmd], capture_output=True, text=True, shell=True)
                if result.returncode == 0:
                    break
                print(f"Fingerprint attempt {attempt}/{max_retries} failed (exit {result.returncode}):")
                if result.stderr:
                    print(f"  stderr: {result.stderr[-500:]}")
                if attempt == max_retries:
                    print(f"  cmd: {fp_cmd}")
                    if result.stdout: print(f"  stdout: {result.stdout[-2000:]}")
                    if result.stderr: print(f"  stderr: {result.stderr[-2000:]}")
                    result.check_returncode()
            end_time = time.perf_counter()

            with open(segment_size_path, "w") as f:
                f.write(f"{chunk_size}\n")
            report_time("generate fingerprint done", start_time, end_time)

        # --- Clustering (same as cluster_then_trace) ---
        simpoints_dir = os.path.join(workload_home, "simpoints")
        if os.path.isdir(simpoints_dir) and os.path.isfile(os.path.join(simpoints_dir, "opt.p")):
            print("clustering.. skipped (already exists)")
        else:
            print("clustering..")
            start_time = time.perf_counter()
            trace_clustering_info = {}
            bbfp_files = glob.glob(os.path.join(f"{workload_home}/fingerprint", "bbfp.*"))
            bbfp_files = [f for f in bbfp_files if not f.endswith('.inscount')]
            num_bbfp = len(bbfp_files)
            if num_bbfp == 0:
                trace_clustering_info["err"] = "No bbfp files found."
                with open(os.path.join(workload_home, "trace_clustering_info.json"), "w") as json_file:
                    json.dump(trace_clustering_info, json_file, indent=2, separators=(",", ":"))
                exit(1)
            if num_bbfp > 1:
                best_file = None
                best_lines = -1
                for f in bbfp_files:
                    with open(f, 'r') as fh:
                        n = sum(1 for line in fh if line.strip())
                    if n > best_lines:
                        best_lines = n
                        best_file = f
                print(f"  Multi-threaded: {num_bbfp} bbfp files, using {os.path.basename(best_file)} ({best_lines} segments)")
                bbfp_file = best_file
            else:
                bbfp_file = bbfp_files[0]
            clustering_cmd = f"/bin/bash /usr/local/bin/run_clustering.sh {bbfp_file} {workload_home}"
            if clustering_userk != None:
                clustering_cmd = f"{clustering_cmd} {clustering_userk}"
            subprocess.run([clustering_cmd], check=True, shell=True, stdin=open(os.devnull, 'r'))

            print("adjusting oversized simpoints..")
            replace_cmd = f"python3 /usr/local/bin/replace_oversized_simpoints.py {workload_home}"
            subprocess.run([replace_cmd], check=True, shell=True)

            end_time = time.perf_counter()
            report_time("clustering", start_time, end_time)

        print("cluster_only (mode 4) complete. Simpoints ready for parallel segment tracing.")
    except Exception as e:
        raise e


def trace_single_segment(workload, suite, simpoint_home, bincmd, client_bincmd, simpoint_mode, drio_args, segment_id, cluster_id):
    """Mode 5: trace + raw2trace + minimize a single segment (Phase 2 of parallel pipeline)."""
    chunk_size = 10000000
    seg_size = 10000000
    warmup_chunks = 5
    try:
        workload_home = f"{simpoint_home}/{workload}"
        dynamorio_home = os.environ.get('DYNAMORIO_HOME')
        dr_home = workload_home

        # Read segment_size from fingerprint (should already exist from Phase 1)
        segment_size_path = os.path.join(workload_home, "fingerprint", "segment_size")
        if os.path.isfile(segment_size_path):
            with open(segment_size_path) as f:
                seg_size = int(f.read().strip())
                chunk_size = seg_size

        seg_dir = os.path.join(workload_home, "traces_simp", str(segment_id))
        os.makedirs(seg_dir, exist_ok=True)

        # --- Check if already fully minimized ---
        dest_trace_dir = os.path.join(workload_home, "traces_simp", "trace")
        dest_zip = os.path.join(dest_trace_dir, f"{segment_id}.zip")
        if os.path.isfile(dest_zip) and os.path.getsize(dest_zip) > 0:
            print(f"segment {segment_id}: already minimized, skipping")
            return

        # --- 1. Trace this segment ---
        raw_dir = os.path.join(seg_dir, "raw")
        if os.path.isdir(raw_dir) and any(
            f.endswith(".raw.lz4")
            for dp, _, fns in os.walk(raw_dir) for f in fns
        ):
            print(f"segment {segment_id}: raw trace exists, skipping trace")
        else:
            print(f"segment {segment_id}: tracing..")
            start_time = time.perf_counter()
            roi_start = segment_id * seg_size
            roi_end = roi_start + seg_size
            warmup = seg_size * warmup_chunks
            if roi_start > warmup:
                roi_start -= warmup
            else:
                roi_start = 0
            roi_length = roi_end - roi_start

            drio_trace_extra = drio_args if drio_args else ""
            drio_trace_extra += " -disable_rseq"
            if roi_start == 0:
                trace_cmd = f"{dynamorio_home}/bin64/drrun -opt_cleancall 2 {drio_trace_extra} -t drcachesim -jobs {DR_JOBS} -outdir {seg_dir} -offline -count_fetched_instrs -trace_for_instrs {roi_length} -- {bincmd}"
            else:
                trace_cmd = f"{dynamorio_home}/bin64/drrun -opt_cleancall 2 {drio_trace_extra} -t drcachesim -jobs {DR_JOBS} -outdir {seg_dir} -offline -count_fetched_instrs -trace_after_instrs {roi_start} -trace_for_instrs {roi_length} -- {bincmd}"

            trace_env = os.environ.copy()
            trace_env["HOME"] = dr_home
            result = subprocess.run(trace_cmd, shell=True, stdout=subprocess.DEVNULL, env=trace_env)
            end_time = time.perf_counter()
            report_time(f"segment {segment_id} tracing done", start_time, end_time)

        # --- 2. Portabilize + thread filter + raw2trace ---
        trace_path = f"{workload_home}/traces_simp/{segment_id}"
        trace_out = os.path.join(trace_path, "trace")
        if os.path.isdir(trace_out) and any(
            f.endswith(".trace.zip")
            for _, _, fns in os.walk(trace_out) for f in fns
        ):
            print(f"segment {segment_id}: trace output exists, skipping raw2trace")
        else:
            print(f"segment {segment_id}: raw2trace..")
            start_time = time.perf_counter()

            # Move raw/ out of drmemtrace dir
            raw_path = os.path.join(trace_path, "raw")
            dr_raw_dirs = glob.glob(os.path.join(trace_path, "dr*/raw"))
            fresh_dr_raw = [d for d in dr_raw_dirs if any(
                f.endswith(".raw.lz4") for _, _, fns in os.walk(d) for f in fns)]
            if fresh_dr_raw:
                if os.path.isdir(raw_path):
                    shutil.rmtree(raw_path)
                subprocess.run(f"mv {fresh_dr_raw[0]}/../raw/ {trace_path}/raw/", check=True, shell=True)
            elif not os.path.isdir(raw_path):
                subprocess.run(f"mv {trace_path}/dr*/raw/ {trace_path}/raw/", check=True, shell=True)

            os.makedirs(f"{trace_path}/bin", exist_ok=True)
            bin_path = os.path.join(trace_path, "bin")
            os.chmod(bin_path, 0o777)
            os.chmod(raw_path, 0o777)

            # Portabilize
            bin_files = [f for f in os.listdir(bin_path) if f.endswith('.so') or f.endswith('.so.1')]
            if len(bin_files) > 10:
                print(f"  segment {segment_id}: portabilize skipped (bin/ has {len(bin_files)} libs)")
            else:
                bak_path = os.path.join(raw_path, "modules.log.bak")
                if os.path.isfile(bak_path):
                    subprocess.run(f"cp {bak_path} {bin_path}/modules.log", check=True, shell=True)
                else:
                    subprocess.run(f"cp {raw_path}/modules.log {bin_path}/modules.log", check=True, shell=True)
                    subprocess.run(f"cp {raw_path}/modules.log {raw_path}/modules.log.bak", check=True, shell=True)
                subprocess.run(["python2", f"{simpoint_home}/scarab/utils/memtrace/portabilize_trace.py", f"{trace_path}"], capture_output=True, text=True, check=True)
                subprocess.run(f"cp {bin_path}/modules.log {raw_path}/modules.log", check=True, shell=True)

            # Thread filter: keep only the main thread
            win_dir = os.path.join(raw_path, "window.0000")
            if os.path.isdir(win_dir):
                raw_files = [f for f in os.listdir(win_dir) if f.endswith(".raw.lz4")]
                if len(raw_files) > 1:
                    main_file = max(raw_files, key=lambda f: os.path.getsize(os.path.join(win_dir, f)))
                    removed = 0
                    for f in raw_files:
                        if f != main_file:
                            os.remove(os.path.join(win_dir, f))
                            removed += 1
                    if removed:
                        print(f"  segment {segment_id}: kept main thread, removed {removed} helper thread raw files")

            raw2trace_cmd = f"{dynamorio_home}/tools/bin64/drraw2trace -jobs 1 -indir {raw_path} -chunk_instr_count {chunk_size}"
            subprocess.run(raw2trace_cmd, shell=True, stdout=subprocess.PIPE)
            end_time = time.perf_counter()
            report_time(f"segment {segment_id} raw2trace done", start_time, end_time)

        # --- 3. Minimize this segment ---
        print(f"segment {segment_id}: minimizing..")
        start_time = time.perf_counter()
        single_cluster_map = {cluster_id: segment_id}
        minimize_simpoint_traces(single_cluster_map, workload_home, warmup_chunks + 1)
        end_time = time.perf_counter()
        report_time(f"segment {segment_id} minimize done", start_time, end_time)

    except Exception as e:
        raise e


def cluster_then_trace(workload, suite, simpoint_home, bincmd, client_bincmd, simpoint_mode, drio_args, clustering_userk, manual_trace=False):
    # 1. collect fingerprints
    # 2. clustering
    # 2. trace segments of the workload
    # 3. drraw2trace
    # 4. minimize traces
    chunk_size = 10000000
    seg_size = 10000000
    warmup_chunks = 5
    try:
        os.makedirs(os.path.join(simpoint_home, workload, "fingerprint"), exist_ok=True)
        os.makedirs(os.path.join(simpoint_home, workload, "traces_simp"), exist_ok=True)
        workload_home = f"{simpoint_home}/{workload}"
        dynamorio_home = os.environ.get('DYNAMORIO_HOME')
        # Isolate DynamoRIO config directory per workload. DR caches
        # per-process configs in ~/.dynamorio/ keyed by binary name + PID.
        # When multiple jobs run concurrently on the same node sharing NFS
        # home, one job's config gets picked up by another job's python
        # process, causing libfpg.so to load with the wrong output path or
        # fail to load entirely. Setting HOME for the drrun command makes
        # ~/.dynamorio resolve to a per-workload directory.
        dr_home = workload_home

        # Resume support: skip fingerprinting if already done
        fingerprint_dir = os.path.join(workload_home, "fingerprint")
        segment_size_path = os.path.join(fingerprint_dir, "segment_size")
        if os.path.isfile(segment_size_path) and glob.glob(os.path.join(fingerprint_dir, "bbfp.*")):
            print("generate fingerprint.. skipped (already exists)")
        else:
            print("generate fingerprint..")
            if client_bincmd:
                subprocess.Popen("exec " + client_bincmd, stdout=subprocess.PIPE, shell=True)
            start_time = time.perf_counter()
            drio_extra = drio_args if drio_args else ""
            # Disable restartable sequences — DR doesn't fully support rseq on
            # newer glibc, causing "entries are not in a loaded segment" errors.
            drio_extra += " -disable_rseq"
            # Force single-threaded execution during fingerprinting to avoid
            # DynamoRIO crashes from concurrent module loads in multi-threaded
            # Python workloads (torch/faiss/sentence-transformers). The crash
            # is in DR's d_r_strcmp called with a corrupted module-name pointer
            # when multiple threads dlopen/dlclose simultaneously. Single-threaded
            # fingerprinting is also more deterministic for SimPoint.
            thread_limit_prefix = "PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false"
            fp_cmd = f"HOME={dr_home} {thread_limit_prefix} {dynamorio_home}/bin64/drrun -max_bb_instrs 4095 -opt_cleancall 2 {drio_extra} -c $tmpdir/libfpg.so -no_use_bb_pc -segment_size {seg_size} -output {workload_home}/fingerprint/bbfp -pcmap_output {workload_home}/fingerprint/pcmap -- {bincmd}"
            # DynamoRIO has a non-deterministic crash in d_r_strcmp when
            # multi-threaded Python apps do concurrent dlopen/dlclose during
            # fingerprinting. Retry up to 3 times on failure.
            max_retries = 3
            for attempt in range(1, max_retries + 1):
                # Clean partial fingerprint output from any prior failed attempt
                if attempt > 1:
                    import glob as _glob
                    for stale in _glob.glob(os.path.join(workload_home, "fingerprint", "bbfp.*")):
                        os.remove(stale)
                    for stale in _glob.glob(os.path.join(workload_home, "fingerprint", "pcmap.*")):
                        os.remove(stale)
                result = subprocess.run([fp_cmd], capture_output=True, text=True, shell=True)
                if result.returncode == 0:
                    break
                print(f"Fingerprint attempt {attempt}/{max_retries} failed (exit {result.returncode}):")
                if result.stderr:
                    print(f"  stderr: {result.stderr[-500:]}")
                if attempt == max_retries:
                    print(f"  cmd: {fp_cmd}")
                    if result.stdout: print(f"  stdout: {result.stdout[-2000:]}")
                    if result.stderr: print(f"  stderr: {result.stderr[-2000:]}")
                    result.check_returncode()
            end_time = time.perf_counter()

            with open(segment_size_path, "w") as f:
                f.write(f"{chunk_size}\n")
            report_time("generate fingerprint done", start_time, end_time)

        # Resume support: skip clustering if simpoints already exist
        simpoints_dir = os.path.join(workload_home, "simpoints")
        if os.path.isdir(simpoints_dir) and os.path.isfile(os.path.join(simpoints_dir, "opt.p")):
            print("clustering.. skipped (already exists)")
        else:
            print("clustering..")
            start_time = time.perf_counter()
            trace_clustering_info = {}
            bbfp_files = glob.glob(os.path.join(f"{workload_home}/fingerprint", "bbfp.*"))
            bbfp_files = [f for f in bbfp_files if not f.endswith('.inscount')]
            num_bbfp = len(bbfp_files)
            if num_bbfp == 0:
                trace_clustering_info["err"] = "No bbfp files found."
                with open(os.path.join(workload_home, "trace_clustering_info.json"), "w") as json_file:
                    json.dump(trace_clustering_info, json_file, indent=2, separators=(",", ":"))
                exit(1)
            if num_bbfp > 1:
                # Multi-threaded workload: pick the main thread's fingerprint
                # (the file with the most lines = most segments). Helper threads
                # (GC, import, signal) contribute negligible instructions.
                # This mirrors get_largest_trace() used by trace_then_cluster.
                best_file = None
                best_lines = -1
                for f in bbfp_files:
                    with open(f, 'r') as fh:
                        n = sum(1 for line in fh if line.strip())
                    if n > best_lines:
                        best_lines = n
                        best_file = f
                print(f"  Multi-threaded: {num_bbfp} bbfp files, using {os.path.basename(best_file)} ({best_lines} segments)")
                bbfp_file = best_file
            else:
                bbfp_file = bbfp_files[0]
            clustering_cmd = f"/bin/bash /usr/local/bin/run_clustering.sh {bbfp_file} {workload_home}"
            if clustering_userk != None:
                clustering_cmd = f"{clustering_cmd} {clustering_userk}"
            subprocess.run([clustering_cmd], check=True, shell=True, stdin=open(os.devnull, 'r'))

            print("adjusting oversized simpoints..")
            replace_cmd = f"python3 /usr/local/bin/replace_oversized_simpoints.py {workload_home}"
            subprocess.run([replace_cmd], check=True, shell=True)

            end_time = time.perf_counter()
            report_time("clustering", start_time, end_time)

        trace_clustering_info = {}
        cluster_map = get_cluster_map(workload_home)

        print("clustering tracing..")
        start_time = time.perf_counter()
        # Batch segment traces to avoid OOM: each drrun process runs the full
        # workload under DynamoRIO (~500MB-1GB each).  Launching all segments
        # in parallel (e.g. 30+) easily exceeds the slurm memory limit.
        max_parallel = int(os.environ.get("TRACE_PARALLEL", "1"))
        trace_cmds = []
        skipped_segments = 0
        for cluster_id, segment_id in cluster_map.items():
            seg_dir = os.path.join(workload_home, "traces_simp", str(segment_id))

            # Resume support: skip segments that already have raw trace data
            raw_dir = os.path.join(seg_dir, "raw")
            if os.path.isdir(raw_dir) and any(
                f.endswith(".raw.lz4")
                for dp, _, fns in os.walk(raw_dir) for f in fns
            ):
                skipped_segments += 1
                continue

            os.makedirs(seg_dir, exist_ok=True)
            # the simulation region, in the unit of chunks
            roi_start = segment_id * seg_size
            # seq is inclusive
            roi_end = roi_start + seg_size

            # assume warm-up length is the segsize
            # this will limit the amount of warmup that can be done during the simulation
            warmup = seg_size * warmup_chunks

            if roi_start > warmup:
                # enough room for warmup, extend roi start to the left
                roi_start -= warmup
            else:
                # no enough preceding instructions, can only warmup till segment start
                # new roi start is the very first instruction of the trace
                roi_start = 0

            roi_length = roi_end - roi_start

            drio_trace_extra = drio_args if drio_args else ""
            drio_trace_extra += " -disable_rseq"
            # Note: do NOT use -max_bb_instrs here. With drcachesim -offline,
            # hitting the BB size limit is fatal (exit 255, empty output).
            # The fingerprint client (libfpg.so) tolerates it as a warning,
            # but drcachesim does not.
            if roi_start == 0:
                trace_cmd = f"{dynamorio_home}/bin64/drrun -opt_cleancall 2 {drio_trace_extra} -t drcachesim -jobs {DR_JOBS} -outdir {seg_dir} -offline -count_fetched_instrs -trace_for_instrs {roi_length} -- {bincmd}"
            else:
                trace_cmd = f"{dynamorio_home}/bin64/drrun -opt_cleancall 2 {drio_trace_extra} -t drcachesim -jobs {DR_JOBS} -outdir {seg_dir} -offline -count_fetched_instrs -trace_after_instrs {roi_start} -trace_for_instrs {roi_length} -- {bincmd}"

            trace_cmds.append(trace_cmd)

        if skipped_segments > 0:
            print(f"  resuming: {skipped_segments} segments already traced, {len(trace_cmds)} remaining")

        # Launch in batches to stay within memory limits
        trace_env = os.environ.copy()
        trace_env["HOME"] = dr_home
        for batch_start in range(0, len(trace_cmds), max_parallel):
            batch = trace_cmds[batch_start:batch_start + max_parallel]
            print(f"  tracing segments {batch_start+1}-{batch_start+len(batch)} of {len(trace_cmds)}")
            procs = []
            for cmd in batch:
                p = subprocess.Popen("exec " + cmd, stdout=subprocess.DEVNULL, shell=True, env=trace_env)
                procs.append(p)
            for p in procs:
                p.wait()

        end_time = time.perf_counter()
        report_time("cluster tracing done", start_time, end_time)

        print("clustered traces raw2trace..")
        start_time = time.perf_counter()
        raw2trace_cmds = []
        for cluster_id, segment_id in cluster_map.items():
            trace_path = f"{workload_home}/traces_simp/{segment_id}"

            # Resume support: skip segments where raw2trace already produced trace output.
            # DR raw2trace puts files inside trace/window.NNNN/ subdirs.
            # Check specifically for drmemtrace.*.trace.zip (not cpu_schedule.bin.zip
            # which persists even after minimize deletes the actual trace).
            trace_out = os.path.join(trace_path, "trace")
            if os.path.isdir(trace_out) and any(
                f.endswith(".trace.zip")
                for _, _, fns in os.walk(trace_out) for f in fns
            ):
                continue

            trace_files, dr_folders = find_trace_files(trace_path)
            num_dr_folder = len(dr_folders)
            if num_dr_folder == 0:
                # raw/ already moved out of drmemtrace dir (prior run did this step)
                dr_folders = []
            if num_dr_folder > 1:
                # Multi-process edge case: pick the folder with the largest raw data
                best_folder = max(dr_folders, key=lambda d: sum(
                    os.path.getsize(os.path.join(dp, f))
                    for dp, _, fns in os.walk(d) for f in fns))
                print(f"  Multiple drmemtrace dirs ({num_dr_folder}), using {os.path.basename(best_folder)}")
                for d in dr_folders:
                    if d != best_folder:
                        import shutil
                        shutil.rmtree(d)

            # Move raw/ out of drmemtrace dir if not already done.
            # If raw/ exists but is stale (no .raw.lz4) and drmemtrace.*.dir has
            # fresh data, replace raw/ with the new data.
            raw_path = os.path.join(trace_path, "raw")
            dr_raw_dirs = glob.glob(os.path.join(trace_path, "dr*/raw"))
            fresh_dr_raw = [d for d in dr_raw_dirs if any(
                f.endswith(".raw.lz4") for _, _, fns in os.walk(d) for f in fns)]
            if fresh_dr_raw:
                if os.path.isdir(raw_path):
                    import shutil as _shutil
                    _shutil.rmtree(raw_path)
                subprocess.run(f"mv {fresh_dr_raw[0]}/../raw/ {trace_path}/raw/", check=True, shell=True)
            elif not os.path.isdir(raw_path):
                subprocess.run(f"mv {trace_path}/dr*/raw/ {trace_path}/raw/", check=True, shell=True)
            os.makedirs(f"{trace_path}/bin", exist_ok=True)
            bin_path = os.path.join(trace_path, "bin")
            os.chmod(bin_path, 0o777)
            os.chmod(raw_path, 0o777)
            # Skip portabilize if bin/ already has libraries (prior run completed it)
            bin_files = [f for f in os.listdir(bin_path) if f.endswith('.so') or f.endswith('.so.1')]
            if len(bin_files) > 10:
                print(f"  segment {segment_id}: portabilize skipped (bin/ has {len(bin_files)} libs)")
            else:
                # Restore original modules.log from backup if a prior run already
                # portabilized it (re-running portabilize on modified paths fails).
                bak_path = os.path.join(raw_path, "modules.log.bak")
                if os.path.isfile(bak_path):
                    subprocess.run(f"cp {bak_path} {bin_path}/modules.log", check=True, shell=True)
                else:
                    subprocess.run(f"cp {raw_path}/modules.log {bin_path}/modules.log", check=True, shell=True)
                    subprocess.run(f"cp {raw_path}/modules.log {raw_path}/modules.log.bak", check=True, shell=True)
                subprocess.run(["python2", f"{simpoint_home}/scarab/utils/memtrace/portabilize_trace.py", f"{trace_path}"], capture_output=True, text=True, check=True)
                subprocess.run(f"cp {bin_path}/modules.log {raw_path}/modules.log", check=True, shell=True)
            # Keep only the main thread's raw file to reduce memory.
            # Python GIL serializes execution; helper threads add no useful data
            # but each one costs ~1GB+ during raw2trace decompression.
            win_dir = os.path.join(raw_path, "window.0000")
            if os.path.isdir(win_dir):
                raw_files = [f for f in os.listdir(win_dir) if f.endswith(".raw.lz4")]
                if len(raw_files) > 1:
                    main_file = max(raw_files, key=lambda f: os.path.getsize(os.path.join(win_dir, f)))
                    removed = 0
                    for f in raw_files:
                        if f != main_file:
                            os.remove(os.path.join(win_dir, f))
                            removed += 1
                    if removed:
                        print(f"  segment {segment_id}: kept main thread, removed {removed} helper thread raw files")
            # Use -jobs 1 for raw2trace to limit memory: a single 500MB
            # compressed trace can decompress to 100GB+ with -jobs 2.
            raw2trace_cmd = f"{dynamorio_home}/tools/bin64/drraw2trace -jobs 1 -indir {raw_path} -chunk_instr_count {chunk_size}"
            raw2trace_cmds.append(raw2trace_cmd)

        if len(raw2trace_cmds) == 0:
            print("  all segments already have trace output, skipping raw2trace")
        else:
            # Launch raw2trace in batches to stay within memory limits
            max_r2t = int(os.environ.get("RAW2TRACE_PARALLEL", "1"))
            for batch_start in range(0, len(raw2trace_cmds), max_r2t):
                batch = raw2trace_cmds[batch_start:batch_start + max_r2t]
                print(f"  raw2trace segments {batch_start+1}-{batch_start+len(batch)} of {len(raw2trace_cmds)}")
                procs = []
                for cmd in batch:
                    p = subprocess.Popen("exec " + cmd, stdout=subprocess.PIPE, shell=True)
                    procs.append(p)
                for p in procs:
                    p.wait()

        end_time = time.perf_counter()
        report_time("clustered traces raw2trace done", start_time, end_time)

        print("minimize traces..")
        start_time = time.perf_counter()
        minimize_simpoint_traces(cluster_map, workload_home, warmup_chunks + 1)
        end_time = time.perf_counter()
        report_time("minimize traces done", start_time, end_time)
    except Exception as e:
        raise e

def iterative(workload, suite, simpoint_home, bincmd, client_bincmd, simpoint_mode, drio_args, clustering_userk):
    # 1. trace timestep event from application
    # 2. drraw2trace
    chunk_size = 10000000
    max_retries = 5

    try:
        workload_home = f"{simpoint_home}/{workload}"
        for i in range(1, 101):
            timestep_dir = os.path.join(workload_home, "traces_simp")
            os.makedirs(timestep_dir, exist_ok=True)

            start_time = time.perf_counter()
            dynamorio_home = os.environ.get('DYNAMORIO_HOME')
            if not dynamorio_home:
                raise EnvironmentError("DYNAMORIO_HOME not set")

            trace_cmd = (f"{dynamorio_home}/bin64/drrun -t drcachesim -jobs {DR_JOBS} -outdir {timestep_dir} -offline")
            if drio_args is not None:
                trace_cmd = f"{trace_cmd} {drio_args}"
            trace_cmd = f"{trace_cmd} -- {bincmd}"
            trace_cmd_list = shlex.split(trace_cmd)

            print(f"[Timestep {i}] tracing..")
            if i == 1 and client_bincmd:
                subprocess.Popen("exec " + client_bincmd, stdout=subprocess.PIPE, shell=True)

            for attempt in range(max_retries):
                try:
                    subprocess.run(trace_cmd_list, check=True, capture_output=True, text=True, timeout=60)
                    break
                except subprocess.TimeoutExpired as e:
                    print(f"[Timestep {i}] tracing attempt {attempt+1} timed out: {e}")
                    if attempt == max_retries - 1:
                        print(f"[Timestep {i}] tracing failed with error: {e}")
                        break
                    else:
                        subdirs = [os.path.join(timestep_dir, d) for d in os.listdir(timestep_dir)
                                   if os.path.isdir(os.path.join(timestep_dir, d))]
                        if subdirs:
                            latest_dir = max(subdirs, key=os.path.getmtime)
                            print(f"[Timestep {i}] removing the most recent folder: {latest_dir}")
                            shutil.rmtree(latest_dir, ignore_errors=True)
                        else:
                            print(f"[Timestep {i}] no subfolder found in {timestep_dir} to remove.")

                        print(f"[Timestep {i}] retrying tracing (attempt {attempt+2})...")

            end_time = time.perf_counter()
            report_time(f"[Timestep {i}] tracing done", start_time, end_time)

        print(f"folder renaming..")
        dir_counter = 1
        for root, dirs, files in os.walk(timestep_dir):
            for directory in dirs:
                if re.match(r"^dr.*", directory):
                    old_path = os.path.join(root, directory)
                    new_folder = os.path.join(root, f"Timestep_{dir_counter}")
                    if not os.path.exists(new_folder):
                        os.makedirs(new_folder)
                    new_path = os.path.join(new_folder, directory)
                    os.rename(old_path, new_path)
                    dir_counter += 1
        print(f"folder renaming done..")

        print(f"raw2trace..")
        start_time = time.perf_counter()
        available_cores = os.cpu_count() or 1
        max_processes = int(available_cores * 0.6)
        print(f"available cores: {available_cores}, max_processes: {max_processes}")

        raw2trace_processes = set()
        for root, dirs, files in os.walk(timestep_dir):
            for directory in dirs:
                if re.match(r"^dr.*", directory):
                    dr_path = os.path.join(root, directory)
                    bin_path = os.path.join(root, directory, "bin")
                    raw_path = os.path.join(root, directory, "raw")
                    os.makedirs(bin_path, exist_ok=True)
                    os.chmod(bin_path, 0o777)
                    os.chmod(raw_path, 0o777)
                    subprocess.run(f"cp {raw_path}/modules.log {bin_path}/modules.log", check=True, shell=True)
                    subprocess.run(f"cp {raw_path}/modules.log {raw_path}/modules.log.bak", check=True, shell=True)
                    subprocess.run(["python2", f"{simpoint_home}/scarab/utils/memtrace/portabilize_trace.py", f"{dr_path}"], capture_output=True, text=True, check=True)
                    subprocess.run(f"cp {bin_path}/modules.log {raw_path}/modules.log", check=True, shell=True)
                    raw2trace_cmd = f"{dynamorio_home}/tools/bin64/drraw2trace -jobs {DR_JOBS} -indir {raw_path} -chunk_instr_count {chunk_size}"
                    process = subprocess.Popen("exec " + raw2trace_cmd, stdout=subprocess.PIPE, shell=True)
                    stdout, stderr = process.communicate()
                    if stderr:
                        print("drraw2trace error:", stderr.decode())
                    raw2trace_processes.add(process)

                    while len(raw2trace_processes) >= max_processes:
                        for p in list(raw2trace_processes):
                            if p.poll() is not None:
                                p.wait()
                                raw2trace_processes.remove(p)
                                break

        for p in raw2trace_processes:
            p.wait()

        end_time = time.perf_counter()
        report_time(f"raw2trace done", start_time, end_time)

        print(f"post processing..")
        start_time = time.perf_counter()
        inst_counts = {}
        dir_counter = 1
        tracefiles = []

        for root, dirs, files in os.walk(timestep_dir):
            if re.match(r"^Timestep_\d+$", os.path.basename(root)):
                for directory in dirs:
                    if re.match(r"^dr.*", directory):
                        trace_clustering_info = {}
                        dr_path = os.path.join(root, directory)
                        whole_trace = get_largest_trace(dr_path, simpoint_mode)
                        dr_folder_path = os.path.dirname(os.path.dirname(whole_trace))
                        print(whole_trace)
                        tracefiles.append(whole_trace)

                        try:
                            with zipfile.ZipFile(whole_trace, 'r') as zf:
                                file_list = zf.namelist()
                                chunk_files = [fname for fname in file_list if re.search(r'chunk\.', fname)]
                                numChunk = len(chunk_files)
                                numInsts = numChunk * chunk_size
                                print(f"{directory}: numInsts = {numInsts}")
                                inst_counts[directory] = numInsts
                        except zipfile.BadZipFile:
                            print(f"invalid zip: {whole_trace}.")
                            shutil.rmtree(os.path.join(root, directory), ignore_errors=True)
                            continue

        total_inst = sum(inst_counts.values())
        print(f"Total instruction count across all directories: {total_inst}")

        trace_weights = {}
        for directory, count in inst_counts.items():
            if total_inst > 0:
                weight = count / total_inst
            else:
                weight = 0
            trace_weights[directory] = weight

        print("Computed trace weights:")
        for directory, weight in trace_weights.items():
            print(f"{directory}: {weight:.4f}")

        sorted_dirs = sorted(trace_weights.keys())

        simpoint_home = f"{workload_home}/simpoints"
        os.makedirs(simpoint_home, exist_ok=True)

        opt_w_path = os.path.join(simpoint_home, "opt.w")
        opt_w_lpt_path = os.path.join(simpoint_home, "opt.w.lpt0.99")
        with open(opt_w_path, "w") as f1, open(opt_w_lpt_path, "w") as f2:
            for idx, directory in enumerate(sorted_dirs):
                weight = trace_weights[directory]
                line = f"{weight:.6f} {idx}\n"
                f1.write(line)
                f2.write(line)

        opt_w2_path = os.path.join(simpoint_home, "opt.w.2")
        opt_w2_lpt_path = os.path.join(simpoint_home, "opt.w.2.lpt0.99")
        with open(opt_w2_path, "w") as f3, open(opt_w2_lpt_path, "w") as f4:
            for idx, directory in enumerate(sorted_dirs):
                inst_count = inst_counts[directory]
                line = f"{inst_count} {idx}\n"
                f3.write(line)
                f4.write(line)

        opt_p_path = os.path.join(simpoint_home, "opt.p")
        opt_p_lpt_path = os.path.join(simpoint_home, "opt.p.lpt0.99")
        with open(opt_p_path, "w") as f1, open(opt_p_lpt_path, "w") as f2:
            for idx, directory in enumerate(sorted_dirs):
                line = f"{idx+1} {idx}\n"
                f1.write(line)
                f2.write(line)

        fingerprint_dir = os.path.join(workload_home, "fingerprint")
        if not os.path.exists(fingerprint_dir):
            os.makedirs(fingerprint_dir)
        segment_size_path = os.path.join(fingerprint_dir, "segment_size")
        with open(segment_size_path, "w") as f:
            f.write(f"{chunk_size}\n")

        modules_log_list = glob.glob(os.path.join(dr_folder_path, "raw", "modules.log"))
        modules_dir = os.path.dirname(modules_log_list[0]) if modules_log_list else ""
        trace_clustering_info["modules_dir"] = modules_dir
        trace_clustering_info["whole_trace"] = None
        trace_clustering_info["dr_folder"] = os.path.basename(dr_folder_path)
        trace_clustering_info["trace_file"] = tracefiles

        if trace_clustering_info is not None:
            clustering_info_path = os.path.join(workload_home, "trace_clustering_info.json")
            with open(clustering_info_path, "w") as json_file:
                json.dump(trace_clustering_info, json_file, indent=2, separators=(",", ":"))
            print(f"trace_clustering_info.json written to {clustering_info_path}")

        end_time = time.perf_counter()
        report_time(f"post processing done", start_time, end_time)
    except Exception as e:
        print(f"[Timestep {i}] failed: {str(e)}")
        raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Runs clustering/tracing on local or a slurm network')

    # Add arguments
    parser.add_argument('-w', '--workload', required=True, help='Workload name to run tracing. Usage: --workload fibers_benchmark')
    parser.add_argument('-s', '--suite', required=True, help='Workload suite name. Usage: --suite dcperf')
    parser.add_argument('-simph', '--simpoint_home', required=True, help='Path to the simpoint_flow home of the tracing. Usage: --simpoint_home /home/surim/simpoint_flow/trace_dcperf')
    parser.add_argument('-b', '--bincmd', required=True, help='A Binary command to run the workload. Usage: --bincmd \$tmpdir/DCPerf/benchmarks/wdl_bench/fibers_fibers_benchmark')
    parser.add_argument('-m', '--simpoint_mode', required=True, help='Simpoint mode. 1 for non-post-processing, 2 for post-processing. Usage: --simpoint_mode 2')
    parser.add_argument('-clb', '--client_bincmd', required=False, default=None, help='A Binary command to run a background client for the workload. Usage: --client_bincmd "nohup /opt/ros/\$ROS_DISTRO/bin/ros2 bag play "$tmpdir/driving_log_replayer_output/yabloc/latest/sample/result_bag" --rate "')
    parser.add_argument('-dr', '--drio_args', required=False, default=None, help='Dynamorio arguments. Usage: --drio_args "-exit_after_tracing 1520000000000"')
    parser.add_argument('-userk', '--clustering_userk', required=False, default=None, help='maxk will use the user provided value if specified. If not specified, maxk will be calculated as the square root of the number of segments.')
    parser.add_argument('-man', '--manual_trace', required=False, default=None, help='manual trace. Usage --manual_trace True')
    parser.add_argument('--segment_id', type=int, default=None, help='Segment ID to trace (mode 5 only)')
    parser.add_argument('--cluster_id', type=int, default=None, help='Cluster ID for the segment (mode 5 only)')

    # Parse the command-line arguments
    args = parser.parse_args()

    workload = args.workload
    suite = args.suite
    simpoint_home = args.simpoint_home
    bincmd = os.path.expandvars(args.bincmd)
    client_bincmd = ""
    if args.client_bincmd:
        client_bincmd = os.path.expandvars(args.client_bincmd)
    simpoint_mode = args.simpoint_mode
    drio_args = args.drio_args
    clustering_userk = args.clustering_userk
    manual_trace = args.manual_trace

    try:
        print("running run_simpoint_trace.py...")
        print(simpoint_mode)
        if simpoint_mode == "1": # clustering then tracing
            cluster_then_trace(workload, suite, simpoint_home, bincmd, client_bincmd, simpoint_mode, drio_args, clustering_userk, manual_trace)
        elif simpoint_mode == "2": # trace then post-process
            trace_then_cluster(workload, suite, simpoint_home, bincmd, client_bincmd, simpoint_mode, drio_args, clustering_userk)
        elif simpoint_mode == "3": # trace each timestep
            iterative(workload, suite, simpoint_home, bincmd, client_bincmd, simpoint_mode, drio_args, clustering_userk)
        elif simpoint_mode == "4": # cluster only (Phase 1 of parallel segment pipeline)
            cluster_only(workload, suite, simpoint_home, bincmd, client_bincmd, simpoint_mode, drio_args, clustering_userk)
        elif simpoint_mode == "5": # trace single segment (Phase 2 of parallel segment pipeline)
            if args.segment_id is None or args.cluster_id is None:
                raise Exception("Mode 5 requires --segment_id and --cluster_id")
            trace_single_segment(workload, suite, simpoint_home, bincmd, client_bincmd, simpoint_mode, drio_args, args.segment_id, args.cluster_id)
        else:
            raise Exception("Invalid simpoint mode")
    except Exception as e:
        traceback.print_exc() # Print the full stack trace
