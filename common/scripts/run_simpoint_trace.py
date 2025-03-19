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

def find_trace_files(base_path):
    trace_files = glob.glob(os.path.join(base_path, "drmemtrace.*.dir/trace/dr*.trace.zip"))
    dr_folders = glob.glob(os.path.join(base_path, "drmemtrace.*.dir"))
    return trace_files, dr_folders

def get_largest_trace(base_path):
    traces = []
    for trace in glob.glob(os.path.join(base_path, "*/trace/dr*.trace.zip")):
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

def minimize_simpoint_traces(cluster_map, workload_home):
    ################################################################
    # minimize traces, rename traces
    # it is possible that SimPoint picks interval zero,
    # in that case the simulation would only need one chunk,
    # but we always keep two regardlessly
    try:
        for cluster_id, segment_id in cluster_map.items():
            trace_dir = os.path.join(workload_home, "traces_simp", segment_id)
            trace_files = subprocess.getoutput(f"find {trace_dir} -name 'dr*.trace.zip' | grep 'drmemtrace.*.trace.zip'").splitlines()
            num_traces = len(trace_files)
            if num_traces != 1:
                trace_clustering_info = {}
                trace_clustering_info["err"] = "There are multiple or no bbfp files. This simpoint flow would not work."
                with open(os.path.join(workload_home, "trace_clustering_info.json"), "w") as json_file:
                    json.dump(trace_clustering_info, json_file, indent=2, separators=(",", ":"))
                exit(1)
            trace_file = trace_files[0]
            big_zip_file = os.path.join(trace_dir, "trace", f"{segment_id}.big.zip")
            subprocess.run(f"mv {trace_file} {big_zip_file}", check=True, shell=True)
            unzip_output = subprocess.getoutput(f"unzip -l {big_zip_file}")
            num_chunk = len([line for line in unzip_output.splitlines() if 'chunk.' in line])

            if num_chunk < 2:
                print(f"WARN: the big trace {segment_id} contains less than 2 chunks: {num_chunk} !")


            # Copy chunk 0 and chunk 1 into a new zip file
            subprocess.run(f"zip {big_zip_file} --copy chunk.0000 chunk.0001 --out {os.path.join(trace_path, f'{segment_id}.zip')}", shell=True)

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
    try:
        start_time = time.perf_counter()
        subprocess.run(["mkdir", "-p", f"{simpoint_home}/{workload}/traces/whole"], check=True, capture_output=True, text=True)
        workload_home = f"{simpoint_home}/{workload}"
        dynamorio_home = os.environ.get('DYNAMORIO_HOME')
        trace_cmd = f"{dynamorio_home}/bin64/drrun -t drcachesim -jobs 40 -outdir {workload_home}/traces/whole -offline"
        if drio_args != None:
            trace_cmd = f"{trace_cmd} {drio_args}"
        trace_cmd = f"{trace_cmd} -- {bincmd}"
        trace_cmd_list = shlex.split(trace_cmd)

        print("whole app tracing..")
        if client_bincmd:
            subprocess.Popen("exec " + client_bincmd, stdout=subprocess.PIPE, shell=True)
        subprocess.run(trace_cmd_list, check=True, capture_output=True, text=True)
        end_time = time.perf_counter()
        report_time("whole app tracing done", start_time, end_time)

        print("whole app raw2trace..")
        start_time = time.perf_counter()
        whole_trace_path = f"{workload_home}/traces/whole"
        raw2trace_processes = set()
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
                    raw2trace_cmd = f"{dynamorio_home}/tools/bin64/drraw2trace -jobs 40 -indir {raw_path} -chunk_instr_count {chunk_size}"
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

        trace_clustering_info = {}
        if num_traces == 1 and num_dr_folder == 1:
            modules_dir = os.path.dirname(glob.glob(os.path.join(whole_trace_path, "drmemtrace.*.dir/raw/modules.log"))[0])
            whole_trace = glob.glob(os.path.join(whole_trace_path, "drmemtrace.*.dir/trace/dr*.zip"))[0]
            dr_folder = os.path.dirname(os.path.dirname(whole_trace))
        elif num_traces == num_dr_folder:
            whole_trace = get_largest_trace(whole_trace_path)
            dr_folder = os.path.dirname(os.path.dirname(whole_trace))
            modules_dir = os.path.dirname(os.path.join(dr_folder, "raw/modules.log"))
        elif num_traces > num_dr_folder:
            whole_trace = get_largest_trace(whole_trace_path)
            dr_folder = os.path.dirname(os.path.dirname(whole_trace))
            modules_dir = os.path.dirname(os.path.join(dr_folder, "raw/modules.log"))
        else:
            trace_clustering_info["err"] = "There are multiple trace files. Decide one and run manually: /usr/local/bin/run_trace_post_processing.sh <OUTDIR> <MODULESDIR> <TRACEFILE> <CHUNKSIZE> <SEG_SIZE> <SIMPOINTHOME>, then run /usr/local/bin/run_clustering.sh <FPFILE> <OUTDIR>"

            with open(os.path.join(workload_home, "trace_clustering_info.json"), "w") as json_file:
                json.dump(trace_clustering_info, json_file, indent=2, separators=(",", ":"))
            exit(1)
        post_processing_cmd = f"/bin/bash /usr/local/bin/run_trace_post_processing.sh {workload_home} {modules_dir} {whole_trace} {chunk_size} {seg_size} {simpoint_home}"
        subprocess.run([post_processing_cmd], check=True, shell=True, stdin=open(os.devnull, 'r'))
        trace_file = os.path.basename(whole_trace)
        dr_folder = os.path.basename(dr_folder)
        trace_clustering_info["modules_dir"] = modules_dir
        trace_clustering_info["whole_trace"] = whole_trace
        trace_clustering_info["dr_folder"] = dr_folder
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
        minimize_trace_cmd = f"/bin/bash /usr/local/bin/minimize_trace.sh {dr_folder}/bin {whole_trace} {workload_home}/simpoints 1 {workload_home}/traces_simp"
        subprocess.run([minimize_trace_cmd], check=True, shell=True, stdin=open(os.devnull, 'r'))
        end_time = time.perf_counter()
        report_time("minimizing traces done", start_time, end_time)
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
    try:
        os.makedirs(os.path.join(simpoint_home, workload, "fingerprint"), exist_ok=True)
        os.makedirs(os.path.join(simpoint_home, workload, "traces_simp"), exist_ok=True)
        workload_home = f"{simpoint_home}/{workload}"
        dynamorio_home = os.environ.get('DYNAMORIO_HOME')
        print("generate fingerprint..")
        if client_bincmd:
            subprocess.Popen("exec " + client_bincmd, stdout=subprocess.PIPE, shell=True)
        start_time = time.perf_counter()
        fp_cmd = f"{dynamorio_home}/bin64/drrun -max_bb_instrs 4096 -opt_cleancall 2 -c $tmpdir/libfpg.so -no_use_bb_pc -no_use_fetched_count -segment_size {seg_size} -output {workload_home}/fingerprint/bbfp -pcmap_output {workload_home}/fingerprint/pcmap -- {bincmd}"
        subprocess.run([fp_cmd], check=True, capture_output=True, text=True)
        end_time = time.perf_counter()
        report_time("generate fingerprint done", start_time, end_time)

        print("clustering..")
        start_time = time.perf_counter()
        trace_clustering_info = {}
        bbfp_files = glob.glob(os.path.join(f"{workload_home}/fingerprint", "bbfp.*"))
        num_bbfp = len(bbfp_files)
        if num_bbfp == 1:
            bbfp_file = bbfp_files[0]
            clustering_cmd = f"/bin/bash /usr/local/bin/run_clustering.sh {bbfp_file} {workload_home}"
            if clustering_userk != None:
                clustering_cmd = f"{clustering_cmd} {clustering_userk}"
            subprocess.run([clustering_cmd], check=True, shell=True, stdin=open(os.devnull, 'r'))
        else:
            trace_clustering_info["err"] = "There are multiple or no bbfp files. This simpoint flow would not work."
            with open(os.path.join(workload_home, "trace_clustering_info.json"), "w") as json_file:
                json.dump(trace_clustering_info, json_file, indent=2, separators=(",", ":"))
            exit(1)
        end_time = time.perf_counter()
        report_time("clustering", start_time, end_time)

        cluster_map = get_cluster_map(workload_home)

        print("clustering tracing..")
        start_time = time.perf_counter()
        cluster_tracing_processes = set()
        for cluster_id, segment_id in cluster_map.items():
            seg_dir = os.path.join(workload_home, "traces_simp", str(segment_id))
            os.makedirs(seg_dir, exist_ok=True)
            # the simulation region, in the unit of chunks
            roi_start = segment_id * seg_size
            # seq is inclusive
            roi_end = roi_start + seg_size

            # assume warm-up length is the segsize
            # this will limit the amount of warmup that can be done during the simulation
            warmup = seg_size

            if roi_start > warmup:
                # enough room for warmup, extend roi start to the left
                roi_start -= warmup
            else:
                # no enough preceding instructions, can only warmup till segment start
                # new roi start is the very first instruction of the trace
                roi_start = 0

            roi_length = roi_end - roi_start
            if manual_trace:
                # because we are using exit_after_tracing,
                # want to to pad saome extra so we can likely have enough instrs
                roi_length += 8 * seg_size
            else:
                # pad even more
                roi_length += 2 * seg_size

            if roi_start == 0:
                trace_cmd = f"{dynamorio_home}/bin64/drrun -t drcachesim -jobs 40 -outdir {seg_dir} -offline -exit_after_tracing {roi_length} -- {bincmd}"
            else:
                trace_cmd = f"{dynamorio_home}/bin64/drrun -t drcachesim -jobs 40 -outdir {seg_dir} -offline -trace_after_instrs {roi_start} -exit_after_tracing {roi_length} -- {bincmd}"

            process = subprocess.Popen("exec " + trace_cmd, stdout=subprocess.PIPE, shell=True)
            cluster_tracing_processes.add(process)

        for p in cluster_tracing_processes:
            p.wait()
        end_time = time.perf_counter()
        report_time("cluster tracing done", start_time, end_time)

        print("clustered traces raw2trace..")
        start_time = time.perf_counter()
        raw2trace_processes = set()
        for cluster_id, segment_id in cluster_map.items():
            trace_path = f"{workload_home}/traces_simp/{segment_id}"
            trace_files, dr_folders = find_trace_files(trace_path)
            num_dr_folder = len(dr_folders)
            if num_dr_folder != 1:
                trace_clustering_info["err"] = "There are multiple or no bbfp files. This simpoint flow would not work."
                with open(os.path.join(workload_home, "trace_clustering_info.json"), "w") as json_file:
                    json.dump(trace_clustering_info, json_file, indent=2, separators=(",", ":"))
                exit(1)

            subprocess.run(f"mv {trace_path}/dr*/raw/ {trace_path}/raw/", check=True, shell=True)
            os.makedirs(f"{trace_path}/bin", exist_ok=True)
            bin_path = os.path.join(trace_path, "bin")
            raw_path = os.path.join(trace_path, "raw")
            os.chmod(bin_path, 0o777)
            os.chmod(raw_path, 0o777)
            subprocess.run(f"cp {raw_path}/modules.log {bin_path}/modules.log", check=True, shell=True)
            subprocess.run(f"cp {raw_path}/modules.log {raw_path}/modules.log.bak", check=True, shell=True)
            subprocess.run(["python2", f"{simpoint_home}/scarab/utils/memtrace/portabilize_trace.py", f"{trace_path}"], capture_output=True, text=True, check=True)
            subprocess.run(f"cp {bin_path}/modules.log {raw_path}/modules.log", check=True, shell=True)
            raw2trace_cmd = f"{dynamorio_home}/tools/bin64/drraw2trace -jobs 40 -indir {raw_path} -chunk_instr_count {chunk_size}"
            process = subprocess.Popen("exec " + raw2trace_cmd, stdout=subprocess.PIPE, shell=True)
            raw2trace_processes.add(process)

        for p in raw2trace_processes:
            p.wait()

        end_time = time.perf_counter()
        report_time("clustered traces raw2trace done", start_time, end_time)

        print("minimize traces..")
        start_time = time.perf_counter()
        minimze_simpoint_traces(cluster_map, workload_home)
        end_time = time.perf_counter()
        report_time("minimize traces done", start_time, end_time)
    except Exception as e:
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
        if simpoint_mode == "1": # clustering then tracing
            cluster_then_trace(workload, suite, simpoint_home, bincmd, client_bincmd, simpoint_mode, drio_args, clustering_userk, manual_trace)
        elif simpoint_mode == "2": # trace then post-process
            trace_then_cluster(workload, suite, simpoint_home, bincmd, client_bincmd, simpoint_mode, drio_args, clustering_userk)
    except Exception as e:
        traceback.print_exc() # Print the full stack trace
