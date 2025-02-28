from gather_cluster_results import *
import os, sys
import subprocess
from time import sleep
import shutil
# pip install plotly==5.18.0
import plotly.graph_objects as go

# ref: https://stackoverflow.com/a/13197763
class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

def get_top_simpoint(simpoints):
    max_weight = 0
    max_index = 0
    for simp_i, simp in enumerate(simpoints):
        if simp.weight > max_weight:
            max_weight = simp.weight
            max_index = simp_i
    
    return simpoints[max_index]

# ub: upper bound in M
def run_vary_warmup_legth(SCARABHOME, MODULESDIR, TRACEFILE, OUTDIR, segID, SEGSIZE, ub):
    p_list = []
    warmup_unit = 1000000
    seg_root = OUTDIR + "/" + str(segID)
    os.makedirs(seg_root, exist_ok=True)
    for WARMUP in range(ub+1):
        warmup_dir = seg_root + "/" + str(WARMUP)
        os.makedirs(warmup_dir)
        
        # convert to actual instruction number
        WARMUP = WARMUP * warmup_unit
        with cd(warmup_dir):
            shutil.copyfile(SCARABHOME +"/src/PARAMS.sunny_cove",
                            warmup_dir + "/PARAMS.in")

            roiStart = segID * SEGSIZE + 1
            roiEnd = segID * SEGSIZE + SEGSIZE

            if roiStart > WARMUP:
                roiStart = roiStart - WARMUP
            else:
                # no enough preceding instructions, can only warmup till segment start
                WARMUP = roiStart - 1
                # new roi start is the very first instruction of the trace
                roiStart = 1

            instLimit = roiEnd - roiStart + 1

            # scarabCmd="$SCARABHOME/src/scarab \
            # --frontend memtrace \
            # --cbp_trace_r0=$TRACEFILE \
            # --memtrace_modules_log=$MODULESDIR \
            # --memtrace_roi_begin=$roiStart \
            # --memtrace_roi_end=$roiEnd \
            # --inst_limit=$instLimit \
            # --full_warmup=$WARMUP \
            # $SCARABPARAMS \
            # &> sim.log"

            executable = SCARABHOME + "/src/scarab"
            scarab_cmd = [executable,
                        "--frontend", "memtrace",
                        "--cbp_trace_r0", TRACEFILE,
                        "--memtrace_modules_log", MODULESDIR,
                        "--memtrace_roi_begin", str(roiStart),
                        "--memtrace_roi_end", str(roiEnd),
                        "--inst_limit", str(instLimit),
                        "--full_warmup", str(WARMUP)
                        ]
            # log file at cur dir
            log_file = "./sim.log"
            print(scarab_cmd, flush=True)
            with open(log_file, "w") as outfile:
                p_list.append(subprocess.Popen(scarab_cmd, stdout=outfile, shell=False))
            while True:
                live = 0
                for p in p_list:
                    if p.poll() is None:
                        live += 1
                if live < 40:
                    break
                else:
                    sleep(120)

    print("wait for all warm-up runs to finish...")
    for p in p_list:
        p.wait()

def plot(OUTDIR, segID, ub):
    seg_root = OUTDIR + "/" + str(segID)
    warmup_dir_list = [seg_root + "/" + str(WARMUP) for WARMUP in range(ub+1)]

    for g in stat_groups:
        fig = go.Figure()
        for s in g.s_list:
            y_vals=[get_acc_stat_from_file(warmup_dir + "/" + g.f_name, s.s_name, s.pos) for warmup_dir in warmup_dir_list]
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(y_vals))),
                    y=y_vals,
                    name=s.s_name,
                    showlegend=True
                    # visible='legendonly'
                )
            )
        fig.update_layout(
            title = g.g_name
        )
        with open("{}/warmup.html".format(seg_root), 'a') as f:
            f.write(fig.to_html(full_html=False, include_plotlyjs="cdn"))

if __name__ == "__main__":
    SIMPOINTDIR=sys.argv[1]
    if not os.path.isdir(SIMPOINTDIR):
        print("simpoint directory {} does not exist!".format(SIMPOINTDIR))
        exit
    SCARABHOME=sys.argv[2]
    if not os.path.isdir(SCARABHOME):
        print("scarab directory {} does not exist!".format(SCARABHOME))
        exit
    MODULESDIR=sys.argv[3]
    if not os.path.isdir(MODULESDIR):
        print("modules directory {} does not exist!".format(MODULESDIR))
        exit
    TRACEFILE=sys.argv[4]
    if not os.path.isfile(TRACEFILE):
        print("trace file {} does not exist!".format(TRACEFILE))
        exit
    OUTDIR=sys.argv[5]
    if not os.path.isdir(OUTDIR):
        print("output directory {} does not exist!".format(OUTDIR))
        exit
    SEGSIZE=int(sys.argv[6])
    print("SEGSIZE is {}".format(SEGSIZE))

    simpoints = read_simpoints(SIMPOINTDIR, "not applicable")
    top_simpoint = get_top_simpoint(simpoints)
    print("top simp: {} {}".format(top_simpoint.seg_id, top_simpoint.weight))
    # def run_vary_warmup_legth(SCARABHOME, MODULESDIR, TRACEFILE, OUTDIR, segID, SEGSIZE, ub):
    run_vary_warmup_legth(SCARABHOME, MODULESDIR, TRACEFILE, OUTDIR, top_simpoint.seg_id, SEGSIZE, 300)
    # def plot(OUTDIR, segID, ub):
    plot(OUTDIR, top_simpoint.seg_id, 300)