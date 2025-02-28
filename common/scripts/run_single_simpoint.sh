#!/bin/bash
source utilities.sh

set -x #echo on

set_app_tracefile () {
  case $APPNAME in
    mysql)
      MODULESDIR=/simpoint_traces/mysql/traces/whole/drmemtrace.mysqld.123776.5088.dir/raw
      TRACEFILE=/simpoint_traces/mysql/traces/whole/drmemtrace.mysqld.123776.5088.dir/trace/drmemtrace.mysqld.123827.6272.trace.zip
      ;;
    postgres)
      MODULESDIR=/simpoint_traces/postgres/traces/whole/drmemtrace.postgres.10865.1082.dir/raw
      TRACEFILE=/simpoint_traces/postgres/traces/whole/drmemtrace.postgres.10865.1082.dir/trace/drmemtrace.postgres.10865.3710.trace.zip
      ;;
    clang)
      MODULESDIR=/simpoint_traces/clang/traces/whole/drmemtrace.clang.03072.7752.dir/raw
      TRACEFILE=/simpoint_traces/clang/traces/whole/drmemtrace.clang.03072.7752.dir/trace/drmemtrace.clang.03072.4467.trace.zip
      ;;
    gcc)
      MODULESDIR=/simpoint_traces/gcc/traces/whole/drmemtrace.cc1.04250.2989.dir/raw
      TRACEFILE=/simpoint_traces/gcc/traces/whole/drmemtrace.cc1.04250.2989.dir/trace/drmemtrace.cc1.04250.5506.trace.zip
      ;;
    mongodb)
      MODULESDIR=/simpoint_traces/mongodb/traces/whole/drmemtrace.mongod.04280.8169.dir/raw
      TRACEFILE=/simpoint_traces/mongodb/traces/whole/drmemtrace.mongod.04280.8169.dir/trace/drmemtrace.mongod.04332.7098.trace.zip
      ;;
    verilator)
      MODULESDIR=/simpoint_traces/verilator/traces/whole/raw
      TRACEFILE=/simpoint_traces/verilator/traces/whole/trace/drmemtrace.emulator-freechips.rocketchip.system-DefaultConfigN8.00025.6005.trace.zip
      ;;
    xgboost)
      MODULESDIR=/simpoint_traces/xgboost/traces/whole/drmemtrace.python3.8.00025.6828.dir/raw
      TRACEFILE=/simpoint_traces/xgboost/traces/whole/drmemtrace.python3.8.00025.6828.dir/trace/drmemtrace.python3.8.00025.0843.trace.zip
      ;;
    memcached)
      MODULESDIR=/simpoint_traces/memcached/traces/whole/drmemtrace.memcached.07432.6868.dir/raw
      TRACEFILE=/simpoint_traces/memcached/traces/whole/drmemtrace.memcached.07432.6868.dir/trace/drmemtrace.memcached.07434.0028.trace.zip
      ;;
    redis)
      MODULESDIR=/simpoint_traces/redis/traces/drmemtrace.redis-server.40792.8757.dir/raw/
      TRACEFILE=/simpoint_traces/redis/traces/drmemtrace.redis-server.40792.8757.dir/trace/drmemtrace.redis-server.40792.6868.trace.zip
      ;;
    rocksdb)
      MODULESDIR=/simpoint_traces/rocksdb/traces_simp/raw/
      TRACEFILE=/simpoint_traces/rocksdb/traces_simp/trace/
      mode="1"
      ;;
    600.perlbench_s)
      MODULESDIR=/simpoint_traces/600.perlbench_s/traces/whole/drmemtrace.perlbench_s_base.memtrace-m64.11679.5983.dir/raw
      TRACEFILE=/simpoint_traces/600.perlbench_s/traces/whole/drmemtrace.perlbench_s_base.memtrace-m64.11679.5983.dir/trace/drmemtrace.perlbench_s_base.memtrace-m64.11679.4703.trace.zip
      ;;
    602.gcc_s)
      MODULESDIR=/simpoint_traces/602.gcc_s/traces/whole/drmemtrace.sgcc_base.memtrace-m64.66312.0508.dir/raw
      TRACEFILE=/simpoint_traces/602.gcc_s/traces/whole/drmemtrace.sgcc_base.memtrace-m64.66312.0508.dir/trace/drmemtrace.sgcc_base.memtrace-m64.66312.8159.trace.zip
      ;;
    605.mcf_s)
      MODULESDIR=/simpoint_traces/605.mcf_s/traces/whole/drmemtrace.mcf_s_base.memtrace-m64.66517.6766.dir/raw
      TRACEFILE=/simpoint_traces/605.mcf_s/traces/whole/drmemtrace.mcf_s_base.memtrace-m64.66517.6766.dir/trace/drmemtrace.mcf_s_base.memtrace-m64.66517.4453.trace.zip
      ;;
    620.omnetpp_s)
      MODULESDIR=/simpoint_traces/620.omnetpp_s/traces/whole/drmemtrace.omnetpp_s_base.memtrace-m64.11305.6465.dir/raw
      TRACEFILE=/simpoint_traces/620.omnetpp_s/traces/whole/drmemtrace.omnetpp_s_base.memtrace-m64.11305.6465.dir/trace/drmemtrace.omnetpp_s_base.memtrace-m64.11305.0389.trace.zip
      ;;
    623.xalancbmk_s)
      MODULESDIR=/simpoint_traces/623.xalancbmk_s/traces/whole/drmemtrace.xalancbmk_s_base.memtrace-m64.68960.4320.dir/raw
      TRACEFILE=/simpoint_traces/623.xalancbmk_s/traces/whole/drmemtrace.xalancbmk_s_base.memtrace-m64.68960.4320.dir/trace/drmemtrace.xalancbmk_s_base.memtrace-m64.68960.4051.trace.zip
      ;;
    625.x264_s)
      MODULESDIR=/simpoint_traces/625.x264_s/traces/whole/drmemtrace.x264_s_base.memtrace-m64.69655.2784.dir/raw
      TRACEFILE=/simpoint_traces/625.x264_s/traces/whole/drmemtrace.x264_s_base.memtrace-m64.69655.2784.dir/trace/drmemtrace.x264_s_base.memtrace-m64.69655.8596.trace.zip
      ;;
    641.leela_s)
      MODULESDIR=/simpoint_traces/641.leela_s/traces/whole/drmemtrace.leela_s_base.memtrace-m64.69890.0911.dir/raw
      TRACEFILE=/simpoint_traces/641.leela_s/traces/whole/drmemtrace.leela_s_base.memtrace-m64.69890.0911.dir/trace/drmemtrace.leela_s_base.memtrace-m64.69890.6754.trace.zip
      ;;
    648.exchange2_s)
      MODULESDIR=/simpoint_traces/648.exchange2_s/traces/whole/drmemtrace.exchange2_s_base.memtrace-m64.70065.6658.dir/raw
      TRACEFILE=/simpoint_traces/648.exchange2_s/traces/whole/drmemtrace.exchange2_s_base.memtrace-m64.70065.6658.dir/trace/drmemtrace.exchange2_s_base.memtrace-m64.70065.5851.trace.zip
      ;;
    657.xz_s)
      MODULESDIR=/simpoint_traces/657.xz_s/traces/whole/drmemtrace.xz_s_base.memtrace-m64.70645.3373.dir/raw
      TRACEFILE=/simpoint_traces/657.xz_s/traces/whole/drmemtrace.xz_s_base.memtrace-m64.70645.3373.dir/trace/drmemtrace.xz_s_base.memtrace-m64.70645.7323.trace.zip
      ;;
    *)
      echo "unknown application"
      ;;
  esac
}

echo "Running on $(hostname)"

# TODO: for other apps?
APPNAME="$1"
APP_GROUPNAME="$2"
SCENARIO="$3"
SCARABPARAMS="$4"
# this is fixed/settled for NON trace post-processing flow.
# for trace post-processing flow, SEGSIZE is read from file
SEGSIZE=100000000
SCARABMODE="$5"
SCARABARCH="$6"
TRACESSIMP="$7"
SCARABHOME="$8"
SEGMENT_ID="$9"

if [ "$APP_GROUPNAME" == "allbench_traces" ] || [ "$APP_GROUPNAME" == "isca2024_udp" ]; then
  # 10M warmup by default
  WARMUP=10000000
else
  WARMUP=50000000
fi

set_app_bincmd
set_app_tracefile

# overwriting
if [ "$TRACESSIMP" == "1" ]; then
  MODULESDIR=/simpoint_traces/$APPNAME/traces_simp/bin
  TRACEFILE=/simpoint_traces/$APPNAME/traces_simp/trace
fi

SIMHOME=$SCENARIO/$APPNAME
mkdir -p $SIMHOME
TRACEHOME=/simpoint_traces/$APPNAME

cd $SIMHOME

cd $TRACEHOME/traces/whole
# continue if only one trace file
###HEERREEE prepare raw dir, trace dir
SPDIR=$TRACEHOME/simpoints/
OUTDIR=$SIMHOME

segmentSizeFile="$TRACEHOME/fingerprint/segment_size"
if [ ! -f $segmentSizeFile ]
then
  echo "$segmentSizeFile does not exist"
  exit
fi
SEGSIZE=$(cat "$segmentSizeFile")
echo "SEGSIZE read from $segmentSizeFile is $SEGSIZE"


# This part comes from the beginning of run_scarab_mode_4_allbench.sh
# if TRACESSIMP is 1,
# TRACEFILE is supposed to be traces_simp FOLDER
if [ "$TRACESSIMP" == "1" ]; then
    if [ ! -d $TRACEFILE ]; then
        echo "TRACEFILE is supposed to be traces_simp FOLDER"
        exit
    fi
fi

# This part is an unrolled version of the loop int run_scarab_mode_4_allbench.sh
echo "SEGMENT ID: $segID"
segID=$SEGMENT_ID
mkdir -p $OUTDIR/$segID
cp $SCARABHOME/src/PARAMS.$SCARABARCH $OUTDIR/$segID/PARAMS.in
cd $OUTDIR/$segID

# roi is initialized by original segment boundary without warmup
roiStart=$(( $segID * $SEGSIZE + 1 ))
roiEnd=$(( $segID * $SEGSIZE + $SEGSIZE ))

# now reset roi start based on warmup:
# roiStart + WARMUP = original segment start
if [ "$roiStart" -gt "$WARMUP" ]; then
    # enough room for warmup, extend roi start to the left
    roiStart=$(( $roiStart - $WARMUP ))
else
    # no enough preceding instructions, can only warmup till segment start
    WARMUP=$(( $roiStart - 1 ))
    # new roi start is the very first instruction of the trace
    roiStart=1
fi

instLimit=$(( $roiEnd - $roiStart + 1 ))

if [ "$TRACESSIMP" != "1" ]; then
    echo "!TRACESSIMP"
    scarabCmd="$SCARABHOME/src/scarab \
    --frontend memtrace \
    --cbp_trace_r0=$TRACEFILE \
    --memtrace_modules_log=$MODULESDIR \
    --memtrace_roi_begin=$roiStart \
    --memtrace_roi_end=$roiEnd \
    --inst_limit=$instLimit \
    --full_warmup=$WARMUP \
    --use_fetched_count=1 \
    $SCARABPARAMS \
    &> sim.log"
elif [ "$TRACESSIMP" == "1" ]; then
    echo "TRACESSIMP"
    # with TRACESSIMP
    # simultion uses the specific trace file
    # the roiStart is the second chunk, which is assumed to be segment size
    #### if chunk zero chunk is part of the simulation, the roiStart is the first chunk
    # the roiEnd is always the end of the trace -- (dynamorio uses 0)
    # the warmup is the same

    # roiStart 1 means simulation starts with chunk 0
    if [ "$roiStart" == "1" ]; then
        echo "ROISTART"
        echo "$TRACEFILE"
        echo "$segID"
        scarabCmd="$SCARABHOME/src/scarab \
        --frontend memtrace \
        --cbp_trace_r0=$TRACEFILE/$segID.zip \
        --memtrace_modules_log=$MODULESDIR \
        --memtrace_roi_begin=1 \
        --memtrace_roi_end=$instLimit \
        --inst_limit=$instLimit \
        --full_warmup=$WARMUP \
        --use_fetched_count=1 \
        $SCARABPARAMS \
        &> sim.log"
    else
        echo "!ROISTART"
        scarabCmd="$SCARABHOME/src/scarab \
        --frontend memtrace \
        --cbp_trace_r0=$TRACEFILE/$segID.zip \
        --memtrace_modules_log=$MODULESDIR \
        --memtrace_roi_begin=$(( $SEGSIZE + 1)) \
        --memtrace_roi_end=$(( $SEGSIZE + $instLimit )) \
        --inst_limit=$instLimit \
        --full_warmup=$WARMUP \
        --use_fetched_count=1 \
        $SCARABPARAMS \
        &> sim.log"
    fi

fi

echo "simulating clusterID ${clusterID}, segment $segID..."
echo "command: ${scarabCmd}"
eval $scarabCmd &
wait $!

# Issues. See sim.log in new_experiment20.
# Failed to open /simpoint_traces/postgres/traces_simp/trace/postgres0.zip
# CMD:  docker exec --user aesymons --workdir /home/aesymons --privileged allbench_traces_aesymons slurm_payload.sh "postgres" "allbench_traces" "" "new_experiment20/fe_ftq_block_num.16" "--inst_limit 99900000 --fdip_enable 1 --fe_ftq_block_num 16" "4" "sunny_cove" "1" /home/aesymons/new_experiment20/scarab "3954"