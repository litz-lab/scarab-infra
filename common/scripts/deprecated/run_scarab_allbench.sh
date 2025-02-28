#!/bin/bash
set -x #echo on

# TODO: for other apps?
APPNAME="$1"
APP_GROUPNAME="$2"
BINCMD="$3"
SCENARIONUM="$4"
SCARABPARAMS="$5"
# this is fixed/settled for NON trace post-processing flow.
# for trace post-processing flow, SEGSIZE is read from file
SEGSIZE=100000000
SCARABMODE="$6"
SCARABARCH="$7"
TRACESSIMP="$8"
# 10M warmup by default
WARMUP=10000000

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


# TODO: get all cmd for spec in advance instead of in place
# Get command to run for Spe17
if [ "$APP_GROUPNAME" == "spec2017" ]; then
  # environment
  cd $tmpdir/cpu2017
  source ./shrc
  # compile and get command for application
  # TODO: this is just for one input
  ./bin/specperl ./bin/harness/runcpu --copies=1 --iterations=1 --threads=1 --config=memtrace --action=runsetup --size=train $APPNAME
  ogo $APPNAME run
  # TODO: the input size
  cd run_base_train*
  BINCMD=$(specinvoke -nn | tail -2 | head -1)
  for sub in $BINCMD; do
    if [[ -f $sub ]]; then
      # ref
      # https://stackoverflow.com/a/13210909
      # https://stackoverflow.com/a/7126780
      replace=$(readlink -f "$sub")
      BINCMD="${BINCMD/"$sub"/"$replace"}"
    fi
  done
fi

SCARABHOME=$HOME/scarab/

if [ "$SCARABMODE" == "4" ]; then
  # overwriting
  if [ "$TRACESSIMP" == "1" ]; then
    MODULESDIR=/simpoint_traces/$APPNAME/traces_simp/bin
    TRACEFILE=/simpoint_traces/$APPNAME/traces_simp/trace
  elif [ "$TRACESSIMP" == "2" ]; then
    MODULESDIR=/simpoint_traces/$APPNAME/traces_simp/
    TRACEFILE=/simpoint_traces/$APPNAME/traces_simp/
  fi

  SIMHOME=$HOME/simpoint_flow/simulations/$APPNAME
  mkdir -p $SIMHOME
  TRACEHOME=/simpoint_traces/$APPNAME

  cd $SIMHOME
  mkdir $SCENARIONUM

  cd $TRACEHOME/traces/whole
  # continue if only one trace file
  ###HEERREEE prepare raw dir, trace dir
  SPDIR=$TRACEHOME/simpoints/
  OUTDIR=$SIMHOME/$SCENARIONUM/

  segmentSizeFile="$TRACEHOME/fingerprint/segment_size"
  if [ ! -f $segmentSizeFile ]
  then
    echo "$segmentSizeFile does not exist"
    exit
  fi
  SEGSIZE=$(cat "$segmentSizeFile")
  echo "SEGSIZE read from $segmentSizeFile is $SEGSIZE"
  bash run_scarab_mode_4_allbench.sh "$SCARABHOME" "$MODULESDIR" "$TRACEFILE" "$SCARABPARAMS" "$SPDIR" "$SEGSIZE" "$OUTDIR" "$WARMUP" "$SCARABARCH" "$TRACESSIMP"
elif [ "$SCARABMODE" == "3" ]; then
  SIMHOME=$HOME/simpoint_flow/simulations/$APPNAME
  EVALHOME=$HOME/simpoint_flow/evaluations/$APPNAME
  mkdir -p $SIMHOME
  mkdir -p $EVALHOME
  TRACEHOME=/simpoint_traces/$APPNAME
  ################################################################
  # read in simpoint
  # ref: https://stackoverflow.com/q/56005842
  # map of cluster - segment
  declare -A clusterMap
  while IFS=" " read -r segID clusterID; do
    clusterMap[$clusterID]=$segID 
  done < $TRACEHOME/simpoints/opt.p.lpt0.99

  ################################################################
  # trace-based or exec-driven simulations
  taskPids=()
  start=`date +%s`
  # simulation in parallel -> use map of trace file
  for clusterID in "${!clusterMap[@]}"
  do
    mkdir -p $SIMHOME/$SCENARIONUM/$clusterID
    cp $SCARABHOME/src/PARAMS.$SCARABARCH $SIMHOME/$SCENARIONUM/$clusterID/PARAMS.in
    cd $SIMHOME/$SCENARIONUM/$clusterID

    segID=${clusterMap[$clusterID]}
    start_inst=$(( $segID * $SEGSIZE ))
    scarabCmd="
    python3 $SCARABHOME/bin/scarab_launch.py --program=\"$BINCMD\" \
    --simdir=\"$SIMHOME/$SCENARIONUM/$clusterID\" \
    --pintool_args=\"-hyper_fast_forward_count $start_inst\" \
    --scarab_args=\"--inst_limit $SEGSIZE $SCARABPARAMS\" \
    --scarab_stdout=\"$SIMHOME/$SCENARIONUM/$clusterID/scarab.out\" \
    --scarab_stderr=\"$SIMHOME/$SCENARIONUM/$clusterID/scarab.err\" \
    --pin_stdout=\"$SIMHOME/$SCENARIONUM/$clusterID/pin.out\" \
    --pin_stderr=\"$SIMHOME/$SCENARIONUM/$clusterID/pin.err\" \
    "

    echo "simulating cluster ${clusterID}..."
    echo "command: ${scarabCmd}"
    eval $scarabCmd &
    taskPids+=($!)
  done

  echo "wait for all simulations to finish..."
  for taskPid in ${taskPids[@]}; do
    if wait $taskPid; then
      echo "simulation process $taskPid success"
    else
      echo "simulation process $taskPid fail"
      exit
    fi
  done
  end=`date +%s`
  runtime=$((end-start))
  hours=$((runtime / 3600));
  minutes=$(( (runtime % 3600) / 60 ));
  seconds=$(( (runtime % 3600) % 60 ));
  echo "simulation Runtime: $hours:$minutes:$seconds (hh:mm:ss)"

  ################################################################
elif [ "$SCARABMODE" == "1" ] || [ "$SCARABMODE" == "2" ] || [ "$SCARABMODE" == "5" ]; then
  SIMHOME=$HOME/nonsimpoint_flow/simulations/$APPNAME
  EVALHOME=$HOME/nonsimpoint_flow/evaluations/$APPNAME
  mkdir -p $SIMHOME
  mkdir -p $EVALHOME
  TRACEHOME=/simpoint_traces/$APPNAME

  if [ "$SCARABMODE" == "2" ]; then
    traceMap=$(ls $TRACEHOME/traces/whole/)
  elif [ "$SCARABMODE" == "5" ]; then
    traceMap="trace.gz"
  fi
  ################################################################
  # trace-based or exec-driven simulations
  taskPids=()
  start=`date +%s`
  mkdir -p $SIMHOME/$SCENARIONUM
  cp $SCARABHOME/src/PARAMS.$SCARABARCH $SIMHOME/$SCENARIONUM/PARAMS.in
  cd $SIMHOME/$SCENARIONUM
  if [ "$SCARABMODE" == "2" ]; then
    scarabCmd="$SCARABHOME/src/scarab --frontend memtrace --cbp_trace_r0=$TRACEHOME/traces/whole/${traceMap} --memtrace_modules_log=$TRACEHOME/traces/raw/ $SCARABPARAMS &> sim.log"
  elif [ "$SCARABMODE" == "5" ]; then
    scarabCmd="$SCARABHOME/src/scarab --full_warmup 49999999 --frontend pt --cbp_trace_r0=$TRACEHOME/${traceMap} $SCARABPARAMS &> sim.log"
  else
    start_inst=100000000
    scarabCmd="
    python3 $SCARABHOME/bin/scarab_launch.py --program=\"$BINCMD\" \
      --simdir=\"$SIMHOME/$SCENARIONUM/\" \
      --pintool_args=\"-hyper_fast_forward_count $start_inst\" \
      --scarab_args=\"--inst_limit $SEGSIZE $SCARABPARAMS\" \
      --scarab_stdout=\"$SIMHOME/$SCENARIONUM/scarab.out\" \
      --scarab_stderr=\"$SIMHOME/$SCENARIONUM/scarab.err\" \
      --pin_stdout=\"$SIMHOME/$SCENARIONUM/pin.out\" \
      --pin_stderr=\"$SIMHOME/$SCENARIONUM/pin.err\" \
      "
  fi
  echo "simulating ..."
  echo "command: ${scarabCmd}"
  eval $scarabCmd &
  taskPids+=($!)

  echo "wait for all simulations to finish..."
  for taskPid in ${taskPids[@]}; do
    if wait $taskPid; then
      echo "simulation process $taskPid success"
    else
      echo "simulation process $taskPid fail"
      exit
    fi
  done
  end=`date +%s`
  runtime=$((end-start))
  hours=$((runtime / 3600));
  minutes=$(( (runtime % 3600) / 60 ));
  seconds=$(( (runtime % 3600) % 60 ));
  echo "simulation Runtime: $hours:$minutes:$seconds (hh:mm:ss)"

  ################################################################
fi
