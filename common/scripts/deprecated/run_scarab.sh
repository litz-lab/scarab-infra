#!/bin/bash

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
# 50M warmup by default
WARMUP=50000000

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

if [ "$SCARABMODE" == "4" ]; then
  cd $HOME/simpoint_flow/$APPNAME
  mkdir -p simulations
  APPHOME=$HOME/simpoint_flow/$APPNAME

  cd simulations
  mkdir $SCENARIONUM

  cd $APPHOME/traces/whole
  # continue if only one trace file
  numTrace=$(find -name "dr*.trace.zip" | grep "drmemtrace.*.trace.zip" | wc -l)
  numDrFolder=$(find -type d -name "drmemtrace.*.dir" | grep "drmemtrace.*.dir" | wc -l)
  if [ "$numTrace" == "1" ] && [ "$numDrFolder" == "1" ]; then
    ###HEERREEE prepare raw dir, trace dir
    SCARABHOME=$HOME/scarab/
    SPDIR=$APPHOME/simpoints/
    OUTDIR=$APPHOME/simulations/$SCENARIONUM/
    modulesDir=$(dirname $(ls $APPHOME/traces/whole/drmemtrace.*.dir/raw/modules.log))
    wholeTrace=$(ls $APPHOME/traces/whole/drmemtrace.*.dir/trace/dr*.zip)
    echo "modulesDIR: $modulesDir"
    echo "wholeTrace: $wholeTrace"

    segmentSizeFile="$APPHOME/fingerprint/segment_size"
    if [ ! -f $segmentSizeFile ]
    then
            echo "$segmentSizeFile does not exist"
            exit
    fi
    SEGSIZE=$(cat "$segmentSizeFile")
    echo "SEGSIZE read from $segmentSizeFile is $SEGSIZE"
    bash run_scarab_mode_4.sh "$SCARABHOME" "$MODULESDIR" "$TRACEFILE" "$SCARABPARAMS" "$SPDIR" "$SEGSIZE" "$OUTDIR" "$WARMUP" "$SCARABARCH"
  else
  # otherwise ask the user to run manually
    echo -e "There are multiple trace files.\n\
    Decide and run \"/usr/local/bin/run_scarab_mode_4.sh <SCARABHOME> <MODULESDIR> <TRACEFILE> "<SCARABPARAMS>" <SPDIR> <SEGSIZE> <OUTDIR> <WARMUP> <SCARABARCH>\""
    exit
  fi
elif [ "$SCARABMODE" == "3" ]; then
  cd $HOME/simpoint_flow/$APPNAME
  mkdir -p simulations evaluations
  APPHOME=$HOME/simpoint_flow/$APPNAME
  ################################################################
  # read in simpoint
  # ref: https://stackoverflow.com/q/56005842
  # map of cluster - segment
  declare -A clusterMap
  while IFS=" " read -r segID clusterID; do
    clusterMap[$clusterID]=$segID 
  done < $APPHOME/simpoints/opt.p.lpt0.99

  ################################################################
  # trace-based or exec-driven simulations
  taskPids=()
  start=`date +%s`
  # simulation in parallel -> use map of trace file
  for clusterID in "${!clusterMap[@]}"
  do
    mkdir -p $APPHOME/simulations/$SCENARIONUM/$clusterID
    cp $HOME/scarab/src/PARAMS.$SCARABARCH $APPHOME/simulations/$SCENARIONUM/$clusterID/PARAMS.in
    cd $APPHOME/simulations/$SCENARIONUM/$clusterID

    segID=${clusterMap[$clusterID]}
    start_inst=$(( $segID * $SEGSIZE ))
    scarabCmd="
    python3 $HOME/scarab/bin/scarab_launch.py --program=\"$BINCMD\" \
    --simdir=\"$APPHOME/simulations/$SCENARIONUM/$clusterID\" \
    --pintool_args=\"-hyper_fast_forward_count $start_inst\" \
    --scarab_args=\"--inst_limit $SEGSIZE $SCARABPARAMS\" \
    --scarab_stdout=\"$APPHOME/simulations/$SCENARIONUM/$clusterID/scarab.out\" \
    --scarab_stderr=\"$APPHOME/simulations/$SCENARIONUM/$clusterID/scarab.err\" \
    --pin_stdout=\"$APPHOME/simulations/$SCENARIONUM/$clusterID/pin.out\" \
    --pin_stderr=\"$APPHOME/simulations/$SCENARIONUM/$clusterID/pin.err\" \
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
elif [ "$SCARABMODE" == "1" ] || [ "$SCARABMODE" == "2" ]; then
  cd $HOME/nonsimpoint_flow/$APPNAME
  mkdir -p simulations evaluations
  APPHOME=$HOME/nonsimpoint_flow/$APPNAME

  if [ "$SCARABMODE" == "2" ]; then
    traceMap=$(ls $APPHOME/traces/trace/)
  fi
  ################################################################
  # trace-based or exec-driven simulations
  taskPids=()
  start=`date +%s`
  mkdir -p $APPHOME/simulations/$SCENARIONUM
  cp $HOME/scarab/src/PARAMS.$SCARABARCH $APPHOME/simulations/$SCENARIONUM/PARAMS.in
  cd $APPHOME/simulations/$SCENARIONUM
  if [ "$SCARABMODE" == "2" ]; then
    scarabCmd="$HOME/scarab/src/scarab --frontend memtrace --cbp_trace_r0=$APPHOME/traces/trace/${traceMap} --memtrace_modules_log=$APPHOME/traces/raw/ $SCARABPARAMS &> sim.log"
  else
    start_inst=100000000
    scarabCmd="
    python3 $HOME/scarab/bin/scarab_launch.py --program=\"$BINCMD\" \
      --simdir=\"$APPHOME/simulations/$SCENARIONUM/\" \
      --pintool_args=\"-hyper_fast_forward_count $start_inst\" \
      --scarab_args=\"--inst_limit $SEGSIZE $SCARABPARAMS\" \
      --scarab_stdout=\"$APPHOME/simulations/$SCENARIONUM/scarab.out\" \
      --scarab_stderr=\"$APPHOME/simulations/$SCENARIONUM/scarab.err\" \
      --pin_stdout=\"$APPHOME/simulations/$SCENARIONUM/pin.out\" \
      --pin_stderr=\"$APPHOME/simulations/$SCENARIONUM/pin.err\" \
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