#!/bin/bash

set -x #echo on

echo "Running on $(hostname)"

APPNAME="$1"
APP_GROUPNAME="$2"
SIMPOINTHOME="$3"
BINCMD="$4"
SEGSIZE=10000000
# chunk size within trace file. Use 10M due to conversion issue.
CHUNKSIZE=10000000
SIMPOINT="$5"
DRIO_ARGS="$6"
# if specified, maxk will use the user provided value;
# if not specified, maxk will be calculated as the square root of the number of segments
CLUSTERING_USERK=${7:-0}
# used by simpoint flow 1 to manually re-trace a simpoint (when the previous one does not have enough instrs)
SIMPOINT_1_MANUAL_TRACE=${8:-NA}

source utilities.sh

# Get command to run for Spec17
if [ "$APP_GROUPNAME" == "spec2017" ] && [ "$APPNAME" != "clang" ] && [ "$APPNAME" != "gcc" ]; then
  # environment
  cd $HOME/cpu2017
  source ./shrc
  # compile and get command for application
  # TODO: this is just for one input
  ./bin/specperl ./bin/harness/runcpu --copies=1 --iterations=1 --threads=1 --config=memtrace --action=runsetup --size=ref $APPNAME
  ogo $APPNAME run
  cd run_base_ref*
  BINCMD=$(specinvoke -nn | tail -2 | head -1)
fi

if [ "$SIMPOINT" == "2" ]; then
  # dir for all relevant data: fingerprint, traces, log, sim stats...
  mkdir -p $SIMPOINTHOME/$APPNAME
  cd $SIMPOINTHOME/$APPNAME
  mkdir -p traces
  APPHOME=$SIMPOINTHOME/$APPNAME


  ################################################################

  # 1. trace the whole application
  # 2. drraw2trace
  # 3. post-process the trace in parallel
  # 4. aggregate the fingerprint
  # 5. clustering

  taskPids=()
  start=`date +%s`

  mkdir -p $APPHOME/traces/whole

  # spec needs to run in its run dir
  if [ "$APP_GROUPNAME" == "spec2017" ] && [ "$APPNAME" != "clang" ] && [ "$APPNAME" != "gcc" ]; then
    ogo $APPNAME run
    cd run_base_ref*
  else
    cd $APPHOME/traces/whole
  fi

  echo ${DRIO_ARGS}
  echo $BINCMD
  if [ "$DRIO_ARGS" == "None" ]; then
    traceCmd="$DYNAMORIO_HOME/bin64/drrun -t drcachesim -jobs 40 -outdir $APPHOME/traces/whole -offline"
  else
    traceCmd="$DYNAMORIO_HOME/bin64/drrun -t drcachesim -jobs 40 -outdir $APPHOME/traces/whole -offline $DRIO_ARGS"
  fi

  if [ "$APPNAME" == "mysql" ] || [ "$APPNAME" == "postgres" ]; then
    sudo chown -R $APPNAME:$APPNAME $APPHOME/traces/whole
    traceCmd="sudo -u $APPNAME $traceCmd -- ${BINCMD}"
  elif [ "$APPNAME" == "long_multi_update" ]; then
    traceCmd="sudo -u $APPNAME $traceCmd -- ${BINCMD}"
  else
    traceCmd="$traceCmd -- ${BINCMD}"
  fi
  echo "tracing whole app..."
  echo "command: ${traceCmd}"
  # if [ "$APP_GROUPNAME" == "spec2017" ]; then
  #   # have to go to that dir for the spec app cmd to work
  #   ogo $APPNAME run
  #   cd run_base_train*
  # fi
  eval $traceCmd &
  taskPids+=($!)

  wait_for "whole app tracing" "${taskPids[@]}"
  end=`date +%s`
  report_time "whole app tracing" "$start" "$end"

  taskPids=()
  start=`date +%s`

  cd $APPHOME/traces/whole
  for dr in dr*;
  do
    cd $dr
    mkdir -p bin
    cp raw/modules.log bin/modules.log
    cp raw/modules.log raw/modules.log.bak
    python2 $SIMPOINTHOME/scarab/utils/memtrace/portabilize_trace.py .
    cp bin/modules.log raw/modules.log
    $DYNAMORIO_HOME/tools/bin64/drraw2trace -jobs 40 -indir ./raw/ -chunk_instr_count $CHUNKSIZE &
    taskPids+=($!)
    cd -
  done

  wait_for "whole app raw2trace" "${taskPids[@]}"
  end=`date +%s`
  report_time "whole app raw2trace" "$start" "$end"

  # continue if only one trace file
  numTrace=$(find -name "dr*.trace.zip" | grep "drmemtrace.*.trace.zip" | wc -l)
  numDrFolder=$(find -type d -name "drmemtrace.*.dir" | grep "drmemtrace.*.dir" | wc -l)
  if [ "$numTrace" == "1" ] && [ "$numDrFolder" == "1" ]; then
    ###HEERREEE prepare raw dir, trace dir
    modulesDir=$(dirname $(ls $APPHOME/traces/whole/drmemtrace.*.dir/raw/modules.log))
    wholeTrace=$(ls $APPHOME/traces/whole/drmemtrace.*.dir/trace/dr*.zip)
    echo "modulesDIR: $modulesDir"
    echo "wholeTrace: $wholeTrace"
    bash run_trace_post_processing.sh $APPHOME $modulesDir $wholeTrace $CHUNKSIZE $SEGSIZE
  else
  # otherwise ask the user to run manually
    echo -e "There are multiple trace files.\n\
    Decide and run \"/usr/local/bin/run_trace_post_processing.sh <OUTDIR> <MODULESDIR> <TRACEFILE> <CHUNKSIZE> <SEGSIZE>\"\n\
    Then run /usr/local/bin/run_clustering.sh <FPFILE> <OUTDIR>"
    exit
  fi

  # clustering
  bash run_clustering.sh $APPHOME/fingerprint/bbfp $APPHOME $CLUSTERING_USERK

elif [ "$SIMPOINT" == "1" ]; then
  # dir for all relevant data: fingerprint, traces, log, sim stats...
  mkdir -p $SIMPOINTHOME/$APPNAME
  cd $SIMPOINTHOME/$APPNAME
  mkdir -p fingerprint traces_simp
  APPHOME=$SIMPOINTHOME/$APPNAME


  ################################################################

  # spec needs to run in its run dir
  if [ "$APP_GROUPNAME" == "spec2017" ] && [ "$APPNAME" != "clang" ] && [ "$APPNAME" != "gcc" ]; then
    ogo $APPNAME run
    cd run_base_ref*
  else
    cd $APPHOME/fingerprint
  fi

  # collect fingerprint
  # TODO: add parameter: size and warm-up
  fpCmd="$DYNAMORIO_HOME/bin64/drrun -max_bb_instrs 4096 -opt_cleancall 2 -c $tmpdir/libfpg.so -no_use_bb_pc -no_use_fetched_count -segment_size $SEGSIZE -output $APPHOME/fingerprint/bbfp -pcmap_output $APPHOME/fingerprint/pcmap -- $BINCMD"
  echo "generate fingerprint..."
  echo "command: ${fpCmd}"
  # if [ "$APP_GROUPNAME" == "spec2017" ]; then
  #   # have to go to that dir for the spec app cmd to work
  #   ogo $APPNAME run
  #   cd run_base_train*
  # fi

  taskPids=()
  start=`date +%s`

  if [ "$SIMPOINT_1_MANUAL_TRACE" == "NA" ]; then
    eval $fpCmd &
    taskPids+=($!)
  fi

  wait_for "online fingerprint" "${taskPids[@]}"
  end=`date +%s`
  report_time "online fingerprint" "$start" "$end"

  if [ "$SIMPOINT_1_MANUAL_TRACE" == "NA" ]; then
    echo "final SEGSIZE is $SEGSIZE, written to $APPHOME/fingerprint/segment_size"
    echo "$SEGSIZE" > $APPHOME/fingerprint/segment_size
  fi

  # continue if only one bbfp file
  cd $APPHOME/fingerprint
  numBBFP=$(find -name "bbfp.*" | grep "bbfp.*" | wc -l)
  if [ "$numBBFP" == "1" ]; then
    bbfpFile=$(ls $APPHOME/fingerprint/bbfp.*)
    echo "bbfpFile: $bbfpFile"
    # run SimPoint clustering
    if [ "$SIMPOINT_1_MANUAL_TRACE" == "NA" ]; then
      bash run_clustering.sh $bbfpFile $APPHOME $CLUSTERING_USERK
    fi
  else
  # otherwise ask the user to run manually
    echo -e "There are multiple or no bbfp files. This simpoint flow would not work."
    exit
  fi

  ################################################################


  ################################################################
  # read in simpoint
  # ref: https://stackoverflow.com/q/56005842
  # map of cluster - segment
  declare -A clusterMap
  while IFS=" " read -r segID clusterID; do
    if [ "$SIMPOINT_1_MANUAL_TRACE" == "NA" ]; then
      clusterMap[$clusterID]=$segID
    elif [ "$segID" == "$SIMPOINT_1_MANUAL_TRACE" ]; then
      clusterMap[$clusterID]=$segID
    fi
  done < $APPHOME/simpoints/opt.p.lpt0.99

  ################################################################
  # collect traces

  # tracing, raw2trace
  taskPids=()
  start=`date +%s`
  for clusterID in "${!clusterMap[@]}"
  do
    segID=${clusterMap[$clusterID]}
    mkdir -p $APPHOME/traces_simp/$segID
    # spec needs to run in its run dir
    if [ "$APP_GROUPNAME" == "spec2017" ] && [ "$APPNAME" != "clang" ] && [ "$APPNAME" != "gcc" ]; then
      ogo $APPNAME run
      cd run_base_ref*
    else
      cd $APPHOME/traces_simp/$segID
    fi

    # the simulation region, in the unit of chunks
    roiStart=$(( $segID * $SEGSIZE ))
    # seq is inclusive
    roiEnd=$(( $segID * $SEGSIZE + $SEGSIZE ))

    # assume warm-up length is the segsize
    # this will limit the amount of warmup that can be done during the simulation
    WARMUP=$SEGSIZE
    if [ "$roiStart" -gt "$WARMUP" ]; then
        # enough room for warmup, extend roi start to the left
        roiStart=$(( $roiStart - $WARMUP ))
    else
        # no enough preceding instructions, can only warmup till segment start
        # new roi start is the very first instruction of the trace
        roiStart=0
    fi

    roiLength=$(( $roiEnd - $roiStart ))

    if [ "$SIMPOINT_1_MANUAL_TRACE" == "NA" ]; then
      # because we are using exit_after_tracing,
      # want to to pad saome extra so we can likely have enough instrs
      roiLength=$(( $roiLength + 2 * $SEGSIZE ))
    else
      # pad even more
      roiLength=$(( $roiLength + 8 * $SEGSIZE ))
    fi

    # which dynamorio to use?
    if [ $roiStart -eq 0 ]; then
      # do not specify trace_after_instrs
      traceCmd="$DYNAMORIO_HOME/bin64/drrun -t drcachesim -jobs 40 -outdir $APPHOME/traces_simp/$segID -offline -exit_after_tracing $roiLength -- ${BINCMD}"
    else
      traceCmd="$DYNAMORIO_HOME/bin64/drrun -t drcachesim -jobs 40 -outdir $APPHOME/traces_simp/$segID -offline -trace_after_instrs $roiStart -exit_after_tracing $roiLength -- ${BINCMD}"
    fi

    echo "tracing cluster ${clusterID}, segment ${segID}..."
    echo "command: ${traceCmd}"

    eval $traceCmd &
    taskPids+=($!)

    # cannot invoke multiple spec apps at once
    if [ "$APP_GROUPNAME" == "spec2017" ]; then
      wait_for "tracing cluster $clusterID" ${taskPids[-1]}
    else
      sleep 2
    fi
  done

  wait_for "cluster tracings" "${taskPids[@]}"
  end=`date +%s`
  report_time "cluster tracings" "$start" "$end"

  taskPids=()
  start=`date +%s`

  # TODO: which dynamorio to use, release or scarab submodule?
  # this assumes that the app is single-thread
  # this flow would only work for single thread anyway
  for clusterID in "${!clusterMap[@]}"
  do
    segID=${clusterMap[$clusterID]}
    cd $APPHOME/traces_simp/$segID
    numDrFolder=$(find -type d -name "drmemtrace.*.dir" | grep "drmemtrace.*.dir" | wc -l)
    if [ "$numDrFolder" != "1" ]; then
      echo "the number of dr folder $numDrFolder is not one, location: $APPHOME/traces_simp/$segID"
      exit
    fi

    mv dr*/raw/ ./raw
    mkdir -p bin
    cp raw/modules.log bin/modules.log
    cp raw/modules.log raw/modules.log.bak
    python2 $SIMPOINTHOME/scarab/utils/memtrace/portabilize_trace.py .
    cp bin/modules.log raw/modules.log
    $DYNAMORIO_HOME/tools/bin64/drraw2trace -jobs 40 -indir ./raw/ -chunk_instr_count $CHUNKSIZE &
    taskPids+=($!)
    sleep 2
  done

  wait_for "cluster raw2trace" "${taskPids[@]}"
  end=`date +%s`
  report_time "cluster raw2trace" "$start" "$end"

  ################################################################
  # minimize traces, rename traces
  # it is possible that SimPoint picks interval zero,
  # in that case the simulation would only need one chunk,
  # but we always keep two regardlessly
  for clusterID in "${!clusterMap[@]}"
  do
    segID=${clusterMap[$clusterID]}
    cd $APPHOME/traces_simp/$segID
    numTrace=$(find -name "dr*.trace.zip" | grep "drmemtrace.*.trace.zip" | wc -l)
    if [ "$numTrace" != "1" ]; then
      echo "the number of trace file $numTrace is not one, location: $APPHOME/traces_simp/$segID"
      exit
    fi
    mv ./trace/dr*.trace.zip ./trace/$segID.big.zip

    numChunk=$(unzip -l ./trace/$segID.big.zip | grep "chunk." | wc -l)
    if [ "$numChunk" -lt 2 ]; then
      echo "WARN: the big trace $segID contains less than 2 chunks: $numChunk !"
    fi

    # copy chunk 0 and chunk 1
    zip ./trace/$segID.big.zip --copy chunk.0000 chunk.0001 --out ./trace/$segID.zip
    rm ./trace/$segID.big.zip
  done

  ################################################################
fi
###############################################################
