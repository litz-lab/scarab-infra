#!/bin/bash
source utilities.sh

#set -x #echo on

#echo "Running on $(hostname)"

# TODO: for other apps?
WORKLOAD_HOME="$1"
SCENARIO="$2"
SCARABPARAMS="$3"
# this is fixed/settled for NON trace post-processing flow.
# for trace post-processing flow, SEGSIZE is read from file
SEGSIZE="$4"
SCARABARCH="$5"
WARMUP="$6"
TRACE_WARMUP="$7"
TRACE_TYPE="$8"
SCARABHOME="$9"
SEGMENT_ID="${10}"
TRACEFILE="${11}"
SCARAB_BIN="${12}"

SIMHOME=$SCENARIO/$WORKLOAD_HOME
mkdir -p $SIMHOME
OUTDIR=$SIMHOME

segID=$SEGMENT_ID
#echo "SEGMENT ID: $segID"
mkdir -p $OUTDIR/$segID
cp $SCARABHOME/src/PARAMS.$SCARABARCH $OUTDIR/$segID/PARAMS.in
cd $OUTDIR/$segID

# SEGMENT_ID = -1 represents whole trace simulation
# SEGMENT_ID >= 0 represents segmented trace (simpoint) simulation
if [ "$SEGMENT_ID" == "-1" ]; then
  traceMap=$(ls $trace_home/$WORKLOAD_HOME/traces/whole/)
  scarabCmd="$SCARABHOME/src/$SCARAB_BIN \
  --frontend memtrace \
  --cbp_trace_r0=$trace_home/$WORKLOAD_HOME/traces/whole/${traceMap} \
  $SCARABPARAMS &> sim.log"
else
  # overwriting
  TRACEFILE=$trace_home/$WORKLOAD_HOME/traces/simp/$segID.zip
  # roi is initialized by original segment boundary without warmup
  roiStart=$(( $segID * $SEGSIZE + 1 ))
  roiEnd=$(( $segID * $SEGSIZE + $SEGSIZE ))

  # now modify roi start based on warmup:
  # roiStart + WARMUP = original segment start
  if [ "$roiStart" -gt "$WARMUP" ]; then
    # enough room for warmup, extend roi start to the left
    roiStart=$(( $roiStart - $WARMUP ))
  else
    # insufficient preceding instructions, can only warmup till segment start
    WARMUP=$(( $roiStart - 1 ))
    # new roi start is the very first instruction of the trace
    roiStart=1
  fi

  instLimit=$(( $roiEnd - $roiStart + 1 ))

  if [ "$TRACE_TYPE" == "iterative_trace" ]; then
    # with no warmup
    # simultion always simulate the whole trace file with no skip

    numChunk=$(unzip -l "$TRACEFILE" 2>/dev/null | grep "chunk." | wc -l)
    instLimit=$((numChunk * 10000000))

    scarabCmd="$SCARABHOME/src/$SCARAB_BIN \
    --frontend memtrace \
    --cbp_trace_r0=$TRACEFILE \
    --inst_limit=$instLimit \
    --full_warmup=$WARMUP \
    --use_fetched_count=0 \
    $SCARABPARAMS \
    &> sim.log"
  elif [ "$TRACE_TYPE" == "trace_then_cluster" ]; then
    # simultion uses the specific trace file
    # the roiStart is the second chunk, which is assumed to be segment size
    #### if chunk zero chunk is part of the simulation, the roiStart is the first chunk
    # the roiEnd is always the end of the trace -- (dynamorio uses 0)
    # the warmup is the same

    # roiStart 1 means simulation starts with chunk 0
    if [ "$roiStart" == "1" ]; then
      #echo "ROISTART"
      #echo "$TRACEFILE"
      #echo "$segID"
      scarabCmd="$SCARABHOME/src/$SCARAB_BIN \
      --frontend memtrace \
      --cbp_trace_r0=$TRACEFILE \
      --memtrace_roi_begin=1 \
      --memtrace_roi_end=$instLimit \
      --inst_limit=$instLimit \
      --full_warmup=$WARMUP \
      --use_fetched_count=1 \
      $SCARABPARAMS \
      &> sim.log"
    else
      #echo "!ROISTART"
      scarabCmd="$SCARABHOME/src/$SCARAB_BIN \
      --frontend memtrace \
      --cbp_trace_r0=$TRACEFILE \
      --memtrace_roi_begin=$(( $SEGSIZE + 1)) \
      --memtrace_roi_end=$(( $SEGSIZE + $instLimit )) \
      --inst_limit=$instLimit \
      --full_warmup=$WARMUP \
      --use_fetched_count=1 \
      $SCARABPARAMS \
      &> sim.log"
    fi
  elif [ "$TRACE_TYPE" == "cluster_then_trace" ]; then
    if [ "$WARMUP" -lt "$TRACE_WARMUP" ]; then
      scarabCmd="$SCARABHOME/src/$SCARAB_BIN \
      --frontend memtrace \
      --cbp_trace_r0=$TRACEFILE \
      --fast_forward=1 \
      --fast_forward_trace_ins=$(( $TRACE_WARMUP - $WARMUP )) \
      --inst_limit=$instLimit \
      --full_warmup=$WARMUP \
      --use_fetched_count=0 \
      $SCARABPARAMS \
      &> sim.log"
    else
      scarabCmd="$SCARABHOME/src/$SCARAB_BIN \
      --frontend memtrace \
      --cbp_trace_r0=$TRACEFILE \
      --inst_limit=$instLimit \
      --full_warmup=$WARMUP \
      --use_fetched_count=0 \
      $SCARABPARAMS \
      &> sim.log"
    fi
  fi
fi

#echo "simulating clusterID ${clusterID}, segment $segID..."
#echo "command: ${scarabCmd}"
eval $scarabCmd &
wait $!

# Issues. See sim.log in new_experiment20.
# Failed to open $trace_home/postgres/traces_simp/trace/postgres0.zip
# CMD:  docker exec --user aesymons --workdir /home/aesymons --privileged allbench_traces_aesymons slurm_payload.sh "postgres" "allbench_traces" "" "new_experiment20/fe_ftq_block_num.16" "--inst_limit 99900000 --fdip_enable 1 --fe_ftq_block_num 16" "4" "sunny_cove" "1" /home/aesymons/new_experiment20/scarab "3954"