#!/bin/bash
source utilities.sh

set -x #echo on

echo "Running on $(hostname)"

# TODO: for other apps?
APPNAME="$1"
APP_GROUPNAME="$2"
SCENARIO="$3"
SCARABPARAMS="$4"
# this is fixed/settled for NON trace post-processing flow.
# for trace post-processing flow, SEGSIZE is read from file
SEGSIZE=100000000
SCARABARCH="$5"
TRACESSIMP="$6"
SCARABHOME="$7"
SEGMENT_ID="$8"
MODULESDIR="$9"
TRACEFILE="$10"

# 10M warmup for segmented memtraces and 50M warmup for whole memtrace
if [ "$SEGMENT_ID" == "0" ]; then
  WARMUP=50000000
else
  WARMUP=10000000
fi

SIMHOME=$SCENARIO/$APPNAME
mkdir -p $SIMHOME
TRACEHOME=/simpoint_traces/$APPNAME
OUTDIR=$SIMHOME

segID=$SEGMENT_ID
echo "SEGMENT ID: $segID"
mkdir -p $OUTDIR/$segID
cp $SCARABHOME/src/PARAMS.$SCARABARCH $OUTDIR/$segID/PARAMS.in
cd $OUTDIR/$segID

# SEGMENT_ID = 0 represents whole trace simulation
# SEGMENT_ID > 0 represents segmented trace (simpoint) simulation
if [ "$SEGMENT_ID" == "0" ]; then
  traceMap=$(ls $TRACEHOME/traces/whole/)
  scarabCmd="$SCARABHOME/src/scarab --frontend memtrace --cbp_trace_r0=$TRACEHOME/traces/whole/${traceMap} --memtrace_modules_log=$TRACEHOME/traces/raw/ $SCARABPARAMS &> sim.log"
else
  # overwriting
  if [ "$TRACESSIMP" == "1" ]; then
    MODULESDIR=/simpoint_traces/$APPNAME/traces_simp/bin
    TRACEFILE=/simpoint_traces/$APPNAME/traces_simp/trace
  elif [ "$TRACESSIMP" == "2" ] || [ "$TRACESSIMP" == "3" ]; then
    MODULESDIR=/simpoint_traces/$APPNAME/traces_simp/
    TRACEFILE=/simpoint_traces/$APPNAME/traces_simp/
  fi


  segmentSizeFile="$TRACEHOME/fingerprint/segment_size"
  if [ ! -f $segmentSizeFile ]
  then
    echo "$segmentSizeFile does not exist"
    exit
  fi
  SEGSIZE=$(cat "$segmentSizeFile")
  echo "SEGSIZE read from $segmentSizeFile is $SEGSIZE"


  # if TRACESSIMP is 1, 2, or 3,
  # TRACEFILE is supposed to be traces_simp FOLDER
  if [ "$TRACESSIMP" != "0" ]; then
    if [ ! -d $TRACEFILE ]; then
      echo "TRACEFILE is supposed to be a FOLDER"
      exit
    fi
  fi

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

  if [ "$TRACESSIMP" == "0" ]; then
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
    # with TRACESSIMP == 1
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
  elif [ "$TRACESSIMP" == "2" ]; then
    # with TRACESSIMP == 2
    #         # simultion always uses the specific trace file with no skip
    #                 # do not use fetch count
    scarabCmd="$SCARABHOME/src/scarab \
    --frontend memtrace \
    --cbp_trace_r0=$TRACEFILE/$segID/trace/$segID.zip \
    --memtrace_modules_log=$MODULESDIR/$segID/raw \
    --inst_limit=$instLimit \
    --full_warmup=$WARMUP \
    --use_fetched_count=0 \
    $SCARABPARAMS \
    &> sim.log"
  elif [ "$TRACESSIMP" == "3" ]; then
    # with TRACESSIMP == 3
    # simultion always simulate the whole trace file with no skip
    modulesDir=$(dirname $(ls $MODULESDIR/Timestep_$segID/drmemtrace.*.dir/bin/modules.log))
    wholeTrace=$(ls $TRACEFILE/Timestep_$segID/drmemtrace.*.dir/trace/dr*.zip)

    scarabCmd="$SCARABHOME/src/scarab \
    --frontend memtrace \
    --cbp_trace_r0=$wholeTrace \
    --memtrace_modules_log=$modulesDir \
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