#!/bin/bash
source utilities.sh

#set -x #echo on

#echo "Running on $(hostname)"

WORKLOAD_HOME="$1"
SCENARIO="$2"
SCARABPARAMS="$3"
SCARABARCH="$4"
WARMUP="$5"
SCARABHOME="$6"
SCARAB_BIN="$7"
SEGMENT_ID=0

if [ "$SEGMENT_ID" != "0" ]; then
  echo -e "PT trace simulation does not support simpoints currently. cluster id should always be 0."
  exit
fi

SIMHOME=$SCENARIO/$WORKLOAD_HOME
mkdir -p $SIMHOME
traceMap="trace.gz"

cd $SIMHOME
OUTDIR=$SIMHOME

segID=$SEGMENT_ID
#echo "SEGMENT ID: $segID"
mkdir -p $OUTDIR/$segID
cp $SCARABHOME/src/PARAMS.$SCARABARCH $OUTDIR/$segID/PARAMS.in
cd $OUTDIR/$segID

scarabCmd="$SCARABHOME/src/$SCARAB_BIN --full_warmup $WARMUP --frontend pt --cbp_trace_r0=$trace_home/$WORKLOAD_HOME/traces/pt/${traceMap} $SCARABPARAMS &> sim.log"

#echo "simulating clusterID ${clusterID}, segment $segID..."
#echo "command: ${scarabCmd}"
eval $scarabCmd &
wait $!
