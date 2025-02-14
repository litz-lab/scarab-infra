#!/bin/bash
source utilities.sh

set -x #echo on

echo "Running on $(hostname)"

APPNAME="$1"
APP_GROUPNAME="$2"
SCENARIO="$3"
SCARABPARAMS="$4"
SCARABARCH="$5"
TRACESSIMP="$6"
SCARABHOME="$7"
SEGMENT_ID="$8"

if [ "$SEGMENT_ID" != "0" ]; then
  echo -e "PT trace simulation does not support simpoints currently. cluster id should always be 0."
  exit
fi

# 50M warmup for PT traces by default
WARMUP=49999999

SIMHOME=$SCENARIO/$APPNAME
mkdir -p $SIMHOME
TRACEHOME=/simpoint_traces/$APPNAME
traceMap="trace.gz"

cd $SIMHOME
OUTDIR=$SIMHOME

segID=$SEGMENT_ID
echo "SEGMENT ID: $segID"
mkdir -p $OUTDIR/$segID
cp $SCARABHOME/src/PARAMS.$SCARABARCH $OUTDIR/$segID/PARAMS.in
cd $OUTDIR/$segID

scarabCmd="$SCARABHOME/src/scarab --full_warmup $WARMUP --frontend pt --cbp_trace_r0=$TRACEHOME/${traceMap} $SCARABPARAMS &> sim.log"

echo "simulating clusterID ${clusterID}, segment $segID..."
echo "command: ${scarabCmd}"
eval $scarabCmd &
wait $!
