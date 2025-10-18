#!/bin/bash
source utilities.sh

#set -x #echo on

#echo "Running on $(hostname)"

WORKLOAD_HOME="$1"
SCENARIO="$2"
SCARABPARAMS="$3"
SEGSIZE="$4"
SCARABARCH="$5"
TRACESSIMP="$6"
SCARABHOME="$7"
SEGMENT_ID="$8"
ENVVAR="$9"
BINCMD="${10}"
CLIENT_BINCMD="${11}"
SCARAB_BIN="${12}"

for token in $ENVVAR;
do
  export $token
done

# 10M warmup for segmented simulation (simpoints) and 50M warmup for whole simulation
if [ "$SEGMENT_ID" == "0" ]; then
  WARMUP=50000000
else
  WARMUP=10000000
fi

SIMHOME=$SCENARIO/$WORKLOAD_HOME
mkdir -p $SIMHOME
OUTDIR=$SIMHOME

segID=$SEGMENT_ID
#echo "SEGMENT ID: $segID"
mkdir -p $OUTDIR/$segID
cp $SCARABHOME/src/PARAMS.$SCARABARCH $OUTDIR/$segID/PARAMS.in
cd $OUTDIR/$segID

# SEGMENT_ID = 0 represents non-simpoint trace simulation
# SEGMENT_ID > 0 represents segmented (simpoint) simulation
if [ "$SEGMENT_ID" == "0" ]; then
  start_inst=100000000
  scarabCmd="
  python3 $SCARABHOME/bin/scarab_launch.py --program=\"$BINCMD\" \
    --scarab=src/$SCARAB_BIN \
    --simdir=\"$SIMHOME/$SCENARIONUM/\" \
    --pintool_args=\"-hyper_fast_forward_count $start_inst\" \
    --scarab_args=\"--inst_limit $SEGSIZE --full_warmup $WARMUP $SCARABPARAMS\" \
    --scarab_stdout=\"$SIMHOME/$SCENARIONUM/scarab.out\" \
    --scarab_stderr=\"$SIMHOME/$SCENARIONUM/scarab.err\" \
    --pin_stdout=\"$SIMHOME/$SCENARIONUM/pin.out\" \
    --pin_stderr=\"$SIMHOME/$SCENARIONUM/pin.err\" \
    "
else
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

  scarabCmd="
  python3 $SCARABHOME/bin/scarab_launch.py --program=\"$BINCMD\" \
  --scarab=src/$SCARAB_BIN \
  --simdir=\"$SIMHOME/$SCENARIONUM/$clusterID\" \
  --pintool_args=\"-hyper_fast_forward_count $roiStart\" \
  --scarab_args=\"--inst_limit $instLimit --full_warmup $WARMUP $SCARABPARAMS\" \
  --scarab_stdout=\"$SIMHOME/$SCENARIONUM/$clusterID/scarab.out\" \
  --scarab_stderr=\"$SIMHOME/$SCENARIONUM/$clusterID/scarab.err\" \
  --pin_stdout=\"$SIMHOME/$SCENARIONUM/$clusterID/pin.out\" \
  --pin_stderr=\"$SIMHOME/$SCENARIONUM/$clusterID/pin.err\" \
  "
fi

#echo "simulating clusterID ${clusterID}, segment $segID..."
#echo "command: ${scarabCmd}"
eval $scarabCmd &
wait $!
