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

WARMUP=50000000

SIMHOME=$SCENARIO/$WORKLOAD_HOME
mkdir -p $SIMHOME
OUTDIR=$SIMHOME

segID=$SEGMENT_ID
#echo "SEGMENT ID: $segID"
mkdir -p $OUTDIR/$segID
cp $SCARABHOME/src/PARAMS.$SCARABARCH $OUTDIR/$segID/PARAMS.in
cp $SCARABHOME/src/pin/pin_exec/obj-intel64/pin_exec.so $OUTDIR/$segID/pin_exec.so
cd $OUTDIR/$segID

BINARY_DIR=$(dirname ${BINCMD%% *})
cd $BINARY_DIR

roiStart=$(( $segID * $SEGSIZE + 1 ))
roiEnd=$(( $segID * $SEGSIZE + $SEGSIZE ))

if [ "$roiStart" -gt "$WARMUP" ]; then
  roiStart=$(( $roiStart - $WARMUP ))
else
  WARMUP=$(( $roiStart - 1 ))
  roiStart=1
fi

instLimit=$(( $roiEnd - $roiStart + 1 ))

scarabCmd="python3 $SCARABHOME/bin/scarab_launch.py --program=\"$BINCMD\" \
  --scarab=\"$SCARABHOME/src/$SCARAB_BIN\" \
  --simdir=\"$SIMHOME/$SCENARIONUM/$segID\" \
  --pintool_args=\"-hyper_fast_forward_count $roiStart\" \
  --scarab_args=\"--use_fetched_count 1 --uop_cache_insert_only_onpath 1 --inst_limit $instLimit --full_warmup $WARMUP $SCARABPARAMS\" \
  --scarab_stdout=\"$SIMHOME/$SCENARIONUM/$segID/scarab.out\" \
  --scarab_stderr=\"$SIMHOME/$SCENARIONUM/$segID/scarab.err\" \
  --pin_stdout=\"$SIMHOME/$SCENARIONUM/$segID/pin.out\" \
  --pin_stderr=\"$SIMHOME/$SCENARIONUM/$segID/pin.err\" \
  "

printf '%q ' "${scarabCmd[@]}" > $SIMHOME/launch_cmd.txt

eval $scarabCmd &
wait $!
