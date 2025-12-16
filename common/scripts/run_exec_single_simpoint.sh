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

SIMHOME=$SCENARIO/$WORKLOAD_HOME
mkdir -p $SIMHOME
OUTDIR=$SIMHOME

segID=$SEGMENT_ID
#echo "SEGMENT ID: $segID"
mkdir -p $OUTDIR/$segID
cp $SCARABHOME/src/PARAMS.$SCARABARCH $OUTDIR/$segID/PARAMS.in
cp $SCARABHOME/src/pin/pin_exec/obj-intel64/pin_exec.so $OUTDIR/$segID/pin_exec.so
cd $OUTDIR/$segID

# Split SCARABPARAMS into scarab vs pintool vs launch flag
enable_aslr=0
pintool_args=()
scarab_args=()

if [ -n "$SCARABPARAMS" ]; then
  read -r -a tokens <<< "$SCARABPARAMS"
  idx=0
  while [ $idx -lt ${#tokens[@]} ]; do
    token="${tokens[$idx]}"
    if [[ "$token" == --enable_aslr || "$token" == --enable_aslr=* ]]; then
      enable_aslr=1
      idx=$((idx + 1))
      continue
    fi

    if [[ "$token" == --* ]]; then
      scarab_args+=("$token")
      idx=$((idx + 1))
      continue
    fi

    if [[ "$token" == -* ]]; then
      if [[ "$token" == *=* ]]; then
        key="${token%%=*}"
        val="${token#*=}"
        pintool_args+=("$key" "$val")
        idx=$((idx + 1))
      else
        if [ $((idx + 1)) -lt ${#tokens[@]} ]; then
          next="${tokens[$((idx + 1))]}"
          if [[ "$next" != --* && "$next" != -* ]]; then
            pintool_args+=("$token" "$next")
            idx=$((idx + 2))
            continue
          fi
        fi
        pintool_args+=("$token")
        idx=$((idx + 1))
      fi
      continue
    fi

    scarab_args+=("$token")
    idx=$((idx + 1))
  done
fi

scarab_args_str=""
if [ ${#scarab_args[@]} -gt 0 ]; then
  scarab_args_str="$scarab_args_str ${scarab_args[*]}"
fi

pintool_args_str="${pintool_args[*]}"

scarabCmd="python3 $SCARABHOME/bin/scarab_launch.py --program=\"$BINCMD\" \
  --scarab=\"$SCARABHOME/src/$SCARAB_BIN\" \
  --simdir=\"$SIMHOME/$SCENARIONUM\$segID\" \
  --pintool_args=\"$pintool_args_str\" \
  --scarab_args=\"$scarab_args_str\" \
  --scarab_stdout=\"$SIMHOME/$SCENARIONUM\$segID\scarab.out\" \
  --scarab_stderr=\"$SIMHOME/$SCENARIONUM\$segID\scarab.err\" \
  --pin_stdout=\"$SIMHOME/$SCENARIONUM\$segID\pin.out\" \
  --pin_stderr=\"$SIMHOME/$SCENARIONUM\$segID\pin.err\" \
  "

if [ "$enable_aslr" -eq 1 ]; then
  scarabCmd="$scarabCmd --enable_aslr"
fi

printf '%q ' "${scarabCmd[@]}" > $SIMHOME/launch_cmd.txt
eval $scarabCmd &
wait $!
