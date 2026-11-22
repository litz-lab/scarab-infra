#!/bin/bash
source utilities.sh

# 1. Map the 20 arguments passed from utilities.py
WORKLOAD_HOME="$1"
SCENARIO="$2"
SCARABPARAMS="$3"
SEGSIZE="$4"
SCARABARCH="$5"
WARMUP="$6"
SCARABHOME="$7"
SEGMENT_ID="$8"
SCARAB_BIN="$9"
SYNTH_TARGET_POOL_SIZE="${10}"
SYNTH_TARGET_SAMPLING_STRATEGY="${11}"
SYNTH_T_NT_RATIO="${12}"
SYNTH_MEM_HIERACHY_LEVEL="${13}"
SYNTH_DEPENDENCE_CHAIN_COUNT="${14}"
SYNTH_INST_SIZE="${15}"
SYNTH_DIRECTION_SAMPLING_STRATEGY="${16}"
SYNTH_WORKLOAD_LENGTH="${17}"
SYNTH_NOP_SIZE="${18}"
SYNTH_STARTING_TARGET="${19}"
SYNTH_TARGET_STRIDE="${20}"

# 2. Locate the correct PARAMS file
PARAMS_FILE="$SCARABHOME/src/PARAMS.$SCARABARCH"
if [[ "$SCARAB_BIN" =~ ^scarab_([0-9a-fA-F]+) ]]; then
  HASH="${BASH_REMATCH[1]}"
  PARAMS_BY_HASH="$SCARABHOME/src/PARAMS.$SCARABARCH.$HASH"
  if [ -f "$PARAMS_BY_HASH" ]; then
    PARAMS_FILE="$PARAMS_BY_HASH"
  fi
fi

# 3. Set up the isolated output directory for this run
SIMHOME=$SCENARIO/$WORKLOAD_HOME
mkdir -p "$SIMHOME"
OUTDIR=$SIMHOME

segID=$SEGMENT_ID
mkdir -p "$OUTDIR/$segID"
cp "$PARAMS_FILE" "$OUTDIR/$segID/PARAMS.in"
cd "$OUTDIR/$segID" || exit 1

instLimit=$(( SEGSIZE + WARMUP ))

# 4. Conditionally append synthetic flags only when set
EXTRA_FLAGS=""
[ -n "$SYNTH_TARGET_POOL_SIZE" ] && [ "$SYNTH_TARGET_POOL_SIZE" != "None" ] && EXTRA_FLAGS="$EXTRA_FLAGS --target_pool_size $SYNTH_TARGET_POOL_SIZE"
[ -n "$SYNTH_TARGET_SAMPLING_STRATEGY" ] && [ "$SYNTH_TARGET_SAMPLING_STRATEGY" != "None" ] && EXTRA_FLAGS="$EXTRA_FLAGS --target_sampling_strategy $SYNTH_TARGET_SAMPLING_STRATEGY"
[ -n "$SYNTH_T_NT_RATIO" ] && [ "$SYNTH_T_NT_RATIO" != "None" ] && EXTRA_FLAGS="$EXTRA_FLAGS --t_nt_ratio $SYNTH_T_NT_RATIO"
[ -n "$SYNTH_MEM_HIERACHY_LEVEL" ] && [ "$SYNTH_MEM_HIERACHY_LEVEL" != "None" ] && EXTRA_FLAGS="$EXTRA_FLAGS --mem_hierachy_level $SYNTH_MEM_HIERACHY_LEVEL"
[ -n "$SYNTH_DEPENDENCE_CHAIN_COUNT" ] && [ "$SYNTH_DEPENDENCE_CHAIN_COUNT" != "None" ] && EXTRA_FLAGS="$EXTRA_FLAGS --dependence_chain_count $SYNTH_DEPENDENCE_CHAIN_COUNT"
[ -n "$SYNTH_INST_SIZE" ] && [ "$SYNTH_INST_SIZE" != "None" ] && EXTRA_FLAGS="$EXTRA_FLAGS --inst_size $SYNTH_INST_SIZE"
[ -n "$SYNTH_DIRECTION_SAMPLING_STRATEGY" ] && [ "$SYNTH_DIRECTION_SAMPLING_STRATEGY" != "None" ] && EXTRA_FLAGS="$EXTRA_FLAGS --direction_sampling_strategy $SYNTH_DIRECTION_SAMPLING_STRATEGY"
[ -n "$SYNTH_WORKLOAD_LENGTH" ] && [ "$SYNTH_WORKLOAD_LENGTH" != "None" ] && EXTRA_FLAGS="$EXTRA_FLAGS --workload_length $SYNTH_WORKLOAD_LENGTH"
[ -n "$SYNTH_NOP_SIZE" ] && [ "$SYNTH_NOP_SIZE" != "None" ] && EXTRA_FLAGS="$EXTRA_FLAGS --nop_size $SYNTH_NOP_SIZE"
[ -n "$SYNTH_STARTING_TARGET" ] && [ "$SYNTH_STARTING_TARGET" != "None" ] && EXTRA_FLAGS="$EXTRA_FLAGS --starting_target $SYNTH_STARTING_TARGET"
[ -n "$SYNTH_TARGET_STRIDE" ] && [ "$SYNTH_TARGET_STRIDE" != "None" ] && EXTRA_FLAGS="$EXTRA_FLAGS --target_stride $SYNTH_TARGET_STRIDE"

# 5. Build and execute command
scarabCmd="$SCARABHOME/src/$SCARAB_BIN \
  --frontend synthetic \
  --full_warmup=$WARMUP \
  --inst_limit $instLimit \
  $EXTRA_FLAGS \
  $SCARABPARAMS &> sim.log"

PERF_PREFIX=$(perf_stat_prefix "$OUTDIR/$segID/perf_stat.txt")
eval $PERF_PREFIX $scarabCmd &
wait $!