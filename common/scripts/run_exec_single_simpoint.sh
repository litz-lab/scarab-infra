#!/bin/bash
source utilities.sh

#set -x #echo on

#echo "Running on $(hostname)"

WORKLOAD_HOME="$1"
SCENARIO="$2"
SCARABPARAMS="$3"
SEGSIZE="$4"
SCARABARCH="$5"
WARMUP="$6"
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
PARAMS_FILE="$SCARABHOME/src/PARAMS.$SCARABARCH"
if [[ "$SCARAB_BIN" =~ ^scarab_([0-9a-fA-F]+) ]]; then
  HASH="${BASH_REMATCH[1]}"
  PARAMS_BY_HASH="$SCARABHOME/src/PARAMS.$SCARABARCH.$HASH"
  if [ -f "$PARAMS_BY_HASH" ]; then
    PARAMS_FILE="$PARAMS_BY_HASH"
  fi
fi
cp "$PARAMS_FILE" "$OUTDIR/$segID/PARAMS.in"

PIN_EXEC="$SCARABHOME/src/pin/pin_exec/obj-intel64/pin_exec_${SCARAB_BIN}.so"
cp "$PIN_EXEC" "$OUTDIR/$segID/pin_exec.so"

cd $OUTDIR/$segID

# Run each simpoint in a PRIVATE, node-local run dir so the benchmark never
# writes into the shared app tree, which is bind-mounted read-only from NFS
# (writes there fail, and concurrent simpoints/users would otherwise clobber
# each other's output files). SPEC binary_cmd embeds ABSOLUTE paths for the
# binary, its inputs AND its outputs (e.g. gcc_r's `-o .../foo.s`), so
# redirecting cwd alone is not enough: we make a private copy of the reference
# run dir and rewrite the command's reference-dir prefix to it, so every output
# lands locally. A real copy (not symlinks) is required because the shared tree
# is often polluted with prior-run output files: a symlink to such a file is a
# link to a read-only NFS target, and overwriting it fails. Run dirs are small
# (tens to ~200 MB) and the copy is removed on exit. Overlayfs/bind copy-on-
# write would avoid the copy but needs root; the benchmark runs unprivileged.

# workloads_db stores paths with a literal $tmpdir; expand it (as the eval
# below and the container env would) before we can match/rewrite the prefix.
[ -n "$tmpdir" ] && BINCMD="${BINCMD//\$tmpdir/$tmpdir}"
REFDIR=$(dirname ${BINCMD%% *})
RUNDIR="${SCARAB_RUN_LOCAL_TMP:-/tmp}/scarab_run_${SCARAB_BIN}_$$_${segID}"
rm -rf "$RUNDIR"
mkdir -p "$RUNDIR"
# Real local copy (dereferencing any source symlinks): every file becomes a
# writable local file owned by us, so the benchmark can create/overwrite its
# outputs. Preserve timestamps but not ownership (we may not be the owner).
cp -rL --preserve=timestamps "$REFDIR"/. "$RUNDIR"/
chmod -R u+rwX "$RUNDIR"
# Point the command's binary/inputs/outputs at the private RUNDIR.
BINCMD="${BINCMD//$REFDIR/$RUNDIR}"
trap 'rm -rf "$RUNDIR"' EXIT
cd "$RUNDIR"

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
  --frontend_pin_tool=\"$OUTDIR/$segID/pin_exec.so\" \
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
