#!/bin/bash

#set -x #echo on
source utilities.sh

OUTDIR=$1
MODULESDIR=$2
TRACEFILE=$3
CHUNKSIZE=$4
SEGSIZE=$5
SIMPOINTHOME=$6

cd $OUTDIR
# RESUMABLE: keep existing fingerprint pieces (a prior pass or retry filled them);
# per-segment loop below skips segments whose piece already exists non-empty.
mkdir -p fingerprint fingerprint/pieces fingerprint/footprint_pieces
cd fingerprint

if [[ "$TRACEFILE" == *.lz4 ]]; then
  # lz4 container has no zip chunk directory; conversion wrote the exact count
  # (drraw2trace -verbose 1 "Final trace instr count") next to the trace.
  numInsts=$(cat "$(dirname $TRACEFILE)/instr_count.txt" 2>/dev/null | head -1)
  if [ -z "$numInsts" ]; then
    # Conversion accepted on the partial-success path (DR worker-shard assert),
    # where raw2trace never prints its aggregate count. Count exactly with the
    # basic_counts analyzer (single decompress pass) and cache the result.
    echo "instr_count.txt missing; counting with basic_counts (one pass)..."
    numInsts=$($DYNAMORIO_HOME/bin64/drrun -t drcachesim -infile $TRACEFILE -simulator_type basic_counts 2>&1                | grep -iE "total \(fetched\)? ?instr|: *total instructions|total instructions"                | grep -oE "[0-9]{4,}" | sort -n | tail -1)
    if [ -z "$numInsts" ]; then echo "ERROR: basic_counts yielded no instruction count for $TRACEFILE"; exit 1; fi
    echo "$numInsts" > "$(dirname $TRACEFILE)/instr_count.txt"
  fi
  numChunk=$(echo "1 + (($numInsts - 1) / $CHUNKSIZE)" | bc)
else
  numChunk=$(unzip -l $TRACEFILE | grep "chunk." | wc -l)
  # rounded-up instr count
  numInsts=$(echo "$numChunk * $CHUNKSIZE" | bc)
fi

echo "total number of trace chunks $numChunk"
echo "total number of instructions ~$numInsts"

# sizeList=("100000000" "50000000" "20000000" "10000000")
# sizeList=("100000000" "10000000")
# for SEGSIZE in "${sizeList[@]}"
# do
#   numSegment=$(echo "1 + (($numInsts - 1) / $SEGSIZE)" | bc)
#   echo "with SEGSIZE $SEGSIZE, number of segments is $numSegment"
#   if [ "$numSegment" -ge 1000 ]; then
#     break
#   elif [ "$SEGSIZE" -eq "${sizeList[-1]}" ]; then
#     echo "with the smallest SEGSIZE $SEGSIZE, number of segments is still not enough for clustering. quit."
#     exit
#   fi
# done

numSegment=$(echo "1 + (($numInsts - 1) / $SEGSIZE)" | bc)
echo "with SEGSIZE $SEGSIZE, number of segments is $numSegment"
if [ $numSegment -lt 1000 ]; then
  echo "WARNING: with SEGSIZE $SEGSIZE, number of segments is less than 1000. Might be to few for clustering."
fi

# if segsize is smaller than the chunksize, the total number of segments
# becomes incorrect. the following steps will fail. so quit.
if [ "$SEGSIZE" -lt "$CHUNKSIZE" ]; then
  echo "SEGSIZE is $SEGSIZE, less than CHUNKSIZE $CHUNKSIZE, which is too small??"

  if [ "$OUTDIR" == "$SIMPOINTHOME/simpoint_10M" ]; then
    echo "This is a hard-coded scenario for verilator, which has chunk size > segment size"
    # verilator last chunk chunk.3945 has 35181523 instructions
    # so the total number of instruction is
    # 3945 * 100M + 35181523 (round up to 4*10M)
    # then the number of segments of 10M is
    numSegment=39454
  else
    exit
  fi
fi

echo "final SEGSIZE is $SEGSIZE, written to $OUTDIR/fingerprint/segment_size"
echo "$SEGSIZE" > $OUTDIR/fingerprint/segment_size

# post-processing
taskPids=()
start=`date +%s`

for segmentID in $(seq 0 $(( $numSegment-1 )))
do
  if [ -s "$OUTDIR/fingerprint/pieces/segment.$segmentID" ]; then continue; fi
  mkdir -p $segmentID
  # do not care about the params file
  cd $segmentID
  scarabCmd="$SIMPOINTHOME/scarab/src/scarab --frontend memtrace \
            --cbp_trace_r0=$TRACEFILE \
            --mode=trace_bbv_distributed \
            --segment_instr_count=$SEGSIZE \
            --memtrace_roi_begin=$(( $segmentID * $SEGSIZE + 1 )) \
            --memtrace_roi_end=$(( $segmentID * $SEGSIZE + $SEGSIZE )) \
            --trace_bbv_output=$OUTDIR/fingerprint/pieces/segment.$segmentID \
            --trace_footprint_output=$OUTDIR/fingerprint/footprint_pieces/segment.$segmentID \
            --use_fetched_count=1 \
            &> sim.log"
  echo "processing segmentID ${segmentID}..."
  echo "command: ${scarabCmd}"
  eval $scarabCmd &
  taskPids+=($!)
  cd -

  # control the number of processing in paralell
  if [ "${#taskPids[@]}" -gt "${FP_JOBS:-40}" ]; then
    echo "post-processing reaches ${FP_JOBS:-40} tasks, wait"
    if wait ${taskPids[0]}; then
      echo "${taskPids[0]} success"
    else
      echo "${taskPids[0]} fail"
    fi
    taskPids=("${taskPids[@]:1}")
  fi

done

wait_for "post-processing" "${taskPids[@]}"
end=`date +%s`
report_time "post-processing" "$start" "$end"

# RETRY missing segments (a few % of bbv sims can fail transiently under 40-way
# parallel zip-seek load; gaps would truncate the fingerprint at the first hole
# and silently exclude the tail from clustering). Two serial-ish retry rounds,
# preserving each failure's sim.log for diagnosis.
for round in 1 2; do
  missing=$(for i in $(seq 0 $(( $numSegment-1 ))); do [ -s "$OUTDIR/fingerprint/pieces/segment.$i" ] || echo $i; done)
  [ -z "$missing" ] && break
  echo "retry round $round: $(echo "$missing" | wc -l) missing segments: $(echo $missing | head -c 200)"
  for segmentID in $missing; do
    mkdir -p $OUTDIR/fingerprint/retry_$segmentID && cd $OUTDIR/fingerprint/retry_$segmentID
    $SIMPOINTHOME/scarab/src/scarab --frontend memtrace --cbp_trace_r0=$TRACEFILE       --mode=trace_bbv_distributed --segment_instr_count=$SEGSIZE       --memtrace_roi_begin=$(( $segmentID * $SEGSIZE + 1 )) --memtrace_roi_end=$(( $segmentID * $SEGSIZE + $SEGSIZE ))       --trace_bbv_output=$OUTDIR/fingerprint/pieces/segment.$segmentID       --trace_footprint_output=$OUTDIR/fingerprint/footprint_pieces/segment.$segmentID       --use_fetched_count=1 &> sim.log
    if [ ! -s "$OUTDIR/fingerprint/pieces/segment.$segmentID" ]; then
      cp sim.log $OUTDIR/fingerprint/failed_segment_$segmentID.round$round.log 2>/dev/null
    fi
    cd - >/dev/null
  done
done
still=$(for i in $(seq 0 $(( $numSegment-1 ))); do [ -s "$OUTDIR/fingerprint/pieces/segment.$i" ] || echo $i; done | wc -l)
if [ "$still" -gt 0 ]; then
  echo "ERROR: $still segments still missing after retries; failed logs in fingerprint/failed_segment_*.log"
  exit 1
fi

# aggregate the fingerprint pieces; bbfp is appended to, so start from scratch, and
# stop on failure: a partial bbfp would cluster only a prefix of the trace
rm -f $OUTDIR/fingerprint/pieces/bbfp
if ! python3 /usr/local/bin/gather_fp_pieces.py $OUTDIR/fingerprint/pieces $numSegment segment; then
  echo "ERROR: gather_fp_pieces failed; not clustering a partial fingerprint"
  exit 1
fi
cp $OUTDIR/fingerprint/pieces/bbfp $OUTDIR/fingerprint/bbfp
exit