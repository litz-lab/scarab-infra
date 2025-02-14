#!/bin/bash

source utilities.sh

SCARABHOME=$1
MODULESDIR=$2
TRACEFILE=$3
SCARABPARAMS=$4
SPDIR=$5
SEGSIZE=$6
OUTDIR=$7
WARMUPORG=$8
SCARABARCH=$9

# if TRACESSIMP is 1,
# TRACEFILE is supposed to be traces_simp FOLDER
TRACESSIMP=$9

if [ "$TRACESSIMP" == "1" ]; then
    if [ ! -d $TRACEFILE ]; then
        echo "TRACEFILE is supposed to be traces_simp FOLDER"
        exit
    fi
fi

cd $OUTDIR

################################################################
# read in simpoint
# ref: https://stackoverflow.com/q/56005842
# map of cluster - segment
declare -A clusterMap
while IFS=" " read -r segID clusterID; do
clusterMap[$clusterID]=$segID 
done < $SPDIR/opt.p.lpt0.99

################################################################
# trace-based simulations
taskPids=()
start=`date +%s`
# simulation in parallel
# actually array would suffice
for clusterID in "${!clusterMap[@]}"
do
    WARMUP=$WARMUPORG
    segID=${clusterMap[$clusterID]}
    mkdir -p $OUTDIR/$segID
    cp $SCARABHOME/src/PARAMS.$SCARABARCH $OUTDIR/$segID/PARAMS.in
    cd $OUTDIR/$segID

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

    if [ "$TRACESSIMP" != "1" ]; then
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
        # with TRACESSIMP
        # simultion uses the specific trace file
        # the roiStart is the second chunk, which is assumed to be segment size
        #### if chunk zero chunk is part of the simulation, the roiStart is the first chunk
        # the roiEnd is always the end of the trace -- (dynamorio uses 0)
        # the warmup is the same

        # roiStart 1 means simulation starts with chunk 0
        if [ "$roiStart" == "1" ]; then
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

    fi

    echo "simulating clusterID ${clusterID}, segment $segID..."
    echo "command: ${scarabCmd}"
    eval $scarabCmd &
    taskPids+=($!)
    cd -
done

wait_for "simpoint simiulations" "${taskPids[@]}"
end=`date +%s`
report_time "simpoint simiulations" "$start" "$end"

# aggregate the simulation results
cd $OUTDIR
python3 gather_cluster_results.py $SPDIR $OUTDIR
