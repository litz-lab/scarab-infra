#!/bin/bash
CONTAINERID=$1

user=$(whoami)

docker cp ./utilities.sh $CONTAINERID:/usr/local/bin
docker cp ../common/scripts/run_clustering.sh $CONTAINERID:/usr/local/bin
docker cp ../common/scripts/run_simpoint_trace.py $CONTAINERID:/usr/local/bin
docker cp ../common/scripts/minimize_trace.sh $CONTAINERID:/usr/local/bin
docker cp ../common/scripts/run_trace_post_processing.sh $CONTAINERID:/usr/local/bin
docker cp ../common/scripts/gather_fp_pieces.py $CONTAINERID:/usr/local/bin
docker cp ../common/scripts/root_entrypoint.sh $CONTAINERID:/usr/local/bin
docker cp ../common/scripts/user_entrypoint.sh $CONTAINERID:/usr/local/bin
docker cp ../common/scripts/run_exec_single_simpoint.sh $CONTAINERID:/usr/local/bin
docker cp ../common/scripts/run_memtrace_single_simpoint.sh $CONTAINERID:/usr/local/bin
docker cp ../common/scripts/run_pt_single_simpoint.sh $CONTAINERID:/usr/local/bin