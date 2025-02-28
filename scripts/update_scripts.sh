#!/bin/bash
CONTAINERID=$1

user=$(whoami)

docker cp ./utilities.sh $CONTAINERID:/usr/local/bin
docker cp ../common/scripts/run_clustering.sh $CONTAINERID:/usr/local/bin
docker cp ../common/scripts/run_simpoint_trace.sh $CONTAINERID:/usr/local/bin
docker cp ../common/scripts/run_trace_post_processing.sh $CONTAINERID:/usr/local/bin
docker cp ../common/scripts/common_entrypoint.sh $CONTAINERID:/usr/local/bin
docker cp ../common/scripts/run_exec_single_simpoint.sh $CONTAINERID:/usr/local/bin
docker cp ../common/scripts/run_memtrace_single_simpoint.sh $CONTAINERID:/usr/local/bin
docker cp ../common/scripts/run_pt_single_simpoint.sh $CONTAINERID:/usr/local/bin