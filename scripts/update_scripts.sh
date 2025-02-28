#!/bin/bash
CONTAINERID=$1

user=$(whoami)

docker cp ../common/scripts/utilities.sh $CONTAINERID:/usr/local/bin
docker cp ../common/scripts/run_clustering.sh $CONTAINERID:/usr/local/bin
docker cp ../common/scripts/run_scarab.sh $CONTAINERID:/usr/local/bin
docker cp ../common/scripts/run_scarab_allbench.sh $CONTAINERID:/usr/local/bin
docker cp ../common/scripts/run_scarab_mode_4.sh $CONTAINERID:/usr/local/bin
docker cp ../common/scripts/run_scarab_mode_4_allbench.sh $CONTAINERID:/usr/local/bin
docker cp ../common/scripts/run_simpoint_trace.sh $CONTAINERID:/usr/local/bin
docker cp ../common/scripts/run_trace_post_processing.sh $CONTAINERID:/usr/local/bin
docker cp ../common/scripts/gather_cluster_results.py $CONTAINERID:/usr/local/bin
docker cp ../common/scripts/gather_fp_pieces.py $CONTAINERID:/usr/local/bin
docker cp ../common/scripts/common/common_entrypoint.sh $CONTAINERID:/usr/local/bin
docker cp ../common/scripts/run_exp_using_descriptor.py $CONTAINERID:/usr/local/bin
docker cp ../common/scripts/gather_cluster_results_using_descriptor.py $CONTAINERID:/usr/local/bin
docker cp ../common/scripts/run_single_simpoint.sh $CONTAINERID:/usr/local/bin

# This script doesn't work unless chmod +x is run
docker exec --user root --workdir /home/$user --privileged $CONTAINERID chmod +x /usr/local/bin/run_single_simpoint.sh