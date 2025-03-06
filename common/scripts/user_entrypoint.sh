#!/bin/bash
#set -x #echo on

export tmpdir="/tmp_home"
export DYNAMORIO_HOME=$tmpdir/DynamoRIO-Linux-10.0.0/
export PIN_ROOT=$tmpdir/pin-3.15-98253-gb56e429b1-gcc-linux
export SCARAB_ENABLE_PT_MEMTRACE=1
export LD_LIBRARY_PATH=$tmpdir/pin-3.15-98253-gb56e429b1-gcc-linux/extras/xed-intel64/lib
export LD_LIBRARY_PATH=$tmpdir/pin-3.15-98253-gb56e429b1-gcc-linux/intel64/runtime/pincrt:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$DYNAMORIO_HOME/lib64/release:$LD_LIBRARY_PATH

export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

if [ -f "/usr/local/bin/workload_user_entrypoint.sh" ]; then
  source /usr/local/bin/workload_user_entrypoint.sh
fi
