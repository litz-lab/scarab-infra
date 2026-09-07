#!/bin/bash
#set -x #echo on

export tmpdir="/tmp_home"
export DYNAMORIO_HOME=$tmpdir/DynamoRIO-Linux-10.0.0/
export PIN_ROOT=$tmpdir/sde-external-9.44.0-2024-08-22-lin/pinkit
export SCARAB_ENABLE_PT_MEMTRACE=1
export SCARAB_ENABLE_PINPLAY=1
# SCARAB_ENABLE_PINPLAY links pin_exec against the SDE runtime, whose init reads
# this and aborts the run without it. The `sde` launcher would set it, but
# scarab_launch.py runs $PIN_ROOT/pin directly. Derived from PIN_ROOT so an SDE
# version bump needs no edit here.
export SDE_GDB_XML="$(dirname "$PIN_ROOT")/misc/gdb-xml"
export LD_LIBRARY_PATH=$tmpdir/sde-external-9.44.0-2024-08-22-lin/pinkit/extras/xed-intel64/lib
export LD_LIBRARY_PATH=$tmpdir/sde-external-9.44.0-2024-08-22-lin/pinkit/intel64/runtime/pincrt:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$DYNAMORIO_HOME/lib64/release:$LD_LIBRARY_PATH

export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

if [ -f "/usr/local/bin/workload_user_entrypoint.sh" ]; then
  source /usr/local/bin/workload_user_entrypoint.sh
fi
