#!/bin/bash
#set -x
APPNAME="$1"

sudo mount -t iso9660 -o exec,loop $tmpdir/spec_env.iso $tmpdir/cpu2017

# # Install spec
# mkdir -p $tmpdir/cpu2017_install
# sudo mount -t iso9660 -o ro,exec,loop $tmpdir/cpu2017-1_0_5.iso $tmpdir/cpu2017_install
# cd $tmpdir/cpu2017_install && echo "yes" | ./install.sh -d $tmpdir/cpu2017
# cp $tmpdir/memtrace.cfg $tmpdir/cpu2017/config/memtrace.cfg
# cp $tmpdir/compile-538-clang.sh $tmpdir/cpu2017/benchspec/CPU

# # Build the app
# cd $tmpdir/cpu2017
# source shrc

# ./bin/specperl ./bin/harness/runcpu --copies=1 --iterations=1 --threads=1 --config=memtrace --action=runsetup --size=train $APPNAME