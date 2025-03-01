#!/bin/bash
#set -x
APPNAME="$1"

mkdir -p $tmpdir/cpu2017_install
sudo mount -t iso9660 -o ro,exec,loop $tmpdir/cpu2017-1_0_5.iso $tmpdir/cpu2017_install
cd $tmpdir/cpu2017_install && echo "yes" | ./install.sh -d $tmpdir/cpu2017
cp $tmpdir/memtrace.cfg $tmpdir/cpu2017/config/memtrace.cfg
cp $tmpdir/compile-538-clang.sh $tmpdir/cpu2017/benchspec/CPU