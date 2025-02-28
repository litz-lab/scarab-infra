#!/bin/bash
set -x

if [ ! -d "/home/$username/cpu2017" ]; then
    mkdir /home/$username/cpu2017
    cd $tmpdir/cpu2017_install && echo "yes" | ./install.sh -d /home/$username/cpu2017

    cp $tmpdir/memtrace.cfg /home/$username/cpu2017/config/memtrace.cfg
    cp $tmpdir/compile-538-clang.sh /home/$username/cpu2017/benchspec/CPU
else
    echo "cpu2017 exists. skip installation."
fi