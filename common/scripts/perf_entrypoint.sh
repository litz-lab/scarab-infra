#!/bin/bash
#set -x #echo on

cd $tmpdir
apt-get install -y linux-tools-common linux-tools-generic linux-tools-`uname -r`
sudo ln -s "$(find /usr/lib/linux-tools/*/perf | head -1)" /usr/local/bin/perf
alias perf=$(find /usr/lib/linux-tools/*/perf | head -1)
cd $tmpdir && git clone https://github.com/andikleen/pmu-tools.git
