#!/bin/bash
#set -x #echo on

set -e

cd "${tmpdir:-/tmp_home}"
export DEBIAN_FRONTEND=noninteractive

apt_package_exists() {
  apt-cache show "$1" >/dev/null 2>&1
}

install_if_available() {
  if apt_package_exists "$1"; then
    apt-get install -y "$1"
  fi
}

apt-get update
install_if_available linux-tools-common
install_if_available linux-tools-generic
install_if_available "linux-tools-$(uname -r)"
install_if_available "linux-cloud-tools-$(uname -r)"
install_if_available linux-cloud-tools-generic
install_if_available linux-perf

perf_bin="$(find /usr/lib/linux-tools /usr/lib -path '*/perf' -type f 2>/dev/null | grep 'linux-tools' | head -1)"
if [ -z "$perf_bin" ] && command -v perf >/dev/null 2>&1; then
  perf_bin="$(command -v perf)"
fi
if [ -n "$perf_bin" ]; then
  ln -sf "$perf_bin" /usr/local/bin/perf
fi

if [ ! -d pmu-tools/.git ]; then
  if [ ! -d pmu-tools ]; then
    git clone https://github.com/andikleen/pmu-tools.git
  fi
fi
