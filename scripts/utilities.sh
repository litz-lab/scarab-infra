#!/bin/bash

# functions
wait_for () {
  # ref: https://askubuntu.com/questions/674333/how-to-pass-an-array-as-function-argument
  # 1: procedure name
  # 2: task list
  local procedure="$1"
  shift
  local taskPids=("$@")
  echo "${taskPids[@]}"
  echo "wait for all $procedure to finish..."
  # ref: https://stackoverflow.com/a/29535256
  for taskPid in ${taskPids[@]}; do
    echo "wait for $taskPid $procedure process"
    if wait $taskPid; then
      echo "$procedure process $taskPid success"
    else
      echo "$procedure process $taskPid fail"
      exit
    fi
  done
}

report_time () {
  # 1: procedure name
  # 2: start
  # 3: end
  local procedure="$1"
  local start="$2"
  local end="$3"
  local runtime=$((end-start))
  local hours=$((runtime / 3600));
  local minutes=$(( (runtime % 3600) / 60 ));
  local seconds=$(( (runtime % 3600) % 60 ));
  echo "$procedure Runtime: $hours:$minutes:$seconds (hh:mm:ss)"
}

wait_for_non_child () {
  # ref: https://askubuntu.com/questions/674333/how-to-pass-an-array-as-function-argument
  # 1: procedure name
  # 2: task list
  local procedure="$1"
  shift
  local taskPids=("$@")
  echo "${taskPids[@]}"
  echo "wait for all $procedure to finish..."
  # ref: https://stackoverflow.com/a/29535256
  for taskPid in ${taskPids[@]}; do
    echo "wait for $taskPid $procedure process"
    while [ -d "/proc/$taskPid" ]; do
      sleep 10 & wait $!
    done
  done
}

perf_stat_prefix () {
  # Prefix that makes `perf stat` count the host instructions a simulation
  # retires, written to $1. Wall-clock speed (KIPS) tracks machine load as much
  # as Scarab; the instruction count does not, so it is what CI regresses on.
  # Prints nothing when perf cannot count here (no perf, restrictive
  # perf_event_paranoid), so the simulation still runs -- just unmeasured.
  local out="$1"
  local perf_bin
  perf_bin=$(command -v perf 2>/dev/null)
  # The image's /usr/bin/perf is a wrapper that refuses to run when it finds no
  # perf for the running kernel; the versioned linux-tools binary works anyway.
  if [ -z "$perf_bin" ] || ! "$perf_bin" --version >/dev/null 2>&1; then
    perf_bin=$(ls -d /usr/lib/linux-tools-*/perf 2>/dev/null | head -1)
  fi
  [ -n "$perf_bin" ] && [ -x "$perf_bin" ] || return 0
  # User-mode only: kernel counting needs perf_event_paranoid <= 1, and syscall
  # instructions are noise here.
  "$perf_bin" stat -e instructions:u -x, -o /dev/null -- true >/dev/null 2>&1 || return 0
  printf '%s stat -e instructions:u -x, -o %q --' "$perf_bin" "$out"
}
