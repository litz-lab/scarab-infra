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