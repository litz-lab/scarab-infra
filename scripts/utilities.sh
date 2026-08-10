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

stage_local_rundir () {
  # 1: bincmd
  # 2: dest
  # Copy the workload's reference run dir (= dir of the binary, the first token of <bincmd>) into
  # <dest>, EXCLUDING the binary itself, dereferencing symlinks and making everything user-writable.
  # Some SPEC CPU2026 workloads open inputs by a hard-coded relative path not on the
  # command line (e.g. ntest uses "resource/solver12.txt", gem5 uses Resource(...,".")),
  # and those resolve only when the process runs with CWD=<dest>.
  # Staging the data here lets the caller point the workload's CWD at <dest> without
  # copying the (possibly ~1GB) binary or writing into the shared, read-only NFS app tree.
  local bincmd="${1//\$tmpdir/$tmpdir}"
  local binary="${bincmd%% *}"
  local refdir; refdir=$(dirname "$binary")
  local dest="$2"
  mkdir -p "$dest"
  ( cd "$refdir" && find . -mindepth 1 -maxdepth 1 ! -name "$(basename "$binary")" \
      -exec cp -rL --preserve=timestamps -t "$dest" {} + )
  chmod -R u+rwX "$dest"
}
