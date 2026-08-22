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

link_local_rundir () {
  # 1: src:  private, node-local copy of the run dir ($RUNDIR)
  # 2: dest: simdir, which is the program's CWD under Pin
  # Symlink each top-level entry of <src> into <dest>, so a workload that opens an input by a
  # hard-coded CWD-relative path resolves it without a second copy of the data.
  # Linking is safe here because the targets are the private copy, not the shared app tree.
  local src="$1" dest="$2" path name
  mkdir -p "$dest"
  for path in "$src"/*; do
    [ -e "$path" ] || continue
    name=$(basename "$path")
    case "$name" in
      PARAMS.in|pin_exec.so|scarab.out|scarab.err|pin.out|pin.err|launch_cmd.txt) continue ;;
    esac
    ln -sfn "$path" "$dest/$name"
  done
}
