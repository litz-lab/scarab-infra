#!/bin/bash
#set -x #echo on

# code to ignore case restrictions
shopt -s nocasematch
source ./scripts/utilities.sh

# help function
help () {
  echo "Usage: ./run.sh [ -h | --help ]
                [ --list ]
                [ -b | --build ]
                [ --run ]
                [ --trace ]
                [ --simulation ]
                [ -k | --kill ]
                [ --status ]
                [ -c | --cleanup ]"
  echo
  echo "!! Modify '<experiment_name>.json' to specify the workloads to run Scarab simulations and Scarab parameters before run !!"
  echo "scarab-infra is an infrastructure that serves an environment where a user analyzes CPU metrics of a datacenter workload, applies SimPoint method to exploit program phase behavior, collects execution traces, and simulates the workload by using scarab microprocessor simulator."
  echo "1) workload setup by building a docker image and launching a docker container"
  echo "2) collect traces with different simpoint workflows for trace-based simulation"
  echo "3) run Scarab simulation in different modes"
  echo "To perform the later step, the previous steps must be performed first, meaning all the necessary options should be set at the same time. However, you can only run earlier step(s) by unsetting the later steps for debugging purposes."

  echo "Options:"
  echo "h           Print this Help."
  echo "list        List workload group names and workload names."
  echo "b           Build a docker image with application setup. The workload group name should be specified. For the available docker image name (workload group name), use -l. e.g) -b allbench_traces"
  echo "run         Run an interactive shell for a docker container. Provide a name of a json file in ./json to provide workload group name. e.g) --run exp (for exp.json)"
  echo "trace       Collect traces. Provide a name of a json file name in ./json. e.g.) --trace trace (for trace.json)"
  echo "simulation  Scarab simulation. Provide a name of a json file name in ./json. e.g) --simulation exp (for exp.json)"
  echo "k           Kill Scarab simulation related to the experiment of a json file. e.g) -k exp (for exp.json)"
  echo "status      Print status of docker/slurm nodes and running experiments/tracing jobs related to a json. Provide a name of a json file name in ./json. e.g) --status exp (for exp.json)"
  echo "c           Clean up all the containers related to an experiment. e.g) -c exp (for exp.json)"
}

# list function: list all the available docker image names
list () {
  workload_db_json_file="${INFRA_ROOT}/workloads/workloads_db.json"

  python3 -m scripts.run_db -dbg 3 -l -wdb ${workload_db_json_file}
}

# build function
build () {
  GROUP_LIST=$(ls -d ${INFRA_ROOT}/workloads/*/ | xargs -n 1 basename)

  FOUND=false
  for APP_GROUPNAME in ${GROUP_LIST}; do
    if [[ "$APP_GROUPNAME" == "$BUILD" ]]; then
      FOUND=true
      break
    fi
  done

  if ! $FOUND; then
    echo "Workload group name should be provided correctly (e.g. -b allbench_traces)."
    exit 1
  fi

  if [[ -n $(git status --porcelain $INFRA_ROOT/common $INFRA_ROOT/workloads/$APP_GROUPNAME | grep '^ M') ]]; then
    echo "There are uncommitted changes."
    echo "The repository is not up to date. Make sure to commit all the local changes for the version of the docker image. githash is used to identify the image."
    echo "If you have an image already in the system you want to overwrite, remove the image first, then build again."
    exit 1
  fi

  # get the latest Git commit hash
  GIT_HASH=$(git rev-parse --short HEAD)

  start=`date +%s`

  # check if the Docker image '$APP_GROUPNAME:$GIT_HASH' exists
  if ! docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "$APP_GROUPNAME:$GIT_HASH"; then
    echo "The image with the current Git commit hash does not exist. Find the same image but with the latest tag.."

    LATEST_HASH=$(cat $INFRA_ROOT/last_built_tag.txt)

    echo "Latest tag of the pre-built image: $LATEST_HASH"
    # check if the latest Docker image exists locally
    if ! docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "$APP_GROUPNAME:$LATEST_HASH"; then
      echo "The image with the latest tag does not exist. Pull first.."

      # Local image not found. Trying to pull pre-built image from GitHub Packages...
      if docker pull ghcr.io/litz-lab/scarab-infra/"$APP_GROUPNAME:$LATEST_HASH"; then
        echo "Successfully pulled pre-built image."
        docker tag ghcr.io/litz-lab/scarab-infra/"$APP_GROUPNAME:$LATEST_HASH" "$APP_GROUPNAME:$LATEST_HASH"
        echo "Tagged pulled image as $APP_GROUPNAME:$LATEST_HASH"
        docker rmi ghcr.io/litz-lab/scarab-infra/"$APP_GROUPNAME:$LATEST_HASH"
      else
        echo "No pre-built image found for $APP_GROUPNAME:$LATEST_HASH (or pull failed). This should not happen as the latest tag retrieved from the registry."
        exit 1
      fi
    fi

    # Run git diff for specific directories
    DIFF_OUTPUT=$(git diff $LATEST_HASH -- $INFRA_ROOT/common $INFRA_ROOT/workloads/$APP_GROUPNAME)

    if [ -n "$DIFF_OUTPUT" ]; then
      echo "Changes detected in ./common or ./workloads/$APP_GROUPNAME since $LATEST_HASH"
      echo "$DIFF_OUTPUT"
      echo "Build docker image locally..."
      # build from the beginning and overwrite whatever image with the same name
      docker build . -f ./workloads/$APP_GROUPNAME/Dockerfile --no-cache -t $APP_GROUPNAME:$GIT_HASH
    else
      echo "No changes in ./common or ./workloads/$APP_GROUPNAME since $LATEST_HASH, tag with $GIT_HASH"
      docker tag $APP_GROUPNAME:$LATEST_HASH $APP_GROUPNAME:$GIT_HASH
    fi
  fi

  end=`date +%s`
  report_time "pull/build-image" "$start" "$end"
}

run () {
  json_file="${INFRA_ROOT}/json/${RUN}.json"

  # Key to extract
  key="descriptor_type"

  # Extract the value using fq
  value=$(jq -r ".$key" "$json_file")

  if [ $value == "simulation" ]; then
    workload_db_json_file="${INFRA_ROOT}/workloads/workloads_db.json"

    APP_GROUPNAME=$(python3 -m scripts.run_db -dbg 1 -g ${json_file} -wdb ${workload_db_json_file})
    echo $APP_GROUPNAME

    BUILD=$APP_GROUPNAME
    build

    # open an interactive shell of docker container
    echo "open an interactive shell.."
    start=`date +%s`

    python3 -m scripts.run_simulation -dbg 3 -l -d ${json_file}

    end=`date +%s`
    report_time "interactive-shell" "$start" "$end"
  elif [ $value == "trace" ]; then
    APP_GROUPNAME=$(jq '.trace_configurations[0].image_name' ${json_file})
    echo $APP_GROUPNAME
    APP_GROUPNAME=$(echo "$APP_GROUPNAME" | tr -d '"')

    BUILD=$APP_GROUPNAME
    build

    # open an interactive shell of docker container
    echo "open an interactive shell.."
    start=`date +%s`

    python3 -m scripts.run_trace -dbg 3 -l -d ${json_file}

    end=`date +%s`
    report_time "interactive-shell" "$start" "$end"
  elif [ $value == "perf" ]; then
    APP_GROUPNAME=$(jq '.image_name' ${json_file})
    echo $APP_GROUPNAME
    APP_GROUPNAME=$(echo "$APP_GROUPNAME" | tr -d '"')

    BUILD=$APP_GROUPNAME
    build

    # open an interactive shell of docker container
    echo "open an interactive shell.."
    start=`date +%s`

    python3 -m scripts.run_perf -dbg 3 -l -d ${json_file}

    end=`date +%s`
    report_time "interactive-shell" "$start" "$end"
  fi
}

simpoint_trace () {
  # run clustering and tracing
  echo "run clustering and tracing.."
  taskPids=()
  start=`date +%s`

  cmd="python3 -m scripts.run_trace -dbg 3 -d ${INFRA_ROOT}/json/${SIMPOINT}.json"
  eval $cmd &
  taskPids+=($!)

  wait_for_non_child "simpoint/tracing" "${taskPids[@]}"
  end=`date +%s`
  report_time "simpoint/tracing" "$start" "$end"
}

run_scarab () {
  json_file="${INFRA_ROOT}/json/${SIMULATION}.json"
  workload_db_json_file="${INFRA_ROOT}/workloads/workloads_db.json"
  APP_GROUPNAME=$(python3 -m scripts.run_db -dbg 1 -g ${json_file} -wdb ${workload_db_json_file})
  echo $APP_GROUPNAME

  # make sure the corresponding docker image exists locally (required for scarab build)
  BUILD=$APP_GROUPNAME
  build

  # run Scarab simulation
  echo "run Scarab simulation.."
  taskPids=()
  start=`date +%s`

  cmd="python3 -m scripts.run_simulation -dbg 1 -d ${json_file}"
  eval $cmd &
  taskPids+=($!)

  wait_for_non_child "Scarab-simulation" "${taskPids[@]}"
  end=`date +%s`
  report_time "Scarab-simulation" "$start" "$end"
}

kill () {
  json_file="${INFRA_ROOT}/json/${KILL}.json"

  # Key to extract
  key="descriptor_type"

  # Extract the value using jq
  value=$(jq -r ".$key" "$json_file")

  if [ $value == "simulation" ]; then
    # kill Scarab simulation
    echo "kill scarab simulation of experiment $KILL .."
    start=`date +%s`
    python3 -m scripts.run_simulation -dbg 1 -k -d ${INFRA_ROOT}/json/${KILL}.json
    end=`date +%s`
    report_time "kill-Scarab-simulation" "$start" "$end"
  elif [ $value == "trace" ]; then
    # kill running clustering+tracing
    echo "kill tracing of $KILL.json .."
    start=`date +%s`
    python3 -m scripts.run_trace -dbg 3 -k -d ${INFRA_ROOT}/json/${KILL}.json
    end=`date +%s`
    report_time "kill-Tracing" "$start" "$end"
  fi
}

status () {
  json_file="${INFRA_ROOT}/json/${STATUS}.json"

  # Key to extract
  key="descriptor_type"

  # Extract the value using jq
  value=$(jq -r ".$key" "$json_file")

  if [ $value == "simulation" ]; then
    # print status of Scarab simulation or docker/slurm nodes
    echo "print docker/slurm node info and status of scarab simulation of experiment ${STATUS} .."
    start=`date +%s`

    python3 -m scripts.run_simulation -dbg 1 -i -d ${json_file}

    end=`date +%s`
    report_time "print-status" "$start" "$end"
  elif [ $value == "trace" ]; then
    # print status of Trace jobs or docker/slurm nodes
    echo "print docker/slurm node info and status of tracing jobs of trace ${STATUS} .."
    start=`date +%s`

    python3 -m scripts.run_trace -dbg 3 -i -d ${json_file}

    end=`date +%s`
    report_time "print-status" "$start" "$end"
  fi
}

cleanup () {
  json_file="${INFRA_ROOT}/json/${CLEANUP}.json"

  # Key to extract
  key="descriptor_type"

  # Extract the value using jq
  value=$(jq -r ".$key" "$json_file")

  if [ $value == "simulation" ]; then
    echo "clean up the containers running simulations from ${CLEANUP}.json .."
    start=`date +%s`

    python3 -m scripts.run_simulation -dbg 1 -c -d ${json_file}

    end=`date +%s`
    report_time "container-cleanup" "$start" "$end"
  elif [ $value == "trace" ]; then
    echo "clean up the containers running tracing from ${CLEANUP}.json .."
    start=`date +%s`

    python3 -m scripts.run_trace -dbg 3 -c -d ${json_file}

    end=`date +%s`
    report_time "container-cleanup" "$start" "$end"
  fi
}

if [ -f "README.md" ]; then
  INFRA_ROOT=$(pwd)
else
  echo "Run this script in the root directory of the repository."
  exit 1
fi

SHORT=h,b:,k:,c:
LONG=help,list,build:,run:,trace:,simulation:,kill:,status:,cleanup:
OPTS=$(getopt -a -n "$(basename "$0")" --options $SHORT --longoptions $LONG -- "$@")

if [ $? -ne 0 ]; then
  echo "Error parsing options."
  exit 1
fi

if [ $# -eq 0 ]; then
  echo "No arguments provided. Showing help."
  help
  exit 0
fi

eval set -- "$OPTS"

# Get the options
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) # display help
      help
      exit 0
      ;;
    --list) # list all the workload groups
      list
      exit 0
      ;;
    -k|--kill) # kill scarab simulation
      KILL="$2"
      kill
      exit 0
      ;;
    --status) # print status of docker/slurm nodes and scarab simulation
      STATUS="$2"
      status
      exit 0
      ;;
    -b|--build) # build a docker image with application setup required during the building time
      BUILD="$2"
      shift 2
      ;;
    --run) # run a docker container with application setup required during the launching time
      RUN="$2"
      shift 2
      ;;
    -t|--trace) # collect traces with simpoint workflows
      SIMPOINT="$2"
      shift 2
      ;;
    --simulation) # scarab simulation
      SIMULATION="$2"
      shift 2
      ;;
    -c|--cleanup) # clean up the containers
      CLEANUP="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *) # unexpected option
      echo "Unexpected option: $1"
      exit 1
      ;;
  esac
done

if [ $BUILD ]; then
  build
fi

if [ $RUN ]; then
  run
  exit 0
fi

if [ $SIMPOINT ]; then
  simpoint_trace
  exit 0
fi

if [ $SIMULATION ]; then
  run_scarab
  exit 0
fi

if [ $CLEANUP ]; then
  cleanup
  exit 0
fi
