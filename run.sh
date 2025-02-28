#!/bin/bash
#set -x #echo on

# code to ignore case restrictions
shopt -s nocasematch
source ./scripts/utilities.sh
declare -A WL_LIST

# help function
help () {
  echo "Usage: ./run.sh [ -h | --help ]
                [ --list ]
                [ -b | --build ]
                [ --run ]
                [ -w | --workloads ]
                [ -t | --trace ]
                [ --simulation ]
                [ --status ]
                [ -k | --kill ]
                [ -c | --cleanup ]"
  echo
  echo "!! Modify '<experiment_name>.json' to specify the workloads to run Scarab simulations and Scarab parameters before run !!"
  echo "scarab-infra is an infrastructure that serves an environment where a user analyzes CPU metrics of a datacenter workload, applies SimPoint method to exploit program phase behavior, collects execution traces, and simulates the workload by using scarab microprocessor simulator."
  echo "This script serves three a center workload is the following."
  echo "1) workload setup by building a docker image and launching a docker container"
  echo "2) collect traces with different simpoint workflows for trace-based simulation; **this will turn off ASLR; the user needs to recover the ASLR setting afterwards manually by running on host**: \`echo 2 | sudo tee /proc/sys/kernel/randomize_va_space\`"
  echo "3) run Scarab simulation in different modes"
  echo "To perform the later step, the previous steps must be performed first, meaning all the necessary options should be set at the same time. However, you can only run earlier step(s) by unsetting the later steps for debugging purposes."

  echo "Options:"
  echo "h           Print this Help."
  echo "l           List workload group names and workload names."
  echo "b           Build a docker image with application setup. The workload group name should be specified. For the available docker image name (workload group name), use -l. e.g) -b allbench_traces"
  echo "r           Run an interactive shell for a docker container. Provide a name of a json file in ./json to provide workload group name. e.g) -r exp (for exp.json)"
  echo "w           List of workloads for simpoint/tracing. Should be used with -t."
  echo "t           Collect traces with different SimPoint workflows. 0: Do not collect traces, 1: Collect traces based on SimPoint workflow - collect fingerprints, do simpoint clustering, trace, 2: Collect traces based on SimPoint post-processing workflow - trace, collect fingerprints, do simpoint clustering, 3: Only collect traces without simpoint clustering. e.g) -t 2"
  echo "s           Scarab simulation. Provide a name of a json file name in ./json. e.g) -s exp (for exp.json)"
  echo "k           Kill Scarab simulation related to the experiment of a json file. e.g) -k exp (for exp.json)"
  echo "i           Print status of docker/slurm nodes and running experiments related to json. e.g) -i exp (for exp.json)"
  echo "c           Clean up all the containers related to an experiment. e.g) -c exp (for exp.json)"
}

# list function: list all the available docker image names
list () {
  WORKLOAD_PATH="./workloads"
  WL_GROUPS=()
  WL_GROUPS=($(find "$WORKLOAD_PATH" -mindepth 1 -maxdepth 1 -type d -exec basename {} \;))

  for GROUP_NAME in "${WL_GROUPS[@]}"; do
    file="$WORKLOAD_PATH/$GROUP_NAME/apps.list"
    if [[ ! -f $file ]]; then
      echo "apps.list not found in $WORKLOAD_PATH/$GROUP_NAME"
      continue
    fi

    WORKLOAD_NAMES=()
    while IFS= read -r line; do
      WORKLOAD_NAMES+=("$line")
    done < "$file"

    WL_LIST["$GROUP_NAME"]="${WORKLOAD_NAMES[@]}"
  done
}

# build function
build () {
  list
  FOUND=false
  for APP_GROUPNAME in "${!WL_LIST[@]}"; do
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

  # check if the Docker image '$APP_GROUPNAME:$GIT_HASH' exists
  if docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "$APP_GROUPNAME:$GIT_HASH"; then
    echo "The image with the same Git commit hash already exists! If you want to overwrite it, remove the old one first and try again."
    exit 1
  fi

  # Local image not found. Trying to pull pre-built image from GitHub Packages...
  if docker pull ghcr.io/litz-lab/scarab-infra/"$APP_GROUPNAME:$GIT_HASH"; then
    echo "Successfully pulled pre-built image."
    docker tag ghcr.io/litz-lab/scarab-infra/"$APP_GROUPNAME:$GIT_HASH" "$APP_GROUPNAME:$GIT_HASH"
    echo "Tagged pulled image as $APP_GROUPNAME:$GIT_HASH"
    docker rmi ghcr.io/litz-lab/scarab-infra/"$APP_GROUPNAME:$GIT_HASH"
    exit 1
  else
    echo "No pre-built image found for $APP_GROUPNAME:$GIT_HASH (or pull failed)."
    echo "Will build locally..."
  fi

  # build a docker image
  echo "build docker image.."
  start=`date +%s`
  # build from the beginning and overwrite whatever image with the same name
  docker build . -f ./workloads/$APP_GROUPNAME/Dockerfile --no-cache -t $APP_GROUPNAME:$GIT_HASH
  end=`date +%s`
  report_time "build-image" "$start" "$end"
}

run () {
  # WL_LIST is filled if build is executed within a single run
  if [ ${#WL_LIST[@]} -eq 0 ]; then
    list
  fi

  json_file="${INFRA_ROOT}/json/${RUN}.json"
  key="workloads_list"
  APP_LIST=$(jq -r ".$key" "$json_file")
  size=${#APP_LIST[@]}
  if (( size > 1 )); then
    echo "Only one workload should be provided for an interactive shell of docker container."
    exit 1
  fi

  NAME=${APP_LIST[0]}
  FOUND=false
  for GROUPNAME in "${!WL_LIST[@]}"; do
    for APPNAME in "${WL_LIST[$GROUPNAME]}"; do
      if [[ "$APPNAME" == "$NAME" ]]; then
        set_app_groupname
        if [[ "$APP_GROUPNAME" == "$GROUPNAME" ]]; then
          FOUND=true
          break
        fi
      fi
    done
  done

  if [ ! -n "$FOUND" ]; then
    echo "Workload name should be provided correctly within the correct workload group (e.g. '-r mysql' within workload group 'sysbench')."
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

  # check if the Docker image '$APP_GROUPNAME:$GIT_HASH' exists
  if ! docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "$APP_GROUPNAME:$GIT_HASH"; then
    echo "The image with the current Git commit hash does not exist! Build the image first by using '-b'."
    exit 1
  fi


  # open an interactive shell of docker container
  echo "open an interactive shell.."
  start=`date +%s`

  python3 ${INFRA_ROOT}/scripts/run_simulation.py -dbg 3 -l -d ${INFRA_ROOT}/json/${RUN}.json

  end=`date +%s`
  report_time "interactive-shell" "$start" "$end"
}

simpoint_trace () {
  if [[ ${#WORKLOADS[@]} -eq 0 ]]; then
    echo "Workloads not provided with -w."
    exit 1
  fi

  # tokenize multiple environment variables
  ENVVARS=""
  echo $ENVVARS
  for token in $ENVVAR; do
    ENVVARS+=" -e ";
    ENVVARS+=$token;
  done

  # run simpoint/trace
  echo "run simpoint/trace.."
  taskPids=()
  start=`date +%s`

  for APPNAME in "${WORKLOADS[@]}"; do
    if [ "$APPNAME" == "allbench" ]; then
      echo "allbench is only for trace-based simulations with the traces from UCSC NFS"
      exit 1
    fi
    set_app_groupname
    set_app_bincmd

    # update the script
    docker cp $INFRA_ROOT/common/scripts/run_simpoint_trace.sh $APP_GROUPNAME\_$USER:/usr/local/bin
    # disable ASLR;
    # the user needs to recover the ASLR setting afterwards manually by running on host:
    # echo 2 | sudo tee /proc/sys/kernel/randomize_va_space
    docker exec --privileged $APP_GROUPNAME\_$USER /bin/bash -c "echo 0 | sudo tee /proc/sys/kernel/randomize_va_space"
    docker exec $ENVVARS --user $USER --workdir /home/$USER --privileged $APP_GROUPNAME\_$USER run_simpoint_trace.sh "$APPNAME" "$APP_GROUPNAME" "$BINCMD" "$SIMPOINT" "$DRIO_ARGS" &
    sleep 2
    while read -r line ;do
      IFS=" " read PID CMD <<< $line
      if [ "$CMD" == "/bin/bash /usr/local/bin/run_simpoint_trace.sh $APPNAME $APP_GROUPNAME $BINCMD $SIMPOINT $DRIO_ARGS" ]; then
        taskPids+=($PID)
      fi
    done < <(docker top $APP_GROUPNAME\_$USER -eo pid,cmd)
  done

  wait_for_non_child "simpoint/tracing" "${taskPids[@]}"
  end=`date +%s`
  report_time "post-processing" "$start" "$end"
}

run_scarab () {
  # run Scarab simulation
  echo "run Scarab simulation.."
  taskPids=()
  start=`date +%s`

  cmd="python3 ${INFRA_ROOT}/scripts/run_simulation.py -dbg 3 -d ${INFRA_ROOT}/json/${SIMULATION}.json"
  eval $cmd &
  taskPids+=($!)

  wait_for_non_child "Scarab-simulation" "${taskPids[@]}"
  end=`date +%s`
  report_time "Scarab-simulation" "$start" "$end"
}

kill () {
  # kill Scarab simulation
  echo "kill scarab simulation of experiment $KILL .."
  start=`date +%s`

  python3 ${INFRA_ROOT}/scripts/run_simulation.py -dbg 3 -k -d ${INFRA_ROOT}/json/${KILL}.json

  end=`date +%s`
  report_time "kill-Scarab-simulation" "$start" "$end"
}

status () {
  # print status of Scarab simulation or docker/slurm nodes
  echo "print docker/slurm node info and status of scarab simulation of experiment $KILL .."
  start=`date +%s`

  python3 ${INFRA_ROOT}/scripts/run_simulation.py -dbg 3 -i -d ${INFRA_ROOT}/json/${STATUS}.json

  end=`date +%s`
  report_time "print-status" "$start" "$end"
}

cleanup () {
  echo "clean up the containers.."
  start=`date +%s`
#  for APPNAME in "${WORKLOADS[@]}"; do
    #set_app_groupname
    #case $APPNAME in
      # solr requires extra cleanup
      #solr)
        #rmCmd="docker rm web_search_client"
        #eval $rmCmd &
        #taskPids+=($!)
        #sleep 2
        #;;
    #esac
    #docker stop $APP_GROUPNAME\_$USER
    #rmCmd="docker rm $APP_GROUPNAME\_$USER"
    #eval $rmCmd &
    #taskPids+=($!)
    #sleep 2
  #done

  python3 ${INFRA_ROOT}/scripts/run_simulation.py -dbg 3 -c -d ${INFRA_ROOT}/json/${CLEANUP}.json

  end=`date +%s`
  report_time "container-cleanup" "$start" "$end"
}

if [ -f "README.md" ]; then
  INFRA_ROOT=$(pwd)
else
  echo "Run this script in the root directory of the repository."
  exit 1
fi

SHORT=h,b:,w:,t:,k:,c:
LONG=help,list,build:,run:,workload:,trace:,simulation:,kill:,status:,cleanup:
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
      echo "WORKLOAD_GROUPNAME: WORKLOAD_NAME1 WORKLOAD_NAME2 ..."
      for GROUP_NAME in "${!WL_LIST[@]}"; do
        echo "$GROUP_NAME: ${WL_LIST[$GROUP_NAME]}"
      done
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
    -w|--workload) # list of workloads for simpoint/tracing
      WORKLOADS=()
      shift
      while [[ $# -gt 0 && $1 != -* ]]; do
        WORKLOADS+=("$1")
        shift
      done
      ;;
    -t|--trace) # collect traces with simpoint workflows
      SIMPOINT=$2
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
