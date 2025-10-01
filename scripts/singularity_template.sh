# BEGIN SINGULARITY TEMPLATE
# This is a fragment of a shell script. Config options will be injected 2 lines above

echo "Running ${CONFIG} ${SUITE}/${SUBSUITE}/${WORKLOAD} ${SIMPOINT}"
echo "Running on $(uname -n)"

# From singularity build allbench_traces_8c3ae56.sif docker-daemon://allbench_traces:8c3ae56
SIF_IMAGE="${INFRA_DIR}/singularity_images/${APP_GROUPNAME}_${GIT_HASH}.sif"
WORKDIR="/home/${USERNAME}"
BIND_TRACES="${TRACES_DIR}:/simpoint_traces:ro"
BIND_HOME="${SIM_HOME}:/home/${USERNAME}:rw"
SCARAB_HOME="${SIM_HOME}/simulations/${EXPERIMENT_NAME}/scarab:/home/${USERNAME}/simulations/${EXPERIMENT_NAME}/scarab:ro"

# NOTE: If a workload isn't working, try uncommenting these and adding --overlay "$OVERLAY" to singularity commands
# Create a writable overlay for the container (optional, for /usr/local/bin writes)
# echo "Creating overlay for $CONTAINER_NAME"
# OVERLAY="${SIM_HOME}/simulations/${EXPERIMENT_NAME}/singularity/${CONTAINER_NAME}_overlay.img"
# rm -f $OVERLAY
# if [ ! -f "$OVERLAY" ]; then
#     singularity overlay create --size 1024 "$OVERLAY"
#     chmod 777 $OVERLAY
# fi

# Scripts already in container. If not, uncomment and add ,"$TMP_SCRIPTS:/usr/local/bin:rw" to binds
# Copy scripts into a temp dir to bind into the container
# TMP_SCRIPTS="tmp/${CONTAINER_NAME}_scripts"
# mkdir -p "$TMP_SCRIPTS"
# echo "Copying scripts to $TMP_SCRIPTS"
# cp ${INFRA_DIR}/scripts/utilities.sh "$TMP_SCRIPTS/"
# cp ${INFRA_DIR}/common/scripts/root_entrypoint.sh "$TMP_SCRIPTS/"
# cp ${INFRA_DIR}/common/scripts/user_entrypoint.sh "$TMP_SCRIPTS/"
# cp ${INFRA_DIR}/common/scripts/run_memtrace_single_simpoint.sh "$TMP_SCRIPTS/"

# Run root entrypoint as root inside the container
echo "Running root entrypoint in $CONTAINER_NAME"
singularity exec \
    --bind "$BIND_TRACES","$BIND_HOME","$SCARAB_HOME" \
    --env username=root \
    --home $WORKDIR \
    --env APP_GROUPNAME=${APP_GROUPNAME} \
    --env APPNAME=${WORKLOAD} \
    "$SIF_IMAGE" \
    /bin/bash -c "/usr/local/bin/root_entrypoint.sh"

# Run user entrypoint as user inside the container
# Last line (scarab run) removed. Will be injected by run script
echo "Running user entrypoint in $CONTAINER_NAME"
singularity exec \
    --bind "$BIND_TRACES","$BIND_HOME","$SCARAB_HOME" \
    --env user_id=${USER_ID} \
    --env group_id=${GID} \
    --env username=${USERNAME} \
    --home $WORKDIR \
    --env APP_GROUPNAME=${APP_GROUPNAME} \
    --env APPNAME=${WORKLOAD} \
    --env trace_home="/simpoint_traces"\
    --pwd "$WORKDIR" \
    "$SIF_IMAGE" \
    /bin/bash -c "source /usr/local/bin/user_entrypoint.sh && 