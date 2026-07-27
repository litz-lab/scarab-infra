#!/bin/bash
#set -x #echo on

# The scarab-infra container-side scripts are NOT baked into the image; they are
# bind-mounted read-only from the host checkout at $INFRA_HOME (see the
# --mount ...target=/scarab_infra in every docker run). That way editing a
# script never invalidates an image layer, and Slurm jobs on every node run the
# exact scripts of the checkout that submitted them.
#
# We publish them into /usr/local/bin as symlinks so that the many existing
# absolute references (/usr/local/bin/run_clustering.sh, .../user_entrypoint.sh,
# the `source` line appended to ~/.bashrc, ...) keep working unchanged. Globbing
# instead of listing filenames means a newly added script needs no bookkeeping
# here. /usr/local/bin itself stays writable, which perf_entrypoint.sh relies on
# for its `perf` symlink.
INFRA_HOME="${INFRA_HOME:-/scarab_infra}"
if [ ! -d "$INFRA_HOME/common/scripts" ]; then
  echo "root_entrypoint.sh: scarab-infra scripts not found at '$INFRA_HOME'." >&2
  echo "The container must be started with:" >&2
  echo "  --mount type=bind,source=<scarab-infra checkout>,target=/scarab_infra,readonly=true" >&2
  exit 1
fi

for f in "$INFRA_HOME"/common/scripts/*.sh \
         "$INFRA_HOME"/common/scripts/*.py \
         "$INFRA_HOME"/scripts/utilities.sh \
         "$INFRA_HOME/workloads/$APP_GROUPNAME"/workload_root_entrypoint.sh \
         "$INFRA_HOME/workloads/$APP_GROUPNAME"/workload_user_entrypoint.sh; do
  [ -f "$f" ] && ln -sf "$f" "/usr/local/bin/$(basename "$f")"
done

if [ -n "$username" ] && [ -n "$group_id" ]; then
  if ! getent group "$username" &>/dev/null; then
    groupadd -g "$group_id" "$username"
  fi
fi

if [ -n "$username" ] && [ -n "$user_id" ]; then
  if ! id -u "$username" &>/dev/null; then
    if getent group "$username" &>/dev/null; then
      useradd -u "$user_id" -g "$username" -M "$username"
    else
      useradd -u "$user_id" -M "$username"
    fi
  fi
fi

if [ -f "/usr/local/bin/workload_root_entrypoint.sh" ]; then
  bash /usr/local/bin/workload_root_entrypoint.sh $APPNAME
fi

if [ -n "$DYNAMORIO_HOME" ] && [ -f "$DYNAMORIO_HOME/lib64/release/libdynamorio.so" ]; then
  chmod 777 "$DYNAMORIO_HOME/lib64/release/libdynamorio.so"
fi
