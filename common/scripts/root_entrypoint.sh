#!/bin/bash
#set -x #echo on

# Check if the user exists, and if not, create it without the home directory
if ! id -u "$username" &>/dev/null; then
  useradd -u "$user_id" -M "$username"
fi

# Check if the group exists, and if not, modify it
if ! getent group "$username" &>/dev/null; then
  groupmod -g "$group_id" "$username"
fi

if [ -f "/usr/local/bin/workload_root_entrypoint.sh" ]; then
  bash /usr/local/bin/workload_root_entrypoint.sh $APPNAME
fi

chmod 777 $DYNAMORIO_HOME/lib64/release/libdynamorio.so
