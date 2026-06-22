#!/bin/bash
#set -x #echo on

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
