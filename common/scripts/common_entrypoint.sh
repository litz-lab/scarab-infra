#!/bin/bash
#set -x #echo on

useradd -u $user_id -m $username && groupmod -g $group_id $username
if [ -f "/usr/local/bin/entrypoint.sh" ]; then
  bash /usr/local/bin/entrypoint.sh $APPNAME
fi
