#!/bin/bash
# Runs as root when an agentcpu container starts (root_entrypoint.sh), from the
# bind-mounted checkout, so it does not change the image.
APPNAME="$1"

# code-py with SWE-ReX's local deployment: SWE-agent installs its tool bundles
# into /root/tools, but trace and perf jobs run as an unprivileged uid and the
# image's /root is 0700.
case "$APPNAME" in
  codepy_*) chmod a+rwx /root ;;
esac
