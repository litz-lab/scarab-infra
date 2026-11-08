#!/bin/bash
# Runs as root when an agentcpu_rust container starts (root_entrypoint.sh), from
# the bind-mounted checkout, so it does not change the image.
APPNAME="$1"

# code-rust tool-side capture (per-process mode): the replayed session builds
# and tests bat in /testbed, whose files (and the partly built target/ from the
# task image) belong to root. Trace jobs run as the invoking uid, and with root
# owning target/ the libgit2-sys build script fails with EPERM when it sets the
# permissions of files it copies there, so `cargo build` and `cargo test` stop
# early (the recording ran as root and built everything). Give the checkout to
# the trace uid. The orchestrator traces (coderust_codex_*, coderust_*_rustc)
# were captured as root and are not affected.
#
# Codex runs each command with the login shell of the user's passwd entry
# (`<shell> -lc`, not $SHELL). The recording ran as root, whose shell is
# /bin/bash; the trace user that root_entrypoint.sh adds gets useradd's default
# /bin/sh, and the commands then run under /bin/sh, which changes their output
# and skips the bash login profile the replay relies on. Give it /bin/bash.
case "$APPNAME" in
  coderust_*_tools*)
    if [ -n "$user_id" ]; then
      chown -R "$user_id:${group_id:-$user_id}" /testbed
    fi
    if [ -n "$username" ] && id -u "$username" >/dev/null 2>&1; then
      usermod -s /bin/bash "$username"
    fi ;;
esac
