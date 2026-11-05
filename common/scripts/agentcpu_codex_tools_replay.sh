#!/usr/bin/env bash
# Replay of a code-rust (Codex) bundle with the whole Codex process tree traced
# and the harness native, for the per-process tool-side capture
# (json/agentcpu_trace_tools_coderust.json).
#
# Same steps as the benchmark's bundles/code-rust/replay_codex_trace.sh, with
# three differences:
#  * Only codex runs under DynamoRIO: it is started through the shell wrapper
#    whose path run_simpoint_trace.py puts in CODEX_TRACE_SHELL
#    (SCARAB_TRACE_SHELL_VAR=CODEX_TRACE_SHELL), and children are followed, so
#    codex and every process it starts (bash, cargo, rustc and its LLVM
#    threads, cc1, ld, build scripts, the test binaries, git) are traced. The
#    replay proxy, the checkout reset and this script run natively, so they do
#    not appear in the capture. Without CODEX_TRACE_SHELL it runs untraced.
#  * The proxy is codex_replay_quiesce.py: the recorded answers, served only
#    once codex has no running descendants, so the slowed build and test
#    commands run to the end instead of being cut when the recorded session
#    moves on or ends (see that script).
#  * Codex's provider stream idle timeout is raised and its retries are off, so
#    a long wait for an answer is not taken as a failed request and re-sent
#    (a re-sent request would take the next recorded answer).
# Usage: agentcpu_codex_tools_replay.sh <bundle_dir> <out_dir>
# Prints RC, MISSES, CALLS, PATCH (as replay_codex_trace.sh) and CONTENT (codex_hash.py).
set -u
CB=${AGENTCPU:-/tmp_home/agentcpu-bench}/bundles/code-rust
B=$1; OUT=$2; mkdir -p "$OUT/target_modules"; RP=$((18800 + RANDOM % 150))
export HOME=/tmp CODEX_HOME=/tmp/codex_home_$$; mkdir -p $CODEX_HOME; cp "$CB/config.toml" $CODEX_HOME/
export PATH=$CB/vendor/bin:$CB/vendor/codex-path:/usr/local/cargo/bin:$PATH RUSTUP_HOME=/usr/local/rustup CARGO_HOME=/usr/local/cargo
export CARGO_NET_OFFLINE=true GIT_TERMINAL_PROMPT=0 npm_config_offline=true npm_config_update_notifier=false NO_UPDATE_NOTIFIER=1 GOPROXY=off AGENTCPU_KEY=x
# Login shells as in the recording. Codex runs every command with `bash -lc`.
# The recording ran as root in the task's own image, whose /etc/profile resets
# PATH for login shells, so /usr/local/cargo/bin was not on the commands' PATH:
# `cargo build` at steps 6 and 8 printed "cargo: command not found" and the
# build ran at step 9, when the model put /usr/local/cargo/bin on PATH itself.
# This image's /etc/profile keeps PATH, so the build would run at step 6 and bat
# would be built again at step 8. A login profile in HOME that drops
# /usr/local/cargo/bin (pure bash, so no extra process is traced) restores the
# recorded sequence; the command-output hash then equals the recording's.
cat > "$HOME/.bash_profile" <<'EOF_PROFILE'
p=":$PATH:"; while [[ $p == *:/usr/local/cargo/bin:* ]]; do p=${p//:\/usr\/local\/cargo\/bin:/:}; done
p=${p#:}; PATH=${p%:}; export PATH; unset p
EOF_PROFILE
QP=$(dirname "$(readlink -f "$0")")/codex_replay_quiesce.py
python3 "$QP" "$CB/codex_responses_proxy.py" $RP "$B/responses" > "$OUT/proxy.log" 2>&1 & PX=$!
for _ in $(seq 1 300); do python3 -c "import socket,sys; s=socket.socket(); s.settimeout(0.2); sys.exit(s.connect_ex((\"127.0.0.1\", int(sys.argv[1]))))" $RP && break; sleep 0.2; done
cd /testbed && git checkout -q -f . && git clean -qfd
# The traced part: one script, run by the wrapper's bash, which execs codex.
cat > "$OUT/run_codex.sh" <<EOF
exec codex exec --json --skip-git-repo-check \\
  -c 'model_providers.agentcpu.base_url="http://127.0.0.1:$RP/v1"' \\
  -c model_providers.agentcpu.stream_idle_timeout_ms=86400000 \\
  -c model_providers.agentcpu.request_max_retries=0 \\
  -c model_providers.agentcpu.stream_max_retries=0 \\
  -C /testbed "\$(cat $B/prompt.txt)" < /dev/null > "$OUT/events.jsonl" 2> "$OUT/codex.err"
EOF
T0=$(date +%s)
"${CODEX_TRACE_SHELL:-/bin/bash}" "$OUT/run_codex.sh"; RC=$?
T1=$(date +%s)
git diff > "$OUT/produced.patch"; kill $PX 2>/dev/null
# drraw2trace reopens every module by path, and the build's own binaries (build
# scripts, proc-macro libraries, test binaries) exist only in this container.
# Keep a copy so that a conversion resumed in a new container can restore them
# to /testbed/target (or pass them with drraw2trace -alt_module_dir).
( cd / && find testbed/target -type f \( -perm -u+x -o -name '*.so' \) -print0 | \
    xargs -0 -r cp --parents -p -t "$OUT/target_modules" 2>/dev/null ) || true
grep '^quiesce' "$OUT/proxy.log"
echo "WALL=$((T1 - T0))s CONTENT=$(python3 "$CB/codex_hash.py" "$OUT/events.jsonl" 2>/dev/null | awk '{print $NF}' | head -1)"
echo "RC=$RC MISSES=$(grep -c MISS $OUT/proxy.log) CALLS=$(grep -c shape $OUT/proxy.log) PATCH=$(sha256sum $OUT/produced.patch | cut -c1-16)"; exit $RC
