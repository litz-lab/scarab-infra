#!/usr/bin/env bash
# Put the AgentCPU benchmark at the commit the image pins into the build
# context: workloads/<group>/agentcpu-bench.tar (git-ignored). The Dockerfile
# checks the tar's commit against AGENTCPU_COMMIT, so a stale tar fails the build.
#   workloads/agentcpu/fetch_bench.sh [repository URL or path]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=${1:-${AGENTCPU_REPO:-https://github.com/litz-lab/agentcpu-bench.git}}
commit=$(sed -n 's/^ARG AGENTCPU_COMMIT=//p' "$here/Dockerfile")
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
git clone -q "$repo" "$tmp/b" && git -C "$tmp/b" archive --format=tar "$commit" > "$here/agentcpu-bench.tar"
for g in agentcpu_rust; do [ -d "$here/../$g" ] && cp "$here/agentcpu-bench.tar" "$here/../$g/"; done
echo "agentcpu-bench $commit -> $here/agentcpu-bench.tar"
