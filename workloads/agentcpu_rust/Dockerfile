# syntax = edrevo/dockerfile-plus
#
# AgentCPU (suite `agentcpu`), code-rust scenario: Codex CLI on SWE-bench
# Multilingual sharkdp/bat. The recorded tool calls run in /testbed with the
# task's Rust toolchain, so the toolchain and the checkout come from the task's
# official evaluation image; the cargo registry is fetched for every commit the
# three-task chain visits, so replays run offline.
FROM ubuntu:jammy

INCLUDE+ ./common/Dockerfile.common

USER root
RUN pip3 install --no-cache-dir uv && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y gcc

COPY --from=swebench/sweb.eval.x86_64.sharkdp_1776_bat-2393:latest /usr/local/rustup /usr/local/rustup
COPY --from=swebench/sweb.eval.x86_64.sharkdp_1776_bat-2393:latest /usr/local/cargo /usr/local/cargo
COPY --from=swebench/sweb.eval.x86_64.sharkdp_1776_bat-2393:latest /testbed /testbed
ENV RUSTUP_HOME=/usr/local/rustup CARGO_HOME=/usr/local/cargo PATH=/usr/local/cargo/bin:$PATH
RUN cd /testbed && head=$(git rev-parse HEAD) && \
    for c in 59d4cfb75c1a17a26b2b38956ee08d8c53f04e04 c14ce4f7caa36fe524975c51f7aae3d5a852d1f9 $head; do \
      git cat-file -e $c^{commit} 2>/dev/null || git fetch -q origin $c; \
      git checkout -q -f $c && cargo fetch -q; done && \
    git checkout -q -f $head && git remote remove origin 2>/dev/null; \
    git config --system --add safe.directory '*' && cargo --version

# 5649d56: 3228ef9 plus codex_hash.py normalising cargo's minute durations
# (the check of the tool-side capture, under DynamoRIO, depends on it).
ARG AGENTCPU_COMMIT=5649d56340a5ae37be8b39afa40e0f7f8080eba2
COPY ./workloads/agentcpu_rust/agentcpu-bench.tar /tmp/agentcpu-bench.tar
RUN test "$(git get-tar-commit-id < /tmp/agentcpu-bench.tar)" = "$AGENTCPU_COMMIT" && \
    mkdir -p $tmpdir/agentcpu-bench && tar -xf /tmp/agentcpu-bench.tar -C $tmpdir/agentcpu-bench && rm /tmp/agentcpu-bench.tar
ENV AGENTCPU=$tmpdir/agentcpu-bench UV_PYTHON_INSTALL_DIR=$tmpdir/uv_python
RUN $AGENTCPU/bundles/code-rust/setup.sh

RUN chmod -R a+rwX /testbed $AGENTCPU && chmod -R a+rX /usr/local/rustup /usr/local/cargo $tmpdir/uv_python

# DynamoRIO 11.3.0 (release) for the tool-side capture of the code-rust session
# (json/agentcpu_trace_tools_coderust.json, SCARAB_TRACE_DR_HOME). The image's
# DR 10.0 (Dockerfile.common) deadlocked in a forked child of the libgit2-sys
# build script and let a rustc die with SIGSEGV when the whole cargo build was
# traced. 11.3.0 writes the same trace format as the DR source Scarab reads
# traces with (cronbuild-11.90.20140: trace version 6, offline file version 8).
# DYNAMORIO_HOME stays DR 10.0 for everything else.
RUN cd $tmpdir && wget -q https://github.com/DynamoRIO/dynamorio/releases/download/release_11.3.0-1/DynamoRIO-Linux-11.3.0.tar.gz -O dr.tar.gz && \
    echo "ae02049df5c4daeb82d4087aaa0d38bf58cdd26b38fb6042dccf27cbc32ceed9  dr.tar.gz" | sha256sum -c - && \
    tar -xzf dr.tar.gz && rm dr.tar.gz && chmod -R a+rX DynamoRIO-Linux-11.3.0-1
ENV AGENTCPU_TRACE_DR_HOME=$tmpdir/DynamoRIO-Linux-11.3.0-1
