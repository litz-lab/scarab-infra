# syntax = edrevo/dockerfile-plus
#
# AgentCPU (suite `agentcpu`): the api, code-py, data and code-node scenarios.
# Clones the benchmark at a pinned commit and builds every scenario's
# environment with the benchmark's own setup scripts, so `--trace` and `--perf` replay the recorded sessions here.
# code-rust needs the task's Rust toolchain at /testbed and lives in
# workloads/agentcpu_rust.
FROM ubuntu:jammy

INCLUDE+ ./common/Dockerfile.common

USER root

# DynamoRIO without annotation support. V8 (code-node) contains the Valgrind
# client-request sequence; a default build intercepts it and drops it from the
# block, which makes raw2trace misattribute memory references. The other
# scenarios trace the same either way.
RUN cd $tmpdir/DynamoRIO-Linux-10.0.0 && cmake -DANNOTATIONS=OFF . > /dev/null && \
    make -j$(nproc) all > /dev/null && make install > /dev/null && \
    cp -r clients/bin64/* tools/bin64/ && chmod 777 $DYNAMORIO_HOME/lib64/release/libdynamorio.so

RUN pip3 install --no-cache-dir uv && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y gcc

# The AgentCPU benchmark (recorded sessions, replay and record scripts) at a pinned commit.
ARG AGENTCPU_REPO=https://github.com/litz-lab/agentcpu-bench.git
ARG AGENTCPU_COMMIT=ddb6471df4e097fc210b3a7657c3696ea47262c6
RUN git clone -q $AGENTCPU_REPO $tmpdir/agentcpu-bench && git -C $tmpdir/agentcpu-bench checkout -q $AGENTCPU_COMMIT
ENV AGENTCPU=$tmpdir/agentcpu-bench UV_PYTHON_INSTALL_DIR=$tmpdir/uv_python

# api: the runners on the image's CPython 3.10 (as captured), AppWorld in its
# own venv with its data, offline tokenizer and embedding model, marker library.
RUN grep -v '^#' $AGENTCPU/bundles/api/requirements.lock > /tmp/api_req.txt && \
    pip3 install --no-cache-dir -r /tmp/api_req.txt
RUN cd $AGENTCPU/bundles/api && \
    uv venv -q --python 3.12 appworld_venv && uv pip install -q --python appworld_venv/bin/python appworld==0.1.3.post1 && \
    mkdir -p appworld && cd appworld && ../appworld_venv/bin/appworld install && ../appworld_venv/bin/appworld download data --root . && \
    cd .. && HF_HOME=$PWD/hf python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('gpt2')" && \
    gcc -shared -fPIC -O2 -o $tmpdir/libagent_markers.so $AGENTCPU/agentcpu/markers/marker.c
ENV PYTHONPATH=$AGENTCPU APPWORLD_ROOT=$AGENTCPU/bundles/api/appworld \
    APPWORLD_PYTHON=$AGENTCPU/bundles/api/appworld_venv/bin/python HF_HOME=$AGENTCPU/bundles/api/hf

# code-py: SWE-agent venv, and the local-deployment checkout at /testbed.
RUN $AGENTCPU/bundles/code-py/setup.sh && $AGENTCPU/bundles/code-py/setup_local.sh && \
    git config --system --add safe.directory '*'

# data: Open Interpreter venv with its Jupyter kernel, DABStep files.
RUN $AGENTCPU/bundles/data/setup.sh

# code-node: the task checkout and venv, both Claude Code builds (node 22 comes
# from common).
RUN $AGENTCPU/agentcpu/anchors/claude_code/tasks/requests_5414/setup.sh

# Replays are offline: a missing model file must fail at start, not stall.
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# Trace and perf jobs run as an unprivileged uid.
RUN chmod -R a+rwX $AGENTCPU /testbed $tmpdir/uv_python && chmod a+rx $tmpdir/libagent_markers.so
