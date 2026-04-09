# syntax = edrevo/dockerfile-plus

FROM ubuntu:focal

INCLUDE+ ./common/Dockerfile.common

USER root
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-pip \
    python3-venv && \
    rm -rf /var/lib/apt/lists/*

WORKDIR $tmpdir

RUN mkdir -p /tmp_home/langchain
COPY workloads/langchain/requirements.txt /tmp_home/langchain/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp_home/langchain/requirements.txt

COPY workloads/langchain/run_langchain.py /tmp_home/langchain/run_langchain.py
COPY workloads/langchain/workload_root_entrypoint.sh /tmp_home/langchain/workload_root_entrypoint.sh
COPY workloads/langchain/workload_user_entrypoint.sh /tmp_home/langchain/workload_user_entrypoint.sh

CMD ["/bin/bash"]
