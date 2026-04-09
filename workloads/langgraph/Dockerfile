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

RUN mkdir -p /tmp_home/langgraph
COPY workloads/langgraph/requirements.txt /tmp_home/langgraph/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp_home/langgraph/requirements.txt

COPY workloads/langgraph/run_langgraph.py /tmp_home/langgraph/run_langgraph.py
COPY workloads/langgraph/workload_root_entrypoint.sh /tmp_home/langgraph/workload_root_entrypoint.sh
COPY workloads/langgraph/workload_user_entrypoint.sh /tmp_home/langgraph/workload_user_entrypoint.sh

CMD ["/bin/bash"]
