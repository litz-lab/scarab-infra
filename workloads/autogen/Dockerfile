# syntax = edrevo/dockerfile-plus

FROM ubuntu:jammy

INCLUDE+ ./common/Dockerfile.common

USER root
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-pip \
    python3-venv && \
    rm -rf /var/lib/apt/lists/*

WORKDIR $tmpdir

RUN mkdir -p /tmp_home/autogen
COPY workloads/autogen/requirements.txt /tmp_home/autogen/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp_home/autogen/requirements.txt

COPY workloads/autogen/run_autogen.py /tmp_home/autogen/run_autogen.py
COPY workloads/autogen/workload_root_entrypoint.sh /tmp_home/autogen/workload_root_entrypoint.sh
COPY workloads/autogen/workload_user_entrypoint.sh /tmp_home/autogen/workload_user_entrypoint.sh

CMD ["/bin/bash"]
