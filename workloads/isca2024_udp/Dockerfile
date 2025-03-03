# syntax = edrevo/dockerfile-plus

# choose a ubuntu version
FROM ubuntu:focal

INCLUDE+ ./common/Dockerfile.common

USER root
RUN pip install gdown
COPY ./workloads/isca2024_udp/workload_root_entrypoint.sh /usr/local/bin/workload_root_entrypoint.sh

WORKDIR /simpoint_traces
RUN gdown https://drive.google.com/uc?id=1tfKL7wYK1mUqpCH8yPaPVvxk2UIAJrOX && tar -xzvf simpoint_traces.tar.gz

WORKDIR $tmpdir
COPY ./workloads/isca2024_udp/plot $tmpdir/plot
COPY ./workloads/isca2024_udp/isca.json $tmpdir/isca.json
COPY ./workloads/isca2024_udp/isca.pt.json $tmpdir/isca.pt.json

# Start your application
CMD ["/bin/bash"]
