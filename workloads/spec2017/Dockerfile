# syntax = edrevo/dockerfile-plus

# choose a ubuntu version
FROM ubuntu:focal

INCLUDE+ ./common/Dockerfile.common

# Copy spec2017
COPY ./workloads/spec2017/cpu2017-1_0_5.iso $tmpdir

USER root
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    gfortran \
    lsb-release \
    software-properties-common


WORKDIR $tmpdir

RUN cd $tmpdir && wget https://apt.llvm.org/llvm.sh
RUN chmod u+x llvm.sh
RUN sudo ./llvm.sh 16

COPY ./workloads/spec2017/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY ./workloads/spec2017/compile-538-clang.sh $tmpdir
COPY ./workloads/spec2017/memtrace.cfg $tmpdir
COPY ./workloads/spec2017/install.sh $tmpdir

# Start your application
CMD ["/bin/bash"]
