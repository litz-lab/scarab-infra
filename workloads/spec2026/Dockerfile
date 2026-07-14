# syntax = edrevo/dockerfile-plus

# choose a ubuntu version
FROM ubuntu:focal

INCLUDE+ ./common/Dockerfile.common

USER root
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    gfortran \
    lsb-release \
    software-properties-common

WORKDIR $tmpdir

# The SPEC CPU2026 installer tarball is not checked in.
# If the build errors with "file not found in build context" for spec2026.tar.gz,
# place your licensed tarball at workloads/spec2026/spec2026.tar.gz or pass
# --build-arg SPEC2026_TARBALL=<relative path inside build context>.
# If you're not Litz-lab member, please match the location described in workload_user_entrypoint.sh with your spec2026 directory. 
ARG SPEC2026_TARBALL=workloads/spec2026/spec2026.tar.gz

RUN cd $tmpdir && wget https://apt.llvm.org/llvm.sh
RUN chmod u+x llvm.sh
RUN sudo ./llvm.sh 16
# ADD should be used here (not COPY + RUN tar -xzf + rm) to avoid a duplicate tarball layer
RUN echo "extracting spec2026.tar.gz to $tmpdir/application"
ADD ${SPEC2026_TARBALL} $tmpdir/application/
# Compilation should be done during ./sci --build-image
RUN echo "building benchmarks"
RUN /bin/bash -c "cd $tmpdir/application/cpu2026 && source shrc && runcpu --config=memtrace.cfg --action=build all"
RUN echo "done"

# Start your application
CMD ["/bin/bash"]
