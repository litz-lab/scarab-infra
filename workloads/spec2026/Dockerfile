# syntax = edrevo/dockerfile-plus

# We follow SPEC CPU2026's own config/Example-gcc-linux-x86.cfg, which SPEC tested with gcc 15.2 on Ubuntu 22.04.
# For reference, here's what each of SPEC CPU2026's own config/Example-*.cfg files reports testing against ("SPEC tested this config file with:"):
#
#   Config                | Compiler            | OS
#   ----------------------|---------------------|----------------
#   gcc-linux-x86         | gcc 15.2            | Ubuntu 22.04
#   gcc-linux-aarch64     | gcc 15.2            | Fedora 42
#   gcc-linux-riscv64     | gcc 15.2            | Ubuntu 24.04
#   gcc-macos             | gcc 15.2            | macOS Sequoia
#   gcc-linux-ppc64le     | GCC 14.2 (IBM AT)   | Ubuntu 22.04
#   llvm-linux-x86        | LLVM 20.1           | Ubuntu 24.04
#   aocc-linux-x86        | AOCC 5.1.0          | Ubuntu 24.04
#   intel-compiler-linux  | Intel oneAPI 2025.2 | Ubuntu 22.04
#   monitors              | LLVM 21.1.8         | Ubuntu 22.04

# choose a ubuntu version
FROM ubuntu:jammy

INCLUDE+ ./common/Dockerfile.common

USER root
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    gnupg \
    lsb-release \
    software-properties-common

# Jammy's native repos only go up to gcc-12, so pull gcc-15 from ubuntu-toolchain-r/test PPA.
RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y gcc-15 g++-15 gfortran-15 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-15 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-15 100 && \
    update-alternatives --install /usr/bin/gfortran gfortran /usr/bin/gfortran-15 100

WORKDIR $tmpdir

# The SPEC CPU2026 installer tarball is not checked in.
# If the build errors with "file not found in build context" for spec2026.tar.gz,
# place your licensed tarball at workloads/spec2026/spec2026.tar.gz or pass
# --build-arg SPEC2026_TARBALL=<relative path inside build context>.
# If you're not Litz-lab member, please match the location described in workload_user_entrypoint.sh with your spec2026 directory. 
ARG SPEC2026_TARBALL=workloads/spec2026/spec2026.tar.gz

# ADD should be used here (not COPY + RUN tar -xzf + rm) to avoid a duplicate tarball layer
RUN echo "extracting spec2026.tar.gz to $tmpdir/application"
ADD ${SPEC2026_TARBALL} $tmpdir/application/
# Compilation should be done during ./sci --build-image
RUN echo "building benchmarks"
# Build all rate benchmarks in PARALLEL.
# runcpu takes no cross-invocation lock, so drive one benchmark per xargs worker instead of building them serially.
# Start the heavy, slow-building benchmarks FIRST so they overlap the many quick builds instead of extending the tail.
ARG BUILD_WORKERS=8
RUN /bin/bash -c "cd $tmpdir/application/cpu2026 && source shrc && \
    { echo 766.femflow_r; echo 723.llvm_r; echo 735.gem5_r; echo 753.ns3_r;  ls -d benchspec/CPU/7[0-9]*_r | xargs -n1 basename | grep -vxF -e 723.llvm_r -e 735.gem5_r -e 753.ns3_r -e 766.femflow_r; } | \
    xargs -P ${BUILD_WORKERS} -I{} runcpu --config=memtrace.cfg --action=build {}"
RUN /bin/bash -c "cd $tmpdir/application/cpu2026 && source shrc && runcpu --config=memtrace.cfg --action=runsetup intrate fprate"
RUN echo "done"

# Start your application
CMD ["/bin/bash"]
