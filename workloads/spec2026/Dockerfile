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
    DEBIAN_FRONTEND=noninteractive apt-get install -y gcc-15 g++-15 gfortran-15

WORKDIR $tmpdir

RUN mkdir -p $tmpdir/application/

# Start your application
CMD ["/bin/bash"]
