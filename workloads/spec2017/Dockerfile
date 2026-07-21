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

# The SPEC CPU2017 binaries are NOT baked into the image. They are bind-mounted
# at runtime from the host onto /tmp_home/application (set the descriptor's
# "application_dir" to a directory that contains the SPEC install as cpu2017/, so
# the entrypoint's `cd /tmp_home/application/cpu2017` resolves). Baking the
# licensed tarball is unnecessary since the mount shadows it anyway; we only
# create the mount point here.
RUN cd $tmpdir && wget https://apt.llvm.org/llvm.sh
RUN chmod u+x llvm.sh
RUN sudo ./llvm.sh 16
RUN mkdir -p $tmpdir/application
RUN echo "done"

# Start your application
CMD ["/bin/bash"]
