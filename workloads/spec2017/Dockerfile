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

# The SPEC CPU2017 installer tarball is not checked in.
# If the build errors with "file not found in build context" for spec2017.tar.gz,
# place your licensed tarball at workloads/spec2017/spec2017.tar.gz or pass
# --build-arg SPEC2017_TARBALL=<relative path inside build context>.
# If you're not Litz-lab member, please match the location described in workload_user_entrypoint.sh with your spec2017 directory. 
ARG SPEC2017_TARBALL=workloads/spec2017/spec2017.tar.gz

RUN cd $tmpdir && wget https://apt.llvm.org/llvm.sh
RUN chmod u+x llvm.sh
RUN sudo ./llvm.sh 16
RUN echo "copying spec2017.tar.gz to $tmpdir/application"
RUN mkdir -p $tmpdir/application
COPY ${SPEC2017_TARBALL} $tmpdir/application/spec2017.tar.gz
RUN echo "extracting spec2017.tar.gz to $tmpdir/application"
RUN tar -xzf $tmpdir/application/spec2017.tar.gz -C $tmpdir/application && rm $tmpdir/application/spec2017.tar.gz
RUN echo "done"

# Start your application
CMD ["/bin/bash"]
