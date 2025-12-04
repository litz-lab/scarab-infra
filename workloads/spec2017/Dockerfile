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

RUN cd $tmpdir && wget https://apt.llvm.org/llvm.sh
RUN chmod u+x llvm.sh
RUN sudo ./llvm.sh 16
RUN echo "copying spec2017.tar.gz to $tmpdir/application"
RUN mkdir -p $tmpdir/application
COPY ./workloads/spec2017/spec2017.tar.gz $tmpdir/application/
RUN echo "extracting spec2017.tar.gz to $tmpdir/application"
RUN tar -xzf $tmpdir/application/spec2017.tar.gz -C $tmpdir/application && rm $tmpdir/application/spec2017.tar.gz
RUN echo "done"

# Start your application
CMD ["/bin/bash"]
