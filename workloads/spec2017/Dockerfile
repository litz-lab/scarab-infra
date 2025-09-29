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
    software-properties-common \
    libarchive-tools

WORKDIR $tmpdir

RUN cd $tmpdir && wget https://apt.llvm.org/llvm.sh
RUN chmod u+x llvm.sh
RUN sudo ./llvm.sh 16

COPY ./workloads/spec2017/compile-538-clang.sh $tmpdir
COPY ./workloads/spec2017/memtrace.cfg $tmpdir

RUN mkdir -p $tmpdir/cpu2017_install
RUN bsdtar -C $tmpdir/cpu2017_install -xf $tmpdir/cpu2017-1_0_5.iso

RUN mkdir -p $tmpdir/cpu2017
RUN cd $tmpdir/cpu2017_install && echo "yes" | ./install.sh -d $tmpdir/cpu2017
RUN cp $tmpdir/memtrace.cfg $tmpdir/cpu2017/config/memtrace.cfg
RUN cp $tmpdir/compile-538-clang.sh $tmpdir/cpu2017/benchspec/CPU
RUN rm -f $tmpdir/cpu2017-1_0_5.iso && rm -rf $tmpdir/cpu2017_install

RUN bash -lc 'cd /tmp_home/cpu2017 && source shrc && \
  runcpu --config=memtrace --noreportable --size=train --action=runsetup intrate fprate'

RUN bash -lc 'find /tmp_home/cpu2017/benchspec/CPU -type d \
  -path "*/run/run_base_train_memtrace-m64.*" -exec chmod -R a+w {} + || true'

# Start your application
CMD ["/bin/bash"]
