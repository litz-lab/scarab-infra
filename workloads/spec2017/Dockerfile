# syntax = edrevo/dockerfile-plus

# choose a ubuntu version
FROM ubuntu:focal

INCLUDE+ ./common/Dockerfile.common

# Copy spec2017

USER root
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    gfortran \
    lsb-release \
    software-properties-common

WORKDIR $tmpdir

RUN cd $tmpdir && wget https://apt.llvm.org/llvm.sh
RUN chmod u+x llvm.sh
RUN sudo ./llvm.sh 16

COPY ./workloads/spec2017/compile-538-clang.sh $tmpdir
COPY ./workloads/spec2017/memtrace.cfg $tmpdir

COPY ./workloads/spec2017/spec2017_install $tmpdir/cpu2017
RUN cp $tmpdir/memtrace.cfg $tmpdir/cpu2017/config/memtrace.cfg
RUN cp $tmpdir/compile-538-clang.sh $tmpdir/cpu2017/benchspec/CPU

# Setup SPEC2017 refrate run directories
WORKDIR $tmpdir/cpu2017
RUN ./bin/runcpu --config=memtrace --action=setup --size=refrate 500.perlbench_r 502.gcc_r 505.mcf_r 520.omnetpp_r 523.xalancbmk_r 525.x264_r 531.deepsjeng_r 541.leela_r 548.exchange2_r 557.xz_r
RUN ./bin/runcpu --config=memtrace --action=setup --size=refrate 503.bwaves_r 507.cactuBSSN_r 508.namd_r 510.parest_r 511.povray_r 519.lbm_r 521.wrf_r 526.blender_r 527.cam4_r 538.imagick_r 544.nab_r 549.fotonik3d_r 554.roms_r

# Start your application
CMD ["/bin/bash"]
