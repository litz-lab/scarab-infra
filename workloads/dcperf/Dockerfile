# syntax = edrevo/dockerfile-plus
FROM ubuntu:22.04

INCLUDE+ ./common/Dockerfile.common

USER root
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    dmidecode \
    lshw \
    curl \
    numactl \
    lsof \
    netcat \
    build-essential \
    socat \
    pciutils

RUN pip install click pyyaml tabulate pandas

WORKDIR $tmpdir
RUN git clone https://github.com/facebookresearch/DCPerf.git && cd ./DCPerf && git checkout 4dc3b5e8836796fb7d80316f43a1147d052dc2e7

WORKDIR $tmpdir/DCPerf
RUN python3 ./benchpress_cli.py install feedsim_autoscale
# Remove three lines where there is a bug as of Feb 16 2025
RUN sed -i '173d' $tmpdir/DCPerf/packages/tao_bench/install_tao_bench_x86_64.sh
RUN sed -i '173d' $tmpdir/DCPerf/packages/tao_bench/install_tao_bench_x86_64.sh
RUN sed -i '173d' $tmpdir/DCPerf/packages/tao_bench/install_tao_bench_x86_64.sh
RUN python3 ./benchpress_cli.py install tao_bench_64g
RUN python3 ./benchpress_cli.py install django_workload_default
RUN python3 ./benchpress_cli.py install video_transcode_bench_svt
# Exclude fbthrift benchmark because there is a bug as of Feb 16 2025
RUN sed -i '161d' $tmpdir/DCPerf/packages/wdl_bench/install_wdl_bench.sh
RUN sed -i '166d' $tmpdir/DCPerf/packages/wdl_bench/install_wdl_bench.sh
RUN sed -i '166d' $tmpdir/DCPerf/packages/wdl_bench/install_wdl_bench.sh
RUN sed -i '170d' $tmpdir/DCPerf/packages/wdl_bench/install_wdl_bench.sh
RUN sed -i '170d' $tmpdir/DCPerf/packages/wdl_bench/install_wdl_bench.sh
RUN sed -i '170d' $tmpdir/DCPerf/packages/wdl_bench/install_wdl_bench.sh
RUN python3 ./benchpress_cli.py -b wdl install folly_single_core

WORKDIR $tmpdir
# Start your application
CMD ["/bin/bash"]
