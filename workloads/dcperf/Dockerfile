# syntax = edrevo/dockerfile-plus
FROM ubuntu:22.04

INCLUDE+ ./common/Dockerfile.common

USER root
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    dmidecode \
    lshw
RUN pip install click pyyaml tabulate pandas

WORKDIR $tmpdir

run cd $tmpdir && git clone https://github.com/facebookresearch/DCPerf.git
run cd DCPerf && ./benchpress_cli.py install feedsim_autoscale
run cd DCPerf && ./benchpress_cli.py install tao_bench_64g
run cd DCPerf && ./benchpress_cli.py install django_workload_default
run cd DCPerf && ./benchpress_cli.py install video_transcode_bench_svt

# Start your application
CMD ["/bin/bash"]
