# syntax = edrevo/dockerfile-plus
FROM ubuntu:20.04

INCLUDE+ ./common/Dockerfile.common
# TODO: Install required packages
USER root
# Switch to the dcuser user
RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y \
 autoconf \
 automake \
 autotools-dev \
 curl \
 libmpc-dev \
 libmpfr-dev \
 libgmp-dev \
 libusb-1.0-0-dev \
 gawk \
 build-essential \
 bison \
 flex \
 texinfo \
 gperf \
 libtool \
 patchutils \
 bc \
 device-tree-compiler \
 pkg-config \
 libexpat-dev \
 libfl-dev \
 libboost-all-dev \
&& apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR $tmpdir

RUN cd $tmpdir && git clone https://github.com/sbeamer/gapbs.git \
&& cd gapbs \
&& make \
&& make bench-graphs
RUN rm -rf /tmp_home/gapbs/benchmark/graphs/raw
# Start your application
CMD ["/bin/bash"]