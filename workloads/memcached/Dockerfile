# syntax = edrevo/dockerfile-plus

FROM ubuntu:focal

INCLUDE+ ./common/Dockerfile.common

# TODO: Install required packages
USER root
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libevent-dev \
    libzmq3-dev
RUN apt-get build-dep memcached

WORKDIR $tmpdir
RUN cd $tmpdir && git clone https://github.com/shaygalon/memcache-perf.git
RUN cd memcache-perf && make

WORKDIR /home/$username
USER $username

# Start your application
CMD ["/bin/bash"]
