# syntax = edrevo/dockerfile-plus

FROM ubuntu:focal

INCLUDE+ ./common/Dockerfile.common

# TODO: Install required packages
USER root
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    make \
    automake \
    libtool \
    pkg-config \
    libaio-dev \
    libmysqlclient-dev \
    libssl-dev \
    libpq-dev \
    mysql-server \
    postgresql

RUN chmod -R 755 /var/run/mysqld
COPY ./workloads/sysbench/workload_root_entrypoint.sh /usr/local/bin/workload_root_entrypoint.sh

WORKDIR $tmpdir
RUN cd $tmpdir && git clone https://github.com/akopytov/sysbench.git

# Start your application
CMD ["/bin/bash"]
