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
# workload_*_entrypoint.sh is bind-mounted from the host checkout and published
# into /usr/local/bin by root_entrypoint.sh; see common/Dockerfile.common.

WORKDIR $tmpdir
RUN cd $tmpdir && git clone https://github.com/akopytov/sysbench.git

# Start your application
CMD ["/bin/bash"]
