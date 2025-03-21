# syntax = edrevo/dockerfile-plus
FROM ubuntu:focal

INCLUDE+ ./common/Dockerfile.common

USER root

# Download Geekbench 6.3.0
RUN wget https://cdn.geekbench.com/Geekbench-6.3.0-Linux.tar.gz -O /tmp/Geekbench.tar.gz && \
    tar -xvf /tmp/Geekbench.tar.gz -C $tmpdir/ && \
    rm /tmp/Geekbench.tar.gz

# Install the required packages
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y jq

# Using Geekbench6 Pro requires a license key and the its user email
# For Litz Lab users, get the license from the traces dir and then copy them into './workloads/geekbench/'
COPY ./workloads/geekbench/license.json $tmpdir/Geekbench-6.3.0-Linux/license.json

# Unlock Geekbench6 using the license json file
RUN cd $tmpdir/Geekbench-6.3.0-Linux && \
    chmod +x ./geekbench6 && \
    ./geekbench6 --unlock "$(jq -r '.license_user' ./license.json)" "$(jq -r '.license_key' ./license.json)"

# Start your application
CMD ["/bin/bash"]