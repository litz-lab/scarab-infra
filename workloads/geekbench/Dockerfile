# syntax = edrevo/dockerfile-plus
FROM ubuntu:focal

INCLUDE+ ./common/Dockerfile.common

USER root

# Download Geekbench 6.3.0
RUN wget https://cdn.geekbench.com/Geekbench-6.3.0-Linux.tar.gz -O /tmp/Geekbench.tar.gz && \
    tar -xvf /tmp/Geekbench.tar.gz -C /opt/ && \
    rm /tmp/Geekbench.tar.gz

# Geekbench6 Pro requires a license key and the its corresponding user email
# For Litz Lab users, get the info from the traces dir and then copy them into current dir
COPY ./LICENSE /opt/Geekbench-6.3.0-Linux/LICENSE
COPY ./EMAIL /opt/Geekbench-6.3.0-Linux/EMAIL

# Set the working directory to the Geekbench folder
WORKDIR /opt/Geekbench-6.3.0-Linux

# Unlock Geekbench6 using and the corresponding email by the license key
RUN chmod +x ./geekbench6 && \
    ./geekbench6 --unlock ./EMAIL ./LICENSE

# Start your application
CMD ["/bin/bash"]