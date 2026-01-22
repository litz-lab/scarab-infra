# syntax = edrevo/dockerfile-plus

# choose a ubuntu version
FROM ubuntu:focal

INCLUDE+ ./common/Dockerfile.common

USER root
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    gfortran \
    lsb-release \
    software-properties-common

# SPEC CPU2006 Build Configuration
# 
# Build command:
#   runspec --config=Example-linux64-amd64-gcc43+ --action=build --size=ref all
#
# Build error fixes:
#   - 400.perlbench: Add -std=gnu89
#   - 416.gamess:    Add -std=legacy
#   - 447.dealII:    Add -std=c++03
#   - 450.soplex:    Add -std=c++03
WORKDIR $tmpdir
RUN mkdir -p $tmpdir/application

# Start your application
CMD ["/bin/bash"]
