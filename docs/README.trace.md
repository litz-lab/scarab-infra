# Scarab-infra

scarab-infra is a set of tools that collect DynamoRIO traces based on SimPoint methodology.
This document describes the steps to collect traces of a new application and add them to scarab-infra workload DB.

## Requirements
1. Install Docker [docker docs](https://docs.docker.com/engine/install/).
2. Configure Docker to run as non-root user ([ref](https://stackoverflow.com/questions/48957195/how-to-fix-docker-got-permission-denied-issue)):
```
sudo chmod 666 /var/run/docker.sock
```
3. Install pip libraries by using conda (create virtual environment `scarabinfra`)
```
conda env create --file quickstart_env.yaml
```
Optional: if you want to remove the existing `scarabinfra` and re-create it, run the following first.
```
conda env remove -n scarabinfra
```
4. Activate the virtual environment.
```
conda activate scarabinfra
```
5. Add the SSH key of the machine(s) running the Docker container to your GitHub account ([link](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent?platform=linux)).
6. Optional: Install [Slurm](docs/slurm_install_guide.md)
7. Prepare new applications/benchmarks where you want to collect traces. We use [DCPerf from Meta](https://github.com/facebookresearch/DCPerf) as a benchmark suite here as an example.

## Limitations
DynamoRIO does not always work for non-C/C++ applications. This README assumes C/C++ benchmarks to be added to this repository.

## Write a Dockerfile to set up the environment (Docker image)
### 1. Carefully read the README of the new applications and find all the prerequisites to install.
Refer to `example/Dockerfile` to see a skeleton of Dockerfile.
[DCPerf](https://github.com/facebookresearch/DCPerf?tab=readme-ov-file#install-prerequisites) provides a detailed README for the requirements.
Based on the description, DCPerf requires one of CentOS 8, CentOS 9, and Ubuntu 22.04 as the operating system version to run the benchmarks. We pick Ubuntu 22.04.
The README also provides the following commands to install the prerequisites on Ubuntu 22.04. Also, find out all required libraries to be installed.
```
sudo apt update
sudo apt install -y python3-pip git
sudo pip3 install click pyyaml tabulate pandas
```
Based on the information, you can create a dockerfile under a new directory called 'workloads/dcperf'.
```
mkdir workloads/dcperf
```
Then, create a Dockerfile inside the directory.
`workloads/dcperf/Dockerfile` should include the following lines.
```
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
```
Here is the line-by-line description.
```
# syntax = edrevo/dockerfile-plus
FROM ubuntu:22.04

INCLUDE+ ./common/Dockerfile.common
```
The first line `# syntax = edrevo/dockerfile-plus` is needed to include the common Dockerfile that includes all the prerequisites for DynamoRIO, SimPoint, and Scarab. The second line `FROM ubuntu:22.04` represents that the docker image will be built for Ubuntu 22.04 system. The line `INCLUDE+ ./common/Dockerfile.common` should always appear after the first two lines (syntax and the OS version).
```
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
```
To install all the prerequisites with sudo, set USER root `USER root`. To install the Ubuntu packages, use the command `RUN DEBIAN_FRONTEND=noninteractive apt-get install -y ..`. python3-pip and git have already installed in `Dockerfile.common`, we omit to add them. To run a UNIX command, prefix `RUN` in the Dockerfile. `$tmpdir` is a temporary directory already created during `Dockerfile.common` build. All the installation should be done in this directory. Download or clone the application and checkout a specific githash for a stable environment for exec-driven simulation. In this example, `RUN git clone https://github.com/facebookresearch/DCPerf.git && cd ./DCPerf && git checkout 4dc3b5e8836796fb7d80316f43a1147d052dc2e7`
```
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
```
DCPerf consists of six benchmark subsuites, but we pick C++ benchmarks `FeedSim`, `TaoBench`, `DjangoBench`, `VideoTranscodeBench` and `wdlbench` to make them work smoothly with DynamoRIO. In DCPerf, each benchmarks requires different commands to be installed.
For example, to install `FeedSim` benchmark, `./benchpress_cli.py install feedsim_autoscale` is used. We recommend you not to include the commands (being commented out) in the example and build the image first. This is usually required because sometimes the prerequisites described in the benchmark README are not sufficient and require additional packages to be installed for a docker image to be successfully built. Just keep them in the comments and now it is time to build the image. If the workload itself has a bug, remove it as possible.

## Set up the environment (Docker image)
### Alternative 1. Download a pre-built Docker image (only available for trace-based simulations)
```
docker pull ghcr.io/litz-lab/scarab-infra/$WORKLOAD_GROUPNAME:<GitHash>
```
### Alternative 2. Build your own Docker image
```
./run.sh -b $WORKLOAD_GROUPNAME
```

## Run tracing based on SimPoint methodology

1. Setup a new trace descriptor
```
cp json/trace.json json/your_trace.json
```
2. Edit your_trace.json to describe your trace scenarios. We strongly recommend starting with a single trace scenario instead of multiple trace configurations, as it may require a significant amount of memory, even for a single scenario, which can cause an out-of-memory issue depending on the system. Please refer to json/trace.json for the describtion. To set the 'workload manager' = slurm, Slurm should be installed.
3. Run tracing with the JSON desciptor.

```
./run.sh --trace your_trace
```
The script will launch all tracings in parallel. The collected simpoint information and traces will be copied to the destination `traces_dir` described in the json. The information for exec-driven/memtrace simulation will be automatically added to the workload DB and suite DB.

## Check the info/status of the tracing
```
./run.sh --status your_trace
```

## Kill the tracing
```
./run.sh --kill your_trace
```

## Run an interactive shell of a docker container for the purpose of debugging or manually collecting traces
```
./run.sh --run your_trace
```

# Publications

```
@inproceedings{oh2024udp,
  author = {Oh, Surim and Xu, Mingsheng and Khan, Tanvir Ahmed and Kasikci, Baris and Litz, Heiner},
  title = {UDP: Utility-Driven Fetch Directed Instruction Prefetching},
  booktitle = {Proceedings of the 51st International Symposium on Computer Architecture (ISCA)},
  series = {ISCA 2024},
  year = {2024},
  month = jun,
}
```
