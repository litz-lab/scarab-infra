# Scarab-infra
Dockerfiles of docker images running data center workloads and run Scarab simulation

> This document still describes the older artifact flow and descriptor layout under `isca2024_udp/`. For the current unified CLI, use `./sci` as documented in the top-level [README](../README.md).

## Docker setup
Install Docker based on the instructions from official [docker docs](https://docs.docker.com/engine/install/).

## Requirements
To run scarab in a docker container, the host machine should use non-root user and the user has a proper GitHub setup to access https://github.com/Litz-Lab/scarab.
To run docker as a non-root user, run: "sudo chmod 666 /var/run/docker.sock" (ref: https://stackoverflow.com/questions/48957195/how-to-fix-docker-got-permission-denied-issue)
To generate a new SSH key and it to the machine's SSH agent: https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent?platform=linux
Add a new SSH key to your GitHub account: https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account?platform=linux

## Clean up any cached docker container/image/builds
```
docker stop isca2024_udp_$USER
docker rm isca2024_udp_$USER
docker rmi isca2024_udp:latest
docker system prune
```

## 1. Build a Docker image and run a container of a built image where all the traces/simpoints are available and ready to run Scarab
```
./run.sh -o <path_to_mount_docker_home> -b 2
```
For example,
```
./run.sh -o /home/$USER/isca2024_home -b 2
```

## 2. Run Scarab simulations by using a descriptor file
The file descriptor for all the simulation scenarios for Artifact Evaluation is already in isca2024_udp/isca.json and isca2024_udp/isca.pt.json
The simulations using memtrace or pt traces should be launched separately with different simulation mode (4 : memtrace, 5 : PT trace). Still the two executions can be parallelized (by launching on different terminals).
Run the following command with <experiment> name for -e.
```
./run.sh -o /home/$USER/isca2024_home -s 4 -e isca
./run.sh -o /home/$USER/isca2024_home -s 5 -e isca.pt
```

The script will launch Scarab simulations in background until it runs all the different scenarios x workloads. It will take approximately 24 hours. You can check if all the simulations are over by checking if there is any 'scarab' process running (UNIX 'top' command).

## 3. Plot the results
If you don't see any 'scarab' runs, you are now ready to generate the figures.
Run a given script to generate Figure 13, 14, 15, 16, 17 in the paper (Figure13.pdf, Figure14.pdf, ...).

```
cd ./isca2024_udp/plot
./plot_figures.sh /home/$USER/isca2024_home/
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
