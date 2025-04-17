# Scarab-infra

scarab-infra is a set of tools that automate the execution of Scarab simulations. It utilizes [Docker](https://www.docker.com/) and [Slurm](https://slurm.schedmd.com/documentation.html) to effectively simulate applications according to the [SimPoint](https://cseweb.ucsd.edu/~calder/simpoint/) methodology. Furthermore, scarab-infra provides tools to analyze generated simulation statistics and to obtain simpoints and execution traces from binary applications.

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
4. Activate the virtual environment.
```
conda activate scarabinfra
```
5. Add the SSH key of the machine(s) running the Docker container to your GitHub account ([link](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent?platform=linux)).
6. Place simpointed instruction traces into $trace_home. scarab-infra offers prepackaged traces that can be downloaded as follows:
```
cd /home/$USER/traces
gdown https://drive.google.com/uc?id=1tfKL7wYK1mUqpCH8yPaPVvxk2UIAJrOX
tar -xzvf simpoint_traces.tar.gz
```
5. Optional: Install [Slurm](docs/slurm_install_guide.md)

## Set up the environment (Docker image)
### Alternative 1. Download a pre-built Docker image where GitHash refers to the short git hash of the current scarab-infra commit.
```
export GIT_HASH=$(git rev-parse --short HEAD)
docker pull ghcr.io/litz-lab/scarab-infra/allbench_traces:$GIT_HASH
docker tag ghcr.io/litz-lab/scarab-infra/allbench_traces:$GIT_HASH allbench_traces:$GIT_HASH
```
Make sure the docker image exists on local by running
```
docker images
```
### Alternative 2. Build your own Docker image
```
./run.sh -b $IMAGE_NAME
```

### List available workload group name
```
./run.sh --list
```

## Run a Scarab experiment

1. Setup a new experiment descriptor
```
cp json/exp.json json/your_experiment.json
```
2. Edit your_experiment.json to describe your experiment. Please refer to json/exp.json for the describtion. To set the 'workload manager' = slurm, Slurm should be installed.
3. Run all experiments
```
./run.sh --simulation your_experiment
```
The script will launch all Scarab simulations in parallel (one process per simpoint). You can check if the experiment is complete if there are no active scarab process running (UNIX 'top' command).

## Check the info/status of the experiment
```
./run.sh --status your_experiment
```

## Kill the experiment
```
./run.sh --kill your_experiment
```

## Run an interactive shell of a docker container for the purpose of debugging/development
```
./run.sh --run your_experiment
```

## Modify the source code and rebuild scarab
A user can update scarab and rebuild for further simulation. Scarab can be updated either 'inside' or 'outside' the container. To exploit already-set simulation/building environment, scarab build itself should be done 'inside' the container.
### Alternative 1. Start an interactive container with pre-installed Scarab environment
```
./run.sh --run your_experiment
cd /scarab/src
make clean && make
```
### Alternative 2. Work outside of the container
When you modify scarab outside the container, cd to the path you provided for 'scarab_path' in your_experiment.json, and modify it.
```
cd /home/$USER/src/scarab
```

## Debug scarab
Start an interactive container with pre-installed Scarab environment
```
./run.sh --run your_experiment
cd /scarab/src
```
If you want to attach gdb to the debug mode binary,
```
make dbg
```
Then, go to the simulation directory where you want to debug such as
```
cd ~/simulations/<exp_name>/baseline/<workload>/<simpoint>
```
Create a debug directory and copy the original PARAMS.out file as a new PARAMS.in, then cut the lines following after `--- Cut out everything below to use this file as PARAMS.in ---`
```
mkdir debug && cd debug
cp ../PARAMS.out ./PARAMS.out
```
Now, you can attach gdb with the same scarab parameters where you want to debug.
```
gdb /scarab/src/scarab
```

## Clean up any cached docker container/image/builds
```
./run.sh -c -w allbench
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
