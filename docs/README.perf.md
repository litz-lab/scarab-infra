# Collecting Microarch Metrics for Performance Analysis
This README describes all the steps to collect system or microarchitectural metrics of an arbitrary application and to collect memtraces by using DynamoRIO.
By following the steps, a user will be able to do top-down analysis with Intel Perf tool to identify performance bottlenecks (frontend/backend/bad speculation/retiring), and extract the region of interest of execution traces with DynamoRIO and SimPoint methodology. The steps described in this README do not start with the complete diff of adding an application, instead, it starts with an incomplete one and revise it step by step.

## Requirements
Prepare new applications/benchmarks where you want to measure performance. We use [DCPerf from Meta](https://github.com/facebookresearch/DCPerf) here as an example.
Make sure the PMU counters readable on your host system (outside of docker container).
```
sudo su
echo -1 > /proc/sys/kernel/perf_event_paranoid
```

## Prepare the Dockerfile to build a docker image as described in [README.trace.md](docs/README.trace.md)
Make sure the perf is installed.
When you run an interactive shell, scarab-infra will install perf and pmu-tools, but you can manually build an image by making sure to include the following lines inside the dockerfile.

We use `pmu-tools` for an effective top-down analysis where it internally runs perf.
```
USER root
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    linux-tools-common \
    linux-tools-generic \
    linux-tools-`uname -r`
```
and
```
RUN alias perf=$(find /usr/lib/linux-tools/*/perf | head -1)
```
to address kernel mismatch for Perf.
Also,
```
RUN cd $tmpdir && git clone https://github.com/andikleen/pmu-tools.git
```
to install the pmu-tools.

## Run an interactive shell of a docker container where you can run Perf
After building a docker image, run an interactive shell of a docker container running on the image by using the perf JSON descriptor.
```
./run.sh --run perf
```

## Run a workload
### Binary command
A binary command will be eventually used to collect traces of the region of interest. Playing around with the binary command is completely application-dependent.

Note that the following steps are only for `feedsim` of `DCPerf`. You should figure out a way to run your new benchmark by playing around with different commands with different options to reproduce the real-world application performance.

Running a binary command does not necessarily require root user, but DCPerf requires root user to have the benchmark running. You can specify the username in the JSON despcriotor.

If your new benchmark is not a server-client application, you need a single the binary command. If it is a server-client application, the binary command should be the one to run a server because we are interested in the performance of the server on a server-client application.

Based on [`wdlbench` README](https://github.com/facebookresearch/DCPerf/blob/main/packages/wdl_bench/README.md), we have installed the benchmark suite. The command `./benchpress_cli.py -b wdl run folly_individual -i '{"name": "fibers_benchmark"}` is used to run an application. To make the command work with DynamoRIO and Perf, try to extract the binary instruction that is executed last. By hacking the scripts, the binary command to execute `fibers_benchmark` of suite `dcperf` and subsuite `wdlbench` is `$tmpdir/DCPerf/benchmarks/wdl_bench/fibers_benchmark`.

### Run with Perf
To simply get the number of cycles and instructions of the application, run
```
perf stat -e cpu-cycles, instruction -a $BINARY_COMMAND
```

To collect performance metrics of a single core application accurately, you need to pin the workload to a single core by using UNIX command `taskset`.
Additionally, to get the top-down analysis, run the pmu-tools by pinning the workload to a specific core.
```
python3 ../pmu-tools/toplev.py --core S0-C0 -l3 -v --no-desc taskset -c 0 ./benchmarks/wdl_bench/fibers_benchmark
```
The benchmark output looks like the following.
```
[...]folly/fibers/test/FibersBenchmark.cpp     relative  time/iter   iters/s
-----------------------------------------
FiberManagerBasicOneAwait                                 420.43ns     2.38M
FiberManagerBasicOneAwaitLogged                             1.19us   838.00K
FiberManagerBasicFiveAwaits                                 1.18us   844.22K
FiberManagerBasicFiveAwaitsLogged                           4.24us   236.07K
FiberManagerGet                                             8.81ns   113.49M
FiberManagerCreateDestroy                                  25.48us    39.25K
FiberManagerAllocateDeallocatePattern                       2.77ms    360.37
FiberManagerAllocateLargeChunk                             23.87ms     41.90
FiberManagerCancelledTimeouts_Single_300                   21.22ms     47.13
FiberManagerCancelledTimeouts_Five                         21.20ms     47.16
FiberManagerCancelledTimeouts_TenThousand                  21.05ms     47.50
```
and the top-down results will follow. This workload shows 30% of frontend boundness, 5.4% of bad speculation, 22.8% of backend boundness, and 42.1% of retiring. You can find the deeper top-down performance results.
```
# 4.8-full-perf on Intel(R) Xeon(R) Gold 6242R CPU @ 3.10GHz [clx/skylake]                                                                                                                                               05:00:27 [250/2648]    
S0-C0    FE               Frontend_Bound                                        % Slots                       30.0    [ 3.3%]   
S0-C0    BAD              Bad_Speculation.Branch_Mispredicts.Other_Mispredicts  % Slots                        0.6  < [ 3.3%]   
S0-C0    BAD              Bad_Speculation.Machine_Clears.Other_Nukes            % Slots                        0.1  < [ 3.3%]   
S0-C0    BAD              Bad_Speculation                                       % Slots                        5.4  < [ 3.3%]   
S0-C0    BE               Backend_Bound                                         % Slots                       22.8    [ 3.3%]   
S0-C0    RET              Retiring                                              % Slots                       42.1  < [ 3.3%]   
S0-C0    FE               Frontend_Bound.Fetch_Latency                          % Slots                       13.7    [ 3.3%]<==    
S0-C0    FE               Frontend_Bound.Fetch_Bandwidth                        % Slots                       16.2  < [ 3.3%]   
S0-C0    BAD              Bad_Speculation.Branch_Mispredicts                    % Slots                        5.2  < [ 3.3%]   
S0-C0    BAD              Bad_Speculation.Machine_Clears                        % Slots                        0.2  < [ 3.3%]   
S0-C0    BE/Mem           Backend_Bound.Memory_Bound                            % Slots                        7.2  < [ 3.3%]   
S0-C0    BE/Core          Backend_Bound.Core_Bound                              % Slots                       15.6    [ 3.3%]   
S0-C0    RET              Retiring.Light_Operations                             % Slots                       33.2  < [ 3.3%]   
S0-C0    RET              Retiring.Heavy_Operations                             % Slots                        8.9  < [ 3.3%]   
S0-C0    RET              Retiring.Light_Operations.Memory_Operations           % Slots                       14.6  < [ 3.3%]   
S0-C0    RET              Retiring.Light_Operations.Fused_Instructions          % Slots                        3.3  < [ 3.3%]   
S0-C0    RET              Retiring.Light_Operations.Non_Fused_Branches          % Slots                        2.9  < [ 3.3%]   
S0-C0    RET              Retiring.Light_Operations.Other_Light_Ops             % Slots                       12.3  < [ 3.3%]   
S0-C0    RET              Retiring.Heavy_Operations.Few_Uops_Instructions       % Slots                        3.5  < [ 3.3%]   
S0-C0    RET              Retiring.Heavy_Operations.Microcode_Sequencer         % Slots                        5.4  < [ 3.3%]   
S0-C0-T0 FE               Frontend_Bound.Fetch_Latency.ICache_Misses            % Clocks                       2.1  < [ 3.3%]   
S0-C0-T0 FE               Frontend_Bound.Fetch_Latency.ITLB_Misses              % Clocks                       3.6  < [ 3.3%]   
S0-C0-T0 FE               Frontend_Bound.Fetch_Latency.Branch_Resteers          % Clocks                       4.3  < [ 3.3%]   
S0-C0-T0 FE               Frontend_Bound.Fetch_Latency.MS_Switches              % Clocks_est                   4.6  < [ 3.3%]   
S0-C0-T0 FE               Frontend_Bound.Fetch_Latency.LCP                      % Clocks                       0.1  < [ 3.3%]   
S0-C0-T0 FE               Frontend_Bound.Fetch_Latency.DSB_Switches             % Clocks                       3.8  < [ 3.3%]   
S0-C0-T0 FE               Frontend_Bound.Fetch_Bandwidth.MITE                   % Slots_est                   28.6  < [ 3.3%]   
S0-C0-T0 FE               Frontend_Bound.Fetch_Bandwidth.DSB                    % Slots_est                    8.8  < [ 3.3%]   
S0-C0-T0 BE/Mem           Backend_Bound.Memory_Bound.L1_Bound                   % Stalls                      11.3  < [ 3.3%]   
S0-C0-T0 BE/Mem           Backend_Bound.Memory_Bound.L2_Bound                   % Stalls                       1.2  < [ 3.3%]   
S0-C0-T0 BE/Mem           Backend_Bound.Memory_Bound.L3_Bound                   % Stalls                       0.8  < [ 3.3%]   
S0-C0-T0 BE/Mem           Backend_Bound.Memory_Bound.DRAM_Bound                 % Stalls                       1.5  < [ 3.3%]   
S0-C0-T0 BE/Mem           Backend_Bound.Memory_Bound.PMM_Bound                  % Stalls                       0.0  <   
S0-C0-T0 BE/Mem           Backend_Bound.Memory_Bound.Store_Bound                % Stalls                       3.0  < [ 3.3%]   
S0-C0-T0 BE/Core          Backend_Bound.Core_Bound.Divider                      % Clocks                       1.7  < [ 3.3%]   
S0-C0-T0 BE/Core          Backend_Bound.Core_Bound.Serializing_Operation        % Clocks                       9.3  < [ 3.3%]   
S0-C0-T0 BE/Core          Backend_Bound.Core_Bound.Ports_Utilization            % Clocks                      35.9    [ 3.3%]   
S0-C0-T0 RET              Retiring.Light_Operations.FP_Arith                    % Uops                         0.0  < [ 3.3%]   
S0-C0-T0 MUX                                                                    %                              3.32 
S0-C0-T1 FE               Frontend_Bound.Fetch_Latency.ICache_Misses            % Clocks                      13.5    [ 3.3%]   
S0-C0-T1 FE               Frontend_Bound.Fetch_Latency.ITLB_Misses              % Clocks                       2.8  < [ 3.3%]   
S0-C0-T1 FE               Frontend_Bound.Fetch_Latency.Branch_Resteers          % Clocks                      14.0    [ 3.3%]   
S0-C0-T1 FE               Frontend_Bound.Fetch_Latency.MS_Switches              % Clocks_est                   9.2    [ 3.3%]   
S0-C0-T1 FE               Frontend_Bound.Fetch_Latency.LCP                      % Clocks                       0.0  < [ 3.3%]   
S0-C0-T1 FE               Frontend_Bound.Fetch_Latency.DSB_Switches             % Clocks                       0.0  < [ 3.3%]   
S0-C0-T1 FE               Frontend_Bound.Fetch_Bandwidth.MITE                   % Slots_est                    0.1  < [ 3.3%]   
S0-C0-T1 FE               Frontend_Bound.Fetch_Bandwidth.DSB                    % Slots_est                    0.0  < [ 3.3%]   
S0-C0-T1 BE/Mem           Backend_Bound.Memory_Bound.L1_Bound                   % Stalls                      15.3    [ 3.3%]   
S0-C0-T1 BE/Mem           Backend_Bound.Memory_Bound.L2_Bound                   % Stalls                       1.9  < [ 3.3%]   
S0-C0-T1 BE/Mem           Backend_Bound.Memory_Bound.L3_Bound                   % Stalls                       6.4  < [ 3.3%]   
S0-C0-T1 BE/Mem           Backend_Bound.Memory_Bound.DRAM_Bound                 % Stalls                       1.2  < [ 3.3%]   
S0-C0-T1 BE/Mem           Backend_Bound.Memory_Bound.PMM_Bound                  % Stalls                       0.0  <   
S0-C0-T1 BE/Mem           Backend_Bound.Memory_Bound.Store_Bound                % Stalls                       0.5  < [ 3.3%]   
S0-C0-T1 BE/Core          Backend_Bound.Core_Bound.Divider                      % Clocks                       0.2  < [ 3.3%]   
S0-C0-T1 BE/Core          Backend_Bound.Core_Bound.Serializing_Operation        % Clocks                      42.7    [ 3.3%]   
S0-C0-T1 BE/Core          Backend_Bound.Core_Bound.Ports_Utilization            % Clocks                      19.8    [ 3.3%]   
```
