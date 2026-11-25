# Synthetic Kernel Generator Documentation

The Synthetic Kernel Generator produces targeted, unrolled instruction loops to stress specific microarchitectural structures and simulate precise performance bottlenecks. It dynamically handles instruction generation, sequence sampling, and execution dispatching.

---

## Table of Contents
1. [Configuration Knobs](#1-configuration-knobs)
2. [Kernel Profiles & Behavior](#2-kernel-profiles--behavior)
3. [Advanced Architecture: Control Flow (CF) Mechanics](#3-advanced-architecture-control-flow-cf-mechanics)

---

## 1. Configuration Knobs

The `Kernel_Factory` class configures the following variables before synthesizing a kernel. These variables control the shape, target addresses, and execution behavior of the generated instruction stream.

| Variable | Type | Description | 
| ----- | ----- | ----- | 
| **frontend** | `enum` | Defines the frontend configuration. | 
| **kernel** | `enum` | Defines the type of synthetic kernel to generate (e.g., `MEM_FU_LIMITED`, `ILP_LIMITED`, `CBR_LIMITED`). | 
| **workload_length** | `UNS64` | The total number of instructions or sequence length of the workload. | 
| **starting_target** | `UNS64` | The starting footprint value (e.g., the first branch target for control-flow workloads, or the first effective memory address for memory workloads). | 
| **target_stride** | `UNS64` | The uniform stride by which a target sequence increases. | 
| **target_pool_size** | `UNS64` | The size or length of the sequence pool. | 
| **t_nt_ratio** | `double` | The distribution ratio of taken (1) vs. not-taken (0) branches in the random pool of branch directions. | 
| **memory_hierarchy_level** | `enum` | Focuses the bottleneck on a specific cache/memory level for latency and bandwidth workloads. | 
| **dependence_chain_count** | `UNS64` | Defines the number of mutually exclusive data dependence chains for ILP workloads. | 
| **inst_size** | `UNS64` | The byte size of the primary instructions utilized in the workload (e.g., size of load, CBR, or ALU op). | 
| **nop_size** | `UNS64` | The byte size of a NOP instruction utilized in the workload. | 
| **pad_length** | `UNS64` | The number of padded NOPs required at backward branches to handle off-path redirect recovery. | 
| **target_sampling_strategy** | `enum` | How targets are selected: `UNIFORM_SEQUENTIAL`, `UNIFORM_RANDOM`, `NORMAL_RANDOM`, `DISCRETE_RANDOM`, or `USER_DEFINED`. | 
| **direction_sampling_strategy** | `enum` | How branch directions are selected, using the same strategies as `target_sampling_strategy`. | 

---

## 2. Kernel Profiles & Behavior

The generator maps the selected `kernel` enum to specific implementation logic that synthesizes bottleneck-inducing instruction streams.

### 2.1 Control Flow (CF) Kernels
Control flow kernels are designed to trigger Top-Down "Bad Speculation" bottlenecks. Control flow kernels are designed to trigger Top-Down "Bad Speculation" bottlenecks. They share structural similarities: every branch is preceded by NOPs to ensure the frontend delivers maximum issue width per cycle. Rather than strictly sizing packets to fill an entire cache line, issue packets are simply bounded by the ICACHE_LINE_SIZE to prevent boundary-crossing stalls, with any leftover dead space cleanly padded with NOPs.

* **CBR_LIMITED (`cbr_limited`)**
  * Generates conditional branches with fixed targets but randomized directions.
  * Directions (taken/not-taken) are weighted according to `t_nt_ratio`.
  * The `Kernel_Dispatcher` rerandomizes the branch direction sequence each time control returns to the beginning of the kernel.

* **UBR_LIMITED / BTB_LIMITED (`btb_limited`)**
  * Generates a sequence of unconditional branches using a set of uniformly generated, shuffled targets.
  * Depending on the configured target pool pattern, this kernel can guarantee either a 100% BTB hit rate, a 100% miss rate, or a specific ratio in between.

* **IBR_LIMITED (`ibr_limited`)**
  * Produces indirect branches branching to targets obtained indirectly from a fixed memory address via a source register.
  * If a randomized target pool is used, the `Kernel_Dispatcher` rerandomizes the targets when execution loops back to the start.

### 2.2 Memory Kernels
These orchestrate latency or bandwidth bottlenecks in the memory subsystem.

* **LOAD_LATENCY_LIMITED (`load_latency_limited`)**
  * Limits IPC using the latency of specific cache hierarchy levels.
  * Uses an unrolled, loop-carried dependence chain of load instructions.
  * Dynamically configures `target_stride` and `target_pool_size` so each load hits the targeted cache level but causes 100% conflict misses in all preceding levels. (e.g., `MLC_LIMITED` misses in DCACHE but hits in MLC).

* **MEM_FU_LIMITED (`mem_fu_limited`)**
  * Stresses memory bandwidth and load execution units.
  * Generates a stream of completely independent loads without loop-carried dependencies, allowing concurrent execution limited only by the available Load Functional Units.

### 2.3 Instruction-Level Parallelism (ILP) Kernels
* **ILP_LIMITED (`ilp_limited`)**
  * Constructs sequences of ALU instructions mapped to source and destination registers to create strict dependence chains.
  * If `dependence_chain_count` is 1, each instruction stalls until the previous one completes.
  * If `dependence_chain_count` > 1, the factory generates mutually exclusive chains, allowing parallel execution up to the configured chain limit.

### 2.4 Instruction Cache Kernels
* **ICACHE_LIMITED (`icache_limited`)**
  * Stresses frontend cache capacity.
  * Forces the workload loop size to explicitly exceed `ICACHE_SIZE`, guaranteeing a 100% I-Cache miss rate upon looping.
  * Requires frontend prefetchers (FDIP) to be disabled for accurate results, and sizes all generated instructions to the `ICACHE_LINE_SIZE`.

---

## 3. Advanced Architecture: Control Flow (CF) Mechanics

The Control Flow workloads (`cbr_limited`, `btb_limited`, `ibr_limited`) are meticulously engineered to isolate the Top-Down "Bad Speculation" bottleneck. Achieving analytical purity requires preventing unwanted "Frontend Bound" noise, such as I-Cache or BTB capacity misses, which would skew metric estimations.

### 3.1 The Master Circular Path & Target Randomization
To guarantee that the frontend structures are stressed solely by prediction logic rather than capacity limits, all CF kernels execute as an unrolled loop that follows a strict, inescapable circular path.

* **Pre-Simulation Scattering:** Before the simulation begins, the target sequence is randomly shuffled. This creates a complex, scattered instruction footprint that defeats basic sequential hardware prefetchers and simple BTB logic.
* **Static Topology:** Once simulation starts, this randomized target sequence is permanently locked. The BTB and I-Cache are allowed to fully warm up and "learn" the spatial footprint of the scattered targets.
* **Isolating the Predictor:** With spatial randomness neutralized by the warm caches, the hardware is forced to mispredict based entirely on dynamic elements: the randomized Taken/Not-Taken branch *directions* (for CBR) or the continually rerandomized indirect targets (for the random IBR variant).

### 3.2 The Short-Circuit Problem & The UBR Safety Net
The `cbr_limited` kernel poses a unique structural risk: conditional branches have both a Taken path and a Not-Taken (fall-through) path. Because the branch direction is randomized to induce mispredictions, a branch that correctly evaluates as Not-Taken would naturally fall through to the next sequential instruction. 

Without strict routing, a random fall-through could accidentally jump to a previously visited target, creating a closed loop. If the program gets trapped in a closed loop, it will iterate indefinitely among a small subset of addresses and never traverse the full target pool or reach the end of the workload.

To guarantee the master circular path is strictly traversed regardless of branch direction, the `cbr_limited` kernel employs an Unconditional Branch (UBR) safety net. Below is a sequence schema demonstrating this dual-entry routing:

```text
// Packet A (Target N)
[64]   NOP  1
[65]   NOP  1
[66]   NOP  1
[67]   NOP  1
[68]   NOP  1
[69]   CBR  1 → TAKEN: Tgt1
[70]   UBR  1 → Tgt1 + 1 
... (nop dead space to end of cache line)

// Packet B (Target N+1)
[128]  Tgt1:     NOP 1  <-- Primary Entry (if Packet A CBR is Taken)
[129]  Tgt1 + 1: NOP 1  <-- Secondary Entry (if Packet A CBR is Not-Taken, routed via UBR)
[130]            NOP 1
[131]            NOP 1
[132]            NOP 1
[133]            NOP 1
[134]            CBR 1 → TAKEN: Tgt2
[135]            UBR 1 → Tgt2 + 1
... (nop dead space to end of cache line)
```

* **Taken Path:** The CBR branches directly to the primary entry point of the next randomized target block (e.g., `Tgt1` at `[128]`).
* **Not-Taken Path:** Execution falls through directly into the UBR, which unconditionally routes control flow to a secondary entry point within the exact same target block (e.g., `Tgt1 + 1` at `[129]`). 

Because the `btb_limited` (UBR) and `ibr_limited` kernels utilize absolute jumps with no sequential fall-through paths, they are natively immune to this closed-loop risk and do not require a safety net.

### 3.3 Dispatch Consistency & The NOP Balancing Act
To ensure that Top-Down metric estimations perfectly match the simulation results, the execution stream must process the exact same number of micro-ops between branch evaluations. Fluctuating retire counts based on branch direction would introduce structural noise. 

To solve this, the generator normalizes all operations to 1-byte sizes (`NOP 1`, `CBR 1`, `UBR 1`) and utilizes a highly specific NOP padding strategy to balance the execution paths.

**Unconditional Kernels (`btb_limited`, `ibr_limited`)**
Because there is no fall-through path to manage, the math is straightforward.
* The issue packet is constructed with exactly **(`DISPATCH_WIDTH - 1`) NOPs + 1 UBR/IBR**. 
* The pipeline flawlessly ingests and executes exactly one `DISPATCH_WIDTH` worth of instructions per target.

**Conditional Kernel (`cbr_limited`)**
The CBR kernel must offset the operational cost of the UBR safety net triggered on the fall-through path. It does this by front-loading **`DISPATCH_WIDTH` NOPs + 1 CBR + 1 UBR** and manipulating the jump targets to balance the final retired operation count:
* **When CBR is Taken:** The branch jumps to the *first* NOP of the next target block (`Tgt1`). The pipeline executes exactly `DISPATCH_WIDTH` NOPs + 1 CBR.
* **When CBR is Not-Taken (Fall-through):** The branch falls through to the UBR (costing 1 operation). To offset this, the UBR is hardcoded to jump to `Tgt1 + 1`, intentionally skipping the first NOP of the destination block. The pipeline executes 1 UBR + `(DISPATCH_WIDTH - 1)` NOPs + 1 CBR.

In both scenarios, the pipeline executes the exact same number of operations. While the very first issue packet of the program may execute one micro-op more than the steady state, perfect dispatch symmetry is achieved for all subsequent iterations.

### 3.4 Configuration Preconditions & Constraints
Because the generator relies on precise mathematical alignment for its padding and routing, the user-defined configuration variables (`target_stride`, `nop_size`, `inst_size`) must satisfy strict preconditions. The `prechecks_for_CF_kernels` validation logic enforces the following:

* **Packet Overlap Prevention:** The total byte size of an issue packet is calculated as `(nop_count * nop_size) + (cf_op_count * inst_size)`. This total size must be less than or equal to the `target_stride`, and `target_stride` must be a perfect multiple of the packet size (`target_stride % packet_size == 0`). If violated, issue packets will physically overwrite each other in memory.
* **Cache Line Padding:** To prevent issue packets from awkwardly crossing cache boundaries and inducing Frontend stalls, packets are generally bounded by the `ICACHE_LINE_SIZE`. If a packet does not perfectly fill the cache line, the leftover "dead space" must be cleanly divisible by `nop_size` (`(ICACHE_LINE_SIZE - packet_size) % nop_size == 0`), allowing the generator to pad the remainder of the line with NOPs.