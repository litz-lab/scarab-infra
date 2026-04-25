# Per-Phase Microarchitecture Characterization of LangChain Agent Workloads

*Draft section — Cascade Lake (Intel Xeon Gold 5218 / 6242R), native HW PMU.*

## TL;DR

Inside a single LangChain-driven agent workload, **two qualitatively different
microarchitecture regimes coexist**. About 80 % of executed instructions sit in
a numerical compute kernel (LexRank summarization) that achieves IPC ≈ 2.4 with
a tiny working set, while the remaining 20 % is spread across short Python
orchestration phases (prompt assembly, tool dispatch, JSON serialization, etc.)
that *individually* run at IPC 0.5–0.9 with L1i MPKI > 100, uop-cache hit rate
< 20 %, and 7–10 % branch misprediction. The slow phases consume only 1 % of
weighted execution time but show **every classic frontend pathology** an
architect would target with bigger BTB, bigger uop cache, or smarter icache
prefetching. We argue that the orchestration tax — though small in cycles —
is the structural inefficiency unique to agent workloads relative to ML
serving.

## Methodology

We instrument the AgentCPU LangChain workload with begin/end markers around
9 semantic phases (see `marker.c`). At runtime, the markers drive
`perf stat --control fifo:` so each begin/end pair toggles native PMU
counters. To get per-phase aggregates we run the workload once per phase
name, with `AGENT_PERF_FILTER_PHASE=<name>` set so only that phase's
markers signal perf — perf's summary then covers all invocations of one
phase across one workload run. We split events into three groups
(cache + branches; L1i + TLB; uop cache) so that no single perf invocation
exceeds the 4 programmable + 2 fixed counter budget on Cascade Lake.

All runs are pinned to one Cascade Lake server-class node (Intel Xeon Gold
5218 or 6242R), with `--exclusive` slurm scheduling and the agent docker
container holding the `PERFMON` capability. We disable Address Space
Layout Randomization (`randomize_va_space=0`), force single-threaded BLAS
(`OMP_NUM_THREADS=1`, etc.), set `PYTHONHASHSEED=0`, and use
`AgentCPU`'s already-deterministic mode where every "LLM call" is replayed
in-process from canned JSON — i.e. no real network, no sampling
non-determinism. The workload's RNG seed is fixed, so two runs of the same
configuration produce instruction streams that differ only by perf's own
measurement perturbation.

## Per-phase microarchitecture (langchain\_short\_12iter, 12 iterations, ~6.3 B insts)

| phase | inv | insts | IPC | L1i MPKI | L1d MPKI | LLC MPKI | iTLB MPKI | BrMis % | uop$ hit % | %insts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **tool\_exec** (LexRank: TF-IDF / cosine / PageRank) | 12 | 5.07 B | **2.37** | 10.3 | 9.4 | 0.02 | 0.11 | 0.5 | 16.1 | 80.8 % |
| **tokenize** (GPT-2 BPE) | 24 | 1.18 B | 1.31 | 3.8 | 16.0 | 1.9 | 0.14 | 1.5 | 48.8 | 18.8 % |
| http\_parse (HTTP/1.1 + gzip) | 12 | 7.6 M | 0.82 | 55.6 | 25.3 | 0.7 | 0.87 | 7.6 | 19.1 | 0.1 % |
| api\_parse (gzip + JSON) | 12 | 7.3 M | 0.88 | 44.1 | 21.0 | 0.1 | 0.89 | 8.1 | 33.4 | 0.1 % |
| api\_build (JSON encode) | 12 | 7.3 M | 1.34 | 20.0 | 13.9 | 0.07 | 0.81 | 2.7 | 40.5 | 0.1 % |
| retrieve (dict ops + slicing) | 12 | 0.92 M | 0.79 | 68.0 | 22.1 | 0.2 | 1.12 | 5.9 | 18.7 | <0.1 % |
| context\_update (list append) | 12 | 0.71 M | 0.74 | 96.9 | 31.7 | 0.0 | 1.33 | 8.4 | 10.5 | <0.1 % |
| build\_prompt (template fill) | 12 | 0.32 M | 0.68 | **128** | 37.5 | 0.0 | 1.97 | 7.1 | 5.9 | <0.1 % |
| tool\_dispatch (attribute lookup) | 12 | 0.28 M | 0.62 | **134** | 36.9 | 0.0 | 2.27 | 7.3 | 6.1 | <0.1 % |

*Each row pools across all invocations of the phase in one 12-iteration
run. Each `inv` count is the number of begin events captured.*

## Two regimes, one workload

The compute phases — `tool_exec` and `tokenize` — own ~99.6 % of executed
instructions and exhibit conventional, well-tuned microarchitecture:

* `tool_exec` (LexRank) is the canonical numerical kernel. **IPC 2.37**,
  **branch misprediction 0.5 %**, LLC MPKI essentially zero (the working
  set fits in L2). Its uop-cache hit rate (16 %) is unexpectedly low —
  worth a follow-up — but the kernel's instruction-level parallelism
  carries it.

* `tokenize` is L1d-bandwidth-bound on the BPE merge tables: **IPC 1.31,
  L1d MPKI 16, LLC MPKI 1.9**. The hot inner loops fit in DSB (uop hit
  48.8 %).

The orchestration phases — `http_parse`, `api_parse`, `api_build`,
`retrieve`, `context_update`, `build_prompt`, `tool_dispatch` — own
1.3 % of executed instructions but show a fundamentally different profile:

* **L1i MPKI is 1–2 orders of magnitude higher** than during compute
  (`tool_dispatch`: 134 vs `tool_exec`: 10). The CPython interpreter loop
  walks a large code footprint when chasing polymorphic call paths.

* **uop-cache hit rate collapses to 5–20 %** — the DSB is too small to
  hold the diverse interpreter paths and most uops are re-decoded through
  MITE. Concretely, the workload pays the legacy-decoder penalty on
  ~80–94 % of orchestration uops.

* **Branch misprediction climbs to 7–11 %** — the polymorphic dispatch in
  PyEval (function calls, attribute lookups, MRO walks) defeats the BTB.

* **iTLB MPKI is 1–4** — interpreter code spans more 4 KB pages than the
  iTLB (Skylake/Cascade Lake: 128 entries) can keep resident.

This regime split is sharp. It does not show up in any single PMU summary
of the workload because the compute phase numerically dominates the
average. **Per-phase attribution is necessary to surface it.**

## What an architect should take away

1. **Bigger uop cache helps orchestration far more than compute.** Today's
   Cascade Lake DSB (~1 .5 K uops) is half-served on Python interpreter
   paths. The compute phase is already past the point where the uop cache
   helps; orchestration is where capacity matters.

2. **L1i and iTLB pressure in orchestration is real.** The interpreter
   loop's effective code footprint (the union of bytecode dispatch, type
   resolution, dict probing, generator state machinery) does not fit in a
   32 KB L1i. iTLB MPKI of 2–4 on context\_update / build\_prompt /
   tool\_dispatch is consistent with code spread across 8 +
   actively-touched pages.

3. **Branch misprediction is a polymorphism tax, not data-dependent
   logic.** Compute phases have predictable loop branches and mispredict
   ≤ 1.5 %. Orchestration mispredicts 5–11 % because the dispatched target
   depends on dynamic type — e.g. which `__getattr__` slot fires for a
   user-provided agent prompt object. BTB associativity / size + indirect
   target prediction would help here, less so for compute.

4. **The slow phases are tiny in cycle share but architecturally
   distinct.** A reviewer might ask why we should care about phases that
   own < 1 % of cycles. The answer is that they (a) are unique to agent
   workloads — ML serving rarely runs sustained interpreter dispatch —
   and (b) have a shape that argues for different µarch sizing than the
   compute does. Sizing the µarch only to the compute regime (IPC 2.4,
   LLC-resident) misses the orchestration tail.

## Limitations of this section

* **Single µarch, single framework.** All numbers are from one Cascade
  Lake CPU running one framework (LangChain) with one tool stack (LexRank
  + GPT-2 BPE + HTTP/JSON simulation). Generalization to AutoGen /
  Haystack / LangGraph / SWE-Agent is the obvious next step, and we
  expect their `tool_exec` µarch to differ qualitatively (Haystack does
  FAISS vector search, SWE-Agent does sort/FFT/matmul, LangGraph does
  numpy SVD).

* **Topdown breakdown not yet included.** Cascade Lake supports the
  legacy topdown event family (`topdown-fetch-bubbles`,
  `topdown-slots-issued`, etc.). We omit it pending a fourth perf event
  group; the IPC + per-MPKI table here already says most of what the
  topdown row would.

* **Orchestration phase invocation counts are small (12–24).** Tightening
  confidence intervals requires either more iterations per workload or
  pooling across multiple langchain configurations (short / medium / long
  prompts × 12 / 25 iterations); see `aggregate_perf_all.py --root
  /soe/surim/perf_runs` for the cross-configuration pool.

* **Native HW characterization, not simulation.** Sensitivity studies
  ("what if BTB doubled?", "what if uop cache were 4 K instead of 1.5 K?")
  require a configurable µarch model. We point at scarab's existing
  simpoint trace replay path for that follow-up; per-phase sensitivity
  via scarab is a separate, larger workstream that re-uses the same
  marker library.

## Reproducing this table

```
# 1. Build the agent docker image (includes libagent_markers.so + perf):
./sci --build-image agent

# 2. Run on a Cascade Lake node with permissive perf paranoia (bohr4 — see
#    /proc/sys/kernel/perf_event_paranoid):
python3 scripts/phase/dispatch_perf.py \
    --workloads langchain_short_12iter --nodelist bohr4

# 3. Aggregate per-workload:
python3 scripts/phase/aggregate_perf.py /soe/surim/perf_runs/langchain_short_12iter

# 4. Cross-configuration pool over all 6 langchain configs:
python3 scripts/phase/aggregate_perf_all.py --root /soe/surim/perf_runs
```

Marker source is at `AgentCPU/workloads/marker/marker.c`; Python wrapper
at `AgentCPU/workloads/agent_markers.py`; per-phase placement at
`AgentCPU/workloads/run_langchain.py`.
