#!/usr/bin/env python3
"""Fan out per-phase native PMU characterization across all 30 agent workloads.

Reads json/agent_trace.json for the workload list + binary_cmd, sbatches one
slurm job per workload constrained to the Cascade Lake nodes (bohr4, bohr5,
ohm). Each job runs the agent container with --cap-add=PERFMON and invokes
workloads/perf/run_with_perf.sh, which produces per-phase perf_*.csv files
plus phase logs.

Output layout:
    /soe/surim/perf_runs/<workload>/perf_<phase>_<group>.csv
                                  /agent_phase_log_<phase>.txt
                                  /sbatch.out

Usage:
    dispatch_perf.py [--workloads <name1>,<name2>,...]
                     [--nodelist bohr4,bohr5,ohm]
                     [--image agent:LATEST]
                     [--outroot /soe/surim/perf_runs]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

INFRA_DIR = Path("/soe/surim/src/infra_agent_perf")
DESC = INFRA_DIR / "json" / "agent_trace.json"
DEFAULT_OUTROOT = Path("/soe/surim/perf_runs")
DEFAULT_NODELIST = "bohr4,bohr5,ohm"
DEFAULT_IMAGE = "agent:f0f6239"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workloads", default="",
                    help="Comma-separated workload names (default: all 30)")
    ap.add_argument("--nodelist", default=DEFAULT_NODELIST)
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--outroot", type=Path, default=DEFAULT_OUTROOT)
    # 5 GB Docker-image load + workload runtime → 4 GB OOMs on first try.
    # 8 GB has comfortable headroom; small workloads still finish quickly.
    ap.add_argument("--mem-mb", type=int, default=8192)
    ap.add_argument("--dry-run", action="store_true",
                    help="Print sbatch commands without submitting")
    return ap.parse_args()


def load_configs() -> list[dict]:
    with DESC.open() as f:
        d = json.load(f)
    return d["trace_configurations"]


def select_configs(all_configs: list[dict], names_csv: str) -> list[dict]:
    if not names_csv:
        return all_configs
    wanted = {n.strip() for n in names_csv.split(",") if n.strip()}
    out = [c for c in all_configs if c["workload"] in wanted]
    missing = wanted - {c["workload"] for c in out}
    if missing:
        print(f"WARN: unknown workload names: {sorted(missing)}", file=sys.stderr)
    return out


def make_inner_cmd(workload: str, binary_cmd: str, outdir_in_container: str) -> str:
    """Build the bash command that runs inside the docker container.

    binary_cmd from the descriptor is the python3 invocation (with \\$tmpdir
    template). We strip the leading "python3 \\$tmpdir/AgentCPU/workloads/run.py"
    and pass everything after as positional args to run_with_perf.sh.
    """
    prefix = "python3 \\$tmpdir/AgentCPU/workloads/run.py "
    if binary_cmd.startswith(prefix):
        py_args = binary_cmd[len(prefix):]
    else:
        # Fallback: try without leading backslash escape
        prefix2 = "python3 $tmpdir/AgentCPU/workloads/run.py "
        if binary_cmd.startswith(prefix2):
            py_args = binary_cmd[len(prefix2):]
        else:
            raise SystemExit(f"unexpected binary_cmd format: {binary_cmd!r}")

    # Hardcode HF cache path (no $-vars) so we can avoid escaping headaches
    # when the inner command is wrapped in sbatch --wrap=\"...\".
    # Cache HF models under /soe/surim (NAS, 7 TB free) instead of /tmp
    # inside the container — heavier workloads (haystack with corpus
    # encoding) otherwise fill the slurm node's /tmp (5 GB image +
    # transformer activations + model files = OOM).
    return (
        "mkdir -p /soe/surim/hf_cache_shared && "
        "cp -rn /root/.cache/huggingface/. /soe/surim/hf_cache_shared/ 2>/dev/null || true; "
        "HF_HOME=/soe/surim/hf_cache_shared "
        f"bash /soe/surim/src/AgentCPU/workloads/perf/run_with_perf.sh "
        f"{outdir_in_container} {py_args}"
    )


def build_sbatch(cfg: dict, args: argparse.Namespace) -> tuple[str, Path]:
    workload = cfg["workload"]
    outdir = args.outroot / workload
    outdir.mkdir(parents=True, exist_ok=True)
    inner = make_inner_cmd(workload, cfg["binary_cmd"], str(outdir))

    # Load the image from the NAS cache if it's not already on this slurm
    # node. The cache tar lives at infra/docker_image_cache/<image_basename>.tar
    # where <image_basename> = image tag with ':' → '_'.
    img_basename = args.image.replace(":", "_")
    cache_tar = INFRA_DIR / "docker_image_cache" / f"{img_basename}.tar"
    load_cmd = (
        f"docker image inspect {args.image} > /dev/null 2>&1 || "
        f"docker load -i {cache_tar} > /dev/null"
    )

    docker_cmd = (
        f"{load_cmd} && "
        f"docker run --rm "
        f"-u $(id -u):$(id -g) "
        f"-e HOME=/tmp "
        f"-v /soe/surim:/soe/surim "
        f"--cap-add=PERFMON --cap-add=SYS_ADMIN "
        f"{args.image} bash -c '{inner}'"
    )

    sbatch_log = outdir / "sbatch.out"
    sbatch_cmd = (
        f"sbatch --nodelist={args.nodelist} --nodes=1 "
        f"--mem={args.mem_mb}M -c 1 --exclusive "
        f"-J perf_{workload} "
        f"-o {sbatch_log} "
        f"--wrap=\"{docker_cmd}\""
    )
    return sbatch_cmd, sbatch_log


def main() -> int:
    args = parse_args()
    args.outroot.mkdir(parents=True, exist_ok=True)
    configs = select_configs(load_configs(), args.workloads)
    print(f"# {len(configs)} workloads → {args.outroot}, nodelist={args.nodelist}, image={args.image}")
    for cfg in configs:
        cmd, log = build_sbatch(cfg, args)
        if args.dry_run:
            print(cmd)
            continue
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FAIL {cfg['workload']}: {result.stderr.strip()}", file=sys.stderr)
            continue
        jid = result.stdout.strip().split()[-1]
        print(f"  {cfg['workload']:<30} → job {jid}  log={log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
