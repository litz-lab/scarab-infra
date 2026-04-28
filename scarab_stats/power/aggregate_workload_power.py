#!/usr/bin/env python3
"""Simpoint-weighted whole-workload power & energy estimation.

For each (suite, subsuite, workload, configuration) under a sim experiment
directory, walk the per-simpoint scarab outputs, run McPAT once per
simpoint (cached), and produce a weighted whole-workload power summary.

Weights come from workloads_db.json's `simpoints[].weight` field.
Energy = weighted_power * total_simulated_time. Total time is computed
from simpoint cycles (also weight-aggregated) at the configured
clock frequency.

Usage:
    aggregate_workload_power.py --sim-root /soe/surim/simulations/agent_sim \
        --config golden_cove_base \
        --workloads-db /soe/surim/src/infra_agent_perf/workloads/workloads_db.json \
        --suite agent \
        [--workloads <name1>,<name2>...]
        [--clock-ghz 4.0]
        [--jobs 8]
        [--out power_summary.csv]

Output CSV columns: suite, subsuite, workload, config,
    avg_runtime_dynamic_W, avg_leakage_W, avg_total_W,
    weighted_cycles, sim_time_s, energy_J
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import re
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(THIS_DIR))

from converter import Converter  # noqa: E402
from parse_power_stat import scarab_csv_to_power_dict  # noqa: E402

import os
XML_TEMPLATE = THIS_DIR / "xml" / "template.xml"
# MCPAT_BIN env var lets users point at a system or pre-built McPAT;
# otherwise fall back to the binary install_mcpat.sh drops next to us.
MCPAT_BIN = Path(os.environ.get("MCPAT_BIN", str(THIS_DIR / "mcpat")))


# === Per-simpoint pipeline ==================================================

def build_mcpat_xml(sim_dir: Path, work_dir: Path) -> Path:
    """Run PR 178's 4-step XML conversion scoped to one simpoint output."""
    work_dir.mkdir(parents=True, exist_ok=True)
    json_out = work_dir / "json"
    json_out.mkdir(exist_ok=True)

    converter = Converter()
    converter.generate_json_from_template(
        xml_path=str(XML_TEMPLATE),
        output_dir=str(json_out),
    )
    converter.update_json_from_scarab_params(
        scarab_params_path=str(sim_dir / "PARAMS.out"),
        mcpat_json_path=str(json_out / "params_table.json"),
    )
    # Use scarab's native power.stat.0.csv totals — no manual power.pkl.
    power_dict = scarab_csv_to_power_dict(sim_dir / "power.stat.0.csv")
    stats_table_path = json_out / "stats_table.json"
    with open(stats_table_path) as f:
        stats_json = json.load(f)
    updated = converter._map_scarab_stats_to_mcpat_json(power_dict, stats_json)
    with open(stats_table_path, "w") as f:
        json.dump(updated, f, indent=2)

    xml_out = work_dir / "mcpat_infile.xml"
    converter.generate_xml_from_json(
        structure_path=str(json_out / "mcpat_structure.json"),
        params_table_path=str(json_out / "params_table.json"),
        stats_table_path=str(json_out / "stats_table.json"),
        output_path=str(xml_out),
    )
    return xml_out


def run_mcpat(xml_path: Path, out_path: Path) -> None:
    """Invoke the mcpat binary; stdout → out_path."""
    if not MCPAT_BIN.exists():
        raise FileNotFoundError(
            f"McPAT binary not found at {MCPAT_BIN}. Build it via "
            f"`{THIS_DIR/'install_mcpat.sh'}` (also wired into "
            f"`./sci --init`), or set MCPAT_BIN to a pre-built binary."
        )
    with open(out_path, "w") as f:
        subprocess.run(
            [str(MCPAT_BIN), "-infile", str(xml_path), "-print_level", "5"],
            stdout=f, stderr=subprocess.STDOUT, check=True,
        )


def parse_mcpat_output(out_path: Path) -> dict[str, float]:
    """Extract Total/Dynamic/Leakage power from McPAT's Processor block."""
    text = out_path.read_text(errors="replace")
    proc_idx = text.find("Processor:")
    if proc_idx < 0:
        return {}
    block = text[proc_idx:proc_idx + 1500]
    out: dict[str, float] = {}
    for label, key in [
        ("Peak Dynamic", "peak_dynamic_W"),
        ("Subthreshold Leakage", "subthreshold_leakage_W"),
        ("Gate Leakage", "gate_leakage_W"),
        ("Runtime Dynamic", "runtime_dynamic_W"),
    ]:
        m = re.search(rf"{label}\s*=\s*([0-9.eE+-]+)\s*W", block)
        if m:
            out[key] = float(m.group(1))
    if "runtime_dynamic_W" in out:
        out["total_W"] = out["runtime_dynamic_W"] + out.get(
            "subthreshold_leakage_W", 0.0) + out.get("gate_leakage_W", 0.0)
    return out


def load_simpoints(workloads_db_path: Path, suite: str) -> dict[str, list[dict]]:
    """Return {workload_name: [{cluster_id, weight, segment_id}, ...]}."""
    with open(workloads_db_path) as f:
        db = json.load(f)
    out: dict[str, list[dict]] = {}
    suite_data = db.get(suite, {})
    for subsuite, ws in suite_data.items():
        if not isinstance(ws, dict):
            continue
        for wl_name, wl_data in ws.items():
            sps = wl_data.get("simpoints", [])
            if sps:
                out[wl_name] = [
                    {"cluster_id": sp["cluster_id"],
                     "weight":     sp["weight"],
                     "subsuite":   subsuite}
                    for sp in sps
                ]
    return out


def power_for_simpoint(args_tuple) -> tuple[int, dict[str, float] | None]:
    """Run McPAT once for a simpoint sim dir; return (cluster_id, summary).

    Caches: if power/power_summary.json already exists, skip rerun.
    """
    sim_dir, cluster_id = args_tuple
    sim_dir = Path(sim_dir)
    cached = sim_dir / "power" / "power_summary.json"
    if cached.is_file():
        with open(cached) as f:
            return cluster_id, json.load(f)
    if not (sim_dir / "PARAMS.out").is_file():
        return cluster_id, None
    if not (sim_dir / "power.stat.0.csv").is_file():
        return cluster_id, None
    work_dir = sim_dir / "power"
    try:
        xml_path = build_mcpat_xml(sim_dir, work_dir)
        mcpat_out = work_dir / "mcpat.out"
        run_mcpat(xml_path, mcpat_out)
        summary = parse_mcpat_output(mcpat_out)
    except Exception as e:
        print(f"# WARN cluster {cluster_id} ({sim_dir}): {e}",
              file=sys.stderr)
        return cluster_id, None
    if summary:
        with open(cached, "w") as f:
            json.dump(summary, f, indent=2)
    return cluster_id, summary


def get_simpoint_cycles(sim_dir: Path) -> float:
    """Pull cycles from power.stat.0.csv (Cumulative_Cycles row)."""
    p = sim_dir / "power.stat.0.csv"
    if not p.is_file():
        return 0.0
    for line in p.read_text(errors="replace").splitlines():
        parts = [s.strip() for s in line.split(",")]
        if parts and parts[0] == "Cumulative_Cycles" and len(parts) >= 3:
            try:
                return float(parts[-1])
            except ValueError:
                pass
    return 0.0


def aggregate_workload(
    sim_root: Path, config: str, suite: str, subsuite: str,
    workload: str, simpoints: list[dict],
    pool: mp.Pool, clock_ghz: float,
) -> dict | None:
    wl_dir = sim_root / config / suite / subsuite / workload
    if not wl_dir.is_dir():
        return None
    work = []
    for sp in simpoints:
        cid = sp["cluster_id"]
        sim_dir = wl_dir / str(cid)
        if sim_dir.is_dir():
            work.append((str(sim_dir), cid))
    if not work:
        return None
    results = pool.map(power_for_simpoint, work)
    cid_to_power = dict(results)

    weight_sum = 0.0
    rd_acc = lk_acc = tot_acc = 0.0
    cycles_acc = 0.0
    for sp in simpoints:
        cid = sp["cluster_id"]
        w = sp["weight"]
        summary = cid_to_power.get(cid)
        if summary is None:
            continue
        sim_dir = wl_dir / str(cid)
        cycles = get_simpoint_cycles(sim_dir)
        weight_sum += w
        rd_acc  += w * summary.get("runtime_dynamic_W", 0.0)
        lk_acc  += w * (summary.get("subthreshold_leakage_W", 0.0)
                        + summary.get("gate_leakage_W", 0.0))
        tot_acc += w * summary.get("total_W", 0.0)
        cycles_acc += w * cycles
    if weight_sum == 0:
        return None
    avg_rd  = rd_acc  / weight_sum
    avg_lk  = lk_acc  / weight_sum
    avg_tot = tot_acc / weight_sum
    sim_time_s = cycles_acc / (clock_ghz * 1e9)
    energy_J = avg_tot * sim_time_s
    return {
        "suite": suite,
        "subsuite": subsuite,
        "workload": workload,
        "config": config,
        "n_simpoints_with_power": sum(1 for r in results if r[1] is not None),
        "weight_covered": weight_sum,
        "avg_runtime_dynamic_W": avg_rd,
        "avg_leakage_W": avg_lk,
        "avg_total_W": avg_tot,
        "weighted_cycles": cycles_acc,
        "sim_time_s": sim_time_s,
        "energy_J": energy_J,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim-root", type=Path, required=True,
                    help="e.g., /soe/surim/simulations/agent_sim")
    ap.add_argument("--config", required=True, help="e.g., golden_cove_base")
    ap.add_argument("--suite", required=True, help="e.g., agent")
    ap.add_argument("--workloads-db", type=Path, required=True)
    ap.add_argument("--workloads", default="",
                    help="comma-separated workload names (default: all)")
    ap.add_argument("--clock-ghz", type=float, default=4.0)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("power_summary.csv"))
    args = ap.parse_args()

    sims = load_simpoints(args.workloads_db, args.suite)
    if args.workloads:
        wanted = {n.strip() for n in args.workloads.split(",") if n.strip()}
        sims = {k: v for k, v in sims.items() if k in wanted}
        missing = wanted - set(sims)
        if missing:
            print(f"# WARN: unknown workloads: {sorted(missing)}",
                  file=sys.stderr)
    if not sims:
        print("# no workloads selected", file=sys.stderr)
        return 1

    rows: list[dict] = []
    with mp.Pool(args.jobs) as pool:
        for wl, sps in sorted(sims.items()):
            subsuite = sps[0]["subsuite"]
            r = aggregate_workload(
                args.sim_root, args.config, args.suite, subsuite,
                wl, sps, pool, args.clock_ghz,
            )
            if r is None:
                print(f"# {wl}: no sim outputs found", file=sys.stderr)
                continue
            rows.append(r)
            print(f"  {wl:<32} cov={r['weight_covered']:.3f}  "
                  f"avg_total={r['avg_total_W']:.2f}W  "
                  f"E={r['energy_J']:.4f}J ({r['sim_time_s']*1e3:.2f}ms)")
    if not rows:
        print("# no power data produced", file=sys.stderr)
        return 1
    cols = list(rows[0].keys())
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"# wrote {args.out} ({len(rows)} workloads)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
