#!/usr/bin/env python3
"""cfg_analyzer.py — Aggregate per-simpoint CFG data into a weighted CFG.

Usage
-----
  python scripts/cfg_analyzer.py --descriptor json/cfg.json \\
      [--config cfg] [--workload mongodb] [--output-dir .]  \\
      [--dot] [--top-n 30] [--focus hot] [--region-depth 1]

Focus modes
-----------
  hot     : top-N nodes by execution frequency (default)
  exec    : frequency × avg execution cost (inter-BBL retire-cycle interval)
  redirect: frequency × avg redirect cost (inter-BBL predict-cycle interval;
            captures predecessor BTB-miss / misprediction-recovery delay)
            requires cfg_predict_BBL
  icache  : frequency × avg icache miss latency (decode_cycle - fetch_cycle,
            first instruction of BBL) — always available
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .utilities import expand_simulation_workloads


# ---------------------------------------------------------------------------
# BBL-level interval definitions
# ---------------------------------------------------------------------------

INTERVAL_LABELS = ["exec",     "redirect", "icache"]
INTERVAL_COLORS = ["#ff8844", "#4488ff",  "#44bb44"]
INTERVAL_TEXT   = ["white",   "white",    "black"]

FOCUS_CHOICES = ["hot", "exec", "redirect", "icache"]
_SCORE_KEY = {"hot":      "frequency",
              "exec":     "exec_score",
              "redirect": "redirect_score",
              "icache":   "icache_score"}

# Fields accumulated with weights during aggregation
_LATENCY_FIELDS = ("uop_count", "inst_count",
                   "exec_cycle_sum",
                   "pred_count", "redirect_cycle_sum",
                   "icache_cycle_sum")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_descriptor(descriptor_path: Path) -> dict:
    data = load_json(descriptor_path)
    assert data.get("descriptor_type") == "simulation", \
        f"Expected descriptor_type='simulation', got {data.get('descriptor_type')!r}"
    return data


def load_workloads_db(infra_root: Path, top_simpoint: bool) -> dict:
    fname = "workloads_top_simp.json" if top_simpoint else "workloads_db.json"
    return load_json(infra_root / "workloads" / fname)


def get_simpoint_weight(workloads_db: dict, suite: str, subsuite: str,
                        workload: str, cluster_id: int) -> float:
    entry = workloads_db.get(suite, {}).get(subsuite, {}).get(workload)
    if entry is None:
        return None
    for sp in entry.get("simpoints", []):
        if sp["cluster_id"] == cluster_id:
            return sp["weight"]
    return None


def find_cfg_data_files(root_dir: Path, experiment: str, config: str,
                        suite: str, subsuite: str,
                        workload: str) -> list[tuple[int, Path]]:
    """Return [(cluster_id, cfg_data.json path), ...] for completed simpoints."""
    base = root_dir / "simulations" / experiment / config / suite / subsuite / workload
    results = []
    if not base.exists():
        return results
    for cluster_dir in sorted(base.iterdir()):
        if not cluster_dir.is_dir():
            continue
        try:
            cluster_id = int(cluster_dir.name)
        except ValueError:
            continue
        cfg_file = cluster_dir / "cfg_data.json"
        if cfg_file.exists():
            results.append((cluster_id, cfg_file))
    return results


def _node_id(pc: str) -> str:
    """Convert a hex PC string to a valid Graphviz node identifier."""
    return "n" + pc.replace("0x", "").replace("0X", "")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_cfg(simpoint_files: list[tuple[int, float, Path]]) -> dict:
    """
    Aggregate CFG data across simpoints with weights.

    Parameters
    ----------
    simpoint_files : list of (cluster_id, weight, cfg_data_path)

    Returns
    -------
    dict with 'nodes', 'edges', 'metadata'.
    """
    nodes: dict[str, dict] = {}
    edges: dict[str, dict] = {}
    total_weight = 0.0
    simpoints_used = []

    for cluster_id, weight, path in simpoint_files:
        data = load_json(path)
        total_weight += weight
        simpoints_used.append({"cluster_id": cluster_id, "weight": weight})

        for n in data.get("nodes", []):
            key = n["start_pc"]
            if key not in nodes:
                nodes[key] = {
                    "start_pc":       n["start_pc"],
                    "end_pc":         n["end_pc"],
                    "cf_type":        n["cf_type"],
                    "weighted_count": 0.0,
                    "simpoint_count": 0,
                    **{f: 0.0 for f in _LATENCY_FIELDS},
                }
            nodes[key]["weighted_count"] += n["count"] * weight
            nodes[key]["simpoint_count"] += 1
            for f in _LATENCY_FIELDS:
                nodes[key][f] += n.get(f, 0) * weight

        for e in data.get("edges", []):
            key = f"{e['from_pc']}->{e['to_pc']}"
            if key not in edges:
                edges[key] = {
                    "from_pc":        e["from_pc"],
                    "to_pc":          e["to_pc"],
                    "cf_type":        e["cf_type"],
                    "weighted_count": 0.0,
                    "simpoint_count": 0,
                }
            edges[key]["weighted_count"] += e["count"] * weight
            edges[key]["simpoint_count"] += 1

    # --- derived BBL-level averages ---
    # retire_delta_sum has (count - 1) terms per simpoint; using weighted_count
    # as denominator gives a negligibly small underestimate, acceptable for
    # large counts.
    total_node_wcount = sum(n["weighted_count"] for n in nodes.values())
    for n in nodes.values():
        n["frequency"] = (n["weighted_count"] / total_node_wcount
                          if total_node_wcount > 0 else 0.0)
        wcount = n["weighted_count"]
        n["avg_inst_per_exec"]   = n["inst_count"]         / wcount if wcount > 0 else 0.0
        n["avg_exec_cycles"]     = n["exec_cycle_sum"]     / wcount if wcount > 0 else 0.0
        pcount = n["pred_count"]
        n["avg_redirect_cycles"] = n["redirect_cycle_sum"] / pcount if pcount > 0 else 0.0
        n["avg_icache_cycles"]   = n["icache_cycle_sum"]   / wcount if wcount > 0 else 0.0

    for e in edges.values():
        src = nodes.get(e["from_pc"])
        src_wcount = src["weighted_count"] if src else 0.0
        e["bias"] = e["weighted_count"] / src_wcount if src_wcount > 0 else 0.0

    # --- dominant-predecessor CF type ---
    in_edges: dict[str, list] = {}
    for e in edges.values():
        in_edges.setdefault(e["to_pc"], []).append(e)
    for pc, n in nodes.items():
        incoming = in_edges.get(pc, [])
        if incoming:
            dom_in     = max(incoming, key=lambda e: e["weighted_count"])
            total_in   = sum(e["weighted_count"] for e in incoming)
            n["pred_cf_type"] = dom_in["cf_type"]
            n["pred_bias"]    = dom_in.get("bias", 1.0)
            n["pred_frac"]    = dom_in["weighted_count"] / total_in if total_in > 0 else 1.0
            n["is_self_loop"] = dom_in["from_pc"] == pc
        else:
            n["pred_cf_type"] = "ENTRY"
            n["pred_bias"]    = 1.0
            n["pred_frac"]    = 1.0
            n["is_self_loop"] = False

    # --- stable BBL IDs (sorted by PC value) ---
    for idx, pc in enumerate(sorted(nodes.keys(), key=lambda p: int(p, 16))):
        nodes[pc]["bbl_id"] = idx

    # --- bottleneck scores ---
    _compute_scores(nodes)

    return {
        "metadata": {
            "simpoints":  simpoints_used,
            "total_weight": total_weight,
            "num_nodes":  len(nodes),
            "num_edges":  len(edges),
        },
        "nodes": nodes,
        "edges": edges,
    }


def _compute_scores(nodes: dict) -> None:
    """Add per-node bottleneck scores: score = frequency × avg_<cost>_cycles."""
    for n in nodes.values():
        n["exec_score"]     = n["frequency"] * n.get("avg_exec_cycles",     0.0)
        n["redirect_score"] = n["frequency"] * n.get("avg_redirect_cycles", 0.0)
        n["icache_score"]   = n["frequency"] * n.get("avg_icache_cycles",   0.0)


# ---------------------------------------------------------------------------
# Seed selection and region extraction
# ---------------------------------------------------------------------------

def _effective_focus(nodes: dict, focus: str) -> tuple[str, str | None]:
    """
    Return (effective_focus, fallback_reason).

    If all scores for the requested focus are zero (e.g. because the
    corresponding scarab API is not yet wired up), fall back to 'hot' and
    return a human-readable reason string.  Otherwise return (focus, None).
    """
    if focus == "hot":
        return focus, None
    key = _SCORE_KEY.get(focus, "frequency")
    if all(n.get(key, 0.0) == 0.0 for n in nodes.values()):
        return "hot", (
            f"focus='{focus}' has no data (all scores are 0) — "
            f"falling back to 'hot'"
        )
    return focus, None


def select_seeds(nodes: dict, focus: str, top_n: int) -> list[str]:
    """Return the top-N node start_pcs ranked by the chosen (effective) focus score."""
    eff_focus, _ = _effective_focus(nodes, focus)
    key = _SCORE_KEY.get(eff_focus, "frequency")
    ranked = sorted(nodes.values(), key=lambda n: n.get(key, 0.0), reverse=True)
    return [n["start_pc"] for n in ranked[:top_n]]


def extract_region(nodes: dict, edges: dict,
                   seed_pcs: list[str], depth: int) -> tuple[dict, dict, set]:
    """
    Return (region_nodes, region_edges, seed_set) where region includes
    seed nodes plus up to `depth` hops of predecessors and successors.
    """
    fwd: dict[str, list[str]] = {}
    bwd: dict[str, list[str]] = {}
    for e in edges.values():
        fwd.setdefault(e["from_pc"], []).append(e["to_pc"])
        bwd.setdefault(e["to_pc"],   []).append(e["from_pc"])

    region_pcs = set(seed_pcs)
    frontier   = set(seed_pcs)
    for _ in range(depth):
        next_frontier: set[str] = set()
        for pc in frontier:
            for nb in fwd.get(pc, []) + bwd.get(pc, []):
                if nb not in region_pcs and nb in nodes:
                    region_pcs.add(nb)
                    next_frontier.add(nb)
        frontier = next_frontier

    region_nodes = {pc: nodes[pc] for pc in region_pcs if pc in nodes}
    region_edges = {k: e for k, e in edges.items()
                    if e["from_pc"] in region_pcs and e["to_pc"] in region_pcs}
    return region_nodes, region_edges, set(seed_pcs)


# ---------------------------------------------------------------------------
# Console summary table
# ---------------------------------------------------------------------------

def print_summary_table(nodes: dict, focus: str, top_n: int) -> None:
    eff_focus, fallback_reason = _effective_focus(nodes, focus)
    key    = _SCORE_KEY.get(eff_focus, "frequency")
    ranked = sorted(nodes.values(),
                    key=lambda n: n.get(key, 0.0), reverse=True)[:top_n]

    ind = "               "
    _score_formulas = {
        "hot":      ("score(=freq)",
                     f"score = freq\n{ind}      = Σ(count×w) / Σ_all_nodes(count×w)"),
        "exec":     ("score(=exec×freq)",
                     f"score = avg_exec_cycles × freq\n"
                     f"{ind}      = [Σ(exec_cycle_sum×w) / Σ(count×w)] × freq"),
        "redirect": ("score(=redirect×freq)",
                     f"score = avg_redirect_cycles × freq\n"
                     f"{ind}      = [Σ(redirect_cycle_sum×w) / Σ(pred_count×w)] × freq"),
        "icache":   ("score(=icache×freq)",
                     f"score = avg_icache_cycles × freq\n"
                     f"{ind}      = [Σ(icache_cycle_sum×w) / Σ(count×w)] × freq\n"
                     f"{ind}where icache_cycles = decode_cycle - fetch_cycle (first inst of BBL)"),
    }
    score_label, score_formula = _score_formulas.get(eff_focus, _score_formulas["hot"])

    has_redirect = any(n.get("pred_count", 0) > 0 for n in nodes.values())

    header = (f"{'Rank':>4}  {'BBL#':>6}  {'start_pc':>18}  {'pred_cf':>14}  "
              f"{'bias':>5}  {'frac':>5}  {'exit_cf':>8}  {'freq':>7}  "
              f"{score_label:>22}  {'exec_cy':>8}  "
              f"{'redirect_cy':>11}  {'icache_cy':>9}  {'inst/exec':>9}")
    focus_hdr = eff_focus.upper()
    if fallback_reason:
        focus_hdr += f" (requested: {focus.upper()})"
    print(f"\n  Focus: {focus_hdr}  —  {score_formula}  (top {len(ranked)} of {len(nodes)} nodes)")
    if fallback_reason:
        print(f"  [warn] {fallback_reason}")
    print("  " + header)
    print("  " + "─" * len(header))

    for i, n in enumerate(ranked, 1):
        score        = n.get(key, 0.0)
        exec_cy      = n.get("avg_exec_cycles",     0.0)
        redirect_cy  = n.get("avg_redirect_cycles", 0.0)
        icache_cy    = n.get("avg_icache_cycles",   0.0)
        inst_exec    = n.get("avg_inst_per_exec",   0.0)
        bias         = n.get("pred_bias", 1.0)
        frac         = n.get("pred_frac", 1.0)
        pred_cf      = n.get("pred_cf_type", "?")
        if n.get("is_self_loop"):
            pred_cf += "[loop]"
        redirect_str = f"{redirect_cy:>10.1f}" if has_redirect else f"{'N/A':>10}"
        print(f"  {i:>4}  {n.get('bbl_id', '?'):>6}  {n['start_pc']:>18}  {pred_cf:>14}  "
              f"{bias:>5.2f}  {frac:>5.2f}  {n['cf_type']:>8}  "
              f"{n['frequency']:>7.4f}  {score:>22.4f}  "
              f"{exec_cy:>8.1f}  {redirect_str}  {icache_cy:>9.1f}  {inst_exec:>9.2f}")
    print()


# ---------------------------------------------------------------------------
# DOT export — HTML-label nodes colored by BBL-level interval stats
# ---------------------------------------------------------------------------

def export_dot(cfg: dict, out_path: Path,
               focus: str = "hot", top_n: int = 50,
               region_depth: int = 1) -> None:
    nodes = cfg["nodes"]
    edges = cfg["edges"]

    eff_focus, _ = _effective_focus(nodes, focus)
    seeds = select_seeds(nodes, eff_focus, top_n)
    region_nodes, region_edges, seed_set = extract_region(nodes, edges, seeds, region_depth)

    has_redirect = any(n.get("pred_count", 0) > 0 for n in nodes.values())

    with open(out_path, "w") as f:
        f.write("digraph CFG {\n")
        f.write("  rankdir=TB;\n")
        f.write("  node [shape=none fontsize=9 fontname=\"Courier\"];\n")
        f.write("  edge [fontsize=8 fontname=\"Courier\"];\n\n")

        # --- nodes ---
        for pc, n in sorted(region_nodes.items(),
                             key=lambda x: -x[1].get("frequency", 0.0)):
            is_seed   = pc in seed_set
            border_w  = 3 if is_seed else 1

            exec_cy     = n.get("avg_exec_cycles",     0.0)
            redirect_cy = n.get("avg_redirect_cycles", 0.0)
            icache_cy   = n.get("avg_icache_cycles",   0.0)

            interval_cells = []
            for i, (label, color, text, val, avail) in enumerate(zip(
                INTERVAL_LABELS, INTERVAL_COLORS, INTERVAL_TEXT,
                [exec_cy, redirect_cy, icache_cy],
                [True, has_redirect, True],
            )):
                val_str = f"{val:.1f} cy" if avail else "N/A"
                interval_cells.append(
                    f'<TD BGCOLOR="{color}">'
                    f'<FONT COLOR="{text}"><B>{label}</B><BR/>{val_str}</FONT>'
                    f'</TD>'
                )
            cells = "".join(interval_cells)
            ncols = len(INTERVAL_LABELS)

            pred_cf   = n.get("pred_cf_type", "ENTRY")
            exit_cf   = n.get("cf_type", "?")
            inst_exec = n.get("avg_inst_per_exec", 0.0)
            label = (
                f'<<TABLE BORDER="{border_w}" CELLBORDER="1" CELLSPACING="0">'
                f'<TR><TD COLSPAN="{ncols}" BGCOLOR="#f4f4f4">'
                f'<B>{n["start_pc"]}</B>&#160;&#160;'
                f'freq={n["frequency"]:.4f}</TD></TR>'
                f'<TR><TD COLSPAN="{ncols}" BGCOLOR="#e8e8e8">'
                f'&#x2192; <B>{pred_cf}</B>&#160;&#160;'
                f'<FONT COLOR="#666666">exit: {exit_cf}</FONT></TD></TR>'
                f'<TR>{cells}</TR>'
                f'<TR><TD COLSPAN="{ncols}" BGCOLOR="#f4f4f4">'
                f'{inst_exec:.1f} inst/exec</TD></TR>'
                f'</TABLE>>'
            )
            f.write(f"  {_node_id(pc)} [label={label}];\n")

        f.write("\n")

        # --- edges ---
        max_wcount = max(
            (e["weighted_count"] for e in region_edges.values()), default=1.0
        )
        for e in sorted(region_edges.values(),
                        key=lambda e: -e.get("weighted_count", 0.0)):
            src  = _node_id(e["from_pc"])
            dst  = _node_id(e["to_pc"])
            bias = e.get("bias", 0.0)
            pw   = max(1, round(e["weighted_count"] / max_wcount * 5))
            color = "#cc2222" if 0.3 < bias < 0.7 else "#333333"
            label = f'{e["cf_type"]}\\nbias={bias:.2f}'
            f.write(f'  {src} -> {dst} '
                    f'[label="{label}" penwidth={pw} color="{color}"];\n')

        f.write("}\n")


# ---------------------------------------------------------------------------
# Per-workload driver
# ---------------------------------------------------------------------------

def process_workload(root_dir: Path, experiment: str, config: str,
                     suite: str, subsuite: str, workload: str,
                     workloads_db: dict, output_dir: Path,
                     emit_dot: bool, top_n: int,
                     focus: str, region_depth: int) -> bool:
    files = find_cfg_data_files(root_dir, experiment, config,
                                suite, subsuite, workload)
    if not files:
        print(f"  [skip] No cfg_data.json found for {config}/{workload}")
        return False

    weighted = []
    for cluster_id, path in files:
        weight = get_simpoint_weight(workloads_db, suite, subsuite,
                                     workload, cluster_id)
        if weight is None:
            if cluster_id == 0:
                weight = 1.0  # PT-trace whole-execution run — no simpoints, weight = 1
            else:
                print(f"  [warn] No weight for {workload} cluster {cluster_id}, skipping")
                continue
        weighted.append((cluster_id, weight, path))

    if not weighted:
        print(f"  [skip] No weighted simpoints for {config}/{workload}")
        return False

    cfg = aggregate_cfg(weighted)
    cfg["metadata"].update(experiment=experiment, config=config, workload=workload)

    stem     = f"aggregated_cfg_{config}_{workload}"
    json_out = output_dir / f"{stem}.json"
    with open(json_out, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  [ok] {workload}: {cfg['metadata']['num_nodes']} nodes, "
          f"{cfg['metadata']['num_edges']} edges → {json_out}")

    print_summary_table(cfg["nodes"], focus, top_n)

    if emit_dot:
        dot_out = output_dir / f"{stem}_{focus}_top{top_n}_depth{region_depth}.dot"
        svg_out = dot_out.with_suffix(".svg")
        export_dot(cfg, dot_out, focus=focus, top_n=top_n, region_depth=region_depth)
        print(f"  DOT ({focus}, top {top_n} seeds, depth {region_depth}) → {dot_out}")
        try:
            subprocess.run(["dot", "-Tsvg", str(dot_out), "-o", str(svg_out)],
                           check=True, capture_output=True, timeout=60)
            print(f"  SVG → {svg_out}")
        except FileNotFoundError:
            print(f"  [warn] 'dot' not found — install graphviz to auto-generate SVG")
        except subprocess.TimeoutExpired:
            print(f"  [warn] dot timed out on {dot_out.name} (graph too large) — .dot written, SVG skipped")
        except subprocess.CalledProcessError as exc:
            print(f"  [warn] dot conversion failed: {exc.stderr.decode().strip()}")

    return True


# ---------------------------------------------------------------------------
# Core driver — callable from sci or directly
# ---------------------------------------------------------------------------

def run(descriptor_path: Path, infra_root: Path, output_dir: Path,
        filter_config: str = None, filter_workload: str = None,
        emit_dot: bool = False, top_n: int = 30,
        focus: str = "hot", region_depth: int = 1) -> int:
    """
    Aggregate CFG data for all (config, workload) combinations in the descriptor.

    Returns 0 on success, 1 on error.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        desc = load_descriptor(descriptor_path)
    except (AssertionError, Exception) as exc:
        print(f"[cfg] Failed to read descriptor: {exc}")
        return 1

    root_dir   = Path(desc["root_dir"])
    experiment = desc["experiment"]
    top_sp     = desc.get("top_simpoint", False)

    workloads_db = load_workloads_db(infra_root, top_sp)

    configs = desc.get("configurations", {})
    if filter_config:
        if filter_config not in configs:
            print(f"[cfg] Config {filter_config!r} not found in descriptor.")
            return 1
        configs = {filter_config: configs[filter_config]}

    cfg_configs = {k: v for k, v in configs.items()
                   if "--debug_cfg 1" in v.get("params", "")}
    if not cfg_configs:
        print("[cfg] No configuration has '--debug_cfg 1' in its params.")
        return 1

    all_targets = expand_simulation_workloads(desc.get("simulations", []), workloads_db)
    targets = [(suite, subsuite, wl) for suite, subsuite, wl in all_targets
               if not filter_workload or wl == filter_workload]

    if not targets:
        print("[cfg] No workloads to process.")
        return 1

    print(f"Experiment   : {experiment}")
    print(f"Configs      : {list(cfg_configs.keys())}")
    print(f"Workloads    : {[t[2] for t in targets]}")
    print(f"Focus        : {focus}")
    print(f"Region depth : {region_depth}")
    print(f"Output dir   : {output_dir}")
    print()

    total_ok = 0
    for config in cfg_configs:
        for suite, subsuite, workload in targets:
            print(f"[{config}] {suite}/{subsuite}/{workload}")
            ok = process_workload(root_dir, experiment, config,
                                  suite, subsuite, workload,
                                  workloads_db, output_dir,
                                  emit_dot, top_n, focus, region_depth)
            if ok:
                total_ok += 1

    print(f"Done. Wrote {total_ok} aggregated CFG file(s) to {output_dir}")

    print(f"""
  Output files
  ────────────
  aggregated_cfg_<config>_<workload>.json   — weighted CFG with BBL-level interval stats
  aggregated_cfg_<config>_<workload>.dot    — Graphviz source (only when --dot)
  aggregated_cfg_<config>_<workload>.svg    — rendered graph  (only when --dot)

  Focus modes  (cfg_analysis.focus in descriptor JSON)
  ─────────────────────────────────────────────────────
  hot       rank by execution frequency
  exec      rank by avg execution cost (retire-cycle interval) × freq
  redirect  rank by avg redirect cost (predict-cycle interval;
            predecessor BTB-miss / misprediction-recovery) × freq  (requires cfg_predict_BBL)
  icache    rank by avg icache miss latency (decode_cycle - fetch_cycle, first inst) × freq

  Configuration (set in json/<descriptor>.json under "cfg_analysis")
  ──────────────────────────────────────────────────────────────────
  dot          true/false  emit .dot/.svg graph (default: false)
  focus        hot/exec/redirect/icache  ranking criterion (default: hot)
  top_n        integer     seed nodes in graph (default: 50)
  region_depth integer     hops around seeds in graph (default: 1)

  Example commands
  ────────────────
  ./sci --cfg cfg                  # console table; dot/focus/top_n from cfg_analysis in JSON
""")
    return 0


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--descriptor", "-d", required=True,
                    help="Path to the simulation descriptor JSON (e.g. json/cfg.json)")
    ap.add_argument("--config",      default=None,
                    help="Only process this configuration")
    ap.add_argument("--workload",    default=None,
                    help="Only process this workload")
    ap.add_argument("--output-dir",  default=".",
                    help="Directory to write output files (default: .)")
    ap.add_argument("--dot",         action="store_true",
                    help="Emit a Graphviz .dot file for each workload")
    ap.add_argument("--top-n",       type=int, default=30,
                    help="Seed nodes for DOT graph and summary table (default: 30)")
    ap.add_argument("--focus",       choices=FOCUS_CHOICES, default="hot",
                    help="Ranking criterion: hot | exec | redirect | icache (default: hot)")
    ap.add_argument("--region-depth", type=int, default=1,
                    help="Hops of predecessors/successors to include in DOT (default: 1)")
    args = ap.parse_args()

    descriptor_path = Path(args.descriptor).resolve()
    infra_root      = descriptor_path.parent.parent
    output_dir      = Path(args.output_dir).resolve()

    sys.exit(run(descriptor_path, infra_root, output_dir,
                 filter_config=args.config, filter_workload=args.workload,
                 emit_dot=args.dot, top_n=args.top_n,
                 focus=args.focus, region_depth=args.region_depth))


if __name__ == "__main__":
    main()
