#!/usr/bin/env python3
"""Generate IPC S-curves from a scarab-infra collected_stats.csv file."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DATA_START_COL = 3
DEFAULT_OUTDIR = "ipc_scurves"
GRAPH_NAME = "ipc_scurve_graph.pdf"
DATA_NAME = "ipc_scurve_data.csv"


def _safe_filename(stem: str) -> str:
    return "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in stem)


def _ordered_unique(values: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _find_row(df: pd.DataFrame, names: Sequence[str]) -> Optional[pd.Series]:
    stats = df["stats"].astype(str)
    lowered = {name.lower() for name in names}
    matches = df.loc[stats.str.lower().isin(lowered)]
    if matches.empty:
        return None
    return matches.iloc[0]


def _row_values(df: pd.DataFrame, names: Sequence[str], data_cols: Sequence[str]) -> Optional[List[object]]:
    row = _find_row(df, names)
    if row is None:
        return None
    return list(row.loc[list(data_cols)])


def _as_float(value: object) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _parse_colname(colname: str) -> Optional[Tuple[str, str, str]]:
    parts = str(colname).strip().split()
    if len(parts) < 3:
        return None
    config = parts[0]
    simpoint = parts[-1]
    workload = " ".join(parts[1:-1])
    return config, workload, simpoint


def _load_column_metadata(df: pd.DataFrame, data_cols: Sequence[str]) -> List[Tuple[str, str, str]]:
    configs = _row_values(df, ["Configuration"], data_cols)
    workloads = _row_values(df, ["Workload"], data_cols)
    simpoints = _row_values(df, ["Cluster Id", "Cluster ID", "cluster_id"], data_cols)

    if configs is not None and workloads is not None and simpoints is not None:
        return [
            (str(configs[i]), str(workloads[i]), str(simpoints[i]))
            for i in range(len(data_cols))
        ]

    parsed: List[Tuple[str, str, str]] = []
    for col in data_cols:
        item = _parse_colname(col)
        if item is None:
            raise ValueError(
                "Could not find collected-stats metadata rows and could not parse "
                f"column name: {col!r}"
            )
        parsed.append(item)
    return parsed


def _load_ipc_by_config(
    csv_path: Path,
) -> Tuple[Dict[str, Dict[Tuple[str, str], float]], List[str]]:
    df = pd.read_csv(csv_path, low_memory=False)
    if "stats" not in df.columns:
        raise ValueError(f"Input CSV must contain a 'stats' column: {csv_path}")

    data_cols = list(df.columns[DATA_START_COL:])
    if not data_cols:
        raise ValueError(f"Input CSV has no simpoint data columns: {csv_path}")

    instr = _row_values(
        df,
        ["Periodic_Instructions", "Periodic Instructions", "PeriodicInstructions"],
        data_cols,
    )
    cycles = _row_values(
        df,
        ["Periodic_Cycles", "Periodic Cycles", "PeriodicCycles"],
        data_cols,
    )
    if instr is None or cycles is None:
        raise ValueError("Missing required Periodic_Instructions and/or Periodic_Cycles rows.")

    failed = _row_values(df, ["Collect Failed"], data_cols)
    metadata = _load_column_metadata(df, data_cols)

    ipc_by_config: Dict[str, Dict[Tuple[str, str], float]] = {}
    configs_in_order: List[str] = []
    for i, (config, workload, simpoint) in enumerate(metadata):
        if failed is not None and _as_bool(failed[i]):
            continue
        inst = _as_float(instr[i])
        cyc = _as_float(cycles[i])
        if inst is None or cyc is None or cyc == 0:
            continue
        if config not in ipc_by_config:
            ipc_by_config[config] = {}
            configs_in_order.append(config)
        ipc_by_config[config][(workload, simpoint)] = inst / cyc

    return ipc_by_config, configs_in_order


def _build_pair_rows(
    ipc_by_config: Dict[str, Dict[Tuple[str, str], float]],
    baseline: str,
    candidate: str,
) -> List[Dict[str, object]]:
    baseline_ipc = ipc_by_config.get(baseline, {})
    candidate_ipc = ipc_by_config.get(candidate, {})
    common = sorted(set(baseline_ipc) & set(candidate_ipc))

    rows: List[Dict[str, object]] = []
    for workload, simpoint in common:
        base = baseline_ipc[(workload, simpoint)]
        cand = candidate_ipc[(workload, simpoint)]
        if base == 0:
            continue
        pct = (cand - base) / base * 100.0
        rows.append(
            {
                "workload": workload,
                "simpoint": simpoint,
                f"ipc_{baseline}": base,
                f"ipc_{candidate}": cand,
                f"percent_change({candidate}_vs_{baseline})": pct,
            }
        )

    pct_col = f"percent_change({candidate}_vs_{baseline})"
    rows.sort(key=lambda row: float(row[pct_col]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def _write_pair_csv(rows: List[Dict[str, object]], csv_path: Path) -> None:
    if not rows:
        return
    fieldnames = ["rank"] + [name for name in rows[0].keys() if name != "rank"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_pair_plot(
    rows: List[Dict[str, object]],
    baseline: str,
    candidate: str,
    graph_path: Path,
) -> None:
    pct_col = f"percent_change({candidate}_vs_{baseline})"
    ranks = [int(row["rank"]) for row in rows]
    deltas = [float(row[pct_col]) for row in rows]

    fig, ax = plt.subplots(figsize=(5.2, 3.0), dpi=300, constrained_layout=True)
    ax.plot(
        ranks,
        deltas,
        linewidth=1.6,
        color="tab:blue",
    )
    ax.axhline(0.0, linestyle="--", linewidth=0.8, color="tab:gray")
    ax.set_title(f"IPC S-curve: {candidate} vs {baseline}", fontsize=10, pad=6)
    ax.set_xlabel(f"Simpoints sorted by IPC change (n={len(rows)})", fontsize=9, labelpad=6)
    ax.set_ylabel("Relative IPC change (%)", fontsize=9, labelpad=6)
    ax.tick_params(axis="both", labelsize=8)
    ax.margins(x=0.01)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
    fig.savefig(graph_path, format="pdf")
    plt.close(fig)


def generate_ipc_scurves(
    csv_path: Path,
    baseline_config: Optional[str] = None,
    configs: Optional[Sequence[str]] = None,
    outdir: Optional[Path] = None,
) -> List[Dict[str, object]]:
    """Generate one IPC S-curve for baseline_config vs each other config."""
    csv_path = Path(csv_path)
    outdir = Path(outdir) if outdir is not None else csv_path.parent / DEFAULT_OUTDIR

    ipc_by_config, configs_in_order = _load_ipc_by_config(csv_path)
    if configs:
        requested = [str(config) for config in configs if str(config) in ipc_by_config]
    else:
        requested = list(configs_in_order)

    requested = _ordered_unique(requested)
    if len(requested) < 2:
        return []

    baseline = baseline_config if baseline_config in requested else requested[0]
    candidates = [config for config in requested if config != baseline]
    outputs: List[Dict[str, object]] = []

    for candidate in candidates:
        rows = _build_pair_rows(ipc_by_config, baseline, candidate)
        if not rows:
            continue

        pair_dir = outdir / f"{_safe_filename(baseline)}_vs_{_safe_filename(candidate)}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        graph_path = pair_dir / GRAPH_NAME
        data_path = pair_dir / DATA_NAME

        _write_pair_csv(rows, data_path)
        _write_pair_plot(rows, baseline, candidate, graph_path)

        outputs.append(
            {
                "baseline": baseline,
                "candidate": candidate,
                "points": len(rows),
                "graph": graph_path,
                "data": data_path,
            }
        )

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate per-simpoint IPC S-curves from collected_stats.csv."
    )
    parser.add_argument("--input", required=True, help="Path to collected_stats.csv.")
    parser.add_argument("--outdir", default=None, help="Output directory.")
    parser.add_argument("--baseline", default=None, help="Baseline configuration name.")
    parser.add_argument(
        "--configs",
        nargs="*",
        default=None,
        help="Optional configuration subset/order.",
    )
    args = parser.parse_args()

    outputs = generate_ipc_scurves(
        Path(args.input),
        baseline_config=args.baseline,
        configs=args.configs,
        outdir=Path(args.outdir) if args.outdir else None,
    )
    if not outputs:
        print("No IPC S-curves generated.")
        return
    for item in outputs:
        print(
            f"{item['baseline']} vs {item['candidate']}: "
            f"{item['graph']} {item['data']} ({item['points']} simpoints)"
        )


if __name__ == "__main__":
    main()
