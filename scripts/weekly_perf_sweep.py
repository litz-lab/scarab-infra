#!/usr/bin/python3
"""Weekly SPEC CPU2017 perf-regression sweep: run 3 modes, append metrics to
history.csv, plot them over time, mail the lab.

Metrics per workload (from ./sci --visualize's aggregates.json, so simpoint
weighting lives in one place):
    ipc                 IPC
    mem_latency_cycles  TOTAL_MEM_LATENCY_pct  (a RATIO over L1_FILL, not a %)
    offpath_cycle_pct   ICACHE_CYCLE_OFFPATH / (ONPATH + OFFPATH)
    kips                Cumulative_Instructions / SIM_HOST_WALL_SECONDS_value / 1e3

The two ratios divide separately-weighted aggregates, so they track drift
correctly but are not exact weighted ratios. Don't quote them as paper numbers.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import fcntl
import json
import os
import shutil
import smtplib
import subprocess
import sys
import tempfile
from email.message import EmailMessage
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

MODES: List[Tuple[str, str]] = [
    ("weekly_spec17_memtrace_opt", "memtrace/opt"),
    ("weekly_spec17_memtrace_dbg", "memtrace/dbg"),
    ("weekly_spec17_exec_opt", "exec/opt"),
]

# Own clone so a sweep never collides with anyone's working tree.
SCARAB_DIR = Path("/soe/hlitz/git/scarab-weekly")
SCARAB_REMOTE = "git@github.com:litz-lab/scarab.git"

HISTORY_DIR = Path("/soe/hlitz/git/scarab_weekly")
HISTORY_REMOTE = "git@github.com:litz-lab/scarab_weekly.git"
HISTORY_CSV = "history.csv"
PLOT_SUBDIR = "plots"

LOCK_PATH = Path("/tmp/scarab_weekly_sweep.lock")

SMTP_HOST = "smtp.soe.ucsc.edu"
SMTP_PORT = 25
MAIL_FROM = "scarab-perf@soe.ucsc.edu"
GITHUB_ORG = "litz-lab"

FIELDNAMES = [
    "date",
    "scarab_sha",
    "infra_sha",
    "mode",
    "workload",
    "ipc",
    "mem_latency_cycles",
    "offpath_cycle_pct",
    "kips",
]

# (csv column, title, y label)
PLOTS = [
    ("ipc", "IPC", "IPC"),
    ("mem_latency_cycles", "Average memory latency", "cycles per L1 fill"),
    ("offpath_cycle_pct", "Off-path icache cycles", "% of icache cycles"),
    ("kips", "Simulation speed", "KIPS"),
]


def log(msg: str) -> None:
    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)


def run(cmd: List[str], *, cwd: Optional[Path] = None, check: bool = True,
        capture: bool = False) -> subprocess.CompletedProcess:
    log("$ " + " ".join(cmd))
    return subprocess.run(
        cmd, cwd=str(cwd) if cwd else None, check=check,
        text=True, capture_output=capture,
    )


def git_sha(repo: Path) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(repo),
                             check=True, text=True, capture_output=True).stdout
        return out.strip()
    except subprocess.CalledProcessError:
        return "unknown"


# ---------------------------------------------------------------------------
# Repositories
# ---------------------------------------------------------------------------

def ensure_clone(path: Path, remote: str) -> None:
    if (path / ".git").is_dir():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", remote, str(path)])


def refresh_scarab() -> str:
    ensure_clone(SCARAB_DIR, SCARAB_REMOTE)
    run(["git", "fetch", "origin", "main"], cwd=SCARAB_DIR)
    # Reset, not pull: a stray local commit must not change what we measure.
    run(["git", "reset", "--hard", "origin/main"], cwd=SCARAB_DIR)
    run(["git", "submodule", "update", "--init", "--recursive"], cwd=SCARAB_DIR, check=False)
    return git_sha(SCARAB_DIR)


# ---------------------------------------------------------------------------
# Running one mode
# ---------------------------------------------------------------------------

def render_descriptor(stem: str, experiment: str) -> str:
    """Dated copy of a descriptor; sci refuses to re-run an existing experiment."""
    src = REPO_ROOT / "json" / f"{stem}.json"
    descriptor = json.loads(src.read_text(encoding="utf-8"))
    descriptor["experiment"] = experiment
    dst_stem = experiment
    dst = REPO_ROOT / "json" / f"{dst_stem}.json"
    dst.write_text(json.dumps(descriptor, indent=2, separators=(",", ":")) + "\n",
                   encoding="utf-8")
    return dst_stem


def sci(args: List[str], *, check: bool = True) -> bool:
    result = run([str(REPO_ROOT / "sci")] + args, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        log(f"WARNING: sci {' '.join(args)} exited {result.returncode}")
        if check:
            return False
    return True


def experiment_dir(descriptor_stem: str) -> Path:
    descriptor = json.loads((REPO_ROOT / "json" / f"{descriptor_stem}.json")
                            .read_text(encoding="utf-8"))
    root = Path(os.path.expandvars(str(descriptor["root_dir"])))
    return root / "simulations" / descriptor["experiment"]


def run_mode(stem: str, experiment: str, *, skip_sim: bool) -> Optional[Path]:
    """Run one mode; return its aggregates.json."""
    descriptor_stem = render_descriptor(stem, experiment)
    exp_dir = experiment_dir(descriptor_stem)

    if not skip_sim:
        if not sci(["--build-scarab", descriptor_stem]):
            return None
        if not sci(["--sim", descriptor_stem]):
            return None
        if not sci(["--collect-stats", descriptor_stem]):
            return None

    if not (exp_dir / "collected_stats.csv").is_file():
        log(f"ERROR: no collected_stats.csv in {exp_dir}")
        return None

    sci(["--visualize", descriptor_stem], check=False)
    aggregates = exp_dir / "aggregates.json"
    if not aggregates.is_file():
        log(f"ERROR: --visualize produced no aggregates.json in {exp_dir}")
        return None
    return aggregates


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _get(aggregates: Dict, stat: str, config: str, workload: str) -> Optional[float]:
    try:
        value = aggregates[stat][config][workload]
    except (KeyError, TypeError):
        return None
    return value if isinstance(value, (int, float)) else None


def derive_metrics(aggregates_path: Path) -> Dict[str, Dict[str, Optional[float]]]:
    """{workload: {metric: value}}, including 'Avg'."""
    payload = json.loads(aggregates_path.read_text(encoding="utf-8"))
    aggregates = payload["aggregates"]
    config = payload["baseline"]
    workloads = list(payload["workloads"]) + ["Avg"]

    out: Dict[str, Dict[str, Optional[float]]] = {}
    for wl in workloads:
        ipc = _get(aggregates, "IPC", config, wl)
        latency = _get(aggregates, "TOTAL_MEM_LATENCY_pct", config, wl)
        on = _get(aggregates, "ICACHE_CYCLE_ONPATH_count", config, wl)
        off = _get(aggregates, "ICACHE_CYCLE_OFFPATH_count", config, wl)
        inst = _get(aggregates, "Cumulative_Instructions", config, wl)
        wall = _get(aggregates, "SIM_HOST_WALL_SECONDS_value", config, wl)

        offpath = None
        if on is not None and off is not None and (on + off) > 0:
            offpath = 100.0 * off / (on + off)
        kips = None
        if inst is not None and wall is not None and wall > 0:
            kips = inst / wall / 1000.0

        out[wl] = {
            "ipc": ipc,
            "mem_latency_cycles": latency,
            "offpath_cycle_pct": offpath,
            "kips": kips,
        }
    return out


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

def append_history(rows: List[Dict[str, object]]) -> Path:
    ensure_clone(HISTORY_DIR, HISTORY_REMOTE)
    run(["git", "pull", "--ff-only"], cwd=HISTORY_DIR, check=False)
    path = HISTORY_DIR / HISTORY_CSV
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
    log(f"appended {len(rows)} rows to {path}")
    return path


def read_history(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_plots(history: List[Dict[str, str]], outdir: Path) -> List[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    modes = [label for _, label in MODES]

    for column, title, ylabel in PLOTS:
        fig, axes = plt.subplots(
            len(modes), 1, figsize=(13, 4.2 * len(modes)), sharex=True, squeeze=False
        )
        drew_anything = False

        for ax, mode in zip((a[0] for a in axes), modes):
            subset = [r for r in history if r["mode"] == mode]
            dates = sorted({r["date"] for r in subset})
            if not dates:
                ax.set_title(f"{title} — {mode} (no data)")
                ax.set_ylabel(ylabel)
                continue

            by_workload: Dict[str, Dict[str, float]] = {}
            for row in subset:
                try:
                    value = float(row[column])
                except (TypeError, ValueError):
                    continue
                by_workload.setdefault(row["workload"], {})[row["date"]] = value

            # 36 apps: thin/translucent context, no per-app legend.
            for workload, series in sorted(by_workload.items()):
                if workload == "Avg":
                    continue
                xs = [d for d in dates if d in series]
                if len(xs) < 1:
                    continue
                ax.plot(xs, [series[d] for d in xs], linewidth=0.8, alpha=0.35)
                drew_anything = True

            avg = by_workload.get("Avg", {})
            xs = [d for d in dates if d in avg]
            if xs:
                ax.plot(xs, [avg[d] for d in xs], color="black", linewidth=2.6,
                        marker="o", markersize=4, label="Average", zorder=5)
                ax.legend(loc="best", fontsize=9)
                drew_anything = True

            ax.set_title(f"{title} — {mode}")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            for tick in ax.get_xticklabels():
                tick.set_rotation(45)
                tick.set_horizontalalignment("right")

        if not drew_anything:
            plt.close(fig)
            continue

        fig.suptitle(f"{title} over time — SPEC CPU2017", fontsize=13)
        fig.tight_layout()
        out = outdir / f"{column}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        written.append(out)
        log(f"wrote plot {out}")

    return written


# ---------------------------------------------------------------------------
# Report + email
# ---------------------------------------------------------------------------

def latest_two_dates(history: List[Dict[str, str]]) -> Tuple[Optional[str], Optional[str]]:
    dates = sorted({r["date"] for r in history})
    if not dates:
        return None, None
    return (dates[-2] if len(dates) > 1 else None), dates[-1]


def build_report(history: List[Dict[str, str]], scarab_sha: str, infra_sha: str) -> str:
    previous, current = latest_two_dates(history)
    lines = [
        f"SPEC CPU2017 weekly sweep — {current}",
        f"scarab {scarab_sha}, scarab-infra {infra_sha}",
        "",
    ]
    def avg_for(mode: str, date: Optional[str], column: str) -> Optional[float]:
        if date is None:
            return None
        for row in history:
            if row["mode"] == mode and row["date"] == date and row["workload"] == "Avg":
                try:
                    return float(row[column])
                except (TypeError, ValueError):
                    return None
        return None

    def drift(mode: str, column: str) -> Optional[float]:
        now, was = avg_for(mode, current, column), avg_for(mode, previous, column)
        if now is None or was in (None, 0):
            return None
        return (now / was - 1.0) * 100.0

    # Headline first: the whole point is seeing drift without opening a plot.
    if previous:
        lines.append(f"Average drift vs previous run ({previous}):")
        for _, mode in MODES:
            parts = []
            for column, _, _ in PLOTS:
                d = drift(mode, column)
                parts.append(f"{column}={'n/a' if d is None else f'{d:+.2f}%'}")
            lines.append(f"  {mode:<14} " + "  ".join(parts))
        flagged = [
            f"{mode} {column} {drift(mode, column):+.2f}%"
            for _, mode in MODES for column, _, _ in PLOTS
            if drift(mode, column) is not None and abs(drift(mode, column)) >= 2.0
        ]
        lines.append("")
        lines.append("  >2% moves: " + (", ".join(flagged) if flagged else "none"))
    else:
        lines.append("First recorded run; no previous data to compare against.")
    lines.append("")

    for _, mode in MODES:
        lines.append(f"== {mode} ==")

        any_row = False
        for column, title, unit in PLOTS:
            now = avg_for(mode, current, column)
            was = avg_for(mode, previous, column)
            if now is None:
                continue
            any_row = True
            if was is None or was == 0:
                lines.append(f"  {title:<26} {now:>12.4g}  ({unit})")
            else:
                delta = (now / was - 1.0) * 100.0
                lines.append(
                    f"  {title:<26} {now:>12.4g}  ({unit})   "
                    f"{delta:+.2f}% vs previous"
                )
        if not any_row:
            lines.append("  no data")
        lines.append("")

    lines.append("Plots and history.csv: https://github.com/litz-lab/scarab_weekly")
    return "\n".join(lines)


def org_recipients() -> List[str]:
    """Org members' public GitHub emails, so joiners/leavers need no code change."""
    try:
        members = subprocess.run(
            ["gh", "api", f"orgs/{GITHUB_ORG}/members", "--jq", ".[].login"],
            check=True, text=True, capture_output=True,
        ).stdout.split()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        log(f"WARNING: could not list org members: {exc}")
        return []

    addresses: List[str] = []
    for login in members:
        try:
            email = subprocess.run(
                ["gh", "api", f"users/{login}", "--jq", ".email"],
                check=True, text=True, capture_output=True,
            ).stdout.strip()
        except subprocess.CalledProcessError:
            continue
        if email and email != "null":
            addresses.append(email)
        else:
            log(f"  {login}: no public email, skipping")
    return addresses


def send_email(subject: str, body: str, attachments: List[Path],
               recipients: List[str]) -> None:
    if not recipients:
        log("no recipients resolved; skipping email")
        return

    for recipient in recipients:
        message = EmailMessage()
        message["From"] = MAIL_FROM
        message["To"] = recipient
        message["Subject"] = subject
        message.set_content(body)
        for path in attachments:
            try:
                message.add_attachment(
                    path.read_bytes(), maintype="image", subtype="png",
                    filename=path.name,
                )
            except OSError as exc:
                log(f"WARNING: could not attach {path}: {exc}")
        try:
            # One msg per recipient: relay rejects multi-domain transactions.
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as smtp:
                smtp.send_message(message)
            log(f"emailed {recipient}")
        except Exception as exc:  # noqa: BLE001 - report and continue
            log(f"WARNING: could not email {recipient}: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the report; don't write, push or email.")
    parser.add_argument("--skip-sim", action="store_true",
                        help="Reuse existing experiment dirs instead of simulating.")
    parser.add_argument("--no-email", action="store_true")
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--experiment-suffix", default=None,
                        help="Override the dated experiment suffix (testing).")
    args = parser.parse_args()

    lock = LOCK_PATH.open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log("another sweep is already running; exiting")
        return 0

    today = _dt.date.today().isoformat()
    suffix = args.experiment_suffix or today.replace("-", "")
    infra_sha = git_sha(REPO_ROOT)
    scarab_sha = "skipped" if args.skip_sim else refresh_scarab()
    if args.skip_sim:
        scarab_sha = git_sha(SCARAB_DIR) if SCARAB_DIR.is_dir() else "unknown"

    log(f"weekly sweep {today}: scarab={scarab_sha} infra={infra_sha}")

    rows: List[Dict[str, object]] = []
    for stem, label in MODES:
        experiment = f"{stem}_{suffix}"
        log(f"--- {label} ({experiment}) ---")
        aggregates_path = run_mode(stem, experiment, skip_sim=args.skip_sim)
        if aggregates_path is None:
            log(f"{label}: no results, continuing with the other modes")
            continue
        metrics = derive_metrics(aggregates_path)
        for workload, values in metrics.items():
            rows.append({
                "date": today,
                "scarab_sha": scarab_sha,
                "infra_sha": infra_sha,
                "mode": label,
                "workload": workload,
                **{k: ("" if v is None else f"{v:.6g}") for k, v in values.items()},
            })

    if not rows:
        log("ERROR: every mode failed; nothing to record")
        return 1

    if args.dry_run:
        history = read_history(HISTORY_DIR / HISTORY_CSV)
        history = history + [{k: str(v) for k, v in row.items()} for row in rows]
        print()
        print(build_report(history, scarab_sha, infra_sha))
        return 0

    csv_path = append_history(rows)
    history = read_history(csv_path)
    plots = make_plots(history, HISTORY_DIR / PLOT_SUBDIR)
    report = build_report(history, scarab_sha, infra_sha)
    print()
    print(report)

    if not args.no_push:
        run(["git", "add", HISTORY_CSV, PLOT_SUBDIR], cwd=HISTORY_DIR, check=False)
        run(["git", "-c", "user.name=scarab-weekly",
             "-c", "user.email=scarab-perf@soe.ucsc.edu",
             "commit", "-m", f"Weekly SPEC17 sweep {today} (scarab {scarab_sha})"],
            cwd=HISTORY_DIR, check=False)
        run(["git", "push"], cwd=HISTORY_DIR, check=False)

    if not args.no_email:
        send_email(f"SPEC17 weekly perf sweep — {today}", report, plots,
                   org_recipients())

    return 0


if __name__ == "__main__":
    sys.exit(main())
