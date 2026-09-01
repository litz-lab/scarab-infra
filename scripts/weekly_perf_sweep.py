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
import getpass
import json
import math
import os
import shutil
import smtplib
import subprocess
import sys
import tempfile
import time
import traceback
from email.message import EmailMessage
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

MODES: List[Tuple[str, str]] = [
    ("weekly_spec17_memtrace_opt", "memtrace/opt"),
    ("weekly_spec17_memtrace_dbg", "memtrace/dbg"),
    ("weekly_spec17_exec_opt", "exec/opt"),
]

# Own clones so a sweep never collides with anyone's working tree, and never
# tests whatever branch someone happened to leave checked out.
SCARAB_DIR = Path("/soe/hlitz/git/scarab-weekly")
SCARAB_REMOTE = "git@github.com:litz-lab/scarab.git"

INFRA_DIR = Path("/soe/hlitz/git/scarab-infra-weekly")
INFRA_REMOTE = "git@github.com:litz-lab/scarab-infra.git"

HISTORY_DIR = Path("/soe/hlitz/git/scarab_weekly")
HISTORY_REMOTE = "git@github.com:litz-lab/scarab_weekly.git"
HISTORY_CSV = "history.csv"
PLOT_SUBDIR = "plots"

LOCK_PATH = Path("/tmp/scarab_weekly_sweep.lock")
# Where the cron entry sends this script's output; quoted in the failure mail.
LOG_PATH = Path("/soe/hlitz/logs/weekly_perf_sweep.log")

POLL_SECONDS = 300
# A full SPEC17 sweep behind a busy queue; past this we take what finished.
SIM_TIMEOUT_SECONDS = 24 * 3600

SMTP_HOST = "smtp.soe.ucsc.edu"
SMTP_PORT = 25
MAIL_FROM = "scarab-perf@soe.ucsc.edu"
GITHUB_ORG = "litz-lab"
# Used when the org lookup comes up empty (no gh on cron's PATH, no token, no
# public emails). Mailing one person beats mailing nobody.
FALLBACK_RECIPIENTS = ["hlitz@ucsc.edu"]

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


# Set once per run; prepended to every mail so a stale checkout is never a
# silent explanation for a weird result.
_INFRA_WARNING = ""


def infra_warning() -> str:
    """Warn when the checkout we ended up in is not main, or is behind it.

    Normally refresh_infra() has already re-execed us from a clean main, so
    this is empty. It fires under --no-self-update, which is how a hand-run
    sweep from a feature branch says so in its own mail.
    """
    run(["git", "fetch", "origin", "main"], cwd=REPO_ROOT, check=False)

    def git(*args: str) -> Optional[str]:
        try:
            return subprocess.run(["git", *args], cwd=str(REPO_ROOT), check=True,
                                  text=True, capture_output=True).stdout.strip()
        except subprocess.CalledProcessError:
            return None

    head = git("rev-parse", "--short", "HEAD")
    target = git("rev-parse", "--short", "origin/main")
    behind = git("rev-list", "--count", "HEAD..origin/main")
    dirty = git("status", "--porcelain")
    if head is None or target is None:
        return ""
    if head == target and not dirty:
        return ""
    detail = [f"checkout {head}, origin/main {target}"]
    if behind and behind != "0":
        detail.append(f"{behind} commits behind")
    if dirty:
        detail.append("uncommitted changes")
    return (f"WARNING: scarab-infra at {REPO_ROOT} is not main ("
            + "; ".join(detail) + ").\n"
            "The images and run scripts under test are not the ones on main.")


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


def refresh_infra() -> None:
    """Re-run ourselves from a fresh scarab-infra main, the way CI does.

    Both repos have to be current: the images, run scripts and descriptors come
    from infra, the simulator from Scarab. Pairing a fresh Scarab with whatever
    infra the sweep happened to be launched from is how 2026-08-30 built a
    PIN-3.31 Scarab inside a PIN-3.15 image and reported nothing.
    """
    ensure_clone(INFRA_DIR, INFRA_REMOTE)
    run(["git", "fetch", "origin", "main"], cwd=INFRA_DIR)
    run(["git", "reset", "--hard", "origin/main"], cwd=INFRA_DIR)

    script = INFRA_DIR / "scripts" / Path(__file__).name
    if not script.is_file():
        log(f"WARNING: {script} missing; continuing from {REPO_ROOT}")
        return
    # The version on main has to understand the flag that stops it re-execing
    # in turn; without that check an older one either dies on an unknown
    # argument (no mail, argparse exits before our handler) or loops forever.
    if "--no-self-update" not in script.read_text(encoding="utf-8", errors="replace"):
        log(f"WARNING: {script} predates --no-self-update; "
            f"continuing from {REPO_ROOT}")
        return
    log(f"running from {INFRA_DIR} at {git_sha(INFRA_DIR)}")
    # exec, not import: the descriptors, sci and run scripts of this run must
    # all come from the tree we just reset.
    os.execv(sys.executable,
             [sys.executable, str(script), "--no-self-update", *sys.argv[1:]])


def refresh_scarab() -> str:
    ensure_clone(SCARAB_DIR, SCARAB_REMOTE)
    run(["git", "fetch", "origin", "main"], cwd=SCARAB_DIR)
    # Reset, not pull: a stray local commit must not change what we measure.
    run(["git", "reset", "--hard", "origin/main"], cwd=SCARAB_DIR)
    # And clean the PIN tool: its objects keep .d files naming the PIN path of
    # the image that built them, so a container upgrade turns into "No rule to
    # make target /tmp_home/pin-3.15-.../algorithm" until they are gone. Only
    # src/pin -- wiping the whole tree would rebuild DynamoRIO every week.
    run(["git", "clean", "-xfd", "--", "src/pin"], cwd=SCARAB_DIR, check=False)
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


def sims_pending(experiment: str) -> bool:
    """Any of this experiment's Slurm jobs still queued or running?"""
    try:
        names = subprocess.run(
            ["squeue", "-u", getpass.getuser(), "-h", "-o", "%j"],
            check=True, text=True, capture_output=True,
        ).stdout.split()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        log(f"WARNING: could not query squeue ({exc}); not waiting")
        return False
    return any(experiment in name for name in names)


def wait_for_sims(experiment: str) -> None:
    """`--sim` returns once Slurm has the jobs; the results land hours later.

    Collecting stats before then reads a half-empty experiment dir and dies on
    the first missing ramulator.stat.out, which is what made every sweep so far
    record nothing.
    """
    deadline = time.monotonic() + SIM_TIMEOUT_SECONDS
    waited = False
    while sims_pending(experiment):
        if time.monotonic() > deadline:
            log(f"{experiment}: jobs still queued after "
                f"{SIM_TIMEOUT_SECONDS // 3600}h; collecting what finished")
            return
        if not waited:
            log(f"{experiment}: waiting for Slurm jobs to finish")
            waited = True
        time.sleep(POLL_SECONDS)
    if waited:
        log(f"{experiment}: all jobs finished")


def run_mode(stem: str, experiment: str, *, skip_sim: bool) -> Optional[Path]:
    """Run one mode; return its aggregates.json."""
    descriptor_stem = render_descriptor(stem, experiment)
    exp_dir = experiment_dir(descriptor_stem)

    if not skip_sim:
        if not sci(["--build-scarab", descriptor_stem]):
            return None
        if not sci(["--sim", descriptor_stem]):
            return None
        wait_for_sims(experiment)
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

def short_name(workload: str) -> str:
    """spec2017/rate_int_v2/gcc_r_2 -> gcc_r_2, so the legend stays narrow."""
    return workload.rsplit("/", 1)[-1]


def app_color(i: int):
    import matplotlib.pyplot as plt
    # tab20+tab20b+tab20c = 60 distinct colors, enough for SPEC17 without reuse.
    palette = [c for name in ("tab20", "tab20b", "tab20c")
               for c in plt.get_cmap(name).colors]
    return palette[i % len(palette)]


def make_plots(history: List[Dict[str, str]], outdir: Path) -> List[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    modes = [label for _, label in MODES]

    for column, title, ylabel in PLOTS:
        fig, axes = plt.subplots(
            len(modes), 1, figsize=(15, 4.6 * len(modes)), sharex=True, squeeze=False
        )
        drew_anything = False
        legend_handles, legend_labels = [], []

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

            apps = [w for w in sorted(by_workload) if w != "Avg"]
            for i, workload in enumerate(apps):
                series = by_workload[workload]
                xs = [d for d in dates if d in series]
                if not xs:
                    continue
                ax.plot(xs, [series[d] for d in xs], linewidth=1.0, alpha=0.75,
                        color=app_color(i), label=short_name(workload))
                drew_anything = True

            avg = by_workload.get("Avg", {})
            xs = [d for d in dates if d in avg]
            if xs:
                ax.plot(xs, [avg[d] for d in xs], color="black", linewidth=2.8,
                        marker="o", markersize=4, label="Average", zorder=5)
                drew_anything = True


            if not legend_handles:
                legend_handles, legend_labels = ax.get_legend_handles_labels()
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
        fig.tight_layout(rect=(0, 0, 0.79, 0.97))
        # One shared legend: the same 36 workloads appear in every subplot.
        if legend_handles:
            fig.legend(legend_handles, legend_labels, loc="center left",
                       bbox_to_anchor=(0.80, 0.5), fontsize=8, ncol=2,
                       frameon=False, handlelength=1.8, labelspacing=0.4)
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
                    value = float(row[column])
                except (TypeError, ValueError):
                    return None
                # A failed simpoint can leave nan in the CSV; never report it.
                return value if math.isfinite(value) else None
        return None

    def drift(mode: str, column: str) -> Optional[float]:
        now, was = avg_for(mode, current, column), avg_for(mode, previous, column)
        if now is None or was is None or was == 0:
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


def notify(subject: str, body: str, attachments: List[Path], args) -> None:
    """Mail the lab, whatever happened.

    The sweep exists to say something once a week. A week that produced no
    mail is indistinguishable from a week nobody looked at, which is how it sat
    broken for two runs -- so failures, crashes and skips all mail too.
    """
    if args.no_email or args.dry_run:
        log(f"not mailing '{subject}' (--no-email/--dry-run)")
        return
    if _INFRA_WARNING:
        body = f"{_INFRA_WARNING}\n\n{body}"
    recipients = org_recipients() or FALLBACK_RECIPIENTS
    send_email(subject, body, attachments, recipients)


def send_email(subject: str, body: str, attachments: List[Path],
               recipients: List[str]) -> None:
    if not recipients:
        log("no recipients resolved; skipping email")
        return

    delivered = 0
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
            delivered += 1
        except Exception as exc:  # noqa: BLE001 - report and continue
            log(f"WARNING: could not email {recipient}: {exc}")

    if not delivered:
        # Nothing else can carry the news out of here; make the log say it.
        log(f"ERROR: '{subject}' reached nobody ({len(recipients)} recipients "
            f"tried via {SMTP_HOST}:{SMTP_PORT})")


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
    parser.add_argument("--no-self-update", action="store_true",
                        help="Run this checkout as-is instead of re-execing "
                             "from a fresh scarab-infra main.")
    args = parser.parse_args()

    today = _dt.date.today().isoformat()
    if not args.no_self_update and not args.dry_run:
        refresh_infra()  # never returns: re-execs from INFRA_DIR
    try:
        return run_sweep(args, today)
    except Exception:  # noqa: BLE001 - any crash must still reach the lab
        report = traceback.format_exc()
        log("CRASHED:\n" + report)
        notify(f"SPEC17 weekly perf sweep CRASHED - {today}",
               f"The sweep raised before it could report.\n\n{report}\n"
               f"Log: {LOG_PATH}",
               [], args)
        return 1


def run_sweep(args, today: str) -> int:
    global _INFRA_WARNING
    _INFRA_WARNING = infra_warning()
    if _INFRA_WARNING:
        log(_INFRA_WARNING)

    lock = LOCK_PATH.open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log("another sweep is already running; exiting")
        notify(f"SPEC17 weekly perf sweep SKIPPED - {today}",
               "Another sweep still held the lock, so this run did nothing.\n"
               f"Log: {LOG_PATH}",
               [], args)
        return 0

    suffix = args.experiment_suffix or today.replace("-", "")
    infra_sha = git_sha(REPO_ROOT)
    scarab_sha = "skipped" if args.skip_sim else refresh_scarab()
    if args.skip_sim:
        scarab_sha = git_sha(SCARAB_DIR) if SCARAB_DIR.is_dir() else "unknown"

    log(f"weekly sweep {today}: scarab={scarab_sha} infra={infra_sha}")

    rows: List[Dict[str, object]] = []
    failures: List[str] = []
    for stem, label in MODES:
        experiment = f"{stem}_{suffix}"
        log(f"--- {label} ({experiment}) ---")
        aggregates_path = run_mode(stem, experiment, skip_sim=args.skip_sim)
        if aggregates_path is None:
            log(f"{label}: no results, continuing with the other modes")
            failures.append(f"{label} ({experiment}): no results")
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
        # Say so out loud: a silent week is indistinguishable from a healthy
        # one, which is how the sweep sat broken without anyone noticing.
        log("ERROR: every mode failed; nothing to record")
        body = "\n".join(
            [f"SPEC CPU2017 weekly sweep {today} recorded nothing.",
             f"scarab {scarab_sha}, scarab-infra {infra_sha}",
             ""] + failures + ["", f"Log: {LOG_PATH}"]
        )
        print()
        print(body)
        notify(f"SPEC17 weekly perf sweep FAILED - {today}", body, [], args)
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

    # A run where some modes died still reports, but says so in the subject.
    if failures:
        report += "\n\nModes that produced nothing:\n" + "\n".join(
            f"  {f}" for f in failures) + f"\nLog: {LOG_PATH}"
    suffix_note = " (partial)" if failures else ""
    notify(f"SPEC17 weekly perf sweep{suffix_note} - {today}", report, plots, args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
