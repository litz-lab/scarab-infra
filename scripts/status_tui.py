#!/usr/bin/env python3
"""Interactive (TUI) front-end for ``./sci --status <DESCRIPTOR>``.

Renders the same summary table that ``--status`` prints, but every count cell is
selectable: pick a (configuration, state) cell and you get the individual
simpoints behind it, with live progress parsed from each simpoint's
``scarab.out`` heartbeat.

Classification is *not* reimplemented here -- it comes from
``print_simulation_status_summary(..., record_sink=...)`` so the TUI and the
plain CLI can never disagree about what "Failed" means.

Requires ``textual``:  pip install textual
"""

from __future__ import annotations

import io
import os
import re
import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------
# heartbeat parsing
# --------------------------------------------------------------------------

# ** Heartbeat:  45% -- { 27000059 } -- 47.62 KIPS (1.48 KIPS)
_HEARTBEAT_RE = re.compile(
    r"\*\*\s*Heartbeat:\s*(\d+)%\s*--\s*\{\s*(\d+)\s*\}"
    r"(?:\s*--\s*([\d.]+|N/A)\s*KIPS\s*\(\s*([\d.]+|N/A)\s*KIPS\s*\))?"
)
# ** Core 0 Finished:    insts:60000002    cycles:23495900 ... -- 2.55 IPC (2.55 IPC)
_FINISHED_RE = re.compile(
    r"\*\*\s*Core\s+\d+\s+Finished:\s*insts:(\d+)\s+cycles:(\d+)"
    r".*?--\s*([\d.]+)\s*IPC"
)
_STARTED_RE = re.compile(r"Scarab started at (.+)")
_GITREV_RE = re.compile(r"Scarab gitrev:\s*(\S+)")

# Exec-driven runs fast-forward through the trace before simulation starts, and
# report that phase in pin.err.<core>.out rather than scarab.out:
#   Entering Hyper Fast Forward Mode: 1677129000001 ins remaining
#   Hyper FF Heartbeat: inst_count=10737418249 (0.64%)
# memtrace runs have no such file.
_FF_HEARTBEAT_RE = re.compile(
    r"Hyper FF Heartbeat:\s*inst_count=(\d+)\s*\(\s*([\d.eE+-]+)\s*%\s*\)"
)
_FF_ENTER_RE = re.compile(r"Entering Hyper Fast Forward Mode:\s*(\d+)\s*ins remaining")

PHASE_FF = "FF"
PHASE_SIM = "sim"

TAIL_BYTES = 16384  # plenty for the last few heartbeat lines


@dataclass
class Progress:
    """What we could learn from a simpoint's scarab.out."""

    pct: Optional[int] = None
    insts: Optional[int] = None
    kips_inst: Optional[float] = None   # instantaneous
    kips_avg: Optional[float] = None    # cumulative
    finished: bool = False
    ipc: Optional[float] = None
    cycles: Optional[int] = None
    gitrev: Optional[str] = None
    mtime: Optional[float] = None
    exists: bool = False
    note: str = ""

    # phase: PHASE_FF while hyper fast-forwarding, PHASE_SIM once scarab.out
    # starts emitting heartbeats, None when neither has produced anything yet.
    phase: Optional[str] = None
    ff_pct: Optional[float] = None
    ff_insts: Optional[int] = None
    ff_total: Optional[int] = None
    # instructions/second observed across polls, used for the FF eta since the
    # FF heartbeats carry no rate of their own
    rate: Optional[float] = None
    sampled_at: Optional[float] = None
    # The last poll at which ff_insts actually changed. FF heartbeats land in
    # bursts, so measuring the rate between adjacent polls would divide a whole
    # burst by one poll interval and overstate the rate by the burst period.
    ff_mark_insts: Optional[int] = None
    ff_mark_at: Optional[float] = None

    @property
    def display_pct(self) -> Optional[float]:
        """Percent within the current phase, for the bar and the % column."""
        if self.phase == PHASE_FF:
            return self.ff_pct
        return None if self.pct is None else float(self.pct)

    @property
    def eta_seconds(self) -> Optional[float]:
        """Remaining wall time for the current phase.

        The simulation phase uses Scarab's own cumulative KIPS. The FF phase has
        no rate in its heartbeat, so it relies on ``rate`` measured across polls.
        """
        if self.finished:
            return None
        if self.phase == PHASE_FF:
            if not self.ff_pct or not self.ff_insts or not self.rate:
                return None
            if self.ff_pct >= 100 or self.rate <= 0:
                return None
            total = self.ff_total or (self.ff_insts / (self.ff_pct / 100.0))
            return max(0.0, (total - self.ff_insts) / self.rate)
        if not self.pct or not self.insts or not self.kips_avg:
            return None
        if self.pct >= 100 or self.kips_avg <= 0:
            return None
        total_insts = self.insts / (self.pct / 100.0)
        remaining = total_insts - self.insts
        return remaining / (self.kips_avg * 1000.0)


def _read_tail(path: Path, nbytes: int = TAIL_BYTES) -> str:
    with path.open("rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        fh.seek(max(0, size - nbytes))
        return fh.read().decode("utf-8", errors="replace")


def _read_head(path: Path, nbytes: int = 2048) -> str:
    with path.open("rb") as fh:
        return fh.read(nbytes).decode("utf-8", errors="replace")


def find_ff_log(sim_dir: Path) -> Optional[Path]:
    """Most recently touched pin.err.<core>.out, if this is an exec-driven run."""
    try:
        candidates = list(sim_dir.glob("pin.err.*.out"))
    except OSError:
        return None
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    try:
        return max(candidates, key=lambda c: c.stat().st_mtime)
    except OSError:
        return candidates[0]


def parse_ff_progress(ff_log: Path, p: Progress) -> bool:
    """Fill in the fast-forward fields from a pin.err log. True if FF data found."""
    try:
        st = ff_log.stat()
        tail = _read_tail(ff_log)
    except OSError:
        return False

    last = None
    for m in _FF_HEARTBEAT_RE.finditer(tail):
        last = m
    if last is None:
        # The "Entering Hyper Fast Forward Mode" banner may be the only thing
        # written so far, and it sits at the head of the file.
        try:
            head = _read_head(ff_log, 4096)
        except OSError:
            return False
        em = _FF_ENTER_RE.search(head)
        if em is None:
            return False
        p.phase = PHASE_FF
        p.ff_total = int(em.group(1))
        p.ff_insts = 0
        p.ff_pct = 0.0
        p.mtime = st.st_mtime
        p.note = "fast-forwarding (no FF heartbeat yet)"
        return True

    p.phase = PHASE_FF
    p.ff_insts = int(last.group(1))
    try:
        p.ff_pct = float(last.group(2))
    except ValueError:
        p.ff_pct = None
    # The banner gives the exact denominator; prefer it over back-computing.
    try:
        em = _FF_ENTER_RE.search(_read_head(ff_log, 4096))
        if em:
            p.ff_total = int(em.group(1))
    except OSError:
        pass
    p.mtime = st.st_mtime
    p.note = "fast-forwarding"
    return True


def parse_progress(scarab_out: Path, prev: Optional[Progress] = None,
                   now: Optional[float] = None) -> Progress:
    """Parse the tail of a scarab.out for the most recent heartbeat.

    Falls back to the pin.err fast-forward log when scarab.out has not started
    emitting heartbeats yet, which is the normal state for the first hours of an
    exec-driven simulation.

    ``prev`` is the previous Progress for the same simpoint; it is used only to
    measure the FF instruction rate across polls, since FF heartbeats carry no
    rate of their own.

    Cheap by design: a stat plus a couple of KB per file, so scanning a few
    hundred simpoints per refresh stays well under a second.
    """
    import time as _time

    p = Progress()
    now = _time.time() if now is None else now
    try:
        if not scarab_out.is_file():
            p.note = "scarab.out not created yet"
            _try_ff(scarab_out.parent, p, prev, now)
            return p
        p.exists = True
        p.mtime = scarab_out.stat().st_mtime
        tail = _read_tail(scarab_out)
    except OSError as exc:
        p.note = f"unreadable: {exc}"
        return p

    fin = None
    for m in _FINISHED_RE.finditer(tail):
        fin = m
    if fin is not None:
        p.finished = True
        p.pct = 100
        p.insts = int(fin.group(1))
        p.cycles = int(fin.group(2))
        p.ipc = float(fin.group(3))
    else:
        last = None
        for m in _HEARTBEAT_RE.finditer(tail):
            last = m
        if last is not None:
            p.pct = int(last.group(1))
            p.insts = int(last.group(2))
            for attr, grp in (("kips_inst", 3), ("kips_avg", 4)):
                raw = last.group(grp)
                if raw and raw != "N/A":
                    setattr(p, attr, float(raw))
        else:
            # No simulation heartbeat: an exec-driven run is probably still
            # fast-forwarding, which is reported in pin.err.<core>.out instead.
            if not _try_ff(scarab_out.parent, p, prev, now):
                p.note = "no heartbeat yet (warmup / trace load?)"

    if p.pct is not None or p.finished:
        p.phase = PHASE_SIM

    try:
        head = _read_head(scarab_out)
        gm = _GITREV_RE.search(head)
        if gm:
            p.gitrev = gm.group(1)
    except OSError:
        pass
    return p


def _try_ff(sim_dir: Path, p: Progress, prev: Optional[Progress],
            now: float) -> bool:
    """Populate FF fields and estimate the FF rate from the previous sample."""
    ff_log = find_ff_log(sim_dir)
    if ff_log is None:
        return False
    if not parse_ff_progress(ff_log, p):
        return False
    p.sampled_at = now
    # Timestamp the sample by when the log was written rather than when we
    # happened to read it: that removes poll jitter from the rate entirely.
    stamp = p.mtime or now

    if p.ff_insts is None:
        return True

    if prev is None or prev.phase != PHASE_FF or prev.ff_mark_insts is None:
        # First sighting: nothing to measure against yet.
        p.ff_mark_insts = p.ff_insts
        p.ff_mark_at = stamp
        return True

    if p.ff_insts > prev.ff_mark_insts and prev.ff_mark_at is not None:
        dt = stamp - prev.ff_mark_at
        dn = p.ff_insts - prev.ff_mark_insts
        if dt > 0:
            inst_rate = dn / dt
            p.rate = (inst_rate if prev.rate is None
                      else 0.3 * inst_rate + 0.7 * prev.rate)
        else:
            p.rate = prev.rate
        p.ff_mark_insts = p.ff_insts
        p.ff_mark_at = stamp
    else:
        # No new instructions since the last mark: carry both forward so the
        # next change measures over the full elapsed interval.
        p.rate = prev.rate
        p.ff_mark_insts = prev.ff_mark_insts
        p.ff_mark_at = prev.ff_mark_at
    return True


# --------------------------------------------------------------------------
# status collection
# --------------------------------------------------------------------------

STATES = ["Completed", "Failed", "Failed - Slurm", "Running", "Pending", "Non-existant"]


@dataclass
class Job:
    config: str
    suite: str
    subsuite: str
    workload: str
    cluster_id: str
    state: str
    log_path: Optional[str] = None
    detail: Optional[str] = None
    node: Optional[str] = None
    progress: Progress = field(default_factory=Progress)

    @property
    def sim_dir(self) -> str:
        return f"{self.suite}/{self.subsuite}/{self.workload}/{self.cluster_id}"


@dataclass
class Snapshot:
    configs: list[str]
    jobs: list[Job]
    available_nodes: list[str] = field(default_factory=list)
    all_nodes: list[str] = field(default_factory=list)
    node_counts: dict[str, int] = field(default_factory=dict)
    raw_output: str = ""

    def count(self, config: str, state: str) -> int:
        return sum(1 for j in self.jobs if j.config == config and j.state == state)

    def select(self, config: str, state: str) -> list[Job]:
        """Jobs for a cell, most-advanced first (simulation phase before FF)."""
        def rank(j: "Job"):
            p = j.progress
            phase_rank = 0 if p.phase == PHASE_SIM else 1 if p.phase == PHASE_FF else 2
            return (phase_rank, -(p.display_pct or -1.0), j.workload, int(j.cluster_id))

        return sorted(
            (j for j in self.jobs if j.config == config and j.state == state),
            key=rank,
        )


def scarab_out_path(sim_root: Path, job: "Job") -> Path:
    return sim_root / job.config / job.suite / job.subsuite / job.workload / job.cluster_id / "scarab.out"


def refresh_running_progress(snapshot: Snapshot, sim_root: Path) -> int:
    """Re-read heartbeats for the Running jobs only. Returns how many just finished.

    This is the cheap tier: a stat plus a small tail per running simpoint, with no
    slurm queries and no re-reading of job logs. Safe to call every second.

    A job whose scarab.out now reports 'Core N Finished' is *not* reclassified here
    -- deciding Completed vs Failed needs the job log and stat files. The count is
    returned so the caller can trigger one full refresh instead of polling.
    """
    newly_done = 0
    for job in snapshot.jobs:
        if job.state != "Running":
            continue
        before = job.progress.finished
        job.progress = parse_progress(scarab_out_path(sim_root, job), prev=job.progress)
        if job.progress.finished and not before:
            newly_done += 1
    return newly_done


def collect_snapshot(descriptor_path: str, infra_dir: str, dbg_lvl: int = 1) -> Snapshot:
    """Run the normal status pass, capture per-job records, then add progress."""
    import subprocess

    from scripts.utilities import (
        read_descriptor_from_json,
        print_simulation_status_summary,
        get_image_list,
    )
    from scripts import slurm_runner

    descriptor_data = read_descriptor_from_json(descriptor_path, dbg_lvl)
    if descriptor_data is None:
        raise RuntimeError(f"Failed to read descriptor {descriptor_path}")

    db_name = (
        "workloads_top_simp.json"
        if descriptor_data.get("top_simpoint")
        else "workloads_db.json"
    )
    workloads_data = read_descriptor_from_json(f"{infra_dir}/workloads/{db_name}", dbg_lvl)
    if workloads_data is None:
        raise RuntimeError("Failed to read workloads database")

    user = subprocess.check_output("whoami").decode().strip()
    experiment = descriptor_data["experiment"]
    docker_prefix_list = get_image_list(descriptor_data.get("simulations") or [], workloads_data)

    available_nodes: list[str] = []
    all_nodes: list[str] = []
    running_sims: list[str] = []
    queued_sims: list[str] = []
    node_of_sim: dict[str, str] = {}
    node_counts: dict[str, int] = {}

    try:
        available_nodes, all_nodes = slurm_runner.check_available_nodes(dbg_lvl)
    except Exception:
        pass

    try:
        per_node = slurm_runner.check_slurm_task_queued_or_running(
            docker_prefix_list, experiment, user, dbg_lvl
        )
        for node, sims in per_node.items():
            if node == "":
                queued_sims += sims
            else:
                running_sims += sims
                node_counts[node] = len(sims)
                for s in sims:
                    node_of_sim[s] = node
    except Exception:
        pass

    records: list[dict] = []
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_simulation_status_summary(
            descriptor_data,
            workloads_data,
            docker_prefix_list,
            user,
            running_sims,
            queued_sims,
            dbg_lvl=dbg_lvl,
            all_nodes=all_nodes,
            log_file_count_buffer=1,
            strict_log_count=False,
            log_count_offset=1,
            prep_failed_label="Failed - Slurm",
            record_sink=records,
        )

    sim_root = Path(descriptor_data["root_dir"]) / "simulations" / experiment
    jobs: list[Job] = []
    # get_simulation_job_identifiers() repeats every (config, simpoint) once per
    # configuration, so the expected-job list upstream contains len(configs)
    # copies of each entry. Upstream only uses it for membership tests, where
    # that is harmless, but we turn it into rows -- so collapse it here or the
    # Pending/Non-existant counts come out multiplied and the detail table gets
    # duplicate rows.
    seen_ids: set = set()
    for r in records:
        ident = (r["config"], r["suite"], r["subsuite"], r["workload"],
                 r["cluster_id"], r["state"])
        if ident in seen_ids:
            continue
        seen_ids.add(ident)
        job = Job(
            config=r["config"],
            suite=r["suite"],
            subsuite=r["subsuite"],
            workload=r["workload"],
            cluster_id=r["cluster_id"],
            state=r["state"],
            log_path=r.get("log_path"),
            detail=r.get("detail"),
        )
        needle = f"_{job.workload}_{experiment}_{job.config}_{job.cluster_id}_"
        for sim, node in node_of_sim.items():
            if needle in sim:
                job.node = node
                break
        if job.state in ("Running", "Failed", "Completed"):
            job.progress = parse_progress(scarab_out_path(sim_root, job))
        jobs.append(job)

    return Snapshot(
        configs=list(descriptor_data["configurations"].keys()),
        jobs=jobs,
        available_nodes=available_nodes,
        all_nodes=all_nodes,
        node_counts=node_counts,
        raw_output=buf.getvalue(),
    )


# --------------------------------------------------------------------------
# formatting helpers
# --------------------------------------------------------------------------

def bar(pct: Optional[float], width: int = 16) -> str:
    if pct is None:
        return "·" * width
    # floor, so anything short of 100% never renders as a full bar
    filled = width if pct >= 100 else int(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)


def fmt_pct(p: Progress) -> str:
    """Percent within the current phase. FF percentages are tiny, so show decimals."""
    v = p.display_pct
    if v is None:
        return "--"
    if p.phase == PHASE_FF:
        return f"{v:.2f}%" if v < 10 else f"{v:.1f}%"
    return f"{v:.0f}%"


def fmt_eta(seconds: Optional[float]) -> str:
    if seconds is None:
        return "--"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def fmt_age(mtime: Optional[float]) -> str:
    """How long ago a file was last written, coarsely."""
    if mtime is None:
        return "--"
    import time

    delta = time.time() - mtime
    if delta < 0:
        return "0s"
    if delta >= 86400:
        return f"{int(delta // 86400)}d"
    return fmt_eta(delta)


def fmt_insts(n: Optional[int]) -> str:
    if n is None:
        return "--"
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n / div:.1f}{unit}"
    return str(n)


# --------------------------------------------------------------------------
# textual app
# --------------------------------------------------------------------------

def build_app(provider, sim_root: Path, sub_title: str = "",
              full_interval: float = 300.0, progress_interval: float = 1.0):
    """Construct the Textual app around a snapshot ``provider`` callable.

    Split out from ``run_tui`` so the UI can be driven in tests with a synthetic
    provider instead of a live slurm cluster.
    """
    try:
        from textual.app import App, ComposeResult
        from textual.binding import Binding
        from textual.containers import Vertical
        from textual.screen import Screen
        from textual.widgets import DataTable, Footer, Header, Static, Log
    except ImportError as exc:
        raise ImportError("textual is required for the interactive status view") from exc

    from textual.coordinate import Coordinate
    from rich.text import Text

    STATE_STYLE = {
        "Completed": "green",
        "Failed": "red",
        "Failed - Slurm": "magenta",
        "Running": "yellow",
        "Pending": "cyan",
        "Non-existant": "dim",
    }

    class DetailScreen(Screen):
        BINDINGS = [
            Binding("escape,q,backspace", "app.pop_screen", "Back"),
            Binding("r", "refresh", "Refresh"),
            Binding("l", "show_log", "Log tail"),
        ]

        def __init__(self, config: str, state: str) -> None:
            super().__init__()
            self.config = config
            self.state = state
            self.rows: list[Job] = []   # row index -> Job, set by populate()

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            yield Static(id="detail-title")
            yield DataTable(id="detail", zebra_stripes=True)
            yield Footer()

        def on_mount(self) -> None:
            table = self.query_one("#detail", DataTable)
            table.cursor_type = "row"
            table.add_columns(
                "Workload", "Simpoint", "Node", "Phase", "Progress", "%",
                "Insts", "Rate", "ETA", "Updated", "Note"
            )
            self.populate()

        def populate(self) -> None:
            snap: Snapshot = self.app.snapshot
            jobs = snap.select(self.config, self.state)
            title = self.query_one("#detail-title", Static)
            style = STATE_STYLE.get(self.state, "white")
            title.update(
                f"[bold]{self.config}[/bold] / [{style}]{self.state}[/{style}]  "
                f"— {len(jobs)} simpoint(s)"
            )
            table = self.query_one("#detail", DataTable)
            table.clear()
            # Rows are identified by position rather than by a composed key: a
            # duplicated key raises DuplicateKey and takes the whole screen down.
            self.rows = list(jobs)
            row_style = STATE_STYLE.get(self.state, "white")
            for j in jobs:
                p = j.progress
                note = j.detail or p.note or ""
                if p.finished and p.ipc is not None:
                    note = note or f"{p.ipc:.2f} IPC"

                if p.phase == PHASE_FF:
                    phase_cell = Text("FF", style="blue")
                    bar_style = "blue"
                    insts = fmt_insts(p.ff_insts)
                    rate = "--" if not p.rate else f"{p.rate / 1e6:.1f}M/s"
                elif p.phase == PHASE_SIM:
                    phase_cell = Text("sim", style=row_style)
                    bar_style = row_style
                    insts = fmt_insts(p.insts)
                    rate = "--" if p.kips_avg is None else f"{p.kips_avg:.2f}K/s"
                else:
                    phase_cell = Text("-", style="dim")
                    bar_style = "dim"
                    insts = "--"
                    rate = "--"

                table.add_row(
                    j.workload,
                    j.cluster_id,
                    j.node or "-",
                    phase_cell,
                    Text(bar(p.display_pct), style=bar_style),
                    fmt_pct(p),
                    insts,
                    rate,
                    fmt_eta(p.eta_seconds),
                    fmt_age(p.mtime),
                    note,
                )

        def action_refresh(self) -> None:
            self.app.action_refresh()

        def action_show_log(self) -> None:
            table = self.query_one("#detail", DataTable)
            if not table.row_count:
                return
            row = table.cursor_coordinate.row
            if 0 <= row < len(self.rows):
                self.app.push_screen(LogScreen(self.rows[row], self.app.sim_root))

    class LogScreen(Screen):
        BINDINGS = [Binding("escape,q,backspace", "app.pop_screen", "Back")]

        def __init__(self, job: Job, sim_root: Path) -> None:
            super().__init__()
            self.job = job
            self.sim_root = sim_root

        def compose(self) -> ComposeResult:
            yield Header()
            yield Static(id="log-title")
            yield Log(id="logview", highlight=True)
            yield Footer()

        def on_mount(self) -> None:
            j = self.job
            out = self.sim_root / j.config / j.suite / j.subsuite / j.workload / j.cluster_id / "scarab.out"
            self.query_one("#log-title", Static).update(f"[bold]{out}[/bold]")
            log = self.query_one("#logview", Log)
            candidates = [out]
            if j.log_path:
                candidates.append(Path(j.log_path))
            for path in candidates:
                try:
                    text = _read_tail(path, 32768)
                except OSError as exc:
                    log.write_line(f"<cannot read {path}: {exc}>")
                    continue
                log.write_line(f"===== tail of {path} =====")
                for line in text.splitlines()[-200:]:
                    log.write_line(line)
                log.write_line("")

    class StatusApp(App):
        CSS = """
        #summary { height: auto; margin: 1 0; }
        #detail { height: 1fr; }
        #nodes, #detail-title, #log-title, #hint { padding: 0 1; }
        #logview { height: 1fr; }
        """
        BINDINGS = [
            Binding("r", "refresh", "Refresh"),
            Binding("q", "quit", "Quit"),
        ]
        TITLE = "scarab-infra status"

        def __init__(self) -> None:
            super().__init__()
            self.snapshot: Snapshot = Snapshot(configs=[], jobs=[])
            self.sim_root = sim_root
            self._collecting = False   # a full refresh worker is in flight
            self._loaded = False       # at least one snapshot has landed
            self._started_at = 0.0
            self._progress_busy = False

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            yield Vertical(
                Static("loading…", id="nodes"),
                DataTable(id="summary", zebra_stripes=True),
                Static("", id="hint"),
            )
            yield Footer()

        def on_mount(self) -> None:
            table = self.query_one("#summary", DataTable)
            table.cursor_type = "cell"
            table.add_columns("Configuration", *STATES, "Total")
            self.query_one("#hint", Static).update(
                "[dim]select a count cell (click or arrows+enter) to drill into simpoints[/dim]"
            )
            self.sub_title = sub_title
            self.action_refresh()
            if full_interval:
                self.set_interval(full_interval, self.action_refresh)
            if progress_interval:
                self.set_interval(progress_interval, self._poll_progress)
            self.set_interval(1.0, self._tick)

        def _tick(self) -> None:
            """Show a progress hint while the very first collection is running."""
            if self._collecting and not self._loaded:
                import time

                waited = int(time.monotonic() - self._started_at)
                self.query_one("#nodes", Static).update(
                    f"collecting status… {waited}s  [dim](squeue + log scan)[/dim]"
                )

        def _poll_progress(self) -> None:
            """Cheap tier: re-read heartbeats of Running jobs only."""
            if not self._loaded or self._collecting or self._progress_busy:
                return
            if not any(j.state == "Running" for j in self.snapshot.jobs):
                return
            self._progress_busy = True
            self.run_worker(self._do_poll_progress, thread=True)

        def _do_poll_progress(self) -> None:
            try:
                newly_done = refresh_running_progress(self.snapshot, self.sim_root)
            except Exception:
                return  # transient NFS error; the next tick will retry
            finally:
                self._progress_busy = False
            self.call_from_thread(self._redraw_progress)
            if newly_done:
                # A simpoint just hit 'Core N Finished'. Deciding Completed vs Failed
                # needs the job log and stat files, so fall back to a full pass once.
                self.call_from_thread(self.action_refresh)

        def _redraw_progress(self) -> None:
            for screen in self.screen_stack:
                if isinstance(screen, DetailScreen):
                    screen.populate()

        def action_refresh(self) -> None:
            # A collection pass can easily outlast a short interval on a
            # busy NFS mount. Skip rather than restart: cancelling an in-flight
            # worker every interval would mean a snapshot never lands at all.
            if self._collecting:
                return
            import time

            self._collecting = True
            self._started_at = time.monotonic()
            self.run_worker(self._refresh, thread=True)

        def _refresh(self) -> None:
            try:
                snap = provider()
            except Exception as exc:  # keep the UI alive on transient slurm hiccups
                self.call_from_thread(
                    self.query_one("#nodes", Static).update,
                    f"[red]refresh failed: {exc}[/red]",
                )
                return
            finally:
                self._collecting = False
            self.call_from_thread(self._apply, snap)

        def _apply(self, snap: Snapshot) -> None:
            self.snapshot = snap
            self._loaded = True
            up = [n for n in snap.all_nodes if n in snap.available_nodes]
            down = [n for n in snap.all_nodes if n not in snap.available_nodes]
            running_total = sum(snap.node_counts.values())
            by_node = "  ".join(f"{k}:{v}" for k, v in sorted(snap.node_counts.items()))
            self.query_one("#nodes", Static).update(
                f"[green]up[/green] {', '.join(up) or '-'}   "
                f"[red]down[/red] {', '.join(down) or '-'}\n"
                f"running {running_total}   {by_node}"
            )

            table = self.query_one("#summary", DataTable)
            table.clear()
            for config in snap.configs:
                cells = []
                total = 0
                for state in STATES:
                    n = snap.count(config, state)
                    total += n
                    style = STATE_STYLE.get(state, "white") if n else "dim"
                    cells.append(Text(str(n), style=style, justify="right"))
                table.add_row(
                    Text(config, style="bold"),
                    *cells,
                    Text(str(total), justify="right"),
                    key=config,
                )

            for screen in self.screen_stack:
                if isinstance(screen, DetailScreen):
                    screen.populate()

        def on_data_table_cell_selected(self, event: DataTable.CellSelected) -> None:
            if event.data_table.id != "summary":
                return
            col = event.coordinate.column
            if col == 0 or col > len(STATES):
                return
            state = STATES[col - 1]
            row_key = event.data_table.coordinate_to_cell_key(event.coordinate).row_key
            config = str(row_key.value)
            if not self.snapshot.select(config, state):
                self.notify(f"no simpoints in {config} / {state}", severity="information")
                return
            self.push_screen(DetailScreen(config, state))

    return StatusApp()


def run_tui(descriptor_path: str, infra_dir: str, dbg_lvl: int = 1,
            full_interval: float = 300.0, progress_interval: float = 1.0,
            refresh_interval: Optional[float] = None) -> int:
    """Entry point used by ``./sci --status <DESCRIPTOR> --tui``.

    ``refresh_interval`` is accepted for compatibility with the single-tier
    version of sci, which passed one period for everything. It maps onto the
    full (slurm + reclassify) tier; heartbeat polling keeps its own default, so
    an older sci still gets live progress without needing to be updated.
    """
    from scripts.utilities import read_descriptor_from_json

    if refresh_interval is not None:
        full_interval = float(refresh_interval)

    d = read_descriptor_from_json(descriptor_path, dbg_lvl)
    if d is None:
        raise RuntimeError(f"Failed to read descriptor {descriptor_path}")
    sim_root = Path(d["root_dir"]) / "simulations" / d["experiment"]

    try:
        app = build_app(
            provider=lambda: collect_snapshot(descriptor_path, infra_dir, dbg_lvl),
            sim_root=sim_root,
            sub_title=d["experiment"],
            full_interval=full_interval,
            progress_interval=progress_interval,
        )
    except ImportError:
        print("The interactive status view needs textual, which is not installed in")
        print("the scarabinfra conda environment. Install it with:\n")
        print("    conda run -n scarabinfra pip install textual\n")
        print("The plain text status output still works:")
        print(f"    ./sci --status {Path(descriptor_path).stem}")
        return 1

    app.run()
    return 0


# --------------------------------------------------------------------------
# self-test:  python -m scripts.status_tui --self-test
#
# Runs entirely on synthetic data in a temp directory -- no slurm, no cluster,
# no real experiment tree -- so it is safe to run anywhere and is the quickest
# way to tell whether this file is healthy after an edit or a bad copy.
# --------------------------------------------------------------------------

_SELFTEST_FAILURES: list = []


def _check(label, cond, extra=""):
    print(f"  [{'ok ' if cond else 'FAIL'}] {label} {extra}")
    if not cond:
        _SELFTEST_FAILURES.append(label)


_SIM_HEAD = """Scarab gitrev: cfbc252
Listening for new clients
Server verified connection.
Scarab started at Wed Aug 12 07:03:59 2026

Initialized Ramulator.
"""
_FF_HEAD = "Entering Hyper Fast Forward Mode: 1677129000001 ins remaining\n"
_FF_TOTAL = 1677129000001


def _make_snapshot(seed: int = 0) -> Snapshot:
    """A snapshot shaped like a real gapbs/pr sweep: 3 configs x 47 simpoints."""
    import random

    rng = random.Random(seed)
    workloads = ["pr_kron", "pr_urand", "pr_twitter", "pr_web", "pr_road"]
    plan = {
        "baseline": [("Completed", 47)],
        "reg_component": [("Completed", 43), ("Running", 4)],
        "upperbound": [("Completed", 41), ("Running", 6)],
    }
    jobs = []
    for config, spec in plan.items():
        cid = 100000
        for state, n in spec:
            for _ in range(n):
                cid += rng.randint(1000, 9000)
                p = Progress()
                if state == "Completed":
                    p = Progress(pct=100, insts=60000002, finished=True, ipc=2.55,
                                 exists=True, mtime=1.7e9, phase=PHASE_SIM)
                elif state == "Running":
                    pct = rng.randint(1, 99)
                    p = Progress(pct=pct, insts=int(60000000 * pct / 100),
                                 kips_avg=rng.uniform(0.5, 3.2), exists=True,
                                 mtime=1.7e9, phase=PHASE_SIM)
                jobs.append(Job(
                    config=config, suite="gapbs", subsuite="pr",
                    workload=rng.choice(workloads), cluster_id=str(cid), state=state,
                    node=rng.choice(["bohr2", "bohr3", "moore", "ohm"])
                    if state == "Running" else None,
                    progress=p,
                ))
    return Snapshot(
        configs=list(plan.keys()), jobs=jobs,
        available_nodes=["bohr2", "bohr3", "moore", "ohm"],
        all_nodes=["bohr2", "bohr3", "moore", "ohm", "twilight"],
        node_counts={"bohr3": 6, "moore": 1, "bohr2": 1, "ohm": 1},
    )


def _write_sim(root: Path, job: Job, pct: int, finished: bool = False) -> Path:
    p = scarab_out_path(root, job)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = _SIM_HEAD
    for i in range(1, pct + 1):
        insts = int(60000000 * i / 100)
        text += f"** Heartbeat: {i:3d}% -- {{ {insts} }} -- 50.00 KIPS ({i * 0.03:.2f} KIPS)\n"
    if finished:
        text += ("** Core 0 Finished:    insts:60000002    cycles:23495900    "
                 "time:7342468750000       -- 2.55 IPC (2.55 IPC) --  N/A  KIPS (3.17 KIPS)\n"
                 "Scarab finished at Wed Aug 12 12:19:26 2026\n")
    p.write_text(text)
    return p


def _write_ff(root: Path, job: Job, insts: int) -> Path:
    p = scarab_out_path(root, job).parent / "pin.err.0.out"
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text(_FF_HEAD)
    with p.open("a") as fh:
        fh.write(f"Hyper FF Heartbeat: inst_count={insts} ({100 * insts / _FF_TOTAL:.2f}%)\n")
    return p


def _test_parser(tmp: Path) -> None:
    print("\n-- heartbeat parser --")
    snap = _make_snapshot()
    job = [j for j in snap.jobs if j.state == "Running"][0]

    _write_sim(tmp, job, 45)
    p = parse_progress(scarab_out_path(tmp, job))
    _check("sim pct parsed", p.pct == 45, f"got {p.pct}")
    _check("phase is sim", p.phase == PHASE_SIM, f"got {p.phase}")
    _check("gitrev parsed", p.gitrev == "cfbc252", f"got {p.gitrev}")
    eta_h = (p.eta_seconds or 0) / 3600
    _check("sim eta computed", eta_h > 0, f"{fmt_eta(p.eta_seconds)}")

    _write_sim(tmp, job, 100, finished=True)
    p = parse_progress(scarab_out_path(tmp, job))
    _check("finish detected", p.finished and p.pct == 100, f"pct={p.pct}")
    _check("ipc parsed", p.ipc == 2.55, f"got {p.ipc}")
    _check("no eta when finished", p.eta_seconds is None)

    p = parse_progress(tmp / "nope" / "scarab.out")
    _check("missing file tolerated", p.pct is None and not p.exists, p.note)

    _check("bar(98) not full", bar(98).count("█") < 16, bar(98))
    _check("bar(100) full", bar(100).count("█") == 16)
    _check("fmt_eta(3725)", fmt_eta(3725) == "1h02m", fmt_eta(3725))


def _test_cheap_tier(tmp: Path) -> None:
    print("\n-- cheap tier touches only Running jobs --")
    snap = _make_snapshot()
    running = [j for j in snap.jobs if j.state == "Running"]
    completed = [j for j in snap.jobs if j.state == "Completed"]

    for j in running:
        _write_sim(tmp, j, 10)
    _write_sim(tmp, completed[0], 3)          # bogus: must not be re-read
    completed[0].progress = Progress(pct=100, finished=True, ipc=2.55, phase=PHASE_SIM)

    done = refresh_running_progress(snap, tmp)
    _check("no false completions", done == 0, f"got {done}")
    _check("running updated", all(j.progress.pct == 10 for j in running),
           f"{[j.progress.pct for j in running]}")
    _check("completed untouched", completed[0].progress.pct == 100,
           f"got {completed[0].progress.pct}")

    _write_sim(tmp, running[0], 40)
    _write_sim(tmp, running[1], 100, finished=True)
    done = refresh_running_progress(snap, tmp)
    _check("advance picked up", running[0].progress.pct == 40, f"got {running[0].progress.pct}")
    _check("one newly finished", done == 1, f"got {done}")
    done = refresh_running_progress(snap, tmp)
    _check("completion reported once", done == 0, f"got {done}")

    snap.jobs.append(Job(config="baseline", suite="gapbs", subsuite="pr",
                         workload="nope", cluster_id="1", state="Running"))
    try:
        refresh_running_progress(snap, tmp)
        _check("missing scarab.out tolerated", True)
    except Exception as exc:
        _check("missing scarab.out tolerated", False, repr(exc))


def _test_ff_phase(tmp: Path) -> None:
    print("\n-- exec-driven fast-forward phase --")
    import os

    snap = _make_snapshot()
    job = [j for j in snap.jobs if j.state == "Running"][0]
    out = scarab_out_path(tmp, job)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_SIM_HEAD)                  # exists, but no heartbeat yet

    (out.parent / "pin.err.0.out").write_text(_FF_HEAD)
    p = parse_progress(out, now=1000.0)
    _check("banner-only detected as FF", p.phase == PHASE_FF, f"phase={p.phase}")
    _check("total from banner", p.ff_total == _FF_TOTAL, f"got {p.ff_total}")
    _check("no misleading sim pct", p.pct is None, f"pct={p.pct}")

    _write_ff(tmp, job, 10737418249)
    p1 = parse_progress(out, now=1000.0)
    _check("ff pct parsed", abs((p1.ff_pct or 0) - 0.64) < 0.01, f"got {p1.ff_pct}")
    _check("ff insts parsed", p1.ff_insts == 10737418249, f"got {p1.ff_insts}")
    _check("no rate on first sample", p1.rate is None)

    # Real conditions: poll at 1 Hz while FF heartbeats land every ~60s. A naive
    # dn/dt between adjacent polls would read a whole burst as one second and
    # overstate the rate by the burst period.
    ffile = out.parent / "pin.err.0.out"
    os.utime(ffile, (0.0, 0.0))
    prev = parse_progress(out, now=0.0)
    insts, CHUNK, PERIOD = 10737418249, 2147483648, 60
    for t in range(1, 181):
        if t % PERIOD == 0:
            insts += CHUNK
            _write_ff(tmp, job, insts)
            os.utime(ffile, (float(t), float(t)))
        prev = parse_progress(out, prev=prev, now=float(t))

    true_rate = CHUNK / PERIOD
    _check("rate accurate under 1Hz polling of 60s bursts",
           prev.rate is not None and abs(prev.rate - true_rate) / true_rate < 0.02,
           f"got {(prev.rate or 0) / 1e6:.1f}M/s, want {true_rate / 1e6:.1f}M/s")
    true_eta_h = (_FF_TOTAL - prev.ff_insts) / true_rate / 3600
    eta_h = (prev.eta_seconds or 0) / 3600
    _check("ff eta accurate", abs(eta_h - true_eta_h) / true_eta_h < 0.05,
           f"got {eta_h:.1f}h, want {true_eta_h:.1f}h")

    stalled = parse_progress(out, prev=prev, now=250.0)
    _check("stalled ff keeps rate", stalled.rate == prev.rate, f"got {stalled.rate}")

    with out.open("a") as fh:
        fh.write("** Heartbeat:   3% -- { 2000004 } -- 66.67 KIPS (0.11 KIPS)\n")
    flipped = parse_progress(out, prev=stalled, now=300.0)
    _check("phase flips to sim", flipped.phase == PHASE_SIM, f"phase={flipped.phase}")
    _check("sim pct used", flipped.pct == 3, f"got {flipped.pct}")
    _check("ff pct dropped in sim", flipped.ff_pct is None, f"got {flipped.ff_pct}")

    # memtrace runs have no pin.err at all
    job2 = [j for j in snap.jobs if j.state == "Running"][1]
    out2 = scarab_out_path(tmp, job2)
    out2.parent.mkdir(parents=True, exist_ok=True)
    out2.write_text(_SIM_HEAD)
    p5 = parse_progress(out2, now=1000.0)
    _check("memtrace: no ff phase", p5.phase is None, f"phase={p5.phase}")
    _check("memtrace: original note", "no heartbeat yet" in p5.note, repr(p5.note))

    # cheap tier must thread prev through so the rate survives
    ffjob = [j for j in snap.jobs if j.state == "Running"][3]
    fout = scarab_out_path(tmp, ffjob)
    fout.parent.mkdir(parents=True, exist_ok=True)
    fout.write_text(_SIM_HEAD)
    _write_ff(tmp, ffjob, 10737418249)
    refresh_running_progress(snap, tmp)
    _check("cheap tier detects ff", ffjob.progress.phase == PHASE_FF,
           f"phase={ffjob.progress.phase}")
    import time as _t
    _t.sleep(1.1)
    _write_ff(tmp, ffjob, 10737418249 + CHUNK)
    refresh_running_progress(snap, tmp)
    _check("cheap tier measures rate across polls", ffjob.progress.rate is not None,
           f"rate={ffjob.progress.rate}")


async def _test_ui() -> None:
    print("\n-- ui: drill-down, refresh, error handling --")
    import tempfile
    import time

    from textual.coordinate import Coordinate
    from textual.widgets import DataTable

    snap = _make_snapshot()
    app = build_app(provider=lambda: snap, sim_root=Path("/tmp/none"),
                    sub_title="selftest", full_interval=0, progress_interval=0)

    def col(table, label):
        for i, c in enumerate(table.ordered_columns):
            if str(c.label) == label:
                return i
        raise AssertionError(f"no column {label!r}")

    async with app.run_test(size=(150, 40)) as pilot:
        await pilot.pause()
        summary = app.screen.query_one("#summary", DataTable)
        _check("3 config rows", summary.row_count == 3, f"got {summary.row_count}")

        run_col = STATES.index("Running") + 1
        summary.cursor_coordinate = Coordinate(1, run_col)
        await pilot.press("enter")
        await pilot.pause()
        _check("detail screen pushed", len(app.screen_stack) == 2)

        detail = app.screen.query_one("#detail", DataTable)
        _check("4 running rows", detail.row_count == 4, f"got {detail.row_count}")
        _check("Phase column present",
               "Phase" in [str(c.label) for c in detail.ordered_columns])
        pc = col(detail, "%")
        pcts = [float(str(detail.get_cell_at(Coordinate(r, pc))).rstrip("%"))
                for r in range(detail.row_count)]
        _check("sorted by pct desc", pcts == sorted(pcts, reverse=True), f"{pcts}")

        await pilot.press("escape")
        await pilot.pause()
        _check("popped back", len(app.screen_stack) == 1)

        fail_col = STATES.index("Failed") + 1
        summary.cursor_coordinate = Coordinate(0, fail_col)
        await pilot.press("enter")
        await pilot.pause()
        _check("empty cell does not drill", len(app.screen_stack) == 1)

        summary.cursor_coordinate = Coordinate(0, 0)
        await pilot.press("enter")
        await pilot.pause()
        _check("config column inert", len(app.screen_stack) == 1)

    # a slow provider must still land a snapshot even with a short full interval
    calls = {"n": 0}

    def slow():
        calls["n"] += 1
        time.sleep(1.2)
        return snap

    app2 = build_app(provider=slow, sim_root=Path("/tmp/none"), sub_title="t",
                     full_interval=0.2, progress_interval=0)
    async with app2.run_test(size=(120, 30)) as pilot:
        for _ in range(40):
            await pilot.pause(0.1)
            if app2._loaded:
                break
        _check("snapshot lands despite short interval", app2._loaded)
        _check("in-flight refresh not restarted", calls["n"] <= 2, f"called {calls['n']}x")

    # a provider blowing up must not kill the app
    state = {"first": True}

    def flaky():
        if state["first"]:
            state["first"] = False
            raise RuntimeError("squeue timed out")
        return snap

    app3 = build_app(provider=flaky, sim_root=Path("/tmp/none"), sub_title="t",
                     full_interval=0, progress_interval=0)
    async with app3.run_test(size=(120, 30)) as pilot:
        for _ in range(20):
            await pilot.pause(0.1)
            if not app3._collecting:
                break
        banner = str(app3.screen.query_one("#nodes").content)
        _check("error surfaced in banner", "squeue timed out" in banner, repr(banner[:60]))
        _check("app still alive", app3.is_running)
        await pilot.press("r")
        for _ in range(20):
            await pilot.pause(0.1)
            if app3._loaded:
                break
        _check("recovers on manual refresh", app3._loaded)


if __name__ == "__main__":
    import sys
    print(__doc__)
    print("This module is driven by:  ./sci --status <DESCRIPTOR> --tui")