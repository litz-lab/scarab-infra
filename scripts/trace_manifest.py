#!/usr/bin/python3
"""List the trace files workloads_db.json refers to, for upload or verification.

The database is what sci downloads against, so it decides what belongs in the
bucket: a zip nobody can reach through workloads_db.json is not a trace we
distribute, and the trace store also holds raw pipeline output, a git checkout
and top-3 copies of traces that already exist under their own suite.

Paths are printed relative to the trace directory, one per line, in the layout
both the store and the bucket use:

    <suite>/<subsuite>/<workload>/traces/simp/<cluster_id>.zip   (memtrace)
    <suite>/<subsuite>/<workload>/traces/pt/trace.gz             (pt)
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Held back from the bucket by size: driving is 465 GB and google 144 GB. The
# uploader skips them and sci tells you they are unavailable, both from here,
# so there is one list rather than a copy published beside the traces.
DEFAULT_SKIP = ("google", "driving")


def entries(db):
    for suite, subsuites in db.items():
        if not isinstance(subsuites, dict):
            continue
        for subsuite, workloads in subsuites.items():
            if not isinstance(workloads, dict):
                continue
            for workload, info in workloads.items():
                if not isinstance(info, dict):
                    continue
                simulation = info.get("simulation") or {}
                # A pt workload is one whole trace, named by run_pt_single_
                # simpoint.sh, and often has no simpoints at all.
                if "pt" in simulation:
                    yield suite, subsuite, workload, "traces/pt/trace.gz"
                # Exec-driven workloads have nothing to distribute; dhrystone is
                # exec-only by design, not a gap in the store.
                if "memtrace" not in simulation:
                    continue
                for simpoint in info.get("simpoints") or []:
                    cluster = (simpoint or {}).get("cluster_id")
                    if cluster is not None:
                        yield suite, subsuite, workload, f"traces/simp/{cluster}.zip"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--traces-dir", default="/soe/hlitz/lab/traces")
    parser.add_argument("--db", default=str(REPO_ROOT / "workloads" / "workloads_db.json"))
    parser.add_argument("--skip", nargs="*", default=list(DEFAULT_SKIP),
                        help="Suites, or suite/subsuite, to leave out "
                             f"(default: {' '.join(DEFAULT_SKIP)}).")
    parser.add_argument("--missing", action="store_true",
                        help="List what the db refers to but the store lacks.")
    parser.add_argument("--summary", action="store_true",
                        help="Per-suite counts instead of paths.")
    args = parser.parse_args()

    db = json.loads(Path(args.db).read_text(encoding="utf-8"))
    traces_dir = Path(args.traces_dir)
    skip = set(args.skip)

    present, missing, skipped = [], [], 0
    for suite, subsuite, workload, tail in entries(db):
        if suite in skip or f"{suite}/{subsuite}" in skip:
            skipped += 1
            continue
        rel = f"{suite}/{subsuite}/{workload}/{tail}"
        (present if (traces_dir / rel).is_file() else missing).append(rel)

    if args.summary:
        from collections import Counter
        have, lack, size = Counter(), Counter(), Counter()
        for rel in present:
            have[rel.split("/")[0]] += 1
            size[rel.split("/")[0]] += (traces_dir / rel).stat().st_size
        for rel in missing:
            lack[rel.split("/")[0]] += 1
        print(f"{'suite':16s} {'in store':>9s} {'absent':>8s} {'size':>10s}")
        for suite in sorted(set(have) | set(lack)):
            gb = size[suite] / 1e9
            print(f"{suite:16s} {have[suite]:9d} {lack[suite]:8d} {gb:9.1f}G")
        print(f"{'TOTAL':16s} {sum(have.values()):9d} {sum(lack.values()):8d} "
              f"{sum(size.values()) / 1e9:9.1f}G")
        print(f"\n{len(present)} files to upload, {len(missing)} referenced but absent, "
              f"{skipped} skipped by --skip")
        return 0

    for rel in (missing if args.missing else present):
        print(rel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
