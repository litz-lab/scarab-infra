#!/usr/bin/python3

# A helper script to stage a workload's run dir so hard-coded relative inputs resolve.
#
# Importable by the trace flow (run_simpoint_trace.py, same directory) and runnable as a script
# by the perf flow, which drives the container from the host and only needs the staged path back:
#
#     python3 /usr/local/bin/stage_local_rundir.py "<bincmd>"   -> prints the staged dir
#
# root_entrypoint.sh publishes every common/scripts/*.py into /usr/local/bin, so both entry points
# resolve to this file. No module-level side effects so importing it never prints anything in the
# trace flow. The perf flow parses stdout to retrieve the staged path.

import os
import tempfile


def stage_local_rundir(bincmd):
    """Stage the run dir as a symlink to resolve hard-coded relative inputs.

    Symlink the workload's reference run dir (= dir of the binary, the first token of bincmd)
    into a private, node-local, writable dir and return its path, excluding the binary.
    Some SPEC CPU2026 workloads open inputs by a hard-coded relative path not on the
    command line (e.g. ntest uses "resource/solver12.txt", gem5 uses Resource(...,".")),
    and those resolve only when drrun runs with this dir as CWD.
    """
    binary = bincmd.split()[0]
    refdir = os.path.dirname(binary)
    if not os.path.isdir(refdir):
        # os.walk() on a missing dir yields nothing and raises nothing.
        # Most likely the app tree is not mounted (see application_dir in the descriptor).
        raise FileNotFoundError(f"{refdir} not found. Wrong application_dir in the descriptor?")
    dest = tempfile.mkdtemp(prefix="scarab_trace_",
                            dir=os.environ.get("SCARAB_RUN_LOCAL_TMP", "/tmp"))
    skip = os.path.basename(binary)
    for root, _, files in os.walk(refdir):
        rel = os.path.relpath(root, refdir)
        target_dir = dest if rel == "." else os.path.join(dest, rel)
        os.makedirs(target_dir, exist_ok=True)
        for name in files:
            if root == refdir and name == skip:
                # Skip the binary itself
                continue
            os.symlink(os.path.join(root, name), os.path.join(target_dir, name))
    return dest


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        sys.exit(f"usage: {os.path.basename(sys.argv[0])} '<bincmd>'")

    # expandvars so a caller can pass the descriptor's literal $tmpdir/... command.
    # tmpdir comes from the image env (Dockerfile.common ENV tmpdir), so no shell is needed.
    print(stage_local_rundir(os.path.expandvars(sys.argv[1])))
