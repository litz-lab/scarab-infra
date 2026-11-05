#!/usr/bin/env python3
"""Replay proxy for a traced Codex session that waits for the tools to finish.

Usage: codex_replay_quiesce.py <codex_responses_proxy.py> <port> <responses_dir>

It runs the benchmark's own replay proxy (codex_responses_proxy.py, replay
mode: the recorded resp_NNN.json served in order, the same bytes) with one
change: before it answers a request, it waits until the codex process has no
running descendants, that is until every command the model started has ended.

Why: Codex's exec_command returns to the model after yield_time_ms (30 s in
the bat-2393 recording) even if the command still runs, and Codex kills the
process group of a still-running command when the session ends. In the
recording `cargo build` yielded after 30 s and finished 2 s into the
write_stdin that followed; under DynamoRIO the build and the test run take
tens of minutes, so without waiting the replay would move on at the recorded
pace and the session would end with the build and the tests cut short. The
traced session would then hold less compiler work than the recorded one.
Waiting here stands in for a slower model: the model's answers are the
recorded ones, and the tools get the time they need. The requests and
answers are unchanged (0 misses and the recorded patch, as before); what
changes is the text of a tool result that reports a command as still running
where the recording reported it finished, or the reverse.

Environment:
  CODEX_QUIESCE_MAX_S   longest wait per request, default 28800 (8 h); after it
                        the request is answered anyway and the log says so
  CODEX_QUIESCE_POLL_S  poll interval, default 1
The proxy log gets one line per request that had to wait:
  quiesce req <i> waited <s> s for <processes>

A watcher thread also sends SIGCONT every 2 s to descendants of codex that are
stopped (state T) and logs each one ("quiesce SIGCONT <name>:<pid>"): in the
code-py tool capture a child was left stopped in the middle of DynamoRIO's
handling of its execve (see write_dr_shell_wrapper in run_simpoint_trace.py),
and here a stopped command would otherwise make the wait last until
CODEX_QUIESCE_MAX_S. Nothing in these sessions stops a process on purpose.
"""
import os
import signal
import sys
import threading
import time

QUIESCE_MAX_S = float(os.environ.get("CODEX_QUIESCE_MAX_S", "28800"))
POLL_S = float(os.environ.get("CODEX_QUIESCE_POLL_S", "1"))


def _procs():
    """{pid: (ppid, comm, state)} for every process visible in /proc."""
    out = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                s = f.read()
        except OSError:
            continue
        # comm is in parentheses and may contain spaces; fields after it are fixed.
        lp, rp = s.find("("), s.rfind(")")
        if lp < 0 or rp < 0:
            continue
        rest = s[rp + 2:].split()
        out[int(d)] = (int(rest[1]), s[lp + 1:rp], rest[0])
    return out


def _descendants(procs):
    """[(pid, comm, state)] of the descendants of every process named codex."""
    roots = [p for p, (_, comm, _) in procs.items() if comm == "codex"]
    children = {}
    for p, (pp, _, _) in procs.items():
        children.setdefault(pp, []).append(p)
    found, stack = [], list(roots)
    while stack:
        for c in children.get(stack.pop(), []):
            stack.append(c)
            found.append((c, procs[c][1], procs[c][2]))
    return found


def running_descendants():
    """Live (non-zombie) descendants of every process named codex, as name:pid."""
    return [f"{comm}:{pid}" for pid, comm, state in _descendants(_procs()) if state != "Z"]


def resume_stopped():
    while True:
        for pid, comm, state in _descendants(_procs()):
            if state == "T":
                try:
                    os.kill(pid, signal.SIGCONT)
                    print(f"quiesce SIGCONT {comm}:{pid}", flush=True)
                except OSError:
                    pass
        time.sleep(2)


def wait_quiescent(index):
    start = time.time()
    seen = set()
    clear = 0
    while True:
        busy = running_descendants()
        seen.update(b.split(":")[0] for b in busy)
        clear = clear + 1 if not busy else 0
        if clear >= 2:
            break
        if time.time() - start > QUIESCE_MAX_S:
            print(f"quiesce req {index} gave up after {time.time() - start:.0f} s; still running: {busy}", flush=True)
            return
        time.sleep(POLL_S)
    waited = time.time() - start
    if seen:
        print(f"quiesce req {index} waited {waited:.0f} s for {sorted(seen)}", flush=True)


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    proxy, port, rec = sys.argv[1:]
    with open(proxy) as f:
        src = f.read()
    serve = 'ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()'
    if serve not in src:
        sys.exit(f"{proxy}: unexpected layout (no serve_forever line); refusing to run")
    sys.argv = [proxy, "replay", port, rec]
    ns = {"__name__": "codex_responses_proxy", "__file__": proxy}
    exec(compile(src.replace(serve, ""), proxy, "exec"), ns)
    base = ns["H"]

    class QuiesceHandler(base):
        def do_POST(self):
            wait_quiescent(ns["N"][0])
            base.do_POST(self)

    threading.Thread(target=resume_stopped, daemon=True).start()
    print(f"codex_replay_quiesce: waiting for codex descendants before each answer "
          f"(max {QUIESCE_MAX_S:.0f} s)", flush=True)
    ns["ThreadingHTTPServer"](("127.0.0.1", int(port)), QuiesceHandler).serve_forever()


if __name__ == "__main__":
    main()
