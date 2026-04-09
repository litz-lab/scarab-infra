from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from langchain_core.tools import tool


@tool
def local_search(query: str) -> str:
    """Deterministic local tool used for CPU-side orchestration."""
    payload = {
        "query": query,
        "hits": [
            "CPU-side tool processing can dominate agent latency.",
            "Branchy orchestration and serialization stress the CPU frontend.",
            "Deterministic replay helps PMU-driven analysis.",
        ],
    }
    encoded = json.dumps(payload, sort_keys=True)
    decoded = json.loads(encoded)
    return json.dumps(decoded, sort_keys=True)


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_json(path: str, payload: dict) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def context_blob(size: str) -> str:
    units = {"short": 32, "medium": 128, "long": 512}[size]
    return "agent-context " * units


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", default="langchain")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trace-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--context-size", choices=("short", "medium", "long"), default="medium")
    parser.add_argument("--iterations", type=int, default=2)
    args = parser.parse_args()

    run_id = f"langchain__{args.task_id}__seed{args.seed}"
    context = context_blob(args.context_size)

    spans = []
    start_run = time.perf_counter_ns()

    def record(phase: str, start_ns: int, meta: dict) -> None:
        end_ns = time.perf_counter_ns()
        spans.append(
            {
                "run_id": run_id,
                "span_id": f"{len(spans)+1:04d}",
                "phase": phase,
                "start_ns": start_ns,
                "end_ns": end_ns,
                "duration_ms": (end_ns - start_ns) / 1_000_000.0,
                "meta": meta,
            }
        )

    t = time.perf_counter_ns()
    plan = {
        "steps": ["inspect_request", "select_tool", "invoke_tool", "synthesize"],
        "iterations": args.iterations,
    }
    record("plan", t, {"context_bytes": len(context)})

    t = time.perf_counter_ns()
    selected_tool = "local_search"
    record("select_tool", t, {"candidate_tools": 1, "selected_tool": selected_tool})

    tool_calls = 0
    last_payload_bytes = 0

    for i in range(args.iterations):
        t = time.perf_counter_ns()
        request = {
            "tool_name": selected_tool,
            "query": "How should an agent benchmark characterize CPU bottlenecks?",
            "iteration": i,
            "plan": plan,
        }
        marshaled = json.dumps(request, sort_keys=True)
        record("marshal_request", t, {"iteration": i, "request_bytes": len(marshaled)})

        t = time.perf_counter_ns()
        response = local_search.invoke(
            {"query": "How should an agent benchmark characterize CPU bottlenecks?"}
        )
        tool_calls += 1
        record("tool_call", t, {"iteration": i, "tool": selected_tool})

        t = time.perf_counter_ns()
        parsed = json.loads(response)
        last_payload_bytes = len(response)
        record("parse_response", t, {"iteration": i, "payload_bytes": last_payload_bytes, "hits": len(parsed["hits"])})

    t = time.perf_counter_ns()
    final_output = (
        "The LangChain-style loop is deterministic and CPU-visible, with "
        "branchy control flow, JSON marshalling, and local tool execution."
    )
    record("synthesize", t, {"context_bytes": len(context), "tool_calls": tool_calls})

    end_run = time.perf_counter_ns()
    total_ms = (end_run - start_run) / 1_000_000.0

    trace = {
        "benchmark": "AgentCPU-Real",
        "workload": "langchain",
        "run_id": run_id,
        "task_id": args.task_id,
        "seed": args.seed,
        "mode": "replay",
        "run_metadata": {
            "framework": "langchain",
            "context_size": args.context_size,
            "iterations": args.iterations,
        },
        "spans": spans,
    }

    summary = {
        "benchmark": "AgentCPU-Real",
        "workload": "langchain",
        "task_id": args.task_id,
        "seed": args.seed,
        "mode": "replay",
        "run_id": run_id,
        "config": {
            "framework": "langchain",
            "context_size": args.context_size,
            "iterations": args.iterations,
        },
        "result": {
            "success": True,
            "latency_ms": total_ms,
            "steps": 4,
            "tool_calls": tool_calls,
            "serialization_bytes": last_payload_bytes,
            "context_bytes": len(context),
            "final_output": final_output,
        },
        "trace_output": args.trace_output,
    }

    write_json(args.trace_output, trace)
    write_json(args.summary_output, summary)

    print(
        json.dumps(
            {
                "summary": summary["result"],
                "trace_path": args.trace_output,
                "run_id": run_id,
            },
            indent=2,
        )
    )

if __name__ == "__main__":
    main()
