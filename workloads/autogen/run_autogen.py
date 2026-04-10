from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ModelFamily, ModelInfo
from autogen_ext.models.replay import ReplayChatCompletionClient


# ---------------------------------------------------------------------------
# Deterministic tools -- pure CPU work, no I/O
# ---------------------------------------------------------------------------

async def topdown_analyze(component: str) -> str:
    """Analyze top-down microarchitecture component."""
    data = {
        "component": component,
        "retiring": 0.34,
        "frontend_bound": 0.23,
        "backend_bound": 0.31,
        "bad_speculation": 0.12,
        "ipc": 1.42,
    }
    return json.dumps(data, sort_keys=True)


async def cache_profile(level: str) -> str:
    """Profile cache hierarchy behavior."""
    profiles = {
        "l1d": {"hit_rate": 0.96, "mpki": 12.3, "access_latency_ns": 1.2},
        "l2": {"hit_rate": 0.82, "mpki": 4.8, "access_latency_ns": 4.5},
        "llc": {"hit_rate": 0.65, "mpki": 1.2, "access_latency_ns": 12.0},
    }
    data = profiles.get(level, profiles["l1d"])
    data["level"] = level
    return json.dumps(data, sort_keys=True)


async def branch_stats(workload: str) -> str:
    """Return branch prediction statistics."""
    data = {
        "workload": workload,
        "total_branches": 142857,
        "mispredictions": 5428,
        "mpki": 7.45,
        "indirect_pct": 0.18,
        "conditional_pct": 0.72,
        "return_pct": 0.10,
    }
    return json.dumps(data, sort_keys=True)


# ---------------------------------------------------------------------------
# Pre-scripted LLM responses for ReplayChatCompletionClient
# ---------------------------------------------------------------------------

SCRIPTED_RESPONSES = [
    "Let me analyze the CPU top-down breakdown first.",
    "The frontend is bound at 23% -- significant instruction cache pressure.",
    "Now checking cache hierarchy behavior at L1D level.",
    "L1D hit rate is 96% but LLC shows 65% -- memory subsystem is stressed.",
    "Let me check branch prediction to understand bad speculation.",
    "Branch MPKI of 7.45 indicates heavy control flow in the agent loop.",
    "TERMINATE",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def context_blob(size: str) -> str:
    units = {"short": 32, "medium": 128, "long": 512}[size]
    return "agent-context " * units


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_json(path: str, payload: dict) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

async def run_benchmark(args) -> None:
    run_id = f"autogen__{args.task_id}__seed{args.seed}"
    context = context_blob(args.context_size)

    model_info: ModelInfo = {
        "vision": False,
        "function_calling": True,
        "json_output": False,
        "family": ModelFamily.UNKNOWN,
        "structured_output": False,
    }

    def build_agent() -> AssistantAgent:
        client = ReplayChatCompletionClient(
            chat_completions=list(SCRIPTED_RESPONSES),
            model_info=model_info,
        )
        return AssistantAgent(
            name="perf_analyst",
            model_client=client,
            tools=[topdown_analyze, cache_profile, branch_stats],
            system_message=(
                "You are a CPU performance analyst. Use the provided tools "
                "to gather microarchitectural data and report findings."
            ),
        )

    spans = []
    start_run = time.perf_counter_ns()

    def record(phase: str, start_ns: int, meta: dict) -> None:
        end_ns = time.perf_counter_ns()
        spans.append({
            "run_id": run_id,
            "span_id": f"{len(spans)+1:04d}",
            "phase": phase,
            "start_ns": start_ns,
            "end_ns": end_ns,
            "duration_ms": (end_ns - start_ns) / 1_000_000.0,
            "meta": meta,
        })

    total_messages = 0

    for i in range(args.iterations):
        agent = build_agent()

        t = time.perf_counter_ns()
        response = await agent.run(
            task=f"Analyze CPU microarchitecture bottlenecks for agent workload, iteration {i}. Context: {context}"
        )
        msg_count = len(response.messages)
        total_messages += msg_count

        record("agent_run", t, {
            "iteration": i,
            "messages": msg_count,
            "context_bytes": len(context),
        })

    end_run = time.perf_counter_ns()
    total_ms = (end_run - start_run) / 1_000_000.0

    trace = {
        "benchmark": "AgentCPU-Real",
        "workload": "autogen",
        "run_id": run_id,
        "task_id": args.task_id,
        "seed": args.seed,
        "mode": "replay",
        "run_metadata": {
            "framework": "autogen",
            "context_size": args.context_size,
            "iterations": args.iterations,
        },
        "spans": spans,
    }

    summary = {
        "benchmark": "AgentCPU-Real",
        "workload": "autogen",
        "task_id": args.task_id,
        "seed": args.seed,
        "mode": "replay",
        "run_id": run_id,
        "config": {
            "framework": "autogen",
            "context_size": args.context_size,
            "iterations": args.iterations,
            "tools": ["topdown_analyze", "cache_profile", "branch_stats"],
        },
        "result": {
            "success": True,
            "latency_ms": total_ms,
            "iterations": args.iterations,
            "total_messages": total_messages,
            "context_bytes": len(context),
        },
        "trace_output": args.trace_output,
    }

    write_json(args.trace_output, trace)
    write_json(args.summary_output, summary)
    print(json.dumps({"summary": summary["result"], "run_id": run_id}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoGen agent CPU benchmark")
    parser.add_argument("--task-id", default="autogen")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trace-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--context-size", choices=("short", "medium", "long"), default="medium")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of full agent invocations")
    args = parser.parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
