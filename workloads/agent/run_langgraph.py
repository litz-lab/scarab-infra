from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent


# ---------------------------------------------------------------------------
# Deterministic tools -- exercise JSON serialization, dict ops, hashing
# ---------------------------------------------------------------------------

@tool
def cpu_analyze(metric: str) -> str:
    """Analyze a CPU performance metric and return structured results."""
    data = {
        "metric": metric,
        "ipc": 1.42,
        "branch_miss_rate": 0.038,
        "frontend_bound": 0.23,
        "backend_bound": 0.31,
        "retiring": 0.34,
        "bad_speculation": 0.12,
    }
    encoded = json.dumps(data, sort_keys=True)
    return json.loads(encoded)


@tool
def memory_profile(region: str) -> str:
    """Profile memory access patterns for a given region."""
    data = {
        "region": region,
        "l1_hit_rate": 0.94,
        "l2_hit_rate": 0.78,
        "llc_hit_rate": 0.62,
        "dtlb_miss_rate": 0.0006,
        "itlb_miss_rate": 0.0001,
        "page_walks_per_ki": 0.03,
    }
    encoded = json.dumps(data, sort_keys=True)
    return json.loads(encoded)


@tool
def branch_analyze(function_name: str) -> str:
    """Analyze branch prediction behavior for a function."""
    data = {
        "function": function_name,
        "total_branches": 48210,
        "mispredictions": 1832,
        "mpki": 7.45,
        "indirect_branches_pct": 0.18,
        "conditional_branches_pct": 0.72,
    }
    encoded = json.dumps(data, sort_keys=True)
    return json.loads(encoded)


# ---------------------------------------------------------------------------
# Fake tool-calling chat model -- cycles through tool calls then final answer
# ---------------------------------------------------------------------------

TOOL_CALL_SEQUENCE = [
    [{"name": "cpu_analyze", "args": {"metric": "frontend_stalls"}, "id": "call_1", "type": "tool_call"}],
    [{"name": "memory_profile", "args": {"region": "heap"}, "id": "call_2", "type": "tool_call"}],
    [{"name": "branch_analyze", "args": {"function_name": "agent_loop"}, "id": "call_3", "type": "tool_call"}],
]


class FakeToolCallingModel(BaseChatModel):
    """Deterministic model that cycles through tool calls then answers."""
    call_count: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.call_count += 1
        idx = self.call_count - 1
        if idx < len(TOOL_CALL_SEQUENCE):
            msg = AIMessage(content="", tool_calls=TOOL_CALL_SEQUENCE[idx])
        else:
            msg = AIMessage(
                content=(
                    f"Analysis complete after {self.call_count} steps. "
                    "IPC=1.42, branch MPKI=7.45, frontend-bound=23%."
                )
            )
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools: list, **kwargs: Any):
        return self

    @property
    def _identifying_params(self):
        return {"model": "fake-tool-calling"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

def context_blob(size: str) -> str:
    units = {"short": 32, "medium": 128, "long": 512}[size]
    return "agent-context " * units


def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph ReAct agent CPU benchmark")
    parser.add_argument("--task-id", default="langgraph")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trace-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--context-size", choices=("short", "medium", "long"), default="medium")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of full agent invocations")
    args = parser.parse_args()

    run_id = f"langgraph__{args.task_id}__seed{args.seed}"
    context = context_blob(args.context_size)
    tools = [cpu_analyze, memory_profile, branch_analyze]
    model = FakeToolCallingModel()
    agent = create_react_agent(model, tools)

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

    total_tool_calls = 0
    total_messages = 0

    for i in range(args.iterations):
        model.call_count = 0

        t = time.perf_counter_ns()
        result = agent.invoke(
            {"messages": [{"role": "user",
                           "content": f"Analyze CPU microarchitecture bottlenecks, iteration {i}. Context: {context}"}]}
        )
        msgs = result["messages"]
        total_messages += len(msgs)
        iter_tool_calls = model.call_count - 1  # last call is the final answer
        total_tool_calls += max(iter_tool_calls, 0)

        record("agent_invoke", t, {
            "iteration": i,
            "messages": len(msgs),
            "tool_calls": iter_tool_calls,
            "context_bytes": len(context),
        })

    end_run = time.perf_counter_ns()
    total_ms = (end_run - start_run) / 1_000_000.0

    trace = {
        "benchmark": "AgentCPU-Real",
        "workload": "langgraph",
        "run_id": run_id,
        "task_id": args.task_id,
        "seed": args.seed,
        "mode": "replay",
        "run_metadata": {
            "framework": "langgraph",
            "context_size": args.context_size,
            "iterations": args.iterations,
        },
        "spans": spans,
    }

    summary = {
        "benchmark": "AgentCPU-Real",
        "workload": "langgraph",
        "task_id": args.task_id,
        "seed": args.seed,
        "mode": "replay",
        "run_id": run_id,
        "config": {
            "framework": "langgraph",
            "context_size": args.context_size,
            "iterations": args.iterations,
            "tools": [t.name for t in tools],
        },
        "result": {
            "success": True,
            "latency_ms": total_ms,
            "iterations": args.iterations,
            "total_tool_calls": total_tool_calls,
            "total_messages": total_messages,
            "context_bytes": len(context),
        },
        "trace_output": args.trace_output,
    }

    write_json(args.trace_output, trace)
    write_json(args.summary_output, summary)
    print(json.dumps({"summary": summary["result"], "run_id": run_id}, indent=2))


if __name__ == "__main__":
    main()
