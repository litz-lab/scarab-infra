from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, List, Optional, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.graph import END, StateGraph


# ---------------------------------------------------------------------------
# Deterministic seed file state for a mock "repository"
# ---------------------------------------------------------------------------

def seed_file_state() -> dict[str, list[str]]:
    return {
        "src/main.py": [
            "def main():",
            "    value = compute(1, 2)",
            "    print('result', value)",
            "",
            "if __name__ == '__main__':",
            "    main()",
        ],
        "src/compute.py": [
            "def compute(a, b):",
            "    # BUG: should be a + b",
            "    return a - b",
        ],
        "src/utils.py": [
            "def helper(x):",
            "    return x * 2",
            "",
            "def normalize(xs):",
            "    return [x / sum(xs) for x in xs]",
        ],
        "tests/test_compute.py": [
            "from compute import compute",
            "",
            "def test_add():",
            "    assert compute(1, 2) == 3",
            "",
            "def test_zero():",
            "    assert compute(0, 0) == 0",
        ],
        "README.md": [
            "# Demo project",
            "",
            "Small project used to characterize multi-step planner agents.",
        ],
    }


# ---------------------------------------------------------------------------
# Pure-CPU mock tools
# ---------------------------------------------------------------------------

def tool_read_file(state: "AgentState", path: str) -> str:
    lines = state["file_state"].get(path, [])
    return "\n".join(lines)


def tool_write_file(state: "AgentState", path: str, content: str) -> str:
    state["file_state"][path] = content.split("\n")
    return f"wrote {len(content)} bytes to {path}"


def tool_list_dir(state: "AgentState", path: str) -> str:
    prefix = path.rstrip("/") + "/"
    hits = [k for k in state["file_state"].keys() if k.startswith(prefix) or path == ""]
    hits.sort()
    return json.dumps(hits)


def tool_grep_files(state: "AgentState", pattern: str) -> str:
    rx = re.compile(pattern)
    matches: list[dict] = []
    for fname, lines in state["file_state"].items():
        for lineno, line in enumerate(lines):
            if rx.search(line):
                matches.append({"file": fname, "line": lineno + 1, "text": line})
    return json.dumps(matches, sort_keys=True)


def tool_run_tests(state: "AgentState") -> str:
    # Pass once a+b fix has been applied to src/compute.py
    compute_body = "\n".join(state["file_state"].get("src/compute.py", []))
    passed = "return a + b" in compute_body
    return json.dumps({
        "passed": passed,
        "total": 2,
        "failures": 0 if passed else 1,
        "step": state["step_count"],
    }, sort_keys=True)


def tool_apply_patch(state: "AgentState", path: str, diff: str) -> str:
    # Deterministic mutation: apply each "old|new" pair from diff
    lines = list(state["file_state"].get(path, []))
    changes = 0
    for spec in diff.split("\n"):
        if "|" not in spec:
            continue
        old, new = spec.split("|", 1)
        for i, line in enumerate(lines):
            if old and old in line:
                lines[i] = line.replace(old, new)
                changes += 1
    state["file_state"][path] = lines
    return f"applied patch to {path}: {changes} changes"


TOOL_FNS = {
    "read_file": tool_read_file,
    "write_file": tool_write_file,
    "list_dir": tool_list_dir,
    "grep_files": tool_grep_files,
    "run_tests": tool_run_tests,
    "apply_patch": tool_apply_patch,
}


# ---------------------------------------------------------------------------
# Scripted tool-call sequence (~20 steps exercising all six tools)
# ---------------------------------------------------------------------------

TOOL_CALL_SEQUENCE: list[dict] = [
    {"name": "list_dir", "args": {"path": "src"}},
    {"name": "read_file", "args": {"path": "README.md"}},
    {"name": "grep_files", "args": {"pattern": "compute"}},
    {"name": "read_file", "args": {"path": "src/main.py"}},
    {"name": "read_file", "args": {"path": "src/compute.py"}},
    {"name": "read_file", "args": {"path": "tests/test_compute.py"}},
    {"name": "run_tests", "args": {}},
    {"name": "grep_files", "args": {"pattern": "BUG"}},
    {"name": "read_file", "args": {"path": "src/utils.py"}},
    {"name": "list_dir", "args": {"path": "tests"}},
    {"name": "apply_patch", "args": {"path": "src/compute.py",
                                     "diff": "return a - b|return a + b"}},
    {"name": "read_file", "args": {"path": "src/compute.py"}},
    {"name": "run_tests", "args": {}},
    {"name": "grep_files", "args": {"pattern": "def "}},
    {"name": "write_file", "args": {"path": "src/notes.md",
                                    "content": "Fix: a+b instead of a-b"}},
    {"name": "read_file", "args": {"path": "src/notes.md"}},
    {"name": "list_dir", "args": {"path": "src"}},
    {"name": "grep_files", "args": {"pattern": "return"}},
    {"name": "run_tests", "args": {}},
    {"name": "read_file", "args": {"path": "src/compute.py"}},
]


# ---------------------------------------------------------------------------
# State + fake model
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    task: str
    plan: list[str]
    completed: list[str]
    file_state: dict[str, list[str]]
    observations: list[str]
    step_count: int
    messages: list[BaseMessage]
    next_tool: Optional[dict]
    done: bool


class FakeToolCallingModel(BaseChatModel):
    """Deterministic model: cycles TOOL_CALL_SEQUENCE, then returns 'done'."""
    call_count: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-swe-tool-calling"

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
            call = TOOL_CALL_SEQUENCE[idx]
            content = json.dumps({"action": "use_tool", **call}, sort_keys=True)
        else:
            content = json.dumps({"action": "done", "summary": "bug fixed"}, sort_keys=True)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    def bind_tools(self, tools: list, **kwargs: Any):
        return self

    @property
    def _identifying_params(self):
        return {"model": "fake-swe-tool-calling"}


# ---------------------------------------------------------------------------
# StateGraph nodes
# ---------------------------------------------------------------------------

MAX_STEPS = len(TOOL_CALL_SEQUENCE) + 2


def make_graph(model: FakeToolCallingModel, record):
    def planner(state: AgentState) -> dict:
        t = time.perf_counter_ns()
        plan_items = [
            "inspect_repo", "locate_bug", "read_sources",
            "propose_fix", "apply_fix", "run_tests", "verify",
        ]
        new_plan = plan_items[state["step_count"] % len(plan_items):] + \
            plan_items[: state["step_count"] % len(plan_items)]
        msg_count = len(state["messages"]) + 1
        new_messages = state["messages"] + [
            HumanMessage(content=f"plan step {state['step_count']}: {new_plan[0]}")
        ]
        record("plan", t, {"step": state["step_count"], "plan_head": new_plan[0],
                           "messages": msg_count})
        return {"plan": new_plan, "messages": new_messages}

    def tool_selector(state: AgentState) -> dict:
        t = time.perf_counter_ns()
        result = model._generate(state["messages"])
        ai_msg = result.generations[0].message
        payload = json.loads(ai_msg.content)
        record("select_tool", t, {"step": state["step_count"],
                                  "action": payload.get("action"),
                                  "tool": payload.get("name", "")})
        return {
            "next_tool": payload,
            "messages": state["messages"] + [ai_msg],
        }

    def tool_executor(state: AgentState) -> dict:
        t = time.perf_counter_ns()
        call = state["next_tool"] or {}
        action = call.get("action")
        if action != "use_tool":
            record("execute_tool", t, {"step": state["step_count"], "tool": "none"})
            return {"done": True}
        name = call.get("name", "")
        kwargs = call.get("args", {}) or {}
        fn = TOOL_FNS.get(name)
        if fn is None:
            out = f"unknown tool {name}"
        else:
            out = fn(state, **kwargs)
        new_obs = state["observations"] + [f"{name}: {out[:120]}"]
        record("execute_tool", t, {"step": state["step_count"], "tool": name,
                                   "out_bytes": len(out)})
        return {"observations": new_obs,
                "messages": state["messages"] + [HumanMessage(content=out)]}

    def observer(state: AgentState) -> dict:
        t = time.perf_counter_ns()
        tail = state["observations"][-3:]
        summary = " | ".join(tail)
        completed = state["completed"] + [state["plan"][0] if state["plan"] else "step"]
        record("observe", t, {"step": state["step_count"],
                              "observations": len(state["observations"]),
                              "summary_bytes": len(summary)})
        return {"completed": completed, "step_count": state["step_count"] + 1}

    def replan(state: AgentState) -> dict:
        t = time.perf_counter_ns()
        # Rotate plan on failure (every even step) -- branchy
        should_rotate = (state["step_count"] % 2 == 0) and bool(state["plan"])
        new_plan = state["plan"][1:] + state["plan"][:1] if should_rotate else state["plan"]
        record("replan", t, {"step": state["step_count"], "rotated": should_rotate,
                             "plan_len": len(new_plan)})
        return {"plan": new_plan}

    def check_done(state: AgentState) -> str:
        if state.get("done"):
            return "synthesize"
        if state["step_count"] >= MAX_STEPS:
            return "synthesize"
        # run_tests indicating passed also ends the loop
        for obs in reversed(state["observations"]):
            if obs.startswith("run_tests:") and '"passed": true' in obs:
                return "synthesize"
            break
        return "planner"

    def synthesize(state: AgentState) -> dict:
        t = time.perf_counter_ns()
        final = {
            "status": "done",
            "steps": state["step_count"],
            "observations": len(state["observations"]),
            "files_touched": len(state["file_state"]),
        }
        record("synthesize", t, final)
        return {"messages": state["messages"] + [AIMessage(content=json.dumps(final))]}

    graph = StateGraph(AgentState)
    graph.add_node("planner", planner)
    graph.add_node("tool_selector", tool_selector)
    graph.add_node("tool_executor", tool_executor)
    graph.add_node("observer", observer)
    graph.add_node("replan", replan)
    graph.add_node("synthesize", synthesize)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "tool_selector")
    graph.add_edge("tool_selector", "tool_executor")
    graph.add_edge("tool_executor", "observer")
    graph.add_edge("observer", "replan")
    graph.add_conditional_edges("replan", check_done,
                                {"planner": "planner", "synthesize": "synthesize"})
    graph.add_edge("synthesize", END)
    return graph.compile()


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


def context_blob(size: str) -> str:
    units = {"short": 32, "medium": 128, "long": 512}[size]
    return "agent-context " * units


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SWE-agent LangGraph CPU benchmark")
    parser.add_argument("--task-id", default="swe_agent")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trace-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--context-size", choices=("short", "medium", "long"), default="medium")
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    run_id = f"swe_agent__{args.task_id}__seed{args.seed}"
    context = context_blob(args.context_size)

    spans: list[dict] = []
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

    model = FakeToolCallingModel()
    app = make_graph(model, record)

    total_steps = 0
    total_tool_calls = 0

    for i in range(args.iterations):
        model.call_count = 0
        init_state: AgentState = {
            "task": f"Fix compute bug, iteration {i}. Context: {context}",
            "plan": [],
            "completed": [],
            "file_state": seed_file_state(),
            "observations": [],
            "step_count": 0,
            "messages": [HumanMessage(content=f"bugfix task {i}")],
            "next_tool": None,
            "done": False,
        }
        final_state = app.invoke(init_state, {"recursion_limit": 200})
        total_steps += final_state["step_count"]
        total_tool_calls += len([o for o in final_state["observations"]])

    end_run = time.perf_counter_ns()
    total_ms = (end_run - start_run) / 1_000_000.0

    trace = {
        "benchmark": "AgentCPU-Real",
        "workload": "swe_agent",
        "run_id": run_id,
        "task_id": args.task_id,
        "seed": args.seed,
        "mode": "replay",
        "run_metadata": {
            "framework": "langgraph",
            "archetype": "swe_agent",
            "context_size": args.context_size,
            "iterations": args.iterations,
        },
        "spans": spans,
    }

    summary = {
        "benchmark": "AgentCPU-Real",
        "workload": "swe_agent",
        "task_id": args.task_id,
        "seed": args.seed,
        "mode": "replay",
        "run_id": run_id,
        "config": {
            "framework": "langgraph",
            "archetype": "swe_agent",
            "context_size": args.context_size,
            "iterations": args.iterations,
            "tools": list(TOOL_FNS.keys()),
        },
        "result": {
            "success": True,
            "latency_ms": total_ms,
            "iterations": args.iterations,
            "total_steps": total_steps,
            "total_tool_calls": total_tool_calls,
            "context_bytes": len(context),
        },
        "trace_output": args.trace_output,
    }

    write_json(args.trace_output, trace)
    write_json(args.summary_output, summary)
    print(json.dumps({"summary": summary["result"], "run_id": run_id}, indent=2))


if __name__ == "__main__":
    main()
