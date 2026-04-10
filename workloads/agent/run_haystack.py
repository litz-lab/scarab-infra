from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOPICS = [
    "CPU frontend stalls", "backend memory pressure", "branch misprediction",
    "instruction cache miss", "TLB pressure", "vector SIMD throughput",
    "LLC capacity", "prefetcher efficiency", "pipeline flushes",
    "speculative execution", "load-store forwarding", "retire width",
    "ROB occupancy", "dispatch stalls", "data cache miss", "page walk latency",
]

VERBS = [
    "dominates", "amplifies", "mitigates", "stresses", "characterizes",
    "bottlenecks", "serializes", "overlaps", "reduces", "exposes",
]

ADJECTIVES = [
    "branchy", "memory-bound", "frontend-bound", "backend-bound", "retiring",
    "deterministic", "agentic", "orchestration-heavy", "compute-bound", "latency-sensitive",
]


def make_corpus(n: int, rng: random.Random) -> list[str]:
    docs = []
    for i in range(n):
        topic = rng.choice(TOPICS)
        verb = rng.choice(VERBS)
        adj = rng.choice(ADJECTIVES)
        num = rng.randint(1, 9999)
        docs.append(
            f"Document {i}: {adj} workloads show that {topic} {verb} "
            f"the critical path by {num} cycles in agent-style pipelines."
        )
    return docs


def make_query(i: int, rng: random.Random) -> str:
    topic = rng.choice(TOPICS)
    adj = rng.choice(ADJECTIVES)
    return f"How does {topic} impact {adj} inference in iteration {i}?"


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


def corpus_size_for(size: str) -> int:
    return {"short": 100, "medium": 500, "long": 2000}[size]


def main() -> None:
    parser = argparse.ArgumentParser(description="Haystack-style RAG CPU benchmark")
    parser.add_argument("--task-id", default="haystack")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trace-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--context-size", choices=("short", "medium", "long"), default="medium")
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    run_id = f"haystack__{args.task_id}__seed{args.seed}"
    context = context_blob(args.context_size)
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

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

    # ---- One-time setup ----
    t = time.perf_counter_ns()
    model = SentenceTransformer(MODEL_NAME)
    embed_dim = model.get_sentence_embedding_dimension()
    record("load_model", t, {"model": MODEL_NAME, "embed_dim": embed_dim})

    t = time.perf_counter_ns()
    corpus = make_corpus(corpus_size_for(args.context_size), rng)
    record("build_corpus", t, {"corpus_size": len(corpus)})

    t = time.perf_counter_ns()
    corpus_emb = model.encode(
        corpus, batch_size=32, convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")
    record("encode_corpus", t, {"corpus_size": len(corpus), "embed_dim": embed_dim})

    t = time.perf_counter_ns()
    index = faiss.IndexFlatIP(embed_dim)
    index.add(corpus_emb)
    record("build_index", t, {"ntotal": index.ntotal})

    # ---- Per-iteration RAG loop ----
    top_k = 10
    total_queries = 0
    last_top_doc = ""
    for i in range(args.iterations):
        query = make_query(i, rng)

        t = time.perf_counter_ns()
        q_emb = model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        record("embed_query", t, {"iteration": i, "query_bytes": len(query)})

        t = time.perf_counter_ns()
        scores, idxs = index.search(q_emb, top_k)
        record("vector_search", t, {"iteration": i, "top_k": top_k})

        t = time.perf_counter_ns()
        retrieved = corpus_emb[idxs[0]]
        rerank_scores = retrieved @ q_emb[0]
        order = np.argsort(-rerank_scores)
        top_idx = int(idxs[0][int(order[0])])
        record("rerank", t, {"iteration": i, "candidates": top_k})

        t = time.perf_counter_ns()
        top_doc = corpus[top_idx]
        answer = (
            f"[iter {i}] Based on top-1 (score={float(rerank_scores[order[0]]):.4f}): "
            f"{top_doc[:160]} | ctx_bytes={len(context)}"
        )
        encoded = json.dumps({"query": query, "answer": answer}, sort_keys=True)
        _ = json.loads(encoded)
        last_top_doc = top_doc
        record("synthesize", t, {"iteration": i, "answer_bytes": len(answer)})

        total_queries += 1

    end_run = time.perf_counter_ns()
    total_ms = (end_run - start_run) / 1_000_000.0

    # Per-phase aggregates
    phase_latency_ms: dict = {}
    for s in spans:
        phase_latency_ms.setdefault(s["phase"], 0.0)
        phase_latency_ms[s["phase"]] += s["duration_ms"]

    trace = {
        "benchmark": "AgentCPU-Real",
        "workload": "haystack",
        "run_id": run_id,
        "task_id": args.task_id,
        "seed": args.seed,
        "mode": "replay",
        "run_metadata": {
            "framework": "haystack",
            "model": MODEL_NAME,
            "context_size": args.context_size,
            "iterations": args.iterations,
        },
        "spans": spans,
    }

    summary = {
        "benchmark": "AgentCPU-Real",
        "workload": "haystack",
        "task_id": args.task_id,
        "seed": args.seed,
        "mode": "replay",
        "run_id": run_id,
        "config": {
            "framework": "haystack",
            "model": MODEL_NAME,
            "context_size": args.context_size,
            "iterations": args.iterations,
            "top_k": top_k,
        },
        "result": {
            "success": True,
            "latency_ms": total_ms,
            "total_queries": total_queries,
            "corpus_size": len(corpus),
            "embed_dim": embed_dim,
            "phase_latency_ms": phase_latency_ms,
            "context_bytes": len(context),
            "last_top_doc": last_top_doc,
        },
        "trace_output": args.trace_output,
    }

    write_json(args.trace_output, trace)
    write_json(args.summary_output, summary)
    print(json.dumps({"summary": summary["result"], "run_id": run_id}, indent=2))


if __name__ == "__main__":
    main()
