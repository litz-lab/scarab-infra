# syntax = edrevo/dockerfile-plus

FROM ubuntu:jammy

INCLUDE+ ./common/Dockerfile.common

USER root
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-pip \
    python3-venv && \
    rm -rf /var/lib/apt/lists/*

WORKDIR $tmpdir

RUN mkdir -p /tmp_home/agent
COPY workloads/agent/requirements.txt /tmp_home/agent/requirements.txt

# Upgrade pip first. Ubuntu jammy ships pip 22 which has a known
# AssertionError in resolvelib.get_topological_weights on complex graphs.
RUN python3 -m pip install --no-cache-dir --upgrade "pip>=24.0"

# Install CPU-only torch BEFORE the main requirements so sentence-transformers
# sees torch already satisfied and does NOT drag in the 2.5+ GB NVIDIA CUDA
# stack (cublas/cudnn/nccl/triton/...). This is a CPU-only benchmark suite.
RUN python3 -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    "torch>=2.1,<2.5"

RUN python3 -m pip install --no-cache-dir -r /tmp_home/agent/requirements.txt

# Pre-download the haystack sentence-transformer model at build time so the
# container is offline-deterministic (important for tracing reproducibility).
RUN python3 -c "from sentence_transformers import SentenceTransformer; \
                SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY workloads/agent/run_langchain.py /tmp_home/agent/run_langchain.py
COPY workloads/agent/run_langgraph.py /tmp_home/agent/run_langgraph.py
COPY workloads/agent/run_autogen.py /tmp_home/agent/run_autogen.py
COPY workloads/agent/run_haystack.py /tmp_home/agent/run_haystack.py
COPY workloads/agent/run_swe_agent.py /tmp_home/agent/run_swe_agent.py
COPY workloads/agent/workload_root_entrypoint.sh /tmp_home/agent/workload_root_entrypoint.sh
COPY workloads/agent/workload_user_entrypoint.sh /tmp_home/agent/workload_user_entrypoint.sh

CMD ["/bin/bash"]
