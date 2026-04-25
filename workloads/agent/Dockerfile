# syntax = edrevo/dockerfile-plus

FROM ubuntu:jammy

INCLUDE+ ./common/Dockerfile.common

USER root
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-pip \
    python3-venv \
    git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR $tmpdir

# Clone the AgentCPU benchmark repo (private).
# --mount=type=ssh forwards the host SSH agent for this RUN only; no keys
# or SSH config are baked into the final image layer.
RUN mkdir -p -m 0700 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts
RUN --mount=type=ssh git clone git@github.com:5surim/AgentCPU.git /tmp_home/AgentCPU && \
    rm -rf /tmp_home/AgentCPU/.git

# Build per-phase marker shared library used by agent_markers.py. -O0 keeps
# each agent_roi_*_* function at a distinct link address so the marker PCs
# are unique per phase. Placed at /tmp_home so agent_markers.py finds it.
RUN gcc -shared -fPIC -O0 -o /tmp_home/libagent_markers.so \
        /tmp_home/AgentCPU/workloads/marker/marker.c && \
    chmod 0755 /tmp_home/libagent_markers.so

# Install perf for per-phase native-HW PMU characterization driven by
# agent_markers --control fifo:. The kernel-version-specific binary in
# linux-tools-generic is for jammy's 5.15 ABI; that perf still works
# against newer host kernels for the basic counter set we use
# (cycles, instructions, branches, L1/LLC misses).
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        linux-tools-common linux-tools-generic && \
    ln -sf /usr/lib/linux-tools-5.15.0-176/perf /usr/local/bin/perf && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip first. Ubuntu jammy ships pip 22 which has a known
# AssertionError in resolvelib.get_topological_weights on complex graphs.
RUN python3 -m pip install --no-cache-dir --upgrade "pip>=24.0"

# Install CPU-only torch BEFORE the main requirements so sentence-transformers
# sees torch already satisfied and does NOT drag in the 2.5+ GB NVIDIA CUDA
# stack (cublas/cudnn/nccl/triton/...). This is a CPU-only benchmark suite.
RUN python3 -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    "torch>=2.1,<2.5"

RUN python3 -m pip install --no-cache-dir -r /tmp_home/AgentCPU/requirements.txt

# Pre-download the haystack sentence-transformer model at build time so the
# container is offline-deterministic (important for tracing reproducibility).
RUN python3 -c "from sentence_transformers import SentenceTransformer; \
                SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Pre-download NLTK tokenizer data for sumy LexRank (langchain workload).
# Download to a world-readable path so non-root users (e.g. DynamoRIO tracing
# which runs as the host user) can access the data.
ENV NLTK_DATA=/usr/local/share/nltk_data
RUN python3 -c "import nltk; nltk.download('punkt', download_dir='/usr/local/share/nltk_data'); \
                nltk.download('punkt_tab', download_dir='/usr/local/share/nltk_data')"

# Pre-download GPT-2 BPE tokenizer for agent_common (all workloads).
RUN python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('gpt2')"

# Keep numpy/scipy/BLAS single-threaded for clean DynamoRIO tracing.
ENV OPENBLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
# Fix Python hash seed so dict/set iteration order is stable across runs.
# Required for deterministic instruction counts across retraces.
ENV PYTHONHASHSEED=0

COPY workloads/agent/workload_root_entrypoint.sh /tmp_home/agent/workload_root_entrypoint.sh
COPY workloads/agent/workload_user_entrypoint.sh /tmp_home/agent/workload_user_entrypoint.sh

CMD ["/bin/bash"]
