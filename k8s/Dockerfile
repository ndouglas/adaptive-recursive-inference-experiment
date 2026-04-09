FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev && \
    rm -rf /var/lib/apt/lists/*

# Bootstrap pip for Python 3.12
RUN python3.12 -m ensurepip --upgrade

# PyTorch with CUDA 12.1 (closest match for driver 535.x / CUDA 12.2)
RUN python3.12 -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121

# Remaining dependencies
RUN python3.12 -m pip install --no-cache-dir transformers accelerate scipy numpy tqdm safetensors

WORKDIR /app
COPY src/ src/
COPY scripts/ scripts/
COPY data/ data/

ENTRYPOINT ["python3.12", "scripts/run_sweep.py"]
CMD ["--max-math", "8", "--max-eq", "8", "--results", "/workspace/sweep_results.json"]
