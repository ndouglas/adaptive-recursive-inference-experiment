FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# PyTorch with CUDA 12.2 (separate install because of --index-url)
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu122

# Remaining dependencies
RUN pip3 install --no-cache-dir transformers accelerate scipy numpy tqdm safetensors

WORKDIR /app
COPY src/ src/
COPY scripts/ scripts/
COPY data/ data/

ENTRYPOINT ["python3", "scripts/run_sweep.py"]
CMD ["--max-math", "8", "--max-eq", "8", "--results", "/workspace/sweep_results.json"]
