FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/paper_experiments
COPY requirements.txt requirements-rag.txt ./
RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install -r requirements.txt
COPY . .

CMD ["python3.11", "-m", "unittest", "discover", "-v"]
