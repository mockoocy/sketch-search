FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin:$PATH"

RUN uv python install 3.13
ENV UV_PYTHON=3.13

COPY pyproject.toml uv.lock ./
COPY server ./server
COPY model ./model

# sync workspace
RUN uv sync --no-install-package torch --no-install-package torchvision

# install CUDA torch
RUN uv pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128

CMD ["uv", "run", "--no-sync", "sketch-search"]
