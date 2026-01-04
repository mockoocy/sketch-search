FROM ghcr.io/astral-sh/uv:python3.13-trixie-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY server ./server
COPY model ./model


RUN uv sync
ENV PYTHONPATH=/app

CMD ["uv", "run", "--no-sync" , "server"]
