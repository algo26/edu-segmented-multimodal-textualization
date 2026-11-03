FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md run.py ./
COPY videoqa ./videoqa
COPY configs ./configs
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"
CMD ["python", "run.py", "test", "--config", "configs/c1.yaml"]
