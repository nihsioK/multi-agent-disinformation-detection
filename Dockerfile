FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /uvx /bin/
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev
COPY app ./app
COPY models/roberta-base ./models/roberta-base
COPY data/liar ./data/liar
EXPOSE 8000
CMD ["/app/.venv/bin/uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
