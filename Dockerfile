FROM ghcr.io/astral-sh/uv:0.9.7 AS uv
FROM python:3.11.14-slim-bookworm AS build
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --extra cpu --no-dev --no-editable

FROM python:3.11.14-slim-bookworm AS runtime
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/tmp/huggingface PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=2
COPY --from=build /app/.venv /app/.venv
COPY LICENSE ./
COPY licenses ./licenses
USER 65532:65532
ENTRYPOINT ["patchdedup"]
CMD ["--help"]

# Optional standalone image. Pass an allowlisted checkpoint with --build-context.
FROM runtime AS bundled
COPY --from=checkpoint --chown=65532:65532 / /model/

# Keep the existing external read-only model mount as the default build.
FROM runtime AS default
