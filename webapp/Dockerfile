# Cloud image for the Food Recs Lab web app — SERVE-ONLY (the "limited" deployment).
#
# It installs the core runtime only (no torch: the neural rankers are training-only, see
# pyproject.toml), bakes in the ~9 MB serve-time data/, and starts recsys.serve_only, which
# never builds the world or writes to Postgres. For the FULL local experience, use
# `pip install -e .[full]` + `python -m recsys` on your machine instead.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8000

WORKDIR /app

# libgomp1 is the OpenMP runtime LightGBM links against. Keep the layer lean and non-interactive.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy everything, then install once. .dockerignore keeps the heavy build-only inputs
# (.npy embeddings, impressions, menu cache), tests, artifacts and secrets out of the context.
COPY . .

RUN python -m pip install --upgrade pip && python -m pip install .

# Secrets (DATABASE_URL, APP_USERNAME, APP_PASSWORD, GEMINI_API_KEY) are provided by the host
# as environment variables — never baked into the image.
EXPOSE 8000
CMD ["python", "-m", "recsys.serve_only"]
