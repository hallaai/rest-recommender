# Cloud image for the Food Recs Lab web app — SERVE-ONLY (the "limited" deployment).
#
# It installs the core runtime only (no torch: the neural rankers are training-only, see
# pyproject.toml), bakes in the ~9 MB serve-time data/, and starts recsys.serve_only, which
# never builds the world or writes to Postgres. For the FULL local experience (generate the
# world, train the neural models, run the eval), use `pip install -e .[full]` + `python -m recsys`
# on your machine instead — this image deliberately leaves that out to stay small and boot fast.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# Build deps for any wheels that need compiling (lightgbm ships manylinux wheels, but libgomp
# is needed at runtime); keep the layer lean.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first (better layer caching): copy only the packaging metadata, install
# the core runtime, then copy the source.
COPY pyproject.toml ./
COPY recsys/__init__.py ./recsys/__init__.py
RUN pip install --upgrade pip && pip install .

# App source, frontend and the baked-in serve-time data (parquet world + truth/*.npz + meta).
# .dockerignore keeps the heavy build-only inputs (.npy embeddings, impressions, menu cache) out.
COPY recsys ./recsys
COPY frontend ./frontend
COPY data ./data

# Re-install so the now-present package source is importable as an installed distribution.
RUN pip install --no-deps .

# Secrets (DATABASE_URL, APP_USERNAME, APP_PASSWORD, GEMINI_API_KEY) are provided by the host
# as environment variables — never baked into the image.
EXPOSE 8000
CMD ["python", "-m", "recsys.serve_only"]
