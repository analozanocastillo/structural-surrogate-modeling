FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ML_STRUCTURAL_ROOT=/app \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-docker.txt ./
RUN python -m pip install --no-cache-dir -r requirements-docker.txt

COPY README.md setup_hannover.sh ./
COPY src/ ./src/
COPY tests/ ./tests/
COPY notebooks/ ./notebooks/
COPY data/reference/ ./data/reference/
RUN chmod +x setup_hannover.sh && mkdir -p data/raw data/extracted

EXPOSE 8888
CMD ["bash", "-lc", "./setup_hannover.sh && exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.root_dir=/app"]
