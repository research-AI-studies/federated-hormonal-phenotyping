FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Data and outputs are bind-mounted at runtime; nothing sensitive is baked in.
ENTRYPOINT ["python", "scripts/run_pipeline.py"]
CMD ["--config", "config/default.yaml", "--data", "data/synthetic/cohort.csv"]
