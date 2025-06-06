FROM python:3.9-slim AS builder

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ cmake libjpeg-dev zlib1g-dev ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# COPY wheelhouse/ ./wheelhouse/

RUN pip install --upgrade pip && \
    pip install --default-timeout=120 --retries=10 --prefix=/install --no-cache-dir -r requirements.txt && \
    find /install -name '*.pyc' -delete && \
    find /install -name '__pycache__' -type d -exec rm -rf {} + && \
    rm -rf /root/.cache/pip

FROM python:3.9-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo libz-dev ffmpeg && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY . .
# Ensure the empty folders exist in the image
RUN mkdir -p /app/temp_files /app/uploads

EXPOSE 3015
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3015"]

