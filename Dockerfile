ARG USE_GPU=false

FROM python:3.10-slim AS base
ARG USE_GPU

RUN if [ "$USE_GPU" = "true" ]; then \
      echo "Using GPU base image" && \
      echo "FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04" > /tmp/Dockerfile.gpu ; \
    else \
      echo "Using CPU base image" && \
      echo "FROM python:3.10-slim" > /tmp/Dockerfile.gpu ; \
    fi

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY configs/config.yaml .

ENTRYPOINT ["python3", "src/main.py"]