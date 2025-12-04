# -------------------------------------------------------
# 1. Base image
# -------------------------------------------------------
FROM python:3.10-slim AS base

ENV DEBIAN_FRONTEND=noninteractive

# -------------------------------------------------------
# 2. System packages for ONNX + OpenCV + BentoML
# -------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libopencv-dev \
        libjpeg-dev \
        curl \
        && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------------
# 3. Work directory
# -------------------------------------------------------
WORKDIR /app

# -------------------------------------------------------
# 4. Dependencies
# -------------------------------------------------------
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# -------------------------------------------------------
# 5. Copy application code INCLUDING benchmark.py
# -------------------------------------------------------
COPY service.py .
COPY benchmark.py .
#COPY yolo11s.onnx .
# If you also export the model inside the container, add the .pt file:
# COPY yolo11s.pt .

# copy an images folder for internal benchmarking
COPY coco-dataset/data/ data/coco-dataset/val2017/

EXPOSE 3000

# -------------------------------------------------------
# 6. Default command (BentoML service)
# -------------------------------------------------------
CMD ["bentoml", "serve", "service", "--port", "3000"]
